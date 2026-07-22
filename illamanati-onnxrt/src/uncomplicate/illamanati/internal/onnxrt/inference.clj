;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.inference
  (:require [uncomplicate.commons
             [core :refer [let-release with-release Releaseable release Info info sizeof view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer]]
            [uncomplicate.neanderthal
             [core :refer [transfer! view-vctr entry! view-ge submatrix copy!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow MemoryContext]]
            [uncomplicate.diamond
             [tensor :refer [input output Transfer tensor shape data-type layout
                             view-tz offset! transformer]]
             [onnxrt :refer [onnx *onnx-options*]]]
            [uncomplicate.diamond.internal.protocols
             :refer [neanderthal-factory DiamondFactoryProvider Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer [onnx-data-type-pointer]]
             [core :as onnx
              :refer [free mutable-data onnx-tensor io-binding input-count output-count cast-type
                      value-tensor-info input-type-info output-type-info tensor-type
                      bind-input! bind-output! runner* synchronize-inputs! synchronize-outputs!
                      options override-dimension! available-providers execution-mode! cpu-mem-arena!
                      disable-per-session-threads! graph-optimization! inter-op-threads!
                      append-provider! threading-options environment memory-info session]]
             [impl :refer [*ort-api* *default-allocator*
                           bind-input* bind-output* input-name* output-name*]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.protocols :refer [TokenizerProvider GeneratorProvider]]
            [uncomplicate.snapdragan :refer [sampler]])
  (:import [clojure.lang IFn AFn]))

(defprotocol KVManager
  (base-tz [this])
  (max-seq-len [this])
  (bind-past-kv!
    [this binding past-seq-len]
    [this binding past-seq-len idx])
  (bind-present-kv!
    [this binding total-seq-len]
    [this binding total-seq-len idx]))

(defn kv-names
  ([sess in-offset out-offset]
   (let [ort-api (safe *ort-api*)
         allo (safe *default-allocator*)
         sess (safe sess)]
     [(doall (mapv #(input-name* ort-api sess allo %)
                   (range in-offset (input-count sess))))
      (doall (mapv #(output-name* ort-api sess allo %)
                   (range out-offset (output-count sess))))])))

(deftype ContiguousKVManager [ort-api
                              free
                              mem-info
                              base-tz
                              ^long layer-stride
                              ^long num-layers
                              ^long max-seq-len
                              layers
                              kv-type
                              shape]
  Releaseable
  (release [_]
    (doseq [[kv info past-name present-name] layers]
      (release kv)
      (release info)
      (free past-name)
      (free present-name))
    (release base-tz))
  KVManager
  (base-tz [_]
    base-tz)
  (max-seq-len [_]
    max-seq-len)
  (bind-past-kv! [this binding! past-seq-len]
    (if (<= 0 (long past-seq-len))
      (let [current-shape (assoc shape 2 past-seq-len)]
        (doseq [[kv _ past-name _] layers]
          (with-release [kv-view (onnx-tensor mem-info current-shape (mutable-data kv) kv-type)]
            (bind-input* ort-api binding! past-name kv-view)))
        this)
      (dragan-says-ex "Please don't try to process prompts leading to negative kv-cache size."
                      {:past-seq-len past-seq-len})))
  (bind-present-kv! [this binding! total-seq-len]
    (if (<= 0 (long total-seq-len))
      (let [current-shape (assoc shape 2 total-seq-len)]
        (doseq [[kv _ _ present-name] layers]
          (with-release [kv-view (onnx-tensor mem-info current-shape (mutable-data kv) kv-type)]
            (bind-output* ort-api binding! present-name kv-view)))
        this)
      (dragan-says-ex "please don't try to process prompts leading to negative kv-cache size."
                      {:total-seq-len total-seq-len}))))

(defn element-alignment ^long [data-type]
  (case data-type
    0 64
    1 128
    2 128
    64))

(defn align-up ^long [^long size ^long element-width]
  (let [target-alignment-bytes 256
        alignment (quot target-alignment-bytes element-width)
        r (rem size alignment)]
    (if (= 0 r)
      size
      (+ size (- alignment r)))))

(defn contiguous-kv-manager [sess mem-info base-tz kv-type
                             input-offset output-offset
                             batch-size num-heads max-seq-len head-dim]
  (let [[past-names present-names] (kv-names sess input-offset output-offset)
        ort-api (safe *ort-api*)
        num-layers (count past-names)
        max-shape [batch-size num-heads max-seq-len head-dim]
        layer-capacity (* batch-size num-heads max-seq-len head-dim)
        layer-stride (align-up layer-capacity (sizeof (buffer base-tz)))
        base-ptr (buffer base-tz)
        layers (mapv (fn [i past-name present-name]
                       (let [kv (onnx-tensor mem-info max-shape
                                             (get-pointer base-ptr (* i layer-stride))
                                             kv-type)]
                         [kv (value-tensor-info kv) past-name present-name]))
                     (range num-layers) past-names present-names)]
    (->ContiguousKVManager ort-api free mem-info base-tz
                           layer-stride num-layers max-seq-len layers kv-type max-shape)))

(defn kv-shifter [src dst ^long seq-len]
  (let [sub-shape (update (shape dst) 3 - seq-len)
        token-shift (* seq-len (get (layout dst) 3))]
    (if (< 0 (get sub-shape 3))
      (let-release [view-dst (view-tz dst sub-shape)
                    view-src (offset! (view-tz src sub-shape) token-shift)]
        (transformer view-src view-dst))
      (dragan-says-ex "you can't shift more tokens than kv-cache holds."
                      {:required seq-len
                       :available (get (shape dst) 3)}))))

(defn bind-kv-linear! [[past present past-seq-len] binding! ^long seq-len]
  (let [total-seq-len (+ (long past-seq-len) seq-len)
        base (base-tz past)
        max-seq-len (long (max-seq-len past))]
    (if (< max-seq-len total-seq-len)
      (dragan-says-ex "kvcache limit reached. this model does not support complex kv management."
                      {:total-seq-len total-seq-len
                       :max-seq-len max-seq-len})
      [(bind-present-kv! present binding! total-seq-len)
       (bind-past-kv! past binding! past-seq-len)
       total-seq-len])))

(defn bind-kv-sliding! [[past present past-seq-len past->present present->past] binding! ^long seq-len]
  (let [total-seq-len (+ (long past-seq-len) seq-len)
        base (base-tz past)
        max-seq-len (long (max-seq-len past))]
    (if (< max-seq-len total-seq-len)
      (let [shift-amount (- total-seq-len max-seq-len)
            effective-past-len (- past-seq-len shift-amount)]
        (if (<= 0 effective-past-len)
          (if (= 1 shift-amount)
            (past->present)
            (with-release [dynamic-shifter (kv-shifter base (base-tz present) shift-amount)]
              (dynamic-shifter)))
          (dragan-says-ex "Please don't try to process prompts leading to negative kv-cache size."
                          {:effective-past-len effective-past-len
                           :total-seq-len total-seq-len
                           :max-seq-len max-seq-len}))
        [(bind-present-kv! past binding! max-seq-len)
         (bind-past-kv! present binding! effective-past-len)
         max-seq-len
         past->present
         present->past])
      [(bind-present-kv! present binding! total-seq-len)
       (bind-past-kv! past binding! past-seq-len)
       total-seq-len
       present->past
       past->present])))

(deftype DecoderModel [fact mem-info
                       sess opt run-session! prefill-bind decode-bind
                       embeds-name decode-embeds onnx-decode-embeds
                       attention-mask-name decode-attention-mask onnx-decode-attention-mask
                       position-ids-name decode-position-ids onnx-decode-position-ids
                       logits-name decode-logits onnx-decode-logits ge-decode-logits
                       kvmans bind-kv
                       attention-shape]
  Releaseable
  (release [_]
    (release onnx-decode-embeds)
    (release onnx-decode-attention-mask)
    (release onnx-decode-position-ids)
    (release onnx-decode-logits)
    (release decode-embeds)
    (release decode-attention-mask)
    (release decode-position-ids)
    (release decode-logits)
    (release prefill-bind)
    (release decode-bind)
    (release run-session!)
    (release sess)
    (release opt)
    (release ge-decode-logits)
    (run! release (deref kvmans)))
  Transfer
  (input [_]
    decode-embeds)
  (output [_]
    decode-logits)
  IFn
  (invoke [_ embeds onnx-embeds
           attention-mask onnx-attention-mask
           position-ids onnx-position-ids
           logits onnx-logits]
    (let [seq-len (long ((shape embeds) 1))
          total-seq-len (long ((shape attention-mask) 1))
          [batch-size _ vocab-size :as sub-shape] (shape logits)
          batch-data-len (* seq-len (long vocab-size))
          last-logits (submatrix (view-ge (view-vctr logits) batch-data-len batch-size)
                                 (- batch-data-len (long vocab-size)) 0 vocab-size batch-size)]
      (bind-input! prefill-bind embeds-name onnx-embeds)
      (bind-input! prefill-bind attention-mask-name onnx-attention-mask)
      (when position-ids
        (transfer! (take (* (long batch-size) seq-len)
                         (cycle (range (- total-seq-len seq-len) total-seq-len)))
                   (view-vctr position-ids)) ;;todo support rolling kvs; todo write specialized kernel;
        (bind-input! prefill-bind position-ids-name onnx-position-ids))
      (bind-output! prefill-bind logits-name onnx-logits)
      (swap! kvmans bind-kv prefill-bind seq-len)
      (swap! attention-shape assoc 1 total-seq-len)
      (synchronize-inputs! prefill-bind)
      (run-session! prefill-bind)
      (synchronize-outputs! prefill-bind);;TODO remove if not needed!
      (copy! last-logits ge-decode-logits)
      decode-logits))
  (invoke [this embeds attention-mask logits!]
    (.invoke this embeds attention-mask nil logits!))
  (invoke [_]
    (swap! attention-shape update 1 (fn ^long [^long x]
                                      (min (max-seq-len ((deref kvmans) 0))
                                           (inc x))))
    (with-release [mask-view (onnx-tensor mem-info (deref attention-shape)
                                          (buffer decode-attention-mask)
                                          (data-type decode-attention-mask))]
      (bind-input! decode-bind attention-mask-name mask-view)
      (when decode-position-ids
        (entry! (view-vctr decode-position-ids) (get (deref kvmans) 2)))
      (swap! kvmans bind-kv decode-bind 1)
      (synchronize-inputs! decode-bind)
      (run-session! decode-bind)
      (synchronize-outputs! decode-bind);;TODO remove if not needed!
      decode-logits))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn decoder-model [fact mem-info sess opt
                     [embeds-name attention-mask-name position-ids-name :as input-names]
                     [logits-name :as output-names]
                     decode-embeds
     max-seq-len]
  (with-release [embeds-type-info (input-type-info sess 0)
                 attention-mask-type-info (input-type-info sess 1)
                 position-ids-type-info (when position-ids-name (input-type-info sess 2))
                 logits-type-info (output-type-info sess 0)
                 input-offset (long (count (filter identity input-names)))
                 output-offset (long (count (filter identity output-names)))
                 kv-type-info (input-type-info sess input-offset)]
    (let [neand-fact (neanderthal-factory fact)
          decode-embeds (view decode-embeds)
          embeds-info (cast-type embeds-type-info)
          attention-mask-info (cast-type attention-mask-type-info)
          position-ids-info (when position-ids-name (cast-type position-ids-type-info))
          logits-info (cast-type logits-type-info)
          kv-info (cast-type kv-type-info)
          [batch-size num-heads _ head-dim] (onnx/shape kv-info)
          hidden-size (get (onnx/shape embeds-info) 2)
          embeds-type (tensor-type embeds-info)
          attention-mask-type (tensor-type attention-mask-info)
          attention-shape (atom [batch-size 0])
          position-ids-type (when position-ids-info (tensor-type position-ids-info))
          logits-type (tensor-type logits-info)
          logits-shape (onnx/shape logits-info)
          vocab-size (peek logits-shape)
          num-layers (- (input-count sess) input-offset)
          kv-type (tensor-type kv-info)
          kv-type-pointer (onnx-data-type-pointer kv-type)
          kv-element-width (with-release [temp (kv-type-pointer 1)]
                             (sizeof temp))
          layer-capacity (* batch-size num-heads max-seq-len head-dim)
          layer-stride (align-up layer-capacity kv-element-width)
          total-elements (* num-layers layer-stride)
          kv-5d-shape [num-layers batch-size num-heads max-seq-len head-dim]
          kv-5d-strides [layer-stride layer-capacity (* max-seq-len head-dim) head-dim 1]]
      (let-release [run-session! (runner* sess)
                    prefill-bind (io-binding sess)
                    decode-bind (io-binding sess)
                    onnx-decode-embeds (onnx-tensor mem-info [batch-size 1 hidden-size]
                                                    (buffer decode-embeds) embeds-type)
                    decode-attention-mask-desc (tensor-desc fact neand-fact [batch-size max-seq-len]
                                                            attention-mask-type)
                    decode-attention-mask (create-tz fact neand-fact decode-attention-mask-desc)
                    onnx-decode-attention-mask (onnx-tensor mem-info [batch-size max-seq-len]
                                                            (buffer decode-attention-mask)
                                                            attention-mask-type)
                    decode-position-ids-desc (when position-ids-type
                                               (tensor-desc fact neand-fact
                                                            [batch-size 1] position-ids-type))
                    decode-position-ids (when decode-position-ids-desc
                                          (create-tz fact neand-fact decode-position-ids-desc))
                    onnx-decode-position-ids (when decode-position-ids
                                               (onnx-tensor mem-info [batch-size 1]
                                                            (buffer decode-position-ids)
                                                            position-ids-type))
                    decode-logits-desc (tensor-desc fact neand-fact [batch-size 1 vocab-size]
                                                    logits-type)
                    decode-logits (create-tz fact neand-fact decode-logits-desc)
                    onnx-decode-logits (onnx-tensor mem-info [batch-size 1 vocab-size]
                                                    (buffer decode-logits) logits-type)
                    ge-decode-logits (view-ge (view-vctr decode-logits) vocab-size batch-size)
                    base-tz-desc (tensor-desc fact neand-fact kv-5d-shape kv-type kv-5d-strides)
                    base-tz-a (create-tz fact neand-fact base-tz-desc)
                    base-tz-b (create-tz fact neand-fact base-tz-desc)
                    a->b (when decode-position-ids (kv-shifter base-tz-a base-tz-b 1))
                    b->a (when decode-position-ids (kv-shifter base-tz-b base-tz-a 1))
                    kvm-a (contiguous-kv-manager sess mem-info base-tz-a kv-type
                                                 input-offset output-offset
                                                 batch-size num-heads max-seq-len head-dim)
                    kvm-b (contiguous-kv-manager sess mem-info base-tz-b kv-type
                                                 input-offset output-offset
                                                 batch-size num-heads max-seq-len head-dim)
                    kvmans (atom [kvm-a kvm-b 0 a->b b->a])]
        (bind-input! decode-bind embeds-name onnx-decode-embeds)
        (entry! (view-vctr decode-attention-mask) 1)
        (when decode-position-ids
          (entry! (view-vctr decode-position-ids) 0)
          (bind-input! decode-bind position-ids-name onnx-decode-position-ids))
        (bind-output! decode-bind logits-name onnx-decode-logits)
        (->DecoderModel fact mem-info sess opt run-session! prefill-bind decode-bind
                        embeds-name decode-embeds onnx-decode-embeds
                        attention-mask-name decode-attention-mask onnx-decode-attention-mask
                        position-ids-name decode-position-ids onnx-decode-position-ids
                        logits-name decode-logits onnx-decode-logits ge-decode-logits
                        kvmans (if decode-position-ids bind-kv-sliding! bind-kv-linear!)
                        attention-shape)))))

(deftype EmbeddingModel [fact mem-info sess opt run-session! prefill-bind decode-bind
                         input-ids-name decode-input-ids onnx-decode-input-ids
                         image-features-name decode-image-features  onnx-decode-image-features
                         embeds-name decode-embeds onnx-decode-embeds]
  Releaseable
  (release [_]
    (release onnx-decode-input-ids)
    (release onnx-decode-image-features)
    (release onnx-decode-embeds)
    (release decode-input-ids)
    (release decode-image-features)
    (release decode-embeds)
    (release prefill-bind)
    (release decode-bind)
    (release run-session!)
    (release sess)
    (release opt))
  Transfer
  (input [_]
    decode-input-ids)
  (output [_]
    decode-embeds)
  IFn
  (invoke [this onnx-input-ids onnx-image-features onnx-embeds]
    (bind-input! prefill-bind input-ids-name onnx-input-ids)
    (bind-input! prefill-bind image-features-name
                 (or onnx-image-features onnx-decode-image-features))
    (bind-output! prefill-bind embeds-name onnx-embeds)
    (synchronize-inputs! prefill-bind)
    (run-session! prefill-bind)
    (synchronize-outputs! prefill-bind);;TODO remove if not needed!
    decode-embeds)
  (invoke [this input-ids embeds!]
    (.invoke this input-ids nil embeds!))
  (invoke [_]
    (synchronize-inputs! decode-bind)
    (run-session! decode-bind)
    (synchronize-outputs! decode-bind);;TODO remove if not needed!
    decode-embeds)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn embedding-model [fact mem-info sess opt
                       [input-ids-name image-features-name]
                       [embeds-name]]
  (with-release [input-ids-type-info (input-type-info sess 0)
                 image-features-type-info (input-type-info sess 1)
                 embeds-type-info (output-type-info sess 0)]
    (let [neand-fact (neanderthal-factory fact)
          input-ids-info (cast-type input-ids-type-info)
          image-features-info (cast-type image-features-type-info)
          embeds-info (cast-type embeds-type-info)
          [batch-size _ hidden-size] (onnx/shape embeds-info)
          input-ids-type (tensor-type input-ids-info)
          image-features-type (tensor-type image-features-info)
          embeds-type (tensor-type embeds-info)]
      (let-release [run-session! (runner* sess)
                    prefill-bind (io-binding sess)
                    decode-bind (io-binding sess)
                    decode-input-ids-desc (tensor-desc fact neand-fact [batch-size 1] input-ids-type)
                    decode-input-ids (create-tz fact neand-fact decode-input-ids-desc)
                    onnx-decode-input-ids (onnx-tensor mem-info [batch-size 1]
                                                       (buffer decode-input-ids) input-ids-type)
                    decode-embeds-desc (tensor-desc fact neand-fact [batch-size 1 hidden-size]
                                                    embeds-type)
                    decode-embeds (create-tz fact neand-fact decode-embeds-desc)
                    onnx-decode-embeds (onnx-tensor mem-info [batch-size 1 hidden-size]
                                                    (buffer decode-embeds) embeds-type)
                    image-features-desc (tensor-desc fact neand-fact
                                                     [0 0 hidden-size] image-features-type)
                    decode-image-features (create-tz fact neand-fact image-features-desc)
                    onnx-decode-image-features (onnx-tensor mem-info [0 0 hidden-size]
                                                            (buffer decode-image-features)
                                                            image-features-type)]
        (bind-input! decode-bind input-ids-name onnx-decode-input-ids)
        (bind-input! decode-bind image-features-name onnx-decode-image-features)
        (bind-output! decode-bind embeds-name onnx-decode-embeds)
        (->EmbeddingModel fact mem-info sess opt run-session! prefill-bind decode-bind
                          input-ids-name decode-input-ids onnx-decode-input-ids
                          image-features-name decode-image-features onnx-decode-image-features
                          embeds-name decode-embeds onnx-decode-embeds)))))

(defn universal-options!
  ([opt! args]
   (let [available-ep (set (available-providers))]
     (doto opt!
       (execution-mode! :sequential)
       (cpu-mem-arena! false)
       (override-dimension! "batch_size" (:batch-size args))
       (override-dimension! "num_images" 0)
       (override-dimension! "image_length" 0)
       (disable-per-session-threads!)
       (graph-optimization! (:graph-optimization args))
       ;; (intra-op-threads! 10)
       (inter-op-threads! 1))
     (doseq [ep (:ep args)]
       (append-provider! opt!
                         (or (available-ep ep)
                             (dragan-says-ex (format "Execution provider %s is not available." ep)
                                             {:requested ep :available available-ep}))
                         (args ep))))
   opt!))

(deftype TokenGenerator [fact
                         tensor-desc create-tz
                         mem-info embedding-model! decoder-model! sample!
                         ^long batch-size
                         ^long hidden-size
                         ^long vocab-size]
  Releaseable
  (release [_]
    (release decoder-model!)
    (release embedding-model!)
    (release sample!)
    (release mem-info))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input embedding-model!))
  (output [_]
    (output decoder-model))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ prefill-ids arg]
    (let [seq-len (if (number? (first prefill-ids))
                    (count prefill-ids)
                    (max (map count prefill-ids)))
          past-seq-len (long (get (deref (.attention-shape decoder-model!)) 1));;TODO reflection
          total-seq-len (+ past-seq-len seq-len)
          ids-shape [batch-size seq-len]
          ids-dt (data-type (input embedding-model!))
          image-features-shape [0 0 hidden-size]
          image-features-dt(data-type (.-decode-image-features embedding-model!));;TODO reflection
          embeds-shape [batch-size seq-len hidden-size]
          embeds-dt (data-type (output embedding-model!))
          mask-shape [batch-size total-seq-len]
          mask-dt (data-type (.-decode-attention-mask decoder-model!));;TODO reflection
          position-ids-shape [batch-size seq-len]
          logits-shape [batch-size seq-len vocab-size]
          logits-dt (data-type (output decoder-model!))]
      (with-release [ids-desc (tensor-desc ids-shape ids-dt)
                     ids (create-tz ids-desc)
                     onnx-ids (onnx-tensor mem-info ids-shape (buffer ids) ids-dt)
                     embeds-desc (tensor-desc embeds-shape embeds-dt)
                     embeds (create-tz embeds-desc)
                     onnx-embeds (onnx-tensor mem-info embeds-shape (buffer embeds) embeds-dt)
                     mask-desc (tensor-desc mask-shape mask-dt)
                     mask (create-tz mask-desc)
                     onnx-mask (onnx-tensor mem-info mask-shape (buffer mask) mask-dt)
                     position-ids-desc (when-let [decode-position-ids (.decode-position-ids decoder-model!)] ;;TODO reflection
                                         (tensor-desc position-ids-shape (data-type decode-position-ids)))
                     position-ids (when position-ids-desc (create-tz position-ids-desc))
                     onnx-position-ids (when position-ids
                                         (onnx-tensor mem-info position-ids-shape (buffer position-ids)
                                                      (data-type position-ids)))
                     logits-desc (tensor-desc logits-shape logits-dt)
                     logits (create-tz logits-desc)
                     onnx-logits (onnx-tensor mem-info logits-shape (buffer logits) logits-dt)
                     image-features-desc (tensor-desc image-features-shape image-features-dt)
                     image-features (create-tz image-features-desc)
                     onnx-image-features (onnx-tensor mem-info image-features-shape
                                                      (buffer image-features)
                                                      image-features-dt)]
        (transfer! prefill-ids (view-ge (view-vctr ids) seq-len batch-size))
        (embedding-model! onnx-ids onnx-image-features onnx-embeds)
        (entry! (view-vctr mask) 1)
        (decoder-model! embeds onnx-embeds mask onnx-mask position-ids onnx-position-ids logits onnx-logits)
        (sample! arg))))
  (invoke [_ arg]
    (embedding-model!)
    (decoder-model!)
    (sample! arg))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defrecord TokenGeneratorProvider [merged-args tok]
  Releaseable
  (release [_]
    (release tok))
  Info
  (info [_]
    (into (info tok) merged-args))
  (info [_ info-key]
    (or (info tok info-key)
        (merged-args info-key)))
  MemoryContext
  (compatible? [this other]
    (= (device this) (device other)))
  (device [_]
    (:device merged-args))
  GeneratorProvider
  (generator [_ fact]
    (let [vect-fact (neanderthal-factory fact)
          {:keys [batch-size hidden-size vocab-size context-len model-path embedding decoder
                  embedding-inputs embedding-outputs decoder-inputs decoder-outputs]} merged-args]
      (with-release [env-options (threading-options (:env-options merged-args))]
        (let-release [env (or (:env merged-args)
                              (environment (:logging-level merged-args)
                                           (:log-name merged-args)
                                           env-options))
                      embedding-opt (universal-options! (if-let [opt (:options merged-args)]
                                                          (options opt)
                                                          (options))
                                                        merged-args)
                      decoder-opt (options embedding-opt)
                      mem-info (memory-info (device (neanderthal-factory fact :float))
                                            :device :default)
                      embedding-sess (session env (format "%s/%s" model-path embedding) embedding-opt)
                      decoder-sess (session env (format "%s/%s" model-path decoder) decoder-opt)
                      embedding (embedding-model fact mem-info embedding-sess embedding-opt
                                                 embedding-inputs embedding-outputs)
                      decoder (decoder-model fact mem-info decoder-sess decoder-opt
                                             decoder-inputs decoder-outputs
                                             (output embedding) context-len)
                      sample (sampler (view-ge (view-vctr (output decoder))
                                               vocab-size batch-size)
                                      (view-vctr (input embedding)))]
          (->TokenGenerator fact
                            (partial tensor-desc fact vect-fact)
                            (partial create-tz fact vect-fact)
                            mem-info
                            embedding decoder sample
                            batch-size hidden-size vocab-size)))))
  TokenizerProvider
  (tokenizer [this]
    tok))

(defn token-generator;;TODO move to onnx? and call it TokenGenerator factory?
  ([model-path args]
   (let-release [tok (let [[tokenizer model-file] (:tokenizer args)]
                       (tokenizer (format "%s/%s" model-path model-file)))]
     (->TokenGeneratorProvider (merge *onnx-options* args args {:model-path model-path}) tok))))

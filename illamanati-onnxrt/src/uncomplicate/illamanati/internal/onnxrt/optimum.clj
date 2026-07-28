;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.optimum
  (:require [uncomplicate.commons
             [core :refer [let-release with-release Releaseable release Info info sizeof view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer]]
            [uncomplicate.neanderthal
             [core :refer [transfer! view-vctr entry! view-ge submatrix copy! vctr]]
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
            [uncomplicate.illamanati.internal.protocols :refer [TokenizerProvider StepEngineProvider]]
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

(deftype CoreDecoderModel [fact neand-fact mem-info sess opt run-session! prefill-bind decode-bind
                           input-x-name decode-input-x onnx-decode-input-x
                           attention-mask-name decode-attention-mask onnx-decode-attention-mask
                           logits-name decode-logits onnx-decode-logits
                           kvmans bind-kv
                           attention-shape]
  Releaseable
  (release [_]
    (release onnx-decode-input-x)
    (release decode-input-x)
    (release onnx-decode-attention-mask)
    (release decode-attention-mask)
    (release onnx-decode-logits)
    (release decode-logits)
    (release prefill-bind)
    (release decode-bind)
    (release run-session!)
    (release sess)
    (release opt))
  Transfer
  (input [_]
    decode-input-x)
  (output [_]
    decode-logits)
  IFn
  (invoke [this input-x onnx-input-x logits onnx-logits]
    (let [seq-len (long (get (shape input-x) 1))
          [batch-size past-seq-len] (deref attention-shape)
          total-seq-len (+ (long past-seq-len) seq-len)
          mask-shape [batch-size total-seq-len]
          mask-dt (data-type decode-attention-mask)]
      (with-release [mask-desc (tensor-desc fact neand-fact mask-shape mask-dt)
                     mask (create-tz fact neand-fact mask-desc)
                     onnx-mask (onnx-tensor mem-info mask-shape (buffer mask) mask-dt)]
        (entry! (view-vctr mask) 1)
        (bind-input! prefill-bind input-x-name onnx-input-x)
        (bind-input! prefill-bind attention-mask-name onnx-mask)
        (bind-output! prefill-bind logits-name onnx-logits)
        (swap! kvmans bind-kv prefill-bind seq-len)
        (swap! attention-shape assoc 1 total-seq-len)
        (synchronize-inputs! prefill-bind)
        (run-session! prefill-bind)
        (synchronize-outputs! prefill-bind)));;TODO remove if not needed!
    decode-logits)
  (invoke [this input-x onnx-input-x]
    (this input-x onnx-input-x decode-logits onnx-decode-logits))
  (invoke [_]
    (swap! attention-shape update 1 (fn ^long [^long x]
                                      (min (max-seq-len ((deref kvmans) 0))
                                           (inc x))))
    (with-release [mask-view (onnx-tensor mem-info (deref attention-shape)
                                          (buffer decode-attention-mask)
                                          (data-type decode-attention-mask))]
      (bind-input! decode-bind attention-mask-name mask-view)
      (swap! kvmans bind-kv decode-bind 1)
      (synchronize-inputs! decode-bind)
      (run-session! decode-bind)
      (synchronize-outputs! decode-bind);;TODO remove if not needed!
      decode-logits))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn core-decoder-model [fact mem-info sess opt
                          [input-x-name attention-mask-name :as input-names]
                          [logits-name :as output-names]
                          decode-input-x max-seq-len]
  (with-release [input-x-type-info (input-type-info sess 0)
                 attention-mask-type-info (input-type-info sess 1)
                 logits-type-info (output-type-info sess 0)
                 input-offset (long (count (filter identity input-names)))
                 output-offset (long (count (filter identity output-names)))
                 kv-type-info (input-type-info sess input-offset)]
    (let [neand-fact (neanderthal-factory fact)
          input-x-info (cast-type input-x-type-info)
          attention-mask-info (cast-type attention-mask-type-info)
          logits-info (cast-type logits-type-info)
          kv-info (cast-type kv-type-info)
          [batch-size num-heads _ head-dim] (onnx/shape kv-info)
          input-x-type (tensor-type input-x-info)
          decode-input-x-shape (assoc (onnx/shape input-x-info) 1 1)
          attention-mask-type (tensor-type attention-mask-info)
          decode-attention-mask-shape (assoc (onnx/shape attention-mask-info) 1 max-seq-len)
          attention-shape (atom [batch-size 0])
          logits-type (tensor-type logits-info)
          decode-logits-shape (assoc (onnx/shape logits-info) 0 batch-size 1 1)
          vocab-size (peek decode-logits-shape)
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
                    decode-input-x (or (view decode-input-x)
                                       (create-tz fact neand-fact
                                                  (tensor-desc fact neand-fact decode-input-x-shape input-x-type)))
                    onnx-decode-input-x (onnx-tensor mem-info decode-input-x-shape
                                                     (buffer decode-input-x) input-x-type)
                    decode-attention-mask-desc (tensor-desc fact neand-fact decode-attention-mask-shape
                                                            attention-mask-type)
                    decode-attention-mask (create-tz fact neand-fact decode-attention-mask-desc)
                    onnx-decode-attention-mask (onnx-tensor mem-info decode-attention-mask-shape
                                                            (buffer decode-attention-mask)
                                                            attention-mask-type)
                    decode-logits-desc (tensor-desc fact neand-fact decode-logits-shape logits-type)
                    decode-logits (create-tz fact neand-fact decode-logits-desc)
                    onnx-decode-logits (onnx-tensor mem-info decode-logits-shape
                                                    (buffer decode-logits) logits-type)
                    base-tz-desc (tensor-desc fact neand-fact kv-5d-shape kv-type kv-5d-strides)
                    base-tz-a (create-tz fact neand-fact base-tz-desc)
                    base-tz-b (create-tz fact neand-fact base-tz-desc)
                    kvm-a (contiguous-kv-manager sess mem-info base-tz-a kv-type
                                                 input-offset output-offset
                                                 batch-size num-heads max-seq-len head-dim)
                    kvm-b (contiguous-kv-manager sess mem-info base-tz-b kv-type
                                                 input-offset output-offset
                                                 batch-size num-heads max-seq-len head-dim)
                    kvmans (atom [kvm-a kvm-b 0])]
        (entry! (view-vctr decode-attention-mask) 1)
        (bind-input! decode-bind input-x-name onnx-decode-input-x)
        (bind-output! decode-bind logits-name onnx-decode-logits)
        (bind-output! prefill-bind logits-name onnx-decode-logits)
        (->CoreDecoderModel fact neand-fact mem-info sess opt run-session! prefill-bind decode-bind
                            input-x-name decode-input-x onnx-decode-input-x
                            attention-mask-name decode-attention-mask onnx-decode-attention-mask
                            logits-name decode-logits onnx-decode-logits
                            kvmans bind-kv-linear!
                            attention-shape)))))

(deftype EmbeddingModel [fact mem-info sess opt run-session! prefill-bind decode-bind
                         input-ids-name decode-input-ids onnx-decode-input-ids
                         embeds-name decode-embeds onnx-decode-embeds]
  Releaseable
  (release [_]
    (release onnx-decode-input-ids)
    (release onnx-decode-embeds)
    (release decode-input-ids)
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
  (invoke [this onnx-input-ids onnx-embeds]
    (bind-input! prefill-bind input-ids-name onnx-input-ids)
    (bind-output! prefill-bind embeds-name onnx-embeds)
    (synchronize-inputs! prefill-bind)
    (run-session! prefill-bind)
    (synchronize-outputs! prefill-bind);;TODO remove if not needed!
    decode-embeds)
  (invoke [_]
    (synchronize-inputs! decode-bind)
    (run-session! decode-bind)
    (synchronize-outputs! decode-bind);;TODO remove if not needed!
    decode-embeds)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn embedding-model [fact mem-info sess opt [input-ids-name] [embeds-name]]
  (with-release [input-ids-type-info (input-type-info sess 0)
                 embeds-type-info (output-type-info sess 0)]
    (let [neand-fact (neanderthal-factory fact)
          input-ids-info (cast-type input-ids-type-info)
          embeds-info (cast-type embeds-type-info)
          [batch-size _ hidden-size] (onnx/shape embeds-info)
          input-ids-type (tensor-type input-ids-info)
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
                                                    (buffer decode-embeds) embeds-type)]
        (bind-input! decode-bind input-ids-name onnx-decode-input-ids)
        (bind-output! decode-bind embeds-name onnx-decode-embeds)
        (->EmbeddingModel fact mem-info sess opt run-session! prefill-bind decode-bind
                          input-ids-name decode-input-ids onnx-decode-input-ids
                          embeds-name decode-embeds onnx-decode-embeds)))))

(deftype EmbeddingDecoderModel [fact neand-fact mem-info embedding-model! decoder-model!
                                num-logits-name decode-num-logits onnx-decode-num-logits]
  Releaseable
  (release [_]
    (release onnx-decode-num-logits)
    (release decode-num-logits)
    (release decoder-model!)
    (release embedding-model!))
  Transfer
  (input [_]
    (input embedding-model!))
  (output [_]
    (output decoder-model!))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ input-ids onnx-input-ids]
    (let [embeds-shape (assoc (shape (output embedding-model!)) 1 (get (shape input-ids) 1))
          embeds-dt (data-type (output embedding-model!))]
      (with-release [embeds-desc (tensor-desc fact neand-fact embeds-shape embeds-dt)
                     embeds (create-tz fact neand-fact embeds-desc)
                     onnx-embeds (onnx-tensor mem-info (take 3 embeds-shape) (buffer embeds) embeds-dt)]
        (embedding-model! onnx-input-ids onnx-embeds)
        (decoder-model! embeds onnx-embeds))))
  (invoke [_]
    (embedding-model!)
    (decoder-model!))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn embedding-decoder-model [fact mem-info embedding-sess embedding-opt decoder-sess decoder-opt
                               embedding-inputs embedding-outputs
                               [_ _ num-logits-name :as decoder-inputs] decoder-outputs
                               context-len]
  (with-release [num-logits-type-info (input-type-info decoder-sess 2)]
    (let [neand-fact (neanderthal-factory fact)
          num-logits-info (cast-type num-logits-type-info)
          num-logits-type (tensor-type num-logits-info)]
      (let-release [decode-num-logits (vctr (neanderthal-factory neand-fact num-logits-type) [1])
                    onnx-decode-num-logits (onnx-tensor mem-info (buffer decode-num-logits))
                    embedding (embedding-model fact mem-info embedding-sess embedding-opt
                                               embedding-inputs embedding-outputs)
                    decoder (core-decoder-model fact mem-info decoder-sess decoder-opt
                                                decoder-inputs decoder-outputs
                                                (output embedding) context-len)]
        (bind-input! (.-prefill-bind decoder) num-logits-name onnx-decode-num-logits)
        (bind-input! (.-decode-bind decoder) num-logits-name onnx-decode-num-logits)
        (->EmbeddingDecoderModel fact neand-fact mem-info embedding decoder
                                 num-logits-name decode-num-logits onnx-decode-num-logits)))))

(deftype CopyLogitsDecoderModel [fact neand-fact mem-info decoder-model!]
  Releaseable
  (release [_]
    (release decoder-model!))
  Transfer
  (input [_]
    (input decoder-model!))
  (output [_]
    (output decoder-model!))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ input-ids onnx-input-ids]
    (let [[batch-size seq-len vocab-size :as logits-shape]
          (assoc (shape (output decoder-model!)) 1 (get (shape input-ids) 1) )
          batch-data-len (* seq-len (long vocab-size))
          logits-dt (data-type (output decoder-model!))]
      (with-release [logits-desc (tensor-desc fact neand-fact logits-shape logits-dt)
                     logits (create-tz fact neand-fact logits-desc)
                     onnx-logits (onnx-tensor mem-info (take 3 logits-shape) (buffer logits) logits-dt)
                     last-logits (submatrix (view-ge (view-vctr logits) batch-data-len batch-size)
                                            (- batch-data-len (long vocab-size)) 0 vocab-size batch-size)
                     ge-decode-logits (view-ge (view-vctr (output decoder-model!)) vocab-size batch-size)]
        (decoder-model! input-ids onnx-input-ids logits onnx-logits)
        (copy! last-logits ge-decode-logits))))
  (invoke [_]
    (decoder-model!))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn copy-logits-decoder-model [fact mem-info decoder-sess decoder-opt
                                 decoder-inputs decoder-outputs
                                 context-len]
  (with-release [num-logits-type-info (input-type-info decoder-sess 2)]
    (let [neand-fact (neanderthal-factory fact)]
      (let-release [decoder (core-decoder-model fact mem-info decoder-sess decoder-opt
                                                decoder-inputs decoder-outputs
                                                nil context-len)]
        (->CopyLogitsDecoderModel fact neand-fact mem-info decoder)))))

(deftype OptimumStepEngine [fact neand-fact
                            mem-info decoder-model! sample!
                            ^long batch-size]
  Releaseable
  (release [_]
    (release decoder-model!)
    (release sample!)
    (release mem-info))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input decoder-model!))
  (output [_]
    (output decoder-model!))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ prefill-ids arg]
    (let [seq-len (if (number? (first prefill-ids))
                    (count prefill-ids)
                    (max (map count prefill-ids)))
          ids-shape [batch-size seq-len]
          ids-dt (data-type (input decoder-model!))]
      (with-release [ids-desc (tensor-desc fact neand-fact ids-shape ids-dt)
                     ids (create-tz fact neand-fact ids-desc)
                     onnx-ids (onnx-tensor mem-info ids-shape (buffer ids) ids-dt)]
        (transfer! prefill-ids (view-ge (view-vctr ids) seq-len batch-size))
        (decoder-model! ids onnx-ids)
        (sample! arg))))
  (invoke [_ arg]
    (decoder-model!)
    (sample! arg))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defrecord OptimumProvider [merged-args tok]
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
  StepEngineProvider
  (step-engine [_ fact]
    (let [vect-fact (neanderthal-factory fact)
          {:keys [batch-size hidden-size vocab-size context-len model-path
                  embedding embedding-inputs embedding-outputs
                  decoder decoder-inputs decoder-outputs]} merged-args]
      (with-release [env-options (threading-options (:env-options merged-args))]
        (let-release [env (or (:env merged-args)
                              (environment (:logging-level merged-args)
                                           (:log-name merged-args)
                                           env-options))
                      mem-info (memory-info (device (neanderthal-factory fact :float))
                                            :device :default)
                      decoder-opt (universal-options! (if-let [opt (:options merged-args)]
                                                        (options opt)
                                                        (options))
                                                      merged-args)

                      decoder-sess (session env (format "%s/%s" model-path decoder) decoder-opt)
                      decoder (if embedding
                                (let-release [embedding-opt (options decoder-opt)
                                              embedding-sess (session env (format "%s/%s" model-path embedding) embedding-opt)]
                                  (embedding-decoder-model fact mem-info
                                                           embedding-sess embedding-opt
                                                           decoder-sess decoder-opt
                                                           embedding-inputs embedding-outputs
                                                           decoder-inputs decoder-outputs
                                                           context-len))
                                (copy-logits-decoder-model fact mem-info decoder-sess decoder-opt
                                                           decoder-inputs decoder-outputs
                                                           context-len))
                      sample (sampler (view-ge (view-vctr (output decoder))
                                               vocab-size batch-size)
                                      (view-vctr (input decoder)))]
          (->OptimumStepEngine fact (neanderthal-factory fact)
                               mem-info decoder sample
                               batch-size)))))
  TokenizerProvider
  (tokenizer [this]
    tok))

(defn optimum-provider
  ([model-path args]
   (let-release [tok (let [[tokenizer model-file] (:tokenizer args)]
                       (tokenizer (format "%s/%s" model-path model-file)))]
     (->OptimumProvider (merge *onnx-options* args {:model-path model-path}) tok))))

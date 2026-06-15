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
             [core :refer [let-release with-release Releaseable release Info size sizeof]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer get-entry position! long-pointer put-entry!]]
            [uncomplicate.neanderthal
             [core :refer [iamax transfer! native view-vctr entry!]]
             [block :refer [buffer]]]
            [uncomplicate.diamond
             [tensor :refer [Transfer tensor output shape data-type layout view-tz offset! transformer]]
             [dnn :refer [network activation]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory DiamondFactoryProvider]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer [onnx-data-type-pointer]]
             [core :as onnx
              :refer [onnx-tensor io-binding input-count output-count cast-type value-tensor-info input-type-info output-type-info tensor-type
                      bind-input! bind-output! runner* options override-dimension! free mutable-data]]
             [impl :refer [*ort-api* *default-allocator* create-tensor* bind-input* bind-output* input-name* output-name*
                           tensor-dimensions*]]
             [model :refer [create-tz tensor-desc]]])
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
    (let [current-shape (assoc shape 2 past-seq-len)]
      (doseq [[kv _ past-name _] layers]
        (with-release [kv-view (onnx-tensor mem-info current-shape (mutable-data kv) kv-type)]
          (bind-input* ort-api binding! past-name kv-view))))
    this)
  (bind-present-kv! [this binding! total-seq-len]
    (let [current-shape (assoc shape 2 total-seq-len)]
      (doseq [[kv _ _ present-name] layers]
        (with-release [kv-view (onnx-tensor mem-info current-shape (mutable-data kv) kv-type)]
          (bind-output* ort-api binding! present-name kv-view))))
    this))

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
  (let [sub-shape (update (shape dst) 3 (fn ^long [^long x] (- x seq-len)))
        token-shift (* seq-len (get (layout dst) 3))]
    (if (< 0 (get sub-shape 3))
      (let-release [view-dst (view-tz dst sub-shape)
                    view-src (offset! (view-tz src sub-shape) token-shift)]
        (transformer view-src view-dst))
      (dragan-says-ex "You can't shift more tokens than kv-cache holds."
                      {:required seq-len
                       :available (get (shape dst) 3)}))))

(defn bind-kv! [[past present past-seq-len past->present present->past] binding! ^long seq-len]
  (let [total-seq-len (+ (long past-seq-len) seq-len)
        base (base-tz past)
        max-seq-len (long (max-seq-len past))]
    (if (< max-seq-len total-seq-len)
      (let [shift-amount (- total-seq-len max-seq-len)
            effective-past-len (- past-seq-len shift-amount)]
        (if (= 1 shift-amount)
          (past->present)
          (with-release [dynamic-shifter (kv-shifter base (base-tz present) shift-amount)]
            (dynamic-shifter)))
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


(deftype TextModel [mem-info
                    prefill-sess prefill! prefill-bind
                    decode-sess decode! decode-bind
                    embeds-name decode-embeds onnx-decode-embeds
                    attention-mask-name decode-attention-mask onnx-decode-attention-mask
                    position-ids-name decode-position-ids onnx-decode-position-ids
                    logits-name decode-logits onnx-decode-logits
                    kvmans
                    attention-shape attention-info]
  Releaseable
  (release [_]
    (release prefill-sess)
    (release prefill!)
    (release prefill-bind)
    (release decode-sess)
    (release decode!)
    (release decode-bind)
    (release onnx-decode-embeds)
    (release onnx-decode-attention-mask)
    (release onnx-decode-position-ids)
    (release onnx-decode-logits)
    (release decode-embeds)
    (release decode-attention-mask)
    (release decode-position-ids)
    (release decode-logits)
    (release attention-info)
    (run! release (deref kvmans)))
  IFn
  (invoke [_ embeds attention-mask position-ids logits!]
    (let [[_ seq-len :as embeds-shape] (shape embeds)
          [_ total-seq-len :as attention-mask-shape] (shape attention-mask)]
      (with-release [onnx-embeds (onnx-tensor mem-info embeds-shape (buffer embeds) (data-type embeds))
                     onnx-attention-mask (onnx-tensor mem-info attention-mask-shape
                                                      (buffer attention-mask)
                                                      (data-type attention-mask))
                     onnx-position-ids (if position-ids
                                         (onnx-tensor mem-info (shape position-ids)
                                                      (buffer position-ids)
                                                      (data-type position-ids))
                                         nil)
                     onnx-logits (onnx-tensor mem-info (shape logits!)
                                              (buffer logits!) (data-type logits!))]
        (bind-input! prefill-bind embeds-name onnx-embeds)
        (bind-input! prefill-bind attention-mask-name onnx-attention-mask)
        (when onnx-position-ids (bind-input! prefill-bind position-ids-name onnx-position-ids))
        (bind-output! prefill-bind logits-name onnx-logits)
        (swap! kvmans bind-kv! prefill-bind seq-len)
        (swap! attention-shape assoc 1 total-seq-len)
        (prefill! prefill-bind)
        logits!)))
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
      (swap! kvmans bind-kv! decode-bind 1)
      (decode! decode-bind)
      decode-logits))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn text-model [fact mem-info prefill-sess decode-sess
                  [embeds-name attention-mask-name position-ids-name :as input-names]
                  [logits-name :as output-names]
                  max-seq-len]
  (with-release [embeds-type-info (input-type-info decode-sess 0)
                 attention-mask-type-info (input-type-info decode-sess 1)
                 position-ids-type-info (when position-ids-name (input-type-info decode-sess 2))
                 logits-type-info (output-type-info decode-sess 0)
                 input-offset (long (count (filter identity input-names)))
                 output-offset (long (count (filter identity output-names)))
                 kv-type-info (input-type-info prefill-sess input-offset)]
    (let [neand-fact (neanderthal-factory fact)
          embeds-info (cast-type embeds-type-info)
          attention-mask-info (cast-type attention-mask-type-info)
          position-ids-info (when position-ids-name (cast-type position-ids-type-info))
          logits-info (cast-type logits-type-info)
          kv-info (cast-type kv-type-info)
          [batch-size num-heads _ head-dim] (onnx/shape kv-info)
          embeds-type (tensor-type embeds-info)
          embeds-dim (peek (onnx/shape embeds-info))
          attention-mask-type (tensor-type attention-mask-info)
          attention-shape (atom [batch-size 0])
          position-ids-type (when position-ids-info (tensor-type position-ids-info))
          logits-type (tensor-type logits-info)
          logits-shape (onnx/shape logits-info)
          logits-dim (peek logits-shape)
          num-layers (- (input-count prefill-sess) input-offset)
          kv-type (tensor-type kv-info)
          kv-type-pointer (onnx-data-type-pointer kv-type)
          kv-element-width (with-release [temp (kv-type-pointer 1)]
                             (sizeof temp))
          layer-capacity (* batch-size num-heads max-seq-len head-dim)
          layer-stride (align-up layer-capacity kv-element-width)
          total-elements (* num-layers layer-stride)
          kv-5d-shape [num-layers batch-size num-heads max-seq-len head-dim]
          kv-5d-strides [layer-stride layer-capacity (* max-seq-len head-dim) head-dim 1]]
      (let-release [prefill! (runner* prefill-sess)
                    prefill-bind (io-binding prefill-sess)
                    decode! (runner* decode-sess)
                    decode-bind (io-binding decode-sess)
                    decode-embeds-desc (tensor-desc fact neand-fact [batch-size 1 embeds-dim] embeds-type)
                    decode-embeds (create-tz fact neand-fact decode-embeds-desc)
                    onnx-decode-embeds (onnx-tensor mem-info [batch-size 1 embeds-dim]
                                                    (buffer decode-embeds) embeds-type)
                    decode-attention-mask-desc (tensor-desc fact neand-fact [batch-size max-seq-len] attention-mask-type)
                    decode-attention-mask (create-tz fact neand-fact decode-attention-mask-desc)
                    onnx-decode-attention-mask (onnx-tensor mem-info [batch-size max-seq-len]
                                                            (buffer decode-attention-mask) attention-mask-type)
                    decode-position-ids-desc (when position-ids-type
                                               (tensor-desc fact neand-fact [batch-size 1] position-ids-type))
                    decode-position-ids (when position-ids-type
                                          (create-tz fact neand-fact decode-position-ids-desc))
                    onnx-decode-position-ids (when position-ids-type
                                               (onnx-tensor mem-info [batch-size 1]
                                                            (buffer decode-position-ids) position-ids-type))
                    decode-logits-desc (tensor-desc fact neand-fact [batch-size 1 logits-dim] logits-type)
                    decode-logits (create-tz fact neand-fact decode-logits-desc)
                    onnx-decode-logits (onnx-tensor mem-info [batch-size 1 logits-dim]
                                                    (buffer decode-logits) logits-type)
                    attention-info (value-tensor-info onnx-decode-attention-mask)
                    base-tz-desc (tensor-desc fact neand-fact kv-5d-shape kv-type kv-5d-strides)
                    base-tz-a (create-tz fact neand-fact base-tz-desc)
                    base-tz-b (create-tz fact neand-fact base-tz-desc)
                    a->b (kv-shifter base-tz-a base-tz-b 1)
                    b->a (kv-shifter base-tz-b base-tz-a 1)
                    kvm-a (contiguous-kv-manager prefill-sess mem-info base-tz-a kv-type
                                                 input-offset output-offset
                                                 batch-size num-heads max-seq-len head-dim)
                    kvm-b (contiguous-kv-manager prefill-sess mem-info base-tz-b kv-type
                                                 input-offset output-offset
                                                 batch-size num-heads max-seq-len head-dim)
                    kvmans (atom [kvm-a kvm-b 0 a->b b->a])]
        (transfer! (repeat 1) decode-attention-mask)
        (bind-input! decode-bind embeds-name onnx-decode-embeds)
        (when decode-position-ids (bind-input! decode-bind position-ids-name onnx-decode-position-ids))
        (bind-output! decode-bind logits-name onnx-decode-logits)
        (->TextModel mem-info
                     prefill-sess prefill! prefill-bind
                     decode-sess decode! decode-bind
                     embeds-name decode-embeds onnx-decode-embeds
                     attention-mask-name decode-attention-mask onnx-decode-attention-mask
                     position-ids-name decode-position-ids onnx-decode-position-ids
                     logits-name decode-logits onnx-decode-logits
                     kvmans
                     attention-shape attention-info)))))

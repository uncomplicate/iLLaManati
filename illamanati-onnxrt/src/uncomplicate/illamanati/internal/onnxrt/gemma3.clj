;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.gemma3
  (:require [uncomplicate.commons
             [core :refer [let-release with-release Releaseable release Info size sizeof]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer get-entry position! long-pointer put-entry!]]
            [uncomplicate.neanderthal
             [core :refer [imax transfer! native view-vctr view-ge entry! col]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow]]
            [uncomplicate.diamond
             [tensor :refer [Transfer tensor input output shape data-type layout view-tz offset! transformer *diamond-factory*]]
             [dnn :refer [network activation]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory DiamondFactoryProvider Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer [onnx-data-type-pointer]]
             [core :as onnx
              :refer [graph-optimization! cpu-mem-arena! execution-mode! spin-control! denormal-as-zero!
                      threading-options session memory-info onnx-tensor
                      io-binding input-count output-count cast-type value-tensor-info
                      input-type-info output-type-info tensor-type
                      bind-input! bind-output! runner* options override-dimension! free mutable-data]]
             [impl :refer [*ort-api* *default-allocator* create-tensor* bind-input* bind-output* input-name* output-name*
                           tensor-dimensions*]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.onnxrt.inference :refer [text-model embedding-model]]
            [uncomplicate.illamanati.tokenizer :refer [TokenizerProvider]]
            [uncomplicate.illamanati.internal.huggingface.tokenizer-fast :refer [hft]])
  (:import [clojure.lang IFn AFn]));;TODO tidy up

(def gemma-3-default {:hidden-size 2560
                      :vocab-size 262208
                      :context-len 128000
                      :gemma-3-cpu-text "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                      :gemma-3-cpu-embedding "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                      :tokenizer "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.json"
                      :tokenizer-config "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer_config.json"
                      :embedding-inputs ["input_ids" "image_features"]
                      :embedding-outputs ["inputs_embeds"]
                      :text-inputs ["inputs_embeds" "attention_mask"]
                      :text-outputs ["logits"]})

(defn prefill-options!
  ([opt! batch-size]
   (-> opt!
       (execution-mode! :sequential)
       (override-dimension! "batch_size" batch-size)
       (cpu-mem-arena! false)
       (graph-optimization! :all))))

(defn decode-options! [opt! batch-size]
  (-> opt!
      (execution-mode! :sequential)
      (override-dimension! "batch_size" batch-size)
      (override-dimension! "sequence_length" 1)
      (override-dimension! "num_images" 0)
      (override-dimension! "image_length" 0)
      (cpu-mem-arena! false)
      (graph-optimization! :all)))

(deftype Gemma3 [fact tokenizer tensor-desc create-tz embedding-model! text-model! sample!
                 ^long batch-size] ;;TODO rename to model-agnostic name and generalize
  Releaseable
  (release [_]
    (release embedding-model!)
    (release text-model!)
    (release sample!))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  TokenizerProvider
  (tokenizer [_]
    (tokenizer))
  Transfer
  (input [_]
    (input embedding-model!))
  (output [_]
    (output text-model))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ ids]
    (let [seq-len (count (if (number? (first ids)) ids (first ids)))
          total-seq-len seq-len
          hidden-size (:hidden-size gemma-3-default)
          vocab-size (:vocab-size gemma-3-default)];;TODO generalize
      (with-release [input-ids-desc (tensor-desc [batch-size seq-len] :long)
                     input-embeds-desc (tensor-desc [batch-size seq-len hidden-size] :float)
                     prefill-mask-desc (tensor-desc [batch-size total-seq-len] :long)
                     prefill-logits-desc (tensor-desc [batch-size seq-len vocab-size] :float)
                     input-ids (create-tz input-ids-desc)
                     input-embeds! (create-tz input-embeds-desc)
                     prefill-mask (create-tz prefill-mask-desc)
                     prefill-logits! (create-tz prefill-logits-desc)]
        (transfer! ids (view-ge (view-vctr input-ids) seq-len batch-size))
        (entry! (view-vctr prefill-mask) 1)
        (embedding-model! input-ids input-embeds!)
        (text-model! input-embeds! prefill-mask prefill-logits!)
        (sample! (output text-model!) (input embedding-model!))
        (input embedding-model!))))
  (invoke [_]
    (embedding-model!)
    (text-model!)
    (sample! (output text-model!) (input embedding-model!))
    (input embedding-model!)))

(defn argmax-sampler [logits input-ids!];; TODO ATM just a naive placeholder. Use snapdragan later.
  (let [[batch-size seq-size vocab-size] (shape logits)]
    (if (and (= 1 seq-size) (contiguous? logits) (contiguous? input-ids!))
      (with-release [logits-ge (view-ge (view-vctr logits) vocab-size batch-size)]
        (dotimes [i batch-size]
          (entry! (col input-ids! i) (imax (col logits-ge i)))))
      (dragan-says-ex "This sampler is intended to sample the last token of a contiguous tensor, not the whole history."
                      {:seq-size seq-size}))))

(defn gemma-3-cpu
  ([fact model-path args]
   (let [{:keys [env batch-size hidden-size vocab-size gemma-3-cpu-text gemma-3-cpu-embedding
                 context-len tokenizer tokenizer-config embedding
                 embedding-inputs embedding-outputs text-inputs text-outputs]
          :or {batch-size 1}
          } (into gemma-3-default args)
         vect-fact (neanderthal-factory fact)]
     (let-release [threading-opts (-> (threading-options)
                                      (denormal-as-zero!)
                                      (spin-control! true))
                   prefill-opts (prefill-options! (options) batch-size)
                   decode-opts (decode-options! (options) batch-size)
                   tokenizer-constructor (partial hft (format "%s/%s" model-path tokenizer))
                   text-prefill-sess (session env (format "%s/%s" model-path gemma-3-cpu-text) prefill-opts)
                   text-decode-sess (session env (format "%s/%s" model-path gemma-3-cpu-text) decode-opts)
                   embedding-prefill-sess (session env (format "%s/%s" model-path gemma-3-cpu-embedding) prefill-opts)
                   embedding-decode-sess (session env (format "%s/%s" model-path gemma-3-cpu-embedding) decode-opts)
                   mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
                   gemma-3-embedding (embedding-model fact mem-info embedding-prefill-sess
                                                      embedding-decode-sess embedding-inputs embedding-outputs)
                   gemma-3-text (text-model fact mem-info text-prefill-sess text-decode-sess
                                            text-inputs text-outputs
                                            (output gemma-3-embedding) context-len)]
       (->Gemma3 fact tokenizer-constructor
                 (partial tensor-desc fact vect-fact) (partial create-tz fact vect-fact)
                 gemma-3-embedding gemma-3-text argmax-sampler
                 batch-size))))
  ([model-path args]
   (gemma-3-cpu *diamond-factory* model-path args))
  ([model-path]
   (gemma-3-cpu model-path nil)))

;; TODO decode is covered by snapdragan, with the note that I should support half-precision through tensors (so I need to extend the implementation to support tensor-based data)

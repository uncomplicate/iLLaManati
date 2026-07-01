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
             [core :refer [let-release with-release Releaseable release Info size sizeof releaseable]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer get-entry position! long-pointer put-entry!]]
            [uncomplicate.neanderthal
             [core :refer [imax transfer! native view-vctr view-ge entry! col dim]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow]]
            [uncomplicate.diamond
             [tensor :refer [Transfer tensor input output shape data-type layout view-tz offset! transformer *diamond-factory*]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory DiamondFactoryProvider Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer [onnx-data-type-pointer]]
             [core :as onnx
              :refer [graph-optimization! cpu-mem-arena! execution-mode! spin-control! denormal-as-zero! inter-op-threads! intra-op-threads! config!
                      threading-options session memory-info onnx-tensor
                      io-binding input-count output-count cast-type value-tensor-info
                      input-type-info output-type-info tensor-type
                      bind-input! bind-output! runner* options override-dimension! free mutable-data
                      append-provider!]]
             [impl :refer [*ort-api* *default-allocator* create-tensor* bind-input* bind-output* input-name* output-name*
                           tensor-dimensions*]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.onnxrt.inference :refer [text-model embedding-model]]
            [uncomplicate.illamanati.tokenizer :refer [TokenizerProvider]]
            [uncomplicate.illamanati.internal.huggingface.tokenizer-fast :refer [hft]])
  (:import [clojure.lang IFn AFn]));;TODO tidy up

(def gemma-3-cpu-default {:hidden-size 2560
                          :vocab-size 262208
                          :context-len 128000
                          :gemma-3-text "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                          :gemma-3-embedding "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                          :tokenizer "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.json"
                          :tokenizer-config "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer_config.json"
                          :embedding-inputs ["input_ids" "image_features"]
                          :embedding-outputs ["inputs_embeds"]
                          :text-inputs ["inputs_embeds" "attention_mask"]
                          :text-outputs ["logits"]
                          :opts {}})

(def gemma-3-gpu-default {:hidden-size 2560
                          :vocab-size 262208
                          :context-len 128000
                          :gemma-3-text "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-text.onnx"
                          :gemma-3-embedding "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-embedding.onnx"
                          :tokenizer "gpu/gpu-fp16-io-int4-rtn-block-32/tokenizer.json"
                          :tokenizer-config "gpu/gpu-fp16-io-int4-rtn-block-32/tokenizer_config.json"
                          :embedding-inputs ["input_ids" "image_features"]
                          :embedding-outputs ["inputs_embeds"]
                          :text-inputs ["inputs_embeds" "attention_mask" "position_ids"]
                          :text-outputs ["logits"]
                          :opts {:device-id 0
                                 :copy-in-default-stream true
                                 ;;:conv-algo-search :exhaustive ;;TODO
                                 ;;:conv-use-max-workspace true
                                 :conv-use-max-workspace false
                                 :enable-cuda-graph false
                                 :conv1d-pad-to-nc1d false
                                 :tunable-op-enable false
                                 :tunable-op-tuning-enable false
                                 :tunable-op-max-tuning-duration-ms 0
                                 :skip-layer-norm-strict-mode false
                                 :prefer-nhwc false
                                 :use-ep-level-unified-stream false
                                 :ep-level-unified-stream false
                                 :tf32 true
                                 :fuse-conv-bias false
                                 :sdpa-kernel false
                                 :arena-extend-strategy :requested}
                          })

(defn prefill-options!
  ([opt! batch-size]
   (-> opt!
       (execution-mode! :sequential)
       (override-dimension! "batch_size" batch-size)
       (cpu-mem-arena! false)
       (graph-optimization! :all))))

(defn universal-options!
  ([opt! batch-size]
   (-> opt!
       (execution-mode! :sequential)
       (cpu-mem-arena! false)
       (override-dimension! "batch_size" batch-size)
       (override-dimension! "num_images" 0)
       (override-dimension! "image_length" 0)
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
                 ^long batch-size
                 position-ids?] ;;TODO rename to model-agnostic name and generalize
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
  (invoke [_ ids arg];;TODO generalize. Reuse common bits so it's easier to support new model types.
    ;;TODO maybe even separate prefill and decode objects in inference...
    (let [seq-len (if (number? (first ids)) (count ids) (max (map count ids)))
          past-seq-len (long (get (deref (.attention-shape text-model!)) 1));;TODO reflection
          total-seq-len (+ past-seq-len seq-len)
          hidden-size (:hidden-size gemma-3-cpu-default)
          vocab-size (:vocab-size gemma-3-cpu-default)];;TODO generalize
      (with-release [input-ids-desc (tensor-desc [batch-size seq-len]
                                                 (data-type (input embedding-model!)))
                     input-embeds-desc (tensor-desc [batch-size seq-len hidden-size]
                                                    (data-type (output embedding-model!)))
                     prefill-mask-desc (tensor-desc [batch-size total-seq-len]
                                                    (data-type (.decode-attention-mask text-model!)));;TODO reflection
                     position-ids-desc (when-let [decode-position-ids (.decode-position-ids text-model!)] ;;TODO reflection
                                         (tensor-desc [batch-size seq-len]
                                                      (data-type decode-position-ids)))
                     prefill-logits-desc (tensor-desc [batch-size seq-len vocab-size]
                                                      (data-type (output text-model!)))
                     input-ids (create-tz input-ids-desc)
                     input-embeds! (create-tz input-embeds-desc);;TODO generalize
                     prefill-mask (create-tz prefill-mask-desc)
                     position-ids (when position-ids-desc (create-tz position-ids-desc))
                     prefill-logits! (create-tz prefill-logits-desc)];;TODO generalize
        (transfer! ids (view-ge (view-vctr input-ids) seq-len batch-size))
        (transfer! (repeat (dim prefill-mask) 1) (view-vctr prefill-mask));;TODO entry! seems to fail with integers...
        (embedding-model! input-ids input-embeds!)
        (text-model! input-embeds! prefill-mask position-ids prefill-logits!)
        (sample! arg))))
  (invoke [_ arg]
    (embedding-model!)
    (text-model!)
    (sample! arg)))

(defn argmax-sampler [logits input-ids!];; TODO ATM just a naive placeholder. Use snapdragan later. Use floats, and convert non-floats to floats. Use connector for easy if-needed transformations.
  (let [[batch-size seq-size vocab-size] (shape logits)]
    (if (and (= 1 seq-size) (contiguous? logits) (contiguous? input-ids!))
      (let-release [logits-ge (view-ge (view-vctr logits) vocab-size batch-size)]
        (releaseable (fn [_]
                       (dotimes [i batch-size]
                         (entry! (col input-ids! i) (imax (col logits-ge i))))
                       (vec (seq (view-vctr input-ids!))))))
      (dragan-says-ex "This sampler is intended to sample the last token of a contiguous tensor, not the whole history."
                      {:seq-size seq-size}))))

(defn gemma-3-cpu
  ([fact model-path args]
   (let [{:keys [env batch-size hidden-size vocab-size gemma-3-text gemma-3-embedding
                 context-len tokenizer tokenizer-config embedding
                 embedding-inputs embedding-outputs text-inputs text-outputs]
          :or {batch-size 1}
          } (into gemma-3-cpu-default args)
         vect-fact (neanderthal-factory fact)]
     (let-release [threading-opts (-> (threading-options)
                                      (denormal-as-zero!)
                                      (spin-control! true))
                   universal-opts (universal-options! (options) batch-size)
                   tokenizer-constructor (partial hft (format "%s/%s" model-path tokenizer))
                   text-sess (session env (format "%s/%s" model-path gemma-3-text) universal-opts)
                   embedding-sess (session env (format "%s/%s" model-path gemma-3-embedding)
                                                  universal-opts)
                   mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
                   gemma-3-embedding (embedding-model fact mem-info
                                                      embedding-sess embedding-inputs embedding-outputs)
                   gemma-3-text (text-model fact mem-info text-sess
                                            text-inputs text-outputs
                                            (output gemma-3-embedding) context-len)
                   sample (argmax-sampler (output gemma-3-text) (input gemma-3-embedding))]
       (->Gemma3 fact tokenizer-constructor
                 (partial tensor-desc fact vect-fact) (partial create-tz fact vect-fact)
                 gemma-3-embedding gemma-3-text sample
                 batch-size false))))
  ([model-path args]
   (gemma-3-cpu *diamond-factory* model-path args))
  ([model-path]
   (gemma-3-cpu model-path nil)))

(use 'uncomplicate.snapdragan)
(use 'uncomplicate.snapdragan.cuda)

(defn gemma-3-gpu
  ([fact model-path args]
   (let [{:keys [env batch-size hidden-size vocab-size gemma-3-text gemma-3-embedding
                 context-len tokenizer tokenizer-config embedding
                 embedding-inputs embedding-outputs text-inputs text-outputs
                 opts]
          :or {batch-size 1}
          } (into gemma-3-gpu-default args)
         vect-fact (neanderthal-factory fact)]
     (let-release [threading-opts (-> (threading-options)
                                      (denormal-as-zero!)
                                      (spin-control! true))
                   universal-opts (-> (universal-options! (options) batch-size)
                                      (intra-op-threads! 1)
                                      (inter-op-threads! 1)
                                      (append-provider! :cuda (into opts {:stream (flow fact)}))
                                      (config! {;; :inter-op-spinning true
                                                :intra-op-spinning true
                                                :denormal-as-zero "1"
                                                :use-ort-model-bytes-directly true
                                                :use-ort-model-bytes-for-initializers true
                                                :use-device-allocator-for-initializers true
                                                :initial-cpu-capacity-bytes 2147483648
                                                :use-env-allocators true}))
                   tokenizer-constructor (partial hft (format "%s/%s" model-path tokenizer))
                   embedding-sess (session env (format "%s/%s" model-path gemma-3-embedding) universal-opts)
                   text-sess (session env (format "%s/%s" model-path gemma-3-text) universal-opts)
                   mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
                   gemma-3-embedding (embedding-model fact mem-info
                                                      embedding-sess embedding-inputs embedding-outputs)
                   gemma-3-text (text-model fact mem-info text-sess
                                            text-inputs text-outputs
                                            (output gemma-3-embedding) context-len)
                   sample (uncomplicate.snapdragan/sampler (.ge-decode-logits gemma-3-text) (view-vctr (input gemma-3-embedding)))];;tODO reflection
       (->Gemma3 fact tokenizer-constructor
                 (partial tensor-desc fact vect-fact) (partial create-tz fact vect-fact)
                 gemma-3-embedding gemma-3-text sample
                 batch-size true))))
  ([model-path args]
   (gemma-3-gpu *diamond-factory* model-path args))
  ([model-path]
   (gemma-3-gpu model-path nil)))

;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.optimum-test
  (:require [midje.sweet :refer [facts => throws]]
            [uncomplicate.commons [core :refer [with-release info let-release]]]
            [uncomplicate.clojurecuda.core :refer [init stream synchronize!]]
            [uncomplicate.neanderthal
             [core :refer [iamax transfer! asum scal! native view-vctr entry!]]
             [vect-math :refer [exp!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow]]
            [uncomplicate.diamond
             [tensor :refer [tensor *diamond-factory* input output]]
             [native :refer []]]
            [uncomplicate.diamond.internal.protocols
             :refer [neanderthal-factory
                     Parameters bias weights ParametersSeq parameters DescriptorProvider;;TODO remove. testing only.
                     DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                     DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                     inf-desc train-desc diff-desc Initializable batch-index create-tensor
                     create-tensor-desc neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer :all]
             [model :refer [tensor-desc create-tz]]]
            [uncomplicate.snapdragan :refer [sampler]]
            [uncomplicate.illamanati.internal.onnxrt.optimum :refer :all])
  (:import clojure.lang.ExceptionInfo))

(facts "align-up must align the byte offset strictly to a 256-byte boundary"
       (let [check-alignment (fn [batch heads seq-len dim width]
                               (let [layer-capacity (* batch heads seq-len dim)
                                     stride (align-up layer-capacity width)
                                     total-bytes (* stride width)]
                                 (rem total-bytes 256)))]
         (check-alignment 1 32 4096 128 2) => 0
         (check-alignment 1 16 2048 64 2) => 0
         (check-alignment 2 32 4096 128 4) => 0
         (check-alignment 1 32 1 128 2) => 0))

(facts "align-up is an idempotent operation"
       (align-up 131072 2) => 131072
       (align-up (align-up 55555 2) 2) => (align-up 55555 2))

(let [fact *diamond-factory*
      vect-fact (neanderthal-factory fact)]
  (with-release [batch-size 1
                 seq-len 3
                 total-seq-len 3
                 hidden-size 2560
                 vocab-size 262208
                 threading-opt (-> (threading-options)
                                   (denormal-as-zero!)
                                   (spin-control! true))
                 env (telemetry! (environment :verbose (name (gensym "illamanati_")) {:inter-op-threads 1
                                                                                      :intra-op-threads 8
                                                                                      :denormal-as-zero true
                                                                                      :spin true}))
                 opts (-> (options)
                          (execution-mode! :sequential)
                          (override-dimension! "batch_size" batch-size)
                          (cpu-mem-arena! false)
                          (graph-optimization! :all))
                 sess (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx" opts)
                 mem-info (memory-info (device (neanderthal-factory fact :float)) :device :default)
                 gemma3 (embedding-model fact mem-info sess opts
                                         ["input_ids" "image_features"] ["inputs_embeds"])
                 input-ids (tensor vect-fact [batch-size seq-len] :long :nc)
                 onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                 prefill-embeds (tensor fact [batch-size seq-len hidden-size] :float :ncw)
                 onnx-prefill-embeds (onnx-tensor mem-info [batch-size seq-len hidden-size] (buffer prefill-embeds) :float)]
    (facts
      "Super-basic embedding prefill + 1 decode with Gemma 3."
      (transfer! (range 1 (* batch-size seq-len)) input-ids)
      (time (gemma3 onnx-input-ids nil onnx-prefill-embeds))
      (seq (transfer! prefill-embeds (double-array 3))) => [-1.3093806505203247 0.15749625861644745 0.29183128476142883]
      (transfer! (range 4) (.decode-input-ids gemma3)) ;;TODO do it in decoder-model initialization
      (seq (transfer! (time (gemma3)) (double-array 3))) => [0.5435164570808411 0.10422546416521072 -0.116578109562397]
      (time (gemma3)) => (time (gemma3)))))

(let [fact *diamond-factory*
      vect-fact (neanderthal-factory fact)]
  (with-release [batch-size 1
                 seq-len 3
                 total-seq-len 3
                 hidden-size 2560
                 vocab-size 262208
                 threading-opt (-> (threading-options)
                                   (denormal-as-zero!)
                                   (spin-control! true))
                 env (telemetry! (environment :verbose (name (gensym "illamanati_")) {:inter-op-threads 1
                                                                                      :intra-op-threads 8
                                                                                      :denormal-as-zero true
                                                                                      :spin true}))
                 opts (-> (options)
                          (execution-mode! :sequential)
                          (override-dimension! "batch_size" batch-size)
                          (cpu-mem-arena! false)
                          (graph-optimization! :all))
                 sess (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                               opts)
                 mem-info (memory-info (device (neanderthal-factory fact :float)) :device :default)
                 input-embeds (tensor fact [batch-size seq-len hidden-size] :float :ncw)
                 onnx-input-embeds (onnx-tensor mem-info [batch-size seq-len hidden-size] (buffer input-embeds) :float)
                 gemma3 (decoder-model fact mem-info sess opts
                                       ["inputs_embeds" "attention_mask"] ["logits"]
                                       input-embeds 6)
                 prefill-mask (tensor vect-fact [batch-size total-seq-len] :long :nc)
                 onnx-prefill-mask (onnx-tensor mem-info [batch-size total-seq-len] (buffer prefill-mask) :long)
                 prefill-logits (tensor fact [batch-size seq-len vocab-size] :float :ncw)
                 onnx-prefill-logits (onnx-tensor mem-info [batch-size seq-len vocab-size] (buffer prefill-logits) :float)]
    (facts
      "Super-basic prefill + 1 decode with Gemma 3."
      (transfer! (repeat 0.1) input-embeds)
      (transfer! (repeat 1) prefill-mask)
      (time (gemma3 input-embeds onnx-input-embeds prefill-mask onnx-prefill-mask nil nil prefill-logits onnx-prefill-logits))
      (seq (transfer! prefill-logits (double-array 3))) => [-12.705020904541016 12.854991912841797 0.18065690994262695]
      (transfer! (repeat 0.1) (.decode-embeds gemma3)) ;;TODO do it in text-model initialization
      (seq (transfer! (time (gemma3)) (double-array 3))) => [-12.020591735839844 13.310209274291992 2.906890869140625]
      (time (gemma3)) => (time (gemma3)))))

(let [fact *diamond-factory*
      vect-fact (neanderthal-factory fact)]
  (with-release [tensor-desc (partial tensor-desc fact vect-fact)
                 create-tz (partial create-tz fact vect-fact)
                 hidden-size 2560
                 vocab-size 262208
                 batch-size 1
                 seq-len 6
                 hidden-size 2560
                 past-sequence-length 0
                 total-sequence-length (+ past-sequence-length seq-len)
                 text-input "Belgrade is the capital"
                 env (telemetry! (environment :verbose (name (gensym "illamanati_onnxrt_")) #_{:inter-op-threads 1
                                                                                               :intra-op-threads 8
                                                                                               :denormal-as-zero true
                                                                                               :spin true}))
                 opt (-> (options)
                         (intra-op-threads! 10)
                         (inter-op-threads! 1)
                         (execution-mode! :sequential)
                         (cpu-mem-arena! false)
                         (graph-optimization! :all)
                         (override-dimension! "batch_size" batch-size)
                         (override-dimension! "num_images" 0)
                         (override-dimension! "image_length" 0)
                         (config! {:use-env-allocators true;;
                                   ;; :inter-op-spinning true
                                   :intra-op-spinning true
                                   :denormal-as-zero "1"
                                   :use-ort-model-bytes-directly true
                                   :use-ort-model-bytes-for-initializers true
                                   :use-device-allocator-for-initializers true
                                   :initial-cpu-capacity-bytes 2147483648
                                   :gelu-approximation true
                                   :aot-function-inlining true
                                   :x64quantprecision "1"
                                   :dynamic-block-base 4
                                   ;;          :disable-cpu-ep-fallback true
                                   :strict_shape_type_inference "1"
                                   :allow-released-opsets-only "1"
                                   :use-lut-gemm "1"
                                   :enable-dq-matmulnbits-fusion "1"}))
                 opt-text (-> (options)
                              (intra-op-threads! 10)
                              (inter-op-threads! 1)
                              (execution-mode! :sequential)
                              (cpu-mem-arena! false)
                              (graph-optimization! :all)
                              (override-dimension! "batch_size" batch-size)
                              (override-dimension! "num_images" 0)
                              (override-dimension! "image_length" 0)
                              (config! {:use-env-allocators true;;
                                        ;; :inter-op-spinning true
                                        :intra-op-spinning true
                                        :denormal-as-zero "1"
                                        :use-ort-model-bytes-directly true
                                        :use-ort-model-bytes-for-initializers true
                                        :use-device-allocator-for-initializers true
                                        :initial-cpu-capacity-bytes 2147483648
                                        :gelu-approximation true
                                        :aot-function-inlining true
                                        :x64quantprecision "1"
                                        :dynamic-block-base 4
                                        ;;          :disable-cpu-ep-fallback true
                                        :strict_shape_type_inference "1"
                                        :allow-released-opsets-only "1"
                                        :use-lut-gemm "1"
                                        :enable-dq-matmulnbits-fusion "1"}))
                 input-ids-desc (tensor-desc [batch-size seq-len] :long)
                 input-ids (create-tz input-ids-desc)
                 image-features-desc (tensor-desc [0 0 hidden-size] :float)
                 image-features (create-tz image-features-desc)
                 embeds-desc (tensor-desc [batch-size seq-len hidden-size] :float)
                 embeds (create-tz embeds-desc)
                 attention-mask-desc (tensor-desc [batch-size total-sequence-length] :long)
                 attention-mask (create-tz attention-mask-desc)
                 logits-desc (tensor-desc [batch-size seq-len vocab-size] :float)
                 logits (create-tz logits-desc)
                 mem-info (memory-info :cpu :device :default)
                 onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                 onnx-image-features (onnx-tensor mem-info [0 0 hidden-size] (buffer image-features) :float)
                 onnx-embeds (onnx-tensor mem-info [batch-size seq-len hidden-size] (buffer embeds) :float)
                 onnx-attention-mask (onnx-tensor mem-info [batch-size total-sequence-length]
                                                  (buffer attention-mask) :long)
                 onnx-logits (onnx-tensor mem-info [batch-size seq-len vocab-size] (buffer logits) :float)
                 sess-embedding (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx" opt)
                 sess-text (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx" opt)
                 gemma-3-embedding! (embedding-model fact mem-info sess-embedding opt
                                                     ["input_ids" "image_features"]
                                                     ["inputs_embeds"])
                 gemma-3-text! (decoder-model fact mem-info sess-text opt-text
                                              ["inputs_embeds" "attention_mask"] ["logits"]
                                              (output gemma-3-embedding!) 12)
                 sample! (sampler (view-vctr (output gemma-3-text!))
                                  (view-vctr (input gemma-3-embedding!)))]
    (facts
      "ONNX Gemma3 embedding test."
      (transfer! [2 19727 9619 563 506 5279] (view-vctr input-ids))
      (gemma-3-embedding! onnx-input-ids onnx-image-features onnx-embeds)
      (count (filter pos? (view-vctr embeds))) => 7649
      (entry! (view-vctr attention-mask) 1)
      (gemma-3-text! embeds onnx-embeds
                     attention-mask onnx-attention-mask
                     nil nil
                     logits onnx-logits)
      (sample! 1.0) => [532]
      (seq (view-vctr (input gemma-3-embedding!))) => [532]
      (gemma-3-embedding!)
      (gemma-3-text!)
      (sample! 1.0) => [7488])))

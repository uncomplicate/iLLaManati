;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.gemma3-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons [core :refer [with-release view]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! entry entry! native view-vctr]]
             [block :refer [buffer]]]
            [uncomplicate.diamond
             [tensor :refer [with-diamond input output *diamond-factory* tensor]]
             [native :refer []]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer [environment telemetry! init-ort-api! synchronize-inputs! synchronize-outputs!
                           session options override-dimension! append-provider! graph-optimization! config!
                           execution-mode! memory-info cpu-mem-arena! inter-op-threads! intra-op-threads!
                           onnx-tensor]]
             [model :refer [tensor-desc create-tz]]]
            [uncomplicate.illamanati.tokenizer :refer [tokenizer encode decoder ids]]
            [uncomplicate.illamanati.internal.onnxrt
             [inference :refer [embedding-model text-model]]
             [gemma3 :refer [gemma-3-cpu argmax-sampler]]]))

(with-release [vect-fact (neanderthal-factory *diamond-factory*)
               tensor-desc (partial tensor-desc *diamond-factory* vect-fact)
               create-tz (partial create-tz *diamond-factory* vect-fact)
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
               mem-info (memory-info :cpu :device 0 :default)
               onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
               onnx-image-features (onnx-tensor mem-info [0 0 hidden-size] (buffer image-features) :float)
               onnx-embeds (onnx-tensor mem-info [batch-size seq-len hidden-size] (buffer embeds) :float)
               onnx-attention-mask (onnx-tensor mem-info [batch-size total-sequence-length]
                                                (buffer attention-mask) :long)
               onnx-logits (onnx-tensor mem-info [batch-size seq-len vocab-size] (buffer logits) :float)
               sess-embedding (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx" opt)
               sess-text (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx" opt)
               gemma-3-embedding! (embedding-model mem-info sess-embedding opt
                                                   ["input_ids" "image_features"]
                                                   ["inputs_embeds"])
               gemma-3-text! (text-model mem-info sess-text opt-text
                                         ["inputs_embeds" "attention_mask"]
                                         ["logits"] (output gemma-3-embedding!) 12)
               sample! (argmax-sampler (output gemma-3-text!) (input gemma-3-embedding!))]
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
   (sample! 1.0) => [7488]))

(with-release [model-path "../data/Gemma-3-ONNX/gemma-3-4b-it"
               text-input "Belgrade is the capital"
               env (telemetry! (environment :verbose (name (gensym "illamanati_onnxrt_"))))
               gemma-3! (gemma-3-cpu model-path {:env env
                                                 :context-len 12
                                                 :batch-size 1})
               gemma-3-tokenizer (tokenizer gemma-3!)
               encoding (encode gemma-3-tokenizer text-input);;TODO this could be function (gemma-3-tokenizer text-input) => encode
               st (decoder gemma-3-tokenizer)] ;;TODO (gemma-3-tokenizer) => gives the decoder
  (facts
    "ONNX Gemma3 inference test."
    (println "----------------- prefill starts ------------------")
    (count (ids encoding)) => 6
    (st (first (time (gemma-3! (ids encoding) nil)))) => " and"
    (println "----------------- prefill ends ------------------")
    (println "----------------- decode starts ------------------")
    (st (first (time (gemma-3! nil)))) => " largest"
    (st (first (time (gemma-3! nil)))) => " city"
    (st (first (time (gemma-3! nil)))) => " of"
    (st (first (time (gemma-3! nil)))) => " Serbia"
    (st (first (time (gemma-3! nil)))) => "."
    (println "----------------- decode ends ------------------")))

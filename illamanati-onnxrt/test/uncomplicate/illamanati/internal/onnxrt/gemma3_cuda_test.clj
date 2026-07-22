;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.gemma3-cuda-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons [core :refer [with-release view let-release]]]
            [uncomplicate.clojurecuda.core
             :refer [init default-stream *headers* in-context with-default current-context default-stream
                     reset-context! device last-error synchronize!]]
            [uncomplicate.neanderthal
             [core :refer [transfer! entry entry! native view-vctr]]
             [block :refer [buffer]]]
            [uncomplicate.diamond
             [tensor :refer [with-diamond input output *diamond-factory* tensor]]
             [cuda :refer [cuda-factory]]]
            [uncomplicate.diamond.internal.onnxrt.core :refer [onnx-tensor]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer [environment telemetry! init-ort-api! synchronize-inputs! synchronize-outputs!
                           session options override-dimension! append-provider! graph-optimization! config!
                           execution-mode! memory-info cpu-mem-arena!]]
             [model :refer [tensor-desc create-tz]]]
            [uncomplicate.snapdragan :refer [sampler]]
            [uncomplicate.snapdragan.cuda :refer []]
            [uncomplicate.illamanati.cuda :refer []]
            [uncomplicate.illamanati.internal.protocols :refer [tokenizer]]
            [uncomplicate.illamanati.internal.onnxrt
             [inference :refer [embedding-model decoder-model]]
             [gemma3 :refer [gemma-3-gpu-default token-generator]]]
            [uncomplicate.illamanati.internal.onnxrt.gemma3-test
             :refer [test-generator test-async-generator]]))

#_(with-default
  (reset-context! (device))
  (binding [*headers* {"cuda_fp16.h" nil}]
    (with-diamond cuda-factory []
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
                             (cpu-mem-arena! false)
                             (append-provider! :cuda {:stream default-stream
                                                      :device-id 0
                                                      :conv-use-max-workspace false
                                                      :enable-cuda-graph false
                                                      :conv1d-pad-to-nc1d false
                                                      :tunable-op-enable false
                                                      :tunable-op-tuning-enable false
                                                      :tunable-op-max-tuning-duration-ms 0
                                                      :skip-layer-norm-strict-mode false
                                                      :prefer-nhwc false
                                                      :use-ep-level-unified-stream false
                                                      ;;:tf32 true
                                                      :tf32 false
                                                      :fuse-conv-bias false
                                                      :sdpa-kernel false
                                                      :arena-extend-strategy :requested})
                             (execution-mode! :sequential)
                             (override-dimension! "batch_size" batch-size)
                             (override-dimension! "num_images" 0)
                             (override-dimension! "image_length" 0)
;;                             (override-dimension! "past_sequence_length" past-sequence-length)
                             (config! {:intra-op-spinning true
                                       :denormal-as-zero "1"
                                       :use-ort-model-bytes-directly true
                                       :use-ort-model-bytes-for-initializers true
                                       :use-device-allocator-for-initializers true
                                       :initial-cpu-capacity-bytes 2147483648
                                       :use-env-allocators true})
                             (graph-optimization! :all))
                     input-ids-desc (tensor-desc [batch-size seq-len] :long)
                     input-ids (create-tz input-ids-desc)
                     image-features-desc (tensor-desc [0 0 hidden-size] :half)
                     image-features (create-tz image-features-desc)
                     embeds-desc (tensor-desc [batch-size seq-len hidden-size] :half)
                     embeds (create-tz embeds-desc)
                     attention-mask-desc (tensor-desc [batch-size total-sequence-length] :long)
                     attention-mask (create-tz attention-mask-desc)
                     position-ids-desc (tensor-desc [batch-size seq-len] :long)
                     position-ids (create-tz position-ids-desc)
                     logits-desc (tensor-desc [batch-size seq-len vocab-size] :half)
                     logits (create-tz logits-desc)
                     mem-info (memory-info :cuda :device 0 :default)
                     onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                     onnx-image-features (onnx-tensor mem-info [0 0 hidden-size] (buffer image-features) :half)
                     onnx-embeds (onnx-tensor mem-info [batch-size seq-len hidden-size] (buffer embeds) :half)
                     onnx-attention-mask (onnx-tensor mem-info [batch-size total-sequence-length]
                                                      (buffer attention-mask) :long)
                     onnx-position-ids (onnx-tensor mem-info [batch-size seq-len] (buffer position-ids) :long)
                     onnx-logits (onnx-tensor mem-info [batch-size seq-len vocab-size] (buffer logits) :half)
                     sess-embedding (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-embedding.onnx" opt)
                     sess-text (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-text.onnx" opt)
                     gemma-3-embedding! (embedding-model mem-info sess-embedding opt
                                                         ["input_ids" "image_features"]
                                                         ["inputs_embeds"])
                     gemma-3-text! (decoder-model mem-info sess-text opt
                                               ["inputs_embeds" "attention_mask" "position_ids"]
                                               ["logits"]  (output gemma-3-embedding!) 16)
                     sample! (sampler (.-ge-decode-logits gemma-3-text!) (view-vctr (input gemma-3-embedding!)))]
        (facts
          "ONNX Gemma3 embedding test."
          (transfer! [2 19727 9619 563 506 5279] (view-vctr input-ids))
          (gemma-3-embedding! onnx-input-ids onnx-image-features onnx-embeds)
          (entry! (view-vctr attention-mask) 1)
          ;; (gemma-3-text! embeds onnx-embeds
          ;;                attention-mask onnx-attention-mask
          ;;                position-ids onnx-position-ids
          ;;                logits onnx-logits)
          ;; (sample! 1.0) => [532]
          ;; (seq (view-vctr (native (input gemma-3-embedding!)))) => [532]
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :a
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :b
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :c
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :d
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :e
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :f
          ;; (gemma-3-embedding!)
          ;; (gemma-3-text!)
          ;; (sample! 1.0) => :g
          )))))

#_(with-default
  (reset-context! (device))
  (binding [*headers* {"cuda_fp16.h" nil}]
    (with-diamond cuda-factory []
      (with-release [model-path "../data/Gemma-3-ONNX/gemma-3-4b-it"
                     text-input "Belgrade is the capital"
                     env (telemetry! (environment :verbose (name (gensym "illamanati_onnxrt_"))))
                     gemma-3! (gemma-3-gpu model-path {:env env
                                                       :context-len 12
                                                       :batch-size 1})
                     gemma-3-tokenizer (tokenizer gemma-3!)
                     ids (cons 2 (gemma-3-tokenizer text-input))
                     st (gemma-3-tokenizer)]
        (facts
          "ONNX Gemma3 inference test."
          (println "----------------- prefill starts ------------------")
          (count ids) => 6
          (st (first (time (gemma-3! ids 1.0)))) => " and"
          (println "----------------- prefill ends ------------------")
          (println "----------------- decode starts ------------------")
          (st (first (time (gemma-3! 1.0)))) => " largest"
          (st (first (time (gemma-3! 1.0)))) => " city"
          (st (first (time (gemma-3! 1.0)))) => " of"
          (st (first (time (gemma-3! 1.0)))) => " Serbia"
          (st (first (time (gemma-3! 1.0)))) => "."
          (println "----------------- decode ends ------------------"))))))


(with-default
  (reset-context! (device))
  (binding [*headers* {"cuda_fp16.h" nil}]
    (with-diamond cuda-factory []
      (test-generator gemma-3-gpu-default))))

(with-default
  ;;(reset-context! (device))
  (test-async-generator gemma-3-gpu-default))

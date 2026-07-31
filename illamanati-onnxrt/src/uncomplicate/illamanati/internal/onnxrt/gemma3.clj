;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.gemma3
  (:require [uncomplicate.illamanati.internal.tokenizer.sentencepiece :refer [spp]]
            [uncomplicate.illamanati.internal.onnxrt
             [optimum :refer [optimum-provider]]
             [genai :refer [genai-provider]]]))

(def gemma-3-default {:hidden-size 2560
                      :vocab-size 262208
                      :context-len 128000
                      :batch-size 1
                      :tokenizer [spp "gemma-3-tokenizer.model"]
                      ;;:embedding-inputs ["input_ids" "image_features"]
                      :embedding-inputs ["input_ids"]
                      :embedding-outputs ["inputs_embeds"]
                      :decoder-inputs ["inputs_embeds" "attention_mask"]
                      :decoder-outputs ["logits"]})
;;TODO the in/out list should be validated at input, as non-existing string would stall/crash the vm.

(def cuda-default {:ep [:cuda]
                   :device :cuda
                   :cuda {:device-id 0
                          :copy-in-default-stream false
                          ;;:conv-algo-search :exhaustive ;;TODO
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
                          :arena-extend-strategy :requested}})

(def gemma-3-optimum (into gemma-3-default
                           {:decoder "onnx/decoder_model_merged_q4f16.onnx"
                            :embedding "onnx/embed_tokens_q4f16.onnx"
                            :decoder-inputs ["inputs_embeds" "attention_mask" "num_logits_to_keep"]
                            :provider optimum-provider
                            :device :cpu}))

(def gemma-3-gqa-optimum (into gemma-3-default
                               {:vocab-size 262144
                                :decoder "onnx/model_q4f16.onnx"
                                :decoder-inputs ["input_ids" "attention_mask"]
                                :provider optimum-provider
                                :device :cpu}))

(def gemma-3-genai-cpu (into gemma-3-default
                             {:decoder "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                              :embedding "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                              :embedding-inputs ["input_ids" "image_features"]
                              :provider genai-provider
                              :device :cpu}))

(def gemma-3-genai-cuda (merge gemma-3-default
                               {:decoder  "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-text.onnx"
                                :embedding "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-embedding.onnx"
                                :embedding-inputs ["input_ids" "image_features"]
                                :decoder-inputs ["inputs_embeds" "attention_mask" "position_ids"]
                                :provider genai-provider}
                             cuda-default))

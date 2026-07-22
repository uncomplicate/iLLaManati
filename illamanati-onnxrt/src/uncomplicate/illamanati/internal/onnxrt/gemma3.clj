;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.gemma3
  (:require [uncomplicate.illamanati.internal.sentencepiece :refer [spp]]))

(def gemma-3-default {:hidden-size 2560
                      :vocab-size 262208
                      :context-len 128000
                      :batch-size 1
                      :tokenizer [spp "gemma-3-tokenizer.model"]
                      :embedding-inputs ["input_ids" "image_features"]
                      :embedding-outputs ["inputs_embeds"]})

(def gemma-3-cpu-default (into gemma-3-default
                               {:decoder "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                                :embedding "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                                :decoder-inputs ["inputs_embeds" "attention_mask"]
                                :decoder-outputs ["logits"]
                                :device :cpu}))

(def gemma-3-gpu-default (into gemma-3-default
                               {:decoder "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-text.onnx"
                                :embedding "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-embedding.onnx"
                                :decoder-inputs ["inputs_embeds" "attention_mask" "position_ids"]
                                :decoder-outputs ["logits"]
                                :ep [:cuda]
                                :device :cuda}))

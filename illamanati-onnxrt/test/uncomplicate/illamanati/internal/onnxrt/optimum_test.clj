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
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [view-vctr entry!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.core :refer [transfer!]]
            [uncomplicate.diamond
             [tensor :refer [tensor *diamond-factory* input output]]
             [native :refer []]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer :all]
             [model :refer [tensor-desc create-tz]]]
            [uncomplicate.snapdragan :refer [sampler]]
            [uncomplicate.illamanati.internal.onnxrt.optimum :refer :all])
  (:import clojure.lang.ExceptionInfo))

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
                 env (telemetry! (environment :verbose (name (gensym "illamanati_onnxrt_"))))
                 opt (-> (options)
                         (intra-op-threads! 10)
                         (inter-op-threads! 1)
                         (execution-mode! :sequential)
                         (cpu-mem-arena! false)
                         (graph-optimization! :all)
                         (override-dimension! "batch_size" batch-size))
                 opt-text (options opt)
                 mem-info (memory-info :cpu :device :default)
                 input-ids-desc (tensor-desc [batch-size seq-len] :long)
                 input-ids (create-tz input-ids-desc)
                 onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                 logits-desc (tensor-desc [batch-size seq-len vocab-size] :float)
                 logits (create-tz logits-desc)
                 onnx-logits (onnx-tensor mem-info [batch-size seq-len vocab-size] (buffer logits) :float)
                 sess-embedding (session env "../data/gemma-3-4b-it-ONNX/onnx/embed_tokens_q4f16.onnx" opt)
                 sess-text (session env "../data/gemma-3-4b-it-ONNX/onnx/decoder_model_merged_q4f16.onnx" opt)
                 gemma-3! (embedding-decoder-model fact mem-info sess-embedding opt sess-text opt-text
                                                   ["input_ids"] ["inputs_embeds"]
                                                   ["inputs_embeds" "attention_mask" "num_logits_to_keep"] ["logits"]
                                                   12)]
    (facts
      "ONNX Gemma3 4b model test."
      (transfer! [2 19727 9619 563 506 5279] (view-vctr input-ids))
      (seq (transfer! (gemma-3! input-ids onnx-input-ids) (double-array 3))) => [-10.5546875 -5.2265625 1.841796875]
      (seq (transfer! (gemma-3!) (double-array 3))) => [-9.78125 -1.9111328125 3.40234375])))

(let [fact *diamond-factory*
      vect-fact (neanderthal-factory fact)]
  (with-release [tensor-desc (partial tensor-desc fact vect-fact)
                 create-tz (partial create-tz fact vect-fact)
                 hidden-size 2560
                 vocab-size 262144
                 batch-size 1
                 seq-len 6
                 hidden-size 2560
                 past-sequence-length 0
                 total-sequence-length (+ past-sequence-length seq-len)
                 text-input "Belgrade is the capital"
                 env (telemetry! (environment :verbose (name (gensym "illamanati_onnxrt_"))))
                 opt-text (-> (options)
                              (intra-op-threads! 10)
                              (inter-op-threads! 1)
                              (execution-mode! :sequential)
                              (cpu-mem-arena! false)
                              (graph-optimization! :all)
                              (override-dimension! "batch_size" batch-size))
                 mem-info (memory-info :cpu :device :default)
                 input-ids-desc (tensor-desc [batch-size seq-len] :long)
                 input-ids (create-tz input-ids-desc)
                 onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                 sess-text (session env "../data/gemma-3-1b-it-ONNX-GQA/onnx/model_q4f16.onnx" opt-text)
                 gemma-3! (copy-logits-decoder-model fact mem-info sess-text opt-text
                                                     ["input_ids" "attention_mask"] ["logits"] 12)]
    (facts
      "ONNX Gemma3 1b model test."
      (transfer! [2 19727 9619 563 506 5279] (view-vctr input-ids))
      (seq (transfer! (gemma-3! input-ids onnx-input-ids) (double-array 3))) => [-11.921875 -4.734375 -4.4296875]
      (seq (transfer! (gemma-3!) (double-array 3))) => [-13.1328125 1.12890625 -0.7568359375])))

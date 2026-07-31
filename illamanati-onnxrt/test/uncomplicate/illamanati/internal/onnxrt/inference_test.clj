;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.inference-test
  (:require [midje.sweet :refer [facts => throws]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [view-vctr entry! transfer!]]
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
            [uncomplicate.diamond.internal.onnxrt.core :refer :all]
            [uncomplicate.illamanati.internal.onnxrt.inference :refer :all])
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
                 sess (session env "../data/gemma-3-4b-it-ONNX/onnx/embed_tokens_q4f16.onnx" opts)
                 mem-info (memory-info (device (neanderthal-factory fact :float)) :device :default)
                 gemma3 (embedding-model fact mem-info sess opts ["input_ids"] ["inputs_embeds"])
                 input-ids (tensor vect-fact [batch-size seq-len] :long :nc)
                 onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                 prefill-embeds (tensor fact [batch-size seq-len hidden-size] :float :ncw)
                 onnx-prefill-embeds (onnx-tensor mem-info [batch-size seq-len hidden-size] (buffer prefill-embeds) :float)]
    (facts
      "Super-basic embedding prefill + 1 decode with Gemma 3."
      (transfer! (range 1 (* batch-size seq-len)) input-ids)
      (gemma3 onnx-input-ids onnx-prefill-embeds)
      (seq (transfer! prefill-embeds (double-array 3))) => [-1.25 0.3125 0.3125]
      (transfer! (range 1) (.decode-input-ids gemma3)) ;;TODO do it in decoder-model initialization
      (seq (transfer! (gemma3) (double-array 3))) => [0.52587890625 0.1314697265625 -0.1314697265625]
      (gemma3) => (gemma3))))

(let [fact *diamond-factory*
      vect-fact (neanderthal-factory fact)]
  (with-release [batch-size 1
                 seq-len 3
                 total-seq-len 3
                 hidden-size 2560
                 vocab-size 262144
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
                 sess (session env "../data/gemma-3-1b-it-ONNX-GQA/onnx/model_q4f16.onnx"
                               opts)
                 mem-info (memory-info (device (neanderthal-factory fact :float)) :device :default)
                 input-ids (tensor vect-fact [batch-size seq-len] :long :nc)
                 onnx-input-ids (onnx-tensor mem-info [batch-size seq-len] (buffer input-ids) :long)
                 gemma3 (core-decoder-model fact mem-info sess opts
                                            ["input_ids" "attention_mask"] ["logits"]
                                            input-ids 12)
                 prefill-mask (tensor vect-fact [batch-size total-seq-len] :long :nc)
                 onnx-prefill-mask (onnx-tensor mem-info [batch-size total-seq-len] (buffer prefill-mask) :long)
                 logits (tensor fact [batch-size seq-len vocab-size] :half :ncw)
                 onnx-logits (onnx-tensor mem-info [batch-size seq-len vocab-size] (buffer logits) :half)]
    (facts
      "Super-basic prefill + 1 decode with Gemma 3."
      (transfer! (range 1 (* batch-size seq-len)) input-ids)
      (transfer! (repeat 1) prefill-mask)
      (gemma3 input-ids onnx-input-ids logits onnx-logits)
      (gemma3 input-ids onnx-input-ids logits onnx-logits)
      (seq (transfer! logits (double-array 3))) => [-12.84375 -1.0361328125 -0.69482421875]
      (transfer! (range 1 batch-size) (.decode-input-x gemma3))
      (seq (transfer! (gemma3) (double-array 3))) => [-16.296875 14.2421875 2.587890625]
      (gemma3) => (gemma3))))

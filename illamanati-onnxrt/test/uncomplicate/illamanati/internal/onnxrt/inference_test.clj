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
            [uncomplicate.commons [core :refer [with-release info let-release]]]
            [uncomplicate.clojurecuda.core :refer [init stream synchronize!]]
            [uncomplicate.neanderthal
             [core :refer [iamax transfer! asum scal! native view-vctr entry!]]
             [vect-math :refer [exp!]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow]]
            [uncomplicate.diamond.tensor :refer [tensor]]
            [uncomplicate.diamond.internal.protocols
             :refer [neanderthal-factory
                     Parameters bias weights ParametersSeq parameters DescriptorProvider;;TODO remove. testing only.
                     DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                     DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                     inf-desc train-desc diff-desc Initializable batch-index create-tensor
                     create-tensor-desc neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer :all]
             [model :refer [onnx-multi-io-model]]]
            [uncomplicate.illamanati.internal.onnxrt
             [inference :refer :all]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]])
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

(with-release [fact (dnnl-factory)
               vect-fact (neanderthal-factory fact)
               batch-size 1
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
               prefill-opts (-> (options)
                                (execution-mode! :sequential)
                                (override-dimension! "batch_size" batch-size)
                                (cpu-mem-arena! false)
                                (graph-optimization! :all))
               prefill-sess (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                                     prefill-opts)
               decode-opts (-> (options)
                               (execution-mode! :sequential)
                               (override-dimension! "batch_size" batch-size)
                               (override-dimension! "sequence_length" 1)
                               (cpu-mem-arena! false)
                               (graph-optimization! :all))
               decode-sess (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                                    decode-opts)
               mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
               gemma3 (text-model fact mem-info prefill-sess decode-sess
                                  ["inputs_embeds" "attention_mask"] ["logits"] 4)
               input-embeds (tensor fact [batch-size seq-len hidden-size] :float :ncw)
               prefill-mask (tensor vect-fact [batch-size total-seq-len] :long :nc)
               prefill-logits (tensor fact [batch-size seq-len vocab-size] :float :ncw)]
  (facts
   "Super-basic prefill + 1 decode with Gemma 3."
   (transfer! (repeat 0.1) input-embeds)
   (transfer! (repeat 1) prefill-mask)
   (time (gemma3 input-embeds prefill-mask prefill-logits)) => prefill-logits
   (seq (transfer! prefill-logits (double-array 3))) => [-12.705020904541016 12.854991912841797 0.18065690994262695]
   (transfer! (repeat 0.1) (.decode-embeds gemma3)) ;;TODO do it in text-model initialization
   (seq (transfer! (time (gemma3)) (double-array 3))) => [-12.020591735839844 13.310209274291992 2.906890869140625]
   (time (gemma3)) => (time (gemma3))))

(with-release [fact (dnnl-factory)
               vect-fact (neanderthal-factory fact)
               batch-size 1
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
               prefill-opts (-> (options)
                                (execution-mode! :sequential)
                                (override-dimension! "batch_size" batch-size)
                                (cpu-mem-arena! false)
                                (graph-optimization! :all))
               prefill-sess (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                                     prefill-opts)
               decode-opts (-> (options)
                               (execution-mode! :sequential)
                               (override-dimension! "batch_size" batch-size)
                               (override-dimension! "sequence_length" 1)
                               (override-dimension! "num_images" 0)
                               (override-dimension! "image_length" 0)
                               (cpu-mem-arena! false)
                               (graph-optimization! :all))
               decode-sess (session env "../data/Gemma-3-ONNX/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                                    decode-opts)
               mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
               gemma3 (embedding-model fact mem-info prefill-sess decode-sess
                                       ["input_ids" "image_features"] "inputs_embeds")
               input-ids (tensor vect-fact [batch-size seq-len] :long :nc)
               prefill-embeds (tensor fact [batch-size seq-len hidden-size] :float :ncw)]
  (facts
   "Super-basic embedding prefill + 1 decode with Gemma 3."
   (transfer! (range 1 (* batch-size seq-len)) input-ids)
   (time (gemma3 input-ids prefill-embeds)) => prefill-embeds
   (seq (transfer! prefill-embeds (double-array 3))) => [-1.3093806505203247 0.15749625861644745 0.29183128476142883]
   (transfer! (range 4) (.decode-input-ids gemma3)) ;;TODO do it in text-model initialization
   (seq (transfer! (time (gemma3)) (double-array 3))) => [0.5435164570808411 0.10422546416521072 -0.116578109562397]
   (time (gemma3)) => (time (gemma3))))

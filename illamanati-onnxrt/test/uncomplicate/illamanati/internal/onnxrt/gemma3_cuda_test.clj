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
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.clojurecuda.core
             :refer [init stream synchronize! *headers* with-default current-context default-stream]]
            [uncomplicate.neanderthal.core :refer [transfer! entry]]
            [uncomplicate.diamond.tensor :refer [input]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [uncomplicate.diamond.internal.onnxrt.core :refer [environment telemetry!]]
            [uncomplicate.illamanati.tokenizer :refer [tokenizer encode decoder ids]]
            [uncomplicate.illamanati.internal.onnxrt.gemma3 :refer [gemma-3-gpu]]))

(with-default
  (binding [*headers* {"cuda_fp16.h" nil}]
    (with-release [fact (cudnn-factory default-stream)
                   model-path "../data/Gemma-3-ONNX/gemma-3-4b-it"
                   text-input "Belgrade is the capital"
                   env (telemetry! (environment :verbose (name (gensym "illamanati_onnxrt_")) {:inter-op-threads 1
                                                                                               :intra-op-threads 8
                                                                                               :denormal-as-zero true
                                                                                               :spin true}))
                   gemma-3! (gemma-3-gpu fact model-path {:env env
                                                          :context-len 12
                                                          :batch-size 1})
                   gemma-3-tokenizer (tokenizer gemma-3!)
                   encoding (encode gemma-3-tokenizer text-input)
                   st (decoder gemma-3-tokenizer)]
      (facts
        "ONNX Gemma3 inference test."
        (println "----------------- prefill starts ------------------")
        (count (ids encoding)) => 6
        (st (first (time (let [res (gemma-3! (ids encoding) 1.0)]
                           (uncomplicate.clojurecuda.core/synchronize!)
                           res)))) => "and"
        (println "----------------- prefill ends ------------------")
        (println "----------------- decode starts ------------------")
        (st (first (time (gemma-3! 1.0)))) => " largest"
        (st (first (time (gemma-3! 1.0)))) => " city"
        (st (first (time (gemma-3! 1.0)))) => " of"
        (st (first (time (gemma-3! 1.0)))) => " Serbia"
        (st (first (time (gemma-3! 1.0)))) => "."
        (println "----------------- decode ends ------------------")
        (uncomplicate.clojurecuda.core/synchronize!)))))

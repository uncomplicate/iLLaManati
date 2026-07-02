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
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal.core :refer [transfer! entry]]
            [uncomplicate.diamond.native :refer []]
            [uncomplicate.diamond.internal.onnxrt.core :refer [environment telemetry!]]
            [uncomplicate.illamanati.tokenizer :refer [tokenizer encode decoder ids]]
            [uncomplicate.illamanati.internal.onnxrt.gemma3 :refer [gemma-3-cpu]]))

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

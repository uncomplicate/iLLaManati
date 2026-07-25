;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.generator-test
  (:require [midje.sweet :refer [facts =>]]
            [clojure.string :refer [join]]
            [clojure.core.async :refer [chan >!! <!! close! thread]]
            [uncomplicate.commons [core :refer [with-release info]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory*]]
             [native :refer []]]
            [uncomplicate.illamanati :refer [async-generator async-encoder async-decoder]]
            [uncomplicate.illamanati.internal.protocols :as api]
            [uncomplicate.illamanati.internal.onnxrt
             [optimum :refer [optimum-provider]]
             [gemma3 :refer [gemma-3-cpu-default]]]))

(defn test-generator [config]
  (with-release [model-path "../data/Gemma-3-ONNX/gemma-3-4b-it"
                 text-input "Belgrade is the capital"
                 provider (optimum-provider model-path (into config {:context-len 12}))
                 gen (api/step-engine provider *diamond-factory*)
                 tok (api/tokenizer provider)
                 ids (cons (info tok :bos) (tok text-input))
                 st (tok)]
    (facts
      "ONNX Gemma3 inference test."
      (println "----------------- prefill starts ------------------")
      (count ids) => 6
      (st (first (time (gen ids 1.0)))) => " and"
      (println "----------------- prefill ends ------------------")
      (println "----------------- decode starts ------------------")
      (st (first (time (gen 1.0)))) => " largest"
      (st (first (time (gen 1.0)))) => " city"
      (st (first (time (gen 1.0)))) => " of"
      (st (first (time (gen 1.0)))) => " Serbia"
      (st (first (time (gen 1.0)))) => "."
      (println "----------------- decode ends ------------------"))))

(test-generator gemma-3-cpu-default)

(defn test-async-generator [config]
  (with-release [model-path "../data/Gemma-3-ONNX/gemma-3-4b-it"
                 prompt "Belgrade is the capital"
                 provider (optimum-provider model-path (into config {:context-len 12}))]
    (let [prompt-chan (chan)
          ids-chan (async-encoder provider prompt-chan)
          id-chan (async-generator provider ids-chan)
          text-chan (async-decoder provider id-chan)]

      (facts
        "ONNX Gemma3 async generator test."
        (>!! prompt-chan prompt)
        (<!! text-chan) => " and"
        (time (join (repeatedly 5 #(<!! text-chan)))) => " largest city of Serbia."
        (close! prompt-chan)
        (<!! text-chan) => nil))))

(test-async-generator gemma-3-cpu-default)

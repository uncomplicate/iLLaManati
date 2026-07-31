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
            [clojure.core.async :refer [chan >!! <!! close! thread poll! alt!! io-thread]]
            [clojure.core.async.flow :as flow :refer [create-flow start resume stop inject process]]
            [uncomplicate.commons [core :refer [with-release info]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory*]]
             [native :refer []]]
            [uncomplicate.illamanati
             :refer [async-generator async-encoder async-decoder encoder decoder generator-process]]
            [uncomplicate.illamanati.internal.protocols :as api]))

(defn test-generator [config model-path answer]
  (with-release [text-input "Belgrade is the capital"
                 provider ((:provider config) model-path (into config {:context-len 4096}))
                 gen (api/step-engine provider *diamond-factory*)
                 tok (api/tokenizer provider)
                 ids (cons (info tok :bos) (tok text-input))
                 st (tok)]
    (facts
      "ONNX Gemma3 inference test."
      (println "----------------- prefill starts ------------------")
      (count ids) => 6
      (st (first (time (gen ids 1.0)))) => (answer 0)
      (println "----------------- prefill ends ------------------")
      (println "----------------- decode starts ------------------")
      (st (first (time (gen 1.0)))) => (answer 1)
      (st (first (time (gen 1.0)))) => (answer 2)
      (st (first (time (gen 1.0)))) => (answer 3)
      (st (first (time (gen 1.0)))) => (answer 4)
      (st (first (time (gen 1.0)))) => (answer 5)
      (println "----------------- decode ends ------------------"))))

(defn test-async-generator [config model-path answer]
  (with-release [provider ((:provider config) model-path (into config {:context-len 12}))]
    (let [prompt "Belgrade is the capital"
          prompt-chan (chan)
          ids-chan (async-encoder provider prompt-chan)
          id-chan (async-generator provider ids-chan)
          text-chan (async-decoder provider id-chan)]
      (facts
        "ONNX Gemma3 async generator test."
        (>!! prompt-chan prompt)
        (time (join (repeatedly 6 #(<!! text-chan)))) => answer
        (close! prompt-chan)
        (<!! text-chan) => nil))))

(defn test-flow-generator [config model-path answer]
  (with-release [provider ((:provider config) model-path (into config {:context-len 12}))]
    (facts "Test token generator flow."
           (let [input "Belgrade is the capital"
                 topology {:procs {:enc {:proc (process #'encoder)
                                         :args {:tokenizer provider}}
                                   :dec {:proc (process #'decoder)
                                         :args {:tokenizer provider}}
                                   :gen {:proc (generator-process provider)
                                         :executor :thread}
                                   :monitor {:proc (process (fn
                                                              ([] {:ins {:in :data}})
                                                              ([s] s)
                                                              ([s _] s)
                                                              ([s _ m] [s {::flow/report [m]}])))}}
                           :conns [[[:enc :out] [:gen :prompt]]
                                   [[:gen :token] [:dec :in]]
                                   [[:gen :next] [:gen :step]]
                                   [[:dec :out] [:monitor :in]]]}
                 f (create-flow topology)
                 running-flow (start f)]
             (resume f)
             (inject f [:enc :in] [input])
             (when-let [err (poll! (:error-chan running-flow))] (stop f) err) => nil
             (time (join (repeatedly 6 #(<!! (:report-chan running-flow))))) => answer
             (stop f)))))

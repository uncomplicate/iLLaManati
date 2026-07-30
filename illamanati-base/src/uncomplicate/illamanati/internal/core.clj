;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.core
  (:require [clojure.core.async :refer [<!! >!! alt!! close! chan thread]]
            [clojure.core.async.flow :as flow]
            [uncomplicate.commons.core :refer [with-release let-release release info]]
            [uncomplicate.neanderthal.internal.api :refer [device]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory*]]
            [uncomplicate.illamanati.internal.protocols :as api]))

;; ======= core.async generator loop ============================================================

(defn generator-loop [eos bos context-len step-engine! in-chan tok-chan]
  (with-release [step-engine! step-engine!]
    (let [prompt (cons bos (<!! in-chan))
          arg 1.0]
      (loop [n (count prompt)
             token (first (step-engine! prompt arg))]
        (when (number? token) (>!! tok-chan token))
        (cond (or (not token) (= :stop token)) (close! tok-chan)
              (= :pause token) (recur n (<!! in-chan));;TODO cover a continuation prompt here when that becomes available.
              (= eos token) (recur (inc n) (<!! in-chan));;TODO cover a continuation prompt here when that becomes available.
              (< n context-len) (recur (inc n);;TODO if I receive a signal I should not increase n
                                       (alt!! in-chan ([signal] signal)
                                              :default (first (step-engine! arg))))
              :default (close! tok-chan))))))

;; ======= core.async Flow step functions ==========================================================

(defn generator-describe []
  {:ins {:prompt "Initial prompt token ids"
         :step "Self-trigger loop counter"}
   :outs {:token "Generated token id"
          :next "Self-trigger loop counter"}})

(defn generator-init [fact provider]
  (into (select-keys (info provider) [:eos :bos :context-len])
        {:step-engine (api/step-engine provider fact)}))

(defn generator-transition [state transition]
  (case transition
    ::flow/stop (update state :step-engine release)
    state))

(defn generator-transform [{:keys [step-engine bos eos context-len] :as state} port data]
  (let [[n [token :as msg]] (case port
                              :prompt [(inc (count data)) (step-engine (cons bos data) 1.0)]
                              :step [(inc data) (step-engine 1.0)]
                              [context-len nil])]
    [state (if (and token (not= eos token) (< n context-len))
             {:token msg :next [n]}
             {:token msg})]))

;; ======= Polymorphic generator dispatch =============================================================

(defmethod api/generator :default
  ([provider in-chan tok-chan]
   (let-release [step-engine! (api/step-engine provider *diamond-factory*)]
     (thread (generator-loop (info provider :eos)
                             (info provider :bos)
                             (info provider :context-len)
                             step-engine!
                             in-chan
                             tok-chan)))
   tok-chan)
  ([provider]
   (fn
     ([]
      (generator-describe))
     ([arg-map]
      (generator-init *diamond-factory* provider))
     ([state trans]
      (generator-transition state trans))
     ([state input msg]
      (generator-transform state input msg)))))

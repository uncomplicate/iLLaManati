;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati
  (:require [clojure.core.async :refer [<!! >!! io-thread chan close!]]
            [clojure.core.async.flow :as flow]
            [uncomplicate.commons.core :refer [with-release release info]]
            [uncomplicate.illamanati.internal
             [protocols :as api]
             [core :refer []]]))

(defn async-encoder
  ([provider text-chan ids-chan]
   (io-thread
    (let [tok (api/tokenizer provider)]
      (loop [text (<!! text-chan)]
        (if text
          (do (with-release [encoding (api/encode tok text)]
                (>!! ids-chan (api/ids encoding)) )
              (recur (<!! text-chan)))
          (close! ids-chan)))))
   ids-chan)
  ([provider text-chan]
   (async-encoder provider text-chan (chan))))

(defn async-decoder
  ([provider id-chan text-chan]
   (io-thread
    (let [decoder ((api/tokenizer provider))]
      (loop [id (<!! id-chan)]
        (if id
          (do (when-let [decoded-part (decoder id)]
                (when-not (= "" decoded-part)
                  (>!! text-chan decoded-part) ))
              (recur (<!! id-chan)))
          (close! text-chan)))))
   text-chan)
  ([tokenizer id-chan]
   (async-decoder tokenizer id-chan (chan))))

(defn async-generator
  ([provider in-chan tok-chan]
   (api/generator provider in-chan tok-chan)
   tok-chan)
  ([provider in-chan]
   (async-generator provider in-chan (chan))))

(defn encoder
  ([]
   {:params {:tokenizer "Tokenizer"}
    :ins {:in "Text"}
    :outs {:out "A sequence of token ids"}})
  ([args]
   (api/tokenizer (:tokenizer args)))
  ([tokenizer _]
   tokenizer)
  ([tokenizer _ text]
   (with-release [encoding (api/encode tokenizer text)]
     [tokenizer {:out [(api/ids encoding)]}])))

(defn decoder
  ([]
   {:params {:tokenizer "Tokenizer"}
    :ins {:in "Token id"}
    :outs {:out "Token text"}})
  ([args]
   ((api/tokenizer (:tokenizer args))))
  ([decoder _]
   decoder)
  ([decoder _ id]
   [decoder (when-let [decoded-part (decoder id)]
              (when-not (= "" decoded-part)
                {:out [decoded-part]}))]))

(defn generator
  ([]
   {:params {:provider "Step engine provider"
             :diamond-factory "Deep Diamond tensor factory"}
    :ins {:prompt "Initial prompt token ids"
          :step "Self-trigger loop counter"}
    :outs {:token "Generated token id"
           :next "Self-trigger loop counter"}})
  ([args]
   (let [provider (:provider args)
         fact (:diamond-factory args)]
     (into (select-keys (info provider) [:eos :bos :context-len])
           {:step-engine (api/step-engine provider fact)
            :fact fact})))
  ([state transition]
   (case transition
     ::flow/stop (update state (:step-engine state) release)
     state))
  ([{:keys [step-engine bos eos context-len] :as state} port data]
   (let [[n [token :as msg]] (case port
                               :prompt [(inc (count data)) (step-engine (cons bos data) 1.0)]
                               :step [(inc data) (step-engine 1.0)]
                               [context-len nil])]
     [state (if (and token (not= eos token) (< n context-len))
              {:token msg :next [n]}
              {:token msg})])))

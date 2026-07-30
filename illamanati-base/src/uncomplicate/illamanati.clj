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
            [clojure.core.async.flow :refer [process]]
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

(defn generator-process [provider]
  (process (api/generator provider)))

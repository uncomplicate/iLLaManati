;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.tokenizer
  (:require [clojure.core.async :refer [<!! >!! io-thread chan close!]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.illamanati.internal.protocols :as api]))

(defn encode [tok text]
  (api/encode tok text))

(defn ids [encoding]
  (seq (api/ids encoding)))

(defn tokens [encoding]
  (seq (api/tokens encoding)))

(defn async-encoder
  ([tokenizer text-chan ids-chan]
   (io-thread
    (loop [text (<!! text-chan)]
      (if text
        (do (with-release [encoding (api/encode tokenizer text)]
              (>!! ids-chan (ids encoding)) )
            (recur (<!! text-chan)))
        (close! ids-chan))))
   ids-chan)
  ([tokenizer text-chan]
   (async-encoder tokenizer text-chan (chan))))

(defn async-decoder
  ([tokenizer id-chan text-chan]
   (io-thread
    (let [decoder (tokenizer)]
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

(defn encoder
  ([]
   {:params {:tokenizer "Tokenizer"}
    :ins {:in "Text"}
    :outs {:out "Token ids"}})
  ([args]
   (:tokenizer args))
  ([tokenizer _]
   tokenizer)
  ([tokenizer _ text]
   (with-release [encoding (api/encode tokenizer text)]
     [tokenizer {:out (ids encoding)}])))

(defn decoder
  ([]
   {:params {:tokenizer "Tokenizer"}
    :ins {:in "Token id"}
    :outs {:out "Token text"}})
  ([args]
   ((:tokenizer args)))
  ([decoder _]
   decoder)
  ([decoder _ id]
   [decoder (when-let [decoded-part (decoder id)]
              (when-not (= "" decoded-part)
                {:out [decoded-part]}))]))

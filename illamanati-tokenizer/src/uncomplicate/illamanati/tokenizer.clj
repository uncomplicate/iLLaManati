;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.tokenizer
  (:require [clojure.core.async :refer [<!! >!! io-thread]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.illamanati.internal.protocols :as api]))

(defn encode [tokenizer-provider text]
  (api/encode (api/tokenizer tokenizer-provider) text))

(defn ids [encoding]
  (seq (api/ids encoding)))

(defn tokens [encoding]
  (seq (api/tokens encoding)))

(defn pad-token [config]
  (api/pad-token config))

(defn pad-id [config]
  "TODO")

(defn async-encoder [tokenizer-provider text-chan ids-chan]
  (io-thread
   (with-release [tzr (api/tokenizer tokenizer-provider)]
     (loop [text (<!! text-chan)]
       (when text
         (with-release [encoding (api/encode tzr text)]
           (>!! ids-chan (api/ids encoding)))
         (recur (<!! text-chan)))))))

(defn async-decoder [tokenizer-provider id-chan text-chan]
  (io-thread
   (with-release [decoder ((api/tokenizer tokenizer-provider))]
     (loop [id (<!! id-chan)]
       (when id
         (let [decoded-part (decoder id)]
           (if (and decoded-part (not= "" decoded-part))
             (>!! text-chan decoded-part)))
         (recur (<!! id-chan)))))))

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
           (>!! ids-chan (ids encoding)))
         (recur (<!! text-chan)))))))

(defn async-decoder [tokenizer-provider id-chan text-chan]
  (io-thread
   (let [decoder ((api/tokenizer tokenizer-provider))]
     (loop [id (<!! id-chan)]
       (when id
         (when-let [decoded-part (decoder id)]
           (when-not (= "" decoded-part)
             (>!! text-chan decoded-part)))
         (recur (<!! id-chan)))))))

(defn encoder
  ([]
   {:params {:tokenizer "Tokenizer Provider"}
    :ins {:in "Text"}
    :outs {:out "Token ids"}})
  ([args]
   (api/tokenizer (:tokenizer args)))
  ([tokenizer _]
   tokenizer)
  ([tokenizer _ text]
   (with-release [encoding (api/encode tokenizer text)]
     [tokenizer {:out (ids encoding)}])))

(defn decoder
  ([]
   {:params {:tokenizer "Tokenizer Provider"}
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

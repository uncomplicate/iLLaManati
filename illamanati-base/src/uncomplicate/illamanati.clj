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
  "Creates an encoder that asynchronously tokenizes textual prompts received from `text-chan`
  and puts the sequence of tokens on the `ids-chan`. `provider` is a tokenizer or any object
  that can provide a tokenizer. Currently supported tokenizers are Sentencepiece and Hugging Face Tokenizer."
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
  "Creates a decoder that asynchronously decodes singular tokens from `id-chan`
  and puts the decoded string on the `text-chan`. `provider` is a tokenizer or any object
  that can provide a tokenizer. The tokens are decoded by a stateful streaming tokenizer,
  which takes care of special cases and multi-id strings.
  "
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
  "Creates a token generator that asynchronously generates tokens until stopped,
  either internally (EOS), or externally (by closing the in-chan, or putting `:stop` on `in-chan`).
  The `provider` determines the model and implementation technology of the token generator.
  For available providers, please see technology-specific iLLaManati sub-projects."
  ([provider in-chan tok-chan]
   (api/generator provider in-chan tok-chan)
   tok-chan)
  ([provider in-chan]
   (async-generator provider in-chan (chan))))

(defn encoder
  "The text encoder step-fn compatible with core.async Flow processes. Please see the `:describe` map."
  ([]
   {:params {:tokenizer "Tokenizer provider for this encoder"}
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
  "The token decoder step-fn compatible with core.async Flow processes. Please see the `:describe` map."
  ([]
   {:params {:tokenizer "Tokenizer provider for this encoder"}
    :ins {:in "Single token id"}
    <:outs {:out "Token text, when the token is complete."}})
  ([args]
   ((api/tokenizer (:tokenizer args))))
  ([decoder _]
   decoder)
  ([decoder _ id]
   [decoder (when-let [decoded-part (decoder id)]
              (when-not (= "" decoded-part)
                {:out [decoded-part]}))]))

(defn generator-process
  "Token generator process compatible with core.async Flow processes. `provider` determines the model
  and implementation technology of the token generator. For available providers, please see
  technology-specific iLLaManati sub-projects."
  [provider]
  (process (api/generator provider)))

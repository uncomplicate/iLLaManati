;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.huggingface-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [path]]]
            [uncomplicate.clojure-cpp :refer [pointer pointer-seq int-pointer fill!]]
            [uncomplicate.neanderthal.core :refer [vctr ge transfer! entry! zero cols rows]]
            [uncomplicate.neanderthal.native :refer [native-int native-long]]
            [uncomplicate.illamanati.tokenizer :refer :all]
            [uncomplicate.illamanati.internal.huggingface :refer [hft]])
  (:import ai.djl.huggingface.tokenizers.TokenizerConfig))

(def hftokenizer (hft "data/gemma-3-4b-pt/tokenizer.json"))
(def tokconf (TokenizerConfig/load (path "data/gemma-3-4b-pt" "tokenizer_config.json")))

(facts "HF Tokenizer Hello World."
       (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                      encoding (encode hftokenizer input)
                      pad-enc (encode hftokenizer (pad-token tokconf))
                      to-llm (int-pointer 16)];27.25 microseconds
         (fill! to-llm (first (ids pad-enc)))
         (take 3 (ids encoding)) => [9259 236764 147224]
         (pointer-seq (ids! encoding to-llm)) => [9259 236764 147224 236888 2088 563 506 7606 528 86221 236881 0 0 0 0 0]
         (take 3 (ids! encoding [])) => [9259 236764 147224];; 1 microsecond or less
         ;; (tokens encoding) => ["<bos>" "Hello" "," "▁Gemma" "!" "▁How" "▁is" "▁the" "▁weather" "▁in" "▁Belgrade" "?"]
         ;; (decoder hftokenizer (ids encoding)) => "<bos>Hello, Gemma! How is the weather in Belgrade?"
         ;; (decoder hftokenizer (ids! encoding to-llm)) => "<bos>Hello, Gemma! How is the weather in Belgrade?<pad><pad><pad><pad>"
         (pad-token tokconf) => "<pad>"))

(facts "Streaming decoder test."
       (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                      encoding (encode hftokenizer input)
                      decodable (ids encoding)
                      st (decoder hftokenizer)]
         (tokens encoding) => ["Hello" "," "▁Gemma" "!" "▁How" "▁is" "▁the" "▁weather" "▁in" "▁Belgrade" "?"]
         (apply str (mapv st decodable)) => input))

(facts "HF Tokenizer with vectors."
       (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                      encoding (encode hftokenizer input)
                      pad-enc (encode hftokenizer (pad-token tokconf))
                      v (vctr native-long 16)
                      st (decoder hftokenizer)];27.25 microseconds
         ;;(seq (entry! v (second (ids pad-enc)))) => (repeat 16 0) TODO
         (take 3 (ids encoding)) => [9259 236764 147224]
         (seq (ids! encoding v)) => (seq (transfer! (ids encoding) (zero v)))
         (apply str (map st (seq v))) => "Hello, Gemma! How is the weather in Belgrade?<pad><pad><pad><pad><pad>"))

(facts "HF Tokenizer with matrices."
       (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                      encoding (encode hftokenizer (repeat 3 input))
                      pad-enc (encode hftokenizer (pad-token tokconf))
                      m (ge native-long 11 3)
                      st (decoder hftokenizer)];27.25 microseconds
         ;;(entry! m (second (ids pad-enc)))
         (map seq (ids encoding)) => (repeat 3 [9259 236764 147224 236888 2088 563 506 7606 528 86221 236881])
         (map seq (cols (ids! encoding m))) => (repeat 3 [9259 236764 147224 236888 2088 563 506 7606 528 86221 236881])
         (map #(apply str (map st %)) (cols m)) => (repeat 3 "Hello, Gemma! How is the weather in Belgrade?")))

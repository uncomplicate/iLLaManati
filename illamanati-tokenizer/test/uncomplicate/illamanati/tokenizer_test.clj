;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.tokenizer-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge zero cols rows]]
             [native :refer [lv iv native-long]]]
            [uncomplicate.illamanati.tokenizer :refer :all]))

(defn test-tokenizer [tokenizer]
  (facts "Tokenizer Hello World."
         (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                        encoding (encode tokenizer input)]
           (take 3 (ids encoding)) => [9259 236764 147224]
           (tokens encoding) => ["Hello" "," "▁Gemma" "!" "▁How" "▁is" "▁the" "▁weather" "▁in" "▁Belgrade" "?"]
           (seq (tokenizer input)) => (seq (ids encoding))
           (tokenizer (ids encoding)) => "Hello, Gemma! How is the weather in Belgrade?"
           ;; (pad-token tokconf) => "<pad>"
           )))

(defn test-streaming-decoder [tokenizer]
  (facts "Streaming decoder test."
         (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                        encoding (encode tokenizer input)
                        decodable (ids encoding)
                        st (tokenizer)]
           (tokens encoding) => ["Hello" "," "▁Gemma" "!" "▁How" "▁is" "▁the" "▁weather" "▁in" "▁Belgrade" "?"]
           (apply str (mapv st decodable)) => input)))

(defn test-vector [tokenizer]
  (facts "Tokenizer with vectors."
         (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                        st (tokenizer)]
           (apply str (map st (iv (seq (tokenizer input))))) => "Hello, Gemma! How is the weather in Belgrade?")))

(defn test-matrix [tokenizer]
  (facts "Tokenizer with matrices."
         (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                        encoding (encode tokenizer (repeat 3 input))
                        m (ge native-long 11 3 (map seq (ids encoding)))
                        st (tokenizer)]
           (map seq (ids encoding)) => (repeat 3 [9259 236764 147224 236888 2088 563 506 7606 528 86221 236881])
           (map seq (tokenizer (repeat 3 input))) => (repeat 3 [9259 236764 147224 236888 2088 563 506 7606 528 86221 236881])
           (map seq (cols m)) => (repeat 3 [9259 236764 147224 236888 2088 563 506 7606 528 86221 236881])
           (map #(apply str (map st %)) (cols m)) => (repeat 3 input))))

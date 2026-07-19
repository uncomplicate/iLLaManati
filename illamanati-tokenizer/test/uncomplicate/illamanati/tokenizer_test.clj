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
            [clojure.core.async :refer [chan io-thread >!! <!! poll! timeout alts!!]]
            [clojure.core.async.flow :as flow :refer [create-flow start resume stop inject process]]
            [clojure.string :refer [join]]
            [uncomplicate.commons.core :refer [with-release info]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge zero cols rows]]
             [native :refer [lv iv native-long]]]
            [uncomplicate.illamanati.tokenizer :refer :all]))

(defn test-config [tzr]
  (facts "Tzr config test."
         (info tzr :pad) => 0
         (info tzr :eos) => 1
         (info tzr :bos) => 2
         (info tzr :unk) => 3))

(defn test-tokenizer [tokenizer]
  (facts "Tokenizer Hello World."
         (with-release [input "Hello, Gemma! How is the weather in Belgrade?"
                        encoding (encode tokenizer input)]
           (take 3 (ids encoding)) => [9259 236764 147224]
           (tokens encoding) => ["Hello" "," "▁Gemma" "!" "▁How" "▁is" "▁the" "▁weather" "▁in" "▁Belgrade" "?"]
           (seq (tokenizer input)) => (seq (ids encoding))
           (tokenizer (ids encoding)) => "Hello, Gemma! How is the weather in Belgrade?")))

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


(defn test-async-tokenizer [tok]
  (let [text-chan (chan 100)
        ids-chan (chan 100)
        id-chan (chan 100)
        text2-chan (chan 100)
        atok (async-encoder tok text-chan ids-chan)
        adetok (async-decoder tok id-chan text2-chan)]
    (facts "Test async-tokenizer"
           (>!! text-chan "Hello there!")
           (doseq [id (<!! ids-chan)]
             (>!! id-chan id))
           (join (repeatedly 3 #(<!! text2-chan))) => "Hello there!")))

(defn test-tokenizer-flow [tok]
  (facts "Test tokenizer flow."
         (let [input "Hello there!"
               topology {:procs {:enc {:proc (process #'encoder)
                                       :args {:tokenizer tok}}
                                 :dec {:proc (process #'decoder)
                                       :args {:tokenizer tok}}
                                 :monitor {:proc (process (fn
                                                            ([] {:ins {:in :data}})
                                                            ([s] s)
                                                            ([s _] s)
                                                            ([s _ m] [s {::flow/report [m]}])))}}
                         :conns [[[:enc :out] [:dec :in]]
                                 [[:dec :out] [:monitor :in]]]}
               f (create-flow topology)
               running-flow (start f)]
           (resume f)
           (inject f [:enc :in] [input])
           (join (repeatedly 3 #(<!! (:report-chan running-flow)))) => input
           (stop f))))

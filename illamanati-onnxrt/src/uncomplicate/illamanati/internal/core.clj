;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.core
  (:require [clojure.core.async :refer [<!! >!! io-thread thread alt!! close! chan]]
            [uncomplicate.commons.core :refer [with-release info]]))

(defn generator-loop [eos bos context-len generator! in-chan tok-chan]
  (with-release [generator! generator!]
    (let [prompt (cons bos (<!! in-chan))
          arg 1.0]
      (loop [n (count prompt)
             token (first (generator! prompt arg))]
        (when (number? token) (>!! tok-chan token))
        (cond (or (not token) (= :stop token)) (close! tok-chan)
              (= :pause token) (recur n (<!! in-chan));;TODO cover a continuation prompt here when that becomes available.
              (= eos token) (recur (inc n) (<!! in-chan));;TODO cover a continuation prompt here when that becomes available.
              (< n context-len) (recur (inc n)
                                       (alt!! in-chan ([signal] signal)
                                              :default (first (generator! arg))))
              :default (close! tok-chan))))))

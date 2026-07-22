;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati
  (:require [clojure.core.async :refer [chan]]
            [uncomplicate.illamanati.internal.core :refer [generator]]))

(defn async-generator
  ([provider in-chan tok-chan]
   (generator provider in-chan tok-chan)
   tok-chan)
  ([provider in-chan]
   (async-generator provider in-chan (chan))))

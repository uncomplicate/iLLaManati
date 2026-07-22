;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.native
  (:require [clojure.core.async :refer [thread io-thread <!!]]
            [uncomplicate.commons.core :refer [info release]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory*]]
            [uncomplicate.illamanati :refer [generator]]
            [uncomplicate.illamanati.internal.core :refer [generator-loop]]
            [uncomplicate.illamanati.internal.protocols :as api]))

(defn native-generator
  ([fact provider in-chan tok-chan]
   (thread (generator-loop (info provider :eos)
                           (info provider :bos)
                           (info provider :context-len)
                           (api/generator provider fact)
                           in-chan
                           tok-chan)
           fact))
  ([provider in-chan tok-chan]
   (native-generator *diamond-factory* provider in-chan tok-chan)))

(defmethod generator :cpu [provider in-chan tok-chan]
  (native-generator provider in-chan tok-chan)
  tok-chan)

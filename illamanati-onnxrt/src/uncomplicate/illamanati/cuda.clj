;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.cuda
  (:require [clojure.core.async :refer [thread io-thread <!!]]
            [uncomplicate.commons.core :refer [let-release info release]]
            [uncomplicate.clojurecuda.core :refer [*headers* current-context in-context]]
            [uncomplicate.neanderthal.internal.api :refer [device]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory*]]
             [cuda :refer [cuda-factory]]]
            [uncomplicate.snapdragan.cuda :refer []]
            [uncomplicate.illamanati.internal.core :refer [generator-loop generator]]
            [uncomplicate.illamanati.internal.protocols :as api]))

(defn cuda-generator
  ([fact provider in-chan tok-chan]
   (let-release [generator! (api/generator provider fact)]
     (thread (binding [*diamond-factory* fact]
               (in-context (.-ctx fact);;TODO reflection
                 (generator-loop (info provider :eos)
                                 (info provider :bos)
                                 (info provider :context-len)
                                 generator!
                                 in-chan
                                 tok-chan))
               fact))))
  ([provider in-chan tok-chan]
   (let [fact *diamond-factory*]
     (if (and fact (= :cuda (device (neanderthal-factory fact :float))))
       (cuda-generator fact provider in-chan tok-chan)
       (let-release [fact (cuda-factory)]
         (let [release-fact (cuda-generator fact provider in-chan tok-chan)]
           (io-thread (release (<!! release-fact))
                      provider)))))))

(defmethod generator :cuda [provider in-chan tok-chan]
  (cuda-generator provider in-chan tok-chan)
  tok-chan)

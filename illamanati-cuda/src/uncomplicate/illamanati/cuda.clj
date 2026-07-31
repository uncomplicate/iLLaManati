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
            [clojure.core.async.flow :as flow]
            [uncomplicate.commons.core :refer [let-release info release]]
            [uncomplicate.clojurecuda.core :refer [*headers* current-context in-context]]
            [uncomplicate.neanderthal.internal.api :refer [device]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory*]]
             [cuda :refer [cuda-factory]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.illamanati.internal
             [protocols :as api]
             [core :refer [generator-loop generator-transform generator-describe
                           generator-init generator-transition generator-transform]]]))

(defn cuda-generator
  ([fact provider in-chan tok-chan]
   (let [ctx (.-ctx fact)];;TODO reflection (insignificant, but still)
     (in-context ctx
       (let-release [step-engine! (api/step-engine provider fact)]
         (thread (binding [*diamond-factory* fact]
                   (in-context ctx
                     (generator-loop (info provider :eos)
                                     (info provider :bos)
                                     (info provider :context-len)
                                     step-engine!
                                     in-chan
                                     tok-chan))
                   fact))))))
  ([provider in-chan tok-chan]
   (let [fact *diamond-factory*]
     (if (and fact (= :cuda (device (neanderthal-factory fact :float))))
       (cuda-generator fact provider in-chan tok-chan)
       (let-release [fact (cuda-factory)]
         (let [release-fact (cuda-generator fact provider in-chan tok-chan)]
           (io-thread (release (<!! release-fact)) provider)))))))

;; ======= core.async Flow step function ==========================================================

(defn cuda-generator-init
  ([fact provider]
   (let [ctx (.-ctx fact)]
     (in-context ctx
       (into (generator-init fact provider)
             {:cuda-context ctx}))))
  ([provider]
   (let [fact *diamond-factory*]
     (if (and fact (= :cuda (device (neanderthal-factory fact :float))))
       (cuda-generator-init fact provider)
       (let-release [fact (cuda-factory)]
         (into (cuda-generator-init fact provider) {:release-fact fact}))))))

(defmethod api/generator :cuda
  ([provider in-chan tok-chan]
   (cuda-generator provider in-chan tok-chan)
   tok-chan)
  ([provider]
   (fn
     ([]
      (generator-describe))
     ([arg-map]
      (cuda-generator-init provider))
     ([state trans]
      (let [ctx (:cuda-context state)]
        (if (= ::flow/stop trans)
          (in-context ctx
             (update state :step-engine release)
             (update state :release-fact release))
          (in-context ctx
            (generator-transition state trans)))))
     ([state input msg]
      (in-context (:cuda-context state)
        (generator-transform state input msg))))))

;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.cuda
  (:require [uncomplicate.commons.core :refer [let-release info]]
            [uncomplicate.clojurecuda.core :refer [*headers* current-context in-context]]
            [uncomplicate.diamond
             [tensor :refer [with-diamond]]
             [cuda :refer [cuda-factory]]]
            [uncomplicate.illamanati :refer [generator]]
            [uncomplicate.illamanati.internal.core :refer [generator-loop]]
            [uncomplicate.illamanati.internal.protocols :as api]))

(defmethod generator :cuda [provider in-chan tok-chan]
  (binding [*headers* {"cuda_fp16.h" nil}]
    (let-release [fact (cuda-factory)
                  generator! (api/generator provider fact)]
      (let [ctx (current-context)]
        (fn []
          (with-diamond identity [fact]
            (in-context ctx
              (generator-loop (info provider :eos)
                              (info provider :bos)
                              (info provider :context-len)
                              generator!
                              in-chan
                              tok-chan))))))))

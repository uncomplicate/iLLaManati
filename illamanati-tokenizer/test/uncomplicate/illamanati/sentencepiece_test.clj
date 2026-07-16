;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.sentencepiece-test
  (:require [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [path]]]
            [uncomplicate.illamanati.internal.sentencepiece :refer [spp]]
            [uncomplicate.illamanati.tokenizer-test :refer :all]))

(with-release [sp (spp "../data/gemma-3-tokenizer.model")]
  (test-tokenizer sp)
  (test-streaming-decoder sp)
  (test-vector sp)
  (test-matrix sp)
  (test-async-tokenizer sp)
  (test-tokenizer-flow sp))

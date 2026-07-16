;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.huggingface-test
  (:require [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [path]]]
            [uncomplicate.illamanati.internal.huggingface :refer [hft]]
            [uncomplicate.illamanati.tokenizer-test :refer :all])
  (:import ai.djl.huggingface.tokenizers.TokenizerConfig))

;;(def tokconf (TokenizerConfig/load (path "data/gemma-3-4b-pt" "tokenizer_config.json")))

(with-release [hf (hft "data/gemma-3-4b-pt/tokenizer.json")]
  (test-tokenizer hf)
  (test-streaming-decoder hf)
  (test-vector hf)
  (test-matrix hf)
  (test-async-tokenizer hf))

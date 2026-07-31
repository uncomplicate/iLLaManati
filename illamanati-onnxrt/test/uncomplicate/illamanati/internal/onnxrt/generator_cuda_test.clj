;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.generator-cuda-test
  (:require [uncomplicate.clojurecuda.core :refer [with-default]]
            [uncomplicate.diamond
             [tensor :refer [with-diamond]]
             [cuda :refer [cuda-factory]]]
            [uncomplicate.illamanati.cuda :refer []]
            [uncomplicate.snapdragan.cuda :refer []]
            [uncomplicate.illamanati.internal.onnxrt
             [generator-test :refer [test-generator test-async-generator test-flow-generator]]
             [gemma3 :refer [gemma-3-optimum gemma-3-gqa-optimum gemma-3-genai-cuda cuda-default]]]))

#_(with-default
  (with-diamond cuda-factory []
    (test-generator gemma-3-gpu-default "../data/gemma-3-4b-it-ONNX" [" and" " largest" " city" " of" " Serbia" "."])
    (test-generator gemma-3-gqa-gpu-default "../data/gemma-3-1b-it-ONNX-GQA" [" of" " Serbia" "," " a" " vibrant" " and"])))

#_(with-default
  (test-async-generator gemma-3-gpu-default "../data/gemma-3-4b-it-ONNX" " and largest city of Serbia.")
  (test-async-generator gemma-3-gqa-gpu-default "../data/gemma-3-1b-it-ONNX-GQA" " of Serbia, a vibrant and"))

#_(with-default
  (test-flow-generator gemma-3-gpu-default "../data/gemma-3-4b-it-ONNX" " and largest city of Serbia.")
  (test-flow-generator gemma-3-gqa-gpu-default "../data/gemma-3-1b-it-ONNX-GQA" " of Serbia, a vibrant and"))

(with-default
  #_(test-generator gemma-3-genai-cuda "../data/Gemma-3-ONNX/gemma-3-4b-it/" [" and" " largest" " city" " of" " Serbia" "."])
  #_(test-async-generator gemma-3-genai-cuda "../data/Gemma-3-ONNX/gemma-3-4b-it/" " of Serbia. It is a")
  (test-flow-generator gemma-3-genai-cuda "../data/Gemma-3-ONNX/gemma-3-4b-it/" " of Serbia. It is a"))

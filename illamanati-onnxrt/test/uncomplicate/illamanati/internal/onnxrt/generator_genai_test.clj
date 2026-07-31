;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.generator-genai-test
  (:require [uncomplicate.illamanati.internal.onnxrt
             [generator-test :refer [test-generator test-async-generator test-flow-generator]]
             [gemma3 :refer [gemma-3-genai-cpu gemma-3-genai-cuda]]]))

(test-generator gemma-3-genai-cpu "../data/Gemma-3-ONNX/gemma-3-4b-it/" [" and" " largest" " city" " of" " Serbia" "."])
(test-async-generator gemma-3-genai-cpu "../data/Gemma-3-ONNX/gemma-3-4b-it/" " and largest city of Serbia.")
(test-flow-generator gemma-3-genai-cpu "../data/Gemma-3-ONNX/gemma-3-4b-it/" " and largest city of Serbia.")

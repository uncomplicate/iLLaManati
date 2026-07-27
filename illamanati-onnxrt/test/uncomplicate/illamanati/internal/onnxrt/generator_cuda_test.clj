;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.generator-cuda-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.clojurecuda.core
             :refer [with-default reset-context! device *headers*]]
            [uncomplicate.diamond
             [tensor :refer [with-diamond]]
             [cuda :refer [cuda-factory]]]
            [uncomplicate.illamanati.cuda :refer []]
            [uncomplicate.illamanati.internal.protocols :refer [tokenizer]]
            [uncomplicate.illamanati.internal.onnxrt.gemma3
             :refer [gemma-3-gpu-default gemma-3-gqa-default gemma-3-gqa-gpu-default]]
            [uncomplicate.illamanati.internal.onnxrt.generator-test
             :refer [test-generator test-async-generator test-gqa-generator]]))

(with-default
  (reset-context! (device))
  (binding [*headers* {"cuda_fp16.h" nil}]
    (with-diamond cuda-factory []
      (test-generator gemma-3-gpu-default))))

(with-default
  (test-async-generator gemma-3-gpu-default))

(with-default
  (reset-context! (device))
  (binding [*headers* {"cuda_fp16.h" nil}]
    (with-diamond cuda-factory []
      (test-gqa-generator gemma-3-gqa-gpu-default))))

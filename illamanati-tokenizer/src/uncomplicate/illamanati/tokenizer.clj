;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.tokenizer
  (:require [uncomplicate.illamanati.internal.protocols :as api]))

(defn encoder [tokenizer text]
  (api/encoder tokenizer text))

(defn decoder [tokenizer]
  (api/decoder tokenizer))

(defn pad-token [tokenizer]
  (api/pad-token tokenizer))

(defn ids [encoding]
  (api/ids encoding))

(defn ids! [encoding dst!]
  (api/ids encoding dst!))

(defn tokens [encoding]
  (seq (api/tokens encoding)))

(defn pad-token [config]
  (api/pad-token config))

;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.optimum
  (:require [uncomplicate.commons
             [core :refer [let-release with-release Releaseable release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! view-vctr view-ge submatrix copy! vctr]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [device MemoryContext]]
            [uncomplicate.diamond
             [tensor :refer [input output Transfer shape data-type]]
             [onnxrt :refer [*onnx-options*]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [core :as onnx
              :refer [onnx-tensor cast-type input-type-info output-type-info tensor-type
                      bind-input! bind-output! environment memory-info session options
                      threading-options]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.protocols :refer [TokenizerProvider StepEngineProvider]]
            [uncomplicate.illamanati.internal.onnxrt.inference
             :refer [embedding-model core-decoder-model universal-options! ->StepEngine]]
            [uncomplicate.snapdragan :refer [sampler]])
  (:import [clojure.lang IFn AFn]))

(deftype EmbeddingDecoderModel [fact neand-fact mem-info embedding-model! decoder-model!
                                num-logits-name decode-num-logits onnx-decode-num-logits]
  Releaseable
  (release [_]
    (release onnx-decode-num-logits)
    (release decode-num-logits)
    (release decoder-model!)
    (release embedding-model!))
  Transfer
  (input [_]
    (input embedding-model!))
  (output [_]
    (output decoder-model!))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ input-ids onnx-input-ids]
    (let [embeds-shape (assoc (shape (output embedding-model!)) 1 (get (shape input-ids) 1))
          embeds-dt (data-type (output embedding-model!))]
      (with-release [embeds-desc (tensor-desc fact neand-fact embeds-shape embeds-dt)
                     embeds (create-tz fact neand-fact embeds-desc)
                     onnx-embeds (onnx-tensor mem-info (take 3 embeds-shape) (buffer embeds) embeds-dt)]
        (embedding-model! onnx-input-ids onnx-embeds)
        (decoder-model! embeds onnx-embeds))))
  (invoke [_]
    (embedding-model!)
    (decoder-model!))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn embedding-decoder-model [fact mem-info embedding-sess embedding-opt decoder-sess decoder-opt
                               embedding-inputs embedding-outputs
                               [_ _ num-logits-name :as decoder-inputs] decoder-outputs
                               context-len]
  (with-release [num-logits-type-info (input-type-info decoder-sess 2)]
    (let [neand-fact (neanderthal-factory fact)
          num-logits-info (cast-type num-logits-type-info)
          num-logits-type (tensor-type num-logits-info)]
      (let-release [decode-num-logits (vctr (neanderthal-factory neand-fact num-logits-type) [1])
                    onnx-decode-num-logits (onnx-tensor mem-info (buffer decode-num-logits))
                    embedding (embedding-model fact mem-info embedding-sess embedding-opt
                                               embedding-inputs embedding-outputs)
                    decoder (core-decoder-model fact mem-info decoder-sess decoder-opt
                                                decoder-inputs decoder-outputs
                                                (output embedding) context-len)]
        (bind-input! (.-decode-bind decoder) num-logits-name onnx-decode-num-logits)
        (bind-input! (.-prefill-bind decoder) num-logits-name onnx-decode-num-logits)
        (->EmbeddingDecoderModel fact neand-fact mem-info embedding decoder
                                 num-logits-name decode-num-logits onnx-decode-num-logits)))))

(deftype CopyLogitsDecoderModel [fact neand-fact mem-info decoder-model!]
  Releaseable
  (release [_]
    (release decoder-model!))
  Transfer
  (input [_]
    (input decoder-model!))
  (output [_]
    (output decoder-model!))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ input-ids onnx-input-ids]
    (let [[batch-size seq-len vocab-size :as logits-shape]
          (assoc (shape (output decoder-model!)) 1 (get (shape input-ids) 1) )
          batch-data-len (* seq-len (long vocab-size))
          logits-dt (data-type (output decoder-model!))]
      (with-release [logits-desc (tensor-desc fact neand-fact logits-shape logits-dt)
                     logits (create-tz fact neand-fact logits-desc)
                     onnx-logits (onnx-tensor mem-info (take 3 logits-shape) (buffer logits) logits-dt)
                     last-logits (submatrix (view-ge (view-vctr logits) batch-data-len batch-size)
                                            (- batch-data-len (long vocab-size)) 0 vocab-size batch-size)
                     ge-decode-logits (view-ge (view-vctr (output decoder-model!)) vocab-size batch-size)]
        (decoder-model! input-ids onnx-input-ids logits onnx-logits)
        (copy! last-logits ge-decode-logits))))
  (invoke [_]
    (decoder-model!))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn copy-logits-decoder-model [fact mem-info decoder-sess decoder-opt
                                 decoder-inputs decoder-outputs
                                 context-len]
  (with-release [num-logits-type-info (input-type-info decoder-sess 2)]
    (let [neand-fact (neanderthal-factory fact)]
      (let-release [decoder (core-decoder-model fact mem-info decoder-sess decoder-opt
                                                decoder-inputs decoder-outputs
                                                nil context-len)]
        (->CopyLogitsDecoderModel fact neand-fact mem-info decoder)))))

(defrecord OptimumProvider [merged-args tok]
  Releaseable
  (release [_]
    (release tok))
  Info
  (info [_]
    (into (info tok) merged-args))
  (info [_ info-key]
    (or (info tok info-key)
        (merged-args info-key)))
  MemoryContext
  (compatible? [this other]
    (= (device this) (device other)))
  (device [_]
    (:device merged-args))
  StepEngineProvider
  (step-engine [_ fact]
    (let [vect-fact (neanderthal-factory fact)
          {:keys [batch-size hidden-size vocab-size context-len model-path
                  embedding embedding-inputs embedding-outputs
                  decoder decoder-inputs decoder-outputs]} merged-args]
      (with-release [env-options (threading-options (:env-options merged-args))]
        (let-release [env (or (:env merged-args)
                              (environment (:logging-level merged-args)
                                           (:log-name merged-args)
                                           env-options))
                      mem-info (memory-info (device (neanderthal-factory fact :float))
                                            :device :default)
                      decoder-opt (universal-options! (if-let [opt (:options merged-args)]
                                                        (options opt)
                                                        (options))
                                                      merged-args)
                      decoder-sess (session env (format "%s/%s" model-path decoder) decoder-opt)
                      decoder (if embedding
                                (let-release [embedding-opt (options decoder-opt)
                                              embedding-sess (session env (format "%s/%s" model-path embedding) embedding-opt)]
                                  (embedding-decoder-model fact mem-info
                                                           embedding-sess embedding-opt
                                                           decoder-sess decoder-opt
                                                           embedding-inputs embedding-outputs
                                                           decoder-inputs decoder-outputs
                                                           context-len))
                                (copy-logits-decoder-model fact mem-info decoder-sess decoder-opt
                                                           decoder-inputs decoder-outputs
                                                           context-len))
                      sample (sampler (view-ge (view-vctr (output decoder))
                                               vocab-size batch-size)
                                      (view-vctr (input decoder)))]
          (->StepEngine fact (neanderthal-factory fact)
                        mem-info decoder sample batch-size)))))
  TokenizerProvider
  (tokenizer [this]
    tok))

(defn optimum-provider
  ([model-path args]
   (let-release [tok (let [[tokenizer model-file] (:tokenizer args)]
                       (tokenizer (format "%s/%s" model-path model-file)))]
     (->OptimumProvider (merge *onnx-options* args {:model-path model-path}) tok))))

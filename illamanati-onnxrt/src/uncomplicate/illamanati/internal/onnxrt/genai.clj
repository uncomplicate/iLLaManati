;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.genai
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
            [uncomplicate.diamond.internal.protocols
             :refer [neanderthal-factory DiamondFactoryProvider Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [core :as onnx
              :refer [onnx-tensor cast-type input-type-info output-type-info tensor-type
                      bind-input! bind-output! override-dimension! available-providers
                      execution-mode! cpu-mem-arena! disable-per-session-threads!
                      graph-optimization! inter-op-threads! append-provider! threading-options
                      environment memory-info session options]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.protocols :refer [TokenizerProvider StepEngineProvider]]
            [uncomplicate.illamanati.internal.onnxrt.inference
             :refer [embedding-model core-decoder-model universal-options! ->StepEngine]]
            [uncomplicate.snapdragan :refer [sampler]])
  (:import [clojure.lang IFn AFn]))

(defn genai-options!
  ([opt! args]
   (doto opt!
     (universal-options! args)
     (override-dimension! "num_images" 0)
     (override-dimension! "image_length" 0))))

(deftype CPUModel [fact neand-fact mem-info embedding-model! decoder-model!
                   image-features-name decode-image-features onnx-decode-image-features]
  Releaseable
  (release [_]
    (release onnx-decode-image-features)
    (release decode-image-features)
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
  (invoke [_ input-ids onnx-input-ids image-features onnx-image-features]
    (let [embeds-shape (assoc (shape (output embedding-model!)) 1 (get (shape input-ids) 1))
          embeds-dt (data-type (output embedding-model!))
          [batch-size seq-len vocab-size :as logits-shape]
          (assoc (shape (output decoder-model!)) 1 (get (shape input-ids) 1) )
          batch-data-len (* seq-len (long vocab-size))
          logits-dt (data-type (output decoder-model!))]
      (with-release [embeds-desc (tensor-desc fact neand-fact embeds-shape embeds-dt)
                     embeds (create-tz fact neand-fact embeds-desc)
                     onnx-embeds (onnx-tensor mem-info (take 3 embeds-shape) (buffer embeds) embeds-dt)
                     logits-desc (tensor-desc fact neand-fact logits-shape logits-dt)
                     logits (create-tz fact neand-fact logits-desc)
                     onnx-logits (onnx-tensor mem-info (take 3 logits-shape) (buffer logits) logits-dt)
                     last-logits (submatrix (view-ge (view-vctr logits) batch-data-len batch-size)
                                            (- batch-data-len (long vocab-size)) 0 vocab-size batch-size)
                     ge-decode-logits (view-ge (view-vctr (output decoder-model!)) vocab-size batch-size)]
        (when onnx-image-features
          (bind-input! (.-prefill-bind embedding-model!) image-features-name onnx-decode-image-features))
        (embedding-model! onnx-input-ids onnx-embeds)
        (decoder-model! embeds onnx-embeds logits onnx-logits)
        (copy! last-logits ge-decode-logits))))
  (invoke [this input-ids onnx-input-ids]
    (this input-ids onnx-input-ids nil nil))
  (invoke [_]
    (embedding-model!)
    (decoder-model!))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn cpu-model [fact mem-info embedding-sess embedding-opt decoder-sess decoder-opt
                 [_ image-features-name :as embedding-inputs] embedding-outputs
                 decoder-inputs decoder-outputs
                 context-len]
  (with-release [image-features-type-info (input-type-info embedding-sess 1)]
    (let [neand-fact (neanderthal-factory fact)
          image-features-info (cast-type image-features-type-info)
          [_ _ hidden-size] (onnx/shape image-features-info)
          image-features-type (tensor-type image-features-info)]
      (let-release [image-features-desc (tensor-desc fact neand-fact
                                                     [0 0 hidden-size] image-features-type)
                    decode-image-features (create-tz fact neand-fact image-features-desc)
                    onnx-decode-image-features (onnx-tensor mem-info [0 0 hidden-size]
                                                            (buffer decode-image-features)
                                                            image-features-type)
                    embedding (embedding-model fact mem-info embedding-sess embedding-opt
                                               embedding-inputs embedding-outputs)
                    decoder (core-decoder-model fact mem-info decoder-sess decoder-opt
                                                decoder-inputs decoder-outputs
                                                (output embedding) context-len)]
        (bind-input! (.-prefill-bind embedding) image-features-name onnx-decode-image-features)
        (bind-input! (.-decode-bind embedding) image-features-name onnx-decode-image-features)
        (->CPUModel fact neand-fact mem-info embedding decoder
                    image-features-name decode-image-features onnx-decode-image-features)))))

(defrecord GenAIProvider [merged-args tok]
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
                      decoder-opt (genai-options! (if-let [opt (:options merged-args)]
                                                    (options opt)
                                                    (options))
                                                  merged-args)
                      decoder-sess (session env (format "%s/%s" model-path decoder) decoder-opt)
                      decoder (let-release [embedding-opt (options decoder-opt)
                                            embedding-sess (session env (format "%s/%s" model-path embedding) embedding-opt)]
                                (cpu-model fact mem-info
                                           embedding-sess embedding-opt
                                           decoder-sess decoder-opt
                                           embedding-inputs embedding-outputs
                                           decoder-inputs decoder-outputs
                                           context-len))
                      sample (sampler (view-ge (view-vctr (output decoder))
                                               vocab-size batch-size)
                                      (view-vctr (input decoder)))]
          (->StepEngine fact (neanderthal-factory fact)
                        mem-info decoder sample
                        batch-size)))))
  TokenizerProvider
  (tokenizer [this]
    tok))

(defn genai-provider
  ([model-path args]
   (let-release [tok (let [[tokenizer model-file] (:tokenizer args)]
                       (tokenizer (format "%s/%s" model-path model-file)))]
     (->GenAIProvider (merge *onnx-options* args {:model-path model-path}) tok))))

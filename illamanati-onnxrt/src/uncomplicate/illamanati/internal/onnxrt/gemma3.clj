;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.onnxrt.gemma3
  (:require [uncomplicate.commons
             [core :refer [let-release with-release Releaseable release Info info size sizeof
                           releaseable view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer get-entry position! long-pointer put-entry!]]
            [uncomplicate.neanderthal
             [core :refer [imax transfer! native view-vctr view-ge entry! col dim]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow MemoryContext]]
            [uncomplicate.diamond
             [tensor :refer [Transfer tensor input output shape data-type layout view-tz offset!
                             transformer *diamond-factory*]]
             [onnxrt :refer [*onnx-options*]]]
            [uncomplicate.diamond.internal.protocols
             :refer [neanderthal-factory DiamondFactoryProvider diamond-factory Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer [onnx-data-type-pointer]]
             [core :as onnx
              :refer [graph-optimization! cpu-mem-arena! execution-mode! spin-control! denormal-as-zero! inter-op-threads! intra-op-threads! config!
                      threading-options session memory-info onnx-tensor environment
                      io-binding input-count output-count cast-type value-tensor-info
                      input-type-info output-type-info tensor-type
                      bind-input! bind-output! runner* options override-dimension! free mutable-data
                      append-provider! available-providers disable-per-session-threads!]]
             [impl :refer [*ort-api* *default-allocator* create-tensor* bind-input* bind-output* input-name* output-name*
                           tensor-dimensions*]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.onnxrt.inference :refer [text-model embedding-model]]
            [uncomplicate.illamanati.internal.protocols :refer [TokenizerProvider GeneratorProvider]]
            [uncomplicate.illamanati.internal.sentencepiece :refer [spp]]
            [uncomplicate.snapdragan :refer [sampler]]
            [uncomplicate.snapdragan.cuda :refer []])
  (:import [clojure.lang IFn AFn]))

(def gemma-3-default {:hidden-size 2560
                      :vocab-size 262208
                      :context-len 128000
                      :batch-size 1
                      :tokenizer "gemma-3-tokenizer.model"
                      :embedding-inputs ["input_ids" "image_features"]
                      :embedding-outputs ["inputs_embeds"]})

(def gemma-3-cpu-default (into gemma-3-default
                               {:gemma-3-text "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                                :gemma-3-embedding "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                                :text-inputs ["inputs_embeds" "attention_mask"]
                                :text-outputs ["logits"]
                                :device :cpu}))

(def gemma-3-gpu-default (into gemma-3-default
                               {:gemma-3-text "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-text.onnx"
                                :gemma-3-embedding "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-embedding.onnx"
                                :text-inputs ["inputs_embeds" "attention_mask" "position_ids"]
                                :text-outputs ["logits"]
                                :ep [:cuda]
                                :device :cuda}))

(defn universal-options!
  ([opt! args]
   (let [available-ep (set (available-providers))]
     (doto opt!
       (execution-mode! :sequential)
       (cpu-mem-arena! false)
       (override-dimension! "batch_size" (:batch-size args))
       (override-dimension! "num_images" 0)
       (override-dimension! "image_length" 0)
       (disable-per-session-threads!)
       (graph-optimization! (:graph-optimization args))
       ;; (intra-op-threads! 10)
       (inter-op-threads! 1))
     (doseq [ep (:ep args)]
       (append-provider! opt!
                         (or (available-ep ep)
                             (dragan-says-ex (format "Execution provider %s is not available." ep)
                                             {:requested ep :available available-ep}))
                         (args ep))))
   opt!))

(deftype Gemma3 [fact
                 tensor-desc create-tz
                 mem-info embedding-model! text-model! sample!
                 ^long batch-size
                 ^long hidden-size
                 ^long vocab-size] ;;TODO rename to model-agnostic name and generalize
  Releaseable
  (release [_]
    (release text-model!)
    (release embedding-model!)
    (release sample!)
    (release mem-info))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input embedding-model!))
  (output [_]
    (output text-model))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_ prefill-ids arg];;TODO generalize. Reuse common bits so it's easier to support new model types.
    ;;TODO maybe even separate prefill and decode objects in inference...
    (let [seq-len (if (number? (first prefill-ids))
                    (count prefill-ids)
                    (max (map count prefill-ids)))
          past-seq-len (long (get (deref (.attention-shape text-model!)) 1));;TODO reflection
          total-seq-len (+ past-seq-len seq-len)
          ids-shape [batch-size seq-len]
          ids-dt (data-type (input embedding-model!))
          image-features-shape [0 0 hidden-size]
          image-features-dt(data-type (.-decode-image-features embedding-model!));;TODO reflection
          embeds-shape [batch-size seq-len hidden-size]
          embeds-dt (data-type (output embedding-model!))
          mask-shape [batch-size total-seq-len]
          mask-dt (data-type (.-decode-attention-mask text-model!));;TODO reflection
          position-ids-shape [batch-size seq-len]
          logits-shape [batch-size seq-len vocab-size]
          logits-dt (data-type (output text-model!))]
      (with-release [ids-desc (tensor-desc ids-shape ids-dt)
                     ids (create-tz ids-desc)
                     onnx-ids (onnx-tensor mem-info ids-shape (buffer ids) ids-dt)
                     embeds-desc (tensor-desc embeds-shape embeds-dt)
                     embeds (create-tz embeds-desc)
                     onnx-embeds (onnx-tensor mem-info embeds-shape (buffer embeds) embeds-dt)
                     mask-desc (tensor-desc mask-shape mask-dt)
                     mask (create-tz mask-desc)
                     onnx-mask (onnx-tensor mem-info mask-shape (buffer mask) mask-dt)
                     position-ids-desc (when-let [decode-position-ids (.decode-position-ids text-model!)] ;;TODO reflection
                                         (tensor-desc position-ids-shape (data-type decode-position-ids)))
                     position-ids (when position-ids-desc (create-tz position-ids-desc))
                     onnx-position-ids (when position-ids
                                         (onnx-tensor mem-info position-ids-shape (buffer position-ids)
                                                      (data-type position-ids)))
                     logits-desc (tensor-desc logits-shape logits-dt)
                     logits (create-tz logits-desc)
                     onnx-logits (onnx-tensor mem-info logits-shape (buffer logits) logits-dt)
                     image-features-desc (tensor-desc image-features-shape image-features-dt)
                     image-features (create-tz image-features-desc)
                     onnx-image-features (onnx-tensor mem-info image-features-shape
                                                      (buffer image-features)
                                                      image-features-dt)]
        (transfer! prefill-ids (view-ge (view-vctr ids) seq-len batch-size))
        (embedding-model! onnx-ids onnx-image-features onnx-embeds)
        (entry! (view-vctr mask) 1)
        (text-model! embeds onnx-embeds mask onnx-mask position-ids onnx-position-ids logits onnx-logits)
        (sample! arg))))
  (invoke [_ arg]
    (embedding-model!)
    (text-model!)
    (sample! arg))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defrecord Gemma3Provider [merged-args tok]
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
  GeneratorProvider
  (generator [_ fact]
    (let [vect-fact (neanderthal-factory fact)
          {:keys [batch-size hidden-size vocab-size context-len model-path
                  gemma-3-embedding gemma-3-text
                  embedding embedding-inputs embedding-outputs text-inputs text-outputs]} merged-args]
      (with-release [env-options (threading-options (:env-options merged-args))]
        (let-release [env (or (:env merged-args)
                              (environment (:logging-level merged-args)
                                           (:log-name merged-args)
                                           env-options))
                      embedding-opt (universal-options! (if-let [opt (:options merged-args)]
                                                          (options opt)
                                                          (options))
                                                        merged-args)
                      text-opt (options embedding-opt)
                      mem-info (memory-info (device (neanderthal-factory fact :float))
                                            :device :default)
                      embedding-sess (session env (format "%s/%s" model-path gemma-3-embedding) embedding-opt)
                      text-sess (session env (format "%s/%s" model-path gemma-3-text) text-opt)
                      gemma-3-embedding (embedding-model fact mem-info embedding-sess embedding-opt
                                                         embedding-inputs embedding-outputs)
                      gemma-3-text (text-model fact mem-info text-sess text-opt
                                               text-inputs text-outputs
                                               (output gemma-3-embedding) context-len)
                      sample (sampler (view-ge (view-vctr (output gemma-3-text))
                                               vocab-size batch-size)
                                      (view-vctr (input gemma-3-embedding)))]
          (->Gemma3 fact
                    (partial tensor-desc fact vect-fact)
                    (partial create-tz fact vect-fact)
                    mem-info
                    gemma-3-embedding gemma-3-text sample
                    batch-size hidden-size vocab-size)))))
  TokenizerProvider
  (tokenizer [this]
    tok))

(defn gemma-3
  ([fact model-path args]
   (let-release [tok (spp (format "%s/%s" model-path (:tokenizer gemma-3-default)))]
     (->Gemma3Provider (merge *onnx-options* gemma-3-default args {:model-path model-path}) tok)))
  ([model-path args]
   (gemma-3 *diamond-factory* model-path args))
  ([model-path]
   (gemma-3 model-path gemma-3-cpu-default)))

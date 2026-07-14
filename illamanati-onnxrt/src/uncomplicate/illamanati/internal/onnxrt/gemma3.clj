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
             [core :refer [let-release with-release Releaseable release Info size sizeof
                           releaseable view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [safe get-pointer get-entry position! long-pointer put-entry!]]
            [uncomplicate.neanderthal
             [core :refer [imax transfer! native view-vctr view-ge entry! col dim]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.neanderthal.internal.api :refer [device flow]]
            [uncomplicate.diamond
             [tensor :refer [Transfer tensor input output shape data-type layout view-tz offset! transformer *diamond-factory*]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory DiamondFactoryProvider Initializable]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer [onnx-data-type-pointer]]
             [core :as onnx
              :refer [graph-optimization! cpu-mem-arena! execution-mode! spin-control! denormal-as-zero! inter-op-threads! intra-op-threads! config!
                      threading-options session memory-info onnx-tensor
                      io-binding input-count output-count cast-type value-tensor-info
                      input-type-info output-type-info tensor-type
                      bind-input! bind-output! runner* options override-dimension! free mutable-data
                      append-provider!]]
             [impl :refer [*ort-api* *default-allocator* create-tensor* bind-input* bind-output* input-name* output-name*
                           tensor-dimensions*]]
             [model :refer [create-tz tensor-desc]]]
            [uncomplicate.illamanati.internal.onnxrt.inference :refer [text-model embedding-model]]
            [uncomplicate.illamanati.tokenizer :refer [TokenizerProvider]]
            [uncomplicate.illamanati.internal.sentencepiece :refer [spp]]
            [uncomplicate.snapdragan :refer [sampler]]
            [uncomplicate.snapdragan.cuda :refer []])
  (:import [clojure.lang IFn AFn]))

(def gemma-3-cpu-default {:hidden-size 2560
                          :vocab-size 262208
                          :context-len 128000
                          :gemma-3-text "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx"
                          :gemma-3-embedding "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx"
                          :tokenizer "gemma-3-tokenizer.model"
                          ;;:tokenizer-config "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer_config.json"
                          :embedding-inputs ["input_ids" "image_features"]
                          :embedding-outputs ["inputs_embeds"]
                          :text-inputs ["inputs_embeds" "attention_mask"]
                          :text-outputs ["logits"]
                          :opts {}})

(def gemma-3-gpu-default {:hidden-size 2560
                          :vocab-size 262208
                          :context-len 128000
                          :gemma-3-text "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-text.onnx"
                          :gemma-3-embedding "gpu/gpu-fp16-io-int4-rtn-block-32/gemma-3-embedding.onnx"
                          :tokenizer "gemma-3-tokenizer.model"
                          ;;:tokenizer-config "gpu/gpu-fp16-io-int4-rtn-block-32/tokenizer_config.json"
                          :embedding-inputs ["input_ids" "image_features"]
                          :embedding-outputs ["inputs_embeds"]
                          :text-inputs ["inputs_embeds" "attention_mask" "position_ids"]
                          :text-outputs ["logits"]
                          :opts {:device-id 0
                                 :copy-in-default-stream true
                                 ;;:conv-algo-search :exhaustive ;;TODO
                                 :conv-use-max-workspace false
                                 :enable-cuda-graph false
                                 :conv1d-pad-to-nc1d false
                                 :tunable-op-enable false
                                 :tunable-op-tuning-enable false
                                 :tunable-op-max-tuning-duration-ms 0
                                 :skip-layer-norm-strict-mode false
                                 :prefer-nhwc false
                                 :ep-level-unified-stream false
                                 ;;:tf32 true
                                 :tf32 false
                                 :fuse-conv-bias false
                                 :sdpa-kernel false
                                 :arena-extend-strategy :requested}})

(defn universal-options!
  ([opt! batch-size]
   (-> opt!
       (execution-mode! :sequential)
       (cpu-mem-arena! false)
       (override-dimension! "batch_size" batch-size)
       (override-dimension! "num_images" 0)
       (override-dimension! "image_length" 0)
       (graph-optimization! :all))))

(deftype Gemma3 [fact tensor-desc create-tz
                 mem-info embedding-model! text-model! sample!
                 ^long batch-size] ;;TODO rename to model-agnostic name and generalize
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
          hidden-size (:hidden-size gemma-3-cpu-default);;TODO generalize cpu/gpu
          vocab-size (:vocab-size gemma-3-cpu-default);;TODO generalize cpu/gpu
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
        (let [res (sample! arg)]
          (uncomplicate.clojurecuda.core/synchronize! (flow fact))
          res))))
  (invoke [_ arg]
    (embedding-model!)
    (text-model!)
    (sample! arg))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn argmax-sampler [logits input-ids!];; TODO ATM just a naive placeholder. Use snapdragan later. Use floats, and convert non-floats to floats. Use connector for easy if-needed transformations.
  (let [[batch-size seq-size vocab-size] (shape logits)]
    (if (and (= 1 seq-size) (contiguous? logits) (contiguous? input-ids!))
      (let-release [logits-ge (view-ge (view-vctr logits) vocab-size batch-size)]
        (releaseable (fn [_]
                       (dotimes [i batch-size]
                         (entry! (col input-ids! i) (imax (col logits-ge i))))
                       (vec (seq (view-vctr input-ids!))))))
      (dragan-says-ex "This sampler is intended to sample the last token of a contiguous tensor, not the whole history."
                      {:seq-size seq-size}))))

(defn gemma-3-tokenizer [model-path]
  (spp (format "%s/%s" model-path (:tokenizer gemma-3-cpu-default))))

(defn gemma-3-cpu
  ([fact model-path args]
   (let [{:keys [env batch-size hidden-size vocab-size gemma-3-embedding gemma-3-text
                 context-len embedding
                 embedding-inputs embedding-outputs text-inputs text-outputs]
          :or {batch-size 1}
          } (into gemma-3-cpu-default args)
         vect-fact (neanderthal-factory fact)]
     (let-release [;; threading-opts (-> (threading-options)
                   ;;                    (denormal-as-zero!)
                   ;;                    (spin-control! true))
                   embedding-opt (universal-options! (-> (options)
                                                         (intra-op-threads! 10)
                                                         (inter-op-threads! 1))
                                                     batch-size)
                   text-opt (universal-options! (-> (options)
                                                    (intra-op-threads! 10)
                                                    (inter-op-threads! 1))
                                                batch-size)
                   embedding-sess (session env (format "%s/%s" model-path gemma-3-embedding) embedding-opt)
                   text-sess (session env (format "%s/%s" model-path gemma-3-text) text-opt)
                   mem-info (memory-info (device (neanderthal-factory fact :float)) :device :default)
                   gemma-3-embedding (embedding-model fact mem-info embedding-sess embedding-opt
                                                      embedding-inputs embedding-outputs)
                   gemma-3-text (text-model fact mem-info text-sess text-opt
                                            text-inputs text-outputs
                                            (output gemma-3-embedding) context-len)
                   sample (argmax-sampler (output gemma-3-text) (input gemma-3-embedding))]
       (->Gemma3 fact
                 (partial tensor-desc fact vect-fact) (partial create-tz fact vect-fact)
                 mem-info
                 gemma-3-embedding gemma-3-text sample
                 batch-size))))
  ([model-path args]
   (gemma-3-cpu *diamond-factory* model-path args))
  ([model-path]
   (gemma-3-cpu model-path nil)))

(defn gemma-3-gpu ;;TODO this should be a protocol that dispatches based on factory!
  ([fact env model-path args]
   (let [{:keys [batch-size hidden-size vocab-size gemma-3-embedding gemma-3-text
                 context-len embedding
                 embedding-inputs embedding-outputs text-inputs text-outputs
                 opts]
          :or {batch-size 1}
          } (into gemma-3-gpu-default args)
         vect-fact (neanderthal-factory fact)]
     (let-release [;; threading-opts (-> (threading-options)
                   ;;                    (denormal-as-zero!)
                   ;;                    (spin-control! true))
                   embedding-opt (-> (universal-options! (options) batch-size)
                                     ;; (intra-op-threads! 1)
                                     ;; (inter-op-threads! 1)
                                     (append-provider! :cuda (into opts {:stream (flow fact)}))
                                     (config! {;; :inter-op-spinning true
                                               :intra-op-spinning true
                                               :denormal-as-zero "1"
                                               :use-ort-model-bytes-directly true
                                               :use-ort-model-bytes-for-initializers true
                                               :use-device-allocator-for-initializers true
                                               :initial-cpu-capacity-bytes 2147483648
                                               :use-env-allocators true}))
                   text-opt (-> (universal-options! (options) batch-size)
                                ;; (intra-op-threads! 1)
                                ;; (inter-op-threads! 1)
                                (append-provider! :cuda (into opts {:stream (flow fact)}))
                                (config! {;; :inter-op-spinning true
                                          :intra-op-spinning true
                                          :denormal-as-zero "1"
                                          :use-ort-model-bytes-directly true
                                          :use-ort-model-bytes-for-initializers true
                                          :use-device-allocator-for-initializers true
                                          :initial-cpu-capacity-bytes 2147483648
                                          :use-env-allocators true}))

                   mem-info (memory-info (device (neanderthal-factory fact :float)) :device :default)
                   sess-embedding (session env (format "%s/%s" model-path gemma-3-embedding) embedding-opt)
                   sess-text (session env (format "%s/%s" model-path gemma-3-text) text-opt)
                   gemma-3-embedding (embedding-model fact mem-info sess-embedding embedding-opt
                                                      embedding-inputs embedding-outputs)
                   gemma-3-text (text-model fact mem-info sess-text text-opt
                                            text-inputs text-outputs
                                            (output gemma-3-embedding) context-len)
                   sample (sampler (.ge-decode-logits gemma-3-text) (view-vctr (input gemma-3-embedding)))];;tODO reflection
       (gemma-3-embedding)
       (->Gemma3 fact
                 (partial tensor-desc fact vect-fact) (partial create-tz fact vect-fact)
                 mem-info gemma-3-embedding gemma-3-text sample
                 batch-size))))
  ([env model-path args]
   (gemma-3-gpu *diamond-factory* env model-path args))
  ([model-path]
   (gemma-3-gpu model-path nil)))

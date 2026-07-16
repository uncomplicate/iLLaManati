;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.sentencepiece
  (:require [clojure.java.io :refer [input-stream]]
            [uncomplicate.commons
             [core :refer [with-release Releaseable release size]]
             [utils :refer [path dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [integer :refer [entry]]]
            [uncomplicate.illamanati.internal
             [protocols :as api]
             [streaming-decoder :refer [streaming-decoder sentencepiece-lookup]]])
  (:import [clojure.lang IFn AFn Seqable]
           [org.bytedeco.javacpp IntPointer LongPointer ShortPointer BytePointer]
           [org.bytedeco.sentencepiece SentencePieceProcessor Status IntVector StringVector]
           [uncomplicate.neanderthal.internal.api IntegerVector IntegerMatrix LayoutNavigator]))

(defprotocol Encodable
  (encode [src encoder]))

(defprotocol Decodable
  (decode [src decoder]))

(declare ->IntVectorEncoder)

;; ================== Encodables ===============================================

(extend-type String
  Encodable
  (encode [text ^SentencePieceProcessor processor]
    (->IntVectorEncoder (.EncodeAsIds processor text)
                        (delay (.EncodeAsPieces processor text)))))

(extend-type (Class/forName "[Ljava.lang.String;")
  Encodable
  (encode [text ^SentencePieceProcessor processor]
    (mapv #(encode % processor) text)))

(extend-type Seqable
  Encodable
  (encode [text ^SentencePieceProcessor processor]
    (mapv #(encode % processor) text)))

;; ----------------- 2D arrays -------------------------------------------------

(extend-type (Class/forName "[[J")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type (Class/forName "[[I")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type (Class/forName "[[S")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

;; -------------- 1D arrays ----------------------------------------------------

(extend-type (Class/forName "[J")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type (Class/forName "[I")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type (Class/forName "[S")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type (Class/forName "[B")
  Decodable
  (decode [srcs processor]
    (let [len (alength srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (aget srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type LongPointer
  Decodable
  (decode [srcs processor]
    (let [len (size srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (.get ^LongPointer srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type IntPointer
  Decodable
  (decode [srcs processor]
    (let [len (size srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (.get ^IntPointer srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type ShortPointer
  Decodable
  (decode [srcs processor]
    (let [len (size srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (.get ^ShortPointer srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type BytePointer
  Decodable
  (decode [srcs processor]
    (let [len (size srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (.get ^BytePointer srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type IntegerVector
  Decodable
  (decode [srcs processor]
    (let [len (dim srcs)]
      (with-release [v (IntVector. len)]
        (dotimes [i len]
          (.put ^IntVector v i (int (entry srcs i))))
        (.DecodeIds ^SentencePieceProcessor processor v)))))

(extend-type Seqable
  Decodable
  (decode [srcs processor]
    (let [len (count srcs)]
      (with-release [v (IntVector. len)]
        (reduce (fn ^long [^long i ^long e]
                  (.put v i (int e))
                  (inc i))
                0
                srcs)
        (.DecodeIds ^SentencePieceProcessor processor v)))))

;; ============== SentencePiece extensions ===============================================

(defn sp-vocabulary-array [^SentencePieceProcessor processor]
  (let [size (.GetPieceSize processor)
        lookup-array ^"[Ljava.lang.String;" (make-array String size)]
    (dotimes [id size]
      (let [token (.IdToPiece processor id)]
        (aset lookup-array id token)))
    [sentencepiece-lookup lookup-array]))

(defn sp-streaming-decoder
  ([^SentencePieceProcessor processor ^long capacity]
   (let [[bpe-lookup vocabulary-lookup] (sp-vocabulary-array processor)]
     (streaming-decoder bpe-lookup vocabulary-lookup capacity)))
  ([processor]
   (sp-streaming-decoder processor 1024)))

(deftype IntVectorEncoder [^IntVector encoding-ids encoding-tokens]
  java.lang.AutoCloseable
  (close [this]
    (release this))
  Releaseable
  (release [_]
    (.close encoding-ids)
    (when (realized? encoding-tokens)
      (.close ^StringVector (deref encoding-tokens)))
    true)
  IFn
  (invoke [_]
    (.get encoding-ids))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  api/EncodingIds
  (ids [this]
    (.get encoding-ids))
  api/EncodingTokens
  (tokens [this]
    (.get ^StringVector (deref encoding-tokens))))

(deftype SPP [^SentencePieceProcessor processor decoder-fn]
  java.lang.AutoCloseable
  (close [_]
    (.close processor))
  Releaseable
  (release [_]
    (.close processor)
    true)
  api/TokenizerProvider
  (tokenizer [this]
    this)
  IFn
  (invoke [_ text-or-token-ids]
    (if (or (string? text-or-token-ids) (string? (first text-or-token-ids)))
      (with-release [enc (encode text-or-token-ids processor)]
        (api/ids enc))
      (decode text-or-token-ids processor)))
  (invoke [_]
    decoder-fn)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  api/Encoder
  (encode [this text]
    (encode text processor)))

(extend-type java.util.Collection
  api/EncodingIds
  (ids [this]
    (mapv api/ids this))
  api/EncodingTokens
  (tokens [this]
    (mapv api/tokens this)))

(defn spp [source]
  (cond
    (bytes? source) (throw (UnsupportedOperationException. "TODO"))
    (string? source) (let [res ^SentencePieceProcessor (SentencePieceProcessor.)
                           status ^Status (.Load res source)]
                       (if (.ok status)
                         (->SPP res (sp-streaming-decoder res))))
    :default (dragan-says-ex "This source type is unsupported." {:requested (type source)
                                                                 :required [String]})))

(extend-type SentencePieceProcessor
  api/Config
  (pad-token [this]
    (let [pad (.pad_id this)]
      (when (< -1 pad)
        (.IdToPiece this pad)))))

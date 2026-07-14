;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.streaming-decoder
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [charred.api :refer [read-json]])
  (:import java.io.InputStream
           [java.nio ByteBuffer CharBuffer BufferOverflowException]
           [java.nio.charset Charset StandardCharsets CodingErrorAction]))

(defn bpe-lookup-array []
  (let [ba (byte-array 324)
        fill! (fn [^long start ^long end ^long unicode-start]
                (dotimes [i (- end start)]
                  (aset-byte ba (+ unicode-start i) (unchecked-byte (+ start i)))))]
    (fill! 33 127 33)
    (fill! 161 173 161)
    (fill! 174 256 174)
    (fill! 0 33 256)
    (fill! 127 161 289)
    (fill! 173 174 323)
    (fn ^long [^long id]
      (long (if (< id 324)
             (aget ba id)
             63)))))

(defn bpe-reconstruct-lookup [^"[Ljava.lang.String;" vocabulary-lookup]
  (let [ba (byte-array 65536)]
    (dotimes [i 128]
      (aset-byte ba i (byte i)))
    (dotimes [id (min 256 (alength vocabulary-lookup))]
      (let [token ^String (aget vocabulary-lookup id)]
        (when (= 1 (.length token))
          (let [unicode-char (int (.charAt token 0))]
            (aset-byte ba unicode-char (unchecked-byte id))))))
    (fn ^long [^long id]
      (long (aget ba id)))))

(defn sentencepiece-lookup ^long [^long c]
  (if (= 9601 c) ;; (▁)
    (long 32)    ;; Space (ASCII 32)
    c))

(defn huf-vocabulary-array [^InputStream is]
  (let [data (read-json is)
        full-vocab (into (get-in data ["model" "vocab"])
                         (map (juxt #(get % "content")
                                    #(get % "id"))
                              (get data "added_tokens")))
        max-id (long (apply max (vals full-vocab)))
        lookup-array ^"[Ljava.lang.String;" (make-array String (inc max-id))]
    (loop [detected nil vocab full-vocab]
      (if-let [[token-str id] (first vocab)]
        (do (aset lookup-array id (str token-str))
            (recur (case token-str
                     "\u2581" :sentence-piece
                     "Ġ" :bpe
                     detected)
                   (rest vocab)))
        [(case detected
           :bpe (bpe-lookup-array)
           :sentence-piece sentencepiece-lookup
           (bpe-reconstruct-lookup lookup-array))
         lookup-array]))))

(defn str->buffer!
  [bpe-lookup ^String token ^ByteBuffer res!]
  (let [len (.length token)
        remaining (.remaining res!)]
    (when (< remaining len)
      (throw (dragan-says-ex "Decoder stream needs more internal buffer capacity for this token."
                             {:capacity (.capacity res!)
                              :remaining remaining
                              :requested len})))
    (dotimes [i (.length token)]
      (let [c (int (.charAt token i))]
        (.put res! (unchecked-byte (bpe-lookup c))))))
  res!)

(defn streaming-decoder
  ([bpe-lookup ^"[Ljava.lang.String;" vocabulary-lookup ^long capacity]
   (let [decoder (doto (.newDecoder StandardCharsets/UTF_8)
                   (.onMalformedInput CodingErrorAction/REPLACE)
                   (.onUnmappableCharacter CodingErrorAction/REPLACE))
         in (ByteBuffer/allocate capacity)
         out (CharBuffer/allocate capacity)]
     (fn [^long id]
       (try
         (str->buffer! bpe-lookup (aget vocabulary-lookup id) in)
         (.flip in)
         (.clear out)
         (.decode decoder in out false)
         (.flip out)
         (.toString out)
         (finally
           (if (.hasRemaining in)
             (.compact in)
             (.clear in)))))))
  ([^InputStream vocabulary-stream ^long capacity]
   (let [[bpe-lookup vocabulary-lookup] (huf-vocabulary-array vocabulary-stream)]
     (streaming-decoder bpe-lookup vocabulary-lookup capacity)))
  ([vocabulary-source]
   (streaming-decoder vocabulary-source 1024)))

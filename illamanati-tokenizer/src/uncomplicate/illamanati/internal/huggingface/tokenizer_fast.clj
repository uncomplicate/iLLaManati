;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.huggingface.tokenizer-fast
  (:require [clojure.java.io :refer [input-stream]]
            [uncomplicate.commons
             [core :refer [with-release Releaseable]]
             [utils :refer [path dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.clojure-cpp :refer [pointer long-pointer int-pointer limit memcpy! get!
                                              put! byte-pointer get-int put-int! get-short]]
            [uncomplicate.neanderthal
             [core :refer [dim transfer! view-vctr]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.neanderthal.internal
             [api :refer [navigator storage region]]
             [navigation :refer [full-storage]]]
            [uncomplicate.illamanati.internal.protocols :as api]
            [uncomplicate.illamanati.internal.huggingface.streaming-decoder :refer [streaming-decoder]])
  (:import java.io.InputStream
           [java.nio ByteBuffer CharBuffer BufferOverflowException]
           [java.nio.charset Charset StandardCharsets CodingErrorAction]
           [ai.djl.huggingface.tokenizers HuggingFaceTokenizer Encoding TokenizerConfig]
           [org.bytedeco.javacpp IntPointer LongPointer ShortPointer BytePointer]
           [uncomplicate.neanderthal.internal.api IntegerVector IntegerMatrix LayoutNavigator]))

;; ======= Array and primitive pointer converter functions =====================

(defn lp->lp
  (^longs [^LongPointer src ^LongPointer dst!]
   (when-not (identical? src dst!)
     (memcpy! src dst!))
   dst!)
  (^longs [^LongPointer src]
   (let [res (long-pointer (limit src))]
     (memcpy! src res)
     res)))

(defn lp->longs
  (^longs [^LongPointer src ^longs dst! ^long dst-offset]
   (get! src dst! dst-offset (min (limit src) (- (alength dst!) dst-offset)))
   dst!)
  (^longs [^LongPointer src ^longs dst!]
   (get! src dst!)
   dst!)
  (^longs [^LongPointer src]
   (get! src (long-array (limit src)))))

(defn longs->lp
  ([^longs src ^LongPointer dst!]
   (put! dst! src)
   dst!)
  ([^longs src]
   (put! (long-pointer (alength src)) src)))

(defn ints->longs
  (^longs [^ints src ^longs dst! ^long dst-offset]
   (let [len (min (alength src) (- (alength dst!) dst-offset))]
     (dotimes [i len]
       (aset ^longs dst! (+ i dst-offset) (aget ^ints src i)))
     dst!))
  (^longs [^ints src ^longs dst!]
   (ints->longs src dst! 0))
  (^longs [^ints src]
   (ints->longs (long-array (alength src)) 0)))

(defn longs->ints
  (^ints [^longs src ^ints dst!]
   (let [len (min (alength src) (limit dst!))]
     (dotimes [i len]
       (aset ^ints dst! i (aget ^longs src i)))
     dst!))
  (^ints [^longs src]
   (longs->ints (int-array (alength src)))))

(defn ip->longs
  (^longs [^IntPointer src ^longs dst! ^long dst-offset]
   (let [len (min (limit src) (- (alength dst!) dst-offset))
         temp (byte-pointer src)]
     (dotimes [i len]
       (aset ^longs dst! (+ i dst-offset) (long (get-int temp i))))
     dst!))
  (^longs [^IntPointer src ^longs dst!]
   (ip->longs src dst! 0))
  (^longs [^IntPointer src]
   (ip->longs src (long-array (limit src)) 0)))

(defn longs->ip
  ([^longs src ^IntPointer dst!]
   (let [len (min (alength src) (limit dst!))
         temp (byte-pointer dst!)]
     (dotimes [i len]
       (put-int! temp i (int (aget ^longs src i)))))
   dst!)
  ([^longs src]
   (longs->ip (int-pointer (alength src)))))

(defn shorts->longs
  (^longs [^shorts src ^longs dst! ^long dst-offset]
   (let [len (min (alength src) (- (alength dst!) dst-offset))]
     (dotimes [i len]
       (aset ^longs dst! (+ i dst-offset) (aget ^shorts src i)))
     dst!))
  (^longs [^shorts src ^longs dst!]
   (shorts->longs src dst! 0))
  (^longs [^shorts src]
   (shorts->longs (long-array (alength src)) 0)))

(defn longs->shorts
  (^shorts [^longs src ^shorts dst!]
   (let [len (alength src)]
     (dotimes [i len]
       (aset ^ints dst! i (aget ^shorts src i)))
     dst!))
  (^shorts [^longs src]
   (longs->shorts (int-array (alength src)))))

(defn sp->longs
  (^longs [^ShortPointer src ^longs dst! ^long dst-offset]
   (let [len (min (limit src) (- (alength dst!) dst-offset))
         temp (byte-pointer src)]
     (dotimes [i len]
       (aset ^longs dst! (+ i dst-offset) (long (get-short temp i))))
     dst!))
  (^longs [^ShortPointer src ^longs dst!]
   (sp->longs src dst! 0))
  (^longs [^ShortPointer src]
   (sp->longs src (long-array (limit src)) 0)))

;; ================== HUFT Protocols ===========================================

(defprotocol CoerceLongArray
  (to-longs [ids] [ids dst!] [ids dst! dst-offset])
  (from-longs [ids dst!]))

(defprotocol Encodable
  (encode [src encoder]))

(defprotocol Decodable
  (decode [src decoder]))

;; ================== Encodables ===============================================

(extend-type String
  Encodable
  (encode [text hft]
    (.encode hft text)))

(extend-type (Class/forName "[Ljava.lang.String;")
  Encodable
  (encode [text hft]
    (.batchEncode hft text)))

;; ================= Decodables ================================================

;; ----------------- 2D arrays -------------------------------------------------

(extend-type (Class/forName "[[J")
  Decodable
  (decode [srcs hft]
    (.batchDecode hft srcs)))

(extend-type (Class/forName "[[I")
  Decodable
  (decode [^"[[I" srcs hft]
    (let [len (alength srcs)
          dest (make-array Long/TYPE len 0)]
      (dotimes [i len]
        (aset dest i (ints->longs (aget srcs i))))
      (.batchDecode hft dest))))

(extend-type (Class/forName "[[S")
  Decodable
  (decode [^"[[S" srcs hft]
    (let [len (alength srcs)
          dest (make-array Long/TYPE len 0)]
      (dotimes [i len]
        (aset dest i (shorts->longs (aget srcs i))))
      (.batchDecode hft dest))))

;; -------------- 1D arrays ----------------------------------------------------

(extend-type (Class/forName "[J")
  CoerceLongArray
  (to-longs
    ([this]
     this)
    ([this ^longs dst!]
     (lp->lp this dst!))
    ([this ^longs dst! ^long dst-offset]
     (lp->lp this dst! dst-offset)))
  (from-longs [this! src]
    (lp->lp src this!))
  Decodable
  (decode [src hft]
    (.decode hft src)))

(extend-type (Class/forName "[I")
  CoerceLongArray
  (to-longs
    ([this]
     (ints->longs this))
    ([this ^longs dst!]
     (ints->longs this dst!))
    ([this ^longs dst! ^long dst-offset]
     (ints->longs this dst! dst-offset)))
  (from-longs [this! src]
    (longs->ints src this!))
  Decodable
  (decode [src hft]
    (.decode hft (ints->longs src))))

(extend-type (Class/forName "[S")
  CoerceLongArray
  (to-longs
    ([this]
     (shorts->longs this))
    ([this ^longs dst!]
     (shorts->longs this dst!))
    ([this ^longs dst! ^long dst-offset]
     (shorts->longs this dst! dst-offset)))
  (from-longs [this! src]
    (longs->shorts src this!))
  Decodable
  (decode [src hft]
    (.decode hft (shorts->longs src))))

(extend-type LongPointer
  CoerceLongArray
  (to-longs
    ([this]
     (lp->longs this))
    ([this ^longs dst!]
     (lp->longs this dst!))
    ([this ^longs dst! ^long dst-offset]
     (lp->longs this dst! dst-offset)))
  (from-longs [this! src]
    (longs->lp src this!))
  Decodable
  (decode [src hft]
    (.decode hft (lp->longs src))))

(extend-type IntPointer
  CoerceLongArray
  (to-longs
    ([this]
     (ip->longs this))
    ([this ^longs dst!]
     (ip->longs this dst!))
    ([this ^longs dst! ^long dst-offset]
     (ip->longs this dst! dst-offset)))
  (from-longs [this! src]
    (longs->ip src this!))
  Decodable
  (decode [src hft]
    (.decode hft (ip->longs src))))

(extend-type ShortPointer
  CoerceLongArray
  (to-longs
    ([this]
     (ip->longs this))
    ([this ^longs dst!]
     (ip->longs this dst!))
    ([this ^longs dst! ^long dst-offset]
     (ip->longs this dst! dst-offset)))
  (from-longs [this! src]
    (longs->ip src this!))
  Decodable
  (decode [src hft]
    (.decode hft (sp->longs src))))

(extend-type java.util.List
  Encodable
  (encode [text hft]
    (.batchEncode hft text))
  Decodable
  (decode [src hft]
    (.batchDecode hft src)))

(extend-type java.util.Collections
  Encodable
  (encode [text hft]
    (.batchEncode hft text)))

(extend-type clojure.lang.Sequential
  CoerceLongArray
  (to-longs
    ([this]
     (long-array this))
    ([this ^longs dst!]
     (lp->lp (long-array this) dst!)))
  (from-longs [this src]
    (into (empty this) src))
  Decodable
  (decode [src hft]
    (.decode hft (long-array src))))

;; ============= Vectors and matrices ==========================================

(extend-type IntegerVector
  CoerceLongArray
  (to-longs
    ([this]
     (to-longs this (long-array (dim this))))
    ([this ^longs dst!]
     (if (contiguous? this)
       (to-longs (buffer this) dst!)
       (transfer! this dst!)))
    ([this ^longs dst! ^longs dst-offset]
     (if (contiguous? this)
       (to-longs (buffer this) dst! dst-offset)
       (dragan-says-ex "Only contiguous vectors can be copied to arrays with offset."))))
  (from-longs [this! src]
    (if (contiguous? this!)
      (from-longs (buffer this!) src)
      (transfer! src this!))
    this!)
  Decodable
  (decode [data hft]
    (.decode hft (to-longs data))))

(extend-type IntegerMatrix
  CoerceLongArray
  (to-longs
    ([data]
     (let [stor (full-storage data)
           batch (.fd stor)
           n (.sd stor)
           res (make-array Long/TYPE batch n)]
       (to-longs data res)))
    ([data ^"[[J" dst!]
     (to-longs data dst! 0))
    ([data ^"[[J" dst! ^long dst-offset]
     (let [nav (navigator data)
           batch (alength dst!)]
       (dotimes [i batch]
         (to-longs (.stripe nav data i) (aget dst! i) dst-offset))
       dst!)))
  (from-longs [this! src]
    (let [stor (full-storage this!)
          nav (navigator this!)
          batch (.fd stor)
          n (.sd stor)]
      (if (contiguous? this!)
        (dotimes [i batch]
          (from-longs (buffer (.stripe nav this! i)) (get (vec src) i)))
        (dotimes [i batch]
          (transfer! (get (vec src) i) (.stripe nav this! i))))
      this!)
    (transfer! (seq src) this!))
  api/Encoding
  (ids [this! encodings]
    (let [stor (full-storage this!)
          nav (navigator this!)
          batch (.fd stor)
          n (.sd stor)]
      (if (contiguous? this!)
        (dotimes [i batch]
          (from-longs (buffer (.stripe nav this! i)) (.getIds ^Encoding (aget encodings i))))
        (dotimes [i batch]
          (transfer! (.getIds ^Encoding (aget encodings i)) (.stripe nav this! i))))
      this!))
  Decodable
  (decode [data hft]
    (.batchDecode hft (to-longs data))))

;; ============== HUF extensions ===============================================

(deftype HFT [^HuggingFaceTokenizer hft decoder-fn]
  java.lang.AutoCloseable
  (close [_]
    (.close hft))
  Releaseable
  (release [_]
    (.close hft))
  api/Encoder
  (encode [this text]
    (encode text hft))
  api/DecoderProvider
  (api/decoder [_]
    decoder-fn))

(defn hft [source]
  (cond
    (bytes? source)
    (->HFT (with-open [is0 (input-stream source)]
             (HuggingFaceTokenizer/newInstance is0 {"addSpecialTokens" "true"}))
           (with-open [is1 (input-stream source)]
             (streaming-decoder is1)))
    (instance? InputStream source)
    (with-open [stream source]
      (hft (.readAllBytes ^InputStream stream)))
    :default (hft (input-stream source))))

(extend-type Encoding
  api/Encoding
  (ids
    ([this]
     (.getIds this))
    ([this dst!]
     (from-longs dst! (.getIds this))))
  (tokens [this]
    (.getTokens this)))

(extend-type (Class/forName "[Lai.djl.huggingface.tokenizers.Encoding;")
  api/Encoding
  (ids
    ([this]
     (map #(.getIds ^Encoding %) this))
    ([this dst!]
     (api/ids dst! this)))
  (tokens [this]
    (map #(.getTokens ^Encoding %) this)))

(extend-type java.util.Collection
  api/Encoding
  (ids
    ([this]
     (fmap #(.getIds ^Encoding %) this))
    ([this dst!]
     (api/ids dst! this)))
  (tokens [this]
    (fmap #(.getTokens ^Encoding %) this)))

(extend-type TokenizerConfig
  api/Config
  (pad-token [this]
    (.getPadToken this)))

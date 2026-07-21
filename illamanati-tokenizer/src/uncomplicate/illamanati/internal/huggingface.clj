;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

 (ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.huggingface
  (:require [clojure.java.io :refer [input-stream copy]]
            [uncomplicate.commons
             [core :refer [with-release let-release Releaseable Info]]
             [utils :refer [path dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal
             [core :refer [dim transfer! view-vctr]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.neanderthal.internal
             [api :refer [navigator storage region]]
             [navigation :refer [full-storage]]]
            [uncomplicate.illamanati.internal
             [protocols :as api]
             [array-conversion :refer :all]
             [streaming-decoder :refer [streaming-decoder]]])
  (:import [java.io InputStream ByteArrayOutputStream]
           [java.nio ByteBuffer CharBuffer BufferOverflowException]
           [java.nio.charset Charset StandardCharsets CodingErrorAction]
           [clojure.lang IFn AFn]
           [ai.djl.huggingface.tokenizers HuggingFaceTokenizer Encoding]
           [org.bytedeco.javacpp IntPointer LongPointer ShortPointer BytePointer]
           [uncomplicate.neanderthal.internal.api IntegerVector IntegerMatrix LayoutNavigator]))

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
  (encode [text ^HuggingFaceTokenizer hft]
    (.encode hft text false false)))

(extend-type (Class/forName "[Ljava.lang.String;")
  Encodable
  (encode [text ^HuggingFaceTokenizer hft]
    (.batchEncode hft ^"[Ljava.lang.String;" text false false)))

;; ================= Decodables ================================================

;; ----------------- 2D arrays -------------------------------------------------

(extend-type long/2
  Decodable
  (decode [srcs hft]
    (.batchDecode ^HuggingFaceTokenizer hft ^long/2 srcs false)))

(extend-type int/2
  Decodable
  (decode [srcs hft]
    (let [long2 (make-array long/2 (alength ^int/2 srcs) (alength ^ints (aget ^int/2 srcs 0)))]
      (dotimes [i (alength ^int/2 srcs)]
        (ints->longs (aget ^int/2 srcs i) (aget ^long/2 long2 i)))
      (.batchDecode ^HuggingFaceTokenizer hft ^long/2 long2 false))))

(extend-type short/2
  Decodable
  (decode [srcs hft]
    (let [long2 (make-array long/2 (alength ^short/2 srcs) (alength ^shorts (aget ^short/2 srcs 0)))]
      (dotimes [i (alength ^short/2 srcs)]
        (shorts->longs (aget ^short/2 srcs i) (aget ^long/2 long2 i)))
      (.batchDecode ^HuggingFaceTokenizer hft ^long/2 long2 false))))

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
  (decode [src ^HuggingFaceTokenizer hft]
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
  (decode [src ^HuggingFaceTokenizer hft]
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
  (decode [src ^HuggingFaceTokenizer hft]
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
  (decode [src ^HuggingFaceTokenizer hft]
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
  (decode [src ^HuggingFaceTokenizer hft]
    (.decode hft (ip->longs src))))

(extend-type ShortPointer
  CoerceLongArray
  (to-longs
    ([this]
     (sp->longs this))
    ([this ^longs dst!]
     (sp->longs this dst!))
    ([this ^longs dst! ^long dst-offset]
     (sp->longs this dst! dst-offset)))
  (from-longs [this! src]
    (longs->ip src this!))
  Decodable
  (decode [src ^HuggingFaceTokenizer hft]
    (.decode hft (sp->longs src))))

(extend-type java.util.List
  Encodable
  (encode [text hft]
    (.batchEncode ^HuggingFaceTokenizer hft text false false))
  Decodable
  (decode [src hft]
    (.batchDecode ^HuggingFaceTokenizer hft src false)))

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
  (decode [src ^HuggingFaceTokenizer hft]
    (.decode hft (long-array src))))

;; ;; ============= Vectors and matrices ==========================================

;; (extend-type IntegerVector
;;   CoerceLongArray
;;   (to-longs
;;     ([this]
;;      (to-longs this (long-array (dim this))))
;;     ([this ^longs dst!]
;;      (if (contiguous? this)
;;        (to-longs (buffer this) dst!)
;;        (transfer! this dst!)))
;;     ([this ^longs dst! ^longs dst-offset]
;;      (if (contiguous? this)
;;        (to-longs (buffer this) dst! dst-offset)
;;        (dragan-says-ex "Only contiguous vectors can be copied to arrays with offset."))))
;;   (from-longs [this! src]
;;     (if (contiguous? this!)
;;       (from-longs (buffer this!) src)
;;       (transfer! src this!))
;;     this!)
;;   Decodable
;;   (decode [data ^HuggingFaceTokenizer hft]
;;     (.decode hft (to-longs data))))

;; (extend-type IntegerMatrix
;;   CoerceLongArray
;;   (to-longs
;;     ([data]
;;      (let [stor (full-storage data)
;;            batch (.fd stor)
;;            n (.sd stor)
;;            res (make-array Long/TYPE batch n)]
;;        (to-longs data res)))
;;     ([data ^"[[J" dst!]
;;      (to-longs data dst! 0))
;;     ([data ^"[[J" dst! ^long dst-offset]
;;      (let [nav (navigator data)
;;            batch (alength dst!)]
;;        (dotimes [i batch]
;;          (to-longs (.stripe nav data i) (aget dst! i) dst-offset))
;;        dst!)))
;;   (from-longs [this! src]
;;     (let [stor (full-storage this!)
;;           nav (navigator this!)
;;           batch (.fd stor)
;;           n (.sd stor)]
;;       (if (contiguous? this!)
;;         (dotimes [i batch]
;;           (from-longs (buffer (.stripe nav this! i)) (get (vec src) i)))
;;         (dotimes [i batch]
;;           (transfer! (get (vec src) i) (.stripe nav this! i))))
;;       this!)
;;     (transfer! (seq src) this!))
;;   api/EncodingIds
;;   (ids [this! encodings]
;;     (let [stor (full-storage this!)
;;           nav (navigator this!)
;;           batch (.fd stor)
;;           n (.sd stor)]
;;       (if (contiguous? this!)
;;         (dotimes [i batch]
;;           (from-longs (buffer (.stripe nav this! i))
;;                       (.getIds ^Encoding (aget ^"[Lai.djl.huggingface.tokenizers.Encoding;" encodings i))))
;;         (dotimes [i batch]
;;           (transfer! (.getIds ^Encoding (aget ^"[Lai.djl.huggingface.tokenizers.Encoding;" encodings i))
;;                      (.stripe nav this! i))))
;;       this!))
;;   Decodable
;;   (decode [data ^HuggingFaceTokenizer hft]
;;     (.batchDecode hft (to-longs data) false false)))

;; ============== HUF extensions ===============================================

(deftype HFT [^HuggingFaceTokenizer hft config decoder-fn]
  java.lang.AutoCloseable
  (close [_]
    (.close hft))
  Releaseable
  (release [_]
    (.close hft)
    true)
  Info
  (info [_]
    config)
  (info [_ info-key]
    (get config info-key))
  api/TokenizerProvider
  (tokenizer [this]
    this)
  IFn
  (invoke [_ text-or-token-ids]
    (if (or (string? text-or-token-ids) (string? (first text-or-token-ids)))
          (with-release [enc (encode text-or-token-ids hft)]
            (api/ids enc))
          (decode text-or-token-ids hft)))
  (invoke [_]
    decoder-fn)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  api/Encoder
  (encode [this text]
    (encode text hft)))

(defn build-config [^HuggingFaceTokenizer hft]
  (with-release [text "<pad><eos><bos><unk><mask>[multimodal]"
                 enc (.encode hft text false false)
                 ids ^longs (api/ids enc)]
    {:pad (aget ids 0)
     :eos (aget ids 1)
     :bos (aget ids 2)
     :unk (aget ids 3)
     :mask (aget ids 4)
     :multimodal (aget ids 5)}))

(defn hft [source]
  (cond
    (bytes? source)
    (let-release [hft (with-open [is0 (input-stream source)]
                        (HuggingFaceTokenizer/newInstance is0 {"addSpecialTokens" "true"}))]
      (->HFT hft
             (build-config hft)
             (with-open [is1 (input-stream source)]
               (streaming-decoder is1))))
    (instance? InputStream source)
    (with-open [^InputStream stream source]
      (let [baos (ByteArrayOutputStream.)]
        (copy stream baos)
        (hft (.toByteArray baos))))
    :default (hft (input-stream source))))

(extend-type Encoding
  api/EncodingIds
  (ids [this]
    (.getIds this))
  api/EncodingTokens
  (tokens [this]
    (.getTokens this)))

(extend-type (Class/forName "[Lai.djl.huggingface.tokenizers.Encoding;")
  api/EncodingIds
  (ids [this]
    (mapv api/ids this))
  api/EncodingTokens
  (tokens [this]
    (mapv api/tokens this)))

(extend-type java.util.Collection
  api/EncodingIds
  (ids [this]
    (fmap api/ids this))
  api/EncodingTokens
  (tokens [this]
    (fmap api/tokens this)))

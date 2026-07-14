;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.illamanati.internal.array-conversion
  (:require [uncomplicate.clojure-cpp
             :refer [pointer long-pointer int-pointer limit memcpy! get!
                     put! byte-pointer get-int put-int! get-short]])
  (:import [org.bytedeco.javacpp IntPointer LongPointer ShortPointer BytePointer]))

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

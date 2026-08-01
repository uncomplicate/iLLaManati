;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.cpu-flow
  (:require [clojure.string :refer [join]]
            [clojure.core.async :refer [thread <!!]]
            [clojure.core.async.flow :as flow :refer [create-flow start resume stop inject process]]
            [uncomplicate.diamond.native :refer []]
            [uncomplicate.illamanati :refer [encoder decoder generator-process]]
            [uncomplicate.illamanati.internal.onnxrt.gemma3 :refer [gemma-3-genai-cpu]]))

;; Before you expect this to work, download this model from Hugging Face: https://huggingface.co/onnxruntime/Gemma-3-ONNX/gemma-3-4b-it
;; That is a multi-GB repository, you only need:
;; - cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-text.onnx
;; - cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/gemma-3-embedding.onnx
;; Also, you have to download the tokenizer model for Gemma 3 from Google's Gemma 3 repository, for example from
;; https://huggingface.co/google/gemma-3-4b-it/blob/main/tokenizer.model
;; put that tokenizer file in the root of the model-path (in the same folder where cpu_and_mobile is)
;; the paths are relative to the folder where project.clj is, so make sure you have the /data/ folder,
;; or edit the path accordingly to the path you put these files in.

(def provider ((:provider gemma-3-genai-cpu)
               "../../../data/Gemma-3-ONNX/gemma-3-4b-it"
               (into gemma-3-genai-cpu {:context-len 4096})))

(def topology {:procs {:enc {:proc (process #'encoder)
                                         :args {:tokenizer provider}}
                                   :dec {:proc (process #'decoder)
                                         :args {:tokenizer provider}}
                                   :gen {:proc (generator-process provider)
                                         :executor :thread}
                                   :monitor {:proc (process (fn
                                                              ([] {:ins {:in :data}})
                                                              ([s] s)
                                                              ([s _] s)
                                                              ([s _ m] [s {::flow/report [m]}])))}}
                           :conns [[[:enc :out] [:gen :prompt]]
                                   [[:gen :token] [:dec :in]]
                                   [[:gen :next] [:gen :step]]
                                   [[:dec :out] [:monitor :in]]]})

(def f (create-flow topology))

;; This will take a few seconds
(def running-flow (start f))

(resume f)

(inject f [:enc :in] ["Clojure is"])

(thread (loop [s (<!! (:report-chan running-flow))]
          (if s
            (if (not= "<end_of_turn" s)
              (do (print s)
                  (recur (<!! (:report-chan running-flow))))
              (stop s))
            (stop f))))

;; My CPU from 2019 gives me stable 15 tokens per second with Gemma 3 4b. Not bad at all.

" a dynamic, general-purpose programming language that runs on the Java Virtual Machine (JVM), .NET Common Language Runtime (CLR), and JavaScript. It's a dialect of Lisp, known for its concurrency features, immutable data structures, and functional programming paradigm.

Here's a breakdown of key aspects of Clojure:

**1. Core Concepts:**

* **Lisp Dialect:**  Clojure inherits the Lisp philosophy of code as data, using S-expressions (parenthesized expressions) for defining programs.
* **Immutability:** Data structures in Clojure are immutable by default. This simplifies reasoning about code and makes concurrency easier.  Changes create new data structures instead of modifying existing ones.
* **Concurrency:** Clojure's core is built around concurrency. It provides powerful primitives for managing concurrent tasks, including:
    * **Agents:**  For asynchronous state management.
    * **Atoms:**  For atomic updates to shared state.
    * **Refs:**  For transactional state management.
    * **STM (Software Transactional Memory):**  A mechanism for managing concurrent state updates safely.
* **Functional Programming:** Clojure encourages functional programming principles like:
    * **Pure Functions:** Functions that have no side effects.
    * **Higher-Order Functions:** Functions that take other functions as arguments or return functions.
    * **Recursion:**  A common technique for solving problems.
* **Dynamic Typing:**  Clojure is dynamically typed, meaning you don't need to declare the types of variables.  The type is inferred at runtime.
* **Data-Oriented:** Clojure is designed to work with data.  It provides powerful data structures like vectors, maps, sets, and sequences.
* **Macros:**  Powerful macro system for code generation and abstraction.

**2. Key Features:**

* **JVM, CLR, and JavaScript Support:**  Clojure can run on any platform that supports these runtimes.
* **Rich Ecosystem:**  A growing ecosystem of libraries and tools.
* **REPL (Read-Eval-Print Loop):**  An interactive environment for experimenting with code and debugging.
* **Persistent Data Structures:**  Data structures are efficiently shared and reused.
* **Automatic Memory Management (Garbage Collection):**  Managed by the JVM/CLR/JavaScript runtime.
* **Protocol and Records:**  Powerful mechanisms for defining data structures and their behavior.
* **Pipelines:**  A way to chain functions together for data transformation.

**3. When to Use Clojure:**

* **Concurrent Systems:**  Clojure's concurrency features make it well-suited for building highly concurrent and scalable applications.
* **Data Processing:**  Its data-oriented nature and efficient data structures are beneficial for data manipulation and analysis.
* **Web Development
"
(stop f)

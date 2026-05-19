(defproject illamanati-tokenizer "0.1.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [com.cnuernber/charred "1.038"]
                 [ai.djl.huggingface/tokenizers "0.36.0"]
                 ;; The following line includes the Ahead-of-Time (AOT) compiled Deep Diamond, for fast start
                 ;; In production, you should prefer the specific Deep Diamond parts that you neeed,
                 ;; and then build them according to your preferences. The functionality is the same,
                 ;; AOT compilation just loads instantly, but requires exact versions of dependencies,
                 ;; which then might clash with the versions that you project includes.
                 ;; If you want to try the Hello World without AOT, just comment out the uncomplicate/deep-diamond
                 ;; dependency!
                 [uncomplicate/deep-diamond "0.44.0"]
                 [org.uncomplicate/neanderthal-base "0.62.0"]
                 [org.uncomplicate/deep-diamond-dnnl "0.44.0"]]

  ;; Most of the following dependencies can be left out if you already have compatible binaries
  ;; installed globally through your operating system's package manager.
  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]]
                       :dependencies [[midje "1.10.10"]]
                       :resource-paths ["data"]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                            "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.62.0"]
                                    [org.bytedeco/mkl "2025.3-1.5.13" :classifier "linux-x86_64-redist"]]}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.62.0"]
                                      [org.bytedeco/mkl "2025.3-1.5.13" :classifier "windows-x86_64-redist"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.62.0"]
                                     [org.bytedeco/openblas "0.3.31-1.5.13" :classifier "macosx-arm64"]]}}

  :resource-paths ["data"]
  ;; Wee need this for the DNNL binaries, for the latest version is not available in the Maven Central yet
  ;; :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )

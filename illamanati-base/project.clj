(defproject org.uncomplicate/illamanati-base "0.4.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [org.clojure/core.async "1.10.874-alpha3"]
                 [com.cnuernber/charred "1.038"]
                 [org.bytedeco/sentencepiece-platform "0.2.1-1.5.13"]
                 [ai.djl.huggingface/tokenizers "0.36.0"]
                 [org.uncomplicate/neanderthal-base "0.64.0"]
                 [org.uncomplicate/deep-diamond-base "0.46.0"]]

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
             :linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.64.0"]
                                    [org.bytedeco/mkl "2025.3-1.5.13" :classifier "linux-x86_64-redist"]]}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.64.0"]
                                      [org.bytedeco/mkl "2025.3-1.5.13" :classifier "windows-x86_64-redist"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.64.0"]
                                     [org.bytedeco/openblas "0.3.31-1.5.13" :classifier "macosx-arm64"]]}}

  :resource-paths ["data"]
  ;; Wee need this for the DNNL binaries, for the latest version is not available in the Maven Central yet
  ;; :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  :javac-options ["--release" "21" "-Xlint:-options"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )

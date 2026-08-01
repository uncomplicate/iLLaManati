(defproject org.uncomplicate/illamanati-onnxrt "0.7.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [org.uncomplicate/neanderthal-base "0.65.0"]
                 [org.uncomplicate/deep-diamond-base "0.46.1"]
                 [org.uncomplicate/deep-diamond-dnnl "0.46.1"]
                 [org.uncomplicate/diamond-onnxrt "0.29.0"]
                 [org.uncomplicate/snapdragan-base "0.8.0"]
                 [org.uncomplicate/illamanati-base "0.6.0"]]

  ;; Most of the following dependencies can be left out if you already have compatible binaries
  ;; installed globally through your operating system's package manager.
  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]]
                       :dependencies [[midje "1.10.10"]]
                       :resource-paths ["../data"]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=tru"
                                            "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.uncomplicate/illamanati-cuda "0.6.0"]
                                    #_[org.uncomplicate/snapdragan-cuda "0.8.0"]
                                    [org.bytedeco/onnxruntime-platform-gpu "1.28.0-1.5.14-20260730.122756-6"]
                                    [org.uncomplicate/neanderthal-mkl "0.65.0"]
                                    [org.bytedeco/mkl "2025.3-1.5.13" :classifier "linux-x86_64-redist"]
                                    [org.uncomplicate/deep-diamond-cuda "0.46.1"]
                                    [org.bytedeco/cuda-redist "13.2-9.21-1.5.14-20260703.164521-11" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "13.2-9.21-1.5.14-20260703.164546-11" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cudnn "13.2-9.21-1.5.14-20260703.164735-11" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-nccl "13.2-9.21-1.5.14-20260703.164522-11" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.uncomplicate/illamanati-cuda "0.6.0"]
                                      #_[org.uncomplicate/snapdragan-cuda "0.8.0"]
                                      [org.bytedeco/onnxruntime-platform-gpu "1.28.0-1.5.14-20260730.122756-6"]
                                      [org.uncomplicate/neanderthal-mkl "0.65.0"]
                                      [org.bytedeco/mkl "2025.3-1.5.13" :classifier "windows-x86_64-redist"]
                                      [org.uncomplicate/deep-diamond-cuda "0.46.1"]
                                      [org.bytedeco/cuda-redist "13.2-9.21-1.5.14-20260703.164521-11" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cublas "13.2-9.21-1.5.14-20260703.164546-11" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cudnn "13.2-9.21-1.5.14-20260703.164735-11" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-nccl "13.2-9.21-1.5.14-20260703.164522-11" :classifier "windows-x86_64"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.65.0"]
                                     [org.bytedeco/openblas "0.3.31-1.5.13" :classifier "macosx-arm64"]]}}

  ;; Wee need this for the DNNL binaries, for the latest version is not available in the Maven Central yet
  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  :javac-options ["--release" "21" "-Xlint:-options"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )

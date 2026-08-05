(defproject org.uncomplicate/illamanati-cuda "0.10.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [org.clojure/core.async "1.10.874-alpha3"]
                 [org.bytedeco/cuda-platform "13.3-9.25-1.5.14-SNAPSHOT"]
                 [org.uncomplicate/deep-diamond-base "0.46.1"]
                 [org.uncomplicate/deep-diamond-cuda "0.46.1"]
                 [org.uncomplicate/illamanati-base "0.9.0"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]]
                   :dependencies [[midje "1.10.10"]]
                   :resource-paths ["data"]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "--enable-native-access=ALL-UNNAMED"]} }

  :resource-paths ["data"]
  ;; Wee need this for the CUDA binaries, for the latest version is not available in the Maven Central yet
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

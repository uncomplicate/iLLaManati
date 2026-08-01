[New books available for subscription](https://aiprobook.com)

<img src="http://aiprobook.com/img/dlfp-cover.png" alt="Deep Learning for Programmers" title="Deep Learning for Programmers" align="left" width="250"/>

<img src="http://aiprobook.com/img/lafp-cover.png" alt="Numerical Linear Algebra for Programmers" title="Numerical Linear Algebra for Programmers" align="right" width="250"/>

# iLLaManati - The clojure CPU & GPU LLM

[Become a patron](https://patreon.com/draganrocks).

iLLaManati is a Clojure library for running Large Language Models on the CPU and GPU.
iLLaManati is powered by [Neanderthal](https://github.com/uncomplicate/neanderthal), [Deep Diamond](https://github.com/uncomplicate/deep-diamond),
and LLM runner backends. Currently, available backends are:
- [Diamond ONNX Runtime](https://github.com/uncomplicate/diamond-onnxrt)

## How to use it

As usual with Uncomplicate projects, there's a [Hello World project to get you started](https://github.com/uncomplicate/iLLaManati/tree/main/illamanati-onnxrt/examples/hello-world).

Until I set up a dedicated website with the documentation and tutorials, please refer to my
blog [Dragan Rocks](https://dragan.rocks), the [ONNX Runtime backend tests](https://github.com/uncomplicate/iLLaManati/blob/main/illamanati-onnxrt/test/uncomplicate/illamanati/internal/onnxrt/generator_test.clj), and [Hello World project](https://github.com/uncomplicate/iLLaManati/tree/main/illamanati-onnxrt/examples/hello-world).

There is not only enough step-by-step material there to get you started, but to deep-dive into both higher-level and lower-level features.

Please note that this library does not even introduce much public API, beyond the core.async and Flow functions, since it does everything automatically under the hood, and the implementation neatly blends into [Neanderthal](https://github.com/uncomplicate/neanderthal) and [Deep Diamond](https://github.com/uncomplicate/deep-diamond), which is very extensively documented in the [Deep Learning for Programmers](https://aiprobook.com) book.

Please also note that the current release is a preview, and not yet ready for production use.

## License

Copyright © 2025-2026 Dragan Djuric

Distributed under the Eclipse Public License either version 1.0 or (at your option) any later version.

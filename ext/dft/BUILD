# Build rules for TPU distributed discrete Fourier transform (DFT).
load("//devtools/python/blaze:pytype.bzl", "pytype_library")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

# TODO(b/133421409.): Convert to pytype_strict_library.
pytype_library(
    name = "dft",
    srcs = ["dft.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

# TODO(b/133421409): Convert to py2and3_strict_test.
py_test(
    name = "dft_test",
    srcs = ["dft_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":dft",
        ":dft_initializer",
        "//learning/brain/google/xla:deepsea_default_hardware_device",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

pytype_library(
    name = "dft_initializer",
    srcs = ["dft_initializer.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:util",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "dft_initializer_test",
    srcs = ["dft_initializer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dft_initializer",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

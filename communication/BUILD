# Contains functions for communicating halo information across cores. These
# functions also apply boundary conditions.

load(
    "//devtools/python/blaze:pytype.bzl",
    "pytype_library",
    "pytype_strict_library",
)
load(
    "//devtools/python/blaze:strict.bzl",
    "py_strict_test",
)

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

# TODO(b/133421409): Convert to pytype_strict_library.
pytype_library(
    name = "halo_exchange",
    srcs = [
        "halo_exchange.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":halo_exchange_utils",
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_strict_test(
    name = "halo_exchange_test",
    timeout = "long",
    srcs = ["halo_exchange_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":halo_exchange",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "halo_exchange_utils",
    srcs = [
        "halo_exchange_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:util",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "halo_exchange_utils_test",
    srcs = [
        "halo_exchange_utils_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":halo_exchange_utils",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

# Contains basic logistics to set up and run a simulation. Modules such as the
# TPU driver, I/O, the simulation configuration context, and the key manager should
# go in this directory.

load(
    "//devtools/python/blaze:pytype.bzl",
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

pytype_strict_library(
    name = "driver_tpu",
    srcs = ["driver_tpu.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "driver_tpu_test",
    srcs = ["driver_tpu_test.py"],
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish",
    ],
    deps = [
        ":driver_tpu",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

pytype_strict_library(
    name = "precipitation",
    srcs = ["precipitation.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/swirl_lm/physics/thermodynamics:water",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "precipitation_test",
    srcs = ["precipitation_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-dragonfish",
    ],
    deps = [
        ":precipitation",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/physics/thermodynamics:water",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

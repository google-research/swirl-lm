load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

py_strict_test(
    name = "thermodynamics_test",
    srcs = ["thermodynamics_test.py"],
    data = [
        "//third_party/py/swirl_lm/utility/post_processing/test_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":thermodynamics",
        "//pyglib:resources",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "thermodynamics",
    srcs = ["thermodynamics.py"],
    srcs_version = "PY3",
    deps = [
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/physics/thermodynamics:water",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

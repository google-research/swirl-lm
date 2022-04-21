load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

proto_library(
    name = "numerics_proto",
    srcs = ["numerics.proto"],
)

py_proto_library(
    name = "numerics_proto_py_pb2",
    api_version = 2,
    deps = [":numerics_proto"],
)

pytype_strict_library(
    name = "algebra",
    srcs = ["algebra.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "convection_test",
    srcs = ["convection_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":convection",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:boundary_condition_utils",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "diffusion_test",
    srcs = ["diffusion_test.py"],
    deps = [
        ":diffusion",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/numerics:numerics_proto_py_pb2",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":filters",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "root_finder_test",
    srcs = ["root_finder_test.py"],
    data = ["//third_party/py/swirl_lm/numerics:testdata/config.textpb"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":root_finder",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/physics/thermodynamics:water",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "convection",
    srcs = ["convection.py"],
    srcs_version = "PY3",
    visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:__pkg__",
        "//third_party/py/swirl_lm:__subpackages__",
        "//third_party/py/swirl_lm/numerics/tf1:__pkg__",
    ],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:boundary_condition_utils",
        "//third_party/py/swirl_lm/equations:common",
        "//third_party/py/swirl_lm/numerics:numerics_proto_py_pb2",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "diffusion",
    srcs = ["diffusion.py"],
    visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
        "//third_party/py/swirl_lm/numerics/tf1:__pkg__",
    ],
    deps = [
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:monin_obukhov_similarity_theory",
        "//third_party/py/swirl_lm/equations:common",
        "//third_party/py/swirl_lm/equations:utils",
        "//third_party/py/swirl_lm/numerics:numerics_proto_py_pb2",
        "//third_party/py/swirl_lm/physics:constants",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "filters",
    srcs = ["filters.py"],
    srcs_version = "PY3",
    visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
        "//third_party/py/swirl_lm/numerics/tf1:__pkg__",
    ],
    deps = [
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "root_finder",
    srcs = ["root_finder.py"],
    srcs_version = "PY3",
    visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
        "//third_party/py/swirl_lm/numerics/tf1:__pkg__",
    ],
    deps = [
        ":algebra",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/tensorflow",
    ],
)

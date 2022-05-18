load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//third_party/py/swirl_lm/utility:build_defs.bzl", "tf1_and_tf2_test")

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

pytype_strict_library(
    name = "calculus",
    srcs = ["calculus.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
    name = "calculus_test",
    srcs = ["calculus_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":calculus",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
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
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:boundary_condition_utils",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/swirl_lm/utility:tf_tpu_test_util",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
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

tf1_and_tf2_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-dragonfish",
    ],
    deps = [
        ":filters",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/swirl_lm/utility:tf_tpu_test_util",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
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

pytype_strict_library(
    name = "time_integration",
    srcs = ["time_integration.py"],
    srcs_version = "PY3",
    deps = [
        ":numerics_proto_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
    name = "time_integration_test",
    srcs = ["time_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":time_integration",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

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
    name = "pressure_proto",
    srcs = ["pressure.proto"],
    deps = ["//third_party/py/swirl_lm/linalg:poisson_solver_proto"],
)

py_proto_library(
    name = "pressure_py_pb2",
    api_version = 2,
    deps = [":pressure_proto"],
)

proto_library(
    name = "scalars_proto",
    srcs = ["scalars.proto"],
    deps = [
        "//third_party/py/swirl_lm/numerics:numerics_proto",
    ],
)

pytype_strict_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
)

py_strict_test(
    name = "pressure_test",
    timeout = "long",
    srcs = ["pressure_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    data = [
        "//third_party/py/swirl_lm/equations:testdata/pressure_config.textpb",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-dragonfish",
    ],
    deps = [
        ":pressure",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:monitor",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "scalars_test",
    srcs = ["scalars_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-dragonfish",
    ],
    deps = [
        ":scalars",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:components_debug",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "velocity_test",
    timeout = "long",
    srcs = ["velocity_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    data = glob([
        "testdata/velocity_config*.textpb",
    ]),
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-dragonfish",
    ],
    deps = [
        ":velocity",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/utility:components_debug",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:monitor",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "pressure",
    srcs = ["pressure.py"],
    srcs_version = "PY3",
    deps = [
        ":pressure_py_pb2",
        ":utils",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:physical_variable_keys_manager",
        "//third_party/py/absl/logging",
        "//third_party/py/attr",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:boundary_condition_utils",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/linalg:poisson_solver",
        "//third_party/py/swirl_lm/numerics:filters",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_py_pb2",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:monitor",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "scalars",
    srcs = ["scalars.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":utils",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_numerics",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:physical_variable_keys_manager",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:sgs_model",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:immersed_boundary_method",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/numerics:convection",
        "//third_party/py/swirl_lm/numerics:diffusion",
        "//third_party/py/swirl_lm/numerics:filters",
        "//third_party/py/swirl_lm/numerics:numerics_proto_py_pb2",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/physics/thermodynamics:water",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:components_debug",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_numerics",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:monin_obukhov_similarity_theory",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "velocity",
    srcs = ["velocity.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":utils",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_numerics",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:physical_variable_keys_manager",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:sgs_model",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:boundary_condition_utils",
        "//third_party/py/swirl_lm/boundary_condition:immersed_boundary_method",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/numerics:convection",
        "//third_party/py/swirl_lm/numerics:diffusion",
        "//third_party/py/swirl_lm/numerics:filters",
        "//third_party/py/swirl_lm/numerics:numerics_proto_py_pb2",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/physics/thermodynamics:water",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:components_debug",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:monitor",
        "//third_party/py/tensorflow",
    ],
)

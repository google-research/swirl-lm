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
load("//third_party/py/swirl_lm/utility:build_defs.bzl", "tf1_and_tf2_test")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "driver",
    srcs = ["driver.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:checkpoint_handler",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/framework/tf1:parallel_io",
        "//research/simulation/tensorflow/fluid/framework/tf1:tpu_simulation_manager",
        # BEGIN GOOGLE-INTERNAL
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:legacy_solvers",
        # END GOOGLE-INTERNAL
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/base:driver_tpu",
        "//third_party/py/swirl_lm/base:parameters",
        "//third_party/py/swirl_lm/core:simulation",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

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

pytype_strict_library(
    name = "parameters",
    srcs = ["parameters.py"],
    srcs_version = "PY3",
    deps = [
        ":parameters_py_pb2",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:boundary_condition_utils",
        "//third_party/py/swirl_lm/boundary_condition:boundary_conditions_py_pb2",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/numerics:numerics_proto_py_pb2",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:grid_parametrization_py_pb2",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

proto_library(
    name = "parameters_proto",
    srcs = ["parameters.proto"],
    cc_api_version = 2,
    deps = [
        "//third_party/py/swirl_lm/boundary_condition:boundary_conditions_proto",
        "//third_party/py/swirl_lm/boundary_condition:boundary_models_proto",
        "//third_party/py/swirl_lm/boundary_condition:rayleigh_damping_layer_proto",
        "//third_party/py/swirl_lm/equations:pressure_proto",
        "//third_party/py/swirl_lm/equations:scalars_proto",
        "//third_party/py/swirl_lm/numerics:numerics_proto",
        "//third_party/py/swirl_lm/physics/combustion:combustion_proto",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_proto",
        "//third_party/py/swirl_lm/utility:grid_parametrization_proto",
        "//third_party/py/swirl_lm/utility:monitor_proto",
        "//third_party/py/swirl_lm/utility:probe_proto",
    ],
)

py_proto_library(
    name = "parameters_py_pb2",
    api_version = 2,
    deps = [
        ":parameters_proto",
    ],
)

tf1_and_tf2_test(
    name = "parameters_test",
    srcs = ["parameters_test.py"],
    data = glob(["testdata/swirl_lm_parameters*.textpb"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":parameters",
        "//pyglib:resources",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "physical_variable_keys_manager",
    srcs = ["physical_variable_keys_manager.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/six",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
    name = "physical_variable_keys_manager_test",
    srcs = ["physical_variable_keys_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":physical_variable_keys_manager",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

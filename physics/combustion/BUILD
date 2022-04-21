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
    name = "combustion_proto",
    srcs = ["combustion.proto"],
    deps = [
        ":wood_proto",
    ],
)

py_proto_library(
    name = "combustion_py_pb2",
    api_version = 2,
    deps = [":combustion_proto"],
)

proto_library(
    name = "wood_proto",
    srcs = ["wood.proto"],
    deps = [
        "//third_party/py/swirl_lm/numerics:numerics_proto",
    ],
)

py_strict_test(
    name = "igniter_test",
    srcs = ["igniter_test.py"],
    deps = [
        ":igniter",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "onestep_test",
    srcs = ["onestep_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":onestep",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "turbulent_kinetic_energy_test",
    srcs = ["turbulent_kinetic_energy_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":turbulent_kinetic_energy",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "wood_test",
    srcs = ["wood_test.py"],
    data = [
        "//third_party/py/swirl_lm/physics/combustion:test_data/dry_wood.textpb",
        "//third_party/py/swirl_lm/physics/combustion:test_data/moist_wood.textpb",
    ],
    python_version = "PY3",
    shard_count = 7,
    srcs_version = "PY3",
    deps = [
        ":wood",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "igniter",
    srcs = ["igniter.py"],
    deps = [
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "onestep",
    srcs = ["onestep.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework/tf1:model_function",
        "//research/simulation/tensorflow/fluid/framework/tf1:step_updater",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_numerics",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_updates",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "turbulent_kinetic_energy",
    srcs = ["turbulent_kinetic_energy.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework/tf1:model_function",
        "//research/simulation/tensorflow/fluid/framework/tf1:step_updater",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_numerics",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "wood",
    srcs = ["wood.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework/tf1:model_function",
        "//research/simulation/tensorflow/fluid/framework/tf1:step_updater",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_numerics",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

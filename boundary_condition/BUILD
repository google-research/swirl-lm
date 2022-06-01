load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

proto_library(
    name = "boundary_conditions_proto",
    srcs = ["boundary_conditions.proto"],
    deps = [
        "//third_party/py/swirl_lm/utility:grid_parametrization_proto",
    ],
)

py_proto_library(
    name = "boundary_conditions_py_pb2",
    api_version = 2,
    deps = [":boundary_conditions_proto"],
)

proto_library(
    name = "boundary_models_proto",
    srcs = ["boundary_models.proto"],
    deps = [
        ":immersed_boundary_method_proto",
        ":monin_obukhov_similarity_theory_proto",
        ":rayleigh_damping_layer_proto",
    ],
)

proto_library(
    name = "immersed_boundary_method_proto",
    srcs = ["immersed_boundary_method.proto"],
)

py_proto_library(
    name = "immersed_boundary_method_py_pb2",
    api_version = 2,
    deps = [":immersed_boundary_method_proto"],
)

proto_library(
    name = "monin_obukhov_similarity_theory_proto",
    srcs = ["monin_obukhov_similarity_theory.proto"],
)

py_proto_library(
    name = "monin_obukhov_similarity_theory_py_pb2",
    api_version = 2,
    deps = [":monin_obukhov_similarity_theory_proto"],
)

proto_library(
    name = "rayleigh_damping_layer_proto",
    srcs = ["rayleigh_damping_layer.proto"],
)

py_proto_library(
    name = "rayleigh_damping_layer_py_pb2",
    api_version = 2,
    deps = [":rayleigh_damping_layer_proto"],
)

py_strict_test(
    name = "boundary_condition_utils_test",
    srcs = ["boundary_condition_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":boundary_condition_utils",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "immersed_boundary_method_test",
    srcs = ["immersed_boundary_method_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":immersed_boundary_method",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "monin_obukhov_similarity_theory_test",
    srcs = ["monin_obukhov_similarity_theory_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    data = [
        "test_data/monin_obukhov_similarity_theory_config.textpb",
        "test_data/monin_obukhov_similarity_theory_config_negative_heat_flux.textpb",
        "test_data/monin_obukhov_similarity_theory_config_reg.textpb",
        "test_data/monin_obukhov_similarity_theory_config_zero_heat_flux.textpb",
    ],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":monin_obukhov_similarity_theory",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:physical_variable_keys_manager",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/equations:common",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "outflow_test",
    srcs = ["outflow_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":outflow",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "rayleigh_damping_layer_test",
    srcs = ["rayleigh_damping_layer_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":rayleigh_damping_layer",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:rayleigh_damping_layer_py_pb2",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "synthetic_turbulent_inflow_test",
    timeout = "long",
    srcs = ["synthetic_turbulent_inflow_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    data = [
        "test_data/inflow_0_replicas112.npy",
        "test_data/inflow_0_replicas121.npy",
        "test_data/inflow_0_replicas211.npy",
        "test_data/inflow_1_replicas112.npy",
        "test_data/inflow_1_replicas121.npy",
        "test_data/inflow_1_replicas211.npy",
        "test_data/inflow_2_replicas112.npy",
        "test_data/inflow_2_replicas121.npy",
        "test_data/inflow_2_replicas211.npy",
    ],
    python_version = "PY3",
    shard_count = 14,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":synthetic_turbulent_inflow",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "boundary_condition_utils",
    srcs = ["boundary_condition_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/util:constants",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:types",
    ],
)

pytype_strict_library(
    name = "immersed_boundary_method",
    srcs = ["immersed_boundary_method.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:immersed_boundary_method_py_pb2",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "monin_obukhov_similarity_theory",
    srcs = ["monin_obukhov_similarity_theory.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:physical_variable_keys_manager",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/boundary_condition:monin_obukhov_similarity_theory_py_pb2",
        "//third_party/py/swirl_lm/equations:common",
        "//third_party/py/swirl_lm/numerics:root_finder",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "outflow",
    srcs = ["outflow.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:composite_types",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "rayleigh_damping_layer",
    srcs = ["rayleigh_damping_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":rayleigh_damping_layer_py_pb2",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "synthetic_turbulent_inflow",
    srcs = ["synthetic_turbulent_inflow.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:physical_variable_keys_manager",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

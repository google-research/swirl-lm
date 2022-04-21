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
    name = "thermodynamics_proto",
    srcs = ["thermodynamics.proto"],
)

py_proto_library(
    name = "thermodynamics_py_pb2",
    api_version = 2,
    deps = [":thermodynamics_proto"],
)

py_strict_test(
    name = "constant_density_test",
    srcs = ["constant_density_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":constant_density",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "ideal_gas_test",
    srcs = ["ideal_gas_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ideal_gas",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "linear_mixing_test",
    srcs = ["linear_mixing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":linear_mixing",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "thermodynamics_generic_test",
    srcs = ["thermodynamics_generic_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_generic",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "thermodynamics_manager_test",
    srcs = ["thermodynamics_manager_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    data = [
        "//third_party/py/swirl_lm/physics/thermodynamics:testdata/ideal_gas_anelastic.textpb",
        "//third_party/py/swirl_lm/physics/thermodynamics:testdata/ideal_gas_low_mach.textpb",
        "//third_party/py/swirl_lm/physics/thermodynamics:testdata/water_geo_static_anelastic.textpb",
        "//third_party/py/swirl_lm/physics/thermodynamics:testdata/water_geo_static_low_mach.textpb",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":thermodynamics_manager",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_py_pb2",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "thermodynamics_utils_test",
    srcs = ["thermodynamics_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_utils",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "water_test",
    srcs = ["water_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":water",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "constant_density",
    srcs = ["constant_density.py"],
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_generic",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "ideal_gas",
    srcs = ["ideal_gas.py"],
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_generic",
        ":thermodynamics_utils",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "linear_mixing",
    srcs = ["linear_mixing.py"],
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_generic",
        ":thermodynamics_utils",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "thermodynamics_generic",
    srcs = ["thermodynamics_generic.py"],
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_utils",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "thermodynamics_manager",
    srcs = ["thermodynamics_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":constant_density",
        ":ideal_gas",
        ":linear_mixing",
        ":thermodynamics_generic",
        ":thermodynamics_utils",
        ":water",
        "//research/simulation/tensorflow/fluid/framework/tf1:model_function",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/numerics:filters",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_py_pb2",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "thermodynamics_utils",
    srcs = ["thermodynamics_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/swirl_lm/physics:constants",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "water",
    srcs = ["water.py"],
    srcs_version = "PY3",
    deps = [
        ":thermodynamics_generic",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/swirl_lm/numerics:root_finder",
        "//third_party/py/swirl_lm/physics:constants",
        "//third_party/py/tensorflow",
    ],
)

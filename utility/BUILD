load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

proto_library(
    name = "monitor_proto",
    srcs = ["monitor.proto"],
)

py_proto_library(
    name = "monitor_py_pb2",
    api_version = 2,
    deps = [":monitor_proto"],
)

proto_library(
    name = "probe_proto",
    srcs = ["probe.proto"],
    deps = ["//third_party/py/swirl_lm/utility:grid_parametrization_proto"],
)

py_proto_library(
    name = "probe_py_pb2",
    api_version = 2,
    deps = [":probe_proto"],
)

pytype_strict_library(
    name = "utility",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    # everything except test_util
    deps = [
        ":common_ops",
        ":grid_parametrization",
        ":grid_parametrization_py_pb2",
        ":tf_test_util",
        ":types",
    ],
)

pytype_strict_library(
    name = "common_ops",
    srcs = ["common_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":types",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/framework/tf1:fluid",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "components_debug",
    srcs = ["components_debug.py"],
    srcs_version = "PY3",
    deps = [
        ":common_ops",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "get_kernel_fn",
    srcs = ["get_kernel_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":common_ops",
        ":types",
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "grid_parametrization",
    srcs = ["grid_parametrization.py"],
    deps = [
        ":grid_parametrization_py_pb2",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "init_fn",
    srcs = ["init_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":get_kernel_fn",
        ":types",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "log_invocation",
    srcs = ["log_invocation.py"],
    srcs_version = "PY3",
    deps = [
        "//file/colossus/public:cns",
        "//file/namespace",
        "//pyglib:build_data",
        "//pyglib:gfile",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
    ],
)

pytype_strict_library(
    name = "monitor",
    srcs = ["monitor.py"],
    srcs_version = "PY3",
    deps = [
        ":monitor_py_pb2",
        ":types",
        "//research/simulation/tensorflow/fluid/framework:analytics_util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "probe",
    srcs = ["probe.py"],
    deps = [
        ":common_ops",
        ":get_kernel_fn",
        ":grid_parametrization",
        ":types",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework/post_processing:data_processing",
        "//research/simulation/tensorflow/fluid/framework/tf1:model_function",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "tf_test_util",
    srcs = ["tf_test_util.py"],
    deps = ["//third_party/py/tensorflow"],
)

pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "tf_test_util_test",
    srcs = ["tf_test_util_test.py"],
    deps = [
        ":tf_test_util",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "tf_test_util_tf1_test",
    srcs = ["tf_test_util_tf1_test.py"],
    deps = [
        ":tf_test_util",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "common_ops_test",
    srcs = ["common_ops_test.py"],
    python_version = "PY3",
    shard_count = 30,
    srcs_version = "PY3",
    tags = ["requires-dragonfish:4"],
    deps = [
        ":common_ops",
        ":grid_parametrization",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "components_debug_test",
    srcs = ["components_debug_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":components_debug",
        ":tf_test_util",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "get_kernel_fn_test",
    srcs = ["get_kernel_fn_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    data = [
        "testdata/big_tile.txt",
        "testdata/big_tile_convop_k4d2x.txt",
        "testdata/big_tile_convop_k4d2y.txt",
        "testdata/big_tile_convop_kD4x.txt",
        "testdata/big_tile_convop_kD4y.txt",
        "testdata/big_tile_convop_kdd16x.txt",
        "testdata/big_tile_convop_kdd16y.txt",
        "testdata/big_tile_convop_kdd16z_0.txt",
        "testdata/big_tile_convop_kdd16z_1.txt",
        "testdata/big_tile_convop_kdd16z_10.txt",
        "testdata/big_tile_convop_kdd16z_11.txt",
        "testdata/big_tile_convop_kdd16z_12.txt",
        "testdata/big_tile_convop_kdd16z_13.txt",
        "testdata/big_tile_convop_kdd16z_14.txt",
        "testdata/big_tile_convop_kdd16z_15.txt",
        "testdata/big_tile_convop_kdd16z_16.txt",
        "testdata/big_tile_convop_kdd16z_17.txt",
        "testdata/big_tile_convop_kdd16z_2.txt",
        "testdata/big_tile_convop_kdd16z_3.txt",
        "testdata/big_tile_convop_kdd16z_4.txt",
        "testdata/big_tile_convop_kdd16z_5.txt",
        "testdata/big_tile_convop_kdd16z_6.txt",
        "testdata/big_tile_convop_kdd16z_7.txt",
        "testdata/big_tile_convop_kdd16z_8.txt",
        "testdata/big_tile_convop_kdd16z_9.txt",
        "testdata/big_tile_convop_kdd8x.txt",
        "testdata/big_tile_convop_kdd8y.txt",
        "testdata/big_tile_convop_kdd8z_0.txt",
        "testdata/big_tile_convop_kdd8z_1.txt",
        "testdata/big_tile_convop_kdd8z_10.txt",
        "testdata/big_tile_convop_kdd8z_11.txt",
        "testdata/big_tile_convop_kdd8z_12.txt",
        "testdata/big_tile_convop_kdd8z_13.txt",
        "testdata/big_tile_convop_kdd8z_14.txt",
        "testdata/big_tile_convop_kdd8z_15.txt",
        "testdata/big_tile_convop_kdd8z_16.txt",
        "testdata/big_tile_convop_kdd8z_17.txt",
        "testdata/big_tile_convop_kdd8z_2.txt",
        "testdata/big_tile_convop_kdd8z_3.txt",
        "testdata/big_tile_convop_kdd8z_4.txt",
        "testdata/big_tile_convop_kdd8z_5.txt",
        "testdata/big_tile_convop_kdd8z_6.txt",
        "testdata/big_tile_convop_kdd8z_7.txt",
        "testdata/big_tile_convop_kdd8z_8.txt",
        "testdata/big_tile_convop_kdd8z_9.txt",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":get_kernel_fn",
        ":tf_test_util",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//pyglib:gfile",
        "//pyglib:resources",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "grid_parametrization_test",
    srcs = ["grid_parametrization_test.py"],
    deps = [
        ":grid_parametrization",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "init_fn_test",
    srcs = ["init_fn_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":init_fn",
        ":tf_test_util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "log_invocation_test",
    srcs = ["log_invocation_test.py"],
    python_version = "PY3",
    deps = [
        ":log_invocation",
        "//pyglib:gfile",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/logging",
    ],
)

py_strict_test(
    name = "monitor_test",
    srcs = ["monitor_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = ["requires-dragonfish:4"],
    deps = [
        ":monitor",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/framework:test_util",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:monitor_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "probe_test",
    srcs = ["probe_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":get_kernel_fn",
        ":probe",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_config",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

proto_library(
    name = "grid_parametrization_proto",
    srcs = ["grid_parametrization.proto"],
)

py_proto_library(
    name = "grid_parametrization_py_pb2",
    api_version = 2,
    deps = [":grid_parametrization_proto"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    parse_tests = False,
    deps = ["//devtools/python/blaze:strict_bzl"],
)

load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

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

proto_library(
    name = "grid_parametrization_proto",
    srcs = ["grid_parametrization.proto"],
)

py_proto_library(
    name = "grid_parametrization_py_pb2",
    api_version = 2,
    deps = [":grid_parametrization_proto"],
)

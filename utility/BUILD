load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = ["//research/simulation/tensorflow/fluid:sim_research_fluid"],
)

licenses(["notice"])

pytype_strict_library(
    name = "utility",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    # everything except test_util
    deps = [
        ":grid_parametrization",
        ":grid_parametrization_py_pb2",
        ":tf_test_util",
        ":types",
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
        "//third_party/py/tensorflow:tensorflow_google",
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

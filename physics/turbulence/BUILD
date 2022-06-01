load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//third_party/py/swirl_lm/utility:build_defs.bzl", "tf1_and_tf2_test")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "sgs_model",
    srcs = ["sgs_model.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/numerics:calculus",
        "//third_party/py/swirl_lm/numerics:filters",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

tf1_and_tf2_test(
    name = "sgs_model_test",
    srcs = ["sgs_model_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":sgs_model",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/models/incompressible_structured_mesh:incompressible_structured_mesh_parameters_py_pb2",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:tf_test_util",
        "//third_party/py/swirl_lm/utility:tf_tpu_test_util",
        "//third_party/py/tensorflow",
    ],
)

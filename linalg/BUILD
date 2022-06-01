load("//devtools/python/blaze:pytype.bzl", "pytype_library", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "base_poisson_solver",
    srcs = ["base_poisson_solver.py"],
    srcs_version = "PY3",
    deps = [
        ":poisson_solver_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/six",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_library(
    name = "fast_diagonalization_solver",
    srcs = ["fast_diagonalization_solver.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/ext/dft",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "conjugate_gradient_solver",
    srcs = ["conjugate_gradient_solver.py"],
    srcs_version = "PY3",
    deps = [
        ":base_poisson_solver",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "conjugate_gradient_solver_test",
    timeout = "long",
    srcs = ["conjugate_gradient_solver_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":conjugate_gradient_solver",
        "//learning/brain/google/xla:deepsea_default_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/framework/tf1:fluid",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "fast_diagonalization_solver_test",
    timeout = "long",
    srcs = ["fast_diagonalization_solver_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":fast_diagonalization_solver",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/framework/tf1:fluid",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_1_core_compare_test",
    timeout = "long",
    srcs = ["multigrid_1_core_compare_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    shard_count = 12,
    srcs_version = "PY3",
    tags = ["requires-dragonfish:4"],
    deps = [
        ":multigrid",
        ":multigrid_3d",
        ":multigrid_utils",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_3d_test",
    srcs = ["multigrid_3d_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    shard_count = 30,
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":multigrid_3d",
        ":multigrid_test_common",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_3d_utils_test",
    timeout = "long",
    srcs = ["multigrid_3d_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish",
    ],
    deps = [
        ":multigrid_3d_utils",
        ":multigrid_test_common",
        ":multigrid_utils",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_multicore_3d_test",
    timeout = "long",
    srcs = ["multigrid_multicore_3d_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = ["requires-dragonfish:4"],
    deps = [
        ":multigrid_3d",
        ":multigrid_3d_utils",
        ":multigrid_utils",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_multicore_test",
    timeout = "long",
    srcs = ["multigrid_multicore_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    shard_count = 31,
    srcs_version = "PY3",
    tags = ["requires-dragonfish:4"],
    deps = [
        ":multigrid",
        ":multigrid_utils",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_test",
    srcs = ["multigrid_test.py"],
    args = ["--xla_jf_conv_full_precision=true"],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":multigrid",
        ":multigrid_test_common",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "multigrid_utils_test",
    timeout = "moderate",
    srcs = ["multigrid_utils_test.py"],
    python_version = "PY3",
    shard_count = 30,
    srcs_version = "PY3",
    tags = ["requires-dragonfish"],
    deps = [
        ":multigrid_test_common",
        ":multigrid_utils",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

proto_library(
    name = "poisson_solver_proto",
    srcs = ["poisson_solver.proto"],
    cc_api_version = 2,
    deps = ["//third_party/py/swirl_lm/utility:grid_parametrization_proto"],
)

py_proto_library(
    name = "poisson_solver_py_pb2",
    api_version = 2,
    deps = [":poisson_solver_proto"],
)

py_strict_test(
    name = "poisson_solver_cg_test",
    timeout = "long",
    srcs = ["poisson_solver_cg_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":base_poisson_solver",
        ":poisson_solver",
        ":poisson_solver_py_pb2",
        ":poisson_solver_testutil",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//net/proto2/python/public",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//research/simulation/tensorflow/fluid/framework/tf1:fluid",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "poisson_solver_residuals_test",
    timeout = "long",
    srcs = ["poisson_solver_residuals_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 14,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":base_poisson_solver",
        ":multigrid_utils",
        ":poisson_solver_py_pb2",
        ":poisson_solver_testutil",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "poisson_solver_test",
    timeout = "long",
    srcs = ["poisson_solver_test.py"],
    args = ["--xla_jf_conv_full_precision=True"],
    python_version = "PY3",
    shard_count = 50,
    srcs_version = "PY3",
    tags = [
        "optonly",
        "requires-dragonfish:4",
    ],
    deps = [
        ":base_poisson_solver",
        ":poisson_solver",
        ":poisson_solver_py_pb2",
        ":poisson_solver_testutil",
        "//learning/brain/google/xla:deepsea_hardware_device",
        "//pyglib:gfile",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:grid_parametrization_py_pb2",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "jacobi_solver",
    srcs = ["jacobi_solver.py"],
    srcs_version = "PY3",
    deps = [
        ":base_poisson_solver",
        ":poisson_solver_py_pb2",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "multigrid",
    srcs = ["multigrid.py"],
    srcs_version = "PY3",
    deps = [
        ":multigrid_utils",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "multigrid_3d",
    srcs = ["multigrid_3d.py"],
    srcs_version = "PY3",
    deps = [
        ":multigrid_3d_utils",
        ":multigrid_utils",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//research/simulation/tensorflow/fluid/framework:util",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "multigrid_3d_utils",
    srcs = ["multigrid_3d_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":multigrid_utils",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "multigrid_test_common",
    srcs = ["multigrid_test_common.py"],
    srcs_version = "PY3",
    deps = [
        ":multigrid_3d_utils",
        ":multigrid_utils",
        "//research/simulation/tensorflow/fluid/framework:tpu_runner",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "multigrid_utils",
    srcs = ["multigrid_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/communication:halo_exchange_utils",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "poisson_solver",
    srcs = ["poisson_solver.py"],
    srcs_version = "PY3",
    deps = [
        ":base_poisson_solver",
        ":conjugate_gradient_solver",
        ":fast_diagonalization_solver",
        ":jacobi_solver",
        ":multigrid_3d",
        ":multigrid_utils",
        ":poisson_solver_py_pb2",
        "//research/simulation/tensorflow/fluid/framework:analytics_util",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/utility:common_ops",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/swirl_lm/utility:grid_parametrization_py_pb2",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "poisson_solver_testutil",
    srcs = ["poisson_solver_testutil.py"],
    srcs_version = "PY3",
    deps = [
        ":multigrid_utils",
        ":poisson_solver",
        "//pyglib:gfile",
        "//research/simulation/tensorflow/fluid/framework:initializer",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/utility:grid_parametrization",
        "//third_party/py/tensorflow",
    ],
)

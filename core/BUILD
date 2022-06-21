# The simulation core: main algorithm for variable density low Mach number flow.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "simulation",
    srcs = ["simulation.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/numpy",
        "//third_party/py/swirl_lm/base:parameters",
        "//third_party/py/swirl_lm/base:physical_variable_keys_manager",
        "//third_party/py/swirl_lm/boundary_condition:immersed_boundary_method",
        "//third_party/py/swirl_lm/communication:halo_exchange",
        "//third_party/py/swirl_lm/equations:pressure",
        "//third_party/py/swirl_lm/equations:scalars",
        "//third_party/py/swirl_lm/equations:velocity",
        "//third_party/py/swirl_lm/physics/thermodynamics:thermodynamics_manager",
        "//third_party/py/swirl_lm/utility:components_debug",
        "//third_party/py/swirl_lm/utility:get_kernel_fn",
        "//third_party/py/swirl_lm/utility:monitor",
        "//third_party/py/swirl_lm/utility:types",
        "//third_party/py/tensorflow",
    ],
)

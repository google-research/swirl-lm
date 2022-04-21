load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//research/simulation/tensorflow/fluid:sim_research_fluid",
        "//third_party/py/swirl_lm:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

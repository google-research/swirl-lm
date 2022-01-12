load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = ["//research/simulation/tensorflow/fluid:sim_research_fluid"],
)

licenses(["notice"])

exports_files(["LICENSE"])

pytype_strict_library(
    name = "swirl_lm",
    srcs = glob(
        ["*.py"],
        exclude = ["*_test.py"],
    ),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/swirl_lm/utility",
    ],
)

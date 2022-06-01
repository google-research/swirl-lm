# swirl_lm (go/swirl_lm)

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
load("//third_party/py/etils:build_defs.bzl", "glob_py_srcs")

package(default_visibility = [":internal"])

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//third_party/py/swirl_lm/...",
    ],
)

# swirl_lm public API
# This is a single py_library rule which centralize all files/deps.
pytype_strict_library(
    name = "swirl_lm",
    # Auto-collect all `.py` files (excluding tests).
    # If using subfolders, sources in subfolders should be collected and added here.
    # See go/oss-kit#single-rule-pattern for documentation.
    srcs = glob_py_srcs(),
    visibility = ["//visibility:public"],
    # Project dependencies (matching the `pip install` deps defined in `pyproject.toml`)
    deps = [
    ],
)

copybara_config_test(
    name = "copybara_test",
    config = "copy.bara.sky",
    deps = [
        "//third_party/py/etils:copybara_utils",
    ],
)

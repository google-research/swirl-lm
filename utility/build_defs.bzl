"""Macro for TF1 and TF2 targets for a single test."""

load("//devtools/python/blaze:strict.bzl", "py_strict_test")

def tf1_and_tf2_test(**kwargs):
    """TF1 and TF2 targets for a single test.

    This macro expands to two py_strict_test targets and a test_suite grouping
    them. The test_suite uses the given 'name', which is expected to be of the
    format
       XXX_test,
    while the two py_strict_test targets are given corresponding names
       XXX_tf1_test,
       XXX_tf2_test.
    The remaining attributes are included in the rules for both test targets,
    with two adjustments:
       * //learning/brain/public:disable_tf2 is added to the deps of the tf1
         target
       * following the semantics of py_test, if there is no 'main' attribute
         and srcs includes "XXX_test.py", this default 'main' is explicitly
         added to the two test targets.

    Args:
      **kwargs: attributes for both test targets. (see above.)
    """

    if "name" not in kwargs:
        fail("'name' required")
    name = kwargs["name"]
    if not name.endswith("_test"):
        fail("name must end with '_test'")
    base = name[:-len("_test")]
    tf1_name = "%s_tf1_test" % base
    tf2_name = "%s_tf2_test" % base
    tf1_deps = ["//learning/brain/public:disable_tf2"] + kwargs["deps"]
    if "main" not in kwargs and "srcs" in kwargs:
        main = "%s.py" % name
        if main in kwargs["srcs"]:
            kwargs["main"] = main
    py_strict_test(**(kwargs | {"name": tf2_name}))
    py_strict_test(**(kwargs | {"name": tf1_name, "deps": tf1_deps}))
    native.test_suite(name = name, tests = [":%s" % tf1_name, ":%s" % tf2_name])

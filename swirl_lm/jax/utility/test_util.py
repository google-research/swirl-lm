# Copyright 2025 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for common operations used in tests."""

from typing import Any, Callable, Iterable, Tuple
import jax
import numpy as np

# axis_order should be one of ['xyz', 'xzy', 'yxz', 'yzx', 'zyx', 'zxy'].
# All the tests are expected to pass for all these orders.
# It is recommended to use this order in all the tests for consistency.
data_axis_order = 'zxy'


def assert_jax_retval_allclose(
    fn: Callable[..., jax.Array | Tuple[jax.Array, ...]],
    *args_sets: Tuple[Any, ...],
    expected_outputs: Tuple[jax.Array, ...] | Tuple[Tuple[jax.Array, ...], ...],
    atol: float = 1e-5,
    rtol: float = 1e-5,
    static_argnames: str | Iterable[str] | None = None,
    include_jit_test: bool = True,
    strict: bool = True,
) -> None:
  """Tests a JAX function with and without JIT compilation.

  Allows multiple or variable-length arguments and single or multiple return
  values. Also allows specifying static argument names for JIT.

  Args:
    fn: The JAX function to test.
    *args_sets: Multiple sets of arguments to be passed to the function. Each
      set should be a tuple of arguments.
    expected_outputs: A tuple of expected outputs. Each element can be a single
      value or a tuple of values, matching the structure of the outputs returned
      by the function.
    atol: Absolute tolerance for numerical comparisons.
    rtol: Relative tolerance for numerical comparisons.
    static_argnames: An optional list of argument names to be treated as static
      during JIT compilation.
    include_jit_test: If True, include the JIT test. Test without JIT is always
      included.
    strict: If True, raise an AssertionError when either the shape or the data
      type of the arguments does not match.
  """

  def _assert_allclose(actual: Any, desired: Any):
    # Check if the returned value is a single value or a tuple
    if isinstance(actual, tuple) and isinstance(desired, tuple):
      for j in range(len(actual)):
        np.testing.assert_allclose(
            actual[j],
            desired[j],
            atol=atol,
            rtol=rtol,
            strict=strict,
        )
    else:
      np.testing.assert_allclose(
          actual,
          desired,
          atol=atol,
          rtol=rtol,
          strict=strict,
      )

  for i, args in enumerate(args_sets):
    _assert_allclose(fn(*args), expected_outputs[i])
  if include_jit_test:
    jit_fn = jax.jit(fn, static_argnames=static_argnames)
    for i, args in enumerate(args_sets):
      _assert_allclose(jit_fn(*args), expected_outputs[i])

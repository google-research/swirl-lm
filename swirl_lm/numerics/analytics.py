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

"""Library for computing basic analytics in a distributed setting."""

from typing import Optional, Sequence

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal

_CTYPE = types.TF_COMPLEX_DTYPE
_DTYPE = types.TF_DTYPE
VectorField = types.VectorField


def moments(
    f1: FlowFieldVal,
    n: Sequence[int],
    halos: Sequence[int],
    homogeneous_dims: Sequence[bool],
    replicas: np.ndarray,
    f2: Optional[FlowFieldVal] = None,
    f1_ref: Optional[FlowFieldVal] = None,
    f2_ref: Optional[FlowFieldVal] = None,
):
  """Calculates moments of the field: E[(f - E[f])^k].

  The calculation will exclude the halo region on each core, and there will be
  one such moment calculated for each of the exponents contained in `n`. The
  moments are calculated relative to the mean in the homogeneous directions of
  the grid, or relative to a reference state if one is provided. If `f2` is
  provided, then the computation will yield a cross moment between variables
  `f1` and `f2`.

  Args:
    f1: The field/variable on the local core. This is expected to be expressed
      in the form of a list of 2D Tensors representing x-y slices, where each
      list element represents the slice of at a given z coordinate in ascending
      z order.
    n: A sequence of the moment orders to compute. They should always be greater
      than 1.
    halos: The width of the (symmetric) halos for each dimension: for example
      [1, 2, 3] means the halos for `f` have width of 1, 2, 3 on both sides in
      x, y, z dimension respectively.
    homogeneous_dims: A sequence of booleans indicating the homogeneous
      dimensions of the physical grid, e.g. homogeneous_dims = [True, False,
      True] denotes that x and z are homogeneous directions.
    replicas: A 3D numpy array representing the mapping from the core replica
      coordinate to the `replica_id`. The number of cores in each dimension is
      the number of splits of the global input for the transformation.
    f2: Optional second field/variable on the local core for computing a cross
      moment with `f1`. It should have the same structure as `f1` (i.e. a list
      of 2D tensors).
    f1_ref: Optional field/variable on the local core to subtract from field `f`
      before the moment computation. If not passed, the mean of `f` is used by
      default. This should have the same structure as `f1` (i.e. a list of 2D
      tensors).
    f2_ref: Optional field/variable on the local core to subtract from field
      `f2` prior to computing the cross moment. If not provided, the mean of
      `f2` is used by default. This should have the same structure as `f2` (i.e.
      a list of 2D tensors).

  Returns:
    A list of scalar tensors representing the n-th order moment of the field(s).
    There will be one such tensor for each element of `n`.
  """
  def subtract(f, ref):
    """Subtracts `ref` from `f`, broadcasting if needed."""
    if len(ref) == 1:
      return tf.nest.map_structure(lambda f: f - ref[0], f)
    else:
      return tf.nest.map_structure(lambda f, ref: f - ref, f, ref)
  reduction_axis = [i for i, active in enumerate(homogeneous_dims) if active]
  # Strip halos from all fields.
  f1 = common_ops.strip_halos(f1, halos)
  f1_mean = common_ops.global_mean(f1, replicas, axis=reduction_axis)
  if f1_ref is not None:
    f1_ref = common_ops.strip_halos(f1_ref, halos)
  else:
    f1_ref = f1_mean

  if f2 is not None:
    f2 = common_ops.strip_halos(f2, halos)
    f2_ref = common_ops.strip_halos(
        f2_ref, halos) if f2_ref is not None else common_ops.global_mean(
            f2, replicas, axis=reduction_axis)
    f2 = subtract(f2, f2_ref)

  moment_lst = []
  for order in n:
    if f2 is None:
      if order == 1:
        moment_lst.append(f1_mean)
      else:
        deviation = subtract(f1, f1_ref)
        moment_n = tf.nest.map_structure(
            # pylint:disable=cell-var-from-loop
            lambda deviation: tf.math.pow(deviation, order),
            deviation)
        moment_lst.append(moment_n)
    else:  # Compute cross moment.
      deviation = subtract(f1, f1_ref)
      f_cross = tf.nest.map_structure(lambda deviation, f2: deviation * f2,
                                      deviation, f2)
      moment_n = tf.nest.map_structure(
          # pylint:disable=cell-var-from-loop
          lambda f_cross: tf.math.pow(f_cross, order),
          f_cross)
      moment_lst.append(moment_n)

  return [
      common_ops.global_mean(moment, replicas, axis=reduction_axis)
      for moment in moment_lst
  ]


def pair_distance_with_tol(
    lhs: FlowFieldVal,
    rhs: FlowFieldVal,
    atol: float,
    rtol: float,
    replicas: np.ndarray,
    halo_width: int,
    symmetric: bool = False,
) -> tf.Tensor:
  r"""Computes max distance of `lhs` vs `rhs` without halos, within tolerance.

  By default using the `rhs` as the reference only, but could also use the `lhs`
  as the reference at the same time if `symmetric` is set.

  1. Compute `tol = atol + rtol * abs(rhs)`
  2. Compute `diff = lhs - rhs`
  3. Define `distance = max(abs(diff) - tol)` and check:
     - Case 1: Close enough if non-positive, where `-tol <= diff <= +tol`:
       *       *          *
       ^-tol
               ^diff
                          ^+tol
     - Case 2: Not close if positive, where `diff > tol` or `diff < -tol`:
       *       *          *
       ^-tol
               ^+tol
                          ^diff

  Note that as the distance is defined as a difference, its range is
  `(-\infinity, \infinity)`.
  - When negative or zero, it implies the two vectors are close enough.
  - When positive, it means the two vectors' difference larger than given
    thresholds, `atol` joint with `rtol`.

  Args:
    lhs: Vector on the left hand side for distance.
    rhs: Vector on the right hand side for distance, and used as reference for
      relative tolerance `rtol`.
    atol: Absolute difference for comparison.
    rtol: Relative difference for comparison, could use `rhs` or together with
      `lhs` as reference if symmetric.
    replicas: A numpy array that maps a replica's grid coordinate to its
      replica_id, e.g. replicas[0, 0, 0] = 0, replicas[0, 0, 1] = 1.
    halo_width: Width of the halo.
    symmetric: Whether to use the `lhs` as reference as well. If enabled will
      use both `lhs` and `rhs` as reference and get the maximum of these two
      relative distances, otherwise using `rhs` only as reference.

  Returns:
    The maximum distance (defined as a max difference) of two vectors.

  Raises:
    ValueError: If `atol` or `rtol` is negative.
  """
  if atol < 0 or rtol < 0:
    raise ValueError('Invalid input of (atol, rtol) = (%g, %g) < 0.' %
                     (atol, rtol))

  lhs = halo_exchange.clear_halos(lhs, halo_width)
  rhs = halo_exchange.clear_halos(rhs, halo_width)

  diff = lhs - rhs

  num_replicas = np.prod(replicas.shape)
  group_assignment = np.array([range(num_replicas)], dtype=np.int32)

  def pair_distance_fn(d: FlowFieldVal, raw: FlowFieldVal) -> tf.Tensor:
    """Max distance from absolute diff and raw vector."""
    tol = atol + rtol * tf.math.abs(raw)
    distance = tf.math.abs(d) - tol
    return common_ops.global_reduce(distance, tf.math.reduce_max,
                                    group_assignment)

  rhs_distance = pair_distance_fn(diff, rhs)
  if symmetric:
    return tf.maximum(pair_distance_fn(diff, lhs), rhs_distance)
  else:
    return rhs_distance

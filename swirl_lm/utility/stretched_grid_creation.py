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

"""Tool to allow flexible creation of stretched grids.

Flexible functionality for geometrically and uniformly spaced grids.

* Geometric grids: Exactly 2 out of 4 of {n, r, z_max, dz_max} must be
specified.
* Uniform grids: Exactly 2 out of 3 of {n, dz, z_max} must be specified.

The function `full_grid_from_spec()` will create a full grid from a sequence of
grid specs and is the primary interface for this module.
"""

from collections.abc import Mapping, Sequence
import enum
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


class GridType(enum.Enum):
  GEOMETRIC = 1
  UNIFORM = 2

GridSpec: TypeAlias = tuple[GridType, Mapping[str, int | float]]


def create_geometric_grid(
    z0: float, dz0: float, n: int, r: float
) -> np.ndarray:
  """Create a geometrically spaced grid.

  Args:
    z0: The first value of the grid.
    dz0: The initial spacing between grid points.
    n: The number of grid points.
    r: The ratio between adjacent spacings.

  Returns:
    A geometrically spaced grid.
  """
  powers = np.arange(-1, n - 1)
  dz = dz0 * r**powers
  dz[0] = 0  # The first slot of z will be just z0.
  z = z0 + np.cumsum(dz)
  return z


def _geometric_sum(r: float, d: int) -> float:
  """Calculate the sum S = 1 + r + r^2 + ... + r^d."""
  if r == 1:
    return d + 1
  return (r ** (d + 1) - 1) / (r - 1)


class GeometricGrid:
  """A class to represent a geometrically spaced grid.

  Let dz, z be arrays, given by:

    dz = dz0 * [0, 1, r, r^2, r^3, ..., r^d]
    z = z0 + [0, dz, dz0 + r*dz0, ..., dz0(1 + r + ... + r^d)]

  z can be rewritten as:
    z = z0 + dz0 * [0, 1, 1 + r, 1 + r + r^2, ..., 1 + r + r^2 + ... + r^d]

  If there are n total elements in dz or z, then
    n = d + 2

  Geometric series formulas:
    dz_max = dz0 * r^d
    z_max = z0 + dz0 * S
  where
    S = 1 + r + r^2 + ... + r^d.
  which can be rewritten as:
    S = (r^(d+1) - 1) / (r - 1)

  Currently:
      - z0 and dz0 are required to be specified
      - Exactly 2 out of 4 of {n, r, z_max, dz_max} must be specified
  """

  def __init__(
      self,
      *,
      z0: float,
      dz0: float,
      n: int | None = None,
      r: float | None = None,
      z_max: float | None = None,
      dz_max: float | None = None,
  ):
    """Initialize a GeometricGrid.

    z0 and dz0 must be provided, and exactly 2 out of 4 of {n, r, z_max, dz_max}
    must be provided.

    Args:
      z0: The first value of the grid.
      dz0: The initial spacing between grid points.
      n: The number of grid points.
      r: The ratio between adjacent spacings.
      z_max: The final value of the grid.
      dz_max: The final spacing between grid points.

    Exceptions:
      ValueError: If the arguments are not specified correctly or if the given
        arguments produce an invalid grid, with n <= 2.
    """
    self.z0 = z0
    self.dz0 = dz0
    # Must specify only 2 out of 4 of n, r, z_max, dz_max.
    # There are 6 possible combinations.
    error_msg = 'Must specify exactly 2 out of the 4 args: n, r, z_max, dz_max'
    n_error_msg = 'The provided arguments are not consistent with n >= 3.'

    if n is not None and r is not None:
      if n <= 2:
        raise ValueError(n_error_msg)
      if z_max is not None or dz_max is not None:
        raise ValueError(error_msg)
      # Calculate z_max, dz_max.
      self.n = n
      self.r = r
      d = n - 2
      s = _geometric_sum(r, d)
      self.z_max = z0 + dz0 * s
      self.dz_max = dz0 * r**d
    elif n is not None and z_max is not None:
      if n <= 2:
        raise ValueError(n_error_msg)
      if r is not None or dz_max is not None:
        raise ValueError(error_msg)
      # Calculate r, dz_max.
      self.n = n
      self.z_max = z_max
      d = n - 2

      def fun(r):
        lhs = (z_max - z0) / dz0
        rhs = (r ** (d + 1) - 1) / (r - 1)
        return lhs - rhs

      # For purposes of calculating r, assume the minimum possible grid ratio is
      # 0.5 and the maximum possible grid ratio is 2.0.
      rmin = 0.5
      rmax = 2.0
      self.r = scipy.optimize.brentq(fun, rmin, rmax)
      self.dz_max = dz0 * self.r**d
    elif n is not None and dz_max is not None:
      if n <= 2:
        raise ValueError(n_error_msg)
      if z_max is not None or r is not None:
        raise ValueError(error_msg)
      self.n = n
      self.dz_max = dz_max
      d = n - 2
      # Calculate r, z_max.
      self.r = (dz_max / dz0) ** (1 / d)
      s = _geometric_sum(self.r, d)
      self.z_max = z0 + dz0 * s
    elif r is not None and z_max is not None:
      if n is not None or dz_max is not None:
        raise ValueError(error_msg)
      self.z_max = z_max
      # Calculate n, dz_max.
      # First, find the integer d (or n) such that given r, z_max, the grid goes
      # just past z_max.
      s = (z_max - z0) / dz0
      numerator = np.log(1 + s * (r - 1))
      denominator = np.log(r)
      d = int(np.ceil(numerator / denominator - 1))
      n = d + 2
      if n <= 2:
        raise ValueError(n_error_msg)
      self.n = n

      # Due to the constraint that n is an integer, if we use the given r, then
      # the grid will go past z_max. But we want the end of the grid to exactly
      # equal z_max. To accomplish this, we instead adjust r slightly.
      # Now, given n, z_max, calculate r.
      g = GeometricGrid(z0=z0, dz0=dz0, n=n, z_max=z_max)
      self.r = g.r
      self.dz_max = g.dz_max
    elif r is not None and dz_max is not None:
      self.dz_max = dz_max
      if n is not None or z_max is not None:
        raise ValueError(error_msg)
      d_float = np.log(dz_max / dz0) / np.log(r)
      d = int(np.ceil(d_float))
      n = d + 2
      if n <= 2:
        raise ValueError(n_error_msg)

      # Due to the constraint that n is an integer, the final dz_max will not be
      # the desired dz_max. To make the final dz_max equal to exactly what is
      # requested, we modify r.
      self.r = (dz_max / dz0) ** (1 / d)
      self.n = n
      s = _geometric_sum(self.r, d)
      self.z_max = z0 + dz0 * s
    elif z_max is not None and dz_max is not None:
      self.z_max = z_max
      if n is not None or r is not None:
        raise ValueError(error_msg)
      # Combine the 2 equations:
      #  1. dz_max = dz0 * r^d
      #  2. z_max = z0 + dz0 * S, with S given above
      # Let a = dz_max / dz0 = r^d
      # Let b = (z_max - z0) / dz0 = S
      # We can eventually simplify to:
      a = dz_max / dz0
      b = (z_max - z0) / dz0
      r = (b - 1) / (b - a)
      d_float = np.log(a) / np.log(r)

      d = int(np.ceil(d_float))
      n = d + 2
      if n <= 2:
        raise ValueError(n_error_msg)
      self.n = n

      # Due to the constraint that n is an integer, recalculate r and dz_max.
      g = GeometricGrid(z0=z0, dz0=dz0, n=n, z_max=z_max)
      self.r = g.r
      self.dz_max = g.dz_max
    else:
      raise ValueError('Something went wrong with the arguments. ' + error_msg)

    # Manifest the actual length-n geometrically spaced grid.
    self.grid = create_geometric_grid(z0, dz0, self.n, self.r)

  def next_z_dz(self) -> tuple[float, float]:
    """Return what the next value for z and dz would be."""
    dz_next = self.dz_max * self.r
    z_next = self.z_max + dz_next
    return z_next, dz_next

  def __repr__(self):
    return (
        f'GeometricGrid(z0={self.z0:.2f}, dz0={self.dz0:.2f}, n={self.n},'
        f' r={self.r:.3f}, z_max={self.z_max:.2f}, dz_max={self.dz_max:.2f})'
    )


class UniformGrid:
  """A class to represent a uniformly spaced grid.

  Fundamental relation: z_max = z0 + (n-1) * dz.
  """

  def __init__(
      self,
      *,
      z0: float,
      n: int | None = None,
      dz: float | None = None,
      z_max: float | None = None,
  ):
    """Initialize a UniformGrid.

    Args:
      z0: The first value of the grid.
      n: The number of grid points.
      dz: The spacing between grid points.
      z_max: The final value of the grid.

    Exceptions:
      ValueError: If the arguments are not specified correctly or if the given
        arguments produce an invalid grid, with n <= 2.
    """
    self.z0 = z0
    # Must specify exactly 2 out of n, dz, z_max.
    error_msg = 'Must specify exactly 2 out of the 3 args: n, dz, z_max.'
    n_error_msg = 'The provided arguments are not consistent with n >= 2.'

    # If n is not specified, then dz will be modified to keep n an integer.
    # The initialization will make dz smaller than requested.
    if n is not None and dz is not None:
      if z_max is not None:
        raise ValueError(error_msg)
      self.n = n
      self.dz = dz
      self.z_max = z0 + (n - 1) * dz
    elif n is not None and z_max is not None:
      if dz is not None:
        raise ValueError(error_msg)
      self.n = n
      self.z_max = z_max
      self.dz = (z_max - z0) / (n - 1)
    elif dz is not None and z_max is not None:
      if n is not None:
        raise ValueError(error_msg)
      self.z_max = z_max
      n_float = 1 + (z_max - z0) / dz
      n = int(np.ceil(n_float))
      if n <= 1:
        raise ValueError(n_error_msg)
      self.n = n

      g = UniformGrid(z0=z0, n=n, z_max=z_max)
      self.dz = g.dz
    else:
      raise ValueError('Something went wrong with the arguments. ' + error_msg)

    self.grid = np.linspace(self.z0, self.z_max, self.n)

  def next_z_dz(self) -> tuple[float, float]:
    """Return what the next value for z and dz would be."""
    z_next = self.z_max + self.dz
    return z_next, self.dz

  def __repr__(self):
    return (
        f'UniformGrid(z0={self.z0:.2f}, n={self.n}, dz={self.dz:.2f},'
        f' z_max={self.z_max:.2f})'
    )


def full_grid_from_spec(
    grid_specs: Sequence[GridSpec], n_total: int | None = None
) -> tuple[np.ndarray, list[GeometricGrid | UniformGrid]]:
  """Create a full grid from a sequence of grid specs.

  A grid spec is a tuple of (grid_type, grid_args), where grid_type is either
  GridType.GEOMETRIC or GridType.UNIFORM, and grid_args is a dictionary of
  arguments that are input for that grid type.

  When using this function, the first grid must have z0 specified (and
  additionally, if the first grid is Geometric, then dz0 must also be
  specified). Subsequent grids after the first do not need to have z0 or dz0
  specified because they are computed automatically. For the final grid, if n=-1
  is used, then the final n will be computed such that the total number of grid
  points in the full grid is n_total.

  Example usage:

  from swirl_lm.utility import stretched_grid_creation as sgc

  n_total = 156
  grid_specs = [
    (sgc.GridType.GEOMETRIC, {'z0': 10, 'dz0': 20., 'r': 1.04, 'dz_max': 150}),
    (sgc.GridType.GEOMETRIC, {'z_max': 15000, 'dz_max': 200}),
    (sgc.GridType.GEOMETRIC, {'r': 1.05, 'z_max': 29000}),
    (sgc.GridType.UNIFORM, {'dz': 880, 'n': -1}),
  ]
  z, grids = sgc.full_grid_from_spec(grid_specs, n_total=n_total)

  fig, axarr = sgc.plot_grid(z)
  print(f'{len(z)=}')
  for g in grids:
    print(g)

  Args:
    grid_specs: A sequence of tuples of (grid_type, grid_args).
    n_total: The total number of grid points in the full grid. Must be specified
      if the last grid in grid_specs has n=-1, otherwise the argument is
      ignored.

  Returns:
    A tuple of the complete grid as a numpy array and a list of the individual
    grids, in the order they appear in `grid_specs`.
  """
  n_grids = len(grid_specs)
  grids: list[GeometricGrid | UniformGrid] = []
  full_z_grid = np.empty(0)
  for j, grid_spec in enumerate(grid_specs):
    grid_type, grid_args = grid_spec
    if j == 0:
      if grid_type is GridType.GEOMETRIC:
        # Note: must have z0, dz0 in grid_args.
        grid = GeometricGrid(**grid_args)
      elif grid_type is GridType.UNIFORM:
        # Note: must have z0 in grid_args.
        grid = UniformGrid(**grid_args)
      else:
        raise ValueError(f'Unknown grid type: {grid_type}')
    else:  # j >= 1
      if j == n_grids - 1:
        # Last grid.
        if 'n' in grid_args and grid_args['n'] == -1:
          if n_total is None:
            raise ValueError(
                'Must supply n_total to full_grid_from_spec() if using n=-1 in'
                ' last grid.'
            )
          # Compute n for the last grid such that total grid points is n_total.
          n_so_far = sum(g.n for g in grids)
          required_n = n_total - n_so_far
          if grid_type is GridType.GEOMETRIC and required_n < 3:
            raise ValueError(
                'Invalid grid spec. Fewer than 3 grid points would be used for'
                ' final geometric grid.'
            )
          elif grid_type is GridType.UNIFORM and required_n < 2:
            raise ValueError(
                'Invalid grid spec. Fewer than 2 grid points would be used for'
                ' final uniform grid.'
            )

          grid_args = dict(grid_args)  # Make a copy to avoid modifying input.
          grid_args['n'] = required_n

      next_z, next_dz = grids[-1].next_z_dz()
      if grid_type is GridType.GEOMETRIC:
        grid = GeometricGrid(z0=next_z, dz0=next_dz, **grid_args)
      elif grid_type is GridType.UNIFORM:
        grid = UniformGrid(z0=next_z, **grid_args)
      else:
        raise ValueError(f'Unknown grid type: {grid_type}')
    grids.append(grid)
    full_z_grid = np.concatenate((full_z_grid, grid.grid))
  return full_z_grid, grids


def plot_grid(
    z: np.ndarray, figsize: tuple[float, float] = (6, 8)
) -> tuple[plt.Figure, list[plt.Axes]]:
  """Plot the grid, the grid spacings, and the ratio of grid spacings.

  Args:
    z: An array of grid coordinates, e.g., produced by `full_grid_from_spec()`.
    figsize: The desired figure size.

  Returns:
    A tuple of the figure and the axes.
  """
  dz = np.diff(z)
  ratios = dz[1:] / dz[0:-1]

  fig, axarr = plt.subplots(3, 1, figsize=figsize)
  ax0, ax1, ax2 = axarr[0], axarr[1], axarr[2]
  ax0.plot(z, np.arange(len(z)))
  ax0.set_xlabel('z')
  ax0.set_ylabel('z index')

  ax1.plot(z[:-1], dz)
  ax1.set_xlabel('z)')
  ax1.set_ylabel('dz')

  ax2.plot(z[1:-1], ratios)
  ax2.set_xlabel('z')
  ax2.set_ylabel(r'Ratio $\Delta z_{j+1}/\Delta z_j$')

  plt.tight_layout()
  return fig, axarr

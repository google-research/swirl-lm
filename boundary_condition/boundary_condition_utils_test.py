"""Tests for swirl_lm.boundary_condition.boundary_condition_utils."""

import numpy as np
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.communication import halo_exchange
import tensorflow as tf
from google3.testing.pybase import parameterized

_NP_DTYPE = np.float32
_TF_DTYPE = tf.float32


class BoundaryConditionUtilsTest(tf.test.TestCase, parameterized.TestCase):

  _UNDEFINED_VELOCITY_BC = [
      {
          'u': None,
          'v': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
          'w': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
      },
      {
          'u': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
          'v': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
          'w': None,
      },
      {
          'u': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
          'v': None,
          'w': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
      },
      {
          'u': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
          'v': None,
          'w': None,
      },
      {
          'u': None,
          'v': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
          'w': None,
      },
      {
          'u': None,
          'v': None,
          'w': [[
              (halo_exchange.BCType.DIRICHLET, 1.0),
          ] * 2] * 3,
      },
      {
          'u': None,
          'v': None,
          'w': None,
      },
  ]

  @parameterized.parameters(*zip(_UNDEFINED_VELOCITY_BC))
  def testFindBCTypeReturnsPeriodicBCIfAnyVelocityBCIsUndefined(self, bc):
    """Checks if boundary type is set to periodic when velocity BC undefined."""
    bc_types = boundary_condition_utils.find_bc_type(
        bc, (True, True, True))

    expected = [[boundary_condition_utils.BoundaryType.PERIODIC,] * 2,] * 3

    self.assertAllEqual(expected, bc_types)

  def testFindBCTypeDerivesCorrectBoundaryConditions(self):
    """Checks if boundary types are found correctly."""
    boundary_conditions = {
        'u': [[(halo_exchange.BCType.DIRICHLET, 1.0),
               (halo_exchange.BCType.NEUMANN, 0.0)],
              [None, None],
              [(halo_exchange.BCType.NEUMANN, 0.0),
               (halo_exchange.BCType.DIRICHLET, 0.0)],],
        'v': [[(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)],
              [None, None],
              [(halo_exchange.BCType.NEUMANN, 0.0),
               (halo_exchange.BCType.DIRICHLET, 0.0)],],
        'w': [[(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)],
              [None, None],
              [(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.DIRICHLET, 0.0)],],
    }

    bc_types = boundary_condition_utils.find_bc_type(
        boundary_conditions, (False, True, False))

    with self.subTest('x0IsInflow'):
      self.assertEqual(bc_types[0][0],
                       boundary_condition_utils.BoundaryType.INFLOW)

    with self.subTest('x1IsOutflow'):
      self.assertEqual(bc_types[0][1],
                       boundary_condition_utils.BoundaryType.OUTFLOW)

    with self.subTest('y0IsPeriodic'):
      self.assertEqual(
          bc_types[1][0],
          boundary_condition_utils.BoundaryType.PERIODIC)

    with self.subTest('y1IsPeriodic'):
      self.assertEqual(
          bc_types[1][1],
          boundary_condition_utils.BoundaryType.PERIODIC)

    with self.subTest('z0IsSlipWall'):
      self.assertEqual(bc_types[2][0],
                       boundary_condition_utils.BoundaryType.SLIP_WALL)

    with self.subTest('z1IsNonSlipWall'):
      self.assertEqual(bc_types[2][1],
                       boundary_condition_utils.BoundaryType.NON_SLIP_WALL)

  def testFindBCTypeDistinguishesSlipWallAndShearWall(self):
    """Checks if boundary types are found correctly."""
    boundary_conditions = {
        'u': [[None, None],
              [None, None],
              [(halo_exchange.BCType.NEUMANN, 20.0),
               (halo_exchange.BCType.NEUMANN, 0.0)],],
        'v': [[None, None],
              [None, None],
              [(halo_exchange.BCType.NEUMANN, 0.0),
               (halo_exchange.BCType.NEUMANN, 0.0)],],
        'w': [[None, None],
              [None, None],
              [(halo_exchange.BCType.DIRICHLET, 0.0),
               (halo_exchange.BCType.DIRICHLET, 0.0)],],
    }

    bc_types = boundary_condition_utils.find_bc_type(
        boundary_conditions, (True, True, False))

    with self.subTest('z0IsShearWall'):
      self.assertEqual(bc_types[2][0],
                       boundary_condition_utils.BoundaryType.SHEAR_WALL)

    with self.subTest('z1IsSlipWall'):
      self.assertEqual(bc_types[2][1],
                       boundary_condition_utils.BoundaryType.SLIP_WALL)

  def testDirichletGhostCellQuickProvidesCorrectValuesInGhostCells(self):
    """Checks if values in ghost cells are corrected for Dirichlet BC."""
    nx = 8
    ny = 10
    nz = 6
    halo_width = 2

    def slice_x(val, halo_val=None):
      """Get a slice perpendicular to the x axis with constant `val`."""
      halo_val = val if halo_val is None else halo_val
      row = halo_val * np.ones((1, ny), dtype=_NP_DTYPE)
      row[0, halo_width:-halo_width] = val
      return [
          halo_val * tf.ones((1, ny), dtype=_TF_DTYPE),
      ] * halo_width + [
          tf.convert_to_tensor(row, dtype=_TF_DTYPE),
      ] * (nz - 2 * halo_width) + [
          halo_val * tf.ones((1, ny), dtype=_TF_DTYPE),
      ] * halo_width

    def slice_y(val, halo_val=None):
      """Get a slice perpendicular to the y axis with constant `val`."""
      halo_val = val if halo_val is None else halo_val
      row = halo_val * np.ones((nx, 1), dtype=_NP_DTYPE)
      row[halo_width:-halo_width, 0] = val
      return [
          halo_val * tf.ones((nx, 1), dtype=_TF_DTYPE),
      ] * halo_width + [
          tf.convert_to_tensor(row, dtype=_TF_DTYPE),
      ] * (nz - 2 * halo_width) + [
          halo_val * tf.ones((nx, 1), dtype=_TF_DTYPE),
      ] * halo_width

    def slice_z(val, halo_val=None):
      """Get a slice perpendicular to the z axis with constant `val`."""
      halo_val = val if halo_val is None else halo_val
      plane = halo_val * np.ones((nx, ny), dtype=_NP_DTYPE)
      plane[halo_width:-halo_width, halo_width:-halo_width] = val
      return tf.convert_to_tensor(plane, dtype=_TF_DTYPE)

    bc = {
        'u': [[(halo_exchange.BCType.DIRICHLET, 1.0),
               (halo_exchange.BCType.DIRICHLET, [slice_x(-2.0),
                                                 slice_x(-1.0),
                                                 slice_x(-3.0),])],
              [(halo_exchange.BCType.NEUMANN, 1.0),
               (halo_exchange.BCType.NEUMANN, -1.0)],
              [(halo_exchange.BCType.DIRICHLET, [slice_z(2.0),
                                                 slice_z(2.0)]),
               (halo_exchange.BCType.DIRICHLET, -2.0)]],
        'v': [[(halo_exchange.BCType.NEUMANN, 1.0),
               (halo_exchange.BCType.NEUMANN, -1.0)],
              [(halo_exchange.BCType.DIRICHLET, [slice_y(1.0),
                                                 slice_y(1.0)]),
               (halo_exchange.BCType.DIRICHLET, -1.0)], [None, None]],
    }

    bc_type = [
        [
            boundary_condition_utils.BoundaryType.SHEAR_WALL,
            boundary_condition_utils.BoundaryType.SHEAR_WALL
        ],
        [
            boundary_condition_utils.BoundaryType.OUTFLOW,
            boundary_condition_utils.BoundaryType.OUTFLOW
        ],
        [boundary_condition_utils.BoundaryType.INFLOW, None],
    ]

    hollow_ones = np.zeros((nz, nx, ny), dtype=_NP_DTYPE)
    hollow_ones[2:-2, 2:-2, 2:-2] = 1.0
    states = {
        'u':
            tf.unstack(
                tf.convert_to_tensor(
                    6.0 * hollow_ones, dtype=_TF_DTYPE)),
        'v':
            tf.unstack(
                tf.convert_to_tensor(8.0 * hollow_ones, dtype=_TF_DTYPE)),
    }

    bc_new = boundary_condition_utils.dirichlet_ghost_cell_quick(
        bc, states, bc_type)

    bc_expected = {
        'u': [[(halo_exchange.BCType.DIRICHLET, [slice_x(-4.0, 2.0),
                                                 slice_x(1.0)]),
               (halo_exchange.BCType.DIRICHLET, [slice_x(-1.0),
                                                 slice_x(-8.0, -2.0)])],
              [(halo_exchange.BCType.NEUMANN, 1.0),
               (halo_exchange.BCType.NEUMANN, -1.0)],
              [(halo_exchange.BCType.DIRICHLET, [slice_z(2.0),
                                                 slice_z(2.0)]),
               (halo_exchange.BCType.DIRICHLET, [slice_z(-2.0),
                                                 slice_z(-10.0, -4.0)])]],
        'v': [[(halo_exchange.BCType.NEUMANN, 1.0),
               (halo_exchange.BCType.NEUMANN, -1.0)],
              [(halo_exchange.BCType.DIRICHLET, [slice_y(-6.0, 2.0),
                                                 slice_y(1.0)]),
               (halo_exchange.BCType.DIRICHLET, [slice_y(-1.0),
                                                 slice_y(-10.0, -2.0)])],
              [None, None]],
    }

    self.assertSequenceEqual(bc_expected.keys(), bc_new.keys())
    for key, val in bc_expected.items():
      for dim in range(3):
        for face in range(2):
          with self.subTest(name='{}Dim{}Face{}'.format(key, dim, face)):
            if val[dim][face] is None:
              self.assertIsNone(bc_new[key][dim][face])
            elif isinstance(val[dim][face][1], float):
              self.assertSequenceAlmostEqual(val[dim][face],
                                             bc_new[key][dim][face])
            else:
              self.assertEqual(val[dim][face][0], bc_new[key][dim][face][0])
              self.assertLen(bc_new[key][dim][face][1], halo_width)
              for i in range(halo_width):
                self.assertAllClose(val[dim][face][1][i],
                                    bc_new[key][dim][face][1][i])


if __name__ == '__main__':
  tf.test.main()

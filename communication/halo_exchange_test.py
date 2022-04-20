"""Tests for Halo Exchange."""
# pylint: disable=bad-whitespace

import functools
import numpy as np
from six.moves import range
from swirl_lm.communication import halo_exchange
from swirl_lm.communication.halo_exchange import SideType
import tensorflow as tf

from google3.research.simulation.tensorflow.fluid.framework.tpu_runner import TpuRunner
from google3.testing.pybase import parameterized


class HaloExchangeTest(tf.test.TestCase):

  def do_halo_exchange(self, xysize, **kwargs):
    strategy = TpuRunner(replicas=[[0], [1]])
    ones = np.ones([xysize, xysize])
    # Create 2 sub-grids of shape (xysize * xysize * 4), adjacent in the
    # indicated dimension in each test,
    # e.g. for dim=[2] the 8 z-slices are in 2 groups of 4, one on each replica
    # See subgrid 0 below, with its 4 z-slices. Subgrid 1 has all values *11
    #
    #    .
    #   .
    #  4 4 ....
    # 4 4 .....
    #
    #    .
    #   .
    #  3 3 ....
    # 3 3 .....
    #
    #    .
    #   .
    #  2 2 ....
    # 2 2 .....
    #
    #    .
    #   .
    #  1 1 ....
    # 1 1 .....
    self.values = [[ones * 1, ones * 2, ones * 3, ones * 4],
                   [ones * 11, ones * 22, ones * 33, ones * 44]]
    output = strategy.run_with_replica_args(
        functools.partial(halo_exchange.inplace_halo_exchange, **kwargs),
        self.values)
    return output[0], output[1]

  def test3DInplaceExchangeWithDim2(self):
    xs, ys = self.do_halo_exchange(xysize=2, dims=[2], replica_dims=[0])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0, 0],  #
                [0, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [0, 0],  #
                [0, 0],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2DirichletNeumann(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.5),
                              (halo_exchange.BCType.NEUMANN, 0.8)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0.5, 0.5],  #
                [0.5, 0.5],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [33.8, 33.8],  #
                [33.8, 33.8],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2DirichletNoTouch(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.5),
                              (halo_exchange.BCType.NO_TOUCH, 0.0)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0.5, 0.5],  #
                [0.5, 0.5],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [44, 44],  #
                [44, 44],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2DirichletAdditive(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.5),
                              (halo_exchange.BCType.ADDITIVE, 4.2)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0.5, 0.5],  #
                [0.5, 0.5],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [48.2, 48.2],  #
                [48.2, 48.2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2NeumannDirichlet(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.NEUMANN, 0.1),
                              (halo_exchange.BCType.DIRICHLET, 0.8)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [1.9, 1.9],  #
                [1.9, 1.9],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [0.8, 0.8],  #
                [0.8, 0.8],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2NoneDirichlet(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[None, (halo_exchange.BCType.DIRICHLET, 0.8)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0, 0],  #
                [0, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [0.8, 0.8],  #
                [0.8, 0.8],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2NoTouchNone(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.NO_TOUCH, 0.0), None]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [1, 1],  #
                [1, 1],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [0, 0],  #
                [0, 0],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2AdditiveNone(self):
    xs, ys = self.do_halo_exchange(
        xysize=2,
        dims=[2],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.ADDITIVE, 4.2), None]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [5.2, 5.2],  #
                [5.2, 5.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [0, 0],  #
                [0, 0],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim2Periodic(self):
    xs, ys = self.do_halo_exchange(
        xysize=2, dims=[2], replica_dims=[0], periodic_dims=[True])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 3],  #
                [3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 22],  #
                [22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 33],  #
                [33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [2, 2],  #
                [2, 2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim0(self):
    xs, ys = self.do_halo_exchange(xysize=4, dims=[0], replica_dims=[0])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0, 0, 0, 0],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [0, 0, 0, 0],  #
                [2, 2, 2, 2],  #
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [0, 0, 0, 0],  #
                [3, 3, 3, 3],  #
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [0, 0, 0, 0],  #
                [4, 4, 4, 4],  #
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
                [11, 11, 11, 11],  #
                [0, 0, 0, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
                [22, 22, 22, 22],  #
                [0, 0, 0, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
                [33, 33, 33, 33],  #
                [0, 0, 0, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
                [44, 44, 44, 44],  #
                [0, 0, 0, 0],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim0DirichletNeumann(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.1),
                              (halo_exchange.BCType.NEUMANN, 0.2)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [2, 2, 2, 2],  #
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [3, 3, 3, 3],  #
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [4, 4, 4, 4],  #
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
                [11, 11, 11, 11],  #
                [11.2, 11.2, 11.2, 11.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
                [22, 22, 22, 22],  #
                [22.2, 22.2, 22.2, 22.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
                [33, 33, 33, 33],  #
                [33.2, 33.2, 33.2, 33.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
                [44, 44, 44, 44],  #
                [44.2, 44.2, 44.2, 44.2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim0DirichletNoTouch(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.1),
                              (halo_exchange.BCType.NO_TOUCH, 0.0)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [2, 2, 2, 2],  #
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [3, 3, 3, 3],  #
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [4, 4, 4, 4],  #
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
                [11, 11, 11, 11],  #
                [11, 11, 11, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
                [22, 22, 22, 22],  #
                [22, 22, 22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
                [33, 33, 33, 33],  #
                [33, 33, 33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
                [44, 44, 44, 44],  #
                [44, 44, 44, 44],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim0DirichletAdditive(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0],
        replica_dims=[0],
        boundary_conditions=[[(halo_exchange.BCType.DIRICHLET, 0.1),
                              (halo_exchange.BCType.ADDITIVE, 4.2)]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [2, 2, 2, 2],  #
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [3, 3, 3, 3],  #
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [0.1, 0.1, 0.1, 0.1],  #
                [4, 4, 4, 4],  #
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
                [11, 11, 11, 11],  #
                [15.2, 15.2, 15.2, 15.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
                [22, 22, 22, 22],  #
                [26.2, 26.2, 26.2, 26.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
                [33, 33, 33, 33],  #
                [37.2, 37.2, 37.2, 37.2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
                [44, 44, 44, 44],  #
                [48.2, 48.2, 48.2, 48.2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim0Periodic(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0], replica_dims=[0], periodic_dims=[True])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [11, 11, 11, 11],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [22, 22, 22, 22],  #
                [2, 2, 2, 2],  #
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [33, 33, 33, 33],  #
                [3, 3, 3, 3],  #
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [44, 44, 44, 44],  #
                [4, 4, 4, 4],  #
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 1, 1, 1],  #
                [11, 11, 11, 11],  #
                [11, 11, 11, 11],  #
                [1, 1, 1, 1],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 2, 2, 2],  #
                [22, 22, 22, 22],  #
                [22, 22, 22, 22],  #
                [2, 2, 2, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 3, 3, 3],  #
                [33, 33, 33, 33],  #
                [33, 33, 33, 33],  #
                [3, 3, 3, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 4, 4, 4],  #
                [44, 44, 44, 44],  #
                [44, 44, 44, 44],  #
                [4, 4, 4, 4],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim1(self):
    xs, ys = self.do_halo_exchange(xysize=4, dims=[1], replica_dims=[0])
    self.assertAllClose(
        np.array(
            [
                [0, 1, 1, 11],  #
                [0, 1, 1, 11],  #
                [0, 1, 1, 11],  #
                [0, 1, 1, 11],  #
            ],
            dtype=np.float32),
        xs[0])
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [0, 2, 2, 22],  #
                [0, 2, 2, 22],  #
                [0, 2, 2, 22],  #
                [0, 2, 2, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [0, 3, 3, 33],  #
                [0, 3, 3, 33],  #
                [0, 3, 3, 33],  #
                [0, 3, 3, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [0, 4, 4, 44],  #
                [0, 4, 4, 44],  #
                [0, 4, 4, 44],  #
                [0, 4, 4, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 11, 11, 0],  #
                [1, 11, 11, 0],  #
                [1, 11, 11, 0],  #
                [1, 11, 11, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 22, 22, 0],  #
                [2, 22, 22, 0],  #
                [2, 22, 22, 0],  #
                [2, 22, 22, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 33, 33, 0],  #
                [3, 33, 33, 0],  #
                [3, 33, 33, 0],  #
                [3, 33, 33, 0],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 44, 44, 0],  #
                [4, 44, 44, 0],  #
                [4, 44, 44, 0],  #
                [4, 44, 44, 0],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithDim1Periodic(self):
    xs, ys = self.do_halo_exchange(
        xysize=4, dims=[1], replica_dims=[0], periodic_dims=[True])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [11, 1, 1, 11],  #
                [11, 1, 1, 11],  #
                [11, 1, 1, 11],  #
                [11, 1, 1, 11],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [22, 2, 2, 22],  #
                [22, 2, 2, 22],  #
                [22, 2, 2, 22],  #
                [22, 2, 2, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [33, 3, 3, 33],  #
                [33, 3, 3, 33],  #
                [33, 3, 3, 33],  #
                [33, 3, 3, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [44, 4, 4, 44],  #
                [44, 4, 4, 44],  #
                [44, 4, 4, 44],  #
                [44, 4, 4, 44],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [1, 11, 11, 1],  #
                [1, 11, 11, 1],  #
                [1, 11, 11, 1],  #
                [1, 11, 11, 1],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [2, 22, 22, 2],  #
                [2, 22, 22, 2],  #
                [2, 22, 22, 2],  #
                [2, 22, 22, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [3, 33, 33, 3],  #
                [3, 33, 33, 3],  #
                [3, 33, 33, 3],  #
                [3, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [4, 44, 44, 4],  #
                [4, 44, 44, 4],  #
                [4, 44, 44, 4],  #
                [4, 44, 44, 4],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithOneThreePeriodicTwoBC(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0, 1, 2],
        replica_dims=[0, 0, 0],
        periodic_dims=[True, False, True],
        boundary_conditions=[[None, None],
                             [None, (halo_exchange.BCType.NEUMANN, 0.3)],
                             [(halo_exchange.BCType.NEUMANN, 0.1), None]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [33, 3, 3, 3.3],  #
                [3, 33, 33, 33.3],  #
                [3, 33, 33, 33.3],  #
                [33, 3, 3, 3.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [0, 22, 22, 2],  #
                [0, 2, 2, 22],  #
                [0, 2, 2, 22],  #
                [0, 22, 22, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [0, 33, 33, 3],  #
                [0, 3, 3, 33],  #
                [0, 3, 3, 33],  #
                [0, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 2, 2, 2.3],  #
                [2, 22, 22, 22.3],  #
                [2, 22, 22, 22.3],  #
                [22, 2, 2, 2.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [0, 33, 33, 3],  #
                [0, 3, 3, 33],  #
                [0, 3, 3, 33],  #
                [0, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 2, 2, 2.3],  #
                [2, 22, 22, 22.3],  #
                [2, 22, 22, 22.3],  #
                [22, 2, 2, 2.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 3, 3, 3.3],  #
                [3, 33, 33, 33.3],  #
                [3, 33, 33, 33.3],  #
                [33, 3, 3, 3.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [0, 22, 22, 2],  #
                [0, 2, 2, 22],  #
                [0, 2, 2, 22],  #
                [0, 22, 22, 2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithOneThreePeriodicTwoNoTouchNeumann(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0, 1, 2],
        replica_dims=[0, 0, 0],
        periodic_dims=[True, False, True],
        boundary_conditions=[[None, None],
                             [(halo_exchange.BCType.NO_TOUCH, 0.0),
                              (halo_exchange.BCType.NEUMANN, 0.3)],
                             [(halo_exchange.BCType.DIRICHLET, 0.1), None]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [33, 3, 3, 3.3],  #
                [3, 33, 33, 33.3],  #
                [3, 33, 33, 33.3],  #
                [33, 3, 3, 3.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [22, 22, 22, 2],  #
                [2, 2, 2, 22],  #
                [2, 2, 2, 22],  #
                [22, 22, 22, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [33, 33, 33, 3],  #
                [3, 3, 3, 33],  #
                [3, 3, 3, 33],  #
                [33, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 2, 2, 2.3],  #
                [2, 22, 22, 22.3],  #
                [2, 22, 22, 22.3],  #
                [22, 2, 2, 2.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [33, 33, 33, 3],  #
                [3, 3, 3, 33],  #
                [3, 3, 3, 33],  #
                [33, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 2, 2, 2.3],  #
                [2, 22, 22, 22.3],  #
                [2, 22, 22, 22.3],  #
                [22, 2, 2, 2.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 3, 3, 3.3],  #
                [3, 33, 33, 33.3],  #
                [3, 33, 33, 33.3],  #
                [33, 3, 3, 3.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [22, 22, 22, 2],  #
                [2, 2, 2, 22],  #
                [2, 2, 2, 22],  #
                [22, 22, 22, 2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithOneThreePeriodicTwoAdditiveNeumann(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0, 1, 2],
        replica_dims=[0, 0, 0],
        periodic_dims=[True, False, True],
        boundary_conditions=[[None, None],
                             [(halo_exchange.BCType.ADDITIVE, 4.2),
                              (halo_exchange.BCType.NEUMANN, 0.3)],
                             [(halo_exchange.BCType.DIRICHLET, 0.1), None]])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [33, 3, 3, 3.3],  #
                [3, 33, 33, 33.3],  #
                [3, 33, 33, 33.3],  #
                [33, 3, 3, 3.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [26.2, 22, 22, 2],  #
                [6.2, 2, 2, 22],  #
                [6.2, 2, 2, 22],  #
                [26.2, 22, 22, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [37.2, 33, 33, 3],  #
                [7.2, 3, 3, 33],  #
                [7.2, 3, 3, 33],  #
                [37.2, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 2, 2, 2.3],  #
                [2, 22, 22, 22.3],  #
                [2, 22, 22, 22.3],  #
                [22, 2, 2, 2.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [37.2, 33, 33, 3],  #
                [7.2, 3, 3, 33],  #
                [7.2, 3, 3, 33],  #
                [37.2, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 2, 2, 2.3],  #
                [2, 22, 22, 22.3],  #
                [2, 22, 22, 22.3],  #
                [22, 2, 2, 2.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 3, 3, 3.3],  #
                [3, 33, 33, 33.3],  #
                [3, 33, 33, 33.3],  #
                [33, 3, 3, 3.3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [26.2, 22, 22, 2],  #
                [6.2, 2, 2, 22],  #
                [6.2, 2, 2, 22],  #
                [26.2, 22, 22, 2],  #
            ],
            dtype=np.float32))

  def test3DInplaceExchangeWithAllTorus(self):
    xs, ys = self.do_halo_exchange(
        xysize=4,
        dims=[0, 1, 2],
        replica_dims=[0, 0, 0],
        periodic_dims=[True, True, True])
    self.assertAllClose(
        xs[0],
        np.array(
            [
                [33, 3, 3, 33],  #
                [3, 33, 33, 3],  #
                [3, 33, 33, 3],  #
                [33, 3, 3, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[1],
        np.array(
            [
                [2, 22, 22, 2],  #
                [22, 2, 2, 22],  #
                [22, 2, 2, 22],  #
                [2, 22, 22, 2],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[2],
        np.array(
            [
                [3, 33, 33, 3],  #
                [33, 3, 3, 33],  #
                [33, 3, 3, 33],  #
                [3, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        xs[3],
        np.array(
            [
                [22, 2, 2, 22],  #
                [2, 22, 22, 2],  #
                [2, 22, 22, 2],  #
                [22, 2, 2, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[0],
        np.array(
            [
                [3, 33, 33, 3],  #
                [33, 3, 3, 33],  #
                [33, 3, 3, 33],  #
                [3, 33, 33, 3],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[1],
        np.array(
            [
                [22, 2, 2, 22],  #
                [2, 22, 22, 2],  #
                [2, 22, 22, 2],  #
                [22, 2, 2, 22],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[2],
        np.array(
            [
                [33, 3, 3, 33],  #
                [3, 33, 33, 3],  #
                [3, 33, 33, 3],  #
                [33, 3, 3, 33],  #
            ],
            dtype=np.float32))
    self.assertAllClose(
        ys[3],
        np.array(
            [
                [2, 22, 22, 2],  #
                [22, 2, 2, 22],  #
                [22, 2, 2, 22],  #
                [2, 22, 22, 2],  #
            ],
            dtype=np.float32))


def list_of_random_tensors(n):
  tensor_list = []
  for _ in range(n):
    tensor_list.append(np.random.rand(n, n).astype(np.float32))
  return tensor_list


# TODO(b/142672777): Reconcile the various halo exchange and boundary condition
# test classes, and in particular, coverage for periodic boundary condition
# cases.
class GridBcHaloExchangeTest(tf.test.TestCase):

  def setUp(self):
    super(GridBcHaloExchangeTest, self).setUp()
    self.strategy = TpuRunner(replicas=[[0], [1]])
    self.grid_size = 6  # Test works for any size > 2 * width.
    self.xlist1 = list_of_random_tensors(self.grid_size)
    self.xlist2 = list_of_random_tensors(self.grid_size)
    self.bclist = list_of_random_tensors(self.grid_size)
    self.zeros = np.zeros((self.grid_size, self.grid_size))
    self.expected1 = np.copy(self.xlist1)
    self.expected2 = np.copy(self.xlist2)

  @staticmethod
  def get_boundary_conditions(bcs, bc_key):
    boundary_conditions = {
        'dirichlet_neumann': [[(halo_exchange.BCType.DIRICHLET, bcs[0]),
                               (halo_exchange.BCType.NEUMANN, bcs[1])]],
        'neumann_dirichlet': [[(halo_exchange.BCType.NEUMANN, bcs[0]),
                               (halo_exchange.BCType.DIRICHLET, bcs[1])]],
        'no_touch_dirichlet': [[(halo_exchange.BCType.NO_TOUCH, bcs[0]),
                                (halo_exchange.BCType.DIRICHLET, bcs[1])]],
        'neumann_no_touch': [[(halo_exchange.BCType.NEUMANN, bcs[0]),
                              (halo_exchange.BCType.NO_TOUCH, bcs[1])]],
        'additive_dirichlet': [[(halo_exchange.BCType.ADDITIVE, bcs[0]),
                                (halo_exchange.BCType.DIRICHLET, bcs[1])]],
        'neumann_additive': [[(halo_exchange.BCType.NEUMANN, bcs[0]),
                              (halo_exchange.BCType.ADDITIVE, bcs[1])]]
    }
    return boundary_conditions[bc_key]

  def do_halo_exchange(self,
                       dim,
                       periodic,
                       set_bcs,
                       width=1,
                       bc_key=None):
    periodic_dims = [periodic]
    if set_bcs:
      bclist = [tf.convert_to_tensor(a) for a in self.bclist]
      bcs = [
          halo_exchange.get_edge_of_3d_field(bclist, dim, SideType.LOW, width),
          halo_exchange.get_edge_of_3d_field(bclist, dim, SideType.HIGH, width)
      ]
      boundary_conditions = self.get_boundary_conditions(bcs, bc_key)
    else:
      boundary_conditions = None

    output = self.strategy.run_with_replica_args(
        functools.partial(
            halo_exchange.inplace_halo_exchange,
            dims=[dim],
            replica_dims=[0],
            periodic_dims=periodic_dims,
            boundary_conditions=boundary_conditions,
            width=width),
        [self.xlist1, self.xlist2])
    return output[0], output[1]

  def testGridBcHaloExchangePeriodicBcsWidth1X(self):
    dim = 0
    periodic = True
    set_bcs = True  # BCs are set but have no effect.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = self.xlist2[iz][-2, :]
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = self.xlist1[iz][1, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicNoBcsWidth1X(self):
    dim = 0
    periodic = True
    set_bcs = False
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = self.xlist2[iz][-2, :]
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = self.xlist1[iz][1, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeDnBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = True
    # Left is Dirichlet, right is Neumann.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = self.bclist[iz][0, :]
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = (
          self.xlist2[iz][-2, :] + self.bclist[iz][-1, :])
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNdBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 1
    # Left is Neumann, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = (self.xlist1[iz][1, :] - self.bclist[iz][0, :])
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = self.bclist[iz][-1, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNtdBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 1
    # Left is No Touch, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = self.xlist1[iz][0, :]
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = self.bclist[iz][-1, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='no_touch_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNntBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 1
    # Left is Neumann, right is No Touch.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = (self.xlist1[iz][1, :] - self.bclist[iz][0, :])
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_no_touch')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeAdBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 1
    # Left is Additive, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] += self.bclist[iz][0, :]
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = self.bclist[iz][-1, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='additive_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNaBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 1
    # Left is Neumann, right is Additive.
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = (self.xlist1[iz][1, :] - self.bclist[iz][0, :])
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] += self.bclist[iz][-1, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_additive')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNonPeriodicNoBcsWidth1X(self):
    dim = 0
    periodic = False
    set_bcs = False
    for iz in range(self.grid_size):
      self.expected1[iz][0, :] = 0
      self.expected1[iz][-1, :] = self.xlist2[iz][1, :]
      self.expected2[iz][0, :] = self.xlist1[iz][-2, :]
      self.expected2[iz][-1, :] = 0
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicBcsWidth2X(self):
    dim = 0
    periodic = True
    set_bcs = True  # BCs are set but have no effect.
    width = 2
    for iz in range(self.grid_size):
      self.expected1[iz][:2, :] = self.xlist2[iz][-4:-2, :]
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] = self.xlist1[iz][2:4, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicNoBcsWidth2X(self):
    dim = 0
    periodic = True
    set_bcs = False
    width = 2
    for iz in range(self.grid_size):
      self.expected1[iz][:2, :] = self.xlist2[iz][-4:-2, :]
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] = self.xlist1[iz][2:4, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeDnBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 2
    # Left is Dirichlet, right is Neumann.
    for iz in range(self.grid_size):
      self.expected1[iz][:2, :] = self.bclist[iz][:2, :]
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2, :] = (
          self.expected2[iz][-3, :] + self.bclist[iz][-2, :])
      self.expected2[iz][-1, :] = (
          self.expected2[iz][-2, :] + self.bclist[iz][-1, :])
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNdBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 2
    # Left is Neumann, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][1, :] = (
          self.expected1[iz][2, :] - self.bclist[iz][1, :])
      self.expected1[iz][0, :] = (
          self.expected1[iz][1, :] - self.bclist[iz][0, :])
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] = self.bclist[iz][-2:, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNtdBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 2
    # Left is No Touch, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][1, :] = self.expected1[iz][1, :]
      self.expected1[iz][0, :] = self.expected1[iz][0, :]
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] = self.bclist[iz][-2:, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='no_touch_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNntBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 2
    # Left is Neumann, right is No Touch.
    for iz in range(self.grid_size):
      self.expected1[iz][1, :] = (
          self.expected1[iz][2, :] - self.bclist[iz][1, :])
      self.expected1[iz][0, :] = (
          self.expected1[iz][1, :] - self.bclist[iz][0, :])
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_no_touch')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeAdBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 2
    # Left is Additive, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:2, :] += self.bclist[iz][:2, :]
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] = self.bclist[iz][-2:, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='additive_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNaBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = True
    width = 2
    # Left is Neumann, right is Additive.
    for iz in range(self.grid_size):
      self.expected1[iz][1, :] = (
          self.expected1[iz][2, :] - self.bclist[iz][1, :])
      self.expected1[iz][0, :] = (
          self.expected1[iz][1, :] - self.bclist[iz][0, :])
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] += self.bclist[iz][-2:, :]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_additive')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNonPeriodicNoBcsWidth2X(self):
    dim = 0
    periodic = False
    set_bcs = False
    width = 2
    for iz in range(self.grid_size):
      self.expected1[iz][:2, :] = 0
      self.expected1[iz][-2:, :] = self.xlist2[iz][2:4, :]
      self.expected2[iz][:2, :] = self.xlist1[iz][-4:-2, :]
      self.expected2[iz][-2:, :] = 0
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicBcsWidth1Y(self):
    dim = 1
    periodic = True
    set_bcs = True  # BCs are set but have no effect.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = self.xlist2[iz][:, -2]
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] = self.xlist1[iz][:, 1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicNoBcsWidth1Y(self):
    dim = 1
    periodic = True
    set_bcs = False
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = self.xlist2[iz][:, -2]
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] = self.xlist1[iz][:, 1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeDnBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    # Left is Dirichlet, right is Neumann.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = self.bclist[iz][:, 0]
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[
          iz][:, -1] = self.expected2[iz][:, -2] + self.bclist[iz][:, -1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNdBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 1
    # Left is Neumann, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = (self.xlist1[iz][:, 1] - self.bclist[iz][:, 0])
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] = self.bclist[iz][:, -1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNtdBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 1
    # Left is No Touch, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = self.xlist1[iz][:, 0]
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] = self.bclist[iz][:, -1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='no_touch_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNntBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 1
    # Left is Neumann, right is No Touch.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = (self.xlist1[iz][:, 1] - self.bclist[iz][:, 0])
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_no_touch')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeAdBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 1
    # Left is Additive, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] += self.bclist[iz][:, 0]
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] = self.bclist[iz][:, -1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='additive_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNaBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 1
    # Left is Neumann, right is Additive.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = (self.xlist1[iz][:, 1] - self.bclist[iz][:, 0])
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] += self.bclist[iz][:, -1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_additive')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNonPeriodicNoBcsWidth1Y(self):
    dim = 1
    periodic = False
    set_bcs = False
    for iz in range(self.grid_size):
      self.expected1[iz][:, 0] = 0
      self.expected1[iz][:, -1] = self.xlist2[iz][:, 1]
      self.expected2[iz][:, 0] = self.xlist1[iz][:, -2]
      self.expected2[iz][:, -1] = 0
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicBcsWidth2Y(self):
    dim = 1
    periodic = True
    set_bcs = True  # BCs are set but have no effect.
    width = 2
    for iz in range(self.grid_size):
      self.expected1[iz][:, :2] = self.xlist2[iz][:, -4:-2]
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] = self.xlist1[iz][:, 2:4]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicNoBcsWidth2Y(self):
    dim = 1
    periodic = True
    set_bcs = False
    width = 2
    for iz in range(self.grid_size):
      self.expected1[iz][:, :2] = self.xlist2[iz][:, -4:-2]
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] = self.xlist1[iz][:, 2:4]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeDnBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 2
    # Left is Dirichlet, right is Neumann.
    for iz in range(self.grid_size):
      self.expected1[iz][:, :2] = self.bclist[iz][:, :2]
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2] = (
          self.xlist2[iz][:, -3] + self.bclist[iz][:, -2])
      self.expected2[iz][:, -1] = (
          self.expected2[iz][:, -2] + self.bclist[iz][:, -1])
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNdBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 2
    # Left is Neumann, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 1] = (self.xlist1[iz][:, 2] - self.bclist[iz][:, 1])
      self.expected1[iz][:, 0] = (
          self.expected1[iz][:, 1] - self.bclist[iz][:, 0])
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] = self.bclist[iz][:, -2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNtdBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 2
    # Left is No Touch, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] = self.bclist[iz][:, -2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='no_touch_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNntBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 2
    # Left is Neumann, right is No Touch.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 1] = (self.xlist1[iz][:, 2] - self.bclist[iz][:, 1])
      self.expected1[iz][:, 0] = (
          self.expected1[iz][:, 1] - self.bclist[iz][:, 0])
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_no_touch')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeAdBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 2
    # Left is Additive, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected1[iz][:, :2] += self.bclist[iz][:, :2]
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] = self.bclist[iz][:, -2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='additive_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNaBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = True
    width = 2
    # Left is Neumann, right is Additive.
    for iz in range(self.grid_size):
      self.expected1[iz][:, 1] = (self.xlist1[iz][:, 2] - self.bclist[iz][:, 1])
      self.expected1[iz][:, 0] = (
          self.expected1[iz][:, 1] - self.bclist[iz][:, 0])
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] += self.bclist[iz][:, -2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_additive')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNonPeriodicNoBcsWidth2Y(self):
    dim = 1
    periodic = False
    set_bcs = False
    width = 2
    for iz in range(self.grid_size):
      self.expected1[iz][:, :2] = 0
      self.expected1[iz][:, -2:] = self.xlist2[iz][:, 2:4]
      self.expected2[iz][:, :2] = self.xlist1[iz][:, -4:-2]
      self.expected2[iz][:, -2:] = 0
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicBcsWidth1Z(self):
    dim = 2
    periodic = True
    set_bcs = True  # BCs are set but have no effect.
    self.expected1[0] = self.xlist2[-2]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.xlist1[1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicNoBcsWidth1Z(self):
    dim = 2
    periodic = True
    set_bcs = False
    self.expected1[0] = self.xlist2[-2]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.xlist1[1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeDnBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    # Bottom is Dirichlet, top is Neumann.
    self.expected1[0] = self.bclist[0]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.expected2[-2] + self.bclist[-1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNdBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 1
    # Bottom is Neumann, top is Dirichlet.
    self.expected1[0] = self.xlist1[1] - self.bclist[0]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.bclist[-1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNtdBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 1
    # Bottom is No Touch, top is Dirichlet.
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.bclist[-1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='no_touch_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNntBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 1
    # Bottom is Neumann, top is No Touch.
    self.expected1[0] = self.xlist1[1] - self.bclist[0]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_no_touch')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeAdBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 1
    # Bottom is Additive, top is Dirichlet.
    self.expected1[0] += self.bclist[0]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.bclist[-1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='additive_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNaBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 1
    # Bottom is Neumann, top is Additive.
    self.expected1[0] = self.xlist1[1] - self.bclist[0]
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] += self.bclist[-1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_additive')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNonPeriodicNoBcsWidth1Z(self):
    dim = 2
    periodic = False
    set_bcs = False
    self.expected1[0] = self.zeros
    self.expected1[-1] = self.xlist2[1]
    self.expected2[0] = self.xlist1[-2]
    self.expected2[-1] = self.zeros
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicBcsWidth2Z(self):
    dim = 2
    periodic = True
    set_bcs = True  # BCs are set but have no effect.
    width = 2
    self.expected1[:2] = self.xlist2[-4:-2]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] = self.xlist1[2:4]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangePeriodicNoBcsWidth2Z(self):
    dim = 2
    periodic = True
    set_bcs = False
    width = 2
    self.expected1[:2] = self.xlist2[-4:-2]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] = self.xlist1[2:4]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeDnBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 2
    # Bottom is Dirichlet, top is Neumann.
    self.expected1[:2] = self.bclist[:2]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2] = self.expected2[-3] + self.bclist[-2]
    self.expected2[-1] = self.expected2[-2] + self.bclist[-1]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='dirichlet_neumann')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNdBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 2
    # Bottom is Neumann, top is Dirichlet.
    self.expected1[1] = self.expected1[2] - self.bclist[1]
    self.expected1[0] = self.expected1[1] - self.bclist[0]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] = self.bclist[-2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNtdBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 2
    # Bottom is No Touch, top is Dirichlet.
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] = self.bclist[-2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='no_touch_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNntBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 2
    # Bottom is Neumann, top is No Touch.
    self.expected1[1] = self.expected1[2] - self.bclist[1]
    self.expected1[0] = self.expected1[1] - self.bclist[0]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_no_touch')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeAdBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 2
    # Bottom is Additive, top is Dirichlet.
    self.expected1[:2] += self.bclist[:2]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] = self.bclist[-2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='additive_dirichlet')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNaBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = True
    width = 2
    # Bottom is Neumann, top is Additive.
    self.expected1[1] = self.expected1[2] - self.bclist[1]
    self.expected1[0] = self.expected1[1] - self.bclist[0]
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] += self.bclist[-2:]
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width,
                                   bc_key='neumann_additive')
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)

  def testGridBcHaloExchangeNonPeriodicNoBcsWidth2Z(self):
    dim = 2
    periodic = False
    set_bcs = False
    width = 2
    self.expected1[:2] = [self.zeros] * 2
    self.expected1[-2:] = self.xlist2[2:4]
    self.expected2[:2] = self.xlist1[-4:-2]
    self.expected2[-2:] = [self.zeros] * 2
    x1, x2 = self.do_halo_exchange(dim, periodic, set_bcs, width)
    self.assertAllClose(self.expected1, x1)
    self.assertAllClose(self.expected2, x2)


class ExplicitSingleCoreHaloExchangeTest(tf.test.TestCase):

  def run_test_with_single_replica_dims(self,
                                        dims,
                                        replica_dims,
                                        periodic_dims,
                                        boundary_conditions,
                                        expected,
                                        dual_core=True):
    xlist = [[
        # a0 is [[111, 112, 113, 114],
        #        [121, 122, 123, 124],
        #        [131, 132, 133, 134],
        #        [141, 142, 143, 144]]
        #
        # b0 is [[211, 212, 213, 214],
        #        [221, 222, 223, 224],
        #         .................
        # and so on.
        [[110 + i for i in range(1, 5)], [120 + i for i in range(1, 5)],
         [130 + i for i in range(1, 5)], [140 + i for i in range(1, 5)]],
        [[210 + i for i in range(1, 5)], [220 + i for i in range(1, 5)],
         [230 + i for i in range(1, 5)], [240 + i for i in range(1, 5)]],
        [[310 + i for i in range(1, 5)], [320 + i for i in range(1, 5)],
         [330 + i for i in range(1, 5)], [340 + i for i in range(1, 5)]],
        [[410 + i for i in range(1, 5)], [420 + i for i in range(1, 5)],
         [430 + i for i in range(1, 5)], [440 + i for i in range(1, 5)]]
    ]]
    if dual_core:
      xlist.append([
          # a1 is [[1111, 1112, 1113, 1114],
          #        [1121, 1122, 1123, 1124],
          #        [1131, 1132, 1133, 1134],
          #        [1141, 1142, 1143, 1144]]
          #
          # b1 is [[1211, 1212, 1213, 1214],
          #        [1221, 1222, 1223, 1224],
          #         .................
          # and so on.
          [[1110 + i for i in range(1, 5)], [1120 + i for i in range(1, 5)],
           [1130 + i for i in range(1, 5)], [1140 + i for i in range(1, 5)]],
          [[1210 + i for i in range(1, 5)], [1220 + i for i in range(1, 5)],
           [1230 + i for i in range(1, 5)], [1240 + i for i in range(1, 5)]],
          [[1310 + i for i in range(1, 5)], [1320 + i for i in range(1, 5)],
           [1330 + i for i in range(1, 5)], [1340 + i for i in range(1, 5)]],
          [[1410 + i for i in range(1, 5)], [1420 + i for i in range(1, 5)],
           [1430 + i for i in range(1, 5)], [1440 + i for i in range(1, 5)]]
      ])
    # Group the list levels beyond the first 2 into tensors, to avoid confusion
    # with a 4-level list of scalars. The first two levels are: by replica;
    # z-slices for a given replica.
    xlist_t = [[tf.convert_to_tensor(a, dtype=tf.float32)
                for a in l] for l in xlist]
    replicas = [[[[0]]], [[[1]]]] if dual_core else [[[[0]]]]
    strategy = TpuRunner(replicas=replicas)
    output = strategy.run_with_replica_args(
        functools.partial(
            halo_exchange.inplace_halo_exchange,
            dims=dims,
            replica_dims=replica_dims,
            periodic_dims=periodic_dims,
            boundary_conditions=boundary_conditions),
        xlist_t)

    self.assertAllClose(output[0][0], expected[0][0])
    self.assertAllClose(output[0][1], expected[0][1])
    self.assertAllClose(output[0][2], expected[0][2])
    self.assertAllClose(output[0][3], expected[0][3])
    if dual_core:
      self.assertAllClose(output[1][0], expected[1][0])
      self.assertAllClose(output[1][1], expected[1][1])
      self.assertAllClose(output[1][2], expected[1][2])
      self.assertAllClose(output[1][3], expected[1][3])

  def test2DInplaceExchangeWithSingleReplicaDim(self):
    self.run_test_with_single_replica_dims(
        dims=[0, 2],
        replica_dims=[0, 2],
        periodic_dims=[True, True],
        boundary_conditions=[None, None],
        expected=[
            [
                # pylint: disable=bad-whitespace
                np.array(
                    [
                        [1331, 1332, 1333, 1334],
                        [ 321,  322,  323,  324],
                        [ 331,  332,  333,  334],
                        [1321, 1322, 1323, 1324],
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        [1231, 1232, 1233, 1234],
                        [ 221,  222,  223,  224],
                        [ 231,  232,  233,  234],
                        [1221, 1222, 1223, 1224],
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        [1331, 1332, 1333, 1334],
                        [ 321,  322,  323,  324],
                        [ 331,  332,  333,  334],
                        [1321, 1322, 1323, 1324],
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        [1231, 1232, 1233, 1234],
                        [ 221,  222,  223,  224],
                        [ 231,  232,  233,  234],
                        [1221, 1222, 1223, 1224],
                    ],
                    dtype=np.float32)
                # pylint: enable=bad-whitespace
            ],
            [
                # pylint: disable=bad-whitespace
                np.array(
                    [
                        [ 331,  332,  333,  334],
                        [1321, 1322, 1323, 1324],
                        [1331, 1332, 1333, 1334],
                        [ 321,  322,  323,  324],
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        [ 231,  232,  233,  234],
                        [1221, 1222, 1223, 1224],
                        [1231, 1232, 1233, 1234],
                        [ 221,  222,  223,  224],
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        [ 331,  332,  333,  334],
                        [1321, 1322, 1323, 1324],
                        [1331, 1332, 1333, 1334],
                        [ 321,  322,  323,  324],
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        [ 231,  232,  233,  234],
                        [1221, 1222, 1223, 1224],
                        [1231, 1232, 1233, 1234],
                        [ 221,  222,  223,  224],
                    ],
                    dtype=np.float32)
                # pylint: enable=bad-whitespace
            ]])

  def test3DInplaceExchangeWithSingleReplicaDimXZPeriodic(self):
    self.run_test_with_single_replica_dims(
        dims=[0, 1, 2],
        replica_dims=[0, 1, 2],
        periodic_dims=[True, False, True],
        boundary_conditions=[
            None,
            [(halo_exchange.BCType.DIRICHLET, 0.),
             (halo_exchange.BCType.NEUMANN, 2.)], None
        ],
        expected=[
            [
                # pylint: disable=bad-whitespace
                np.array([
                    [0, 1332, 1333, 1335],
                    [0, 322, 323, 325],
                    [0, 332, 333, 335],
                    [0, 1322, 1323, 1325],
                ],
                         dtype=np.float32),
                np.array([
                    [0, 1232, 1233, 1235],
                    [0, 222, 223, 225],
                    [0, 232, 233, 235],
                    [0, 1222, 1223, 1225],
                ],
                         dtype=np.float32),
                np.array([
                    [0, 1332, 1333, 1335],
                    [0, 322, 323, 325],
                    [0, 332, 333, 335],
                    [0, 1322, 1323, 1325],
                ],
                         dtype=np.float32),
                np.array([
                    [0, 1232, 1233, 1235],
                    [0, 222, 223, 225],
                    [0, 232, 233, 235],
                    [0, 1222, 1223, 1225],
                ],
                         dtype=np.float32)
                # pylint: enable=bad-whitespace
            ],
            [
                # pylint: disable=bad-whitespace
                np.array([
                    [0, 332, 333, 335],
                    [0, 1322, 1323, 1325],
                    [0, 1332, 1333, 1335],
                    [0, 322, 323, 325],
                ],
                         dtype=np.float32),
                np.array([
                    [0, 232, 233, 235],
                    [0, 1222, 1223, 1225],
                    [0, 1232, 1233, 1235],
                    [0, 222, 223, 225],
                ],
                         dtype=np.float32),
                np.array([
                    [0, 332, 333, 335],
                    [0, 1322, 1323, 1325],
                    [0, 1332, 1333, 1335],
                    [0, 322, 323, 325],
                ],
                         dtype=np.float32),
                np.array([
                    [0, 232, 233, 235],
                    [0, 1222, 1223, 1225],
                    [0, 1232, 1233, 1235],
                    [0, 222, 223, 225],
                ],
                         dtype=np.float32)
                # pylint: enable=bad-whitespace
            ]
        ])

  def test3DInplaceExchangeWithSingleReplicaDimXYPeriodic(self):
    self.run_test_with_single_replica_dims(
        dims=[0, 1, 2],
        replica_dims=[0, 1, 2],
        periodic_dims=[True, True, False],
        boundary_conditions=[
            None, None,
            [(halo_exchange.BCType.DIRICHLET, 0.),
             (halo_exchange.BCType.NEUMANN, 2.)]
        ],
        expected=[
            [
                np.zeros((4, 4)),
                # pylint: disable=bad-whitespace
                np.array([
                    [1233, 1232, 1233, 1232],
                    [223, 222, 223, 222],
                    [233, 232, 233, 232],
                    [1223, 1222, 1223, 1222],
                ],
                         dtype=np.float32),
                np.array([
                    [1333, 1332, 1333, 1332],
                    [323, 322, 323, 322],
                    [333, 332, 333, 332],
                    [1323, 1322, 1323, 1322],
                ],
                         dtype=np.float32),
                np.array([
                    [1335, 1334, 1335, 1334],
                    [325, 324, 325, 324],
                    [335, 334, 335, 334],
                    [1325, 1324, 1325, 1324],
                ],
                         dtype=np.float32)
                # pylint: enable=bad-whitespace
            ],
            [
                np.zeros((4, 4)),
                # pylint: disable=bad-whitespace
                np.array([
                    [233, 232, 233, 232],
                    [1223, 1222, 1223, 1222],
                    [1233, 1232, 1233, 1232],
                    [223, 222, 223, 222],
                ],
                         dtype=np.float32),
                np.array([
                    [333, 332, 333, 332],
                    [1323, 1322, 1323, 1322],
                    [1333, 1332, 1333, 1332],
                    [323, 322, 323, 322],
                ],
                         dtype=np.float32),
                np.array([
                    [335, 334, 335, 334],
                    [1325, 1324, 1325, 1324],
                    [1335, 1334, 1335, 1334],
                    [325, 324, 325, 324],
                ],
                         dtype=np.float32)
                # pylint: enable=bad-whitespace
            ]
        ])

  def test3DInplaceExchangeWithSingleCoreDimXZPeriodic(self):
    self.run_test_with_single_replica_dims(
        dims=[0, 1, 2],
        replica_dims=[0, 1, 2],
        periodic_dims=[True, False, True],
        boundary_conditions=[
            None,
            [(halo_exchange.BCType.DIRICHLET, 0.),
             (halo_exchange.BCType.NEUMANN, 2.)], None
        ],
        expected=[[
            np.array([
                [0, 332, 333, 335],
                [0, 322, 323, 325],
                [0, 332, 333, 335],
                [0, 322, 323, 325],
            ],
                     dtype=np.float32),
            np.array([
                [0, 232, 233, 235],
                [0, 222, 223, 225],
                [0, 232, 233, 235],
                [0, 222, 223, 225],
            ],
                     dtype=np.float32),
            np.array([
                [0, 332, 333, 335],
                [0, 322, 323, 325],
                [0, 332, 333, 335],
                [0, 322, 323, 325],
            ],
                     dtype=np.float32),
            np.array([
                [0, 232, 233, 235],
                [0, 222, 223, 225],
                [0, 232, 233, 235],
                [0, 222, 223, 225],
            ],
                     dtype=np.float32)
            # pylint: enable=bad-whitespace
        ]],
        dual_core=False)


class ParameterizedSingleCoreHaloExchangeTest(tf.test.TestCase,
                                              parameterized.TestCase):

  def setUp(self):
    super(ParameterizedSingleCoreHaloExchangeTest, self).setUp()
    self.grid_size = 16  # Test works for any size > 2 * width.
    self.xlist = list_of_random_tensors(self.grid_size)
    self.bclist = list_of_random_tensors(self.grid_size)
    self.zeros = np.zeros((self.grid_size, self.grid_size))
    self.expected = np.copy(self.xlist)

  def do_halo_exchange(self,
                       dim,
                       periodic,
                       set_bcs,
                       width,
                       dirichlet_neumann=True):

    periodic_dims = [periodic]
    if set_bcs:
      bclist = [tf.convert_to_tensor(a) for a in self.bclist]
      bcs = [
          halo_exchange.get_edge_of_3d_field(bclist, dim, SideType.LOW, width),
          halo_exchange.get_edge_of_3d_field(bclist, dim, SideType.HIGH, width)
      ]
      if dirichlet_neumann:
        boundary_conditions = [[(halo_exchange.BCType.DIRICHLET, bcs[0]),
                                (halo_exchange.BCType.NEUMANN, bcs[1])]]
      else:
        boundary_conditions = [[(halo_exchange.BCType.NEUMANN, bcs[0]),
                                (halo_exchange.BCType.DIRICHLET, bcs[1])]]
    else:
      boundary_conditions = None

    strategy = TpuRunner(replicas=[0])
    output = strategy.run_with_replica_args(
        functools.partial(
            halo_exchange.inplace_halo_exchange,
            dims=[dim],
            replica_dims=[0],
            periodic_dims=periodic_dims,
            boundary_conditions=boundary_conditions,
            width=width),
        [self.xlist])
    return output[0]

  def singleCoreHaloExchangePeriodicX(self, set_bcs, width):
    dim = 0
    periodic = True
    for iz in range(self.grid_size):
      self.expected[iz][:width, :] = self.xlist[iz][-2 * width:-width, :]
      self.expected[iz][-width:, :] = self.xlist[iz][width:2 * width, :]
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_BCs_Width1', True, 1),
      ('_NoBCs_Width1', False, 1),
      ('_BCs_Width2', True, 2),
      ('_NoBCs_Width2', False, 2),
      ('_BCs_Width5', True, 5),
      ('_NoBCs_Width5', False, 5),
  )
  def testSingleCoreHaloExchangePeriodicX(self, set_bcs, width):
    x = self.singleCoreHaloExchangePeriodicX(set_bcs, width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangePeriodicY(self, set_bcs, width):
    dim = 1
    periodic = True
    for iz in range(self.grid_size):
      self.expected[iz][:, :width] = self.xlist[iz][:, -2 * width:-width]
      self.expected[iz][:, -width:] = self.xlist[iz][:, width:2 * width]
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_BCs_Width1', True, 1),
      ('_NoBCs_Width1', False, 1),
      ('_BCs_Width2', True, 2),
      ('_NoBCs_Width2', False, 2),
      ('_BCs_Width5', True, 5),
      ('_NoBCs_Width5', False, 5),
  )
  def testSingleCoreHaloExchangePeriodicY(self, set_bcs, width):
    x = self.singleCoreHaloExchangePeriodicY(set_bcs, width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangePeriodicZ(self, set_bcs, width):
    dim = 2
    periodic = True
    self.expected[:width] = self.xlist[-2 * width:-width]
    self.expected[-width:] = self.xlist[width:2 * width]
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_BCs_Width1', True, 1),
      ('_NoBCs_Width1', False, 1),
      ('_BCs_Width2', True, 2),
      ('_NoBCs_Width2', False, 2),
      ('_BCs_Width5', True, 5),
      ('_NoBCs_Width5', False, 5),
  )
  def testSingleCoreHaloExchangePeriodicZ(self, set_bcs, width):
    x = self.singleCoreHaloExchangePeriodicZ(set_bcs, width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeNonPeriodicNoBcsX(self, width):
    dim = 0
    periodic = False
    set_bcs = False
    for iz in range(self.grid_size):
      self.expected[iz][:width, :] = 0
      self.expected[iz][-width:, :] = 0
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeNonPeriodicNoBcsX(self, width):
    x = self.singleCoreHaloExchangeNonPeriodicNoBcsX(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeNonPeriodicNoBcsY(self, width):
    dim = 1
    periodic = False
    set_bcs = False
    for iz in range(self.grid_size):
      self.expected[iz][:, :width] = 0
      self.expected[iz][:, -width:] = 0
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeNonPeriodicNoBcsY(self, width):
    x = self.singleCoreHaloExchangeNonPeriodicNoBcsY(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeNonPeriodicNoBcsZ(self, width):
    dim = 2
    periodic = False
    set_bcs = False
    self.expected[:width] = [self.zeros] * width
    self.expected[-width:] = [self.zeros] * width
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeNonPeriodicNoBcsZ(self, width):
    x = self.singleCoreHaloExchangeNonPeriodicNoBcsZ(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeDnBcsX(self, width):
    dim = 0
    periodic = False
    set_bcs = True
    # Left is Dirichlet, right is Neumann.
    for iz in range(self.grid_size):
      self.expected[iz][:width, :] = self.bclist[iz][:width, :]
      for w in range(width):
        self.expected[iz][-width + w:, :] = (
            self.expected[iz][-width + w - 1, :] +
            self.bclist[iz][-width + w, :])
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeDnBcsX(self, width):
    x = self.singleCoreHaloExchangeDnBcsX(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeDnBcsY(self, width):
    dim = 1
    periodic = False
    set_bcs = True
    # Left is Dirichlet, right is Neumann.
    for iz in range(self.grid_size):
      self.expected[iz][:, :width] = self.bclist[iz][:, :width]
      for w in range(width):
        self.expected[iz][:, -width + w] = (
            self.expected[iz][:, -width + w - 1] +
            self.bclist[iz][:, -width + w])
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeDnBcsY(self, width):
    x = self.singleCoreHaloExchangeDnBcsY(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeDnBcsZ(self, width):
    dim = 2
    periodic = False
    set_bcs = True
    self.expected[:width] = self.bclist[:width]
    for w in range(width):
      self.expected[-width + w] = (
          self.expected[-width + w - 1] + self.bclist[-width + w])
    return self.do_halo_exchange(dim, periodic, set_bcs, width)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeDnBcsZ(self, width):
    x = self.singleCoreHaloExchangeDnBcsZ(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeNdBcsX(self, width):
    dim = 0
    periodic = False
    set_bcs = True
    dirichlet_neumann = False
    # Left is Neumann, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected[iz][-width:, :] = self.bclist[iz][-width:, :]
      for w in range(width - 1, -1, -1):
        self.expected[iz][w, :] = (
            self.expected[iz][w + 1, :] - self.bclist[iz][w, :])
    return self.do_halo_exchange(dim, periodic, set_bcs, width,
                                 dirichlet_neumann)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeNdBcsX(self, width):
    x = self.singleCoreHaloExchangeNdBcsX(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeNdBcsY(self, width):
    dim = 1
    periodic = False
    set_bcs = True
    dirichlet_neumann = False
    # Left is Neumann, right is Dirichlet.
    for iz in range(self.grid_size):
      self.expected[iz][:, -width:] = self.bclist[iz][:, -width:]
      for w in range(width - 1, -1, -1):
        self.expected[iz][:, w] = (
            self.expected[iz][:, w + 1] - self.bclist[iz][:, w])
    return self.do_halo_exchange(dim, periodic, set_bcs, width,
                                 dirichlet_neumann)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeNdBcsY(self, width):
    x = self.singleCoreHaloExchangeNdBcsY(width)
    self.assertAllClose(self.expected, x)

  def singleCoreHaloExchangeNdBcsZ(self, width):
    dim = 2
    periodic = False
    set_bcs = True
    dirichlet_neumann = False
    self.expected[-width:] = self.bclist[-width:]
    for w in range(width - 1, -1, -1):
      self.expected[w] = self.expected[w + 1] - self.bclist[w]
    return self.do_halo_exchange(dim, periodic, set_bcs, width,
                                 dirichlet_neumann)

  @parameterized.named_parameters(
      ('_Width1', 1),
      ('_Width2', 2),
      ('_Width5', 5),
  )
  def testSingleCoreHaloExchangeNdBcsZ(self, width):
    x = self.singleCoreHaloExchangeNdBcsZ(width)
    self.assertAllClose(self.expected, x)

  @parameterized.named_parameters(
      # nx = ny = nz = 3
      ('3x3x3Width0', (3, 3, 3), 0,
       (np.ones(shape=(3, 3), dtype=np.float32),) * 3),
      (
          '3x3x3Width1',
          (3, 3, 3),
          1,
          (np.zeros(shape=(3, 3), dtype=np.float32),
           np.array([
               [0] * 3,
               [0, 1, 0],
               [0] * 3,
           ], dtype=np.float32), np.zeros(shape=(3, 3), dtype=np.float32)),
      ),
      # nx = ny = nz = 4
      ('4x4x4Width0', (4, 4, 4), 0,
       (np.ones(shape=(4, 4), dtype=np.float32),) * 4),
      ('4x4x4Width1', (4, 4, 4), 1,
       ((np.zeros(shape=(4, 4), dtype=np.float32),) +
        (np.array([
            [0] * 4,
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0] * 4,
        ],
                  dtype=np.float32),) * 2 +
        (np.zeros(shape=(4, 4), dtype=np.float32),))),
      # nx != ny != nz
      ('3x4x5Width1', (3, 4, 5), 1,
       ((np.zeros(shape=(3, 4), dtype=np.float32),) +
        (np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.float32),) * 3 +
        (np.zeros(shape=(3, 4), dtype=np.float32),))),
      ('4x3x5Width1', (4, 3, 5), 1,
       ((np.zeros(shape=(4, 3), dtype=np.float32),) +
        (np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
                  dtype=np.float32),) * 3 +
        (np.zeros(shape=(4, 3), dtype=np.float32),))),
  )
  def testClearHalos(
      self,
      n_xyz,
      halo_width,
      expected_x_pad,
  ):
    """Tests `clear_halos` resets boundary values to 0 correctly."""
    nx, ny, nz = n_xyz

    x = [tf.ones(shape=(nx, ny), dtype=tf.float32)] * nz

    x_pad = [t.numpy() for t in halo_exchange.clear_halos(x, halo_width)]

    self.assertLen(x_pad, nz)
    for i in range(nz):
      self.assertAllEqual(expected_x_pad[i], x_pad[i],
                          'The {}th z plane does not match!'.format(i))

  @parameterized.named_parameters(
      # nx = ny = nz = 3
      ('3x3x3Width2', (3, 3, 3), 2),
      # nx = ny = nz = 4
      ('4x4x4Width2', (4, 4, 4), 2),
      # nx != ny != nz
      ('4x3x12Width5', (4, 3, 12), 5),
  )
  def testClearHalosFailure(self, n_xyz, halo_width):
    """Tests `clear_halos` failure cases without any interior points."""
    nx, ny, nz = n_xyz
    x = [tf.ones(shape=(nx, ny), dtype=tf.float32)] * nz

    with self.assertRaisesRegex(ValueError, 'there are no interior points'):
      _ = [t.numpy() for t in halo_exchange.clear_halos(x, halo_width)]


if __name__ == '__main__':
  tf.test.main()

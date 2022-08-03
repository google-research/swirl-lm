# Copyright 2022 The swirl_lm Authors.
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

"""Utility functions to compute thermodynamic related quantities."""

from typing import Mapping, Text
from swirl_lm.physics import constants
from swirl_lm.utility import types
import tensorflow as tf

TF_DTYPE = types.TF_DTYPE

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap
_MolecularWeightMap = Mapping[Text, float]

# Universal gas constant, in units of J/mol/K.
R_UNIVERSAL = constants.R_UNIVERSAL
# The precomputed gas constant for dry air, in units of J/kg/K.
R_D = constants.R_D
# The gravitational acceleration constant, in units of N/kg.
G = constants.G

# Name of the inert species.
INERT_SPECIES = 'ambient'


def regularize_scalar_bound(phi: FlowFieldVal) -> FlowFieldVal:
  """Enforces a bound of [0, 1] on the scalar `phi`.

  Args:
    phi: The scalar values that need to be bounded between [0, 1].

  Returns:
    `phi` with 0 <= `phi` <= 1 enforced.
  """
  phi_lower_bounded = [
      tf.compat.v1.where(phi_i < 0., tf.zeros_like(phi_i), phi_i)
      for phi_i in phi
  ]
  return [
      tf.compat.v1.where(phi_i > 1., tf.ones_like(phi_i), phi_i)
      for phi_i in phi_lower_bounded
  ]


def regularize_scalar_sum(phi: FlowFieldMap) -> FlowFieldMap:
  """Rescales the scalars so that their sum at a point is 1.

  Args:
    phi: The state of all scalars.

  Returns:
    The regularized scalars such that the sum of all scalars at each point is 1.
  """
  sc_total = [
      tf.zeros_like(sc_i, dtype=TF_DTYPE) for sc_i in list(phi.values())[0]
  ]
  for sc_val in phi.values():
    sc_total = [
        sc_total_i + sc_val_i for sc_total_i, sc_val_i in zip(sc_total, sc_val)
    ]
  sc_reg = {}
  for sc_name, sc_val in phi.items():
    sc_reg.update({
        sc_name: [
            sc_val_i / sc_total_i
            for sc_val_i, sc_total_i in zip(sc_val, sc_total)
        ]
    })
  return sc_reg


def compute_ambient_air_fraction(phi: FlowFieldMap) -> FlowFieldVal:
  """Computes the mass fraction of the ambient air.

  The total mass fraction at each grid point for all scalars should be 1.

  Args:
    phi: The mass fraction of scalars other than the ambient air.

  Returns:
    The mass fraction of the ambient air.
  """
  y_ambient = [
      tf.ones_like(sc_i, dtype=TF_DTYPE) for sc_i in list(phi.values())[0]
  ]
  for sc_val in phi.values():
    y_ambient = [
        y_ambient_i - sc_val_i
        for y_ambient_i, sc_val_i in zip(y_ambient, sc_val)
    ]
  return regularize_scalar_bound(y_ambient)


def compute_mixture_molecular_weight(
    molecular_weights: _MolecularWeightMap,
    massfractions: FlowFieldMap) -> FlowFieldVal:
  """Computes the mixture molecular weight based on species' massfractions.

  Args:
    molecular_weights: A dictionary with keys being the names of the species,
      and values being the molecular weight of that species.
    massfractions: A dictionary with keys being the names of the species, and
      values being the field of mass fractions.

  Returns:
    The molecular weight of the mixture.
  """
  w_mix_inv = [
      tf.zeros_like(y_i, dtype=TF_DTYPE)
      for y_i in list(massfractions.values())[0]
  ]
  for sc_name, w_sc in molecular_weights.items():
    w_mix_inv = [
        w_mix_inv_i + y_sc / w_sc
        for w_mix_inv_i, y_sc in zip(w_mix_inv, massfractions[sc_name])
    ]

  return [1.0 / w_mix_inv_i for w_mix_inv_i in w_mix_inv]

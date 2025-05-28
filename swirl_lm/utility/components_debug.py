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

"""A library for debugging components in governing equations.

Equations considered in this tool includes the momentum equations and all scalar
transport equations. The names of the variables are rhou, rhov, rhow, and all
scalars listed as `transport scalars` in the parameter context.

Terms considered in each equation, and their naming rules are (take variable ):
1. For convection terms: dbg_[w+]_conv_x, dbg_[w+]_conv_y, dbg_[w+]_conv_z;
2. For diffusion terms: dbg_[w+]_diff_x, dbg_[w+]_diff_y, dbg_[w+]_diff_z;
3. For the source: dbg_[w+]_src
4. For gravitational term (rhou, rhov, and rhow only): dbg_[w+]_gravity
5. For turbulent diffusivity (transported scalars only, when sgs is used):
   dbg_[w+]_D_t
"""

import re
from typing import Optional, Sequence, Text
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

_MOMEMTUM_VARIABLES = ['rhou', 'rhov', 'rhow']


class ComponentsDebug(object):
  """A library for debugging components in governing equations."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the debugging tool."""
    self._params = params

  @property
  def debugging_states_names_full(self):
    """Generates names for all debugging terms."""
    var_names = []
    for sc_name in self._params.transport_scalars_names + _MOMEMTUM_VARIABLES:
      var_names += [
          'dbg_{}_conv_x'.format(sc_name),
          'dbg_{}_conv_y'.format(sc_name),
          'dbg_{}_conv_z'.format(sc_name),
          'dbg_{}_diff_x'.format(sc_name),
          'dbg_{}_diff_y'.format(sc_name),
          'dbg_{}_diff_z'.format(sc_name),
          'dbg_{}_src'.format(sc_name),
      ]
    var_names += [
        'dbg_rhou_gravity',
        'dbg_rhov_gravity',
        'dbg_rhow_gravity',
    ]
    if self._params.use_sgs:
      for sc_name in self._params.transport_scalars_names:
        var_names.append('dbg_{}_D_t'.format(sc_name))

    return var_names

  def debugging_states_names(self):
    """Retrieves names for debugging states from the config file."""
    return [
        key for key in self._params.additional_state_keys
        if re.match('dbg_*', key)
    ]

  def generate_initial_states_full(self):
    """Initializes all debugging terms to zeros and returns."""
    output = {}
    for key in self.debugging_states_names_full:
      output.update(
          common_ops.gen_field(key, self._params.nx, self._params.ny,
                               self._params.nz))
    return output

  def generate_initial_states(self, split_in_z=False):
    """Initializes debugging terms specified in config to zeros and returns."""
    output = {}
    for key in self.debugging_states_names():
      output.update(
          common_ops.gen_field(key, self._params.nx, self._params.ny,
                               self._params.nz))
    return output if not split_in_z else {
        key: tf.unstack(val) for key, val in output.items()
    }

  def update_scalar_terms(
      self,
      key: Text,
      terms: FlowFieldMap,
      diff_t: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Updates the debugging terms for variable named `key`.

    Note that the name of the source term from the input `terms` has the key
    'source'. This will be converted to `dbg_[key]_src` in the output
    dictionary.

    Args:
      key: The name of the scalar.
      terms: All terms for debugging generated for this scalar.
      diff_t: The turbulent diffusivity. It's provided when SGS model is used,
        otherwise it's `None`. When it's provided, the `dbg_[key]_D_t` term will
        be added for debug.

    Returns:
      A dictionary of all debugging terms following the naming rule specified
      in this `ComponentDebug` library.

    Raises:
      ValueError: If `dbg_[key]_D_t` is in the debugging states namelist but no
        `diff_t` is provided, i.e. `diff_t` is None.
    """
    dbg_terms = {}
    for term_key, term_val in terms.items():
      # Rename the term's name to "src" if it's the source term.
      term_key = 'src' if term_key == 'source' else term_key
      term_name = 'dbg_{}_{}'.format(key, term_key)
      if term_name not in self.debugging_states_names():
        continue
      dbg_terms.update({term_name: term_val})

    d_t_name = 'dbg_{}_D_t'.format(key)
    if d_t_name in self.debugging_states_names():
      if diff_t is None:
        raise ValueError(
            'D_t for {} is requested for debug but is not provided'.format(key))
      dbg_terms.update({d_t_name: diff_t})

    return dbg_terms

  def update_momentum_terms(self,
                            terms: Sequence[FlowFieldMap]) -> FlowFieldMap:
    """Updates the debugging terms for the momentum.

    Note that the name of the forcing term from the input `terms` has the key
    'force'. This will be converted to `dbg_[key]_src` in the output
    dictionary, for key being 'rhou', 'rhov', and 'rhow'.

    Args:
      terms: All terms for debugging generated for the momentum. This variable
        is a sequence of length 3, with component 0, 1, and 2 corresponds to
        momentum component in dimension 0, 1, and 2, respectively.

    Returns:
      A dictionary of all debugging terms following the naming rule specified
      in this `ComponentDebug` library.
    """
    dbg_terms = {}
    for i in range(3):
      var_name = _MOMEMTUM_VARIABLES[i]
      for term_key, term_val in terms[i].items():
        # Rename the term's name to "src" if it's the forcing term.
        term_key = 'src' if term_key == 'force' else term_key
        term_name = 'dbg_{}_{}'.format(var_name, term_key)
        if term_name not in self.debugging_states_names():
          continue
        dbg_terms.update({term_name: term_val})

    return dbg_terms

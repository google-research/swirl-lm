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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library that processes data from the LandFire database.

The format of the .lcp file is described on
https://gdal.org/drivers/raster/lcp.html
"""
import struct
from typing import Any, Dict

import numpy as np

VARNAMES = ('elev', 'slope', 'aspect', 'fuel', 'cover', 'height', 'base',
            'density', 'duff', 'woody')


def _read_short(f):
  """Reads one short integer (4 bytes) from the file handler."""
  return struct.unpack('h', f.read(2))[0]


def _read_long(f):
  """Reads one long integer (4 bytes) from the file handler."""
  return struct.unpack('i', f.read(4))[0]


def _read_double(f):
  """Reads one double (8 bytes) from the file handler."""
  return struct.unpack('d', f.read(8))[0]


def _read_string(f, buf_size):
  """Reads a string of `buf_size`."""
  return ''.join([
      c[0].decode('utf-8')
      for c in struct.iter_unpack('c', f.read(buf_size))
      if c[0] != b'\x00'
  ])


def _read_grouped_data(f, varname):
  """Reads grouped data from a LCP file."""
  data = dict()
  data.update({f'lo{varname}': _read_long(f)})
  data.update({f'hi{varname}': _read_long(f)})
  data.update({f'num{varname}': _read_long(f)})
  data.update({
      f'{varname}_values': [h[0] for h in struct.iter_unpack('i', f.read(400))]
  })
  return data


def read_lcp(filename: str) -> Dict[str, Any]:
  """Reads an LCP file and returns all information in a dictionary."""
  data = {}
  with open(filename, 'rb') as f:
    # Read the header of the landscape file.
    data.update({'crown_fuels': _read_long(f)})
    data.update({'ground_fuels': _read_long(f)})
    data.update({'latitude': _read_long(f)})
    data.update({'loeast': _read_double(f)})
    data.update({'hieast': _read_double(f)})
    data.update({'lonorth': _read_double(f)})
    data.update({'hinorth': _read_double(f)})

    for varname in VARNAMES:
      data.update(_read_grouped_data(f, varname))

    data.update({'num_east': _read_long(f)})
    data.update({'num_north': _read_long(f)})
    data.update({'east_utm': _read_double(f)})
    data.update({'west_utm': _read_double(f)})
    data.update({'north_utm': _read_double(f)})
    data.update({'south_utm': _read_double(f)})
    data.update({'grid_units': _read_long(f)})
    data.update({'x_res': _read_double(f)})
    data.update({'y_res': _read_double(f)})

    for varname in VARNAMES:
      data.update({f'{varname}_units': _read_short(f)})

    for varname in VARNAMES:
      data.update({f'{varname}_file': _read_string(f, 256)})

    data.update({'description': _read_string(f, 512)})

    # Read the data in raster format.
    nx = int((data['east_utm'] - data['west_utm']) / data['x_res'])
    ny = int((data['north_utm'] - data['south_utm']) / data['y_res'])

    raster = []
    while buf := [
        val[0] for val in struct.iter_unpack('h', f.read(nx * ny * 2))
    ]:
      raster += buf
    raster = np.reshape(np.array(raster), (ny, nx, -1))
    nvar = raster.shape[2]
    data.update({VARNAMES[i]: raster[..., i] for i in range(nvar)})

  return data

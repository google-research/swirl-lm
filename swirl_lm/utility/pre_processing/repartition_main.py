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

"""Interpolates a checkpoint file (ser) and repartitions the data."""

from collections.abc import Sequence
import itertools

from absl import app
from absl import flags
import apache_beam as beam
import fancyflags as ff
from swirl_lm.utility.pre_processing import repartition
import xarray_beam as xbeam


ff.DEFINE_auto(
    'source',
    repartition.DataInfo,
    'Information about the mesh and partition for the source data.',
)
ff.DEFINE_auto(
    'target',
    repartition.DataInfo,
    'Information about the mesh and partition for the target data.',
)
flags.DEFINE_list('varnames', [], 'Variables that requires repartitioning.')
flags.DEFINE_list(
    'pipeline_options',
    ['--runner=DirectRunner'],
    'A comma-separated list of command line arguments to be used as options'
    ' for the Beam Pipeline.',
)


def main(argv: Sequence[str]) -> None:
  del argv

  # Get the mesh information from the source and target.
  mesh_source = repartition.get_mesh(flags.FLAGS.source())
  mesh_target = repartition.get_mesh(flags.FLAGS.target())

  n_cores_source = repartition.coords_to_dims(
      {
          'x': flags.FLAGS.source().cx,
          'y': flags.FLAGS.source().cy,
          'z': flags.FLAGS.source().cz,
      },
      flags.FLAGS.source().tensor_orientation,
  )
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      flags.FLAGS.pipeline_options
  )

  with beam.Pipeline(options=pipeline_options) as root:
    _ = (
        root
        | beam.Create(
            itertools.product(
                range(flags.FLAGS.source().cx),
                range(flags.FLAGS.source().cy),
                range(flags.FLAGS.source().cz),
                flags.FLAGS.varnames,
            )
        )
        # 1. Load the ser files and convert them to xbeam chunks,  keeping the
        # same partitioning as the ser files.
        | 'Reshuffle 0' >> beam.Reshuffle()
        | 'Load source'
        >> beam.FlatMap(
            repartition.load_source, flags.FLAGS.source(), mesh_source
        )
        # 2. Consolidate the dataset so that it is chunked along the 0th
        # dimension only. After this step, each chunk represents a full 2D plane
        # perpendicular to the 0th dimension.
        | 'Rechunk for 2D interpolation'
        >> xbeam.Rechunk(
            dim_sizes={
                'dim_0': len(mesh_source['dim_0']),
                'dim_1': len(mesh_source['dim_1']),
                'dim_2': len(mesh_source['dim_2']),
            },
            source_chunks={
                dim: int(len(mesh) / n_cores_source[dim])
                for dim, mesh in mesh_source.items()
            },
            target_chunks={'dim_0': 1, 'dim_1': -1, 'dim_2': -1},
            itemsize=8,
            max_mem=2 ** 31,
        )
        # 3. Perform interpolation on the 2D plane in each chunk.
        | 'Interpolate dim 1 and 2'
        >> beam.FlatMapTuple(
            repartition.interpolate,
            mesh_target={
                'dim_1': mesh_target['dim_1'],
                'dim_2': mesh_target['dim_2'],
            },
        )
        # 4. Rechunk the data again, with each chunk corresponds to an index
        # pair of the 2D plane generated in step 3.
        | 'Rechunk for 1D interpolation'
        >> xbeam.Rechunk(
            dim_sizes={
                'dim_0': len(mesh_source['dim_0']),
                'dim_1': len(mesh_target['dim_1']),
                'dim_2': len(mesh_target['dim_2']),
            },
            source_chunks={'dim_0': 1, 'dim_1': -1, 'dim_2': -1},
            target_chunks={'dim_0': -1, 'dim_1': 1, 'dim_2': 1},
            itemsize=8,
        )
        # 5. Perform interpolation along the single dimension in each chunk.
        | 'Interpolate dim 0'
        >> beam.FlatMapTuple(
            repartition.interpolate, mesh_target={'dim_0': mesh_target['dim_0']}
        )
        # 6. Rechunk the data one last time to fit the target partition.
        | 'Rechunk to target'
        >> xbeam.Rechunk(
            dim_sizes={dim: len(mesh) for dim, mesh in mesh_target.items()},
            source_chunks={'dim_0': -1, 'dim_1': 1, 'dim_2': 1},
            target_chunks=repartition.get_chunks(flags.FLAGS.target()),
            itemsize=8,
        )
        # 7. Write the chunked data as ser files.
        | 'Write ser'
        >> beam.FlatMapTuple(
            repartition.write_target_to_ser, target=flags.FLAGS.target()
        )
    )


if __name__ == '__main__':
  app.run(main)

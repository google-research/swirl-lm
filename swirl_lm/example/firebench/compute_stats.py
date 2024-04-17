# Copyright 2024 The swirl_lm Authors.
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

r"""Calculates min/max/mean values for a Swirl-LM dataset.

For an overview of running Apache Beam pipelines on Google Cloud, see:

https://cloud.google.com/dataflow/docs/guides/use-beam
https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python

Steps to run:

1. Build a custom container:

   * Change to the directory swirl_lm/example/firebench/docker.

   * Run:

     gcloud builds submit --region=<region> \
       --tag <region>-docker.pkg.dev/<project_id>/<repository>/<image>:<tag>

     <region>: For example, us-central1.
     <project_id>: Google Cloud project id. If the id contains ':'s, replace
                   them with slashes.
     <repository>: A new or existing repository name.
     <image>: A new unique name for the image.
     <tag>: The tag for the version being created.

     For more info see:
     https://cloud.google.com/dataflow/docs/guides/build-container-image

   * Verify that the image was built successfully by viewing the "Artifact
     Registry" pages in the Cloud console.

   * The image does not need to be rebuilt as long as new python dependencies
     are not added.

2. Launch the dataflow job:

   * Verify that the machine used for the launch has the same version of python3
     as the Beam image (see docker/Dockerfile) and all the requirements
     (docker/requirements.txt) are installed. Using a virtual env is recommended
     for setting up python and dependencies on the launch machine.

     An alternative to setting up a virtual env is to start up a shell using the
     docker image and launch from the docker image, though this is currently not
     well tested.

   * Change to the directory swirl_lm/example/firebench.

     python3 compute_stats.py \
       --input_path=gs://firebench/<path_to_zarr> \
       --output_path=gs://<path_to_output_zarr> \
       --pipeline_options="--runner=apache_beam.runners.dataflow.dataflow_runner.DataflowRunner,--project=<project_id>,--temp_location=gs://<temp_location>,--staging_location=gs://<staging_location>,--region=<region>,--sdk_container_image=<image_path>@<digest>,--sdk_location=container,--save_main_session"

     <path_to_zarr>: Path to input zarr dataset.
     <path_to_output_zarr>: Path to output zarr dataset.
     <project_id>: Google Cloud project id.
     <temp_location>: Writable path in a GCS bucket.
     <staging_location>: Writable path in a GCS bucket.
     <region>: For example, us-central1.
     <image_path>: <region>-docker.pkg.dev/<project_id>/<repository>/<image>
                   from step 1 *without* the tag.
     <digest>: Digest as output by gcloud builds or as shown on the "Artifact
               Repository", e.g., sha256:6e1cf2a963132a240fd06f535c9f9e8cfb1353ca510b2df31cf2f32ff658a8c9

"""


from typing import Tuple

from absl import app
from absl import flags
import apache_beam as beam
import xarray
import xarray_beam as xbeam


# NOTE: To make top-level imports available to workers, we need to have
# --save_main_sesion=True, but then Beam refuses to save flag values (via
# pickling) so we can't assign flags to global variables as we normally do.
flags.DEFINE_string('input_path', None, help='Input Zarr path')
flags.DEFINE_string('output_path', None, help='Output Zarr path')
flags.DEFINE_list(
    'pipeline_options', ['--runner=DirectRunner'],
    'A comma-separated list of command line arguments to be used as options'
    ' for the Beam Pipeline.'
)


def compute_stats(
    key: xbeam.Key, dataset: xarray.Dataset
) -> Tuple[xbeam.Key, xarray.Dataset]:
  """Computes spatial mean/min/max for all variables."""
  spatial_dims = set(dataset.dims) - {'t'}
  return (
      key.with_offsets(x=None, y=None, z=None, stat=0),
      xarray.concat(
          [
              dataset.mean(spatial_dims).assign_coords(stat='mean'),
              dataset.min(spatial_dims).assign_coords(stat='min'),
              dataset.max(spatial_dims).assign_coords(stat='max'),
          ],
          dim='stat',
      ),
  )


class CombineStatsFn(beam.CombineFn):
  """Combiner for mean/min/max.

  Keeps track of count of datasets and the accumulated dataset. The accumulated
  dataset keeps the sum of means (at stat='mean') from the input datasets, and
  the min and the max. At the end of the combine stage, the mean is calculated
  from the sum of the means and the count.
  """

  def create_accumulator(self, *args, **kwargs):
    return 0, None  # Count of datasets, accumulated dataset

  def _merge_stats(self, left, right):
    if left[0] == 0:
      return right
    if right[0] == 0:
      return left
    accumulator = xarray.concat(
        [left[1].sel(stat='mean') + right[1].sel(stat='mean'),
         xarray.where((left[1].sel(stat='min') <
                       right[1].sel(stat='min')),
                      left[1].sel(stat='min'),
                      right[1].sel(stat='min')),
         xarray.where((left[1].sel(stat='max') >
                       right[1].sel(stat='max')),
                      left[1].sel(stat='max'),
                      right[1].sel(stat='max'))], dim='stat')
    return left[0] + right[0], accumulator

  def add_input(self, mutable_accumulator, element, *args, **kwargs):
    return self._merge_stats(mutable_accumulator, (1, element))

  def merge_accumulators(self, accumulators, *args, **kwargs):
    out = 0, None
    for accumulator in accumulators:
      out = self._merge_stats(out, accumulator)
    return out

  def extract_output(self, accumulator, *args, **kwargs):
    return xarray.concat(
        [
            accumulator[1].sel(stat='mean') / accumulator[0],
            accumulator[1].sel(stat='min'),
            accumulator[1].sel(stat='max'),
        ],
        dim='stat',
    )


class ComputeStats(beam.PTransform):
  """Main pipeline as a PTransform to make testing easier."""

  def __init__(self, input_path: str, output_path: str):
    self.input_path = input_path
    self.output_path = output_path

  def expand(self, pcoll):
    source_dataset, source_chunks = xbeam.open_zarr(self.input_path)

    template = (
        xbeam.make_template(source_dataset)
        .isel(x=0, y=0, z=0, drop=True)
        .expand_dims(stat=['mean', 'min', 'max'])
    )

    compute_stats_sizes = dict(source_dataset.sizes)
    del compute_stats_sizes['x']
    del compute_stats_sizes['y']
    del compute_stats_sizes['z']
    compute_stats_sizes['stat'] = 3

    compute_stats_chunks = dict(source_chunks)
    del compute_stats_chunks['x']
    del compute_stats_chunks['y']
    del compute_stats_chunks['z']
    compute_stats_chunks['stat'] = -1

    output_chunks = {'t': compute_stats_sizes['t'], 'stat': 3}

    return (
        pcoll
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | beam.MapTuple(compute_stats)
        | beam.CombinePerKey(CombineStatsFn())
        | xbeam.Rechunk(
            compute_stats_sizes,
            compute_stats_chunks,
            output_chunks,
            itemsize=8,
            min_mem=0,  # Small chunks are OK.
        )
        | xbeam.ChunksToZarr(self.output_path, template, output_chunks)
    )


def main(args):
  del args
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      flags.FLAGS.pipeline_options)
  with beam.Pipeline(options=pipeline_options) as root:
    _ = (
        root
        | ComputeStats(flags.FLAGS.input_path, flags.FLAGS.output_path)
    )


if __name__ == '__main__':
  app.run(main)

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

r"""Example XManager script that launches a JAX TPU or GPU job on Borg.

Examples:

* Launch a job with 16 A100:

google_xmanager launch dist_fft_xm_launcher.py -- \
--xm_resource_pool=research-training \
--xm_resource_alloc=group:research-training/sim-research-xm \
--cell=oa --resource_quantity=16 --platform=a100 \
--experiment_title=fft_test_a100 \
--flags=cx=4,cy=4,nx=32768,ny=32768,backend=gpu,num_cycles=2000,num_transforms=3


* Launch a job with 2x2x2 TPU v5:

xmanager launch launch dist_fft_xm_launcher.py -- \
--xm_resource_pool=research-training \
--xm_resource_alloc=group:research-training/sim-research-xm --cell=nz \
--resource_quantity=2x2x2 --platform=viperfish --experiment_title=fft_test_vf \
--flags=cx=4,cy=2,nx=32768,ny=32768,backend=tpu,num_cycles=4000,\
num_transforms=3,deepsea_chip_config_name=megacore


* Launch a job with 16x16 TPU v5e:

xmanager launch launch dist_fft_xm_launcher.py -- \
--xm_resource_pool=research-training \
--xm_resource_alloc=group:research-training/sim-research-xm --cell=eb \
--resource_quantity=16x16 --platform=viperlite_pod \
--experiment_title=fft_test_vlp \
--flags=cx=16,cy=16,nx=65536,ny=65536,backend=tpu,num_cycles=300,\
num_transforms=10

"""

import re
from typing import Any, Dict, List, Union

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc


_CELL = flags.DEFINE_string('cell', None, 'Tpu cell')
_PLATFORM = flags.DEFINE_string(
    'platform', 'jellyfish', 'Accelerator type'
)
_RESOURCE_QUANTITY = flags.DEFINE_string(
    'resource_quantity', '2x2', 'Amount of GPUs or the topology of the TPU.'
)
_BUILD_TARGET_PATH = flags.DEFINE_string(
    'build_target_path',
    '//swirl_lm/ext/fft:dist_fft_perf_main.par',
    'The build target to run.',
)
_EXPERIMENT_TITLE = flags.DEFINE_string(
    'experiment_title', 'fft', 'The title of the experiment.'
)
_FLAGS = flags.DEFINE_string(
    'flags', '', 'Comma separated list of flags to pass through.'
)


def parse_flags(flags_str: str) -> Dict[str, Union[List[str], str]]:
  """Parses comma separated flags into a Dict suitable for BuildTarget args."""
  out: Dict[str, Any] = {}
  # Actually Dict[str, Union[List[str], str]] but
  # pytype gets confused with that below.
  # Split on commas just before the next var so that commas in values are
  # allowed, e.g., 'input=a,b,output=c' means '--input=a,b --output=c'.
  for flag in re.split(r',(?=[^,]*=)', flags_str):
    if not flag:
      continue
    name, value = flag.split('=', 1)
    if name not in out:
      out[name] = value
    else:
      if not isinstance(out[name], list):
        out[name] = [out[name]]
      out[name].append(value)
  out['xprof_port'] = '%port_xprof%'
  return out


def main(_):
  bazel_args = xm_abc.bazel_args.for_resource(_PLATFORM.value)
  # This is necessary for linking to succeed when using GPUs.
  # In some informal testing, it did not seem to slow down compilation of
  # non-GPU workloads, so we're always enabling it for simplicity.
  bazel_args = bazel_args + ('--define=cuda_compress=1',)
  with xm_abc.create_experiment(
      experiment_title=_EXPERIMENT_TITLE.value,) as experiment:
    [executable] = experiment.package([
        xm.bazel_binary(
            label=_BUILD_TARGET_PATH.value,
            executor_spec=xm_abc.Borg.Spec(),
            bazel_args=bazel_args,
        ),
    ])

    executor = xm_abc.Borg(
        requirements=xm.JobRequirements(
            {_PLATFORM.value: _RESOURCE_QUANTITY.value},
            location=_CELL.value,
            service_tier=xm.ServiceTier.PROD,
        ),
        logs_read_access_roles=['all'],

    )

    job = xm.Job(executable, executor)
    experiment.add(job, args={'args': parse_flags(_FLAGS.value)},)


if __name__ == '__main__':
  app.run(main)

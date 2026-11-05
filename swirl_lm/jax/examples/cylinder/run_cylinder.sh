#!/bin/bash
#
# Runs the cylinder validation case for the ghost-point immersed boundary.
#
#   ./run_cylinder.sh                              # conjugate gradient, Re = 100
#   CONFIG=.../cylinder_multigrid.pbtxt ./run_cylinder.sh
#   RE=40 ./run_cylinder.sh                        # the steady validation point
#   JAX_PLATFORMS=cuda ./run_cylinder.sh
#   PYTHON=/path/to/python ./run_cylinder.sh
#
# The case setup is read from any of the config files which differ only in the
# solver block. Only the Reynolds number is overridable from here, because the
# viscosity follows from it and nothing else has to move.
#
# Note: On CPU this script derives the simulated device count from the
# decomposition; on GPU the scheduler decides, and cylinder_main warns if the
# two disagree.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

CONFIG="${CONFIG:-${HERE}/cylinder_cg.pbtxt}"
OUTPUT="${OUTPUT:-${HERE}/SOLUT}"
NUM_CYCLES="${NUM_CYCLES:-40}"
NUM_STEPS="${NUM_STEPS:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
PYTHON="${PYTHON:-python}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Resolve the decomposition the same way the run will, by reading the case
# file. Reading it rather than guessing is what keeps the simulated CPU device
# count in step with it.
read -r CX CY CZ NCORES < <(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from google.protobuf import text_format
from swirl_lm.jax.examples.cylinder import cylinder_pb2

case = cylinder_pb2.CylinderCase()
with open(sys.argv[1]) as f:
  text_format.Parse(f.read(), case)
cx, cy, cz = case.cores_x, case.cores_y, case.cores_z
print(cx, cy, cz, cx * cy * cz)
PY
)

# On GPU the scheduler decides how many devices exist; cylinder_main compares
# that against the decomposition and says so if they differ. On CPU the device
# count has to be asked for.
if [[ "${JAX_PLATFORMS}" != *"cuda"* && "${JAX_PLATFORMS}" != *"gpu"* ]]; then
  export XLA_FLAGS="--xla_force_host_platform_device_count=${NCORES} ${XLA_FLAGS:-}"
fi

ARGS=(
  --config_filepath="${CONFIG}"
  --checkpoint_dir="${OUTPUT}"
  --checkpoint_interval="${CHECKPOINT_INTERVAL}"
  --num_cycles="${NUM_CYCLES}"
  --num_steps="${NUM_STEPS}"
)
[[ -n "${RE:-}" ]] && ARGS+=(--reynolds_number="${RE}")

mkdir -p "${OUTPUT}"
cd "${ROOT}"
echo "backend ${JAX_PLATFORMS}, decomposition ${CX}x${CY}x${CZ} = ${NCORES} device(s)"
echo "running: ${PYTHON} -m swirl_lm.jax.examples.cylinder.cylinder_main ${ARGS[*]}"
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" exec "${PYTHON}" -m \
  swirl_lm.jax.examples.cylinder.cylinder_main "${ARGS[@]}"

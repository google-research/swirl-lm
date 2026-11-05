#!/bin/bash
#
# Runs the cylinder validation case for the ghost-point immersed boundary.
#
#   ./run_cylinder.sh                     # Re = 100, shedding, one device
#   RE=40 ./run_cylinder.sh               # the steady validation point
#   CELLS=40 ./run_cylinder.sh            # twice the resolution
#   CORES_X=2 CORES_Y=2 ./run_cylinder.sh # four devices
#   JAX_PLATFORMS=cuda CORES_X=4 ./run_cylinder.sh
#   PYTHON=/path/to/python ./run_cylinder.sh
#
# The decomposition is CORES_X x CORES_Y x CORES_Z, and it is the only thing
# that decides how many devices are used -- on CPU and on GPU alike. There is
# deliberately no separate "number of devices" knob: `driver.create_mesh`
# takes `jax.devices()[:cores_x*cores_y*cores_z]`, so a device count that
# disagrees with the decomposition either fails or, worse, silently leaves
# devices idle. On CPU this script derives the simulated device count from the
# decomposition; on GPU the scheduler decides, and cylinder_main warns if the
# two disagree.
#
# Not every decomposition is legal. Each per-core extent, halos included, must
# divide by the kernel widths (see MEMORY.md), which for the default case
# allows cores_x in {1,2,4,5,8} and cores_y in {1,2,4,5}. The span is the
# cheapest axis for the immersed boundary -- the cylinder is invariant along
# it, so no ghost-point stencil ever crosses a span seam -- but splitting it
# needs spanwise_cells >= 4*cores_z, so cores_z > 1 means raising
# `spanwise_cells` in the case file. The script reports the resolved
# decomposition before starting, and `Cylinder` raises on an illegal one.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

CONFIG="${CONFIG:-${HERE}/cylinder.pbtxt}"
OUTPUT="${OUTPUT:-${HERE}/SOLUT}"
NUM_CYCLES="${NUM_CYCLES:-40}"
NUM_STEPS="${NUM_STEPS:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
PYTHON="${PYTHON:-python}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Resolve the decomposition the same way the run will: the case file, with any
# environment override applied. Reading it here rather than guessing is what
# keeps the simulated CPU device count in step with it.
read -r CX CY CZ NCORES < <(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON}" - "${CONFIG}" <<'PY'
import sys, os
from google.protobuf import text_format
from swirl_lm.jax.examples.cylinder import cylinder_pb2

case = cylinder_pb2.CylinderCase()
with open(sys.argv[1]) as f:
  text_format.Parse(f.read(), case)
cx = int(os.environ.get('CORES_X') or case.cores_x)
cy = int(os.environ.get('CORES_Y') or case.cores_y)
cz = int(os.environ.get('CORES_Z') or case.cores_z)
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
  --cores_x="${CX}"
  --cores_y="${CY}"
  --cores_z="${CZ}"
)
[[ -n "${RE:-}" ]] && ARGS+=(--reynolds_number="${RE}")
[[ -n "${CELLS:-}" ]] && ARGS+=(--cells_per_diameter="${CELLS}")

mkdir -p "${OUTPUT}"
cd "${ROOT}"
echo "backend ${JAX_PLATFORMS}, decomposition ${CX}x${CY}x${CZ} = ${NCORES} device(s)"
echo "running: ${PYTHON} -m swirl_lm.jax.examples.cylinder.cylinder_main ${ARGS[*]}"
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" exec "${PYTHON}" -m \
  swirl_lm.jax.examples.cylinder.cylinder_main "${ARGS[@]}"

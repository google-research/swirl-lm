# Flow over cylinder case

This case demonstrates a flow over a cylinder following the validation
case in [Ghias et al., J. Comput. Phys., 2007](https://linkinghub.elsevier.com/retrieve/pii/S0021999106006048).
The cylinder has a diameter of 1 m, and the flow can be studied at
different Reynolds numbers.
The present case is used as validation of the immersed boundary method
and as a simple test case to examine the convergence of different
Poisson solvers.

## How to run

The case can be run via the provided runner `run_cylinder.sh` or by
calling `cylinder_main.py` with command line arguments.
By default, the runner will use the conjugate gradient solver and
the `cylinder_cg.pbtxt` config file:

```bash
# Using the run_cylinder.sh script on GPUs
JAX_PLATFORMS=cuda ./run_cylinder.sh
```

The above command will run the simulation for a total of 20000 steps,
with the output written to `./SOLUT` every 500 steps. The flow
Reynolds number is `Re = 100` (default value).
The simulation can be customized by providing the following optional
arguments to the runner script:

```bash
# Use a different Reynolds number and checkpointing
JAX_PLATFORMS=cuda RE=40 OUTPUT=./SOLUT_RE40 \
  CHECKPOINT_INTERVAL=100 NUM_STEPS=100 NUM_CYCLES=10 \
  CONFIG=./cylinder_cg.pbtxt ./run_cylinder.sh
```

This will run the simulation at a Reynolds number of `Re = 40`.
**Note:** On CPU, either set `JAX_PLATFORMS=cpu` or omit the
environment variable altogether.

To execute the simulation via the main simulation file, call it as a
module from the repository root:

```bash
JAX_PLATFORMS=cuda python -m swirl_lm.jax.examples.cylinder.cylinder_main \
  --config_filepath=swirl_lm/jax/examples/cylinder/cylinder_cg.pbtxt \
  --num_steps=100 --num_cycles=10 --checkpoint_interval=10 \
  --checkpoint_dir=./SOLUT_CG \
  --reynolds_number=100
```

For mesh decomposition across multiple GPUs (or CPUs), set the desired
`cores_x:`, `cores_y:` and `cores_z:` in the `cylinder_cg.pbtxt`
config file. The runner reads them from there, so the decomposition
cannot drift between the script and the case.

## Running across several nodes

More than one node needs one process per GPU, and those processes have to
join into a single job before any of them touches a device. `cylinder_main.py`
joins on startup, so the launcher only has to start the tasks under `srun`,
which is what lets `jax.distributed.initialize()` find its peers.

How the devices are spread over processes is a separate choice from the
decomposition. JAX pins each Slurm task to one GPU, so `--ntasks-per-node=4`
gives four single-device processes; `--ntasks-per-node=1` instead gives one
process holding all four, which keeps a node's collectives inside that process
and leaves only the hop between nodes crossing a process boundary. Both run the
same decomposition, and on this mesh the second is the cheaper of the two.

The decomposition itself comes from the config file, so `cores_x * cores_y *
cores_z` must equal the total number of *devices*; a mismatch is refused to
avoid stalling, because a process holding no part of the state would still
enter the compiled cycle. Legal splits of this mesh are `cores_x` in
`{1,2,4,5,8}` and `cores_y` in `{1,2,4,5}`, with `cores_z: 1` while
`spanwise_cells: 4`, so eight devices can be arranged as `4x2x1`, `2x4x1` or
`8x1x1`.

Nothing about the physics changes with the decomposition: the mesh follows
from `cells_per_diameter` and the domain extents, so the Strouhal number is
the same on one device as on eight, and a disagreement means a halo exchange
is wrong.

### A batch script for NERSC Perlmutter

Note: Batch scripts are site-specific. Example below runs this case on
Perlmutter; save it as `sbatch_cylinder.sh` beside `run_cylinder.sh` and submit
it from that directory.

```bash
#!/bin/bash
#
# Runs the cylinder case across several GPU nodes on Perlmutter.
#
#   cd swirl_lm/jax/examples/cylinder
#   sbatch --nodes=1 --ntasks-per-node=1 sbatch_cylinder.sh   # single device
#   sbatch --nodes=1 --ntasks-per-node=4 sbatch_cylinder.sh   # one node
#   sbatch --nodes=2 --ntasks-per-node=4 sbatch_cylinder.sh   # two nodes
#   sbatch --nodes=2 --ntasks-per-node=1 sbatch_cylinder.sh   # two, one proc each
#
# The decomposition comes from the config file, not from here: set
# `cores_x/cores_y/cores_z` so their product equals the device count, which is
# nodes x gpus-per-node however the tasks are arranged.
# A mismatch is refused by `driver.create_mesh` rather than left to hang.
#
# One task per GPU: JAX reads SLURM_LOCALID to pick the device, so every task
# may see all four and no CUDA_VISIBLE_DEVICES juggling is needed. One task per
# *node* works too and claims all four, which keeps a node's collectives inside
# one process and leaves only the hop between nodes crossing processes.
#
#SBATCH --job-name=cylinder
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=log_cylinder_%j.log
#SBATCH --qos=regular
## Set your project before submitting:
##SBATCH --account=mXXXX

set -euo pipefail

# Slurm runs a *copy* of this script out of its spool directory, so
# `BASH_SOURCE` points there and not at the case. `sbatch` therefore has to be
# invoked from this directory, which is what SLURM_SUBMIT_DIR then reports.
HERE="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
if [[ ! -f "${HERE}/run_cylinder.sh" ]]; then
  echo "${HERE} holds no run_cylinder.sh: submit from the cylinder example" \
       "directory." >&2
  exit 1
fi
ROOT="$(cd "${HERE}/../../../.." && pwd)"

export JAX_PLATFORMS=cuda
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# The pip `nvidia-*-cu12` wheels ship their own CUDA; let them own it.
module unload cudatoolkit 2>/dev/null || true

# NCCL over Slingshot. Set NCCL_PLUGIN_DIR to the `plugin/lib` of an `nccl/*`
# module -- that subdirectory only, never `module load nccl/...`, which would
# shadow the pip NCCL and undo the line above. The plugin is built per CUDA
# major version, so try a cu12 one first to match the wheels.
#
#   module show nccl/2.24.3        # find the install prefix
#   ls <prefix>/plugin/lib         # expect libnccl-net*.so
NCCL_PLUGIN_DIR="${NCCL_PLUGIN_DIR:-}"
if [[ -n "${NCCL_PLUGIN_DIR}" ]]; then
  if compgen -G "${NCCL_PLUGIN_DIR}/libnccl-net*.so" >/dev/null; then
    export LD_LIBRARY_PATH="${NCCL_PLUGIN_DIR}:${LD_LIBRARY_PATH:-}"
    export NCCL_NET="AWS Libfabric"
    export NCCL_NET_GDR_LEVEL=LOC
    export NCCL_CROSS_NIC=2
    export NCCL_SOCKET_IFNAME=hsn
    export FI_CXI_DISABLE_HOST_REGISTER=1
    echo "NCCL OFI plugin: ${NCCL_PLUGIN_DIR}"
  else
    echo "NCCL_PLUGIN_DIR=${NCCL_PLUGIN_DIR} holds no libnccl-net*.so" >&2
    exit 1
  fi
else
  echo "NCCL_PLUGIN_DIR unset: expect TCP socket fallback, not Slingshot." >&2
fi

# Reports the transport that was actually selected.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# Slingshot fork safety, should an MPI-style fork error appear.
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

# ---------------------------------------------------------------------------
# Case setup. The runner uses the conjugate gradient config by default.
# `cores_x/cores_y/cores_z` can be changed in the config file. Use absolute
# paths only. 40 cycles of 200 steps is the measured
# reference: 8000 steps, whose second half gives the 4000 Strouhal samples.
# ---------------------------------------------------------------------------
export CONFIG="${CONFIG:-${HERE}/cylinder_multigrid.pbtxt}"
export NUM_CYCLES="${NUM_CYCLES:-40}"
export NUM_STEPS="${NUM_STEPS:-200}"
# In cycles, not steps. The final checkpoint is written regardless.
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
# Per job, and on storage every node can see: each bring-up step writes its own
# `wake_probe.npy`, and that is the number being compared between them.
export OUTPUT="${OUTPUT:-${SCRATCH:-${HERE}}/SOLUT_cylinder_${SLURM_JOB_ID}}"

# Which code is about to run: an installed copy can shadow the repo, so print
# the resolved path rather than assume the two agree.
echo "job ${SLURM_JOB_ID}: ${SLURM_NNODES} node(s) x ${SLURM_NTASKS_PER_NODE} task(s) = ${SLURM_NTASKS} process(es)"
echo "nodes: ${SLURM_STEP_NODELIST:-${SLURM_JOB_NODELIST}}"
echo "config: ${CONFIG}"
echo "output: ${OUTPUT}"
python -c 'from swirl_lm.jax.core import driver; print("driver:", driver.__file__)'
python -c 'import jax; print("jax:", jax.__version__)'

srun bash "${HERE}/run_cylinder.sh"
```

Two parts of that script are environment-specific rather than Slurm-specific.
The pip `nvidia-*-cu12` wheels ship their own CUDA, and NERSC warns that
loading the `cudatoolkit` module on top of a conda CUDA install conflicts with
it, so the module is unloaded. And Slingshot-11 (Cassini) NICs do not speak IB
verbs, so without the NCCL OFI plugin NCCL falls back to TCP sockets: correct,
and several times slower. Use `NCCL_DEBUG=INFO`, which reports the transport
that was actually selected.

## Validating the simulation

At the end of every run, the simulation will report a Strouhal number (St)
of the downstream von Karman street (requires `Re > 40` to develop). By
default, the initial 50% of the total simulation steps will be discarded
to compute `St` over an established flow field without initialization
effects. A cache file is also written as `wake_probe.npy` to the
`checkpoint_dir` for later analysis.
At `Re = 100`, the reported Strouhal number in Ghias et al. (2007) ranges
between 0.162 and 0.168 (see Fig. 12, p. 546).

### Measured results

For reference, Strouhal numbers at different solver settings are provided
in the table below. All simulations were run at `Re = 100` on the same mesh,
640 x 400 x 4 points decomposed over 2x2x1 GPUs (A100), with
`num_sub_iterations: 3` (cluster: NERSC Perlmutter, Lawrence Berkeley National
Laboratory, interactive queue).
Times are `compute_time` per 200-step cycle at steady state; cycle 0 carries
the JIT compile and is excluded. The Poisson column is the total minus the
6.49 s that is not the pressure solve. St is computed from 4000 samples,
about ten shedding periods.

| Solver | Levels | `n_smooth` | Iterations | s / 200 steps | vs CG | Poisson | St |
|---|---|---|---|---|---|---|---|
| Conjugate gradient | - | - | 100 | 33.80 | 1.00x | 27.31 | 0.165 |
| Multigrid | 5 | 2 | 8 V-cycles | 144-150 | 0.23x | 138-144 | 0.1662 |
| Multigrid | 5 | 2 | 5 V-cycles | 92.54 | 0.37x | 86.05 | (not computed) |
| Multigrid | 5 | 2 | 1 V-cycle | 23.70 | 1.43x | 17.21 | 0.1661 |
| Multigrid | 3 | 2 | 1 V-cycle | 13.88 | 2.44x | 7.39 | 0.1659 |
| Multigrid | 3 | 1 | 1 V-cycle | 9.91 | 3.41x | 3.42 | 0.1659 |
| Multigrid | 2 | 2 | 1 V-cycle | 9.82 | 3.44x | 3.33 | 0.1658 |
| **Multigrid** | **2** | **1** | **1 V-cycle** | **7.42** | **4.56x** | **0.93** | **0.1659** |

The last row is obtained by running the simulation with
`cylinder_multigrid.pbtxt`.
At that setting the pressure solve is 12% of the step, against 81% for the
validated conjugate gradient setting, and the remaining headroom over a free
Poisson solve is 1.14x.

**Do not read the St column as a ranking.** One FFT bin is
`1/T = 1/(4000 x 0.015 s) = 0.0167` in St, and the spread across every row
above is 0.0011, which is fifteen times smaller than the resolution.
Those values are one measurement, and their ordering comes from how the peak
is interpolated rather than from the solver. The table shows that any
multigrid depth from 2 to 5 levels accurately reproduces the reported
Strouhal number in Ghias et al. Therefore, one may prefer a lower multigrid
level count for faster execution in this specific case without degrading
accuracy.

Note that few iterations with the Jacobi solver are insufficient and cannot
reproduce the reported Strouhal number as the solution is under-converged:
at 10, 20 and 40 Jacobi iterations, the computed Strouhal numbers are
0.1921, 0.1865 and 0.1814, respectively.
Specifically, at 10 Jacobi iterations, the Strouhal number is 16% above
the converged result (1.6 FFT bins), where the whole spread in the table
above is 0.07 of a bin.

## Logging the residual of the Poisson solver

To log the residual of the Poisson solver, `cylinder_main.py` needs
to be called with the additional argument `--log_poisson_residual`.
Each sub-iteration then reports the relative and absolute residual, the
right-hand side norm, and what the pressure correction did to the
divergence. It costs one extra operator application per solve.
The quantities are reduced across devices, so every process reports the
same numbers: in a multi-node run each line appears once per process, and
the repeats are not separate measurements.
**Note:** Logging is done on the CPU even when the simulation is
executed on GPUs. Therefore, `JAX_PLATFORMS` needs to be extended
as follows (note the appended `,cpu`):

```bash
# Log the residual of the Poisson solver
JAX_PLATFORMS=cuda,cpu python -m swirl_lm.jax.examples.cylinder.cylinder_main \
  --config_filepath=swirl_lm/jax/examples/cylinder/cylinder_cg.pbtxt \
  --num_steps=100 --num_cycles=10 --checkpoint_interval=10 \
  --checkpoint_dir=./SOLUT_CG \
  --reynolds_number=100 --log_poisson_residual
```

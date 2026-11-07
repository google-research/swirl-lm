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
640 x 400 x 4 points decomposed over 2x2x1 GPUs (A100), with `num_sub_iterations: 3`
(cluster: NERSC Perlmutter, Lawrence Berkeley National Laboratory,
interactive queue).
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

The last row is obtained by running the simulation with `cylinder_multigrid.pbtxt`.
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

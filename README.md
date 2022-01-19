# Swirl-LM: Computational Fluid Dynamics in TensorFlow

Swirl-LM is a computational fluid dynamics (CFD) simulation framework that is
accelerated by the Tensor Processing Unit (TPU). It solves the three dimensional
variable-density Navier-Stokes equation using a low-Mach approximation, and the
governing equations are discretized by a finite-difference method on a
collocated structured mesh. It is implemented in TensorFlow.

## Citation

If you extend or use this package in your work, please cite the [paper](https://arxiv.org/pdf/2108.11076)
as

```
@misc{
      title={A TensorFlow Simulation Framework for Scientific Computing of
Fluid Flows on Tensor Processing Units},
      author={Wang, Qing and Ihme, Matthias and Chen, Yi-Fan and Anderson, John},
      year={2021},
      eprint={2108.11076},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph}
}
```

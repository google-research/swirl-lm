# Swirl-LM: Computational Fluid Dynamics in TensorFlow

Swirl-LM is a computational fluid dynamics (CFD) simulation framework that is
accelerated by the Tensor Processing Unit (TPU). It solves the three dimensional
variable-density Navier-Stokes equation using a low-Mach approximation, and the
governing equations are discretized by a finite-difference method on a
collocated structured mesh. It is implemented in TensorFlow.

## Citation

If you extend or use this package in your work, please cite the
[paper](https://www.sciencedirect.com/science/article/abs/pii/S0010465522000108)
as

```
@ARTICLE{Wang2022-ln,
  title     = "A {TensorFlow} simulation framework for scientific computing of
               fluid flows on tensor processing units",
  author    = "Wang, Qing and Ihme, Matthias and Chen, Yi-Fan and Anderson,
               John",
  abstract  = "A computational fluid dynamics (CFD) simulation framework for
               fluid-flow prediction is developed on the Tensor Processing Unit
               (TPU) platform. The TPU architecture is featured with
               accelerated dense matrix multiplication, large high bandwidth
               memory, and a fast inter-chip interconnect, making it attractive
               for high-performance scientific computing. The CFD framework
               solves the variable-density Navier-Stokes equation using a
               low-Mach approximation, and the governing equations are
               discretized by a finite-difference method on a collocated
               structured mesh. It uses the graph-based TensorFlow as the
               programming paradigm. The accuracy and performance of this
               framework is studied both numerically and analytically,
               specifically focusing on effects of TPU-native single precision
               floating point arithmetic. The algorithm and implementation are
               validated with canonical 2D and 3D Taylor-Green vortex
               simulations. To demonstrate the capability for simulating
               turbulent flows, simulations are conducted for two
               configurations, namely decaying homogeneous isotropic turbulence
               and a turbulent planar jet. Both simulations show good
               statistical agreement with reference solutions. The performance
               analysis shows a linear weak scaling and a superlinear strong
               scaling up to a full TPU v3 pod with 2048 cores.",
  journal   = "Comput. Phys. Commun.",
  publisher = "North-Holland",
  volume    =  274,
  pages     = "108292",
  month     =  may,
  year      =  2022,
  issn = {0010-4655},
  doi = {https://doi.org/10.1016/j.cpc.2022.108292},
  url = {https://www.sciencedirect.com/science/article/pii/S0010465522000108},
  keywords  = "Tensor processing unit; TensorFlow; Computational fluid
               dynamics; High performance computing"
}

```

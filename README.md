# CUDA_Parallel_Programming
Course _CUDA Parallel Programming_ from NTU.

## Quick Search and Tags

* Assignments

|     Folder     |                              Tags                              |                      CUDA                     |
|:--------------:|:--------------------------------------------------------------:|:---------------------------------------------:|
|  Problem Set 1 |                Vector Addition, Matrix Addition                |                                               |
|  Problem Set 2 |           Find Maximum, Dot Product, Vector Addition           | Shared Memory, Parallel Reduction, Multi-GPUs |
|  Problem Set 3 |                        Laplace Equation                        |                 Texture Memory                |
|  Problem Set 4 |                           Dot Product                          | Shared Memory, Parallel Reduction, Multi-GPUs |
|  Problem Set 5 |                         Heat Diffusion                         |                   Multi-GPUs                  |
| Problem Set 6  |                    Random Number, Histogram                    |                 Shared Memory                 |
| Problem Set 7  | Monte Carlo Integration, Simple Sampling, Metropolis Algorithm |                     cuRAND                    |
| Problem Set 8  |                        Multigrids Method                       |                                               |
| Problem Set 9  |                           Ising Model                          |                   Multi-GPUs                  |
| Problem Set 10 |                        Poisson Equation                        |                     cuFFT                     |

## Assignments
Problem sets are named as `hwX_2020.pdf`, my reports are named as `r08244002_psX.pdf`.

### Problem Set 1
* Vector Addition
* Matrix Addition

### Problem Set 2
* Find Maximum
  * Parallel Reduction
* Dot Product
  * Parallel Reduction
* Vector Addition
  * Add two arbitrary large vector together.
  * Work with multiple GPUs.

### Problem Set 3
* Laplace Equation on 2D lattice
  * Solve laplace equation on the lattice.
    * CPU only
    * GPU with device memory
    * GPU with texture memory
* Laplace Equation on 2D lattice with boundary condition
  * Solve laplace equation on the lattice.
    * CPU only
    * GPU with device memory
    * GPU with texture memory
* Laplace Equation on 3D lattice with boundary condition
  * Solve laplace equation on the lattice.
    * CPU only
    * GPU with device memory
    * GPU with texture memory

### Problem Set 4
* Dot Product
  * Parallel Reduction
  * Work with multiple GPUs.

### Problem Set 5
* Heat Diffusion
  * Solve for the thermal equilibrium temperature distribution on a square plate.
    * Work with multiple GPUs.

### Problem Set 6
* Random Number
  * Generate random numbers with exponential decay distribution e<sup>-x</sup>.
* Histogram
  * Collect a series of data, and build a histogram.
    * CPU
    * GPU with global memory
    * GPU with shared memory

### Problem Set 7
* Monte Carlo Integration
  * Do integration in 10-dim
    * CPU only
      * Simple Sampling
      * Metropolis Algorithm
    * Work with multiple GPUs.
      * Simple Sampling
      * Metropolis Algorithm

### Problem Set 8
* Multigrids Method
  * Project ![link](https://github.com/cindytsai/multigrid_poisson_solver)

### Problem Set 9
* 2D Ising Model on a Torus
  * Simulation of 2D Ising model on a torus
    * CPU only
    * Work with multiple GPUs.
  * Do measurements and estimate the errors
    * Energy
    * Magnetization

### Problem Set 10
* Poisson Equation
  * Solve Poisson Equation using FFT
    * GPU, cuFFT

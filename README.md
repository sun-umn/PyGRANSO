# PyGRANSO

Toward a user-friendly and scalable numerical package for nonsmooth, nonconvex, constrained optimization

## Table of contents
* [General Information](#general-information)
* [Dependencies](#dependencies)
* [Updating Summary](#updating-summary)
* [Usage](#usage)
* [References](#references)
* [Contact](#contact)

## General Information

The goals of our high-level project are: 

1) Build scalable and user-friendly numerical optimization packages for solving general nonconvex, nonsmooth, constrained optimization problems. This software package will be distributed and actively maintained for wider community usage. 

2) Perform a computational and theoretical study of constrained deep learning. Current deep learning frameworks, albeit powerful, cannot integrate physical constraints or prior knowledge. This is because the standard software frameworks such as TensorFlow and PyTorch cannot handle constrained problems. The proposed software package will enable a systematic study of the computational and theoretical foundations of constrained deep learning. 

GRANSO is the first numerical optimization package that can handle general nonconvex, nonsmooth, constrained optimization problems based on MATLAB. The package is very stable and produces meaningful results when other carefully crafted solvers fail [4]. However, several limitations of GRANSO preclude its potential broad deployment by general users: 

1) It only allows vector variables but not matrices or tensors, while the latter two are common computational units in modern optimization problems such as machine/deep learning.   

2) The default MATLAB quadratic programming solver struggles to scale up to medium- to large-scale problems, which is a bottleneck for scalability. 

3) GRANSO requires deriving analytic subgradients for the objective and constraint functions, which is challenging and even infeasible, especially in deep learning.  

4) MATLAB that GRANSO is written in is a proprietary programming language and entails considerable license fees for researchers and developers. 

We will tackle the severe limitations of GRANSO, and build a user-friendly and scalable numerical optimization package by revamping several key components of GRANSO. We will also obtain preliminary results on constrained deep learning. Main tasks include: 

1) User-friendly MATLAB GRANSO: Allow matrix and tensor optimization variables in the solver. This is relatively straightforward.  

2) Scalable MATLAB GARNSO: Replace the current MATLAB builtin quadratic solver with the QPALM package. The QPALM package is a great alternative to MATLAB’s slow quadratic solver,  and has consistently and significantly outperformed popular commercial solvers in terms of speed and scalability. Besides the scalability advantage, QPALM supports multiple programming languages, particularly Python, which we plan to use to revamp GRANSO. 

3) User-friendly and scalable Python GRANSO: Central to the project, we will revamp GRANSO and translate it into Python, for a couple of critical reasons: a) modern Python computational frameworks, such as Jax and Pytorch, enable highly optimized and parallelizable matrix/tensor computations that take the full advantage of modern massively parallel hardware, e.g., GPUs, and b) Jax and Pytorch provide first-rate autodifferentiation capabilities, removing the pain of deriving analytic subgradients. Powerful matrix/tensor computation and autodifferentiation will substantially boost the usability and scalability of GRANSO for non-experts, and turn the impossible, e.g., constrained deep learning, into possible. We will translate GRANSO into Jax or Pytorch, or maybe both. QPALM will be interfaced through its Python binding. 

4) Preliminary test on constrained deep learning: We will use the solver to explore constrained deep learning problems arising in robust image recognition, which is often formulated as a constrained min-max optimization problem.

## Dependencies

Python 3.7.10
numpy-1.20.3 
scipy-1.6.2
pytorch-1.5.0
gurobi

## Updating Summary

mat2vec.m: a new function that could 

gransoOptions.m: update the quadprog options to QPALM options.

./private/solveQP.m: replace the MATLAB default solver (quadprog) with QPALM.

./private/qpSteeringStrategy.m: provided the input args required by QPALM solver.

./private/qpTerminationCondition.m: provided the input args required by QPALM solver.

./example/*: examples of several optimization problems.

./QPALM4GRANSO: the modified QPALM solver for the GRANSO package.

## Usage

1. Download the latest version of revamped GRANSO.

2. Install the QPALM solver: https://github.com/sun-umn/qpalm4granso. Follow the instructions in README.

3. Add the revamped GRANSO package to the search path of MATLAB.

4. Check the documentation by using command *help*, e.g., 
```bash
help granso
help gransoOptions
help mat2vec
help private/solveQP
help private/qpTerminationCondition
```

## References
Ben Hermans, Andreas Themelis, and Panagiotis Patrinos. "QPALM: a Newton-type proximal augmented Lagrangian method for quadratic programs." 2019 IEEE 58th Conference on Decision and Control (CDC). IEEE, 2019.

Frank E. Curtis, Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.

## Contact
Created by Buyun Liang [liang664@umn.edu] (https://www.linkedin.com/in/buyun-liang/) - feel free to contact me if you have any questions!

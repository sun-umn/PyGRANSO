# PyGRANSO

PyGRANSO: a Python numerical package using GRadient-based Algorithm for Non-Smooth Optimization

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

## Dependencies

```bash
conda install -c oxfordcontrol osqp

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install gurobipy
```

osqp

Python 3.7.10

numpy-1.20.3

scipy-1.6.2

pytorch-1.5.0

gurobipy

## Updating Summary

TODO

## Usage

1. TODO

2. TODO

```bash
TODO
```

## References
Ben Hermans, Andreas Themelis, and Panagiotis Patrinos. "QPALM: a Newton-type proximal augmented Lagrangian method for quadratic programs." 2019 IEEE 58th Conference on Decision and Control (CDC). IEEE, 2019.

Frank E. Curtis, Tim Mitchell, and Michael L. Overton. "A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles." Optimization Methods and Software 32.1 (2017): 148-181.

## Contact
Created by Buyun Liang [liang664@umn.edu] (https://www.linkedin.com/in/buyun-liang/) - feel free to contact me if you have any questions!

# Change Log

PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation

## Version: 1.2.0 --- 2022-07-26

Description: major fixes and improvements on LBFGS. 

**Fixed** 
- Reducing memory usage for LBFGS. Now PyGRANSO can solve problem with ~15k parameters by using 14 GB memory. 
- Update example: ortho RNN with max folding and orthonormal initialization.
- Allow high precision for QP solver.
- Allow part of optimization variables not showing up in objective (see SVM example).
- Fixed Code 12: terminated with steering failure.
- Fixed stationary failure: try different stationarity calculation, or set stationarity measure to be inf if encounter numerical issue

**Added**
- Reorganize and add examples: perceptual/lp norm attack on ImageNet images. trace optimization with orthogonal constraints; unconstrained deep learning with LeNet5; logistic regression.



## Version: 1.1.0 --- 2022-02-20

Description: major fixes and improvements. 

**Fixed** 
- Avoid gradient accumulating in deep learning problem; 
- Prevent memory leak problem when using torch tensor. See ex6 perceptual attack.

**Changed**
- Update format of user-defined variables when using `pygranso` interface. 

**Packaging**
- Publish pygranso package on [Pypi](https://pypi.org/project/pygranso/).

**Added**
- Add examples: ex 10 dictionary learning with torch.nn module; ex 11 orthogonal recurrent neural networks.

## Version: 1.0.0 --- 2021-12-27

Description: initial public release of PyGRANSO. 

**Main features:** auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages. Multiple new examples added.

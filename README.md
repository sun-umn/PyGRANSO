# NCVX

![Example screenshot](./ncvx_logo.png)

NCVX: A User-Friendly and Scalable Package for Nonconvex Optimization in Machine Learning.

Please check https://ncvx.org for detailed instructions (introduction, installation, settings, examples...).

## Brief Introduction

Optimization for nonconvex problems, especially those nonsmooth and with constraints, is an essential part of machine learning and deep learning, which is hard to reliably solve without optimization expertise. Also existing general purpose non-convex optimization software packages usually cannot handle non-smoothness and require analytical gradients, which raised the technical barrier for general users. GRANSO is the first numerical optimization package that can handle this type of problems. However, it has some limitations such as lack of auto-differentiation and GPU acceleration, which preclude the potential broad deployment by general users. Thus, we introduce NCVX (NonConVeX), a user-friendly and scalable python package for NCVX optimization, which is revamped and translated from the GRANSO package. In order to lower the barriers to general users and solve modern machine/deep learning problems, we introduce several main features including auto-differentiation, GPU acceleration, tensor input, scalable QP solver and open-source dependencies.

## Update Logs

v1.1.1-alpha: Rename the package from “PyGRANSO” to “NCVX”; multiple examples added: unconstrained DL, feasibility problem, sphere manifold.

v1.1.0-alpha: Cleaned code, added L-BFGS, updated tutorials and documentation.

v1.0.2-alpha: Update installation guide for Linux and windows users

v1.0.1-alpha: Update contirbutions, limitations and acknowledgement sections in docs.

v1.0.0-alpha: Initial release of PyGRANSO. Main features: Python translation, autodifferentiation, GPU-support with PyTorch, matrix/tensor inputs, more powerful solver and several new settings to avoid numerical issues in deep learning problem.

## Acknowledgements

Buyun Liang was supported by the UMII Seed Grant Program (https://research.umn.edu/units/umii).

## Contact
Codes written by Buyun Liang (https://buyunliang.org). Questions or bug reports please send email to Buyun Liang, liang664@umn.edu.

Thanks to bug reporters: 

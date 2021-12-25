# PyGRANSO

![PyGRANSO](./PyGRANSO_logo_banner.png)


PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation

Please check https://ncvx.org for detailed instructions (introduction, installation, settings, examples...).

## Brief Introduction

Optimizing nonconvex (NCVX) problems, especially those nonsmooth and constrained, is an essential part of machine learning and deep learning. But it is hard to reliably solve this type of problems without optimization expertise. Existing general-purpose NCVX optimization packages are powerful, but typically cannot handle nonsmoothness. GRANSO is among the first packages targeting NCVX, nonsmooth, constrained problems. However, it has several limitations such as the lack of auto-differentiation and GPU acceleration, which preclude the potential broad deployment by non-experts. To lower the technical barrier for the machine learning community, we revamp GRANSO into a user-friendly and scalable python package named PyGRANSO, featuring auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages. As a highlight, PyGRANSO can solve general constrained deep learning problems, the first of its kind.

## Installation

Installing NCVX is simple. Here is a step-by-step instruction:

1. Download the latest version of NCVX on GitHub (https://github.com/sun-umn/NCVX)

2. Change the name and prefix in environment.yml.

    (GPU and CPU) Simply run:

        conda env create -f environment_cuda.yml
        conda activate ncvx_cuda_env

    (CPU only) Simply run:

        conda env create -f environment_cpu.yml
        conda activate ncvx_cpu_env

3. (GPU and CPU) Run test to make sure the dependency installation is correct:

        python test_cuda.py

    (CPU only) Run test to make sure the dependency installation is correct:

        python test_cpu.py

4. Check the Examples section (https://ncvx.org/examples) in the documentation website to get started.

## Dependencies
    Python-3.9.7

    numpy-1.20.3

    scipy-1.7.1

    pytorch-1.9.0

    osqp-0.6.2

    Jupyter Notebook-6.4.5

## Update Logs

v1.0.0: Initial release of PyGRANSO. Main features: auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages. Multiple examples added: unconstrained DL, feasibility problem, sphere manifold.

## Acknowledgements

We would like to thank the GRANSO developers. This work was supported by UMII Seed Grant Program and NSF CMMI 2038403.

## Citation

If you publish work that uses or refers to PyGRANSO, please cite both
PyGRANSO and GRANSO paper:

*[1] Buyun Liang, Tim Mitchell and Ju Sun.
    NCVX: A User-Friendly and Scalable Package for Nonconvex
    Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).*
    Available at https://arxiv.org/abs/2111.13984
    
*[2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton
    A BFGS-SQP method for nonsmooth, nonconvex, constrained
    optimization and its evaluation using relative minimization
    profiles, Optimization Methods and Software, 32(1):148-181, 2017.*
    Available at https://dx.doi.org/10.1080/10556788.2016.1208749    

BibTex:

    @article{liang2021ncvx,
        title={NCVX: A User-Friendly and Scalable Package for Nonconvex 
        Optimization in Machine Learning}, 
        author={Buyun Liang, Tim Mitchell and Ju Sun},
        year={2021},
        eprint={2111.13984},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    
    @article{curtis2017bfgs,
        title={A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles},
        author={Frank E. Curtis, Tim Mitchell, and Michael L. Overton},
        journal={Optimization Methods and Software},
        volume={32},
        number={1},
        pages={148--181},
        year={2017},
        publisher={Taylor \& Francis}
    }

## Contact
For questions or bug reports, please either:
- raise issues in the PyGRANSO repository (https://github.com/sun-umn/PyGRANSO/) or
- send an email to:
  - Buyun Liang (*liang664 an_at_symbol umn a_dot_symbol edu*)
  - Tim Mitchell (*tim an_at_symbol timmitchell a_dot_symbol com*)
  - Ju Sun (*jusun an_at_symbol umn a_dot_symbol edu*)

Thanks to bug reporters: 

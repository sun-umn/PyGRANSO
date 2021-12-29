# PyGRANSO

![PyGRANSO](./PyGRANSO_logo_banner.png)


PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation

Please check https://ncvx.org/PyGRANSO/ for detailed instructions (introduction, installation, settings, examples...).

## Brief Introduction

Optimizing nonconvex (NCVX) problems, especially those nonsmooth and constrained, is an essential part of machine learning and deep learning. But it is hard to reliably solve this type of problems without optimization expertise. Existing general-purpose NCVX optimization packages are powerful, but typically cannot handle nonsmoothness. GRANSO is among the first packages targeting nonsmooth NCVX problems with nonsmooth constraints. However, it has several limitations such as the lack of auto-differentiation and GPU acceleration, which preclude the potential broad deployment by non-experts. To lower the technical barrier for the machine learning community, we revamp GRANSO into a user-friendly and scalable python package named PyGRANSO, featuring auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages. As a highlight, PyGRANSO can solve general constrained deep learning problems, the first of its kind.

## Installation

**NOTE: The installation process is tested on Ubuntu 20.04**

Installing PyGRANSO is simple. Here is a step-by-step instruction:

0. Prerequisite: install Anaconda on your system (recommend: Ubuntu 20.04). Detailed guidance: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

1. Download the latest version of PyGRANSO on GitHub (https://github.com/sun-umn/PyGRANSO)

2. Change the name and prefix in environment.yml.

    (GPU and CPU) Simply run:

        conda env create -f environment_cuda.yml
        conda activate pygranso_cuda_env

    (CPU only) Simply run:

        conda env create -f environment_cpu.yml
        conda activate pygranso_cpu_env

3. (GPU and CPU) Run test to make sure the dependency installation is correct:

        python test_cuda.py

    (CPU only) Run test to make sure the dependency installation is correct:

        python test_cpu.py

4. Check the Examples section (https://ncvx.org/PyGRANSO/examples) in the documentation website to get started.

## Dependencies
    Python-3.9.7

    numpy-1.20.3

    scipy-1.7.1

    pytorch-1.9.0

    osqp-0.6.2

    Jupyter Notebook-6.4.5

## Change Logs

Please check [CHANGELOG.md](./CHANGELOG.md) in the main folder.

## Notes on Documentation

PyGRANSO is a PyTorch-enabled port of GRANSO with auto-differentiation, and some of its documentations are in the MATLAB format. Please check the notes for the data structures used in PyGRANSO.

* `struct` refers to `pygransoStruct`, which is a dummy class.
* Vector and matrix refer to PyTorch tensor with *(n,1)* and *(m,n)* dimension 


## Acknowledgements

We would like to thank [Frank E. Curtis](https://coral.ise.lehigh.edu/frankecurtis/) and [Michael L. Overton](https://cs.nyu.edu/~overton/) for their involvement in creating the BFGS-SQP algorithm that is 
implemented in the software package [GRANSO](http://www.timmitchell.com/software/GRANSO). This work was supported by UMII Seed Grant Program and NSF CMMI 2038403.

## Citation

If you publish work that uses or refers to PyGRANSO, please cite the following two papers,
which respectively introduced PyGRANSO and GRANSO:

*[1] Buyun Liang, Tim Mitchell, and Ju Sun,
    NCVX: A User-Friendly and Scalable Package for Nonconvex
    Optimization in Machine Learning, arXiv preprint arXiv:2111.13984 (2021).*
    Available at https://arxiv.org/abs/2111.13984

*[2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
    A BFGS-SQP method for nonsmooth, nonconvex, constrained
    optimization and its evaluation using relative minimization
    profiles, Optimization Methods and Software, 32(1):148-181, 2017.*
    Available at https://dx.doi.org/10.1080/10556788.2016.1208749  

BibTex:

    @article{liang2021ncvx,
        title={NCVX: A User-Friendly and Scalable Package for Nonconvex 
        Optimization in Machine Learning}, 
        author={Buyun Liang, Tim Mitchell, and Ju Sun},
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

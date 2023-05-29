# PyGRANSO

![PyGRANSO](./PyGRANSO_logo_banner.png)


PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation

Please check https://ncvx.org for detailed instructions (introduction, installation, settings, examples...).

PyGRANSO is AGPL v3.0 licensed, but it also contains a small portion of GPL code.  Please see the LICENSE_INFO folder for more information.

## Brief Introduction

Optimizing nonconvex (NCVX) problems, especially nonsmooth and constrained ones, is an essential part of machine learning. However, it can be hard to reliably solve such problems without optimization expertise. Existing general-purpose NCVX optimization packages are powerful but typically cannot handle nonsmoothness. GRANSO is among the first optimization solvers targeting general nonsmooth NCVX problems with nonsmooth constraints, but, as it is implemented in MATLAB and requires the user to provide analytical gradients, GRANSO is often not a convenient choice in machine learning (especially deep learning) applications. To greatly lower the technical barrier, we introduce a new software package called NCVX, whose initial release contains the solver PyGRANSO, a PyTorch-enabled port of GRANSO incorporating auto-differentiation, GPU acceleration, tensor input, and support for new QP solvers. NCVX is built on freely available and widely used open-source frameworks, and as a highlight, can solve general constrained deep learning problems, the first of its kind.

## Installation

Installing PyGRANSO is simple. Here is a step-by-step instruction:

1. Install [Python >= 3.9](https://www.python.org/)

2. Get the most recent PyGRANSO package (including examples and requirements file):

        git clone https://github.com/sun-umn/PyGRANSO.git
        cd PyGRANSO

3.  Install PyGRANSO solver from PyPI:

        pip install git+https://github.com/sun-umn/PyGRANSO.git

4. Install Dependencies from PyPI: 

    OS: **Linux** OR **Windows**; Compute Platform: **CUDA**:

        pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html

    OS: **Linux**; Compute Platform: **CPU**:

        pip install -r requirements_linux_cpu.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

    OS: **Mac** OR **Windows**; Compute Platform: **CPU**:

        pip install -r requirements_cpu.txt
    
5. (CUDA) Run test to make sure the dependency installation is correct:

        python test_cuda.py

    (CPU) Run test to make sure the dependency installation is correct:

        python test_cpu.py

6. Check the [example folder](./examples) in the source code or [example section](https://ncvx.org/examples) on the documentation website to get started.

## Dependencies
    Python-3.9.7

    numpy-1.20.3

    scipy-1.7.1

    pytorch-1.9.0

    osqp-0.6.2

    Jupyter Notebook-6.4.5

    gurobi

## Change Logs

Please check [CHANGELOG.md](./CHANGELOG.md) in the main folder.

## Notes on Documentation

PyGRANSO is a PyTorch-enabled port of GRANSO with auto-differentiation, and some of its documentation uses MATLAB conventions. In the PyGRANSO source code docstrings, please note that:

* `struct` refers to `pygransoStruct`, which is a simple class that users need to use for specifying their problems and options.
* Vector and matrix refer to PyTorch tensor with *(n,1)* and *(m,n)* dimension, respectively. 


## Acknowledgements

We would like to thank [Frank E. Curtis](https://coral.ise.lehigh.edu/frankecurtis/) and [Michael L. Overton](https://cs.nyu.edu/~overton/) for their involvement in creating the BFGS-SQP algorithm that is 
implemented in the software package [GRANSO](http://www.timmitchell.com/software/GRANSO). This work was supported by UMII Seed Grant Program and NSF CMMI 2038403.

## Citation

If you publish work that uses or refers to PyGRANSO, please cite the following two papers,
which respectively introduced PyGRANSO and GRANSO:

*[1] Buyun Liang, Tim Mitchell, and Ju Sun, NCVX: A General-Purpose Optimization Solver for Constrained Machine and Deep Learning, arXiv preprint arXiv:2210.00973 (2022). Available at https://arxiv.org/abs/2210.00973*

*[2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
    A BFGS-SQP method for nonsmooth, nonconvex, constrained
    optimization and its evaluation using relative minimization
    profiles, Optimization Methods and Software, 32(1):148-181, 2017.*
    Available at https://dx.doi.org/10.1080/10556788.2016.1208749  

BibTex:

    @article{liang2021ncvx,
        title={{NCVX}: {A} User-Friendly and Scalable Package for Nonconvex 
        Optimization in Machine Learning}, 
        author={Buyun Liang, Tim Mitchell, and Ju Sun},
        year={2021},
        eprint={2111.13984},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    
    @article{curtis2017bfgssqp,
        title={A {BFGS-SQP} method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles},
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
- raise issues in the [PyGRANSO repository](https://github.com/sun-umn/PyGRANSO/) or
- send an email to our [NCVX PyGRANSO forum](https://groups.google.com/a/umn.edu/g/ncvx): ncvx@umn.edu

Main authors:

  - [Buyun Liang](https://buyunliang.org/) (*byliang an_at_symbol seas a_dot_symbol upenn a_dot_symbol edu*)
  - [Tim Mitchell](http://www.timmitchell.com/) (*tim an_at_symbol timmitchell a_dot_symbol com*)
  - [Ju Sun](https://sunju.org/) (*jusun an_at_symbol umn a_dot_symbol edu*)

Thanks to other contributors and bug reporters: 

- [Hengyue Liang](https://hengyuel.github.io/): Applied PyGRANSO on adversarial robustness problems. Tested PyGRANSO across multiple platforms. Debugged several functions.

- Wenjie Zhang: Tested practical techniques on various constrained deep learning problems, which can be used to accelerate the convergence of PyGRANSO.

- [Ryan de Vera](https://github.com/rydevera3): Applied PyGRANSO on neural topology optimization problems.

- Yash Travadi: Applied PyGRANSO on imbalanced classification problems.

- [Ying Cui](https://sites.google.com/site/optyingcui/home): Advised the adversarial robustness problems.

- [Chen Jiang](https://github.com/shoopshoop): Tested perceptual attack example (ex6). Tested PyGRANSO on Win10. Debugged updatePenaltyParameter function.


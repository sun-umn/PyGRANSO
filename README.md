# PyGRANSO

![PyGRANSO](./PyGRANSO_logo_banner.png)


PyGRANSO: A PyTorch-enabled port of GRANSO with auto-differentiation

Please check https://ncvx.org for detailed instructions (introduction, installation, settings, examples...).

PyGRANSO is AGPL v3.0 licensed, but it also contains a small portion of GPL code.  Please see the LICENSE_INFO folder for more information.

## Brief Introduction

Optimizing nonconvex (NCVX) problems, especially nonsmooth and constrained ones, is an essential part of machine learning. However, it can be hard to reliably solve such problems without optimization expertise. Existing general-purpose NCVX optimization packages are powerful but typically cannot handle nonsmoothness. GRANSO is among the first optimization solvers targeting general nonsmooth NCVX problems with nonsmooth constraints, but, as it is implemented in MATLAB and requires the user to provide analytical gradients, GRANSO is often not a convenient choice in machine learning (especially deep learning) applications. To greatly lower the technical barrier, we introduce a new software package called NCVX, whose initial release contains the solver PyGRANSO, a PyTorch-enabled port of GRANSO incorporating auto-differentiation, GPU acceleration, tensor input, and support for new QP solvers. NCVX is built on freely available and widely used open-source frameworks, and as a highlight, can solve general constrained deep learning problems, the first of its kind.

## Installation

PyGRANSO requires **Python 3.10+** (3.13+ recommended). Dependencies are managed via `pyproject.toml`.

### Option 1: Install from source with uv (recommended)

1. Install [uv](https://docs.astral.sh/uv/) (Python package installer and resolver).

2. Clone the repository and install in editable mode with dependencies:

        git clone https://github.com/sun-umn/PyGRANSO.git
        cd PyGRANSO
        uv sync

   Or install the package in your current environment:

        uv pip install -e .

### Option 2: Install with pip

1. Clone the repository:

        git clone https://github.com/sun-umn/PyGRANSO.git
        cd PyGRANSO

2. Create a virtual environment (recommended), then install:

        pip install -e .

   Dependencies (including PyTorch, SciPy, OSQP, etc.) will be installed from `pyproject.toml`.

### PyTorch with CUDA

By default, `pip` or `uv` may install a CPU-only build of PyTorch. For **GPU (CUDA) support**, install a CUDA-enabled PyTorch first, then install PyGRANSO:

    # Example: PyTorch with CUDA 12.x (check https://pytorch.org for your CUDA version)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    cd PyGRANSO
    pip install -e .   # or: uv pip install -e .

Set `opts.torch_device = torch.device("cuda")` when calling PyGRANSO to use the GPU.

### Verify installation

- **CPU:** `python test_cpu.py`
- **CUDA:** `python test_cuda.py`

Then check the [example folder](./examples) or the [example section](https://ncvx.org/examples) on the documentation website to get started.

## Dependencies

Core dependencies (see `pyproject.toml` for full list and versions):

- **Python** 3.10+ (3.13+ recommended)
- **PyTorch** >= 2.8.0 (install with CUDA support separately if needed)
- **SciPy** >= 1.16
- **OSQP** >= 1.0.4 (QP solver)
- **NumPy** (compatible with NumPy 2.x)
- **Jupyter** (for examples)
- **Gurobi** (optional; `gurobipy` for alternative QP solver)

Optional extras used by some examples: `pandas`, `polars`, `scikit-learn`, `wandb`, `optuna`, etc.

## Change Logs

Please check [CHANGELOG.md](./CHANGELOG.md) in the main folder.

## Additional documentation

The [docs/](./docs/) folder contains short notes on algorithm behavior and using PyGRANSO with common PyTorch features:

- **[Unconstrained problems and OSQP](./docs/UNCONSTRAINED_AND_OSQP.md)** — How unconstrained problems are handled, stationarity and gradient samples, and when CPU vs CUDA OSQP helps.
- **[Mixed precision and `torch.autocast`](./docs/MIXED_PRECISION.md)** — What to expect when using autocast (speed, memory, impact on PyGRANSO) and how to wrap it inside your `combined_fn`.
- **[`torch.compile`](./docs/TORCH_COMPILE.md)** — Impact of compiling your model or combined_fn with `torch.compile` (recompilation, speed, and recommendations).

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
  - Ryan Devera (*dever120 an_at_symbol umn a_dot_symbol edu*) — enhancements and maintainer of PyGRANSO 2.0

Thanks to other contributors and bug reporters:

- [Hengyue Liang](https://hengyuel.github.io/): Applied PyGRANSO on adversarial robustness problems. Tested PyGRANSO across multiple platforms. Debugged several functions.

- Wenjie Zhang: Tested practical techniques on various constrained deep learning problems, which can be used to accelerate the convergence of PyGRANSO.

- Ryan Devera: Applied PyGRANSO on neural topology optimization problems.

- Yash Travadi: Applied PyGRANSO on imbalanced classification problems.

- [Ying Cui](https://sites.google.com/site/optyingcui/home): Advised the adversarial robustness problems.

- [Chen Jiang](https://github.com/shoopshoop): Tested perceptual attack example (ex6). Tested PyGRANSO on Win10. Debugged updatePenaltyParameter function.


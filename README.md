# NCVX

![Example screenshot](./NCVX_logo_banner.png)

NCVX: A User-Friendly and Scalable Package for Nonconvex Optimization in Machine Learning.

Please check https://ncvx.org for detailed instructions (introduction, installation, settings, examples...).

## Brief Introduction

Optimizing nonconvex (NCVX) problems, especially those nonsmooth and constrained, is an essential part of machine learning and deep learning. But it is hard to reliably solve this type of problems without optimization expertise. Existing general-purpose NCVX optimization packages are powerful, but typically cannot handle nonsmoothness. GRANSO is among the first packages targeting NCVX, nonsmooth, constrained problems. However, it has several limitations such as the lack of auto-differentiation and GPU acceleration, which preclude the potential broad deployment by non-experts. To lower the technical barrier for the machine learning community, we revamp GRANSO into a user-friendly and scalable python package named NCVX, featuring auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages. As a highlight, NCVX can solve general constrained deep learning problems, the first of its kind.

## Update Logs

v1.1.1: Multiple examples added: unconstrained DL, feasibility problem, sphere manifold.

v1.1.0: L-BFGS Added.

v1.0.0: Initial release of NCVX. Main features: auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages.

## Acknowledgements

We would like to thank the GRANSO developers. This work was supported by UMII Seed Grant Program and NSF CMMI 2038403.

## Citation

    @article{liang2021ncvx,
        title={NCVX: A User-Friendly and Scalable Package for Nonconvex 
        Optimization in Machine Learning}, 
        author={Buyun Liang and Ju Sun},
        year={2021},
        eprint={2111.13984},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }


## Contact
Codes written by Buyun Liang (https://buyunliang.org). Questions or bug reports please send email to Buyun Liang, liang664@umn.edu.

Thanks to bug reporters: 

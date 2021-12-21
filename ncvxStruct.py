"""
    ncvxStruct: 
        Use dummy class to replace MATLAB struct

        If you publish work that uses or refers to NCVX, please cite both
        NCVX and GRANSO paper:

        [1] Buyun Liang, and Ju Sun. 
            NCVX: A User-Friendly and Scalable Package for Nonconvex 
            Optimization in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
            Available at https://arxiv.org/abs/2111.13984

        [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton 
            A BFGS-SQP method for nonsmooth, nonconvex, constrained 
            optimization and its evaluation using relative minimization 
            profiles, Optimization Methods and Software, 32(1):148-181, 2017.
            Available at https://dx.doi.org/10.1080/10556788.2016.1208749

        Change Log:
            
            Buyun Dec 20, 2021 (NCVX Version 1.0.0):
                ncvxStruct.py is introduced in NCVX


        For comments/bug reports, please visit the NCVX webpage:
        https://github.com/sun-umn/NCVX
"""


# strcu for options in NCVX
class Options:
    pass

# struct for Data
class Data:
    pass

# struct for sub validators 
class sub_validators_struct:
    pass

# struct for general settings
class GeneralStruct:
    pass
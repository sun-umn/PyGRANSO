def copyrightNotice():
    """
    copyrightNotice:
        This file returns a cell array of strings for printing NCVX's
        name, version, and copyright information.  Each string specifies
        one line. The file should be modified accordingly when ever the
        version number is increased.

    This file is part of NCVX: https://ncvx.org/ (doc) 
    https://github.com/sun-umn/NCVX (GitHub)

    This code is translated and revamped from the GRANSO code:
    GRANSO: GRadient-based Algorithm for Non-Smooth Optimization
    Version 1.6.4
    Licensed under the AGPLv3, Copyright (C) 2016-2020 Tim Mitchell
    http://www.timmitchell.com/software/GRANSO/

    If you publish work that uses or refers to NCVX, please cite the
    NCVX paper as well as the GRANSO paper:

    [1] Buyun Liang and Ju Sun 
        NCVX: A User-Friendly and Scalable Package for Nonconvex Optimization 
        in Machine Learning. arXiv preprint arXiv:2111.13984 (2021).
        Available at https://arxiv.org/abs/2111.13984

    [2] Frank E. Curtis, Tim Mitchell, and Michael L. Overton
        A BFGS-SQP method for nonsmooth, nonconvex, constrained
        optimization and its evaluation using relative minimization
        profiles, Optimization Methods and Software, 32(1):148-181, 2017.
        Available at https://dx.doi.org/10.1080/10556788.2016.1208749

    For comments/bug reports, please visit the NCVX GitHub page:
    https://github.com/sun-umn/NCVX

    NCVX is created by Buyun Liang https://buyunliang.org 
    Please contact liang664@umn.edu if you have any questions

    Change log: 
    Add copyrightNotice.py in v1.0.0

    =========================================================================
    |  NCVX: A User-Friendly and Scalable Package for Nonconvex             |
    |  Optimization in Machine Learning                                     |
    |                                                                       |
    |  Copyright (C) 2021 Buyun Liang                                       |
    |                                                                       |
    |  This file is part of NCVX.                                           |
    |                                                                       |
    |  NCVX is free software: you can redistribute it and/or modify         |
    |  it under the terms of the GNU Affero General Public License as       |
    |  published by the Free Software Foundation, either version 3 of       |
    |  the License, or (at your option) any later version.                  |
    |                                                                       |
    |  NCVX is distributed in the hope that it will be useful,              |
    |  but WITHOUT ANY WARRANTY; without even the implied warranty of       |
    |  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        |
    |  GNU Affero General Public License for more details.                  |
    |                                                                       |
    |  You should have received a copy of the GNU Affero General Public     |
    |  License along with this program.  If not, see                        |
    |  <http://www.gnu.org/licenses/agpl.html>.                             |
    =========================================================================

    """

    msg = ["NCVX: A User-Friendly and Scalable Package for Nonconvex Optimization in Machine Learning",
           "Version 1.1.1", "MIT License Copyright (c) 2021 SUN Group @ UMN"]
    return msg

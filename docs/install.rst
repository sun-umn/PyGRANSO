Installation
============

Installing PyGRANSO is simple. Here is a step by step plan on how to do it:

1. Download the latest version of PyGRANSO on GitHub (https://github.com/sun-umn/PyGRANSO)

2. We recommend creating a new conda environment to manage PyGRANSO dependencies::

    conda create --name pygranso_env
    
    conda activate pygranso_env

3. For Linux user, simply run::

    conda env create -f environment_linux.yml
    
Don't forget to change the env name and path in the first and last lines of environment.yml. Linux users can also run::
    
    conda install pytorch==1.9.0 torchvision torchaudio cudatoolkit=11 -c pytorch -c conda-forge

    conda install -c conda-forge osqp

For Windows User, simply run::

     conda env create -f environment_windows.yml
     
Don't forget to change the env name and path in the first and last lines of environment.yml. Windows users can also run::
    
     conda install -c oxfordcontrol osqp

     conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch


For Mac user, run::

    TODO

4. Check the examples page in the documentation and example codes in PyGRANSO package for detailed instruction.

5. Modify the working directory used in example codes.
    
Dependencies
-----------------

osqp-0.6.2

Python-3.9.7

numpy-1.20.3

scipy-1.7.1

pytorch-1.9.0
    


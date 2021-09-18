Installation
============

Installing PyGRANSO is simple. Here is a step by step plan on how to do it:

1. Download the latest version of PyGRANSO on GitHub (https://github.com/sun-umn/PyGRANSO)

2. We recommend creating a new conda environment to manage PyGRANSO dependencies::

    conda create -n pygranso python=3.9.6

3. In new conda environment, run::

     conda install -c oxfordcontrol osqp

     conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch

4. Check the examples page in the documentation and example codes in PyGRANSO package for detailed instruction.

5. Modify the working directory used in example codes.
    
Dependencies
-----------------

osqp-0.6.2

Python-3.9.6

numpy-1.21.1

scipy-1.7.0

pytorch-1.9.0
    


# Introduction to PyGRANSO: some simple constrained optimization problems and advanced usages in PyGRANSO

## Example A1: 2-variable nonsmooth Rosenbrock objective function

Modified based on GRANSO demo examples 1, 2, and 3.

2-variable nonsmooth Rosenbrock objective function, subject to simple bound constraints.

Reference: http://www.timmitchell.com/software/GRANSO/

Including tutorials about (L)BFGS restarting and PyGRANSO results logs.

## Example A2: Eigenvalue Optimization

Modified based on GRANSO demo example 4.

Reference: http://www.timmitchell.com/software/GRANSO/

Including tutorials about LBFGS and feasibility related options.

## Example A3: Dictionary learning 

Subgradient Descent Learns Orthogonal Dictionaries

Reference: Bai, Yu, Qijia Jiang, and Ju Sun. "Subgradient descent learns orthogonal dictionaries." arXiv preprint arXiv:1810.10702 (2018).

Including tutorials about auto-differentiation (AD), user provided analytical gradients and other PyGRANSO advanced settings.

Also including implementation of the same problem by using PyGRANSO default style or PyTorch nn module.

# Classical (Constrained) Optimization Problems

## Example B1: Nonlinear Feasiblity Problem

Reference: https://www.mathworks.com/help/optim/ug/solve-feasibility-problem.html

## Example B2: Optimization on Sphere Manifold

Reference: https://www.manopt.org/manifold_documentation_sphere.html

## Example B3: Trace Optimization

Trace optimization with orthogonal constraints.

Reference: Effrosini Kokiopoulou, Jie Chen, and Yousef Saad. "Trace optimization and eigenproblems in dimension reduction methods." Numerical Linear Algebra with Applications 18.3 (2011): 565-602.

# (Constrained) Machine Learning Problems

## Example C1: Robust PCA 

Reference: Yi, Xinyang, et al. "Fast algorithms for robust PCA via gradient descent." Advances in neural information processing systems. 2016.

## Example C2: Generalized LASSO

Generalized LASSO: total variation denoising

Reference: Boyd, Stephen, Neal Parikh, and Eric Chu. Distributed optimization and statistical learning via the alternating direction method of multipliers. Now Publishers Inc, 2011.

## Example C3: Logistic Regression

Reference: Sören Laue, Matthias Mitterreiter, and Joachim Giesen. "GENO--GENeric Optimization for Classical Machine Learning." Advances in Neural Information Processing Systems 32 (2019). and https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

## Example C4: Support Vector Machine

TODO

# (Constrained) Deep Learning Problems

## Example D1: Unconstrained Deep Learning: LeNet-5

Reference: LeCun, Yann. "LeNet-5, convolutional neural networks." URL: http://yann. lecun. com/exdb/lenet 20.5 (2015): 14.

## Example D2: Perceptual Attack

Adversarial Perceptual Attack on CIFAR-10 and ImageNet datasets.

Reference: Laidlaw, Cassidy, Sahil Singla, and Soheil Feizi. "Perceptual adversarial robustness: Defense against unseen threat models." arXiv preprint arXiv:2006.12655 (2020).

## Example D3: Orthogonal RNN

Reference: Lezcano-Casado, Mario, and David Martınez-Rubio. "Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group." International Conference on Machine Learning. PMLR, 2019.

## Example D4: Robustness Problems

TODO

Maximum Loss function or Minimum perturbation.

References: 
[1] Croce, Francesco, and Matthias Hein. "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." International conference on machine learning. PMLR, 2020.

[2] Croce, Francesco, and Matthias Hein. "Minimally distorted adversarial examples with a fast adaptive boundary attack." International Conference on Machine Learning. PMLR, 2020.

# Primary Authors:

* Tim Mitchell

    Tim Mitchell (http://www.timmitchell.com/) is the author of the original GRANSO package (http://www.timmitchell.com/software/GRANSO/).

* Buyun Liang

    Buyun Liang (https://buyunliang.org/) translates and revamps the original GRANSO package into a Python version package PyGRANSO (https://github.com/sun-umn/PyGRANSO), featuring auto-differentiation, GPU acceleration, tensor input, scalable QP solver, and zero dependency on proprietary packages. He also prepares numerous examples for PyGRANSO and maintains a documentation website (https://ncvx.org/) for it.  

# Contribution Details

## Tim Mitchell and Buyun Liang
    examples/1.demo_Rosenbrock.ipynb
    examples/2.demo_SpectralRadiusOpt.ipynb
    private/bfgsHessianInverseLimitedMem.py
    private/bfgssqp.py
    private/centerString.py
    private/copyrightNotice.py
    private/double2FixedWidthStr.py
    private/formatOrange.py
    private/isBlankStr.py
    private/isFiniteValued.py
    private/isRealValued.py
    private/linesearchWeakWolfe.py
    private/makePenaltyFunction.py
    private/makeStructWithFields.py
    private/neighborhoodCache.py
    private/optionValidator.py
    private/printMessageBox.py
    private/printOrange.py
    private/pygransoPrinter.py
    private/pygransoPrinterColumns.py
    private/qpSteeringStrategy.py
    private/qpTerminationCondition.py
    private/solveQP.py
    private/tablePrinter.py
    private/truncate.py
    private/wrapToLines.py
    pygranso.py
    pygransoOptions.py
    pygransoOptionsAdvanced.py

## Buyun Liang
    environment_cpu.yml
    environment_cuda.yml
    examples/3.demo_DictLearning.ipynb
    examples/4.demo_RobustPCA.ipynb
    examples/5.demo_GeneralizedLASSO.ipynb
    examples/6.demo_PerceptualAttack.ipynb
    examples/7.demo_UnconstrainedDL.ipynb
    examples/8.demo_nonlinear_feasiblity.ipynb
    examples/9.demo_sphere_manifold.ipynb
    private/getCiGradVec.py
    private/getCiVec.py
    private/getNvar.py
    private/getObjGrad.py
    private/isRestartData.py
    private/isString.py
    private/tensor2vec.py
    private/vec2tensor.py
    pygransoStruct.py
    README.md
    test_cpu.py
    test_cuda.py

## Tim Mitchell
    private/bfgsDamping.py
    private/bfgsHessianInverse.py
    private/isAnInteger.py
    private/isARealNumber.py
    private/isColumn.py
    private/isMbyN.py
    private/isPositiveDefinite.py
    private/isRow.py
    private/nDigitsInWholePart.py
    private/pygransoConstants.py
    private/regularizePosDefMatrix.py

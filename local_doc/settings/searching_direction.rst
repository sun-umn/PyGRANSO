Searching Direction Strategies
========================

max_fallback_level     
----------------        

Positive integer. Default value: 4

Max number of strategy to be employed (>= min_fallback_level)
NOTE: fallback levels 0 and 1 are only relevant for constrained problems. 

        SEARCH DIRECTION STRATEGIES
        If a step cannot be taken with the current search direction (e.g.
        computed an invalid search direction or the line search failed on a
        valid search direction), PyGRANSO may attempt up to four optional 
        fallback strategies to try to continue making progress from the current
        iterate.  The strategies are as follows and are attempted in order:
            
            0. BFGS-SQP steering 
                - default for constrained problems
                - irrelevant for unconstrained problems
            1. BFGS-SQP steering with BFGS's inverse Hessian approximation
                replaced by the identity. If strategy #0 failed because quadprog
                failed on the QPs, this "steepest descent" version of the 
                steering QPs may be easier to solve.
                - irrelevant for unconstrained problems
            2. Standard BFGS update on penalty/objective function, no steering
                - default for unconstrained problems
            3. Steepest descent on penalty/objective function, no steering
            4. Randomly generated search direction 






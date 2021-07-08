class qpSS:
    def __init__(self):
        pass

    def qpSteeringStrategy( self,penaltyfn_at_x, apply_Hinv, l1_model, ineq_margin, maxit, c_viol, c_mu, quadprog_options):
        """
        qpSteeringStrategy:
        attempts to find a search direction which promotes progress towards
        feasibility.  
        """

        d = -1
        mu = -1
        reduction = -1
        print("TODO: qpSteeringStrategy")
        return [d,mu,reduction]
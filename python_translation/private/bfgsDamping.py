def bfgsDamping(damping,applyH,s,y,sty):
    """
    bfgsDamping:
       This function implements Procedure 18.2 from Nocedal and Wright,
       which ensures that the BFGS update is always well defined.
    """

    damped      = False
    Hs          = applyH(s)
    stHs        = s.T@Hs 
    
    if sty < damping * stHs:
        theta   = ((1 - damping) * stHs) / (stHs - sty)
        y       = theta * y + (1 - theta) * Hs
        sty     = theta * sty + (1- theta) * stHs; # s.T@y;
        damped  = True

    return [y,sty,damped]
def ncvxConstants():
    """
    ncvxConstants:
    Simple routine for defining constants for use in:
        1) ncvx 
        2) ncvxOptions 
        3) ncvxOptionsAdvanced
    """
    # Number of first fallback level after QP approaches have failed
    POSTQP_FALLBACK_LEVEL       = 2
    # Number of last fallback level (randomly generated search directions)
    LAST_FALLBACK_LEVEL         = 4

    return [POSTQP_FALLBACK_LEVEL, LAST_FALLBACK_LEVEL]
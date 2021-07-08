def pygransoConstants():
    """
    pygransoConstants:
    Simple routine for defining constants for use in:
        1) granso 
        2) gransoOptions 
        3) gransoOptionsAdvanced
    """
    # Number of first fallback level after QP approaches have failed
    POSTQP_FALLBACK_LEVEL       = 2
    # Number of last fallback level (randomly generated search directions)
    LAST_FALLBACK_LEVEL         = 4

    return [POSTQP_FALLBACK_LEVEL, LAST_FALLBACK_LEVEL]
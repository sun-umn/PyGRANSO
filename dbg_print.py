def dbg_print(args):
    # debug_flag = True
    debug_flag = False
    OKCYAN = '\033[96m'
    W  = '\033[0m'  # white (normal)
    if debug_flag:
        print(OKCYAN + args + W)
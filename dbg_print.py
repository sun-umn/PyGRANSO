def dbg_print(args):
    # debug_flag = True
    debug_flag = False
    OKCYAN = '\033[96m'
    W  = '\033[0m'  # white (normal)
    if debug_flag:
        print(OKCYAN + args + W)

def dbg_print_1(args):
    # debug_flag = True
    debug_flag = False
    OKCYAN = '\033[96m'
    W  = '\033[0m'  # white (normal)
    if debug_flag:
        print(OKCYAN + args + W)

def dbg_print_2(args):
    # debug_flag = True
    debug_flag = False
    WARNING = '\033[93m'
    W  = '\033[0m'  # white (normal)
    if debug_flag:
        print(WARNING + args + W)
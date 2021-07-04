from pygranso import pygranso
from mat2vec import mat2vec

# options for pygranso
class Options:
    def __init__(self):
        self.QPsolver = 'qpalm'

# help(pygranso)

# variable and corresponding dimensions
var_dim_map = {"x1": (1,1), "x2": (1,1)}

# calculate total number of scalar variables
nvar = 0
for dim in var_dim_map.values():
    nvar = nvar + dim[0] * dim[1]

print(nvar)

opts = Options()
opts.QPsolver = 'gurobi'

# call mat2vec to enable GRANSO using matrix input
combined_fn = lambda x: mat2vec(x,var_dim_map,nvar)

pygranso(1,2,opts)

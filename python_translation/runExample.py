from pygranso import pygranso

# options for pygranso
class opts():
    pass

# help(pygranso)

# variable and corresponding dimensions
var_dim_map = {"x1": (1,1), "x2": (1,1)}

# calculate total number of scalar variables
nvar = 0
for dim in var_dim_map.values():
    nvar = nvar + dim[0] * dim[1]

print(nvar)

# opts.quadprog_opts.QPsolver = 'qpalm'
opts.quadprog_opts.QPsolver = 'gurobi'

pygranso(1,2,3)
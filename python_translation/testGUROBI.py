#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple QP model:
#  minimize
#      x^2 + x*y + y^2 + y*z + z^2 + 2 x
#  subject to
#      x + 2 y + 3 z >= 4
#      x +   y       >= 1
#      x, y, z non-negative
#
# It solves it once as a continuous model, and once as an integer model.

import gurobipy as gp
# from gurobipy import GRB
import numpy as np

# Create a new model
m = gp.Model("qp")

# Create variables
# x = m.addVar(ub=1.0, name="x")
# y = m.addVar(ub=1.0, name="y")
# z = m.addVar(ub=1.0, name="z")
x = m.addVars(3,1)
lst = []
for i in range(len(x)):
    lst.append(x[i,0])

x_vec = np.array(lst)
H = np.array([[1,0.5,0],[0.5,1,0.5],[0,0.5,1]])
f = np.array([2,0,0])

# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
# obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
obj = x_vec @ H @ x_vec.T + f @ x_vec
m.setObjective(obj)

# Add constraint: x + 2 y + 3 z <= 4
# m.addConstr(np.array([1,2,3]) @ x_vec == 4, "c0")

# Add constraint: x + y >= 1
# m.addConstr(np.array([1,1,0]) @ x_vec == 2, "c1")

LB = [1,2,3]
m.addConstrs(x_vec[i] >= LB[i] for i in range(len(LB)))

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())




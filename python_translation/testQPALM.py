import qpalm as qp
import numpy as np
import scipy as sc
import scipy.sparse as sp


solver = qp.Qpalm()
solver._settings.contents.eps_abs = 1e-10
solver._settings.contents.eps_rel = 1e-10

row = np.array([0, 0, 1, 1])
col = np.array([0, 1, 0, 1])
data = np.array([1, -1, -1, 2])
Q = sp.csc_matrix((data, (row, col)), shape=(3, 3))

q = np.array([-2, -6, 1])
bmin = np.array([0.5, -10, -10, -10])
bmax = np.array([0.5, 10, 10, 10])

row = np.array([0, 1, 0, 2, 0, 3])
col = np.array([0, 0, 1, 1, 2, 2])
data = np.array([1, 1, 1, 1, 1, 1])
A = sp.csc_matrix((data, (row, col)), shape=(4, 3))

solver.set_data(Q=Q, A=A, q=q, bmin=bmin, bmax=bmax)

solver._solve()
sol_x = solver._work.contents.solution.contents.x
tol = 1e-5

assert(abs(sol_x[0] - 5.5) < tol)
assert(abs(sol_x[1] - 5.0) < tol)
assert(abs(sol_x[2] - (-10)) < tol)

# Warm start with solution to check whether the solver exits immediately
solver._warm_start(solver._work.contents.solution.contents.x, solver._work.contents.solution.contents.y)
solver._solve()
assert(solver._work.contents.info.contents.iter == 0)

# Update functions
# It is possible to update the bounds, the linear part of the cost (q) and the settings
solver._settings.contents.eps_abs = 1e-10
solver._settings.contents.eps_rel = 0

solver._update_settings()
solver._solve()
sol_x = solver._work.contents.solution.contents.x
strict_tol = 1e-10
assert(abs(sol_x[0] - 5.5) < strict_tol)
assert(abs(sol_x[1] - 5.0) < strict_tol)
assert(abs(sol_x[2] - (-10)) < strict_tol)

solver._settings.contents.eps_abs = 1e-6
solver._settings.contents.eps_rel = 1e-6
solver._update_settings()

solver._data.contents.bmin[3] = -15
solver._update_bounds()
solver._solve()
sol_x = solver._work.contents.solution.contents.x
assert(abs(sol_x[0] - 8.5) < tol)
assert(abs(sol_x[1] - 7) < tol)
assert(abs(sol_x[2] - (-15)) < tol)

sol_x[0] = 0
sol_x[1] = 0
sol_x[2] = 0

solver._data.contents.q[0] = 0
solver._data.contents.q[1] = 0
solver._data.contents.q[2] = 0
solver._update_q()
solver._solve()
sol_x = solver._work.contents.solution.contents.x
assert(abs(sol_x[0] - 0) < tol)
assert(abs(sol_x[1] - 0) < tol)
assert(abs(sol_x[2] - (0.5)) < tol)

print("All tests succeeded!\n")
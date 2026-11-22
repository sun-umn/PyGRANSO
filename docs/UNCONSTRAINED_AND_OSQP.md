# Unconstrained Problem Handling and OSQP in PyGRANSO

Notes on how PyGRANSO handles unconstrained problems, how stationarity is computed, and practical implications for the QP solver (including when CUDA/OSQP helps or does not).

---

## Unconstrained Problem Handling in PyGRANSO

### Behavior changes

- **No steering QP:** Steering QP (levels 0–1) is disabled. The algorithm uses BFGS directions directly (level 2) or steepest descent (level 3).
- **Penalty parameter fixed:** `mu = 1` and cannot be adjusted. The penalty function equals the objective function.
- **Always feasible:** `feasible_to_tol = True` (no constraints to violate).
- **No penalty parameter retries:** Line search does not retry with lower `mu` values.

### Algorithm simplification

- Effectively reduces to **standard BFGS with line search**.
- No constraint violation handling.
- **Termination code 0** is still possible based on stationarity (`stat_val <= opt_tol`).

### Code locations

| Topic | File | Lines |
|-------|------|--------|
| Steering QP disabled | `bfgssqp.py` | 265–275 |
| Search direction computation (skips steering) | `bfgssqp.py` | 376–409 |
| Penalty parameter setup (`mu = 1`, always feasible) | `makePenaltyFunction.py` | 327–352 |
| Convergence checking (always feasible) | `bfgssqp.py` | 740–760 |

For unconstrained problems, PyGRANSO behaves like a **standard BFGS optimizer with nonsmooth stationarity checking**, without the constraint-handling overhead.

---

## Stationarity Computation and Gradient Sample Selection

### Unconstrained problems and QP solver usage

For unconstrained problems, PyGRANSO uses a **two-stage stationarity check**.

1. **Stage 1** checks if the gradient norm is below the tolerance (smooth case).
2. If not, **Stage 2** uses a QP solver to compute a **nonsmooth stationarity measure**, even for unconstrained problems. The QP finds the smallest vector in the convex hull of nearby gradient samples, which requires multiple gradient samples from the optimization history.

### How `l` (number of gradient samples) is computed

- `l` is the number of gradient samples returned by the **neighborhood cache**.
- Each sample contains gradients (objective, inequality constraints, equality constraints) evaluated at a previous iterate.
- The cache stores up to `ngrad` previous iterates and their gradients.
- On each iteration it returns all cached samples whose iterate is within Euclidean distance **`evaldist`** of the current point.
- The current iterate is always included (distance = 0), so **`l ≥ 1`**.
- The QP Hessian `H` has shape `(q + l + p) × (q + l + p)`; for **unconstrained** problems, **`H` is `l × l`** (one row/column per gradient sample).

### What `evaldist` and `ngrad` mean in practice

- **`evaldist`** (default: `1e-4`): Radius in variable space. Only iterates within this distance of the current point are used. A larger `evaldist` (e.g. `1e6` or `inf`) includes more history, but only if the cache has accumulated enough samples. Early iterations will still have small `l` because the cache has not built up yet.
- **`ngrad`** (default: `min(100, 2*n, n+10)`): Maximum number of gradient samples the cache can store. This caps `l` even with a large `evaldist`. After `ngrad` iterations, the cache uses a circular buffer, keeping exactly `ngrad` samples (the most recent ones).

### Practical implications

- `l` is limited by **both** `ngrad` and the **number of iterations** completed.
- Early iterations typically have **`l = 1–10`** as the cache builds.
- Later iterations may reach **`l = 10–100`** if steps are small and iterates cluster.
- With **`evaldist = inf`**, `l` equals the number of cached samples (up to `ngrad`), but you still need enough iterations to accumulate samples.
- Setting **`ngrad = 60000`** does not help if you only run 100 iterations; **`l` will be at most 100**.

---

## How More Gradient Samples Improve Stationarity Accuracy

- For **nonsmooth** problems, stationarity means **0 is in the subgradient set** (a set of vectors, not a single gradient).
- The QP approximates this by finding the **smallest vector in the convex hull** of nearby gradient samples.
- Using **more samples (larger `l`)** better approximates the subgradient set and improves the stationarity measure.
  - With **`l = 1`**, only the current gradient is used, which can miss that 0 is in the subgradient.
  - With **`l = 10–50`**, nearby gradients capture more of the subgradient structure, giving a more accurate measure.
- **Trade-off:** computational cost. Larger `l` means a **larger QP** (`H` is `l × l`), so there is a balance between accuracy and efficiency.
- For **smooth** problems, **`l = 1`** is often sufficient because the smooth check (Stage 1) typically passes; for **nonsmooth** problems, using more history improves the stationarity assessment.

---

## QP Size, CUDA OSQP, and Memory

Given the above, the **QP dimension** fed to the QP solver is on the order of **`l`** (for unconstrained) or **`q + l + p`** in general, often in the **hundreds to low thousands** (e.g. ~1000), not the full variable dimension `n`.

- **CUDA-based OSQP** is aimed at **large-scale** QPs where GPU parallelism pays off.
- At **~1000 variables**, there is **virtually no timing benefit** from the CUDA algebra compared to the built-in (CPU) solver, and the GPU path can be **more memory intensive**.
- So for typical PyGRANSO use (moderate `l`, QP size ~hundreds to ~1k), **CPU OSQP (`algebra="builtin"`) is appropriate**; enabling CUDA OSQP is unlikely to help and may use more memory.

---

## OSQP Experiment: ~1000 Variables (CPU vs CUDA)

Example run with **~1000 QP variables** (903 variables, 900 constraints). Timings are effectively the same between CPU and CUDA OSQP.

**Run 1 (CUDA):**

```
-----------------------------------------------------------------
           OSQP v1.0.0  -  Operator Splitting QP Solver
              (c) The OSQP Developer Team
-----------------------------------------------------------------
problem:  variables n = 903, constraints m = 900
          nnz(P) + nnz(A) = 1935
settings: algebra = CUDA 12.5,
          OSQPInt = 4 bytes, OSQPFloat = 4 bytes,
          device = Tesla T4 (Compute capability 7.5),
          linear system solver = CUDA Conjugate Gradient - Diagonal preconditioner,
          eps_abs = 1.0e-03, eps_rel = 1.0e-03,
          eps_prim_inf = 1.0e-15, eps_dual_inf = 1.0e-15,
          rho = 1.00e-01 (adaptive: 50 iterations),
          sigma = 1.00e-06, alpha = 1.60, max_iter = 1000000000
          check_termination: on (interval 5, duality gap: off),
          time_limit: 1.00e+03 sec,
          scaling: on (10 iterations), scaled_termination: off
          warm starting: on, polishing: off,
Solving using OSQP with algebra=cuda (indirect)
iter   objective    prim res   dual res   gap        rel kkt    rho         time
   1  -9.5998e+03   1.64e+01   4.15e+00  -9.66e+03   1.64e+01   1.00e-01    2.02e-02s
 110   1.1065e+02   8.89e-03   2.57e-04  -2.29e-01   8.89e-03   1.00e-01    2.07e-01s

status:               solved
number of iterations: 110
optimal objective:    110.6477
dual objective:       110.8764
duality gap:          -2.2873e-01
primal-dual integral: 1.6890e+04
run time:             2.07e-01s
optimal rho estimate: 1.46e-01
```

**Run 2 (CUDA, repeated):**

```
-----------------------------------------------------------------
           OSQP v1.0.0  -  Operator Splitting QP Solver
              (c) The OSQP Developer Team
-----------------------------------------------------------------
problem:  variables n = 903, constraints m = 900
          nnz(P) + nnz(A) = 1935
settings: algebra = CUDA 12.5,
          ...
Solving using OSQP with algebra=cuda (indirect)
iter   objective    prim res   dual res   gap        rel kkt    rho         time
   1  -9.5998e+03   1.64e+01   4.15e+00  -9.66e+03   1.64e+01   1.00e-01    1.74e-02s
 110   1.1065e+02   7.81e-03   1.77e-04  -2.24e-01   7.81e-03   1.00e-01    2.07e-01s

status:               solved
run time:             2.08e-01s
optimal rho estimate: 1.81e-01
```

**Conclusion:** At this problem size there is **virtually no difference in timing** between runs, and CPU OSQP is typically sufficient and less memory-intensive than CUDA for PyGRANSO’s QP subproblems.

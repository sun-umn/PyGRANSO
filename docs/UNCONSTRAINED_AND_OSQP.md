# Unconstrained Problem Handling and OSQP in PyGRANSO

Notes on how PyGRANSO handles unconstrained problems, how stationarity is
computed, and how the dense Torch reference and builtin OSQP policies apply.

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

## QP Size, Dense Torch OSQP, and Memory

The QP dimension is on the order of `l` for unconstrained stationarity checks
or `q + l + p` in general. It is usually much smaller than the original model
dimension, but can still reach the low thousands.

- The Torch-direct route is a correctness-first **dense reference solver**, not
  a scalable sparse solver.
- `opts.osqp_algebra = "auto"` keeps CPU-targeted work on builtin OSQP and uses
  Torch only for independently promoted accelerator backends.
- `opts.osqp_algebra = "torch"` explicitly requests the dense Torch route.
- Automatic Torch selection is limited to `n + m <= 2400` plus a conservative
  dense-memory preflight. Explicit Torch requests above the envelope warn and
  attempt the requested backend.
- Torch reuses `torch.linalg.lu_factor_ex`/`lu_solve` factors and includes Ruiz
  scaling, deterministic adaptive rho, strict polishing, and per-run warm state.
- Automatic exceptions or unsolved statuses produce a warned builtin retry with
  causal telemetry. MPS float64 requests return a builtin CPU float64 result.
- Sparse-CG, Jacobi, sparse operators, and CUDA Graph execution exist only on
  the research archive branch; their former settings produce a migration error.

Local NVIDIA correctness evidence passed, but representative B1/B2/B3 runs
were 12.48x, 21.33x, and 43.46x slower than builtin CPU OSQP. CUDA therefore
remains unpromoted for `auto`. Sparse acceleration and performance-oriented
backends are later milestones behind the same private factorization boundary.

PyGRANSO does not differentiate through a QP solve. Autograd forms objective
and constraint gradients before QP construction.

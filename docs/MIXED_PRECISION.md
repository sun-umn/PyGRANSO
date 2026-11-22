# 🎯 Mixed precision and `torch.autocast` with PyGRANSO

This note describes what to expect when using **`torch.autocast`** (or other mixed-precision setups) around the code that PyGRANSO calls (e.g. your `combined_fn`), and how it interacts with PyGRANSO's own precision and behavior.

---

## ⚙️ What PyGRANSO controls

- **`opts.double_precision`** (default: `True`) sets the dtype for PyGRANSO's **internal** data:
  - Optimization variable `x`, BFGS/L-BFGS state, penalty function values and gradients, line-search state, etc.
  - With `double_precision=True` → `torch.float64`; with `False` → `torch.float32`.
- The **QP solver** (OSQP/Gurobi) receives data after `.cpu().numpy()`; its precision is whatever dtype those tensors had (float32 or float64) before conversion.
- So: PyGRANSO's **algorithm state** is always in a single dtype (float32 or float64). It does **not** by default run your model in float16 or mixed precision.

---

## 🔬 What `torch.autocast` does (briefly)

- **Autocast** typically runs selected ops (e.g. matmuls, convolutions) in **float16** on GPU to:
  - Reduce **memory** (activations and some weights in float16).
  - Often **speed up** those ops on modern GPUs.
- It usually keeps other ops (e.g. reductions, loss scalers) in float32. So you get "mixed" precision inside the autocast region.

---

## 📍 Where autocast can apply with PyGRANSO

- You pass **`combined_fn`** into PyGRANSO (e.g. `soln = pygranso(var_spec=..., combined_fn=combined_fn, user_opts=opts)`), and PyGRANSO **calls it internally** during optimization. So you do not control the call site—PyGRANSO does.
- The only way to use autocast without changing PyGRANSO's code is to **wrap inside the function you pass**: define `combined_fn` so that **inside its body** you run the actual objective/constraint computation under `torch.autocast(...)`. Then whenever PyGRANSO calls `combined_fn(X)`, your code runs and the autocast context is active.
- Example: define `combined_fn` with autocast around the computation, then pass it to PyGRANSO as usual:

  ```python
  def combined_fn(X):
      with torch.autocast(device_type="cuda", dtype=torch.float16):
          # your actual objective and constraints
          f = ...
          ci = ...
          ce = ...
      return f, ci, ce

  soln = pygranso(var_spec=var_spec, combined_fn=combined_fn, user_opts=opts)
  ```

- PyGRANSO does **not** wrap your function in autocast; it does not change your model's dtypes. So any mixed precision is entirely under your control by wrapping inside `combined_fn`.

---

## ⚡ Will it speed up computations?

- **Possibly, but only in your objective/constraint evaluation.**
- Speedups are most likely when the **cost of your combined_fn** (and its backward) dominates total time (e.g. large neural nets, many matmuls/convs). Autocast can make those GPU ops faster.
- PyGRANSO's own work (BFGS updates, QP solves, line search, penalty and stationarity computations) is **not** inside autocast and uses `opts.double_precision` (float32 or float64). So:
  - If most time is in **PyGRANSO internals** (e.g. QP, many small evals), autocast will have **little or no** effect on total runtime.
  - If most time is in **your combined_fn** (e.g. big model forward/backward), autocast can **reduce that part** of the runtime.

---

## 💾 Will it decrease memory?

- **It can reduce memory use inside your model and during your combined_fn.**
- Float16 activations and (where applicable) weights use less memory than float32; that can lower peak memory during the forward/backward of `combined_fn`.
- PyGRANSO's internal tensors (x, H, gradients, penalty terms, etc.) are still in float32 or float64, so their memory is **unchanged** by autocast. Overall memory use is: **your model + combined_fn** (possibly lower with autocast) **+ PyGRANSO state** (unchanged).

---

## 🎯 Impact on using PyGRANSO

### 1. 🔢 Dtypes and gradients

- Autocast can produce **float16** (or bfloat16) outputs and gradients inside your graph. PyGRANSO then uses those values (e.g. it pulls gradients from `parameter.grad` or from your returned tensors) and typically works with them in its own dtype (float32/64) for the solver state.
- So you get **mixed precision**: float16 (or bfloat16) in your model, float32/64 in PyGRANSO's vectors and QP. This is generally workable, but you should be aware that:
  - Very small or very large magnitudes in float16 can underflow/overflow; if your problem is sensitive, you might need to keep some parts in float32 (e.g. loss scaling or disabling autocast for fragile ops).
  - PyGRANSO's **convergence and termination** logic (stationarity, line search, etc.) is written assuming the values it receives are numerically consistent; extreme mixed-precision effects could, in theory, affect behavior (e.g. rare cases where gradients are truncated or rounded in an unhelpful way).

### 2. ⚙️ `opts.double_precision`

- **`double_precision=True`** (default): PyGRANSO keeps its state in float64. Your combined_fn can still run under autocast (float16); PyGRANSO will use the resulting values/gradients (possibly upcast or stored in float64 internally). So you can get "mixed": float16 in your model, float64 in the optimizer state.
- **`double_precision=False`**: PyGRANSO uses float32. This pairs more naturally with autocast (float16 in the model, float32 in PyGRANSO) and can reduce memory further. For many ML problems this is a good setting to try with autocast.

### 3. 🛡️ Numerical stability

- BFGS, line search, and termination checks assume **consistent** precision and reasonable gradient/function values. Autocast is usually safe when:
  - Your combined_fn is numerically stable in float16 (or you use a loss scaler / careful scaling), and
  - You use **`double_precision=False`** if you want to avoid unnecessary float64↔float16 mixing in the same pipeline.
- If you see odd convergence (e.g. line search failures, stationarity flapping, or NaNs), try **disabling autocast** or narrowing the autocast region to confirm it's not due to mixed precision.

### 4. 🔧 QP and other internals

- The QP subproblems are built from tensors that PyGRANSO has already created (in float32 or float64), then converted to NumPy on CPU. Autocast does **not** change how the QP is built or solved; it only affects the **user-facing** objective/constraint and their gradients. So:
  - **No** mixed precision inside the QP solver itself.
  - Impact of autocast is **only** on the quality and cost of the function/gradient values that PyGRANSO feeds into the QP and the rest of the algorithm.

---

## 💡 Practical recommendations

| Goal              | Suggestion |
|-------------------|------------|
| 🧪 Try mixed precision | Use `torch.autocast` only around the body of your `combined_fn` (or the parts that run on GPU). Keep PyGRANSO's API as-is. |
| 🔗 Match dtypes      | Prefer **`opts.double_precision = False`** when using autocast, so PyGRANSO state is float32 and closer to what autocast produces. |
| ⚡ Speed             | Use autocast when **combined_fn** (and backward) dominates runtime; don't expect much gain if most time is in QP or other PyGRANSO internals. |
| 💾 Memory            | Autocast can reduce memory in **your model and combined_fn**; PyGRANSO's memory use is unchanged. |
| 🛡️ Stability         | If you see convergence or numerical issues, disable autocast or restrict it to a smaller region to see if mixed precision is the cause. |

---

## 📝 Summary

- **Autocast** can **speed up** ⚡ and **reduce memory** 💾 for the **objective/constraint evaluation** (your `combined_fn`) when that part is heavy and runs on GPU.
- It has **no direct effect** on PyGRANSO's internal steps (QP, BFGS, line search, etc.); those still run in float32 or float64 as set by `opts.double_precision`.
- **Expectations:** possible speedup and lower memory in your model; no change in PyGRANSO's own algorithm speed or memory; small risk of numerical effects if float16 is unstable in your problem.
- **Recommendation:** use autocast only around your combined_fn, set `double_precision=False` when using it, and if something breaks, try without autocast to confirm.

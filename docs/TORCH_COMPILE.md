# ⚡ `torch.compile` and PyGRANSO

This note describes what to expect if you use **`torch.compile`** (e.g. wrapping your model or the code inside your `combined_fn`) when solving problems with PyGRANSO, and how it can affect the optimization process.

---

## 🔬 What `torch.compile` does (briefly)

- **`torch.compile`** (PyTorch 2.0+) compiles a callable (e.g. a `nn.Module` or a function) so that its execution is optimized—e.g. via kernel fusion, fewer Python overheads, and backend-specific optimizations (e.g. Inductor).
- The **first few runs** are often slower (compilation/tracing); later runs with the **same** “graph” can be faster.
- If inputs, shapes, or control flow change in a way the compiler doesn’t expect, it may **recompile**, which can add overhead and sometimes reduce or cancel gains.

---

## 📍 Where compile can apply with PyGRANSO

- You pass **`combined_fn`** into PyGRANSO (e.g. `soln = pygranso(var_spec=..., combined_fn=combined_fn, user_opts=opts)`), and PyGRANSO **calls it internally** many times during optimization. You do not control the call site.
- To use `torch.compile` without changing PyGRANSO’s code, you apply it **inside the callable you pass**—either by compiling your **model** or by compiling a **function** that runs the objective/constraint and using that inside `combined_fn`.
- Example: compile your model, then use it inside `combined_fn`:

  ```python
  model = MyModel(...)
  model = torch.compile(model, mode="reduce-overhead")  # or "default", "max-autotune"

  def combined_fn(X):
      # X may be the model (var_spec is the model) or vec2tensor(x, var_dim_map)
      f = objective(X)
      ci = ineq_constraints(X)
      ce = eq_constraints(X)
      return f, ci, ce

  soln = pygranso(var_spec=var_spec, combined_fn=combined_fn, user_opts=opts)
  ```

- Alternatively, you can wrap only the heavy part in a compiled function and call that from inside `combined_fn`. PyGRANSO does **not** compile anything; any use of `torch.compile` is under your control inside the code path that runs when PyGRANSO calls `combined_fn`.

---

## ⚡ Will it speed up the PyGRANSO process?

- **Possibly, but only for the work inside your `combined_fn`** (and its backward). PyGRANSO’s own steps (BFGS, QP, line search, penalty and stationarity computations) are **not** compiled and are unchanged.
- **When it helps:** If the **same** (or very similar) graph is executed many times, compilation overhead is amortized and you can see a clear speedup. That is more likely when:
  - Your model and combined_fn have **stable shapes and control flow** across iterations, and
  - The cost of **combined_fn** (forward + backward) dominates total time.
- **When it can hurt or be neutral:**
  - **Recompilation:** PyGRANSO calls `combined_fn` with **different `x`** (and thus different `X`) every iteration. If the compiler treats these as new graphs (e.g. different shapes, different branches), you may get **recompilation often**, so compilation cost can dominate and total time can increase or stay similar.
  - **Cold start:** The first few iterations are slower; for short runs or when PyGRANSO does few evaluations, you may see no net benefit.
- So: **expect speedup only when** (1) most time is in your combined_fn, and (2) the compiled graph is reused across many calls (e.g. same tensor shapes and control flow). Otherwise, `torch.compile` may have **little effect or even slow things down** due to recompilation.

---

## 💾 Will it change memory use?

- **Usually modest impact.** Compiled code can reduce temporary allocations and kernel launches, which might **slightly lower** peak memory during your combined_fn. It does not change how PyGRANSO allocates its internal state (x, H, QP data, etc.).
- In practice, don’t rely on `torch.compile` for large memory savings; use it mainly for **speed** when the conditions above are met.

---

## 🎯 Impact on the PyGRANSO process

### 1. 🔢 Correctness and numerics

- **In principle**, compiled code should produce the same values as eager mode (same dtype and logic). In practice, **TorchDynamo/Inductor** can have edge cases (e.g. certain ops, control flow, or in-place updates) where behavior or numerics differ slightly.
- PyGRANSO’s logic (BFGS, line search, termination) assumes that the objective and constraints (and their gradients) are the same as in eager mode. If compilation changes outputs or gradients, convergence can change (e.g. different iterates, different termination iteration or code).
- **Recommendation:** If you use `torch.compile`, compare a short run (or key statistics) with and without it to confirm that results and convergence behavior are acceptable.

### 2. 📐 Dynamic shapes and recompilation

- PyGRANSO updates **x** every iteration; your **combined_fn** may see different tensor shapes or different branches (e.g. different numbers of active constraints). Compilers often **recompile** when shapes or control flow change, which can cause:
  - **Extra overhead** and variability in per-iteration time.
  - **More memory** if multiple compiled variants are cached.
- Using **static shapes** (e.g. fixed-size batches, fixed problem dimensions) and avoiding shape-dependent control flow inside the compiled region helps reuse one compiled graph and get the best benefit.

### 3. 🔧 Backward pass and gradients

- PyGRANSO needs gradients of the objective and constraints from your combined_fn (via autograd). `torch.compile` can compile both forward and backward; that is supported, but the same recompilation and correctness caveats apply to the backward graph.
- If you see **wrong gradients** (e.g. NaNs, or clearly different from eager), try disabling compile for the backward or for the whole combined_fn and report the issue.

### 4. 🛠️ PyGRANSO internals unchanged

- The QP solver, BFGS updates, line search, and penalty/stationarity computations are **not** compiled. Their speed and memory use are unchanged. So the **only** part of the “PyGRANSO process” that can change when you use `torch.compile` is the **cost and behavior of your combined_fn** (and thus the values and gradients PyGRANSO gets from it).

---

## 💡 Practical recommendations

| Goal              | Suggestion |
|-------------------|------------|
| 🧪 Try compile    | Use `torch.compile` on your **model** or on a **function** called from inside the **`combined_fn`** you pass to PyGRANSO. Keep PyGRANSO’s API as-is. |
| ⚡ Maximize speed | Prefer **stable shapes and control flow** in the compiled region so the same graph runs many times; avoid recompilation. Use `mode="reduce-overhead"` or `"max-autotune"` if appropriate. |
| 🔍 Check results  | Compare a run with and without compile (same opts, same seed) to confirm convergence and numerics are acceptable. |
| 🛡️ If it’s slow   | If total time goes up, recompilation is likely. Simplify the compiled region, fix shapes, or disable compile and rely on eager mode. |
| 📝 Backends       | Default backend is usually Inductor; you can try others (e.g. `backend="eager"` to disable compile) for debugging. |

---

## 📝 Summary

- **`torch.compile`** can **speed up** the part of the PyGRANSO process that runs **inside your combined_fn** (and its backward), when the same graph is reused many times. It has **no direct effect** on PyGRANSO’s own steps (QP, BFGS, line search, etc.).
- **Risk:** Recompilation (e.g. due to changing `x`/shapes or control flow) can add overhead and reduce or cancel gains; cold start can make short runs slower.
- **Impact on the process:** Mainly **runtime** of your combined_fn; possibly **slight** memory reduction there. Correctness and convergence should match eager mode in principle, but compiler edge cases can cause small differences—so validate if you rely on compile for production runs.
- **Recommendation:** Use `torch.compile` on your model or on the heavy part of `combined_fn` if that part dominates time and has stable shapes; otherwise, prefer eager mode to avoid recompilation overhead.

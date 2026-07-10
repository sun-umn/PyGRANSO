# Phase 1.1 Implementation Plan: Public QP Contract

## Goal

Implement **the public PyGRANSO-to-OSQP QP contract**.

This feature should allow the system to:

1. Accept PyGRANSO QP inputs consistently.
2. Convert them to canonical OSQP form.
3. Reject invalid shapes, dtypes, devices, bounds, and nonfinite values early.

Keep the implementation modular, easy to test, and consistent with the existing project structure.

---

## Current State

The project already has:

* `pygranso/private/solveQP.py`: receives PyGRANSO QP arguments.
* `pygranso/private/osqpTorchAdapter.py`: routes QPs to builtin or Torch OSQP.
* `_build_constraints_torch` and `_build_constraints_numpy`: build OSQP constraints.

The missing part is:

* A standalone implementation plan for the canonical QP boundary.
* Clear input/output type definitions for the adapter contract.
* A checklist tying validation behavior to files and tests.

---

## New Components to Add

### Component 1

```text
PygransoQPInput
```

Responsibility:

```text
Represent the raw QP data produced by PyGRANSO before OSQP canonicalization.
```

### Component 2

```text
CanonicalOSQPProblem
```

Responsibility:

```text
Represent the validated OSQP problem `min 0.5*x'Px + q'x` subject to `l <= Ax <= u`.
```

---

## Class / Registry Diagrams

### Diagram 1: PyGRANSO QP Input

```text
+-------------------------------------------------------------------------------+
|                                 PygransoQPInput                                |
+-------------------------------------------------------------------------------+
|  - H: torch.Tensor                                                             |
|  - f: torch.Tensor                                                             |
|  - A: torch.Tensor | None                                                      |
|  - b: torch.Tensor | None                                                      |
|  - LB: torch.Tensor                                                            |
|  - UB: torch.Tensor                                                            |
|  - torchDevice: torch.device                                                   |
|  - doublePrecision: boolean                                                    |
+-------------------------------------------------------------------------------+
|  + validateShape(): void                  --> Checks dimensions and columns    |
|  + validateDtype(): void                  --> Checks float32/float64 contract  |
|  + validateDevice(): void                 --> Checks compatible devices        |
|  + variableCount(): number                --> Returns QP dimension n           |
+-------------------------------------------------------------------------------+
```

### Diagram 2: Canonical OSQP Problem

```text
+-------------------------------------------------------------------------------+
|                              CanonicalOSQPProblem                              |
+-------------------------------------------------------------------------------+
|  - P: torch.Tensor                                                             |
|  - q: torch.Tensor                                                             |
|  - A_osqp: torch.Tensor                                                        |
|  - l: torch.Tensor                                                             |
|  - u: torch.Tensor                                                             |
|  - constraintOrderSignature: tuple                                             |
+-------------------------------------------------------------------------------+
|  + validateBounds(): void                 --> Rejects NaN and l > u            |
|  + objective(x): float                    --> Computes canonical objective     |
|  + kktDimension(): number                 --> Returns n + m                    |
|  + toBuiltinArrays(): tuple               --> Produces NumPy/scipy inputs      |
+-------------------------------------------------------------------------------+
```

### Diagram 3: Constraint Assembler

```text
+-------------------------------------------------------------------------------+
|                                ConstraintAssembler                             |
+-------------------------------------------------------------------------------+
|  - No persistent internal state                                                |
+-------------------------------------------------------------------------------+
|  + buildTorch(A,b,LB,UB,n): tuple         --> Builds Torch A,l,u rows          |
|  + buildNumpy(A,b,LB,UB,n): tuple         --> Builds builtin sparse rows       |
|  + orderSignature(A,b,n): tuple           --> Records equality/bound ordering  |
+-------------------------------------------------------------------------------+
```

---

## Class Diagram Rules

* The public contract should remain small and stable.
* Validation classes are internal helpers, not new public APIs.
* Backend-specific implementation details must not leak into the public QP option surface.

---

## Data Model

```ts
type QPContractInput = {
  id: string;
  name: string;
  H: "Tensor[n,n]";
  f: "Tensor[n] or Tensor[n,1]";
  A?: "Tensor[p,n]";
  b?: "Tensor[p] or Tensor[p,1]";
  LB: "Tensor[n,1]";
  UB: "Tensor[n,1]";
  dtype: "float32" | "float64";
  device: string;
};
```

---

## Storage / State

This feature is stateless. It receives input, returns canonical output, and does not persist data.

Temporary state:

* Canonical tensors during one solve.
* Constraint order signature passed into solver settings.

---

## Required Methods

```ts
function canonicalizePygransoQP(input: QPContractInput): CanonicalOSQPProblem
```

```ts
function validateCanonicalProblem(problem: CanonicalOSQPProblem): void
```

---

## Validation Rules

Before processing data, check:

1. `H` is square and matches `len(f)`.
2. `LB` and `UB` are column vectors with `n` entries.
3. Equality rows have compatible `A` and `b`.
4. Matrices and finite vectors contain no NaN/Inf.
5. Infinite values are permitted only in bounds.
6. Every lower bound is less than or equal to its upper bound.

---

## UI / API Integration

This feature is internal.

Callers:

* `solveQP(...)` passes raw QP data.
* `solve_osqp_torch_qp(...)` normalizes and selects backend.
* Builtin and Torch paths consume canonical problem data.

---

## Workflow

1. Receive PyGRANSO QP input.
2. Validate base shapes and dtype.
3. Build equality rows if present.
4. Append variable-bound identity rows.
5. Produce `P`, `q`, `A_osqp`, `l`, `u`.
6. Pass canonical data to selected backend.

---

## Files to Create

```text
None
```

---

## Files to Modify

```text
pygranso/private/osqpTorchAdapter.py
pygranso/private/solveQP.py
tests/test_torch_osqp_direct.py
tests/test_torch_osqp_policy.py
```

---

## Error Handling

Handle these cases:

* Missing required tensor.
* Incompatible shapes.
* Incompatible dtype or device.
* Material asymmetry in `H`/`P`.
* Invalid bounds.
* Nonfinite data where not allowed.

---

## Testing Checklist

Test the following:

* [ ] Bound-only QP canonicalizes correctly.
* [ ] Equality-plus-bounds QP canonicalizes correctly.
* [ ] Infinite bounds are preserved.
* [ ] NaNs are rejected.
* [ ] Shape mismatch raises a clear error.
* [ ] Return solution shape remains `(n, 1)`.

---

## Acceptance Criteria

This phase is complete when:

1. `CanonicalOSQPProblem` behavior is implemented through adapter functions.
2. Builtin and Torch routes receive equivalent QP data.
3. Validation prevents invalid input from reaching backend solvers.
4. Tests or manual checks confirm expected behavior.

# XAD-Forge JIT Limitations

## Overview

The XAD-Forge JIT backend compiles a recorded XAD tape into native x86-64 machine
code for fast repeated evaluation. Its core paradigm is **record once → compile once
→ evaluate many**. This document describes a fundamental limitation of that approach
when applied to code with data-dependent branching — which includes most QuantLib
pricing engines.

---

## The branching problem

### How XAD tape recording works

When XAD records a computation, it traces the **actual execution path** through the
C++ code. Every arithmetic operation on `AReal` values appends a statement to a
linear tape — a flat list of `{operation, operand_slots, result_slot}` entries.

C++ `if` statements that depend on `AReal` values are **not recorded on the tape**.
The `if` is evaluated at record time using the current numeric value, one branch is
taken, and only the operations inside that branch appear on the tape. The other branch
is never visited and leaves no trace.

For standard XAD (non-JIT), this is fine: the tape is used once, the branch was correct
for that evaluation, and `computeAdjoints()` differentiates the path that was actually
taken. This is standard practice in AAD — it computes the derivative of the function
*as evaluated*, not the derivative of all possible branches.

### What Forge does differently

Forge takes the recorded tape and compiles it to native x86-64 machine code — a
`ForgedKernel`. The selling point is **reuse**: record once, compile once, evaluate
many times with different inputs.

But the compiled kernel is a fixed sequence of instructions. It encodes exactly one
control flow path — whichever branch was taken during the original recording. There are
no `if` instructions in the kernel. It is a straight-line computation.

### The problem

When the kernel is re-evaluated with new inputs, stale branch decisions produce wrong
results or crashes:

```
Record:   rate = 0.05  →  if (rate > 0.04) { path A }  →  kernel encodes path A
Replay:   rate = 0.03  →  kernel still executes path A  →  WRONG result
```

The kernel does not know path B exists. The `if` was resolved at record time and only
path A's operations were captured.

---

## Impact on QuantLib pricing engines

### Concrete example: TreeCallableFixedRateBondEngine

A trinomial tree engine with 40 time steps contains branching at multiple levels:

1. **Coupon/callability alignment**:
   `if (withinNextWeek(callabilityTime, couponTime))` — whether a coupon date falls
   near a callability date depends on the term structure, which changes with the rate
   input.

2. **Exercise decisions**:
   `if (callPrice < continuationValue)` — the optimal exercise boundary shifts when
   rates change. Under new inputs the bond might be called at step 15 instead of
   step 20, but the kernel still follows the original exercise pattern.

3. **Boundary conditions**:
   Various `if (i == 0)`, `if (time > maturity)` checks that may depend on how the
   grid aligns with input-dependent dates.

Each of these is a plain C++ `if` evaluated at record time. The compiled kernel bakes
in all of them. With different market inputs:

- **Wrong results**: The kernel follows stale branches, computing prices/greeks for a
  decision tree that does not match the new inputs.
- **Crashes**: Some branches guard array bounds or pointer validity. Following the wrong
  branch can read out-of-bounds memory or dereference invalid pointers (SIGSEGV).

### Affected engine categories

Nearly every QuantLib engine has data-dependent branching:

| Engine type | Branching source |
|---|---|
| **Tree/lattice engines** | Exercise decisions, coupon timing, boundary handling |
| **Monte Carlo engines** | Path-dependent payoffs, barrier crossings, early exercise |
| **PDE/FD engines** | Grid adaptation, boundary conditions |
| **Calibration** | Convergence checks, step-size adaptation, solver iterations |
| **Curve bootstrapping** | Solver iterations, bracket selection |

### What works safely

Engines where control flow is **entirely independent of AReal input values** — essentially
pure straight-line arithmetic:

- Black-Scholes closed-form formula
- Simple analytic formulas with no conditional logic on inputs
- Any computation graph that is structurally identical for all input values

---

## Forge's intended solution: `ABool::If`

Forge provides `ABool::If(condition, trueValue, falseValue)` — a branchless conditional
select that records **both** paths as a single node in the computation graph. The kernel
evaluates both sides and selects the correct one at runtime, similar to a CPU `cmov`
instruction.

However, adopting this in QuantLib would require:

1. Rewriting all `if` statements that depend on `AReal` values to use `ABool::If`.
2. Patterns that cannot be expressed as `ABool::If` — loops with data-dependent iteration
   counts, early returns, exception-guarded paths — would need deeper restructuring.
3. The performance impact of always evaluating both branches may negate the JIT benefit
   for complex engines.

This is a substantial undertaking and is not currently implemented in QuantLib-Risks.

---

## Observed behaviour

### AAD replay (fixed tape, backward sweep only) — works

Recording the tape once at base parameters and replaying backward sweeps works correctly
under JIT. The computation graph is fixed, no re-recording occurs, and all branches
match the original recording. The `run_benchmarks.py` callable bond result (6.74×
JIT speedup) uses this mode.

### AAD re-record (fresh recording per scenario) — crashes or wrong results

When `newRecording()` is called per scenario with different market inputs under JIT:

- **Crashes** at `computeAdjoints()` — the Forge-compiled kernel accesses invalid
  memory because array indices or pointer offsets from the original recording no longer
  apply.
- **Wrong derivatives** — even when the kernel does not crash, it follows stale branches
  and produces garbage values (observed: 0.0, or values like 124,730,936 instead of
  the correct ~-440).

### Upstream status

Both `lib/xad-forge/README.md` and `lib/forge/README.md` document the system as
**experimental / proof-of-concept**, not hardened for production use.

---

## Recommendations

1. **Use JIT only for AAD replay** (fixed tape, `clearDerivatives()` +
   `computeAdjoints()` loop) where the computation graph does not change between
   evaluations.

2. **Do not use JIT for re-record workflows** (`newRecording()` per scenario) with
   engines that contain data-dependent branching — which is most QuantLib engines.

3. **For Monte Carlo / scenario risk**, use the standard non-JIT XAD tape. The
   re-record overhead is modest (the IRS benchmark shows FD÷AAD-re-record ≈ 5.6×)
   and produces correct derivatives.

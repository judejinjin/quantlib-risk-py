# AAD & JIT in QuantLib-Risks-Py

*A technical overview of how this project enables Adjoint Algorithmic Differentiation (AAD)
in QuantLib, how JIT compilation accelerates it, current JIT limitations, and benchmark
evidence showing how AAD scales vs finite differences as the number of inputs grows.*

---

## 1 — Why AAD?

Pricing a derivative requires computing sensitivities (Greeks) with respect to every market
input.  The traditional **finite-difference (FD)** approach bumps each input one at a time
and re-prices:

$$
\frac{\partial V}{\partial x_i} \approx \frac{V(x_i + h) - V(x_i)}{h}
$$

This costs **N + 1 pricings** for N inputs — prohibitively expensive when N is large
(e.g. a yield curve with 20 pillar quotes).

**Adjoint Algorithmic Differentiation (AAD)** computes the entire gradient
$\nabla V = \bigl(\partial V/\partial x_1, \dots, \partial V/\partial x_N\bigr)$
in a **single backward sweep** whose cost is bounded by a small constant multiple of
one forward pricing.  The theoretical bound is ≤ 4× the forward cost, independent of N.

---

## 2 — How QuantLib-Risks Enables AAD

### 2.1  Type replacement: `double` → `xad::AReal<double>`

QuantLib defines its floating-point type via a preprocessor macro `QL_REAL`, which
defaults to `double`.  The project replaces it **at compile time** — zero QuantLib
source modifications are needed:

```
QuantLib-Risks-Cpp  provides  ql/qlrisks.hpp
    └─ #define QL_REAL  xad::AReal<double>
    └─ ~500 lines of Boost / QuantLib compatibility specialisations
```

Through `QL_INCLUDE_FIRST=ql/qlrisks.hpp`, every QuantLib translation unit automatically
picks up the new type.  The XAD library's operator overloading then records every arithmetic
operation on `Real` values onto an internal **tape** data structure.

### 2.2  The tape-based workflow

| Step | What happens |
|------|-------------|
| **1. Activate tape** | `tape = Tape(); tape.activate()` — sets a thread-local recording target |
| **2. Register inputs** | `tape.registerInput(x)` — marks variables as independent |
| **3. Forward pass** | Build curves, price instruments — XAD records every `+`, `*`, `sin`, … as `{opcode, operands, result}` entries on the tape |
| **4. Seed output** | `derivative(npv) = 1.0` — set ∂V/∂V = 1 |
| **5. Backward sweep** | `tape.computeAdjoints()` — walk the tape in reverse, applying the chain rule |
| **6. Read gradients** | `tape.derivative(x_i)` returns ∂V/∂xᵢ for every registered input |

**Key property:** Steps 4–6 (the *backward pass*) are O(1) in N — one sweep yields all
N sensitivities.

### 2.3  Replay vs re-record

After the initial recording, the tape can be **replayed** many times with different adjoint
seeds without re-executing the forward pass:

```python
for scenario in scenarios:
    tape.clearDerivatives()
    derivative(npv) = 1.0
    tape.computeAdjoints()        # re-uses the same tape
    greeks[scenario] = tape.derivative(x)
```

This is vastly cheaper than **re-recording** (calling `tape.newRecording()` and
re-executing the forward pass), which must rebuild the tape for each scenario.

> **Rule of thumb:** Use *replay* for scenario sweeps where the computation graph does
> not change.  Use *re-record* only when inputs change the control flow (e.g. different
> solver paths).

### 2.4  Python integration via SWIG

The SWIG bindings include custom typemaps (`types.i`) that convert transparently between
Python floats and `xad::AReal<double>`.  Python users interact with the tape through the
`xad` Python package:

```python
import QuantLib_Risks as ql
from xad.adj_1st import Tape

tape = Tape()
tape.activate()
# … build curves, price instruments — all recorded on tape …
tape.computeAdjoints()
delta = tape.derivative(spot)
```

No special Python-side configuration is needed; the C++ operator overloading handles
everything transparently.

### 2.5  Build dependency graph

```
forge          (JIT compiler, AsmJit-based)
  └── forge/api/c    (C API for binary compatibility)
xad            (AAD library, operator-overloading)
xad-forge      (bridge: XAD tape → Forge graph → native code)
QuantLib-Risks-Cpp   (INTERFACE lib: redefines Real, adds specialisations)
QuantLib       (compiled with Real = xad::AReal<double>)
QuantLib-Risks-Py    (SWIG bindings → Python extension module)
xad-autodiff   (Python bindings for the XAD tape API)
```

All C++ dependencies are injected into QuantLib's build via a single CMake variable:
`QL_EXTERNAL_SUBDIRECTORIES`.

---

## 3 — How JIT Speeds Up AAD

### 3.1  The problem with tape interpretation

The standard backward sweep *interprets* the tape — reading each entry, dispatching by
opcode, and propagating adjoints.  This involves pointer chasing, branch mispredictions,
and poor cache locality, especially for large tapes (millions of entries for solver-heavy
instruments like bootstrapped curves).

### 3.2  Forge: record once → compile once → evaluate many

The Forge JIT compiler transforms the tape into a **compiled native kernel**:

```
XAD Tape (linear list of operations)
    │
    ▼   xad-forge transformation
Forge Graph (DAG of operations)
    │
    ▼   Forge optimisation passes
         • common subexpression elimination
         • constant folding
         • algebraic simplification
    │
    ▼   AsmJit code generation
ForgedKernel (native x86-64 machine code)
```

The compiled kernel replaces tape interpretation with a single function call.  No
interpreter overhead, no opcode dispatch.

### 3.3  Two execution backends

| Backend | Instructions | Parallelism | Use case |
|---------|-------------|-------------|----------|
| **ScalarBackend** | SSE2 scalar | 1 eval at a time | Drop-in tape replacement |
| **AVXBackend** | AVX2 packed (256-bit) | **4 evals in parallel** | Batch pricing, scenario grids |

The AVX backend evaluates 4 independent input sets simultaneously using SIMD, which is
ideal for portfolio-level risk computation.

### 3.4  Where JIT helps most

JIT benefits scale with the **regularity and size of the backward sweep kernel**, not
with N.  The biggest wins come from:

- **Tight regular loops** (e.g. trinomial tree's 40-step backward induction → 6.7× JIT
  speedup over interpreted AAD)
- **Large tapes** where interpretation overhead dominates

For small, smooth computations (e.g. Black-Scholes with 4 inputs), the tape is already
tiny and JIT adds negligible benefit (~1×).

---

## 4 — JIT Limitations

### 4.1  Root cause: data-dependent branching

XAD records a **linear tape** — C++ `if` statements that branch on `AReal<double>` values
are evaluated at record time, and only the taken branch appears on the tape.  The compiled
ForgedKernel is a **straight-line computation** with no conditional instructions.

When the kernel is re-evaluated with new inputs that would have taken a different branch:
- **Wrong results** — stale branch decisions produce incorrect prices/Greeks
- **Crashes (SIGSEGV)** — stale array indices or pointer offsets access invalid memory

### 4.2  Affected QuantLib engine types

| Engine type | Branching source | Example |
|-------------|-----------------|---------|
| **PDE / Finite-Difference** | Grid adaptation, boundary conditions, early-exercise checks | `FdBlackScholesVanillaEngine`, `FdSimpleBSSwingEngine` |
| **Monte Carlo** | Path-dependent payoffs, barrier crossings, random path generation | `MCEuropeanBasketEngine` |
| **Tree / Lattice** | Exercise decisions, coupon timing, boundary handling | `TreeCallableFixedRateBondEngine`† |
| **ISDA CDS** | Data-dependent `if (fhphh < 1E-4)` on `Real` | `IsdaCdsEngine` |

> †The callable-bond tree engine works with JIT (and shows a large speedup) because its
> 40-step trinomial loop is regular enough that the branch pattern is stable across
> scenarios.  However, this is not guaranteed for all tree engines.

### 4.3  Potential solution: branchless selects

Forge supports `ABool::If(condition, trueValue, falseValue)` — a branchless select that
records **both** paths.  Adopting this across QuantLib would require rewriting all
data-dependent `if` statements — a large undertaking not currently implemented.

### 4.4  JIT eligibility summary

| Instrument / Engine | JIT eligible | Reason |
|---|:---:|---|
| European Call — AnalyticEuropeanEngine | ✅ | Smooth BSM formula, no branching |
| American Put — BAW / Bjerksund-Stensland / QD+ | ✅ | Closed-form / quasi-analytic approximations |
| American Put — FdBlackScholesVanillaEngine | ❌ | PDE boundary conditions, early exercise |
| Callable Bond — TreeCallableFixedRateBondEngine | ✅† | Regular tree loop; high variance |
| IR Cap — BlackCapFloorEngine | ✅ | Caplet Black formula evaluations |
| European Swaption — JamshidianSwaptionEngine | ✅ | Zero-bond option decomposition |
| Vanilla IRS — DiscountingSwapEngine | ✅ | Bootstrap solver on tape |
| SOFR OIS IRS — DiscountingSwapEngine | ✅ | Small tape |
| CDS — MidPointCdsEngine | ✅ | Small tape |
| CDS — IsdaCdsEngine | ❌ | `if` on `Real` in `isdacdsengine.cpp` |
| Risky Bond — RiskyBondEngine | ✅ | Branching on dates only (not `AReal`) |
| Basket Option — MCEuropeanBasketEngine | ❌ | MC path generation has data-dependent branching |
| Swing Option — FdSimpleBSSwingEngine | ❌ | PDE solver + exercise logic |

---

## 5 — Benchmark Results

All benchmarks run on Linux x86-64, Python 3.13.

### 5.1  Single-instrument pricing

*30 repetitions, median. AAD backward pass = tape replay.*

| Instrument | Engine | N | FD (ms) | AAD (ms) | FD ÷ AAD | JIT eligible |
|---|---|---:|---:|---:|---:|:---:|
| European Call | AnalyticEuropeanEngine | 4 | 0.024 | 0.001 | **17×** | ✅ |
| American Put (BAW) | BaroneAdesiWhaleyApprox | 4 | 0.020 | 0.003 | **8×** | ✅ |
| American Put (B-S) | BjerksundStenslandApprox | 4 | 0.128 | 0.003 | **43×** | ✅ |
| American Put (PDE) | FdBlackScholesVanilla | 4 | 354.3 | 168.9 | **2.1×** | ❌ |
| American Put (QD+) | QdPlusAmericanEngine | 4 | 0.183 | 0.029 | **6.2×** | ✅ |
| Swaption (HW) | JamshidianSwaptionEngine | 3 | 0.136 | 0.007 | **18×** | ✅ |
| Basket Option (MC) | MCEuropeanBasketEngine | 5 | 254.5 | 7.2 | **36×** | ❌ |
| Swing Option (PDE) | FdSimpleBSSwingEngine | 3 | 110.6 | 31.9 | **3.5×** | ❌ |
| Risky Bond | RiskyBondEngine | 14 | 7.417 | 0.064 | **116×** | ✅ |
| IR Cap (Black) | BlackCapFloorEngine | 18 | 7.682 | 0.009 | **888×** | ✅ |
| Vanilla IRS | DiscountingSwapEngine | 17 | 6.571 | 0.034 | **191×** | ✅ |

**Observation:** For analytic engines, FD ÷ AAD scales roughly linearly with N:

```
N =  3 →   18×  (swaption)
N =  4 →  8–43× (american options, european)
N =  5 →   36×  (basket MC — AAD efficient despite MC overhead)
N = 14 →  116×  (risky bond — OIS + CDS bootstrapped curve)
N = 17 →  191×  (vanilla IRS — 17 curve quote inputs)
N = 18 →  888×  (IR cap — 18 inputs, large forward evaluation)
```

The PDE engines (N = 3–4) show only 2–3.5× because the backward sweep cost is dominated
by PDE-specific overhead (grid sweeps, boundary logic), not by the number of inputs.

### 5.2  Scenario-batch benchmarks (100 scenarios)

*5 outer repetitions, median.*

| Instrument | Engine | N | FD batch (ms) | AAD replay (ms) | FD ÷ AAD | AAD re-record (ms) | FD ÷ re-record |
|---|---|---:|---:|---:|---:|---:|---:|
| Vanilla IRS | DiscountingSwap | 17 | 726.4 | 3.4 | **212×** | 130.5 | 5.6× |
| CDS (MidPoint) | MidPointCdsEngine | 6 | 78.3 | 1.2 | **63×** | 42.6 | 1.8× |
| CDS (ISDA) | IsdaCdsEngine | 20 | 6,599.8 | 26.0 | **254×** | 1,580.5 | 4.2× |
| SOFR OIS IRS | DiscountingSwap | 9 | 404.4 | 3.0 | **136×** | 5,143.1 | 0.1× |
| Callable Bond | TreeCallableBond | 3 | 335.0 | 69.7 | **4.8×** | 675.3 | 0.5× |

**Key insight — replay vs re-record:**
- AAD **replay** is always dramatically faster than FD (63–254×)
- AAD **re-record** can be *slower* than FD for solver-heavy instruments (OIS: 0.1×,
  callable bond: 0.5×) because re-recording rebuilds the tape through iterative solvers
- Rule: always prefer replay when the computation graph is stable across scenarios

### 5.3  Jacobian benchmarks (curve sensitivities)

These benchmarks compute full N×N Jacobian matrices and demonstrate two approaches:

- **Approach 1 (direct):** Differentiate the entire pipeline (curve build + pricing) end
  to end.  AAD replay gives a single row of the Jacobian per sweep.
- **Approach 2 (chain rule):** Factor the Jacobian as $J = J_{\text{curve}} \times
  J_{\text{pricing}}$, differentiate each piece separately, and multiply.

#### 5.3.1  Interest-rate Jacobians (9×9 OIS curve, 100 scenarios)

**Zero-rate sensitivities (∂NPV/∂z) — direct ZeroCurve, no solver:**

| Method | Time (ms) | vs FD |
|--------|----------:|------:|
| FD (N+1 pricings × 100 scenarios) | 4,647.8 | 1× |
| AAD re-record | 468.8 | 10× |
| Jacobian chain-rule (9 sweeps + solve) | 55.3 | 84× |
| **AAD replay** | **0.1** | **35,548×** |

**Par-rate sensitivities (∂NPV/∂r) — through bootstrap solver:**

| Method | Time (ms) | vs FD |
|--------|----------:|------:|
| FD | 437.3 | 1× |
| AAD re-record | 5,344.9 | 0.1× *(slower)* |
| Jacobian chain-rule | 48.9 | 9× |
| **AAD replay** | **2.6** | **168×** |

#### 5.3.2  Credit Jacobians (4×4 CDS curve, 100 scenarios)

**Hazard-rate sensitivities (∂NPV/∂h) — direct HazardRateCurve:**

| Method | Time (ms) | vs FD |
|--------|----------:|------:|
| FD | 17.0 | 1× |
| AAD re-record | 5.4 | 3× |
| Jacobian chain-rule | 4.6 | 4× |
| **AAD replay** | **0.2** | **85×** |

**CDS-spread sensitivities (∂NPV/∂s) — through bootstrap solver:**

| Method | Time (ms) | vs FD |
|--------|----------:|------:|
| FD | 110.7 | 1× |
| AAD re-record | 57.5 | 2× |
| Jacobian chain-rule | 1.2 | 89× |
| **AAD replay** | **2.1** | **53×** |

### 5.4  JIT speedup on AAD backward pass

| Instrument | AAD Non-JIT (ms) | AAD JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| European Call | 0.0014 | 0.0013 | 1.08× |
| American Put (BAW) | 0.0025 | 0.0022 | 1.13× |
| American Put (B-S) | 0.0030 | 0.0026 | 1.16× |
| **Callable Bond (tree)** | **0.5656** | **0.0839** | **6.74×** |
| IR Cap (Black) | 0.0086 | 0.0081 | 1.06× |
| Vanilla IRS | 0.0343 | 0.0309 | 1.11× |

JIT's biggest win is on the callable bond's trinomial tree (6.74×), where the tight
regular backward-induction loop is ideal for native compilation.  For small analytic
formulas, the tape is already tiny and JIT adds ≈1×.

---

## 6 — Scaling Summary

### 6.1  FD ÷ AAD ratio grows with N

The core theoretical result — AAD is O(1) vs FD's O(N) — is clearly visible in the
benchmarks:

```
                     FD ÷ AAD
  N (inputs)    (analytic engines)
  ─────────────────────────────────
       3              18×
       4           8 – 43×
       5              36×
       6              63×
       9             136×
      14             116×
      17             191×
      18             888×
      20             254×
```

At N = 18 (IR Cap), AAD is **888× faster** than FD.  The slight non-monotonicity
(N = 14 at 116× vs N = 9 at 136×) reflects per-instrument overhead differences
(the risky bond has a heavier forward pass than the OIS IRS).

### 6.2  Jacobian extreme: 35,548×

For full N×N Jacobian computation (zero-rate sensitivities, 9 inputs, 100 scenarios),
AAD replay achieves a **35,548× speedup** over FD.  This is because:
1. The zero-rate curve is built directly (no iterative solver) → small tape
2. Replay cost is ~microseconds per sweep
3. FD must rebuild the curve N+1 times per scenario × 100 scenarios

### 6.3  When AAD does *not* help

| Scenario | FD ÷ AAD | Why |
|----------|----------|-----|
| PDE engines (N = 3–4) | 2–3.5× | Backward sweep dominated by grid overhead, not N |
| AAD re-record through solver | 0.1–0.5× | Tape rebuild through iterative solver is more expensive than FD bumping |
| Square Jacobians (N×N, approach 2) | ~1× | Both AAD and FD need ~N sweeps for N×N; no asymptotic advantage |

### 6.4  When to use which approach

| Situation | Recommended approach | Expected speedup |
|-----------|---------------------|-----------------|
| Few inputs (N ≤ 3), no solver | FD | Simpler, no tape overhead |
| Many inputs, analytic engine | AAD replay | 10–900× over FD |
| Scenario sweep, stable graph | AAD replay | 50–250× over FD |
| Full N×N Jacobian, direct curve | AAD replay | 85–35,548× over FD |
| Full Jacobian, through solver | Chain-rule factorisation | 4–89× over FD |
| PDE / MC engines | AAD re-record | 2–36× over FD |
| JIT-eligible + regular loop | JIT-compiled AAD | Additional 1–7× over interpreted AAD |

---

## 7 — Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Python User Code                      │
│   tape = Tape()                                          │
│   tape.registerInput(spot, vol, …)                       │
│   npv = instrument.NPV()        ← forward recorded       │
│   tape.computeAdjoints()        ← backward sweep          │
│   greeks = [tape.derivative(x) for x in inputs]           │
├─────────────────────────────────────────────────────────┤
│                  SWIG Bindings Layer                      │
│   float ↔ xad::AReal<double> typemaps                    │
├─────────────────────────────────────────────────────────┤
│                    QuantLib C++                           │
│   Real = xad::AReal<double>  (via QL_INCLUDE_FIRST)      │
│   All arithmetic recorded on XAD tape                     │
├──────────────────────┬──────────────────────────────────┤
│    XAD (tape-based)  │   Forge JIT (optional)            │
│    Record → Replay   │   Tape → Graph → Native x86-64   │
│    O(1) per sweep    │   Eliminates interpreter overhead  │
│                      │   AVX backend: 4× SIMD parallel    │
└──────────────────────┴──────────────────────────────────┘
```

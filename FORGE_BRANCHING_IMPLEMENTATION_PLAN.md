# Implementation Plan: Overcoming Forge JIT Branching Limitations in QuantLib

## Executive Summary

The Forge JIT backend compiles XAD's tape into a **fixed, straight-line x86-64
kernel** — no runtime branches based on input values.  When QuantLib's C++ code
uses `if (aReal < threshold)`, the branch is evaluated **once at recording time**
and baked in.  Re-evaluating the kernel with different inputs follows the stale
path, producing **wrong results or crashes**.

This document presents a detailed plan to convert QuantLib's branching patterns
into Forge-compatible branchless equivalents, starting with the Callable Bond
(TreeCallableFixedRateBondEngine + HullWhite model) as a proof-of-concept, then
generalizing to other engines.

---

## 1. Architecture Recap

```
┌─────────────────┐   record    ┌──────────┐  compile   ┌──────────────┐
│  QuantLib C++   │ ──────────► │ XAD Tape │ ────────►  │ Forge Kernel │
│  (uses Real)    │             │ (linear) │            │ (x86-64 asm) │
└─────────────────┘             └──────────┘            └──────────────┘
         │                           │                         │
    if (r > x)                  NOT recorded              Baked path only
    ← branch ─►                                           ← wrong ─►
```

**Root cause**: C++ `if` statements on `AReal` values are invisible to the tape.
The tape sees only the arithmetic of whichever path was taken.

**Solution mechanisms available**:

| Mechanism | Where | Semantics |
|---|---|---|
| `ABool::If(trueVal, falseVal)` | XAD JIT extension | Records `OpCode::If` — branchless select at runtime |
| `xad::less(x, y)`, `xad::greater(x, y)` | XAD JIT extension | Returns `ABool` comparison node |
| `OpCode::Min`, `OpCode::Max` | Forge graph | Native branchless min/max (hardware `minsd`/`vminpd`) |
| `max_op<Scalar>`, `min_op<Scalar>` | XAD `BinaryMathFunctors.hpp` | Already branchless for AD types: $(a+b \pm |a-b|)/2$ |
| `smooth_max`, `smooth_min` | XAD `MathFunctions.hpp` | C³-smooth near discontinuity (tunable cutoff `c`) |

---

## 2. Complete Branching Audit: Callable Bond Pricing Path

The callable bond pricing executes this call chain:

```
TreeCallableFixedRateBondEngine::calculateWithSpread(spread)
  └─ DiscretizedCallableFixedRateBond ctor    [schedule/date logic]
  └─ HullWhite::tree(grid)                   [tree construction + φ(t) fitting]
  └─ OneFactorModel::ShortRateTree::setSpread(s)
  └─ TreeLattice::initialize()
  └─ TreeLattice::rollback()                 ← main pricing loop
       ├─ stepback()                          [branch-free: weighted sum × discount]
       ├─ adjustValues()
       │   ├─ preAdjustValuesImpl()           [coupon addition]
       │   ├─ postAdjustValuesImpl()
       │   │   ├─ applyCallability()          ← std::min/std::max on Real  ⚠️
       │   │   └─ addCoupon()
       └─ ShortRateTree::discount(i, j)      [branch-free: exp(-r×dt)]
  └─ TreeLattice::presentValue()             [dot product]
```

### Category A: Data-dependent branches on `Real` (MUST FIX)

| # | Location | Code | Impact | Fix Strategy |
|---|---|---|---|---|
| A1 | `applyCallability()` L180 | `std::min(callPrice, values_[j])` | Exercise decision at every tree node | XAD `min_op` is **already branchless** for AD types — verify it records correctly on JIT tape |
| A2 | `applyCallability()` L185 | `std::max(values_[j], putPrice)` | Put exercise decision | Same as A1 — already branchless |
| A3 | `HullWhite::FittingParameter::value()` L160 in hpp | `a_ < sqrt(QL_EPSILON) ? sigma_*t : sigma_*(1-exp(-a_*t))/a_` | φ(t) fitting — ternary on model param `a` | Convert to `ABool::If` |
| A4 | `HullWhite::discountBondOption()` L95 in cpp | `if (_a < sqrt(QL_EPSILON))` | Bond option vol — branch on `a` | Convert to `ABool::If` |
| A5 | `HullWhite::discountBondOption()` L113 in cpp | `if (_a < sqrt(QL_EPSILON))` (overload) | Same as A4 (different overload) | Convert to `ABool::If` |
| A6 | `HullWhite::discountBondOption()` L125 in cpp | `sqrt(std::max(c, 0.0))` | Floor negative variance | Already branchless via `max_op` |
| A7 | `TreeCallableFixedRateBondEngine::calculateWithSpread()` L69 | `if (s != 0.0)` | Spread setup — skips `setSpread` if zero | Always call `setSpread` (zero spread is harmless) |

### Category B: Structural branches (date/index logic — SAFE for JIT)

| # | Location | Code | Why safe |
|---|---|---|---|
| B1 | `withinNextWeek()` | `t1 <= t2 && t2 <= t1 + dt` | Operates on `Time` (double from dates), not `AReal` |
| B2 | Constructor L64 | `if (withinNextWeek(...) && callabilityDate < couponDate)` | Date comparison — invariant across scenarios |
| B3 | `mandatoryTimes()` | `if (t >= 0.0)` | Fixed time values |
| B4 | `preAdjustValuesImpl()` | `if (couponAdjustments_[i] == CouponAdjustment::pre)` | Enum comparison |
| B5 | `postAdjustValuesImpl()` | `if (t >= 0.0 && isOnTime(t))` | Time comparison |
| B6 | `TreeLattice::partialRollback()` | `if (close(from, to))`, `if (i != iTo)` | Loop/time control |
| B7 | `switch (type())` in `applyCallability()` | `Callability::Call` vs `Callability::Put` | Enum — determined by bond structure |
| B8 | HullWhite::tree() fitting loop | `for (Size i=0; ...)` | Fixed iteration count from time grid |

### Category C: Deep branches (not on direct pricing path — DEFER)

| # | Location | Notes |
|---|---|---|
| C1 | `TrinomialTree` constructor | min/max on integer indices — construction-time only |
| C2 | `convexityBias()` | Static utility, not called during pricing |
| C3 | Various `QL_REQUIRE` checks | Validation only — throw on failure |

---

## 3. Detailed Fix Specifications

### Fix A1/A2: `applyCallability()` — `std::min` / `std::max` on `Real`

**Current code** ([discretizedcallablefixedratebond.cpp](lib/QuantLib/ql/experimental/callablebonds/discretizedcallablefixedratebond.cpp#L175-L189)):
```cpp
case Callability::Call:
    for (j = 0; j < values_.size(); j++)
        values_[j] = std::min(adjustedCallabilityPrices_[i], values_[j]);
    break;
case Callability::Put:
    for (j = 0; j < values_.size(); j++)
        values_[j] = std::max(values_[j], adjustedCallabilityPrices_[i]);
    break;
```

**Analysis**: When `Real = xad::AReal<double>` (AAD mode), `std::min` and
`std::max` resolve to XAD's overloads in `BinaryMathFunctors.hpp`.  For AD
types, `max_op::operator()` uses the **branchless formula**:

```cpp
// isExpr == true for AReal
return (a + b + abs(a - b)) / Scalar(2);   // max
return (a + b - abs(a - b)) / Scalar(2);   // min
```

This formula produces an expression tree that is fully recorded on the tape
with no C++ `if`.  It also has a well-defined subgradient at the tie point
(0.5 each).

**Verification needed**: Confirm that when XAD is compiled with JIT support, the
branchless `max_op`/`min_op` expression tree is correctly translated into Forge
graph nodes.  The expression `(a + b + abs(a - b)) / 2` decomposes into
`OpCode::Add`, `OpCode::Sub`, `OpCode::Abs`, `OpCode::Add`, `OpCode::Mul` (÷2 as
×0.5) — all of which Forge supports.

**Alternatively**: If Forge's `OpCode::Min` and `OpCode::Max` are exposed at the
XAD JIT graph level, the tape could record these directly instead of the
expanded formula.  This would be more efficient (single instruction vs 5 ops)
and should be investigated.

**Action**: **No C++ change needed** — the existing XAD `min`/`max` overloads
are already branchless.  Need to verify JIT graph translation works correctly
by writing a unit test:

```cpp
// Test: min/max with JIT re-evaluation
xad::AD x(5.0), y(3.0);
tape.registerInputs({x, y});
tape.newRecording();
xad::AD z = std::min(x, y);   // Should record branchless expression
tape.registerOutput(z);
// Verify: change x to 1.0, re-evaluate → z should be 1.0
```

**Risk**: LOW — branchless formula is unconditional.

---

### Fix A3: `HullWhite::FittingParameter::value()` — ternary on `a`

**Current code** ([hullwhite.hpp](lib/QuantLib/ql/models/shortrate/onefactormodels/hullwhite.hpp#L158-L160)):
```cpp
Real temp = a_ < std::sqrt(QL_EPSILON) ?
            Real(sigma_*t) :
            Real(sigma_*(1.0 - std::exp(-a_*t))/a_);
```

**Problem**: `a_ < std::sqrt(QL_EPSILON)` is a C++ ternary evaluated at
recording time.  If `a = 0.06` at recording, the `else` branch is baked in.
If the kernel is later re-evaluated with `a ≈ 0`, it computes
`sigma * (1 - exp(-0*t)) / 0` → **division by zero / NaN**.

**Purpose of the branch**: When `a → 0`, the Vasicek mean-reversion formula
`σ(1 - e^{-at})/a` has a removable singularity.  The limit is `σt` (by
L'Hôpital).  The branch avoids numerical instability near zero.

**Proposed fix** — two options:

#### Option 1: `ABool::If` (exact branch tracking)

```cpp
Real value(const Array&, Time t) const override {
    auto cond = xad::less(a_, std::sqrt(QL_EPSILON));
    Real temp_small = sigma_ * t;
    Real temp_large = sigma_ * (1.0 - std::exp(-a_ * t)) / a_;
    Real temp = cond.If(temp_small, temp_large);
    return (forwardRate + 0.5 * temp * temp);
}
```

**Pros**: Exact; Forge evaluates both paths and selects at runtime (like `cmov`).

**Cons**: Both branches are always evaluated.  The `else` branch computes
`(1 - exp(-a*t))/a` which still divides by `a` even when `a ≈ 0`.  Forge will
execute the division producing ±Inf/NaN, then `If` selects the safe path — but
the NaN propagation through Forge's adjoint code **may** corrupt gradients.

**Mitigation**: Compute `temp_large` with a guarded denominator:
```cpp
Real safe_a = std::max(a_, Real(std::sqrt(QL_EPSILON)));  // floor at ε
Real temp_large = sigma_ * (1.0 - std::exp(-safe_a * t)) / safe_a;
```

#### Option 2: Numerically stable reformulation (NO branch needed)

The function `(1 - e^{-x})/x` can be computed stably for all `x ≥ 0` using the
Taylor series or a Padé approximant.  Define:

```cpp
// expm1_over_x(x) = (1 - exp(-x)) / x, stable for x → 0
// = 1 - x/2 + x²/6 - x³/24 + ...  (Taylor around x=0)
Real expm1_over_x(Real x) {
    // std::expm1(-x) = exp(-x) - 1, numerically stable
    // -(exp(-x) - 1) / x = (1 - exp(-x)) / x
    return -std::expm1(-x) / x;   // stable via expm1
}
```

Wait — this still divides by `x = a*t` which can be zero.  Better:

```cpp
// Use the identity: (1 - exp(-at))/a = t * sinch(at/2) * exp(-at/2)
// where sinch(x) = sinh(x)/x → 1 as x → 0
// But this is complex.  Simplest: use the Padé approximant or smooth blend.
```

**Recommended approach**: Use `ABool::If` with guarded denominator (Option 1 +
mitigation).  This is explicit, correct, and minimal-diff.

**Complexity**: LOW — single conditional, called once per tree construction.

---

### Fix A4/A5: `HullWhite::discountBondOption()` — branch on `a`

**Current code** (two overloads at L95 and L113):
```cpp
Real _a = a();
Real v;
if (_a < std::sqrt(QL_EPSILON)) {
    v = sigma()*B(maturity, bondMaturity) * std::sqrt(maturity);
} else {
    v = sigma()*B(maturity, bondMaturity) *
        std::sqrt(0.5*(1.0 - std::exp(-2.0*_a*maturity))/_a);
}
```

**Same pattern as A3**: singularity guard for `a → 0`.

**Proposed fix**: Apply the same `ABool::If` + guarded denominator pattern:

```cpp
Real _a = a();
auto cond = xad::less(_a, Real(std::sqrt(QL_EPSILON)));
Real safe_a = std::max(_a, Real(std::sqrt(QL_EPSILON)));

Real v_small = sigma() * B(maturity, bondMaturity) * std::sqrt(maturity);
Real v_large = sigma() * B(maturity, bondMaturity) *
               std::sqrt(0.5 * (1.0 - std::exp(-2.0 * safe_a * maturity)) / safe_a);
Real v = cond.If(v_small, v_large);
```

For the second overload (L113), apply the same pattern to the more complex 
variance expression at L118-L123, flooring `_a`:

```cpp
Real safe_a = std::max(_a, Real(std::sqrt(QL_EPSILON)));
Real c = exp(-2.0*safe_a*(bondStart-maturity))
       - exp(-2.0*safe_a*bondStart)
       - 2.0*(exp(-safe_a*(bondStart+bondMaturity-2.0*maturity))
             - exp(-safe_a*(bondStart+bondMaturity)))
       + exp(-2.0*safe_a*(bondMaturity-maturity))
       - exp(-2.0*safe_a*bondMaturity);
Real v_large = sigma()/(safe_a*sqrt(2.0*safe_a)) * sqrt(std::max(c, Real(0.0)));
Real v_small = sigma()*B(bondStart, bondMaturity) * std::sqrt(maturity);
Real v = cond.If(v_small, v_large);
```

**Note**: `discountBondOption()` is not called during tree construction or rollback.
It is used for closed-form bond option pricing (e.g., calibration).  For the
TreeCallableFixedRateBondEngine pricing path, this function is **not invoked**.
However, fixing it enables JIT for HullWhite calibration workflows.

**Priority**: MEDIUM — not on the critical pricing path but needed for full
HullWhite JIT support.

---

### Fix A6: `sqrt(std::max(c, 0.0))` in `discountBondOption()` L125

**Already branchless**: `std::max(c, 0.0)` with `c` as `Real` uses XAD's
branchless `max_op`.  No change needed.

---

### Fix A7: Spread check in `TreeCallableFixedRateBondEngine`

**Current code** ([treecallablebondengine.cpp](lib/QuantLib/ql/experimental/callablebonds/treecallablebondengine.cpp#L69)):
```cpp
if (s != 0.0) {
    auto* sr = dynamic_cast<OneFactorModel::ShortRateTree*>(&(*lattice));
    QL_REQUIRE(sr, "Spread is not supported for trees other than OneFactorModel");
    sr->setSpread(s);
}
```

**Problem**: `s` is `Spread` (= `Real`).  When `Real = AReal`, `s != 0.0`
extracts the value and branches.  If the spread changes between recording and
re-evaluation, the kernel may skip or incorrectly apply the spread.

**Proposed fix**: Always call `setSpread(s)`.  Setting a zero spread is a no-op
mathematically (`discount *= exp(-0 * dt) = 1`):

```cpp
auto* sr = dynamic_cast<OneFactorModel::ShortRateTree*>(&(*lattice));
if (sr != nullptr) {
    sr->setSpread(s);    // Always set; zero spread is harmless
}
```

The `dynamic_cast` and null check are on a pointer (structural), not on `Real`
values — safe for JIT.

**Alternative**: Since the engine is constructed with a known lattice type, the
`dynamic_cast` could be moved to construction time + stored as a member.

**Complexity**: TRIVIAL.

---

## 4. Template Abstraction for `ABool::If`

The `ABool::If` pattern requires the XAD JIT extension.  When QuantLib is
compiled with standard (non-JIT) XAD, `xad::less()` and `ABool` may not exist.

**Solution**: A compatibility header that dispatches to the right mechanism:

```cpp
// ql/xad_compat.hpp  (or similar)
#ifndef QL_XAD_COMPAT_HPP
#define QL_XAD_COMPAT_HPP

#include <ql/types.hpp>

namespace QuantLib {

// Branchless conditional select for Real values.
// JIT mode: uses ABool::If (records conditional node in JIT graph).
// Non-JIT mode: uses regular ternary (safe since tape is used once).
#ifdef QL_XAD_JIT
    // ABool::If version
    inline Real conditional(bool cond_passive,
                            const Real& true_val,
                            const Real& false_val) {
        // At JIT record time, we need ABool to track the condition
        // This requires the comparison to produce an ABool, so we
        // need the caller to pass the ABool directly.
        // Overload for ABool:
        // auto result = abool_cond.If(true_val, false_val);
    }

    template <typename ABoolT>
    inline Real conditional(const ABoolT& cond,
                            const Real& true_val,
                            const Real& false_val) {
        return cond.If(true_val, false_val);
    }
#else
    // Standard mode: plain ternary
    inline Real conditional(bool cond,
                            const Real& true_val,
                            const Real& false_val) {
        return cond ? true_val : false_val;
    }
#endif

}  // namespace QuantLib

#endif
```

**Better alternative**: Since `ABool::If` is a template method on a
comparison expression, we can use SFINAE/concepts to make it work regardless:

```cpp
// Works for both ABool (has .If) and plain bool (fallback)
template <typename Cond, typename T>
auto branchless_select(const Cond& cond, const T& t, const T& f)
    -> decltype(cond.If(t, f))
{
    return cond.If(t, f);
}

// Fallback for plain bool
template <typename T>
T branchless_select(bool cond, const T& t, const T& f) {
    return cond ? t : f;
}
```

---

## 5. Implementation Phases

### Phase 1: Verify existing min/max branchlessness (1-2 days)

**Goal**: Confirm that `std::min`/`std::max` on `AReal` already work with JIT.

1. Write a standalone C++ test (using xad-forge test infrastructure):
   - Record `z = min(x, y)` with `x > y`
   - Compile with ForgeBackend
   - Re-evaluate with `x < y`
   - Verify `z` and gradients are correct
2. If min/max work correctly → A1/A2/A6 need **no QuantLib changes**
3. If not → investigate whether Forge's `OpCode::Min`/`OpCode::Max` can be
   emitted directly from the XAD JIT tape translator

**Deliverable**: Test + pass/fail report.

### Phase 2: Fix HullWhite `a → 0` singularity guards (2-3 days)

**Goal**: Convert A3, A4, A5 from `if` to `ABool::If` with guarded denominator.

1. Create `ql/xad_compat.hpp` with `branchless_select()` utility
2. Modify `HullWhite::FittingParameter::Impl::value()` in `hullwhite.hpp`:
   - Compute both branches with `safe_a = max(a, eps)`
   - Select with `branchless_select(xad::less(a_, eps), temp_small, temp_large)`
3. Modify both `HullWhite::discountBondOption()` overloads in `hullwhite.cpp`
4. Apply Fix A7 (always call `setSpread`)
5. Write unit tests:
   - Verify FittingParameter produces correct φ(t) for a = 0.001, 0.06, 0.5
   - Verify gradients dφ/da, dφ/dσ are correct
   - Verify JIT re-evaluation across the a → 0 boundary

**Deliverable**: Modified `hullwhite.hpp`, `hullwhite.cpp`,
`treecallablebondengine.cpp`, + tests.

### Phase 3: End-to-end callable bond JIT test (2-3 days)

**Goal**: Price a callable bond with Forge JIT and verify:
- NPV matches non-JIT pricing (within tolerance)
- Sensitivities (dr, da, dσ) match tape-based AAD
- JIT re-evaluation with different parameters produces correct results

1. Record tape at base parameters (r=0.0465, a=0.06, σ=0.20)
2. Compile with ForgeBackend
3. Re-evaluate at perturbed parameters (r=0.05, a=0.03, σ=0.15)
4. Compare NPV and gradients vs fresh tape-based pricing

**Risk**: The callable bond tree engine has ~40 time steps × 81 nodes × rollback
operations.  The resulting JIT graph will be very large (potentially 100K+ nodes).
Forge compilation time may be significant.  Need to measure and report.

**Deliverable**: Working JIT callable bond example + benchmark.

### Phase 4: Benchmark JIT callable bond (1-2 days)

**Goal**: Add JIT column back to `monte_carlo_bond_benchmarks.py`.

1. Re-add `VENV_JIT` and JIT wheel setup to the benchmark
2. Add H4 — AAD JIT replay mode (compile once, evaluate N_SCENARIOS times)
3. Compare: FD vs AAD replay vs AAD re-record vs JIT replay
4. Report speedup/slowdown

**Expected outcome**: JIT replay should be faster than tape-based replay for
large scenario counts (>1000) but slower for small counts due to compilation
overhead.  The crossover point depends on graph size.

### Phase 5: Generalize to other engines (ongoing)

**Goal**: Extend the branchless conversion to other QuantLib pricing engines.

| Engine | Branching complexity | Effort |
|---|---|---|
| `TreeSwaptionEngine` | Low — similar exercise decision only | 1-2 days |
| `TreeCapFloorEngine` | Low — similar | 1-2 days |
| `FDBlackScholesVanillaEngine` | Medium — boundary conditions | 3-5 days |
| `MCEuropeanEngine` | Low — payoff only | 1 day |
| `MCBarrierEngine` | High — barrier monitoring per path | 5-10 days |
| `MarkovFunctional` | Very high — interpolation + solver | Not recommended |

---

## 6. Risk Analysis

### Risk 1: `ABool::If` evaluates both branches — NaN contamination

**Severity**: HIGH

When `ABool::If(true_val, false_val)` is used, Forge compiles code that
computes **both** `true_val` and `false_val`, then selects.  If one branch
produces NaN/Inf (e.g., division by zero when `a ≈ 0`), the NaN may propagate
through the gradient computation even though the NaN value is not selected by
the forward pass.

**Mitigation**: Always use guarded denominators (`safe_a = max(a, eps)`) in
both branches so neither can produce NaN.  This introduces a tiny numerical
perturbation (< 1e-8) for the branch that is ultimately discarded.

### Risk 2: JIT graph size explosion

**Severity**: MEDIUM

A 40-step trinomial tree with 81 nodes at the widest point generates O(40 ×
81 × ops_per_node) graph nodes.  Each `stepback()` involves 3 multiplies + 2
adds + 1 multiply (discount) per tree node, plus the `min`/`max` exercise
decisions in `applyCallability()`.  Rough estimate: ~30,000–50,000 Forge graph
nodes.

**Impact**: Compilation time may be 50–200ms (one-time cost).  The compiled
kernel should be fast to evaluate (~10× faster than tape interpretation for
large evaluation counts).

**Mitigation**: Benchmark compilation time.  If excessive, consider reducing
tree steps (20 instead of 40) for the JIT path.

### Risk 3: `ABool` API not available in open-source XAD

**Severity**: HIGH

The `xad::less()`, `xad::greater()`, and `ABool::If()` functions are part of
XAD's proprietary JIT extension.  They are **not present** in the open-source
`lib/xad/src/XAD/` headers.  The JIT headers (`JITGraph.hpp`,
`JITBackendInterface.hpp`, `JITCompiler.hpp`) are included by xad-forge's
`ForgeBackend.hpp` but do not exist in the workspace's `lib/xad/` tree.

**Implication**: The ABool-based fixes require the proprietary XAD JIT
extension to be installed.  QuantLib must be conditionally compiled:

```cpp
#ifdef XAD_HAS_JIT   // or similar macro
    auto cond = xad::less(a_, eps);
    Real temp = cond.If(temp_small, temp_large);
#else
    Real temp = (a_ < eps) ? temp_small : temp_large;
#endif
```

**Mitigation**: Use the `branchless_select()` template from Section 4, which
falls back to plain `bool` dispatch when `ABool` is unavailable.

### Risk 4: Expression template lifetime issues in Both-branch evaluation

**Severity**: MEDIUM

When both branches of `ABool::If(t, f)` are evaluated, any expression-template
temporaries must remain alive until the `If` node is recorded.  The same
dangling-reference issue fixed in `CALLABLE_BOND_CRASH_FIX.md` applies here:
always assign branch expressions to `Real` variables, never use `auto`.

**Mitigation**: Enforce the `Real temp_small = ...; Real temp_large = ...;`
pattern — never `auto`.

---

## 7. Alternative Approaches Considered

### Alternative A: Branchless mathematical reformulations

Instead of `ABool::If`, reformulate the math to eliminate all branches.

For the HullWhite `(1 - exp(-at))/a` singularity:

```cpp
// Use Horner form of Taylor series for (1 - exp(-x))/x:
//   f(x) = 1 - x/2 + x²/6 - x³/24 + ...
// Blend: for |x| < 0.01, use Taylor; else use standard formula
// But the blend itself is a branch...
```

**Verdict**: Not viable without `ABool::If` for the blend.  The Taylor approach
works for small x but introduces its own branch to select between Taylor and
standard formula.

### Alternative B: Re-record the full tree per evaluation

Instead of JIT, use the AAD re-record approach: for each evaluation, rebuild
all QL objects + tape.

**Verdict**: Already implemented as H3 in the benchmark.  Works but is ~10×
slower than replay.  JIT exists specifically to avoid this cost.

### Alternative C: Automatic branch detection + graph splitting

Split the Forge graph at branch points into multiple sub-kernels.  At runtime,
evaluate the branch condition, then dispatch to the appropriate sub-kernel.

**Verdict**: This is a Forge compiler redesign — orders of magnitude more work
than fixing individual branches.  Not practical for this project.

### Alternative D: Whole-branch smoothing

Replace `if (a < ε) expr1 else expr2` with a smooth blend:

```cpp
Real blend = smooth_step(a, sqrt(QL_EPSILON));  // 0→1 transition
Real temp = (1 - blend) * (sigma * t) + blend * (sigma * (1 - exp(-a*t)) / max(a, eps));
```

**Verdict**: Introduces numerical error proportional to the smoothing width.
For the FittingParameter, this changes φ(t) slightly near `a = 0`, which
affects the term-structure fit.  Not recommended for production pricing.

---

## 8. Testing Strategy

### Unit tests (per fix)

| Test | What it verifies |
|---|---|
| `MinMaxJIT` | `std::min`/`std::max` on `AReal` produce correct JIT graph |
| `FittingParameterJIT` | φ(t) correct across a ∈ {1e-10, 1e-5, 0.06, 0.5} with JIT |
| `DiscountBondOptionJIT` | Bond option vol correct across a range |
| `SpreadAlwaysApplied` | Zero/nonzero spread both work |

### Integration tests

| Test | What it verifies |
|---|---|
| `CallableBondJIT_NPV` | NPV matches tape-based pricing (tol < 1e-10) |
| `CallableBondJIT_Gradients` | dr, da, dσ match FD with tight tolerance |
| `CallableBondJIT_ReEvaluate` | JIT re-evaluation with different params |
| `CallableBondJIT_ManyScenarios` | 1000 scenarios, all NPVs and gradients correct |

### Regression tests

Compare all existing QuantLib test results before and after the changes.  The
branchless changes should produce **bit-for-bit identical** results in non-JIT
mode (the `branchless_select` fallback uses the same ternary).

---

## 9. Estimated Timeline

| Phase | Work | Duration | Dependencies |
|---|---|---|---|
| 1 | Verify min/max branchlessness | 1-2 days | None |
| 2 | Fix HullWhite branching | 2-3 days | Phase 1 |
| 3 | End-to-end JIT callable bond | 2-3 days | Phase 2 |
| 4 | Benchmark | 1-2 days | Phase 3 |
| 5 | Generalize to other engines | ongoing | Phase 3 |

**Total for Phases 1-4**: ~8-10 working days.

---

## 10. Success Criteria

1. **Correctness**: JIT-compiled callable bond NPV matches tape-based NPV within
   machine epsilon (~1e-12 relative error)
2. **Gradients**: JIT adjoint derivatives match FD sensitivities within 1e-6
   relative error (limited by FD accuracy, not AAD)
3. **Re-evaluation**: JIT kernel produces correct results when re-evaluated
   with different (r, a, σ) parameters, including across the `a → 0` boundary
4. **Performance**: JIT re-evaluation speedup > 1× vs tape interpretation for
   ≥ 1000 scenario evaluations
5. **Regression**: All existing QuantLib tests pass unchanged

---

## Appendix A: Forge OpCode Reference (relevant subset)

| OpCode | Semantics | x86-64 instruction | Differentiable? |
|---|---|---|---|
| `Add` | a + b | `addsd` / `vaddpd` | ✅ da=1, db=1 |
| `Sub` | a - b | `subsd` / `vsubpd` | ✅ da=1, db=-1 |
| `Mul` | a × b | `mulsd` / `vmulpd` | ✅ da=b, db=a |
| `Div` | a / b | `divsd` / `vdivpd` | ✅ da=1/b, db=-a/b² |
| `Exp` | eᵃ | `call exp` | ✅ da=eᵃ |
| `Log` | ln(a) | `call log` | ✅ da=1/a |
| `Sqrt` | √a | `sqrtsd` / `vsqrtpd` | ✅ da=1/(2√a) |
| `Abs` | \|a\| | bit mask | ✅ da=sign(a) |
| `Min` | min(a,b) | `minsd` / `vminpd` | ✅ subgradient |
| `Max` | max(a,b) | `maxsd` / `vmaxpd` | ✅ subgradient |
| `If` | c ? a : b | `blendvpd` / `cmov` | ✅ propagates to selected branch |
| `CmpLT` | a < b | `cmpltsd` | Bool (no gradient) |

## Appendix B: File Change Summary

| File | Change type | Lines affected |
|---|---|---|
| `ql/xad_compat.hpp` | **NEW** | ~30 lines — `branchless_select()` utility |
| `ql/models/shortrate/onefactormodels/hullwhite.hpp` | MODIFY | L158-160 — FittingParameter::value() |
| `ql/models/shortrate/onefactormodels/hullwhite.cpp` | MODIFY | L95-100, L113-130 — discountBondOption() |
| `ql/experimental/callablebonds/treecallablebondengine.cpp` | MODIFY | L69-73 — spread check |
| `ql/experimental/callablebonds/discretizedcallablefixedratebond.cpp` | NONE | min/max already branchless |

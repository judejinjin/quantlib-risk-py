# Callable Bond Benchmark Crash Fix

## Summary

`benchmarks/monte_carlo_bond_benchmarks.py` crashed with `SIGSEGV` (exit code -11) in the
non-JIT worker subprocess during `tape.computeAdjoints()` inside `_bond_aad_record`.
The root cause was a dangling reference bug in XAD expression templates triggered by
use of `auto` in a lambda within `DiscretizedCallableFixedRateBond`'s constructor.

## Root Cause

### File

`lib/QuantLib/ql/experimental/callablebonds/discretizedcallablefixedratebond.cpp`
— the `calcDiscountFactorInclSpread` lambda (formerly around line 79).

### Mechanism

With XAD, `Real` is `xad::AReal<double>`. Arithmetic on `AReal` values does **not**
return an `AReal` — it returns a lightweight *expression template*
(e.g. `UnaryExpr<BinaryExpr<ADVar, ADVar>>`). At the leaves of these expression
trees sit `ADVar` objects, which store a **const reference** to their `AReal` operand:

```cpp
// lib/xad/src/XAD/Literals.hpp line 349
areal_type const& ar_;
```

The original code used `auto` throughout the lambda:

```cpp
auto calcDiscountFactorInclSpread = [&termStructure, spread](Date date) {
    auto time = termStructure->timeFromReference(date);
    auto zeroRateInclSpread =
        termStructure->zeroRate(date, termStructure->dayCounter(),
                                Continuous, NoFrequency) + spread;
    auto df = std::exp(-zeroRateInclSpread * time);
    return df;
};
auto dfTillCallDate = calcDiscountFactorInclSpread(callabilityDate);
auto dfTillCouponDate = calcDiscountFactorInclSpread(couponDate);
```

Because of `auto`:

1. `time`, `zeroRateInclSpread`, and `df` are deduced as expression template types,
   not `AReal`.
2. The lambda's return type is deduced as an expression template containing `ADVar`
   references to the lambda's stack-local `AReal` temporaries (created implicitly
   during intermediate evaluations).
3. After the lambda returns, those stack locals are destroyed, but the returned
   expression template still holds references to them — **dangling references**.
4. When the caller evaluates `dfTillCallDate`, it reads dead stack memory, corrupting
   the XAD tape with garbage slot indices.
5. The subsequent `tape.computeAdjoints()` traverses these invalid slots and
   segfaults.

### Why the crash was intermittent

- **First call succeeds, second crashes**: On the first invocation the dead stack memory
  contains benign residuals. On subsequent calls, leftover data from the previous tree
  computation produces out-of-bounds tape slot references.
- **Only with active callabilities**: The affected code path is reached only when
  `callabilityDate < couponDate` within the next week, which only occurs when callabilities
  are in the future relative to the evaluation date. With `calcDate = 2016-08-16` the bond
  has matured and this path is never hit.

### Valgrind confirmation

Valgrind reported invalid reads inside the `DiscretizedCallableFixedRateBond` constructor
at addresses "on thread 1's stack, 312/320/232 bytes below stack pointer" — the classic
signature of reading deallocated stack memory.

## Fix

Changed all `auto` types to explicit `Real` and added a `-> Real` return type on the
lambda. This forces expression template evaluation while locals are still alive:

```cpp
auto calcDiscountFactorInclSpread = [&termStructure, spread](Date date) -> Real {
    Real time = termStructure->timeFromReference(date);
    Real zeroRateInclSpread =
        termStructure->zeroRate(date, termStructure->dayCounter(),
                                Continuous, NoFrequency) + spread;
    Real df = std::exp(-zeroRateInclSpread * time);
    return df;
};
Real dfTillCallDate = calcDiscountFactorInclSpread(callabilityDate);
Real dfTillCouponDate = calcDiscountFactorInclSpread(couponDate);
```

A comment in the source explains the rationale.

## Verification

| Test | Result |
|------|--------|
| Non-JIT 100-scenario loop | Pass — stable derivatives (~-440) |
| Non-JIT benchmark worker (`--worker 1`) | Pass — all three modes (FD, AAD replay, AAD record) |
| All reproduction scripts | Pass |
| Codebase-wide search for similar `auto` patterns | No other instances found |

## JIT Worker (Separate Issue)

The JIT worker (`--worker 2`) still crashes after this fix. This is an unrelated,
pre-existing issue in the experimental Forge/xad-forge JIT system:

- **Branching limitation**: Forge bakes C++ `if` decisions at record time.
  `TreeCallableFixedRateBondEngine` uses extensive data-dependent branching in
  its lattice construction, violating this constraint.
- **Wrong derivatives**: Even when JIT doesn't crash (small iteration counts), it
  produces incorrect derivatives (0.0 or garbage values).
- **Documented as experimental**: Both `lib/xad-forge/README.md` and `lib/forge/README.md`
  note the system is a proof-of-concept that has not been hardened for production use.

This is a distinct issue that requires changes to Forge/xad-forge, not QuantLib.

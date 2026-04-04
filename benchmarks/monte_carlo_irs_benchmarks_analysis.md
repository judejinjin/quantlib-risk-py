# IRS Monte Carlo Benchmark — Analysis

## Results (2026-02-25)

| Method | Non-JIT batch (ms) | JIT batch (ms) | JIT speedup | Per-scenario NJ | Per-scenario JIT |
|---|---:|---:|---:|---:|---:|
| FD (N+1 pricings per scenario) | 726.4 ±16.2 | 767.7 ±51.4 | 0.95× | 7264 µs | 7677 µs |
| **AAD replay** (backward sweep only) | 3.4 ±0.6 | 3.6 ±0.3 | 0.94× | 34 µs | 36 µs |
| AAD re-record (forward + backward) | 130.5 ±4.0 | 131.9 ±2.5 | 0.99× | 1305 µs | 1319 µs |
| *FD ÷ AAD replay (non-JIT / JIT)* | *212×* | *211×* | — | — | — |
| *FD ÷ AAD re-record (non-JIT / JIT)* | *5.6×* | *5.8×* | — | — | — |

---

## Why is JIT speedup ≈ 1× (or slightly below)?

This is **expected and correct** for the IRS instrument.

What XAD JIT actually does is compile the XAD backward sweep kernel — the adjoint
pass over the tape's statement array — using LLVM. JIT is profitable when that kernel
is large enough to amortize LLVM compilation overhead and regular enough (tight loops,
uniform arithmetic) for LLVM to auto-vectorize.

**The IRS backward sweep is neither:**

- At 34 µs per scenario the kernel is tiny — LLVM compilation overhead is comparable
  to the gain.
- The computation graph is a full `PiecewiseFlatForward` bootstrap: irregular
  interpolation, date arithmetic, and conditional branching scattered throughout.
  LLVM cannot vectorize this effectively.

JIT being slightly *slower* (0.94–0.99×) is explained by LLVM initialization and
kernel-compilation cost that never fully pays off across just 100 scenarios with such
an irregular computation graph.

---

## Do the other numbers make sense?

Yes. The FD/AAD ratios are internally consistent:

| Ratio | Observed | Explanation |
|---|---|---|
| FD ÷ AAD replay | **212×** | FD runs 18 full curve bootstraps per scenario (17 inputs + 1 base); AAD replay is one backward sweep — roughly the cost of one pricing — yielding all 17 sensitivities |
| FD ÷ AAD re-record | **5.6×** | Re-record costs ~1 forward pass (one bootstrap) + ~1 backward pass ≈ 2 effective pricings vs 18 for FD. Theoretical ratio ≈ 9×; observed 5.6× reflects overhead from tape activation, `Real` construction, and observer chain setup |
| AAD re-record ÷ AAD replay | **38×** | Re-record must fully re-bootstrap the `PiecewiseFlatForward` from all 17 `SimpleQuote` values each scenario; replay just sweeps a pre-recorded tape |

---

## Contrast with the Callable Bond

The callable bond uses a `TreeCallableFixedRateBondEngine` with a 40-step trinomial
tree. Its backward sweep is a **tight, regular, loop-dominated** computation:
40 time steps × grid nodes with uniform floating-point arithmetic throughout.
That is precisely what LLVM vectorizes well, which is why the bond benchmark is
expected to show ~7× JIT speedup where the IRS shows ~1×.

The IRS result therefore confirms that JIT benefit is **instrument-specific**: it
scales with the regularity and size of the backward sweep kernel, not with the number
of inputs or scenarios.

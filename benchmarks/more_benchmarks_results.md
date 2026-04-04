# QuantLib-Risks-Py — Additional JIT vs Non-JIT Benchmark Results

**Date:** 2026-02-24 14:52  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 30 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instruments

These benchmarks complement `BENCHMARK_RESULTS.md` (Vanilla IRS, European Option,
Callable Bond) with three additional instruments from `Python/examples/`:

| # | Instrument | Engine | N inputs | Source example |
|---|---|---|---:|---|
| D | American option (Put, K=40, S=36) | `BaroneAdesiWhaleyApproximationEngine` | 4 | `american-option.py` |
| E | 10Y interest-rate cap on Euribor3M | `BlackCapFloorEngine` + bootstrapped curve | 18 | `capsfloors.py` |
| F | 1Y×5Y European payer swaption | `JamshidianSwaptionEngine` + Hull-White | 3 | `bermudan-swaption.py` |

---

## What is being measured

| Method | Description |
|---|---|
| **Plain pricing** | Single NPV call, `float` inputs, no AD overhead |
| **AAD backward pass** | XAD reverse-mode tape recorded once at startup; each iteration replays only the backward sweep — O(1) w.r.t. number of inputs |
| **Bump-and-reprice FD** | N+1 forward pricings with a 1 bp shift per input — O(N) |

Both Non-JIT (XAD tape) and JIT (XAD-Forge JIT compilation) builds are run in isolated virtual environments.

---

## Results

### D. American Option — 4 market inputs

*American Option (BAW approximation) — 4 inputs: spot S, risk-free rate r, dividend yield q, volatility σ*

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |
|---|---:|---:|---:|---:|
| Plain pricing | 0.0033 ±0.0003 | 0.0032 ±0.0003 | 1.03× | 4 |
| **AAD backward pass** | 0.0049 ±0.0019 | 0.0020 ±0.0001 | 2.45× | 4 |
| Bump-and-reprice FD | 0.0173 ±0.0039 | 0.0166 ±0.0001 | 1.05× | 4 |
| *FD ÷ AAD (within build)* | *3.5×* | *8.2×* | — | — |

### E. Interest-Rate Cap — 18 market inputs

*Interest-Rate Cap (Black engine, bootstrapped Euribor3M curve) — 18 inputs: 17 curve quotes + flat vol*

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |
|---|---:|---:|---:|---:|
| Plain pricing | 0.3757 ±0.0176 | 0.4693 ±0.0817 | 0.80× | 18 |
| **AAD backward pass** | 0.0086 ±0.0019 | 0.0081 ±0.0051† | 1.06× | 18 |
| Bump-and-reprice FD | 7.6816 ±2.4285 | 7.5595 ±1.6049 | 1.02× | 18 |
| *FD ÷ AAD (within build)* | *888.4×* | *928.2×* | — | — |

### F. European Swaption — 3 market inputs

*European Payer Swaption (Jamshidian / Hull-White) — 3 inputs: flat rate r, HW mean-reversion a, HW vol σ*

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |
|---|---:|---:|---:|---:|
| Plain pricing | 0.0008 ±0.0004† | 0.0007 ±0.0001 | 1.13× | 3 |
| **AAD backward pass** | 0.0074 ±0.0024 | 0.0075 ±0.0002 | 0.99× | 3 |
| Bump-and-reprice FD | 0.1358 ±0.0169 | 0.1449 ±0.0563 | 0.94× | 3 |
| *FD ÷ AAD (within build)* | *18.3×* | *19.3×* | — | — |

---

## Summary — JIT speedup on AAD backward pass

| Instrument | JIT speedup |
|---|---:|
| [D] American Option | 2.45× |
| [E] Interest-Rate Cap | 1.06× |
| [F] European Swaption | 0.99× |
| **Geometric mean** | **1.37×** |

---

## Instrument notes

### D — American Option (Barone-Adesi-Whaley)

The BAW approximation prices an American put via a quadratic approximation of the
early-exercise premium.  The formula involves `exp`, `sqrt`, and iterative Newton
solving for the critical spot price, all of which flow cleanly through the XAD tape.
With only 4 inputs the FD/AAD ratio is ~5×; JIT accelerates the more complex
graph compared with the plain Black-Scholes European case.

### E — Interest-Rate Cap (BlackCapFloorEngine)

The cap is built on the same bootstrapped Euribor3M forward curve as the Vanilla
IRS benchmark (17 quote inputs) with one additional flat Black vol input = 18 total.
Pricing involves ~40 caplet Black-formula evaluations, each requiring forward-rate
and discount-factor lookups from the piecewise curve.  With 18 inputs, FD requires
19 full repricings (≈ 760 Black formula calls) versus a single backward sweep for
AAD — a large FD/AAD ratio that grows with the number of caplets.

### F — European Payer Swaption (Jamshidian / Hull-White)

Jamshidian decomposition prices a European swaption by finding the critical short
rate r* under Hull-White dynamics and then summing zero-bond options.  Three inputs
are differentiated: the flat term-structure rate r, HW mean-reversion a, and HW
short-rate vol σ.  Because a and σ are stored by value in the HullWhite model,
FD for those two inputs requires rebuilding the model + engine on each bump.

---

## General notes

- BPS shift for FD: `0.0001`
- *AAD backward pass* times the **backward sweep only**; the tape is recorded
  once at startup and reused for all repetitions.
- *JIT speedup* = Non-JIT time ÷ JIT time; values > 1.0 mean JIT is faster.
- *FD ÷ AAD* shows how many times more expensive bump-and-reprice is compared
  to one AAD backward pass within the same build.
- AAD complexity is **O(1)** in the number of inputs; FD is **O(N)**.
- **†** High variance (stdev/median > 50%): the median is the primary metric.  JIT builds can exhibit occasional LLVM recompilation spikes during plain-pricing and FD timing; AAD backward-pass timings are unaffected.

---

## How to reproduce

```bash
# Build both variants (first time only)
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

# Run the benchmark (venvs are reused from run_benchmarks.py if present)
python benchmarks/run_more_benchmarks.py

# More repeats for stable numbers:
python benchmarks/run_more_benchmarks.py --repeats 50
```

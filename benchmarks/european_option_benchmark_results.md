# European Option — FD vs AAD vs AAD+JIT Benchmark Results

**Date:** 2026-02-27 10:20  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 30 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

| Parameter | Value |
|---|---|
| Type | European Call |
| Spot (S) | 7.0 |
| Strike (K) | 8.0 |
| Volatility (σ) | 0.1 |
| Risk-free rate (r) | 0.05 |
| Dividend yield (q) | 0.05 |
| Maturity | 1 year |
| Engine | `AnalyticEuropeanEngine` (BSM closed-form) |
| JIT eligible | **Yes** — no branching in analytic formula |

---

## Greeks validation (AAD vs FD)

NPV = 0.0303344207

| Input | FD (1 bp bump) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S (spot) | 0.09511176 | 0.09509987 | 1.19e-05 |
| q (div yield) | -0.66872443 | -0.66934673 | 6.22e-04 |
| σ (vol) | 1.17251409 | 1.17147727 | 1.04e-03 |
| r (rate) | 0.63940316 | 0.63884610 | 5.57e-04 |

---

## Timing results

N = 4 market inputs, 30 repetitions, BPS = 0.0001

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing (1 NPV) | 0.0047 ±0.0086 | 0.0041 ±0.0003 | 1.13× |
| Bump-and-reprice FD (N+1 NPVs) | 0.0242 ±0.0170 | 0.0228 ±0.0042 | 1.06× |
| **AAD backward pass** | 0.0014 ±0.0001 | 0.0013 ±0.0001 | 1.08× |
| *FD ÷ AAD* | *17.1×* | *17.3×* | — |

---

## Analysis

The **AnalyticEuropeanEngine** uses the Black-Scholes-Merton closed-form formula,
which involves only smooth mathematical operations (`exp`, `log`, `erfc`) with no
branching (if/else) in the computation graph. This makes it an ideal candidate for
JIT compilation — the XAD-Forge compiler can translate the entire AAD tape to
optimised native machine code.

With only 4 inputs, both FD (4+1 = 5 forward pricings) and AAD (1 backward
sweep) are fast.  The AAD advantage grows with the number of inputs (O(1) vs O(N)).

---

## How to reproduce

```bash
# Build both variants (first time only)
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

# Run this benchmark
python benchmarks/european_option_benchmarks.py
python benchmarks/european_option_benchmarks.py --repeats 50
```

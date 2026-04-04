# Basket Option — FD vs AAD Benchmark Results

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
| Type | European Basket Call (MaxBasketPayoff) |
| Assets | 2 |
| Spot (S1, S2) | 7.0, 7.0 |
| Strike (K) | 8.0 |
| Volatility (σ1, σ2) | 0.1, 0.1 |
| Risk-free rate (r) | 0.05 |
| Dividend yields (q1, q2) | 0.05, 0.05 |
| Correlation (ρ) | 0.5 |
| Maturity | 1 year |
| Engine | `MCEuropeanBasketEngine` (low-discrepancy, 32768 samples) |
| JIT eligible | **No** — MC engine has branching (random path generation) |

---

## Greeks validation (AAD vs FD)

NPV = 0.0533985523

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S1 (spot1) | 0.08059217 | 0.08055131 | 4.09e-05 |
| S2 (spot2) | 0.08068309 | 0.08068309 | 1.29e-12 |
| σ1 (vol1) | 1.00452223 | 1.00327787 | 1.24e-03 |
| σ2 (vol2) | 1.00494371 | 1.00429483 | 6.49e-04 |
| r (rate) | 1.07605257 | 1.07524224 | 8.10e-04 |

---

## Timing results

N = 5 inputs, 30 repetitions, BPS = 0.0001

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing (1 NPV) | 42.4261 ±5.6013 | 38.9493 ±7.7722 | 1.09× |
| Bump-and-reprice FD (N+1 NPVs) | 254.5055 ±43.9056 | 229.3918 ±21.3338 | 1.11× |
| **AAD backward pass** | 7.1681 ±0.7048 | 6.6898 ±1.4788 | 1.07× |
| *FD ÷ AAD* | *35.5×* | *34.3×* | — |

---

## Analysis

The **MCEuropeanBasketEngine** uses Monte Carlo simulation with pseudo-random
or quasi-random sequences.  The path generation and payoff evaluation involve
branching (max/min payoffs, early termination checks), so the engine is **not
eligible for JIT compilation** — the Forge compiler cannot trace through
data-dependent branches in the MC loop.

The JIT column therefore shows the Forge build falling back to interpreted
AD, and the JIT speedup should be ≈ 1.0×.

Despite this, AAD still provides a benefit over FD: one backward sweep
gives all 5 Greeks simultaneously, whereas FD requires 5+1 = 6
full MC simulations.  For MC engines (which are typically the slowest),
this O(1) vs O(N) advantage is the primary value of AAD.

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)
python benchmarks/basket_option_benchmarks.py
```

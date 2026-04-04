# Swing Option — FD vs AAD Benchmark Results

**Date:** 2026-02-27 10:49  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 30 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

| Parameter | Value |
|---|---|
| Type | Swing Option (VanillaForwardPayoff, Call) |
| Spot (S) | 30.0 |
| Strike (K) | 30.0 (forward) |
| Volatility (σ) | 0.2 |
| Risk-free rate (r) | 0.05 |
| Dividend yield (q) | 0.0 |
| Exercise dates | 31 (Jan 1–31, 2019) |
| Min exercises | 0 |
| Max exercises | 31 |
| Engine | `FdSimpleBSSwingEngine` (PDE finite-differences) |
| JIT eligible | **No** — PDE solver has branching (boundary conditions, exercise logic) |

---

## Greeks validation (AAD vs FD)

NPV = 47.1723310913

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S (spot) | 18.37786964 | 18.37786964 | 8.98e-10 |
| σ (vol) | 197.15762483 | 197.15714711 | 4.78e-04 |
| r (rate) | 147.09084277 | 147.08300931 | 7.83e-03 |

---

## Timing results

N = 3 inputs, 30 repetitions, BPS = 0.0001

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing (1 NPV) | 27.2453 ±3.5845 | 25.9916 ±4.1015 | 1.05× |
| Bump-and-reprice FD (N+1 NPVs) | 110.5523 ±7.6388 | 111.4978 ±11.3698 | 0.99× |
| **AAD backward pass** | 31.8797 ±2.8503 | 31.9550 ±2.5163 | 1.00× |
| *FD ÷ AAD* | *3.5×* | *3.5×* | — |

---

## Analysis

The **FdSimpleBSSwingEngine** solves a PDE on a finite-difference grid with
conditional logic for boundary conditions and early-exercise decisions at each
of the 31 exercise dates.  This branching makes the engine **not eligible for
JIT compilation** — the Forge compiler cannot trace through data-dependent branches.

The JIT column shows the Forge build falling back to interpreted AD (expect
speedup ≈ 1.0×).

With 3 inputs, FD requires 4 = 4 PDE solves while AAD needs only
1 backward sweep.  The PDE solver is moderately expensive, so the FD ÷ AAD
ratio shows the efficiency gain from AAD even without JIT.

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)
python benchmarks/swing_option_benchmarks.py
```

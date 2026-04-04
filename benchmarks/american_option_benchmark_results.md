# American Option — FD vs AAD vs AAD+JIT Benchmark Results

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
| Type | American Put |
| Spot (S) | 36.0 |
| Strike (K) | 40.0 |
| Volatility (σ) | 0.2 |
| Risk-free rate (r) | 0.06 |
| Dividend yield (q) | 0.0 |
| Maturity | 1 year |
| Inputs | 4 (S (spot), r (rate), q (div yield), σ (vol)) |

---

## BAW — `BaroneAdesiWhaleyApproximationEngine`

JIT eligible: Yes  
NPV = 4.4622354670

### Greeks validation

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S (spot) | -0.69066864 | -0.69067319 | 4.55e-06 |
| r (rate) | -10.36123649 | -10.36827519 | 7.04e-03 |
| q (div yield) | 9.30789573 | 9.30261661 | 5.28e-03 |
| σ (vol) | 11.00166807 | 10.99871367 | 2.95e-03 |

### Timing

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing | 0.0039 ±0.0002 | 0.0035 ±0.0046 | 1.13× |
| Bump-and-reprice FD (N+1) | 0.0198 ±0.0101 | 0.0175 ±0.0001 | 1.13× |
| **AAD backward pass** | 0.0025 ±0.0001 | 0.0022 ±0.0081 | 1.13× |
| *FD ÷ AAD* | *8.0×* | *8.0×* | — |

---

## Bjerksund-Stensl — `BjerksundStenslandApproximationEngine`

JIT eligible: Yes  
NPV = 4.4556626587

### Greeks validation

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S (spot) | -0.70289384 | -0.70289832 | 4.48e-06 |
| r (rate) | -9.71304756 | -9.72031378 | 7.27e-03 |
| q (div yield) | 8.71414555 | 8.70843841 | 5.71e-03 |
| σ (vol) | 10.59598840 | 10.59297879 | 3.01e-03 |

### Timing

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing | 0.0427 ±0.0304 | 0.0180 ±0.0044 | 2.38× |
| Bump-and-reprice FD (N+1) | 0.1279 ±0.0585 | 0.0889 ±0.0075 | 1.44× |
| **AAD backward pass** | 0.0030 ±0.0002 | 0.0026 ±0.0052 | 1.16× |
| *FD ÷ AAD* | *43.0×* | *34.5×* | — |

---

## FD-BS (PDE) — `FdBlackScholesVanillaEngine`

JIT eligible: **No** (PDE branching)  
NPV = 4.4887013833

### Greeks validation

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S (spot) | -0.69604618 | 10.27923161 | 1.10e+01 |
| r (rate) | -10.38348863 | -10.39027756 | 6.79e-03 |
| q (div yield) | 9.09112495 | 0.00009164 | 9.09e+00 |
| σ (vol) | 10.97816159 | 0.00000000 | 1.10e+01 |

### Timing

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing | 70.7425 ±13.8123 | 71.6810 ±5.7269 | 0.99× |
| Bump-and-reprice FD (N+1) | 354.2780 ±23.4962 | 380.0037 ±71.0176 | 0.93× |
| **AAD backward pass** | 168.8591 ±11.2612 | 170.9924 ±34.1668 | 0.99× |
| *FD ÷ AAD* | *2.1×* | *2.2×* | — |

---

## QD+ — `QdPlusAmericanEngine`

JIT eligible: Yes  
NPV = 4.4997148851

### Greeks validation

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| S (spot) | -0.69811074 | 10.26212974 | 1.10e+01 |
| r (rate) | -10.28946049 | -10.29616883 | 6.71e-03 |
| q (div yield) | 8.99721347 | 8.99250184 | 4.71e-03 |
| σ (vol) | 10.96302492 | 0.00000000 | 1.10e+01 |

### Timing

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing | 0.0342 ±0.0152 | 0.0387 ±0.0100 | 0.88× |
| Bump-and-reprice FD (N+1) | 0.1834 ±0.0737 | 0.1816 ±0.0364 | 1.01× |
| **AAD backward pass** | 0.0294 ±0.0027 | 0.0309 ±0.0167 | 0.95× |
| *FD ÷ AAD* | *6.2×* | *5.9×* | — |

---

## Analysis

The **analytic approximation** engines (BAW, Bjerksund-Stensland, QD+) use
closed-form or quasi-analytic formulae with no branching in the computation
graph, making them **JIT eligible**.  The **FdBlackScholesVanillaEngine** solves
a PDE on a grid with conditional logic (boundary conditions, early exercise
checks), so it is **not JIT eligible** — the Forge compiler cannot trace
through branches.

With only 4 inputs (S, r, q, σ), FD requires 5 forward pricings while
AAD needs only 1 backward sweep regardless of input count.

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)
python benchmarks/american_option_benchmarks.py
```

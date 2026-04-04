# Risky Bond — OIS + CDS Bootstrapped Benchmark Results

**Date:** 2026-02-27 12:04  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 30 (median reported)  
**OIS rates:** US Treasury daily par yield curve (2026-02-26)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

| Parameter | Value |
|---|---|
| Type | 5Y Fixed-Rate Coupon Bond |
| Notional | 100 |
| Coupon | 5% semiannual |
| Day counter | Actual/365 Fixed |
| Engine | `RiskyBondEngine` (survival-weighted discounting) |
| JIT eligible | **Yes** — branching on dates only, not on AReal inputs |

### Interest-rate curve (OIS-bootstrapped)

| Tenor | Rate |
|---|---:|
| 1M | 3.74% |
| 3M | 3.68% |
| 6M | 3.61% |
| 1Y | 3.52% |
| 2Y | 3.42% |
| 3Y | 3.46% |
| 5Y | 3.57% |
| 10Y | 4.02% |
| 30Y | 4.67% |

Bootstrap: `OISRateHelper` → `PiecewiseLogLinearDiscount` (SOFR index, US Treasury daily par yield curve (2026-02-26))

### Credit curve (CDS-bootstrapped)

| Tenor | CDS Spread |
|---|---:|
| CDS 1Y | 50 bp |
| CDS 2Y | 75 bp |
| CDS 3Y | 100 bp |
| CDS 5Y | 125 bp |

Recovery: 40%  
Bootstrap: `SpreadCdsHelper` → `PiecewiseFlatHazardRate`

---

## Pricing formula

The `RiskyBondEngine` computes the risky NPV as:

$$
\text{NPV} = \sum_i CF_i \cdot P(0, T_i) \cdot Q(T_i)
+ R \sum_i N(T_i^{\text{mid}}) \cdot P(0, T_i^{\text{mid}})
\cdot [Q(T_{i-1}) - Q(T_i)]
$$

where $P(0, t)$ is the OIS discount factor, $Q(t)$ is the CDS-implied
survival probability, $R$ is the recovery rate, and $N(t)$ is the notional.

---

## Greeks validation (AAD vs FD)

NPV = 100.6119512006

### Interest-rate sensitivities (OIS)

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| OIS 1M | -1.117421 | -1.117426 | 5.33e-06 |
| OIS 3M | -0.047320 | -0.047321 | 4.95e-07 |
| OIS 6M | -1.052444 | -1.052495 | 5.05e-05 |
| OIS 1Y | -0.225570 | -0.225549 | 2.01e-05 |
| OIS 2Y | -4.177902 | -4.177931 | 2.92e-05 |
| OIS 3Y | -12.906363 | -12.906128 | 2.34e-04 |
| OIS 5Y | -417.837543 | -417.897127 | 5.96e-02 |
| OIS 10Y | 0.023434 | 0.023429 | 4.88e-06 |
| OIS 30Y | -0.000001 | 0.000000 | 7.67e-07 |

### Credit sensitivities (CDS spreads + recovery)

| Input | FD (1 bp) | AAD | \|Δ\| |
|---|---:|---:|---:|
| CDS 1Y | -0.442502 | -0.442539 | 3.71e-05 |
| CDS 2Y | -0.894657 | -0.894709 | 5.20e-05 |
| CDS 3Y | -8.148631 | -8.146991 | 1.64e-03 |
| CDS 5Y | -439.253599 | -439.319624 | 6.60e-02 |
| Recovery | -0.200430 | -0.200397 | 3.29e-05 |

---

## Timing results

N = 14 market inputs (9 OIS + 4 CDS + 1 recovery), 30 repetitions, BPS = 0.0001

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Plain pricing (1 NPV) | 0.3936 ±0.1133 | 0.3635 ±0.1069 | 1.08× |
| Bump-and-reprice FD (N+1 NPVs) | 7.4168 ±0.7441 | 7.4262 ±0.3973 | 1.00× |
| **AAD backward pass** | 0.0637 ±0.0306 | 0.0659 ±0.0065 | 0.97× |
| *FD ÷ AAD* | *116.4×* | *112.7×* | — |

---

## Analysis

This benchmark demonstrates AAD applied to a **realistic credit-risky bond**
priced against production-grade bootstrapped curves:

- **Interest-rate curve**: 9 SOFR OIS par rates bootstrapped via
  `OISRateHelper` + `PiecewiseLogLinearDiscount` (live-scraped from
  US Treasury or hardcoded fallback)
- **Credit curve**: 4 CDS spread quotes bootstrapped via
  `SpreadCdsHelper` + `PiecewiseFlatHazardRate`
- **Recovery rate**: scalar input to `RiskyBondEngine`

With 14 inputs, FD requires 14+1 = 15 full pricings (each involving
curve re-bootstrap), while AAD needs only 1 backward sweep on the pre-recorded
tape.  The AAD advantage is amplified by the bootstrap cost and grows with N.

All branching in `RiskyBondEngine::calculate()` is on dates (not AReal),
making the tape structure fixed for a given bond schedule → **JIT eligible**.

---

## How to reproduce

```bash
# Build both variants (first time only)
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

# Run with live Treasury rates
python benchmarks/risky_bond_benchmarks.py

# Run with hardcoded rates (offline)
python benchmarks/risky_bond_benchmarks.py --offline
python benchmarks/risky_bond_benchmarks.py --repeats 50
```

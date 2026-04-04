# QuantLib-Risks-Py — Hazard-Rate Sensitivity & Credit Jacobian Benchmark

**Date:** 2026-02-27 13:23  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**Risk-free rate:** 3.5% (flat)  
**Recovery rate:** 40%  

---

## Instrument

- **5-year CDS** (Protection Buyer, 100 bp running coupon, $10M notional)
- Hazard curve: `PiecewiseFlatHazardRate` (bootstrap) or `HazardRateCurve` (direct)
- Risk-free: `FlatForward` at 3.5% (held fixed)

### Market data

| Tenor | CDS spread (bp) | Hazard rate (bp) |
|-------|----------------:|-----------------:|
| 1Y | 50.0 | 82.77 |
| 2Y | 75.0 | 172.63 |
| 3Y | 100.0 | 260.52 |
| 5Y | 125.0 | 281.87 |

---

## Approach 1 — Direct HazardRateCurve

Bootstrap once (plain float), extract hazard rates at pillar dates,
then build an interpolated `HazardRateCurve` (BackwardFlat) and
differentiate through it.  **No solver on the AD tape.**

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| FD (N+1 pricings per scenario) | 17.0 ±0.2 | 22.7 ±4.3 | 0.75× |
| **AAD replay** (backward sweep) | 0.2 ±0.0 | 0.3 ±0.4 | 0.84× |
| AAD re-record (forward + backward) | 5.4 ±0.7 | 5.2 ±2.3 | 1.04× |

---

## Approach 2 — Jacobian conversion

Compute ∂NPV/∂s via AAD replay on the bootstrap tape (solver on tape),
then compute J = ∂h/∂s via 4 backward sweeps through the bootstrap.
Solve for ∂NPV/∂h:
$$J^T \cdot \nabla_h \text{NPV} = \nabla_s \text{NPV}$$

| Step | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Spread AAD replay (100×) | 2.2 ±0.1 | 2.1 ±0.5 | 1.07× |
| Jacobian AAD (4 sweeps) | 0.7 ±0.0 | 0.8 ±0.1 | 0.94× |
| Jacobian FD  (4 bumps) | 1.3 ±0.1 | 1.7 ±0.5 | 0.78× |
| Matrix solve (100×) | 0.8 ±0.0 | 0.9 ±0.5 | 0.88× |
| **Total Approach 2** | 4.6 ±0.8 | 3.7 ±0.6 | 1.23× |

> **Jacobian: FD ÷ AAD = 1.8× (non-JIT), 2.2× (JIT)**

---

## Cross-approach comparison

All methods compute **∂NPV/∂(hazard rate)** for 100 scenarios.

| Method | Non-JIT (ms) | vs FD | JIT (ms) | vs FD |
|---|---:|---:|---:|---:|
| Approach 1: FD | 17.0 | 1.0× | 22.7 | 1.0× |
| Approach 1: AAD re-record | 5.4 | 3.1× | 5.2 | 4.3× |
| Approach 2: total | 4.6 | 4× | 3.7 | 6× |
| **Approach 1: AAD replay** | **0.2** | **81×** | **0.3** | **90×** |

> **Approach 1 AAD replay** (0.2 ms) is **22× faster** than Approach 2 total (4.6 ms).
>
> For hazard-rate sensitivities, **bypassing the solver** (using a direct
> `HazardRateCurve`) is the clear winner.  The tiny tape means replay is
> near-instant, far cheaper than the 4-sweep Jacobian + matrix solve in
> Approach 2.  This mirrors the zero-rate benchmark for interest rates.

---

## Hazard-rate sensitivity validation

| Pillar | FD | AAD (HazardRateCurve) | Jacobian solve |
|--------|---:|---:|---:|
| 2025-12-22 | 6300670.14 | 6301013.91 | 6301013.91 |
| 2026-12-21 | 5396216.07 | 5396482.92 | 5396482.92 |
| 2027-12-20 | 5118094.60 | 5118347.71 | 5118347.71 |
| 2029-12-20 | 9501729.15 | 9502664.34 | 9502664.34 |

> **FD** and **AAD** agree closely.  The **Jacobian solve** column also
> agrees because both `PiecewiseFlatHazardRate` and `HazardRateCurve`
> use BackwardFlat interpolation on hazard rates.

---

## Round-trip:  Jᵀ × ∂NPV/∂h = ∂NPV/∂s

| Tenor | ∂NPV/∂s (AAD) | Jᵀ × ∂NPV/∂h |
|-------|---:|---:|
| 1Y | -37805.21 | -37805.21 |
| 2Y | -70354.80 | -70354.80 |
| 3Y | -160755.56 | -160755.56 |
| 5Y | 44392831.43 | 44392831.43 |

---

## Bootstrap Jacobian  J = ∂h/∂s

J is the 4×4 matrix of partial derivatives ∂h_j/∂s_i.
It is **lower-triangular** because the CDS bootstrap is sequential:
the hazard rate at each pillar depends only on shorter-tenor CDS spreads.

### AAD Jacobian

| | s\_1Y | s\_2Y | s\_3Y | s\_5Y |
|---|---:|---:|---:|---:|
| h(2025-12-22) | 1.655384 | 0.000000 | 0.000000 | 0.000000 |
| h(2026-12-21) | -1.927029 | 3.594861 | 0.000000 | 0.000000 |
| h(2027-12-20) | -0.003908 | -3.786058 | 5.473905 | 0.000000 |
| h(2029-12-20) | -0.005180 | -0.009640 | -2.965285 | 4.671619 |

### FD Jacobian

| | s\_1Y | s\_2Y | s\_3Y | s\_5Y |
|---|---:|---:|---:|---:|
| h(2025-12-22) | 1.655384 | 0.000000 | 0.000000 | 0.000000 |
| h(2026-12-21) | -1.927023 | 3.595204 | -0.000000 | -0.000000 |
| h(2027-12-20) | -0.003907 | -3.786394 | 5.474937 | 0.000000 |
| h(2029-12-20) | -0.005180 | -0.009639 | -2.965796 | 4.673000 |

---

## Notes

- For **hazard-rate sensitivities**, Approach 1 (direct `HazardRateCurve`
  AAD) is the clear winner — just like the zero-rate benchmark for
  interest rates.
- The **credit bootstrap Jacobian** J = ∂h/∂s is lower-triangular and
  analogous to J = ∂z/∂r for interest rates.
- Both AAD and FD Jacobians agree closely, validating the AAD tape
  through the Brent solver in `PiecewiseFlatHazardRate`.
- The **round-trip** Jᵀ × ∂NPV/∂h = ∂NPV/∂s confirms that the
  Jacobian conversion is mathematically exact.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/hazard_rate_jacobian_benchmarks.py
python benchmarks/hazard_rate_jacobian_benchmarks.py --repeats 10
```

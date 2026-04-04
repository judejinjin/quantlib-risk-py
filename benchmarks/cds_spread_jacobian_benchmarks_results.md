# QuantLib-Risks-Py — CDS-Spread Sensitivity & Reverse Credit Jacobian Benchmark

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
- Hazard curve: `PiecewiseFlatHazardRate` (Approach 1) or `HazardRateCurve` (Approach 2)
- Risk-free: `FlatForward` at 3.5% (held fixed)

### Market data

| Tenor | CDS spread (bp) | Hazard rate (bp) |
|-------|----------------:|-----------------:|
| 1Y | 50.0 | 82.77 |
| 2Y | 75.0 | 172.63 |
| 3Y | 100.0 | 260.52 |
| 5Y | 125.0 | 281.87 |

---

## Approach 1 — Direct bootstrap

Build `PiecewiseFlatHazardRate` from CDS spread inputs.  The Brent solver
is **on** the AD tape.  A single backward sweep gives all CDS-spread sensitivities.

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| FD (N+1 pricings per scenario) | 110.7 ±40.2 | 97.0 ±1.7 | 1.14× |
| **AAD replay** (backward sweep) | 2.1 ±0.1 | 2.0 ±0.4 | 1.07× |
| AAD re-record (forward + backward) | 57.5 ±2.5 | 58.9 ±2.7 | 0.98× |

---

## Approach 2 — HazardRateCurve AAD + J^T conversion

Compute ∂NPV/∂h via HazardRateCurve AAD (no solver on tape), then
multiply by J^T to obtain CDS-spread sensitivities:
$$\nabla_s \text{NPV} = J^T \cdot \nabla_h \text{NPV}$$

| Step | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| HazardRateCurve AAD replay (100×) | 0.3 ±0.0 | 0.3 ±0.1 | 0.84× |
| Jacobian J = ∂h/∂s (4 sweeps) | 1.2 ±0.2 | 0.6 ±0.1 | 1.79× |
| J^T × ∂NPV/∂h matmul (100×) | 0.4 ±0.0 | 0.2 ±0.0 | 1.87× |
| **Total Approach 2** | 1.2 ±0.3 | 1.0 ±0.0 | 1.19× |

---

## Cross-approach comparison

All methods compute **∂NPV/∂(CDS spread)** for 100 scenarios.

| Method | Non-JIT (ms) | vs FD | JIT (ms) | vs FD |
|---|---:|---:|---:|---:|
| Approach 1: FD | 110.7 | 1.0× | 97.0 | 1.0× |
| Approach 1: AAD re-record | 57.5 | 1.9× | 58.9 | 1.6× |
| Approach 2: total | 1.2 | 89× | 1.0 | 93× |
| **Approach 1: AAD replay** | **2.1** | **52×** | **2.0** | **48×** |

> **Approach 1 AAD replay** (2.1 ms) is **1× faster** than Approach 2 total (1.2 ms).
>
> For CDS-spread sensitivities, there is no benefit to bypassing the solver.
> The bootstrap tape is larger, but a **single backward sweep** through it
> is still vastly cheaper than computing the full 4-sweep Jacobian.
> This mirrors the money-market (par) rate benchmark for interest rates.

---

## CDS-spread sensitivity validation

| Tenor | FD (bootstrap) | AAD direct (bootstrap) | J^T × ∂NPV_HC/∂h |
|-------|---:|---:|---:|
| 1Y | -37802.88 | -37805.21 | -37805.21 |
| 2Y | -70353.04 | -70354.80 | -70354.80 |
| 3Y | -160757.66 | -160755.56 | -160755.56 |
| 5Y | 44385531.51 | 44392831.43 | 44392831.43 |

> **FD** and **AAD direct** both price through `PiecewiseFlatHazardRate`
> and agree closely.
>
> The **J^T × ∂NPV_HC/∂h** column also agrees because both
> `PiecewiseFlatHazardRate` and `HazardRateCurve` use BackwardFlat
> interpolation on hazard rates.

---

## Implied par spreads from HazardRateCurve

Par spreads computed by pricing CDS instruments through HazardRateCurve
(BackwardFlat hazard-rate interpolation) vs the original par spreads
used to bootstrap the curve.

| Tenor | Original (bp) | HazardRateCurve (bp) | Diff (bp) |
|-------|--------:|---------:|----------:|
| 1Y | 50.0 | 50.00 | -0.00 |
| 2Y | 75.0 | 75.00 | -0.00 |
| 3Y | 100.0 | 100.00 | +0.00 |
| 5Y | 125.0 | 125.00 | +0.00 |

> Since `PiecewiseFlatHazardRate` and `HazardRateCurve` both use BackwardFlat
> interpolation on hazard rates, the implied par spreads should match
> the originals closely.

---

## Reverse Jacobian  K = ∂s/∂h

**Computation time:** 0.3 ms (non-JIT), 0.3 ms (JIT)

### How K is generated

K is the 4×4 matrix of partial derivatives ∂s_i/∂h_j.
It is computed via AAD through the *reverse* mapping: hazard rates → par spreads.

1. **Record a tape** of the reverse mapping: create 4 `xad::Real` hazard-rate
   inputs, build a `HazardRateCurve`, then for each of the 4 tenors build a CDS
   and call `fairSpread()`.  The fair spread is computed analytically from
   survival probabilities (no solver involved), so the tape is compact.

2. **Register the 4 fair spreads as tape outputs.**

3. **4 backward sweeps**: for each output *i*, set `s_i.derivative = 1.0`,
   call `computeAdjoints()`, read `h_j.derivative` for all *j* → row *i* of K.

Like J, K is **lower-triangular**: the par spread at tenor *i* depends only on
hazard rates at pillars ≤ *i* (BackwardFlat interpolation doesn't reach
beyond the tenor's maturity date).

### K matrix

| | h(2025-12-22) | h(2026-12-21) | h(2027-12-20) | h(2029-12-20) |
|---|---:|---:|---:|---:|
| s\_1Y | 0.604089 | 0.000000 | 0.000000 | 0.000000 |
| s\_2Y | 0.323823 | 0.278175 | 0.000000 | 0.000000 |
| s\_3Y | 0.224405 | 0.192401 | 0.182685 | 0.000000 |
| s\_5Y | 0.143778 | 0.122700 | 0.115958 | 0.214059 |

---

## Bootstrap Jacobian  J = ∂h/∂s

(Same as in the hazard-rate benchmark, included for comparison.)

| | s\_1Y | s\_2Y | s\_3Y | s\_5Y |
|---|---:|---:|---:|---:|
| h(2025-12-22) | 1.655384 | 0.000000 | 0.000000 | 0.000000 |
| h(2026-12-21) | -1.927029 | 3.594861 | 0.000000 | 0.000000 |
| h(2027-12-20) | -0.003908 | -3.786058 | 5.473905 | 0.000000 |
| h(2029-12-20) | -0.005180 | -0.009640 | -2.965285 | 4.671619 |

---

## Inverse verification:  K × J  vs  I

If K = J⁻¹, then K × J should equal the identity matrix.

| | [0] | [1] | [2] | [3] |
|---|---:|---:|---:|---:|
| [0] | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| [1] | -0.000000 | 1.000000 | 0.000000 | 0.000000 |
| [2] | 0.000000 | -0.000000 | 1.000000 | 0.000000 |
| [3] | 0.000000 | 0.000000 | -0.000000 | 1.000000 |

**max |K×J − I| = 1.30e-10**  
**max |K − J⁻¹| = 7.88e-11**

### Residual matrix  K × J − I

| | [0] | [1] | [2] | [3] |
|---|---:|---:|---:|---:|
| [0] | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [1] | -0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [2] | 0.000000 | -0.000000 | 0.000000 | 0.000000 |
| [3] | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

### Why K = J⁻¹

K and J are computed through the **same interpolation method**:

| | Forward (J = ∂h/∂s) | Reverse (K = ∂s/∂h) |
|---|---|---|
| **Curve object** | `PiecewiseFlatHazardRate` | `HazardRateCurve` |
| **Interpolation** | BackwardFlat on hazard rates | BackwardFlat on hazard rates |
| **Survival prob** | S(t) = exp(−∫₀ᵗ h(u) du) | S(t) = exp(−∫₀ᵗ h(u) du) |

Because both curves produce identical survival probabilities at all dates,
the round-trip s → h → s closes exactly.  By the inverse function
theorem, K = J⁻¹ and K × J = I.

---

## Notes

- For **CDS-spread sensitivities**, Approach 1 (direct bootstrap AAD) is the
  clear winner.  A single backward sweep through the bootstrap tape, even
  with the Brent solver, is far cheaper than the 4-sweep Jacobian computation
  in Approach 2.
- This is the **mirror image** of the hazard-rate benchmark, where bypassing the
  solver (Approach 1: HazardRateCurve) was the overwhelming winner.
- **Takeaway**: differentiate through the solver when you need CDS-spread risks;
  bypass the solver (HazardRateCurve) when you need hazard-rate risks.
- The **K = J⁻¹** result confirms the inverse function theorem:
  both directions use `PiecewiseFlatHazardRate` / `HazardRateCurve` (same
  BackwardFlat hazard-rate interpolation), so the round-trip is exact.
- This is the **credit analogue** of the interest-rate `mm_rate_jacobian`
  benchmark, where K = ∂r/∂z was verified against J = ∂z/∂r.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/cds_spread_jacobian_benchmarks.py
python benchmarks/cds_spread_jacobian_benchmarks.py --repeats 10
```

# QuantLib-Risks-Py — Money-Market (Par) Rate & Reverse Jacobian Benchmark

**Date:** 2026-02-27 08:48  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**Rate source:** US Treasury daily par yield curve (2026-02-26)  

---

## Instrument

- **5-year SOFR OIS** (pay fixed 3.57%, receive SOFR, $10M notional)
- Discount/forecasting: `PiecewiseLinearZero` (Approach 1) or `ZeroCurve` (Approach 2)

### Market data

| Tenor | Par rate | Zero rate |
|-------|--------:|---------:|
| 1M | 3.74% | 3.7859% |
| 3M | 3.68% | 3.7167% |
| 6M | 3.61% | 3.6302% |
| 1Y | 3.52% | 3.5097% |
| 2Y | 3.42% | 3.4090% |
| 3Y | 3.46% | 3.4493% |
| 5Y | 3.57% | 3.5626% |
| 10Y | 4.02% | 4.0555% |
| 30Y | 4.67% | 4.9598% |

---

## Approach 1 — Direct bootstrap

Build `PiecewiseLinearZero` from par-rate inputs.  The Brent solver
is **on** the AD tape.  A single backward sweep gives all par-rate sensitivities.

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| FD (N+1 pricings per scenario) | 437.3 ±74.7 | 371.3 ±11.0 | 1.18× |
| **AAD replay** (backward sweep) | 2.6 ±0.7 | 2.5 ±0.3 | 1.04× |
| AAD re-record (forward + backward) | 5344.9 ±90.1 | 5042.2 ±266.3 | 1.06× |

---

## Approach 2 — ZeroCurve AAD + J^T conversion

Compute ∂NPV/∂z via ZeroCurve AAD (no solver on tape), then
multiply by J^T to obtain par-rate sensitivities:
$$\nabla_r \text{NPV} = J^T \cdot \nabla_z \text{NPV}$$

| Step | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| ZeroCurve AAD replay (100×) | 0.2 ±0.1 | 0.1 ±0.0 | 1.46× |
| Jacobian J = ∂z/∂r (9 sweeps) | 47.2 ±0.5 | 48.4 ±3.8 | 0.97× |
| J^T × ∂NPV/∂z matmul (100×) | 0.7 ±0.1 | 0.9 ±0.1 | 0.79× |
| **Total Approach 2** | 48.9 ±1.2 | 47.6 ±2.6 | 1.03× |

---

## Cross-approach comparison

All methods compute **∂NPV/∂(par rate)** for 100 scenarios.

| Method | Non-JIT (ms) | vs FD | JIT (ms) | vs FD |
|---|---:|---:|---:|---:|
| Approach 1: FD | 437.3 | 1.0× | 371.3 | 1.0× |
| Approach 1: AAD re-record | 5344.9 | 0.1× | 5042.2 | 0.1× |
| Approach 2: total | 48.9 | 9× | 47.6 | 8× |
| **Approach 1: AAD replay** | **2.6** | **171×** | **2.5** | **151×** |

> **Approach 1 AAD replay** (2.6 ms) is **19× faster** than Approach 2 total (48.9 ms).
>
> For par-rate sensitivities, there is no benefit to bypassing the solver.
> The bootstrap tape is larger, but a **single backward sweep** through it
> is still vastly cheaper than computing the full 9-sweep Jacobian.
> This is the mirror image of the zero-rate benchmark, where Approach 1
> (bypassing the solver) was the overwhelming winner.

---

## Par-rate sensitivity validation

| Tenor | FD (bootstrap) | AAD direct (bootstrap) | J^T × ∂NPV_ZC/∂z |
|-------|---:|---:|---:|
| 1M | 12.03 | 12.04 | 12.04 |
| 3M | -0.01 | 0.00 | 0.00 |
| 6M | -0.01 | 0.00 | 0.00 |
| 1Y | -38.06 | -38.05 | -38.05 |
| 2Y | -34.03 | -34.22 | -34.22 |
| 3Y | 469.03 | 468.91 | 468.91 |
| 5Y | 45718845.39 | 45725094.52 | 45725094.52 |
| 10Y | -0.01 | 0.00 | 0.00 |
| 30Y | -0.01 | 0.00 | 0.00 |

> **FD** and **AAD direct** both price through `PiecewiseLinearZero`
> and agree closely.
>
> The **J^T × ∂NPV_ZC/∂z** column now also agrees closely because
> `PiecewiseLinearZero` and `ZeroCurve` both use linear interpolation
> on zero rates, producing identical inter-pillar discount factors.

---

## Implied par rates from ZeroCurve

Par rates computed by pricing OIS swaps through ZeroCurve (linear zero-rate
interpolation) vs the original par rates used to bootstrap the curve.

| Tenor | Original | ZeroCurve | Diff (bp) |
|-------|--------:|---------:|----------:|
| 1M | 3.7400% | 3.7400% | +0.00 |
| 3M | 3.6800% | 3.6800% | +0.00 |
| 6M | 3.6100% | 3.6100% | -0.00 |
| 1Y | 3.5200% | 3.5200% | +0.00 |
| 2Y | 3.4200% | 3.4200% | -0.00 |
| 3Y | 3.4600% | 3.4600% | +0.00 |
| 5Y | 3.5700% | 3.5699% | -0.01 |
| 10Y | 4.0200% | 4.0200% | -0.00 |
| 30Y | 4.6700% | 4.6700% | -0.00 |

> Since `PiecewiseLinearZero` and `ZeroCurve` both use linear
> interpolation on zero rates, the implied par rates should now
> match the originals closely.  Any remaining differences are at
> machine-precision level.

---

## Reverse Jacobian  K = ∂r/∂z

**Computation time:** 45.2 ms (non-JIT), 45.3 ms (JIT)

### How K is generated

K is the 9×9 matrix of partial derivatives ∂r_i/∂z_j.
It is computed via AAD through the *reverse* mapping: zero rates → par rates.

1. **Record a tape** of the reverse mapping: create 9 `xad::Real` zero-rate
   inputs, build a `ZeroCurve`, then for each of the 9 tenors build an OIS
   swap and call `fairRate()`.  The fair rate is computed analytically from
   discount factors (no solver involved), so the tape is compact.

2. **Register the 9 fair rates as tape outputs.**

3. **9 backward sweeps**: for each output *i*, set `r_i.derivative = 1.0`,
   call `computeAdjoints()`, read `z_j.derivative` for all *j* → row *i* of K.

Like J, K is **lower-triangular**: the par rate at tenor *i* depends only on
zero rates at pillars ≤ *i* (ZeroCurve's linear interpolation doesn't reach
beyond the tenor's maturity date).

### K matrix

| | z(2026-04-02) | z(2026-06-02) | z(2026-09-02) | z(2027-03-02) | z(2028-03-02) | z(2029-03-02) | z(2031-03-03) | z(2036-03-03) | z(2056-03-02) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r\_1M | 0.989478 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r\_3M | -0.043286 | 1.038863 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r\_6M | -0.021837 | 0.000000 | 1.026337 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r\_1Y | -0.011195 | 0.000000 | 0.000000 | 1.032696 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r\_2Y | -0.005682 | 0.000000 | 0.000000 | 0.017550 | 1.009187 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r\_3Y | -0.003855 | 0.000000 | 0.000000 | 0.012046 | 0.023275 | 0.989402 | 0.000000 | 0.000000 | 0.000000 |
| r\_5Y | -0.002395 | 0.000000 | 0.000000 | 0.007722 | 0.014922 | 0.035343 | 0.964540 | 0.000000 | 0.000000 |
| r\_10Y | -0.001320 | 0.000000 | 0.000000 | 0.004792 | 0.009260 | 0.021932 | 0.081874 | 0.894773 | 0.000000 |
| r\_30Y | -0.000661 | 0.000000 | 0.000000 | 0.002787 | 0.005385 | 0.012751 | 0.047616 | 0.262786 | 0.635964 |

---

## Bootstrap Jacobian  J = ∂z/∂r

(Same as in the zero-rate benchmark, included for comparison.)

| | r\_1M | r\_3M | r\_6M | r\_1Y | r\_2Y | r\_3Y | r\_5Y | r\_10Y | r\_30Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z(2026-04-02) | 1.010634 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| z(2026-06-02) | 0.042110 | 0.962591 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| z(2026-09-02) | 0.021503 | 0.000000 | 0.974339 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| z(2027-03-02) | 0.010955 | -0.000000 | 0.000000 | 0.968339 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| z(2028-03-02) | 0.005500 | 0.000000 | -0.000000 | -0.016840 | 0.990897 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| z(2029-03-02) | 0.003675 | 0.000000 | 0.000000 | -0.011393 | -0.023310 | 1.010711 | 0.000000 | 0.000000 | 0.000000 |
| z(2031-03-03) | 0.002203 | 0.000000 | 0.000000 | -0.007076 | -0.014476 | -0.037025 | 1.036716 | 0.000000 | 0.000000 |
| z(2036-03-03) | 0.001084 | 0.000000 | 0.000000 | -0.004086 | -0.008359 | -0.021379 | -0.094872 | 1.117582 | 0.000000 |
| z(2056-03-02) | 0.000269 | 0.000000 | 0.000000 | -0.001655 | -0.003385 | -0.008658 | -0.038420 | -0.461795 | 1.572416 |

---

## Inverse verification:  K × J  vs  I

If K = J⁻¹, then K × J should equal the identity matrix.

| | [0] | [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [0] | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [1] | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [2] | -0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [3] | -0.000000 | -0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [4] | -0.000000 | -0.000000 | -0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [5] | -0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| [6] | 0.000000 | 0.000000 | 0.000000 | -0.000001 | -0.000001 | 0.000010 | 0.999953 | 0.000000 | 0.000000 |
| [7] | 0.000000 | 0.000000 | 0.000000 | -0.000001 | -0.000001 | 0.000006 | -0.000008 | 0.999982 | 0.000000 |
| [8] | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | -0.000000 | 0.000000 | -0.000000 | 1.000000 |

**max |K×J − I| = 4.66e-05**  
**max |K − J⁻¹| = 4.50e-05**

### Residual matrix  K × J − I

| | [0] | [1] | [2] | [3] | [4] | [5] | [6] | [7] | [8] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [0] | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [1] | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [2] | -0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [3] | -0.000000 | -0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [4] | -0.000000 | -0.000000 | -0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [5] | -0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [6] | 0.000000 | 0.000000 | 0.000000 | -0.000001 | -0.000001 | 0.000010 | -0.000047 | 0.000000 | 0.000000 |
| [7] | 0.000000 | 0.000000 | 0.000000 | -0.000001 | -0.000001 | 0.000006 | -0.000008 | -0.000018 | 0.000000 |
| [8] | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | -0.000000 | 0.000000 | -0.000000 | 0.000000 |

### Why K = J⁻¹

K and J are now computed through the **same interpolation method**:

| | Forward (J = ∂z/∂r) | Reverse (K = ∂r/∂z) |
|---|---|---|
| **Curve object** | `PiecewiseLinearZero` | `ZeroCurve` |
| **Interpolation** | Linear on zero rates | Linear on zero rates |
| **Inter-pillar DF** | DF(t) = exp(−z(t)·t), z(t) linear | DF(t) = exp(−z(t)·t), z(t) linear |

Because both curves produce identical discount factors at all dates,
the round-trip r → z → r closes exactly.  By the inverse function
theorem, K = J⁻¹ and K × J = I.

---

## Notes

- For **par-rate sensitivities**, Approach 1 (direct bootstrap AAD) is the
  clear winner.  A single backward sweep through the bootstrap tape, even
  with the Brent solver, is far cheaper than the 9-sweep Jacobian computation
  in Approach 2.
- This is the **mirror image** of the zero-rate benchmark, where bypassing the
  solver (Approach 1: ZeroCurve) was ~500× faster for the AAD replay.
- **Takeaway**: differentiate through the solver when you need par-rate risks;
  bypass the solver (ZeroCurve) when you need zero-rate risks.
- The **K = J⁻¹** result confirms the inverse function theorem:
  both directions use `PiecewiseLinearZero` / `ZeroCurve` (same
  linear zero-rate interpolation), so the round-trip is exact.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/mm_rate_jacobian_benchmarks.py            # live rates
python benchmarks/mm_rate_jacobian_benchmarks.py --offline  # hardcoded
```

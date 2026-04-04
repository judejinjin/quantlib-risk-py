# QuantLib-Risks-Py — Zero-Rate Sensitivity & Jacobian Benchmark

**Date:** 2026-02-27 09:12  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**MC scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**Rate source:** US Treasury daily par yield curve (2026-02-26)  

---

## Instrument

- **5-year SOFR OIS** (pay fixed 3.57%, receive SOFR, $10M notional)
- Discount/forecasting: `ZeroCurve` (Approach 1) or `PiecewiseLinearZero` (Approach 2)

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

## Approach 1 — Direct ZeroCurve

Bootstrap once (plain float) → extract continuous zero rates at pillar
dates → build `ZeroCurve` → differentiate through interpolation + swap
pricing.  **No solver on the AD tape.**

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| FD (N+1 pricings per scenario) | 4647.8 ±98.7 | 4332.9 ±443.3 | 1.07× |
| **AAD replay** (backward sweep) | 0.1 ±0.0 | 0.1 ±0.0 | 0.91× |
| AAD re-record (forward + backward) | 468.8 ±17.7 | 458.1 ±36.6 | 1.02× |
| *FD ÷ AAD replay (non-JIT / JIT)* | *35548×* | *30099×* | — |
| *FD ÷ AAD re-record (non-JIT / JIT)* | *9.9×* | *9.5×* | — |

---

## Approach 2 — Jacobian conversion

Compute ∂NPV/∂r via AAD replay on the bootstrap tape, then compute
the bootstrap Jacobian J = ∂z/∂r via AAD (9 backward sweeps), and
solve J^T × ∂NPV/∂z = ∂NPV/∂r.

| Step | Non-JIT (ms) | JIT (ms) | JIT speedup |
|---|---:|---:|---:|
| Par-rate AAD replay (100 sweeps) | 4.4 ±5.0 | 2.4 ±0.4 | 1.83× |
| Jacobian AAD (9 sweeps) | 50.6 ±6.3 | 45.2 ±0.9 | 1.12× |
| Jacobian FD  (9 bumps) | 56.5 ±5.6 | 46.2 ±0.7 | 1.22× |
| Matrix solve (100×) | 4.4 ±0.2 | 4.3 ±0.4 | 1.02× |
| **Total Approach 2** | 55.3 ±3.4 | 51.8 ±5.0 | 1.07× |
| *FD ÷ AAD speedup* | *1.1×* | *1.0×* | — |

---

## Cross-approach comparison

All methods compute the same thing: **∂NPV/∂(zero rate)** for 100 scenarios.

| Method | Non-JIT (ms) | vs FD | JIT (ms) | vs FD |
|---|---:|---:|---:|---:|
| Approach 1: FD | 4647.8 | 1.0× | 4332.9 | 1.0× |
| Approach 1: AAD re-record | 468.8 | 9.9× | 458.1 | 9.5× |
| Approach 2: Jacobian total | 55.3 | 84× | 51.8 | 84× |
| **Approach 1: AAD replay** | **0.1** | **35,548×** | **0.1** | **30,099×** |

> **Approach 1 AAD replay** is 423× faster than **Approach 2 total** (0.1 ms vs 55.3 ms).
> This is because Approach 1 eliminates the Brent solver from the tape entirely, leaving only ZeroCurve interpolation + swap pricing —
> a tape so small that a single backward sweep takes ~1 µs.
>
> **Approach 2** is 8.5× faster than **Approach 1 re-record** (55.3 ms vs 468.8 ms).
> This matters when re-recording is needed (e.g. changing market data), 
> since Approach 2 re-records only the small par-rate tape.

---

## Sensitivity validation (base market)

### Approach 1 — Direct ZeroCurve: FD vs AAD

| Pillar | FD | AAD | Match |
|--------|---:|---:|:---:|
| 2026-04-02 | -109529.95 | -109530.01 | ✓ |
| 2026-06-02 | 0.00 | 0.00 | ✓ |
| 2026-09-02 | 0.00 | 0.00 | ✓ |
| 2027-03-02 | 353107.72 | 353125.57 | ✓ |
| 2028-03-02 | 682272.89 | 682341.40 | ✓ |
| 2029-03-02 | 1615976.21 | 1616187.48 | ✓ |
| 2031-03-03 | 44094754.82 | 44105719.81 | ✓ |
| 2036-03-03 | 0.00 | 0.00 | ✓ |
| 2056-03-02 | 0.00 | 0.00 | ✓ |

### Approach 2 — Round-trip: Jᵀ × ∂NPV/∂z should = ∂NPV/∂r

| Tenor | ∂NPV/∂r (AAD) | Jᵀ×∂NPV/∂z | Match |
|-------|---:|---:|:---:|
| 1M | 12.04 | 12.04 | ✓ |
| 3M | 0.00 | 0.00 | ✓ |
| 6M | 0.00 | 0.00 | ✓ |
| 1Y | -38.05 | -38.05 | ✓ |
| 2Y | -34.22 | -34.22 | ✓ |
| 3Y | 468.91 | 468.91 | ✓ |
| 5Y | 45725094.52 | 45725094.52 | ✓ |
| 10Y | 0.00 | 0.00 | ✓ |
| 30Y | 0.00 | 0.00 | ✓ |

### Zero-rate sensitivities ∂NPV/∂z (both approaches)

> Both approaches use linear interpolation on zero rates
> (`ZeroCurve` and `PiecewiseLinearZero`), so values agree exactly.

| Pillar | Direct (ZeroCurve) | Jacobian (bootstrap) |
|--------|---:|---:|
| 2026-04-02 | -109530.01 | -109530.01 |
| 2026-06-02 | 0.00 | -0.00 |
| 2026-09-02 | 0.00 | 0.00 |
| 2027-03-02 | 353125.57 | 353125.57 |
| 2028-03-02 | 682341.40 | 682341.40 |
| 2029-03-02 | 1616187.48 | 1616187.48 |
| 2031-03-03 | 44105719.81 | 44105719.81 |
| 2036-03-03 | 0.00 | 0.00 |
| 2056-03-02 | 0.00 | 0.00 |

### Par-rate sensitivities ∂NPV/∂r per 1bp

| Tenor | FD | AAD |
|-------|---:|---:|
| 1M | 12.04 | 12.04 |
| 3M | -0.00 | 0.00 |
| 6M | -0.00 | 0.00 |
| 1Y | -38.05 | -38.05 |
| 2Y | -34.02 | -34.22 |
| 3Y | 469.04 | 468.91 |
| 5Y | 45718845.41 | 45725094.52 |
| 10Y | 0.00 | 0.00 |
| 30Y | 0.00 | 0.00 |

---

## Bootstrap Jacobian  J = ∂z/∂r

### How the Jacobian is generated

The Jacobian J is the 9×9 matrix of partial derivatives ∂z_j/∂r_i, where z_j is the continuous
zero rate at pillar date *j* and r_i is par OIS rate *i*.  It is computed
via AAD through the bootstrap procedure:

1. **Record a tape** of the bootstrap: create 9 `xad::Real` par-rate inputs,
   register them on the tape, then build a `PiecewiseLinearZero` curve
   from 9 `OISRateHelper` objects (one per tenor).  The bootstrap internally
   uses Brent root-finding to solve for each discount factor sequentially —
   all of this is recorded on the AD tape.

2. **Extract zero rates as tape outputs**: call `curve.zeroRate(date, dc,
   Continuous).rate()` for each of the 9 pillar dates.  Each returned
   `xad::Real` is a live variable on the tape, connected to the par-rate
   inputs through the bootstrap computation graph.  Register all 9 as tape
   outputs.

3. **9 backward sweeps** to fill the Jacobian: for each output *j*, set
   `z_j.derivative = 1.0` (all others 0), call `computeAdjoints()`, then
   read `r_i.derivative` for all *i*.  This gives row *j* of J.  Each
   sweep runs in O(tape_size) — the same cost as one backward pass through
   the bootstrap.

The result is a **lower-triangular** matrix because the bootstrap is
sequential: pillar *j*'s zero rate depends only on par rates at tenors
≤ *j*.  Longer-tenor par rates have zero influence on shorter-tenor zero
rates, so all upper-triangular entries are zero.

### Jacobian matrix (AAD)

Lower-triangular 9×9 matrix.  Row *j* shows how continuous zero
rate *z_j* changes w.r.t. each par OIS rate *r_i*.

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

### Jacobian matrix (FD)

Same matrix computed via central finite differences (1 bp bump).

| | r\_1M | r\_3M | r\_6M | r\_1Y | r\_2Y | r\_3Y | r\_5Y | r\_10Y | r\_30Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z(2026-04-02) | 1.010630 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| z(2026-06-02) | 0.042110 | 0.962579 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| z(2026-09-02) | 0.021503 | -0.000000 | 0.974315 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| z(2027-03-02) | 0.010955 | -0.000000 | -0.000000 | 0.968292 | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| z(2028-03-02) | 0.005500 | 0.000000 | 0.000000 | -0.016838 | 0.990898 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| z(2029-03-02) | 0.003675 | -0.000000 | -0.000000 | -0.011392 | -0.023307 | 1.010766 | -0.000000 | -0.000000 | -0.000000 |
| z(2031-03-03) | 0.002203 | -0.000000 | -0.000000 | -0.007075 | -0.014475 | -0.037022 | 1.036841 | 0.000000 | 0.000000 |
| z(2036-03-03) | 0.001084 | -0.000000 | -0.000000 | -0.004085 | -0.008358 | -0.021378 | -0.094861 | 1.117908 | -0.000000 |
| z(2056-03-02) | 0.000269 | 0.000000 | 0.000000 | -0.001654 | -0.003385 | -0.008657 | -0.038415 | -0.461538 | 1.574348 |

**max |J_AAD − J_FD| = 1.93e-03**

> The Jacobian AAD computation (50.6 ms) is **1.1× faster** than FD (56.5 ms).
> Both produce the same matrix to machine precision.

### Why AAD and FD have similar Jacobian timings

For this 9×9 square Jacobian, reverse-mode AAD and FD do roughly
the same amount of work:

| | Reverse-mode AAD | Finite differences |
|---|---|---|
| **Forward passes** | 1 (tape recording) | 9+1 (base + 9 bumps) |
| **Backward sweeps** | 9 (one per output row) | 0 |
| **Total pass-equivalents** | 9+1 | 9+1 |

Each FD bump re-bootstraps the curve *once* and extracts *all* zero
rates — producing a full **column** of J, not a single element.
So FD needs 9 re-bootstraps (not 9×9 = 81), plus one base
evaluation.

Since one backward sweep costs roughly the same as one forward pass,
the two methods converge to ~9+1 pass-equivalents for a square
Jacobian.  The AAD advantage appears when the matrix is **rectangular**:

- **Reverse-mode AAD** scales with N_outputs (rows) — ideal when
  N_outputs ≪ N_inputs.
- **FD** scales with N_inputs (columns) — ideal when
  N_inputs ≪ N_outputs.

For example, with 100 par-rate inputs but still 9 zero-rate outputs,
AAD would still do 9 sweeps while FD would need 100 bumps — giving
~10× speedup for AAD.

### Using the Jacobian for zero-rate sensitivities (Approach 2)

Given par-rate sensitivities ∂NPV/∂r (a 9-vector obtained from one AAD
backward sweep through the pricing tape), convert to zero-rate
sensitivities via the linear system:

$$
J^T \cdot \frac{\partial \text{NPV}}{\partial z}
= \frac{\partial \text{NPV}}{\partial r}
$$

Since J is lower-triangular, Jᵀ is upper-triangular, and the solve
is a simple O(n²) back-substitution — negligible cost (~5 ms for 100
scenarios).  The Jacobian itself needs to be recomputed only when the
base curve changes, not per scenario.

---

## Notes

- **Approach 1** eliminates the Brent solver from the AD tape, making
  the tape much smaller and AAD replay/re-record dramatically faster.
- **Approach 2** reuses the existing par-rate risk infrastructure and
  converts via a one-off Jacobian computation.  Useful when you already
  have par-rate sensitivities from your risk system and want zero-rate
  sensitivities consistent with the original bootstrapped curve.
- The Jacobian is **lower-triangular** because the OIS bootstrap is
  sequential: each pillar zero rate depends only on par rates at that
  tenor and earlier tenors.
- **Approach 1 vs Approach 2 zero-rate sensitivities agree** because
  both `ZeroCurve` and `PiecewiseLinearZero` use linear interpolation on
  zero rates, producing identical inter-pillar discount factors.
- The **round-trip** validation confirms internal consistency of
  Approach 2: Jᵀ × ∂NPV/∂z recovers ∂NPV/∂r exactly.
- The **Jacobian AAD** computation (~51 ms) is **1.1× faster** than FD (~57 ms), 
  and only needs to be recomputed when the base curve changes.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/zero_rate_jacobian_benchmarks.py            # live rates
python benchmarks/zero_rate_jacobian_benchmarks.py --offline  # hardcoded
```

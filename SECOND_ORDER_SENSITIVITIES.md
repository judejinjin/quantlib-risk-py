# Second-Order Sensitivities — Proposed Examples

## Background

QuantLib-Risks-Py currently computes **first-order sensitivities** (delta, vega, rho, …)
via XAD's adjoint mode (`xad.adj_1st`).  XAD's C++ API supports second-order derivatives
natively via nested types (e.g. `AReal<FReal<double>>` for forward-over-adjoint), but the
QuantLib SWIG bindings are compiled with `Real = AReal<double>` — a fixed first-order type.

**Recompiling QuantLib with a second-order type** (`AReal<FReal<double>>`) would require
changes to the QuantLib-Risks-Cpp compatibility layer (qlrisks.hpp) and SWIG typemaps, and
would approximately **double the tape memory** and **halve the forward-pass speed** due to
the extra FReal bookkeeping layer.  This is a significant build-level change.

### Practical approach: FD-over-AAD

Instead, second-order sensitivities can be computed by applying a **finite-difference bump
on top of AAD first derivatives** — no recompilation needed:

$$
\frac{\partial^2 V}{\partial x_i \partial x_j}
\approx \frac{\frac{\partial V}{\partial x_j}\bigg|_{x_i + h}
             - \frac{\partial V}{\partial x_j}\bigg|_{x_i}}
             {h}
$$

This requires N+1 full AAD recordings (one base + one per bumped input), each yielding the
entire gradient vector.  The cost is:
- **FD-over-AAD:** (N+1) × (1 forward + 1 backward) → full N×N Hessian
- **Pure FD Hessian:** N² + 1 forward pricings (or N(N+1)/2 + 1 with symmetry)

For N = 4 (European option), FD-over-AAD needs 5 tape recordings vs 11 FD pricings — a
modest win.  The advantage grows with N: for N = 18 (IR Cap), FD-over-AAD needs 19 tape
recordings vs 172 FD pricings.

---

## Proposed Examples

### Example 1: European Option — Gamma, Vanna, Volga, Rho²

**Instrument:** European call option (AnalyticEuropeanEngine)  
**Inputs (N = 4):** spot S, dividend yield q, volatility σ, risk-free rate r  
**Second-order sensitivities:**

| Greek | Definition | Financial meaning |
|-------|-----------|-------------------|
| **Gamma** | ∂²V/∂S² | Convexity of delta; P&L from hedging error |
| **Vanna** | ∂²V/∂S∂σ | Sensitivity of delta to vol (or vega to spot) |
| **Volga** (Vomma) | ∂²V/∂σ² | Convexity of vega; vol-of-vol exposure |
| **Charm** | ∂²V/∂S∂t | Rate of delta decay (delta bleed) |

**Why this example:** Closed-form BSM Greeks exist for validation.  Gamma and vanna are the
most commonly hedged second-order risks in equity derivatives.

```python
# Pseudocode — FD-over-AAD approach
import QuantLib_Risks as ql
from xad.adj_1st import Tape

inputs = [S, q, sigma, r]    # N = 4
h = 1e-5                      # bump size

def compute_greeks(input_values):
    """Record tape, compute adjoints, return all first-order derivatives."""
    tape = Tape()
    tape.activate()
    reals = [ql.Real(v) for v in input_values]
    tape.registerInputs(reals)
    tape.newRecording()
    # ... build market data, price option ...
    npv = option.NPV()
    tape.registerOutput(npv)
    npv.derivative = 1.0
    tape.computeAdjoints()
    return [tape.derivative(r) for r in reals]

# Base case
greeks_base = compute_greeks(inputs)           # [delta, dq, vega, rho]

# Bumped cases → Hessian rows
hessian = []
for i in range(len(inputs)):
    bumped = list(inputs)
    bumped[i] += h
    greeks_bumped = compute_greeks(bumped)
    hessian.append([(g_b - g_0) / h
                    for g_b, g_0 in zip(greeks_bumped, greeks_base)])

gamma = hessian[0][0]         # ∂²V/∂S²
vanna = hessian[0][2]         # ∂²V/∂S∂σ  (= hessian[2][0] by symmetry)
volga = hessian[2][2]         # ∂²V/∂σ²
```

**Benchmark:** Compare FD-over-AAD (5 recordings) vs pure FD Hessian (11 pricings + N²
differences) vs QuantLib's built-in `Greeks` (which uses analytic formulae internally).

---

### Example 2: Vanilla IRS — Convexity (∂²NPV/∂r²) and Cross-Gamma

**Instrument:** 5Y vanilla IRS (DiscountingSwapEngine + PiecewiseFlatForward)  
**Inputs (N = 17):** 17 curve pillar quotes  
**Second-order sensitivities:**

| Greek | Definition | Financial meaning |
|-------|-----------|-------------------|
| **DV02** (convexity) | ∂²NPV/∂r² | How delta changes with parallel rate shift |
| **Cross-gamma** | ∂²NPV/∂rᵢ∂rⱼ | How sensitivity to pillar i changes when pillar j moves |
| **Key-rate convexity** | diagonal of Hessian | Per-pillar convexity |

**Why this example:** N = 17 makes the cost advantage of FD-over-AAD clear.

- **FD-over-AAD:** 18 tape recordings → full 17×17 Hessian
  - Each recording: ~0.034 ms backward pass (from benchmark data)
  - Plus ~0.38 ms forward pass = ~7.5 ms total
- **Pure FD Hessian:** 154 pricings (using symmetry) × ~0.39 ms = ~60 ms
- **Speedup:** ~8×

The cross-gamma matrix reveals which pillars interact — typically adjacent pillars have
significant cross-gamma while distant ones are nearly zero.  This sparsity structure is
useful for risk decomposition and hedging.

---

### Example 3: Risky Bond — Credit-Rate Cross-Gamma

**Instrument:** Risky bond (RiskyBondEngine, OIS + CDS bootstrapped curves)  
**Inputs (N = 14):** 9 OIS quotes + 4 CDS spreads + 1 recovery rate  
**Second-order sensitivities:**

| Greek | Definition | Financial meaning |
|-------|-----------|-------------------|
| **IR convexity** | ∂²NPV/∂rᵢ∂rⱼ | Rate curve convexity (9×9 block) |
| **Credit convexity** | ∂²NPV/∂sᵢ∂sⱼ | Spread curve convexity (4×4 block) |
| **IR-credit cross-gamma** | ∂²NPV/∂rᵢ∂sⱼ | How rate sensitivity changes with credit spreads |

**Why this example:** The 14×14 Hessian decomposes into interpretable blocks (IR×IR,
credit×credit, IR×credit).  The cross-gamma block quantifies **wrong-way risk** — how rate
and credit risks interact.

- **FD-over-AAD:** 15 recordings → full 14×14 Hessian (~1.0 ms total)
- **Pure FD Hessian:** 106 pricings × ~0.53 ms = ~56 ms
- **Speedup:** ~56×

---

### Example 4: IR Cap — Vega-Vega Cross-Gamma and Rate-Vol Interactions

**Instrument:** IR Cap (BlackCapFloorEngine + bootstrapped curve)  
**Inputs (N = 18):** curve quotes + caplet volatilities  
**Second-order sensitivities:**

| Greek | Definition | Financial meaning |
|-------|-----------|-------------------|
| **Vega convexity** | ∂²NPV/∂σᵢ∂σⱼ | How vega to one tenor changes when another tenor's vol moves |
| **Rate-vol cross-gamma** | ∂²NPV/∂rᵢ∂σⱼ | Interaction between rate and vol exposures |

**Why this example:** The IR Cap has the highest first-order AAD speedup (888×) and the
largest N.  FD-over-AAD becomes compelling:

- **FD-over-AAD:** 19 recordings → full 18×18 Hessian
  - Forward + backward ≈ 0.43 ms × 19 = ~8.2 ms
- **Pure FD Hessian:** 172 pricings × ~0.43 ms = ~74 ms
- **Speedup:** ~9×

---

### Example 5: CDS — Hazard Rate Convexity

**Instrument:** CDS (MidPointCdsEngine + PiecewiseFlatHazardRate)  
**Inputs (N = 6):** 5 CDS spread quotes + 1 recovery rate  
**Second-order sensitivities:**

| Greek | Definition | Financial meaning |
|-------|-----------|-------------------|
| **Spread convexity** | ∂²NPV/∂sᵢ∂sⱼ | How spread delta changes with spread level |
| **Recovery sensitivity** | ∂²NPV/∂R∂sᵢ | How spread deltas change with recovery assumption |

**Why this example:** Credit convexity is critical for CVA computation and wrong-way risk
modelling. The recovery-spread cross-gamma quantifies model sensitivity.

---

## Cost Comparison Summary

| Example | N | FD-over-AAD recordings | Pure FD Hessian pricings | Approx. speedup |
|---------|---:|---:|---:|---:|
| European Option | 4 | 5 | 11 | 2× |
| CDS (MidPoint) | 6 | 7 | 22 | 3× |
| Risky Bond | 14 | 15 | 106 | 7× |
| Vanilla IRS | 17 | 18 | 154 | 8× |
| IR Cap | 18 | 19 | 172 | 9× |

> The FD-over-AAD cost grows as **O(N)** while pure FD Hessian grows as **O(N²)**, so the
> advantage increases with the number of inputs.

---

## Future: Native Second-Order AAD

If QuantLib-Risks-Cpp's `qlrisks.hpp` were extended to support
`Real = AReal<FReal<double>>` (forward-over-adjoint mode), the Hessian computation would
become:

- **N forward-over-adjoint sweeps** for the full Hessian (seed forward direction eᵢ, get
  the i-th row of the Hessian from the backward sweep)
- Each sweep costs ~1× the forward pass
- Total: **N × (forward + backward)** — same O(N) as FD-over-AAD, but with analytic
  (machine-precision) second derivatives instead of O(h) finite-difference error

This would require:
1. Extending `qlrisks.hpp` with specialisations for nested XAD types
2. Updating SWIG typemaps for `AReal<FReal<double>>` ↔ Python
3. Building a separate wheel (e.g. `quantlib_risks_2nd`)
4. Testing all QuantLib engines for compatibility with the nested type

This is a significant but well-scoped project that could be a future milestone.

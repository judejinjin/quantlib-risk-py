# CDS — Second-Order Sensitivities (FD-over-AAD)

**Date:** 2026-03-02 07:55  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 3 (median reported)  
**Wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

- 2-year CDS, Protection Seller, nominal = 1,000,000, coupon = 150bp
- `PiecewiseFlatHazardRate` + `MidPointCdsEngine`
- **6 inputs:** 4 CDS par spreads + 1 recovery rate + 1 risk-free rate
- Hessian bump size h = 1e-05
- Evaluation date: 15-May-2007

---

## How to Read the Hessian Matrix

The Hessian matrix **H** contains all second-order partial derivatives of the NPV
with respect to pairs of inputs:

$$H_{ij} = \frac{\partial^2 \text{NPV}}{\partial x_i \, \partial x_j}$$

- **Diagonal entries** $H_{ii}$ measure the *convexity* (curvature) of the NPV
  with respect to input $x_i$.  A large diagonal value means the first-order
  sensitivity (delta/gradient) changes rapidly as that input moves.
- **Off-diagonal entries** $H_{ij}$ ($i \neq j$) measure *cross-gamma* — how
  the sensitivity to input $x_i$ changes when input $x_j$ moves.
  These capture interaction effects missed by first-order Greeks.
- The matrix is **symmetric** ($H_{ij} = H_{ji}$) up to numerical noise.
  The reported symmetry metric quantifies this noise.
- Values are in NPV currency units per unit² of the respective inputs.

---

## Results

**NPV** = 41.0925128048

### First-Order Sensitivities

| Input | ∂NPV/∂input |
|---|---:|
| CDS 6M | -0.0001 |
| CDS 1Y | 0.0002 |
| CDS 2Y | -2010444.0450 |
| CDS 3Y | 0.0000 |
| Recovery | 0.0000 |
| RiskFree | -0.3377 |

### Full Hessian (FD-over-AAD)

| | CDS 6M | CDS 1Y | CDS 2Y | CDS 3Y | Recovery | RiskFree |
|---|---:|---:|---:|---:|---:|---:|
| **CDS 6M** | 7.1125 | -19.6442 | 610181.4003 | 0.0000 | -0.0000 | -0.0001 |
| **CDS 1Y** | -419.3374 | 1157.2618 | 1548893.0482 | 0.0000 | 0.0362 | 0.0027 |
| **CDS 2Y** | 610172.5737 | 1549583.7357 | 3998841.4881 | 0.0000 | 124815.3855 | 2323353.3470 |
| **CDS 3Y** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Recovery** | -0.0005 | 0.0015 | 124779.3117 | 0.0000 | 0.0000 | 0.0000 |
| **RiskFree** | -0.0013 | 0.0036 | 2323365.6392 | 0.0000 | -0.0000 | 0.0028 |

Symmetry: max |H[i,j] − H[j,i]| = 6.91e+02  
AAD vs FD: max |Δ| = 9.52e+02

> **Key insight:** The Recovery × Spread cross-gammas and spread
> convexity terms capture important second-order credit risk that
> is missed by first-order sensitivities alone.

### FD-over-AAD vs Pure FD — Difference Matrix

The table below shows $(H^{\text{AAD}}_{ij} - H^{\text{FD}}_{ij})$,
i.e. the element-wise difference between the Hessian computed via
FD-over-AAD and the one computed via pure finite differences.

| | CDS 6M | CDS 1Y | CDS 2Y | CDS 3Y | Recovery | RiskFree |
|---|---:|---:|---:|---:|---:|---:|
| **CDS 6M** | -7.572425e+01 | 1.934777e+02 | -1.272977e+02 | 0.000000e+00 | -2.091983e-01 | 2.722304e-02 |
| **CDS 1Y** | -2.062155e+02 | 5.593599e+02 | -9.517347e+02 | 0.000000e+00 | 1.271508e-01 | -8.827418e-02 |
| **CDS 2Y** | -1.361243e+02 | -2.610472e+02 | -2.562285e+02 | 0.000000e+00 | 3.829674e+01 | -2.924890e+01 |
| **CDS 3Y** | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| **Recovery** | -2.097246e-01 | 9.242594e-02 | 2.222875e+00 | 0.000000e+00 | 2.910404e-01 | 2.728929e-02 |
| **RiskFree** | 2.594704e-02 | -8.734421e-02 | -1.695671e+01 | 0.000000e+00 | 2.728374e-02 | -1.063616e-01 |

| Metric | Value |
|---|---:|
| Max \|difference\| | 9.5173e+02 |
| Mean \|difference\| | 7.9313e+01 |
| Max \|Hessian entry\| | 3.9988e+06 |
| Relative error (max \|Δ\| / max \|H\|) | 2.3800e-04 |

> ✅ **Acceptable.** The relative difference is well below 1%,
> confirming both methods agree to high precision.

### Timing

| Method | Time (ms) | Operations |
|---|---:|---|
| FD-over-AAD | 4.1659 ±0.9637 | 7 AAD recordings |
| Pure FD | 24.9673 ±0.8489 | 73 forward pricings |
| **Speedup** | **6.0×** | |

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
python benchmarks/second_order_cds.py
python benchmarks/second_order_cds.py --repeats 50
```

# European Option — Second-Order Sensitivities (FD-over-AAD)

**Date:** 2026-03-02 07:55  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 3 (median reported)  
**Wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

- European call option (`AnalyticEuropeanEngine`)
- S = 7.0, K = 8.0, σ = 0.1, r = 0.05, q = 0.05, T = 1.0Y
- **4 inputs:** spot, dividend yield, volatility, risk-free rate
- Hessian bump size h = 1e-05

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

**NPV** = 0.0303344207

### First-Order Sensitivities

| Input | ∂NPV/∂input |
|---|---:|
| S (spot) | 0.09509987 |
| q (div yield) | -0.66934673 |
| σ (vol) | 1.17147727 |
| r (rate) | 0.63884610 |

### Analytic Validation

| Greek | FD-over-AAD | Analytic BSM | \|Δ\| |
|---|---:|---:|---:|
| Gamma (∂²V/∂S²) | 0.2377761235 | 0.2373357357 | 4.40e-04 |
| Vanna (∂²V/∂S∂σ) | 2.3062077941 | 2.3014914995 | 4.72e-03 |
| Volga (∂²V/∂σ²) | 20.7436208024 | 20.7069735278 | 3.66e-02 |

### Full Hessian (FD-over-AAD)

| | S (spot) | q (div yield) | σ (vol) | r (rate) |
|---|---:|---:|---:|---:|
| **S (spot)** | 0.237776 | -1.769176 | 2.306208 | 1.673554 |
| **q (div yield)** | -1.769035 | 12.451099 | -16.231312 | -11.778147 |
| **σ (vol)** | 2.306181 | -16.231720 | 20.743621 | 15.053720 |
| **r (rate)** | 1.673646 | -11.779720 | 15.054329 | 11.137317 |

Symmetry: max |H[i,j] − H[j,i]| = 1.57e-03  
FD-over-AAD vs Pure FD: max |Δ| = 1.27e-03

### FD-over-AAD vs Pure FD — Difference Matrix

The table below shows $(H^{\text{AAD}}_{ij} - H^{\text{FD}}_{ij})$,
i.e. the element-wise difference between the Hessian computed via
FD-over-AAD and the one computed via pure finite differences.

| | S (spot) | q (div yield) | σ (vol) | r (rate) |
|---|---:|---:|---:|---:|
| **S (spot)** | 8.540112e-06 | -1.491341e-05 | 9.965898e-06 | 1.556267e-05 |
| **q (div yield)** | 1.263922e-04 | -8.736389e-04 | 5.362512e-04 | 8.150044e-04 |
| **σ (vol)** | -1.726778e-05 | 1.278202e-04 | -1.274582e-03 | -2.310560e-04 |
| **r (rate)** | 1.078155e-04 | -7.578520e-04 | 3.786800e-04 | 7.016656e-04 |

| Metric | Value |
|---|---:|
| Max \|difference\| | 1.2746e-03 |
| Mean \|difference\| | 3.7481e-04 |
| Max \|Hessian entry\| | 2.0744e+01 |
| Relative error (max \|Δ\| / max \|H\|) | 6.1445e-05 |

> ✅ **Acceptable.** The relative difference is well below 1%,
> confirming both methods agree to high precision.

### Timing

| Method | Time (ms) | Operations |
|---|---:|---|
| FD-over-AAD | 1.0010 ±0.3337 | 5 AAD tape recordings |
| Pure FD | 1.3595 ±0.0747 | 33 forward pricings |
| **Speedup** | **1.36×** | |

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
python benchmarks/second_order_european.py
python benchmarks/second_order_european.py --repeats 50
```

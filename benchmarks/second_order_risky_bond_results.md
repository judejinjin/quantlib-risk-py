# Risky Bond — Second-Order Sensitivities (FD-over-AAD)

**Date:** 2026-03-02 07:58  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 3 (median reported)  
**Wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

- 5-year fixed-rate bond, 5% coupon, semiannual, 100 face value
- `PiecewiseLogLinearDiscount` OIS curve + `PiecewiseFlatHazardRate` + `RiskyBondEngine`
- **14 inputs:** 9 OIS rates + 4 CDS spreads + 1 recovery rate
- Hessian bump size h = 1e-05
- Evaluation date: 15-Nov-2024

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

**NPV** = 102.3269249507

### First-Order Sensitivities

| Input | ∂NPV/∂input |
|---|---:|
| OIS 1Y | -2.5507 |
| OIS 2Y | -3.5784 |
| OIS 3Y | -6.3743 |
| OIS 4Y | -13.4241 |
| OIS 5Y | -415.2311 |
| OIS 7Y | 0.0604 |
| OIS 10Y | 0.0000 |
| OIS 15Y | 0.0000 |
| OIS 20Y | 0.0000 |
| CDS 1Y | 1.5333 |
| CDS 3Y | -23.5991 |
| CDS 5Y | -444.3009 |
| CDS 7Y | 0.0000 |
| Recovery | -0.4489 |

### Hessian Block Analysis

| Block | Max \|entry\| | Dominant pair | Value |
|---|---:|---|---:|
| IR × IR | 776.8948 | OIS 5Y × OIS 5Y | 776.8948 |
| CR × CR | 1242.4242 | CDS 5Y × CDS 5Y | 1242.4242 |
| IR × CR | 2884.1547 | OIS 5Y × CDS 5Y | 2884.1547 |
| IR × Rec | 50.6674 | OIS 5Y × Recovery | 50.6674 |
| CR × Rec | 15.4429 | CDS 3Y × Recovery | -15.4429 |
| Rec × Rec | 1.4560 | Recovery × Recovery | -1.4560 |

Symmetry: max |H[i,j] − H[j,i]| = 3.51e-02  
AAD vs FD: max |Δ| = 2.46e-02

### Full Hessian (FD-over-AAD)

<details>
<summary>Click to expand 14×14 Hessian matrix</summary>

| | OIS 1Y | OIS 2Y | OIS 3Y | OIS 4Y | OIS 5Y | OIS 7Y | OIS 10Y | OIS 15Y | OIS 20Y | CDS 1Y | CDS 3Y | CDS 5Y | CDS 7Y | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **OIS 1Y** | 2.7723 | 3.0532 | 2.3475 | 3.9347 | 75.6381 | -0.0078 | 0.0000 | 0.0000 | 0.0000 | -54.4328 | -96.6631 | 84.1901 | 0.0000 | -1.6248 |
| **OIS 2Y** | 3.0533 | 0.5867 | 8.2250 | 7.9286 | 149.6399 | -0.0153 | 0.0000 | 0.0000 | 0.0000 | 185.3095 | -474.4398 | 165.4568 | 0.0000 | -4.2557 |
| **OIS 3Y** | 2.3475 | 8.2252 | -1.4538 | 16.8273 | 232.4987 | 0.0449 | 0.0000 | 0.0000 | 0.0000 | 159.0438 | -61.0342 | -257.5643 | 0.0000 | -7.9025 |
| **OIS 4Y** | 3.9348 | 7.9289 | 16.8277 | -23.3962 | 360.5467 | 0.1653 | 0.0000 | 0.0000 | 0.0000 | 3.3343 | 985.1518 | -1135.6216 | 0.0000 | -12.1345 |
| **OIS 5Y** | 75.6381 | 149.6400 | 232.4989 | 360.5469 | 776.8948 | -0.7271 | 0.0000 | 0.0000 | 0.0000 | -71.6283 | 565.7263 | 2884.1547 | 0.0000 | 50.6674 |
| **OIS 7Y** | -0.0078 | -0.0153 | 0.0449 | 0.1653 | -0.7271 | 0.2470 | 0.0000 | 0.0000 | 0.0000 | 0.0107 | 0.0102 | 5.4147 | 0.0000 | -0.0043 |
| **OIS 10Y** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **OIS 15Y** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **OIS 20Y** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **CDS 1Y** | -54.4327 | 185.3072 | 159.0429 | 3.3343 | -71.6285 | 0.0107 | 0.0000 | 0.0000 | 0.0000 | -2.0200 | 20.0437 | 217.7609 | 0.0000 | 2.6383 |
| **CDS 3Y** | -96.6639 | -474.4334 | -61.0323 | 985.1167 | 565.7528 | 0.0102 | 0.0000 | 0.0000 | 0.0000 | 20.0441 | -11.4210 | 998.8022 | 0.0000 | -15.4429 |
| **CDS 5Y** | 84.1899 | 165.4561 | -257.5722 | -1135.5915 | 2884.1418 | 5.4140 | 0.0000 | 0.0000 | 0.0000 | 217.7598 | 998.7859 | 1242.4242 | 0.0000 | -8.6844 |
| **CDS 7Y** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Recovery** | -1.6249 | -4.2558 | -7.9027 | -12.1347 | 50.6687 | -0.0043 | 0.0000 | 0.0000 | 0.0000 | 2.6384 | -15.4432 | -8.6947 | 0.0000 | -1.4560 |

</details>

> **Key insight:** The IR×CR cross-gamma block shows significant
> second-order interaction between interest rate and credit risk,
> demonstrating non-trivial curvature that is missed by first-order sensitivities.

### FD-over-AAD vs Pure FD — Difference Matrix

The table below shows $(H^{\text{AAD}}_{ij} - H^{\text{FD}}_{ij})$,
i.e. the element-wise difference between the Hessian computed via
FD-over-AAD and the one computed via pure finite differences.

<details>
<summary>Click to expand 14×14 difference matrix</summary>

| | OIS 1Y | OIS 2Y | OIS 3Y | OIS 4Y | OIS 5Y | OIS 7Y | OIS 10Y | OIS 15Y | OIS 20Y | CDS 1Y | CDS 3Y | CDS 5Y | CDS 7Y | Recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **OIS 1Y** | -4.7705e-04 | -3.8036e-07 | -2.0847e-05 | -6.7576e-05 | -7.4183e-04 | -1.0077e-04 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 4.1523e-04 | 8.0066e-04 | -6.8962e-04 | 0.0000e+00 | 2.7993e-05 |
| **OIS 2Y** | 4.1940e-05 | 2.6654e-04 | -1.4035e-04 | -6.1130e-05 | -1.3017e-03 | -5.7605e-06 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 6.1013e-04 | 1.2389e-03 | -1.0576e-03 | 0.0000e+00 | 1.7048e-05 |
| **OIS 3Y** | 2.9259e-05 | -4.7741e-06 | -3.0868e-04 | -1.1799e-04 | -2.1677e-03 | -3.2051e-05 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 5.2242e-04 | -2.3463e-04 | 7.9644e-04 | 0.0000e+00 | -1.5384e-05 |
| **OIS 4Y** | 7.6217e-05 | 2.3267e-04 | 3.2332e-04 | -1.0118e-03 | -3.1447e-03 | -3.0726e-05 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 3.6401e-05 | 1.5960e-02 | -1.4791e-02 | 0.0000e+00 | -3.5864e-05 |
| **OIS 5Y** | -7.1071e-04 | -1.2321e-03 | -1.9500e-03 | -2.9200e-03 | -1.2194e-02 | 1.1894e-04 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 6.3098e-04 | -2.4329e-02 | -1.1733e-02 | 0.0000e+00 | -4.4981e-04 |
| **OIS 7Y** | -1.0133e-04 | -6.8868e-06 | -3.2267e-05 | -3.1345e-05 | 7.4363e-05 | -1.1868e-04 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | -3.5174e-05 | 5.5011e-05 | 9.9510e-05 | 0.0000e+00 | 8.6826e-06 |
| **OIS 10Y** | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| **OIS 15Y** | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| **OIS 20Y** | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| **CDS 1Y** | 5.1966e-04 | -1.7132e-03 | -4.0426e-04 | -2.3828e-06 | 4.3489e-04 | -3.5448e-05 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0769e-04 | -5.9537e-05 | -1.3297e-03 | 0.0000e+00 | 6.6529e-05 |
| **CDS 3Y** | -1.7026e-05 | 7.6106e-03 | 1.6810e-03 | -1.9136e-02 | 2.2145e-03 | 5.4405e-05 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 3.0983e-04 | -8.8227e-04 | -8.2946e-03 | 0.0000e+00 | 2.2044e-05 |
| **CDS 5Y** | -9.5044e-04 | -1.8487e-03 | -7.1036e-03 | 1.5259e-02 | -2.4615e-02 | -6.2083e-04 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | -2.3847e-03 | -2.4549e-02 | -4.1295e-03 | 0.0000e+00 | 1.0097e-02 |
| **CDS 7Y** | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| **Recovery** | -1.2442e-05 | -7.0316e-05 | -1.7035e-04 | -1.8755e-04 | 8.6171e-04 | 8.6959e-06 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.4925e-04 | -2.2797e-04 | -1.7575e-04 | 0.0000e+00 | 5.7699e-05 |

</details>

| Metric | Value |
|---|---:|
| Max \|difference\| | 2.4615e-02 |
| Mean \|difference\| | 1.2406e-03 |
| Max \|Hessian entry\| | 2.8842e+03 |
| Relative error (max \|Δ\| / max \|H\|) | 8.5345e-06 |

> ✅ **Acceptable.** The relative difference is well below 1%,
> confirming both methods agree to high precision.

### Timing

| Method | Time (ms) | Operations |
|---|---:|---|
| FD-over-AAD | 925.09 ±27.53 | 15 AAD recordings |
| Pure FD | 23934.71 ±561.16 | 393 forward pricings |
| **Speedup** | **25.9×** | |

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
python benchmarks/second_order_risky_bond.py
python benchmarks/second_order_risky_bond.py --repeats 50
```

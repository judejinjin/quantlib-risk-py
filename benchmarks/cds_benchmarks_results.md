# QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: CDS

**Date:** 2026-02-26 14:42  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**MC scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Instrument

- **CDS** priced with **MidPointCdsEngine**
- 4 quoted spreads (3M, 6M, 1Y, 2Y) + recovery rate + risk-free rate = **6 inputs**
- Hazard curve bootstrap via `PiecewiseFlatHazardRate` + `SpreadCdsHelper`
- Nominal: 1,000,000

---

## Results

### CDS (MidPointCdsEngine)  (6 inputs, 100 scenarios per batch)

| Method | Non-JIT batch (ms) | JIT batch (ms) | JIT speedup | Per-scenario NJ | Per-scenario JIT |
|---|---:|---:|---:|---:|---:|
| FD (N+1 pricings per scenario) | 78.3 ±50.7† | 77.6 ±4.7 | 1.01× | 783 µs | 776 µs |
| **AAD replay** (backward sweep only) | 1.2 ±0.3 | 1.3 ±0.4 | 0.98× | 12 µs | 13 µs |
| AAD re-record (forward + backward) | 42.6 ±7.9 | 45.7 ±5.2 | 0.93× | 426 µs | 457 µs |
| *FD ÷ AAD replay (non-JIT / JIT)* | *63×* | *61×* | — | — | — |
| *FD ÷ AAD re-record (non-JIT / JIT)* | *1.8×* | *1.7×* | — | — | — |

---

**†** High variance (stdev/median > 50%).

## Notes

- The **MidPointCdsEngine** core NPV computation is straight-line arithmetic
  with no data-dependent branching on `Real`, making it fully JIT-compatible.
- Convenience outputs like `fairSpread()` and `fairUpfront()` do branch on `Real`
  but are not used in the NPV benchmark.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/cds_benchmarks.py            # default 5 repeats
python benchmarks/cds_benchmarks.py --repeats 10
```

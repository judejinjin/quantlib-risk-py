# QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: Vanilla IRS

**Date:** 2026-02-25 19:26  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**MC scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## Results

### Vanilla IRS  (17 inputs, 100 scenarios per batch)

*FD detail: 18 complete curve bootstraps + swap valuations per scenario*

| Method | Non-JIT batch (ms) | JIT batch (ms) | JIT speedup | Per-scenario NJ | Per-scenario JIT |
|---|---:|---:|---:|---:|---:|
| FD (N+1 pricings per scenario) | 726.4 ±16.2 | 767.7 ±51.4 | 0.95× | 7264 µs | 7677 µs |
| **AAD replay** (backward sweep only) | 3.4 ±0.6 | 3.6 ±0.3 | 0.94× | 34 µs | 36 µs |
| AAD re-record (forward + backward) | 130.5 ±4.0 | 131.9 ±2.5 | 0.99× | 1305 µs | 1319 µs |
| *FD ÷ AAD replay (non-JIT / JIT)* | *212×* | *211×* | — | — | — |
| *FD ÷ AAD re-record (non-JIT / JIT)* | *5.6×* | *5.8×* | — | — | — |

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/monte_carlo_irs_benchmarks.py            # default 5 repeats
python benchmarks/monte_carlo_irs_benchmarks.py --repeats 10
```

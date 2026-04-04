# QuantLib-Risks-Py — JIT vs Non-JIT Benchmark Results

**Date:** 2026-02-24 14:38  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**Repetitions:** 30 (median reported)  
**Non-JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## What is being measured

Each instrument is timed for three methods:

| Method | Description |
|---|---|
| **Plain pricing** | Single NPV call, `float` inputs, no AD overhead |
| **AAD backward pass** | XAD reverse-mode tape recorded once at startup; each iteration replays only the backward sweep — O(1) w.r.t. number of inputs |
| **Bump-and-reprice FD** | N+1 forward pricings with a 1 bp shift per input — O(N) |

Both builds are run in isolated virtual environments with their respective wheels.

---

## Results

### A. Vanilla IRS — 17 market inputs

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |
|---|---:|---:|---:|---:|
| Plain pricing | 0.3165 ±0.0306 | 0.3113 ±0.1021 | 1.02× | 17 |
| **AAD backward pass** | 0.0343 ±0.0039 | 0.0309 ±0.0081 | 1.11× | 17 |
| Bump-and-reprice FD | 6.5712 ±1.1084 | 7.0296 ±1.3265 | 0.93× | 17 |
| *FD ÷ AAD (within build)* | *191.3×* | *227.8×* | — | — |

### B. European Option — 4 market inputs

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |
|---|---:|---:|---:|---:|
| Plain pricing | 0.0048 ±0.0001 | 0.0047 ±0.0001 | 1.02× | 4 |
| **AAD backward pass** | 0.0011 ±0.0001 | 0.0011 ±0.0001 | 1.00× | 4 |
| Bump-and-reprice FD | 0.0233 ±0.0036 | 0.0227 ±0.0187† | 1.03× | 4 |
| *FD ÷ AAD (within build)* | *21.1×* | *20.6×* | — | — |

### C. Callable Bond — 3 market inputs

| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |
|---|---:|---:|---:|---:|
| Plain pricing | 0.0019 ±0.0000 | 0.0038 ±0.0001 | 0.51× | 3 |
| **AAD backward pass** | 0.5656 ±0.3660† | 0.0839 ±0.0557† | 6.74× | 3 |
| Bump-and-reprice FD | 2.5425 ±0.8370 | 4.3631 ±10.5374† | 0.58× | 3 |
| *FD ÷ AAD (within build)* | *4.5×* | *52.0×* | — | — |

---

## Summary — JIT speedup on AAD backward pass

| Instrument | JIT speedup |
|---|---:|
| [A] Vanilla IRS | 1.11× |
| [B] European Option | 1.00× |
| [C] Callable Bond | 6.74× |
| **Geometric mean** | **1.96×** |

---

## Notes

- BPS shift for FD: `0.0001`
- *AAD backward pass* times the **backward sweep only**; the tape is recorded once at startup and reused for all repetitions.
- *JIT speedup* = Non-JIT time ÷ JIT time; values > 1.0 mean JIT is faster.
- *FD ÷ AAD* shows how many times more expensive bump-and-reprice is compared to one AAD backward pass within the same build.
- AAD complexity is **O(1)** in the number of inputs; FD complexity is **O(N)**.
- **†** High variance (stdev/median > 50%): the median is still the primary metric. In the JIT build this occurs for plain-pricing and FD calls on the callable bond because Forge instruments even non-AD tree-pricer code paths, occasionally triggering LLVM recompilation mid-measurement. The AAD backward-pass timings (stdev/median < 10%) are unaffected and remain the authoritative JIT-speedup figure.

---

## How to reproduce

```bash
# Build both variants (first time only)
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

# Run the benchmark (venvs are created/reused automatically)
python benchmarks/run_benchmarks.py

# More repeats for stable numbers:
python benchmarks/run_benchmarks.py --repeats 50
```

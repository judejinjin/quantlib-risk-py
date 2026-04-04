# QuantLib-Risks-Py — Callable Bond Benchmark: FD vs AAD

**Date:** 2026-02-26 12:31  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**MC scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  

---

## What is being measured

Three methods for computing sensitivities of a Callable Fixed-Rate Bond
(HullWhite tree engine, 40 steps, 3 inputs: flat rate r, mean-reversion a,
volatility σ) across 100 randomly perturbed Monte Carlo scenarios:

| Method | Description |
|---|---|
| **FD (bump-and-reprice)** | N+1 = 4 tree pricings per scenario (base + 1 bp bump per input). HullWhite model is rebuilt for a and σ bumps. |
| **AAD replay** | Tape recorded once at base parameters; each scenario replays only the backward sweep — O(1) w.r.t. number of inputs. |
| **AAD re-record** | Per-scenario fresh `Real` inputs, QL objects, and tape recording + backward sweep. Correct per-scenario sensitivities. |

---

## Results

### Callable Bond — 3 inputs, 100 scenarios per batch

| Method | Batch time (ms) | Per-scenario | FD ÷ method |
|---|---:|---:|---:|
| **FD** (bump-and-reprice) | 335.0 ±7.7 | 3350 µs | 1.0× |
| **AAD replay** (backward sweep) | 69.7 ±1.6 | 697 µs | 4.8× |
| **AAD re-record** (forward+backward) | 675.3 ±12.5 | 6753 µs | 0.5× |

---

## Key ratios

| Ratio | Value | Interpretation |
|---|---:|---|
| FD ÷ AAD replay | **5×** | Replay gets all 3 sensitivities in one backward sweep |
| FD ÷ AAD re-record | **0.5×** | Re-record = full forward + backward per scenario |
| Re-record ÷ replay | **10×** | Cost of re-recording the tape each scenario |

---

## Notes

- BPS shift for FD: `0.0001`
- *AAD replay* is fastest but uses sensitivities at base parameters for all scenarios (valid for small perturbations around a single market state).
- *AAD re-record* produces correct per-scenario sensitivities at the cost of re-recording the full computation graph each time.
- AAD complexity is **O(1)** in the number of inputs; FD complexity is **O(N)**.
- JIT/Forge backend is not used: the TreeCallableFixedRateBondEngine contains data-dependent branching incompatible with Forge's record-once-replay-many paradigm (see `JIT_LIMITATIONS.md`).

---

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)

python benchmarks/monte_carlo_bond_benchmarks.py            # default 5 repeats
python benchmarks/monte_carlo_bond_benchmarks.py --repeats 10
```

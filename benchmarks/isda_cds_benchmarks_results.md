# QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: ISDA CDS Engine

**Date:** 2026-03-01 19:17  
**Platform:** Linux x86_64  
**Python:** 3.13.5  
**MC scenarios per batch:** 100  
**Outer repetitions:** 5 (median reported)  
**Wheel:** `quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl`  
**JIT:** Not used (IsdaCdsEngine has data-dependent branching on Real)  

---

## Instrument

- **CDS** priced with **IsdaCdsEngine**
- 6 deposit rates (1M–12M) + 14 swap rates (2Y–30Y) = **20 curve inputs**
- Discount curve: `PiecewiseFlatForward` bootstrapped from deposit + swap helpers
- CDS: 10Y term, spread 10 bps, recovery 40%, notional 10M
- Pipeline: curve bootstrap → `impliedHazardRate` → `FlatHazardRate` → `IsdaCdsEngine` → NPV

---

## Results

### CDS (IsdaCdsEngine)  (20 inputs, 100 scenarios per batch)

| Method | Batch (ms) | Per-scenario |
|---|---:|---:|
| FD (N+1 pricings per scenario) | 6573.6 ±219.8 | 65736 µs |
| **AAD replay** (backward sweep only) | 27.4 ±1.4 | 274 µs |
| AAD re-record (forward + backward) | 1654.9 ±24.0 | 16549 µs |
| *FD ÷ AAD replay* | *240×* | — |
| *FD ÷ AAD re-record* | *4.0×* | — |

---

## Why no JIT?

The `IsdaCdsEngine` contains a data-dependent branch on `Real`:

```cpp
// isdacdsengine.cpp, line ~193 and ~262:
Real fhphh = log(P0) - log(P1) + log(Q0) - log(Q1);
if (fhphh < 1E-4 && numericalFix_ == Taylor) { ... }
```

`fhphh` depends on discount/survival factors which are functions of the AD
inputs. The Forge JIT backend evaluates `if` at record time and bakes the
decision into the compiled kernel. Replaying with different inputs may take
the wrong branch, producing incorrect results or crashing.

See [JIT_LIMITATIONS.md](../JIT_LIMITATIONS.md) for full details.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)

python benchmarks/isda_cds_benchmarks.py            # default 5 repeats
python benchmarks/isda_cds_benchmarks.py --repeats 10
```

# QuantLib-Risks-Py — MC Scenario Risk Benchmark: OIS-Bootstrapped IRS

**Date:** 2026-02-26 16:19  
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
- OIS discount/forecasting curve: `PiecewiseLogLinearDiscount` bootstrapped
  from 9 `OISRateHelper` instruments using the `Sofr` overnight index
- Engine: `DiscountingSwapEngine` discounting off the OIS curve

### Market data (US Treasury daily par yield curve (2026-02-26))

| Tenor | Rate |
|-------|-----:|
| 1M | 3.74% |
| 3M | 3.68% |
| 6M | 3.61% |
| 1Y | 3.52% |
| 2Y | 3.42% |
| 3Y | 3.46% |
| 5Y | 3.57% |
| 10Y | 4.02% |
| 30Y | 4.67% |

---

## Results

### SOFR OIS (5Y)  (9 inputs, 100 scenarios per batch)

| Method | Non-JIT batch (ms) | JIT batch (ms) | JIT speedup | Per-scenario NJ | Per-scenario JIT |
|---|---:|---:|---:|---:|---:|
| FD (N+1 pricings per scenario) | 404.4 ±3.5 | 420.3 ±10.5 | 0.96× | 4044 µs | 4203 µs |
| **AAD replay** (backward sweep only) | 3.0 ±0.2 | 2.9 ±0.6 | 1.03× | 30 µs | 29 µs |
| AAD re-record (forward + backward) | 5143.1 ±40.8 | 5176.1 ±72.8 | 0.99× | 51431 µs | 51761 µs |
| *FD ÷ AAD replay (non-JIT / JIT)* | *136×* | *145×* | — | — | — |
| *FD ÷ AAD re-record (non-JIT / JIT)* | *0.1×* | *0.1×* | — | — | — |

---

## Notes

- By default, rates are scraped live from the **US Treasury daily par yield curve**
  at treasury.gov.  If scraping fails, the script warns and falls back to
  hardcoded Nov 2024 SOFR OIS rates.  Use `--offline` to skip scraping.
- Treasury par yields are a close proxy for SOFR OIS swap rates;
  the AD performance comparison is valid with either source.
- The OIS curve bootstrap via `PiecewiseLogLinearDiscount` and
  `DiscountingSwapEngine` are **straight-line arithmetic** with no data-dependent
  branching on `Real`, making this pipeline fully JIT-compatible.
- The same OIS curve serves as both the discounting and forecasting curve,
  consistent with single-curve SOFR pricing methodology.

## How to reproduce

```bash
./build.sh --no-jit -j$(nproc)
./build.sh --jit    -j$(nproc)

python benchmarks/ois_bootstrapped_IRS_benchmarks.py            # live rates (default)
python benchmarks/ois_bootstrapped_IRS_benchmarks.py --offline  # hardcoded Nov 2024
python benchmarks/ois_bootstrapped_IRS_benchmarks.py --repeats 10
```

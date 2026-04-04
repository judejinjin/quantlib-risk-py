# Callable Bond Benchmark Analysis: FD vs AAD Replay vs AAD Re-record

## Why three modes?

The benchmark compares three methods for computing sensitivities of a Callable
Fixed-Rate Bond (HullWhite tree engine, 3 inputs: flat rate _r_, mean-reversion
_a_, volatility _σ_) across 100 Monte Carlo scenarios.

---

## Mode descriptions

### FD — Finite Differences (bump-and-reprice)

- For each scenario _(rᵢ, aᵢ, σᵢ)_, price at the scenario point, then bump
  each input by 1 bp and reprice → **N+1 = 4 tree pricings per scenario**.
- Produces correct per-scenario sensitivities.
- Cost scales as **O(N)** in the number of inputs.

### AAD Replay — Backward sweep on a fixed tape

- The tape (computation graph) is recorded **once** at base parameters
  _(r₀, a₀, σ₀)_.
- For each scenario, only `computeAdjoints()` is called — a single backward
  sweep on the same tape.
- **Fast**, but the derivatives ∂NPV/∂r, ∂NPV/∂a, ∂NPV/∂σ are always
  evaluated **at the base point**, not at the scenario's perturbed parameters.
- Valid when perturbations are small (linear regime) and you only need
  base-point sensitivities. **Incorrect** if you need exact sensitivities at
  each scenario's market state.

### AAD Re-record — Per-scenario tape recording + backward sweep

- For each scenario _(rᵢ, aᵢ, σᵢ)_, creates fresh XAD `Real` inputs, rebuilds
  all QuantLib objects, records a new tape, then calls `computeAdjoints()`.
- The derivatives are computed **at the actual scenario parameters** —
  mathematically correct per-scenario sensitivities.
- Much slower because you pay the full forward pricing + tape recording cost
  each time.

---

## Correctness vs cost trade-off

| Mode | Correct per-scenario? | Cost per scenario | Scaling with N inputs |
|---|---|---|---|
| **FD** | ✅ Yes (reprices at each point) | N+1 pricings | O(N) |
| **AAD replay** | ❌ No (base-point sensitivities) | 1 backward sweep | O(1) |
| **AAD re-record** | ✅ Yes (tape at actual point) | 1 forward + 1 backward | O(1) |

---

## Observed results (100 scenarios, 5 repeats, median)

| Method | Batch time | Per-scenario | FD ÷ method |
|---|---:|---:|---:|
| FD (bump-and-reprice) | 335.0 ms | 3350 µs | 1.0× |
| AAD replay (backward sweep) | 69.7 ms | 697 µs | **4.8×** |
| AAD re-record (forward+backward) | 675.3 ms | 6753 µs | 0.5× |

### Key ratios

| Ratio | Value | Interpretation |
|---|---:|---|
| FD ÷ AAD replay | **~5×** | Replay gets all 3 sensitivities in one backward sweep vs 4 tree pricings |
| FD ÷ AAD re-record | **~0.5×** | With only 3 inputs, FD's 4 pricings are cheaper than full tape construction |
| Re-record ÷ replay | **~10×** | Quantifies the cost of recording the tape each scenario |

---

## Why include re-record if it's slower than FD?

For this particular instrument (3 inputs), FD wins over re-record because:

- FD requires only 4 tree pricings — a low constant.
- Re-record must construct all QuantLib objects, record the full tape (which
  tracks every intermediate operation through the tree engine), and then run the
  backward sweep.

However, **AAD complexity is O(1) in the number of inputs** while FD is O(N).
As the number of inputs grows, re-record becomes increasingly favorable:

| N inputs | FD pricings | AAD re-record | FD ÷ re-record |
|---:|---:|---|---|
| 3 | 4 | 1 forward + 1 backward | ~0.5× (FD wins) |
| 10 | 11 | 1 forward + 1 backward | ~1.4× (break-even) |
| 17 (IRS) | 18 | 1 forward + 1 backward | ~2.3× (AAD wins) |
| 50 | 51 | 1 forward + 1 backward | ~6.5× (AAD wins decisively) |

The cross-over point depends on the instrument and engine, but typically AAD
re-record overtakes FD somewhere around **6–10 inputs**.

The benchmark includes re-record to:

1. **Quantify the tape-recording overhead** (the 10× gap vs replay).
2. **Show the full cost picture** — users can decide whether replay's
   approximation is acceptable or whether they need correct per-scenario
   sensitivities.
3. **Demonstrate the O(1) scaling argument** that makes AAD dominant for
   higher-dimensional problems.

---

## When to use each mode

| Use case | Recommended mode |
|---|---|
| Risk at a single market state (Greeks) | AAD replay |
| Stress testing with small perturbations | AAD replay (approximate) |
| Full Monte Carlo VaR / CVA with per-path Greeks | AAD re-record |
| Few inputs (≤5), many scenarios | FD (simpler, competitive speed) |
| Many inputs (≥10), many scenarios | AAD re-record |

---

## Note on JIT/Forge

The JIT (Forge) backend is **not used** in this benchmark. Forge's
record-once-compile-once-evaluate-many paradigm is incompatible with the
`TreeCallableFixedRateBondEngine`, which contains data-dependent branching
(`if` statements evaluated at pricing time). See `JIT_LIMITATIONS.md` for
details.

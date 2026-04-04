#!/usr/bin/env python3
"""
QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: Credit Default Swap
==============================================================================

Benchmarks a CDS priced with the MidPointCdsEngine under N_SCENARIOS = 100
random Monte Carlo scenarios, comparing:

  C1) FD             – bump-and-reprice (6 inputs → 7 full re-pricings per scenario)
  C2) AAD replay     – N backward sweeps on a tape recorded once at base market
  C3) AAD re-record  – per-scenario fresh recording via SimpleQuote.setValue(Real)

The MidPointCdsEngine core NPV computation is straight-line arithmetic (no
data-dependent branching on Real), so JIT (Forge) can be used for the replay
benchmark.  Both non-JIT and JIT builds are tested.

Inputs (6):
  - 4 quoted CDS spreads   (3M, 6M, 1Y, 2Y)
  - 1 recovery rate
  - 1 risk-free rate

Usage
-----
  python benchmarks/cds_benchmarks.py               # default 5 repeats
  python benchmarks/cds_benchmarks.py --repeats 10
  python benchmarks/cds_benchmarks.py --clean-venvs

  # Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/cds_benchmarks.py --worker REPEATS
"""

import argparse
import datetime
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
BUILD      = ROOT / "build"
VENV_NOJIT = BUILD / "bench-venv-nojit"
VENV_JIT   = BUILD / "bench-venv-jit"

SEPARATOR = "=" * 84
BPS = 1e-4

# ------- scenario parameters -----------------------------------------------
N_SCENARIOS   = 100
SCENARIO_SEED = 123          # distinct seed from IRS/bond benchmarks

# CDS base market  (6 inputs)
_CDS_BASE_SPREADS = [0.0150, 0.0150, 0.0150, 0.0150]
_CDS_BASE_RECOVERY = 0.5
_CDS_BASE_RISKFREE = 0.01
_CDS_BASE = _CDS_BASE_SPREADS + [_CDS_BASE_RECOVERY, _CDS_BASE_RISKFREE]
_N_INPUTS = len(_CDS_BASE)    # 6

_CDS_TENORS = [
    (3, "Months"),
    (6, "Months"),
    (1, "Years"),
    (2, "Years"),
]


def _gen_scenarios():
    """Pre-generate CDS MC scenarios deterministically."""
    rng = random.Random(SCENARIO_SEED)
    scenarios = []
    for _ in range(N_SCENARIOS):
        spreads = [max(1e-5, s + rng.gauss(0, 2e-3)) for s in _CDS_BASE_SPREADS]
        rec     = min(0.99, max(0.01, _CDS_BASE_RECOVERY + rng.gauss(0, 0.02)))
        rfr     = max(0.0001, _CDS_BASE_RISKFREE + rng.gauss(0, 5e-4))
        scenarios.append(spreads + [rec, rfr])
    return scenarios


# ============================================================================
# Wheel / venv helpers  (shared pattern with IRS benchmarks)
# ============================================================================

def find_wheels(build_root: Path) -> dict:
    def latest(pattern):
        matches = sorted(build_root.glob(pattern))
        return matches[-1] if matches else None
    return {
        "nojit": {
            "xad": latest("xad-python/dist/xad_autodiff-*.whl"),
            "ql":  latest("linux-xad-gcc-ninja-release/Python/dist/quantlib_risks-*.whl"),
        },
        "jit": {
            "xad": latest("xad-python-jit/dist/xad_autodiff-*.whl"),
            "ql":  latest("linux-xad-jit-gcc-ninja-release/Python/dist/quantlib_risks-*.whl"),
        },
    }


def python_in(venv: Path) -> Path:
    return venv / "bin" / "python"


def _install_xad_shim(py: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "xad-shim"
        for subdir in ["xad/adj_1st", "xad/fwd_1st", "xad/math", "xad/exceptions"]:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        (base / "xad/__init__.py").write_text(
            "from xad_autodiff import *\n"
            "from xad_autodiff import _xad_autodiff, adj_1st, fwd_1st, value, derivative\n"
        )
        (base / "xad/adj_1st/__init__.py").write_text(
            "from xad_autodiff.adj_1st import *\n"
            "from xad_autodiff.adj_1st import Real, Tape\n"
        )
        (base / "xad/fwd_1st/__init__.py").write_text("from xad_autodiff.fwd_1st import *\n")
        (base / "xad/math/__init__.py").write_text("from xad_autodiff.math import *\n")
        (base / "xad/exceptions/__init__.py").write_text(
            "from xad_autodiff.exceptions import *\n"
        )
        (base / "pyproject.toml").write_text(textwrap.dedent("""\
            [build-system]
            requires = ["setuptools>=42"]
            build-backend = "setuptools.build_meta"
            [project]
            name = "xad"
            version = "1.5.2"
            description = "Compatibility shim: maps import xad to xad_autodiff"
            requires-python = ">=3.8"
            dependencies = ["xad-autodiff>=1.5.1"]
        """))
        subprocess.check_call(
            [py, "-m", "pip", "install", "--quiet", "--force-reinstall",
             "--no-deps", "--no-build-isolation", str(base)]
        )


def install_wheels(venv: Path, xad_wheel: Path, ql_wheel: Path):
    py = str(python_in(venv))
    print("    • pip install setuptools wheel")
    subprocess.check_call([py, "-m", "pip", "install", "--quiet", "setuptools", "wheel"])
    print(f"    • pip install {xad_wheel.name}")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps",
         str(xad_wheel)]
    )
    print("    • pip install xad compatibility shim")
    _install_xad_shim(py)
    print(f"    • pip install {ql_wheel.name}")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps",
         str(ql_wheel)]
    )


def venv_is_ready(venv: Path) -> bool:
    py = python_in(venv)
    if not py.exists():
        return False
    result = subprocess.run(
        [str(py), "-c", "import QuantLib_Risks; import xad"],
        capture_output=True,
    )
    return result.returncode == 0


def setup_venv(venv: Path, xad_wheel: Path, ql_wheel: Path, force: bool = False):
    if force and venv.exists():
        print(f"    --clean-venvs: removing {venv}")
        shutil.rmtree(venv)
    if venv_is_ready(venv):
        print(f"    Reusing existing venv: {venv.name}")
        return
    print(f"    Creating venv: {venv}")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    install_wheels(venv, xad_wheel, ql_wheel)
    print(f"    Venv ready: {venv.name}")


def _clean_env() -> dict:
    drop = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHON_BASIC_REPL",
            "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT"}
    return {k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith(("CONDA_", "PYTHON_"))}


def run_worker_in_venv(venv: Path, repeats: int) -> dict:
    py = str(python_in(venv))
    result = subprocess.run(
        [py, str(Path(__file__).resolve()), "--worker", str(repeats)],
        capture_output=True, text=True,
        env=_clean_env(),
    )
    if result.returncode != 0:
        print(f"\n  Worker exit code: {result.returncode}")
        print(f"  Worker STDOUT (last 2000 chars):\n{result.stdout[-2000:]}")
        print(f"  Worker STDERR (last 4000 chars):\n{result.stderr[-4000:]}")
        raise RuntimeError(f"Worker failed in {venv.name}")
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(
        f"No JSON found in worker output ({venv.name}):\n{result.stdout[-2000:]}"
    )


# ============================================================================
# WORKER MODE — benchmark implementations
# ============================================================================

def _median_ms(func, n: int, warmup: int = 1):
    """Return (median_ms, stdev_ms) over n timed calls."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ---- CDS with MidPointCdsEngine (6 inputs) ---------------------------------

def _build_cds_structures():
    """
    Build the CDS plain-pricing structure (float SimpleQuotes) and the base
    AAD tape (for replay).

    Returns:
        cds_list    – list of 4 CreditDefaultSwap objects (float engine)
        cds_quotes  – list of 6 SimpleQuote objects (4 spreads + recovery + rfr)
        tape_cds    – Tape recorded at base market (for AAD replay)
        npv_cds     – xad Real output registered on tape_cds
        cds_inputs  – list of 6 xad Reals registered on tape_cds
    """
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape, Real

    calendar   = ql.TARGET()
    todaysDate = ql.Date(15, ql.May, 2007)
    ql.Settings.instance().evaluationDate = todaysDate

    # --- plain-float structure for FD ---
    spread_quotes = [ql.SimpleQuote(s) for s in _CDS_BASE_SPREADS]
    recovery_quote = ql.SimpleQuote(_CDS_BASE_RECOVERY)
    riskfree_quote = ql.SimpleQuote(_CDS_BASE_RISKFREE)
    cds_quotes = spread_quotes + [recovery_quote, riskfree_quote]

    risk_free_rate = ql.YieldTermStructureHandle(
        ql.FlatForward(todaysDate, ql.QuoteHandle(riskfree_quote), ql.Actual365Fixed()))

    tenors = [ql.Period(n, getattr(ql, u)) for n, u in _CDS_TENORS]
    maturities = [calendar.adjust(todaysDate + t, ql.Following) for t in tenors]

    instruments = [
        ql.SpreadCdsHelper(
            ql.QuoteHandle(sq), tenor, 0, calendar, ql.Quarterly,
            ql.Following, ql.DateGeneration.TwentiethIMM,
            ql.Actual365Fixed(), recovery_quote.value(), risk_free_rate,
        )
        for sq, tenor in zip(spread_quotes, tenors)
    ]
    hazard_curve = ql.PiecewiseFlatHazardRate(todaysDate, instruments, ql.Actual365Fixed())
    probability = ql.DefaultProbabilityTermStructureHandle(hazard_curve)

    nominal = 1_000_000.0
    cds_list = []
    for maturity, s in zip(maturities, _CDS_BASE_SPREADS):
        schedule = ql.Schedule(
            todaysDate, maturity, ql.Period(ql.Quarterly), calendar,
            ql.Following, ql.Unadjusted, ql.DateGeneration.TwentiethIMM, False,
        )
        cds = ql.CreditDefaultSwap(
            ql.Protection.Seller, nominal, s, schedule,
            ql.Following, ql.Actual365Fixed(),
        )
        engine = ql.MidPointCdsEngine(probability, recovery_quote.value(), risk_free_rate)
        cds.setPricingEngine(engine)
        cds_list.append(cds)

    # --- AAD tape for replay (recorded once at base market) ---
    # We price the 2Y CDS (last one) as the benchmark target
    tape_cds = Tape()
    tape_cds.activate()

    spread_r = [Real(s) for s in _CDS_BASE_SPREADS]
    recovery_r = Real(_CDS_BASE_RECOVERY)
    riskfree_r = Real(_CDS_BASE_RISKFREE)
    cds_inputs = spread_r + [recovery_r, riskfree_r]
    tape_cds.registerInputs(cds_inputs)
    tape_cds.newRecording()

    risk_free_rate_r = ql.YieldTermStructureHandle(
        ql.FlatForward(todaysDate, riskfree_r, ql.Actual365Fixed()))

    instruments_r = [
        ql.SpreadCdsHelper(
            ql.QuoteHandle(ql.SimpleQuote(sr)), tenor, 0, calendar, ql.Quarterly,
            ql.Following, ql.DateGeneration.TwentiethIMM,
            ql.Actual365Fixed(), recovery_r, risk_free_rate_r,
        )
        for sr, tenor in zip(spread_r, tenors)
    ]
    hazard_curve_r = ql.PiecewiseFlatHazardRate(
        todaysDate, instruments_r, ql.Actual365Fixed())
    probability_r = ql.DefaultProbabilityTermStructureHandle(hazard_curve_r)

    # Price the 2Y CDS
    maturity_2y = maturities[-1]
    schedule_r = ql.Schedule(
        todaysDate, maturity_2y, ql.Period(ql.Quarterly), calendar,
        ql.Following, ql.Unadjusted, ql.DateGeneration.TwentiethIMM, False,
    )
    cds_r = ql.CreditDefaultSwap(
        ql.Protection.Seller, nominal, spread_r[-1], schedule_r,
        ql.Following, ql.Actual365Fixed(),
    )
    engine_r = ql.MidPointCdsEngine(probability_r, recovery_r, risk_free_rate_r)
    cds_r.setPricingEngine(engine_r)
    npv_cds = cds_r.NPV()
    tape_cds.registerOutput(npv_cds)

    return cds_list, cds_quotes, tape_cds, npv_cds, cds_inputs


def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    from xad.adj_1st import Real, Tape

    scenarios = _gen_scenarios()
    results = {}

    (cds_list, cds_quotes,
     tape_cds, npv_cds, cds_inputs) = _build_cds_structures()

    # We benchmark FD on the 2Y CDS (last in list)
    cds_2y = cds_list[-1]

    # C1 — FD batch: 6-input bump-and-reprice for every scenario
    def _cds_fd_mc():
        for scene in scenarios:
            # Set base values
            for sq, v in zip(cds_quotes, scene):
                sq.setValue(v)
            cds_2y.NPV()
            # Bump each input
            for sq, v in zip(cds_quotes, scene):
                sq.setValue(v + BPS)
                cds_2y.NPV()
                sq.setValue(v)
        # Restore base
        for sq, v in zip(cds_quotes, _CDS_BASE):
            sq.setValue(v)

    m, s = _median_ms(_cds_fd_mc, repeats)
    results["cds_fd_mc"] = {"median": m, "stdev": s,
                             "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    # C2 — AAD replay: N_SCENARIOS backward sweeps on a fixed tape
    def _cds_aad_replay():
        for _ in range(N_SCENARIOS):
            tape_cds.clearDerivatives()
            npv_cds.derivative = 1.0
            tape_cds.computeAdjoints()

    m, s = _median_ms(_cds_aad_replay, repeats)
    results["cds_aad_replay"] = {"median": m, "stdev": s,
                                  "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}
    tape_cds.deactivate()

    # C3 — AAD re-record: per-scenario fresh recording
    def _cds_aad_record():
        calendar   = ql.TARGET()
        todaysDate = ql.Date(15, ql.May, 2007)
        nominal    = 1_000_000.0
        tenors = [ql.Period(n, getattr(ql, u)) for n, u in _CDS_TENORS]
        maturities = [calendar.adjust(todaysDate + t, ql.Following) for t in tenors]
        maturity_2y = maturities[-1]

        tape = Tape()
        tape.activate()
        for scene in scenarios:
            spread_r = [Real(v) for v in scene[:4]]
            recovery_r = Real(scene[4])
            riskfree_r = Real(scene[5])
            reals = spread_r + [recovery_r, riskfree_r]
            tape.registerInputs(reals)
            tape.newRecording()

            rfr = ql.YieldTermStructureHandle(
                ql.FlatForward(todaysDate, riskfree_r, ql.Actual365Fixed()))

            instr = [
                ql.SpreadCdsHelper(
                    ql.QuoteHandle(ql.SimpleQuote(sr)), tenor, 0, calendar,
                    ql.Quarterly, ql.Following, ql.DateGeneration.TwentiethIMM,
                    ql.Actual365Fixed(), recovery_r, rfr,
                )
                for sr, tenor in zip(spread_r, tenors)
            ]
            hc = ql.PiecewiseFlatHazardRate(todaysDate, instr, ql.Actual365Fixed())
            prob = ql.DefaultProbabilityTermStructureHandle(hc)

            sch = ql.Schedule(
                todaysDate, maturity_2y, ql.Period(ql.Quarterly), calendar,
                ql.Following, ql.Unadjusted, ql.DateGeneration.TwentiethIMM, False,
            )
            cds = ql.CreditDefaultSwap(
                ql.Protection.Seller, nominal, spread_r[-1], sch,
                ql.Following, ql.Actual365Fixed(),
            )
            eng = ql.MidPointCdsEngine(prob, recovery_r, rfr)
            cds.setPricingEngine(eng)
            npv = cds.NPV()
            tape.registerOutput(npv)
            npv.derivative = 1.0
            tape.computeAdjoints()
        tape.deactivate()

    m, s = _median_ms(_cds_aad_record, repeats)
    results["cds_aad_record"] = {"median": m, "stdev": s,
                                  "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison table
# ============================================================================

INSTRUMENTS = [
    # (id, label,     fd_key,       replay_key,       record_key)
    ("C", "CDS (MidPointCdsEngine)", "cds_fd_mc", "cds_aad_replay", "cds_aad_record"),
]


def _sp(nojit, jit):
    return f"{nojit / jit:.2f}×" if jit > 0 else "—"


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def print_comparison(nojit: dict, jit: dict, repeats: int, wheels: dict):
    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Monte Carlo Scenario Risk Benchmark: CDS")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  MC scenarios per batch : {N_SCENARIOS}")
    print(f"  Outer repeats          : {repeats}")
    print(f"  BPS shift (FD)         : {BPS}")
    print(f"  Non-JIT : {wheels['nojit']['ql'].name}")
    print(f"  JIT     : {wheels['jit']['ql'].name}")
    print()

    COL_M = 24
    COL_T = 22

    for _, label, fd_k, rpl_k, rec_k in INSTRUMENTS:
        n_in   = nojit[fd_k]["n_inputs"]
        n_scen = nojit[fd_k]["n_scenarios"]

        print(f"  ── {label}  ({n_in} inputs, {n_scen} scenarios per batch) ──")
        hdr = (f"  {'Method':<{COL_M}}"
               f"  {'Non-JIT batch':>{COL_T}}"
               f"  {'JIT batch':>{COL_T}}"
               f"  {'JIT speedup':>12}"
               f"  {'per-scenario (NJ)':>18}"
               f"  {'per-scenario (JIT)':>19}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for method, key in [
            ("FD (N+1 pricings)",       fd_k),
            ("AAD replay   (backward)", rpl_k),
            ("AAD re-record (fwd+bwd)", rec_k),
        ]:
            nj  = nojit[key]["median"]
            jt  = jit[key]["median"]
            njs = nojit[key]["stdev"]
            jts = jit[key]["stdev"]
            sp  = _sp(nj, jt)
            psnj = f"{nj / n_scen * 1000:.1f} µs"
            psjt = f"{jt / n_scen * 1000:.1f} µs"
            print(f"  {method:<{COL_M}}"
                  f"  {nj:>8.1f} ±{njs:>6.1f} ms"
                  f"  {jt:>8.1f} ±{jts:>6.1f} ms"
                  f"  {sp:>12}"
                  f"  {psnj:>18}"
                  f"  {psjt:>19}")

        fd_nj  = nojit[fd_k]["median"]
        fd_jt  = jit[fd_k]["median"]
        rpl_nj = nojit[rpl_k]["median"]
        rpl_jt = jit[rpl_k]["median"]
        rec_nj = nojit[rec_k]["median"]
        rec_jt = jit[rec_k]["median"]
        print(f"  {'FD/replay   (non-JIT / JIT)':<{COL_M}}"
              f"  {fd_nj/rpl_nj:>8.1f}×"
              f"  {fd_jt/rpl_jt:>12.1f}×")
        print(f"  {'FD/re-record(non-JIT / JIT)':<{COL_M}}"
              f"  {fd_nj/rec_nj:>8.1f}×"
              f"  {fd_jt/rec_jt:>12.1f}×")
        print()

    print(SEPARATOR)


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "cds_benchmarks_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: CDS")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**MC scenarios per batch:** {N_SCENARIOS}  ")
    w(f"**Outer repetitions:** {repeats} (median reported)  ")
    w(f"**Non-JIT wheel:** `{wheels['nojit']['ql'].name}`  ")
    w(f"**JIT wheel:** `{wheels['jit']['ql'].name}`  ")
    w("")
    w("---")
    w("")
    w("## Instrument")
    w("")
    w("- **CDS** priced with **MidPointCdsEngine**")
    w("- 4 quoted spreads (3M, 6M, 1Y, 2Y) + recovery rate + risk-free rate = **6 inputs**")
    w("- Hazard curve bootstrap via `PiecewiseFlatHazardRate` + `SpreadCdsHelper`")
    w("- Nominal: 1,000,000")
    w("")
    w("---")
    w("")
    w("## Results")
    w("")

    HIGH_CV = 0.5
    has_high_cv = False

    for _, label, fd_k, rpl_k, rec_k in INSTRUMENTS:
        n_in   = nojit[fd_k]["n_inputs"]
        n_scen = nojit[fd_k]["n_scenarios"]

        w(f"### {label}  ({n_in} inputs, {n_scen} scenarios per batch)")
        w("")
        w("| Method | Non-JIT batch (ms) | JIT batch (ms) | JIT speedup "
          "| Per-scenario NJ | Per-scenario JIT |")
        w("|---|---:|---:|---:|---:|---:|")

        for method, key in [
            ("FD (N+1 pricings per scenario)",       fd_k),
            ("**AAD replay** (backward sweep only)", rpl_k),
            ("AAD re-record (forward + backward)",   rec_k),
        ]:
            nj  = nojit[key]["median"]
            jt  = jit[key]["median"]
            njs = nojit[key]["stdev"]
            jts = jit[key]["stdev"]
            sp  = f"{nj/jt:.2f}×" if jt > 0 else "—"
            nj_flag = "†" if nj and njs / nj > HIGH_CV else ""
            jt_flag = "†" if jt and jts / jt > HIGH_CV else ""
            if nj_flag or jt_flag:
                has_high_cv = True
            psnj = f"{nj / n_scen * 1000:.0f} µs"
            psjt = f"{jt / n_scen * 1000:.0f} µs"
            w(f"| {method} "
              f"| {nj:.1f} ±{njs:.1f}{nj_flag} "
              f"| {jt:.1f} ±{jts:.1f}{jt_flag} "
              f"| {sp} | {psnj} | {psjt} |")

        fd_nj  = nojit[fd_k]["median"]
        fd_jt  = jit[fd_k]["median"]
        rpl_nj = nojit[rpl_k]["median"]
        rpl_jt = jit[rpl_k]["median"]
        rec_nj = nojit[rec_k]["median"]
        rec_jt = jit[rec_k]["median"]
        w(f"| *FD ÷ AAD replay (non-JIT / JIT)* "
          f"| *{fd_nj/rpl_nj:.0f}×* | *{fd_jt/rpl_jt:.0f}×* | — | — | — |")
        w(f"| *FD ÷ AAD re-record (non-JIT / JIT)* "
          f"| *{fd_nj/rec_nj:.1f}×* | *{fd_jt/rec_jt:.1f}×* | — | — | — |")
        w("")

    w("---")
    w("")
    if has_high_cv:
        w(f"**†** High variance (stdev/median > {HIGH_CV:.0%}).")
        w("")

    w("## Notes")
    w("")
    w("- The **MidPointCdsEngine** core NPV computation is straight-line arithmetic")
    w("  with no data-dependent branching on `Real`, making it fully JIT-compatible.")
    w("- Convenience outputs like `fairSpread()` and `fairUpfront()` do branch on `Real`")
    w("  but are not used in the NPV benchmark.")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("")
    w("python benchmarks/cds_benchmarks.py            # default 5 repeats")
    w("python benchmarks/cds_benchmarks.py --repeats 10")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks CDS (MidPointCdsEngine) benchmark",
    )
    parser.add_argument("--worker", metavar="REPEATS", type=int, default=None,
                        help="Internal worker mode: run benchmarks and print JSON")
    parser.add_argument("--repeats", "-r", type=int, default=5,
                        help=f"Outer repetitions per batch of {N_SCENARIOS} scenarios "
                             "(default: 5)")
    parser.add_argument("--clean-venvs", action="store_true",
                        help="Destroy and recreate benchmark venvs")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write results to markdown file")
    args = parser.parse_args()

    if args.worker is not None:
        worker_main(args.worker)
        return

    repeats = args.repeats
    wheels  = find_wheels(BUILD)

    missing = [f"{mode}/{kind}"
               for mode in ("nojit", "jit")
               for kind in ("xad", "ql")
               if wheels[mode][kind] is None]
    if missing:
        print("ERROR: Missing wheels for:", ", ".join(missing))
        print("  Run both builds first:")
        print("    ./build.sh --no-jit -j$(nproc)")
        print("    ./build.sh --jit -j$(nproc)")
        sys.exit(1)

    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Monte Carlo Scenario Risk Benchmark: CDS")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"  {N_SCENARIOS} scenarios per batch, {repeats} outer repetitions")
    print(SEPARATOR)

    print("\nSetting up virtual environments")
    print("-" * 50)
    print(f"\n[1/2] Non-JIT venv  ({VENV_NOJIT.name})")
    setup_venv(VENV_NOJIT, wheels["nojit"]["xad"], wheels["nojit"]["ql"],
               force=args.clean_venvs)
    print(f"\n[2/2] JIT venv      ({VENV_JIT.name})")
    setup_venv(VENV_JIT, wheels["jit"]["xad"], wheels["jit"]["ql"],
               force=args.clean_venvs)

    print(f"\nRunning benchmarks  ({repeats} outer repeats, {N_SCENARIOS} scenarios each)")
    print("-" * 50)
    print("\n  [1/2] Non-JIT worker …")
    nojit = run_worker_in_venv(VENV_NOJIT, repeats)
    print("        done.")
    print("\n  [2/2] JIT worker …")
    jit = run_worker_in_venv(VENV_JIT, repeats)
    print("        done.")

    print_comparison(nojit, jit, repeats, wheels)

    if not args.no_save:
        write_markdown(nojit, jit, repeats, wheels)


if __name__ == "__main__":
    main()

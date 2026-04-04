#!/usr/bin/env python3
"""
QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: Vanilla IRS
======================================================================

Benchmarks the Vanilla IRS (17 Euribor3M bootstrapped inputs) under
N_SCENARIOS = 100 random Monte Carlo scenarios, comparing:

  G1) FD             – bump-and-reprice (17+1 full curve bootstraps per scenario)
  G2) AAD replay     – N backward sweeps on a tape recorded once at base market
  G3) AAD re-record  – per-scenario fresh recording via SimpleQuote.setValue(Real)

Usage
-----
  python benchmarks/monte_carlo_irs_benchmarks.py               # default 5 repeats
  python benchmarks/monte_carlo_irs_benchmarks.py --repeats 10
  python benchmarks/monte_carlo_irs_benchmarks.py --clean-venvs

  # Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/monte_carlo_irs_benchmarks.py --worker REPEATS
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
SCENARIO_SEED = 42

# IRS base quotes (17 values – deposit, FRAs, futures, swaps)
_IRS_BASE = [
    0.0363,
    0.037125, 0.037125, 0.037125,
    96.2875, 96.7875, 96.9875, 96.6875,
    96.4875, 96.3875, 96.2875, 96.0875,
    0.037125, 0.0398, 0.0443, 0.05165, 0.055175,
]


def _gen_scenarios():
    """Pre-generate IRS MC scenarios deterministically."""
    rng = random.Random(SCENARIO_SEED)
    irs_scenarios = []
    for _ in range(N_SCENARIOS):
        scene = [v + rng.gauss(0, 5e-4) for v in _IRS_BASE]
        irs_scenarios.append(scene)
    return irs_scenarios


# ============================================================================
# Wheel / venv helpers
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
    """
    Return (median_ms, stdev_ms) over n timed calls.
    Each call processes all N_SCENARIOS — only 1 warmup needed.
    """
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ---- G.  Vanilla IRS  (17 bootstrapped Euribor3M inputs) -------------------

def _build_irs_mc_structures():
    """
    Build the IRS plain-pricing structure (float SimpleQuotes) and the base
    AAD tape (for replay).

    Returns:
        swap      – VanillaSwap (float engine)
        irs_quotes – list of 17 SimpleQuote objects
        tape_irs   – Tape recorded at base market (for AAD replay)
        npv_irs    – xad Real output registered on tape_irs
        irs_inputs – list of 17 xad Reals registered on tape_irs
    """
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape, Real

    calendar       = ql.TARGET()
    todaysDate     = ql.Date(6, ql.November, 2001)
    settlementDate = ql.Date(8, ql.November, 2001)
    ql.Settings.instance().evaluationDate = todaysDate

    # --- plain-float structure ---
    deps_f = {(3, ql.Months): ql.SimpleQuote(0.0363)}
    fras_f = {(3, 6): ql.SimpleQuote(0.037125), (6, 9): ql.SimpleQuote(0.037125),
              (9, 12): ql.SimpleQuote(0.037125)}
    futs_f = {
        ql.Date(19, 12, 2001): ql.SimpleQuote(96.2875),
        ql.Date(20,  3, 2002): ql.SimpleQuote(96.7875),
        ql.Date(19,  6, 2002): ql.SimpleQuote(96.9875),
        ql.Date(18,  9, 2002): ql.SimpleQuote(96.6875),
        ql.Date(18, 12, 2002): ql.SimpleQuote(96.4875),
        ql.Date(19,  3, 2003): ql.SimpleQuote(96.3875),
        ql.Date(18,  6, 2003): ql.SimpleQuote(96.2875),
        ql.Date(17,  9, 2003): ql.SimpleQuote(96.0875),
    }
    swps_f = {
        (2,  ql.Years): ql.SimpleQuote(0.037125),
        (3,  ql.Years): ql.SimpleQuote(0.0398),
        (5,  ql.Years): ql.SimpleQuote(0.0443),
        (10, ql.Years): ql.SimpleQuote(0.05165),
        (15, ql.Years): ql.SimpleQuote(0.055175),
    }
    irs_quotes = (list(deps_f.values()) + list(fras_f.values())
                  + list(futs_f.values()) + list(swps_f.values()))

    dayCounter = ql.Actual360()
    depH  = [ql.DepositRateHelper(ql.QuoteHandle(deps_f[(n, u)]), ql.Period(n, u),
                2, calendar, ql.ModifiedFollowing, False, dayCounter)
             for n, u in deps_f]
    fraH  = [ql.FraRateHelper(ql.QuoteHandle(fras_f[(n, m)]), n, m,
                2, calendar, ql.ModifiedFollowing, False, dayCounter)
             for n, m in fras_f]
    futH  = [ql.FuturesRateHelper(ql.QuoteHandle(futs_f[d]), d, 3,
                calendar, ql.ModifiedFollowing, True, dayCounter,
                ql.QuoteHandle(ql.SimpleQuote(0.0)))
             for d in futs_f]
    discountTS = ql.YieldTermStructureHandle(
        ql.FlatForward(settlementDate, 0.04, ql.Actual360()))
    swpH  = [ql.SwapRateHelper(ql.QuoteHandle(swps_f[(n, u)]), ql.Period(n, u),
                calendar, ql.Annual, ql.Unadjusted,
                ql.Thirty360(ql.Thirty360.BondBasis), ql.Euribor3M(),
                ql.QuoteHandle(), ql.Period("0D"), discountTS)
             for n, u in swps_f]
    fcastH = ql.RelinkableYieldTermStructureHandle()
    fcastH.linkTo(ql.PiecewiseFlatForward(
        settlementDate, depH + futH + swpH[1:], ql.Actual360()))
    maturity = calendar.advance(settlementDate, 5, ql.Years)
    index    = ql.Euribor3M(fcastH)
    fxSch = ql.Schedule(settlementDate, maturity, ql.Period(1, ql.Years), calendar,
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    flSch = ql.Schedule(settlementDate, maturity, ql.Period(3, ql.Months), calendar,
        ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.Swap.Payer, 1_000_000, fxSch, 0.04,
        ql.Thirty360(ql.Thirty360.BondBasis), flSch, index, 0.0, index.dayCounter())
    swap.setPricingEngine(ql.DiscountingSwapEngine(discountTS))

    # --- AAD tape for replay (recorded once at base market) ---
    tape_irs = Tape()
    tape_irs.activate()

    deps_r = {(3, ql.Months): Real(0.0363)}
    fras_r = {(3, 6): Real(0.037125), (6, 9): Real(0.037125), (9, 12): Real(0.037125)}
    futs_r = {
        ql.Date(19, 12, 2001): Real(96.2875), ql.Date(20,  3, 2002): Real(96.7875),
        ql.Date(19,  6, 2002): Real(96.9875), ql.Date(18,  9, 2002): Real(96.6875),
        ql.Date(18, 12, 2002): Real(96.4875), ql.Date(19,  3, 2003): Real(96.3875),
        ql.Date(18,  6, 2003): Real(96.2875), ql.Date(17,  9, 2003): Real(96.0875),
    }
    swps_r = {
        (2,  ql.Years): Real(0.037125), (3,  ql.Years): Real(0.0398),
        (5,  ql.Years): Real(0.0443),   (10, ql.Years): Real(0.05165),
        (15, ql.Years): Real(0.055175),
    }
    irs_inputs = (list(deps_r.values()) + list(fras_r.values())
                  + list(futs_r.values()) + list(swps_r.values()))
    tape_irs.registerInputs(irs_inputs)
    tape_irs.newRecording()

    depH_r  = [ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(deps_r[(n, u)])),
                  ql.Period(n, u), 2, calendar, ql.ModifiedFollowing, False, dayCounter)
               for n, u in deps_r]
    fraH_r  = [ql.FraRateHelper(ql.QuoteHandle(ql.SimpleQuote(fras_r[(n, m)])),
                  n, m, 2, calendar, ql.ModifiedFollowing, False, dayCounter)
               for n, m in fras_r]
    futH_r  = [ql.FuturesRateHelper(ql.QuoteHandle(ql.SimpleQuote(futs_r[d])), d, 3,
                  calendar, ql.ModifiedFollowing, True, dayCounter,
                  ql.QuoteHandle(ql.SimpleQuote(0.0)))
               for d in futs_r]
    swpH_r  = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(swps_r[(n, u)])),
                  ql.Period(n, u), calendar, ql.Annual, ql.Unadjusted,
                  ql.Thirty360(ql.Thirty360.BondBasis), ql.Euribor3M(),
                  ql.QuoteHandle(), ql.Period("0D"), discountTS)
               for n, u in swps_r]
    fcastH_r = ql.RelinkableYieldTermStructureHandle()
    fcastH_r.linkTo(ql.PiecewiseFlatForward(
        settlementDate, depH_r + futH_r + swpH_r[1:], ql.Actual360()))
    index_r = ql.Euribor3M(fcastH_r)
    swap_r  = ql.VanillaSwap(ql.Swap.Payer, 1_000_000, fxSch, 0.04,
        ql.Thirty360(ql.Thirty360.BondBasis), flSch, index_r, 0.0,
        index_r.dayCounter())
    swap_r.setPricingEngine(ql.DiscountingSwapEngine(discountTS))
    npv_irs = swap_r.NPV()
    tape_irs.registerOutput(npv_irs)

    return swap, irs_quotes, tape_irs, npv_irs, irs_inputs


# ============================================================================
# Worker: run all IRS benchmarks, return JSON
# ============================================================================

def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    from xad.adj_1st import Real, Tape

    irs_scenarios = _gen_scenarios()
    results = {}

    # ============================================================ IRS MC ===

    (irs_swap, irs_quotes,
     tape_irs, npv_irs, irs_inputs) = _build_irs_mc_structures()

    # G1 — FD batch: 17-input bump-and-reprice for every scenario
    def _irs_fd_mc():
        for scene in irs_scenarios:
            for sq, v in zip(irs_quotes, scene):
                sq.setValue(v)
            irs_swap.NPV()
            for sq, v in zip(irs_quotes, scene):
                sq.setValue(v + BPS)
                irs_swap.NPV()
                sq.setValue(v)
        for sq, v in zip(irs_quotes, _IRS_BASE):
            sq.setValue(v)

    m, s = _median_ms(_irs_fd_mc, repeats)
    results["irs_fd_mc"] = {"median": m, "stdev": s,
                             "n_scenarios": N_SCENARIOS, "n_inputs": 17}

    # G2 — AAD replay: N_SCENARIOS backward sweeps on a fixed tape
    def _irs_aad_replay():
        for _ in range(N_SCENARIOS):
            tape_irs.clearDerivatives()
            npv_irs.derivative = 1.0
            tape_irs.computeAdjoints()

    m, s = _median_ms(_irs_aad_replay, repeats)
    results["irs_aad_replay"] = {"median": m, "stdev": s,
                                  "n_scenarios": N_SCENARIOS, "n_inputs": 17}
    tape_irs.deactivate()

    # G3 — AAD re-record: single persistent Tape, newRecording() per scenario.
    # A single Tape is kept active for the entire batch. SimpleQuote.setValue(Real)
    # pushes xad Reals into the pre-built QL structure so QuantLib's lazy observer
    # chain re-bootstraps the PiecewiseFlatForward and records the full computation
    # graph.  Quotes are restored to float values BEFORE tape.deactivate() to
    # avoid close_enough() on a detached AReal (null tape pointer → SIGSEGV).
    def _irs_aad_record():
        tape = Tape()
        tape.activate()
        for scene in irs_scenarios:
            reals = [Real(v) for v in scene]
            tape.registerInputs(reals)
            tape.newRecording()
            for sq, r in zip(irs_quotes, reals):
                sq.setValue(r)
            npv = irs_swap.NPV()
            tape.registerOutput(npv)
            npv.derivative = 1.0
            tape.computeAdjoints()
            # Restore quotes to float while tape is still active (safe)
            for sq, v in zip(irs_quotes, _IRS_BASE):
                sq.setValue(v)
        tape.deactivate()

    m, s = _median_ms(_irs_aad_record, repeats)
    results["irs_aad_record"] = {"median": m, "stdev": s,
                                  "n_scenarios": N_SCENARIOS, "n_inputs": 17}

    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison table
# ============================================================================

INSTRUMENTS = [
    # (id, label,         n_in_key, fd_key,       replay_key,       record_key)
    ("G", "Vanilla IRS",  "irs",   "irs_fd_mc",  "irs_aad_replay", "irs_aad_record"),
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
    print("QuantLib-Risks-Py  –  Monte Carlo Scenario Risk Benchmark: Vanilla IRS")
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

    jit_speedup_replay = []
    jit_speedup_record = []

    for _, label, prefix, fd_k, rpl_k, rec_k in INSTRUMENTS:
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
            if key == rpl_k:
                jit_speedup_replay.append(nj / jt if jt > 0 else None)
            if key == rec_k:
                jit_speedup_record.append(nj / jt if jt > 0 else None)

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
    print("  SUMMARY")
    print(SEPARATOR)
    print(f"  {'':30s}   {'Replay':>8}   {'Re-record':>10}  (JIT speedup)")
    for (_, label, _, _, _, _), spr, src in zip(
            INSTRUMENTS, jit_speedup_replay, jit_speedup_record):
        print(f"  {label:<30s}   {spr:>6.2f}×      {src:>6.2f}×")
    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "monte_carlo_irs_benchmarks_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: Vanilla IRS")
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
    w("## Results")
    w("")

    HIGH_CV = 0.5
    has_high_cv = False
    jit_speedup_replay = []
    jit_speedup_record = []

    for _, label, prefix, fd_k, rpl_k, rec_k in INSTRUMENTS:
        n_in   = nojit[fd_k]["n_inputs"]
        n_scen = nojit[fd_k]["n_scenarios"]

        w(f"### {label}  ({n_in} inputs, {n_scen} scenarios per batch)")
        w("")
        w("*FD detail: 18 complete curve bootstraps + swap valuations per scenario*")
        w("")
        w("| Method | Non-JIT batch (ms) | JIT batch (ms) | JIT speedup | Per-scenario NJ | Per-scenario JIT |")
        w("|---|---:|---:|---:|---:|---:|")

        for method, key in [
            ("FD (N+1 pricings per scenario)",         fd_k),
            ("**AAD replay** (backward sweep only)",   rpl_k),
            ("AAD re-record (forward + backward)",     rec_k),
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
            if key == rpl_k:
                jit_speedup_replay.append(nj / jt if jt > 0 else None)
            if key == rec_k:
                jit_speedup_record.append(nj / jt if jt > 0 else None)
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

    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("")
    w("python benchmarks/monte_carlo_irs_benchmarks.py            # default 5 repeats")
    w("python benchmarks/monte_carlo_irs_benchmarks.py --repeats 10")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks Monte Carlo IRS benchmark",
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
    print("QuantLib-Risks-Py  –  Monte Carlo Scenario Risk Benchmark: Vanilla IRS")
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

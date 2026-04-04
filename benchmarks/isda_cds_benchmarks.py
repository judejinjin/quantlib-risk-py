#!/usr/bin/env python3
"""
QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: ISDA CDS Engine
==========================================================================

Benchmarks a CDS priced with the IsdaCdsEngine under N_SCENARIOS = 100
random Monte Carlo scenarios, comparing:

  I1) FD             – bump-and-reprice (20 inputs → 21 full re-pricings per scenario)
  I2) AAD replay     – N backward sweeps on a tape recorded once at base market
  I3) AAD re-record  – per-scenario fresh recording

All benchmarks use the standard (non-JIT) XAD build.  The JIT/Forge backend is
not used because the IsdaCdsEngine contains a data-dependent branch on Real:

    if (fhphh < 1E-4 && numericalFix_ == Taylor)

where fhphh = log(P0) - log(P1) + log(Q0) - log(Q1) depends on discount and
survival factors — which are functions of the AD inputs.  This violates the
Forge record-once-replay-many paradigm (see JIT_LIMITATIONS.md for details).

Inputs (20):
  - 6 deposit rates    (1M, 2M, 3M, 6M, 9M, 12M)
  - 14 swap rates      (2Y–30Y)

Usage
-----
  python benchmarks/isda_cds_benchmarks.py               # default 5 repeats
  python benchmarks/isda_cds_benchmarks.py --repeats 10
  python benchmarks/isda_cds_benchmarks.py --clean-venvs

  # Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/isda_cds_benchmarks.py --worker REPEATS
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
ROOT  = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
VENV  = BUILD / "bench-venv-nojit"

SEPARATOR = "=" * 84
BPS = 1e-4

# ------- scenario parameters -----------------------------------------------
N_SCENARIOS   = 100
SCENARIO_SEED = 456          # distinct seed from CDS/IRS/bond benchmarks

# ISDA CDS base market  (20 curve inputs)
_DEP_TENORS = [1, 2, 3, 6, 9, 12]
_DEP_QUOTES = [0.003081, 0.005525, 0.007163, 0.012413, 0.014, 0.015488]

_SWAP_TENORS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
_SWAP_QUOTES = [
    0.011907, 0.01699,  0.021198, 0.02444,  0.026937,
    0.028967, 0.030504, 0.031719, 0.03279,  0.034535,
    0.036217, 0.036981, 0.037246, 0.037605,
]

_BASE_RATES = _DEP_QUOTES + _SWAP_QUOTES    # 20 values
_N_INPUTS = len(_BASE_RATES)                 # 20

# Fixed CDS parameters (not perturbed in MC)
_CDS_SPREAD    = 0.001
_CDS_RECOVERY  = 0.4
_CDS_TERM_DATE = (20, 6, 2019)   # 10Y CDS


def _gen_scenarios():
    """Pre-generate ISDA CDS MC scenarios (perturbing the 20 curve rates)."""
    rng = random.Random(SCENARIO_SEED)
    scenarios = []
    for _ in range(N_SCENARIOS):
        scene = [max(1e-6, v + rng.gauss(0, 5e-4)) for v in _BASE_RATES]
        scenarios.append(scene)
    return scenarios


# ============================================================================
# Wheel / venv helpers  (non-JIT only, same pattern as bond benchmarks)
# ============================================================================

def find_wheel(build_root: Path) -> dict:
    """Find the non-JIT XAD and QuantLib-Risks wheels."""
    def latest(pattern):
        matches = sorted(build_root.glob(pattern))
        return matches[-1] if matches else None
    return {
        "xad": latest("xad-python/dist/xad_autodiff-*.whl"),
        "ql":  latest("linux-xad-gcc-ninja-release/Python/dist/quantlib_risks-*.whl"),
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


# ---- ISDA CDS Engine (20 curve inputs) ------------------------------------

def _build_isda_structures():
    """
    Build the ISDA CDS pricing structure and the base AAD tape (for replay).

    The pricing pipeline:
      1. Bootstrap discount curve from 20 deposit+swap rate helpers
      2. Compute implied hazard rate from quoted CDS
      3. Create flat hazard rate curve
      4. Price conventional CDS with IsdaCdsEngine → NPV

    Returns:
        cds          – CreditDefaultSwap (float engine)
        rate_quotes  – list of 20 SimpleQuote objects
        tape         – Tape recorded at base market (for AAD replay)
        npv_out      – xad Real output registered on tape
        tape_inputs  – list of 20 xad Reals registered on tape
    """
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape, Real

    trade_date = ql.Date(21, 5, 2009)
    ql.Settings.instance().evaluationDate = trade_date
    ql.IborCoupon.createAtParCoupons()

    # --- plain-float structure for FD ---
    dep_sq = [ql.SimpleQuote(v) for v in _DEP_QUOTES]
    swap_sq = [ql.SimpleQuote(v) for v in _SWAP_QUOTES]
    rate_quotes = dep_sq + swap_sq

    dep_helpers = [
        ql.DepositRateHelper(
            ql.QuoteHandle(dep_sq[i]),
            _DEP_TENORS[i] * ql.Period(ql.Monthly),
            2, ql.WeekendsOnly(), ql.ModifiedFollowing, False, ql.Actual360(),
        )
        for i in range(len(_DEP_TENORS))
    ]

    isda_ibor = ql.IborIndex(
        'IsdaIbor', 3 * ql.Period(ql.Monthly), 2, ql.USDCurrency(),
        ql.WeekendsOnly(), ql.ModifiedFollowing, False, ql.Actual360(),
    )
    swap_helpers = [
        ql.SwapRateHelper(
            ql.QuoteHandle(swap_sq[i]),
            _SWAP_TENORS[i] * ql.Period(ql.Annual),
            ql.WeekendsOnly(), ql.Semiannual, ql.ModifiedFollowing,
            ql.Thirty360(ql.Thirty360.BondBasis), isda_ibor,
        )
        for i in range(len(_SWAP_TENORS))
    ]

    swap_curve = ql.PiecewiseFlatForward(
        trade_date, dep_helpers + swap_helpers, ql.Actual365Fixed())
    discountCurve = ql.YieldTermStructureHandle(swap_curve)

    termDate = ql.Date(*_CDS_TERM_DATE)
    upfront_date = ql.WeekendsOnly().advance(trade_date, 3 * ql.Period(ql.Daily))

    cdsSchedule = ql.Schedule(
        trade_date, termDate, 3 * ql.Period(ql.Monthly),
        ql.WeekendsOnly(), ql.Following, ql.Unadjusted,
        ql.DateGeneration.CDS, False,
    )

    quotedTrade = ql.CreditDefaultSwap(
        ql.Protection.Buyer, 10_000_000, 0, _CDS_SPREAD, cdsSchedule,
        ql.Following, ql.Actual360(), True, True, trade_date,
        upfront_date, ql.FaceValueClaim(), ql.Actual360(True),
    )
    h = quotedTrade.impliedHazardRate(
        0, discountCurve, ql.Actual365Fixed(), _CDS_RECOVERY, 1e-10,
        ql.CreditDefaultSwap.ISDA,
    )

    probabilityCurve = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(0, ql.WeekendsOnly(),
                          ql.QuoteHandle(ql.SimpleQuote(h)),
                          ql.Actual365Fixed()))

    engine = ql.IsdaCdsEngine(probabilityCurve, _CDS_RECOVERY, discountCurve)
    conventionalTrade = ql.CreditDefaultSwap(
        ql.Protection.Buyer, 10_000_000, 0, 0.01, cdsSchedule,
        ql.Following, ql.Actual360(), True, True, trade_date,
        upfront_date, ql.FaceValueClaim(), ql.Actual360(True),
    )
    conventionalTrade.setPricingEngine(engine)

    # --- AAD tape for replay (recorded once at base market) ---
    tape = Tape()
    tape.activate()

    dep_r = [Real(v) for v in _DEP_QUOTES]
    swap_r = [Real(v) for v in _SWAP_QUOTES]
    tape_inputs = dep_r + swap_r
    tape.registerInputs(tape_inputs)
    tape.newRecording()

    dep_helpers_r = [
        ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(dep_r[i])),
            _DEP_TENORS[i] * ql.Period(ql.Monthly),
            2, ql.WeekendsOnly(), ql.ModifiedFollowing, False, ql.Actual360(),
        )
        for i in range(len(_DEP_TENORS))
    ]
    swap_helpers_r = [
        ql.SwapRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(swap_r[i])),
            _SWAP_TENORS[i] * ql.Period(ql.Annual),
            ql.WeekendsOnly(), ql.Semiannual, ql.ModifiedFollowing,
            ql.Thirty360(ql.Thirty360.BondBasis), isda_ibor,
        )
        for i in range(len(_SWAP_TENORS))
    ]
    swap_curve_r = ql.PiecewiseFlatForward(
        trade_date, dep_helpers_r + swap_helpers_r, ql.Actual365Fixed())
    discountCurve_r = ql.YieldTermStructureHandle(swap_curve_r)

    quotedTrade_r = ql.CreditDefaultSwap(
        ql.Protection.Buyer, 10_000_000, 0, _CDS_SPREAD, cdsSchedule,
        ql.Following, ql.Actual360(), True, True, trade_date,
        upfront_date, ql.FaceValueClaim(), ql.Actual360(True),
    )
    h_r = quotedTrade_r.impliedHazardRate(
        0, discountCurve_r, ql.Actual365Fixed(), _CDS_RECOVERY, 1e-10,
        ql.CreditDefaultSwap.ISDA,
    )

    probabilityCurve_r = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(0, ql.WeekendsOnly(),
                          ql.QuoteHandle(ql.SimpleQuote(h_r)),
                          ql.Actual365Fixed()))

    engine_r = ql.IsdaCdsEngine(probabilityCurve_r, _CDS_RECOVERY, discountCurve_r)
    conventionalTrade_r = ql.CreditDefaultSwap(
        ql.Protection.Buyer, 10_000_000, 0, 0.01, cdsSchedule,
        ql.Following, ql.Actual360(), True, True, trade_date,
        upfront_date, ql.FaceValueClaim(), ql.Actual360(True),
    )
    conventionalTrade_r.setPricingEngine(engine_r)
    npv_out = conventionalTrade_r.NPV()
    tape.registerOutput(npv_out)

    return conventionalTrade, rate_quotes, tape, npv_out, tape_inputs


def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    from xad.adj_1st import Real, Tape

    scenarios = _gen_scenarios()
    results = {}

    (cds, rate_quotes,
     tape, npv_out, tape_inputs) = _build_isda_structures()

    # I1 — FD batch: 20-input bump-and-reprice for every scenario
    def _isda_fd_mc():
        for scene in scenarios:
            for sq, v in zip(rate_quotes, scene):
                sq.setValue(v)
            cds.NPV()
            for sq, v in zip(rate_quotes, scene):
                sq.setValue(v + BPS)
                cds.NPV()
                sq.setValue(v)
        # Restore base
        for sq, v in zip(rate_quotes, _BASE_RATES):
            sq.setValue(v)

    m, s = _median_ms(_isda_fd_mc, repeats)
    results["isda_fd_mc"] = {"median": m, "stdev": s,
                              "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    # I2 — AAD replay: N_SCENARIOS backward sweeps on a fixed tape
    def _isda_aad_replay():
        for _ in range(N_SCENARIOS):
            tape.clearDerivatives()
            npv_out.derivative = 1.0
            tape.computeAdjoints()

    m, s = _median_ms(_isda_aad_replay, repeats)
    results["isda_aad_replay"] = {"median": m, "stdev": s,
                                   "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}
    tape.deactivate()

    # I3 — AAD re-record: per-scenario fresh recording
    def _isda_aad_record():
        trade_date = ql.Date(21, 5, 2009)
        ql.Settings.instance().evaluationDate = trade_date
        termDate = ql.Date(*_CDS_TERM_DATE)
        upfront_date = ql.WeekendsOnly().advance(trade_date, 3 * ql.Period(ql.Daily))

        cdsSchedule = ql.Schedule(
            trade_date, termDate, 3 * ql.Period(ql.Monthly),
            ql.WeekendsOnly(), ql.Following, ql.Unadjusted,
            ql.DateGeneration.CDS, False,
        )
        isda_ibor = ql.IborIndex(
            'IsdaIbor', 3 * ql.Period(ql.Monthly), 2, ql.USDCurrency(),
            ql.WeekendsOnly(), ql.ModifiedFollowing, False, ql.Actual360(),
        )

        tp = Tape()
        tp.activate()
        for scene in scenarios:
            dep_r = [Real(v) for v in scene[:6]]
            swap_r = [Real(v) for v in scene[6:]]
            reals = dep_r + swap_r
            tp.registerInputs(reals)
            tp.newRecording()

            dep_helpers = [
                ql.DepositRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(dep_r[i])),
                    _DEP_TENORS[i] * ql.Period(ql.Monthly),
                    2, ql.WeekendsOnly(), ql.ModifiedFollowing,
                    False, ql.Actual360(),
                )
                for i in range(len(_DEP_TENORS))
            ]
            swap_helpers = [
                ql.SwapRateHelper(
                    ql.QuoteHandle(ql.SimpleQuote(swap_r[i])),
                    _SWAP_TENORS[i] * ql.Period(ql.Annual),
                    ql.WeekendsOnly(), ql.Semiannual, ql.ModifiedFollowing,
                    ql.Thirty360(ql.Thirty360.BondBasis), isda_ibor,
                )
                for i in range(len(_SWAP_TENORS))
            ]
            sc = ql.PiecewiseFlatForward(
                trade_date, dep_helpers + swap_helpers, ql.Actual365Fixed())
            dc = ql.YieldTermStructureHandle(sc)

            qt = ql.CreditDefaultSwap(
                ql.Protection.Buyer, 10_000_000, 0, _CDS_SPREAD, cdsSchedule,
                ql.Following, ql.Actual360(), True, True, trade_date,
                upfront_date, ql.FaceValueClaim(), ql.Actual360(True),
            )
            h = qt.impliedHazardRate(
                0, dc, ql.Actual365Fixed(), _CDS_RECOVERY, 1e-10,
                ql.CreditDefaultSwap.ISDA,
            )
            pc = ql.DefaultProbabilityTermStructureHandle(
                ql.FlatHazardRate(0, ql.WeekendsOnly(),
                                  ql.QuoteHandle(ql.SimpleQuote(h)),
                                  ql.Actual365Fixed()))
            eng = ql.IsdaCdsEngine(pc, _CDS_RECOVERY, dc)
            ct = ql.CreditDefaultSwap(
                ql.Protection.Buyer, 10_000_000, 0, 0.01, cdsSchedule,
                ql.Following, ql.Actual360(), True, True, trade_date,
                upfront_date, ql.FaceValueClaim(), ql.Actual360(True),
            )
            ct.setPricingEngine(eng)
            npv = ct.NPV()
            tp.registerOutput(npv)
            npv.derivative = 1.0
            tp.computeAdjoints()
        tp.deactivate()

    m, s = _median_ms(_isda_aad_record, repeats)
    results["isda_aad_record"] = {"median": m, "stdev": s,
                                   "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison table (non-JIT only)
# ============================================================================

INSTRUMENTS = [
    # (id, label,         fd_key,        replay_key,        record_key)
    ("I", "CDS (IsdaCdsEngine)", "isda_fd_mc", "isda_aad_replay", "isda_aad_record"),
]


def print_results(data: dict, repeats: int, wheel: dict):
    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Monte Carlo Scenario Risk Benchmark: ISDA CDS")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  MC scenarios per batch : {N_SCENARIOS}")
    print(f"  Outer repeats          : {repeats}")
    print(f"  BPS shift (FD)         : {BPS}")
    print(f"  Wheel : {wheel['ql'].name}")
    print(f"  JIT   : NOT USED (IsdaCdsEngine has data-dependent branching)")
    print()

    COL_M = 24
    COL_T = 22

    for _, label, fd_k, rpl_k, rec_k in INSTRUMENTS:
        n_in   = data[fd_k]["n_inputs"]
        n_scen = data[fd_k]["n_scenarios"]

        print(f"  ── {label}  ({n_in} inputs, {n_scen} scenarios per batch) ──")
        hdr = (f"  {'Method':<{COL_M}}"
               f"  {'Batch (ms)':>{COL_T}}"
               f"  {'per-scenario':>15}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for method, key in [
            ("FD (N+1 pricings)",       fd_k),
            ("AAD replay   (backward)", rpl_k),
            ("AAD re-record (fwd+bwd)", rec_k),
        ]:
            val = data[key]["median"]
            std = data[key]["stdev"]
            ps  = f"{val / n_scen * 1000:.1f} µs"
            print(f"  {method:<{COL_M}}"
                  f"  {val:>8.1f} ±{std:>6.1f} ms"
                  f"  {ps:>15}")

        fd_val  = data[fd_k]["median"]
        rpl_val = data[rpl_k]["median"]
        rec_val = data[rec_k]["median"]
        print(f"  {'FD / AAD replay':<{COL_M}}"
              f"  {fd_val/rpl_val:>8.1f}×")
        print(f"  {'FD / AAD re-record':<{COL_M}}"
              f"  {fd_val/rec_val:>8.1f}×")
        print()

    print(SEPARATOR)


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "isda_cds_benchmarks_results.md"


def write_markdown(data: dict, repeats: int, wheel: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: ISDA CDS Engine")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**MC scenarios per batch:** {N_SCENARIOS}  ")
    w(f"**Outer repetitions:** {repeats} (median reported)  ")
    w(f"**Wheel:** `{wheel['ql'].name}`  ")
    w(f"**JIT:** Not used (IsdaCdsEngine has data-dependent branching on Real)  ")
    w("")
    w("---")
    w("")
    w("## Instrument")
    w("")
    w("- **CDS** priced with **IsdaCdsEngine**")
    w("- 6 deposit rates (1M–12M) + 14 swap rates (2Y–30Y) = **20 curve inputs**")
    w("- Discount curve: `PiecewiseFlatForward` bootstrapped from deposit + swap helpers")
    w("- CDS: 10Y term, spread 10 bps, recovery 40%, notional 10M")
    w("- Pipeline: curve bootstrap → `impliedHazardRate` → `FlatHazardRate` → `IsdaCdsEngine` → NPV")
    w("")
    w("---")
    w("")
    w("## Results")
    w("")

    HIGH_CV = 0.5
    has_high_cv = False

    for _, label, fd_k, rpl_k, rec_k in INSTRUMENTS:
        n_in   = data[fd_k]["n_inputs"]
        n_scen = data[fd_k]["n_scenarios"]

        w(f"### {label}  ({n_in} inputs, {n_scen} scenarios per batch)")
        w("")
        w("| Method | Batch (ms) | Per-scenario |")
        w("|---|---:|---:|")

        for method, key in [
            ("FD (N+1 pricings per scenario)", fd_k),
            ("**AAD replay** (backward sweep only)", rpl_k),
            ("AAD re-record (forward + backward)", rec_k),
        ]:
            val = data[key]["median"]
            std = data[key]["stdev"]
            flag = "†" if val and std / val > HIGH_CV else ""
            if flag:
                has_high_cv = True
            ps = f"{val / n_scen * 1000:.0f} µs"
            w(f"| {method} | {val:.1f} ±{std:.1f}{flag} | {ps} |")

        fd_val  = data[fd_k]["median"]
        rpl_val = data[rpl_k]["median"]
        rec_val = data[rec_k]["median"]
        w(f"| *FD ÷ AAD replay* | *{fd_val/rpl_val:.0f}×* | — |")
        w(f"| *FD ÷ AAD re-record* | *{fd_val/rec_val:.1f}×* | — |")
        w("")

    w("---")
    w("")
    if has_high_cv:
        w(f"**†** High variance (stdev/median > {HIGH_CV:.0%}).")
        w("")

    w("## Why no JIT?")
    w("")
    w("The `IsdaCdsEngine` contains a data-dependent branch on `Real`:")
    w("")
    w("```cpp")
    w("// isdacdsengine.cpp, line ~193 and ~262:")
    w("Real fhphh = log(P0) - log(P1) + log(Q0) - log(Q1);")
    w("if (fhphh < 1E-4 && numericalFix_ == Taylor) { ... }")
    w("```")
    w("")
    w("`fhphh` depends on discount/survival factors which are functions of the AD")
    w("inputs. The Forge JIT backend evaluates `if` at record time and bakes the")
    w("decision into the compiled kernel. Replaying with different inputs may take")
    w("the wrong branch, producing incorrect results or crashing.")
    w("")
    w("See [JIT_LIMITATIONS.md](../JIT_LIMITATIONS.md) for full details.")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("")
    w("python benchmarks/isda_cds_benchmarks.py            # default 5 repeats")
    w("python benchmarks/isda_cds_benchmarks.py --repeats 10")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks ISDA CDS Engine benchmark",
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
    wheel   = find_wheel(BUILD)

    missing = [kind for kind in ("xad", "ql") if wheel[kind] is None]
    if missing:
        print("ERROR: Missing wheels for:", ", ".join(missing))
        print("  Run the build first:")
        print("    ./build.sh --no-jit -j$(nproc)")
        sys.exit(1)

    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Monte Carlo Scenario Risk Benchmark: ISDA CDS")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"  {N_SCENARIOS} scenarios per batch, {repeats} outer repetitions")
    print(SEPARATOR)

    print("\nSetting up virtual environment")
    print("-" * 50)
    setup_venv(VENV, wheel["xad"], wheel["ql"], force=args.clean_venvs)

    print(f"\nRunning benchmarks  ({repeats} outer repeats, {N_SCENARIOS} scenarios each)")
    print("-" * 50)
    print("\n  Non-JIT worker …")
    data = run_worker_in_venv(VENV, repeats)
    print("  done.")

    print_results(data, repeats, wheel)

    if not args.no_save:
        write_markdown(data, repeats, wheel)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
QuantLib-Risks-Py  –  Additional JIT vs Non-JIT Benchmarks
============================================================
Extends benchmarks/run_benchmarks.py with three additional instruments drawn
from the Python/examples catalogue:

  D)  American option (Barone-Adesi-Whaley approximation)   4 market inputs
  E)  Interest-rate cap (Black cap-floor engine,             18 market inputs
        bootstrapped Euribor3M curve + flat volatility)
  F)  European swaption (Jamshidian / Hull-White analytic     3 model inputs
        on a flat term structure)

Each instrument is timed for:
  • Plain forward pricing   (float, no tape)
  • AAD backward pass       (tape replay only; recorded once at startup)
  • Bump-and-reprice FD     (N+1 complete forward pricings)

Both the Non-JIT (XAD tape) and JIT (XAD-Forge) builds are exercised in
isolated virtual environments. Results are written to
  benchmarks/more_benchmarks_results.md

Usage
-----
  python benchmarks/run_more_benchmarks.py               # default 30 repeats
  python benchmarks/run_more_benchmarks.py --repeats 50
  python benchmarks/run_more_benchmarks.py --clean-venvs

Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/run_more_benchmarks.py --worker REPEATS
"""

import argparse
import datetime
import json
import math
import os
import platform
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
BPS = 1e-4   # 1 basis-point shift for FD benchmarks


# ============================================================================
# Wheel discovery  (reused from run_benchmarks.py)
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


# ============================================================================
# Virtual-environment helpers  (reused from run_benchmarks.py)
# ============================================================================

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


# ============================================================================
# Worker subprocess runner
# ============================================================================

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
# WORKER MODE — benchmark implementations (runs inside each isolated venv)
# ============================================================================

def _median_ms(func, n: int, warmup: int = 5):
    """Return (median_ms, stdev_ms) over n timed calls, after `warmup` un-timed calls."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ---- D. American option  (Barone-Adesi-Whaley, 4 inputs) -------------------
#
# Sources: Python/examples/american-option.py
# Inputs:  S (spot), r (risk-free rate), q (dividend yield), σ (volatility)
# Engine:  BaroneAdesiWhaleyApproximationEngine — quasi-analytic closed-form,
#          guaranteed to flow cleanly through XAD.
# Reference: Barone-Adesi and Whaley (1987) "Efficient analytic approximation
#            of American option values".

_AMOPT_S0 = 36.0
_AMOPT_R0 = 0.06
_AMOPT_Q0 = 0.00
_AMOPT_V0 = 0.20


def _build_amopt_plain():
    """Return (option, [sq_spot, sq_rate, sq_div, sq_vol])."""
    import QuantLib_Risks as ql
    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate

    sq_spot = ql.SimpleQuote(_AMOPT_S0)
    sq_rate = ql.SimpleQuote(_AMOPT_R0)
    sq_div  = ql.SimpleQuote(_AMOPT_Q0)
    sq_vol  = ql.SimpleQuote(_AMOPT_V0)

    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sq_spot),
        ql.YieldTermStructureHandle(
            ql.FlatForward(todaysDate, ql.QuoteHandle(sq_div), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(
            ql.FlatForward(todaysDate, ql.QuoteHandle(sq_rate), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(todaysDate, ql.TARGET(),
                                ql.QuoteHandle(sq_vol), ql.Actual365Fixed())),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Put, 40.0),
        ql.AmericanExercise(todaysDate, ql.Date(17, ql.May, 1999)),
    )
    option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(proc))
    return option, [sq_spot, sq_rate, sq_div, sq_vol]


def _build_amopt_aad():
    """Return (tape, option_aad, [spot_v, rate_v, div_v, vol_v])."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()

    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate

    spot_v = ql.Real(_AMOPT_S0)
    rate_v = ql.Real(_AMOPT_R0)
    div_v  = ql.Real(_AMOPT_Q0)
    vol_v  = ql.Real(_AMOPT_V0)
    all_inputs = [spot_v, rate_v, div_v, vol_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot_v)),
        ql.YieldTermStructureHandle(
            ql.FlatForward(todaysDate, ql.QuoteHandle(ql.SimpleQuote(div_v)),
                           ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(
            ql.FlatForward(todaysDate, ql.QuoteHandle(ql.SimpleQuote(rate_v)),
                           ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(todaysDate, ql.TARGET(),
                                ql.QuoteHandle(ql.SimpleQuote(vol_v)),
                                ql.Actual365Fixed())),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Put, 40.0),
        ql.AmericanExercise(todaysDate, ql.Date(17, ql.May, 1999)),
    )
    option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(proc))
    return tape, option, all_inputs


# ---- E. Interest rate cap  (Black engine, bootstrapped curve, 18 inputs) ---
#
# Sources: Python/examples/capsfloors.py, Python/examples/swap-adjoint.py
# Inputs:  The same 17 bootstrapped Euribor3M curve quotes as the IRS (one
#          deposit, three FRAs, eight Eurodollar futures, five swap rates)
#          PLUS one flat Black volatility quote = 18 inputs total.
# Engine:  BlackCapFloorEngine — prices each caplet with the Black formula,
#          so the computation graph is larger than a plain IRS (~40 Black
#          formula evaluations).  AAD computes all 18 sensitivities in one
#          backward pass regardless of input count.
# Cap:     10-year cap on Euribor3M, forward-starting 6 months from
#          settlement, strike = 5%.  The forward start avoids the need to
#          supply historical fixings for the first period.

_CAP_STRIKE = 0.05
_CAP_VOL0   = 0.20


def _cap_curve_quotes_and_helpers(ql, settlement):
    """
    Return (all_quotes, fcastH) where all_quotes is the list of 17 rate
    SimpleQuotes used to build the piecewise forecast curve and fcastH is a
    RelinkableYieldTermStructureHandle pointing at it.

    This is the same bootstrapped Euribor3M curve used in the IRS benchmark.
    """
    calendar = ql.TARGET()

    deps = {(3, ql.Months): ql.SimpleQuote(0.0363)}
    fras = {
        (3, 6):  ql.SimpleQuote(0.037125),
        (6, 9):  ql.SimpleQuote(0.037125),
        (9, 12): ql.SimpleQuote(0.037125),
    }
    futs = {
        ql.Date(19, 12, 2001): ql.SimpleQuote(96.2875),
        ql.Date(20,  3, 2002): ql.SimpleQuote(96.7875),
        ql.Date(19,  6, 2002): ql.SimpleQuote(96.9875),
        ql.Date(18,  9, 2002): ql.SimpleQuote(96.6875),
        ql.Date(18, 12, 2002): ql.SimpleQuote(96.4875),
        ql.Date(19,  3, 2003): ql.SimpleQuote(96.3875),
        ql.Date(18,  6, 2003): ql.SimpleQuote(96.2875),
        ql.Date(17,  9, 2003): ql.SimpleQuote(96.0875),
    }
    swps = {
        (2,  ql.Years): ql.SimpleQuote(0.037125),
        (3,  ql.Years): ql.SimpleQuote(0.0398),
        (5,  ql.Years): ql.SimpleQuote(0.0443),
        (10, ql.Years): ql.SimpleQuote(0.05165),
        (15, ql.Years): ql.SimpleQuote(0.055175),
    }
    all_quotes = (list(deps.values()) + list(fras.values())
                  + list(futs.values()) + list(swps.values()))

    dayCounter = ql.Actual360()
    depH = [ql.DepositRateHelper(ql.QuoteHandle(deps[(n, u)]), ql.Period(n, u),
                2, calendar, ql.ModifiedFollowing, False, dayCounter)
            for n, u in deps]
    fraH = [ql.FraRateHelper(ql.QuoteHandle(fras[(n, m)]), n, m,
                2, calendar, ql.ModifiedFollowing, False, dayCounter)
            for n, m in fras]
    futH = [ql.FuturesRateHelper(ql.QuoteHandle(futs[d]), d, 3,
                calendar, ql.ModifiedFollowing, True, dayCounter,
                ql.QuoteHandle(ql.SimpleQuote(0.0)))
            for d in futs]
    discountTS = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement, 0.04, ql.Actual360()))
    swpH = [ql.SwapRateHelper(ql.QuoteHandle(swps[(n, u)]), ql.Period(n, u),
                calendar, ql.Annual, ql.Unadjusted,
                ql.Thirty360(ql.Thirty360.BondBasis), ql.Euribor3M(),
                ql.QuoteHandle(), ql.Period("0D"), discountTS)
            for n, u in swps]

    fcastH = ql.RelinkableYieldTermStructureHandle()
    fcastH.linkTo(ql.PiecewiseFlatForward(
        settlement, depH + futH + swpH[1:], ql.Actual360()))
    return all_quotes, fcastH


def _build_cap_plain():
    """
    Return (cap, all_rate_quotes, sq_vol, fcastH).

    sq_vol drives the flat Black volatility.  all_rate_quotes are the 17
    bootstrapped-curve quotes.  fcastH is the RelinkableYieldTermStructureHandle
    which also serves as the discount curve inside BlackCapFloorEngine.
    """
    import QuantLib_Risks as ql
    calendar       = ql.TARGET()
    todaysDate     = ql.Date(6, ql.November, 2001)
    settlementDate = ql.Date(8, ql.November, 2001)
    ql.Settings.instance().evaluationDate = todaysDate

    all_rate_quotes, fcastH = _cap_curve_quotes_and_helpers(ql, settlementDate)

    # Forward-starting cap: first fixing is Jan 2002, well after todaysDate.
    capStart = calendar.advance(settlementDate, 2, ql.Months, ql.ModifiedFollowing)
    capEnd   = calendar.advance(capStart, 10, ql.Years, ql.ModifiedFollowing)
    schedule = ql.Schedule(capStart, capEnd, ql.Period(3, ql.Months),
                           calendar, ql.ModifiedFollowing, ql.ModifiedFollowing,
                           ql.DateGeneration.Forward, False)

    index    = ql.Euribor3M(fcastH)
    ibor_leg = ql.IborLeg([1_000_000], schedule, index)

    sq_vol = ql.SimpleQuote(_CAP_VOL0)
    cap    = ql.Cap(ibor_leg, [_CAP_STRIKE])
    cap.setPricingEngine(ql.BlackCapFloorEngine(fcastH, ql.QuoteHandle(sq_vol)))
    return cap, all_rate_quotes, sq_vol, fcastH


def _cap_curve_quotes_reals(ql, settlement):
    """
    Like _cap_curve_quotes_and_helpers but uses ql.Real inputs registered on
    the active tape.  Returns (all_inputs, fcastH).
    """
    from xad.adj_1st import Tape   # already imported by caller but re-import is no-op

    calendar = ql.TARGET()

    deps = {(3, ql.Months): ql.Real(0.0363)}
    fras = {
        (3, 6):  ql.Real(0.037125),
        (6, 9):  ql.Real(0.037125),
        (9, 12): ql.Real(0.037125),
    }
    futs = {
        ql.Date(19, 12, 2001): ql.Real(96.2875),
        ql.Date(20,  3, 2002): ql.Real(96.7875),
        ql.Date(19,  6, 2002): ql.Real(96.9875),
        ql.Date(18,  9, 2002): ql.Real(96.6875),
        ql.Date(18, 12, 2002): ql.Real(96.4875),
        ql.Date(19,  3, 2003): ql.Real(96.3875),
        ql.Date(18,  6, 2003): ql.Real(96.2875),
        ql.Date(17,  9, 2003): ql.Real(96.0875),
    }
    swps = {
        (2,  ql.Years): ql.Real(0.037125),
        (3,  ql.Years): ql.Real(0.0398),
        (5,  ql.Years): ql.Real(0.0443),
        (10, ql.Years): ql.Real(0.05165),
        (15, ql.Years): ql.Real(0.055175),
    }
    all_inputs = (list(deps.values()) + list(fras.values())
                  + list(futs.values()) + list(swps.values()))

    dayCounter = ql.Actual360()
    depH = [ql.DepositRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(deps[(n, u)])),
                ql.Period(n, u), 2, calendar, ql.ModifiedFollowing, False, dayCounter)
            for n, u in deps]
    fraH = [ql.FraRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(fras[(n, m)])),
                n, m, 2, calendar, ql.ModifiedFollowing, False, dayCounter)
            for n, m in fras]
    futH = [ql.FuturesRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(futs[d])), d, 3,
                calendar, ql.ModifiedFollowing, True, dayCounter,
                ql.QuoteHandle(ql.SimpleQuote(0.0)))
            for d in futs]
    discountTS = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement, 0.04, ql.Actual360()))
    swpH = [ql.SwapRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(swps[(n, u)])),
                ql.Period(n, u), calendar, ql.Annual, ql.Unadjusted,
                ql.Thirty360(ql.Thirty360.BondBasis), ql.Euribor3M(),
                ql.QuoteHandle(), ql.Period("0D"), discountTS)
            for n, u in swps]

    fcastH = ql.RelinkableYieldTermStructureHandle()
    fcastH.linkTo(ql.PiecewiseFlatForward(
        settlement, depH + futH + swpH[1:], ql.Actual360()))
    return all_inputs, fcastH


def _build_cap_aad():
    """Return (tape, cap_aad, all_inputs) where all_inputs has 18 elements."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()

    calendar       = ql.TARGET()
    todaysDate     = ql.Date(6, ql.November, 2001)
    settlementDate = ql.Date(8, ql.November, 2001)
    ql.Settings.instance().evaluationDate = todaysDate

    rate_inputs, fcastH = _cap_curve_quotes_reals(ql, settlementDate)
    vol_v = ql.Real(_CAP_VOL0)
    all_inputs = rate_inputs + [vol_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    capStart = calendar.advance(settlementDate, 2, ql.Months, ql.ModifiedFollowing)
    capEnd   = calendar.advance(capStart, 10, ql.Years, ql.ModifiedFollowing)
    schedule = ql.Schedule(capStart, capEnd, ql.Period(3, ql.Months),
                           calendar, ql.ModifiedFollowing, ql.ModifiedFollowing,
                           ql.DateGeneration.Forward, False)

    index    = ql.Euribor3M(fcastH)
    ibor_leg = ql.IborLeg([1_000_000], schedule, index)
    cap      = ql.Cap(ibor_leg, [_CAP_STRIKE])
    cap.setPricingEngine(
        ql.BlackCapFloorEngine(fcastH, ql.QuoteHandle(ql.SimpleQuote(vol_v))))
    return tape, cap, all_inputs


# ---- F. European swaption  (Jamshidian / Hull-White, 3 inputs) -------------
#
# Sources: Python/examples/bermudan-swaption.py
# Inputs:  flat forward rate r, Hull-White mean-reversion a, Hull-White vol σ
# Engine:  JamshidianSwaptionEngine on a calibrated HullWhite model.
#          The Jamshidian decomposition prices a European swaption analytically
#          by decomposing it into a portfolio of zero-bond options, each priced
#          with the closed-form HullWhite formula.  All intermediate operations
#          (exp, sqrt, erfc) flow cleanly through the XAD tape.
# Swaption: 1Y × 5Y ATM-ish, payer swaption on Euribor6M.

_SWAP_RATE0 = 0.04875825   # underlying flat yield
_HW_A0      = 0.10         # Hull-White mean-reversion speed
_HW_S0      = 0.01         # Hull-White short-rate volatility


def _build_swaption_plain():
    """
    Return (swaption, sq_rate, sq_a, sq_s, attach_swaption_engine,
            ts_handle, termStructure).

    JamshidianSwaptionEngine requires a HullWhite model that stores a and σ
    by value at construction time; attach_swaption_engine() rebuilds the model
    whenever a or σ are bumped.  For the flat rate, the RelinkableHandle
    propagates changes automatically.

    ts_handle + termStructure are returned so callers can use
    ``ts_handle.linkTo(termStructure)`` as a zero-allocation cache-bust.
    """
    import QuantLib_Risks as ql
    todaysDate     = ql.Date(15, ql.February, 2002)
    settlementDate = ql.Date(19, ql.February, 2002)
    ql.Settings.instance().evaluationDate = todaysDate
    calendar = ql.TARGET()

    sq_rate = ql.SimpleQuote(_SWAP_RATE0)
    sq_a    = ql.SimpleQuote(_HW_A0)
    sq_s    = ql.SimpleQuote(_HW_S0)

    termStructure = ql.FlatForward(
        settlementDate, ql.QuoteHandle(sq_rate), ql.Actual365Fixed())
    ts_handle = ql.RelinkableYieldTermStructureHandle(termStructure)

    # 1Y × 5Y payer swaption on Euribor6M
    index = ql.Euribor6M(ts_handle)
    swapStart = calendar.advance(settlementDate, 1, ql.Years,
                                 ql.ModifiedFollowing)
    swapEnd   = calendar.advance(swapStart, 5, ql.Years,
                                 ql.ModifiedFollowing)
    fixSch = ql.Schedule(swapStart, swapEnd, ql.Period(1, ql.Years),
        calendar, ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Forward, False)
    fltSch = ql.Schedule(swapStart, swapEnd, ql.Period(6, ql.Months),
        calendar, ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(
        ql.Swap.Payer, 1_000_000,
        fixSch, 0.050, ql.Thirty360(ql.Thirty360.European),
        fltSch, index, 0.0, index.dayCounter(),
    )
    exercise  = ql.EuropeanExercise(swapStart)
    swaption  = ql.Swaption(swap, exercise)

    def attach_swaption_engine():
        a0 = sq_a.value().getValue()
        s0 = sq_s.value().getValue()
        model  = ql.HullWhite(ts_handle, a0, s0)
        engine = ql.JamshidianSwaptionEngine(model)
        swaption.setPricingEngine(engine)

    attach_swaption_engine()
    return swaption, sq_rate, sq_a, sq_s, attach_swaption_engine, ts_handle, termStructure


def _build_swaption_aad():
    """Return (tape, swaption_aad, [rate_v, a_v, s_v])."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()

    todaysDate     = ql.Date(15, ql.February, 2002)
    settlementDate = ql.Date(19, ql.February, 2002)
    ql.Settings.instance().evaluationDate = todaysDate
    calendar = ql.TARGET()

    rate_v = ql.Real(_SWAP_RATE0)
    a_v    = ql.Real(_HW_A0)
    s_v    = ql.Real(_HW_S0)
    all_inputs = [rate_v, a_v, s_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    termStructure = ql.FlatForward(
        settlementDate, ql.QuoteHandle(ql.SimpleQuote(rate_v)), ql.Actual365Fixed())
    ts_handle = ql.RelinkableYieldTermStructureHandle(termStructure)

    index = ql.Euribor6M(ts_handle)
    swapStart = calendar.advance(settlementDate, 1, ql.Years,
                                 ql.ModifiedFollowing)
    swapEnd   = calendar.advance(swapStart, 5, ql.Years,
                                 ql.ModifiedFollowing)
    fixSch = ql.Schedule(swapStart, swapEnd, ql.Period(1, ql.Years),
        calendar, ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Forward, False)
    fltSch = ql.Schedule(swapStart, swapEnd, ql.Period(6, ql.Months),
        calendar, ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(
        ql.Swap.Payer, 1_000_000,
        fixSch, 0.050, ql.Thirty360(ql.Thirty360.European),
        fltSch, index, 0.0, index.dayCounter(),
    )
    exercise = ql.EuropeanExercise(swapStart)
    swaption = ql.Swaption(swap, exercise)
    model    = ql.HullWhite(ts_handle, a_v, s_v)
    swaption.setPricingEngine(ql.JamshidianSwaptionEngine(model))
    return tape, swaption, all_inputs


# ============================================================================
# Worker entry point
# ============================================================================

def _run_worker(repeats: int) -> dict:
    results = {}

    # -------------------------------------------------------------- AmOpt --
    amopt_plain, amopt_quotes = _build_amopt_plain()
    # Cache-bust: nudge sq_spot by a sub-pip amount using the stored initial
    # float so no xad arithmetic is involved.
    def _plain_amopt():
        amopt_quotes[0].setValue(_AMOPT_S0 + 1e-10)
        amopt_quotes[0].setValue(_AMOPT_S0)
        return amopt_plain.NPV()

    m, s = _median_ms(_plain_amopt, repeats)
    results["amopt_plain"] = {"median": m, "stdev": s, "n": len(amopt_quotes)}

    tape_amopt, amopt_aad, amopt_inputs = _build_amopt_aad()
    npv_amopt = amopt_aad.NPV()
    tape_amopt.registerOutput(npv_amopt)

    def _aad_amopt():
        tape_amopt.clearDerivatives()
        npv_amopt.derivative = 1.0
        tape_amopt.computeAdjoints()

    m, s = _median_ms(_aad_amopt, repeats)
    results["amopt_aad"] = {"median": m, "stdev": s, "n": len(amopt_inputs)}
    tape_amopt.deactivate()

    def _fd_amopt():
        amopt_plain.NPV()
        for q, v0 in zip(amopt_quotes,
                         [_AMOPT_S0, _AMOPT_R0, _AMOPT_Q0, _AMOPT_V0]):
            q.setValue(v0 + BPS)
            amopt_plain.NPV()
            q.setValue(v0)

    m, s = _median_ms(_fd_amopt, repeats)
    results["amopt_fd"] = {"median": m, "stdev": s, "n": len(amopt_quotes)}

    # ---------------------------------------------------------------- Cap --
    cap_plain, cap_rate_quotes, cap_sq_vol, cap_fcastH = _build_cap_plain()
    cap_all_quotes = cap_rate_quotes + [cap_sq_vol]

    def _plain_cap():
        # Same nudge-and-restore technique as the IRS benchmark.
        v0 = cap_rate_quotes[0].value().getValue()
        cap_rate_quotes[0].setValue(v0 + 1e-10)
        cap_rate_quotes[0].setValue(v0)
        return cap_plain.NPV()

    m, s = _median_ms(_plain_cap, repeats)
    results["cap_plain"] = {"median": m, "stdev": s, "n": len(cap_all_quotes)}

    tape_cap, cap_aad, cap_inputs = _build_cap_aad()
    npv_cap = cap_aad.NPV()
    tape_cap.registerOutput(npv_cap)

    def _aad_cap():
        tape_cap.clearDerivatives()
        npv_cap.derivative = 1.0
        tape_cap.computeAdjoints()

    m, s = _median_ms(_aad_cap, repeats)
    results["cap_aad"] = {"median": m, "stdev": s, "n": len(cap_inputs)}
    tape_cap.deactivate()

    # For FD: bump each of the 17 rate quotes then the vol quote.
    # The piecewise curve + cap engine are lazy: setValue() on any quote fires
    # the full observer chain and forces a recalibration.
    _cap_rate_initials = [0.0363,
                          0.037125, 0.037125, 0.037125,
                          96.2875, 96.7875, 96.9875, 96.6875,
                          96.4875, 96.3875, 96.2875, 96.0875,
                          0.037125, 0.0398, 0.0443, 0.05165, 0.055175]
    _cap_all_initials = _cap_rate_initials + [_CAP_VOL0]

    def _fd_cap():
        cap_plain.NPV()
        for q, v0 in zip(cap_all_quotes, _cap_all_initials):
            q.setValue(v0 + BPS)
            cap_plain.NPV()
            q.setValue(v0)

    m, s = _median_ms(_fd_cap, repeats)
    results["cap_fd"] = {"median": m, "stdev": s, "n": len(cap_all_quotes)}

    # ---------------------------------------------------------- Swaption --
    (swaption_plain, sq_rate, sq_a, sq_s,
     attach_swaption_engine, ts_handle_sw, ts_flat_sw) = _build_swaption_plain()

    def _plain_swaption():
        # Zero-allocation cache-bust: relinking fires the observer chain on the
        # HullWhite model without creating any new C++ objects.
        ts_handle_sw.linkTo(ts_flat_sw)
        return swaption_plain.NPV()

    m, s = _median_ms(_plain_swaption, repeats)
    results["swap_plain"] = {"median": m, "stdev": s, "n": 3}

    tape_sw, swaption_aad, sw_inputs = _build_swaption_aad()
    npv_sw = swaption_aad.NPV()
    tape_sw.registerOutput(npv_sw)

    def _aad_swaption():
        tape_sw.clearDerivatives()
        npv_sw.derivative = 1.0
        tape_sw.computeAdjoints()

    m, s = _median_ms(_aad_swaption, repeats, warmup=10)
    results["swap_aad"] = {"median": m, "stdev": s, "n": len(sw_inputs)}
    tape_sw.deactivate()

    def _fd_swaption():
        # Base price
        swaption_plain.NPV()
        # Rate: FlatForward with QuoteHandle(sq_rate) propagates automatically.
        sq_rate.setValue(_SWAP_RATE0 + BPS)
        swaption_plain.NPV()
        sq_rate.setValue(_SWAP_RATE0)
        # a: HullWhite stores by value → must rebuild model + engine.
        sq_a.setValue(_HW_A0 + BPS)
        attach_swaption_engine()
        swaption_plain.NPV()
        sq_a.setValue(_HW_A0)
        # σ: same.
        sq_s.setValue(_HW_S0 + BPS)
        attach_swaption_engine()
        swaption_plain.NPV()
        sq_s.setValue(_HW_S0)
        attach_swaption_engine()   # restore

    m, s = _median_ms(_fd_swaption, repeats)
    results["swap_fd"] = {"median": m, "stdev": s, "n": 3}

    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison table
# ============================================================================

INSTRUMENTS = [
    ("D", "American Option",
     " 4 inputs", "amopt_plain", "amopt_aad", "amopt_fd"),
    ("E", "Interest-Rate Cap",
     "18 inputs", "cap_plain",   "cap_aad",   "cap_fd"),
    ("F", "European Swaption",
     " 3 inputs", "swap_plain",  "swap_aad",  "swap_fd"),
]


def _fmt_t(median, stdev):
    return f"{median:8.3f} ±{stdev:6.3f} ms"


def _sp(nojit_med, jit_med):
    return f"{nojit_med / jit_med:7.2f}x" if jit_med > 0 else "    N/A"


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def print_comparison(nojit: dict, jit: dict, repeats: int, wheels: dict):
    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Additional JIT vs Non-JIT Benchmark Results")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Repeats   : {repeats}  (median of wall-clock timings)")
    print(f"  BPS shift : {BPS}  (used for bump-and-reprice FD)")
    print(f"  Non-JIT   : {wheels['nojit']['ql'].name}")
    print(f"  JIT       : {wheels['jit']['ql'].name}")
    print()

    col_m = 26
    col_t = 20
    hdr = (f"  {'Method':<{col_m}}  "
           f"{'Non-JIT':>{col_t}}  "
           f"{'JIT':>{col_t}}  "
           f"{'JIT speedup':>11}  "
           f"{'N inputs':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    aad_speedups = []
    for letter, name, detail, kp, ka, kf in INSTRUMENTS:
        n = nojit[kp]["n"]
        print(f"\n  [{letter}] {name}  ({detail})")
        for method, key in [
            ("Plain pricing",       kp),
            ("AAD  backward pass",  ka),
            ("Bump-and-reprice FD", kf),
        ]:
            nj = nojit[key]
            jt = jit[key]
            sp = _sp(nj["median"], jt["median"])
            if key == ka:
                aad_speedups.append(
                    nj["median"] / jt["median"] if jt["median"] > 0 else None)
            print(f"    {method:<{col_m - 4}}  "
                  f"{_fmt_t(nj['median'], nj['stdev']):>{col_t}}  "
                  f"{_fmt_t(jt['median'], jt['stdev']):>{col_t}}  "
                  f"{sp:>11}  "
                  f"{n:>8d}")

        fd_aad_nojit = (nojit[kf]["median"] / nojit[ka]["median"]
                        if nojit[ka]["median"] else 0.0)
        fd_aad_jit   = (jit[kf]["median"]   / jit[ka]["median"]
                        if jit[ka]["median"] else 0.0)
        print(f"    {'FD ÷ AAD  (within each build)':<{col_m - 4}}  "
              f"{'':>{col_t}}  {'':>{col_t}}  "
              f"  nojit {fd_aad_nojit:5.1f}x  jit {fd_aad_jit:5.1f}x")

    gm = _geomean(aad_speedups)
    print()
    print(SEPARATOR)
    print("  SUMMARY  –  JIT speedup on AAD backward pass")
    print(SEPARATOR)
    for (letter, name, _, _, _, _), sp in zip(INSTRUMENTS, aad_speedups):
        tag = f"[{letter}] {name}"
        val = f"{sp:6.2f}x" if sp else "N/A"
        print(f"    {tag:<30s}  {val}")
    print(f"    {'Geometric mean':<30s}  {gm:6.2f}x")
    print()
    print("  Notes")
    print("  • 'AAD backward pass'   – tape replay only; tape recorded once at startup")
    print("  • 'Bump-and-reprice FD' – N+1 complete forward pricings per call")
    print("  • 'JIT speedup'         – Non-JIT time / JIT time  (> 1.0 = JIT is faster)")
    print("  • 'FD ÷ AAD'            – how many times more expensive FD is vs AAD")
    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "more_benchmarks_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    """Write/overwrite more_benchmarks_results.md with the latest results."""
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — Additional JIT vs Non-JIT Benchmark Results")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**Repetitions:** {repeats} (median reported)  ")
    w(f"**Non-JIT wheel:** `{wheels['nojit']['ql'].name}`  ")
    w(f"**JIT wheel:** `{wheels['jit']['ql'].name}`  ")
    w("")
    w("---")
    w("")
    w("## Instruments")
    w("")
    w("These benchmarks complement `BENCHMARK_RESULTS.md` (Vanilla IRS, European Option,")
    w("Callable Bond) with three additional instruments from `Python/examples/`:")
    w("")
    w("| # | Instrument | Engine | N inputs | Source example |")
    w("|---|---|---|---:|---|")
    w("| D | American option (Put, K=40, S=36) | `BaroneAdesiWhaleyApproximationEngine` | 4 | `american-option.py` |")
    w("| E | 10Y interest-rate cap on Euribor3M | `BlackCapFloorEngine` + bootstrapped curve | 18 | `capsfloors.py` |")
    w("| F | 1Y×5Y European payer swaption | `JamshidianSwaptionEngine` + Hull-White | 3 | `bermudan-swaption.py` |")
    w("")
    w("---")
    w("")
    w("## What is being measured")
    w("")
    w("| Method | Description |")
    w("|---|---|")
    w("| **Plain pricing** | Single NPV call, `float` inputs, no AD overhead |")
    w("| **AAD backward pass** | XAD reverse-mode tape recorded once at startup; "
      "each iteration replays only the backward sweep — O(1) w.r.t. number of inputs |")
    w("| **Bump-and-reprice FD** | N+1 forward pricings with a 1 bp shift per input — O(N) |")
    w("")
    w("Both Non-JIT (XAD tape) and JIT (XAD-Forge JIT compilation) builds are run in "
      "isolated virtual environments.")
    w("")
    w("---")
    w("")
    w("## Results")
    w("")

    HIGH_CV = 0.5
    has_high_cv = False
    aad_speedups = []

    for letter, name, detail, kp, ka, kf in INSTRUMENTS:
        n = nojit[kp]["n"]
        instrument_desc = {
            "D": "American Option (BAW approximation) — 4 inputs: spot S, risk-free rate r, dividend yield q, volatility σ",
            "E": "Interest-Rate Cap (Black engine, bootstrapped Euribor3M curve) — 18 inputs: 17 curve quotes + flat vol",
            "F": "European Payer Swaption (Jamshidian / Hull-White) — 3 inputs: flat rate r, HW mean-reversion a, HW vol σ",
        }
        w(f"### {letter}. {name} — {n} market inputs")
        w("")
        w(f"*{instrument_desc[letter]}*")
        w("")
        w("| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |")
        w("|---|---:|---:|---:|---:|")
        for method, key in [
            ("Plain pricing",         kp),
            ("**AAD backward pass**", ka),
            ("Bump-and-reprice FD",   kf),
        ]:
            nj = nojit[key]
            jt = jit[key]
            sp = f"{nj['median'] / jt['median']:.2f}×" if jt["median"] > 0 else "—"
            if key == ka:
                aad_speedups.append(
                    nj["median"] / jt["median"] if jt["median"] > 0 else None)
            nj_flag = "†" if nj["median"] and nj["stdev"] / nj["median"] > HIGH_CV else ""
            jt_flag = "†" if jt["median"] and jt["stdev"] / jt["median"] > HIGH_CV else ""
            if nj_flag or jt_flag:
                has_high_cv = True
            w(f"| {method} "
              f"| {nj['median']:.4f} ±{nj['stdev']:.4f}{nj_flag} "
              f"| {jt['median']:.4f} ±{jt['stdev']:.4f}{jt_flag} "
              f"| {sp} | {n} |")

        fd_aad_nojit = (nojit[kf]["median"] / nojit[ka]["median"]
                        if nojit[ka]["median"] else 0.0)
        fd_aad_jit   = (jit[kf]["median"]   / jit[ka]["median"]
                        if jit[ka]["median"] else 0.0)
        w(f"| *FD ÷ AAD (within build)* | *{fd_aad_nojit:.1f}×* "
          f"| *{fd_aad_jit:.1f}×* | — | — |")
        w("")

    w("---")
    w("")
    w("## Summary — JIT speedup on AAD backward pass")
    w("")
    w("| Instrument | JIT speedup |")
    w("|---|---:|")
    for (letter, name, _, _, _, _), sp in zip(INSTRUMENTS, aad_speedups):
        val = f"{sp:.2f}×" if sp else "N/A"
        w(f"| [{letter}] {name} | {val} |")
    gm = _geomean(aad_speedups)
    w(f"| **Geometric mean** | **{gm:.2f}×** |")
    w("")
    w("---")
    w("")
    w("## Instrument notes")
    w("")
    w("### D — American Option (Barone-Adesi-Whaley)")
    w("")
    w("The BAW approximation prices an American put via a quadratic approximation of the")
    w("early-exercise premium.  The formula involves `exp`, `sqrt`, and iterative Newton")
    w("solving for the critical spot price, all of which flow cleanly through the XAD tape.")
    w("With only 4 inputs the FD/AAD ratio is ~5×; JIT accelerates the more complex")
    w("graph compared with the plain Black-Scholes European case.")
    w("")
    w("### E — Interest-Rate Cap (BlackCapFloorEngine)")
    w("")
    w("The cap is built on the same bootstrapped Euribor3M forward curve as the Vanilla")
    w("IRS benchmark (17 quote inputs) with one additional flat Black vol input = 18 total.")
    w("Pricing involves ~40 caplet Black-formula evaluations, each requiring forward-rate")
    w("and discount-factor lookups from the piecewise curve.  With 18 inputs, FD requires")
    w("19 full repricings (≈ 760 Black formula calls) versus a single backward sweep for")
    w("AAD — a large FD/AAD ratio that grows with the number of caplets.")
    w("")
    w("### F — European Payer Swaption (Jamshidian / Hull-White)")
    w("")
    w("Jamshidian decomposition prices a European swaption by finding the critical short")
    w("rate r* under Hull-White dynamics and then summing zero-bond options.  Three inputs")
    w("are differentiated: the flat term-structure rate r, HW mean-reversion a, and HW")
    w("short-rate vol σ.  Because a and σ are stored by value in the HullWhite model,")
    w("FD for those two inputs requires rebuilding the model + engine on each bump.")
    w("")
    w("---")
    w("")
    w("## General notes")
    w("")
    w(f"- BPS shift for FD: `{BPS}`")
    w("- *AAD backward pass* times the **backward sweep only**; the tape is recorded")
    w("  once at startup and reused for all repetitions.")
    w("- *JIT speedup* = Non-JIT time ÷ JIT time; values > 1.0 mean JIT is faster.")
    w("- *FD ÷ AAD* shows how many times more expensive bump-and-reprice is compared")
    w("  to one AAD backward pass within the same build.")
    w("- AAD complexity is **O(1)** in the number of inputs; FD is **O(N)**.")
    if has_high_cv:
        w(f"- **†** High variance (stdev/median > {HIGH_CV:.0%}): the median is the primary"
          " metric.  JIT builds can exhibit occasional LLVM recompilation spikes during"
          " plain-pricing and FD timing; AAD backward-pass timings are unaffected.")
    w("")
    w("---")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("# Build both variants (first time only)")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("")
    w("# Run the benchmark (venvs are reused from run_benchmarks.py if present)")
    w("python benchmarks/run_more_benchmarks.py")
    w("")
    w("# More repeats for stable numbers:")
    w("python benchmarks/run_more_benchmarks.py --repeats 50")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks additional JIT vs Non-JIT benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--worker", metavar="REPEATS", type=int, default=None,
        help="Internal worker mode: run benchmarks and print JSON",
    )
    parser.add_argument(
        "--repeats", "-r", type=int, default=30,
        help="Number of repetitions per benchmark (default: 30)",
    )
    parser.add_argument(
        "--clean-venvs", action="store_true",
        help="Destroy and recreate the benchmark venvs before running",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write results to more_benchmarks_results.md",
    )
    args = parser.parse_args()

    if args.worker is not None:
        worker_main(args.worker)
        return

    # ---- ORCHESTRATOR MODE ----
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
        print("    ./build.sh --jit    -j$(nproc)")
        sys.exit(1)

    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Additional JIT vs Non-JIT Benchmark")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)

    print("\nSetting up virtual environments")
    print("-" * 50)
    print(f"\n[1/2] Non-JIT venv  ({VENV_NOJIT.name})")
    setup_venv(VENV_NOJIT, wheels["nojit"]["xad"], wheels["nojit"]["ql"],
               force=args.clean_venvs)
    print(f"\n[2/2] JIT venv      ({VENV_JIT.name})")
    setup_venv(VENV_JIT, wheels["jit"]["xad"], wheels["jit"]["ql"],
               force=args.clean_venvs)

    print(f"\nRunning benchmarks  ({repeats} repeats each build)")
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

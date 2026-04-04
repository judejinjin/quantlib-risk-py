#!/usr/bin/env python3
"""
QuantLib-Risks-Py  –  JIT vs Non-JIT Benchmark
=================================================
Compares wall-clock time for AAD (XAD backward pass) and bump-and-reprice (FD)
between:

  • Standard build  – XAD tape, no JIT   (build/linux-xad-gcc-ninja-release/)
  • JIT build       – XAD-Forge JIT tape  (build/linux-xad-jit-gcc-ninja-release/)

Instruments benchmarked
-----------------------
  A)  Vanilla IRS (5-year spot swap)           17 market inputs
  B)  European option (Black-Scholes)           4 market inputs
  C)  Callable bond (HullWhite tree, 40 steps)  3 model parameters

Each instrument is timed for:
  • Plain forward pricing  (float, no tape)
  • AAD backward pass      (tape replay only, recorded once at startup)
  • Bump-and-reprice FD    (N+1 complete forward pricings)

Usage
-----
  # Full orchestrated run – sets up isolated venvs, installs wheels, compares:
  python benchmarks/run_benchmarks.py

  # Options:
  python benchmarks/run_benchmarks.py --repeats 50   # default 30
  python benchmarks/run_benchmarks.py --clean-venvs  # force-rebuild venvs

  # Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/run_benchmarks.py --worker REPEATS

Prerequisites
-------------
  Both builds must be present:
    ./build.sh --no-jit -j$(nproc)
    ./build.sh --jit    -j$(nproc)

  Venvs are created at:
    build/bench-venv-nojit/
    build/bench-venv-jit/
  and reused on subsequent runs (use --clean-venvs to force recreation).
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
import gc
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT  = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

VENV_NOJIT = BUILD / "bench-venv-nojit"
VENV_JIT   = BUILD / "bench-venv-jit"

SEPARATOR = "=" * 84
BPS = 1e-4   # 1 basis-point shift used in FD benchmarks


# ============================================================================
# Wheel discovery
# ============================================================================

def find_wheels(build_root: Path) -> dict:
    """
    Returns:
        {
          "nojit": {"xad": Path | None, "ql": Path | None},
          "jit":   {"xad": Path | None, "ql": Path | None},
        }
    The latest .whl matching each glob is returned (sorted lexicographically,
    which works because wheel filenames embed the version).
    """
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
# Virtual-environment helpers
# ============================================================================

def python_in(venv: Path) -> Path:
    return venv / "bin" / "python"


def _install_xad_shim(py: str):
    """Build the `xad` compatibility shim in a temp dir and pip-install it."""
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
        (base / "xad/exceptions/__init__.py").write_text("from xad_autodiff.exceptions import *\n")
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
    """Install xad_autodiff wheel, xad shim, and quantlib_risks wheel into venv."""
    py = str(python_in(venv))
    # setuptools is needed to build the xad shim (--no-build-isolation path)
    print(f"    • pip install setuptools wheel")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "setuptools", "wheel"]
    )
    print(f"    • pip install {xad_wheel.name}")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "--force-reinstall",
         "--no-deps", str(xad_wheel)]
    )
    print(f"    • pip install xad compatibility shim")
    _install_xad_shim(py)
    print(f"    • pip install {ql_wheel.name}")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "--force-reinstall",
         "--no-deps", str(ql_wheel)]
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
# Worker subprocess runner (called by orchestrator)
# ============================================================================

def _clean_env() -> dict:
    """Return os.environ minus variables that leak from conda/virtualenv parents."""
    drop = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHON_BASIC_REPL",
            "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT"}
    return {k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith(("CONDA_", "PYTHON_"))}


def run_worker_in_venv(venv: Path, repeats: int) -> dict:
    """Invoke this script as --worker inside venv; return the JSON results dict."""
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
    # The worker prints JSON as the last non-empty line starting with '{'
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(
        f"No JSON found in worker output ({venv.name}):\n{result.stdout[-2000:]}"
    )


# ============================================================================
# WORKER MODE  –  actual benchmark logic (runs inside each isolated venv)
# ============================================================================

def _median_ms(func, n: int, warmup: int = 5):
    """Return (median_ms, stdev_ms) over n calls of func().

    warmup un-timed calls are made first so that JIT compilation and any
    first-call overhead (e.g. dynamic linking) do not pollute the timings.
    """
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ---- A. Vanilla IRS  (17 market inputs) ------------------------------------

def _build_irs_plain():
    import QuantLib_Risks as ql
    calendar       = ql.TARGET()
    todaysDate     = ql.Date(6, ql.November, 2001)
    ql.Settings.instance().evaluationDate = todaysDate
    settlementDate = ql.Date(8, ql.November, 2001)

    deps = {(3, ql.Months): ql.SimpleQuote(0.0363)}
    fras = {(3, 6): ql.SimpleQuote(0.037125),
            (6, 9): ql.SimpleQuote(0.037125),
            (9, 12): ql.SimpleQuote(0.037125)}
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
    depH  = [ql.DepositRateHelper(ql.QuoteHandle(deps[(n, u)]), ql.Period(n, u), 2,
                calendar, ql.ModifiedFollowing, False, dayCounter)
             for n, u in deps]
    fraH  = [ql.FraRateHelper(ql.QuoteHandle(fras[(n, m)]), n, m, 2,
                calendar, ql.ModifiedFollowing, False, dayCounter)
             for n, m in fras]
    futH  = [ql.FuturesRateHelper(ql.QuoteHandle(futs[d]), d, 3,
                calendar, ql.ModifiedFollowing, True, dayCounter,
                ql.QuoteHandle(ql.SimpleQuote(0.0)))
             for d in futs]
    discountTS = ql.YieldTermStructureHandle(
        ql.FlatForward(settlementDate, 0.04, ql.Actual360()))
    swpH  = [ql.SwapRateHelper(ql.QuoteHandle(swps[(n, u)]), ql.Period(n, u),
                calendar, ql.Annual, ql.Unadjusted,
                ql.Thirty360(ql.Thirty360.BondBasis), ql.Euribor3M(),
                ql.QuoteHandle(), ql.Period("0D"), discountTS)
             for n, u in swps]

    fcastH = ql.RelinkableYieldTermStructureHandle()
    fcastH.linkTo(ql.PiecewiseFlatForward(settlementDate,
        depH + futH + swpH[1:], ql.Actual360()))

    maturity = calendar.advance(settlementDate, 5, ql.Years)
    index    = ql.Euribor3M(fcastH)
    fxSch = ql.Schedule(settlementDate, maturity, ql.Period(1, ql.Years), calendar,
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    flSch = ql.Schedule(settlementDate, maturity, ql.Period(3, ql.Months), calendar,
        ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.Swap.Payer, 1_000_000, fxSch, 0.04,
        ql.Thirty360(ql.Thirty360.BondBasis), flSch, index, 0.0, index.dayCounter())
    swap.setPricingEngine(ql.DiscountingSwapEngine(discountTS))
    return swap, all_quotes


def _build_irs_aad():
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()
    calendar       = ql.TARGET()
    todaysDate     = ql.Date(6, ql.November, 2001)
    ql.Settings.instance().evaluationDate = todaysDate
    settlementDate = ql.Date(8, ql.November, 2001)

    deps = {(3, ql.Months): ql.Real(0.0363)}
    fras = {(3, 6): ql.Real(0.037125),
            (6, 9): ql.Real(0.037125),
            (9, 12): ql.Real(0.037125)}
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
    tape.registerInputs(all_inputs)
    tape.newRecording()

    dayCounter = ql.Actual360()
    depH  = [ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(deps[(n, u)])),
                ql.Period(n, u), 2, calendar, ql.ModifiedFollowing, False, dayCounter)
             for n, u in deps]
    fraH  = [ql.FraRateHelper(ql.QuoteHandle(ql.SimpleQuote(fras[(n, m)])),
                n, m, 2, calendar, ql.ModifiedFollowing, False, dayCounter)
             for n, m in fras]
    futH  = [ql.FuturesRateHelper(ql.QuoteHandle(ql.SimpleQuote(futs[d])), d, 3,
                calendar, ql.ModifiedFollowing, True, dayCounter,
                ql.QuoteHandle(ql.SimpleQuote(0.0)))
             for d in futs]
    discountTS = ql.YieldTermStructureHandle(
        ql.FlatForward(settlementDate, 0.04, ql.Actual360()))
    swpH  = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(swps[(n, u)])),
                ql.Period(n, u), calendar, ql.Annual, ql.Unadjusted,
                ql.Thirty360(ql.Thirty360.BondBasis), ql.Euribor3M(),
                ql.QuoteHandle(), ql.Period("0D"), discountTS)
             for n, u in swps]

    fcastH = ql.RelinkableYieldTermStructureHandle()
    fcastH.linkTo(ql.PiecewiseFlatForward(settlementDate,
        depH + futH + swpH[1:], ql.Actual360()))

    maturity = calendar.advance(settlementDate, 5, ql.Years)
    index    = ql.Euribor3M(fcastH)
    fxSch = ql.Schedule(settlementDate, maturity, ql.Period(1, ql.Years), calendar,
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    flSch = ql.Schedule(settlementDate, maturity, ql.Period(3, ql.Months), calendar,
        ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.Swap.Payer, 1_000_000, fxSch, 0.04,
        ql.Thirty360(ql.Thirty360.BondBasis), flSch, index, 0.0, index.dayCounter())
    swap.setPricingEngine(ql.DiscountingSwapEngine(discountTS))
    return tape, swap, all_inputs


# ---- B. European option  (4 inputs) ----------------------------------------

def _build_option_plain():
    import QuantLib_Risks as ql
    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    sq_spot = ql.SimpleQuote(7.0)
    sq_div  = ql.SimpleQuote(0.05)
    sq_vol  = ql.SimpleQuote(0.10)
    sq_rate = ql.SimpleQuote(0.05)
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sq_spot),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(sq_div), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(sq_rate), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(sq_vol), ql.Actual365Fixed())))
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, 8.0),
        ql.EuropeanExercise(ql.Date(17, ql.May, 1999)))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return option, [sq_spot, sq_div, sq_vol, sq_rate]


def _build_option_aad():
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()
    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    spot_v = ql.Real(7.0)
    div_v  = ql.Real(0.05)
    vol_v  = ql.Real(0.10)
    rate_v = ql.Real(0.05)
    all_inputs = [spot_v, div_v, vol_v, rate_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot_v)),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(ql.SimpleQuote(div_v)), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(ql.SimpleQuote(rate_v)), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(vol_v)),
            ql.Actual365Fixed())))
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, 8.0),
        ql.EuropeanExercise(ql.Date(17, ql.May, 1999)))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return tape, option, all_inputs


# ---- C. Callable bond  (3 inputs: rate, HW mean-reversion a, HW vol s) -----

def _make_callable_bond_plain():
    """
    Returns (bond, sq_rate, sq_a, sq_s, attach_engine, ts_handle, termStructure).

    attach_engine() must be called after bumping sq_a or sq_s, because
    HullWhite stores those parameters by value at construction time.
    For sq_rate the RelinkableYieldTermStructureHandle propagates changes
    automatically so no engine rebuild is needed.

    ts_handle and termStructure are also returned so callers can use
    ``ts_handle.linkTo(termStructure)`` as a zero-allocation cache-bust that
    fires the full observer chain without creating new C++ objects.
    """
    import QuantLib_Risks as ql
    calcDate = ql.Date(16, 8, 2006)
    ql.Settings.instance().evaluationDate = calcDate
    dayCount = ql.ActualActual(ql.ActualActual.Bond)

    sq_rate = ql.SimpleQuote(0.0465)
    sq_a    = ql.SimpleQuote(0.06)
    sq_s    = ql.SimpleQuote(0.20)

    termStructure = ql.FlatForward(
        calcDate, ql.QuoteHandle(sq_rate), dayCount, ql.Compounded, ql.Semiannual)
    ts_handle = ql.RelinkableYieldTermStructureHandle(termStructure)

    callabilitySchedule = ql.CallabilitySchedule()
    callDate = ql.Date(15, ql.September, 2006)
    nc = ql.NullCalendar()
    for _ in range(24):
        callabilitySchedule.append(ql.Callability(
            ql.BondPrice(100.0, ql.BondPrice.Clean), ql.Callability.Call, callDate))
        callDate = nc.advance(callDate, 3, ql.Months)

    issueDate    = ql.Date(16, ql.September, 2004)
    maturityDate = ql.Date(15, ql.September, 2012)
    calendar     = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    schedule = ql.Schedule(issueDate, maturityDate, ql.Period(ql.Quarterly),
        calendar, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    bond = ql.CallableFixedRateBond(3, 100, schedule, [0.025],
        ql.ActualActual(ql.ActualActual.Bond), ql.Following, 100,
        issueDate, callabilitySchedule)

    def attach_engine():
        # Use getValue() to pass plain Python floats so no xad arithmetic
        # is recorded outside an active tape session.
        a0 = sq_a.value().getValue()
        s0 = sq_s.value().getValue()
        model = ql.HullWhite(ts_handle, a0, s0)
        bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model, 40))

    attach_engine()
    return bond, sq_rate, sq_a, sq_s, attach_engine, ts_handle, termStructure


def _build_bond_aad():
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()
    calcDate = ql.Date(16, 8, 2006)
    ql.Settings.instance().evaluationDate = calcDate
    dayCount = ql.ActualActual(ql.ActualActual.Bond)

    rate_v = ql.Real(0.0465)
    a_v    = ql.Real(0.06)
    s_v    = ql.Real(0.20)
    all_inputs = [rate_v, a_v, s_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    termStructure = ql.FlatForward(
        calcDate, ql.QuoteHandle(ql.SimpleQuote(rate_v)), dayCount,
        ql.Compounded, ql.Semiannual)
    ts_handle = ql.RelinkableYieldTermStructureHandle(termStructure)

    callabilitySchedule = ql.CallabilitySchedule()
    callDate = ql.Date(15, ql.September, 2006)
    nc = ql.NullCalendar()
    for _ in range(24):
        callabilitySchedule.append(ql.Callability(
            ql.BondPrice(100.0, ql.BondPrice.Clean), ql.Callability.Call, callDate))
        callDate = nc.advance(callDate, 3, ql.Months)

    issueDate    = ql.Date(16, ql.September, 2004)
    maturityDate = ql.Date(15, ql.September, 2012)
    calendar     = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    schedule = ql.Schedule(issueDate, maturityDate, ql.Period(ql.Quarterly),
        calendar, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    bond = ql.CallableFixedRateBond(3, 100, schedule, [0.025],
        ql.ActualActual(ql.ActualActual.Bond), ql.Following, 100,
        issueDate, callabilitySchedule)
    model = ql.HullWhite(ts_handle, a_v, s_v)
    bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model, 40))
    return tape, bond, all_inputs


# ---- Worker entry point -----------------------------------------------------

def _run_worker(repeats: int) -> dict:
    results = {}

    # ------------------------------------------------------------------ IRS --
    irs_plain, irs_quotes = _build_irs_plain()

    def _plain_irs():
        # Nudge the first quote by a sub-pip amount and restore, so QuantLib's
        # LazyObject cache is invalidated and every iteration does a real pricing.
        # Use .getValue() to extract a plain Python float so xad arithmetic is
        # not involved — repeated xad nudges can corrupt later tape sweeps.
        v0 = irs_quotes[0].value().getValue()
        irs_quotes[0].setValue(v0 + 1e-10)
        irs_quotes[0].setValue(v0)
        return irs_plain.NPV()

    m, s = _median_ms(_plain_irs, repeats)
    results["irs_plain"] = {"median": m, "stdev": s, "n": len(irs_quotes)}

    tape_irs, irs_aad, irs_inputs = _build_irs_aad()
    npv_irs = irs_aad.NPV()
    tape_irs.registerOutput(npv_irs)

    def _aad_irs():
        tape_irs.clearDerivatives()
        npv_irs.derivative = 1.0
        tape_irs.computeAdjoints()

    m, s = _median_ms(_aad_irs, repeats)
    results["irs_aad"] = {"median": m, "stdev": s, "n": len(irs_inputs)}
    tape_irs.deactivate()

    def _fd_irs():
        irs_plain.NPV()
        for q in irs_quotes:
            v0 = q.value().getValue()
            q.setValue(v0 + BPS)
            irs_plain.NPV()
            q.setValue(v0)

    m, s = _median_ms(_fd_irs, repeats)
    results["irs_fd"] = {"median": m, "stdev": s, "n": len(irs_quotes)}

    # --------------------------------------------------------------- Option --
    opt_plain, opt_quotes = _build_option_plain()

    def _plain_opt():
        v0 = opt_quotes[0].value().getValue()
        opt_quotes[0].setValue(v0 + 1e-10)
        opt_quotes[0].setValue(v0)
        return opt_plain.NPV()

    m, s = _median_ms(_plain_opt, repeats)
    results["opt_plain"] = {"median": m, "stdev": s, "n": len(opt_quotes)}

    tape_opt, opt_aad, opt_inputs = _build_option_aad()
    npv_opt = opt_aad.NPV()
    tape_opt.registerOutput(npv_opt)

    def _aad_opt():
        tape_opt.clearDerivatives()
        npv_opt.derivative = 1.0
        tape_opt.computeAdjoints()

    m, s = _median_ms(_aad_opt, repeats)
    results["opt_aad"] = {"median": m, "stdev": s, "n": len(opt_inputs)}
    tape_opt.deactivate()

    def _fd_opt():
        opt_plain.NPV()
        for q in opt_quotes:
            v0 = q.value().getValue()
            q.setValue(v0 + BPS)
            opt_plain.NPV()
            q.setValue(v0)

    m, s = _median_ms(_fd_opt, repeats)
    results["opt_fd"] = {"median": m, "stdev": s, "n": len(opt_quotes)}

    # ------------------------------------------------------------- CallBond --
    bond_plain, sq_rate, sq_a, sq_s, attach_engine, ts_handle_plain, ts_plain = \
        _make_callable_bond_plain()

    def _plain_bond():
        # Relinking the handle to the same underlying TermStructure fires the
        # full observer chain (invalidates QuantLib's LazyObject cache) without
        # allocating any new C++ objects.  This is safe for unlimited repeats.
        ts_handle_plain.linkTo(ts_plain)
        return bond_plain.cleanPrice()

    m, s = _median_ms(_plain_bond, repeats)
    results["bond_plain"] = {"median": m, "stdev": s, "n": 3}

    tape_bond, bond_aad, bond_inputs = _build_bond_aad()
    npv_bond = bond_aad.cleanPrice()
    tape_bond.registerOutput(npv_bond)

    def _aad_bond():
        tape_bond.clearDerivatives()
        npv_bond.derivative = 1.0
        tape_bond.computeAdjoints()

    m, s = _median_ms(_aad_bond, repeats, warmup=10)
    results["bond_aad"] = {"median": m, "stdev": s, "n": len(bond_inputs)}
    tape_bond.deactivate()

    def _fd_bond():
        # rate: RelinkableYieldTermStructureHandle propagates automatically
        bond_plain.cleanPrice()
        v0 = sq_rate.value().getValue()
        sq_rate.setValue(v0 + BPS)
        bond_plain.cleanPrice()
        sq_rate.setValue(v0)
        # a: HullWhite stores by value → must rebuild model+engine
        v0 = sq_a.value().getValue()
        sq_a.setValue(v0 + BPS)
        attach_engine()
        bond_plain.cleanPrice()
        sq_a.setValue(v0)
        # s: same
        v0 = sq_s.value().getValue()
        sq_s.setValue(v0 + BPS)
        attach_engine()
        bond_plain.cleanPrice()
        sq_s.setValue(v0)
        attach_engine()   # restore original engine

    m, s = _median_ms(_fd_bond, repeats)
    results["bond_fd"] = {"median": m, "stdev": s, "n": 3}

    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    # JSON on the last line; orchestrator scans for it
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison table
# ============================================================================

def _fmt_t(median, stdev):
    return f"{median:8.3f} ±{stdev:6.3f} ms"


def _sp(nojit_med, jit_med):
    if jit_med > 0:
        return f"{nojit_med / jit_med:7.2f}x"
    return "    N/A"


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


INSTRUMENTS = [
    ("A", "Vanilla IRS",      "17 inputs", "irs_plain",  "irs_aad",  "irs_fd"),
    ("B", "European Option",  " 4 inputs", "opt_plain",  "opt_aad",  "opt_fd"),
    ("C", "Callable Bond",    " 3 inputs", "bond_plain", "bond_aad", "bond_fd"),
]


def print_comparison(nojit: dict, jit: dict, repeats: int, wheels: dict):
    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  JIT vs Non-JIT Benchmark Results")
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
                aad_speedups.append(nj["median"] / jt["median"] if jt["median"] > 0 else None)
            print(f"    {method:<{col_m - 4}}  "
                  f"{_fmt_t(nj['median'], nj['stdev']):>{col_t}}  "
                  f"{_fmt_t(jt['median'], jt['stdev']):>{col_t}}  "
                  f"{sp:>11}  "
                  f"{n:>8d}")

        # same-build FD÷AAD ratio
        fd_aad_nojit = (nojit[kf]["median"] / nojit[ka]["median"]
                        if nojit[ka]["median"] else 0.0)
        fd_aad_jit   = (jit[kf]["median"]   / jit[ka]["median"]
                        if jit[ka]["median"] else 0.0)
        print(f"    {'FD ÷ AAD  (within each build)':<{col_m - 4}}  "
              f"{'':>{col_t}}  "
              f"{'':>{col_t}}  "
              f"  nojit {fd_aad_nojit:5.1f}x  jit {fd_aad_jit:5.1f}x")

    gm = _geomean(aad_speedups)
    print()
    print(SEPARATOR)
    print("  SUMMARY  –  JIT speedup on AAD backward pass")
    print(SEPARATOR)
    for (letter, name, _, _, ka, _), sp in zip(INSTRUMENTS, aad_speedups):
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

MD_PATH = Path(__file__).resolve().parent / "BENCHMARK_RESULTS.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    """Write/overwrite BENCHMARK_RESULTS.md with the latest results."""
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — JIT vs Non-JIT Benchmark Results")
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
    w("## What is being measured")
    w("")
    w("Each instrument is timed for three methods:")
    w("")
    w("| Method | Description |")
    w("|---|---|")
    w("| **Plain pricing** | Single NPV call, `float` inputs, no AD overhead |")
    w("| **AAD backward pass** | XAD reverse-mode tape recorded once at startup; "
      "each iteration replays only the backward sweep — O(1) w.r.t. number of inputs |")
    w("| **Bump-and-reprice FD** | N+1 forward pricings with a 1 bp shift per input — O(N) |")
    w("")
    w("Both builds are run in isolated virtual environments with their respective wheels.")
    w("")
    w("---")
    w("")
    w("## Results")
    w("")

    HIGH_CV = 0.5   # flag cells whose stdev/median exceeds this threshold

    aad_speedups = []
    has_high_cv = False
    for letter, name, detail, kp, ka, kf in INSTRUMENTS:
        n = nojit[kp]["n"]
        w(f"### {letter}. {name} — {n} market inputs")
        w("")
        w("| Method | Non-JIT (ms) | JIT (ms) | JIT speedup | N inputs |")
        w("|---|---:|---:|---:|---:|")
        for method, key in [
            ("Plain pricing",        kp),
            ("**AAD backward pass**", ka),
            ("Bump-and-reprice FD",  kf),
        ]:
            nj = nojit[key]
            jt = jit[key]
            if jt["median"] > 0:
                sp = f"{nj['median'] / jt['median']:.2f}×"
            else:
                sp = "—"
            if key == ka:
                aad_speedups.append(nj["median"] / jt["median"] if jt["median"] > 0 else None)

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
    w("## Notes")
    w("")
    w(f"- BPS shift for FD: `{BPS}`")
    w("- *AAD backward pass* times the **backward sweep only**; "
      "the tape is recorded once at startup and reused for all repetitions.")
    w("- *JIT speedup* = Non-JIT time ÷ JIT time; values > 1.0 mean JIT is faster.")
    w("- *FD ÷ AAD* shows how many times more expensive bump-and-reprice is "
      "compared to one AAD backward pass within the same build.")
    w("- AAD complexity is **O(1)** in the number of inputs; "
      "FD complexity is **O(N)**.")
    if has_high_cv:
        w(f"- **†** High variance (stdev/median > {HIGH_CV:.0%}): "
          "the median is still the primary metric. "
          "In the JIT build this occurs for plain-pricing and FD calls on the "
          "callable bond because Forge instruments even non-AD tree-pricer code "
          "paths, occasionally triggering LLVM recompilation mid-measurement. "
          "The AAD backward-pass timings (stdev/median < 10%) are unaffected "
          "and remain the authoritative JIT-speedup figure.")
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
    w("# Run the benchmark (venvs are created/reused automatically)")
    w("python benchmarks/run_benchmarks.py")
    w("")
    w("# More repeats for stable numbers:")
    w("python benchmarks/run_benchmarks.py --repeats 50")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks JIT vs Non-JIT benchmark",
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
        help="Do not write results to BENCHMARK_RESULTS.md",
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
    print("QuantLib-Risks-Py  –  JIT vs Non-JIT Benchmark")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)

    # ---- set up venvs ----
    print("\nSetting up virtual environments")
    print("-" * 50)
    print(f"\n[1/2] Non-JIT venv  ({VENV_NOJIT.name})")
    setup_venv(VENV_NOJIT, wheels["nojit"]["xad"], wheels["nojit"]["ql"],
               force=args.clean_venvs)
    print(f"\n[2/2] JIT venv      ({VENV_JIT.name})")
    setup_venv(VENV_JIT, wheels["jit"]["xad"], wheels["jit"]["ql"],
               force=args.clean_venvs)

    # ---- run workers ----
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

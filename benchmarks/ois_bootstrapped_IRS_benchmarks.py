#!/usr/bin/env python3
"""
QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: OIS-Bootstrapped IRS
===============================================================================

Benchmarks a SOFR-based interest-rate swap priced against an OIS-bootstrapped
discount/forecasting curve under N_SCENARIOS = 100 random Monte Carlo scenarios,
comparing:

  O1) FD             – bump-and-reprice (9 inputs → 10 pricings per scenario)
  O2) AAD replay     – N backward sweeps on a tape recorded once at base market
  O3) AAD re-record  – per-scenario fresh recording via SimpleQuote.setValue(Real)

The OIS bootstrap pipeline (OISRateHelper + Sofr + PiecewiseLogLinearDiscount)
and the DiscountingSwapEngine used for pricing are straight-line arithmetic with
no data-dependent branching on Real, so JIT (Forge) can be used.  Both non-JIT
and JIT builds are tested.

Market data: 9 interest-rate inputs at tenors 1M–30Y.  By default, rates are
scraped live from the US Treasury daily par yield curve at treasury.gov.
If scraping fails, the script warns and falls back to hardcoded Nov 2024 rates.
Use ``--offline`` to skip scraping and use hardcoded rates directly.
Note: Treasury par yields are a close proxy for SOFR OIS swap rates;
the benchmark performance comparison (FD vs AAD) is valid with either.

Instrument: 5-year SOFR OIS (pay fixed at-market, receive SOFR, $10M notional)

Inputs (9): OIS par rates for 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 30Y

Usage
-----
  python benchmarks/ois_bootstrapped_IRS_benchmarks.py               # live rates (default)
  python benchmarks/ois_bootstrapped_IRS_benchmarks.py --offline      # hardcoded Nov 2024
  python benchmarks/ois_bootstrapped_IRS_benchmarks.py --repeats 10
  python benchmarks/ois_bootstrapped_IRS_benchmarks.py --clean-venvs

  # Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/ois_bootstrapped_IRS_benchmarks.py --worker REPEATS --market-data '{...}'
"""

import argparse
import csv
import datetime
import io
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
SCENARIO_SEED = 789          # distinct seed from other benchmarks

# OIS base market — scraped Nov 2024 SOFR OIS rates (9 inputs)
_OIS_TENOR_LABELS = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y', '30Y']
_OIS_TENORS = [
    (1,  'Months'),
    (3,  'Months'),
    (6,  'Months'),
    (1,  'Years'),
    (2,  'Years'),
    (3,  'Years'),
    (5,  'Years'),
    (10, 'Years'),
    (30, 'Years'),
]
_OIS_BASE_RATES = [
    0.0483,   # 1M — 4.83%
    0.0455,   # 3M — 4.55%
    0.0425,   # 6M — 4.25%
    0.0357,   # 1Y — 3.57%
    0.0340,   # 2Y — 3.40%
    0.0335,   # 3Y — 3.35%
    0.0350,   # 5Y — 3.50%
    0.0385,   # 10Y — 3.85%
    0.0415,   # 30Y — 4.15%
]
_N_INPUTS = len(_OIS_BASE_RATES)   # 9

# Swap parameters
_SWAP_TENOR_YEARS = 5
_SWAP_NOMINAL     = 10_000_000

# Mutable slot: overwritten when --live or --market-data is used
_ACTIVE_RATES     = list(_OIS_BASE_RATES)
_RATES_SOURCE     = "hardcoded Nov 2024 SOFR OIS snapshot"
_RATES_DATE       = "2024-11-15"
_EVAL_DATE        = (15, 11, 2024)      # (day, month, year) for ql.Date
_SWAP_FIXED_RATE  = 0.0350              # at-market for the 5Y rate


# ---------------------------------------------------------------------------
# Live-rate scraping
# ---------------------------------------------------------------------------
# The US Treasury publishes daily par yield curves in CSV format.  These are
# Treasury yields (not SOFR OIS swap rates), but the tenors align closely
# and the benchmarking goal — comparing FD vs AAD performance — is valid
# with either data source.

# Mapping: (CSV column header) → index in our 9-rate vector
_TREASURY_COLUMN_MAP = {
    "1 Mo":  0,   # 1M
    "3 Mo":  1,   # 3M
    "6 Mo":  2,   # 6M
    "1 Yr":  3,   # 1Y
    "2 Yr":  4,   # 2Y
    "3 Yr":  5,   # 3Y
    "5 Yr":  6,   # 5Y
    "10 Yr": 7,   # 10Y
    "30 Yr": 8,   # 30Y
}


def scrape_live_rates() -> tuple:
    """
    Scrape the latest US Treasury par yield curve from treasury.gov.

    Returns
    -------
    (rates, source_label, date_str)
        rates : list of 9 floats (decimal, e.g. 0.0367 for 3.67%)
        source_label : human-readable data-source description
        date_str : YYYY-MM-DD of the observation
    """
    import requests  # only needed when --live is used

    now = datetime.datetime.now()
    year = now.year

    url = (
        f"https://home.treasury.gov/resource-center/data-chart-center/"
        f"interest-rates/daily-treasury-rates.csv/{year}/all"
        f"?type=daily_treasury_yield_curve"
        f"&field_tdr_date_value={year}&page&_format=csv"
    )

    print(f"  Fetching live rates from treasury.gov ({year}) …")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)
    if not rows:
        raise RuntimeError("Treasury CSV returned no rows")

    # First row is the most recent business day
    latest = rows[0]
    date_raw = latest.get("Date", "unknown")

    rates = [None] * 9
    for col_hdr, idx in _TREASURY_COLUMN_MAP.items():
        val = latest.get(col_hdr, "").strip()
        if val:
            rates[idx] = float(val) / 100.0   # percent → decimal

    missing = [_OIS_TENOR_LABELS[i] for i in range(9) if rates[i] is None]
    if missing:
        raise RuntimeError(
            f"Treasury CSV missing tenors: {', '.join(missing)} "
            f"(row date={date_raw})"
        )

    # Parse the date
    try:
        dt = datetime.datetime.strptime(date_raw, "%m/%d/%Y")
        date_str = dt.strftime("%Y-%m-%d")
    except ValueError:
        date_str = date_raw

    source = f"US Treasury daily par yield curve ({date_str})"
    return rates, source, date_str


def _set_active_rates(rates: list, source: str, date_str: str):
    """Replace the module-level active rates, eval date, and fixed rate."""
    global _ACTIVE_RATES, _RATES_SOURCE, _RATES_DATE, _EVAL_DATE, _SWAP_FIXED_RATE
    _ACTIVE_RATES = list(rates)
    _RATES_SOURCE = source
    _RATES_DATE   = date_str
    # Set the 5Y rate as the swap fixed rate (at-the-money)
    _SWAP_FIXED_RATE = rates[6]   # index 6 = 5Y
    # Derive evaluation date from the rate observation date
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        _EVAL_DATE = (dt.day, dt.month, dt.year)
    except ValueError:
        pass  # keep the default


def _gen_scenarios():
    """Pre-generate OIS MC scenarios deterministically."""
    rng = random.Random(SCENARIO_SEED)
    scenarios = []
    for _ in range(N_SCENARIOS):
        scene = [max(1e-5, v + rng.gauss(0, 5e-4)) for v in _ACTIVE_RATES]
        scenarios.append(scene)
    return scenarios


# ============================================================================
# Wheel / venv helpers  (dual-venv pattern for JIT comparison)
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


def run_worker_in_venv(venv: Path, repeats: int, market_data: dict = None) -> dict:
    py = str(python_in(venv))
    cmd = [py, str(Path(__file__).resolve()), "--worker", str(repeats)]
    if market_data:
        cmd.extend(["--market-data", json.dumps(market_data)])
    result = subprocess.run(
        cmd,
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


# ---- OIS-bootstrapped IRS (9 inputs) --------------------------------------

def _build_ois_irs_structures():
    """
    Build the OIS-bootstrapped IRS pricing structures.

    Plain-float structure:
      - 9 SimpleQuotes → 9 OISRateHelpers → PiecewiseLogLinearDiscount
      - 5Y OvernightIndexedSwap (pay fixed at 5Y rate, receive SOFR)
      - DiscountingSwapEngine discounting off the same OIS curve

    AAD structure:
      - Same pipeline built with xad Reals to record the tape

    Returns:
        swap         – OvernightIndexedSwap (float engine)
        ois_quotes   – list of 9 SimpleQuote objects
        tape         – Tape recorded at base market (for AAD replay)
        npv_out      – xad Real output registered on tape
        tape_inputs  – list of 9 xad Reals registered on tape
    """
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape, Real

    calendar = ql.UnitedStates(ql.UnitedStates.SOFR)
    todaysDate = ql.Date(_EVAL_DATE[0], _EVAL_DATE[1], _EVAL_DATE[2])
    ql.Settings.instance().evaluationDate = todaysDate
    settlementDays = 2

    # --- plain-float structure for FD ---
    ois_quotes = [ql.SimpleQuote(r) for r in _ACTIVE_RATES]

    sofr = ql.Sofr()
    helpers = []
    for i, (n, unit) in enumerate(_OIS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        helper = ql.OISRateHelper(
            settlementDays,
            period,
            ql.QuoteHandle(ois_quotes[i]),
            sofr,
        )
        helpers.append(helper)

    ois_curve = ql.PiecewiseLogLinearDiscount(todaysDate, helpers, ql.Actual365Fixed())
    ois_curve.enableExtrapolation()
    curve_handle = ql.YieldTermStructureHandle(ois_curve)

    # Link SOFR index to the OIS curve for forecasting
    sofr_fcast = ql.Sofr(curve_handle)

    # Build 5Y OIS: pay fixed, receive SOFR
    settlement = calendar.advance(todaysDate, settlementDays, ql.Days)
    maturity   = calendar.advance(settlement, _SWAP_TENOR_YEARS, ql.Years)
    schedule   = ql.Schedule(
        settlement, maturity,
        ql.Period(ql.Annual),
        calendar,
        ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Backward, False,
    )
    swap = ql.OvernightIndexedSwap(
        ql.OvernightIndexedSwap.Payer,
        _SWAP_NOMINAL,
        schedule,
        _SWAP_FIXED_RATE,
        ql.Actual360(),
        sofr_fcast,
    )
    engine = ql.DiscountingSwapEngine(curve_handle)
    swap.setPricingEngine(engine)

    # --- AAD tape for replay (recorded once at base market) ---
    tape = Tape()
    tape.activate()

    ois_r = [Real(r) for r in _ACTIVE_RATES]
    tape.registerInputs(ois_r)
    tape.newRecording()

    helpers_r = []
    for i, (n, unit) in enumerate(_OIS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        helper = ql.OISRateHelper(
            settlementDays,
            period,
            ql.QuoteHandle(ql.SimpleQuote(ois_r[i])),
            sofr,
        )
        helpers_r.append(helper)

    ois_curve_r = ql.PiecewiseLogLinearDiscount(todaysDate, helpers_r, ql.Actual365Fixed())
    ois_curve_r.enableExtrapolation()
    curve_handle_r = ql.YieldTermStructureHandle(ois_curve_r)

    sofr_fcast_r = ql.Sofr(curve_handle_r)

    swap_r = ql.OvernightIndexedSwap(
        ql.OvernightIndexedSwap.Payer,
        _SWAP_NOMINAL,
        schedule,
        _SWAP_FIXED_RATE,
        ql.Actual360(),
        sofr_fcast_r,
    )
    engine_r = ql.DiscountingSwapEngine(curve_handle_r)
    swap_r.setPricingEngine(engine_r)
    npv_out = swap_r.NPV()
    tape.registerOutput(npv_out)

    return swap, ois_quotes, tape, npv_out, ois_r


def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    from xad.adj_1st import Real, Tape

    scenarios = _gen_scenarios()
    results = {}

    (swap, ois_quotes,
     tape, npv_out, tape_inputs) = _build_ois_irs_structures()

    # O1 — FD batch: 9-input bump-and-reprice for every scenario
    def _ois_fd_mc():
        for scene in scenarios:
            for sq, v in zip(ois_quotes, scene):
                sq.setValue(v)
            swap.NPV()
            for sq, v in zip(ois_quotes, scene):
                sq.setValue(v + BPS)
                swap.NPV()
                sq.setValue(v)
        # Restore base
        for sq, v in zip(ois_quotes, _ACTIVE_RATES):
            sq.setValue(v)

    m, s = _median_ms(_ois_fd_mc, repeats)
    results["ois_fd_mc"] = {"median": m, "stdev": s,
                             "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    # O2 — AAD replay: N_SCENARIOS backward sweeps on a fixed tape
    def _ois_aad_replay():
        for _ in range(N_SCENARIOS):
            tape.clearDerivatives()
            npv_out.derivative = 1.0
            tape.computeAdjoints()

    m, s = _median_ms(_ois_aad_replay, repeats)
    results["ois_aad_replay"] = {"median": m, "stdev": s,
                                  "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}
    tape.deactivate()

    # O3 — AAD re-record: per-scenario fresh recording
    def _ois_aad_record():
        calendar = ql.UnitedStates(ql.UnitedStates.SOFR)
        todaysDate = ql.Date(_EVAL_DATE[0], _EVAL_DATE[1], _EVAL_DATE[2])
        settlementDays = 2
        sofr = ql.Sofr()
        settlement = calendar.advance(todaysDate, settlementDays, ql.Days)
        maturity   = calendar.advance(settlement, _SWAP_TENOR_YEARS, ql.Years)
        schedule   = ql.Schedule(
            settlement, maturity,
            ql.Period(ql.Annual),
            calendar,
            ql.ModifiedFollowing, ql.ModifiedFollowing,
            ql.DateGeneration.Backward, False,
        )

        tp = Tape()
        tp.activate()
        for scene in scenarios:
            reals = [Real(v) for v in scene]
            tp.registerInputs(reals)
            tp.newRecording()

            helpers = []
            for i, (n, unit) in enumerate(_OIS_TENORS):
                period = ql.Period(n, getattr(ql, unit))
                helper = ql.OISRateHelper(
                    settlementDays,
                    period,
                    ql.QuoteHandle(ql.SimpleQuote(reals[i])),
                    sofr,
                )
                helpers.append(helper)

            crv = ql.PiecewiseLogLinearDiscount(
                todaysDate, helpers, ql.Actual365Fixed())
            crv.enableExtrapolation()
            ch = ql.YieldTermStructureHandle(crv)

            sofr_f = ql.Sofr(ch)
            sw = ql.OvernightIndexedSwap(
                ql.OvernightIndexedSwap.Payer,
                _SWAP_NOMINAL,
                schedule,
                _SWAP_FIXED_RATE,
                ql.Actual360(),
                sofr_f,
            )
            eng = ql.DiscountingSwapEngine(ch)
            sw.setPricingEngine(eng)
            npv = sw.NPV()
            tp.registerOutput(npv)
            npv.derivative = 1.0
            tp.computeAdjoints()
        tp.deactivate()

    m, s = _median_ms(_ois_aad_record, repeats)
    results["ois_aad_record"] = {"median": m, "stdev": s,
                                  "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    return results


def worker_main(repeats: int, market_data_json: str = None):
    if market_data_json:
        md = json.loads(market_data_json)
        _set_active_rates(md["rates"], md["source"], md["date"])
    # Print active config for debugging
    print(f"Worker: eval_date={_EVAL_DATE}, fixed_rate={_SWAP_FIXED_RATE:.4f}, "
          f"source={_RATES_SOURCE}", file=sys.stderr)
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison table
# ============================================================================

INSTRUMENTS = [
    # (id, label,          fd_key,       replay_key,       record_key)
    ("O", "SOFR OIS (5Y)", "ois_fd_mc", "ois_aad_replay", "ois_aad_record"),
]


def _sp(nojit, jit):
    return f"{nojit / jit:.2f}×" if jit > 0 else "—"


def print_comparison(nojit: dict, jit: dict, repeats: int, wheels: dict):
    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  MC Scenario Risk Benchmark: OIS-Bootstrapped IRS")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  MC scenarios per batch : {N_SCENARIOS}")
    print(f"  Outer repeats          : {repeats}")
    print(f"  BPS shift (FD)         : {BPS}")
    print(f"  Non-JIT : {wheels['nojit']['ql'].name}")
    print(f"  JIT     : {wheels['jit']['ql'].name}")
    print(f"  Rate src: {_RATES_SOURCE}")
    print(f"  Eval dt : {_EVAL_DATE[2]}-{_EVAL_DATE[1]:02d}-{_EVAL_DATE[0]:02d}")
    print(f"  Fix rate: {_SWAP_FIXED_RATE*100:.2f}%")
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

MD_PATH = Path(__file__).resolve().parent / "ois_bootstrapped_IRS_benchmarks_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict,
                   rates: list, rates_source: str,
                   fallback_warning: str = None):
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — MC Scenario Risk Benchmark: OIS-Bootstrapped IRS")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**MC scenarios per batch:** {N_SCENARIOS}  ")
    w(f"**Outer repetitions:** {repeats} (median reported)  ")
    w(f"**Non-JIT wheel:** `{wheels['nojit']['ql'].name}`  ")
    w(f"**JIT wheel:** `{wheels['jit']['ql'].name}`  ")
    w(f"**Rate source:** {rates_source}  ")
    if fallback_warning:
        w(f"**⚠️ Live scraping failed:** {fallback_warning}  ")
        w("**Fallback:** hardcoded Nov 2024 SOFR OIS rates used instead  ")
    w("")
    w("---")
    w("")
    w("## Instrument")
    w("")
    w(f"- **5-year SOFR OIS** (pay fixed {_SWAP_FIXED_RATE*100:.2f}%, receive SOFR, $10M notional)")
    w("- OIS discount/forecasting curve: `PiecewiseLogLinearDiscount` bootstrapped")
    w("  from 9 `OISRateHelper` instruments using the `Sofr` overnight index")
    w("- Engine: `DiscountingSwapEngine` discounting off the OIS curve")
    w("")
    w(f"### Market data ({rates_source})")
    w("")
    w("| Tenor | Rate |")
    w("|-------|-----:|")
    for lbl, r in zip(_OIS_TENOR_LABELS, rates):
        w(f"| {lbl} | {r*100:.2f}% |")
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
    w("- By default, rates are scraped live from the **US Treasury daily par yield curve**")
    w("  at treasury.gov.  If scraping fails, the script warns and falls back to")
    w("  hardcoded Nov 2024 SOFR OIS rates.  Use `--offline` to skip scraping.")
    w("- Treasury par yields are a close proxy for SOFR OIS swap rates;")
    w("  the AD performance comparison is valid with either source.")
    w("- The OIS curve bootstrap via `PiecewiseLogLinearDiscount` and")
    w("  `DiscountingSwapEngine` are **straight-line arithmetic** with no data-dependent")
    w("  branching on `Real`, making this pipeline fully JIT-compatible.")
    w("- The same OIS curve serves as both the discounting and forecasting curve,")
    w("  consistent with single-curve SOFR pricing methodology.")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("")
    w("python benchmarks/ois_bootstrapped_IRS_benchmarks.py            # live rates (default)")
    w("python benchmarks/ois_bootstrapped_IRS_benchmarks.py --offline  # hardcoded Nov 2024")
    w("python benchmarks/ois_bootstrapped_IRS_benchmarks.py --repeats 10")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks OIS-bootstrapped IRS benchmark",
    )
    parser.add_argument("--worker", metavar="REPEATS", type=int, default=None,
                        help="Internal worker mode: run benchmarks and print JSON")
    parser.add_argument("--market-data", type=str, default=None,
                        help="JSON string with {rates, source, date} for worker")
    parser.add_argument("--repeats", "-r", type=int, default=5,
                        help=f"Outer repetitions per batch of {N_SCENARIOS} scenarios "
                             "(default: 5)")
    parser.add_argument("--offline", action="store_true",
                        help="Skip live scraping; use hardcoded Nov 2024 rates")
    parser.add_argument("--clean-venvs", action="store_true",
                        help="Destroy and recreate benchmark venvs")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write results to markdown file")
    args = parser.parse_args()

    if args.worker is not None:
        worker_main(args.worker, args.market_data)
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

    # --- Resolve market data -------------------------------------------------
    market_data = None       # None → worker uses module-level hardcoded rates
    fallback_warning = None  # set if live scraping was attempted but failed
    if not args.offline:
        try:
            rates, source, date_str = scrape_live_rates()
            _set_active_rates(rates, source, date_str)
            market_data = {"rates": rates, "source": source, "date": date_str}
            print(f"  Live rates loaded: {source}")
            for lbl, r in zip(_OIS_TENOR_LABELS, rates):
                print(f"    {lbl:>4s}: {r*100:.2f}%")
        except Exception as exc:
            fallback_warning = str(exc)
            print(f"  WARNING: live scraping failed: {exc}")
            print(f"           falling back to hardcoded Nov 2024 rates")
    else:
        print(f"  --offline: using hardcoded Nov 2024 rates")

    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  MC Scenario Risk Benchmark: OIS-Bootstrapped IRS")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"  {N_SCENARIOS} scenarios per batch, {repeats} outer repetitions")
    print(f"  Rate source: {_RATES_SOURCE}")
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
    nojit = run_worker_in_venv(VENV_NOJIT, repeats, market_data)
    print("        done.")
    print("\n  [2/2] JIT worker …")
    jit = run_worker_in_venv(VENV_JIT, repeats, market_data)
    print("        done.")

    print_comparison(nojit, jit, repeats, wheels)

    if not args.no_save:
        write_markdown(nojit, jit, repeats, wheels,
                       _ACTIVE_RATES, _RATES_SOURCE, fallback_warning)


if __name__ == "__main__":
    main()

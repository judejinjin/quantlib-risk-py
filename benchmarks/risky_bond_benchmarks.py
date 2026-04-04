#!/usr/bin/env python3
"""
QuantLib-Risks-Py — Risky Bond Benchmark (OIS-Bootstrapped + CDS Credit Curve)
================================================================================

Benchmarks first-order sensitivities for a 5Y fixed-rate coupon bond priced
with the ``RiskyBondEngine`` against realistic, bootstrapped curves:

  • **Interest-rate risk** — 9-point OIS curve bootstrapped from SOFR par rates
    (live-scraped from US Treasury or hardcoded fallback)
  • **Credit risk** — 4-point hazard-rate curve bootstrapped from hypothetical
    CDS spread quotes via ``SpreadCdsHelper`` + ``PiecewiseFlatHazardRate``
  • **Recovery risk** — scalar recovery rate (0.40)

This gives **14 inputs** total (9 OIS + 4 CDS + 1 recovery).

Three methods are compared:

  • **FD** — bump-and-reprice each of the 14 inputs by 1 bp
  • **AAD** — XAD reverse-mode tape; one backward sweep gives all 14 Greeks
  • **AAD + JIT** — same tape compiled to native code via XAD-Forge

The ``RiskyBondEngine``'s ``calculate()`` branches only on dates (not on AReal
inputs), and the OIS/CDS bootstrap pipelines are straight-line arithmetic for
a given convergence path → **JIT eligible**.

Instrument
----------
  5Y Fixed-rate coupon bond, semiannual, 100 notional, 5% coupon
  Priced with survival-weighted discounting + recovery-on-default term

Market data  (14 inputs)
---------
  OIS rates (9):      1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 30Y
  CDS spreads (4):    1Y=50bp, 2Y=75bp, 3Y=100bp, 5Y=125bp
  Recovery (1):       40%

Usage
-----
  python benchmarks/risky_bond_benchmarks.py                    # live rates
  python benchmarks/risky_bond_benchmarks.py --offline          # hardcoded
  python benchmarks/risky_bond_benchmarks.py --repeats 50
  python benchmarks/risky_bond_benchmarks.py --worker REPEATS --market-data '{...}'
"""

import argparse
import csv
import datetime
import io
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
BPS = 1e-4   # 1 basis-point shift for FD

# ---------------------------------------------------------------------------
# OIS rate tenors & hardcoded fallback (Nov 2024 SOFR OIS snapshot)
# ---------------------------------------------------------------------------
_OIS_TENOR_LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "10Y", "30Y"]
_OIS_TENORS = [
    (1,  "Months"),
    (3,  "Months"),
    (6,  "Months"),
    (1,  "Years"),
    (2,  "Years"),
    (3,  "Years"),
    (5,  "Years"),
    (10, "Years"),
    (30, "Years"),
]
_OIS_FALLBACK_RATES = [
    0.0483,   # 1M
    0.0455,   # 3M
    0.0425,   # 6M
    0.0357,   # 1Y
    0.0340,   # 2Y
    0.0335,   # 3Y
    0.0350,   # 5Y
    0.0385,   # 10Y
    0.0415,   # 30Y
]

# ---------------------------------------------------------------------------
# CDS spread tenors & hypothetical quotes (investment-grade issuer)
# ---------------------------------------------------------------------------
_CDS_TENOR_LABELS = ["CDS 1Y", "CDS 2Y", "CDS 3Y", "CDS 5Y"]
_CDS_TENORS = [
    (1,  "Years"),
    (2,  "Years"),
    (3,  "Years"),
    (5,  "Years"),
]
_CDS_BASE_SPREADS = [0.0050, 0.0075, 0.0100, 0.0125]   # 50–125 bp

# Bond parameters
_COUPON   = 0.05
_NOTIONAL = 100.0
_MATURITY = "5Y"
_RECOVERY = 0.40

# All input labels (9 OIS + 4 CDS + 1 recovery = 14)
_INPUT_NAMES = (
    [f"OIS {l}" for l in _OIS_TENOR_LABELS]
    + _CDS_TENOR_LABELS
    + ["Recovery"]
)

# Mutable module-level state (overwritten by --live / --market-data)
_ACTIVE_OIS = list(_OIS_FALLBACK_RATES)
_ACTIVE_CDS = list(_CDS_BASE_SPREADS)
_ACTIVE_REC = _RECOVERY
_RATES_SOURCE = "hardcoded Nov 2024 SOFR OIS snapshot"
_RATES_DATE   = "2024-11-15"
_EVAL_DATE    = (15, 11, 2024)      # (day, month, year) for ql.Date


# ============================================================================
# Live-rate scraping (US Treasury daily par yield curve)
# ============================================================================

_TREASURY_COLUMN_MAP = {
    "1 Mo":  0,
    "3 Mo":  1,
    "6 Mo":  2,
    "1 Yr":  3,
    "2 Yr":  4,
    "3 Yr":  5,
    "5 Yr":  6,
    "10 Yr": 7,
    "30 Yr": 8,
}


def scrape_live_rates() -> tuple:
    """Scrape the latest US Treasury par yield curve from treasury.gov.

    Returns (rates_9, source_label, date_str).
    """
    import requests

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

    latest = rows[0]
    date_raw = latest.get("Date", "unknown")

    rates = [None] * 9
    for col_hdr, idx in _TREASURY_COLUMN_MAP.items():
        val = latest.get(col_hdr, "").strip()
        if val:
            rates[idx] = float(val) / 100.0

    missing = [_OIS_TENOR_LABELS[i] for i in range(9) if rates[i] is None]
    if missing:
        raise RuntimeError(f"Treasury CSV missing tenors: {', '.join(missing)}")

    try:
        dt = datetime.datetime.strptime(date_raw, "%m/%d/%Y")
        date_str = dt.strftime("%Y-%m-%d")
    except ValueError:
        date_str = date_raw

    source = f"US Treasury daily par yield curve ({date_str})"
    return rates, source, date_str


def _set_active_rates(rates: list, source: str, date_str: str):
    """Replace the module-level active OIS rates and eval date."""
    global _ACTIVE_OIS, _RATES_SOURCE, _RATES_DATE, _EVAL_DATE
    _ACTIVE_OIS   = list(rates)
    _RATES_SOURCE = source
    _RATES_DATE   = date_str
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        _EVAL_DATE = (dt.day, dt.month, dt.year)
    except ValueError:
        pass


def _all_input_vals():
    """Return the full 14-element input vector."""
    return _ACTIVE_OIS + _ACTIVE_CDS + [_ACTIVE_REC]


# ============================================================================
# Wheel / venv helpers  (dual-venv pattern)
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
    print("    \u2022 pip install setuptools wheel")
    subprocess.check_call([py, "-m", "pip", "install", "--quiet", "setuptools", "wheel"])
    print(f"    \u2022 pip install {xad_wheel.name}")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps",
         str(xad_wheel)]
    )
    print("    \u2022 pip install xad compatibility shim")
    _install_xad_shim(py)
    print(f"    \u2022 pip install {ql_wheel.name}")
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
# WORKER MODE  –  benchmark implementations
# ============================================================================

def _median_ms(func, n: int, warmup: int = 5):
    """Return (median_ms, stdev_ms) over n timed calls, after warmup."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ---- Build helpers ----------------------------------------------------------

def _build_bond_plain():
    """Build risky bond with plain-float bootstrapped curves.

    Returns (bond, all_quotes, rfHandle, defHandle)
      all_quotes: list of 14 SimpleQuote objects
        [0..8]   = 9 OIS rate quotes
        [9..12]  = 4 CDS spread quotes
        [13]     = 1 recovery quote  (bumped via engine rebuild)
    """
    import QuantLib_Risks as ql

    todaysDate = ql.Date(_EVAL_DATE[0], _EVAL_DATE[1], _EVAL_DATE[2])
    ql.Settings.instance().evaluationDate = todaysDate
    calendar = ql.TARGET()

    # --- OIS curve (9 inputs) ---
    ois_quotes = [ql.SimpleQuote(r) for r in _ACTIVE_OIS]
    sofr = ql.Sofr()
    ois_helpers = []
    for i, (n, unit) in enumerate(_OIS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        helper = ql.OISRateHelper(
            2, period, ql.QuoteHandle(ois_quotes[i]), sofr)
        ois_helpers.append(helper)

    ois_curve = ql.PiecewiseLogLinearDiscount(
        todaysDate, ois_helpers, ql.Actual365Fixed())
    ois_curve.enableExtrapolation()
    rfHandle = ql.YieldTermStructureHandle(ois_curve)

    # --- CDS hazard curve (4 inputs) ---
    cds_quotes = [ql.SimpleQuote(s) for s in _ACTIVE_CDS]
    cds_helpers = []
    for sq, (n, unit) in zip(cds_quotes, _CDS_TENORS):
        tenor = ql.Period(n, getattr(ql, unit))
        h = ql.SpreadCdsHelper(
            ql.QuoteHandle(sq), tenor, 0, calendar, ql.Quarterly,
            ql.Following, ql.DateGeneration.TwentiethIMM,
            ql.Actual365Fixed(), _ACTIVE_REC, rfHandle)
        cds_helpers.append(h)

    hazard_curve = ql.PiecewiseFlatHazardRate(
        todaysDate, cds_helpers, ql.Actual365Fixed())
    hazard_curve.enableExtrapolation()
    defHandle = ql.DefaultProbabilityTermStructureHandle(hazard_curve)

    # --- Bond ---
    schedule = ql.Schedule(
        todaysDate, todaysDate + ql.Period(_MATURITY),
        ql.Period(ql.Semiannual),
        calendar, ql.Following, ql.Following,
        ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(2, _NOTIONAL, schedule, [_COUPON],
                            ql.Actual365Fixed())
    bond.setPricingEngine(
        ql.RiskyBondEngine(defHandle, _ACTIVE_REC, rfHandle))

    recovery_quote = ql.SimpleQuote(_ACTIVE_REC)
    all_quotes = ois_quotes + cds_quotes + [recovery_quote]
    return bond, all_quotes, rfHandle, defHandle


def _build_bond_aad():
    """Build risky bond with AAD tape for all 14 inputs.

    Returns (tape, bond, all_inputs)
      all_inputs: list of 14 xad Reals
        [0..8]   = 9 OIS rate inputs
        [9..12]  = 4 CDS spread inputs
        [13]     = 1 recovery input
    """
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape, Real

    tape = Tape()
    tape.activate()

    todaysDate = ql.Date(_EVAL_DATE[0], _EVAL_DATE[1], _EVAL_DATE[2])
    ql.Settings.instance().evaluationDate = todaysDate
    calendar = ql.TARGET()

    # Register all 14 inputs
    ois_r = [Real(r) for r in _ACTIVE_OIS]
    cds_r = [Real(s) for s in _ACTIVE_CDS]
    rec_r = Real(_ACTIVE_REC)
    all_inputs = ois_r + cds_r + [rec_r]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    # --- OIS curve ---
    sofr = ql.Sofr()
    ois_helpers = []
    for i, (n, unit) in enumerate(_OIS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        helper = ql.OISRateHelper(
            2, period, ql.QuoteHandle(ql.SimpleQuote(ois_r[i])), sofr)
        ois_helpers.append(helper)

    ois_curve = ql.PiecewiseLogLinearDiscount(
        todaysDate, ois_helpers, ql.Actual365Fixed())
    ois_curve.enableExtrapolation()
    rfHandle = ql.YieldTermStructureHandle(ois_curve)

    # --- CDS hazard curve ---
    cds_helpers = []
    for sr, (n, unit) in zip(cds_r, _CDS_TENORS):
        tenor = ql.Period(n, getattr(ql, unit))
        h = ql.SpreadCdsHelper(
            ql.QuoteHandle(ql.SimpleQuote(sr)), tenor, 0, calendar, ql.Quarterly,
            ql.Following, ql.DateGeneration.TwentiethIMM,
            ql.Actual365Fixed(), rec_r, rfHandle)
        cds_helpers.append(h)

    hazard_curve = ql.PiecewiseFlatHazardRate(
        todaysDate, cds_helpers, ql.Actual365Fixed())
    hazard_curve.enableExtrapolation()
    defHandle = ql.DefaultProbabilityTermStructureHandle(hazard_curve)

    # --- Bond ---
    schedule = ql.Schedule(
        todaysDate, todaysDate + ql.Period(_MATURITY),
        ql.Period(ql.Semiannual),
        calendar, ql.Following, ql.Following,
        ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(2, _NOTIONAL, schedule, [_COUPON],
                            ql.Actual365Fixed())
    bond.setPricingEngine(
        ql.RiskyBondEngine(defHandle, rec_r, rfHandle))

    return tape, bond, all_inputs


# ---- Worker entry point -----------------------------------------------------

def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    import xad
    V = lambda x: float(xad.value(x))
    results = {}
    input_vals = _all_input_vals()
    n = len(input_vals)
    results["n_inputs"] = n

    # ---- plain / FD ----
    bond_plain, all_quotes, rfHandle, defHandle = _build_bond_plain()

    base_npv = V(bond_plain.NPV())
    results["npv"] = base_npv

    # FD Greeks: bump each of 14 inputs
    # For inputs 0..12 (OIS + CDS quotes): setValue bump works via observer chain
    # For input 13 (recovery): full rebuild required because recovery enters both
    #   the CDS bootstrap (SpreadCdsHelper) and the RiskyBondEngine constructor
    calendar = ql.TARGET()
    todaysDate = ql.Settings.instance().evaluationDate
    fd_greeks = []
    for i, (q, v0) in enumerate(zip(all_quotes, input_vals)):
        if i < 13:
            # OIS or CDS quote — observer chain works
            q.setValue(v0 + BPS)
            npv_up = V(bond_plain.NPV())
            q.setValue(v0)
        else:
            # Recovery — full rebuild (CDS helpers + hazard curve + engine)
            rec_bumped = v0 + BPS
            cds_helpers_b = []
            cds_quotes_b = [ql.SimpleQuote(s) for s in _ACTIVE_CDS]
            for sq_b, (nn, uu) in zip(cds_quotes_b, _CDS_TENORS):
                tenor_b = ql.Period(nn, getattr(ql, uu))
                hh = ql.SpreadCdsHelper(
                    ql.QuoteHandle(sq_b), tenor_b, 0, calendar, ql.Quarterly,
                    ql.Following, ql.DateGeneration.TwentiethIMM,
                    ql.Actual365Fixed(), rec_bumped, rfHandle)
                cds_helpers_b.append(hh)
            haz_b = ql.PiecewiseFlatHazardRate(
                todaysDate, cds_helpers_b, ql.Actual365Fixed())
            haz_b.enableExtrapolation()
            def_b = ql.DefaultProbabilityTermStructureHandle(haz_b)
            bond_plain.setPricingEngine(
                ql.RiskyBondEngine(def_b, rec_bumped, rfHandle))
            npv_up = V(bond_plain.NPV())
            # restore original engine
            bond_plain.setPricingEngine(
                ql.RiskyBondEngine(defHandle, v0, rfHandle))
        fd_greeks.append((npv_up - base_npv) / BPS)
    results["fd_greeks"] = fd_greeks

    # ---- Plain timing (single NPV) ----
    def _plain():
        all_quotes[0].setValue(_ACTIVE_OIS[0] + 1e-10)
        all_quotes[0].setValue(_ACTIVE_OIS[0])
        return bond_plain.NPV()

    m, s = _median_ms(_plain, repeats)
    results["plain"] = {"median": m, "stdev": s}

    # ---- FD timing (N+1 pricings) ----
    def _fd():
        bond_plain.NPV()
        for i, (q, v0) in enumerate(zip(all_quotes, input_vals)):
            if i < 13:
                q.setValue(v0 + BPS)
                bond_plain.NPV()
                q.setValue(v0)
            else:
                rec_b = v0 + BPS
                cds_h_b = []
                cds_q_b = [ql.SimpleQuote(s) for s in _ACTIVE_CDS]
                for sq_b, (nn, uu) in zip(cds_q_b, _CDS_TENORS):
                    tn = ql.Period(nn, getattr(ql, uu))
                    hh = ql.SpreadCdsHelper(
                        ql.QuoteHandle(sq_b), tn, 0, calendar, ql.Quarterly,
                        ql.Following, ql.DateGeneration.TwentiethIMM,
                        ql.Actual365Fixed(), rec_b, rfHandle)
                    cds_h_b.append(hh)
                haz_b = ql.PiecewiseFlatHazardRate(
                    todaysDate, cds_h_b, ql.Actual365Fixed())
                haz_b.enableExtrapolation()
                def_b = ql.DefaultProbabilityTermStructureHandle(haz_b)
                bond_plain.setPricingEngine(
                    ql.RiskyBondEngine(def_b, rec_b, rfHandle))
                bond_plain.NPV()
                bond_plain.setPricingEngine(
                    ql.RiskyBondEngine(defHandle, v0, rfHandle))

    m, s = _median_ms(_fd, repeats)
    results["fd"] = {"median": m, "stdev": s}

    # ---- AAD ----
    tape, bond_aad, inputs = _build_bond_aad()
    npv_aad = bond_aad.NPV()
    tape.registerOutput(npv_aad)
    results["aad_npv"] = V(npv_aad)

    tape.clearDerivatives()
    npv_aad.derivative = 1.0
    tape.computeAdjoints()
    aad_greeks = [V(xad.derivative(inp)) for inp in inputs]
    results["aad_greeks"] = aad_greeks

    def _aad():
        tape.clearDerivatives()
        npv_aad.derivative = 1.0
        tape.computeAdjoints()

    m, s = _median_ms(_aad, repeats)
    results["aad"] = {"median": m, "stdev": s}
    tape.deactivate()

    return results


def worker_main(repeats: int, market_data_json: str = None):
    if market_data_json:
        md = json.loads(market_data_json)
        _set_active_rates(md["rates"], md["source"], md["date"])
    print(f"Worker: eval_date={_EVAL_DATE}, source={_RATES_SOURCE}",
          file=sys.stderr)
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison output
# ============================================================================

def _fmt_t(median, stdev):
    return f"{median:8.4f} \u00b1{stdev:6.4f} ms"


def _sp(a, b):
    return f"{a / b:6.2f}x" if b > 0 else "   N/A"


def print_comparison(nojit: dict, jit: dict, repeats: int):
    print()
    print(SEPARATOR)
    print("Risky Bond  \u2013  OIS + CDS Bootstrapped  \u2013  FD vs AAD vs AAD+JIT")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Instrument  : {_MATURITY} Fixed-Rate Coupon Bond ({_COUPON*100:.0f}% "
          f"semiannual, {_NOTIONAL:.0f} notional)")
    print(f"  OIS curve   : 9-point SOFR OIS ({_RATES_SOURCE})")
    print(f"  Credit curve: 4-point CDS-bootstrapped hazard "
          f"(1Y={_ACTIVE_CDS[0]*1e4:.0f}bp, 2Y={_ACTIVE_CDS[1]*1e4:.0f}bp, "
          f"3Y={_ACTIVE_CDS[2]*1e4:.0f}bp, 5Y={_ACTIVE_CDS[3]*1e4:.0f}bp)")
    print(f"  Recovery    : {_ACTIVE_REC*100:.0f}%")
    print(f"  Engine      : RiskyBondEngine  (survival-weighted discounting)")
    print(f"  JIT         : Eligible  (date-only branching)")
    print(f"  Repeats     : {repeats}")
    print(f"  BPS shift   : {BPS}")
    n = nojit["n_inputs"]
    print(f"  Inputs      : {n}  (9 OIS + 4 CDS + 1 recovery)")
    print()

    # NPV
    print(f"  NPV (FD build)  : {nojit['npv']:.10f}")
    print(f"  NPV (AAD build) : {nojit['aad_npv']:.10f}")
    print()

    # Greeks
    print("  Greeks comparison (AAD vs FD):")
    print(f"    {'Input':<16s}  {'FD':>14s}  {'AAD':>14s}  {'|\u0394|':>12s}")
    print("    " + "-" * 60)
    for i, name in enumerate(_INPUT_NAMES):
        fd_g  = nojit["fd_greeks"][i]
        aad_g = nojit["aad_greeks"][i]
        diff  = abs(fd_g - aad_g)
        print(f"    {name:<16s}  {fd_g:14.6f}  {aad_g:14.6f}  {diff:12.2e}")
    print()

    # Timing
    col = 22
    hdr = (f"  {'Method':<28s}  "
           f"{'Non-JIT':>{col}}  "
           f"{'JIT':>{col}}  "
           f"{'JIT speedup':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, key in [
        ("Plain pricing (1 NPV)",      "plain"),
        ("Bump-and-reprice FD (N+1)",   "fd"),
        ("AAD backward pass",           "aad"),
    ]:
        nj = nojit[key]
        jt = jit[key]
        sp = _sp(nj["median"], jt["median"])
        print(f"    {label:<26s}  "
              f"{_fmt_t(nj['median'], nj['stdev']):>{col}}  "
              f"{_fmt_t(jt['median'], jt['stdev']):>{col}}  "
              f"{sp:>11}")

    fd_aad_nojit = nojit["fd"]["median"] / nojit["aad"]["median"] if nojit["aad"]["median"] else 0
    fd_aad_jit   = jit["fd"]["median"]   / jit["aad"]["median"]   if jit["aad"]["median"]   else 0
    print()
    print(f"  FD \u00f7 AAD ratio:  Non-JIT {fd_aad_nojit:.1f}x  |  JIT {fd_aad_jit:.1f}x")
    print()
    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "risky_bond_benchmark_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append
    n = nojit["n_inputs"]

    w("# Risky Bond \u2014 OIS + CDS Bootstrapped Benchmark Results")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**Repetitions:** {repeats} (median reported)  ")
    w(f"**OIS rates:** {_RATES_SOURCE}  ")
    w(f"**Non-JIT wheel:** `{wheels['nojit']['ql'].name}`  ")
    w(f"**JIT wheel:** `{wheels['jit']['ql'].name}`  ")
    w("")
    w("---")
    w("")
    w("## Instrument")
    w("")
    w("| Parameter | Value |")
    w("|---|---|")
    w(f"| Type | {_MATURITY} Fixed-Rate Coupon Bond |")
    w(f"| Notional | {_NOTIONAL:.0f} |")
    w(f"| Coupon | {_COUPON*100:.0f}% semiannual |")
    w(f"| Day counter | Actual/365 Fixed |")
    w(f"| Engine | `RiskyBondEngine` (survival-weighted discounting) |")
    w(f"| JIT eligible | **Yes** \u2014 branching on dates only, not on AReal inputs |")
    w("")
    w("### Interest-rate curve (OIS-bootstrapped)")
    w("")
    w("| Tenor | Rate |")
    w("|---|---:|")
    for i, label in enumerate(_OIS_TENOR_LABELS):
        w(f"| {label} | {_ACTIVE_OIS[i]*100:.2f}% |")
    w("")
    w(f"Bootstrap: `OISRateHelper` \u2192 `PiecewiseLogLinearDiscount` "
      f"(SOFR index, {_RATES_SOURCE})")
    w("")
    w("### Credit curve (CDS-bootstrapped)")
    w("")
    w("| Tenor | CDS Spread |")
    w("|---|---:|")
    for i, label in enumerate(_CDS_TENOR_LABELS):
        w(f"| {label} | {_ACTIVE_CDS[i]*1e4:.0f} bp |")
    w("")
    w(f"Recovery: {_ACTIVE_REC*100:.0f}%  ")
    w(f"Bootstrap: `SpreadCdsHelper` \u2192 `PiecewiseFlatHazardRate`")
    w("")
    w("---")
    w("")
    w("## Pricing formula")
    w("")
    w("The `RiskyBondEngine` computes the risky NPV as:")
    w("")
    w("$$")
    w("\\text{NPV} = \\sum_i CF_i \\cdot P(0, T_i) \\cdot Q(T_i)")
    w("+ R \\sum_i N(T_i^{\\text{mid}}) \\cdot P(0, T_i^{\\text{mid}})")
    w("\\cdot [Q(T_{i-1}) - Q(T_i)]")
    w("$$")
    w("")
    w("where $P(0, t)$ is the OIS discount factor, $Q(t)$ is the CDS-implied")
    w("survival probability, $R$ is the recovery rate, and $N(t)$ is the notional.")
    w("")
    w("---")
    w("")
    w("## Greeks validation (AAD vs FD)")
    w("")
    w(f"NPV = {nojit['npv']:.10f}")
    w("")
    w("### Interest-rate sensitivities (OIS)")
    w("")
    w("| Input | FD (1 bp) | AAD | \\|\u0394\\| |")
    w("|---|---:|---:|---:|")
    for i in range(9):
        name  = _INPUT_NAMES[i]
        fd_g  = nojit["fd_greeks"][i]
        aad_g = nojit["aad_greeks"][i]
        diff  = abs(fd_g - aad_g)
        w(f"| {name} | {fd_g:.6f} | {aad_g:.6f} | {diff:.2e} |")
    w("")
    w("### Credit sensitivities (CDS spreads + recovery)")
    w("")
    w("| Input | FD (1 bp) | AAD | \\|\u0394\\| |")
    w("|---|---:|---:|---:|")
    for i in range(9, n):
        name  = _INPUT_NAMES[i]
        fd_g  = nojit["fd_greeks"][i]
        aad_g = nojit["aad_greeks"][i]
        diff  = abs(fd_g - aad_g)
        w(f"| {name} | {fd_g:.6f} | {aad_g:.6f} | {diff:.2e} |")
    w("")
    w("---")
    w("")
    w("## Timing results")
    w("")
    w(f"N = {n} market inputs (9 OIS + 4 CDS + 1 recovery), "
      f"{repeats} repetitions, BPS = {BPS}")
    w("")
    w("| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |")
    w("|---|---:|---:|---:|")
    for label, key in [
        ("Plain pricing (1 NPV)", "plain"),
        ("Bump-and-reprice FD (N+1 NPVs)", "fd"),
        ("**AAD backward pass**", "aad"),
    ]:
        nj = nojit[key]
        jt = jit[key]
        sp = f"{nj['median'] / jt['median']:.2f}\u00d7" if jt["median"] > 0 else "\u2014"
        w(f"| {label} | {nj['median']:.4f} \u00b1{nj['stdev']:.4f} "
          f"| {jt['median']:.4f} \u00b1{jt['stdev']:.4f} | {sp} |")

    fd_aad_nojit = nojit["fd"]["median"] / nojit["aad"]["median"] if nojit["aad"]["median"] else 0
    fd_aad_jit   = jit["fd"]["median"]   / jit["aad"]["median"]   if jit["aad"]["median"]   else 0
    w(f"| *FD \u00f7 AAD* | *{fd_aad_nojit:.1f}\u00d7* | *{fd_aad_jit:.1f}\u00d7* | \u2014 |")
    w("")
    w("---")
    w("")
    w("## Analysis")
    w("")
    w("This benchmark demonstrates AAD applied to a **realistic credit-risky bond**")
    w("priced against production-grade bootstrapped curves:")
    w("")
    w("- **Interest-rate curve**: 9 SOFR OIS par rates bootstrapped via")
    w("  `OISRateHelper` + `PiecewiseLogLinearDiscount` (live-scraped from")
    w("  US Treasury or hardcoded fallback)")
    w("- **Credit curve**: 4 CDS spread quotes bootstrapped via")
    w("  `SpreadCdsHelper` + `PiecewiseFlatHazardRate`")
    w("- **Recovery rate**: scalar input to `RiskyBondEngine`")
    w("")
    w(f"With {n} inputs, FD requires {n}+1 = {n+1} full pricings (each involving")
    w(f"curve re-bootstrap), while AAD needs only 1 backward sweep on the pre-recorded")
    w(f"tape.  The AAD advantage is amplified by the bootstrap cost and grows with N.")
    w("")
    w("All branching in `RiskyBondEngine::calculate()` is on dates (not AReal),")
    w("making the tape structure fixed for a given bond schedule \u2192 **JIT eligible**.")
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
    w("# Run with live Treasury rates")
    w("python benchmarks/risky_bond_benchmarks.py")
    w("")
    w("# Run with hardcoded rates (offline)")
    w("python benchmarks/risky_bond_benchmarks.py --offline")
    w("python benchmarks/risky_bond_benchmarks.py --repeats 50")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Risky bond (OIS + CDS bootstrapped) FD vs AAD vs AAD+JIT benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--worker", metavar="REPEATS", type=int, default=None,
        help="Internal worker mode: run benchmarks and print JSON",
    )
    parser.add_argument(
        "--market-data", type=str, default=None,
        help="JSON market data (internal, passed by orchestrator)",
    )
    parser.add_argument(
        "--repeats", "-r", type=int, default=30,
        help="Number of repetitions per benchmark (default: 30)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use hardcoded rates only (skip live scraping)",
    )
    parser.add_argument(
        "--clean-venvs", action="store_true",
        help="Destroy and recreate benchmark venvs before running",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write results markdown",
    )
    args = parser.parse_args()

    if args.worker is not None:
        worker_main(args.worker, args.market_data)
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
    print("Risky Bond  \u2013  OIS + CDS Bootstrapped  \u2013  FD vs AAD vs AAD+JIT")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)

    # --- Scrape live rates (or use hardcoded) ---
    market_data = None
    if not args.offline:
        try:
            rates, source, date_str = scrape_live_rates()
            _set_active_rates(rates, source, date_str)
            market_data = {"rates": rates, "source": source, "date": date_str}
            print(f"  \u2713 Live rates: {source}")
        except Exception as e:
            print(f"  \u26a0 Live scraping failed: {e}")
            print(f"    Falling back to hardcoded rates ({_RATES_SOURCE})")
    else:
        print(f"  --offline: using {_RATES_SOURCE}")

    for i, label in enumerate(_OIS_TENOR_LABELS):
        print(f"    {label:>4s}  {_ACTIVE_OIS[i]*100:6.2f}%")
    print()
    print("  CDS spreads (hypothetical):")
    for i, label in enumerate(_CDS_TENOR_LABELS):
        print(f"    {label}  {_ACTIVE_CDS[i]*1e4:5.0f} bp")
    print(f"    Recovery  {_ACTIVE_REC*100:.0f}%")
    print()

    # ---- set up venvs ----
    print("Setting up virtual environments")
    print("-" * 50)
    print(f"\n[1/2] Non-JIT venv  ({VENV_NOJIT.name})")
    setup_venv(VENV_NOJIT, wheels["nojit"]["xad"], wheels["nojit"]["ql"],
               force=args.clean_venvs)
    print(f"\n[2/2] JIT venv      ({VENV_JIT.name})")
    setup_venv(VENV_JIT, wheels["jit"]["xad"], wheels["jit"]["ql"],
               force=args.clean_venvs)

    # ---- run workers ----
    print(f"\nRunning benchmarks  ({repeats} repeats per build)")
    print("-" * 50)
    print("\n  [1/2] Non-JIT worker \u2026")
    nojit = run_worker_in_venv(VENV_NOJIT, repeats, market_data)
    print("        done.")
    print("\n  [2/2] JIT worker \u2026")
    jit = run_worker_in_venv(VENV_JIT, repeats, market_data)
    print("        done.")

    print_comparison(nojit, jit, repeats)

    if not args.no_save:
        write_markdown(nojit, jit, repeats, wheels)


if __name__ == "__main__":
    main()

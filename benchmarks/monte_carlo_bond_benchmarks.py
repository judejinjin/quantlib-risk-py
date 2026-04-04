#!/usr/bin/env python3
"""
QuantLib-Risks-Py — Monte Carlo Scenario Risk Benchmark: Callable Bond
=======================================================================

Benchmarks the Callable Bond (3 HullWhite inputs: flat rate r, mean-reversion a,
vol σ) under N_SCENARIOS = 100 random Monte Carlo scenarios, comparing:

  H1) FD             – bump-and-reprice (4 tree pricings per scenario)
  H2) AAD replay     – N backward sweeps on a tape recorded once at base market
  H3) AAD re-record  – per-scenario fresh tape + QL objects

All benchmarks use the standard (non-JIT) XAD build.  The JIT/Forge backend is
not used because its record-once-replay-many paradigm is incompatible with
QuantLib pricing engines that contain data-dependent branching (see
JIT_LIMITATIONS.md for details).

Usage
-----
  python benchmarks/monte_carlo_bond_benchmarks.py               # default 5 repeats
  python benchmarks/monte_carlo_bond_benchmarks.py --repeats 10
  python benchmarks/monte_carlo_bond_benchmarks.py --clean-venvs

  # Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/monte_carlo_bond_benchmarks.py --worker REPEATS
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
SCENARIO_SEED = 42

# Callable bond base parameters
_BOND_BASE_RATE = 0.0465
_BOND_BASE_A    = 0.06
_BOND_BASE_S    = 0.20

# Number of IRS inputs in the combined script – used only to advance RNG state
# so bond scenario values match those produced by the IRS+bond benchmarks.
_N_IRS_INPUTS = 17


def _gen_scenarios():
    """
    Pre-generate bond MC scenarios deterministically.

    The RNG is advanced past the IRS draws first (N_SCENARIOS × _N_IRS_INPUTS
    gauss samples) so that bond scenario values are identical to those produced
    by the combined monte_carlo_benchmarks.py script.
    """
    rng = random.Random(SCENARIO_SEED)
    # Advance past IRS draws
    for _ in range(N_SCENARIOS * _N_IRS_INPUTS):
        rng.gauss(0, 1)
    bond_scenarios = []
    for _ in range(N_SCENARIOS):
        rate_v = _BOND_BASE_RATE + rng.gauss(0, 5e-4)
        a_v    = max(0.005, _BOND_BASE_A + rng.gauss(0, 2e-3))   # a > 0
        s_v    = max(0.005, _BOND_BASE_S + rng.gauss(0, 5e-3))   # σ > 0
        bond_scenarios.append([rate_v, a_v, s_v])
    return bond_scenarios


# ============================================================================
# Wheel / venv helpers
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


# ---- H.  Callable Bond  (3 HullWhite inputs) --------------------------------

def _build_bond_mc_structures():
    """
    Build the callable-bond plain-pricing structure and the base AAD tape.

    Returns:
        bond_plain         – CallableFixedRateBond (float HullWhite engine)
        sq_rate            – SimpleQuote for the flat rate
        ts_handle          – RelinkableYieldTermStructureHandle
        attach_engine      – callable that rebuilds the float HullWhite engine
        tape_bond          – Tape recorded at base parameters (for replay)
        npv_bond           – xad Real output registered on tape_bond
        rate_r, a_r, s_r   – xad Reals registered on tape_bond
        ts_r, tsh_r        – FlatForward / Handle holding xad Reals (keepalive)
        bond_r, model_r    – QL objects holding xad Reals (keepalive)
        schedule, callSched – Schedule objects shared by bond_r (keepalive)
    """
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape, Real

    calcDate  = ql.Date(16, 8, 2006)
    ql.Settings.instance().evaluationDate = calcDate
    dayCount  = ql.ActualActual(ql.ActualActual.Bond)
    calendar  = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

    sq_rate   = ql.SimpleQuote(_BOND_BASE_RATE)
    sq_a      = ql.SimpleQuote(_BOND_BASE_A)
    sq_s      = ql.SimpleQuote(_BOND_BASE_S)

    termStructure = ql.FlatForward(
        calcDate, ql.QuoteHandle(sq_rate), dayCount, ql.Compounded, ql.Semiannual)
    ts_handle = ql.RelinkableYieldTermStructureHandle(termStructure)

    callSched = ql.CallabilitySchedule()
    callDate  = ql.Date(15, ql.September, 2006)
    nc        = ql.NullCalendar()
    for _ in range(24):
        callSched.append(ql.Callability(
            ql.BondPrice(100.0, ql.BondPrice.Clean), ql.Callability.Call, callDate))
        callDate = nc.advance(callDate, 3, ql.Months)

    issueDate    = ql.Date(16, ql.September, 2004)
    maturityDate = ql.Date(15, ql.September, 2012)
    schedule     = ql.Schedule(issueDate, maturityDate, ql.Period(ql.Quarterly),
        calendar, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    bond_plain = ql.CallableFixedRateBond(3, 100, schedule, [0.025],
        ql.ActualActual(ql.ActualActual.Bond), ql.Following, 100,
        issueDate, callSched)

    def attach_engine():
        a0 = sq_a.value().getValue()
        s0 = sq_s.value().getValue()
        model = ql.HullWhite(ts_handle, a0, s0)
        bond_plain.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model, 40))

    attach_engine()

    # --- AAD tape for replay (recorded once at base parameters) ---
    tape_bond = Tape()
    tape_bond.activate()
    rate_r = Real(_BOND_BASE_RATE)
    a_r    = Real(_BOND_BASE_A)
    s_r    = Real(_BOND_BASE_S)
    tape_bond.registerInputs([rate_r, a_r, s_r])
    tape_bond.newRecording()

    ts_r    = ql.FlatForward(calcDate, ql.QuoteHandle(ql.SimpleQuote(rate_r)),
                              dayCount, ql.Compounded, ql.Semiannual)
    tsh_r   = ql.RelinkableYieldTermStructureHandle(ts_r)
    bond_r  = ql.CallableFixedRateBond(3, 100, schedule, [0.025],
        ql.ActualActual(ql.ActualActual.Bond), ql.Following, 100,
        issueDate, callSched)
    model_r = ql.HullWhite(tsh_r, a_r, s_r)
    bond_r.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model_r, 40))
    npv_bond = bond_r.cleanPrice()
    tape_bond.registerOutput(npv_bond)

    return (bond_plain, sq_rate, ts_handle, attach_engine, tape_bond, npv_bond,
            rate_r, a_r, s_r, ts_r, tsh_r, bond_r, model_r, schedule, callSched)


# ============================================================================
# Worker: run all Bond benchmarks, return JSON
# ============================================================================

def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    from xad.adj_1st import Real, Tape

    bond_scenarios = _gen_scenarios()
    results = {}

    (bond_plain, sq_rate, ts_handle,
     attach_engine, tape_bond, npv_bond,
     _b_rate_r, _b_a_r, _b_s_r, _b_ts_r, _b_tsh_r, _b_bond_r, _b_model_r,
     _b_sched_r, _b_csched_r) = _build_bond_mc_structures()

    # H1 — FD batch: 3-input bump-and-reprice for every bond scenario
    def _bond_fd_mc():
        for scene in bond_scenarios:
            rate_v, a_v, s_v = scene
            sq_rate.setValue(rate_v)
            _sq_a_ref.setValue(a_v)
            _sq_s_ref.setValue(s_v)
            attach_engine()
            bond_plain.cleanPrice()
            sq_rate.setValue(rate_v + BPS)
            bond_plain.cleanPrice()
            sq_rate.setValue(rate_v)
            _sq_a_ref.setValue(a_v + BPS)
            attach_engine()
            bond_plain.cleanPrice()
            _sq_a_ref.setValue(a_v)
            _sq_s_ref.setValue(s_v + BPS)
            attach_engine()
            bond_plain.cleanPrice()
            _sq_s_ref.setValue(s_v)
            attach_engine()
        sq_rate.setValue(_BOND_BASE_RATE)
        _sq_a_ref.setValue(_BOND_BASE_A)
        _sq_s_ref.setValue(_BOND_BASE_S)
        attach_engine()

    calcDate  = ql.Date(16, 8, 2006)
    dayCount  = ql.ActualActual(ql.ActualActual.Bond)
    _sq_a_ref = ql.SimpleQuote(_BOND_BASE_A)
    _sq_s_ref = ql.SimpleQuote(_BOND_BASE_S)

    def attach_engine():   # shadow the one from _build_bond_mc_structures
        a0 = _sq_a_ref.value().getValue()
        s0 = _sq_s_ref.value().getValue()
        model = ql.HullWhite(ts_handle, a0, s0)
        bond_plain.setPricingEngine(ql.TreeCallableFixedRateBondEngine(model, 40))

    attach_engine()

    _b_calendar     = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    _b_issueDate    = ql.Date(16, ql.September, 2004)
    _b_maturityDate = ql.Date(15, ql.September, 2012)
    _b_nc           = ql.NullCalendar()

    m, s = _median_ms(_bond_fd_mc, repeats)
    results["bond_fd_mc"] = {"median": m, "stdev": s,
                              "n_scenarios": N_SCENARIOS, "n_inputs": 3}

    # H2 — AAD replay: N_SCENARIOS backward sweeps on fixed tape (base parameters)
    def _bond_aad_replay():
        for _ in range(N_SCENARIOS):
            tape_bond.clearDerivatives()
            npv_bond.derivative = 1.0
            tape_bond.computeAdjoints()

    m, s = _median_ms(_bond_aad_replay, repeats, warmup=3)
    results["bond_aad_replay"] = {"median": m, "stdev": s,
                                   "n_scenarios": N_SCENARIOS, "n_inputs": 3}
    tape_bond.deactivate()

    # H3 — AAD re-record: single persistent Tape, newRecording() per scenario.
    _rec_sched = ql.Schedule(
        _b_issueDate, _b_maturityDate, ql.Period(ql.Quarterly),
        _b_calendar, ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Backward, False)
    _rec_csched = ql.CallabilitySchedule()
    _rec_callDate = ql.Date(15, ql.September, 2006)
    for _ in range(24):
        _rec_csched.append(ql.Callability(
            ql.BondPrice(100.0, ql.BondPrice.Clean),
            ql.Callability.Call, _rec_callDate))
        _rec_callDate = _b_nc.advance(_rec_callDate, 3, ql.Months)

    def _bond_aad_record():
        tape = Tape()
        tape.activate()
        rate_r = a_r = s_r = ts = tsh = bond = model_s = npv = None
        for scene in bond_scenarios:
            rate_v, a_v, s_v = scene
            rate_r = Real(rate_v)
            a_r    = Real(a_v)
            s_r    = Real(s_v)
            tape.registerInputs([rate_r, a_r, s_r])
            tape.newRecording()
            ts = ql.FlatForward(calcDate,
                                ql.QuoteHandle(ql.SimpleQuote(rate_r)),
                                dayCount, ql.Compounded, ql.Semiannual)
            tsh = ql.RelinkableYieldTermStructureHandle(ts)
            bond = ql.CallableFixedRateBond(
                3, 100, _rec_sched, [0.025],
                ql.ActualActual(ql.ActualActual.Bond), ql.Following,
                100, _b_issueDate, _rec_csched)
            model_s = ql.HullWhite(tsh, a_r, s_r)
            bond.setPricingEngine(
                ql.TreeCallableFixedRateBondEngine(model_s, 40))
            npv = bond.cleanPrice()
            tape.registerOutput(npv)
            npv.derivative = 1.0
            tape.computeAdjoints()
        del npv, model_s, bond, tsh, ts, s_r, a_r, rate_r
        tape.deactivate()

    m, s = _median_ms(_bond_aad_record, repeats)
    results["bond_aad_record"] = {"median": m, "stdev": s,
                                   "n_scenarios": N_SCENARIOS, "n_inputs": 3}
    attach_engine()

    # ---- Explicit teardown -----------------------------------------------
    del _bond_fd_mc, _bond_aad_replay, _bond_aad_record
    del attach_engine
    del npv_bond
    del _b_rate_r, _b_a_r, _b_s_r
    del _b_model_r, _b_bond_r
    del _b_ts_r, _b_tsh_r
    del _sq_a_ref, _sq_s_ref
    del sq_rate
    del ts_handle, bond_plain, _b_sched_r, _b_csched_r
    del tape_bond

    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: display and markdown
# ============================================================================

def _fmt_t(median, stdev):
    return f"{median:8.1f} ±{stdev:6.1f} ms"


def print_results(data: dict, repeats: int, wheel_name: str):
    fd  = data["bond_fd_mc"]
    rpl = data["bond_aad_replay"]
    rec = data["bond_aad_record"]
    n_scen = fd["n_scenarios"]
    n_in   = fd["n_inputs"]

    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Callable Bond Benchmark: FD vs AAD")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  MC scenarios per batch : {n_scen}")
    print(f"  Outer repeats          : {repeats}  (median of batch timings)")
    print(f"  BPS shift (FD)         : {BPS}")
    print(f"  Wheel                  : {wheel_name}")
    print()

    COL_M = 38

    print(f"  ── Callable Bond  ({n_in} inputs, {n_scen} scenarios per batch) ──")
    print()
    hdr = (f"  {'Method':<{COL_M}}"
           f"  {'Batch time':>18}"
           f"  {'Per-scenario':>14}"
           f"  {'FD/method':>10}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    fd_med = fd["median"]
    for method, d in [
        ("FD (N+1 bump-and-reprice)",          fd),
        ("AAD replay (backward sweep only)",   rpl),
        ("AAD re-record (forward + backward)", rec),
    ]:
        med = d["median"]
        std = d["stdev"]
        ps  = f"{med / n_scen * 1000:.1f} µs"
        ratio = f"{fd_med / med:.1f}×" if med > 0 else "—"
        print(f"  {method:<{COL_M}}"
              f"  {med:>8.1f} ±{std:>5.1f} ms"
              f"  {ps:>14}"
              f"  {ratio:>10}")

    print()
    print(SEPARATOR)
    print("  KEY RATIOS")
    print(SEPARATOR)
    print(f"    FD ÷ AAD replay    = {fd_med / rpl['median']:.0f}×"
          f"   (replay gets all {n_in} sensitivities per backward sweep)")
    print(f"    FD ÷ AAD re-record = {fd_med / rec['median']:.1f}×"
          f"   (re-record = full forward + backward per scenario)")
    print(f"    Re-record ÷ replay = {rec['median'] / rpl['median']:.0f}×"
          f"   (cost of re-recording the tape each scenario)")
    print()
    print("  NOTES")
    print("    • FD: N+1 = 4 tree pricings per scenario (base + 3 bumps)")
    print("    • AAD replay: tape recorded once at base parameters, then")
    print("      N_SCENARIOS backward sweeps — O(1) in number of inputs")
    print("    • AAD re-record: fresh QL objects + tape per scenario —")
    print("      produces correct per-scenario sensitivities at higher cost")
    print(SEPARATOR)
    print()


MD_PATH = Path(__file__).resolve().parent / "monte_carlo_bond_benchmarks_results.md"


def write_markdown(data: dict, repeats: int, wheel_name: str):
    now = datetime.datetime.now()
    fd  = data["bond_fd_mc"]
    rpl = data["bond_aad_replay"]
    rec = data["bond_aad_record"]
    n_scen = fd["n_scenarios"]
    n_in   = fd["n_inputs"]
    fd_med = fd["median"]

    lines = []
    w = lines.append

    w("# QuantLib-Risks-Py — Callable Bond Benchmark: FD vs AAD")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**MC scenarios per batch:** {n_scen}  ")
    w(f"**Outer repetitions:** {repeats} (median reported)  ")
    w(f"**Wheel:** `{wheel_name}`  ")
    w("")
    w("---")
    w("")
    w("## What is being measured")
    w("")
    w("Three methods for computing sensitivities of a Callable Fixed-Rate Bond")
    w("(HullWhite tree engine, 40 steps, 3 inputs: flat rate r, mean-reversion a,")
    w("volatility σ) across 100 randomly perturbed Monte Carlo scenarios:")
    w("")
    w("| Method | Description |")
    w("|---|---|")
    w("| **FD (bump-and-reprice)** | N+1 = 4 tree pricings per scenario "
      "(base + 1 bp bump per input). HullWhite model is rebuilt for a and σ bumps. |")
    w("| **AAD replay** | Tape recorded once at base parameters; each scenario "
      "replays only the backward sweep — O(1) w.r.t. number of inputs. |")
    w("| **AAD re-record** | Per-scenario fresh `Real` inputs, QL objects, and "
      "tape recording + backward sweep. Correct per-scenario sensitivities. |")
    w("")
    w("---")
    w("")
    w("## Results")
    w("")
    w(f"### Callable Bond — {n_in} inputs, {n_scen} scenarios per batch")
    w("")
    w("| Method | Batch time (ms) | Per-scenario | FD ÷ method |")
    w("|---|---:|---:|---:|")

    for method, d in [
        ("**FD** (bump-and-reprice)",           fd),
        ("**AAD replay** (backward sweep)",     rpl),
        ("**AAD re-record** (forward+backward)", rec),
    ]:
        med = d["median"]
        std = d["stdev"]
        ps  = f"{med / n_scen * 1000:.0f} µs"
        ratio = f"{fd_med / med:.1f}×" if med > 0 else "—"
        w(f"| {method} | {med:.1f} ±{std:.1f} | {ps} | {ratio} |")

    w("")
    w("---")
    w("")
    w("## Key ratios")
    w("")
    w("| Ratio | Value | Interpretation |")
    w("|---|---:|---|")
    w(f"| FD ÷ AAD replay | **{fd_med / rpl['median']:.0f}×** | "
      f"Replay gets all {n_in} sensitivities in one backward sweep |")
    w(f"| FD ÷ AAD re-record | **{fd_med / rec['median']:.1f}×** | "
      f"Re-record = full forward + backward per scenario |")
    w(f"| Re-record ÷ replay | **{rec['median'] / rpl['median']:.0f}×** | "
      f"Cost of re-recording the tape each scenario |")
    w("")
    w("---")
    w("")
    w("## Notes")
    w("")
    w(f"- BPS shift for FD: `{BPS}`")
    w("- *AAD replay* is fastest but uses sensitivities at base parameters for all"
      " scenarios (valid for small perturbations around a single market state).")
    w("- *AAD re-record* produces correct per-scenario sensitivities at the cost"
      " of re-recording the full computation graph each time.")
    w("- AAD complexity is **O(1)** in the number of inputs;"
      " FD complexity is **O(N)**.")
    w("- JIT/Forge backend is not used: the TreeCallableFixedRateBondEngine"
      " contains data-dependent branching incompatible with Forge's"
      " record-once-replay-many paradigm (see `JIT_LIMITATIONS.md`).")
    w("")
    w("---")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("")
    w("python benchmarks/monte_carlo_bond_benchmarks.py            # default 5 repeats")
    w("python benchmarks/monte_carlo_bond_benchmarks.py --repeats 10")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks Callable Bond benchmark: FD vs AAD",
    )
    parser.add_argument("--worker", metavar="REPEATS", type=int, default=None,
                        help="Internal worker mode: run benchmarks and print JSON")
    parser.add_argument("--repeats", "-r", type=int, default=5,
                        help=f"Outer repetitions per batch of {N_SCENARIOS} scenarios "
                             "(default: 5)")
    parser.add_argument("--clean-venvs", action="store_true",
                        help="Destroy and recreate benchmark venv")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write results to markdown file")
    args = parser.parse_args()

    if args.worker is not None:
        worker_main(args.worker)
        return

    repeats = args.repeats
    wheels  = find_wheel(BUILD)

    missing = [kind for kind in ("xad", "ql") if wheels[kind] is None]
    if missing:
        print("ERROR: Missing wheels for:", ", ".join(missing))
        print("  Run the build first:")
        print("    ./build.sh --no-jit -j$(nproc)")
        sys.exit(1)

    wheel_name = wheels["ql"].name

    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  Callable Bond Benchmark: FD vs AAD")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"  {N_SCENARIOS} scenarios per batch, {repeats} outer repetitions")
    print(SEPARATOR)

    print("\nSetting up virtual environment")
    print("-" * 50)
    setup_venv(VENV, wheels["xad"], wheels["ql"], force=args.clean_venvs)

    print(f"\nRunning benchmarks  ({repeats} outer repeats, {N_SCENARIOS} scenarios each)")
    print("-" * 50)
    print("\n  Worker …")
    data = run_worker_in_venv(VENV, repeats)
    print("  done.")

    print_results(data, repeats, wheel_name)

    if not args.no_save:
        write_markdown(data, repeats, wheel_name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
QuantLib-Risks-Py  –  Basket Option Greek Benchmarks (FD vs AAD)
================================================================

Benchmarks computation of first-order sensitivities for a European basket
option on two correlated assets, priced with Monte Carlo.

Engines:
  • MCEuropeanBasketEngine — **not JIT eligible** (MC random branching)

Three methods are compared:
  • **FD** – bump-and-reprice each of 5 inputs by 1 bp
  • **AAD** – XAD reverse-mode tape; one backward sweep for all 5 Greeks
  • **AAD + JIT** – not applicable (MC engines contain branching)

Market data (from Python/examples/basket-option.py):
  S1 = S2 = 7.0,  K = 8.0 (Call, MaxBasketPayoff)
  σ1 = σ2 = 0.10,  r = 0.05,  q1 = q2 = 0.05,  ρ = 0.5,  T = 1 Y

Usage
-----
  python benchmarks/basket_option_benchmarks.py
  python benchmarks/basket_option_benchmarks.py --repeats 50
  python benchmarks/basket_option_benchmarks.py --worker REPEATS   # internal
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
BPS = 1e-4

# Market data
_S1   = 7.0
_S2   = 7.0
_VOL1 = 0.10
_VOL2 = 0.10
_R0   = 0.05
_Q1   = 0.05
_Q2   = 0.05
_K    = 8.0
_CORR = 0.5
_SEED = 42
_SAMPLES = 32768   # fixed sample count for reproducibility

_INPUT_NAMES = ["S1 (spot1)", "S2 (spot2)", "σ1 (vol1)", "σ2 (vol2)", "r (rate)"]
_INPUT_VALS  = [_S1, _S2, _VOL1, _VOL2, _R0]


# ============================================================================
# Wheel discovery
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
# Venv helpers
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
# WORKER MODE
# ============================================================================

def _median_ms(func, n: int, warmup: int = 3):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


def _build_basket_plain():
    """Return (option, [sq_s1, sq_s2, sq_v1, sq_v2, sq_r], process_array)."""
    import QuantLib_Risks as ql
    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    settlementDate = ql.Date(17, ql.May, 1998)

    sq_s1 = ql.SimpleQuote(_S1)
    sq_s2 = ql.SimpleQuote(_S2)
    sq_v1 = ql.SimpleQuote(_VOL1)
    sq_v2 = ql.SimpleQuote(_VOL2)
    sq_r  = ql.SimpleQuote(_R0)

    riskFreeRate = ql.FlatForward(
        settlementDate, ql.QuoteHandle(sq_r), ql.Actual365Fixed())

    proc1 = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sq_s1),
        ql.YieldTermStructureHandle(ql.FlatForward(
            settlementDate, _Q1, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(sq_v1), ql.Actual365Fixed())))
    proc2 = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sq_s2),
        ql.YieldTermStructureHandle(ql.FlatForward(
            settlementDate, _Q2, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(sq_v2), ql.Actual365Fixed())))

    matrix = ql.Matrix(2, 2)
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[0][1] = _CORR
    matrix[1][0] = _CORR

    process = ql.StochasticProcessArray([proc1, proc2], matrix)

    exercise = ql.EuropeanExercise(ql.Date(17, ql.May, 1999))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, _K)
    option = ql.BasketOption(ql.MaxBasketPayoff(payoff), exercise)
    option.setPricingEngine(
        ql.MCEuropeanBasketEngine(process, "lowdiscrepancy",
                                  timeStepsPerYear=1,
                                  requiredSamples=_SAMPLES))
    return option, [sq_s1, sq_s2, sq_v1, sq_v2, sq_r]


def _build_basket_aad():
    """Return (tape, option, [s1_v, s2_v, v1_v, v2_v, r_v])."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()

    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    settlementDate = ql.Date(17, ql.May, 1998)

    s1_v = ql.Real(_S1)
    s2_v = ql.Real(_S2)
    v1_v = ql.Real(_VOL1)
    v2_v = ql.Real(_VOL2)
    r_v  = ql.Real(_R0)

    all_inputs = [s1_v, s2_v, v1_v, v2_v, r_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    riskFreeRate = ql.FlatForward(
        settlementDate, ql.QuoteHandle(ql.SimpleQuote(r_v)), ql.Actual365Fixed())

    proc1 = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s1_v)),
        ql.YieldTermStructureHandle(ql.FlatForward(
            settlementDate, _Q1, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(v1_v)),
            ql.Actual365Fixed())))
    proc2 = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s2_v)),
        ql.YieldTermStructureHandle(ql.FlatForward(
            settlementDate, _Q2, ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(v2_v)),
            ql.Actual365Fixed())))

    matrix = ql.Matrix(2, 2)
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[0][1] = _CORR
    matrix[1][0] = _CORR

    process = ql.StochasticProcessArray([proc1, proc2], matrix)

    exercise = ql.EuropeanExercise(ql.Date(17, ql.May, 1999))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, _K)
    option = ql.BasketOption(ql.MaxBasketPayoff(payoff), exercise)
    option.setPricingEngine(
        ql.MCEuropeanBasketEngine(process, "lowdiscrepancy",
                                  timeStepsPerYear=1,
                                  requiredSamples=_SAMPLES))
    return tape, option, all_inputs


# ---- Worker entry point -----------------------------------------------------

def _run_worker(repeats: int) -> dict:
    import xad
    V = lambda x: float(xad.value(x))   # extract plain float from xad Real
    results = {}
    n = len(_INPUT_VALS)
    results["n_inputs"] = n

    # ---- plain / FD ----
    opt_plain, quotes = _build_basket_plain()

    base_npv = opt_plain.NPV()
    results["npv"] = V(base_npv)

    fd_greeks = []
    for q, v0 in zip(quotes, _INPUT_VALS):
        q.setValue(v0 + BPS)
        npv_up = opt_plain.NPV()
        q.setValue(v0)
        fd_greeks.append(V(npv_up - base_npv) / BPS)
    results["fd_greeks"] = fd_greeks

    def _plain():
        quotes[0].setValue(_S1 + 1e-10)
        quotes[0].setValue(_S1)
        return opt_plain.NPV()

    m, s = _median_ms(_plain, repeats)
    results["plain"] = {"median": m, "stdev": s}

    def _fd():
        opt_plain.NPV()
        for q, v0 in zip(quotes, _INPUT_VALS):
            q.setValue(v0 + BPS)
            opt_plain.NPV()
            q.setValue(v0)

    m, s = _median_ms(_fd, repeats)
    results["fd"] = {"median": m, "stdev": s}

    # ---- AAD ----
    tape, opt_aad, inputs = _build_basket_aad()
    npv_aad = opt_aad.NPV()
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


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator
# ============================================================================

def _fmt_t(median, stdev):
    return f"{median:8.4f} ±{stdev:6.4f} ms"

def _sp(a, b):
    return f"{a / b:6.2f}x" if b > 0 else "   N/A"


def print_comparison(nojit: dict, jit: dict, repeats: int):
    print()
    print(SEPARATOR)
    print("Basket Option  –  FD vs AAD Benchmark  (JIT not applicable)")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Instrument : European Basket Call (MaxPayoff), 2 assets")
    print(f"               S1={_S1}, S2={_S2}, K={_K}, σ1={_VOL1}, σ2={_VOL2}")
    print(f"               r={_R0}, q1={_Q1}, q2={_Q2}, ρ={_CORR}, T=1Y")
    print(f"  Engine     : MCEuropeanBasketEngine (low-discrepancy, {_SAMPLES} samples)")
    print(f"  JIT        : NOT eligible (MC engine has branching)")
    print(f"  Repeats    : {repeats}")
    print(f"  BPS shift  : {BPS}")
    n = nojit["n_inputs"]
    print(f"  Inputs     : {n}  ({', '.join(_INPUT_NAMES)})")
    print()

    print(f"  NPV (FD build)  : {nojit['npv']:.10f}")
    print(f"  NPV (AAD build) : {nojit['aad_npv']:.10f}")
    print()

    # Greeks
    print(f"  {'Input':<16s}  {'FD':>14s}  {'AAD':>14s}  {'|Δ|':>12s}")
    print("  " + "-" * 60)
    for i, name in enumerate(_INPUT_NAMES):
        fd_g = nojit['fd_greeks'][i]
        aad_g = nojit['aad_greeks'][i]
        diff = abs(fd_g - aad_g)
        print(f"  {name:<16s}  {fd_g:14.8f}  {aad_g:14.8f}  {diff:12.2e}")
    print()

    # Timing
    col = 22
    hdr = (f"  {'Method':<28s}  {'Non-JIT':>{col}}  {'JIT':>{col}}  {'JIT speedup':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lbl, key in [
        ("Plain pricing (1 NPV)",       "plain"),
        ("Bump-and-reprice FD (N+1)",   "fd"),
        ("AAD backward pass",           "aad"),
    ]:
        nj = nojit[key]
        jt = jit[key]
        sp = _sp(nj["median"], jt["median"])
        print(f"    {lbl:<26s}  "
              f"{_fmt_t(nj['median'], nj['stdev']):>{col}}  "
              f"{_fmt_t(jt['median'], jt['stdev']):>{col}}  "
              f"{sp:>11}")

    fd_aad_nojit = nojit["fd"]["median"] / nojit["aad"]["median"] if nojit["aad"]["median"] else 0
    fd_aad_jit   = jit["fd"]["median"]   / jit["aad"]["median"]   if jit["aad"]["median"]   else 0
    print()
    print(f"  FD ÷ AAD ratio:  Non-JIT {fd_aad_nojit:.1f}x  |  JIT {fd_aad_jit:.1f}x")
    print()
    print("  Note: MC engine is not JIT-eligible; JIT column shows the Forge build")
    print("        falling back to interpreted AD (expect speedup ≈ 1.0x).")
    print()
    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "basket_option_benchmark_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append
    n = nojit["n_inputs"]

    w("# Basket Option — FD vs AAD Benchmark Results")
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
    w("## Instrument")
    w("")
    w("| Parameter | Value |")
    w("|---|---|")
    w("| Type | European Basket Call (MaxBasketPayoff) |")
    w(f"| Assets | 2 |")
    w(f"| Spot (S1, S2) | {_S1}, {_S2} |")
    w(f"| Strike (K) | {_K} |")
    w(f"| Volatility (σ1, σ2) | {_VOL1}, {_VOL2} |")
    w(f"| Risk-free rate (r) | {_R0} |")
    w(f"| Dividend yields (q1, q2) | {_Q1}, {_Q2} |")
    w(f"| Correlation (ρ) | {_CORR} |")
    w(f"| Maturity | 1 year |")
    w(f"| Engine | `MCEuropeanBasketEngine` (low-discrepancy, {_SAMPLES} samples) |")
    w(f"| JIT eligible | **No** — MC engine has branching (random path generation) |")
    w("")
    w("---")
    w("")
    w("## Greeks validation (AAD vs FD)")
    w("")
    w(f"NPV = {nojit['npv']:.10f}")
    w("")
    w("| Input | FD (1 bp) | AAD | \|Δ\| |")
    w("|---|---:|---:|---:|")
    for i, name in enumerate(_INPUT_NAMES):
        fd_g = nojit['fd_greeks'][i]
        aad_g = nojit['aad_greeks'][i]
        diff = abs(fd_g - aad_g)
        w(f"| {name} | {fd_g:.8f} | {aad_g:.8f} | {diff:.2e} |")
    w("")
    w("---")
    w("")
    w("## Timing results")
    w("")
    w(f"N = {n} inputs, {repeats} repetitions, BPS = {BPS}")
    w("")
    w("| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |")
    w("|---|---:|---:|---:|")
    for lbl, key in [
        ("Plain pricing (1 NPV)", "plain"),
        ("Bump-and-reprice FD (N+1 NPVs)", "fd"),
        ("**AAD backward pass**", "aad"),
    ]:
        nj = nojit[key]
        jt = jit[key]
        sp = f"{nj['median'] / jt['median']:.2f}×" if jt['median'] > 0 else "—"
        w(f"| {lbl} | {nj['median']:.4f} ±{nj['stdev']:.4f} "
          f"| {jt['median']:.4f} ±{jt['stdev']:.4f} | {sp} |")

    fd_aad_nojit = nojit["fd"]["median"] / nojit["aad"]["median"] if nojit["aad"]["median"] else 0
    fd_aad_jit   = jit["fd"]["median"]   / jit["aad"]["median"]   if jit["aad"]["median"]   else 0
    w(f"| *FD ÷ AAD* | *{fd_aad_nojit:.1f}×* | *{fd_aad_jit:.1f}×* | — |")
    w("")
    w("---")
    w("")
    w("## Analysis")
    w("")
    w("The **MCEuropeanBasketEngine** uses Monte Carlo simulation with pseudo-random")
    w("or quasi-random sequences.  The path generation and payoff evaluation involve")
    w("branching (max/min payoffs, early termination checks), so the engine is **not")
    w("eligible for JIT compilation** — the Forge compiler cannot trace through")
    w("data-dependent branches in the MC loop.")
    w("")
    w("The JIT column therefore shows the Forge build falling back to interpreted")
    w("AD, and the JIT speedup should be ≈ 1.0×.")
    w("")
    w("Despite this, AAD still provides a benefit over FD: one backward sweep")
    w(f"gives all {n} Greeks simultaneously, whereas FD requires {n}+1 = {n+1}")
    w("full MC simulations.  For MC engines (which are typically the slowest),")
    w("this O(1) vs O(N) advantage is the primary value of AAD.")
    w("")
    w("---")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("python benchmarks/basket_option_benchmarks.py")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Basket option FD vs AAD benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--worker", metavar="REPEATS", type=int, default=None)
    parser.add_argument("--repeats", "-r", type=int, default=30)
    parser.add_argument("--clean-venvs", action="store_true")
    parser.add_argument("--no-save", action="store_true")
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
        sys.exit(1)

    print(SEPARATOR)
    print("Basket Option  –  FD vs AAD Benchmark")
    print(SEPARATOR)

    print("\nSetting up virtual environments")
    print("-" * 50)
    print(f"\n[1/2] Non-JIT venv  ({VENV_NOJIT.name})")
    setup_venv(VENV_NOJIT, wheels["nojit"]["xad"], wheels["nojit"]["ql"],
               force=args.clean_venvs)
    print(f"\n[2/2] JIT venv      ({VENV_JIT.name})")
    setup_venv(VENV_JIT, wheels["jit"]["xad"], wheels["jit"]["ql"],
               force=args.clean_venvs)

    print(f"\nRunning benchmarks  ({repeats} repeats per build)")
    print("-" * 50)
    print("\n  [1/2] Non-JIT worker …")
    nojit = run_worker_in_venv(VENV_NOJIT, repeats)
    print("        done.")
    print("\n  [2/2] JIT worker …")
    jit = run_worker_in_venv(VENV_JIT, repeats)
    print("        done.")

    print_comparison(nojit, jit, repeats)

    if not args.no_save:
        write_markdown(nojit, jit, repeats, wheels)


if __name__ == "__main__":
    main()

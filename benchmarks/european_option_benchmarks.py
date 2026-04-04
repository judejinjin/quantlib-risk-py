#!/usr/bin/env python3
"""
QuantLib-Risks-Py  –  European Option Greek Benchmarks (FD vs AAD vs AAD+JIT)
==============================================================================

Benchmarks the computation of first-order sensitivities (delta, dividend-rho,
vega, rho) for a European call option priced with the Black-Scholes-Merton
analytic engine (``AnalyticEuropeanEngine``).

Three methods are compared:

  • **Finite Differences (FD)** – bump-and-reprice each input by 1 bp
  • **AAD** – XAD reverse-mode tape; one backward sweep gives all Greeks
  • **AAD + JIT** – same tape compiled to native code via XAD-Forge

The analytic BSM engine is a closed-form formula with no branching, so it is
fully eligible for JIT compilation.

Market data (from Python/examples/european-option.py):
  S = 7.0,  K = 8.0,  σ = 0.10,  r = 0.05,  q = 0.05,  T = 1 Y

Usage
-----
  python benchmarks/european_option_benchmarks.py               # 30 repeats
  python benchmarks/european_option_benchmarks.py --repeats 50
  python benchmarks/european_option_benchmarks.py --clean-venvs
  python benchmarks/european_option_benchmarks.py --worker REPEATS   # internal
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
BPS = 1e-4   # 1 basis-point shift for FD

# Market data
_S0   = 7.0    # spot
_Q0   = 0.05   # dividend yield
_VOL0 = 0.10   # volatility
_R0   = 0.05   # risk-free rate
_K    = 8.0    # strike
_INPUT_NAMES = ["S (spot)", "q (div yield)", "σ (vol)", "r (rate)"]
_INPUT_VALS  = [_S0, _Q0, _VOL0, _R0]


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
# Venv helpers  (shared with other benchmark scripts)
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
# WORKER MODE  –  benchmark implementations
# ============================================================================

def _median_ms(func, n: int, warmup: int = 5):
    """Return (median_ms, stdev_ms) over n timed calls, after warmup un-timed calls."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ---- Build helpers ----------------------------------------------------------

def _build_option_plain():
    """Return (option, [sq_spot, sq_div, sq_vol, sq_rate])."""
    import QuantLib_Risks as ql
    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    sq_spot = ql.SimpleQuote(_S0)
    sq_div  = ql.SimpleQuote(_Q0)
    sq_vol  = ql.SimpleQuote(_VOL0)
    sq_rate = ql.SimpleQuote(_R0)
    proc = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sq_spot),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(sq_div), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(sq_rate), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(sq_vol), ql.Actual365Fixed())))
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, _K),
        ql.EuropeanExercise(ql.Date(17, ql.May, 1999)))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return option, [sq_spot, sq_div, sq_vol, sq_rate]


def _build_option_aad():
    """Return (tape, option, [spot_v, div_v, vol_v, rate_v])."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()
    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    spot_v = ql.Real(_S0)
    div_v  = ql.Real(_Q0)
    vol_v  = ql.Real(_VOL0)
    rate_v = ql.Real(_R0)
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
        ql.PlainVanillaPayoff(ql.Option.Call, _K),
        ql.EuropeanExercise(ql.Date(17, ql.May, 1999)))
    option.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return tape, option, all_inputs


# ---- Worker entry point -----------------------------------------------------

def _run_worker(repeats: int) -> dict:
    results = {}
    import xad
    V = lambda x: float(xad.value(x))   # extract plain float from xad Real

    # ---- plain pricing ----
    opt_plain, opt_quotes = _build_option_plain()

    # compute baseline NPV and FD Greeks
    base_npv = opt_plain.NPV()
    results["npv"] = V(base_npv)

    fd_greeks = []
    for q, v0 in zip(opt_quotes, _INPUT_VALS):
        q.setValue(v0 + BPS)
        npv_up = opt_plain.NPV()
        q.setValue(v0)
        fd_greeks.append(V(npv_up - base_npv) / BPS)
    results["fd_greeks"] = fd_greeks

    def _plain_opt():
        opt_quotes[0].setValue(_S0 + 1e-10)
        opt_quotes[0].setValue(_S0)
        return opt_plain.NPV()

    m, s = _median_ms(_plain_opt, repeats)
    results["plain"] = {"median": m, "stdev": s}

    # ---- FD timing (N+1 pricings) ----
    def _fd_opt():
        opt_plain.NPV()
        for q, v0 in zip(opt_quotes, _INPUT_VALS):
            q.setValue(v0 + BPS)
            opt_plain.NPV()
            q.setValue(v0)

    m, s = _median_ms(_fd_opt, repeats)
    results["fd"] = {"median": m, "stdev": s}

    # ---- AAD ----
    tape, opt_aad, opt_inputs = _build_option_aad()
    npv_aad = opt_aad.NPV()
    tape.registerOutput(npv_aad)

    aad_greeks = []
    tape.clearDerivatives()
    npv_aad.derivative = 1.0
    tape.computeAdjoints()
    for inp in opt_inputs:
        aad_greeks.append(V(xad.derivative(inp)))
    results["aad_greeks"] = aad_greeks
    results["aad_npv"] = V(npv_aad)

    def _aad_opt():
        tape.clearDerivatives()
        npv_aad.derivative = 1.0
        tape.computeAdjoints()

    m, s = _median_ms(_aad_opt, repeats)
    results["aad"] = {"median": m, "stdev": s}
    tape.deactivate()

    results["n_inputs"] = len(opt_inputs)
    return results


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator: comparison output
# ============================================================================

def _fmt_t(median, stdev):
    return f"{median:8.4f} ±{stdev:6.4f} ms"


def _sp(a, b):
    if b > 0:
        return f"{a / b:6.2f}x"
    return "   N/A"


def print_comparison(nojit: dict, jit: dict, repeats: int):
    print()
    print(SEPARATOR)
    print("European Option  –  FD vs AAD vs AAD+JIT Benchmark")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Instrument : European Call  S={_S0}, K={_K}, σ={_VOL0}, "
          f"r={_R0}, q={_Q0}, T=1Y")
    print(f"  Engine     : AnalyticEuropeanEngine (Black-Scholes-Merton)")
    print(f"  JIT        : Eligible (closed-form formula, no branching)")
    print(f"  Repeats    : {repeats}  (median of wall-clock timings)")
    print(f"  BPS shift  : {BPS}")
    print(f"  Inputs     : {nojit['n_inputs']}  ({', '.join(_INPUT_NAMES)})")
    n = nojit['n_inputs']
    print()

    # NPV
    print(f"  NPV (FD build)  : {nojit['npv']:.10f}")
    print(f"  NPV (AAD build) : {nojit['aad_npv']:.10f}")
    print()

    # Greeks comparison
    print("  Greeks comparison (AAD vs FD):")
    print(f"    {'Input':<16s}  {'FD':>14s}  {'AAD':>14s}  {'|Δ|':>12s}")
    print("    " + "-" * 60)
    for i, name in enumerate(_INPUT_NAMES):
        fd_g = nojit['fd_greeks'][i]
        aad_g = nojit['aad_greeks'][i]
        diff = abs(fd_g - aad_g)
        print(f"    {name:<16s}  {fd_g:14.8f}  {aad_g:14.8f}  {diff:12.2e}")
    print()

    # Timing table
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

    # FD ÷ AAD ratio
    fd_aad_nojit = nojit["fd"]["median"] / nojit["aad"]["median"] if nojit["aad"]["median"] else 0
    fd_aad_jit   = jit["fd"]["median"]   / jit["aad"]["median"]   if jit["aad"]["median"]   else 0
    print()
    print(f"  FD ÷ AAD ratio:  Non-JIT {fd_aad_nojit:.1f}x  |  JIT {fd_aad_jit:.1f}x")
    print()
    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "european_option_benchmark_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append
    n = nojit["n_inputs"]

    w("# European Option — FD vs AAD vs AAD+JIT Benchmark Results")
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
    w(f"| Type | European Call |")
    w(f"| Spot (S) | {_S0} |")
    w(f"| Strike (K) | {_K} |")
    w(f"| Volatility (σ) | {_VOL0} |")
    w(f"| Risk-free rate (r) | {_R0} |")
    w(f"| Dividend yield (q) | {_Q0} |")
    w(f"| Maturity | 1 year |")
    w(f"| Engine | `AnalyticEuropeanEngine` (BSM closed-form) |")
    w(f"| JIT eligible | **Yes** — no branching in analytic formula |")
    w("")
    w("---")
    w("")
    w("## Greeks validation (AAD vs FD)")
    w("")
    w(f"NPV = {nojit['npv']:.10f}")
    w("")
    w("| Input | FD (1 bp bump) | AAD | \|Δ\| |")
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
    w(f"N = {n} market inputs, {repeats} repetitions, BPS = {BPS}")
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
        sp = f"{nj['median'] / jt['median']:.2f}×" if jt['median'] > 0 else "—"
        w(f"| {label} | {nj['median']:.4f} ±{nj['stdev']:.4f} "
          f"| {jt['median']:.4f} ±{jt['stdev']:.4f} | {sp} |")

    fd_aad_nojit = nojit["fd"]["median"] / nojit["aad"]["median"] if nojit["aad"]["median"] else 0
    fd_aad_jit   = jit["fd"]["median"]   / jit["aad"]["median"]   if jit["aad"]["median"]   else 0
    w(f"| *FD ÷ AAD* | *{fd_aad_nojit:.1f}×* | *{fd_aad_jit:.1f}×* | — |")
    w("")
    w("---")
    w("")
    w("## Analysis")
    w("")
    w("The **AnalyticEuropeanEngine** uses the Black-Scholes-Merton closed-form formula,")
    w("which involves only smooth mathematical operations (`exp`, `log`, `erfc`) with no")
    w("branching (if/else) in the computation graph. This makes it an ideal candidate for")
    w("JIT compilation — the XAD-Forge compiler can translate the entire AAD tape to")
    w("optimised native machine code.")
    w("")
    w(f"With only {n} inputs, both FD ({n}+1 = {n+1} forward pricings) and AAD (1 backward")
    w(f"sweep) are fast.  The AAD advantage grows with the number of inputs (O(1) vs O(N)).")
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
    w("# Run this benchmark")
    w("python benchmarks/european_option_benchmarks.py")
    w("python benchmarks/european_option_benchmarks.py --repeats 50")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="European option FD vs AAD vs AAD+JIT benchmark",
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
        help="Destroy and recreate benchmark venvs before running",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write results markdown",
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
    print("European Option  –  FD vs AAD vs AAD+JIT Benchmark")
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

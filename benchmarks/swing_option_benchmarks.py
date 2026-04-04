#!/usr/bin/env python3
"""
QuantLib-Risks-Py  –  Swing Option Greek Benchmarks (FD vs AAD)
================================================================

Benchmarks computation of first-order sensitivities for a swing option priced
with the finite-difference PDE engine ``FdSimpleBSSwingEngine``.

Methods compared:
  • **FD** – bump-and-reprice each of 3 inputs by 1 bp
  • **AAD** – XAD reverse-mode tape; one backward sweep for all Greeks
  • **AAD + JIT** – NOT eligible (PDE FD solver has branching)

Market data (from Python/examples/swing.py):
  S = 30.0,  K = 30.0 (forward payoff),  σ = 0.20,  r = 0.05,  q = 0.00
  31 exercise dates (Jan 1–31, 2019), min exercises = 0, max = 31

Usage
-----
  python benchmarks/swing_option_benchmarks.py
  python benchmarks/swing_option_benchmarks.py --repeats 50
  python benchmarks/swing_option_benchmarks.py --worker REPEATS   # internal
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
_S0   = 30.0
_VOL0 = 0.20
_R0   = 0.05          # non-zero to avoid branch-at-zero in FlatForward
_Q0   = 0.00
_INPUT_NAMES = ["S (spot)", "σ (vol)", "r (rate)"]
_INPUT_VALS  = [_S0, _VOL0, _R0]


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


def _price_swing_fresh(spot, vol, rate):
    """Build & price a fresh swing option (full rebuild, no observer chain).

    The payoff strike is always ``_S0`` so that FD Greeks match AAD
    (which keeps the strike off-tape).  Only the BSM process inputs
    are varied.

    NOTE: VanillaSwingOption's observer chain does NOT invalidate its
    LazyObject cache when SimpleQuote values change, so a full rebuild
    is the only reliable way to get correct bump-and-reprice Greeks.
    """
    import QuantLib_Risks as ql
    import xad
    todaysDate = ql.Date(30, ql.September, 2018)
    ql.Settings.instance().evaluationDate = todaysDate

    riskFreeRate  = ql.FlatForward(todaysDate, rate, ql.Actual365Fixed())
    dividendYield = ql.FlatForward(todaysDate, _Q0, ql.Actual365Fixed())
    volatility    = ql.BlackConstantVol(
        todaysDate, ql.TARGET(), vol, ql.Actual365Fixed())

    exerciseDates = [ql.Date(1, ql.January, 2019) + i for i in range(31)]

    swingOption = ql.VanillaSwingOption(
        ql.VanillaForwardPayoff(ql.Option.Call, _S0),   # fixed strike
        ql.SwingExercise(exerciseDates),
        0, len(exerciseDates))

    bsProcess = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(dividendYield),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(volatility))

    swingOption.setPricingEngine(ql.FdSimpleBSSwingEngine(bsProcess))
    return float(xad.value(swingOption.NPV()))


def _build_swing_aad():
    """Return (tape, option, [spot_v, vol_v, rate_v])."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    tape = Tape()
    tape.activate()

    todaysDate = ql.Date(30, ql.September, 2018)
    ql.Settings.instance().evaluationDate = todaysDate

    spot_v = ql.Real(_S0)
    vol_v  = ql.Real(_VOL0)
    rate_v = ql.Real(_R0)
    all_inputs = [spot_v, vol_v, rate_v]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    riskFreeRate = ql.FlatForward(
        todaysDate, ql.QuoteHandle(ql.SimpleQuote(rate_v)), ql.Actual365Fixed())
    dividendYield = ql.FlatForward(todaysDate, _Q0, ql.Actual365Fixed())
    volatility = ql.BlackConstantVol(
        todaysDate, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(vol_v)),
        ql.Actual365Fixed())

    exerciseDates = [ql.Date(1, ql.January, 2019) + i for i in range(31)]

    import xad
    swingOption = ql.VanillaSwingOption(
        ql.VanillaForwardPayoff(ql.Option.Call, float(xad.value(spot_v))),
        ql.SwingExercise(exerciseDates),
        0, len(exerciseDates))

    bsProcess = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot_v)),
        ql.YieldTermStructureHandle(dividendYield),
        ql.YieldTermStructureHandle(riskFreeRate),
        ql.BlackVolTermStructureHandle(volatility))

    swingOption.setPricingEngine(ql.FdSimpleBSSwingEngine(bsProcess))
    return tape, swingOption, all_inputs


# ---- Worker entry point -----------------------------------------------------

def _run_worker(repeats: int) -> dict:
    import xad
    V = lambda x: float(xad.value(x))   # extract plain float from xad Real
    results = {}
    n = len(_INPUT_VALS)
    results["n_inputs"] = n

    # ---- plain / FD (full rebuild – observer chain broken for swing) ----
    base_npv = _price_swing_fresh(_S0, _VOL0, _R0)
    results["npv"] = base_npv

    fd_greeks = []
    for i, v0 in enumerate(_INPUT_VALS):
        bumped = list(_INPUT_VALS)
        bumped[i] = v0 + BPS
        npv_up = _price_swing_fresh(*bumped)
        fd_greeks.append((npv_up - base_npv) / BPS)
    results["fd_greeks"] = fd_greeks

    def _plain():
        return _price_swing_fresh(_S0, _VOL0, _R0)

    m, s = _median_ms(_plain, repeats)
    results["plain"] = {"median": m, "stdev": s}

    def _fd():
        _price_swing_fresh(_S0, _VOL0, _R0)
        for i, v0 in enumerate(_INPUT_VALS):
            bumped = list(_INPUT_VALS)
            bumped[i] = v0 + BPS
            _price_swing_fresh(*bumped)

    m, s = _median_ms(_fd, repeats)
    results["fd"] = {"median": m, "stdev": s}

    # ---- AAD ----
    tape, opt_aad, inputs = _build_swing_aad()
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
    print("Swing Option  –  FD vs AAD Benchmark  (JIT not applicable)")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Instrument : Swing Option (VanillaForwardPayoff, Call)")
    print(f"               S={_S0}, K={_S0} (forward), σ={_VOL0}, "
          f"r={_R0}, q={_Q0}")
    print(f"               31 exercise dates, min=0, max=31")
    print(f"  Engine     : FdSimpleBSSwingEngine (PDE finite-differences)")
    print(f"  JIT        : NOT eligible (PDE branching)")
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
    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "swing_option_benchmark_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append
    n = nojit["n_inputs"]

    w("# Swing Option — FD vs AAD Benchmark Results")
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
    w("| Type | Swing Option (VanillaForwardPayoff, Call) |")
    w(f"| Spot (S) | {_S0} |")
    w(f"| Strike (K) | {_S0} (forward) |")
    w(f"| Volatility (σ) | {_VOL0} |")
    w(f"| Risk-free rate (r) | {_R0} |")
    w(f"| Dividend yield (q) | {_Q0} |")
    w("| Exercise dates | 31 (Jan 1–31, 2019) |")
    w("| Min exercises | 0 |")
    w("| Max exercises | 31 |")
    w(f"| Engine | `FdSimpleBSSwingEngine` (PDE finite-differences) |")
    w("| JIT eligible | **No** — PDE solver has branching (boundary conditions, exercise logic) |")
    w("")
    w("---")
    w("")
    w("## Greeks validation (AAD vs FD)")
    w("")
    w(f"NPV = {nojit['npv']:.10f}")
    w("")
    w("| Input | FD (1 bp) | AAD | \\|Δ\\| |")
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
    w("The **FdSimpleBSSwingEngine** solves a PDE on a finite-difference grid with")
    w("conditional logic for boundary conditions and early-exercise decisions at each")
    w("of the 31 exercise dates.  This branching makes the engine **not eligible for")
    w("JIT compilation** — the Forge compiler cannot trace through data-dependent branches.")
    w("")
    w("The JIT column shows the Forge build falling back to interpreted AD (expect")
    w("speedup ≈ 1.0×).")
    w("")
    w(f"With {n} inputs, FD requires {n+1} = {n+1} PDE solves while AAD needs only")
    w("1 backward sweep.  The PDE solver is moderately expensive, so the FD ÷ AAD")
    w("ratio shows the efficiency gain from AAD even without JIT.")
    w("")
    w("---")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("python benchmarks/swing_option_benchmarks.py")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Swing option FD vs AAD benchmark",
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
    print("Swing Option  –  FD vs AAD Benchmark")
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

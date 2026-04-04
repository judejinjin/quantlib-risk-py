#!/usr/bin/env python3
"""
QuantLib-Risks-Py  –  American Option Greek Benchmarks (FD vs AAD vs AAD+JIT)
==============================================================================

Benchmarks computation of first-order sensitivities for an American put option
using multiple pricing engines:

  A)  BaroneAdesiWhaleyApproximationEngine   — quasi-analytic, **JIT eligible**
  B)  BjerksundStenslandApproximationEngine  — quasi-analytic, **JIT eligible**
  C)  FdBlackScholesVanillaEngine            — PDE finite-differences, **no JIT**
  D)  QdPlusAmericanEngine                   — analytic, **JIT eligible**

Each engine is timed for:
  • **Finite differences (FD)** – bump-and-reprice each input by 1 bp
  • **AAD** – XAD reverse-mode tape; one backward sweep for all Greeks
  • **AAD + JIT** – tape compiled to native code (where eligible)

Market data (from Python/examples/american-option.py):
  S = 36.0,  K = 40.0 (Put),  σ = 0.20,  r = 0.06,  q = 0.00,  T = 1 Y

Usage
-----
  python benchmarks/american_option_benchmarks.py
  python benchmarks/american_option_benchmarks.py --repeats 50
  python benchmarks/american_option_benchmarks.py --worker REPEATS   # internal
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
_S0   = 36.0
_R0   = 0.06
_Q0   = 0.00
_VOL0 = 0.20
_K    = 40.0
_INPUT_NAMES = ["S (spot)", "r (rate)", "q (div yield)", "σ (vol)"]
_INPUT_VALS  = [_S0, _R0, _Q0, _VOL0]

# Engines to benchmark: (label, engine_constructor_name, jit_eligible)
_ENGINES = [
    ("BAW",              "BaroneAdesiWhaleyApproximationEngine",  True),
    ("Bjerksund-Stensl", "BjerksundStenslandApproximationEngine", True),
    ("FD-BS (PDE)",      "FdBlackScholesVanillaEngine",          False),
    ("QD+",              "QdPlusAmericanEngine",                  True),
]


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
# WORKER MODE  –  benchmark implementations
# ============================================================================

def _median_ms(func, n: int, warmup: int = 5):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


def _make_process_plain(ql, todaysDate, sq_spot, sq_rate, sq_div, sq_vol):
    """Build a BSM process from SimpleQuote handles."""
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sq_spot),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(sq_div), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(sq_rate), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(sq_vol), ql.Actual365Fixed())))


def _make_process_aad(ql, todaysDate, spot_v, rate_v, div_v, vol_v):
    """Build a BSM process from ql.Real values (for taping)."""
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot_v)),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(ql.SimpleQuote(div_v)), ql.Actual365Fixed())),
        ql.YieldTermStructureHandle(ql.FlatForward(
            todaysDate, ql.QuoteHandle(ql.SimpleQuote(rate_v)), ql.Actual365Fixed())),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            todaysDate, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(vol_v)),
            ql.Actual365Fixed())))


def _attach_engine(ql, option, proc, engine_name):
    """Attach a named engine to the option."""
    if engine_name == "FdBlackScholesVanillaEngine":
        option.setPricingEngine(getattr(ql, engine_name)(proc, 801, 800))
    else:
        option.setPricingEngine(getattr(ql, engine_name)(proc))


def _run_worker(repeats: int) -> dict:
    import QuantLib_Risks as ql
    import xad
    from xad.adj_1st import Tape

    todaysDate = ql.Date(15, ql.May, 1998)
    ql.Settings.instance().evaluationDate = todaysDate
    n = len(_INPUT_VALS)

    V = lambda x: float(xad.value(x))   # extract plain float from xad Real
    results = {"n_inputs": n, "engines": {}}

    for label, engine_name, jit_ok in _ENGINES:
        eng_results = {"label": label, "engine": engine_name, "jit_eligible": jit_ok}

        # ---- plain / FD ----
        sq_spot = ql.SimpleQuote(_S0)
        sq_rate = ql.SimpleQuote(_R0)
        sq_div  = ql.SimpleQuote(_Q0)
        sq_vol  = ql.SimpleQuote(_VOL0)
        quotes  = [sq_spot, sq_rate, sq_div, sq_vol]

        proc = _make_process_plain(ql, todaysDate, sq_spot, sq_rate, sq_div, sq_vol)
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, _K),
            ql.AmericanExercise(todaysDate, ql.Date(17, ql.May, 1999)))
        _attach_engine(ql, option, proc, engine_name)

        base_npv = option.NPV()
        eng_results["npv"] = V(base_npv)

        # FD Greeks
        fd_greeks = []
        for q, v0 in zip(quotes, _INPUT_VALS):
            q.setValue(v0 + BPS)
            npv_up = option.NPV()
            q.setValue(v0)
            fd_greeks.append(V(npv_up - base_npv) / BPS)
        eng_results["fd_greeks"] = fd_greeks

        # Plain timing
        def _plain(qq=quotes):
            qq[0].setValue(_S0 + 1e-10)
            qq[0].setValue(_S0)
            return option.NPV()

        m, s = _median_ms(_plain, repeats)
        eng_results["plain"] = {"median": m, "stdev": s}

        # FD timing
        def _fd(opt=option, qq=quotes):
            opt.NPV()
            for q, v0 in zip(qq, _INPUT_VALS):
                q.setValue(v0 + BPS)
                opt.NPV()
                q.setValue(v0)

        m, s = _median_ms(_fd, repeats)
        eng_results["fd"] = {"median": m, "stdev": s}

        # ---- AAD ----
        tape = Tape()
        tape.activate()

        spot_v = ql.Real(_S0)
        rate_v = ql.Real(_R0)
        div_v  = ql.Real(_Q0)
        vol_v  = ql.Real(_VOL0)
        all_inputs = [spot_v, rate_v, div_v, vol_v]
        tape.registerInputs(all_inputs)
        tape.newRecording()

        proc_aad = _make_process_aad(ql, todaysDate, spot_v, rate_v, div_v, vol_v)
        option_aad = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, _K),
            ql.AmericanExercise(todaysDate, ql.Date(17, ql.May, 1999)))
        _attach_engine(ql, option_aad, proc_aad, engine_name)

        npv_aad = option_aad.NPV()
        tape.registerOutput(npv_aad)
        eng_results["aad_npv"] = V(npv_aad)

        # Get AAD Greeks
        tape.clearDerivatives()
        npv_aad.derivative = 1.0
        tape.computeAdjoints()
        aad_greeks = [V(xad.derivative(inp)) for inp in all_inputs]
        eng_results["aad_greeks"] = aad_greeks

        # AAD timing
        def _aad(t=tape, npv=npv_aad):
            t.clearDerivatives()
            npv.derivative = 1.0
            t.computeAdjoints()

        m, s = _median_ms(_aad, repeats)
        eng_results["aad"] = {"median": m, "stdev": s}
        tape.deactivate()

        results["engines"][label] = eng_results

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
    if b > 0:
        return f"{a / b:6.2f}x"
    return "   N/A"


def print_comparison(nojit: dict, jit: dict, repeats: int):
    print()
    print(SEPARATOR)
    print("American Option  –  FD vs AAD vs AAD+JIT Benchmark")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Instrument : American Put  S={_S0}, K={_K}, σ={_VOL0}, "
          f"r={_R0}, q={_Q0}, T=1Y")
    print(f"  Inputs     : {nojit['n_inputs']}  ({', '.join(_INPUT_NAMES)})")
    print(f"  Repeats    : {repeats}")
    print(f"  BPS shift  : {BPS}")
    print()

    for label, _, jit_ok in _ENGINES:
        nj_e = nojit["engines"][label]
        jt_e = jit["engines"][label]
        jit_tag = "JIT eligible" if jit_ok else "NOT JIT eligible (branching)"
        print(f"  [{label}]  {nj_e['engine']}  ({jit_tag})")
        print(f"    NPV = {nj_e['npv']:.10f}")
        print()

        # Greeks
        print(f"    {'Input':<16s}  {'FD':>14s}  {'AAD':>14s}  {'|Δ|':>12s}")
        print("    " + "-" * 60)
        for i, name in enumerate(_INPUT_NAMES):
            fd_g = nj_e['fd_greeks'][i]
            aad_g = nj_e['aad_greeks'][i]
            diff = abs(fd_g - aad_g)
            print(f"    {name:<16s}  {fd_g:14.8f}  {aad_g:14.8f}  {diff:12.2e}")
        print()

        # Timing
        col = 22
        for lbl, key in [
            ("Plain pricing",            "plain"),
            ("Bump-and-reprice FD (N+1)", "fd"),
            ("AAD backward pass",        "aad"),
        ]:
            nj = nj_e[key]
            jt = jt_e[key]
            sp = _sp(nj["median"], jt["median"])
            print(f"    {lbl:<28s}  "
                  f"{_fmt_t(nj['median'], nj['stdev'])}  "
                  f"{_fmt_t(jt['median'], jt['stdev'])}  "
                  f"{sp}")

        fd_aad_nojit = nj_e["fd"]["median"] / nj_e["aad"]["median"] if nj_e["aad"]["median"] else 0
        fd_aad_jit   = jt_e["fd"]["median"] / jt_e["aad"]["median"] if jt_e["aad"]["median"] else 0
        print(f"    FD÷AAD:  Non-JIT {fd_aad_nojit:.1f}x  |  JIT {fd_aad_jit:.1f}x")
        print()

    print(SEPARATOR)
    print()


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "american_option_benchmark_results.md"


def write_markdown(nojit: dict, jit: dict, repeats: int, wheels: dict):
    now = datetime.datetime.now()
    lines = []
    w = lines.append
    n = nojit["n_inputs"]

    w("# American Option — FD vs AAD vs AAD+JIT Benchmark Results")
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
    w(f"| Type | American Put |")
    w(f"| Spot (S) | {_S0} |")
    w(f"| Strike (K) | {_K} |")
    w(f"| Volatility (σ) | {_VOL0} |")
    w(f"| Risk-free rate (r) | {_R0} |")
    w(f"| Dividend yield (q) | {_Q0} |")
    w(f"| Maturity | 1 year |")
    w(f"| Inputs | {n} ({', '.join(_INPUT_NAMES)}) |")
    w("")
    w("---")
    w("")

    for label, engine_name, jit_ok in _ENGINES:
        nj_e = nojit["engines"][label]
        jt_e = jit["engines"][label]
        jit_tag = "Yes" if jit_ok else "**No** (PDE branching)"

        w(f"## {label} — `{engine_name}`")
        w("")
        w(f"JIT eligible: {jit_tag}  ")
        w(f"NPV = {nj_e['npv']:.10f}")
        w("")

        w("### Greeks validation")
        w("")
        w("| Input | FD (1 bp) | AAD | \|Δ\| |")
        w("|---|---:|---:|---:|")
        for i, name in enumerate(_INPUT_NAMES):
            fd_g = nj_e['fd_greeks'][i]
            aad_g = nj_e['aad_greeks'][i]
            diff = abs(fd_g - aad_g)
            w(f"| {name} | {fd_g:.8f} | {aad_g:.8f} | {diff:.2e} |")
        w("")

        w("### Timing")
        w("")
        w("| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |")
        w("|---|---:|---:|---:|")
        for lbl, key in [
            ("Plain pricing", "plain"),
            ("Bump-and-reprice FD (N+1)", "fd"),
            ("**AAD backward pass**", "aad"),
        ]:
            nj = nj_e[key]
            jt = jt_e[key]
            sp = f"{nj['median'] / jt['median']:.2f}×" if jt['median'] > 0 else "—"
            w(f"| {lbl} | {nj['median']:.4f} ±{nj['stdev']:.4f} "
              f"| {jt['median']:.4f} ±{jt['stdev']:.4f} | {sp} |")

        fd_aad_nojit = nj_e["fd"]["median"] / nj_e["aad"]["median"] if nj_e["aad"]["median"] else 0
        fd_aad_jit   = jt_e["fd"]["median"] / jt_e["aad"]["median"] if jt_e["aad"]["median"] else 0
        w(f"| *FD ÷ AAD* | *{fd_aad_nojit:.1f}×* | *{fd_aad_jit:.1f}×* | — |")
        w("")
        w("---")
        w("")

    w("## Analysis")
    w("")
    w("The **analytic approximation** engines (BAW, Bjerksund-Stensland, QD+) use")
    w("closed-form or quasi-analytic formulae with no branching in the computation")
    w("graph, making them **JIT eligible**.  The **FdBlackScholesVanillaEngine** solves")
    w("a PDE on a grid with conditional logic (boundary conditions, early exercise")
    w("checks), so it is **not JIT eligible** — the Forge compiler cannot trace")
    w("through branches.")
    w("")
    w(f"With only {n} inputs (S, r, q, σ), FD requires {n+1} forward pricings while")
    w(f"AAD needs only 1 backward sweep regardless of input count.")
    w("")
    w("---")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("python benchmarks/american_option_benchmarks.py")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="American option FD vs AAD vs AAD+JIT benchmark",
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
    print("American Option  –  FD vs AAD vs AAD+JIT Benchmark")
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

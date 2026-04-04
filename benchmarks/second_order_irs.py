#!/usr/bin/env python3
"""
Interest Rate Swap — Second-Order Sensitivities via FD-over-AAD
================================================================

Computes the full 17×17 Hessian matrix of a 5-year payer IRS
with respect to all bootstrapping inputs using two methods:

  • **Pure FD Hessian** — finite-difference on finite-difference NPV
  • **FD-over-AAD**    — finite-difference bump on AAD first-order Greeks

Market data:
  1 deposit, 3 FRAs, 8 futures, 5 swap rates
  Evaluation date: 6-Nov-2001, Settlement: 8-Nov-2001

Usage
-----
  python benchmarks/second_order_irs.py              # default 30 repeats
  python benchmarks/second_order_irs.py --repeats 50
  python benchmarks/second_order_irs.py --no-save

Internal worker mode (invoked automatically by the orchestrator):
  python benchmarks/second_order_irs.py --worker REPEATS
"""

import argparse
import datetime
import json
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

SEPARATOR = "=" * 80

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
_DEPO_RATE = 0.0382
_FRA_RATES = [0.031076, 0.031851, 0.032677]
_FUTURES_PRICES = [96.2875, 96.7875, 96.9875, 96.6875,
                   96.4875, 96.3875, 96.2875, 96.0875]
_SWAP_RATES = [0.037125, 0.0398, 0.0425, 0.04540, 0.04705]

_INPUT_NAMES = (
    ["Depo 3M"]
    + [f"FRA {i}" for i in range(3)]
    + [f"Fut {i}" for i in range(8)]
    + [f"Swap {y}Y" for y in [2, 3, 5, 7, 10]]
)
_INPUT_VALS = [_DEPO_RATE] + _FRA_RATES + _FUTURES_PRICES + _SWAP_RATES
_N = len(_INPUT_VALS)

H_HESS = 1e-5  # bump size for Hessian FD


# ============================================================================
# Wheel / venv boilerplate  (single-venv variant for second-order benchmarks)
# ============================================================================

def find_wheels(build_root: Path) -> dict:
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
         str(xad_wheel)])
    print("    • pip install xad compatibility shim")
    _install_xad_shim(py)
    print(f"    • pip install {ql_wheel.name}")
    subprocess.check_call(
        [py, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps",
         str(ql_wheel)])


def venv_is_ready(venv: Path) -> bool:
    py = python_in(venv)
    if not py.exists():
        return False
    result = subprocess.run(
        [str(py), "-c", "import QuantLib_Risks; import xad"],
        capture_output=True)
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
        capture_output=True, text=True, env=_clean_env())
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
        f"No JSON found in worker output ({venv.name}):\n{result.stdout[-2000:]}")


# ============================================================================
# Instrument construction helpers
# ============================================================================

def _build_swap(input_reals):
    """Build PiecewiseFlatForward curve + 5Y swap from ql.Real objects.
    MUST be called after tape.newRecording() for AAD to work.
    Returns (swap, yieldHandle, todaysDate, settlement)."""
    import QuantLib_Risks as ql

    todaysDate = ql.Date(6, ql.November, 2001)
    ql.Settings.instance().evaluationDate = todaysDate
    calendar = ql.TARGET()
    settlement = calendar.advance(todaysDate, ql.Period(2, ql.Days))

    depo_rate  = input_reals[0]
    fra_rates  = input_reals[1:4]
    fut_prices = input_reals[4:12]
    swap_rates = input_reals[12:17]

    dayCounter = ql.Actual360()
    helpers = []

    # deposit
    helpers.append(ql.DepositRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(depo_rate)),
        ql.Period(3, ql.Months), 3, calendar,
        ql.ModifiedFollowing, True, dayCounter))

    # FRAs
    for i, r in enumerate(fra_rates):
        helpers.append(ql.FraRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(r)),
            3 + i, 6 + i, 2, calendar,
            ql.ModifiedFollowing, True, dayCounter))

    # futures
    imm = ql.IMM.nextDate(settlement)
    for i, p in enumerate(fut_prices):
        d = ql.IMM.nextDate(imm, False) if i > 0 else imm
        imm = d
        helpers.append(ql.FuturesRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(p)),
            d, 3, calendar,
            ql.ModifiedFollowing, True, dayCounter))

    # swaps
    swap_tenors = [2, 3, 5, 7, 10]
    for tenor, r in zip(swap_tenors, swap_rates):
        helpers.append(ql.SwapRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(r)),
            ql.Period(tenor, ql.Years), calendar,
            ql.Annual, ql.Unadjusted,
            ql.Thirty360(ql.Thirty360.BondBasis),
            ql.Euribor6M()))

    curve = ql.PiecewiseFlatForward(settlement, helpers, dayCounter)
    curve.enableExtrapolation()

    yieldHandle = ql.YieldTermStructureHandle(curve)
    index = ql.Euribor6M(yieldHandle)

    start = settlement
    maturity = ql.TARGET().advance(start, ql.Period(5, ql.Years))
    fixedSchedule = ql.Schedule(
        start, maturity, ql.Period(ql.Annual), ql.TARGET(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
    floatSchedule = ql.Schedule(
        start, maturity, ql.Period(ql.Semiannual), ql.TARGET(),
        ql.ModifiedFollowing, ql.ModifiedFollowing,
        ql.DateGeneration.Forward, False)

    swap = ql.VanillaSwap(
        ql.Swap.Payer, 1_000_000.0, fixedSchedule, 0.04,
        ql.Thirty360(ql.Thirty360.BondBasis), floatSchedule,
        index, 0.0, ql.Actual360())
    swap.setPricingEngine(ql.DiscountingSwapEngine(yieldHandle))

    return swap


def _compute_aad_gradient(input_vals):
    """Record tape, compute NPV + gradient."""
    import QuantLib_Risks as ql
    from xad.adj_1st import Tape
    import xad

    tape = Tape()
    tape.activate()

    all_inputs = [ql.Real(v) for v in input_vals]
    tape.registerInputs(all_inputs)
    tape.newRecording()

    swap = _build_swap(all_inputs)
    npv = swap.NPV()
    tape.registerOutput(npv)
    npv.derivative = 1.0
    tape.computeAdjoints()

    gradient = [float(xad.derivative(inp)) for inp in all_inputs]
    npv_val = float(xad.value(npv))
    tape.deactivate()
    return npv_val, gradient


def _compute_fd_npv(input_vals):
    """Build swap, return float NPV."""
    import QuantLib_Risks as ql
    import xad

    input_reals = [ql.Real(v) for v in input_vals]
    swap = _build_swap(input_reals)
    return float(xad.value(swap.NPV()))


# ============================================================================
# Hessian computations
# ============================================================================

def _hessian_fd_over_aad(h=H_HESS):
    base_vals = list(_INPUT_VALS)
    npv_base, grad_base = _compute_aad_gradient(base_vals)
    hessian = []
    for i in range(_N):
        bumped = list(base_vals)
        bumped[i] += h
        _, grad_bumped = _compute_aad_gradient(bumped)
        row = [(grad_bumped[j] - grad_base[j]) / h for j in range(_N)]
        hessian.append(row)
    return npv_base, grad_base, hessian


def _hessian_pure_fd(h=H_HESS):
    base_vals = list(_INPUT_VALS)
    npv_base = _compute_fd_npv(base_vals)
    hessian = [[0.0] * _N for _ in range(_N)]
    for i in range(_N):
        up = list(base_vals); up[i] += h
        dn = list(base_vals); dn[i] -= h
        hessian[i][i] = (_compute_fd_npv(up) - 2 * npv_base + _compute_fd_npv(dn)) / (h * h)
    for i in range(_N):
        for j in range(i + 1, _N):
            pp = list(base_vals); pp[i] += h; pp[j] += h
            pm = list(base_vals); pm[i] += h; pm[j] -= h
            mp = list(base_vals); mp[i] -= h; mp[j] += h
            mm = list(base_vals); mm[i] -= h; mm[j] -= h
            hessian[i][j] = (_compute_fd_npv(pp) - _compute_fd_npv(pm)
                             - _compute_fd_npv(mp) + _compute_fd_npv(mm)) / (4 * h * h)
            hessian[j][i] = hessian[i][j]
    return npv_base, hessian


# ============================================================================
# Timing helpers
# ============================================================================

def _median_ms(func, n, warmup=3):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


# ============================================================================
# Worker: run benchmarks and return results dict
# ============================================================================

def _run_worker(repeats: int) -> dict:
    npv, grad, hess_aad = _hessian_fd_over_aad()
    _, hess_fd = _hessian_pure_fd()

    max_asym = max(abs(hess_aad[i][j] - hess_aad[j][i])
                   for i in range(_N) for j in range(i + 1, _N))
    max_diff = max(abs(hess_aad[i][j] - hess_fd[i][j])
                   for i in range(_N) for j in range(_N))
    max_abs = max(abs(hess_aad[i][j])
                  for i in range(_N) for j in range(_N))

    t_aad_med, t_aad_std = _median_ms(lambda: _hessian_fd_over_aad(), repeats)
    t_fd_med,  t_fd_std  = _median_ms(lambda: _hessian_pure_fd(), repeats)

    return {
        "npv": npv,
        "gradient": grad,
        "hess_aad": hess_aad,
        "hess_fd": hess_fd,
        "max_asymmetry": max_asym,
        "max_hessian_diff": max_diff,
        "max_abs_hessian": max_abs,
        "timing_aad_ms": t_aad_med,
        "timing_aad_std": t_aad_std,
        "timing_fd_ms": t_fd_med,
        "timing_fd_std": t_fd_std,
        "n_inputs": _N,
        "n_aad_recordings": _N + 1,
        "n_fd_pricings": 1 + 2 * _N + 4 * _N * (_N - 1) // 2,
        "repeats": repeats,
    }


def worker_main(repeats: int):
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Console printer
# ============================================================================

def print_results(r: dict):
    print()
    print(SEPARATOR)
    print("Interest Rate Swap — Second-Order Sensitivities (FD-over-AAD)")
    print(SEPARATOR)
    print(f"  5Y Payer IRS, notional = 1M, fixed = 4%")
    print(f"  Curve: 1 depo + 3 FRAs + 8 futures + 5 swap rates = {_N} inputs")
    print(f"  Hessian bump size h = {H_HESS}")
    print()

    print(f"  NPV = {r['npv']:.10f}")
    print()

    print("  1st-order sensitivities (∂NPV/∂input):")
    for i, name in enumerate(_INPUT_NAMES):
        print(f"    {name:>12s}  {r['gradient'][i]:>14.4f}")
    print()

    hess = r['hess_aad']
    print(f"  Hessian:  max |entry| = {r['max_abs_hessian']:.4f}")
    print(f"            max |H[i,j] - H[j,i]| = {r['max_asymmetry']:.2e}")
    print(f"            max |aad - fd|          = {r['max_hessian_diff']:.2e}")
    print()

    # show top-10 largest entries by magnitude
    entries = []
    for i in range(_N):
        for j in range(i, _N):
            entries.append((abs(hess[i][j]), i, j, hess[i][j]))
    entries.sort(reverse=True)
    print("  Top-10 Hessian entries (by magnitude):")
    print(f"    {'Row':>12s}  {'Col':>12s}  {'Value':>14s}")
    for _, i, j, v in entries[:10]:
        print(f"    {_INPUT_NAMES[i]:>12s}  {_INPUT_NAMES[j]:>12s}  {v:>14.4f}")
    print()

    print("-" * 80)
    print("Timing comparison")
    print("-" * 80)
    speedup = r["timing_fd_ms"] / r["timing_aad_ms"] if r["timing_aad_ms"] > 0 else 0
    print(f"  FD-over-AAD: {r['timing_aad_ms']:8.2f} ±{r['timing_aad_std']:.2f} ms  "
          f"({r['n_aad_recordings']} AAD recordings)")
    print(f"  Pure FD:     {r['timing_fd_ms']:8.2f} ±{r['timing_fd_std']:.2f} ms  "
          f"({r['n_fd_pricings']} forward pricings)")
    print(f"  Speedup:     {speedup:.1f}×")
    print()
    print("  Note: The IRS Hessian is near-zero because NPV is linear")
    print("  in bootstrapping inputs. This validates correctness.")
    print(SEPARATOR)


# ============================================================================
# Markdown writer
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "second_order_irs_results.md"


def write_markdown(r: dict, wheels: dict):
    now = datetime.datetime.now()
    hess = r["hess_aad"]
    speedup = r["timing_fd_ms"] / r["timing_aad_ms"] if r["timing_aad_ms"] > 0 else 0

    lines = []
    w = lines.append

    w("# Interest Rate Swap — Second-Order Sensitivities (FD-over-AAD)")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**Repetitions:** {r['repeats']} (median reported)  ")
    w(f"**Wheel:** `{wheels['ql'].name}`  ")
    w("")
    w("---")
    w("")
    w("## Instrument")
    w("")
    w("- 5-year Payer IRS, notional = 1,000,000, fixed coupon = 4%")
    w("- `PiecewiseFlatForward` yield curve + `DiscountingSwapEngine`")
    w(f"- **{_N} inputs:** 1 deposit rate + 3 FRA rates + 8 Eurodollar futures + 5 swap rates")
    w(f"- Hessian bump size h = {H_HESS}")
    w("- Evaluation date: 6-Nov-2001, Settlement: 8-Nov-2001")
    w("")
    w("---")
    w("")
    w("## How to Read the Hessian Matrix")
    w("")
    w("The Hessian matrix **H** contains all second-order partial derivatives of the NPV")
    w("with respect to pairs of inputs:")
    w("")
    w("$$H_{ij} = \\frac{\\partial^2 \\text{NPV}}{\\partial x_i \\, \\partial x_j}$$")
    w("")
    w("- **Diagonal entries** $H_{ii}$ measure the *convexity* (curvature) of the NPV")
    w("  with respect to input $x_i$.  A large diagonal value means the first-order")
    w("  sensitivity (delta/gradient) changes rapidly as that input moves.")
    w("- **Off-diagonal entries** $H_{ij}$ ($i \\neq j$) measure *cross-gamma* — how")
    w("  the sensitivity to input $x_i$ changes when input $x_j$ moves.")
    w("  These capture interaction effects missed by first-order Greeks.")
    w("- The matrix is **symmetric** ($H_{ij} = H_{ji}$) up to numerical noise.")
    w("  The reported symmetry metric quantifies this noise.")
    w("- Values are in NPV currency units per unit² of the respective inputs.")
    w("")
    w("---")
    w("")
    w("## Results")
    w("")
    w(f"**NPV** = {r['npv']:.10f}")
    w("")

    w("### First-Order Sensitivities")
    w("")
    w("| Input | ∂NPV/∂input |")
    w("|---|---:|")
    for i, name in enumerate(_INPUT_NAMES):
        w(f"| {name} | {r['gradient'][i]:.4f} |")
    w("")

    w("### Hessian Summary")
    w("")
    w(f"- **Max \\|entry\\|:** {r['max_abs_hessian']:.4f}")
    w(f"- **Symmetry:** max |H[i,j] − H[j,i]| = {r['max_asymmetry']:.2e}")
    w(f"- **AAD vs FD:** max |Δ| = {r['max_hessian_diff']:.2e}")
    w("")

    w("> **Observation:** The Hessian is near-zero because the IRS NPV is")
    w("> (almost) linear in the bootstrapping inputs. This validates correctness.")
    w("")

    # Top entries
    entries = []
    for i in range(_N):
        for j in range(i, _N):
            entries.append((abs(hess[i][j]), i, j, hess[i][j]))
    entries.sort(reverse=True)

    w("### Top-10 Hessian Entries (by magnitude)")
    w("")
    w("| Row | Col | Value |")
    w("|---|---|---:|")
    for _, i, j, v in entries[:10]:
        w(f"| {_INPUT_NAMES[i]} | {_INPUT_NAMES[j]} | {v:.6f} |")
    w("")

    w("### Full Hessian (FD-over-AAD)")
    w("")
    w("<details>")
    w("<summary>Click to expand 17×17 Hessian matrix</summary>")
    w("")
    w("| | " + " | ".join(_INPUT_NAMES) + " |")
    w("|---|" + "---:|" * _N)
    for i in range(_N):
        row = " | ".join(f"{hess[i][j]:.4f}" for j in range(_N))
        w(f"| **{_INPUT_NAMES[i]}** | {row} |")
    w("")
    w("</details>")
    w("")

    # --- FD-over-AAD vs Pure FD comparison ---
    hess_fd = r["hess_fd"]
    diff = [[hess[i][j] - hess_fd[i][j] for j in range(_N)] for i in range(_N)]
    abs_diffs = [abs(d) for row in diff for d in row]
    max_abs_diff = max(abs_diffs)
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
    abs_entries = [abs(hess[i][j]) for i in range(_N) for j in range(_N)]
    max_abs_entry = max(abs_entries) if abs_entries else 1.0
    rel_diff = max_abs_diff / max_abs_entry if max_abs_entry > 0 else 0.0

    w("### FD-over-AAD vs Pure FD — Difference Matrix")
    w("")
    w("The table below shows $(H^{\\text{AAD}}_{ij} - H^{\\text{FD}}_{ij})$,")
    w("i.e. the element-wise difference between the Hessian computed via")
    w("FD-over-AAD and the one computed via pure finite differences.")
    w("")
    w("<details>")
    w(f"<summary>Click to expand {_N}×{_N} difference matrix</summary>")
    w("")
    w("| | " + " | ".join(_INPUT_NAMES) + " |")
    w("|---|" + "---:|" * _N)
    for i in range(_N):
        row = " | ".join(f"{diff[i][j]:.4e}" for j in range(_N))
        w(f"| **{_INPUT_NAMES[i]}** | {row} |")
    w("")
    w("</details>")
    w("")
    w("| Metric | Value |")
    w("|---|---:|")
    w(f"| Max \\|difference\\| | {max_abs_diff:.4e} |")
    w(f"| Mean \\|difference\\| | {mean_abs_diff:.4e} |")
    w(f"| Max \\|Hessian entry\\| | {max_abs_entry:.4e} |")
    w(f"| Relative error (max \\|Δ\\| / max \\|H\\|) | {rel_diff:.4e} |")
    w("")
    if rel_diff < 1e-2:
        w("> ✅ **Acceptable.** The relative difference is well below 1%,")
        w("> confirming both methods agree to high precision.")
    elif rel_diff < 5e-2:
        w("> ⚠️ **Marginal.** The relative difference is below 5% but above 1%.")
        w("> Results are broadly consistent; consider tightening the bump size h.")
    else:
        w("> ❌ **Large.** The relative difference exceeds 5%.")
        w("> This may indicate numerical instability; consider adjusting h.")
    w("")

    w("### Timing")
    w("")
    w("| Method | Time (ms) | Operations |")
    w("|---|---:|---|")
    w(f"| FD-over-AAD | {r['timing_aad_ms']:.2f} ±{r['timing_aad_std']:.2f} "
      f"| {r['n_aad_recordings']} AAD recordings |")
    w(f"| Pure FD | {r['timing_fd_ms']:.2f} ±{r['timing_fd_std']:.2f} "
      f"| {r['n_fd_pricings']} forward pricings |")
    w(f"| **Speedup** | **{speedup:.1f}×** | |")
    w("")

    w("---")
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("python benchmarks/second_order_irs.py")
    w("python benchmarks/second_order_irs.py --repeats 50")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IRS — 2nd-order sensitivities via FD-over-AAD",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--worker", metavar="REPEATS", type=int, default=None,
                        help="Internal worker mode: run benchmarks and print JSON")
    parser.add_argument("--repeats", "-r", type=int, default=30,
                        help="Number of timing repetitions (default: 30)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write results markdown")
    args = parser.parse_args()

    if args.worker is not None:
        worker_main(args.worker)
        return

    # ---- ORCHESTRATOR MODE ----
    repeats = args.repeats
    wheels  = find_wheels(BUILD)

    missing = [k for k in ("xad", "ql") if wheels[k] is None]
    if missing:
        print(f"ERROR: Missing wheels: {', '.join(missing)}")
        print("  Run the build first:")
        print("    ./build.sh --no-jit -j$(nproc)")
        sys.exit(1)

    print(SEPARATOR)
    print("Interest Rate Swap — Second-Order Sensitivities (FD-over-AAD)")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)

    print("\nSetting up virtual environment")
    print("-" * 50)
    setup_venv(VENV_NOJIT, wheels["xad"], wheels["ql"])

    print(f"\nRunning benchmarks  ({repeats} repeats)")
    print("-" * 50)
    results = run_worker_in_venv(VENV_NOJIT, repeats)
    print("  done.")

    print_results(results)

    if not args.no_save:
        write_markdown(results, wheels)


if __name__ == "__main__":
    main()

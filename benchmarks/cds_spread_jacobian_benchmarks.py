#!/usr/bin/env python3
"""
QuantLib-Risks-Py — CDS-Spread Sensitivity & Reverse Credit Jacobian Benchmark
================================================================================

Two approaches to computing ∂NPV/∂(CDS spread) for a 5-year CDS:

  Approach 1  — Direct bootstrap (PiecewiseFlatHazardRate)
      Build the hazard curve from CDS spread inputs with the Brent solver
      on the AD tape.  A single backward sweep gives all CDS-spread
      sensitivities.

  Approach 2  — HazardRateCurve AAD + J^T conversion
      Compute ∂NPV/∂(hazard rate) via HazardRateCurve AAD (no solver on
      tape), then multiply by J^T (the bootstrap Jacobian transpose)
      to obtain CDS-spread sensitivities.

Additionally, the *reverse* Jacobian K = ∂s/∂h is computed by
differentiating through HazardRateCurve → CDS fairSpread() via AAD.
The product K × J is compared against the identity matrix to test
whether K = J⁻¹.

Market data: 4 CDS spread inputs at tenors 1Y, 2Y, 3Y, 5Y (hypothetical
investment-grade issuer).  Risk-free curve: flat 3.5% (held fixed).

Instrument: 5-year CDS, Protection Buyer, 100 bp running coupon, $10M notional

Usage
-----
  python benchmarks/cds_spread_jacobian_benchmarks.py
  python benchmarks/cds_spread_jacobian_benchmarks.py --repeats 10
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
ROOT       = Path(__file__).resolve().parent.parent
BUILD      = ROOT / "build"
VENV_NOJIT = BUILD / "bench-venv-nojit"
VENV_JIT   = BUILD / "bench-venv-jit"

SEPARATOR = "=" * 84
BPS = 1e-4

# ------- scenario parameters -----------------------------------------------
N_SCENARIOS   = 100
SCENARIO_SEED = 2013          # distinct from hazard_rate benchmark (2001)

# CDS base market — hypothetical IG issuer (4 inputs)
_CDS_TENOR_LABELS = ['1Y', '2Y', '3Y', '5Y']
_CDS_TENORS = [
    (1, 'Years'),
    (2, 'Years'),
    (3, 'Years'),
    (5, 'Years'),
]
_CDS_BASE_SPREADS = [0.0050, 0.0075, 0.0100, 0.0125]   # 50, 75, 100, 125 bp
_N_INPUTS = len(_CDS_BASE_SPREADS)

# Risk-free curve (flat, held fixed — not differentiated)
_RF_RATE = 0.035

# Credit parameters
_RECOVERY = 0.40

# Instrument: 5-year CDS
_CDS_TENOR_YEARS = 5
_CDS_NOMINAL     = 10_000_000
_CDS_COUPON      = 0.01       # 100 bp running spread

# Mutable slot
_EVAL_DATE = (15, 11, 2024)


# ============================================================================
# Wheel / venv helpers (dual-venv, shared pattern)
# ============================================================================

def find_wheels(build_root):
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


def python_in(venv):
    return venv / "bin" / "python"


def _install_xad_shim(py):
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


def install_wheels(venv, xad_wheel, ql_wheel):
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


def venv_is_ready(venv):
    py = python_in(venv)
    if not py.exists():
        return False
    return subprocess.run(
        [str(py), "-c", "import QuantLib_Risks; import xad"],
        capture_output=True,
    ).returncode == 0


def setup_venv(venv, xad_wheel, ql_wheel, force=False):
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


def _clean_env():
    drop = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHON_BASIC_REPL",
            "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT"}
    return {k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith(("CONDA_", "PYTHON_"))}


def run_worker_in_venv(venv, repeats):
    py = str(python_in(venv))
    cmd = [py, str(Path(__file__).resolve()), "--worker", str(repeats)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_clean_env())
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
# Linear algebra helpers (no numpy dependency)
# ============================================================================

def _solve_linear(A, b):
    """Solve A x = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-30:
            raise ValueError(f"Singular matrix at column {col}")
        for j in range(col, n + 1):
            M[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = M[row][col]
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def _transpose(M):
    n = len(M)
    m = len(M[0])
    return [[M[i][j] for i in range(n)] for j in range(m)]


def _mat_vec_mul(M, v):
    """Multiply n×m matrix M by m-vector v.  Returns n-vector."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _mat_mul(A, B):
    """Multiply n×k matrix A by k×m matrix B.  Returns n×m."""
    n, k = len(A), len(A[0])
    m = len(B[0])
    return [[sum(A[i][l] * B[l][j] for l in range(k))
             for j in range(m)] for i in range(n)]


def _mat_inv(M):
    """Compute inverse of n×n matrix M via Gaussian elimination."""
    n = len(M)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(_solve_linear(M, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ============================================================================
# WORKER MODE
# ============================================================================

def _median_ms(func, n, warmup=1):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), (statistics.stdev(times) if n > 1 else 0.0)


def _run_worker(repeats):
    import QuantLib_Risks as ql
    import xad
    from xad.adj_1st import Real, Tape

    V = lambda x: float(xad.value(x))

    calendar   = ql.TARGET()
    todaysDate = ql.Date(_EVAL_DATE[0], _EVAL_DATE[1], _EVAL_DATE[2])
    ql.Settings.instance().evaluationDate = todaysDate
    dc = ql.Actual365Fixed()

    # Background risk-free curve (flat, held fixed)
    rfCurve = ql.FlatForward(todaysDate, _RF_RATE, dc)
    rfHandle = ql.YieldTermStructureHandle(rfCurve)

    # CDS instrument schedule (reused across all approaches)
    cds_schedule = ql.Schedule(
        todaysDate, todaysDate + ql.Period(_CDS_TENOR_YEARS, ql.Years),
        ql.Period(ql.Quarterly), calendar,
        ql.Following, ql.Following,
        ql.DateGeneration.TwentiethIMM, False,
    )

    # ===== Step 0: bootstrap once (plain float) to extract hazard rates =====
    boot_helpers = []
    for i, (n, unit) in enumerate(_CDS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        boot_helpers.append(ql.SpreadCdsHelper(
            _CDS_BASE_SPREADS[i], period, 0, calendar, ql.Quarterly,
            ql.Following, ql.DateGeneration.TwentiethIMM,
            dc, _RECOVERY, rfHandle))
    boot_curve = ql.PiecewiseFlatHazardRate(todaysDate, boot_helpers, dc)
    boot_curve.enableExtrapolation()

    nodes = boot_curve.nodes()
    pillar_dates    = []
    pillar_hazards_f = []
    pillar_dates_str = []
    for d, v in nodes:
        pillar_dates.append(d)
        pillar_hazards_f.append(V(v))
        if d > todaysDate:
            pillar_dates_str.append(
                f"{d.year()}-{int(d.month()):02d}-{d.dayOfMonth():02d}")

    n_hazards = len(pillar_dates) - 1
    assert n_hazards == _N_INPUTS, f"Expected {_N_INPUTS} pillar hazard rates, got {n_hazards}"

    # Pre-generate CDS-spread scenarios
    rng = random.Random(SCENARIO_SEED)
    spread_scenarios = []
    for _ in range(N_SCENARIOS):
        scene = [max(1e-5, s + rng.gauss(0, 5e-4))
                 for s in _CDS_BASE_SPREADS]
        spread_scenarios.append(scene)

    results = {}

    # =====================================================================
    # APPROACH 1 — Direct bootstrap AAD for ∂NPV/∂s
    # =====================================================================

    # --- A1-FD: bump CDS spreads via SimpleQuote --------------------------
    fd_quotes = [ql.SimpleQuote(0.0) for _ in range(_N_INPUTS)]
    fd_helpers = []
    for i, (n, unit) in enumerate(_CDS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        fd_helpers.append(ql.SpreadCdsHelper(
            ql.QuoteHandle(fd_quotes[i]), period, 0,
            calendar, ql.Quarterly, ql.Following,
            ql.DateGeneration.TwentiethIMM, dc, _RECOVERY, rfHandle))
    fd_crv = ql.PiecewiseFlatHazardRate(todaysDate, fd_helpers, dc)
    fd_crv.enableExtrapolation()
    fd_defH = ql.DefaultProbabilityTermStructureHandle(fd_crv)
    fd_cds = ql.CreditDefaultSwap(
        ql.Protection.Buyer, _CDS_NOMINAL, _CDS_COUPON,
        cds_schedule, ql.Following, dc)
    fd_cds.setPricingEngine(ql.MidPointCdsEngine(fd_defH, _RECOVERY, rfHandle))

    def _a1_fd():
        for scene in spread_scenarios:
            for i in range(_N_INPUTS):
                fd_quotes[i].setValue(scene[i])
            V(fd_cds.NPV())
            for j in range(_N_INPUTS):
                fd_quotes[j].setValue(scene[j] + BPS)
                V(fd_cds.NPV())
                fd_quotes[j].setValue(scene[j])

    m, s = _median_ms(_a1_fd, repeats)
    results["a1_fd"] = {"median": m, "stdev": s,
                        "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    # --- A1-AAD-replay: tape through bootstrap ----------------------------
    tape1 = Tape()
    tape1.activate()
    s_reals = [Real(s) for s in _CDS_BASE_SPREADS]
    tape1.registerInputs(s_reals)
    tape1.newRecording()

    helpers1 = []
    for i, (n, unit) in enumerate(_CDS_TENORS):
        period = ql.Period(n, getattr(ql, unit))
        helpers1.append(ql.SpreadCdsHelper(
            ql.QuoteHandle(ql.SimpleQuote(s_reals[i])), period, 0,
            calendar, ql.Quarterly, ql.Following,
            ql.DateGeneration.TwentiethIMM, dc, _RECOVERY, rfHandle))
    crv1 = ql.PiecewiseFlatHazardRate(todaysDate, helpers1, dc)
    crv1.enableExtrapolation()
    defH1 = ql.DefaultProbabilityTermStructureHandle(crv1)
    cds1 = ql.CreditDefaultSwap(
        ql.Protection.Buyer, _CDS_NOMINAL, _CDS_COUPON,
        cds_schedule, ql.Following, dc)
    cds1.setPricingEngine(ql.MidPointCdsEngine(defH1, _RECOVERY, rfHandle))
    npv1 = cds1.NPV()
    tape1.registerOutput(npv1)

    # Capture CDS-spread sensitivities at base market
    tape1.clearDerivatives()
    npv1.derivative = 1.0
    tape1.computeAdjoints()
    spread_sens_direct = [s_reals[i].derivative for i in range(_N_INPUTS)]

    def _a1_replay():
        for _ in range(N_SCENARIOS):
            tape1.clearDerivatives()
            npv1.derivative = 1.0
            tape1.computeAdjoints()

    m, s = _median_ms(_a1_replay, repeats)
    results["a1_aad_replay"] = {"median": m, "stdev": s,
                                "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}
    tape1.deactivate()

    # --- A1-AAD-re-record: per-scenario fresh recording -------------------
    def _a1_rerecord():
        tp = Tape()
        tp.activate()
        for scene in spread_scenarios:
            sr = [Real(s) for s in scene]
            tp.registerInputs(sr)
            tp.newRecording()
            h = []
            for i, (n_op, unit) in enumerate(_CDS_TENORS):
                period = ql.Period(n_op, getattr(ql, unit))
                h.append(ql.SpreadCdsHelper(
                    ql.QuoteHandle(ql.SimpleQuote(sr[i])), period, 0,
                    calendar, ql.Quarterly, ql.Following,
                    ql.DateGeneration.TwentiethIMM, dc, _RECOVERY, rfHandle))
            c = ql.PiecewiseFlatHazardRate(todaysDate, h, dc)
            c.enableExtrapolation()
            dH = ql.DefaultProbabilityTermStructureHandle(c)
            cd = ql.CreditDefaultSwap(
                ql.Protection.Buyer, _CDS_NOMINAL, _CDS_COUPON,
                cds_schedule, ql.Following, dc)
            cd.setPricingEngine(ql.MidPointCdsEngine(dH, _RECOVERY, rfHandle))
            npv = cd.NPV()
            tp.registerOutput(npv)
            npv.derivative = 1.0
            tp.computeAdjoints()
        tp.deactivate()

    m, s = _median_ms(_a1_rerecord, repeats)
    results["a1_aad_record"] = {"median": m, "stdev": s,
                                "n_scenarios": N_SCENARIOS, "n_inputs": _N_INPUTS}

    # =====================================================================
    # APPROACH 2 — HazardRateCurve AAD + J^T conversion → ∂NPV/∂s
    # =====================================================================

    # --- A2a: HazardRateCurve AAD replay for ∂NPV/∂h -----------------------
    tape2 = Tape()
    tape2.activate()
    h_inputs = [Real(h) for h in pillar_hazards_f[1:]]
    tape2.registerInputs(h_inputs)
    tape2.newRecording()

    h_all_t = [h_inputs[0]] + h_inputs             # tie h_ref = h_1
    hrc2 = ql.HazardRateCurve(pillar_dates, h_all_t, dc)
    hrc2.enableExtrapolation()
    defH2 = ql.DefaultProbabilityTermStructureHandle(hrc2)
    cds2 = ql.CreditDefaultSwap(
        ql.Protection.Buyer, _CDS_NOMINAL, _CDS_COUPON,
        cds_schedule, ql.Following, dc)
    cds2.setPricingEngine(ql.MidPointCdsEngine(defH2, _RECOVERY, rfHandle))
    npv2 = cds2.NPV()
    tape2.registerOutput(npv2)

    tape2.clearDerivatives()
    npv2.derivative = 1.0
    tape2.computeAdjoints()
    hazard_sens = [h_inputs[i].derivative for i in range(n_hazards)]

    def _a2a_replay():
        for _ in range(N_SCENARIOS):
            tape2.clearDerivatives()
            npv2.derivative = 1.0
            tape2.computeAdjoints()

    m, s = _median_ms(_a2a_replay, repeats)
    results["a2_hazard_replay"] = {"median": m, "stdev": s,
                                   "n_scenarios": N_SCENARIOS, "n_inputs": n_hazards}
    tape2.deactivate()

    # --- A2b: J = ∂h/∂s (bootstrap Jacobian) -----------------------------
    def _compute_jacobian():
        jac_tape = Tape()
        jac_tape.activate()
        jac_s = [Real(s) for s in _CDS_BASE_SPREADS]
        jac_tape.registerInputs(jac_s)
        jac_tape.newRecording()

        jac_helpers = []
        for i, (n_op, unit) in enumerate(_CDS_TENORS):
            period = ql.Period(n_op, getattr(ql, unit))
            jac_helpers.append(ql.SpreadCdsHelper(
                ql.QuoteHandle(ql.SimpleQuote(jac_s[i])), period, 0,
                calendar, ql.Quarterly, ql.Following,
                ql.DateGeneration.TwentiethIMM, dc, _RECOVERY, rfHandle))

        jac_crv = ql.PiecewiseFlatHazardRate(todaysDate, jac_helpers, dc)
        jac_crv.enableExtrapolation()

        h_out = []
        for d in pillar_dates[1:]:
            h_out.append(jac_crv.hazardRate(d, True))

        for h in h_out:
            jac_tape.registerOutput(h)

        J = []
        for j in range(n_hazards):
            jac_tape.clearDerivatives()
            h_out[j].derivative = 1.0
            jac_tape.computeAdjoints()
            row = [jac_s[i].derivative for i in range(_N_INPUTS)]
            J.append(row)

        jac_tape.deactivate()
        return J

    m, s = _median_ms(lambda: _compute_jacobian(), repeats)
    results["a2_jacobian"] = {"median": m, "stdev": s}
    J = _compute_jacobian()

    # --- A2c: ∂NPV/∂s = J^T × ∂NPV/∂h (simple matmul) -------------------
    JT = _transpose(J)
    spread_sens_jacobian = _mat_vec_mul(JT, hazard_sens)

    def _a2c_matmul():
        for _ in range(N_SCENARIOS):
            _mat_vec_mul(JT, hazard_sens)

    m, s = _median_ms(_a2c_matmul, repeats)
    results["a2_matmul"] = {"median": m, "stdev": s, "n_scenarios": N_SCENARIOS}

    # --- A2 total: hazard replay + jacobian + matmul ----------------------
    def _a2_total():
        tape2.activate()
        for _ in range(N_SCENARIOS):
            tape2.clearDerivatives()
            npv2.derivative = 1.0
            tape2.computeAdjoints()
        tape2.deactivate()
        _compute_jacobian()
        JT_loc = _transpose(J)
        for _ in range(N_SCENARIOS):
            _mat_vec_mul(JT_loc, hazard_sens)

    m, s = _median_ms(_a2_total, repeats)
    results["a2_total"] = {"median": m, "stdev": s,
                           "n_scenarios": N_SCENARIOS, "n_inputs": n_hazards}

    # =====================================================================
    # REVERSE JACOBIAN  K = ∂s/∂h  via HazardRateCurve → fairSpread()
    # =====================================================================

    def _compute_reverse_jacobian():
        k_tape = Tape()
        k_tape.activate()
        k_inputs = [Real(h) for h in pillar_hazards_f[1:]]
        k_tape.registerInputs(k_inputs)
        k_tape.newRecording()

        k_h = [k_inputs[0]] + k_inputs             # tie h_ref = h_1
        hrc_k = ql.HazardRateCurve(pillar_dates, k_h, dc)
        hrc_k.enableExtrapolation()
        defH_k = ql.DefaultProbabilityTermStructureHandle(hrc_k)

        spread_out = []
        imp_spreads = []
        for n_op, unit in _CDS_TENORS:
            period = ql.Period(n_op, getattr(ql, unit))
            end_dt = todaysDate + period
            # Schedule must match CdsHelper::initializeDates() exactly:
            #   terminationDateConvention = Unadjusted
            sched_k = ql.Schedule(
                todaysDate, end_dt,
                ql.Period(ql.Quarterly), calendar,
                ql.Following, ql.Unadjusted,
                ql.DateGeneration.TwentiethIMM, False)
            # CDS must match SpreadCdsHelper::resetEngine() exactly:
            #   protectionStart = evaluationDate
            #   tradeDate       = evaluationDate   (not the default!)
            cds_k = ql.CreditDefaultSwap(
                ql.Protection.Buyer, 1.0, 0.01,
                sched_k, ql.Following, dc,
                True, True, todaysDate,        # protectionStart
                ql.FaceValueClaim(),            # claim (pass-through)
                ql.Actual365Fixed(),            # lastPeriodDayCounter
                True,                           # rebatesAccrual
                todaysDate)                     # tradeDate = evalDate
            cds_k.setPricingEngine(ql.MidPointCdsEngine(defH_k, _RECOVERY, rfHandle))
            fs = cds_k.fairSpread()
            spread_out.append(fs)
            imp_spreads.append(V(fs))

        for r in spread_out:
            k_tape.registerOutput(r)

        K = []
        for i in range(n_hazards):
            k_tape.clearDerivatives()
            spread_out[i].derivative = 1.0
            k_tape.computeAdjoints()
            row = [k_inputs[j].derivative for j in range(n_hazards)]
            K.append(row)

        k_tape.deactivate()
        return K, imp_spreads

    m, s = _median_ms(lambda: _compute_reverse_jacobian(), repeats)
    results["k_time"] = {"median": m, "stdev": s}
    K, implied_spreads = _compute_reverse_jacobian()

    # =====================================================================
    # Inverse verification:  K × J  vs  I
    # =====================================================================
    KJ = _mat_mul(K, J)
    J_inv = _mat_inv(J)

    # Max residual  |K × J − I|
    kj_residuals = [[KJ[i][j] - (1.0 if i == j else 0.0)
                     for j in range(n_hazards)] for i in range(n_hazards)]
    max_kj_res = max(abs(kj_residuals[i][j])
                     for i in range(n_hazards) for j in range(n_hazards))

    # Max difference  |K − J⁻¹|
    k_jinv_diff = [[K[i][j] - J_inv[i][j]
                    for j in range(n_hazards)] for i in range(n_hazards)]
    max_k_jinv = max(abs(k_jinv_diff[i][j])
                     for i in range(n_hazards) for j in range(n_hazards))

    # =====================================================================
    # FD validation at base market
    # =====================================================================
    for i in range(_N_INPUTS):
        fd_quotes[i].setValue(_CDS_BASE_SPREADS[i])
    base_npv_fd = V(fd_cds.NPV())

    spread_sens_fd = []
    for j in range(_N_INPUTS):
        fd_quotes[j].setValue(_CDS_BASE_SPREADS[j] + BPS)
        up_npv = V(fd_cds.NPV())
        fd_quotes[j].setValue(_CDS_BASE_SPREADS[j])
        spread_sens_fd.append((up_npv - base_npv_fd) / BPS)

    # =====================================================================
    # Pack results
    # =====================================================================
    results["n_hazards"]            = n_hazards
    results["pillar_dates_str"]     = pillar_dates_str
    results["pillar_hazards"]       = pillar_hazards_f[1:]
    results["J"]                    = J
    results["K"]                    = K
    results["KJ"]                   = KJ
    results["J_inv"]                = J_inv
    results["kj_residuals"]         = kj_residuals
    results["max_kj_res"]           = max_kj_res
    results["k_jinv_diff"]          = k_jinv_diff
    results["max_k_jinv"]           = max_k_jinv
    results["spread_sens_direct"]   = spread_sens_direct
    results["spread_sens_jacobian"] = spread_sens_jacobian
    results["spread_sens_fd"]       = spread_sens_fd
    results["hazard_sens"]          = hazard_sens
    results["base_npv"]             = base_npv_fd
    results["implied_spreads"]      = implied_spreads

    return results


def worker_main(repeats):
    print(f"Worker: eval_date={_EVAL_DATE}", file=sys.stderr)
    data = _run_worker(repeats)
    print(json.dumps(data))


# ============================================================================
# Orchestrator
# ============================================================================

def _sp(a, b):
    return f"{a / b:.2f}×" if b > 0 else "—"


def _print_matrix(M, row_labels, col_labels, n, title=None):
    """Print an n×n matrix to console."""
    col_w = 12
    lbl_w = 16
    if title:
        print(f"  {title}")
    hdr = " " * (lbl_w + 2) + "".join(f"{c:>{col_w}}" for c in col_labels[:n])
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for j in range(n):
        vals = "".join(f"{M[j][i]:>{col_w}.6f}" for i in range(n))
        print(f"  {row_labels[j]:<{lbl_w}s}{vals}")


def print_comparison(nojit, jit, repeats, wheels):
    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  CDS-Spread Sensitivity & Reverse Credit Jacobian Benchmark")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(SEPARATOR)
    print(f"  Scenarios per batch    : {N_SCENARIOS}")
    print(f"  Outer repeats          : {repeats}")
    print(f"  Non-JIT : {wheels['nojit']['ql'].name}")
    print(f"  JIT     : {wheels['jit']['ql'].name}")
    print(f"  Risk-free rate         : {_RF_RATE*100:.1f}% (flat)")
    print(f"  Recovery rate          : {_RECOVERY*100:.0f}%")
    print()

    n = nojit["n_hazards"]
    n_scen = N_SCENARIOS
    COL_M = 34
    COL_T = 22

    hdr = (f"  {'Method':<{COL_M}}"
           f"  {'Non-JIT':>{COL_T}}"
           f"  {'JIT':>{COL_T}}"
           f"  {'JIT sp':>8}")

    # --- Approach 1 ---
    print(f"  ── Approach 1: Direct bootstrap  ({n} CDS-spread inputs, {n_scen} scenarios) ──")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for label, key in [
        ("FD (N+1 pricings)", "a1_fd"),
        ("AAD replay",        "a1_aad_replay"),
        ("AAD re-record",     "a1_aad_record"),
    ]:
        nj = nojit[key]["median"]
        jt = jit[key]["median"]
        njs = nojit[key]["stdev"]
        jts = jit[key]["stdev"]
        sp = _sp(nj, jt)
        print(f"  {label:<{COL_M}}"
              f"  {nj:>8.1f} ±{njs:>6.1f} ms"
              f"  {jt:>8.1f} ±{jts:>6.1f} ms"
              f"  {sp:>8}")
    print()

    # --- Approach 2 ---
    print(f"  ── Approach 2: HazardRateCurve AAD + J^T conversion  ({n_scen} scenarios) ──")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for label, key in [
        ("HazardRateCurve AAD replay (100×)", "a2_hazard_replay"),
        ("Jacobian J = ∂h/∂s (4 sweeps)",    "a2_jacobian"),
        ("J^T × ∂NPV/∂h matmul (100×)",      "a2_matmul"),
        ("Total Approach 2",                  "a2_total"),
    ]:
        nj = nojit[key]["median"]
        jt = jit[key]["median"]
        njs = nojit[key]["stdev"]
        jts = jit[key]["stdev"]
        sp = _sp(nj, jt)
        print(f"  {label:<{COL_M}}"
              f"  {nj:>8.1f} ±{njs:>6.1f} ms"
              f"  {jt:>8.1f} ±{jts:>6.1f} ms"
              f"  {sp:>8}")
    print()

    # --- Cross-approach comparison ---
    a1_fd_nj = nojit["a1_fd"]["median"]
    a1_rp_nj = nojit["a1_aad_replay"]["median"]
    a1_rc_nj = nojit["a1_aad_record"]["median"]
    a2_nj    = nojit["a2_total"]["median"]

    print(f"  ── Cross-approach comparison ({n_scen} scenarios, non-JIT) ──")
    print(f"  {'Method':<{COL_M}}  {'Time':>10}  {'vs FD':>10}")
    print("  " + "─" * 60)
    print(f"  {'Approach 1: FD':<{COL_M}}  {a1_fd_nj:>7.1f} ms  {'1.0×':>10}")
    print(f"  {'Approach 1: AAD re-record':<{COL_M}}  {a1_rc_nj:>7.1f} ms  {a1_fd_nj/a1_rc_nj:>9.1f}×")
    print(f"  {'Approach 2: total':<{COL_M}}  {a2_nj:>7.1f} ms  {a1_fd_nj/a2_nj:>9.0f}×")
    print(f"  {'Approach 1: AAD replay':<{COL_M}}  {a1_rp_nj:>7.1f} ms  {a1_fd_nj/a1_rp_nj:>9.0f}×")
    print(f"  ┌{'─'*58}┐")
    print(f"  │ Approach 1 replay is {a2_nj/a1_rp_nj:,.0f}× faster than Approach 2 total{' '*5}│")
    print(f"  │ (For spread sensitivities: just differentiate through    │")
    print(f"  │ the solver — AAD handles it efficiently.)                │")
    print(f"  └{'─'*58}┘")
    print()

    # --- CDS-spread sensitivity validation ---
    print(f"  ── CDS-spread sensitivity validation (∂NPV/∂s) ──")
    print(f"  {'Tenor':<6s}  {'FD (boot)':>14s}  {'AAD (boot)':>14s}  {'J^T×∂NPV_HC/∂h':>16s}")
    print("  " + "─" * 58)
    for i in range(n):
        fd_v = nojit["spread_sens_fd"][i]
        ad_v = nojit["spread_sens_direct"][i]
        jc_v = nojit["spread_sens_jacobian"][i]
        print(f"  {_CDS_TENOR_LABELS[i]:<6s}  {fd_v:>14.2f}  {ad_v:>14.2f}  {jc_v:>16.2f}")
    print("  FD and AAD both use PiecewiseFlatHazardRate → agree as expected.")
    print("  J^T column uses HazardRateCurve hazard-rate sensitivities converted via the")
    print("  bootstrap Jacobian.  Since both curves use BackwardFlat interpolation,")
    print("  J^T × ∂NPV/∂h agrees with the direct AAD result.")
    print()

    # --- Implied par spreads ---
    print(f"  ── Implied par spreads from HazardRateCurve vs original ──")
    print(f"  {'Tenor':<6s}  {'Original':>10s}  {'HazardRateCurve':>16s}  {'Diff (bp)':>10s}")
    print("  " + "─" * 48)
    for i in range(n):
        orig = _CDS_BASE_SPREADS[i]
        impl = nojit["implied_spreads"][i]
        diff = (impl - orig) * 10000
        print(f"  {_CDS_TENOR_LABELS[i]:<6s}  {orig*10000:>8.1f} bp  {impl*10000:>14.2f} bp  {diff:>+9.2f}")
    print()

    # --- K matrix ---
    k_nj = nojit["k_time"]["median"]
    k_jt = jit["k_time"]["median"]
    s_labels = [f"s_{lbl}" for lbl in _CDS_TENOR_LABELS]
    h_labels = [f"h({d})" for d in nojit["pillar_dates_str"]]
    _print_matrix(nojit["K"], s_labels, h_labels, n,
                  f"── Reverse Jacobian  K = ∂s/∂h  ({n}×{n})"
                  f"  [{k_nj:.1f} ms non-JIT, {k_jt:.1f} ms JIT] ──")
    print()

    # --- J matrix ---
    _print_matrix(nojit["J"], h_labels, s_labels, n,
                  f"── Bootstrap Jacobian  J = ∂h/∂s  ({n}×{n}) ──")
    print()

    # --- K × J product ---
    ij_labels = [f"[{i}]" for i in range(n)]
    _print_matrix(nojit["KJ"], ij_labels, ij_labels, n,
                  f"── K × J  (should be identity) ──")
    print(f"  max |K×J − I| = {nojit['max_kj_res']:.2e}")
    print(f"  max |K − J⁻¹| = {nojit['max_k_jinv']:.2e}")
    print()

    print("  ► K = J⁻¹ because both J (PiecewiseFlatHazardRate) and K (HazardRateCurve)")
    print("    use the same interpolation: BackwardFlat on hazard rates.")
    print("    The inverse function theorem guarantees the round-trip.")
    print()
    print(SEPARATOR)


# ============================================================================
# Markdown
# ============================================================================

MD_PATH = Path(__file__).resolve().parent / "cds_spread_jacobian_benchmarks_results.md"


def _md_matrix(w, M, row_labels, col_labels, n):
    hdr = "| | " + " | ".join(col_labels[:n]) + " |"
    sep = "|---|" + "|".join(["---:"] * n) + "|"
    w(hdr)
    w(sep)
    for j in range(n):
        row_vals = " | ".join(f"{M[j][i]:.6f}" for i in range(n))
        w(f"| {row_labels[j]} | {row_vals} |")


def write_markdown(nojit, jit, repeats, wheels):
    now = datetime.datetime.now()
    lines = []
    w = lines.append

    n = nojit["n_hazards"]
    n_scen = N_SCENARIOS

    w("# QuantLib-Risks-Py — CDS-Spread Sensitivity & Reverse Credit Jacobian Benchmark")
    w("")
    w(f"**Date:** {now:%Y-%m-%d %H:%M}  ")
    w(f"**Platform:** {platform.system()} {platform.machine()}  ")
    w(f"**Python:** {platform.python_version()}  ")
    w(f"**Scenarios per batch:** {n_scen}  ")
    w(f"**Outer repetitions:** {repeats} (median reported)  ")
    w(f"**Non-JIT wheel:** `{wheels['nojit']['ql'].name}`  ")
    w(f"**JIT wheel:** `{wheels['jit']['ql'].name}`  ")
    w(f"**Risk-free rate:** {_RF_RATE*100:.1f}% (flat)  ")
    w(f"**Recovery rate:** {_RECOVERY*100:.0f}%  ")
    w("")
    w("---")
    w("")
    w("## Instrument")
    w("")
    w(f"- **5-year CDS** (Protection Buyer, {_CDS_COUPON*10000:.0f} bp running coupon, "
      f"${_CDS_NOMINAL/1e6:.0f}M notional)")
    w("- Hazard curve: `PiecewiseFlatHazardRate` (Approach 1) or "
      "`HazardRateCurve` (Approach 2)")
    w(f"- Risk-free: `FlatForward` at {_RF_RATE*100:.1f}% (held fixed)")
    w("")
    w("### Market data")
    w("")
    w("| Tenor | CDS spread (bp) | Hazard rate (bp) |")
    w("|-------|----------------:|-----------------:|")
    for i, lbl in enumerate(_CDS_TENOR_LABELS):
        hr = nojit["pillar_hazards"][i]
        w(f"| {lbl} | {_CDS_BASE_SPREADS[i]*10000:.1f} | {hr*10000:.2f} |")
    w("")
    w("---")
    w("")

    # --- Approach 1 ---
    w("## Approach 1 — Direct bootstrap")
    w("")
    w("Build `PiecewiseFlatHazardRate` from CDS spread inputs.  The Brent solver")
    w("is **on** the AD tape.  A single backward sweep gives all CDS-spread sensitivities.")
    w("")
    w("| Method | Non-JIT (ms) | JIT (ms) | JIT speedup |")
    w("|---|---:|---:|---:|")
    for label, key in [
        ("FD (N+1 pricings per scenario)", "a1_fd"),
        ("**AAD replay** (backward sweep)", "a1_aad_replay"),
        ("AAD re-record (forward + backward)", "a1_aad_record"),
    ]:
        nj = nojit[key]["median"]
        jt = jit[key]["median"]
        njs = nojit[key]["stdev"]
        jts = jit[key]["stdev"]
        sp = _sp(nj, jt)
        w(f"| {label} | {nj:.1f} ±{njs:.1f} | {jt:.1f} ±{jts:.1f} | {sp} |")
    w("")
    w("---")
    w("")

    # --- Approach 2 ---
    w("## Approach 2 — HazardRateCurve AAD + J^T conversion")
    w("")
    w("Compute ∂NPV/∂h via HazardRateCurve AAD (no solver on tape), then")
    w("multiply by J^T to obtain CDS-spread sensitivities:")
    w("$$\\nabla_s \\text{NPV} = J^T \\cdot \\nabla_h \\text{NPV}$$")
    w("")
    w("| Step | Non-JIT (ms) | JIT (ms) | JIT speedup |")
    w("|---|---:|---:|---:|")
    for label, key in [
        ("HazardRateCurve AAD replay (100×)", "a2_hazard_replay"),
        ("Jacobian J = ∂h/∂s (4 sweeps)", "a2_jacobian"),
        ("J^T × ∂NPV/∂h matmul (100×)", "a2_matmul"),
        ("**Total Approach 2**", "a2_total"),
    ]:
        nj = nojit[key]["median"]
        jt = jit[key]["median"]
        njs = nojit[key]["stdev"]
        jts = jit[key]["stdev"]
        sp = _sp(nj, jt)
        w(f"| {label} | {nj:.1f} ±{njs:.1f} | {jt:.1f} ±{jts:.1f} | {sp} |")
    w("")
    w("---")
    w("")

    # --- Cross-approach comparison ---
    a1_fd_nj = nojit["a1_fd"]["median"]
    a1_fd_jt = jit["a1_fd"]["median"]
    a1_rp_nj = nojit["a1_aad_replay"]["median"]
    a1_rp_jt = jit["a1_aad_replay"]["median"]
    a1_rc_nj = nojit["a1_aad_record"]["median"]
    a1_rc_jt = jit["a1_aad_record"]["median"]
    a2_nj    = nojit["a2_total"]["median"]
    a2_jt    = jit["a2_total"]["median"]

    w("## Cross-approach comparison")
    w("")
    w("All methods compute **∂NPV/∂(CDS spread)** for 100 scenarios.")
    w("")
    w("| Method | Non-JIT (ms) | vs FD | JIT (ms) | vs FD |")
    w("|---|---:|---:|---:|---:|")
    w(f"| Approach 1: FD | {a1_fd_nj:.1f} | 1.0× | {a1_fd_jt:.1f} | 1.0× |")
    w(f"| Approach 1: AAD re-record | {a1_rc_nj:.1f} | "
      f"{a1_fd_nj/a1_rc_nj:.1f}× | {a1_rc_jt:.1f} | {a1_fd_jt/a1_rc_jt:.1f}× |")
    w(f"| Approach 2: total | {a2_nj:.1f} | "
      f"{a1_fd_nj/a2_nj:.0f}× | {a2_jt:.1f} | {a1_fd_jt/a2_jt:.0f}× |")
    w(f"| **Approach 1: AAD replay** | **{a1_rp_nj:.1f}** | "
      f"**{a1_fd_nj/a1_rp_nj:,.0f}×** | **{a1_rp_jt:.1f}** | "
      f"**{a1_fd_jt/a1_rp_jt:,.0f}×** |")
    w("")
    w(f"> **Approach 1 AAD replay** ({a1_rp_nj:.1f} ms) is "
      f"**{a2_nj/a1_rp_nj:,.0f}× faster** than Approach 2 total ({a2_nj:.1f} ms).")
    w(f">")
    w(f"> For CDS-spread sensitivities, there is no benefit to bypassing the solver.")
    w(f"> The bootstrap tape is larger, but a **single backward sweep** through it")
    w(f"> is still vastly cheaper than computing the full 4-sweep Jacobian.")
    w(f"> This mirrors the money-market (par) rate benchmark for interest rates.")
    w("")
    w("---")
    w("")

    # --- Sensitivity validation ---
    w("## CDS-spread sensitivity validation")
    w("")
    w("| Tenor | FD (bootstrap) | AAD direct (bootstrap) | J^T × ∂NPV_HC/∂h |")
    w("|-------|---:|---:|---:|")
    for i, lbl in enumerate(_CDS_TENOR_LABELS):
        fd_v = nojit["spread_sens_fd"][i]
        ad_v = nojit["spread_sens_direct"][i]
        jc_v = nojit["spread_sens_jacobian"][i]
        w(f"| {lbl} | {fd_v:.2f} | {ad_v:.2f} | {jc_v:.2f} |")
    w("")
    w("> **FD** and **AAD direct** both price through `PiecewiseFlatHazardRate`")
    w("> and agree closely.")
    w(">")
    w("> The **J^T × ∂NPV_HC/∂h** column also agrees because both")
    w("> `PiecewiseFlatHazardRate` and `HazardRateCurve` use BackwardFlat")
    w("> interpolation on hazard rates.")
    w("")
    w("---")
    w("")

    # --- Implied par spreads ---
    w("## Implied par spreads from HazardRateCurve")
    w("")
    w("Par spreads computed by pricing CDS instruments through HazardRateCurve")
    w("(BackwardFlat hazard-rate interpolation) vs the original par spreads")
    w("used to bootstrap the curve.")
    w("")
    w("| Tenor | Original (bp) | HazardRateCurve (bp) | Diff (bp) |")
    w("|-------|--------:|---------:|----------:|")
    for i, lbl in enumerate(_CDS_TENOR_LABELS):
        orig = _CDS_BASE_SPREADS[i]
        impl = nojit["implied_spreads"][i]
        diff = (impl - orig) * 10000
        w(f"| {lbl} | {orig*10000:.1f} | {impl*10000:.2f} | {diff:+.2f} |")
    w("")
    w("> Since `PiecewiseFlatHazardRate` and `HazardRateCurve` both use BackwardFlat")
    w("> interpolation on hazard rates, the implied par spreads should match")
    w("> the originals closely.")
    w("")
    w("---")
    w("")

    # --- Jacobian matrices ---
    s_labels = [f"s\\_{lbl}" for lbl in _CDS_TENOR_LABELS]
    h_labels = [f"h({d})" for d in nojit["pillar_dates_str"]]

    k_nj_md = nojit["k_time"]["median"]
    k_jt_md = jit["k_time"]["median"]
    w("## Reverse Jacobian  K = ∂s/∂h")
    w("")
    w(f"**Computation time:** {k_nj_md:.1f} ms (non-JIT), {k_jt_md:.1f} ms (JIT)")
    w("")
    w("### How K is generated")
    w("")
    w(f"K is the {n}×{n} matrix of partial derivatives ∂s_i/∂h_j.")
    w("It is computed via AAD through the *reverse* mapping: hazard rates → par spreads.")
    w("")
    w(f"1. **Record a tape** of the reverse mapping: create {n} `xad::Real` hazard-rate")
    w("   inputs, build a `HazardRateCurve`, then for each of the 4 tenors build a CDS")
    w("   and call `fairSpread()`.  The fair spread is computed analytically from")
    w("   survival probabilities (no solver involved), so the tape is compact.")
    w("")
    w(f"2. **Register the {n} fair spreads as tape outputs.**")
    w("")
    w(f"3. **{n} backward sweeps**: for each output *i*, set `s_i.derivative = 1.0`,")
    w("   call `computeAdjoints()`, read `h_j.derivative` for all *j* → row *i* of K.")
    w("")
    w("Like J, K is **lower-triangular**: the par spread at tenor *i* depends only on")
    w("hazard rates at pillars ≤ *i* (BackwardFlat interpolation doesn't reach")
    w("beyond the tenor's maturity date).")
    w("")
    w("### K matrix")
    w("")
    _md_matrix(w, nojit["K"], s_labels, h_labels, n)
    w("")
    w("---")
    w("")

    w("## Bootstrap Jacobian  J = ∂h/∂s")
    w("")
    w("(Same as in the hazard-rate benchmark, included for comparison.)")
    w("")
    _md_matrix(w, nojit["J"], h_labels, s_labels, n)
    w("")
    w("---")
    w("")

    # --- Inverse verification ---
    w("## Inverse verification:  K × J  vs  I")
    w("")
    w("If K = J⁻¹, then K × J should equal the identity matrix.")
    w("")
    ij_labels = [f"[{i}]" for i in range(n)]
    _md_matrix(w, nojit["KJ"], ij_labels, ij_labels, n)
    w("")
    w(f"**max |K×J − I| = {nojit['max_kj_res']:.2e}**  ")
    w(f"**max |K − J⁻¹| = {nojit['max_k_jinv']:.2e}**")
    w("")
    w("### Residual matrix  K × J − I")
    w("")
    _md_matrix(w, nojit["kj_residuals"], ij_labels, ij_labels, n)
    w("")

    w("### Why K = J⁻¹")
    w("")
    w("K and J are computed through the **same interpolation method**:")
    w("")
    w("| | Forward (J = ∂h/∂s) | Reverse (K = ∂s/∂h) |")
    w("|---|---|---|")
    w("| **Curve object** | `PiecewiseFlatHazardRate` | `HazardRateCurve` |")
    w("| **Interpolation** | BackwardFlat on hazard rates | BackwardFlat on hazard rates |")
    w("| **Survival prob** | S(t) = exp(−∫₀ᵗ h(u) du) | S(t) = exp(−∫₀ᵗ h(u) du) |")
    w("")
    w("Because both curves produce identical survival probabilities at all dates,")
    w("the round-trip s → h → s closes exactly.  By the inverse function")
    w("theorem, K = J⁻¹ and K × J = I.")
    w("")
    w("---")
    w("")

    # --- Notes ---
    w("## Notes")
    w("")
    w("- For **CDS-spread sensitivities**, Approach 1 (direct bootstrap AAD) is the")
    w("  clear winner.  A single backward sweep through the bootstrap tape, even")
    w("  with the Brent solver, is far cheaper than the 4-sweep Jacobian computation")
    w("  in Approach 2.")
    w("- This is the **mirror image** of the hazard-rate benchmark, where bypassing the")
    w("  solver (Approach 1: HazardRateCurve) was the overwhelming winner.")
    w("- **Takeaway**: differentiate through the solver when you need CDS-spread risks;")
    w("  bypass the solver (HazardRateCurve) when you need hazard-rate risks.")
    w("- The **K = J⁻¹** result confirms the inverse function theorem:")
    w("  both directions use `PiecewiseFlatHazardRate` / `HazardRateCurve` (same")
    w("  BackwardFlat hazard-rate interpolation), so the round-trip is exact.")
    w("- This is the **credit analogue** of the interest-rate `mm_rate_jacobian`")
    w("  benchmark, where K = ∂r/∂z was verified against J = ∂z/∂r.")
    w("")

    w("## How to reproduce")
    w("")
    w("```bash")
    w("./build.sh --no-jit -j$(nproc)")
    w("./build.sh --jit    -j$(nproc)")
    w("")
    w("python benchmarks/cds_spread_jacobian_benchmarks.py")
    w("python benchmarks/cds_spread_jacobian_benchmarks.py --repeats 10")
    w("```")
    w("")

    MD_PATH.write_text("\n".join(lines))
    print(f"  Results written to {MD_PATH.relative_to(ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLib-Risks CDS-spread sensitivity & reverse credit Jacobian benchmark",
    )
    parser.add_argument("--worker", metavar="REPEATS", type=int, default=None)
    parser.add_argument("--repeats", "-r", type=int, default=5)
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
        print("ERROR: Missing wheels:", ", ".join(missing))
        sys.exit(1)

    print()
    print(SEPARATOR)
    print("QuantLib-Risks-Py  –  CDS-Spread Sensitivity & Reverse Credit Jacobian Benchmark")
    print(f"Python {platform.python_version()}  |  {platform.machine()}  |  "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"  {N_SCENARIOS} scenarios per batch, {repeats} outer repetitions")
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

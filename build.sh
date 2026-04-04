#!/usr/bin/env bash
# =============================================================================
# build.sh  —  Full build script for QuantLib-Risks-Py
#
# Usage:
#   bash build.sh --jit          Build with XAD-Forge JIT backends (Forge + xad-forge)
#   bash build.sh --no-jit       Build standard AAD tape only (no Forge dependency)
#
# Options:
#   --jit              Enable XAD-Forge JIT (requires lib/forge and lib/xad-forge)
#   --no-jit           Standard XAD tape-based AAD only (default)
#   -j N               Parallel build jobs (default: nproc)
#   --clean            Wipe the relevant CMake build directories before configuring
#   --skip-apt         Skip the apt-get install step
#   --skip-pip         Skip pip install of build tools (poetry, setuptools, etc.)
#   --skip-install     Skip cmake --install (already installed to prefix)
#   --skip-verify      Skip the final import sanity check
#
# Build artefacts
#   JIT build:    build/quantlib-linux-xad-jit-gcc-ninja-release/
#                 build/linux-xad-jit-gcc-ninja-release/
#                 build/xad-python-jit/
#                 build/prefix-jit/
#
#   No-JIT build: build/quantlib-linux-xad-gcc-ninja-release/
#                 build/linux-xad-gcc-ninja-release/
#                 build/xad-python/
#                 build/prefix/
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

step()  { echo; echo "${CYAN}${BOLD}==> $*${RESET}"; }
info()  { echo "    ${GREEN}$*${RESET}"; }
warn()  { echo "    ${YELLOW}WARNING: $*${RESET}"; }
die()   { echo "${RED}${BOLD}ERROR: $*${RESET}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
JIT=0
JOBS=$(nproc 2>/dev/null || echo 8)
CLEAN=0
SKIP_APT=0
SKIP_PIP=0
SKIP_INSTALL=0
SKIP_VERIFY=0

usage() {
    # Print the header comment block at the top of this file (up to the first blank line)
    awk '/^# =/{p=1} p && /^[^#]/{exit} p{sub(/^# ?/,""); print}' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --jit)         JIT=1 ;;
        --no-jit)      JIT=0 ;;
        -j)            JOBS="$2"; shift ;;
        -j*)           JOBS="${1#-j}" ;;
        --jobs=*)      JOBS="${1#*=}" ;;
        --clean)       CLEAN=1 ;;
        --skip-apt)    SKIP_APT=1 ;;
        --skip-pip)    SKIP_PIP=1 ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --skip-verify) SKIP_VERIFY=1 ;;
        -h|--help)     usage ;;
        *) die "Unknown argument: $1  (run with --help for usage)" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ $JIT -eq 1 ]]; then
    BUILD_TAG="linux-xad-jit-gcc-ninja-release"
    PREFIX="$ROOT/build/prefix-jit"
    MODE_LABEL="XAD-Forge JIT  (Forge + xad-forge)"
else
    BUILD_TAG="linux-xad-gcc-ninja-release"
    PREFIX="$ROOT/build/prefix"
    MODE_LABEL="Standard XAD tape (no JIT)"
fi

QL_BUILD="$ROOT/build/quantlib-$BUILD_TAG"
PY_BUILD="$ROOT/build/$BUILD_TAG"
XAD_PY_BUILD="$ROOT/build/xad-python$([[ $JIT -eq 1 ]] && echo '-jit' || true)"

echo
echo "${BOLD}QuantLib-Risks-Py build script${RESET}"
echo "  Mode          : ${BOLD}$MODE_LABEL${RESET}"
echo "  Parallel jobs : $JOBS"
echo "  QL cmake build: $QL_BUILD"
echo "  Py build dir  : $PY_BUILD"
echo "  xad-py build  : $XAD_PY_BUILD"
echo "  Install prefix: $PREFIX"
echo "  Repo root     : $ROOT"
echo

# ---------------------------------------------------------------------------
# Step 0 – System packages
# ---------------------------------------------------------------------------
step "0/9  System packages"
if [[ $SKIP_APT -eq 1 ]]; then
    info "Skipped (--skip-apt)"
else
    sudo apt-get update -qq
    sudo apt-get install -y \
        ninja-build g++ cmake libboost-all-dev libssl-dev libgomp1
    info "System packages OK"
fi

# ---------------------------------------------------------------------------
# Step 1 – Git submodules
# ---------------------------------------------------------------------------
step "1/9  Git submodules"
git submodule update --init --recursive
info "Submodules OK: lib/QuantLib  lib/xad  lib/QuantLib-Risks-Cpp"

# Optional JIT repos
if [[ $JIT -eq 1 ]]; then
    if [[ ! -d "$ROOT/lib/forge/.git" ]]; then
        step "     Cloning lib/forge (not a submodule)"
        git clone https://github.com/da-roth/forge.git "$ROOT/lib/forge"
    else
        info "lib/forge already present"
    fi
    if [[ ! -d "$ROOT/lib/xad-forge/.git" ]]; then
        step "     Cloning lib/xad-forge (not a submodule)"
        git clone https://github.com/da-roth/xad-forge.git "$ROOT/lib/xad-forge"
    else
        info "lib/xad-forge already present"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2 – Python build tools (pip)
# ---------------------------------------------------------------------------
step "2/9  Python build tools"
if [[ $SKIP_PIP -eq 1 ]]; then
    info "Skipped (--skip-pip)"
else
    pip install --quiet poetry poetry-core poetry-dynamic-versioning setuptools wheel build
    info "pip packages OK"
fi

# ---------------------------------------------------------------------------
# Step 3 – Configure QuantLib
# ---------------------------------------------------------------------------
step "3/9  Configure QuantLib"

if [[ $CLEAN -eq 1 && -d "$QL_BUILD" ]]; then
    warn "--clean: removing $QL_BUILD"
    rm -rf "$QL_BUILD"
fi

if [[ $JIT -eq 1 ]]; then
    QL_EXT_DIRS="$ROOT/lib/forge/api/c;$ROOT/lib/xad;$ROOT/lib/xad-forge;$ROOT/lib/QuantLib-Risks-Cpp"
    info "QL_EXTERNAL_SUBDIRECTORIES (JIT): forge/api/c  xad  xad-forge  QuantLib-Risks-Cpp"
else
    QL_EXT_DIRS="$ROOT/lib/xad;$ROOT/lib/QuantLib-Risks-Cpp"
    info "QL_EXTERNAL_SUBDIRECTORIES (no JIT): xad  QuantLib-Risks-Cpp"
fi

cmake -B "$QL_BUILD" \
      -S "$ROOT/lib/QuantLib" \
      -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  "-DQL_EXTERNAL_SUBDIRECTORIES=$QL_EXT_DIRS" \
  -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Risks" \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_NO_THREADLOCAL=ON \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"

info "QuantLib configured"

# ---------------------------------------------------------------------------
# Step 4 – Build QuantLib (first pass)
# ---------------------------------------------------------------------------
step "4/9  Build QuantLib"
cmake --build "$QL_BUILD" --parallel "$JOBS"
info "QuantLib built"

# ---------------------------------------------------------------------------
# Step 5 – Apply QuantLib v1.33 cherry-pick patch
# ---------------------------------------------------------------------------
step "5/9  QuantLib patch (GeometricBrownianMotionProcess real-type fix)"
(
    cd "$ROOT/lib/QuantLib"
    git config user.email "build@localhost"
    git config user.name  "build"
    if git log --oneline | grep -q "Uses Real in GemoetricBrownianMotionProcess"; then
        info "Patch already applied – skipping cherry-pick"
    else
        git fetch --all
        git cherry-pick 6bb9c1f18ff6d4c47f06a66136fb83411207e67c
        info "Cherry-pick applied"
        # Rebuild now that the patched file is included
        cmake --build "$QL_BUILD" --parallel "$JOBS"
        info "QuantLib rebuilt after patch"
    fi
)

# ---------------------------------------------------------------------------
# Step 6 – Install QuantLib
# ---------------------------------------------------------------------------
step "6/9  Install QuantLib to $PREFIX"
if [[ $SKIP_INSTALL -eq 1 && -f "$PREFIX/lib/libQuantLib.a" ]]; then
    info "Skipped (--skip-install + prefix already populated)"
else
    cmake --install "$QL_BUILD"
    info "QuantLib installed: $PREFIX/lib/libQuantLib.a"
fi

# ---------------------------------------------------------------------------
# Step 7 – Configure and build the Python SWIG bindings (.so)
# ---------------------------------------------------------------------------
step "7/9  Configure Python SWIG bindings"

if [[ $CLEAN -eq 1 && -d "$PY_BUILD" ]]; then
    warn "--clean: removing $PY_BUILD"
    rm -rf "$PY_BUILD"
fi

mkdir -p "$PY_BUILD"
cmake -B "$PY_BUILD" \
      -S "$ROOT" \
      -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$PREFIX" \
  "-DCMAKE_INSTALL_PREFIX=$PREFIX" \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fopenmp" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fopenmp"

step "     Build Python SWIG bindings"
cmake --build "$PY_BUILD" --parallel "$JOBS"

SO_FILE=$(find "$PY_BUILD/Python/QuantLib_Risks" -name "_QuantLib_Risks*.so" | head -1)
if [[ -z "$SO_FILE" ]]; then
    die "Expected _QuantLib_Risks*.so not found under $PY_BUILD/Python/QuantLib_Risks/"
fi
info "Built: $(basename "$SO_FILE")"

# ---------------------------------------------------------------------------
# Step 8 – Build xad Python C extension and wheel
# ---------------------------------------------------------------------------
step "8/9  Build xad Python bindings"

if [[ $CLEAN -eq 1 && -d "$XAD_PY_BUILD" ]]; then
    warn "--clean: removing $XAD_PY_BUILD"
    rm -rf "$XAD_PY_BUILD"
fi

cmake -B "$XAD_PY_BUILD" \
      -S "$ROOT/lib/xad" \
      -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXAD_ENABLE_PYTHON=ON \
  -DXAD_NO_THREADLOCAL=ON \
  -DXAD_BUILD_TESTS=OFF \
  -DXAD_BUILD_DOCS=OFF

# Build only the C extension .so — skip cmake's internal wheel-build target
# (which runs `poetry install` before the .so exists and would fail)
cmake --build "$XAD_PY_BUILD" --parallel "$JOBS" --target _xad_autodiff

XAD_SO=$(find "$XAD_PY_BUILD" -name "_xad_autodiff*.so" | head -1)
if [[ -z "$XAD_SO" ]]; then
    die "Expected _xad_autodiff*.so not found under $XAD_PY_BUILD"
fi
info "Built: $(basename "$XAD_SO")"

# Build and install xad-autodiff wheel (pip-driven, no cmake involvement)
step "     Build and install xad-autodiff wheel"
(
    XAD_WHEEL_DIR="$XAD_PY_BUILD/dist"
    mkdir -p "$XAD_WHEEL_DIR"
    pip wheel "$ROOT/lib/xad/bindings/python" --no-build-isolation --quiet \
        --wheel-dir "$XAD_WHEEL_DIR"
    XAD_WHEEL=$(ls "$XAD_WHEEL_DIR"/xad_autodiff-*.whl | head -1)
    pip install --quiet --force-reinstall "$XAD_WHEEL"
    info "Installed: $(basename "$XAD_WHEEL")"
)

# Install xad compatibility shim  (xad_autodiff installs as 'xad_autodiff';
# QuantLib-Risks examples import 'from xad.adj_1st import Tape')
step "     Install xad compatibility shim"
python3 - << 'PYEOF'
import pathlib, subprocess, sys

base = pathlib.Path("/tmp/xad-shim")
for d in ["xad/adj_1st", "xad/fwd_1st", "xad/math", "xad/exceptions"]:
    (base / d).mkdir(parents=True, exist_ok=True)

files = {
    "xad/__init__.py": (
        "from xad_autodiff import *\n"
        "from xad_autodiff import _xad_autodiff, adj_1st, fwd_1st, value, derivative\n"
    ),
    "xad/adj_1st/__init__.py": (
        "from xad_autodiff.adj_1st import *\n"
        "from xad_autodiff.adj_1st import Real, Tape\n"
    ),
    "xad/fwd_1st/__init__.py":    "from xad_autodiff.fwd_1st import *\n",
    "xad/math/__init__.py":       "from xad_autodiff.math import *\n",
    "xad/exceptions/__init__.py": "from xad_autodiff.exceptions import *\n",
    "pyproject.toml": (
        '[build-system]\n'
        'requires = ["setuptools>=42"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\n'
        'name = "xad"\n'
        'version = "1.5.2"\n'
        'description = "Compatibility shim: maps import xad to xad_autodiff"\n'
        'requires-python = ">=3.8"\n'
        'dependencies = ["xad-autodiff>=1.5.1"]\n'
    ),
}
for rel, content in files.items():
    (base / rel).write_text(content)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
     "--no-deps", str(base), "--no-build-isolation"]
)
print("    xad shim installed OK")
PYEOF

# ---------------------------------------------------------------------------
# Step 9 – Build and install QuantLib_Risks pip wheel
# ---------------------------------------------------------------------------
step "9/9  Build and install QuantLib_Risks wheel"
(
    cd "$PY_BUILD/Python"
    mkdir -p dist
    # Use `python -m build` (PEP 517) instead of `pip wheel` so that pip's
    # online dependency resolver never runs — all deps are already installed.
    python3 -m build --wheel --no-isolation --outdir dist
    QL_WHEEL=$(ls dist/quantlib_risks-*.whl | head -1)
    pip install --quiet --force-reinstall --no-deps "$QL_WHEEL"
    info "Installed: $(basename "$QL_WHEEL")"
    echo "    Wheel path: $PY_BUILD/Python/dist/$(basename "$QL_WHEEL")"
)

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
if [[ $SKIP_VERIFY -eq 1 ]]; then
    step "Verification skipped (--skip-verify)"
else
    step "Verification"
    cd /tmp
    python3 - << 'PYEOF'
import QuantLib_Risks as ql
from xad.adj_1st import Tape, Real

tape = Tape()
tape.activate()
x = Real(1.5)                   # create Real while tape is active, before recording
tape.registerInput(x)
tape.newRecording()
y = x * x
tape.registerOutput(y)
y.derivative = 1.0
tape.computeAdjoints()
y_val    = float(y.value)       # capture before deactivate
x_deriv  = float(x.derivative)
tape.deactivate()
assert abs(y_val   - 2.25) < 1e-12, f"Bad NPV: {y_val}"
assert abs(x_deriv - 3.0)  < 1e-12, f"Bad adjoint: {x_deriv}"
print(f"    y = {y_val}   dy/dx = {x_deriv}  <- correct")
print("    QuantLib_Risks + XAD import and AD verification: PASSED")
PYEOF
    cd "$ROOT"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "${BOLD}${GREEN}Build complete.${RESET}"
echo
echo "  Mode     : $MODE_LABEL"
echo "  QL build : $QL_BUILD"
echo "  Py build : $PY_BUILD"
echo "  xad-py   : $XAD_PY_BUILD"
echo "  Prefix   : $PREFIX"
echo
echo "  Run the examples:"
echo "    cd $ROOT/Python/examples"
echo "    python3 swap-adjoint.py"
echo
echo "  Run the benchmark:"
echo "    python3 $ROOT/benchmarks/run_benchmarks.py"
echo
echo "  Run the test suite:"
echo "    bash $ROOT/Python/run_tests.sh"
echo

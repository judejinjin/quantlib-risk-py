# Building QuantLib-Risks-Py with XAD-Forge on Linux

This guide covers how to build the project from scratch on Linux (including WSL2),
incorporating all fixes required to get a clean build with XAD-Forge JIT support.

---

## Repository setup

### 1. Clone the main repository

```bash
git clone https://github.com/auto-differentiation/QuantLib-Risks-Py.git
cd QuantLib-Risks-Py
```

### 2. Initialise git submodules

Three dependencies are managed as git submodules (`lib/QuantLib`, `lib/xad`,
`lib/QuantLib-Risks-Cpp`):

```bash
git submodule update --init --recursive
```

### 3. Clone Forge and XAD-Forge

These two repositories are **not** submodules and must be cloned manually into `lib/`:

```bash
git clone https://github.com/da-roth/forge.git     lib/forge
git clone https://github.com/da-roth/xad-forge.git lib/xad-forge
```

After this step the `lib/` directory should look like:

```
lib/
├── forge/               # https://github.com/da-roth/forge.git
├── QuantLib/            # submodule — https://github.com/lballabio/QuantLib.git
├── QuantLib-Risks-Cpp/  # submodule — https://github.com/auto-differentiation/QuantLib-Risks-Cpp.git
├── xad/                 # submodule — https://github.com/auto-differentiation/xad.git
└── xad-forge/           # https://github.com/da-roth/xad-forge.git
```

---

## Prerequisites

### Install required system packages

```bash
sudo apt-get update
sudo apt-get install -y \
    ninja-build \
    g++ \
    cmake \
    libboost-all-dev \
    libssl-dev
```

> **Note:** `ninja-build` is required. CMake will fail with
> `CMAKE_MAKE_PROGRAM is not set` and `CMAKE_CXX_COMPILER not set` if it is missing.

---

## Step 1: Configure QuantLib with XAD-Forge

Run from the **repo root** (`QuantLib-Risks-Py/`), not from inside `lib/QuantLib/`.
All environment variables must be set before invoking cmake.

```bash
ROOT=$(pwd)   # must be run from QuantLib-Risks-Py/

cmake -B lib/QuantLib/build/linux-xad-gcc-ninja-release \
      -S lib/QuantLib \
      -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ROOT/build/prefix" \
  -DQL_EXTERNAL_SUBDIRECTORIES="$ROOT/lib/forge/api/c;$ROOT/lib/xad;$ROOT/lib/xad-forge;$ROOT/lib/QuantLib-Risks-Cpp" \
  -DQL_EXTRA_LINK_LIBRARIES="QuantLib-Risks" \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_NO_THREADLOCAL=ON \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
```

### Key cmake variables explained

| Variable | Value | Why |
|---|---|---|
| `QL_EXTERNAL_SUBDIRECTORIES` | `forge/api/c;xad;xad-forge;QuantLib-Risks-Cpp` | Adds all four dependency subdirectories in the correct order |
| `QL_EXTRA_LINK_LIBRARIES` | `QuantLib-Risks` | **Must be `QuantLib-Risks`, not `QuantLibAAD`**. This INTERFACE target propagates `-DQL_INCLUDE_FIRST=ql/qlrisks.hpp`, which force-includes `qlrisks.hpp` and redefines `Real` as `xad::AReal<double>`. Without it, `Real::tape_type` is invalid (see §Errors below). |
| `QL_NULL_AS_FUNCTIONS` | `ON` | Required for XAD compatibility |
| `XAD_NO_THREADLOCAL` | `ON` | Required on Linux/WSL |
| `CMAKE_EXE_LINKER_FLAGS` | `-fopenmp` | Links `libgomp` into every example executable, resolving undefined references to `GOMP_parallel`, `omp_get_thread_num`, `omp_get_num_threads` from `libQuantLib.a` (see §Errors below). Also causes cmake to use the static `libQuantLib.a` instead of the shared `.so`, avoiding deferred-resolution issues. |
| `QL_BUILD_TEST_SUITE=OFF` etc. | `OFF` | Skip building test suite, examples (QuantLib built-in ones), and benchmarks to save time |

---

## Step 2: Build

```bash
cmake --build lib/QuantLib/build/linux-xad-gcc-ninja-release --parallel 8
```

Or from inside the build directory:

```bash
cd lib/QuantLib/build/linux-xad-gcc-ninja-release
cmake --build /mnt/c/cplusplus/QuantLib-Risks-Py/lib/QuantLib/build/linux-xad-gcc-ninja-release --parallel 8
```

> **Note:** `cmake --build . --parallel 8` (with a `.`) does **not** work on this version of
> CMake (3.28). Use the absolute path to the build directory.

---

## Step 3: Apply required QuantLib patch

QuantLib v1.33 has a bug where `GeometricBrownianMotionProcess` uses `double`
instead of `Real` in its constructor, which breaks compilation against the XAD
`Real` type. A fix was cherry-picked from upstream and must be applied before
installing:

```bash
cd lib/QuantLib
git config user.email "build@localhost"
git config user.name "build"
git log --oneline | grep -q "Uses Real in GemoetricBrownianMotionProcess" || \
  (git fetch --all && git cherry-pick 6bb9c1f18ff6d4c47f06a66136fb83411207e67c)
cd ../..
```

After the cherry-pick, rebuild QuantLib to pick up the change:

```bash
cmake --build lib/QuantLib/build/linux-xad-gcc-ninja-release --parallel 8
```

---

## Step 4: Install QuantLib

```bash
cmake --install lib/QuantLib/build/linux-xad-gcc-ninja-release
```

This installs headers, the static library (`libQuantLib.a`), cmake config files,
and the Adjoint example binaries into `build/prefix/`.

---

## Step 5: Configure the Python bindings

Run from the **repo root** (`QuantLib-Risks-Py/`). The `-DCMAKE_PREFIX_PATH` must
point at the install prefix so `find_package(QuantLib-Risks)` can find the cmake
config files installed in the previous step. `-fopenmp` must be added to all three
linker flag variables so the Python `.so` extension resolves OpenMP symbols from
`libQuantLib.a`:

```bash
ROOT=$(pwd)
mkdir -p build/linux-xad-gcc-ninja-release
cmake -B build/linux-xad-gcc-ninja-release \
      -S . \
      -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROOT/build/prefix" \
  -DCMAKE_INSTALL_PREFIX="$ROOT/build/prefix" \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fopenmp" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fopenmp"
```

---

## Step 6: Build the Python bindings

```bash
cmake --build build/linux-xad-gcc-ninja-release --parallel 8
```

This produces `_QuantLib_Risks.cpython-*-x86_64-linux-gnu.so` inside
`build/linux-xad-gcc-ninja-release/Python/QuantLib_Risks/`.

---

## Step 7: Verify the Python module

```bash
cd build/linux-xad-gcc-ninja-release/Python/QuantLib_Risks
LD_LIBRARY_PATH=/path/to/QuantLib-Risks-Py/build/prefix/lib:$LD_LIBRARY_PATH \
  python3 -c "import _QuantLib_Risks; print('Import OK')"
```

---

## Step 8: Build the xad Python bindings

`QuantLib_Risks` depends on the `xad` Python package, which must be built from the `lib/xad`
submodule.  It is a pybind11 C extension and requires its own CMake configure + build.

### Install build tools (once)

```bash
pip install poetry poetry-core poetry-dynamic-versioning setuptools wheel build
```

### Configure and build the xad C extension

```bash
ROOT=$(pwd)   # must be run from QuantLib-Risks-Py/

cmake -B build/xad-python \
      -S lib/xad \
      -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DXAD_ENABLE_PYTHON=ON \
  -DXAD_NO_THREADLOCAL=ON \
  -DXAD_BUILD_TESTS=OFF \
  -DXAD_BUILD_DOCS=OFF

cmake --build build/xad-python --parallel 8
```

This produces `build/xad-python/bindings/python/src/_xad_autodiff.cpython-*-x86_64-linux-gnu.so`
and writes `lib/xad/bindings/python/prebuilt_file.txt` (used by the wheel build script).

### Build and install the xad-autodiff wheel

```bash
cd lib/xad/bindings/python
pip wheel . --no-build-isolation --wheel-dir dist
pip install dist/xad_autodiff-*.whl
cd -
```

### Install the xad compatibility shim

The installed package is called `xad_autodiff` (module `xad_autodiff`), but
`QuantLib_Risks` and its examples import `from xad.adj_1st import Tape`.  A thin shim
package named `xad` is required to bridge this.  Build and install it with:

```bash
python3 - << 'EOF'
import pathlib, subprocess, sys

base = pathlib.Path("/tmp/xad-shim")
for d in ["xad/adj_1st", "xad/fwd_1st", "xad/math", "xad/exceptions"]:
    (base / d).mkdir(parents=True, exist_ok=True)

files = {
    "xad/__init__.py":            "from xad_autodiff import *\nfrom xad_autodiff import _xad_autodiff, adj_1st, fwd_1st, value, derivative\n",
    "xad/adj_1st/__init__.py":    "from xad_autodiff.adj_1st import *\nfrom xad_autodiff.adj_1st import Real, Tape\n",
    "xad/fwd_1st/__init__.py":    "from xad_autodiff.fwd_1st import *\n",
    "xad/math/__init__.py":       "from xad_autodiff.math import *\n",
    "xad/exceptions/__init__.py": "from xad_autodiff.exceptions import *\n",
    "pyproject.toml": (
        '[build-system]\nrequires = ["setuptools>=42"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "xad"\nversion = "1.5.2"\n'
        'description = "Compatibility shim: maps import xad to xad_autodiff"\n'
        'requires-python = ">=3.8"\ndependencies = ["xad-autodiff>=1.5.1"]\n'
    ),
}
for path, content in files.items():
    (base / path).write_text(content)

subprocess.check_call([sys.executable, "-m", "pip", "install", str(base), "--no-build-isolation"])
print("xad shim installed OK")
EOF
```

---

## Step 9: Build and install the QuantLib_Risks pip wheel

CMake generates a `pyproject.toml` (with the version substituted) and a `build_extensions.py`
inside the Python build output directory.  The custom build script copies the pre-built `.so`
into the wheel instead of recompiling it, so `--no-isolation` is required.

### Build the wheel

```bash
cd build/linux-xad-gcc-ninja-release/Python
pip wheel . --no-build-isolation --wheel-dir dist
```

This produces a platform wheel such as:

```
dist/quantlib_risks-1.33.3-cp313-cp313-manylinux_2_39_x86_64.whl
```

### Install the wheel

```bash
pip install dist/quantlib_risks-1.33.3-*.whl
```

### Verify

```bash
cd /tmp
python3 -c "
import QuantLib_Risks as ql
from xad.adj_1st import Tape
tape = Tape()
tape.activate()
x = ql.Real(1.5)
tape.registerInput(x)
tape.newRecording()
y = x * x
tape.registerOutput(y)
y.derivative = 1.0
tape.computeAdjoints()
print('y =', y.value, '  dy/dx =', x.derivative)   # expect: y=2.25  dy/dx=3.0
"
```

### Runtime requirement on the target machine

The wheel statically links `libQuantLib.a` but still requires the OpenMP runtime at load time.
Install it if not already present:

```bash
sudo apt-get install -y libgomp1
```

---

## Errors and fixes reference

### `CMAKE_MAKE_PROGRAM is not set` / `CMAKE_CXX_COMPILER not set`

**Cause:** `ninja` is not installed.

**Fix:**
```bash
sudo apt-get install -y ninja-build
```

The `CMAKE_CXX_COMPILER` error is a cascade failure from Ninja being missing — it is not
a separate problem.

---

### `error: expected ';' before '::' token` in `using tape_type = Real::tape_type`

**Cause:** `QL_EXTRA_LINK_LIBRARIES` was set to `QuantLibAAD` (which does not exist as a
CMake target). Without the `QuantLib-Risks` INTERFACE target, the define
`-DQL_INCLUDE_FIRST=ql/qlrisks.hpp` is never propagated to consumer translation units.
As a result, `Real` stays as `double` instead of `xad::AReal<double>`, and
`Real::tape_type` is invalid.

**Fix:** Use `-DQL_EXTRA_LINK_LIBRARIES="QuantLib-Risks"`.

The propagation chain:
1. `QuantLib-Risks` INTERFACE target → `-DQL_INCLUDE_FIRST=ql/qlrisks.hpp`
2. `ql/qldefines.hpp` line 56 → `#include INCLUDE_FILE(QL_INCLUDE_FIRST)` (force-includes `qlrisks.hpp`)
3. `qlrisks.hpp` → `#define QL_REAL xad::AReal<double>`
4. `ql/types.hpp` → `typedef QL_REAL Real;` → `Real::tape_type` is valid

---

### Undefined references to `GOMP_parallel` / `omp_get_thread_num` when linking examples

**Cause:** `libQuantLib.a` is compiled with `-fopenmp`. The Adjoint example executables
were not linked with `-fopenmp`/`-lgomp`, leaving OpenMP symbols unresolved at link time.

**Fix:** Pass `-DCMAKE_EXE_LINKER_FLAGS="-fopenmp"` when configuring the QuantLib build
(Step 1). This adds `-fopenmp` to every executable link command and additionally causes
cmake to use the static `libQuantLib.a` instead of the shared `.so`.

---

### `ImportError: undefined symbol: GOMP_parallel` when importing the Python module

**Cause:** The Python `.so` extension links against `libQuantLib.a` which was compiled with
OpenMP. The shared module linker flags did not include `-fopenmp`, leaving the OpenMP
runtime symbols unresolved at import time.

**Fix:** Pass all three OpenMP linker flag variables at configure time for the Python bindings step:
```
-DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
-DCMAKE_SHARED_LINKER_FLAGS="-fopenmp"
-DCMAKE_MODULE_LINKER_FLAGS="-fopenmp"
```

---

### `no matching function for call to 'GeometricBrownianMotionProcess(Real&, Real&, Real&)'`

**Cause:** QuantLib v1.33's `geometricbrownianprocess.hpp` declares the constructor with
`double` parameters instead of `Real`. When `Real = xad::AReal<double>`, the SWIG-generated
wrapper cannot pass `Real` arguments to a `double` constructor.

**Fix:** Cherry-pick the upstream fix before installing:
```bash
cd lib/QuantLib
git fetch --all
git cherry-pick 6bb9c1f18ff6d4c47f06a66136fb83411207e67c
```
Then rebuild and reinstall QuantLib before building the Python bindings.

> **Note:** This cherry-pick is only required for QuantLib v1.33. It is expected
> to be merged in v1.34 and above.

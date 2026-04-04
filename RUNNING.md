# Running QuantLib-Risks-Py Examples with XAD-Forge JIT

This guide assumes the project has already been built following [BUILDING.md](BUILDING.md).  
The expected artefacts are:

| Path | What it is |
|---|---|
| `build/linux-xad-gcc-ninja-release/Python/QuantLib_Risks/` | Python package directory (contains `_QuantLib_Risks.cpython-*.so`) |
| `build/prefix/lib/` | Installed static library (`libQuantLib.a`) and cmake config files |
| `build/prefix/include/` | Installed QuantLib and XAD headers |

---

## Background: tape-based AAD vs. XAD-Forge JIT

QuantLib-Risks-Py uses [XAD](https://github.com/auto-differentiation/xad) for automatic adjoint differentiation (AAD).
By default XAD records a computation graph on a **tape** and replays it in reverse to compute gradients.
This is optimal when each pricing computation is performed once or a handful of times.

[XAD-Forge](https://github.com/da-roth/xad-forge) integrates [Forge](https://github.com/da-roth/forge)
as a JIT backend: record the computation graph **once**, compile it to native x86-64 machine code, then
**re-evaluate as many times as needed** with different inputs.  The upfront compilation cost is amortised
across all subsequent evaluations, making this approach significantly faster for:

- Monte Carlo simulations (many paths, same graph structure)
- Risk scenario grids (stress-testing across hundreds of market moves)
- XVA / CVA batch pricing
- Model calibration (repeated gradient evaluations during optimisation)

The crossover point is typically **5–20 evaluations** depending on graph complexity; beyond that the
JIT approach outperforms tape replay by a wide margin.

Forge itself follows a **record-once → compile-once → evaluate-many** paradigm and provides two backends
bundled via xad-forge:

| Backend | Description |
|---|---|
| `ScalarBackend` | Compiles to scalar x86-64 native code (drop-in tape replacement) |
| `AVXBackend` | Compiles to AVX2 SIMD, evaluates **4 inputs in parallel** (batch pricing) |

---

## Step 1: Set up your shell environment

All commands below should be run from the **repo root** (`QuantLib-Risks-Py/`).

```bash
ROOT=$(pwd)

# Put the built Python package on the Python path
export PYTHONPATH="$ROOT/build/linux-xad-gcc-ninja-release/Python/QuantLib_Risks:$PYTHONPATH"

# Expose any shared libraries installed under build/prefix (needed if .so files are there)
export LD_LIBRARY_PATH="$ROOT/build/prefix/lib:$LD_LIBRARY_PATH"
```

> **WSL2 note:** you may also need to ensure that `libgomp.so.1` is visible.  It ships with GCC:
> ```bash
> sudo apt-get install -y libgomp1
> ```

Verify the module imports cleanly:

```bash
python3 -c "import QuantLib_Risks as ql; print('QL version:', ql.QuantLib_version())"
```

Expected output (version may differ):

```
QL version: 1.33
```

---

## Step 2: Run the test suite

The test suite validates the Python bindings, the AAD tape integration, and all supported instrument
types.  Run it with:

```bash
cd Python
python test/QuantLibTestSuite.py
```

Or use the convenience script (which also runs the three canonical examples):

```bash
bash Python/run_tests.sh
```

The script runs `QuantLibTestSuite.py` followed by the `swap-adjoint.py`, `swap.py`, and
`multicurve-bootstrapping.py` examples.

---

## Step 3: Run the AAD risk examples

The Python examples live in `Python/examples/`.  They are written in
[Jupytext percent format](https://jupytext.readthedocs.io/en/latest/) so they can be executed
directly as plain Python scripts **or** opened as Jupyter notebooks.

### Prerequisite: change directory

```bash
cd Python/examples
```

### 3a. Interest-rate swap with full risk vector (swap-adjoint.py)

This is the primary example demonstrating AAD risk computation.  It prices a vanilla IRS and
derives the **complete risk vector** (DV01 per market input) in a single backward pass using
XAD's reverse-mode tape.

```bash
python swap-adjoint.py
```

What it does:

1. Activates an XAD tape: `tape = Tape(); tape.activate()`
2. Wraps market quotes as `ql.Real` (i.e. `xad::AReal<double>`) and registers them as tape inputs
3. Bootstraps a yield curve and prices the swap — the entire computation is recorded
4. Calls `tape.computeAdjoints()` in a single reverse sweep
5. Prints `tape.derivative(q)` for every input quote — one call, all sensitivities

Expected output (abridged):

```
Swap net present value:  -14.262...
Derivatives w.r.t. market quotes:
  3m deposit  :  ...
  3x6 FRA     :  ...
  2y swap     :  ...
  ...
```

### 3b. European option (european-option.py)

```bash
python european-option.py
```

Prices a European vanilla option using the Black-Scholes-Merton model and several numerical
engines (analytic, finite difference, Monte Carlo).  No AAD tape is activated here — it
demonstrates plain pricing using the `QuantLib_Risks` module as a drop-in for standard QuantLib.

### 3c. Multi-curve bootstrapping (multicurve-bootstrapping.py)

```bash
python multicurve-bootstrapping.py
```

Bootstraps OIS and LIBOR curves simultaneously and prices cross-currency instruments.

### 3d. Bermudan swaption (bermudan-swaption.py)

```bash
python bermudan-swaption.py
```

Prices a Bermudan swaption using the G1++ short-rate model (tree and Monte Carlo engines).

### 3e. Other examples

All scripts in `Python/examples/` can be run the same way:

```bash
for f in *.py; do echo "=== $f ==="; python "$f"; done
```

---

## Step 4: Understanding the XAD-Forge JIT speedup

The JIT speedup operates at the **C++ layer** (inside `libQuantLib.a`).  When the library is
built with `xad-forge` as a subdirectory via `QL_EXTERNAL_SUBDIRECTORIES`, QuantLib-Risks-Cpp
can select the `ScalarBackend` or `AVXBackend` instead of XAD's default tape-interpreter for
repeated-evaluation workloads.

The workflow is:

```
  Python (ql.Real inputs, Tape)
        │
        ▼
  XAD records computation graph
        │
        ▼  [first evaluation]
  xad-forge: compile graph → native x86-64 (ForgedKernel)
        │
        ▼  [every subsequent evaluation]
  ForgedKernel.execute(new inputs)   ← no tape overhead, pure native code
        │
        ▼
  Gradients available via kernel.getGradient(input_node)
```

### Scalar backend (single input, fast replay)

Replaces the tape interpreter with a compiled native function.  Useful whenever the same
pricing computation is repeated (e.g. bump-and-reprice loop replaced by a single AAD Jacobian).

### AVX2 backend (4-wide SIMD, batch pricing)

The `AVXBackend` evaluates **four independent sets of inputs simultaneously** using AVX2 SIMD
instructions.  This is ideal when pricing a portfolio of similar instruments or running a
scenario grid:

```
inputs[4]  →  [AVX2 kernel]  →  outputs[4] + gradients[4]
                (1 pass)
```

Compared to running four separate tape-based evaluations, the AVX2 backend delivers close to
4× throughput on compatible hardware (any Intel/AMD CPU manufactured after ~2013).

### Build-time opt-in

The JIT backends are compiled into `libQuantLib.a` automatically when the CMake configure step
(Step 1 in BUILDING.md) includes `xad-forge` and `forge/api/c` in
`QL_EXTERNAL_SUBDIRECTORIES`:

```
-DQL_EXTERNAL_SUBDIRECTORIES="…/forge/api/c;…/xad;…/xad-forge;…/QuantLib-Risks-Cpp"
```

No additional Python-side configuration is needed: the backend selection happens in C++ before
the result surfaces to Python.

---

## Step 5: Run all examples and tests in one shot

```bash
ROOT=$(pwd)
export PYTHONPATH="$ROOT/build/linux-xad-gcc-ninja-release/Python/QuantLib_Risks:$PYTHONPATH"
export LD_LIBRARY_PATH="$ROOT/build/prefix/lib:$LD_LIBRARY_PATH"
bash Python/run_tests.sh
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'QuantLib_Risks'`

`PYTHONPATH` does not include the build output directory.  Set it as shown in Step 1.

### `ImportError: undefined symbol: GOMP_parallel`

The Python `.so` was not linked with `-fopenmp`.  Reconfigure the Python bindings build (Step 5
in BUILDING.md) with all three linker flag variables:

```bash
-DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
-DCMAKE_SHARED_LINKER_FLAGS="-fopenmp"
-DCMAKE_MODULE_LINKER_FLAGS="-fopenmp"
```

Then rebuild:

```bash
cmake --build build/linux-xad-gcc-ninja-release --parallel 8
```

### `ImportError: cannot open shared object file: libQuantLib.so`

The build links against the **static** `libQuantLib.a` when `-fopenmp` is passed at configure
time (CMake prefers static over shared when OpenMP is involved).  If the shared library is
somehow picked up and cannot be found, set:

```bash
export LD_LIBRARY_PATH="$ROOT/build/prefix/lib:$LD_LIBRARY_PATH"
```

### `ModuleNotFoundError: No module named 'xad'`

The `xad` Python package (separate from the C++ XAD library used internally) must be installed:

```bash
pip install xad
```

The `swap-adjoint.py` example imports `from xad.adj_1st import Tape` — this requires the
`xad` PyPI package.

---

## Quick reference

| Task | Command |
|---|---|
| Verify import | `python3 -c "import QuantLib_Risks as ql; print(ql.QuantLib_version())"` |
| Run test suite | `python Python/test/QuantLibTestSuite.py` |
| Run swap AAD example | `cd Python/examples && python swap-adjoint.py` |
| Run all canonical examples + tests | `bash Python/run_tests.sh` |
| Run every example | `cd Python/examples && for f in *.py; do python "$f"; done` |

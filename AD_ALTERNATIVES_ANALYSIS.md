# Open-Source AD Alternatives to XAD + Forge

## The Fundamental Divide

There are two architectural approaches to AD:

| Approach | How it works | Branching behavior |
|---|---|---|
| **Operator overloading** (XAD, CppAD, CoDiPack, ADOL-C) | Replace `double` with AD type; tape records operations at runtime | Branches are invisible to tape — same problem as XAD |
| **Compiler transformation** (Enzyme, Tapenade) | Transform the actual compiled code (IR or source) to generate adjoint code | Branches are differentiated natively — **no branching problem** |

---

## 1. Enzyme (LLVM-based) — the strongest alternative

**What it is**: An LLVM compiler plugin that differentiates LLVM IR directly.
You write normal C++ with `double`, compile with Clang + Enzyme plugin, and it
generates adjoint code at compile time.

**Why it matters for QuantLib**:

- **No type change**: QuantLib stays `double` — no `AReal<double>`, no
  expression templates, no SWIG type mapping headaches
- **Branches just work**: Since Enzyme operates on the compiled IR,
  `if/else`, `switch`, loops with data-dependent iteration counts, virtual
  dispatch — all differentiated correctly and automatically
- **No tape overhead**: No runtime tape allocation/recording; adjoint code is
  generated statically
- **Handles the exact HullWhite pattern natively**:
  ```cpp
  // This "just works" with Enzyme — no ABool::If needed
  if (_a < std::sqrt(QL_EPSILON)) {
      v = sigma()*B(maturity, bondMaturity) * std::sqrt(maturity);
  } else {
      v = sigma()*B(maturity, bondMaturity) *
          std::sqrt(0.5*(1.0-std::exp(-2.0*_a*maturity))/_a);
  }
  ```
- **Performance**: Generates optimized adjoint code with LLVM's full
  optimization pipeline (inlining, vectorization, dead code elimination).
  Benchmarks consistently show it matching or beating hand-written adjoints.

**Downsides**:

- Requires **Clang** (not GCC) — the current build uses GCC
- Integration complexity: need to annotate which functions to differentiate
  with `__enzyme_autodiff()`
- Doesn't support all C++ patterns (some virtual dispatch, exceptions, and
  complex allocators can cause issues)
- QuantLib's heavy use of `ext::shared_ptr`, `Handle<T>`, and the Observer
  pattern may need careful handling
- No Python-level tape API — the AD happens entirely at the C++ compilation
  level, so the Python user can't register inputs/outputs dynamically

**Maturity**: Active development at MIT, used in production by several
research groups. Well-tested on scientific C++ codes. Under active
development — LLVM version compatibility can be fragile.

**Verdict**: **Best theoretical fit** for QuantLib's branching problem. The
main barriers are (a) GCC→Clang migration, (b) QuantLib's OOP patterns
(shared_ptr graphs, virtual dispatch), and (c) loss of the Python-level tape
API that lets users choose what to differentiate at runtime.

---

## 2. CppAD — most mature operator-overloading alternative

**What it is**: Bell Labs-originated operator-overloading AD library for C++.
Records a tape (called "ADFun"), supports forward and reverse mode.

**Branching**: Has the **same fundamental problem** as XAD — branches are
invisible to the tape. However, CppAD provides:

```cpp
// Conditional expression recorded on tape (like ABool::If)
AD<double> result = CppAD::CondExpLt(a, epsilon, true_val, false_val);
```

This is equivalent to `xad::less(a, eps).If(true_val, false_val)`.

**JIT support**: CppAD has `cppad_jit` which compiles the recorded tape to C
source code, then compiles with the system C compiler. Slower compilation
than Forge (spawns a compiler process) but the generated code handles
`CondExp` correctly.

**Advantages over XAD**:

- Fully open source (EPL-2.0), no proprietary JIT extension needed
- `CondExpLt/Le/Gt/Ge/Eq` are first-class tape operations — not a separate
  "ABool" extension
- More mature ecosystem, better documentation
- `CppAD::checkpoint` for tape segmentation (reduces memory)
- `CppAD::atomic` for hand-coded derivatives of complex sub-functions
- Sparsity pattern detection for efficient Jacobian/Hessian computation

**Disadvantages**:

- Same operator-overloading overhead as XAD (type change required)
- Same branching limitation — still need `CondExp` instead of `if`
- JIT compilation is slower (compiler subprocess vs Forge's in-process
  AsmJit)
- Expression templates are less optimized than XAD's (more tape entries per
  operation)

**Verdict**: Lateral move — same paradigm, same branching problem. Better
`CondExp` API but no fundamental advantage.

---

## 3. CoDiPack — modern, high-performance operator overloading

**What it is**: Developed at TU Kaiserslautern, designed for
high-performance computing. Very efficient tape implementation with
primal-value taping.

**Branching**: Same problem. CoDiPack does **not** have a built-in
`CondExp`/`If` mechanism. It focuses on tape efficiency rather than JIT
re-evaluation.

**Advantages**:

- Very fast tape recording (primal-value taping avoids expression template
  overhead)
- Low memory footprint
- Excellent OpenMP parallel adjoint support
- Well-suited for CFD/PDE codes with fixed control flow

**Disadvantages**:

- No JIT backend
- No conditional expression support — worse than XAD for the branching
  problem
- Less financial-domain usage

**Verdict**: Better raw tape performance but worse for the specific
JIT+branching problem.

---

## 4. ADOL-C — oldest, most established

**What it is**: The original operator-overloading AD library (since 1996).
Records a tape in a custom binary format.

**Branching**: Has `condassign()`:

```cpp
condassign(result, condition, true_val, false_val);
// Records conditional on tape
```

**Disadvantages**:

- Oldest codebase, C-style API
- Slowest tape recording of all modern alternatives
- No JIT (tape is interpreted)
- `condassign` API is clunkier than CppAD's `CondExp` or XAD's `ABool::If`

**Verdict**: Legacy tool. No advantage over XAD.

---

## 5. Sacado (Trilinos) — scientific computing AD

**What it is**: Part of Sandia's Trilinos framework. Supports forward mode
(dual numbers) and reverse mode via operator overloading.

**Advantages**:

- Excellent forward-mode performance (template metaprogramming, no tape)
- Supports nested differentiation (Hessians via forward-over-reverse)
- Battle-tested in large-scale scientific computing

**Disadvantages**:

- Primarily designed for forward mode; reverse mode less mature
- Massive dependency (Trilinos ecosystem)
- No JIT, no conditional expression support
- Overkill for financial pricing

**Verdict**: Not a fit for this use case.

---

## 6. JAX / PyTorch — Python-native AD

Worth mentioning but fundamentally different paradigm:

**JAX** (`jax.grad`): Traces Python functions, applies source-to-source
transformation on the traced computation graph. Handles branches via
`jax.lax.cond()` (like `ABool::If`). XLA JIT compilation to CPU/GPU.
Incredibly mature.

**Problem**: QuantLib is C++. You'd need to rewrite all pricing logic in
Python/JAX, losing QuantLib entirely. Not viable for an existing C++
codebase.

---

## Summary Matrix

| Tool | Language | Approach | Branches | JIT | Open Source | QuantLib Integration |
|---|---|---|---|---|---|---|
| **XAD + Forge** | C++ | Operator overloading + JIT | ❌ (needs ABool::If) | ✅ (AsmJit) | Partial (Forge/ABool proprietary) | ✅ Current |
| **Enzyme** | C++ (Clang) | LLVM IR transformation | ✅ Native | N/A (compile-time) | ✅ (Apache-2.0) | ⚠️ Requires Clang, no runtime tape API |
| **CppAD** | C++ | Operator overloading | ❌ (needs CondExp) | ✅ (C codegen) | ✅ (EPL-2.0) | ⚠️ Type change like XAD |
| **CoDiPack** | C++ | Operator overloading | ❌ (no CondExp) | ❌ | ✅ (GPL-3.0) | ⚠️ Type change, no branch support |
| **ADOL-C** | C++ | Operator overloading | ❌ (needs condassign) | ❌ | ✅ (EPL/GPL) | ⚠️ Legacy API |
| **Sacado** | C++ | Operator overloading | ❌ | ❌ | ✅ (BSD) | ❌ Heavy deps |

---

## Recommendation

**Enzyme is the only tool that fundamentally solves the branching problem.**
Every operator-overloading library (XAD, CppAD, CoDiPack, ADOL-C) has the
same issue — C++ `if` statements are invisible to the tape. They all require
manual conversion to `CondExp`/`condassign`/`ABool::If`. Enzyme
differentiates the compiled LLVM IR directly, so branches, loops, and virtual
dispatch just work.

**However**, Enzyme has practical integration barriers:

1. Requires **Clang** (the current build uses GCC)
2. **No Python-level tape API** — users can't dynamically choose inputs at
   runtime
3. QuantLib's `shared_ptr` / `Handle<T>` / Observer patterns need careful
   testing
4. LLVM version coupling makes the build fragile

**Pragmatic path**: Stick with XAD + Forge for now, implement the
`ABool::If` fixes from the implementation plan (only 3 code changes needed),
and prototype an Enzyme integration as a parallel R&D effort to evaluate
feasibility. If the Enzyme prototype works, it would eliminate the entire
class of branching problems permanently.

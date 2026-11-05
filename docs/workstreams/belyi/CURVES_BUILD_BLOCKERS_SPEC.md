# `rustmath-curves` build blockers — specification (what blocks adding to `belyi`)

**Provenance (2026-07-02):** produced while wiring the M23 Belyi construction into
`rustmath-curves/src/belyi/`. The crate **library does not compile** (25 errors),
so *any* new `belyi` code — and the existing `belyi` module itself — cannot be
built or tested in place. Every error is in the **genus-≥2 hyperelliptic /
Jacobian curve-arithmetic stack**, which `belyi` does not use. Companion:
`M23_BELYI_CONIC_PORT_SPEC.md` (§P1 depends on this being cleared).

---

## 0. Executive summary

| | |
|---|---|
| **Lib-build errors** | **25** (`cargo build -p rustmath-curves`) |
| **Files with errors** | 4: `cantor.rs`, `divisor.rs`, `jacobian.rs`, `hyperelliptic.rs` |
| **Error codes** | `E0308` ×14 (type mismatch), `E0277` ×8 (`F: EuclideanDomain`), `E0599` ×3 (`contains_point` / `F: NumericConversion`) |
| **Root causes** | (1) foundation-API drift to `Option`/`Result` returns; (2) missing `EuclideanDomain` bound; (3) missing `NumericConversion` bound |
| **`belyi` dependency on these** | **NONE** — verified: no `belyi/*.rs` imports `cantor`/`jacobian`/`hyperelliptic`/`divisor`/`riemann_roch`/… |
| **Why `belyi` is blocked anyway** | `lib.rs` declares every module `pub mod`; the crate is all-or-nothing |
| **Additional `cargo test`-only breakage** | `riemann_roch`, `singularities`, `weierstrass`, `parameterization`, `special_divisors`, `lfunction` fail only when test code compiles (do **not** block the lib or `tests/` integration binaries) |

**Bottom line:** the blockers are a *self-contained, unrelated* subsystem. They
can be fixed (≈25 mechanical edits) or gated behind a feature; either unblocks
`belyi` immediately. Recommendation in §6.

---

## 1. Root-cause taxonomy (with the general fix)

### RC-1 — Foundation-API drift to `Option`/`Result` (all 14 `E0308`)
`rustmath-polynomials` / `rustmath-core` evolved so that fallible/partial
operations now return wrapped types, but the curve code still consumes bare
values. Observed diagnostics:
- `expected usize, found Option<usize>` — `Polynomial::degree()` now returns
  `Option<usize>` (`None` for the zero polynomial). Call sites do `deg() > 0`,
  `fn degree(&self)->usize { self.u.degree() }`.
- `expected type parameter F, found Option<F>` / `found Result<F, MathError>` —
  a coefficient/eval accessor now returns `Option<F>`/`Result<F,_>`.
- `expected Option<usize>, found integer` — a helper is expected to return
  `Option<usize>` but returns a bare integer (`Some(_)` missing).
- `expected Result<(_,_),_>, found (_,_)` — `Polynomial::div_rem` now returns
  `Result<(Q,R),_>` (or `Option`); code destructures `let (q,r) = a.div_rem(&b);`.

**Fix pattern:** thread the wrapper — `?` in `Result`-returning fns, `.unwrap()`/
`.expect("…")` where an invariant guarantees success (e.g. monic divisor u),
`.unwrap_or(0)` for `degree()` of possibly-zero polynomials, and `Some(_)`/`Ok(_)`
on the return paths. Prefer `?`/typed errors over `unwrap` on genuinely fallible
paths (status-honesty rule).

### RC-2 — Missing `F: EuclideanDomain` bound (all 8 `E0277`)
`UnivariatePolynomial::<R>::is_square_free` and `CantorAlgorithm::{reduce, add,
double, scalar_multiply, order}` require `R/F: EuclideanDomain` (they use `gcd` /
`div_rem`). The callers bound `F` only by `Field` (`use rustmath_core::{Ring,
Field}`), so the tighter bound is unmet. Sites: `hyperelliptic.rs:45,94`
(`f.is_square_free()`), `jacobian.rs` (Cantor calls).

**Fix pattern (choose one):**
- **Local:** add `+ rustmath_core::EuclideanDomain` to the `F` bound on the
  affected `impl` blocks / methods in `hyperelliptic.rs` and `jacobian.rs`.
- **Root (preferred, but foundation-touch):** add a blanket
  `impl<F: Field> EuclideanDomain for F` in `rustmath-core`/`rustmath-rings`
  (every field is a Euclidean domain with trivial valuation). Fixes the whole
  class at once — but edits shared foundation, so coordinate per MASTER_PORT_PLAN
  Wave-0 rules.

### RC-3 — Missing `F: NumericConversion` bound (all 3 `E0599`)
`HyperellipticCurve::<F>::contains_point` lives in
`impl<F: Field + Clone + PartialEq + rustmath_core::NumericConversion>`
(hyperelliptic.rs:30). `JacobianGroup::<F>` methods (`jacobian.rs:59,83,86`) call
`self.curve.contains_point(..)` with `F: Field` only, so the method is not in
scope.

**Fix pattern:** propagate `+ Clone + PartialEq + NumericConversion` to the `F`
bound on `JacobianGroup`'s `impl`/methods (or relax `contains_point`'s own bounds
if `NumericConversion` is not actually needed by its body).

---

## 2. Per-module specification

### 2.1 `hyperelliptic.rs` — base of the stack (4 lib errors)
`y² = f(x)`, genus, `contains_point`. Imports: none of the curve stack.
| Lines | Code | Cause | Fix |
|---|---|---|---|
| 45, 94 | E0277 | `f.is_square_free()` needs `F: EuclideanDomain` | RC-2 |
| 61, 88 | E0308 | `Option`/`Result` drift (degree / eval) | RC-1 |
The `impl` at line 30 already carries `NumericConversion`; the struct at line 23
is `F: Field`. Fixing this file first unblocks its dependents.

### 2.2 `divisor.rs` — Mumford divisors (6 lib errors, all E0308)
Lines **37, 61, 66, 109, 134, 177**. All RC-1: `Polynomial::degree()` →
`Option<usize>` (`u.degree() > 0` at :37, `fn degree(&self)->usize` at :66) and
`div_rem`/eval wrapper drift. No trait-bound issues. Depends on nothing in-stack.

### 2.3 `cantor.rs` — Cantor reduction/composition (6 lib errors, all E0308)
Lines **89, 98, 108, 128, 173, 191**. All RC-1: `let (q,r) = u.div_rem(&d);`
against a now-`Result`/`Option` `div_rem` (see :89,:98), plus degree/`is_zero`
wrapper drift. Uses `divisor` (fix 2.2 first). `CantorAlgorithm` methods also need
`F: EuclideanDomain` (RC-2) once callers compile.

### 2.4 `jacobian.rs` — Jacobian group law (9 lib errors)
Depends on `cantor`, `divisor`, `hyperelliptic` (fix those first).
| Lines | Code | Cause | Fix |
|---|---|---|---|
| 59, 83, 86 | E0599 | `contains_point` — `F` lacks `NumericConversion` | RC-3 |
| 20, 91, 103, 113, 131, 148, 215 | E0277 | Cantor calls need `F: EuclideanDomain` | RC-2 |

### 2.5 Test-only breakage (does NOT block the lib or `tests/` integration)
`riemann_roch`, `singularities`, `weierstrass`, `parameterization`,
`special_divisors`, `lfunction` compile in the **library** but fail under
`cargo test -p rustmath-curves` (errors in their `#[cfg(test)]`/example code, same
RC-1/RC-2 flavours). **Consequence:** `belyi` can be exercised via a
`rustmath-curves/tests/belyi_*.rs` **integration** binary (compiles against the
green lib) even before these are fixed — the same tactic used for
`rustmath-numerical` (`tests/nonlinear_system_test.rs`). Full `cargo test` green
needs them fixed too, but that is not on the `belyi` critical path.

---

## 3. Dependency graph (why order matters)

```
hyperelliptic ──┐
                ├──► jacobian        (jacobian uses cantor, divisor, hyperelliptic)
divisor ──► cantor ──┘

belyi ── (independent) ──► rustmath-quadraticforms, Rational   [NO edge to the stack]
```
Fix order to reach a green lib fastest: **divisor → hyperelliptic → cantor →
jacobian** (leaves first).

---

## 4. Unblock strategies

**A. Fix the 25 lib errors (recommended for correctness).**
Mechanical: ~14 `?`/`unwrap`/`Some`/`Ok`/`unwrap_or(0)` edits (RC-1) + ~8 bound
additions (RC-2) + ~3 bound additions (RC-3). Touches 4 files. Small, local, no
new logic. Restores the whole genus-≥2 stack. *Then* optionally fix the 6
test-only modules for `cargo test` green.

**B. Feature-gate the genus-≥2 stack (fastest unblock for `belyi`).**
In `lib.rs`, put `#[cfg(feature = "genus2")]` on `pub mod {hyperelliptic,
divisor, cantor, jacobian}` (and the test-only-broken modules if needed), and add
`genus2` to `[features]` (off by default until fixed). The lib then compiles with
`belyi` present; `belyi` work proceeds immediately. **Cost:** genus-≥2 API is
absent unless `--features genus2`; must be tracked as debt, not silently dropped
(status-honesty).

**C. Split `belyi` into its own crate `rustmath-belyi`.**
Cleanest isolation (belyi only needs `Rational` + `rustmath-quadraticforms`), and
matches "one crate per concern". **Cost:** a new crate + workspace/member edits +
moving files; larger than A or B; revisit `DESSIN_REFACTOR_PLAN` (which chose "no
new crate, fold into existing").

**Recommendation:** **B now, A soon.** Feature-gate to unblock `belyi`
construction work this session; land the 25-error fix (A) as a separate,
self-contained PR on `rustmath-curves` so the genus-≥2 stack returns to green
without entangling the M23 build. Avoid C unless the coupling proves painful.

---

## 5. Per-line fix checklist

```
divisor.rs      : 37 61 66 109 134 177      RC-1  (Option/Result unwrap; degree()->Option)
hyperelliptic.rs: 61 88                      RC-1
hyperelliptic.rs: 45 94                      RC-2  (+ EuclideanDomain on F)
cantor.rs       : 89 98 108 128 173 191      RC-1  (div_rem now Result/Option)
jacobian.rs     : 59 83 86                    RC-3  (+ NumericConversion on JacobianGroup F)
jacobian.rs     : 20 91 103 113 131 148 215   RC-2  (+ EuclideanDomain on F)
```

Verify after each file: `cargo build -p rustmath-curves 2>&1 | grep -c '^error'`
should monotonically decrease to 0. Then `cargo build -p rustmath-curves` green ⇒
`belyi` buildable ⇒ add construction code + `tests/belyi_*.rs` integration tests.

---

## 6. Discipline / ownership

- `rustmath-curves` is a **firewall crate** (active dessin/IGP24 workers,
  MASTER_PORT_PLAN). Strategy A/B here is *its* owner's work, not a port-worker
  fan-out; keep edits to the 4 named files (+ `lib.rs` feature lines).
- **New files only** rule is respected by strategy B (only `lib.rs` cfg lines) and
  by adding `belyi` code as new modules; strategy A edits existing broken logic,
  which is permitted for the crate owner repairing build debt (cite this spec).
- **Status honesty:** if B is chosen, the gated genus-≥2 absence is *debt*, logged
  here and in `BUILD_STATUS.md`; a feature silently off must never read as "done".
- **Root-cause note for the survey:** RC-1/RC-2/RC-3 are the same "foundation
  evolved, dependents not re-adopted" gap MASTER_PORT_PLAN §1 calls out; a
  workspace-wide sweep for `degree()`-returns-`Option` and `Field`-vs-
  `EuclideanDomain` would likely clear similar breakage elsewhere.

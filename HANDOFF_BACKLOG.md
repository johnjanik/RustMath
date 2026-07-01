# RustMath hand-off backlog (for an independent instance)

Items found during the dessin_engine→RustMath refactor (see DESSIN_REFACTOR_PLAN.md)
that are **orthogonal to the M23 solve pipeline** — safe to fix independently.

**Coordination rule:** the focused M23 thread is actively porting into these crates
— `rustmath-quadraticforms`, `rustmath-numberfields`, `rustmath-groups`,
`rustmath-polynomials`, `rustmath-numerical`, `rustmath-curves`. Prefer to work
*outside* those, or coordinate before editing them. All items below are
**non-blocking** for the M23 work (that path needs the crate *libraries* to
compile, which they do — not the broken test modules or the full DB).

---

## 1. `rustmath-groups`: pre-existing test-build breakage (~158 errors)
The crate's own `#[cfg(test)]` modules (`free_group.rs`,
`additive_abelian_wrapper.rs`, `group_exp.rs`, …) fail to compile from API drift
(e.g. `a.mul(&b)` type mismatches). This predates our work (confirmed by stashing
our edits — 158 errors at HEAD with our files unreferenced). Consequence:
`cargo test -p rustmath-groups` can't build the unit-test binary. Our new code was
verified via a `tests/` integration test instead. Fix: update those test modules
to the current API. Library itself is fine.

## 2. Missing `rustmath-groups/data/transitive_24.jsonl` (full transitive-24 DB)
Only `data/transitive24_cycletypes.jsonl` is present. Two integration tests
(`galois_narrowing::db_and_support_load`, `native_closure_matches_precomputed_support`)
`.expect()` the absent generator DB and fail. `Db::load_default` now fails
gracefully (fixed), but the full DB should be regenerated/committed if the
generator-level API is wanted. `igp24` and our pipeline use `CycleTypeSupport`
(present), so this is not on the critical path.

## 3. `rustmath-polynomials`: `groebner.rs::reduce` placeholder coefficient division
`groebner.rs`'s reducer uses a unit quotient coefficient (documented placeholder).
It returns correct bases for monomial / unit-leading-coefficient ideals but
**infinite-loops** on ideals needing true coefficient division (e.g. `(x²−1, y²−1)`).
Consequences: two pre-existing `quotient.rs` tests (`test_multivariate_quotient`,
`test_quotient_reduction`) hang forever (predate our work, commit f7ec5ad), so
`cargo test -p rustmath-polynomials` never terminates (use `--lib` + skip those two,
which is green: 216 pass). Also limits the new `staircase.rs` (vdim) and blocked a
real `elimination.rs`. Fix: implement true polynomial coefficient division in the
reducer. NOT on the M23 critical path — Gröbner is demoted in favor of parameter
homotopy for the solve; `PolySystem` (the piece Wave 2/3 need) is done and green.

## 4. `rustmath-numerical`: pre-existing `src/lib.rs` test-module breakage
The crate's own `#[cfg(test)]` block in `src/lib.rs` fails to compile — `f.eval`
is ambiguous between the `Integrable` and `Optimizable` traits (lines ~216/222).
Predates our work (confirmed by `git stash`). Consequence: `cargo test -p
rustmath-numerical --lib` won't build. Our Wave-2 code was verified via a `tests/`
integration test (`homotopy_exactify.rs`, green); the library builds clean. Fix:
disambiguate the trait method call (e.g. `Integrable::eval(&f, ...)`). Not on the
M23 critical path.

## 5. `rustmath-curves`: pre-existing foundation breakage BLOCKS in-tree build (26 errors)
`cargo build -p rustmath-curves` fails with ~26 committed errors in unrelated
modules (`cantor`, `divisor`, `hyperelliptic`, `jacobian`, `lfunction`,
`parameterization`, `plane_curve`, `riemann_roch`, `singularities`, `weierstrass`)
from a foundation trait-bound migration (`NumericConversion`/`EuclideanDomain`).
Verified identical (26) with and without our `belyi/` changes — zero attributable
to `belyi/`. Because the breakage is committed (not stash-able), the whole lib
can't link, so our `belyi/` code cannot build/test in-tree; it was verified via a
standalone `#[path]`-include harness (22 tests green). **This is likely resolved by
the `magma-port/wave0-foundation` migration once it reaches `rustmath-curves`.**
BLOCKING for running the M23 solve pipeline in-tree — coordinate with the
magma-port effort. (Our belyi source is committed and correct; it just can't link
until curves' other modules are migrated.)

## 6. Coordination: shared main working tree vs the magma-port worktrees
The magma-port instance uses isolated `.claude/worktrees/agent-*` (good). Our
dessin refactor has been working in the SHARED main tree on branch
`refactor/dessin-to-rustmath`; mid-session an external checkout switched the main
tree (refactor → wave0-foundation → main), which broke an agent's imports until
switched back. Two efforts must not share one working tree with branch switches.
Resolution options (user decision): put the dessin refactor in its own dedicated
worktree too, or reserve the main tree for one effort. Our work is committed to
`refactor/dessin-to-rustmath`, so it survives switches, but active agents can be
disrupted.

---
*(append new orthogonal breakage/dependency items here as they are found)*

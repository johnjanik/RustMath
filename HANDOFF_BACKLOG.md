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

---
*(append new orthogonal breakage/dependency items here as they are found)*

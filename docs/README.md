# RustMath docs — start here

This directory is the **single source of truth** for RustMath's status and direction, as of **2026-07-09**. It replaces the 70 overlapping status/plan/TODO/analysis documents that had accumulated at the repo root (now in [`archive/`](./archive/), preserved in git).

## The two documents that matter

| Doc | What it is |
|---|---|
| **[SURVEY.md](./SURVEY.md)** | Code-grounded survey of **all 66 crates** as a shared backend for **both SageMath and MAGMA**. Master coverage matrix (crate × Sage module × MAGMA chapter × verdict), build/test health, cross-cutting structural findings, and a per-crate appendix. |
| **[PLAN.md](./PLAN.md)** | The unified phased implementation plan to full Sage + MAGMA parity. Phase 0 (green + honest) → Phase 6 (front-end interop), with sequencing and priority tiers. |

## Reference (kept, still useful)

- [`reference/magma_coverage.md`](./reference/magma_coverage.md) — MAGMA handbook ch 17–159 → RustMath coverage matrix (per-chapter).
- [`reference/build_backlogs.md`](./reference/build_backlogs.md) — detailed per-crate MAGMA backlog (from the port survey).
- [`inverse_galois_mvp.md`](./inverse_galois_mvp.md) — the IGP24 inverse-Galois MVP spec.

## Headline status (see SURVEY for detail)

- **64 / 66 crate libs build clean.** `databases` needs native `libssl-dev` (env-only); `interfaces` has 3 trivial `test_long.rs` type errors.
- **Test suites are not fully green:** `modules`/`category`/`schemes` test builds fail; `topology` has 7 failures. The ~11.9K figure counts declared `#[test]` markers, not passes.
- **Verdict roll-up:** ~6 Complete · ~24 Substantial · ~28 Partial · ~3 Stub/Skeleton · 3 tooling.
- **Strong:** integers, matrix, polynomials, numberfields, groups, algebras, liealgebras, combinatorics, symbolic. **Weak:** arithmetic geometry (schemes/curves/elliptic/modular), function fields, class field theory, general Galois/group theory, arbitrary-precision analysis.
- **Biggest risk:** category-3 **facades** — real-looking APIs that return plausible constants (function fields, LP/MIP, Hecke eigenvalues, `find_singularities`, …). Phase 0 converts these to explicit non-results.

## Layout (reorganized 2026-11-05)

- **Project direction:** MAGMA is the primary specification; SageMath is a secondary reference.
- **Repo root** keeps only `CLAUDE.md`, `README.md`, `Cargo.toml`, `Cargo.lock`, `PROJECT_TRACKER.md`.
- [`magma/`](./magma/): per-intrinsic MAGMA tracker generated from the handbook chapter digests.
- [`BRANCHES.md`](./BRANCHES.md): audit of all 20 remote branches and the belyi + lava-galois merge.
- [`workstreams/belyi/`](./workstreams/belyi/): Belyi/dessin/curves specs and build handoffs
  (`DESSIN_REFACTOR_PLAN.md`, `M23_*`, `CURVES_*`, `RUSTMATH_BUILD_HANDOFF*.md`, `BELYI_SESSION_SOW.md`).
- [`algorithm_notes/`](./algorithm_notes/), `cft_*.md`: design notes (Stauduhar, S-units, CFT engine).
- [`archive/`](./archive/): superseded status docs. `archive/root-legacy/` holds the 2025 root files
  (SageMath trackers, logs, old README/CLAUDE.md, `THINGS_TO_DO.md`); scripts are in `../tools/legacy/`.
- [`paper/`](./paper/): `rustmath_paper.tex`.
- The `*.pdf` files in this directory are SageMath reference manuals (194 MB), kept pending a decision.

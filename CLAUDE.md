# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this project is

RustMath is a pure-Rust, zero-`unsafe` port of the **MAGMA** computational algebra system. The
specification is the *MAGMA Handbook of Functions*, digested per chapter in
`rustmath-lava/docs/Chapter_NN_*.md` (intrinsics, signatures, algorithms, bibliography keys).
SageMath is a secondary reference (API shape, cross-check values). The 2025 SageMath-only framing is
archived in `docs/archive/root-legacy/`.

Implement from the **published algorithms** cited in each chapter digest. Use the handbook as the
interface spec only; do not copy MAGMA package (`.m`) source.

## Start of every session

1. Read `docs/README.md` (status index) and `docs/PLAN.md` (phases).
2. For MAGMA coverage, use `docs/magma/` (per-intrinsic tracker). For what the code actually does,
   use `PROJECT_TRACKER.md` (algorithm inventory with `file:line`, facade/broken flags).
3. Cloud containers are ephemeral: commit and push before ending. Update the tracker rows you touched.

## Commands

```bash
cargo check --workspace                 # whole workspace (first build compiles GMP via rug: minutes)
cargo test -p rustmath-numberfields     # one crate
cargo test -p rustmath-galois --test descent
cargo clippy -p <crate>
cargo fmt
python3 tools/magma_tracker/build.py    # regenerate docs/magma/ from the chapter digests
```

## Rules for code changes

- **Honesty over coverage.** A *facade* (an API returning a plausible fabricated value) is a defect.
  Missing functionality returns `Err`/`unimplemented!()`. Certifying routines distinguish
  `Decided` from `Unresolved`; a bounded search that finds nothing never decides.
- Consume `rustmath-core` traits and existing numeric crates; no new private `num-*` wrappers.
- Cite the MAGMA chapter (and paper) in each new file's header. Port handbook examples as tests,
  with reference values from LMFDB, PARI/GP or Sage.
- One crate per PR where possible; keep that crate build-green and test-green.

## Architecture (bottom-up)

```
L0  Exact kernel       core, integers, rationals, finitefields (GF(p^n), Z/mZ, Galois rings)
L1  Polys/linalg       polynomials (factorization, resultants), matrix (LLL/HNF/SNF), powerseries
L2  Fields             numberfields (round2, ideals, class/unit/S-unit groups, CFT), padics +
                       rings::padics (OM/Montes, extensions), functionfields
L3  Groups             groups (perm groups, Schreier-Sims, degree-24 transitive atlas)
L4  Galois             galois (exact low-degree engine + Stauduhar descent, deg-24 narrowing)
L5+ Geometry, modular  curves (incl. Belyi/dessins), ellipticcurves, modular, schemes
L7  Front-end          rustmath-lava (Magma-language tooling), rustmath-igp24 (CLI)
```

Non-obvious points:

- `rustmath-padics` is a re-export shim over `rustmath_rings::padics`, plus its own `montes` /
  `newton_polygon` modules.
- `rustmath-finitefields` carries two GF(p^n) stacks (`finite_field`/`poly_factor` canonical at the
  crate root; `ff_poly`/`cantor_zassenhaus`/`irreducible` used by `rustmath-galois` via module paths).
- `rustmath-galois::galois_group` is the exact engine; `stauduhar_galois_group` and `descent24` are
  the numeric/p-adic descent stack.
- `Ring` is not dyn-safe by design; object-safe maps go through `rustmath_core::morphism`.
- The degree-24 atlas data (`rustmath-groups/data/*.jsonl`) is gitignored; regenerate with
  `gap -q build_transitive_24.g` and `build_cycletypes.py` (see `rustmath-groups/data/README.md`).

## Important files

| File | Purpose |
|---|---|
| `docs/README.md` | Status index, start here |
| `docs/PLAN.md` | Phased plan |
| `docs/magma/` | MAGMA intrinsic tracker (generated) and status summary |
| `PROJECT_TRACKER.md` | Algorithm inventory (ground truth for what the code does) |
| `docs/BRANCHES.md` | Branch audit and merge record |
| `docs/workstreams/belyi/` | Belyi/dessin/curves workstream specs and handoffs |
| `rustmath-lava/docs/` | MAGMA handbook chapter digests (the spec) |

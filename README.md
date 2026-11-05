# RustMath

A pure-Rust, zero-`unsafe` port of the **MAGMA** computational algebra system: the mathematics of the
*MAGMA Handbook of Functions* (159 chapters, ~14,200 intrinsics), reimplemented from the published
algorithms the handbook cites. SageMath remains a secondary reference for API shape and test values.

Status: research-grade and incomplete. The authoritative, code-grounded status lives in the tracker
files below, not in this README.

## Where things are

| What | Where |
|---|---|
| Start here (status, plan, layout) | [`docs/README.md`](docs/README.md) |
| The phased plan | [`docs/PLAN.md`](docs/PLAN.md) |
| MAGMA intrinsic tracker (one row per handbook intrinsic) | [`docs/magma/`](docs/magma/) |
| Algorithm inventory (one row per implemented algorithm, with `file:line`) | [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) |
| Per-chapter MAGMA handbook digests (the spec) | [`rustmath-lava/docs/`](rustmath-lava/docs/) |
| Branch history and consolidation | [`docs/BRANCHES.md`](docs/BRANCHES.md) |
| Archived 2025 SageMath-era docs and scripts | [`docs/archive/`](docs/archive/), [`tools/legacy/`](tools/legacy/) |

## Layout

A Cargo workspace of ~68 crates. The computational spine, bottom-up:

- **Exact kernel:** `rustmath-core` (traits), `-integers`, `-rationals`, `-finitefields`, `-reals`, `-complex`
- **Polynomials, linear algebra, lattices:** `-polynomials`, `-matrix`, `-powerseries`
- **Global and local fields:** `-numberfields`, `-padics` / `-rings::padics`, `-functionfields`
- **Groups and Galois theory:** `-groups` (incl. the degree-24 transitive atlas), `-galois`
- **Arithmetic geometry:** `-curves`, `-ellipticcurves`, `-modular`, `-schemes`
- **Front-end:** `rustmath-lava` (Magma-language formatter/highlighter, MIT, vendored), `rustmath-igp24` (Galois-ID CLI)

## Build

```bash
cargo check --workspace
cargo test -p rustmath-numberfields
```

`rustmath-reals`, `-complex` and `-rings` depend on `rug`, which builds GMP/MPFR from source on a
fresh machine (several minutes). The degree-24 group atlas data is not in git; regenerate it with
`rustmath-groups/data/fetch_transitive_24.py`.

## License

GPL-2.0-or-later for RustMath; `rustmath-lava` keeps its MIT `LICENSE`.

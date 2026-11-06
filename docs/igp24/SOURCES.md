# IGP24 sources (the `inverse_galois` repo)

The IGP24 competition work lives at **https://codeberg.org/johnjanik/inverse_galois**, pinned here at
commit **`703488c`** (2026-10-07, `main`). RustMath code that says it was ported from
`inverse_galois/...` refers to this repo. Permalink pattern:

`https://codeberg.org/johnjanik/inverse_galois/src/commit/703488c1de80d825b784f8a3eb96479ee741df82/<path>`

The repo stays the canonical home for this material. RustMath copies only what its build or tests
need (currently: nothing is copied; `transitive24_cycletypes.jsonl` is regenerated from it).

## Data RustMath derives from it

| RustMath file | Built from | How |
|---|---|---|
| `rustmath-groups/data/transitive24_cycletypes.jsonl` (gitignored) | `frobenius/cycle_types.jsonl` (LMFDB cycle-type distributions, made by `frobenius/compute.sage`) | `python3 rustmath-groups/data/build_cycletypes.py <inverse_galois>/frobenius/cycle_types.jsonl` |
| `rustmath-groups/data/transitive_24.jsonl` (gitignored) | Not in this repo. GAP's TransGrp library (same numbering as LMFDB) | `gap -q rustmath-groups/data/build_transitive_24.g` (validated against LMFDB order/parity/solvability for all 25,000 groups) |

## Code ported into RustMath (originals to check ports against)

| RustMath | Original in `inverse_galois` |
|---|---|
| `rustmath-curves/src/belyi/*` (15 files), `rustmath-quadraticforms/src/{conic,hilbert,quaternion,ternary}.rs`, `rustmath-numerical/src/{homotopy,exactify}.rs`, `rustmath-numerical/src/root_finding/nonlinear_system.rs`, `rustmath-polynomials/src/poly_system.rs` | `M23/dessin_engine/` (incl. `src/conic.rs`, `src/hilbert.rs`, `src/quadratic_form.rs`), `M23/dessin_build/` |
| `rustmath-curves` Belyi high-precision path | `M23/route_a/belyi_shakedown/src/phi_hp.rs`, `M23/solve_runner/` |
| `rustmath-curves/tests/flags_m24_2_12_5.rs` | `M23/triple_2_12_5.json`, `M23/find_triple_2_12_5.sage` |
| `rustmath-numberfields/src/disc_score.rs` | `frobenius/disc_score.sage` |
| Construction engines (not yet ported; `docs/inverse_galois_mvp.md` L5) | `frobenius/generate.sage`, `frobenius/mestre_{an,n24,run,solve}.sage` |

## Notes and design documents

`inverse_galois_notes.tex` (entries 1–33), `P2_note.md`, `M23/m23_hidden_law.tex`,
`RUSTMATH_HANDOFF_PROMPT.md`, `rustmath_machinery_status.md`, `padelic_reuse_report.md`.

## Candidate test data (reference values)

- `frobenius/lmfdb_nf12_seeds.json`: 301 degree-12 fields keyed by Galois group `12Tj`.
- `frobenius/blind_classes.json`, `frobenius/open_cells.json`, `igp24-coverage/data/t24_atlas.json`
  (per-group invariants, Frobenius-blind classes).
- `FMFDB_datasets/lmfdb_gps_transitive_{4,6,8,12}.csv`, `lmfdb_gps_transitive_0618_0605.txt`
  (degree-24 group list without generators).

## Referenced but never committed (lost)

These are named in `docs/inverse_galois_mvp.md` but appear in no commit of the repo. They were only
on the original Mac, which was archived and wiped (2026-11); treat them as lost unless found in that archive:
`frobenius/ghost_classify.sage`, `frobenius/sunit_batch_v2.sage`, `frobenius/fingerprints.jsonl`,
`frobenius/fingerprint_audit.sage`, `frobenius/build3_support_vector.sage`,
`frobenius/directed_engine.sage`, `strategy_update_20260623.md`.

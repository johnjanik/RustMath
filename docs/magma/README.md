# MAGMA intrinsic tracker

One row per intrinsic in the *MAGMA Handbook of Functions*, generated from the chapter digests in
`rustmath-lava/docs/` (159 chapters, 13,948 rows; 13,341 in the mathematics scope, chapters 17–159).

| File | Contents | Edit? |
|---|---|---|
| `intrinsics.csv` | Every intrinsic: chapter, section, signature, description, algorithm, cited keys, home crate, status, candidate Rust location | Generated |
| `bibliography.csv` | Every bibliography key across all chapters, with the chapters citing it | Generated |
| `STATUS.md` | Per-chapter roll-up | Generated |
| `status_overrides.csv` | Hand- or agent-verified statuses, keyed by `id` | **Edit this** |

Regenerate after any code or override change:

```bash
python3 tools/magma_tracker/build.py
```

## Columns of `intrinsics.csv`

- `id`: stable key `chNNN.<FirstName>[.k]`. Use it in `status_overrides.csv` and commit messages.
- `scope`: `math` (ch 17–159) or `language` (ch 1–16: Magma language/runtime, a `rustmath-lava` concern).
- `refs`: bibliography keys cited in the description/algorithm cells (see `bibliography.csv`).
- `home_crates`: from `docs/reference/magma_coverage.md`.
- `status`: see below. `match_scope`: `home` (match in a home crate) or `elsewhere`.
- `rust_path`: candidate (or verified) `file:line`.
- `inventory_status`: if `rust_path` is a row in `PROJECT_TRACKER.md`'s Algorithm Inventory, its
  status there (`done`, `partial`, `facade`, `broken`, `missing`).

## Status values

| Status | Set by | Meaning |
|---|---|---|
| `missing` | generator | No Rust function with the snake_case name exists |
| `candidate` | generator | A same-named function exists. A lead, not evidence |
| `stub` | override | Exists but returns `Err`/`unimplemented!` |
| `partial` | override | Implemented for a restricted input class |
| `facade` / `broken` | override | Returns fabricated values / wrong or panics |
| `done` | override | Implemented with the handbook semantics, tests present |
| `verified` | override | `done` + tests against external reference values (LMFDB, PARI/GP, Sage) |
| `n/a` | override | Not meaningful for a Rust library (e.g. pure language syntax) |

Overrides always win over the generator. Put the evidence (test name, reference source) in `note`.

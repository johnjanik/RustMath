# Session handoff (2026-11-06, updated after the test-fix session)

Read this first, then `CLAUDE.md` and `docs/README.md`. It records where the project stands and
what to do next. **Tasks 0 and 1 are done** (see their status notes below). **Task 2 is next and
needs the user's go-ahead** before fanning out (it costs about 4-5M tokens).

## Where things stand

- **Direction:** RustMath is a pure-Rust port of MAGMA. The handbook digests in
  `rustmath-lava/docs/Chapter_NN_*.md` are the spec. SageMath is a secondary reference only.
  IGP24 is over; its code lives at https://codeberg.org/johnjanik/inverse_galois (map:
  `docs/igp24/SOURCES.md`).
- **Git:** one line of history. `main` contains everything (PRs #569, #570). The 19 old branches
  were deleted; their tips are tags `archive/<branch>` (`docs/BRANCHES.md`). Work on a new branch
  off `main`; cloud sessions can push only their own branch (no tags, no other branches).
- **Build and tests:** the whole workspace compiles, and `cargo test --workspace` is green:
  13,772 passed, 0 failed, 332 ignored at `e0a9c23` (`docs/TEST_BASELINE.md`). First build
  compiles GMP/MPFR for `rug` (minutes).
- **Environment:** the RustMath cloud environment's setup script installs GAP (`gap-core`,
  `gap-libs`, `gap-transgrp`) and builds the gitignored degree-24 atlas files in
  `rustmath-groups/data/`. Verified on its first real run (2026-11-06): GAP at `/usr/bin/gap`, both
  atlas files 25,000 rows, `galois_narrowing` 4/4.
- **Lost for good:** seven IGP24 files never committed anywhere (`docs/igp24/SOURCES.md`).
  Nothing in the port depends on them.

## Key files

| File | What |
|---|---|
| `docs/magma/intrinsics.csv` | 13,948 handbook intrinsics (13,341 math scope, ch 17-159), one row each, with status |
| `docs/magma/status_overrides.csv` | Reviewed statuses with evidence; the generator never overwrites these |
| `docs/magma/STATUS.md` | Per-chapter roll-up (generated) |
| `tools/magma_tracker/build.py` | Regenerates the three generated files above |
| `docs/magma/reading_list_core.md` | 330 papers cited by core chapters (179 priority A); user is collecting PDFs as `<key>.pdf` |
| `docs/TEST_BASELINE.md` | Workspace test results and failures by crate |
| `PROJECT_TRACKER.md` | Algorithm inventory by `file:line`, with facade/broken flags |
| `docs/PLAN.md` | Phased plan (see the 2026-11-05 update at the top) |

## Task 0: verify the environment (5 minutes). DONE 2026-11-06

Passed as written; the setup script worked. Re-run it at the start of a session if in doubt.

```bash
which gap && wc -l rustmath-groups/data/transitive_24.jsonl rustmath-groups/data/transitive24_cycletypes.jsonl
cargo test -p rustmath-groups --test galois_narrowing     # expect 4 passed
```

If GAP or either file is missing: run `apt-get install -y gap-core gap-libs gap-transgrp`, then
the two commands in `rustmath-groups/data/README.md`, and tell the user the setup script failed.

## Task 1: fix the 57 failing tests. DONE 2026-11-06

All 58 fixed (the 57 below, plus `perm23_classifier::transitive24_db_load_default_fails_gracefully`,
which assumed the atlas data was absent). There are 13 commits, `eb50f9b`..`e0a9c23`, one or two
per crate. Each fix was checked by an independent reviewer agent before it was committed. Each of the
21 library-test failures involved a code bug; three of those tests also had a wrong fixture or
expected value, corrected with proof (GAP or the definition). Most of the doctest failures were stale
examples, but giving them real assertions exposed further bugs and facades (Artin relators, the
Mathieu groups, ghost brackets, multimodular Bernoulli). `docs/TEST_BASELINE.md` has the per-crate
causes, and the `PROJECT_TRACKER.md` Change Log row (2026-11-06) and inventory rows have the
details. Things to know:
- `rustmath-algebras` doctests behind the `full` feature run only in a workspace build (through
  `rustmath-jupyter`). Test that crate alone with `--features full`.
- `rustmath-misc` doctests no longer write `output.txt` or leave `.tmp_dir_*` behind.
- Facades found but **left open** (rows in the tracker): the `GroupElement::order` default (a
  bounded search that decides "infinite"), the `Group::is_abelian` default `false`, the
  `Group::exponent` default `None`, `GenericParentWrapper::contains_element` (always true),
  `FreeModuleWithBasis::cardinality`, lieconformal `degree()` (None for every nonzero element),
  affine `compute_structure_data` (empty tables, so its bracket panics), the f.p.-group SNF, and the
  Artin/braid word problem. The Virasoro, N=2, Neveu-Schwarz and structure-coefficient
  lambda-brackets now panic honestly instead of returning empty maps; they are unimplemented.
- Behaviour and API changes with no callers outside their crate: `ClusterQuiver::mutation_type`
  now returns `Result`, `is_mutation_equivalent` returns `Option<bool>`, and `to_quiver` returns
  `Result`. `recover_forms*` return phase-normalised kernel vectors. `GroupLibGAP::contains` and
  `elements` panic (`unimplemented!`) when a group cannot be enumerated; use `try_contains`.
- Commit `3fab92c` lacks the blank line after its subject (cosmetic; history not rewritten).

The original task text follows.

Baseline (`docs/TEST_BASELINE.md`, with atlas data present): 57 failures.

| Target | Count | Notes |
|---|--:|---|
| `rustmath-groups` lib | 18 | `libgap_group`, `libgap_mixin` (pure-Rust GAP name-alikes), `semidirect_product`, `nilpotent_lie_group`, `generic::test_merge_elements` |
| `rustmath-groups` doctests | 8 | `groups_catalog`, `generic`, `semimonomial_transformation*` |
| `rustmath-lieconformal` doctests | 13 | stale examples |
| `rustmath-trees` / `-numbertheory` / `-modules` / `-algebras` doctests | 5 / 4 / 2 / 2 | stale examples |
| `rustmath-stats`, `rustmath-misc` doctests | 1 each | |
| `rustmath-quivers` lib | 2 | `cluster_quiver::test_cartan_companion`, `test_type_d_quiver` |
| `rustmath-curves` lib | 1 | `belyi::modular_forms_hp::recover_forms_centered_z_a_matches_uncentered`, 3.36e-10 divergence |

Priority: `rustmath-groups` first (it underlies Galois work). Rules:
- Fix the code when the test is right. Never weaken an assertion just to go green. For a stale
  doctest, update the example to the current API and make sure it still checks something real.
- If a test exposes a **facade** (fabricated return value), make the function return an honest
  `Err`/`unimplemented!()` and record it, rather than making the facade pass.
- Whether the groups failures predate the belyi/lava merge is unknown; `git log` on the files will tell.
- `rustmath-misc` tests write `rustmath-misc/output.txt` and a `.tmp_dir_*`; do not commit those.
- Run `cargo test -p <crate>` per crate; update `docs/TEST_BASELINE.md` at the end.

## Task 2: review the tracker for the 45 remaining core chapters

The automatic statuses (`candidate` = a same-named Rust fn exists, `missing` = none) are unreliable.
The pilot on ch 19, 38, 46 (231 rows) found only 17-42% of candidates were real, and some `missing`
rows were implemented under other names. Reviewed statuses go in `status_overrides.csv`.

**Remaining core chapters (4,930 rows):** 17, 18, 20-37, 39-45, 47-52, 57-68.

Method used in the pilot (about 90k Sonnet tokens per chapter; this task is about 4-5M tokens, so
confirm with the user before fanning out):
- One Sonnet subagent per chapter. It reads that chapter's rows from `intrinsics.csv`, searches
  the home crate (column `home_crates`) by **meaning**, not only by name, and writes
  `id,status,rust_path,note` for every row to a scratch CSV.
- Status vocabulary: `missing`, `stub`, `partial`, `facade`, `broken`, `done` (handbook semantics
  **and** a test exercising it, cite the test), `n/a`. Reserve `verified` for tests against
  external reference values (LMFDB, PARI/GP, Sage, GAP, handbook examples).
- Merge each CSV into `status_overrides.csv` (append `[review 2026-11-xx]` to `note`), spot-check
  2-3 claims per chapter against the code, run `python3 tools/magma_tracker/build.py`, commit.
- Subagents should write files rather than rely on their final message; in the pilot their
  reports did not come back, but the CSVs were complete.

## After these two tasks

Use the reviewed tracker to choose the first build targets, bottom-up through the core. Likely
candidates (`docs/PLAN.md`): complete GF(p^n); general Schreier-Sims (the keystone for groups and
Galois chapters); and removing the duplicates left by the merge: two GF(p^n) stacks in
`rustmath-finitefields`, two Montes/OM implementations (`rustmath-padics::montes` vs
`rustmath-rings::padics`), and the placeholder `rustmath-galois::stauduhar` vs the real descent.
Implement from the cited papers; test against LMFDB, PARI or GAP (GAP is installed).

## Working rules learned this session

- Commit and push often; the container can restart mid-task (it did several times). Detach long
  jobs with `nohup` so a restart does not kill them.
- Full `cargo test --workspace` takes about an hour; run it detached and log to the scratchpad.
- Crossref rate-limits after about 300 quick queries; `reading_list.py` caches and backs off.
- The user prefers no em-dashes and no stock phrases in prose.

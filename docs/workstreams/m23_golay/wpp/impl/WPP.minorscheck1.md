# WP-P: check of the fixes of the minor items of review 1

Date: 2026-11-16. Worktree `/home/user/RustMath-wpp`, branch `claude/m23-wpp`, base `main`
`28c4b69`, all changes uncommitted. Inputs: `WPP.review1.md` (items m1 to m8) and Section 12 of
`WPP.md` ("Minor items of review 1"). Scratch files of this check: `WPP.minors1/` (this
directory). I edited nothing in the worktree. I built `target/release` for the opt-in tests and
deleted it afterwards, and deleted `target/doc` and my probe binaries.

## Verdict

**Approve.** No blockers. All eight fixes are right and complete. Nothing regressed. No
undecided result became decided: every code change adds a refusal (`Err`) or a checked
operation, and the rest are docs. Two small optional items (n1, n2) and one note are listed at
the end. None of them changes a reported number.

There is no snapshot of the sources before the fixes (the files are untracked). I checked each
fix against the code as it stands, against review 1's description of the old behaviour, and by
rerunning review 1's own probe (`WPP.rev1/probe/probe2.rs`) on the fixed code. The files
modified after review 1 was written are exactly the ten the report names: `class_function.rs`,
`representation.rs`, `src/m23/{atlas,classes,cohomology,nielsen}.rs` and the four `m23_*` tests.
`three_point.rs`, `group.rs`, `bsgs.rs`, `mod.rs`, the example and the data are untouched.

## The eight fixes

| # | Fix | Check | Result |
|---|---|---|---|
| m1 | `F2Module` keeps `quo_span`; `matrix` returns `Err(NotInModule)` when the image of a quotient basis vector leaves `span(quo)` (`cohomology.rs:235`, `:299` to `:305`, doc `:296` to `:298`) | Together with the old check on the complement vectors this tests `g(sub) in sub` on a basis of `sub` and `g(quo) in quo` on a basis of `quo`, which is exactly "`g` acts on `sub / quo`". Probe: on `<e0+e1, e2+e3> / <e0+e1>`, `(1 3)(2 4)` is refused and `(1 2)` gives `Ok([1])`; 15 of 15 M23 elements are `Ok` on `G0`, `Q`, `P0`. Review 1's probe now gives `Err(NotInModule)` for `Q.matrix(t)` and `TripleCohomology::new(Q, (t, t, 1))` (before: `Ok`, dims 0 and 9), and for `Q` with a 23-cycle outside M23 (before: `CoboundaryNotCocycle`) | right, complete |
| m2 | New `F2Error::NotARealStructure`; `conjugation_fixed_dim` checks `c0^2 = 1`, `c0 g1 c0 = g1^-1`, `c0 g2 c0 = g3^-1` first (`cohomology.rs:51`, `:71`, `:440` to `:447`, doc `:436` to `:439`) | Same three conditions as `real_components` (`nielsen.rs:263` to `:265`; there `conjugate(x, c0) = c0^-1 x c0`, equal to `c0 x c0` once `c0^2 = 1` is known, so the two tests agree). Probe on the three real classes 2, 3, 5 of `(2A, 23A, 23B)` and all three modules: the real structure gives `G0` 2, 2, 3 and `P0` 4, 4, 5 (the `dim_W_real` and `dim_J2_real` of `group_facts.json`); **0 of the other 3,794 elements of 2A** are accepted (before: 2 of 399 gave `Ok(5)`, `Ok(6)`); `1`, `g1`, `g2` are refused | right, complete |
| m3 | `size == u64::MAX \|\| size > budget` (`classes.rs:727`, doc `:701` to `:703`) | Exact: no product `prod m! l^m` with `sum m l = 23` equals `2^64 - 1` (an odd number needs every `m <= 1`, so distinct odd `l`, and its factor 65,537 exceeds 23). Probe: identity with budget `u64::MAX` and `u64::MAX - 1` is refused at once; 23A with `u64::MAX` still gives `Ok(23)`; 2A reports `needed 52,022,476,800` | right, complete |
| m4 | Docs only: `Character::with_class_sizes` (`representation.rs:201` to `:203`), `try_inner_product` (`:288` to `:290`), `ClassFunction::with_class_sizes` (`class_function.rs:98` to `:101`) | All three state that the keys must be all the classes and that otherwise the value is a weighted mean. The S3 example is right: `(1 * 4 + 3 * 0) / 4 = 1`. The intra-doc links resolve (`cargo doc`, no warning) | right, complete |
| m5 | Chapter lines in `tests/m23_three_point.rs:5` (ch. 58, 91) and `tests/m23_nielsen.rs:3` to `:4` (ch. 58, 68) | The digests `Chapter_58_Permutation_Groups.md`, `Chapter_68_Cohomology_and_Extensions.md`, `Chapter_91_Characters_of_Finite_Groups.md` exist | right, complete |
| m6 | `unsigned_list::<T>` with `T::try_from` for `orders`, `centralizer_orders`, `class_sizes`; `usize::try_from` for power-map entries; `u32::try_from` for `d`; `checked_add`/`checked_mul` in `check_orthogonality` (`atlas.rs:247` to `:255`, `:300` to `:305`, `:323` to `:327`, `:354` to `:355`, `:458` to `:466`) | Probe: `orders` 2^32 and -1, `class_sizes` -1, 2^63 and 1.5, `d` = 2^32 + 23 and -7, power-map entries 0, 18, 2^64 - 1 and -1 are all `Err`, each naming the field. The remaining `as i64` casts on sizes and centralizers (`structure_constant`, the orthogonality sums) cannot wrap: `try_from(i64)` bounds the values by `i64::MAX`, and the sums run only after the size checks. The embedded table still parses and is orthogonal | right, complete (see n1 for a pre-existing panic outside the item) |
| m7 | `classes.rs:912` to `:913`: 23 bytes per element, at most 235 MB for all 17 classes, plus spare capacity | `10,200,960 x 23 = 234.6 MB`; the six rational classes hold 3,184,511 elements, 73.2 MB, as stated | right, complete |
| m8 | "Panics if `i >= len()`" on `real_structure`, `real_points_above_0`, `real_components` (`nielsen.rs:224` to `:225`, `:249`, `:260`) | These are the only callers of `triple(i)` besides `complex_conjugation`, which iterates over `0..len()` | right, complete |

The new tests are the ones the review asked for (`m23_nielsen.rs:356` to `:365` and `:370` to
`:378`, `m23_classes.rs:257` to `:264`, `m23_atlas.rs:137` to `:160`). Each asserts the new `Err`
where the old code gave `Ok` (m1, m2, m6), tried to build 23! vectors (m3), or panicked in debug
(m6, `3795 (2^63 - 1)`), so each fails on the old code (review 1's probe output confirms the old
`Ok` values). The existing assertions in those tests are still there (`mods[0].matrix(t)`,
`real_components(2, g1)`, the budget asserts with `SIFT_BUDGET`). Test counts per binary are as
in review 1. One public addition: the variant `F2Error::NotARealStructure`. No `match` on
`F2Error` outside `cohomology.rs` (grep).

## Commands run

| Command | Result |
|---|---|
| `cargo test -j 2 -p rustmath-groups --no-fail-fast` (debug, no rebuild needed) | 22.6 s wall. lib 558 pass; doc-tests 79 pass (4 ignored); `m23_atlas` 6 (1.71 s), `m23_classes` 5 + 2 ignored (1.84 s), `m23_nielsen` 4 + 3 ignored (0.29 s), `m23_three_point` 3 + 1 ignored (1.93 s); all other targets pass except `galois_narrowing` 3 of 4, which panic because the gitignored `data/transitive_24.jsonl` is absent (baseline, WPP-9) |
| `cargo test -j 2 --release -p rustmath-groups --test m23_three_point --test m23_classes --test m23_nielsen --test m23_atlas --example m23_class_vectors --no-run` | 1 min 46 s; `target/release` 259 MB, deleted afterwards |
| `full_57_rows` (release, `--exact`, one thread) | pass, 3.18 s, peak RSS 78.5 MB |
| `full_c_23a_23b` (release) | pass, 2.05 s, 78.0 MB |
| `full_census` (release) | pass, 10.43 s, 10.7 MB (process peak RSS) |
| `full_family_11` (release) | pass, 41.22 s, 10.6 MB |
| `m23_class_vectors --check rustmath-groups/tests/data/m23` (release) | "57 reproduced, 0 mismatches, 57 reference rows", 57 rows `Decided`, exit 0, 3.07 s, 75.1 MB |
| `env_` tests (debug) with `M23ISGALOIS` and `LELAY_M23` set | both pass |
| `env_` tests without the variables | both fail: "set M23ISGALOIS to a checkout of shaowuz/m23isgalois: NotPresent", "set LELAY_M23 to a checkout of research-reflexivity/frontier-math-m23-cover-investigation: NotPresent" |
| `cargo clippy -j 2 -p rustmath-groups --tests --examples --no-deps -- -A clippy::derived_hash_with_manual_eq` | no warning in `src/m23/`, the `m23_*` tests or the example. The two warnings in the changed files are on lines present at `HEAD` (`class_function.rs:207`, HEAD `:158`; `representation.rs:239`, HEAD `:109`) |
| `rustfmt --edition 2021 --check` on the four touched `src/m23` files and the four `m23_*` tests | clean |
| `rustfmt --check` on `representation.rs`, `class_function.rs`, and on their `HEAD` versions | 4 and 2 hunks in both, so none from the fixes |
| `cargo doc -j 2 -p rustmath-groups --no-deps` | no warning in the changed files; `target/doc` deleted |
| `probe.rs` (this directory; compiled with `rustc` against the debug rlib, which cargo reported fresh) | output in `probe.out`; results in the table above |
| review 1's `probe2.rs`, rerun on the fixed code | `review1_probe2.out`: every case review 1 reported now gives the refusal |
| grep for `unsafe` and em-dashes in the new and changed code | none |

The fixer did not rerun `full_57_rows`, `full_census`, `full_family_11` or the example after the
fixes (report Section 12). The fixes touch their path (the parse of the embedded table, and
`centralizer_order_by_sifting` through `centralizer_order`), so I ran all four: they pass with
the same numbers and times close to the earlier runs.

## Optional items (not blockers)

**n1. `AtlasTable::to_float_table` panics on a table that `parse` accepts.** `atlas.rs:563` and
`:567` call `expect`. With a class size 0, or sizes that do not sum to `|G|` (for example
`class_sizes[0] = 2^63 - 1`), `parse` is `Ok`, `check_orthogonality` returns `Err`, and
`to_float_table` panics (`ZeroSize`, "Class sizes sum to ..."; probe). This is older than the
fixes, and m6 did not ask for it. The only caller uses the embedded, checked table. Suggested
change: a `# Panics` section (when `check_orthogonality` fails), or return a `Result`; it can go
to the follow-up list.

**n2. The `Display` text of `F2Error::NotInModule` still reads "the vector is not in the
module".** After m1 the variant also means "a permutation does not preserve the module", as its
doc says (`cohomology.rs:47`). Review 1's probe now prints "Err(the vector is not in the
module)" for a transposition on `Q`. Suggested change: "the vector is not in the module, or the
permutation does not preserve it".

Note (no action needed): the field doc of `M23Error::BudgetExceeded::needed` (`group.rs:74`)
says it is a lower bound only for an orbit search. A saturated sifting size (`u64::MAX`) is also
only a lower bound. The doc of `centralizer_order_by_sifting` says so.

## Files of this check

`WPP.minors1/`: `test_debug_crate.log`, `build_release.log`, `release_full.log`,
`release_classes.log`, `example.log`, `env_set.log`, `env_unset.log`, `clippy.log`,
`rustfmt_m23.log`, `rustfmt_changed.log`, `head/` (the `HEAD` versions for the rustfmt
comparison), `doc.log`, `rss.py`, `probe.rs`, `probe.out`, `review1_probe2.rs`,
`review1_probe2.out`.

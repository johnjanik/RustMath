# WP-P implementation review 1

Date: 2026-11-16. Worktree `/home/user/RustMath-wpp`, branch `claude/m23-wpp`, base `main` `28c4b69`,
all changes uncommitted. Design: `../design/WPP.md` (Revision 1). Report: `WPP.md` (this directory).
Scratch files of this review: `WPP.rev1/` (scripts, GAP inputs and outputs, logs). I edited nothing
in the worktree. I rebuilt `target/release` for the opt-in tests and deleted it afterwards.

## Verdict

**Approve.** No blockers. Eight minor items (m1 to m8), none of which changes a reported number.

The design is implemented as written, with the deviations listed and justified in report Section 8.
The 57 rows, the six `(C, 23A, 23B)` classes and the three dictionaries are reproduced. I confirmed
the class identification, the table's column order and a sample of the rows with GAP, by methods
that differ from RustMath's.

## Commands run

| Command | Result |
|---|---|
| `cargo test -j 2 -p rustmath-groups --tests --examples --no-run` (debug) | builds; no warning in `src/m23/`, the new tests, the example, or the changed files (lib: 52 warnings, as at base) |
| the four debug test binaries, `--test-threads 1` | `m23_classes` 5 pass (2 ignored) 1.96 s; `m23_atlas` 6 pass 1.80 s; `m23_three_point` 3 pass (1 ignored) 1.71 s; `m23_nielsen` 4 pass (3 ignored) 0.37 s. Total 5.8 s. Peak RSS 10.4 MB per binary (`posix_spawn` and `wait4`) |
| `cargo test -j 2 -p rustmath-groups --no-fail-fast` | 14.8 s wall. lib 558 pass, doc-tests 79 pass (4 ignored), all new targets pass. `galois_narrowing` 3 of 4 fail: `data/transitive_24.jsonl` is gitignored and absent in this worktree (the panic message says so). The file and its test are untouched, so this is the base state (report follow-up WPP-9) |
| release build of the three test targets and the example | 1 min 25 s (`-j 2`), `target/release` 256 MB, deleted afterwards |
| `full_census` alone (`--exact`) | pass, 11.63 s, VmHWM 3 MB |
| `full_family_11` alone | pass, 38.51 s, VmHWM 3 MB |
| `full_57_rows` alone | pass, 3.29 s (lists 0.74 s, 57 rows 2.35 s, slowest `(8A, 11A, 11B)` 0.40 s, `a = c` 0.19 s), VmHWM 78 MB. The `a = c` counts are 98, 30, 12, 5, 6 and 0 elsewhere, none generating |
| `full_c_23a_23b` alone | pass, 1.83 s, VmHWM 78 MB; sizes 7, 134, 660, 1422, 1632, 2272; real 3, 16, 42, 32, 72, 0 |
| `m23_class_vectors --check rustmath-groups/tests/data/m23` (release) | "57 reproduced, 0 mismatches, 57 reference rows", exit 0, 3.07 s, 75 MB |
| the example against a copy of the reference with one `real` changed | the row prints `MISMATCH`, "56 reproduced, 1 mismatches", exit 1 |
| `env_` tests with `M23ISGALOIS` and `LELAY_M23` set | both pass (18.7 ms and 17.7 ms, debug): `[5, 4, 0, 2, 1, 3, 6]` and `[0, 1, 2, 4, 6, 5, 3]` |
| `env_` tests without the variables | both panic: "set M23ISGALOIS to a checkout of shaowuz/m23isgalois" and "set LELAY_M23 to a checkout of research-reflexivity/frontier-math-m23-cover-investigation" |
| `env_` tests with a wrong path | both panic and name the missing file |
| `gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null` | byte-identical to the committed JSON (3,471 bytes, sha256 `f9503e80...0f5196eb`) |
| `python3 -I extract_janik_table.py /home/user/johnjanik/m23 <scratch>` | sha256 `3db9d440...2dac`, equal to the vendored file |
| sha256 of the four vendored owner files against `johnjanik/m23` at `2d11038` | all equal to the README |
| `cargo clippy -p rustmath-groups --tests --examples --no-deps -- -A clippy::derived_hash_with_manual_eq` | no warning in new code. The two warnings in the changed files (`class_function.rs:202`, `representation.rs:235`) are on unchanged lines. Without `-A`, the deny error at `indexed_free_group.rs:181` is in an untouched file (report WPP-8) |
| `rustfmt --edition 2021 --check` on the 13 new `.rs` files | clean |

## Independent checks

Scripts and outputs are in `WPP.rev1/probe/` and `WPP.rev1/scripts/`. The Rust probes were compiled
with `rustc` against the existing debug rlibs (read only), so they call the code under review.

1. **Class identification against GAP on random elements.** `probe.rs` draws 20,000 uniform elements
   of M23 from the stabilizer chain. A third are then multiplied by `g2`, and a third are replaced
   by the square of their inverse. Each gets `class_of_checked`, asserted equal to
   `class_of_member`. All 17 labels occur (counts 1 to 1,901). In GAP (`check_classes.g`),
   `G = Group(g1, g2)` and its own `ConjugacyClasses`: RustMath's 17 representatives lie in 17
   distinct classes, with GAP's sizes. Every sample has the cycle type of its label's
   representative and `IsConjugate(G, p, rep[label])` is true. **0 of 20,000 misclassified**
   (11.7 s in GAP).
2. **The table's columns on RustMath's classes.** GAP computes the character table of the actual
   group (`CharacterTable(G)`, Dixon-Schneider). Read on RustMath's classes in ATLAS order, its set
   of irreducible rows equals the set of rows of `CharacterTable("M23")`. So the stored column
   order is correct on RustMath's classes literally, not only up to a table automorphism. The 57
   structure constants computed from the group's own table equal the owner's file (0 mismatches).
3. **Twelve rows by a different method** (`nielsen_rows.g`). Classes are GAP's conjugacy tests
   against RustMath's representatives. Generation is `Size(Group(a, b)) = |G|`, not the ATLAS
   criterion. The Nielsen classes are `Centralizer(G, b)`-orbits, checked free. A class counts as
   real when `RepresentativeAction(G, [a, b], [a^-1, a b], OnTuples)` (conjugate pairs) or
   `[a^-1, b^-1]` (rational) is not `fail`. That is a tuple-conjugacy search, not RustMath's count
   over involutions. For the rational rows I fixed `b` in a different class from RustMath's
   `roles`. All twelve equal the owner's file:

   | Row | `b` | survivors | generating | size | real |
   |---|---|---:|---:|---:|---:|
   | (2A, 23A, 23B) | 23A | 161 | 161 | 7 | 3 |
   | (2A, 14A, 14B) | 14A | 301 | 238 | 17 | 3 |
   | (2A, 11A, 11B) | 11A | 341 | 242 | 22 | 4 |
   | (2A, 15A, 15B) | 15A | 405 | 360 | 24 | 2 |
   | (2A, 7A, 7B) | 7A | 301 | 0 | 0 | 0 |
   | (3A, 23A, 23B) | 23A | 3082 | 3082 | 134 | 16 |
   | (3A, 11A, 11B) | 11A | 5126 | 3784 | 344 | 18 |
   | (2A, 8A, 8A) | 8A | 491 | 224 | 28 | 0 |
   | (3A, 5A, 6A) | 5A (RustMath: 6A) | 4315 = 863/3 x 15 | 1050 | 70 | 4 |
   | (3A, 4A, 8A) | 4A (RustMath: 8A) | 7056 = 441/2 x 32 | 1728 | 54 | 0 |
   | (4A, 4A, 5A) | 5A, `a` in 4A (RustMath: `b` in 4A) | 13635 = 909 x 15 | 1560 | 104 | 2 |
   | (3A, 6A, 6A) | 6A | 5271 | 1440 | 120 | 16 |

   This also confirms the identity `#survivors = n |C(b)|` for roles that RustMath does not use.
4. **Nothing of HJLPPZ or Le Lay is copied.** `nocopy.py` takes the 12 permutations of the HJLPPZ
   file other than the published `g1`, `g2`, `g3`, and the 11 cycle-notation permutations of Le
   Lay's certificate. It searches for each, in seven encodings (1-based and 0-based cycles, owner
   style, JSON, image list), through `src/m23`, `tests` (including `tests/data`), `examples` and
   `data/m23`: 0 hits. A positive control (a permutation taken from the vendored
   `passports_C_23A_23B.json`) is found. The HJLPPZ test reads the file at run time only, and its
   one hard-coded comparison is with the published `g3`.
5. **Generators.** `G1_CYCLES`, `G2_CYCLE`, `G3_CYCLE` in `group.rs:43` equal the owner's `m23.py`
   (HJLPPZ 3.1), and `g1 g2 g3 = 1` holds.

## Against the brief and the review checklist

| Item | Finding |
|---|---|
| Design implemented | Yes. The module tree, the API and the tests are as in design Sections 8 and 10. Deviations (report Section 8) are justified, and none weakens a check. All line numbers the report hands to the tracker step are right (I checked 62 of them) |
| Table right | Yes: Section "Independent checks" 2 and 3, and the example's 57 `reproduced` |
| Class identification exact | Yes. The argument in the `classes.rs` header is sound. `full_census` confirms step 3 (17 sizes from the labels, sign = trace on 887,040 elements, 17 centralizers by scan). GAP finds 0 misclassified of 20,000 |
| Character table data and class matching | The data are reproducible byte for byte. `check_against` checks orders, centralizers computed by RustMath, the permutation character and all seven power maps. The group's own table confirms the column order exactly |
| No undecided result reported as decided | `CountStatus::Decided` is the only success value. If all three classes have order in `{1, 11, 23}` the count is `Err(GenerationCriterionNotApplicable)`. The identity, the divisibility and the parity are `Err` on failure. The two conjugation criteria and the ATLAS theorem are named in `trusted` on every count, and the example prints them |
| Unbounded loops or allocations | None without a documented cost, except the saturated-budget edge case m3 and the undocumented upper bound m7 |
| Tests can fail on wrong code, with external references | Yes. Rows are compared with the owner's file, sizes and centralizers with GAP's constants, the dictionaries with `group_facts.json`. A wrong generation test, real count, trace or 11 test would break `eighteen_small_rows`, `split_invariants_separate` or the identity. The tampered-reference run shows the example fails too |
| HJLPPZ test fails loudly; no data copied | Yes (both checked) |
| Cost claims | Hold, within the machine's load: `full_census` 11.6 s (design 8.9 s, report 10.2 to 12.3 s), the others within 0.5 s of the report, memory as designed |
| No `unsafe` | None in the new or changed code |
| Sources cited | Every new `src/m23` file cites its MAGMA chapter digest and its papers. Two test files do not (m5) |
| T-215 | Both inner products are fixed in place and tested (S3 now gives 1; the M23 table is orthonormal). No existing assertion is removed or weakened (`git diff` removes no `assert` line). No crate outside `rustmath-groups` uses these types (grep) |
| Do-not-call list | Not used: no `conjugacy_classes.rs`, `block_systems`, `SteinerSystem::new` or `rustmath-homology` |
| One crate | Yes (`rustmath-groups`, new dependency `rustmath-coding`, dev-dependency `rustmath-finitefields`) |

## Minor items

**m1. The quotient module accepts permutations that do not act on it.**
`cohomology.rs:290` (`F2Module::matrix`) only checks that the image of each complement vector lies
in `span(sub)`. For `Q = P0 / G0` it does not check that `g` maps `G0` to itself. Probe:
`mods[1].matrix(transposition)` is `Ok` with 11 rows (`mods[0]` gives `Err(NotInModule)`), and
`TripleCohomology::new(&mods[1], &(t, t, 1))` is `Ok` with `dim_parabolic 0`, `dim_orbifold 9`.
The docs of `matrix` and `TripleCohomology::new` (`:288`, `:322`) promise `Err(NotInModule)` when
an entry does not preserve the module. The WP-P callers pass only M23 triples, so no reported
number is affected. Required change: keep the quotient span in `F2Module` and have `matrix` return
`Err(NotInModule)` when `mask_image(q, g)` leaves it for a quotient basis vector `q`. Add a test
with a transposition on `Q`.

**m2. `conjugation_fixed_dim` does not check that `c0` is a real structure.**
`cohomology.rs:425`. Probe: on class 5 of `(2A, 23A, 23B)` (module `P0`), the identity gives
`Ok(8)`, and 2 of 399 other involutions of M23 give `Ok(5)` and `Ok(6)`. `real_components`
(`nielsen.rs:262`) refuses a wrong `c0`. Required change: check `c0 g1 c0 = g1^-1`,
`c0 g2 c0 = g3^-1` and `c0^2 = 1` as `real_components` does (new `F2Error` variant), or state the
precondition in the doc. Add a test with `c0 = 1`.

**m3. A budget of `u64::MAX` defeats the saturating size.**
`classes.rs:725`: `if size > budget`. For the identity the size saturates to `u64::MAX`, so with
`budget = u64::MAX` the check passes and `perms(23)` tries to build 23! vectors (out of memory).
Required change: treat a saturated size as above every budget (for example
`size == u64::MAX || size > budget`), and document that. Add a test with `u64::MAX`.

**m4. A character with class sizes on only some classes gives a weighted mean, not an inner
product.** `representation.rs:283` divides by `sum |C|` over the keys. For the S3 standard
character on `{e: 1, (12): 3}` only, `<chi, chi> = 4 / 4 = 1` and `is_irreducible` is true.
`ClassFunction::with_class_sizes` (`class_function.rs:97`) sets the group order to the same partial
sum, so `inner_product` has the same reading. The design table defines this formula, so this is a
documentation gap, not a regression. Required change: state in `with_class_sizes` (both types) and
`try_inner_product` that the keys must be all classes of the group, or else the result is only a
weighted mean. The old size-less reading already carries the analogous sentence.

**m5. Two new test files do not cite a MAGMA chapter.**
`tests/m23_three_point.rs:1` (chapters 58, 91) and `tests/m23_nielsen.rs:1` (chapters 58, 68)
cite the HANDOFF and the owner's files only. CLAUDE.md asks for the chapter in each new file's
header. Required change: add the chapter line, as `m23_classes.rs` and `m23_atlas.rs` do.

**m6. `AtlasTable::parse` wraps out-of-range integers.**
`atlas.rs:291-311` and `:359` use `as u32` / `as u64` on `i64` / `u64`. A centralizer of `-1`
parses as `18446744073709551615`. `check_orthogonality` then returns `Err`, but a large class size
could overflow `class_size * centralizer_order` (`:466`) and panic in debug. Required change: use
`u64::try_from` / `u32::try_from` and return `Err` naming the field.

**m7. The memory bound of `ClassLists::by_enumeration` is documented for one choice only.**
`classes.rs:909` gives 73 MB for the six rational classes. Any choice is public, and all 17 classes
store `|G| x 23` bytes (235 MB). Required change: one sentence with the bound (23 bytes per stored
element, at most 235 MB).

**m8. Undocumented panics on an index out of range.**
`nielsen.rs:225` (`real_structure`), `:249` (`real_points_above_0`) and `:260`
(`real_components`) call `triple(i)`, which panics for `i >= len()`. Only `triple` documents this.
Required change: document the panic, or return an `Err`.

## Notes (no action needed)

- The report's debug peak RSS (12.7 MB) was measured with a fork-based wrapper, which counts the
  forked Python's pages. With `posix_spawn` and `wait4` the debug binaries peak at 10.4 MB. The
  claim is an upper bound, so it stands.
- `census` says "about 9 s" in release. I measured 11.6 s on the shared machine. That fits
  "about".
- The 11A/11B, 7A/7B, 14A/14B and 15A/15B names are conventions (the HANDOFF says so).
  Independent check 2 shows they coincide with the stored ATLAS columns exactly. No test pins
  `class_of_member(h11) == 11A` directly, but `h11` is private and the convention changes no
  number.
- The `classes.rs` header attributes the two-cyclic-systems lemma to Le Lay, Section 5.4. That is
  the attribution of HANDOFF Section 7. I could not find it by searching his `paper/main.tex`
  for "Steiner", which suggests a different section number or file. The later docs step should
  confirm the reference.

## Files of this review

`WPP.rev1/`: `build_debug.log`, `build_release.log`, `test_debug_m23.log`, `test_debug_crate.log`,
`test_release_full.log`, `example.log`, `example_tampered.log`, `tampered/`, `clippy.log`,
`rustfmt.log`, `atlas_regen.json`, `janik/janik_table.json`, `nocopy.log`, `scripts/` (`rss.py`,
`rss2.py`, `nocopy.py`, `extract_janik_table.py` copy), `probe/` (`probe.rs`, `probe2.rs`,
`probe2.out`, `samples.g`, `vectors.g`, `check_classes.g`, `check_classes.out`, `nielsen_rows.g`,
`nielsen_rows.out`, `fake_gap.g`). I deleted the probe binaries and `target/release`.

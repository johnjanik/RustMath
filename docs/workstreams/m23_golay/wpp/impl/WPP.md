# WP-P implementation report: the three-point class vectors of M23

Date: 2026-11-16. Worktree `/home/user/RustMath-wpp`, local branch `claude/m23-wpp`, base `main`
`28c4b69`. Nothing is committed (as instructed). Design: `../design/WPP.md` (Revision 1 and its
body), approved by `../design/WPP.review2.md`. Logs of every run named here:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/wpp/impl_logs/`
(the final runs in `final/`).

## 1. Summary

Everything mandatory and both optional parts are done, in one crate (`rustmath-groups`):

- M23 on the published generators, Schreier-Sims order 10,200,960, all elements from the
  stabilizer chain (new public `StabilizerChain::for_each_element`).
- The 17 classes with the split pairs identified exactly (Golay trace, heptad-position family,
  cyclic Steiner sign); class sizes and centralizer orders equal GAP's (debug and release).
- The ATLAS table of M23 as data (GAP script, byte-for-byte reproducible), exact arithmetic,
  exact row and column orthogonality, class order matched by orders, sizes, centralizers, all
  power maps and the permutation character; the 57 structure constants equal the owner's file.
- All 57 rows of `data/m23_three_point_nielsen.json` reproduced exactly (n, size, real, genus23),
  with `#survivors = n |C(b)|` asserted on every row and `|C(b)|` computed by RustMath.
- 4.2 step 7 and 4.3: the six `(C, 23A, 23B)` Nielsen classes (6,127 triples), conjugation, real
  structures, ovals, `dim W = dim W^dual = g`, `dim Jac[2] = 2g` on every triple,
  `dim Jac(X)(R)[2] = g + r - 1` on every real cover, all equal to `passports_C_23A_23B.json`
  entry by entry, and the `group_facts.json` `nielsen` fields.
- The three dictionaries hold and are bijective: HJLPPZ rows 1..7 -> `[5, 4, 0, 2, 1, 3, 6]`,
  Janik rows 0..6 -> `[3, 5, 1, 6, 2, 4, 0]`, Le Lay `x_1..x_7` -> `[0, 1, 2, 4, 6, 5, 3]`.
- T-215: `Character::inner_product` and `ClassFunction::inner_product` fixed in place (weighted
  by class sizes), with the edge cases refused (review 2, n3 included).

Fast suite: 18 new integration tests and 10 new unit tests, 5.5 s in debug on one thread, peak
RSS 12.7 MB per test binary. Opt-in release tests: `full_census` 12.3 s, `full_family_11` 38.4 s
(both 3 MB), `full_57_rows` 3.2 s, `full_c_23a_23b` 1.9 s (both 78 MB).

## 2. What changed, per item

All paths are under `rustmath-groups/`. Line numbers are those of the working tree now.

| Item | Change |
|---|---|
| Element enumeration (design 8.1) | `src/bsgs.rs:318` `num_levels`, `:324` `orbit`, `:330` `transversal_element`, `:354` `for_each_element` (odometer over the transversals, one buffer per level, only the levels below the changed one recomputed). Module docs gain the enumeration bullet. Tests `:624` `accessors_and_level_out_of_range`, `:646` `for_each_element_s4_a5_m11` (S4 24, A5 60, M11 7,920 with GAP's `MathieuGroup(11)` generators; count, distinct, member), `:661` `for_each_element_trivial_group`, and a doctest. |
| T-215 `Character` (design 4.4, review 2 n3) | `src/representation.rs:95` `CharacterError` (`SizesOnOneSide`, `SizeMismatch`, `KeyMismatch`, `SizeKeysDiffer`, `ZeroSize`, `MissingValue`, `NoValues`); `Character` gains `class_sizes`; `:200` `with_class_sizes`; `:227` `class_sizes`; `:283` `try_inner_product` (the table of 4.4, plus `KeyMismatch` for different key sets without sizes, n3); `:342` `inner_product` keeps its signature and panics with the error text; `is_irreducible` is false and `norm_squared` is 0 for a character with no values (no panic). `CharacterTable::cyclic_group` and `symmetric_group_s3` attach their sizes; `:457` `CharacterTable::class_sizes`; `:465` `CharacterTable::inner_product` (table sizes, `MissingValue`, `SizeMismatch`). Tests `:943`, `:958`, `:998`, `:1039`, `:1068` (S3 from GAP: the tracker's failure case now gives 1; orthonormal rows; the edge cases; `#[should_panic]`). |
| T-215 `ClassFunction` (design 4.4, n3) | `src/class_function.rs`: `class_sizes` field; `:97` `with_class_sizes` (group order = sum of sizes; refuses other keys or size 0); `:119` `class_sizes`; `:259` `inner_product` weights by the sizes and returns `None` unless the order is known (and equal on both sides), the class sets are equal (n3), and the sizes are known and sum to the order, or the number of classes equals the order (all sizes 1, forced); binary operations keep sizes only when both sides agree; `from_character`/`to_character` carry sizes. Tests `:625`, `:643`. The two old tests (`test_inner_product`, `test_norm`) pass unchanged through the forced case. |
| Dependencies | `Cargo.toml`: `rustmath-coding` (cyclic Golay codes), dev-dependency `rustmath-finitefields` (`PrimeField` cross-check). `cargo tree` of both does not reach `rustmath-groups`. `Cargo.lock` gains the two edges. `src/lib.rs`: `pub mod m23;` and `CharacterError` re-exported. |
| M23 (8.2) | `src/m23/group.rs`: `Perm23` (`transitive23::Perm`), `rmul` `:108`, `inverse`, `conjugate`, `cycles`, `order`, `power`, `mask_image`, `fixed_mask`, `generates_transitive` `:215`, `from_cycles_1based` `:234`, `from_image_list`, `to_image_list`, `M23Error` `:64`; `M23::published` `:330` (checks `g1 g2 g3 = 1` with the published `g3`, order 10,200,960, 253 heptads forming an S(4,7,23) by `is_steiner_system`, `dim G0 = 11`), accessors, `contains`, `for_each_element` `:435` (wraps the chain's enumeration, one reused `[u8; 23]` buffer), `heptads`, `even_golay_basis`. |
| Classes (8.3, 3.1 to 3.4) | `src/m23/classes.rs`: module docs with the exactness argument and the naming (with the `tr(x^2) = tr(x)` remark of review 2, n5); `M23Class` `:53` (ATLAS order, names, orders, cycle types, inverse, rational, partner); `signature` `:242`; `ClassIdentifier::new` `:364` (deterministic walk for type representatives, conventions of 3.3, every separation checked, `Err(Construction)` otherwise); `golay_trace` `:484`; `heptad_family_11` `:534` and `cyclic_steiner_sign` `:624` (the only sources of `InvariantFailed`); `class_of_member` `:638` (fast 11 test; unspecified on non-members, documented); `class_of_checked` `:658`; `centralizer_order` `:675` (sifting within `10^6`, else orbit within `10^5`, cached in a `OnceLock`); free `centralizer_order_by_sifting` `:704` (saturating `prod m! l^m`) and `conjugacy_class_by_orbit` `:807`; `Census` and `census` `:851`; `ClassLists` `:904` (`by_enumeration`, `by_orbit`, `get`). |
| ATLAS table (8.4, 4.1 to 4.3) | `data/m23/m23_atlas_table.g` (our GAP script; command and versions in its header), `data/m23/m23_atlas_table.json` (regenerated here: 3,471 bytes, sha256 `f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb`, equal to the design's), `data/m23/README.md`. `src/m23/atlas.rs`: `AtlasNumber` `:48` (16 `Rational` coordinates, `from_atlas`, `conj`, `rational_part`, `scale`, `to_complex_f64`, `+ - *`), `AtlasTable::m23` `:263` (`include_str!`, parsed once), `parse` `:271` (refuses a wrong class order, wrong lengths, a value outside the four fields), accessors, `structure_constant` `:440` (doc states the relation to MAGMA's `a_ijk`, review 2 n4), `check_orthogonality` `:461` (exact rows and columns), `check_against` `:512` (orders, centralizers computed by RustMath, permutation character, all 7 power maps, review 2 n5), `to_float_table` `:541`. |
| Three-point counting (8.5, 5.1 to 5.4) | `src/m23/three_point.rs`: header with the counting, the generation theorem, and the derivation of the rational real criterion (design 5.3, m4). `ClassVector::validate` `:107` (review 2, n1); `enumerate_class_vectors` `:203` (owner's `m23_passports.g`); `genus23` `:253`; `CountStatus::Decided`, `TrustedResult`; `StructureCount` and `structure_count` `:437` (review 2, n2); `ClassVectorCount`; `count_class_vector` `:450` (identity `Err(CountIdentityFailed)`, `GenerationCriterionNotApplicable`, divisibility and parity `Err(NotDivisible)`, `u = 1` enumerated and counted in `a_equals_c_*`). |
| Nielsen classes and ovals (8.6, 6.1, 6.2) | `src/m23/nielsen.rs`: `NielsenClass23::new` `:131` (candidates only from `ClassLists`; free action checked), `index_of` `:202`, `complex_conjugation` `:215`, `real_structure` `:225`, `real_points_above_0` `:249`, `real_components` `:260` (degree-2 check and `c0` check, `Err(OvalGraph)`). |
| Parabolic cohomology (6.3, m6) | `src/m23/cohomology.rs`: `F2Error` `:39`, `F2Span` `:83` (`add` refuses a 129th vector), `F2Module` `:225` (`subquotient`, `new`, `coords`, `matrix`; `3 dim > 128` refused), `standard_modules` `:299` (`G0`, `P0/G0`, `P0`), `TripleCohomology` `:310` (`new` `:326`, `dim_parabolic`, `dim_orbifold`, `conjugation_fixed_dim` `:425`, involution checked). |
| Module root | `src/m23/mod.rs` (map of the modules, conventions, MAGMA chapters, "no `unsafe`"). |
| Example (8.7) | `examples/m23_class_vectors.rs` (prints the 57 rows with genus, n, survivors, `|C(b)|`, size, real, status, time, trusted results; `--check <dir>` adds `reproduced`/`MISMATCH`, exit 1 on a mismatch). |
| Vendored data (11) | `tests/data/m23/`: the four owner files (sha256 equal to the design), `janik_table.json` regenerated by `extract_janik_table.py` (sha256 `3db9d440...2dac`, equal), `README.md` (sources, licences, sha256, commands, the two run-time-only files with their sha256, rechecked here). Nothing of HJLPPZ or Le Lay is vendored. |
| Readers (m10) | `tests/m23_common/mod.rs` (`#![allow(dead_code)]`): JSON readers with path in every error, `owner_cycles`, `read_janik_table`, `parse_magma_perms`, `gap_assignment`, `gap_perms`, `vm_hwm_mb`. Used as `mod m23_common;` by the tests and through `#[path]` by the example. |

## 3. Tests and reference sources

| Test (file) | Checks | Reference |
|---|---|---|
| `m23_published_order_and_base` (`m23_classes.rs`) | order, base orbit lengths `[23, 22, 21, 20, 3, 16]`, `g1 g2 g3 = 1`, an odd permutation is not a member | owner `m23.py` `ORDER`; HANDOFF 4.2 step 1; GAP `Size` |
| `steiner_and_even_golay_code` | 253 heptads, exact S(4,7,23), `dim G0 = 11`, weights 1 + 506 + 1288 + 253, generators preserve the system | Witt 1938; MacWilliams, Sloane ch. 20 |
| `split_invariants_separate` | signs `g2 -> -1`, `g3 -> +1`; conventions of 3.3; every representative in M23 and in its class; inverses in the inverse class; cycle types; trace separates 7, 14, 15, 23; family separates 11 and is preserved by `g -> g^k` exactly for the squares mod 11 | `group_facts.json` `signs`; HANDOFF 4.2 step 3 |
| `centralizers_match_gap` | `centralizer_order` for all 17 (orbits for 1A, 2A, 3A); orbit lengths 1, 3,795, 56,672 | GAP `SizesCentralizers`, `SizesConjugacyClasses` (constants in the file) |
| `degenerate_class_cases` | non-member involution (2A by `class_of_member`, `Err` by `class_of_checked`), 22-cycle, transposition, 23-cycle outside M23 (`InvariantFailed`), budgets (23! saturates to `u64::MAX`), malformed cycles, names | behaviour (design 9) |
| `full_census` (ignored) | all 10,200,960 labels give GAP's 17 sizes; sign = trace on all 887,040 elements of order 23; 17 centralizers by scan | GAP |
| `full_family_11` (ignored) | fast 11 test = family test on all 1,854,720 elements of order 11 (927,360 in 11A) | redundant by 3.2 |
| `atlas_file_is_the_gap_output` (`m23_atlas.rs`) | source string, GAP centralizers, orders, rational classes, power map -1, the 17 degrees | GAP; ATLAS p. 71 |
| `atlas_orthogonality_and_57_constants` | exact row and column orthogonality; the 57 `n` equal the owner's GAP file, symmetric; `(2A,23A,23B)`: n = 7, `a = 161` | owner `m23_three_point_passports.json` |
| `class_data_match_gap` | `check_against`: orders, centralizers, permutation character, 7 power maps | GAP table |
| `atlas_numbers`, `atlas_parse_rejects_bad_files` | `b7^2 + b7 + 2 = 0`, conjugation, f64 value; parse refusals | algebra |
| `t215_m23_float_table_is_orthonormal` | the 17 irreducibles orthonormal to `1e-9` by both inner products; column orthogonality | GAP table |
| `class_vector_enumeration_is_the_57` (`m23_three_point.rs`) | the set of 57 vectors equals both owner files; 27 rational; genus and n per row | owner files |
| `eighteen_small_rows` | the 18 rows with `a` in 2A or 3A, every row check (identity, `|C(b)|` = GAP, divisibility, parity, `a = c` adds nothing, trusted results); named rows `(2A,23A,23B)` 7/3, `(2A,14A,14B)` 17/3, `(2A,7A,7B)` 0/0 with 301 survivors and n = 43/2, `(3A,5A,6A)` 70/4 | owner `m23_three_point_nielsen.json`; ten rows also recounted in GAP by review 2 |
| `degenerate_cases` | n1 shapes refused; `(1A,1A,1A)`: `GenerationCriterionNotApplicable`, `structure_count` gives 1 survivor, n = 1/|G|; `(2A,2A,8A)`: n = 0, 0 survivors, negative genus; missing lists | behaviour |
| `full_57_rows` (ignored) | all 57 rows, every row check, the `a = c` case on the 21 rational multisets with 2A | owner file |
| `nielsen_2a_23a_23b_and_janik` (`m23_nielsen.rs`) | 7 classes, 161 candidates, conjugation `[1,0,2,3,6,5,4]`, dims 4/4/8, orbifold 7, real data of the three fixed classes, Janik `[3,5,1,6,2,4,0]`, `representatives_g1` and `real` of `"2A"` | `group_facts.json`; `janik_table.json`; `passports_C_23A_23B.json` |
| `cohomology_cross_check_matrix_gf2` | dims 4, 4, 8 of class 5 by `Matrix<PrimeField>` ranks and kernels over GF(2) (independent route) | same |
| `gap_reader_rejects` | the readers refuse malformed input | behaviour |
| `degenerate_nielsen_cases` | `NotARationalClass`, `ClassListMissing`, `ProductNotOne`, `SecondEntryNotIn23A`, `NotInClass`, `OvalGraph`, `CapacityExceeded`, `NotInModule`, F2 `ProductNotOne`; `(3A,23A,23B)` has 134 classes | behaviour |
| `full_c_23a_23b` (ignored) | six classes entry by entry: sizes, `representatives_g1`, `real_classes`, `real`; dims on all 6,127 triples; `g + r - 1`; size = n | `passports_C_23A_23B.json` |
| `env_hjlppz_dictionary`, `env_lelay_dictionary` (ignored) | the two run-time dictionaries; loud failure without the variable | HJLPPZ file; Le Lay's certificate; `group_facts.json` |
| unit tests in `bsgs.rs`, `representation.rs`, `class_function.rs` | section 2 | GAP (S3, M11) |

No existing assertion was changed or weakened.

## 4. The 57 rows (final release run of the example with `--check`)

Command: `cargo run --release -p rustmath-groups --example m23_class_vectors -- --check rustmath-groups/tests/data/m23`.
Every row: status `Decided`; `#survivors = n |C(b)|` held; `|C(b)|` computed by RustMath and
equal to GAP's; reference column against `m23_three_point_nielsen.json`. "trusted" lists the
results each count rests on besides exact enumeration (`TrustedResult`); the rational `real`
values rest on the derivation of `three_point.rs` (classical sources not quoted, WPP-7).
Output: "57 reproduced, 0 mismatches, 57 reference rows".

| # | type | class vector | genus | n | survivors | \|C(b)\| | size | real | status | trusted | reference |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | conjugate_pair | (2A, 7A, 7B) | 0 | 43/2 | 301 | 14 | 0 | 0 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 2 | conjugate_pair | (2A, 23A, 23B) | 4 | 7 | 161 | 23 | 7 | 3 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 3 | conjugate_pair | (2A, 14A, 14B) | 2 | 43/2 | 301 | 14 | 17 | 3 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 4 | conjugate_pair | (2A, 11A, 11B) | 2 | 31 | 341 | 11 | 22 | 4 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 5 | conjugate_pair | (2A, 15A, 15B) | 2 | 27 | 405 | 15 | 24 | 2 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 6 | conjugate_pair | (3A, 7A, 7B) | 2 | 349 | 4886 | 14 | 100 | 8 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 7 | conjugate_pair | (3A, 23A, 23B) | 6 | 134 | 3082 | 23 | 134 | 16 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 8 | conjugate_pair | (3A, 15A, 15B) | 4 | 830/3 | 4150 | 15 | 264 | 20 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 9 | conjugate_pair | (3A, 14A, 14B) | 4 | 349 | 4886 | 14 | 303 | 15 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 10 | conjugate_pair | (3A, 11A, 11B) | 4 | 466 | 5126 | 11 | 344 | 18 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 11 | conjugate_pair | (4A, 23A, 23B) | 7 | 660 | 15180 | 23 | 660 | 42 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 12 | conjugate_pair | (4A, 7A, 7B) | 3 | 1951 | 27314 | 14 | 995 | 21 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 13 | conjugate_pair | (5A, 23A, 23B) | 8 | 1422 | 32706 | 23 | 1422 | 32 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 14 | conjugate_pair | (4A, 15A, 15B) | 5 | 1496 | 22440 | 15 | 1480 | 54 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 15 | conjugate_pair | (6A, 23A, 23B) | 8 | 1632 | 37536 | 23 | 1632 | 72 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 16 | conjugate_pair | (4A, 14A, 14B) | 5 | 1951 | 27314 | 14 | 1879 | 55 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 17 | conjugate_pair | (5A, 7A, 7B) | 4 | 3756 | 52584 | 14 | 1994 | 22 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 18 | conjugate_pair | (8A, 23A, 23B) | 9 | 2272 | 52256 | 23 | 2272 | 0 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 19 | conjugate_pair | (4A, 11A, 11B) | 5 | 2772 | 30492 | 11 | 2390 | 58 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 20 | conjugate_pair | (5A, 15A, 15B) | 6 | 16982/5 | 50946 | 15 | 3381 | 43 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 21 | conjugate_pair | (6A, 7A, 7B) | 4 | 4371 | 61194 | 14 | 3470 | 40 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 22 | conjugate_pair | (5A, 14A, 14B) | 6 | 3756 | 52584 | 14 | 3627 | 37 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 23 | conjugate_pair | (6A, 15A, 15B) | 6 | 11668/3 | 58340 | 15 | 3859 | 103 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 24 | conjugate_pair | (6A, 14A, 14B) | 6 | 4371 | 61194 | 14 | 4328 | 102 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 25 | conjugate_pair | (8A, 7A, 7B) | 5 | 5908 | 82712 | 14 | 4848 | 0 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 26 | conjugate_pair | (5A, 11A, 11B) | 6 | 5658 | 62238 | 11 | 4874 | 54 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 27 | conjugate_pair | (8A, 15A, 15B) | 7 | 5424 | 81360 | 15 | 5424 | 0 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 28 | conjugate_pair | (8A, 14A, 14B) | 7 | 5908 | 82712 | 14 | 5888 | 0 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 29 | conjugate_pair | (6A, 11A, 11B) | 6 | 7056 | 77616 | 11 | 6750 | 96 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 30 | conjugate_pair | (8A, 11A, 11B) | 7 | 10080 | 110880 | 11 | 9648 | 0 | Decided | ATLAS max. subgroups; HJLPPZ 3.1 | reproduced |
| 31 | rational | (2A, 8A, 8A) | 0 | 491/8 | 491 | 8 | 28 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 32 | rational | (3A, 5A, 5A) | 0 | 1310/3 | 6550 | 15 | 30 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 33 | rational | (3A, 4A, 8A) | 0 | 441/2 | 1764 | 8 | 54 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 34 | rational | (3A, 5A, 6A) | 0 | 863/3 | 3452 | 12 | 70 | 4 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 35 | rational | (4A, 4A, 5A) | 0 | 909 | 13635 | 15 | 104 | 2 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 36 | rational | (3A, 6A, 6A) | 0 | 1757/4 | 5271 | 12 | 120 | 16 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 37 | rational | (3A, 5A, 8A) | 1 | 476 | 3808 | 8 | 170 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 38 | rational | (4A, 4A, 6A) | 0 | 1713/2 | 10278 | 12 | 192 | 14 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 39 | rational | (3A, 6A, 8A) | 1 | 634 | 5072 | 8 | 350 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 40 | rational | (4A, 5A, 5A) | 1 | 1936 | 29040 | 15 | 396 | 4 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 41 | rational | (3A, 8A, 8A) | 2 | 889 | 7112 | 8 | 568 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 42 | rational | (4A, 4A, 8A) | 1 | 1170 | 9360 | 8 | 568 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 43 | rational | (4A, 5A, 6A) | 1 | 1757 | 21084 | 12 | 776 | 22 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 44 | rational | (5A, 5A, 5A) | 2 | 61058/15 | 61058 | 15 | 972 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 45 | rational | (4A, 6A, 6A) | 1 | 8691/4 | 26073 | 12 | 1220 | 56 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 46 | rational | (4A, 5A, 8A) | 2 | 2712 | 21696 | 8 | 1484 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 47 | rational | (5A, 5A, 6A) | 2 | 3836 | 46032 | 12 | 2178 | 24 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 48 | rational | (4A, 6A, 8A) | 2 | 6615/2 | 26460 | 8 | 2534 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 49 | rational | (5A, 5A, 8A) | 3 | 5504 | 44032 | 8 | 3288 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 50 | rational | (5A, 6A, 6A) | 2 | 14417/3 | 57668 | 12 | 3416 | 90 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 51 | rational | (4A, 8A, 8A) | 3 | 10305/2 | 41220 | 8 | 4208 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 52 | rational | (6A, 6A, 6A) | 2 | 69737/12 | 69737 | 12 | 4860 | 114 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 53 | rational | (5A, 6A, 8A) | 3 | 7140 | 57120 | 8 | 5666 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 54 | rational | (6A, 6A, 8A) | 3 | 8590 | 68720 | 8 | 7940 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 55 | rational | (5A, 8A, 8A) | 4 | 10680 | 85440 | 8 | 8816 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 56 | rational | (6A, 8A, 8A) | 4 | 13335 | 106680 | 8 | 12584 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |
| 57 | rational | (8A, 8A, 8A) | 5 | 18720 | 149760 | 8 | 17760 | 0 | Decided | ATLAS max. subgroups; three real points (derived) | reproduced |

`a = c` (u = 1) candidates on the 21 rational multisets with 2A, none generating (asserted):
`(2A,2A,2A)` 98, `(2A,2A,3A)` 30, `(2A,2A,4A)` 12, `(2A,2A,5A)` 5, `(2A,2A,6A)` 6, all others 0.
These are RustMath's own counts, equal to the design's measurement; only "none generating" is
asserted.

## 5. `(C, 23A, 23B)` and the dictionaries

| C | genus | size (= n) | real classes | release time | reference |
|---|---:|---:|---:|---:|---|
| 2A | 4 | 7 | 3 | 1.7 ms | equal entry by entry |
| 3A | 6 | 134 | 16 | 23 ms | equal |
| 4A | 7 | 660 | 42 | 136 ms | equal |
| 5A | 8 | 1422 | 32 | 256 ms | equal |
| 6A | 8 | 1632 | 72 | 362 ms | equal |
| 8A | 9 | 2272 | 0 | 455 ms | equal |

On all 6,127 triples: `dim H^1_par(G0) = dim H^1_par(P0/G0) = g`, `dim H^1_par(P0) = 2g`; on
every real cover `dim Jac(X)(R)[2] = g + r - 1`; `components`, `dim_J2_real`, `dim_W_real`,
`real_points_above_0` and `representatives_g1` equal `passports_C_23A_23B.json`.

`(2A, 23A, 23B)` against `group_facts.json` `nielsen` (debug test):

| index | conjugate | real | dim W / W^dual / Jac[2] | orbifold G0 | components | Fix(c0) cap Fix(g1) | dim J2 real | dim W real | HJLPPZ row | Janik row | Le Lay id |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | no | 4 / 4 / 8 | 7 | | | | | 3 | 6 | 1 |
| 1 | 0 | no | 4 / 4 / 8 | 7 | | | | | 5 | 2 | 2 |
| 2 | 2 | yes | 4 / 4 / 8 | 7 | 1 | 3 | 4 | 2 | 4 | 4 | 3 |
| 3 | 3 | yes | 4 / 4 / 8 | 7 | 1 | 3 | 4 | 2 | 6 | 0 | 7 |
| 4 | 6 | no | 4 / 4 / 8 | 7 | | | | | 2 | 5 | 4 |
| 5 | 5 | yes | 4 / 4 / 8 | 7 | 2 | 3 | 5 | 3 | 1 | 1 | 6 |
| 6 | 4 | no | 4 / 4 / 8 | 7 | | | | | 7 | 3 | 5 |

The three dictionaries, each bijective onto the seven classes and equal to `group_facts.json`:

| Source | Reading | Result | Test, time (debug) |
|---|---|---|---|
| HJLPPZ `seven_nielsen_class_representatives.m`, `$M23ISGALOIS=/home/user/external/m23isgalois` (sha256 `2735178d...4fbb`, rechecked) | file fixed `g2` = published `g3` (asserted); triple (file g1, file g3, file g2) | rows 1..7 -> `[5, 4, 0, 2, 1, 3, 6]` | `env_hjlppz_dictionary`, 26 ms |
| Janik's Table 1 (vendored `janik_table.json`) | direct reading `g1 g2 g3 = 1` with the published `g1` (asserted per row) | rows 0..6 -> `[3, 5, 1, 6, 2, 4, 0]` | `nielsen_2a_23a_23b_and_janik` |
| Le Lay `verification/certify_geometric_reflection_obstruction.g`, `$LELAY_M23=/home/user/external/frontier-math-m23-cover-investigation` (sha256 `fad8d332...0fe61`, rechecked) | `y` = published `g3` (asserted), each `x_k` in 2A of M23 (`class_of_checked`), braid `(x, y z y^-1, y)` | `[0, 1, 2, 4, 6, 5, 3]`; his real classes 3, 6, 7 are our fixed 2, 5, 3 | `env_lelay_dictionary`, 21 ms |

Without the variables both `env_` tests fail with "set M23ISGALOIS to a checkout of
shaowuz/m23isgalois" and "set LELAY_M23 to a checkout of
research-reflexivity/frontier-math-m23-cover-investigation" (checked, `run_env_unset.log`).

## 6. Measured times and memory (this machine, shared with another job)

| Item | Time | Peak memory |
|---|---|---|
| Debug build of the tests (incremental, after the first) | a few seconds | |
| Release build of `rustmath-groups` with dependencies, from clean (`-j 2`) | 1 min 16 s | target/release 256 MB (deleted afterwards) |
| Release rebuild after a change | 11 to 14 s | |
| `m23_classes` fast tests, debug, one thread | 1.65 s (most of it `centralizers_match_gap`; 1.6 to 2.0 s across runs) | 12.6 MB |
| `m23_atlas`, debug, one thread | 1.74 s (`check_against` 1.60 s, orthogonality 104 ms) | 12.6 MB |
| `m23_three_point`, debug, one thread | 1.84 s (18 rows 1.61 s) | 12.7 MB |
| `m23_nielsen`, debug, one thread | 0.30 s (seven classes 25 ms, `Matrix<PrimeField>` cross-check 55 ms) | 12.7 MB |
| Whole crate `cargo test -p rustmath-groups` (debug, parallel) | 13.1 s wall, doc-tests included | |
| `full_census` (release) | 12.3 s (10.2 s in an earlier run; design 8.9 s) | VmHWM 3 MB |
| `full_family_11` (release) | 38.4 s (40.5 s earlier; design 40.3 s) | 3 MB |
| `full_57_rows` (release) | 3.2 s: lists 0.71 s, rows 2.31 s, slowest `(8A,11A,11B)` 0.43 s, `a = c` 0.19 s | 78 MB |
| `full_c_23a_23b` (release) | 1.9 s (design 1.3 s) | 78 MB |
| Example with `--check` (release) | 3.1 s inside, 4.0 s wall | |
| `env_hjlppz_dictionary`, `env_lelay_dictionary` (debug) | 26 ms, 21 ms | |

Each `full_` test ran alone (`--exact`, `--test-threads 1`), so its VmHWM is its own. The debug
peaks come from `getrusage` of each test binary run with `--test-threads 1`.

## 7. API (public, `rustmath_groups::m23`)

- `group`: `Perm23`, `M23_ORDER`, `M23Error`, `rmul`, `inverse`, `conjugate`, `identity23`,
  `cycles`, `num_cycles`, `order`, `power`, `mask_image`, `fixed_mask`, `generates_transitive`,
  `from_cycles_1based`, `from_image_list`, `to_image_list`, `M23` (`published`, `g1`, `g2`,
  `g3`, `chain`, `base_orbit_lengths`, `contains`, `for_each_element`, `heptads`,
  `even_golay_basis`).
- `classes`: `M23Class` (`ALL`, `index`, `from_index`, `name`, `from_name`, `element_order`,
  `cycle_type`, `num_cycles`, `inverse`, `is_rational`, `partner`), `signature`, `SIFT_BUDGET`,
  `ORBIT_CAP`, `ClassIdentifier` (`new`, `golay_trace`, `heptad_family_11`,
  `cyclic_steiner_sign`, `class_of_member`, `class_of_checked`, `representative`,
  `centralizer_order`), `centralizer_order_by_sifting`, `conjugacy_class_by_orbit`, `Census`,
  `census`, `ClassLists` (`by_enumeration`, `by_orbit`, `get`).
- `atlas`: `AtlasNumber`, `AtlasTable` (`m23`, `parse`, `source`, `num_irreducibles`, `value`,
  `class_size`, `centralizer_order`, `element_order`, `power_map_keys`, `power_map`,
  `permutation_character_23`, `is_rational_class`, `structure_constant`, `check_orthogonality`,
  `check_against`, `to_float_table`).
- `three_point`: `VectorKind`, `ClassVector` (`validate`), `ThreePointError`,
  `enumerate_class_vectors`, `genus23`, `CountStatus`, `TrustedResult`, `StructureCount`,
  `structure_count`, `ClassVectorCount`, `count_class_vector`.
- `nielsen`: `Triple`, `NielsenError`, `complex_conjugate`, `NielsenClass23` (`new`, `class`,
  `len`, `is_empty`, `num_candidates`, `representatives`, `triple`, `index_of`,
  `complex_conjugation`, `real_structure`, `real_points_above_0`, `real_components`).
- `cohomology`: `F2Error`, `F2Span`, `F2Module`, `standard_modules`, `TripleCohomology`.
- Outside `m23`: `StabilizerChain::{num_levels, orbit, transversal_element, for_each_element}`;
  `CharacterError`, `Character::{with_class_sizes, class_sizes, try_inner_product}`,
  `CharacterTable::{class_sizes, inner_product}`; `ClassFunction::{with_class_sizes,
  class_sizes}`.

## 8. Deviations from the design, with evidence

1. `class_data_match_gap` is in `tests/m23_atlas.rs`, not `m23_classes.rs`: it needs the table,
   which step 4 adds. `m23_classes.rs` checks the centralizers against GAP's values written out.
   The design's single `degenerate_cases` is split into one test per file
   (`degenerate_class_cases`, `degenerate_cases`, `degenerate_nielsen_cases`).
2. `from_cycles_1based<C: AsRef<[u8]>>(cycles: &[C])` instead of `&[&[u8]]`: it accepts both
   arrays and `Vec<Vec<u8>>` (the readers' form).
3. `enumerate_class_vectors` returns `Result` and `genus23` returns
   `Result<i64, ThreePointError>`: the structure constant can fail in principle.
4. `ThreePointError`: `NotRationalVector` (whose condition the design left open, review 2 n1) is
   replaced by `InvalidClassVector(String)` with the conditions of n1, and `Table(String)` for a
   table error. `M23Class::is_rational` is true for 1A (mathematically rational); the
   enumeration excludes 1A explicitly, as the owner's GAP script does, and validation accepts it,
   so `(1A, 1A, 1A)` exercises `GenerationCriterionNotApplicable` with an exact
   `structure_count` (1 survivor, n = 1/10,200,960).
5. `ClassVectorCount` gains `a_equals_c_candidates` and `a_equals_c_generating` (the `u = 1`
   counts, m3), and `StructureCount`/`structure_count` exist (n2).
6. `NielsenClass23<'a>` holds `&'a M23`, so `index_of`, `complex_conjugation` and
   `real_structure` need no group argument (the design's signatures have none). `new` refuses a
   class that is not rational or is 1A (`NotARationalClass`), which is what makes "every triple
   generates" true for every accepted class.
7. `F2Error` gains `NotParabolic` (the image of a parabolic cocycle under conjugation is not
   parabolic) and `ProductNotOne`; `F2Module::new(sub)` is a shorthand. With 23-bit inputs a
   module has dimension at most 23, so `DimensionTooLarge` cannot be reached through the public
   constructors; the check stays, as m6 asked, and `CapacityExceeded` is tested.
8. T-215, with n3: `Character::inner_product` now panics for two size-less characters with
   different key sets (before: a plausible wrong mean). No caller does this (no use of
   `Character` or `ClassFunction` outside these two files in the workspace, checked by grep).
   `is_irreducible` is false and `norm_squared` is 0 for a character with no values, instead of
   a panic. `ClassFunction::with_class_sizes` sets the group order to the sum of the sizes; a
   later `set_group_order` with another value makes `inner_product` return `None` (tested).
9. `M23::for_each_element` uses the chain's own transversals; the prototype built its own. The
   order of the elements differs; no result depends on it (all counts; the Nielsen
   representatives are sorted).
10. Times: `full_census` 10.2 to 12.3 s against 8.9 s, `full_c_23a_23b` 1.9 s against 1.3 s
    (`complex_conjugation` and `real_structure` each sift up to 23 conjugators per class; the
    machine is shared with another job). Memory is as designed.
11. Clippy on `rustmath-groups` stops at a deny lint that exists at `28c4b69`
    (`derived_hash_with_manual_eq`, `src/indexed_free_group.rs:181`), so the test and example
    targets were linted with `-A clippy::derived_hash_with_manual_eq`. Not fixed (unrelated to
    WP-P; follow-up WPP-8).
12. rustfmt: run on all new files (clean). In the changed files `representation.rs` and
    `class_function.rs` only my hunks were formatted (19 and 3); their pre-existing unformatted
    hunks (4 and 2, the same count as at `28c4b69`) and the 8 in `bsgs.rs` (my code there needed
    none) were left, to keep the diff on WP-P. `lib.rs` was not passed to rustfmt, which would
    format the whole crate through its module tree; its two changed lines are formatted.
13. Dependent crates were not rebuilt (disk). No public item was removed or changed in
    signature; the new fields are private; a grep finds no glob import of `rustmath_groups` and
    no `CharacterError` name in the eight dependent crates.

Review 2 minor notes: n1, n2, n3, n4, n5 (permutation character and orders checked; debug peak
memory recorded; "no `unsafe`" stated in `m23/mod.rs`; the `tr(x^2) = tr(x)` remark in
`classes.rs`) are all folded in. None skipped. One correction while folding n5: review 2 says
"2 is a square mod 7, 15 and 23"; 2 is not a square mod 15, so the module docs state only what
follows from `tr(x^2) = tr(x)` (squaring preserves each trace class, as GAP's power map 2 does on
7AB, 15AB and 23AB).

## 9. Acceptance commands run

| Command | Result |
|---|---|
| `cargo test -p rustmath-groups` (baseline at `28c4b69`) | lib 548, doc 78 (4 ignored), `galois_narrowing` 3 of 4 fail: the gitignored `data/transitive_24.jsonl` is absent in this worktree (environment, not WP-P) |
| `cargo test -p rustmath-groups --lib bsgs` / `-- representation class_function` | 13 / 42 pass |
| `gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null \| sha256sum` | `f9503e80...0f5196eb`, 3,471 bytes |
| `python3 -I .../extract_janik_table.py <owner checkout> janik_table.json` | sha256 `3db9d440...2dac` |
| `cargo test -p rustmath-groups --test m23_classes` / `m23_atlas` / `m23_three_point` / `m23_nielsen` | 5 (2 ignored) / 6 / 3 (1 ignored) / 4 (3 ignored) pass |
| `cargo test --release -p rustmath-groups --test m23_classes -- --ignored --exact full_census` and `full_family_11` | pass (twice) |
| `cargo test --release -p rustmath-groups --test m23_three_point -- --ignored --exact full_57_rows` | pass (twice) |
| `cargo test --release -p rustmath-groups --test m23_nielsen -- --ignored --exact full_c_23a_23b` | pass (twice) |
| `M23ISGALOIS=/home/user/external/m23isgalois LELAY_M23=/home/user/external/frontier-math-m23-cover-investigation cargo test -p rustmath-groups --test m23_nielsen -- --ignored env_` | 2 pass; without the variables both fail loudly |
| `cargo run --release -p rustmath-groups --example m23_class_vectors -- --check rustmath-groups/tests/data/m23` | 57 reproduced, exit 0 |
| `cargo test -p rustmath-groups` (all targets, after) | lib 558, doc 79 (4 ignored), all new tests pass; `galois_narrowing` as in the baseline |
| `cargo clippy -p rustmath-groups --tests --examples --no-deps -- -A clippy::derived_hash_with_manual_eq` | no warning in `src/m23/`, the new tests, the example or the new code of `bsgs.rs`, `representation.rs`, `class_function.rs`; lib warnings 105 (baseline 106) |
| `rustfmt --edition 2021 --check` on the new files | clean |

`target/release` was deleted at the end (free disk 6.8 GB). Zero `unsafe` in the new code.

## 10. Left undone

- HANDOFF 4.4 (twists, small Hurwitz schemes, four branch points): out of scope (WPP-1, WPP-2).
- The rest of T-215 (SNF abelianization, Artin and braid word problem, trait-default facades,
  chief series, homomorphism check, `block_systems`, abelian basis): not used by WP-P (WPP-3).
- General centralizers and conjugacy over a BSGS (Leon's backtrack): WP-P's centralizer search is
  exact but limited to small `C_Sym(x)`, with the orbit fallback (WPP-4).
- A general character-table algorithm (Dixon-Schneider): the M23 table is data (WPP-6).
- `docs/` and `PROJECT_TRACKER.md` are not edited (a later step does that); nothing committed.

## 11. Handoffs

### PROJECT_TRACKER.md rows

- `rustmath-groups/src/representation.rs:148` (Character inner product, 🔴): fixed. Now
  `representation.rs:283` `try_inner_product` and `:342` `inner_product`, weighted by class
  sizes when the characters carry them; `:465` `CharacterTable::inner_product`; S3 standard
  character gives 1; M23 table orthonormal. Rate done for characters with class sizes (f64);
  the size-less element-keyed reading is documented and refuses different key sets.
- `rustmath-groups/src/class_function.rs:211` (class-function inner product, 🛑): fixed at
  `class_function.rs:259`; `None` unless decided.
- T-215 (line 246): the two inner products are done; the other items stay open.
- New rows: `bsgs.rs:354` `for_each_element` (+ `:318` to `:330` accessors), done (S4, A5, M11,
  M23); `src/m23/group.rs:330` `M23::published` (public M23 constructor), `classes.rs:364`
  `ClassIdentifier` (exact class map of M23), `:675` and `:704` centralizer orders, `:851`
  `census`, `atlas.rs:263` ATLAS table of M23 (data, exact), `:440` structure constants,
  `three_point.rs:450` `count_class_vector`, `nielsen.rs:131` `NielsenClass23`,
  `cohomology.rs:310` `TripleCohomology` (parabolic `H^1` over `F_2` for triples), all done
  against the owner's data and GAP. The Section 2 lines "no public M23 or M24 constructor" and
  "the chain keeps its transversals private, so there is no element iterator" no longer hold
  for M23.

### MAGMA tracker rows (`docs/magma/intrinsics.csv`, via `status_overrides.csv`)

- `ch091.InnerProduct`: candidate at `representation.rs:148` -> partial or done at
  `representation.rs:283` / `:465` (weighted; f64 values).
- `ch091.IsIrreducible`, `ch091.Norm`: now correct when class sizes are attached
  (`representation.rs`, `class_function.rs:259`).
- `ch091.CharacterTable`: partial, M23 only, as data (`m23/atlas.rs:263`).
- `ch091.StructureConstant`: currently mapped to `rustmath-manifolds/src/lie_algebra.rs:200`, a
  different object. `AtlasTable::structure_constant` (`atlas.rs:440`) is the normalized `n`, with
  `n(C1, C2, C3) = a_{C1, C2, C3^-1} / |C_G(c3)|` documented; partial (M23 only, normalized).
- `ch058.ClassMap`: partial, M23 only (`ClassIdentifier::class_of_member`, `classes.rs:638`).
- `ch058.ClassCentraliser`: partial, orders only, M23 only (`classes.rs:675`).
- `ch068.CohomologyGroup` / `CohomologyModule`: partial, parabolic `H^1` over `F_2` of the three
  Golay modules for triples of M23 only (`cohomology.rs`).

### Workstream handoff (`docs/workstreams/m23_golay/HANDOFF.md`)

- Add a "WP-P: Status (2026-11-16)" block to Section 4 in the style of WP-F's: code
  (section 2 of this report), tests, data and README, commands (section 9), the 57-row result,
  the six `(C, 23A, 23B)` classes, the three dictionaries, the statuses (`Decided`, with
  `TrustedResult`), times.
- Section 2: the T-215 inner-product line and the "no public M23 constructor / no element
  iterator" line move to "Usable as is"; `conjugacy_classes.rs:161` stays in "Do not call".
- Section 8, step 3: done (57 rows and three dictionaries; also 4.3).
- Copy this report, the design and its reviews into `docs/workstreams/m23_golay/wpp/`.

### Follow-ups

- WPP-1: 4.4 item 1, the swap action on Nielsen classes for the twists of `(R, D, D)`; the
  machinery of `three_point.rs` and `nielsen.rs` applies.
- WPP-2: 4.4 items 2 and 3.
- WPP-3: the rest of T-215.
- WPP-4: Leon's backtrack for centralizers and conjugacy (MAGMA ch. 58).
- WPP-6: Dixon-Schneider (MAGMA ch. 91).
- WPP-7: check the rational real criterion against Fried 1977, Krull and Neukirch 1971, Debes
  and Fried 1994; until then the rational `real` values carry `ConjugationThreeRealPoints`.
- WPP-8 (new): the pre-existing clippy deny error `derived_hash_with_manual_eq` at
  `src/indexed_free_group.rs:181` blocks `cargo clippy -p rustmath-groups --tests`.
- WPP-9 (new): `tests/galois_narrowing.rs` fails in a fresh worktree because the gitignored
  degree-24 data is absent; regenerate per `data/README.md` or gate the three tests.
- WPP-10 (new): after the merge, `cargo check` of the eight crates that depend on
  `rustmath-groups` (not run here for disk; no breaking change expected, section 8 item 13).
- WPP-Q1 (owner): the licence footing of the generated table (`tests/data/m23/README.md`).
- WPP-Q2 (owner): vendoring his `LELAY_X` transcription instead of reading Le Lay's certificate
  at run time; the default (not vendored) is implemented.

## 12. Minor items of review 1

All eight items of `WPP.review1.md` are accepted and fixed. None is rejected. Each change is
small. No WP-P number changes: the 57 rows, the six `(C, 23A, 23B)` classes and the three
dictionaries are as in sections 4 and 5. Line numbers are after the fix. Paths are relative to
`rustmath-groups/`.

| # | Item | Change | Test (fails before the fix) |
|---|---|---|---|
| 1 | `F2Module::matrix` did not check that `g` preserves the quotient `G0` of `Q = P0 / G0` | `src/m23/cohomology.rs:235` keeps `quo_span`; `:299` to `:305` return `Err(NotInModule)` when the image of a quotient basis vector leaves `span(quo)`. Doc at `:296` to `:298` says so. | `tests/m23_nielsen.rs:356` to `:365`: a transposition acts on `P0` (22 rows, `Ok`), is refused on `Q`, and `TripleCohomology::new(&Q, &(t, t, 1))` is `Err(NotInModule)` (before: `Ok`, `dim_parabolic` 0, `dim_orbifold` 9) |
| 2 | `conjugation_fixed_dim` did not check that `c0` is a real structure | New variant `F2Error::NotARealStructure` (`cohomology.rs:51`, Display `:71`). `:440` to `:447` check `c0^2 = 1`, `c0 g1 c0 = g1^-1`, `c0 g2 c0 = g3^-1`, the same test as `real_components` (`nielsen.rs:262`). Doc at `:436` to `:439`. | `tests/m23_nielsen.rs:370` to `:378`: `c0 = 1` on class 5 of `(2A, 23A, 23B)`, module `P0`, is `Err(NotARealStructure)` (before: `Ok(8)`); the computed real structure is `Ok` |
| 3 | `centralizer_order_by_sifting` with `budget = u64::MAX` let the saturated size of the identity through | `src/m23/classes.rs:727`: `size == u64::MAX \|\| size > budget`. Doc at `:701` to `:703`: a saturated size counts as above every budget. | `tests/m23_classes.rs:257` to `:264`: identity, budget `u64::MAX`, gives `Err(BudgetExceeded { needed: u64::MAX, budget: u64::MAX })` at once (before: an attempt to build the 23! permutations) |
| 4 | With class sizes on only some classes, the T-215 inner product is a weighted mean | Docs only, as the review asks: `src/representation.rs:201` to `:203` (`Character::with_class_sizes`), `:288` to `:290` (`try_inner_product`, with the review's S3 example: 1 on `{e: 1, (12): 3}`), `src/class_function.rs:98` to `:101` (`ClassFunction::with_class_sizes`, naming `inner_product`, `norm`, `is_irreducible`). A check is not possible here: a `Character` does not carry the group order. | docs |
| 5 | `tests/m23_three_point.rs` and `tests/m23_nielsen.rs` cited no MAGMA chapter | `m23_three_point.rs:5`: ch. 58 and 91. `m23_nielsen.rs:3` to `:4`: ch. 58 and 68. | headers |
| 6 | `AtlasTable::parse` cast with `as` | `src/m23/atlas.rs:247` to `:255`: `unsigned_list::<T>` converts with `T::try_from` and returns `Err("<field>: entry x out of range")`; `:300` to `:305` use it for `orders` (`u32`), `centralizer_orders` and `class_sizes` (`u64`); `:323` to `:327` the power map index uses `usize::try_from`; `:354` to `:355` `d` uses `u32::try_from` (`Err("d: ... out of range")`). Also `check_orthogonality` (`:458` to `:466`) now uses `checked_add` and `checked_mul`: `try_from` still admits values up to `2^63 - 1`, and `3795 (2^63 - 1)` overflows `u64`. | `tests/m23_atlas.rs:137` to `:160`: a centralizer of -1 is an `Err` naming `centralizer_orders`; `d = 2^32 + 7` is an `Err` (before: `as u32` made it 7 and the file parsed); a centralizer of `2^63 - 1` at 2A parses, and `check_orthogonality` returns `Err` (before: overflow panic in debug at `|C| |C(c)|`) |
| 7 | The `by_enumeration` doc gave only the 73 MB of the six rational classes | `src/m23/classes.rs:912` to `:913`: 23 bytes per stored element, at most 235 MB for all 17 classes, plus the spare capacity of the growing vectors. | doc |
| 8 | `real_structure`, `real_points_above_0`, `real_components` panic for `i >= len()` | Documented, as `triple` does (`src/m23/nielsen.rs:224` to `:225`, `:249`, `:260`). Documented rather than an `Err`: `NielsenError` has no index variant, `real_points_above_0` returns `usize`, and every caller iterates over `0..len()`. | doc |

Public API: one new enum variant (`F2Error::NotARealStructure`). A grep finds no exhaustive
match on `F2Error` outside `cohomology.rs`. No signature changed.

Reruns (this machine, shared with another job, `-j 2`):

| Command | Result |
|---|---|
| `cargo test -p rustmath-groups --test m23_atlas --test m23_classes --test m23_nielsen --test m23_three_point` (debug) | 6 / 5 (2 ignored) / 4 (3 ignored) / 3 (1 ignored) pass; 1.54 s, 1.66 s, 0.20 s, 1.50 s |
| `M23ISGALOIS=... LELAY_M23=... cargo test -p rustmath-groups --test m23_nielsen -- --ignored env_` | 2 pass |
| `cargo test -p rustmath-groups --lib -- representation class_function` | 42 pass |
| `cargo test --release -p rustmath-groups --test m23_nielsen -- --ignored --exact full_c_23a_23b --test-threads 1` | pass, 1.88 s, peak RSS 78 MB (as before: 1.9 s, 78 MB); release build 1 min 32 s; `target/release` deleted afterwards |
| `cargo test -p rustmath-groups --no-fail-fast` (debug) | lib 558, doc 79 (4 ignored), all `m23_*` pass; `galois_narrowing` 3 of 4 fail as in the baseline (WPP-9, gitignored data) |
| `cargo clippy -p rustmath-groups --tests --examples --no-deps -- -A clippy::derived_hash_with_manual_eq` | no warning in `src/m23/`, the `m23_*` tests or the example; the two warnings that point into `representation.rs` and `class_function.rs` are pre-existing lines (`:239` `for_kv_map`, `:207` `cast_abs_to_unsigned`, both in `HEAD`) |
| `rustfmt --edition 2021 --check` on `src/m23/{atlas,classes,cohomology,nielsen}.rs` and the four `m23_*` tests | clean |
| `rustfmt --check` on `representation.rs`, `class_function.rs` | only the pre-existing diffs (4 and 2 hunks, as in `HEAD`); none on the changed lines |
| `cargo doc -p rustmath-groups --no-deps` | no warning in the changed docs (new intra-doc links resolve); `target/doc` deleted |

The release `full_57_rows`, `full_census`, `full_family_11` and the example were not rerun. They
do not call `F2Module::matrix` or `conjugation_fixed_dim`. They reach the other changed code only
through `AtlasTable::m23()` (the parse of the embedded file) and `centralizer_order` with
`SIFT_BUDGET`, and the debug tests `atlas_file_is_the_gap_output`,
`atlas_orthogonality_and_57_constants`, `class_data_match_gap` and `centralizers_match_gap` run
these on the same inputs, with all 17 classes and all 57 structure constants checked. Nothing
committed. Free disk 6.4 GB at the end.

# Implementation report: Unit C1-combinatorics (rustmath-combinatorics)

Base: HEAD `dc367a2` (branch `claude/gracious-gates-e6b6l7`), with I3, I1, I2, P1, F1, R1, M1a, M1b
committed. No S1 or MF1 commit or report exists yet, so nothing of theirs was rebased over. The
working tree was clean when this unit started. Design: `design/C1-combinatorics.md` (body with
Revisions 1-3). Reviews 1-4 read; review 4 approved. Scratch:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/C1-work/`
(gp checks `check1.gp`, `check2.gp`; logs `test_crate.log`, `t_*.log`, `check_ws.log`,
`clippy*.log`; lint matcher `lintmatch.py`; HEAD check in `headcheck/`).

Handoffs read from earlier reports: I3 hands C1 "the u32 `binomial` can call
`SageInteger::binomial`" (done, §3.8). No other report names C1. Nothing in M1a/M1b/F1/R1 touches
this crate.

## Items, in plan order (file:line at the end of this unit)

### Step 1. Cargo
- `rustmath-combinatorics/Cargo.toml`: `rug = { version = "1.24", default-features = false,
  features = ["float", "integer"] }` (the rustmath-rationals declaration) and
  `num-bigint = { workspace = true }`, each with a one-line comment.
- `Cargo.lock`: the `rustmath-combinatorics` entry gains `num-bigint` and `rug`; no new package.

### Step 2. `src/machine_word.rs` (new, private; `mod machine_word;` at `lib.rs:105`)
Header cites ch. 144 §144.2/§144.3. `checked_pow_usize` (:12), `binomial_usize_checked` (:28,
`Some(0)` for j > m), `falling_factorial_usize_checked` (:50), `u32_arg` (:66),
`subset_mask_count` (:79). Unit tests (7): the §6.6 values, `binomial_usize_checked` against
`SageInteger::binomial` for m <= 100 and every j, the moved debruijn helper assertions, the u32 and
mask panics (64-bit gated).

### Step 3. `src/factorials.rs` (new, `pub mod factorials`)
Header: ch. 144 §144.2 `Factorial`/`NumberOfPermutations` (digest :36-37), ch. 49 Pochhammer
(:292), Concrete Mathematics §2.6, vzGG §10.1, Sage. `product_tree` (:28), `product_u64s` (:50),
`product_u128s` (:68, new helper, see Deviations), `product_range_u64` (:89), `rising_run` (:98),
`falling_factorial_integer` (:130), `rising_factorial_integer` (:149), `falling_factorial_in`
(:199), `rising_factorial_in` (:207). The `_in` versions take `one` (design §4.2), stop at a zero
factor and multiply by a balanced tree over R. Unit tests (5): range edges (empty, across the u64
packing boundary 20!/21!, lo = 2^64 - 2: gp `(2^64-2)*(2^64-1)` =
340282366920938463408034375210639556610), u64/u128 packing, `rising_run` with a big start, the
integer versions against a direct product for x in -30..=30 and k <= 40, the generic versions
against the integer ones.

### Step 4. `src/partition_number.rs` (new, `pub mod partition_number`)
Header: ch. 145 §145.2, ch. 18 §18.8, Johansson 2012 (1.2), (1.4)-(1.8), (2.1), Algorithm 1;
Rademacher 1938; Whiteman 1956; MPFR directed rounding. `NUMBER_OF_PARTITIONS_MAX_N` (:40),
`number_of_partitions` (:78: step-0 domain assert "above 10^16", Euler for n <= 1000, exponent
check against `rug::float::exp_max()`), `euler_list`, `Enclosure` (:142; every op through
`Float::with_val_round` Down/Up, `cos`/`U` rules of §2.4), `remainder_bound_f64`,
`remainder_bound_upper` (:297), `choose_terms` (:326, f64 scan then the rigorous 1/4 recheck),
`selberg_sum` (:345, Algorithm 1), `rug_to_integer` (:370, digits, Lsf), `partitions_hrr` (:385,
u64 v, `prec_k` with `ceil`, retry cap with the "internal error" panic). Unit tests (6, §6.3):
`hrr_equals_euler_up_to_3000`, `remainder_bound_is_below_a_quarter`,
`enclosure_operations_contain_the_value` (gp `\p 70; Pi`), `selberg_sum_matches_the_dedekind_definition`
(A_k(n) from (1.6)/(1.7) with exact Dedekind sums, k 3..=12, n 0..=40),
`rug_integer_converts_through_digits`, `exponent_range_covers_the_domain`.

### Step 5. `src/lib.rs`
- Modules and re-exports (`:103-105`, `:369-373`): `factorials`, `partition_number`,
  `machine_word`; `number_of_partitions`, `NUMBER_OF_PARTITIONS_MAX_N`, the four factorial
  functions, `number_of_standard_tableaux`.
- Citation comment (`:375-379`, ch. 144 §144.2 and ch. 18 §18.8).
- `factorial` (:392, `product_range_u64`), `binomial` (:401, `SageInteger::binomial`),
  `multinomial` (:416, u128 sum, product of binomials largest k first, `# Panics` I3 line),
  `catalan` (:442, u64 2n and n + 1, `# Panics` I3 line), `fibonacci` (:451) and `lucas` (:458) by
  the doubling pair, `falling_factorial` (:466) and `rising_factorial` (:475) as wrappers (the
  "Pochhammer" name moved to the rising one, review 3 m7), `RowKind` + `band_rows` (:502),
  `stirling_first` (:546), `stirling_second` (:564), `bell_number` (:584) with the in-place
  `bell_triangle_step` (:593), `eulerian` (:612), `motzkin` (:710, three-term recurrence),
  `schroder_step_factors` (:725), `schroder_large` (:740, two rolling values, u64 multipliers).
  `narayana`, `delannoy*`, `schroder_small` unchanged.
- Unit tests: `schroder_step_factors_do_not_wrap` (4294967301 and 12884901891; HEAD 5 and 3),
  `band_rows_stay_band_sized` ((10^5, 10^5-1), (2000, 1990), (500, 250), (500, 1); buffer
  <= w + 1 for the three kinds; s1 = s2 = 4999950000 = gp `binomial(10^5,2)`;
  `eulerian(10^5, 10^5-2) = 2^n - n - 1` with 100000 bits).

### Step 6. `src/ch144_enumerative.rs`
`number_of_permutations` (:39, wrapper of `falling_factorial_integer`), `fibonacci_pair` (:49,
pub(crate)), `generalized_fibonacci_number` (:78, `G_n = g0 F_(n-1) + g1 F_n`, negative indices
through `n.unsigned_abs()`); module table updated.

### Step 7. `src/derangements.rs`
`count_derangements` (:41) by binary splitting (`derangement_split`); doc and header name the
method (Haible-Papanikolaou). `count_derangements_recurrence` unchanged. The `Rational` import
moved into the test module (its only user now).

### Step 8. `src/set_partition.rs`
`bell_number_optimized` calls `crate::bell_number`; `bell_numbers_up_to` uses the same in-place
row (`bell_triangle_step`). Signatures and values unchanged.

### Step 9. `src/partitions.rs`, `src/young_representations.rs`
- Header (ch. 145 §145.2, §145.4.9, [Ful97] §4.3).
- `Partition::dimension` (:117): usize form of `number_of_standard_tableaux`, documented panic.
- `partition_count` (:1263): `number_of_partitions(n).try_into()`, panics for n > 1000 before any
  work and for p(n) > usize::MAX ("partition_count: p(n) exceeds usize::MAX; use
  number_of_partitions"); the "Hardy-Ramanujan would be needed" doc is replaced.
- `count_partitions_with_max_parts` (:1586): `checked_add`, documented panic (exactness note).
- `number_of_standard_tableaux` (:1649): empty partition 1 first, conjugate when rows > columns,
  Frobenius with cancellation when r(r-1)/2 <= n, hook formula otherwise; n and factors u128.
  Doc example H145E13 (3, 2) = 5 and (11, 10) = 58786.
- `young_representations.rs:55 hook_length_dimension` calls `number_of_standard_tableaux`; its
  doc and the module table name MAGMA `NumberOfStandardTableaux` (ch. 145 §145.4.9).

### Step 10. The other §1.6 items
| item | file:line | change |
|---|---|---|
| 3, 17, 18, 28 | `subset.rs` (header :9-16; `from_bitvector` :77; `rank_binary` :155; `rank_k_subset` :189; `unrank_k_subset` :218; `next_binary` :335; `prev_binary` :361; `count_k_subsets` :683) | `binomial_usize` deleted; checked ranks with the stated panics; unrank treats None as "larger than any rank"; successors on the sorted element list (O(n), every n); `from_bitvector` reads bits below usize::BITS only; `count_k_subsets` through `SageInteger::binomial` with u64 arguments |
| 4 | `path_tableaux.rs:297` | `count_lattice_paths_ne`: u64 m + n |
| 5, 6, 23, 25 | `enumeration.rs` (`PermutationGrayCodeIterator::size_hint`, `CombinationGrayCodeIterator::size_hint`, `InfiniteCartesianProduct::count_partitions`, `stars_and_bars`) | private `factorial` and `binomial` deleted; checked bounds; `stars_and_bars` panics instead of returning 0 |
| 7, 13, 30, 32 | `integer_vectors.rs` | checked `count_integer_vectors_with_sum` (u128 binomial), `count_integer_vectors_in_box` (empty side first, i128 widths, checked u128 product), the sign-pattern mask through `subset_mask_count` (negation on the Integer), five `as i64` casts now `as u64` |
| 8 | `six_vertex_model.rs:533` | `u32_arg(n, "six_vertex partition_function")` |
| 9, 10 | `parking_function.rs:316`, `:334` | checked `(n+1)^(n-1)`; Catalan through `crate::catalan` with the usize::BITS early bound; the private wrapping `catalan_number` deleted |
| 11, 12 | `tuple.rs` (rank :77, unrank :92, type doc, `count` :274, `checked_total`, `size_hint`) | Horner with checked ops; `checked_pow_usize`; `(0, None)` hint when n^k does not fit; `len()` note |
| 14 | step 9 above | |
| 15 | `group_action.rs:183`, `:344` | `num_colors as u64` |
| 16 | `tiling.rs:282` | `u32_arg(n, "count_tilings")` before the transfer matrix is built |
| 19, 20 | `debruijn.rs` (`new`, `is_valid`, `universal_cycle_subsets`, `universal_cycle_permutations`) | checked lengths; helpers deleted; their tests moved to `machine_word.rs` |
| 21, 35 | `designs.rs` (`replication_number`, `OrthogonalArray::new`, `verify_orthogonality`) | u128 numerator, `try_from` panic; checked v^t |
| 22, 30 | `set_system.rs` (`shadow`, `shade`, `is_shattered`, `sunflower_bound`, `sunflower_bound_checked`, `exceeds_sunflower_bound`) | usize masks via `subset_mask_count`; checked bound; `exceeds` is false on None |
| 24 | `combinations.rs` (`rank`, `unrank`) | `binomial_usize` deleted; checked |
| 26 | `superpartitions.rs:453` | `count_superpartitions` by the checked O(n^2) DP of prod 1/(1 - 2x^i) |
| 27, 30 | `binary_words.rs` (`from_usize`, `to_usize`, `all_binary_words`) | bits below usize::BITS; panic on a set bit >= usize::BITS; mask count |
| 29 | `permutations.rs` (`order` :134, `order_integer` :153, `lcm` :601) | checked lcm `(a / gcd) * b`; new `order_integer` (`Integer::lcm`), MAGMA `Order(g)` ch. 58 |
| 30 | `posets.rs` (`all_order_ideals`, `all_antichains`), `subword.rs` (`all_subwords`, `compute_faces` + `SubwordComplex::new` doc), `subword_complex.rs` (`all_subsets`), `grossman_larson.rs` (`admissible_cuts`), `multiset_partition_into_sets_ordered.rs` (`finer`) | usize masks via `subset_mask_count`, `# Panics` and cost lines |
| 30 | `composition.rs` (`signed_compositions`, `signed_compositions_k`, new private `push_uniformly_signed`) | the 2^k sign loop is gone; one uniformly signed composition per composition; `n.unsigned_abs()` |
| 31 | `q_analogue.rs:243`, `t_sequences.rs:531` | u128 sum |
| 33 | `integer_lists.rs` (`integer_lists`, `generate_with_length`, `propagate_constraints`, `satisfies_constraints`) | saturating bounds, i128 slopes |
| 34 | `vector_partition.rs` | caps of 1000 deleted; one remainder per state (`subtract_part`); `checked_count_add` with the function name; `# Panics` on the four public counts |
| 36 | `dyck_word.rs` (`checked_sum_i32`, `checked_sub_i32`; `area`, `bounce_statistic`, `total_area`, `bounce_statistics`, `dinv`, `detailed_bounce_statistics`) | checked i32 sums and differences |
| §5 | `symmetric_group_representations.rs` (`IrreducibleRepresentation::new`, `all_irreducible_representations`, `SymmetricGroupCharacterTable::new`) | `# Panics` docs (n >= 36) |

New unit tests in sources besides steps 2-5: `set_system.rs` (`is_shattered_mask_is_a_usize`,
`is_shattered_refuses_word_size_subsets`), `subword_complex.rs` (`all_subsets_lists_two_to_the_n`,
`all_subsets_refuses_word_size_sets`), `permutations.rs` (`lcm_is_checked`), `superpartitions.rs`
(`count_superpartitions_dp_matches_the_enumeration`). `enumeration.rs::test_factorial` keeps its
seven assertions, now on `falling_factorial_usize_checked(n, n)` (the private factorial is gone).
`debruijn.rs` loses its two helper tests; their nine assertions are in
`machine_word.rs::falling_factorial_usize_checked_values`.

### Step 11. Integration tests and fixtures
- `tests/data/gen_counting.gp`, `tests/data/gen_numbpart.gp` (copies of the design generators);
  `tests/data/ch144_counting.csv` (116 rows) and `tests/data/numbpart.csv` (71 rows) were
  regenerated here with `gp -q tests/data/gen_counting.gp > ...` (34 s) and
  `gp -q tests/data/gen_numbpart.gp > ...` (1 s) and are byte-identical to the design's copies.
- `tests/ch144_counting.rs` (14 tests, §6.1), `tests/ch145_number_of_partitions.rs` (6 tests,
  §6.2, H145E1 with GAP `NrConjugacyClasses(SymmetricGroup(n))` and `NrPartitions`, re-run here),
  `tests/machine_word_counts.rs` (50 tests, §6.5 revisions 1-3 plus review 4 m1). Each file has
  the header of §6 (chapters, Johansson for ch145, the generator commands); every other reference
  value quotes its gp command. All gp values were recomputed in this session
  (`C1-work/check1.gp`, `check2.gp`).

## Review 4 minors
- m1 (tautological f^lambda test, no hook-branch coverage): folded. New
  `number_of_standard_tableaux_matches_the_hook_formula` compares with a test-local copy of HEAD's
  `hook_length_dimension` body for every partition of n <= 20. `standard_tableaux_counts` adds
  f(6, 1^5) = 252 (gp `binomial(10,5)`) and f(2001, 1^2000) (gp `binomial(4000,2000)`: residue
  85248900849712891484894804346834018665, 3994 bits). The [11, 10] row and the n <= 12 equality
  test stay.
- m2 (the `--release` acceptance run): folded. Not run; the debug runs catch everything it would.
  No `cfg_attr(debug_assertions, ignore)` was added.
- m3 nits: 64-bit marks on `count_parking_functions(16)` and `sunflower_bound(20, 2)`: folded.
  `prev_binary` doc: "Time and memory are linear in i plus the subset size": folded. The Verdicts
  rewording is design text, not code: skipped (the implementer does not edit the design).

## Deviations from the design, with evidence
1. `lcm` unit test (§6.6): the design says lcm(2^62, 3) is None. It is not: gp `3*2^62` =
   13835058055282163712 < 2^64. The test uses lcm(2^62, 5) = None (gp 5*2^62 =
   23058430092136939520), lcm(2^62, 3) = Some(13835058055282163712), and lcm(2^62, 4) = Some(2^62),
   which is the case HEAD's `a * b / gcd` wrapped (2^64 = 0 mod 2^64).
2. `replication_number` also returns None for k = 0 (k - 1 negative; HEAD underflowed), not only
   for v = 0.
3. `product_u128s` (factorials.rs:68): a helper for the u128 factors of item 14 ("a factor of
   2^64 or more becomes its own Integer leaf"), in place of inline code.
4. `bell_triangle_step` (lib.rs:593, pub(crate)) shares the in-place row between `bell_number` and
   `bell_numbers_up_to`.
5. HRR retry: 11 attempts (guard 32, 64, ..., 32·2^10), then the internal-error panic. The prototype
   truncated the precision estimate with `as u32`; the code takes `ceil` as §2.4 step 3 says.
   `partitions_hrr` accepts 2 <= n (debug_assert), so the unit test compares it with Euler for every
   n in 2..=3000 directly.
6. `signed_compositions(_k)`: the single uniformly signed pattern still goes through the existing
   `sum() == n` filter, so the output is identical to HEAD's even for parts beyond isize::MAX (R6).
7. Tests that the design writes as one row with both a value and a should_panic are split in two
   (for example `six_vertex_partition_function_keeps_small_values` and `..._rejects_huge_n`,
   `tilings_keep_small_values` and `tilings_reject_n_above_u32`), since one test cannot do both.
8. Line endings: `integer_lists.rs`, `superpartitions.rs`, `t_sequences.rs` are CRLF at HEAD; they
   stay CRLF (the diff shows only the changed lines).
9. Clippy on changed lines: four `is_multiple_of`/`from_ref` hints fixed (ch144_enumerative.rs:87,
   derangements.rs:53, partition_number.rs:417, machine_word_counts.rs:593).

## HEAD check (review 2 m5, review 3 m4)
Built against the HEAD debug rlib `librustmath_combinatorics-ae69d44228029ff8.rlib` (the one the
design's `rev3/rd.sh` uses) with `rustc --test`, sources in `C1-work/headcheck/`.
- `machine_word_counts.rs` at HEAD: removed `number_of_standard_tableaux_matches_the_hook_formula`
  and `hook_length_dimension_is_number_of_standard_tableaux` (new function), the three
  `order_integer` lines of `permutation_order_is_exact`, and in `standard_tableaux_counts` replaced
  `number_of_standard_tableaux` by HEAD's exact `hook_length_dimension`. Each of the 48 remaining
  tests ran alone under `ulimit -v 4000000; timeout 60` (`per_test.txt`). 42 fail: wrong values,
  other panic messages ("attempt to ... with overflow"), `standard_tableaux_counts` times out (HEAD
  forms (10^9)!), `non_decreasing_parking_count_refuses_huge_n` aborts, and
  `universal_cycle_subsets_refuse_lengths_above_usize` and `q_and_t_multinomials_check_the_sum` fail
  under the limit. The 6 that pass pin values that HEAD already had right, as the design's table says:
  `dyck_area_does_not_wrap`, `parking_function_counts`, `six_vertex_partition_function_keeps_small_values`,
  `sunflower_bound_is_checked`, `superpartition_counts_are_checked`, `tilings_keep_small_values`.
- `ch144_counting.rs` at HEAD: `counting_fixture_agrees_with_pari`, `falling_rising_*` and
  `u32_wrappers_*` do not compile (new functions). Of the rest, `rising_factorial_u32_does_not_overflow`,
  `multinomial_sum_does_not_wrap`, `sibling_counts_do_not_truncate`, `stars_and_bars_overflow_panics`,
  `count_partitions_with_max_parts_overflow_panics` fail; the five identity tests pass at HEAD (design:
  "passes") and were not timed there.
- `ch145_number_of_partitions.rs` at HEAD: four tests do not compile (`number_of_partitions`);
  `partition_count_overflow_panics` fails, `partition_count_matches_the_old_dp` passes (it pins HEAD).

## Acceptance (debug builds; one release build for timing only, deleted)
| command | result |
|---|---|
| `cargo build -p rustmath-combinatorics` | ok |
| `cargo test -p rustmath-combinatorics --no-fail-fast` | unit 1509 passed, 0 failed, 7 ignored (baseline 1485: +26 new, -2 moved debruijn helper tests); `ch144_counting` 14 passed (5.5 s); `ch145_number_of_partitions` 6 passed (1.3 s); `machine_word_counts` 50 passed (0.3 s); doctests 189 passed, 11 ignored (baseline 187: +2) |
| `cargo clippy -p rustmath-combinatorics` | exits 101 on exactly the five pre-existing deny lints of design §8.2 (now at `binary_words.rs:336`, `debruijn.rs:101`, `path_tableaux.rs:454`, `super_tableaux.rs:297`, `word.rs:252`; lines moved by this unit's edits above them, code unchanged). With those three lint kinds allowed (plus the pre-existing `erasing_op` in the `composition.rs:775` test), `cargo clippy -p rustmath-combinatorics --all-targets` finishes, and `C1-work/lintmatch.py` (lint locations against `git diff -U0` and the new files) finds 0 lints on changed lines. The unused-import warnings in touched files are all in the HEAD log of review 4 |
| `cargo doc -p rustmath-combinatorics --no-deps` | 0 warnings on changed lines (three escaped after the first run); `target/doc` deleted |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 64 passed, 0 failed (412 s) |
| `cargo test -p rustmath-symmetricfunctions -p rustmath-quantumgroups --no-fail-fast` | 200 passed, 0 failed, 1 ignored |
| `cargo test -p rustmath-groups --no-fail-fast` | 668 passed, 0 failed, 4 ignored |
| `cargo test -p rustmath-algebras -p rustmath-liealgebras -p rustmath-crystals -p rustmath-topology --no-fail-fast` | 1403 passed, 0 failed, 2 ignored |
| `cargo check --workspace --tests` | ok (0 errors), re-run after the last doc edit |

The `--release` test run of §8.2 was dropped (review 4 m2). One release build was made only for a
timing check of the documented costs (`C1-work/timing/timing.rs`, `timing.txt`), then
`target/release` was deleted. Release times here: p(10^6) 4 ms, p(10^8) 0.10 s, p(10^9) 0.81 s
(116993 bits), `fibonacci_signed(10^7)` 0.37 s, `bell_number(2000)` 0.31 s,
`stirling_first(1000, 500)` 0.033 s, `stirling_second(10^7, 10^7-1)` 0.43 s,
`eulerian(1000, 500)` 0.046 s, `motzkin(10^5)` 0.90 s, `schroder_large(10^5)` 1.6 s,
`count_derangements(10^5)` 0.065 s, `factorial(10^5)` 0.029 s, `falling_factorial(2^32-1, 10^5)`
0.14 s, `multinomial(10^5, [5e4, 5e4])` 3 ms, `catalan(10^6)` 0.12 s, f(2001, 1^2000) 1 ms. All at
or below the design's figures (§2.7, §3), so the cost lines in the docs stand.

## API and behaviour changes, callers checked
- New public items: `number_of_partitions`, `NUMBER_OF_PARTITIONS_MAX_N`,
  `falling_factorial_integer`, `rising_factorial_integer`, `falling_factorial_in`,
  `rising_factorial_in`, `number_of_standard_tableaux` (all at the crate root),
  `Permutation::order_integer`, modules `factorials` and `partition_number`. Private module
  `machine_word`. New dependencies rug and num-bigint (already in the graph).
- Every signature is unchanged. Values change only where HEAD was wrong (design §5); the new
  documented panics are exactly those of §5 (plus `replication_number` None for k = 0).
- Callers: `rustmath-jupyter/src/repl.rs` (the 18 imported counting functions, same signatures;
  its 64 tests pass), `rustmath-symmetricfunctions` (`Partition::dimension`, n <= 6) and
  `rustmath-quantumgroups`, `rustmath-groups` (`Permutation::order` test, orders 6 and 2), and
  rustmath-algebras, -liealgebras, -crystals, -topology: their test suites pass (table above).
  `cargo check --workspace --tests` covers rustmath-interfaces.

## Left undone, and why
- Nothing of the design's plan is left undone in code. The trackers, `docs/` and the HANDOFF edits
  are the orchestrator's (below).
- The REPL limits are not recalibrated here (J1b's, design §7); the release spot check above
  confirms the new costs.

## Handoffs

### J1b (REPL)
- `falling_factorial_integer` / `rising_factorial_integer` (any Integer x, u64 k) and
  `number_of_partitions(n: u64)` (n <= `NUMBER_OF_PARTITIONS_MAX_N` = 10^16, documented panic
  above) are in place with the design's names (factorials.rs:130, :149; partition_number.rs:78,
  :40). The u32 `falling_factorial` / `rising_factorial` / `partition_count` keep their signatures,
  so the REPL is unchanged until J1b switches the arms. `partition_count` now panics for n >= 417
  (the REPL limits it to 416).
- The `repl_limits` costs of design §7 now apply: the counting callees are 100x to 4000x faster
  and the Stirling/Eulerian memory is (w + 1)(B/8 + 32) bytes, so the table-size model only
  over-estimates.

### Trackers and HANDOFF (orchestrator; design §8.1 step 12)
Line numbers are at the end of this unit (re-grep after the commit).
- `docs/magma/status_overrides.csv`: `ch018.NumberOfPartitions` partial -> done, path
  `rustmath-combinatorics/src/partition_number.rs:78`, note "Euler (1.2) for n <= 1000, certified
  HRR (Johansson 2012) above, n <= 10^16 (documented panic above); tests
  ch145_number_of_partitions.rs (71 gp rows, H145E1 with GAP), unit hrr_equals_euler_up_to_3000";
  add `ch145.NumberOfPartitions` (done, same path); add `ch145.NumberOfStandardTableaux` (done,
  `partitions.rs:1649`, note "Frobenius formula with cancellation or the hook length formula,
  exact Integer; `hook_length_dimension` is the same function; tests standard_tableaux_counts,
  number_of_standard_tableaux_matches_the_hook_formula (n <= 20), H145E13");
  `ch018.Multinomial` (note: u128 sum, product of binomials, tests multinomial_sum_does_not_wrap);
  optionally `ch058.Order` for permutations (`Permutation::order_integer`, `permutations.rs:153`).
  Then `python3 tools/magma_tracker/build.py`.
- `PROJECT_TRACKER.md` (lines at `dc367a2`): `:1316` classical counting (now `lib.rs:392-740`:
  product-tree factorial, Goetgheluck binomial, product-of-binomials multinomial, Fibonacci/Lucas
  doubling, band-sized rolling Stirling/Eulerian rows, in-place Bell triangle, three-term Motzkin,
  rolling Schroder, derangements by binary splitting; tests ch144_counting.rs with 116 gp rows);
  `:1323` hook length formula (was wrong for most shapes of n >= 21 through usize n!; now exact via
  `number_of_standard_tableaux`, `partitions.rs:1649`; `dimension` is its usize form with a
  documented panic; `hook_length_dimension` is the same count); a new row for
  `number_of_partitions` (`partition_number.rs:78`, 🟢, certified); the `partitions.rs` rows that
  cite `partition_count` (now `:1263`, usize form, panics from 417); new rows or notes for
  `falling_factorial_integer` / `rising_factorial_integer` / `_in` (`factorials.rs:130-207`);
  `:1275` hall polynomials (one-row case wrong too), `:1283` plane partitions (box count wrong),
  `:1287` similarity class types (`centralizer_group_cardinality` wrong), `:1304`
  (`parking_functions.rs` not compiled), a note for `polya_enumeration`; `:1331` posets (order
  ideals and antichains were empty from 31 elements, now exact to 63 with a panic at 64; `mobius`
  is i64, R6); vector partitions (counts checked, 1000-copy cap removed); `integer_lists` (silent
  caps of 100, empty `IntegerListIterator`); `signed_compositions` semantics;
  `Permutation::order` / `order_integer`; Dyck-word statistics (checked i32 sums); `:1291` designs
  (`BlockDesign::new` checks only v <= 10); `partition_to_dyck_path` (§1.8); subset successors
  (`next_binary`, `prev_binary` right for every n).
- `docs/HANDOFF.md:335-342`: S2, S3, S4 and the library part of S10 are done (with the commit). Add
  the design's §8.1 step 12 lines verbatim: the machine-word sweep line (patterns P1-P11 checked;
  unfixed hits bounded by memory, enumeration or 2^32 steps of work, guarded, or in flagged
  modules; known to remain: the R5 size invariant and the R6 signed values; arithmetic outside the
  patterns not checked); still open for J1b: "REPL falling_factorial / rising_factorial still take
  u32 n (switch to falling_factorial_integer / rising_factorial_integer)" and "REPL partitions:
  switch to number_of_partitions (u64 n, at most 10^16)", plus the REPL limit recalibration; the
  new items of §1.8 (revisions 1-3) with their reproductions: `count_plane_partitions_in_box`
  (2,2,2) = 15 (gp 20); `polya_enumeration` uses w^L; `hall_polynomial` one-row case and
  `centralizer_group_cardinality`; `parking_functions.rs` not compiled; the size invariant
  (`Partition::new(vec![1 << 63, 1 << 63]).sum() == 0`); the signed values (`Poset::mobius` of the
  194-element layered poset is 0, truth -2^64; `degree_with_multicharge`,
  `SignedComposition::from_composition`, `AffinePermutation::apply`,
  `SixVertexConfiguration::energy`, `sequence_sum` / `sequence_product` with `unwrap_or(0)`);
  `integer_lists` caps; `IntegerListIterator::next` always None;
  `integer_vectors_with_l1_norm_bounded` duplicates ((2, 1): 12 entries, 5 distinct);
  `signed_compositions` semantics vs its doc and Sage; `BlockDesign::new` checks only v <= 10;
  `ColoredPermutation::order` returns 1000001; outside the crate
  `rustmath-interfaces/src/gap_parser.rs:108` `Permutation::order` lcm overflow and
  `rustmath-algebras/src/temperley_lieb_algebra.rs:317` `catalan_number` wrapping from n = 33;
  `partition_to_dyck_path` never returns for (2) and gives 2 N / 1 E for (1, 1).
- New facts from this implementation for the trackers: `cargo clippy -p rustmath-combinatorics
  --all-targets` also stops on a pre-existing `erasing_op` in the `composition.rs:775` test (only
  the five lib lints were listed); C1 keeps its own `pub(crate)` product tree over `Integer`
  (`factorials.rs:28`) next to I3's private BigUint tree in rustmath-integers (possible merge).

## Fix-up round 1 (review 1)

Review: `impl/C1-combinatorics.review1.md` (not approved: one blocker, four minors). All of them are
fixed as asked; nothing is skipped and there is no disagreement. Logs:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/C1-work/fix1/`
(`run.sh`, `status.txt`, `test_crate.log`, `clippy.log`, `clippy_allowed.log`, `lintmatch.txt`,
`check_ws.log`, `t_*.log`).

### B1 (blocker): a mismatched `max_part` made two public functions loop forever
- Agreed. Removing the 1000-copy cap needs every part to have the length of the vector, and the two
  `_with_max_part` functions built the parts from the caller's `max_part` without checking it.
- `vector_partition.rs:111 fast_vector_partitions_with_max_part` and
  `:286 vector_partitions_with_max_part` now check `actual_max.len() == vector.len()` with
  `assert_eq!` before any work (`:118`, `:303`), so a release build refuses too. Messages:
  "fast_vector_partitions_with_max_part: max_part must have as many components as the vector" and
  "vector_partitions_with_max_part: max_part must have as many components as the vector". Both
  functions document it under `# Panics`. The two loop comments (`:165`, `:322`) now name the
  length invariant next to "no part is the zero vector".
- New tests in `tests/machine_word_counts.rs`:
  `vector_partition_count_refuses_a_longer_max_part` (`:631`, `(&[5], Some(&[1, 1]))`; HEAD
  panicked with an index out of bounds), `vector_partition_list_refuses_a_longer_max_part` (`:641`,
  the same call to the list version), and `vector_partition_count_refuses_a_shorter_max_part`
  (`:649`, `(&[5, 5], Some(&[1]))`; HEAD and the round-0 tree returned 0, a value with no clear
  meaning, as the review notes). All three panic at once (the whole file runs in 0.33 s).
- Behaviour change: a `max_part` of the wrong length is now a documented panic. At HEAD a longer one
  panicked with an index error and a shorter one returned 0. No caller in the workspace passes a
  `max_part` (grep: only the crate's own tests call the two functions, always with equal lengths).

### m1: `integer_lists` saturation comments
- Agreed: `sum(usize::MAX).min_part(1 << 63)` has one list, but the length loop runs over 2^64
  lengths. `integer_lists.rs:293-296` now says that saturation only keeps sum + 1 from wrapping to
  0, and that the bound sum + 1 is one of the silent caps of design §1.8 (with min_part 0 it drops
  the longer lists, and the loop runs over every length up to it even when only one list exists).
  `:336-338` says the same for `length * max_part` and its default of 100. The file stays CRLF.

### m2: `remainder_bound_upper` divisions
- Took the first option: both divisions are now directed, so no dyadic-gap argument is needed.
  `partition_number.rs:303`, `:315`, `:318` hold N, n - 1 and 2n exactly in 128-bit floats;
  `:316 up(p, &nn_exact / &nm1_exact)` and `:319 up(p, &two_n_exact / 3u32)` are each one division
  rounded up at 64 bits. The doc (`:295-298`) says so. `sqrt N` reuses the exact N.
  `remainder_bound_is_below_a_quarter`, `hrr_equals_euler_up_to_3000` and the 71-row gp fixture
  still pass.

### m3: nits
- `tiling.rs:280-281`: "since `Matrix::pow` takes a u32 exponent"; the bit count is gone.
- `symmetric_group_representations.rs:37-38`, `:422-423`, `:457`: "on 64-bit targets" added to the
  three n = 36 statements (`IrreducibleRepresentation::new` had the same wording, so it is fixed
  too).
- `set_system.rs:864`: "The family {{}} shatters the empty set".
- `enumeration.rs:1053-1057`: `k < ways` is right for every position k below usize::MAX; the
  position k = usize::MAX is not reached, since it needs about 2^64 tuples enumerated first
  (design C1 §1.7, rule R2).

### Acceptance (debug)
| command | result |
|---|---|
| `cargo build -p rustmath-combinatorics` | ok |
| `cargo test -p rustmath-combinatorics --no-fail-fast` | unit 1509 passed, 0 failed, 7 ignored; `ch144_counting` 14 passed (6.0 s); `ch145_number_of_partitions` 6 passed (1.6 s); `machine_word_counts` 53 passed (50 + the 3 new, 0.33 s); doctests 189 passed, 11 ignored |
| `cargo clippy -p rustmath-combinatorics` | exits 101 on exactly the five older deny lints (`binary_words.rs:336`, `debruijn.rs:101`, `path_tableaux.rs:454`, `super_tableaux.rs:297`, `word.rs:252`) |
| `cargo clippy -p rustmath-combinatorics --all-targets -- -A clippy::inherent_to_string_shadow_display -A clippy::while_immutable_condition -A clippy::erasing_op` | ok; `lintmatch.py`: 0 lint sites on changed or new lines |
| `cargo check --workspace --tests` | ok, 0 errors |
| `cargo test -p rustmath-symmetricfunctions -p rustmath-quantumgroups --no-fail-fast` | 200 passed, 0 failed, 1 ignored |
| `cargo test -p rustmath-groups --no-fail-fast` | 668 passed, 0 failed, 4 ignored |
| `cargo test -p rustmath-algebras -p rustmath-liealgebras -p rustmath-crystals -p rustmath-topology --no-fail-fast` | 1403 passed, 0 failed, 2 ignored |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 64 passed, 0 failed (429 s) |

The `--release` run of §8.2 stays dropped (review 4 m2; the new checks are `assert_eq!`, which
release builds keep). No release build was made in this round. No change to PROJECT_TRACKER.md,
`docs/`, `rustmath-misc/output.txt` or a `.tmp_dir_*`; no em-dash in added prose.

### Tracker note (adds to the handoff above)
- Vector partitions: counts checked, the 1000-copy cap removed, and a `max_part` whose length
  differs from the vector's is now a documented panic in `fast_vector_partitions_with_max_part`
  and `vector_partitions_with_max_part`.

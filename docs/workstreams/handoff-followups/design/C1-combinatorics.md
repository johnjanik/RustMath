# Design C1-combinatorics (rustmath-combinatorics)

Unit C1 of the HANDOFF "Open follow-ups" (`docs/HANDOFF.md:296-305` at `9c24389`; HEAD `c79a582`
has the same code). Items: S2, S3, S4, S10 (library part), and the list of REPL limits for unit
J1b. Spec: MAGMA Handbook ch. 144 §144.2 (digest
`rustmath-lava/docs/Chapter_144_Enumerative_Combinatorics.md:34-53`), ch. 145 §145.2
`NumberOfPartitions` (`Chapter_145_Partitions_Words_and_Young_Tableaux.md:55`) and §145.4.9
`NumberOfStandardTableaux` (same digest, :299; revision 1), ch. 144 §144.3 (subsets; revision 1),
ch. 58 `Order(g)` (`Chapter_58_Permutation_Groups.md:204`; revision 2),
ch. 18 §18.8
(`Chapter_18_Ring_of_Integers.md:261-275`), and the Pochhammer symbol of ch. 49
`HypergeometricSeries` (`Chapter_49_Power_Laurent_and_Puiseux_Series.md:292`).

Experiments (all under
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/C1-combinatorics/`):

| file | what |
|---|---|
| `count.rs` | verbatim copies of the HEAD loops (`h_*`) and the proposed replacements; `count check` compares them, `count time <fn> <n> [k] [head]` times one call and prints the peak RSS, `count fixture <csv>` checks a gp fixture |
| `hrr.rs` | p(n) by Euler's recurrence and by the Hardy-Ramanujan-Rademacher formula with MPFR enclosures; `hrr check N`, `hrr time n`, `hrr gp file`, `hrr fixture csv` |
| `repro.rs` | the HEAD overflow defects, run in a release and a debug build |
| `conv.rs` | cost of `rug::Integer` to `Integer` conversion |
| `rr.sh`, `rd.sh`, `rd2.sh` | release build against `target/release/deps` (HEAD rlibs); unoptimised builds against the release and the debug rlibs |
| `gp/gen_counting.gp` -> `gp/ch144_counting.csv` | the counting fixture (116 rows) |
| `gp/gen_numbpart.gp` -> `gp/numbpart.csv` | the p(n) fixture (71 rows) |
| `gp/rand_full.gp` -> `gp/rand_full.txt` | 500 random full values of p(n), n < 10^7 (design check only) |
| `gp/siblings.gp`, `gp/h145e1.g` | single reference values (gp) and the H145E1 values (GAP) |
| `paper/johansson2012.txt` | text of Johansson 2012 (arXiv:1205.5991), from which (1.2), (1.4)-(1.8), (2.1) and Algorithm 1 are quoted |
| `rev1/rc.sh` | revision 1: release build against the HEAD release rlib of rustmath-combinatorics |
| `rev1/sweep.rs` ... `rev1/sweep4.rs` | revision 1: HEAD reproductions of the machine-word defects found by the sweep (§1.6), and of the non-overflow defects of §1.8 |
| `rev1/fix.rs` | revision 1: prototypes of `checked_pow_usize`, `binomial_usize_checked`, `number_of_standard_tableaux` (Frobenius formula and the two-branch version), the band-sized rolling rows, the digit conversion, the HRR domain probe; `fix check` runs every comparison |
| `rev1/syt.rs`, `rev1/shadow.rs` | revision 1: `standard_tableaux(p).len()` against `dimension()` for n <= 9; the method resolution of `count` |
| `rev1/rev1.gp` -> `rev1/rev1.txt`, `rev1/maxf.gp` -> `rev1/maxf.txt` | revision 1: gp reference values of §1.6, and the largest f^lambda for each n in 20..=36 |

| `rev2/rc.sh` | revision 2: release build against the HEAD release rlib (same as `rev1/rc.sh`) |
| `rev2/strip.py`, `rev2/funcs.py`, `rev2/flag.py`, `rev2/flag2.py`, `rev2/flag3.py`, `rev2/show.py` | revision 2: the widened sweep. `strip.py` writes the 35576 non-test, non-comment lines to `nontest.txt`; `funcs.py` extracts the 2409 non-test function bodies to `funcs.json`; `flag.py` lists the 76 functions that return a machine word and use `*`, `*=`, `<<`, `pow`, `product` or saturating/wrapping arithmetic; `flag2.py` lists the 37 counting or ranking functions that return a machine word and use only `+`; `flag3.py` lists the 142 other functions that return a machine word and add |
| `rev2/sweep6.rs` -> `rev2/sweep6.txt`; `rev2/misc6.rs`, `shr.rs`, `qmult.rs`, `qm2.rs`, `vpart.rs`, `vpart2.rs`, `repl_num.rs`, `size_inv.rs`, `valrange.rs`, `ck.rs`, `superp.rs`, `dyck.rs`, `mobius.rs`, `tile.rs`, `l1.rs`, `sc.rs`, `wreath.rs` | revision 2: HEAD reproductions of §1.6 items 24-36 and of the new §1.8 entries (run under `ulimit -v 4000000` and `timeout`); outputs in `rev2/sweep6.txt` and `rev2/outputs.txt` |
| `rev2/fix2.rs` | revision 2: prototypes (checked `Combination` rank/unrank against brute force for n <= 12, `subset_mask_count`, checked lcm, the vector-partition count without the cap against a HEAD copy on 154 vectors, the full mask); `bin_fix2` passes in 0.2 s |
| `rev2/rev2.gp` -> `rev2/rev2.txt`, `rev2/rev2.gp.extra` -> `rev2/rev2_extra.txt` | revision 2: gp reference values of §1.6 items 24-36 |

Conventions: "gp" is PARI/GP 2.15 (`gp -q`), "GAP" is GAP 4.12. All timings are release builds
(`rustc -O`, the HEAD release rlibs) on this machine unless marked debug. M = 2^127 - 1.

## Revision 2 (after review 2)

Review: `C1-combinatorics.review2.md`. All experiments of this revision are in
`C1-combinatorics/rev2/`; `bin_fix2` passes (0.2 s).

| review issue | what changed |
|---|---|
| B1 (blocker): the revision-1 sweep was incomplete, and §1.7 and the HANDOFF line said it was complete | Agreed. I added the six functions of the review (§1.6 items 24-28) and swept the crate again with ten stated patterns (§1.6, opening paragraph): the three of revision 1, plus `<<`, `>>`, `.pow(`, `.product()`, `.sum()`, narrowing `as` casts other than `as u32`, every function that returns a machine word and multiplies, shifts or saturates (76 functions, `rev2/flag.py`), every counting or ranking function that returns a machine word (37 more, `rev2/flag2.py`), every other function that returns a machine word and adds (142 more, `rev2/flag3.py`), and private helpers named like `binomial`, `factorial`, `count_`. The new patterns found more sites of the same class, now items 29-36: `Permutation::order` (wrong when the order fits but an intermediate product does not, and from 381 points), the 2^n mask loops of 13 public functions (an `i32` literal makes most of them return an empty or one-element result from n = 31), `q_multinomial` / `t_multinomial` (the twin of the `multinomial` sum), `usize as i64` into `Integer` in four integer-vector functions, the pruning bounds of `integer_lists`, the vector-partition counts (p(417) wraps after 0.1 s, and a cap of 1000 copies drops partitions), `BlockDesign::replication_number`, and the Dyck-word statistics, which are counts in i32 that wrap from semilength 65537 (item 36). They also found right shifts past `usize::BITS` in `BinaryWord::from_usize` and `Subset::from_bitvector`, which are folded into items 27 and 28. Each is reproduced at HEAD with a gp value. Two of the new fixes go slightly beyond a checked operation, because they remove the overflow instead of refusing: `count_superpartitions` becomes an O(n^2) checked DP of its own generating function, and `signed_compositions(_k)` stop enumerating 2^k sign patterns of which only one can pass. §1.7 now states eight rules (R1-R8) and lists every remaining hit of every pattern under one of them. §1.8 adds the defects of other classes that the new sweep found (among them a size invariant that no type checks, value-range limits of signed machine-word APIs, an iterator that never yields, and silent caps of 100). The HANDOFF line (§8.1 step 12) now names the patterns and the rules and lists what is known to remain. It no longer claims completeness. |
| m1: `binomial_usize_checked` omits j > m | Fixed in the text: it returns `Some(0)` first when j > m, as the prototype does. Unit test C(3, 5) = 0 (§6.6). |
| m2: `TupleIterator` is an `ExactSizeIterator` | Kept the impl (no caller of `len()` in the workspace) and documented it: `len()` panics when n^k > usize::MAX, as `CartesianProduct` already does (`enumeration.rs:917-929`). `PermutationGrayCodeIterator` is not an `ExactSizeIterator`, so its new `(0, None)` hint needs no such note. |
| m3: two behaviour changes not listed | §5 now lists `all_irreducible_representations(n)` and `SymmetricGroupCharacterTable::new(n)` (they panic for n >= 36 through `IrreducibleRepresentation::new`), and the I3 panic "binomial: C(n, k) has more than 2^32 bits" that `count_k_subsets`, `count_lattice_paths_ne`, `catalan` and `multinomial` can reach for absurd arguments. Their `# Panics` docs say so (§8.1 steps 5, 10). |
| m4: the REPL half of S10 now has an owner | Agreed. J1b's revision takes both REPL switches (`J1b-repl-values-and-limits.md:18`, its §5 and plan step 7). §4.2, §7 and §8.1 step 12 now say so. C1 still records the HANDOFF line as open, because J1b lands later and closes it. |
| m5: a should_panic test exhausts memory on HEAD | §6.5 and §8.2: when the implementer confirms that the new tests fail on HEAD, `universal_cycle_subsets(70, 35)` and `q_multinomial(1, &[u32::MAX, 2])` (and `t_multinomial`) run under `ulimit -v 4000000` and `timeout 60`, or are skipped. At HEAD the first runs out of memory; the other two did not return within 30 s under that limit (`rev2/qmult.rs`). |

## Revision 1 (after review 1)

Review: `C1-combinatorics.review1.md`. All experiments of this revision are in
`C1-combinatorics/rev1/`; `bin_fix check` passes (0.7 s).

| review issue | what changed |
|---|---|
| B1 (blocker): §1.7 dismissed public counts of the S2 class | Agreed. I swept the whole crate: every non-test `as u32`, every `saturating_*` / `wrapping_*`, every `unwrap_or(0)` on a conversion, and every closed-form count that returns a machine word. §1.6 now fixes the five functions of the review (items 7-11) and 17 more public functions found by the sweep (items 12-22: `Tuple::rank` / `unrank`, `count_integer_vectors_in_box`, `Partition::dimension`, which is wrong for most partitions of n >= 21, `burnside_lemma`, `count_fixed_colorings`, `count_tilings`, `Subset::rank_k_subset` / `unrank_k_subset` / `rank_binary`, `DeBruijnSequence::new` / `is_valid`, `universal_cycle_subsets` / `_permutations`, `OrthogonalArray::new`, `SetSystem::sunflower_bound` / `exceeds_sunflower_bound`), plus the private binomial of `InfiniteCartesianProduct` (item 23). Each one is reproduced at HEAD (`rev1/sweep*.rs`) with a gp reference value. A new private module `src/machine_word.rs` holds the checked helpers they share. §1.7 now classifies every remaining site with its reason. §1.8 lists the defects that the sweep found but that are not integer-width defects (four wrong values and a file that is not compiled); they are reported as new HANDOFF items. The HANDOFF line of §8.1 step 12 is rewritten to match. (Revision 2: this sweep was not complete. Revision 2, B1, widens it.) |
| m1: HRR domain unstated | §2.2, §2.4 step 0, §2.5: `number_of_partitions` panics (documented) for n > 10^16 and checks the MPFR exponent range before any work. Measured: cosh(C) is finite at 64 bits for n = 8·10^16 and +inf for n = 8.5·10^16 (`bin_fix hrrdom`, `exp_max() = 2^30 - 1`). At 10^16, log2 of the k = 1 term is 3.7·10^8, a factor 2.9 below `exp_max`. Since 24n - 1 < 2^64 in the domain, v is a u64 again. The process-wide exponent range is not touched. |
| m2: rolling rows hold k + 1 entries | Took the recommendation. §3.3 now keeps a `VecDeque` of at most w + 1 entries (w is the band width), indexed by j - base. Prototype: identical values to the HEAD tables (n <= 60, k <= 62, seven larger pairs). The deque never exceeds w + 1 entries. `stirling_second(10^8, 10^8 - 1)` takes 7.9 s at 2.7 MB; the k + 1 row would need 3.2 GB. Memory at (10^4, 5000) drops from 64 MB to 28 MB, and the time is unchanged. §7 gives J1b the memory model (w + 1)(B/8 + 32) bytes. |
| m3: REPL half of S10 has no owner | Agreed (J1b's design never names `falling_factorial_integer` / `rising_factorial_integer`; J1a moves only the `pochhammer` name). §4.2, §7 and §8.1 step 12 now keep "REPL falling_factorial / rising_factorial still take u32 n" open in HANDOFF, and the summary reports the gap to the orchestrator. (Revision 2: J1b's revision has since taken both switches. See Revision 2, m4.) |
| m4: `partition_count_wrapper` is circular | §6.2: the test now compares `partition_count(n)` with a test-local copy of HEAD's usize DP for every n <= 416 (64-bit). Checked in the prototype: the HEAD DP equals Euler's recurrence for n <= 416. |
| m5: catalan / schroder u32 fixes untested | §1.5: private `schroder_step_factors(i: u64) -> (u64, u64, u64)` with a unit test at i = 715827884 (4294967301; HEAD 5) and i = 2147483649 (12884901891; HEAD 3), checked in the prototype. In `catalan` the only arithmetic is `2 * u64::from(n)` and `u64::from(n) + 1`, which cannot overflow for any u32 n. A computed value would need 2^32 bits, so the fix stays checked by type. |
| m6: decimal conversion is quadratic | §2.4 step 6: the digit route `significant_digits::<u8>()` + `write_digits(.., Order::Lsf)` + `BigInt::from_bytes_le`. The review suggested `to_digits`, but rug offers it only with its `std` feature, which the dependency declaration (`default-features = false`) leaves off. Measured: 0.17 ms against 20.7 ms at 370 kbits, and 1.9 ms against 1.7 s at 4 Mbits. |
| m7: nits | Cargo.lock is listed in §8.1 step 1. The `partition_count` doc says "no usize of 64 bits or fewer". The shadowing of `TupleIterator::count` is documented in the fix (§1.6) and in its tests; `rev1/shadow.rs` confirms that `it.count()` on a value calls `Iterator::count`. On product trees: I3's `product_tree` is private and works over BigUint. Making it public would add API to rustmath-integers inside I3's unit, so C1 keeps one `pub(crate)` tree over `Integer` (§3.7). The possible merge is reported to the orchestrator. |

## Verdicts

| item | verdict |
|---|---|
| S2 u32 overflow (lib.rs:430 catalan, :498 rising, :410 multinomial, :749 schroder, partitions.rs:1261) | fix all five; also every integer-width defect of the same class that the sweep of the crate with the ten patterns of §1.6 found in a public function (§1.6 items 1-35: 28 public functions in 13 files in revision 1, and 42 more in 20 files in revision 2; among them `Partition::dimension`, which is wrong for most partitions of n >= 21, with a new exact `number_of_standard_tableaux`, and `Permutation::order`, with a new `order_integer`). §1.7 classifies every remaining hit of those patterns under eight stated rules; it does not claim more than that |
| S3 Integer-valued p(n) | fix: new `number_of_partitions(n: u64) -> Integer` (Euler's recurrence for n <= 1000, certified Hardy-Ramanujan-Rademacher above, documented panic above n = 10^16); p(10^9) in 0.94 s |
| S4 faster callees | fix: Fibonacci doubling, in-place Bell triangle, rolling Stirling and Eulerian rows in a band-sized buffer, three-term Motzkin recurrence, rolling Schroder, derangements by binary splitting, product-tree factorials, binomial and catalan through `SageInteger::binomial` (I3), multinomial as a product of binomials. 100x to 4000x at the current REPL limits |
| S10 (library) falling/rising factorial take u32 n only | fix: `falling_factorial_integer` / `rising_factorial_integer` (any Integer x) and `falling_factorial_in` / `rising_factorial_in` (any ring element, Sage semantics); the u32 functions stay as wrappers |
| REPL limits for J1b | §7: every limit that depends on these callees, with old and new costs |

Nothing is declared out of scope. Not changed, with reasons (§1.7): casts whose value is the size of
an object already in memory, or that a loop reaches only after 2^32 steps of big-integer work, and
casts inside modules that PROJECT_TRACKER already flags as broken (`ranking.rs`:1273,
`hall_polynomial.rs`:1264, `similarity_class_type.rs`:1276, `ncsym.rs`:1270,
`symmetric_group_representations.rs` `verify_orthogonality`:1257); those casts belong to the
modules' rewrites. Revision 2 names these rules (R1-R8, §1.7) and adds two kinds of
integer-width defect that are reported, not fixed: the size invariant of partition-like types
(R5), which no constructor checks, and signed machine-word values whose type is part of an API
(R6). Each goes to HANDOFF as one new item with reproductions (§1.8).
The sweep also found wrong values that are not integer-width defects
(`count_plane_partitions_in_box`, `polya_enumeration`, the one-row case of `hall_polynomial`,
`centralizer_group_cardinality`; revision 2: `IntegerListIterator`, which never yields, the silent
caps of 100 in `integer_lists`, duplicates in `integer_vectors_with_l1_norm_bounded`, and the
semantics of `signed_compositions`), and a source file that is not compiled. They are outside the four items of this unit and go to
HANDOFF as new items (§1.8).

The handbook has no worked example for the ch. 144 counting functions (the only one, H144E1, is
about `Subsets`). Ch. 145 has H145E1 (partitions of n and the classes of Sym(n)); it is ported in
§6 with GAP values.

---

## 0. Baseline

`cargo test -p rustmath-combinatorics --no-fail-fast` at HEAD: 1485 unit tests passed, 7 ignored;
187 doctests passed, 11 ignored. The crate has no `tests/` directory yet.

Callers of the affected functions (grep of the whole workspace, `--include=*.rs`, excluding
`target`):

- Outside the crate, only `rustmath-jupyter/src/repl.rs:18-23` imports counting functions:
  `catalan, fibonacci_signed, lucas_signed, bell_number, stirling_first, stirling_second,
  multinomial, falling_factorial, rising_factorial, eulerian, narayana, motzkin, delannoy,
  delannoy_central, schroder_large, schroder_small, partition_count, count_derangements`. Every
  one keeps its signature (§5). The other dependants (`rustmath-symmetricfunctions`,
  `-quantumgroups`, `-algebras`, `-groups`, `-liealgebras`, `-crystals`, `-topology`,
  `-interfaces`) use only `Partition`, `partitions`, `Composition`, `Permutation`,
  `SetPartition`, `PerfectMatching`, `PartitionTuple`, `Tableau`.
- Inside the crate: `crate::binomial` (`enumeration.rs:613`, `:1190`, `expnums.rs:349`, `:426`,
  `:447`, `:565`, `:609`, `integer_vectors.rs:248`, `path_tableaux.rs:293`, `ranking.rs:6`,
  `species.rs:224`, `:506`, `subset.rs:649`, `ch144_enumerative.rs:27`, `narayana`, `delannoy`,
  `delannoy_central`, `catalan`); `crate::factorial` (`derangements.rs:47`, `:52`, `expnums.rs:239`,
  `multinomial`); `crate::catalan` (`path_tableaux.rs:300`, `ranking.rs:422`, `species.rs:479`,
  `tamari_lattice.rs:690`); `crate::stirling_second` (`species.rs:263`, `expnums.rs:166`,
  `:238`, `bell_number`); `crate::stirling_first` (`expnums.rs:180`); `crate::bell_number`
  (`expnums.rs:152`, `set_partition.rs:537` test); `count_derangements` (`expnums.rs:195`,
  derangements tests). `debruijn.rs:445` has its own private usize `falling_factorial`; revision 1
  replaces it with a checked helper (§1.6 item 20). Values change only where HEAD was wrong (§5).
- Revision 1 (the functions of §1.6 items 7-23): the only caller outside rustmath-combinatorics is
  the test `rustmath-symmetricfunctions/src/classical_bases.rs:806` (`Partition::dimension`,
  n <= 6). Inside the crate, `IrreducibleRepresentation::new` (`symmetric_group_representations.rs:35`)
  and the test `specht_module.rs:673` call `Partition::dimension`; `exceeds_sunflower_bound` calls
  `sunflower_bound`; `TupleIterator::size_hint` calls `count`.
- Revision 2 (the functions of §1.6 items 24-36; grep of the 30 files outside the crate that name
  `rustmath_combinatorics`, list in `rev2/dependents.txt`): no non-test caller outside the crate.
  `Permutation::order` is called by the test `rustmath-groups/src/permutation_group.rs:441-442`
  (orders 6 and 2, unchanged). `rustmath-interfaces/src/gap_parser.rs:108` has its own
  `Permutation::order` with the same `a * b / gcd` lcm (`:321-326`); it is outside this crate and is
  reported (§1.8). Inside the crate: `lyndon_words` and `binary_words_with_weight`
  (`binary_words.rs:393`, `:401`) call `all_binary_words`; `Poset::width` (`posets.rs:645`) calls
  `all_antichains`; `SetSystem::hereditary_closure` (`set_system.rs:373`) and `:337` call `shadow`;
  `vc_dimension` calls `is_shattered`; `count_distinct_subwords` (`subword.rs:187`) calls
  `all_subwords`; the Grossman-Larson coproduct (`grossman_larson.rs:433`) calls `admissible_cuts`.
  The Dyck-word statistics of item 36 have no caller outside `dyck_word.rs`. Their values change
  only where HEAD was wrong.

---

## 1. S2: u32 overflow

### 1.1 Defects at HEAD (reproduced)

`repro.rs` copies the HEAD expressions and runs them in a release and a debug build:

| site | input | release | debug | reference |
|---|---|---|---|---|
| `lib.rs:498` `Integer::from(n + i)` in `rising_factorial` | `rising_factorial(4294967295, 2)` | `0` (n + 1 wraps to 0) | panic "attempt to add with overflow" | 18446744069414584320 (gp `(2^32-1)*2^32`) |
| `lib.rs:430` `binomial(2 * n, n)` in `catalan` | `catalan(2147483648)` | 2n wraps to 0, `binomial(0, 2^31) = 0`, result 0 | panic "attempt to multiply with overflow" | C(2^32, 2^31)/(2^31+1), not computable (2^32 bits) |
| `lib.rs:410` `let sum: u32 = ks.iter().sum()` in `multinomial` | `multinomial(1, &[4294967295, 2])` | sum wraps to 1 = n, so it computes `factorial(4294967295)` (does not finish) | panic "attempt to add with overflow" | 0 (the sum 4294967297 is not n) |
| same | `multinomial(4294967295, &[4294967294, 1])` | computes `(2^32-1)!` (does not finish) | same | 4294967295 (gp `binomial(2^32-1, 1)`) |
| `lib.rs:749` `3 * (2 * i - 1) as u32` in `schroder_large` | i = 715827884 | factor 5 (wrapped) | panic "attempt to multiply with overflow" | 3(2i-1) = 4294967301 |
| same, `as` truncation | i = 2147483649 | factor 3 instead of 12884901891 | the same silent 3 (an `as` cast never panics) | |
| same, `(i + 1) as u32` | i = 4294967295 | divisor 0 (division-by-zero panic; by inspection) | same | |
| `partitions.rs:1272` `dp[sum] += dp[sum - part]` in `partition_count` | `partition_count(417)` | 541220193622112941 | panic "attempt to add with overflow" | 18987964267331664557 (gp `numbpart(417)`) |

The catalan and schroder inputs need results of 2^32 and about 2^31 bits (and schroder keeps n + 1
such values at HEAD), so those two defects cannot be shown by a test with a computed result. They are
fixed by doing the arithmetic in u64 (§1.3, §1.5). The Schroder multipliers are pinned by the unit
test of §1.5, which HEAD's expressions fail; `catalan` is checked by type (§1.5). Their value tests
cover the new code paths.

Also found in the same functions (the `vec!` sizes): `vec![..; (n + 1) as usize]` in `motzkin`
(`lib.rs:708`), `schroder_large` (`:742`), `stirling_first`/`stirling_second`/`eulerian`
(`:518`, `:550`, `:603`, n + 1 and k + 1 in u32) overflow at n = u32::MAX. The rewrites of §3 remove
these allocations.

### 1.2 rising_factorial (`lib.rs:495-501`)

Becomes a wrapper: `rising_factorial(n: u32, k: u32) -> Integer` returns
`rising_factorial_integer(&Integer::from(n), u64::from(k))` (§4). Nothing is added in u32.

### 1.3 catalan (`lib.rs:425-431`)

```rust
pub fn catalan(n: u32) -> Integer {
    let n = u64::from(n);
    // C_n = C(2n, n) / (n + 1); 2n and n + 1 are formed in u64 (no overflow for any u32 n).
    <Integer as SageInteger>::binomial(&Integer::from(2 * n), &Integer::from(n)) / Integer::from(n + 1)
}
```

The division is exact and by a one-word divisor (num-bigint's single-digit path, O(size)). The
MAGMA definition `C_{n+1} = C_n (4n+2)/(n+2)` (digest :42) is the same sequence; it is tested as an
identity (§6).

### 1.4 multinomial (`lib.rs:406-420`)

```rust
pub fn multinomial(n: u32, ks: &[u32]) -> Integer {
    let sum: u128 = ks.iter().map(|&k| u128::from(k)).sum();   // cannot wrap: len < 2^64
    if sum != u128::from(n) {
        return Integer::zero();   // unchanged convention (documented): coefficient of a
    }                             // monomial of another degree
    // n! / (k_1! ... k_m!) = prod_i C(s_i, k_i), s_i = k_(1) + ... + k_(i), largest k first.
    let mut sorted: Vec<u64> = ks.iter().map(|&k| u64::from(k)).collect();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let mut s = 0u64;
    let parts: Vec<Integer> = sorted.into_iter().map(|k| { s += k; <Integer as SageInteger>::binomial(&Integer::from(s), &Integer::from(k)) }).collect();
    factorials::product_tree(parts)
}
```

Correctness: `n!/(k_1!...k_m!) = prod_i s_i!/(s_(i-1)! k_i!) = prod_i C(s_i, k_i)` (telescoping).
With the largest k first, `s_i - k_i = s_(i-1) >= k_(1) >= k_i`, so each binomial runs over its
lower index k_i, and the total window is n - k_max: `multinomial(2^32-1, [2^32-2, 1])` costs one
C(2^32-1, 1). The zero-for-wrong-sum convention is kept (the existing test `lib.rs:937` and the
REPL, which checks the sum itself first).

### 1.5 schroder_large (`lib.rs:734-753`)

Rolling two values, every multiplier in u64 (§3.6). For i <= u32::MAX + 1, 3(2i - 1) < 2^35,
i - 2 and i + 1 fit u64; nothing is cast to u32. The multipliers come from one private function, so
that a unit test pins them:

```rust
/// (3(2i - 1), i - 2, i + 1): the multipliers of (i+1) S_i = 3(2i-1) S_(i-1) - (i-2) S_(i-2), i >= 2.
fn schroder_step_factors(i: u64) -> (u64, u64, u64)
```

Unit test `schroder_step_factors_do_not_wrap`: i = 715827884 gives 4294967301 (HEAD's
`3 * (2 * i - 1) as u32` gives 5), i = 2147483649 gives 12884901891 (HEAD 3), i = 2 gives (9, 0, 3).
Checked in `rev1/fix.rs`. `catalan` (§1.3) has no such helper: its only arithmetic is
`2 * u64::from(n)` and `u64::from(n) + 1`, which cannot overflow for a u32 n, and a computed value
would need 2^32 bits.

### 1.6 The siblings of the same class (sweep of the crate)

The class. A count, a rank, a size bound or an enumeration bound (the number of objects of some
kind, the position of an object in a listing, or the bound of a loop over all of them) that a
function computes in a machine word, where the true value can exceed the word. A site is fixed
here when a public function reaches it with a small input (a few machine words, or a stored object
of at most a few megabytes, such as the 381-point permutation of item 29 or the 128 KB Dyck word of
item 36) and then returns a wrong value (or wraps) without first doing 2^32 steps of work. Every
other site of the patterns below is in §1.7 under one of its rules.

Patterns, each run over the non-test, non-comment lines of `rustmath-combinatorics/src/*.rs` at
HEAD (`rev2/strip.py` writes them to `rev2/nontest.txt`, 35576 lines; a test module is taken to
start at the first `#[cfg(test)]` followed by `mod`, and no file has an item after it):

- P1 (revision 1): every `as u32` (110 lines in `rev2/nontest.txt`; revision 1 counted 125 with a
  different test-module cut).
- P2 (revision 1): every `saturating_*` and `wrapping_*`.
- P3 (revision 1): every `unwrap_or(0)` or `unwrap_or(usize::MAX)` after a conversion or a checked
  operation.
- P4 (revision 2): every `<<` (43 lines) and every `>>` that is a shift (13 lines).
- P5 (revision 2): every `.pow(`, `checked_pow`, `saturating_pow` (25 lines).
- P6 (revision 2): every `.product()` or `product::<` over machine words (3 lines; the other 14
  matches are methods named `product` of algebra elements).
- P7 (revision 2): every `.sum()` or `sum::<` (130 lines).
- P8 (revision 2): every `as i32`, `as i64`, `as isize`, `as u8`, `as u64`, `as usize`, `as u128`
  (214 lines).
- P9 (revision 2): every function that returns a machine word, or a tuple, `Option` or `Vec` of one,
  and whose body uses `*`, `*=`, `<<`, `pow`, `product` or saturating or wrapping arithmetic (76
  functions, `rev2/flag.py`), and every function of that kind whose name says it counts, ranks or
  sizes (`count`, `rank`, `number`, `total`, `size`, `dim`, `index`, ...) and that uses only `+`
  (37 functions, `rev2/flag2.py`), and every other function that returns a machine word and adds
  (142 functions, `rev2/flag3.py`).
- P10 (revision 2): private helpers by name (`fn .*(binomial|factorial|catalan|choose|pow|power|
  multinomial|count_|num_|number_)`) that return a machine word, and every `fn rank`, `fn unrank`,
  `fn position`.

Every hit is either an item below or a row of §1.7. Arithmetic that none of these patterns matches
(for example a plain `a * b` in a function that returns `Integer` or `bool`, outside P9) is not
claimed; §8.1 step 12 says so in HANDOFF.

Shared helpers: a new private module `src/machine_word.rs` (`mod machine_word;` in lib.rs, not
`pub`). Header: MAGMA ch. 144 §144.2 (the counting functions it serves).

```rust
/// base^exp, or None when it exceeds usize::MAX. Any exp for base 0 and 1.
pub(crate) fn checked_pow_usize(base: usize, exp: usize) -> Option<usize>
/// C(m, j), or None when it exceeds usize::MAX; 0 when j > m. m is a u128 so that n + k - 1 of
/// two usize values fits. At most min(j, m - j, usize::BITS + 1) steps.
pub(crate) fn binomial_usize_checked(m: u128, j: u128) -> Option<usize>
/// n (n-1) ... (n-k+1), or None when it exceeds usize::MAX. 0 when k > n.
pub(crate) fn falling_factorial_usize_checked(n: usize, k: usize) -> Option<usize>
/// x as a u32 argument; panics with "{what}: {x} exceeds u32::MAX".
pub(crate) fn u32_arg(x: usize, what: &str) -> u32
/// 2^n, the number of subsets (masks) of an n-element set, as a usize (revision 2). Panics with
/// "{what}: 2^{n} subsets exceed usize::MAX" when n >= usize::BITS.
pub(crate) fn subset_mask_count(n: usize, what: &str) -> usize
```

- `checked_pow_usize`: base 0 gives 1 for exp = 0 and 0 otherwise, base 1 gives 1; otherwise
  `u32::try_from(exp)` (an exponent of 2^32 or more overflows any usize when base >= 2) and then
  `checked_pow`.
- `subset_mask_count`: `u32::try_from(n).ok().and_then(|s| 1usize.checked_shl(s))`, else the
  panic. `checked_shl` fails exactly when the shift is at least `usize::BITS`, and 2^(BITS-1) fits,
  so the result is 2^n whenever 2^n <= usize::MAX. Unit tests (63) = 2^63 and a panic at 64
  (`rev2/fix2.rs`).
- `binomial_usize_checked`: if j > m, return `Some(0)` (no j-subset of an m-set; this also keeps
  m - j from underflowing). Otherwise j <- min(j, m - j); c_0 = 1 and c_i = c_(i-1) (m - j + i) / i =
  C(m - j + i, i), an exact division. The sequence does not decrease ((m - j + i)/i >= 1), so
  once c_i > usize::MAX the result is too, and the function returns None. Since m - j >= j >= i,
  c_i >= C(2i, i) >= 2^i, so c_i <= usize::MAX forces i < usize::BITS: the loop ends within
  usize::BITS + 1 steps, whatever m and j are. If c_(i-1) (m - j + i) overflows u128, then
  c_i >= 2^128 / i >= 2^128 / 65 > usize::MAX, so None is again right. Checked against exact
  binomials for m <= 200 and every j, and for m in {2^40, 2^63, 2^64, 2^70 + 3} with j <= 70
  (`bin_fix check`); C(67, 33) = 14226520737620288370 fits, C(68, 34) does not (gp).
- `falling_factorial_usize_checked`: `checked_mul` over the k factors. Every factor except possibly
  the last is at least 2, so the product at least doubles per step and the loop ends within
  usize::BITS + 1 steps.

Fixed functions (HEAD outputs from `repro.rs`, `review1/sib*.rs`, `rev1/sweep*.rs`, release; gp
values from `gp/siblings.gp` and `rev1/rev1.gp`):

1. `partitions.rs:1261 partition_count(n: usize) -> usize` keeps its signature (the REPL calls it)
   and becomes honest: `number_of_partitions` (§2) for n <= 1000, converted with `to_usize()`; if
   the value does not fit, or n > 1000 (p(1000) > 2^64, so no usize of 64 bits or fewer can hold
   it), it panics with "partition_count: p(n) exceeds usize::MAX; use number_of_partitions" (a
   documented `# Panics`). HEAD wraps (release) or panics with an overflow message (debug).
2. `partitions.rs:1587 count_partitions_with_max_parts(n, k) -> usize`: the same usize DP
   (`dp[i][j] += dp[i - j][j]`). Use `checked_add` and panic with
   "count_partitions_with_max_parts: the count exceeds usize::MAX" (documented `# Panics`).
3. `subset.rs:649 count_k_subsets(n: usize, k: usize) -> Integer`: `crate::binomial(n as u32, k as u32)`
   truncates n and k. Use `<Integer as SageInteger>::binomial(&Integer::from(n as u64),
   &Integer::from(k as u64))` (usize is at most 64 bits on every Rust target). HEAD:
   `count_k_subsets(4294967298, 2) = C(2, 2) = 1`; gp `binomial(2^32+2, 2)` = 9223372043297226753.
4. `path_tableaux.rs:293 count_lattice_paths_ne(m: u32, n: u32) -> Integer`: `m + n` in u32.
   Use `binomial` of `u64::from(m) + u64::from(n)` and m via `SageInteger::binomial`. HEAD:
   `count_lattice_paths_ne(u32::MAX, 1)` panics (debug) or returns `binomial(0, u32::MAX) = 0`
   (release); gp `binomial(2^32, 1)` = 4294967296.
5. `enumeration.rs:1181-1193 stars_and_bars(n: usize, k: usize) -> usize`: `(n + k - 1) as u32`
   truncates, `n + k - 1` can overflow usize, and `.to_usize().unwrap_or(0)` turns an overflow into
   a count of 0 (a facade). New body: `binomial_usize_checked(n as u128 + k as u128 - 1, k as u128 - 1)`
   with `.expect("stars_and_bars: C(n+k-1, k-1) exceeds usize::MAX")` (documented). The check costs
   at most 65 steps, so (2^40, 2^40) panics at once instead of first forming a 2^41-bit binomial.
   HEAD: `stars_and_bars(4294967296, 2) = C(1, 1) = 1`; gp `binomial(2^32+1, 1)` = 4294967297;
   `stars_and_bars(1000, 1000)` returns 0 at HEAD (C(1999, 999) has 1994 bits, gp
   `logint(binomial(1999,999),2)+1`).
6. `enumeration.rs:613` `CombinationGrayCodeIterator::size_hint`: `crate::binomial(self.n as u32,
   self.k as u32)` truncates, so the upper bound can be too small. Return
   `(0, binomial_usize_checked(n as u128, k as u128))` (None when it does not fit; no big binomial
   is formed). Not testable: `new(n, k)` allocates n bytes, so n > 2^32 needs 4 GB.
7. `integer_vectors.rs:239-251 count_integer_vectors_with_sum(length, sum) -> usize` (re-exported
   `lib.rs:256`): the same body as `stars_and_bars`. HEAD (3, 2^32) -> 1, (1000, 1000) -> 0; gp
   `binomial(2^32+2, 2)` = 9223372043297226753. Fix: `binomial_usize_checked(sum as u128 + length
   as u128 - 1, length as u128 - 1)` with `.expect("count_integer_vectors_with_sum:
   C(sum+length-1, length-1) exceeds usize::MAX")`, documented.
8. `six_vertex_model.rs:528-535 partition_function(n: usize) -> Integer` (re-exported as
   `six_vertex_partition_function`): `count_dwbc_configurations(n as u32)`. HEAD (2^32 + 3) -> 7 =
   A(3), at once. Fix: `count_dwbc_configurations(u32_arg(n, "six_vertex partition_function"))`,
   documented: A(n) for n >= 2^32 has about 0.38 n^2 > 7·10^18 bits.
9. `parking_function.rs:311-316 count_parking_functions(n) -> usize`: `(n + 1).pow((n - 1) as u32)`
   wraps (release) or panics (debug). HEAD 17 -> 10759066654336942080; gp `18^16` =
   121439531096594251776. Fix: n = 0 gives 1; otherwise `n.checked_add(1).and_then(|b|
   checked_pow_usize(b, n - 1)).expect("count_parking_functions: (n+1)^(n-1) exceeds usize::MAX")`.
   16 gives 2862423051509815793 (gp `17^15`).
10. `parking_function.rs:321-345 count_non_decreasing_parking_functions(n) -> usize`: a private
    O(n^2) usize Catalan DP that wraps. HEAD 37 -> 9057316177202639132; gp `binomial(74,37)/38` =
    45950804324621742364. Fix: delete `catalan_number`; panic when n >= usize::BITS, otherwise
    `crate::catalan(n as u32).to_usize()` with the same panic message ("C_n exceeds usize::MAX").
    The early bound is valid because C_n > 2^n for n >= 5 (C_5 = 42 > 32, and C_(n+1)/C_n =
    (4n+2)/(n+2) >= 2 for n >= 1; checked for n < 400), so no Catalan number of index
    >= usize::BITS fits. It keeps `catalan` from building a 2^33-bit number for n = 2^32 - 1 before
    failing. 36 gives 11959798385860453492 (gp `binomial(72,36)/37`).
11. `tuple.rs:252-257 TupleIterator::count(&self) -> usize` and its `size_hint` (`:314-334`):
    `self.n.saturating_pow(self.k as u32)` saturates and truncates k. HEAD (`rev1/sweep2.rs`):
    (2, 64) -> 18446744073709551615, (3, 41) -> the same (gp `3^41` = 36472996377170786403),
    (0, 2^32) -> 1 (0^0 after the truncation; the iterator yields nothing). `size_hint` then reports
    a wrong exact size. Fix: private `fn checked_total(&self) -> Option<usize>` (1 for k = 0, else
    `checked_pow_usize(n, k)`); `count` returns it or panics with "TupleIterator::count: n^k
    exceeds usize::MAX" (documented); `size_hint` returns `(usize::MAX, None)` on None and never
    panics. The doc of `count` says: `it.count()` on a value resolves to `Iterator::count`, which
    enumerates every tuple; write `TupleIterator::count(&it)` or `(&it).count()` for this closed
    form (`rev1/shadow.rs` shows the resolution on a toy type: by value 3, by reference 1000). The
    existing test `tuple.rs:504` calls `Iterator::count` and stays; the new tests call the inherent
    method explicitly. `TupleIterator` implements `ExactSizeIterator` (`tuple.rs:335`), so with the
    `(usize::MAX, None)` hint `len()` fails std's `assert_eq!(upper, Some(lower))`. The impl stays
    (no caller of `len()` in the workspace; `CartesianProduct` does the same,
    `enumeration.rs:917-929`), and the type's doc says: "`len()` panics when n^k > usize::MAX".
12. `tuple.rs:63-75 Tuple::rank(&self) -> usize`: `rank += indices[i] * power; power *= n` in usize;
    the last `power *= n` overflows even when the rank fits. HEAD: indices [1, 1, 1] with n = 2^40
    -> 1099511627777 (the rank 2^80 + 2^40 + 1 does not fit). Fix: Horner from the most significant
    index with `checked_mul` / `checked_add`; every partial value is at most the rank, so it fails
    only when the rank does not fit; panic "Tuple::rank: the rank exceeds usize::MAX".
    `Tuple::unrank(rank, n, k)` (`:81-89`): `n.checked_pow(k as u32)` truncates k. Use
    `checked_pow_usize(n, k)`; None means n^k > usize::MAX >= rank, so every rank is valid.
13. `integer_vectors.rs:404-417 count_integer_vectors_in_box(min, max) -> usize`: `(mx - mn + 1)`
    overflows i64 and the usize product wraps. HEAD ([0; 3], [2^32 - 1; 3]) -> 0 (gp `2^96` =
    79228162514264337593543950336), ([i64::MIN], [i64::MAX]) -> 0 (gp `2^64`). Fix: return 0 when
    some max < min, before any product; otherwise widths `i128::from(mx) - i128::from(mn) + 1`
    (at most 2^64), a u128 product with `checked_mul`, then `usize::try_from`; panic
    "count_integer_vectors_in_box: the count exceeds usize::MAX". ([i64::MIN + 1], [i64::MAX]) =
    2^64 - 1 = usize::MAX on 64-bit (HEAD gets it right only because it wraps twice).
14. `partitions.rs:103-126 Partition::dimension(&self) -> usize` (PROJECT_TRACKER.md:1312 rates it
    🟢): n! and the hook product in usize. 21! > 2^64, so most partitions of n >= 21 give a wrong
    value (release) or panic (debug). HEAD (`rev1/sweep.rs`): [11, 10] -> 16335 (gp 58786),
    [20, 1] -> 5 (gp 20), [5, 5, 5, 5, 5] -> 317 (gp 701149020), [8, 7, 6, 5, 4, 3, 2, 1] -> 2 (gp
    29258366996258488320, above 2^64); [21] -> 1 only because both products wrap to the same value.
    Every f^lambda with n <= 35 fits 64 bits (n = 35: at most 7821859115070000000, at
    (9,7,6,4,3,2,2,1,1)); n = 36 is the first n with a larger one (40971642983700000000 at
    (9,7,6,4,3,3,2,1,1); `rev1/maxf.gp`). So HEAD is wrong for most shapes with 21 <= n <= 35,
    where every value still fits.
    Fix: a new function for MAGMA `NumberOfStandardTableaux(P)` (ch. 145 §145.4.9, digest :299),
    re-exported at the crate root:

    ```rust
    /// MAGMA NumberOfStandardTableaux(P) (ch. 145 §145.4.9): f^lambda, the number of standard
    /// tableaux of shape lambda, exactly: Frobenius's formula with cancellation for thin shapes,
    /// the hook length formula for fat ones.
    pub fn number_of_standard_tableaux(p: &Partition) -> Integer
    ```

    Algorithm, two branches on r = the number of rows after the conjugation below. f^lambda is
    invariant under conjugation, so lambda is first replaced by its conjugate when it has more rows
    than columns (r <= c afterwards).
    - If r(r-1)/2 <= n: Frobenius's formula ([Ful97] §4.3, where the hook length formula of Frame,
      Robinson and Thrall, Canad. J. Math. 6 (1954) 316-324, is derived from it). With
      l_i = lambda_i + r - i (strictly decreasing), f^lambda = n! prod_{i<j} (l_i - l_j) /
      prod_i l_i!. Since l_1 <= n, n!/l_1! = prod_{t = l_1+1}^{n} t, so f^lambda =
      [prod_{t=l_1+1}^{n} t] · [prod_{i<j} (l_i - l_j)] / [prod_{i>=2} l_i!], with one exact
      division at the end. The factors number (n - lambda_1 - r + 1) + r(r-1)/2 +
      sum_{i>=2} l_i <= 4n, and only O(n - lambda_1 + r^2) of them are above 1, so thin shapes are
      cheap: for lambda = (n) or (1^n) nothing is multiplied and f = 1 for any n, where HEAD forms
      n!.
    - Otherwise (fat shapes such as the hook (m, 1^(m-1)), where r(r-1)/2 would reach n^2/8
      factors): the hook length formula n! / prod_cells h(c) directly, 2n factors. Products use
      `factorials::product_range_u64`, `product_u64s` and `product_tree` (§3.7).
    Measured (`bin_fix time ...`): (10^9) 10 µs; (1^(10^6)) 6 ms (the conjugation); (1000^1000),
    n = 10^6, 4.2 s for an 8.7 Mbit result (the hook branch alone: 3.2 s); the staircase with
    n = 10^6 5.7 s; the fat hook (2001, 1^2000) 1.7 ms, where Frobenius alone took 2.7 s;
    (200001, 1^200000) 13.9 s (the hook branch divides a 7 Mbit n! by a 6.6 Mbit hook product).
    The REPL does not call `dimension`, so no REPL limit depends on this cost. Checked
    (`bin_fix check`): Frobenius equals n!/prod(hooks) for all 7338 partitions of n <= 24, the
    two-branch function equals it for all partitions of n <= 22 and on (200, 1^199), and both match
    the gp values above.
    The parts come from `Partition::parts()` (positive, non-increasing: `Partition::new` normalises
    them); the conjugate comes from `Partition::conjugate` (`partitions.rs:55`) and has c < r parts.
    n, the l_i and the factors are u128, summed directly (not through `Partition::sum`, which is a
    usize sum); a factor of 2^64 or more becomes its own `Integer` leaf instead of being packed. Such
    shapes are absurd, but no arithmetic may wrap on them.

    `dimension()` keeps its signature: `number_of_standard_tableaux(self).to_usize()` with
    `.expect("Partition::dimension: f^lambda exceeds usize::MAX; use number_of_standard_tableaux")`,
    documented. Callers: `symmetric_group_representations.rs:35` (`IrreducibleRepresentation::new`
    now panics for an f^lambda above usize::MAX, n >= 36, instead of caching a wrong dimension; the
    module is 🔴, PROJECT_TRACKER.md:1255/1257), `specht_module.rs:673` (test, n <= 5) and
    `rustmath-symmetricfunctions/src/classical_bases.rs:806` (test, n <= 6): values unchanged.
15. `group_action.rs:172-188 burnside_lemma(group, num_colors)` and `:339-343
    count_fixed_colorings(perm, num_colors)`: `Integer::from(num_colors as u32)`. HEAD
    burnside(C_1, 2^32 + 1) -> 1 (truth 4294967297); count_fixed_colorings(id_2, 2^32 + 2) -> 4 (gp
    `(2^32+2)^2` = 18446744090889420804). Fix: `Integer::from(num_colors as u64)` (lossless on every
    Rust target). The exponent `num_cycles as u32` counts the cycles of a stored permutation (§1.7).
16. `tiling.rs:277-290 count_tilings(m, n, pieces)`: `transfer.pow(n as u32)`. HEAD
    count_tilings(2, 2^32 + 1, dominoes) -> 1 = count_tilings(2, 1). Fix:
    `transfer.pow(u32_arg(n, "count_tilings"))`, documented (the count has about 0.69 n bits and
    `Matrix::pow` takes a u32). count_tilings(2, 10) = 89 (gp `fibonacci(11)`) and (3, 4) = 11
    (OEIS A001835) stay; HEAD gives both.
17. `subset.rs:164-187 Subset::rank_k_subset`, `:190-230 Subset::unrank_k_subset`, and the private
    `binomial_usize` (`:627-640`): `result.saturating_mul(n - i) / (i + 1)` gives a wrong finite
    value as soon as an intermediate saturates, even when the binomial fits, and `rank +=` wraps.
    HEAD: the rank of {1, 2, 3} among the 3-subsets of 5·10^9 + 1 -> 9223372036854775807 (gp
    `binomial(5*10^9, 2)` = 12499999997500000000); unrank of 12499999997500000000 -> {1, 705032705,
    3557549058} after 3.6 s (truth {1, 2, 3}); the rank of {9999997, 9999998, 9999999} in 10^7 ->
    645920003284035455 (truth C(10^7, 3) - 1, about 1.7·10^20, which does not fit). Fix: delete
    `binomial_usize`. `rank_k_subset` adds `binomial_usize_checked` terms with `checked_add` and
    panics with "rank_k_subset: the rank exceeds usize::MAX" (every term is at most the rank, so a
    None term means the rank does not fit). `unrank_k_subset` treats None as "larger than any usize
    rank": `count.map_or(true, |c| remaining_rank < c)`.
18. `subset.rs:143-145 Subset::rank_binary(&self) -> usize`: `self.to_bitvector().unwrap_or(0)`,
    and `to_bitvector` returns None for every subset when n > usize::BITS. HEAD {0} in n = 100 -> 0
    (truth 1). Fix: sum 2^e over the elements; panic "rank_binary: the rank exceeds usize::MAX"
    when an element is >= usize::BITS. `to_bitvector` keeps its Option.
19. `debruijn.rs:49-62 DeBruijnSequence::new` and `:65-82 is_valid`: `k.pow(n as u32)` wraps or
    truncates, so invalid sequences are accepted. HEAD: new([], 2, 64) -> Some (release; debug
    panics), new([0], 2, 2^32) -> Some. Fix: `checked_pow_usize(k, n)`; None means k^n >
    usize::MAX >= len, so `new` returns None and `is_valid` returns false.
20. `debruijn.rs:295-341 universal_cycle_subsets(n, k)` and `:353-402
    universal_cycle_permutations(n, k)`, with the private `binomial_coefficient` and
    `falling_factorial` (`:427-457`): wrapping usize products give the target count of the greedy
    loop. HEAD: universal_cycle_permutations(70, 70) -> Some([0]) at once (70! = 0 mod 2^64);
    universal_cycle_subsets(70, 35) ran out of memory in the experiment. Fix: delete both helpers;
    use `binomial_usize_checked` and `falling_factorial_usize_checked` and panic with
    "universal_cycle_subsets: C(n, k) exceeds usize::MAX" (resp. "universal_cycle_permutations:
    n!/(n-k)! exceeds usize::MAX"), documented: such a cycle cannot be stored. The helpers' unit
    tests (`debruijn.rs:691-703`) move to `machine_word.rs`.
21. `designs.rs:300-325 OrthogonalArray::new` and its private `verify_orthogonality` (`:330-350`):
    `v.pow(t as u32)`. HEAD new(64, 1, 2, []) -> Some. Fix: `checked_pow_usize(v, t)`; None means
    the array cannot have that many rows, so `new` returns None and the check returns false.
22. `set_system.rs:687-706 SetSystem::sunflower_bound(k, r)` and `:730-740
    exceeds_sunflower_bound`: saturating products. HEAD sunflower_bound(30, 3) ->
    18446744073709551615 (gp `30!*2^30` = 284813089515958324736640819941867520000000). Fix: private
    `fn sunflower_bound_checked(k, r) -> Option<usize>` (`checked_mul`); `sunflower_bound` panics with
    "sunflower_bound: k!(r-1)^k exceeds usize::MAX" on None (documented); `exceeds_sunflower_bound`
    returns false on None (no family held in memory has more than usize::MAX sets). The r <= 1
    convention (usize::MAX) is not an overflow and stays. PROJECT_TRACKER.md:1275 flags this module
    for `vc_dimension`, not for this bound.
23. `enumeration.rs:1071-1088`, the private `binomial` of `InfiniteCartesianProduct` (via
    `count_partitions`, `:1058`): `saturating_mul` then `/ (i + 1)` gives wrong finite values. It
    is reached only through the diagonals of the enumeration, so a wrong value needs about 10^17
    elements enumerated first (bounded, as in §1.7). It is replaced anyway, to remove a third
    private binomial: `binomial_usize_checked(..).unwrap_or(usize::MAX)`, where usize::MAX is a
    documented "larger than any position" value; `get_indices_for_position` compares `k < ways`,
    which is right for any true count above usize::MAX.

Revision 2 (review 2, B1, and the widened patterns). HEAD outputs are release builds from
`rev2/sweep6.rs` (output `rev2/sweep6.txt`) unless another file is named; gp values from
`rev2/rev2.gp` and `rev2/rev2.gp.extra`. A debug build of HEAD panics with "attempt to ... with
overflow" on most of them.

24. `combinations.rs:62-81 Combination::rank` and `:84-110 Combination::unrank` (re-exported
    `lib.rs:112`), with the private `binomial_usize` (`:159-175`): `result * (n - i) / (i + 1)` in
    usize, the twin of item 17. HEAD: the rank of [1, 2, 3] among the 3-subsets of 5·10^9 + 1 is
    3276627960645224192 (gp `binomial(5*10^9,2)` = 12499999997500000000, which fits);
    `unrank(12499999997500000000, 5·10^9 + 1, 3)` gives [3, 566066340, 1487858385] (truth
    [1, 2, 3]). Fix: delete `binomial_usize`. In `rank`, each term is
    `binomial_usize_checked(n - 1 - j, k - 1 - i)` added with `checked_add`; on None, panic with
    "Combination::rank: the rank exceeds usize::MAX" (documented; every term is at most the rank,
    so a None term or a failed add means the rank does not fit). In `unrank`, a None count is
    "larger than any usize rank": `count.map_or(true, |c| remaining < c)`, as in item 17.
    Prototype (`rev2/fix2.rs`): rank and unrank equal the lexicographic position for every
    k-subset of every n <= 12, the two cases above are right, and the rank of
    [9999997, 9999998, 9999999] in 10^7 (gp `binomial(10^7,3)-1` > 2^64) panics after 1.8 ms.
25. `enumeration.rs:486-495 PermutationGrayCodeIterator::size_hint`, with the private usize
    `factorial` (`:499-505`): `(2..=n).product()` wraps, the twin of item 6. HEAD:
    `new(21).size_hint()` = (0, Some(14197454024290336768)), an upper bound below 21! =
    51090942171709440000 (gp), so it is wrong. Fix: `(0, falling_factorial_usize_checked(self.n,
    self.n))`, which is None when n! does not fit; delete `factorial`. `new(20)` keeps
    (0, Some(2432902008176640000)). The type is not an `ExactSizeIterator`.
26. `superpartitions.rs:444-451 count_superpartitions(n) -> usize`: `1usize << p.length()` and a
    usize `.sum()` over the p(n) partitions of n. The function's own definition (each partition
    with k parts gives 2^k superpartitions) makes the count the coefficient of x^n in
    prod_{i>=1} 1/(1 - 2x^i) (gp `polcoef(prod(i=1,n,1/(1-2*x^i))+O(x^(n+1)),n)`). HEAD: (62) =
    15969100102326684686 (right, and equal to gp), (63) = 13491456169319932838 after 0.21 s (gp
    31938200243029484454 > 2^64). Fix: evaluate that coefficient by the partition DP
    `a[0] = 1; for i in 1..=n { for j in i..=n { a[j] += 2 a[j - i] } }` with `checked_mul` and
    `checked_add`, and panic with "count_superpartitions: the count exceeds usize::MAX"
    (documented). After the pass for i, a[j] is the weighted count of the partitions of j into
    parts at most i (one factor 1/(1 - 2x^i) per pass). The check is exact: the final count c is
    increasing (adding a part 1 maps the partitions of m - 1 injectively into those of m and doubles
    the weight, so c(m) >= 2 c(m-1)); each intermediate a[j] is at most c(j) <= c(n); and each
    product 2 a[j - i] is a summand of a later a[j]. So a failed operation means c(n) > usize::MAX.
    The doc's "using dynamic programming" becomes true: O(n^2) word steps, 3.6 µs at n = 62,
    where HEAD enumerated 1300156 partitions in 0.27 s. Checked (`rev2/superp.rs`): equal to
    `superpartitions(n).len()` for n <= 16 and to HEAD for n <= 62; (63) overflows.
27. `binary_words.rs:39-45 BinaryWord::to_usize` and `:32-35 BinaryWord::from_usize(value,
    length)`: `1 << i` and `value >> i` for i >= 64, the twin of item 18. HEAD (`rev2/sweep6.rs`,
    `rev2/shr.rs`): a 65-bit word with bits 0 and 64 set gives 2 (the shift by 64 wraps to a shift
    by 0); `from_usize(1, 65)` sets bits 0 and 64. Truth: 1 + 2^64 does not fit;
    `from_usize(1, 65)` is bit 0 followed by 64 zeros. Fix: `to_usize` ORs `1 << i` over the set
    bits and panics with "BinaryWord::to_usize: bit {i} is set; the value exceeds usize::MAX" when a
    set bit has i >= usize::BITS (documented; any number of leading zeros stays allowed: a 100-bit
    word with only bit 63 set gives 2^63, as at HEAD). `from_usize` sets bit i only for
    i < usize::BITS.
28. `subset.rs:302-308 Subset::next_binary` and `:71-79 Subset::from_bitvector(bits, n)` (also
    reached through `unrank_binary`, `:156`): `(1 << self.n) - 1` at n = 64 = usize::BITS
    (`to_bitvector` accepts n <= 64), and `bits >> i` for i >= 64. HEAD: the empty subset of a
    64-set gives None ("this is the full set"; truth {0}); the full subset of a 64-set gives
    Some(empty set) (truth None); `from_bitvector(1, 65)` = {0, 64} (truth {0}). n = 63 is right at
    HEAD. Fix: the full mask is `if n == 0 { 0 } else { usize::MAX >> (usize::BITS as usize - n) }`
    (n <= usize::BITS there); `from_bitvector` reads bits `0..n.min(usize::BITS as usize)`.
29. `permutations.rs:122-135 Permutation::order`, with the private `lcm` (`:576-581`):
    `(a * b) / gcd(a, b)` in usize, so the product wraps even when the lcm fits. HEAD: cycles of
    the 15 primes up to 47 and one cycle of length 94 (422 points) give 26163907895633379 (gp
    `lcm(concat(primes(15),[94]))` = 614889782588491410, which fits); cycles of the 16 primes up to
    53 (381 points) give 14142414403480493114 (gp `lcm(primes(16))` = 32589158477190044730,
    above 2^64). Fix: `lcm(a, b) = (a / gcd(a, b)).checked_mul(b)` (exact: no intermediate exceeds the
    lcm), and `order` panics with "Permutation::order: the order exceeds usize::MAX; use
    order_integer" (documented). New `pub fn order_integer(&self) -> Integer`: the same fold with
    `Integer::lcm` (`rustmath-integers/src/integer.rs:116`), MAGMA `Order(g)` for a permutation
    (ch. 58, digest `Chapter_58_Permutation_Groups.md:204`), which returns an integer.
30. The 2^n mask loops. Each loop `for mask in 0..(1 << n)` lists all subsets of an n-element set.
    Where the literal `1` is untyped and the mask is only used in `mask & (1 << i)` or
    `(mask >> i) & 1`, Rust gives it the type `i32`, so `1 << 31` is negative and the range is
    empty, and `1 << 32` wraps to 1 in a release build. HEAD (`rev2/sweep6.rs`, `rev2/misc6.rs`):

    | site | public function | mask type | HEAD | truth |
    |---|---|---|---|---|
    | `binary_words.rs:344` | `all_binary_words(n)` | usize | n = 64: 1 word | 2^64 words (cannot be stored) |
    | `posets.rs:351` | `Poset::all_order_ideals` | i32 | chain of 31: 0 ideals; chain of 32: 1 | 32; 33 (a chain of n has n + 1 ideals) |
    | `posets.rs:617` | `Poset::all_antichains` | i32 | chain of 31: 0; chain of 64: 1 | 32; 65 (the empty set and the singletons) |
    | `subword.rs:155-161` | `all_subwords` | usize capacity, i32 mask | [0; 64]: 1 subword | 2^64 |
    | `subword.rs:253` | `subword::SubwordComplex::new` (`compute_faces`) | i32 | q = 0101... of 31 letters, w = [0]: 0 faces; 32 letters: 1 face | 2^31 - 2^15 = 2147450880 and 2^32 - 2^16 (a removed set is a face unless it contains every position of a 0) |
    | `subword_complex.rs:343` | `subword_complex::SubwordComplex::new` (private `all_subsets` of each facet) | i32 | a facet of 31 or more positions: no faces from it | 2^(facet size) faces |
    | `set_system.rs:263` | `SetSystem::shadow` | i32 | one 31-set: 0 sets; 32-set: 1; 64-set: 1 | 2^31; 2^32; 2^64 |
    | `set_system.rs:296` | `SetSystem::shade` | i32 | the same over the complement of each set | |
    | `set_system.rs:805` | private `is_shattered` (`vc_dimension`) | i32 | reached only with subsets of at most 20 elements (`vc_dimension` stops at 20, PROJECT_TRACKER.md:1275) | fixed with the others for uniformity |
    | `grossman_larson.rs:195` | `RootedTree::admissible_cuts` | i32 | a path of 32 vertices (31 edges): 0 cuts; 33 vertices: 1 | 32; 33 (the empty cut and each single edge; HEAD gives 5 for 5 vertices) |
    | `multiset_partition_into_sets_ordered.rs:386` | `OrderedMultisetPartitionIntoSets::finer` | i32 | one block of 32 elements: 0 | 2^32 - 2 splits |
    | `composition.rs:259`, `:294` | `signed_compositions`, `signed_compositions_k` | i32 | `signed_compositions_k(31, 31)`, `(32, 32)`, `(64, 64)`: 0 each | 1 each ([1; k]) |
    | `integer_vectors.rs:455` | private `generate_all_sign_patterns` (`integer_vectors_with_l1_norm_bounded`) | i32 | (31, 0): 0 vectors; (32, 0): 1 | by its own rule 2^31 and 2^32 copies of the zero vector (§1.8) |

    Fix: every such bound becomes `0..machine_word::subset_mask_count(n, "<function name>")`, the
    mask is a usize (`let count: usize = subset_mask_count(..); for mask in 0..count`), and the
    inner tests are `mask & (1usize << i) != 0` or `(mask >> i) & 1 == 1` with i < n < usize::BITS.
    `all_subwords` calls the helper before `Vec::with_capacity`. `finer` loops over
    `1..count - 1` (its blocks have at least 2 elements there). Each public function gets a
    `# Panics` line ("if the set has usize::BITS or more elements: its 2^n subsets cannot be
    listed") and a cost line ("2^n iterations"). After the fix an n from 31 to 63 runs its 2^n
    iterations (and then runs out of time or memory, which is honest) instead of returning an empty
    or one-element result. The exception is `composition.rs`: `signed_compositions` and
    `signed_compositions_k` drop the loop. For a composition c of |n| and signs s, the signed sum
    sum s_i c_i equals n only when every s_i is the sign of n, because sum c_i = |n|. So exactly one
    of the 2^k patterns passes the existing `sum() == n` filter, and the new body pushes that one
    signed composition for each composition (all parts positive for n > 0, all negative for n < 0;
    n = 0 and k = 0 as now). On every input where HEAD's loop was right the output is identical and
    in the same order (one push per composition), at O(k) per composition. What these functions
    should return is a different question (§1.8).
31. `q_analogue.rs:243-246 q_multinomial(n, ks)` and `t_sequences.rs:531-534 t_multinomial(n,
    ks)`: `let sum: u32 = ks.iter().sum()`, the twin of `multinomial` (§1.4). HEAD
    (`rev2/qmult.rs`): for (1, [u32::MAX, 2]) the sum wraps to 1 = n, so the function goes on to
    build `q_factorial(u32::MAX)`; it did not return within 30 s under a 4 GB limit. A debug build
    panics with "attempt to add with overflow". Truth: the zero polynomial (2^32 + 1 is not 1).
    Fix: a u128 sum, as in §1.4.
32. `integer_vectors.rs:215`, `:222` (`integer_vectors_with_sum`), `:334`
    (`integer_vectors_with_weighted_sum`), `:460` (`integer_vectors_with_l1_norm_bounded`), `:549`
    (`IntegerVectorSumIter::next`): `Integer::from(x as i64)` of a usize value. HEAD
    (`rev2/sweep6.rs`, `rev2/misc6.rs`): `integer_vectors_with_sum(1, 1 << 63)` =
    [[-9223372036854775808]] (truth [[9223372036854775808]], gp `2^63`);
    `integer_vector_sum_iter(2, 1 << 63)` yields [0, -9223372036854775808] first. Fix:
    `Integer::from(x as u64)` (lossless: usize has at most 64 bits on every Rust target), and at
    `:460` the negation is done on the `Integer`. `:222` and `:334` are reached only after 2^63
    loop steps, but take the same one-line fix.
33. `integer_lists.rs:382-468 propagate_constraints` (via `integer_lists`):
    `max_future * (remaining_positions - 1)` and `min_future * (..)` wrap, and
    `prev + slope as usize` can wrap; `satisfies_constraints` (`:524`) compares slopes as
    `parts[i] as isize - parts[i - 1] as isize`; `generate_with_length` bounds the sum by
    `length * max_part` (`:333`) and `integer_lists` bounds the length by `sum + 1` (`:292`). HEAD:
    lists of length 3, sum 3·2^62, parts in [2^62, 2^63] gives [] (truth [[2^62, 2^62, 2^62]], the
    only list: three parts of at least 2^62 with sum 3·2^62). Fix: `saturating_mul` for the two
    products, `saturating_add` for the two slope bounds, `sum.saturating_add(1)` and
    `length.saturating_mul(max_part)`. Saturation is exact for these bounds. `max_remaining` is
    only compared with `remaining_sum <= usize::MAX`, so a true value above usize::MAX and
    usize::MAX take the same branch; the same holds for `min_remaining` under `saturating_sub`. A
    saturated lower slope bound of usize::MAX can still admit the value usize::MAX, but the final
    `satisfies_constraints` check (now with slopes in i128) rejects it. A saturated length or sum
    bound only lengthens an enumeration that has more than usize::MAX lists anyway. The two
    defaults of 100 (`:292`, `:333`) are a different defect (§1.8).
34. `vector_partition.rs`: `count_partitions_with_parts` (`:113-178`, via `fast_vector_partitions`
    and `fast_vector_partitions_with_max_part`), `count_with_max_parts_helper` (`:364-421`, via
    `count_vector_partitions_max_parts`) and `count_with_exact_parts_helper` (`:445-506`, via
    `count_vector_partitions_exact_parts`) sum in usize with `total +=`; `count_partitions_with_parts`
    and `generate_partitions_with_parts` (`:336`, via `vector_partitions` and
    `vector_partitions_with_max_part`) also stop after 1000 copies of a part (`if k > 1000 { break }`,
    "safety check"). HEAD (`rev2/vpart.rs`, `rev2/vpart2.rs`): `fast_vector_partitions(&[417])` =
    541220193622112941 after 0.10 s (gp `numbpart(417)` = 18987964267331664557, whose residue mod
    2^64 is HEAD's value); `fast_vector_partitions_with_max_part(&[1001], Some(&[1]))` = 0 and
    `vector_partitions_with_max_part(&[1001], Some(&[1]))` is empty (truth 1, the partition
    1^1001; gp `polcoef(1/(1-x)+O(x^1003),1001)`); `fast_vector_partitions_with_max_part(&[1001, 1],
    Some(&[1, 1]))` = 1 (truth 2: the coefficient of x^1001 y in 1/((1-x)(1-y)(1-xy)),
    `rev2/rev2.gp.extra`). Fix: delete both caps. The loops end without them because no part is
    zero (`generate_all_parts` and `generate_parts_up_to` remove the zero vector, `:188`, `:211`),
    so the subtraction fails after at most max_i v_i + 1 copies. Keep one remainder and subtract
    one more copy per iteration, instead of rebuilding it from the vector for every k (same values;
    O(k) instead of O(k^2) per state). Sum with `checked_add` and panic with
    "<function>: the count exceeds usize::MAX" (documented in the four public counting functions).
    The check is exact: every memo state (r, i) is reached from v by subtracting multiples of the
    parts P_0 .. P_(i-1), so its count is a summand of the top count along that path, and a partial
    `total` is at most its state's count; so a failed add means the top count exceeds usize::MAX.
    Prototype (`rev2/fix2.rs`): equal to a copy of HEAD on 154 vectors where HEAD has no cap or
    wrap (one component up to 40, two up to 6, three up to 3); [416] = 17873792969689876004 (gp
    `numbpart(416)`); [417] panics; the two cap cases give 1 and 2. Cost: [416] takes 0.11 s in
    release and 1.24 s unoptimised at HEAD (`rev2/bin_vpart_dbg`); the fix does not add work.
35. `designs.rs:178-196 BlockDesign::replication_number`: `self.lambda * (self.v - 1)` in usize.
    HEAD (`rev2/repl_num.rs`): v = 2^33, k = 3, t = 2, lambda = 2^32 and no blocks
    (`BlockDesign::new` checks the design only for v <= 10, §1.8) gives Some(9223372034707292160);
    gp `2^32*(2^33-1)/2` = 18446744071562067968, which fits. Fix: the numerator in u128
    (`u128::from(lambda) * (u128::from(v) - 1)`, with None for v = 0, where HEAD underflows), then
    `usize::try_from(numerator / denominator)` with the panic "replication_number: r exceeds
    usize::MAX" (documented). None keeps its meaning "not a 2-design, or r is not an integer". The
    Fano plane still gives Some(3).
36. `dyck_word.rs`: `DyckWord::area` (`:164-178`), `DyckWord::bounce_statistic` (`:183-185`),
    `NuDyckWord::total_area` (`:445-447`), `NuDyckWord::bounce_statistics` (`:493-515`),
    `NuDyckWord::dinv` (`:576-590`) and `NuDyckWord::detailed_bounce_statistics` (`:595-625`) add
    heights in i32. These are counts (cells under a path, pairs), so they belong to this class even
    though the type is signed. Each height is at most the length of the stored word, so it fits
    (R1), but the sums grow with the square of the length, so they wrap from semilength 65537, a word
    of 128 KB. HEAD (`rev2/dyck.rs`): `area()` of N^65537 E^65537 is -2147450880 (gp
    `65537*65536/2` = 2147516416 > 2^31 - 1); semilength 65536 gives 2147450880, right. Fix: keep
    the i32 return types (no caller outside this file, §0) and do every sum and difference with
    `checked_add` / `checked_sub` through one private helper
    `fn checked_sum_i32(values: impl Iterator<Item = i32>, what: &str) -> i32`, which panics with
    "{what}: the value leaves the i32 range" (documented `# Panics` on the six functions). Values
    that fit are unchanged.

### 1.7 Not changed, with reasons

Every remaining non-test hit of the patterns P1-P10 (§1.6) falls under one of these rules.

- R1 (memory). The value is the length or size of an object held in memory, an index into one,
  or a count bounded by such a size. A wrong value needs an object of at least 2^31 elements (for
  an `i32`), 2^32 (for a `u32`) or 2^63 (for a usize).
- R2 (enumeration). The value counts or indexes objects that the function lists first. A wrong
  value needs at least 2^31 objects listed first.
- R3 (work). The value is reached only after at least 2^32 steps of big-integer work (revision 1).
- R4 (flagged). The module or function is 🔴 or 🛑 in PROJECT_TRACKER. Its arithmetic goes with its
  rewrite.
- R5 (size invariant). The value is the size of a partition-like object: the usize sum of its
  stored parts. Every such type assumes that its size fits a usize, and no constructor checks it:
  `Partition::new(vec![1 << 63, 1 << 63]).sum()` is 0 at HEAD, and `Composition::new` accepts the
  same parts with sum 0 (`rev2/size_inv.rs`). A site whose overflow needs a size above usize::MAX
  falls under this rule. The invariant is one new HANDOFF item (§1.8): checked constructors for
  about ten types.
- R6 (API values). The function returns or stores a machine-word value whose range is part of its
  signature, and the true value can leave that range. Most are signed (`i32`, `i64`, `isize`); the
  one unsigned case is the usize terms of user sequences in `regular_sequences.rs`. These
  are values (Möbius values, characters, energies, degrees, values of affine permutations, signed
  parts, terms of user sequences), not counts or ranks; the counts among the signed results, the
  Dyck-word statistics, are item 36. Widening a value type changes each API and its consumers, so
  these go to HANDOFF as one item with reproductions (§1.8).
- R7 (guarded). A bound checked before the arithmetic keeps it in range.
- R8 (cannot overflow). A widening cast, a shift by 1, a u128 product of u64 values, or
  saturation used as a conservative bound.

Revision-1 rows (P1-P3), with the rule each one falls under in brackets where it is not stated:

| site | why it is left as it is |
|---|---|
| `alternating_sign_matrix.rs:462`, `cyclic_sieving.rs:304`, `tiling.rs:252` | counts of objects already enumerated in memory (2^32 objects first) |
| `group_action.rs:182`, `:342` (`num_cycles as u32`), `:187`, `:261` (`group.order()`, the number of stored elements, `:113`), `:281` (`cycle_len`) | bounded by a stored permutation or group. `:281-282` also belongs to `polya_enumeration`, which is wrong for another reason (§1.8) |
| `schubert.rs:181` | runs after `Permutation::from_vec((0..n)...)` has allocated n entries |
| `ribbon_tableau.rs:250`, `:253` | the spin of a stored ribbon, at most its number of cells |
| `designs.rs:601` `HadamardMatrix::sylvester(k)` | builds a 2^k x 2^k matrix; from k = 16 on memory runs out long before 2^k can wrap at k = 64 (a debug build panics at k = 64) |
| `debruijn.rs:103` | a digit of a validated element x < k; a valid sequence over k >= 2^32 symbols has at least 2^32 entries |
| `expnums.rs:136-654`, `species.rs:119-565` | loop indices n < degree while a Vec of `degree` Integers is filled (2^32 entries are 128 GB). Several of these functions are 🛑 (PROJECT_TRACKER.md:1260, 1261, 1269, 1277) |
| `fully_packed_loop.rs:516-517`; `asm_count` (`alternating_sign_matrix.rs:220-229`, u32 `3*k+1` and `n+k`) | inside an O(n) loop of factorial products: the first step for n >= 2^31 already multiplies n! (more than 2^36 bits), so no truncated value is ever returned. FPL is 🛑 (:1263) |
| `plane_partition.rs:400-515` | `j as u32` truncates only after O(n^2) big multiplications with n >= 1.4·10^9. `:371` belongs to `count_plane_partitions_in_box`, which is wrong for small inputs (§1.8) |
| `q_analogue.rs:169`, `t_sequences.rs:159-505` | DP over an (n+1) x (k+1) table of polynomials; memory runs out first |
| `symmetric_group_representations.rs:364-382` (`conjugacy_class_size`, 🟢 :1306) | the casts follow an n-step factorial loop (n >= 2^32 never finishes; parts are at most n) |
| `symmetric_group_representations.rs:526-542` (`verify_orthogonality`, usize n!) | 🔴 (PROJECT_TRACKER.md:1257); stays with the Murnaghan-Nakayama rewrite |
| `similarity_class_type.rs:154-466` | module 🛑 (:1276). `fq` and `order_of_general_linear_group` truncate only after 2^32 big multiplications. `centralizer_group_cardinality` (`:136-162`) truncates a part size >= 2^32 at once, but its formula is wrong even for small types (§1.8), so its casts go with the module's rewrite |
| `hall_polynomial.rs:213-405` | module 🛑 (:1264); even the one-row case is wrong (§1.8) |
| `ranking.rs:158-730` | module 🛑 (:1273) |
| `ncsym.rs:258` (`matchings`, a usize product) | a placeholder ("return a simple bound"), 🛑 (:1270) |
| `hadamard.rs:270-272` | a u128 work estimate compared with a budget; saturation is conservative there (a saturated cost is refused), so it is correct |
| `parking_functions.rs:333` (`ParkingFunction::count`, a twin of item 9) | the file is not compiled: lib.rs has no `mod parking_functions` (PROJECT_TRACKER.md:1293 calls it a parallel module). §1.8 asks HANDOFF to delete it or wire it in |
| `lib.rs:522-613`, `:749-751` | removed by the rewrites of §3.3 and §3.6 |
| tests: `derangements.rs:323`, `path_tableaux.rs:621`, `plactic_monoid.rs:372`, `q_analogue.rs:410`, `set_partition.rs:564-565`, `six_vertex_model.rs:694-696`, `t_sequences.rs:656-782`, `tamari_lattice.rs:690`, `debruijn.rs:681` | test code with small sizes |

In the language of the rules: the first two rows and `schubert.rs`, `ribbon_tableau.rs`,
`designs.rs:601`, `debruijn.rs:103`, `expnums.rs`/`species.rs`, `q_analogue.rs`/`t_sequences.rs`
are R1 or R2; `fully_packed_loop.rs`, `asm_count`, `plane_partition.rs`,
`conjugacy_class_size` and `similarity_class_type.rs` are R3 (and R4 where flagged);
`verify_orthogonality`, `hall_polynomial.rs`, `ranking.rs`, `ncsym.rs` are R4; `hadamard.rs:270-272`
is R8.

Revision-2 rows (P4-P10). Sites that are items of §1.6 are not repeated.

| pattern | sites | rule and reason |
|---|---|---|
| P4 | `enumeration.rs:76`, `:96` | R8: shifts by 1 (Gray code conversion) |
| P4 | `enumeration.rs:121`, `:208`, `:214`, `:282` | R7: `gray_code_sequence` and `GrayCodeIterator` check n <= 30, `BinaryGrayCodeIterator` n <= 63, first |
| P4 | `subset.rs:122`, `:457`, `:586` | R7: `to_bitvector` checks n <= usize::BITS (elements are < n), `all_subsets` n <= 20, `SubsetIterator` n <= 30 |
| P4 | `hadamard.rs:272`, `:282`, `:285` | R8 and R7: a u128 cost with `n.min(63)`; the u32 mask runs only when n!·2^n <= 2·10^7, so n <= 9 |
| P4 | `tiling.rs:176-227` (u32 `ColumnState` bits), `:246` (`1 << height`) | R1: the transfer matrix has 4^height entries, so memory runs out from height 16. At heights 32 and 64 `num_states * num_states` wraps and the first write panics with an index error at once (`rev2/tile.rs`: "the len is 0 but the index is 201855969"); no value is returned |
| P4 | `ranking.rs:593`, `:609`, `:626` | R4 (🛑, PROJECT_TRACKER.md:1273) |
| P4 | `parking_functions.rs:162`, `:244` | the file is not compiled (§1.8) |
| P5 | `lib.rs:668`, `:686` | R8: `Integer::pow` with a u32 exponent |
| P5 | every other hit | a revision-1 item (9, 11, 12, 16, 19, 21) or row (`designs.rs:601`, `expnums.rs`, `group_action.rs`, `ribbon_tableau.rs`, `similarity_class_type.rs`, `symmetric_group_representations.rs:377`), `ranking.rs:730` (R4), `parking_functions.rs:333` (not compiled) |
| P6 | `ranking.rs:324` | R4. The other two hits are items 13 and 25 |
| P7 | sums of stored parts: `partitions.rs:22` (`Partition::sum`) and its uses (`:42`, `:108`, `:133`, `:162`, `:460`, `:535`, `:654-690`, `:772`, `:1196`), `composition.rs:24`, `:165`, `:355`, `:436`, `:453`, `superpartitions.rs:91`, `:121`, `:151`, `plane_partition.rs:90`, `:97`, `integer_lists.rs:51`, `:352`, `:389`, `:476`, `:558`, `key_polynomial.rs:50`, `:122`, `multiset_partition_into_sets_ordered.rs:130`, `:199`, `:249`, `skew_partition.rs:65`, `:770-841`, `descent_algebra.rs:189`, `affine_permutations.rs:148`, `ribbon_tableau.rs:173`, `:326`, `super_tableaux.rs:345`, `:481`, `:524`, `tableau_tuple.rs:282`, `tableaux.rs:420`, `vector_partition.rs:63`, `young_representations.rs:53`, `:249-250`, `symmetric_group_representations.rs:54-354` | R5. (`knutson_tao.rs`, `hall_polynomial.rs`, `ncsym.rs`, `similarity_class_type.rs`, `cluster_complex.rs:70` are also R4) |
| P7 | sums of lengths or counts of stored or listed objects: `composition_tableau.rs:75`, `:243`, `free_prelie_algebra.rs:92`, `group_action.rs:140`, `:178`, `:341`, `multiset_partition_into_sets_ordered.rs:108`, `parking_function.rs:93`, `:105`, `:171`, `ribbon_tableau.rs:172`, `:266`, `subword_complex.rs:476`, `super_tableaux.rs:138`, `tableau_tuple.rs:46`, `tableaux.rs:52`, `young_representations.rs:196`, `bijectionist.rs:122`, `:147`, `integer_vectors.rs:523`, `:559` | R1 or R2 (a valid parking function has preferences at most its length) |
| P7 | sums of ±1 entries of a stored matrix: `alternating_sign_matrix.rs:53`, `:61`, `:368`, `fully_packed_loop.rs:65`, `:71`, `:480` | R1 |
| P7 | `integer_matrices.rs:163-164`, `:228-229` (u32 totals) | the totals only feed an early exit. When one wraps, the generator still enforces every row and column sum (in i64), so the result is right: `integer_matrices(&[u32::MAX, 1], &[0])`, `(&[u32::MAX, 1], &[0, 0])` and `(&[u32::MAX, 2], &[1])` all give no matrix, at once (`rev2/size_inv.rs`) |
| P7 | `partitions.rs:809-811` | R6 (§1.8). (`dyck_word.rs:184`, `:446`, `:504`, `:604` are item 36) |
| P8 | `affine_permutations.rs` index and rank casts (`:105-169`, `:230-260`, `:366-511`) | R1 (the window holds rank + 1 entries); the values of `apply` are R6 |
| P8 | `composition.rs:215`, `:217` (`part as isize`) | R6 |
| P8 | `composition.rs:227`, `:247`, `:286` (`abs() as usize`) | `:247` and `:286` become `n.unsigned_abs()` in item 30's rewrite (for `isize::MIN`, `abs()` panics in debug); `:227` (`abs()` of a stored part) can only fail on a part equal to `isize::MIN`, which arises only through the R6 cast of `:215` |
| P8 | `alternating_sign_matrix.rs`, `constellation.rs:207-220`, `designs.rs:572`, `fully_commutative_elements.rs:110`, `gelfand_tsetlin.rs:271`, `parallelogram_polyomino.rs`, `partitions.rs:829-1139` (cell residues), `path_tableaux.rs:335-398`, `set_partition.rs:335`, `:380`, `skew_partition.rs:171-172`, `:914-915`, `tableaux.rs:203-204`, `tiling.rs:200-203`, `triangles_fhm.rs`, `young_representations.rs:120-121`, `sidon_sets.rs:321`, `:365` | R1: positions, contents and counts of stored objects (the greedy B_h run stores up to n values) |
| P8 | `subword.rs:376`, `:406`, `subword_complex.rs:475`, `cyclic_sieving.rs:207` | R2: faces listed first (after item 30, a face of 60 or more positions needs 2^60 masks first); the cyclic sieving value is compared with a count of listed objects |
| P8 | `hadamard.rs`, `integer_matrices.rs:179-331`, `young_representations.rs:56`, `:61`, `partition_shifting_algebras.rs:222`, `:474`, `:495`, `ch144_enumerative.rs:109`, `enumeration.rs:143` | R8: widening casts (u32 to i64, u64 to u128, usize to u64), casts after a non-negativity check, `trailing_zeros`, a u32 capacity; `integer_matrices.rs:318`, `:331` saturate a conservative bound |
| P8 | `knutson_tao.rs:216`, `:299` (`as u8` labels), `symmetric_group_representations.rs:108-588`, `fully_packed_loop.rs`, `ranking.rs`, `similarity_class_type.rs`, `hall_polynomial.rs`, `ncsym.rs`, `expnums.rs`, `species.rs` | R4 (`knutson_tao.rs` is 🛑, PROJECT_TRACKER.md:1268) or revision-1 rows |
| P9 | `flag.py`: `alternating_sign_matrix::descending_diagonal_sums`, `dyck_word::peaks`/`valleys`/`returns`/`bounce_path` (indices), `debruijn_sequence`, `debruijn_prefer_one` (`vec![0; k * n]`: a wrapped size gives an index panic, never a value; the sequence has k^n entries), `perfect_matching::to_chord_diagram`, `permutations::descents`/`ascents`, `pipe_dream::num_elbows`, `regular_sequences::detect_periodicity`, `skew_partition::spin`, `subword::dimension`, `subword_complex::dimension`, `superpartitions::frobenius_coordinates`, `word::christoffel_word`/`sturmian_word` (a word of length p + q is built), `young_representations::adjacent_word` | R1 |
| P9 | `flag.py`: `enumeration` `CartesianProduct::size_hint` | R8: the total is a checked product, and `produced` <= total, so its steps fit |
| P9 | `flag.py`: `composition::weighted_sum` | R5 |
| P9 | `flag.py`: `affine_permutations::apply_type_*`, `partitions::degree_with_multicharge`, `six_vertex_model::energy`, `regular_sequences::sequence_product`, `young_representations::mn_character` (also R3: the recursion has at least abs(chi) leaves), `subword::h_vector` / `binomial_coefficient` (also R2) | R6 |
| P9 | `flag.py`: `group_action::polya_enumeration` / `multiply_polynomials`, `interval_posets::interval_representation` (f64, not an integer), `hadamard::legendre` (u128) | §1.8, not integer arithmetic, R8 |
| P9 | `flag.py`: `ncsym::matchings`, `ranking`, `similarity_class_type`, `symmetric_group_representations::murnaghan_nakayama` / `decompose_character` | R4 |
| P9 | `flag2.py`: `razumov_stroganov_refined_count`, `backtrack_count` (two), `burnside_count`, `count_partitions` (`InfiniteCartesianProduct`, item 23), `count_fc_elements_by_length`, `tamari_lattice::rank` | R2 |
| P9 | `flag2.py`: `composition_tableau::size`/`major_index`, `constellation::count_cycles`, `group_action::cycle_index`, `incidence_structure::intersection_number`, `parking_function::major_index`, `ribbon_tableau::total_spin`, `six_vertex_model::vertex_type_counts`, `super_tableaux::size`, `tableau_tuple::size`, `tableaux::size`, `tamari_lattice::size`, `words::count_factor` | R1 |
| P9 | `flag2.py`: `multiset_partition_into_sets_ordered::size`/`shape_from_size`, `partitions::beta_numbers`, `skew_partition::size` | R5 |
| P9 | `flag2.py`: `dyck_word::total_area` | item 36 |
| P9 | `flag2.py`: `ranking::rank` (two), `similarity_class_type::size`/`centralizer_algebra_dim` | R4 |
| P9 | `flag3.py` (142 functions that return a machine word and add): positions, lengths, indices, degrees and counts of stored objects (`affine_permutations::length_type_*` from the window, `binary_words::num_runs`, `constellation::genus`, `permutations::inversions`, `parking_function::diagonal_inversions`, `word::period`, the residue, corner, rim and KMP helpers, `hadamard_invariant`, ...) | R1 |
| P9 | `flag3.py`: the `sum`, `volume`, `trace`, `degree`, `weight` and `get_min_sum` / `get_max_sum` methods of partition-like types (`partitions.rs:21`, `:771`, `composition.rs:23`, `:164`, `integer_lists.rs:50`, `:203-220`, `key_polynomial.rs:49`, `:121`, `plane_partition.rs:89-98`, `superpartitions.rs:90`, `:145`, `vector_partition.rs:44`, `symmetric_group_representations.rs:53`) | R5 |
| P9 | `flag3.py`: `posets::mobius_helper`, `regular_sequences::nth` / `sequence_sum`, `symmetric_group_representations::character` | R6 (and R4 for the last) |
| P9 | `flag3.py`: `subword_complex::euler_characteristic` | R2 (an alternating sum of listed faces) |
| P9 | `flag3.py`: `cyclic_sieving::order` | R7: it panics when the orbit exceeds 10^6 steps |
| P9 | `flag3.py`: `wreath_product::ColoredPermutation::order` | not an integer-width defect: it stops after 10^6 steps and returns 1000001 as the order (§1.8) |
| P9 | `flag3.py`: `cluster_complex`, `ncsym`, `knutson_tao`, `ranking` functions | R4 |
| P9 | `flag3.py`: `dyck_word` sums, `stars_and_bars`, `universal_cycle_permutations` | items 36, 5, 20 |
| P10 | `fully_packed_loop::factorial_big` (`Integer`), `q_analogue::divide_q_factorial_polynomials`, `t_sequences::divide_t_factorial_polynomials` (polynomials) | R8: no machine-word count |
| P10 | `fn rank` / `fn position`: `affine_permutations::rank`, `enumeration::position`, `fully_commutative_elements::rank`, `pipe_dream::rank`, `subword::rank` (a count of grid cells), `tamari::position_at` | R1; `cluster_complex::rank` R4 |
| P4-P10 | `parking_functions.rs` | not compiled (§1.8) |

What remains known after this design is therefore: the R5 size invariant, the R6 signed values,
and the R4 modules, each recorded in HANDOFF (§8.1 step 12) with its reproduction. Arithmetic that
none of the patterns P1-P10 matches is not claimed either way. The HANDOFF line says exactly this,
and it does not call the sweep complete.

### 1.8 Found by the sweep, reported and not fixed here

These give wrong values for small inputs, and C1 does not change them. Most are not integer-width
defects, so they are not part of S2, S3, S4 or S10. The two integer-width entries of revision 2
(R5, R6) need a checked constructor for about ten types or a wider return type in about ten APIs
and their consumers (§1.7), which is a redesign rather than a sibling fix. §8.1 step 12 adds every
entry to HANDOFF as a new open item, each with this reproduction.

- `plane_partition.rs:358-376 count_plane_partitions_in_box(a, b, c)`: MacMahon's product is
  evaluated with a truncating integer division of non-integral partial products. HEAD (2,2,2) ->
  15, (3,3,3) -> 874, (2,3,4) -> 393; gp
  `prod(i=1,a,prod(j=1,b,prod(k=1,c,(i+j+k-1)/(i+j+k-2))))` gives 20, 980, 490.
  PROJECT_TRACKER.md:1272 flags only the q-series and the generator of this file.
- `group_action.rs:255-305 polya_enumeration`: a cycle of length L contributes w^L instead of L·w
  to the exponent (the keys add in `multiply_polynomials`, `:309-325`). HEAD, C_2 on two points
  with two colours of weight 1: {1: 1, 2: 2}; Polya's theorem, (p_1^2 + p_2)/2 with p_L = 2t^L,
  gives {2: 3}. Its `unwrap_or(0)` key (`:282`) goes with that fix.
- `hall_polynomial.rs:117-133`, the one-part case, returns q^(k·m) for lambda = (k + m),
  mu = (k), nu = (m). HEAD g^(2)_((1),(1)) = x, but a cyclic group of order p^2 has exactly one
  subgroup of order p, so the value is 1 (the function's own definition, `:75-79`, counts such
  subgroups). PROJECT_TRACKER.md:1264 calls the one-part cases real. In the same spirit, `similarity_class_type::centralizer_group_cardinality`
  gives q^2 - 1 for one 2x2 Jordan block, whose centralizer {aI + bN : a != 0} has q^2 - q
  elements.
- `parking_functions.rs` is not compiled (no `mod` in lib.rs) and duplicates `parking_function.rs`.

Revision 2 adds the following. The first two are integer-width defects of other kinds (rules R5 and
R6 of §1.7); the rest are not integer-width defects.

- R5, the size invariant. Partition-like types store machine-word parts and report their size as
  the usize sum, but no constructor checks that the sum fits: `Partition::new(vec![1 << 63,
  1 << 63]).sum()` is 0, and `Composition::new` of the same parts returns `Some` with sum 0
  (`rev2/size_inv.rs`). Every function that uses such a size is then wrong. A fix is a checked
  constructor (or a checked `sum`) for each type: `Partition`, `Composition`, `SuperPartition`,
  `PlanePartition`, `IntegerList`, `SkewPartition`, `WeightedIntegerVector`, `PartitionTuple`,
  `OrderedMultisetPartitionIntoSets`, `VectorPartition`.
- R6, machine-word values of an API (mostly signed). Each returns a value whose true size can
  leave its type, for a small input:
  - `Poset::mobius` (`posets.rs:168-202`, memoised, so fast) is 0 for the poset 0 < L antichains
    of 3 elements < 1 with L = 64 (194 elements, 0.9 s), and -9223372036854775808 for L = 63. By
    Hall's theorem mu(0, 1) is the reduced Euler characteristic of the join of L three-point sets,
    (-1)^(L-1) 2^L: -18446744073709551616 and 9223372036854775808 (gp); L = 3, 4 and 62 give 8,
    -16 and -2^62 at HEAD, as the formula says (`rev2/mobius.rs`).
  - `PartitionTuple::degree_with_multicharge` (`partitions.rs:803-812`) is -2147483648 for the
    tuple ([2^31]) (`p.sum() as i32`; truth 2^31, `rev2/valrange.rs`).
  - `SignedComposition::from_composition` (`composition.rs:204-222`) turns the part 2^63 into
    -9223372036854775808 (`part as isize`).
  - `AffinePermutation::apply` (`affine_permutations.rs:216-297`): type A with window [2, 1] gives
    `apply(i64::MAX)` = -9223372036854775808 (truth 2^63, which no i64 holds).
  - `SixVertexConfiguration::energy` (`six_vertex_model.rs:323-332`) multiplies user `i32` weights
    by vertex counts in i32.
  - `regular_sequences::sequence_sum` / `sequence_product` (`:449-475`) add and multiply usize
    terms of user sequences, and they turn an undefined term (`nth` = None) into 0 with
    `unwrap_or(0)`, which makes a sum or product of an undefined term look defined.
  - The characters `murnaghan_nakayama` and `mn_character` and the `h_vector` of `subword.rs` are
    `i64`; they also fall under R4 or R2 (§1.7).
  A fix widens each return type (to i64 or `Integer`) and adapts the consumers, so it is one new
  HANDOFF item, with these reproductions.
- `integer_lists.rs:283-300`, `:331-335`: without a sum or length constraint, `integer_lists`
  silently caps the sum at `length * 100` and the length at 101 (`unwrap_or(100)`), so it returns
  a truncated family as if it were complete. The doc does not say so.
- `integer_lists.rs:583-608 IntegerListIterator::next` always returns None ("This is a simplified
  iterator"): the iterator yields nothing for every constraint set, while `integer_lists` of the
  same constraints is non-empty.
- `integer_vectors.rs:425-470 integer_vectors_with_l1_norm_bounded`: it applies all 2^n sign
  patterns to every vector, including its zero components, so it returns duplicates: (2, 1) has 12
  entries at HEAD, of which 5 are distinct (its own doc lists the 5). Item 30 fixes only the mask
  width.
- `composition.rs:240-309 signed_compositions(n)`: it keeps only the compositions of |n| with the
  sign of n (exactly one of the 2^k sign patterns can pass its `sum() == n` filter, item 30). Its
  doc describes all sequences of non-zero integers that sum to n (an infinite family), its tests
  expect [-1, 3] and [3, -1] in a comment but never assert them, and Sage's `SignedCompositions(n)`
  is the compositions of n with each part signed (2·3^(n-1) of them).
- `designs.rs:20-60 BlockDesign::new` checks the design property only for t <= 2 and v <= 10, and
  otherwise accepts any parameters with valid blocks, even none (item 35 uses this).
- `wreath_product.rs:159-175 ColoredPermutation::order` composes the element with itself until it
  reaches the identity, and after 10^6 steps returns the step count 1000001 as if it were the
  order. HEAD (`rev2/wreath.rs`): one colour and cycles of lengths 1009 and 1013 give 1000001
  after 7.5 s; `Permutation::order` of the same permutation gives 1022117 (gp `lcm(1009,1013)`).
  The order of a coloured permutation is computable from its colored cycles (the lcm over cycles
  of the cycle length times the order of the cycle's colour sum in Z/r).
- Outside this crate: `rustmath-interfaces/src/gap_parser.rs:108-118 Permutation::order` folds the
  same `a * b / gcd(a, b)` lcm (`:321-326`) as item 29, so it has the same defect.

---

## 2. S3: an Integer-valued p(n)

### 2.1 Defect

`partition_count(n: usize) -> usize` (`partitions.rs:1261`) is the only p(n) in the workspace
(grep `fn .*partition` finds no other count). It is usize-valued: wrong or panicking from n = 417
(§1.1). Its doc says "For large n, more sophisticated methods like Hardy-Ramanujan or Rademacher
formulas would be needed". MAGMA `NumberOfPartitions(n)` (ch. 145 §145.2, ch. 18 §18.8) returns an
integer. The REPL bounds n by 416 for this reason (`repl.rs:10554-10556`).

### 2.2 API

New file `src/partition_number.rs`, re-exported from `lib.rs`:

```rust
/// MAGMA `NumberOfPartitions(n)` (ch. 145 §145.2, ch. 18 §18.8): p(n), the number of
/// partitions of n. p(0) = 1. Exact for every n <= 10^16: Euler's recurrence for n <= 1000,
/// the Hardy-Ramanujan-Rademacher series above, returned only when a rigorous enclosure of the
/// series plus Rademacher's remainder bound contains exactly one integer.
/// Cost (release): 8 ms at 10^6, 0.15 s at 10^8, 0.94 s at 10^9, 5.4 s at 10^10; it grows
/// linearly in n (the search for the terms of Selberg's formula), so n must be small in
/// MAGMA's sense.
///
/// # Panics
///
/// If n > 10^16 (`NUMBER_OF_PARTITIONS_MAX_N`): near n = 8.4e16 the first term of the series,
/// about e^C with C = (pi/6) sqrt(24n - 1), overflows MPFR's exponent range (2^30 - 1 by
/// default), and at 10^16 the run already takes at least 5e6 s. Also if the process-wide MPFR
/// exponent range has been narrowed so far that the first term does not fit.
pub fn number_of_partitions(n: u64) -> Integer
/// The largest n that `number_of_partitions` accepts.
pub const NUMBER_OF_PARTITIONS_MAX_N: u64 = 10_000_000_000_000_000;
```

In its domain it always returns the exact value; there is no Unresolved state to report, because
the loop of §2.4 terminates for every n <= 10^16 (argument in §2.5). It never returns an unproved
value. Outside the domain it panics before any work, with the message of §2.4 step 0; it never
reaches the "internal error" of step 7.

Private items in the same file: `const EULER_MAX: u64 = 1000;`, `fn partitions_euler(n: usize) ->
Integer`, `fn partitions_hrr(n: u64) -> Integer`, `struct Enclosure { lo: rug::Float, hi:
rug::Float }` with the operations listed in §2.4, `fn remainder_bound_upper(n: u64, terms: u64) ->
rug::Float`, `fn selberg_sum(n: u64, k: u64, pi: &Enclosure, prec: u32) -> Enclosure`.

Dependency: add `rug = { version = "1.24", default-features = false, features = ["float",
"integer"] }` to `rustmath-combinatorics/Cargo.toml`, the same declaration as
`rustmath-rationals/Cargo.toml:16` (Cargo.lock already has rug 1.30.0; the crate already depends
on it through rustmath-rationals and rustmath-complex, so the build and the C libraries do not
change). Why not an existing type: `rustmath_reals::RealMPFR` rounds to nearest only (none of its
operations takes a rounding direction, `rustmath-reals/src/mpfr.rs`; `RoundingMode` in
`rounding.rs` only rounds an f64 to an integer); `rustmath_rings::real_mpfi` cannot be used
(rustmath-rings depends on rustmath-combinatorics through rustmath-numberfields and
rustmath-groups: `cargo tree -p rustmath-rings`); `rustmath_complex::BigRealBall` documents that
it is not rigorous (`rustmath-complex/src/ball.rs:11-15`). `Enclosure` is about 100 lines, private,
and does only what the series needs; it is a candidate for the Phase-2 ball canonicalisation
(`ball.rs:17-35`, docs/PLAN.md).

### 2.3 Euler's recurrence (n <= 1000)

Johansson (1.2): `p(n) = sum_{k>=1} (-1)^(k+1) [p(n - k(3k-1)/2) + p(n - k(3k+1)/2)]`, from
Euler's pentagonal number theorem (1.1). `partitions_euler(n)` fills p(0..=n) in a `Vec<Integer>`:
O(n^1.5) additions of numbers of at most 105 bits. Measured: 1 ms at n = 1000, 2.35 s at 10^5
(`count time partitions`), which is why it is only used below 1000.

### 2.4 Hardy-Ramanujan-Rademacher (n > 1000)

From Johansson, *Efficient implementation of the Hardy-Ramanujan-Rademacher formula*, LMS J.
Comput. Math. 15 (2012) 341-359 (arXiv:1205.5991), quoting Rademacher, *On the partition function
p(n)*, Proc. London Math. Soc. 43 (1938) 241-254, and Whiteman, *A sum connected with the series
for the partition function*, Pacific J. Math. 6 (1956) 159-176:

- (1.4)-(1.5): `p(n) = sum_{k=1}^{N} sqrt(3/k) (4/(24n-1)) A_k(n) U(C/k) + R(n, N)`,
  `U(x) = cosh x - sinh(x)/x`, `C = (pi/6) sqrt(24n - 1)`.
- (1.8): `|R(n, N)| < M(n, N) = (44 pi^2/(225 sqrt 3)) N^(-1/2) + (pi sqrt 2/75) (N/(n-1))^(1/2)
  sinh((pi/N) sqrt(2n/3))`.
- (2.1) (Selberg, via Whiteman): `A_k(n) = (k/3)^(1/2) S_k`,
  `S_k = sum (-1)^l cos((6l+1) pi/(6k))` over `0 <= l < 2k` with `(3l^2 + l)/2 = -n (mod k)`.

Substituting (2.1) into (1.4), the square roots cancel:

    t_k = 4 U(C/k) S_k / (24n - 1),   p(n) = sum_{k=1}^{N} t_k + R(n, N).

S_1 = sqrt 3 (A_1 = 1) and S_2 = (-1)^n sqrt(3/2) (A_2 = (-1)^n) are set directly. For k >= 3,
S_k uses Algorithm 1 of the paper: `m` runs over `n + (3l^2+l)/2 mod k` and `r = 3l + 2 mod k`,
each updated with one addition and at most one subtraction of k (valid because m, r < k and
k >= 3); a cosine is evaluated only when m = 0 (O(k^(1/2)) of the 2k values of l).

Algorithm `partitions_hrr(n)` (n >= 2):

0. Domain (in `number_of_partitions`, before dispatching). `assert!(n <= NUMBER_OF_PARTITIONS_MAX_N,
   "number_of_partitions: n = {n} is above 10^16, outside the supported domain (the series terms
   would overflow the MPFR exponent range near 8.4e16; the run would take at least 5e6 s at 10^16)")`.
   Then the exponent check: `e1 = C / ln 2 + 64` in f64 (C = (pi/6) sqrt(24n - 1); log2 of the
   k = 1 term is below C / ln 2, and 64 bits cover the f64 rounding many times over), and
   `assert!(e1 < rug::float::exp_max() as f64, "number_of_partitions: the MPFR exponent range
   ({}) is too small for n = {n}")`. With MPFR's default range (`exp_max()` = 2^30 - 1 =
   1073741823, measured) this never fires in the domain: at 10^16, C / ln 2 = 3.70·10^8. The
   process-wide range is read, never changed. Measured with `bin_fix hrrdom` (cosh(C) rounded up at
   64 bits): finite at n = 10^16, 2·10^16 and 8·10^16, +inf at 8.5·10^16, 9·10^16 and 10^17.
   The precision estimate of step 3 is then at most about 3.7·10^8 + 33000 bits, below
   `u32::MAX` and `rug::float::prec_max()` (2^32 - 1), so its `as u32` cannot saturate.
1. N: the first N >= 1 with `M_f64(n, N) < 0.2` (a heuristic in f64), then while
   `remainder_bound_upper(n, N) >= 1/4`, N += 1. `remainder_bound_upper` evaluates (1.8) at
   64 bits with every operation rounded up and pi, sqrt 2 rounded up, sqrt 3 rounded down (it is
   in a denominator). So `|R(n, N)| < Mb < 1/4` with `Mb` the computed bound.
2. `v = 24n - 1` as a u64 (in the domain v < 2.4·10^17 < 2^64), loaded exactly into a 128-bit
   Float.
3. Precision of term k: `prec_k = max(64, ceil(max(0, C/(k ln 2) + log2 k + 4 - log2 v)) +
   bitlen(N) + guard + 16)` with `guard = 32` at first. The first part bounds log2 |t_k|
   (U(x) < e^x/2, |S_k| <= 2k). `prec_acc = prec_1 + 16` for the accumulator.
4. `pi = [pi down, pi up]` and `C = pi * sqrt(v) / 6` at `prec_acc`.
5. For k = 1..=N: `x = C / k`; `t = U(x) * S_k * 4 / v` at `prec_k`; skip k when the enclosure
   of S_k is exactly [0, 0] (no l satisfies the congruence: A_k(n) = 0); add t into the
   accumulator (lower ends rounded down, upper ends up).
6. `lo = acc.lo - Mb` (down), `hi = acc.hi + Mb` (up), `L = ceil(lo)`, `H = floor(hi)`.
   If L == H, return L, converted to `Integer` through its digits: `let mut bytes = vec![0u8;
   z.significant_digits::<u8>()]; z.write_digits(&mut bytes, Order::Lsf);
   Integer::new(BigInt::from_bytes_le(sign, &bytes))` (L > 0 here; the sign is still taken from
   `cmp0()`). This is the inverse of `rustmath-reals/src/mpfr.rs:368`. rug's `to_digits` would be
   shorter but exists only with rug's `std` feature, which the declaration of §2.2 leaves off.
   Measured (`bin_fix conv`): 0.06 ms at 117 kbits, 0.17 ms at 370 kbits, 0.50 ms at 1 Mbit and
   1.9 ms at 4 Mbits, against 2.7 ms, 20.7 ms, 117 ms and 1.7 s through a decimal string.
7. Otherwise double `guard` and repeat from 3. After 10 rounds (guard 32 * 2^10), panic with
   "number_of_partitions: internal error: the enclosure of p(n) did not narrow" (a bug, never a
   mathematical state; every value checked for this design (§2.6) succeeded in the first round).

`Enclosure` operations, each with MPFR directed rounding (`Float::with_val_round(prec, expr,
Round::Down / Round::Up)`, `cos_round`, `cosh_round`, `sinh_round`, `sqrt_round`):

| op | lower end | upper end |
|---|---|---|
| add, sub | `a.lo + b.lo`, `a.lo - b.hi` (down) | `a.hi + b.hi`, `a.hi - b.lo` (up) |
| neg | `-a.hi` | `-a.lo` (exact) |
| mul | min of the four end products, each rounded down | max of the four, each rounded up |
| mul or div by a positive exact u64 / Float | `a.lo * m`, `a.lo / d` (down) | `a.hi * m`, `a.hi / d` (up) |
| sqrt of an exact v | `sqrt(v)` down | `sqrt(v)` up |
| pi | `Constant::Pi` down | up |
| cos on [a, b] | `cos(a)` down minus `(b - a)` up | `cos(a)` up plus `(b - a)` up |
| U on [a, b], a > 0 | `cosh(a)` down minus (`sinh(a)` up / a, up), down | `cosh(b)` up minus (`sinh(b)` down / b, down), up |

The cos rule uses |cos t - cos a| <= |t - a| (|cos'| <= 1). The U rule uses that U is increasing
on x > 0: `U(x) = sum_{m>=1} 2m x^(2m)/(2m+1)!` (the Taylor coefficients of cosh x minus those of
sinh(x)/x), all positive. The angle of a cosine is `pi * (6l+1) / (6k)` with `6l + 1 < 12k + 1`
in u64.

### 2.5 Correctness and termination

Every MPFR operation used is correctly rounded in the requested direction (MPFR guarantees this
for `+ - * /`, `sqrt`, `cos`, `cosh`, `sinh` and `const_pi` in every rounding mode), so each
`Enclosure` contains the exact real quantity it stands for, and `[acc.lo, acc.hi]` contains
`sum_{k<=N} t_k`. By (1.8), `p(n)` lies in `[lo, hi]`. When `L == H` the interval contains exactly
one integer, which is p(n). Termination: `hi - lo <= 2 Mb + w` where w is the width of the
accumulated enclosure. `Mb < 1/4`, so as soon as `w < 1/2` the interval is shorter than 1 and
contains exactly one integer, so `L == H`. Each round adds `guard` bits to every term; w shrinks
geometrically, so the loop ends. Selberg's formula (2.1) is an identity (Whiteman 1956), and
`S_k` is summed over exactly the l of (2.1) (Algorithm 1 of the paper, with its preconditions
m < k, r < k, k >= 3 checked above). The only analytic input is Rademacher's bound (1.8) as stated
by Johansson; it needs n >= 2, which holds (n > 1000). Termination also needs every quantity to stay
inside MPFR's exponent range, which step 0 guarantees: the largest is cosh(C), with log2 below
C / ln 2 <= 3.7·10^8 < `exp_max()`, and the smallest are 1/v and the remainder bound, far above
2^(-2^30). Outside the domain the function panics in step 0 instead.

### 2.6 Validation done for this design

- `hrr check 3000`: HRR equals Euler's recurrence for every n in 2..=3000 (each in the first
  round; 1.4 s release, 2.0 s unoptimised).
- `hrr gp gp/rand_full.txt`: 500 random n < 10^7 (gp `numbpart`, full values) agree.
- `hrr fixture gp/numbpart.csv`: 71 rows (n up to 10^8) agree, 1.0 s release, 1.5 s
  unoptimised.
- p(417), p(2000), p(10^4), p(10^5), p(10^6), p(10^7), p(10^8), p(10^9) agree with gp
  `numbpart(n) % (2^127-1)` and its bit length (`gp -q gp/numbpart.gp`; p(10^9) has 116993 bits,
  residue 34163845937351525108516332470403485670).

### 2.7 Cost

| n | N (terms) | prec_1 | time | (Euler, Integer) |
|---|---|---|---|---|
| 417 | 43 | 136 | 0.5 ms | < 1 ms |
| 10^4 | 77 | 427 | 0.9 ms | 55 ms |
| 10^5 | 179 | 1225 | 3 ms | 2.35 s |
| 10^6 | 472 | 3753 | 8 ms | |
| 10^7 | 1306 | 11753 | 29 ms | |
| 10^8 | 3710 | 37055 | 0.15 s (0.23 s unoptimised) | |
| 10^9 | 10706 | 117072 | 0.94 s | |
| 3·10^9 | 17821 | 202740 | 1.97 s | |
| 10^10 | 31220 | 370110 | 5.4 s | |

N is about 0.34 n^(1/2); the Selberg search costs about N^2 = 0.12 n word steps and dominates
from 10^9. Memory: a few MB (one accumulator of prec_1 bits). Johansson's Algorithm 2 (Whiteman's
factorisation of A_k, O(n^(1/2+o(1)))) would remove the linear term; it is not needed for any n the
REPL or the tests use, and is left as a possible later speed-up (one line in the module doc).

The domain bound n <= 10^16 (§2.4 step 0) lies far beyond any feasible run: the linear term alone
gives about 5·10^-10 n s, so 5·10^6 s at 10^16, and the first terms then need 3.7·10^8 bits of
precision.

---

## 3. S4: faster callees

Each rewrite returns the same values (checked against the HEAD copies in `count.rs check` on the
grids below, and against gp on the 116-row fixture). Before/after, release, at the current REPL
limit and above it:

| function | HEAD at the limit | new at the same n | new, larger n |
|---|---|---|---|
| fibonacci_signed / lucas_signed | n = 1.3·10^6: 11.8 s | 0.027 s | 10^7: 0.50 s; 3·10^7: 2.9 s; 5·10^7: 5.4 s; 10^8: 14.3 s |
| bell_number | n = 600: 11.3 s | 0.013 s | 2000: 0.42 s; 3000: 1.40 s; 4000: 3.25 s; 5000: 6.96 s (30 MB) |
| stirling_first (1000, 999) | 0.23 s, 262 MB | < 1 ms, 2 MB | see §3.3 |
| stirling_second (1000, 999) | 0.19 s, 222 MB | < 1 ms, 2 MB | |
| eulerian (1000, 999) | 0.32 s, 371 MB | < 1 ms, 2 MB | |
| stirling_first / second / eulerian (1000, 500) | 0.22 / 0.16 / 0.21 s, 211 / 172 / 251 MB | 0.036 / 0.046 / 0.079 s, 3 MB | |
| motzkin | n = 4500: 12.0 s | 0.003 s | 10^5: 1.10 s; 2·10^5: 4.26 s; 3·10^5: 10.0 s |
| schroder_large | n = 50000: 0.62 s, 390 MB | 0.44 s, 2 MB | 10^5: 1.83 s; 2·10^5: 7.2 s; 3·10^5: 16.4 s |
| count_derangements | n = 1400: 12.9 s | < 1 ms | 10^5: 0.08 s; 2·10^5: 0.33 s; 10^6: 3.4 s; 2·10^6: 10.7 s |
| falling_factorial (2^32-1, k) | k = 10^4: 0.018 s; 10^5: 2.18 s | 0.004 s; 0.14 s | k = 10^6: 3.3 s (rising 3.5 s) |
| factorial | 10^5: 0.87 s | 0.041 s | 10^6: 1.7 s |
| multinomial | (10^4, [2]*5000): 0.071 s; (10^5, [5·10^4]*2): 1.28 s | 0.002 s; 0.003 s | (10^6, [1000]*1000): 0.79 s; (10^6, [2]*5·10^5): 1.6 s |
| catalan (with I3's binomial) | n = 316227: 15.8 s | 0.043 s | 10^6: 0.10 s; 4·10^6: 0.86 s; 10^7: 2.8 s |
| narayana (via binomial, I3) | (10^5, 5·10^4): 0.75 s | 0.006 s | (10^6, 5·10^5): 0.094 s |
| delannoy (via binomial, I3) | (2000, 2000): 0.43 s | 0.072 s | (10^4, 10^4): 2.0 s |

Unoptimised (debug, num-bigint at opt-level 0, `rd2.sh`): fibonacci(10^6) 0.33 s, bell(1000)
0.54 s, stirling1(1000, 500) 0.27 s, motzkin(10^4) 0.05 s, schroder(10^4) 0.1 s,
derangements(10^5) 1.3 s, falling(2^32-1, 10^5) 2.2 s, catalan(10^5) 0.04 s.

### 3.1 Fibonacci and Lucas by doubling (`ch144_enumerative.rs`, `lib.rs:434`, `:457`)

New `pub(crate) fn fibonacci_pair(m: u64) -> (Integer, Integer)` in `ch144_enumerative.rs`,
returning `(F_m, F_(m+1))`: from the top bit of m down, with `(a, b) = (F_k, F_(k+1))`,
`F_2k = a (2b - a)`, `F_(2k+1) = a^2 + b^2`, and on a set bit `(F_(2k+1), F_2k + F_(2k+1))`. These
are the cases m = n and m = n + 1 of the addition formula `F_(m+n) = F_m F_(n+1) + F_(m-1) F_n`
(Knuth, TAOCP vol. 1, §1.2.8); see also D. Takahashi, *A fast algorithm for computing large
Fibonacci numbers*, Inform. Process. Lett. 75 (2000) 243-246. Three multiplications per bit;
O(M(n)) overall.

- `generalized_fibonacci_number(g0, g1, n)` (`:54`): `G_n = g0 F_(n-1) + g1 F_n` for every
  integer n (it is G_0 = g0 at n = 0 because F_(-1) = 1, G_1 = g1, and it satisfies the
  recurrence by linearity). For n >= 1, `fibonacci_pair(n - 1)`. For n <= 0, m = |n| (as
  `n.unsigned_abs()`, so i64::MIN is fine), `(F_m, F_(m+1)) = fibonacci_pair(m)`,
  `F_n = (-1)^(m+1) F_m`, `F_(n-1) = F_(-(m+1)) = (-1)^m F_(m+1)`.
- `fibonacci_signed`, `lucas_signed` (`:92`, `:97`): unchanged bodies (they call the generalized
  function).
- `fibonacci(n: u32)` (`lib.rs:434`): `fibonacci_pair(u64::from(n)).0`.
- `lucas(n: u32)` (`lib.rs:457`): `L_n = 2 F_(n+1) - F_n` from `fibonacci_pair(n)`.

Checked: equal to the HEAD loops for n in -80..=80 with (g0, g1) in {(0,1), (2,1), (3,5),
(-7,11)}, and at n = 1000, 4097, -4097; fixture rows (gp `fibonacci(n)`, Lucas as
`fibonacci(n-1)+fibonacci(n+1)`).

### 3.2 Bell numbers by the Bell triangle (`lib.rs:571`, `set_partition.rs:323`, `:367`)

Aitken's array (A. C. Aitken, *A problem in combinations*, Math. Notes Edinburgh 28 (1933)
xviii-xxiii; OEIS A011971): row i starts with the last entry of row i - 1, and each next entry is
its left neighbour plus the entry above it; B_n is the first entry of row n. One vector, updated
in place:

```rust
pub fn bell_number(n: u32) -> Integer {
    let mut row = vec![Integer::one()];
    for _ in 0..n {
        let mut cur = row[row.len() - 1].clone();
        for j in 0..row.len() {
            let next = &cur + &row[j];
            row[j] = std::mem::replace(&mut cur, next);
        }
        row.push(cur);
    }
    row.swap_remove(0)
}
```

O(n^2) additions of numbers of at most bits(B_n) (n log2 n bits); memory n + 1 entries. HEAD summed
n + 1 Stirling tables (O(n^3) big operations and an (n+1)(k+1) table each).
`bell_number_optimized` (already a triangle, but allocating a row per step) becomes
`crate::bell_number(n)`; `bell_numbers_up_to` uses the same in-place row and pushes `row[0]` after
each step. The ch. 144 definition (`Bell(n) = sum_k StirlingSecond(n, k)`, digest :47) is tested as
an identity.

### 3.3 Stirling and Eulerian numbers by rolling rows (`lib.rs:506`, `:538`, `:591`)

Recurrences (Graham, Knuth, Patashnik, *Concrete Mathematics*, 2nd ed., §6.1, §6.2):
`[i j] = (i-1)[i-1 j] + [i-1 j-1]`, `{i j} = j{i-1 j} + {i-1 j-1}`,
`<i j> = (j+1)<i-1 j> + (i-j)<i-1 j-1>`, and the symmetry `<n k> = <n n-1-k>`.

Band trimming: (n, k) depends on (i, j) only if `k - (n - i) <= j <= k` (each row step lowers j by
at most 1), so row i needs only `j in lo(i) ..= hi(i)` with `lo(i) = max(1, k + i - n)` and
`hi(i) = min(i, k)` (Eulerian: `hi(i) = min(i - 1, k)`), and the recurrence at j = lo(i) also reads
the entry j - 1 of row i - 1. Eulerian first replaces k by `min(k, n - 1 - k)` (after the k >= n and
n = 0 cases). All index arithmetic is in u64.

Storage (revision 1): a band-sized buffer instead of a row of k + 1 entries. One shared private
function serves the three numbers:

```rust
/// The row recurrence of stirling_first, stirling_second and eulerian, kept in a buffer of at
/// most (band width + 1) entries. Returns the (n, k) entry and the largest buffer length (the
/// length is for the unit test that pins the memory bound).
fn band_rows(n: u64, k: u64, kind: RowKind) -> (Integer, usize)
```

`d: VecDeque<Integer>` holds the row value at j = base + t in `d[t]`; positions past `d.len()`
stand for 0. Start (row 0 for Stirling, row 1 for Eulerian): `d = [1]`, base = 0. For each row i:

1. While `base + 1 < lo(i)`: `d.pop_front()`, `base += 1` (at most once per row, since lo rises by
   at most 1).
2. While `base + d.len() <= hi(i)`: `d.push_back(0)`. These are the correct values
   `[i-1 j] = 0` for j > i - 1.
3. For j from hi(i) down to lo(i): `d[j - base] = m(i, j) d[j - base] + c(i, j) d[j - base - 1]`
   (Stirling first: m = i - 1, c = 1; second: m = j, c = 1; Eulerian: m = j + 1, c = i - j).
   Descending j means every entry read still holds row i - 1.
4. Stirling only: if base = 0, set `d[0] = 0` ([i 0] = 0 for i >= 1). For Eulerian, `d[0]` stays 1
   while base = 0 (A(i, 0) = 1).

Why the front entry is right: after step 1, base = lo(i) - 1. If base moved in step 1, the front is
j = lo(i-1), the first band entry of row i - 1, written in that row. If base did not move, then
lo(i) = lo(i-1); since lo rises by one per row once it exceeds 1, this forces lo(i) = 1 and base = 0,
and the front is [i-1 0], which step 4 keeps right (Eulerian: A(i-1, 0) = 1). Length: `d` covers base ..= hi(i), so
`d.len() = hi(i) - lo(i) + 2 <= w + 1` with w = `min(k, n - k + 1)` (Eulerian `min(k', n - k')`).
The answer is `d[k - base]` after row n.

Every entry of the band is at most the final value: each enters it with a positive integer
coefficient (m and c are at least 1 inside the band from row 2 on). So memory is at most
(w + 1)(B/8 + 32) bytes for a result of B bits, and work is O(n · w) big operations, the same as with
the full row.

Checked (`bin_fix check`, `bin_fix time`): equal to the HEAD tables for all n <= 60, k <= 62 and at
(500, 499), (500, 250), (500, 1), (300, 7), (301, 150), (301, 299), (400, 398); the buffer never
exceeds w + 1 entries (`SHOWLEN`: 3 at (10^7, 10^7 - 1), 102 at (20000, 19900), 2501 at
(5000, 2500)). Measured against the k + 1 row of the first design: `stirling_second(10^8, 10^8 - 1)`
7.9 s at 2.7 MB, where the row would need 3.2 GB; `stirling_first(10^7, 10^7 - 1)` 0.82 s at 2.7 MB;
(10^4, 5000) 34.5 s at 28 MB (the row: 34.3 s, 64 MB); (5000, 2500) 4.1 s at 9 MB (row: 4.0 s,
17 MB); Eulerian (5000, 2500) 8.4 s at 13 MB (row: 8.2 s, 21 MB). Time is unchanged within noise.
Fixture rows: gp `abs(stirling(n,k,1))`, `stirling(n,k,2)`, and
`sum(j=0,k,(-1)^j*binomial(n+1,j)*(k+1-j)^n)`.

New costs (release), for J1b's time model (memory, (w + 1)(B/8 + 32) bytes, is no longer the binding limit; the MB figures below are from the k + 1 row and are upper bounds now):

| (n, k) | stirling_first | stirling_second | eulerian |
|---|---|---|---|
| (1000, 500) | 0.036 s | 0.046 s | 0.079 s |
| (2000, 1000) | 0.29 s | 0.23 s | 0.46 s |
| (3000, 1500) | 0.91 s, 7 MB | 0.87 s | 1.73 s, 9 MB |
| (5000, 2500) | 4.0 s, 17 MB | 3.9 s | 8.2 s, 21 MB |
| (10000, 5000) | 34.3 s, 64 MB | 30.6 s | 73.3 s, 82 MB |
| (10000, 100) | 1.87 s | 0.93 s | 1.60 s |
| (20000, 19900) | 0.33 s | 0.28 s | 5.32 s (k' = 99) |
| (30000, 10) | 2.35 s | 0.29 s | 0.66 s |
| (30000, 3) / (50000, 3) / (100000, 3) | 0.57 / 1.74 / 7.67 s | (100000, 3): 0.24 s | (100000, 3): 0.65 s |

A rough fit: about 1e-9 s per unit of `n · w · B / 64`, where w is the band width
(`min(k, n - k + 1)`; Eulerian `min(k', n - k')`) and B the bit length of the result. It is within
30% on the rows with big entries ((1000, 500), (10000, 100), (10000, 5000), (100000, 3) for
`stirling_first`) and under-estimates by a factor 3 to 4 where the entries are small
((20000, 19900), `stirling_second(100000, 3)`), where a per-operation term `c0 · n · w` dominates.
J1b also needs an a-priori bound for B (for example log2(n!) for `stirling_first`); that model is
J1b's.

### 3.4 Motzkin numbers (`lib.rs:703`)

Three-term recurrence `(i+2) M_i = (2i+1) M_(i-1) + 3(i-1) M_(i-2)`, M_0 = M_1 = 1 (R. Donaghey,
L. W. Shapiro, *Motzkin numbers*, J. Combin. Theory Ser. A 23 (1977) 291-301; OEIS A001006).
Two rolling values; each step is two one-word multiplications, an addition and an exact one-word
division (`(2i+1)`, `3(i-1)`, `i+2` in u64). O(n^2 / 64) word operations instead of HEAD's
O(n^2) big multiplications. Checked equal to HEAD for n <= 60, 100, 257, 1000; fixture rows (gp
`sum(k=0,n\2,binomial(n,2*k)*binomial(2*k,k)/(k+1))`). The doc comment keeps the convolution
identity as a property and names the recurrence used.

### 3.5 Derangements by binary splitting (`derangements.rs:33`)

`D_n = sum_{k=0}^{n} (-1)^k n!/k! = sum_k (-1)^k prod_{j=k+1}^{n} j`. With
`T(a, b) = sum_{k=a}^{b} (-1)^k prod_{j=k+1}^{b} j` and `P(a, b) = prod_{j=a+1}^{b} j`:
`T(a, b) = T(a, m) P(m, b) + T(m+1, b)`, `P(m, b) = (m+1) P(m+1, b)`, `P(a, b) = P(a, m) P(m, b)`,
`D_n = T(0, n)`. Leaves of fewer than 32 terms are summed directly from k = b down to a. This is
binary splitting of Euler's recurrence `D_n = n D_(n-1) + (-1)^n` (B. Haible, T. Papanikolaou,
*Fast multiprecision evaluation of series of rational numbers*, ANTS-III, LNCS 1423 (1998)
338-350). Recursion depth log2(n/32) <= 27. O(M(B) log n) for a B-bit result. The plain
recurrence costs 1.27 s at 10^5 and 4.1 s at 2·10^5 (measured), binary splitting 0.08 s and
0.33 s. HEAD summed n + 1 rationals, each addition running Euclid's gcd on numbers of about
2 bits(n!) bits. `count_derangements_recurrence` (`:75`) is unchanged.

### 3.6 Schroder numbers (`lib.rs:734`, `:763`)

The existing recurrence `(i+1) S_i = 3(2i-1) S_(i-1) - (i-2) S_(i-2)`, S_0 = 1, S_1 = 2, with two
rolling values and u64 multipliers (§1.5). Time is about HEAD's; memory drops from n + 1 values
(390 MB at 50000) to two (2 MB). `schroder_small` unchanged (`S_n / 2` for n >= 1).

### 3.7 Products: factorial, falling and rising factorials (`lib.rs:377`)

New `src/factorials.rs` (§4) holds `pub(crate) fn product_tree(Vec<Integer>) -> Integer`
(pairwise products, level by level), `pub(crate) fn product_range_u64(lo, hi) -> Integer`
(lo >= 1; packs consecutive factors into one u64 while the product fits, then `product_tree`)
and `pub(crate) fn rising_run(a: &Integer, k: u64) -> Integer` (a >= 1; `product_range_u64` when
`a + k - 1 < 2^64`, else one leaf per factor). Balanced products: von zur Gathen and Gerhard,
*Modern Computer Algebra* (3rd ed., 2013) §10.1. `factorial(n)` becomes
`product_range_u64(2, u64::from(n))` (1 for n < 2). Revision 1 adds `pub(crate) fn
product_u64s(factors: impl Iterator<Item = u64>) -> Integer` (the same packing for any list of
word factors; `number_of_standard_tableaux` uses it for the differences l_i - l_j, §1.6 item 14).

I3 (`I3-integers-arith.md` §7) adds its own private `product_tree` over BigUint inside
rustmath-integers. Merging the two would mean new public API in rustmath-integers, which belongs to
I3's unit, so C1 keeps its `pub(crate)` tree over `Integer`; the possible merge is reported to the
orchestrator.

### 3.8 binomial, catalan, narayana, delannoy, multinomial

`binomial(n: u32, k: u32)` (`lib.rs:386`) becomes `<Integer as SageInteger>::binomial(
&Integer::from(n), &Integer::from(k))`: same values (0 for k > n; the trait function returns 0 for
0 <= n < k), and after unit I3 it is Goetgheluck's prime-splitting algorithm (I3 design §7). All
crate callers (`narayana`, `delannoy`, `delannoy_central`, `bernoulli_polynomial`, the lists in §0)
get it. `catalan` §1.3, `multinomial` §1.4. The measured "new" costs above use I3's prototype
(`binom_window`, copied into `count.rs`); with HEAD's `SageInteger::binomial` (the K-step loop)
catalan(10^5) takes 1.48 s (`count time catalan_sageint 100000`), the same as HEAD.

---

## 4. S10 (library): falling and rising factorials of any integer or ring element

### 4.1 Defect

`falling_factorial(n: u32, k: u32)` and `rising_factorial(n: u32, k: u32)` (`lib.rs:480`,
`:495`) take u32 n only; negative n and n >= 2^32 have no API, and `rising_factorial` overflows
(§1.1). `number_of_permutations(n: u32, k: u32)` (`ch144_enumerative.rs:38`) duplicates the
falling loop. Sage (secondary reference, sage.arith.misc) defines `falling_factorial(x, a)` and
`rising_factorial(x, a)` for "x - element of a ring" and integer a >= 0 as `x(x-1)...(x-a+1)` and
`x(x+1)...(x+a-1)`; Sage calls the rising one the Pochhammer symbol, as does MAGMA ch. 49
(`(a)_n = a(a+1)...(a+n-1)`, digest :292). MAGMA ch. 144 has only `NumberOfPermutations(n, k)`
(non-negative n).

### 4.2 API (new file `src/factorials.rs`, re-exported)

```rust
/// x (x-1) ... (x-k+1) for any integer x; 1 when k = 0 (Sage falling_factorial(x, k)).
pub fn falling_factorial_integer(x: &Integer, k: u64) -> Integer
/// x (x+1) ... (x+k-1) for any integer x; 1 when k = 0. The Pochhammer symbol (x)_k of MAGMA
/// ch. 49 and Sage rising_factorial(x, k).
pub fn rising_factorial_integer(x: &Integer, k: u64) -> Integer
/// The same products in any ring. `one` is the identity of the ring x belongs to; it is passed
/// in because `R::one()` panics or is an unbound constant for parent-carrying element types
/// (rustmath-core `Ring` docs, traits.rs:94-100). Factor i is x - i·one (x + i·one), built by
/// repeated subtraction (addition); a zero factor ends the product early.
pub fn falling_factorial_in<R: Ring>(x: &R, k: u64, one: &R) -> R
pub fn rising_factorial_in<R: Ring>(x: &R, k: u64, one: &R) -> R
```

Integer semantics (signs and zeros, all reduced to `rising_run` on a run of positive integers):

- falling, x >= 0: 0 if k > x (the factor 0 occurs), else `rising_run(x - k + 1, k)`.
- falling, x < 0: `(-1)^k rising_run(-x, k)`.
- rising, x > 0: `rising_run(x, k)`.
- rising, x <= 0: with m = -x, 0 if k > m, else `(-1)^k falling(m, k)`.

The generic versions collect the k factors (stopping at a zero factor, which is returned since
0 times anything is 0) and multiply them with a balanced product over R; k = 0 returns
`one.clone()`. The factors x - i·1 commute with each other, so the order of the product does not
matter even for a non-commutative R.

The existing u32 functions become wrappers (same values): `falling_factorial(n, k) =
falling_factorial_integer(&Integer::from(n), u64::from(k))`, `rising_factorial` likewise,
`number_of_permutations(n, k) = falling_factorial_integer(..)`. Keeping them avoids touching
`rustmath-jupyter` in this unit. The REPL half of S10 belongs to J1b (revision 2, review 2 m4):
J1b's revision switches the `falling_factorial` / `pochhammer` and `rising_factorial` arms to
`falling_factorial_integer` / `rising_factorial_integer` (`J1b-repl-values-and-limits.md:18`, its
§5 and plan step 7), after J1a has moved the `pochhammer` name to the rising arm
(`J1a-repl-parser-and-small.md` §3). C1 lands first, so its HANDOFF update (§8.1 step 12) records
"REPL falling_factorial / rising_factorial still take u32 n" as open and names J1b §5 as its owner;
J1b's HANDOFF step closes it.

### 4.3 Checks done

`count check`: `falling_int`/`rising_int` equal a direct product for x in -30..=30, k in 0..=40,
and for x = 2^64 - 4 (big leaves), k in {0, 1, 2, 5, 9}; equal to the HEAD loops on n <= 60,
k <= 62. Fixture rows `falling_factorial` (10 rows) and `rising_factorial` (9 rows), including
x = -4294967296, 2^64 + 5, -(2^64 + 5), 2^64 - 1, -(2^64), and k = 20000 at x = 2^32 - 1.

---

## 5. Behaviour changes and callers

- Values: none change. Every public counting function keeps its signature and returns the
  same Integer on its old domain, except where HEAD was wrong: `rising_factorial` for
  n + k - 1 >= 2^32, `multinomial` when the u32 sum wrapped, `catalan` for n >= 2^31,
  `schroder_large` for n > 715827883, `partition_count` for n >= 417 (now a documented panic
  instead of a wrong value), `count_partitions_with_max_parts` overflow (documented panic),
  `count_k_subsets`, `count_lattice_paths_ne`, `stars_and_bars` (truncation; overflow now a
  documented panic instead of 0), `CombinationGrayCodeIterator::size_hint` (upper bound None
  instead of a possibly wrong number).
- Revision 1, the sweep of §1.6 (items 7-23): every change is on inputs where HEAD returned a
  wrong value, wrapped, or (in debug) panicked with an overflow message. A wrong value becomes the
  right one where it fits (`Partition::dimension` for 21 <= n <= 35 and beyond while
  f^lambda fits, `count_integer_vectors_with_sum`, `burnside_lemma`, `count_fixed_colorings`,
  `rank_k_subset` / `unrank_k_subset` when the rank fits, `rank_binary` when n > usize::BITS,
  `TupleIterator::count` for n = 0 with k >= 2^32, `Tuple::unrank` for k >= 2^32). A value that
  cannot be represented becomes a documented panic (`count_integer_vectors_with_sum`,
  `count_integer_vectors_in_box`, `six_vertex_partition_function` and `count_tilings` for
  n > u32::MAX, `count_parking_functions`, `count_non_decreasing_parking_functions`,
  `TupleIterator::count`, `Tuple::rank`, `Partition::dimension`, `rank_k_subset`, `rank_binary`,
  `universal_cycle_subsets` / `_permutations`, `sunflower_bound`). A validation that accepted an
  impossible object now refuses it (`DeBruijnSequence::new` / `is_valid`, `OrthogonalArray::new`).
  `TupleIterator::size_hint` returns `(usize::MAX, None)` instead of a wrong exact size.
  `IrreducibleRepresentation::new` (`symmetric_group_representations.rs:35`) panics for a shape
  with f^lambda > usize::MAX (n >= 36) instead of caching a wrong dimension, and so do its two
  callers (revision 2, review 2 m3): `all_irreducible_representations(n)` (`:414`) and
  `SymmetricGroupCharacterTable::new(n)` (`:444`, through `all_irreducible_representations`) panic
  for every n >= 36, because some shape of n has f^lambda above usize::MAX (§1.6 item 14). At HEAD
  they stored wrong dimensions from n = 21. Their `# Panics` docs say so; the module is 🔴, and
  enumerating the p(36) = 17977 shapes was never cheap. On every input where HEAD was right,
  values are unchanged (the existing unit tests of these files stay).
- After unit I3 (revision 2, review 2 m3): `SageInteger::binomial` panics with "binomial: C(n, k)
  has more than 2^32 bits; not computed" when I3's lower bound on the result exceeds 2^32 bits.
  `count_k_subsets` (usize arguments), `count_lattice_paths_ne` (m + n up to 2^33), `catalan`
  (C(2n, n) with 2n up to 2^33 - 2) and `multinomial` (its binomial factors) can reach that panic
  for absurd arguments, where HEAD truncated the arguments or overflowed. Each `# Panics` doc
  gets the line "Panics (in `SageInteger::binomial`) if the result has more than about 2^32
  bits." No test reaches it in this unit; I3's own test pins the message.
- Revision 2, §1.6 items 24-36. As before, every change is on an input where HEAD returned a wrong
  value, wrapped, panicked with an overflow message in debug, or (`q_multinomial`,
  `t_multinomial`) did not return. A wrong value becomes the right one where it fits:
  `Combination::rank` / `unrank`, `Permutation::order` when the order fits, `count_superpartitions`
  (n <= 62 unchanged), `BinaryWord::from_usize` for length > usize::BITS, `Subset::next_binary` at
  n = 64, `Subset::from_bitvector` / `unrank_binary` for n > 64, `q_multinomial` /
  `t_multinomial` when the u32 sum wraps (now the zero polynomial), the four integer-vector
  functions for values of 2^63 or more, `integer_lists` when a pruning product wraps, the six
  vector-partition functions when a part is needed more than 1000 times, and
  `BlockDesign::replication_number` when lambda (v - 1) wraps but r fits. A value that cannot be
  represented becomes a documented panic: `Combination::rank`, `Permutation::order` (with the new
  `order_integer` as the exact alternative), `count_superpartitions` from n = 63,
  `BinaryWord::to_usize`, the four vector-partition counts, `replication_number`, the six Dyck-word
  statistics of item 36 (from semilength 65537), and the 11
  mask-loop functions of item 30 that keep a loop (all but `signed_compositions(_k)`) at
  n >= usize::BITS. `PermutationGrayCodeIterator::size_hint`
  returns an upper bound of None from n = 21 instead of a wrong number. The mask-loop functions
  of item 30 with 31 <= n < 64 now run their 2^n iterations (they used to return an empty or
  one-element result at once); `signed_compositions(_k)` instead become O(k) per composition with
  the same output. `count_superpartitions` becomes O(n^2) word steps instead of enumerating p(n)
  partitions. `fast_vector_partitions` and its relatives no longer rebuild the remainder for every
  copy (O(k) instead of O(k^2) per state). No caller outside the crate depends on any of these
  values (§0).
- Callers of the revision-1 functions outside rustmath-combinatorics (grep of the workspace): only
  `Partition::dimension`, in the test `rustmath-symmetricfunctions/src/classical_bases.rs:806`
  (n <= 6, unchanged). The REPL imports none of them.
- New public items: `number_of_partitions`, `NUMBER_OF_PARTITIONS_MAX_N`,
  `falling_factorial_integer`, `rising_factorial_integer`, `falling_factorial_in`,
  `rising_factorial_in`, `number_of_standard_tableaux` (all re-exported at the crate root),
  `Permutation::order_integer` (revision 2), modules `factorials`, `partition_number`. New private
  module `machine_word`.
- Speed and memory: §3. `rustmath-jupyter` calls the same functions with the same arguments, so
  its outputs and refusals are unchanged; its limits and the measured-time comments in
  `repl_limits` become stale (they only over-estimate now). Recalibration is J1b's (§7).
- `bell_number_optimized` and `bell_numbers_up_to` keep their signatures and values.
- New dependency `rug` (already in the dependency graph, §2.2).

---

## 6. Tests

New integration tests (the crate gets a `tests/` directory; fixtures under `tests/data/`, read with
`include_str!`). Expected values are from gp or GAP; each gp command is the one in
`gp/gen_counting.gp`, `gp/gen_numbpart.gp`, `gp/siblings.gp`; the CSV files are copied verbatim.

### 6.1 `tests/ch144_counting.rs` (fixture `tests/data/ch144_counting.csv` = `gp/ch144_counting.csv`, 116 rows, 5.3 KB)

Columns `fn,a,b,sign,residue,bits`: `residue = lift(Mod(v, 2^127-1))`, `bits = logint(|v|,2)+1`
(0 for v = 0). The test computes each row with the crate function, then compares the sign, the
non-negative residue mod M and the bit length. Rows (gp expression of v): `catalan`
(`binomial(2*n,n)/(n+1)`; n up to 316228), `fibonacci` (`fibonacci(n)`; -10^6..1300001),
`lucas` (`fibonacci(n-1)+fibonacci(n+1)`), `bell` (`sum(k=0,n,stirling(n,k,2))`; up to 1000),
`stirling1` (`abs(stirling(n,k,1))`), `stirling2` (`stirling(n,k,2)`), `eulerian`
(`if(k>=n, n==0&&k==0, sum(j=0,k,(-1)^j*binomial(n+1,j)*(k+1-j)^n))`; up to (1201, 600)),
`motzkin` (`sum(k=0,n\2,binomial(n,2*k)*binomial(2*k,k)/(k+1))`; up to 10^4), `schroder_large`
(`sum(k=0,n,binomial(n+k,n-k)*binomial(2*k,k)/(k+1))`; up to 20000), `schroder_small` (half of
it), `derangements` (`d=1; for(i=1,n,d=i*d+(-1)^i); d`; up to 30000), `factorial` (`n!`),
`falling_factorial` (`prod(i=0,k-1,x-i)`), `rising_factorial` (`prod(i=0,k-1,x+i)`), `binomial`
(`binomial(n,k)`). The a column is a decimal string (parsed with `Integer::from_decimal_str` for
the falling/rising rows, as u32 or i64 otherwise). The `fibonacci` and `lucas` rows call
`fibonacci_signed` / `lucas_signed`, and for a >= 0 also `fibonacci` / `lucas` (u32).

| test | content | HEAD |
|---|---|---|
| `counting_fixture_agrees_with_pari` | all 116 rows (5.6 s unoptimised with I3's binomial) | does not compile (no `falling_factorial_integer`); with that removed, rising(2^32-1, 2) panics in debug; fibonacci(±10^6), bell(1000), motzkin(10^4), derangements(10^4, 3·10^4), catalan(316228) take minutes (HEAD release: derangements(1400) 12.9 s, bell(600) 11.3 s, motzkin(4500) 12.0 s, catalan(316227) 15.8 s) |
| `rising_factorial_u32_does_not_overflow` | `rising_factorial(4294967295, 2) == 18446744069414584320` (gp `(2^32-1)*2^32`) | debug panic "attempt to add with overflow"; release 0 |
| `multinomial_sum_does_not_wrap` | `multinomial(1, &[u32::MAX, 2]) == 0`; `multinomial(0, &[1 << 31, 1 << 31]) == 0`; `multinomial(u32::MAX, &[u32::MAX - 1, 1]) == 4294967295` (gp `binomial(2^32-1,1)`); `multinomial(30, &[1,2,3,4,5,6,9]) == 29375875019771280000` (gp `30!/(1!*2!*3!*4!*5!*6!*9!)`); `multinomial(10000, &[2500; 4])` residue 129213650714704638181892182263183187668, 19981 bits (gp `10000!/(2500!^4)`); `multinomial(5, &[2, 2]) == 0` | debug panics on the first two; the third does not finish (it computes (2^32-1)!) |
| `catalan_satisfies_the_magma_recursion` | `(n+2) C_(n+1) == (4n+2) C_n` for n in 0..=400 (ch. 144 definition, digest :42) | passes |
| `fibonacci_doubling_matches_the_recurrence` | `fibonacci_signed(n)`, `lucas_signed(n)`, `generalized_fibonacci_number(g0, g1, n)` equal a test-local loop for n in -300..=300, (g0, g1) in {(0,1), (2,1), (3,5), (-7,11)}; `fibonacci(n) == fibonacci_signed(n)`, `lucas(n) == lucas_signed(n)` for n <= 300; `generalized_fibonacci_number(3, 5, 100000)` residue 97280742721680003836121413480268365629, sign 1 (gp `3*fibonacci(99999)+5*fibonacci(100000)`); `(3, 5, -100000)` residue 103771081190773177687255222800512674149, sign -1 (gp `3*fibonacci(-100001)+5*fibonacci(-100000)`); `(-7, 11, -10) == -1228` (gp) | passes (slowly) |
| `stirling_rows_match_the_full_table` | `stirling_first`, `stirling_second`, `eulerian` equal test-local copies of the HEAD table code for n in 0..=60, k in 0..=62 | passes |
| `bell_triangle_matches_the_stirling_sum` | `bell_number(n) == sum_k stirling_second(n, k)` for n <= 120 (ch. 144 definition, digest :47); `bell_number_optimized(n) == bell_number(n)`; `bell_numbers_up_to(150)[i] == bell_number(i)` | passes |
| `derangements_binary_splitting_matches_the_recurrence` | `count_derangements(n) == count_derangements_recurrence(n)` for n <= 500 | passes |
| `falling_rising_integer_values` | Sage doc examples `falling_factorial(10,3) = 720`, `falling_factorial(10,10) = 3628800`, `rising_factorial(10,3) = 1320`; k = 0 gives 1 for x in {-3, 0, 5}; `falling(5, 6) = 0`, `rising(0, 3) = 0`, `rising(-5, 6) = 0`, `rising(-5, 5) = -120` (gp `prod(i=0,4,-5+i)`) | does not compile |
| `falling_rising_in_a_polynomial_ring` | over `UnivariatePolynomial<Integer>` with x = `var()`, one = `UnivariatePolynomial::constant(Integer::one())`: falling(x, 4) has coefficients [0, -6, 11, -6, 1] and rising(x, 4) [0, 6, 11, 6, 1] (Sage doc examples; gp `prod(i=0,3,x-i)`, `prod(i=0,3,x+i)`); over `Rational`: falling(1/2, 3) = 3/8, rising(1/2, 3) = 15/8 (gp `prod(i=0,2,1/2-i)`, `prod(i=0,2,1/2+i)`); `falling_factorial_in(&Integer::from(3), 10, &one) == 0` (early zero) | does not compile |
| `u32_wrappers_agree_with_the_integer_versions` | `falling_factorial`, `rising_factorial`, `number_of_permutations` equal the Integer versions for n <= 40, k <= 45; `number_of_permutations(4294967295, 3) == 79228162403583873198531280890` (gp `(2^32-1)*(2^32-2)*(2^32-3)`) | does not compile |
| `sibling_counts_do_not_truncate` | `count_k_subsets(4294967298, 2) == 9223372043297226753` (gp `binomial(2^32+2,2)`); `count_lattice_paths_ne(u32::MAX, 1) == 4294967296` (gp `binomial(2^32,1)`); `stars_and_bars(4294967296, 2) == 4294967297` (gp `binomial(2^32+1,1)`); `count_partitions_with_max_parts(100, 5) == 46262` (gp `polcoef(prod(i=1,5,1/(1-x^i))+O(x^101),100)`); `count_partitions_with_max_parts(416, 416) == 17873792969689876004` (gp `numbpart(416)`) | 1; debug panic or 0; 1; passes; passes |
| `stars_and_bars_overflow_panics` | `#[should_panic(expected = "exceeds usize::MAX")] stars_and_bars(1000, 1000)` | returns 0 |
| `count_partitions_with_max_parts_overflow_panics` | `#[should_panic(expected = "exceeds usize::MAX")]` on (417, 417) | debug: other message ("attempt to add with overflow"); release: wraps |

The 64-bit-only rows (`count_k_subsets(4294967298, 2)`, `stars_and_bars(4294967296, 2)`,
`count_partitions_with_max_parts(416, 416)`, the should_panic rows) are under
`#[cfg(target_pointer_width = "64")]`.

### 6.2 `tests/ch145_number_of_partitions.rs` (fixture `tests/data/numbpart.csv` = `gp/numbpart.csv`, 71 rows)

| test | content | HEAD |
|---|---|---|
| `number_of_partitions_fixture` | rows `n,residue,bits` (gp `numbpart(n)`): n in {0..5, 100, 127, 128, 416, 417, 1000, 1001, 2000, 2001, 3000, 10^4, 10^5, 10^6, 10^7, 10^8}, 40 random n < 10^6 and 10 random n in [10^6, 10^8) (`setrand(20261112)`); 1.5 s unoptimised | does not compile (no `number_of_partitions`) |
| `number_of_partitions_small_values` | p(416) = 17873792969689876004, p(417) = 18987964267331664557, p(1000) = 24061467864032622473692149727991 (gp `numbpart`) | does not compile |
| `h145e1_partitions_count_the_classes_of_sym_n` | for n in 1..=12: `number_of_partitions(n) == partitions(n).len() ==` [1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77] (GAP `List([1..12], n -> NrConjugacyClasses(SymmetricGroup(n)))`, also `NrPartitions`) | does not compile |
| `partition_count_matches_the_old_dp` | 64-bit only. A test-local copy of HEAD's usize DP (`partitions.rs:1261-1276`: `dp[sum] += dp[sum - part]` over parts 1..=416, filled once) gives p(m) for every m <= 416; `partition_count(m)` equals it for every m in 0..=416, and `partition_count(416) == 17873792969689876004` (gp `numbpart(416)`). This is independent of `number_of_partitions` and pins the old values (checked in `rev1/fix.rs`: the HEAD DP equals Euler's recurrence for m <= 416) | passes (it pins HEAD's values) |
| `partition_count_overflow_panics` | `#[should_panic(expected = "number_of_partitions")] partition_count(417)` | debug: other message; release: 541220193622112941 |
| `number_of_partitions_domain_is_checked` | `#[should_panic(expected = "above 10^16")] number_of_partitions(NUMBER_OF_PARTITIONS_MAX_N + 1)`; it panics in step 0, before any work | does not compile |

### 6.3 Unit tests in `src/partition_number.rs`

- `hrr_equals_euler_up_to_3000`: `partitions_hrr(n) == partitions_euler(n)` for every n in
  2..=3000 (2.0 s unoptimised, `hrr_dbg check 3000`).
- `remainder_bound_is_below_a_quarter`: for n in {1001, 10^4, 10^6, 10^9} the chosen N has
  `remainder_bound_upper(n, N) < 0.25`; for n in {10^4, 10^6, 10^9} also `N <= ceil(n^(1/2))`
  (measured N: 47 at n = 1001, 77, 472, 10706; Johansson's N = ceil(n^(1/2)) is for the weaker
  target |R| < 1/2, so small n may need more terms than that).
- `enclosure_operations_contain_the_value`: pi at 200 bits has `lo < hi`, `hi - lo <= 2^-196`,
  and meets the interval `d ± 10^-68` around
  d = 3.141592653589793238462643383279502884197169399375105820974944592307816 (gp `\p 70` then
  `Pi`, 69 decimals printed); `mul` of [-1, 2] and [-3, 4] is exactly [-6, 8]; `cos` of
  the point [0, 0] is [1, 1]; U of the point 3 contains `cosh(3) - sinh(3)/3` evaluated at 300
  bits (round to nearest) and has width below 2^-55 at 64 bits.
- `selberg_sum_matches_the_dedekind_definition`: for k in 3..=12 and n in 0..=40, the midpoint of
  `sqrt(k/3) S_k` equals A_k(n) from (1.6), evaluated in f64 with the Dedekind sums s(h, k) of
  (1.7) as exact rationals, to within 1e-9 (a consistency check of Algorithm 1, not a proof).
- `rug_integer_converts_through_digits`: the private converter of §2.4 step 6 maps 0, 1, -1,
  2^64, -(2^64 + 5) and 3^100000 (a `rug::Integer`) to the `Integer` built with
  `Integer::from_decimal_str` of the same decimal string.
- `exponent_range_covers_the_domain`: the step-0 bound `C / ln 2 + 64` at
  n = NUMBER_OF_PARTITIONS_MAX_N is below `rug::float::exp_max()` (3.7·10^8 + 64 against
  1073741823), and `cosh` of C at 64 bits, rounded up, is finite there.

### 6.5 `tests/machine_word_counts.rs` (revisions 1 and 2: the sweep of §1.6)

Every value is from gp (`rev1/rev1.gp`, `gp/siblings.gp`) unless stated; "64" marks
`#[cfg(target_pointer_width = "64")]`. Each should_panic test names the expected message
("exceeds usize::MAX" or "exceeds u32::MAX"; revision 2 adds "subsets exceed usize::MAX" and
"leaves the i32 range"). HEAD outputs are release builds (`rev1/sweep*.rs`,
`review1/sib*.rs`); a debug build panics with "attempt to ... with overflow" in most of them, which
does not match the expected message.

| test | content | HEAD |
|---|---|---|
| `integer_vectors_with_sum_do_not_truncate` (64) | `count_integer_vectors_with_sum(3, 1 << 32) == 9223372043297226753` (`binomial(2^32+2,2)`); `(3, 4) == 15` | 1 |
| `integer_vectors_with_sum_overflow_panics` | should_panic on (1000, 1000) (C(1999, 999) has 1994 bits) | returns 0 |
| `integer_vectors_in_box_values` | `([0,0],[2,3]) == 12`; `([0, 5], [(1<<32) - 1, 4]) == 0` (an empty side wins before any product); (64) `([i64::MIN + 1], [i64::MAX]) == usize::MAX` (`2^64-1`) | passes (the last only by wrapping twice) |
| `integer_vectors_in_box_overflow_panics` (64) | should_panic on `([0; 3], [(1<<32) - 1; 3])` (`2^96`) and, in a second test, on `([i64::MIN], [i64::MAX])` (`2^64`) | both return 0 |
| `six_vertex_partition_function_rejects_huge_n` (64) | should_panic "exceeds u32::MAX" on `six_vertex_partition_function((1 << 32) + 3)`; `six_vertex_partition_function(5) == 429` stays | returns 7 |
| `parking_function_counts` | `count_parking_functions(16) == 2862423051509815793` (`17^15`); `count_non_decreasing_parking_functions(36) == 11959798385860453492` (64; `binomial(72,36)/37`); `(5) == 42` | passes (16 and 36 still fit at HEAD) |
| `parking_function_counts_overflow_panic` | should_panic on `count_parking_functions(17)` (`18^16` = 121439531096594251776); separate tests on `count_non_decreasing_parking_functions(37)` (`binomial(74,37)/38` = 45950804324621742364) and, 64, on `(1 << 40)` (must panic at once, through the usize::BITS bound) | 10759066654336942080; 9057316177202639132; the last aborts trying to allocate 2^40 + 1 words for its DP |
| `tuple_counts_are_checked` (64) | `TupleIterator::count(&TupleIterator::new(2, 63)) == 1 << 63`; `TupleIterator::count(&TupleIterator::new(0, 1 << 32)) == 0`; `TupleIterator::new(2, 64).size_hint() == (usize::MAX, None)`; `Tuple::from_vec(vec![1, 1], 1 << 31).unwrap().rank() == (1 << 31) + 1`; `Tuple::unrank(5, 1 << 40, 3).unwrap().indices() == [0, 0, 5]` | 2^63 passes; (0, 2^32) gives 1; size_hint gives (usize::MAX, Some(usize::MAX)); unrank gives None |
| `tuple_counts_overflow_panic` (64) | should_panic on `TupleIterator::count(&TupleIterator::new(2, 64))` (`2^64`), and in a second test on `Tuple::from_vec(vec![1, 1, 1], 1 << 40).unwrap().rank()` (`2^80+2^40+1`) | 18446744073709551615; 1099511627777 |
| `standard_tableaux_counts` | `number_of_standard_tableaux` of [3, 2] = 5 (H145E13), [11, 10] = 58786, [20, 1] = 20, [5, 5, 5, 5, 5] = 701149020, [9,7,6,4,3,3,2,1,1] = 40971642983700000000, [10,9,...,1] = 44261486084874072183645699204710400, [7; 7] = 475073684264389879228560, [10^9] = 1, [1; 30] = 1 (gp `hookdim` of `rev1/rev1.gp`, `rev1/maxf.gp`); for every partition of n <= 9, `number_of_standard_tableaux(p) == standard_tableaux(&p).len()` (an independent enumeration, `tableaux.rs:419`; it agrees with HEAD's `dimension` there, `rev1/syt.rs`); `Partition::dimension` equals it for [11, 10], [20, 1], [5; 5] and (64) [9,7,6,4,3,2,2,1,1] = 7821859115070000000 | does not compile (new function); HEAD `dimension`: 16335, 5, 317 |
| `partition_dimension_overflow_panics` | should_panic on `Partition::new(vec![8,7,6,5,4,3,2,1]).dimension()` (29258366996258488320 > 2^64) | returns 2 |
| `colorings_keep_the_colour_count` (64) | `burnside_lemma(&PermutationGroup::cyclic(1), (1 << 32) + 1) == 4294967297`; `count_fixed_colorings(&Permutation::identity(2), (1 << 32) + 2) == 18446744090889420804` (`(2^32+2)^2`); `burnside_lemma(&PermutationGroup::cyclic(4), 2) == 6` stays | 1; 4 |
| `tilings_reject_n_above_u32` (64) | should_panic "exceeds u32::MAX" on `count_tilings(2, (1 << 32) + 1, &Polyomino::dominoes())`; values `count_tilings(2, 10, ..) == 89` (`fibonacci(11)`), `(3, 4) == 11` | returns 1 |
| `subset_ranks_are_exact` (64) | `Subset::from_elements(vec![1, 2, 3], 5_000_000_001).unwrap().rank_k_subset(3) == Some(12499999997500000000)` (`binomial(5*10^9,2)`); `Subset::unrank_k_subset(12499999997500000000, 5_000_000_001, 3)` has elements [1, 2, 3]; `Subset::from_elements(vec![0], 100).unwrap().rank_binary() == 1`; `from_elements(vec![63], 64).rank_binary() == 1 << 63` | 9223372036854775807; [1, 705032705, 3557549058] after 3.6 s (a debug build loops for minutes); 0 |
| `subset_ranks_overflow_panic` | should_panic on the rank of {9999997, 9999998, 9999999} in 10^7 (`binomial(10^7,3)-1` > 2^64), and in a second test on `rank_binary` of {64} in 100 | 645920003284035455; 0 |
| `sequence_validations_check_the_length` | `DeBruijnSequence::new(vec![], 2, 64).is_none()`; (64) `DeBruijnSequence::new(vec![0], 2, 1 << 32).is_none()`; `OrthogonalArray::new(64, 1, 2, vec![]).is_none()`; `DeBruijnSequence::new(debruijn_sequence(2, 3), 2, 3)` is still Some | Some; Some; Some |
| `universal_cycles_refuse_lengths_above_usize` | should_panic on `universal_cycle_permutations(70, 70)` (70! > 2^64), and in a second test on `universal_cycle_subsets(70, 35)` (C(70, 35) = 112186277816662845432 > 2^64) | Some([0]) at once; runs out of memory |
| `sunflower_bound_is_checked` | `SetSystem::sunflower_bound(20, 2) == 2432902008176640000` (`20!`); the existing (3, 3) = 48 stays; should_panic in a second test on `sunflower_bound(30, 3)` (`30!*2^30`) | 20! passes; (30, 3) returns 18446744073709551615 |

Revision 2 adds these tests to the same file (items 24-36; gp commands in `rev2/rev2.gp`,
`rev2/rev2.gp.extra`, `rev2/qm.gp`; HEAD outputs from the `rev2/*.rs` reproductions, release):

| test | content | HEAD |
|---|---|---|
| `combination_ranks_are_exact` (64) | `Combination::from_vec(vec![1, 2, 3], 5_000_000_001).unwrap().rank() == 12499999997500000000` (`binomial(5*10^9,2)`); `Combination::unrank(12499999997500000000, 5_000_000_001, 3).unwrap().indices() == [1, 2, 3]`; for n <= 8 and every k, the ranks of `combinations(n, k)` are 0, 1, ... in order and `unrank` inverts them | 3276627960645224192; [3, 566066340, 1487858385]; the small cases pass |
| `combination_rank_overflow_panics` (64) | should_panic ("exceeds usize::MAX") on the rank of [9999997, 9999998, 9999999] in 10^7 (`binomial(10^7,3)-1` > 2^64) | wraps |
| `permutation_gray_code_size_hint_is_an_upper_bound` (64) | `PermutationGrayCodeIterator::new(20).size_hint() == (0, Some(2432902008176640000))` (`20!`); `new(21).size_hint() == (0, None)` (`21!` = 51090942171709440000 > 2^64) | (0, Some(14197454024290336768)) for 21 |
| `superpartition_counts_are_checked` | `count_superpartitions(n) == superpartitions(n).len()` for n <= 12; (64) `count_superpartitions(62) == 15969100102326684686` (`polcoef(prod(i=1,62,1/(1-2*x^i))+O(x^63),62)`); should_panic in a second test (64) on `count_superpartitions(63)` (31938200243029484454) | 62 passes; 63 returns 13491456169319932838 |
| `binary_words_respect_usize_bits` (64) | a 100-bit word with only bit 63 set has `to_usize() == 1 << 63`; `BinaryWord::from_usize(usize::MAX, 64).to_usize() == usize::MAX`; `BinaryWord::from_usize(1, 65)` has exactly one set bit, bit 0; should_panic in a second test on `to_usize` of a 65-bit word with bits 0 and 64 set | `from_usize(1, 65)` sets bits 0 and 64; `to_usize` gives 2 |
| `subset_binary_order_at_the_word_size` (64) | `Subset::from_elements(vec![], 64).unwrap().next_binary()` has elements [0]; the full subset of 64 has `next_binary() == None`; `Subset::from_bitvector(1, 65).elements() == [0]`; `Subset::unrank_binary(5, 100).elements() == [0, 2]`; the empty subset of 63 still gives [0] | None; Some(empty set); [0, 64]; [0, 2, 64, 66] |
| `permutation_order_is_exact` | a test-local permutation with one cycle of each length in L: L = the primes up to 47 and 94 (422 points) gives `order() == 614889782588491410` (64) and `order_integer() == 614889782588491410` (`lcm(concat(primes(15),[94]))`); L = the primes up to 53 (381 points) gives `order_integer() == 32589158477190044730` (`lcm(primes(16))`); should_panic in a second test (64) on `order()` of the latter | 26163907895633379; 14142414403480493114 |
| `subset_mask_loops_refuse_word_size_sets` (64) | each call below is run in `std::panic::catch_unwind` and must panic with a message containing "subsets exceed usize::MAX": `all_binary_words(64)`; `all_order_ideals` and `all_antichains` of the chain 0 < 1 < ... < 63; `all_subwords(&[0u8; 64])`; `subword::SubwordComplex::new(vec![0u8; 64], vec![0u8])`; `shadow()` of `SetSystem::from_sets(64, vec![(0..64).collect()])`; `shade()` of `SetSystem::from_sets(64, vec![vec![]])` (the complement of the empty set in the ground set 0..64 has 64 elements); `admissible_cuts` of a path of 65 vertices built with `RootedTree::from_children`; `OrderedMultisetPartitionIntoSets::new(vec![(0..64u32).collect()]).finer()`; `integer_vectors_with_l1_norm_bounded(64, 0)`. Small values stay: the chain of 5 has 6 ideals and 6 antichains, the path of 5 vertices has 5 admissible cuts, `all_binary_words(3).len() == 8` | release: one-element or empty results at once (`rev2/sweep6.rs`: 1, 1, 1, ...); debug: "attempt to shift left with overflow", another message |
| `signed_compositions_need_no_sign_loop` | `signed_compositions_k(31, 31)` and `(64, 64)` are each the single composition [1; k]; `signed_compositions_k(-3, 2)` is [[-1, -2], [-2, -1]]; `signed_compositions(4).len() == 8` (the 2^3 compositions of 4, all positive) and `signed_compositions(-4)` is the same list negated; the existing tests (`composition.rs:657-740`) stay | (31, 31) and (64, 64) give []; the others pass |
| `q_and_t_multinomials_check_the_sum` | `q_multinomial(1, &[u32::MAX, 2])` and `t_multinomial(1, &[u32::MAX, 2])` are the zero polynomial; `q_multinomial(5, &[2, 2, 1])` and `t_multinomial(5, &[2, 2, 1])` have coefficients [1, 2, 4, 5, 6, 5, 4, 2, 1] (gp `Vecrev(qf(5)/(qf(2)^2*qf(1)))`, `rev2/qm.gp`) | the first two do not return (30 s under 4 GB, `rev2/qmult.rs`); debug: "attempt to add with overflow" |
| `integer_vectors_keep_values_above_i64` (64) | `integer_vectors_with_sum(1, 1 << 63)` is the one vector [9223372036854775808] (`2^63`); `integer_vector_sum_iter(2, 1 << 63).next()` has components [0, 9223372036854775808] | [-9223372036854775808]; [0, -9223372036854775808] |
| `integer_lists_prune_without_wrapping` (64) | `integer_lists(&IntegerListConstraints::new().sum(3 << 62).length(3).min_part(1 << 62).max_part(1 << 63))` is [[2^62, 2^62, 2^62]] (the only list: three parts of at least 2^62 with sum 3·2^62); the existing doc example (length 2, sum 3: 4 lists) stays | [] |
| `vector_partition_counts_are_exact` | `fast_vector_partitions(&[100]) == 190569292` (`numbpart(100)`); `count_vector_partitions_max_parts(&[100], 100) == 190569292`; `fast_vector_partitions_with_max_part(&[1001], Some(&[1])) == 1` and `vector_partitions_with_max_part(&[1001], Some(&[1])).len() == 1` (`polcoef(1/(1-x)+O(x^1003),1001)`); `fast_vector_partitions_with_max_part(&[1001, 1], Some(&[1, 1])) == 2` (`rev2/rev2.gp.extra`) | 190569292 passes; 0; 0; 1 |
| `vector_partition_count_overflow_panics` (64) | should_panic ("exceeds usize::MAX") on `fast_vector_partitions(&[417])` (`numbpart(417)` = 18987964267331664557); about 1.2 s unoptimised | 541220193622112941 |
| `replication_number_is_exact` (64) | `BlockDesign::new(1 << 33, 3, 2, 1 << 32, vec![]).unwrap().replication_number() == Some(18446744071562067968)` (`2^32*(2^33-1)/2`); the Fano plane (the 7 lines of `rev2/repl_num.rs`) gives Some(3) | Some(9223372034707292160); Some(3) |
| `dyck_area_does_not_wrap` | `DyckWord::new` of N^65536 E^65536 (`true` 65536 times, then `false` 65536 times) has `area() == 2147450880` (`65536*65535/2`); should_panic ("leaves the i32 range") in a second test on the same word of semilength 65537 (`65537*65536/2` = 2147516416 > 2^31 - 1); the existing `dyck_word.rs` tests stay | 2147450880 passes; -2147450880 |

Checking that the new tests fail on HEAD (review 2, m5). `universal_cycle_subsets(70, 35)` runs out
of memory at HEAD, and `q_multinomial(1, &[u32::MAX, 2])` / `t_multinomial(..)` do not return
(`rev2/qmult.rs`: 30 s under a 4 GB limit). On the shared machine, build the test binary first
(`cargo test -p rustmath-combinatorics --test machine_word_counts --no-run`, without a limit,
because rustc needs more address space), then run only those tests under the limit: `bash -c
'ulimit -v 4000000; timeout 60 target/debug/deps/machine_word_counts-<hash> universal_cycles
q_and_t'`. Or skip them in the HEAD check. Every other new test fails on HEAD within a second, in release or debug.

### 6.6 New unit tests in the source files

- `machine_word.rs`: `checked_pow_usize` at (2, 63), (2, 64), (3, 40), (3, 41), (0, 0), (0, 2^40),
  (1, 2^40), (2, 2^32 + 1), (17, 15), (18, 16) (values in `rev1/fix.rs`); `binomial_usize_checked`
  equals an exact `Integer` binomial for m <= 100 and every j (debug-cheap), C(67, 33) =
  14226520737620288370 fits, C(68, 34) is None, C(2^32 + 2, 2) = 9223372043297226753, C(2 usize::MAX - 1,
  usize::MAX - 1) is None (64); `falling_factorial_usize_checked` (5, 3) = 60, (10, 2) = 90, (5, 0) = 1,
  (5, 6) = 0, (20, 20) = 2432902008176640000, (21, 21) None (64), (70, 70) None (the tests moved from
  `debruijn.rs:691-703`); `u32_arg` panics on 2^32 (64). Revision 2: `binomial_usize_checked(3, 5)
  == Some(0)` and `(0, 1) == Some(0)` (j > m, review 2 m1); `subset_mask_count` gives 1 at 0,
  2^63 at 63 (64) and panics at 64 (64) with "subsets exceed usize::MAX".
- Revision 2: `subword_complex.rs`, a unit test of the private `all_subsets`: a 3-element set gives
  8 subsets, a 64-element set panics (64). `set_system.rs`, a unit test of the private
  `is_shattered` with a 64-element subset: panics (64). `superpartitions.rs`: the DP equals the
  enumeration for n <= 12 (also in §6.5). `permutations.rs`: `lcm` of (2^62, 3) is None (64) and of
  (6, 4) is 12.
- `lib.rs`: `schroder_step_factors_do_not_wrap` (§1.5); `band_rows_stay_band_sized`: for (n, k) in
  (10^5, 10^5 - 1), (2000, 1990), (500, 250), (500, 1), the buffer length returned by `band_rows`
  is at most min(k, n - k + 1) + 1 for both Stirling kinds and min(k', n - k') + 1 for Eulerian
  (k' = min(k, n - 1 - k)), and `stirling_second(10^5, 10^5 - 1) == stirling_first(10^5, 10^5 - 1)
  == 4999950000` (both are C(n, 2); gp `binomial(10^5,2)`), `eulerian(10^5, 10^5 - 2)` has
  100000 bits (A(n, 1) = 2^n - n - 1).

### 6.4 Existing tests

All stay: `lib.rs` tests (values unchanged), `set_partition.rs:517`, `derangements.rs:200-340`,
`ch144_enumerative.rs` tests, `partitions.rs:1647` (`test_partition_count`), `path_tableaux.rs:584`,
`enumeration.rs:1764`. Revision 1: the tests of the files of §1.6 stay with their values
(`partitions.rs:1681-1689` dimensions 2, 16, 1; `tuple.rs:504`; `set_system.rs:1392-1403`;
`six_vertex_model.rs:602-660`; `parking_function.rs:430-455`; `integer_vectors.rs:749-754`; the
subset, designs and group_action tests), except `debruijn.rs:691-703`, whose helpers are deleted:
those assertions move to `machine_word.rs` (§6.6). Revision 2: the tests of the files of items
24-36 stay with their values (`combinations.rs`, `enumeration.rs`, `superpartitions.rs`,
`binary_words.rs:627`, `subset.rs`, `permutations.rs`, `posets.rs:865`, `:923`, `:1090`,
`subword.rs:615`, `subword_complex.rs`, `set_system.rs`, `grossman_larson.rs:613-627`,
`multiset_partition_into_sets_ordered.rs:625`, `composition.rs:657-740`, `integer_vectors.rs`,
`q_analogue.rs`, `t_sequences.rs`, `integer_lists.rs`, `vector_partition.rs`, `designs.rs`). None
of them uses an input where HEAD was wrong.

---

## 7. For unit J1b: REPL limits that depend on these callees

`rustmath-jupyter/src/repl.rs`, `mod repl_limits` (`:10463-10561`) and the arms that use them.
"Old" is HEAD at the limit (release); "new" is after C1 (and I3 for the binomial rows).

| limit (line) | REPL arm (line) | callee | old cost at the limit | new cost at the old limit | new costs above it |
|---|---|---|---|---|---|
| `CATALAN_N = 316_227` (:10492) | catalan (:2207) | `catalan` | 15.8 s | 0.043 s | 10^6: 0.10 s; 4·10^6: 0.86 s; 10^7: 2.8 s (34 MB). The `4n^2 <= BINOMIAL_WORK` link to the binomial model no longer describes the cost |
| `FIBONACCI_N = 1_300_000` (:10495) | fibonacci/fib (:2224), lucas (:2241) | `fibonacci_signed`, `lucas_signed` | 11.8 s | 0.027 s | 10^7: 0.50 s; 3·10^7: 2.9 s; 5·10^7: 5.4 s (36 MB); 10^8: 14.3 s |
| `BELL_N = 600` (:10497) | bell (:2256) | `bell_number` | 11.3 s | 0.013 s | 2000: 0.42 s; 3000: 1.40 s; 4000: 3.25 s; 5000: 6.96 s (30 MB) |
| `TABLE_BYTES = 400_000_000` (:10503) with `table_bytes`/`check_table` | stirling1 (:2276), stirling2 (:2295), eulerian (:2413) | `stirling_first`, `stirling_second`, `eulerian` | (1000, 999): 0.23 / 0.19 / 0.32 s at 262 / 222 / 371 MB | < 1 ms, 2 MB | the memory is now at most (w + 1)(B/8 + 32) bytes, w = min(k, n - k + 1) (Eulerian min(k', n - k')), B the result's bits: 28 MB at (10^4, 5·10^3), 2.7 MB at (10^8, 10^8 - 1) (7.9 s). The binding cost is time, table in §3.3; the table-size model `(n+1)(k+1)` should become a time model, with this memory bound as a second check if J1b wants one |
| `MOTZKIN_N = 4_500` (:10505) | motzkin (:2453) | `motzkin` | 12.0 s | 0.003 s | 10^5: 1.10 s; 2·10^5: 4.26 s; 3·10^5: 10.0 s (3 MB) |
| `SCHRODER_N = 50_000` (:10508) | schroder/schroder_large (:2509), schroder_small (:2524) | `schroder_large`, `schroder_small` | 0.62 s, 390 MB | 0.44 s, 2 MB | 10^5: 1.83 s; 2·10^5: 7.2 s; 3·10^5: 16.4 s (3 MB) |
| `DERANGEMENTS_N = 1_400` (:10510) | derangements/subfactorial (:2556) | `count_derangements` | 12.9 s | < 1 ms | 10^5: 0.08 s; 2·10^5: 0.33 s; 10^6: 3.4 s (18 MB); 2·10^6: 10.7 s |
| `FACTORIAL_WORK = 2.5e12` (:10517) | multinomial (:2309-2341), falling_factorial/pochhammer (:2355-2366), rising_factorial (:2380-2395) | `multinomial`, `falling_factorial`, `rising_factorial` | falling(2^32-1, 10^5) (work 3.2e11): 2.18 s | 0.14 s | falling(2^32-1, 10^6): 3.3 s; rising: 3.5 s (63 MB); 10^6! 1.6 s; multinomial (10^6, [2]*5·10^5): 1.6 s; (10^6, [1000]*1000): 0.79 s; (10^5, [5·10^4]*2): 0.003 s (HEAD 1.28 s). The k·(k·bits) model now over-estimates by about 15x at k = 10^5; the product-tree cost is about M(B) log k with B = k·bits |
| `PARTITIONS_N = 416` (:10556) | partitions/partition_count (:2540-2542) | `partition_count` (usize) | small | same | switch to `number_of_partitions(n: u64)` (n <= 10^16, §2.2): 10^6: 8 ms; 10^8: 0.15 s; 10^9: 0.94 s; 3·10^9: 2.0 s; 10^10: 5.4 s; cost about 5e-10·n s from 10^9 |
| `BINOMIAL_WORK = 4e11` (:10489), `narayana` arm (:2428-2439) | narayana | `narayana` (via `binomial`) | (10^5, 5·10^4): 0.75 s | 0.006 s | (10^6, 5·10^5): 0.094 s |
| `DELANNOY_SUM_CAP = 20_000` (:10559), `delannoy_work` | delannoy (:2467-2490) | `delannoy`, `delannoy_central` (via `binomial`) | (2000, 2000): 0.43 s | 0.072 s | (10^4, 10^4): 2.0 s |

REPL arm changes that C1 makes possible (not done here):

- `pochhammer` is an alias of `falling_factorial` (`repl.rs:2348`); MAGMA ch. 49 and Sage use the
  rising factorial for the Pochhammer symbol. J1a moves the name (its §3).
- `falling_factorial` and `rising_factorial` can take any integer n (`falling_factorial_integer`,
  `rising_factorial_integer`); the u32 n range (`:2355`) and the rising guard
  `0..=u32::MAX - k + 1` with "n + k - 1 must fit in 32 bits" (`:2382`) are then not needed.
  Owner: J1b §5 and plan step 7 (revision 2; J1b's revision took it). C1 records it as open in
  HANDOFF until J1b lands (§8.1 step 12).
- partitions: call `number_of_partitions(n)` with a u64 n (at most `NUMBER_OF_PARTITIONS_MAX_N`)
  and a time limit; drop the machine-word rationale of `PARTITIONS_N`. Owner: J1b §5 as well
  (`PARTITIONS_N: u64`, provisional 10^10 from §2.7).
- The `repl_limits` comments that quote HEAD times (`:10479-10561`) and the help line become
  stale; J1b recalibrates them (J1b design, the protocol after its limits table).

---

## 8. Plan

### 8.1 Files, in implementation order

1. `rustmath-combinatorics/Cargo.toml`: add `rug` (§2.2). `Cargo.lock` changes with it (the
   `rustmath-combinatorics` entry gains `rug`; no new package).
2. `src/machine_word.rs` (new, private; `mod machine_word;` in lib.rs). Header: MAGMA ch. 144
   §144.2 and §144.3 (the counts and subset ranks it serves). Contents and unit tests: §1.6, §6.6,
   with `subset_mask_count` (revision 2).
3. `src/factorials.rs` (new). Header: MAGMA ch. 144 §144.2 `Factorial`, `NumberOfPermutations`
   (digest `Chapter_144_Enumerative_Combinatorics.md:36-37`), the Pochhammer symbol of ch. 49
   (`Chapter_49_...md:292`); Concrete Mathematics §2.6 (falling and rising factorial powers);
   von zur Gathen and Gerhard §10.1 (balanced products); Sage `falling_factorial`,
   `rising_factorial` as the secondary reference. Contents §3.7 (with `product_u64s`) and §4.2,
   unit tests for `product_range_u64` (empty range 1, a range crossing the u64 packing boundary,
   lo near u64::MAX), `product_u64s` and `rising_run` with big a.
4. `src/partition_number.rs` (new). Header: MAGMA ch. 145 §145.2 and ch. 18 §18.8
   `NumberOfPartitions`; Johansson 2012 (equations (1.2), (1.4)-(1.8), (2.1), Algorithm 1);
   Rademacher 1938; Whiteman 1956; MPFR directed rounding. Contents §2.2-§2.4 with
   `NUMBER_OF_PARTITIONS_MAX_N` and the step-0 checks, the digit conversion of step 6, unit tests
   §6.3. Port from `hrr.rs`: keep its linear f64 scan for N, the rigorous recheck and its u64 v
   (enough in the domain); add the retry cap of §2.4 step 7 (the prototype loops without one).
5. `src/lib.rs`: `mod machine_word; pub mod factorials; pub mod partition_number;`, re-export the
   new public functions and the constant; add the ch. 144 / ch. 18 citation to the comment at
   `:371`; rewrite `factorial` (§3.7), `binomial` (§3.8, `use rustmath_integers::SageInteger`),
   `multinomial` (§1.4), `catalan` (§1.3), `fibonacci`, `lucas` (§3.1), `falling_factorial`,
   `rising_factorial` (§4.2 wrappers), `stirling_first`, `stirling_second`, `eulerian` through
   `band_rows` (§3.3), `bell_number` (§3.2), `motzkin` (§3.4), `schroder_large` with
   `schroder_step_factors` (§1.5, §3.6); doc comments name the algorithm and its source; unit
   tests of §6.6.
6. `src/ch144_enumerative.rs`: `fibonacci_pair`, `generalized_fibonacci_number` (§3.1),
   `number_of_permutations` (wrapper); update the module table (`:12-20`).
7. `src/derangements.rs`: `count_derangements` (§3.5); update its doc (cost, method) and the
   header.
8. `src/set_partition.rs`: `bell_number_optimized`, `bell_numbers_up_to` (§3.2).
9. `src/partitions.rs`: `partition_count` (doc: "no usize of 64 bits or fewer can hold p(n) for
   n > 1000"), `count_partitions_with_max_parts` (§1.6 items 1-2); fix the doc at `:1257-1260`;
   `number_of_standard_tableaux` and `Partition::dimension` (§1.6 item 14; header line citing
   MAGMA ch. 145 §145.4.9 and [Ful97] §4.3).
10. The other files of §1.6, each with a documented `# Panics` where one is added:
    `src/subset.rs` (items 3, 17, 18), `src/path_tableaux.rs:293` (item 4), `src/enumeration.rs`
    (items 5, 6, 23), `src/integer_vectors.rs` (items 7, 13), `src/six_vertex_model.rs` (item 8),
    `src/parking_function.rs` (items 9, 10), `src/tuple.rs` (items 11, 12), `src/group_action.rs`
    (item 15), `src/tiling.rs` (item 16), `src/debruijn.rs` (items 19, 20; the moved helper tests),
    `src/designs.rs` (item 21), `src/set_system.rs` (item 22).
    Revision 2 (items 24-36, same rule for `# Panics` docs, plus a cost line where item 30 says
    so): `src/combinations.rs` (item 24; delete `binomial_usize`), `src/enumeration.rs` (item 25;
    delete the private `factorial`), `src/superpartitions.rs` (item 26), `src/binary_words.rs`
    (items 27, 30), `src/subset.rs` (item 28), `src/permutations.rs` (item 29, with
    `order_integer`; header line citing MAGMA ch. 58 `Order`), `src/posets.rs`, `src/subword.rs`,
    `src/subword_complex.rs`, `src/set_system.rs`, `src/grossman_larson.rs`,
    `src/multiset_partition_into_sets_ordered.rs`, `src/composition.rs` (item 30; in
    `composition.rs` the sign loop goes and `n.abs()` becomes `n.unsigned_abs()`),
    `src/q_analogue.rs`, `src/t_sequences.rs` (item 31), `src/integer_vectors.rs` (items 30, 32),
    `src/integer_lists.rs` (item 33), `src/vector_partition.rs` (item 34), `src/designs.rs`
    (item 35), `src/dyck_word.rs` (item 36). The `# Panics` docs of `count_k_subsets`, `count_lattice_paths_ne`, `catalan`,
    `multinomial` get I3's binomial line, and those of `all_irreducible_representations` and
    `SymmetricGroupCharacterTable::new` get the n >= 36 line (§5).
11. `tests/data/ch144_counting.csv`, `tests/data/numbpart.csv` (copy the two gp outputs), then
    `tests/ch144_counting.rs`, `tests/ch145_number_of_partitions.rs`,
    `tests/machine_word_counts.rs` (§6).
12. Trackers. `docs/magma/status_overrides.csv`: `ch018.NumberOfPartitions` (partial -> done,
    path `partition_number.rs:<line of number_of_partitions>`, note "Euler (1.2) for n <= 1000,
    certified HRR (Johansson 2012) above, n <= 10^16; tests ..."), add `ch145.NumberOfPartitions`
    (done, same path), `ch145.NumberOfStandardTableaux` (done, path
    `partitions.rs:<line of number_of_standard_tableaux>`, note "Frobenius formula with
    cancellation or the hook length formula, exact Integer; tests standard_tableaux_counts,
    H145E13"), `ch018.Multinomial` (new note: u128 sum, product of binomials, tests), then
    `python3 tools/magma_tracker/build.py`.
    `PROJECT_TRACKER.md`: `:1305` (classical counting: name the algorithms and the new tests),
    `:1312` (hook length formula: was wrong from n = 21, now exact via `number_of_standard_tableaux`),
    a new row for `number_of_partitions`, the `partitions.rs` rows that cite `partition_count`,
    `:1264` (hall polynomials: the one-row case is wrong too), `:1272` (plane partitions: the box
    count is wrong), `:1276` (similarity class types: `centralizer_group_cardinality` is wrong),
    `:1293` (`parking_functions.rs` is not compiled), and a row or note for `polya_enumeration`.
    Revision 2: a note on `:1320` (posets: the order-ideal and antichain lists were empty from 31
    elements, now exact up to 63, and `mobius` is i64, R6), rows or notes for vector partitions
    (counts now checked, the 1000-copy cap removed), `integer_lists` (the silent caps of 100, the
    empty `IntegerListIterator`), `signed_compositions` (semantics), `Permutation::order` /
    `order_integer`, the Dyck-word statistics (i32 sums now checked, item 36), and `:1280` (designs: `BlockDesign::new`
    checks only v <= 10).
    `docs/HANDOFF.md:296-305`:
    - S2, S3, S4 and the library part of S10: done (with the commit).
    - One line for §1.7 (revision 2; it does not claim completeness): "Machine-word sweep of
      rustmath-combinatorics (design C1 §1.6-§1.7): the patterns `as u32`, saturating and
      wrapping arithmetic, `unwrap_or(0)` after a conversion, `<<`, `>>`, `pow`, `product`,
      `sum`, narrowing `as` casts, functions that return a machine word and multiply or count,
      and helpers named like binomial or factorial were checked. The hits that are not fixed are
      bounded by memory, by an enumeration or by 2^32 steps of work, guarded, or in modules flagged
      broken (ranking, hall_polynomial, similarity_class_type, ncsym, knutson_tao, cluster_complex,
      the symmetric-group characters). Known to remain: the size invariant of partition-like types
      and the signed i32/i64 values (two items below). Arithmetic that these patterns do not
      match was not checked."
    - Still open, owned by J1b §5 and plan step 7 (revision 2, review 2 m4): "REPL
      falling_factorial / rising_factorial still take u32 n (S10, REPL part): switch the arms to
      `falling_factorial_integer` / `rising_factorial_integer`"; "REPL partitions: switch to
      `number_of_partitions` (u64 n, at most 10^16)". J1b's HANDOFF step closes both. The REPL
      limit recalibration stays with J1b (its design covers it).
    - New items from §1.8: `count_plane_partitions_in_box` wrong for (2,2,2) (15, gp 20);
      `polya_enumeration` uses w^L for a cycle of length L (C_2 example); `hall_polynomial`
      one-row case and `centralizer_group_cardinality` wrong; `parking_functions.rs` not
      compiled.
    - New items from §1.8, revision 2, each with its reproduction: the size invariant
      (`Partition::new(vec![1 << 63, 1 << 63]).sum() == 0`; checked constructors for about ten
      types); signed machine-word values (`Poset::mobius` of the 194-element layered poset is 0, truth -2^64;
      `degree_with_multicharge`, `SignedComposition::from_composition`, `AffinePermutation::apply`,
      `SixVertexConfiguration::energy`, `sequence_sum` / `sequence_product` with `unwrap_or(0)`);
      `integer_lists` caps sums at 100 per part and lengths at 101 without saying so;
      `IntegerListIterator::next` always returns None; `integer_vectors_with_l1_norm_bounded`
      returns duplicates ((2, 1): 12 entries, 5 distinct); `signed_compositions` keeps only
      uniform signs, unlike its doc and Sage; `BlockDesign::new` checks designs only for v <= 10;
      `ColoredPermutation::order` returns 1000001 when the order exceeds 10^6;
      and, outside the crate, `rustmath-interfaces/src/gap_parser.rs:108` `Permutation::order`
      has the lcm overflow of item 29.

### 8.2 Acceptance

```bash
cargo build -p rustmath-combinatorics
cargo test -p rustmath-combinatorics --no-fail-fast          # 1485 + new unit, 187 + doc, 3 integration files
cargo test -p rustmath-combinatorics --release --test ch144_counting --test ch145_number_of_partitions --test machine_word_counts
cargo clippy -p rustmath-combinatorics                        # no new warnings in the touched files
cargo test -p rustmath-jupyter --no-fail-fast                 # REPL callers: values and refusals unchanged
cargo test -p rustmath-symmetricfunctions -p rustmath-quantumgroups --no-fail-fast   # Partition users
cargo test -p rustmath-groups --no-fail-fast                  # revision 2: Permutation::order users (tests)
cargo check --workspace --tests
python3 tools/magma_tracker/build.py
```

When confirming that the new tests fail on HEAD, run `universal_cycles_refuse_lengths_above_usize`
and `q_and_t_multinomials_check_the_sum` only under a memory limit, as §6.5 says (review 2, m5).

### 8.3 Risks

- Order with I3. C1 relies on `SageInteger::binomial` keeping its signature (I3 changes only its
  body). If I3's fast body is not merged when C1 starts, values are unchanged but the fixture
  rows catalan(31622..316228) and binomial(200000, 100000) take minutes in a debug build; then
  drop those rows to n <= 10^4 and say so in the commit.
- Rademacher's bound. The proof of exactness depends on (1.8) as quoted by Johansson. The
  checks of §2.6 (every n in 2..=3000 against Euler, 579 gp values) would very likely expose a
  wrong bound; the tests keep the 2..=3000 check and 71 gp rows.
- MPFR semantics through rug. Directed rounding of `cos`, `cosh`, `sinh`, `sqrt`, `const_pi` is
  an MPFR guarantee; the code must use the `_round` variants or `with_val_round` everywhere in
  `Enclosure` (a `Float::with_val` without a direction would round to nearest and break the
  enclosure). Review point.
- Test time. New tests add about 9-10 s in a debug build (§6). The counting fixture is the
  largest part (5.6 s); rows can move to `--release`-only `#[ignore]` tests if the budget needs
  it.
- `partition_count(417)` now panics instead of returning a wrong number. No caller passes n > 416
  (the REPL limits it to 416).
- Revision 1 adds panics to 14 public functions (§5). Each replaces a wrong value or a wrap, and
  none has a caller outside its file except `Partition::dimension`, whose only outside caller is a
  test with n <= 6. Inside the crate, `IrreducibleRepresentation::new` now panics for n >= 36
  shapes with f^lambda > usize::MAX; no test builds one. The should_panic tests name the new
  messages, so a debug build of HEAD (which panics with "attempt to ... with overflow") fails them.
- 64-bit assumptions: the tests marked 64 in §6.5 are under `#[cfg(target_pointer_width = "64")]`;
  the helpers themselves use `usize::BITS` and u128, so they are right on any width.
- Revision 1 test time: below 1 s in a debug build (the slowest is the rank of the last 3-subset of
  10^7, which panics after about 4·10^5 loop steps).
- Clippy may ask for `u64::from` over `as` in the new code; follow it.
- Revision 2, item 30: for 31 <= n < 64 the mask-loop functions now run their 2^n iterations,
  where HEAD returned an empty or one-element result at once. A caller that relied on the quick
  (wrong) answer would now run for a long time. No such caller exists (§0: the in-crate callers
  pass small sets, and nothing outside the crate calls these functions).
- Revision 2 test time: about 1.5 s more in a debug build. The slowest test is
  `fast_vector_partitions(&[417])` (about 1.2 s unoptimised, `rev2/bin_vpart_dbg`); the others are
  instant. If the budget needs it, that one test can move behind `#[cfg_attr(debug_assertions,
  ignore)]`, since §8.2 runs `machine_word_counts` in release as well.
- Revision 2 leaves two integer-width classes open on purpose (R5 size invariant, R6 signed
  values, §1.7). Both are recorded in HANDOFF with reproductions, and the HANDOFF line says the
  sweep covers stated patterns only, so no later reader takes the crate as overflow-free.

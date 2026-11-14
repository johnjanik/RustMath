# Review 4 of design C1-combinatorics (rustmath-combinatorics), including Revisions 1 to 3

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/C1-combinatorics/review4/`.
Files there: `rd.sh` builds against the HEAD debug rlib. `r4.rs` / `r4.txt` hold the HEAD reproductions.
`sample4.gp` / `sample4.txt` hold 30 gp reference values. `mine4.gp` and `hrr_vs_gp.txt` compare 5 fresh
p(n) with the HRR prototype. `h145.g` is the GAP H145E1 check. `branch.py` / `branch.txt` count the
partitions that take the hook branch. `hookbranch.gp` / `hookbranch.txt` hold gp values for hook-branch
shapes. `clippy_head.txt` is clippy at HEAD. `times4.txt` holds the prototype timings. The binaries are
deleted. Tree: HEAD `dc367a2`, clean before and after.

Verdict: **approved**. No blockers. The blocker of review 3 is resolved, and so are its eight minors.
The minors below are small and can be folded into the implementation.

## Review 3 blocker and minors: resolved

- B1 (`next_binary` / `prev_binary` for n > 64). Reproduced at HEAD (`r4.txt`, debug rlib). For n = 65,
  `next_binary` of {} and of {0..63} gives None. `prev_binary` of {0} and of {64} gives None.
  `next_binary` at n = 64 panics, and so do `from_bitvector(1, 65)` and `unrank_binary(5, 100)`.
  `rank_binary` of {0} in 100 gives 0. Item 28 now computes both successors on the sorted element
  list. `Subset.elements` is private, and every constructor keeps it sorted (`from_elements`
  sorts and dedups), so the precondition holds. I re-derived both rules. Increment: drop the run
  0..i-1 and add i, the first non-element; it is None iff i = n. Decrement: remove the smallest
  element i and add 0..i-1; it is None iff the set is empty. Both outputs are sorted. n = 0 gives
  None both ways, as at HEAD. The prototype `rev3/nb3.rs` implements exactly the text. The tests
  are the ones review 3 asked for (§6.5 `subset_binary_successors_for_any_n`), and `prev_binary`
  is named in §1.6, §5 and §8.1 step 10.
- m1: `hook_length_dimension` becomes a call of `number_of_standard_tableaux`. Its only callers
  are its own test and the re-export (my grep agrees). But the new equality test is now a tautology
  (m1 below).
- m2: num-bigint. `Cargo.toml:86` declares it at workspace level, and `rustmath-rationals/Cargo.toml:14`
  declares it the same way. `Integer::new(BigInt)` is at `integer.rs:70`, and rustmath-integers does
  not re-export `BigInt`. Resolved.
- m3: `(0, None)` is a valid hint in every state. At HEAD a debug build panics in `Tuple::rank`, and
  (2, 63) after two calls gives (2^63 - 2, Some(2^63 - 2)) (`r4.txt`). n = 0 never allocates (`new`
  sets `current = None`).
- m4: the HEAD check runs in three steps. I checked that every other test of `machine_word_counts`
  fails fast at HEAD in debug. Its first failing assertion or panic comes before any long loop
  (for example, `rank_k_subset` fails before the slow unrank). The `--skip` substrings match only
  the three intended tests.
- m5: an early return for the empty partition, with the test rows. At HEAD both functions give 1
  (`r4.txt`).
- m6: §8.2 adds the four other dependents. I checked the list: nine Cargo.toml files declare the
  dependency, and no file of rustmath-interfaces or rustmath-topology names the crate.
- m7: "Pochhammer" moves to the rising factorial (`lib.rs:477`, `:492`; the ch. 49 digest at :292
  says (a)_n = a(a+1)...(a+n-1)). The new test files get headers.
- m8: the Temperley-Lieb `catalan_number` (`temperley_lieb_algebra.rs:317`) is reported. gp's
  simulation of the wrapping loop gives 3793109252770211, and C_33 = 212336130412243110.

## What I reproduced or re-checked (agrees with the design)

- Clippy (revision 3 changes the acceptance). `cargo clippy -p rustmath-combinatorics` exits 101 at
  HEAD with exactly the five deny-level errors named in §8.2 (`binary_words.rs:317`,
  `debruijn.rs:100`, `path_tableaux.rs:446`, `super_tableaux.rs:297`, `word.rs:252`) and 320
  warnings. So "no lint on a changed line" is a fair acceptance.
- `partition_to_dyck_path` (§1.8). The code makes total_north the number of parts and total_east
  the largest part, so the `while` loop never ends when lambda_1 > length. (1, 1) gives 2 N and 1 E
  (`r4.txt`).
- P11. My own grep of the sources, which also matches `.ok()?`, `to_u32()?` and `try_into()?`,
  finds exactly `gelfand_tsetlin.rs:219`, `subset.rs:304`, `:315` and `tuple.rs:87`. The `as u32`
  sites of `rev2/nontest.txt` fall into 26 files. Each file is an item or a §1.7 row. I checked
  the lib, subset, tuple, tiling, six_vertex, integer_vectors, schubert, ribbon, ASM and
  cyclic_sieving lines one by one.
- Line citations at `dc367a2`: `subset.rs:303-309`, `:314-320`; `tuple.rs:63-89`, `:252-334`;
  `young_representations.rs:13`, `:52`, `:286`; `lib.rs:362`, `:477`, `:492`; the REPL arms (:2219
  to :2568) and `repl_limits` (:10491 to :10587); `docs/HANDOFF.md:335-342`; the PROJECT_TRACKER
  rows :1266 to :1331. All hold.
- HANDOFF coverage. S2 covers the five named sites, and the u32 `binomial` goes through
  `SageInteger::binomial` (§3.8). S3, S4 and the library half of S10 are covered. The REPL half is
  J1b's and is recorded as open. Nothing is declared out of scope.
- HRR. `hrr.rs`, built against the HEAD rug rlib, agrees with five fresh gp values: n = 1003,
  77777, 5000011, 31415926 and 271828182 (residue mod 2^127 - 1 and bit length, `hrr_vs_gp.txt`).
  It also passes the 71-row fixture. p(10^9) takes 0.81 s, with N = 10706, prec_1 = 117072 and the
  residue that §2.6 quotes. p(10^8) takes 0.10 s. Both are at or below §2.7.
- Prototypes: `count check` passes ("check ok"), and `count fixture` passes on all 116 rows.
- S4 costs, measured as HEAD/new ratios with the same build (`times4.txt`; num-bigint unoptimised,
  so absolute figures run above the design's release figures): bell(600) 44.8 s against 0.067 s;
  derangements(1400) 4.3 s against 1 ms; eulerian(1000, 999) 1.0 s at 371 MB against under 1 ms
  at 2 MB. The claimed speed-ups hold.
- Algorithms re-derived again: Fibonacci doubling with the negative-index signs, Lucas
  L_n = 2F_(n+1) - F_n, the in-place Aitken array (traced to B_3), the derangement splitting
  identities, the Motzkin (M_4 = 9) and Schroder (S_3 = 22) recurrences, the falling and rising
  sign and zero cases, the band-row starts for k = 0, Frobenius f(2, 1) = 2, and the checked DP of
  `count_partitions_with_max_parts`. That check is exact: p_<=j(i) does not decrease in i or j.
- gp (`sample4.txt`): every value I sampled equals the design. That includes the hook dimensions
  (40971642983700000000, 475073684264389879228560, 44261486084874072183645699204710400, 58786,
  701149020, 29258366996258488320, 7821859115070000000, 20, 5, and 1 for []). It also includes
  binomial(5·10^9, 2), C(2^32+2, 2), 17^15, 18^16, C_36, C_37, 3^41, both lcm rows, the
  superpartition coefficients 62 and 63, 2^32(2^33-1)/2, the Dyck areas, 46262, the multinomials,
  the generalized Fibonacci residues (the negative row is `lift(Mod(v, M))` of the negative v, as
  §6.1 defines), C(67, 33), the falling and rising rows, the q-multinomial, numbpart(416), (417),
  (1000), 30!·2^30, C(70, 35) and the MacMahon (2,2,2) = 20.
- GAP: `NrConjugacyClasses(SymmetricGroup(n))` and `NrPartitions` for n = 1..12 both give
  [1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77] (H145E1).
- Callers: 30 files outside the crate name it, the same as review 3. No file outside `subset.rs`
  calls `next_binary`, `prev_binary`, `rank_binary`, `unrank_binary` or `from_bitvector`. No file
  outside the crate calls `TupleIterator`, `Tuple::rank` or `hook_length_dimension`.
- No facade or silent truncation. Every new failure is a documented panic, a None with its stated
  meaning, or a valid bound. The HRR retry cap panics as an "internal error" and never returns an
  unproved value. No bounded search decides anything.

## Minor

### m1. The new f^lambda equality test is a tautology, and the hook branch has almost no independent coverage

After item 14, `hook_length_dimension(p)` is literally `number_of_standard_tableaux(p)`. So
`hook_length_dimension_is_number_of_standard_tableaux` (§6.5, n <= 12) cannot fail. The only
independent checks of `number_of_standard_tableaux` are the enumeration for n <= 9 and the listed
values. No listed value takes the hook branch: [20,1], [5^5], [7^7], the staircase and
[9,7,6,4,3,3,2,1,1] all satisfy r(r-1)/2 <= n. The enumeration for n <= 9 reaches the hook branch on
one shape only, (5, 1^4) (`branch.txt`: 1 of 97). For n <= 12 there are 5 such shapes, and for n <= 20
there are 311 of 2714.

Required change: make the test independent. Compare `number_of_standard_tableaux` with a test-local
copy of HEAD's `hook_length_dimension` body (n! / prod hooks in `Integer`), in the style of
`partition_count_matches_the_old_dp`, for every partition of n <= 20. That is cheap in debug: 20! has
62 bits. Also add one hook-branch value with its gp command: f(6, 1^5) = 252 = `binomial(10,5)`, or
f(2001, 1^2000) = `binomial(4000,2000)` (residue 85248900849712891484894804346834018665 mod 2^127 - 1,
3994 bits; `hookbranch.txt`). Keep the [11, 10] row.

### m2. The `--release` acceptance run is not a timing check, and the debug run already catches more

§8.2 runs the three integration files with `--release` "because a release build wraps where a debug
build panics". This round's rule allows release builds only for timing checks. A debug run already
fails on everything the release run would catch. Suppose the new code still had an unchecked
overflow before its check. In debug it panics with "attempt to ... with overflow", which fails
every `should_panic(expected = ...)` test and every value test. In release it can wrap to a value
that passes by luck. An example is `count_integer_vectors_in_box` at HEAD, which is right only
because it wraps twice. Required change: drop the `--release` line. Also drop the §8.3 suggestion
to `cfg_attr(debug_assertions, ignore)` the 1.2 s `fast_vector_partitions(&[417])` test, since it
relies on that run. Otherwise, mark the run optional and say why.

### m3. Nits

- Two §6.5 rows need the 64-bit mark: `count_parking_functions(16) == 2862423051509815793`
  (`parking_function_counts`) and `sunflower_bound(20, 2) == 2432902008176640000`
  (`sunflower_bound_is_checked`). On a 32-bit target both values exceed usize::MAX, so the new code
  panics there.
- The Verdicts row for S2 lists `hook_length_dimension` with the "integer-width defects found" of
  revision 3. At HEAD it is exact: it computes in `Integer`, and `rev3/syt3.txt` shows equal values.
  Reword it as a merge with no defect.
- `prev_binary` of {i} returns the i elements 0..i-1. For a universe near usize::MAX the result
  cannot be allocated. Allocation then fails, or `Vec` panics with "capacity overflow". The true
  answer cannot be stored, so this is honest, and it is better than HEAD's None. The doc's "O(n)"
  could say that time and memory are linear in the smallest element plus the subset size.

## Checks that the design passes

- Coverage of S2, S3, S4 and S10 (library). §7 covers every REPL limit for J1b. The revision-3
  additions are reproduced, and their fixes are sound.
- Tests use gp and GAP reference values; I sampled them above. The new tests fail on HEAD: they
  do not compile, they get a wrong value, they get another panic message, or they abort or time out
  (the three tests that run alone). The one exception is m1.
- The plan order can be implemented. The acceptance commands cover the crate, the REPL, the
  Partition, Permutation and other direct dependents, and `cargo check --workspace --tests`.

# Review 2 of the implementation of unit C1-combinatorics

Tree: HEAD `dc367a2` plus the uncommitted C1 changes (35 modified files, 3 new source files, a new
`tests/` directory, `Cargo.lock`). Design: `design/C1-combinatorics.md` with Revisions 1-3 (review 4
approved it). Report: `impl/C1-combinatorics.md`, including "Fix-up round 1". Previous review:
`impl/C1-combinatorics.review1.md` (one blocker B1, minors m1-m3). Scratch:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/C1-combinatorics.rev2/`
(`src/rb.sh` builds a scratch program against the current debug rlib; `src/chk.rs` holds my
independent checks; `src/tm.rs` is a debug timing sample; `spot.gp`, `vp.gp` hold my gp values;
`numbpart.csv` / `counting.csv` are the regenerated fixtures; `*.log` are the acceptance runs;
`lintshort.py` matches clippy lints with changed lines). I did not edit, commit, stash or check out
anything.

Verdict: **approved**. No blocker. The B1 fix is right, the review 1 minors are all folded, and
every acceptance command passes. Two minors and one nit below; none of them blocks.

## Review 1 findings, re-checked

### B1 (mismatched `max_part` made two public functions loop forever): fixed

- `vector_partition.rs:118` (`fast_vector_partitions_with_max_part`) and `:303`
  (`vector_partitions_with_max_part`) check `actual_max.len() == vector.len()` with `assert_eq!`
  before any work, so release builds refuse too. Both functions document it under `# Panics`.
- The other four public entry points (`fast_vector_partitions`, `vector_partitions`,
  `count_vector_partitions_max_parts`, `count_vector_partitions_exact_parts`) build their parts from
  the vector itself, so every part has the vector's length and is non-zero. The copy loops then end
  after at most max_i v_i + 1 copies; the two bounded helpers also stop at `max_parts` /
  `exact_parts`. No other caller can reach `count_partitions_with_parts` or
  `generate_partitions_with_parts` with other parts (both private).
- New tests `vector_partition_count_refuses_a_longer_max_part`,
  `vector_partition_list_refuses_a_longer_max_part` and
  `vector_partition_count_refuses_a_shorter_max_part` (`tests/machine_word_counts.rs:626-652`)
  name the new messages. They fail at HEAD (an index-out-of-bounds panic for the first two, the
  value 0 for the third) and would hang on the round-0 tree. All three pass in 0 s.
- No workspace caller passes a `max_part` (grep), so the behaviour change has no caller to adapt.
- Values: `fast_vector_partitions(&[300])` = 9253082936723602 = gp `numbpart(300)`;
  `fast_vector_partitions(&[12, 12])` = 379693 and `vector_partitions(&[6, 6]).len()` = 1043, both
  equal to an independent gp DP over all parts (a, b) != (0, 0) (`vp.gp`).

### m1-m3: folded as asked

- m1: `integer_lists.rs:293-297` and `:336-339` now say that saturation only keeps the bound from
  wrapping, and that the bounds themselves are the reported caps of §1.8. The file stays CRLF on
  every line (830 of 830).
- m2: `remainder_bound_upper` (`partition_number.rs:299-327`) now holds N, n - 1 and 2n exactly in
  128-bit floats and rounds each of the two divisions up at 64 bits (`up(p, &nn_exact /
  &nm1_exact)`, `up(p, &two_n_exact / 3u32)`). I traced every operation: each positive factor is
  rounded up, each denominator (sqrt 3, sqrt N) down, `sinh` is increasing, so the result is an
  upper bound of M(n, N) as stated in the doc.
- m3: `tiling.rs:280-281`, the three "on 64-bit targets" notes in
  `symmetric_group_representations.rs`, `set_system.rs:864` and `enumeration.rs:1053-1057` read
  right now.

## Minor

### m1. `count_superpartitions` allocates n + 1 words before its first check

`superpartitions.rs:453-463` allocates `vec![0usize; n + 1]` and only then runs the DP whose
checked additions give the documented panic. The count exceeds usize::MAX from n = 63 (gp:
c(62) = 15969100102326684686, c(63) = 31938200243029484454, `spot.gp`), so every n >= 63 first
allocates 8(n + 1) bytes for nothing. For n of about 2^32 or more the allocation itself fails, and
the process aborts instead of panicking with the documented message:

| call (debug rlib, `ulimit -v 4000000`) | result |
|---|---|
| `count_superpartitions(62)` | 15969100102326684686 (gp) |
| `count_superpartitions(10^8)` | allocates 800 MB, then the documented panic (exit 101) |
| `count_superpartitions(1 << 40)` | "memory allocation of 8796093022216 bytes failed", aborted (exit 134) |

This is not a regression: HEAD enumerated `partitions(n)`, which exhausts memory far sooner. But
the design applied exactly this rule to its sibling `count_non_decreasing_parking_functions`
(panic at once for n >= usize::BITS, test `non_decreasing_parking_count_refuses_huge_n`), and the
fix is one line. Suggested change: before the allocation,
`if n >= usize::BITS as usize { panic!("{MSG}") }`. It is exact by the doc's own argument: c(m + 1)
>= 2 c(m) (append a part 1, circled or not), so c(n) >= 2^n > usize::MAX. A should_panic test on
`1 << 40` would pin it.

### m2. `falling_factorial_in` / `rising_factorial_in` hold all k factors at once

`factorials.rs:166-189` (`product_of_steps_in`) pushes every factor into a `Vec` before the
balanced product. For a ring whose elements stay small and never reach zero (for example an
element of GF(p^n) outside the prime field), `falling_factorial_in(x, 10^9, one)` holds 10^9
elements at once, although the result is one field element. Time is O(k) anyway, so this is only a
memory cost, bounded by the argument, and not a correctness defect. Optional: a binary-counter stack
of partial products (merge the top two entries while they cover equal counts) gives the same
balanced product with O(log k) live elements. Or say in the doc that memory is linear in k.

## Nit

- `factorials.rs:13`: "so n! is about n / 4 leaves" holds only for n below 2^16 (four 16-bit
  factors per u64). In general a leaf holds about 64 / log2(n) factors, for example 3 near 10^5.

## What I checked, and found right

- Every item of §1.6 (1-36), §2, §3 and §4 that review 1 listed is still in place after the fix-up;
  the only code changes of the fix-up are the two asserts and the two directed divisions, plus
  comments.
- HRR (`partition_number.rs`): re-derived independently. The term 4 U(C/k) S_k / (24n - 1) is
  (1.4) with A_k = (k/3)^(1/2) S_k, so the square roots cancel; S_1 = sqrt 3 and
  S_2 = (-1)^n sqrt(6)/2 follow from A_1 = 1 and A_2 = (-1)^n. `selberg_sum` is Algorithm 1
  (`paper/johansson2012.txt:196-215`); one subtraction suffices for m and r since both stay below k
  and k >= 3. Every `Enclosure` operation is directed; the cos rule uses |cos'| <= 1, U is
  increasing. The final test ceil(lo) = floor(hi) proves that [lo, hi] holds exactly one integer;
  that holds even if the precision were below the magnitude (then lo and hi are integers). p(n) is
  in it because |R| < M <= Mb. An exactly zero S_k encloses a true 0, so skipping it is right. The
  retry loop runs 11 rounds and then panics as an internal error; it never returns an unproved
  value. `choose_terms` ends because M(n, N) tends to 0 like N^(-1/2). The domain and exponent checks
  run before any work.
- Independent value checks (`src/chk.rs`, all pass): `stirling_first`, `stirling_second` and
  `eulerian` against a naive full table for every (n, k) with n <= 40 and k <= n + 1;
  `number_of_standard_tableaux` against the plain hook product on 400 random shapes with n in
  30..=250 (391 take the Frobenius branch, 9 the hook branch); `next_binary` / `prev_binary` as
  inverses on 2000 subsets of a 70-element set; `falling_factorial_integer`,
  `falling_factorial_in` and `rising_factorial_integer` for x = ±123456789012345678901234567890;
  `rising_factorial(u32::MAX, 3)`. gp values (`spot.gp`): `partition_count(416)` =
  17873792969689876004 and 417 panics; p(10^6) and p(1234567) mod 10^20; `stirling(30, 12, 1)`,
  `stirling(30, 12, 2)`, A(12, 5) = 162512286, M_20 = 50852019, S_20 = 17518619320890, D(25) =
  `round(25!/exp(1))`, B(30); the max f^lambda exceeds 2^64 - 1 at n = 36 and not at n = 35
  (the `Partition::dimension` doc); C_36 fits and C_37 does not (the parking-function doc).
- Fixtures: `gp -q tests/data/gen_numbpart.gp` and `gp -q tests/data/gen_counting.gp` give files
  byte-identical to `tests/data/numbpart.csv` and `tests/data/ch144_counting.csv` (36 s).
- Algebra re-derived: derangement binary splitting (T(a, b) = T(a, m) P(m, b) + T(m + 1, b), the leaf
  loop, D(0) = 1, D(1) = 0); Fibonacci doubling and the signs of F_(n-1), F_n for n <= 0; Lucas;
  Motzkin's three-term recurrence (M_2..M_4); the Bell triangle step; the Frobenius formula with
  l_1 <= n and its O(n - lambda_1 + r^2) factor count; `binomial_usize_checked` (monotone exact
  divisions, at most usize::BITS + 1 steps, the u128 overflow case); Horner rank in `Tuple::rank`;
  the subset successors; the lcm (a / gcd) b; the exactness notes of the vector-partition,
  superpartition and `count_partitions_with_max_parts` checks; `signed_compositions` (`true` means
  positive in `from_composition`, so the uniform pattern has the sign of n).
- No facade and no undecided value returned as decided: every refusal is a documented panic, a None
  with its stated meaning, or a valid size-hint bound. `exceeds_sunflower_bound`,
  `DeBruijnSequence::is_valid` and `OrthogonalArray::new` decide correctly when the bound exceeds
  usize::MAX (no object in memory can have that many entries).
- No new unbounded loop: every new loop is bounded by its arguments (and the vector-partition loops
  now also by the length check). The only new up-front allocation larger than the result is m1.
- Tests: no assertion was weakened. The only removed assertions are the nine of the two debruijn
  helper tests, all present in `machine_word.rs::falling_factorial_usize_checked_values`. The new
  tests fail on HEAD as the report's HEAD check says, and quote gp or GAP commands.
- Callers: the only caller outside the crate of a function whose behaviour changed is
  `rustmath-jupyter/src/repl.rs:2554` (`partition_count`, limited to 416 by `repl_limits`); its
  tests pass.
- Costs (debug sample, `src/tm.rs`): p(10^8) 0.19 s, p(10^9) 1.46 s (doc: 0.94 s release),
  fibonacci(10^6) 0.20 s, bell(1000) 0.31 s, stirling_first(1000, 500) 0.20 s,
  count_derangements(10^5) 0.99 s, factorial(10^5) 0.56 s, f(2001, 1^2000) 4 ms. All in line with
  the documented release costs.
- Headers cite the MAGMA chapters and the papers; no `unsafe`; no em-dash in added lines; CRLF files
  stay CRLF; `Cargo.lock` gains only the two dependency names of the crate's entry; no change to
  PROJECT_TRACKER.md, `docs/`, `rustmath-misc/output.txt` or a `.tmp_dir_*`.
- Unused-import warnings in touched files are all present at HEAD (review 4's
  `clippy_head.txt`) or in unchanged test code (`set_partition.rs:497`).

## Acceptance (debug)

| command | result |
|---|---|
| `cargo build -p rustmath-combinatorics` | ok (`build.log`) |
| `cargo test -p rustmath-combinatorics --no-fail-fast` | unit 1509 passed, 0 failed, 7 ignored; `ch144_counting` 14 passed (6.6 s); `ch145_number_of_partitions` 6 passed (1.5 s); `machine_word_counts` 53 passed (0.3 s); doctests 189 passed, 11 ignored (`test_crate.log`) |
| `cargo clippy -p rustmath-combinatorics` | exits 101 on the five older deny lints only (`binary_words.rs:336`, `debruijn.rs:101`, `path_tableaux.rs:454`, `super_tableaux.rs:297`, `word.rs:252`; `clippy.log`) |
| `cargo clippy -p rustmath-combinatorics --all-targets -- -A clippy::inherent_to_string_shadow_display -A clippy::while_immutable_condition -A clippy::erasing_op` | ok; 0 of 401 lint sites on changed or new lines (`clippy_allowed.log`, `lintshort.py`) |
| `cargo test -p rustmath-symmetricfunctions -p rustmath-quantumgroups --no-fail-fast` | 200 passed, 0 failed, 1 ignored |
| `cargo test -p rustmath-groups --no-fail-fast` | 668 passed, 0 failed, 4 ignored |
| `cargo test -p rustmath-algebras -p rustmath-liealgebras -p rustmath-crystals -p rustmath-topology --no-fail-fast` | 1403 passed, 0 failed, 2 ignored |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 64 passed, 0 failed (442 s; `t_jupyter.log`) |
| `cargo check --workspace --tests` | ok, 0 errors (`check_ws.log`) |

The `--release` run of §8.2 stays dropped, as review 4 m2 asked and this round's rule allows
release builds only for timing checks. The debug runs are the stronger check: an unchecked overflow
before a new check would panic with "attempt to ... with overflow" and fail the
`should_panic(expected = ...)` tests. No release build was made.

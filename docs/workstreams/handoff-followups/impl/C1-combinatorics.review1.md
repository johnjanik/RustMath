# Review 1 of the implementation of unit C1-combinatorics

Tree: HEAD `dc367a2` plus the uncommitted C1 changes (35 modified files, 3 new source files, a new
`tests/` directory). Design: `design/C1-combinatorics.md` with Revisions 1-3; review 4 approved it.
Report: `impl/C1-combinatorics.md`. Scratch:
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/C1-combinatorics.rev1/`
(`src/rb.sh` builds a scratch program against the current debug rlib; `src/vp.rs` is the
reproduction of B1; `src/tm.rs` is a debug timing sample; `check.gp` / `check_gp.txt` hold my gp
values; `counting.csv` / `numbpart.csv` are my regenerated fixtures; `*.log` are the acceptance
runs). I did not edit, commit, stash or check out anything.

Verdict: **not approved**. One blocker (B1). The rest of the unit is sound, and the fix is small.

## Blocker

### B1. Removing the 1000-copy cap makes two public vector-partition functions loop forever

`vector_partition.rs`: the design (§1.6 item 34) deletes the caps `if k > 1000 { break }` because
"no part is zero, so the subtraction fails after at most max_i v_i + 1 copies". That holds only
when the parts have as many components as the vector. `fast_vector_partitions_with_max_part` and
`vector_partitions_with_max_part` build the parts from the caller's `max_part`
(`generate_all_parts(actual_max)`, `:113`, `:285`), whose length is not checked. The new
`subtract_part` (`:175`) zips `remainder` with `part`, so it silently ignores the extra
components. A part such as [0, 1] then acts as the zero vector on a 1-component vector, the
subtraction never fails, and the `loop` (`:159`, `:316`) never ends.

Reproduced (`src/vp.rs`, debug rlib, `ulimit -v 2000000; timeout 10`):

| call | HEAD `dc367a2` | this tree |
|---|---|---|
| `fast_vector_partitions_with_max_part(&[5], Some(&[1, 1]))` | panics at once: "index out of bounds: the len is 1 but the index is 1" (`vector_partition.rs:148`) | does not return (exit 124) |
| `vector_partitions_with_max_part(&[5], Some(&[1, 1]))` | the same panic (`:307`) | does not return (exit 124), and `next_parts` grows by one vector per turn |
| `fast_vector_partitions_with_max_part(&[5, 5], Some(&[1]))` | 0 | 0 |

So a malformed argument that used to fail at once now hangs: a new unbounded loop reachable from
two public functions (both re-exported at `lib.rs:266-267`). The last row gives 0 in both trees.
With a shorter `max_part` the extra components of the vector can never reach 0, so that 0 has no
clear meaning either.

Required change: check the length in both public functions before any work, with a documented
`# Panics` line, for example
`assert_eq!(actual_max.len(), vector.len(), "fast_vector_partitions_with_max_part: max_part must have as many components as the vector")`
(and the same for `vector_partitions_with_max_part`). That also turns the wrong 0 of the last row
into a stated refusal. Add a should_panic test for the first two rows. A `debug_assert` in
`subtract_part` is not enough: a release build would still loop.

## Minor

### m1. The `integer_lists` saturation comment claims more than holds

`integer_lists.rs:293-295` says "a length bound of usize::MAX only lengthens an enumeration that
has more than usize::MAX lists anyway". That is false in general:
`IntegerListConstraints::new().sum(usize::MAX).min_part(1 << 63)` has exactly one list,
[usize::MAX], but the loop `for length in min_len..=max_len` runs over 2^64 lengths. The behaviour
is no worse than HEAD's for `sum(usize::MAX - 1)`, which already loops over sum + 1 lengths (the
pre-existing bound that §1.8 reports with the caps of 100). Reword the comment: saturation only
keeps `sum + 1` from wrapping, and the length bound sum + 1 is a separate, reported cost defect.
The sum bound `length.saturating_mul(..)` of `generate_with_length` (`:335`) is in the same position.

### m2. `remainder_bound_upper` doc: two divisions are not directed

`partition_number.rs:295-296` says "Evaluated at 64 bits with every operation rounded up". Two
operations are not: `Float::with_val(128, terms) / Float::with_val(128, n - 1)` (`:312`) and
`Float::with_val(128, 2n) / 3u32` (`:314`) are owned-value divisions, which rug rounds to nearest
at 128 bits; only the following `up(64, ..)` is directed. The bound is still an upper bound: N/(n-1)
and 2n/3 are rationals with denominators below 2^54, so when such a value is not itself a 64-bit
float it lies at relative distance at least 2^-118 from every 64-bit float, which is far more than
the 2^-128 error of the 128-bit rounding; so its 64-bit upward rounding is the same before and
after. Either say that in the doc, or make the two divisions directed directly
(`up(p, &a / &b)` on two 128-bit `Float`s, and `up(p, &t / 3u32)`), which removes the argument.

### m3. Nits

- `tiling.rs:280-281`: "the count then has about 0.69 n bits" is true for 2 x n domino tilings
  only. Say that `Matrix::pow` takes a u32 exponent, and leave out the bit count or name the
  domino case.
- `symmetric_group_representations.rs` (`all_irreducible_representations`,
  `SymmetricGroupCharacterTable::new`): "For every n >= 36" is the 64-bit threshold; on a 32-bit
  target it is smaller. Add "on 64-bit targets".
- `set_system.rs` test `is_shattered_mask_is_a_usize`: the comment "The empty family shatters only
  the empty set" describes the call `is_shattered(&[], &[vec![]])`, whose family is {{}}, not the
  empty family.
- `enumeration.rs:1054-1056`: "`k < ways`, which is right for any true count above usize::MAX"
  misses the position k = usize::MAX itself. It is unreachable (it needs about 2^64 elements
  enumerated first, R2), so only the wording is off.

## What I checked, and found right

- Every item of §1.6 (1-36), §2, §3 and §4 is implemented as the revised design says, with the
  deviations of the report (all justified; deviation 1 is right: gp `3*2^62` =
  13835058055282163712 < 2^64, so the design's lcm(2^62, 3) = None was wrong and lcm(2^62, 5) is
  the overflowing case).
- HRR (`partition_number.rs`): (1.4), (1.5), (1.8) and Algorithm 1 agree with the paper text
  (`design/C1-combinatorics/paper/johansson2012.txt:79-110`, `:196-215`). Every `Enclosure`
  operation is directed (`with_val_round` Down/Up on incomplete values, `cos_ref`, `cosh_ref`,
  `sinh_ref`, `sqrt_ref`), apart from the two divisions of m2. The cos and U rules are right (U is
  increasing on x > 0). The final test L = ceil(lo) = floor(hi) = H proves that [lo, hi] holds
  exactly one integer, and p(n) lies in it because |R| < M <= Mb. The precision estimate is
  decreasing in k for k <= N < C, so `prec_acc` covers every term. The retry cap panics with
  "internal error" and never returns an unproved value. The domain check runs before any work.
- `band_rows`: I traced the front entry, the pops and the pushes. base = lo(i) - 1 always, so
  `t - 1 >= 0`; lo(i) = lo(i-1) only when lo = 1, so the stale front is j = 0, which is 0 for
  Stirling and 1 for Eulerian. The end index k - base is inside the buffer.
- Derangement splitting, Bell triangle, Motzkin (M_4 = 9), Schroder (S_3 = 22), Fibonacci doubling
  and the negative-index signs, Lucas, falling/rising signs and zeros, `binomial_usize_checked`
  (the u128 overflow case and the loop bound), the checked Horner rank, the subset successors, the
  superpartition and `count_partitions_with_max_parts` exactness arguments: re-derived, all right.
- `number_of_standard_tableaux`: l_1 = lambda_1 + r - 1 <= n after conjugation, so the tail
  product is n!/l_1!; both branches are exact divisions. My gp `hookdim` gives every listed value
  (`check_gp.txt`), and f(2001, 1^2000) agrees with `binomial(4000,2000)` (residue
  85248900849712891484894804346834018665, 3994 bits).
- Reference values: I recomputed all quoted gp values of the three test files (`check.gp` ->
  `check_gp.txt`; they all agree), regenerated both fixtures with the shipped generators
  (`counting.csv`, `numbpart.csv`: byte-identical to `tests/data/`), and re-ran the GAP H145E1
  check (`NrConjugacyClasses(SymmetricGroup(n))` and `NrPartitions` for n = 1..12 both give
  [1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77]).
- No facade: every refusal is a documented panic with a stated message, a None with its stated
  meaning, or a valid size-hint bound; no bounded search decides anything.
- Tests: no assertion was weakened. The only deleted tests are the two debruijn helper tests, and
  their nine assertions are in `machine_word.rs::falling_factorial_usize_checked_values`;
  `enumeration.rs::test_factorial` keeps its seven assertions. The impl's HEAD check
  (`C1-work/headcheck/per_test.txt`) shows 42 of 48 runnable `machine_word_counts` tests failing at
  HEAD; the 6 passing ones pin values HEAD already had right, as the design says. Review 4 m1 is
  folded in (`number_of_standard_tableaux_matches_the_hook_formula`, n <= 20, plus the two
  hook-branch values).
- Costs (debug sample, `src/tm.rs`, against the design's unoptimised figures): fibonacci(10^6)
  0.25 s (0.33), bell(1000) 0.31 s (0.54), stirling_first(1000, 500) 0.19 s (0.27), motzkin(10^4)
  0.04 s (0.05), schroder(10^4) 0.06 s (0.1), derangements(10^5) 0.90 s (1.3),
  falling(2^32-1, 10^5) 1.46 s (2.2), catalan(10^5) 0.035 s (0.04), p(10^8) 0.20 s (0.23). The
  release figures of the report are at or below the documented costs.
- Headers cite the chapters and the papers; no new `unsafe`; no em-dash in added prose; no change
  to PROJECT_TRACKER.md, docs/, rustmath-misc/output.txt or a `.tmp_dir_*`.
- Review 4 minors: m1 and m2 folded, m3 folded except the Verdicts wording, which is design text
  (a fair skip).

## Acceptance (debug)

| command | result |
|---|---|
| `cargo build -p rustmath-combinatorics` | ok (`build.log`) |
| `cargo test -p rustmath-combinatorics --no-fail-fast` | unit 1509 passed, 0 failed, 7 ignored; `ch144_counting` 14 passed (5.7 s); `ch145_number_of_partitions` 6 passed (1.3 s); `machine_word_counts` 50 passed (0.4 s); doctests 189 passed, 11 ignored (`test_crate.log`) |
| `cargo clippy -p rustmath-combinatorics --all-targets` | exits 101 on the five older deny lints of design §8.2 (now `binary_words.rs:336`, `debruijn.rs:101`, `path_tableaux.rs:454`, `super_tableaux.rs:297`, `word.rs:252`) and the older `erasing_op` in the `composition.rs:775` test (`clippy.log`). With those three lint kinds allowed it finishes, and the impl's `C1-work/lintmatch.py` finds 0 lint sites on changed or new lines (`clippy_allowed.log`) |
| `cargo test -p rustmath-jupyter --no-fail-fast` | 64 passed, 0 failed (418 s) |
| `cargo test -p rustmath-symmetricfunctions -p rustmath-quantumgroups --no-fail-fast` | 200 passed, 0 failed, 1 ignored |
| `cargo test -p rustmath-groups --no-fail-fast` | 668 passed, 0 failed, 4 ignored |
| `cargo test -p rustmath-algebras -p rustmath-liealgebras -p rustmath-crystals -p rustmath-topology --no-fail-fast` | 1403 passed, 0 failed, 2 ignored |
| `cargo check --workspace --tests` | ok, 0 errors (`check_ws.log`) |
| gp fixtures regenerated with the shipped generators | byte-identical (`gp_regen.txt`) |

No release build was made. The `--release` run of §8.2 was dropped as review 4 m2 asked. The tests
pass, so the only reason for not approving is B1.

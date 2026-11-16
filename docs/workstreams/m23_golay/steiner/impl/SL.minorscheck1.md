# Check of the fixes for the minor items of review 1 (Steiner labelling)

Date: 2026-11-15. Reviewer: independent. Object: the uncommitted worktree
`/home/user/RustMath-steiner` (branch `claude/m23-steiner` on `d91f880`) after the round
"Minor items of review 1" of `impl/SL.md`, against the items m1 to m5 of `impl/SL.review1.md`
(Section 8). Scratch: `impl/SL.minors1/` (paths relative to
`/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/m23/sl/`).

No file of the worktree was edited. As in review 1, I touched the four `lib.rs` before each cargo
run (mtime only; the target directory is shared with the WP-F worktree). `git status --short`,
`git diff --stat` and the sha256 of all 24 changed or new files are the same before and after
this check (`SL.minors1/status_before.txt`, `diffstat_before.txt`, `sha_before.txt`).
`target/release` was deleted after use. Free disk at the end: 7.9 GB.

## 0. Verdict

**Approve. All five items are fixed correctly and completely. No blocker. No regression. All
tests pass, including the `env_` tests in release. No undecided result became decided.**

Four optional nits are listed in Section 7. None needs a change before the commit.

## 1. What changed in this round

By mtime (after the review at 12:30) and by content: `rustmath-combinatorics/src/designs.rs`
(only the hunk from line 961; the diff against `d91f880` is still additive, two hunks),
`rustmath-combinatorics/tests/designs_exact.rs`, `rustmath-finitefields/tests/galois_ring_frobenius.rs`
(one comment), `rustmath-galois/src/integrality.rs`, `rustmath-galois/src/steiner_labelling.rs`,
`rustmath-galois/tests/steiner_labelling.rs`. The `lib.rs` diffs are unchanged in content. The
other files (`galois_ring.rs`, `galois_ring_frobenius.rs`, `cyclic_golay.rs`, the data files)
predate the review. Line numbers quoted in the fixer's notes match the files.

## 2. Item by item

### m1. Panic for `p < 2`: fixed

- `steiner_labelling.rs:452` `require_prime(p)` runs the same `try_is_prime` as
  `FrobeniusRoots::new` (`galois_ring_frobenius.rs:588`) and maps `Ok(false)` and `Err` to
  `SteinerError::Frobenius(FrobeniusRootsError::Prime(..))`. `try_is_prime` returns `Ok(false)`
  for every `n < 2`, negatives included (`rustmath-integers/src/prime.rs:110`), so no panic.
- It is called before `precision_for` in both public entry points that take a caller's prime:
  `duad_resolvent` (line 660) and `steiner_labelling` (line 763, for a given and for a searched
  prime). The other `precision_for` calls (lines 926, 964, 985) use `self.prime()`, a prime
  already proven by `FrobeniusRoots`. `frobenius_sign_given_t5` and
  `SteinerLabelling::frobenius_sign` never call `precision_for`; `FrobeniusRoots::new` refuses
  `q < 2` there before any arithmetic with `q` (read, lines 578 to 596). So every public path
  that returns `Result` now refuses `p < 2` with `Err`.
- `precision_for` keeps its documented panic; that is acceptable (its doc says so, and the
  review did not ask to change it).
- Behaviour change: a composite `p` with an explicit precision `N` and `p^N <= 2 B5` now gives
  `Prime` instead of `PrecisionTooLow`. Both are refusals; no existing assertion depends on the
  old order (the full test suite passes).
- Test: `tests/steiner_labelling.rs:514`, inside `refusals`: `steiner_labelling` with `p = 0, 1`
  and `duad_resolvent` with `p = -23, 0, 1, 25`, all `Err(Frobenius(Prime(_)))`. This is the
  review's required change.

### m2. Memory of `t_design_index`: fixed with a dense counter array

The hash map is replaced by a `Vec<u32>` of `C(v, t)` counters indexed by the colex rank
`sum_i C(c_i, i + 1)`. I checked the code by hand and by a brute-force comparison.

- **Rank.** `table[i * width + d] = C(pts[i + d], i + 1)` with `width = n - t + 1`; position `i`
  of an index vector satisfies `i <= idx[i] <= n - t + i`, so `d = idx[i] - i` is in range. The
  combinatorial number system is a bijection from the `t`-subsets onto `[0, C(v, t))`.
- **`u32` bounds.** Each table entry is the position-`i` term of the rank of some `t`-subset of
  `{0, ..., v-1}` (it has `i` points below and, since `pts[i + d] <= v - t + i`, at least
  `t - 1 - i` points above), so it is below `C(v, t) <= 10^8 < 2^27`. A count is at most the
  number of blocks with at least `t` points, at most `S <= 10^8`. The intermediate products of
  `binomial_u128_saturating` stay below `2^27 v`, far from `u128::MAX`.
- **Early exits.** `S = 0`: every `t`-subset lies in 0 blocks, `Some(0)` (correct, also for
  `t = 0` with no blocks). `0 < S < C(v, t)`: some subset is met, some is not, `None` (correct).
  A saturated `C(v, t)` only makes `S < C(v, t)` true, which is also correct then.
- **Enumeration.** The successor rule (rightmost position below its maximum `n - t + i - 1`,
  then reset to the right) is the standard lexicographic one; `t = 0` gives one empty subset.
- **Brute force** (`SL.minors1/probe/design_bf.rs`, linked against this worktree's rlib, debug
  and release): for every `t`-subset of `{0, ..., v-1}`, count the blocks containing it.
  20,000 random instances with `v <= 10`, `0 <= t <= v`, random blocks (unsorted, repeated,
  of mixed sizes), complete designs with one block removed or repeated, multiplied full blocks.
  Every `t_design_index` and `is_steiner_system` value agrees (about 48 % `Some(> 0)`, 28 %
  `Some(0)`, 22 % `None`; 65 % on the dense path). Output `probe/design_bf_out.txt`.
- **Near the limit, release** (`probe/design_big_out.txt`): one block of 29 points, `t = 14`
  (77,558,760 counters): `Some(1)` in 8.3 s; 28 points, `t = 14`: 3.9 s; two copies of 27
  points, `t = 13`: `Some(2)`, 3.1 s. So mid-range `t` near the limit is exact and bounded.
- **Docs.** `DESIGN_CHECK_LIMIT` (doc at lines 1010 to 1011) states the counter bound (400 MB). The doc of
  `t_design_index` (lines 1030 to 1056) states the cost, the memory (counters plus one sorted
  block and its table) and the measurements, as the review required. The fixer's measured
  numbers (120 points, `t = 4`: 0.16 s and 35 MB, against 6.1 s and 1.13 GB before) are
  consistent with `C(120, 4) * 4` bytes = 33 MB.
- **Test.** `designs_exact.rs:104` `colex_ranks_cover_every_subset_once`: a full block given
  unsorted has index 1 and twice it index 2 for eight `(v, t)` pairs, including `t = 0`,
  `t = v - 1`, `t = v`; the complete design of the 4-subsets of 9 points has
  `lambda = 21, 6, 1, 0` for `t = 2, 3, 4, 5` (`C(9 - t, 4 - t)`, correct), and without one block
  `None` through the dense path (`125 * 6 = 750 >= 36`, so not the early exit). The four
  existing tests of the file are unchanged (same test names; their assertions read).

### m3. Costs of the public entry points: documented

One cost line each, as required: `duad_resolvent` (line 650), `search_limit` (676), `extra_digits`
(681), `precision` (685), `retries` (688), `golay_invariants` (914), `heptad_resolvent` (953),
`SteinerLabelling::duad_resolvent` (977). Checked:

- `N` formulas: `253 log_p(2M)` gives 266.5 for f2 at 29 (actual 267); `253 log_p(7M)` gives
  359 (actual 360); `7 log_p M` plus `log_p 506` gives 7.7 (actual 8). CC0 "about 12,400" and
  "about 12,500" agree with the design's 12,416 and 12,510 (`design/SL.md:800`), marked "not
  run".
- `search_limit`: 1228 primes in `[3, 10000)` (`pi(10^4) = 1229` minus the prime 2), correct;
  the range `min(search_limit, MAX_P + 1)` is what `first_inert_prime` does.
- `extra_digits`: `10 / log10 29 = 6.8`, "7 at 29" is right; f1 `N = 31`, f2 `N = 30`.
- Times against my release run: heptad f2 1.89 s, f1 2.21 s (doc 1.9 s, 2.1 s); duad f2
  1.30 s, f1 1.69 s (doc 1.3 s, "1.4 to 1.6 s"). The f1 duad time is slightly above the stated
  range on this shared machine; review 1 measured 1.61 s. Noise, not a defect.

### m4. Empty or unequal-length candidates: fixed

- `integrality.rs:54, 60`: `EmptyCandidate { index }`, `CandidateLengths { index, len,
  expected }`, with `Display`. `integrality_dichotomy` (line 170) checks the precision first,
  then refuses any empty candidate and any candidate whose length differs from the first one.
  The first candidate empty gives `EmptyCandidate { index: 0 }` (the empty check comes before
  the length check). An empty list gives `Unresolved { NoneWithinBound }`, documented.
- The decision logic after the checks is unchanged, so the change only turns former outputs into
  `Err`; it cannot create a `Decided`.
- `SteinerError::Integrality` (steiner_labelling.rs:218) now says `PrecisionTooLow` is the only
  `IntegralityError` the module can cause. True: its only calls are `integrality_dichotomy`
  with two candidates of length 1 (line 787) and `bounded_integer` (lines 606, 930), which can
  only fail with `PrecisionTooLow`.
- Test `integrality.rs:314` `malformed_candidates_are_refused`: the review's probe input, an
  empty last candidate, unequal lengths, the empty list. Covers the required change.

### m5. Test comment: fixed

`galois_ring_frobenius.rs:319` reads "(Z - w)(Z + w) = Z^2 - w^2: the constant term -w^2 is not
in Z/125.", the review's wording. Correct for `GR(5^3, 3)` (`w^2` has degree 2 < 3).

## 3. Tests run

Debug, `cargo test -j 2 -p <crate>` after the touch (`SL.minors1/run_debug.sh`, logs
`debug_*.log`):

| Crate | Result | Wall (with rebuild) |
|---|---|---|
| rustmath-combinatorics | lib 1509 (7 ignored), `ch144_counting` 14, `ch145_number_of_partitions` 6, `designs_exact` 6, `machine_word_counts` 53, doc 189 (11 ignored): all pass | 50 s |
| rustmath-coding | lib 67, `cyclic_golay` 7, doc 5: all pass | 16 s |
| rustmath-finitefields | lib 205, 12 integration files (`galois_ring_frobenius` 11), doc 8: all pass | 24 s |
| rustmath-galois | lib 66 (2 ignored), `descent` 12, `steiner_labelling` 13 (10 ignored), `perm` 5, `subgroups_and_labels` 5, doc 1: all pass | 54 s |

The list of test names per crate equals review 1's list plus exactly the two new tests
(`integrality::tests::malformed_candidates_are_refused`, `colex_ranks_cover_every_subset_once`).
Debug fast set of `steiner_labelling`: 15.9 s of test time.

Release (`SL.minors1/run_release.sh`, logs `rel_*.log`), because the fixes touch the `env_`
paths (`require_prime` in `steiner_labelling` and `duad_resolvent`, the new checks in
`integrality_dichotomy`, the rewritten `is_steiner_system` behind `golay_systems_checked`):

| Command | Result | Time |
|---|---|---|
| build `--release -p rustmath-galois --test steiner_labelling --no-run` | built | 118 s |
| `M23ISGALOIS=/home/user/external/m23isgalois <bin> --ignored env_ --test-threads 1` | 8 passed: f2 `T5 = -1059840`, `J5 = 61916`, `J7 = -83030`, sign -, rejected 44 digits; f1 `T5 = 92747040`, `J5 = 524492`, `J7 = -2281968`, sign -, rejected 45 digits; resolvents at N = 360, 395, 267, 302; second primes 73 and 223 | 29.25 s |
| same without `M23ISGALOIS` | 8 FAILED, each "set M23ISGALOIS to a checkout of shaowuz/m23isgalois" | 0.06 s |
| `<bin> --ignored full_` | 2 passed | 1.68 s |
| `<bin>` (fast set) | 13 passed | 1.79 s |

Other checks:

- `cargo clippy -j 2 --no-deps -p rustmath-galois -p rustmath-finitefields --all-targets` and
  `cargo clippy -j 2 --no-deps -p rustmath-combinatorics --lib --test designs_exact -- --cap-lints
  warn`: no warning in `integrality.rs`, `steiner_labelling.rs` (src, tests),
  `galois_ring_frobenius.rs` (src, tests), `designs_exact.rs`, or `designs.rs` lines 958 to
  1170 (logs `clippy_galois.log`, `clippy_comb.log`).
- `rustfmt --edition 2021 --check` on `integrality.rs`, `steiner_labelling.rs` (src, tests),
  `galois_ring_frobenius.rs` (tests), `designs_exact.rs`: clean. On a copy of `designs.rs` the
  only diffs are at pre-existing lines (37, 57, 519, 547, 773, and 1179 in the old test module).
- No em-dash and no `unsafe` in the changed files or added lines.

## 4. No undecided result became decided

Every change of this round only adds refusals (`require_prime`, the two new
`IntegralityError` variants) or replaces the implementation of `t_design_index` with one that
agrees with brute force. The decision code of `integrality_dichotomy` and of
`steiner_labelling` is unchanged after the new checks. The `Unresolved` and `NotInM23` tests
(C23 at 2 and 3 with retries, f2 at `N = 9`, `x^23 - x - 1` at 23 and 157) pass with the same
residues. The run-time S(4,7,23) check still gives `Ok(true)` for both signs, and the
`cyclic_golay` test still checks `lambda_3 = 5`, `lambda_2 = 21`, `lambda_1 = 77` through the new
code.

## 5. Licence

No data file changed in this round. The new test values (`lambda = C(9 - t, 4 - t)`) are cited
(Beth, Jungnickel, Lenz). No f1 or f2 coefficient was added.

## 6. Disk and processes

Built only the crates needed. `target/release` (this worktree's, through the shared `target`
symlink) was deleted after the release run; the F-9 job uses its own `target-f9rel`. The probe
binaries were deleted; the probe source and outputs remain in `SL.minors1/probe/`.

## 7. Optional nits (no change required)

- **n1 (comment).** `designs.rs:1095`, dense-path comment before `let mut counts`: "every count (at
  most the number of blocks with t points)" should read "with at least t points".
- **n2 (cost, not a regression).** `DESIGN_CHECK_LIMIT` bounds `S`, not the work `S t`. For
  `t` close to `|B|` the time is quadratic in the block size: one block of 20,000 points with
  `t = 19,999`: 0.40 s; 40,000 points: 1.48 s (release); 10^6 points would take about 15 minutes
  by extrapolation. The doc says "`S` steps of `O(t)`", which is honest; it could add that the
  limit does not bound `S t`, or enumerate complements when `2t > |B|`. The old version had the
  same time and more memory.
- **n3 (pre-existing, made visible by the m3 doc).** `steiner_labelling` reports
  `NoInertPrime { below: opts.search_limit }`, but the search stops at `fp_factor::MAX_P`
  (3,037,000,499), as the new `search_limit` doc says. For `search_limit > MAX_P + 1` the error
  claims a range larger than the one searched. Unreachable in practice (about 1.4 * 10^8
  factorizations first). Report `min(search_limit, MAX_P + 1)` in a later round.
- **n4 (doc).** `require_prime`'s doc says the second primality test in `FrobeniusRoots::new` is
  "free" above `2^64`. It still runs trial division and BPSW before the cache lookup; cheap, not
  free. "cheap" would be exact.

## 8. Files of this check

`impl/SL.minors1/`: `run_debug.sh`, `run_release.sh`, `debug_*.log`, `debug_summary.txt`,
`rel_build.log`, `rel_env.log`, `rel_noenv.log`, `rel_full.log`, `rel_fast.log`,
`rel_summary.txt`, `clippy_galois.log`, `clippy_comb.log`, `old_*.txt` and `new_*.txt` (test name
lists), `status_before.txt`, `diffstat_before.txt`, `sha_before.txt`; `probe/design_bf.rs`,
`probe/design_bf_out.txt`, `probe/design_big_out.txt`.

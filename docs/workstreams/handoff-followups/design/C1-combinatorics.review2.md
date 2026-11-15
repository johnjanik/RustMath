# Review 2 of design C1-combinatorics (rustmath-combinatorics), including Revision 1

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/C1-combinatorics/review2/`
(`rc.sh` builds against the HEAD release rlib; `sweep_copy.rs`, `sp.rs`, `sp2.rs`, `bw.rs`, `comb.rs`:
HEAD reproductions; `mine.gp` / `mine.txt`: 36 fresh gp values of p(n); `vals.gp`: gp reference values;
`head_times.txt`, `hrr_times.txt`: timings). Tree at HEAD c79a582, clean.

Verdict: **not approved**. One blocker: the revision-1 sweep, which the design presents as complete
and which step 12 of the plan writes into HANDOFF as complete, misses at least six public functions of
the same machine-word class. Two of them are direct twins of items the design fixes, through private
helpers of the kind it deletes. Everything else holds up; the minors below can be folded in with the
blocker fix.

## What I reproduced (agrees with the design)

- §1.1 and the §1.6 sweep items: I rebuilt `rev1/sweep.rs` against the HEAD release rlib and got every
  HEAD output in the design. Examples: dimension [11,10] -> 16335, [5^5] -> 317, [8..1] -> 2; box count -> 0;
  burnside 1; count_fixed_colorings 4; count_tilings(2, 2^32+1) 1; rank_k_subset 9223372036854775807;
  rank_binary 0; the DeBruijn and OrthogonalArray Some; universal_cycle_permutations(70,70) Some([0]);
  sunflower_bound(30,3) u64::MAX; plane partitions 15/874/393; Polya {1:1, 2:2}; hall x.
- `rev1/bin_fix check` passes in 0.66 s. It checks binomial_usize_checked against exact binomials, the
  Frobenius formula against n!/prod(hooks) for all 7338 partitions of n <= 24, the two-branch version for
  n <= 22, the deque rows against the HEAD tables, the HEAD partition DP against Euler for n <= 416, and
  the Schroder multipliers. `bin_fix hrrdom` gives exp_max 1073741823; cosh(C) is finite at 8e16 and
  +inf at 8.5e16.
- gp (`vals.gp`), all equal to the design: hook dimensions 58786, 20, 701149020, 40971642983700000000,
  44261486084874072183645699204710400, 475073684264389879228560, 7821859115070000000,
  29258366996258488320, 5. The largest f^lambda for n = 35 is 7821859115070000000 at (9,7,6,4,3,2,2,1,1)
  and for n = 36 is 40971642983700000000 at (9,7,6,4,3,3,2,1,1), both checked by `forpart`. Also 3^41,
  18^16, 17^15, Catalan 37 and 36, 30!·2^30, 20!, C(5e9,2), C(10^7,3)-1, C(2^32+2,2), C(70,35),
  C(10^5,2), numbpart(416/417/1000), 46262, the multinomial residue and bits, the two generalised
  Fibonacci residues (the negative one as M - 66370102269696054044432080915371431578), -1228, 3/8, 15/8,
  -120, (2^32-1)(2^32-2)(2^32-3), and 1994 bits.
- HRR: (1.4), (1.5), (1.8), (2.1) and Algorithm 1 match the paper text (`paper/johansson2012.txt:70-110,
  155-215`). I re-derived these steps:
  - t_k = 4 U(C/k) S_k / v, with S_1 = sqrt 3 and S_2 = (-1)^n sqrt(3/2);
  - the m/r recurrences (m_(l+1) - m_l = 3l + 2 = r_l, and one subtraction suffices for k >= 3);
  - every outward rounding: cos through |cos'| <= 1, and U increasing on x > 0;
  - Rademacher's bound rounded up (the sqrt 3 and sqrt N in the denominators are rounded down);
  - M(n, N) decreases in N (d/dN of sqrt N sinh(c/N) has the sign of sinh y - 2y cosh y < 0), so the
    linear scan for N ends;
  - the L == H test proves the value;
  - termination: the width shrinks by 2^-guard; the amplification by x = C/k is at most 2^28 at 10^16,
    inside the 48-bit margin.
  - No bounded search decides anything, so Decided/Unresolved does not apply.
- 36 fresh gp values (`setrand(777123)`, n in {1001, 1002, 1003, 1500, 1729, 2048, 4095, 9999, 100001,
  123456, 7777777} and 25 random n < 5·10^7) all agree with `bin/hrr gp` (1.1 s).
- rug: `write_digits` and `significant_digits` are not behind rug's `std` feature, and `to_digits` is
  (rug 1.30.0 `integer/big.rs:654,685,732`). So the digit route of §2.4 step 6 builds under the declared
  features. Only `rustmath-rings/src/real_mpfi.rs` uses directed rounding in the workspace, so the
  private `Enclosure` is justified, as review 1 found.
- Costs (release, this machine; claims in brackets):
  - new: fib(10^7) 0.44 s [0.50]; fib(1.3·10^6) 0.022 s [0.027]; bell(3000) 1.19 s [1.40];
    motzkin(10^5) 1.06 s [1.10]; schroder(10^5) 1.60 s [1.83]; derangements(10^6) 2.77 s [3.4];
    falling(2^32-1, 10^6) 3.06 s [3.3]; catalan(10^6) 0.096 s [0.10]; stirling1(3000, 1500) 0.85 s [0.91];
    eulerian(3000, 1500) 1.59 s [1.73]; stirling2(10^4, 100) 0.90 s [0.93];
    multinomial(10^6, [2]·5·10^5) 1.49 s [1.6]; p(10^9) 0.95 s [0.94]; p(10^10) 4.8 s [5.4].
  - HEAD at the REPL limits: bell(600) 10.0 s [11.3]; fib(1.3·10^6) 10.9 s [11.8]; schroder(50000)
    0.61 s, 390 MB [0.62 s, 390 MB]; falling(2^32-1, 10^5) 1.80 s [2.18].
- Algorithms re-derived: the Fibonacci doubling identities and negative-index signs; the in-place Aitken
  array (traced to row 3); the band-deque rows. For the deque, the front entry after a pop holds
  row (i-1)'s value at lo(i-1), a band entry of that row, so the memory bound (w+1)(B/8+32) holds. The
  Motzkin and Schroder recurrences are checked against M_2, M_3, S_2, S_3. Also checked: the
  derangement binary-splitting identities; the multinomial telescoping product; the falling and rising
  sign and zero cases; Frobenius's formula with l_1 <= n; Horner for Tuple::rank (r_j n <= r_(j+1) <= rank).
  The bound C_n > 2^n for n >= 5 holds, and the dp of count_partitions_with_max_parts is monotone in i
  and j, so its checked_add fails only when the result does not fit.
- I3 interplay: I3 keeps `SageInteger::binomial(&Integer, &Integer) -> Integer` and adds a panic above
  2^32 result bits. For u32 arguments its lower bound is K·1 <= 2^31, so `binomial`, `catalan` and the
  u32 wrappers are never refused.
- J1b interplay: J1b's revision consumes `number_of_partitions`, `NUMBER_OF_PARTITIONS_MAX_N`,
  `falling_factorial_integer` and `rising_factorial_integer` under those names, and uses C1 §7's costs.
  §7 covers all 18 imported callees. `FACTORIAL_N` is not C1's: the REPL `factorial` arm calls
  `Integer::factorial`, `repl.rs:854`.

## Blocker

### B1. The revision-1 sweep is incomplete, and §1.7 and the HANDOFF line say it is complete

Revision 1 says "I swept the whole crate" and "§1.7 now classifies every remaining site". The HANDOFF line
of §8.1 step 12 says the machine-word code left in the crate is bounded by memory, by an enumeration or
by 2^32 steps, or sits in flagged modules. That is false for these public functions. None of them is in
§1.6 or §1.7. Each one is reached with O(1)-size input and returns a wrong value at once or within
0.2 s. HEAD release outputs come from `review2/comb.rs`, `sp.rs` and `bw.rs`; a debug build panics with
"attempt to ... with overflow".

| function | defect | HEAD output | reference (gp) |
|---|---|---|---|
| `combinations.rs:62-81 Combination::rank` (re-exported `lib.rs:112`), private `binomial_usize` `:159-173` | `result * (n - i) / (i + 1)` in usize: the twin of §1.6 item 17 (`Subset::rank_k_subset`). It wraps even when the binomial fits | rank of [1,2,3] among 3-subsets of 5e9+1: 3276627960645224192 | `binomial(5*10^9,2)` = 12499999997500000000 (fits) |
| `combinations.rs:84-110 Combination::unrank` | same helper | `unrank(12499999997500000000, 5e9+1, 3)` = [3, 566066340, 1487858385] | [1, 2, 3] |
| `enumeration.rs:487-494 PermutationGrayCodeIterator::size_hint`, private usize `factorial` `:499-504` | `(2..=n).product()` in usize: the twin of §1.6 item 6. The upper bound wraps to a value that is too small | `new(21).size_hint()` = (0, Some(14197454024290336768)) | `21!` = 51090942171709440000 > 2^64, so the bound must be None |
| `superpartitions.rs:444-451 count_superpartitions` | `1usize << p.length()` and a usize `.sum()` | (63) = 13491456169319932838 in 0.19 s; (64) = 8536168323873208747 | The function's own formula is the coefficient of `prod(i=1,n,1/(1-2*x^i))`: (62) = 15969100102326684686 (equal to HEAD; it fits), (63) = 31938200243029484454 > 2^64. I checked it against `superpartitions(n).len()` for n <= 8 |
| `binary_words.rs:39-44 BinaryWord::to_usize` | `1 << i` for i >= 64: the twin of item 18 (`rank_binary`) | a 65-bit word with only bit 64 set -> 1 | 2^64 (does not fit) |
| `subset.rs:302-308 Subset::next_binary` | `(1 << self.n) - 1` at n = 64 = usize::BITS (`to_bitvector` accepts n <= 64) | the empty subset of a 64-set -> None (claims "this is the full set") | Some({0}); n = 63 is right at HEAD |

Required change:

- Add these to §1.6 with fixes, tests and gp references, or justify each one in §1.7.
  - `Combination::rank` / `unrank`: delete `combinations.rs::binomial_usize` and use
    `binomial_usize_checked`. Copy the rank / unrank handling of item 17: `checked_add` with a documented
    panic in `rank`, and None counts as "larger than any rank" in `unrank`.
  - `PermutationGrayCodeIterator::size_hint`: `(0, falling_factorial_usize_checked(n, n))`, so the upper
    bound is None when n! does not fit. Delete the private `factorial`.
  - `count_superpartitions`: checked shift and checked sum, with a documented panic ("exceeds
    usize::MAX"). n >= 63 overflows, and n = 63 enumerates only p(63) = 1505499 partitions, so the
    panic is reached at once. Pin it with (62) = 15969100102326684686 and a should_panic on (63).
  - `BinaryWord::to_usize`: a documented panic when a set bit has index >= usize::BITS. Any number of
    leading zeros stays allowed.
  - `Subset::next_binary`: handle n == usize::BITS, for example with `usize::MAX >> (usize::BITS - n)`
    for the full mask, or `checked_shl`. Test that the empty subset of a 64-set gives Some({0}) and that
    the full set gives None (64-bit only).
- Widen the sweep's patterns and say which ones were used: plain `*`, `+` and `<<` in usize or u64
  helpers (`fn .*binomial.*usize`, `fn factorial(n: usize)`, `1 <<`, `1usize <<`), and `.product()` /
  `.sum()` over machine words, not only `as u32`, `saturating_*`, `wrapping_*` and `unwrap_or(0)`.
- If the designer stops short of a complete sweep, reword the HANDOFF line of §8.1 step 12. It must not
  claim completeness. It should list what is known to remain, with these reproductions.

Every value above was reproduced at HEAD and checked with gp in `review2/vals.gp` and the one-liners
recorded there. The fixes are small and reuse `machine_word.rs`.

## Minor

### m1. `binomial_usize_checked`: the text omits the j > m case
§1.6 says "j <- min(j, m - j)". For j > m, m - j underflows u128: a panic in debug, a wrap in release.
The prototype (`rev1/fix.rs:19-20`) returns `Some(0)` first. Put that line in the design. No current
caller passes j > m (stars_and_bars handles k = 0, CombinationGrayCodeIterator returns early for k > n,
and the rank loops guard `choices >= remaining - 1`). The helper is shared, so it must be total anyway.

### m2. `TupleIterator` is an `ExactSizeIterator`
`tuple.rs:335` has `impl ExactSizeIterator for TupleIterator {}`. With the new `size_hint` of
(usize::MAX, None), `len()` fails std's `assert_eq!(upper, Some(lower))` with a generic message. The
crate already does this for `CartesianProduct` (`enumeration.rs:917-929`), so it is acceptable. Say so in
the doc: `len()` panics when n^k > usize::MAX. The same holds for `PermutationGrayCodeIterator` if it is
an ExactSizeIterator (it is not at HEAD).

### m3. Behaviour changes not listed
- `all_irreducible_representations(n)` (`symmetric_group_representations.rs:414`) and the character-table
  constructor that calls it (`:445`) now panic for n >= 36, through `IrreducibleRepresentation::new`.
  At HEAD they silently stored wrong dimensions from n = 21. List them in §5 next to `new`.
- After I3, `count_k_subsets`, `count_lattice_paths_ne`, `catalan` and `multinomial` reach I3's panic
  "binomial: C(n, k) has more than 2^32 bits" for absurd usize or u64 arguments. Put it in their
  `# Panics` docs, or one line in §5. HEAD truncated those arguments silently.

### m4. The REPL half of S10 now has an owner
§4.2, §7 and §8.1 step 12 say "No unit owns this" and "reported to the orchestrator". J1b's revision took
both switches (`J1b-repl-values-and-limits.md:18`, its §5, plan step 7), and its HANDOFF step removes C1's
open line (`:1368`). Say that J1b §5 owns them. C1 can still record the line as open, because J1b lands
later.

### m5. A should_panic test exhausts memory on HEAD
`universal_cycles_refuse_lengths_above_usize` on `universal_cycle_subsets(70, 35)` runs out of memory
at HEAD (design §6.5). When the implementer confirms that the new tests fail on HEAD, run that one under
a memory limit (`ulimit -v`) or skip it, so it does not take down the shared machine.

## Checks that the design passes

- Coverage: S2 (the five named sites), S3 (certified HRR with a stated domain), S4 (every named callee,
  measured before and after in release), S10 (Integer and ring-element versions; Sage semantics and the
  ch. 49 Pochhammer symbol) and the §7 list for J1b are complete. Only the sibling sweep (B1) falls short.
  Nothing is declared out of scope.
- No facade: `number_of_partitions` returns only a value the enclosure proves. The usize functions
  panic with documented messages instead of wrapping or returning 0. `size_hint` returns None instead of
  a wrong bound. The retry cap is a bug guard, not a mathematical state.
- Tests use gp and GAP values (sampled and re-run above). The new tests fail on HEAD: they do not
  compile, panic in debug with another message, or get a wrong value in release.
- Plan order is implementable: Cargo.toml (rug is already in the graph), machine_word, factorials,
  partition_number, lib.rs, the sibling files, tests, trackers. The acceptance commands cover the crate,
  the REPL, the Partition users and `cargo check --workspace --tests`. The tracker rows the plan edits
  exist: `status_overrides.csv:969`, `:973`; ch145 rows in `intrinsics.csv`; `PROJECT_TRACKER.md:1305`,
  `:1312`.

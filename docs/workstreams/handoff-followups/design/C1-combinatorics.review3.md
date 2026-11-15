# Review 3 of design C1-combinatorics (rustmath-combinatorics), including Revisions 1 and 2

Reviewer scratch: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/C1-combinatorics/review3/`
(`rc.sh` builds against the HEAD release rlib; `r3.rs`, `nb.rs`: HEAD reproductions, output in `nb.txt`;
`ht.rs` / `head_times.txt`: HEAD library timings; `new_times.txt`: prototype timings; `mine.gp` /
`mine.txt`: 19 fresh gp values of p(n); `hook.gp`, `qm.gp`: gp reference values; `dependents.txt`: the
30 files outside the crate that name `rustmath_combinatorics`). Tree at HEAD c79a582, clean.

Verdict: **not approved**. One blocker: the fix of §1.6 item 28 (`Subset::next_binary`) is incomplete.
After it, `next_binary` is still wrong for every ground set of more than 64 elements, and its twin
`prev_binary` has the same defect and appears nowhere in §1.6 or §1.7. The fix is about ten lines. The
rest of the design holds up under reproduction. The minors below can be folded in with it.

## What I reproduced (agrees with the design)

- §1.1 at HEAD (release rlib): `rising_factorial(u32::MAX, 2)` = 0; `partition_count(416)` =
  17873792969689876004 and `partition_count(417)` = 541220193622112941 (gp `numbpart(417)` =
  18987964267331664557).
- Revision-2 items at HEAD: `Partition::dimension` of [11,10] = 16335 (gp 58786) and of
  [9,7,6,4,3,2,2,1,1] = 0 (gp 7821859115070000000, which fits; item 14). Dyck `area()` of N^65536 E^65536
  = 2147450880 and of N^65537 E^65537 = -2147450880 (item 36). `fast_vector_partitions_with_max_part`
  on [1001] with parts <= [1] gives 0, on [1001, 1] with [1, 1] gives 1, and on [417] gives
  541220193622112941; [416] takes 0.09 s (item 34). `Permutation::order` of the 16 prime cycles up to 53
  gives 14142414403480493114, and of the primes up to 47 plus 94 gives 26163907895633379 (item 29).
  `BlockDesign::new(2^33, 3, 2, 2^32, [])` is Some with r = 9223372034707292160 (item 35).
- gp (`hook.gp`, `qm.gp`), all equal to the design: hook dimensions 40971642983700000000,
  475073684264389879228560, 44261486084874072183645699204710400, 58786, 7821859115070000000, 5;
  `lcm(concat(primes(15),[94]))` = 614889782588491410; `lcm(primes(16))` = 32589158477190044730;
  superpartition coefficients 31938200243029484454 (63) and 15969100102326684686 (62); 2147516416;
  18446744071562067968; `binomial(5*10^9,2)` = 12499999997500000000; `numbpart(1000)`; the q-multinomial
  [1, 2, 4, 5, 6, 5, 4, 2, 1].
- HRR. I re-read (1.4)-(1.8) against `paper/johansson2012.txt` and re-derived t_k = 4 U(C/k) S_k / v,
  S_1 = sqrt 3, S_2 = (-1)^n sqrt(6)/2, the m/r recurrences (m_(l+1) - m_l = 3l + 2, one subtraction
  suffices for k >= 3), the outward rounding of `cos`, `u`, `mul` and `m_upper` in `hrr.rs`, and the
  L == H argument. 19 fresh gp values (`setrand(424242)`, including 1001, 1002, 1729, 65537, 2^24+1,
  123456789, 10^9+7, 2^31-1 and 8 random n < 10^8) all agree with `bin/hrr` (residue mod 2^127 - 1
  and bit length). p(10^9 + 7) took 0.85 s and p(2^31 - 1) 1.50 s, matching §2.7. No bounded search
  decides anything, so Decided/Unresolved does not apply.
- Prototype checks pass: `rev1/bin_fix check` and `rev2/bin_fix2`.
- Algorithms re-derived: the derangement binary splitting (leaf loop and the three identities), the
  Motzkin and Schroder recurrences (M_2 = 2, M_3 = 4, S_2 = 6, S_3 = 22), the in-place Aitken array to
  row 3, Fibonacci doubling with the negative-index signs, the falling/rising sign and zero cases, the
  multinomial telescoping, the superpartition DP (item 26: one factor 1/(1 - 2x^i) per pass; its
  overflow check is exact), the uniform-sign argument for `signed_compositions` (item 30), the
  saturation argument of item 33 (I traced the test case through the new bounds: 2^62 at each step),
  and the exactness argument for the vector-partition checked sums (item 34).
- Costs. HEAD library at the REPL limits (this run): motzkin(4500) 14.4 s, derangements(1400) 16.8 s,
  eulerian(1000, 999) 0.49 s, falling(2^32-1, 10^5) 2.69 s, multinomial(10^4, [2]·5000) 0.08 s.
  Prototype: motzkin 4500 / 10^5: 0.004 / 1.37 s; derangements 1400 / 10^6: < 1 ms / 3.88 s;
  eulerian (1000, 999) / (3000, 1500): < 1 ms / 2.26 s; schroder(10^5) 2.52 s; catalan(10^6) 0.18 s;
  falling(2^32-1, 10^5) 0.18 s; fib(10^7) 0.63 s; bell(3000) 1.88 s; multinomial(10^6, [2]·5·10^5)
  2.39 s. The machine was loaded by other builds: both HEAD and new figures are 20 to 50 percent above
  the design's, so the speed-ups hold. J1b's protocol re-measures, so this does not change §7.
- Callers. My grep of the whole workspace (30 dependent files, `dependents.txt`; rustmath-py does not use
  the crate): only `rustmath-jupyter/src/repl.rs:18-23` imports counting functions (the 18 callees,
  all in §7). `Partition::dimension` is called outside the crate only by the test
  `rustmath-symmetricfunctions/src/classical_bases.rs:806`. The only combinatorics `Permutation::order`
  call outside the crate is the test `rustmath-groups/src/permutation_group.rs:441`. No crate re-exports
  the combinatorics `Permutation`. The `.order()` calls in liealgebras and algebras are other types.
- §7 covers every `repl_limits` constant whose arm calls a C1 callee: CATALAN_N, FIBONACCI_N, BELL_N,
  TABLE_BYTES (with `table_bytes` / `check_table`), MOTZKIN_N, SCHRODER_N, DERANGEMENTS_N,
  FACTORIAL_WORK (with `product_work`), PARTITIONS_N, BINOMIAL_WORK for narayana and delannoy (with
  `binomial_work_u32`), DELANNOY_SUM_CAP (with `delannoy_work`). FACTORIAL_N is not C1's: the
  `factorial` arm calls `Integer::factorial` (`repl.rs:855`). J1b consumes the C1 names and signatures
  as written (`J1b-repl-values-and-limits.md:115-117`, `:1115-1140`).
- Digest citations are right: ch. 144 `:34-53`, ch. 145 `:55` and `:299`, ch. 18 `:261-275`
  (`NumberOfPartitions`, "non-negative integer n", so p(0) = 1 is right), ch. 49 `:292` (the
  Pochhammer symbol is the rising factorial), ch. 58 `:204`. The dependency-cycle reason for a private
  `Enclosure` holds (rustmath-rings -> numberfields -> groups -> combinatorics).

## Blocker

### B1. Item 28 fixes `Subset::next_binary` only at n = 64; it stays wrong for n > 64, and `prev_binary` is missed

`subset.rs:302-308 next_binary` starts with `let bits = self.to_bitvector()?;`, and `to_bitvector`
(`:115-125`) returns None for every subset when n > usize::BITS. So `next_binary` returns None, which its
doc defines as "this is the full set", for every subset of a ground set of 65 or more elements. The twin
`prev_binary` (`:313-319`, same `?`) returns None, documented as "this is the empty set", for all of them.
HEAD release (`review3/nb.rs`, output `nb.txt`):

| call | HEAD | truth |
|---|---|---|
| `Subset::from_elements(vec![], 65).unwrap().next_binary()` | None ("full set") | Some({0}) |
| `Subset::from_elements(vec![0], 65).unwrap().prev_binary()` | None ("empty set") | Some({}) |
| `Subset::from_elements(vec![0], 64).unwrap().prev_binary()` | Some({}) | right |

Item 28 changes only the full mask at n = 64. After it, the first row is still None. This is the
class rule of §1.6 itself: a public function, a small input (an empty subset), a wrong value at once. It
is also the twin of item 18 (`rank_binary`, `to_bitvector().unwrap_or(0)`), which the design fixes. The
`?` on a machine-word conversion is not one of the patterns P1-P10. I grepped `rev2/nontest.txt` for
`(checked_*(..)|to_usize()|to_u64()|try_from(..)|try_into()|to_bitvector())?`: the only hits are
these two lines and `tuple.rs:87` (item 12, fixed). So this change closes that pattern.

Required change (item 28 and §6.5):

- Implement both on the sorted element list, with no machine word, so they are right for every n:
  - `next_binary`: let i be the smallest index in 0..n that is not an element. If i = n (the full set),
    return None. Otherwise return (S minus {0, ..., i-1}) plus {i}. This is binary increment: the
    trailing ones become zeros and the first zero becomes one.
  - `prev_binary`: let i be the smallest element. If S is empty, return None. Otherwise return
    (S minus {i}) plus {0, ..., i-1}. This is binary decrement.
  - Both are O(n). The n = 64 full-mask expression of item 28 is then not needed.
- Tests (64-bit only where noted; each fails on HEAD):
  - next of the empty subset of 65 is {0};
  - prev of {0} in 65 is the empty set;
  - next of {0, ..., 63} in 65 is {64};
  - prev of {64} in 65 is {0, ..., 63};
  - the full subset of 65 has no next, and the empty subset of 65 has no prev;
  - for every subset of every n <= 10, next and prev equal `from_bitvector(bits + 1, n)` and
    `from_bitvector(bits - 1, n)`;
  - the existing n = 64 rows of `subset_binary_order_at_the_word_size` stay.
- Add `prev_binary` to §1.6, §5 and the §8.1 step 10 file list.

## Minor

### m1. `hook_length_dimension` already computes f^lambda exactly; the new `number_of_standard_tableaux` duplicates it

`young_representations.rs:52 pub fn hook_length_dimension(partition: &Partition) -> Integer` computes
n! / prod(hooks) in `Integer`. It is re-exported at the crate root (`lib.rs:361-362`) and tested
(`:286-296`). The design does not mention it. It adds a second public f^lambda (§1.6 item 14) and marks
`ch145.NumberOfStandardTableaux` as new. Reconcile the two:

- `hook_length_dimension` becomes a call of `number_of_standard_tableaux` (same values, faster). Its doc
  names the MAGMA intrinsic.
- Add a test that the two are equal for every partition of n <= 12.
- The status row and the PROJECT_TRACKER `:1312` note mention both names.

### m2. The digit conversion needs `num-bigint` as a direct dependency

§2.4 step 6 calls `BigInt::from_bytes_le(sign, &bytes)` and `Integer::new(BigInt)`. rustmath-combinatorics
has no `num-bigint` dependency, and rustmath-integers does not re-export it. So step 4 does not compile
as planned. Add `num-bigint = { workspace = true }` to plan step 1, next to `rug` (the crate is already
in the graph; this is not a wrapper).

### m3. `TupleIterator::size_hint` can report a lower bound above the remaining count

Item 11 returns `(usize::MAX, None)` whenever n^k does not fit. For n^k = 2^64 (for example
`TupleIterator::new(2, 64)`, or 4^32, 16^16), after two calls of `next()` only 2^64 - 2 = usize::MAX - 1
tuples remain. The lower bound then exceeds the true count, which breaks the `Iterator::size_hint`
contract. Fix: compute the remaining count as an `Integer` (n^k minus the rank of `current`, both
exact) and return `(min(remaining, usize::MAX), None)`. Or return `(0, None)` when n^k does not fit.
Test: (2, 64) after two `next()` gives a lower bound of at most usize::MAX - 1.

### m4. One more should_panic test cannot run safely on HEAD

At HEAD, `count_non_decreasing_parking_functions(1 << 40)` aborts the process when it tries to allocate
its 2^40 + 1 word DP (§6.5, its own HEAD column). Add it to the list of §6.5 and §8.2 that runs only
under `ulimit -v` and `timeout`, or is skipped in the HEAD check. Otherwise an abort ends the whole test
binary and hides the other results.

### m5. The empty partition

`number_of_standard_tableaux` defines l_1 = lambda_1 + r - 1, which does not exist for r = 0. State
f(empty partition) = 1 (MAGMA, and `hook_length_dimension` today) as an early return, and add the row to
`standard_tableaux_counts`. `Partition::dimension()` of the empty partition must stay 1.

### m6. Acceptance: the other direct dependents

§8.2 runs the tests of jupyter, symmetricfunctions, quantumgroups and groups. rustmath-algebras,
-liealgebras, -crystals and -topology also depend on the crate directly. My grep finds no call of a
changed function there, so `cargo check --workspace --tests` covers them. Still, add
`cargo test -p rustmath-algebras -p rustmath-liealgebras -p rustmath-crystals -p rustmath-topology
--no-fail-fast`, or say in §8.2 why it is not needed.

### m7. Doc fixes to name

- `lib.rs:478` calls `falling_factorial` "(Pochhammer symbol)". MAGMA ch. 49 and Sage use that name for
  the rising factorial (§4.1). Plan step 5 should remove it there and put it on `rising_factorial`.
- The three new test files are new files: their headers should cite the MAGMA chapters (and Johansson
  2012 for ch145), as CLAUDE.md asks for each new file. Say so in plan step 11.

### m8. Nit: an outside-crate twin to report next to `gap_parser.rs`

`rustmath-algebras/src/temperley_lieb_algebra.rs:317 catalan_number(n: usize) -> usize` is the same
wrapping `result * (2n - i) / (i + 1)` as the private Catalan helpers that items 10 and 17 delete. It
returns 3793109252770211 for n = 33 (truth C_33 = 212336130412243110, which fits; I simulated the wrapping
loop in Python). `TemperleyLiebAlgebra::new(33, delta).dimension()` reaches it at once, and `new` only
stores the order. Report it in §1.8 with the `gap_parser.rs` lcm. It is outside this unit.

## Checks that the design passes

- Coverage. The five named S2 sites; S3 (certified HRR with a stated domain and a documented panic);
  S4 (each named callee, measured before and after); S10 (Integer and ring-element versions, with the
  `one` argument for parent-carrying rings); the §7 list for J1b. Nothing is declared out of scope.
  Except for B1, the revision-2 items 24-36 are reproduced and their fixes are sound.
- No facade. `number_of_partitions` returns only an enclosure-proved value. The machine-word functions
  panic with documented messages instead of wrapping. The usize::MAX sentinel of item 23 and the
  `false` of `exceeds_sunflower_bound` are documented comparison values. The mask loops now run their
  2^n steps instead of returning a truncated result. The multinomial zero for a wrong sum is the
  coefficient interpretation, as at HEAD.
- Tests use gp and GAP values (sampled above). The new tests fail on HEAD: they do not compile, or a
  wrong value or another panic message appears.
- The plan order is implementable once m2 is added, and the acceptance commands cover the crate, the REPL,
  the Partition and Permutation users, and `cargo check --workspace --tests`.

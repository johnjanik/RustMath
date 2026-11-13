# Implementation report: Unit P1 (rustmath-polynomials, with the pgalois, maclane and typed-factor callers)

Design: `fu2/design/P1-polynomials.md` (Revision 1 body), approved by review 2 (8 minor notes).
Base: HEAD `0ceaeaa` (I3, I1, I2 committed). The working tree held steps 1-2 of an interrupted
attempt at this unit (core `static_is_commutative` and the Integer/Rational overrides with their
test assertions); I checked them against the design and kept them unchanged.

Handoffs from earlier units: I1 hands P1 "`berlekamp::check_prime` uses `try_is_prime`; the two
`internal` texts are untouched (P1's B.7); `prime::is_prime_u64` is public". Both hold at HEAD
and are used below. I3 and I2 hand nothing to P1. I3's `to_i64` migrations in `pgalois.rs`,
`padic_factor.rs`, `zassenhaus.rs`, `sieve.rs`, `galois_ctx.rs` were kept (pgalois's reduction
loop is rewritten, it now uses `Integer::modulo` + `i64::try_from`).

## Per item: what changed

### A. Distinct-degree factorization checks its input

- `rustmath-polynomials/src/fp_factor.rs:435` `distinct_degree_factor`: checks `p`
  (`check_modulus`), reads `f` modulo `p`, makes it monic, then `squarefree_check`; panics
  `"fp_factor::distinct_degree_factor: {why}"`. The HEAD loop moved to the private
  `ddf_squarefree` (`:451`), which `factor` calls on the radical.
- New private helpers: `reduce` (`:306`), `monic_nonzero` (`:312`, the zero check of the
  checked entries), `squarefree_check` (`:323`). `block_contract` uses `squarefree_check`
  (same message "f is not squarefree", after its own `f = 0` check).
- Tests (`fp_factor.rs` tests): `ddf_rejects_non_squarefree`, `ddf_rejects_hidden_repeated_factor`,
  `ddf_rejects_zero_derivative`, `ddf_rejects_zero`, `ddf_rejects_composite_p`,
  `ddf_constant_and_squarefree`, `ddf_reads_coefficients_mod_p` (gp commands in the block
  comment above them).

### B. Prime checks

- `fp_factor.rs:592` `pub fn check_modulus(p)`: `(2..=MAX_P).contains(&p) &&
  rustmath_integers::prime::is_prime_u64(p as u64)` (I1's exact word test); replaces the private
  `check_prime` (same message, so T5 keeps its expectations).
- Checked entries, each `check_modulus` first, then reduce, then panic on `f ≡ 0`
  ("f = 0 has no factorization", review 2 m1): `factor` (`:809`), `factor_with_multiplicity`
  (`:854`), `squarefree_factor` (`:379`, wrapper over the private `radical`, `:389`),
  `equal_degree_factor` (`:769`). New `pub(crate) factor_prime` (`:824`, no prime check, used by
  `zassenhaus::choose_prime`) and the private `factor_monic` (`:829`).
- Module header (`:33-53`) rewritten: checked entries vs unchecked arithmetic helpers. One
  em-dash in the header removed.
- `padic_factor.rs:72, :143, :185`: `cycle_type`, `padic_factor`, `ramification_type` check `p`
  first, panic with their own prefix; `# Panics` sections, and for the `Result` entries the reason
  it is a panic and not a `PadicError` (review 2 m7: `ideals.rs:150-158` reads `Err` as
  `montes_resolved = false`). The doc of `cycle_type` now says that a composite `p` gave a wrong
  value at HEAD (`x^2+1` at 15 gave `[2]`, review 2 m5).
- `rustmath-groups/src/pgalois.rs`: `unramified_local_galois_group` (B.4 steps 1-7: primitive
  part of the trimmed f, degree check, `check_modulus`, `disc(g)`, `g mod p`, the root at
  infinity when `deg(g mod p) = n - 1`, `unreachable!` below that), with the proof in its doc;
  `local_galois_candidates` (prime check first, then the primitive part; constants and the empty
  slice give `([], false)`; the ramified branch uses `disc(g)`); `is_padic_square` panics unless
  `is_prime_u64(p)`. Doc of `UnramifiedLocalGalois` and of its `frobenius_cycle_type` field
  updated; module header cites MAGMA ch. 23 §23.8.1 and states the prime check.
- `rustmath-igp24/src/main.rs`: the parsed prime list is validated with `check_modulus`; a bad
  entry prints `igp24: p = 15 is not a prime <= 3037000499` on stderr and exits with status 2.
  Checked by hand: `rustmath-igp24 '[1,0,1]' '[5,15]'` prints that line, status 2;
  `'[1,0,1]' '[5,7]'` prints the JSON (`cycle_types {"5":[1,1],"7":[2]}`), status 0.
- B.7 (`berlekamp.rs`): kept I1's `check_prime` structure (review 2 m4) and split its result:
  `Ok(false)` stays `InvalidArgument("... is not a prime")`, an unresolved proof becomes
  `NotSupported("berlekamp: the primality of the modulus {p} is unresolved ({e})")` (`:78`).
  `internal` (`:67`) now says "the modulus is a proven prime, so this is a bug"; `not_prime` is
  renamed `no_inverse` (`:61`) and is `internal("a nonzero coefficient has no inverse modulo the
  proven prime {p}")`. I1 had already added the doc sentence on the unresolved case to `factor`
  and `is_irreducible`. The stale "3.19e23" sentence in `factorization::berlekamp_factor_gf`'s doc
  is rewritten.
- Tests: `fp_factor.rs` `factor_rejects_composite_p`, `factor_rejects_p_one`,
  `factor_rejects_large_prime`, `factor_with_multiplicity_rejects_composite_p`,
  `squarefree_factor_rejects_composite_p`, `factor_rejects_zero_mod_p` (`[5,10]` mod 5),
  `factor_with_multiplicity_rejects_zero`, `squarefree_factor_rejects_zero_mod_p`,
  `factor_reads_coefficients_mod_p`, `factor_reduces_leading_coefficient`,
  `factor_with_multiplicity_reduces_leading_coefficient`, `equal_degree_factor_rejects_zero_mod_p`,
  `check_modulus_boundary` (2047, 1373653, 25326001, 3215031751 and the range cases; gp
  `isprime` and strong tests by hand); `padic_factor.rs` `cycle_type_rejects_composite_p`,
  `padic_factor_rejects_p_one`, `ramification_type_rejects_composite_p`;
  `tests/ch23_berlekamp.rs:717` `berlekamp_rejects_strong_pseudoprime_psi12` (gp `isprime` 0,
  `factor` 399165290221 · 798330580441); `rustmath-groups/tests/pgalois.rs:92-175` (12 tests: the
  B.6 table, with p = 0 and p = 1 as two tests; gp `poldisc`, `factormod`, `factorpadic(f,p,8)`
  rerun for every row, command block above the tests).

### C. Equal-degree cost and the rings callers

- `fp_factor.rs:632` `block_contract`: `prime_divisors(d)` (`:339`) first, then the d Frobenius
  steps keeping `X_0`, the current rung and the rungs `X_(d/l)` only; checks and messages in HEAD
  order.
- `rustmath-rings/src/valuation/maclane.rs`: the `!is_irreducible_fp(&q, p)?` clause is gone (the
  `degree(&q) != d` check stays); `factor_fp_certified`'s doc and the comment block before
  `is_irreducible_fp` rewritten (fp_factor handles inseparable inputs since `5ff8749`).
  `residue_tower.rs:19-23` rewritten the same way. `function_field/typed/factor.rs:156` uses
  `rustmath_polynomials::fp_factor::MAX_P`.
- Tests: `certificate_composite_d_accepts`, `certificate_composite_d_rejects_divisor_degree`,
  `certificate_d6_two_cubics`, `certificate_d6_irreducible`, `prime_divisors_small` (gp
  `factormod`, `polisirreducible`, `Vecrev(lift(...))` in the block comment).

### D. Phase 1 removed, sorted output

- Deleted `LEGACY_TRIES`, `random_monic_polynomial`, `SplitMix64::below_legacy` and the phase-1
  loop. `edf_rec(f, d, p, out)` (`:694`): at most 256 uniform tries, `contract_or_panic` before try
  32 (`TRIES_BEFORE_CERTIFICATE`, `:494`).
- `pub fn factor_order` (`:604`); `factor_monic`, `equal_degree_factor` sort with it;
  `factor_with_multiplicity` inherits the order. `zassenhaus::factor` sorts with
  `zx_factor_order` (`zassenhaus.rs:412`).
- Tests: `factor_order_is_unchanged` replaced by `factor_order_is_pari_order` (the reference rows
  of `edf_reference_rows` are now stored in PARI's printed order, gp `apply(le, factormod(f,p)[,1]~)`
  rerun for all 19 rows; plus the three mixed rows, also through `factor_with_multiplicity`);
  `phase_two_alone_matches_pari` became `edf_rec_matches_pari`; T1 compares as sets (both sides
  sorted); `edf_rec_certificate_panics_on_bad_block` uses the new signature and now pins the full
  reason; `split_with_exhaustive_counts`'s comment no longer says "phase 1" (review 2 m8);
  `ch23_berlekamp.rs::factor_agrees_with_cantor_zassenhaus` asserts that `fp_factor`'s output is
  already in order before the sort; `zassenhaus.rs::factor_order_matches_pari` (12 polynomials,
  gp `factor(f)` rerun).

### E. Zassenhaus error type

- `zassenhaus.rs:29` `pub enum ZassenhausError { NoGoodPrime, RecombinationLimit,
  HenselLiftFailed }`, `Display` with the design texts, `std::error::Error`, and
  `impl From<ZassenhausError> for MathError` (`:72`): `HenselLiftFailed` gives
  `InvalidOperation` (review 2 m8, optional, taken), the two limits give `NotSupported`.
- `choose_prime(f, prime_bound)` (`:163`, loop `p <= prime_bound`, calls `factor_prime`),
  `recombine(.., cap)` (`:203`), `factor_monic_squarefree_with(f, prime_bound, cap)` (`:292`),
  `factor` returns `Result<Factorization, ZassenhausError>` (`:385`) with a rewritten doc. `MAX_RECOMBINE`
  is now `pub` (the public doc links it).
- Callers: `factorization.rs` (private `zassenhaus_error(who, e)`, `:239`; messages
  "is_irreducible: ...", "factor_over_integers: ...", "is_irreducible_over_integers: ..."; doc of
  `factor_over_integers` rewritten, it delegates to Zassenhaus; doc of `is_irreducible` names the
  causes), galois `cayley.rs:314`, `lowdegree.rs:64`, `quintic.rs:251`, `lib.rs:180`
  (`.map_err(MathError::from)`), modular `decomposition.rs:247` (`e.to_string()`), curves
  `belyi/regular_certify.rs:215` (`Err(e) =>`, the message carries `{e}`).
- Tests: `factor_scans_past_bad_small_primes`, `choose_prime_respects_bound`,
  `recombination_limit_is_reported`, `error_messages`, and
  `factorization::tests::is_irreducible_scans_past_bad_small_primes` (gp: `#Str(P) = 43293`,
  `polisirreducible(x^2-P) = 1`, `nextprime(10^5) = 100003`, `kronecker(P,100003) = -1`). The
  primorial is built with `rustmath_integers::sieve::primes_up_to(99_999)`.

### F. Cohen citations

- `zx.rs:12-22`: MAGMA ch. 23 sections added; "3.4.2 (square-free)" replaced by Yun 1976
  (SYMSAC, 26-35) with the remark that Cohen Alg. 3.4.2 is its characteristic-p form (Sage
  `finite_field_base.pyx` quote); `squarefree_decomposition` doc cites Yun. One em-dash in the
  header replaced by a colon.
- `univariate.rs:28-32` and `:1286`: "§3.2" became "§3.2.3-3.2.4", with the library catalogue
  record in the header.

### G. Ring hooks

- Core, integers, rationals: unchanged from the interrupted attempt (`traits.rs:191`
  `static_is_commutative`, overrides for `i32`, `i64`, `Integer`, `Rational`; one assertion each in
  `test_i32_i64_element_predicates` and the two `tests/ring_predicates.rs`).
- `univariate.rs:873-1117`: `UNIT_INVERSE_WORK = 1 << 16`, a private `convolve` (products seeded
  by their first term, no `R::zero()`, review 2 m6), and the hooks `unit_status`,
  `nilpotent_status`, `unit_inverse` (power-series recurrence with the n-zeros stop rule, an early
  `Err(NotImplemented)` when `n(n+1)/2 > 65536`, both products checked through `convolve`),
  `multiplicative_order_status`, `static_characteristic`, `static_is_commutative`.
- `laurent.rs:215-495`: the same hooks for `LaurentPolynomial` (monomials delegate; non-monomial
  rules of G.2 plus review 2 m2: every ring, two unit extreme coefficients give `Some(false)`;
  commutative, nilpotent iff every coefficient is). The commutative rules run only when
  `R::static_is_commutative() == Some(true)`. Inverse from the low side, then the high side, with
  checked exponents and a span bound before the dense vector is built (gaps filled with `c - c`).
  `add_term` (`:68`) no longer calls `R::zero()` (needed by the no-static-constructor test; same
  behaviour).
- `rustmath-algebras/src/algebra_with_parent.rs:275-278`: `test_simple_algebra_finiteness` now
  expects `Some(false)` for `SimpleAlgebra<UnivariatePolynomial<Integer>>`, with the new comment.
  This replaces an undecided `None` by a decided value; GAP `IsFinite(PolynomialRing(Integers, 1))`
  is `false` (rerun). The module is behind the `full` feature, so I ran it with
  `--features full` (the workspace build enables `full` through `rustmath-jupyter`).
- Tests: new `rustmath-polynomials/tests/ring_predicates.rs` (10 tests: the 9 rows of G.4 plus
  `hooks_call_no_static_constructor`). GAP references rerun (`units_p1.g` in my scratch, command
  block in the file header): orders of `1+2t` over Z/8 (4) and `5+6u` over Z/12 (2), both
  products of the Z/8 inverses, `(1+2s)·h = 1` over Z/2^62, both M2 products I, `4^k mod 12 = 4`.

### H. MAGMA conventions (documentation)

- `univariate.rs` `div_rem` doc: V2.19 text, Sage/FLINT `quo_rem` doctests (develop source on
  GitHub, review 2's note); `content` doc: V2.19 wording and Sage `content_ideal` / `QQ.ideal(2)`.
- `factorization.rs:9-14` (`content`) and `:115-123` (`is_irreducible`, Sage `is_irreducible`
  doctests). `berlekamp.rs:322-330` `q_matrix` doc: the CompanionMatrix "degree d - 1" evidence
  and Knuth's d × d Q.
- Tests: `tests/ch23_quotrem_gcd.rs` header and two new cases in
  `quotrem_over_z_minimal_remainder_is_not_unique` (`(5x+7).div_rem(3) = (0, 5x+7)` with the FLINT
  alternative `(x+2, 2x+1)` checked to be a valid quotient and remainder of degree 1;
  `(x^2).div_rem(2x-1) = (0, x^2)`); `tests/ch23_is_irreducible.rs:868` `sage_content_examples`.

## API and behaviour changes (callers checked)

- New public API: `fp_factor::check_modulus`, `fp_factor::factor_order`,
  `zassenhaus::ZassenhausError`, `zassenhaus::Factorization` (type alias of the `Ok` value),
  `zassenhaus::MAX_RECOMBINE` (now pub), `From<ZassenhausError> for MathError`,
  `Ring::static_is_commutative`.
- Breaking: `zassenhaus::factor` returns `Result<_, ZassenhausError>` (was `Result<_, ()>`). All 12
  call sites compile; only `regular_certify.rs` needed a code change (`Err(()) =>`). `audit.rs:390`
  and functionfields `genus.rs:223` match `Err(_)`.
- Panics on a bad `p` (composite, < 2, > MAX_P) in: `fp_factor::{factor,
  factor_with_multiplicity, squarefree_factor, distinct_degree_factor}`,
  `padic_factor::{cycle_type, padic_factor, ramification_type}`, `pgalois::{unramified_local_galois_group,
  local_galois_candidates, is_padic_square}`, and through them galois `sieve::cycle_type_mod_p`,
  `galois_ctx::build_ctx`, numberfields `prime_decomposition`, `prime_ideals`,
  `local_decomposition`, `quartic_local_galois_group`. Every workspace caller passes primes (the
  acceptance runs below are green); the igp24 command line validates its list.
- Panics on `f ≡ 0 (mod p)` in `factor`, `factor_with_multiplicity`, `squarefree_factor` (HEAD:
  `[]`, `[0]` or a `pth_root` panic). No caller passes 0 (review 2 m1 lists the guards).
- Output order: `fp_factor::factor`, `factor_with_multiplicity`, `equal_degree_factor` and
  `zassenhaus::factor` are in PARI order. The D.3 consumers' tests pass unchanged.
- pgalois: inputs with p | content, p | lc, or untrimmed coefficients now give the true local group
  (B.1 table); a constant `f` panics in `unramified_local_galois_group` (HEAD `None`).
- Ring hooks: polynomial and Laurent types now decide where they returned `None`; the one test
  that pinned the old `None` is the algebras finiteness test above.
- `distinct_degree_factor` and `squarefree_factor` read coefficients modulo `p` (HEAD echoed
  unreduced input).

## Deviations from the design, with evidence

1. Review 2 m3 implemented (not only recorded): for a proven unit of degree >= 1 the order is
   searched (`f^k = 1`, work bound 65536 products), so `5 + 6u` over Z/12 is `Finite(2)` (GAP
   `(5+6u)^2 = 1`), not the design's documented `Undecided`; `1 + 2t` over Z/8 is `Finite(4)`.
   The same search runs for Laurent non-monomial units (`1 + 2x^-1` over Z/8: `Finite(4)`).
   A search that finds nothing stays `Undecided` (test: `x^-3 + 2x^-1`).
2. Laurent `unit_status`, commutative case, has one more rule than G.3: an extreme unit
   coefficient with another proven non-nilpotent coefficient gives `Some(false)` (the G.2
   argument: f unit and c_a unit force c_(a+1..b) into every prime). Without it, a `NotInvertible`
   from the inner polynomial inverse would be reported by `unit_inverse` but not by
   `unit_status`.
3. `unit_inverse`: when `f·h = 1` but `h·f != 1` over a coefficient type marked commutative, the
   answer is `Err(InvalidOperation)` (inconsistent arithmetic), not `Err(NotInvertible)`: in a
   commutative ring the two products are equal, so a difference can only come from the arithmetic.
   The work-bound message is "no inverse of degree < {k - n} found within 65536 coefficient
   products" (k the step reached; the design's "< {k}" overstated what is proven: an inverse of
   degree d is only seen at step d + n).
4. `UnivariatePolynomial::unit_inverse` and the Laurent inverse return `Err(NotImplemented)` at
   once when `n(n+1)/2 > 65536` (no inverse can be finished within the bound). This also keeps the
   Laurent path from allocating a dense vector over a huge exponent span.
5. The pgalois `is_padic_square` and `check_prime` panics are as designed; `local_galois_candidates`
   panics with its own prefix `pgalois::local_galois_candidates` (the design's test expects only
   "is not a prime"; I pinned the full prefix).
6. The HEAD baseline of the new tests was not rerun: the rules forbid checkout/stash. The designer
   (`$X/t*.rs`, `$X/rev1/r1_head.rs`, `r2_psi12.rs`) and review 2 (`$R/h1.rs`, `h2.rs`, `h3.rs`)
   reproduced every HEAD value the tests contradict, against the HEAD rlibs. Review 2 m5 notes that
   on HEAD `factor(&[1,0,1], 1)` overflows the stack, which aborts a whole test binary.
7. `squarefree_factor` panics on `f ≡ 0` with "f = 0 has no factorization" (review 2 m1 asks for
   the same message in all three entries; HEAD returned `[0]`).

## Review 2 minor notes

- m1 (zero mod p): done, three `should_panic` tests.
- m2 (Laurent rules): done, test rows `2x^-1 + 4x` (nilpotent), an interior non-nilpotent row,
  `I x^-1 + I x` over M2 (not a unit).
- m3 (order search): done (deviation 1).
- m4 (B.7 vs I1): done as described in B.
- m5 (HEAD baseline, padic sentence): the padic doc says what HEAD returned; the baseline note is
  deviation 6.
- m6 (static constructors): done (`convolve`; Laurent `dense` and `sparse_product`; `add_term`);
  test `hooks_call_no_static_constructor` with a coefficient type whose `zero()`/`one()` panic.
- m7 (`# Panics` on the Result entries): done.
- m8: the `split_with_exhaustive_counts` comment and `HenselLiftFailed -> InvalidOperation` done.
  The `branch_cycles.rs:27` row belongs to the design's D.3 table, not to code; noted here:
  functionfields `branch_cycles.rs:27` imports `fp_factor as fp` for arithmetic helpers only.
- Skipped: none.

## Test counts (acceptance commands)

| Command | Passed | Failed | Ignored |
|---|---|---|---|
| `cargo test -p rustmath-core --no-fail-fast` | 49 | 0 | 1 |
| `cargo test -p rustmath-integers --test ring_predicates` | 1 | 0 | 0 |
| `cargo test -p rustmath-rationals --test ring_predicates` | 3 | 0 | 0 |
| `cargo test -p rustmath-polynomials --no-fail-fast` (rerun after the last edits) | 424 | 0 | 4 |
| `cargo test -p rustmath-groups --no-fail-fast` | 668 | 0 | 4 |
| `cargo test -p rustmath-rings --no-fail-fast` | 1914 | 0 | 82 |
| `cargo test -p rustmath-galois --no-fail-fast` (includes `sieve`) | 85 | 0 | 5 |
| `cargo test -p rustmath-numberfields --no-fail-fast` | 166 | 0 | 1 |
| `cargo test -p rustmath-curves --no-fail-fast` | 305 | 0 | 13 |
| `cargo test -p rustmath-modular --no-fail-fast` | 221 | 0 | 1 |
| `cargo test -p rustmath-functionfields --no-fail-fast` | 27 | 0 | 0 |
| `cargo test -p rustmath-symbolic --no-fail-fast` | 772 | 0 | 16 |
| `cargo test -p rustmath-category -p rustmath-modules -p rustmath-algebras --no-fail-fast` | 1112 | 0 | 1 |
| `cargo test -p rustmath-algebras --features full --no-fail-fast` (the changed finiteness test) | 868 | 0 | 1 |
| `cargo build -p rustmath-igp24` | ok | | |

The polynomials total went from 377 (I1 report) to 424: 47 new tests (fp_factor 26, zassenhaus 5,
factorization 1, padic_factor 3, berlekamp 1, is_irreducible 1, ring_predicates 10, with the
pinned-order test replaced and `phase_two_alone_matches_pari` renamed). groups +12 (pgalois).

`cargo check --workspace --tests`: green (0 errors). `cargo clippy -p rustmath-polynomials -p
rustmath-groups`: the command as written fails on deny lints in code this unit does not touch
(`rustmath-combinatorics` `binary_words.rs:317` and four more, `rustmath-groups`
`indexed_free_group.rs:181`; unchanged since `c79a582`). With `--no-deps` (and `--all-targets`
for polynomials) no lint points at a line this unit added; I fixed the two `type_complexity`
warnings my code raised (`laurent.rs` `Term` alias, `zassenhaus::Factorization` alias). The
remaining warnings in touched files are on older lines (`fp_factor.rs` `div_mod`, `pow_mod`,
`derivative_of`; `padic_factor.rs` `(g.len() - 1).max(0)`; `factorization.rs`).
`cargo doc -p rustmath-polynomials --no-deps`: no warning on an added line (the 21 warnings are
the old `Z[x]`-style links); `target/doc` deleted afterwards.

## Left undone

- Nothing of the design. Not attempted (out of scope, listed for later): the facade-like `Err`
  handling of Zassenhaus results (design section I), `static_is_commutative` overrides for the
  finitefields, reals, complex and p-adic types.

## Handoffs

- F1 (finitefields): `Ring::static_is_commutative` exists (default `None`); the finite-field
  element types can return `Some(true)` (one line each), which lets `UnivariatePolynomial` over
  them use the commutative unit/nilpotent rules.
- R1 (rings, p-adics): the polynomial and Laurent hooks call no static constructor of the
  coefficient type, so they are safe over the p-adic types; `Real` and `Complex` (and the p-adic
  types) can return `static_is_commutative() = Some(true)`; then a mismatch of the two products in
  `unit_inverse` over inexact types is reported as `InvalidOperation`, not `NotInvertible`.
- Trackers (orchestrator), per design Plan step 11: `PROJECT_TRACKER.md` rows for fp_factor
  (checked entries, `check_modulus`, DDF squarefree check, sparse Rabin ladder, no phase 1, PARI
  order), zassenhaus (typed `ZassenhausError`, prime scan to MAX_P, PARI order), pgalois
  (primitive part, root at infinity, prime check), maclane (redundant Rabin removed), the core
  three-valued predicates row (polynomials and Laurent polynomials no longer keep the defaults;
  new `static_is_commutative`), the MAGMA convention notes (H.1-H.4 evidence). `docs/HANDOFF.md`:
  remove the P1 follow-ups; add design section I: (1) facade-like `Err` handling of Zassenhaus
  results at `rustmath-groups/src/galois_resolvent.rs:159-169`, `rustmath-curves/src/belyi/audit.rs:390-398`,
  `rustmath-igp24/src/main.rs:117` (`unwrap_or(false)`), `rustmath-functionfields/src/factor.rs:70`,
  `genus.rs:240-244`; (2) `local_galois_candidates` assumes a transitive local group at ramified
  primes; (3) `LaurentPolynomial::is_one` is false for the zero polynomial of the zero ring;
  (4) `rustmath-numberfields/src/panayi.rs:326` duplicates `is_padic_square` without the prime check.
  New: (5) `LaurentPolynomial`'s `Mul` adds `i32` exponents unchecked (`laurent.rs`, `p1 + p2`;
  overflow panics in debug); the hooks use a checked product.
- Note for the full workspace run: `cargo test -p rustmath-numberfields` spends about 12 minutes in
  two `recognize::tests::hp_*` tests (debug build, high-precision LLL; they do not use the code of
  this unit).

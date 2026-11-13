# Design P1: rustmath-polynomials (with the named callers in rustmath-groups and rustmath-rings)

## Revision 1

This revision answers review 1 (`P1-polynomials.review1.md`: 2 blockers, 11 minor issues). I accept every
issue. For m10 the review allowed a documented panic; I took the other option and avoided the panic. For m5
I replaced the algorithm instead of only adding a bound. The body below is updated so that it reads as one
design. The experiments for this revision are in `$X/rev1/` (`$X` is defined below).

| Issue | What changed | Where |
|---|---|---|
| Blocker 1: the pgalois `unreachable!` is reachable for linear f with p \| content | Both pgalois entries now work on the primitive part `g = pp(trim(f))`. With p ∤ content(g), the binary-form argument holds for every n >= 1, so `deg(g mod p) >= n - 1` and the `unreachable!` is sound. I reproduced HEAD (`$X/rev1/r1_head.rs`): `6x+6` at 3, `2x` at 2 and `-2x+4` at 2 give `[]`, not transitive. Side effects: `3x^2+3` at 3 is now decided (`[2]`, transitive; HEAD `None`). `local_galois_candidates(3(x^4+x+1), 3)` is now `([], true)`; HEAD gives `(["4T2","4T4"], false)`, and neither group is the true local group. A randomized gp check of the revised rule (`$X/rev1/pgrand_pp.gp`) covers 28,213 pairs (f, p) with p ∤ disc(pp f), including 2,231 with p \| content(f) and 7,356 linear ones. It found 0 mismatches against `factorpadic` and 0 cases with `deg(g mod p) < n - 1`. | B.1, B.4, B.5, B.6, Risks |
| Blocker 2: the G delegation breaks `rustmath-algebras` `test_simple_algebra_finiteness` | The test (`algebra_with_parent.rs:274-277`) changes to `Some(false)` with a new comment. The new value is right (GAP `IsFinite(PolynomialRing(Integers,1))` is false). I redid the consumer grep over all Rust code, unit tests in `src/` included (G.3): this is the only test that changes. | G.3, G.4, Plan step 7 |
| m1: `reduce` is not covered by the tests | New rows `factor([1,1,5],5) = [[1,1]]` (HEAD panics in `div_mod`), `factor([1,5],5) = []` (HEAD panics in `pth_root`), `factor_with_multiplicity([1,1,5],5) = [([1,1],1)]` (HEAD panics), `equal_degree_factor([5],1,5)` panics "f = 0" (HEAD returns `[]`). The `[7,0,6]` row stays as a guard; it passes on HEAD, and the table now says so. | B.6 |
| m2: no strong pseudoprime below MAX_P is tested | `check_modulus_boundary` adds 2047, 1373653 and 25326001 (each `Err`). gp, `$X/rev1/rev1.gp`: each is composite; they are strong to the bases [2], [2,3] and [2,3,5] but not to the next base. 3215031751 stays, labelled as the range check. | B.6 |
| m3: overlap with I1 | The Plan says "rebase on I1" and lists I1's edits in P1 files. I accept the I1 §7 hand-off: `berlekamp::check_prime` uses `try_is_prime`, and an unresolved proof becomes `Err(NotSupported)` instead of a panic. The two berlekamp error texts that mention "a composite modulus that passed the primality test" are rewritten, because after I1 no composite passes. | new B.7, Plan |
| m4: the Plan breaks the build between steps 4 and 10 | The one-line fix in `rustmath-curves/src/belyi/regular_certify.rs:215` moves into step 4 (the only caller that stops compiling). Each step now leaves `cargo check --workspace --tests` green. | E.3, Plan |
| m5: the `unit_inverse` cap bounds terms, not work | The finite series `Σ (-M)^k` is replaced by the power-series recurrence `h_k = -b Σ_{i=1}^{min(k,n)} a_i h_{k-i}`. It stops exactly after n consecutive zero coefficients. The work bound is `UNIT_INVERSE_WORK = 2^16` coefficient products, so time and memory are both bounded: at most 65536 products and 65537 stored coefficients, so an inverse of degree d is found whenever n·(d + n) <= 65536. Past the bound the result is `Err(NotImplemented)`. The recurrence needs no commutativity, and a two-sided check makes every returned inverse a proof. The Laurent inverse of a non-monomial reuses it on the side whose extreme coefficient is a unit. Prototype `$X/rev1/unit_inverse_proto.py`: Z/8, Z/12, the M_2(Z) example, Z/2^e for e up to 20000, and the cap. | G.2, G.3, G.4, Risks |
| m6: the Laurent exponent can overflow | `k.checked_neg()` (and checked exponent sums in the non-monomial inverse); overflow gives `Err(NotImplemented)`. New test. | G.3, G.4 |
| m7: the scope wording | G.1 now says that no polynomial parent type implements `Parent` or `RingParentMethods`. The only `PolynomialRing` struct (`rustmath-algebras/src/cached_algebra.rs:15`) is a caching demo. So the element-level hooks are the whole scope. | G.1 |
| m8: the Cohen §3.2 citation | Cite "§3.2.3-3.2.4". A library catalogue of GTM 138 (bib-bvb.de, record BV013376443) lists 3.2.3 "Polynomials over Unique Factorization Domains" (p. 116) and 3.2.4 "Euclid's Algorithm for Polynomials over a UFD" (p. 117). | F |
| m9: public entries that now panic on a bad p | Listed as behaviour changes: `rustmath_galois::sieve::cycle_type_mod_p` (through the DDF), and numberfields `prime_decomposition`, `prime_ideals`, `local_decomposition`, `quartic_local_galois_group` (through `fp_factor::factor`). No caller passes such a p. | A.3, B.5 |
| m10: `local_galois_candidates` on a constant | I chose not to add this panic. `local_galois_candidates` keeps returning `([], false)` for a constant f, as at HEAD ("no claim"). The empty slice, which underflows at HEAD, now returns the same. Only `unramified_local_galois_group` panics on a constant; at HEAD it returns `None` (disc 0, "ramified"), which is a misleading claim. Listed in B.5. | B.4, B.5 |
| m11: no DDF test with unreduced input | New test `ddf_reads_coefficients_mod_p`: `[-4,1]` mod 5 gives `[(1,[1,1])]`. HEAD returns `[(1,[-4,1])]`, so it fails on HEAD. | A.4 |
| m12: the timing claim | The text now says "comparable within measurement noise", and quotes the reviewer's spread (up to about ±40 % between reruns). | 0, D.2 |

Unit P1 of the HANDOFF follow-ups (`docs/HANDOFF.md`, "Open follow-ups", lines at `9c24389`, same code
as HEAD `c79a582`). Implemented after I3, I1 and I2, before F1, on top of I1's changes (see the Plan).

Experiments, prototypes and logs: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/P1-polynomials/`
(called `$X` below). The build script `$X/rc.sh prog.rs [opt]` links a standalone program against the
debug rlibs of HEAD. The files that matter:

| File | What it is |
|---|---|
| `$X/t1_composite.rs` | HEAD reproductions: `fp_factor::factor(x^2+1, 15)`, DDF on non-squarefree inputs |
| `$X/t2_pgal.rs` | HEAD reproductions for `unramified_local_galois_group` (composite p, p = 0, 1, non-monic f, untrimmed f) |
| `$X/t3_zass.rs` | HEAD: `zassenhaus::factor(x^2 - primorial(10^5))` returns `Err(())` |
| `$X/t4_zorder.rs` | HEAD Zassenhaus output order against PARI order (9 of 12 differ) |
| `$X/t5_isprime.rs` | Cost of HEAD `Integer::is_prime` on small primes (2.3 to 2.7 ms per call, debug) |
| `$X/t6_unreduced.rs` | HEAD `factor` echoes unreduced coefficients (`factor([-3,0,1], 5) = [[-3,0,1]]`) |
| `$X/t7_quotrem.rs`, `t8_ddf.rs` | HEAD values for H (Quotrem over Z, IsIrreducible content cases, Content over Q) and the DDF expectation of A.4 |
| `$X/psw_scan.rs`, `psw_scan*.log` | Exhaustive scan of odd composites for strong pseudoprimes to 2, 3, 5, 7 |
| `$X/psw.gp`, `order.gp`, `order_z.gp`, `pgal.gp`, `units.g` | PARI/GP and GAP reference scripts |
| `$X/proto/patch.py`, `proto/fp_new_patched.rs`, `proto/driver.rs` | Prototype of the new `fp_factor` (no phase 1, sorted output, checked p, checked DDF, sparse ladder) and its differential and timing driver |
| `$X/dl_sage/*`, `$X/dl_magma/vol02.pdf`, `dl_magma/ch23.txt` | Downloaded Sage sources (GitHub raw) and the MAGMA V2.19 handbook volume 2 (math.uzh.ch), data only |
| `$X/rev1/r1_head.rs`, `r2_psi12.rs` (`$X/rev1/rc.sh`) | Revision 1: HEAD values for the new rows (unreduced input, linear and content cases of pgalois, Berlekamp at psi_12) |
| `$X/rev1/pgrand_pp.gp`, `pgal_rev1.gp`, `rev1.gp` | Revision 1: randomized check of the primitive-part rule (28,213 cases), the pgalois reference rows, the strong pseudoprimes and the factormod rows |
| `$X/rev1/unit_inverse_proto.py`, `units_rev1.g` | Revision 1: prototype of the recurrence `unit_inverse` with its work bound; GAP references for its rows |

## 0. Summary

| Item | Verdict at HEAD | Fix |
|---|---|---|
| A. DDF does not check squarefree | Confirmed. `distinct_degree_factor((x+1)^2 (x^2+2)(x^2+3), 5)` returns `[(1, x+1), (2, x^5+x^4+x+1)]`: degree label 2 on a degree-5 block that holds `x+1`. | Eager squarefree check (one gcd) and prime check, panics with the reason. |
| B. Unchecked p in `fp_factor::factor` and `unramified_local_galois_group` | Confirmed (`factor(x^2+1, 15) = [[1,0,1]]`, pgalois cycle type `[2]` "transitive" at p = 15; p = 0 divides by zero; p = 1 gives "ramified"). Also found: pgalois is wrong for non-monic f with p \| lc (`2x^2+x+1` at 2 gives `[1]`, true `[1,1]`), for untrimmed input, and for f with p \| content (`6x+6` at 3 gives `[]`; `3(x^4+x+1)` at 3 gives wrong ramified candidates). | `check_modulus(p)` at every factorization entry (I1's word-size `is_prime_u64`); pgalois checks p, works on the primitive part of the trimmed f and adds the root at infinity. The Miller-Rabin claim is verified: no composite <= 3215031750 is a strong pseudoprime to 2, 3, 5, 7, and MAX_P = 3037000499 < 3215031751. Berlekamp's prime check uses I1's `try_is_prime` (B.7). |
| C. Equal-degree cost | Confirmed: `block_contract` stores d + 1 rungs; maclane runs Rabin on every piece after the certificate; the rings literal. | Keep only rungs d/l; maclane drops its redundant Rabin; `fp_factor::MAX_P` in rings. |
| D. Drop phase 1, sort `factor` | Feasible now. Prototype: same factor sets and multiplicities as HEAD on 600 random cases, PARI order on 18 reference rows, timings comparable to HEAD within measurement noise (D.2). | Phase 1 removed, lazy certificate after 32 uniform tries, output in PARI `factormod` order; `zassenhaus::factor` output in PARI `factor` order. Downstream audit: no test outside `fp_factor.rs` pins an order that changes. |
| E. Zassenhaus error text | Confirmed: `factor` returns `Err(())` also when no good prime exists below 100000 (`x^2 - primorial(10^5)`), and every caller says "recombination exceeded the subset bound". | Typed `ZassenhausError`; the prime scan runs to `MAX_P` (the "no good prime" case becomes unreachable in practice, so that input now factors); callers print the real cause. |
| F. Cohen GTM 138 numbers | Alg. 3.1.2 (pseudo-division) and 3.3.1 (sub-resultant gcd) confirmed (Sage source citations, step comparison); Alg. 3.4.2 is Cohen's characteristic-p square-free algorithm, not the Yun algorithm over Z it is cited for; §3.2 confirmed (§3.2.3 "Polynomials over Unique Factorization Domains", §3.2.4 "Euclid's Algorithm for Polynomials over a UFD"). | Doc fixes in `zx.rs`; `univariate.rs` citations made precise (§3.2.3-3.2.4), with the evidence noted. |
| G. Ring hooks for polynomials | Confirmed: `UnivariatePolynomial` and `LaurentPolynomial` keep all defaults (`2x+1` over Z is `None`). These are the only polynomial `Ring` types, and no polynomial parent type exists, so the element-level hooks are the whole scope. | Hooks decided from the coefficients: rules valid in every ring, plus Atiyah-Macdonald Ch. 1 Ex. 2 when the coefficient type is commutative (new core hook `Ring::static_is_commutative`); `unit_inverse` by an exact power-series recurrence with a work bound. One existing test changes (`rustmath-algebras`, `None` becomes `Some(false)`). |
| H. MAGMA conventions | Quotrem over Z, Content over Q, content in IsIrreducible over Z, QMatrix shape: all still undocumented by MAGMA beyond the text we had. New evidence: the V2.19 handbook text (same wording as V2.3), Sage sources. | Docs record the evidence; no behaviour change (no MAGMA evidence contradicts the current choices). |

Nothing is declared out of scope.

---

## A. Distinct-degree factorization does not check its input

### A.1 Defect at HEAD

`rustmath-polynomials/src/fp_factor.rs:340-366`: the doc says "of a squarefree polynomial", nothing checks
it, and p is not checked either. On a non-squarefree input the result breaks the documented contract
("`g_d` is the product of all irreducible factors of degree exactly d") in two ways (`$X/t1_composite.rs`,
`$X/proto/driver.rs` item 3i):

| Input (mod p) | HEAD | Reference |
|---|---|---|
| `(x+1)^2` mod 5 = `[1,2,1]` | `[(1,[1,1]), (1,[1,1])]` (degree 1 twice) | not squarefree |
| `(x^2+1)^2` mod 3 | `[(2,[1,0,1]), (2,[1,0,1])]` | not squarefree |
| `(x+1)^2 (x^2+2)(x^2+3)` mod 5 = `[1,2,1,0,1,2,1]` | `[(1,[1,1]), (2,[1,1,0,0,1,1])]` | the "degree 2" block `x^5+x^4+x+1` is `(x+1)(x^2+2)(x^2+3)` (gp `factormod(x^5+x^4+x+1,5)`): a wrong label, and an EDF with d = 2 on it panics |

Why: after step k only one copy of a degree-k factor g is removed; the second copy stays in `f_rem` and
is caught at the next multiple of k, or lands in the tail block (`:362-364`).

### A.2 Fix

MAGMA ch. 23 §23.8.1 `DistinctDegreeFactorization(f)` requires f squarefree (digest line 299). The fix
checks it eagerly, plus the prime (item B):

```rust
/// MAGMA `DistinctDegreeFactorization(f)` (ch. 23 §23.8.1). Coefficients are read modulo p.
/// Returns `(d, g_d)`, `d` increasing, `g_d` the monic product of the irreducible factors of
/// degree exactly d. A nonzero constant gives `vec![]`.
///
/// # Panics
/// `p` is not a prime <= MAX_P (checked first); `f ≡ 0`; `f` is not squarefree.
pub fn distinct_degree_factor(f: &[i64], p: i64) -> Vec<(i64, Vec<i64>)>
```

Body: `check_modulus(p)` (panic `"fp_factor::distinct_degree_factor: {why}"`), `f_m = make_monic(&reduce(f, p), p)`,
`squarefree_check(&f_m, p)` (panic with the same prefix), then the HEAD loop moved unchanged into a
private `fn ddf_squarefree(f_m: &[i64], p: i64) -> Vec<(i64, Vec<i64>)>`, which `factor` calls on the
radical (already squarefree, p already checked).

New private helpers:

```rust
/// Coefficients reduced into [0, p), trimmed.
fn reduce(f: &[i64], p: i64) -> Vec<i64>;
/// Ok(()) iff the monic, reduced f is squarefree: f != 0, and deg f <= 0 or (f' != 0 and gcd(f, f') = 1).
/// Err("f = 0 is not squarefree") / Err("f is not squarefree").
fn squarefree_check(f: &[i64], p: i64) -> Result<(), String>;
```

`block_contract` (`:542-545`) uses `squarefree_check` too (same message "f is not squarefree"; its own
`f = 0` check comes first and keeps its message "f = 0 has no factorization").

Correctness: over F_p, f is squarefree iff `f' != 0` and `gcd(f, f') = 1` (for `f' = 0`, f = g(x^p) = g^p).
The DDF loop is then exact (vzGG §14.2, Rabin 1980): after step d every factor of degree <= d is gone,
and the tail of degree < 2(d+1) is irreducible.

### A.3 Callers (grep `distinct_degree_factor`)

| Caller | Input | Change |
|---|---|---|
| `fp_factor.rs:742` (`factor`) | radical of f over a checked prime | calls `ddf_squarefree` directly, no extra cost |
| `rustmath-galois/src/sieve.rs:82` (in the public `cycle_type_mod_p(f, p: u64)`) | `reduce_squarefree_mod_p` already proved squarefree; inside the crate p comes from `is_prime_u64` (`frobenius_cycle_types`) | one extra prime test and gcd per prime (microseconds with I1's `is_prime_u64`) |
| `rustmath-rings/src/valuation/maclane.rs:1686` | `sf = work / gcd(work, work')`, squarefree; p proved prime by `PAdicBaseValuation::new` | none |
| `fp_factor.rs:859` (test) | `x^2+2` mod 5 | none |

No caller passes a non-squarefree polynomial or 0. One behaviour change reaches a public entry:
`rustmath_galois::sieve::cycle_type_mod_p` takes a caller-supplied `p: u64`, and a composite p or p > MAX_P
now panics in the DDF (`"fp_factor::distinct_degree_factor: p = .. is not a prime <= 3037000499"`). Before,
it returned a meaningless cycle type, `None`, or a `div_mod` panic. Its callers in the workspace pass primes
(`frobenius_cycle_types`, and the tests with the literal primes 2, 3, 5, 7), so no current output changes.

### A.4 Tests (all in `fp_factor.rs` tests)

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `ddf_rejects_non_squarefree` | `[1,2,1]`, 5 | `should_panic(expected = "fp_factor::distinct_degree_factor: f is not squarefree")` | fails (returns) |
| `ddf_rejects_hidden_repeated_factor` | `[1,2,1,0,1,2,1]`, 5 (gp `Vecrev(lift(Mod(1,5)*(x+1)^2*(x^2+2)*(x^2+3)))`) | same panic | fails (returns the wrong label) |
| `ddf_rejects_zero_derivative` | `[0,0,1]`, 2 | same panic | fails |
| `ddf_rejects_zero` | `[0]`, 5 | `"f = 0 is not squarefree"` | fails (returns `[]`) |
| `ddf_rejects_composite_p` | `[1,0,1]`, 15 | `"p = 15 is not a prime <= 3037000499"` | fails |
| `ddf_constant_and_squarefree` | `[3]`, 5 gives `[]`; `[1,1,0,0,1,1]`, 5 (squarefree, gp above) gives `[(1,[1,1]), (2,[1,0,0,0,1])]` (gp `factormod(x^5+x^4+x+1,5)`: x+1 and x^4+1 = (x^2+2)(x^2+3)) | passes on HEAD too |
| `ddf_reads_coefficients_mod_p` | `[-4,1]`, 5 (monic, unreduced) | `[(1,[1,1])]` (gp `factormod(x-4,5)` = `x+1`) | fails (HEAD `[(1,[-4,1])]`, `$X/rev1/r1_head.rs`) |

---

## B. Prime checks: `fp_factor::factor` and `unramified_local_galois_group`

### B.1 Defects at HEAD

1. `fp_factor::factor` (`fp_factor.rs:735`) documents "p a prime <= MAX_P" (`:721-724`) and checks
   nothing. `factor(&[1,0,1], 15) = [[1,0,1]]`; `factor(x^2+1, 21)` and `factor(x^2-1, 15)` panic inside
   `div_mod` ("leading coefficient not invertible", `$X/t1_composite.rs`). The same holds for
   `factor_with_multiplicity` (`:761`) and `squarefree_factor` (`:304`).
2. `factor` echoes unreduced coefficients of a monic input: `factor([-3,0,1], 5) = [[-3,0,1]]` while
   `factor([2,0,1], 5) = [[2,0,1]]` (`$X/t6_unreduced.rs`). The module header asks for reduced input, but
   no entry enforces it; with a sorted output (item D) the output must be canonical.
3. `unramified_local_galois_group` (`rustmath-groups/src/pgalois.rs:122-156`), `$X/t2_pgal.rs`:

   | Call | HEAD | Reference |
   |---|---|---|
   | `x^2+1`, p = 15 | `Some { [2], order 2, transitive: true }` | meaningless; mod 3 the type is `[2]`, mod 5 it is `[1,1]` (gp `factormod(x^2+1,3)`, `factormod(x^2+1,5)`) |
   | `x^2+1`, p = 1 | `None` ("ramified") | p = 1 is not a prime |
   | `x^2+1`, p = 0 | panic "attempt to divide by zero" | |
   | `local_galois_candidates(x^5-x-1, 15)` | panic in `div_mod` | |
   | `2x^2+x+1`, p = 2 | `[1]`, order 1, not transitive | `[1, 1]`: gp `poldisc(2*x^2+x+1) = -7`, `factorpadic(2*x^2+x+1, 2, 8)` has degrees `[1, 1]` |
   | `2x^3+x^2+x+1`, p = 2 | `[2]` | `[1, 2]`: `poldisc = -83`, `factorpadic(...,2,8)` degrees `[1, 2]` |
   | `3x^4+x^3+2x+1`, p = 3 | `[3]` | `[1, 3]`: `poldisc = -563`, `factorpadic(...,3,8)` degrees `[1, 3]` |
   | `[1,0,1,0]` (x^2+1 untrimmed), p = 7 | `[2]`, transitive **false** | `[2]`, transitive true (n = 2, gp `factormod(x^2+1,7)` irreducible) |
   | `6x+6` at 3; `2x` at 2; `-2x+4` at 2 | `[]`, order 1, not transitive | `[1]`, order 1, transitive: the root is rational (`poldisc` = 1 for all three; `factorpadic(f,p,8)` has one factor of degree 1) |
   | `3x^2+3`, p = 3 | `None` ("ramified") | `[2]`, order 2, transitive: `poldisc(3x^2+3) = -36`, but the primitive part `x^2+1` has `poldisc = -4` and is irreducible mod 3; `factorpadic(3*x^2+3,3,8)` degrees `[2]` |
   | `local_galois_candidates(3(x^4+x+1), 3)` | `(["4T2","4T4"], false)` | `([], true)` as for `x^4+x+1`: `poldisc(x^4+x+1) = 229`, `factormod(x^4+x+1,3)` degrees `[1,3]` (intransitive cyclic group; neither 4T2 nor 4T4 is the local group) |
   | `local_galois_candidates(2(x^5-x-1), 2)` | `(["5T3","5T5"], false)` | `([], true)`: `poldisc(x^5-x-1) = 2869 = 19·151`, `factormod(x^5-x-1,2)` degrees `[2,3]` |

   The non-monic and untrimmed cases come from `n = fbar.len() - 1` (`:152`) and from dropping the root
   at infinity when p | lc(f). The content cases come from testing `p | disc(f)` on f itself: for n >= 2,
   p | content(f) forces p^(2n-2) | disc(f) (`disc(c·g) = c^(2n-2) disc(g)`), so HEAD calls the prime
   "ramified" although the roots of f are those of its primitive part; for n = 1, disc = 1 and `f mod p = 0`.
   gp: `$X/pgal.gp`, `$X/rev1/pgal_rev1.gp`; HEAD values: `$X/t2_pgal.rs`, `$X/rev1/r1_head.rs`.

### B.2 The Miller-Rabin bound (verified)

The HANDOFF says strong tests to the bases 2, 3, 5, 7 are exact for p <= MAX_P.

- gp (`$X/psw.gp`): `factor(3215031751) = 151 · 751 · 28351`; it is a strong probable prime to 2, 3, 5
  and 7, not to 11. `sqrtint(2^63-1) = 3037000499 = MAX_P < 3215031751`.
- Exhaustive check (`$X/psw_scan.rs`, release build, sieve of Eratosthenes on the odd numbers, then strong
  tests on every odd composite): below 3215031752 there are 2045 strong pseudoprimes to base 2, and the
  only one that is also a strong pseudoprime to 3, 5 and 7 is 3215031751 (`psw_scan.log`). Below
  3037000500 (= MAX_P + 1) there are 1998 and none passes all four (`psw_scan_maxp.log`). The program is
  validated by the published count of strong base-2 pseudoprimes below 10^9, 1282 (Pomerance, Selfridge,
  Wagstaff, *The pseudoprimes to 25·10^9*, Math. Comp. 35 (1980) 1003-1026): `psw_scan_1e9.log` gives 1282.
  The least strong pseudoprime to 2, 3, 5 is 25326001 = 2251 · 11251 (gp: strong to 2, 3, 5, not 7).

So a deterministic 4-base test would be exact on [2, MAX_P]. It is **not** needed as new code: I1 makes
the word-size test public (`rustmath_integers::prime::is_prime_u64`, I1 design §2.1, exact for every
u64). A private Miller-Rabin in `fp_factor` would duplicate it (CLAUDE.md: consume the existing crates).

### B.3 Fix in `fp_factor`

```rust
/// `Ok(())` iff `p` is a prime <= [`MAX_P`] (the module's exact-arithmetic bound); otherwise
/// `Err("p = {p} is not a prime <= 3037000499")`. Exact: `is_prime_u64` (I1) is a proof for every u64.
pub fn check_modulus(p: i64) -> Result<(), String>
```

It replaces the private `check_prime` (`:504-510`, same message, so T5 keeps its expectations; I1 rewrites
that function's doc comment, and P1 replaces the function and its doc). Body:
`(2..=MAX_P).contains(&p) && rustmath_integers::prime::is_prime_u64(p as u64)`.
P1 is rebased on I1, which makes `is_prime_u64` public (I1 §2.1). The fallback in case it is missing is in
the Risks section. HEAD's `Integer::is_prime` costs 2.3 to 2.7 ms per call on primes above 1000
(`$X/t5_isprime.rs`, debug), which is why the word-size test is used.

Entry points (every one checks p first, then reads the coefficients modulo p with `reduce`):

| Function | New behaviour on bad p | Other change |
|---|---|---|
| `factor(f, p)` | panic `"fp_factor::factor: {why}"` | reduces f; output in PARI order (item D) |
| `factor_with_multiplicity(f, p)` | panic `"fp_factor::factor_with_multiplicity: {why}"` | reduces f; order of `factor` |
| `squarefree_factor(f, p)` | panic `"fp_factor::squarefree_factor: {why}"` | reduces f; the recursion moves to a private `fn radical(f_m, p)` (HEAD body) |
| `distinct_degree_factor(f, p)` | item A | |
| `equal_degree_factor(f, d, p)` | unchanged (already checks) | reduces f; PARI order |
| `pub(crate) fn factor_prime(f, p)` | **no check**: the caller has proved p prime <= MAX_P | the body of `factor`; used by `zassenhaus::choose_prime` (its p comes from trial division, a proof, and is <= MAX_P by the loop bound) |

The arithmetic helpers (`add`, `sub`, `mul`, `div_mod`, `gcd`, `extended_gcd`, `pow_mod`, `make_monic`,
`mod_inv`, `derivative_of`, `trim`, `degree`, `is_zero`) stay unchecked: they run in inner loops of
`zp_hensel`, `sieve`, `maclane`, `ideals`. The module header (`:33-42`) is rewritten to say exactly this.

### B.4 Fix in `pgalois`

MAGMA ch. 23 §23.8.1 (`Factorization` over F_p) and the p-adic Galois chapter digest cited in the file
header.

Theory. f and its primitive part `g = f / content(f)` have the same roots, so they have the same local Galois
group, and the module works on g. Let n = deg g >= 1 and write `G(X, Y) = Σ a_i X^i Y^(n-i)` for the
coefficients `a_i` of g. Then `G mod p` is a nonzero binary form of degree n, because p ∤ content(g).

- n >= 2. The discriminant of G as a binary form, `a_n^(2n-2) ∏_{i<j} (α_i - α_j)^2`, is a universal
  integer polynomial in the `a_i`. Up to sign it is `disc(g)`: MAGMA's definition (digest §23.8.2,
  `D = c_n^(2n−2) · ∏_{i≠j}(αᵢ − αⱼ)`) differs from it by `(−1)^(n(n−1)/2)`, and RustMath's
  `disc::discriminant` (`disc.rs:199-213`, `(−1)^(n(n−1)/2) Res(g, g')/lc(g)`) is equal to it. Only
  divisibility by p matters here. Reduction mod p commutes with the polynomial. So if p ∤ disc(g),
  then `G mod p` has n distinct roots on P^1 over F̄_p. The root at infinity has multiplicity
  n - deg(g mod p), so this is at most 1.
- n = 1. disc(g) = 1, and `G mod p` is a nonzero linear form. It has one root on P^1, so
  deg(g mod p) ∈ {0, 1} = {n - 1, n}.

In both cases deg(g mod p) >= n - 1, and `V(G) ⊂ P^1_{Z_p}` is finite étale of degree n over Z_p (the
special fibre is reduced of degree n). Its generic fibre is `Spec Q_p[x]/(g)`, since `a_n ≠ 0`. So the
splitting field is unramified, and Frobenius permutes the n roots with the cycle type given by the closed
points of the special fibre. These are the factor degrees of `g mod p`, plus one fixed point when
deg(g mod p) = n - 1. That root lies in Q_p: Hensel on the reversed polynomial, whose reduction has a
simple root at 0. Review 1 found that the earlier version of this argument, on f instead of g, fails for
n = 1 with p | content(f). The revised rule was checked with gp on 28,213 random pairs (f, p) with
p ∤ disc(g), 2,231 of them with p | content(f) and 7,356 linear (`$X/rev1/pgrand_pp.gp`). It matched
`factorpadic(f, p, 10)` in every case, and `deg(g mod p) >= n - 1` held in every case.

```rust
/// Local Galois group at a prime p that does not divide disc(g), where g is the primitive part of f.
/// Returns `None` iff p | disc(g) (ramified or an index divisor; not handled here).
/// # Panics
/// `p` is not a prime <= `fp_factor::MAX_P` (`fp_factor::check_modulus`); `f` has degree < 1 after trimming.
pub fn unramified_local_galois_group(f: &[Integer], p: i64) -> Option<UnramifiedLocalGalois>
```

Steps:
1. `g = zx::primitive_part(&zx::trim(f))` (`zx.rs:135`; it trims, keeps the sign, divides by the content).
2. `n = zx::degree(&g)`. Panic `"pgalois::unramified_local_galois_group: f must have degree >= 1"` if n < 1.
3. `fp_factor::check_modulus(p)`. On `Err(why)` panic `"pgalois::unramified_local_galois_group: {why}"`.
4. If `disc(g) % p == 0`, return `None`.
5. `gbar` = g mod p in [0, p), trimmed. `nbar = degree(gbar)`, which is `>= 0` because p ∤ content(g).
6. `degs` = degrees of `fp_factor::factor(&gbar, p)` (empty when nbar = 0), plus `1` when `nbar == n - 1`.
   `nbar < n - 1` is `unreachable!("pgalois: deg(g mod p) < deg g - 1 with p ∤ content(g)·disc(g)")`, by the
   argument above.
7. Sort `degs`, `order = lcm`, `transitive = degs == [n]`.

The doc comment of `UnramifiedLocalGalois` (`:113-116`) changes "Returns `None` if `p | disc(f)`" to
"... if p divides the discriminant of the primitive part of f".

`local_galois_candidates` (`:165`):
- Step 1 and the modulus check first. `n = degree(g)`.
- `n < 1`, or n has no table (`transitive_groups(n)` empty): return `([], false)`, as at HEAD. This is a
  "no claim" answer. A constant f keeps this answer (no new panic), and the empty slice, which underflows
  at HEAD (`f.len() - 1`), gets it too.
- Otherwise call `unramified_local_galois_group(&g, p)`, and in the ramified branch use
  `is_padic_square(&disc(g), p)`. The square class is that of disc(f): `disc(f) = c^(2n-2) disc(g)`, and
  c^(2n-2) is a nonzero square.

`is_padic_square` (`:202`): panic `"pgalois::is_padic_square: p = {p} is not a prime"` unless
`p >= 2 && rustmath_integers::prime::is_prime_u64(p as u64)`. This is exact for every i64. There is no
MAX_P limit, since it uses `Integer` arithmetic.

### B.5 Callers and behaviour changes

- `fp_factor::factor` (grep `fp_factor::factor(`): `zassenhaus.rs:117` (switches to `factor_prime`),
  `padic_factor.rs:76, :145`, `rustmath-numberfields/src/{ideals.rs:118, panayi.rs:374, local_field.rs:42}`,
  `rustmath-curves/src/igp_closure.rs:76`, `rustmath-galois/src/galois_ctx.rs:451`,
  `rustmath-groups/src/pgalois.rs:135`, `rustmath-rings/src/function_field/typed/factor.rs:171`. Every one
  passes a prime <= MAX_P (rings checks `modulus_is_prime` and `P <= MAX_P` before; numberfields, galois,
  curves iterate over primes), so the only change they see is the order (item D). Several are themselves
  public entries with a caller-supplied p. They now panic on a composite p or p > MAX_P inside
  `fp_factor::factor`, where before they returned a value computed from a meaningless factorization (or hit
  a `div_mod` panic):
  - `rustmath_galois::galois_ctx::build_ctx(f, p, k)` (`:425-451`, "assumes p is good");
  - numberfields `ideals::prime_decomposition(f, p)` (`:126`, through `factor_with_mult`, `:118`),
    `ideals::prime_ideals(f, p)` (`:350`), `local_field::local_decomposition(f, p)` (`:114`, through
    `prime_decomposition`), and `panayi::quartic_local_galois_group(f, p)` (`:414`, through
    `irreducible_mod_p`, `:356-374`).

  Their callers in the workspace pass primes: they iterate over primes, factor a norm, or use the literal
  primes 2, 3, 5, 7 in tests (review 1 grepped the literals). So no current output changes.
- `padic_factor::{cycle_type, padic_factor, ramification_type}` (public, user p): check p first and panic
  with their own prefix, instead of a `div_mod` panic inside `fp_factor::gcd`. Callers:
  `rustmath-groups/src/galois_resolvent.rs:211`, `rustmath-galois/src/{deg24.rs, descent24.rs}`,
  `rustmath-curves/src/belyi/audit.rs`, `rustmath-igp24/src/main.rs`, `rustmath-numberfields/src/ideals.rs:152`,
  tests. All library callers pass primes (`first_primes`, `small_primes`, constant lists such as
  `audit.rs:640`, `regular_certify.rs:722`). The one exception is the igp24 command line: `main.rs:266-275`
  parses a user prime list with `filter_map(parse)` and passes it to `cycle_type` / `ramification_type`
  (`:137-160`). A composite entry gave a meaningless cycle type (or a `div_mod` panic) and would now panic;
  the CLI validates the list at parse time with `fp_factor::check_modulus` and exits with
  `"igp24: {why}"` (status 2) naming the entry.
- pgalois callers: only `rustmath-groups/tests/pgalois.rs` (primes 7, 11, 13, 19, 23, 47 and 2 on monic,
  trimmed, primitive inputs): unchanged values. Behaviour changes of the pgalois entries themselves:
  - f with p | content(f) is now handled through its primitive part. Linear f gets `[1]`, transitive
    (HEAD `[]`, not transitive). For n >= 2, a prime that divides only the content is no longer "ramified":
    `3x^2+3` at 3 gives `Some([2])` (HEAD `None`), and `local_galois_candidates(3(x^4+x+1), 3)` gives
    `([], true)` (HEAD returned the wrong candidates `["4T2","4T4"]`).
  - `unramified_local_galois_group` on a constant f now panics ("f must have degree >= 1"). At HEAD it
    returned `None` (disc 0, so "ramified"), a claim that means nothing for a constant.
    `local_galois_candidates` on a constant keeps `([], false)`.
  - A composite p, p = 0, p = 1 or p > MAX_P now panics in both entries and in `is_padic_square`. The
    check also runs for degrees with no table, where HEAD returned `([], false)` without looking at p.

### B.6 Tests

`fp_factor.rs`:

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `factor_rejects_composite_p` | `[1,0,1]`, 15 | `should_panic(expected = "fp_factor::factor: p = 15 is not a prime <= 3037000499")` | fails (returns `[[1,0,1]]`) |
| `factor_rejects_p_one_and_large_prime` | `[1,0,1]`, 1; `[1,1]`, 3037000507 (gp `nextprime(3037000500)`) | panics, same message pattern (two tests) | fail |
| `factor_with_multiplicity_rejects_composite_p` | `[1,0,1]`, 21 | `"fp_factor::factor_with_multiplicity: p = 21 ..."` | fails (div_mod panic, other message) |
| `squarefree_factor_rejects_composite_p` | `[1,0,1]`, 15 | `"fp_factor::squarefree_factor: p = 15 ..."` | fails |
| `factor_reads_coefficients_mod_p` | `[-3,0,1]` and `[7,0,6]`, 5 | both `[[2,0,1]]` (gp `factormod(x^2-3,5)`: irreducible, monic `x^2+2`) | fails for `[-3,0,1]`; `[7,0,6]` passes on HEAD (`make_monic` reduces when lc ≢ 1) and stays as a guard |
| `factor_reduces_leading_coefficient` | `[1,1,5]`, 5 (`5x^2+x+1`); `[1,5]`, 5 (`5x+1`) | `[[1,1]]` (gp `factormod(5*x^2+x+1,5)` = `x+1`); `[]` (gp `factormod(5*x+1,5)` is empty: a nonzero constant) | fails: HEAD panics in `div_mod` ("leading coefficient not invertible") and in `pth_root` (`$X/rev1/r1_head.rs`) |
| `factor_with_multiplicity_reduces_leading_coefficient` | `[1,1,5]`, 5 | `[([1,1], 1)]` | fails (HEAD panics in `div_mod`) |
| `equal_degree_factor_rejects_zero_mod_p` | `[5]`, d = 1, 5 | `should_panic(expected = "fp_factor::equal_degree_factor: f = 0 has no factorization")` (gp `Mod(5,5)` = 0) | fails (HEAD returns `[]`) |
| `check_modulus_boundary` | Ok: 2, 3037000493 (`precprime(3037000499)`). Err: 0, 1, 4, 15, -7; the strong pseudoprimes 2047 = 23·89 (strong to 2), 1373653 = 829·1657 (strong to 2, 3), 25326001 = 2251·11251 (strong to 2, 3, 5); the range checks 3037000507 (`nextprime(3037000500)`) and 3215031751 (strong to 2, 3, 5, 7, but above MAX_P) | gp `$X/rev1/rev1.gp`: `isprime` 0 for the three; strong to the bases [2,3,5,7] = [1,0,0,0], [1,1,0,0], [1,1,1,0] | new API |

`rustmath-groups/tests/pgalois.rs` (new tests):

| Name | Input | Expected (gp) | HEAD |
|---|---|---|---|
| `unramified_rejects_composite_p` | `x^2+1`, 15 | `should_panic(expected = "pgalois::unramified_local_galois_group: p = 15 is not a prime")` | fails |
| `unramified_rejects_p_zero_and_one` | p = 0, p = 1 | panics with "is not a prime" | fails (division by zero / `None`) |
| `candidates_reject_composite_p` | `x^5-x-1`, 15 | panic "is not a prime" | fails (other message) |
| `unramified_non_monic_adds_root_at_infinity` | `2x^2+x+1` at 2; `2x^3+x^2+x+1` at 2; `3x^4+x^3+2x+1` at 3; `2x+1` at 2 | `[1,1]` order 1; `[1,2]` order 2; `[1,3]` order 3; `[1]` order 1 transitive | fails on the first three (`[1]`, `[2]`, `[3]`) and the last (not transitive) |
| `unramified_untrimmed_input` | `[1,0,1,0]` at 7 | `[2]`, order 2, transitive | fails (transitive false) |
| `unramified_linear_with_content` | `6x+6` at 3; `2x` at 2; `-2x+4` at 2 | each `[1]`, order 1, transitive (gp `poldisc` 1; `factorpadic(f,p,8)` one factor of degree 1) | fails (HEAD `[]`, not transitive) |
| `unramified_uses_primitive_part` | `3x^2+3` at 3 | `[2]`, order 2, transitive (gp `poldisc(x^2+1) = -4`, `factorpadic(3*x^2+3,3,8)` degrees `[2]`) | fails (HEAD `None`) |
| `candidates_use_primitive_part` | `3(x^4+x+1)` at 3; `2(x^5-x-1)` at 2; `2(x^5-x-1)` at 3 | `([], true)`; `([], true)`; `(["5T1"], true)` (gp: `poldisc` 229 and 2869; `factormod` degrees `[1,3]`, `[2,3]`, `[5]`; `factorpadic(f,p,8)` the same degrees) | fails on the first two (HEAD `(["4T2","4T4"], false)` and `(["5T3","5T5"], false)`); the third passes on HEAD (3 ∤ 2) |
| `candidates_constant_makes_no_claim` | `[5]` at 3; `[]` at 3 | `([], false)` both | `[5]` passes on HEAD; `[]` fails (HEAD panics: subtract with overflow) |
| `unramified_rejects_constant` | `[5]` at 3 | `should_panic(expected = "f must have degree >= 1")` | fails (HEAD `None`) |
| `is_padic_square_rejects_composite` | `is_padic_square(4, 15)` | panic "is not a prime" | fails |

gp commands: `$X/pgal.gp` and `$X/rev1/pgal_rev1.gp` (`poldisc`, `factorpadic(f, p, 8)`, `factormod`).

### B.7 Berlekamp's prime check (I1 hand-off, accepted)

I1 §7 hands P1 this item: `berlekamp::check_prime` (`berlekamp.rs:73-80`) can map an unresolved primality
proof to `Err` with `try_is_prime`. I accept it. After I1, `Integer::is_prime` is a proof, and it panics
when ECPP finds no certificate (I1 §2.4; only above 2^64, and in I1's experiments only above 512 bits).
`berlekamp::factor`, `squarefree_decomposition`, `q_matrix` and `is_irreducible` return `Result`, so such a
panic would break their contract. Change:

```rust
/// `Ok(())` iff `p` is a proven prime (I1's `try_is_prime`: a proof for every size).
/// `p < 2` or `p` composite: `Err(InvalidArgument("berlekamp: the modulus {p} is not a prime"))` (text unchanged).
/// Proof unresolved: `Err(NotSupported("berlekamp: the primality of the modulus {p} is unresolved ({e})"))`.
fn check_prime(p: &Integer) -> Result<()>
```

After I1 no composite modulus passes the check, so two error texts become false and are rewritten:
- `internal` (`:65-71`): "berlekamp: internal check failed ({what}); expected only for a composite modulus
  that passed the primality test" becomes "berlekamp: internal check failed ({what}); the modulus is a
  proven prime, so this is a bug". The error kind (`InvalidOperation`) is unchanged.
- `not_prime` (`:59-63`, used where a nonzero leading coefficient has no inverse, `:127`, `:157`, `:409`):
  "berlekamp: {p} is not prime (a leading coefficient has no inverse modulo {p})" becomes
  `internal("a nonzero coefficient has no inverse modulo the proven prime {p}")`.

The docs at `:260-266` and `:337-341` (which I1 rewrites to drop the 3.19e23 caveat) get one sentence: "An
unresolved primality proof of `p` gives `Err(NotSupported)`." No test or caller matches these texts (grep of
the workspace: `berlekamp::` is used only inside rustmath-polynomials, and no test pins these messages).

Tests (`tests/ch23_berlekamp.rs`):
- `berlekamp_rejects_strong_pseudoprime_psi12`: `berlekamp::factor(x^2+1, 318665857834031151167461)` and
  `berlekamp::is_irreducible` of the same input are `Err(InvalidArgument)` with "is not a prime". The
  modulus is psi_12, the strong pseudoprime to the bases 2..37 (gp `isprime(318665857834031151167461)` = 0,
  `factor` = 399165290221 · 798330580441). On HEAD `is_prime(psi_12)` is true, and Berlekamp returns
  `Ok` with 2 "factors", and `is_irreducible` returns `Ok(false)` (`$X/rev1/r2_psi12.rs`). So the test fails
  on HEAD and passes once I1 is in: it pins I1's semantics at the Berlekamp level.
- No test for the unresolved branch. Whether ECPP finds a certificate depends on I1's class-polynomial
  table, and I1 pins no unresolved prime. I1's own `try_is_prime` tests cover the `Err` it returns. If I1's
  final tests do pin such a prime, add `berlekamp_unresolved_modulus_is_err` with it.

---

## C. Equal-degree cost and the rings callers

### C.1 Defects at HEAD

- `block_contract` (`fp_factor.rs:546-553`) stores `ladder[0..=d]`, d + 1 polynomials of degree < n, but
  reads only `ladder[d]`, `ladder[0]` and `ladder[d/l]` for the primes l | d (`:554`, `:571`). For a block
  with d = n = 1000 that is 1001 · 1000 · 8 bytes ≈ 8 MB where 2 + ω(d) rungs suffice.
- `rustmath-rings/src/valuation/maclane.rs:1687-1694`: every piece of `fp_factor::equal_degree_factor(&gd,
  d, p)` is tested again with `is_irreducible_fp` (Rabin, `:1579`), although the public entry has already
  certified the block (`fp_factor.rs:700-703`): a divisor of degree d of a certified block is
  irreducible. The comment block `:1562-1574` and `residue_tower.rs:19-22` still say that
  `fp_factor::factor` is wrong on inseparable inputs; that was fixed in `5ff8749`
  (`test_factor_inseparable_gf2`, `_gf3`, `_gf5`).
- `rustmath-rings/src/function_field/typed/factor.rs:155-161` writes the literal `3_037_000_499`.

### C.2 Fix

`block_contract`: compute the prime divisors of d first (private `fn prime_divisors(d: i64) -> Vec<i64>`,
trial division, increasing), `wanted = sorted { d / l }`, then run the d Frobenius steps
`X_k = X_(k-1)^p rem f` keeping `X_0`, the current rung and the rungs in `wanted` only. Checks and messages
in HEAD order: `X_d != X_0` → `"f has an irreducible factor whose degree does not divide {d}"`; then for l
increasing, `gcd(X_(d/l) - X_0, f) != 1` → `"f has an irreducible factor of degree dividing {d/l}"`. Time is
unchanged (d Frobenius steps); memory O((ω(d) + 2) n). The prototype (`$X/proto/driver.rs` item 4) gives
identical results and messages to HEAD on 12 inputs, including d = 6 with factors of degree 2 and 4.

maclane: delete the `!is_irreducible_fp(&q, p)?` clause at `:1689` (keep `degree(&q) != d`); the
function `is_irreducible_fp` stays (used at `:1257`). Rewrite the doc of `factor_fp_certified` (`:1642-1646`)
to "irreducibility of each piece is certified by `fp_factor::equal_degree_factor` (Rabin's criterion for
products, checked on the block)". Rewrite the comment block `:1562-1574` and `residue_tower.rs:19-22`:
`fp_factor::factor` handles inseparable inputs since `5ff8749`; this module keeps its own pipeline for
its multiplicities and its reconstruction certificate.

typed/factor.rs: `if P > rustmath_polynomials::fp_factor::MAX_P as u64` and the comment "fp_factor's i64
arithmetic is exact for p <= fp_factor::MAX_P". Behaviour unchanged (same number).

### C.3 Callers, cost

- `block_contract` is private: callers `equal_degree_contract` (lazy certificate) and `equal_degree_factor`.
- maclane change: removes one Rabin test (d Frobenius steps on a degree-d polynomial plus ω(d) gcds) per
  piece; results unchanged.

### C.4 Tests

| Name | Input | Expected (gp) |
|---|---|---|
| `certificate_composite_d_accepts` | `(x^12+x^3+1)(x^12+x^6+x^4+x+1)` over F_2, d = 12: `[1,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1]` | `[x^12+x^3+1, x^12+x^6+x^4+x+1]` in this order (gp `factormod(...,2)`; `polisirreducible` 1 for both) |
| `certificate_composite_d_rejects_divisor_degree` | `(x^2+x+1)(x^4+x+1)(x^6+x+1)` over F_2, d = 12: `[1,1,0,1,0,0,1,1,0,1,1,1,1]` | panic `"degree dividing 6"` |
| `certificate_d6_two_cubics` | A2 row `[1,1,1,1,1,1,1]`, d = 6, p = 2 | panic `"degree dividing 3"` |
| `certificate_d6_irreducible` | `x^6+x+1` over F_2, d = 6 | `[[1,1,0,0,0,0,1]]` (gp `polisirreducible(Mod(1,2)*(x^6+x+1)) = 1`) |
| `prime_divisors_small` | 1, 12, 60, 97 | `[]`, `[2,3]`, `[2,3,5]`, `[97]` |

All pass on HEAD as well except `prime_divisors_small` (new helper); they guard the rewrite. The maclane
tests (`cargo test -p rustmath-rings valuation`) must pass unchanged.

---

## D. Drop phase 1 and sort `factor`'s output

### D.1 State at HEAD

`edf_rec` (`fp_factor.rs:595-652`) runs up to 200 legacy tries (monic samples of degree n - 1, biased
draw), then the certificate, then 256 uniform vzGG 14.8 tries. Phase 1 exists only to keep HEAD's
discovery order, which `factor_order_is_unchanged` (`:1188`) pins. `factor` returns that order.

### D.2 Fix

- Delete `LEGACY_TRIES`, `random_monic_polynomial`, `SplitMix64::below_legacy` and the phase-1 loop.
- `edf_rec(f, d, p, out)` (no `legacy_tries`): up to `LAS_VEGAS_TRIES = 256` uniform tries of
  `split_with` (vzGG Algorithm 14.8 with step 3); before try number `TRIES_BEFORE_CERTIFICATE = 32` it runs
  `contract_or_panic(f, d, p)` (lazy certificate: on a valid block 32 consecutive failures have probability
  <= 2^-32, vzGG Theorem 14.9). No split in 256 tries: panic as now. `n == d` pushes f; `d < 1 || n < d`
  as now.
- New `pub fn factor_order(a: &[i64], b: &[i64]) -> std::cmp::Ordering`: degree, then coefficients from the
  top down (the order of `berlekamp::factor_order`, `berlekamp.rs:205`, on i64).
- `factor` / `factor_prime` sort their output with `factor_order`; `equal_degree_factor` sorts its output;
  `factor_with_multiplicity` inherits the order of `factor`.
- `zassenhaus::factor` sorts its `(g, e)` list by `g` with the same rule on `Integer` (degree, then
  coefficients from the top down, signed comparison; `Integer` is `Ord`).

This is PARI's order on both sides. Evidence: `$X/order.gp` checks `apply(le, factormod(f,p)[,1]~)` against
the sort on all 16 reference rows of `fp_factor.rs` and 3 more (all equal), and `$X/order_z.gp` checks
`factor(f)[,1]` over Z on 12 polynomials (all equal). HEAD's Zassenhaus order differs from it on 9 of
those 12 (`$X/t4_zorder.rs`: for example `(x^2+x+1)(x^2-x+1)(x^2+3)(x^2-3)` comes out as
`x^2+x+1, x^2+3, x^2-x+1, x^2-3`, and `(x-1)^3 (x+1)^2 x` as `x, x+1, x-1`). MAGMA fixes no order (V2.19
handbook §23.8.1 `Factorization`: "each factor is normalized"); PARI's order is the documented reference
here.

Prototype (`$X/proto/fp_new_patched.rs`, driver `$X/proto/driver.rs`): factor sets and multiplicities
equal HEAD's on 600 random products (primes 2, 3, 5, 7, 11, 13, 101, 65537, 2^31 - 1, 3037000493, up to 5
factors of degree <= 6 with powers <= 3), the new output is already in PARI order, and the 18 rows match
PARI. Timing (debug, same machine, `$X/proto/driver.rs` item 5):

| Workload | HEAD | new |
|---|---|---|
| 20 random monic, degree 120, p = 2 | 366 ms | 296 ms |
| 20 random monic, degree 80, p = 3 | 242 ms | 223 ms |
| 20 random monic, degree 60, p = 5 | 155 ms | 159 ms |
| 10 random monic, degree 40, p = 2^31 - 1 | 407 ms | 384 ms |
| 10 random monic, degree 40, p = 3037000493 | 334 ms | 355 ms |

The two columns are comparable within measurement noise. Review 1 reran the driver and saw spreads of up to
about ±40 % between runs, in both directions (degree 60 mod 5: HEAD 150/144/154 ms, new 207/138/151 ms;
degree 40 mod 2^31 - 1: HEAD 500/475/414 ms, new 413/398/599 ms). So the table shows no slowdown, and no
speed-up either. The prototype's prime check is a 4-base Miller-Rabin. With I1's `is_prime_u64` the check
costs about 10 µs per call (I1 §2.2).

### D.3 Downstream order audit

Where the order of `fp_factor::factor`, `equal_degree_factor`, `zassenhaus::factor` flows (grep of the
workspace for `fp_factor::factor`, `equal_degree_factor`, `zassenhaus::factor`, `factor_over_integers`,
`padic_factor(`, `factor_with_mult`, `prime_ideals(`, `factor_poly`, `factor_monic_rational_certified`,
`factor_fp_certified`, `mac_lane_approximants`, then the tests of each consumer):

| Consumer | Where the order goes | Tests | Verdict |
|---|---|---|---|
| `fp_factor.rs` tests | `factor_order_is_unchanged` (`:1188`) pins HEAD order; `phase_two_alone_matches_pari` (`:1278`) passes `legacy_tries = 0`; `edf_rec_certificate_panics_on_bad_block` (`:1433`) | rewritten (D.4) | **change** |
| other `fp_factor.rs`, `zassenhaus.rs`, `padic_factor.rs`, `factorization.rs` tests | sort, `contains`, lengths, single factors | | unaffected |
| `tests/ch23_berlekamp.rs:635` | sorts `fp_factor` output before comparing | strengthen: assert it is already sorted | unaffected |
| `tests/lmfdb_degree24.rs`, `tests/lmfdb_local.rs` | degree filters, sums, sorted `ramification_type` | | unaffected |
| `resolvent.rs:359` | sorted degrees | | unaffected |
| numberfields `ideals.rs` `factor_with_mult` → `prime_decomposition` (`ef()` sorts), `prime_ideals` order | `ideals.rs` tests use products, `ef()`, valuations against "the conjugate prime"; `s_units.rs:90`, `s_unit_gens.rs:504` take the first prime above p, and every assertion there holds for either prime (norms, ranks, `is_s_unit` with the same prime, both primes above 29 in Q(√-5) principal, both above 2 in Q(√-23) of order 3); class group, ray class, Artin: factor base order, assertions on invariants and on classes of named ideals | unaffected |
| numberfields `panayi.rs:374`, `local_field.rs:42` | factor count | | unaffected |
| curves `igp_closure.rs:76` | sorts | | unaffected |
| curves `belyi/regular_certify.rs:214`, `audit.rs:390` (Zassenhaus) | counts, multiplicities | | unaffected (error type: item E) |
| galois `galois_ctx.rs:451` | sorts | | unaffected |
| galois `cayley.rs:314`, `quintic.rs:251` | sorted degrees | | unaffected |
| galois `lowdegree.rs:64` `monic_integer_roots` | `roots.len()` | | unaffected |
| galois `lib.rs:180` → `reducible::classify_composite` | notes print the factor degrees and discriminants in factor order (`reducible.rs:66`, `:86-91`); no test matches those strings (grep `degrees [`, `discriminants [`) | | unaffected; note text order changes |
| groups `pgalois.rs:135`, `galois_resolvent.rs:159, :211` | sorted degrees, counts | | unaffected |
| rings `typed/factor.rs:118, :171` → `place.rs:102`, `divisor.rs:123` | tests use `any`, lengths, single-factor equality; divisors are maps | | unaffected |
| rings `maclane.rs:1687` → `mac_lane_step` children order | `maclane.rs` tests: one-leaf cases or loops over all leaves (`:2590-2601` asserts the two keys differ) | | unaffected |
| modular `decomposition.rs:247` | summands sorted by (dimension, charpoly key) (`:505-516`); `Mixed` lists sorted in tests (`:897`, `:949`, `degeneracy.rs:2092-2111`, `:2152-2160`) | | unaffected |
| functionfields `factor.rs:69`, `genus.rs:223` | lengths, multiplicities, degree-1 factor (`lib.rs:105`) | | unaffected |
| symbolic `factor.rs:92`, `risch.rs:808` | expression structure tests (`matches!`), single log terms | | unaffected |
| igp24 `main.rs:119` | JSON output order (no tests) | | output order changes (canonical now) |

So the only tests that change are the three in `fp_factor.rs`. The acceptance run (Plan) runs every crate
in this table; any further failure must be analysed: if the test asserts a set, make it order-free; if a
library contract promises an order, sort in the library.

### D.4 Tests

- Replace `factor_order_is_unchanged` by `factor_order_is_pari_order`: for every reference row, `factor(f, p)`
  and `equal_degree_factor(f, d, p)` equal the PARI list below **unsorted** (gp, `$X/order.gp`:
  `le(g) = my(v = Vec(lift(g))); vector(#v, i, v[#v + 1 - i]); apply(le, factormod(f, p)[,1]~)`):

  | Row | PARI order (little-endian) |
  |---|---|
  | A1 | `[0,1] [1,1]` |
  | A2 | `[1,1,0,1] [1,0,1,1]` |
  | A3 | `[1,1,0,0,1] [1,0,0,1,1] [1,1,1,1,1]` |
  | A4 | `[1,0,1,0,0,1] [1,0,0,1,0,1] [1,1,1,1,0,1] [1,1,1,0,1,1] [1,1,0,1,1,1] [1,0,1,1,1,1]` |
  | B1 | `[0,1] [1,1] [2,1]` |
  | B2 | `[1,0,1] [2,1,1] [2,2,1]` |
  | B3 | `[1,2,0,1] [2,2,0,1]` |
  | C1 | `[0,1] [1,1] [2,1] [3,1] [4,1]` |
  | C2 | `[2,0,1] [3,0,1]` |
  | C3 | `[1,1,0,1] [4,1,0,1] [1,2,0,1]` |
  | D1 | `[2147483642,1] ... [2147483646,1]` (increasing) |
  | D2 | `[1,0,1] [2,0,1] [4,0,1]` |
  | D3 | `[4,1,0,1] [13,1,0,1]` |
  | E1 | `[1,0,1] [2,0,1] [4,0,1] [3,1,1] [4,1,1] [6,1,1] [2,2,1] [3,2,1] [5,2,1] [1,3,1] [5,3,1] [6,3,1] [1,4,1] [5,4,1] [6,4,1] [2,5,1] [3,5,1] [5,5,1] [3,6,1] [4,6,1] [6,6,1]` |
  | D4 | `[3037000490,1] [3037000491,1] [3037000492,1]` |
  | D5 | `[2,0,1] [3,0,1]` |

  plus `factor(x^5+4x+1, 11) = [6,1] [10,2,1] [9,3,1]`, `factor(x^8+x^4+x^2+x+1, 7) = [1,1,1,1,1] [1,0,0,6,1]`,
  `factor((x-1)(x^2+2)(x^2+3), 5) = [4,1] [2,0,1] [3,0,1]` (same gp file). Fails on HEAD (A1: HEAD gives
  `[1,1] [0,1]`).
- `phase_two_alone_matches_pari` becomes `edf_rec_matches_pari` (`edf_rec(&f_m, d, p, &mut out)`, sorted,
  equals PARI): passes.
- `edf_rec_certificate_panics_on_bad_block`: new signature; still panics "violates its contract" (32
  failed tries, then the certificate).
- `tests/ch23_berlekamp.rs::factor_agrees_with_cantor_zassenhaus`: assert that
  `fp_factor::factor_with_multiplicity` is already in `order` before the existing sort (240 random cases).
- `zassenhaus.rs`: `factor_order_matches_pari` on the 12 polynomials of `$X/order_z.gp` (expected lists in
  E.5). Fails on HEAD for 9 of them.
- Remove the sort from `zassenhaus.rs::tests::factors_only`? Keep it (harmless); the new test covers the order.

---

## E. Zassenhaus failures and their messages

### E.1 Defect at HEAD

`zassenhaus::factor` (`zassenhaus.rs:273`) returns `Result<_, ()>`, documented "`Err(())` only if
recombination exceeds MAX_RECOMBINE" (`:271-272`). `Err(())` also comes from:

1. `choose_prime` (`:96-129`) finding no prime `p < 100_000` with `f mod p` squarefree of full degree
   (`:101`). Reproduction (`$X/t3_zass.rs`): `f = x^2 - P`, P the product of the primes below 10^5
   (43293 digits): `Err(())` after 0.2 s. `x^2 - P` is irreducible (P squarefree, not a square; gp
   `polisirreducible(x^2 - prod)` is 1). `factorization::is_irreducible` on it returns
   `Err(NotSupported("is_irreducible: Zassenhaus recombination exceeded the subset bound"))`.
2. `zp_hensel::hensel_lift_all` returning `None` (`:219`), an internal failure.

Every caller blames the subset bound: `factorization.rs:233-237`, `:438-443`, `:474-476`, the doc at `:131`;
`rustmath-galois/src/{cayley.rs:314-316, lowdegree.rs:64-66, quintic.rs:251-253, lib.rs:180-182}`
("factor recombination limit exceeded"); `rustmath-modular/src/modsym/decomposition.rs:247-248` ("its
factor-count bound"). `factor_over_integers`' doc (`factorization.rs:405-423`) still says it "does not
implement Zassenhaus".

### E.2 Fix

```rust
/// Why [`factor`] stopped. None of the variants says anything about `f`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZassenhausError {
    /// No prime p <= `bound` keeps f mod p squarefree of full degree `degree` (the bad primes divide
    /// lc(f)·disc(f) != 0; reaching `bound` = MAX_P needs a discriminant with more than 10^8 prime factors).
    NoGoodPrime { degree: i64, bound: i64 },
    /// Recombination tested more than `limit` subsets of the `lifted` modular factors (exponential worst
    /// case of Berlekamp-Zassenhaus; van Hoeij's knapsack recombination, MAGMA's method, is not implemented).
    RecombinationLimit { limit: u64, lifted: usize },
    /// Hensel lifting rejected the factorization mod `p` (internal error: the factors of a squarefree
    /// reduction are pairwise coprime).
    HenselLiftFailed { p: i64, factors: usize },
}
impl std::fmt::Display for ZassenhausError { ... }
impl std::error::Error for ZassenhausError {}
pub fn factor(f: &[Integer]) -> Result<(Integer, Vec<(Vec<Integer>, u32)>), ZassenhausError>;
```

Display texts (tests pin them):
- `"Zassenhaus: no prime p <= {bound} keeps f mod p squarefree of degree {degree}"`
- `"Zassenhaus: recombination tested more than {limit} subsets of {lifted} modular factors (Berlekamp-Zassenhaus is exponential in the number of modular factors; van Hoeij's lattice recombination is not implemented)"`
- `"Zassenhaus: Hensel lifting of {factors} factors mod {p} failed (internal error)"`

`choose_prime(f, prime_bound: i64)`: the loop condition `p < 100_000` becomes `p <= prime_bound`; `factor`
passes `fp_factor::MAX_P`. Termination: f is squarefree, so `disc(f) != 0`, and a prime is bad only if it
divides `lc(f)·disc(f)`; after at most ω(lc·disc) + MAX_PRIME_SCAN primes the scan has 30 good ones.
Typical inputs stop exactly where they stop now (30 good primes or one irreducible reduction). It calls
`fp_factor::factor_prime` (p proved prime by `is_small_prime`, trial division, and <= MAX_P).
`recombine(f, lifted, pk, cap: u64)` returns `RecombinationLimit { limit: cap, lifted: lifted.len() }`;
`factor_monic_squarefree` becomes `factor_monic_squarefree_with(f, prime_bound, cap)` with the public path
`(MAX_P, MAX_RECOMBINE)`; the Hensel `None` maps to `HenselLiftFailed { p, factors: fp_factors.len() }`.
`factor` sorts its output (item D). Doc `:267-272` rewritten.

### E.3 Callers (grep `zassenhaus::factor`, 12 sites)

| Site | Change |
|---|---|
| `factorization.rs:233` | `.map_err(\|e\| MathError::NotSupported(format!("is_irreducible: {e}")))`; doc `:131` names the three causes |
| `factorization.rs:438` | `format!("factor_over_integers: {e}")`; doc `:405-423` rewritten (it delegates to Zassenhaus; limits = the error variants) |
| `factorization.rs:474` | `format!("is_irreducible_over_integers: {e}")` |
| galois `cayley.rs:314`, `lowdegree.rs:64`, `quintic.rs:251`, `lib.rs:180` | `MathError::NotSupported(e.to_string())`; `cayley.rs:460` `.unwrap()` in a test needs `Debug` (derived) |
| modular `decomposition.rs:247` | `.map_err(\|e\| e.to_string())` |
| curves `belyi/regular_certify.rs:214` | **required** (`Err(()) =>` at `:215` no longer compiles; the only call site that breaks, by grep: every other site uses `map_err(\|_\| ..)`, `Err(_)` or `unwrap()`): `Err(e) => { ... why: format!("Zassenhaus factorization failed on the fibre polynomial: {e}") }`. Done in the same Plan step as the error type, so the workspace never stops compiling |
| curves `belyi/audit.rs:390`, functionfields `genus.rs:223` | compile unchanged (`Err(_)`) |

Behaviour: same `Err` variants of `MathError` as before, with the true cause; inputs whose bad primes
cover [2, 10^5) now factor instead of failing.

### E.4 Tests (`zassenhaus.rs` unless stated)

| Name | Input | Expected | HEAD |
|---|---|---|---|
| `factor_scans_past_bad_small_primes` | `x^2 - P`, P = ∏_{p < 10^5} p (built by a sieve in the test) | `Ok`, one factor `x^2 - P`, multiplicity 1, content 1 (gp `polisirreducible`) | fails (`Err(())`); about 0.3 s debug |
| `factorization::tests::is_irreducible_scans_past_bad_small_primes` | same through `is_irreducible` | `Ok(true)` | fails |
| `choose_prime_respects_bound` | `x^2 - 210` (`disc = 840 = 2^3·3·5·7`), `factor_monic_squarefree_with(f, 10, MAX_RECOMBINE)` | `Err(NoGoodPrime { degree: 2, bound: 10 })` | new internal API |
| `recombination_limit_is_reported` | `x^4 - 10x^2 + 1` (H23E6: splits mod every prime), `factor_monic_squarefree_with(f, MAX_P, 0)` | `Err(RecombinationLimit { limit: 0, .. })` | new internal API |
| `error_messages` | the three variants | the Display strings above | new |
| `factor_order_matches_pari` | 12 polynomials (below) | PARI order | fails for 9 |

PARI order (`$X/order_z.gp`, `factor(f)`), multiplicities in brackets when not 1:
`x^2-1`: x-1, x+1. `(x-1)^2(x+2)`: x-1 [2], x+2. `(x^2+1)(x^2+2)(x^3+x+1)`: in that order. `6x^2-6`:
content 6; x-1, x+1. `2x^2+3x+1`: x+1, 2x+1. `(x+3)(x-3)(x+1)(x-2)(x^2-2)(x^2+2)`: x-3, x-2, x+1, x+3,
x^2-2, x^2+2. `(x^2+x+1)(x^2-x+1)(x^2+3)(x^2-3)`: x^2-x+1, x^2-3, x^2+3, x^2+x+1. `(2x-1)(3x+1)(x-5)(5x+4)`:
x-5, 2x-1, 3x+1, 5x+4. `(x^4-10x^2+1)(x^2+x-1)(x^2-x-1)`: x^2-x-1, x^2+x-1, x^4-10x^2+1. `x^6-1`: x-1, x+1,
x^2-x+1, x^2+x+1. `x^12-1`: x-1, x+1, x^2-x+1, x^2+1, x^2+x+1, x^4-x^2+1. `(x-1)^3(x+1)^2 x`: x-1 [3], x,
x+1 [2].

---

## F. Cohen GTM 138 algorithm numbers

Citations checked (`grep -n -i "cohen\|GTM" rustmath-polynomials/src/{zx,univariate}.rs`):

| Site | Citation | Evidence | Verdict |
|---|---|---|---|
| `zx.rs:13`, `:183`; `univariate.rs:16`, `:1188` | Alg. 3.1.2 pseudo-division | Sage `polynomial_element.pyx`, `pseudo_quo_rem`: "ALGORITHM: Algorithm 3.1.2 in [Coh1993]" (`$X/dl_sage`, GitHub raw, develop) | correct |
| `zx.rs:13`, `:250`; `univariate.rs:16`, `:1043`, `:1384`, `:1448` | Alg. 3.3.1 sub-resultant gcd | Sage `unique_factorization_domains.py`, `_gcd_univariate_polynomial`: "Algorithm 3.3.1 in [Coh1993], based on pseudo-division", and its code is the same loop as `zx::subresultant_gcd` (content split, `g, h` updates, `h <- h^(1-δ) g^δ`); a library table of contents (web search) lists "3.3 The Sub-Resultant Algorithm, p. 118" | correct |
| `zx.rs:13` | "3.4.2 (square-free)" for Yun's algorithm over Z (`zx.rs:306`) | Sage `finite_field_base.pyx`: "[Coh1993], Algorithm 3.4.2 which is basically the algorithm in [Yun1976] with special treatment for powers divisible by p": a characteristic-p algorithm (§3.4 is factorization mod p) | **correct the wording**: "square-free decomposition by Yun's algorithm (D. Y. Y. Yun, *On square-free decomposition algorithms*, SYMSAC 1976, 26-35); Cohen Alg. 3.4.2 is its characteristic-p form" |
| `univariate.rs:18-19`, `:1042` | §3.2 (primitive parts and gcds over a UFD and its fraction field) | Sage `fields.py` cites "[Coh1993], Algorithm 3.2.2" for the extended Euclidean algorithm, so §3.2 holds Euclid's algorithms. A library catalogue of the book (Bibliotheksverbund Bayern, record BV013376443, `https://bvbvf1.bib-bvb.de/vufind/Record/DE-604.BV013376443/Details`, found by web search) lists "3.2.3 Polynomials over Unique Factorization Domains, p. 116" (content, primitive part, Gauss's lemma) and "3.2.4 Euclid's Algorithm for Polynomials over a UFD, p. 117". Review 1 confirms the same from its knowledge of the book | correct; make it "§3.2.3-3.2.4 (primitive parts over a UFD; Euclid's algorithm over a UFD)" at both sites |

The Knuth citations next to them (TAOCP vol. 2 §4.6.1: Algorithm D, R, C, E, Lemma G) are outside the item;
they agree with the book as far as known and are left alone. Also fix `berlekamp.rs`/`factorization.rs`?
They cite only "§3.4" (factorization mod p), which is right; no change.

Tests: none (documentation). `cargo doc -p rustmath-polynomials --no-deps` must stay warning-free for the
touched files.

---

## G. Ring hooks for polynomial rings

### G.1 State at HEAD

`impl<R: Ring> Ring for UnivariatePolynomial<R>` (`univariate.rs:857-873`) and
`impl<R: Ring> Ring for LaurentPolynomial<R>` (`laurent.rs:195-214`) override none of the hooks of
`rustmath-core/src/traits.rs:97-190` (`unit_status`, `nilpotent_status`, `unit_inverse`,
`multiplicative_order_status`, `static_characteristic`). So `2x + 1` in Z[x] has `unit_status() = None`,
`static_characteristic() = None` for Z[x] (Z has `Some(0)`), `UnivariatePolynomial::<Integer>::new(vec![2])`
has `None` (Z says `Some(false)`).

Scope. The task names parent-level hooks too (`characteristic`, `is_field`, `is_finite`). No polynomial
parent type exists in the workspace. No type implements `Parent` or `RingParentMethods` with a polynomial
`Element` (grep for `type Element = UnivariatePolynomial`/`LaurentPolynomial` and for `Parent for` on
polynomial types: no hit). The only struct named `PolynomialRing` (`rustmath-algebras/src/cached_algebra.rs:15`)
is a caching demo with a variable count and names, and no ring structure. `UnivariatePolynomial` and
`LaurentPolynomial` are the only polynomial types with an `impl Ring` (grep `Ring for` over the workspace).
The parent-level answers come from the element type through `RingParentMethods::characteristic`
(`rustmath-rings/src/category_methods.rs:328`, which reads `Element::static_characteristic()`) and the
`is_finite` of the generic containers (G.3). So the element-level hooks below, `static_characteristic`
included, are the whole scope.

### G.2 Theory (MAGMA ch. 23 §23.4.4 `IsUnit`, `IsNilpotent`; ch. 17 §17.4.2 `Characteristic`, §17.4.3 `IsCommutative`)

For `f = a_0 + ... + a_n x^n` (trimmed, `a_n != 0`), x central:

1. R ⊂ R[x] is a subring with the same 1, and `f ↦ f(0)` is a ring map R[x] → R that is the identity on R.
   So a constant c has in R[x] the unit status, nilpotency, inverse and multiplicative order it has in R;
   and `char R[x] = char R`.
2. Every ring, n >= 1: if `a_0` is a proven non-unit, f is not a unit (apply `f ↦ f(0)`). If `a_n` is a
   proven unit, f is not a unit: for `g != 0`, `a_n b_m != 0` (u b = 0 implies b = 0), so
   `deg(fg) = n + deg g >= 1`. The coefficient of `x^(kn)` in `f^k` is `a_n^k` and the constant term is
   `a_0^k`, so if `a_n` or `a_0` is a proven non-nilpotent, f is not nilpotent.
3. Commutative R (Atiyah-Macdonald, *Introduction to Commutative Algebra*, Ch. 1, Exercise 2; also Sage
   `Polynomial.is_unit` / `is_nilpotent` docstrings): f is a unit iff `a_0` is a unit and `a_1, ..., a_n` are
   nilpotent; f is nilpotent iff every `a_i` is nilpotent.
4. Not commutative: rule 3 fails. In `M_2(Z)[x]` the polynomial `I + A x + E_11 x^2` with
   `A = [[0,1],[1,0]]` is the matrix `[[1+x^2, x],[x, 1]]` of determinant 1, a unit (inverse
   `[[1, -x],[-x, 1+x^2]]`, GAP `$X/units.g`), and its coefficient A is not nilpotent. The recurrence of rule 5
   finds this inverse (`h = I - A x + E_22 x^2`, `$X/rev1/unit_inverse_proto.py`), and both products are I
   (GAP `$X/rev1/units_rev1.g`).
5. The inverse, in any ring, when `a_0` has a two-sided inverse b. The power series
   `h = Σ h_k x^k` with `h_0 = b` and `h_k = -b · Σ_{i=1}^{min(k,n)} a_i h_{k-i}` (k >= 1) is the unique
   right inverse of f in R[[x]]: `(f h)_k = a_0 h_k + Σ_{i>=1} a_i h_{k-i} = 0` for k >= 1. The
   recurrence has order n. So if n consecutive coefficients `h_{m+1}, ..., h_{m+n}` vanish, every later
   one vanishes, `h = h_0 + ... + h_m x^m` is a polynomial, and `f h = 1` exactly. Conversely, if f is a
   unit with inverse g of degree d, then h = g (uniqueness of the right inverse), and the run of n zeros
   starts at d + 1. So the recurrence finds every unit's inverse after d + n steps and
   `Σ_{k=1}^{d+n} min(k, n) <= n (d + n)` coefficient products. If `f h = 1` but `h f ≠ 1`, then f is not a
   unit: a unit's right inverse is its inverse. This replaces the earlier `a_0^(-1) Σ_k (-M)^k`
   (`M = a_0^(-1) f - 1`). That series is correct in the commutative case, but its terms grow to degree
   k·n and its cost is not bounded by the size of the answer (review 1, m5). The recurrence costs
   O(n·(d + n)) products, and needs no commutativity.

Laurent polynomials `f = Σ_{i=a}^{b} c_i x^i` (`c_a, c_b != 0`): a monomial `c x^k` is a unit (nilpotent)
iff c is (x^k is a central unit; `x ↦ 1` is a ring map to R); `(c x^k)^m = c^m x^(km)`, so a unit monomial
with k != 0 has infinite order. Every ring: `c_b` or `c_a` a proven non-nilpotent ⇒ f not nilpotent.
Commutative R, non-monomial: if `c_a c_b` is a proven non-nilpotent, choose a prime P with `c_a c_b ∉ P`;
f mod P is a non-monomial in `(R/P)[x, x^(-1)]`, whose units are monomials (domain), so f is not a unit.
If `c_a` (or `c_b`) is a unit and all other coefficients are nilpotent, f is a unit.

Laurent inverse. Write `f = x^a p(x)` with `p = Σ_{j=0}^{b-a} c_{a+j} x^j ∈ R[x]`, and
`f = x^b q(x^(-1))` with `q = Σ_{j=0}^{b-a} c_{b-j} y^j ∈ R[y]`. If p has a polynomial inverse u (rule 5,
which needs `p(0) = c_a` invertible), then `f^(-1) = x^(-a) u(x)`. If q has one, w (needs `c_b`
invertible), then `f^(-1) = x^(-b) w(x^(-1))`. For commutative R this finds every inverse whose
unit status the rules above decide. If c_a is a unit and f is a unit, then for every prime P the image of p
in the domain `(R/P)[x, x^(-1)]` is a unit with nonzero constant term, hence a constant. So
`c_{a+1}, ..., c_b` lie in every prime and are nilpotent, and p is a unit of R[x] (rule 3). The case of
c_b is the same with q. Not every Laurent unit has a unit extreme coefficient: over Z/6, `3 + 4x` has
the inverse `3 + 4x^(-1)` (GAP `$X/rev1/units_rev1.g`: `(3+4y)(4+3y) = y` over `ZmodnZ(6)`), and both
3 and 4 are zero divisors. There `unit_status` stays `None` and `unit_inverse` is `Err(NotImplemented)`.
Both are honest answers. Deciding such cases needs the idempotent decomposition of R, which a generic
coefficient type does not expose.

The rules in 3 need to know that R is commutative, and a generic `impl<R: Ring>` cannot see the marker
trait `CommutativeRing` (no specialization; rustmath has non-commutative `Ring` types: `FreeAlgebra`,
`CliffordAlgebra`, `GroupAlgebra`, `MatrixAlgebra`, `FQSym`, `NCSF`). Without them Z[x] stays undecided on
`2x + 1`.

### G.3 Fix

rustmath-core (`traits.rs`, after `static_characteristic`):

```rust
/// MAGMA `IsCommutative(R)` (Ch. 17 §17.4.3) for every ring this element type represents, when the
/// type alone determines it. `Some(true)` only if every value the type can hold, including any
/// deferred constant from `zero()`/`one()`, lies in a commutative ring. `None` (default) when it is
/// unknown or depends on runtime data. Containers use it to apply results that hold only over
/// commutative rings (`UnivariatePolynomial`: Atiyah-Macdonald Ch. 1 Ex. 2).
fn static_is_commutative() -> Option<bool> { None }
```

Overrides in P1 (each `Some(true)`): `i32`, `i64` (`traits.rs:413`, `:457`), `Integer`
(`rustmath-integers/src/integer.rs:1484` area), `Rational` (`rustmath-rationals/src/rational.rs:448` area).
Delegating: `UnivariatePolynomial<R>` and `LaurentPolynomial<R>` return `R::static_is_commutative()` and
`R::static_characteristic()`. Every other type keeps `None`, and its polynomial ring gets the rules of G.2.1
and G.2.2 only (still decisive over fields: a nonzero leading coefficient is a unit). Overrides for the
`rustmath-finitefields`, reals, complex and p-adic types are a follow-up for their crates (F1 can add the
finite-field ones: one line each).

`UnivariatePolynomial<R>` (no static constructor is called: the hooks only read coefficients. The 1 that
`unit_inverse` compares against is `b·a_0` with `b = a_0^(-1)`, so it lives in the coefficients' runtime
parent):

| Hook | Constant `c` | Degree >= 1 |
|---|---|---|
| `unit_status` | `c.unit_status()` | `Some(false)` if `a_0.unit_status() == Some(false)` or `a_n.unit_status() == Some(true)`; else if `R::static_is_commutative() == Some(true)`: `Some(false)` if some `a_i` (i >= 1) has `nilpotent_status() == Some(false)`, `Some(true)` if `a_0` is a proven unit and all `a_i` (i >= 1) proven nilpotent; else `None` |
| `nilpotent_status` | `c.nilpotent_status()` | `Some(false)` if `a_0` or `a_n` is a proven non-nilpotent; else, commutative: `Some(false)` if any coefficient is, `Some(true)` if all are proven nilpotent; else `None` |
| `unit_inverse` | `c.unit_inverse()` as a constant | `unit_status() == Some(false)`: `Err(NotInvertible)`. Otherwise `b = a_0.unit_inverse()`: `Err(NotInvertible)` passes through (f(0) is a non-unit, rule 2); any other `Err` gives `Err(NotImplemented)`. With b: the recurrence of G.2.5, counting coefficient products, at most `UNIT_INVERSE_WORK = 1 << 16`. Stop after n consecutive zero coefficients and drop them. Then check `f·h == 1` (else `Err(InvalidOperation)`: the coefficient arithmetic is inexact or inconsistent) and `h·f == 1` (else `Err(NotInvertible)`: a right inverse that is not a left inverse, so f is not a unit, G.2.5). Return h. Work bound reached first: `Err(NotImplemented("UnivariatePolynomial::unit_inverse: no inverse of degree < {k} within 65536 coefficient products"))`. This runs also when `unit_status()` is `None`, for example over a coefficient type of unknown commutativity. A returned h is a proven two-sided inverse in every case. |
| `multiplicative_order_status` | `c.multiplicative_order_status()` | `NotAUnit` if `unit_status() == Some(false)`, else `Undecided` |
| `static_characteristic` | `R::static_characteristic()` | |
| `static_is_commutative` | `R::static_is_commutative()` | |

`LaurentPolynomial<R>`:
- `static_*` delegate.
- Monomials `c x^k` delegate `unit_status` and `nilpotent_status`. `unit_inverse` is `c^(-1) x^(-k)`, with
  `k.checked_neg()`: for `k = i32::MIN` it returns
  `Err(NotImplemented("LaurentPolynomial::unit_inverse: the exponent {k} has no negative in i32"))`. The
  monomial is still a unit; only the result type cannot hold the inverse. Order: `k = 0` delegates;
  `k != 0` with a proven unit c is `Infinite`, and with a proven non-unit c it is `NotAUnit`.
- Non-monomials: `nilpotent_status` is `Some(false)` by the extreme coefficients. `unit_status` follows
  the commutative rules of G.2 (non-unit from `c_a c_b`, unit from an extreme unit plus a nilpotent rest).
  `unit_inverse`: `Err(NotInvertible)` on a proven non-unit. Otherwise, if `c_a` is invertible, take
  `p = Σ c_{a+j} x^j` and its `UnivariatePolynomial::unit_inverse` u, and return `x^(-a) u`. Else, if
  `c_b` is invertible, take the reversed `q` and its inverse w, and return `x^(-b) w(x^(-1))` (G.2,
  "Laurent inverse"). All exponent arithmetic is checked (`checked_neg`, `checked_add`), and overflow
  gives `Err(NotImplemented)`. If neither side gives an inverse, `Err(NotImplemented)`. A returned inverse
  is two-sided, because u (or w) is.
- Order of a non-monomial: `NotAUnit` or `Undecided`.
- The empty map (zero) keeps the defaults.

Consumers of the hooks. The grep covers all Rust code in the workspace, `src/` unit tests included. It
looks for the hook names (`unit_status()`, `nilpotent_status()`, `unit_inverse()`,
`multiplicative_order_status()`, `static_characteristic()`) and for their generic wrappers. The generic
consumers are:
- `rustmath-category/src/ring_category.rs:170-192`, the `RingElementMethods` blanket (`is_unit`,
  `is_nilpotent`, `multiplicative_order`);
- `rustmath-rings/src/category_methods.rs:70, :95` (`is_unit_element`, `inverse_of_unit`), and `:328`
  (`RingParentMethods::characteristic`, from `Element::static_characteristic()`);
- `rustmath-modules/src/with_basis/parent.rs:195, :272`, `rustmath-matrix/src/matrix_algebra.rs:383`
  and `rustmath-algebras/src/algebra_with_parent.rs:142` (`is_finite` is `Some(false)` when
  `R::static_characteristic() == Some(0)`).

The other hits are concrete types deciding their own hooks: finitefields, rationals and the core primitives.
Then I grepped every file that names `UnivariatePolynomial` or `LaurentPolynomial` for these wrappers
(`is_unit()`, `is_nilpotent()`, `multiplicative_order()`, `is_unit_element()`, `inverse_of_unit()`,
`characteristic()`, `is_finite()`). Exactly one test instantiates a consumer with a polynomial type:

- `rustmath-algebras/src/algebra_with_parent.rs:274-277` (`test_simple_algebra_finiteness`) asserts
  `SimpleAlgebra::<UnivariatePolynomial<Integer>>::new(2).is_finite() == None`, with the comment "The
  polynomial element type states no characteristic: undecided, not "infinite"". After the delegation,
  `static_characteristic()` is `Some(0)` and `is_finite()` is `Some(false)`. That is correct: Z[x]^2
  contains Z (GAP `IsFinite(PolynomialRing(Integers, 1))` is `false`). The test changes to `Some(false)`,
  and the comment becomes "The polynomial element type has characteristic 0 from its coefficients, so
  Z[x]^2 is infinite (GAP: IsFinite(PolynomialRing(Integers, 1)) -> false)."

The other hits in those files are unrelated: `rustmath-rings/src/augmented_valuation.rs` (`is_finite` of
valuation values), `rustmath-polynomials/tests/ch23_is_irreducible.rs:762-808` (a coefficient type with its
own `unit_status`, read by `factorization::is_irreducible`), `rustmath-jupyter`, `root_label.rs` (`f64::is_finite`),
and the finitefields files. No `RingParentMethods` implementor has a polynomial `Element`, and no test builds
a matrix algebra or free module over a polynomial type. `rustmath-polynomials/src/factorization.rs:243-251`
(`is_proven_unit`) and `:280` apply to coefficient types, not to polynomials: a coefficient type there must
be a `EuclideanDomain`, which polynomials are not. Every other consumer now gets decided answers with
polynomial coefficients where it got `None`. Each new answer is a theorem of G.2.

### G.4 Tests (new `rustmath-polynomials/tests/ring_predicates.rs`; plus one assertion in each of
`rustmath-core/src/traits.rs::test_i32_i64_element_predicates`, `rustmath-integers/tests/ring_predicates.rs`,
`rustmath-rationals/tests/ring_predicates.rs` for `static_is_commutative() == Some(true)`; and the changed
`rustmath-algebras` test of G.3)

References: GAP `$X/units.g` (`IsUnit(PolynomialRing(Integers), 2*x+1) = false`, `2*x^2-1` false, `-1` true,
`x` false; `IsUnit(PolynomialRing(Rationals), y+1) = false`, `Inverse(2) = 1/2`; over ZmodnZ(8):
`(1+2t)(1-2t+4t^2) = 1`, `(2+4t)^3 = 0`; over ZmodnZ(12): `(5+6u)^2 = 1`, `(6+6u^2)^2 = 0`, the powers of
`5+6u` alternate (order 2); `det [[1+x^2, x],[x,1]] = 1` and the product with `[[1,-x],[-x,1+x^2]]` is I).
GAP `$X/rev1/units_rev1.g`: `det [[1+x, 0],[0, 1]] = x+1` and `IsUnit` of it in Z[x] is false; both products of
`[[1+x^2, x],[x,1]]` and `[[1,-x],[-x,1+x^2]]` are I; `(1+2t)(1+6t+4t^2) = 1` over `ZmodnZ(8)`;
`(3+4y)(4+3y) = y` over `ZmodnZ(6)`.
Sage docstrings (`$X/dl_sage/polynomial_element.pyx`, `is_nilpotent`): over `Integers(12)`, `5 + 6x` is not
nilpotent, `6 + 6x^2` is. GAP's `IsUnit(PolynomialRing(Integers), 2)` answers `true` (its constant lands in
the rational-function family): not used; the constant 2 follows `Integer::unit_status` (units of Z[x] are ±1).

| Test | Coefficients | Assertions | HEAD |
|---|---|---|---|
| `z_polynomial_units` | `Integer` | `2x+1`, `2x^2-1`, `x`, `0`, `2` → `Some(false)`; `-1`, `1` → `Some(true)`; `unit_inverse(-1) = -1`; `unit_inverse(2x+1) = Err(NotInvertible)`; orders: `-1` `Finite(2)`, `x` `NotAUnit` | `None` for `2x+1`, `2x^2-1`, `x`, `2` |
| `z_polynomial_nilpotents_and_statics` | `Integer`, `i64` | `2x`, `x+3` → `nilpotent Some(false)`; `0` → `Some(true)`; `static_characteristic() == Some(0)`, `static_is_commutative() == Some(true)` | `None`s |
| `q_polynomial_units` | `Rational` | `2` unit with inverse `1/2`; `y+1` → `Some(false)` | `None` |
| `zmod8_units_and_inverse` | test type `Zm<M>` (const generic modulus, i128 products, commutative, exact hooks, `static_is_commutative = Some(true)`), M = 8 | `1+2t` unit, `unit_inverse = 1 + 6t + 4t^2` (i.e. 1-2t+4t^2), product 1; `1+4t+2t^2` unit, inverse `1 + 4t + 6t^2 + 4t^4` (an interior zero coefficient: the stop rule needs n = 2 consecutive zeros; `$X/rev1/unit_inverse_proto.py`), product 1; `2+4t` nilpotent; `1+3t` not a unit, not nilpotent | `None` |
| `zmod12_sage_examples` | `Zm<12>` | `5+6u` unit, inverse `5+6u`, not nilpotent; `6+6u^2` nilpotent; order of `5+6u` `Undecided` (documented limit) | `None` |
| `commutativity_unknown_stays_undecided` | `ZmNoFlag<8>` (same arithmetic, default `static_is_commutative`) | `1+2t` → `unit_status None`, but `unit_inverse = Ok(1 + 6t + 4t^2)` (the recurrence needs no commutativity; both products are 1); `t+1` (lc unit) → `Some(false)` | `None`; `unit_inverse` `Err` |
| `noncommutative_matrix_coefficients` | test type `M2` (2×2 over i64, `unit_status` by det ±1, `nilpotent_status` by `A^2 = 0`, default commutativity) | `I + A x + E11 x^2` → `unit_status None`, `unit_inverse = Ok(I - A x + E22 x^2)`, and the test checks both products are I; `I + E11 x` → `unit_status None`, `unit_inverse` is `Err(NotImplemented)` after the work bound (true answer: not a unit, det = x+1); `I + I x` → `Some(false)`; `E12 + E12 x` → nilpotent `None` | `None`, `Err` |
| `unit_inverse_work_bound` | `Zm<2^62>`, `ZmNoFlag<12>`, `Zm<12>` | over `Zm<2^62>`: `1+2t` has an inverse of degree 61, `h_k = (-2)^k mod 2^62`, product 1 (62 coefficient products); over `ZmNoFlag<12>`: `1+4t` → `unit_status None`, `unit_inverse` `Err(NotImplemented)` whose message names 65536 (4 is not nilpotent mod 12, so f is not a unit; the recurrence never stops); over `Zm<12>`: `1+4t` → `Some(false)` and `Err(NotInvertible)` at once | new |
| `laurent_hooks` | `Integer`, `Zm<8>`, `Zm<6>` | `-x^5` unit, inverse `-x^-5`, order `Infinite`; `3x^-2` not a unit; `x + x^-1` not a unit; `x^(i32::MAX)` inverse `x^(-i32::MAX)`; `x^(i32::MIN)` unit, order `Infinite`, `unit_inverse` `Err(NotImplemented)` (no i32 negative); over `Zm<8>`: `1 + 2x^-1` unit with inverse `1 + 6x^-1 + 4x^-2` (high side), `x^-3 + 2x^-2` unit with inverse `x^3 + 6x^4 + 4x^5` (low side), products 1; `2x^3` nilpotent; over `Zm<6>`: `3 + 4x` → `unit_status None`, `unit_inverse` `Err(NotImplemented)` (documented limit: its inverse is `3 + 4x^-1`, GAP above) | `None`; HEAD has no Laurent hooks |

---

## H. MAGMA conventions (documentation; no behaviour change)

Sources: the MAGMA V2.19 handbook, volume 2 (`https://www.math.uzh.ch/sepp/magma-2.19.8-cr/HandbookVolume02.pdf`,
reachable; `$X/dl_magma/ch23.txt` is its chapter 23 as text), the ch. 23 digest, Sage sources
(`$X/dl_sage`, GitHub raw, develop branch) and PARI/GP.

1. **Quotrem over Z.** V2.19 §23.4.10 has the V2.3 text word for word, including "Over the integers
   (R = Z) this will be true in general when the leading coefficient of g divides that of f." No example.
   Sage over ZZ calls FLINT `fmpz_poly_divrem` (`polynomial_integer_dense_flint.pyx:691-757`): its doctests
   give `(5*x+7).quo_rem(3) == (x + 2, 2*x + 1)` and `(x^2).quo_rem(2*x - 1) == (0, x^2)`. RustMath gives
   `(0, 5x + 7)` and `(0, x^2)`; both remainders have the minimal degree 1 resp. 2 (the x coefficient of
   `5x + 7 - 3q` is `5 - 3t != 0`). PARI `divrem(5*x+7, 3)` divides over Q (`[5/3*x + 7/3, 0]`). No source
   says which minimal remainder MAGMA returns: keep, and add the Sage examples and the V2.19 confirmation to
   the doc of `UnivariatePolynomial::div_rem` (`univariate.rs:1061-1095`) and the header of
   `tests/ch23_quotrem_gcd.rs`; pin `(5x+7).div_rem(3) = (0, 5x+7)` with the FLINT alternative checked
   (new case in `quotrem_over_z_minimal_remainder_is_not_unique`).
2. **Content over Q.** V2.19 §23.5.2: "The content of p, that is, the greatest common divisor of the
   coefficients of p as an element of the coefficient ring." Chapter 20 (Rational Field) defines no gcd of
   rationals. Sage: `Polynomial.content_ideal` is `self.base_ring().ideal(self.coefficients())`, and
   `QQ.ideal(2)` is "Principal ideal (1) of Rational Field" (`fields.py:644-655`): the unit ideal, as
   RustMath's `content = 1`. PARI `content(x/2+1/3) = 1/6`, `content(2/3*x+4/5) = 2/15` (the Z-module
   content). Keep 1; add the Sage evidence to `univariate.rs:1330-1341` and `factorization.rs:9-12`.
3. **Content in IsIrreducible over Z.** V2.19 §23.8.1: "returns true if and only f is irreducible over R";
   no example. Sage `Polynomial.is_irreducible` doctests (`polynomial_element.pyx:10039-10100`): over ZZ
   `R(2*x).is_irreducible()` is False, `R(1)`, `R(4)` False, `R(5)` True; over QQ `R(2*x)` True. RustMath's
   `is_irreducible` agrees on all five (content counts over Z; prime constants are irreducible). Keep; add
   the Sage evidence to `factorization.rs:115-119` and port the five cases as a test
   (`tests/ch23_is_irreducible.rs::sage_content_examples`; passes on HEAD).
4. **Shape of QMatrix.** V2.19 §23.8.1: "the Berlekamp Q-matrix associated with f, which is an element of
   the degree d − 1 matrix algebra over F" (same as V2.3). On the same page `CompanionMatrix` is "an
   element of the full matrix algebra of degree d − 1" while the displayed matrix has d rows and the d
   entries `−a_0 … −a_(d−1)` in its last row, so "degree d − 1" is not the matrix size there (MAGMA's
   "degree n" is n × n: "Full Matrix Algebra of degree 3 over GF(23)", ch. 26). MAGMA cites Knuth §4.6.2
   for Berlekamp (V2.19 §23.8.1, `Factorization`), whose Q is d × d with row k the coefficients of
   `x^(kq) mod f`, which is `berlekamp::q_matrix`. Keep; rewrite `berlekamp.rs:312-314` with this
   evidence.

---

## I. Adjacent findings (not in this unit; for the HANDOFF)

- Facade-like `Err` handling of Zassenhaus results: `rustmath-groups/src/galois_resolvent.rs:159-169`
  (`is_irreducible` returns false on `Err`), `rustmath-curves/src/belyi/audit.rs:390-398` (false on `Err`),
  `rustmath-igp24/src/main.rs:117` (`unwrap_or(false)`), `rustmath-functionfields/src/factor.rs:70` and
  `genus.rs:240-244` (treat a failed factorization as irreducible). After item E, `Err` means the
  recombination limit (or an internal error); these sites should propagate it.
- `pgalois::local_galois_candidates` assumes `Gal(f/Q_p)` transitive; at a ramified prime an f that is
  irreducible over Q but reducible over Q_p has an intransitive local group, so "the true group is always
  included" (`pgalois.rs:160`) can fail.
- `LaurentPolynomial::is_one` is false for the zero polynomial of the zero ring.
- `rustmath-numberfields/src/panayi.rs:326` duplicates `is_padic_square` without the prime check.

---

## Plan

**Rebase on I1.** I1 is implemented before P1 and touches P1's files. Start P1 from the branch state after
I1, and keep I1's changes:
- I1 makes `rustmath_integers::prime::is_prime_u64` public and adds `try_is_prime` (I1 §2.1). P1 uses both
  (B.3, B.4, B.7).
- `rustmath-polynomials/src/fp_factor.rs`: I1 rewrites the doc comment of `check_prime` (`:501-503`). P1
  replaces the function with `check_modulus` and writes its own doc (B.3).
- `src/factorization.rs`: I1 rewrites `integer_constant_is_irreducible` (`:172-196`) and its doc (`:122-127`).
  P1 edits other parts of the file: the messages at `:233`, `:438`, `:474`, the docs at `:131` and
  `:405-423`, and the Sage evidence next to `:115-119` (H.3). Merge by hand around `:115-131`.
- `tests/ch23_is_irreducible.rs`: I1 changes `:596-598` (2^89 - 1 is now `Ok(true)`). P1 adds
  `sage_content_examples` (H.3), whose values agree with I1's semantics (`R(5)` is a proven prime).
- `src/berlekamp.rs`: I1 rewrites the docs at `:260-266` and `:337-341`. P1 then applies B.7 on top (the
  I1 §7 hand-off, accepted).

Implementation order (one crate group). Each step leaves `cargo check --workspace --tests` green and the
crates it touches test-green before the next.

1. **rustmath-core** `src/traits.rs`: add `Ring::static_is_commutative` (default `None`), overrides for
   `i32`, `i64`; extend `test_i32_i64_element_predicates`.
2. **rustmath-integers** `src/integer.rs` (Ring impl next to `static_characteristic`), **rustmath-rationals**
   `src/rational.rs`: `Some(true)`; one assertion each in `tests/ring_predicates.rs`.
3. **rustmath-polynomials** `src/fp_factor.rs`: header (MAGMA ch. 23 §23.8.1, papers as now, plus
   Pomerance-Selfridge-Wagstaff 1980 and the exhaustive scan of B.2 cited in the `check_modulus` doc only
   if the 4-base fallback of the Risks section is used);
   `reduce`, `squarefree_check`, `prime_divisors`, `pub check_modulus`, `pub factor_order`,
   `pub(crate) factor_prime`; `distinct_degree_factor` (A); `squarefree_factor` → wrapper + `radical`;
   `block_contract` sparse ladder (C); `edf_rec` without phase 1, `TRIES_BEFORE_CERTIFICATE`; delete legacy
   items; sorted outputs; docs and `# Panics`; tests of A.4, B.6, C.4, D.4. The DDF check reaches
   `rustmath-galois` `sieve::cycle_type_mod_p`: run `cargo test -p rustmath-galois --lib sieve` here.
4. `src/zassenhaus.rs`: `ZassenhausError`, `choose_prime(f, bound)`, `recombine(.., cap)`,
   `factor_monic_squarefree_with`, `factor_prime`, sorted output, docs; tests of E.4. In the same step, the
   one caller that stops compiling: **rustmath-curves** `src/belyi/regular_certify.rs:215` (`Err(()) =>`
   becomes `Err(e) =>` with the message of E.3). Every other call site still compiles (E.3), so
   `cargo check --workspace --tests` stays green.
5. `src/factorization.rs`: messages (E.3), doc of `factor_over_integers`, Sage evidence (H.3), new test.
   Merge with I1's `integer_constant_is_irreducible` edit (see "Rebase on I1").
5b. `src/berlekamp.rs`: B.7 (`check_prime` with `try_is_prime`, the two error texts, one doc sentence);
   test `berlekamp_rejects_strong_pseudoprime_psi12` in `tests/ch23_berlekamp.rs`.
6. `src/padic_factor.rs`: `check_modulus` at the three public entries; test
   `cycle_type_rejects_composite_p` (`should_panic`, `x^2+1`, 15).
7. `src/univariate.rs`, `src/laurent.rs`: hooks (G); docs of `div_rem`, `content` (H.1, H.2); Cohen
   citations (F). `src/zx.rs`: header and Yun citation (F). `src/berlekamp.rs:312`: QMatrix evidence (H.4).
   `tests/ring_predicates.rs` (new), `tests/ch23_quotrem_gcd.rs` (H.1 case and header),
   `tests/ch23_is_irreducible.rs` (H.3), `tests/ch23_berlekamp.rs` (sortedness assertion). In the same
   step, **rustmath-algebras** `src/algebra_with_parent.rs:274-277`: `test_simple_algebra_finiteness`
   expects `Some(false)` for `SimpleAlgebra<UnivariatePolynomial<Integer>>`, with the new comment (G.3).
   Run `cargo test -p rustmath-algebras --lib algebra_with_parent` here.
8. **rustmath-groups** `src/pgalois.rs` (B.4: primitive part, trimmed degree, root at infinity,
   `check_modulus`, `is_prime_u64` in `is_padic_square`, constants in `local_galois_candidates`) and
   `tests/pgalois.rs` (B.6, including the content, linear and constant rows).
9. **rustmath-rings** `src/valuation/maclane.rs` (C.2), `src/valuation/residue_tower.rs` (doc),
   `src/function_field/typed/factor.rs` (MAX_P).
10. Messages at the callers of the error type (they compile since step 4, but still blame the subset
    bound): **rustmath-galois** `cayley.rs`, `lowdegree.rs`, `quintic.rs`, `lib.rs`; **rustmath-modular**
    `modsym/decomposition.rs`. **rustmath-igp24** `src/main.rs:266-275`: validate the prime
    list (B.5); check by hand that `cargo run -p rustmath-igp24 -- '[1,0,1]' '[5,15]'` prints the error
    and `'[5,7]'` still prints the JSON.
11. Trackers: `PROJECT_TRACKER.md` rows for fp_factor, zassenhaus, pgalois, maclane, the polynomial Ring
    hooks (the core three-valued predicates row: polynomials no longer "keep the defaults"), the MAGMA
    convention notes; `docs/HANDOFF.md` follow-ups (remove the P1 items, add section I).

Acceptance commands:

```bash
cargo test -p rustmath-core --no-fail-fast
cargo test -p rustmath-integers --test ring_predicates
cargo test -p rustmath-rationals --test ring_predicates
cargo test -p rustmath-polynomials --no-fail-fast
cargo test -p rustmath-groups --no-fail-fast            # pgalois, galois_resolvent, transitive24
cargo test -p rustmath-rings --no-fail-fast             # maclane, residue tower, typed function fields
cargo test -p rustmath-galois --no-fail-fast            # Zassenhaus order and errors, sieve DDF check
cargo test -p rustmath-numberfields --no-fail-fast      # prime order of ideals, class groups, s-units
cargo test -p rustmath-curves --no-fail-fast            # belyi, igp_closure
cargo test -p rustmath-modular --no-fail-fast
cargo test -p rustmath-functionfields --no-fail-fast
cargo test -p rustmath-symbolic --no-fail-fast          # factor, Risch
cargo test -p rustmath-category -p rustmath-modules -p rustmath-algebras --no-fail-fast   # hook consumers
cargo build -p rustmath-igp24
cargo check --workspace --tests
cargo clippy -p rustmath-polynomials -p rustmath-groups
```

Baseline: the same commands on HEAD first (all green at `c79a582`, 14,173 passed workspace-wide); the
tests listed as failing on HEAD must fail there before the change.

Risks:

- **I1 dependency.** `check_modulus` needs a fast exact word-size primality test, and B.7 needs
  `try_is_prime`. Both come from I1, which is implemented first. If I1's public
  `rustmath_integers::prime::is_prime_u64` is missing after all, `Integer::is_prime` is exact but costs
  about 2.5 ms per call in debug. The internal paths (`zassenhaus`) are unchecked anyway, but `galois::sieve`
  and `numberfields` call checked entries per prime. Then measure `cargo test -p rustmath-galois` time
  before and after, and only if it regresses by more than 10 % add the 4-base strong test (exact on
  [2, MAX_P], B.2) as a private function, citing the scan. Without `try_is_prime`, B.7 is skipped and
  recorded in the HANDOFF.
- **Order changes** reach outputs that no test pins (galois notes text, igp24 JSON, numberfields prime
  order). Audited in D.3; the acceptance run is the check. Do not restore HEAD order anywhere: the new
  order is PARI's.
- **Core trait change** rebuilds the whole workspace (a defaulted method; no compile break expected; a type
  with an inherent `static_is_commutative` would shadow it: grep finds none).
- **choose_prime past 10^5**: only reached when every prime below 10^5 is bad; bounded by ω(lc·disc).
- **`unreachable!` in pgalois** relies on two facts. First, `disc` is the binary-form discriminant (MAGMA's
  definition, `disc.rs:199-213`), which holds for g of exact degree n after trimming. Second, g is
  primitive, so `g mod p != 0`. Review 1 showed that without the second fact the branch is reachable
  (linear f with p | content). With the primitive part, 28,213 randomized gp cases found no violation
  (B.4).
- **`unit_inverse` work bound** (`1 << 16` products). An inverse of degree d is found when
  `n·(d + n) <= 65536`. Past that the answer is `Err(NotImplemented)`, never a wrong value. The memory
  bound is at most 65537 coefficients of R. Over a coefficient type with huge elements (for example
  Integer-backed residues modulo a 10^4-digit number), this can reach tens of megabytes before the `Err`.
  That is no more than the size of an inverse of that degree.
- **Test runtime**: `factor_scans_past_bad_small_primes` about 0.3 s debug (measured 0.2 s for the scan at
  HEAD).

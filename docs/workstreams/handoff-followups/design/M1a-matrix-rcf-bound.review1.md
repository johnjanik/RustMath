# Review 1 of M1a-matrix-rcf-bound (design `M1a-matrix-rcf-bound.md`)

Reviewer scratch: `$R = .../fu2/design/M1a-matrix-rcf-bound/review1/` (harnesses in `$R/src/`, built with the
designer's `exp/build.sh` against the HEAD debug rlibs; no repository file touched).

Verdict: **approve**. No blockers. Eight minor issues, each with a required change below; all are
documentation, test or argument additions, none changes the algorithm.

## 1. What I reproduced

| Claim | Reproduction | Result |
|---|---|---|
| Item 1 defect: `CONJDIAG20_S7` slow at HEAD | `$R/t4 old cd7` (HEAD public functions) | IF 93.6 s, RCF 166.4 s (design: 101.6 s / 177.7 s; HANDOFF 81 s). Confirmed |
| `CONJDIAG20_S7` constant is the timed matrix, R 2×2 with degrees [13, 7] | `$R/src/c7.rs`: constant == `conjdiag(20, 7)`; `krylov_presentation` r = 2, degrees [13, 7]; Rust constant == `cd7_const.gp` (400 entries) | Confirmed |
| r = n worst case slow at HEAD | `$R/t4 old wbw14`, cap 250 s | IF + RCF did not finish in 250 s. Confirmed |
| `PrimeField` GF(101), n = 60, HEAD RCF ≈ 99 s | `$R/t7 old wbws 60 101` | IF 4.9 s, RCF 84.9 s. Confirmed |
| New path fast | prototype `t4 new`: cd7 0.22 s IF / +0.14 s T+cert; wbw20 0.27 / 0.21; wbwrep20 0.48 / 0.89; arrow20 0.28 / 0.04 (18 merges); cd6 0.91 / 0.41. `t7 new wbws 60 101`: IF 0.84 s, T+cert 0.90 s | Matches section 1.4 |
| gp references | `bref.gp`, `cd7chk.gp`, `gf101.gp`, `cmp.gp` | every line prints 1 |
| Fuzz agreement with HEAD | `exp/t2.rs`: Q 600/0 bad, GF(p) 1200/0 bad, sentinel families both = 19, old only = 24, neither = 2, disagree = 0; `exp/t10.rs`: GF(p^k) 180/0 bad/0 err | Confirmed |
| Item 2 defect | `exp/treal.rs`: 8×8 `Real`: presentation Err, `rational_canonical_form` Err "internal: rational form certificate failed" (so it ran the classical fallback); `companion.rs:540-545` says the fallback "only happens" for sentinels; `polynomial_matrix.rs:24` and `:176-178` name only `PrimeField`; `FiniteFieldElement` has the same Z-semantics sentinels (`finite_field.rs:859-875`, `:922-939`) | Confirmed |
| Line references | `companion.rs:466/564/737/1101/1408/1591/1669/1858/2134/2236`, `krylov.rs:265/486`, `polynomial_matrix.rs:203/223/238`, tracker and csv rows, HANDOFF `:312-315` | All correct |
| Callers | my grep of the whole workspace for the affected names | Same as section 1.3: no production caller outside rustmath-matrix (`polynomial_ops.rs:88` `Matrix::minimal_polynomial` is an unrelated eigenvalue method; numberfields/rings/algebraic/finitefields hits are other functions; `rnfeq.rs:197` uses `characteristic_polynomial`). Only test caller outside the crate: `rustmath-finitefields/tests/matrix_forms_gfp.rs` |
| Citation: Steel 1997 is O(n⁴) | `dl/diss.txt:4085-4112` (Storjohann thesis Fig. 9.6, row 9) | Confirmed |
| MAGMA digest | `Chapter_26_Matrices.md:271-312`: §26.11, §26.12.2, [Ste97], H26E9 without the matrix | Confirmed |

## 2. Algorithm check (re-derived)

* (F1), (F3), (F5): standard module facts; correct. The level-by-level conclusion (invariant factors of V
  are f_1 together with those of V/U_1, and so on) needs only the PID structure theorem, not admissibility.
* (F4) coprime split: invariants αβ = lcm(f, a), α | f, β | a hold (per prime, e_α + e_β stays e_a once
  β holds the prime). The worked examples of `coprime_split_examples` are right (I traced all four).
* (F6): I re-derived it with the Hoffman-Kunze Step 2 argument (any representation f(A)v = Σ g_i(A)v_i with
  the Step 1 vectors forces f | g_i, by the degree contradiction with the maximal conductor of v_j). The
  design's own proof via the invariant complement is also valid. (F7): R' = Y·D, ψ = φ∘Y, ker ψ = im D:
  correct.
* Exponent certificate (1.2.4 item 3): every live candidate gets an h with h(A)y ∈ U from an exact
  check; dropped candidates lie in U; candidates plus U generate V; so f annihilates V/U and equals
  ann_U(v) = exp(V/U). Sound. Only an `Err`, never a value, on a failed check: no facade.
* Single span with `reduce_prefix(y, m)`: the first m echelon rows are never modified by later pushes and
  are an echelon basis of U, so a zero reduction decides membership exactly. Correct.
* Termination for any `Field` impl: at most n stored vectors per span, one pushed vector at least per
  level, merges must raise deg f, Euclid degrees strictly drop (`div_rem_field` keeps `r.len() <= db`),
  capped coprime split. Correct.
* Cost O((r+1)·n³) field operations: re-derived (per level, r candidates at O(D_ℓ·n² + n²), merges at most
  min(D_ℓ, r) at O(D_ℓ·n²), gcds of degree at most D_ℓ). Correct.

## 3. Minor issues (required changes)

### m1. The Z-semantics argument for the new first stage is missing

The new stage answers sentinel input (19 natural families, and random cases), and the docs of section 2.2
keep the claim "the result over Z and, reduced, over GF(p) for every p". The design argues that claim only
for the Krylov path and for T (`det T = ±1`). For `invariant_factors_peeled` (no T) and for
`minimal_polynomial_peeled` (only level 1, the span is not full) no argument is given, and section 1.6 puts
no "Exactness" paragraph in the `frobenius.rs` header.

The claim is true; I checked it. My harness `$R/src/z1.rs` (3000 random sentinel matrices, entries in
{-1, 0, 1}, n = 2..8) and `z2.rs` (4000, entries in {-2..2}) compare every peeled result (Full and
MinimalPolynomial depth, no RCF check) with HEAD's `invariant_factors` of the bound matrix over GF(2), GF(3),
GF(5), GF(7): 761 + 509 answered, 0 disagreements, `generators` never failed. The argument:

* Full depth: success means every push of the final span inverted a pivot ±1, so the matrix of the stored
  vectors A^t·v_ℓ is unit lower triangular times a matrix that is unit upper triangular in pivot column
  order: unimodular over Z. The verified relations hold over Z, the f_ℓ are monic, and the (F6) quotients
  are exact divisions by monic polynomials, so in Z[x]. Hence R' = Y·D over Z[x] with Y unimodular, the
  presentation Z^n ≅ Z[x]^s / im R' holds over Z and modulo every p, and the invariant factors mod p are the
  f_ℓ mod p.
* MinimalPolynomial depth: the spin of v_1 had unit pivots, so v_1, ..., A^{D-1}·v_1 stay independent mod p
  (a D×D minor is ±1), giving deg ann_p(v_1) ≥ D; and f_1(A) = 0 over Z (f_1 = μ over Q, certified), so
  μ_p | f_1 mod p. Hence μ_p = f_1 mod p.

Required: add this argument to the design (section 1.2.4 or a new 1.2.9) and to the `frobenius.rs` header
("Exactness", as `krylov.rs` has one), and add a test that runs `invariant_factors_peeled` and
`minimal_polynomial_peeled` on the 19 natural sentinel families it answers and compares with the bound
GF(2), GF(3), GF(5), GF(7) results (the existing `check_forms_sentinel` only sees the public chain).

### m2. A behaviour change is not listed: sentinel inputs that HEAD rejects are now answered

Section 1.3 item 4 says the chain "answers every input HEAD answered" and implies nothing else changes.
`$R/src/z4.rs` (6000 random sentinel matrices) finds Err → Ok changes: the 4×4 sentinel matrix
`[0,0,0,0; -1,1,0,0; 1,0,0,0; 0,-1,1,0]` gives `Err(InvalidArgument("No inverse exists: gcd(2, 0) = 2"))`
at HEAD for both `invariant_factors` and `minimal_polynomial`, and `x⁴ - x³` on the new path (`z5.rs`),
which is right: gp `matfrobenius(A,1)` = `[x^4 - x^3]` and `lift(matfrobenius(A*Mod(1,p),1))` =
`[x^4 + (p-1)*x^3]` for p = 2, 3, 5, 7, equal to the bound results. The prototype can also make
`minimal_polynomial` answer where `invariant_factors` errs (Full depth fails at level 2 while level 1
succeeds; seen 2 times in 3000, both still answered by HEAD's fallback, so no Ok → Err change was found).
Required: list both in section 1.3 (Err → Ok in Z semantics, correct by m1), add the 4×4 example as a test
with its gp values, and keep the wording of `forms_sentinel_natural_families` ("43 answer") as a lower
bound only where it is not exact.

### m3. The slowdown on scalar-like input grows with n; the documented ratios are only valid at n ≤ 60

Section 1.3 item 3 and the risks give 10× (`invariant_factors`) and "within 1.5×"
(`rational_canonical_form`) at n = 40-60. For r ≈ s ≈ n the new path is Θ(n⁴) while HEAD's Smith step on a
diagonal R is cheap, so the ratio grows like n. Measured (`$R/t7`, `PrimeField` GF(7), `2·I`, n = 100): new
IF 3.5 s (plus presentation) against HEAD 0.18 s (about 20×); new RCF about 6 s (peel 3.5 s + T and
certificate 2.5 s) against HEAD 2.4 s (2.5×). Required: state in section 1.3, in the risks and in the docs of
the public functions that inputs with many small Krylov blocks and many invariant factors (scalar, diagonal
with few distinct values) cost Θ(n⁴) on the new path against about n³ at HEAD, with the n = 100 figures.
(Optional, not required: a shortcut when R is diagonal, or peeling in Krylov coordinates over finite fields,
which the design measured as faster when the entries are small.)

### m4. Port the H26E9 invariant factors here, not only in M1b

CLAUDE.md asks to port handbook examples as tests, and H26E9 is the example of `InvariantFactors` and
`RationalForm`, the functions this unit rewrites. M1b already has the matrix (`M1b design :715-719`, MAGMA's
printed f). On the prototype (`$R/src/h26.rs`): r = 4, three peeling levels, factors `[x+4, x²+4, x²+4]`,
equal to MAGMA's f, HEAD and gp `matfrobenius(a,1)` (`[x^2 + Mod(4,5), x^2 + Mod(4,5), x + Mod(4,5)]`). It
exercises the multi-level path over `PrimeField`, which no other new test with a MAGMA reference does.
Required: add `h26e9_invariant_factors_and_rational_form` (`invariant_factors`, `invariant_factors_peeled`,
`rational_canonical_form` certified, `canonical_matrix` = block companion of MAGMA's f) to section 1.5;
M1b keeps the primary-form parts.

### m5. Section 1.2.7 (Over Q) misses the growth inside one level

"The generators of (F7) are built from the uncorrected v's, so no size compounds over levels" is right, but
within a level each merge applies (f/α)(A) to the current vector, so after several merges the level's vector
is a polynomial in A of the summed cofactor degrees applied to the first candidate. Per prime the absorbed
exponents strictly increase, so the summed degree is O(n·max e) and at most O(n²). This is still a
description, not a bound, so nothing breaks, but the text must not suggest that only divisors of χ_A are
ever applied. Measured on structured r = n inputs (`$R/src/adv.rs`, A = W·Q·J·Q⁻¹·W⁻¹, J Jordan with
eigenvalues 1, 2, 5): n = 30 up to 9 merges and 7.75 s for `invariant_factors` in debug (the design's
table maximum at n = 30 is 3.1 s). Required: reword 1.2.7 and the docs accordingly, and add this n = 30
figure (or a similar one) to section 1.4 so the stated times are not read as a ceiling.

### m6. The coprime-split cap needs the right bound

(F4) states "at most log2(n) + 2 rounds", but the API caps the loop at deg f + 2, which can be smaller than
log2(n) + 2 (deg f = 1, n = 60). The cap is still safe: per prime the loop needs at most
log2(1 + e_f) + 2 gcds (e_f the exponent in f), and log2(1 + e_f) + 2 ≤ deg f + 2. I checked the
iteration count for all e_f, e_a < 200 (worst case well below e_f + 2). Required: state this bound in
(F4) so the cap is justified by the text, and say that the cap counts the gcds of the loop, the final one
included.

### m7. Complexity and timing wording in the docs

* The bound O((r+1)·n³) holds for the first stage only. The fallbacks (Z semantics, rounding fields) keep no
  bound. Section 1.6 must say so in the module docs and in the "Complexity" paragraph of each public
  function, not only in `polynomial_matrix.rs`.
* The proposed `polynomial_matrix.rs` header says "one 2×2 relation matrix took 101 s in debug"; the HANDOFF
  says 81 s and I measured 93.6 s. Use the measured range ("81 to 102 s in debug, depending on load") or
  "about 90 s", consistently in all touched docs.
* The cost sum in 1.2.6 omits the coprime-split term; with gcd degrees at most D_ℓ it is
  O(r·n²·log n), inside the bound. Add it for completeness.

### m8. Test of the HEAD failure for the arrow case has a thin margin

`peeled_merges_arrow_n20` bounds IF + RCF by 10 s; HEAD takes 21.4 s (design measurement), new 0.6 s. That is
fine for the new code, but if the test is meant to show the HEAD failure, say in its comment that the
HEAD time is about 2× the bound (measured under load), so nobody reads a pass on a fast machine as a
regression of the fix. No change of the bound is needed.

## 4. Plan and risks

* The plan order (krylov.rs helpers, then `frobenius.rs`, then `companion.rs`, docs, trackers) builds green
  at each step (the crate has no `deny(warnings)`; unused helpers in step 1 only warn).
* M1b relies on `field_constants`, `linearly_independent`, `mat_vec`, `div_rem_field` and the documented
  contracts of the public functions; all are kept. Moving `poly_at_matrix_times_vector` to
  `krylov::poly_mat_vec` touches nothing M1b uses.
* Only two timing assertions exist in the affected tests (`companion.rs:1416`, 60 s; `:1875`, 15 s); the
  new path is well inside both.
* No new `unsafe`: the prototype's `static mut STATS` must not be copied (the design says so).
* No Decided/Unresolved issue: the algorithm is deterministic and complete over an exact field; failed checks
  are `Err`.

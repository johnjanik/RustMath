# Review 1 of the M1b-matrix-forms design

Reviewer scratch: `$R = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/M1b-matrix-forms/review1`.
Design read in full (753 lines, no Revision sections). HEAD `c79a582`, tree clean.

Verdict: **approve**. No blockers. Five minor items below; none changes an algorithm or an expected value
that the design text specifies.

## 1. What I reproduced

### Defects at HEAD

| Claim | Check | Result |
|---|---|---|
| A: only one-vector `solve`, `solve_left`, `solve_left_with_kernel` | `linear_solve.rs:157`, `:198`, `:218`; grep for `solve_matrix`, `solve_left_matrix`, `solve_left_sequence` over all `.rs` | Confirmed, no hits. Tracker rows `status_overrides.csv:3150-3153` say what the design quotes |
| A: k eliminations cost | rebuilt the designer's `t7.rs` against the same rlibs (`$R/t7.rs`) | n = k = 20: 0.54 s against 1.22 s; n = k = 30: 3.9 s against 11.4 s; same V. Matches the table |
| B: no primary forms | grep `primary_rational\|primary_invariant\|PrimaryRationalForm\|FactorizationField` over `.rs` | No hits. Rows `:3188`, `:3191` "missing" |
| C: dead Jordan sketch, unconstructible struct | `eigenvalues.rs:22-383` one `/* */` block; `jordan_form` doc at `:253`, fn ends `:381`, `}` `:382`, `*/` `:383`; struct `:385-392`; `lib.rs:49` | Confirmed. Deleting `:253-381` leaves the block well formed. All tests in `eigenvalues.rs` are also commented out, so removing `use crate::Matrix;` is safe |
| D: no `is_similar` | grep `fn [a-z_]*similar` | Only `rustmath-combinatorics` similarity-class counting, unrelated |
| Callers | grep `JordanForm\|jordan_form` (all files), `\.solve(` | `JordanForm`: only `lib.rs:49` and the dead code. `Matrix::solve`: `vector_space.rs:144/237/362`, `companion.rs:1086`, `nfctx.rs:243`, `repl.rs:1435`, `matrix_forms_gfp.rs`, the crate's tests. The design's list is complete |

### References

All nine gp scripts in `$S/gp/` rerun with PARI/GP 2.15. The output is byte-identical to the designer's logs
(`$R/*.rerun.log`). The GAP script `pif.g` also reruns identically (`$R/pif.rerun.log`). I checked the
expected values in the test tables by hand against these logs: the multi-RHS values (V, N, X; also by hand
RREF for the GF(3) case), H26E9 (`t*a*t^-1 == r`, PIF order `x+1, x+1, x+4, x+4, x+4`), the GF(3) 7×7 J,
the GF(3) generalised block, the large prime `2^64 + 13` and its factormod order, Sage ex. 1-3, and the
`J₂ ⊕ J₂` vs `J₂ ⊕ 0 ⊕ 0` pair. All agree.

PARI `factor` order over Q (claimed: degree, then the primitive integer coefficients from the top down).
New discriminating cases (`$R/order.gp`): `x^2+2x+5` before `x^2+3x+1` (bottom-up would invert it),
`x-7, x+5, 2x-3, 2x+1`, `2x+5` before `3x+1`, `x^2-2, x^2+2, x^2+x+1, 2x^2+1`. All consistent with the claim.

### Algorithm

Re-derived:

* RCF contract: `M·T = T·C` with C block column-companion forces the columns of block i to be
  `g_i, M g_i, ..., M^{k-1} g_i`, and the last column gives `f_i(M) g_i = 0`, so `ann(g_i) = f_i`
  (the columns are independent). M1a keeps this (its section 1.3 and `certified_form`).
* Primary block basis: `deg(x^t q) ≤ deg f_i - 1` and `deg(x^t p^j q) ≤ deg f_i - 1`, so the coordinates
  are the coefficients. `ann(q(M) g_i) = p^e`. The CRT direct sum holds. The action of x in the basis
  `x^t p^j q` puts all subdiagonal entries 1 and `-c_s` in column `jd+d-1`, rows `jd..jd+d-1`.
  Appendix A indexes this correctly.
* Transposition: `M P = P G` gives `Pᵀ A = Gᵀ Pᵀ`, so `T = Pᵀ`, `F = Gᵀ`. The row layout of 0.2 is right
  (for `(x²+1)²` it gives `[0,1,0,0; -1,0,1,0; 0,0,0,1; 0,0,-1,0]`).
* Preusser Def. 5 (`$S/dl/a.txt:81-95`, `:134-170`): `[P]` is column style, ξ has a 1 at (1, n), in the
  block below the diagonal. That is exactly the design's column-convention G. The citation is accurate.
  The single-1 coupling is valid over any field (separability is not needed), unlike the `I_d` coupling.
* is_similar: `T·P_A = P_B` and the RCF certificates give `T = P_B P_A⁻¹`, invertible, `T A T⁻¹ = B`.
  This matches MAGMA's `t * a * t^(-1) = b`. Equal invariant factors decide similarity exactly. No
  bounded search is involved anywhere. Zassenhaus' cap and Berlekamp's give-up pass through as `Err`.
* Multi-RHS: the RREF pivots in the A block do not depend on B, and "no pivot in a B column" is equivalent
  to "every b_j in col(A)" (take the first inconsistent j). For a consistent B each column equals what
  `solve` returns. The early `break` in `row_echelon_form` (`linear_solve.rs:38`) is harmless.

Stress test of the prototype on structures the design's tests do not cover (`$R/r2.rs`, `$R/r2.log`):
over Q, `(x-1)^{1,2,3} (x+2) (x²+1)^{1,2}` (n = 13, three invariant factors sharing primes with different
exponents), `(x²-2)^{1,2} (x³-2)`, repeated blocks `x², x², (x+1), (x+1)³`; over GF(3), GF(2), GF(7)
mixed linear and quadratic and cubic factors, conjugated by random unimodular matrices. Every run returns
the expected J and primary rational form (exactly, after sorting my expected blocks by the design's rule),
the certificate passes, and `is_similar(A, J)` returns `Some`. Also `J₂(1/2) ⊕ C(x² - 1/2)` over Q and a
1×1 matrix (`$R/r3.log`).

Cost: rebuilt `t2.rs`: n = 8 / 12 / 16 dense Jordan 10 / 68 / 432 ms, n = 16 Jordan structure 66 ms,
`is_similar(A, J)` n = 16 dense 8.8 s (two certificates). Same order as the design's 2.6 and 4.4.

### Sentinels and binding

`PrimeField` and `FiniteFieldElement` both return `Err` from `inverse` of an unbound 2 (`prime_field.rs:493`,
`finite_field.rs:923-937`), so the probe rejects sentinel-only input for both types and never rejects a real
field (2 is 0 or a unit). The `x + z` binding works with HEAD's coercion (`prime_field.rs:357`).
`$R/r1.log`: for bound GF(5) inputs (zero, identity, scalar, `J₃(0)`, a 3-cycle) every invariant factor at
HEAD has bound coefficients, so the "p unknown" error of the `PrimeField` oracle does not fire on them.

### Crate edges and coordination

* rings depends on matrix (`rustmath-rings/Cargo.toml:18`); finitefields has matrix only as a
  dev-dependency; polynomials depends on rationals. So the trait in polynomials with impls for `Rational`
  (polynomials) and `PrimeField` (finitefields) creates no new edge and no orphan problem.
* `zassenhaus::factor` returns primitive factors with positive leading coefficient (`zassenhaus.rs:267-273`),
  and the content of `D·f` is 1 for monic f (D = lcm of denominators). The sort key is well defined.
* M1a keeps `field_constants`, `linearly_independent`, `div_rem_field`, `mat_vec`, the block layout and the
  column structure of `change_of_basis` (M1a design `:269-270`, `:347-354`). F1 makes
  `UnivariatePolynomial::new` bind mixed coefficients (F1 design 7.2), which only helps here. I1/P1 may
  change what `berlekamp::factor` does for huge p; M1b passes it through and the large-prime test uses
  `2^64 + 13`, inside the proven range.
* Line references checked: `krylov.rs:74`, `:234`, `polynomial_matrix.rs:524`, `companion.rs:466`, `:564`,
  `Chapter_23_...md:294` (Factorization), `:313` (CompanionMatrix: superdiagonal 1s, last row),
  `Chapter_26_...md:222-225`, `:299-312`, `Chapter_83_...md:321-326`, tracker rows, `PROJECT_TRACKER.md:940/959/979`.

## 2. Findings

### M1 (minor). The prototype's `Rational` sort contradicts the design text and its own test

`exp/forms.rs` sorts the factors by the **monic rational** coefficients from the top down. The design text
(2.3) sorts by the **primitive integer** coefficients from the top down (PARI order), and the test
`rational_factorization_pari_order` expects `[(x-3,1), (x+3,1), (x-1/2,1), (x²+1,2)]`. Measured
(`$R/r3.log`): the prototype returns `[x-3, x-1/2, x+3, (x²+1)²]`. The two keys agree on every other test
in the design (all those factors are monic over Z), which is why the prototype runs did not show it.
Appendix A calls the prototype "the core"; an implementer who copies the `Rational` impl from `forms.rs`
gets a failing test and a non-PARI order.
Required change: state in 2.3 that the sort compares the primitive integer factors as returned by
`factor_over_integers` (positive leading coefficient), before division by the leading coefficient, and
that `forms.rs` uses a different key and must not be copied there.

### M2 (minor). Make the `PrimeField` oracle independent of how the invariant factors were built

`FactorizationField for PrimeField` returns `Err` when every coefficient is an unbound sentinel. At HEAD
the invariant factors of bound matrices are bound (`$R/r1.log`), and F1 will keep polynomials uniformly
bound, but this is an assumption about M1a's and F1's internals. The matrix side knows the bound `z`.
Required change: in `primary_blocks` (and in `primary_invariant_factors`), map the coefficients of `f_r`
(and of the `f_i` used for the valuations) through `c ↦ c + z` before calling the oracle. One line, and
the "p unknown" error then only fires for genuinely field-less input, which the probe already rejects.

### M3 (minor). Odd composite `PrimeField` moduli pass the probe

`PrimeField::new` does not check primality. For modulus 9 the probe passes (2 is a unit). The forms are
safe (Berlekamp rejects 9), but `is_similar` returns an answer computed in Z/9. It is out of contract, but
the doc should say so.
Required change: one sentence in the `is_similar` docs (and in the probe error text's neighbourhood):
"for a `PrimeField` with a composite modulus the result is unspecified; the probe only detects even
composite moduli and sentinel-only input".

### M4 (minor). Cost wording in 4.2

"the second check `T·A == B·T` would add about 35% to the cost on Q". From `t3.log`: inv 0.22 + RCF 0.36 +
solve 2.08 + `T·P_A` check 2.98 = 5.6 s; `T·A == B·T` is 3.36 s more, so it adds about 60% to the
one-certificate cost (it is about 37% of the two-certificate total, which is what 4.4's "saves about a
third" says).
Required change: reword 4.2 to "would add about 60% (3.4 s on 5.6 s, n = 16 dense)".

### M5 (minor). Test table detail

`solve_left_matrix_shapes` lists "W 0×2" with expected `Some(0×3)` but does not name the A (it must be
3×2, while the first row of the same cell uses a 3-column A).
Required change: name the A of each shape case in the table (as `$S/exp/t5.rs` does).

## 3. Items covered

| HANDOFF item | Covered by | Out of scope claims |
|---|---|---|
| multi-RHS `IsConsistent` (matrix W and sequence Q), `Solution` | Item A, five methods, one elimination | Euclidean rings (Z) left as a follow-up: MAGMA allows them, the existing one-vector forms are fields only too; reasonable |
| `PrimaryRationalForm`, `PrimaryInvariantFactors` | Item B, Q and GF(p) via `FactorizationField` | GF(p^n) impls as a follow-up; the task asks for Q and GF(p) |
| `JordanForm` (split and generalised) | Item C, certified | MAGMA's d ≥ 2 layout cannot be checked (no MAGMA access); documented, and the chosen layout is the standard one valid over any field |
| `IsSimilar` with transform | Item D, exact for any field | none |
| Coordinate with M1a | Section 0.3, consistent with the M1a design | none |

No facade, no silent truncation, no unsafe. The plan's order (polynomials, finitefields, matrix solve,
matrix forms, GF tests, trackers) builds green at every step; `cargo check --workspace --tests` covers the
changed `JordanForm` fields and the new re-exports (no workspace user of either).

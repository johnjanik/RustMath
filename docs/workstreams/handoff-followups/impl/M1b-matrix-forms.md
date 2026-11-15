# M1b-matrix-forms: implementation report

Unit M1b of the HANDOFF follow-ups: multi-RHS `IsConsistent` / `Solution`, `PrimaryInvariantFactors`,
`PrimaryRationalForm`, `JordanForm`, `IsSimilar`. Design: `design/M1b-matrix-forms.md` (no Revision
sections; review 1 approved it with five minor notes). Base: HEAD `d3a133b` (M1a committed). The tree
was clean at the start, so there was no interrupted attempt. Nothing is committed (the orchestrator
commits). Work files (timing logs, stress driver): `impl/M1b-work/`.

## Handoffs read

* **M1a** (`impl/M1a-matrix-rcf-bound.md`, Handoffs): the contracts M1b uses are kept
  (`krylov::field_constants`, `krylov::linearly_independent`, `polynomial_matrix::div_rem_field`, the
  documented `invariant_factors` / `rational_canonical_form` behaviour, `change_of_basis` columns
  `g, A·g, ...` block by block in increasing factor order). Used as stated; no rename to follow.
  `poly_at_matrix_times_vector` (now `krylov::poly_mat_vec`) is not used: the columns are read off
  `P_R` as the design says, with no evaluation at M.
* **F1** (Handoffs): the container binding of `Matrix<R>` by `Ring::bind_to_parent_of` is not in this
  unit's design. M1b binds its own inputs by `x ↦ x + z` (design section 5), as M1a did. F1's
  `UnivariatePolynomial::new` binding is used implicitly (bound coefficients stay bound).
* **R1**: `MatrixAlgebra::is_full` is informational; not used.
* **I1**: `berlekamp::check_prime` now proves primality for every size (`try_is_prime`), so the
  design's remark "2^64 + 13 is inside the proven range" is moot: every prime is proven now.
* **P1**: `factor_over_integers` now returns factors in PARI order already. The `Rational` impl still
  sorts (same key, stable), as the design asks, so it does not depend on that.

## Items

### Step 1. `FactorizationField` (rustmath-polynomials)

* New `rustmath-polynomials/src/factorization_field.rs` (228 lines): header (MAGMA ch. 23 §23.8.1, ch. 26
  §26.12.2 precondition; Zassenhaus 1969), trait `FactorizationField: Field` (`:28`) with
  `factor_monic_polynomial`, public helpers `check_monic` (`:46`) and `check_product` (`:69`), and
  `impl FactorizationField for Rational` (`:99`): lcm of denominators, `factor_over_integers`, drop the
  content, sort the **primitive integer** factors by (degree, coefficients top-down as integers), then
  divide by the leading coefficient, multiply back (review M1 folded in: the prototype's monic-rational
  key was not copied).
* `src/lib.rs`: `pub mod factorization_field;` (`:11`), `pub use factorization_field::FactorizationField;`
  (`:44`).
* Tests (4): `rational_factorization_pari_order` (gp `factor((x-3)*(x+3)*(2*x-1)*(x^2+1)^2)`),
  `rational_factorization_charpoly` (gp `factor(x^4 - 11*x^3 + 42*x^2 - 64*x + 32)`),
  `rational_factorization_irreducible_with_denominator` (extra; gp `factor(x^2 + 1/2)` =
  `Mat([2*x^2 + 1, 1])`), `rational_factorization_rejects`.

### Step 2. `FactorizationField for PrimeField` (rustmath-finitefields)

* New private module `rustmath-finitefields/src/prime_field_factor.rs` (148 lines), impl at `:30`;
  `src/lib.rs:24` `mod prime_field_factor;`. p from the bound coefficients; sentinel-only input is
  `Err(InvalidArgument("... the prime p is unknown ..."))`; two different moduli are
  `Err(InvalidArgument)` (extra check, cheap); `berlekamp::factor` on the `value()` integers (prime
  check, PARI `factormod` order); unit must be 1 (`InvalidOperation` otherwise); factors built with
  `PrimeField::new(c, p)` (all bound); multiply back with a bound one.
* Tests (3): `prime_field_factorization_matches_factormod` (gp `lift(factormod(x^5+x^4+x+1, 5))`,
  `lift(factormod((x+1)^2*(x^2+2), 5))`), `prime_field_factorization_mixed_coefficients` (extra; gp
  `lift(factormod(x^2+1, 5))`), `prime_field_factorization_rejects` (p unknown; mod 15; zero, constant,
  non-monic).

### Step 3. Multi-RHS solvers (rustmath-matrix `linear_solve.rs`)

* Module header (`:1-34`): table of the four MAGMA forms and their methods, the one-elimination
  algorithm (Hoffman-Kunze §1.4), cost with the new measurement, "fields only".
* `solve` (`:175`) now builds the m×1 matrix of b and calls the shared `solve_columns` (`:191`): same
  augmented matrix, same pivots, `F::zero()` for free variables, so identical output (all 9 existing
  `solve_*` tests and `matrix_forms_gfp::solve_gf7` unchanged and green).
* New: `solve_matrix` (`:236`, A·X = B), `solve_left_matrix` (`:295`, MAGMA `IsConsistent(A, W)`),
  `solve_left_matrix_with_kernel` (`:319`, `Solution(A, W)`), `solve_left_sequence` (`:337`,
  `IsConsistent(A, Q)`, error names the index), `solve_left_sequence_with_kernel` (`:360`,
  `Solution(A, Q)`). `solve_left`'s doc points to them.
* Tests (7, `linear_solve.rs:968-1100`): `solve_left_matrix_two_rhs`, `..._with_kernel_two_rhs`,
  `..._one_bad_row`, `..._unique`, `solve_matrix_column_convention`, `solve_left_sequence_matches_rows`,
  `solve_left_matrix_shapes` (review M5 folded in: each shape case names its A). References: gp
  `matinverseimage(A~, W~)~`, `matker(A~)`, `W*A^-1`, `matinverseimage(A, B)` (commands in comments).
* The file was not rustfmt-clean at HEAD; I ran rustfmt on it, which also joined three pre-existing
  wrapped lines (whitespace only, `:106`, `:140`, `:375` region).

### Step 4. Primary forms (rustmath-matrix `primary_forms.rs`, new, 949 lines)

* Module docs: MAGMA ch. 26 §26.12.2 and ch. 83 §83.11.2 digest lines; [Ste97] named and stated as not
  implemented; the algorithm (Dummit-Foote §12.1-12.3, Jacobson BA I §3.9-3.10), the certificate, the
  conventions of design 0.2 in full (row convention, MAGMA `CompanionMatrix` layout, generalised Jordan
  block with Preusser arXiv:2006.06502 Def. 5 transposed, the `d >= 2` caveat, block order, "multiplicity"
  = exponent), binding and the field probe, cost.
* Public: structs `PrimaryRationalForm` (`:109`), `JordanForm` (`:121`) with fields
  `primary_invariant_factors`, `canonical_matrix`, `transform`; `primary_invariant_factors` (`:460`),
  `primary_rational_form` (`:492`), `jordan_form` (`:538`, doc test = Sage `a`), `is_similar` (`:576`);
  `pub use rustmath_polynomials::FactorizationField`.
* Private: `field_of` (probe), `bind`, `bind_poly`, `prepared`, `poly_pow`, `exact_div`,
  `primary_blocks` (`:218`), `primary_columns` (`:300`), `primary_core` (`:333`), `magma_form` (`:433`),
  as in the design's Appendix A. G is built by hand from the bound `z`, `one` (no `companion_matrix`).
* `primary_blocks` binds the coefficients of every `f_i` by `c ↦ c + z` before the oracle and the
  valuations (review M2 folded in). It validates the oracle (monic, degree >= 1, multiplicity >= 1,
  pairwise distinct), checks each `f_i / Π p^e = 1`, and also (extra) that the valuations of `f_r` equal
  the oracle's multiplicities.
* Certificate in `primary_core`: `M·P == P·G` and `linearly_independent(columns of P)`; failure is
  `Err(InvalidOperation("internal: ..."))`. `is_similar`: equal invariant factors decide; T from
  `P_A.solve_left_matrix(P_B)`, certified by `T·P_A == P_B`.
* Docs: review M3 folded in (composite odd `PrimeField` modulus: unspecified, module docs and
  `is_similar` docs); review M4 folded in (the "about 60% (3.4 s on 5.6 s)" wording, as a code comment
  at the certificate in `is_similar`, labelled as the design prototype's measurement).
* `eigenvalues.rs`: the commented-out `jordan_form` sketch (HEAD `:253-381`) and the struct `JordanForm`
  (HEAD `:385-392`) are deleted; `use crate::Matrix;` removed; module doc points to
  `primary_forms::jordan_form`. The rest of the comment block stays.
* `lib.rs`: `pub mod primary_forms;`; `pub use eigenvalues::{EigenDecomposition, Eigenvector};`;
  `pub use primary_forms::{is_similar, jordan_form, primary_invariant_factors, primary_rational_form,
  FactorizationField, JordanForm, PrimaryRationalForm};`.
* Tests (12 + doc test): `g_is_u_c_u_inverse` (extra; gp `U*C*U^-1 == G` = 1, `matdet(U)` = 1),
  `primary_invariant_factors_q`, `primary_rational_form_q`, `primary_forms_agree`, `primary_forms_shapes`
  (plus a 1×1 case), `jordan_form_sage_examples` (plus Sage's `P = T⁻¹` convention),
  `jordan_form_generalized_block_q`, `jordan_form_sage_similarity_matrices`, `is_similar_sage_examples`,
  `is_similar_same_char_and_min_poly`, `is_similar_to_own_forms`, `is_similar_shapes` (plus 1×1 cases).
  References: gp `matfrobenius(A,1)` and `factor`, `matrank((A-l)^k)`, `T*A*T^-1 == F`, Sage docs
  (`matrix2.pyx` `jordan_form`, `is_similar`), GAP `pif.g` multisets (design `gp/`); the commands are
  in the test comments.

### Step 5. GF(p) tests (`rustmath-finitefields/tests/matrix_forms_gfp.rs`)

* Header lists §26.8 (vector, matrix and sequence forms) and the four §26.12.2 intrinsics; imports.
* Tests (8): `solve_left_matrix_gf3_all_solutions` (gp `gf3_multi.gp`: V, `matker(A~)`, brute-force
  counts 9 and 9, `[;]`), `h26e9_magma_handbook_example` (MAGMA's printed output, gp `h51e8.gp`, GAP),
  `jordan_form_gf3_mixed_blocks`, `generalized_jordan_gf3`, `jordan_form_large_prime` (p = 2^64 + 13,
  gp `nextprime(2^64)`, `factormod` order), `primary_forms_reject_unbound_entries`,
  `primary_forms_bind_mixed_entries` (gp `matfrobenius(Mod([2,1;0,2],2),1)` = `[x^2]`),
  `is_similar_gf5_negative`. All transforms are checked by property only.

## API and behaviour changes, callers checked

* New public API only, except `rustmath_matrix::JordanForm`, which changes from `{ j, p }` (no
  constructor anywhere) to the MAGMA struct. Workspace grep for `JordanForm|jordan_form`: only
  `lib.rs` and the dead code used it; no user of the fields `j`, `p`.
* `Matrix::solve`: same output. Callers (`vector_space.rs`, `companion.rs` test,
  `rustmath-rings/src/dedekind/nfctx.rs`, `rustmath-jupyter/src/repl.rs`, `matrix_forms_gfp.rs`) need
  no change; `cargo test -p rustmath-rings --lib dedekind` green (26 passed).
* New names collide with nothing (grep for `solve_matrix`, `solve_left_matrix`, `solve_left_sequence`,
  `FactorizationField`, `is_similar(`: only the new code).
* New public helpers `rustmath_polynomials::factorization_field::{check_monic, check_product}` (not in the
  design; shared by the two impls).

## Deviations from the design (with evidence)

1. `check_monic` / `check_product` are public in rustmath-polynomials, so the finitefields impl does not
   copy them. Small public API addition.
2. Extra validations, all failing as `Err`, never changing a value: two different moduli in the PrimeField
   impl; Berlekamp's unit must be 1; the valuations of `f_r` must equal the oracle's multiplicities; an
   exact division must lower the degree.
3. Costs: the design's numbers were measured at c79a582 with the prototype. M1a's peeled RCF is faster,
   so the docs carry new measurements of this implementation (debug, this machine;
   `impl/M1b-work/timing_q.log`, `timing_gfp.log`). Over Q, dense n = 8 / 12 / 16: RCF(Aᵀ) 2 / 12 / 49 ms,
   Jordan 4 / 19 / 72 ms, `is_similar(A, J)` 16 / 137 / 610 ms (T entries 29 / 73 / 135 digits); n = 16
   with six Jordan blocks: RCF 59 ms, Jordan 80 ms, `is_similar` 0.42 s. GF(101), GF(10⁹+7),
   GF(10²⁴+7), dense n = 20: Jordan 0.07 / 0.09 / 0.73 s, `is_similar` 0.10 / 0.10 / 0.34 s. Multi-RHS
   n = k = 20: 59 ms against 224 ms; n = k = 30: 0.35 s against 1.6 s (same V). The design's 2.6 table
   (4 / 38 / 212 ms RCF, 9 / 59 / 356 ms Jordan) is superseded.
4. Binding uses `x ↦ x + z` as designed, not F1's `Ring::bind_to_parent_of`; both work, the design's
   choice is generic over every `Field`.

## Review notes (review 1)

All five folded in: M1 (sort key: primitive integer factors), M2 (bind the invariant factors before the
oracle), M3 (doc sentence on composite moduli), M4 (60% wording, as a comment at the certificate), M5
(test names its A per shape case). None skipped.

## Extra verification (not committed)

`impl/M1b-work/m1b_stress_tmp.rs`, copied into `rustmath-finitefields/tests`, run and deleted: 48 random
structures over GF(2), GF(3), GF(5), GF(7) (direct sums of `C(p^e)` for linear, quadratic and cubic
irreducible p, e <= 3, n <= 14, conjugated by random unimodular matrices). For each: the pairs equal the
constructed multiset, the three pair lists agree, both transforms satisfy `T·A = F·T`, and
`is_similar(A, C)` and `is_similar(J, F)` return certified transforms. Log `stress_gfp.log`: 48 runs, pass.

## Acceptance commands

Logs in `impl/M1b-work/`.

| Command | Passed | Failed | Ignored |
|---|---|---|---|
| `cargo test -p rustmath-polynomials --no-fail-fast` | 438 (+4 new) | 0 | 4 |
| `cargo test -p rustmath-finitefields --no-fail-fast` | 329 (+3 lib, +8 in `matrix_forms_gfp`: 19 -> 27) | 0 | 0 |
| `cargo test -p rustmath-matrix --no-fail-fast` | 331 (was 311 after M1a: +7 `linear_solve`, +12 `primary_forms`, +1 doc test) | 0 | 1 |
| `cargo test -p rustmath-rings --lib dedekind --no-fail-fast` | 26 | 0 | 0 |
| `cargo check --workspace --tests` | exit 0 | | |
| `cargo clippy -p rustmath-matrix -p rustmath-polynomials -p rustmath-finitefields` | exit 0, no warning in a touched file | | |

With `--tests` added, clippy reports nothing in the new code; the only hits in touched files are
pre-existing test code (`linear_solve.rs:732`, `:797` needless range loops; `eigenvalues.rs:259-260`
unused imports of the commented-out tests, `:396-397` at HEAD). No release build was made; no
`unsafe`. New files are rustfmt-clean; `matrix_forms_gfp.rs` was rustfmt-clean at HEAD and still is.

## Left undone

* Full `cargo test --workspace` (the orchestrator runs it).
* Trackers and docs (design Plan step 6): not edited by instruction; see Handoffs.

## Handoffs

* **Trackers (orchestrator; design Plan step 6, line numbers at HEAD `d3a133b`)**:
  * `PROJECT_TRACKER.md:951` (QR / power iteration / eigenvectors / Jordan in the dead block): the
    Jordan form is no longer there (now `rustmath-matrix/src/primary_forms.rs:538`), the struct
    `JordanForm` is the new constructible one; the row keeps the QR/power-iteration status.
    `:970` (linear solve): add `solve_matrix` (`linear_solve.rs:236`), `solve_left_matrix` (`:295`),
    `solve_left_matrix_with_kernel` (`:319`), `solve_left_sequence` (`:337`),
    `solve_left_sequence_with_kernel` (`:360`), one elimination for all right-hand sides, and the tests
    `solve_left_matrix_two_rhs`, `solve_left_matrix_with_kernel_two_rhs`, `solve_left_matrix_one_bad_row`,
    `solve_left_matrix_unique`, `solve_matrix_column_convention`, `solve_left_sequence_matches_rows`,
    `solve_left_matrix_shapes`, `matrix_forms_gfp::solve_left_matrix_gf3_all_solutions`; covers
    ch026.IsConsistent.2, ch026.Solution.2; row line refs `:157/:198/:218` are now `:175/:255/:275`.
    `:990` ("Jordan canonical form (working)" missing bullet): remove. New 🟢 row: "Primary rational
    form, generalised Jordan form, primary invariant factors, similarity with transform (from the
    certified rational form and the factored minimal polynomial)", `rustmath-matrix/src/primary_forms.rs:333`
    (`primary_core`), tests as listed in Step 4 and Step 5, stated limits: generalised Jordan layout for
    `deg p >= 2` unverified against MAGMA, block order over Q is PARI's (MAGMA's undocumented),
    `FactorizationField` for Q and GF(p) only. New row for `FactorizationField`
    (`rustmath-polynomials/src/factorization_field.rs:28`, impls `:99` and
    `rustmath-finitefields/src/prime_field_factor.rs:30`). Change Log entry.
  * `docs/magma/status_overrides.csv`: `ch026.IsConsistent` (`:3150`), `ch026.Solution` (`:3152`) stay
    `partial` (fields only), notes name the matrix and sequence forms and the new line numbers;
    `ch026.IsConsistent.2` (`:3151`), `ch026.Solution.2` (`:3153`) become `partial` (fields only);
    `ch026.PrimaryRationalForm` (`:3188`), `ch026.JordanForm` (`:3189`), `ch026.PrimaryInvariantFactors`
    (`:3191`) become `partial` ("Q and GF(p) through FactorizationField; MAGMA allows any field with
    factorization; generalised Jordan layout for degree >= 2 unverified"); `ch026.IsSimilar` (`:3193`)
    becomes `done` (any exact field); add the four `ch083.*` rows (`intrinsics.csv:7207`, `:7208`,
    `:7210`, `:7212`) with the same statuses. Then `python3 tools/magma_tracker/build.py`.
  * `docs/HANDOFF.md:385`: drop "the multi-RHS `IsConsistent`, `PrimaryRationalForm`, `JordanForm`,
    `IsSimilar`". Add the follow-ups (design 7.4): `FactorizationField` impls for `FiniteFieldElement`
    (GF(p^n), through `poly_factor::factor`) and for the rings `GFp<P>`; `FactorableConstantField`
    (rustmath-rings `function_field/typed/factor.rs`) can become an alias of `FactorizationField`; a
    MAGMA-convention `RationalForm` (`T·A·T⁻¹ = Cᵀ`) is now a few lines (`rational_canonical_form(&Aᵀ)`,
    transposed), for the rational-form owner; multi-RHS `IsConsistent` over Euclidean rings (Z); Steel's
    [Ste97] algorithm itself is not implemented (the forms come from the rational form).
* **F1's container hand-off stays open**: `Matrix<R>` still does not bind unbound constants; M1b only
  binds its own inputs. The multi-RHS solvers, like `solve`, return unbound `F::zero()` for free
  variables over `PrimeField` (equal to the bound zero under the coercing `==`; unchanged behaviour).
* **J1a/J1b (REPL)**: nothing required. The REPL could expose `jordan_form`, `is_similar` later.
* **Pre-existing, not this unit** (from M1a): `clippy::erasing_op` in `integer_forms.rs` tests makes
  `cargo clippy -p rustmath-matrix --tests` fail; the acceptance clippy command has no `--tests`.

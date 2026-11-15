# MF1-manifolds: review 2 (independent, adversarial)

Design reviewed: `fu2/design/MF1-manifolds.md` (1062 lines, with "Revision 1"). HEAD `c79a582`, tree clean.
Scratch: `$R = .../fu2/design/MF1-manifolds/review2/`. Programs are built with the designer's `../build.sh`
against the HEAD debug rlibs (the rlibs date from Nov 12 and the tree is clean, so they are HEAD). Binaries
were deleted afterwards; outputs are kept as `.txt`.

Verdict: **not approved**. One blocker: the new exterior derivative, and the new three-valued
`closedness` / `vanishes(dω)` / `is_closed` decisions, inherit the derivative facades of
rustmath-symbolic and return fabricated values and false decisions. The fix is small and local. One
further cost hole in the reference prototype (uncharged negation) must be closed, and some minor points.
The rest of the design is sound and well evidenced, and both blockers of review 1 are fixed.

## What I reproduced and found to hold

* HEAD defects (`$R/repro_head`, `repro_head3`, `repro_head4`, rebuilt from the designer's sources):
  d(xy dx + x^2 dy) gives `[(y + x), (2 * x), 0, 0]`; dx∧dx gives `(1 * 1)` nodes; the trace of diag(5, 7)
  is `[1]`; the (0,1)⊗(1,0) product is `[3, 6, 4, 8]`; panics `j % 0` (`diff_form.rs:151`), multiply
  overflow (`tensor_field.rs:158`, `:105`), add overflow (`diff_form.rs:132`, `:182`); rank 2^32 + 1
  gives 2 components; `is_closed` of the closed form x1 dx1∧dx2 + dx3∧dx4 is Ok(false), also through
  `SymplecticForm`; `ContactForm::new(dz)` is accepted; `apply_to_form(dA)` has degree 1;
  `commutes_with_exterior_derivative(A)` is Ok(false). All as stated.
* HEAD cost: d(d(dx_0)) at n = 100 is 1.46 s, 444 MB (`measure_head d2 100`). Prototype
  (`measure_new2`): d(dx_0) n = 1000 0.13 s / 66 MB; d(d(dx_0)) n = 100 0.11 s / 66 MB; dx_0∧dx_0
  n = 1000 0.20 s / 66 MB; d∘d of x_{i+1}x_{i+2} dx_i at n = 60 0.03 s, all literal 0; d of the 2-form
  x_i x_j at n = 100 0.48 s. The claims of 1.9 hold for these inputs.
* Review 1 B1 (reserved names, infinities): `t_b1` output is identical to the designer's `b1.txt`.
  I re-derived the soundness argument of 1.6.4. The induction (e = N/D wherever no met denominator
  vanishes) is right, no denominator is ever cancelled (so an opaque atom met in a denominator stays in
  V unless N = 0), and the single-named-constant rule follows from transcendence as stated.
  `simplify` never folds `Real` values (`simplify.rs` has no `Real` arm), so the exact-binary-value
  semantics are not undermined by float rounding in `normalize`.
* Review 1 B2 (time): a sample of `sweep.txt` reproduces (univ 50 64: 0.032 s; prod 12: 0.088 s;
  multi 30 3: 0.092 s; ratf 100: 0.069 s; bigu 60 2040: 0.040 s). `MultivariatePolynomial::add` inserts
  the right operand into the left, so the "smaller into larger" charge is right.
* References: `gp -q refs.gp` prints every value quoted in 1.8 and 3.3. I re-derived dA, dB, A∧B, the
  2D wedge and d, a∧b (1-form by 2-form), i_X(dx∧dy) = (xz, yz, 0), X(f) and the ω^k and contact signs
  by hand; the Sage strings quoted in 1.8 expand to the gp values.
* Callers: `git grep` over the workspace agrees with 3.4. Only rustmath-jupyter depends on the crate
  (rustmath-py is excluded from the workspace and does not use it). No `match` on `ManifoldError`.
  `to_flat`, `total_rank`, `num_components` (TensorField) have no callers outside the crate besides
  `lie_derivative.rs:225`.
* REPL (item 3): the line numbers of 4.1 and 4.4 are right (`:4608`, `:4649-4650`, `:4686-4687`,
  `:10548`, `:10549-10552`, `:10553`, `:10869-10902`, `:12204-12209`, `:12345-12389`, `:12393-12426`;
  the stale comments at `:4651`, `:4689`, `:10884`). Only `test_form_component_limits` `:12359-12366`
  reaches the new chart check; `exterior_derivative(fs, CS)` errs on the lookup first; the `Cm` lines use
  the p >= n and p + q > n shortcuts. The 4.3 replacement block works against the arms as written (the
  REPL chart is named `user_chart`, the components are found, then the dimension check fires and the
  arm maps it to ComputationError). J1a (§5, the R^3 block restored with `chart(E3, "x,y,z")`) and J1b
  (§6 table rows for `FORM_COMPONENTS`, `DIFF_FORM_DEGREE`) already consume 4.3 and 4.4.
* Citations exist: Chapter_24 §24.4.2 and §24.4.7, Chapter_41, Chapter_42 §42.15, Chapter_82 §82.3,
  Chapter_111. The paper citations (Buchberger-Loos 1982, Richardson 1968, Lindemann 1882, Hermite
  1873, Lee 2013 ch. 12 and 14) are correct.
* Coordination: S1 (before MF1) adds `impl Drop for Expr` (E0509 on by-value matches; the prototype
  matches on references) and keeps `simplify` results unchanged; I3 makes `Integer::to_i64` checked (the
  design already uses `NumericConversion::to_i64`). Neither breaks the plan.

## Blocker

### B1. d returns fabricated derivatives, and closedness certifies them

`rustmath-symbolic/src/differentiate.rs` is not the derivative on several node kinds. It returns
`Expr::from(0)` for every `Function(name, args)` other than `root_sum` (`:230-234`, tracker 🛑
`PROJECT_TRACKER.md:2106`), for `Binary(Mod, ..)` (`:90`), for `Unary(Gamma | Factorial | Zeta, ..)`
(`:184-188`, `:191`, `:204`), whatever the arguments, and it uses the rational approximation 1000/886
of 2/sqrt(pi) for `erf` (`:194-198`). The new d (1.4 step 6) differentiates every component with it, and the
correctness argument ("step 6 computes the formula for every increasing I") assumes it is correct.
`normalize` then stores a literal 0, `vanishes` reads it as decided, and `closedness` returns `Zero`,
`is_closed` Ok(true). These are new decisions that the design presents as proofs (3.3).

Reproduced with the revised prototype (`$R/r_dfacade.rs`, output `$R/dfacade.txt`), ω = f dy on R^2,
true (dω)_{xy} = ∂f/∂x, nonzero in every row:

| f | prototype (dω)_{xy} | consequence |
|---|---|---|
| f(x) (user function), bessel_j(0, x) | literal 0 | closedness Zero, is_closed Ok(true) |
| gamma(x), zeta(x), factorial(x) | literal 0 | closedness Zero, is_closed Ok(true) |
| x mod 3 | literal 0 | closedness Zero, is_closed Ok(true) |
| erf(x) | `500/443 * exp(-(x ^ 2))` (true 2/sqrt(pi) exp(-x^2)) | a wrong value stored as a component |

The same false decisions reach `SymplecticManifold::new` (a non-closed form with such components is
accepted as closed), `DeRhamCohomology::is_closed`, `is_contact` (dα = 0, so a contact form is refused),
`commutes_with_exterior_derivative`, and the Lie derivative through Cartan. CLAUDE.md: a facade is a
defect, missing functionality returns `Err`, and certifying routines must not claim decisions they do
not have. S1 handles the same differentiate facade by refusing such input (`Unsupported` for gamma, zeta,
factorial and named functions of the variable, S1 §7 item 3), so the project pattern is a refusal.

Required change (small, local to rustmath-manifolds):

1. In `diff_form.rs`, a crate-private check `derivative_supported(e: &Expr, var: &Symbol) -> bool`
   (iterative scan, explicit stack): false when `e` has a subterm that contains `var` and is one of
   `Function(name, args)` other than `root_sum` with 2 arguments, `Unary(Gamma | Factorial | Zeta |
   Erf, _)`, or `Binary(Mod, _, _)`. A subterm that does not contain `var` is differentiated correctly
   (to 0) and is allowed. `Sign` (0 almost everywhere) and `Abs` (sign(f) f') are correct where defined.
2. d (1.4 step 6): before `w_J.differentiate(&symbols[i_kk])`, if the check fails, return
   `Err(ManifoldError::NotImplemented(format!("the derivative of {} with respect to {} is not
   implemented in rustmath-symbolic (differentiate.rs; PROJECT_TRACKER.md:2106)", w_J, var)))`. Every
   caller of d (interior product is unaffected, Lie derivative, closedness, is_closed, contact,
   symplectic, de Rham, Stokes) then propagates the error instead of a fabricated 0.
3. The same check in `LieDerivative::apply_to_one_form` (`lie_derivative.rs:130-176`), which keeps
   HEAD's explicit formula for p = 1 and also calls `differentiate` on ω_i and X^j. Otherwise
   `commutes_with_exterior_derivative` and `apply_to_form` keep fabricated terms.
4. Tests (new, in `diff_form.rs`): `test_d_refuses_unsupported_derivatives`: on R^2, d of f(x) dy,
   bessel_j(0, x) dy, gamma(x) dy, zeta(x) dy, factorial(x) dy, (x mod 3) dy, erf(x) dy is
   Err(NotImplemented) each, and `closedness` / `is_closed` propagate the Err; d of f(y) dy is Ok with
   (dω)_{xy} literal 0 (no x in the argument, and ω_x = 0 is constant, so no derivative in y is taken).
   On HEAD each of these d calls returns Ok (fabricated), so the test fails on HEAD. One test for
   `apply_to_one_form` with X = ∂_x and ω = f(x) dy: Err(NotImplemented).
5. Correctness argument of 1.4: state the precondition that `differentiate` is the true derivative on
   the node kinds the check admits. Section 5 and the HANDOFF entry: "rustmath-symbolic `differentiate`
   returns 0 for named functions, Mod, gamma, zeta, factorial and approximates erf; MF1's d refuses such
   components; wiring the FunctionRegistry derivative hooks (tracker :2106) lifts the refusal".

## Minor issues (required changes where stated)

1. **Negation is not charged in the reference prototype.** The table of 1.6.3 charges "negation ... its
   number of terms", but `proto2.rs` negates for free in the `Unary(Neg)` arm (`negp(r.num)`) and in
   `rf_add` for a subtraction (`negp(b.num)`). Each negation rebuilds the `BTreeMap` of the polynomial.
   Nested negations around a product of 10^4 terms (`$R/r_adv.rs neg`, `$R/neg.txt`; a tree of about
   600 nodes, 51396 units charged):

   | depth | time (debug) |
   |---|---|
   | 0 | 0.107 s |
   | 50 | 3.42 s |
   | 100 | 6.60 s |
   | 197 | 12.87 s |

   That is 128 times the "at most 0.1 s" of 1.6.3 and 1.9. `normalize` is protected in practice, because
   `simplify` folds `-(-x)` to `x` (`simplify.rs:623-626`), but `zero_test` is public and the
   antisymmetry check of `from_components` (3.1) runs it on unsimplified user input; right-nested
   `0 - (0 - (0 - P))` hits the `rf_add` path the same way. Required: charge every polynomial negation by
   its number of terms before it runs (or carry a lazy sign in `RF` and negate once), say in 6.1 step 3
   that `proto2.rs` lacks this charge, and add to `test_zero_test_budget`: 197 nested `Neg` around
   (a0 + ... + a99)(b0 + ... + b99) is `Unresolved` within 1 s (it fails on the prototype: 12.9 s).
2. **The pushforward has the same facade as the pullback the design fixes.**
   `PushForward::apply_to_contravariant_tensor` for p >= 2 (`maps.rs:442-494`, the copy at `:485-487`) computes the Jacobian and
   returns the source components unchanged on the target. The design turns the identical pullback code
   (`maps.rs:388-412`) into `Err(NotImplemented)` but does not mention this one. Either give it the same
   two-line treatment (no caller in the workspace) or list it in section 5 and the HANDOFF. Also note in
   section 5 that `SymplecticForm::evaluate` / `evaluate_on_vectors` (`symplectic.rs:273-301`) read the
   same `matrix_at` facade as `is_non_degenerate`, and that the doc of `SymplecticForm::new`
   (`symplectic.rs:164`) claims checks it does not make.
3. **`tracker_08_09_tests.rs` is not compiled.** No `mod` in `lib.rs` includes it (`git grep
   tracker_08_09` finds only the file). 3.4 says its tests "keep passing"; they never run. Say so (and
   add it to section 5; its `test_lie_derivative_one_form` asserts a structural zero that would fail).
4. **Other unchecked products of the N4 class** in the crate, not listed in section 5:
   `almost_complex.rs:160`, `finsler.rs:153`, `:300`, `symmetries.rs:110`, `symplectic.rs:258`
   (`dim * dim`, `dim * dim * dim`). Add them to the `spin.rs` bullet.
5. **`test_d_chart_dimension` construction.** The lookup precedes the dimension check (1.4 steps 3
   and 4), so the form's components must be stored under the name of the 0-coordinate chart (for
   example built with `from_components(m, &Chart::new("c", 0, ..), 1, ..)` on R^2), else the test gets
   `NoComponentsInChart`. Spell this out.
6. **Tree walks on shared subtrees.** `is_finite_expr` and the constructor pass walk the tree, not the
   DAG of `Arc`s: a component of a few KB with 2^40 tree nodes makes `from_components` hang where HEAD
   returned at once. `simplify` and `differentiate` behave the same at HEAD, so this is not new for d,
   but it is new for the constructors. Either memoize the scan by `Arc` pointer or document it in the
   risks.

## Files

`$R/r_dfacade.rs` and `dfacade.txt` (B1), `$R/r_adv.rs` and `neg.txt` (minor 1; also the trig, power
and `Mod` probes: `sin(x)^2 + cos(x)^2 - 1` is not folded by `simplify`, so the last row of
`test_is_closed_three_valued` is `Unresolved` as designed), `$R/b1_mine.txt` (identical to the designer's
`b1.txt`), copies of the designer's `proto2.rs`, `t_b1.rs`, `t_b2.rs`, `t_simp.rs`, `measure_new2.rs`,
`measure_head.rs`, `repro_head3.rs`, `repro_head4.rs` used unchanged.

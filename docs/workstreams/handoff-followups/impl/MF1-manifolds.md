# MF1-manifolds: implementation report

Unit MF1 of the HANDOFF follow-ups (items F2 library part, N4 / S8, REPL guards). Design:
`fu2/design/MF1-manifolds.md` (Revision 3, approved by review 4). Implemented on HEAD `0c2f368`,
whose code equals `36c2170` (the design's base). S1 is not committed; MF1 uses nothing of it.
Nothing is committed. Work files: `fu2/impl/MF1-work/` (`build.sh`, `rlibs.txt`,
`measure_impl.rs` and `measure_impl.txt`, harness outputs, gp outputs). Harness binaries deleted.
I did not build in release, so I left `target/release` alone (not mine).

## Files changed

| File | Change |
|---|---|
| `rustmath-manifolds/Cargo.toml` | `rustmath-polynomials` dependency (`Cargo.lock` gains one line) |
| `rustmath-manifolds/src/errors.rs:86-102` | `ComponentCountOverflow`, `ComponentAllocationFailed`, `DegreeOverflow`, `Undecided` |
| `rustmath-manifolds/src/expr_zero.rs` (new, 1.6) | zero test, scans, `sum_signed`, `normalize`, derivative guard, tests |
| `rustmath-manifolds/src/lib.rs` | `pub mod expr_zero;`, `pub use expr_zero::{zero_test, zero_test_with_budget, ZeroTest}` |
| `rustmath-manifolds/src/tensor_field.rs` | checked counts, crate helpers, `contract` and `tensor_product` rewritten, tests |
| `rustmath-manifolds/src/diff_form.rs` | rewritten: d, wedge, interior product, Lie derivative, Add, Neg, scale, vanishes, closedness, constructors with checks, tests |
| `rustmath-manifolds/src/lie_derivative.rs` | `apply_to_form`, guarded `apply_to_one_form`, `apply_to_tensor` NotImplemented, `commutes_with_exterior_derivative`, tests |
| `rustmath-manifolds/src/symplectic.rs` | `SymplecticForm::new` doc, `standard_form` checked, `is_closed` delegates, `volume_form` exact, `is_non_degenerate` doc names the facade, tests |
| `rustmath-manifolds/src/contact.rs` | `standard` uses chart coordinates, checked dims, `contact_volume`, three-valued `is_contact`, `ContactManifold::standard` checked, tests |
| `rustmath-manifolds/src/topology.rs:82-89` | `DeRhamCohomology::is_closed` delegates, test |
| `rustmath-manifolds/src/maps.rs` | three facades return `Err(NotImplemented)`, test |
| `rustmath-manifolds/src/catalog.rs:624-626` | duplicate `||` operand removed (see Deviations 9) |
| `rustmath-jupyter/src/repl.rs:4903-4904`, `:12326-12327` | the two comments of design 4.3 (no code, no test) |

`contact.rs` and `symplectic.rs` use CRLF line endings at HEAD; they keep them.

## Per item

### F2 (library): d and wedge (design 1.3 to 1.9)

* `expr_zero.rs` (new). Module doc: references (MAGMA Ch. 24 §24.4.2, §24.4.7, Ch. 41,
  Buchberger-Loos 1982, Richardson 1968, Lindemann 1882, Hermite 1873, `parser.rs:216-237`,
  `differentiate.rs` lines, tracker `:2117`), semantics, reserved names, exact `Real` value,
  precondition, algorithm. Items: `ZeroTest` (`:79`), `reserved` (`:90`), `ZERO_TEST_WORK`
  (`:100`), private limits `COEFF_BITS`, `MAX_DEGREE`, `MAX_TREE_DEPTH`, `MAX_NESTING`,
  `MAX_OPAQUE`, `MEMO_AFTER`; the charged conversion `Conv` (`:177`, negation charged in
  `negate`), `exact_real` (`:454`), `prescan` (`:509`), `zero_test` (`:557`),
  `zero_test_with_budget` (`:563`), `is_finite_expr` (`:609`, memoized), `normalize` (`:655`),
  `sum_signed` (`:673`), `unsupported_derivative` (`:726`, with the review 3 `root_sum` rule),
  `check_derivative` (`:789`). It follows `rev3/proto4.rs` without the `ZT_*` overrides, the
  counters and the `limb_div` factor (always 1 under `COEFF_BITS`).
* `diff_form.rs`: module doc with the convention, I1, I2, the d preconditions and the citations
  (Lee 2013 ch. 14, Spivak 1965 ch. 4, MAGMA Ch. 82 §82.3, Ch. 42 §42.15, Ch. 111, the Sage
  module; MAGMA has no smooth-manifold chapter and no examples to port). Helpers `first_comb`,
  `next_comb`, `for_each_perm` (Heap), `neg`, `scatter`, `decode`, `binomial`
  (`:92-178`), `check_coordinate_chart` (`:182`), `validate_components` (`:220`).
  `exterior_derivative` (`:512`) follows 1.4 steps 1 to 7 in that order (overflow, p >= n,
  lookup, chart check, count, loop with `check_derivative` before every `differentiate`).
  `wedge` (`:580`) follows 1.5.
* Measured on the implementation (`MF1-work/measure_impl.txt`, debug rlibs, public API):
  d(dx_0) n = 1000: 0.23 s; dx_0∧dx_0 n = 1000: 0.28 s; d(d(dx_0)) n = 100: 0.19 s (all
  literal zeros; VmHWM 128 MB includes the harness's clone of the result, the result is 64 MB);
  d of ω_i = x_{i+1} n = 1000: 0.18 s, 66 MB; d of the 2-form x_i x_j on R^100: 0.43 s, 69 MB;
  d∘d of ω_i = x_{i+1} x_{i+2} n = 60: 0.06 s, all literal 0; d of (0, s^64): 0.07 s;
  (x_0..x_999)∧(x_1..x_0): 11.6 s, 869 MB, 999000 nonzero slots. All within design 1.9.

### N4 / S8 (design 2)

* `tensor_field.rs`: `component_count` (`:36`, doc test), `flat_index` (`:60`, crate),
  `zeroed_components` (`:66`), checked `MultiIndex::to_flat` (`:94`), `total_rank` (`:228`)
  and `num_components` (`:235`) return `Result`, `from_components` checks the rank sum and the
  count, `component` (`:315`) borrows the components and checks the index rank,
  `set_component` (`:328`) checks the rank and allocates with `zeroed_components`.
  Crate helpers `components_ref` (`:247`), `from_dense_unchecked` (`:256`),
  `chart_components` (`:275`), `chart_components_mut` (`:280`, see Deviations 2),
  `from_chart_components` (`:285`).
* `diff_form.rs`: every degree sum is `checked_add`; every allocation goes through
  `component_count` and `zeroed_components`; d checks the chart.
* `contract` (`:362`) and `tensor_product` (`:425`) rewritten (3.2), with checked rank sums.
* `symplectic.rs:185` `standard_form`: `n.checked_mul(2)`, `component_count`, `zeroed_components`.
  `contact.rs:105` `ContactForm::standard` and `:259` `ContactManifold::standard`: 2n + 1 checked.

### REPL (design 4)

No REPL code or test change. Two comments corrected (4.3). Every REPL test passes (109 passed,
1 ignored, plus 4 in the integration test, as before). The follow-ups of 4.4 are in Handoffs.

### Adjacent facades and callers (design 3)

* `diff_form.rs`: `from_tensor` (`:342`) and `from_components` (`:373`) check finiteness and, for
  degree >= 2, antisymmetry with the zero test, then rebuild (I1). New
  `from_independent_components` (`:391`), `independent_components` (`:438`). `interior_product`
  (`:662`), `lie_derivative` (`:737`, Cartan), `scale` (`:772`), `vanishes` (`:821`),
  `closedness` (`:859`), three-valued `is_closed` (`:881`), `Add` (`:916`), `Neg` (`:973`).
  `is_zero` (`:850`) keeps its structural meaning; its doc points to `vanishes`.
* `lie_derivative.rs`: `apply_to_form` (`:118`) delegates degree >= 2 to Cartan;
  `apply_to_one_form` (`:144`) has the chart check, the literal-zero skips, `check_derivative`,
  `sum_signed` and `normalize`; `apply_to_tensor` (`:209`) returns `Err(NotImplemented)` after its
  three rank checks; `commutes_with_exterior_derivative` (`:249`) decides with `vanishes`.
* `symplectic.rs`: `SymplecticForm::new` doc (`:164-168`, "Checks only that the form has degree
  2 ..."), `is_closed` (`:232`) delegates, `volume_form` (`:318`) is ω^n scaled by the exact 1/n!
  (n = 0: the 0-form 1); `is_non_degenerate` doc names the `matrix_at` facade (not fixed).
* `contact.rs`: `standard` writes `-Expr::Symbol(chart.coordinate_symbol(2i + 1))`;
  `contact_volume` (`:151`) and `is_contact` (`:175`): `Ok(true)` for a nonzero rational
  constant, `Ok(false)` when it vanishes, else `Err(Undecided)`; n = 0 uses α itself.
* `topology.rs:86` `DeRhamCohomology::is_closed` delegates.
* `maps.rs:346` (`PullBack::apply_to_form`, p > 1), `:391` (`apply_to_covariant_tensor`,
  q >= 2), `:457` (`PushForward::apply_to_contravariant_tensor`, p >= 2): `Err(NotImplemented)`
  with the missing formula, after the rank checks, before the Jacobian.

## Tests and reference sources

All new tests pass; none fails on the implementation. 49 new unit tests and 2 new doc tests.
Reference values: PARI/GP 2.15, `gp -q refs.gp < /dev/null`, `gp -q rev2/refs2.gp < /dev/null`,
`gp -q rev3/refs4.gp < /dev/null` (outputs in `MF1-work/refs*.out`; `refs4.out` is identical to
the design's); the Sage values quoted in the design. The gp lines are quoted in the test comments.

* `expr_zero` (6): `test_zero_test_cases` (every row of 1.6.6, and the 64-term `sin` sum is
  `Zero`), `test_zero_test_budget` (all bullets of 1.6.6, 2 MB threads, wall-clock bounds 1 s and
  5 s), `test_scans_on_shared_subtrees` (both DAG shapes, five leaves, 1 s bound),
  `test_normalize_and_finiteness`, `test_sum_signed_shape`, `test_exact_real` (0.1 is
  3602879701896397/2^55; 5e-324 is 2^-1074).
* `tensor_field` (7): `test_component_count_checked`, `test_num_components_checked`,
  `test_from_components_huge_rank`, `test_zeroed_components_allocation_failure` (64-bit only),
  `test_multi_index_checked`, `test_contract_trace` (gp `trace(matdiagonal([5,7]))` = 12;
  δ^i_j a_k gives 3a_k and a_j), `test_tensor_product_index_order` ([3, 4, 6, 8], and a
  (1,1)⊗(0,1) case).
* `diff_form` (27): every test of 1.8, 2.3 (except the tensor_field ones) and 3.3:
  `test_d_one_form_r3`, `test_dd_is_literal_zero`, `test_d_of_coordinate_form_is_zero`,
  `test_d_trig_sage`, `test_d_two_dim`, `test_d_zero_form_gradient`, `test_wedge_sage_r3`,
  `test_wedge_one_two`, `test_wedge_two_dim`, `test_wedge_antisymmetry`, `test_wedge_powers`,
  `test_leibniz`, `test_d_budget`, `test_d_reserved_constants`,
  `test_interior_product_long_sum` (R^20000, 2 MB thread),
  `test_d_refuses_unsupported_derivatives` (the 12 refused forms, the 0-form, 6 accepted with
  literal 0, 4 accepted with a value, the 8 refused `root_sum` forms, `root_sum(t^2 - x, t) dx`;
  `closedness` and `is_closed` return the same Err), `test_degree_overflow`,
  `test_d_chart_dimension`, `test_d_chart_identity` (with the review 4 repeated-coordinate row),
  `test_zero_by_dimension`, `test_interior_product_gp`, `test_lie_derivative_cartan_gp` (plus the
  top-form case: L_X of dx∧dy∧dz is 0, div X = 0), `test_add_neg_scale`,
  `test_from_components_antisymmetry`, `test_non_finite_components_refused`,
  `test_from_independent_components`, `test_is_closed_three_valued`.
* `lie_derivative` (3): `test_apply_to_tensor_not_implemented`, `test_apply_to_two_form`
  (gp `d(L_X A)`, and `commutes_with_exterior_derivative` is `Ok(true)`),
  `test_lie_one_form_guard` (all cases of 3.4, gp `refs2.gp`, `refs4.gp`).
* `symplectic` (2): `test_symplectic_is_closed_nonconstant`, `test_symplectic_volume_form`
  (gp `(-1)^(k*(k-1)/2)`). `contact` (2): `test_contact_volume` (gp `k!`),
  `test_dz_is_not_contact` (plus α = dz - y dx accepted, dz - xy dx Undecided). `topology` (1):
  `test_de_rham_is_closed`. `maps` (1): `test_pull_push_facades_not_implemented`.

The design harnesses, rebuilt against the current debug rlibs (`MF1-work/build.sh`, new
`rlibs.txt`): `t_rootsum3` prints `rev3/rootsum3.txt` (apart from the timing), `t_dguard` prints
`rev2/dguard.txt` exactly, `t_b2 negp 80 197` is `Unresolved` after 100000 units in 0.144 s
(sweep4: same units). These run the prototype; my unit tests run the same inputs on the
implementation and give the same answers.

HEAD-fail check: the new tests use new APIs (`zero_test`, `independent_components`,
`from_independent_components`, `vanishes`, `Result` returns), so they do not compile on HEAD.
I did not apply the test hunks to HEAD (that needs a checkout or stash, which the task forbids).
The HEAD evidence is the design's repro programs (`repro_head*.rs`, review 2, 3 and 4 probes).

No existing assertion was weakened or changed. The existing tests at `tensor_field.rs`
(`to_flat`, `total_rank`, `num_components` x2) gained `.unwrap()` for the new `Result` returns.

## API and behaviour changes, callers checked

`git grep` over the workspace; only rustmath-jupyter depends on rustmath-manifolds.

* `MultiIndex::to_flat`, `TensorField::total_rank`, `TensorField::num_components` return
  `Result`. Callers: `tensor_field.rs` itself, `lie_derivative.rs` (`apply_to_tensor`, now
  without the call), three tests. `TensorFieldModule::total_rank` (`modules.rs:605`) and
  `traits.rs:124` are other methods and unchanged.
* `DiffForm::from_components` / `from_tensor` now refuse non-finite components
  (`ValidationError`), non-antisymmetric ones (`ValidationError`) and undecidable ones
  (`Undecided`). Callers: `standard_form` (constant ±1, passes), `contact.rs`, `maps.rs:321`
  (1-form pullback), `integration.rs:213` (`VolumeForm::from_metric`: n = 1 passes; n >= 2 already
  failed on the length), tests.
* `DiffForm::is_closed` can return `Err(Undecided)` and `Err(NotImplemented)`. Callers:
  `symplectic.rs` (`SymplecticManifold::new`, `SymplecticForm::is_closed`), `topology.rs`,
  `kahler.rs:95`, `:239` (unreachable: `compute_kahler_form` is `unimplemented!`).
* `exterior_derivative` refuses a chart of the wrong dimension, with repeated coordinates, or with
  coordinates that differ from the registered chart of that name. Every in-crate caller passes
  `default_chart()` or its caller's chart. REPL charts are never registered and J1a's
  `check_chart` checks the dimension first; the REPL tests pass.
* `LieDerivative::apply_to_tensor` and the three `maps.rs` methods return `Err(NotImplemented)`;
  no caller in the workspace.
* `integration.rs:390`, `:423` (Stokes) now integrate `components[0]` of the true dω, which is 0;
  integration is wrong before and after (design section 5).
* New public items: `expr_zero` module (`zero_test`, `zero_test_with_budget`, `ZeroTest`,
  `is_finite_expr`, `reserved`, `ZERO_TEST_WORK`), `tensor_field::component_count`,
  `DiffForm::{from_independent_components, independent_components, scale, vanishes,
  closedness}`, four `ManifoldError` variants. No `match` on `ManifoldError` exists outside the
  crate.

## Deviations from the design, with evidence

1. Review 4 minor 3 folded in: `check_coordinate_chart` also refuses a chart that repeats a
   coordinate (`Err(InvalidChart("the chart 'r' repeats the coordinate x"))`), tested in
   `test_d_chart_identity`; the module doc says the first registered chart of a name is used. It
   also compares `coordinate_names().len()` with n (equal by `Chart::new` anyway).
2. `TensorField::chart_components_mut` (crate-private) added to 1.7's helpers: `from_tensor`
   rewrites every chart's vector in place, and `Neg` negates in place, as the design asks.
3. `scale` treats a `Rational` equal to 1 like the literal 1 (a clone). `simplify` keeps
   `Rational(1)` as a `Rational`, so ω/1! came back with a `Rational(1)` component; the volume
   test compares by value.
4. `apply_to_one_form` returns `DimensionMismatch` when the form or vector components do not have
   n entries (HEAD would index out of bounds and panic for a vector field of another manifold).
5. `ContactManifold::standard` (`contact.rs:259`) computes 2n + 1 through `ContactForm::standard`
   (checked) instead of unchecked; not in the design, same N4 class.
6. `test_d_reserved_constants` and `test_wedge_powers` compare the slots by value
   (`zero_test(slot - reference) == Zero`, plus not literal 0) rather than structurally. Sound for
   opaque `I` too (a `Zero` answer holds for any value of it). The other structural assertions of
   1.8 (`test_d_budget`, the accepted `root_sum`, `abs`, `x^y` slots, the `-root_sum(.., 0)`
   slot) are structural as designed and pass.
7. `from_independent_components` with p > n and nonempty values: `DimensionMismatch { expected:
   0, .. }` (the design says the values must be empty).
8. `DiffForm::lie_derivative` looks the form's components up before the n = 0 shortcut, so a form
   without components errs as everywhere else.
9. `catalog.rs:624-626`: `(self.k == 2 && self.n == 4) || (self.k == 2 && self.n == 4)` became
   `self.k == 2 && self.n == 4` (identical behaviour). It is a deny-level clippy lint (`eq_op`)
   that made the acceptance command `cargo clippy -p rustmath-manifolds` fail at HEAD.
10. Clippy was run as `cargo clippy -p rustmath-manifolds --no-deps`. Without `--no-deps` clippy
    stops in rustmath-combinatorics (a dependency through rustmath-category and rustmath-groups,
    since R1) on pre-existing deny lints: `inherent_to_string_shadow_display` at
    `binary_words.rs:336`, `debruijn.rs:101`, `super_tableaux.rs:297`, `word.rs:252`, and
    `while_immutable_condition` at `path_tableaux.rs:454`. With `--tests`, clippy also stops on
    pre-existing `approx_constant` in `rustmath-manifolds/src/examples.rs:520`, `:522` (not fixed:
    test code outside the unit, not an acceptance target). The only clippy warning in the new
    code is the pre-existing unread `TensorField::name` field.

## Review 4 minor notes

Folded in: 3 (repeated coordinates, doc sentence on the first registered chart), 4 (the
`unsupported_derivative` doc says `_t` (or `_t0`), which no chart in the workspace uses),
5 (the (1, 0) tensor of `test_apply_to_tensor_not_implemented` has components). Skipped: 1, 2
and 6, which are slips in the design text (the prototype name in plan step 3, the HEAD column
of `test_d_chart_identity`, line numbers in section 5); no code depends on them. My test comment
for `test_d_chart_identity` does not claim a HEAD value.

## Final test counts

* `cargo test -p rustmath-manifolds --no-fail-fast`: 308 passed, 0 failed, 11 ignored; doc tests
  33 passed (baseline 259 / 11 ignored / 31 doc tests).
* `cargo test -p rustmath-jupyter --no-fail-fast`: lib 109 passed, 1 ignored; integration 4
  passed (unchanged).
* `cargo check --workspace --tests`: ok.
* `cargo clippy -p rustmath-manifolds --no-deps`: ok (warnings only, see Deviations 10).
* gp scripts and the three harnesses: as above.

## Left undone

* Nothing of the design's plan steps 1 to 7. Step 8 (trackers, HANDOFF) is the orchestrator's.
* The HEAD-fail check by applying the test hunks to HEAD (forbidden checkout); see above.
* Section 5 defects of the design, by design.

## Handoffs

For the trackers (plan step 8, rows by title, lines at `36c2170`):

* "Exterior derivative d on p-forms" `:2531`, "Tensor contraction (Einstein summation)" `:2532`,
  "Wedge product" `:2533` to 🟢: `diff_form.rs:512`, `:580`, `tensor_field.rs:362`, exact zero
  test `expr_zero.rs:557`.
* "Lie derivative of scalars/vectors/forms" `:2548`: `apply_to_form` correct (Cartan for p >= 2,
  guarded coordinate formula for p = 1); `apply_to_tensor` returns `Err(NotImplemented)`; stays ⏳.
* "De Rham cohomology" `:2554`: `is_closed` is three-valued (Undecided).
* "Integration on manifolds" `:2547`: Stokes now integrates `components[0]` of the true dω (0);
  the quadrature, `evaluate_at` and `VolumeForm::from_metric` defects of design section 5 remain.
* "Pullback / pushforward of forms and vectors along maps" `:2539`: the p-form and
  covariant-tensor pullbacks and the rank >= 2 pushforward return `Err(NotImplemented)`; the
  base-point issue `maps.rs:193-195` and the 1-form pullback without ω∘f remain; stays 🛑. Bullet
  `:2572` the same; remove the "Absent entirely" bullets `:2566`, `:2567`; tally `:2527`.
* "Symplectic form non-degeneracy test" `:2542` stays 🛑 (`matrix_at` facade; doc now says so).
* "Derivative of a named function" `:2117` (rustmath-symbolic): stale `:205` is `:230-234`; add the
  other facade kinds (Mod `:90`, gamma `:184-188`, factorial `:191`, zeta `:204`, erf
  `:194-198`, `root_sum` with the variable in its polynomial `:218-221`) and "MF1's d refuses such
  components (`expr_zero::unsupported_derivative`)".

For HANDOFF (design 4.4 and section 5; all still hold):

* REPL (design 4.4): drop `DIFF_FORM_DEGREE` and its test lines; delete the arms' redundant
  overflow checks (`repl.rs:4862-4863`, `:4901-4902`) or keep them for wording; compute
  `check_form_components` and `missing_components_error` with
  `rustmath_manifolds::tensor_field::component_count` ("too many components to store");
  re-measure `FORM_COMPONENTS` (release, J1b's harness; 64 bytes a slot, held once) and rewrite
  its doc and the `repl_limits` overview; unique REPL chart names or registration before forms
  with symbolic components; optionally expose `is_closed`, `closedness`, `interior_product`,
  `lie_derivative` and component reading.
* rustmath-symbolic `differentiate` facades (above) and the unguarded callers in
  rustmath-manifolds: `vector_field.rs:169`, `:198`, `:202`, `scalar_field.rs:149`,
  `maps.rs:126`, `transition.rs:143`, `utilities.rs:134`, `symplectic.rs` `from_hamiltonian`,
  `symmetries.rs:121-139`, `finsler.rs:160-161`, `:309-311`.
* Integration quadrature and top component; pullback / pushforward formulas; symplectic
  non-degeneracy and `evaluate` (`matrix_at`); a checking `SymplecticForm` constructor; the Lie
  derivative of general tensors (formula in `apply_to_tensor`'s doc and error).
* Charts identified by name only (store coordinate names with components).
* Unchecked counts: `spin.rs:76`, `:78`, `:341`, `:343`; `almost_complex.rs:160`,
  `finsler.rs:153`, `:300`, `symmetries.rs:110`, `symplectic.rs:258` (`matrix_at`); module rank
  sums `modules.rs:605`, `traits.rs:124`.
* `DiffForm::new` means both "zero" and "no components"; dense storage limit (ω^4 on R^8 needs
  8^8 slots; R^10 and above end in `ComponentAllocationFailed`).
* `tracker_08_09_tests.rs` is still not compiled (private field `PullBack::map`).
* One public zero test in rustmath-symbolic (move `expr_zero` there; with S1, merge with S1's
  constants). `simplify` folds `0 * a` to 0 for any a (including `oo`, NaN, `1/(y - y)`).
* New, found here: `Expr::from(f64)` maps NaN and infinities to `Integer(0)`
  (`rustmath-symbolic/src/expression.rs:399-404`), so a non-finite float entered that way
  becomes a literal 0 before MF1's finiteness checks can see it. `simplify` keeps `Rational(1)`
  as a `Rational` (does not normalize to `Integer`). Pre-existing deny-level clippy lints in
  rustmath-combinatorics (Deviations 10) block plain `cargo clippy -p` for every crate that
  depends on rustmath-category; `examples.rs:520`, `:522` block clippy `--tests` on
  rustmath-manifolds.
* For S1 (if it lands after MF1): every `match` on `Expr` in MF1 is on a reference and no field
  is moved out of an owned `Expr`, so S1's `impl Drop for Expr` compiles against it. MF1's tests
  build long sums with `sum_signed` and compare printed subterms through `format!("{}", u)` of
  expressions built in the test; the `root_sum`, `abs`, `x^y` and `d_budget` slot assertions are
  structural and depend on HEAD's `simplify` results, which S1 states it keeps (S1 §2.10).

# MF1-manifolds: review 3 (independent, adversarial)

Design reviewed: `fu2/design/MF1-manifolds.md` (1286 lines, with "Revision 1" and "Revision 2"). HEAD `c79a582`,
tree clean. Scratch: `$R = .../fu2/design/MF1-manifolds/review3/`. Programs are built with the designer's
`../build.sh` against the HEAD debug rlibs; the binaries were deleted afterwards and the outputs kept as `.txt`.

Verdict: **not approved**. One blocker, in the derivative guard that Revision 2 added for review 2's B1: it admits
every two-argument `root_sum`, and `differentiate` fabricates 0 for a `root_sum` whose polynomial contains the
variable. So d, `closedness`, `is_closed` and the Lie derivative still return fabricated values and false
decisions for such input. The fix is three lines and a few tests. The rest of the design is sound and well
evidenced, and the blockers of reviews 1 and 2 are fixed.

## What I reproduced and found to hold

* HEAD defects (`$R/repro_head.rs`, the designer's source): d(xy dx + x^2 dy) is `[(y + x), (2 * x), 0, 0]`;
  dx∧dx has `(1 * 1)` nodes; the trace of diag(5, 7) is `[1]`; the (0,1)⊗(1,0) product is `[3, 6, 4, 8]`; panics
  `j % 0` (`diff_form.rs:151`), multiply overflow (`tensor_field.rs:158`, `:105`), add overflow (`diff_form.rs:132`,
  `:182`); rank 2^32 + 1 gives 2 components. Line numbers of 1.1 and 2.1 match HEAD (`grep -n`).
* HEAD cost (`measure_head`, debug): d(dx_0), n = 1000: 8.16 s, 253 MB; d(d(dx_0)), n = 100: 1.86 s, 444 MB.
  Prototype (`measure_new2` on `proto3.rs`): 0.15 s / 66 MB, 0.15 s / 66 MB, dx_0∧dx_0 n = 1000 0.23 s / 66 MB, the
  2-form x_i x_j on R^100 0.69 s / 70 MB. The claims of 1.9 hold.
* Zero-test cost (`t_b2`, `r_adv` on `proto3.rs`): univ 50 64 0.047 s, negp 80 197 0.231 s, bil 80 0.174 s,
  prod 12 0.144 s, all `Unresolved` at 100000 units. My own worst case for coefficient size (`$R/r_bigab.rs`: one
  product of two distinct 100- or 120-term univariate polynomials with 2040-bit coefficients, `bigab.txt`):
  0.087 s and 0.112 s. The budget charges at least about 5 units per coefficient product, so at most about
  20000 products of at most 2048-bit factors run per call. The 0.25 s worst case of 1.6.3 holds.
* DAG scans (`t_dag`): 2^40-leaf trees in 0.1 ms, as in `dag.txt`.
* Soundness of the zero test (1.6.4), re-derived: the N/D induction, the fast paths, opaque atoms compared by
  `==` (equal structure, equal value), `Symbol` ids interned by name (`symbol.rs:41-47`, so chart symbols and
  `Expr::symbol` agree), named constants by transcendence. `simplify` (`simplify.rs:376-652`) has no rule that
  changes a value on a dense open set, apart from the `0 * a` folding that `normalize` guards. No false `Zero`
  or `NonZero` found within the stated precondition.
* Formulas re-derived: d, wedge (the `inv += p - x` shuffle sign), Heap's sign, contraction digit insertion, the
  tensor-product output index, interior product (first slot), Cartan cases, ω^k = k! (-1)^(k(k-1)/2) for
  the (q, p) layout of `standard_form`, α∧(dα)^k = k! for the interleaved layout of `ContactForm::standard`.
* References: `gp -q refs.gp` and `gp -q rev2/refs2.gp` print every value quoted in 1.8, 3.3 and 3.4. The
  Sage strings expand to the gp values (a∧b, dA, 2D wedge).
* `test_d_budget`: slot (0,1) equals `(s^64).differentiate(x).simplify()` structurally (`misc.txt`).
* Callers (`git grep` over the workspace): only rustmath-jupyter depends on the crate; every caller of d,
  wedge, `is_closed`, `contract`, `tensor_product`, `to_flat`, `total_rank`, `num_components`,
  `from_components`/`from_tensor` of forms is in 3.4 or the plan. The plan keeps rustmath-manifolds green after
  each step (I traced the HEAD bodies of `SymplecticForm::is_closed`, `is_contact` and `apply_to_one_form`
  against the new d and wedge between steps 5 and 6).
* REPL (item 3): the line numbers of 4.1 and 4.4 are right. Only `test_form_component_limits` `:12359-12367`
  reaches the new chart check; the R^1 block of 4.3 works against the arms as written (`chart` takes its
  dimension from the number of names, `repl.rs:4585-4590`). J1a (§5, the R^3 block back with
  `chart(E3, "x,y,z")`) and J1b (§6 table rows `:1265-1266`, plan step 8) consume 4.3 and 4.4.
* Baseline: `cargo test -p rustmath-manifolds --no-fail-fast` at HEAD: 258 passed, 11 ignored, 31 doc tests.
* Citations: Chapter_24 §24.4.2 and §24.4.7, Chapter_41, Chapter_42 §42.15, Chapter_82 §82.3, Chapter_111 exist.
* Coordination: S1 (before MF1) adds `impl Drop for Expr` and keeps `simplify` results; I3 makes
  `Integer::to_i64` return `Option` (the design uses `NumericConversion::to_i64`); P1 does not touch
  `MultivariatePolynomial`. None breaks the plan.

## Blocker

### B1. The guard admits `root_sum(f, g)` with the variable in f; d and closedness fabricate 0

`differentiate.rs:218-222`: for `Function("root_sum", [f, g])`, `if args[0].contains_symbol(var) { 0 }`. The
comment assumes that a variable in f *is* the bound variable τ. That is right only when τ is the only symbol
of f. For `root_sum(t^2 - x, t^2)` (the sum of c^2 over the roots c of t^2 - x, which is 2x) the derivative in x
is 2, and `differentiate` returns 0. The design's guard (1.4, `derivative_unsupported_kind` in `proto3.rs`)
admits every two-argument `root_sum`, and 1.4 states the precondition "on every node kind that
`unsupported_derivative` admits, `differentiate` returns the derivative ... This holds by the rules listed
above". It does not hold here. Such input is reachable: `parse("root_sum(t^2 - x, t^2)")` builds it, and the
design's own accepted-forms test uses a user-chosen bound symbol (`root_sum(t^2 - 2, t*x)`).

Reproduced on `proto3.rs` (`$R/r_rootsum.rs`, `rootsum.txt`; `$R/r_misc.rs`, `misc.txt`), on R^2:

| Component of ω | prototype (dω)_{xy} | zero test | true (dω)_{xy} (gp, `$R/refs3.gp`) |
|---|---|---|---|
| `root_sum(t^2 - x, t^2)` dy (built, and by `parse`) | literal 0 | Zero | 2 (`polsym(x^2 - a, 4)` = `[2, 0, 2*a, 0, 2*a^2]`) |
| `root_sum(t^2 - x, t^4)` dy | literal 0 | Zero | 4x |
| `root_sum(t - x, t)` dy (= x dy) | literal 0 | Zero | 1 |
| `sin(root_sum(t^2 - x, t^2))` dy | literal 0 | Zero | 2 cos(2x) |
| L_∂x of `root_sum(t - x, t)` dy (`lie_one_dense`) | `[0, 0]` | | dy, i.e. `[0, 1]` |

So `closedness` returns `Zero` and `is_closed` `Ok(true)` for non-closed forms, and the same reaches
`SymplecticManifold::new`, `DeRhamCohomology::is_closed`, `is_contact`, `commutes_with_exterior_derivative` and
the Lie derivative. This is the class of review 2's B1 (CLAUDE.md: a facade is a defect; a certifying routine
must not claim a decision it does not have).

Required change:

1. In `unsupported_derivative`, a two-argument `root_sum` whose first argument contains `var` is an unsupported
   node (returned as the offending subterm). A narrower rule is also sound: refuse only when the first argument
   contains `var` and another symbol (when `var` is its only symbol, `var` is the bound variable and 0 is
   right, `misc.txt`: `root_sum(x^2 - 2, x*y)`). The simpler rule loses only a chart coordinate that is itself
   the bound symbol. Do the `contains_symbol` check without an unmemoized tree walk (reuse the scan), so the
   scan stays linear in the distinct nodes.
2. Correct the table and the precondition of 1.4: the `root_sum` rule is the derivative only when the roots of
   f do not depend on `var`.
3. Tests: add to `test_d_refuses_unsupported_derivatives` the forms `root_sum(t^2 - x, t^2)` dy (also via
   `parse`), `root_sum(t - x, t)` dy and `sin(root_sum(t^2 - x, t^2))` dy, each `Err(NotImplemented)` with
   `closedness` and `is_closed` propagating it (true values 2, 1, 2 cos(2x) from gp `polsym`); add to
   `test_lie_one_form_guard` X = ∂_x, ω = `root_sum(t - x, t)` dy: `Err(NotImplemented)` (true value dy). Keep
   `root_sum(t^2 - 2, t*x)` dy accepted. Each of these returns `Ok` with a literal 0 on `proto3.rs`, so the
   tests fail before the change.
4. Section 5 and the HANDOFF bullet on the `differentiate` facades: add this rule (`differentiate.rs:218-222`
   trusts that a variable of f is the bound variable).

## Minor issues

1. **A facade is left next to an edited line.** Plan step 4 edits `LieDerivative::apply_to_tensor`
   (`lie_derivative.rs:225-226`) only to make it build, but the function returns values without the
   correction terms (`:229-244`; section 5 lists it). The design's own rule turned the identical situation in
   `maps.rs` into `Err(NotImplemented)`. `apply_to_tensor` has no caller in the workspace (`git grep`). Either
   return `Err(NotImplemented)` after the rank checks (two lines), or implement the formula of its doc comment
   ((L_X T) = X^k ∂_k T - Σ_a T^{..k..} ∂_k X^{i_a} + Σ_b T_{..k..} ∂_{j_b} X^k, Lee 2013 ch. 12) with the guard,
   `sum_signed` and a gp test. Update tracker row `:2537` accordingly.
2. **Tracker row `:2528` is not in plan step 8.** The 🛑 row "Pullback / pushforward of forms and vectors along
   maps" (`maps.rs:188`) cites the `:402/:485` "pull back componentwise" facades that MF1 turns into
   `Err(NotImplemented)`. Plan step 8 updates only the bullet `:2561`. Update the row too (the base-point issue
   at `:193-195` and the 1-form pullback without ω∘f remain).
3. **d trusts the chart name.** Components are keyed by `chart.name()` (`tensor_field.rs:115`). A chart with the
   same name and dimension but other coordinates passes the new dimension check, and d differentiates in the
   wrong symbols: ω = x dy stored under chart "c" (x, y), then `ω.is_closed(&Chart::new("c", 2, ["u", "v"]))`
   gives `Ok(true)` (the same at HEAD, but now it is a "decided" answer). Every REPL chart is named
   `user_chart` (`repl.rs:4590`), so once J1a lets names be listed, two REPL charts on one manifold share a key
   (harmless today: REPL forms have constant components). Either compare the chart with
   `manifold.chart(name)` when the manifold has one of that name (`differentiable.rs:125`), or state the
   precondition in the docs of `exterior_derivative`, `closedness` and `is_closed` and add a section 5 / J1a
   note (unique REPL chart names).
4. **A test input can abort the test binary before S1 lands.** `test_zero_test_budget` builds "a sum of 30000
   distinct symbols". Built left-deep (a `fold` of `+`), dropping it overflows a 2 MB test thread at HEAD
   (`$R/r_drop.rs`, `drop.txt`: 5000 terms ok, 30000 aborts the process). S1's iterative `Drop` (S1 §2.14,
   implemented before MF1) removes this, but an abort kills every test of the binary. Say in 1.6.6 that the test
   builds the sum with `sum_signed`, or name the dependency on S1 §2.14.
5. **The R1 hand-off is not mentioned.** `R1-rings-padics-category.md` F.2 hands `DiffScalarFieldAlgebra`
   (`modules.rs:94`, ring predicates of C^∞(M)) "to MF1", and R1 plan step 13 records it as a new HANDOFF
   follow-up. It is not one of MF1's items. Add one line to section 5 saying so, so that it does not fall between
   the units.
6. **Wording (3.3).** "A non-finite X^k gives Err(ValidationError) through normalize" holds only when the term
   reaches `normalize`. A term whose form factor is a literal 0 is skipped, and in `apply_to_one_form`
   `simplify` folds `X^j * 0` to 0 first. `VectorField` components are not validated. Say so (the results are
   right for finite X; vector fields with `oo` are outside the contract).
7. **Plan order across crates.** Between plan steps 5 and 7, `cargo test -p rustmath-jupyter` fails in
   `test_form_component_limits` (the new chart check). Say that step 7 lands in the same commit as step 5.

## Files

`$R/r_rootsum.rs` and `rootsum.txt`, `$R/r_misc.rs` and `misc.txt` (B1, also the `test_d_budget` structural
check and `root_sum(x^2 - 2, x*y)`), `$R/refs3.gp` (gp reference values for B1), `$R/r_drop.rs` and `drop.txt`
(minor 4), `$R/r_bigab.rs` and `bigab.txt` (coefficient-size cost), `baseline.txt` (HEAD test counts), copies of
the designer's `proto3.rs`, `repro_head.rs`, `measure_head.rs`, `measure_new2.rs`, `t_b2.rs`, `r_adv.rs`,
`t_dag.rs` used unchanged.

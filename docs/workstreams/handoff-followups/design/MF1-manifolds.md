# MF1-manifolds: correct and bounded exterior algebra on forms, checked component counts

Unit MF1 of the HANDOFF follow-ups. Crate: `rustmath-manifolds` (plus one test block in
`rustmath-jupyter/src/repl.rs`, the only downstream crate, adapted to a library behaviour change).
HEAD `c79a582` (same code as `9c24389`).

## Revision 2 (after review 2)

Review: `MF1-manifolds.review2.md`. New experiments are in `$S/rev2/` (listed at the end of the scratch
table). `$S/rev2/proto3.rs` is `rev1/proto2.rs` with the changes below. It replaces `proto2.rs` as the
reference implementation. The rev1 harnesses were rerun on it unchanged (copies in `$S/rev2/`): `t_b1`
prints `b1.txt` exactly (`b1_3.txt`), and `t_ref2`, `t_ref22`, `t_contact2`, `t_zero2` print the same
values as on `proto2.rs`; only the work counters differ (`ref3.out` against `old/ref2_all.out`). The body
below is updated to match. Each issue and what changed:

| Issue | What changed |
|---|---|
| B1 (blocker): d inherits the `differentiate` facades of rustmath-symbolic | Agreed and reproduced. `differentiate` returns 0 for every `Function` but `root_sum` with 2 arguments (`differentiate.rs:230-234`, tracker 🛑 `PROJECT_TRACKER.md:2106`), for `Mod` (`:90`), `Gamma` (`:184-188`), `Factorial` (`:191`) and `Zeta` (`:204`), and uses 500/443 for 2/sqrt(pi) in `Erf` (`:194-198`; gp: 1.12867 against 1.12838, `$S/rev2/refs2.gp`). New crate-private scan `unsupported_derivative(e, var) -> Option<&Expr>` (1.4): the outermost subterm of one of these kinds that contains `var`. d (1.4 step 6) and the Lie derivative of a 1-form (3.4) call it before every `differentiate` and return `Err(NotImplemented)` naming that subterm. Every caller of d (closedness, `is_closed`, Lie derivative, contact, symplectic, de Rham) propagates the error instead of a fabricated 0. 1.4 states the precondition. On `proto3.rs` (`$S/rev2/t_dguard.rs`, `dguard.txt`): the review's seven forms and five nested ones (`sin(f(x^2 + y))`, `y*gamma(1 + x)`, `root_sum(t^2 - 2, t*f(x))`, the parser's `gamma(x)` and `f(x)`) are refused. `f(y) dy`, `f(x) dx`, `gamma(3) dy`, `gamma(y) dy`, `(y mod 3) dy` give a literal-zero d, which is correct: no derivative of an unsupported node in the variable is taken. `abs`, `sign`, `x^y` and `root_sum(t^2 - 2, t*x)` are differentiated. New tests: `test_d_refuses_unsupported_derivatives` (1.8) and `test_lie_one_form_guard` (3.4). Section 5 and the HANDOFF entry point to tracker `:2106`. Cost: d of the 2-form x_i x_j on R^100 (485100 scans) goes from 0.58 s to 0.68 s; the other rows of 1.9 do not change (`measure3.txt`) |
| Minor 1: negation not charged in the prototype | Charged now: weight + 1 units, before it runs (1.6.3). `MultivariatePolynomial::neg` rebuilds the map term by term with `add_term` (`multivariate.rs:804-813`). The number of terms would undercount it: one negation of 10^4 two-variable terms takes 0.08 s, 2.7 µs a weight unit. Measured on `proto3.rs` (`$S/rev2/neg3.txt`): 197 nested `Neg` around (a0 + ... + a99)(b0 + ... + b99) give `Unresolved` in 0.19 s (`proto2.rs`: `NonZero` after 12.9 s). The right-nested 0 - (0 - (... - P)) with 197 levels gives `Unresolved` in 0.18 s, one `Neg` gives `NonZero` in 0.19 s, and 197 nested `Neg` around x - y keep their decisions (2 ms). There is no lazy sign. It would make negation free, but an error in its sign rules could turn a `Zero` into a `NonZero` or the reverse, while a charge changes only the cost. The 20 families of `sweep.txt` give the same answers (`sweep3.txt`). The worst case per call is now 0.25 s in debug, not 0.10 s. The new families `bil` (P - P*1 with P a product of two sums of k symbols) and `negp` (nested `Neg` around such a product) cost up to 2.5 µs a unit. `bil 80` already took 0.17 s on `proto2.rs` (`old/bil_proto2.txt`), so the 0.10 s of Revision 1 was too low before the charge. `bil 80` is `Zero` on `proto2.rs` and `Unresolved` on `proto3.rs`: the negation of the 6400-term operand now counts. 1.6.3, 1.9, 4.2 and 6.3 are restated. Plan step 3 says that `proto2.rs` lacks this charge. New items in `test_zero_test_budget` (1.6.6) |
| Minor 2: pushforward, symplectic evaluation, `SymplecticForm::new` doc | `PushForward::apply_to_contravariant_tensor` for p >= 2 (`maps.rs:442-494`; it returns the source components at `:485-487`) now returns `Err(NotImplemented)` after the rank checks, like the pullback (3.4). Its only caller is `PushForward::apply_to_tensor` (`:502-515`), which has none. Section 5 now says that `SymplecticForm::evaluate` and `evaluate_on_vectors` (`symplectic.rs:273-301`) read the `matrix_at` facade too. MF1 corrects the doc of `SymplecticForm::new` (`:164`, "Verifies that the form is closed and non-degenerate"; it checks only the degree), because it edits `symplectic.rs` anyway (3.4) |
| Minor 3: `tracker_08_09_tests.rs` is not compiled | Agreed: no `mod` declares it (`git grep tracker_08_09` finds the file and `docs/magma/REVIEW_2026-11-06.md:69`, which lists it as uncompiled). It would not compile as a child of the crate root either: it reads the private field `PullBack::map` (`:102`). 3.4 no longer counts it among the tests that keep passing. Section 5 entry |
| Minor 4: more unchecked products of the N4 class | `almost_complex.rs:160`, `finsler.rs:153`, `:300`, `symmetries.rs:110`, `symplectic.rs:258` added to the `spin.rs` bullet of section 5 and to the HANDOFF entry |
| Minor 5: `test_d_chart_dimension` construction | Spelled out (2.3): the form is built with the 2-coordinate chart named `c`, and d is called with the 0- and 1-coordinate charts of the same name, so the lookup succeeds and the dimension check fires |
| Minor 6: scans walk shared subtrees as trees | `is_finite_expr` and `unsupported_derivative` memoize shared nodes. After 64 visited nodes, a node held by an `Arc` with a strong count above 1 is visited once (per flag), keyed by its address. A node with strong count 1 is reached only through its single parent, so the scans are linear in the number of distinct nodes. Measured (`$S/rev2/t_dag.rs`, `dag.txt`): a DAG of fewer than 100 nodes whose tree has 2^41 leaves takes under 1 ms for both scans, with the children in distinct `Arc`s or in one shared `Arc`. A leaf returns without allocating; other scans allocate their stack, and the set only after 64 nodes. The other tree walks of the constructors for degree >= 2 (`==`, the `simplify` in `neg`, the budgeted zero test) are documented in 6.3. New test `test_scans_on_shared_subtrees` (1.6.6) |

Disagreements with the review, each small:

* The scan lives in `expr_zero.rs`, not `diff_form.rs`. It shares the memoized walk with
  `is_finite_expr`, and `lie_derivative.rs` uses it too.
* The error message names the subterm and the variable, but no file lines or tracker lines. Line
  numbers go stale in a runtime message. The code comment carries the references.
* The negation test bound is 5 s, not 1 s. The measured time is 0.19 s, and the other wall-clock bounds
  of the design keep a margin of at least 25. The test still fails on `proto2.rs`, by its answer
  (`NonZero`) as well as by its time (12.9 s).
* The review's test list is extended. It gains nested cases, cases that must be accepted (`f(y) dy`,
  `gamma(3) dy`, `abs`, `x^y`, `root_sum` in the coordinate), and a 0-form case.

## Revision 1 (after review 1)

Review: `MF1-manifolds.review1.md`. New experiments are in `$S/rev1/` (`$S` is defined below; the rev1
files are listed at the end of the scratch table).
`$S/rev1/proto2.rs` is the revised prototype; it replaced `proto.rs` as the reference implementation
(and is replaced by `$S/rev2/proto3.rs` in Revision 2). The worst-case time of 0.10 s per zero test
stated in this table was too low; Revision 2 restates it as 0.25 s.
The body below is updated to match. Each issue and what changed:

| Issue | What changed |
|---|---|
| B1: the zero test treated `I`, `pi`, `e`, `oo` as free indeterminates and every leaf as finite | Section 1.6 rewritten. The parser's reserved names (`parser.rs:216-237`) are now in one place (`expr_zero::reserved`). `I` is an opaque atom (never `NonZero`). `pi` and `e` are named constants: `NonZero` is claimed only when at most one of them occurs, by the transcendence of pi (Lindemann 1882) and of e (Hermite 1873); both together give `Unresolved`. `Symbol("oo")` or a NaN or infinite `Real` anywhere in the tree, also inside an opaque subterm, gives `Unresolved` at once. A finite `Real` is its exact binary value (a dyadic rational). The precondition is stated: opaque atoms are finite on a dense open set. On `$S/rev1/t_b1.rs` all six inputs of the review give `Unresolved`, and d((I^2 + 1) x dy) is kept as `(I^2 + 1)` and its `closedness` is `Unresolved`. The review's fix is not enough for `normalize`: `simplify` folds `0 * oo`, `0 * inf` and `0 * NaN` to `0` before any zero test runs (`$S/rev1/t_simp.rs`). So `normalize` now returns `Err(ValidationError)` for such input, and the form constructors refuse non-finite components (3.1). Alignment with S1's kernel classes is in 1.6.5. New or extended tests: `test_zero_test_cases` (the six inputs and more, 1.6.6), `test_normalize_and_finiteness`, `test_d_reserved_constants` (1.8), `test_non_finite_components_refused` (3.3). |
| B2: the caps bounded memory, not time | Section 1.6.3 rewritten. One cumulative work budget per call (`ZERO_TEST_WORK = 100_000` units: nodes, term products, term insertions, atom comparisons). Coefficients are integers (N and D in Z[t]), so the bound `COEFF_BITS = 4096` is checked before each product, not after. With rational coefficients and a check after each operation, one product could still run unbounded: a test input ran over 100 s and was killed. A total-degree cap (2^20), a nesting cap (200, after flattening sum and product chains), a tree-depth cap (4096) and a cap of 64 distinct opaque atoms complete it. `a - a` with structurally equal operands is `Zero` at once when the tree has no division. Measured (debug): `zero_test(s^64 - (s*1)^64)`, s with 50 terms: 53.1 s to 0.044 s (`Unresolved`); d of (0, s^64) on R^2: 26.8 s to 0.05 s. Over 20 adversarial families the slowest call takes 0.10 s. Cost claims in 1.9 and 6.3 restated with this worst case. New tests: `test_zero_test_budget` (1.6.6), `test_d_budget` (1.8), both with wall-clock bounds 100 times the measured time. |
| Found while fixing B2 (not in the review) | Sums of many terms were built left-deep. `i_X ω` of a 1-form on R^1000 is a sum of 1000 terms and overflows a 2 MB test-thread stack on the old prototype (`$S/rev1/t_int_left.rs`). All sums built by the new code now go through `sum_signed` (balanced, depth ceil(log2 m) + 1). The revised prototype handles R^100000. The results of the reference programs are unchanged. New test: `test_interior_product_long_sum` (1.8). |
| Minor 1: I1 stated structurally, but `Neg` writes v into the odd slots | I1 is now stated up to value (1.3). Readers and the antisymmetry check only need values. |
| Minor 2: compile order | `tensor_field.rs:361` added to the list of test fixes (3.4). The `lie_derivative.rs:225` change moves to plan step 4 (6.1). |
| Minor 3: the REPL block only reaches shortcuts on R^1 | The comment in 4.3 now says so. No REPL test computes a nontrivial d until J1a fixes S9 and S14. |
| Minor 4: J1b recommendation 3 | Now "too many components to store" (4.4). |
| Minor 5: boundary rows assume `size_of::<Expr>() == 64` | `test_component_count_checked` derives the boundary from `size_of::<Expr>()` (2.3). |
| Minor 6: adjacent facades and limits | `PullBack::apply_to_form` for p > 1 (`maps.rs:346-350`) and `apply_to_covariant_tensor` for q >= 2 (`maps.rs:388-412`, the same facade) now return `Err(NotImplemented)`. Neither has a caller. Section 5 records the symplectic non-degeneracy facade (`PROJECT_TRACKER.md:2531`), the dense-storage limit (`volume_form` works up to R^8, where ω^4 needs 1.1 GB, and fails on R^10), and the 1-form pullback, which does not compose ω with the map. |
| Minor 7: two zero tests (MF1 and S1) | 1.6.5: the semantics agree (the two tests never contradict each other). A HANDOFF entry plans one public zero test in rustmath-symbolic. |
| Minor 8: small inaccuracies | `ContactForm::standard` wording fixed (3.4). The exact-binary-value rule for `Real` is documented (1.6.1). Test tables say "fails or does not compile on HEAD". |

Disagreements with the review: none of substance. Two choices within the review's options: `I` is
opaque rather than reduced with I^2 = -1, because `Chart::new` accepts a coordinate named `I`, and the
reduction would then be wrong. So `I*I + 1` is `Unresolved`, as the review expects. And `NonZero` stays
available for expressions with a single named constant, which the review's argument (pi and e
jointly) does not cover.

Items (docs/HANDOFF.md, "Open follow-ups", REPL split-outs of the X2 design):

1. **F2 (library part).** "repeated `d` and `wedge` build ever larger expression trees
   (`rustmath-manifolds/src/diff_form.rs:126`, `:173`; d(d(dx_0)) peaks at 574 MB at 10^6 components; the
   inner loop of `exterior_derivative` is also O(n^(p+2)) for O(n^(p+1)) components)". The REPL part of F2
   (`repl.rs:649`, `:706` clone a value on every assignment and lookup) belongs to J1b.
2. **N4 / S8.** "the library computes `n.pow(p + 1)` and `n.pow(p + q)` unchecked (`diff_form.rs:139`,
   `tensor_field.rs:105`, `:237`, `:280`); only the REPL arms check them" (checked arithmetic and a
   component guard returning an error).
3. **REPL guards.** Check the arms in rustmath-jupyter that guard these today (`FORM_COMPONENTS`,
   `DIFF_FORM_DEGREE` in `repl.rs`) and say what J1b should change once the library checks itself.

Scratch: `$S = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/MF1-manifolds`.
Programs are standalone `rustc` builds against the debug rlibs of HEAD (`$S/build.sh src.rs out`,
`$S/rlibs.txt`; the binaries were deleted to save disk, rebuild with `build.sh`). The crate sources were
not modified.

| Path | Content |
|---|---|
| `$S/repro_head.rs` | HEAD: wrong d, wedge, contraction, tensor-product layout; panics (chart without coordinates, `pow` overflow, `p + 1` / `p + q` overflow); silent `as u32` truncation |
| `$S/repro_head2.rs`, `repro_head3.rs`, `repro_head4.rs` | HEAD: Add/Neg return the left operand, `lie_derivative` of a 0-form returns the form, `interior_product` always errs, non-antisymmetric input accepted, `ContactForm::new(dz)` accepted, a closed form reported not closed, `LieDerivative::apply_to_form` turns a 2-form into a 1-form, `commutes_with_exterior_derivative` says false, integration reads the wrong slot |
| `$S/measure_head.rs` | HEAD time and VmHWM of d and wedge (`./measure_head d1 1000` etc.) |
| `$S/proto.rs` | Original prototype of every new algorithm (d, wedge, contraction, tensor product, interior product, add, zero test) on dense flat vectors in the crate's layout. Superseded by `$S/rev1/proto2.rs` (zero test and sums); it has an `NOZT` environment switch for one measurement only (do not copy) |
| `$S/measure_new.rs` | Prototype time and VmHWM on the same inputs |
| `$S/t_ref.rs`, `t_ref2.rs`, `t_contact.rs`, `t_zero.rs` | Prototype checks: Sage doc examples, gp values of `refs.gp`, the contact volumes, zero-test edge cases (all print `Zero` where a difference with the reference is tested) |
| `$S/refs.gp` | PARI/GP 2.15 reference values (`gp -q refs.gp`) |
| `$S/rev1/proto2.rs` | Revision 1 prototype: the zero test of 1.6 (reserved names, prescan, integer polynomials, work budget, caps) and the operations with balanced sums (`sum_signed`). The reference implementation now. Its limits can be overridden with `ZT_WORK` / `ZT_BITS` for experiments only; its `normalize` lacks the finiteness check of 1.4, which the design adds |
| `$S/rev1/t_b1.rs` (output `b1.txt`) | B1: reserved names, infinities, exact `Real` values, d((I^2 + 1) x dy) |
| `$S/rev1/t_simp.rs` | `simplify` at HEAD folds `0 * oo`, `0 * inf`, `0 * NaN`, `0 * (1/(y-y))` to `0` |
| `$S/rev1/t_b2.rs` (output `sweep.txt`), `r_dcost2.rs` | B2: time and work units of one zero test on 20 adversarial families; the review's d case |
| `$S/rev1/t_stack.rs`, `t_int_left.rs`, `t_int_bal.rs` | Recursion in a 2 MB thread: left-deep sums (HEAD `simplify` overflows at 5000 terms, the old prototype's `i_X ω` at n = 1000), balanced sums (n = 100000 works), the tree-depth cap |
| `$S/rev1/t_ref2.rs`, `t_ref22.rs`, `t_contact2.rs`, `t_zero2.rs`, `measure_new2.rs` | The checks and measurements of the original design, rerun on `proto2.rs` (same results; `ref2.out`) |
| `$S/rev2/proto3.rs` | Revision 2 prototype, the reference implementation now: `proto2.rs` plus the negation charge (`pneg`), `unsupported_derivative` and `check_derivative` (the guard of 1.4), the memoized `is_finite_expr`, `d_dense` returning the guard's error, and `lie_one_dense` (the Lie derivative of a 1-form of 3.4). Experiment counters as before (do not copy) |
| `$S/rev2/t_dguard.rs` (output `dguard.txt`) | B1 of review 2: forms d refuses and forms d accepts, 0-forms, the Lie derivative of 1-forms |
| `$S/rev2/r_adv.rs` (output `neg3.txt`) | Minor 1 of review 2: nested `Neg`, right-nested subtraction, small polynomials under 197 `Neg`; the reviewer's other probes rerun |
| `$S/rev2/t_b2.rs` (output `sweep3.txt`) | The 20 families of `sweep.txt` on `proto3.rs`, plus `bil` and `negp`; `old/bil_proto2.txt` is `bil` on `proto2.rs` |
| `$S/rev2/t_dag.rs` (output `dag.txt`) | Minor 6 of review 2: the scans on a DAG of fewer than 100 nodes whose tree has 2^41 leaves |
| `$S/rev2/measure3.txt`, `parse.txt`, `refs2.gp` | 1.9 rows on `proto2.rs` and `proto3.rs` side by side; which node kinds `parse` produces and what `differentiate` gives for them; gp values of Revision 2 (`gp -q refs2.gp < /dev/null`) |
| `$S/rev2/t_b1.rs`, `t_ref2.rs`, `t_ref22.rs`, `t_contact2.rs`, `t_zero2.rs`, `measure_new2.rs`, `t_stack.rs`, `t_int_bal.rs`, `r_dcost2.rs`; `old/` | The rev1 harnesses pointed at `proto3.rs` (`b1_3.txt`, `ref3.out`); `old/` holds the same harnesses on `proto2.rs` for comparison |

## 0. Summary

| Item | At HEAD | Fix |
|---|---|---|
| F2 | `exterior_derivative` is not the exterior derivative: slot i gets the sum over all j of ∂_j of `components[i]` (`diff_form.rs:145-156`), unsimplified, so each of the first n^p slots is a left-deep tree of n `Add` nodes. `wedge` returns the unsimplified tensor product (`(1 * 0)` nodes) with the degree relabelled. Both rows are already 🔴 in PROJECT_TRACKER.md (`:2520`, `:2522`). Measured (debug, standalone): d(dx_0) at n = 1000 takes 6.85 s and 253 MB; d(d(dx_0)) at n = 100 takes 1.39 s and 444 MB (574 MB in the REPL with its copies); the loop runs n^(p+2) iterations | The exterior derivative and the wedge product from their coordinate formulas [Lee 2013, ch. 14], computed on the C(n, k) strictly increasing index tuples only, each sum built balanced, simplified and then put through a new three-valued exact zero test (`expr_zero.rs`) with a fixed work budget per call. d∘d of a polynomial form is then literally 0, and d(dx_0) is all `Integer(0)`. Cost: (p+1)·C(n, p+1) derivatives plus n^(p+1) slot writes, plus at most one budgeted zero test (about 20 µs typical, at most 0.25 s measured, debug) per nonzero non-constant independent component. A component whose derivative rustmath-symbolic does not implement (named functions, `mod`, gamma, factorial, zeta, erf of the coordinate) makes d return `Err(NotImplemented)` instead of a fabricated 0. Measured on the prototype: d(dx_0), n = 1000: 0.12 s, 66 MB (the dense output itself, 64 bytes a slot); d(d(dx_0)), n = 100: 0.10 s, 66 MB |
| N4 / S8 | `n.pow(k as u32)` at `diff_form.rs:139`, `tensor_field.rs:105`, `:158`, `:237`, `:280`; `p + 1` (`diff_form.rs:132`, `:161`), `p + q` (`:182`), rank sums (`tensor_field.rs:105`, `:153`, `:273-274`) and `MultiIndex::to_flat` (`:31-38`) unchecked. Debug: panics. Release: wraps. Both: `rank as u32` truncates (rank 2^32 + 1 gives n^1). Also `j % symbols.len()` divides by zero for a chart without coordinates (`diff_form.rs:151`) | `tensor_field::component_count(n, rank) -> Result<usize>` (checked power, bytes bounded by `isize::MAX`), `zeroed_components(count) -> Result<Vec<Expr>>` (`try_reserve_exact`), `checked_add` on every degree and rank sum, `MultiIndex::to_flat` checked. New errors `ComponentCountOverflow`, `ComponentAllocationFailed`, `DegreeOverflow`. A chart whose dimension is not dim(M) is refused by `exterior_derivative` (`InvalidChart`) |
| REPL | `check_form_components` (n^degree <= 10^6), `p.checked_add(1)` / `p.checked_add(q)` in the arms, `DIFF_FORM_DEGREE = usize::MAX / 2` | Section 4: what J1b should change. MF1 itself changes only one REPL test block (the library now refuses the REPL's one-name chart on R^3) |

Adjacent facades are fixed because the rewrite of `diff_form.rs` and `tensor_field.rs` touches them and
leaving them would keep a function returning fabricated values next to the new code (section 3):
tensor contraction (the `:237` line of N4 sits in it; it returns n(n-1)/2 for every input), the
tensor-product index layout for mixed ranks, `interior_product`, `lie_derivative`, `Add`, `Neg`, the
antisymmetry check in `from_components` / `from_tensor`, the callers whose answers change with
d and wedge (`SymplecticForm::is_closed` and `volume_form`, `DeRhamCohomology::is_closed`,
`ContactForm::is_contact` and `standard`, `LieDerivative::apply_to_form` and
`commutes_with_exterior_derivative`), the pullback of p-forms, p > 1, which returned its input
and now returns `Err(NotImplemented)` (`maps.rs:346-350`, with the same facade at `:388-412`), and the
pushforward of contravariant tensors of rank >= 2, which has the same facade (`maps.rs:485-487`) and
now returns `Err(NotImplemented)` too. The Lie derivative of a 1-form keeps its formula and gains the
derivative guard of d (3.4).

Nothing of the three items is declared out of scope. Section 5 lists defects found on the way that this
unit does not fix (integration quadrature, the missing pullback and pushforward of higher-rank tensors,
`apply_to_tensor`, unchecked `dim * dim` and `spin.rs` powers, the symplectic non-degeneracy and
evaluation facades, the dense-storage limit, the `differentiate` facades of rustmath-symbolic, the
uncompiled `tracker_08_09_tests.rs`).

## 1. Item F2: repeated d and wedge

### 1.1 The defect at HEAD

Code (`rustmath-manifolds/src/diff_form.rs`):

```rust
// :139-156
let new_size = n.pow((p + 1) as u32);
let mut d_components = vec![Expr::from(0); new_size];
for i in 0..new_size {
    let mut sum = Expr::from(0);
    for j in 0..n {
        if i < components.len() {
            sum = sum + components[i].differentiate(&symbols[j % symbols.len()]);
        }
    }
    d_components[i] = sum;
}
```

* It is not d. Slot i of the (p+1)-form gets Σ_j ∂_j ω_i for the first n^p slots and 0 elsewhere. There
  is no antisymmetrization and no sign. `$S/repro_head.rs` on R^2 (chart x, y):
  `d(xy dx + x^2 dy)` gives the dense components `[(y + x), (2 * x), 0, 0]` (shown after `simplify`). The exterior derivative is
  x dx∧dy, dense `[0, x, -x, 0]` (gp: `deriv(x^2, x) - deriv(x*y, y)` prints `x`).
* `wedge` (`:173-198`) returns `tensor_product` (`tensor_field.rs:268-298`) with the degree relabelled.
  `dx∧dx` gives `[(1 * 1), (1 * 0), (0 * 1), (0 * 0)]` (true: 0), `dx∧dy` gives
  `[(1 * 0), (1 * 1), (0 * 0), (0 * 1)]` (true: `[0, 1, -1, 0]`).
* Nothing is simplified. `Expr`'s `+` and `*` build nodes (`rustmath-symbolic/src/expression.rs:506-536`),
  `differentiate` does not simplify (`differentiate.rs:24`). So d(dx_0) puts a left-deep tree of n zeros in
  each of its first n slots, and every further d differentiates those trees again. That is the growth of
  F2: d(d(dx_0)) holds 10^6 dense slots plus about 2·10^6 tree nodes at n = 100.
* The loop is `for i in 0..n^(p+1) { for j in 0..n {...} }`: n^(p+2) iterations, n^(p+1) of them do
  nothing but test `i < components.len()`.

Measured (`$S/measure_head.rs`, debug build of HEAD, standalone program, this container):

| Input | HEAD time | HEAD VmHWM |
|---|---|---|
| d(dx_0), n = 1000 (10^6 slots) | 6.85 s | 253 MB |
| dx_0 ∧ dx_0, n = 1000 | 0.29 s | 253 MB |
| d(d(dx_0)), n = 100 (10^6 slots) | 1.39 s | 444 MB |
| d of ω_i = x_{i+1}, n = 300 / 600 | 0.30 s / 1.54 s | 25 MB / 93 MB |

The HANDOFF figure (574 MB) is the REPL, which holds up to three copies of a value (repl.rs comments at
`:10538-10552`).

### 1.2 Why the fix is the correct algorithm

A faster or simplified version of the current loop would still return fabricated values (CLAUDE.md:
a facade is a defect). The growth also has a mathematical cause: d∘d = 0 and α∧α = 0 for a 1-form, so
the correct results collapse to zero, while the facade keeps summing derivatives of garbage. The fix is
therefore the exterior derivative and the wedge product themselves, computed on independent components,
with each result simplified and zero-tested.

### 1.3 Conventions and storage invariant

Convention (the "determinant convention" of J. M. Lee, *Introduction to Smooth Manifolds*, 2nd ed.,
GTM 218, Springer 2013, ch. 14, sections "The Algebra of Alternating Tensors", "The Wedge Product"; it is
also SageMath's, `sage.manifolds.differentiable.diff_form`, and the one the doc comments at
`diff_form.rs:121-125`, `:169-172` already state): a p-form is ω = Σ_{I increasing} ω_I dx^I, and its
components are those of the alternating tensor, ω_{i_1...i_p} = ω(∂_{i_1}, ..., ∂_{i_p}). So
ω_{σ(I)} = sgn(σ) ω_I and a component with a repeated index is 0.

Storage stays dense: per chart a `Vec<Expr>` of n^p slots in the layout of `MultiIndex::to_flat` (first
index most significant). The callers read `form.tensor().components(chart)` (maps.rs, integration.rs,
lie_derivative.rs, the REPL's `missing_components_error`), so a change of representation would ripple into
rustmath-jupyter, which J1b edits. Antisymmetric storage (C(n, p) slots, Sage's `CompFullyAntiSym`) is a
possible later step; it is not needed to fix the items, because the measured growth was the trees, not
the slots.

Invariant (I1), new, stated up to value: in every chart, the slot of an increasing tuple I holds an
expression v_I; the slot σ(I) holds an expression equal in value to sgn(σ) v_I; and a slot with a
repeated index holds the literal `Expr::Integer(0)`. "Equal in value" means equal as functions on the
chart domain where both are defined. The operations below write exactly `v_I` (even σ) and
`neg(v_I) = (-v_I).simplify()` (odd σ) through `scatter`; `Neg` writes `neg(v_I)` and the old `v_I`
(section 3.3), which is equal in value to `neg(neg(v_I))`. Every constructor establishes (I1)
(section 3.1: `from_components` and `from_tensor` check antisymmetry with the zero test and rebuild the
vector). The operations read only increasing tuples, except the interior product, which reads (k, J)
with k anywhere; (I1) makes those reads consistent in value, which is all any reader needs. Structural
equality of forms (`PartialEq`) is not part of the invariant.

Invariant (I2), kept from HEAD: a form of degree > n is zero. d, wedge, interior product and Lie
derivative return such forms as `DiffForm::new(manifold, degree)` (no components, nothing allocated),
as HEAD does at `:130-133` and `:182-185`.

### 1.4 The exterior derivative

Formula (Lee 2013, ch. 14, "The Exterior Derivative": dω = Σ_J dω_J ∧ dx^J, written per component; it
agrees with the comment at `diff_form.rs:124`, (dω)_{i_0...i_p} = (p+1) ∂_{[i_0} ω_{i_1...i_p]}):

    (dω)_I = Σ_{k=0}^{p} (-1)^k ∂_{i_k} ω_{I \ i_k}      for I = (i_0 < i_1 < ... < i_p).

I \ i_k is again increasing, so only stored increasing tuples are read.

Algorithm (`$S/rev2/proto3.rs`, `d_dense`):

1. `k = p.checked_add(1)`, else `Err(DegreeOverflow { left: p, right: 1 })`.
2. If p >= n: `Ok(DiffForm::new(manifold, k))` (I2; HEAD order, before any lookup).
3. `w = self.tensor.components_ref(chart)?` (`NoComponentsInChart`; a new crate-private borrowing
   accessor, so the input is not cloned as `components()` does at `tensor_field.rs:162-166`).
4. If `chart.dimension() != n`: `Err(InvalidChart(format!("the chart '{}' has dimension {}, but the
   manifold has dimension {}", ...)))`. This replaces `symbols[j % symbols.len()]` (division by zero for a
   chart without coordinates, silent reuse of coordinates for a short chart).
5. `count = component_count(n, k)?`; `out = zeroed_components(count)?` (section 2).
6. For each increasing k-tuple I in lexicographic order (`next_comb`): for each position kk, take
   `w_J = w[flat(I without i_kk)]`; skip it if `w_J.is_constant()`; else
   `check_derivative(w_J, &symbols[i_kk])?` (the derivative guard below: `Err(NotImplemented)` when
   `differentiate` would return a fabricated value), then
   `t = w_J.differentiate(&symbols[i_kk]).simplify()`; skip a literal zero; push `(t, kk odd)` onto a
   vector of signed terms. If any term remains: `v = normalize(sum_signed(terms))?` and
   `scatter(out, I, v)`.
7. `Ok(DiffForm { tensor: TensorField::from_dense_unchecked(manifold, 0, k, chart, out), degree: k })`
   (a crate-private constructor: the length is right by construction, (I1) holds by construction).

`sum_signed(terms: Vec<(Expr, bool)>) -> Option<Expr>` (crate-private, in `expr_zero.rs` so that
`tensor_field.rs` can use it too): the sum of ±t (true means negated), combined pairwise so that the tree has depth ceil(log2 m) + 1 and only `Add`
and `Sub` nodes (one `Neg` when every term is negated). Pairs combine as a + b, a - b, b - a, or
(a + b) negated. Every sum built by the new code (d, wedge, interior product, contraction) uses it, so no
consumer (`simplify`, the zero test, `Display`, `Drop`) recurses once per term. Evidence
(`$S/rev1/t_stack.rs`, `t_int_left.rs`, `t_int_bal.rs`, 2 MB thread as in `cargo test`): HEAD `simplify`
overflows the stack on a left-deep sum of 5000 terms; the original prototype's `i_X ω` of a 1-form on
R^1000 overflows; with `sum_signed`, R^100000 works. For d and wedge, whose sums have at most
C(p+q, p) terms, the shape is the same as before for up to three terms.

`normalize(e) -> Result<Expr>` (`expr_zero.rs`, section 1.6): if `!is_finite_expr(&e)` (the symbol `oo`
or a NaN or infinite `Real` occurs anywhere), `Err(ValidationError(format!("the expression {} is not
finite", e)))`; this check comes first because `simplify` folds `0 * oo` to `0` (1.6.1). Otherwise
`s = e.simplify()`; a rational constant is returned as is; otherwise if `zero_test(&s) == Zero` return
`Expr::Integer(0)`, else `s`.

`scatter(out, I, v)`: nothing for a literal-zero v. Else `nv = (-v).simplify()` once, then Heap's
algorithm over the permutations of I, flipping the sign at each swap, writes `v` or `nv` (`Expr` clones of
`Binary`/`Unary` nodes are `Arc` clones). (p+1)! writes per tuple, n!/(n-p-1)! <= n^(p+1) in total.

The derivative guard (Revision 2). rustmath-symbolic's `differentiate` is not the derivative on six node
kinds. It returns a value anyway:

| Node | `differentiate` at HEAD | True derivative |
|---|---|---|
| `Function(name, args)`, except `root_sum` with 2 arguments | 0 (`differentiate.rs:230-234`; tracker 🛑 `PROJECT_TRACKER.md:2106`) | the chain rule through the unknown derivative of the function |
| `Binary(Mod, a, b)` | 0 (`:90`) | a' wherever a mod b is continuous (x mod 3 has derivative 1 off 3Z) |
| `Unary(Gamma, a)` | 0 (`:184-188`) | Γ(a) ψ(a) a' |
| `Unary(Factorial, a)` | 0 (`:191`) | Γ(a + 1) ψ(a + 1) a' |
| `Unary(Zeta, a)` | 0 (`:204`) | ζ'(a) a' |
| `Unary(Erf, a)` | (500/443) exp(-a^2) a' (`:194-198`) | (2/sqrt(π)) exp(-a^2) a'; 500/443 = 1.12867, 2/sqrt(π) = 1.12838 (`$S/rev2/refs2.gp`) |

Every other kind is differentiated by its rule in `differentiate.rs:24-206`: the field operations, `Pow`
(a constant exponent, or f^g (g' log f + g f'/f)), the elementary `Unary` functions, `Abs` (sign(f) f')
and `Sign` (0, true off the zero set of f), and `root_sum` with two arguments (termwise, `:218-229`). The
parser produces `Unary` nodes for the elementary functions and for gamma, zeta, erf and factorial, and a
`Function` node for any other name (`$S/rev2/parse.txt`: `bessel_j(0, x)` and `f(x)` are `Function`). A
hand-built `Expr::Function("sin", ..)`, as `catalog.rs:197` builds one, is a `Function` too.

```rust
// expr_zero.rs, crate-private (used by diff_form.rs and lie_derivative.rs)

/// The outermost subterm of `e` that contains `var` and is a `Function` other than `root_sum` with two
/// arguments, a `Unary` Gamma, Factorial, Zeta or Erf, or a `Binary` Mod; None when there is none.
/// Iterative pre-order scan with an explicit stack; each entry carries its outermost ancestor of these
/// kinds, and a `Symbol` equal to `var` below such an ancestor returns that ancestor. A leaf returns at
/// once. After MEMO_AFTER = 64 visited nodes, a child held by an `Arc` with strong count > 1 is pushed at
/// most once per (address, below-an-ancestor) pair, and not again below no ancestor once it was visited
/// below one. A child with strong count 1 is reachable only through its single parent, so the scan is
/// linear in the number of distinct nodes.
pub(crate) fn unsupported_derivative<'a>(e: &'a Expr, var: &Symbol) -> Option<&'a Expr>;

/// Ok(()) when unsupported_derivative(e, var) is None. Otherwise, for that subterm u:
///   Err(ManifoldError::NotImplemented(format!("the derivative of {} with respect to {} is not
///   available: rustmath-symbolic's differentiate does not implement it", u, var.name())))
/// The code comment cites differentiate.rs and PROJECT_TRACKER.md:2106; the message has no line numbers.
pub(crate) fn check_derivative(e: &Expr, var: &Symbol) -> Result<()>;
```

A subterm of these kinds that does not contain `var` is constant in `var`, and `differentiate` gives 0
for it, which is right. So it is allowed: d(f(y) dy) and d(gamma(3) dy) are the literal 0, and so is
d(f(x) dx), because ∂_y f(x) = 0. The arguments of `root_sum` are scanned like any other subterm:
`root_sum(t^2 - 2, t*f(x))` is refused, `root_sum(t^2 - 2, t*x)` is differentiated. Evidence:
`$S/rev2/dguard.txt` (the 12 refused and 9 accepted 1-forms of `test_d_refuses_unsupported_derivatives`,
two 0-forms, the Lie derivative cases of 3.4). The scan runs only on non-constant components (step 6
skips constants first). It costs about 0.25 µs on a 3-node component in debug (1.9).

Precondition: on every node kind that `unsupported_derivative` admits, `differentiate` returns the
derivative wherever the expression is defined. This holds by the rules listed above. A new node kind
or a new facade in rustmath-symbolic must be added to the scan. Once the FunctionRegistry derivative
hooks (tracker `:2106`) are wired into `differentiate`, the scan can admit the functions that have one
(section 5).

Correctness: under the precondition, step 6 computes the formula for every increasing I, or returns
`Err(NotImplemented)` before it would use a derivative that `differentiate` does not compute. The
antisymmetric extension of a function on increasing tuples is unique, and `scatter` writes exactly it;
repeated-index slots stay 0. The zero test only replaces a component by 0 when it is decided zero
(soundness in 1.6.4). Checked against external values in 1.8.

### 1.5 The wedge product

Formula (Lee 2013, ch. 14, "The Wedge Product", determinant convention; Sage's check
`s[1,2,3] == a[1]*b[2,3] + a[2]*b[3,1] + a[3]*b[1,2]`): for I increasing of length p + q,

    (ω ∧ η)_I = Σ_{A ⊂ I, |A| = p} sgn(A, I \ A) ω_A η_{I \ A},

where A and I \ A are increasing and sgn(A, I \ A) is the sign of the shuffle that sorts the concatenation:
(-1) to the number of pairs (a ∈ A, b ∈ I \ A) with b before a in I.

Algorithm (`$S/rev2/proto3.rs`, `wedge_dense`, the same as in `proto2.rs`):

1. Same manifold (`Arc::ptr_eq`), else `DifferentManifolds` (as HEAD).
2. `k = p.checked_add(q)`, else `DegreeOverflow { left: p, right: q }`.
3. If k > n: `Ok(DiffForm::new(manifold, k))` (I2; HEAD order).
4. `a = self` components, then `b = other` components (`NoComponentsInChart`, HEAD order: self first).
   No chart-dimension check: the wedge does not use coordinates (keeps the REPL test of section 4 working).
5. `count = component_count(n, k)?`, `out = zeroed_components(count)?`.
6. For each increasing I, for each p-subset S of positions 0..k (lexicographic), split I into
   `ia`, `ib`, count the inversions (`inv += p - x` when a b-element is placed with x a-elements before it),
   skip a literal-zero factor, `t = (a[flat(ia)] * b[flat(ib)]).simplify()`, push `(t, inv odd)`.
   `v = normalize(sum_signed(terms))?`, `scatter(out, I, v)`. p = 0 or q = 0 work through the same loop
   (one split).
7. `Ok(DiffForm { tensor: from_dense_unchecked(.., 0, k, chart, out), degree: k })`.

Cost: C(n, p+q)·C(p+q, p) products, plus n^(p+q) slot writes.

### 1.6 The exact zero test (new module `expr_zero.rs`)

The simplifier of rustmath-symbolic only folds constants and identities (`simplify.rs:376-628`): it does
not cancel `x - x`. Without a zero test, d∘d of a polynomial form stays a tree like `(2 * x) - (2 * x)`,
and `is_closed` (structural `is_zero`, `diff_form.rs:251`, `tensor_field.rs:301-310`) would answer false
for a closed form. The zero test makes `normalize` collapse such components, and gives the three-valued
answers of section 3.3. Reference implementation: `$S/rev2/proto3.rs` (`zero_test`, `Conv`, `prescan`).

```rust
/// rustmath-manifolds/src/expr_zero.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroTest { Zero, NonZero, Unresolved }

/// The symbol names that rustmath-symbolic's parser gives a fixed meaning (parser.rs:216-237).
/// Kept here in one place; see 1.6.5 for the relation to S1's kernel classes.
pub mod reserved {
    pub const IMAGINARY_UNIT: &str = "I";            // parser: "I" and "i"
    pub const NAMED_CONSTANTS: [&str; 2] = ["pi", "e"]; // parser: "pi", "π", "e"
    pub const INFINITY: &str = "oo";                 // parser: "oo", "inf", "infinity"
}

/// Work units one call may spend (1.6.3).
pub const ZERO_TEST_WORK: u64 = 100_000;

/// Decides whether `e` is the zero function. `Zero` and `NonZero` are proofs under the precondition of
/// 1.6.1; `Unresolved` is not a decision. Equals `zero_test_with_budget(e, ZERO_TEST_WORK)`.
pub fn zero_test(e: &Expr) -> ZeroTest;

/// The same with another work budget. A smaller budget can only turn answers into `Unresolved`.
pub fn zero_test_with_budget(e: &Expr, work: u64) -> ZeroTest;

/// False when the symbol `oo` or a NaN or infinite `Real` occurs anywhere in `e`. Iterative scan,
/// memoized on shared nodes like `unsupported_derivative` (1.4): linear in the number of distinct nodes.
pub fn is_finite_expr(e: &Expr) -> bool;

/// Err(ValidationError) when `!is_finite_expr(&e)`; else `e.simplify()`, replaced by
/// `Expr::Integer(0)` when `zero_test` decides it is zero.
pub(crate) fn normalize(e: Expr) -> Result<Expr>;
```

#### 1.6.1 Semantics, reserved names, precondition

What the answers are about: `e` denotes a function of independent real indeterminates. Every `Symbol`
other than the reserved names is such an indeterminate (chart coordinates and free parameters, as in
Sage's symbolic ring and as S1's `Indeterminate` class). The reserved names:

| Name | Meaning in rustmath-symbolic | Treatment in `expr_zero` |
|---|---|---|
| `I` | imaginary unit (parser maps `I` and `i` to it, `parser.rs:227-232`; S1 reduces I^2 = -1) | opaque atom: a `Zero` answer holds for any value of it; `NonZero` is never claimed when it occurs |
| `pi`, `e` | π and Euler's number (`parser.rs:216-226`) | named constant atoms: `NonZero` is claimed only when at most one of them occurs (1.6.4) |
| `oo` | infinity (parser maps `oo`, `inf`, `infinity` to it, `parser.rs:233-237`) | `Unresolved` at once, wherever it occurs, also inside an opaque subterm |

`Real(f)`: a finite f is its exact binary value, a dyadic rational (every finite IEEE 754 double is one).
So `0.5*x - x/2` is `Zero`, `0.1*x - x/10` is `NonZero` (0.1 is 3602879701896397/2^55, not 1/10), and a
float residue of inexact arithmetic, such as `0.1 + 0.2 - 0.3` folded by the caller to 5.55e-17, is
`NonZero`. A NaN or infinite `Real` anywhere gives `Unresolved` at once. Documented in the module doc.
The exact value is decoded from `f64::to_bits` (sign, exponent, mantissa: m·2^e; `exact_real` in
`$S/rev2/proto3.rs`). `Rational::from_f64` (`rustmath-rationals/src/rational.rs:166-199`) must not be
used: it returns an approximation, found by trying every denominator up to 10^6.

`I` is opaque rather than reduced with I^2 = -1 because `Chart::new` accepts any coordinate names (the
REPL's `chart(E, "I")` creates one). With a coordinate named `I` the reduction would decide `I*I + 1` to
be zero, which is false. The opaque treatment is sound in both readings. `pi` and `e` are sound in both
readings too (1.6.4). A coordinate named `oo` makes every expression that contains it non-finite; the
module doc says that such a chart cannot be used with forms.

Precondition (module doc): every opaque subterm (function applications, `Unary` other than `Neg`,
`Mod`, powers whose exponent is not an `Integer` of absolute value at most 2^20, the symbol `I`) is finite
on a dense open subset of the domain. So `log(0)` and `1 / (sin(x)^2 + cos(x)^2 - 1)` are outside the
contract, like a divisor that vanishes identically without being the zero polynomial. A divisor that
converts to the zero polynomial is caught (`Unresolved`).

Why `normalize` checks finiteness itself: `simplify` folds `0 * a` to `0` for every a, so
`normalize(0 * oo)` would store a literal 0 before any zero test runs (`$S/rev1/t_simp.rs`, HEAD):

| Input | `simplify()` at HEAD |
|---|---|
| `0 * oo`, `oo * 0`, `0 * Real(inf)`, `Real(NaN) * 0`, `0 * (1/(y - y))` | `0` |
| `oo - oo`, `Real(inf) - Real(inf)`, `inf*x - inf*x`, `0 / 0` | unchanged |

So `normalize` returns `Err(ValidationError)` for non-finite input, and the form constructors refuse
non-finite components (3.1). `0 * (1/(y - y))` is finite in this sense and folds to 0; it is outside the
precondition (the divisor vanishes identically), as at HEAD.

#### 1.6.2 Algorithm

Buchberger and Loos, "Algebraic Simplification", in *Computer Algebra: Symbolic and Algebraic
Computation*, Springer 1982, pp. 11-43 (canonical simplifier for rational functions); MAGMA Chapter_24
§24.4.2 (equality of multivariate polynomials) and Chapter_41 (rational function fields). Steps:

1. Prescan, iterative with an explicit stack, one work unit a node: `Unresolved` when `oo` or a NaN or
   infinite `Real` occurs, when the node count exceeds the budget, or when the depth exceeds
   `MAX_TREE_DEPTH = 4096`. It records whether a `Div`, a negative `Integer` power or a non-`Integer`
   power occurs ("has division").
2. Fast paths: `Integer` and `Rational` by value; `Real` by exact value (0.0 is `Zero`, any other finite
   value `NonZero`); `Sub(l, r)` with `l == r` structurally and no division in the tree is `Zero`.
3. Conversion to a fraction N/D with N, D in `rustmath_polynomials::MultivariatePolynomial<Integer>`
   (integer coefficients; `Integer` implements `Ring`, `integer.rs:1432`) over atoms t_1..t_m:

| `Expr` | N/D |
|---|---|
| `Integer(n)` | n / 1 |
| `Rational(u/v)` | u / v (two constant polynomials) |
| finite `Real(f)` | exact value u / 2^k |
| `Symbol` | its atom (`HashMap` keyed by `Symbol::id`) / 1; the kind is Indeterminate, Named (`pi`, `e`) or Opaque (`I`) |
| `Function(..)`, `Unary(op ≠ Neg, ..)`, `Binary(Mod, ..)`, `Binary(Pow, b, k)` with k not an `Integer` of absolute value at most 2^20 | opaque atom / 1; compared with `==` against the opaque atoms seen so far (at most `MAX_OPAQUE = 64`) |
| `Unary(Neg, a)` | -N_a / D_a |
| chain of `+` and `-` along the left spine | flattened iteratively; then N_a ± N_b over D when D_a == D_b, else (N_a D_b ± N_b D_a) / (D_a D_b) |
| chain of `*` along the left spine | flattened iteratively; N_a N_b / D_a D_b |
| `a / b` | N_a D_b / D_a N_b; `Undefined` when N_b is the zero polynomial |
| `Pow(b, k)`, `Integer` k with abs(k) <= 2^20 | binary powering; for k < 0 swap N and D (`Undefined` if N = 0) |

   The exponent is read with `rustmath_core::NumericConversion::to_i64 -> Option<i64>`
   (`integer.rs:1588`), not the panicking inherent `Integer::to_i64` (`integer.rs:88`, HANDOFF S7). The
   recursion that remains after flattening (an operand of a different kind of node) is at most
   `MAX_NESTING = 200` deep.
4. Answer. `Undefined` or an exhausted budget or limit: `Unresolved`. N = 0: `Zero`. Otherwise let V be
   the atoms that occur in N or D (atoms that cancelled do not count: `sin(x) - sin(x) + y` is `NonZero`).
   An Opaque atom in V: `Unresolved`. Two named constants in V: `Unresolved`. Otherwise `NonZero`.

#### 1.6.3 The work budget

One budget of `ZERO_TEST_WORK = 100_000` units per call. Every step is charged before it runs:

| Step | Units |
|---|---|
| prescan | 1 a node |
| an opaque atom of s nodes | s × (1 + number of opaque atoms so far): its node count and the `==` comparisons |
| sum of polynomials | weight(smaller) + 1; the smaller is inserted into the larger |
| product of polynomials with s and t terms | s·t + s·weight(b) + t·weight(a) |
| negation of a polynomial (`Unary(Neg)`, the right operand of a subtraction, a negated first term of a chain) | weight + 1: `MultivariatePolynomial::neg` rebuilds the map with one `add_term` per term (`multivariate.rs:804-813`) |
| clone of an operand (powering) | weight |
| comparison D_a == D_b | number of terms of the smaller |
| recomputing a coefficient bound | number of terms |

Every operation that touches the terms of a polynomial is in this table, except the final scan of the
variables of N and D (step 4), which reads polynomials whose construction was charged, and the drop of
the intermediate polynomials, likewise. `$S/rev1/proto2.rs` did not charge negation (review 2): 197 nested `Neg` around a product of 10^4 terms took 12.9 s there for 51396
charged units. `$S/rev2/proto3.rs` charges it (`pneg`).

weight(p) = Σ over the terms of (1 + number of variables in the monomial). It covers the cost of
`Monomial::mul` (a `BTreeMap` merge) and of the degree scan.

Limits, each exhausted limit gives `Unresolved`:

* `COEFF_BITS = 4096`. Each polynomial carries an upper bound of the bit length of its coefficients. A sum
  has at most max + 1 bits; a product at most b_a + b_b + bitlen(min(s, t)) bits. The operation is refused
  when the bound exceeds 4096, after recomputing the exact bounds once. With integer coefficients the
  bound holds before the operation runs. With rational coefficients it would not: a sum of m rationals
  has a denominator up to the product of the m denominators. Evidence: the first version of this
  revision used `Rational` coefficients with the bit check after each operation, and
  (Σ_{i<60} ((2i + 1) / (2^999 + 2i + 3)) x^i)^2 - same ran over 100 s and was killed. With integer
  coefficients it is `Unresolved` after 881 units, under 1 ms (`$S/rev1/sweep.txt`, `bigr 60 1000`).
* `MAX_DEGREE = 2^20` (total degree of a polynomial). `Monomial` stores `u32` exponents and adds them
  unchecked (`multivariate.rs:59-65`), so this cap keeps the conversion from overflowing.
* `MAX_TREE_DEPTH = 4096` (prescan), `MAX_NESTING = 200` (conversion), `MAX_OPAQUE = 64`. The derived
  `PartialEq` of `Expr` recurses; the depth cap keeps the `==` comparisons safe in a 2 MB test thread
  (measured: sin applied 4000 times works, `$S/rev1/t_stack.rs`).

Measured (`$S/rev2/t_b2.rs` on `proto3.rs`, output `sweep3.txt`; debug rlibs, prototype at opt-level 0,
this container; units used and time of one call):

| Input | Answer | Units | Time |
|---|---|---|---|
| s^64 - (s*1)^64, s = 1 + x + ... + x^49 (review 1 B2; 53.1 s on the original prototype) | Unresolved | 100000 | 0.048 s |
| same, s with 10 terms (`Zero` in 1.78 s on the original prototype; the budget gives this decision up) | Unresolved | 100000 | 0.024 s |
| (x1 + ... + x10)^4 - ((x1 + ... + x10)*1)^4 | Zero | 53294 | 0.056 s |
| (x1 + ... + x30)^3 - (same*1)^3 | Unresolved | 100000 | 0.119 s |
| Π_{i<=12} (1 + x_i) - (same*1) (4096 terms, 12 variables a monomial) | Unresolved | 100000 | 0.115 s |
| Σ_{i<=100} 1/(x + i) - (same*1) | Unresolved | 100000 | 0.088 s |
| (Σ_{i<60} c_i x^i)^2 - same*1, c_i of 2040 bits | Zero | 44840 | 0.050 s |
| Σ of 30000 distinct symbols - (same*1) | Unresolved (prescan) | - | < 1 ms |
| Σ_{i<64} sin(x + i) - (same*1) / with 65 terms | Zero / Unresolved (MAX_OPAQUE) | 26372 / 9489 | 2 ms / 1 ms |
| P - P*1, P = (a_1 + ... + a_k)(b_1 + ... + b_k), k = 70 / 80 (`bil`) | Zero / Unresolved | 80084 / 100000 | 0.141 s / 0.163 s |
| 197 nested `Neg` around P, k = 80 / 100 (`negp`; review 2) | Unresolved / Unresolved | 100000 / 100000 | 0.251 s / 0.189 s |
| one `Neg` around P, k = 100 | NonZero | 81201 | 0.192 s |

The 20 families of `$S/rev1/sweep.txt` give the same answers as on `proto2.rs`; a decided call uses up
to 14 % more units (`lin 1000`: 14003 to 16004).
`bil 80` is `Zero` on `proto2.rs` (85123 units, 0.174 s, `$S/rev2/old/bil_proto2.txt`) and `Unresolved`
on `proto3.rs`, because the negation of the 6400-term operand now counts. The slowest of the 33 calls of
`sweep3.txt` takes 0.25 s, at 0.06 to 2.5 µs a unit; the slow units are the `BTreeMap` inserts of large
polynomials in two-variable monomials. So one call costs at most about 0.25 s in a debug build. A typical
component of d or wedge costs about 20 µs (1.9).

#### 1.6.4 Soundness

By induction on the conversion: for every assignment of values to the atoms at which no denominator met
in the conversion vanishes, the value of `e` is N/D at that assignment. The fast paths are exact
(Integer, Rational, the exact value of a Real; `a - a` is 0 wherever a is defined, and the tree has no
division).

(a) `Zero`: N is the zero polynomial, so e = 0 at every assignment where it is defined, whatever the
values of the opaque atoms, of `I`, `pi` and `e`. By the precondition, e is defined on a dense open set,
so e is the zero function there and, being continuous where defined, the zero function.

(b) `NonZero`: N ≠ 0, no opaque atom occurs in N or D, and at most one named constant c occurs. Write N
as a polynomial in the indeterminates with coefficients in Z[c]. Some coefficient is a nonzero
polynomial in c, and c is transcendental (π: F. Lindemann, "Über die Zahl π", Math. Ann. 20 (1882)
213-225; e: C. Hermite, "Sur la fonction exponentielle", C. R. Acad. Sci. 77 (1873)). So N(x, c) is a
nonzero real polynomial in the indeterminates, and so is D(x, c). A nonzero polynomial does not vanish on
any nonempty open set, so N/D is not identically zero on any chart domain. If a reserved name is used as
a coordinate instead, it is one more indeterminate and the argument holds as well. With both π and e the
argument fails (their algebraic independence is open), so that case is `Unresolved`.

(c) Opaque atoms may satisfy relations (sin^2 + cos^2 = 1), so N ≠ 0 proves nothing then: `Unresolved`.
By D. Richardson, "Some undecidable problems involving elementary functions of a real variable",
J. Symbolic Logic 33 (1968) 514-520, zero equivalence is undecidable for such expressions in general, so
`Unresolved` cannot be avoided.

(d) Infinities: `oo - oo`, `0 * oo` and `inf*x - inf*x` have no value, so neither answer is claimed. They
are caught in the prescan, before any fast path, also inside opaque subterms (`sin(oo) - sin(oo)`).
An opaque atom is not enough here, since t - t would give `Zero`.

(e) Budget: an exhausted budget only returns `Unresolved`, so the budget never changes a decision into
a wrong one; `normalize` then keeps the simplified expression.

#### 1.6.5 Relation to S1's zero test

Unit S1 (implemented before MF1) adds a three-valued zero test on `SeriesConst` in rustmath-symbolic
(`S1-symbolic.md` section 2.2, crate-private). The classes agree:

| Input | S1 (`KernelClass`, `zero_status`) | MF1 (`expr_zero`) |
|---|---|---|
| other `Symbol` | `Indeterminate` (generic parameter) | indeterminate |
| `I` | `ImaginaryUnit`, I^2 = -1 | opaque atom (no reduction, see 1.6.1) |
| `pi`, `e` | `NamedConstant`; one named constant alone gives `NonZero` (Lindemann, Hermite) | the same rule |
| `oo` | `Err(InvalidPoint)` | `Unresolved` |
| finite `Real` | a `NonZero` kernel | its exact value (also `NonZero` when f ≠ 0) |
| non-finite `Real` | `Err(Unsupported)` | `Unresolved` |

Both are sound for the same semantics, so they never contradict each other (one `Zero` where the other
says `NonZero`). They differ in reach: S1 decides more about constants (I^2 = -1, radicals, ball
arithmetic); MF1 handles rational functions of many indeterminates under a work budget. One public zero
test in rustmath-symbolic would serve both. That is a HANDOFF entry (section 5); MF1 keeps
`expr_zero.rs` crate-local because S1's type is crate-private and one crate per PR is the rule.

#### 1.6.6 Tests (module `expr_zero`)

`test_zero_test_cases` (all from `$S/rev1/b1.txt`, `$S/rev1/ref2.out`, `$S/rev1/t_zero2.rs`; "fails or
does not compile on HEAD": the module is new):

| Expected | Inputs |
|---|---|
| `Zero` | (x+y)^2 - x^2 - 2xy - y^2; 2x - x*2; x/y - x*y^(-1); sin(x) - sin(x); x^1000 - x^1000; x^(10^30) - x^(10^30); Real(0.5) - Real(0.5); x^0 - 1; I*x - x*I; pi*x - x*pi; pi*e - e*pi; Real(0.5)*x - x/2; Real(0.0) |
| `NonZero` | x - y; pi*x; pi - 3; e*x + 1; (pi - pi) + e*x; sin(x) - sin(x) + y; Real(0.1)*x - x/10; Real(0.1 + 0.2 - 0.3); Real(0.1)*x |
| `Unresolved` | sin(x)^2 + cos(x)^2 - 1; sin(x); x/(y - y); (0*x)^(-1); x^(1/2)*x^(1/2) - x; `parse("I*I + 1")` and its `simplify()`; `parse("i^2 + 1")`; `parse("I")`; oo - oo; `Expr::from(0) * parse("oo")` (not simplified); Real(inf)*x - Real(inf)*x; Real(NaN)*x - Real(NaN)*x; sin(oo) - sin(oo); `parse("infinity") - parse("inf")`; pi - e; 1/(y - y) - 1/(y - y); Σ_{i<65} sin(x + i) - (same*1) |

`test_zero_test_budget` (wall-clock bounds are at least 25 times the measured debug time, so the test is
not flaky):

* s = 1 + x + ... + x^49: `zero_test(s^64 - (s*1)^64)` is `Unresolved` within 5 s (measured 0.048 s).
* (x1 + ... + x10)^10 - ((x1 + ... + x10)*1)^10: `Unresolved`; the same with exponent 4: `Zero`.
* `zero_test_with_budget((x+y)^2 - x^2 - 2xy - y^2, 10)`: `Unresolved` (the default budget gives `Zero`).
* a sum of 30000 distinct symbols minus (the same times 1): `Unresolved` within 1 s.
* sin applied 4000 times to x, minus (the same times 1): `Zero`; applied 5000 times: `Unresolved`
  (`MAX_TREE_DEPTH`). Run in a thread with a 2 MB stack.
* (Σ_{i<60} ((2i + 1) / (2^999 + 2i + 3)) x^i)^2 - (same*1)^2 (rational coefficients with 1000-bit
  denominators): `Unresolved` within 1 s.
* Negation (review 2), P = (a0 + ... + a99)(b0 + ... + b99), 10^4 terms: 197 nested `Neg` around P is
  `Unresolved` within 5 s (measured 0.19 s). On `proto2.rs`, which does not charge negation, it is
  `NonZero` after 12.9 s, so this item fails there by its answer and by its time. 197 right-nested
  subtractions 0 - (0 - (... - P)): `Unresolved` within 5 s (0.18 s). -P alone: `NonZero` (81201 units,
  0.19 s). 197 nested `Neg` around x - y is `NonZero`, and that plus (x - y) is `Zero` (197 is odd; 2 ms;
  `$S/rev2/neg3.txt`).

`test_scans_on_shared_subtrees` (review 2; `$S/rev2/t_dag.rs`, `dag.txt`): starting from a leaf L, 40
times e = e + e or e = e * e (alternating; once with `e.clone()` twice, once with one `Arc` used for both
children), so the tree has 2^40 copies of L and the DAG fewer than 100 nodes. With L = x + y:
`is_finite_expr` is true and `unsupported_derivative(e, x)` is None. With L = gamma(y) + x: None. With
L = gamma(x) + y: `Some(gamma(x))`. With L = x + oo: `is_finite_expr` is false. Each call within 1 s
(measured under 1 ms; a tree walk would visit 2^41 leaves).

`test_normalize_and_finiteness`: `is_finite_expr` is false for `oo`, `sin(oo)`, `Real(NaN) * x`,
`Real(-inf)` and true for `x + pi + I`. `normalize(0 * oo)` is `Err(ValidationError)` (`simplify` alone
gives 0); `normalize(2x - x*2)` is `Integer(0)`; `normalize(x - y)` is `x - y`.

### 1.7 API changes for F2 (exact signatures)

`diff_form.rs` (signatures of the existing methods are unchanged):

```rust
pub fn exterior_derivative(&self, chart: &Chart) -> Result<DiffForm>;   // 1.4
pub fn wedge(&self, other: &DiffForm, chart: &Chart) -> Result<DiffForm>; // 1.5
```

`tensor_field.rs` (crate-private helpers used by diff_form.rs):

```rust
pub(crate) fn components_ref(&self, chart: &Chart) -> Result<&[Expr]>;
pub(crate) fn from_dense_unchecked(manifold: Arc<DifferentiableManifold>, contravariant_rank: usize,
    covariant_rank: usize, chart: &Chart, components: Vec<Expr>) -> TensorField; // debug_assert on the length
pub(crate) fn chart_components(&self) -> &HashMap<ChartId, Vec<Expr>>;      // for Add, Neg, scale
pub(crate) fn from_chart_components(manifold: Arc<DifferentiableManifold>, contravariant_rank: usize,
    covariant_rank: usize, map: HashMap<ChartId, Vec<Expr>>) -> TensorField;
```

`expr_zero.rs`: the items of 1.6 (`ZeroTest`, `reserved`, `ZERO_TEST_WORK`, `zero_test`,
`zero_test_with_budget`, `is_finite_expr`, crate-private `normalize`; private `COEFF_BITS`,
`MAX_DEGREE`, `MAX_TREE_DEPTH`, `MAX_NESTING`, `MAX_OPAQUE`, `MEMO_AFTER`) and the crate-private
helpers of 1.4:

```rust
pub(crate) fn sum_signed(terms: Vec<(Expr, bool)>) -> Option<Expr>;   // balanced, Add/Sub only
pub(crate) fn unsupported_derivative<'a>(e: &'a Expr, var: &Symbol) -> Option<&'a Expr>;
pub(crate) fn check_derivative(e: &Expr, var: &Symbol) -> Result<()>; // Err(NotImplemented)
```

`lib.rs`: `pub mod expr_zero;` and `pub use expr_zero::{zero_test, zero_test_with_budget, ZeroTest};`.

`Cargo.toml`: add `rustmath-polynomials = { path = "../rustmath-polynomials" }` (already in the graph
through rustmath-symbolic; no cycle).

### 1.8 Tests for F2 (new, in `diff_form.rs` unless stated)

A test chart `c` on a 3-manifold with coordinates x, y, z (`Chart::new("c", 3, vec!["x", "y", "z"])`,
added to a `DifferentiableManifold::new("R3", 3)`), and on R^2 the `EuclideanSpace::new(2)` default chart
(coordinates x, y). "Equal to the reference" means `zero_test(&(ours - reference)) == ZeroTest::Zero`.
Reference commands are in `$S/refs.gp` (run `gp -q refs.gp`); the Sage values are quoted from the Sage
doc page of `sage.manifolds.differentiable.diff_form` (doc.sagemath.org, fetched for this design).
The HEAD column says what the same computation gives at HEAD (`$S/repro_head*.rs`); the tests use new
APIs (`zero_test`, `from_independent_components`, `vanishes`), so "fails" means "fails or does not
compile on HEAD".

| Test | Input | Expected | Reference | HEAD |
|---|---|---|---|---|
| `test_d_one_form_r3` | A = (x*y*z, -z*x, y*z) | independent components [0,1], [0,2], [1,2] equal `-z*x - z`, `-y*x`, `x + z`; dense slot (1,0) is the `neg` of slot (0,1); slot (0,0) is literal 0 | gp `[D(A[2],1) - D(A[1],2), ...]` prints `[-z*x - z, -y*x, x + z]`; Sage `dA = -(x + 1)*z dx∧dy - x*y dx∧dz + (x + z) dy∧dz` | fails |
| `test_dd_is_literal_zero` | A above; f = x^2*y*z^3 + x*y (0-form) | every slot of d(dA) and of d(df) is `Expr::Integer(0)` (structural) | gp `ddA_{xyz} = 0` | fails (garbage trees) |
| `test_d_of_coordinate_form_is_zero` | dx_0 on a 300-dimensional manifold, chart x0..x299 | d(dx_0) has 90000 slots, all `Expr::Integer(0)`; d(d(dx_0)) on n = 30 likewise | d of a constant form | fails (first n slots are trees of n zeros) |
| `test_d_trig_sage` | B = (cos z, sin x, cos y) | [0,1] = cos(x), [0,2] = sin(z), [1,2] = -sin(y) | Sage `dB = cos(x) dx∧dy + sin(z) dx∧dz - sin(y) dy∧dz` | fails |
| `test_d_two_dim` | on R^2: xy dx + x^2 dy; -y^2 dx + x^2 dy | dense `[0, x, -x, 0]`; [0,1] equal to `2*x + 2*y` | gp `deriv(x^2,x) - deriv(x*y,y)` = `x`; `deriv(x^2,x) - deriv(-y^2,y)` = `2*x + 2*y`; Sage `da = (2*x + 2*y) dx∧dy` | fails (`[(y + x), (2 * x), 0, 0]`) |
| `test_d_zero_form_gradient` | f = x*y^2 on R^2 | d f = (y^2, 2xy) | gp `[deriv(x*y^2,x), deriv(x*y^2,y)]` | fails (`[y^2 + 2xy, 0]`, unsimplified) |
| `test_wedge_sage_r3` | A above, B above | [0,1] = x*y*z*sin(x) + x*z*cos(z), [0,2] = x*y*z*cos(y) - y*z*cos(z), [1,2] = -(x*cos(y) + y*sin(x))*z | Sage `A∧B` display | fails |
| `test_wedge_one_two` | a = (2, 1+x, y*z), b = from_independent [y^2, z+x, z^2] | [0,1,2] equal to `-x^2 + (-z - 1)*x + (z*y^3 + (2*z^2 - z))`; b∧a equals a∧b | gp `a[1]*b2[3] - a[2]*b2[2] + a[3]*b2[1]`; Sage `a∧b = (-x^2 + (y^3 - x - 1)*z + 2*z^2 - x) dx∧dy∧dz` | fails |
| `test_wedge_two_dim` | on R^2: (y, x) ∧ (x^2 + y^2, y) | [0,1] equal to `-x^3 - y^2*x + y^2` | gp `a2[1]*bb[2] - a2[2]*bb[1]`; Sage `(-x^3 - (x - 1)*y^2) dx∧dy` | fails |
| `test_wedge_antisymmetry` | dx, dy on R^2 | dx∧dx all literal 0; dx∧dy = `[0, 1, -1, 0]`; dy∧dx = `[0, -1, 1, 0]`; A∧A all literal 0 | definition | fails |
| `test_wedge_powers` | ω = Σ dq_i∧dp_i on R^{2k}, coordinates (q_1..q_k, p_1..p_k), k = 1, 2, 3 | top component of ω^k is 1, -2, -6 | gp `k! * (-1)^(k*(k-1)/2)` | fails |
| `test_leibniz` | A above, P = (y^2, z + x, x*y) | d(A∧P) equal to dA∧P - A∧dP at [0,1,2] | Lee 2013 ch. 14 (graded Leibniz rule) | fails |
| `test_d_budget` | on R^2: ω = (0, s^64), s = 1 + x + ... + x^49 | `Ok` within 5 s (measured 0.05 s; 26.8 s on the original prototype); slot (0,1) is not literal 0 and is structurally `(s^64).differentiate(x).simplify()`; `closedness` is `Ok(Unresolved)`, `is_closed` is `Err(Undecided)` | review B2 (`$S/rev1/r_dcost2.rs`) | 1-form formula of HEAD, wrong |
| `test_d_reserved_constants` | on R^2: (0, (I^2 + 1) x), (0, I x), (0, pi x), (0, (pi - e) x) | the [0,1] components are `(I^2) + 1`, `I`, `pi`, `pi - e` (not literal 0); `closedness` is `Unresolved`, `Unresolved`, `NonZero`, `Unresolved` | 1.6.1 (`$S/rev1/b1.txt`) | fails |
| `test_interior_product_long_sum` | i_X ω on R^20000 for ω = Σ x_{i+1} dx_i, X = Σ x_i ∂_i, in a thread with a 2 MB stack | `Ok` (measured 0.08 s), a 0-form whose component is not literal 0 (its zero test is `Unresolved`: the budget runs out) | 1.4 (`sum_signed`; `$S/rev1/t_int_bal.rs`) | `interior_product` errs |
| `test_d_refuses_unsupported_derivatives` (review 2) | on R^2, ω = f dy with f = `Function("f", [x])`, `bessel_j(0, x)` (a `Function` with 2 arguments), gamma(x), zeta(x), factorial(x) (the `Unary` nodes), `Binary(Mod, x, 3)`, erf(x), sin(f(x^2 + y)), y*gamma(1 + x), `root_sum(t^2 - 2, t*f(x))`, `parse("gamma(x)")`, `parse("f(x)")`; the 0-form gamma(x) | `exterior_derivative` is `Err(NotImplemented(m))` for each, with m containing "the derivative of" and the offending subterm (`f(x)`, `gamma(x)`, `(x % 3)`, ...); `closedness` and `is_closed` of each form return the same Err | 1.4 (the true derivatives are nonzero: f'(x), Γ(x)ψ(x), ζ'(x), 1 off 3Z, 2/sqrt(π) exp(-x^2), ...; `$S/rev2/dguard.txt`) | every d returns Ok with fabricated components (`$S/review2/dfacade.txt`) |
| same test, accepted forms | on R^2: f(y) dy, f(x) dx, gamma(3) dy, gamma(y) dy, `Binary(Mod, y, 3)` dy, sign(x) dy | `Ok`; slot (0,1) is literal 0 and `closedness` is `Zero` (no unsupported node contains the variable it is differentiated in, and ∂_x of a function of y is 0) | definition | Ok, but by the facade |
| same test, accepted forms | abs(x) dy; x^y dy; `root_sum(t^2 - 2, t*x)` dy | `Ok`; slot (0,1) is `sign(x)`, `(x ^ y) * (y / x)` (equal to y x^(y-1)), `root_sum((t ^ 2) - 2, t)`; `closedness` `Unresolved` each (opaque atoms) | by hand: d/dx |x| = sign(x) off 0, d/dx x^y = y x^(y-1) for x > 0, Σ_{c^2 = 2} c = 0 (the zero test cannot see this: `Unresolved` is honest) | fabricated layout |

At HEAD a 0-form has one slot; d builds n slots and puts Σ_j ∂_j f into slot 0 only.

### 1.9 Cost after the fix

Prototype (`$S/measure_new.rs`; rerun with `$S/rev1/measure_new2.rs` on the revised prototype, same
times within noise; the d rows rerun with `$S/rev2/measure_new2.rs` on `proto3.rs`, which adds the
derivative guard, next to `proto2.rs` in the same session, `$S/rev2/measure3.txt`; debug rlibs, the
prototype compiled at opt-level 0, this container, other jobs running):

| Input | HEAD | Prototype |
|---|---|---|
| d(dx_0), n = 1000 | 6.85 s, 253 MB | 0.12-0.18 s, 66 MB |
| dx_0 ∧ dx_0, n = 1000 | 0.29 s, 253 MB | 0.19-0.27 s, 66 MB |
| d(d(dx_0)), n = 100 | 1.39 s, 444 MB | 0.10-0.13 s, 66 MB |
| d of ω_i = x_{i+1}, n = 300 / 600 / 1000 | 0.30 s / 1.54 s / not run | 0.02 s / 0.08 s / 0.20-0.26 s (66 MB at 1000; 0.24 s without the guard, 0.26 s with it) |
| d of ω_{ij} = x_i x_j (2-form, C(100,3) triples), n = 100 | wrong result | 0.68 s, 70 MB (0.58 s without the guard in the same session; 0.43-0.48 s in earlier sessions) |
| d∘d of ω_i = x_{i+1} x_{i+2}, n = 60 | wrong result | 0.03 s, all slots literal 0 |
| (x_0, ..., x_999) ∧ (x_1, ..., x_0): 999000 nonzero symbolic slots | wrong result | 13.1 s, 806 MB (3.1 s without the zero test; 10.2 s and 2.5 s on the original prototype) |
| d of (0, s^64) on R^2, s = 1 + x + ... + x^49 (review B2) | wrong result | 0.05 s (26.8 s on the original prototype) |

What remains is the dense output (64 bytes a slot: `size_of::<Expr>() == 64`) and, for symbolic results,
the trees of rustmath-symbolic (a `Symbol` clone allocates its name). The zero test runs once per nonzero
non-constant independent component. Typical cost: about 20 µs in debug (the wedge row: 499500 calls,
10 s of the 13.1 s, 22 units at most each). Worst case: one call stops at `ZERO_TEST_WORK` units, at most
0.25 s in debug over the 33 measured calls of 1.6.3. So a d or wedge with m nonzero non-constant
independent components spends at most about m × 0.25 s in the zero test, whatever the components are.
It is what turns d∘d and α∧α into literal zeros, so it stays. The derivative guard of d runs once per
derivative taken, before `differentiate`; on the 3-node components of the 2-form row it costs about
0.25 µs a call (485100 calls, 0.1 s). Memory: the input is borrowed
(`components_ref`), the output is allocated once with `try_reserve_exact`, the `±` slots share their
trees, and the zero test holds at most the polynomials its budget allows.

## 2. Item N4 / S8: checked counts and a component guard

### 2.1 The defect at HEAD

`$S/repro_head.rs` (debug build of HEAD):

* `TensorField::new(M, 0, 2).num_components()` with dim M = 2^33: panic "attempt to multiply with
  overflow" (`tensor_field.rs:158`). `TensorField::from_components(M, 0, 2, ..)`: the same panic
  (`:105`).
* `DiffForm::new(M, usize::MAX).exterior_derivative(c)`: panic "attempt to add with overflow"
  (`diff_form.rs:132`). `DiffForm::new(R2, usize::MAX).wedge(&DiffForm::new(R2, 1), c)`: the same
  (`:182`).
* `TensorField::new(R2, 0, 2^32 + 1).num_components()` returns 2: `rank as u32` truncates, in debug and
  release.
* `exterior_derivative` with a chart of 0 coordinates: panic "attempt to calculate the remainder with a
  divisor of zero" (`:151`).
* In release (`overflow-checks` off) the powers wrap silently and the loops run over a wrong size.
* Further unchecked arithmetic of the same kind: `contravariant_rank + covariant_rank`
  (`tensor_field.rs:105`, `:153`), `new_p`/`new_q` sums (`:273-274`), `MultiIndex::to_flat`
  (`stride *= dimension` after the last index overflows even when the index fits, and an index >= n
  aliases another slot: (0, 3) with n = 3 is slot (1, 0); `set_component` at `:187` writes it).

### 2.2 Fix

```rust
// tensor_field.rs

/// n^rank, the number of components of a rank-`rank` tensor on an n-dimensional manifold. Err when
/// it does not fit in `usize` or when its storage (n^rank * size_of::<Expr>() bytes) exceeds
/// `isize::MAX` bytes, the largest allocation Rust allows. Not a policy limit (the REPL has its own).
pub fn component_count(dimension: usize, rank: usize) -> Result<usize>;
//   n == 0: Ok(if rank == 0 { 1 } else { 0 }); n == 1: Ok(1) (any rank, also above u32::MAX);
//   else: u32::try_from(rank) and n.checked_pow(..), then the byte product checked against
//   isize::MAX; failure: Err(ComponentCountOverflow { dimension, rank }).

/// A vector of `count` literal zeros; Err when the allocator refuses (try_reserve_exact).
pub(crate) fn zeroed_components(count: usize) -> Result<Vec<Expr>>;
//   failure: Err(ComponentAllocationFailed { count }).

impl MultiIndex {
    /// Horner evaluation with checked arithmetic. Err(InvalidIndex) when an index is >= dimension
    /// or the value overflows.
    pub fn to_flat(&self, dimension: usize) -> Result<usize>;   // was -> usize
}

impl TensorField {
    pub fn total_rank(&self) -> Result<usize>;       // was -> usize; Err(DegreeOverflow)
    pub fn num_components(&self) -> Result<usize>;   // was -> usize; component_count(n, total_rank()?)
    // from_components: expected = component_count(n, p.checked_add(q)?)?, then the length check
    // component / set_component: indices.len() == total_rank, else Err(InvalidIndex); then to_flat()?
    // set_component on a chart without components: zeroed_components(num_components()?)?
}
```

`errors.rs`, new variants (no `match` on `ManifoldError` exists anywhere in the workspace, checked with
`git grep`):

```rust
#[error("A rank-{rank} tensor on a manifold of dimension {dimension} has too many components to store \
         (more than isize::MAX bytes)")]
ComponentCountOverflow { dimension: usize, rank: usize },
#[error("Could not allocate {count} tensor components")]
ComponentAllocationFailed { count: usize },
#[error("The degree {left} + {right} does not fit in a machine word")]
DegreeOverflow { left: usize, right: usize },
#[error("Undecided: {0}")]
Undecided(String),
```

`diff_form.rs`: every degree sum is `checked_add` (1.4 step 1, 1.5 step 2); every allocation of n^k slots
goes through `component_count` and `zeroed_components`; d checks the chart dimension (1.4 step 4).
`MultiIndex::from_flat` is unchanged (it panics for dimension 0 with rank >= 1, where no index exists; it
is used only in a test).

Contraction (`:237`) and tensor product (`:280`) get the same checks; their bodies are also rewritten
(section 3.2), because the contraction is a facade.

### 2.3 Tests for N4 / S8 (in `tensor_field.rs` unless stated)

| Test | Input | Expected | HEAD |
|---|---|---|---|
| `test_component_count_checked` | `component_count(1 << 33, 2)`, `(2, (1 << 32) + 1)`, `(0, 0)`, `(0, 5)`, `(1, usize::MAX)`, `(1000, 2)`; and at the byte boundary: `max = isize::MAX as usize / size_of::<Expr>()`, `k = max.ilog2()`, then `(2, k)` and `(2, k + 1)` | Err, Err, Ok(1), Ok(0), Ok(1), Ok(10^6); `Ok(1 << k)` and Err. The boundary is derived from `size_of::<Expr>()` in the test, not hard-coded (with today's 64-byte `Expr`: k = 56, and 2^57 slots need 2^63 bytes > `isize::MAX`) | new function |
| `test_num_components_checked` | `TensorField::new(R2, 0, (1 << 32) + 1).num_components()`; `TensorField::new(M_{2^33}, 0, 2).num_components()`; `TensorField::new(R2, usize::MAX, 1).total_rank()` | Err(ComponentCountOverflow), Err(ComponentCountOverflow), Err(DegreeOverflow) | 2; panic; panic |
| `test_from_components_huge_rank` | `TensorField::from_components(M_{2^33}, 0, 2, c, vec![])` | Err(ComponentCountOverflow) | panic |
| `test_zeroed_components_allocation_failure` (`#[cfg(target_pointer_width = "64")]`) | `zeroed_components(max)` with `max` as above (about 2^63 bytes, beyond any 64-bit address space) | Err(ComponentAllocationFailed) | new function |
| `test_multi_index_checked` | `to_flat` of [0, 3] with n = 3; `set_component(c, [0, 3], ..)`; `component(c, [1])` on a rank-2 tensor | Err(InvalidIndex) each | Ok(3) (aliases slot (1, 0)) |
| `test_degree_overflow` (diff_form.rs) | `DiffForm::new(M, usize::MAX).exterior_derivative(c)`; `DiffForm::new(R2, usize::MAX).wedge(&DiffForm::new(R2, 1), c)` | Err(DegreeOverflow { left: usize::MAX, right: 1 }) each | panics |
| `test_d_chart_dimension` (diff_form.rs) | `m` = the manifold of `EuclideanSpace::new(2)`, `c2 = Chart::new("c", 2, vec!["x", "y"])`, ω = `DiffForm::from_components(m, &c2, 1, vec![x*y, x^2])`. Components are keyed by the chart name (`tensor_field.rs:115`, `chart.name()`), so the lookup of step 3 finds them for any chart named `c`. Then `ω.exterior_derivative(&Chart::new("c", 0, Vec::<String>::new()).unwrap())` and `ω.exterior_derivative(&Chart::new("c", 1, vec!["x"]).unwrap())` | Err(InvalidChart(..)) containing "the chart 'c' has dimension 0, but the manifold has dimension 2", and "has dimension 1, but the manifold has dimension 2". With `c2` itself: Ok (the d of `test_d_two_dim`) | panic (`j % 0`, `diff_form.rs:151`); silent reuse of `x` |
| `test_zero_by_dimension` (diff_form.rs) | d of a top form on R^2; dx∧(dx∧dy) | zero forms of degree 3; `vanishes(c) == Ok(Zero)` | degree only |

## 3. Adjacent facades in the rewritten files, and changed callers

### 3.1 Construction and accessors (`diff_form.rs`)

At HEAD `from_components` and `from_tensor` accept any vector: `[5, 1, 7, 0]` as a 2-form on R^2 is
accepted (`$S/repro_head2.rs`). The new d and wedge read only increasing tuples, so such input would give
results that depend on which half was read. New behaviour, for degree >= 2:

* One pass over the n^p slots: decode the tuple; with a repeated index the slot must be decided zero;
  otherwise sort it with its sign and compare with the increasing slot. A structurally equal pair
  (`c == v` or `c == neg(v)`) passes at once; otherwise `zero_test(c - sgn·v)`: `Zero` passes, `NonZero`
  gives `Err(ValidationError(format!("the components are not antisymmetric: {:?} is {} and {:?} is {}", ..)))`,
  `Unresolved` gives `Err(Undecided(format!("cannot verify the antisymmetry of {:?} and {:?}", ..)))`.
* Then the vector is rewritten in place from the independent values with `scatter` (invariant I1;
  no second vector is allocated). `from_tensor` does this in every chart the tensor has components in.
* Degree 0 and 1: no antisymmetry constraint, stored as given (as HEAD).
* Every degree: each component must be finite in the sense of `is_finite_expr` (no `oo`, no NaN or
  infinite `Real`), else `Err(ValidationError(format!("the component {:?} is {}, which is not finite",
  ..)))`. Checked in the same pass. A form's components are functions on the chart domain, and
  `simplify` would turn `0 * oo` into 0 (1.6.1). `is_finite_expr` is memoized on shared nodes (1.6), so
  for degree 0 and 1 the constructors stay linear in the number of distinct nodes, as HEAD's (which
  did not look at the components). For degree >= 2 the antisymmetry pass also runs `==`, `neg` (a
  `simplify`) and the budgeted zero test; the first two walk the tree (6.3). `from_independent_components` checks the same. No
  caller in the workspace passes such components (`git grep` for `oo`, `INFINITY`, `NAN` in
  rustmath-manifolds finds only `to_f64` fallbacks in `tensor_field.rs:328-329`, `vector_field.rs:241-242`).

New constructor and accessor:

```rust
/// A p-form from its independent components: `values[r]` is ω_I for the r-th strictly increasing
/// p-tuple I in lexicographic order. Err(DimensionMismatch) unless values.len() == C(n, p)
/// (a checked binomial); for p > n, values must be empty and the result is DiffForm::new.
pub fn from_independent_components(manifold: Arc<DifferentiableManifold>, chart: &Chart,
    degree: usize, values: Vec<Expr>) -> Result<Self>;

/// The independent components in `chart`: (I, ω_I) for the increasing p-tuples, lexicographic.
pub fn independent_components(&self, chart: &Chart) -> Result<Vec<(Vec<usize>, Expr)>>;
```

### 3.2 Contraction and tensor product (`tensor_field.rs`)

* `contract` (`:216-265`) sums `Expr::from(k as i64)`, the index values: every slot is n(n-1)/2. The
  trace of diag(5, 7) comes out `[1]` (`$S/repro_head.rs`; gp `trace(matdiagonal([5,7]))` = 12). Tracker
  row `PROJECT_TRACKER.md:2521` (🔴). New body (Lee 2013, ch. 12, "Tensors", contraction): rank checks as
  HEAD, `components_ref`, `out_rank = r - 2`, `count = component_count(n, out_rank)?`,
  `zeroed_components`; for each output slot decode its digits, insert k at position `contra_index` and at
  position `p + cov_index` of the input index, sum over k (literal zeros skipped) with `sum_signed`,
  `simplify`. Cost
  n^(r-1)·r. The `% n` of the decode is never reached for n = 0 (count is 0 or the rank is 0).
* `tensor_product` (`:268-298`) writes slot `i * |S| + j`, which is the layout
  [contra(T), cov(T), contra(S), cov(S)], but labels the result (p1+p2, q1+q2), whose layout is
  [contra(T), contra(S), cov(T), cov(S)]. For (0,1) ⊗ (1,0) with t = (3, 4), s = (1, 2) the slots come out
  `[3, 6, 4, 8]`; (t⊗s)^i_j = s^i t_j is `[3, 4, 6, 8]`. New body: output index
  `((u_T * n^{p2} + u_S) * n^{q1} + l_T) * n^{q2} + l_S` with u_T, l_T = t / n^{q1}, t % n^{q1} and
  likewise for S; products of literal zeros skipped, `simplify` otherwise; rank sums checked. Its only
  caller was `wedge`; it stays public API.

### 3.3 Interior product, Lie derivative, Add, Neg, scale, zero decisions (`diff_form.rs`)

At HEAD (`$S/repro_head2.rs`): `interior_product` always returns
`Err(InvalidIndex("Contravariant index 0 out of range for rank 0"))` (it contracts a (0, p) tensor with
itself, `:217`); `lie_derivative` of a 0-form returns the form (`:235-239`; L_X(x^2) with X = ∂_x gives
`x * x`, true 2x) and of a p-form errs through `interior_product`; `(x dx) + (3 dy)` returns `[x, 0]`
(`:289-306` returns the left operand); `-(x dx)` returns `[x, 0]` (`:309-316`). None has a caller outside
`diff_form.rs`.

```rust
/// i_X ω: (i_X ω)_J = Σ_k X^k ω_{kJ} (Lee 2013, ch. 14, "Interior Multiplication"; Sage contracts the
/// first index the same way). Err for a 0-form (HEAD message). p > n + 1: DiffForm::new(m, p - 1);
/// p == n + 1: the zero n-form, dense zeros in `chart` (guarded allocation).
pub fn interior_product(&self, vector: &VectorField, chart: &Chart) -> Result<DiffForm>;
//   X = vector.components(chart)?; for each increasing (p-1)-tuple J: the terms
//   (X^k * ω[flat(k, J)]).simplify() over k ∉ J, combined with sum_signed, normalize(..)?, scatter.
//   A non-finite X^k gives Err(ValidationError) through normalize.

/// L_X ω by Cartan's formula L_X = i_X d + d i_X (Lee 2013, ch. 14, "Lie Derivatives of Differential
/// Forms"). p == 0: i_X(dω), that is X(f). 1 <= p < n: i_X(dω) + d(i_X ω). p == n: d(i_X ω) (dω has
/// degree n + 1). p > n: DiffForm::new(m, p). n == 0: the 0-form 0.
pub fn lie_derivative(&self, vector: &VectorField, chart: &Chart) -> Result<DiffForm>;

/// Componentwise sum in every chart where both summands have components; the sums are computed on the
/// independent components (normalize) and scattered. Err(DifferentManifolds), Err(DimensionMismatch)
/// for different degrees (as HEAD), Err(NoComponentsInChart) when no chart is shared. Degree > n:
/// DiffForm::new. A form from DiffForm::new / DiffForm::zero has no components, so adding it errs
/// (documented).
impl Add for DiffForm { type Output = Result<Self>; }

/// Negates every component in every chart, in place: for each increasing tuple, nv = neg(v), then the
/// slots get nv (even) and the old v (odd), which is equal in value to neg(nv) (invariant I1 is up to
/// value, 1.3).
impl Neg for DiffForm { type Output = Self; }

/// factor * ω in every chart (normalize per independent component, then scatter). Literal 0 gives the
/// zero form in the same charts, literal 1 a clone. Err(ValidationError) for a factor that is not
/// finite (`is_finite_expr`), checked first; otherwise it may fail only on allocation.
pub fn scale(&self, factor: &Expr) -> Result<DiffForm>;

/// Decides whether ω vanishes identically in `chart`. Degree > n: Zero. Else NoComponentsInChart or
/// the zero test over the independent components: NonZero as soon as one is NonZero, else Unresolved
/// if one is Unresolved, else Zero.
pub fn vanishes(&self, chart: &Chart) -> Result<ZeroTest>;

/// vanishes(d ω). The errors of d propagate: Err(NotImplemented) from the derivative guard (1.4),
/// InvalidChart, NoComponentsInChart. A fabricated derivative never reaches the zero test.
pub fn closedness(&self, chart: &Chart) -> Result<ZeroTest>;

/// Ok(true) for Zero, Ok(false) for NonZero, Err(Undecided("d(ω) has components the zero test cannot
/// decide")) for Unresolved; the errors of d propagate as in `closedness`. Signature unchanged.
pub fn is_closed(&self, chart: &Chart) -> Result<bool>;
```

At HEAD `DiffForm::is_closed` (`:267-270`) is the structural `is_zero` of the facade d: the closed form
x1 dx1∧dx2 + dx3∧dx4 on R^4 gives Ok(false) (`$S/repro_head4.rs`). `PartialEq` stays structural.

`is_zero(&self) -> bool` keeps its structural meaning (every stored component is literally 0; a form
without components counts as zero, the convention of `DiffForm::new` and `DiffForm::zero`). Its doc now
says that false does not prove the form nonzero and points to `vanishes`.

Tests (gp values from `$S/refs.gp`; X = (y*z, -x*z, x*y), A as in 1.8; all fail or do not compile on
HEAD):

| Test | Expected | Reference |
|---|---|---|
| `test_interior_product_gp` | i_X A equal to `z^2*x^2 + (z^2 + z)*y^2*x`; i_X dA equal to `[(y^2 - z^2)*x^2 - z^2*x, -y*x^2 + (-z^2 - z)*y*x - z^2*y, -z*x^2 + (-z*y^2 - z^2)*x]`; i_X(dx∧dy) = [x*z, y*z, 0] | gp `i_X A`, `i_X dA` lines |
| `test_lie_derivative_cartan_gp` | L_X A equal to `[(y^2 - z^2)*x^2 + z^2*x + (z^2 + z)*y^2, -y*x^2 + (z^2 + z)*y*x - z^2*y, z*x^2 + ((z + 1)*y^2 - z^2)*x]`; L_X dA independent components equal to `[-2*y*x^2 - 2*y*x + (-z^2 - z)*y, 2*z*x^2 + (-z*y^2 - z^2), y*x + 2*z*y]`; L_X f for f = x^2*y equal to `-z*x^3 + 2*z*y^2*x` | gp `L_X A`, `d(L_X A)`, `X(f)` lines (the gp formula (L_X A)_i = X^j ∂_j A_i + A_j ∂_i X^j is independent of Cartan's) |
| `test_add_neg_scale` | (x dx) + (3 dy) = [x, 3]; -(x dx) = [-x, 0]; degree mismatch Err; no shared chart Err(NoComponentsInChart); scale by 1/2 of dx∧dy gives [0, 1/2, -1/2, 0] | definition |
| `test_from_components_antisymmetry` | `[5, 1, 7, 0]` as a 2-form: Err(ValidationError); `[0, 1, -1, 0]`: Ok; `[0, sin(x)^2, cos(x)^2 - 1, 0]`: Err(Undecided); a 1-form `[5, 7]`: Ok | definition |
| `test_non_finite_components_refused` | 1-forms `[oo, 0]`, `[0 * oo, x]` (not simplified), `[Real(NaN), 0]`, `[sin(oo), 0]` and the 2-form `[0, oo, -oo, 0]`: Err(ValidationError) each; `from_independent_components` with `[Real(inf)]`: Err(ValidationError); `(dx∧dy).scale(&oo)`: Err(ValidationError); `interior_product` of dx with X = (oo, 0): Err(ValidationError) | 1.6.1 (`simplify(0 * oo)` is 0 at HEAD, `$S/rev1/t_simp.rs`) |
| `test_from_independent_components` | R^3, [y^2, z + x, z^2] gives dense `[0, y^2, z+x, -y^2, 0, z^2, -(z+x), -z^2, 0]` (up to `neg`); 2 values: Err(DimensionMismatch) | definition |
| `test_is_closed_three_valued` | A: `is_closed` Ok(false), `closedness` NonZero; dA and d f: Ok(true); x1 dx1∧dx2 + dx3∧dx4 on R^4: Ok(true); f dx on R^2 with f = sin(x)^2 + cos(x)^2 + y*(sin(x)^2 + cos(x)^2 - 1): Err(Undecided), `closedness` Unresolved | d(x1 dx1∧dx2) = dx1∧dx1∧dx2 = 0; the last form is closed but not decidable here |

### 3.4 Callers whose behaviour changes (found with `git grep` over the whole workspace)

Only `rustmath-jupyter` depends on rustmath-manifolds (`git grep -l rustmath-manifolds -- '*/Cargo.toml'`).

| Caller | At HEAD | After MF1 | Change in MF1 |
|---|---|---|---|
| `symplectic.rs:222` `SymplecticForm::is_closed` | its own simplify + structural zero over the facade d. x1 dx1∧dx2 + dx3∧dx4 on R^4 (closed) gives Ok(false) (`$S/repro_head4.rs`) | delegate to `self.form.is_closed(chart)` (three-valued, Err when undecided) | yes; test `test_symplectic_is_closed_nonconstant`: that form gives Ok(true), q2 dq1∧dp1 + dq2∧dp2 (`x2 dx1∧dx3 + dx2∧dx4`) gives Ok(false) |
| `symplectic.rs:69` `SymplecticManifold::new` | `is_closed()?` | an undecided form now errs with Undecided, and a form with a component whose derivative is not implemented (1.4 guard) with NotImplemented, instead of a guess | none |
| `symplectic.rs:162-176` `SymplecticForm::new` | doc says "Verifies that the form is closed and non-degenerate" (`:164`); the body checks only the degree | doc corrected: "Checks only that the form has degree 2. Closedness is checked by `is_closed` (and by `SymplecticManifold::new`); non-degeneracy is not checked (`is_non_degenerate` is a facade, section 5)." Behaviour unchanged | doc only |
| `symplectic.rs:306` `SymplecticForm::volume_form` | ω^n of the facade wedge, and "Divide by n!" is a comment (`:318-321`) | ω^n by the new wedge, then `scale` by 1/n! (an exact `Rational`); n = 0 gives the 0-form 1 | yes; test `test_symplectic_volume_form`: k = 1, 2, 3 top component 1, -1, -1 (gp `(-1)^(k*(k-1)/2)`). Dense storage limits it (section 5): R^8 needs 1.1 GB for ω^4; on R^10 and above it ends in `ComponentAllocationFailed` |
| `symplectic.rs:181-210` `standard_form` | `vec![..; n * n * 4]`, `2 * n` unchecked | `component_count(dim, 2)?` and `zeroed_components`, `n.checked_mul(2)`; the vector passes the antisymmetry check | yes; existing tests unchanged |
| `topology.rs:82` `DeRhamCohomology::is_closed` | structural `is_zero` of the facade d | `form.is_closed(chart)` | yes; test: dA closed, A not closed |
| `contact.rs:148` `is_contact` | `!contact_volume.is_zero()`: dz on R^3 is accepted as a contact form (`$S/repro_head3.rs`) | the contact condition is "α∧(dα)^n vanishes nowhere". It is decided only when the single independent component is a nonzero rational constant (Ok(true)) or `vanishes` gives Zero (Ok(false)); otherwise Err(Undecided("α∧(dα)^n = ... is not constant; whether it vanishes nowhere is not decided")). n = 0 (dim 1): the condition is α itself | yes |
| `contact.rs:118` `ContactForm::standard` | uses `Expr::symbol("y{i}")`, which is not a chart coordinate (the chart is `Chart::standard`, x1..x_{2n+1}); the facade hid it | component 2i is `-Expr::Symbol(chart.coordinate_symbol(2 * i + 1))`. `standard` never calls `is_contact` (it builds the form directly), so HEAD's error is invisible there; but with `y{i}` dα = 0, and the new `test_contact_volume` would get α∧(dα)^k = 0 instead of k! | yes; tests `test_standard_contact_form_3d`, `_5d`, `test_standard_contact_manifold` keep passing; new `test_contact_volume`: standard(k) gives α∧(dα)^k top component k! for k = 1, 2, 3 (`$S/t_contact.rs`; gp `k!`); new `test_dz_is_not_contact`: `ContactForm::new(R3, dz)` is Err |
| `lie_derivative.rs:107` `apply_to_form` | p >= 2 runs the 1-form formula: a 2-form comes back as a 1-form (`$S/repro_head3.rs`) | p >= 2 delegates to `form.lie_derivative(&self.vector_field, chart)` (Cartan, through the guarded d); p = 1 calls `apply_to_one_form` (next row); p = 0 keeps its Err | yes; test `test_apply_to_two_form`: equal to gp `d(L_X A)` |
| `lie_derivative.rs:130-176` `apply_to_one_form` (private; reached from `apply_to_form` and `commutes_with_exterior_derivative`) | (L_X ω)_i = Σ_j X^j ∂_j ω_i + ω_j ∂_i X^j as a left-deep unsimplified sum; `differentiate` of named functions, gamma, ... gives fabricated 0 terms; `chart.coordinate_symbol(j)` falls back to `x` for a short chart (`chart.rs:125-130`) | the same formula, with: the chart dimension check of d after the two lookups (`InvalidChart`); a term whose other factor (X^j, or ω_j) is a literal 0 is skipped before its derivative is taken; `check_derivative` before every other `differentiate` (`Err(NotImplemented)`); each term simplified, combined with `sum_signed` and put through `normalize`. So L_X of a constant form is literal zero (`$S/rev2/proto3.rs`, `lie_one_dense`) | yes; new test `test_lie_one_form_guard` on R^2 (chart `c2 = Chart::new("c", 2, vec!["x", "y"])`): X = ∂_x, ω = f(x) dy: Err(NotImplemented) (true value f'(x) dy); X = (gamma(y), 0), ω = x dx: Err(NotImplemented) (needs ∂_y gamma(y)); X = ∂_x, ω = gamma(y) dy: Ok, both components literal 0; X = (gamma(y), 0), ω = dy: Ok, literal [0, 0]; X = (y, x), ω = x y dx: equal to `[x^2 + y^2, y*x]` (gp, `$S/rev2/refs2.gp`); a 1-coordinate chart named `c`: Err(InvalidChart). Evidence `$S/rev2/dguard.txt`. On HEAD the first two are Ok with fabricated terms and the last is silent |
| `lie_derivative.rs:253` `commutes_with_exterior_derivative` | structural `!=`; gives Ok(false) for A (`$S/repro_head3.rs`) | `(L_X dω + (-(d L_X ω)))?.vanishes(chart)?`: Zero → Ok(true), NonZero → Ok(false), Unresolved → Err(Undecided) | yes; test: A gives Ok(true) |
| `kahler.rs:95`, `:239` | `is_closed` | same signature; unreachable today (`compute_kahler_form` is `unimplemented!`, `:112-119`) | none |
| `integration.rs:390`, `:423` (Stokes) | integrates `components[0]` of the facade dω | integrates `components[0]` of the true dω, which is ω_{0...0} = 0 | none: integration is wrong before and after (section 5) |
| `maps.rs:346-350` `PullBack::apply_to_form` (p > 1) | returns the input form, relabelled (facade; tracker `PROJECT_TRACKER.md:2561`) | `Err(NotImplemented("the pullback of a p-form for p > 1 needs the Jacobian minors (f^*ω)_I = Σ_J (ω_J ∘ f) det(∂f^J/∂x^I)"))`. No caller in the workspace | yes; test `test_pull_push_facades_not_implemented` (a 2-form on R^2 under `SmoothMap::new(r2, r2, "id")`: Err(NotImplemented)) |
| `maps.rs:388-412` `PullBack::apply_to_covariant_tensor` (q >= 2) | computes the Jacobian, then returns the target components unchanged on the source manifold (facade) | `Err(NotImplemented(..))` right after the rank checks, before the Jacobian lookup. Only caller: `apply_to_tensor` (`:431`), which has none | yes; same test, a (0, 2) tensor: Err(NotImplemented) |
| `maps.rs:442-494` `PushForward::apply_to_contravariant_tensor` (p >= 2; review 2) | computes the Jacobian, then returns the source components unchanged on the target manifold (`:485-487`, the same facade) | `Err(NotImplemented("the pushforward of a (p, 0) tensor field for p >= 2 needs (f_* T)^J = Σ_I T^I Π_a ∂f^{j_a}/∂x^{i_a}, composed with the inverse map"))` right after the rank checks. Only caller: `PushForward::apply_to_tensor` (`:502-515`), which has none (`tracker_08_09_tests.rs:111` builds a `PushForward` but is not compiled) | yes; same test, a (2, 0) tensor: Err(NotImplemented) |
| `utilities.rs:117` `exterior_derivative` (HashMap forms) | a correct sparse d of its own | unchanged | none |
| `rustmath-jupyter/src/repl.rs` | see section 4 | `exterior_derivative` refuses the one-name REPL chart on a manifold of dimension >= 2 | one test block (section 4.3) |

Existing manifolds tests that use d or wedge (`diff_form.rs:347`, `:359`; `symplectic.rs:493-499`;
`contact.rs:325-375`; `lie_derivative.rs:355-388`) keep passing with the changes above. Their inputs are
zero or constant forms (constants skip the guard), or the constant 1-form dx on the `apply_to_one_form`
path, whose result is now literal zero and is tested with `simplify().is_zero()`.
`tracker_08_09_tests.rs` is not compiled: no `mod` declares it (`git grep tracker_08_09` finds only the
file and `docs/magma/REVIEW_2026-11-06.md:69`, which lists it as uncompiled). It would not compile as a
child of the crate root either, because it reads the private field `PullBack::map` (`:102`). Its tests
never ran, so they neither pass nor fail; section 5. `test_contact_manifold_creation` (`contact.rs:346-368`) uses `-y` on the chart x1, x2, x3, so
after MF1 its form is not contact (dα = 0) and its `if let Ok(..)` body is skipped; it passes before and
after. The tests at `tensor_field.rs:361` (`to_flat`), `:373`, `:374` and `:396` need `.unwrap()` for
the new `Result` returns, and `lie_derivative.rs:225` (`num_components()?`) must change in the same step
as `tensor_field.rs` (plan step 4) so that the crate builds after each step.

## 4. The REPL arms (item 3) and what J1b should change

### 4.1 What the REPL checks today

| Place | Check |
|---|---|
| `repl.rs:4599-4615` `diff_form` | degree in `0..=DIFF_FORM_DEGREE` (`usize::MAX / 2`, `:10548`) so that p + 1 and p + q of forms built directly cannot overflow in the library |
| `repl.rs:4641-4662` `exterior_derivative` | `p.checked_add(1)` (`:4649`); for p < n, `check_form_components(n, p + 1)` (n^degree <= `FORM_COMPONENTS` = 10^6, `:10553`), with `missing_components_error` first so that a form without components gets the library's error |
| `repl.rs:4680-4699` `wedge` | same manifold, `p.checked_add(q)` (`:4686`), for p + q <= n the same component check |
| `repl.rs:10869-10879` `check_form_components` | `u32::try_from(degree)` and `checked_pow`, then the limit |
| `repl.rs:10891-10902` `missing_components_error` | looks the components up only for forms of at most 10^6 slots |
| Tests | `:12204-12209` (diff_form degree range), `test_form_component_limits` `:12345-12389`, `test_form_without_components` `:12393-12426` |

### 4.2 What MF1 changes in the library that the REPL sees

* Degree sums are checked in the library: `Err(DegreeOverflow)` for p + 1 and p + q.
* The library refuses only what cannot be stored (overflow of n^k or of the byte count, or a refused
  allocation). It has no memory policy: `FORM_COMPONENTS` stays the REPL's.
* The library's lookup order is kept: d checks p >= n first, then looks the components up, then the chart
  dimension, then counts and allocates; wedge checks p + q > n first, then self's and other's components.
  So `missing_components_error` stays accurate.
* New: `exterior_derivative` refuses a chart whose dimension is not dim(M):
  `Invalid chart: the chart 'user_chart' has dimension 1, but the manifold has dimension 3`. A REPL chart
  has one coordinate per name, and a list of names cannot be written (S9, commas inside strings), and
  `chart` does not compare names with dim(M) (S14). So in the REPL, d now works only on 1-dimensional
  manifolds until J1a fixes S9 and S14. HEAD computed a wrong d there anyway.
* Memory: the expression growth is gone. d(d(dx_0)) at n = 100 is 66 MB standalone (HEAD 444 MB); the peak
  is now the dense output (64 bytes a slot) plus symbolic trees for symbolic forms (about 800 MB for
  999000 nonzero slots of x_i x_{j+1} - x_j x_{i+1}, section 1.9).
* Time: each nonzero non-constant independent component costs at most one budgeted zero test (at most
  about 0.25 s in debug, 1.6.3), so d and wedge time is bounded by the component count, whatever the
  expressions are.
* d refuses a component whose derivative rustmath-symbolic does not implement (a named function, `mod`,
  gamma, factorial, zeta or erf of the coordinate): `Err(NotImplemented)`, which the arm maps to
  `ComputationError` like every library error. REPL forms have constant components today, so no REPL test
  reaches it. Once forms with symbolic components can be built in the REPL, its tests should expect this
  error for such components, not a value.
* Forms whose components contain `oo` or a NaN or infinite `Real` are refused by the constructors
  (`ValidationError`). The REPL builds forms only with `coordinate_form` and `diff_form` (constants or
  no components), so no REPL test changes.

### 4.3 The one REPL change made by MF1

`test_form_component_limits` (`repl.rs:12359-12366`) computes `da = exterior_derivative(a, C3)` with
`C3 = chart(E3, "t")`, a one-coordinate chart on R^3. It now errs. Replace the block by:

```rust
        // A REPL chart has one coordinate per name, and a list of names cannot be written yet (S9).
        // On R^3 the one-name chart has the wrong dimension, so rustmath-manifolds refuses d; wedge
        // does not use coordinates. On R^1, d of a 1-form and wedge(u, du) are the degree shortcuts
        // (p >= n, p + q > n). So no REPL test computes a nontrivial d until S9 and S14 are fixed;
        // the rustmath-manifolds tests cover d itself.
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        ctx.eval("C3 = chart(E3, \"t\")").unwrap();
        ctx.eval("a = coordinate_form(E3, C3, 0)").unwrap();
        ctx.eval("b = coordinate_form(E3, C3, 1)").unwrap();
        let e = ctx.eval("exterior_derivative(a, C3)").unwrap_err();
        assert_eq!(e.name, "ComputationError");
        assert!(e.message.contains("has dimension 1, but the manifold has dimension 3"), "{}", e.message);
        ctx.eval("ab = wedge(a, b, C3)").unwrap();
        assert!(ctx.eval("wedge(ab, b, C3)").is_ok());
        ctx.eval("E1 = EuclideanSpace(1)").unwrap();
        ctx.eval("C1 = chart(E1, \"t\")").unwrap();
        ctx.eval("u = coordinate_form(E1, C1, 0)").unwrap();
        ctx.eval("du = exterior_derivative(u, C1)").unwrap();
        assert!(ctx.eval("exterior_derivative(du, C1)").is_ok());
        assert!(ctx.eval("wedge(u, du, C1)").is_ok());
```

No other REPL test changes: every other form test is refused by the REPL limit before the library runs,
or hits `NoComponentsInChart` before the chart check (`exterior_derivative(fs, CS)` at `:12418`), or uses
`p >= n` / `p + q > n` shortcuts (`:12377-12388`).

### 4.4 Recommendations for J1b

1. `DIFF_FORM_DEGREE` (`:10548`, used at `:4608`) can go: any usize degree is safe now, d and wedge
   return `DegreeOverflow`. Drop the bound and its test lines (`:12204-12207`); keep `diff_form(M, -1)`
   refused (a usize conversion).
2. The arms' own overflow checks (`:4649-4650`, `:4686-4687`) are redundant. Either delete them and let
   the library error through (then `:12382-12388` assert the library message "The degree ... does not fit
   in a machine word"), or keep them for the wording. Deleting is simpler.
3. Keep `check_form_components` and `FORM_COMPONENTS` as the REPL memory policy, but compute the count
   with `rustmath_manifolds::tensor_field::component_count` (one source of truth). Its Err means "too
   many components to store": either n^k does not fit in a `usize`, or it fits but its storage exceeds
   `isize::MAX` bytes. So the REPL message should say "too many components to store", or report the
   count n^k when `n.checked_pow(k)` gives one. Re-measure with `ReplContext` (VmHWM) and rewrite the doc comment at
   `:10549-10552`: the "each further d enlarges the expressions" sentence is no longer true, and the
   peak per 10^6 slots is 64 MB per copy plus symbolic trees. The limit may rise if the clones of
   `:649`, `:706` (the REPL part of F2) are removed.
4. Stale line references: `:4651` ("diff_form.rs:139") and `:4689` ("tensor_field.rs:268-292"),
   `:10884` ("tensor_field.rs:162-166") change with MF1; cite the new functions instead of lines.
5. For J1a, which owns them: S14 (`chart` must compare the number of names with dim(M)) and S9 (commas
   inside strings). Once `chart(E3, "x,y,z")` can be written, restore the R^3 version of the block in
   4.3 with a real d.
6. Optional: expose `is_closed` (it can now answer Undecided), `interior_product` and `lie_derivative`
   for forms; the REPL's `interior_product` name is taken by Clifford algebras (`:4241`).

## 5. Defects found on the way and not fixed in this unit

These are outside the items. They should go to HANDOFF (plan step 8 adds them).

* rustmath-symbolic `differentiate` returns 0 for named functions (`differentiate.rs:230-234`, tracker 🛑
  `PROJECT_TRACKER.md:2106`, which still cites the old line `:205`), `Mod` (`:90`), gamma (`:184-188`),
  factorial (`:191`) and zeta (`:204`), and approximates 2/sqrt(π) by 500/443 for erf (`:194-198`).
  MF1's d and the Lie derivative of a 1-form refuse such components (`Err(NotImplemented)`, 1.4).
  Wiring the derivative hooks of `rustmath-symbolic/src/function.rs` (the trait method at `:22-23`,
  `with_derivative` at `:59-64`, the `FunctionRegistry` at `:204-245`) into
  `differentiate`, with digamma for gamma and factorial and an exact 2/sqrt(π) for erf, lifts the
  refusal; `unsupported_derivative` must then admit exactly the kinds that gained a derivative. The other
  callers of `differentiate` in rustmath-manifolds are not guarded by MF1 and inherit the facade:
  `vector_field.rs:169` (`apply_to_scalar`), `:198`, `:202` (`lie_bracket`), `scalar_field.rs:149`
  (`differential`), `maps.rs:126` (`jacobian`), `transition.rs:143`, `utilities.rs:134` (the HashMap d),
  `lie_derivative.rs:234` (`apply_to_tensor`), `symplectic.rs:379`, `:386` (`from_hamiltonian`),
  `symmetries.rs:121-139`, `finsler.rs:160-161`, `:309-311`. Each needs the same guard or the fixed
  `differentiate`. S1 refuses the same facade in its series engine.
* `integration.rs`: the quadrature for n >= 2 is wrong: ∫∫ 1 over the default domain [-10, 10]^2 gives
  20.0 (true 400; `$S/repro_head3.rs`, a form whose `components[0]` is 1). `integrate_form_on_chart`
  (`:309-310`) and `evaluate_at` (`:254`) read `components[0]`, which is ω_{0...0} = 0 for an
  antisymmetric top form (∫ dx∧dy gives 0.0 at HEAD), and `VolumeForm::from_metric` builds `factorial(n)`
  slots instead of n^n (`:201`; Err for n >= 2). Stokes' `verify` therefore integrates 0 for dω after
  MF1 (garbage at HEAD). Tracker row `PROJECT_TRACKER.md:2536` (⏳) should say so. A fix needs the top
  component (flat index of (0, 1, ..., n-1), or `from_independent_components`) and a correct tensor-product
  quadrature.
* Pullback and pushforward (`maps.rs`). MF1 turns the three facades into `Err(NotImplemented)` (3.4):
  the pullback of p-forms with p > 1 (`:346-350`) and of covariant tensors of rank >= 2 (`:388-412`), and
  the pushforward of contravariant tensors of rank >= 2 (`:442-494`). The missing pieces are the
  Jacobian-minor formula (f^*ω)_I = Σ_J (ω_J ∘ f) det(∂f^J/∂x^I) (Lee 2013, ch. 14, "Pullbacks"), the
  Jacobian products for covariant tensors (Lee 2013, ch. 12), and for the pushforward the inverse map
  as well. The 1-form pullback
  (`apply_to_one_form`, `:278-322`) is also wrong for non-constant ω: it multiplies by `ω_j` without
  substituting the map's coordinate expressions (`ω_j ∘ f`), unlike `apply_to_scalar_field`
  (`:259-272`, which calls `substitute_in_expr`). Found by reading the code; no caller.
* Symplectic non-degeneracy (`symplectic.rs:236-268`, tracker 🛑 row `PROJECT_TRACKER.md:2531`):
  `matrix_at` ignores the point and the chart and returns the standard block matrix, so every 2-form is
  "non-degenerate". MF1 edits `symplectic.rs` (3.4) but not this. An honest version can use the pattern of
  `is_contact`: ω^n has one independent component; a nonzero rational constant gives Ok(true), a
  component decided zero gives Ok(false), anything else Err(Undecided). The same `matrix_at` feeds
  `SymplecticForm::evaluate` and `evaluate_on_vectors` (`symplectic.rs:273-301`), so ω(X, Y) is the
  standard form's value for every ω; it needs the components evaluated at the point (`evaluate_at`-style
  substitution) instead. `SymplecticForm::new` checks only the degree; MF1 corrects its doc (3.4), and a
  checking constructor would call `is_closed` and an honest non-degeneracy test.
* Dense storage. A p-form stores n^p slots of 64 bytes, not C(n, p) (Sage's `CompFullyAntiSym`).
  `SymplecticForm::volume_form` on R^{2k} computes ω^j up to j = k: R^8 needs 8^8 slots (1.1 GB) for ω^4;
  on R^10 ω^4 takes 6.4 GB and ω^5 (640 GB) ends in `ComponentAllocationFailed`; on R^12 ω^4 (27.5 GB)
  already does. The REPL caps forms at 10^6 slots (`FORM_COMPONENTS`). Antisymmetric storage would lift
  this; it changes the representation that `maps.rs`, `integration.rs`, `lie_derivative.rs` and the
  REPL read (1.3).
* Two zero tests: S1's crate-private `SeriesConst::zero_status` (rustmath-symbolic) and MF1's
  `expr_zero` (1.6.5). Plan: one public zero test in rustmath-symbolic (S1's constants as coefficients,
  MF1's budgeted rational-function normal form), then `expr_zero.rs` re-exports it.
* rustmath-symbolic's `simplify` folds `0 * a` to `0` for any a, including `oo`, NaN and `1/(y - y)`
  (`$S/rev1/t_simp.rs`). MF1 guards against it for forms (1.6.1); the rule itself is S1's area.
* `lie_derivative.rs:189-247` `apply_to_tensor`: no correction terms (tracker ⏳ row `:2537`).
* Unchecked component counts of the N4 class outside the files MF1 rewrites (not listed in the item):
  `spin.rs:76`, `:78`, `:341`, `:343` (`2_usize.pow((dimension / 2) as u32)` overflows for dimension
  >= 128); `dim * dim` at `almost_complex.rs:160` (`vec![Expr::from(0); dim * dim]`), `finsler.rs:153`,
  `symmetries.rs:110` (`Vec::with_capacity`) and `symplectic.rs:258` (`matrix_at`); `dim * dim * dim` at
  `finsler.rs:300`. Each should use `checked_mul` or `component_count`.
* `modules.rs:573` and `traits.rs:124` `total_rank`: unchecked rank sums of module types (no allocation
  follows).
* `DiffForm::new` / `DiffForm::zero` mean both "zero" (`is_zero` is true) and "no components in this
  chart" (d errs). A sparse antisymmetric storage with an explicit zero would remove the ambiguity.
* `tracker_08_09_tests.rs` is not compiled (no `mod` declares it; already listed in
  `docs/magma/REVIEW_2026-11-06.md:69`). As a child of the crate root it would not compile, because it
  reads the private field `PullBack::map` (`:102`, also `PushForward::map` at `:114`). Its
  `test_lie_derivative_one_form` (`:186-212`) asserts that L_∂x dx is structurally zero; that fails at
  HEAD (unsimplified terms) and holds after MF1 (`apply_to_one_form` normalizes, 3.4). Wiring the file
  in needs accessors for the map, or the tests moved into `maps.rs`.

## 6. Plan

### 6.1 Files, in implementation order

1. `rustmath-manifolds/Cargo.toml`: add `rustmath-polynomials`.
2. `rustmath-manifolds/src/errors.rs`: the four variants of 2.2.
3. `rustmath-manifolds/src/expr_zero.rs` (new): header citing MAGMA Chapter_24 §24.4.2 and §24.4.7,
   Chapter_41, Buchberger-Loos 1982, Richardson 1968, Lindemann 1882 and Hermite 1873, and
   `rustmath-symbolic/src/parser.rs:216-237` for the reserved names; the module doc states the semantics,
   the reserved names, the exact value of a `Real` and the precondition (1.6.1). Contents: `ZeroTest`,
   `reserved`, `ZERO_TEST_WORK`, `zero_test`, `zero_test_with_budget`, `is_finite_expr`, `normalize`, and
   the private prescan, conversion (`MultivariatePolynomial<Integer>` with a cached coefficient-size bound)
   and limits of 1.6.2 and 1.6.3, following `$S/rev2/proto3.rs` (without its `ZT_*` environment
   overrides and counters). `$S/rev1/proto2.rs` lacks the negation charge of 1.6.3 (review 2): every
   negation of a polynomial, in the `Unary(Neg)` arm, for the right operand of a subtraction and for a
   negated first term, goes through one charged helper (`pneg` in `proto3.rs`). Also `sum_signed` (1.4),
   the memoized scans `is_finite_expr` and `unsupported_derivative` / `check_derivative` (1.4; the
   header also cites `rustmath-symbolic/src/differentiate.rs` and `PROJECT_TRACKER.md:2106` for the
   node kinds it refuses). Module tests of 1.6.6 (`test_zero_test_cases`, `test_zero_test_budget`,
   `test_normalize_and_finiteness`, `test_scans_on_shared_subtrees`). `lib.rs`: `pub mod expr_zero;` and
   the re-export.
4. `rustmath-manifolds/src/tensor_field.rs`: header gains the citations (Lee 2013 ch. 12); `component_count`,
   `zeroed_components`, checked `to_flat`, `total_rank`, `num_components`, `from_components`,
   `component`, `set_component`; the crate-private helpers of 1.7; `contract` and `tensor_product`
   rewritten (3.2); tests of 2.3 and 3.2 (`test_contract_trace` with gp `trace(matdiagonal([5,7]))` = 12
   and the (1,2) tensor δ^i_j a_k: contract(0,0) = 3a_k, contract(0,1) = a_j;
   `test_tensor_product_index_order` = [3, 4, 6, 8]). In the same step, so that the crate builds:
   `.unwrap()` in the tests at `tensor_field.rs:361`, `:373`, `:374`, `:396`, and
   `lie_derivative.rs:225` becomes `tensor.num_components()?` (with `zeroed_components` for the vector
   at `:226`). `contract` uses `expr_zero::sum_signed` (step 3).
5. `rustmath-manifolds/src/diff_form.rs`: header citing MAGMA Chapter_82 §82.3 (exterior algebras),
   Chapter_42 §42.15 (`Differential(a)`), Chapter_111 (derivations), Lee 2013 ch. 14, Spivak, *Calculus on
   Manifolds* (1965) ch. 4, and the Sage module as the API reference; the convention and invariants I1, I2
   in the module doc; the combinatorial helpers (`first_comb`/`next_comb`, Heap's permutations with sign,
   `scatter`, `neg`); construction with the antisymmetry and finiteness checks (3.1), d
   (1.4, with `check_derivative` before every `differentiate` and the precondition in its doc), wedge
   (1.5), 3.3; tests of 1.8 (including `test_d_refuses_unsupported_derivatives`), 2.3, 3.3. Note in the module doc that MAGMA has no
   smooth-manifold chapter and that the handbook has no examples to port.
6. Callers (3.4): `symplectic.rs` (also the doc of `SymplecticForm::new`), `topology.rs`, `contact.rs`,
   `lie_derivative.rs` (`apply_to_form`, `apply_to_one_form` with the guard, the chart check and
   `sum_signed`, `commutes_with_exterior_derivative`; `test_lie_one_form_guard`), `maps.rs` (the three
   `Err(NotImplemented)`: two pullbacks and the pushforward; `test_pull_push_facades_not_implemented`),
   with their tests.
7. `rustmath-jupyter/src/repl.rs`: the test block of 4.3 only.
8. Docs: PROJECT_TRACKER.md rows `:2520` (d), `:2521` (contraction), `:2522` (wedge) to 🟢 with the new
   locations; `:2537` (Lie derivative: `apply_to_form` correct, `apply_to_tensor` still without
   correction terms, stays ⏳); `:2543` (de Rham: `is_closed` is three-valued now); `:2536` (integration,
   section 5); remove the two "Absent entirely" bullets `:2555`, `:2556`; the bullet `:2561` (pullback)
   says that the p-form and covariant-tensor pullback facades and the pushforward facade for rank >= 2
   now return `Err(NotImplemented)`, and that the 1-form pullback lacks ω∘f; the tally line `:2516`. Row `:2531` (symplectic non-degeneracy) stays 🛑.
   The tracker row `:2106` (rustmath-symbolic, derivative of a named function) gains "MF1's d refuses
   such components (rustmath-manifolds `expr_zero::unsupported_derivative`)" and its stale line `:205`
   becomes `:230-234`; the list of other facade kinds (Mod, gamma, factorial, zeta, erf) is added to it.
   HANDOFF: remove the library part of F2 and N4 / S8, keep the REPL part of F2 for J1b, add section 4.4
   and every bullet of section 5 (the `differentiate` facades and the unguarded callers, integration,
   pullback and pushforward, symplectic non-degeneracy and evaluation and the `SymplecticForm::new`
   checks, `apply_to_tensor`, the unchecked counts of `spin.rs`, `almost_complex.rs`, `finsler.rs`,
   `symmetries.rs`, `symplectic.rs:258`, module rank sums, the `DiffForm::new` ambiguity, the
   uncompiled `tracker_08_09_tests.rs`, the dense-storage limit, the two zero tests, `simplify`'s
   `0 * a` rule). No docs/magma rows change (MAGMA has no intrinsics for smooth forms).

### 6.2 Acceptance

```bash
cargo test -p rustmath-manifolds --no-fail-fast        # HEAD baseline: 258 passed, 11 ignored; 31 doc tests
cargo test -p rustmath-jupyter --no-fail-fast          # the only downstream crate
cargo check --workspace --tests
cargo clippy -p rustmath-manifolds
gp -q $S/refs.gp                                       # the reference values quoted in the tests
gp -q $S/rev2/refs2.gp < /dev/null                     # the Revision 2 values (L_X(xy dx), the erf coefficient)
cd $S/rev2 && ../build.sh t_b1.rs t_b1 && ./t_b1       # expected output: $S/rev1/b1.txt (also t_b2 <family> <a> <b>: sweep3.txt)
cd $S/rev2 && ../build.sh t_dguard.rs t_dguard && ./t_dguard   # expected output: dguard.txt
```

Every test of 1.8, 2.3 and 3.3 named "fails" must fail or not compile on HEAD and pass after (apply only
the test hunks to HEAD once to check). Most use new APIs and cannot compile on HEAD; for those the
prototype programs (`$S/repro_head*.rs`, `$S/rev1/*.rs`, `$S/review2/r_dfacade.rs`) are the HEAD
evidence. The tests with wall-clock bounds (`test_zero_test_budget`, `test_d_budget`,
`test_scans_on_shared_subtrees`) allow at least 25 times the measured debug time.

### 6.3 Risks

* `DiffForm::from_components` and `from_tensor` now reject non-antisymmetric input and refuse input whose
  antisymmetry cannot be verified (Undecided). No caller in the workspace passes such input (the only
  degree >= 2 caller is `standard_form`, constant ±1).
* Public signatures change: `MultiIndex::to_flat`, `TensorField::total_rank` and `num_components` now
  return `Result` (callers: `tensor_field.rs` itself, `lie_derivative.rs:225`, three tests). `Neg`
  keeps its signature; `scale` is new.
* The zero test treats free symbols as indeterminates: a form a dx∧dy with a parameter a is NonZero, as in
  Sage. Its budget and limits make large inputs Unresolved, never a wrong decision. The reserved names
  follow the parser: a chart coordinate named `I`, `pi` or `e` gets the conservative treatment of 1.6.1
  (sound, fewer decisions); one named `oo` cannot be used with forms.
* Forms with components that contain `oo` or a NaN or infinite `Real` are now refused (3.1). No caller in
  the workspace builds such forms.
* Cost of the zero test in `normalize`: typical about 20 µs a call in debug, about 3 times the time of the
  rest of a wedge of generic symbolic 1-forms (13.1 s against 3.1 s for 999000 slots, 1.9). Worst case
  about 0.25 s a call (the work budget, 1.6.3; the slowest units are inserts into large polynomials of
  two-variable monomials). It runs once per nonzero non-constant independent component, so a form with
  many components whose zero tests all exhaust the budget costs up to 0.25 s each. Lowering
  `ZERO_TEST_WORK` would trade decisions for time. The unit costs are measured on the prototype; an
  implementation that adds an uncharged pass over the terms of a polynomial breaks the bound, as the
  uncharged negation did (review 2). Every such pass must be in the table of 1.6.3.
* d refuses components whose derivative rustmath-symbolic does not implement (1.4): any `Function` of
  the coordinate other than `root_sum`, and `mod`, gamma, factorial, zeta, erf of it. Such forms gave a
  fabricated d at HEAD; now they give `Err(NotImplemented)`, and so do `closedness`, `is_closed`, the Lie
  derivative, `is_contact`, `SymplecticManifold::new` and `commutes_with_exterior_derivative` on them. No
  caller in the workspace builds such forms (`git grep` for `Expr::Function`, gamma, zeta, erf, factorial
  and `Mod` in rustmath-manifolds finds only `catalog.rs` metrics, integration and transition helpers,
  none of which reaches d). A hand-built `Expr::Function("sin", ..)` is refused too: the parser's
  `sin(x)` is `Unary(Sin)` and works.
* Tree walks on shared subtrees. `is_finite_expr` and `unsupported_derivative` are linear in the
  number of distinct nodes (1.4, 1.6). The other walks are linear in the tree size, which can be
  exponential in the DAG size (a component built by repeated `e.clone() + e.clone()`): `is_constant`,
  `differentiate` and `simplify` in d (as at HEAD), and in the constructors for degree >= 2 the
  structural `==` (`Expr` derives `PartialEq` but not `Eq`, so `Arc` comparison has no pointer
  shortcut) and the `simplify` of `neg` (3.1). On such a component `from_components` of degree >= 2 can
  take as long as `simplify` of it, where HEAD returned at once; degree 0 and 1 stay as fast as HEAD.
  The zero test itself stops after 100000 tree nodes (`Unresolved`). A memoized `simplify` and an
  `Arc::ptr_eq` shortcut in the structural comparison would remove this; both belong to
  rustmath-symbolic.
* The REPL can no longer take d on manifolds of dimension >= 2 until J1a fixes S9 and S14 (4.2). This is
  the honest answer: the REPL chart does not match the manifold.
* Recursion: `simplify` and `differentiate` recurse over the input trees (as HEAD; S1 makes `simplify`
  iterative along the left spine). The sums built by MF1 are balanced (`sum_signed`), the zero test's
  prescan is iterative, its conversion flattens chains and is capped at 200 nested levels, and trees deeper
  than 4096 are `Unresolved`. A user component that is itself a very deep tree can still overflow the
  stack in `simplify` or `Drop`, as at HEAD.

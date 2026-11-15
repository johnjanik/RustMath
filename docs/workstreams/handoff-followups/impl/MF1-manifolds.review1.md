# MF1-manifolds: implementation review 1 (adversarial)

Reviewed: the uncommitted diff on HEAD `0c2f368` (code equal to `36c2170`), against
`fu2/design/MF1-manifolds.md` (Revision 3, approved by review 4) and the report
`fu2/impl/MF1-manifolds.md`. Every uncommitted change was read: `expr_zero.rs` (new, in full),
`diff_form.rs` (in full), the diffs of `tensor_field.rs`, `lie_derivative.rs`, `maps.rs`,
`symplectic.rs`, `contact.rs`, `topology.rs`, `catalog.rs`, `errors.rs`, `lib.rs`, `Cargo.toml`,
`Cargo.lock` and the two comments in `rustmath-jupyter/src/repl.rs`. No file was edited.
Scratch: `$R = fu2/impl/MF1-manifolds.rev1/` (sources in `src/`, outputs as `.txt`; binaries
deleted).

Verdict: **approved**. No blocker. Three minor issues follow; none changes a value the code
computes.

## Acceptance commands (run by me)

| Command | Result |
|---|---|
| `cargo test -p rustmath-manifolds --no-fail-fast` | 308 passed, 0 failed, 11 ignored; 33 doc tests passed (baseline 259 / 11 / 31) |
| `cargo test -p rustmath-jupyter --no-fail-fast` | lib 109 passed, 1 ignored; integration 4 passed (unchanged) |
| `cargo check --workspace --tests` | ok |
| `cargo clippy -p rustmath-manifolds --no-deps` | ok; no warning in `expr_zero.rs`, `diff_form.rs`, `lie_derivative.rs`, nor on a changed line elsewhere |
| `cargo clippy -p rustmath-manifolds` | stops in rustmath-combinatorics on 5 deny lints (`binary_words.rs`, `debruijn.rs`, `path_tableaux.rs`, `super_tableaux.rs`, `word.rs`), outside this unit, as the report says |
| `gp -q refs.gp`, `rev2/refs2.gp`, `rev3/refs4.gp` (`< /dev/null`) | every value quoted in the tests; `refs4.out` identical to the design's |
| design harnesses rebuilt on fresh debug rlibs (`$R/rlibs.txt`) | `t_rootsum3` prints `rev3/rootsum3.txt` (timings aside), `t_dguard` prints `rev2/dguard.txt` exactly, `t_b2 negp 80 197` Unresolved after 100000 units in 0.143 s |

## What I checked, with evidence

* **Every design item is implemented.** 1.4 (d, steps 1 to 7 in order: overflow, p >= n,
  lookup, chart check, count, loop with `check_derivative` before each `differentiate`), 1.5
  (wedge, shuffle sign `inv += p - x` re-derived), 1.6 (zero test, prescan, conversion, charges
  of the 1.6.3 table incl. negation, limits, answer rule), 2.2 (checked counts, fallible
  allocation, checked `to_flat`, `Result` returns), 3.1 to 3.4 (constructors, contraction,
  tensor product layout, interior product, Cartan Lie derivative, Add/Neg/scale,
  vanishes/closedness/is_closed, all callers), 4.3 (the two REPL comments). Public items
  diffed against HEAD: only the planned additions and the three `Result` signatures.
* **Zero test soundness, fuzzed** (`$R/src/fuzz.rs`): 4000 random rational expressions in x, y
  (`+ - * /`, integer powers -1..2, `Neg`), half compared with an equal-valued rewrite, half with
  an unrelated expression; every answer checked by exact `eval_rational` at 6 random rational
  points. Result: 1572 Zero, 2042 NonZero, 386 Unresolved, **0 contradictions** (17366
  evaluations). `MultivariatePolynomial::add_term` removes zero coefficients, so `is_zero` and
  `variables()` (atoms that cancelled) are reliable.
* **Cost on the implementation, not the prototype** (`$R/src/t_impl.rs`, `sweep_impl.txt`): the
  33 calls of `rev3/sweep4.txt` through `rustmath_manifolds::zero_test` give the same 33 answers;
  slowest 0.161 s (`negp 100 197`). The 0.25 s worst case of 1.6.3 holds. The 1.9 rows
  (`$R/measure_rev1.txt`, report's harness rebuilt): d(dx_0) n = 1000 0.24 s; d(d(dx_0))
  n = 100 0.19 s; d of x_{i+1} dx_i n = 1000 0.18 s / 66 MB; d of the 2-form x_i x_j on R^100
  0.45 s / 69 MB. All within the design.
* **Probes** (`$R/src/probe.rs`, `probe2.rs`, outputs `probe.txt`, `probe2.txt`): L_X dx with
  X = gamma(x) ∂_x is refused (needs ∂_x gamma); a 2-form [5] on R^1 is refused (repeated
  index); L_X and d on a 0-dimensional manifold give the 0-form 0 and the empty 1-form;
  `[0, sin x, -1*sin x, 0]` is accepted (zero test, not structure); `simplify` never folds
  `Real` arithmetic, so a float overflow cannot put `inf` into a component; `(1/0)*x dy`
  gives an Unresolved (honest) d; `(x - x)^0` is NonZero (0^0 = 1, the convention `simplify`
  uses too).
* **Memo scans.** Re-derived the rule of `unsupported_derivative`: a node seen below an
  ancestor finds every occurrence of `var`, so skipping a later visit without one loses nothing;
  a strong count of 1 means a single parent, so the walk is linear in distinct nodes.
  `test_scans_on_shared_subtrees` covers both DAG shapes.
* **No undecided result returned as decided.** `is_closed`, `commutes_with_exterior_derivative`
  and `is_contact` map Unresolved to `Err(Undecided)`; `is_contact` claims true only for a
  nonzero rational constant; antisymmetry refusals are by `NonZero`, undecidable input is
  `Undecided`; every exhausted budget or limit is Unresolved.
* **Bounded.** Every new loop is over C(n, k) tuples times at most k! permutations or n
  terms, all at most n^k slots that `component_count` admitted and `zeroed_components`
  allocated fallibly. `binomial` cannot overflow u128 (it runs after `component_count`).
  `from_independent_components` checks the count before it allocates.
* **Callers.** `git grep` over the workspace for `from_components`, `from_tensor`,
  `num_components`, `total_rank`, `to_flat`, `apply_to_tensor`, `is_closed`: only the callers of
  design 3.4 and 6.3 (the other `num_components` / `total_rank` hits are other types).
  `VectorField::from_components` checks the length, so the interior product cannot index out of
  range. CRLF endings of `contact.rs` and `symplectic.rs` are kept on every line.
* **Assertions.** The only removed assertion lines are the three `tensor_field.rs` lines that
  gained `.unwrap()`. All five old `diff_form.rs` tests and the old `lie_derivative.rs`,
  `symplectic.rs`, `contact.rs` tests are kept unchanged.
* **HEAD-fail.** The new tests reference `crate::expr_zero` or other new APIs, so the test
  target does not compile on HEAD. Of the tests that would compile alone, `test_de_rham_is_closed`
  (`is_closed(dA)` is the structural zero of the facade d: false), `test_dz_is_not_contact`,
  `test_apply_to_tensor_not_implemented`, `test_lie_one_form_guard`, `test_contract_trace`
  ([1], not [12]) and `test_tensor_product_index_order` ([3, 6, 4, 8]) fail on HEAD's behaviour.
* **Rules.** No `unsafe`, no em-dash in added prose. Headers cite MAGMA Ch. 24 §24.4.2, §24.4.7,
  Ch. 41, Ch. 42 §42.15, Ch. 82 §82.3, Ch. 111 (all exist in `rustmath-lava/docs/`) and the
  papers. No `rustmath-misc/output.txt` or `.tmp_dir_*` change. Every `match` on an `Expr` is on
  a reference (S1's E0509 rule).
* **Review 4 minors.** Folded: 3 (repeated coordinates, tested), 4 (`_t` or `_t0`), 5 (the (1, 0)
  tensor has components). Skipped 1, 2, 6: slips in the design text, no code depends on them.
  I agree.
* **Handoffs to MF1.** R1 (ring predicates, nothing to do) and J1b (the 4.4 REPL items, left
  for HANDOFF as the design says). The report's Handoffs list all of them.

## Minor issues

1. **Two negation items of `test_zero_test_budget` pass for the wrong reason**
   (`expr_zero.rs:955-972`). P is built from two *balanced* sums (`sum_of`, depth 8), so
   197 nested `Neg` around P, and 197 right-nested subtractions, put the inner sub-sums of P at
   conversion depth above `MAX_NESTING = 200` (the flattening covers only the left spine). `convert` returns `Budget` before any polynomial is
   built: Unresolved in 0.000 s, even with a budget of 10^10. So these items do not reach the
   negation charge, and they would pass on `proto2.rs` (no charge) too; design 1.6.6 says the
   item must fail there by its answer. Measured (`$R/src/probe3.rs`, `probe3.txt`): balanced
   k = 197: Unresolved 0.000 s with either budget (also right-nested); left-deep k = 197:
   Unresolved in 0.12 to 0.15 s with the default budget (the charge), NonZero after 11.2 to
   11.6 s with budget 10^10 (right-nested: NonZero after 11.6 s); balanced k = 150: Unresolved
   in 0.12 s (the charge), NonZero after 8.5 to 8.9 s with budget 10^10 (two runs each). The code is right;
   the test is weaker than designed.
   Required change: build the two factors of P left-deep (100 terms each, as the design's
   evidence `neg3.txt` and `t_b2 negp` did; 100 terms is within the "few hundred" rule of 6.1),
   so that 197 levels stay at depth 199. Alternatively keep `sum_of` and use 150 levels.
   Apply the same to the right-nested item.
2. **`Symbol("π")` is an indeterminate, but rustmath-symbolic uses it for π.**
   `reserved::NAMED_CONSTANTS` is `["pi", "e"]`. The parser maps `π` to `pi`, but
   `solve.rs:911`, `:919`, `:938`, `:952`, `:977`, `:985` build `Expr::symbol("π")` for π. So
   `zero_test(π - e)` is NonZero while `zero_test(pi - e)` is Unresolved (`probe.txt` item 2).
   For an integer polynomial P, P(π, e) ≠ 0 is unproven (the algebraic independence of π and e
   is open). Under the module's stated semantics `π` is just a free symbol, so no documented
   answer is wrong, and only a solve result used as a form component reaches it. Required
   change (cheap): add `"π"` to `NAMED_CONSTANTS` (counted as its own named constant, so `pi`
   with `π` is Unresolved, which is sound), say so in the module doc, and add a row to
   `test_zero_test_cases` (`π - e` Unresolved, `π*x` NonZero).
3. **Doc wording.** `diff_form.rs:49-50` says "Every result component is simplified and put
   through the zero test". `Add` copies a summand's component unchanged when the other one is
   a literal 0 (`:945-948`), and `Neg` only simplifies. Say "every computed component", or
   name the exceptions.

## Notes (no change required)

* `repl.rs:11102-11107` (`FORM_COMPONENTS` doc) still says "each further d enlarges the
  expressions; that growth is not bounded here". After MF1 that is false. Design 4.4 item 3
  hands the doc rewrite and the re-measurement to HANDOFF, and the report's Handoffs carry
  it; the orchestrator should keep that entry.
* `check_coordinate_chart` reports "has dimension d, but the manifold has dimension d" in the
  impossible case that `Chart` has a coordinate count different from its dimension
  (`Chart::new` checks the count). Harmless.
* The report's deviations 1 to 10 are each justified; I checked 4 (length check in
  `apply_to_one_form`), 8 (lookup before the n = 0 shortcut) and 9 (`catalog.rs`, identical
  behaviour).

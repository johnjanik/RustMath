# MF1-manifolds: review 4 (independent, adversarial)

Design reviewed: `fu2/design/MF1-manifolds.md` (1476 lines, with Revisions 1 to 3). HEAD `92906d4`. It
differs from `36c2170` only in `docs/` (`git diff 36c2170 HEAD --name-only`), so the code facts of the
design hold at HEAD. Tree clean. Scratch: `$R = .../fu2/design/MF1-manifolds/review4/`. Programs were
built with a copy of `rev3/build.sh` against the `36c2170` debug rlibs. The binaries were deleted
afterwards; the outputs are kept as `.txt`.

Verdict: **approved**. The blocker of review 3 is fixed, in the prototype and in the design text, and
every minor issue of review 3 is handled. I found no blocker. Six minor issues follow; none changes a
value the design computes on a valid chart.

## Blocker of review 3 (B1): resolved

* `$R/proto4.rs` is `rev2/proto3.rs` plus one match arm (`diff` checked): a two-argument `root_sum`
  pushes f with `anc.or(Some(node))` and g with `anc`. So a `var` anywhere in f returns the outermost
  unsupported ancestor, or the `root_sum` itself. No `contains_symbol` walk is added.
* Reproduced: `t_rootsum3` prints `rev3/rootsum3.txt` exactly (`rootsum3_mine.txt`, apart from the
  timing). `t_dguard` prints `rev2/dguard.txt` exactly (`dguard_mine.txt`). Review 3's own probes
  (`r_rootsum`, `r_misc`) now print Err for every `root_sum` case (`rootsum_mine.txt`,
  `misc_mine.txt`). `t_dag` gives the answers of `dag.txt`; the 2^40-copy `root_sum` DAG takes 75 µs.
* My own probes (`$R/r4_probe.rs`, `r4_probe.txt`), all as the design says:
  * Memo soundness. The shared f = t^2 - x is pushed first with no ancestor after the memo started
    (a 100-level chain visited first), and only later as f of `root_sum(f, t^2)`. The key
    `(addr, true)` is not yet in the set, so f is pushed again and the `root_sum` is returned. The
    reverse order and `root_sum(N, N)` with one `Arc` are refused too. I re-derived the memo rule:
    a visit below an ancestor finds every occurrence of `var`, which is a superset of what a visit
    with no ancestor finds. So skipping `(addr, false)` after `(addr, true)` loses nothing, and the
    result is `Some` exactly when an offending subterm exists.
  * One- and three-argument `root_sum` with x: refused (generic `Function`). `root_sum(t^2 - x, t*x)`,
    `x^root_sum(..)`, `-root_sum(..)`, and `root_sum(t^2 - abs(x), t^2)`: refused.
  * Lie derivative with the `root_sum` in X, `X = (root_sum(t - y, t), 0)`, ω = x dx: refused. It
    needs ∂_y of the `root_sum`; the true value is (L_X ω)_y = x.
* The precondition of 1.4 is right as restated. When f has no `var`, the roots of f do not move with
  `var`, and termwise differentiation of the finite sum is exact. This holds whichever symbol of f
  is bound. Refusing a `var` that is the bound symbol (`root_sum(x^2 - 2, x*y)`) only loses an
  answer.
* `differentiate.rs` is unchanged since `c79a582`. Its matches on `UnaryOp` and `BinaryOp` are
  exhaustive, so the table of 1.4 lists every kind where it is not the derivative.
* gp: `gp -q rev3/refs4.gp < /dev/null` prints `refs4.out` exactly (`refs4_mine.out`). `refs.gp` and
  `rev2/refs2.gp` print every value quoted in 1.8 and 3.3. 2Γ(2)ψ(2) = 2(1 - γ) = 0.8456 checks.
* Section 5, the HANDOFF bullet and the plan-step-8 text for tracker row `:2117` name the rule.

## Minor issues of review 3: resolved

1. `apply_to_tensor` returns `Err(NotImplemented)` after the rank checks (3.4, plan step 4). It is
   indeed a facade at HEAD (`r4_head.txt`): for T = x dx⊗dx and X = (y, x) it gives `[y, 0, 0, 0]`.
   By hand, (L_X T)_ij = X^k ∂_k T_ij + T_kj ∂_i X^k + T_ik ∂_j X^k = `[y, x, x, 0]`. Its only
   `num_components` use is `lie_derivative.rs:225`. `git grep apply_to_tensor` shows no caller.
2. Tracker row "Pullback / pushforward" (`:2539`) is in plan step 8. The tracker lines quoted at
   `36c2170` (`:2117`, `:2527`, `:2531-2533`, `:2539`, `:2542`, `:2547`, `:2548`, `:2554`,
   `:2566`, `:2567`, `:2572`) hold at HEAD.
3. `check_coordinate_chart` (1.4 step 4) checks the dimension, then the coordinates against
   `DifferentiableManifold::chart(name)` (`differentiable.rs:125`, which returns `Result<&Chart>`).
   An unregistered chart is trusted, and the docs state the precondition. Every in-crate caller
   passes `default_chart()`, the registered chart itself (`examples.rs:98-115`, `contact.rs:107`,
   `symplectic.rs:188`), or its caller's chart (`topology.rs:82`). REPL charts are `user_chart` and
   never registered (`repl.rs:4801`). See minor 2 and 3 below.
4. `test_zero_test_budget` builds its 30000-term sum with `sum_signed`. `drop3.txt` shows why.
5. R1 hand-off: `modules.rs:197-225` and test `scalar_field_algebra_ring_predicates` (`:812`) exist.
   Nothing is left for MF1.
6. 3.3 wording restated correctly (`vector_field.rs:68-83` checks the length only).
7. Moot. J1a's `check_chart` (`repl.rs:11455-11462`) refuses a wrong-dimension chart first. The
   R^3 block of `test_form_component_limits` (`:14791-14834`) is back. I traced each REPL test of
   4.3 against the new d and wedge order. Each passes, and the two comments of 4.3 are the only
   stale text (`:4903-4904`, `:12326-12327`).

## Re-checks at HEAD

* Baseline: `cargo test -p rustmath-manifolds --no-fail-fast` gives 259 passed, 0 failed,
  11 ignored, and 31 doc tests, as the design says.
* The moved lines of Revision 3 hold:
  * `MultivariatePolynomial::neg` is `multivariate.rs:891-900`, and `add_term` with the
    `HAS_UNBOUND_CONSTANTS` branch is `:385-404`.
  * `Integer::to_i64 -> Option<i64>` is `integer.rs:100`, and `impl Ring for Integer` is `:1524`.
  * `Rational::from_f64` is `rational.rs:176`, and `risch.rs:33-38` and `:1151-1158` match.
  * The REPL lines of 4.1 hold: `:213`, `:4793-4807`, `:4853-4877`, `:4880-4920`, `:11098`,
    `:11101`, `:11108`, `:12314`, `:12334`, and the tests.
* Cost (`measure4.txt`): d(dx_0), n = 1000, 0.11 s / 66 MB; d(d(dx_0)), n = 100, 0.08 s / 66 MB;
  the 2-form x_i x_j on R^100, 0.42 s / 70 MB; d of ω_i = x_{i+1}, n = 1000, 0.18 s. All are within
  the rows of 1.9. `t_b2 negp 80 197` is `Unresolved` after 100000 units in 0.148 s, and the 33
  calls of `sweep4.txt` give the answers and units of 1.6.3. The 0.25 s worst case stands.
* Callers (`git grep` over the workspace): the callers of d, wedge, `is_closed`, `from_tensor`,
  `from_components`, `num_components`, `to_flat` and `apply_to_tensor` are those of 3.4 and 6.3.
  `TensorFieldModuleTrait::total_rank` (`traits.rs:124`) is a different type, so no trait
  conflicts with the new `Result` return.
* Citations: Chapter_24 §24.4.2 and §24.4.7, Chapter_41, Chapter_42 §42.15, Chapter_82 §82.3 and
  Chapter_111 exist in `rustmath-lava/docs/`.
* S1 independence: S1's design keeps every `simplify` result (§2.10). Its parser changes (§2.12
  literal folding, §2.13 `arcsinh` and friends as `UnaryOp`) touch no input of an MF1 test, and the
  reserved names keep their meaning. With S1, `parse("arcsinh(x)")` becomes a `Unary` that the
  guard admits and `differentiate` handles. Without S1 it is a `Function`, which is refused. Both
  are sound.

## Minor issues (none blocks)

1. **Plan step 3 names the wrong prototype.** Lines 1318-1319 say "`$S/rev3/proto4.rs` lacks the
   `root_sum` rule of `unsupported_derivative` (review 3)". `proto4.rs` has the rule (it is the only
   difference from `proto3.rs`). The sentence should name `$S/rev2/proto3.rs`, or be dropped. As
   written, it tells the implementer that the reference implementation is incomplete.
2. **`test_d_chart_identity`: the HEAD column is wrong.** The design says that at HEAD "`is_closed(&u)`
   answers Ok(true)". At HEAD it answers `Ok(false)` for both `u` and `c` (`$R/r4_head.rs`,
   `r4_head.txt`). The facade d leaves unsimplified `(0 + 0) + 0` trees, and `is_zero` is
   structural. `Ok(true)` is what the new d would answer without the chart check, which is review
   3's point. The test still fails on HEAD (it expects `Err(InvalidChart)`), so only the column
   needs correcting.
3. **`check_coordinate_chart` accepts repeated coordinate names.** `Chart::new("c", 2, vec!["x", "x"])`
   is accepted (`chart.rs:44-67` checks only the count). On it, d(x dx1) gives (dω)_{01} = -1 on
   `proto4.rs` (`r4_probe.txt`: `[0, -1, 1, 0]`). In any real chart, d(x1 dx1) = 0. So `is_closed`
   would answer `Ok(false)`, a decided answer for an input that is not a chart. HEAD is wrong here
   too, the REPL refuses such charts (J1a S14), and no workspace caller builds one. Since the design
   adds a function whose job is to check the chart's coordinates, the cheap fix belongs there:
   `Err(InvalidChart("the chart 'c' repeats the coordinate x"))` from a `HashSet` pass (O(n)), plus
   a row in `test_d_chart_identity`. A one-line precondition in the docs and in section 5 is the
   alternative. Related: `manifold.chart(name)` returns the first registered chart of that name
   (`manifold.rs:112-117`), so with two registered charts of one name, d in the second one errs.
   That is an honest Err; one doc sentence covers it.
4. **Wording: the bound symbol of the Risch integrator.** 1.4 says the integrator "binds the reserved
   symbol `_t` ..., never a chart coordinate". It binds `_t0` when the integration variable is `_t`
   (`risch.rs:1151-1158`), and `Chart::new` accepts a coordinate named `_t`. The consequence is only
   a refusal, never a wrong value. Say "`_t` (or `_t0`), which no chart in the workspace uses".
5. **`test_apply_to_tensor_not_implemented`, "(1, 0): Err(InvalidOperation), as at HEAD".** HEAD looks
   the components up before the rank checks. So a (1, 0) tensor without components in the chart
   gives `NoComponentsInChart` at HEAD, and only one with components gives `InvalidOperation`
   (`r4_head.txt`). Build the (1, 0) tensor of the test with components, or drop "as at HEAD".
   (After MF1 both give `InvalidOperation`, because the lookups go.)
6. **Small text slips.** Section 5 cites only `_t` for the Risch bound symbol (see 4). The R1
   ring-predicate impl is `modules.rs:197-225`, not `:195-225`.

## Files

`$R/r4_probe.rs` and `r4_probe.txt` (memo order, one- and three-argument `root_sum`, `root_sum` in X
of the Lie derivative, repeated coordinates); `$R/r4_head.rs` and `r4_head.txt` (HEAD:
`is_closed(&u)` is `Ok(false)`, the root_sum forms, `apply_to_tensor`); `$R/measure_new4.rs` and
`measure4.txt` (1.9 rows and one budget call on `proto4.rs`, and the baseline); the designer's
harnesses copied unchanged and rerun (`t_rootsum3.rs`, `t_dguard.rs`, `t_dag.rs`, `t_b2.rs`,
`r_rootsum.rs`, `r_misc.rs`, outputs `*_mine.txt`); `refs4_mine.out`.

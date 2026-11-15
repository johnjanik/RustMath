# S1-symbolic design: adversarial review 5

Design reviewed: `fu2/design/S1-symbolic.md` (2161 lines, with "Revision 4" to "Revision 1"). HEAD dc367a2
(`git log c79a582..HEAD` ends at M1b; working tree clean). My experiments are in
`fu2/design/S1-symbolic/review5/` (below `R5/`): `build.sh` links a program against the HEAD rlibs (release
rlibs from `cargo build --release -p rustmath-symbolic -p rustmath-reals`, list in `rlibs_rel.txt`; debug
rlibs as in `rev3/rlibs_dbg.txt`), `p5/` is the revision 4 prototype `rev4/proto5/` unchanged plus three
probe drivers (`deep6.rs`, `longmono.rs`, `dag_input.rs`), `runp.py` times one call (address space capped at
6 GiB), `src/btree_mono.rs` is a micro-benchmark. Timings are release with budget 1e8 unless stated.

Verdict: **not approved**. Two new blockers. Revision 4 fixes both blockers of review 4 and its six minor
issues, and every number I reran matches. The new holes are outside the paths revision 4 changed: the
expression depth of a monomial (B1) and the monomial work of a constant product (B2).

## What I verified (holds)

- Review 4 B1 and B2, rerun on `R5/bin/proto5_rel` (`R5/repro_r4_rel.txt`): `(K + x + ... + x)*0` with 400
  `+x` `Ok` in 0.34 s (6.71e7 units, 97 MB); the 160 `-x` variant 0.23 s; 100 `x^0` at the point K 0.28 s in
  the engine; the `+y` control refused at 1.01e8 after 0.33 s; `exp((1+exp(N_16))^511 + x)` `Ok` in 31 ms,
  1.8e7 units, 4 MB; N_30 30 ms; the N_6, N_8, N_10 products refused after 0.24 to 0.25 s; `1/((1+sin(M_30))
  + x)` `Ok` in 0.6 ms (5.1e4 units) with and without a budget, M_60 in 1.8 ms; `1/((1+exp(N_16))^511 + x)`
  `UndecidedZero` in 0.05 s. The code matches §2.2 and §2.7: `SeriesConst` behind `Arc`, `copy_units` before
  every merge, negation, grouping and split, `expr_units` before every `to_expr`, 4 units per copied series
  term, keys from the normal forms with sub-kernel ids (one copy, in the table), `has_indet` from the
  arguments' kernels, `ball_memo`/`ball_cost` keyed by `Arc` address. The memo is sound: the evaluated
  `Expr` is borrowed for the whole call and the ball code builds no temporary `Expr`.
- Review 4 minors: m1 (`x + ((2+sin(1))^(-1099511627776))^(-1)`, `^(-5000)` and `x + (2*3^(-300000))^(4095/2)`
  unbounded in 0.13 ms, 0.14 ms and 4.3 ms, kernels as §4.1 says), m2 (200 cube roots: 0.11 s, 1.27e7 units),
  m3 to m5 as text. m6 is contradicted by B2 below.
- Oracles: `rev2/gpcheck3.py` on proto5 release gives 107 OK, 0 failures (`R5/gpcheck3_r5.out`);
  `revchecks.py` and `rev2/revchecks3.py` on the debug build give 0 bad; `rev4/tools/budget4.py` (debug) gives
  all 31 budget outcomes, the slowest 5.6 s (`R5/budget4_debug_r5.txt`); `rev4/tools/calib4.py` gives the 54
  outcomes of §5.3, slowest rate 5.4e7 units/s (`1/(exp(x)-1)^150`), slowest refusal 1.50 s (`tan(x)` at 1,
  order 200) (`R5/calib4_r5.txt`).
- gp (PARI/GP, `R5/refs5.gp`, `R5/s50.gp`, output `R5/refs5.out`): tan to 15, cot to 9, H49E3 (`== 1`),
  `1/(x+x^2)^200` (valuation -200, [1, -200, 20100], x^2 coefficient `== binomial(401,199)`),
  `1/(x+sin(1)+I)`, acosh at 2, `1/(2+sin(x))` at 1, the triple pole, `(8+x)^(1/3)`, `-1/tan(x)`,
  `1/(exp(x)-1)^150` ([1, -75, 11225/4]), `#binary(3^340000)` = 538 888, and `1/(sin(x)^50 - x^50)`
  (valuation -52, 28 nonzero, -3/25, -62/125, -9113/13125, equal to the prototype). Every value equals §4.1.
- Callers (grep outside `target`): `repl.rs:1981`, `limits.rs:430`, `:554`, the `lib.rs` and `series.rs`
  tests. The REPL `limit` arm calls `Expr::limit`, which never reaches `limit_via_series`. Line references
  of §1, §3.1, §3.3 (`integrate_exact.rs:96`, `:148`, `plane_curve_ops.rs:492`, `repl.rs:1973`,
  `series.rs:795`, `repl.rs:10555-10562`) are right at HEAD.
- HANDOFF coverage: F1 (algorithm, simplify and Drop, REPL bound) and S7's last part (the literal parser) are
  covered (`docs/HANDOFF.md:324-339`); I3 did S7's point part and the design keeps its test.
- Nesting depth: for nested kernels (`sin(sin(...))`, `1/(x + sin(...sin(1)...) + 2)`, `exp(...exp(x)...)`,
  `(2+sin(` chains) the engine reaches the parser's own limit (4100 levels on 8 MB, 1025 on 2 MB release, 159
  of 162 on 2 MB debug; `R5/longmono.txt`). So the ball recursion is not a limit there.
- Radical folds with many radicals (`x + R*R`, R a balanced product of 4000 square roots) run at 3e8 units/s:
  the `out.remove` loop of `reduce_mono` is not a hole.

## Blockers

### B1. A monomial's expression is a left-deep product: recursion depth is the monomial length, not the input depth (§2.2 `to_expr`, §2.3, §2.8, §5.1)

`to_expr` builds each monomial with `product()` (`rev4/proto5/konst.rs:139`, `mono_expr` `:903`), a
left-deep `((k1 * k2) * k3) * ...`. The monomial length L is not bounded by the input depth: a balanced
input `((sin(1)*sin(2))*(sin(3)*sin(4)))*...` of depth about 2 log2 N gives one monomial of N kernels. So
every coefficient, kernel expression and `1/P0` denominator that holds it has depth N, and two recursions
reach it:

- the numeric zero test: `ball_memo`/`ball_child` recurse once per level (gdb backtrace in `R5/longmono.txt`);
- the result: Display (and the derived `PartialEq`, `substitute`) of a coefficient of depth N.

Reproduction (`R5/p5/longmono.rs`, `R5/probe2.py`, `R5/longmono.txt`; P is the balanced product of
`sin(1)..sin(N)`, point 0, order 1):

| input | thread | the prototype | HEAD `taylor` |
|---|---|---|---|
| `1/(x + P - 1/2)`, N = 431 (4 580 characters) | 2 MB, debug | **stack overflow, abort** (ok at 425) | returns, result depth 16 |
| the same, N = 2200 (25 KB) | 2 MB, release | **abort** (ok at 2150) | |
| the same, N = 8800 (102 KB) | 8 MB, release | **abort** | |
| `x + P`, N = 1975 (22 KB), then Display | 2 MB, debug | **abort** (result depth 1952 at N = 1950) | N = 4000: returns, depth 15, prints 55 KB |
| `x + P`, N = 8200 (95 KB), then Display | 2 MB, release | **abort** | |

This is a regression. HEAD returns a result of depth 15 to 18 on every row. The REPL doc comment says that an
abort kills the kernel (`repl.rs:10474-10475`), the REPL runs on tokio threads until J1b moves it, and cargo
test threads have 2 MB. §0.4 and §5.1 claim that no consumer recurses once per term and that the result is
fine on 2 MB threads. That holds for the terms of the sum, not for the factors of a monomial.

Required change:

1. Build every product of kernels balanced, as `balanced_sum` does for sums: the numerator and the
   denominator of `mono_expr`, so every coefficient, kernel expression and `Inverse` denominator. A monomial
   then has depth about log2 L + 3. State in §2.2 and §2.8 that a coefficient has depth O(log t + log L) per
   level of kernel nesting, and that kernel nesting is at most the input's.
2. Make `enclose_real` iterative (an explicit post-order stack, as `ball_cost` already is), or give it a depth
   limit beyond which it returns `None` (no enclosure: `Unknown`, sound). Say which in §2.3. The same for any
   other recursive walk of a kernel's expression (`display_estimate` stops after about 34 nodes, so it is
   fine; say so).
3. Tests (debug, on a 2 MB thread, outcome asserted): `1/(x + P - 1/2)` and `x + P` with N = 4096 at 0,
   order 1: no stack overflow (`Ok` or `UndecidedZero` for the first, `Ok` for the second), the result's
   tree depth at most 64, and `format!("{}")` of the second succeeds. Both abort on the revision 4
   prototype.

### B2. `SeriesConst::mul` charges one unit per monomial factor, but its map search costs O(L log t) per pair and each factor takes 16 bytes (§2.7)

`mul` charges `ta Lb + tb La` for the monomial work, then `q_mul` and `q_add` per pair
(`rev4/proto5/konst.rs:549-597`). Two costs are not in it:

- The accumulation map. Each product monomial is looked up and inserted in a `BTreeMap<Mono, Rational>`.
  Comparing two monomials that share a long prefix walks the prefix (`Ord` of `Vec<(Kernel, i64)>`, each step
  dereferencing both kernels' `KData`), and a B-tree search makes about 25 to 36 comparisons. So a pair costs
  O(L log t) against a charge of L. `R5/src/btree_mono.rs`: inserting 2^15 keys of length 1415 that share a
  1400-factor prefix takes 3.3 s, 100 us per insert.
- Memory. Each new monomial is a `Vec` of 16 bytes per factor, charged 1 unit per factor. §2.7 sizes every
  other copy at about 4 bytes per unit (`copy_units` charges `4 L` per term).

Reproduction (`R5/mem_mono2.py`, `R5/mem_mono3.py`, outputs `.txt`; Q is the balanced product of
`sin(1)..sin(M)`, S the product `(1+cos(1))*...*(1+cos(k))`, point 0, order 1, budget 1e8):

| input | length | outcome | wall | units | peak memory |
|---|---|---|---|---|---|
| `x + Q*S*0`, M = 1400, k = 15 | 15 867 | `Ok` | **13.3 s** | 9.71e7 | **1.09 GB** |
| `x + Q*S*0`, M = 2900, k = 14 | 33 855 | `Ok` | **13.4 s** | 9.81e7 | **1.13 GB** |
| `x + Q*(S)*0`, M = 1400, k = 15 (one product) | 15 869 | `Ok` | 7.8 s | 5.37e7 | 740 MB |
| `x + Q*S*0`, M = 1400, k = 16 | 15 879 | refused at 1.89e8 | 13.4 s | | |

The rate is 7.3e6 units/s, below the floor of 1e7 that §5.3 assumes and 8 times below its slowest row, and
the memory is 11 bytes per unit against the "at most 3.0" of §2.7, the `TAYLOR_WORK` doc comment (§3.1) and
revision 4's m6. So 1e8 units hold over 1 GB, not "at most about 300 MB" (the REPL policy is about 400 MB,
`repl.rs:10477-10480`). The calibration has no long-monomial row (K and S have at most 18 factors per term),
so the recalibration rule of §5.3 would not catch it.

Required change:

1. Charge the product's monomials by their memory, `4 L` per new monomial as `copy_units` does, and the map
   search by its comparisons, for example `L * (bits(t_acc) + 1)` per lookup, where t_acc is the size of the
   accumulator. Or make the comparisons O(1): intern monomials per expansion (hash-consed ids, as kernels
   are), so a map key compares in O(1) and only a new monomial costs O(L). The same search term applies to
   every other insertion of a term into a map (the merge of `add`, `indet_groups`, `split_imag`): their
   copies are charged, their searches are not (measured fine at 4 L today, about 4e7 units/s, but state the
   bound).
2. Add the two `x + Q*S*0` rows to §5.3 with their bytes per unit, and redo the memory figure of §2.7 and of
   the `TAYLOR_WORK` doc comment from the worst row.
3. Budget test (debug, outcome asserted, "returns within 20 s"): `x + Q*S*0` with M = 1400, k = 15 at 0,
   order 1, budget 1e8: `BudgetExceeded` (or `Ok`) within 20 s. It runs 13.3 s in release on the revision 4
   prototype, longer in debug.

## Minor issues (fix in the revision)

- **m1. DAG inputs.** A REPL variable can hold a DAG: `f = f + f` repeated (J1b measured it), and
  `parse_symbolic_arg` passes the stored `Expr` unchanged (`repl.rs:4942-4945`). The dependence pre-pass of
  §2.5 is keyed by node address but never checks the map before descending (`rev4/proto5/ser.rs:781`), so it
  walks the tree, uncharged, before the first charge. `R5/p5/dag_input.rs` (`R5/dag.txt`): with
  `e = x; e = e + e` k times, `expansion(.., 1e8)` returns `BudgetExceeded` after 0.14 s at k = 20 and 0.62 s
  at k = 24, doubling per k, so at k = 40 it does not return. This is not a REPL regression: the arm's
  `args[0].symbols()` (`repl.rs:1965`) walks the same tree first (1.14 s at k = 24) at HEAD. But §2.7 says
  `taylor_with_budget` returns `BudgetExceeded` at the first charge that passes the budget, and the
  `TAYLOR_WORK` doc comment says it bounds the work of one call. Required: check the pre-pass map before
  descending (one line), test the point for the variable with the same memoized walk instead of
  `contains_symbol`, find the REPL arm's variable with a DAG-safe walk (S1 rewrites that arm), and state in
  §2.5 that a DAG input is expanded once per occurrence, each occurrence charged. Test: the k = 40 DAG through
  `taylor_with_budget(.., 1e8)` returns `BudgetExceeded` within 20 s.
- **m2. The rendering rebuilds the point.** §2.8 decides `x` versus `x - a` by `SeriesConst::from_expr(point)`
  being zero, and the prototype's `to_expr` calls `konst_of(a)` again after the expansion
  (`ser.rs:980-990`). `SeriesExpansion::to_expr()` (§2.9) has no `Work`, so a port rebuilds the point
  uncharged at every call (0.25 s for the point K). Record whether the point is zero when the expansion is
  made (the driver already knows it from the variable's series) and keep it in `SeriesExpansion`.

## Summary for the orchestrator

Revision 4 resolves review 4: shared coefficients and the copy charges close B1, normal-form keys and the
memoized ball code close B2, and every review 4 input, gp value, oracle and budget test I reran matches. The
algorithm, the zero test, the API, the callers, the parser and Display, Drop and HANDOFF coverage hold. Two
new blockers stop approval.

- B1: monomials are rendered as left-deep products, so a balanced 4.6 KB input aborts the process with a stack
  overflow in the ball evaluation (debug, 2 MB thread; 25 KB in release), and a result prints at depth N
  where HEAD gives depth 15 to 18. Fix: balanced products and an iterative (or depth-capped) `enclose_real`.
- B2: a constant product charges one unit per monomial factor, but its map search is O(L log t) and its
  monomials take 16 bytes per factor. A 15.9 KB input is admitted after 13.3 s with 1.09 GB under
  `TAYLOR_WORK` (7.3e6 units/s, 11 bytes per unit). Fix: charge the search and the memory, or intern
  monomials, and calibrate the family.

The minors are DAG inputs walked as trees by the uncharged pre-pass (m1) and the rendering's rebuild of the
point (m2).

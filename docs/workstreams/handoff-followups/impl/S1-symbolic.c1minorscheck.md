# S1 C-1 minor items: independent check

Checked: branch `claude/gracious-gates-e6b6l7`, HEAD 226a7ec (clean; its Rust code is that of
a4f7791). The delta of the minors step is `c1mcheck/c1minors_delta.patch`. The state before the
minors step ("pm" below) is a4f7791 with that patch reversed (`c1mcheck/pm/`); its seven source files
are byte-identical to 52708d7 plus `c1m/diff_before.patch`, so the patch is the whole code change
of the step. Read: the delta, `tests/s1_cuts.rs` whole, section "C-1 minor items" of `S1_impl.md`,
and the two C-1 reviews. Paths below are relative to `fu2/s1impl/c1mcheck/` unless they start with
`rustmath-`.

Nothing in the repository was edited, committed, pushed, stashed or reset. `git status --short` is
empty at the end, `rustmath-misc/output.txt` has md5 746308829575e17c3331bbcb00c0898b, no
`.tmp_dir_*` exists and `tests/data` has no `__pycache__`. `target/release` was deleted after the
release runs; the probe executables and the pre-minors rlib built in scratch were deleted.

## Verdict

Approve. No blocker under the narrowed bar.

- (a) No wrong, guessed or fabricated value. Every value that the step turned from a refusal into a
  value (888 on my probes, 52 on the reviewers' corpora) passes a two-sided check against gp's scalar
  functions at three generic complex parameter values; the only exceptions are real parameter
  values, which the documented convention allows. The new rules (`gnc`, `kpos`,
  `scaled_enclosed`, `pow_general`'s class) are sound by the argument in section 2, and the
  constants in disguise I built all refuse.
- (b) No value changed and no value became a refusal on 91 773 probe rows (the reviewers' corpora
  and two of my own). One minor budget regression: `scaled_enclosed` encloses the constant of every
  product chain, whether or not a cut check reads the class, so a 99.8 KB input at order 30 that
  the approved C-1 code expands under the REPL budget now exceeds it (m1). The C-1 regressions
  review graded the same kind of effect (r-m4) minor; I grade it the same way, but it is the
  owner's call.
- (c) The new paths are within the bar: at most 1.46 s per entry point and 45 MB in release on an
  8 MB thread, and no stack overflow on a 2 MB debug thread, on 41 inputs of up to 100 KB.
- Tests: debug 909 passed, 0 failed, 35 ignored; release `s1_cuts` 14 passed, 1 ignored, and the
  ignored release test passes in 4.16 s.

## 1. The claimed changes against the code

Each claim of the report matches the code:

- s-m2, r-m3: `Display` of `NotLaurent` reads "no Laurent expansion found at the point: ..."
  (`series.rs:261`). No crate or test of the workspace matches the old text (grep). In `position`,
  c0 = i t on an imaginary-axis cut, with t not written out and its position undecided, reports
  `undecided(1 + c0^2)` (`series_engine.rs:1452`); the product is charged before it is built, and a
  budget fault wins in `undecided_expr`.
- s-m3: `Tail::scaled_enclosed` (`series_engine.rs:1225`, called at `:2576`) reads the class from one
  64-bit `cball_konst` (charged), only when the structure decides nothing, the tail has a class, and
  the constant's tree has at most 256 nodes.
- s-m4, s-m5: `s1_gp.py` snaps the scalar F's branch arguments below 1e-250 and `AB` is gp's `abs` on
  a scalar. My run of `python3 -I -B s1_gp.py --two-sided` on the three data files exits 0 with the
  report's counts (cuts 54 value outcomes, 0 fail; singular 367, 0 fail; fuzz 394, 0 fail).
- s-m6: `s1_cuts.rs:178` asserts `LimitResult::Unknown` for `limit_with_error`.
- r-m1: `KData::gnc` (`series_const.rs:113`, set at `:3474` by `kernel_generic_nonconstant`,
  `:3592`) and `Konst::generic_nonconstant` (`:2699`), used by `position` (`series_engine.rs:1384`).
- r-m2: `pow_general` (`series_engine.rs:2609` to `:2617`) gives f^g the real class for a real f
  whose constant term is positive by `struct_positive` or by `real_sign`, and a real g.
- r-m4 (partial): `KData::kpos` (`:109`, set at `:3472` by `kernel_struct_pos`, `:3542`),
  `struct_positive` and `struct_gt1` through `struct_lower` (`:3505` to `:3535`), `arccosh` of a
  structural c > 1 is real (`:3707`).
- `position_undecided` returns the budget fault after each enclosure (`series_engine.rs:1366`).
- `tests/s1_cuts.rs` has the five new tests and the stronger assertion; eight rows appended to
  `s1_cuts.tsv`, and the 82 rows of C-1 are unchanged.

## 2. Bar (a): attacks on r-m1, r-m2, s-m3 and kpos

### 2.1 Why the rules hold

- `gnc` by induction: a parameter is a nonconstant function; 1/P, P^r (r a nonzero rational), q^A
  (q a rational other than 0 and 1) and f(A) for an analytic f in `GENERIC_ANALYTIC` are nonconstant
  on every open set where P or A is, since a nonconstant holomorphic function is open and each f is
  nonconstant on every open set of its domain. `generic_nonconstant` accepts only a polynomial in the
  parameters with a nonconstant monomial of decided nonzero coefficient, or R(k) for one `gnc`
  kernel k, coefficients free of parameters, and a decided nonzero coefficient of some k^e, e != 0.
  R is then a nonconstant Laurent polynomial and R(k) a nonconstant function, so its preimage of a
  cut has no interior. Any second kernel with a parameter, any parameter as a factor, abs, sign, mod
  and named functions are rejected; `exp(2*pi*I)^a` is rejected because the base is not a rational.
- `kpos`: each rule (pi, e, radicals n^(1/d) with n > 1, positive floats, 1/P and P^a for P > 0 and
  real a, exp and cosh of a real, sqrt, sinh, tanh, arctan, arcsinh, erf, gamma, factorial of a
  positive constant, log and arccosh of a constant > 1) gives a positive number, and `struct_lower`
  requires every non-rational term to have a positive coefficient and positive factors.
- `scaled_enclosed`: an exact 0 part of a `CBall` arises only from exact arithmetic (Ball::add and
  Ball::mul keep exact zeros only when the exact result is 0, and the real step of a total function
  keeps the imaginary part exact); a constant with a parameter has no enclosure.
- `pow_general`: f real with f(a) > 0 and g real make exp(g log f) real near the point.

### 2.2 Probes

Release probe `drv/p.rs` (the C-1 regressions reviewer's probe: `series_expansion` with the REPL
budget, `taylor_with_budget`, `limit_via_series`, `limit_with_error`) built against the current
library and against the pre-minors library compiled in scratch (`buildpm.sh`, `drv/build.sh`).

| corpus | rows | changed outcomes | values that appeared |
|---|---:|---|---:|
| `par/tgt.tsv` (made by `par/tgt.py`): each branch function of each inner function of a parameter at a, at sin(a), Laurent polynomials in one kernel, nested kernels, 26 constants in disguise, variable exponents of 23 bases, 28 scaling constants real or imaginary only by enclosure | 12 856 | 1 743: 937 `NotLaurent` to `UndecidedZero`, 806 `NotLaurent` to a value | 806 |
| `par/g.tsv` (made by `par/gen.py`, seeds 1 and 2): random trees with parameters a, b, y and points 0, a, b, sin(a), log(a), exp(a), a + b, I, ... | 8 000 | 988: 903 to `UndecidedZero`, 3 to `Unsupported`, 82 to a value | 82 |

Two-sided check (`par/head.gp`, `par/mk.py`: the series against the function at a + h and a - h,
h = 10^-20, realprecision 400, the parameters set to 3/7 + 2i/9, -5/7 + i/3, 0.31 - 0.87i (set 0),
-5/7 + i/3, 1.3 - 2.2i, -2.1 + 0.05i (set 1), 0.913 - 1.37i, 0.2 + 0.6i, 1.7 + 0.4i (set 2)):

- The 82 new values of `g`: 82 of 82 pass at each of the three complex sets.
- The 806 new values of `tgt`: 770 pass at each set; the other 36 have the base
  `e - 2.718281828459045` (1.4e-16 > 0, positive by its enclosure), whose radius of convergence is
  below h. Compared coefficient by coefficient with gp's formal series (`par/formal.gp`), 31 agree to
  1e-100 and the 5 `arccosh` rows agree with the principal `ACOSH` of `s1_gp.py` (gp's own series of
  acosh takes the other sign left of -1).
- All 3 947 values of `g` (old and new) pass at sets 0 and 2; two rows could not be evaluated by gp
  (`.../zeta(-2)` and `.../arctan(sqrt(-1))`, the documented convention for a constant the input
  divides by, unchanged by this step).
- Constants in disguise refuse: `sin^2 + cos^2`, `exp(x)*exp(-x)`, `exp(x)^2/exp(2*x)`,
  `sqrt(exp(2*x))/exp(x)`, `tan*cos/sin`, `cosh^2 - sinh^2`, `exp(log(sin(x))) - sin(x)`,
  `exp(I*pi*x)^2*exp(-2*I*pi*x)`, `abs(exp(I*x))`, `sign(x)^2`, `abs(x)` and the others of
  `tgt.py`, inside each branch function at a and at 0: none became a value.

At real parameter values (a = 3/7, b = -5/7, y = 2/3 and a = -5/7, b = 3/7, y = -2/3) 24 and 10 new
values hold on one side only, all of the form F(-g(x) + I*(x - a)) or F(sqrt(x)*sqrt(x)) at a
(the latter is gp's rounding of (a^(1/2))^2 on the cut; the engine's value is right). This is the
real exception set that `series.rs:76` to `:87` documents (observation O1).

## 3. Bar (b): regressions

| corpus | rows | changed | changes |
|---|---:|---:|---|
| `c1-1-regressions/hunt/h.tsv` | 1 807 | 2 | `NotLaurent` to a value (`log(-5 + (pi + x)^x)`, `sqrt(-4 + (pi + x)^x)`) |
| `c1-1-regressions/rnd/r.tsv` | 6 000 | 0 | |
| `c1-1-regressions/corp/all.tsv` | 63 110 distinct | 128 | 76 to `UndecidedZero`, 2 to `Unsupported`, 50 to a value; the 50 equal the reviewers' output of the code before C-1 (`all.pre.gz`) and pass two-sided at all five parameter sets (one is one-sided at a real set) |
| `par/tgt.tsv`, `par/g.tsv` | 20 856 | 2 731 | as in section 2.2 |

In every corpus no value changed (the series, the precision, the limit and the limit with error of
every row that was a value before are identical), and no value became a refusal. These match the
report's counts. Outputs: `out/*.gz`, `out/all.diff`, `par/*.gz`, `par/*.res`.

Budget. `drv/mb.rs` finds by bisection the least budget under which `series_expansion` gives its
unlimited outcome. On 1 455 rows sampled from the corpus that are values in both versions, the ratio
current / pre-minors has median 1.00, 99th percentile 1.05, maximum 2.82 (`out/mbc.*`). On 500 rows
of my corpora: median 1.00, 90th percentile 1.34, maximum 1.97 (`par/mb.*`). The large ratios come
from `scaled_enclosed` (m1); the `gnc` flags add at most 1.2 on inputs whose kernels have large
parametric arguments (`par/kp.*`).

## 4. Bar (c): resources of the new paths

Release driver `drv/ep.rs` (all four entry points with the REPL budget, plus `to_expr`, `Display`,
`Debug` and re-parse of the series) on an 8 MB thread, and its debug build on a 2 MB thread, on 41
inputs of up to 100 KB made by `res/mk.py`: nested kernels of a parameter at a (log, log(sin),
sqrt(log), arcsinh(sin), log(2 + sin), arctan(log)), the same inside a constant term, Laurent
polynomials in sin(a) with 2 245 undecided coefficients, many cut checks of one Laurent polynomial,
kernels of large parametric arguments, `kpos` chains (sqrt, log(2 + ...), arccosh, sqrt(1 + exp(-...)),
(1 + ...)^pi, gamma(1 + ...)), wide sums of radicals, `pow_general` with positive enclosures (many,
large, nested), products scaled by constants real by enclosure (many, large, long monomials), and
imaginary c0 with t not written out (many, large, and an undecided t with 3 000 terms).

| | release, 8 MB | debug, 2 MB |
|---|---|---|
| largest time of one entry point | 1.46 s (`sc_many`, 3 159 terms `log(-k + log(-1)^2*sin(x))`, `BudgetExceeded`) | 7.77 s, same input |
| largest VmHWM | 45 MB (`gnc_poly_many`) | 46 MB |
| stack overflow, abort, panic | none | none |

Logs: `res/ep_cur.log`, `res/ep_pm.log`, `res/ep_dbg.log`.

## 5. Tests

- `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug,
  `sym_dbg.log`): unit 686 (5 ignored), `s1_cuts` 14 (1), `s1_depth` 14, `s1_finish1` 11 (1),
  `s1_fuzz_data` 7, `s1_payload` 2 (1), `s1_regressions` 26 (2), `s1_review2` 2, `s1_review3` 7 (7),
  `s1_review4` 4 (2), `s1_review4_memory` 1 (1), `s1_review5` 4 (2), `s1_review5_memory` 2 (2),
  `s1_review6` 4, `s1_singular_points` 9, doc 116 (11): 909 passed, 0 failed, 35 ignored.
- `cargo test --release -p rustmath-symbolic --test s1_cuts`: 14 passed, 0 failed, 1 ignored.
- The same with `-- --ignored --test-threads=1`: `cut_rule_within_the_bar_release` passes in
  4.16 s (0.44 s, 0.62 s and 3.06 s for its three inputs).

## 6. Minor items

**m1. `scaled_enclosed` encloses every product's constant, needed or not**
(`series_engine.rs:1225` to `:1241`, called at `:2576`). Every product chain whose constant is not
structurally real or imaginary (log(-k), arcsin(k), log(k + I), (-k)^(1/3), ...) and whose series
factor has a class pays a 64-bit complex enclosure, even when no branch function ever reads the
class. On such inputs the work is 4.4 to 6 times the pre-minors work at orders 1 and 2 (`par/sc*.`).
Evidence: `res/scflip.txt`, 99 800 bytes, the sum over k = 1..486 of
`(log(1+I)+log(2+I)+...+log(20+I))*sin(k*x)`, at 0, order 30, REPL budget 1e8: the approved C-1
code gives `Ok` (least budget 6.20e7) from `taylor_with_budget` and `series_expansion`, and
`limit_with_error` gives `Finite(0)` with an error bound; the current code gives `BudgetExceeded`
(least budget 1.03e8) from both, and `limit_with_error` gives `Finite(0)` without a bound. The
extra work is charged and bounded (about 4e7 units for 100 KB), so bar (c) holds; it is a value
that became a refusal at the budget's edge, the kind the C-1 review graded minor (r-m4). Change:
read the enclosure only where the class is consumed (for example keep the constant's class
undecided in `Tail` and enclose it in `cut_check` when `tb` does not decide), or cache it per
constant; otherwise record the cost in the doc of `scaled_enclosed` and in the handoff.

**m2. `generic_nonconstant` walks before it charges** (`series_const.rs:2716` to `:2735`). The doc
says it is charged as `indet_groups`, but in the one-kernel path the walk over every factor runs
first and returns false without a charge when it meets a parameter factor or a second kernel. The
walk is linear in a constant whose construction (or kernel key) was charged, and no input of
section 4 shows a cost; the doc should say so.

**Observation O1 (owner's convention, as s-m1).** Under r-m1 a constant term that is a kernel of a
parameter is generic, so `log(-sin(x) + I*(x - a))` at a now returns the formal series, which for
every real a with sin(a) > 0 (an open set of real values) is the function on one side only; the C-1
code refused it, HEAD-before-S1 returned the same series. The module conventions (`series.rs:76` to
`:87`) document this; if the owner chooses real parameters, the rule must refuse these.

## 7. Files

Scratch `fu2/s1impl/c1mcheck/`: `c1minors_delta.patch` (given); `pm/` (the pre-minors source),
`buildpm.sh`, `rlibs_rel.txt`, `rlibs_dbg.txt`, `relbuild.*`; `drv/` (`p.rs`, `ep.rs`, `mb.rs`,
their per-library copies, `build.sh`); `out/` (outputs of the reviewers' corpora, `cmp.py`,
`all.diff`, `mbc.*`); `par/` (`tgt.py`, `gen.py`, `head.gp`, `formal.gp`, `mk.py`, the corpora,
outputs, gp results, `mb.*`, `sc*.`, `kp.*`); `res/` (`mk.py`, the 41 inputs, `scflip.txt`,
`ep_*.log`, `run_dbg.sh`); `sym_dbg.log`, `oracle.log`.

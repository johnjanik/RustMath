# S1 C-1 minors check items (m1, m2): independent check, semantics lens

Checked: branch `claude/gracious-gates-e6b6l7`, HEAD 226a7ec (clean at the start; its Rust code is
that of a4f7791), plus the uncommitted fix of m1 and m2 (`git diff`: `series_engine.rs`,
`series_const.rs`, `tests/s1_cuts.rs`, `tests/data/gen_s1_cuts.py`, `tests/data/s1_cuts.tsv`). Read:
the diff, section "C-1 minors check items" of `S1_impl.md`, `S1.c1minorscheck.md`, and the engine
code that builds and reads the classes (`Tail`, `cut_check`, `atan_like`, `log`, `pow_const`,
`pow_by_const`, `pow_general`, `pow_const_base`, `expand_unary`, `EFrame::resume`, `expand`,
`finish_product`, `cball_konst`, the budget functions, the driver `expansion`).

Three libraries built in release from separate source trees in scratch: HEAD (`git archive
226a7ec`), the C-1 approved state (`git archive a4f7791` with `c1mcheck/c1minors_delta.patch`
reversed; byte-identical to the checker's `c1mcheck/pm/`), and the fix (the working tree). A fourth,
instrumented library (below) was built from the fix. Paths below are relative to
`fu2/s1impl/closeout/check1-semantics/` unless they start with `rustmath-` or `c1mcheck/`.

Nothing in the repository was edited, committed, pushed, stashed or reset; `git status --short`
lists only the five files of the fix. `rustmath-misc/output.txt` has md5
746308829575e17c3331bbcb00c0898b, no `.tmp_dir_*` exists, `tests/data` has no `__pycache__`.

## Verdict

**Approve. No blocker under the narrowed bar.**

- (a) No wrong, guessed or fabricated value and no undecided answer reported as decided. The fix
  changes no decision: where a cut check can read a product's class the code is HEAD's call, at the
  same precision, with no cache; elsewhere the class it computes is read by nothing, which an
  instrumented build confirms on 107 527 rows (0 reads of a class the fix computes differently).
  Every value that the fix adds over HEAD (the scflip rows) equals the closed form or gp's series.
- (b) No regression. On 107 527 rows (the five corpora of the previous rounds and my own enclosure
  corpus), every output field of the fix equals HEAD's; against the C-1 approved state, no value
  changed and no value became a refusal (the differences are the known C-1 minor-item moves from
  `NotLaurent`). The only outcomes that move are at the budget's edge, from `BudgetExceeded`:
  the m1 input to a value, and two variants I built, one to a value (checked against gp) and one to
  a decided `NotLaurent` that is right and equals the C-1 approved state's (observation O1).
- m2 is a doc-only change and its text matches the code.
- Debug suite: 911 passed, 0 failed, 36 ignored.

## 1. The fix against the code

**m1.** `expand` (`series_engine.rs:2527`) counts the records on its work stack whose own step
checks a cut (`cut_records`); a product record notes at its push whether one lies below it
(`EProd::class_read`), and `finish_product` (`:2607`) calls `Tail::scaled_enclosed` (unchanged body,
64 bits, `SCALE_ENCLOSE_TREE` limit) only then, `Tail::scaled` otherwise. I checked the argument the
fix rests on:

1. *Only cut checks decide from a class.* The classes flow from a child's `SV` into its parent's
   class (`Sum` by `and`, `Prod` by `full_class`/`times`, `Unary` by `expand_unary`, `Tower` by
   `pow_by_const`, `pow_general`, `pow_const_base`). The only reads that change an outcome are
   `cut_check` (`:1531`), reached from `log` (`:779`), `pow_const` (`:895`, only with
   `Some(tb)`: `pow_by_const` for a non-integer exponent and `Sqrt`) and `atan_like` (`:1587` to
   `:1589`, all inverse functions but erf). `pow_general`'s real-class test and `const_base_class`
   read classes only to compute a class (the former may charge `real_sign`), and both run inside a
   tower record, which `checks_a_cut` counts. `expand` returns the series without its class, and
   every entry point (`series_expansion`, `taylor_with_budget`, `limit_via_series`,
   `limit_with_error`, `to_expr` of the result) goes through `series_engine::expansion`.
2. *The ancestors are exactly the records below on the stack.* Frames are created only by
   `expand_visit` and pushed only in `expand`; a frame is popped only when it is done, so while a
   product record is on the stack the records below it are its ancestors and do not change.
   `checks_a_cut` is a function of the frame's variant and operator, so the counter is raised and
   lowered by the same frame (no underflow; the debug suite runs it with overflow checks).
3. *The set counted is a superset of the readers.* `Log`, `Sqrt`, the six inverse functions other
   than erf, and every tower (including integer exponents and constant bases, which check no cut:
   only extra enclosures, as on HEAD). Every `UnaryOp` is matched in `expand_unary` (`Neg`, `Exp`,
   the trigonometric and hyperbolic functions, `Erf`, `Abs`, `Sign`, `Gamma`/`Factorial`/`Zeta`
   refused), so no other unary step reaches a cut check.
4. *No state crosses contexts.* `cball_konst` has no side effect but the budget (`COST`, the sticky
   `FAULT`); `cball_expr` keeps a local memo; `to_expr` reads a hint but sets none; there is no memo
   of series or classes per input node (each occurrence is expanded and charged anew), so a DAG input
   shares no class between occurrences. The fix adds no cache, so there is no cache key that two
   constants could share and no stale entry; the enclosure precision is unchanged.
5. *Fewer charges change only budget faults.* The other uses of the budget (`remaining()` in
   `ball_cost`, the kernel key walk and the product pair count) only bound walks before a fault, and
   the driver shares one budget across its attempts and returns a fault at once. So the fix can only
   turn a `BudgetExceeded` into whatever the unlimited computation gives.

**m2.** The `series_const.rs` diff touches comment lines only. The new doc of
`Konst::generic_nonconstant` (`:2687` to `:2705`) is true: the polynomial case calls
`indet_groups`, which charges before its walk; the kernel walk runs uncharged, holds only an
`Option<&Kernel>` (no allocation), compares kernels by id (`Kernel::eq`, O(1)), is linear in the
terms and factors, and returns false at an indeterminate factor, a second kernel with a parameter,
or no `gnc` kernel before any charge; the grouping charge is `indet_groups`'s formula, before the
grouping runs.

## 2. Attacks on the lazy enclosure

### 2.1 Taint instrumentation

`mktaint.py` makes an instrumented copy of the fixed `series_engine.rs` (`src/taint/`): `Tail` gets
a taint bit; a product with `class_read` false computes the enclosure anyway (so its charges are
HEAD's), returns the structural class (the fix's), and is tainted when the two differ; the taint is
or-ed through `and`, `times`, `scaled`, `scaled_enclosed`, `full_class`, `atan_like`, and at every
record from each operand's class into the result's; every read of a tainted class by `cut_check`, by
`pow_general`'s class test or by `const_base_class` is counted. The probe `drv/pt.rs` prints the
counts per row. Expected: outputs equal to HEAD's (HEAD's charges, and the fix's classes where no one
reads them) and no tainted read.

### 2.2 My enclosure corpus

`gen_enc.py` (seeds 1, 2, 3; 15 000 rows, `t/enc.tsv`): products scaled by 33 constants, 17 of them
real or imaginary only by an exact zero part of their enclosure (`log(-1)^2`, `I*log(-1)^2`,
`log(-1)^3`, `pi*log(-1)`, `sqrt(-2)*sqrt(-3)`, `I*sqrt(-2)`, `cos(I)`, `sin(I)`, `tan(I)`, ...),
plus complex and structural controls; wrapped in 0 to 3 layers of class-keeping or class-changing
nodes (exp, sin, sinh, tan, tanh, erf, cos - 1, integer powers, products, quotients, sums with other
products, `abs(x)*...`, and the branch functions themselves); then placed under one of 15 branch
functions with constant terms on and off their cuts (log, sqrt, the powers 1/3, 2/3, pi, x,
x + 1/2, arcsin, arccos, arctanh, arctan, arcsinh, arccosh, 2^(...), (-2)^(...)), beside them, above
them (an outer product, sum or exp), nested in a second branch function, or under none; at 0, 1, -1,
pi, 1/2, a and infinity, orders 2 to 4.

| | rows | result |
|---|---:|---|
| fix against HEAD | 15 000 | identical in every field (series, precision, taylor, limit, limit with error) |
| taint build against HEAD | 15 000 | identical; 6 430 products in 1 311 rows had a skipped enclosure that would change their class; **0 reads** |
| fix against the C-1 approved state | 15 000 | 4 904 values identical; 824 `NotLaurent` became 428 values and 396 `UndecidedZero` (the C-1 minor items, as on HEAD); no value changed or became a refusal |

Outcomes of the fix: 5 332 values, 4 071 `NotLaurent`, 5 447 `UndecidedZero`, 150 `Unsupported`.

**gp, two-sided** (`gp/`; `c1mcheck/par/head.gp` and `mk.py`: the series against the function at
a + h and a - h, h = 1e-20, realprecision 400, parameter sets 0 and 2 of `mk.py`, both complex):
5 288 values at finite points. 5 234 (set 0) and 5 227 (set 2) pass directly. The rest:

- With the branch arguments of F snapped to their cut's axis when the other part is a rounding
  residue below 1e-250 (`gp/snaphead.gp`, `gp/snapmk.py`, the rule of `s1_gp.py`'s SNR/SNI), 10
  more pass. Without the snap, gp's `log(-1)^2 = -pi^2 + 0.E-37*I` puts the argument on a random
  side of the cut and both sides fail by O(1).
- 31 (set 0) and 35 (set 2) fail with a ratio that is the same at h = 1e-20 and h = 1e-40
  (`gp/encs40.*.res`): the error scales as h^P, so it is the first omitted term with a large
  coefficient (e^(pi^2) scalings, sinh(2 pi), nested exponentials), not a wrong coefficient; 37 of
  them also agree with gp's formal series to 1e-100 relative (`gp/encf.*.res`).
- 16 (set 0) and 20 (set 2) cannot be evaluated by gp (overflow of exp, cosh or erfc at
  realprecision 400, or erfc of a series).

The 428 values that the C-1 minor items added over the C-1 approved state (HEAD's, unchanged by the
fix): 414 pass directly, 10 with the snap, 3 fail only by an h-independent truncation ratio (two
also agree with gp's formal series), 1 is at infinity.

### 2.3 The corpora of the previous rounds

Probe `c1mcheck/drv/p.rs` built against each library (`build/bin/p_*`, `run_corp.sh`, outputs
`out/*.gz`, comparison `cmp.py`, which ignores only the time and taint fields):

| corpus | rows | fix vs HEAD (mine and the checker's `cur` outputs) | taint build vs HEAD | taint reads | fix vs C-1 approved |
|---|---:|---|---|---:|---|
| `c1-1-regressions/hunt/h.tsv` | 1 807 | 0 differ | 0 differ | 0 | 2 `NotLaurent` to a value; the rest message text |
| `c1-1-regressions/rnd/r.tsv` | 6 000 | 0 | 0 | 0 (132 tainted) | message text only |
| `c1mcheck/par/tgt.tsv` | 12 856 | 0 | 0 | 0 | 806 to a value, 937 to `UndecidedZero` |
| `c1mcheck/par/g.tsv` | 8 000 | 0 | 0 | 0 (20 tainted) | 82 to a value, 903 to `UndecidedZero`, 3 to `Unsupported` |
| `c1-1-regressions/corp/all.tsv` | 63 864 (63 110 ids) | 0 | 0 | 0 (45 tainted) | 50 to a value, 76 to `UndecidedZero`, 2 to `Unsupported`, 6 `BudgetExceeded` with other unit counts |

My own HEAD build (`out/*.head.gz`) gives outputs identical to the fix's and to the checker's HEAD
outputs on all five corpora. No value of the C-1 approved state changed in any corpus. The one
driver panic (the point `1e400`, a parse in the driver) is in every library.

### 2.4 Shared nodes (DAG inputs)

`drv/dag.rs` builds inputs whose product node P = c*sin(x) is one `Arc` shared by a branch function
and a context that reads no class, in both orders (`exp(P) + log(-1 + P)`, `log(-1 + P) + exp(P)`,
`P*log(-1 + P)`), and by two branch functions (`arctan(2I + P) + log(-1 + P)`,
`P*P + arctan(2I + P)`), for six constants. Series (`Display` of `to_expr`), taylor, limit and limit
with error are identical in HEAD and the fix (`t/dag.head`, `t/dag.fix`, `t/dag.pm`). The values are those of
the C-1 minor items: for c = log(-1)^2, `exp(P) + log(-1 + P)` is `1 + log(-1) + (-1/6)log(-1)^6 x^3`,
which I checked by hand (the x^3 coefficient is pi^6/6); `arctan(2I + P)` with a real P leaves its
cut and refuses, with an imaginary P (c = I*log(-1)^2, log(-1)^3) it stays and is a value.

### 2.5 The m1 input and inputs at the budget's edge

`c1mcheck/res/scflip.txt` (99 800 bytes, 485 products, at 0, order 30, REPL budget):

| | `series_expansion` | `taylor_with_budget` | `limit_via_series` | `limit_with_error` |
|---|---|---|---|---|
| HEAD | `BudgetExceeded` (100 000 104 units) | `BudgetExceeded` | `Finite(0)` | `Finite(0)`, with a bound |
| fix | Ok, precision 31 | Ok | `Finite(0)` | `Finite(0)`, with a bound |
| C-1 approved | Ok, precision 31 | Ok | `Finite(0)` | `Finite(0)`, with a bound |
| taint build | as HEAD (its charges), 0 tainted products, 0 reads | | | |

These are from the probe `p.rs`, whose `limit_with_error` is at order 5. With the entry-point
driver `c1mcheck/drv/ep.rs` (order 30 for every entry point, REPL budget, 8 MB thread, `build/bin/ep_*`)
HEAD's `limit_with_error` is `Finite(0)` without a bound, as the item says; the fix's and the C-1
approved state's are `Finite(0)` with a bound, and their taylor (21 824 characters), series,
`to_expr`, `Display`, `Debug` and re-parse agree (0.35 s, 34 MB in release). The fix's `p.rs` row
equals the C-1 approved state's in every field (`t/flip.fix.gz`, `t/flip.pm.gz`). Its 15
stored terms (odd exponents 1 to 29) equal the closed form C (-1)^m (1^j + ... + 485^j)/j!,
C = log(1+I) + ... + log(20+I), to a relative error of 7.0e-135 (gp, realprecision 120).

Variants at the same budget (`t/fv.tsv.gz`, outputs `t/fv.*.gz`; HEAD, fix, C-1 approved):

| input | HEAD | fix | C-1 approved |
|---|---|---|---|
| v1 `log(2 + FLIP)` at 0, order 30 (every product under log) | `BudgetExceeded` | `BudgetExceeded`, same units | `BudgetExceeded` |
| v2 `FLIP + log(-1 + log(-1)^2*sin(x))`, order 30 | `BudgetExceeded` | value, precision 31 | `NotLaurent` (cannot decide) |
| v3 the same, order 3 | value | value, identical | `NotLaurent` (cannot decide) |
| v4 `exp(FLIP) + sqrt(-1 + I*log(-1)^2*sin(x))`, order 5 | `BudgetExceeded` | `NotLaurent` (leaves the cut) | `NotLaurent` (leaves the cut) |

v1 shows that products under a branch function keep HEAD's enclosure and charges. v2's value equals
gp's series of C (sin x + ... + sin 485x) + I pi + log(1 + pi^2 sin x) at every exponent 0 to 30
(relative error 1.8e-134). v4 is O1 below.

## 3. Bar (b): regressions

Against HEAD: no value changes and no value becomes a refusal on any of the 107 527 rows above; the
only outcomes that differ from HEAD are `BudgetExceeded` rows that now finish (scflip, v2, v4).
Against HEAD-before-S1 (8a0b310): the fix differs from HEAD only by such rows, so it adds no
regression to those that the S1 reviews measured against 8a0b310.

**O1 (observation, not a blocker).** A `BudgetExceeded` at HEAD can now become a decided refusal:
v4 is `NotLaurent` ("a non-integer power: the argument leaves a branch cut at the point"). It is
right: the argument -1 - i pi^2 sin x has its constant term on the cut of sqrt and an imaginary
coefficient, so sqrt is near -i for x > 0 and near +i for x < 0 (gp), and the C-1 approved state
gives the same refusal. The m1 item names only moves from `BudgetExceeded` to a value; this is the
same mechanism (fewer charges) reaching a refusal instead.

## 4. Tests

`cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg.log`,
1 min 46 s): unit 686 (5 ignored), `s1_cuts` 16 (2 ignored), `s1_depth` 14, `s1_finish1` 11 (1),
`s1_fuzz_data` 7, `s1_payload` 2 (1), `s1_regressions` 26 (2), `s1_review2` 2, `s1_review3` 7 (7),
`s1_review4` 4 (2), `s1_review4_memory` 1 (1), `s1_review5` 4 (2), `s1_review5_memory` 2 (2),
`s1_review6` 4, `s1_singular_points` 9, doc 116 (11): **911 passed, 0 failed, 36 ignored**.

The test diff adds three tests, three helpers and rows `enc:1` to `enc:12`; no line of the tests or
data is removed or changed. The two-sided oracle (`python3 -I -B s1_gp.py --two-sided` on copies of
the three data files, `oracle.log`) exits 0: cuts 64 value outcomes, 0 fail, 37 refusal rows (27
with the formal series wrong on a side, `enc:11` and `enc:12` among them); singular points 367, 0
fail; fuzz 394, 0 fail.

## 5. Minor items (doc precision, no outcome affected)

- n1. `EFrame::checks_a_cut` (`series_engine.rs:2292` to `:2298`): "a tower with an integer exponent
  is counted too, which only computes a class that no check may read" reads as if that class could
  never be read; what is meant is that counting the tower only adds enclosures whose classes its own
  step does not check. A tower with a constant base and a series exponent (`pow_const_base`, no cut
  check: log of a constant) is also counted and is not mentioned.
- n2. The same doc and `finish_product`'s say that only a cut check reads a class; `pow_general`'s
  real-class test (which may charge `real_sign`) and `const_base_class` also read classes, to compute
  a class. Both run under a tower record, which is counted, so the rule holds; the sentence could
  say "only a cut check decides from a class".

## 6. Files and housekeeping

Scripts and inputs: `mktaint.py`, `gen_enc.py`, `cmp.py`, `buildlib.sh`, `builddrv.sh`,
`run_corp.sh`, `drv/` (`p.rs`, `pt.rs`, `dag.rs`, copies of `ep.rs` and `mb.rs`), `t/` (the
enclosure corpus `enc.tsv` and its outputs `enc.*.gz`, `flip.*.gz`, `fv.*.gz`, `dag.*`, `v2.T`,
`flip.T`, `enc.newvals.ids`), `gp/` (`snaphead.gp`, `snapmk.py`, the check lines `*.chk*` and the
results `*.res`; a gp script is `c1mcheck/par/head.gp` (with its `PW` line replaced by
`snaphead.gp` for the snapped runs) followed by the check lines), `out/` (corpus outputs),
`oracle.log`, `sym_dbg.log`, `rlibs_rel.txt`.

Removed at the end: `build/` (the copy of the release dependencies, the four rlibs, the drivers'
sources and executables), the source trees `src/` (`mktaint.py` and `git archive` rebuild them), and
the concatenated gp scripts. No executable is left in the scratchpad.

`target/release` (219 MB) was created by my `cargo build --release -p rustmath-symbolic` at 16:52;
the concurrent resources check then built release tests in it (rustmath-complex and the symbolic
test binaries, 17:06 to 17:07). No process was using it at the end of this check, but I left it in
place so as not to remove binaries that check may still run; it should be deleted when that check
ends. Disk: 6.0 GB free at the end.

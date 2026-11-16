# S1-symbolic: design (crate rustmath-symbolic)

**Prototype of the current revision (revision 7rt4; attackers and reviewers measure this one).**
`EXP/rev7rt4/proto13/`, with `EXP` = `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/S1-symbolic`.
It is revision 8's `proto12` plus the changes of "Revision 7, red-team round 4" below: the precision out
of band (`proto13/ser.rs::Prec`, `Abs(i64)` or `Exact`, with `covers`, `add` and `at_most`; every
`prec` of `Ser`, `mul`'s cutoff, `inv`, `pow_int` items 2 and 5, `trunc`, `integrate`, the driver and the
flag `PREC_CLAMPED`, which stops the driver after a clamped attempt that did not raise the
precision; series H1), saturating units (`proto13/konst.rs::Work::charge`, the `q_*` and
`i_*` charges) and the slot cap of `ser.rs::Ctx::alloc` (`MAX_SERIES_SLOTS`; series H2), the poles of a
rational argument in `konst.rs::kapply_src` (numeric m1), exact literal balls and the exact-zero rules
in `proto13/ball.rs` (`ball_expr` leaves, `Sqrt`, `big_int_pow`, the real power; numeric m3) with
`exp(k log a)` as an integer power (`exp_int_pow`, `int_pow_ball`; found while revising), the tree
of a computed kernel per kind (`konst.rs::mk_kernel_inner`) and the margin of the hint test
(`ser.rs::maybe_hint`, `hint_exp_arg`; structure m1), and the entry `api::limit_with_error` with the
driver mode `NO_TERMS` (memory m2). The public precision of `SeriesExpansion` is T = order + 1 always
(`api.rs`). Build script: `EXP/rev7rt4/build.sh release proto13/main.rs proto13` writes
`EXP/rev7rt4/bin/proto13_rel`; `XFLAGS="--cfg memcount" EXP/rev7rt4/build.sh release proto13/main.rs
proto13m` writes the counting build `bin/proto13m_rel` (heap peak); `debug` in place of `release` writes
the debug build `bin/proto13`. Each build takes a few seconds against the HEAD rlibs listed in
`EXP/rev5/rlibs_rel.txt` and `rlibs_dbg.txt` (rustmath-symbolic and its dependencies are unchanged since
80130e7; if a listed path is missing, run `cargo build [--release] -p rustmath-symbolic -p
rustmath-reals` and regenerate the list from `cargo build --message-format=json`).
`EXP/rev7rt4/tools/runall_rt4.sh` rebuilds every binary of the list below. Run every command from
`EXP/rev7rt4`. Every call of the driver goes through the public layer of §2.9 (`proto13/api.rs`:
`SeriesExpansion` with shared coefficients, `series_expansion`, `to_expr`, `taylor_with_budget`,
`limit_via_series`, `limit_with_error`). One call, timed, with its units, VmHWM and heap peak (malloc
chunks above the heap before the call):

- `taylor_with_budget`: `python3 tools/run7.py bin/proto13m_rel '<f>' <point> <order> [budget]` (budget
  1e8 by default; `@<path>` in place of `<f>` reads the input from a file), or directly
  `BUDGET=100000000 FASTEXIT=1 bin/proto13m_rel measure '<f>' <point> <order>` (without `BUDGET` it is the
  unbounded `taylor`). The line gives `work` (units), `kernels`, `hints` (rendering hints set, §2.5),
  `seeds`, `VmHWM` and, with the counting build, `heap_base` and `peak`.
- `series_expansion`, then `to_expr()`: the same with `ENTRY=se` in the environment; the output line
  adds `after_se [time VmHWM live peak]`, the state when `series_expansion` has returned.
- the `limit_via_series` path: `ENTRY=lim` (order 5 and its own budget `LIMIT_SERIES_WORK = 1e8`; the
  order and the budget of the command are ignored; the driver renders the first stored term only); the
  line ends `result Finite(..)` or `result Unknown (<engine error>)`. `ENTRY=lwe` (revision 7rt4) is
  `limit_with_error`: the limit path, then for a finite limit the expansion to the command's order with
  no term rendered; the line ends `result .. bigo Some(O(h^k))` or `bigo None (<engine error>)`.
- `TREE_TOO=1` prints the result's size as a tree (`tree`, counted with a memo by node address) and its
  distinct nodes; `TREECAP=<n>` replaces `MAX_RESULT_TREE` (experiments only). `STACK=<bytes>` runs the
  whole call on a thread of that size (`STACK=2097152` with `bin/proto13` is the 2 MB debug thread);
  `DISPLAY_TOO=1` adds the result's depth and its Display length, with HEAD's recursive Display by
  default (the stricter check, which the depth scripts keep) and with the work-list writer of §2.11 when
  `DISPLAY=iter`; `BIGLIT=<digits>` replaces the symbol L by that integer (the prototype links HEAD's
  parser, which reads i64 literals only, reads `asinh` as a named function and cannot parse `1/x`; write
  `(1)/x`). `bin/proto13_rel series|coeffs '<f>' <point> <order>` prints the stored terms (`series`: the
  public precision and, since revision 7rt4, the internal one, `internal Exact` or `internal Abs(p)`;
  `coeffs`: one `exponent<TAB>coefficient` line per term, with the rules of §2.11 when
  `DISPLAY=iter`), and `bin/proto13_rel ball '<constant>' <p>` the enclosure at p bits or its reason
  `None(Never)`, `None(TooWide)` (the midpoint is printed as an f64, so a value above 1.8e308 prints
  `inf`). `bin/powtime_rel '<constant>' <p> [reps]` times the enclosure against its a-priori charge;
  `rt4num/bin/rt4_rel rate|scan|ball|balls FILE P [REPS]` (red team 4's numeric probe, linked against
  proto13; `rt4num/build.sh release src/rt4.rs rt4`) times one node against its charge.
- Parse and call on one thread, or each alone: `bin/split_rel MODE FILE POINT ORDER STACK` (HEAD's
  parser), MODE `both` (parse and `taylor_with_budget` on one STACK thread), `parse`, `engine` (parse
  on a 1 GiB thread, the call on a STACK thread), `head` and `headboth` (HEAD's `Expr::taylor`);
  `ENTRY=se|lim` as above; `DISPLAY_TOO`, `LATEX_TOO`, `EQ_TOO` (PartialEq against a clone) and
  `CHECKSAME` (the work-list writers against HEAD's) on the same thread. `bin/rt2_rel` has the same
  modes behind a parser that follows §2.12 (red team 2's probe, linked against proto13 as
  `EXP/rev7rt4/probes/rt2.rs`; `PRINT`, `GPCHECK`, `EQ_TOO`, and `ROUNDTRIP=1|gp`: the input and the
  result written with §2.11's rules and read back by the §2.12 parser), and the mode
  `bin/rt2_rel ball FILE P 0 0` (the enclosure of the constant in FILE at P bits, with asinh, acosh and
  atanh as unary functions); `bin/rt2t_rel` (revision 7rt4, red team 4's probe) is `rt2_rel` with
  `TREE_TOO` and `TREECAP`. `python3 checks/gpres.py bin/rt2_rel FILE POINT ORDER` compares every
  coefficient of the result up to the order with gp's series of the input (300 digits, gp with a 1 GB
  stack), and `python3 checks/c0check.py bin/rt2_rel FILE POINT ORDER REF0 [REF1 ..]` evaluates the
  result's coefficients in gp against gp references and flags `sign(0)`, `/ 0` and `(1 - 1)` in its
  text. `python3 ../redteam4-structure/tools/treeest.py BIN FILE POINT ORDER` bisects `TREECAP` for the
  driver's `render_tree` and prints it with the exact tree; `python3 tools/treesafe_rt4.py FILES..`
  (revision 7rt4) checks on red team 3's random inputs that `render_tree` is never below the tree and
  compares the outcome kinds with revision 8's; `python3 tools/maxfuzz_rt4.py SEED N` (revision 7rt4)
  checks random products near 2^63 against the same inputs shifted down.
  Inputs built with the `Expr` API in the structure §2.12's parser gives them: `bin/apishape_rel SHAPE N
  STACK` (shapes `divl`, `divlc`, `divlx`, `muldiv`, `divsin`, `powneg`, `tow1`, `towneg2`, `xtow2`,
  `towx`, `towx2`, `towsin`; `POINT`, `ORDER`, `ENTRY`, `DISPLAY_TOO`). The REPL taylor line (J1b's
  `display_size`, then the text and the LaTeX): `FILE=<input> LINEAR=iter bin/replshow_rel @ <point>
  <order>` on a 64 MiB thread (`FCHAIN=<n>` builds `(f(f(...f(1)...)))/(1-x)` with the API instead;
  without `LINEAR`, HEAD's `expr_to_latex`). A REPL line that hands a result to a consumer (red team 3's
  probe): `bin/consumers_rel design|head FILE POINT ORDER simplify|subs|eq|none` on a 64 MiB thread. The
  Drop on a mirror enum with HEAD's field types: `bin/dropm SHAPE N STACK design|glue|tls` (`twin`,
  `sib`, `chain`, `left`, `dagsum`, or `file:PATH:POINT:ORDER` for a result; `tls` drops the value in the
  thread's TLS teardown after `LEAF`, and `OLDLEAF=1` uses revision 7rt3's `LEAF.with`).
  Without `_rel` each is the debug build. `python3 tools/bsearch.py BIN MODE SHAPE STACK LO HI`
  (generators in `tools/shapes.py`), `tools/bsapi.py BIN SHAPE STACK LO HI` and `tools/mixdepth_rt2.py`
  (the mixed shapes of red team 2, generators in `EXP/redteam2-structure/gen.py`) find the deepest input
  handled; `python3 tools/timed.py CMD..` times one run with its maximum RSS (4 GiB cap, 60 s);
  `python3 checks/coefcheck.py BIN '<f>' '<gp f>' <point> <order>` compares every stored coefficient
  with gp's series.
- `tools/runall_rt4.sh` (revision 7rt4) rebuilds every binary, reruns every check of revision 8 on
  `proto13` (the oracles, with `checks/revchecks_rt4.py`, a copy of `EXP/revchecks.py` whose exact-zero
  line reads the public precision T and `internal Exact` in place of the sentinel's value; the
  calibration and memory families, the self-attack, the depth searches, the inputs of reviews 3 to 8,
  the scans, `tools/redteam1_rt2.sh`, `tools/redteam2_rt2.sh`, `tools/redteam3_rt3.sh`,
  `tools/cons3.sh`, `tools/hintscan.py`, `tools/review8_r8.sh` and `tools/randhint_rt4.py`, review 8's
  random hinted towers against gp with revision 8's and revision 7rt2's outcomes beside each), then the
  reproductions of red team 4: `EXP/redteam4-series/tools/repro.sh` with proto13's binaries (output
  `rt4series_repro.txt`), `rt4num/repro.sh` (the numeric report's, output `rt4num_repro.txt`),
  `tools/rt4_ms.sh` (the memory and structure reports', output `rt4_ms.txt`) and `tools/rt4_extra.sh`
  (the class checks of this revision: the `exact_sentinel` rows, buffers from 2^57 to 2^63 slots
  through every recurrence, `constant_poles`, the exact-zero balls and `maxfuzz_rt4.py`, output
  `rt4_extra.txt`). About 25 minutes on an idle machine; the binaries stay in `bin/` (about 480 MB).
  The outputs carry the suffix `_rt4` (`gpcheck3_rt4.out`, `calibrt4_final.txt`, `redteam3_rt4.txt`,
  ...) and are in `EXP/rev7rt4/`. Revision 8's prototype, scripts and outputs stay in `EXP/rev8/`
  (`proto12`, `tools/runall_r8.sh`, `tools/seeds_r8.sh`, `review8_r8.txt`, `randhint_r8.txt`), revision
  7rt3's in `EXP/rev7rt3/` (`proto11`), revision 7rt2's in `EXP/rev7rt2/` (`proto10`), revision 7rt1's in
  `EXP/rev7rt1/` (`proto9`), revision 7's in `EXP/rev7/` (`proto8`).

Unit items (docs/HANDOFF.md "Open follow-ups", REPL split-outs of the X2 design):

- **F1.** `Expr::taylor` (`rustmath-symbolic/src/series.rs:24`) differentiates without simplifying, so memory
  grows exponentially with the order, and a debug build overflows the stack at order 20000
  (`TAYLOR_ORDER`, `rustmath-jupyter/src/repl.rs:11087` at HEAD 92906d4). No REPL bound prevents it. Fix the algorithm
  and the deep recursion (simplify and Drop of left-deep sums).
- **S7 (symbolic part).** The expansion point in `series.rs:34`, and the i64-only literal parser
  (`rustmath-symbolic/src/parser.rs:148-154`).
- Measure before/after for x^2, exp(x), sin(x), 2*x+1 at several orders; reference series from gp.
- **expand under a minus** (revision 6: added to the S1 bullet of `docs/HANDOFF.md` by the tracker
  commit `0c2f368`, after review 6; tracker row "Polynomial expansion", 🔴): `distribute_negation`
  (`rustmath-symbolic/src/expand.rs:250-263`) negates one term in each arm, so `expand(-(x - 1))`
  gives `x - 1`. §1.7, §2.17.

Revisions 0 to 2 were written against c79a582. Revision 3 re-checked every fact against HEAD dc367a2,
which adds units I3, I1, I2, P1, F1, R1, M1a and M1b. Two of them change this design's inputs:
`Integer::gcd` is now Lehmer's algorithm (I3, `rustmath-integers/src/lehmer.rs`, 10 to 60 times
faster than the Euclid that review 3 measured), and `Integer::exact_root(k)` exists (I3,
`integer.rs:458`, power-residue sieve and 2-adic root). The line numbers below are at HEAD dc367a2
where revision 3 re-checked them (series.rs, limits.rs, simplify.rs, repl.rs, plane_curve_ops.rs,
integrate_exact.rs); the E0509 test sites of §3.3 are at c79a582 and are found again by compiling.
Revision 4 was written against the same HEAD dc367a2. Revision 5 re-checked every fact against HEAD
92906d4, which adds C1, J1a and J1b (`git log dc367a2..HEAD`). rustmath-symbolic, rustmath-integers,
rustmath-reals, rustmath-calculus and rustmath-geometry are unchanged since dc367a2, so the line
numbers in them stand. rustmath-rationals changed only `Rational::to_f64`, which the engine does not
call. J1a and J1b rewrote much of `rustmath-jupyter/src/repl.rs`: the taylor arm is now `:2118-2145`
and `TAYLOR_ORDER` is at `:11087` (§3.1); the REPL evaluates on a thread of 64 MiB
(`rustmath-jupyter/src/kernel.rs:45`, `EVAL_STACK_BYTES`); and J1b's display budget is in
(`display_size`, `repl.rs:11569`, an explicit stack that stops past `OUTPUT_BYTES`, §3.4).

Experiments: `/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/S1-symbolic/`
(below: `EXP/`). The working prototype is `EXP/rev7rt4/proto13/` (the block at the top; its scripts are
in `EXP/rev7rt4/tools/`, its outputs in `EXP/rev7rt4/`, its own inputs in `EXP/rev7rt4/inputs4/`, red
team 4's inputs in `EXP/redteam4-*/inputs/`; `EXP/rev7rt4/inputs` is a link to `EXP/rev7rt2/inputs`,
`EXP/rev7rt4/inputs3` to `EXP/rev7rt3/inputs3`, and `inputs8` a copy of revision 8's). The revision 8
prototype is `EXP/rev8/proto12/` (its scripts in `EXP/rev8/tools/`, its outputs in `EXP/rev8/`, its own
inputs in `EXP/rev8/inputs8/`, review 8's inputs in `EXP/review8/inputs/`). The revision 7rt3 prototype is `EXP/rev7rt3/proto11/`
(its scripts in `EXP/rev7rt3/tools/`, its outputs in `EXP/rev7rt3/`, its own inputs in
`EXP/rev7rt3/inputs3/`, red team 3's inputs in `EXP/redteam3-*/inputs/`). The
revision 7rt2 prototype is `EXP/rev7rt2/proto10/` (its scripts in `EXP/rev7rt2/tools/`, its outputs in
`EXP/rev7rt2/`, red team 2's inputs in `EXP/rev7rt2/inputs/series/`, `structure2/`, `memory2/` and
`numeric2/`). The revision 7rt1 prototype is
`EXP/rev7rt1/proto9/` (its scripts in `EXP/rev7rt1/tools/`, its outputs in `EXP/rev7rt1/`, red team 1's
inputs in `EXP/rev7rt1/inputs/structure/` and `inputs/memory/`). The revision 7 prototype is `EXP/rev7/proto8/`
(the revision 6 prototype plus the erf rule, the result floor of the charged arithmetic and the public
layer `api.rs`; its scripts are in `EXP/rev7/tools/`, its outputs in `EXP/rev7/`). The revision 6
prototype is `EXP/rev6/proto7/` (revision 6: `konst.rs`,
`ser.rs`, `ball.rs`, `main.rs`, the counting allocator `memcount.rs` and the probe drivers of
revision 5, `dag_input.rs` with two more shapes), built with `EXP/rev6/build.sh debug|release SRC
NAME` (`XFLAGS="--cfg memcount"` for the counting build) against the HEAD 92906d4 rlibs of
`EXP/rev5/rlibs_*.txt`, binaries in `EXP/rev6/bin/` (deleted after the runs to save disk; `build.sh`
rebuilds each in seconds, and `runall6.sh` starts by rebuilding them). Its scripts are in `EXP/rev6/tools/`;
`runall6.sh` runs all of them and the oracles (gp, revchecks, `budget6.py`), outputs in
`EXP/rev6/` (§5.3). Revision 6 re-checked every fact against HEAD 0c2f368 (92906d4 plus a tracker
commit that changes docs only): nothing that S1 reads has changed since revision 5, and HANDOFF's S1
bullet gained "expand under a minus" (§1.7). Revision 7 re-checked every fact against HEAD f29c492
(`git log 0c2f368..HEAD`: MF1). It changes rustmath-manifolds and two comments of
`rustmath-jupyter/src/repl.rs` (`:4903`, `:12327`, the same number of lines), so every line number
cited here stands; rustmath-symbolic and the crates it reads are unchanged. The tracker commit
2e15d1c that followed changes `PROJECT_TRACKER.md` and `docs/` only; it adds to HANDOFF's S1 bullet a
note from MF1 (§3.4) and moves the notes of §1.8 to `docs/HANDOFF.md:461-463`. The revision 5 prototype is `EXP/rev5/proto6/` (revision 5: `konst.rs`,
`ser.rs`, `ball.rs`, `main.rs`, and the probe drivers `head.rs` (HEAD's `taylor` on the same input),
`longmono.rs`, `dag_input.rs`, `enginedepth.rs` and `enginedepth_head.rs`), built with
`EXP/rev5/build.sh debug|release SRC NAME` against the HEAD 92906d4 rlibs (`EXP/rev5/rlibs_dbg.txt`,
`rlibs_rel.txt`; binaries in `EXP/rev5/bin/`, deleted after the runs and rebuilt by `build.sh` in
seconds after `cargo build [--release] -p rustmath-symbolic -p rustmath-reals`). The driver reads
`FILE=<path>` (the input from a file), `STACK=<bytes>` (the whole call on a thread of that size),
`DISPLAY_TOO=1` (the result's depth and `format!` length) and `BIGLIT=<digits>` (the symbol L becomes
that literal; the prototype links HEAD's parser, which reads i64 literals only). Revision 5's scripts
are in `EXP/rev5/tools/`; `runall5.sh` runs all of them: `attack5.py` (the self-attack of §5.3,
`EXP/rev5/attack5_final.txt`), `calib5.py` (§5.3, `calib5_final.txt`), `mem5.py` (bytes per unit,
`mem5_final.txt`), `depthcmp.py` (the deepest input handled, the prototype against HEAD,
`depthcmp_rel_final.txt`, `depthcmp_dbg_final.txt`), `deep2.py`, `nkx.py`, `leftdeep.py`, `b2.py`,
`newtests_dbg.py` (the new tests of §4.1 in debug) and `rev4cmp.py` (the revision 4 prototype on the
families revision 5 found, `EXP/rev5/rev4cmp.txt`, binary `bin/proto5_rel` from `EXP/rev5/rev4cmp/`).
On the final prototype the gp oracle `EXP/rev2/gpcheck3.py` gives 107 OK and 0 failures
(`EXP/rev5/gpcheck3_r5.out`), `EXP/revchecks.py` and `EXP/rev2/revchecks3.py` give 0 bad
(`EXP/rev5/revchecks_r5.out`, `revchecks3_r5.out`), and `EXP/rev4/tools/budget4.py` gives all 31
budget outcomes, the slowest in 5.5 s debug (`EXP/rev5/budget4_debug_r5.txt`). The revision 4
prototype is `EXP/rev4/proto5/` (`konst.rs`,
`ser.rs`, `ball.rs`, `main.rs`, built with `EXP/rev4/build.sh debug|release proto5/main.rs proto5`,
release rlibs in `EXP/rev4/rlibs_rel.txt`, debug rlibs in `EXP/rev3/rlibs_dbg.txt`; binaries
`EXP/rev4/bin/proto5` and `proto5_rel`, deleted to save disk and rebuilt by `build.sh` in a few
seconds as described below; with `FASTEXIT=1` the driver prints the engine result and exits, without
the tree walks of its `measure` report). The revision 4 scripts are in
`EXP/rev4/tools/`: `budget4.py` (every budget test of §4.1, output `EXP/rev4/budget4_debug.txt`),
`repro_r4.py` (review 4's inputs, outputs `EXP/rev4/repro_r4_*.txt`), `calib4.py` (§5.3, outputs
`EXP/rev4/calib4_run1.txt` and `calib4_run2.txt`), `mem_r4.py` (memory per unit, `EXP/rev4/mem_r4_a.txt`), `minors_r4.py`
(`EXP/rev4/minors_r4.txt`) and the depth probes (`EXP/rev4/depth_r4.txt`). On it the gp oracle
`EXP/rev2/gpcheck3.py` gives 107 OK and 0 failures (`EXP/rev4/gpcheck3_r4.out`), and
`EXP/revchecks.py` and `EXP/rev2/revchecks3.py` give 0 bad (`EXP/rev4/revchecks_r4.out`,
`revchecks3_r4.out`). The revision 3 prototype is `EXP/rev3/proto4/` (`konst.rs`, `ser.rs`,
`ball.rs`, `main.rs`, built with `EXP/rev3/build.sh debug|release`, rustc against the HEAD dc367a2
workspace rlibs listed in `EXP/rev3/rlibs_dbg.txt` and `rlibs_rel.txt`; binaries `EXP/rev3/bin/proto4`
(debug) and `EXP/rev3/bin/proto4_rel`). `EXP/rev3/proto3_head/` is the revision 2 prototype
unchanged, built against HEAD (`EXP/rev3/bin/proto3_head_rel`, `build.sh release proto3_head/main.rs
proto3_head`) for before/after runs at HEAD. The binaries were deleted to save disk; `build.sh`
rebuilds each in a few seconds after `cargo build -p rustmath-symbolic -p rustmath-reals` (and
`--release` for the release ones, whose `target/release` is deleted after the timing runs; the
rlib paths are regenerated from `cargo build --message-format=json`, as `rlibs_*.txt` were).
`EXP/rev2/proto3/` (revision 2), `EXP/proto2/` (revision 1) and `EXP/proto/` (revision 0) are kept for
comparison. The gp cross-check harness is `EXP/rev2/gpcheck3.py` (107 cases; on the revision 3
prototype 0 failures, output `EXP/rev3/gpcheck3_r3.out`; it contains the 82 cases of
`EXP/gpcheck2.py`). Expected errors, exact structures and budget timings are checked by
`EXP/revchecks.py` (revision 1 checks, output on the revision 3 prototype `EXP/rev3/revchecks_r3.out`,
0 bad) and `EXP/rev2/revchecks3.py` (revision 2 checks, output `EXP/rev3/revchecks3_r3.out`, 0 bad).
Revision 3 measurements: `EXP/rev3/micro*.txt` (big-integer operation costs at HEAD),
`EXP/rev3/calib3_final.txt` (the §5.3 calibration, `EXP/rev3/calib3.py`) and
`EXP/rev3/budget_debug.txt` (the debug budget tests). The iterative Drop experiments are in
`EXP/rev1/`. `EXP/rev2/run.py` times one prototype call (`python3 EXP/rev2/run.py BIN measure f a n
[budget]`).

## Revision 7, red-team round 4

Reports: `fu2/design/S1-symbolic.redteam4-memory.md`, `-numeric.md`, `-series.md` and `-structure.md`
(four lenses on the revision 8 prototype `EXP/rev8/proto12/`). Two holes are over the review bar, both
in the series lens; seven are below it. Each is fixed in the body below, or, where the fix is not cheap,
recorded with its measured cost (§7 items 8 and 10), and each fix closes the class of its hole, not only
the measured inputs. The prototype is now `EXP/rev7rt4/proto13/` (the block at the top). Every
reproduction of the four reports was rerun on it (`EXP/redteam4-series/tools/repro.sh` with proto13's
binaries, `EXP/rev7rt4/rt4num/repro.sh`, `tools/rt4_ms.sh`, outputs `rt4series_repro.txt`,
`rt4num_repro.txt`, `rt4_ms.txt`), with the class checks of `tools/rt4_extra.sh` (`rt4_extra.txt`) and
every check of revision 8 (`tools/runall_rt4.sh`, §5.3). The facts were rechecked at HEAD 6e03987:
rustmath-symbolic and the crates it reads are unchanged. Numbers are release on an 8 MB thread, or debug
on a 2 MB thread, budget 1e8 (`LIMIT_SERIES_WORK` on the limit paths), unless stated.

| hole | change | where | evidence (proto13) |
|---|---|---|---|
| series H1 (over the bar, a wrong decided result): the sentinel `EXACT = i64::MAX` was also a legal exponent, a legal finite precision and a legal working precision. Five producers conflated them: `pow_int` item 2 stored an exact term at `n v = i64::MAX`, which `add` dropped while keeping the sum exact; `mul` with the cutoff `EXACT` dropped a pair summing to i64::MAX and returned `EXACT`; `padd` returned the finite sum i64::MAX; `pow_int` item 5 returned the precision `n vb + 1 = i64::MAX`; the loss-driven step set P = i64::MAX. `(x^(2^62)*x^(2^62-1))/x^(2^63-1)` at 0 (32 bytes) was a decided 0 for the value 1 (the limit path `Finite(0)`), `(x^9223372036854775807 + 1)/x^9223372036854775807` lost its `+ 1`, `(x^(2^63-4)*exp(x))/x^(2^63-2)` at order 2 claimed `1/x^2 + 1/x + 1/2` exact, and the limit path gave `Finite(0)` for the limit 2^63 - 2 of `(((x+x^2)^9223372036854775806)/x^9223372036854775806 - 1)/x`; each in under 4 ms, the same on 2 MB debug | The class is "a sentinel that is also a value". The precision is out of band: `enum Prec { Abs(i64), Exact }` (ordered so that the minimum is a sum's precision), every i64 a legal exponent and a legal finite precision. Every precision sum goes through `Prec::add`: `Exact` absorbs, a finite sum that overflows upwards is `Abs(i64::MAX)` (a weaker true claim; `shift` puts the terms at and above it in the O-term), one that overflows downwards is the overflow error. `mul`'s cutoff is a `Prec`: an `Exact` cutoff drops no pair, and `Exact` survives only when no pair was dropped. `pow_int` item 2 stores the exact term at i64::MAX; item 5 checks `n vb + 1` and returns `O(h^(n vb))` at i64::MAX; `inv`'s `-vb = i64::MAX` is legal; `trunc` and `integrate` check `e + 1`. P = i64::MAX is a legal working precision (its slot counts are charged with saturating units, H2). A loss after an attempt that clamped a precision and did not raise it cannot be repaired by P: `Unsupported("a series precision overflows i64")`. The public precision is T (`SeriesExpansion::precision`), so no public value is both an exponent and a sentinel. The other sentinels of the engine were audited: saturated sizes (u64::MAX) only refuse; the fault kernel's id u64::MAX is never an id (at most 2^16 kernels) | §0 item 1, §2.1 (the `Prec` block, "Why out of band", the truncation rules), §2.4 (`mul`, `inv`, `pow_int` items 2 and 5), §2.6, §2.9 (`precision`), §2.16 (`mul`, `pow_int`), §4.1 `exact_sentinel`, §6 step 4 | every row of the report is right or an honest refusal, through `taylor_with_budget`, `series_expansion`, the limit path, the §2.12 probe and on a 2 MB debug thread, each in under 0.3 ms and 4.5 MB: `(x^(2^62)*x^(2^62-1))/x^(2^63-1)` and `(x^9223372036854775806*x)/x^9223372036854775807` are `1` (the limit path `Finite(1)`), `(x^9223372036854775807 + 1)/x^9223372036854775807` is `1/x^9223372036854775807 + 1`, `((1)/x^(2^62)*(1)/x^(2^62-1))*x^(2^63-1)` at oo is `1`, and the control at i64::MAX - 1 stays `1` (gp: 1 and the two terms); `((1)/(1-x^9223372036854775807) - 1)/x^9223372036854775807` is `PrecisionExhausted(1)`, the two products that need the exponent 2^63 `PrecisionExhausted`, `(x^(2^63-4)*exp(x))/x^(2^63-2)` and `(x^(2^63-4)*((1)/(1-x)))/x^(2^63-2)` `Unsupported("a series precision overflows i64")` after two attempts, the power rows `PrecisionExhausted` (their value needs P above i64::MAX), and the limit path `Unknown` for each; random products and quotients near 2^63 (`tools/maxfuzz_rt4.py`, 900 inputs in six seeds): 900 inputs in six seeds, 222 `Ok`, each equal to the same input with its exponents moved down by one shift, 678 honest refusals (`Unsupported`, `PrecisionExhausted`, `BudgetExceeded`), nothing else; the gp oracle 107 OK, the revision checks 0 bad (`rt4series_repro.txt`, `rt4_extra.txt`) |
| series H2 (over the bar, a release crash on inputs HEAD answers): the units of a dense buffer of 2^60 slots or more saturated to u64::MAX in `alloc`, and the base of `Work::charge` (`16 + w`) wrapped them to 15, so the budget admitted the buffer and `Vec::with_capacity` panicked ("capacity overflow"; debug: "attempt to add with overflow" in `Work::charge`). `exp(x)/x^(2^61)` (15 bytes), `sin(x)/x^1152921504606846976`, `(1+x)^(1/3)/x^1152921504606846976`, `((1)/x^100000000000000000 + 1)^65`, `(1)/(x^2305843009213693952+x^2305843009213693953)` and `exp(x)` through `series_expansion` at order 2^60 - 1 panicked after 0.06 to 0.09 s through every entry point; HEAD answers each in 6 to 13 us | The class is "a units formula that can wrap". §2.7 states the rule: every units formula saturates, and `Work::charge` adds its base with `saturating_add` and compares the saturated sum; the plain sums left combine sizes of objects in memory with constant factors. `alloc` charges `16 n` saturating before the cast, so a budget refuses a buffer above `remaining / 16` slots before any allocation, and without a budget a buffer above `MAX_SERIES_SLOTS = 2^57` is `Unsupported`, so no slot count from the input reaches `Vec::with_capacity` beyond what it can hold. Every dense buffer of the engine is allocated through `alloc` (audited: the other `with_capacity` calls take lengths of vectors in memory, a pair count under the budget, or a constant) | §2.7 (`Work::charge`, the row "dense buffer of n slots", "Units arithmetic"), §2.15 (`MAX_SERIES_SLOTS`), §2.16 (`Work::charge`, `alloc`), §4.1 `dense_buffer_units`, §6 steps 3 and 4 | every row of the report is `BudgetExceeded` (the units saturated at u64::MAX) in under 0.3 ms through `taylor_with_budget` and `series_expansion`, the limit path `Unknown (BudgetExceeded)` (the 57-byte row `PrecisionExhausted(6)`, as before), and the same on a 2 MB debug thread, no panic; every recurrence (exp, sin, cosh, tan, log, atan, erf, a rational power, Miller and binomial powers, `inv`, a product) divided by x^(2^k) for k = 57 to 62, on 8 MB release and 2 MB debug: `BudgetExceeded`; `exp(x)` through `series_expansion` at orders 2^57 - 1 to 2^63 - 2: `BudgetExceeded`; without a budget, `exp(x)/x^(2^60)`, `sin(x)/x^(2^61)`, `(1)/(x^(2^61)+x^(2^61+1))` and the `^65` row: `Unsupported("a series buffer of more than 2^57 slots")` in under 0.1 ms (`rt4series_repro.txt`, `rt4_extra.txt`) |
| numeric m1 (below the bar): arctanh(1), arctanh(-1) and zeta(1) were kernels in the prototype, not the `DivisionByZero` that §2.2 prescribes: `(1)/(x + atanh(1))` inverted the pole, `x*atanh(1)` was `Finite(0)` on the limit path (HEAD the same) | the rational branch of `apply` returns `DivisionByZero` for arctanh at +-1 and zeta at 1, as for log(0) and the gamma poles; a pole of a non-rational argument is recorded | §2.2, §4.1 `constant_poles`, §6 step 3, §7 item 10 | the six rows of the report are `DivisionByZero("arctanh at 1 or -1")` or `("zeta(1)")` through `taylor_with_budget` and `series_expansion`, and `Unknown (DivisionByZero)` through the limit path (never `Finite(0)`), each in under 0.3 ms and 3 MB, the same through the §2.12 probe; `x + atanh(1/2)` and `x + zeta(2)` stay kernels; HEAD's `limit_via_series` of `x*arctanh(1)` is `Finite(0)` (`rt4num_repro.txt`, `rt4_extra.txt`) |
| numeric m2 (below the bar, a stated figure too high): short-mantissa arguments run cosh, sinh and the integer-valued power below the stated scan rates (MPFR's Ziv loop retries when `1 + x^2/2` or `1 + x` is representable) | the short-mantissa arguments join the scans of §2.3, and the slowest rate is restated as 1.6e7 units/s, 1.6 times the floor; the charges stay (the floor holds) | §2.3, §2.7 | red team 4's slowest nodes rerun on proto13 (1001 calls each, the node's own time): `cosh(3/2^130)` at 256 bits 1.60e7 units/s, `sinh(3/2^130)` 1.74e7, `cosh(1/2^31)` at 64 bits 1.87e7, `sinh(3/2^31)` 2.07e7, `cosh(15/2^470)` at 1024 bits 1.81e7, `(1 - 2^-30)^(-(5))` 1.99e7; the floor of 1e7 holds, so 1e8 units of numeric work take at most 6.3 s; 200 such kernels in 64 groups (10.7 KB) are `UndecidedZero` after 0.16 s (`rt4num_repro.txt`) |
| numeric m3 (below the bar, an honest refusal and a regression of revision 7rt3 against 7rt2): a hint or a kernel's source that writes an exactly-zero operand as `1 - 1` lost the enclosure the normal form has, since every literal had a radius and `1 - 1` was `0 +- 2^-63` | a literal exact at p bits is an exact ball, and the exact 0 has the rules `sqrt(0) = 0`, `0^k = 0` (k > 0), `0^b` by the sign of b's ball; non-dyadic rationals (`1/3 - 1/3`) keep a radius, recorded | §2.3, §2.5, §4.1 `exact_zero_operand`, §6 step 2, §7 item 10 | `1 - 1`, `sqrt(1-1)`, `(1-1)^(2*2)` and `(1-1)^(1/3)` are the exact 0 at 64 bits, `(1-1)^(-(2*2))` `None(Never)`; the three rows at 0, order 1, are `Ok` through `taylor_with_budget` and `series_expansion` (c0 = 1/(sin(1) - 1/2), gp `sin(1) - 1/2 = 0.34147098480789650665`) in under 0.3 ms; the hinted 6-level tower at 1, order 0, `Ok` in 0.4 ms (revisions 7rt3 and 8: `UndecidedZero`; 7rt2: `Ok`); the limit path `Unknown` (the constant terms are not rational); `(1)/(x + sin(sqrt(1/3-1/3) + 1) - 1/2)` stays `UndecidedZero` (§7 item 10) |
| memory m1 (below the bar, a stale figure): distinct coefficients that each render a long monomial reach 2.83 bytes per unit, above the 2.6 that §0, §2.7, §2.16 and §3.1 stated | the measured maximum is restated (2.83, an admitted call), with the family; the bound of the table "Memory" (3.0) holds, so no code change | §0 item 5, §2.7, §2.16, §3.1, §5.3 | `(-1)*(qa*...)/(1-x)^2` with 1000 names at order 1043: 117.8 MB of heap at 4.16e7 units, 2.83 bytes per unit, VmHWM 122 MB, 0.11 s, the same through `series_expansion`; order 1044 the same, 1045 `LimitExceeded(TreeSize)` (as in revision 8); `(qa^2*...)/(1-x)^2` at 521: 2.64; the limit path 1.18 bytes per unit with 5 MB (`rt4_ms.txt`) |
| memory m2 (below the bar): `limit_with_error` rendered every stored term to read only the precision, so `1 + x*S_17` was `(Finite(1), None)` (the tree cap) where `O(x^6)` is known | an internal entry `expansion_reaches` runs the driver and renders no term; §2.13 says which entries render what | §2.13, §2.16, §4.1 `limit_with_error_renders_none`, §6 steps 4 and 5 | `limit_with_error` of `1 + x*S_17` at 0, order 5: `(Finite(1), Some(O(x^6)))` in 0.33 s, 3.19e7 units, VmHWM 69 MB (revision 8: `(Finite(1), None)`); `series_expansion` of the same stays `LimitExceeded(TreeSize)`; `1 + x*S_18` likewise within `LIMIT_SERIES_WORK` (`rt4_ms.txt`) |
| structure m1 (below the bar, a constant factor in a cap; refusals of inputs HEAD answers): a computed kernel's cached tree counted 3 nodes above its arguments where its expression has 1, so `render_tree` reached 3 times the result's tree on a chain of computed kernels, and `MAX_RESULT_TREE` refused `sin^1200(x+1+0*(y0+...+y499))`, `sin^2000(x+1)`, the 20-caret tower at 1/2 and the 100-level cos tower, whose trees are under the cap (HEAD answers in at most 0.46 s) | the tree of a computed kernel is counted per kind (1 for an operator or a power, 2 for a radical, 0 for an inverse, 1 for a float leaf), and the hint test of §2.5 takes a margin (`HINT_MIN`) on the exact trees in place of a factor 2 that fired only through the over-count (red team 4's `fix2/`) | §2.2 (`KernelData::tree`), §2.5, §2.15, §2.16, §4.1 `kernel_tree_count`, §6 steps 3 and 4 | the five refused rows are `Ok` through `taylor_with_budget` and `series_expansion`: `sin^1200(x+1+0*(y0+...+y499))` in 0.05 s (a tree of 724 203 nodes), `sin^2000(x+1)` in 0.12 s (2 007 003), `sin^1700(x+1)` in 0.09 s (1 450 953), the 20-caret tower at 1/2, order 2, in 0.03 s (1 494 018), the 100-level cos tower in 0.01 s (1 754 050), each under 11 MB of VmHWM; `render_tree` exceeds the exact tree by 8 to 10 nodes on these five (`treeest.py`), and; on 1143 random inputs of red team 3 (`r3_s2` to `r3_s5`, `r3b_s11` to `r3b_s16`; `tools/treesafe_rt4.py`) the call with `TREECAP` set to the exact tree minus 1 is refused every time (`render_tree` is never below the tree) and every outcome kind equals revision 8's; the gp values: `sin^1700(x+1)` 6.5e-312, `sinDj_1200` 2.0e-312, the 20-caret tower 3.4e-328, the cos tower 2.8e-309 (`checks/gpres.py`, 300 digits); of the 107 gp-oracle inputs and the 119 displaycheck runs one each sets a hint, as in revision 8 (`hintscan_rt4.txt`); the hint margin makes most hinted trees smaller (`F_10` at 1, order 1: 462 903 nodes, revision 8 468 023; `F_11` 1 036 403) and one larger (review 8's `sing_16` at 1, order 0: 221 nodes, revision 8 113) |
| structure m2 (below the bar by the precedent of review 8 m4; a missing record): towers and nested functions give expanded normal forms polynomially larger than HEAD's nested derivative, and the cap refuses them from about 22 levels at order 2 | recorded with the measured limits: §2.15's row, §7 item 8 and the HANDOFF row (the remedy is the one for `F_k`: coefficients that keep products, or consumers memoized by `Arc` address) | §2.15, §5.3, §7 item 8, §4.1 `tower_tree_cap` | proto13 admits `x^x^...^x` at 1/2, order 2, up to 21 carets (a tree of 1 867 473 nodes in 0.04 s) and refuses 22 (`LimitExceeded(TreeSize)` in 0.04 s, 45 bytes; HEAD 8 ms); at 2, order 1, up to 136 carets (2.05e6 nodes at 135), refusing 137; the 150-level cos tower is refused in 0.06 s; revision 8 admitted 19 and 118 carets; the 21-caret tower at 1/2 equals gp's to 3.0e-328 (the towers at 2 overflow gp's reals); HEAD answers the refused towers in 8 to 36 ms |

Also changed while revising (each is in the body): `exp(k log a)` with an integer-valued k is enclosed
as the integer power a^k (§2.3, `ball.rs::exp_int_pow`, §4.1 `exp_integer_log`). The exact kernel counts
set fewer hints on one of review 8's 360 random towers, `((-3) + tanh(..))^(x+1)` at -2, whose constant
term then rendered as its normal form `exp(-2 log c)` with c < 0, which had no enclosure (revision 8
answered it through the hint `c^(-2)`); the identity `exp(k log a) = a^k` for integer k closes that side
of the numeric m3 class, and the tower is `Ok` and equal to gp's value again. Review 8's 360 random towers (`randhint_rt4.txt`): 130 `Ok` (revision 8: 103), of which 27 revision 8 refused and none it answered lost; 129 equal gp's series at 300 digits, and the 130th, `((-1) + sqrt((1/2) + sin(..)^2)^2)^(1/2)` around a power `(c)^(x+1)` with c < 0 at 1, order 1, differs from gp's series only by gp's branch: its inner constant term is exactly the real `c^2`, which gp computes as `exp(2 log c)` with an imaginary part of -1.4e-115, so gp's principal square root of the negative constant term is `-0.372 I`; with `c^2 exp((x-1) log c)` in place of `c^(x+1)` (the same function) gp gives `+0.372 I` and `0.0652 - 0.0118 I`, the engine's two coefficients (`EXP/rev7rt4/checks/rh131.gp`; §2.4, "Branch at the point"); the other 229 are errors in both revisions. The internal precision is printed by the `series`
mode of the driver (`internal Exact` or `internal Abs(p)`), and `checks/revchecks_rt4.py` reads the
exact zero `0^x` at 1 as `val 0 prec 4 internal Exact` (revision 8's line printed the sentinel's value
as the precision); new probes and checks: `api::limit_with_error` with the driver mode `NO_TERMS`
(`ENTRY=lwe`), `probes/rt2t.rs` (red team 4's probe with `TREE_TOO`), `tools/treesafe_rt4.py`,
`tools/maxfuzz_rt4.py`, `tools/rt4_ms.sh`, `tools/rt4_extra.sh`, `tools/randhint_rt4.py` (revision 8's
and revision 7rt2's outcomes beside each input), and `rt4num/` (red team 4's numeric probe and
reproductions linked against proto13).

Where I did not take the red team's literal wording:

- Series H1 offered the out-of-band form, or a reserved sentinel with clamps and finite limits. I took the
  out-of-band form: it keeps the control at i64::MAX - 1 that the reserved form refuses, and it removes
  the sentinel from the public layer too. Where the red team clamped a finite precision to i64::MAX - 1,
  `Prec::add` clamps to `Abs(i64::MAX)` (the representation allows it), and the driver stops once a
  clamped attempt failed to raise the precision (the red team's sketch let the doubling run into the
  budget for producer 3: 0.1 to 0.18 s and 104 MB).
- Series H2 offered `LimitExceeded` or `Unsupported` for a buffer that does not fit without a budget. I
  took `Unsupported` with the cap `isize::MAX / 64 = 2^57` slots, the red team's bound, so no new
  `StructureLimit` variant is needed (the cap is reached only without a budget).
- Numeric m2 offered a restated rate or a charge of two operations at `2p + 64` bits for sinh and cosh.
  I restated the rate (1.6 times the floor holds the bar) and left the charge to the port's calibration.
- Numeric m3 offered a retry on the normal form, or the cheaper exact literals with the exact-zero rules.
  I took the cheaper fix, which closes the measured rows and the hinted tower; the non-dyadic remainder
  is recorded (§7 item 10).
- Structure m1 counted 1 for an `Inverse` kernel in its `fix2/`; its report says 0, which is the count of
  the expression (P0's rendering), and I took 0 (the property check on red team 3's random inputs holds
  with it).

## Revision 8 (response to review 8)

Review: `fu2/design/S1-symbolic.review8.md`, on the revision 7rt3 prototype `EXP/rev7rt3/proto11/`. One
blocker (B1, a wrong result) and four minor issues. Each is fixed in the body below, and the fix of B1
closes its class, "a node marked regular although a node of its subtree is not", not only the measured
inputs. The prototype of revision 8 is `EXP/rev8/proto12/` (revision 7rt4's is at the top). The facts were rechecked at HEAD
6e03987: `git diff --stat 2e15d1c HEAD` touches only `docs/` (HANDOFF's build bullet, README,
TEST_BASELINE, the m23_golay handoff), so rustmath-symbolic, the crates it reads and the cited lines of
`repl.rs` and `docs/HANDOFF.md` (`:461-463`, the S1 bullet at `:509-519`) are unchanged. Every check of
revision 7rt3 was rerun on proto12 (`EXP/rev8/tools/runall_r8.sh`), with review 8's reproductions
(`tools/review8_r8.sh`, output `EXP/rev8/review8_r8.txt`) and its random hinted towers against gp
(`tools/randhint_r8.py`, output `randhint_r8.txt`, revisions 7rt3 and 7rt2 beside each input). Numbers
are release on an 8 MB thread, budget 1e8, unless stated.

| issue | change | where | evidence (proto12) |
|---|---|---|---|
| B1 (a wrong result): a tower level whose exponent depends on the variable but expands to an exact constant took its regular flag from the base alone, so the value-at-the-point hint substituted a subtree that is not regular. `x + (1+sin(1))^(500*sign(x^2))` at 0 printed c0 = `(1 + sin(1)) ^ (500 * sign((0 ^ 2)))`, whose value is 1, not (1 + sin 1)^500 = 3.82e132; `(1+sin(1))^(((1)/x)*0 + 500)` printed `1 / 0`, `x + (1+sin(1))^(500*((x-1)/(x-1)))` at 1 `0 / 0`, and the 4-level `W_4` (exponent `2*sign((x-1)^2)`) a c0 of value 1 | The class is "a node marked regular although a node of its subtree is not". §2.5 now states the invariant: the flag is the AND of a node's own conditions and of its children's flags, and no rule drops a child's flag; the tower's constant-exponent state carries the exponent's flag (review 8's `p8.diff`), and every record was audited against the AND. Second, a node whose series is an exact constant c is regular whatever its subtree holds, with the rendering of c as its substitution (a seed of the substitution memo, `seed_const`): it equals c near the point, and the seed is the engine's own value. Without seeds the flag fix alone refuses towers HEAD answers (below). A hint is set only when it is smaller than the normal form | §2.5 "Value at the point", §2.16, §4.1 `value_at_point_irregular_exponent`, §6 step 4 | the review's five inputs and V_3 (the tower with exponent `2+((1)/(x-1))*0`, 3 levels at 1) give gp's values (`checks/c0check.py`: c0 = 3.8233154825e132 for the first three, W_4's c0 = 2.89697852518268829051 and c1 = 18.2587797964534041670, V_3's c0 = 5.746165117285e886330929042 to 63 digits and c1 to 47 digits of gp's `derivnum`), with no `sign`, `/ 0` or `(1 - 1)` in any result; each call under 0.06 s and 6 MB through `taylor_with_budget` and `series_expansion`, and under 0.2 s and 15 MB through the limit path (order 5; `Unknown`, the constant terms are not rational). With seeds, V_20 and V_30 at 1, order 0, are trees of 277 and 815 nodes in 0.2 and 0.4 ms (`p8.diff` alone: `LimitExceeded(TreeSize)`; HEAD 37 and 48 us), the `sign` towers likewise; a 3000-level seeded tower (81 KB) is `Ok` at order 0 in under 0.1 s and 31 MB and refused by the budget at order 1 in 0.4 s, on a 2 MB debug thread too, and 6000 seeded summands (96 KB) are `Ok` in 0.04 s with 3.6 MB of heap (`seeds_r8.txt`, `tools/seeds_r8.sh`). Regular inputs are unchanged: `F_10` and `F_30` at order 0 trees of 51 and 151 nodes, `F_10` and `F_11` at order 1 468 023 and 1 046 643 (no seeds), the hinted families at 1/3 and `F_2`, `F_3` equal to gp's to 1e-295, and 1 of the 107 gp-oracle inputs sets a hint, as before |
| m1 (below the bar, an honest refusal and a regression of revision 7rt3): the ball code took only an `Integer` literal exponent as an integer power, so `(-3 + 1)^(-2)`, which a hint or a kernel made from an input node renders, had no enclosure, where the normal form's `c^-2` had one; 6 of the 102 random towers that revision 7rt2 expands were `UndecidedZero` in revision 7rt3, and `(1)/(x + sin((-2)^(-2)) - 1/4)` at 0 was `UndecidedZero` in both, where HEAD answers | `small_int_value` reads an exponent's exact integer value in i64 (a literal under negations, sums, differences and products, each checked; at most 64 nodes and 8 levels, no allocation); an exponent with such a value is an integer power: binary powering for \|k\| <= 3, else `exp(k log \|a\|)` with the sign and the inverse, a^0 = 1; the a-priori charge of a non-literal exponent is `4u` (was `3u`). §2.5's sentence on the zero test is corrected: a hint can change an `Unknown` of the numeric test, never its soundness | §2.3, §2.5, §2.16, §4.1 `integer_valued_exponent_enclosure`, §6 step 2 | the review's enclosures exclude 0 around gp's values (1/4, 4, -8, -243, 1; revision 7rt3: none); `(1)/(x + sin((-3+1)^(-2)) - 1/4)` and `(1)/(x + sin((-2)^(-2)) - 1/4)` at 0 are `Ok` with HEAD's and gp's c0 = -385.20196639526608917 and c1 = -148380.55491477970544; review 8's 360 random towers: 103 `Ok` (proto11: 96), 0 different from gp, among them the 6 that revision 7rt3 refused and revision 7rt2 expands and one that both refused; the other 257 are errors in all three revisions (mostly non-real constants, which have no enclosure). The eight constants of `powtime_r8.txt` run at 8.3e7 units/s or more; the heap is unchanged (a 16 000-literal sum inside a kernel: 10.7 MB, as in revision 7rt3, `seeds_r8.txt`); the calibration keeps every outcome and every row's units within 0.5 %, its slowest refusal (2000 64th roots) at 2.83 s (2.72 s) |
| m2 (below the bar): `x + 0^pi`, `x + (x-x)^pi` and `x + (x-x)^(x-x+pi)` were `UndecidedZero(0^pi)` while `x + 0^(pi + x)` was x | `0^alpha` for a constant alpha is decided by the rule of `0^r` on Re alpha, one function `zero_pow_sign` shared by `kpow_const` and `zero_pow` | §2.2, §2.5 `0^r`, §2.16, §4.1 `zero_power_constant`, §6 step 3 | `x + 0^pi`, `x + (x-x)^pi`, `x + (x-x)^(x-x+pi)` and `x + 0^(sqrt(2)-1)` at 0 are `x` (the limit path `Finite(0)`), `x + 0^(-pi)` `DivisionByZero`, `x + 0^(sin(1)^2+cos(1)^2-1)` `UndecidedZero` of its real part; revision 7rt3 gave `UndecidedZero(0^alpha)` for each |
| m3 (below the bar, a port note): in the thread's TLS teardown, `Drop` took the iterative branch and `detach` called `LEAF.with`, which panics there, and the teardown does not finish (measured: panic after panic until the process was killed after 20 s) | `LEAF.try_with`, and a fresh `Arc::new(Expr::Real(0.0))` when `LEAF` is gone | §2.14, §2.16, §4.1 `drop_during_tls_teardown`, §5.4, §6 step 7 | mode `tls`: a unique chain, `twin`, `sib` and a left-deep sum of 2 x 10^5 levels drop on a 2 MB debug thread with at most 16.5 KB of stack, 10^6 levels on a 1 MiB release thread in at most 0.40 s; with `LEAF.with` (revision 7rt3) the same drop panics in the teardown |
| m4 (below the bar, wording): §2.15 did not say that the tree cap refuses `F_12` to `F_15` at order 1 although the engine's own work is small | §2.15's row and §7 item 8 say so, and name one remedy for the cap and the budget on this family: a representation of coefficients that keeps products | §2.15, §7 item 8 | `F_10` at order 1: 2.4e6 units, 5 ms; `F_11`: 5.0e6, 11 ms; `F_12` to `F_15` refused after 36 ms to 0.26 s; `F_16` `BudgetExceeded` after 0.52 s |

Also changed while revising (each is in the body): `maybe_hint` and `hint_exp_arg` set a hint only when
its tree is smaller than the normal form's (a seed can make a substitution larger than its estimate;
before seeds the estimate was exact, so no earlier hint changes); new probes and checks:
`proto12/powtime.rs` (the enclosure of a constant timed against its charge, `powtime_r8.txt`),
`checks/c0check.py` (a result's coefficients evaluated in gp against gp references, and a scan of its
text for `sign(0)`, `/ 0` and `(1 - 1)`), the mode `tls` of `probes/dropm.rs`, `tools/randhint_r8.py`
(review 8's generator, with the outcomes of revisions 7rt3 and 7rt2 beside each input), `hs/headsimp.rs`
(review 8's HEAD `taylor` on a file), and a 1 GB stack for gp in `checks/gpres.py` (its default 8 MB
overflowed on the 10-level hinted families).

Where I did not take the review's literal wording:

- B1 asked for the four-line change of `p8.diff`. I took it, and added the exact-constant seeds, because
  the flag fix alone turns answers into refusals where HEAD answers: with `p8.diff`, V_20 and V_30 (the
  tower `(1+exp(V))^(2+((1)/(x-1))*0)` with 20 and 30 levels, at 1, order 0) were
  `LimitExceeded(TreeSize)` (the normal form doubles per level), while HEAD prints the input with the
  point substituted and `(1/(1-1))*0` folded, in under 0.1 ms. The review's assertions hold on proto12:
  no `sign` node and no division by zero in any result, and the gp values.
- m1 offered "an exponent whose own ball is exact and integral (radius 0)". A literal's ball has a
  nonzero radius (`Ball::rounded` gives |m| 2^-p), so I read the exponent's exact integer value instead,
  by a bounded walk (a first version kept a map of values per node, which added up to 40 bytes per node
  to the enclosure's 280 against its 96 units, past the 3 bytes per unit of §2.16 "Memory").
  It follows literals, negations, sums, differences and products, at most 64 nodes and 8 levels; a power or a quotient node in an
  exponent (`(-2)^(2^1)`) is not followed and stays without an enclosure, an honest `Unknown` (a
  literal `4/2` is folded to 2 by the parser, so `(-2)^(4/2)` is enclosed).
  Binary powering is used only up to |k| = 3, so that the a-priori charge of a non-literal exponent stays
  a constant (`4u`); larger values take `exp(k log |a|)`, which is as precise for a nonzero test.
- m2 did not say what `0^alpha` with Re alpha = 0 gives; it is `NotLaurent`, as in the series rule (the
  value has no limit), and an undecided real part is `UndecidedZero` of that part.

## Revision 7, red-team round 3

Reports: `fu2/design/S1-symbolic.redteam3-memory.md`, `-numeric.md`, `-series.md` and `-structure.md`
(four lenses on the revision 7rt2 prototype `EXP/rev7rt2/proto10/`). Three holes are over the review bar,
ten are below it. Each is fixed in the body below, except memory m2, which is recorded (§7 item 9), and
each fix closes the class of its hole, not only the measured input. The prototype is now
`EXP/rev7rt3/proto11/` (then the block at the top). Every reproduction of the four reports was rerun on it
(`EXP/rev7rt3/tools/redteam3_rt3.sh`, output `EXP/rev7rt3/redteam3_rt3.txt`; the numbers below are from
that run unless stated), and so was every check of revision 7rt2 with the reproductions of red teams 1
and 2 (`EXP/rev7rt3/tools/runall_rt3.sh`, §5.3). The facts were rechecked at HEAD 2e15d1c: nothing that
S1 reads has changed since revision 7. Numbers are release on an 8 MB thread, or debug on a 2 MB thread,
budget 1e8 (`LIMIT_SERIES_WORK` on the limit path), unless stated.

| hole | change | where | evidence (proto11) |
|---|---|---|---|
| series H1 (over the bar, a wrong decided result): an empty truncated argument `O(h^q)` of negative precision met `exp`, `sin_cos`, `sinh_cosh`, the atan family with erf, `integrate` and `0^r` as if its valuation were at least 0. `x^10*exp((sin(x)-x)/x^4)` at 0, order 2, was a decided `0`, `(1)/(1-x) + x^10*exp(..)` was `1 + x + x^2`, `x^30001*exp((exp(x^30000)-1)/x^30001)` at `TAYLOR_ORDER` was `0`, and the limit path gave `Finite(1)` for `1 + x^10*exp((sin(x^2)-x^2)/x^9)`, whose left limit is +infinity (gp: "domain error in exp: valuation < 0"); `0^((sin(x))/x)` at order 0 was a decided `NotLaurent`, although its value is 0 | The class is "an operation that decides from an operand whose valuation is undecided". §2.1 states the rule: an empty truncated series `O(h^q)` has an undecided valuation of at least q, so it meets "valuation >= 0" only when q >= 0 and no precondition on its leading coefficient; an operation whose precondition it does not meet returns `NeedPrecision`, as `inv`, `log` and `pow_const` already did. Applied to `exp`, `sin_cos`, `sinh_cosh` (and `tan`, `tanh`), the atan family with erf, `integrate` (an empty integrand of negative precision may hold an h^-1 term) and `0^r`; every other operation that reads a valuation or a leading coefficient was checked against the rule (`mul`, `add`, `pow_int` item 4, `abs_sign`, `exact_constant`: already conforming) | §2.1, §2.4 rows `exp`, `sin_cos`, arctan..erf, `deriv`/`integrate`, §2.5 `0^r`, §2.6, §2.16, §4.1 `empty_argument_negative_precision`, `zero_power_rule` | every row of the red team's table is `NotLaurent` through `taylor_with_budget` and `series_expansion` and `Unknown` through the limit path, in at most 0.03 s and 4 MB (the order-20 000 input: `NotLaurent` in 25 ms; its limit path `PrecisionExhausted(134)`, an honest refusal), the same on the 2 MB debug thread and through the §2.12 probe parser; HEAD prints `0` and `1 + x + x^2`. `0^((sin(x))/x)` is the exact 0 at orders 0 and 3; `exp((sin(x)-x)/x^2)` stays `1 - x/6 + x^2/72`. gp oracle 107 OK |
| structure H1 (over the bar, a decided error where HEAD answers): the rule of `Pow` was chosen from the syntax of the base, so a base that depends on the variable but expands to the exact zero took `exp(r log l)` and `log(0)` (`exp(x) + ((x+1)^2 - x^2 - 2*x - 1)^(x+1)`, `cos(x) + (x*0)^(1+x^2)`, `(x-x)^x`, `1 + (x*0)^(2^x)`: `DivisionByZero("log(0)")`), and `0^r` read `val` (a lower bound) and accepted only a positive rational constant term (`x + 0^(2^x)` at 1 and `x + 0^(x+sqrt(2))` at 0: `NotLaurent`; an undecided pole term of r reported as decided) | The class is "a rule chosen from the syntax, or decided by a test weaker than the engine's". A base that expands to an exact constant takes the constant-base rule with that constant, as an exact constant exponent already took the constant-exponent rule. `0^r` is decided from the series of r with the engine's own tests: an empty truncated r asks for precision; a pole of r or `r(a) = 0` with a decided lead is `NotLaurent`, an undecided lead `UndecidedZero`; else the sign of Re r(a) by `split_imag` and rule 5's enclosures (`real_sign`): positive gives the exact 0, negative `DivisionByZero("0 to a power with negative real part")`, Re = 0 with a decided imaginary part `NotLaurent`, anything undecided `UndecidedZero` | §2.5, §2.9 (`DivisionByZero`), §2.16, §4.1 `zero_power_rule`, §6 step 4 | the eight inputs of the red team give `1 - x^2/2`, `0`, `1`, `1 + x + x^2/2 + x^3/6`, `-3^(1/3)`, `1 + (x - 1)`, `x` and `UndecidedZero` (the undecided pole term), each the red team's gp value, in at most 2 ms on 2 MB debug; the limit path `Finite` where the constant term is rational; eight more zero bases (§5.3) give the rule's outcomes, `x + (x*0)^(cos(4) + x)` `DivisionByZero` (cos 4 = -0.65) |
| structure H2 (over the bar under (b)): a kernel made from a computed value rendered its arguments' normal forms, which name a kernel twice for a power or a product of a sum, so a result's tree doubled per level of input. `simplify` and `substitute` of `taylor(F_10, x, 1, 1)` (112 bytes, a tree of 1.04e7 nodes) took 0.8 s and 960 MB where HEAD takes 18 us, `F_12` aborted at 4 GiB, and `==` of the order-0 result of `F_30` (1.07e10 nodes) with its clone took 28.7 s | The class is "a result whose tree is exponential in its DAG". Two changes. First, the value at the point of a regular input node (every subexpression analytic at the point, every division by a value nonzero there; tracked by a flag carried with each value of the work stack) renders as that node with the variable replaced by the point when its normal form's tree is more than twice that and 64 more (a hint in a side table of `Work`; the substitution copies each distinct input node once, charged `SUBST_NODE`); the normal form, the key and the zero test are unchanged, and a removable singularity is never substituted (`sin(x)/x` at 0 is not regular), which answers red team 1's objection to substitution. Second, every constant and kernel caches the size of its tree, and the result's tree and the constant of `UndecidedZero` are capped (`MAX_RESULT_TREE = 2^21`, `LimitExceeded(TreeSize)`), so a consumer that walks a result is bounded by the cap times its cost per node (about 200 MB for `simplify`) | §2.2, §2.5 "Value at the point", §2.7, §2.8, §2.9, §2.15, §2.16, §4.1 `result_tree_cap`, §6 steps 3 and 4, §7 items 5 and 8 | `F_30` at order 0 is the input with x = 1, a tree of 151 nodes (HEAD's has 151); `simplify`, `substitute` and `==` of the order-0 results of `F_22` to `F_30` take at most 32 us (HEAD 6.5 us); `F_10` at order 1 is a tree of 468 023 nodes, `simplify` 52 ms and 48 MB; `F_12` to `F_15` at order 1 are `LimitExceeded(TreeSize)` in at most 0.2 s (their x coefficients have 2^k terms; the budget refuses `F_16` on, as it refused `F_20` on before, §7 item 8); values equal gp's to 1e-295 (`F_2`, `F_3`, orders 0 to 2) and 1e-307 on hinted families with bounded values; of the 107 gp-oracle inputs one sets a hint. The cap also refuses results HEAD does not finish: the sparse-rendering family of review 6 (where the budget refused it), red team 2's long monomials, `(F)/(1-x)` with a 33 000-call F from order 63 |
| numeric m1 (below the bar, an honest refusal): asinh, acosh, atanh by their log formulas lost the argument's relative precision for x << -1, |x| < 2^-p and |x| > 2^(2^27) | MPFR's own functions with Lipschitz bounds (asinh `min(1, 1/d)`, acosh `1/sqrt((lo-1)(lo+1))`, atanh `1/((1-a)(1+a))`); the scan adds asinh at `-2^k` and acosh at `L 2^k` | §2.3, §2.16, §4.1, §6 step 2 | the red team's nine inputs `Ok` in at most 0.3 ms, the coefficients equal to gp's; the enclosures contain gp's values; slowest scan rates 2.15e7 to 4.68e7 units/s at 64 bits |
| numeric m2 (below the bar): an integer exponent above 2^32 had no enclosure, while the same exponent written `2^33` did | binary powering for every exponent in i64, `exp(k log |a|)` with the sign beyond it, charged `u (2 min(bits(k), 64) + 1)` | §2.3, §2.16, §4.1 `integer_power_beyond_2_32` | the red team's three inputs `Ok`; the enclosures contain gp's values; `(1 + 10^-9)^(2^63 - 1)` stays without one (outside the window); a beyond-i64 power of a negative base is enclosed at 256 bits |
| memory m1 (below the bar, a stated figure exceeded): one shared constant rendered once per term held 2.93 bytes per unit, 296 MB | `series_expansion` renders each distinct coefficient once (a map by the constant's address), and the tree cap refuses the large members of the family before they are rendered; the stated figures are the new measured maxima | §0 item 5, §2.7, §2.8, §2.16, §3.1 | `(-1)*(a*b*...)/(1-x)` with 2700 names: order 380 `Ok` with 1.9 MB, orders above about 400 `LimitExceeded`; `L*((1)/(1-x))` at order 6000 2.5 MB (revision 7rt2: 252 MB); the largest heap peak now 2.59 bytes per unit at a refusal, the largest admitted VmHWM 230 MB |
| memory m2 (below the bar per call): a REPL line keeps every result | recorded: a per-cell policy is J1b's; the tree cap bounds a result's nodes but not its digits | §3.1 (`TAYLOR_WORK` comment), §7 item 9 | the 400-call line holds 2.26 GB in 7.4 s, as before; HEAD finishes neither line |
| memory m3 (below the bar): the limit path rendered every stored term to read the first | an internal entry that charges, bounds and renders the first stored term only | §2.13, §2.16, §4.1 `limit_first_term_only` | `1 + x*S_18`: `Finite(1)` at 5.41e7 units in 0.37 s (revision 7rt2: `Unknown (BudgetExceeded)`) |
| memory m4 (below the bar, a missing row): the variable's series copy | a row of the table "Memory" | §2.16 | 48 bytes for 24 units; the chains at most 1.50 bytes per unit |
| series m1 (below the bar, an honest refusal): the loss-driven step crawled when a kept leading term's precision did not grow with P | a stalled step doubles the extra precision | §2.6, §2.16, §4.1 `stalled_loss_step` | `((x+x^2)^1000)/x^999` `Ok` at every order from 1 to 400 in at most 0.01 s, equal to gp's `x (1 + x)^1000`; the limit path `Finite(0)` |
| series m2 (below the bar, a debug crash where HEAD fails): `inv` subtracted exponents unchecked | unsigned distances below the slot count; the other casts audited | §2.1, §2.4, §2.16, §4.1 `inv_offsets_unsigned` | `x^2 + O(x^4)` on 2 MB debug and in release |
| structure m1 (below the bar, a false claim): §2.14's Drop recursed through a child shared within a node | every child `Arc` with children moved into a list of `Arc`s, taken with `Arc::into_inner` | §2.14, §2.16, §4.1 `drop_shared_children` | `twin` and `sib` with 2 x 10^5 levels drop on 2 MB debug with 17 KB of stack (revision 7rt2's Drop: 14 513 and 14 455 levels) |
| structure m2 (below the bar): an integral float at or above 2^53 printed as an Integer of another value | `.0` after an integral float | §2.11, §4.3 `test_display_integral_float` | the three inputs round-trip with their values |

Also changed while revising (each is in the body): the tree size of a constant is counted as `mono_expr`
builds its rendering (a first estimate of L + 3 nodes per term set hints on plain sums such as
`x + y1 + ... + y5000` and over-counted the cap); the hint table is a side table of `Work`, not a field of
`ConstData` (a field would have grown every constant's header to a 240-byte chunk, 3.3 bytes per unit);
`KernelData` gains `tree` (a 128-byte chunk); the `expand` records name their node and carry the regular
flag (`EProd` 104 bytes, a 112-byte chunk, 3.0 bytes per unit with its slot); new probes
(`probes/consumers.rs`, `probes/dropm.rs`, `probes/replline.rs`, `probes/twolines.rs`, `probes/ksize.rs`,
the `ball` mode of `probes/rt2.rs`, `checks/gpres.py`, `tools/cons3.sh`, `tools/hintscan.py`,
`tools/rt3mem/`).

Where I did not take the red team's literal wording:

- Structure H2 offered a tree cap, checked in `mk_kernel` and in `render_units`, or consumers memoized by
  `Arc` address. I took the cap, checked on the result and on the constant of `UndecidedZero`, not in
  `mk_kernel`: a kernel with a large tree that never reaches the result (a factor 0) costs a consumer
  nothing, and refusing it would refuse inputs whose answer is small. A cap alone would refuse what HEAD
  answers from about 8 levels of `F_k` at order 1 and 18 at order 0 (revision 7rt2's trees had 1.04e7
  nodes at 10 levels and order 1 and 4.19e7 at 22 levels and order 0, doubling per level); so I added the rendering of values
  at the point from the input, which makes order-0 results HEAD's size and leaves the cap to the
  expanded normal forms of 2^k terms. Memoized consumers stay outside S1's item (§7 item 8).
- Memory m1 offered `EXPR_NODE = 36` as a margin; the shared rendering and the tree cap remove the
  family instead, so the constant and the calibration stay.
- Memory m2 offered a per-cell `TAYLOR_WORK` or a cap on `render_units`; both need REPL state per cell
  or refuse results HEAD cannot produce anyway, and the other REPL commands have the same exposure, so
  the per-cell policy is left to J1b and recorded (§7 item 9).
- Memory m3 offered an order-0 expansion first; I took the first-term entry (one expansion, never two).
- Numeric m1's cheaper alternative (`asinh(x) = -asinh(-x)`) covers one regime of three; I took MPFR's
  functions.

## Revision 7, red-team round 2

Reports: `fu2/design/S1-symbolic.redteam2-memory.md`, `-numeric.md`, `-series.md` and `-structure.md`
(four lenses on the revision 7rt1 prototype `EXP/rev7rt1/proto9/`). Four holes are over the review bar,
six are below it. Each is fixed in the body below, and each fix closes the class of its hole, not only
the measured input; where a hole below the bar is a stated figure or a claim, the claim is corrected.
The prototype is now `EXP/rev7rt2/proto10/` (then the block at the top). Every reproduction of the four
reports was rerun on it (`EXP/rev7rt2/tools/redteam2_rt2.sh`, output `EXP/rev7rt2/redteam2_rt2.txt`; the
numbers below are from that run unless stated), and so was every check of revision 7rt1 with red team
1's reproductions (`EXP/rev7rt2/tools/runall_rt2.sh`, §5.3). The facts were rechecked at HEAD 2e15d1c:
nothing that S1 reads has changed since revision 7. Numbers are release on an 8 MB thread, or debug on a
2 MB thread, budget 1e8 (`LIMIT_SERIES_WORK` on the limit path), unless stated.

| hole | change | where | evidence (proto10) |
|---|---|---|---|
| series S-B3 (over the bar, a wrong result marked decided): the binomial path of `pow_int` (an undecided leading coefficient, n > 64) returned `acc.trunc(S)`, and `trunc` keeps an exact series of at most 64 terms exact, so a truncated `(c0 + u')^n` claimed to be the whole polynomial. After a pole shift the driver saw no precision loss: `taylor((x+1+sin(y))^100/x^5, x, 0, 3)` returned the exponents -5 to -2 marked exact, without the nonzero terms -1 to 3 (and at order 70 without x^66 to x^70) | The class is "an operation that marks `EXACT` a series whose terms it did not all compute". §2.1 now states the rule: `EXACT` means every term of the value is present, and an operation that leaves terms out returns the precision up to which it computed them. Both large-n paths of `pow_int` return precision S (the binomial sum is truncated hard at S, as Miller's `from_dense(0, g, S)` is), and the binomial loop stops at m = n (the terms m > n are zero, and `c0^(n-m)` with n - m < 0 would invert c0). The other producers of `EXACT` were checked against the rule: an exact constant, the variable, an exact monomial and its powers, the products, sums, shifts and inverses of exact operands computed in full (`mul` keeps `EXACT` only when no pair was dropped), the binary powering of an exact series (every product exact) | §2.1, §2.4 item 7, §2.16 `pow_int`, §4.1 `binomial_power_precision` | order 3: the exponents -5 to 3 with precision 4, `Ok` in 0.02 s with 9.9e6 units (the limit path 1.3e7); the nine coefficients equal gp's `binomial(100, e+5) (1+s)^(95-e)` with s = sin(y) (`checks/coefcheck`-style check in gp: 9 of 9); order 70: the exponents -5 to 70 with precision 71, 76 of 76 equal; `(x+1+sin(y))^100` and `(x+1+exp(1))^100` at order 3 both have precision 4 (revision 7rt1: `EXACT` for the first). Through `taylor_with_budget`, `series_expansion` and the limit path |
| series S-B2 (over the bar, a crash where HEAD answers): `mul` computed the span `hi - val + 1` of a product with plain `i64` arithmetic; `((1)/x^N + x^N)*((1)/x^N + x^N)` with N = 4e18 (97 bytes) wrapped it, took the dense branch with an empty buffer and panicked in release | The class is "exponent arithmetic that is not checked". `mul` checks the span (a span that does not fit takes the heap merge, whose size is the pair count) and every pair sum (`checked_add`: a sum overflows only upwards, since it is at least the product's valuation, which fits, so an overflow lies above every cutoff; an exact product whose top exponent does not fit takes the working cutoff instead of being kept in full). The audit of the engine found seven more unchecked or wrapping sites, all fixed: `deriv`'s `e - 1` and `prec - 1`, `sincos_u`'s `2 * prec`, `pow_int`'s `P - n vb`, the driver's `target - s.prec`, `target + next` and the cast of its doubling cap, and the rendering's `-e` (an exponent `i64::MIN` is now the overflow error) | §2.1, §2.4 `mul` row, §2.16 `mul` row, §4.1 `mul_exponent_span` | the 97-byte input: `Ok` in 0.1 ms with 3.6e3 units, terms `[(-8000000000000000000, 1), (0, 2)]`, precision 4 (the value `x^-(8 * 10^18) + 2 + O(x^4)`), the same through the three entry points (the limit path: `Unknown`, a pole); the red team's heap-merge input `ovf_heap`: `Ok` with 1.5e5 units, first exponent -940, through the three entry points; HEAD answers the first in 1.1 ms. proto9 on the first: `index out of bounds` (the control) |
| series S-B1 (over the bar, a budget hole): the first loop of the `log` recurrence step scanned every offset of the argument at every step, to find w_k and to sum the charge, which counted only the offsets below k. A sparse argument whose offsets lie far above the low steps did n times the offset count of work for about 17 units per step: `log(1 + x^1000000*(1+x)*(1+x^2)*...*(1+x^65536))/x^1131072` at the REPL's order 0 (203 bytes) ran 78 s before the budget refused it, 70 s through the limit path | The class is "a loop of a recurrence step longer than its charge". Every loop of every step now visits only the offsets j <= k: the offsets are sorted, the charge loop of `log` stops at the first offset >= k (as its summing loop and every other recurrence already did) and w_k is found by one binary search (charged `bits(offsets) + 1`). §2.4 states the rule for every recurrence; the rows of `inv`, `exp_u`, `sincos_u`, `log` and `miller` in §2.16 say so, and each was checked against it | §2.4, §2.16, §4.1 `budget_log_sparse_offsets` | the 203-byte input: refused at 1e8 after 0.28 s, VmHWM 21 MB, through `taylor_with_budget`, `Unknown (Budget)` after 0.18 s through the limit path; the other four of the red team's table refused after at most 0.21 s; the `exp` control as before (0.18 s). Values: `log(1 + x^5*(1+x)*(1+x^2)*(1+x^4))/x^9` at order 12 and the 40-offset analogue at order 40 equal gp's series (16 and 32 coefficients) |
| structure H1 (over the bar, a stack overflow): between two parentheses §2.12's grammar has up to six levels (a `+ -` chain, a `* /` chain, a negation, a tower, a negated exponent, the atom). Revision 7rt1 read each level that repeats without a parenthesis by a loop, but still recursed once per nested operand, through a dispatcher frame and a chain frame for each level a parenthesis used: 11 frames per parenthesis, 15.5 to 18.9 KB of debug stack against the parser's 9.7 KB. `x + -(-(...(1)^2*3+4)...)^2*3+4` with 133 levels (1.2 KB) aborted a 2 MB debug thread after the parser had read it, five other mixed shapes likewise at 111 to 192 levels, and a 39.6 KB line killed the REPL's 64 MiB debug thread | The class is "engine recursion over the input". `expand` and `from_expr` run one loop over an explicit work stack each, with one record per pending node of the current path (a chain, a tower, a call or a `%` node holds its items and the values read so far in a boxed record, at most 96 bytes, charged `RECORD_UNITS = 32` when it is pushed; a unary node is a 16-byte record), and the value of a finished node is handed to the record on top. The order of the operations, the charges, the checks and the source nodes of the kernels are those of the recursive reading. The engine has no recursion over the input left (§2.16 fact 3 rewritten), so on every thread the parser is the only recursion over a text, whatever its shape. The six mixed shapes join the table of §2.5, `deep_mixed_nesting` (§4.1) and §6 step 8 | §0, §2.2, §2.5, §2.16, §4.1, §6 steps 3, 4 and 8, §7 item 5 | `mix5cs_133` on one 2 MB debug thread: `Ok` in 2.5 ms (1.6e5 units) through `taylor_with_budget` and `series_expansion`, `Finite(1)` through the limit path (proto9 on the same thread: stack overflow, the control); 150 and 200 levels: `Ok` in at most 3.5 ms; `mix5_180`, `mix4_200`, `mix6c_112` (the §2.12 probe parser): `Ok` in 6.3 ms; refused by the budget after 0.74 s; `Ok` in 4.6 ms (the limit path `Unknown`: its constant is not rational). The deepest of each shape on one thread is the parser's (HEAD's parser 216 levels on 2 MB debug and 4124 on 8 MB release, the §2.12 probe parser 266 and 4438; gdb at one level more shows the overflow in nom), and 100 KB of every shape is handled by the engine alone on 2 MB debug (nine shapes with HEAD's parser and three with the probe parser, each 84 to 100 KB: `Ok`, the budget or an honest error in at most 3.9 s on 2 MB debug and 1.2 s on 8 MB release, through `taylor_with_budget` and the limit path). The REPL's 64 MiB debug thread: `mix5cs` with 4400 levels (39.6 KB) `Ok` (`1 + x`) in 0.14 s and `Finite(1)`, with 6000 (54 KB) `Ok` in 0.18 s, VmHWM 69 MB. Values equal gp's at 2, 3 and 5 levels (maxdiff 0) |
| numeric m1 (below the bar, an honest refusal of decidable constants): the zero test and the integer test stopped at the first precision without an enclosure, although rule 5 lets any of the three decide; at 64 bits many enclosures are missing only because a ball is too wide (`tan(10^18)`: argument radius 0.054) | `enclose_real` returns why there is no enclosure, `NoBall::Never` or `NoBall::TooWide` (a domain or guard check failed on a ball that still meets the domain); both tests go on to the next precision on `TooWide`. The classification decides nothing by itself (a `NonZero` still needs a ball that excludes 0), so a wrong class only costs a higher precision's charge or gives up early, and the extra cost is at most the three charges that an undecided constant already pays | §2.2 rule 5, §2.3, §2.16, §4.1 `ball_too_wide_goes_on`, §6 step 2 | the seven constants of the red team's table are decided, each coefficient equal to gp's series (1e-60 relative at 300 digits): each `Ok` in under 0.01 s with at most 1.6e4 units through `taylor_with_budget`, `series_expansion` and the limit path (the limit path `Unknown`: the constant terms are not rational); `x^(log(PQ - e) + 84)` is `NotLaurent`; `(x + tan(10^18) + 1)^200` at order 5 now takes Miller's path, 3.58e6 units (revision 7rt1: 1.84e7 by the binomial path). The exact zeros `sin(1)^2 + cos(1)^2 - 1`, `exp(log(3)) - 3`, `(sin(4)^2)^(1/2) + sin(4)` stay `UndecidedZero` |
| numeric m2 (below the bar, semantics): a float literal is decided by its exact binary value, so `0.1 - 1/10` is nonzero | Kept, and stated (§2.2): a float literal denotes its exact binary value, as at HEAD; the `taylor` doc comment and the REPL help say so | §2.2, §4.1 `float_literal_exact_value` | `1/(x + 0.1 - 1/10)` `Ok` with c0 = `1/(0.1 - 1/10)`; `0.5 - 1/2` `UndecidedZero` |
| memory m1 (below the bar, a stated figure exceeded): the rendering of long monomials with exponents other than +-1 holds 2.55 to 2.61 bytes per unit, VmHWM 261 MB, above the stated 2.60 and 256 MB | The claim is now the bound of the table "Memory" (every object at most 3 bytes per unit, so at most about 300 MB plus the parsed input at 1e8 units), with this family as the measured maximum; the family is in §5.3 and `EXP/rev7rt2/tools/memA_rt2.py` | §0 item 5, §2.7, §2.16, §3.1 (`TAYLOR_WORK` comment) | `(y0^2*...*y7999^2)/(1-x)` at order 107: 253 MB of heap at 9.94e7 units, 2.55 bytes per unit, VmHWM 261 MB, 0.20 s; at order 200 refused at 1.79e8 in 0.03 s; N = 2000 at order 200: 2.60 bytes per unit (117 MB); the REPL line at order 100: 0.46 s, VmHWM 265 MB (the same as proto9: no change to the engine's memory) |
| structure m1 (below the bar, a Display round trip): `Neg(Rational(p/q))` printed `-p/q`, which §2.12 reads as `(-p)/q` under `^` or as a divisor (`(pi ^ -2/3)` read back as 0.0338 instead of 0.4662) | `rule_neg` also parenthesizes a `Rational` with denominator above 1, whatever its sign: `-(2/3)`. By operator and operand it was the only pair the rules missed (every binary node is in parentheses, and `+`, `-`, `*` give the same value either way) | §2.11, §4.3 `test_display_negated_fraction`, `test_display_round_trip_values` | the four inputs print `(pi ^ -(2/3))`, `(3 / -(1/2))`, `(y ^ -(1/2))`, `(2 ^ -(1/3))` and read back as printed; gp reading the text: maxdiff 0 (revision 7rt1: 0.42 to 0.51). Random round trip (`tools/roundtrip_rt2.py`, 8 seeds of the red team's generator, the value of each tree against the value of the tree read back by the §2.12 parser, by gp): 551 inputs, 954 trees (the input and the result of each): 944 with the value of the tree read back, 0 different, 10 that gp cannot evaluate (7 not defined at the point, 3 whose written tree of 1 to 9 MB overflows gp's stack); with proto9's Display 78 different |
| structure m2 (below the bar, a false claim): §2.15 claimed that `PartialEq` handles the deepest result of any text the parser reads on the same thread, but a kernel made from an input node takes the input's depth, and a left-deep chain inside a call gives a result as deep as the input | The claim is dropped. §2.15 states the bound of a result's depth (the larger of the input's depth and `2 D + 2`, plus the rendering's levels) and the measured limits of `PartialEq`, and §7 item 5 says that it is not claimed for every result of a parsed text | §2.15, §7 item 5 | `x + sin(1+...+1)` with 20 000 ones on 2 MB debug: `Ok` in 77 ms, result depth 20 003; `PartialEq` of that result aborts there (the input is as deep); with 5000 ones it compares (in 0.9 ms) |
| structure m3 (below the bar, an honest error where HEAD answers): an exact monomial to a power above i64 was the overflow error although the series is `O(h^P)`; `exp(x)^(2^100)` kept a kernel `1 ^ N` | `pow_int` returns `O(h^P)` for an exact monomial whose exponent n v > 0 does not fit (only a pole of that order stays the overflow error), and `pow_const` folds `1^n` and `(-1)^n` for any n | §2.2, §2.4 item 2, §4.1 `exact_monomial_huge_power` | `x^(2^100) + x`, `x^(2^63) + x`, `(2*x)^(2^100) + x`, `x^(2^100)*sin(x) + x` at 0, order 1: `x` (the controls `x^(2^62) + x` and `(x+x^2)^(2^100) + x` as before); `exp(x)^(2^100)`: `1 + 2^100 x`, no kernel `1 ^ N`; `(1+0*x)^(2^100) + x`: `1 + x`; `(-1+0*x)^(2^100+1) + x`: `-1 + x`; `(1)/x^(2^100) + x`: `PrecisionExhausted` (the pole does not fit) |

Also changed while revising (each is in the body):

- **The records of the work stack are charged.** The recursion of revision 7rt1 used native stack,
  which the budget did not see; the boxed records of revision 7rt2 are heap objects, so each is charged
  `RECORD_UNITS = 32` when it is pushed (with the node visit's 16 units: 48 units for at most 128 bytes),
  and the table "Memory" has their row (`bin/sizes_rel` prints the sizes). The calibration of §5.3 moved
  by at most 4.3 %.
- **The probes of red team 2** are part of the prototype: `probes/rt2.rs` (the §2.12 parser probe,
  linked against proto10, with `ROUNDTRIP`), `probes/rt2_olddisp.rs` (the same with proto9's Display, the
  control of the round trip), `checks/coefcheck.py` (every stored coefficient against gp), the scripts
  `tools/mixdepth_rt2.py`, `tools/roundtrip_rt2.py`, `tools/memA_rt2.py`, `tools/mr.py`,
  `tools/gpcheck_rt2.py`, and their inputs in `EXP/rev7rt2/inputs/`.
- **`main.rs`**: `ball` prints the reason of a missing enclosure, and `coeffs` with `DISPLAY=iter`
  writes each coefficient with the rules of §2.11 (gp misreads HEAD's `(a ^ 1/3)`).

Where I did not take the red team's literal wording:

- H1 offered an iterative engine or smaller frames with a margin of 1.5 over the parser. I took the
  iterative engine: frame sizes depend on the compiler and the optimization level, so a margin would be
  a property to re-measure on every toolchain, and it would still leave a recursion that the next
  grammar level could outgrow. The work stack removes the dependence on the input's shape.
- S-B2 offered an error or the heap merge for a span that does not fit; I took the heap merge (the
  result fits, HEAD answers), and the overflow error only where a needed exponent does not fit.
- S-B3 offered `EXACT` when `n * deg(u') < S`; I return precision S in that case too, as Miller's path
  always did. An exact claim there would save one retry at most, and a single rule is easier to keep.
- numeric m2 offered a radius of half an ulp for a float leaf; I kept the exact binary value (HEAD's
  semantics, exact and documented) and stated it.
- structure m1 asked for a round trip over random trees; the crate test compares values by the ball
  code of §2.3 (the prototype's check uses gp), since a text round trip of `(a * 1/3)` gives another
  tree with the same value.

## Revision 7, red-team round 1

Reports: `fu2/design/S1-symbolic.redteam1-memory.md`, `-numeric.md`, `-series.md` and `-structure.md`
(four lenses on the revision 7 prototype `EXP/rev7/proto8/`). Four holes are over the review bar and
three are below it. Each is fixed in the body below, and each fix closes the class of its hole, not
only the measured input. The prototype is now `EXP/rev7rt1/proto9/` (then the block at the top), and every
reproduction of the four reports was rerun on it (`EXP/rev7rt1/tools/redteam1_rt1.sh`, output
`EXP/rev7rt1/redteam1_rt1.txt`; the numbers below are from that run unless stated). Every other check
of revision 7 was rerun too (`EXP/rev7rt1/tools/runall_rt1.sh`, §5.3). The facts were rechecked at
HEAD 2e15d1c: nothing that S1 reads has changed since revision 7. Numbers are release on an 8 MB thread,
or debug on a 2 MB thread, budget 1e8, unless stated.

| hole | change | where | evidence (proto9) |
|---|---|---|---|
| memory H1 (over the bar): a REPL `taylor` line spends 32 s writing the LaTeX of a result that both budgets admit. A deep kernel shared by every term (`(F)/(1-x)` with F = `f(f(...f(1)...))`, 33 000 calls, 99 KB, order 100) prints 10 MB of text, and HEAD's `expr_to_latex` formats each node from its children's finished strings, so it copies each subtree's text once per ancestor: O(text x depth) | The class is "a writer of an `Expr`'s text whose time is not linear in the text". The REPL's `expr_to_latex` becomes a work-list writer into one `String` (no `format!` of child strings, no `join`, no recursion), with the same output. Display, the other writer the REPL calls, is a work-list writer too (H2). J1b's `display_size` estimates the Display text and stops past `OUTPUT_BYTES`; the LaTeX of a node is at most about 3.4 times its estimate, so the display budget now bounds the time to print, not only the text | §2.7, §2.11, §3.1, §3.4, §4.5, §6 step 10 | `fchain_33000` at order 100 (the red team's input): `taylor_with_budget` 0.07 s and 1.12e7 units, `display_size` 16.7 MB (printed) in 0.04 s, text 10.0 MB in 0.09 s, LaTeX 33.3 MB in 0.09 s (HEAD's writer 31.9 s), the whole line 0.33 s, VmHWM 139 MB. Debug, the same chain built with the `Expr` API on a 256 MiB thread: 1.17 s in all (HEAD's writer 35.9 s); depth 8000 at order 400 on the REPL's 64 MiB debug thread: 0.99 s. The work-list writers give HEAD's strings byte for byte on 368 expressions (`displaycheck.rs`: the 107 inputs of the gp oracle, their points and their results, and every node kind) and on the red team's two `CHECKSAME` inputs |
| structure H1 (over the bar): right-deep `^` chains overflow the engine on text that HEAD parses and handles. `konst_of` and `expand_pow` recursed once per `^`, while the parser's `^` rule costs less stack per level: `x+1^1^...^1` with 400 carets (804 bytes) aborted a 2 MB debug thread, 12 000 carets an 8 MB release thread, where HEAD answers | `konst_of` (`konst_pow_chain`) and `expand` (`expand_pow_chain`) read a tower `b_1^(b_2^(...^t))` along its right spine by a loop, with any number of negations between the levels (`x^-x^-x`, which §2.12 reads). The bases are converted left to right, then the top t, then the powers are folded from the inside out, each kernel with its own input node: the order of operations and the charges of the recursive reading. Only the bases and t recurse, and in text a nested base or top needs parentheses or a call. The class rule, now fact 3 of §2.16: the engine recurses only where the text nests through parentheses or a call; every other rule of the parser (sign runs, `+ -` chains, `* /` chains, the `^` rule) has an engine loop | §2.2, §2.5, §2.16, §4.1, §6 steps 3, 4 and 8 | `tow1_400`, debug 2 MB, parse and call on one thread: `Ok` in 2 ms, 46 060 units, through `taylor_with_budget`, `series_expansion` and the limit path (`Finite(1)`; gp `x + 1`); HEAD 0.4 ms. 12 000 to 22 000 carets on 8 MB release: `Ok` in at most 10 ms with at most 18 MB. 13 000 carets on the REPL's 64 MiB debug thread: `Ok` in 42 ms. With the parse on a 1 GiB thread, the engine handles 50 000 levels (100 KB) of `x+1^...^1`, `x^2^...^2` at 1, `x+sin(1)^...`, `x^x^...^x` at 1 and `x+2^-2^-...` on a 2 MB debug thread (at most 0.70 s) and an 8 MB release thread (at most 0.15 s), and `x^-x^-...^-x` at 1 built with the API on 2 MB debug (0.86 s). On one thread HEAD's parser stops first on every tower shape, at 1438 carets on 2 MB debug (1427 for `sin(1)^...`) and 22 695 on 8 MB release |
| structure H2 (over the bar, if printing counts): results twice as deep as their input abort Display where HEAD's result prints. The value of `x^x^...^x` at 2 is a computed kernel `exp(log(2) K)` per level, depth 2 D + 2, and with H1 fixed the parser reads D = 22 695 on 8 MB release, so the claim of §2.15 that any result prints on the thread where its input parses was false | Display for `Expr` (and `Debug`, which calls it) becomes a work-list writer (§2.11), so printing needs no stack per level of the result, whatever its depth; the REPL's LaTeX writer likewise (memory H1). The kernel-depth cap stays `max(1024, 4 D)`. §2.15 now names the consumers that still recurse (the derived `PartialEq`, `substitute`, `simplify`), gives the measured limits of `PartialEq` against the deepest results a parsed text gives, and drops the false claim | §2.8, §2.11, §2.15, §2.16, §3.2, §4.1, §4.3, §7 item 5 | `towx_18000` at 2, order 0: `Ok` in 60 ms, 3.32e7 units, kernel and result depth 36 002, prints 288 KB in 3 ms (HEAD's recursive Display aborts on the same result, the control); 20 000 carets: depth 40 002, prints in 3 ms; 4500 and 5500 carets on a 2 MB release thread print. The tower with 22 695 carets, the deepest a text gives on 8 MB release, is parsed, expanded (94 ms, depth 45 392), printed (363 KB in 4 ms) and compared with `PartialEq` (1.5 ms) on one thread; on 2 MB debug `PartialEq` compares depth 5892, and the deepest result of a text that parses there (1438 carets, depth 2878) is parsed, expanded, printed and compared on one thread (`partialeq_rt1.txt`). gp: `x^(x^x)` at 2 has c0 = 16 and c1 = 107.1104124466... = 16 (4 log 2 + 4 log^2 2 + 2), the prototype's coefficients |
| structure H3 (over the bar): left-deep `/` chains, which §2.12's parser reads by its loop, overflow the engine. `expand` read `Div(l, r)` as `expand(l)` per `/`, and `konst_of` likewise: `x + (1/2/.../2)` aborted a 2 MB debug thread at 400 slashes (0.8 KB) | `expand_product` and `konst_of` (`konst_product_chain`) read `*` and `/` as one chain along the left spine, with a flag per factor. A constant divisor is inverted as it is read (`inverse_src(false, its node)`, so a zero divisor is `DivisionByZero` before a zero factor is seen); a series divisor is inverted (strict) and multiplied in order (`chain_step`). That is the loop of the `*` chain, so no recursion per factor | §2.2, §2.5, §2.16, §4.1, §6 steps 3 and 4 | `x/2/.../2`, `x + (1/2/.../2)`, `1/x/.../x` at 1 and `(x + sin(1))/sin(1)/...` with 50 000 divisions (100 KB), and `x*3/2*3/2...` with 25 000 pairs: `Ok` on 2 MB debug in at most 0.54 s and on 8 MB release in at most 0.09 s, at most 50 MB; through `series_expansion` and the limit path the same (the limit of `1/x^50000` at 1, order 5, is refused by the budget). Values: `x/2^10` is `1/1024 x`, `x*(3/2)^10` is `59049/1024 x` (gp `1/2^10`, `(3/2)^10`); `1/x^50000` at 1 is `1 - 50000 (x - 1)`. HEAD's parse of `x+(1/2/.../2)` with 12 000 slashes: 3 ms |
| numeric m1 (below the bar): sin near +-1 and log near 1 run 4 to 18 times slower than the scan table said. MPFR's sin computes `sqrt(1 - cos^2)` and doubles its precision when the result rounds to +-1; log adds the cancellation near 1 | sin is charged as an operation at `3p + 64` bits, `((3p + 64)/64 + 1)^2 * 8` units (cos keeps `2p + 64`: it has no such path). The scan adds the arguments where MPFR raises its precision (`nearscan.rs`: `round_p((k + 1/2) pi)` and `round_p(k pi)` for sin, cos and tan; `1 +- 2^-k` for log and acosh; `L 2^-k` with L = 3/7 for asinh, atanh, sinh and tanh; k = p/2, p - 2, p - 1 and 1022), timed as medians of 200 calls; the table of §2.3 is corrected | §2.3, §2.16, §6 step 2 | slowest median rates: sin 3.5e7, 3.4e7 and 6.2e7 units/s at 64, 256 and 1024 bits (with revision 7's charge 2.7e7, 1.6e7 and 4.2e7), log 2.6e7, 2.5e7 and 4.9e7, the rest at least 3.4e7. End to end, `(1)/(x + (sin(L) + 1)*(1+y1)*...*(1+y13))` with the 1023-bit convergent L of pi/2 (8192 groups) is refused at 1e8 after 0.79 s with 15 MB; with 4096 groups it is now refused after 0.72 s (revision 7: `UndecidedZero` after 0.93 s; both honest) |
| numeric m2 (below the bar): `literal_abs_bound` converted the whole literal to f64 (a copy and a division of both parts), uncharged, once per erf node per walk | `literal_abs_bound` returns bounds `lo <= abs(v) <= hi` from the bit lengths and the top 64 bits of the numerator and the denominator, in O(1). The lower bound decides the cheap regime (`lo >= X_p + 1`: `2u`), the upper bound the charge | §2.3, §2.16 | 34 ns for the 2 x 1M-bit rational `t/(t + 2)` (revision 7: 611 us); 210 rationals with numerators up to 2^2063: every exact value inside its bounds, every pair within a factor 1 + 1e-12 (`litbound.rs`) |
| series m3 (below the bar, a port trap): the binomial path of `pow_int` built a second context with `Work { used: 0, max: 0 }`, harmless only with the prototype's thread-local counter | The loop runs in the caller's context with the working precision set to the slots (`cx.p` saved and restored), so it charges the caller's `Work`; the port passes the same `&mut Work`. §6 names the trap at its new place | §2.4, §6 (port traps) | `(x + sin(1)^2 + cos(1)^2 - 1)^2000` at orders 1200 and 3000 and `^100000` at 1500 are refused at 1.0e8 in at most 0.17 s with at most 1.2 MB of heap, as before; `^70` at order 3, unbounded, gives the binomial expansion `c^70 + 70 c^69 x + 2415 c^68 x^2 + ...` with c the undecided constant |

Also changed while revising (each is in the body):

- **The frame of `konst_of`.** Its `+ -` chain, its `* /` chain and its `Function` arm are separate
  non-inlined functions (`konst_sum_chain`, `konst_product_chain`, `konst_function`), so the frame that
  recurses once per level of nested input holds no chain vectors: `x + sin^k(1)` reaches 710 levels
  on a 2 MB debug thread (revision 7: 395) and 32 729 on 8 MB release (11 636). The `/` chain now goes
  through `expand_product`, whose frame holds the item vectors: `1/(1 + 1/(1 + ... x))` reaches 342
  levels on 2 MB debug (revision 7: 357; HEAD's `taylor` 47, HEAD's parser 212). The depth table of
  §2.5 has the new values and a tower row.
- **A heap peak above the stated figure.** `L*((1)/(1-x))` with a 99 990-digit literal L holds 2.56
  bytes per unit (252 MB of heap at 9.84e7 units, VmHWM 256 MB, order 6000), in revision 7 as now: each
  stored coefficient is a new copy of L, charged `COPY_WORD` per word, and its rendering copies it again
  (`EXPR_WORD` per word), 2.67 bytes per unit by the row "digit vector" of §2.16. Revision 7 stated
  2.29 bytes per unit and 230 MB from families without this one. Since `/` joined the `*` chain,
  `(L)/(1-x)` and `(L)/((1-x)*x^k)` take the same path (revision 7 multiplied by an exact constant
  series and charged its pairs, so it refused `(L)/((1-x)*x^1000)` at 1.12e8 units; now `Ok` at 4.92e7
  with 126 MB), and `((L)/((1-x)*x^7000))*0` is `Ok` at 5.1e6 units (the zero factor of the chain
  makes the product 0 before L scales the quotient; revision 7 refused it at 1e8). `(L*x)/(1-x)^2` holds
  2.60. All are below the table's 3 bytes per unit and the bar's 512 MB; §0, §2.7, §2.16 and the
  `TAYLOR_WORK` comment now say at most 2.6 bytes per unit and 256 MB of VmHWM.
- **The probes of the red team** are part of the prototype (`split.rs`, `apishape.rs` with more shapes,
  `replshow.rs` with the work-list writers), with their scripts in `EXP/rev7rt1/tools/` and their inputs
  in `EXP/rev7rt1/inputs/`.
- **The notes of the series lens** (HEAD's parser reads `atanh(x)` as a named function and `x^3/6` as
  `x^(1/2)`): both are fixed by §2.12 (its table entries for the inverse hyperbolic functions, and `a/b`
  as an operator); no change.

Where I did not take the red team's literal wording:

- H2 offered three fixes: the kernel of a value rendered from the input node with the point
  substituted, a cap of `max(1024, D + 64)` on computed kernels, or an iterative Display. I took the
  iterative Display. The cap refuses towers that HEAD answers (`x^x^...^x` at 2 with 18 000 carets, 7 ms
  at HEAD). Substitution is not valid where a subexpression has a removable singularity at the point
  (`exp(sin(x)/x)` at 0 would print `sin(0)/0`), and it would be a second rendering rule for every
  computed kernel. A work-list Display removes the dependence on the depth for every result, whatever
  made it deep.
- Memory H1 offered a linear writer that recurses (`latex_into`) or a size estimate of text times
  depth; I took a work-list writer, which is linear and needs no stack per level, for the same reason
  as Display.
- m1 offered a charge at `3p + 64` bits or `sin x = cos(x - pi/2)`; I took the charge (one formula; the
  identity needs a pi ball at the argument's exponent for every sin).
- H1 left the parser's `^` rule as it is (HEAD's nesting, §2.12). The engine now reads a tower by a
  loop, so the parser's recursion per `^` can only stop a text before the engine would.

## Revision 7 (response to review 7)

Review: `fu2/design/S1-symbolic.review7.md`. Its two blockers and four minor issues are fixed in the
body below, with the orchestrator's items: the families of reviews 3 to 7 rerun on the updated
prototype (§5.3, "Revision 7 probes"), the "Invariants" and caps tables kept current (§2.15, §2.16),
and the first lines of this file naming the prototype and how to time one call through each public
entry point. The facts were rechecked at HEAD f29c492 (MF1 after the tracker commit 0c2f368; it changes
rustmath-manifolds and two comments of `repl.rs`, so no cited line moves) and 2e15d1c (a tracker
commit, docs only; its HANDOFF note from MF1 for S1 is answered in §3.4). The prototype is
`EXP/rev7/proto8/`: revision 6's `proto7` plus the changes below, and a public layer `api.rs` (§2.9)
through which every driver now runs, so every figure of this revision is measured through
`taylor_with_budget`, `series_expansion` with `to_expr`, or the `limit_via_series` path. Numbers are
release, budget 1e8, point 0, order 1, unless stated; memory is the heap peak of the whole call from the
counting allocator (`--cfg memcount`) and VmHWM. The machine was shared with a workspace test run
during the scans, so their rates are lower bounds.

| issue | change | where | evidence |
|---|---|---|---|
| B1 erf charged like a cheap function | Two regimes with the threshold `X_p` (7.5, 14 and 27 at 64, 256, 1024 bits, `X_p^2 >= (p + 16) ln 2`). Above it, the cheap rigorous enclosure `+-1 +- 2^-(p+16)` (Abramowitz and Stegun 7.1.13: `erfc(x) < exp(-x^2)/(x sqrt(pi))`), no MPFR call; a ball across `+-X_p` gets `[-1, 1]`. Below it, MPFR's erf, charged a priori by `erf_units(p, a) = 2u (8 + ceil(min(a, X_p)^2))`, where `a` is the bound of a literal argument (through at most 4 negations) and `X_p` for any other argument; a literal at or above `X_p + 1` is charged `2u` | §2.3, §2.7, §2.16 | review 7's family: `erf(975/38) + erf(-975/38)` in 3700 groups (12.0 KB) refused after 0.15 s release, 0.21 s debug (revision 6: `Ok` after 32.1 s); 3600 groups refused after 0.15 s; 100 groups with `erf(26)`, `975/38`, `960/38`, `586/25`, `280/11` refused within 0.17 s; 3 groups `Ok` in 0.03 s |
| B1 item 2: scan of every enclosed function | `fnscan.rs` times every operation of the ball code (18 unary functions, `+ * /`, integer and real powers, integer and rational leaves, pi, e) at 64, 256 and 1024 bits over a grid with long mantissas, rationals and values just inside each guard; `erfbins.rs` adds 300 random arguments per half-unit bin of `[0, X_p + 1)` for erf. The table of §2.3 gives the slowest rate of each | §2.3 | over three runs, slowest erf rate 4.0e7, 8.9e7 and 1.4e8 units/s at 64, 256 and 1024 bits; slowest of all operations 2.1e7 (atan at 64 bits, 6.1 us for 128 units), above the floor of 1e7 that `TAYLOR_WORK` assumes; the longest single erf call 6.2 ms (`erf(24.6)` at 1024 bits, charged 2.84e6); one preempted scan cell checked by 1000 repeated calls (`onecall_r7.txt`) |
| B1 item 3: test | `budget_erf_groups` | §4.1 | 0.21 s, 0.04 s, 2 ms, 0.2 ms debug |
| B2 owned coefficients in `SeriesExpansion` | `terms: Vec<(i64, Arc<Expr>)>` holds the `Arc` each coefficient's rendering built (a bare kernel is the kernel's own `Arc`), h is built once, `to_expr()` builds from the `Arc`s (one increment per term and the `RENDER_TERM` nodes), `coefficient(k)` returns the `Arc`, `Debug` is written by hand. The only copies left are once per call: the header's `var` and `point`, the point inside h, and the top node when the result is one shared coefficient | §2.8, §2.9, §2.16 | `(Y)/((1-x)*x^3000)` at order 2000, 100 KB: 2 MB of heap through `taylor_with_budget` and through `series_expansion` + `to_expr` (review 7's model of revision 6: 983 MB); `(Y)/((1-x)*x^10000)`: 5 MB (2.35 GB); `Y/(1-x)`: 1 MB (396 MB); the limit path of each 3 MB or less (`EXP/rev7/b2probe_r7.txt`) |
| B2 item 3: test | `series_expansion_shares_kernels`: the 5001 leaves Y of the result are one allocation, and every stored coefficient is `Arc::ptr_eq` to `coefficient(-3000)` | §4.1 | `EXP/rev7/proto8/sharecheck.rs`: 0.02 s release, 0.06 s debug |
| m1 `TAYLOR_ORDER` 20000 to 2000 | Kept at HEAD's 20000; its doc comment says what it bounds | §0, §3.1, §4.1, §4.5, §5.1, §6 | at 20000: `x + 1` `Ok` with 860 units (HEAD: 0.48 s, 324 MB); `1/(1-x)` `Ok`, 1.68e7 units, 12 MB; `exp(x)` and `sin(x)` refused after 0.14 s and 0.12 s (0.28 s and 0.24 s debug) |
| m2 two HANDOFF notes | §1.8 states both. The Drop half of the `w + w + ... + w` note is fixed by §2.14, the Display half stays out of scope (§7 item 5); `-(-x)` now displays `-(-x)` (§2.11) | §1.8, §2.11, §3.2, §4.5, §7 | `--x` also parses back to x under §2.12 |
| m3 parser depth measured with HEAD's parser | §6 step 8 measures the new parser's limits on the six shapes of §2.5 (2 MB debug, 8 MB release) and checks `taylor_with_budget` at each; if the new parser reads deeper than the engine expands, it gets a nesting limit below the engine's and not below HEAD's | §2.15, §6 | HEAD's parser on the prototype: 160 to 212 levels on 2 MB debug, 2048 to 4096 on 8 MB release; the engine on API-built inputs 319 to 1298 and 10 908 to 40 282 (`EXP/rev7/depthcmp7_*.txt`, `enginedepth_r7.txt`) |
| m4 `q_mul` with a one-word operand | `q_mul`, `q_add`, `q_sub` and `i_mul` charge at least `COPY_WORD` per word of their result (a `max`, so only lopsided operations change) | §2.7, §2.16 | `q_mul(7, L)` with L of 100 words: 309 units (revision 6: 266); calibration: 68 of 70 rows within 0.5 % of revision 6, `(1+x)^10000` at 2000 +1.6 %, `sin(2^340000 + k)` +0.9 %; `(L)/((1-x)*x^1000)` with a 99 990-digit L, which review 7 measured `Ok` at 251 MB, is now refused at 1.12e8 after 0.15 s with 126 MB |
| orchestrator: reproduction families of reviews 3 to 7 | `EXP/rev7/tools/runall7.sh`: review 3's `repro_blockers.sh`, review 4's `repro4.py`, review 5's `repro_r4.py`, `longmono_rate.py`, `radrem.py`, `mem_mono2.py`, `mem_mono3.py`, `probe.py`, `probe2.py`, review 6's inputs and families, review 7's `fam_r6.py`, `gen7.py` and inputs, plus the self-attack scripts of revisions 5 and 6 and the oracles | §5.3 | about 760 runs, plus about 500 in the binary searches of the depth probes, three times: no budgeted call longer than 2.8 s or above 230 MB, no stack overflow in the engine (the two overflows are HEAD's parser, as in revision 6); oracles 107 OK, 26 + 18 checks, 31 budget inputs |

Also changed while revising (each is in the body):

- **`BALL_NODE` in the text.** §2.3 and §2.7 said `BALL_NODE = 32`; the revision 6 prototype and the
  memory table use 64. The text now says 64.
- **A heap peak above the stated figure.** Measured through the public layer, `x + S_19*sin(x)` (the
  largest admitted product of the caps table) peaks at 2.29 bytes per unit (227 MB at 9.92e7 units);
  §0, §2.7 and §2.16 said 2.21. They now say 2.29, still below the 3 bytes per unit of the table
  "Memory", and VmHWM at most 230 MB.
- **The name an input kernel copies.** A kernel made from an input symbol or function node takes a
  shallow clone of the node, which copies its name: a 99 990-character Y was a 100 KB copy for 16
  units (bounded by the input, since each distinct name gives one kernel, but outside the table
  "Memory"). It is now charged `COPY_WORD` per 8 bytes of the payload (`node_payload_units`, §2.7);
  `taylor(Y, x, 0, 5)` holds 2.2 bytes per unit, the uncharged copy of the returned top node included.
- **Two stale figures.** `(1+x)^(10^100)` at order 1150 is charged 9.55e7 units (m4), not 9.22e7; the
  `TAYLOR_ORDER` comment keeps order 1150 (at 1200 it is refused at 1.04e8). The order-20000 figures of
  §5.1 come from this revision.

Where I did not take the reviewer's literal wording:

- B1 offered an argument-dependent charge or a cheap enclosure for large arguments; I did both. The
  cheap enclosure alone leaves the cost of MPFR below the threshold unbounded by the old `2u`
  (`erf(6.5)` at 64 bits: 19 to 74 us for 96 units), and a charge alone would have to grow without
  bound as the argument approaches the point where MPFR stops being expensive. The suggested
  `u (2 + a^2)`, and twice it, left erf at 1.2e7 and 2.1e7 units/s at 64 bits for small arguments,
  where MPFR's Ziv loop retries (`erf(18/13)`: 15.5 us for 192 and 320 units), so the charge is
  `2u (8 + a^2)`. The argument bound is
  read a priori from a literal argument only, so `enclose_real` stays a function of `Expr` alone and
  `ball_cost` stays an a-priori walk; a computed argument is charged at `X_p`.
- B2 offered shared `Arc` coefficients or a charge per copied top node; I took the shared `Arc`s,
  which leaves nothing per term to charge.
- m2 offered to state the two notes or to fix the Display of a negated negation; I did both (the fix
  is one rule of §2.11).
- m4 offered to correct the table row or to charge `COPY_WORD` per result word; I charged it, as a
  floor, so the "at most 3 bytes per unit" of the table holds.

## Revision 6 (response to review 6)

Review: `fu2/design/S1-symbolic.review6.md`. Its three blockers and five minor issues are fixed in the
body below. The orchestrator's brief asked again to close the classes behind reviews 3 to 6: the
caps (§2.15, now with the error of each cap and the walks over kernel expressions), review 5's B1, B2,
m1 and m2 (kept and rechecked), the invariants (§2.16, now with a table "Memory" of every kind of
heap object and its units), and a self-attack at the bar (§5.3, "Revision 6 probes"). The facts were
rechecked at HEAD 0c2f368: 92906d4 (the HEAD of review 6) plus the tracker commit of part 2, which
changes only `PROJECT_TRACKER.md` and `docs/` (the crates S1 reads are unchanged since revision 5).
That commit adds one item to the S1 bullet of `docs/HANDOFF.md`, expand under a minus, which this
revision covers (§1.7, §2.17, §4.6). The prototype is `EXP/rev6/proto7/` (revision 5's `proto6`
plus every fix below), built with `EXP/rev6/build.sh` against the same rlibs; `EXP/rev6/base6/` is
revision 5's code with the counting driver, for before/after runs. Numbers are release, 8 MB main
thread, budget 1e8, unless they say debug (2 MB thread). Memory is the heap peak of the expansion and
its rendering, in malloc chunks, from the counting allocator `EXP/rev6/proto7/memcount.rs` (built
with `--cfg memcount`), and VmHWM. `fz`, `two`, `negz`, `sparseg` and the other families are review
6's generators, in `EXP/rev6/tools/gen6.py`.

| issue | change | where | evidence |
|---|---|---|---|
| B1 sparse series product not charged | The sparse path of the series product is a heap merge of the rows of the shorter operand (Johnson 1974): the products come out in increasing exponent, so the only containers are a heap of one cursor per row and the output. Charged before it runs: `SER_TERM + SEARCH_CMP (bits(rows) + 1)` per pair, 16 per row. The same class in `sum_series` (a `BTreeMap` of vectors per exponent, 176 bytes per exponent for 4 units per term): a heap merge too, charged per term and 24 per series | §2.4, §2.5, §2.7, §2.16 | `fz(64, 186)` (4.6 KB): refused after 0.06 s with 8 MB (revision 5: `Ok` after 15.5 s with 2.53 GB); `fz(64, 32)`: refused after 0.03 s (452 MB); `fz(64, 8)`: `Ok` in 0.12 s with 40 MB, 0.66 bytes per unit |
| B1 test | `budget_sparse_products` | §4.1 | 0.76 s and 0.97 s debug |
| B2 small constants | A constant of several terms is a sorted boxed slice (88 bytes per term, no node slack), built by one constructor, `from_sorted`, which charges `CONST_BASE = 72` for the header before it allocates it; 0 and 1 are shared. A term copy is charged `52 + 6 L + 3 w` (`COPY_TERM`, `COPY_FACTOR`, `COPY_WORD`; revision 5: 32 + 4 L + 2 w), and a one-term constant made by `rat` or `kernel` also its digit vectors and monomial. Every operation builds its result in order: `merge2` for `add` and `sub`, maps for `neg` and `scale`, a heap merge for `sum_many`; the product keeps a transient map accumulator, charged `ACC_ENTRY = 32` per pair and converted to the slice in one pass | §2.2, §2.7, §2.16 | `two(64, 4)`: refused after 0.21 s with 58 MB (revision 5: `Ok` with 1.35 GB, 13.8 bytes per unit); `negz(64, 8)`: refused after 0.31 s with 102 MB (574 MB); 3 to 16 terms per coefficient (`kterm`): at most 1.5 bytes per unit |
| B2 test | `budget_small_constants` | §4.1 | 1.46 s, 1.69 s, 1.03 s, 0.22 s debug |
| B3 rendering charged the coefficients only | The driver charges `render_units` before the result is built: `2 EXPR_NODE` for h, then per stored term its coefficient's `expr_units` and `RENDER_TERM = 4 EXPR_NODE + 2 EXPR_INT + SER_TERM = 156`. h is one shared node (revision 5 cloned x, a `String`, per term). `EXPR_NODE = 32` (an `Arc<Expr>` is a 96-byte chunk), with the node bound `2 + L + 2 pf` per term (revision 5: 20 for `3 + 2 L`, nothing for the exponent nodes) | §2.7, §2.8, §2.9 | `sparseg(60, 4)` (1.4 KB): refused after 0.12 s with 35 MB (revision 5: `Ok` with 743 MB, 7.5 bytes per unit); `sparseg(60, 1)`: `Ok` in 0.10 s with 126 MB, 1.66 bytes per unit |
| B3 test | `budget_rendering` | §4.1 | 0.83 s, 0.21 s, 0.07 s debug |
| B1 to B3, bytes per unit | Every kind of heap object has a row in the table "Memory" of §2.16, at most 3 bytes per unit with glibc's chunk overhead, and the `TAYLOR_WORK` doc comment gives the measured figures. The memory scripts now use the counting allocator, which measures live heap bytes independently of fragmentation | §2.7, §2.16, §3.1, §5.3 | heap peak at most 2.21 bytes per unit on every family of §5.3, at most 1.9 on review 6's families; VmHWM at most 183 MB on every budgeted probe (`fam6_final.txt`, `mem6_final.txt`, `m2probe_final.txt`, `review6_inputs_final.txt`) |
| m1 kernel depth reached by a cheap constant | `inverse_src`: the kernel `1/P0` of a divisor that the input names (the node r of `l / r`, the constant divisor of the series `Div`, the base of `c^n` with n < 0) takes `r / c` as its expression, so its depth is the input's. §2.15 states the remaining case, strict inverses of computed constants, and measures it | §2.2, §2.5, §2.15 | `invpow(80)` without a budget: `Ok` in 2.8 s, kernel depth 241, result depth 243, the input's depth (revision 5: `LimitExceeded(KernelDepth)` after 19.3 s; HEAD: depth 248). `invpowx`: the budget comes first (0.31 s), without one the term cap (2.5 s) |
| m1 test | `input_divisor_kernels` | §4.1 | 4.2 s and 1.4 s debug |
| m2 outcome depends on the tree shape | `product_many` returns 0 before any round when a factor is 0; the test runs both splits | §2.2, §4.1 | `x + Q_1400*(S_15)*0`: `Ok` with both splits in 0.03 s; `x + Q_1400*(S_15)`: refused at 1.29e9 and 2.43e9 with both |
| m3 rate of the B1 family | fixed by the B1 charge | §5.3 | refused within 0.06 s |
| m4 pair weight | §2.4 and §2.7 state the prototype's rule: `wa + wb` when either coefficient is one word, else `wa wb` | §2.4, §2.7 | |
| m5 the `scale` row | The series `scale` now charges `16 + SER_TERM n` itself (a product by a coefficient 1 is the shared constant, free), and the row says so | §2.16 | |
| caps (orchestrator item 1) | Values unchanged: factors of one monomial 2^15, kernels 2^16, terms of one constant 2^20, kernel depth max(1024, 4 D), each `LimitExceeded` with its `StructureLimit`. The table gives the error, the margins on the revision 6 probes and the measured refusals; a paragraph lists every walk over a kernel expression (none recurses) | §2.15 | margin at least 2.5 on every cap |
| review 5 B1, B2, m1, m2 (item 2) | Kept: balanced monomial products and the iterative `enclose_real`; the monomial and search charges (`MONO_FACTOR_UNITS` raised from 5 to 6 to match the memory table); the DAG pre-pass and `KONST_NODE`; the recorded point | §2.2, §2.3, §2.5, §2.7 | the long, DAG and point probes of §5.3 rerun on revision 6 |
| invariants (item 3) | §2.16 rewritten against `proto7`, function by function, and the table "Memory" | §2.16 | |
| HANDOFF item added after review 6 | expand under a minus: `distribute_negation` (`expand.rs:250-263`) becomes `-(a + b) = (-a) + (-b)` and `-(a - b) = (-a) + b`; the empty `test_expand_negation` gets assertions, and a REPL test is added | §1.7, §2.17, §4.6, §6 step 6 | `EXP/rev6/probes/expandneg.rs`: HEAD gives -4/7, 4/7, 34/35, -36/35, 16/49 on the five inputs of §1.7, the fix 4/7, -10/7, -64/35, 34/35, 40/49, which are gp's |
| self-attack (item 4) | About 560 runs: the families of reviews 3 to 6 and new ones (long names, literal chains, sums of 12 000 series, numeric tests of large constants, computed inverses); every probe is in §5.3 | §5.3 | slowest budgeted call 2.88 s (2000 64th roots); largest VmHWM 183 MB; no stack overflow in the engine (two in HEAD's parser, which HEAD shares) |

Also found and fixed while attacking the prototype (each is in the body):

- **Names copied into every monomial.** The rendering cloned each kernel's top node into the product,
  and a `Symbol` node owns its name's `String`. `(1 + Y*x + Z*x)^300` with names of 40 000 characters
  (80 KB of input) held 3.68 GB at 2.9e7 units in revision 5 (127 bytes per unit). Kernel expressions
  are now `Arc<Expr>` and every rendering refers to them; the same input holds 49 MB.
- **Literal chains.** A literal was charged one unit per word for an 8-byte copy. With the `Expr` API,
  `x + L + ... + L` with 25 000 references to one 50 000-digit literal held 519 MB under the budget.
  A literal is now charged `COPY_WORD` per word (and a one-term constant its digit vectors):
  refused at 1e8 with 137 MB.
- **An uncharged walk in the zero test.** `const_status` walked the factors at every test to find
  rule 4 (only I and one named constant), uncharged; a constant at the point, tested again at every
  `1/x`, would have repeated it. The flag is now cached (`one_named_imag`), with `any_imag`, so
  `split_imag` and the grouping by indeterminates copy nothing when there is nothing to split or to
  group; the grouping itself is one sorted vector charged two copies (revision 5 allocated a vector
  of 4 slots per group: about 600 bytes per one-term group for 60 units).
- **Smaller ones.** The ball code charges `BALL_NODE = 64` per distinct node for its memo; a kernel key
  is allocated with its charged size as capacity; `reduce_mono` compacts in place; the vectors of a
  chain get exact capacities; the series `deriv`, `integrate` and the offsets of `inv` charge
  `SER_TERM` per term; the work functions that the per-kind functions of `expand` call are never
  inlined, so the new merge code adds nothing to a frame on the recursive path (the engine's depth on
  every API-built shape is at least revision 5's: §2.5, `EXP/rev6/enginedepth_r6.txt`).

Where I did not take the reviewer's literal wording:

- B1 offered a charged map or a sorted vector of pairs. I took a heap merge (Johnson 1974): it holds no
  container per pair at all, so its memory is the output, and it is charged per pair before it runs.
- B2 offered a charge per B-tree node or a sorted vector for at most 16 terms. I took the sorted
  slice for every size, which has no slack at any size, plus `CONST_BASE` per constant. The product
  keeps a transient B-tree accumulator, because it checks the term cap before each new entry (a
  vector of all pairs would hold every pair before it could know the result's size); its slots are
  charged `ACC_ENTRY`.
- B3 suggested `EXPR_NODE * 6 + 2 w(e) + 16` per term. With h shared and `EXPR_NODE = 32`, a term
  beyond its coefficient allocates at most 4 nodes and one digit vector, so `RENDER_TERM` charges 4
  nodes, two digit vectors and its slot.
- m1 offered two fixes; I did both: the input divisor's kernel takes the input node, and §2.15 states
  and measures the case that remains (computed inverses), which the budget and the term cap reach
  first.
- The orchestrator's item 2 allowed interned monomials or charges; revision 5's charges stay. Interned
  monomials would order the terms by creation, which changes every printed result.

## Revision 5 (response to review 5)

Review: `fu2/design/S1-symbolic.review5.md`. Both blockers and both minor issues are fixed in the body
below. The orchestrator's brief for this revision asked to close the classes behind reviews 3 to 5,
not only their instances. So the body also gains hard structural caps (§2.15), an invariant for every
loop, allocation and recursion of the engine (§2.16), and a self-attack of the prototype in eleven
input families (§5.3, "Revision 5 probes"). The prototype is `EXP/rev5/proto6/` (revision 4's `proto5`
plus every fix below), built against HEAD 92906d4 (see the top of this document). Numbers are release, 8 MB main thread,
budget 1e8 (the provisional `TAYLOR_WORK`), unless they say debug and 2 MB (a cargo test thread).
P_N is the balanced product of `sin(1)` to `sin(N)`, Q_M the same with M factors, S_k the product
`(1+cos(1))*...*(1+cos(k))` (or of `(1+sin(j))` where it says so), N_k review 4's nested
`(1+exp(...))^2`.

| issue | change | where | evidence |
|---|---|---|---|
| B1 left-deep monomials | `to_expr` builds the numerator and the denominator of a monomial as balanced products (`balanced_product`, the rule of `balanced_sum`), so a monomial of L factors has depth `ceil(log2 L) + 3`. `enclose_real` is iterative: a post-order walk with an explicit stack and the address memo; the first child without an enclosure ends the call (every child is needed by its parent). No other engine walk of a kernel expression recurses: keys are written from ids, `ball_cost` and `display_estimate` use explicit stacks (the second stops after about 34 nodes), and `balanced_sum` / `balanced_product` recurse `log2` of their length. Kernel depth is capped (§2.15) | §2.2, §2.3, §2.8, §2.15, §2.16 | debug, 2 MB thread, N = 4096 (48 KB): `x + P_4096` is `Ok` in 0.10 s with result depth 16, `format!` 56 KB; `1/(x + P_4096 - 1/2)` is `Ok`, depth 21 (revision 4: abort at N = 431 and 1975). Release, N = 8800 (104 KB): depth 17 and 22 (HEAD: 17) |
| B1 tests | `budget_long_monomials` (§4.1) | §4.1 | |
| B2 monomial work | A product charges `MONO_FACTOR_UNITS = 5` per factor of every product monomial (its 16 bytes of heap and the merge step), and its accumulator's searches by `search_units`: `SEARCH_CMP = 3` comparisons per level of the B-tree for every factor beyond `SEARCH_FREE = 16` of each operand's monomials. The same search charge applies to every other insertion into a term map (`add`, the new `sum_many`, the zero test's grouping and split). `scale` charges `copy_units` (it copies every monomial), and the new monomials of `inverse` and `pow_u` charge 5 per factor | §2.2, §2.7 | `x + Q_1400*(S_15)*0` is refused at 1.26e9 in 39 ms (revision 4: `Ok` after 13.3 s with 1.09 GB). `x + Q_1400*S_15*0` (one chain) is now `Ok` in 8 ms: its zero is a factor of the same chain and absorbs the product (§2.5). M = 2900, k = 14: the same |
| B2 memory | The memory per unit is remeasured on every family of §5.3 and on the self-attack: at most 2.9 bytes per unit (`x + (sum of sin(1..550))*(sum of cos(1..550))`, 219 MB at 7.5e7 units). Three changes were needed to get there: one-term constants are stored inline (revision 4 allocated a B-tree node of about 1 KB for each, a memory hole of revision 4); `EXPR_NODE` is 20 (16 gave 3.5 bytes per unit on rendering-heavy rows); a new kernel charges `KERNEL_UNITS = 128` before it is created (kernel-heavy inputs held 12 to 14 bytes per unit) | §2.2, §2.7, §3.1, §5.3 | revision 4 (`EXP/rev5/rev4cmp.txt`): `(1/(1-x))/x^300000` at 0, order 1 held 629 MB and `-(1/(1-x))/x^250000` 524 MB before their refusals at 1e8, above the 512 MB of the bar; now 80 MB and 113 MB |
| B2 test | `budget_monomial_products` (§4.1) | §4.1 | |
| m1 DAG inputs | The pre-pass is an explicit-stack walk that looks a node up before it descends, so it visits each distinct node once and does not recurse; it also records each node's depth. `expand` and `from_expr` charge `KONST_NODE = 16` per node visit before anything else (a literal also its words), so a DAG input is expanded once per occurrence and each occurrence is charged. The point is tested for the variable with the same memoized walk. The REPL arm finds its variable without a walk (`Symbol::new` is interned by name, `symbol.rs:40`) and checks that the expression mentions it with the DAG-safe `mentions_symbol` | §2.5, §2.7, §3.1 | k = 40 (`e = e + e` and four other shapes): `BudgetExceeded` after 0.11 to 0.45 s release, 1.0 to 3.3 s debug on a 2 MB thread (revision 4: did not return at k = 40) |
| m2 point rebuilt | `SeriesExpansion` records whether the point is zero (`point_is_zero`, set by the driver from the variable's series); `to_expr()` reads it | §2.8, §2.9 | |
| caps (orchestrator item 1) | Four hard caps, checked before the work they bound, each with the typed error `SeriesError::LimitExceeded`: factors of one monomial 2^15, distinct kernels of one expansion 2^16, terms of one constant 2^20, depth of a kernel expression max(1024, 4 x input depth). None is reached inside the zero test (it copies and evaluates, it builds no larger constant and no kernel) | §2.15 | every §4.1 and §5.3 input stays below each cap with a margin of at least 2.5; the inputs a cap refuses are listed in §2.15 |
| invariants (item 3) | Every loop, allocation and recursion of the prototype, by function, with the charge, cap or input bound that bounds it | §2.16 | |
| self-attack (item 4) | About 300 runs in eleven families (the families of reviews 3 to 5, and chains, DAG inputs, huge literals, deep nesting, many radicals, high orders, poles of high order, small coefficients); every result is in §5.3 | §5.3 | slowest admitted call 3.0 s (200 64th roots of 4096-bit radicands, 3.2e7 units/s); slowest refused 2.8 s; peak memory 250 MB; no stack overflow except in HEAD's own parser, which stops HEAD at the same depth. The attack found the memory hole of the "B2 memory" row and the holes below, all fixed |

Also found and fixed while attacking the prototype (each is in the body):

- **Quadratic chains.** `x + y1 + ... + y3000` (9 KB) was refused at 1e8: each left-deep `add` copied the
  growing constant, O(N^2) copies (revision 4 had the same). Sums and products are now read along the
  left spine of the input with a loop: the constant summands go into one merge (`sum_many`), the
  series into one k-way merge, and constant factors are multiplied in rounds of pairs
  (`product_many`, O(N log N) factor copies). 5000 symbols: `Ok` in 7 ms with 1.8e6 units. HEAD: 17 ms
  for `x + sin(1) + ... + sin(11000)`, the engine 31 ms. Neither the spine nor the pre-pass recurses
  now, so a left-deep sum no longer bounds the input depth (§7 item 5): 65 536 terms `x + ... + x` are
  `Ok` on an 8 MB release and a 2 MB debug thread (HEAD overflows at 12 800 and 190).
- **The pair count of a series product ran before any charge** (a budget hole of revision 4). It was
  a loop over all pairs, O(pairs), before the one charge. When a loss raises the working precision,
  two cheap series of P terms meet in one product: `((1/(1-x))^2)/x^250000` (31 characters) ran 34.7 s
  with 524 MB in revision 4 before its refusal at 6.3e10 units. The count now stops once its sum
  passes the remaining budget (every pair adds at least 2): refused after 0.21 s with 67 MB (debug,
  2 MB thread: 1.2 s).
- **The debug frame of `expand`** (a regression of revision 4). Revision 4's `expand` used about
  12 KB of a debug stack per level of the input: an input built with the `Expr` API,
  `sin(sin(...(x)))`, overflowed a 2 MB debug thread at 172 levels, where HEAD's `taylor` reaches 188.
  `expand` is now a thin dispatcher; each node kind's work is a separate non-inlined function, off the
  stack while the children expand. The same input now reaches 1280 levels (HEAD 188), and every shape
  of §2.5 reaches at least HEAD's depth on a 2 MB debug thread and on an 8 MB release thread.
- **Exponential text of input constants.** A kernel's expression was rendered from its arguments'
  normal forms, so `(1+exp(E))^2 = 1 + 2 exp(E) + exp(E)^2` named E twice and the text doubled per
  level: `x + N_300` (3.3 KB) had a result whose text has 2^300 nodes. A kernel made from an input node
  now takes that node as its expression (an O(1) clone; its children are shared), so its text is the
  input's text. `x + N_300` prints 10 KB in 0.14 ms (HEAD: 5 KB). Kernels the engine makes from computed
  values (values at a point, inverses of leading coefficients) still render normal forms (§2.2).
- **The parser's sign rule.** §2.12 read `('-' | '+') unary` recursively, so 100 KB of minus signs would
  recurse 100 000 levels. A run of signs is now read by a loop and folds to one `Neg` when the number
  of minus signs is odd.
- Smaller ones: every size the charges read (`words`, `copy_units`, `expr_units`, the key bound, the
  class flags) is computed once when a constant is built, so every a-priori charge is O(1); the
  one-term zero test reads a cached flag instead of walking the monomial; `reduce_mono` is one pass
  (it removed folded factors one by one, O(L^2)); `scale` by 1 shares the map; the kernel table is
  dropped newest kernel first, so the drop of a chain of `Inverse` kernels never recurses through the
  chain; `log(exp(r))` is folded from the kernel's recorded argument, not from its expression.

Where I did not take the reviewer's literal wording:

- B2.1 offered interned monomials with O(1) comparisons or charged searches. I took the charges.
  Interned monomials order a map by creation order, which would change the canonical order of the
  terms in every printed result and in the tests. The search charge has a free prefix of 16 factors
  per operand, because shorter comparisons are covered by the per-pair base: with 32 factors in common
  a search costs about 4 us, and the pair is charged about 300 units for it.
- B2.3 asked for `x + Q*S*0` to be `BudgetExceeded` or `Ok` within 20 s. It is now `Ok` in 8 ms, for
  the reason above, so the test also asserts `BudgetExceeded` for `x + Q*(S)*0`, where S is built and
  multiplied.
- m1 allowed a DAG input to be expanded once per occurrence. I kept that (each occurrence is charged)
  rather than memoize series per node, which would keep every node's series alive.
- The kernel-depth cap is relative to the input: max(1024, 4 x input depth). An absolute 1024 would
  refuse `x + sin(sin(...sin(1)...))` with 1025 to 4100 levels, which HEAD and revision 4 handle on an
  8 MB thread. The factor 4 covers the measured growth of a kernel's depth over its input's (§2.15).

## Revision 4 (response to review 4)

Review: `fu2/design/S1-symbolic.review4.md`. Both blockers and the six minor issues are fixed in the
body below, and the prototype `EXP/rev4/proto5/` carries each code fix. It is review 4's `p4` (the
revision 3 prototype with a fast-exit driver) plus the fixes, so "rev 3" below is `p4` built against
the same rlibs (`EXP/rev4/bin/p4_rel`). Numbers are release with budget 1e8 (the provisional
`TAYLOR_WORK`) unless stated; K is `(1+pi)^511*(1+e)^511` (262 144 terms with large coefficients), S
is `(1+sin(1))*(1+sin(2))*...*(1+sin(18))` (262 144 terms with coefficient 1, 9 kernels per term on
average), N_k is review 4's `(1+exp(...(1+exp(sin(1)))^2...))^2` nested k times, and M_k the same
with `sin` in place of `exp` and 1 inside.

| issue | change | where | evidence |
|---|---|---|---|
| B1 copies of coefficients charged one unit per term | A `SeriesConst` holds its term map behind an `Arc` and is immutable once built, so a copy is O(1) (one `Arc` increment) and takes no memory. Every operation that builds a new term map charges before it runs: the arithmetic ones term by term through the charged layer (`scale`, `mul`, `inverse`, powers), the others by `copy_units(c) = sum over terms of (32 + 4 L + 2 (w(num) + w(den)))`, with L the monomial length (the `add`/`sub` merge, `neg`, the zero test's grouping by indeterminates and its A + B I split). `to_expr` is charged `expr_units(c) = 16 + sum over terms of (16 (3 + 2 L) + 2 (w(num) + w(den)))` before it runs (kernel creation, the numeric test, an `UndecidedZero` constant, the rendering of the result). Both are sized at about 4 bytes of heap per unit. Series operations charge 4 units per term they copy (`add`, `neg`, `shift`, `split0`, `offsets`), and each occurrence of the variable 8. The grep rule of §6 covers the new constructions | §2.2, §2.4, §2.7, §6 | `(K + x + ... + x)*0` with 400 `+x`: `Ok` in 0.25 s, 6.71e7 units, 97 MB (rev 3: 33.1 s); the 160 `-x` variant 0.24 s (13.2 s); 100 `x^0` at the point K 0.24 s in the engine (8.4 s); the `+y` control refused at 1.01e8 after 0.34 s (1.2 s). Debug, two runs: 2.1 to 2.2 s, 2.1 to 2.4 s, 4.3 to 4.9 s (`EXP/rev4/repro_r4_proto5*.txt`, `budget4_debug.txt`) |
| B1 calibration | §5.3 rerun with 13 new rows: the four copy inputs, merges and a right-deep sum at the point S, two kernels of S, the shared-kernel inputs, the 40 large keys and the trial division of m2. Slowest rate 5.8e7 units/s (`1/sin(x)^200`); the slowest refused call stops after 1.41 s (`tan(x)` at 1, order 200). Every revision 3 row keeps its outcome | §5.3 | `EXP/rev4/calib4_run1.txt` and `calib4_run2.txt` (54 rows each) |
| B1 tests | `budget_coefficient_copies`: the three inputs above at order 1, budget 1e8, `Ok` within 20 s | §4.1 | debug 2.1 to 2.2 s, 2.1 to 2.4 s, 4.3 to 4.9 s |
| B2 kernel keys and shared expressions | A kernel's key is (kind, class, the arguments' normal forms), with every sub-kernel written as its interning id: linear in the normal forms, whatever the size of the expression trees. Its size bound is a walk over the normal forms that stops at the remaining budget, and it is charged 1 unit per byte before it is written. The intern table holds the only copy (`KernelData` has no key). A new kernel's expression is charged `expr_units` of its arguments before it is built. `has_indet` is an OR over the arguments' kernels. §2.2 now says that kernel expressions are DAGs whose trees can be exponentially larger, and which walks remain. The numeric test goes further than review 4 asked: `ball_cost` and `enclose_real` evaluate a node reached through a shared `Arc` once (memoized by address), so the test is linear in the DAG, also without a budget. `Debug for SeriesError` is bounded like its `Display` | §2.2, §2.3, §2.7, §2.9, §3.4 | `exp((1+exp(N_16))^511 + x)`: `Ok` in 31 ms, 1.8e7 units, 4 MB (rev 3: 11.4 s, 2.56 GB); N_30: 52 ms, 4 MB; the N_10 product refused at 1e8 after 0.26 s (rev 3: aborted at 6 GiB); found while testing: `1/((1+sin(M_30)) + x)` at 0, order 2 is `Ok` in 0.7 ms (rev 3: killed for lack of memory after 74 s); `1/((1+exp(N_16))^511 + x)` is `UndecidedZero` in 0.05 s (its ball leaves the window) |
| B2 tests | `budget_shared_kernels`: the N_16 and N_30 inputs (`Ok`), the N_10 product (`BudgetExceeded`), `1/((1+sin(M_30)) + x)` (`Ok`) and `1/((1+exp(N_16))^511 + x)` (`UndecidedZero`), within 20 s. Peak memory is checked by `EXP/rev4/tools/budget4.py` (VmHWM 4 to 5 MB), not by the unit test: cargo runs the tests as threads of one process | §4.1 | debug 0.3 to 0.5 s, 0.3 to 0.4 s, 2.1 to 2.4 s, 4 ms, 0.3 s |
| m1 two paths skip the caps | The inverse of a monomial with `(1/P0)^e` and the rational power `(m r^(1/d)/den)^p` both go through `exact_pow_if_small`, the caps of §2.2. Above them they keep the kernel `P0^e` (`NonZero`, since P0 is nonzero under the convention that made `1/P0`) or `c^(p/d)` (`NonZero`) | §2.2 | unbounded: `x + ((2+sin(1))^(-1099511627776))^(-1)` in 0.1 ms (rev 3: over 60 s); `x + ((2+sin(1))^(-5000))^(-1)` 0.1 ms (15.9 s, 1.65e10 units); `x + (2*3^(-300000))^(4095/2)` 4.5 ms, 5.7e7 units, all of it the square-root test of the 475 490-bit radicand (rev 3: over 60 s) |
| m2 trial division without a per-prime base | `(w(n) + 8)` units per candidate prime; the 6542 primes are a table built once (the port: `rustmath_integers::prime_range`) | §2.2, §2.7 | 200 cube roots of 63-bit radicands: 0.10 s for 1.27e7 units (1.2e8 units/s; rev 3: 0.58 s for 1.4e6 units, most of it the primality loop) |
| m3 `expand` depth, quadratic `contains_symbol` | Dependence on the variable is computed once per node by a pre-pass, so the walk is linear. The recursion depth stays the depth of the input (as at HEAD); `expand`, `from_expr` and the pre-pass are listed in §7 item 5 and its HANDOFF row, with the measured limits | §2.5, §7 | `x + 1 + ... + 1` with 10 000 terms: 8 ms (rev 3: 0.20 s). Depth (`EXP/rev4/depth_r4.txt`, left-deep `x + x + ... + x`, release): HEAD's `taylor` overflows between 12 659 and 13 048 terms on an 8 MB thread, the prototype between 13 048 and 13 437; on a 2 MB thread both between 2 942 and 3 331 |
| m4 line-number sentence | §3.3 now says the library sites are at HEAD and the test sites at c79a582 | §3.3 | |
| m5 port traps | §6 lists the dummy `Work` of the binomial path, the per-pair `words()` of the `mul` pre-charge (now computed once per term in the prototype), the `-q.clone()` tests for -1, and the budget error inside the zero test | §6 | |
| m6 memory in numbers | The `TAYLOR_WORK` doc comment records the measured bytes per unit: at most 3.0 (S alone: 125 MB at 4.1e7 units), K 1.4, merges and right-deep sums at the point S 1.5 to 2.0, kernels of S 1.8 to 1.9, kernel keys at most 1 byte per unit. So 1e8 units hold at most about 300 MB, the REPL's policy of a few hundred MB | §2.7, §3.1 | `EXP/rev4/mem_r4_a.txt` |

Also found and fixed while porting the review's fixes:

- The prototype reported a budget refusal inside the zero test as `UndecidedZero(0)`: the ball
  charge failed, the thread-local fault was set, and the error constant came out of a refused
  `to_expr`. The design's `zero_status` returns `Result` and propagates `BudgetExceeded` with `?`; the
  prototype now does the same (`undecided` in `konst.rs`), and §6 lists the trap.
- `(1+x)^(10^100)` at order 1400 is now charged 1.10e8 units, 1.0e7 of them for the rendering, so
  it is refused at 1e8. As revision 3 anticipated, the `TAYLOR_ORDER` doc comment uses order 1300:
  9.5e7 units, 0.2 s, 83 MB.
- `budget_kernel_keys` moves to the new totals: 2.279e7 units in all, 3.40e6 of them for the keys and
  4.3e5 for the kernel expressions, so it is refused at 2.1e7 and `Ok` at 2.3e7.
- The structural charges raised the units of the inputs at transcendental points by up to 2 times
  (`1/(2+sin(x))` at 1, order 50: 2.9e7, revision 3 1.5e7), mostly through the merges of the
  recurrence sums; each is still admitted, and the engine got faster (0.15 s, revision 3 0.18 s),
  because a copy is now O(1).

Where I did not take the reviewer's literal wording:

- B1 offered two fixes. I took the second (shared coefficients) and charge what a mutation copies.
  The first (charge `words(c)` per copy) leaves every copy in memory, and `words` undercounts small
  terms: a term with a 1-word coefficient and one kernel counts 2 words but is about 150 bytes in 3
  allocations. `copy_units` adds a per-term base for the copies that remain (merges, negations, the
  zero test's grouping and split).
- B2.1 asked to stop the key walk at `8 * remaining` bytes for a charge of one unit per 8 bytes. A key
  is now charged one unit per byte, so the walk stops at `remaining` bytes, and the keys hold at most
  about 1 byte per unit (review 4 estimated 1.6 GB of keys under the old charge).
- B2.3 asked for J1b's size estimate to be memoized per shared node. J1b's `display_size` already
  stops once it passes `OUTPUT_BYTES` (J1b §3.2), so its cost is O(`OUTPUT_BYTES`) whatever the DAG;
  §3.4 says that either the stop or a memo is required, and a plain walk without either is
  exponential.
- m3: `expand` is not made iterative. That would also need `from_expr`, the pre-pass and the REPL
  arm's own `args[0].symbols()` (a recursive walk, `substitute.rs:345`) to be iterative, and the
  limit is the same as HEAD's. It is listed instead, as the review allowed.

## Revision 3 (response to review 3)

Review: `fu2/design/S1-symbolic.review3.md`. Both blockers and the six minor issues are fixed in the
body below, and the prototype `EXP/rev3/proto4/` carries each code fix. Release numbers are from
`EXP/rev3/bin/proto4_rel` at HEAD with budget 1e8 (the provisional `TAYLOR_WORK`); "rev 2 at HEAD"
is the unchanged revision 2 prototype built against HEAD, so that the gain of Lehmer's gcd and the
gain of the new cost model are told apart.

| issue | change | where | evidence |
|---|---|---|---|
| B1 rational normalization not charged | A single layer of charged big-number operations: `q_add`, `q_sub`, `q_mul`, `q_pow`, `q_new`, `i_mul`, `i_pow`, `i_gcd`, `i_lcm`, `i_exact_root` in `series_const.rs`. Each charges an a-priori bound of its cost right before it runs: `OP_BASE = 64` per call, `a*b` for each product of a-word and b-word integers, and `N*D` for the normalization of an unreduced N-word / D-word fraction (`N` when the denominator is 1). No other code does `Rational` or `Integer` arithmetic on coefficients; a grep rule enforces it (§6 risks). So `SeriesConst::add`, `mul`, `scale`, `pow_u`, `inverse`, the radical fold, `log_exact`, `kapply`, every recurrence step and the binomial path are charged by the sizes they really normalize. The gcd constant is measured at HEAD (Lehmer): 0.8e8 to 2.1e8 word pairs per second for `Rational::new` from 257 to 8192 words, so 1 unit per pair | §2.7, §2.2 | `x + (3/5)^200000`: `Ok`, 3.7e7 units, 0.14 s (rev 2 at HEAD: 0.14 s for 15 644 units; at c79a582 12.7 s). `x + (3^330000)/(5^200000)*2*3`: `BudgetExceeded` at 1.2e8 after 0.33 s; unbounded 1.0 s, 1.8e8 units (rev 2 at HEAD: `Ok` in 0.96 s for 5.9e7 units; at c79a582 75.7 s). The same with `*2*3*5*...*29`: refused after 0.34 s (rev 2 at HEAD: 3.9 s for 5.95e7 units, and each further factor adds about 0.36 s for 15 000 units). `(2/3+x)^(1/7)` at 2000: 4.4e7 units in 0.57 s (rev 2 at HEAD: 1.5e6 units in 0.51 s). The 67-character zero-multiplied sum at 2000: refused at 1e8 after 1.26 s (rev 2 at HEAD: `Ok` after 1.5 s for 5.0e6 units) |
| B1 recalibration | §5.3 redone at HEAD in release, with large numerators and denominators, rational powers and transcendental points. Slowest measured rate 3.7e7 units/s (`tan(x)` at 1, order 200, refused after 2.7 s). `TAYLOR_WORK` stays 100 000 000: at most about 2.7 s on every measured input, and at most 10 s for any input that runs at 1e7 units/s or more, a rate 3.7 times below the slowest measured | §5.3 | `EXP/rev3/calib3_final.txt` (41 inputs); the slowest isolated operation class, a sum of 4-word rationals, charges 2.5e7 units/s (`EXP/rev3/micro_small.txt`) |
| B1, B2, m1, m2 tests | New budget tests (debug, outcome asserted, within 20 s): `budget_rational_normalization` (B1), `budget_radicals` (B2), `budget_radical_fold` (m2), `budget_transcendental_point` (m1); unit tests of the charged arithmetic (each formula, and a refusal before the power or root runs); `budget_kernel_keys` moved to the new totals | §4.1 | debug: `x + (3/5)^200000` `Ok` in 1.3 s; `x + (3^330000)/(5^200000)*2*3` `BudgetExceeded` in 2.8 s; the 67-character sum `BudgetExceeded` in 5.4 s (`EXP/rev3/budget_debug.txt`) |
| B2 roots not charged | Radicals use `Integer::exact_root` (I3) through `i_exact_root`, charged `OP_BASE + w^2` for a w-word radicand before it runs (Newton square roots after the sieve; measured 9e8 units/s at 8421 words). The perfect-power test runs once per radical: revision 2 ran it twice (`rat_root`, then the cofactor test of the same number), so `rat_root` is gone and `dth_power_part` decides both the exact case (cofactor 1) and the radical | §2.2, §2.7 | 100 terms `((3^340000+k)^(1/2))^0` (2593 characters): `BudgetExceeded` after 7 ms release, 0.13 s debug (rev 2 at HEAD: `Ok` after 10.5 s for 4.1e6 units; at c79a582 14.3 s). The variant whose radicands pass the sieve (k from 3): refused after 1.1 s debug. One radical `x + (3^340000+1)^(1/2)` is admitted: 7.3e7 units, 5 ms |
| m1 add over-charges | `SeriesConst::add` charges its merge linearly (`words(a) + words(b)`) and each common monomial's sum by `q_add` | §2.7 | `1/(2+sin(x))` at 1: `Ok` at order 50 in 0.18 s (revision 2: refused at 50 after 27 ms); `exp(sin(x))` at 1, order 50: `Ok` in 0.3 s; `log(cos(x))` and `tan(x)` at 1, order 100: `Ok` in 0.64 s and 0.95 s (revision 2: refused in 0.42 s and 0.41 s) |
| m2 powers and the radical fold | Integer powers are charged by the result size `W` in words, `OP_BASE + W^2/64 + W` (measured 5.9e7 to 2.7e8 units/s from 50 to 16 370 words). `W` comes from `n log2 abs(b) + 1` bits, with log2 from the leading 53 bits: `bits(b) * n` doubled the charge of `2^n`. The radical fold of `reduce_mono` uses `q_pow` and `q_mul` | §2.2, §2.7 | `x + 3^340000`: 1.1e6 units, 6 ms; 200 copies of `(sqrt(3)^524288)^0`: refused at 1e8 after 0.4 s release, 4.3 s debug (revision 2: 5 397 units, 0.54 s) |
| m3 valuation | `SeriesExpansion::valuation` becomes `first_exponent`, documented as the exponent of the first stored term: every term below it is exactly zero, so it is a lower bound of the valuation, equal to it when that coefficient is decided nonzero | §2.9, §2.13 | test `first_exponent_lower_bound` |
| m4 erf ball | `Erf` arm in the ball code (`Float::erf`, L = 2, R2 zero at 0) | §2.3 | `1/(x + erf(1) - 1/2)` at 0 is `Ok` with c0 = `-2/(1 - 2 erf(1))` (revision 2: `UndecidedZero`); ball unit tests `2 + erf(1)`, `erf(1) - 1/2` |
| m5 round-trip check | The check is stated: hand-built cases compare `simplify` of both sides structurally; engine outputs are re-expanded from the parsed text at the same point and order, and the coefficients compared as `SeriesConst` normal forms in one `Work` | §4.3 | |
| m6 error text | `Display for SeriesError` prints a constant only when its estimated text (a linear walk that stops at the limit, no decimal conversion) is at most 200 characters; otherwise it prints the estimated length. The variant keeps the full `Expr` | §2.9, §4.1 | review 3's 407-character input (404 characters in `EXP/rev3`'s reconstruction) prints about 110 characters instead of 5.7 MB |

Also found and fixed while porting the review's fixes:

- The charge for an exact power used `bits(b) * n`, twice the true size for `b = 2` and 26 % above it
  for `b = 3`, so `2^340000` was charged like a number of 680 000 bits. The bound is now
  `n log2|b| + 1` (above).
- The trial division of `dth_power_part` was charged `bits(n)/64 * 6542 + 1`, which is 1 unit for any
  radicand below 64 bits, whatever the number of primes tried. It is now `(bits(n)/64 + 1)` times the
  number of candidate primes, `min(6542, 2^(bits(n)/d + 1))`.
- At HEAD the high-order pole cases run in debug in 1.1 s (`1/sin(x)^200`) and 2.1 s
  (`1/(exp(x)-1)^150`), so `high_order_poles_release` is no longer ignored in debug; it is merged
  into `high_order_poles` (§4.1).
- Kernel-key budget test: the new power charge dominates the total (`2^340000` is charged 4.5e5
  units per kernel), so the budgets of `budget_kernel_keys` move to 1.89e7 (refused) and 1.95e7
  (`Ok`); the keys are charged about 4.25e5 of the 1.914e7 total.

Where I did not take the reviewer's literal wording:

- B1 asked for `GCD_UNITS >= 5` per word pair, measured with Euclid at c79a582 (2.8e6 pairs/s). At HEAD
  the gcd is Lehmer's (1.4e8 to 2.1e8 pairs/s for large balanced operands), so 1 unit per pair already
  runs at 8e7 units/s or more, above the slowest end-to-end rate. The per-call overhead, which
  dominates small operands (a 2-word rational sum takes 3.6 us), is charged by `OP_BASE = 64`.
- B1 suggested an a-priori charge per operation in each `SeriesConst` function. The charge is instead
  made once, in the arithmetic layer that every function calls, so a new call site cannot forget it.
  The series-level pre-charges stay (pairs of a product, recurrence steps, dense buffers): they refuse
  a large product at once (`exp(x)*sin(x)` at 2000 is still refused by one charge of 5.2e9).
- `TAYLOR_WORK`: §5.3 prescribes about 10 s times the slowest rate, which would be 3.7e8. I keep 1e8.
  Each review so far found an input family 30 to 10 000 times slower per unit than the calibration,
  so I keep a factor of 3.7 in reserve. 1e8 still admits every calibration input that revision 2
  admitted.

## Revision 2 (response to review 2)

Review: `fu2/design/S1-symbolic.review2.md`. Both blockers and all five minor issues are fixed in the
body below, and the prototype carries each fix (`EXP/rev2/proto3/`). Summary:

| issue | change | where | evidence |
|---|---|---|---|
| B1 MPFR underflow | Every MPFR value the ball code computes (midpoints, radii, Lipschitz bounds) must be 0 or have its exponent in a window [-W, W], W = 2^28; else there is no enclosure. MPFR's exponent range is [1 - 2^30, 2^30 - 1], so no product, sum or quotient of values in the window can underflow or overflow. A computed midpoint 0 is accepted only when the exact result is 0 (a factor is 0, the summands are exact negatives, the argument is a zero of f). exp, sinh and cosh require \|m\| + r <= 2^27. Every enclosure that is accepted is therefore free of underflow and overflow, and the radius bound of every operation holds. | §2.3 | `revchecks3`: `1/(x + (1/2)^1073741827*2^1073741822 - 1/32)` and `1/(x + exp(-744261120)*exp(744261117) - exp(-3))` at 0 give `UndecidedZero` (revision 1: a series of 1/0 terms); also the overflow twin `exp(744261120)*exp(-744261117) - exp(3)`, `(1/3)^536870912*3^536870912 - 1` and `2^(-300000000)*2^300000000 - 1`; each in 2 to 13 ms debug |
| B2 MPFR sin/cos/tan cost | MPFR sin and cos are called only when the ball has r < 1 and exponent(m) <= p + 64; otherwise sin and cos get the trivial enclosure [-1, 1] and tan none. The a-priori charge of the numeric test is `ball_cost`, a walk over the expression actually evaluated (every node, shared kernel subtrees once per occurrence), with sin, cos, tan charged at precision 2p + 64. The walk stops at the remaining budget. Measured first: only sin and cos (and tan through them) cost more with a larger exponent; exp, log, sqrt, atan, tanh, erf, sinh take microseconds at 2^(10^7) (`EXP/rev2/mpfr_cost/out.txt`). | §2.3, §2.7 | debug, budget 1e8: `1/(x + sin(2^100000000) + 2)` is `Ok` in 2 ms (revision 1: 111.7 s release); the `2^1000000000` variant is `UndecidedZero` in 2 ms (its kernel's ball leaves the window); `cos`, `tan` variants, the `pow_const` kernel fallback `x + (sin(2^100000000) + 3)^1000000000` and the non-integer test `x^(2^100000000*sin(1))` all return in 2 ms |
| m1 high-order poles | `pow_int` step 5 (n vb >= P) keeps the leading term `c0^n h^(n vb) + O(h^(n vb + 1))`; the working truncation `trunc(P)` keeps the first term when every term is at or above P; `mul` computes at least its first term (cutoff `min(prec, max(P, val + 1))`). The loss-driven retry then finds the precision. | §2.1, §2.4 | `gpcheck3` (release): `1/sin(x)^200`, `1/(x+x^2)^200`, `1/(exp(x)-1)^150`, `2/(3*sin(x)^200)`, `1/(sin(x)^200+x^300)` at 0, order 2, equal gp at every coefficient (revision 1: `PrecisionExhausted(131)`); the first three take 1.3 to 1.5 s, 15 to 17 ms and 2.3 to 3.0 s release (two runs) |
| m2 kernel creation | Kernel keys are a prefix code built in linear time with integers in hexadecimal (no decimal conversion); a kernel's creation is charged by its key length before the key is built; kernels are interned per expansion and compare by (class rank, id), so a comparison is O(1) whatever the key length. | §2.2, §2.7 | release: 40 kernels `sin(2^340000 + k)` take 15 ms and 1.28e6 units (8.6e7 units/s; revision 1: 504 ms for 8.5e5 units); with each kernel squared, 24 ms (revision 1: 786 ms) |
| m3 `log(e^k)` | Implemented and extended: `log(q e^k prod exp(r_i)^(j_i)) = log(q) + k + sum j_i r_i` for a rational q > 0, integers k, j_i and rationals r_i (all factors are positive reals). | §2.2 | `1/(x + log(e^3) - 3)` and `1/(x + log(exp(2)) - 2)` at 0 give exactly `1/x`; `1/(x + log(2*e) - 1)` gives the series of `1/(x + log(2))` (gp); `log(-e)` is not folded |
| m4 Drop wording | The table row and the doc comment of `Drop for Expr` state that the guarantee is relative to the shallowest drop seen on the thread, so a thread whose first `Expr` drop happens deep in its stack may use up to 16 KiB of glue frames below that point. | §2.14 | |
| m5 output size | `TAYLOR_ORDER` and `TAYLOR_WORK` doc comments say what each bounds. I agree that the order does not bound the output. I disagree in part with "the budget bounds the output": the units track the words of the coefficients in memory, but a kernel is stored once and printed once per coefficient that contains it. So the budget bounds the work and the memory of the result, not the printed text. The printed size is bounded by the REPL display budget (J1b, `OUTPUT_BYTES` = 16 MiB, estimated before formatting). | §3.1 | release: `(1+x)^(10^100)` at order 1400 uses 7.4e7 units and prints 95.8 MB (review 2); `(x + sin(2^100000))^1000` at order 1000 uses 1.4e5 units, 4 ms and 5.7 MB peak memory, and prints 30.4 MB in 1.7 s |

Also found and fixed while porting the review's fixes:

- `non_integer_status` converted floor(mid(A)) through decimal strings (rug to num-bigint), which is
  quadratic in num-bigint. It now converts through binary digits, and only when |mid(A)| < 2^p (a
  midpoint of magnitude at least 2^p has p significant bits, so it is itself an integer and the ball
  cannot lie strictly between two integers at that precision). When no precision gives a nearest
  integer, the error is `UndecidedZero(alpha)`.
- The `NotLaurent` message of a non-integer power formatted the exponent eagerly. No engine message
  formats a constant now ("a non-integer power of an argument of valuation v has a branch point
  there").
- The prototype built 2^-p by p multiplications; the ball code builds powers of two by shifts.
- The m1 change was rerun on review 2's cases of negative intermediate precision (`exp`, `log`, `sqrt`,
  `^100` of `1/sin(x) - 1/x (+1)` at orders 0, 1, 3) and branch cuts (`acos`, `asin` at 2, `sqrt`, `log`
  at -2): all 16 equal gp (`gpcheck3`). The harness now parenthesizes a negative base, which HEAD's
  Display prints without parentheses (the Display fix of §2.11 removes that).

Where I did not take the reviewer's literal wording:

- B1: the review offered (a) an absolute underflow term on every radius or (b) a refusal outside an
  exponent window. I took (b) and added the exact-zero rule. With (b) every operation sees operands
  of bounded exponent, which keeps the no-underflow argument of §2.3 to a few lines, and it refuses
  overflow by the same check.
- B2: the trivial enclosure is used from r >= 1, not from r >= pi. For r >= 1 the Lipschitz enclosure
  [sin m - r, sin m + r] has width at least 2 and always contains 0, so [-1, 1] is never wider.
  The `2^1000000000` variant ends in `UndecidedZero`, not `Ok`, as the review allowed.
- m1: the leading-term rule does not require the lead to be decided. The term `c0^n h^(n vb)` is exact
  whatever c0 is; an undecided c0 is reported by the strict inverse that needs it.
- m5: see the table (the budget bounds memory, not text).

## Revision 1 (response to review 1)

Review: `fu2/design/S1-symbolic.review1.md`. Every blocker and every minor issue is fixed in the body
below. Summary of the changes, with the evidence:

| issue | change | where | evidence |
|---|---|---|---|
| B1 zero test rule 4 unsound | Rule 4 now splits c = A + B*I and answers `NonZero` only when A and B **both** have real ball enclosures and one of them excludes 0 (a successful enclosure proves the value real). A non-real kernel such as `(-4)^(1/2)` or `exp(I*pi/2)` has no enclosure, so the answer is `Unknown`. This is sound and slightly stronger than the revision 0 prototype, which refused every constant with `I` (it now decides `1/(x + sin(1) + I)`). | §2.2 zero test, §2.3 | `revchecks`: `1/(x+sqrt(-4)-2*I)`, `1/(x+exp(I*pi/2)-I)`, `1/(x+sqrt(-1)-I)`, `1/(x+log(-1)-I*pi)` at 0 give `UndecidedZero`; `gpcheck2`: `1/(x+sin(1)+I)`, `1/(x+pi+I)` match gp |
| B2 i32 exponents overflow | Kernel exponents are `i64` with checked arithmetic everywhere (monomial products, powers, inverses). Series exponents (valuation, precision, shifts) are checked too. A constant power whose exponent product overflows stays a kernel `c^n`; an overflow inside a monomial product or a series exponent is `Err(Unsupported)`. | §2.1, §2.2 | debug `proto2`, no panic: `(sin(1)*x)^4294967296` at 1 gives `sin(1)^4294967296 + 4294967296*sin(1)^4294967296*(x - 1)`, `(sin(1)*x)^3000000000` likewise, `x + ((sin(1)^4096)^4096)^4096` gives `sin(1)^68719476736 + x`; all three match gp (`gpcheck2`); `(sin(1)^4*x)^(2^62)` at 1 keeps the kernel `(sin(1)^4)^(2^62)`; `x + sin(1)^(2^62)*sin(1)^(2^62)` and `(x^(-2^62))^4` are `Unsupported` |
| B3a constant arithmetic not charged | `&mut Work` is threaded through **every** `SeriesConst` operation (`from_expr`, `pow_const`, `pow_u`, `apply`, `inverse`, `mul`, `add`, `scale`, the numeric zero test, radical extraction), each returns `Result`, and each charges an a-priori size **before** it runs. Exact constant powers are capped by result size (terms and bits) and fall back to a kernel `c^n`. Large series powers use Miller's recurrence (cost independent of n) when the leading coefficient is decided, else the binomial theorem with capped constant powers. | §2.2, §2.4, §2.7 | `revchecks` (debug, budget 1e6): `x+(sin(1)+cos(1)+exp(1))^50`, `^100`, `x+(sin(1)+cos(1))^4096` return `Ok` with a kernel in 2 ms (revision 0: 13.8 s, over 120 s, over 30 s) |
| B3b dense allocation before any charge | Series are **sparse**: `Vec<(i64, SeriesConst)>` of structurally nonzero terms. The only dense buffers are the outputs of the recurrences, whose length is charged (16 units per slot) before allocation. `SeriesExpansion` is sparse too. | §2.1, §2.9 | `x^(-1000000000)+1` at 0 gives exactly `1 + 1/x^1000000000` in 2 ms, 4 MB (revision 0: 2.35 GB at 10^8, about 24 GB at 10^9) |
| B4 iterative Drop out of scope | Implemented in this unit: `impl Drop for Expr` with bounded stack use (§2.14). It costs nothing measurable on ordinary workloads: the drop glue recurses as before while the drop runs less than 16 KiB below the shallowest drop seen on the thread, and only deeper subtrees are torn down iteratively. E0509 sites fixed by matching on references: 4 library sites and 98 test sites (listed in §3.3). The `mem::forget` is gone from the simplify test. | §2.14, §3.3, §4.1, §6 | `EXP/rev1/drop_measure.txt`: a 10^6-term left-deep sum is built and dropped on a 1 MiB thread in debug (drop 0.38 s); right-deep, unary, function and zig-zag chains of 10^6 too; a drop that starts with less than 16 KiB of stack left; release cost equal to HEAD (parse+differentiate+simplify loop 106.0 ms vs 107.5 ms, min of 8) |
| m1 NotLaurent semantics | `NotLaurent` is documented as "a subexpression has no Laurent expansion at the point"; the REPL maps it to `ComputationError`. A non-integer power of a series of nonzero valuation needs the exponent **decided** non-integer (rational, a nonzero indeterminate part, or a ball strictly between two integers or off the real axis), else `UndecidedZero`. A `Pow` whose exponent expands to an exact constant uses the constant-exponent path. `0^r` is exact 0 when r(a) is a positive rational. | §2.4, §2.5, §2.9, §3 | `x^(x-x+2)` at 0 is `x^2`; `x^(sin(1)^2+cos(1)^2)` is `UndecidedZero`; `x^pi`, `x^y`, `x^(1/2+I)` are `NotLaurent`; `0^x` at 1 is `0` |
| m2 precision cap | The driver separates the two kinds of retry: a decided precision loss raises P by the exact missing amount, uncapped (bounded by 24 attempts and the budget); an undecided valuation (`NeedPrecision`) doubles the extra precision up to `max(4 abs(T), 128)`. Exact series with at most 64 terms are no longer truncated at P, so exact denominators of high valuation need no search. | §2.6 | `1/(sin(x)^50 - x^50)` at 0, order 3, gives valuation -52 equal to gp (0.12 s release); `1/(x^300*(1+x))` matches gp; `1/(sin(x)^2+cos(x)^2-1)` is `PrecisionExhausted` after 0.5 s debug |
| m3 Display of a negated base | A `Pow` base that is `Unary(Neg, _)` or a negative number is printed in parentheses. | §2.11, §4.3 | `sqrt(x - y)` added to the round-trip test |
| m4 harness false pass | `gpcheck2.py` maps `erf` to `ERF0(z) = 1 - erfc(z)` and treats any gp error as a failure (gp's own stack-growth notices excepted). Two float artifacts of gp (`sqrt(2)*sqrt(2)-2`, `sqrt(8)-2*sqrt(2)`) use exact aliases, as the pi points already did. | §4 | `erf(x)` at 0 and at 1 now genuinely pass |
| m5 constant base | `l^r` with l free of the variable: l decided zero gives the `0^r` rule, l undecided is `UndecidedZero(l)`, only a nonzero l gives `exp(r log l)`. | §2.5 | `(sin(1)^2+cos(1)^2-1)^x` at 1 is `UndecidedZero` |
| m6 slow REPL test | Replaced by `taylor("exp(x)*sin(x)", x, 0, 2000)`: the product is pre-charged at 5.2e9 units and refused at once (0.18 s debug with budget 1e8). `(1+x)^10000` at 2000 is now admitted (Miller: 11.5 ms release, 45 ms debug). | §4.5 | `revchecks`, `EXP/calib_rev1.txt` |
| m7 plan order | Ball arithmetic first (it works on `Expr`, no dependency on the constant field), then the constant field; the `Result` API, its callers in `limits.rs` and the REPL arm in one step; Display, Drop and the E0509 fixes in one step. | §6 | |
| m8 timing asserts | No timing assertion in unit tests. Only the budget tests carry a loose 20 s bound, which guards against a hang, not a speed. | §4 | |
| m9 abs/sign, messages | abs and sign documented as assuming a real variable and a real point (non-rational coefficients give `Unsupported`). Messages: "exp of an argument with a pole at the point has no Laurent expansion there" (also for arctan(1/x), sin(1/x) ...), "abs of an argument with a zero of odd order is not analytic there". | §2.4 | `revchecks` |
| m10 radicals | Radical kernels n^(1/d) are made d-th-power free by trial division with the primes below 2^16 (when n has at most 4096 bits) plus a perfect-power test of the cofactor. `sqrt(8) - 2*sqrt(2)` is now `Zero`. Products of radicals are not split (`sqrt(6) - sqrt(2)*sqrt(3)` stays `UndecidedZero`: sound, incomplete, documented). | §2.2 | `gpcheck2`: `1/(x+sqrt(8)-2*sqrt(2))` is `1/x` |
| m11 H49E3 | `x/(exp(x)-1)` to order 50 against gp (`vector(51,k,polcoef(s,k-1)) == vector(51,k,bernfrac(k-1)/(k-1)!)` is 1). | §4.1 | `gpcheck2` |

Where I did not take the reviewer's literal wording:

- B3 asked that the three constant-power inputs return `BudgetExceeded` with budget 1e6. With the
  size cap they now return the correct value at once, with the large power kept as a kernel
  (2 ms debug), which is better than a refusal and is what "fall back to a kernel" gives. The
  budget path is still tested, on inputs below the caps whose work exceeds the budget:
  `(x+sin(1)+cos(1)+exp(1)+pi)^64` (`BudgetExceeded` after 19 ms debug, budget 1e6) and
  `exp(x)*sin(x)` at order 2000.
- B4 asked for the classic always-iterative Drop. I measured three such variants first
  (`EXP/rev1/dropbench2.rs`, `EXP/rev1/workload.rs`): they cost 18 to 28 % on a
  parse+differentiate+simplify loop (+46 % instructions), because every internal node pays an
  `Arc::get_mut` and two moves. The design uses the stack-gated variant instead (§2.14): same
  guarantee, no measurable cost.

## 0. Summary of decisions

1. Replace repeated differentiation by **truncated Laurent series arithmetic over a field of
   symbolic constants** (MAGMA Handbook ch. 49, "Power, Laurent and Puiseux Series", §49.1.3-49.1.4
   precision model, §49.4.7-49.4.8, §49.5 transcendental functions; digest
   `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`). Each node of the expression
   becomes a sparse series in h = x - a with exact coefficients and an absolute precision, or the
   out-of-band precision `Exact` (revision 7rt4: no exponent or finite precision can be read as exact,
   and every units formula saturates, so no slot count from the input can wrap the budget). Products
   are truncated Cauchy products; exp, log, powers, sin/cos, sinh/cosh, inverse trig/hyperbolic
   functions and erf use the classical first-order recurrences derived from their differential
   equations (Knuth, TAOCP vol. 2, 3rd ed., §4.7; Brent and Kung 1978). Cost is O(n^2) coefficient
   operations per node instead of exponential.
2. Coefficients live in a canonical normal form: Q-linear combinations of monomials in **kernels**
   (constants such as `exp(1)`, `sin(1)`, `2^(1/2)`, `y`, `pi`) with `i64` exponents and checked
   exponent arithmetic. Every division by a derived leading coefficient goes through a **sound
   three-valued zero test** (`Zero` / `NonZero` / `Unknown`): symbolic rules first, then a rigorous
   MPFR ball enclosure of the real and imaginary parts (via `rustmath-reals::RealMPFR`), which
   refuses every value outside a safe exponent window, so that no accepted enclosure has underflowed
   or overflowed, and which is charged by the expression it evaluates. erf, whose MPFR cost grows
   like the square of its argument, gets a cheap enclosure above a threshold and is charged by its
   argument below it (revision 7). `Unknown` gives
   `Err(SeriesError::UndecidedZero)`. Nothing is guessed. Kernels are interned per expansion by a
   key built from their arguments' normal forms (sub-kernels written as ids, so the key is linear
   even though kernel expressions share subtrees), and compare in O(1). A constant's term map is
   shared (`Arc`), so a copy of a coefficient is O(1) (review 4). A kernel made from an input node
   takes that node as its expression, so a result prints like its input (revision 5); that includes
   the kernel of a divisor the input names (revision 6). Several terms are a sorted slice, and every
   rendering refers to the kernels' expressions through `Arc` (revision 6). `SeriesExpansion` keeps
   each coefficient as that shared `Arc<Expr>`, so no public entry point copies a node per term
   (revision 7).
3. `taylor`, `maclaurin`, `series`, `laurent`, `asymptotic`, `series_coefficients` and
   `series_with_big_o` return `Result<_, SeriesError>`. Poles give Laurent terms (as Sage/Maxima
   `taylor` and PARI do). Branch points, log and essential singularities, unsupported functions
   (gamma, zeta, factorial, named functions, mod of the variable) are `Err`, not fabricated values.
   This also fixes the `laurent` facade (PROJECT_TRACKER.md row "Laurent series",
   `series.rs:89`) and the "inherits the differentiate facade" note of the row "Taylor / Maclaurin
   series" (`series.rs:24`).
4. The output is a **balanced** sum (depth O(log n)) of terms whose monomials are balanced products
   (review 5, B1), and the depth of a kernel expression is capped (§2.15), so no consumer (simplify,
   Display, Drop, PartialEq, substitute) recurses once per term or per factor. `simplify` becomes
   iterative along the left spine of binary chains, and `Drop for Expr` has bounded stack use for
   every tree shape (§2.14). The engine reads chains of `+ -` and `*` along the left spine with a
   loop and sums them by one merge (revision 5), so a long input sum costs linear work; `/` belongs to
   the `*` chain, and a power tower is read along its right spine by a loop (revision 7rt1). `expand`
   and `from_expr` run one loop over an explicit work stack each (revision 7rt2), so the engine has no
   recursion over the input, and on any thread the parser is the only recursion over a text, whatever
   grammar levels its parentheses use. Display and the REPL's LaTeX writer use a work list (revision
   7rt1), so a result of any depth prints, in time linear in its text.
5. A deterministic **work budget** (`taylor_with_budget`) lets the REPL refuse expansions that would
   run too long. Every big-number operation on coefficients (products, sums with their gcd
   normalization, powers, gcds, exact roots) goes through one charged arithmetic layer, which
   charges an a-priori bound of its cost right before it runs (review 3). Every other construction
   of a term map (merges, negations, the zero test's copies), every `to_expr`, every kernel key and
   every term a series operation copies is charged by its size before it is made (review 4). Series
   operations add structural pre-charges, and every dense buffer is charged before it is allocated.
   Exact constant powers are capped by result size, on every path. The REPL taylor arm uses the
   budget, and `TAYLOR_ORDER` stays 20000, HEAD's value (an argument check; revision 7). The budget bounds
   the work and the memory of the call: every kind of heap object is charged at most 3 bytes per unit
   before it is allocated (revision 6, §2.16 "Memory"), so a call charged 1e8 units holds at most about
   300 MB of engine data plus the parsed input. That is the claim. The largest measured heap peak is
   2.83 bytes per unit (revision 7rt4, red team 4 memory m1: `(-1)*(qa*qb*...)/(1-x)^2` with 1000
   distinct names at order 1043, an admitted call, 117.8 MB of heap at 4.16e7 units, VmHWM 122 MB;
   distinct coefficients that each render a long monomial, which the rendering memo by address cannot
   share, so the row "Expr node of a rendering" of §2.16, 3.0 bytes per unit, dominates; the next order
   is refused by the tree cap); revision 7rt3 stated 2.6 (`(L*x)/(1-x)^2` with a 99 990-digit L, 259 MB
   of heap at the budget's refusal, VmHWM 147 MB); the largest admitted VmHWM stays 230 MB,
   `x + S_19*sin(x)`. Revisions 7 to 7rt2 stated their own largest, 2.29, 2.60 and 2.61, as if they were bounds,
   and a family they had not measured exceeded each (red team 3: 2.93 for one constant rendered once
   per term, which revision 7rt3 renders once, §2.8; the long monomials of red team 2 and that family
   now pass the tree cap before they are rendered, §2.15). Neither bounds the printed text, which the REPL display budget of J1b bounds (§3.1); the writers are linear (revision 7rt1), so that budget also bounds the time to print. Five hard caps bound
   the length of a monomial, the number of kernels, the size of a constant, the depth of a kernel
   expression and the size of a result as a tree (revision 7rt3), each with a typed error (§2.15), and
   every loop, allocation and recursion of the engine has a stated bound (§2.16). The value at the
   point of a regular input node renders as the node with the point substituted when its expanded
   normal form is much larger (revision 7rt3, §2.5), so results whose expanded forms name a kernel
   twice per level no longer have trees exponential in their input. A node is regular only if every
   node of its subtree is, and a node whose series is an exact constant counts as regular with that
   constant as its substitution (revision 8, review 8 B1), so a substitution never evaluates a
   function where it is not continuous or a division by zero.
6. S7: the point test uses the engine (any size of integer). The literal parser reads bignum integers;
   `a/b` stops being a literal token (it was the cause of `x^2/4` parsing as `x^(1/2)` and `1/x`
   failing to parse); a value-preserving fold keeps `parse("1/3") == Rational(1/3)`.
7. Declared out of scope (section 7): the duplicate `rustmath-calculus::taylor` (other crate, same
   defect plus an i64 factorial overflow); Puiseux and logarithmic expansions; series of gamma and
   zeta; iterative PartialEq and substitute (not in the item, which names simplify and Drop; both are
   consumers of a result, §2.15). The engine's own recursion over the input is gone since revision 7rt2
   (§2.5).

## 1. Defects confirmed at HEAD

All runs: standalone programs in `EXP/` linked against the workspace rlibs. Debug = the workspace
debug profile, main thread (8 MB stack). Release = `-O` against `target/release` rlibs. Memory is
VmHWM from `/proc/self/status`.

### 1.1 F1: exponential memory and time (`series.rs:24-58`)

`taylor` keeps `derivative = derivative.differentiate(var)` without simplification, substitutes the
point into the whole tree, and accumulates `result = result + term`. `differentiate` builds
`0*x + 2*1`-style trees whose size doubles at each product rule.

Measured with `EXP/measure_head.rs` (`bin/measure_head f 0 n`), point 0:

| f | n | debug time | debug peak | release time | release peak |
|---|---|---|---|---|---|
| x^2 | 7 | 0.15 s | 33 MB | | |
| x^2 | 8 | 1.46 s | 233 MB | 0.46 s | 232 MB |
| x^2 | 9 | 11.2 s | 1.92 GB | 3.92 s | 1.92 GB |
| exp(x) | 10 | 0.35 s | 140 MB | 0.22 s | 139 MB |
| exp(x) | 12 | 12.6 s | 4.64 GB | 7.86 s | 4.64 GB |
| exp(x) | 14 | out of memory at 6 GB | | | |
| sin(x) | 12 | 14.1 s | 5.23 GB | 9.08 s | 5.23 GB |
| 2*x+1 | 20 | 3.95 s | 1.18 GB | 2.41 s | 1.18 GB |
| 2*x+1 | 40 (debug), 24 (release) | out of memory at 6 GB | | out of memory at 6 GB | |

The growth factor is about 7 to 8 per order for x^2. The REPL accepts orders up to 20000.
`series_coefficients` (`series.rs:148-169`) has the same loop and the same growth.

### 1.2 F1: stack overflow at TAYLOR_ORDER, and the recursive consumers

`bin/measure_head x 0 20000` (debug): `thread 'main' has overflowed its stack` (order 10000: 0.5 s,
85 MB). The result is a left-deep sum of 20001 terms; `simplify` (`simplify.rs:376-400`) recurses
once per term.

Depth at which each recursive consumer overflows on a left-deep sum, debug, 8 MB stack
(`EXP/deep_head.rs`, sum of `x^k`):

| consumer | ok | overflows |
|---|---|---|
| Display (`expression.rs:437`) | 5000 | 10000 |
| simplify (`simplify.rs:376`) | 10000 | 20000 |
| substitute (`substitute.rs:26`) | 10000 | 20000 |
| PartialEq (derived) | 20000 | 40000 |
| Drop (drop glue through `Arc`) | 40000 | 80000 |

The REPL runs on tokio worker threads and `cargo test` on 2 MB threads, so the limits there are about
four times lower. The REPL also prints the result, so Display at about 2500 terms is the real limit.
Release builds are not safe either: the HEAD build of `EXP/rev1/workload.rs` aborts with a stack
overflow when it drops a 2*10^5-term sum on the 8 MB main thread.

### 1.3 F1: no REPL bound

`repl.rs:2118-2145` (HEAD 92906d4; `:1955-1983` at dc367a2) checks `0..=TAYLOR_ORDER` (20000) only. The doc comment at `repl.rs:11078-11086`
says that the order cannot bound the cost. That is true for the old algorithm and stays partly true
for the new one (the cost of dense rational series grows like n^2 times the coefficient size, see
§5), so a work budget is needed, and it must cover every operation (review 1, B3).

### 1.4 F1 (found): wrong output at singular points and for named functions

Same program, order 3 (the numerators are written `(1)/...` because HEAD cannot parse `1/sin(x)`, §1.6):

```
taylor(1/sin(x), x, 0, 3) = ((((1 / 0) + ((-1 / 0) * x)) + (((0 / 0) * (x ^ 2)) / 2)) + ...
taylor(sin(x)/x, x, 0, 3) = ((((0 / 0) + ((0 / 0) * x)) + ...
taylor(log(x), x, 0, 3)   = (((log(0) + ((1 / 0) * x)) + ...
taylor(sqrt(x), x, 0, 3)  = ((((1 / 0) * x) + ...          (the constant sqrt(0) = 0 folded away)
taylor(gamma(x), x, 1, 3) = gamma(1)                       (all higher terms silently 0)
taylor(zeta(x), x, 2, 3)  = zeta(2)
taylor(abs(x), x, 0, 3)   = (abs(0) + (sign(0) * x))       (|x| is not analytic at 0)
```

The gamma/zeta rows come from the `differentiate` facades (PROJECT_TRACKER.md row "Derivative of a
named function": d/dx of
`Function`, `Gamma`, `Zeta`, `Factorial`, `Mod` is 0; d/dx erf uses a rational approximation of
2/sqrt(pi), `differentiate.rs:197`). `laurent` (`series.rs:89-119`) returns
`taylor(num)/taylor(den)` or a plain Taylor series (tracker 🛑, row "Laurent series"). `asymptotic` (`:129-142`)
calls `taylor` at t = 0 of `f(1/t)`, which has a pole for every non-constant polynomial.
`limit_via_series` (`limits.rs:428-443`) only ever sees these `1/0` forms.

### 1.5 S7: the expansion point (`series.rs:34`, also `:206`, `:233`)

`matches!(point, Expr::Integer(i) if i.to_i64() == 0)` uses the panicking inherent `Integer::to_i64`
(`rustmath-integers/src/integer.rs:88`):

```
taylor(x^2, x, 10^30, 2)  ->  thread 'main' panicked at rustmath-integers/src/integer.rs:90:29:
                              Integer value too large for i64
```

Unit I3 (landed, commit 80130e7) changed `Integer::to_i64` to return `Option<i64>`, edited these
three sites to `i.is_zero()` (`series.rs:34`, `:206`, `:233` at HEAD dc367a2), and added the test
`taylor_at_a_huge_point` (`series.rs:795`). So the panic above is the c79a582 behaviour; HEAD no
longer panics. S1 rewrites these functions anyway; the new code never calls `to_i64` on an Integer to
decide anything about the point, and keeps I3's test, adapted to the `Result` API.

### 1.6 S7: the literal parser (`parser.rs:148-171`, `:286-298`)

`integer` parses `opt('-') digit1` with `str::parse::<i64>`; `rational` parses `integer '/'
integer` as one token and is tried by `number` before any operator. Probe `EXP/parse_head.rs`:

| input | HEAD | correct |
|---|---|---|
| `x + 18446744073709551616` | parse error | `x + 2^64` |
| `9223372036854775808` | parse error | 2^63 |
| `123456789012345678901234567890` | parse error | the integer |
| `1/x`, `1/sin(x)`, `1/(y+x)`, `exp(1/x)` | **parse error** (`rational` consumes `1/` then fails with `?`, no backtrack) | `1/x` ... |
| `x^2/4` | `(x ^ 1/2)`, i.e. x^(1/2) | (x^2)/4 |
| `2^3/4` | `2^(3/4)` | 2 |
| `2/3^2` | `(2/3)^2 = 4/9` | 2/9 |
| `3/4^2`, `x*2/3^2` | 9/16, x*4/9 | 3/16, 2x/9 |
| `x/2/3`, `2^-1/3` | x/(2/3), 2^(-1/3) | x/6, 1/6 |
| `2^-x` | parse error | 2^(-x) |
| `1/0` | parse error | parse error (kept) |

The first three rows are the HANDOFF item. The others have the same cause (the `a/b` literal token)
and the fix of the literal parser removes them; they are in scope as part of S7.
The REPL parses every symbolic argument with this parser (`repl.rs:14`, `parse_expr`), so
`taylor("x^2/4", x, 0, 3)` expands x^(1/2) today.

### 1.7 expand under a minus (`expand.rs:250-263`; HANDOFF since `0c2f368`)

`Expr::expand` sends `-u` to `distribute_negation(expand(u))`, which maps `-(a + b)` to
`(-a) - (-b)` and `-(a - b)` to `(-b) - (-a)`, so it negates one operand only. Probe
`EXP/rev6/probes/expandneg.rs` (HEAD rlibs; values at x = 3/7, y = 2/5 after `substitute` and
`simplify`, against gp `print(-(3/7-1))` and the like):

| input | HEAD's `expand` | its value | gp |
|---|---|---|---|
| `-(x - 1)` | `(-1 - -x)` | -4/7 | 4/7 |
| `-(x + 1)` | `(-x - -1)` | 4/7 | -10/7 |
| `-(x + y + 1)` | `((-x - -y) - -1)` | 34/35 | -64/35 |
| `-(x - y - 1)` | `(-1 - (-y - -x))` | -36/35 | 34/35 |
| `-((x - 1)*(x + 1))` | `((-(1 * x) - -(x * x)) - (-(1 * 1) - -(x * 1)))` | 16/49 | 40/49 |

The REPL reaches it through `expand(..)` (`repl.rs:2052-2056`), and `simplify_full`
(`simplify.rs:36`) calls `expand` first. The existing `test_expand_negation` (`expand.rs:319`)
asserts nothing (`match expanded { Expr::Binary(Sub, ..) => {}, _ => {} }`).

### 1.8 Two HANDOFF notes in S1's area (`docs/HANDOFF.md:461-463` at 2e15d1c, since `0c2f368`; review 7, m2)

The J1a review notes in HANDOFF end with two items marked "(S1's area)":

- "a symbolic chain `w + w + ... + w` of 3000 terms aborts the debug driver in `Expr` Display and
  Drop". The REPL parses such a line into a left-deep sum, and both consumers recurse once per term
  (§1.2: on an 8 MB debug thread Display overflows between 5000 and 10 000 terms, Drop between 40 000
  and 80 000, and the REPL's frames are larger). The Drop part is fixed by §2.14 (bounded stack for
  every shape). The Display part is fixed too since revision 7rt1: Display writes with a work list
  (§2.11), because an engine result can be twice as deep as its input (red team 1, H2), and
  `test_display_work_list` (§4.3) prints a left-deep sum of 10^5 terms on a 2 MB debug thread.
- "`-(-x)` displays `--x`". Since §2.12 reads runs of signs, `--x` parses back to x, so the text
  round-trips. §2.11 still prints the inner negation in parentheses, `-(-x)`, which removes the note.

## 2. Algorithm

### 2.1 Model (MAGMA ch. 49 §49.1.3, §49.1.4)

A series is stored as a sparse list of terms `c_e h^e` plus `O(h^p)`, with coefficients in the
constant field K (§2.2) and absolute precision `Abs(p)`, or `Exact` for an exact finite expansion
(constants, the variable, sums and products of them that were not truncated). Since series
arithmetic has no carries, every stored coefficient is exact (ch. 49 overview). h is `x - a` for a
finite point a, and `1/x` for the point at infinity (`asymptotic`).

```rust
// rustmath-symbolic/src/series_engine.rs (pub(crate))
/// Revision 7rt4 (red team 4 series, H1): the precision out of band. `Abs(p)`: every term of
/// exponent below p is stored or is zero, nothing is known at p and above. `Exact`: every term of
/// the value is present. Ordered `Abs(a) < Abs(b)` for a < b and every `Abs` below `Exact`, so the
/// precision of a sum is the minimum. Every i64 is a legal exponent and a legal finite precision,
/// i64::MAX included (revision 8 encoded `Exact` as the exponent i64::MAX, below).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Prec { Abs(i64), Exact }
impl Prec {
    pub fn covers(self, e: i64) -> bool;                // e < p, or true for Exact
    /// the precision times h^k: Exact stays; a finite sum that overflows upwards is Abs(i64::MAX)
    /// (a weaker true claim) and marks the attempt clamped (§2.6); downwards, the overflow error
    pub fn add(self, k: i64) -> Result<Prec, SeriesError>;
    pub fn at_most(self, p: i64) -> i64;                // min(self, P) as a finite precision
}
/// A clone copies the term vector: one pointer per coefficient (the coefficients are shared,
/// §2.2). An operation that copies terms charges SER_TERM = 6 units per term (§2.4; review 4, B1;
/// revision 6: 4 units).
#[derive(Clone)]
pub(crate) struct Laurent {
    /// (exponent, coefficient): increasing exponents, every coefficient structurally nonzero,
    /// every exponent covered by `prec`
    pub terms: Vec<(i64, SeriesConst)>,
    pub prec: Prec,
}
impl Laurent {
    /// first exponent (p when empty with precision Abs(p), 0 for the exact zero)
    pub fn val(&self) -> i64;
    pub fn coef(&self, e: i64) -> SeriesConst;          // binary search; zero when absent
    pub fn trunc(self, p: i64) -> Laurent;              // working truncation (below)
    pub fn trunc_hard(self, p: i64) -> Laurent;         // result truncation
}
```

Why out of band (revision 7rt4, red team 4 series H1). Revision 8 stored `prec = EXACT = i64::MAX`,
and i64::MAX is also a legal exponent and a legal result of a checked precision sum. Five producers
conflated them: `pow_int` item 2 stored an exact monomial at `n v = i64::MAX`, which `add` then dropped
(`e >= prec`) while keeping the sum exact; `mul` with the cutoff `EXACT` dropped a pair summing to
i64::MAX and still returned `EXACT`; `padd` returned a finite sum equal to i64::MAX; `pow_int` item 5
returned the precision `n vb + 1 = i64::MAX`; the loss-driven step of §2.6 set P = i64::MAX. So
`(x^(2^62)*x^(2^62-1))/x^(2^63-1)` at 0 (32 bytes) was a decided 0 for the value 1,
`(x^9223372036854775807 + 1)/x^9223372036854775807` lost its `+ 1`, `(x^(2^63-4)*exp(x))/x^(2^63-2)`
at order 2 claimed `1/x^2 + 1/x + 1/2` exact, and the limit path gave `Finite(0)` for limits 1 and
2^63 - 2, each in under 4 ms. Reserving i64::MAX instead would also refuse the exact product term at
i64::MAX - 1 (`(x^(2^62-1))^2/x^(2^63-2)`, whose value 1 has no finite precision above it); with the
precision out of band, that control and the five rows whose value is representable are right, and the
others are honest refusals (§4.1 `exact_sentinel`).

Why sparse (review 1, B3b): a dense vector from the valuation to the precision made
`x^(-10^9) + 1` allocate about 24 GB for a two-term answer. With sparse terms, memory is proportional
to the number of nonzero terms, except in the outputs of the recurrences (§2.4), whose length is the
working span and is charged before allocation.

`val` is a lower bound of the true valuation, and equal to it when the first coefficient is decided
nonzero. All exponent arithmetic (sums of valuations, precision `p - 2v`, shifts `n * v`, the span
and the pair sums of a product, the derivative's `e - 1`, the slots of a power, the driver's
precision steps, the rendering's `-e`) uses checked or saturating operations on `i64`; an overflow
that the result needs is `Err(Unsupported("a series exponent overflows i64"))`, and one that lies
above the working cutoff is no error: every pair sum of a product is at least its valuation, which
fits, so a sum can only overflow upwards, past every cutoff. `Exact` absorbs in precision sums
(`Prec::add`). Revision 7rt4: an exponent sum that reaches i64::MAX is an exponent like any other
(the exact term `x^(2^63-1)` is stored and divides out); a finite precision sum that overflows
upwards is clamped to `Abs(i64::MAX)`, a true claim, and the terms at and above it go into the
O-term (`shift` drops them, never errs for them); one that overflows downwards, and an exponent of an
exact series that overflows, is the overflow error.
Revision 7rt1 computed the span `hi - val + 1` of a product without a check (red team 2 series,
S-B2): `((1)/x^N + x^N)*((1)/x^N + x^N)` with N = 4e18 (97 bytes) wrapped the span, took the dense
branch with an empty buffer and panicked in release, where HEAD answers in 0.5 ms. Revision 7rt2
checked every `+`, `-` and `*` on an exponent or a precision in the engine; the plain ones left are
bounded by a check just before them (an index `e - val` below a span that fits; revision 7rt4: the
`e + 1` of `trunc` and `integrate` is checked, since an exact series may hold a term at i64::MAX).
Revision 7rt3 found one more (red team 3 series, m2): `inv` took the
offset `e - vb` of each term in `i64`, which overflows for `(1)/((1)/x^2 + x^9223372036854775806)`
(38 bytes; a panic in debug, a wrapped but correct offset in release). It is now the unsigned distance
`e.abs_diff(vb)`, and only the offsets below the slot count are kept (the loop reads no other). The
other `as usize` casts of the engine take a difference that a check just before them bounds (the
dense index of `mul`) or an exponent at least 1 and below the slot count (`offsets`).

Empty truncated series (revision 7rt3, red team 3 series H1). An empty truncated series `O(h^q)` has
an undecided valuation of at least q. It meets the precondition "valuation >= 0" only when q >= 0, and
it meets no precondition on its leading coefficient (decided nonzero, a real part of known sign,
valuation exactly 0). An operation whose precondition such an operand does not meet returns
`NeedPrecision`, so the driver raises P (§2.6); none decides anything from it. This covers `inv`,
`log`, `pow_const`, `pow_int` item 4 and `abs`/`sign` (since revision 1), and since revision 7rt3
`exp`, `sin_cos` and `sinh_cosh` (so `tan` and `tanh`), the atan family with erf, `integrate` (an empty
integrand of negative precision may hold an h^-1 term) and the rule `0^r` of §2.5. Revision 7rt2 let
`exp(O(h^-1))` be `O(h^-1)`: `x^10*exp((sin(x)-x)/x^4)` at 0, order 2, met the argument `O(h^-1)` at
P = 3, the factor x^10 lifted it to precision 9 >= T, and the result was a decided 0 for a function
with an essential singularity at 0 (gp: "domain error in exp: valuation < 0"); through the limit path
`1 + x^10*exp((sin(x^2)-x^2)/x^9)` was `Finite(1)`, although its left limit is +infinity. Now the
first attempt asks for precision and the second sees the pole: `NotLaurent`.

Working precision P and truncation. Every operation truncates its result at P:

- `trunc(P)`: an inexact series drops its terms of exponent >= P and takes `prec = min(prec, P)`,
  **except that it keeps its first term** (review 2, m1): when every term has exponent >= P, the
  first term e0 stays and `prec = min(prec, Abs(e0 + 1))` (e0 + 1 checked: e0 = i64::MAX only in an
  exact series whose single term it is, which keeps `Exact`). This is
  correct because the series' own precision exceeds e0, so its coefficients below e0 + 1 are known.
  It keeps a decided valuation above P instead of turning it into `O(h^P)`, which an inverse could
  only meet with `NeedPrecision`. An exact series is left exact when all its terms are below P **or
  when it has at most `EXACT_KEEP_TERMS = 64` terms**; a larger exact series is truncated like an
  inexact one. Keeping small exact series exact means that exact denominators of high valuation
  (`1/(x^300*(1+x))`, `1/(x^1000 + x^1001)`) need no precision search. The bound keeps the cost of
  exact arithmetic small: a product of two kept series has at most 64 * 64 pairs.
- `mul` of two exact series with at most `EXACT_KEEP_PAIRS = 4096` pairs whose top exponent fits in
  i64 is computed in full, then `trunc(P)`. Other products are computed only up to
  `min(prec, Abs(max(P, va + vb + 1)))`: at least the first term, for the same reason. An `Exact`
  cutoff drops no pair (every pair sum lies between val and the top sum, which fit), so `mul` keeps
  `Exact` only when it computed every pair; a finite cutoff that dropped a pair is the result's
  precision (revision 8 compared pair sums with the cutoff `EXACT = i64::MAX` and dropped the pair at
  i64::MAX while keeping `EXACT`).
- `Exact` means that every term of the value is present. An operation that leaves out terms it did
  not compute returns the precision up to which it computed them, never `Exact` (revision 7rt2: the
  binomial path of `pow_int` broke this, §2.4). Since revision 7rt4 no finite precision or exponent can
  be read as `Exact`: the variant is the only way to say it.
- `trunc_hard(T)` (the final result) drops every term of exponent >= T. The public precision of the
  result is T (`SeriesExpansion::precision`, §2.9): exactness stays inside the engine, so no public
  value is both an exponent and a sentinel.

Every operation computes at least the first term of its result when the result's precision allows
it; this costs one coefficient per operation.

The variable's own series is exact: `[(0, a), (1, 1)]` at a nonzero point, `[(1, 1)]` at 0,
`[(-1, 1)]` at infinity.

Why not `rustmath-powerseries`: its `PowerSeries<R: Ring>` and `LaurentSeries<R>` need a
context-free `R::zero()`, a two-valued `Ring::is_zero` and `Field::inverse` on R. The symbolic
constant field has only a three-valued zero test. `LaurentSeries::try_inverse`
(`rustmath-powerseries/src/laurent.rs:287`) finds the valuation with
`position(|c| !c.is_zero())`, so an undecided leading coefficient would be silently taken as zero or
nonzero. Implementing the core trait tower for symbolic constants would force that unsound
`is_zero`, and it would have no place for the work budget. The recurrences are the same textbook
ones; the engine below is specific to symbolic constants (absolute precision with a retry driver,
strict and generic inverses, a work budget).

### 2.2 The constant field K (`series_const.rs`)

```rust
/// Shared and immutable once built (review 4, B1): `Clone` is one `Arc` increment, so a copy of a
/// coefficient costs O(1) and no memory. Every operation that builds a new constant charges first (§2.7).
#[derive(Clone)] pub(crate) struct SeriesConst(Arc<ConstData>);
pub(crate) struct ConstData { terms: Terms, sizes: ConstSizes }
/// Revision 7rt3 (red team 3 structure, H2): a rendering of the same value whose tree is much smaller
/// than the normal form's: the input node with the variable replaced by the point (§2.5, "value at
/// the point"), with its tree size and depth. `Work` keeps the hints of the current attempt in a
/// table keyed by the constant's address, holding a clone of the constant so that the address stays
/// valid (`HashMap<*const ConstData, (SeriesConst, RenderHint)>`, cleared at every attempt and when the
/// expansion returns; a hint is set at most once per constant, charged `HINT_UNITS = 32`). Hints are
/// rare, and a field in `ConstData` would add 32 bytes to every constant (a 240-byte chunk for
/// `CONST_BASE = 72`, 3.3 bytes per unit). `to_expr` returns the hint (an `Arc` increment), and the
/// effective tree size and depth of the constant are the hint's; with an empty table every lookup is
/// skipped.
pub(crate) struct RenderHint { expr: Arc<Expr>, tree: u64, depth: u32 }
/// Revision 5 (B2 memory): one term is stored inline. Revision 6 (review 6, B2): several terms are a
/// sorted boxed slice, 88 bytes per term without slack. A BTreeMap of 2 to 11 terms allocated a leaf
/// of about 1 KB, so a series of 2-term coefficients held 1.3 KB per coefficient (13.5 bytes per unit).
/// Lookups are binary searches; no operation inserts into a stored constant (§2.2, arithmetic).
pub(crate) type Term = (Monomial, Rational);
pub(crate) enum Terms { Zero, One(Term), Many(Box<[Term]>) }  // monomial order, no zero coefficient
/// Revision 6: the only constructor. It charges CONST_BASE (the Arc, the header and the cached sizes,
/// a 208-byte chunk) before it allocates them; the terms were charged by the operation that built
/// them. 0 and 1 are shared constants and allocate nothing.
pub(crate) fn from_sorted(w: &mut Work, terms: Vec<Term>) -> Result<SeriesConst, SeriesError>;
/// Revision 5: computed by the one constructor in one pass, O(t + F), so every charge and every class
/// test is O(1): words, copy_units, expr_units, the total and the largest monomial length, the excess
/// factors of the search charge (§2.7), the bits of the key, the depth of to_expr, the largest term
/// size and exponent of an exact power, and the flags any_opaque, any_indet, nonindet_dep,
/// only_indet and real_free; revision 6 adds any_imag (some factor is I) and one_named_imag (every
/// kernel is I or one and the same named constant: rule 4 of the zero test); revision 7rt3 adds
/// `tree`, the nodes of `to_expr` as a tree (saturating), counted as `mono_expr` builds them: per term
/// each factor's kernel `tree` (plus a `Pow` and an `Integer` node when its exponent is not +-1), the
/// numerator's products and its coefficient (none for 1, a negation for -1, a product and a rational
/// node otherwise; a rational node alone without factors), the denominator's products and its
/// quotient; plus t - 1 for the sum. An overestimate would set hints and refuse results for nothing.
pub(crate) struct ConstSizes { .. }
pub(crate) type Monomial = Vec<(Kernel, i64)>;                           // sorted, exponents != 0, at most 2^15 factors
#[derive(Clone)] pub(crate) struct Kernel(Arc<KernelData>);              // Eq by id, Ord by (class rank, id)
pub(crate) struct KernelData {
    id: u64,              // interning id, unique per key within one expansion (below)
    expr: Arc<Expr>,      // the input node it was made from, or built from the arguments (below);
                          // shared (revision 6): a rendering refers to it, it never clones its top node
    depth: u32,           // the depth of expr (capped, §2.15)
    tree: u64,            // revision 7rt3: the nodes of expr as a tree (saturating; §2.15), O(1) at
                          // creation: the input node's (from the pre-pass), else the nodes the
                          // expression builds above the arguments' effective tree sizes, per kind
                          // (revision 7rt4, red team 4 structure m1): 1 for an operator f(c) or a
                          // power c^alpha (its exponent is an argument), 2 for a radical n^(1/d)
                          // (the Pow node and the exponent, not an argument), 0 for an Inverse
                          // kernel (its expression is P0's rendering; the quotient node is the
                          // monomial's), 1 for a float leaf. Revision 7rt3 counted 3 for every
                          // kind, so a chain of k computed kernels was counted about 3k for a tree
                          // of k: `sin^1200(x+1+0*(y0+...+y499))` at 0, order 1, had a result tree
                          // of 724 203 nodes counted 2 167 811, refused by the tree cap where HEAD
                          // answers in 0.46 s. A unit test builds one kernel of each kind and
                          // compares its cached tree with a memoized count of its expression
    class: KernelClass,
    has_indet: bool,      // the arguments contain an indeterminate (an OR over their kernels)
    unary: Option<(UnaryOp, SeriesConst)>,  // f and c for a kernel f(c) (log_exact reads it)
    inv_of: Option<SeriesConst>,      // Inverse kernels: kernel = 1 / P0
    root: Option<(Integer, u32)>,     // radical kernels: kernel = n^(1/d), n > 1 integer, d-th-power free (below)
}
pub(crate) enum KernelClass { Indeterminate, ImaginaryUnit, NamedConstant, NonZero, Opaque, Inverse }
pub(crate) enum ZeroStatus { Zero, NonZero, Unknown }
```

Every operation below takes `w: &mut Work` and returns `Result<_, SeriesError>`; the first thing it
does is charge its a-priori cost (§2.7). The signatures are written without `w` for brevity.

Kernel expressions (review 4, B2; revision 5). A kernel made while converting an input node
(`from_expr` of a `Unary`, of a `^` that stays a kernel, of a `Function`, a `Mod`, a `Symbol` or a
`Real`) takes that node as its expression: an O(1) clone of the node, whose children are the input's
shared `Arc`s. Its text is the input's text, and its depth is the node's depth, which the pre-pass
recorded (§2.5). Revision 4 rendered every kernel from its arguments' normal forms. Since
`(1+exp(E))^2 = 1 + 2 exp(E) + exp(E)^2` names E twice, the text doubled at every level: the result
of `x + N_300` (3.3 KB) had a text of 2^300 nodes; now it prints 10 KB (HEAD: 5 KB). Kernels that the
engine makes from computed constants (the value `exp(b0)` of a series at the point, the inverse of a
leading coefficient, a power kept by the caps, a radical) render their arguments,
reusing each sub-kernel's `Expr` through `Arc`. Their expressions are DAGs, and a tree can be
exponentially larger than its DAG: `(1 + K)^2` has the normal form `1 + 2 K + K^2`, which names K
twice, so `F_k = (1+exp((1+exp(...x...))^2))^2` with k levels at the point 1 had a result whose tree
doubled per level. Revision 7rt2 treated that as a cost of the printed text only (J1b's display budget
summarizes it); red team 3 (structure H2) showed that every consumer that walks a result as a tree
pays it too: `simplify` and `substitute` of `taylor(F_10, x, 1, 1)` (112 bytes, a tree of 1.04e7 nodes
with 22 613 distinct) took 0.8 s and 960 MB, `F_12` aborted at 4 GiB, and `PartialEq` of the order-0
result of `F_30` (a tree of 1.07e10 nodes) with its clone took 28.7 s, where HEAD's results of 605,
1125 and 151 nodes take microseconds. Revision 7rt3 bounds the tree in two ways. First, the value at
the point of a regular input node renders as that node with the variable replaced by the point when
its normal form's tree is much larger (§2.5, "value at the point"; since revision 8 a node of its
subtree whose series is an exact constant is replaced by that constant, and the flag is the AND over
the rest of the subtree): a `RenderHint` in the hint table of
`Work`, so a kernel
made from that value and a result coefficient equal to it render the substituted input, as HEAD's
result does. `F_30` at order 0 is now the input with x = 1 (a tree of 151 nodes, HEAD's result has 151),
and `F_10` at order 1 a tree of 462 903 nodes (468 023 before the exact kernel counts of revision 7rt4). Second, every constant and kernel caches the size of its tree (saturating,
O(1) from its parts), and the result's tree is capped before it is rendered (`MAX_RESULT_TREE`,
§2.15), so a consumer that walks it is bounded by the cap times its cost per node. What the cap still
refuses is a normal form that itself has exponentially many terms: the x coefficient of `F_k` at order
1 has 2^k terms, so `F_12` to `F_15` are `LimitExceeded(TreeSize)` and the budget refuses `F_16` on, as
it refused `F_20` on before (§7 item 8); and, polynomially, a normal form that names deep kernels many
times, such as a tower `x^x^...^x` at order n (about k^(n+2) nodes for k levels; refused from 22 carets
at order 2, red team 4 structure m2, §2.15). No engine code walks a kernel's expression as a tree, except
where the walk is bounded: the numeric test evaluates each shared node once with an explicit stack
(§2.3), the substitution copies each distinct input node once (§2.5), and `display_estimate` (§2.9)
stops at its limit. Display and `Debug` write with a work list (revision 7rt1, §2.11); J1b's display
budget bounds the printed text in the REPL (§3.4), and `Debug for SeriesError` is bounded (§2.9).

Kernel identity (review 2, m2; review 4, B2; `EXP/rev6/proto7/konst.rs`, `mk_kernel_src`,
`mk_kernel_inner`, `write_nf`, `intern`):

- The key is (kind, class, the arguments' normal forms), written in one linear pass: the kind
  (the `UnaryOp` name, `^` for a power, `inv`, `root` and d for a radical, `%` for mod, `@len:name` for
  a function, `$len:name` for a symbol, `r` and the 16 hex digits of `f64::to_bits` for a float),
  `:`, the class, `:`, then each argument's normal form as `{`, per term `[`, per factor `k`, the
  sub-kernel's interning id in hexadecimal, `^`, the exponent, `;`, then `#` num `/` den in
  hexadecimal, `]`, and `}`. Each token is self-delimiting, so the code is injective by
  construction. Integers are written in base 16 (`{:x}` of num-bigint, linear time), never in
  decimal, whose num-bigint conversion is quadratic. Kernels are interned once per expansion, so
  ids are stable, and two kernels of one kind and class whose arguments have equal normal forms are
  the same kernel. The
  key is linear in the arguments' normal forms, whatever the size of the expression trees (revision
  3 wrote the expression tree, so a 204-character input used 2.56 GB, review 4). The class
  is part of the key, so `c^n` as a `NonZero` and as an `Opaque` kernel are two kernels (two names
  for one number are sound, only less canonical).
- Creation is checked and charged before the key is written: first the depth cap of §2.15 (O(1): the
  input node's depth, or one more than the cached depth of the arguments), then a size bound of the
  key from the arguments' cached sizes (36 bytes per factor, the hexadecimal digits of both parts
  plus 13 per term; O(1) since revision 5), at 1 unit per byte. A key that does not fit is refused
  before any byte is written; the key's `String` is allocated with that bound as its capacity
  (revision 6: no doubling slack). When the key is new, the kernel cap and `KERNEL_UNITS = 128` (the
  `KernelData` and the table entry, about 250 bytes; the constant that holds the kernel is charged
  `CONST_BASE` by its constructor) come next, then the kernel's expression: 16 units for an input
  node, `16 + expr_units` of the arguments otherwise (§2.7), and for the `Inverse` kernel of an input
  divisor (`inverse_src`, below) 16 plus the quotient node. `has_indet` is an OR over the arguments'
  cached flags; no walk of an expression.
- Kernels are **interned** in a table owned by `Work` (one table per expansion, shared by all
  attempts of the driver): `HashMap<String, Kernel>`, which holds the only copy of each key
  (`KernelData` has none); a new key gets the next id. Two kernels are
  equal iff their ids are equal, and they are ordered by (class rank, id), with the indeterminates
  first. Comparisons are O(1) whatever the key length, so a monomial product or a map insertion never
  scans a long key. A comparison of two monomials walks their common prefix (review 5, B2), which
  the search charge of §2.7 covers. The normal form is canonical within one expansion; the order of
  terms in the output follows the creation order, which is deterministic. The table is dropped
  newest kernel first: an `Inverse` kernel holds its P0, whose monomials hold older kernels, so the
  table still owns every kernel that a dropped one refers to, and no drop recurses through a chain
  of kernels.

Classes:

- `Indeterminate`: any `Symbol` other than the variable, `pi`, `e`, `I`, `oo`. Free symbols are
  **generic parameters** (independent indeterminates), as PARI treats `y` in `1/(y+x)`, and as the
  crate's Risch normalizer already does (`risch.rs:73-81`, `:319-326`, generic-identity convention).
  Documented on every public function.
- `ImaginaryUnit`: `Symbol("I")` (the parser maps `i` and `I` to it, `parser.rs:227-232`). Monomials
  reduce `I^e` to `I^(e mod 4)` with sign: `I^2 = -1`.
- `NamedConstant`: `Symbol("pi")`, `Symbol("e")` (the parser's constants, `parser.rs:216-226`).
- `NonZero`: a constant proved nonzero (list below). `Opaque`: anything else.
- `Inverse`: `1/P0` for a constant P that the input divides by or that a decided division needs.
- `Symbol("oo")` anywhere gives `Err(SeriesError::InvalidPoint("infinity ..."))`.

Arithmetic: `add`, `sub`, `neg`, `scale(&Rational)`, `mul`, `pow_u(u64)`, `inverse(strict: bool)`
(and `inverse_src`, below), plus `pow_const` and `apply(UnaryOp)` below, and (revision 5, §2.5)
`sum_many(&[(bool, SeriesConst)])`, the sum of a chain of signed summands by one merge, and
`product_many(Vec<SeriesConst>)`, the product of a chain of factors by rounds of pairwise products.
Their rational coefficients are combined only through the charged arithmetic layer of §2.7 (`q_add`,
`q_mul`, `q_pow`, `q_new`, `i_gcd`, ...), never with the `+ - * /` operators of `Rational` and
`Integer` directly (review 3, B1). Every result is built in monomial order and handed to
`from_sorted` (revision 6, review 6 B2; `EXP/rev6/proto7/konst.rs`):

- `add` and `sub`: one two-pointer merge of the two slices (`merge2`), `q_add` or `q_sub` per common
  monomial. Charged first: `copy_units` of both operands. That also covers the comparisons: each
  comparison walks a common prefix no longer than the monomial it moves past, so the merge compares
  at most F factors in all.
- `neg` and `scale`: one map over the slice (`scale` with one `q_mul` per term), charged
  `copy_units` first. `scale` by 1 shares the constant.
- `sum_many`: a k-way merge of the k summands with a binary heap of their cursors, charged first
  `copy_units` of every summand, `SER_TERM` per summand for its cursor and `search_units(excess, k)`
  for the heap's comparisons (about 2 log2 k per term, each walking a common prefix).
- `mul`: the ta x tb pairs are accumulated in a transient `BTreeMap<Monomial, Rational>`, with the
  term cap checked before each new entry, then turned into the sorted slice in one linear pass that
  frees the map's nodes as it goes. Charged first: `MONO_FACTOR_UNITS` per factor of the product
  monomials, `search_units` for the accumulator's searches, and `ACC_ENTRY = 32` per pair for its
  slot in the map (a B-tree node holds 11 slots of 88 bytes, at least half full).
- `product_many`: a zero factor gives 0 before any round (revision 6, review 6 m2), so the outcome
  of a chain with a zero factor does not depend on where the zero sits.
- `split_imag` and the grouping by indeterminates of the zero test (below) build vectors of exact
  capacity in one pass: the imaginary part is sorted after `I` is removed from each monomial, and the
  grouping sorts its (indeterminate part, rest, coefficient) triples in place, then cuts them into
  groups (`search_units` for the comparisons of both). Neither runs when there is nothing to split or
  to group (the cached flags `any_imag` and `any_indet`).

The new monomial of `inverse` and `pow_u` is charged `MONO_FACTOR_UNITS` per factor. Exponents are
`i64`; every exponent sum or product is checked.
Canonical reductions inside `mul` (sound identities only, in one pass over the monomial: revision 4
removed each folded factor with `Vec::remove`, O(L^2)):

- `I^e -> (+-1) I^(e mod 4)`.
- Radical kernels `n^(1/d)`: exponent e reduced to `e mod d` and the factor `n^(e div d)` moved
  into the rational coefficient (so `sqrt(2) * sqrt(2) = 2` and `1/sqrt(2) = sqrt(2)/2`), but only
  when `|e div d| * bits(n) <= MAX_FOLD_BITS = 2^20`; a larger fold keeps the unreduced exponent
  (still a true identity, so the normal form stays sound; it is only less canonical). The fold
  computes `n^(e div d)` with `q_pow` and multiplies it in with `q_mul`, both charged (review 3,
  m2: revision 2 computed it uncharged, so 200 copies of `(sqrt(3)^524288)^0` took 0.54 s for 5 397
  units).
- `Inverse` kernels only ever carry positive exponents; inverting a monomial that contains
  `(1/P0)^e` multiplies by `P0^e`, computed under the caps of an exact power below
  (`exact_pow_if_small`) and otherwise kept as the kernel `P0^e`, of class `NonZero`: P0 is nonzero
  under the same convention that made the kernel `1/P0` (review 4, m1: revision 3 computed `P0^e`
  without the caps, so the unbounded `taylor` of `x + ((2+sin(1))^(-1099511627776))^(-1)` did not
  return in 60 s; now 0.1 ms).
- An exponent overflow in a monomial product is `Err(Unsupported("a kernel exponent overflows
  i64"))`, and a product monomial longer than `MAX_MONO_FACTORS` is `Err(LimitExceeded)` (§2.15).

Construction of constants from expressions free of the variable (`SeriesConst::from_expr`):
`Integer`/`Rational` exact; `Real(f)`: 0.0 is zero, a finite f is a `NonZero` kernel (no float
folding), non-finite is `Err(Unsupported)`. A float literal denotes its exact binary value, as at HEAD:
`0.1 - 1/10` is decided nonzero (its value is `1/180143985094819840`), so `1/(x + 0.1 - 1/10)` expands,
where gp, which reads 0.1 as an approximation, refuses the inverse; `0.5 - 1/2` is `Unknown` (the float
kernel is not folded). The `taylor` doc comment and the REPL help say so (revision 7rt2, red team 2
numeric m2); `Symbol` by class; `+ -` and `* /` exact, each a
chain along the left spine read by a loop and combined by `sum_many` or `product_many` (revision 5); in
the `* /` chain a divisor r is inverted as it is read, `inverse_src(false, r)` (revision 7rt1, red team 1
H3: revision 7 recursed once per `/`, and `x + (1/2/.../2)` with 400 slashes aborted a 2 MB debug
thread); `^` is `pow_const`, and a tower `b_1^(b_2^(...^t))` (with negations between the levels) is read
along its right spine by a loop: the bases left to right, then t, then `pow_const` from the inside out,
each with its own node as the kernel's source (revision 7rt1, red team 1 H1: revision 7 recursed once
per `^`, and `x+1^1^...^1` with 400 carets aborted a 2 MB debug thread); `Unary` is `apply`;
`Function(name, args)` and `Mod` are `Opaque` kernels with canonical arguments. `from_expr` does not
recurse over the input (revision 7rt2, red team 2 structure H1). One loop runs an explicit work stack
with one record per pending node of the current path: a chain, a tower, a call or a `%` node holds its
items and the values read so far in a boxed record (at most 96 bytes, charged `RECORD_UNITS = 32` when
it is pushed), a unary node is a 16-byte record. The value of a finished node is handed to the record
on top, which names its next child or finishes. The order of the conversions, the charges, the checks
and the source node of every kernel are those of the recursive reading (prototype
`EXP/rev7rt2/proto10/konst.rs`: `konst_of`, `konst_visit`, `KFrame::resume`). Revision 7rt1 read the
chains and towers by loops but recursed once per nested operand, through a dispatcher frame and a chain
frame for each grammar level a parenthesis used, so a text that used five levels per parenthesis cost
15.5 to 18.9 KB of debug stack per parenthesis against the parser's 9.7 KB, and
`x + -(-(...(1)^2*3+4)...)^2*3+4` with 133 levels (1.2 KB) aborted a 2 MB debug thread that had parsed
it. A kernel made here takes its input node as its expression (above). Each step charges before it runs, starting with `KONST_NODE = 16` per node visit
plus the words of a literal (revision 5, review 5 m1: a DAG input is converted once per occurrence,
each occurrence charged), so a large constant in the input is refused by the budget before any large
computation (review 1, B3a).

`pow_const(c, alpha)` (c^alpha for constants):

- alpha an integer n, c = 0: `0^n` is 0 for n > 0, 1 for n = 0, `Err(DivisionByZero)` for n < 0.
- alpha an integer n, c = 1 or c = -1: 1, or `(-1)^n` by the parity of n, whatever the size of n
  (revision 7rt2, red team 2 structure m3: revision 7rt1 kept the kernel `1 ^ N` when N did not fit
  in i64, so `exp(x)^(2^100)` carried it in its coefficients).
- alpha an integer n, c != 0. Let b = c for n > 0 and b = `inverse(false)` of c for n < 0 (a single
  monomial in all cases except a multi-term c with n > 0). The power is **exact** when its a-priori
  size is small, and a kernel `c^n` otherwise (`NonZero` when c is decided nonzero, else `Opaque`):
  - b a single monomial `q * m`: exact when `|n| * B <= MAX_EXACT_POWER_BITS = 2^20`, where
    `B = bits(q) + sum over radical kernels of bits(n_rad) * e / d` (B = 0 for q = +-1 and no
    radical); the exponents of m are multiplied by n with `checked_mul` (an overflow gives the kernel
    `c^n`). So `2^100000` is an exact integer, `sin(1)^(2^36)` is the monomial `sin(1)^68719476736`,
    and `(sqrt(2))^(2^40)` stays a kernel (its fold would need 2^39 bits).
  - b with t >= 2 terms: exact when `|n| <= 4096`, the multinomial term count
    `C(|n| + t - 1, t - 1) <= MAX_EXACT_POWER_TERMS = 512`, and `|n| * (B + log2 t + 1) <= 2^16`
    bits; computed by binary powering with `mul` (each product pre-charged). So
    `(sin(1)+cos(1))^4096` and `(sin(1)+cos(1)+exp(1))^50` are kernels (review 1 inputs), and
    `(1+sqrt(2))^20` is expanded.
  - These two tests are one function, `exact_pow_if_small(b, |n|)`, and every exact power of a
    constant goes through it: this case, the inverse of `(1/P0)^e` above and the rational power
    below (review 4, m1).
- alpha = p/d (d > 1), c rational > 0: `c^(1/d) = (num * den^(d-1))^(1/d) / den`; write
  `num * den^(d-1) = m^d * r` with r d-th-power free (below); then `c^(p/d) = (m * r^(1/d) / den)^p`
  with the radical kernel `r^(1/d)`, computed for `|p| <= 4096` and only when
  `exact_pow_if_small` admits it (the single-monomial cap, `|p| * (bits(m/den) + bits(r)/d) <=
  2^20`), else the `NonZero` kernel `c^(p/d)` (review 4, m1: revision 3 had no bit cap here, so the
  unbounded `taylor` of `x + (2*3^(-300000))^(4095/2)` did not return in 60 s; now 4.5 ms, most of it
  the square-root test of the radicand). When r = 1, c is a perfect d-th power and `c^(p/d)` is the
  exact power `(m/den)^p` under the size caps above (so `4^(10001/2) = 2^10001`). This one
  decomposition is the only perfect-power test of the radical (review 3, B2: revision 2 first tried
  `rat_root`, a floor root of the numerator and of the denominator, and then tested the same
  cofactor again, two uncharged roots per radical). Negative rational c, or d > 64: a `NonZero`
  kernel `c^(p/d)` (principal value).
- other alpha: `1^alpha = 1`; c decided zero: the rule of `0^r` (§2.5) with r = alpha, one
  function shared by both (`zero_pow_sign`, revision 8, review 8 m2): alpha = A + B I by
  `split_imag`, the sign of A by rule 5's enclosures (`real_sign`); A > 0 gives the exact 0, A < 0
  `Err(DivisionByZero("0 to a power with negative real part"))`, A exactly 0 with B decided nonzero
  `Err(NotLaurent)`, anything undecided `Err(UndecidedZero(A))` (or of B). Revision 7rt3 returned
  `UndecidedZero(0^alpha)` here, so one constant got two answers by the route it took: `x + 0^pi`,
  `x + (x-x)^pi` and `x + (x-x)^(x-x+pi)` were `UndecidedZero` while `x + 0^(pi + x)` was x. Else a
  kernel `c^alpha` (`NonZero` when c is decided nonzero, else `Opaque`).

d-th-power free radicals (review 1, m10): `n = m^d * r` by trial division with the 6542 primes below
2^16 (a table from `rustmath_integers::prime_range`, built once; the loop never tests the
primality of each p) when n has at most 4096 bits, charged first by
`(w(n) + 8) * min(6542, 2^(bits(n)/d + 1))` units: the number of candidate primes p with p^d <= n,
times the words of n plus a base of 8 for the power p^d, the copy and the remainder of each
candidate (review 4, m2: without the base a 63-bit radicand with d = 3 ran 6 542 candidates for
6 542 units, 1.1e7 units/s; now 200 such cube roots take 0.10 s for 1.27e7 units). The exact
divisions by a found p^d (at most bits(n)/d of them, n below 4096 bits) are covered by the same
charge. Then a cofactor that is a perfect d-th power moves into m. The
cofactor test is `Integer::exact_root(d)` (unit I3, MAGMA `IsPower(n, k)`, handbook ch. 18 §18.4.5:
a power-residue sieve, the 2-adic root for the odd part of d, then Newton square roots; every "no" is
a proof), called through `i_exact_root`, which first charges `OP_BASE + w^2` units for a w-word
cofactor (review 3, B2). The square-root path is the costly one: 80 ms for a sieve-passing
539 000-bit radicand in release, 7.1e7 units, so about 9e8 units/s; odd d and sieve-refused
radicands take under 1 ms. Above 4096 bits there is no trial division, so the radicand of a kept
kernel may still have small d-th-power factors (sound, less canonical). So `sqrt(8) = 2*sqrt(2)` and
`sqrt(8) - 2*sqrt(2)` is `Zero`. Products are not split into prime radicals (that needs a full
factorization): `sqrt(6) - sqrt(2)*sqrt(3)` stays `Unknown` (sound, incomplete, documented).

`apply(op, c)` (the value f(c)):

- Exact values at rational q: exp(0) = 1, log(1) = 0, sin/tan/sinh/tanh/arcsin/arctan/arcsinh/
  arctanh/erf(0) = 0, cos(0) = cosh(0) = 1, arccos(1) = arccosh(1) = 0, abs and sign of q,
  sqrt via `pow_const(c, 1/2)` (so `sqrt(2)` and `2^(1/2)` are the same kernel). The rational
  arithmetic of this step (`q + 1/2` for cos, `12 q` for the twelfths of `sin(q pi)`, `q + 1` for
  the factorial pole test, `j r` in `log_exact`) uses `q_add` and `q_mul`, since q can be large
  (review 3, B1). Poles of the
  constant are `Err(DivisionByZero)`: log(0), arctanh(1), arctanh(-1), zeta(1), gamma and factorial
  at a pole. Revision 7rt4 (red team 4 numeric, m1): the prototype of revision 8 tested log, gamma and
  factorial only, so arctanh(+-1) was a `NonZero` kernel and zeta(1) an `Opaque` one:
  `(1)/(x + atanh(1))` inverted the pole, and the limit path gave `Finite(0)` for `x*atanh(1)` (gp:
  "domain error" for both constants; HEAD gives the same outputs). The test is on the rational
  argument, so `atanh(sin(pi/2))` (sin(pi/2) folds to 1) is refused too (§4.1 `constant_poles`). At a
  non-rational argument the pole test would be a zero test of `c - 1` (or `c + 1`); f(c) stays a kernel
  of the class below, a known cost (§7 item 10).
- Exact trig at rational multiples of pi (`c = q*pi`, a single monomial): sin, cos and tan for
  denominators 1, 2, 3, 4, 6 (values in Q(sqrt 2, sqrt 3); tan at an odd multiple of pi/2 is
  `Err(DivisionByZero)`). Other q: a `NonZero` kernel (sin(q pi) = 0 iff q is an integer, cos(q pi) =
  0 iff q - 1/2 is an integer).
- Exact logs (review 2, m3; `EXP/rev6/proto7/konst.rs::log_exact`): for c a single monomial
  `q * e^k * exp(r_1)^(j_1) * ... * exp(r_m)^(j_m)` with q a positive rational, k and j_i integers and
  r_i rationals, `log(c) = log(q) + k + j_1 r_1 + ... + j_m r_m` (`log(q)` is 0 for q = 1, else the
  `NonZero` kernel `log(q)`). Every factor is a positive real, so the principal log is additive. So
  `log(e^3) = 3`, `log(exp(2)) = 2`, `log(2*e) = 1 + log(2)`. A negative q or any other kernel is not
  folded (`log(-e)` stays a kernel; it is 1 + I pi). Since revision 5 a kernel `exp(r)` is recognised
  by its recorded `unary = (Exp, c)` with c rational, not by its expression, which may be the input's
  own node: `log(exp(1+1))` folds to 2.
- Classification of the remaining kernels f(c): for rational q, f(q) is `NonZero` for exp, log
  (q != 1), sin, cos, tan, sinh, cosh, tanh, arcsin, arccos (q != 1), arctan, arcsinh, arccosh
  (q != 1), arctanh, erf, gamma, factorial (pi is irrational, Lambert 1761, so sin q = 0 and
  cos q = 0 have no rational solution q != 0; the hyperbolic and inverse functions vanish only at the
  point listed; Gamma has no zeros). zeta(q) is `Opaque`. For non-rational c: exp(c) is always
  `NonZero`; sqrt(c) is `NonZero` when c is; f(c) for c a non-constant polynomial in indeterminates
  only is `NonZero` (generic); otherwise `Opaque`.

**Zero test** `zero_status(&self) -> Result<ZeroStatus, SeriesError>` (sound;
`EXP/rev6/proto7/konst.rs::zero_status`, `const_status`, `numeric`):

1. No terms: `Zero`.
2. One term `q * m`: `NonZero` if m has no `Opaque` kernel (a product of nonzero numbers, real or
   complex, is nonzero), else the numeric test (rule 5).
3. Several terms. If some non-indeterminate kernel depends on an indeterminate: the numeric test
   (which cannot evaluate indeterminates, so in practice `Unknown`). Else group the terms by their
   indeterminate monomial. With a non-empty indeterminate part, the constant is a Laurent polynomial
   in independent indeterminates over the constants, so it is `NonZero` iff some group coefficient
   is (by rules 4 and 5), else `Unknown`.
4. Constants (no indeterminates) whose kernels are only `I` and one single named constant (only pi,
   or only e): `NonZero`. Proof: write c = A(t) + B(t) I with A, B Laurent polynomials over Q and t
   = pi or e. t is real and transcendental (Lindemann 1882, Hermite 1873), so A(t) and B(t) are real
   and c = 0 forces A(t) = B(t) = 0, hence A = B = 0 as polynomials; but the normal form is not
   empty.
5. Numeric test (review 1, B1): split c = A + B I, where A collects the terms without the kernel `I`
   and B the terms with `I` (with `I` removed). The answer is `NonZero` iff **A and B both have real
   ball enclosures** (§2.3) and one of them excludes 0, at one of the precisions 64, 256, 1024 bits;
   otherwise `Unknown`. The precisions are tried in order. A precision at which an enclosure is
   missing only because a ball is too wide (`NoBall::TooWide`, §2.3) goes on to the next one; any other
   missing enclosure (`NoBall::Never`) ends the test with `Unknown` (revision 7rt2, red team 2 numeric
   m1: revision 7rt1 ended the test at the first precision without an enclosure, so
   `1/(x + tan(10^18) + 1)`, `1/(x + log(PQ - e) + 84)` with PQ a convergent of e, and five more
   constants that 256 bits decide were `UndecidedZero`, where HEAD prints the series). An enclosure exists only for an expression built from rationals, pi, e and
   real functions on balls inside their real domains, so its value is a real number. Then A and B
   are real, c = 0 iff A = B = 0, and a ball that excludes 0 proves one of them nonzero. A kernel
   that may be non-real (`(-4)^(1/2)`, `exp(I*pi/2)`, `log(-1)`, an `Inverse` of a complex constant)
   has no enclosure, so the answer is `Unknown`, whatever B is. This replaces revision 0's
   unsound rule "NonZero if either part is proved nonzero". An enclosure is also refused when any
   value computed for it leaves the exponent window of §2.3 (review 2, B1), so a constant whose
   evaluation would underflow or overflow in MPFR gets `Unknown`: `(1/2)^1073741827 * 2^1073741822 -
   1/32` and `exp(-744261120) * exp(744261117) - exp(-3)` (both exactly 0) are `Unknown`.

The grouping of rule 3 and the split of rule 5 copy every term, so each charges first (review 4, B1):
the grouping `2 copy_units(c) + 2 search_units` (one vector of triples, sorted in place, then one
constant per group; revision 6), the split `copy_units(c) + search_units`. Revision 6 enters neither
when there is nothing to do: a constant without indeterminates is one group, itself, and a constant
without the kernel I is its own real part, both O(1) from the cached flags; rule 4 reads
`one_named_imag`. The numeric test charges `expr_units` of A and B for their `to_expr`, then
`ball_cost` (§2.3). A refusal inside the test is `Err(BudgetExceeded)`, returned
with `?` before any `UndecidedZero` constant is built (the revision 3 prototype reported it as
`UndecidedZero(0)`, an artifact of its fault flag).

Soundness: every rewriting above is a true identity, so a zero normal form is a zero value
(`Zero` is sound). Each `NonZero` answer is one of: a product of a nonzero rational and kernels
proved nonzero; a nonzero polynomial in algebraically independent indeterminates (the generic
convention); a nonzero polynomial in one transcendental real and I; or rigorous real enclosures of
both parts with one excluding 0. `sin(1)^2 + cos(1)^2 - 1` stays `Unknown` (it is zero; the numeric
enclosure always contains 0). `sqrt(-4) - 2*I` and `exp(I*pi/2) - I` stay `Unknown`.

**Decided non-integer** `non_integer_status(alpha) -> Result<(ZeroStatus, Option<Integer>),
SeriesError>` (for m1 of review 1, used by `pow_const` of a series of nonzero valuation): `NonZero`
when alpha is decided not to be an integer: alpha a non-integer rational; or alpha has an
indeterminate group with a nonzero coefficient (then alpha - n is nonzero for every integer n,
generic convention); or alpha = A + B I with real enclosures (the precisions as in rule 5: too wide
goes on) where B excludes 0; or B is exactly 0
and, with k = floor(mid(A)), the balls of A - k and A - (k + 1) both exclude 0 (the enclosure of A is
an interval containing its midpoint, so it lies in (k, k + 1)). `Zero` when alpha is an integer (a
rational integer), with that integer. Otherwise `Unknown`, with the integer n nearest to mid(A) when
one was computed; the caller reports `UndecidedZero(alpha - n)`, or `UndecidedZero(alpha)` without
n. The floor is computed only when |mid(A)| < 2^p (a midpoint of magnitude at least 2^p has p
significant bits, so it is itself an integer, the ball contains it, and the test cannot succeed at
that precision), and it is converted from MPFR through
binary digits (`rug::Integer::write_digits`), never through a decimal string: the decimal parse of
num-bigint is quadratic, and a 2^28-bit midpoint would take hours (found while fixing review 2, B2).
`x^(2^100000000 * sin(1))` at 0 is `UndecidedZero(2^100000000 * sin(1))` in 2 ms.

**Inverse.** `inverse(strict)`: `Zero` is `Err(DivisionByZero)`. A single monomial without `Opaque`
kernels is inverted exactly (exponents negated with `checked_neg`). Otherwise a kernel `1/P0` is
created, where `P = c * P0`, `P0` has coprime integer coefficients and a positive leading coefficient
(`EXP/rev6/proto7/konst.rs`, so `1/(2+sin(1))` prints as such); the content c comes from `i_gcd` of
the numerators and `i_lcm` of the denominators, and `q_new`, all charged. `strict = true` (a derived leading
coefficient) requires `NonZero`, else `Err(UndecidedZero(c))`. `strict = false` is used only for
constants that the **input itself** divides by (a variable-free denominator): the input is assumed
defined, and an inverse of a defined constant is nonzero. This is the distinction between "the input
divides by a constant" (generic, like Risch) and "the expansion must know a valuation" (decided,
else refused).

`inverse_src(strict, src)` (revision 6, review 6 m1): when the input names the divisor, `src` is
that input node r (`from_expr` of `l / r`, the constant divisor of the series `Div` case, and the
base of `c^n` with n < 0). The value of r is P = c P0, so the kernel `1/P0` takes `r / c` as its
expression (r itself when c = 1), and its depth is the depth of r in the input (plus 1 when c is not
1), which the pre-pass recorded. Revision 5 rendered P0's normal form there: with
`(1)/((1 + inner)^511)` nested 80 times (1.1 KB), every level's kernel rendered a 512-term P0, 13
levels of kernel per 3 levels of input, and the unbounded `taylor` met the kernel-depth cap after
19.3 s (review 6). Now it is `Ok` in about 3 s with kernel depth 241 and result depth 243, the
input's depth (HEAD's result: depth 248). Strict inverses of computed constants still render P0 (§2.15).

`to_expr()`: a balanced sum of monomials. A monomial is `q * N / D`, with N and D the **balanced**
products (`balanced_product`, the shape rule of `balanced_sum`) of the positive-exponent kernels and
of the negative-exponent kernels and `Inverse` denominators; q = 1 and q = -1 print without the
factor (review 5, B1: revision 4 built left-deep products, so a monomial of L factors had depth L, and
a 4.6 KB balanced input aborted a 2 MB debug thread). The tree is built from `Arc<Expr>` nodes
(revision 6): a kernel factor with exponent +-1 is the kernel's own `Arc` (an increment, no
allocation), so a term allocates at most `2 + L + 2 pf` nodes, pf the factors whose exponent is not
+-1 (a `Pow` node and an `Integer` node each). Revision 5 cloned each kernel's top node into the
product; a `Symbol` node owns its name's `String`, so `(1 + Y*x + Z*x)^300` with 40 KB names (80 KB of
input) copied the names once per monomial: 3.68 GB at 2.9e7 units (found by the self-attack, §5.3;
now 49 MB). With `bal(n)` the depth that a balanced tree adds
over n leaves (0 for n = 1, n - 1 up to 4, then `1 + bal(ceil(n/2))`; `bal(2^15) = 16`,
`bal(2^20) = 21`), a constant has depth at most its deepest kernel + `bal(t) + bal(L) + 3`, so at most
the deepest kernel + 40 under the caps of §2.15.

### 2.3 Rigorous numeric enclosures (`series_ball.rs`)

The module works on `Expr` only (no dependency on the constant field):

```rust
/// Evaluates every distinct node once: a node reached through an `Arc` is memoized by its address
/// (`Arc::as_ptr(..).addr()`, compared, never dereferenced), so the cost is linear in the DAG.
/// Iterative (revision 5, review 5 B1): a post-order walk with an explicit stack, so it uses O(1)
/// native stack whatever the depth of `e`. Every child of a node is needed by it, so the first child
/// without an enclosure ends the call. An integer exponent is read, not evaluated.
/// Revision 7rt2: the error says why there is no enclosure (below).
pub(crate) fn enclose_real(e: &Expr, prec: u32) -> Result<Ball, NoBall>;
/// `Never`: no precision gives one (a non-real or opaque node, a function, an integer exponent
/// above 2^32, a value outside the window, R2, a domain check failed by the whole ball, the budget).
/// `TooWide`: a domain or guard check failed on a ball that still meets the domain, so a higher
/// precision may give one.
pub(crate) enum NoBall { Never, TooWide }
/// A-priori cost of enclose_real(e, prec) in work units: a walk that counts every distinct node
/// once (the same memo key), as enclose_real evaluates it; it stops once the sum passes `limit`
/// and then returns a value above `limit`.
pub(crate) fn ball_cost(e: &Expr, prec: u32, limit: u64) -> u64;
```

The memo is sound: `e` is borrowed for the whole call, so the address of a live `Arc` allocation
names one node, and a node's ball at precision p does not depend on where the node occurs. Revision
4 recursed once per level (`ball_memo`, `ball_child`), and a monomial rendered as a left-deep product
of 431 sines overflowed a 2 MB debug thread there (review 5, B1); the explicit stack removes the
recursion, and the balanced products of §2.2 remove the depth. It is
needed because kernel expressions share subtrees (§2.2): revision 3 walked the tree, which is
exponential in the nesting (review 4, B2). Within the budget that walk was refused at the budget,
which is honest, but `taylor` without a budget would walk 2^30 nodes for a 355-character input. With
the memo, `1/((1+sin(M_30)) + x)` at 0, order 2 (M_k is `(1+sin(...(1+sin(1))^2...))^2` nested k
times) is `Ok` in 0.7 ms with 5.1e4 units, with or without a budget; the revision 3 prototype was
killed for lack of memory after 74 s (its kernel keys and its `symbols()` walk, §2.2).

`series_const.rs` calls them on `A.to_expr()` and `B.to_expr()` (rule 5) after checking that A and B
contain no `I` and no indeterminate kernel: first `w.charge(ball_cost(&e, p, w.remaining()))?`, then
`enclose_real(&e, p)`, at each precision p of 64, 256, 1024 (review 2, B2: the revision 1 charge
`words(A) * (p/64 + 1)^2 * 8` counted a kernel as one word, so `sin(2^100000000)` cost 2 words; the
walk counts the expression that is evaluated). A `TooWide` precision goes on to the next one, a
`Never` stops (rule 5, revision 7rt2), so the cost is at most the three charges that a constant
undecided at every precision already pays.

Why there is no enclosure (revision 7rt2, red team 2 numeric m1). The checks that fail on a ball that
still contains a point of the domain give `TooWide`: `lo <= 0 < hi` for log, sqrt and the real powers
`exp(b log a)`; `|m| <= r` for `1/x` (and so for tan's cosine and atanh's `1 - x`), except an exact 0;
`gap <= 0` in `1/x` (the inflated radius reaches `|m|`); tan's guard (`r >= 1`, or an exponent above
`p + 64`); `|m| - r < 1 <= |m| + r` for asin, acos and atanh; `lo <= 1 < hi` for acosh; the exp, sinh
and cosh guard failed with `|m| - r` at most `2^27`; a ball across 0 raised to an integer beyond i64
(revision 7rt3). Every other missing enclosure is `Never`: a node that is not a real function of
rationals, pi and e, R1 (a value outside the window
is so at every precision) and R2 (a flush to zero is an underflow, which R1 refuses at every precision),
a domain check failed by the whole ball (`hi <= 0` for log), the charge refused by the budget. The
classification decides nothing by itself: `NonZero` still needs a ball that excludes 0, so a wrong
`TooWide` only costs the charge of a higher precision, and a wrong `Never` only gives up early. So the
tests may use rounded values (`|m| - r`). At 64 bits, `tan(10^18)` has the argument radius
`10^18 2^-64 = 0.054`, and the tan ball is too wide to separate `tan(10^18) + 1 = -7.3885` from 0; at
256 bits the radius is `2.5e-56` (gp: `tan(10^18) + 1 = -7.38854968059368800013`). The prototype
(`EXP/rev7rt2/proto10/ball.rs`, `ball_expr_why`) records the reason in a thread-local flag set where the
check fails; the port returns it.

Midpoint-radius balls over `rustmath_reals::RealMPFR` (MPFR rounds every operation correctly to
nearest: Fousse et al., ACM TOMS 33(2), 2007; ball arithmetic as in Johansson, IEEE Trans. Comput.
66(8), 2017). For a result r the rounding error is at most `|r| 2^-p` **when no underflow or
overflow happens**; an input radius propagates through f with a Lipschitz bound L of f on the ball;
every computed radius is multiplied by `1 + 2^(4-p)` so that it stays an upper bound despite its own
rounding. Powers of two (2^-p, 2^(4-p)) are built exactly by shifts.

**Exponent range (review 2, B1).** MPFR's default exponent range is [emin, emax] = [1 - 2^30,
2^30 - 1] (checked: `rug::float::exp_min()`). A result below 2^(emin-1) in magnitude underflows to 0
or to 2^(emin-1), and one above 2^emax overflows to infinity; then the bound above fails. Revision 1
decided the exact zero `(1/2)^1073741827 * 2^1073741822 - 1/32` nonzero this way: the first power
underflowed to [0, 0]. Two checks make every accepted enclosure free of underflow and overflow:

- **R1, window.** Every MPFR value the module computes (midpoints, radii, Lipschitz bounds,
  intermediate terms of radii) must be 0 or have its exponent in [-W, W], W = 2^28; otherwise the
  operation returns `None` and there is no enclosure. Inputs are checked when they are built
  (integers, rationals, floats, pi, e).
- **R2, exact zeros.** A computed midpoint equal to 0 is accepted only when the exact result is 0:
  for a product, a factor's midpoint is 0; for a sum, the two midpoints are exact negatives
  (`a == -b`); for f(m), m is a zero of f (sin, tan, atan, tanh, sinh, erf, sqrt, asin, asinh,
  atanh, abs at 0; log, acos, acosh at 1; exp, cos, cosh never). For an integer or a rational input,
  the number is 0. Otherwise `None`.
- exp, sinh and cosh require `|m| + r <= 2^27` before MPFR is called (exp(2^27) has exponent about
  1.94 * 10^8 < W).

Why this suffices: with operands in the window, a nonzero exact product has exponent at least -2W - 1
= -2^29 - 1 > emin, a nonzero exact sum is a multiple of the unit in the last place of the operand
with the smaller exponent, so at least 2^(-W-p) > 2^emin, a quotient 1/m is at least 2^(-W-1), and
exp, sinh, cosh stay inside the window by their guard; log, sqrt, atan, tanh, erf, asin, acos and the trig functions of a windowed argument cannot
overflow. So the only possible underflow is a nonzero exact result that MPFR flushes to 0 or to
2^(emin-1): R2 refuses the first, R1 the second (its exponent is emin < -W). A representable result
outside the window is refused by R1 before any later operation uses it. Every value an accepted
enclosure was computed from is therefore a correctly rounded MPFR result, and the radius bounds hold.
R1 is conservative: values of magnitude below 2^-W or above 2^W (such as `2^(2^29)` or
`exp(-744261120)`) get no enclosure, and constants built from them stay `Unknown` (sound). The
five test inputs of §4.1 `undecided_after_underflow` are exactly zero and give `UndecidedZero`.

**Argument size of sin and cos (review 2, B2).** MPFR's sin and cos reduce the argument modulo 2 pi
with as many bits of pi as the argument's exponent: at p = 64, sin(1.5 * 2^(10^7)) takes 3.8 s,
while exp, log, sqrt, atan, tanh and sinh take at most 10 us at any exponent
(`EXP/rev2/mpfr_cost/out.txt`, at p = 64 only). Revision 2 listed erf there too, which is false
(review 7, B1): its cost grows like the square of its argument, see "erf" below. The scan of every
operation at 64, 256 and 1024 bits is at the end of this section. So MPFR sin and cos are called only when `r < 1` and the midpoint is
0 or has exponent at most `p + 64`. Otherwise sin and cos get the trivial enclosure [-1, 1]
(midpoint 0, radius 1) and tan gets none. For r >= 1 the Lipschitz enclosure [sin m - r, sin m + r]
has width at least 2 and contains 0, so [-1, 1] is never wider. A rounded ball whose midpoint has
exponent above p has r >= 1, so the second condition only matters for exact inputs (a float such as
`1e300`). Then `sin(2^100000000) + 2` is enclosed in [1, 3] and decided `NonZero` at once.

Operations:

- `+ - *`: the usual midpoint-radius formulas. `1/x` requires `|m| > r`, radius `r / (|m| (|m| - r))`.
- exp: L = exp(m + r). log, sqrt: require `m - r > 0`, L = 1/(m - r), 1/(2 sqrt(m - r)).
  sin, cos (guarded as above), atan, tanh: L = 1. erf: L = 2 (true bound 1.13). sinh, cosh:
  L = cosh(|m| + r). asin, acos: require `|m| + r < 1`, L = 1/sqrt(1 - (|m|+r)^2). asinh, acosh,
  atanh by MPFR's own functions (revision 7rt3, red team 3 numeric m1): asinh with
  L = min(1, 1/d), d = `|m| - r` shrunk by `2^(4-p)` relative (the least |x| on the ball); acosh
  requires `lo > 1` (lo = `m - r` shrunk the same way), L = 1/sqrt((lo - 1)(lo + 1)) (`lo - 1` is
  exact for lo <= 2, Sterbenz); atanh requires `a = |m| + r < 1`, L = 1/((1 - a)(1 + a)); each L
  inflated by `2^(4-p)` relative, as asin's. Revision 7rt2 used the log formulas, which lose the
  argument's relative precision at every precision the test reaches: `x + sqrt(x^2 + 1)` cancels for
  x << -1 (no enclosure of `asinh(-2^600) + 400` even at 1024 bits), `1 + x` rounds to 1 for
  |x| < 2^-p (`asinh(10^-400) 10^400 + 1/2` and `atanh(...)` likewise), and x^2 leaves the window past
  2^(2^27). tan = sin/cos (no enclosure when the guard fails). Integer powers by binary powering for
  every exponent that fits in i64 (at most 64 squarings and 64 products, each checked by R1 and R2;
  revision 7rt3, red team 3 numeric m2: revision 7rt2 refused |k| > 2^32, so
  `(1 + 1/10^9)^4294967297 - 5` was `UndecidedZero` while the same exponent written `2^32 + 1` was
  decided); an integer exponent beyond i64 by `exp(k log |a|)`, negated when the ball of a lies below 0
  and k is odd, inverted for k < 0 (a ball across 0 is too wide). An exponent that is not an integer
  literal but whose value is an exact integer k in i64 is an integer power too (revision 8, review 8
  m1): `small_int_value` reads an integer literal under negations, sums, differences and products,
  each operation checked in i64 (`-2`, `(-(-(5)))`, `3 - 1`, `2*3 - 3`), by a walk of at most 64 nodes
  and 8 levels (no allocation; a constant amount of work per power node, covered by its charge); any
  other shape, a deeper or larger one, or an overflow is not an integer value. Such an exponent gives
  binary powering for |k| <= 3 and `exp(k log |a|)` with the sign and the inverse of the beyond-i64
  case above for |k| >= 4, and a^0 = 1. Revision 7rt3 sent every non-literal exponent
  to `exp(b log a)`, which has no enclosure for a < 0, so `(-3 + 1)^(-2)` (the parsers read `(-2)` as
  `Neg(Integer 2)`) had none while the normal form's `c^-2` had one; a hint (§2.5) or a kernel made
  from an input node renders the input's syntax, so `(1)/(x + sin((-2)^(-2)) - 1/4)` at 0 was
  `UndecidedZero` where HEAD gives `1/(sin(1/4) - 1/4) - x/(sin(1/4) - 1/4)^2`. Other powers
  `exp(b log a)` (requires a > 0). pi and e: MPFR constants. Integers and rationals:
  `with_val_integer` / `with_val_rational` (correctly rounded; both convert through binary digits,
  in time linear in the size). Anything else (gamma, zeta, named functions, indeterminates, `I`, a
  log of a non-positive ball): no enclosure.
- Exact literals and the exact zero (revision 7rt4, red team 4 numeric m3). A literal whose value is
  exact at p bits, an integer of at most p bits or a rational `n / 2^k` with n of at most p bits, is
  an exact ball (radius 0); revision 8 gave every literal the radius `|m| 2^-p`, so `1 - 1` was the
  ball `0 +- 2^-63` and never the exact 0. A sum or product of exact balls whose midpoint is exactly 0
  is the exact 0 (R2's `a == -b` and the radius formulas give radius 0). On the exact 0: `sqrt(0) =
  0`; an integer power k > 0 is 0 (binary powering, and `exp(k log |a|)` beyond |k| = 3 or beyond
  i64 returns 0 at once), k < 0 has no value (`Never`); a real power `0^b` is 0 when the ball of b
  lies above 0, has no value when it lies at or below 0 (`Never`), and is `TooWide` across 0; log of
  it has no value (`Never`). So a hint or a kernel's source that writes an exactly-zero operand as
  `1 - 1` under sqrt, a fractional power or a power with abs(k) >= 4 keeps the enclosure the normal
  form has: `(1)/(x + sin(sqrt(1-1) + 1) - 1/2)` at 0 is `Ok` with gp's `sin(1) - 1/2 =
  0.34147098480789650665` in the denominator (revision 8: `UndecidedZero`, TooWide at every
  precision), and the hinted 6-level tower of red team 4 is `Ok` again (as in revision 7rt2). A
  rational with another denominator (`1/3 - 1/3`) keeps a radius, so its difference is a ball around
  0: such a hint stays `UndecidedZero`, a known cost (§7 item 10), as before.
- `exp(k log a)` with k an exact integer value (`small_int_value` of either factor of the product) is
  enclosed as the integer power a^k (binary powering for |k| <= 3, else `exp(k log |a|)` with the sign,
  a^0 = 1; a's ball must exclude 0), without evaluating `log a` (revision 7rt4). The identity holds for
  every nonzero real or complex a and every branch of log: branches differ by 2 pi i m, and k times that
  is a multiple of 2 pi i. It is the normal form's side of the gap above: a constant base to a series
  exponent renders as `exp(r log l)` (§2.5), so at a point where r is the integer k the constant term
  `exp(k log c)` of a negative c had no enclosure (log of a negative ball), while the input syntax
  `c^k` had one. Which of the two renderings a constant has depends on the hints, and the exact kernel
  counts of revision 7rt4 (§2.2) set fewer hints on one of review 8's random towers, `((-3) +
  tanh(((1/2) + sqrt(((1/2) + sqrt(((1/2) + log(((-3) + sin(x))^(-2)))^x))^3))^(2/3)))^(x+1)` at -2,
  order 0, which then was `UndecidedZero` (revision 8: `Ok`); with this rule it is `Ok` and equal to gp's
  value, and another tower that revision 8 refused is `Ok` too. Charged within the existing units: the
  nodes `exp`, `*`, `log` and the leaf of k are all counted by `ball_cost` (at least `5u`), the power at
  most four operations. Checked against gp at 3500 digits on 4500 random constants of this form
  (negative and positive rational, transcendental and near-1 bases, integer-valued exponents from -9
  to 9 written as literals, negations, sums and products; `rt4num/gen_exppow.py`, `check_rigor.py`):
  13 500 enclosures at three precisions, 0 outside the ball.
- erf uses `RealMPFR::from_float(x.as_float().clone().erf())` (inherent rug method, no new
  dependency), with L = 2 and the exact zero of R2 at 0, only when `|m| < X_p`; above the threshold
  it takes the cheap enclosure of the paragraph "erf" below (revision 7); asinh, acosh and atanh use
  rug's `Float::asinh`, `acosh` and `atanh` the same way (revision 7rt3; `RealMPFR` has no direct
  methods). The revision 2 prototype had no `Erf` arm, so
  `1/(x + erf(1) - 1/2)` was `UndecidedZero` (review 3, m4); `EXP/rev3/proto4/ball.rs` has it, and
  that input now expands with c0 = `-2/(1 - 2 erf(1))`.
- `ball_cost` units per node, with `u = (p/64 + 1)^2 * 8`: integer and rational leaves `u + bits/64 +
  1` (the conversion); floats, symbols, `+ - * /`, functions `u`; `^` with an integer exponent k
  `u * (2 * min(bits(k), 64) + 1)` (binary powering; revision 7rt3: 64 in place of 33, and an
  exponent beyond i64 is charged the same, at least the `3u` of its `exp(k log |a|)`), other `^` `4u`
  (revision 8: at most four operations, `exp(b log a)` or, for an exponent with an exact integer
  value, three products and an inverse or log, product, exp and inverse; revision 7rt3 charged `3u`;
  `EXP/rev8/powtime_r8.txt`: the slowest of eight such constants at 8.3e7 units/s, at 64 bits); cos `t = ((2p + 64)/64 +
  1)^2 * 8` (the argument reduction at exponent at most p + 64), sin `((3p + 64)/64 + 1)^2 * 8`
  (revision 7rt1, red team 1 m1: MPFR's `sin.c` computes `sqrt(1 - cos^2)` and doubles its working
  precision when the result rounds to +-1, so a midpoint within an ulp of `(k + 1/2) pi` costs an
  operation at about 3p + 64 bits; with `t` it ran at 1.6e7 units/s at 256 bits), tan `3t`; asinh,
  acosh, atanh `6u`;
  erf `erf_units(p, a)` (below, revision 7); other unary functions `2u`; plus `BALL_NODE = 64` per
  distinct node for its memory (revision 6: its memo entry, its stack entry and its entry in the
  walk's set; at p = 64 a memo entry holds two MPFR values, about 160 bytes with its hash slot,
  which `u = 32` alone left at 5 bytes per unit; earlier text of this section said 32, the
  prototype has 64). Every MPFR call of `enclose_real` is covered by its node's units (each distinct
  node is counted and evaluated once). Each call other than erf costs at most a constant times the
  cost of an operation at precision 2p + 64 (sin: 3p + 64); erf costs at most its `erf_units` at the
  rates of the scans below.

**erf (revision 7, review 7 B1).** MPFR's erf (mpfr 4.2.2, `erf.c`) returns +-1 at once when
`x^2 / ln 2 > p + 1`, and otherwise sums the Taylor series at 0 (`mpfr_erf_0`) at a working precision
of `p + x^2 / ln 2 + 8 + log2 p` bits, with at least `x^2` terms, inside a Ziv loop. So its cost grows
like `x^2` times an operation at up to about 2p bits, up to the threshold where erf rounds to +-1:
review 7 measured 6.4 ms for `erf(975/38)` at 1024 bits, which revision 6 charged 4656 units, and an
11.6 KB input whose zero test evaluates `erf(975/38) + erf(-975/38)` once per indeterminate group ran
31.7 s under the budget. The enclosure now has two regimes, with the threshold
`X_p = ceil(2 sqrt((p + 16) ln 2)) / 2` (7.5, 14 and 27 at 64, 256 and 1024 bits, exact in binary):

- **Above the threshold, no MPFR call.** If the ball's lower end `m - r` exceeds `X_p`, the
  enclosure is the ball `1 +- 2^-(p+16)`, and `-1 +- 2^-(p+16)` when `m + r < -X_p`. Proof: for
  `x >= X >= 1`, `0 < 1 - erf(x) = erfc(x) < exp(-x^2) / (x sqrt(pi)) <= exp(-X^2) <= 2^-(p+16)`
  (Abramowitz and Stegun 7.1.13, the upper bound `1/(x + sqrt(x^2 + 4/pi))` of `exp(x^2) int_x^oo
  exp(-t^2) dt`), and erf is odd. `m - r` and `m + r` are rounded to nearest, which is monotone, and
  `X_p` is representable, so `rounded(m - r) > X_p` implies `m - r > X_p` exactly. MPFR at precision p
  returns +-1 there (`X_p^2 / ln 2 > p + 1`, its own cutoff), and the ball the old path built around it
  had a radius of at least 2^-p, so nothing is lost: `erf(28) - 1` is `Unknown` both ways (erfc(28) is
  about 6.6e-343, below 2^-1024).
- **A ball across `+-X_p`** (`|m| >= X_p` but not the case above, so `r >= |m| - X_p`): the range of
  erf, `[-1, 1]` (midpoint 0, radius 1), as sin and cos past their guard.
- **Below the threshold** (`|m| < X_p`): MPFR's erf with `L = 2`, as before. Its cost grows with `x^2`
  until MPFR's own cutoff `x^2 / ln 2 > p + 1`, which lies below `X_p`, and `ball_cost` charges it a
  priori by
  `erf_units(p, a) = ERF_FACTOR u (ERF_BASE + ceil(min(a, X_p)^2))`, with `ERF_FACTOR = 2`,
  `ERF_BASE = 8` and `u = (p/64 + 1)^2 * 8`. The bound `a` is known a priori when the argument of the
  erf node is a literal (an `Integer`, `Rational` or `Real`, under at most 4 negations):
  `literal_abs_bound` gives `lo <= |v| <= hi` from the bit lengths and the top 64 bits of the numerator
  and the denominator, in O(1) whatever their size, and `a = hi`; a literal argument with
  `lo >= X_p + 1` takes the first regime and is charged `2u` (revision 7rt1, red team 1 m2: revision 7
  converted the whole literal to an f64, an uncharged copy and division of both parts, 611 us per erf
  node for a 2 x 1M-bit rational; now 34 ns, and on 210 rationals up to 2^2063 each exact value lies in
  its bounds, which are within a factor 1 + 1e-12; `EXP/rev7rt1/proto9/litbound.rs`). For any other
  argument `a = X_p`. So
  `erf(1/2)` is charged 576 units at 64 bits and 41 616 at 1024 bits, `erf(975/38)` 3.08e6 at 1024
  bits, and erf of a computed argument 4160, 81 600 and 3.41e6 units at 64, 256 and 1024 bits.

The factor and the base come from the scan below. The charge review 7 suggested, `u (2 + a^2)`, and
twice it left erf at 1.2e7 and 2.1e7 units/s at 64 bits for `erf(18/13)`, where MPFR retries its Ziv
loop (15.5 us for 192 and 320 units with `BALL_NODE`); with base 8 the slowest erf rate is at least
that of the other functions at the same precision.

Scan of every enclosed operation (revision 7, review 7 B1 item 2; `EXP/rev7/proto8/fnscan.rs`, output
`EXP/rev7/fnscan_r7.txt`, and `EXP/rev7/proto8/erfbins.rs`, output `EXP/rev7/erfbins_r7.txt`; release,
each argument timed as the minimum of three batches, through `Ball::unary` and the binary `Ball`
operations, so the radius work is included). The arguments: short mantissas `k/65537 * 2^e` and
p-bit pseudo-random mantissas, for exponents e from `-(W - 2)` to `W - 2` within each operation's
domain and guard; rationals `n/d` with `d <= 12` (`d <= 40` for erf) over the range where the
operation is slowest, whose balls have full p-bit mantissas; and values just inside every guard
(`2^27 - 1` for exp, sinh and cosh; a p-bit mantissa times `2^(p+64)` for sin, cos and tan, as an exact
ball; `1 - 2^-k` for asin, acos and atanh and `1 + 2^-k` for acosh, k up to p - 2; for erf `X_p - 2^-k`
and `X_p` itself, rounded and exact); and their negatives where the domain allows. For erf the binned
scan adds 300 random p-bit arguments in each half-unit bin of `[0, X_p + 1)`. A rate is the node's
units (`unary_units + BALL_NODE`; for erf the charge of its own argument as a literal, the smallest
that applies) over the time. Slowest rate in units per second over three runs of `runall7.sh` (`EXP/rev7/run1/`, `run2/` and
`EXP/rev7/` itself, files `fnscan_r7.txt` and `erfbins_r7.txt`, and the earlier `erfbins_r7_run1.txt`),
and the longest single call at any precision:

| operation | 64 bits | 256 bits | 1024 bits | longest call |
|---|---|---|---|---|
| exp | 4.2e7 | 6.6e7 | 2.4e8 | 19 us |
| log | 3.6e7 | 7.0e7 | 3.2e8 | 15 us |
| sqrt | 1.0e8 | 1.8e8 | 9.6e8 | 4.9 us |
| sin, cos | 4.5e7, 5.3e7 | 1.0e8, 1.5e8 | 4.7e8, 9.1e8 | 20 us |
| tan | 2.5e7 | 1.5e8 | 8.7e8 | 32 us |
| atan | 2.1e7 | 3.2e7 | 1.6e8 | 29 us |
| erf, charged by its argument | 4.0e7 (`erf(5.96)`, 71 us; see below) | 8.9e7 (`erf(346/30)`, 639 us) | 1.4e8 (`erf(2.17)`, 438 us) | 6.2 ms (`erf(24.6)` at 1024 bits, charged 2.84e6) |
| erf above `X_p` (cheap enclosure; a literal at or above `X_p + 1` is charged `2u + BALL_NODE`) | 2.1e8 | 7.7e8 | 7.8e9 | 0.6 us |
| tanh | 4.6e7 | 8.7e7 | 2.3e8 | 20 us |
| sinh, cosh | 2.4e7, 3.0e7 | 3.5e7, 5.4e7 | 5.5e7, 1.0e8 | 85 us |
| asin, acos | 4.9e7, 3.7e7 | 4.1e7, 4.4e7 | 9.8e7, 9.0e7 | 52 us |
| asinh, acosh, atanh | 3.5e7, 5.2e7, 4.6e7 | 1.2e8, 1.3e8, 1.2e8 | 3.1e8, 4.6e8, 4.6e8 | 45 us |
| abs, neg | 1.2e9 | 3.6e9 | 3.3e10 | 0.1 us |
| `+`, `*`, `/` | 2.8e8, 1.9e8, 7.7e7 | 6.0e8, 3.4e8, 1.6e8 | 3.7e9, 1.8e9, 7.1e8 | 3.4 us |
| integer power, exponent up to 2^32 | 8.2e7 | 3.8e8 | 1.9e9 | 80 us |
| other power, `exp(b log a)` | 3.9e7 | 7.1e7 | 2.1e8 | 34 us |
| integer and rational leaves, up to 2 x 2^20 bits | 6.9e7, 5.9e7 | 6.6e7, 5.7e7 | 7.7e7, 3.3e7 | 1.1 ms |
| pi, e | 2.0e9, 1.4e8 | 4.4e9, 1.2e8 | 4.2e10, 2.9e8 | 8.3 us |

The rerun on the revision 7rt3 prototype (`EXP/rev7rt3/fnscan_rt3.txt`, a machine shared with another
workload for part of the run), with MPFR's asinh, acosh and atanh, gives for them 2.15e7, 2.60e7 and
4.68e7 at 64 bits, 5.0e7, 4.9e7 and 8.1e7 at 256 and 1.6e8, 9.7e7 and 1.9e8 at 1024 (the longest call
0.14 ms, acosh at `1 + 2^-512`); the integer power over the grid (exponents up to 2^32 - 1) 6.5e7,
3.7e8 and 1.3e9; the slowest cell of all 1.57e7 units/s (sinh at 64 bits; 2.4e7 in revision 7rt2's run).
Each cell is the slowest of thousands of arguments, each timed as the minimum of three batches, so one
preempted argument lowers a cell: in the third run `erf(110/20)` at 64 bits scanned at 141 us (1.8e7
units/s), while 1000 calls of it take 11.3 us at the median and 91 us at most (2.3e8 units/s at the
median; `EXP/rev7/proto8/onecall.rs`, output `EXP/rev7/onecall_r7.txt`, which also times the other
slowest cells: `erf(139/22)` 40 us, `tan(262/7)` 3.7 us, `atan(44/3)` 3.8 us, `sinh(77/9)` 5.5 us,
`erf(975/38)` at 1024 bits 4.2 ms at the median). Apart from that outlier every operation charges at
least 2.1e7 units/s in every run (atan at 64 bits, where the radius work of the ball dominates:
6.1 us for 128 units), above the floor of 1e7 units/s that `TAYLOR_WORK` assumes (§5.3), so a numeric
test charged 1e8 units runs at most about 5 s. Revision 6's charge of erf, `2u`, ran at 4.9e6 units/s
at 64 bits and 7.3e5 at 1024 bits (review 7). The rerun on the revision 7rt1 prototype
(`EXP/rev7rt1/fnscan_rt1.txt`, `erfbins_rt1.txt`) gives the slowest rate 2.6e7 (sinh at 64 bits); some
cells scanned lower than in revision 7's three runs (exp and log at 64 bits at 2.8e7 and 2.7e7, the
integer leaf at 256 bits at 4.0e7), the spread between runs of a minimum over thousands of timings,
and sin scanned higher by its new charge.

The grid above has no midpoint where MPFR raises its working precision (red team 1, m1). Revision 7rt1
adds a scan of those arguments (`EXP/rev7rt1/proto9/nearscan.rs`, in `EXP/rev7rt1/redteam1_rt1.txt`),
timed as the median of 200 calls (the minimum of batch means hides the slow calls): sin, cos and tan at
`round_p((k + 1/2) pi)` and `round_p(k pi)` for k below 60 (MPFR's sin doubles its precision when the
result rounds to +-1); log and acosh at `1 +- 2^-k`, asinh, atanh, sinh and tanh at `L 2^-k` with
`L = 3/7`, for k = p/2, p - 2, p - 1 and 1022 (log adds the cancellation near 1 to its precision).
Slowest median rate in units per second, with the charge of revision 7rt1, and the longest single call:

| operation | 64 bits | 256 bits | 1024 bits | longest call |
|---|---|---|---|---|
| sin near +-1 (revision 7's charge: 2.7e7, 1.6e7, 4.2e7) | 3.5e7 | 3.4e7 | 6.2e7 | 3.1 ms (one preempted call; the median there 7.6 us) |
| cos near 0, tan near both | 1.0e8, 3.6e7 | 3.6e8, 6.1e7 | 1.1e9, 7.7e7 | 2.0 ms |
| log at `1 +- 2^-k` | 2.6e7 | 2.5e7 | 4.9e7 | 0.53 ms |
| acosh at `1 + 2^-k`; asinh, atanh at `L 2^-k` | 3.4e7, 3.6e7, 3.5e7 | 1.2e8, 1.3e8, 5.8e7 | 4.5e8, 1.4e8, 1.4e8 | 0.63 ms |
| sinh, tanh at `L 2^-k` | 1.3e8, 2.0e8 | 5.4e8, 8.9e8 | 3.2e9, 5.3e9 | 33 us |

Revision 7rt3 (red team 3, numeric m1) adds asinh at `-2^k` and `-L 2^k` and acosh at `L 2^k` (k = p/2,
p - 1, 1022 and 2^20: the large arguments where the log formulas cancelled or left the window) and runs
the scan on MPFR's own functions (`EXP/rev7rt3/proto11/nearscan.rs`, in `redteam1_rt3.txt`): the slowest
medians are acosh at `1 + 2^-k` 3.7e7, 5.2e7 and 9.9e7 units/s at 64, 256 and 1024 bits (0.37 ms the
longest call), asinh at `-2^k` 1.7e8 and more, acosh at `L 2^k` 1.6e8 and more, atanh at `L 2^-k` 3.1e8
and more.

Revision 7rt4 (red team 4 numeric, m2) adds short-mantissa arguments, which every scan above missed
(their rationals `n/d`, `k/65537 2^e` and random p-bit mantissas all round to full mantissas):
`M 2^-k` with M odd below 2^8 and k from p/4 to p + 1, and `1 +- M 2^-k`. There MPFR's Ziv loop
retries two to four times, because `1 + x^2/2` (cosh) or `1 + x` (the exp inside sinh and cosh) is
representable and the next term lies past the working precision, and the ball calls MPFR twice for
sinh and cosh (`cosh(|m| + r)` for L, and f(m)), both retrying. Slowest medians of 1001 calls, node
units over the node's own time (`EXP/rev7rt4/rt4num_repro.txt`, red team 4's `rt4.rs scan` linked
against proto13): `cosh(3/2^130)` at 256 bits 1.60e7 units/s (464 units in 29 us), `sinh` there
1.74e7, `cosh(1/2^31)` at 64 bits 1.87e7, `cosh(15/2^470)` at 1024 bits 1.81e7, and the revision 8
power path `(1 - 2^-30)^(-(5))` (`exp(k log |a|)`) at 64 bits 1.99e7. Every other function of their
scans (38 114 arguments) is at 2.9e7 or more, the acosh, asinh, asin and log near 1 at 2.1e7 or
more. The charges stay (`2u` for sinh and cosh, `4u` for the power): the slowest median of every scan
is now 1.6e7 units/s (cosh at 256 bits on a short mantissa), 1.6 times the floor, so any 1e8 units of
numeric work run at most 6.3 s; end to end the node is diluted by its leaf, the combining nodes and
the other precisions (200 such kernels in 64 groups, 10.7 KB, are `UndecidedZero` after 0.16 s at
2.7e7 units, 1.7e8 units/s). Charging sinh and cosh as two operations at `2p + 64` bits would put
them at 4e7 or more; it is not needed for the bar and is left to the port's calibration.
Before revision 7rt4 the slowest median of either scan was 2.5e7 units/s (log at 256 bits). End to
end, the red team's input `(1)/(x + (sin(L) + 1)*(1+y1)*...*(1+y13))` with L the 1023-bit numerator of
a convergent of pi/2 (`sin(L) = -1 + 2^-2049`, 8192 indeterminate groups, each tested at 64, 256 and
1024 bits) is refused at 1e8 after 0.79 s with 15 MB; with 12 factors (4096 groups) it is refused
after 0.72 s (revision 7: `UndecidedZero` after 0.93 s; both outcomes are honest).

New dependency: `rustmath-reals = { path = "../rustmath-reals" }` in `rustmath-symbolic/Cargo.toml`.
No cycle: rustmath-reals depends on core, integers, rationals, num-traits, rug only. This consumes an
existing numeric crate instead of wrapping rug privately.

Effect (prototype): `1/(2 + sin(x))` and `log(1 + sin(x))` at x = 1, `1/(x + e - pi)` at 0,
`1/(x + sin(1) + I)` at 0 expand; `1/(x + sin(1)^2 + cos(1)^2 - 1)` at 0 is
`Err(UndecidedZero(-1 + cos(1)^2 + sin(1)^2))`; `1/(x + sqrt(-4) - 2*I)` is `UndecidedZero`.
Revision 2 prototype (`EXP/rev2/revchecks3.out`, debug): the five underflow and overflow inputs of
§4.1 are `UndecidedZero` in 2 to 13 ms; with budget 1e8, `1/(x + sin(2^100000000) + 2)` and its
`cos` twin are `Ok` in 2 ms, the `tan` twin and `1/(x + sin(2^1000000000) + 2)` are `UndecidedZero`
in 2 ms (revision 1: 111.7 s release for the first, over 30 s for the last).

### 2.4 Series operations (`series_engine.rs`)

Context: `struct Ctx<'a> { var: &'a Symbol, xser: Laurent, p: i64, work: &'a mut Work }`. Every
`SeriesConst` operation receives `cx.work`. Sparse loops: only structurally nonzero terms are
visited. A recurrence that fills a dense buffer of n slots first calls `cx.alloc(n)`, which charges
`16 n` units, then allocates; its output is compacted to sparse terms. Before step k of any
recurrence, the step's cost `sum_j words(u_j) * words(g_{k-j})` (plus a base of 1) is charged; the
constant products, sums and scalings inside the step are charged again by the arithmetic layer of
§2.7, by the sizes they normalize (review 3, B1: a step `g_k = (1/k) ...` on coefficients with large
numerators and denominators costs a quadratic gcd per scaling). Every loop of a step visits only
the offsets j <= k of the sparse argument (the offsets are sorted; each loop breaks at the first offset
above k, and `log` finds w_k by a binary search), so the work of a step is bounded by its charge
(revision 7rt2, red team 2 series S-B1: revision 7rt1's `log` scanned every offset at every step to find
w_k and to sum the charge, which counted only the offsets below k; with 131 072 offsets above
10^6, `log(1 + x^1000000*(1+x)*(1+x^2)*...*(1+x^65536))/x^1131072` at the REPL's order 0, 203 bytes,
ran 78 s before the budget refused it, about 17 units per step for a scan of every offset).

Copies of terms (review 4, B1). A coefficient is shared (§2.2), so a series operation that copies a
term copies a pointer, whatever the size of the coefficient. Each such operation charges
`SER_TERM = 6` units per term it copies (a 16-byte slot; revision 5 charged 4, 4 bytes per unit),
plus the base of `charge`: `add` (the terms of both operands, common or not), `neg` (which also
charges `copy_units` of each coefficient it negates), `shift`, `split0`, `offsets` and the slices
`bs.t[1..]` of `pow_int`; the variable's series is copied at each occurrence of the variable for 8
units (two terms). `coef` (a binary search) and `exact_constant` copy one pointer.
Revision 3 copied each coefficient in full for one unit per term, so a 344-character sum that added
`x` 160 times to a 262 144-term constant ran 15 s, and 100 occurrences of `x^0` at such a point 8.4 s
(review 4); now both take the time of building the constant (0.25 s).

| operation | precondition | result valuation and precision | recurrence |
|---|---|---|---|
| `add(a, b)` | | merge, prec = min(pa, pb), then `trunc(P)` | termwise |
| `neg`, `scale(k)` | | unchanged | termwise |
| `mul(a, b)` | | val = va + vb (checked), prec = min(va + pb, vb + pa) (`Exact` absorbs; `Prec::add`); computed up to the cutoff `min(prec, Abs(max(P, val + 1)))` (at least the first term, review 2 m1), or in full for small exact operands (the cutoff `Exact`, which drops no pair; §2.1); `Exact` only when no pair was dropped, else the finite cutoff (revision 7rt4) | first pass: sum the pair weights of the contributing pairs (`wa + wb` when either coefficient is one word, else `wa wb`, review 6 m4), stopping as soon as the sum passes the remaining budget, and charge it (revision 5: the pass was O(pairs) before any charge, and `((1/(1-x))^2)/x^250000` counted 3e10 pairs in 34.7 s); second pass: products, accumulated in a dense buffer when the span (checked, revision 7rt2) is at most `2 * pairs + 64` slots (charged 16 per slot), else (a span that does not fit in i64 included) by a heap merge of the rows of the shorter operand (revision 6, review 6 B1: Johnson 1974; the products come out in increasing exponent, so the only containers are a heap of one cursor per row and the output; charged first `SER_TERM + SEARCH_CMP (bits(rows) + 1)` per pair and 16 per row; revision 5 accumulated every pair in an uncharged `BTreeMap`, and a 4.6 KB input held 2.53 GB) |
| `inv(b)` | b not empty (exact empty: `DivisionByZero`; truncated empty: `NeedPrecision`); c0 `NonZero` (strict inverse) | val = -vb (checked: vb = i64::MIN is the overflow error; -vb = i64::MAX is a legal exponent since revision 7rt4); prec = pb - 2 vb in two `Prec::add` steps, at most P, or P when b is exact; an exact monomial inverts exactly | g0 = 1/c0, g_k = -g0 sum_{j=1..k} b_j g_{k-j} over the span `prec - val`; the offsets j = e - vb as unsigned distances, those below the span only (revision 7rt3) |
| `pow_int(b, n)` | n any Integer | see "integer powers" below | |
| `pow_const(b, alpha)` | alpha a non-integer constant; b not empty; c0 `NonZero`; vb = 0, else see "non-integer powers" below | val 0, prec = min(pb, P) | w = b/c0; Miller: g0 = 1, g_k = (1/k) sum_{j=1..k} ((alpha+1) j - k) w_j g_{k-j} (Knuth §4.7 eq. (9)); times `pow_const(c0, alpha)` |
| `exp(b)` | vb >= 0 (vb < 0 with lead `NonZero`: `NotLaurent`; lead `Unknown`: `UndecidedZero`; b empty and truncated with precision below 0: `NeedPrecision`, §2.1, revision 7rt3) | val 0, prec = min(pb, P) | b = b0 + u; E_0 = 1, E_k = (1/k) sum j u_j E_{k-j}; times `apply(Exp, b0)` |
| `log(b)` | b not empty; c0 `NonZero`; vb = 0 (else `NotLaurent` log singularity) | val 0, prec = min(pb, P) | w = b/c0; k L_k = k w_k - sum_{j=1..k-1} j L_j w_{k-j}; plus `apply(Log, c0)`; w_k by a binary search of the sorted offsets, the sum over the offsets below k only (revision 7rt2) |
| `sin_cos(b)`, `sinh_cosh(b)` | vb >= 0, as `exp` (an empty b of negative precision: `NeedPrecision`) | prec = min(pb, P) | b = b0 + u; S_0 = 0, C_0 = 1, S_k = (1/k) sum j u_j C_{k-j}; C_k = -(1/k) sum j u_j S_{k-j} (trig) or +(1/k) sum j u_j S_{k-j} (hyperbolic); sin(b0 + u) = sin b0 C + cos b0 S, cos(b0 + u) = cos b0 C - sin b0 S; sinh(b0 + u) = sinh b0 C + cosh b0 S, cosh(b0 + u) = cosh b0 C + sinh b0 S |
| `tan`, `tanh` | | | sin/cos via `mul(s, inv(c))` (a pole if cos b0 = 0 is decided, e.g. at pi/2) |
| arctan, arcsin, arccos, arcsinh, arccosh, arctanh, erf | vb >= 0, as `exp` (an empty b of negative precision: `NeedPrecision`) | prec = pb (integration adds one, the derivative loses one) | G = f(b0) + integral of Q, Q = b' / (1 + b^2), b' (1 - b^2)^(-1/2), -b' (1 - b^2)^(-1/2), b' (1 + b^2)^(-1/2), b' (b - 1)^(-1/2) (b + 1)^(-1/2) (principal branch, not (b^2 - 1)^(-1/2)), b' / (1 - b^2), (2/sqrt(pi)) exp(-b^2) b'. Q must have valuation >= 0 (else `NotLaurent` log branch point; `pow_const` already refuses arcsin at 1) |
| `abs`, `sign` | **real variable and real point assumed** (documented): all coefficients rational, vb even | abs: sign(c0) b; sign: exact constant sign(c0) | odd vb: `NotLaurent`; non-rational coefficients: `Unsupported` |
| `deriv`, `integrate` | integrate: no h^-1 term (an empty integrand of negative precision may hold one: `NeedPrecision`, revision 7rt3) | prec - 1, prec + 1 | formal |

Integer powers `pow_int(b, n)` (in this order):

1. n = 0: exact 1. b exact zero: 0 for n > 0, `DivisionByZero` for n < 0.
2. b an exact monomial `c h^v`: exact `pow_const(c, n) h^(n v)` (n v checked), kept whatever P is;
   n v = i64::MAX is a legal exponent of the exact term (revision 7rt4; revision 8 stored it at the
   sentinel's value, and `add` then dropped it from a sum it kept exact).
   When n v > 0 does not fit in i64 the power lies above every working precision and is `O(h^P)`;
   only a pole of that order is the overflow error (revision 7rt2, red team 2 structure m3: revision
   7rt1 returned the error, so `x^(2^100) + x` at 0 was `Unsupported` where HEAD's value is x).
3. n < 0: `pow_int(inv(b), -n)` (strict inverse of the lead).
4. b empty and truncated, `O(h^q)`: `O(h^(n q))` for q > 0 (capped at P), else `NeedPrecision`.
5. vb > 0 and n vb >= P (n compared as an `Integer`): the leading term
   `pow_const(c0, n) h^(n vb) + O(h^(n vb + 1))` (review 2, m1), or `O(h^P)` when n vb does not fit
   in i64. The precision `Abs(n vb + 1)` is checked: at n vb = i64::MAX the term has no finite
   precision above it, and the power is `O(h^(n vb))`, a true claim (revision 8 returned the precision
   `n vb + 1 = i64::MAX`, which read as `EXACT`, for n vb = i64::MAX - 1). The term is exact whatever c0 is (an undecided c0 is reported by the strict inverse that
   needs it, not here), and its precision `n vb + 1` is at most `n vb + (pb - vb)`, the true
   precision of b^n. Revision 1 returned `O(h^P)`, which forgot the valuation: the inverse of
   `sin(x)^200` met an empty series, the `NeedPrecision` doubling stopped at extra 128, and
   `1/sin(x)^200` was `PrecisionExhausted(131)`.
6. n <= `POW_BINARY_MAX = 64`: binary powering with `mul` (exact canonical coefficients).
7. n > 64: write b = h^vb (c0 + u') with u' of valuation >= 1, and compute S = min(relative
   precision of b, P - n vb) slots (charged; `P - n vb` saturating), then shift by n vb. Both branches
   return the series of precision S, `O(h^(n vb + S))`, never `Exact` (§2.1):
   - c0 decided `NonZero`: Miller's recurrence with alpha = n on w = u'/c0, times `pow_const(c0, n)`
     (size-capped, kernel fallback). Cost O(S^2) independent of n: `(1+x)^10000` at order 2000 takes
     11.5 ms (release; revision 0 binary powering: 12.9 s). When c0 has several terms the output
     contains `c0^n * (1/c0)^k`, which is correct but not reduced: `(x + sin(1) + cos(1))^4096` at 0
     has `4096 * (cos(1) + sin(1))^4096 / (cos(1) + sin(1))` as its x coefficient.
   - c0 `Unknown`: the binomial theorem, no division and no zero test:
     `sum_{m < min(S, n + 1)} C(n, m) pow_const(c0, n - m) u'^m` (the terms m > n are zero, and
     `pow_const(c0, n - m)` with n - m < 0 would invert c0). Cost O(S^3); charged. The loop runs in the
     caller's context with the working precision set to S, so every product and sum charges the
     caller's `Work` (revision 7rt1, red team 1 m3: the prototype built a second context with a dummy
     `Work`, which only its thread-local counter made harmless). The partial sum is truncated at S and
     takes precision S, as Miller's result does (`from_dense(0, g, S)`). Revision 7rt1 returned
     `acc.trunc(S)`, and `trunc` leaves an exact series of at most 64 terms exact, so the result
     claimed to be the whole polynomial (red team 2 series, S-B3): after a pole shift the driver saw no
     loss to retry, and `taylor((x+1+sin(y))^100/x^5, x, 0, 3)` returned the exponents -5 to -2 as a
     complete finite series, without the nonzero terms -1 to 3. Now it returns all nine, equal to
     gp's `binomial(100, e + 5) (1 + s)^(95 - e)` with s = sin(y), and the 76 terms at order 70.

Non-integer powers of a series of nonzero valuation (review 1, m1): `non_integer_status(alpha)`:
`NonZero` gives `NotLaurent` ("a non-integer power of an argument of valuation v has a branch point
there"; no engine message formats a constant, since a constant may hold a 2^20-bit integer whose
num-bigint decimal conversion is quadratic); `Zero` (alpha is an integer after all) gives `pow_int`;
`Unknown` gives `UndecidedZero(alpha - n)`, or `UndecidedZero(alpha)` when no nearest integer n was
computed (|mid(alpha)| >= 2^1024). So `x^pi`, `x^y` and `x^(1/2 + I)` at 0 are `NotLaurent`, and
`x^(sin(1)^2 + cos(1)^2)` at 0 is `UndecidedZero(-1 + cos(1)^2 + sin(1)^2)`.

Branch at the point (revision 7rt4, stated; the rule is unchanged since revision 1). `pow_const`, `log`
and the atan family take the principal value of their constant c0 (`apply`, §2.2) and expand the
branch that is analytic at the point through it (Miller's recurrence on `b/c0`, the log recurrence on
`b/c0`). When c0 lies on the branch cut (a negative real c0 for log and the non-integer powers) and a
higher coefficient of b is not real, the principal-branch function of a real variable is discontinuous
at the point, and the expansion is that of the branch whose value at the point is principal, which is
what gp computes from an exact constant term. gp's own series of the same text can take the other
branch when its floating constant term falls below the cut: one of review 8's random towers has the
inner constant term `exp(2 log c) = c^2` with c < 0, which gp evaluates with the imaginary part
-1.4e-115, so its `sqrt` of the negative constant is `-0.372 I` where the engine's is `+0.372 I`; with
the factor `c^2` written exactly gp gives the engine's coefficients (`EXP/rev7rt4/checks/rh131.gp`).

Messages of `NotLaurent` (review 1, m9): an argument of negative valuation inside exp, sin, cos,
sinh, cosh, arctan, arcsin, ... gives "<f> of an argument with a pole at the point has no Laurent
expansion there" (this covers the essential singularity of exp(1/x) and the jump of arctan(1/x));
log gives "log has a logarithmic singularity at the point"; abs and sign give "<f> of an argument
with a zero of odd order is not analytic there".

Not expanded (`Err(Unsupported)`): `Gamma`, `Factorial`, `Zeta`, `Function(..)` and `Mod` applied to
an expression in the variable. (Gamma's series needs polygamma constants; a later unit can add it.)
When the argument is free of the variable they are kernels.

### 2.5 Expanding an `Expr`

`expand(e, cx)` charges `KONST_NODE = 16` per node visit before anything else (revision 5, review 5
m1), then:

- `e` free of the variable: `SeriesConst::from_expr(e)` as an exact constant series.
- the variable: `cx.xser` (an O(1) copy, charged 8 units).
- Chains (revision 5). A chain of `+` and `-` along the left spine of the input
  (`((a + b) - c) + d`, as the parser builds it) is read by a loop, not by recursion. Its summands
  free of the variable go to `from_expr` and one `sum_many`; the others are expanded and summed with
  that constant by one k-way merge (`sum_series`: a binary heap of the k series' cursors yields the
  exponents in increasing order, and each exponent's coefficients are summed by one `sum_many`;
  charged 24 per series and `SER_TERM + SEARCH_CMP (bits(k) + 1)` per term; revision 5 grouped the
  terms in a `BTreeMap` of vectors, about 176 bytes per exponent for 4 units per term). A chain of
  `*` and `/` is read the same way (revision 7rt1, red team 1 H3: revision 7 read `/` by recursion, so
  `x/2/.../2` and `x + (1/2/.../2)`, which §2.12's parser reads by its loop, aborted a 2 MB debug thread
  at 1302 and 400 slashes): the constant factors by `product_many`, a constant divisor r entering as
  `inverse_src(false, r)`, computed as it is read; the series factors multiplied in order, a series
  divisor through `inv` (strict) and `mul` (`chain_step`); then one scaling. A constant factor 0 gives
  the exact 0 after the series factors are expanded, so their errors still surface (`log(x)*0` at 0 is
  `NotLaurent`). Revision 4 added the summands one at a time and copied the growing constant at each
  step, O(N^2): `x + y1 + ... + y3000` (9 KB) was refused at 1e8; now 5000 symbols take 7 ms and
  1.8e6 units. A chain is one record of the work stack (below), and its items are read as that
  record's children: `x + x + ... + x` with 65 536 terms is `Ok` on an 8 MB release thread and on a 2 MB
  debug thread (HEAD overflows at 12 800 and 190).
- `Neg`: as above.
- `Div(l, r)`: an item of the `* /` chain above. r free of the variable: the factor
  `from_expr(r).inverse_src(false, r)` (the input divides by that constant, and its kernel takes the
  node r as its expression, §2.2); else the product so far times `inv(expand(r))` (strict).
- `Pow(l, r)`: a tower `b_1^(b_2^(...^t))`, with any number of `Neg` nodes between the levels
  (`x^-x^-x`, which §2.12 reads), is read along its right spine by a loop while the exponent depends on
  the variable (the tower record `ETower`; revision 7rt1, red team 1 H1: revision 7 recursed once per
  `^`, and `x+1^1^...^1` with 400 carets, 804 bytes, aborted a 2 MB debug thread where HEAD answers in
  0.4 ms). The top level `b_k^t` is expanded by the rules below; then each level's exponent is the
  series of the level above it, negated as the input says, and the rules from "else" on apply. The
  order of operations is t, then `b_k`, ..., `b_1`, and the charges are one `KONST_NODE` per spine
  node and the record's `RECORD_UNITS`. The bases and t are read as the record's children (revision
  7rt2: no recursion). The rules of one level:
  - r free of the variable: alpha = `from_expr(r)`; an integer gives `pow_int`, any other
    `pow_const`.
  - else expand r first. If its series is exact with no term other than h^0 (for example
    `x - x + 2`), alpha is that constant and the previous case applies (review 1, m1: `x^(x-x+2)`
    at 0 is `x^2`).
  - else l free of the variable (review 1, m5), or l depending on the variable with a series that is
    an exact constant (revision 7rt3): lc = `from_expr(l)`, or that constant, and its strict zero
    status: `Zero`: the rule `0^r` below; `Unknown`: `Err(UndecidedZero(lc))`; `NonZero`:
    `exp(expand(r) * apply(Log, lc))`.
  - else (both depend on the variable): `exp(expand(r) * log(expand(l)))`, so x^x at 0 is
    `NotLaurent`, at 1 it expands.
  - The rule is chosen from the series, not from the syntax, on both sides (revision 7rt3, red team 3
    structure H1): an exponent that expands to an exact constant takes the constant-exponent rule
    (review 1, m1), and a base that expands to an exact constant (`x*0`, `x - x`,
    `(x+1)^2 - x^2 - 2*x - 1`) takes the constant-base rule. Revision 7rt2 took `log` of the exact zero
    there, `DivisionByZero("log(0)")`, a decided error where 0^r = 0: `exp(x) + ((x+1)^2 - x^2 - 2*x -
    1)^(x+1)` at 0, `cos(x) + (x*0)^(1+x^2)` at 0 and `(x-x)^x` at 1, which HEAD answers.
  - `0^r` (revision 7rt3, red team 3 structure H1), decided with the engine's own tests on the series
    of r (0^r = 0 where Re r > 0 and is undefined where Re r < 0, so it has a Laurent expansion at the
    point only when Re r > 0 near the point, or r = 0 identically). r exact and empty is an exact
    constant (0^0 = 1, above); r empty and truncated: `NeedPrecision` (§2.1). Else with `e0` the first
    stored exponent and c its coefficient: `e0 < 0` (a pole of r) or `e0 > 0` (r vanishes at the point
    and is not constant): `NotLaurent` when c is decided nonzero (Re r takes both signs in every
    punctured neighbourhood), else `UndecidedZero(c)`; `e0 = 0`: split c = A + B I (`split_imag`) and
    decide the sign of A by rule 5's enclosures (`real_sign`: a rational's sign, or a ball at 64, 256
    or 1024 bits that excludes 0; a too-wide precision goes on): A > 0 gives the exact 0, A < 0
    `DivisionByZero("0 to a power with negative real part")`, A exactly 0 with B decided nonzero
    `NotLaurent`, and anything undecided `UndecidedZero(A)` (or `UndecidedZero(B)`). Revision 7rt2 read
    `val`, a lower bound when the lead is undecided, and accepted only a positive rational constant
    term, so `x + 0^(2^x)` at 1 and `x + 0^(x + sqrt(2))` at 0 were a decided
    `NotLaurent("0^r with r not positive at the point")` where the value is x, and
    `x + 0^((sin(1)^2+cos(1)^2-1)/x + 2)` (an undecided pole term) reported an undecided valuation as a
    decided one; the first is now x, the second `UndecidedZero`. The case `e0 = 0` and a constant
    `0^alpha` (§2.2) are one function, `zero_pow_sign` (revision 8, review 8 m2), so an exponent that
    is a constant, or expands to an exact constant (`(x-x)^(x-x+pi)`), gets the answer of the series
    rule: `x + 0^pi` is x, `x + 0^(-pi)` `DivisionByZero`.
- `Unary`: the table of §2.4; `Sqrt` is `pow_const(b, 1/2)`.

"Free of the variable" is computed once per node, by a pre-pass over the input before the first
attempt, kept for all attempts: a map from each node's address (`ptr::from_ref(e).addr()`; the
input is borrowed for the whole expansion, so the addresses are stable) to its flag and its depth
(review 4, m3). The pass is a post-order walk with an explicit stack (revision 5), and it looks a
node up before it descends (review 5 m1), so it visits each distinct node once and does not
recurse. An input built in the REPL by `e = e + e` is a DAG whose tree doubles per step, and
revision 4's pass walked the tree uncharged before the first charge (0.62 s at k = 24, doubling per
k). The point goes through the same pass: its dependence on the variable is the test
for `InvalidPoint` (revision 4 called `contains_symbol`, a tree walk), and the larger of the two
depths is the input depth D of the kernel-depth cap (§2.15). Revision 3 called `contains_symbol` at
every node, which is quadratic on a left-deep chain: `x + 1 + ... + 1` with 10 000 terms took 0.20 s,
now 8 ms. The `Div` and `Pow` cases above read the same flags. A DAG input is expanded once per
occurrence of each shared node, and each occurrence charges `KONST_NODE` first (review 5 allowed this;
a memo of series per node would keep every node's series alive): `e = x; e = e + e` 40 times is
`BudgetExceeded` after 0.11 s release, 1.0 s debug, and the four other DAG shapes of §5.3 after at most
0.45 s and 3.3 s.

The engine expands every subexpression on its own. So a function whose subexpression has no Laurent
expansion is refused even when the function itself has one: `sqrt(x)^2` and `exp(log(x))` at 0 are
`NotLaurent` (documented on `NotLaurent`; Puiseux series are out of scope, §7).

Value at the point (revision 7rt3, red team 3 structure H2). The pre-pass also records, per distinct
node, the size of its tree and the occurrences of the variable in it (saturating), so the size of the
node with the variable replaced by the point is known in O(1). Each finished node of `expand` carries a
flag "regular", set in one of two ways.

1. By the rules: its series has a positive precision and a decided valuation >= 0 (the first stored
   exponent is at least 0), every series divisor of its `* /` chain and every base of a negative power
   has valuation 0 (implied by the power's own valuation >= 0), `sign` has an argument of valuation 0,
   a power is not the exact zero, and every child that depends on the variable is regular. A constant
   node is regular; the variable is regular at a finite point. So the flag is the AND of the node's own
   conditions and of its children's flags, and no rule may drop a child's flag (revision 8, review 8
   B1). One rule did: a tower level whose exponent depends on the variable but expands to an exact
   constant takes the constant-exponent rule for its arithmetic (review 1 m1: `x^(x-x+2)` at 0 is
   `x^2`), and revision 7rt3 set the level's flag from the base alone there. So
   `x + (1+sin(1))^(500*sign(x^2))` at 0 (sign is not continuous at a zero, so `sign(x^2)` is not
   regular by the rules, although its series is the exact 1) printed the constant term
   `(1 + sin(1)) ^ (500 * sign((0 ^ 2)))`, whose value is 1 and not (1 + sin 1)^500 = 3.82e132 (gp
   `(1+sin(1))^500`); `(1+sin(1))^(((1)/x)*0 + 500)` at 0 printed `1 / 0`, and
   `x + (1+sin(1))^(500*((x-1)/(x-1)))` at 1 `0 / 0`. The rule of the arithmetic (an exact constant
   exponent takes the constant-exponent rule) and the rule of the flag are separate: the tower's
   constant-exponent state carries the exponent's flag (true for a top exponent free of the variable),
   and the level ANDs it in. Every record that sets the flag was checked against the AND: the `+ -`
   record ANDs each series summand's flag (constant summands are regular); the `* /` record ANDs each
   series factor's flag, and a series divisor must have valuation 0 (a constant factor 0 keeps the
   factors' flags, so `((1)/x)*0` at 0 is not regular by the rules); the unary record
   ANDs its operand's flag, and `sign` needs an operand of valuation 0; a tower level ANDs its base's
   and its exponent's flags on all four paths (a constant exponent, an exact-constant exponent, a
   constant base, the general `exp(r log l)`, whose exact-constant-base case also needs both flags);
   the variable is regular at a finite point only; constants are regular.
2. As an exact constant (revision 8): a node whose series is an exact constant c (exact precision, no
   term other than h^0: `sign(x^2)` at 0, `((1)/x)*0`, `(x-1)/(x-1)` at 1, `(x - x)^3`, `0^(x+1)`)
   equals c near the point whatever its subtree holds, so at a finite point it is regular, with the
   rendering of c as its substitution. `seed_const` puts that rendering in the substitution memo, and a
   later substitution reads it and does not descend below the node. A seed is the engine's own value of
   the node, not a re-evaluation of its subtree, so it is right whenever the normal form is. A node is
   seeded only when c's rendering is not much larger than the node's substitution would be (a tree of
   at most twice the estimate plus `HINT_MIN`, the hint rule in reverse), once per node and expansion,
   charged `SUBST_NODE + expr_units(c)` (its memo entry and the rendering); above that it is not
   regular, and its parents keep their normal forms.
   Without seeds, the fix of B1 alone refused towers that HEAD answers: with the exponent
   `2 + ((1)/(x-1))*0` at each level, 20 levels at 1, order 0, passed the tree cap (the normal form
   doubles per level), where HEAD returns the input with the point substituted and its `(1/0)*0` folded
   in 55 us; with seeds the result is a tree of 277 nodes, and 30 levels of 815
   (`EXP/rev8/review8_r8.txt`).

Then every node of a regular node's subtree, down to the seeds, is regular by the rules, the subtree
is analytic at the point there, every division in it is by a value that is nonzero there, and the
node's substitution (the variable replaced by the point, each seeded node replaced by its constant) is
an expression whose value is the series' constant term b0: the engine's rules are true identities of
analytic functions, and the substitution evaluates the same functions at the same point (principal
branches, as `apply` and `pow_const` take them) on the values of the seeds. A removable singularity is
never substituted: `sin(x)/x` at 0 has a series divisor of valuation 1 and a series that is not
constant, so it is not regular (red team 1's objection to substitution, `sin(0)/0`), and
`(x-1)/(x-1)` at 1 is replaced by its value 1, never by `0/0`. When a regular node's b0 is not rational
and its normal form's tree exceeds the substitution's estimate by more than `HINT_MIN = 64` (a margin,
revision 7rt4; revisions 7rt3 and 8 asked for twice the estimate plus 64, a factor that the over-count
of computed kernels made fire, and with exact kernel counts the factor alone lost those hints:
`x^x^...^x` at 2, order 1, fell from 118 admitted carets to 110), b0 gets the substitution as its
rendering hint (§2.2), if the substitution's tree is smaller than the normal form's (a seed can make
it larger than its estimate; revision 8). The substitution is built by an
explicit-stack copy of the nodes that depend on the variable, memoized per node for the whole
expansion (a DAG costs its distinct nodes once; a node free of the variable is the input's own `Arc`,
the variable the point's one shared node, a seeded node its seed), charged `SUBST_NODE = 96` per new
node. For `l^r = exp(r log l)` with l and r regular, the constant term of `r log l` gets the hint
`r(a) * log(l(a))` by the same test (`hint_exp_arg`). The hint changes no normal form and no key. It
is a second rendering of the same value, used by `to_expr` and so by a kernel made from b0 and by a
result coefficient equal to b0, and it is that only because every node of the substituted subtree is
regular, with the seeds as leaves whose substitution is their own value (above). The numeric zero test
evaluates `to_expr` of a constant (§2.3 `enclose`), so it evaluates the hint: its soundness rests on
the hint having the value b0, not on the ball code refusing a wrong hint (revision 7rt3's wrong hints
were refused only because `sign(0)`, `1/0` and `0/0` have no enclosure). A hint can change an
`Unknown` of the numeric test in either direction, since the ball code reads the input's syntax and
not a normal form; revision 8 closes the known gap, an exponent written `(-2)` or `3 - 1` (§2.3,
review 8 m1: 6 of the 102 random towers of review 8 that revision 7rt2 expands were `UndecidedZero` in
revision 7rt3), and revision 7rt4 the next one, an exactly-zero operand written `1 - 1` under sqrt, a
fractional power or a power with abs(k) >= 4 (§2.3, red team 4 numeric m3: the hinted 6-level tower
`(1)/(x - 1 + G_6(x + (x-1)^(2*2)) - 3)` at 1 was `UndecidedZero` in revisions 7rt3 and 8, `Ok` in
7rt2). A kernel created in an attempt before its argument was regular keeps its rendering in
later attempts (the table is shared); that costs tree size only. Ordinary results keep their expanded
normal form: of the 107 inputs of the gp oracle (which contain the displaycheck corpus's inputs) one
sets a hint, `(sin(1)^2 + cos(1)^2 + x)^100` at 0, whose constant term now prints
`(((sin(1) ^ 2) + (cos(1) ^ 2)) + 0) ^ 100` in place of its 101-term expansion, and its value still
equals gp's (`EXP/rev8/tools/hintscan.py`, `hintscan_r8.txt`, the same as revision 7rt3's).

Recursion depth (revision 7rt2, red team 2 structure H1). `expand` does not recurse over the input. It
runs one loop over an explicit work stack: a node visit either gives a series (a constant through
`from_expr`, which has its own work stack and never calls `expand`; the variable) or pushes one record,
and the series of a finished node is handed to the record on top, which names its next child or
finishes. A unary node's record is its operator (16 bytes); a `+ -` or `* /` chain and a tower hold
their items, the constants and the series read so far, and the state of the tower's level in a boxed
record (at most 96 bytes, charged `RECORD_UNITS = 32` when it is pushed, §2.16 "Memory"). The stack
holds at most one record per node of the current path, so at most D records, and the native stack of a
call is O(1) whatever the input's shape. The order of operations (the items of a chain left to right,
a constant divisor inverted as it is read; a tower's top, then its bases from the inside out), the
charges and the checks are those of the recursive reading of revision 7rt1 (prototype
`EXP/rev7rt2/proto10/ser.rs`: `expand`, `expand_visit`, `EFrame::resume`, `ETower`). Revision 7rt1
read every rule of the grammar that repeats without a parenthesis (the sign runs, the `+ -` and `* /`
chains, the `^` rule) by a loop, but recursed once per nested operand, with a dispatcher frame and a
chain or tower frame for each grammar level a parenthesis used. A text that used every level in each
parenthesis (red team 2: `x + -(-(...(1)^2*3+4)...)^2*3+4`, a sum, a product, a negation, a tower and an
atom per level) cost 11 frames and 15.5 to 18.9 KB of debug stack per parenthesis against the 9.7 KB of
HEAD's parser, so the engine overflowed a 2 MB debug thread at 132 levels where the parser reads 216;
revisions 5 to 7 had tried to keep each frame small (a thin dispatcher, non-inlined work functions),
which a compiler and a new grammar level can always outgrow.

Measured on the revision 7rt2 prototype (`EXP/rev7rt2/enginedepth_rt2.txt`, a binary search with
`EXP/rev7rt2/tools/edepth.sh` that counts a budget refusal as handled, up to 60 000 levels on 2 MB
debug and 200 000 on 8 MB release; `EXP/rev7rt2/redteam1_rt2.txt` for the towers and chains with the
red team's searches up to 50 000; `EXP/rev7rt2/redteam2_rt2.txt` and `mixdepth_rt2.txt` for the mixed
shapes, 100 KB of text each; HEAD from `EXP/rev5/enginedepth_*`, the red teams' reports and
`bin/enginedepth_head`; inputs built with the `Expr` API, or parsed on a 1 GiB thread, so the parser is
not involved; a 2 MB debug thread and an 8 MB release thread; revision 7rt1's values in parentheses;
"at least" means the search's upper end):

| shape (order 1 at 0, order 2 for `1/(1 + ...)` and `((x*2 + 1)*2 + 1)`) | engine, debug 2 MB | HEAD `taylor`, debug 2 MB | engine, release 8 MB | HEAD, release 8 MB |
|---|---|---|---|---|
| `sin(sin(...(x)))` | at least 59 996 (1727) | 188 | at least 199 996 (47 603) | 9 088 |
| `x + sin(sin(...(1)))` | at least 59 996 (710) | 184 | at least 199 996 (32 729) | 6 400 |
| `-(-(...(x)))` | at least 59 996 (1731) | 188 | at least 199 996 (47 606) | 12 672 |
| `x + (x + (... + x))` (right-deep) | at least 59 996 (740) | 188 | at least 199 996 (26 184) | 12 672 |
| `1/(1 + 1/(1 + ... x))` | at least 59 996 (342) | 47 | at least 199 996 (13 090) | 196 |
| `((x*2 + 1)*2 + 1)...` | at least 59 996 (345) | 62 | at least 199 996 (13 090) | 312 |
| `x^(x^(...^x))` at 1 | at least 59 996 (the same) | 188 | at least 199 996 (the same) | 4 570 |
| `x + 1^1^...^1`, `x^2^...^2` at 1, `x + sin(1)^...`, `x + 2^-2^-...` | at least 50 000 each (the same) | 2582, 2582, 1400 and more, 1296 | at least 50 000 (the same) | 28 880 for the first |
| `x/2/.../2`, `x + (1/2/.../2)`, `1/x/.../x` at 1 | at least 50 000 each (the same) | 188, 187, 188 | at least 50 000 (the same) | 6 579 |
| `x*3/2*3/2...` (pairs) | at least 25 000 (the same) | 94 | at least 25 000 (the same) | 3 290 |
| the six mixed shapes of red team 2 (`mix5cs`, `mix5`, `mix4`, `mix5cdiv` with HEAD's parser, `mix6c`, `mix6` with the §2.12 probe parser; 100 KB of text each, parsed on a 1 GiB thread) | at least 100 KB of each: 11 111, 9 090, 12 500, 11 111, 10 000 and 8 333 levels (132, 173, 192, 132, 111, 168) | 45 to 60 | at least 100 KB of each (5133, 5454, 6159, and more) | |

(The release HEAD runs had 4 GB and 60 s; the rows `1/(1 + ...)` and `((x*2 + 1)*2 + 1)` of HEAD may
stop by memory or time, the others stop by a stack overflow.)

Parsed inputs stop in the parser, and only there. The parser of these figures is HEAD's (the prototype
links it) or, for the mixed shapes `mix6c` and `mix6` that need §2.12's signed exponents, the §2.12
probe parser of red team 2 (`EXP/rev7rt2/probes/rt2.rs`); §6 step 8 measures the new parser. On an 8 MB
release thread HEAD's parser reads 4096 nested calls and stops at 4160 (2048 for `((..)*2+1)`), 22 695
carets of every tower and 4124 levels of every mixed shape (the §2.12 probe parser 4438); on a 2 MB debug
thread 160 to 162 nested calls, 1438 carets and 216 levels of a mixed shape (the probe parser 266); gdb
shows each overflow in the parser (nom). Parse and call on one thread stop exactly there (one level
less where the driver's own frame is larger: 215 and 4123), through `taylor_with_budget`,
`series_expansion` and the limit path (`EXP/rev7rt2/mixdepth_rt2.txt`, `depthcmp_rt2_*.txt`). On the
REPL's 64 MiB debug thread `mix5cs` with 6000 levels (54 KB) is parsed and expanded in 0.1 s (revision
7rt1: a 39.6 KB line killed the thread). So on the thread that parsed an input, `taylor_with_budget`
cannot overflow, whatever grammar levels the text uses. An `Expr` built through the API or by REPL
assignments can be deeper than any text; the engine handles it within the budget and the kernel-depth
cap (§2.15). The consumers of a result that still recurse are listed in §7 item 5.

### 2.6 Driver: working precision search

```rust
pub(crate) fn expansion(f, var, point: &ExpansionPoint, max_exp: i64, work: &mut Work)
    -> Result<Laurent, SeriesError>
```

Target T = max_exp + 1 (checked; overflow is `Unsupported("order too large")`). P starts at T. Each
attempt expands with precision P, all attempts charging the same `Work`:

- result `prec >= T`: return `trunc_hard(T)`.
- result `prec < T`: a decided precision loss (an `inv` with a decided leading coefficient loses
  exactly 2 v; cancellations). The missing amount is known: `P += T - prec`. Not capped (review 1,
  m2); bounded by the attempt limit and the budget. When an attempt did not raise the result's
  precision over the previous loss-driven attempt, the extra precision at least doubles:
  `P += max(T - prec, P - T)` (revision 7rt3, red team 3 series m1). A kept leading term of
  `pow_int` item 5 has precision `n vb + 1` whatever P is, so after a division by a high power of h
  the loss stays the same until P passes `n vb`: `((x+x^2)^1000)/x^999` (which is `x (1 + x)^1000`)
  at orders 2 to 40 ran out of its 24 attempts (`PrecisionExhausted(27)` to `(977)`), where now the
  doubling reaches P = 1000 in a few attempts and the result equals gp's.
  P is checked (`checked_add`, else `PrecisionExhausted`); P = i64::MAX is a legal finite working
  precision, since `Exact` is out of band (§2.1) and every slot count it gives is charged with
  saturating units before any allocation (§2.7; revision 8 could set P to the sentinel's value, and
  with the wrapped units of a 2^60-slot buffer the next attempt panicked). Revision 7rt4: when an
  attempt clamped a precision at i64::MAX (`Prec::add`), lost precision and did not raise the result's
  precision over the previous loss-driven attempt, a larger P cannot repair the loss (the clamped
  precision does not grow with P), so the driver returns `Unsupported("a series precision overflows
  i64")`: `(x^(2^63-4)*exp(x))/x^(2^63-2)` at order 2 is refused after its second attempt in under
  0.1 ms, where without the rule the doubling ran into the budget (0.18 s, 104 MB). A clamp in an
  attempt whose precision still grows (a loss elsewhere in the input) lets the search go on.
- `NeedPrecision` (an operation met an empty truncated series whose valuation it needs: `inv`, `log`,
  `pow_const`, `pow_int` item 4, `abs`/`sign`, and since revision 7rt3 `exp`, `sin_cos`, `sinh_cosh`,
  the atan family with erf, `integrate` and `0^r`; §2.1): double the extra precision,
  `P = T + max(2 (P - T), 8)`, while the extra stays at most `max(4 |T|, 128)`; beyond that,
  `Err(PrecisionExhausted { precision: P })`.
- any other error: return it.

At most 24 attempts; then `Err(PrecisionExhausted { precision: P })`. This is a bounded search: it
never claims the denominator is zero.

Examples (prototype): `1/(sin(x)^2 + cos(x)^2 - 1)` at 0, order 3 (identically zero denominator,
every rational coefficient cancels) is `PrecisionExhausted(132)` after 0.5 s debug, 0.08 s release at c79a582 (0.15 s and 16 ms at HEAD
dc367a2, revision 3 prototype);
`1/(x - x)` (exact zero) is `DivisionByZero`. `1/(sin(x) - x)` at order 3 needs one loss-driven
retry (P = 4, then 10). `1/(sin(x)^50 - x^50)` at order 3 (review 1, m2) runs the NeedPrecision
doubling 4, 12, 20, 36, 68 (the denominator becomes visible at 68 > 52), then one loss-driven step
to P = 108, and returns valuation -52 with the gp coefficients (1.1 s debug, 0.12 s release at c79a582; 0.21 s and
35 ms at HEAD).
`1/(x^300*(1+x))` needs no search (the exact denominator is kept, §2.1). `1/sin(x)^200` at order 2
(review 2, m1): at P = 3, `pow_int` keeps the leading term `x^200 + O(x^201)`, the inverse has
precision -199, and two loss-driven steps (P = 205, then 403) give the result with valuation -200,
equal to gp (1.3 s release at c79a582, 0.23 s at HEAD; revision 1: `PrecisionExhausted(131)`). `1/(x+x^2)^200` and
`1/(exp(x)-1)^150` take the same three attempts (15 ms and 2.3 s release at c79a582, 0.56 s for the
second at HEAD; `S1TRACE=1 EXP/rev3/bin/proto4_rel measure f 0 2` prints the attempts). The loss-driven step undershoots once
here because the precision of the kept leading term (201) does not grow with P until P passes 200;
the next attempt corrects it.

The point must not contain the variable (`InvalidPoint`). The point is any constant: `10^30`,
`pi/2`, `I`, a symbol `a`, `exp(1)`.

### 2.7 Work budget and cost model

```rust
/// Per-expansion state that every constant operation needs: the work counter, the kernel
/// interning table (§2.2; it holds the only copy of each key), the caps of §2.15 (the defaults;
/// unit tests set small ones) and the allowed kernel depth of this input. One `Work` serves all
/// attempts of the driver. Its `Drop` drops the kernels newest first (§2.2).
pub(crate) struct Work { used: u64, max: u64, kernels: HashMap<String, Kernel>, caps: Caps, kernel_depth: u32 }
impl Work {
    pub fn new(max: u64) -> Work;
    pub fn unbounded() -> Work;                       // max = u64::MAX; the counter still counts
    /// adds `units` plus the base 16 with `saturating_add` and compares the saturated sum (revision
    /// 7rt4, red team 4 series H2: `16 + units` wrapped a saturated u64::MAX to 15)
    pub fn charge(&mut self, units: u64) -> Result<(), SeriesError>;  // Err(BudgetExceeded) once used > max
    pub fn remaining(&self) -> u64;                   // max - used (the limit of the a-priori walks)
    /// the kernel with this key; `make` (given the new id) runs only for a new key, and charges
    /// the kernel's expression before it builds it
    pub fn intern(&mut self, key: String, make: impl FnOnce(&mut Work, u64) -> Result<KernelData, SeriesError>)
        -> Result<Kernel, SeriesError>;
}
```

Every constant operation and every series step calls `w.charge(units)?` **before** it does the
work, with units an a-priori size in 64-bit words. Sizes: `w(n) = bits(n)/64 + 1` for an integer,
and `words(c) = sum over terms of (bits(num) + bits(den))/64 + 1 + monomial length` (at least 1)
for a constant. Two more sizes count copies (review 4, B1), each at about 3 to 4 bytes of heap per
unit. All three are cached in the constant's `ConstSizes` (revision 5), so a charge reads them in
O(1):

- `copy_units(c) = sum over terms of (COPY_TERM + COPY_FACTOR L + COPY_WORD (w(num) + w(den)))`,
  with `COPY_TERM = 48`, `COPY_FACTOR = 6`, `COPY_WORD = 3` and L the monomial length (revision 6;
  revision 5: 32, 4, 2): one term of a new constant, its 88-byte slot, the monomial vector (16 bytes
  per factor) and the two digit vectors (8 bytes per word), with glibc's chunk overhead (a request
  plus 8, rounded up to 16, at least 32 bytes), so at most 3 bytes per unit (§2.16, memory). `words`
  alone undercounts small terms: a term with a 1-word coefficient and one kernel is 2 words but
  184 bytes in 3 allocations and its slot.
- `expr_units(c) = 16 + sum over terms of (EXPR_NODE (2 + L + 2 pf) + EXPR_WORD (w(num) + w(den)) +
  EXPR_INT (2 + pf))`, with `EXPR_NODE = 32`, `EXPR_WORD = 3`, `EXPR_INT = 11` and pf the factors
  whose exponent is not +-1: what `to_expr(c)` allocates (an `Arc<Expr>` is an 80-byte request, a
  96-byte chunk; at most `2 + L + 2 pf` nodes per term, §2.2), the integer words it copies and the
  digit vectors of the copied integers. The kernels' own expressions are shared, not copied.
  (Revision 5: `EXPR_NODE = 20` for `3 + 2 L` nodes, 4.8 bytes per unit per node, and nothing for
  the `Pow` and exponent nodes of a factor with exponent other than +-1.)

Revision 5 (review 5, B2) adds the units of monomial work and of map searches:

- `MONO_FACTOR_UNITS = 5` per factor of a monomial that an operation builds (16 bytes of heap and the
  merge step).
- `search_units(excess, t) = SEARCH_CMP (bits(t) + 1) excess`, with `SEARCH_CMP = 3`, for insertions
  into a map that holds at most t keys, where `excess` is the sum, over the inserted monomials, of
  their factors beyond `SEARCH_FREE = 16`. A comparison of two monomials walks their common prefix,
  and a B-tree search makes about 2.3 (bits(t) + 1) comparisons (review 5 measured 25 to 36 for 2^15
  keys). The first 16 factors of each operand are covered by the per-pair base (`OP_BASE` of `q_mul`
  and `q_add`, and `MONO_FACTOR_UNITS`): with 32 factors in common a search of a 2^20-key map costs
  about 4 us, and the pair is charged about 300 units. Revision 4 charged one unit per factor and no
  search, so `x + Q_1400*(S_15)*0` ran 13.3 s with 1.09 GB (7.3e6 units/s, 11 bytes per unit).
- `KERNEL_UNITS = 128` per new kernel, before it is created, and `KONST_NODE = 16` per node visit of
  `expand` and `from_expr` (plus the words of a literal, which it copies).
- `RECORD_UNITS = 32` per boxed record of the work stack of `expand` and `from_expr` (a chain, a tower,
  a call or a `%` node), when it is pushed (revision 7rt2: the records replace the native stack of the
  recursion, which the budget did not see; with the node visit, 48 units for at most 128 bytes, and
  since revision 7rt3, whose `expand` records name their node and carry a regular flag, at most 144).
- `SUBST_NODE = 96` per new node of a substitution (revision 7rt3, §2.5: the node and its memo entry),
  before it is built, and `HINT_UNITS = 32` per entry of the hint table.

Revision 6 (review 6, B1 to B3) charges every heap object the engine allocates at most 3 bytes per
unit, allocator overhead included; the table "Memory" of §2.16 lists each object with its bytes and
its units. The new constants:

- `CONST_BASE = 72` per constant built, charged by `from_sorted` before it allocates the header (an
  `Arc` of the header and the cached sizes, a 208-byte chunk). A one-term constant made from a
  rational (`rat`) or a kernel (`kernel`) also charges its digit vectors (`COPY_WORD w + 16`) or its
  monomial and coefficient (32): 94 and 104 units for 272 and 304 bytes. Revision 5 charged a negation
  `copy_units` only (40 units for a one-term constant of about 300 bytes) and a 2-term merge 80 units
  for a 1.3 KB constant (a B-tree leaf), which review 6 measured at 5.8 and 13.5 bytes per unit.
- `ACC_ENTRY = 32` per pair of a constant product, for its slot in the transient accumulator.
- `SER_TERM = 6` per term a series operation copies or produces (16 bytes; revision 5: 4).
- `RENDER_TERM = 4 EXPR_NODE + 2 EXPR_INT + SER_TERM = 156` per stored term of the result, beyond
  its coefficient's `expr_units`: the power of h, the exponent with its digit vector, the product or
  quotient and the term's share of the balanced sum; and `2 EXPR_NODE` once for h (§2.8).
- `BALL_NODE = 64` per distinct node of a numeric enclosure (§2.3; the revision 6 value, which this
  list gave as 32).
- `erf_units(p, a) = ERF_FACTOR u (ERF_BASE + ceil(min(a, X_p)^2))` per erf node, with `ERF_FACTOR = 2`,
  `ERF_BASE = 8`, `a` the bound of a literal argument or `X_p`, and `2u` for a literal argument at or
  above `X_p + 1` (revision 7, review 7 B1; §2.3).

**Charged big-number arithmetic (review 3, B1, B2, m1, m2).** Every `Rational` or `Integer`
operation on coefficients goes through one of the functions below (`series_const.rs`, section
"charged arithmetic"; prototype `EXP/rev6/proto7/konst.rs`, `q_mul` to `i_exact_root`, as in revision 3). Each charges
the bound in the table right before it calls `rustmath-rationals` or `rustmath-integers`, and on
`Err` it does nothing. No other code in `series_const.rs`, `series_engine.rs` or `series.rs` applies
`+ - * /`, `pow`, `gcd`, `lcm`, `Rational::new` or a root to a coefficient. `Rational::new` and the
`Rational` operators always reduce by a gcd of the unreduced numerator and denominator
(`rustmath-rationals/src/rational.rs:19-35`, `:279-367`), and that gcd is quadratic, so a charge
linear in the input sizes is a budget hole: revision 2 charged `q * 7` by `wa + wb` while the gcd of
`7 num` and `den` (about 1000 words each) ran 0.3 s at c79a582 (review 3, B1).

| function | computes | units charged before it runs |
|---|---|---|
| `q_mul(a, b)` | `a * b` | `OP_BASE + na nb + da db + N D`, with `N = na + nb`, `D = da + db` (`N` alone when both denominators are 1); at least `COPY_WORD (N + D)` (revision 7) |
| `q_add(a, b)`, `q_sub` | `a + b`, `a - b` | `OP_BASE + na db + nb da + da db + N D`, with `N = max(na + db, nb + da) + 1`, `D = da + db` (`N` alone when both denominators are 1); at least `COPY_WORD (N + D)` (revision 7) |
| `q_pow(q, n)` | `q^n` | `OP_BASE + P(Wn) + P(Wd) + Wn Wd` (`Wn` alone for an integer), with `Wn`, `Wd` the words of `num^n`, `den^n` and `P(W) = W^2/64 + W` |
| `i_pow(b, e)`, `i_mul(a, b)` | `b^e`, `a b` | `OP_BASE + P(W)`; `OP_BASE + w(a) w(b)`, at least `COPY_WORD (w(a) + w(b))` (revision 7) |
| `i_gcd(a, b)`, `i_lcm(a, b)` | gcd, lcm | `OP_BASE + w(a) w(b)`, `OP_BASE + 3 w(a) w(b)` |
| `q_new(n, d)` | `Rational::new(n, d)` | `OP_BASE + w(n) w(d)` (`w(n)` when d = 1) |
| `i_exact_root(n, d)` | `n.exact_root(d)` (I3) | `OP_BASE + w(n)^2` |

Here `na`, `da` are the words of the numerator and the denominator of a, and `OP_BASE = 64`. The
floor `COPY_WORD` per result word (revision 7, review 7 m4) makes every result of the layer at most 3
bytes per unit: the product of a one-word operand by an L-word one was charged `OP_BASE + 2L + 2`
for an `(L + 1)`-word result, 4 bytes per unit, and the sum of a one-word and an L-word integer about
the same. The floor is a `max`, so it changes only lopsided operations: the calibration of §5.3 moves
by at most 1.6 % (`(1+x)^10000` at 2000). The
words of a power come from a bound of its bits, `pow_bits(b, n) = ceil(n lg (1 + 10^-12)) + 2`,
where `lg = log2(top + 1) + s` for `b = top 2^s + rest` with a 53-bit `top`, and `lg = log2|b|` when
`|b| < 2^53` (then `top = |b|` exactly). The earlier bound `bits(b) n` doubled the charge of
`2^n`.

Why these constants (release, HEAD, `EXP/rev3/micro_gcd.txt`, `micro_norm_pow.txt`,
`micro_root.txt`, `micro_small.txt`; `EXP/rev3/micro.rs` and `micro_small.rs`):

| operation class | measured | rate of its charge |
|---|---|---|
| Lehmer gcd, balanced, 16 to 16 384 words | 12 us to 1.95 s | 2.6e7 (16 words) to 1.9e8 units/s |
| `Rational::new`, 257 to 8192 words per part | 0.8 ms to 0.42 s | 0.8e8 to 2.1e8 units/s |
| rational sum, 1, 2, 4, 8, 16 words per part | 1.1 us, 3.6 us, 7.3 us, 16.7 us, 38 us | 6.8e7, 2.7e7, 2.5e7, 3.2e7, 5.0e7 units/s |
| rational product, 2 and 4 words per part | 2.1 us, 2.9 us | 4.2e7, 5.5e7 units/s |
| integer power, result 50 to 16 370 words | 2 us to 15 ms | 5.9e7 to 2.7e8 units/s |
| `exact_root(n, 2)`, sieve passed, 8421 words | 50 to 80 ms | about 9e8 units/s |
| integer product, 257 to 16 384 words | 45 us to 22 ms | 1.5e9 to 1.2e10 units/s |

The per-call overhead dominates small operands (a 2-word rational sum costs 3.6 us, of which the
gcd 1.4 us), and `OP_BASE = 64` covers it: with 32, the 2-word sum charged 1.8e7 units/s. Every
class charges at least 2.5e7 units/s in isolation, and the slowest numeric node 1.6e7 (§2.3, revision
7rt4: cosh on a short mantissa), above the floor of 1e7 units/s that `TAYLOR_WORK` assumes (§5.3). Products are over-charged (schoolbook bound against Karatsuba), which
errs towards refusing.

On top of that layer, the structural charges:

| operation | units charged before it runs |
|---|---|
| constant construction (`from_sorted`, revision 6) | `CONST_BASE` before the header is allocated; 0 and 1 are shared and charge nothing |
| `SeriesConst::add`, `sub` | `copy_units(a) + copy_units(b)` for the merge (a new constant with every term of both; review 4, B1: revision 3 charged `words`), which also covers its comparisons (revision 6: a two-pointer merge, no search), plus `q_add` for each common monomial; adding to zero returns the other operand (an O(1) copy). Revision 2 charged `(wa + wb)^2` over all terms when a denominator was not 1, which refused small expansions at transcendental points in milliseconds (review 3, m1) |
| `SeriesConst::sum_many` (revision 5) | `copy_units` of every summand plus `SER_TERM` per summand plus `search_units(excess of all, k)` for the heap of the k cursors (revision 6; revision 5: a map search per term), plus `q_add` per collision |
| `SeriesConst::product_many` (revision 5) | each `mul` of its rounds; a zero factor returns 0 before any (revision 6) |
| `SeriesConst::neg` | `copy_units(a)` (revision 3: nothing) |
| `SeriesConst` copy (`Clone`) | nothing: an `Arc` increment (review 4, B1) |
| `SeriesConst::scale` | `copy_units(a)` (revision 5; it copies every monomial; revision 4 charged `words(a)`, one unit per factor), plus `q_mul` per term; by 1: nothing (the constant is shared) |
| `SeriesConst::mul` | `MONO_FACTOR_UNITS (ta Lb + tb La)` for the monomial work and memory of all pairs (`t` the number of terms, `L` the sum of the monomial lengths plus 1; revision 4: `ta Lb + tb La`), plus `search_units(tb excess(a) + ta excess(b), min(ta tb, 2^20))` for the accumulator (review 5, B2), plus `ACC_ENTRY` per pair for its accumulator slot (revision 6), plus `q_mul` (and `q_add` for the accumulation) per pair |
| `inverse` and `pow_u` of one monomial | `MONO_FACTOR_UNITS (L + 1)` for the new monomial (revision 5), plus the `q_*` they call |
| exact power of a rational or a single monomial | `q_pow` (the size cap of §2.2 applies first) |
| multi-term exact power | each `mul` of the binary powering, as above |
| radical extraction | trial division `(w(n) + 8) min(6542, 2^(bits(n)/d + 1))` when n has at most 4096 bits (review 4, m2: the base 8 per candidate prime), then `i_exact_root` |
| zero test, several terms | `2 copy_units(c) + 2 search_units(excess(c), t)` for the grouping by indeterminates, only when c has one (revision 6: two copies, the triples and the groups), and `copy_units(c) + search_units(excess(c), t)` for the A + B I split, only when c has the kernel I (review 4, B1; review 5, B2); the class tests read cached flags |
| numeric zero test, per precision p | `expr_units` of A and of B (their `to_expr`), then `ball_cost(A.to_expr(), p, remaining)` and the same for B (§2.3; review 2, B2; the walk counts each distinct node once, review 4, B2; an erf node by `erf_units`, revision 7) |
| `to_expr` of a constant, anywhere (the numeric test, an `UndecidedZero` constant, kernel creation, rendering) | `expr_units(c)` (review 4, B1) |
| kernel creation | the depth cap first; the key's size bound in bytes, 1 unit per byte, before the key is written (§2.2; review 2, m2; review 4, B2: revision 3 charged one unit per 8 bytes of a key that wrote the expression tree); for a new key, the kernel cap and `KERNEL_UNITS = 128` (revision 5), then 16 for an input node or `16 + expr_units` of the arguments before the expression is built; for an input node also `node_payload_units`, `COPY_WORD` per 8 bytes of what its shallow clone copies (a symbol's or a function's name, a function's argument pointers; revision 7: a 100 KB name was 16 units) |
| node visit of `expand` and `from_expr` | `KONST_NODE = 16`, plus the words of a literal, plus 16 per item of a chain (revision 5, review 5 m1), plus `RECORD_UNITS = 32` for a boxed record of the work stack (revision 7rt2) |
| rendering of the result | `render_units`: `2 EXPR_NODE` for h, then per stored term `expr_units` of its coefficient and `RENDER_TERM`, before any node is built (review 4, B1; revision 6, review 6 B3: revision 5 charged the coefficients only, and the `h^e`, product and sum nodes of 983 040 terms with coefficient 1 held 725 MB). `series_expansion` builds the coefficients' `Arc`s and h under this charge, and `to_expr()` the other nodes of one rendering (revision 7, §2.8) |
| series `add` | 16 plus `SER_TERM` per term of both operands (an O(1) copy each; review 4, B1), plus the constant adds |
| series `sum_series` (a chain, revision 5) | 24 per series plus `SER_TERM + SEARCH_CMP (bits(k) + 1)` per term (revision 6: a heap merge), plus one `sum_many` per exponent |
| series `neg`, `shift`, `split0`, `offsets`, the slices of `pow_int` | 16 plus `SER_TERM` per term copied; `neg` also `copy_units` of each coefficient |
| the variable's series, per occurrence | 16 + 8 (two O(1) copies) |
| series `mul` | 16 plus the sum over contributing pairs of their weights (`wa + wb` when either coefficient is one word, else `wa wb`; review 6 m4), in one charge before the second pass, plus the constant operations; the loop that sums it stops once the sum passes the remaining budget (revision 5); then 16 per slot of a dense buffer, or for the heap merge `SER_TERM + SEARCH_CMP (bits(rows) + 1)` per pair and 16 per row (revision 6, review 6 B1) |
| recurrence step k | `16 + 1 + sum_j words(u_j) * words(g_{k-j})` (Miller: times `words(alpha + 1)`), plus the constant operations of the step |
| dense buffer of n slots | `16 + 16 n`, saturating, before the cast and the allocation (`Ctx::alloc`); with a budget a buffer above `remaining / 16` slots is `BudgetExceeded`, and without one a buffer of more than `MAX_SERIES_SLOTS = isize::MAX / 64 = 2^57` slots is `Unsupported("a series buffer of more than 2^57 slots")`, so no allocation of a slot count from the input can panic (revision 7rt4) |
| driver | nothing extra: all attempts charge the same counter |

The series-level charges count the pairs a product or a step will combine; the constant operations
inside are charged again by their own sizes. The double count is deliberate: the series-level charge
is made before any pair is computed, so a large product is refused at once (`exp(x)*sin(x)` at order
2000 by one charge of 5.2e9 units), and the constant-level charges bound what each pair really costs.

The model is deterministic (tests can rely on it). Its normalization term is essential: the revision
2 model, which charged most normalizations linearly or not at all, let `x + (3^330000)/(5^200000)*2*3`
run 75.7 s at c79a582 for 5.9e7 units (review 3), and still 0.96 s at HEAD with Lehmer's gcd, where
every further small factor adds about 0.36 s for 15 000 units. With the model above, the same input
is refused at 1.2e8 units after 0.33 s. The size caps of §2.2 (exact constant powers, on every
path through `exact_pow_if_small`: the integer power, the inverse of `(1/P0)^e` and the rational
power, review 4 m1) and §2.1 (exact series kept, exact products in full) are structural and apply
with or without a budget; they keep `taylor` (unbounded) from expanding huge constants. `taylor` and friends use
`Work::unbounded()`; `taylor_with_budget` and `series_expansion(.., Some(budget))` return
`Err(BudgetExceeded { work, budget })` at the first charge that passes the budget (no partial
result).

Inherent output size is not bounded by the caps: `1/(x^(10^9)*(1+x))` up to x^3 has 10^9 + 4 terms,
and `x^(-10^9)*sin(x)` up to x^1 has 5*10^8. Unbounded calls compute them; the budget refuses them
before the dense buffer is allocated (the buffer is charged first).

Units arithmetic (revision 7rt4, red team 4 series H2). Every units formula is computed with
saturating `u64` arithmetic (`saturating_add`, `saturating_mul`), and `Work::charge` adds its base with
`saturating_add`, so a size from the input (a slot count, a pole order, an exponent, a precision, a bit
length times an exponent) can only saturate to u64::MAX, which every budget refuses. The plain `+` and
`*` left in the formulas combine sizes of objects already in memory (word counts, lengths, the
precision of an enclosure, at most 1024 bits) with constant factors, each below 2^40. Revision 8 computed
the slot units with `saturating_mul` and then `16 + w`, which wrapped u64::MAX to 15: a slot count of
2^60 or more from the text (the loss step sets P = T + a pole order, `inv` of an exact series has
`P + vb` slots, a power pole `P - n vb`, and the order of `series_expansion`) was admitted for 15
units, and `Vec::with_capacity` panicked with "capacity overflow" in release (debug: "attempt to add
with overflow" in `Work::charge`), on 15-byte inputs such as `exp(x)/x^(2^61)` that HEAD answers in
6 us. Now each such row is `BudgetExceeded` in at most 0.3 ms, on 8 MB release and 2 MB debug threads,
and through the limit path `Unknown (BudgetExceeded)`; the boundary between 2^59 (refused before, too)
and 2^60 slots is gone (§4.1 `dense_buffer_units`).

What the budget bounds (review 2, m5; review 3, B1; review 4, B1, m6; review 5, B2; review 6, B1 to
B3). Every big-number operation is charged by a bound of its cost before it runs; every constant,
term, monomial, `Expr` node, kernel, kernel key, series term, buffer slot, merge cursor and
accumulator slot is charged by its size before it is allocated, at most 3 bytes per unit (the table
"Memory" of §2.16); every map search and heap comparison beyond the free prefix is charged; and a copy
of a coefficient or of a kernel's expression is shared, not duplicated. So the budget bounds the work
and the memory of the call, its result included. Measured (release, revision 7, through the public
entry points of §2.9: `taylor_with_budget`, `series_expansion` with `to_expr`, and the
`limit_via_series` path; the heap peak of the call with the counting allocator of
`EXP/rev7/proto8/memcount.rs`, and VmHWM, of which about 4 MB is the process and up to 9 MB the parse
of a 100 KB input; `EXP/rev7/fam7_final.txt`, `mem7_final.txt`, `m2probe7_final.txt`,
`calib7_final.txt`, `attack7_final.txt`, `b2probe_r7.txt`; rerun on the revision 7rt1 prototype in
`EXP/rev7rt1/`): the heap peak is at most 2.29 bytes per unit on every family of §5.3 (`x +
S_19*sin(x)`: 227 MB at 9.92e7 units; building S: 120 MB at 5.4e7 units, 2.21; a product of 2^15 terms
by a 1400-factor monomial, then multiplied by 0: 13.9 MB at 6.35e6, 2.19), at most 2.0 on the families
of reviews 6 and 7, and VmHWM is at most 230 MB on every budgeted probe of those families. One family
holds more per unit (found while revising for red team 1, revision 7rt1): a 99 990-digit literal L times
a series of unit coefficients, `L*((1)/(1-x))`, and since `/` joined the `*` chain also `(L)/(1-x)` and
`(L)/((1-x)*x^k)`. Each stored coefficient is a new copy of L (the product `q_mul(L, 1)`, charged
`COPY_WORD` per result word since revision 7), and its rendering copies the digits again (`EXPR_WORD`
per word): 16 bytes for 6 units per word, 2.67 bytes per unit, the row "digit vector" of the table
"Memory". Measured 2.56 bytes per unit, and at the budget's edge `L*((1)/(1-x))` and `(L)/(1-x)` at
order 6000 hold 252 MB of heap at 9.84e7 units, VmHWM 256 MB (revision 7 measured the same for
`L*((1)/(1-x))`; it charged `(L)/(1-x)` through a product by an exact constant series, so it refused
that input from order 2700); `(L*x)/(1-x)^2` holds 2.60 bytes per unit. Revision 6 measured the engine's
internal rendering only; review 7 (B2) showed that the public `SeriesExpansion` of revision 6, which
held owned `Expr` coefficients, copied a bare kernel's top node (a 100 KB name) per term, twice:
983 MB at 4.5e6 units. With shared coefficients (§2.8) the same input holds 2 MB. A third family
holds slightly more (red team 2 memory, m1): a coefficient that is one monomial of N factors with an
exponent other than +-1 renders, per term and per factor, a `Mul`, a `Pow` and an `Integer` node,
charged at about 3 bytes per unit by the row "Expr node of a rendering", so the rendering dominates:
`(y0^2*y1^2*...*y7999^2)/(1-x)` (62 897 bytes) at order 107 holds 2.55 bytes per unit (253 MB of heap
at 9.94e7 units, VmHWM 261 MB, 0.20 s), and with N = 2000 at order 200 2.60 to 2.61 bytes per unit
(117 MB; 2.61 in the red team's run, 2.60 in revision 7rt2's);
the REPL line of the order-100 input reaches 265 MB of VmHWM (the text and the LaTeX, 12.8 MB each,
on top). The claim is the bound of the table "Memory", not a measured maximum: every kind of object at
most 3 bytes per unit, so `TAYLOR_WORK = 1e8` holds at most about 300 MB of engine data plus the
process and the parsed input, within the REPL's policy of about 400 MB per admitted call
(`repl.rs:10988`) and the 512 MB of the review bar. The largest measured are 2.61 bytes per unit and
265 MB of VmHWM (revision 7rt2; revisions 7 and 7rt1 stated their own largest, 2.29 and 2.60, as the
figure, and a family they had not measured passed each). Revision 7rt3 renders each distinct coefficient
once (§2.8) and caps the result's tree (§2.15): the long monomials above are `LimitExceeded(TreeSize)`
from N = 8000 at order 107 (a tree of about 3.5e6 nodes) and red team 3's `(-1)*(a*b*...)/(1-x)` from
about order 400 with 2700 names, `L*((1)/(1-x))` at order 6000 holds 2.5 MB (one rendering of L), and
the largest measured are 2.59 bytes per unit for `(L*x)/(1-x)^2` at the budget's refusal (259 MB of
heap, VmHWM 147 MB, order 3500) and 230 MB of VmHWM for an admitted call (`x + S_19*sin(x)`, 2.29
bytes per unit; `EXP/rev7rt3/lit_rt3.txt`, `caps_rt3.txt`). Revision 7rt4 (red team 4 memory, m1):
distinct coefficients that each render a long monomial reach 2.83 bytes per unit,
`(-1)*(qa*qb*...)/(1-x)^2` with N = 1000 short names at order 1043 (4.3 KB; 117.8 MB of heap at
4.16e7 units, VmHWM 122 MB, 0.11 s; the same through `series_expansion`, 1.18 bytes per unit through
the limit path), 2.81 with N = 4000 at order 258, 2.71 with N = 8000 at order 129, 2.64 for the squared
names with N = 1000 at order 521 (159.5 MB): each copy of the monomial is L `Mul` nodes of 96 bytes for
`EXPR_NODE = 32` units, the row "Expr node of a rendering" at 3.0, and the memo of §2.8 shares a
rendering only by the constant's address. The next order of each is `LimitExceeded(TreeSize)`, so the
tree cap binds before the budget and the family stays near 120 to 170 MB; on a 2 MB debug thread it is
`Ok` with at most 168 MB, and the REPL line of N = 1000 at order 1043 prints 8.7 MB of text and 10.8 MB
of LaTeX with VmHWM 138 MB. So the largest measured heap peak is 2.83 bytes per unit, an admitted call,
still below the 3.0 of the table "Memory" (`EXP/rev7rt4/rt4_ms.txt`). Revision 5 measured at most 2.9 bytes per unit with VmHWM on its
families but missed four, each a memory hole of review 6 or of this revision's self-attack: the
uncharged map of a sparse series product (18 bytes per unit, 2.53 GB), constants of 2 to 11 terms in
a 1 KB B-tree leaf and one-term constants copied for 40 units (13.8 and 5.9), the per-term nodes of
the rendering (7.5), and names copied into every monomial of a rendering (127 bytes per unit, 3.68 GB
from an 80 KB input); all four are fixed (§2.2, §2.4, §2.8). Revision 6 missed a fifth, the owned
coefficients of `SeriesExpansion` (review 7, B2: 220 bytes per unit through `to_expr()`), fixed in
revision 7 (§2.8, §2.9).

The budget does not bound the printed text: a kernel is stored once (its `Expr` is shared through
`Arc`) and printed once per coefficient that contains it, a kernel that the engine made from computed
values prints its shared subtrees once per occurrence (§2.2), and each 64-bit word of a coefficient
prints as about 19 decimal digits. Measured (release, revision 6): `(1+x)^(10^100)` at order 1150
uses 9.22e7 units, 0.12 s and 127 MB, and prints 64.7 MB (at order 1200 it is refused at 1.002e8;
revision 5 admitted order 1250); `(1+exp(...(1+exp(x))^2...))^2` with 10 levels at the point 1,
order 1, uses 2.4e6 units, 4 ms and 41 MB with its text, and prints 35 MB; `exp((1+exp(N_30))^511 +
x)` at order 1 uses 2.4e7 units and 4 MB, and its text has more than 2^30 nodes. (Revision 5 also
quoted `(x + sin(2^100000))^1000` at order 1000 printing 30.4 MB, a figure from revision 2: since
revision 5 its kernel's expression is the input's node `sin(2^100000)`, and it prints 265 KB.) The
printed size is the REPL's concern: J1b estimates a value's text before formatting it and shows a
value above `OUTPUT_BYTES` (16 MiB) as a summary (§3.4). That bounds the time to print only when the
writers are linear in the text they write. Display and the REPL's LaTeX writer write into one buffer
with a work list (revision 7rt1, §2.11 and §3.1; red team 1, memory H1: HEAD's `expr_to_latex` formats
each node from its children's finished strings, so a 99 KB input whose result shares a depth-33 000
kernel among 101 terms, 1.1e7 units and 62 ms in the engine, spent 32 s in the LaTeX of its 10 MB of
text; the work-list writer takes 0.13 s).

### 2.8 Output rendering

`Laurent -> Expr`: terms in increasing exponent, coefficient k:

- exponent 0: `c`; exponent e > 0: `h^e` (`h` when e = 1), `-h^e` when c = -1, `c * h^e` otherwise;
  exponent e < 0: `c / h^|e|`.
- h is `x` when the point is zero, `x - a` otherwise (a as the caller gave it), and `x` with exponent
  -e at infinity. Whether the point is zero is decided once, by the driver, when it builds the
  variable's series from `SeriesConst::from_expr(point)` (the exact zero, not a match on
  `Expr::Integer`), and `SeriesExpansion` keeps it (`point_is_zero`, review 5 m2: revision 4's
  rendering rebuilt the point's constant at every `to_expr()`, uncharged, 0.25 s for the point K).
- h is one shared node (`Arc<Expr>`), and every term refers to it (revision 6: revision 5 cloned x,
  a `Symbol` with its name's `String`, for each term). The driver charges `render_units` (§2.7)
  before the result is built: `2 EXPR_NODE` for h, then per stored term its coefficient's
  `expr_units` and `RENDER_TERM` (review 6, B3).
- Which code renders, and what it copies (revision 7, review 7 B2). `series_expansion`, under that
  charge, builds h once and turns each stored coefficient into the `Arc<Expr>` of
  `to_expr_arc_raw` (§2.2): a coefficient that is a bare kernel is the kernel's own `Arc` (one
  increment, no node), any other the nodes `expr_units` counted, which refer to the kernels' `Arc`s.
  It stores those `Arc`s in `SeriesExpansion::terms` and drops the `Work` (the kernel table); the
  coefficients keep the kernels' expressions alive. Revision 7rt3 (red team 3 memory, m1) renders each
  distinct coefficient once: a map from the constant's `Arc` address to its rendering, for the pass
  only, so a result that names one shared constant at every term (`(-1)*(a*b*...)/(1-x)`, whose
  coefficients are one constant) holds one rendering of it instead of one per term. The charge
  (`render_units`, per term) is unchanged, so it now over-counts such results.
- The result's tree is bounded before it is rendered (revision 7rt3, red team 3 structure H2):
  `render_tree(s)`, O(n) from the cached sizes, is per term the coefficient's effective tree (its
  hint's when it has one, §2.2), at most 4 nodes and h's tree (1, or 2 plus the point's), and a result
  above `MAX_RESULT_TREE = 2^21` nodes is `LimitExceeded(TreeSize)` (§2.15). The constant of an
  `UndecidedZero` is bounded the same way (a constant above the cap gives the same error), since the
  caller can walk it too. `SeriesExpansion::to_expr(&self)` then builds,
  per term, one `Arc` increment for the coefficient, one for h, and at most the 4 nodes of
  `RENDER_TERM` (the power of h, its exponent, the product or quotient, its share of the balanced
  sum). No public entry point copies a node per term: `taylor_with_budget` is
  `series_expansion(.., Some(budget))?.to_expr()`, and the `limit_via_series` path reads the first
  stored term only (§2.13). Revision 6 stored owned `Expr` coefficients, so each bare-kernel
  coefficient cloned the kernel's top node into the `SeriesExpansion` (a `Symbol` clones its name's
  `String`) and `to_expr()` cloned it again: `(Y)/((1-x)*x^3000)` at order 2000 with a 99 990-character
  Y held 983 MB at 4.5e6 units (review 7). The prototype of revision 7 has this layer
  (`EXP/rev7/proto8/api.rs`), and every measurement of §5.3 goes through it: the same input holds
  2 MB, and `(Y)/((1-x)*x^10000)` 5 MB at 1.06e7 units (`EXP/rev7/b2probe_r7.txt`).
- What remains copied, once per call and not per term (not charged; the table "Memory" of §2.16):
  the `SeriesExpansion` header (`var`, a copy of the variable's name, and `point`, a copy of the
  caller's point node), the point node again inside h when it is not zero, and, when the whole result
  is one shared coefficient (`taylor(Y, x, 0, 5)` is Y), the top node that `to_expr()` copies because
  it returns an owned `Expr`: a kernel's node (a name, charged once already as the kernel's payload,
  §2.7) or a rational coefficient's digits (charged once by `render_units`). Each is at most one node
  of the caller's arguments or of the charged rendering.
- the terms are combined by a **balanced** sum (`balanced_sum`: sums of at most 4 terms stay
  left-deep `((a + b) + c) + d`; longer ones split in halves). The sum has depth about log2 n plus
  the depth of a term (3 plus its coefficient's depth); the whole result had depth 18 at order 20000
  in the prototype. A coefficient's monomials are balanced products too (§2.2, review 5 B1), so a
  coefficient has depth at most its deepest kernel + 40, and the result at most the allowed kernel
  depth of §2.15 + 41 + `bal(n)`: about 1080 when the kernel depth is at most 1024. A result can be
  deeper than its input: a kernel the engine makes from computed values renders their normal forms, and
  `x^x^...^x` at 2 has the value kernels `exp(log(2) K)`, two levels per level of input, so depth
  2 D + 2 (red team 1, H2: 36 002 for 18 000 carets, where HEAD's result has 17 998). Since revision
  7rt3 a level's value at the point renders as the input with the point substituted when its normal
  form is much larger (§2.5), so the same tower has depth 20 800 (24 800 for 20 000 carets, 30 190 for
  22 695); 2 D + 2 stays the bound. Display needs no stack per level (§2.11), so such a result prints on
  any thread; §2.15 gives the limits of the consumers that still recurse.
- no `simplify` call on the output (it is already canonical).

Examples (prototype output):
`taylor(exp(x), x, 0, 5) = (((1 + x) + (1/2 * (x ^ 2))) + (((1/6 * (x ^ 3)) + (1/24 * (x ^ 4))) + (1/120 * (x ^ 5))))`,
`taylor(1/sin(x), x, 0, 5) = ((((1 / x) + (1/6 * x)) + (7/360 * (x ^ 3))) + (31/15120 * (x ^ 5)))`,
`taylor(exp(x), x, 1, 2) = ((exp(1) + (exp(1) * (x - 1))) + ((1/2 * exp(1)) * ((x - 1) ^ 2)))`,
`taylor(x^(-1000000000) + 1, x, 0, 1) = ((1 / (x ^ 1000000000)) + 1)`.
(These are the HEAD Display strings; §2.11 adds parentheses around rational exponents.)

### 2.9 Public API (`series.rs`)

```rust
/// Errors of the series engine (MAGMA ch. 49).
#[derive(Clone, PartialEq)]
pub enum SeriesError {
    /// A subexpression has no Laurent expansion at the point: a branch point (sqrt(x) at 0), a
    /// logarithmic singularity (log(x) at 0), an argument with a pole inside exp, sin, arctan, ...
    /// (exp(1/x) at 0), or abs/sign at a zero of odd order. The engine expands each subexpression
    /// on its own, so the function itself may still have one (sqrt(x)^2 and exp(log(x)) at 0).
    NotLaurent(String),
    /// gamma, factorial, zeta, a named function or mod applied to an expression in the variable,
    /// abs/sign of a series with non-rational coefficients, an exponent that overflows i64.
    Unsupported(String),
    /// The expansion must divide by this constant (or read its sign) and cannot decide whether it
    /// is 0. For a power whose exponent alpha must be known to be an integer or not, the constant
    /// is alpha - n with n the nearest integer, or alpha itself when no nearest integer was
    /// computed. The numeric test refuses values outside a safe exponent window, so a constant
    /// whose evaluation would underflow or overflow is reported here, never decided. The variant
    /// holds the whole constant; `Display` prints it only when it is short (below).
    UndecidedZero(Expr),
    /// The expression divides by an exact zero (1/(x - x), log(0), 0^-1, tan(pi/2) as a constant, and
    /// since revision 7rt3 0^r with Re r decided negative at the point).
    DivisionByZero(String),
    /// The point contains the variable, or infinity (oo) appears.
    InvalidPoint(String),
    /// Bounded search: a denominator vanished to this working precision. It may be identically 0.
    PrecisionExhausted { precision: i64 },
    BudgetExceeded { work: u64, budget: u64 },
    /// A structural cap of the engine (§2.15) was reached before the work it bounds: a monomial of
    /// more than 2^15 factors, more than 2^16 kernels, a constant of more than 2^20 terms, a
    /// kernel expression deeper than max(1024, 4 x the input's depth), or (revision 7rt3) a result,
    /// or the constant of an `UndecidedZero`, whose tree has more than 2^21 nodes. `value` is the cap.
    LimitExceeded { limit: StructureLimit, value: u64 },
    /// laurent(): the expansion has a nonzero term of exponent `valuation` below `min_order`.
    ValuationBelowMinimum { valuation: i64, min_order: i64 },
    /// series_coefficients(): the expansion has a pole of this order (valuation < 0).
    Pole { valuation: i64 },
}
/// Which cap of §2.15 `LimitExceeded` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureLimit { MonomialFactors, Kernels, ConstantTerms, KernelDepth, TreeSize }
/// One plain sentence each. The constant of `UndecidedZero` is printed only when
/// `display_estimate(c, 200)` is at most 200 characters; otherwise the sentence gives the
/// estimate: "cannot decide whether a constant is zero (its printed form has about 5700000
/// characters; the error value holds it)". `display_estimate` is a walk with an explicit stack
/// that adds 0.302 bits + 2 per integer (both parts of a rational), 24 per float, the name length
/// per symbol or function and 6 per operator node, and stops once the sum passes its limit (about
/// 34 nodes for 200). It does no decimal conversion, whose num-bigint cost is quadratic (review 3,
/// m6: a 407-character input gave a 5.7 MB message). `LimitExceeded` prints "a structural limit of
/// the series engine was reached: more than {value} {factors in one monomial | distinct kernels |
/// terms in one constant | levels in one kernel expression | nodes in the tree of the result}". The other variants hold fixed engine
/// texts, which never format a constant.
impl fmt::Display for SeriesError { ... }
/// Written by hand (review 4, B2): the variant name, then the Display text. A derived Debug would
/// print the whole `UndecidedZero` constant, whose tree can be exponentially larger than its DAG
/// (§2.2), so a failing `unwrap()` in a test could print gigabytes.
impl fmt::Debug for SeriesError { ... }
impl std::error::Error for SeriesError {}

#[derive(Debug, Clone, PartialEq)]
pub enum ExpansionPoint { At(Expr), Infinity }

/// A truncated Laurent expansion sum c_k h^k + O(h^precision), h = var - a (or 1/var at infinity).
/// `Debug` is written by hand, as for `SeriesError`: the variable, the point's estimate, the
/// precision and the exponents, never a coefficient's tree (a coefficient's tree can be
/// exponentially larger than its DAG, §2.2).
#[derive(Clone, PartialEq)]
pub struct SeriesExpansion {
    pub var: Symbol,
    pub point: ExpansionPoint,
    /// (exponent, coefficient) in increasing exponent; canonical constants, none structurally 0;
    /// every exponent below `precision`. Sparse: x^(-10^9) + 1 has two entries.
    /// Revision 7 (review 7, B2): each coefficient is the shared node its rendering built
    /// (`to_expr_arc_raw`; a bare kernel is the kernel's own `Arc`), so no node is copied per term.
    pub terms: Vec<(i64, Arc<Expr>)>,
    /// max_exponent + 1, always: an exact internal result (`Prec::Exact`) has this public precision
    /// too (revision 7rt4; the prototype of revision 8 returned i64::MAX for an exact result, a value
    /// a finite precision can also take)
    pub precision: i64,
    /// Revision 5 (review 5, m2): whether the point is the exact zero, recorded by the driver, so
    /// `to_expr` renders `x` or `x - a` without rebuilding the point. Not public: a
    /// `SeriesExpansion` is made only by `series_expansion`.
    point_is_zero: bool,
    /// Revision 7: h (`x`, or `x - a`), built once by `series_expansion` and charged there.
    h: Arc<Expr>,
}
impl SeriesExpansion {
    /// The exponent of the first stored term (None when no term is stored). Every term below it
    /// is exactly zero, so it is a lower bound of the valuation. It equals the valuation when that
    /// coefficient is decided nonzero (for example a rational); an undecided coefficient may be 0:
    /// `(sin(1)^2 + cos(1)^2 - 1)/x + 1` at 0 has first exponent -1 and valuation 0 (review 3, m3).
    pub fn first_exponent(&self) -> Option<i64>;
    /// None for k >= precision; Some(0) when absent; otherwise the shared coefficient (an `Arc`
    /// increment; revision 7, review 7 B2: revision 6 returned an owned clone)
    pub fn coefficient(&self, k: i64) -> Option<Arc<Expr>>;
    /// §2.8: per term an `Arc` increment for the coefficient and at most 4 new nodes, all charged by
    /// the expansion (render_units); the top node is copied once only when the result is a single
    /// shared coefficient
    pub fn to_expr(&self) -> Expr;
    pub fn big_o(&self) -> BigO;                        // O(h^precision)
}

impl Expr {
    pub fn series_expansion(&self, var: &Symbol, point: &ExpansionPoint, max_exponent: i64,
                            budget: Option<u64>) -> Result<SeriesExpansion, SeriesError>;
    /// All terms of exponent <= order; poles give negative exponents (as Sage/Maxima taylor).
    /// A float literal denotes its exact binary value: `0.1 - 1/10` is a nonzero constant
    /// (revision 7rt2, §2.2).
    pub fn taylor(&self, var: &Symbol, point: &Expr, order: usize) -> Result<Expr, SeriesError>;
    pub fn taylor_with_budget(&self, var: &Symbol, point: &Expr, order: usize, budget: u64)
        -> Result<Expr, SeriesError>;
    pub fn maclaurin(&self, var: &Symbol, order: usize) -> Result<Expr, SeriesError>;
    pub fn series(&self, var: &Symbol, point: &Expr, order: usize) -> Result<Expr, SeriesError>; // = taylor
    /// Terms of exponent in [min_order, max_order]; Err(ValuationBelowMinimum) if a term below
    /// min_order is decided nonzero (instead of dropping the principal part), Err(UndecidedZero)
    /// if the terms below min_order cannot be decided.
    pub fn laurent(&self, var: &Symbol, point: &Expr, min_order: i64, max_order: i64)
        -> Result<Expr, SeriesError>;
    /// Expansion at infinity in powers of 1/var, terms (1/var)^k for k <= order.
    pub fn asymptotic(&self, var: &Symbol, order: usize) -> Result<Expr, SeriesError>;
    /// [c_0, ..., c_order] (dense; absent terms are 0); Err(Pole) when a term of negative exponent
    /// is decided nonzero, Err(UndecidedZero) when it cannot be decided. Like `taylor` it has no
    /// budget; each entry is an owned copy of its coefficient's top node (subtrees shared).
    pub fn series_coefficients(&self, var: &Symbol, point: &Expr, order: usize)
        -> Result<Vec<Expr>, SeriesError>;
    pub fn series_with_big_o(&self, var: &Symbol, point: &Expr, order: usize)
        -> Result<(Expr, BigO), SeriesError>;
}
```

`order` above `i64::MAX - 1` is `Err(Unsupported("order too large"))`. `lib.rs` exports
`SeriesError`, `SeriesExpansion`, `ExpansionPoint`. Every public doc comment states the generic
convention for free symbols, and abs/sign state the real-variable assumption.

Header of each new file: MAGMA ch. 49 (digest path, §§ above), Knuth TAOCP 2 §4.7, Brent and Kung,
"Fast algorithms for manipulating formal power series", J. ACM 25(4) 1978, 581-595; series_engine.rs
and series_const.rs also: S. C. Johnson, "Sparse polynomial arithmetic", ACM SIGSAM Bulletin 8(3),
1974, 63-71, and Monagan and Pearce, "Sparse polynomial division using a heap", J. Symbolic Comput.
46(7), 2011, 807-822 (the heap merges of revision 6); series_ball.rs:
Fousse et al. 2007 and Johansson 2017; series_const.rs: Richardson 1968 (zero equivalence is
undecidable in general, which is why the test is three-valued), Lindemann 1882, Hermite 1873.

### 2.10 `simplify` along the left spine (`simplify.rs:376-400`)

```rust
pub fn simplify(expr: &Expr) -> Expr {
    // Binary chains are usually left-deep (a + b + c + ..., built by the parser and by
    // accumulating loops): walk the left spine with an explicit stack.
    let mut spine: Vec<(BinaryOp, &Arc<Expr>)> = Vec::new();
    let mut cur = expr;
    while let Expr::Binary(op, l, r) = cur {
        spine.push((*op, r));
        cur = l;
    }
    let mut acc = simplify_non_binary(cur);          // the old Integer/../Unary/Function arms
    while let Some((op, r)) = spine.pop() {
        acc = simplify_binary(op, acc, simplify(r));
    }
    acc
}
```

The evaluation order (left before right, inner before outer) and every result are unchanged; only
the recursion over left children goes. `simplify_binary` (`simplify.rs:470`) inspects its operands
only one level deep (`as_rational_const` matches `Integer`/`Rational`, `simplify.rs:450`), so a deep
accumulator costs nothing extra. Right-deep chains (power towers) still recurse.

### 2.11 Display: parenthesize rationals and negative bases under `^` and `/`, and a negated negation (`expression.rs:444-454`, `:481`)

The parser change (§2.12) makes `a/b` an operator, so today's Display strings `(x ^ 1/2)` and
`(x / 1/2)` would re-parse as `(x^1)/2` and `(x/1)/2`. The engine outputs such kernels (`2^(1/2)`).
Display prints:

- an `Expr::Rational` with denominator > 1 in parentheses when it is an operand of `Pow` or the
  right operand of `Div`: `(x ^ (1/2))`, `((3/4) ^ (-1/2))`, `(x / (1/2))`;
- the base of a `Pow` in parentheses when it is `Unary(Neg, _)` or a negative `Integer`,
  `Rational` or `Real` (review 1, m3): `((-y) ^ (1/2))`, `((-2) ^ 2)`, `((-1.5) ^ 2)`;
- the operand of a `Unary(Neg, _)` in parentheses when it is itself a `Unary(Neg, _)` or a negative
  `Integer`, `Rational` or `Real` (revision 7, the HANDOFF note of §1.8): `-(-x)`, `-(-2)`,
  `-(-1/2)`, where HEAD prints `--x`, `--2`, `--1/2`. Both forms parse back to the same value under
  §2.12 (sign runs); the parenthesized one also reads as the tree it prints. Also when it is a
  `Rational` with denominator above 1, whatever its sign (revision 7rt2, red team 2 structure m1):
  `-(2/3)`, where revision 7rt1 printed `-2/3`, which §2.12 reads as `(-2)/3` under `^` or as a
  divisor. A kernel made from an input node keeps the input's subtree, so results printed such nodes:
  `x*pi^-(2/3)` at 1 printed `(pi ^ -2/3)`, read back as `pi^(-2)/3` = 0.0338 instead of
  `pi^(-2/3)` = 0.4662, and `x + atan(3/-(1/2))` printed `(3 / -1/2)`, read back as `-3/2`. By operator
  and operand this is the only pair the other rules miss: every binary node is in parentheses, and
  under `+`, `-` and `*` both readings of `-p/q` have the same value (`(a * 1/3)` reads back as
  `((a * 1) / 3)`, another tree with the same value).

A negative exponent needs no parentheses (`x ^ -2` parses through the signed exponent rule). A
workspace grep finds no test string that depends on the old forms (a grep for `^ p/q`, `p/q ^`,
`/ p/q` and `(-n ^` inside string literals of `*.rs` files is empty). The last rule also fixes the
pre-existing round-trip errors `Pow(-2, 2)` displayed `(-2 ^ 2)` (parses as -4) and
`sqrt(x - y)` at 0, whose kernel `(-y)^(1/2)` displayed `(-y ^ (1/2))` (parses as `-(y^(1/2))`).

Display writes with a work list (revision 7rt1, red team 1 H2). HEAD's `fmt` recurses once per level, so
a result deeper than its input aborted the thread that had parsed the input: `x^x^...^x` with 18 000
carets at 2, order 0, has depth 36 002 (§2.8), and its Display overflowed an 8 MB release thread where
HEAD's own result, of depth 17 998, prints. The writer below has the same output (HEAD's text plus the
parentheses of the rules above) and keeps its pending items on the heap:

```rust
impl fmt::Display for Expr {
    /// Writes with an explicit work list: the stack use does not depend on the depth of the
    /// expression (a result can be twice as deep as its input, §2.8, and a sum built by a loop is as
    /// deep as it is long). The list holds at most the depth plus the pending arguments of calls.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Item<'a> { Node(&'a Expr, bool), Str(&'static str) }   // the bool: in parentheses (rules above)
        let mut stack = vec![Item::Node(self, false)];
        while let Some(it) = stack.pop() {
            match it {
                Item::Str(t) => f.write_str(t)?,
                Item::Node(e, paren) => {
                    if paren { f.write_str("(")?; stack.push(Item::Str(")")); }
                    match e {
                        Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => { /* as today */ }
                        Expr::Binary(op, l, r) => {        // "(l op r)": pushed in reverse order
                            f.write_str("(")?;
                            stack.extend([Item::Str(")"), Item::Node(r, rule_right(*op, r)),
                                          Item::Str(op_text(*op)), Item::Node(l, rule_left(*op, l))]);
                        }
                        Expr::Unary(UnaryOp::Neg, i) => { f.write_str("-")?; stack.push(Item::Node(i, rule_neg(i))); }
                        Expr::Unary(op, i) => { f.write_str(op_name(*op))?; f.write_str("(")?;
                                                stack.extend([Item::Str(")"), Item::Node(i, false)]); }
                        Expr::Function(name, args) => {
                            f.write_str(name)?; f.write_str("(")?; stack.push(Item::Str(")"));
                            for (k, a) in args.iter().enumerate().rev() {
                                stack.push(Item::Node(a, false));
                                if k > 0 { stack.push(Item::Str(", ")); }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
```

`rule_left`, `rule_right` and `rule_neg` are the three rules of this section (a fraction under `^` or as
a divisor, a negative base of `^`, a negated negative operand or fraction of `-`); `Debug` calls Display,
as at HEAD. A fourth rule is on the leaf `Real(f)` (revision 7rt3, red team 3 structure m2): a finite
integral f is written with Rust's `{}` (the shortest text that reads back as f) followed by `.0`, so
§2.12 reads it back as the same f64 and not as an `Integer` of another value: `{}` writes the f64
`99999999999999991611392` of the literal `100000000000000000000000.0` as `100000000000000000000000`,
which §2.12's bignum literals read as 10^23, and `(1)/(x + 100000000000000000000000.0 -
100000000000000000000000)` (c0 = `1/(-8388608)`, decided) printed a text that reads back as a division
by zero. A non-integral float already has its `.` (Rust's `{}` never uses an exponent). The value round trip (write with the rules, read back with §2.12's parser, the same value)
is checked on random trees of the §2.12 grammar (`EXP/rev7rt2/tools/roundtrip_rt2.py`, red team 2's
generator, gp at x = 37/100 and y = 61/100: 551 inputs, 954 trees, 944 keep their value, 0 differ, 10 cannot be evaluated by gp; revision 7rt1's Display: 78 differ).
Prototype: `EXP/rev7rt4/proto13/display.rs` (`display_iter`; with the rules off it is HEAD's Display, and
`displaycheck.rs` finds it byte-identical to `format!("{}")` on 368 expressions: the 107 inputs of the gp
oracle, their points and their results, and every node kind built with the API; with the rules on it
gives the forms listed above). The deepest results measured print in a few milliseconds: depth 36 002
(288 KB) in 3 ms and depth 40 002 in 3 ms on an 8 MB release thread, a 10 MB result of depth 33 000 in
0.14 s (HEAD's recursive Display: 0.17 s).

### 2.12 The literal parser (S7)

```rust
// Unsigned integer literal of any size.
fn integer_literal(input: &str) -> IResult<&str, Integer> {
    let (rest, digits) = digit1(input)?;
    Ok((rest, Integer::from_decimal_str(digits).expect("digit1 yields decimal digits")))
}
fn number(input: &str) -> IResult<&str, Expr>   // float (no sign) | integer_literal; no a/b token
fn unary(input)          // ('-' | '+')* power, the run of signs read by a loop (revision 5): an odd
                         // number of '-' gives one Neg, an even number none ("--5" = 5, "---x" = Neg(x))
fn power(input)          // atom ('^' exponent)?, exponent := ('-' | '+')* power, the signs by a loop as above
                         // Neg(Integer(n)) as an exponent folds to Integer(-n): "x^-2" = Pow(x, -2) as today
fn multiplicative(input) // unary (('*' | '/') unary)*, with the fold below
```

Fold in `multiplicative`: after reading `l / r`, if both l and r are signed integer literals
(`Integer(n)` or `Neg(Integer(n))` as the parser built them) and r != 0, the result is the exact
rational (an `Integer` when it divides). r = 0 is a parse error ("division by zero in a literal"), as
today. The fold is value-preserving, so it changes no meaning, and keeps `parse("1/3") ==
Rational(1/3)` (`parser.rs:423` test), `"6/4"` = `Rational(3/2)`, `"1/-3"` = `Rational(-1/3)` as
today. New structures: `"-2/3"` is `Rational(-2/3)` (was `Neg(Rational(2/3))`, same value), `"2^-1"`
is `Pow(2, Integer(-1))` as today.

The runs of signs are loops, not recursion (revision 5, self-attack): with a rule
`('-' | '+') unary`, 100 KB of minus signs would recurse 100 000 levels and overflow any thread; HEAD
accepts one sign only (`--5` is a parse error at HEAD). The parser's other recursion (nested
parentheses, function calls, the right-associative `^`) is HEAD's: on a 2 MB debug thread it reads
about 160 nested calls, more than 200 nested parentheses and 1438 carets of a tower, on an 8 MB release
thread about 4100 nested calls and 22 695 carets (§2.5). The engine reads a tower by a loop and the
chains of `multiplicative` and `additive` by loops (§2.5, revision 7rt1), so of these only parentheses
and calls cost the engine stack, and less per level than they cost the parser.

Also (one-line table entries, same function `function_call`, `parser.rs:254-276`): map `arcsinh` /
`asinh`, `arccosh` / `acosh`, `arctanh` / `atanh` to their `UnaryOp`. Display prints `arcsinh(..)`
(`expression.rs:472`) but the parser read it back as an opaque `Function`, so taylor refused it.

Results (new): every row of §1.6 gets the "correct" column; `1/0` stays an error.

### 2.13 Callers in the crate

- `limits.rs:428-443` `limit_via_series`: use `expansion_first_term(var, At(point), order,
  LIMIT_SERIES_WORK)` with `LIMIT_SERIES_WORK = 100_000_000` (revision 5: no library path runs
  the engine without a budget; without one, `exp(x)^2/x^1000000` exhausts memory, §5.3). Revision 7rt3
  (red team 3 memory, m3): an internal entry of `series.rs` that runs the driver at the order and
  charges, bounds and renders the first stored term only (`render_units` and `render_tree` of that
  term), and returns its exponent and, when it is rational, its value; revision 7rt2 called
  `series_expansion`, which rendered every stored term to read the first, so `1 + x*S_18` (S_k =
  `(1+cos(1))*...*(1+cos(k))`) spent 1e8 units on the rendering of S_18 that no one read and was
  `Unknown (BudgetExceeded)`, where its order-0 expansion costs 5.4e7. `Err(_)` gives `Unknown`; no
  term with precision >= 1, or a first exponent above 0, gives `Finite(0)`; first exponent 0 with a
  rational coefficient gives `Finite(c0)` (one copy of that rational's digits, charged by the term's
  `render_units`);
  anything else `Unknown` (a pole, an undecided leading coefficient or a symbolic constant, as
  before). Correct because terms below the first stored one are exact zeros, a rational stored
  coefficient is nonzero, and a Laurent expansion with valuation >= 0 is continuous at the point
  after removing the removable singularity. A first exponent below 0 is never read as a pole
  without a decided coefficient (review 3, m3).
- `limits.rs:554` `limit_with_error`: the limit as above, then, for a finite limit, the `BigO`
  `O(h^(order + 1))` when the expansion reaches `order + 1` with the same budget; `Err(_)` gives
  `None`. Revision 7rt4 (red team 4 memory, m2): an internal entry `expansion_reaches(var, At(point),
  order, LIMIT_SERIES_WORK)` runs the driver and renders no term (it charges and bounds none, since the
  caller reads only that the expansion succeeded); revision 8 called `series_expansion`, which charged,
  bounded and rendered every stored term, so `1 + x*S_17` was `(Finite(1), None)` (the tree cap after
  0.19 s) where `O(x^6)` is known. Which entries render: `series_expansion` (and so `taylor`,
  `taylor_with_budget`, `laurent`, `asymptotic`, `series_coefficients`, `series_with_big_o`) every
  stored term; `expansion_first_term` the first; `expansion_reaches` none.
- `known_series::{exp, sin, cos, log_one_plus, binomial}` (`series.rs:247-354`): build their sums
  with `balanced_sum` (the same depth problem); `exp` keeps its `simplify`.

### 2.14 `Drop for Expr` with bounded stack use (`expression.rs`, review 1, B4)

The derived drop glue drops a node, then each child `Arc`, and the last owner of a child drops it in
turn: one stack frame chain per tree level. HEAD overflows at 80000 levels (debug, 8 MB), and a
release build overflows at 2*10^5 (§1.2).

```rust
thread_local! {
    /// Highest stack address seen on entry to `Drop for Expr` on this thread.
    static EXPR_DROP_STACK_TOP: Cell<usize> = const { Cell::new(0) };
}

impl Drop for Expr {
    /// Teardown with bounded stack use. While the drop runs less than GLUE_STACK_BUDGET bytes
    /// below the shallowest drop seen on this thread, the drop glue recurses as before. Deeper, the
    /// subtree is torn down iteratively: every child `Arc` that has children, shared or not, is
    /// moved to a heap list of `Arc`s and a shared leaf takes its place, and the loop takes each with
    /// `Arc::into_inner`: the last owner gets the node and moves its children in turn, any other owner
    /// only decrements (revision 7rt3). So the glue below any node drops leaves only.
    ///
    /// The bound is relative to the shallowest `Expr` drop seen on the thread, not to the top of
    /// the thread's stack: a thread whose first `Expr` drop happens deep in its stack treats that
    /// point as the top, and the glue may use up to GLUE_STACK_BUDGET bytes below it before the
    /// iterative teardown starts. A thread with less than 16 KiB of stack left at its first drop
    /// can therefore still overflow there.
    fn drop(&mut self) {
        const GLUE_STACK_BUDGET: usize = 16 * 1024;
        if !has_children(self) { return; }                   // leaves: no other work
        let marker = 0u8;
        let here = std::ptr::from_ref(std::hint::black_box(&marker)).addr();  // compared, never dereferenced
        let used = EXPR_DROP_STACK_TOP
            .try_with(|top| { if here > top.get() { top.set(here) } top.get().saturating_sub(here) })
            .unwrap_or(usize::MAX);                          // TLS being destroyed: iterative
        if used < GLUE_STACK_BUDGET { return; }              // the glue recursion is shallow here
        let mut list: Vec<Arc<Expr>> = Vec::new();
        detach(self, &mut list);
        while let Some(a) = list.pop() {
            if let Some(mut e) = Arc::into_inner(a) {       // the last owner: tear it down here
                detach(&mut e, &mut list);                   // e then drops with leaf children only
            }                                                // another owner: one decrement
        }
    }
}
thread_local! { static LEAF: Arc<Expr> = Arc::new(Expr::Real(0.0)); }
fn has_children(e: &Expr) -> bool   // Binary, Unary, or Function with arguments
fn detach(e: &mut Expr, list: &mut Vec<Arc<Expr>>) {
    // for each child Arc `a` of e with has_children(a): push std::mem::replace(a, leaf), where
    // leaf = LEAF.try_with(Arc::clone).unwrap_or_else(|_| Arc::new(Expr::Real(0.0)))
    // (revision 8: LEAF may already be destroyed while the thread's locals are torn down)
}
```

Correctness:

- Every child `Arc` with children is moved, shared or not (revision 7rt3, red team 3 structure m1).
  Revision 7rt2 moved only uniquely owned children (`Arc::get_mut`) and left a shared one in place, to
  be dropped by its last owner; when that last owner was the node's own glue (a node that holds one
  child twice, `Binary(Mul, a, a)`, or a child shared with its sibling's subtree), the glue tore the
  child down one level deeper, so a chain of such nodes recursed as HEAD does: 14 513 levels of `twin`
  and 14 455 of `sib` overflowed a 2 MB debug thread. `Arc::into_inner` returns the node only to the
  last owner (a strong count of 1, no `Weak`; the workspace has no `Weak<Expr>`), so a shared node is
  torn down exactly once, by whichever reference goes last, in this loop or in another `Drop`.
- Every node is dropped exactly once: a moved `Arc` is owned by `list`, its old place holds a clone of
  the shared leaf `Real(0.0)` (an increment), and the node is dropped at the end of its loop iteration
  after its children with children were moved.
- Teardown of the thread's locals (revision 8, review 8 m3): an `Expr` held in a thread-local is
  dropped by the thread's TLS destructors, possibly after `LEAF` was destroyed. `EXPR_DROP_STACK_TOP`
  is a `const` `Cell<usize>` with no destructor, so its `try_with` succeeds and the drop takes the
  iterative branch when it is deep; revision 7rt3's `detach` then called `LEAF.with`, which panics
  there ("cannot access a Thread Local Storage value during or after destruction"); the thread does
  not finish its teardown (in the measurement the process printed panic after panic until it was
  killed). `detach` reads `LEAF` with `try_with` and makes a
  fresh leaf when it is gone (one allocation per moved child, freed with its parent in the same loop
  iteration). No `Expr` lives in a thread-local in the workspace today (one `Cell<i64>` in
  rustmath-integers), so this is for future callers. Measured with `EXP/rev8/probes/dropm.rs`, mode
  `tls` (the value is put in a thread-local whose destructor is registered before `LEAF`'s; §5.4).
- Stack bound: in the iterative branch, the glue below a node drops only leaves, and a node another
  owner still holds is not dropped here. Above the switch, at most 16 KiB of glue frames (about 80 to
  160 levels) are used. Measured on the mirror enum of red team 3 (`EXP/rev7rt3/probes/dropm.rs`, HEAD's
  field types, a result converted with its sharing): `twin`, `sib`, a unique chain and a left-deep sum
  with 2 x 10^5 levels each drop on a 2 MB debug thread with at most 17 KB of stack, and 10^6 levels on a
  1 MiB release thread in at most 0.11 s (HEAD's glue: `twin` overflows a 2 MB debug thread at 20 000).
- The address of a local is a safe proxy for the stack depth (`ptr::from_ref(..).addr()`, no
  `unsafe`). The stack grows downward on all Rust tier-1 targets. On a target where it grows
  upward, `saturating_sub` gives 0 and the drop is the derived glue, as at HEAD. A thread whose first
  `Expr` drop happens deep in its stack treats that point as the top, so it may use 16 KiB more
  before switching; documented.

Measured (`EXP/rev1/`, the variant is `rev1/sym_v10/src/expression.rs`, compiled into a copy of the
crate with `rev1/build_sym.py`; `drop_measure.txt`):

| check | HEAD | this design |
|---|---|---|
| left-deep sum of 10^6 terms, debug, 1 MiB thread | overflows | built 0.50 s, dropped 0.38 s |
| right-deep power tower, unary chain sin(sin(..)), function chain f(f(..), x), zig-zag `+`/`*`, each 10^6 deep, debug, 1 MiB | overflow | ok |
| 10^6-term sum dropped after 520 frames of about 1.6 KiB on a 1 MiB thread (less than 16 KiB left: 530 frames alone overflow), after shallow drops on the same thread had recorded the stack top (`rev1/workload.rs`, `shapes`); when a thread's first `Expr` drop happens deep in its stack, the glue may still use 16 KiB there (review 2, m4) | overflow | ok |
| a shared deep subtree (two owners), dropped one owner at a time | | ok |
| 3*10^4-term sum, drop only, release, min of 7 | 4.4 ms | 3.8 ms |
| 3000 x (parse, 3 x (differentiate, simplify)), release, min of 8 runs | 107.5 ms | 106.0 ms |
| same loop, 300 iterations, instructions (cachegrind) | 96.5 M | 103.4 M (+7 %) |
| HEAD `taylor(exp(x), 0, 10)` (allocation heavy), release | 188 ms | 179 ms |

Rejected variants (same directory): always-iterative with a `Vec` per drop (`v1`, `v3`), with a
thread-local list (`v8`), depth-counted recursion (`v9`): +18 % to +28 % wall time and +19 % to
+48 % instructions on the differentiate loop, because every internal node pays an `Arc::get_mut`
(an atomic compare-exchange) and two moves.

Consequences: implementing `Drop` forbids moving fields out of an owned `Expr` (E0509). The full
list of sites (found by compiling the workspace chain with this `Drop`, `EXP/dropchain2.txt`) is in
§3.3; every one is fixed by matching on a reference and cloning the `Arc` contents where a value is
needed.

### 2.15 Structural caps (revision 5; revised in revisions 6, 7, 7rt2, 7rt3, 8 and 7rt4)

Five caps bound the structures that the budget alone bounds only through their cost (and since
revision 7rt4 a slot cap of `alloc` for calls without a budget, the last row). Each is checked
before the work it bounds and gives `Err(SeriesError::LimitExceeded { limit, value })`, with `limit` one
of `StructureLimit::{MonomialFactors, Kernels, ConstantTerms, KernelDepth, TreeSize}` and `value` the
cap. The fifth (revision 7rt3, red team 3 structure H2) bounds what a consumer of the result walks:
the budget bounds a result's DAG, but a tree-walking consumer (`simplify`, `substitute`, `symbols`, the
derived `PartialEq`, the REPL arms that call them) pays for its tree. The
REPL maps it to "ComputationError" (§3.1). No cap can be reached inside the zero test: the test copies
terms and evaluates expressions, and it builds no larger constant, no longer monomial and no kernel. So
the zero test never turns a cap into `Unknown`; a cap reached by an operation is returned with `?`, as
`BudgetExceeded` is. The values are unchanged since revision 5; the margins and the measured refusals
below are those of the revision 7 prototype (`EXP/rev7/attack7_final.txt`, `newtests7_dbg.txt`; the
same outcomes as revision 6, times within the noise), rerun on the revision 7rt1, 7rt2 and 7rt3 prototypes
with the same outcomes (`EXP/rev7rt3/caps_rt3.txt`, `attackrt3_final.txt`, `newtests_rt2_dbg.txt` there;
revision 7rt3 adds the fifth cap, whose refusals are in its row). The
work stack of revision 7rt2 needs no cap: it holds at most one record per node of the current path, at
most D, each charged when it is made (§2.16).

| cap | value | error | checked where, before what | bounds | largest in §4.1 and §5.3 (margin) | refuses (measured) |
|---|---|---|---|---|---|---|
| `MAX_MONO_FACTORS` | 2^15 = 32 768 | `LimitExceeded(MonomialFactors)` | `mono_mul`, before each factor is pushed; no other operation lengthens a monomial (`pow_u` and `inverse` keep the length, a kernel is one factor) | one monomial: 512 KB of factors, the depth of its product (`bal(2^15) = 16` levels), the work of each pair | 13 000 (the symbol product of §5.3), 8 192 (`x + P_4096*P'_4096`), 4 096 (`budget_long_monomials`): 2.5 | a product of more than 32 768 distinct kernels. A 100 KB input names at most about 26 000 distinct symbols, so no parsed input of that size reaches it |
| `MAX_KERNELS` | 2^16 = 65 536 | `LimitExceeded(Kernels)` | `Work::intern`, before a new kernel is created | the interning table (ids, keys, `KernelData`), and the length of every monomial (its factors are distinct kernels) | 20 001 (`1/(x + sum of 20 000 symbols)`), 11 000 (`sin(1)` to `sin(11000)`), 9 001 (the numeric probe of §5.3): 3.3 | 40 powers `(x + sin(k)^2 + cos(k)^2 - 1)^1000000` at order 2000 without a budget (2000 kernels `c0^(n-m)` each): after 107 s, 198 MB. With the REPL budget, 6 such powers are refused by the budget after 0.18 s |
| `MAX_CONST_TERMS` | 2^20 = 1 048 576 | `LimitExceeded(ConstantTerms)` | `mul` (before a new accumulator entry), `merge2` and `sum_many` (before a new term is pushed) | one constant: its sum has depth at most `bal(2^20) = 21`, a search at most 21 levels of comparisons, and the accumulator of a product at most 2^20 entries | 262 144 (K and S): 4 | `x + S_21*sin(x)` at 0, order 0, without a budget: after 1.1 s, 388 MB (S_k here with `(1+sin(j))`; HEAD returns the product unexpanded). `invpowx(4)` without a budget (§2.2, below): after 2.5 s, 674 MB. With the REPL budget, `x + S_20*sin(x)` is refused by the budget first (after 0.06 s); S_19 is `Ok` (9.92e7 units, 227 MB of heap, 230 MB VmHWM) |
| `MAX_RESULT_TREE` (revision 7rt3) | 2^21 = 2 097 152 nodes | `LimitExceeded(TreeSize)` | the driver, before the result is rendered (`render_tree`, O(n) from the cached tree sizes of the coefficients and h), and `UndecidedZero` before its constant is rendered (O(1)); since revision 7rt4 the cached sizes count each kind of kernel exactly (§2.2, `KernelData::tree`): `render_tree` is never below the result's tree and exceeds it by the point's nodes and a few per term (8 to 68 nodes on red team 4's families), so the margins below are exact counts | the nodes of the result as a tree, so a consumer that walks it costs at most the cap times its cost per node: HEAD's `simplify` and `substitute` copy about 92 bytes per node (red team 3), at most about 200 MB | results that HEAD also answers: `x^x^...^x` at 2 with 18 000 carets (a tree of 41 991 nodes), `(1)/(1-x)` at order 20 000 (79 999; `exp(x)` there without a budget 119 997), `F_30` at 1, order 0 (151), `F_11` at 1, order 1 (1 036 403 since revision 7rt4, 1 046 643 before): 2.0 | `F_12` to `F_15` at 1, order 1 (trees of 2.6e6 to about 3e7: the x coefficient has 2^k terms; HEAD keeps the product and answers in 0.2 ms; the budget refuses `F_16` on, as it refused `F_20` on before). The cap refuses these although the engine's own work is small (revision 8, review 8 m4): `F_10` at order 1 expands in 2.4e6 units and 5 ms, `F_11` in 5.0e6 and 11 ms, and `F_12` to `F_15` are refused after 36 ms to 0.26 s, below the budget; the cost is the expanded normal form of the x coefficient, not the computation. The remedy for both the cap and the budget on this family is a representation that keeps products, which §7 item 8 and the HANDOFF row name; `(-1)*(a*b*...)/(1-x)` with 2700 names from about order 400 (a tree of 2.06e6 at order 380; HEAD does not finish); `(F)/(1-x)` with a 33 000-call F from order 63 (2.01e6 at order 60; HEAD does not finish). Revision 7rt4 (red team 4 structure, m2): towers and nested functions, whose expanded normal form names deep kernels many times, polynomially in the levels: the order-n coefficient of a k-level tower `x^x^...^x` at a point names the kernels `E_j = exp(log(a) E_(j-1))` about k^(n+1) times, a tree of about k^(n+2) nodes with few distinct ones (59 458 at 20 carets, order 2), while HEAD keeps the derivative nested (about k^2). Measured on proto13: `x^x^...^x` at 1/2, order 2, `Ok` up to 21 carets (a tree of 1 867 473 nodes in 0.04 s), refused from 22 (45 bytes; HEAD 8 ms and 11 MB); at 2, order 1, `Ok` up to 136 carets, refused from 137 (HEAD 21 ms at 200 carets); `cos(x+1)^...^(x+2)` with 150 levels at 0, order 1 (HEAD 19 ms). Each refusal takes at most 0.25 s and 26 MB. Revision 8 refused from 20 carets at 1/2 and 119 at 2 (the over-count of computed kernels, §2.2); the remedy is the one for `F_k` (§7 item 8) |
| kernel depth | max(1024, 4 D), D the depth of the input and of the point (the pre-pass computes both) | `LimitExceeded(KernelDepth)` | `mk_kernel`, before the key is written (O(1): the depth of an input node is known from the pre-pass, the depth of a computed argument is cached in its constant) | the depth of every kernel expression, so of every coefficient (at most the largest kernel depth + 40: `bal(t) <= 21`, `bal(L) <= 16`, 3 operator levels, §2.2) and of the result (+ 1 + `bal(n)`, n the stored terms; `bal(2001) = 12`) | `x^x^...^x` at 2, 18 000 carets (revision 7rt1): 36 002 for an allowed 72 004; `x + sin^4000(1)`: 4001 for 16 008; `invpow(80)`: 241 for 1024; N_30: 106 for 1024; `budget_long_monomials`: 18 for 1024: 2 | none measured (below) |
| `MAX_SERIES_SLOTS` (revision 7rt4) | 2^57 = isize::MAX / 64 slots | `Unsupported("a series buffer of more than 2^57 slots")` (not a `LimitExceeded`: it is reached only without a budget) | `Ctx::alloc`, after the saturating charge of `16 n` and before the cast and the allocation of a dense buffer | the dense buffer of every recurrence and of a product, so that no slot count from the input (a pole order, a valuation, a power pole, the order of `series_expansion`) reaches `Vec::with_capacity` beyond what it can hold; with a budget the charge refuses first (`16 n > remaining`) | 40 000 (`sin_cos` at `TAYLOR_ORDER`, two buffers of 20 001): 3.6e12 | without a budget, `exp(x)/x^(2^60)`, `sin(x)/x^(2^61)`, `(1)/(x^(2^61)+x^(2^61+1))` in under 0.1 ms; revision 8 panicked ("capacity overflow") on these from 2^60 slots, with or without a budget (red team 4 series, H2) |

Which inputs could reach the kernel-depth cap (review 6, m1). A kernel made from an input node takes
that node as its expression (§2.2), so its depth is at most D. Since revision 6 that includes the
`Inverse` kernel of a divisor the input names (`inverse_src`): revision 5 rendered P0's normal form
there, and `invpow(80)` (`x + (1)/((1 + ... (1)/((1 + sin(1))^511) ...)^511)`, 80 levels, 1.1 KB)
added 13 levels of kernel per 3 levels of input; the unbounded `taylor` met the cap after 19.3 s, where
HEAD returns in 10 ms. Now it is `Ok` in about 3 s, kernel depth 241, result depth 243, the input's
depth (HEAD: 248); with the REPL budget it is refused after 0.15 s. Only kernels made from computed values
still render normal forms: the value of a series at the point, the strict inverse of a leading
coefficient, a power kept by the caps, a radical. Their depth grows by at most `bal(t) + bal(L) + 4`
per level of kernel. A level of kernel needs at least two levels of input (a function or a division,
and the sum or power that makes the computed constant), except for `f^g` with both sides depending on
the variable, which is `exp(g log f)`: at a point a other than 0 and 1, `x^x^...^x` makes one kernel
`exp(log(a) K)` of two levels per level of input, so its result has depth 2 D + 2 (red team 1, H2:
36 002 for 18 000 carets, within the allowed 72 004; 20 800 since revision 7rt3, whose values at the
point render from the input). Otherwise the growth exceeds 4 D only when
each such pair of input levels expands into hundreds of terms. The measured case is `invpowx(k)`, the
nest above with `(x + sin(1))` inside, so that every inverse is the strict inverse of a computed
512-term leading coefficient: its x coefficient multiplies 512-term constants at every level, so the
term cap (unbounded, 4 levels, 2.6 s) and the budget (0.35 s) come first; HEAD keeps the products
unexpanded and returns in 1 ms (§7 item 8).

Recursive walks over kernel expressions (orchestrator item 1). The depth cap bounds how deep a
consumer of the result recurses (`PartialEq`, `substitute`, `simplify`); the tree cap bounds how many
nodes it visits. The engine itself never recurses over a kernel
expression: the ball enclosure and its cost walk use explicit stacks with a memo of distinct nodes
(§2.3); a key names sub-kernels by id and walks only the normal forms of the arguments, iteratively
(§2.2); a rendering refers to each kernel's `Arc` and recurses only over the balanced trees it builds,
at most 21 levels (`balanced_arc`); `display_estimate` uses an explicit stack and stops after about 34
nodes (§2.9).

Why these values:

- Display needs no stack per level of the result (a work list since revision 7rt1, §2.11), nor does the
  REPL's LaTeX writer (§3.1), so the depth of a result does not limit its printing. Revision 7 argued
  instead that a result prints on the thread where its input could be parsed, because the parser reads
  at most about 160 nested levels on a 2 MB debug thread; with towers read by a loop (§2.5) the parser
  reads 1438 carets there and 22 695 on 8 MB release, and the tower at 2 above, of depth 2 D + 2,
  aborted HEAD's recursive Display on an 8 MB release thread at 18 000 carets (red team 1, H2). The
  consumers that still recurse once per level of a result are the derived `PartialEq`, `substitute` and
  `simplify` (§7 item 5). `PartialEq` compares a result of depth 50 002 in 0.6 ms on an 8 MB release
  thread and one of depth 5892 on a 2 MB debug thread (`EXP/rev7rt1/partialeq_rt1.txt`). A result is
  at most as deep as the larger of the input's depth and `2 D + 2` (a tower at a point other than 0
  and 1), plus the 40 levels of a coefficient's rendering and the result's sum. Both can pass those
  limits on the thread that parsed the text: the tower at 2 with 22 695 carets, the deepest a text
  gives on 8 MB release, has depth 45 392 (parsed, expanded, printed in 4 ms and compared in 1.5 ms
  there), and with 1438 carets on 2 MB debug 2878; but a kernel made from an input node takes that
  node, and the parser reads a `+ -` chain by a loop, so `x + sin(1+1+...+1)` with 20 000 ones (40 KB)
  gives a result of depth 20 003 on 2 MB debug, which `PartialEq` cannot compare there (red team 2
  structure, m2; the input is as deep, and `PartialEq` cannot compare it either; HEAD's `taylor`
  aborts at 5000 ones). So `PartialEq` is not claimed for every result of a parsed text, only for
  results within the measured depths above (revision 7rt2; revision 7rt1 claimed it for every
  result of a text the parser reads on the same thread). HEAD's own result of such an input has
  about the input's depth. The parser figures are HEAD's parser, which the prototype links; §2.12 rewrites
  `unary`, `power` and `multiplicative`, so the implementer measures the new parser's limits and checks
  them against the engine's (§6 step 8, review 7 m3).
- The factor 4 relative to the input: see "Which inputs could reach the kernel-depth cap" above.
- The other three caps have a margin of at least 2.5 over every input of §4.1 and §5.3 and over what a
  100 KB input can name. Under the REPL budget the budget is reached first on every probe of §5.3; the
  caps matter for `taylor` without a budget, where they turn an exhaustion of memory into an error.

Tests (`structural_caps`, unit tests, outcome asserted): `Work` carries the five caps (the values
above by default), so each test sets a small one and checks that the next operation is
`LimitExceeded` with the right `StructureLimit` before the work it bounds: a product monomial past a
factor cap of 8; the ninth kernel with a kernel cap of 8; a product, a `merge2` and a `sum_many` past
a term cap of 16; a kernel whose argument depth passes an allowed depth of 8; a result whose tree
passes a tree cap of 64 (`exp(x)` at order 20), and an `UndecidedZero` constant past it. With the default caps,
`x + S_21*sin(x)` at 0, order 0, without a budget, is `LimitExceeded(ConstantTerms)` after 1.1 s in
release (7.9 s debug and 380 MB, so it is measured by `EXP/rev6/tools/newtests_dbg.py`, not run by
`cargo test`).

### 2.16 Invariants: what bounds every loop, allocation and recursion (revision 5; revised in revisions 6, 7, 7rt1, 7rt2, 7rt3, 8 and 7rt4)

Every loop, allocation and recursion of the engine is listed here by function: the design name, with
the prototype function of `EXP/rev7rt4/proto13/` in parentheses where it differs (revision 7rt3: the
revision 7rt2 prototype plus the rule for empty truncated operands, the rule of `0^r` and of a base
that expands to a constant, the tree sizes, the rendering hints and the tree cap, MPFR's inverse
hyperbolic functions, integer exponents up to i64, the doubling of a stalled loss-driven step, the
unsigned offsets of `inv`, the first-term limit path, one rendering per distinct coefficient and the
float Display; revision 8: the exponent's flag in the tower's constant-exponent state, the exact
integer value of an exponent in the enclosures, `zero_pow_sign` for a constant `0^alpha`, and the
fallback leaf of `Drop`; revision 7rt4: the precision out of band (`Prec`), saturating units and the
slot cap of `alloc`, the poles of a rational argument, exact literal balls and the exact-zero rules,
`exp(k log a)` as an integer power, the
per-kind kernel tree, the hint margin and `expansion_reaches`; none adds a loop or a recursion). Anything not on this
list is a hole, and the port adds no loop, allocation or recursion without a row. Notation: B is the
budget; t, L and F are the terms of a constant, the factors of one monomial and the factors of all its
monomials; w the words of a coefficient; n is the number of terms of a series or the slots of a
buffer; D and N are the depth and the number of distinct nodes of the input together with the point
(the pre-pass computes both). "Charged" means charged before the work, by the formula of §2.7. The
caps are those of §2.15: L <= 2^15, at most 2^16 kernels, t <= 2^20, kernel depth <= max(1024, 4 D).
"Covered" means that the work is O(the units of a charge made just before it). The table "Memory"
at the end gives, for every kind of heap object, its bytes and its units.

Three facts carry most rows:

1. Every size a charge reads (`words`, `copy_units`, `expr_units`, the key bound, the class flags,
   the depth, and since revision 7rt3 the tree size) is cached in `ConstData` when the constant is
   built (§2.2), so computing any charge is
   O(1) or O(number of operands). Revision 6 caches two more class flags (`any_imag`, and
   `one_named_imag` for rule 4 of the zero test), so every class test of the zero test is O(1);
   revision 5 walked the factors in `const_status` at every test, uncharged, which a shared constant
   at the point (each `1/x` at the point K tests K again) would have repeated.
2. Every constant is built by one constructor, `from_sorted`, which charges `CONST_BASE` before it
   allocates the header. Its terms were built by an operation that charged at least `copy_units` of
   them or, for a product, `MONO_FACTOR_UNITS` per factor, `ACC_ENTRY` and `OP_BASE` per pair. So the
   constructor's own pass over the terms (`ksize`, O(t + F)) is covered, and so is the memory of the
   terms (table "Memory").
3. The engine has no recursion over the input (revision 7rt2, red team 2 structure H1): `expand` and
   `from_expr` each run one loop over an explicit work stack on the heap, with at most one record per
   node of the current path (at most D), so a call uses O(1) native stack whatever the input's shape.
   The only recursions of the engine are over `log2` of a length (balanced trees, at most 21 levels)
   and a constant depth (at most 2 levels: a function calling itself once). Outside the engine, the
   parser recurses over its nesting and `simplify` over right children, so on the thread that parsed a
   text the parser is the only recursion over it and stops first, whatever grammar levels the text
   uses between two parentheses (§2.5; revision 7rt1 had engine loops for the rules that repeat
   without a parenthesis but recursed once per nested operand, and a text using five levels per
   parenthesis overflowed the engine first). A port adds no recursion over the input: a new node kind
   gets a record of the work stack. Every walk of a kernel expression uses an explicit stack, and so
   does the substitution of a regular node (revision 7rt3, memoized per distinct input node). Display
   and the REPL's LaTeX writer use work lists (revision 7rt1); the other consumers of the result
   (`PartialEq`, `substitute`, `simplify`) recurse over its depth and visit its tree, which §2.15
   bounds (the depth cap and, since revision 7rt3, the tree cap).

`series_const.rs` (prototype `konst.rs`):

| function | loops and allocations | recursion | bound |
|---|---|---|---|
| `Work::charge`, `remaining` | none | none | O(1); the base 16 added with `saturating_add`, the saturated sum compared (revision 7rt4) |
| `Work::intern` (`intern`) | a hash and a lookup of the key; for a new key one table entry and one `KernelData` in an `Arc` | none | the key is charged 1 unit per byte before it is written (its `String` has that bound as capacity); the kernel cap and `KERNEL_UNITS` before a new entry |
| drop of `Work` (`drop_table`) | sorts the kernels by id, O(k log k); drops them newest first | none: an `Inverse` kernel's P0 refers only to older kernels, which the table still holds when it drops | k <= 2^16 |
| charged arithmetic (`q_mul` to `i_exact_root`, `pow_bits`) | the big-number operation and its result | none | charged by its §2.7 formula right before it runs; `q_mul`, `q_add`, `q_sub` and `i_mul` at least `COPY_WORD` per result word (revision 7, `result_units`) |
| `words`, `copy_units`, `expr_units`, key bound (`nf_key_size`), class flags (`pure_indet_nonconstant`, `real_free`, `any_opaque`, `any_imag`, `one_named_imag`) | none | none | O(1), cached |
| `is_zero`, `as_rational`, `get` | a binary search, O(log t) comparisons; `as_rational` clones a one-term rational | none | O(log t + w), in an operation charged at least w |
| `zero`, `one` | none: shared constants | none | O(1) |
| `rat`, `kernel` | one constant of one term | none | its constructor's `CONST_BASE`, and the term's own allocations, charged by `rat` (`COPY_WORD w + 16`) and `kernel` (32) before it |
| constructor `from_sorted` (`from_sorted`, `raw`, `ksize`) | one pass over terms and factors, O(t + F); one `Arc` header; an in-place shrink of the vector to a boxed slice | none | `CONST_BASE` charged before the header; the pass and the terms covered (fact 2) |
| `add`, `sub` (`merge2`) | one two-pointer merge, at most ta + tb steps, each a monomial comparison, a clone and at most one `q_add`; one vector of capacity ta + tb | none | `copy_units` of both charged (also the comparisons: at most F factor comparisons in all); every `q_*` charged; the term cap before each push |
| `neg`, `scale` | one map over the slice, exact size; one `q_mul` per term (`scale`) | none | `copy_units` charged; each `q_mul` charged; `scale` by 1 and adding 0 share the constant |
| `sum_many` | a heap of k cursors: per term one pop, at most one push, one clone and at most one `q_add`; vectors of k entries and of at most the summands' terms | none | `copy_units` of every summand, `SER_TERM` per summand and `search_units(excess, k)` charged; the term cap before each push |
| `product_many` | one zero test per factor (a length check); `ceil(log2 n)` rounds of pairwise `mul`, one vector per round | none | each `mul` charged; the factors were charged as chain items (`KONST_NODE`) |
| `mul` | ta x tb pairs, each a `mono_mul`, a `q_mul`, at most one `q_add` and one map search; the accumulator, at most 2^20 entries; one linear conversion to the slice that frees the map's nodes as it goes | none | `MONO_FACTOR_UNITS (ta Lb + tb La) + search_units + ACC_ENTRY ta tb` charged first; every `q_*` charged; the term cap before each new entry |
| `mono_mul` | a merge, O(La + Lb), one vector | none | called only by `mul` (covered); the factor cap before each push |
| `reduce_mono` | one pass in place, O(L); one `q_pow` and one `q_mul` per folded radical | none | inside a charged `mul`, `pow_u` or `inverse`; each `q_*` charged |
| `zero_status` | one term: none (a cached flag); several: without indeterminates `const_status`, else `indet_groups`, then `const_status` per group | none | below |
| `indet_groups` | one vector of t (indeterminate part, rest, coefficient) triples, an in-place sort (t log t comparisons), one constant per group | none | `2 copy_units + 2 search_units` charged; each group's constant by its constructor; O(1) without indeterminates (a single group, the constant itself) |
| `const_status` | none (cached flags) | none | O(1), then the numeric test |
| `split_imag` | a counting pass over the factors; two vectors of exact capacity; a sort of the imaginary part | none | `copy_units + search_units` charged; O(1) without the kernel I (A is the constant itself) |
| numeric test (`numeric`, `proves_nonzero_ab`), `non_integer_status` (`non_integer`) | at most 3 precisions, 2 enclosures each; a `TooWide` precision goes on, a `Never` stops (revision 7rt2) | none | `expr_units` and `ball_cost` charged per precision |
| `inverse`, `inverse_src` | one term (no Opaque factor, the cached flag): one pass building the new monomial (exact capacity), one `exact_pow_if_small` per `Inverse` factor; several terms: one `i_gcd` and one `i_lcm` per term, one `scale`, one kernel | none | `rat_words + MONO_FACTOR_UNITS (L + 1)` charged; each `i_*`, `q_*`, `scale`, `mul` and the kernel charged; the expression of an input divisor's kernel is a clone of the input node and at most one quotient node (16 units plus the quotient's) |
| `pow_u` | one term: one pass building the new monomial; several: at most 2 x 12 `mul` (n <= 4096) | none | `MONO_FACTOR_UNITS (L + 1)` and `q_pow` charged; each `mul` charged |
| `exact_pow_if_small` (`power_size`) | the multinomial count, at most min(t, 513) iterations of `u128` arithmetic | none | O(513), cached sizes otherwise |
| `to_expr`, `to_expr_raw`, `to_expr_arc_raw`, `mono_expr`, `kpow`, `balanced_arc` | at most `2 + L + 2 pf` nodes per term, two vectors of one pointer per term and per factor; a constant with a hint returns it (one `Arc` increment, revision 7rt3) | `balanced_arc` recurses `ceil(log2 n)` levels | `expr_units` charged before (`to_expr`; kernel creation charges its arguments' `expr_units`; the driver charges `render_units`), which a hint only over-counts; recursion at most 21 levels |
| `tree`, `depth_eff`, `set_hint` (revision 7rt3) | one lookup in the hint table of `Work` (none while it is empty); `set_hint` inserts one entry, at most once per constant | none | O(1); `HINT_UNITS` per entry; the hint's node was charged by its builder (`subst_at`, `hint_exp_arg`, the quotient of `inverse`) |
| `bal_depth` | none | `log2 n` | at most 21 levels |
| key writing (`write_nf`, `hex_int`) | O(key bytes) | none | charged 1 unit per byte |
| `mk_kernel` (`mk_kernel_src`, `mk_kernel_full`, `mk_kernel_inner`) | the key; for a new kernel its expression: a clone of the input node (shallow: its children are shared `Arc`s), or `to_expr_arc_raw` of the arguments (a hint where one is set); one `Arc` | none | the depth cap (O(1), from the arguments' effective depths) and the key charge first; `16 + node_payload_units` for an input node (revision 7) or `16 + expr_units` of the arguments before the expression; the tree size O(1) from the pre-pass or the arguments (revision 7rt3), counted per kind of kernel (revision 7rt4) |
| `dth_power_part` | at most 6542 candidate primes; at most bits(n)/d exact divisions | none | `(w(n) + 8)` per candidate charged first (n <= 4096 bits); `i_exact_root` charged |
| `small_primes` | a sieve to 2^16, once per process | none | O(65 536) once |
| `pow_const` (`kpow_const`, `kpow_const_src`), `apply` (`kapply`, `kapply_src`), `sin_q_pi`, `as_q_pi`, `radical_kernel` | none beyond the operations they call | `kpow_const` calls itself once (a perfect d-th power) | each operation charged |
| `log_exact` | one `add` per factor of one monomial | none | each `add` charged |
| `from_expr` (`konst_of`, `konst_visit`, `KFrame::resume`; revision 7rt2) | one loop over a work stack: per node visit a leaf's constant, or one record pushed (16 bytes; a chain, a tower, a call or a `%` node also a boxed record of at most 96 bytes); a finished value is handed to the record on top | none (an explicit stack on the heap, at most one record per node of the current path) | `KONST_NODE` per node visit, plus `COPY_WORD` per word of a literal (revision 6; revision 5: 1 per word), charged first, and `RECORD_UNITS = 32` per boxed record; at most D records; an input symbol's kernel is made by the non-inlined `symbol_kernel` (with its name's `node_payload_units`, revision 7) |
| chain records (`KSum`, `KProd`, `chain_items`; revision 5, `* /` one chain since revision 7rt1) | the left spine of `+ -`, or of `*` and `/`, by a loop into one vector of (flag, node); one constant per item, a divisor inverted as it is read (`inverse_src(false, node)`); one `sum_many` or `product_many` | none | `KONST_NODE` per item charged first; each operation charged |
| tower record (`KTower`, `tower_levels`; revision 7rt1) | the right spine of a tower by a loop (through `Neg` nodes) into one vector of (node, base, negations); one constant per base, then the top; `pow_const` per level from the inside out and one `neg` per negation | none | `KONST_NODE` per spine node charged first; each `pow_const` and `neg` charged |
| call and `%` records (`KFunc`, `KMod`) | one constant per argument; one kernel | none | the kernel charged as above |

`series_ball.rs` (prototype `ball.rs`):

| function | loops and allocations | recursion | bound |
|---|---|---|---|
| `enclose_real` (`ball_expr`) | a post-order walk with an explicit stack; one memo entry, one stack entry and a constant number of MPFR operations per distinct node; since revision 8, at a power node with a non-literal exponent, `small_int_value`: at most 64 nodes read, no allocation | `small_int_value` recurses at most 8 levels | `ball_cost` charged first counts every distinct node, its operations and its memory (`BALL_NODE`); a non-literal exponent's power `4u` | `ball_cost` charged first counts every distinct node, its operations and its memory (`BALL_NODE`) |
| `ball_cost` | an explicit stack; one `seen` entry per distinct node; for an erf node, a look at its argument through at most 4 negations, and for a literal the bounds `lo <= abs(v) <= hi` from the bit lengths and the top 64 bits of its parts (`literal_abs_bound`, O(1) since revision 7rt1; revision 7 converted the literal to f64, O(n)) | none | stops once its sum passes the remaining budget, and every node adds at least `u + BALL_NODE` |
| `Ball` operations (`add` ... `unary`) | O(1) MPFR operations at precision at most 2p + 64 (sin: up to about 3p + 64 near +-1, where MPFR doubles its precision, and so charged since revision 7rt1), except erf: MPFR's erf only when `abs(m) < X_p` (its Taylor sum, at least `x^2` terms at up to about 2p bits, inside a Ziv loop), else the cheap enclosure, which calls no MPFR function (revision 7); asinh, acosh, atanh one MPFR call each since revision 7rt3 | `unary` calls itself once (tan; for an integer exponent beyond i64, abs, log and exp) | covered by the node's units; erf by `erf_units`, measured at 4.0e7 units/s or more, every operation at 2.1e7 or more (the scan of §2.3) |
| integer power of a ball | at most 2 x 64 products for an exponent in i64 (revision 7rt3: 33), else one `abs`, `log`, product and `exp` (`big_int_pow`) | none | `u (2 min(bits(k), 64) + 1)` |
| `real_sign` (revision 7rt3, the sign of A for `0^r`; since revision 8 also for a constant `0^alpha`, through `zero_pow_sign`) | at most 3 precisions, one enclosure each; a `TooWide` precision goes on, a `Never` stops | none | `expr_units` and `ball_cost` charged per precision (`enclose`), as the zero test |
| `floor_small` | at most p/8 digit steps, p <= 1024 | none | O(128) |

`series_engine.rs` (prototype `ser.rs`):

| function | loops and allocations | recursion | bound |
|---|---|---|---|
| `Laurent::trunc`, `trunc_hard`, `coef`, `val`, `from_dense` | O(n) retain or filter; a binary search | none | after an operation that charged at least `SER_TERM` per term or 16 per slot |
| `alloc` | none | none | charges 16 per slot (saturating) before the cast and before the caller allocates; above `MAX_SERIES_SLOTS = 2^57` slots `Unsupported` without a budget, so no slot count from the input reaches `Vec::with_capacity` unbounded (revision 7rt4) |
| `add` | a merge, O(na + nb); the constant adds | none | `16 + SER_TERM (na + nb)` charged; each constant add charged |
| `sum_series` | a heap of k cursors: per term one pop and at most one push; one vector per exponent's coefficients, reused; one `sum_many` per exponent | none | 24 per series and `SER_TERM + SEARCH_CMP (bits(k) + 1)` per term charged |
| `neg`, `shift`, `split0`, `offsets`, the slices of `pow_int`, `scale`, `deriv`, `integrate`, the offsets of `inv` (unsigned distances below the slot count, revision 7rt3) | O(n) | none | `16 + SER_TERM n` charged (`neg`: plus `copy_units` per coefficient; `scale`, `deriv`, `integrate`: plus each constant operation; revision 6 adds the charge to `scale`, whose product by a coefficient 1 is free, and to `deriv`, `integrate` and `inv`'s offsets) |
| `mul` | the pair count, then the pairs into a dense buffer or by a heap merge (a heap of one cursor per row of the shorter operand, one output vector); the span and every pair sum checked, a span that does not fit taking the heap merge (revision 7rt2); the cutoff a `Prec` (`covers`), `Exact` only when no pair was dropped (revision 7rt4) | none | the count adds at least 2 per pair and stops once its sum passes the remaining budget; the pair weights charged in one sum; the buffer charged per slot, the heap merge per pair (`SER_TERM + SEARCH_CMP (bits(rows) + 1)`) and per row (16) before it runs |
| `inv`, `exp_u`, `sincos_u`, `log`, `miller` | n slots; per step a loop over the sorted offsets j <= k, which breaks at the first offset above k (in `log`, w_k by a binary search, revision 7rt2: revision 7rt1 scanned every offset at every step) | none | `alloc(n)` first; per step `17 + sum of words` charged before its constant operations (`log`: plus `bits(offsets) + 1` for the search); every loop of a step is O(offsets below k), at most the charge |
| `pow_int` | binary powering, at most 2 x 7 `mul`; Miller; binomial: one iteration per slot, at most n + 1 (revision 7rt2), each a `mul`, a `pow_const`, a `binom` (m steps of `q_new` and `q_mul`), a `scale` and an `add`, in the caller's context and `Work` with the working precision set to the slots (revision 7rt1); both large-n paths return precision `slots`, never `Exact` (revision 7rt2); item 2 may store an exact term at i64::MAX, item 5 returns `Abs(n vb + 1)` checked, or `O(h^(n vb))` at n vb = i64::MAX (revision 7rt4) | calls itself once (n < 0) | the slots charged by `alloc`; every operation charged to the caller's `Work` |
| `pow_const`, `exp`, `log`, `sin_cos`, `atan_like`, `abs_sign` | O(n) plus the recurrences above; an empty truncated operand that does not meet the precondition returns `NeedPrecision` first (`require_nonneg_val`, `integrate`; revision 7rt3) | `atan_like` calls `pow_const`, `inv` and `exp` once each | charged as above |
| `expand` (`expand`, `expand_visit`, `EFrame::resume`; revision 7rt2) | one loop over a work stack: per node visit a series (a constant through `from_expr`, the variable), or one record pushed (16 bytes; a chain or a tower also a boxed record of at most 104 bytes, 112 as a chunk, since revision 7rt3 holds its node and a regular flag); a finished series is handed, with its flag, to the record on top | none (an explicit stack on the heap, at most one record per node of the current path) | `KONST_NODE` per visit charged first, and `RECORD_UNITS` per boxed record; at most D records |
| `maybe_hint`, `hint_exp_arg`, `subst_at`, `subst_or_node` (revision 7rt3), `seed_const`, `finish_flag` (revision 8) | O(1) tests from the cached sizes and the pre-pass; a seed renders one exact constant of at most twice its node's substitution estimate plus `HINT_MIN` nodes, once per node and expansion (charged `SUBST_NODE + expr_units`); for a hint, one substitution: a post-order walk with an explicit stack over the nodes that depend on the variable, one new node and one memo entry per distinct node not yet substituted | none | `SUBST_NODE = 96` per new node charged before it is built; the memo lives for the expansion, so all substitutions together build at most N nodes |
| chain records (`ESum`, `EProd`, `chain_items_e`; `/` in the product chain since revision 7rt1) | the spine as a loop; one item per summand or factor in three vectors; a constant divisor inverted as it is read, a series divisor through `inv` and `mul` (`chain_step`) | none | `KONST_NODE` per item charged |
| tower record (`ETower`, `tower_record`, `with_exponent`, `level_done`; revision 7rt1) | the right spine of a tower by a loop (through `Neg` nodes) into one vector of (base, exponent node, negations; 24 bytes, revision 7rt3); the top level `b_k^t`, then per level one `neg` per negation and the power by the rules of §2.5 | none | `KONST_NODE` per spine node charged first; each operation charged |
| `div_series`, `pow_general`, `pow_const_base`, `pow_const_base_k`, `zero_pow`, `pow_by_const`, `expand_unary`, `expand_const`, `expand_var`, `chain_step`, `finish_product` | the operations they call (`zero_pow`: one `split_imag`, one `real_sign` and at most one zero test) | none (a constant operand goes to `from_expr`, which has its own loop and never calls `expand`) | charged as above |
| pre-pass (`dep_pass`) | a post-order walk with an explicit stack; one map entry per distinct node (since revision 7rt3 with its tree size and the variable's occurrences, saturating) | none | looks a node up before it descends: N entries |
| driver (`expansion`) | at most 24 attempts; a stalled loss-driven step doubles the extra precision (revision 7rt3) | none | each attempt charged |
| rendering (`render_units`, `render_tree`, `to_expr`) | one pass over the stored terms (the first only for the limit path, revision 7rt3); per term at most 4 nodes beyond its coefficient's; one shared h | `balanced_arc`, `log2 n` levels | `render_tree` (O(n), cached sizes) checked against `MAX_RESULT_TREE` and `render_units` charged by the driver before the result is built; since revision 7 the pass is split between `series_expansion` and `SeriesExpansion::to_expr` (`series.rs`, below) |
| `same_expansion` (test helper) | two expansions | as `expand` | test only, unbounded `Work` |

`series.rs`, `parser.rs`, `simplify.rs`, `expression.rs` and the REPL arm:

| function | loops and allocations | recursion | bound |
|---|---|---|---|
| `series_expansion` (prototype `api::series_expansion`, revision 7) | the driver; then one pass over the stored terms: one `to_expr_arc_raw` per distinct coefficient (a map from the constant's address to its rendering, for the pass only; revision 7rt3) into one vector of exact capacity, one h; then the `Work` is dropped | none | the driver's charges, `render_units` among them; the pass and the map (one entry per term at most) are covered by it |
| `SeriesExpansion::to_expr` (`api::SeriesExpansion::to_expr`) | one pass over the terms: per term one `Arc` increment for the coefficient and one for h, at most 4 new nodes, one vector slot; one `balanced_arc`; one top-node copy when the result is a single shared coefficient | `balanced_arc`, `log2 n` levels | covered by `render_units` of the expansion (one rendering per call; a caller that calls it k times holds k results) |
| `coefficient`, `first_exponent` | a binary search, O(log n); one `Arc` increment (a new 0 when absent) | none | O(log n) |
| `limit_via_series` (`api::limit_via_series`) | one driver run with `LIMIT_SERIES_WORK` that charges, bounds and renders the first stored term only (`expansion_first_term`, revision 7rt3); one copy of a rational coefficient | none | the expansion's budget |
| `limit_with_error` (`api::limit_with_error`, revision 7rt4) | `limit_via_series`, then for a finite limit one driver run with `LIMIT_SERIES_WORK` that renders no term (`expansion_reaches`; revision 8 called `series_expansion`, which rendered every term) | none | each run's budget |
| `series_coefficients` | one `series_expansion` (no budget, as `taylor`); a dense vector of order + 1 entries, one top-node copy per stored coefficient | none | unbudgeted, as every entry point without a budget |
| `display_estimate` | an explicit stack | none | stops once the estimate passes 200 characters (about 34 nodes) |
| `Display`, `Debug` of `SeriesError` | a bounded text | none | O(200) |
| `Debug` of `SeriesExpansion` (revision 7) | the variable, the point's `display_estimate`, the precision and the stored exponents; no coefficient's tree | none | O(n + 200) |
| `mentions_symbol` (REPL arm) | an explicit stack; one visited entry per distinct node | none | linear in the DAG |
| parser (§2.12) | runs of signs and of `* /` and `+ -` by loops | one level per nested parenthesis, function call or `^` | the input's nesting, as HEAD; the only recursion over a text (fact 3, revision 7rt2); §6 step 8 measures its limits and checks `taylor_with_budget` at each (revision 7, review 7 m3; revision 7rt1) |
| `Display for Expr` and `Debug` (§2.11, revision 7rt1) | a work list of at most the depth plus the pending arguments, on the heap; one write per node and per token | none | linear in the text written |
| REPL `expr_to_latex` (§3.1, revision 7rt1) | a work list as Display, into one `String` | none | linear in the LaTeX written, at most about 3.4 times J1b's estimate of the Display text, which stops past `OUTPUT_BYTES` |
| `simplify` (§2.10) | the left spine by a loop | the right children | the input's depth, as HEAD |
| `Drop for Expr` (§2.14) | a heap list of `Arc`s; one `Arc::into_inner` per moved child (revision 7rt3); during the thread's TLS teardown one fresh leaf per moved child, freed in the same iteration (revision 8) | the glue, at most 16 KiB of stack | bounded for every shape and every sharing, a child shared within its node included (revision 7rt3; revision 7rt2 moved only uniquely owned children) |

Outside the engine, Display and `Debug` write a result with a work list (revision 7rt1), and the other
consumers of a result (the derived `PartialEq`, `substitute`, `simplify`) recurse once per level of it;
§2.15 bounds its depth and gives their measured limits. J1b's `display_size`
(`repl.rs:11569`, `:11620`) uses an explicit stack and stops past `OUTPUT_BYTES`.

Memory (revision 6, review 6 B1 to B3). Every kind of heap object the engine allocates, with its size
on x86-64 (`EXP/rev6/probes/sizes.rs`, `sizes2.rs`) as a glibc chunk (the request plus 8, rounded up
to 16, at least 32 bytes) and the units charged for it before it is allocated. Every row is at most 3
bytes per unit, so a call that is charged U units holds at most about 3 U bytes of engine data, plus
the input it was given and the pre-pass map. A port that adds a kind of object adds its row.

| object | bytes | units charged, by | bytes per unit |
|---|---|---|---|
| constant header (`Arc` of `ConstData`: terms enum 88, cached sizes 88) | 208 | `CONST_BASE = 72`, `from_sorted` | 2.9 |
| one-term constant made by `rat` (the header with its inline term, and two digit vectors) or by `kernel` (also the one-factor monomial) | 272, 304 | `CONST_BASE` plus `COPY_WORD w + 16` (94 for one-word parts), `CONST_BASE + 32` (104) | 2.9 |
| term slot (`(Monomial, Rational)`, in the boxed slice or the one-term header) | 88 | `COPY_TERM = 52` (with the 3 chunk overheads below), `copy_units` | together 2.9 for a rational term (152 bytes) |
| monomial vector (16 per factor) and its chunk overhead | 16 L + at most 16 | `COPY_FACTOR = 6` per factor (`copy_units`), `MONO_FACTOR_UNITS = 6` (a product, a power, an inverse); the overhead in `COPY_TERM` | 2.7 |
| digit vector of a numerator or denominator (8 per word) and its overhead | 8 w + at most 24 | `COPY_WORD = 3` per word; the overhead in `COPY_TERM`; results of the charged arithmetic by their operation, at least `OP_BASE` and, since revision 7, at least `COPY_WORD` per word (review 7, m4: `q_mul(1, L)` was charged about 2 units per result word, 4 bytes per unit) | 2.7 |
| accumulator slot of a product (a B-tree node holds 11 slots of 88 bytes, at least half full) | at most 190 | `ACC_ENTRY = 32` plus the pair's `q_mul` (at least 68) and `MONO_FACTOR_UNITS` per factor | at most 1.9 |
| `Expr` node of a rendering (`Arc<Expr>`, 80 bytes) | 96 | `EXPR_NODE = 32`, `expr_units` and `RENDER_TERM` | 3.0 |
| integer digits of an `Expr` node (a copied coefficient, an exponent) | 8 w + at most 24 | `EXPR_WORD = 3` per word, `EXPR_INT = 11` per integer | 2.9 |
| pointer vectors of a rendering (terms, factors, the balanced tree's work vector) | 16 per term, 8 per factor | inside `expr_units` (at least 2 nodes per term) | below 1 |
| term slot of `SeriesExpansion` (`(i64, Arc<Expr>)`) and its coefficient's `Arc` (revision 7) | 16, and the coefficient's nodes (none for a bare kernel: the kernel's own `Arc`) | the slot in `RENDER_TERM` (its `SER_TERM`); the nodes by `expr_units` | 2.7, 3.0 |
| per call, not per term: the `SeriesExpansion` header (`var`, a copy of the variable's name; `point`, a copy of the caller's point node), the point node again inside h, and the top node `to_expr()` copies when the result is one shared coefficient (revision 7) | at most four nodes of the caller's arguments or of the charged rendering | none: a constant number of copies per call of nodes that were charged or that the caller holds | |
| series term slot (`(i64, SeriesConst)`) | 16 | `SER_TERM = 6` | 2.7 |
| dense buffer slot of a recurrence or a product (8), and its compacted output (16, at most doubled while it grows) | 8 + at most 32 | 16 per slot, `alloc` (saturating, before the cast; at most 2^57 slots without a budget, revision 7rt4) | 2.5 |
| cursor of a merge: `sum_many` (its summand's slot, pointers and heap entry), `sum_series` (the series' slot, 40 bytes, and its cursor), the sparse product (a row's cursor) | 56, 64, 24 | `SER_TERM` plus the summand's `copy_units` (at least 58); 24 per series; 16 per row | below 1, 2.7, 1.5 |
| kernel (`KernelData` 96 in an `Arc`, 128 since revision 7rt3 adds `tree`; table slot of 32 bytes at most doubled, 64) and its expression (a clone of an input node in an `Arc`, 96; for an input divisor with c != 1 also the quotient and the rational node, 96 more and the digits) | 288, at most 416 | `KERNEL_UNITS = 128`, plus 16 for the expression (plus `2 EXPR_NODE` and the rational's units for the quotient) | 2.0 |
| payload of that clone (a symbol's or a function's name, 1 byte per character; a function's argument vector, 8 per argument; revision 7) | the payload, plus at most 24 | `node_payload_units`: `COPY_WORD` per 8 bytes | 2.7 |
| kernel key (one byte per byte, exact capacity) | 1 per byte | 1 per byte | 1.0 |
| chain item of `from_expr` and `expand` (its spine slot, at most doubled while the spine is read, and its constant or series slot, exact capacity) | at most 88 | `KONST_NODE = 16` per item and 16 for the item's own node visit | 2.8 |
| work-stack record of `from_expr` and `expand` (revision 7rt2): its 16-byte slot, at most doubled while the stack grows, and for a chain, a tower, a call or a `%` node a boxed record (`KSum` 56, `KProd` 56, `KFunc` 64, `KTower` 56, `KMod` 16, and since revision 7rt3, with the node and the regular flag, `ESum` 96, `EProd` 104, `ETower` 88 bytes; `bin/sizes_rel`), at most 112 bytes as a chunk | 32 per pending node, plus at most 112 per boxed one | the node's visit (`KONST_NODE = 16`), and `RECORD_UNITS = 32` for a boxed record when it is pushed | 2.0 and 3.0 |
| substitution node of a regular node (revision 7rt3): the new `Arc<Expr>` (96) and its memo entry (an address and a node, tree and depth, about 40 bytes, at most doubled while the table grows) | at most 192 | `SUBST_NODE = 96` | 2.0 |
| hint table entry (revision 7rt3; the constant's address, a clone of the constant that keeps the address valid, the `RenderHint`: about 40 bytes, at most doubled while the table grows; cleared at every attempt and when the expansion returns) | at most 96 | `HINT_UNITS = 32` | 3.0 |
| tower level of `from_expr` and `expand` (its spine slot of 24 or 16 bytes, at most doubled while the spine is read, and its base's constant slot of 8; revision 7rt1) | at most 56 | `KONST_NODE = 16` per spine node, and the base's own node visit | 1.8 |
| work list of Display and of the REPL's LaTeX writer (24 bytes per pending item; revision 7rt1) | at most 24 times the result's depth plus its widest argument list, transient | none: outside the engine, at most a fraction of the result it writes, whose nodes were charged | |
| memo entry, stack entry and `seen` entry of a numeric enclosure (two MPFR values at p = 64, hash slots at their lowest load) | about 280 | `u + BALL_NODE` = 32 + 64 at p = 64, more at higher precision | 2.9 |
| seed of an exact-constant node (revision 8): its memo entry (as a substitution node's) and the rendering of its constant | at most 96, plus the rendering | `SUBST_NODE = 96` plus `expr_units` of the constant | 1.0, 3.0 |
| pre-pass map entry (one per distinct node of the input; since revision 7rt3 with a tree size and an occurrence count, 32 bytes) | at most 80 | none: less than the memory of the input node it describes (96 bytes), which the caller holds | |
| the variable's series, copied at each occurrence (`expand_var`: a vector of 1 or 2 terms, a 32- or 48-byte chunk; held by a pending chain record; red team 3 memory m4) | at most 48 | 8 units for the copy and the occurrence's node visit (16) | 2.0 |
| rendering map of `series_expansion` (one entry per term, for the pass only; revision 7rt3) | at most 48 per term, transient | inside `RENDER_TERM` (`SER_TERM`, 6, per term) and the term's other units | below 1 |
| prime table | 51 KB once per process | none | |

Measured (release, the revision 7 prototype built with a counting allocator, `EXP/rev7/proto8/memcount.rs`,
`--cfg memcount`: the heap peak of the whole call through a public entry point, in malloc chunks, so
independent of fragmentation; `EXP/rev7/fam7_final.txt`, `mem7_final.txt`, `calib7_final.txt`,
`b2probe_r7.txt`; rerun on the revision 7rt1 prototype in `EXP/rev7rt1/` and on the revision 7rt2
prototype in `EXP/rev7rt2/`, and on the revision 7rt3 prototype in `EXP/rev7rt3/`, whose largest are
2.59 bytes per unit at a refusal and 230 MB of VmHWM for an admitted call, §2.7; on the revision 7rt4
prototype the largest heap peak is 2.83 bytes per unit, red team 4's distinct long-monomial
coefficients, an admitted call, §2.7): at most 2.29 bytes per unit over the families of revision 7, the most
for `x + S_19*sin(x)` (2.29, 227 MB at 9.92e7 units) and building S (2.21: the product's transient
accumulator next to its result); at most 2.60 for long literals times series of unit coefficients
(`(L*x)/(1-x)^2` 2.60; `L*((1)/(1-x))` and `(L)/(1-x)` 2.56, 252 MB at 9.84e7 units; revision 7rt1,
§2.7); and at most 2.61 for the rendering of long monomials with exponents other than +-1 (red team 2,
§2.7). VmHWM, which also counts the process (about 4 MB), the parsed input (up to 9 MB) and memory
that glibc keeps after a free, is at most 261 MB on every probe with a budget, 265 MB for a whole REPL
line. These are measurements; the bound is the table's 3 bytes per unit. The counting allocator is measurement code only (an `unsafe impl
GlobalAlloc`); it is not part of the port.

### 2.17 expand under a minus (`expand.rs:250-263`, revision 6)

`distribute_negation` becomes `-(a + b) = (-a) + (-b)` and `-(a - b) = (-a) + b`, each `-a` again by
`distribute_negation` (so a sum chain of any shape has every summand negated), and any other node
`-u` as today:

```rust
fn distribute_negation(expr: &Expr) -> Expr {
    match expr {
        Expr::Binary(BinaryOp::Add, a, b) => distribute_negation(a) + distribute_negation(b),
        Expr::Binary(BinaryOp::Sub, a, b) => distribute_negation(a) + (**b).clone(),
        _ => -expr.clone(),
    }
}
```

The recursion follows the left operand and, for `+`, the right one, as HEAD's does (`expand` itself
recurses over the whole tree at HEAD, so this adds no depth); the `-` arm no longer descends into its
right operand. The probe of §1.7 gives `(-x + 1)`, `(-x + -1)`, `((-x + -y) + -1)`,
`((-x + y) + 1)` and, for the product, a sum of value 40/49: every value equals gp. The doc comment of
`expand` states the rule. Callers (`grep -rn '\.expand()'`): the REPL `expand` arm, `simplify_full`,
`rustmath-py/src/symbolic.rs:135` and `:291`, `rustmath-benchmarks/src/bench_symbolic.rs:136`; their
results under a minus change from wrong to right, and no test asserts the old forms
(`test_expand_nested` expands `(x + 1)^2`, `test_simplify_full` asserts nothing).

## 3. Callers and behaviour changes (whole workspace)

### 3.1 Series API

`grep -rn "\.taylor(\|\.maclaurin(\|\.series_coefficients(\|\.laurent(\|\.asymptotic(\|\.series_with_big_o(\|\.series(&" --include=*.rs`:

| site | change |
|---|---|
| `rustmath-jupyter/src/repl.rs:2143` (REPL `taylor`; arm `:2118-2145` at HEAD 92906d4) | `taylor_with_budget(&var, &args[2], order, repl_limits::TAYLOR_WORK)`; map errors: `DivisionByZero`, `InvalidPoint`, `Pole`, `ValuationBelowMinimum` to "ValueError" (properties of the input); `NotLaurent`, `Unsupported`, `UndecidedZero`, `PrecisionExhausted`, `BudgetExceeded`, `LimitExceeded` to "ComputationError" (the engine could not expand; review 1, m1); message `taylor: {e}`. The variable (revision 5, review 5 m1): `let var = Symbol::new(var_name)` (symbols are interned by name, `symbol.rs:40`, so it equals the expression's symbol) instead of `args[0].symbols()` (`:2127`, a recursive tree walk, exponential on a REPL DAG such as `e = e + e`); the "Variable 'x' not found in expression" error stays, decided by the new DAG-safe `rustmath_symbolic::mentions_symbol(&args[0], &var)` (an explicit stack and a visited set of node addresses, linear in the DAG; the engine's own point test uses its pre-pass, §2.5). The `match args[3].simplify()` at `:2135` becomes `match &args[3].simplify()` (an E0509 site, §3.3) |
| `rustmath-jupyter/src/repl.rs:11078-11087` (the `TAYLOR_ORDER` doc comment and constant at HEAD 92906d4, unchanged at f29c492; J1b's comment says that the limit waits for S1) | `TAYLOR_ORDER` stays 20000 (revision 7, review 7 m1: revision 6 lowered it to 2000, an argument check with no cost behind it that would refuse calls HEAD handles, such as `taylor(x + 1, x, 0, 20000)`, which HEAD returns in 0.48 s with 324 MB and the engine in 16 us with 860 units); new `TAYLOR_WORK` (§5.3, provisional 100_000_000); doc comments rewritten with the new measurements, and they say what each constant bounds (review 2, m5). `TAYLOR_ORDER`: "An argument check: a clear refusal of absurd orders before any work, HEAD's value. It bounds neither the work nor the output: `TAYLOR_WORK` refuses `exp(x)` at order 20000 after 0.14 s, while `x + 1` costs 860 units there; `(1+x)^(10^100)` at order 1150 is admitted and prints 64.7 MB." (The revision 7 prototype charges that call 9.55e7 units; at order 1200 it is refused at 1.04e8, so the comment uses order 1150; the implementer checks the numbers with the final code.) `TAYLOR_WORK`: "Bounds the work of one call and its memory: every big-number operation is charged by a bound of its cost, and every constant, term, monomial, expression node, kernel, kernel key, series term and buffer slot by its size, before it is made, at most 3 bytes per unit; copies of coefficients and of kernels' expressions are shared. So 1e8 units hold at most about 300 MB plus the parsed input. Measured: a heap peak of at most 2.83 bytes per unit, and at most 230 MB of resident memory at 1e8 units on every probe of rustmath-symbolic §5.3. Hard caps bound the length of a monomial, the number of kernels, the size of a constant, the depth of a kernel expression and the result's size as a tree (2^21 nodes, so that `simplify`, `subs` and `==` of a result stay within about 200 MB; §2.15). The result is most of a call's memory, and it outlives the call: a line or a cell that keeps several results holds each (rustmath-symbolic §7 item 9). It does not bound the printed text: a kernel is stored once and printed in every coefficient that contains it, so `(f(...f(1)...))/(1-x)` with 33 000 calls at order 60 prints 6.0 MB of text and 20 MB of LaTeX. The display budget (`OUTPUT_BYTES`) bounds the text, and the time to print it, because Display and `expr_to_latex` write linearly into one buffer." (review 4, m6; reviews 5 and 6; red team 1.) Its doc comment also records the §5.3 table (slowest rate 3.5e7 units/s in revision 7, the margin kept). For plain series the order still sets the size: exp(x) prints 5.3 MB at 2000 and 38 MB at 5000; at 20000 the budget refuses it after 0.14 s (it would print 730 MB, §5). J1b left `TAYLOR_ORDER` to S1 (its doc comment says so); S1 keeps 20000 and rewrites the comment |
| `rustmath-jupyter/src/repl.rs:10883-10935` (`expr_to_latex`, called by `to_display` at `:383` for every `Expr` value; revision 7rt1, red team 1 memory H1) | rewritten as a work-list writer into one `String`, with HEAD's output byte for byte: each node pushes its prefix, then its children and separators on an explicit list, and its suffix (no `format!` of a child's finished string, no `join`, no recursion), as `latex_iter` in `EXP/rev7rt1/proto9/display.rs`. HEAD formats each node from its children's strings, so it copies every subtree's text once per ancestor: a taylor line whose 99 KB input shares a depth-33 000 kernel among 101 terms (`(f(f(...f(1)...)))/(1-x)` at order 100) spent 32 s in it, after 62 ms in the engine; the work-list writer takes 0.13 s (the whole line 0.44 s, VmHWM 139 MB). Since revision 7rt3 that result (a tree of 3.3e6 nodes) is `LimitExceeded(TreeSize)`; at order 60 (a tree of 2.0e6) the line takes 0.40 s: the estimate 10.1 MB, the text 6.0 MB in 0.09 s, the LaTeX 20.1 MB in 0.09 s, VmHWM 120 MB (`EXP/rev7rt3/bin/replshow_rel`). With `display_size` (which stops past `OUTPUT_BYTES`) and the LaTeX of a node at most about 3.4 times its estimate, the display budget bounds the time to print. The doc comment says so |
| `rustmath-symbolic/src/limits.rs:430`, `:554` | §2.13 |
| `rustmath-symbolic/src/lib.rs:291-311`, `:366-368`, `:433-465` | tests: `.unwrap()` the Results; assertions unchanged (`laurent(1/x, .., -1, 2)` is `1/x`, not constant; `asymptotic(x^2)` is `x^2`) |
| `rustmath-symbolic/src/series.rs:813-1003` (tests; the module is `:790-1089` at HEAD, with I3's `taylor_at_a_huge_point` at `:795` and the Fourier tests after `:1003`) | rewritten against gp values (§4); the Fourier tests are untouched |
| `rustmath-calculus/src/taylor.rs` | none (own implementation, see §7) |
| `rustmath-py` (outside the workspace) | does not call these functions |

### 3.2 Parser and Display

Parser change (`parse`) callers: `rustmath-jupyter/src/repl.rs:14` (`parse_expr` for every symbolic
REPL argument, also `:2018`, `:5086`, `:5116` at HEAD 92906d4), `rustmath-py/src/symbolic.rs:6`, and the crate's own
tests. REPL behaviour changes: `x^2/4`, `2^3/4`, `2/3^2` now mean what they say; `1/x`, `1/sin(x)`,
`exp(1/x)` now parse; huge integer literals parse.
`grep -nE '\^[0-9]+/[0-9]|[^a-z0-9_.][0-9]+/[a-z(]|\^-[0-9]' repl.rs` finds no REPL test using these
forms.

Display change: only `Pow` operands and `Div` right operands that are rationals, `Pow` bases that
are negated or negative, and the operand of a negation that is itself negated or negative (`-(-x)`,
revision 7). Display is a work-list writer (§2.11, revision 7rt1); it writes the same text, so callers
see no difference except that a deep expression no longer overflows the stack. A grep for `"--` in the string literals of the workspace's tests finds only the REPL's
numeric evaluator (`repl.rs:13733`, `("--5", "5")`, a value, not an `Expr` Display).

### 3.3 `Drop for Expr`: E0509 sites

Compiling the reverse-dependency chain of rustmath-symbolic (symbolic, calculus, functions,
geometry, manifolds, curves, plot3d, jupyter; libraries and their test targets) with the `Drop` of
§2.14 (`EXP/dropchain.py`, output `EXP/dropchain2.txt`) gives E0509 at exactly these sites. Each is
fixed by matching on `&expr` (and `(**inner).clone()` where an owned value is needed); the fixed
library versions compile (`EXP/dropcheck/src/integrate_exact.rs`, `EXP/dropcheck_geom/plane_curve_ops.rs`,
`EXP/dropcheck_jup/repl.rs`).

Library code (4 sites):

- `rustmath-symbolic/src/integrate_exact.rs:96` and `:148` (the second becomes `let next = match
  &cur { .. }; cur = next;`, returning `(c, cur)` in the non-rational arms);
- `rustmath-geometry/src/plane_curve_ops.rs:492` (`match &simplified`; `:489` at c79a582);
- `rustmath-jupyter/src/repl.rs:2135` at HEAD 92906d4 (`match &args[3].simplify()`, part of the taylor
  arm rewrite; `:1973` at dc367a2).

Test code (98 sites, all `#[cfg(test)]` modules; `match x {` / `if let Expr::.. = x` on an owned
result becomes `match &x {` with derefs adjusted):

| file | lines |
|---|---|
| `rustmath-symbolic/src/parser.rs` | 454 488 508 528 584 605 626 646 711 735 771 784 870 880 890 900 908 919 936 953 973 |
| `rustmath-symbolic/src/specialfunctions/orthogonal_polys.rs` | 629 662 669 676 683 690 697 710 722 734 |
| `rustmath-symbolic/src/specialfunctions/other.rs` | 466 532 (twice) 558 619 634 641 648 |
| `rustmath-symbolic/src/specialfunctions/special.rs` | 318 324 332 339 345 353 361 370 |
| `rustmath-symbolic/src/specialfunctions/airy.rs` | 300 306 315 321 330 338 394 |
| `rustmath-symbolic/src/specialfunctions/generalized.rs` | 271 279 297 338 400 442 |
| `rustmath-symbolic/src/specialfunctions/min_max.rs` | 252 272 276 299 319 323 |
| `rustmath-symbolic/src/specialfunctions/transcendental.rs` | 315 328 336 343 350 365 |
| `rustmath-symbolic/src/specialfunctions/wigner.rs` | 373 387 404 418 432 446 |
| `rustmath-symbolic/src/specialfunctions/jacobi.rs` | 185 192 199 206 213 |
| `rustmath-symbolic/src/specialfunctions/hypergeometric.rs` | 296 304 313 |
| `rustmath-symbolic/src/radical.rs` | 318 338 367 |
| `rustmath-symbolic/src/specialfunctions/prime_pi.rs` | 255 283 |
| `rustmath-symbolic/src/function.rs` 380, `walker.rs` 685, `specialfunctions/piecewise.rs` 94, `specialfunctions/spike_function.rs` 68 | one each |
| `rustmath-calculus/src/integration.rs` 427, `limits.rs` 309, `sum.rs` 232 | one each |

No other crate has a site: functions, manifolds, curves (library and `tests/genus2_jacobian.rs`),
plot3d and jupyter compile unchanged; rustmath-benchmarks and rustmath-py do not pattern-match on
`Expr`; rustmath-symbolic has no `tests/` directory today. The library sites above are at HEAD
92906d4 (rustmath-symbolic and rustmath-geometry are unchanged since dc367a2; the jupyter site
moved). The test lines in the table are at c79a582: in the files that I3 edited they have moved
(for example `orthogonal_polys.rs:629` is now `use rustmath_integers::Integer;`, review 4 m4). The
implementer recompiles (`cargo check --workspace --tests`), which finds every site at its current
line, including any site added by the units implemented before S1.

### 3.4 Coordination

MF1 (f29c492, before S1; HANDOFF's S1 bullet since 2e15d1c): every match on `Expr` in rustmath-manifolds
is on a reference, so the `Drop` of §2.14 compiles there, and MF1's `root_sum`, `abs`, `x^y` and
`d_budget` slot assertions are structural and depend on HEAD's `simplify` results. S1 keeps those
results: §2.10 changes how `simplify` walks a left spine, not what it returns, §2.17 changes `expand`
only, which rustmath-manifolds does not call (a grep for `.expand()` and `simplify_full` there is
empty), and §2.11 changes Display only, which no rustmath-manifolds test reads on an `Expr` with a
negated negation. The acceptance of §6 runs `cargo test -p rustmath-manifolds`.

I3 (before S1) changes `Integer::to_i64` to `Option` and adds the symbolic test for taylor at 10^30;
S1 adapts that test. J1a and J1b landed before S1 in this round (the order of the round moved S1
late): they left the REPL taylor arm, `TAYLOR_ORDER` and `test_taylor_order_checked` to S1 (J1b's
`TAYLOR_ORDER` comment says so), so S1 makes those changes at their HEAD 92906d4 lines (§3.1). Any unit implemented after S1
that adds a by-value match on an owned `Expr` gets E0509 at compile time, which says how to fix it.
J1b's display budget (`display_size`, `OUTPUT_BYTES`) is what bounds the printed text of a taylor
result: `TAYLOR_WORK` bounds the work and the memory, not the text (§2.7, review 2 m5). It bounds the
time to print only if the writers are linear in their text: Display is (§2.11), and S1 rewrites the
REPL's `expr_to_latex`, which was O(text x depth), as a work-list writer (§3.1; red team 1, memory H1). Its estimate
must count a shared subtree once per occurrence (as Display prints it), which J1b's DAG case already
requires; `(1+exp(...(1+exp(x))^2...))^2` with 10 levels at the point 1, order 1 (35 MB of text
from 2.4e6 units in revision 6; 1.6 MB since revision 7rt3, whose values at the point render from the
input, §2.5) is a good check for it. Engine results are DAGs whose trees can be larger than the DAG
(§2.2, review 4 B2): `exp((1+exp(N_30))^511 + x)` at 0, order 1 is computed in 52 ms with 4 MB,
and its text had more than 2^30 nodes before revision 7rt3. Since revision 7rt3 the result's tree is
capped at 2^21 nodes (§2.15), so the text is at most about 2^21 tokens and every other REPL arm that
walks a result (`simplify`, `subs`, `expand`, `diff`) visits at most that many nodes; the display
budget remains the bound of the printed text, and the cost of the other arms is not only text (red
team 3 structure H2: `simplify` of revision 7rt2's `F_10` result took 960 MB). So the estimate must not be a plain walk of the tree: it
either stops once it passes `OUTPUT_BYTES`, as J1b's `display_size` does (J1b §3.2: "stopped once
bytes passes limit, so it costs O(limit) at most whatever the DAG"), or it memoizes the size per
shared node (by `Arc` address), which gives the exact tree size in DAG time. That input is a good
second check. J1b has landed: `display_size` (`repl.rs:11569`) and `expr_display_size` (`:11620`) use
an explicit stack and stop once the bound passes `OUTPUT_BYTES`, so such a result is shown as a
one-line summary, and the check above holds. Since revision 5, kernels made from input constants
print as the input does (§2.2), so a summary is needed only for kernels the engine makes from
computed values (the k = 10 nesting at a point of §5.3 printed 35 MB before revision 7rt3, 1.6 MB now). Error messages are S1's own concern: `Display for SeriesError` prints
an `UndecidedZero` constant only when its estimated text is at most 200 characters (§2.9, review 3
m6), so the REPL message `taylor: {e}` stays short whatever the constant, and J1b's
`OUTPUT_BYTES`, which covers values, is not needed for it.

## 4. Tests

Reference values come from PARI/GP 2.15.4 (`/usr/bin/gp`). Helper used for all of them
(`EXP/refs.gp`):

```
L(F, a, n) = my(s = F(a + x + O(x^(n + 20))), v = valuation(s, x)); [v, vector(n - v + 1, i, polcoef(s, v + i - 1, x))];
```

It returns `[valuation, coefficients up to exponent n]` (raise the `20` when the valuation is very
negative, as for `1/(sin(x)^50 - x^50)`, where gp needs `O(x^(n+120))`). Outputs are in
`EXP/refs.txt` and `EXP/gpcheck2.out`. A test helper `fn coeffs(f, a, n) -> SeriesExpansion` calls
`series_expansion` and compares `Expr` values exactly (rational coefficients compare as
`Expr::Integer`/`Expr::Rational`; parameter cases substitute a rational for y and `simplify`;
transcendental cases evaluate with `eval_float` after substituting `pi`, `e` with floats, tolerance
1e-12; coefficients too small for f64, as `sin(1)^4294967296`, are compared as structures).

The porting oracle is `EXP/rev2/gpcheck3.py` (107 cases: the 82 of `EXP/gpcheck2.py`, 9 for review 2
(m1 poles, m3 logs, a huge trig kernel) and review 2's 16 cases of negative intermediate precision
and branch cuts). It runs the prototype and compares every
coefficient with gp in an 80-digit gp session: exactly for rational, parameter and `I` cases, to a
relative 1e-60 for transcendental constants. It maps gp's missing `erf` to `ERF0(z) = 1 - erfc(z)`
and counts any gp error as a failure (review 1, m4); gp's own float artifacts (`sin(Pi)`,
`sqrt(2)*sqrt(2)-2`, `sqrt(8)-2*sqrt(2)`) and gp's refusal of a series exponent (`x^(x-x+2)`) use exact
aliases, as do `log(e^3) - 3` and `log(exp(2)) - 2`, which gp evaluates as floats. The harness
parenthesizes a negative base, which HEAD's Display prints without parentheses. The implementer ports
these cases as tests and can run the harness against a small driver binary of the real crate.
`EXP/rev2/gp/m1.gp` (output `m1.out`) gives the reference values of the m1 tests below with
`L(F, a, n, extra) = my(s = F(a + x + O(x^(n + extra))), v = valuation(s, x), w = vector(n - v + 1,
i, polcoef(s, v + i - 1, x))); [v, #select(t -> t != 0, w), w[1..3], w[#w]]` (valuation, number of
nonzero coefficients up to x^n, the first three, the last).

No unit test asserts a duration (review 1, m8). The budget tests below carry one loose bound,
"returns within 20 s", which guards against a hang only. It also catches the review 2 B2 regression
(revision 1 took 111.7 s on `budget_huge_trig_kernels`; the fix takes 2 ms), so review 2's "within 1
s" is met by the measurement and guarded by the 20 s bound without a timing-sensitive assertion.

### 4.1 Engine (new file `rustmath-symbolic/tests/series_engine.rs`, plus unit tests in the modules)

Each test checks values. Each fails on HEAD: HEAD returns `1/0` forms, fabricated zeros, panics,
runs out of memory, or (for the new entry points) has no such function.

| test | input | expected (gp) |
|---|---|---|
| `exp_at_0` | exp(x), 0, 12 | `L(x->exp(x),0,12)` = [0, [1, 1, 1/2, 1/6, ..., 1/479001600]] |
| `sin_at_0` | sin(x), 0, 15 | [1, [1, 0, -1/6, 0, 1/120, ..., -1/1307674368000]] |
| `tan_at_0` | tan(x), 0, 15 | [1, [1, 0, 1/3, 0, 2/15, 0, 17/315, 0, 62/2835, 0, 1382/155925, 0, 21844/6081075, 0, 929569/638512875]] |
| `tanh_sinh_cosh_at_0` | tanh 9, sinh 7, cosh 6 | [1,[1,0,-1/3,0,2/15,0,-17/315,0,62/2835]], [1,[1,0,1/6,0,1/120,0,1/5040]], [0,[1,0,1/2,0,1/24,0,1/720]] |
| `csc_cot_at_0` (poles) | 1/sin(x), 1/tan(x), 0, 9 | [-1, [1, 0, 1/6, 0, 7/360, 0, 31/15120, 0, 127/604800, 0, 73/3421440]]; [-1, [1, 0, -1/3, 0, -1/45, 0, -2/945, 0, -1/4725, 0, -2/93555]] |
| `bernoulli_gf` | 1/(exp(x)-1), 0, 8 | [-1, [1, -1/2, 1/12, 0, -1/720, 0, 1/30240, 0, -1/1209600, 0]] |
| `bernoulli_h49e3` (m11) | x/(exp(x)-1), 0, 50 | coefficient of x^k is `bernfrac(k)/k!` for k = 0..50: gp `s = x/(exp(x+O(x^60))-1); vector(51,k,polcoef(s,k-1)) == vector(51,k,bernfrac(k-1)/(k-1)!)` is 1 |
| `log_cos`, `exp_sin` | 0, 10 / 0, 8 | [2, [-1/2, 0, -1/12, 0, -1/45, 0, -17/2520, 0, -31/14175]]; [0, [1, 1, 1/2, 0, -1/8, -1/15, -1/240, 1/90, 31/5760]] |
| `removable_singularity` | (sin(x)-x)/x^3, 0, 6; sin(x)/x, 0, 10 | [0, [-1/6, 0, 1/120, 0, -1/5040, 0, 1/362880]]; sin(x)/x from `L(x->sin(x)/x,0,10)` |
| `triple_pole_needs_retry` | 1/(sin(x)-x), 0, 3 | [-3, [-6, 0, -3/10, 0, -11/1400, 0, -17/126000]] |
| `high_valuation_denominator` (m2) | 1/(sin(x)^50 - x^50), 0, 3 | valuation -52, 28 nonzero terms (exponents -52, -50, ..., 2), first ones -3/25, -62/125, -9113/13125; all from gp `F(x)=1/(sin(x)^50-x^50); s=F(x+O(x^124)); vector(56,i,polcoef(s,-53+i))` (exact rationals) |
| `exact_denominator_no_search` | 1/(x^300*(1+x)), 0, 3; 1/(x^3*(1+x)), 0, 3 | [-300, [1, -1, 1, ..., ]] (304 terms, alternating +-1); [-3, [1, -1, 1, -1, 1, -1, 1]] |
| `binomial_series` | sqrt(1+x) 6; (8+x)^(1/3) 4; (1+x)^(-5) 8 | [0,[1,1/2,-1/8,1/16,-5/128,7/256,-21/1024]]; [0,[2,1/12,-1/288,5/20736,-5/248832]]; `L(x->(1+x)^-5,0,8)` |
| `large_integer_powers` | (1+x)^10000, 0, 30; (1+x+x^2)^100, 0, 20; (2+x)^(-70), 0, 10 | `L(x->(1+x)^10000,0,30)` (= binomial(10000, k)); `L(x->(1+x+x^2)^100,0,20)`; `L(x->(2+x)^(-70),0,10)` |
| `large_power_symbolic_lead` | (x+sin(1)+cos(1))^4096, 0, 2 (Miller); (sin(1)^2+cos(1)^2+x)^100, 0, 3 and (sin(1)^2+cos(1)^2-1+x)^100, 0, 3 (binomial, undecided lead) | numeric values of `L(x->(x+sin(1)+cos(1))^4096,0,2)`; `L(x->(1+x)^100,0,3)` = [0,[1,100,4950,161700]] numerically; all coefficients of the last are numerically 0 up to x^3 (it is x^100) |
| `inverse_trig_at_0` | atan, asin 9; asinh, atanh 9 (built with `Expr::arcsinh` etc.) | [1,[1,0,-1/3,0,1/5,0,-1/7,0,1/9]]; [1,[1,0,1/6,0,3/40,0,5/112,0,35/1152]]; [1,[1,0,-1/6,0,3/40,0,-5/112,0,35/1152]]; [1,[1,0,1/3,0,1/5,0,1/7,0,1/9]] |
| `acos_at_0_has_pi_over_2` | acos(x), 0, 3 | c0 = pi/2 exactly (`Expr` check), then [-1, 0, -1/6] |
| `x_pow_x_at_1` | x^x, 1, 5 | [0, [1, 1, 1, 1/2, 1/3, 1/12]] |
| `exact_constant_exponent` (m1) | x^(x-x+2), 0, 3 | `x^2` (gp refuses the series exponent; exact reference) |
| `zero_base` (m1) | 0^x, 1, 3 | exact 0 |
| `removable_at_1`, `pole_free_at_2` | (x-1)/(x^2-1) at 1, 4; 1/(1-x) at 2, 4 | [0,[1/2,-1/4,1/8,-1/16,1/32]]; [0,[-1,1,-1,1,-1]] |
| `generic_parameter` | 1/(y+x), 0, 4; x^y at 1, 3; exp(y*x)/(1-x) 0, 5 | [0,[1/y,-1/y^2,1/y^3,-1/y^4,1/y^5]]; [0,[1,y,1/2*y^2-1/2*y,1/6*y^3-1/2*y^2+1/3*y]]; `L(x->exp(y*x)/(1-x),0,5)` (compare after y = 3/7) |
| `imaginary_unit` | exp(I*x), 0, 6; 1/(x^2+1) at I, 3 | [0,[1,I,-1/2,-1/6*I,1/24,1/120*I,-1/720]]; [-1,[-1/2*I,1/4,1/8*I,-1/16,-1/32*I]] |
| `complex_constant_decided` (B1) | 1/(x+sin(1)+I), 0, 2; 1/(x+pi+I), 0, 4 | `L(x->1/(x+sin(1)+I),0,2)` = [0, [0.49264333476860154590 - 0.58545492793321861568*I, 0.10006001734936179013 + 0.57684093610746422169*I, -0.38700826930171954901 - 0.22559621214875578048*I]]; `L(x->1/(x+Pi+I),0,4)` |
| `exact_pi_points` | 1/sin(x) at pi, 5; tan(x) at pi/2, 5; sin(x) at pi/4, 2 | `L(x->-1/sin(x),0,5)` = [-1,[-1,0,-1/6,0,-7/360,0,-31/15120]]; `L(x->-1/tan(x),0,5)` = [-1,[-1,0,1/3,0,1/45,0,2/945]]; c = [sqrt(2)/2, sqrt(2)/2, -sqrt(2)/4] exactly (gp numeric `L(x->sin(x),Pi/4,2)`). (gp's own floating Pi gives 5.3e38 here, which is why exact pi matters) |
| `radical_reduction` (m10) | 1/(x+sqrt(8)-2*sqrt(2)), 0, 2; zero_status of sqrt(8) - 2*sqrt(2) | `1/x`; `Zero` |
| `asymptotic_rational` | (x^2+x)/(x^2+1) at oo, 5 | `subst((x^2+x)/(x^2+1),x,1/x)+O(x^6)` = 1 + x - x^2 - x^3 + x^4 + x^5: `1 + 1/x - 1/x^2 - 1/x^3 + 1/x^4 + 1/x^5` |
| `asymptotic_polynomial` | x^2 at oo, 3 | `x^2` (valuation -2) |
| `transcendental_coefficients` | exp(x) at 1, 4; 1/(2+sin(x)) at 1, 3; acosh(x) at 2, 3; erf(x) at 0, 7 | `L(x->exp(x),1,4)`; [0.35193039286572446133, -0.066919142861945019958, 0.064834789907861362567, -0.011083758395192097310]; [1.3169578969248167086, 0.57735026918962576451, -0.19245008972987525484, 0.096225044864937627418]; `2/sqrt(Pi)*intformal(exp(-x^2+O(x^9)))` = (2/sqrt(pi)) (x - x^3/3 + x^5/10 - x^7/42) |
| `big_point_s7` | x^2 at 10^30, 3 | [0, [10^60, 2*10^30, 1]] (gp `L(x->x^2,10^30,3)`); at c79a582 `taylor` panicked; at HEAD dc367a2 (after I3) the old `taylor` returns it, and the test fails only because `series_expansion` does not exist yet |
| `symbolic_point` | exp(x) at a (symbol), 2 | `exp(a) + exp(a)*(x - a) + 1/2*exp(a)*(x - a)^2` (structure check) |
| `huge_exponents_b2` (debug and release) | (sin(1)*x)^4294967296 at 1, 1; (sin(1)*x)^3000000000 at 1, 1; x + ((sin(1)^4096)^4096)^4096 at 0, 1 | structures `[sin(1)^4294967296, 4294967296*sin(1)^4294967296]`, `[sin(1)^3000000000, 3000000000*sin(1)^3000000000]`, `[sin(1)^68719476736, 1]` (gp agrees numerically: `exponent(sin(1)^4294967296)` = -1069509430, `gpcheck2`); HEAD returns them unevaluated, revision 0 returned `1 + 4294967296*(x - 1)`, `1/sin(1)^1294967296 + ..`, `1 + x` |
| `exponent_overflow` | (sin(1)^4*x)^(2^62) at 1, 1; x + sin(1)^(2^62)*sin(1)^(2^62) at 0, 1; (x^(-2^62))^4 at 0, 1 | the first is `Ok` with the kernel `(sin(1)^4)^4611686018427387904` (the exponent product would overflow); the second is `Unsupported` (kernel exponent in a monomial product), the third `Unsupported` (series exponent); no panic in debug |
| `sparse_exact` (B3b) | x^(-1000000000)+1 at 0, 1 | terms `[(-1000000000, 1), (0, 1)]` (HEAD: garbage; revision 0: 24 GB) |
| `high_order_poles` (review 2, m1; review 3: one test, no longer release-only) | 1/(x+x^2)^200 at 0, order 2; 1/sin(x)^140 at 0, order 0; 1/sin(x)^200, 1/(exp(x)-1)^150, 2/(3*sin(x)^200), 1/(sin(x)^200+x^300), all at 0, order 2 | `m1.out`: [-200, 203, [1, -200, 20100], 203359206710512721649984375195710539823858829997596852086963562829147609878495814974301634417563537134018932503818612000] (the coefficient of x^k is (-1)^(k+200) binomial(k+399, 199)); [-140, 71, [1, 0, 70/3], last as printed by `m1.out`]; [-200, 102, [1, 0, 100/3], ..]; [-150, 153, [1, -75, 11225/4], ..]; [-200, 102, [2/3, 0, 200/9], ..]; [-200, 102, [1, 0, 100/3], ..], last coefficients as printed; every coefficient compared with gp (`gpcheck3`). Revision 1: `PrecisionExhausted(131)` and `PrecisionExhausted(129)`. Debug at HEAD (revision 3 prototype): 0.11 s, 0.32 s, 1.1 s, 2.1 s, 0.07 s, 1.1 s (at c79a582 the four order-2 cases took 15 to 30 s in debug, hence the former `high_order_poles_release`) |
| `exact_logs` (review 2, m3) | 1/(x+log(e^3)-3), 1/(x+log(exp(2))-2), 1/(x+log(2*e)-1), x+log(e), all at 0, order 2 | terms `[(-1, 1)]` (exactly 1/x) twice; the series of 1/(x + log(2)): c_k = (-1)^k log(2)^(-k-1), gp `L(x->1/(x+log(2)),0,2)`; `[(0, 1), (1, 1)]`. Revision 1: `UndecidedZero(-3 + log(e^3))` |
| `cap_paths_unbounded` (review 4, m1; `taylor`, no budget, plus "returns within 20 s") | x + ((2+sin(1))^(-1099511627776))^(-1), x + ((2+sin(1))^(-5000))^(-1), x + (2*3^(-300000))^(4095/2), all at 0, order 1 | terms `[(0, (2 + sin(1))^1099511627776), (1, 1)]`, `[(0, (2 + sin(1))^5000), (1, 1)]` and `[(0, c^(4095/2)), (1, 1)]` with c = `2/3^300000` (structures: the kernels of the caps). Revision 3: over 60 s, 15.9 s (1.65e10 units) and over 60 s release; revision 4: 0.1 ms, 0.1 ms and 4.5 ms |

Errors (each a `matches!` on the variant):

| test | input | expected |
|---|---|---|
| `log_singularity` | log(x) at 0 | `NotLaurent` |
| `branch_point` | sqrt(x) at 0, x^(1/2), sqrt(x^2) at 0, asin(x) at 1, x^pi at 0, x^y at 0, x^(1/2+I) at 0 | `NotLaurent` |
| `subexpression_not_laurent` | sqrt(x)^2 at 0, exp(log(x)) at 0 | `NotLaurent` (documented: a subexpression) |
| `essential_singularity` | exp(1/x) at 0, atan(1/x) at 0 | `NotLaurent`, message contains "of an argument with a pole" |
| `abs_at_zero` | abs(x) at 0 (`abs(x-2)` at 0 is `2 - x`) | `NotLaurent`, message contains "zero of odd order" |
| `unsupported_functions` | gamma(x) at 1, zeta(x) at 2, bessel_j(0, x) at 0 | `Unsupported` (HEAD: `gamma(1)`, `zeta(2)`, 0) |
| `identically_zero_denominator` | 1/(x - x) | `DivisionByZero` |
| `undecidable_zero_denominator` | 1/(sin(x)^2 + cos(x)^2 - 1) at 0, order 2 | `PrecisionExhausted` (bounded search; 0.15 s debug at HEAD) |
| `undecided_constant` | 1/(x + sin(1)^2 + cos(1)^2 - 1) at 0 | `UndecidedZero(c)` with c the canonical constant |
| `complex_constant_undecided` (B1) | 1/(x + sqrt(-4) - 2*I), 1/(x + exp(I*pi/2) - I), 1/(x + sqrt(-1) - I), 1/(x + log(-1) - I*pi), all at 0 | `UndecidedZero` (all four are 0; revision 0's text rule would have called the first two nonzero) |
| `undecided_exponent` (m1) | x^(sin(1)^2 + cos(1)^2) at 0 | `UndecidedZero` |
| `undecided_constant_base` (m5) | (sin(1)^2 + cos(1)^2 - 1)^x at 1 | `UndecidedZero` |
| `numeric_zero_test_decides` | 1/(2 + sin(x)) at 1 | `Ok` (ball test) |
| `undecided_after_underflow` (review 2, B1) | 1/(x + (1/2)^1073741827*2^1073741822 - 1/32) at 0, order 1; 1/(x + exp(-744261120)*exp(744261117) - exp(-3)) at 0, order 2; 1/(x + exp(744261120)*exp(-744261117) - exp(3)) (overflow); 1/(x + (1/3)^536870912*3^536870912 - 1); 1/(x + 2^(-300000000)*2^300000000 - 1) | `UndecidedZero` for each (all five constants are exactly 0, so the function is 1/x; gp at 77 digits gives the second constant as 5.4e-79, review 2 `underflow.gp`). Revision 1 returned a finite series of 1/0 terms for the first two (the other three were already `UndecidedZero` there, through its finiteness checks and radius growth; they guard the new checks). 2 to 13 ms debug |
| `non_integer_huge_midpoint` | x^(2^100000000*sin(1)) at 0, order 2 | `UndecidedZero(2^100000000*sin(1))` (no nearest integer at any precision; 2 ms debug) |
| `log_of_negative_not_folded` (m3) | 1/(x + log(-e) - 1) at 0 | `UndecidedZero` (log(-e) = 1 + I pi is not folded; the constant has no enclosure) |
| `point_contains_variable` | point = x + 1 | `InvalidPoint` |
| `laurent_min_order` | laurent(1/x^2, x, 0, -1, 2) | `ValuationBelowMinimum { valuation: -2, min_order: -1 }` |
| `series_coefficients_pole` | series_coefficients(1/x, x, 0, 3) | `Pole { valuation: -1 }` |
| `first_exponent_lower_bound` (review 3, m3) | (sin(1)^2+cos(1)^2-1)/x + 1 at 0, order 1 | `Ok`; `first_exponent()` is `Some(-1)` with the undecided coefficient `-1 + cos(1)^2 + sin(1)^2` at -1 (the valuation is 0); `limit_via_series` of it at 0 is `Unknown` |
| `undecided_error_text_bounded` (review 3, m6) | 1/(x + (sin(1)^2+cos(1)^2-1)*(3^200002*sin(2) + 3^200003*sin(3) + ... + 3^200021*sin(21))) at 0 (404 characters); 1/(x + sin(1)^2 + cos(1)^2 - 1) at 0 | both `UndecidedZero`; `format!("{e}")` of the first is under 300 characters and contains "about" (the constant prints as about 5.7 MB); the second prints the constant `-1 + cos(1)^2 + sin(1)^2` |

Budget (each with `taylor_with_budget`, debug, outcome asserted, plus "returns within 20 s"). The last
column gives the revision 2 prototype at c79a582 unless it says HEAD (the revision 3 prototype at
dc367a2, `EXP/rev3/budget_debug.txt`) or rev 4 or rev 5. Every row was rerun on the revision 4
prototype (`EXP/rev4/tools/budget4.py`, output `EXP/rev4/budget4_debug.txt`) and on the revision 5
prototype (`EXP/rev5/budget4_debug_r5.txt`) and on the revision 6 prototype with
`EXP/rev6/tools/budget6.py` (`budget4.py` with the new high budget of `budget_kernel_keys`; output
`EXP/rev6/budget6_debug_r6.txt`): all 31 inputs give the expected outcome, the slowest in 5.1 s debug
(the zero-multiplied sum of `budget_rational_normalization`). The rows added by revision 5 were run
with `EXP/rev5/tools/newtests_dbg.py` and `dag_input.rs` (`EXP/rev5/newtests_dbg.txt`,
`dag_final.txt`, `enginedepth_dbg_final.txt`; rerun on revision 6 in `EXP/rev6/newtests_dbg.txt` and
`dag_final.txt`), the rows added by revision 6 with `EXP/rev6/tools/newtests6_dbg.py`
(`EXP/rev6/newtests6_dbg.txt`):

| test | input, budget | expected | prototype, debug |
|---|---|---|---|
| `budget_constant_powers` (B3a) | x+(sin(1)+cos(1)+exp(1))^50, x+(sin(1)+cos(1)+exp(1))^100, x+(sin(1)+cos(1))^4096 at 0, order 1, budget 1e6 | `Ok`, the power kept as a kernel (`gpcheck2` checks the values) | 2 ms each (revision 0: 13.8 s, over 120 s, over 30 s) |
| `budget_sparse` (B3b) | x^(-1000000000)+1 at 0, order 1, budget 1e6 | `Ok(1/x^1000000000 + 1)` | 1 ms, 4 MB |
| `budget_multinomial` | (x+sin(1)+cos(1)+exp(1)+pi)^64 at 0, order 1, budget 1e6 | `BudgetExceeded` | 19 ms at c79a582; 12 ms at HEAD (refused at 4.0e6) |
| `budget_product_precharge` | exp(x)*sin(x) at 0, order 2000, budget 1e8 | `BudgetExceeded` (the product alone is charged 5.2e9) | 0.18 s; 0.21 s at HEAD |
| `budget_recurrence` | exp(sin(x)) at 0, order 300, budget 1e6 | `BudgetExceeded` | 0.9 s; 0.08 s at HEAD |
| `budget_huge_trig_kernels` (review 2, B2) | at 0, order 2, budget 1e8: 1/(x + sin(2^100000000) + 2); 1/(x + cos(2^100000000) + 2); 1/(x + tan(2^100000000) + 2); 1/(x + sin(2^1000000000) + 2); x + (sin(2^100000000) + 3)^1000000000 (order 1, the `pow_const` kernel fallback) | `Ok` with c0 = `1/(2 + sin(2^100000000))` (structure); `Ok`; `UndecidedZero`; `UndecidedZero` (the kernel's ball leaves the window); `Ok` with the kernel `(3 + sin(2^100000000))^1000000000` | 2 ms each (revision 1: 111.7 s release for the first, over 30 s for the fourth) |
| `budget_kernel_keys` (review 2, m2; review 4, B2) | x + sin(2^340000+1) + ... + sin(2^340000+40) at 0, order 1, budget 2.1e7, then budget 2.4e7 | `BudgetExceeded` with 2.1e7 (the revision 6 prototype charges 2.322e7 in all, of which 3.40e6 for the keys at one unit per byte; the kernel expressions are the input's nodes, 16 units each; without the key charge 1.98e7 would pass); `Ok` with 2.4e7 (revision 5: 2.3e7; revision 6 charges each use of a kernel its constant, `CONST_BASE + 32`). With the final model, put the low budget between the total without the key charge and the total | rev 6: 0.09 s and 0.09 s debug, 17 ms release (revision 1: 504 ms) |
| `budget_rational_normalization` (review 3, B1) | at 0, budget 1e8: x + (3/5)^200000, order 1; x + (3^330000)/(5^200000)*2*3, order 1; x + (3^330000)/(5^200000)*2*3*5*7*11*13*17*19*23*29, order 1; x+(1+x/3)^(1/7)*0+(1+x/5)^(1/7)*0+(1+x/11)^(1/9)*0+(1+x/13)^(1/3)*0, order 2000 | `Ok` (`x + 3^200000/5^200000`, exact); `BudgetExceeded`; `BudgetExceeded`; `BudgetExceeded` (each zero-multiplied power is still expanded to order 2000 before the product) | 1.3 s, 2.8 s, 2.5 s, 5.4 s debug; 0.14 s, 0.33 s, 0.34 s, 1.26 s release (revision 2 at HEAD: `Ok` in 0.14 s, 0.96 s, 3.9 s, 1.5 s release, charged 15 644, 5.9e7, 5.95e7 and 5.0e6 units; at c79a582 12.7 s, 75.7 s, about 275 s (review 3 estimate) and 14.3 s) |
| `budget_radicals` (review 3, B2) | at 0, order 1, budget 1e8: x + ((3^340000+1)^(1/2))^0 + ... + ((3^340000+100)^(1/2))^0 (2593 characters); the same with k = 3 .. 102 (radicands that pass the quadratic-residue sieve); x + (3^340000+1)^(1/2) | `BudgetExceeded`; `BudgetExceeded`; `Ok` (one root charged 7.1e7) | 0.13 s and 1.1 s debug, 7 ms and 68 ms release (revision 2 at HEAD: `Ok` after 10.5 s release for 4.1e6 units; at c79a582 14.3 s) |
| `budget_radical_fold` (review 3, m2) | x + (sqrt(3)^524288)^0 + ... (200 copies) at 0, order 1, budget 1e8 | `BudgetExceeded` (each fold charges `q_pow` of a 2^20-bit power, 6.9e5 units) | 4.3 s debug, 0.4 s release (revision 2: 5 397 units, 0.54 s) |
| `budget_transcendental_point` (review 3, m1) | 1/(2+sin(x)) at 1, order 30, budget 1e8 | `Ok` (4.1e6 units in revision 4, 2.1e6 in revision 3; revision 2 charged 8.5e7 here and refused order 50) | 0.1 s debug |
| `budget_coefficient_copies` (review 4, B1) | at order 1, budget 1e8, with K = `(1+pi)^450*(1+e)^450` (revision 6: a 203 401-term constant charged 6.7e7; with the exponents 511 of revisions 4 and 5 the revision 6 model charges 9.2e7, too close to the budget for a test): `(K + x + x + ... + x)*0` with 400 `+x` at 0; the same with 160 `-x` at 0; `x^0 + ... + x^0` (100 terms) at the point K | `Ok` (exact 0) for the first two; `Ok` (exactly 100) for the third. Each copy of K is O(1); the units are those of building K. Control: `K + x + y + ... + y` with 160 `+y` is `BudgetExceeded` (1.55e8) | rev 6: 1.41 s, 1.46 s, 1.40 s debug, 2 MB thread; 0.29 s, 0.28 s, 0.24 s release with 93 MB (rev 4, with exponents 511: 2.1 to 2.2 s, 2.1 to 2.4 s, 4.3 to 4.9 s debug; revision 3: 33.1 s, 13.2 s, 8.4 s release) |
| `budget_long_monomials` (review 5, B1) | debug, on a 2 MB thread, at 0, order 1, budget 1e8, P_N the balanced product of `sin(1)` to `sin(N)` (48 KB for N = 4096): `x + P_4096`; `1/(x + P_4096 - 1/2)` | `Ok` for both; the result's tree depth at most 64 (measured 16 and 21); `format!("{}")` of both succeeds (56 KB and 113 KB) | rev 5: 0.10 s and 0.15 s debug (revision 4: a stack overflow abort from N = 1975 and from N = 431) |
| `budget_monomial_products` (review 5, B2; revision 6, review 6 m2) | at 0, order 1, budget 1e8, Q_1400 the balanced product of 1400 sines, S_15 = `(1+cos(1))*...*(1+cos(15))`, each input built with both splits of a balanced product (left half `len//2` and `(lo+hi)//2`, the generators of `EXP/rev6/tools/m2probe.py`): `x + Q_1400*(S_15)*0`; `x + Q_1400*S_15*0`; `x + Q_1400*(S_15)` | `Ok` (exactly `x`) for the first two with both splits: a zero factor of a chain gives 0 before any round of `product_many`, so the outcome no longer depends on the split (revision 5: `BudgetExceeded` with one split, `Ok` with the other, review 6 m2); `BudgetExceeded` for the third with both splits (charged 1.27e9 and 2.39e9 at once: the monomial work of 2^15 products of 1415 factors) | rev 6: 0.20 s, 0.03 s and 0.20 s debug with either split (revision 4: `x + Q_1400*S_15*0` `Ok` after 13.3 s release with 1.09 GB, review 5) |
| `budget_sparse_products` (review 6, B1) | at 0, order 1, budget 1e8, `fz(d, m)` = `(A*B*D*E)*0` with A, B, D, E sparse polynomials in negative powers of x of 64, 64, d and m terms (exponents multiples of 3, so every exponent gets one pair; the generator of `EXP/rev6/tools/gen6.py`): `fz(64, 186)` (4.6 KB); `fz(64, 8)` | `BudgetExceeded` (the heap merge of 48.8 million pairs is charged first); `Ok` (exact 0) | rev 6: 0.76 s and 0.97 s debug; release 0.08 s with 7.8 MB and 0.11 s with 40 MB (revision 5: the first `Ok` after 15.5 s with 2.53 GB) |
| `budget_small_constants` (review 6, B2) | at 0, order 1, budget 1e8, F = `fz` without the `*0` (2^20 terms with coefficient 1 for d = 64, m = 4): `(F*sin(1) + F*cos(1))*0` (a 2-term coefficient per exponent, `two(64, 4)`); `(-(F*sin(1)))*0` with m = 8 (`negz(64, 8)`); `kterm(8, 1, 16)` (a 16-term coefficient per exponent); `negz(32, 1)` | `BudgetExceeded`; `BudgetExceeded`; `Ok` (exact 0); `Ok` (exact 0). Peak memory is checked by `EXP/rev6/tools/fam6.py` (heap peak at most 1.9 bytes per unit on the family), not by the test | rev 6: 1.46 s, 1.69 s, 1.03 s and 0.22 s debug; release 59 MB, 103 MB, 82 MB, 39 MB VmHWM (revision 5: the first two `Ok` with 1.32 GB and 561 MB) |
| `budget_rendering` (review 6, B3) | at 0, order 1, budget 1e8: `sparseg(60, 4)` = `(A*B*D*E)/x^K` with 983 040 stored terms of coefficient 1 (1.4 KB); `sparseg(30, 1)`; `(1 + Y*x + Z*x)^300` at order 120 with Y and Z symbols of 40 000 characters | `BudgetExceeded` (`render_units` is charged before the result is built); `Ok` (61 441 terms); `Ok` (each monomial of the result refers to the shared nodes of Y and Z) | rev 6: 0.83 s, 0.21 s and 0.07 s debug; release 34 MB, 67 MB and 11 MB VmHWM (revision 5: the first `Ok` with 725 MB; the third 594 MB, and 3.67 GB at order 300) |
| `input_divisor_kernels` (review 6, m1) | `invpow(k)` = `x + (1)/((1 + ... (1)/((1 + sin(1))^511) ...)^511)` with k levels: k = 12 at 0, order 1, without a budget (`taylor`); k = 80 with budget 1e8 | `Ok`, and the result's tree depth is at most the input's depth plus 4 (measured: equal to it; revision 5 rendered each level's 512-term P0, 13 levels of kernel per 3 of input); `BudgetExceeded` | rev 6: 4.2 s (result depth 39, the input's; HEAD 44) and 1.4 s debug, 2 MB thread |
| `budget_literal_chain` (revision 6, found by the self-attack) | built with the `Expr` API: `x + L + L + ... + L` with 25 000 items that share one node L, a literal of 50 000 digits, at 0, order 1, budget 1e8 | `BudgetExceeded` (each conversion of L copies its 2596 words and is charged `KONST_NODE + COPY_WORD w`; peak memory, 137 MB, by `EXP/rev6/litchain_final.txt`) | rev 6: 0.09 s release (revision 5's charge of 1 unit per word: 519 MB) |
| `budget_dag_inputs` (review 5, m1) | built with the `Expr` API, k = 40, at 0, order 1, budget 1e8: `e = x; e = e + e`; `e = sin(1); e = e*e`, then `x + e`; `e = x*0; e = e*e`; `e = 1 + x; e = e*e`; `exp(x)` at the point `e` of the second; `e = x; e = e + x*e` (revision 6: a chain whose items are DAG products) | `BudgetExceeded` for each | rev 6: 0.56 to 1.45 s debug, 2 MB thread (rev 5: 1.0 to 3.3 s; revision 4: did not return) |
| `budget_pair_count` (revision 5) | `((1)/(1-x))^2/x^250000` at 0, order 1, budget 1e8 | `BudgetExceeded` | rev 5: 1.2 s debug, 0.21 s release with 67 MB (revision 4: 34.7 s release, 524 MB) |
| `budget_small_coefficients` (revision 5) | `-((1)/(1-x))/x^250000` at 0, order 1, budget 1e8 | `BudgetExceeded` (its peak memory, 113 MB, is checked by `EXP/rev5/tools/mem5.py`, not by the test) | rev 5: 1.3 s debug (revision 4: 524 MB release) |
| `budget_chains` (revision 5) | 5000 distinct symbols y_k (17 KB): `x + y1 + ... + y5000`; `x*y1*...*y5000`, at 0, order 1, budget 1e8 | `Ok` for both (exactly the input, its terms in canonical order) | rev 5: 27 ms and 39 ms debug (revision 4: the first refused from 3000 symbols) |
| `deep_api_inputs` (revision 5; revision 7rt1, red team 1 H1 and H3) | built with the `Expr` API, debug, on a 2 MB thread, at 0, order 1: `sin(sin(...(x)))` with 1000 levels; `x + sin^300(1)`; a right-deep sum of 500 `x`. Revision 7rt1, each with 50 000 levels (as long as a 100 KB text): `x + 1^1^...^1`; `x^2^...^2` at 1; `x^x^...^x` at 1; `x + 2^-2^-...^-2`; `x/2/.../2`; `x + (1/2/.../2)`; `1/x/.../x` at 1; and `x*3/2*3/2...` with 25 000 pairs, in the structure §2.12's parser gives the text | `Ok` for each (`x`, `x + sin^300(1)`, `500 x`); then `x + 1`; c0 = 1 and c1 the tower's kernel; `1 + (x - 1)`; `x + c` with c the tower's constant; `2^-50000 x` (an exact rational coefficient); `2^-50000 + x`; `1 - 50000 (x - 1)`; `(3/2)^25000 x` (gp `1/2^10` and `(3/2)^10` for the n = 10 cases) | rev 5 reaches 1280, 368 and 648 levels there; HEAD's `taylor` 188, 184 and 188; revision 4 172 for the first. Rev 7rt1: 0.16 to 0.52 s each in debug. Revision 7 handled at most 397, 397, 604, 198, 1292, 397, 1292 and 423 levels there and overflowed the stack beyond; HEAD's `taylor` 2582, 2582, 188, 1296, 188, 187, 188 and 94 (`EXP/rev7rt1/depth_rev7_vs_head_api.txt`, the red team's report). Revision 7rt2: the first three shapes also with 50 000 levels (the engine has no recursion over the input; `Ok` in 0.28 to 0.40 s debug, `bin/enginedepth`) |
| `deep_mixed_nesting` (red team 2 structure, H1; revision 7rt2) | the six mixed shapes, each parenthesis using every grammar level: `x + -(-(...(1)^2*3+4)...)^2*3+4` at 0, `-(-(...(x)^x*x+2*x)...)^x*x+2*x` at 1, `((...(x)^x*x+x)...)^x*x+x` at 1, `x + -(-(...(1)^2/3+4)...)^2/3+4` at 0, `x + -2^-(-2^-(...(1))*3+4)*3+4` at 0 and `-x^-(-x^-(...(x))*x+2*x)*x+2*x` at 1, order 1, budget 1e8: (a) built with the `Expr` API in the structure §2.12 gives them, 10 000 levels, debug on a 2 MB thread; (b) parsed and expanded on one 2 MB debug thread at the deepest level the new parser reads there (§6 step 8); (c) `mix5cs` with 6000 levels parsed and expanded on a 64 MiB debug thread (the REPL's), through `taylor_with_budget`, `series_expansion` and `limit_via_series` | (a) `Ok` or `BudgetExceeded`, no overflow; (b) `Ok`; (c) `Ok` (`1 + x`) and `Finite(1)`; at 2, 3 and 5 levels the values equal gp's (`x+-(-(1)^2*3+4)^2*3+4` at 0 is `1 + x`) | revision 7rt1: the engine overflowed first at 132 (mix5cs), 173, 192, 132, 111 and 168 levels on 2 MB debug, and a 39.6 KB line (4400 levels) killed the REPL's debug thread after parsing; HEAD's `taylor` overflows at 45 to 60 levels. Prototype: 100 KB of each shape handled on 2 MB debug (`EXP/rev7rt2/redteam2_rt2.txt`) |
| `binomial_power_precision` (red team 2 series, S-B3; revision 7rt2) | `(x+1+sin(y))^100/x^5` at 0, orders 3 and 70; `(x+1+sin(y))^100` and `(x+1+exp(1))^100` at 0, order 3 | the nine coefficients of x^-5 to x^3, each `binomial(100, e+5) (1+sin(y))^(95-e)` (gp: `polcoef((x+1+s)^100/x^5, e)` with s a variable), and 76 at order 70; the power without the pole has precision 4, not `EXACT`, with the undecided and with the decided base | revision 7rt1 returned only x^-5 to x^-2, marked exact |
| `mul_exponent_span` (red team 2 series, S-B2; revision 7rt2) | `((1)/x^N + x^N)*((1)/x^N + x^N)` with N = 4 * 10^18 at 0, order 3, through the three entry points; a product of `x^60 + ... + x^124` and `(1)/x^1000 + x^1000 + ... + x^1061 + x^(2^63 - 51)` | `x^-(8 * 10^18) + 2` (terms `[(-8000000000000000000, 1), (0, 2)]`, precision 4); the second `Ok` with first exponent -940 | revision 7rt1 panicked (`index out of bounds`) in release; HEAD answers the first in 0.5 ms |
| `budget_log_sparse_offsets` (red team 2 series, S-B1; revision 7rt2) | `log(1 + x^1000000*(1+x)*(1+x^2)*...*(1+x^65536))/x^1131072` at 0, order 0, budget 1e8, through `taylor_with_budget` and `limit_via_series`; `log(1 + x^5*(1+x)*(1+x^2)*(1+x^4))/x^9` at 0, order 12 | `BudgetExceeded`, and `Unknown` with that error; 16 coefficients equal to gp's `log(1 + x^5*(1+x)*(1+x^2)*(1+x^4))/x^9 + O(x^13)` | release: 0.22 s (revision 7rt1: 78 s and 70 s) |
| `ball_too_wide_goes_on` (red team 2 numeric, m1; revision 7rt2) | at 0, order 5, PQ = 2922842896378005707/1075253811351460636 (a convergent of e): `1/(x + tan(10^18) + 1)`, `1/(x + tan(2^100) + 1/2)`, `1/(x + log(PQ - e) + 84)`, `1/(x + sqrt(PQ - e) - 1)`, `1/(x + 1/(PQ - e) + 1)`, `1/(x + (PQ - e)^(1/3) - 1)`, `1/(x + arcsin(1 - (PQ - e)) - 1)`; `x^(log(PQ - e) + 84)`; and unit tests of `enclose_real`: `tan(10^18) + 1` is `TooWide` at 64 bits and enclosed at 256, `log(-1)` and `gamma(1/3)` are `Never` | `Ok`, each coefficient within 1e-60 of gp's series at 300 digits (gp `tan(10^18)+1` = -7.38854968059368800013); `NotLaurent` (alpha is decided to lie in (0, 1)); the exact zeros of `undecided_constant`, `undecided_after_underflow` and `complex_constant_undecided` stay `UndecidedZero` | revision 7rt1: `UndecidedZero` for every one; HEAD prints the series |
| `exact_monomial_huge_power` (red team 2 structure, m3; revision 7rt2) | at 0, order 1: `x^(2^100) + x`, `x^(2^63) + x`, `(2*x)^(2^100) + x`, `x^(2^100)*sin(x) + x`, `exp(x)^(2^100)`, `(1+0*x)^(2^100) + x`, `(-1+0*x)^(2^100+1) + x`, `(1)/x^(2^100) + x` | `x` four times; `1 + 2^100 x` with no kernel `1 ^ N`; `1 + x`; `-1 + x`; an error (`PrecisionExhausted`: the pole does not fit) | revision 7rt1: `Unsupported` for the first four |
| `float_literal_exact_value` (red team 2 numeric, m2; revision 7rt2) | `1/(x + 0.1 - 1/10)` and `1/(x + 0.5 - 1/2)` at 0, order 1 | `Ok`, c0 = `1/(0.1 - 1/10)` (the binary value of 0.1 minus 1/10 is 1/180143985094819840, documented); `UndecidedZero(-1/2 + 0.5)` | HEAD prints the same series for the first |
| `deep_tower_display` (red team 1, H2; revision 7rt1) | `x^x^...^x` built with the API at 2, order 0: in release (the `--release` run of §6) with 18 000 carets on an 8 MB thread, in debug with 1400 carets on a 2 MB thread; and `x^(x^x)` at 2, order 1 | `Ok`; tree depth at most 2 n + 2 (revision 7rt3: 20 800 and 1499, the values at the point rendered from the input, §2.5; revision 7rt1: 36 002 and 2802); `format!("{}")` succeeds on the same thread (135 991 and 9381 bytes; revision 7rt1: 288 001 and 22 401); `PartialEq` with a second expansion is true. For `x^(x^x)`: `eval_float` of c0 is 16 and of c1 107.110412446601... within 1e-12 (gp `f(x)=x^(x^x); s=f(2+x+O(x^2)); polcoef(s,0)` and `polcoef(s,1)`; c1 = 16 (4 log 2 + 4 log^2 2 + 2)) | rev 7rt1: 60 ms and 3 ms to print in release (revision 7: Display aborts the thread; HEAD's result of depth 17 998 prints); 36 ms debug |
| `input_node_kernels` (revision 5) | on a 64 MiB thread (the debug parser reads about 91 levels of this shape on 2 MB): `x + N_300` (N_k with `^2` at each level, `sin(1)` inside, 3.3 KB) at 0, order 1; `1/(x + log(exp(1+1)) - 2)` at 0, order 2 | `Ok`; `format!` of the first has at most 4 times the input's length (measured 10 217 bytes); the second is exactly `1/x` | rev 5: 1.4 ms release, 1.4 ms debug for N_60 (revision 4: the first result's text has 2^300 nodes) |
| `structural_caps` (§2.15) | unit tests with small caps in `Work` (§2.7): `mono_mul` past the factor cap; `intern` past the kernel cap; `mul` and `sum_many` past the term cap; a kernel past the allowed depth | `LimitExceeded` with the right `StructureLimit`, returned before the bounded work (the counter shows no charge for it) | |
| `point_is_zero_recorded` (review 5, m2) | `series_expansion` of `exp(x)` at the point `pi - pi`, and at `pi` | `to_expr()` renders `x` and `x - pi` from `point_is_zero` (a unit test in `series.rs`) | |
| `budget_shared_kernels` (review 4, B2) | at 0, budget 1e8, with N_k = `(1+exp(...(1+exp(sin(1)))^2...))^2` nested k times and M_k the same with `sin` and 1: `exp((1+exp(N_16))^511 + x)`, order 1; `exp((1+exp(N_30))^511 + x)`, order 1; `exp((1+exp(N_10))^511*(1+exp(N_10'))^511 + x)` (N_10' with `sin(2)`), order 1; `1/((1+sin(M_30)) + x)`, order 2; `1/((1+exp(N_16))^511 + x)`, order 1 | `Ok` (1.8e7 units); `Ok`; `BudgetExceeded` (a 262 144-term argument: its key alone passes the budget); `Ok` (5.1e4 units: the numeric test is memoized per shared node); `UndecidedZero` (the ball of exp(N_16) leaves the window) | rev 4: 0.3 to 0.5 s, 0.3 to 0.4 s, 2.1 to 2.4 s, 4 ms, 0.3 s debug (two runs); peak 4 to 5 MB (VmHWM in `budget4.py`) (revision 3: 11.4 s and 2.56 GB for the first, an abort at 6 GiB for the third, killed for lack of memory after 74 s for the fourth, release) |
| `budget_erf_groups` (review 7, B1) | at 0, order 1, budget 1e8, with `G = erf(975/38) + erf(-975/38)` (exactly 0): `1/(x + (y1 + ... + yn)*G + y0)` with n = 3700 (12 KB, `EXP/review7/gen7.py erfgroups`); the same with n = 3; `1/(x + erf(26) - 1)`; `1/(x + erf(28) - 1)` | `BudgetExceeded` (rule 3 tests one group per indeterminate, and each group's G reaches 1024 bits, where each of its erf nodes is charged `erf_units(1024, 25.66) = 3 084 208`); `Ok` with c0 = 1/y0 and c1 = -1/y0^2 after the zero groups (gp `print(truncate(1/(x+(a+b+c)*(1-erfc(975/38)+(-1+erfc(975/38)))+d)+O(x^2)))` gives 1/d - x/d^2 up to zeros of order 1e-327); `Ok` with c0 = `-1/(1 - erf(26))`, decided at 1024 bits (gp `erfc(26)` = 5.663e-296); `UndecidedZero(-1 + erf(28))` (gp `erfc(28)` = 6.56e-343, below 2^-1024: the cheap enclosure is as wide as MPFR's) | rev 7: 0.21 s, 0.04 s, 2 ms and 0.2 ms debug, 2 MB thread; release 0.15 s for n = 3700 (revision 6: `Ok` after 32.1 s release, review 7) |
| `series_expansion_shares_kernels` (review 7, B2) | `(Y)/((1-x)*x^3000)` at 0, order 2000, budget 1e8, Y a symbol of 99 990 characters (100 KB): through `taylor_with_budget`, through `series_expansion`, and `limit_via_series` at 0 | `Ok` with 5001 stored terms; every leaf of the `taylor_with_budget` result that is the symbol Y is one allocation (`Arc::ptr_eq`, 5001 leaves, one address); every coefficient of the `SeriesExpansion` is `Arc::ptr_eq` to `coefficient(-3000).unwrap()`; the limit is `Unknown` (a pole). The heap of the call is checked by `EXP/rev7/tools/b2probe.py`, not by the test | rev 7: 0.06 s debug (`EXP/rev7/proto8/sharecheck.rs`); release 2 MB of heap at 4.5e6 units (revision 6's owned coefficients: 983 MB, review 7) |
| `empty_argument_negative_precision` (red team 3 series, H1; revision 7rt3) | at 0, order 2: `x^10*exp((sin(x)-x)/x^4)`, the same with `sin`, `cos`, `sinh`, `cosh`, `erf`, `2^(..)`, `(1)/(1-x) + x^10*exp((sin(x)-x)/x^4)`; at 1 `(x-1)^10*exp((sin(x-1)-(x-1))/(x-1)^4)`; at infinity `exp((sin((1)/x)-(1)/x)*x^4)/x^10`; at 0, order 20 000, `x^30001*exp((exp(x^30000)-1)/x^30001)`; through `taylor_with_budget`, `series_expansion` and the limit path; controls `exp((sin(x)-x)/x^2)` at order 2 and `x^10*exp((sin(x)^2+cos(x)^2-1)/x^4)` | `NotLaurent` each (gp: "domain error in exp: valuation < 0"), the limit path `Unknown` (`1 + x^10*exp((sin(x^2)-x^2)/x^9)` too: its left limit is +infinity, gp f(-0.1) = 2.4e62); the controls `1 - x/6 + x^2/72` (gp) and `O(x^3)` | `0`, `1 + x + x^2`, `Finite(1)` (revision 7rt2 and HEAD) |
| `zero_power_rule` (red team 3 series H1 and structure H1; revision 7rt3) | `0^((sin(x))/x)` at 0, orders 0 and 3; `exp(x) + ((x+1)^2 - x^2 - 2*x - 1)^(x+1)` and `cos(x) + (x*0)^(1+x^2)` at 0, order 3; `(x-x)^x` and `1 + (x*0)^(2^x)` at 1, order 2; `x + 0^(2^x)` at 1 and `x + 0^(x+sqrt(2))` at 0, order 2; `x + 0^((sin(1)^2+cos(1)^2-1)/x + 2)` at 0; `x + 0^(1/x)`, `x + 0^x`, `x + 0^(I + x)`, `x + 0^(-1 + x)`, `x + 0^(y + x)` at 0 | the exact 0 twice; `1 + x + x^2/2 + x^3/6`, `1 - x^2/2` (gp); `0`, `1`; `x` (written `1 + (x - 1)` at 1) and `x`; `UndecidedZero`; `NotLaurent`, `NotLaurent`, `NotLaurent`, `DivisionByZero`, `UndecidedZero(y)`; the same through the three entry points (the limit path `Finite` only where the constant term is rational) | `NotLaurent` at order 0; `DivisionByZero("log(0)")` (HEAD answers); `NotLaurent` |
| `result_tree_cap` (red team 3 structure, H2; revision 7rt3) | `F_k = (1+exp(...(1+exp(x))^2...))^2` at 1: k = 30 at order 0, k = 10 and 11 at order 1, k = 12 at order 1; then `simplify`, `substitute(x, 2)` and `==` with a clone of each `Ok` result | the order-0 result is the input with x = 1 (a tree of 151 nodes, equal to gp's value); trees of 462 903 and 1 036 403 nodes (revision 8: 468 023 and 1 046 643; coefficients equal to gp's at k = 2 and 3); `LimitExceeded(TreeSize)` before any rendering; each consumer returns (on the order-0 result in microseconds) | a result whose tree has 1.07e10 nodes (`==` 28.7 s) and 1.04e7 (`simplify` 0.8 s, 960 MB) |
| `asinh_acosh_atanh_enclosures` (red team 3 numeric, m1; revision 7rt3) | at 0, order 3: `(1)/(x + asinh(-2^600) + 400)`, `(1)/(x + asinh(-10^200) + 400)`, `(1)/(x + asinh(1/10^400)*10^400 + 1/2)`, the same with atanh, `(1)/(x + asinh(2^(2^27)) - 93032640)`, the same with acosh | `Ok` each, every coefficient equal to gp's where gp evaluates the constant (-16.581455516527, -61.210165779369, 1.5); the enclosures of `asinh(2^(2^27)) - 93032640` and `acosh(..)` contain gp's `(2^27 + 1) log 2 - 93032640` = 0.437508807794 | `UndecidedZero` (log formulas) |
| `integer_power_beyond_2_32` (red team 3 numeric, m2; revision 7rt3) | at 0, order 3: `(1)/(x + (1+(1)/10^9)^4294967297 - 5)`, `^8589934592 - 5`, `(1-(1)/10^9)^8589934592 - 1/2`, `(1+(1)/10^9)^(-8589934592) - 1`; the ball of `(1+1/10^9)^9223372036854775807` and of `(-1-1/10^30)^36893488147419103233 + 1` (beyond i64) | `Ok` each; the enclosures contain gp's `exp(k log(1 + 10^-9)) - c`: 68.3298161434, 5372.2619249, -0.4998140317, -0.9998140317; `(1+1/10^9)^9223372036854775807` has no enclosure (its value, 1.4e4005659578, is outside the window), and the beyond-i64 power is too wide at 64 bits and enclosed at 256 bits around gp's `1 - exp(k log(1 + 10^-30))` = -3.6893488148e-11 | `UndecidedZero` |
| `stalled_loss_step` (red team 3 series, m1; revision 7rt3) | `((x+x^2)^1000)/x^999` at 0, orders 1, 2, 5, 10, 40, 50 and 400; `(sin(x)^1000)/x^999` at order 10; `sin(x)^(1000000)/x^1000000` at order 2; the limit path of the first | `Ok`, every coefficient equal to gp's `x (1 + x)^1000` (`checks/coefcheck.py`); the limit path `Finite(0)` | `PrecisionExhausted(27)` to `(977)` at orders 2 to 40 |
| `inv_offsets_unsigned` (red team 3 series, m2; revision 7rt3) | `(1)/((1)/x^2 + x^9223372036854775806)` at 0, order 3, in debug on a 2 MB thread | `x^2 + O(x^4)` | panic `attempt to subtract with overflow` (revision 7rt2, debug) |
| `limit_first_term_only` (red team 3 memory, m3; revision 7rt3) | the limit path of `1 + x*S_17` and `1 + x*S_18` at 0 (S_k = `(1+cos(1))*...*(1+cos(k))`), `LIMIT_SERIES_WORK` | `Finite(1)` for both, below the order-0 expansion's units plus one rendering of the constant term | `Unknown (BudgetExceeded)` for S_18 |
| `drop_shared_children` (red team 3 structure, m1; revision 7rt3) | on a 2 MB thread in debug: `N_{k+1} = Binary(Mul, a, a)` with `a = Arc(N_k)` and `N_{k+1} = Binary(Add, a, Arc(Unary(Neg, a)))`, 10^5 levels each, dropped | the thread joins | stack overflow at 14 513 and 14 455 levels (revision 7rt2's Drop) |
| `value_at_point_irregular_exponent` (review 8, B1; revision 8) | through `taylor_with_budget`, `series_expansion` and the limit path: `x + (1+sin(1))^(500*sign(x^2))` at 0, order 1; `(1+sin(1))^(((1)/x)*0 + 500)` at 0, order 1; `x + (1+sin(1))^(500*((x-1)/(x-1)))` at 1, order 1; W_4 at 1, order 1, with W_0 = x and W_{k+1} = `(1+sin(W_k))^(2*sign((x-1)^2))`; V_3 at 1, order 1, and V_20 and V_30 at 1, order 0, with V_0 = x and V_{k+1} = `(1+exp(V_k))^(2+((1)/(x-1))*0)`; control `x + (1+sin(1))^(2 + log(x)*0)` at 1, order 1 | outcome and values asserted: the first c0 has no `sign` node and its enclosure contains gp's `(1+sin(1))^500` = 3.8233154825e132, c1 = 1; the second has no `Div` node with a zero denominator, c0 the same value, c1 = 0; the third c0 = 1 + (1+sin 1)^500, c1 = 1; W_4 has c0 = 2.89697852518268829051 and c1 = 18.2587797964534041670 (gp `G(t)=my(s=t);for(i=1,4,s=(1+sin(s))^2);s; [G(1), derivnum(u=1,G(u))]`); V_3 has c0 = G(1) and c1 = G'(1) with G(t)=my(s=t);for(i=1,3,s=(1+exp(s))^2);s (gp, 5.746165117285e886330929042 and 4.741175615507e886330929056); V_20 and V_30 are `Ok` with trees of at most 1000 nodes (the exponents seeded with 2; 277 and 815 in the prototype) and no `Div` node with a zero denominator (HEAD answers both in under 0.1 ms); the control c0 = 1 + (1+sin 1)^2, c1 = 1; the limit path `Unknown` for these (a constant term that is not rational, §2.13), never a value | c0 printed `(1 + sin(1)) ^ (500 * sign((0 ^ 2)))` (value 1, gp `sign(0)` = 0), `... ^ (((1 / 0) * 0) + 500)`, `... ^ (500 * ((1 - 1) / (1 - 1)))`; W_4's c0 evaluates to 1 (revision 7rt3; HEAD prints the same substituted text for the first and fourth); with the flag fix alone (review 8's `p8.diff`), V_20 and V_30 are `LimitExceeded(TreeSize)` |
| `integer_valued_exponent_enclosure` (review 8, m1; revision 8) | the enclosures at 64 bits of `(-3 + 1)^(-2)`, `(-2)^(3-1)`, `(-2)^(2*3-3)`, `(-3)^(-(-(5)))` and `(sin(1)^2+cos(1)^2-1)^(3-1) + 1`; `(1)/(x + sin((-3+1)^(-2)) - 1/4)` and `(1)/(x + sin((-2)^(-2)) - 1/4)` at 0, order 2; review 8's random tower `(2 + cos((1 + atan(((-3) + atan((1 + log((1 + tanh(x))^(-2)))^2))^(-2)))^2))^(2/3)` at -2, order 0 | balls that exclude 0 around 1/4, 4, -8, -243 and 1; `Ok` with c0 = 1/(sin(1/4) - 1/4) = -385.20196639526608917, c1 = -1/(sin(1/4) - 1/4)^2 = -148380.55491477970544, c2 = 1/(sin(1/4) - 1/4)^3 = -57156481.527993906641 (gp `s=sin(1/4)-1/4; [1/s, -1/s^2, 1/s^3]`, HEAD's values) for both; the tower `Ok`, equal to gp's value (`checks/gpres.py`) | no enclosure (`None(Never)`: `exp(b log a)` with a < 0); `UndecidedZero` for the two inputs and the tower (revision 7rt3) |
| `zero_power_constant` (review 8, m2; revision 8) | at 0, order 2, through the three entry points: `x + 0^pi`, `x + (x-x)^pi`, `x + (x-x)^(x-x+pi)`, `x + 0^(sqrt(2)-1)`, `x + 0^(-pi)`, `x + 0^(sin(1)^2+cos(1)^2-1)`; control `x + 0^(pi + x)` | `x` four times (the limit path `Finite(0)`); `DivisionByZero("0 to a power with negative real part")`; `UndecidedZero`; the control `x` | `UndecidedZero(0^pi)` for the first four and `UndecidedZero(0^-pi)` (revision 7rt3) |
| `drop_during_tls_teardown` (review 8, m3; revision 8) | four threads (debug, 2 MB each): each first touches a thread-local `RefCell<Option<Arc<Expr>>>`, then drops a deep `Expr` (more than 16 KiB of glue, so the iterative branch creates `LEAF` after the thread-local), then stores in it a unique chain, `twin`, `sib` or a left-deep sum of 2 x 10^5 levels, and exits | each thread joins (on Linux the TLS destructors run in reverse order of registration, so `LEAF` goes first and the value is dropped through the fallback leaf; the test asserts the join whatever the order) | panic in `Drop` ("cannot access a Thread Local Storage value during or after destruction") during the teardown, which does not finish: the thread never joins (revision 7rt3's `LEAF.with`) |
| `exact_sentinel` (red team 4 series, H1; revision 7rt4) | at 0 unless stated, through `taylor_with_budget`, `series_expansion` and the limit path, release and on a 2 MB debug thread: `(x^(2^62)*x^(2^62-1))/x^(2^63-1)` and `(x^9223372036854775806*x)/x^9223372036854775807` (order 0), `(x^9223372036854775807 + 1)/x^9223372036854775807` (order 0), `((1)/x^(2^62)*(1)/x^(2^62-1))*x^(2^63-1)` at oo (order 0), the control `(x^4611686018427387903*x^4611686018427387903)/x^9223372036854775806`; then `((1)/(1-x^9223372036854775807) - 1)/x^9223372036854775807` (order 0), `((1+x)*x^(2^62)*x^(2^62-1))/x^(2^63-1)` and `((x^(2^62)+x^(2^62+1))*(x^(2^62-1)+x^(2^62)))/x^(2^63-1)` (order 2), `(x^(2^63-4)*exp(x))/x^(2^63-2)` and `(x^(2^63-4)*((1)/(1-x)))/x^(2^63-2)` (order 2), `((x+x^2)^9223372036854775806)/x^9223372036854775806` (order 1) and the same minus 1 over x (order 0); and the shift property: random products and quotients of monomials with exponents near 2^62 and 2^63 - 1 times a function of x, each `Ok` equal to the same input with every large exponent moved down by one shift (`tools/maxfuzz_rt4.py`) | `1`, `1`, `1/x^9223372036854775807 + 1`, `1`, `1` (gp: the value 1 and the two terms; the limit path `Finite(1)` for the first two); every other row an honest refusal, never a stored term: `PrecisionExhausted` for the first, the overflow of the exponent `2^63` for the two products (`PrecisionExhausted`), `Unsupported("a series precision overflows i64")` for the two clamped precisions, `PrecisionExhausted` for the power (its value needs P above i64::MAX); the limit path `Unknown` for each; the property holds on every `Ok` | a decided `0` (`Finite(0)`) for the value 1, `1/x^9223372036854775807` without the `+ 1`, the truncated `1/x^2 + 1/x + 1/2` and `1` claimed exact, `Finite(0)` for 2^63 - 2 (revision 8) |
| `dense_buffer_units` (red team 4 series, H2; revision 7rt4) | at 0, order 0, with the budget 1e8, release and on a 2 MB debug thread: `exp(x)/x^(2^61)`, `sin(x)/x^1152921504606846976`, `(1+x)^(1/3)/x^1152921504606846976`, `((1)/x^100000000000000000 + 1)^65`, `(1)/(x^2305843009213693952+x^2305843009213693953)`, `((1)/(1-x^9223372036854775806) - 1)/x^9223372036854775806`; every recurrence (`exp`, `sin`, `cosh`, `tan`, `log(1+x)`, `atan`, `erf`, `(1+x)^(1/3)`, Miller and binomial powers, `inv`) divided by x^(2^k) for k = 57 to 62; `exp(x)` through `series_expansion` at orders 2^57 - 1 to 2^63 - 2; and without a budget `exp(x)/x^(2^60)` and `sin(x)/x^(2^61)` | `BudgetExceeded` for every budgeted row (`work` saturated at u64::MAX), the limit path `Unknown (BudgetExceeded)` (for the last text row `Unknown (PrecisionExhausted)`, its loss step overflows first), in at most 0.3 ms, no panic; without a budget `Unsupported("a series buffer of more than 2^57 slots")` | a release panic "capacity overflow" (debug: "attempt to add with overflow" in `Work::charge`) from 2^60 slots (revision 8) |
| `constant_poles` (red team 4 numeric, m1; revision 7rt4) | at 0, order 2, the three entry points: `(1)/(x + atanh(1))`, `(1)/(x + atanh(-1)) + x`, `x + atanh(sin(pi/2))`, `x*atanh(1)`, `x + zeta(1)`, `(1)/(x + log(1-1))`; controls `x + atanh(1/2)`, `x + zeta(2)`, `x + gamma(0)` | `DivisionByZero` for the six (the limit path `Unknown (DivisionByZero)`, never `Finite(0)`); the controls a kernel plus x, and `gamma(0)` `DivisionByZero` as before | `Ok` with `1/arctanh(1) - x/arctanh(1)^2 + ...`, `Finite(0)` for `x*atanh(1)` (revision 8; HEAD the same) |
| `exact_zero_operand` (red team 4 numeric, m3; revision 7rt4) | the enclosures at 64 bits of `1 - 1`, `sqrt(1-1)`, `(1-1)^(2*2)`, `(1-1)^(1/3)`, `(1-1)^(-(2*2))`, `3/4 - 3/4`; `(1)/(x + sin(sqrt(1-1) + 1) - 1/2)`, `(1)/(x + sin((1-1)^(2*2) + 1) - 1/2)`, `(1)/(x + sin((1-1)^(1/3) + 1) - 1/2)` at 0, order 1; red team 4's hinted tower `(1)/(x - 1 + G_6(x + (x-1)^(2*2)) - 3)` (G_k = `(1+sin(..))^2` k times) at 1, order 0 | the exact 0 (radius 0) for the first four and the last, `None(Never)` for `0^-4`; `Ok` for the three with c0 = 1/(sin(1) - 1/2) (gp: `sin(1) - 1/2 = 0.34147098480789650665`), and the tower `Ok` with c0 = 1/(G_6(1) - 3) (gp: `G_6(1) - 3 = 0.99845675697654779266`) | `None(TooWide)` and `UndecidedZero` (revision 8) |
| `exp_integer_log` (revision 7rt4, found while revising) | the enclosures at 64, 256 and 1024 bits of `exp((-2)*log(-3 + sin(1)))`, `exp(log(-7/3)*(3-1))`, `exp((2*(-5) - (-5))*log(1/2 + tanh(-2)))`, `exp(0*log(-2))`; review 8's random tower `((-3) + tanh(((1/2) + sqrt(((1/2) + sqrt(((1/2) + log(((-3) + sin(x))^(-2)))^x))^3))^(2/3)))^(x+1)` at -2, order 0 | balls that exclude 0 around gp's `(-3 + sin(1))^-2`, `(-7/3)^2`, `(1/2 + tanh(-2))^-5` and 1; the tower `Ok`, equal to gp's value (`checks/gpres.py`) | `None(Never)` (log of a negative ball); the tower `UndecidedZero` with the exact kernel counts alone |
| `kernel_tree_count` (red team 4 structure, m1; revision 7rt4) | a unit test that builds one computed kernel of each kind (an operator `sin(c)`, a power `c^alpha`, a radical `2^(1/3)`, an inverse `1/(1 + sin(1))`, a float leaf) and one from an input node; integration: `sin^1200(x+1+0*(y0+...+y499))`, `sin^2000(x+1)`, `sin^1700(x+1)` at 0, order 1, `x^x^...^x` with 20 carets at 1/2, order 2, `cos(x+1)^...^(x+2)` with 100 levels at 0, order 1; the property on red team 3's random inputs: the call with `MAX_RESULT_TREE` set to the exact tree minus 1 is refused | each cached tree equals a memoized count of its expression; the five inputs `Ok` with trees of 724 203, 2 007 003, 1 450 953, 1 494 018 and 1 754 050 nodes, and their coefficients equal to gp's (`checks/gpres.py`); the property holds on every input (`render_tree` is never below the tree) | `LimitExceeded(TreeSize)` for the five (revision 8 counted 3 nodes above the arguments of every kind: `render_tree` 2 167 811 for a tree of 724 203) |
| `limit_with_error_renders_none` (red team 4 memory, m2; revision 7rt4) | `limit_with_error` of `1 + x*S_17` and `1 + x*S_18` (S_k = `(1+cos(1))*...*(1+cos(k))`) at 0, order 5; controls `exp(x)` and `(1)/x` | `(Finite(1), Some(O(x^6)))` for both, within `LIMIT_SERIES_WORK`; `(Finite(1), Some(O(x^6)))` and `(Unknown, None)` | `(Finite(1), None)`: the tree cap of the rendering that no one read (revision 8) |
| `tower_tree_cap` (red team 4 structure, m2; revision 7rt4) | `x^x^...^x` at 1/2, order 2, with 21 and 22 carets; at 2, order 1, with 136 and 137 | 21: `Ok`, equal to gp's (`checks/gpres.py`, 300 digits); 136: `Ok` (the tower's value at 2 overflows gp's reals, so only its structure is checked); 22 and 137: `LimitExceeded(TreeSize)` below the budget (the cap, not the budget, is the limit; §2.15, §7 item 8) | 19 and 118 were the last admitted (revision 8) |


Zero test (unit tests in `series_const.rs`): `sqrt(2)*sqrt(2) - 2` is `Zero`; `sqrt(8) - 2*sqrt(2)`
is `Zero`; `I^2 + 1` is `Zero`; `pi^2 - 10` is `NonZero` (single named constant); `pi + I` is
`NonZero` (rule 4); `y + sin(1)` is `NonZero` (indeterminate group); `sin(1)^2 + cos(1)^2 - 1` is
`Unknown`; `2 + sin(1)` is `NonZero` (numeric); `sin(1) + I` is `NonZero` (both parts enclosed);
`sqrt(-4) - 2*I`, `exp(I*pi/2) - I` are `Unknown`; `sqrt(6) - sqrt(2)*sqrt(3)` is `Unknown`
(documented incompleteness); `sin(pi)` folds to 0; `exp(pi) - pi` is `NonZero` (numeric);
`log(-1)` has no enclosure. `non_integer_status`: `pi`, `y`, `1/2 + I`, `sqrt(2)` are decided
non-integers; `sin(1)^2 + cos(1)^2` is `Unknown` with nearest integer 1; `2^100000000 * sin(1)` is
`Unknown` without one.

Ball unit tests (in `series_ball.rs`; review 2, B1 and B2): `enclose_real` is `None` for
`(1/2)^1073741827`, `exp(-744261120)`, `exp(744261120)`, `2^(2^29)` (outside the window) and for
`tan(2^100000000)`; it is `[0 ± 1]` for `sin(2^100000000)` and `cos(2^100000000)`; R2 accepts the
zero midpoints of `sin(0)` (the exact ball [0, 0]) and `log(1)` (a ball around 0 of radius about
2^-p); `2 + sin(2^100000000)` excludes 0. Review 3, m4: `2 + erf(1)` and `erf(1) - 1/2` are
enclosed away from 0, and `erf(0)` is the exact ball [0, 0].
`ball_cost(sin(2^100000000), 64, u64::MAX)` is at least the trig unit `((2*64 + 64)/64 + 1)^2 * 8 =
128`, and `ball_cost` of a long sum returns as soon as it passes its limit. Revision 7 (review 7, B1):
`erf_threshold` is 7.5, 14 and 27 at 64, 256 and 1024 bits; `enclose_real(erf(30), 1024)` is the ball
`1 +- 2^-1040` and `erf(-30)` is `-1 +- 2^-1040` (the cheap enclosure: both return in microseconds);
`erf(26) - 1` is enclosed away from 0 at 1024 bits and `erf(55/2) - 1` is not; a ball of midpoint
27.25 and radius 1 gives `[-1, 1]` for erf; the erf node of `erf(975/38)` adds
`erf_units(1024, 25.66) = 3 084 208` units to `ball_cost` at 1024 bits, that of `erf(sin(1))` adds
`erf_units(1024, None) = 3 407 888`, and that of `erf(30)` adds `2u = 4624`.

Kernel unit tests (in `series_const.rs`; review 2, m2; review 4, B2): creating `sin(2^340000 + 1)`
charges at least one unit per key byte (about 85 000) before the key exists; the key contains the
integer in hexadecimal; creating it twice in one `Work` gives the same kernel (same id), and the
second creation charges the key but no expression units; `sin(1)` and `sin(2)` compare unequal;
`Real(2.0)` and `Integer(2)` inside a kernel give different keys; the key of `sin(sin(1))` names
`sin(1)` by its id (it does not contain the text `sin(1)`); the kernel of the input node `sin(1)` has
that node as its expression (its child is pointer-equal to the input's, revision 5), while the kernel
`exp(b0)` of a series value renders b0's normal form; for N_k built by `from_expr` in one
`Work`, the key of the outermost kernel is under 100 bytes at k = 10 and at k = 20 (each level
refers to the one below by id), while the expression tree doubles per level; `has_indet` is set for
`sin(y + 1)` and not for `sin(pi + 1)`, and `KernelData` has no key field (one copy, in the table).

Copy unit tests (in `series_const.rs` and `series_engine.rs`; review 4, B1): `copy_units` and
`expr_units` follow their formulas (a 1-term constant `1/3`: `52 + 0 + 3 * 2 = 58` and
`16 + 32 * 2 + 3 * 2 + 11 * 2 = 108`, and building it charges `72 + 3 * 2 + 16 = 94`, revision 6); a
`Clone` of a constant charges nothing and shares
its terms (`Arc::ptr_eq`); `neg` charges `copy_units` and `CONST_BASE`; `add` of a 2-term and a 1-term
constant with one common monomial charges `copy_units` of both, one `q_add` and `CONST_BASE` (no
search units: a merge); the series `add` of two 3-term series charges 16 + 36 plus its constant adds;
each occurrence of the variable charges 24 plus its node visit (16). Revision 5: a one-term constant
holds its term inline (`Terms::One`); `scale` by 1 returns a shared constant; `mul` of two constants
with 20-factor monomials charges `search_units` for 4 excess factors per operand monomial. Revision 6:
a constant of several terms is a boxed slice in monomial order (`Terms::Many`); `zero()` and `one()`
allocate nothing (pointer-equal across calls); `sum_many` of three constants charges `copy_units` of
the three, 3 `SER_TERM` and `CONST_BASE`; `mul` of a 2-term by a 3-term constant charges `6 ACC_ENTRY`;
`to_expr` of `sin(1)*sin(2)` allocates the product node only (its factors are pointer-equal to the
kernels' expressions); `render_units` of `[(0, 1), (1, 1)]` is `2 EXPR_NODE + 2 RENDER_TERM` plus the two
coefficients' `expr_units`; `zero_status` of a 2-term constant without indeterminates and without `I`
copies nothing (`indet_groups` and `split_imag` are not entered); the series `scale` of a 3-term series
by `sin(1)` charges 16 + 18 even though each product is a shared constant.

Ball memo unit tests (in `series_ball.rs`; review 4, B2): for `s = Arc::new(sin(1))` and
`e = Binary(Add, s.clone(), s.clone())`, `ball_cost(&e, 64, u64::MAX)` counts `sin(1)` once (it
equals the cost of `sin(1)` plus one `+` node), and `enclose_real(&e, 64)` equals `2 sin(1)`'s ball;
`enclose_real` of the expression of `sin(M_30)` (built by `from_expr`, a tree of more than 2^30
nodes) returns an enclosure in milliseconds. Revision 5 (review 5, B1): `enclose_real` of a sum
chain of depth 100 000, built with the `Expr` API, returns on a 2 MB thread in debug; and of an
expression with a `Function` leaf deep inside, returns `None` (the first missing child ends it).

Charged arithmetic unit tests (in `series_const.rs`; review 3, B1, B2, m2): each function charges
exactly the formula of §2.7 (for example `q_add` of 1/3 and 1/5 charges 64 + 3 + 3 * 2 = 73 units,
and `q_mul` of two 2-word integers 64 + 4 + 1 + 4 = 73); with `Work::new(1000)`, `q_pow(3/5,
200000)` and `i_exact_root(3^340000 + 3, 2)` return `Err(BudgetExceeded)` and leave the counter at
the charged amount (the power and the root never run); revision 7 (review 7, m4): `q_mul` of 7 by an
integer of 6336 bits (100 words) charges `max(64 + 100 + 1 + 101, 3 (101 + 2)) = 309` units, where
revision 6 charged 266; `pow_bits(2, 340000)` is between 340 001
(the bits of 2^340000) and 340 003, and `pow_bits(3, 340000)` is at least `bits(3^340000) = 538 888`
(gp `#binary(3^340000)`) and below 1.0001 times it. The trial division of a 63-bit radicand with
d = 3 charges `(1 + 8) * 6542` units before it runs (review 4, m2).

Large orders (unit tests; run on the default 2 MB test thread, debug):

| test | input | expected | HEAD |
|---|---|---|---|
| `taylor_order_20000_polynomials` | x^2, 2*x+1, x at order 20000 | `x^2`, `1 + 2*x`, `x` | x^2: 1.92 GB at order 9; x: stack overflow |
| `taylor_order_20000_budget` (revision 7, review 7 m1) | with `taylor_with_budget` and budget 1e8 at order 20000: `x + 1`; `1/(1-x)`; `exp(x)`; `sin(x)` | `(1 + x)`; `Ok` with 20001 coefficients 1; `BudgetExceeded` twice, each "returns within 20 s" | `x + 1`: 0.48 s and 324 MB (release, 64 MiB thread); the others: out of memory. Prototype: 0.1 ms, 0.09 s, 0.28 s and 0.24 s debug |
| `taylor_exp_order_2000_depth` | exp(x), 0, 2000 | coefficient of x^2000 is `1/2000!` (gp `1/2000!`); tree depth of the result <= 16; `format!("{}")` succeeds | out of memory |
| `simplify_long_left_deep_sum` | `1 + 1 + ... + 1` (100000 terms, built in a loop) simplified in a thread with a 1 MiB stack; the input is dropped normally (no `mem::forget`) | `Integer(100000)` | stack overflow |
| `drop_left_deep_sum_1e6` (B4) | a left-deep sum of 10^6 terms `x^(k mod 5)`, built and dropped in a thread with a 1 MiB stack | the thread joins without panic | stack overflow |
| `drop_other_shapes` (B4) | right-deep power tower, unary chain, function chain, zig-zag, each 10^5 deep, 1 MiB thread; a shared deep subtree dropped one owner at a time | joins without panic | stack overflow |

### 4.2 Existing tests to rewrite (`series.rs:813-1003`, `lib.rs`)

- `test_taylor_polynomial`: `Ok(x^2)`.
- `test_maclaurin_exp`, `test_unified_series_function`: compare the coefficient list with gp
  (structural equality with `known_series::exp` no longer holds; known_series builds
  `x^2 / 2`, the engine `1/2 * x^2`).
- `test_series_coefficients`, `test_calculus_series_coefficients`: `[0, 0, 1, 0]` exactly (the old
  test never checked `coeffs[2]`).
- `test_taylor_around_nonzero`, `test_series_at_nonzero_point`: `1 + 2*(x-1) + (x-1)^2`.
- `test_laurent_positive_orders`: `laurent(exp(x), 0, 0, 3) == taylor(exp(x), 0, 3)`.
- `test_series_with_big_o`: BigO is `O(x^4)`.
- `test_asymptotic_expansion`: `1 + 1/x - 1/x^2 + O(..)` coefficients (the old test only checked
  "not constant").
- `lib.rs` tests: `.unwrap()`, assertions unchanged.

### 4.3 Parser and Display (`parser.rs` tests)

| test | input | expected | HEAD |
|---|---|---|---|
| `test_parse_big_integer` | `x + 18446744073709551616`, `-9223372036854775808`, `123456789012345678901234567890` | Integer values (`Integer::from_decimal_str`) | parse error |
| `test_parse_division_by_non_literal` | `1/x`, `1/sin(x)`, `exp(1/x)`, `1/(y+x)` | `Div(1, ..)` | parse error |
| `test_parse_rational_precedence` | `x^2/4`, `2^3/4`, `2/3^2`, `3/4^2`, `x*2/3^2`, `x/2/3`, `2^-1/3` | `Div(Pow(x,2),4)`; simplify gives 2, 2/9, 3/16, `x*2/9`, `x/6`, 1/6 (gp: `2^3/4`, `2/3^2`, `3/4^2`, `2^-1/3`) | x^(1/2), 2^(3/4), 4/9, 9/16, 4x/9, x/(2/3), 2^(-1/3) |
| `test_parse_signed_exponent` | `2^-1`, `x^-2`, `2^-x` | `Pow(2, Integer(-1))`, `Pow(x, Integer(-2))`, `Pow(2, Neg(x))` | third: error |
| `test_parse_literal_fold` | `1/3`, `6/4`, `1/-3`, `-2/3`, `1/0` | `Rational(1/3)`, `Rational(3/2)`, `Rational(-1/3)`, `Rational(-2/3)`, error | (as today except `-2/3` structure) |
| `test_parse_inverse_hyperbolic` | `asinh(x)`, `arccosh(x)`, `atanh(x)` | `Unary(Arcsinh, x)` etc. | `Function` |
| `test_parse_sign_runs` (revision 5) | `--5`, `---x`, `x^--2`, `+-+x`, and 100 000 `-` before `x` parsed on a 2 MB thread | `Integer(5)`, `Neg(x)`, `Pow(x, 2)`, `Neg(x)`, `x` (an even run) | parse error for every run of two or more signs |
| `test_display_work_list` (red team 1, H2; revision 7rt1) | on a 2 MB thread in debug, each built with the `Expr` API: a left-deep sum of 10^5 terms `x` (the HANDOFF note of §1.8 for 3000), a tower `x^(x^(...))` of 10^5 levels, `sin(sin(...(x)))` and `f(f(...(x)))` of 10^5 levels; and every node kind (each `UnaryOp` and `BinaryOp`, `Function` with 0, 1 and 3 arguments, the cases of `test_display_parse_round_trip`) and every `Ok` engine output of §4.1 except those of `budget_shared_kernels` and `cap_paths_unbounded`, compared with a test-only copy of the recursive writer of §2.11 (`display_reference`: HEAD's `fmt` with the three rules) | every `format!` returns; the sum prints `(((x + x) + x) ... + x)`, 6 * 10^5 - 5 bytes; every string equals `display_reference`'s byte for byte | stack overflow at HEAD from about 3000 terms in debug (§1.8); prototype: `displaycheck.rs` finds the work-list writer equal to HEAD's on 368 expressions |
| `test_display_integral_float` (red team 3 structure, m2; revision 7rt3) | `Real(1e23)`, `Real(2.0)`, `Real(-3.0)`, `Real(0.5)`, and the engine output of `(1)/(x + 100000000000000000000000.0 - 100000000000000000000000)` at 0, order 1, written and read back by §2.12 | `100000000000000000000000.0`, `2.0`, `-3.0` (in parentheses where a negative literal needs them), `0.5`; each reads back as the same f64, and the output's value equals its own (c0 = `1/(-8388608)`) | `100000000000000000000000` reads back as 10^23, a division by zero |
| `test_display_negated_fraction` (red team 2 structure, m1; revision 7rt2) | `Neg(Rational(2/3))`, `Neg(Rational(-2/3))` alone, under `^` and as a divisor; the engine outputs of `x*pi^-(2/3)` at 1, `x + atan(3/-(1/2))` at 1, `x + y^-(1/2)` at 0 and `x + atan(2^-(1/3))` at 0, order 1 | `-(2/3)`, `-(-2/3)`, `(x ^ -(2/3))`, `(x / -(2/3))`; the results print `(pi ^ -(2/3))`, `(3 / -(1/2))`, `(y ^ -(1/2))`, `(2 ^ -(1/3))`, and parse back to the same tree | revision 7rt1 printed `(pi ^ -2/3)`, which §2.12 reads as `pi^(-2)/3` |
| `test_display_round_trip_values` (red team 2 structure, m1; revision 7rt2) | 400 random trees of the §2.12 grammar (a seeded port of `EXP/redteam2-structure/randgen.py`: chains, towers with signed exponents, sign runs, calls, the literals `1/2` and `(1/3)`), each and its engine output at its point written with Display and parsed back | the two trees have equal values: their 256-bit balls at x = 37/100, y = 61/100 overlap (§2.3, after substitution), or neither has an enclosure | the prototype check: 551 inputs, 954 trees, 944 keep their value, 0 differ, 10 cannot be evaluated by gp; with revision 7rt1's Display 78 differ |
| `test_parse_depth_not_below_head` (revision 5) | 160 nested `sin(` and 200 nested parentheses, parsed on a 2 MB thread in debug | parses | the same (HEAD reaches 162 and 186 there; the new rules add no frame per nesting level) |
| `test_display_parse_round_trip` | `Pow(x, 1/2)`, `Pow(3/4, -1/2)`, `Div(x, 1/2)`, `Pow(-2, 2)`, `Pow(Neg(y), 1/2)`, `Pow(Real(-1.5), 2)`, the engine output of `sqrt(x - y)` at 0 (review 1, m3), every `Ok` engine output of §4.1 except those of `budget_shared_kernels` (their text has 2^16 to 2^30 nodes, §3.4) and `cap_paths_unbounded` (a 475 490-bit integer in decimal) | hand-built cases: `parse(&format!("{e}")).simplify() == e.simplify()` (structural equality; `simplify` folds `Neg(Integer(2))` and `Real(-1.5)`, which the parser rebuilds as negations). Engine outputs (review 3, m5): for each `Ok` case `(f, a, n)` the test takes `t = taylor(f, a, n)` and `t2 = parse(&format!("{t}"))`, and the crate helper `series_engine::same_expansion(&t, &t2, x, a, n)` expands both at the same point and order in one `Work` (unbounded) and compares each coefficient's `SeriesConst` normal form. Kernels are interned in that `Work`, so two equal constants have equal term maps (equivalently, the difference has the zero normal form). No numeric comparison is involved, so it also covers kernels such as `(3 + sin(2^100000000))^1000000000`. The helper is `pub(crate)`, so this part of the test is a unit test in `series_engine.rs` | `(x ^ 1/2)` re-parses as `(x^1)/2` under the new parser; `(-y ^ (1/2))` as `-(y^(1/2))` |

Existing parser tests stay green (checked against the grammar: `-17` is `Neg(17)`, `-1.5` is
`Neg(Real(1.5))`, `1.5e-3`, `2 ^ 3 ^ 2` right-associative, error cases unchanged). Their 21 by-value
matches are rewritten for E0509 (§3.3) without changing what they assert.

### 4.4 Limits

`limit_via_series`: `(1 - cos(x))/x^2` at 0 is `Finite(1/2)`; `(exp(x) - 1 - x)/x^2` at 0 is
`Finite(1/2)`; `x^3/x` at 0 is `Finite(0)`; `x*log(x)` is `Unknown`; `1/x` is `Unknown`
(gp: `limit` values from `L(..)[2][1]`). Existing limit tests stay green.

### 4.5 REPL (`rustmath-jupyter/src/repl.rs` tests)

- `test_taylor_order_checked` unchanged (it reads `TAYLOR_ORDER`).
- new `test_taylor_engine`: `taylor("1/sin(x)", x, 0, 3)` is `Ok` with the Display string
  `(((1 / x) + (1/6 * x)) + (7/360 * (x ^ 3)))` (HEAD: the argument does not parse);
  `taylor("log(x)", x, 0, 3)` is a ComputationError containing "logarithmic singularity";
  `taylor("gamma(x)", x, 1, 3)` is a ComputationError; `taylor("exp(x)*sin(x)", x, 0, 2000)` is a
  ComputationError (budget: the product is pre-charged at 5.2e9 units, far above any plausible
  `TAYLOR_WORK`; refused after 0.18 s in the debug prototype; review 1, m6);
  `taylor("x^2/4", x, 0, 2)` is `(1/4 * (x ^ 2))`; `taylor("1/(x-x)", x, 0, 2)` is a ValueError;
  `taylor("1/(x + sin(2^100000000) + 2)", x, 0, 2)` is `Ok` (review 2, B2: 2 ms in the debug
  prototype under the REPL budget; revision 1 ran 111.7 s in release);
  `taylor("x + (3^330000)/(5^200000)*2*3", x, 0, 1)` is a ComputationError (review 3, B1: refused
  at 1.2e8 units after 2.8 s in the debug prototype); a `taylor` call whose `UndecidedZero` constant
  is long (the 404-character input of `undecided_error_text_bounded`) gives a ComputationError whose
  message is under 300 characters (review 3, m6). Revision 5: after `e = x` and 40 lines `e = e + e`,
  `taylor(e, x, 0, 1)` is a ComputationError (budget) within 20 s, and the arm's variable lookup and
  "not found" test (`mentions_symbol`) return at once (review 5 m1; at HEAD the arm's `symbols()` walks
  2^40 nodes); `taylor("5", x, 0, 3)` keeps the ValueError "Variable 'x' not found in expression".
  Do not put `exp(sin(x))` at a high order in a debug test: reaching the budget takes about 1 s in
  release at HEAD and about 10 times that in debug. Revision 7: `taylor("x + 1", x, 0, 20000)` is `Ok`
  `(1 + x)` (the order check keeps HEAD's 20000, review 7 m1) and `taylor("exp(x)", x, 0, 20000)` is a
  ComputationError (budget; 0.28 s in the debug prototype); `expr("-(-x)")` displays `-(-x)` (§2.11,
  the HANDOFF note of §1.8).
- Revision 7rt1 (red team 1, memory H1): `test_expr_latex_work_list`: the work-list `expr_to_latex`
  equals a test-only copy of HEAD's recursive one (`expr_to_latex_reference`, `repl.rs:10883-10935` at
  HEAD) on every node kind (each `UnaryOp` and `BinaryOp`, `Function` with 0, 1 and 3 arguments,
  `Rational`, `Real`, an `Integer` of 10^5 digits through `decimal_string`) and on the result of
  `taylor((sin(x)+sign(pi)+f(2,3)+(2/3)^x)/(1-x), x, 0, 20)`. `test_taylor_display_linear`: F =
  `f(f(...f(1)...))` with 33 000 levels built with the `Expr` API, on a 256 MiB thread in debug (the
  debug engine needs about 5 KB per nested call; a release REPL parses the 99 KB text of the red team on
  its 64 MiB thread): `taylor_with_budget(F/(1-x), x, 0, 100, TAYLOR_WORK)`, then
  `RustMathValue::Expr(..).to_display()`, return within 20 s, and the LaTeX field has 33 331 691 bytes
  (the prototype: 1.2 s in debug, of which 0.49 s the LaTeX; HEAD's writer 36 s); the same with 8000
  levels at order 400 on the REPL's own 64 MiB thread (`EVAL_STACK_BYTES`; 0.9 s debug, HEAD's writer
  10.7 s).

### 4.6 expand under a minus (`expand.rs` tests, revision 6)

`test_expand_negation` is rewritten with assertions, and a REPL test is added. Reference values are
gp's (`printf 'print(-(3/7-1))\nprint(-(3/7+1))\nprint(-(3/7+2/5+1))\nprint(-(3/7-2/5-1))\nprint(-((3/7-1)*(3/7+1)))\n' | gp -q`
gives 4/7, -10/7, -64/35, 34/35, 40/49):

| test | input | expected | HEAD |
|---|---|---|---|
| `test_expand_negation` | `expand` of `-(x - 1)`, `-(x + 1)`, `-(x + y + 1)`, `-(x - y - 1)`, `-((x - 1)*(x + 1))`, each substituted at x = 3/7, y = 2/5 and simplified | 4/7, -10/7, -64/35, 34/35, 40/49; and no `Unary(Neg, _)` node of the result has a sum or a difference as its operand | -4/7, 4/7, 34/35, -36/35, 16/49 |
| `test_expand_negation_structure` | `expand(-(x - 1))` | `Binary(Add, Unary(Neg, x), Integer(1))`, displayed `(-x + 1)` | `(-1 - -x)` |
| `test_expand_under_minus` (REPL) | `expand(expr("-(x - 1)"))`, then the value at x = 3/7 through `subs` | 4/7 | -4/7 |

## 5. Cost: before and after

### 5.1 The four functions of the item (point 0)

Debug (revision 1 prototype `bin/proto2`, budget 1e8): x^2 and 2*x+1 take 36 us and 47 us at order
20000 (HEAD: 1.92 GB at order 9 for x^2, 1.18 GB at order 20 for 2*x+1); exp(x) 0.13 ms at 12 (HEAD
12.6 s, 4.64 GB) and 32 ms at 2000; sin(x) 0.13 ms at 12 (HEAD 14.1 s, 5.23 GB). Revision 0 debug
(`EXP/after_debug*.txt`): exp(x) 0.17 s at 5000, 2.4 s at 20000; sin(x) 2.0 s at 20000 (806 MB, the
coefficients themselves). Revision 2 debug (`EXP/rev2/after_debug_r2.txt`, `bin/proto3`, budget
1e8): x^2 and 2*x+1 80 us and 92 us at 20000; exp(x) 0.24 ms at 12 and 39 ms at 2000; sin(x) 0.23 ms
at 12 and 37 ms at 2000 (the same work units as revision 1; the small differences are noise of a
shared machine). Revision 3 debug at HEAD dc367a2 (`EXP/rev3/bin/proto4`, budget 1e8): x^2 and 2*x+1
46 us and 60 us at 20000; exp(x) 0.13 ms at 12 and 27 ms at 2000; sin(x) 0.17 ms at 12 and 53 ms at
2000 (5.0e6 and 3.8e6 units: the new per-operation base charges more units for the same work).
The HEAD figures of the old algorithm (§1.1) are unchanged at dc367a2: its cost is the size of the
derivative trees, which Lehmer's gcd does not touch.

Release (`EXP/after_release.txt`, revision 0; the sparse revision 1 engine is within noise of it for
these inputs: exp(x) at 2000 takes 14.7 ms, sin(x) 14.1 ms, `EXP/calib_rev1.txt`):

| f | n | engine | to_expr | Display | output length | peak |
|---|---|---|---|---|---|---|
| x^2 | 9 / 20000 | 19 us / 7 us | | | 7 | 3.1 MB |
| 2*x+1 | 20000 | 10 us | | | 13 | 3.2 MB |
| exp(x) | 12 | 26 us | 7 us | 8 us | 268 | 3.2 MB |
| exp(x) | 1000 | 5.2 ms | 0.6 ms | 16 ms | 1.2 MB | 7.4 MB |
| exp(x) | 2000 | 13 ms | 1.7 ms | 83 ms | 5.3 MB | 17 MB |
| exp(x) | 5000 | 77 ms | 8 ms | 0.94 s | 38 MB | 86 MB |
| exp(x) | 20000 | 1.41 s | 78 ms | 39 s | 730 MB | 1.43 GB |
| sin(x) | 1000 / 5000 / 20000 | 4.3 ms / 73 ms / 1.25 s | | 7 ms / 0.47 s / 20 s | 0.6 / 19 / 365 MB | 6.8 MB / 65 MB / 1.0 GB |

The output size of exp(x) is inherent (sum of log10 k! digits for k <= n), and Display of big
integers dominates past order 2000. Revision 6 set `TAYLOR_ORDER = 2000` for that reason; revision 7
keeps HEAD's 20000 (review 7, m1): the budget refuses exp(x) and sin(x) at order 20000 after 0.14 s and
0.12 s in release (0.28 s and 0.24 s debug), polynomials and rational functions with small
coefficients pass (`x + 1`: 860 units; `1/(1-x)`: 1.68e7 units and 12 MB; `1/(1-x)^2`: 2.2e7 units),
and J1b's display budget bounds the text (`EXP/rev7/attack7_final.txt`, family `rev7`). The order is
not an output bound in general (review 2, m5): other admitted inputs print far more at a lower order
(§2.7, §3.1), and the REPL display budget handles them.

Output depth: 18 at order 20000, so Display, simplify, Drop, substitute and PartialEq of the result
are all fine on 2 MB threads. Monomials are balanced products too (review 5, B1), so this also holds
for coefficients with long monomials (depth 16 for 4096 factors), and §2.15 bounds the depth of the
kernels.

### 5.2 General functions (release, point 0)

Revision 0 and 1 figures, measured at c79a582 (Euclid gcd):

| f | n | engine | work units (first model, no base charge) | gp (same series) |
|---|---|---|---|---|
| exp(sin(x)) | 100 / 200 / 300 | 0.06-0.11 s / 0.9-1.05 s / 4.0-4.2 s | 8.5e7 at 300 | 0.18 s at 300, 10.5 s at 1000 |
| tan(x) | 100 / 200 / 300 | 0.08 s / 1.0-1.2 s / 4.5-4.8 s | 1.1e8 at 300 | 15 ms at 300, 0.27 s at 1000 |
| 1/(exp(x)-1) | 200 | 1.04 s | 2.5e7 | |
| log(cos(x)) | 200 | 0.53-0.58 s | 8.1e6 | |
| sqrt(1+x+x^2) | 600 | 0.17 s | 1.2e6 | |
| exp(x)*sin(x) | 500 / 1000 | 2.7 s / 41 s | 4.1e8 at 500 | |
| (1+x)^10000 | 2000 | revision 0: 12.9 s; revision 1 (Miller): 11.5 ms | 6.4e5 | |
| 1/(1+x+x^2), atan(x) | 2000 | 3-4 ms | | |

At HEAD dc367a2 the same engine is 3 to 6 times faster on these inputs, because I3 replaced
Euclid's gcd by Lehmer's (revision 3 prototype, release): exp(sin(x)) at 250 0.43 s, tan(x) at 250
0.37 s, sqrt(1+x+x^2) at 2000 0.47 s, exp(x)*sin(x) at 500 0.70 s (3.4e8 units).

Dense rational series cost O(n^2) operations on coefficients of O(n log n) bits; the gcd in every
rational addition dominates. gp is 20 to 300 times faster (GMP, fast algorithms). Faster series
arithmetic (Newton iteration for inverse/exp/log with fast multiplication, common-denominator
integer arithmetic) is a performance follow-up, not a correctness defect.

### 5.3 REPL budget calibration

**Revision 7rt4** (red team 4), release, HEAD 6e03987, budget 1e8, every call through a public entry
point: `EXP/rev7rt4/tools/runall_rt4.sh` (every check of revision 8 rerun on `proto13`, then red team 4's
reproductions and the class checks of this revision; about 30 minutes on a machine that other builds
shared, load 1.5 to 2); outputs in `EXP/rev7rt4/` with the suffix `_rt4`, compared with revision 8's by
`tools/cmp_outcomes.py`. Every row keeps its revision 8 outcome (0 outcome differences in every file): the
oracles, the calibration, the memory and family probes, the self-attack (165 rows), the DAG and
literal-chain families, the depth searches, the inputs of reviews 3 to 7, the caps, the consumers of
`F_k`, the scans, red teams 1 to 3 and review 8's reproductions; the largest time and VmHWM of each
file are within the noise of the shared machine. The oracles: gp 107 OK and 0 failures (`gpcheck3_rt4.out`),
`revchecks_rt4.out` and `revchecks3_rt4.out` 0 bad (`checks/revchecks_rt4.py` reads the exact zero `0^x`
at 1 as the public precision 4 and `internal Exact`, where `EXP/revchecks.py` reads the sentinel's
value), the debug budget tests 0 bad (`budget_rt4_debug.txt`). The calibration keeps every outcome
and every row's units (70 rows, identical units; its slowest call, the 2000 64th roots, 2.76 s and
3.25 s in two runs on the shared machine, 2.83 s in revision 8). The
heap peak per unit of the memory families is as in revision 8 (`memrt4_final.txt`, `famrt4_final.txt`);
red team 4's family of distinct long-monomial coefficients is the new largest, 2.83 bytes per unit for
an admitted call (§2.7). The scans: the slowest cell of `fnscan_rt4.txt` is sinh at 64 bits, 2.47e7
units/s (revision 8: 2.35e7); a first run on the busier machine scanned sinh(357/9) at 4.8e6 for one
argument, of which 2000 calls take 2.2 us at the median, 5.8e7 units/s (`bin/onecall_rel sinh 357 9
64 2000`), a preemption; the short-mantissa arguments of red team 4 give the slowest medians, 1.6e7
units/s (§2.3). The powers of
`powtime_rt4.txt` run at 8.2e7 units/s or more (revision 8: 8.3e7). Red teams 1 to 3 and review 8's reproductions keep every outcome
(`redteam1_rt4.txt`, `redteam2_rt4.txt` (through `tools/exps_rt4.py`, which reads the new
`internal` field of the `series` line), `redteam3_rt4.txt` and `review8_rt4.txt`, 0 outcome differences;
the hint margin changes some trees of review 8's families, most smaller (`F_10` at 1, order 1: 462 903
nodes, revision 8 468 023) and one larger (`sing_16` at 1, order 0: 221 nodes, revision 8 113)); review 8's 360 random hinted towers (`randhint_rt4.txt`): 130 `Ok` (revision 8: 103), of which 27 revision 8 refused and none it answered lost; 129 equal gp's series at 300 digits, and the 130th, `((-1) + sqrt((1/2) + sin(..)^2)^2)^(1/2)` around a power `(c)^(x+1)` with c < 0 at 1, order 1, differs from gp's series only by gp's branch: its inner constant term is exactly the real `c^2`, which gp computes as `exp(2 log c)` with an imaginary part of -1.4e-115, so gp's principal square root of the negative constant term is `-0.372 I`; with `c^2 exp((x-1) log c)` in place of `c^(x+1)` (the same function) gp gives `+0.372 I` and `0.0652 - 0.0118 I`, the engine's two coefficients (`EXP/rev7rt4/checks/rh131.gp`; §2.4, "Branch at the point"); the other 229 are errors in both revisions. Red team 4's reproductions are in
`rt4series_repro.txt`, `rt4num_repro.txt`, `rt4_ms.txt` and `rt4_extra.txt` (the table of "Revision 7,
red-team round 4" at the top). `TAYLOR_WORK` stays 1e8.

**Revision 7rt4 probes** (red team 4's reproductions on proto13, release on an 8 MB thread unless stated;
the three entry points agree wherever both are listed):

| probe | input (at 0 unless stated) | proto13 | revision 8 | HEAD |
|---|---|---|---|---|
| series H1 | `(x^(2^62)*x^(2^62-1))/x^(2^63-1)`, order 0 | `1`, `Exact` inside, 0.05 ms, 3.5 MB; limit path `Finite(1)`; 2 MB debug `Ok` | `0` claimed exact, `Finite(0)` | `0 ^ N` forms |
| series H1 | `(x^9223372036854775807 + 1)/x^9223372036854775807`, order 0 | `1/x^9223372036854775807 + 1` | the `+ 1` lost | `0 ^ N` forms |
| series H1 | `(x^(2^63-4)*exp(x))/x^(2^63-2)`, order 2 | `Unsupported("a series precision overflows i64")` after two attempts, 0.07 ms | `1/x^2 + 1/x + 1/2` claimed exact | `0 ^ N` forms |
| series H1 | `(((x+x^2)^9223372036854775806)/x^9223372036854775806 - 1)/x`, order 0 | `PrecisionExhausted(8388609)`, 0.2 ms; limit path `Unknown` | `0`, `Finite(0)` (gp: 9223372036854775806) | `... / 0` |
| series H1 | 900 random products and quotients near 2^63 (`maxfuzz_rt4.py`) | 222 `Ok`, each equal to the shifted input; 678 honest refusals | | |
| series H2 | `exp(x)/x^(2^61)` (15 bytes), order 0 | `BudgetExceeded` in 0.05 ms, 3.5 MB; 2 MB debug the same; without a budget `Unsupported` (the slot cap) | release panic "capacity overflow" | `(1 / (0 ^ 2305843009213693952))`, 15 us |
| series H2 | every recurrence over x^(2^k), k = 57 to 62; `exp(x)` at orders 2^57 - 1 to 2^63 - 2 | `BudgetExceeded`, at most 2 ms on 2 MB debug | panic from 2^60 slots | |
| numeric m1 | `x*atanh(1)`, `x + zeta(1)`, order 2 | `DivisionByZero`, limit path `Unknown (DivisionByZero)` | `Ok`, `Finite(0)` | `Finite(0)` |
| numeric m2 | `cosh(3/2^130)` at 256 bits (1001 calls) | 1.60e7 units/s (the slowest median) | 1.61e7 | |
| numeric m3 | `(1)/(x + sin(sqrt(1-1) + 1) - 1/2)`, order 1; the hinted 6-level tower at 1, order 0 | `Ok` in 0.15 ms and 0.3 ms | `UndecidedZero` | a series (the first with a `0/0` term) |
| memory m1 | `(-1)*(qa*...)/(1-x)^2`, 1000 names, order 1043 | `Ok`, 117.8 MB heap at 4.16e7 units (2.83 bytes per unit), VmHWM 122 MB, 0.11 s; 1045 `LimitExceeded(TreeSize)` | the same | does not finish |
| memory m2 | `limit_with_error` of `1 + x*S_17`, order 5 | `(Finite(1), Some(O(x^6)))`, 0.37 s, 69 MB | `(Finite(1), None)` | aborts at 4 GiB |
| structure m1 | `sin^2000(x+1)` at 0, order 1 | `Ok`, a tree of 2 007 003 (`render_tree` 2 007 011), 0.14 s, 10 MB, gp 1e-312 | `LimitExceeded(TreeSize)` | 0.46 s, 386 MB |
| structure m1 | 1143 random inputs (red team 3's), `TREECAP` = exact tree - 1 | refused every time; outcome kinds as revision 8 | | |
| structure m2 | `x^x^...^x` at 1/2, order 2: 21 and 22 carets | `Ok` (gp 3.0e-328) and `LimitExceeded(TreeSize)` in 0.06 s | 19 admitted | 22 carets: 7.6 ms, 11 MB |

**Revision 8** (review 8), release, HEAD 6e03987, budget 1e8, every call through a public entry point:
`EXP/rev8/tools/runall_r8.sh` (every check of revision 7rt3 rerun on `proto12`, then
`tools/review8_r8.sh` and `tools/randhint_r8.py`; 20 minutes in all); outputs in `EXP/rev8/` with the
suffix `_r8`, compared with revision 7rt3's by `tools/cmp_outcomes.py`. Every row keeps its revision
7rt3 outcome: the calibration (70 rows, every row's units within 0.5 %; the slowest call is the
refusal of the 2000 64th roots after 2.83 s, 2.72 s before, the charge of a non-literal exponent being
`4u`), the memory and family probes (`memr8_final.txt`, `famr8_final.txt`, `m2prober8_final.txt`,
`b2probe_r8.txt`), the self-attack (165 rows), the DAG and literal-chain families, the depth searches
(`enginedepth_r8.txt`, `depthcmp_r8_*.txt`, `newtests*_r8_dbg.txt`), the inputs of reviews 3 to 7, the
caps, the consumers of `F_k` (`cons3_r8.txt`), the scans, and red teams 1, 2 and 3 (red team 3's file
differs only where gp, now given a 1 GB stack, no longer overflows). The largest time and VmHWM of each
file are within noise of revision 7rt3's. The oracles: gp 107 OK and 0 failures (`gpcheck3_r8.out`),
`revchecks_r8.out` and `revchecks3_r8.out` 0 bad, the debug budget tests 0 bad (`budget_r8_debug.txt`,
the slowest 5.0 s in debug). The slowest scan cell is atan at 64 bits, 2.0e7 units/s, then sinh at
2.35e7 (in a first run of revision 8, on a busier machine, erf(130/29) scanned at 1.28e7 for one
preempted argument; 1000 calls of it take 15.6 us at the median, 1.2e8 units/s, `bin/onecall_rel`). The powers whose exponent has an exact integer
value run at 8.3e7 units/s or more (`powtime_r8.txt`). Review 8's reproductions, the seeded towers and
the random towers against gp are in `review8_r8.txt` and `randhint_r8.txt` (the table of "Revision 8"
at the top). `TAYLOR_WORK` stays 1e8.

**Revision 7rt3** (red team 3), release, HEAD 2e15d1c, the cost model of §2.7 with `SUBST_NODE` and
`HINT_UNITS`, budget 1e8, every call through a public entry point: `EXP/rev7rt3/tools/runall_rt3.sh`
(every check of revision 7rt2 rerun on `proto11`, then `tools/redteam1_rt2.sh`, `tools/redteam2_rt2.sh`,
`tools/redteam3_rt3.sh` and `tools/hintscan.py`; 19 minutes in all); outputs in `EXP/rev7rt3/`,
compared with revision 7rt2's by `tools/cmp_outcomes.py`. The machine was shared with another workload
for part of the run, so a few scan cells and HEAD's controls are slower than in revision 7rt2.

Calibration (`calibrt3_final.txt`, the 70 rows): every row keeps its revision 7rt2 outcome except one
refusal that changes its reason (`(1+sin(1))*...*(1+sin(k))` at order 1, refused by the budget at
1.83e8 before, now `LimitExceeded(TreeSize)` before the rendering); 68 rows are within 0.5 % of
revision 7rt2's units, and `(1)/(x + a balanced product of sines)` drops by 37 % (fewer attempts: the
doubling of a stalled loss-driven step, §2.6). The slowest call stops after 2.72 s (the 2000 64th roots,
as before), the slowest rate is 3.6e7 units/s (calls under 10 ms aside); `TAYLOR_WORK` stays 1e8.

**Revision 7rt3 probes** (every reproduction of the four reports of red team 3, those of red teams 1 and
2, and the families of reviews 3 to 7 and of the self-attacks rerun). The bar as before. Every outcome is
honest; no budgeted call ran longer than 2.72 s in release (4.5 s in debug) or held more than 230 MB of
VmHWM; no call of the engine overflowed a stack. Outcomes that changed against revision 7rt2, besides
the holes below: the sparse-rendering family of review 6 B3 (`sparseg`, 7 rows of
`review6_inputs_rt3.txt` and one row each of the attack, `newtests6`, `review7_fam_r6` and
`review7_inputs`) is refused by the tree cap where the budget refused it (983 040 stored terms); red team
2's long monomials (`(y0^2*...*y7999^2)/(1-x)` at orders 100 to 200 and its REPL line) are
`LimitExceeded(TreeSize)` (about 3.5e6 nodes; HEAD does not finish them); towers `x^x^...^x` at 2 render
their values at the point from the input (the result of 18 000 carets has depth 20 800, not 36 002,
and HEAD's recursive Display now prints it). Every other row of `attackrt3_final.txt`, `famrt3_final.txt`,
`memrt3_final.txt`, `m2probert3_final.txt`, `review3`-`review5` and red teams 1 and 2 keeps its outcome.
The oracles: gp 107 OK and 0 failures (`gpcheck3_rt3.out`), `revchecks_rt3.out` and
`revchecks3_rt3.out` 0 bad, the debug budget tests 0 bad (`budget_rt2_debug.txt` in `EXP/rev7rt3/`).
The tables below are `redteam3_rt3.txt` unless stated; times are release on an 8 MB thread.

| family | probe | outcome |
|---|---|---|
| series H1: empty arguments of negative precision | the 10 inputs of the red team (`EXP/redteam3-series/inputs/h1_*.txt`, at 0, 1 and infinity, order 2, 5 and 20 000) through the three entry points; the stored series; the limit path on 2 MB debug; the §2.12 parser and HEAD; `0^((sin(x))/x)` at orders 0 and 3; the controls | `NotLaurent` for each through `taylor_with_budget` and `series_expansion` (at most 0.03 s, 4 MB), `Unknown` through the limit path (the pole; `PrecisionExhausted(134)` for the order-20 000 input, whose pole at x^30000 lies above order 5), the same on 2 MB debug and through the §2.12 parser; HEAD prints `0` and `1 + x + x^2`; `0^((sin(x))/x)` is the exact 0 at orders 0 and 3; `exp((sin(x)-x)/x^2)` is `1 - x/6 + x^2/72` and `x^10*exp((sin(x)^2+cos(x)^2-1)/x^4)` is `O(x^3)` |
| structure H1: `0^r` and zero bases | `zb_1` to `zb_8` of the red team through the three entry points and on 2 MB debug; eight more zero bases | `1 - x^2/2`, `0`, `1`, `1 + x + x^2/2 + x^3/6`, `-3^(1/3)`, `1 + (x - 1)`, `x` (the red team's gp references; gp cannot expand `0^r`), and `UndecidedZero` for the undecided pole term of `zb_8`; each in at most 2 ms on 2 MB debug; the limit path `Finite` where the constant term is rational. `x + 0^(1/x)` and `x + 0^x` `NotLaurent`, `x + 0^(I + x)` `NotLaurent` (Re r = 0), `x + 0^(-1 + x)` and `x + (x*0)^(cos(4) + x)` `DivisionByZero` (cos 4 < 0), `x + 0^(y + x)` `UndecidedZero(y)`, `x + (x - x)^(sin(1) + x)` `x`, `(x - x + 2)^x` `2^x` |
| structure H2: result trees and their consumers (`cons3.sh`, `TREE_TOO`) | `F_k` at 1, k = 2 to 30, orders 0 and 1; `simplify`, `substitute` and `==` of the results, design and HEAD; values against gp (`checks/gpres.py`, at x = 1 + h + O(h^(n+1))); the REPL line of `F_10` at order 1; hinted families with bounded values (`(1+sin(...))^2` and `(1+atan(..))^3`, `(2+cos(..))^(1/2)` alternating, 6 to 10 levels at 1/3, `hinted_gp_rt3.txt`) | order 0: trees of 16 to 151 nodes (k = 30: 151, HEAD's result has 151), in at most 0.4 ms; `simplify`, `substitute` and `==` of them in at most 32 us (HEAD 6.5 us; revision 7rt2: 2.95 s and 3.84 GB for k = 22, 28.7 s for `==` at k = 30). Order 1: k = 10 a tree of 468 023 nodes (`simplify` 52 ms and 48 MB, `substitute` 60 ms, `==` 2.5 ms; revision 7rt2: 0.83 s and 960 MB), k = 11 1 046 643, k = 12 to 15 `LimitExceeded(TreeSize)` in at most 0.2 s, k >= 16 `BudgetExceeded`. gp: k = 2 and 3 at orders 0 to 2 equal to 1e-295; the hinted families equal to 1e-307 at orders 0 and 1. The REPL line of `F_10` at order 1: estimate 3.2 MB, text 1.6 MB, LaTeX 1.2 MB, 0.09 s (revision 7rt2: a 35 MB text) |
| numeric m1: asinh, acosh, atanh | the 9 inputs of the red team through the three entry points; the enclosures of 13 constants at 64 and 1024 bits; `fnscan`, `nearscan` (with asinh at `-2^k`, acosh at `L 2^k`), `onecall` | each `Ok` in at most 0.3 ms and 1.0e4 units (revision 7rt2: `UndecidedZero`), every coefficient equal to gp's where gp evaluates the constant (1e-309); the enclosures contain gp's values (`asinh(-2^600) + 400` = -16.5814555165, `(2^27 + 1) log 2 - 93032640` = 0.4375088078, ...); `acosh(1 + 2^-1000)` and `atanh(1 - 2^-1000)` too wide at 64 bits and enclosed at 1024; slowest scan rates 2.15e7, 2.60e7 and 4.68e7 units/s at 64 bits (asinh, acosh, atanh) |
| numeric m2: integer exponents above 2^32 | the red team's inputs; the enclosures of `(1 + 10^-9)^k - c` for k = 2^32 + 1, 2^33, -2^33, of `(1 + 10^-9)^(2^63 - 1)` and of `(-1 - 10^-30)^36893488147419103233 + 1` (beyond i64) | `Ok` each; the enclosures contain gp's `exp(k log(1 + 10^-9)) - c` (68.3298161434, 5372.2619249, -0.4998140317, -0.9998140317); `(1 + 10^-9)^(2^63 - 1)` has none (its value, 1.4e4005659578, is outside the window); the beyond-i64 power is too wide at 64 bits and -3.689348814809966796685e-11 at 256 bits (gp: -3.6893488148099667966850e-11) |
| memory m1: one shared constant per term (`tools/rt3mem/mr3.py`) | `(-1)*(a*b*...)/(1-x)` with 2700 names at orders 380, 1100 and 1120, with 1000 at 380 and 3000, with 8000 at 360, with 200 at 2000; `((1)/(a*...))/(1-x)` at 1040; `(a^2*...)/(1-x)` at 330; `L*((1)/(1-x))` and `(L*x)/(1-x)^2` with a 99 990-digit L (`lit_rt3.txt`) | the orders above about 400 with 2700 names (and the other large ones) `LimitExceeded(TreeSize)` in at most 0.04 s and 13 MB; order 380: `Ok`, 3.48e7 units, 1.9 MB of heap (one rendering of the constant); `L*((1)/(1-x))` at 6000: `Ok`, 9.84e7 units, 2.5 MB (revision 7rt2: 252 MB); `(L*x)/(1-x)^2` refused by the budget with at most 259 MB of heap (2.59 bytes per unit) and 147 MB of VmHWM |
| memory m3: the first-term limit path | `1 + x*S_17`, `1 + x*S_18`, `1 + x*S_12` through the limit path and at order 0 | `Finite(1)` each, at the units of the order-0 expansion: S_18 5.41e7 in 0.37 s (revision 7rt2: `Unknown (BudgetExceeded)` at 1.54e8), S_17 3.19e7 (revision 7rt2: 7.97e7) |
| memory m2 and m4 | a line of 400 calls of `taylor((1)/(1-x), x, 0, 20000)`; `x + ... + x` with 10 000 and 49 999 terms at 0 and 1 | unchanged: 2.26 GB in 7.4 s (§7 item 9); the chains at most 1.50 bytes per unit, 29 MB |
| series m1: a stalled loss-driven step | `((x+x^2)^1000)/x^999` at orders 1, 2, 5, 10, 40, 50 and 400 and through the limit path; `(sin(x)^1000)/x^999` at order 10; `sin(x)^(1000000)/x^1000000` and `(x+x^2)^(-1000000)*(x+x^2)^1000000` at order 2 | `Ok` at every order in at most 0.01 s and 1.05e6 units, every coefficient equal to gp's `x (1 + x)^1000` (10 and 40 terms); the limit path `Finite(0)`; the other three refused by the budget in at most 0.2 s (revision 7rt2: `PrecisionExhausted`, both honest) |
| series m2: the offsets of `inv` | `(1)/((1)/x^2 + x^9223372036854775806)` at 0, order 3, on 2 MB debug and release | `x^2 + O(x^4)` on both (revision 7rt2: a panic in debug) |
| structure m1: Drop of shared children (`bin/dropm`) | `twin`, `sib`, a chain and a left-deep sum with 2 x 10^5 levels on 2 MB debug, 10^6 on a 1 MiB release thread; HEAD's glue; the results of `F_30` at order 0 and `F_10` at order 1 | each dropped with at most 17 KB of stack, the 10^6 levels in at most 0.13 s; HEAD's glue overflows `twin` at 20 000 levels; the results with at most 13 KB |
| structure m2: integral floats | the red team's three inputs, `ROUNDTRIP=1` | the input and the result read back as printed (`100000000000000000000000.0`), the value kept |

**Revision 7rt2** (red team 2), release, HEAD 2e15d1c, the cost model of §2.7 with `RECORD_UNITS`,
budget 1e8, every call through a public entry point: `EXP/rev7rt2/tools/runall_rt2.sh` (every check of
revision 7rt1 rerun on `proto10`, then `tools/redteam1_rt2.sh` and `tools/redteam2_rt2.sh`; about
20 minutes in all); outputs in `EXP/rev7rt2/`, compared with revision 7rt1's by
`tools/cmp_outcomes.py`. The first minutes of the run shared the machine with a stray process of an
aborted run, so the calibration was timed again alone (`calibrt2_idle.txt`).

Calibration (`calibrt2_final.txt`, `calibrt2_idle.txt`, the 70 rows): every row keeps its revision 7rt1
outcome; 65 rows are within 0.5 % of revision 7rt1's units, and five move up by 0.9 % to 4.3 % (the
records of the work stack on balanced products and sums of sines: `RECORD_UNITS` per nested chain). The
slowest call stops after 2.75 s (the 2000 64th roots, as in revisions 5 to 7rt1), the slowest rate is 3.6e7 units/s;
`TAYLOR_WORK` stays 1e8.

**Revision 7rt2 probes** (every reproduction of the four reports of red team 2, every reproduction of red
team 1, and the families of reviews 3 to 7 and of the self-attacks rerun). The bar as before. Every
outcome is honest; no budgeted call ran longer than 2.75 s in release (3.9 s in debug) or held more than 265 MB of
VmHWM; no call of the engine overflowed a stack (the only overflows are the parser's, past its limits).

| family | probe | outcome |
|---|---|---|
| series S-B3: the binomial precision (`redteam2_rt2.txt`) | `(x+1+sin(y))^100*((1)/x^5)` at 0, orders 3 and 70, and the powers without the pole, against proto9 and gp | proto10: exponents -5 to 3 with precision 4 (and -5 to 70 with precision 71), the nine and the 76 coefficients equal to gp's, both powers without the pole precision 4, 0.02 s; proto9: -5 to -2 marked exact (the control) |
| series S-B2: the exponent span (`redteam2_rt2.txt`) | `ovf_span`, `ovf_heap` and its control through the three entry points; proto9 and HEAD on `ovf_span` | `x^-(8 * 10^18) + 2`, precision 4, in 0.1 ms; `ovf_heap` `Ok`; proto9 panics, HEAD answers in 1.1 ms |
| series S-B1: the sparse log (`redteam2_rt2.txt`) | the five inputs of the red team's table and `log_1000000_17` through `taylor_with_budget` and the limit path; the `exp` control; three sparse logs against gp | refused at 1e8 after at most 0.28 s (revision 7rt1: 8.4 to 78 s), VmHWM at most 21 MB; `exp` 0.18 s; the logs equal gp's (16, 32 and 7 coefficients) |
| structure H1: mixed nesting (`redteam2_rt2.txt`, `mixdepth_rt2.txt`) | the red team's inputs on one 2 MB debug thread through the three entry points; the REPL's 64 MiB debug thread up to 6000 levels; 100 KB of nine mixed shapes, the engine alone on 2 MB debug and 8 MB release; the deepest input of six shapes for the parser, the engine and both on one thread; values against gp | every input `Ok` on the thread that parsed it (`mix4_200` refused by the budget after 0.74 s); 6000 levels on 64 MiB in 0.18 s; 100 KB of every shape handled in at most 3.9 s debug and 1.2 s release; the parser's limits (216 and 4124 levels, the probe parser 266 and 4438) are those of parse and call on one thread; 12 values equal gp's (maxdiff 0) |
| numeric m1: too wide goes on (`redteam2_rt2.txt`) | the seven constants of the red team's table and the power, through the three entry points, every coefficient against gp; the balls at 64, 256 and 1024 bits; four exact zeros | every constant decided at 256 bits and every coefficient equal to gp's, in under 0.01 s; `TooWide` at 64 bits for `log(PQ - e) + 84`, a ball across 0 for `tan(10^18) + 1`; the exact zeros `UndecidedZero` |
| numeric m2, structure m2 and m3 (`redteam2_rt2.txt`) | the float literals; `x + sin(1+...+1)` with 20 000 and 5000 ones on 2 MB debug with `PartialEq`; ten huge monomial powers | `0.1 - 1/10` decided nonzero and `0.5 - 1/2` undecided (documented); the deep result `Ok` in 77 ms with depth 20 003, its `PartialEq` aborts on 2 MB debug as stated (5000 ones compare in 0.9 ms); the monomial powers as in §4.1 `exact_monomial_huge_power` |
| memory m1: long monomials (`redteam2_rt2.txt`) | `(y0^2*...*y7999^2)/(1-x)` at orders 100, 106, 107, 200 and N = 2000 at 100 and 200, through `taylor_with_budget` and `series_expansion`; the REPL line at order 100 | at most 2.55 bytes per unit at N = 8000 (253 MB at 9.94e7 units, VmHWM 261 MB), 2.60 at N = 2000, the REPL line 265 MB of VmHWM, all as with proto9 |
| structure m1: the Display round trip (`redteam2_rt2.txt`) | the four inputs, gp reading the text; the round trip of 8 x 400 random inputs with proto10's Display and with proto9's | gp reads every text as the value (maxdiff 0; revision 7rt1 0.42 to 0.51); 944 of 954 trees keep their value, 0 differ, 10 not evaluable by gp; proto9: 78 differ |
| memory of the work stack (`workstack_mem_rt2.txt`, the counting build, parse on a 1 GiB thread) | 30 000 nested negations, `sin(...(1))`, right-deep sums and calls `f(...(1))`, 11 000 levels of `mix5cs`, 3000 of `mix4` | at most 2.16 bytes per unit (the nested sines: 21 MB at 9.9e6 units), the others 0.05 to 1.37 |
| correctness of chains and towers (`checks/cases1_rt2.out`, `checks/rand1_rt2.out`) | the red team's 31 hand cases and 74 random inputs in the §2.12 grammar, against gp (`GPFORM=1`) | every coefficient equal to gp's to 60 digits (maxdiff at most 7e-69), except the two values of size 1e120 and 1e214 that agree to 76 relative digits, as with proto9; the same outcomes as proto9 on all 121 |
| red team 1, every reproduction (`redteam1_rt2.txt`) | towers, chains, deep results, `PartialEq`, the REPL taylor line, the writers, the near points, the literal bound, the binomial budget | every outcome as in revision 7rt1; the engine searches reach their upper ends (50 000) on every tower and chain; `PartialEq` compares depth 5892 on 2 MB debug; the writers 0 differences on 368 expressions; the scans of §2.3 as before |
| the engine's depth (`enginedepth_rt2.txt`, `depthcmp_rt2_rel.txt`, `depthcmp_rt2_dbg.txt`) | the six API-built shapes and the tower; the ten parsed shapes against HEAD | the engine handles at least 59 996 levels of each on 2 MB debug and 199 996 on 8 MB release (the searches' upper ends); parsed, the same outcomes as revision 7rt1 (the parser stops first; HEAD's `taylor` stops at the same level or earlier) |
| everything else rerun (`attackrt2_final.txt`, `review*_rt2.txt`, `famrt2_final.txt`, `memrt2_final.txt`, `m2probert2_final.txt`, `dagrt2_final.txt`, `litchainrt2_final.txt`, `leftdeeprt2_final.txt`, `attack_deep2_rt2.txt`, `attack_nkx_rt2.txt`, `b2probe_rt2.txt`, `caps_rt2.txt`, `newtests_rt2_dbg.txt`, `newtests6_rt2_dbg.txt`, `sharecheck_rt2.txt`, the scans) | as in revision 7rt1 | 0 outcome differences against revision 7rt1 in every file (`cmp_outcomes.py`); the slowest budgeted call 2.7 s; the scans: slowest rate 2.1e7 units/s on the grid (atan at 64 bits), 2.5e7 at the median near the precision-doubling points |
| oracles (`gpcheck3_rt2.out`, `revchecks_rt2.out`, `revchecks3_rt2.out`, `budget_rt2_debug.txt`) | the gp oracle, the revision 1 and 2 checks, the 31 budget inputs | 107 OK and 0 failures; 0 bad; all 31 outcomes as before |

**Revision 7rt1** (red team 1), release, HEAD 2e15d1c, the cost model of §2.7 with the revision 7rt1
charge of sin, budget 1e8, every call through a public entry point, on an idle machine:
`EXP/rev7rt1/tools/runall_rt1.sh` (every check of revision 7 rerun on `proto9`, then
`tools/redteam1_rt1.sh`; about 15 minutes in all), and `tools/partialeq_rt1.py`; outputs in `EXP/rev7rt1/`, compared with
revision 7's by `tools/cmp_outcomes.py`.

Calibration (`calibrt1_final.txt`, the 70 rows): every row keeps its revision 7 outcome, and six rows
move by more than 0.5 %. `x + (3^82000)/(5^50000)*2*...*29` is 19.9 % lower (the `/` joined the product
chain: one `product_many` of all factors, not a quotient and then a product); `(1)/(1+x+x^2)` at 2000,
`(1)/(x+sin(1)+I)` at 400 and `(1)/(1-x-...-x^7)` at 2000 are 3.9 %, 2.5 % and 0.6 % lower (a series
divisor is inverted without a product by the exact constant 1); a sum of sines in a divisor is 4.0 %
higher (sin at 256 and 1024 bits). The slowest call stops after 2.72 s (the 2000 64th roots, 3.7e7
units/s, as in revision 7); `TAYLOR_WORK` stays 1e8. The scans of §2.3: slowest rate 2.6e7 on the grid,
2.5e7 at the median near the precision-doubling points.

**Revision 7rt1 probes** (every reproduction of the four red-team reports, and the families of reviews 3
to 7 and of the self-attacks rerun). The review bar as in revision 7: an input of at most 100 KB through
`taylor_with_budget` with budget 1e8, through `series_expansion` with its budget, or through the
`limit_via_series` path must not run longer than 20 s in release, use more than 512 MB, or overflow an
8 MB release thread or a 2 MB debug thread. Every outcome is honest; no budgeted call ran longer than
2.8 s or held more than 256 MB of VmHWM; no call of the engine overflowed a stack.

| family | probe | outcome |
|---|---|---|
| structure H1: towers (`redteam1_rt1.txt`) | `x+1^1^...^1` with 400 carets, parse and call on one 2 MB debug thread, through the three entry points; 12 000 to 22 000 carets on 8 MB release; 13 000 on the REPL's 64 MiB debug thread; `x^2^...^2` (500) and `x^x^...^x` (700) at 1 on 2 MB debug | `Ok` in 1.5 ms (46 060 units; the limit path `Finite(1)`; HEAD 0.4 ms); `Ok` in at most 10 ms with at most 18 MB; `Ok` in 42 ms; `Ok` in 5 ms and 8 ms (revision 7: stack overflows) |
| | the parser alone and the engine alone (`bsearch.py parse` and `engine`), five tower shapes | the parser reads 1438 carets on 2 MB debug (1427 for `sin(1)^...`) and 22 695 on 8 MB release; the engine handles 50 000 (100 KB) of each on both threads, at most 0.70 s and 0.15 s; built with the API, 50 000 levels of six shapes on 2 MB debug, at most 0.86 s (`x^-x^-...` at 1) |
| structure H2: deep results (`redteam1_rt1.txt`, `partialeq_rt1.txt`) | `x^x^...^x` at 2, order 0, 16 000 to 20 000 carets, 8 MB release, Display with the work list and `PartialEq`; the same 18 000 with HEAD's recursive Display; 4500 and 5500 on 2 MB release; 1438 and 22 695 carets (the deepest a text gives) on one thread with parse, call, Display and `PartialEq` | depth 32 002 to 40 002, printed in at most 3.6 ms, compared in at most 1.0 ms; HEAD's Display aborts (the control); printed; depth 2878 (debug 2 MB) and 45 392 (release 8 MB), all four steps on one thread, 31 ms and 94 ms for the call |
| | `PartialEq` alone, `x^2^...^2` at 1 on 2 MB debug | compares up to depth 5892 |
| towers with nested bases (self-check, `depth_extra_rt1.txt`) | `((x^2)^2)^...` and `((x^x)^x)^...` at 1, `((x)^(x^x))^...` at 2, 2 MB debug | parse and call on one thread: the parser stops first at 214 levels of each; the engine alone handles 536, 436 and 560 |
| structure H3: `/` chains (`bsapi.py`, API-built in §2.12's structure) | `x/2/.../2`, `x + (1/2/.../2)`, `1/x/.../x` at 1, `(x + sin(1))/sin(1)/...` with 50 000 divisions, `x*3/2*3/2...` with 25 000 pairs, 2 MB debug and 8 MB release; through `series_expansion` and the limit path; HEAD's parse of `x+(1/2/.../2)` with 12 000 slashes | `Ok`, at most 0.54 s debug and 0.09 s release; the same outcomes (the limit of `1/x^50000` at order 5 refused by the budget, the others `Finite(..)`); 3 ms |
| memory H1: the REPL taylor line (`replshow.rs`) | `fchain_33000` at order 100 with the work-list writers, and with HEAD's `expr_to_latex`; `sinchain_20000` at 51; the API-built chain of depth 33 000 in debug on a 256 MiB thread; depth 8000 at order 400 on the REPL's 64 MiB debug thread; two `CHECKSAME` inputs | 0.33 s in all, VmHWM 139 MB (HEAD's writer: 31.9 s of LaTeX); 0.12 s; 1.17 s (HEAD's writer 35.9 s); 0.99 s; the same strings |
| writers (`displaycheck.rs`) | the work-list Display and LaTeX writers against HEAD's on 368 expressions | 0 differences |
| numeric m1, m2 (`nearscan.rs`, `litbound.rs`) | the near points at 64, 256 and 1024 bits; the 4096- and 8192-group `sin(L) + 1` inputs; the literal bound | slowest median 2.5e7 units/s (log at 256 bits; sin 3.4e7 to 6.2e7); both refused at 1e8 after 0.72 s and 0.79 s; 34 ns, 0 bad bounds of 210 |
| series m3 | the binomial path, `(x + sin(1)^2 + cos(1)^2 - 1)^2000` at 1200 and 3000, `^100000` at 1500 | refused at 1.0e8 within 0.17 s with at most 1.2 MB |
| depth of the six nested shapes and the tower (`enginedepth_rt1.txt`, `depthcmp_rt1_rel.txt`, `depthcmp_rt1_dbg.txt`) | the engine alone on API-built inputs; the ten parsed shapes against HEAD | the table of §2.5; parsed, the same outcomes as revision 7 on every shape (the parser stops first) |
| long literals (found while revising) | `L*((1)/(1-x))`, `(L)/(1-x)` at orders up to 6200, `(L)/((1-x)*x^k)` for k up to 3900, `(L*x)/(1-x)^2` up to 1400, L of 99 990 digits | `Ok` up to the budget, at most 2.60 bytes per unit, 252 MB of heap and 256 MB VmHWM at 9.84e7 units (§2.7); order 6200 refused at 1.02e8 |
| everything else rerun (`attackrt1_final.txt`, `review*_rt1.txt`, `famrt1_final.txt`, `memrt1_final.txt`, `m2probert1_final.txt`, `dagrt1_final.txt`, `litchainrt1_final.txt`, `leftdeeprt1_final.txt`, `b2probe_rt1.txt`, `caps_rt1.txt`, `newtests_rt1_dbg.txt`, `newtests6_rt1_dbg.txt`, `attack_deep2_rt1.txt`, `attack_nkx_rt1.txt`) | the families of reviews 3 to 7 and the self-attacks of revisions 5 to 7 | the outcomes of revision 7 on every line except two long-literal inputs now `Ok` (above: `(L)/((1-x)*x^1000)` at 4.92e7 units with 126 MB, `((L)/((1-x)*x^7000))*0` at 5.1e6); the slowest budgeted call 1.0 s in release (review 6's `srch_256_1024`, refused at 3.7e8) and 1.4 s in debug (the DAG inputs); the unbudgeted probes as in revision 7 (the 40 binomial powers `LimitExceeded(Kernels)` after 107 s; `exp(x)^2/x^1000000` without a budget exhausts memory, the reason for `LIMIT_SERIES_WORK`, §2.13) |
| oracles (`gpcheck3_rt1.out`, `revchecks_rt1.out`, `revchecks3_rt1.out`, `budget_rt1_debug.txt`, `sharecheck_rt1.txt`) | the gp oracle, the revision 1 and 2 checks, the 31 budget inputs, the sharing check | 107 OK and 0 failures; 0 bad; all 31 outcomes, the slowest 5.3 s debug; one allocation for the 5001 leaves Y |

**Revision 7** (review 7, B1, B2, m1 to m4, and the families of reviews 3 to 7), release, HEAD f29c492,
the cost model of §2.7 with the revision 7 charges (the erf rule, the result floor), budget 1e8, every
call through a public entry point: `EXP/rev7/tools/runall7.sh`, 12 to 14 minutes, outputs in
`EXP/rev7/` (the file of each row is given; `EXP/rev7/run1/` and `run2/` hold two earlier full runs,
whose outcomes are the same). The machine ran a workspace test in parallel during the first run, so
times are upper bounds and rates lower bounds.

Calibration (`python3 EXP/rev7/tools/calib6.py EXP/rev7/bin/proto8_rel 100000000`,
`EXP/rev7/calib7_final.txt`, the 70 rows of revisions 5 and 6): every row keeps its revision 6 outcome;
68 rows are within 0.5 % of revision 6's units, `(1+x)^10000` at 2000 is 1.6 % higher and the 40 kernels
`sin(2^340000 + k)` 0.9 % higher (the result floor, m4; no row of the table has an erf). The slowest
rate is 3.5e7 to 3.8e7 units/s over three runs (the 200 and 2000 64th roots, as in revisions 5 and 6),
and the slowest call stops after at most 2.76 s. `TAYLOR_WORK` stays 1e8 by the rule below. The
enclosure scan of §2.3 finds slower single operations, 2.1e7 units/s at worst (atan at 64 bits) and
4.0e7 for erf, still above the floor of 1e7.

Memory per unit (`fam7_final.txt`, `mem7_final.txt`, `m2probe7_final.txt`, `b2probe_r7.txt`, the counting
build through `taylor_with_budget`, and through `series_expansion` with `to_expr` where it says so):
the heap peak is at most 2.29 bytes per unit (`x + S_19*sin(x)`, 227 MB at 9.92e7 units; building S
2.21, `x + Q_1400*(S_15)*0` 2.19, the families and inputs of reviews 6 and 7 at most 2.0), and VmHWM
at most 230 MB on every budgeted probe (`x + S_19*sin(x)`, `caps7.txt`; next 211 MB, the review 6
input `srch_256_1024`, refused at 3.7e8 after 1.0 s).

**Revision 7 probes** (the families of reviews 3 to 7 and of the self-attacks of revisions 5 and 6, on
the revision 7 prototype; about 760 runs, plus about 500 in the binary searches of the depth probes).
The review bar: an input of at most 100 KB through `taylor_with_budget` with budget 1e8, through
`series_expansion` with its budget, or through the `limit_via_series` path must not run longer than 20 s
in release, use more than 512 MB, or overflow an 8 MB release thread or a 2 MB debug thread. Every
outcome is honest (`Ok` with the gp-checked algorithm, or a typed refusal); no budgeted call ran longer
than 2.8 s or held more than 230 MB; no call of the engine overflowed a stack (the two overflows are in
HEAD's parser, which also stops HEAD there, as in revision 6). Release on an 8 MB thread unless it says
debug (2 MB thread).

| family | probe | outcome |
|---|---|---|
| review 7 B1: erf groups (`attack7_final.txt`, family `rev7`; `review7_inputs_r7.txt`) | `1/(x + (y1 + ... + yn)*(erf(a) + erf(-a)) + y0)`: n = 3700 and 3600 with a = 975/38 (12.0 and 11.6 KB); n = 100 with a = 26, 975/38, 960/38, 586/25, 280/11; the sum form with n = 3700 (112 KB); `erf(x + 975/38)` in the groups | refused after 0.15 s (debug 0.21 s), within 0.17 s, after 0.16 s, after 0.15 s; at most 14 MB. Revision 6: `Ok` after 32.1 s (review 7). Through the limit path: `Unknown` (budget) within 0.15 s |
| | n = 3 | `Ok` in 0.03 s, 1.86e7 units, c0 = 1/y0 (gp, §4.1 `budget_erf_groups`) |
| erf at the edges (`rev7`) | `1/(x + erf(a) - 1)` for a = 7, 15/2, 29/4, 13, 27/2, 14, 26, 975/38 | `Ok` in at most 4.4 ms, decided at 64, 256 or 1024 bits (`erf(26)`: 3.18e6 units) |
| | a = 27, 55/2, 28 | `UndecidedZero` in 0.1 ms (erfc(a) is below 2^-1024 there; MPFR's erf rounds to 1 as well) |
| | `1/(x + sum of k terms (erf(sin(j)^2 + cos(j)^2) - erf(1)) + y)` (computed arguments, each term exactly 0), k = 1, 10, 30, 100 | `Ok` (3.6e6 and 3.6e7 units); refused after 2 ms and 5 ms (each erf node of a computed argument is charged 3.41e6 at 1024 bits) |
| | `erf(x)` at 26 and at 975/38, order 30; `1/(erf(x) - 1)` at 26, order 3; `x + erf(1/77) + ... + erf(2000/77)`; `1/(x + that sum - 1)` | `Ok` in 0.2 ms, 0.4 ms, 1.4 ms, 4 ms and 14 ms |
| review 7 B2: long names (`b2probe_r7.txt`, `sharecheck_r7.txt`) | Y of 99 990 characters, order 2000 (100 KB): `Y/(1-x)`, `(Y)/((1-x)*x^3000)`, `(Y)/((1-x)*x^10000)` through `taylor_with_budget`, through `series_expansion` then `to_expr`, and through the limit path | `Ok` in at most 0.02 s with 1, 2 and 5 MB of heap through both (after `series_expansion` alone 1.0, 1.9 and 3.7 MB); the limit path `Unknown` (a coefficient that is not rational, a pole) with at most 3 MB. Review 7's model of revision 6: 396 MB, 983 MB, 2.35 GB. The 5001 leaves Y of the second share one allocation |
| | `Y` alone at order 5 (one shared coefficient: `to_expr` copies its top node once) | `Ok`, 0.3 MB of heap at 1.38e5 units (2.2 bytes per unit, the kernel's charged copy of the name and the returned copy included) |
| review 7 m4 (`rev7`, `b2probe_r7.txt`) | L a 99 990-digit literal, order 2000: `(L)/((1-x)*x^1000)`; `((L)/((1-x)*x^7000))*0`; `(L)/(1-x)`; `(L*x)/(1-x)^2` at order 300 | refused at 1.12e8 after 0.15 s with 126 MB (review 7: `Ok` with 251 MB); refused after 0.17 s with 130 MB (the limit path 164 MB); `Ok` in 0.15 s with 167 MB; `Ok` in 0.03 s |
| review 7 m1: order 20000 (`rev7`) | `x + 1`, `x^2`, `2*x+1`, `1/(1-x)`, `1/(1-x)^2`, `exp(x)`, `sin(x)` | `Ok` in 16 us to 19 ms (at most 14 MB); `exp(x)` and `sin(x)` refused after 0.14 s and 0.12 s with 123 and 105 MB. HEAD: `x + 1` in 0.48 s with 324 MB |
| review 7's rerun of review 6 (`review7_fam_r6.txt`) | `fz(64, 186)`, `fz(64, 32)`, `fz(64, 8)`, `two(64, 4)`, `negz(64, 8)`, `sparseg(60, 4)`, `sparseg(60, 1)`, `negC(4096)` at 560, `kterm` with 3 and 11 terms | the outcomes of review 7's run: refused within 0.31 s, `fz(64, 8)` `Ok` in 0.11 s with 38 MB of heap, `sparseg(60, 1)` `Ok` with 126 MB; at most 183 MB |
| review 6 inputs (`review6_inputs_r7.txt`, `review6_invP_r7.txt`) | the 49 files of `EXP/review6/inputs` (the m2 products, `fz`, `two`, `negz`, `sparse`, `spg`, `negC`, `invpow`, `srch`) | every outcome as revision 6: refused within 1.0 s or `Ok` within 0.23 s, at most 211 MB. The three `1/(x + P_k)` files do not parse with HEAD's parser (it cannot read `1/(`); written `(1)/(x + P_k)` they are `Ok` within 0.07 s |
| review 5 (`review5_repro_r7.txt`, `review5_depth_r7.txt`) | `repro_r4.py` (K, the B1 copies, the B2 keys, products and shared trees); `longmono_rate.py`, `radrem.py`, `mem_mono2.py`, `mem_mono3.py` (long monomials, radicals); `probe.py` and `probe2.py` (depth, debug 2 MB) | `Ok` or refused within 0.33 s, at most 119 MB; the shared-tree numeric tests `UndecidedZero` in 0.2 ms or less (their balls leave the window). Two inputs of `longmono_rate.py` and `radrem.py` are 200 KB and more, above the bar, and do not fit the scripts' command line. Depth: the parser stops first at 159 to 162 nested calls (92 for `(2+sin(...))`); `x + P_k` reaches `LimitExceeded(MonomialFactors)` past 32 768 factors (honest; 654 KB inputs) |
| review 4 (`review4_repro_r7.txt`) | `repro4.py` (B1 copies of K, B2 keys and products) | `Ok` within 0.33 s or refused at 1.62e8 after 0.24 s; at most 119 MB |
| review 3 (`review3_repro_r7.txt`) | `repro_blockers.sh` (rational normalization, radicals) | `Ok` or refused within 1.04 s |
| self-attack of revisions 5 and 6 (`attack7_final.txt`, `attack7_deep2_final.txt`, `attack7_nkx_final.txt`, `leftdeep7_final.txt`, `dag7_final.txt`, `litchain7_final.txt`) | the families `rev6`, `long`, `kernels`, `wide`, `literal`, `deep`, `radical`, `order`, `pole`, the DAG inputs (k = 20, 40, release and debug 2 MB) and the literal chains | the outcomes of the revision 6 table below, with times within the noise: the slowest budgeted call at most 2.75 s (2000 64th roots), the largest VmHWM 183 MB; the 40 binomial powers without a budget `LimitExceeded(Kernels)` after 107 s (unbounded); `invpowx(4)` without a budget `LimitExceeded(ConstantTerms)` after 2.5 s with 674 MB (unbounded) |
| depth (`depthcmp7_rel_final.txt`, `depthcmp7_dbg_final.txt`, `enginedepth_r7.txt`) | the ten shapes against HEAD; the engine alone on API-built inputs | as revision 6: release 8 MB, the prototype reaches HEAD's parser limit on every shape (4096, 2048 for `((..)*2+1)` and N_k) and HEAD's `taylor` stops earlier on two (364, 228); debug 2 MB, at least HEAD's depth on every shape except nested calls, where both stop in the parser at 160 to 164. The engine alone: 1294, 395, 1298, 647, 357 and 319 levels on 2 MB debug (395 for `x + sin^k(1)`, revision 6 365: the symbol arm of `from_expr` left its frame), 40 279 to 10 908 on 8 MB release |
| oracles (`gpcheck3_r7.out`, `revchecks_r7.out`, `revchecks3_r7.out`, `budget7_debug.txt`, `newtests7_dbg.txt`, `newtests6_r7_dbg.txt`) | the gp oracle, the revision 1 and 2 checks, the 31 budget inputs, the tests of revisions 5 and 6 | 107 OK and 0 failures; 0 bad of 26 + 10 and of 18; all 31 outcomes, the slowest 5.3 s debug; every outcome of revisions 5 and 6 |

**Revision 6** (review 6, B1 to B3, and the self-attack), release, HEAD 92906d4, the cost model of
§2.7 with the revision 6 charges, budget 1e8: `python3 EXP/rev6/tools/calib6.py
EXP/rev6/bin/proto7_rel 100000000`, output `EXP/rev6/calib6_final.txt`, the 70 rows of revision 5.
The units of a row are 1.0 to 2.0 times revision 5's (the constructor's `CONST_BASE`, the larger
copy and node constants, `SER_TERM`, `ACC_ENTRY`), and smaller where a zero factor now ends a chain
or a rendering shares the kernels' nodes. Three rows change outcome: `x + Q_1400*(S_15)*0` (refused,
now `Ok` with 6.35e6 units: review 6 m2), `1/(x + P_4096*S_6)` (refused at 1.05e8, now `Ok` with
6.5e7 units in 0.47 s and 113 MB: the zero test no longer copies a constant without indeterminates)
and `tan(x)` at 1, order 100 (`Ok` with 8.2e7, now refused at 1e8 after 1.09 s; order 98 is `Ok`,
9.9e7 units, 0.87 s). Selected rows:

| f | point | n | result | units | time | units/s | peak |
|---|---|---|---|---|---|---|---|
| exp(x) | 0 | 2000 | ok | 4.97e6 | 10 ms | 5.2e8 | 9.3 MB |
| exp(sin(x)), tan(x) | 0 | 250 | ok | 3.99e7, 4.04e7 | 0.33 s, 0.41 s | 1.2e8, 9.9e7 | 4 MB |
| (5/7+x)^(-3/11) | 0 | 2000 | ok | 8.89e7 | 0.96 s | 9.3e7 | 12 MB |
| tan(x) | 1 | 100 / 200 | refused / refused | 1e8 | 1.09 s / 1.06 s | 9.2e7 / 9.4e7 | 5 MB |
| `(K + x + ... + x)*0`, 400 `+x` | 0 | 1 | ok | 9.22e7 | 0.29 s | 3.2e8 | 119 MB |
| `x + Q_1400*(S_15)*0` | 0 | 1 | ok | 6.35e6 | 46 ms | 1.4e8 | 17 MB |
| `1/(x + P_4096*S_6)` | 0 | 1 | ok | 6.54e7 | 0.47 s | 1.4e8 | 113 MB |
| `x` plus 200 64th roots of `2^4095 + k` | 0 | 1 | ok | 9.71e7 | 2.58 s | 3.8e7 | 4.4 MB |
| the same with 2000 roots | 0 | 1 | refused | 1e8 | 2.88 s | 3.5e7 | 6.5 MB |
| 1/sin(x)^2000, 1/(exp(x)-1)^1000 | 0 | 2 | refused | 1e8 | 0.72 s, 1.01 s | 1.4e8, 9.9e7 | 26 MB, 9 MB |

The slowest rate is 3.5e7 to 3.7e7 units/s over two runs (the 64th roots, as in revision 5), and the
slowest call stops after at most 2.9 s. `TAYLOR_WORK` stays 1e8 by the rule below.

Memory per unit (revision 6, review 6 B1 to B3). The prototype driver built with `--cfg memcount`
(`EXP/rev6/bin/proto7m_rel`, `EXP/rev6/proto7/memcount.rs`, a counting global allocator) prints the
heap peak of the expansion and its rendering, in malloc chunks, above the live heap before the call
(so the parsed input is excluded); `EXP/rev6/tools/fam6.py`, `mem6.py` and `m2probe.py`, outputs
`fam6_final.txt`, `mem6_final.txt`, `m2probe_final.txt`. The heap peak is at most 2.21 bytes per unit
on every family (building S: 120 MB at 5.4e7 units; `x + Q_1400*(S_15)*0`: 13.9 MB at 6.35e6, 2.19),
and at most 1.9 on the families of review 6. VmHWM is at most 183 MB on every probe of this section
that has a budget (`negC` at order 560, refused at 1e8). The same inputs on the revision 5 prototype
(`EXP/rev6/bin/base6m_rel`, revision 5's code with the counting driver; `EXP/rev6/review6_inputs_final.txt`):

| family (review 6) | input | revision 5 | revision 6 |
|---|---|---|---|
| sparse product (B1) | `fz(64, 32)`, 2.4 KB | `Ok` after 2.45 s, 452 MB (18.1 bytes per unit) | refused after 0.03 s, 8 MB |
| | `fz(64, 186)`, 4.6 KB | `Ok` after 15.5 s, 2.53 GB (review 6) | refused after 0.06 s, 8 MB |
| | `fz(64, 8)` | | `Ok` in 0.12 s, 40 MB (0.66) |
| 2-term coefficients (B2) | `two(64, 4)`, 4.0 KB | `Ok` after 2.13 s, 1.35 GB (13.8) | refused after 0.21 s, 58 MB |
| | `two(64, 1)`; `kterm(16, 1, 11)` (11 terms per coefficient) | | `Ok` in 0.22 s, 135 MB (1.71); `Ok` in 0.27 s, 115 MB (1.22) |
| negated one-term coefficients (B2) | `negz(64, 8)`, 2.0 KB | `Ok` after 1.27 s, 574 MB (5.9) | refused after 0.31 s, 102 MB |
| | `negz(64, 2)` | | `Ok` in 0.21 s, 143 MB (1.53) |
| | `-((sin(1)+...+sin(4096))/(1-x))` at order 560, 40 KB | refused at the render charge, 466 MB | refused after 0.22 s, 183 MB (1.77) |
| rendering (B3) | `sparseg(60, 4)`, 1.4 KB | `Ok` after 1.18 s, 743 MB (7.5) | refused after 0.12 s, 35 MB |
| | `sparse1(3)`; `sparse(2)` (review 6's generator) | `Ok`, 595 MB (7.7); `Ok`, 574 MB (5.9) | refused after 0.09 s, 29 MB; refused after 0.24 s, 146 MB |
| | `sparseg(60, 1)` | | `Ok` in 0.10 s, 126 MB (1.66) |
| long names (found) | `(1 + Y*x + Z*x)^300` at order 300, Y and Z of 40 000 characters | `Ok` after 1.72 s, 3.68 GB (127) | `Ok` in 0.08 s, 49 MB (1.17) |
| | `(Y + x)^2000` at order 2000, a 90 000-character Y | `Ok`, 187 MB at 2.2e6 units (85) | `Ok`, 7 MB (0.80) |
| literal chains (found) | `x + L + ... + L` (API), 25 000 references to one 50 000-digit literal | | refused after 0.09 s, 137 MB (1.4; with revision 5's charge of 1 unit per word of a literal: 519 MB) |

**Revision 6 probes** (the self-attack of the orchestrator's brief, item 4). The families of reviews
3 to 6 and new ones, at the review bar: an input of at most 100 KB through `taylor_with_budget` with
budget 1e8 must not run longer than 20 s in release, use more than 512 MB, or overflow an 8 MB release
thread or a 2 MB debug thread. About 560 runs (`EXP/rev6/tools/runall6.sh`; outputs
`attack6_final.txt`, `attack6_deep2_final.txt`, `attack6_nkx_final.txt`, `leftdeep_final.txt`,
`dag_final.txt`, `litchain_final.txt`, `fam6_final.txt`, `mem6_final.txt`, `m2probe_final.txt`,
`calib6_final.txt`, `depthcmp_rel_final.txt`, `depthcmp_dbg_final.txt`, `newtests_dbg.txt`,
`newtests6_dbg.txt`, and the oracles `gpcheck3_r6.out`, `revchecks_r6.out`, `revchecks3_r6.out`,
`budget6_debug_r6.txt`). Release on an 8 MB thread unless it says debug (2 MB thread); HEAD is HEAD's
`taylor` on the same input (`EXP/rev6/proto7/head.rs`). Every outcome is honest: `Ok` with the
gp-checked algorithm (107 OK, 0 failures), or a typed refusal. No budgeted call ran longer than 2.9 s
or held more than 183 MB, and no call of the engine overflowed a stack; the only overflows are in
HEAD's parser, which also stops HEAD there (`EXP/rev6/probes/parseonly.rs`: `x + N_150` on a 2 MB
debug thread and 3000 levels of `((..)*2+1)` on an 8 MB release thread overflow in `parse` alone).

| family | probe | outcome |
|---|---|---|
| review 6 families | `fz`, `two`, `kterm`, `negz`, `negC`, `sparseg`, `sparse1`, `negsparse` (the table above) | refused within 0.31 s, or `Ok` in at most 0.27 s; at most 183 MB, at most 1.9 bytes per unit |
| long names | `(1 + Y*x + Z*x)^300` (80 KB of names) at 300; the same `^2000` at 60; `exp(Y*x)` at 2000 (80 KB); `x + Y` (+ Display, 90 KB); `f_long(1) + x` (90 KB function name) | `Ok` in at most 0.08 s, at most 49 MB; Display of `x + Y` prints 90 KB once |
| | `1/(x + sin(Y) + 2)`, 90 KB name | `UndecidedZero` in 0.2 ms (Y is an indeterminate and `sin(Y)` an `Opaque` value) |
| long monomials | `x + P_4096`, `1/(x + P_4096 - 1/2)`, debug | `Ok` in 0.10 s and 0.20 s, result depth 16 and 21, `format!` 56 KB and 113 KB |
| | `x + P_8800`, `1/(x + P_8800 - 1/2)` (104 KB) | `Ok` in 0.04 s and 0.07 s, depth 17 and 22 (HEAD: 0.10 s, depth 17) |
| | balanced product of 13 000 symbols (75 KB); `x + P_4096*P'_4096` | `Ok`, 0.03 s, depth 17, both |
| | `x + (P_4096 + 1)^64`; `1/(x + P_4096*S_6)` | refused at 1.06e8 after 0.02 s; `Ok` in 0.45 s, 113 MB |
| | `x + Q_1400*(S_15)*0` and `x + Q_1400*(S_15)`, both splits | `Ok` in 0.03 s; refused at 1.29e9 and 2.43e9 after 0.03 s |
| many kernels | `x + sin(1) + ... + sin(11000)` (110 KB, left-deep) | `Ok`, 0.05 s, 18 MB (HEAD 0.03 s) |
| | balanced sums of 11 000 sines and of 20 000 symbols; `1/(x + the symbols)` | `Ok` in 0.04 s, 0.05 s and 0.06 s, depth 17 |
| | `1/(x + sin(1) + ... + sin(9000) - 5)` (numeric test of a 9000-term constant) | `Ok`, 0.07 s, 20 MB |
| | 6 binomial powers `(x + sin(k)^2 + cos(k)^2 - 1)^1000000` at order 2000 | refused after 0.18 s |
| | 40 such powers, no budget | `LimitExceeded(Kernels)` after 108 s, 197 MB (unbounded) |
| wide and deep constants | `x + (sum of 1000 sines)*(sum of 1000 cosines)` | refused after 0.28 s, 139 MB |
| | `x + (sum of 300 sines)*(sum of 300 cosines)` | `Ok`, 0.05 s, 35 MB |
| | S_18, S_19, S_21 as one chain with `*0` | `Ok` in 0.1 ms (the zero factor ends the chain before any product) |
| | `x + S_21*sin(x)`, order 0, no budget; `x + S_20*sin(x)`; `x + S_19*sin(x)` | `LimitExceeded(ConstantTerms)` after 1.1 s, 388 MB; refused after 0.06 s; `Ok` in 0.56 s, 230 MB |
| | `x + sin(1)*cos(1) + ... + sin(5000)*cos(5000)` (98 KB) | `Ok`, 0.03 s |
| | 12 nested `(...)^2 + e`; `1/(x + S_12 - S'_12)` | `Ok`, 1 ms and 5 ms |
| chains | `x + y1 + ... + yN`, `x*y1*...*yN`, N up to 5000 (17 KB) | `Ok` in at most 10 ms release, 31 ms and 42 ms debug |
| | `x + x^-1 + ... + x^-12000` (12 000 series in one sum, 121 KB); `sum of 6000 sin(k)*x^(-k)` (118 KB) | `Ok` in 0.04 s and 0.03 s, 21 MB and 17 MB |
| | `x + x + ... + x`, 65 536 terms | `Ok` on an 8 MB release and a 2 MB debug thread (HEAD overflows at 12 800 and 190) |
| DAG inputs | `e = e + e`, `e*e` of `sin(1)`, of `x*0`, of `1 + x`, a DAG point, and `e = e + x*e`, k = 20 and 40 | refused at 1e8 after 0.07 to 0.17 s release, 0.56 to 1.45 s debug on a 2 MB thread |
| | `x + L + ... + L`, 25 000 references to one 50 000-digit literal (API) | refused after 0.09 s, 137 MB |
| huge literals | a 99 990-digit literal L: `x + L` (+ Display), `x^L` at 1 and `(1+x)^L` at orders 3 and 2000, `sin(x + L)`, `x + 2^L`, `x + L/7^99999`, `1/(x + L*sin(1) - 1)`, `1/(x + sin(L) + 2)`, `x + L^(1/2)`, `x + L^(1/3)` | all within 0.13 s: `Ok`, or refused at 1.08e8 for `x^L` and `(1+x)^L` (an honest over-charge, §6) |
| deep nesting | `x + sin^4000(1)` (+ Display), `1/(x + sin^4000(1) + 2)` | `Ok` in 4 ms and 10 ms, kernel depth 4001, result depth 4002 (HEAD: 1.2 s and 1.5 GB) |
| | `x + sin^150(1)`, debug | `Ok`, 1 ms |
| | `exp^1000(x)` at 0, order 3; the continued fraction `1/(1 + 1/(...(sin(1) + x)))`, 1500 levels | refused after 0.16 s and 0.20 s (HEAD: stack overflow for both) |
| | `exp^70(x)`, debug | refused after 0.88 s (HEAD 2.1 s, 1.4 GB) |
| | continued fraction, 60 levels, debug | `Ok`, 0.13 s, result depth 137 (its text is 1.4 MB: engine-made kernels, §2.2) |
| | `x + N_300` (+ Display); `x + N_1000`, no budget | `Ok` in 1 ms (text 10 KB) and 3 ms, kernel depth 900 and 3000 |
| | `(1+exp(...))^511` at every level, k = 60 and 100; k = 100 without a budget | refused after 0.13 s; `Ok` in 3.2 s, kernel depth 300 |
| | `invpow(80)` (input divisors); without a budget; HEAD | refused after 0.14 s; `Ok` in 2.8 s, kernel depth 241, result depth 243 = the input's (revision 5: `LimitExceeded(KernelDepth)` after 19.3 s); HEAD 9 ms, depth 248 |
| | `invpowx(20)` (computed inverses); `invpowx(4)` without a budget; HEAD | refused after 0.31 s; `LimitExceeded(ConstantTerms)` after 2.5 s, 674 MB (unbounded, §2.15); HEAD 0.6 ms (§7 item 8) |
| | `(1+exp(...(1+exp(x))^2...))^2` at 1, k = 10 (+ Display); k = 20 to 60 | `Ok` in 4 ms, text 35 MB (HEAD 2 KB: kernels from computed values, §2.2); refused after 0.45 to 0.5 s |
| | `x^(x^(...x))` 300 levels at 1; `x/(1/(1/(...)))` 3000 levels; `((1+x)^2)^2...` 3000 levels | `Ok` in 2 ms, 4 ms, 21 ms |
| | the deepest input handled, ten shapes (`depthcmp.py`) | release 8 MB: the prototype reaches HEAD's parser limit (4096 nested calls, 2048 for `((..)*2+1)` and for N_k) on every shape, and HEAD's `taylor` stops earlier on two (364, 228). Debug 2 MB: the prototype reaches at least HEAD's depth on every shape except nested calls, where both stop in the parser at 160 to 164 |
| many radicals | `x + R*R`, R the product of the square roots of 4000 primes; `x + sqrt(2) + ... + sqrt(10001)` (109 KB) | `Ok` in 0.03 s and 0.08 s |
| | 200 64th roots, 200 cube roots of 4096-bit radicands; 2000 64th roots | `Ok` in 2.66 s and 0.54 s; refused after 2.73 s |
| | `(sqrt(2) + ... + sqrt(30))^64`; `1/(x + sqrt(2)*sqrt(3) - sqrt(6))` | `Ok` (the kernel of the cap); `UndecidedZero` (documented) |
| high orders | `exp(x)^2/x^250000` and `/x^1000000`, `(exp(x)*sin(x))/sin(x)^100000` | refused after 0.12 to 0.14 s, at most 122 MB |
| | `exp(x)*sin(x)` at 2000, `(1+x)^(10^100)` at 2000, `tan(x)` at 1 at 2000, `exp(x)` at the point K at 2000, `sin(x)^2000/x^2000` at 2000 | refused after 13 ms to 0.97 s |
| | `exp(x)^2/x^1000000` without a budget | aborted by the 6 GiB address-space cap after 5.9 s (unbounded `taylor`; `limit_via_series` passes a budget, §2.13) |
| poles of high order | `exp(x)/x^1000000000`; 1/sin(x)^2000; 1/sin(x)^50000; 1/(exp(x)-1)^1000; 1/(x+x^2)^5000; 1/(x^1000000*(1+x)) | refused after 36 us to 0.79 s, at most 104 MB |
| | 1/(sin(x)^200 - x^200) at order 2 | `PrecisionExhausted(131)` in 0.2 ms (the valuation -202 lies beyond the bounded search at this order; honest) |
| | 1/sin(x)^200 at order 2, debug | `Ok`, 1.18 s |

**Revision 5** (review 5, B1 and B2, and the self-attack), release, HEAD 92906d4, the cost model of
§2.7 with the revision 5 charges, budget 1e8: `python3 EXP/rev5/tools/calib5.py EXP/rev5/bin/proto6_rel
100000000`, output `EXP/rev5/calib5_final.txt`, 70 rows: the 54 of revision 4 and 16 for review 5 and
the self-attack. Every revision 4 row keeps its outcome (`ok` or `refused`), with one deliberate
change of input meaning: `(S)*0` is now the chain `S*0` (parentheses make no node), whose zero absorbs
the product in `product_many`, so building S is measured by `S*x` at order 0 in `mem5.py`. Selected
rows (the revision 4 table below has the others; their units are 0.19 to 1.16 times revision 4's:
down where `scale` by 1 now shares its map and where `product_many` multiplies a chain in rounds, up
by the new charges):

| f | point | n | result | units | time | units/s | peak |
|---|---|---|---|---|---|---|---|
| exp(x) | 0 | 2000 | ok | 3.70e6 | 10 ms | 3.6e8 | 9.6 MB |
| exp(sin(x)), tan(x) | 0 | 250 | ok | 3.40e7, 3.61e7 | 0.34 s, 0.39 s | 1.0e8, 9.3e7 | 4 MB |
| (5/7+x)^(-3/11) | 0 | 2000 | ok | 8.62e7 | 0.90 s | 9.5e7 | 12.5 MB |
| tan(x) | 1 | 100 / 200 | ok / refused | 8.23e7 / 1e8 | 1.03 s / 1.26 s | 8.0e7 | 7 MB |
| `(K + x + ... + x)*0`, 400 `+x` | 0 | 1 | ok | 7.48e7 | 0.25 s | 2.9e8 | 96.5 MB |
| `((x+y1) + ((x+y2) + ...))*0`, 6 levels | S | 1 | refused | 1e8 | 0.77 s | 1.3e8 | 250 MB |
| `x + P_4096` (B1) | 0 | 1 | ok | 2.02e6 | 20 ms | 1.0e8 | 8.2 MB |
| `1/(x + P_4096 - 1/2)` (B1) | 0 | 1 | ok | 3.33e6 | 40 ms | 8.3e7 | 10.8 MB |
| `x + Q_1400*(S_15)*0` (B2) | 0 | 1 | refused | 1.26e9 | 39 ms | | 14.6 MB |
| `x + Q_1400*S_15*0` (B2, one chain) | 0 | 1 | ok | 2.82e6 | 8 ms | 3.3e8 | 6 MB |
| `x + P_4096*S_6` | 0 | 1 | ok | 2.15e7 | 89 ms | 2.4e8 | 60.7 MB |
| `1/(x + P_4096*S_6)` | 0 | 1 | refused | 1.05e8 | 1.36 s | 7.7e7 | 223 MB |
| `x + sin(1) + ... + sin(11000)` (110 KB) | 0 | 1 | ok | 4.25e6 | 36 ms | 1.2e8 | 20.8 MB |
| `x + (sum of sin(1..300))*(sum of cos(1..300))` | 0 | 1 | ok | 2.25e7 | 79 ms | 2.8e8 | 67.8 MB |
| `x` plus 200 64th roots of `2^4095 + k` | 0 | 1 | ok | 9.69e7 | 3.01 s | 3.2e7 | 4.7 MB |
| the same with 2000 roots | 0 | 1 | refused | 1e8 | 2.81 s | 3.6e7 | 6.6 MB |
| 1/sin(x)^2000, 1/(exp(x)-1)^1000 | 0 | 2 | refused | 1e8 | 0.86 s, 0.95 s | 1.1e8 | 26 MB, 9 MB |
| `exp(x)^2/x^250000` | 0 | 1 | refused | 1e8 | 0.18 s | 5.5e8 | 149 MB |

The slowest rate is 3.2e7 units/s (the 64th roots: trial division by p^64 for 6542 primes per
radicand; revision 4's slowest was 5.8e7), and the slowest refused call stops after 2.81 s. At that
rate 1e8 units take 3.1 s, under the X2 policy of about 10 s. `TAYLOR_WORK` stays 1e8 by the rule
below (the slowest rate is above 1e7 units/s).

Memory per unit, revision 5 (`EXP/rev5/tools/mem5.py`, output `mem5_final.txt`, VmHWM, the parse of a
100 KB input takes up to 9 MB of it; superseded by the revision 6 figures above, which also cover the
families revision 5 missed): at most 2.9 bytes per unit, `x + (sum of sin(1..550))*(sum of cos(1..550))`
(219 MB at 7.53e7 units, most of it the rendering of 302 500 terms); `x + P_4096*S_8` 2.7 (221 MB at
8.21e7); building S 2.6 (103 MB at 4.0e7); K 1.3 (96 MB at 7.47e7); `(1/(1-x))/x^200000` 1.8 (150 MB
at 8.2e7, one-term coefficients); kernel-heavy inputs 1.4 to 2.8. So `TAYLOR_WORK = 1e8` holds at most
about 290 MB plus the parsed input. The highest peak of every probe below is 250 MB.

**Revision 5 probes** (the self-attack of the orchestrator's brief, item 4). The families of reviews
3 to 5 and new ones, at the review bar: an input of at most 100 KB through `taylor_with_budget` with
budget 1e8 must not run longer than 20 s in release, use more than 512 MB, or overflow an 8 MB release
thread or a 2 MB debug thread. About 300 runs (`EXP/rev5/tools/runall5.sh`; outputs `attack5_final.txt`,
`attack5_deep2_final.txt`, `attack5_nkx_final.txt`, `leftdeep_final.txt`, `dag_final.txt`,
`mem5_final.txt`, `calib5_final.txt`, `depthcmp_rel_final.txt`, `depthcmp_dbg_final.txt`,
`newtests_dbg.txt`, `enginedepth_dbg_final.txt`, `enginedepth_rel.txt`, `rev4cmp.txt`, `b2_rel.txt`).
Release on an 8 MB thread unless it says debug (2 MB thread); HEAD is HEAD's `taylor` on the same input
(`EXP/rev5/proto6/head.rs`). Every outcome is honest: `Ok` with the gp-checked algorithm, or a typed
refusal.

| family | probe | outcome |
|---|---|---|
| long monomials | `x + P_4096`, `1/(x + P_4096 - 1/2)`, debug | `Ok` in 0.10 s and 0.15 s, result depth 16 and 21, `format!` 56 KB and 113 KB (revision 4: abort) |
| | `x + P_8800`, `1/(x + P_8800 - 1/2)` (104 KB) | `Ok` in 17 ms and 63 ms, depth 17 and 22 (HEAD: 77 ms, depth 17) |
| | balanced product of 13 000 symbols (75 KB) | `Ok`, 18 ms, depth 17 |
| | `x + (P_4096 + 1)^64` | refused at 1.02e8 after 18 ms |
| | `x + P_4096*P'_4096` (8192 factors) | `Ok`, 17 ms |
| | `1/(x + P_4096*S_6)` | refused after 1.31 s, 222 MB |
| | `x + Q_1400*(S_15)*0`, M = 2900 with k = 14 | refused at 1.26e9 after 39 ms (revision 4: `Ok` after 13.3 s, 1.09 GB) |
| many kernels | `x + sin(1) + ... + sin(11000)` (110 KB, left-deep) | `Ok`, 31 ms, 20.9 MB (HEAD 18 ms; revision 4: refused, quadratic merges) |
| | balanced sums of 11 000 sines and of 20 000 symbols; `1/(x + the symbols)` | `Ok` in 29 ms, 43 ms and 67 ms, depth 17 |
| | 6 binomial powers `(x + sin(k)^2 + cos(k)^2 - 1)^1000000` at order 2000 | refused after 0.18 s |
| | 40 such powers, no budget | `LimitExceeded(Kernels)` after 108 s, 200 MB (unbounded) |
| wide and deep constants | `x + (sum of 1000 sines)*(sum of 1000 cosines)` | refused at 2.49e8 after 0.51 s |
| | S_18, S_19, S_21 as one chain with `*0` | `Ok` in 50 ms (the zero absorbs) |
| | `x + S_21*sin(x)`, order 0, no budget | `LimitExceeded(ConstantTerms)` after 1.1 s release, 384 MB (7.3 s debug) |
| | `x + S_20*sin(x)` | refused after 0.44 s, 169 MB |
| | `x + sin(1)*cos(1) + ... + sin(5000)*cos(5000)` (98 KB) | `Ok`, 18 ms |
| | 12 nested `(...)^2 + e`; `1/(x + S_12 - S'_12)` | `Ok`, 1.5 ms and 7 ms |
| chains | `x + y1 + ... + y5000`, `x*y1*...*y5000` (17 KB) | `Ok` in 7 ms and 6 ms release, 27 ms and 39 ms debug (revision 4: refused at 3000 symbols) |
| | `x + x + ... + x`, 65 536 terms | `Ok` on an 8 MB release and a 2 MB debug thread (HEAD overflows at 12 800 and 190) |
| DAG inputs | `e = e + e`, `e*e` of `sin(1)`, of `x*0`, of `1 + x`, and a DAG point, k = 20 and 40 | refused at 1e8 after 0.11 to 0.45 s release, 1.0 to 3.3 s debug (revision 4: did not return at k = 40) |
| huge literals | a 99 990-digit literal L: `x + L` (+ Display), `x^L` at 1 and `(1+x)^L` at orders 3 and 2000, `sin(x + L)`, `x + 2^L`, `x + L/7^99999`, `1/(x + L*sin(1) - 1)`, `1/(x + sin(L) + 2)`, `x + L^(1/2)`, `x + L^(1/3)` | all within 0.14 s: `Ok`, or refused at 1.08e8 for `x^L` and `(1+x)^L` (Miller's steps multiply 5200-word coefficients, charged by the schoolbook bound; HEAD returns them in 3 ms: an honest over-charge, the risk §6 names) |
| deep nesting | `x + sin^4000(1)` (+ Display), `1/(x + sin^4000(1) + 2)` | `Ok` in 4 ms and 10 ms, kernel depth 4001, result depth 4002 (HEAD: 1.4 s and 1.5 GB) |
| | `x + sin^150(1)`, debug | `Ok`, 1 ms |
| | `exp^1000(x)` at 0, order 3; continued fraction `1/(1 + 1/(...(sin(1) + x)))`, 1500 levels | refused after 0.17 s and 0.19 s (HEAD: stack overflow for both) |
| | `exp^70(x)`, debug | refused after 0.88 s (HEAD 2.2 s, 1.4 GB) |
| | continued fraction, 60 levels, debug | `Ok`, 81 ms, result depth 137 (its text is 1.4 MB: engine-made kernels, §2.2) |
| | `x + N_300` (^2 nesting, 3.3 KB), + Display | `Ok` in 1.4 ms, text 10 KB (revision 4: text of 2^300 nodes; HEAD 5 KB) |
| | `x + N_1000`, no budget | `Ok`, 5 ms, kernel depth 3000 |
| | `(1+exp(...))^511` at every level, k = 60 and 100 | refused after 0.26 s and 0.22 s; k = 100 without a budget `Ok` in 3.4 s (kernel depth 300) |
| | `(1+exp(...(1+exp(x))^2...))^2` at 1, k = 10 (+ Display) | `Ok`, 4 ms, but the text is 35 MB (HEAD 2 KB): kernels from computed values (§2.2); k = 20 and more: refused after 0.5 s (2^k terms) |
| | `x^(x^(...x))` 300 levels at 1; `x/(1/(1/(...)))` 3000 levels; `((1+x)^2)^2...` 3000 levels | `Ok` in 2 ms, 9 ms, 22 ms |
| | the deepest input handled, eight shapes (`depthcmp.py`) | release 8 MB: the prototype reaches HEAD's parser limit (4096 nested calls, 2048 for `((..)*2+1)`) on every shape, and HEAD's `taylor` stops earlier on two (364, 228). Debug 2 MB: the prototype reaches at least HEAD's depth on every shape; both stop in the parser for nested calls (160 to 162; gdb shows the overflow in nom) |
| | the engine alone, API-built inputs, debug 2 MB (`enginedepth.rs`) | `sin(sin(...x))` 1280 levels, `x + sin^k(1)` 368, a `Neg` chain 1296, a right-deep sum 648, a `/` chain 344, `((..)*2+1)` 320. HEAD: 188, 184, 188, 188, 47, 62. Revision 4: 172 for the first two (a regression, fixed) |
| many radicals | `x + R*R`, R the product of the square roots of 4000 primes | `Ok`, 26 ms |
| | `x + sqrt(2) + ... + sqrt(10001)` (109 KB) | `Ok`, 32 ms |
| | 200 64th roots, 200 cube roots of 4096-bit radicands; 2000 64th roots | `Ok` in 2.56 s and 0.51 s; refused after 2.67 s |
| | `(sqrt(2) + ... + sqrt(30))^64`; `1/(x + sqrt(2)*sqrt(3) - sqrt(6))` | `Ok` (the kernel of the cap); `UndecidedZero` (documented) |
| high orders | `exp(x)^2/x^250000` and `/x^1000000`, `(exp(x)*sin(x))/sin(x)^100000`, `((1/(1-x))^2)/x^250000` | refused after 0.16 to 0.21 s (the last: revision 4 ran 34.7 s and 524 MB, the uncharged pair count) |
| | `exp(x)*sin(x)` at 2000, `(1+x)^(10^100)` at 2000, `tan(x)` at 1 at 2000, `exp(x)` at the point K at 2000, `sin(x)^2000/x^2000` at 2000 | refused after 14 ms to 1.65 s |
| | `exp(x)^2/x^1000000` without a budget | aborted by the 6 GiB address-space cap after 6.3 s (unbounded `taylor`; `limit_via_series` now passes a budget, §2.13) |
| poles of high order | `exp(x)/x^1000000000`; 1/sin(x)^2000; 1/sin(x)^50000; 1/(exp(x)-1)^1000; 1/(x+x^2)^5000; 1/(x^1000000*(1+x)) | refused after 40 us to 0.91 s, at most 170 MB |
| | 1/(sin(x)^200 - x^200) at order 2 and at order 50 | `PrecisionExhausted(131)` and `(179)` in 0.4 ms (the valuation -202 lies beyond the bounded precision search of §2.6 at these orders; a bounded search, honest) |
| | 1/sin(x)^200 at order 2, debug | `Ok`, 1.07 s |
| small coefficients (memory) | `(1/(1-x))/x^k` for k = 2e5 and 3e5, `-(1/(1-x))/x^250000`, `(1/(1+x^2))/x^300000`, `(1/(1-x-x^2))/x^100000` | `Ok` or refused, at most 150 MB (revision 4: 629 MB and 524 MB, above the bar; one B-tree node per one-term coefficient) |

Rerunning with the final code (the implementer, §6): `EXP/rev7/tools/runall7.sh` (a superset of
`runall6.sh` and `runall5.sh`, with the reproduction scripts of reviews 3 to 7) with its binary paths
replaced by drivers built from the crate, one of them with the counting allocator; the drivers call
the public entry points, as `EXP/rev7/proto8/main.rs` does (`ENTRY=tw`, `se`, `lim`). If the slowest rate falls below 1e7 units/s, `TAYLOR_WORK` becomes 10 s times that
rate, rounded down; otherwise it stays 1e8. If a heap peak exceeds 3 bytes per unit, the constant of
the object that dominates it (the table "Memory" of §2.16) grows until it does not.

**Revision 4** (review 4, B1 and B2), release, HEAD dc367a2, cost model of §2.7, budget 1e8:
`python3 EXP/rev4/tools/calib4.py EXP/rev4/bin/proto5_rel 100000000`, outputs
`EXP/rev4/calib4_run1.txt` (the table below) and `EXP/rev4/calib4_run2.txt` (a second run after the
last prototype change, the 4-unit charge of the `pow_int` slices: the same units within 0.1 % and
the same outcomes), 54 rows each: the 41 of revision 3 and 13 for review 4. Wall time includes
about 2 ms of process start, so rows under 10 ms say little about the rate. K, S and N_k are as in
the Revision 4 table. Selected rows:

| f | point | n | result | units | time | units/s |
|---|---|---|---|---|---|---|
| exp(x) | 0 | 2000 | ok | 5.71e6 | 16 ms | 3.5e8 |
| sin(x) | 0 | 2000 | ok | 4.48e6 | 17 ms | 2.7e8 |
| 1/(1+x+x^2) | 0 | 2000 | ok | 7.06e5 | 6 ms | 1.3e8 |
| exp(sin(x)) | 0 | 250 | ok | 3.26e7 | 0.34 s | 9.5e7 |
| tan(x) | 0 | 250 | ok | 3.54e7 | 0.42 s | 8.5e7 |
| sqrt(1+x+x^2) | 0 | 2000 | ok | 3.39e7 | 0.42 s | 8.1e7 |
| exp(x)*sin(x) | 0 | 300 | ok | 4.26e7 | 0.14 s | 3.0e8 |
| (1+x)^10000 | 0 | 2000 | ok | 2.89e6 | 14 ms | 2.1e8 |
| exp(sin(x)), tan(x), log(cos(x)) | 0 | 2000 | refused | 1e8 | 0.87 s, 0.80 s, 0.87 s | 1.15e8 to 1.26e8 |
| (2/3+x)^(1/7) | 0 | 2000 | ok | 4.53e7 | 0.57 s | 8.0e7 |
| sqrt(1+x/3) | 0 | 2000 | ok | 1.73e7 | 0.23 s | 7.5e7 |
| (1+x/3)^(1/7) | 0 | 2000 | ok | 3.53e7 | 0.52 s | 6.7e7 |
| (5/7+x)^(-3/11) | 0 | 2000 | ok | 8.52e7 | 0.93 s | 9.2e7 |
| x + (3/5)^200000 | 0 | 1 | ok | 3.72e7 | 0.18 s | 2.1e8 |
| x + (3^330000)/(5^200000)*2*3 | 0 | 1 | refused | 1.21e8 | 0.31 s | 3.9e8 |
| the 67-character zero-multiplied sum of review 3 | 0 | 2000 | refused | 1e8 | 0.99 s | 1.0e8 |
| 1/(2+sin(x)) | 1 | 50 / 100 | ok / refused | 2.92e7 / 1e8 | 0.15 s / 0.71 s | 2.0e8 / 1.4e8 |
| exp(sin(x)) | 1 | 50 / 100 | ok / refused | 3.48e7 / 1e8 | 0.21 s / 0.81 s | 1.7e8 / 1.2e8 |
| log(cos(x)) | 1 | 100 | ok | 4.81e7 | 0.59 s | 8.2e7 |
| tan(x) | 1 | 100 / 200 | ok / refused | 7.86e7 / 1e8 | 1.00 s / 1.41 s | 7.9e7 / 7.1e7 |
| exp(x) | 1 | 2000 | ok | 5.77e6 | 15 ms | 3.7e8 |
| 1/sin(x)^200 | 0 | 2 | ok | 1.62e7 | 0.28 s | 5.8e7 |
| 1/(exp(x)-1)^150 | 0 | 2 | ok | 2.85e7 | 0.36 s | 8.0e7 |
| `(K + x + ... + x)*0`, 400 `+x` (B1) | 0 | 1 | ok | 6.71e7 | 0.26 s | 2.6e8 |
| `(K - x - ... - x)*0`, 160 `-x` (B1) | 0 | 1 | ok | 6.71e7 | 0.27 s | 2.5e8 |
| `x^0 + ... + x^0`, 100 terms (B1) | K | 1 | ok | 6.70e7 | 0.60 s (0.24 s in the engine; the driver builds the point first) | 1.1e8 |
| `K + x + y + ... + y`, 160 `+y` (B1 control) | 0 | 1 | refused | 1.02e8 | 0.37 s | 2.8e8 |
| `S + y + ... + y`, 8 `+y` (merges of small terms) | 0 | 1 | refused | 1.17e8 | 0.80 s | 1.5e8 |
| `((x+y1) + ((x+y2) + ...))*0`, 6 levels | S | 1 | refused | 1.17e8 | 0.87 s | 1.3e8 |
| `sin(x) + cos(x)` (kernels of S) | S | 0 | refused | 1e8 | 0.55 s | 1.8e8 |
| `S*0` | 0 | 1 | ok | 4.12e7 | 0.56 s | 7.3e7 |
| `exp((1+exp(N_16))^511 + x)` (B2) | 0 | 1 | ok | 1.81e7 | 36 ms | 5.1e8 |
| `exp((1+exp(N_30))^511 + x)` (B2) | 0 | 1 | ok | 1.81e7 | 34 ms | 5.2e8 |
| the N_10 product (B2) | 0 | 1 | refused | 1e8 | 0.24 s | 4.2e8 |
| `x + sin(2^340000+1) + ... + sin(2^340000+40)` | 0 | 1 | ok | 2.28e7 | 19 ms | 1.2e9 |
| x plus 200 cube roots of 63-bit radicands (m2) | 0 | 1 | ok | 1.27e7 | 0.10 s | 1.2e8 |

Every revision 3 row keeps its outcome (`ok` or `refused`). The slowest rate measured is 5.8e7
units/s (`1/sin(x)^200`; 5.75e7 in an earlier run; the slowest of the second run is 6.8e7, `S*0`),
and the slowest refused call stops after 1.41 s (`tan(x)` at 1, order 200; 1.65 s in an earlier
run). The copy and key paths of review 4 now run at
1.1e8 to 1.2e9 units/s. `TAYLOR_WORK = 100_000_000`: at the slowest measured rate a call stops after
at most about 1.7 s, and it stays under the X2 policy of about 10 s for any input whose rate is at
least 1e7 units/s. §5.3 used to prescribe 10 s times the slowest rate, which would give 5.8e8; I
keep 1e8 and a factor of 5.8 in reserve, because each review so far found an input family 30 to
10 000 times slower per unit than the calibration of its time, or one that was not charged at all
(review 1, B3; review 2, B2; review 3, B1 and B2; review 4, B1 at 1 100 units/s and B2). The
implementer reruns the revision 5 calibration (`EXP/rev5/tools/calib5.py`, a superset of `calib4.py`)
with the final code (release) and records the table in the doc comment of `TAYLOR_WORK`. If the slowest rate falls below 1e7 units/s, `TAYLOR_WORK` becomes 10 s
times that rate, rounded down; otherwise it stays 1e8. The model over-charges integer-only and
product-heavy inputs; that errs towards refusing, with an honest `BudgetExceeded`.

Revision 3 calibration (`EXP/rev3/calib3_final.txt`, the same 41 rows on the revision 3 model): the
slowest rate was 3.7e7 units/s (`tan(x)` at 1, order 200, refused after 2.67 s). Revision 4 charges
the inputs at transcendental points up to 2 times more (the merges of the recurrence sums), and runs
them faster (a copy is O(1)).

Revision 1 and 2 calibration, at c79a582 (Euclid gcd; `EXP/calib_rev1.txt`, `EXP/rev2/calib_r2.txt`):
exp(x) and sin(x) at 2000 ran at 8.6e7 to 9.9e7 units/s, exp(sin(x)) and tan(x) at 250 at 1.7e7 to
2.0e7, sqrt(1+x+x^2) at 2000 at 1.28e7 to 1.45e7. Those rates were measured on inputs with small
coefficients only; review 3 showed that the revision 2 model charged large numerators and
denominators 30 to 10 000 times too little per unit (B1), and big roots not at all (B2).

### 5.4 Iterative Drop

See the table of §2.14: no measurable cost on ordinary workloads (+7 % instructions, wall time
within noise), and deep trees of any shape are dropped in bounded stack. A deep drop in the
iterative branch costs about the same as the glue (3*10^4-term sum: 3.8 ms vs 4.4 ms release).
Revision 8 (review 8 m3): a value dropped by the thread's TLS teardown after `LEAF` is gone
(`EXP/rev8/probes/dropm.rs` mode `tls`, `EXP/rev8/review8_r8.txt`) drops a unique chain, `twin`, `sib`
and a left-deep sum of 2 x 10^5 levels on a 2 MB debug thread (at most 16.5 KB of stack seen in
`Drop`), and of 10^6 levels on a 1 MiB release thread in at most 0.40 s (one fresh leaf per moved
child); revision 7rt3's `LEAF.with` panics in that teardown, and the process was still printing
panics when it was killed after 20 s.

## 6. Plan

Implementation order (each step builds and tests green; `cargo test -p rustmath-symbolic` after
each step, plus the crates named in the step):

1. `rustmath-symbolic/Cargo.toml`: add `rustmath-reals`.
2. `src/series_ball.rs` (new): `Ball`, `enclose_real(&Expr, prec) -> Option<Ball>` and
   `ball_cost(&Expr, prec, limit) -> u64` on `Expr` only, with the window R1, the exact-zero rule
   R2, the exp/sinh/cosh guard and the sin/cos guard of §2.3; powers of two by shifts.
   Unit tests: `2 + sin(1)`, `exp(pi) - pi`, `sin(1) + 1/3` enclosed away from 0;
   `sin(1)^2 + cos(1)^2 - 1` encloses 0; `log(-1)`, `(-4)^(1/2)`, `I`, `y`, `gamma(1/3)` have no
   enclosure; the ball unit tests of §4.1 (review 2, `erf` of review 3, m4, and the memo of review
   4, B2). Revision 7 (review 7, B1): the erf rule of §2.3, the threshold `X_p` (`erf_threshold`),
   the cheap enclosure above it, `erf_units` with `ERF_FACTOR = 2`, `ERF_BASE = 8` and
   `literal_abs_bound`, and `unary_units` shared by `ball_cost`. Revision 7rt1 (red team 1, m1 and m2):
   sin charged `((3p + 64)/64 + 1)^2 * 8` (cos keeps `2p + 64`), and `literal_abs_bound` returning
   bounds `(lo, hi)` from the bit lengths and the top 64 bits of the literal's parts, with `lo` deciding
   the cheap regime and `hi` the charge. Revision 7rt2 (red team 2 numeric m1): `enclose_real` returns
   `Result<Ball, NoBall>` with the reasons `Never` and `TooWide` of §2.3 (the prototype's thread-local
   flag becomes the error value), and the unit test `ball_too_wide_goes_on` of §4.1; port from
   `EXP/rev7rt2/proto10/ball.rs`. `enclose_real` and `ball_cost` evaluate and count each node reached through an `Arc` once
   (memoized by address), both with an explicit stack (revision 5: a unit test encloses a sum chain
   of depth 100 000 on a 2 MB thread). Port from `EXP/rev7rt2/proto10/ball.rs` (`ball_expr`, the
   iterative post-order; `ball_cost` with `BALL_NODE` per distinct node, revision 6; its `enclose`,
   `proves_nonzero_ab`, `non_integer`, `floor_small` move to step 3, which owns the class checks and
   the charging; `enclose` charges `expr_units` and `ball_cost` first, and the two tests go on to the
   next precision on `TooWide`). Revision 7rt3 (red team 3 numeric, m1 and m2): asinh, acosh and atanh
   by rug's `Float::asinh`, `acosh` and `atanh` with the Lipschitz bounds of §2.3, every integer
   exponent in i64 by binary powering and one beyond i64 by `exp(k log |a|)` (`big_int_pow`), the
   charge `u (2 min(bits(k), 64) + 1)`, and `real_sign` (rule 5's enclosures, for `0^r`). Revision 8
   (review 8, m1): `small_int_value` (an exponent's exact integer value in i64, at most 64 nodes and 8
   levels read), an exponent with such a value an integer power (`bin_pow` for |k| <= 3, `big_int_pow`
   above, a^0 = 1), the charge `4u` of a non-literal exponent, and the unit test
   `integer_valued_exponent_enclosure` of §4.1. Revision 7rt4 (red team 4 numeric, m3): an exact ball
   for a literal exact at p bits (an integer of at most p bits, a dyadic rational with such a
   numerator), `sqrt(0) = 0`, `0^k = 0` for k > 0 on the `exp(k log |a|)` path, `0^b` for a real b by
   the sign of b's ball, `exp(k log a)` with an integer-valued k as the power a^k (`exp_int_pow`,
   `int_pow_ball`), and the unit tests `exact_zero_operand` and `exp_integer_log` of §4.1. Port from
   `EXP/rev7rt4/proto13/ball.rs`.
3. `src/series_const.rs` (new): `Work`; first the charged arithmetic layer of §2.7 (`OP_BASE`,
   `pow_bits`, `q_add`, `q_sub`, `q_mul`, `q_pow`, `q_new`, `i_mul`, `i_pow`, `i_gcd`, `i_lcm`,
   `i_exact_root`, each `fn(w: &mut Work, ..) -> Result<_, SeriesError>`), `copy_units` and
   `expr_units`, then `SeriesConst` (its map behind `Arc`), `Kernel`, classes, canonical reductions
   with checked `i64` exponents (the radical fold through `q_pow`), `from_expr`, `apply`, `pow_const`
   with the size caps of `exact_pow_if_small` on every path, d-th-power-free radicals (one
   `i_exact_root` per radical, no `rat_root`, the prime table built once), `inverse`, `zero_status`
   with the A + B I numeric rule, `non_integer_status`, `to_expr` (charged) and `to_expr_raw` (for
   kernel creation, which charges its arguments), `balanced_sum`; every operation takes `&mut Work`
   and returns `Result`. Kernel keys from the arguments' normal forms with sub-kernel ids, charged one
   unit per byte before they are written, interned in `Work` (the only copy; ids, O(1)
   comparisons), the expression charged before it is built, `has_indet` from the normal forms;
   `log_exact`. Unit tests of §4.1 (zero test, non-integer status, kernel keys, copies, charged
   arithmetic). Revision 5: `ConstData` with the inline one-term `Terms` and the cached
   `ConstSizes` (one constructor computes them); `KernelData::depth` and `unary`; kernels of input
   nodes take the node as their expression; the caps of §2.15 in `Work` (`Caps`, defaults, small
   ones in tests) and `SeriesError::LimitExceeded`; `MONO_FACTOR_UNITS`, `search_units`,
   `KERNEL_UNITS`, `EXPR_NODE = 20`; `sum_many`, `product_many`; the one-pass `reduce_mono`; `scale` by
   1 shares; the newest-first drop of the kernel table; `balanced_product` in `mono_expr`. Revision 6:
   `Terms::Many` as a sorted boxed slice and the one constructor `from_sorted` with `CONST_BASE`; 0
   and 1 shared; `merge2` for `add` and `sub`, the heap merge of `sum_many`, the map accumulator of
   `mul` with `ACC_ENTRY` and its linear conversion; `product_many` testing for a zero factor first;
   `inverse_src` and its callers; `KernelData::expr` as `Arc<Expr>` and the `Arc`-sharing rendering
   (`to_expr_arc_raw`, `balanced_arc`, `kpow`, `mono_expr`); the cached flags `any_imag` and
   `one_named_imag` with the fast paths of `zero_status`, `indet_groups` and `split_imag`; the
   in-place `reduce_mono`; the constants `COPY_TERM = 52`, `COPY_FACTOR = 6`, `COPY_WORD = 3`,
   `MONO_FACTOR_UNITS = 6`, `EXPR_NODE = 32`, `EXPR_WORD = 3`, `EXPR_INT = 11`, `SER_TERM = 6`,
   `ACC_ENTRY = 32`, `CONST_BASE = 72`; `KONST_NODE + COPY_WORD w` per literal; `as_rational`
   returning `Option<&Rational>` (the prototype clones). Revision 7 (review 7, m4): the floor
   `COPY_WORD` per result word in `q_mul`, `q_add`, `q_sub` and `i_mul` (`result_units`), and
   `node_payload_units` for the clone of an input node in kernel creation. Revision 7rt1 (red team 1,
   H1 and H3) and revision 7rt2 (red team 2, H1): `from_expr` as one loop over an explicit work stack
   (`konst_of`, `konst_visit`, `KFrame::resume`), with the records of the `+ -` chain, of the `* /`
   chain (a divisor inverted as it is read by `inverse_src(false, its node)`), of a tower (the right
   spine through `Neg` nodes: the bases left to right, then the top, then `pow_const` from the inside
   out with each level's node as the source), of a call and of `%`, `RECORD_UNITS` per boxed record;
   `pow_const` of 1 and -1 to any integer (revision 7rt2, structure m3). Revision 7rt3 (red team 3
   structure H2): `ConstSizes::tree` and `KernelData::tree` (O(1) from the parts; an input node's from
   the pre-pass), the effective tree and depth through the hint table of `Work` (`RenderHint`,
   `HINT_UNITS`, cleared per attempt), `to_expr_arc_raw` returning a hint, the hint of P0 in `inverse`,
   `MAX_RESULT_TREE` and `StructureLimit::TreeSize`, and the tree check of an `UndecidedZero`
   constant. Revision 8 (review 8, m2): `zero_pow_sign`, the rule of `0^r` for a constant `0^alpha`
   (shared with `zero_pow` of step 4), and the unit test `zero_power_constant`. Revision 7rt4 (red
   team 4): `Work::charge` with `saturating_add` and every units formula saturating (series H2); the
   poles arctanh(+-1) and zeta(1) of a rational argument in `apply` (numeric m1, `constant_poles`);
   `KernelData::tree` counted per kind of kernel, and the unit test `kernel_tree_count` (structure
   m1). Port from `EXP/rev7rt4/proto13/konst.rs`
   (`mk_kernel_src`, `mk_kernel_inner`, `write_nf`, `nf_key_size`, `ksize`, `from_sorted`, `Terms`,
   `merge2`, `sum_many`, `product_many`, `exact_pow_if_small`, `small_primes`, `drop_table`),
   replacing its thread-local counter, fault flag, kernel table, allowed depth (`KDEPTH_ALLOWED`), the
   side table of unary arguments (`UNARY`; the design stores it in `KernelData`) and instrumentation
   counters (`KEYCOST`, `KEXPRCOST`, `table_stats`; prototype emulations) by the explicit `&mut Work`
   and `?`.
4. `src/series_engine.rs` (new): sparse `Laurent`, `trunc` (keeping the first term) and
   `trunc_hard`, operations (the `mul` cutoff of §2.1), integer power paths (the leading term of step
   5, binary, Miller, binomial), `expand`, the `expansion` driver, rendering, and the
   `pub(crate) same_expansion` helper of the round-trip test (§4.3), with the per-term copy charges
   of §2.4, the dependence pre-pass of §2.5 and the rendering charge of §2.7. Revision 5: the thin
   `expand` dispatcher with one non-inlined function per node kind (§2.5); the chains (`expand_sum`,
   `expand_product`, `sum_series`); the iterative, memoized pre-pass with node depths and the point
   test; `KONST_NODE` per visit; the pair count of `mul` that stops at the remaining budget;
   `point_is_zero` recorded by the driver. Revision 6: the heap merge of a sparse product and of
   `sum_series`; `SER_TERM` in every term copy, also `scale`, `deriv`, `integrate` and the offsets of
   `inv`; `render_units` in the driver and the shared h of the rendering; `inverse_src` in the
   constant `Div` case; exact capacities of the chain vectors. Revision 7rt1 (red team 1, H1, H3 and
   m3): `/` in the product chain (`chain_step`), the right spine of a tower read by a loop, and the
   binomial loop of `pow_int` in the caller's context and `Work`. Revision 7rt2 (red team 2): `expand`
   as one loop over an explicit work stack (`expand`, `expand_visit`, `EFrame::resume`, the records
   `ESum`, `EProd`, `ETower` and the unary record, `RECORD_UNITS` per boxed record), so the dispatcher
   and its frame-size rules of revisions 5 to 7rt1 go; the binomial branch of `pow_int` returning
   precision `slots` and stopping at m = n (S-B3); the checked span and pair sums of `mul` and the
   other sites of the exponent audit (§2.1, S-B2); the early break and the binary search of `log`
   (S-B1); `O(h^P)` for an exact monomial to a power above i64 (structure m3). Revision 7rt3 (red
   team 3): `NeedPrecision` for an empty truncated operand of negative precision in
   `require_nonneg_val` and `integrate` (series H1); `pow_general` and `pow_const_base_k` choosing the
   constant-base rule for a base that expands to an exact constant, and `zero_pow` (structure H1); the
   regular flag carried with every value of the work stack, the records naming their node, the pre-pass
   recording tree sizes and the variable's occurrences, `maybe_hint`, `hint_exp_arg`, `subst_at` with its
   memo (`SUBST_NODE`), and `render_tree` checked against `MAX_RESULT_TREE` in the driver (structure
   H2); the doubling of a stalled loss-driven step (series m1); the unsigned offsets of `inv` (series
   m2); the first-term entry `expansion_first_term` for the limit path (memory m3). Revision 8
   (review 8, B1): the tower's constant-exponent state carries the exponent's regular flag
   (`TSt::BaseConst(alpha, reg_e)`, true for a top free of the variable) and the level ANDs it in, so
   every flag is the AND over its subtree (§2.5); a node whose series is an exact constant is seeded
   with its rendering (`seed_const`, called by `finish_flag` at the end of every record and by
   `with_exponent` for a tower exponent); `maybe_hint` and `hint_exp_arg` set a hint only when it is
   smaller than the normal form; `zero_pow` calls `zero_pow_sign`. Revision 7rt4 (red team 4): the
   precision `Prec` out of band (`covers`, `add` with the upward clamp, `at_most`), `mul`'s `Prec`
   cutoff, `inv`, `pow_int` items 2 and 5, `trunc` and `integrate` checked at i64::MAX, the driver's
   stop after a clamped loss (`PREC_CLAMPED`; series H1); `alloc` with saturating units and
   `MAX_SERIES_SLOTS` (series H2); the hint margin of `maybe_hint` and `hint_exp_arg` (structure m1);
   the entry `expansion_reaches`, which renders no term (memory m2). Unit tests
   `empty_argument_negative_precision`, `zero_power_rule`, `result_tree_cap`, `stalled_loss_step`,
   `inv_offsets_unsigned`, `value_at_point_irregular_exponent`, `exact_sentinel`,
   `dense_buffer_units`, `tower_tree_cap` of §4.1.
   Port from `EXP/rev7rt4/proto13/ser.rs`
   (drop its `S1TRACE`
   print; the binomial coefficients use `q_new` and `q_mul`; its `DEP` and `POINT_ZERO` thread-locals
   become fields of the expansion state). Unit tests that build expressions with the `Expr` API (the
   parser cannot yet read `1/sin(x)`), among them `deep_api_inputs` (§4.1), which now asserts that
   `expand` handles every shape of §2.5 at 50 000 levels on a 2 MB debug thread (the engine has no
   recursion over the input, so no frame size is guarded any more).
5. One step, because `taylor` changes its signature: `src/series.rs` (`SeriesError` with the
   bounded `Display` of §2.9 and its `display_estimate`, the bounded hand-written `Debug`,
   `ExpansionPoint`, sparse `SeriesExpansion`
   with `first_exponent` and the private `point_is_zero`, its coefficients as shared `Arc<Expr>`, the
   private h, `coefficient` returning `Option<Arc<Expr>>` and a hand-written `Debug` (revision 7,
   review 7 B2; port the layer from `EXP/rev7rt4/proto13/api.rs`: `series_expansion` builds the `Arc`s
   under `render_units`, one per distinct coefficient (revision 7rt3, memory m1), and drops the `Work`,
   `to_expr` builds from them; `limit_via_series` reads the first term of `expansion_first_term`,
   revision 7rt3, memory m3; `limit_with_error` takes its `BigO` from `expansion_reaches`, revision
   7rt4, memory m2; the public precision is T, revision 7rt4), `StructureLimit`, the DAG-safe
   `mentions_symbol` (pub, for the REPL arm), the new public functions; delete the derivative loops
   `:24-58`, `:148-169` and the facade `laurent` body `:89-119`; `known_series` with
   `balanced_sum`; rewrite the tests of §4.2; adapt I3's 10^30 test `taylor_at_a_huge_point` (`series.rs:795`) to the `Result` API, keeping its three assertions, next to `big_point_s7`), `lib.rs`
   exports and test unwraps, `src/limits.rs` (§2.13 with `LIMIT_SERIES_WORK`, §4.4 tests), and the
   REPL taylor arm (`rustmath-jupyter/src/repl.rs:2118-2145` at HEAD 92906d4: `taylor_with_budget`,
   the error mapping of §3.1 with `LimitExceeded`, the variable by `Symbol::new` and
   `mentions_symbol`, `match &args[3].simplify()`, a provisional `TAYLOR_WORK = 100_000_000`).
   Then `cargo test -p rustmath-jupyter`.
6. `src/simplify.rs`: §2.10 and its test (with the input still forgotten until step 7 lands, or run
   the test from step 7 on). `src/expand.rs`: §2.17 and the tests of §4.6 (revision 6).
7. One step, because `Drop` breaks compilation of the E0509 sites: `src/expression.rs` (Display
   §2.11, written with a work list as in `EXP/rev7rt4/proto13/display.rs`, `display_iter` (with the
   parentheses of a negated fraction, revision 7rt2, and the `.0` of an integral float, revision 7rt3),
   with `test_display_work_list` and `test_display_integral_float` of §4.3; `Drop for Expr` §2.14 (the
   list of `Arc`s and `Arc::into_inner` of revision 7rt3, and since revision 8 `LEAF.try_with` with a
   fresh leaf during the thread's TLS teardown, as `EXP/rev8/probes/dropm.rs`) with its unit tests
   `drop_left_deep_sum_1e6`, `drop_other_shapes`, `drop_shared_children`, `drop_during_tls_teardown`);
   the 4 library sites and 98 test sites of §3.3 (`rustmath-symbolic`, `rustmath-calculus`,
   `rustmath-geometry`; the jupyter site is already gone in step 5); remove `mem::forget` from
   `simplify_long_left_deep_sum`. Then `cargo check --workspace --tests`.
8. `src/parser.rs`: §2.12 (with the sign runs as loops) and §4.3 tests. Then measure the new parser
   (review 7, m3; revisions 7rt1 and 7rt2): the deepest input of each shape of the table of §2.5 that
   `parse` reads, the six nested shapes, the six mixed shapes of red team 2 (`mix5cs`, `mix5`, `mix4`,
   `mix5cdiv`, `mix6c`, `mix6` of `EXP/redteam2-structure/gen.py`, every grammar level in each
   parenthesis), the four towers (`x + 1^...^1`, `x^2^...^2` and `x^x^...^x` at 1, `x^-x^-...^-x`) and
   the four chains (`x/2/.../2`, `x + (1/2/.../2)`, `1/x/.../x`, `x*3/2*3/2...`), on a 2 MB debug
   thread, an 8 MB release thread and the REPL's 64 MiB debug thread (`EXP/rev7rt2/tools/depthcmp.py`,
   `tools/bsearch.py ... both` and `tools/mixdepth_rt2.py`, with their binary paths pointed at a driver
   built from the crate; the probe `EXP/rev6/probes/parseonly.rs` for the parser alone), and check that
   `taylor_with_budget`, `series_expansion` and `limit_via_series` handle that input on the same
   thread. Since revision 7rt2 the engine has no recursion over the input (§2.16, fact 3), so each must:
   a failure is a recursion the port added, and its remedy is a record of the work stack, never a
   parser limit. Record the limits in the report. On the prototype, HEAD's parser reads 216 levels of
   every mixed shape on 2 MB debug and 4124 on 8 MB release, and the §2.12 probe parser 266 and 4438;
   at each limit the parse and the call on one thread are `Ok` (gdb puts the overflow one level deeper
   in nom), and the engine alone handles 100 KB of every shape on 2 MB debug.
9. `tests/series_engine.rs` (new): §4.1 value, error and budget tests (they use the new parser);
   also run under `--release` (review 1 B2 structures). The round-trip part on engine outputs is a
   unit test in `series_engine.rs` (§4.3).
10. `rustmath-jupyter/src/repl.rs`: `TAYLOR_ORDER` stays 20000 (revision 7), calibrate `TAYLOR_WORK` in release with
    `EXP/rev3/calib3.py` against a driver built from the final crate (§5.3), doc comments of both
    constants with what each bounds (§3.1, review 2 m5) and the calibration table; `expr_to_latex` as a
    work-list writer (§3.1, revision 7rt1, red team 1 memory H1; port `latex_iter` from
    `EXP/rev7rt2/proto10/display.rs`, with `decimal_string` for integers); §4.5 tests.
11. Docs (the orchestrator's tracker pass after the unit, not the implementer, who edits no tracker):
    `PROJECT_TRACKER.md` rows by name (their line numbers move with every pass): "Taylor / Maclaurin
    series" (`series.rs:24`; truncated Laurent series arithmetic over symbolic constants, 🟢),
    "Laurent series" (`series.rs:89`; 🛑 to 🟢), "Asymptotic expansion" (`series.rs:129`), the
    expression parser row (`parser.rs:112`; bignum literals, `a/b` precedence), "Polynomial expansion"
    (`expand.rs:25`; 🔴 to 🟢, §2.17), "Derivative of a named function" (drop "Contaminates taylor(),
    series()"), the Expr row (iterative Drop); new rows for the out-of-scope items of §7;
    `docs/HANDOFF.md`: remove the S1 bullet (F1, the symbolic part of S7 and expand under a minus),
    add §7's new items (item 5 names the consumers of a result that still recurse, PartialEq and
    substitute, review 4 m3; the engine's own recursion over the input is gone since revision 7rt2;
    item 8 the limits revisions 5 and 6 found). `docs/magma/`: no row changes (the ch. 49 intrinsics map to
    rustmath-powerseries; the symbolic engine is not a MAGMA intrinsic).

Acceptance:

```
cargo test -p rustmath-symbolic --no-fail-fast
cargo test -p rustmath-symbolic --release --test series_engine      # review 1 B2 structures in release
cargo test -p rustmath-jupyter --no-fail-fast
cargo test -p rustmath-calculus -p rustmath-geometry -p rustmath-functions -p rustmath-manifolds -p rustmath-curves -p rustmath-plot3d --no-fail-fast
cargo check -p rustmath-benchmarks
cargo check --workspace --tests
cargo clippy -p rustmath-symbolic
python3 EXP/rev2/gpcheck3.py     # oracle (107 cases), against a driver built from the final crate if wanted
python3 EXP/revchecks.py; python3 EXP/rev2/revchecks3.py   # PROTO=<driver>: errors, structures, budget timings
PROTO=<debug driver> python3 EXP/rev7rt3/tools/budget6.py       # the budget tests of revisions 1 to 5, outcome, time and VmHWM
python3 EXP/rev7rt3/tools/newtests6_dbg.py                      # the budget tests of revision 6 (PROTO-style path in the script)
python3 EXP/rev7rt3/tools/calib6.py <release driver> 100000000 # §5.3 table for the TAYLOR_WORK doc comment
python3 EXP/rev7rt3/tools/b2probe.py <counting release driver> # review 7 B2: the three public entry points
<release driver built like EXP/rev7rt4/proto13/fnscan.rs>       # review 7 B1: the enclosure scan of §2.3
<release driver built like EXP/rev7rt4/proto13/nearscan.rs>     # red team 1 m1: the enclosures near MPFR's precision doubling
EXP/rev7rt3/tools/runall_rt3.sh  # with its binary paths pointed at drivers built from the crate (one of
                               # them with the counting allocator of EXP/rev7rt4/proto13/memcount.rs, as
                               # test tooling outside the crate): the self-attack, the heap peak per
                               # unit, depth against HEAD, the families of reviews 3 to 7 (§5.3),
                               # tools/redteam1_rt2.sh, every reproduction of red team 1 (towers, chains,
                               # deep results, the REPL taylor line, the near points, the literal bound),
                               # and tools/redteam2_rt2.sh, every reproduction of red team 2 (the binomial
                               # precision, the exponent span, the sparse log, the mixed nesting on the
                               # three threads, the too-wide enclosures, the long monomials, the Display
                               # round trip with a driver that links the crate's parser, PartialEq, the
                               # huge monomial powers), and tools/redteam3_rt3.sh, every
                               # reproduction of red team 3 (the empty arguments, the zero bases and
                               # 0^r, the result trees and their consumers, the shared-child Drop, the
                               # integral float, the shared renderings, the first-term limit path, the
                               # inverse hyperbolic enclosures, the large integer exponents)
EXP/rev7rt4/tools/rt4_extra.sh   # revision 7rt4: the exact_sentinel rows and maxfuzz_rt4.py (random
                               # products near 2^63 against the same inputs shifted down), buffers of
                               # 2^57 to 2^63 slots through every recurrence on 8 MB release and 2 MB
                               # debug threads, constant_poles, the exact-zero balls; with
                               # EXP/redteam4-series/tools/repro.sh, EXP/rev7rt4/rt4num/repro.sh and
                               # EXP/rev7rt4/tools/rt4_ms.sh (red team 4's reproductions, the tree
                               # estimate against the exact tree on red team 3's random inputs)
```

Risks:

- The engine is about 5700 lines in the prototype (with the public layer `api.rs` and the writers of
  `display.rs`). The revision 7rt4 prototype passes the same oracles (`EXP/rev7rt4/gpcheck3_rt4.out`,
  `revchecks_rt4.out` with the public precision in its exact-zero line, `revchecks3_rt4.out`,
  `budget_rt4_debug.txt`), as did revision 8's (`EXP/rev8/`). The revision 7rt3 prototype passes the same oracles (`EXP/rev7rt3/gpcheck3_rt3.out`,
  `revchecks_rt3.out`, `revchecks3_rt3.out`, `budget_rt2_debug.txt` in `EXP/rev7rt3/`), as did revision
  7rt2's (`EXP/rev7rt2/gpcheck3_rt2.out`, `revchecks_rt2.out`, `revchecks3_rt2.out`,
  `budget_rt2_debug.txt`) and revision 7rt1's
  (`EXP/rev7rt1/gpcheck3_rt1.out`, `revchecks_rt1.out`, `revchecks3_rt1.out`, `budget_rt1_debug.txt`)
  and revision 7's
  (`EXP/rev7/gpcheck3_r7.out`, `revchecks_r7.out`, `revchecks3_r7.out`, `budget7_debug.txt`). The revision 6 prototype passes 107 gp
  cross-checks (`python3 EXP/rev2/gpcheck3.py`, output `EXP/rev6/gpcheck3_r6.out`; exact for
  rational, parameter and `I` cases, 1e-60 relative for transcendental constants), the 26 checks of
  `EXP/revchecks.py` (plus its 10 budget timings; `EXP/rev6/revchecks_r6.out`) and the 18 checks of
  `EXP/rev2/revchecks3.py` (`EXP/rev6/revchecks3_r6.out`); keep all three as the oracle while
  porting, with the 31 budget inputs of `EXP/rev6/tools/budget6.py` (`EXP/rev6/budget6_debug_r6.txt`),
  the revision 6 budget tests of `EXP/rev6/tools/newtests6_dbg.py` and the probes of §5.3.
- Invariants (revision 5). §2.16 lists every loop, allocation and recursion of the engine with its
  bound. A port that adds one adds its row; the reviewer checks the port against the list (a grep for
  `for `, `while `, `loop`, `Vec::`, `BTreeMap`, `HashMap`, `BinaryHeap`, `collect`, `clone()` and
  recursive calls in the three engine files). Since revision 7rt2 the list has no recursion over the
  input: a recursive call of `expand` or `from_expr` in the port is a defect, whatever its frame size.
  Every `+`, `-` and `*` on an exponent or a precision is checked or saturating (§2.1), and every loop
  of a recurrence step stops at the offsets below k (§2.4). Each a-priori charge is computed from cached sizes in
  O(1) (or O(operands)); a charge computed by a walk over terms or pairs is a hole of the kind of review
  5's pair count, and so is an uncharged walk over a constant that the engine may test again and again
  (revision 6: `const_status`).
- Memory (revision 6). Every kind of heap object has a row in the table "Memory" of §2.16, at most 3
  bytes per unit with glibc's chunk overhead. A port that allocates a new kind of object, or a
  container that can grow per term or per pair (a map, a vector without its capacity, a clone of a
  name or of a top node), adds its row and its charge; the memory script of §5.3 with the counting
  driver checks the heap peak per unit on every family. Reviews 4 to 7 each found a container the
  charges missed (copies of coefficients, a one-entry B-tree, the sparse product's map, small
  constants, the rendering's nodes, owned coefficients in the public `SeriesExpansion`); the table is
  the list to check against. The memory script must measure through the public entry points
  (`taylor_with_budget`, `series_expansion` with `to_expr`, `limit_via_series`), as
  `EXP/rev7rt1/tools/b2probe.py` does, not through an internal rendering.
- Enclosure cost (revision 7). An MPFR function whose cost depends on its argument (erf, review 7 B1)
  is either guarded so that it runs only where its cost is bounded, or charged by its argument. A new
  enclosed function needs a row in the scan of §2.3 (`EXP/rev7rt2/proto10/fnscan.rs`) with its slowest
  rate at 64, 256 and 1024 bits, and the arguments near which MPFR raises its working precision in
  `nearscan.rs` (red team 1, m1: sin near +-1, log near 1), timed as medians.
- Recursion over the input (revisions 5 to 7rt2). Revisions 5 to 7rt1 kept the recursive `expand` and
  `from_expr` within the parser's stack by keeping their frames small, which red team 2 broke with a
  text that used every grammar level in each parenthesis (H1). Since revision 7rt2 neither recurses:
  each is one loop over an explicit work stack. A port that reintroduces a recursive call (for a new
  node kind, or by calling `expand` from inside a record) fails `deep_api_inputs` and
  `deep_mixed_nesting` (50 000 and 10 000 levels on a 2 MB debug thread); the remedy is a record.
- The prototype emulates `Result` threading with a thread-local fault flag. The port must charge at
  the same points; a missed charge is a budget hole. A grep for constant arithmetic calls that do not
  pass `w` finds them, since every operation takes `&mut Work`.
- Big-number arithmetic outside the charged layer is a budget hole (review 3, B1 and B2). In
  `series_const.rs`, `series_engine.rs` and `series.rs`, outside the section "charged arithmetic",
  a grep for `Rational::new`, `.gcd(`, `.lcm(`, `.pow(`, `nth_root`, `exact_root`, `.sqrt(` and the
  operators `+ - * /` applied to `Rational` or `Integer` values must find only exponent arithmetic on
  `i64` and constant-size values (`qi(k)`, `1/k` for a recurrence index k, the parser's literals),
  and the trial division of §2.2 (charged as a whole before it runs). The implementer runs it before
  the budget tests and lists the remaining hits in the report.
- Copies are a budget and memory hole when they are not shared or not charged (review 4, B1). Two
  more greps, in the same files: every construction of a term map (`BTreeMap::new`,
  `Arc::new(` of a map, `.collect()` into a map, `(*c.terms).clone()`, `Arc::make_mut`) must sit in
  a function that charged `copy_units` or a charged-layer operation per term first; every `to_expr(`
  outside `to_expr` itself must be the charged one (`to_expr_raw` only inside kernel creation); and
  every copy of terms of a `Laurent` (`.clone()`, `.to_vec()`, `.cloned()`, a slice copy) must sit in
  an operation that charges `SER_TERM` per term. The implementer lists the remaining hits.
- Kernel expressions are DAGs (§2.2). Any new walk of a kernel's `Expr` (a size estimate, a
  predicate, a printer) must stop at a limit or memoize per shared node; a plain recursive or
  stack walk is exponential in the nesting (review 4, B2: the revision 3 key and `symbols()` walks
  took 2.56 GB for a 204-character input). A grep for `.symbols()`, `contains_symbol` and
  hand-written `match` walks over a kernel's `expr` in the three engine files finds them.
- Port traps (review 4, m5, and the prototype's emulation). Up to revision 7 the binomial path of
  `pow_int` built a second context with a dummy `Work { used: 0, max: 0 }` (`EXP/rev7/proto8/ser.rs:770`,
  red team 1 m3); that is harmless with the prototype's thread-local counter, but a literal port with an
  explicit `&mut Work` would refuse at once or charge nothing inside an O(S^3) loop, a budget hole.
  `EXP/rev7rt2/proto10/ser.rs` runs the loop in the caller's context with the working precision set to
  the slots; the port passes the same `&mut Work` (a `Ctx` that reborrows it with another `p`). The `mul` pre-charge
  must compute each term's `words()` once (the revision 3 loop recomputed it for every pair, which
  is O(terms) per pair; `EXP/rev4/proto5/ser.rs` precomputes them). The tests `(-q.clone()).is_one()`
  in `pow_u`, `power_size` and the rendering copy a coefficient to compare it with -1: compare the
  sign and the magnitude instead. A budget refusal inside the zero test must be returned as
  `BudgetExceeded` before any `UndecidedZero` constant is built (§2.2; the prototype's fault flag
  needed the `undecided` helper for this, a port with `?` gets it for free).
- The cost constants (`OP_BASE`, 1 unit per word pair, `P(W) = W^2/64 + W`, `w^2` per root,
  `MONO_FACTOR_UNITS`, `SEARCH_CMP`, `EXPR_NODE`, `KERNEL_UNITS`) were measured with Lehmer's gcd (I3),
  num-bigint and the standard B-tree. Products are charged by the schoolbook bound, so very large
  coefficients are over-charged (`x^L` at 1, L a 99 990-digit literal, is refused where HEAD takes
  3 ms; §5.3): an honest refusal, not a hole. A later change of `Integer::gcd` or of the
  multiplication algorithm changes the rates; the calibration of §5.3 and the micro-benchmarks
  (`EXP/rev3/micro.rs`, `micro_small.rs`) are the check.
- Kernel keys are the prefix code of §2.2 over the arguments' normal forms, injective by
  construction, so two kernels with different arguments never merge. The interning table lives as
  long as one expansion, and the keys contain ids, so a key and an id must never be compared across
  two `Work`s (the API converts every constant to `Expr` before it returns, so none escapes).
- Ball soundness rests on R1 and R2 of §2.3 being applied to every MPFR value. Any new ball
  operation must pass its results through the same checked constructor; a raw `RealMPFR` operation
  whose result is not checked is a soundness hole (B1 of review 2). A grep for `RealMPFR` arithmetic
  outside the checked helpers finds them.
- Zero-test soundness rests on the rule list of §2.2. A reviewer should check each `NonZero` rule
  against its proof; `Opaque` is the safe default for anything new; a constant with a possibly
  non-real kernel never gets a numeric `NonZero`.
- Generic parameters: a result for `1/(x + a - b)` is valid for a != b only. This is the documented
  convention (gp, Maxima, the crate's Risch module), stated in every doc comment.
- The stack-gated `Drop` relies on stack addresses decreasing with depth; elsewhere it degrades to
  the HEAD behaviour. Units implemented after S1 must match on references (E0509 tells them).
- Display/parse: §2.11 changes printed forms of `Pow` with rational or negative operands; grep shows
  no test depends on them.
- REPL: `TAYLOR_ORDER` keeps HEAD's value, so the order check refuses what HEAD refuses (revision 7;
  revision 6 lowered it, review 7 m1).
- Build: `rustmath-symbolic` now links MPFR through rustmath-reals (already built in the workspace).

## 7. Out of scope (with honest interim behaviour)

1. **`rustmath-calculus/src/taylor.rs:35`** duplicates the old algorithm (same exponential growth) and
   overflows its i64 factorial at order 21: `taylor(5*x, "x", 0, 25)` panics with "attempt to multiply
   with overflow" (`taylor.rs:62`, debug; wraps in release). Its `laurent` (`:131`, tracker 🛑, the rustmath-calculus row "Laurent series")
   is the same facade. It is another crate, not named in the item, and returns `Expr`; delegating to
   `Expr::taylor` needs a signature change there. New HANDOFF row.
2. **Puiseux and logarithmic expansions** (sqrt(x), log(x), x^x at 0, and so sqrt(x)^2 or exp(log(x))
   expanded bottom-up): MAGMA ch. 49 §49.1.2 Puiseux series need fractional exponents, and log terms
   are not series at all. Interim: `NotLaurent` (documented as "a subexpression has no Laurent
   expansion").
3. **gamma, zeta, factorial, named functions of the variable**: `Unsupported`. Needed: their series
   (polygamma constants) or derivative hooks (the `FunctionRegistry` derivative is not wired,
   tracker row "Derivative of a named function").
4. **Faster series arithmetic** (§5.2): Newton iteration and fast multiplication; the budget keeps the
   REPL honest meanwhile.
5. **Iterative PartialEq and substitute** for deep trees from other sources (PartialEq and
   substitute overflow at 40000 and 20000 left-deep terms, §1.2). Not named in the item (which names
   simplify and Drop), and the engine's output is balanced with a capped kernel depth (§2.15). Each
   needs its own explicit-stack rewrite.
   Display is no longer on this list: it writes with a work list since revision 7rt1 (§2.11), because a
   result can be twice as deep as its input (red team 1, H2), and the REPL's LaTeX writer likewise
   (§3.1). PartialEq is not claimed for every result of a parsed text: a result can be as deep as
   its input (a left-deep chain inside a call: `x + sin(1+...+1)` with 20 000 ones has depth 20 003,
   which PartialEq cannot compare on a 2 MB debug thread, nor the input; red team 2 structure m2), and
   §2.15 gives its measured limits. How many nodes these consumers visit is bounded since revision
   7rt3: a result's tree has at most `MAX_RESULT_TREE = 2^21` nodes (§2.15, red team 3 structure H2).
   Since revision 5 the engine reads chains of `+ -` and `*` with a loop and its pre-pass is
   iterative, so a left-deep sum costs no recursion (65 536 terms on a 2 MB debug thread; HEAD's
   `taylor` overflows at 12 800 on an 8 MB release thread); since revision 7rt1 `/` belongs to the `*`
   chain and towers are read along their right spine; since revision 7rt2 `expand` and `from_expr`
   run one loop over an explicit work stack, so the engine has no recursion over the input at all
   (red team 2 structure H1: revision 7rt1 still recursed once per nested operand, and a text using
   five grammar levels per parenthesis overflowed the engine before the parser). The parser keeps
   HEAD's recursion over its nesting, the only one over a text (§2.16 fact 3). The REPL arm no longer
   calls `args[0].symbols()` (§3.1). New HANDOFF row, naming `symbols()` (`substitute.rs:345`) in the
   other REPL arms, PartialEq and substitute. HANDOFF's note
   "a symbolic chain `w + w + ... + w` of 3000 terms aborts the debug driver in `Expr` Display and Drop
   (S1's area)" (§1.8) is fixed in both halves: Drop by §2.14, Display by its work list (§2.11, revision
   7rt1); and the note "`-(-x)` displays `--x`" by §2.11 (revision 7, review 7 m2).
6. **Products of radicals** (`sqrt(6) - sqrt(2)*sqrt(3)`): needs a prime factorization of every
   radicand. Interim: `UndecidedZero` when such a constant must be inverted (sound). The same holds
   for large rational powers kept as kernels: `2^(-300000000) * 2^300000000 - 1` and
   `(1/2)^1073741827 * 2^1073741822 - 1/32` are 0, but the two powers are distinct kernels and their
   balls leave the window, so they are `UndecidedZero` (review 2 test inputs). Needed: kernels
   `b^n` over a canonical base b (a positive integer that is not a perfect power, as for radicals)
   with a folding rule back to a rational for small exponents, and `exp(q)` for rational q
   represented as `e^q`. Neither is needed for soundness.
7. Other defects seen but not touched: `differentiate` of erf uses a rational approximation of
   2/sqrt(pi) (`differentiate.rs:197`); `eval_float` returns None for `pi`/`e`.
8. **Revision 5 limits, honest but costly** (found by the self-attack, §5.3; new HANDOFF rows).
   Kernels the engine makes from computed values render their arguments' normal forms, and a tree
   costs every consumer that walks it, not only the text: revision 7rt2's result of
   `(1+exp(...x...))^2` with 10 levels at 1, order 1, was a tree of 1.04e7 nodes, and `simplify` of
   it took 960 MB (red team 3, structure H2). Since revision 7rt3 the value at the point of a regular
   input node renders as that node with the point substituted when its normal form is much larger
   (§2.5), so a value at the point no longer doubles per level (the order-0 result of 30 levels is
   the input with x = 1), and the result's tree is capped (`MAX_RESULT_TREE = 2^21`, §2.15). What
   remains is the expanded normal form itself: the x coefficient of the same input has 2^k terms, so
   from 12 levels the result's tree passes the cap (`LimitExceeded(TreeSize)`; 10 levels give a tree
   of 462 903 nodes, 11 levels 1 036 403; revision 8 468 023 and 1 046 643) and from 16 levels the budget refuses it, where HEAD keeps the
   product and returns in 0.2 ms. The cap refuses 12 to 15 levels although the engine's own work there
   is small (revision 8, review 8 m4: 10 levels take 2.4e6 units, 11 levels 5.0e6; 12 to 15 levels are
   refused after 36 ms to 0.26 s, below the budget): the cost is the expanded normal form, not the
   computation. The HANDOFF row records one remedy for both refusals, the cap's and the budget's: a
   product-preserving representation of coefficients; so would consumers memoized by `Arc` address (`simplify`, `substitute`, `symbols`, a
   `PartialEq` with `Arc::ptr_eq` and a pair memo), which are outside S1's item. Products of big
   coefficients are charged by the schoolbook bound, so `x^L` at 1 with a 99 990-digit literal L is
   refused, where HEAD takes 3 ms; a Karatsuba-aware charge would admit it. Revision 6 adds one more
   instance of the expanded normal form: strict inverses of computed leading coefficients render
   P0's normal form, and `invpowx(k)` (`(1)/((1 + ...)^511)` nested k times around `x + sin(1)`)
   multiplies 512-term constants at every level, so it is refused by the budget after 0.35 s and,
   without a budget, by the term cap after 2.5 s (674 MB), where HEAD returns in 1 ms. Revision 6's
   charges also refuse `tan(x)` at 1 from order 99 on (revision 5 admitted order 100 with 8.2e7
   units), the price of charging every constant its header. Revision 7rt4 (red team 4 structure, m2):
   the expanded normal form is also polynomially larger than HEAD's nested derivative for towers and
   nested functions, whose coefficients name deep kernels many times (a k-level tower at order n names
   them about k^(n+1) times, a tree of about k^(n+2) nodes, few of them distinct): `x^x^...^x` at 1/2,
   order 2, passes the cap from 22 carets (45 bytes; HEAD 8 ms), at 2, order 1, from 137 carets, and
   the 150-level `cos(x+1)^...^(x+2)` at 0, order 1 (HEAD 19 ms); each refusal takes at most 0.25 s
   (§2.15). The remedy is the same (coefficients that keep products, or consumers memoized by `Arc`
   address with a cap on the DAG); the HANDOFF row of this item should name the towers too (the
   orchestrator's tracker pass).
9. **Results kept by one REPL line or cell** (red team 3 memory, m2; revision 7rt3). `TAYLOR_WORK`
   bounds each call, and a result is most of its call's memory and outlives it: a line that adds 11
   taylor calls of about 9.8e7 units each (91.7 KB) held 3.08 GB, and 400 calls of `taylor((1)/(1-x),
   x, 0, 20000)` in one line 2.26 GB in 6.1 s (about 17 GB and 46 s at 100 KB, extrapolated); two lines
   with the first kept as `_` hold both. HEAD finishes neither line (its taylor is exponential), so
   this is no regression, and J1b's REPL model is per call (an interrupt or `time_limit` stops a cell
   between calls; two `EuclideanSpace(7000000)` values hold 385 MB each). The tree cap (§2.15) bounds a
   result's nodes at about 200 MB, but not its integer digits. Needed: a per-cell resource policy for
   every command of the REPL (J1b's area), or, for taylor alone, one `TAYLOR_WORK` shared by the taylor
   calls of a cell (the arm would keep the units left in the cell and pass them as the budget, which
   needs `series_expansion` to return the units it used). New HANDOFF row.
10. **Two numeric gaps left** (red team 4 numeric, m1 and m3; revision 7rt4). First, a function at a
   pole of a non-rational argument: `arctanh(c)`, `zeta(c)`, `gamma(c)`, `log(c)` or `tan(c)` for a
   computed c that equals a pole stays a kernel (`Opaque`, or `NonZero` by the rules of §2.2); its
   inverse is `UndecidedZero` (the ball of the argument meets the pole), but a product `x f(c)` has a
   first term of exponent 1, and the limit path reports `Finite(0)`. The rational case is refused
   since revision 7rt4 (§2.2). Needed: a pole test (a zero test of `c - 1`, `c + 1`, `c`, `cos c`, or
   the distance of c to the non-positive integers) before such a kernel is created, at the cost of a
   zero test per kernel; HEAD gives the same outputs. Second, a hint or a kernel's source that writes an
   exactly-zero operand as a difference of equal non-dyadic rationals (`sqrt(1/3 - 1/3)`) keeps a ball
   around 0, so `(1)/(x + sin(sqrt(1/3-1/3) + 1) - 1/2)` at 0 is `UndecidedZero` where the normal form
   is decided; the dyadic and integer literals are exact balls since revision 7rt4 (§2.3). Needed: an
   exact rational value of a small literal subtree (as `small_int_value` reads an integer one), or a
   retry of the enclosure on the normal form of the constant.

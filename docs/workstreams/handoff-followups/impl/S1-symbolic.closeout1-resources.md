# S1 closeout check 1 (resources lens): the fix of m1 and m2

Checked: branch `claude/gracious-gates-e6b6l7`, HEAD 226a7ec, plus the uncommitted change of the
step "C-1 minors check items" (`git diff`, five files, saved as
`closeout/check1-resources/diff_checked.patch`, md5 d22ecedec8e73c3d6983f4adb1b015b4; the diff was
the same at the end of the check). Compared against HEAD (its Rust code is a4f7791) and against the
approved C-1 state ("C-1 code" below: a4f7791 with `c1mcheck/c1minors_delta.patch` reversed, rebuilt
here; its sources are byte-identical to `c1mcheck/pm/`), and on the residual inputs against
HEAD-before-S1 (8a0b310, `fin1-resources/headsrc`). Read: the diff whole, the section "C-1 minors
check items" of `S1_impl.md`, `S1.c1minorscheck.md`, the engine code around every reader of a class
(`Tail`), `Konst::generic_nonconstant` and its callers, `intern`, `cball_konst` and `ball_cost`.
Paths below are relative to `fu2/s1impl/closeout/check1-resources/` unless they start with
`rustmath-`.

Nothing in the repository was edited, committed, pushed, stashed or reset. `git status --short`
lists the five files of the step only, `rustmath-misc/output.txt` has md5
746308829575e17c3331bbcb00c0898b, no `.tmp_dir_*` exists, `tests/data` has no `__pycache__`.
`target/release` was deleted after the release runs; the scratch libraries, executables and source
copies were deleted (`target/debug` is kept, as it was).

The machine was shared: the load average was 4 to 8 during most release runs (another job in
`RustMath-wpf`, the semantics checker, my own debug runs). Times below are wall times under that load;
the units (work charged) do not depend on it.

## Verdict

Approve. No blocker under the narrowed bar.

- (c) Every entry point (`taylor_with_budget`, `series_expansion` with `to_expr`, `Display`,
  `Debug`, re-parse and `Drop` of the result, `limit_via_series`, `limit_with_error`) on 109 rows
  (input and order) of up to 100 000 bytes, 67 of them built for this check to defeat the laziness
  of the fix: at most
  2.76 s per entry point in release on an 8 MB thread (1.6 to 1.9 s on reruns at load 2), at most
  45 MB VmHWM (46 MB in debug), no stack overflow, abort or panic on a 2 MB debug thread (at most 11.3 s per entry
  point there). The new code adds O(1) work per record of the work stack, which is charged.
- (b) No value changed and no value became a refusal against HEAD on any of these inputs; the only
  outcome changes against HEAD are refusals that became values (`scflip.txt` and two more inputs of
  the same kind, and `limit_via_series` / `limit_with_error` of `exp(...)` of the scflip sum).
- (a) No new value comes from the change: where a cut check can read a product's class it is the
  class HEAD computed (same enclosure); elsewhere nothing reads it.
- `res/scflip.txt` at 0, order 30, REPL budget: `taylor_with_budget` Ok, `series_expansion` Ok with
  the C-1 code's value (same output hash), `limit_with_error` `Finite(0)` with its bound; least
  budget 61 988 831, the C-1 code's exactly (HEAD 102 996 826).
- m2: the new doc of `generic_nonconstant` is true.
- Tests: debug suite 911 passed, 0 failed, 36 ignored; release `s1_cuts` 16 passed and both ignored
  release tests pass (`scaled_products_keep_their_value_at_the_repl_budget_release` in 1.24 s).

One minor item (r1: the m1 cost remains for products under a branch function whose argument does
not need the class; a 100 KB input that the C-1 code expands is still refused by the budget, as on
HEAD) and three nits.

## 1. The change against the code

### m1 (`rustmath-symbolic/src/series_engine.rs`)

- `expand` (`:2527`) counts the records on its stack whose step checks a cut (`cut_records`): raised
  when `expand_visit` returns such a record (`:2541`), before it is pushed, lowered when the record
  is popped after `EStep::Done` (`:2565`). Records are pushed in one place and popped in one place,
  `checks_a_cut` (`:2299`) is a function of the record's kind and unary operator, which do not
  change while the record is on the stack, so the counter cannot drift or underflow. An error leaves
  `expand` with `?` and drops the stack and the counter together.
- A product record notes `cut_records > 0` when it is pushed (`EProd::class_read`, `:2175`, set at
  `:2539`); `finish_product` (`:2607`) calls `Tail::scaled_enclosed` (`:1228`) only then and
  `Tail::scaled` otherwise (`:2623`).
- Readers of a class. I listed every use of `Tail` fields: the only decision is `cut_check`
  (`:1531`, `tb.re` / `tb.im` read first), called from `log` (`:779`), `pow_const` with `Some(tb)`
  (`:895`; callers `Sqrt` `:2892` and `pow_by_const` `:1729`, the tower's constant exponent) and
  `atan_like` (`:1587` to `:1589`); `pow_general_log` (`:2681`) calls `log` for a tower with a series
  exponent. Every other read (`expand_unary`'s odd/even classes, `pow_class`, `pow_int_class`,
  `full_class`, the chain's `times`, the sum's `and`, erf's class) only computes the class of the
  record's own value. A value is handed only to the record below it on the stack, so a product's
  class reaches a cut check only through records that are below the product while it is on the
  stack. `checks_a_cut` is true for exactly the records that can call `cut_check` (log, sqrt, the
  inverse functions other than erf, every tower). No series value is memoized across occurrences
  (`SUBST` holds substituted expressions, not series), so a class computed without the enclosure
  cannot be reused under a cut check later. The rule is therefore exact in what it spares and
  conservative in what it keeps (see r1).
- Record sizes (`size_of`, measured on scratch copies of HEAD and of the fix): `EProd` 112 bytes,
  `ESum` 96, `ETower` 96, `EFrame` 16, the same on HEAD and on the fix (the new `bool` sits in the
  padding of the three flags already there). Memory per record is unchanged.
- Order of the charges (nit n1). HEAD computed the class after scaling the series
  (`Ok((scale(&a, &c)?, tail.scaled_enclosed(&c)))`); the fix computes it before `scale`. The set of
  charges is the same, so outcomes and least budgets are the same (identical least budgets on the
  class-read inputs of section 4; the implementer reports the 91 773 rows of the corpora identical), but the charge
  that crosses the budget differs, so a `BudgetExceeded` message can report another unit total:
  `deep_prod_60` (99 814 bytes, all products under one log) reports 100 000 054 units on the fix and
  100 000 019 on HEAD. A scratch build of the fix with HEAD's order (`scale` first) reports HEAD's
  100 000 019 and 100 000 084 (`deep_prod_120`) exactly, and a build that prints every product left
  unenclosed prints none on this input, so the difference is the order and not a missed enclosure.

### m2 (`rustmath-symbolic/src/series_const.rs:2697` to `:2705`)

The code is unchanged (the diff of `series_const.rs` is the doc comment only). The doc is true:

- the polynomial path calls `indet_groups`, which charges two copies and the sort before its loop;
- the one-kernel walk (`:2723` to `:2736`) reads each factor of each term once by reference, and
  compares kernels by id (`Kernel::eq` is `id == id`, O(1)), so it is linear in the constant and
  allocates nothing; the grouping after it is charged before it runs (`:2743`), as `indet_groups`
  charges;
- its callers are `position` (`series_engine.rs:1387`), on the constant term of the argument, which
  is a value that its construction paid for (`konst_of` does not memoize by address, so a shared
  subtree of a DAG input is converted and charged per occurrence), and `kernel_generic_nonconstant`
  (`series_const.rs:3599`), which runs inside `intern`'s closure only for a new key, after the key
  (at least the size of the argument's normal form) was charged.

Nit n2: the doc names two uncharged early exits of the walk (a parameter factor, a second kernel
with a parameter); the walk also returns false without a charge when the one kernel is not `gnc`
(`:2739`). The sentence before ("the walk ... is not charged") covers it, so the cost statement is
true; the list is just not complete.

## 2. `res/scflip.txt` at the REPL budget

99 800 bytes (md5 669766839a384d03a4163fa6a43fe6c8), at 0, order 30, budget 1e8
(`out/scflip_rerun.log`, release, 8 MB thread, load about 2):

| | fix | HEAD | C-1 code |
|---|---|---|---|
| `taylor_with_budget` | Ok, 0.36 to 0.46 s | `BudgetExceeded` (100 000 104 units), 0.52 to 0.58 s | Ok, 0.37 s |
| `series_expansion` (+ `to_expr`, `Display` 21 824 chars, `Debug`, re-parse true, `Drop`) | Ok, 0.33 to 0.42 s, the C-1 code's output (hash 1272ee4ef0be16fd) | `BudgetExceeded` | Ok, same hash |
| `limit_via_series` | `Finite(0)`, 0.05 s | `Finite(0)`, 0.20 s | `Finite(0)` |
| `limit_with_error` | `Finite(0)` with a bound, 0.38 to 0.44 s | `Finite(0)` without a bound | `Finite(0)` with a bound (same hash as the fix) |
| VmHWM | 34 MB | 35 to 37 MB | 34 MB |
| least budget (`series_expansion`, `drv/mb.rs`) | 61 988 831 | 102 996 826 | 61 988 831 |

Debug build on a 2 MB thread (`bin/ep_dbg`, linked against cargo's fresh
`target/debug/librustmath_symbolic.rlib`): all four entry points as in release, the same output
hashes, at most 2.57 s per entry point, 35 MB. At order 3 the least budget is 14 543 533 on the fix
and on the C-1 code (HEAD 55 313 110).

## 3. Bar (c): inputs built against the change

Drivers: `drv/ep.rs` (the minors check's driver plus `Drop` timing and an FNV hash of each printed
outcome, so that outcomes can be compared across libraries), `drv/mb.rs` (least budget by
bisection). Libraries: the fix, HEAD and the C-1 code compiled from source copies as release rlibs
with the same flags (`buildlib.sh`, `builddrv.sh`), and the fix in debug. Inputs (`gen.py`,
`gen2.py`, all at most 100 000 bytes; `inp/`, `inp2/`; lists `list.txt`, `list2.txt`, `list3.txt`):

- the scflip sum under each kind of record: `log(2 + S)`, `log(-1 + S)`, `sqrt(2 + S)`,
  `(2 + sin(x))^S` (class read), `exp(S)`, `cos(erf(S))` (not read), `log(2 + exp(cos(S)))` (a cut
  record far below every product), orders 3 and 30;
- many distinct constants real or imaginary only by their enclosure, on a cut, where the class
  decides (`log(-1 + sum (log(-1)^2+k) sin(kx))`, the same in `arctan(2I + ...)`,
  `arcsinh(3I + ...)`, 3 500 separate `log(-1 + ...)` terms, at infinity);
- constants that differ only deep inside (`log(-1)^2 * sin^d(k)` for d = 20, 100, 240, an arctan
  chain of 100), constants just under and over the 256-node tree cap of the enclosure;
- deep products: right-nested chains `c1*(c2*(...*(c_d*sin(x))))` of depth 10, 60, 120 under one
  log, and a chain with an `exp` between the products;
- repeated cut checks whose class decides: nests of depth 5, 30, 100 of
  `log(-1 + c*f) - log(-1)`, `arctan(2I + c*g) - arctan(2I)`, `sqrt(-1 + I*c*h) - sqrt(-1)` (each
  level's constant term cancels exactly, so every level's argument lies on its cut and is a value
  only through the enclosed class; checked on one item: the fix and HEAD give the value, the C-1
  code refuses);
- 30 levels of `exp` around each product, with and without a log at the top (the counter stays 1
  across 30 non-cut records);
- constants with large normal forms in every product (`(log(2+I)+log(3+I)+1)^4`, `^6`) bare, under
  exp and under log, and `log(2 + S_n)` for n = 150, 250, 350 at order 5.

Results (release, 8 MB thread; debug, 2 MB thread; `table.py`; logs `out/adv_*.log`,
`out/adv2_*.log`, `out/rel3_*.log`, `out/dbg_*.log`):

| | release (fix) | release (HEAD) | debug (fix) |
|---|---|---|---|
| largest time of one entry point | 2.76 s (`nest_atan_100`, order 3, `BudgetExceeded`, load 7.7); 1.59 to 1.83 s on two reruns at load 2 (`out/nest_atan_100_rerun.log`) | 1.79 s (`nest_atan_30`), 1.52 to 1.66 s on the reruns | 11.3 s (`nest_atan_100`) |
| largest VmHWM | 41 MB | 41 MB | 42 MB |
| stack overflow, abort, panic, nonzero exit | none | none | none |

The 41 inputs of `c1mcheck/res/res_list.txt` plus `scflip.txt` (`out/res_*.log`, `out/dbg_res.log`):
fix at most 1.46 s per entry point (`sc_many`, `BudgetExceeded`), 45 MB (`gnc_poly_many`); HEAD
2.09 s on the same input; debug 7.41 s, 46 MB; no overflow.

Every `BudgetExceeded` above is reached in under 3 s in release: the enclosures are charged before
they run (`cball_konst`: `COMPLEX_COST_FACTOR` times `ball_cost`, which counts every node, the bits
of integer leaves and the trig and erf costs), and the observed rate on these inputs is 4 to 30 ns per
unit, far from the 200 ns per unit at which 1e8 units would take 20 s.

## 4. Bar (b) and cost: outcomes and least budgets

Outcomes (series kind and hash, taylor, both limits) of the fix against HEAD:

| inputs | rows | differ | how |
|---|---:|---:|---|
| `list.txt` | 39 | 1 input | `flip_exp` (orders 3 and 30): `limit_via_series` and `limit_with_error` `Unknown` on HEAD, `Finite(1)` on the fix (with a bound at order 3); exp of a series with constant term 0, so 1 is right. Series and taylor identical. |
| `list2.txt` | 22 | 0 | |
| `list3.txt` | 6 | 2 | `powc4_exp`, `powc4_bare`: `BudgetExceeded` on HEAD, values on the fix, equal to the C-1 code's (same hashes) |
| `res_list.txt` | 42 | 1 | `scflip` (section 2) |
| debug against release (fix), all rows | 109 | 0 | |

Least budgets of `series_expansion` (`out/mb_*.tsv`, `out/mb4.log`, `out/mb3.log`; "read" = a cut
record lies below the products):

| input (order 3 unless said) | class | fix | HEAD | C-1 code |
|---|---|---:|---:|---:|
| `scflip` (order 30) | not read | 61 988 831 | 102 996 826 | 61 988 831 |
| `scflip` | not read | 14 543 533 | 55 313 110 | 14 543 533 |
| `flip_exp` | not read | 15 974 045 | 56 743 621 | 15 974 045 |
| `flip_erf_cos` | not read | 17 881 393 | 58 650 970 | 17 881 393 |
| `powc4_bare` | not read | 54 359 436 | 257 492 065 | 54 359 436 |
| `powc4_exp` | not read | 54 836 273 | 257 492 065 | 54 836 273 |
| `flip_log2` | read | 57 220 459 | 57 220 459 | 16 212 464 |
| `flip_sqrt` | read | 57 220 459 | 57 220 459 | 16 331 673 |
| `flip_tower` | read | 57 697 296 | 57 697 296 | 16 808 510 |
| `flip_far_below` | read | 56 743 621 | 56 743 621 | 15 974 045 |
| `powc4_log2` | read | 257 492 065 | 257 492 065 | 54 836 273 |
| `log(2 + S_150)` (order 5) | read | 105 857 849 | 105 857 849 | 92 506 409 |
| `real_many_logm1`, `imag_many_atan`, `deep_prod_10`, `big_const_40` | read, the class decides | equal to HEAD | | refused (`NotLaurent`) |

Where nothing reads the class the fix needs exactly the C-1 code's budget; where a cut record lies
below, exactly HEAD's (the charge order of n1 does not move a least budget).

## 5. Uncharged work

- New code: `checks_a_cut` is a match on the record kind, the counter an increment or decrement,
  `class_read` one store; each happens once per record push or pop, and every push follows a
  charged node visit (`KONST_NODE`) and, for a chain or a tower, `RECORD_UNITS`. Nothing new loops.
- `Tail::scaled` (the path the fix now takes more often) reads two cached flags.
- m2: as in section 1, the uncharged walk is linear in a constant whose construction or key was
  charged, and runs once per cut check or per new kernel.

## 6. Tests

- `cargo test -j 2 -p rustmath-symbolic --no-fail-fast -- --test-threads=2` (debug, `sym_dbg.log`,
  139 s): unit 686 (5 ignored), `s1_cuts` 16 (2), `s1_depth` 14, `s1_finish1` 11 (1),
  `s1_fuzz_data` 7, `s1_payload` 2 (1), `s1_regressions` 26 (2), `s1_review2` 2, `s1_review3` 7 (7),
  `s1_review4` 4 (2), `s1_review4_memory` 1 (1), `s1_review5` 4 (2), `s1_review5_memory` 2 (2),
  `s1_review6` 4, `s1_singular_points` 9, doc 116 (11): 911 passed, 0 failed, 36 ignored. The three
  new tests are among them (`cut_decisions_on_products_scaled_by_an_enclosure_are_kept` and
  `products_whose_class_no_cut_check_reads_are_not_enclosed` pass in debug).
- `cargo test --release -j 2 -p rustmath-symbolic --test s1_cuts` (`rel_cuts.log`): 16 passed,
  2 ignored; with `-- --ignored --test-threads=1` (`rel_cuts_ign.log`): 2 passed,
  `cut_rule_within_the_bar_release` (0.40, 0.49, 2.88 s) and
  `scaled_products_keep_their_value_at_the_repl_budget_release` (1.24 s).
- The test diff only adds lines (163 in `s1_cuts.rs`, 12 rows in `s1_cuts.tsv`, 20 in the
  generator); no assertion was changed. `rustfmt --edition 2021 --check` is clean on the three Rust
  files; no added line has an em-dash or `unsafe`.

## 7. Minor items and nits

**r1 (minor). The m1 cost remains under a branch function whose argument does not need the class.**
The rule encloses every product with a cut record below it, also when the cut check would decide
without the class (`log(2 + ...)`: the constant term 2 is off the cut, and `position` decides that
at once, but `cut_check` reads `tb` first). Example: `powc4_log2` (`inp/powc4_log2.txt`, 99 979
bytes: `log(2 + sum_k (log(2+I)+log(3+I)+1)^4*sin(k*x))`, 2 660 products with one shared
constant) at 0, order 3, REPL budget: the C-1 code gives a value (least budget 5.48e7), HEAD and the
fix `BudgetExceeded` in 0.36 s (least budget 2.57e8, 4.7 times); HEAD-before-S1's `taylor` returns
a 5.9 MB expression in 4.8 s (`out/pre.log`; at order 5 it exhausts 4 GB). Also `log(2 + S_150)`
at order 5 (30 798 bytes): C-1 code Ok, HEAD and the fix `BudgetExceeded`. This is not a regression
against HEAD (the fix gives exactly HEAD's outcome and budget on every class-read input) and not
part of m1's stated requirement (`scflip.txt`, which the fix repairs); against HEAD-before-S1 it is
the kind of honest budget refusal the earlier reviews graded minor (`S1.c1-1-regressions.md` 4.2),
and HEAD already has it. The implementer's handoff names the finer rule (read the deferred class in
`cut_check` only when the position of the constant term does not decide); a cache of the enclosed
class per constant (the minors check's other option) would also repair this input, since all its
products share one constant. Owner's call.

**n1 (nit). Charge order.** The class is now computed before `scale` (HEAD: after), so a
`BudgetExceeded` message on a class-read input can report a different unit total (section 1); the
report's "computed exactly as on HEAD" holds for the class and the total work, not for the order.
No outcome or least budget moves.

**n2 (nit). m2 doc list.** The early exits named in the doc omit the non-`gnc` kernel (section 1).

**O1 (pre-existing, unchanged).** The doc of `EFrame` (`series_engine.rs:2194`) says 24 bytes; it is
16. The doc of `RECORD_UNITS` (`series_const.rs:4607`) says a boxed record is at most 96 bytes;
`EProd` is 112 (on HEAD too). Memory stays far inside the bar (45 MB at most here); not graded.

## 8. Files

`closeout/check1-resources/`: `diff_checked.patch` (the diff checked); `buildlib.sh`, `builddrv.sh`,
`run.sh`, `cmp.py`, `table.py`; `drv/` (`ep.rs`, `mb.rs`, `pre.rs` for HEAD-before-S1, the size
probe `sz_*.rs`, per-library copies); `gen.py`, `gen2.py`, `inp/`, `inp2/`, `list*.txt`,
`res_list.txt`, `mb*.tsv`; `out/` (all logs: `scflip_rel.log`, `scflip_rerun.log`, `adv_*`,
`adv2_*`, `rel3_*`, `res_*`, `dbg_*`, `mb*`, `powc.log`, `log2flip_n.log`, `flip_log2_orders.log`,
`nest_atan_100_rerun.log`, `pre.log`); `sym_dbg.log`, `rel_cuts*.log`, `rlibs_*.txt`,
`relbuild.*`, `dbgbuild.*`.

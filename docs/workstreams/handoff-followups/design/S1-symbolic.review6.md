# S1-symbolic design: adversarial review 6

Design reviewed: `fu2/design/S1-symbolic.md` (2773 lines, with "Revision 5" to "Revision 1"). HEAD 92906d4
(`git log c79a582..HEAD` ends at the docs commit after J1b; rustmath-symbolic unchanged since 80130e7).
My experiments are in `fu2/design/S1-symbolic/review6/` (below `R6/`): `p6/` is the revision 5 prototype
`rev5/proto6/` copied unchanged, `build.sh` builds it against the HEAD rlibs (`rev5/rlibs_*.txt`),
`gen.py` writes the probe inputs to `inputs/`, `run.py` times one call (address space capped at 6 GiB, VmHWM
from the driver), `probes6.txt` has the outputs of the main probes, `refs6.gp` and `refs6b.gp` the gp
checks, `gpcheck3_r6.out` the oracle run. Timings are release, budget 1e8 (the provisional `TAYLOR_WORK`),
point 0, order 1, unless stated.

Verdict: **not approved**. Three new blockers, all of one class: the claim that every allocation is
charged and that the budget holds at most 2.9 bytes per unit (§2.7, §2.16, §3.1, §5.3) is false for inputs
with many small terms. Each blocker is a 1.4 KB to 4.6 KB input that `taylor_with_budget` admits under
1e8 units with more than 512 MB, up to 2.53 GB. Revision 5 resolves both blockers and both minor issues of
review 5, and the algorithm, the oracles and the callers hold.

## What I verified (holds)

- Review 5 B1, on the rebuilt prototype (`STACK=2097152`, debug): `x + P_4096` is `Ok`, result depth 16,
  `format!` 56 238 bytes; `(1)/(x + P_4096 - 1/2)` is `Ok`, depth 21, 112 521 bytes; N = 431 gives depth 13
  and 18. Release, N = 8800 (104 KB): depth 17 and 22. Revision 4 aborted on all of them.
- Review 5 B2: `x + Q_1400*S_15*0` is `Ok` in 3.9 ms (1.6e6 units); `x + Q_1400*(S_15)*0` with the design's
  `P` generator (left half `len//2`) is refused at 1.26e9 after 24 ms; M = 2900, k = 14 is refused at 1.25e9.
  See m2 for a shape that is `Ok` instead.
- Review 5 m1: `dag_input.rs`, k = 40, all five shapes `BudgetExceeded`: 0.12 to 0.45 s release, 1.0 to
  3.2 s debug. The pre-pass looks a node up before it descends (`ser.rs:819-864`). m2: `POINT_ZERO` is set
  once by the driver and read by `to_expr`.
- Oracles: `rev2/gpcheck3.py` on the rebuilt release prototype gives 107 OK and 0 failures
  (`R6/gpcheck3_r6.out`). gp 2.15 (`R6/refs6.gp`, `refs6b.gp`): tan to 15, the triple pole, `(8+x)^(1/3)`,
  `1/(exp(x)-1)`, cot and `-cot`, `log(cos(x))`, `exp(sin(x))`, H49E3 (`== 1`), acosh at 2,
  `1/(2+sin(x))` at 1, `1/(x+x^2)^200` (valuation -200, [1, -200, 20100], `polcoef(s,2) ==
  binomial(401,199)`), `1/(sin(x)^50 - x^50)` (valuation -52, -3/25, -62/125, 28 nonzero). Every value equals
  §4.1, and the prototype prints the same coefficients.
- Input-node kernels (§2.2, `konst.rs:1390`, `:1565`, `:1760`, `:1853`): `src` is passed only where the
  kernel is the value of that node (`f(c)`, `c^alpha` kept as a kernel, `Function`, `Mod`); radicals,
  inverses and values at a point render normal forms. Sound.
- Callers at HEAD: `repl.rs:2118-2145` (the arm, `symbols()` at `:2127`, `match args[3].simplify()` at
  `:2135`, `taylor` at `:2143`), `TAYLOR_ORDER` at `:11087`, `OUTPUT_BYTES` at `:11199`, `display_size` at
  `:11569`, `EVAL_STACK_BYTES` at `kernel.rs:45`, `Symbol::new` interned at `symbol.rs:40`,
  `limits.rs:428-430`, `:554`, the `series.rs` and `lib.rs` tests. No other workspace caller.
- HANDOFF coverage: F1 (`docs/HANDOFF.md:324-326`) and S7's last part (`:339`) are covered; S7's point part
  was done by I3 and its test is kept.

## Blockers

### B1. The sparse path of the series product is not charged (§2.4, §2.7, §2.16)

`mul` charges the pair weights in one sum (`ser.rs:239-256`), then, when the span exceeds `2 pairs + 64`,
accumulates in a `BTreeMap<i64, SeriesConst>` (`ser.rs:277-291`): a get and an insert per pair, about
40 bytes per map entry, then a result vector of 16 bytes per term. Neither the map nor the result terms
are charged. A pair of 1-word coefficients is charged `wa + wb = 2` in the prototype, and `wa * wb = 1`
by the formula of §2.7. The coefficients themselves are shared (`1 * c` is `scale(1)`, an `Arc` copy).

Reproduction (`gen.py fz D M`): `(A*B*D*E)*0`, with A, B, D, E exact sparse polynomials in negative
powers of x (exponents multiples of 3, so every exponent gets one pair and the map path is taken; negative
exponents keep every term below the working precision, so one attempt suffices). The zero factor makes
the result `0`, after the series factor is expanded (§2.5).

| input | length | outcome | wall | units | peak |
|---|---|---|---|---|---|
| `fz 64 32` (2^23 terms) | 2.4 KB | `Ok` (0) | 2.3 s | 1.74e7 | 441 MB |
| `fz 64 128` (2^25 terms) | 3.7 KB | `Ok` (0) | 10.2 s | 6.77e7 | 1.74 GB |
| `fz 64 186` | 4.6 KB | `Ok` (0) | **15.5 s** | 9.81e7 | **2.53 GB** |

That is 25.8 bytes per unit and 6.3e6 units/s, below the floor of 1e7 that §5.3 assumes. Without the
`*0` the render charge is refused after the same expansion, with the same peak.

Required change:

1. Charge the map path of the series product per pair (a search of an `i64` map: about
   `SEARCH_CMP (bits(pairs) + 1)`) and per new entry and output term for its memory (16 units, as a dense
   slot), before the second pass; or accumulate the sparse pairs in a vector and sort it, charged
   `pairs (bits(pairs) + 1)` plus 16 per entry. State in §2.7 which per-pair weight is used (m4).
2. Add the row to §2.16, and the `fz` family to the §5.3 calibration and the memory table.
3. Budget test (debug, outcome asserted, "returns within 20 s"): `fz 64 186` at 0, order 1, budget 1e8 is
   `BudgetExceeded`; its peak memory is checked by the memory script.

### B2. Small constants cost far more memory than `copy_units` charges (§2.2, §2.7, §2.16 fact 2)

Every constant is an `Arc<ConstData>` with its `ConstSizes` (about 200 bytes with the header), and a
constant of 2 to 11 terms is one `BTreeMap` leaf of 11 slots (about 1 KB). The operations that build a
constant by copying only (`neg`, the merges of `add` and `sum_many`, so the series sum and difference,
and a negated summand) charge `copy_units`, about 40 units per short term. Fact 2 of §2.16 ("the
constructor's own pass ... is covered") counts the pass, not the allocation. Revision 5 stored one-term
constants inline, but the header remains, and 2-term constants still allocate the 1 KB leaf.

Reproduction (`gen.py two 64 4`, `gen.py negz 64 8`; F is the product above with coefficient 1 and 2^20
or 2^21 terms):

| input | length | outcome | wall | units | peak |
|---|---|---|---|---|---|
| `(F*sin(1) + F*cos(1))*0`: one 2-term constant per exponent | 4.0 KB | `Ok` (0) | 1.7 s | 9.76e7 | **1.32 GB** |
| `(-(F*sin(1)))*0`: one negated one-term constant per exponent | 2.0 KB | `Ok` (0) | 1.1 s | 9.70e7 | **561 MB** |
| `-((sin(1) + ... + sin(4096))/(1-x))` at order 560: 561 negated 4096-term copies | 40 KB | refused at the render charge | 0.6 s | about 9.7e7 | 455 MB |

That is 13.5, 5.8 and 4.7 bytes per unit. The first two exceed the bar.

Required change:

1. Charge every constructed constant its allocation before it is built: a base of about 64 units (the
   `Arc`, `ConstData` and `ConstSizes`), and for `Terms::Many` about 256 units per B-tree node
   (`ceil(t / 6)` leaves plus the internal nodes); or store constants of up to 16 terms as a sorted
   `Vec<(Monomial, Rational)>`, whose cost per term `copy_units` already models. Check `COPY_TERM` against
   the measured size of one map entry.
2. Rerun the memory table with the two probes above and a few variants (3 to 11 terms per coefficient).

### B3. The rendering charges the coefficients but not the terms (§2.7, §2.8, §2.16)

The driver charges `expr_units` of every coefficient before rendering (`ser.rs:1118-1119`). The rendering
of each stored term also builds `h^e` (a `Pow` node, an `Integer` exponent with its limb, a clone of the
`Symbol`, which holds a `String`), the `*` or `/` node, and its share of the balanced sum: about 6 `Arc`
nodes and three heap blocks, about 700 bytes. A coefficient 1 is charged 96 units, a one-term kernel
coefficient 136.

Reproduction (`gen.py sparseg 60 4`): `(A*B*D*E)/x^2949130`, A to E as above with positive exponents,
983 040 terms with coefficient 1. The input is 1.4 KB; the result is `Ok` after 0.9 s with 9.89e7 units
and **725 MB** (7.3 bytes per unit). `gen.py sparse1 3` gives 581 MB at 7.9e7 units, and the negated
variant `sparse 2` 560 MB at 9.76e7.

Required change: charge each rendered term before it is built, for example `EXPR_NODE * 6 + 2 w(e) + 16`
plus the point's `x - a` node, in the driver and in `SeriesExpansion::to_expr()`; add this row to §2.16
and remeasure.

Common to B1 to B3: the `TAYLOR_WORK` doc comment, §2.7 and §5.3 must give the bytes per unit measured
with these families. The bar (512 MB at 1e8 units, including the process and the parsed input) needs
at most about 5 bytes per unit on every family; the design states 2.9.

## Minor issues (fix in the revision)

- **m1. The kernel-depth cap is reached by a cheap constant (unbounded `taylor`).** §2.15 says that no
  input reaches it and that 4 D covers the growth. `gen.py invpow 80`, `x + (1)/((1 + ... (1)/((1 +
  sin(1))^511) ...)^511)` with 80 levels (1.1 KB): each level's `Inverse` kernel renders a 512-term P0, so
  13 levels of kernel per 3 levels of input. Unbounded: `LimitExceeded(KernelDepth)` after 19.3 s; HEAD's
  `taylor` returns in 10.5 ms (result depth 248). Under the REPL budget it is refused at 8 levels after
  0.2 s, so this is honest and within the bar. Either give the `Inverse` kernel of an input divisor r the
  expression `c / r` (the input node; P = c P0), so its depth is the input's, or correct §2.15 and state
  the case.
- **m2. `budget_monomial_products` depends on the tree shape of Q.** With the split `(lo+hi)//2` (left half
  larger; review 5's generator), `x + Q_1400*(S_15)*0` is `Ok` in 24 ms with 4.4e6 units, because S meets
  the zero factor early in `product_many`'s rounds. With the design's split `len//2` it is refused at
  1.26e9. Both are honest, but the test must pin the generator. A zero test of the factors before the
  rounds would make the outcome independent of the shape.
- **m3. The B1 family runs at 6.3e6 units/s.** That is below the 1e7 floor of §5.3, so 1e8 units take
  15.5 s here. It is within the 20 s bar on this machine; the charge of B1 fixes it, and the §5.3 rule
  catches it once the family is calibrated.
- **m4. The series product's pair weight.** §2.7 says `wa * wb` per pair; the prototype charges `wa + wb`
  when either operand is at most one word (`ser.rs:250`). For 1-word coefficients the design formula
  gives 1 unit per pair. Write the prototype's rule into §2.7.
- **m5. The `scale` row of §2.16** says "each `mul` charged (at least `copy_units`)". When a coefficient
  is rational 1, `mul` is `scale(1)`, an `Arc` copy charged nothing. The walk over the series is still
  bounded by its creation charge, but the row is wrong.

## Summary for the orchestrator

Revision 5 fixes review 5: balanced monomials and the iterative ball code remove the stack overflows, the
monomial and search charges refuse the long-monomial products, the DAG pre-pass and the recorded point
work, and every gp value, oracle case and caller I checked holds. Three new blockers stop approval, all
memory holes under the REPL budget:

- B1: the sparse path of the series product charges 1 or 2 units per pair for a map entry and an output
  term. A 4.6 KB input is admitted after 15.5 s with 2.53 GB.
- B2: constants cost a header of about 200 bytes and, with 2 to 11 terms, a 1 KB B-tree leaf, while copies
  charge about 40 units per term. A 4.0 KB input is admitted with 1.32 GB, a 2.0 KB one with 561 MB.
- B3: rendering charges the coefficients but not the `h^e`, product and sum nodes of each term. A 1.4 KB
  input is admitted with 725 MB.

The fixes are charges (or a small-vector representation for B2) plus a memory calibration on families
with many small terms. The minors are the kernel-depth cap reached without a budget (m1), a test whose
outcome depends on the tree shape (m2), the rate of the B1 family (m3) and two text corrections (m4, m5).

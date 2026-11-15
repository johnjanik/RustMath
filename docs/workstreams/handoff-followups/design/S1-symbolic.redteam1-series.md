# S1-symbolic, revision 7: red team 1 (series arithmetic and charged cost vs measured time)

Design attacked: `fu2/design/S1-symbolic.md` revision 7, all sections, with focus on
§2.1 (model/trunc), §2.4 (series operations), §2.5 (expand), §2.6 (precision driver),
§2.7 (cost model) and the caps/invariants of §2.15–§2.16. Prototype:
`EXP/rev7/proto8/` (EXP = `fu2/design/S1-symbolic`), copied unchanged to
`EXP/redteam1-series/src/` and built with `EXP/redteam1-series/build.sh release src/main.rs p8`
(release `bin/p8_rel`; `XFLAGS="--cfg memcount"` -> `bin/p8m_rel`), against the HEAD rlibs of
`EXP/rev5/rlibs_rel.txt`. Every run went through a public entry point (`taylor_with_budget`,
`series_expansion` + `to_expr`, `limit_via_series`) with budget 1e8, a timeout and a 4 GiB
address-space cap (`tools/run.py`, `rate.py`). Correctness checked coefficient-by-coefficient against
gp 2.15 (`tools/gpcheck.py`, `default(seriesprecision, n+3)`). HEAD is 2e15d1c.

Lens: products (dense and sparse paths), division and inversion, composition `f^g`, powers with
large / negative / rational exponents, the exp/log/trig/erf recurrences, high orders up to
TAYLOR_ORDER, high pole orders and wide exponent ranges, very sparse and very dense inputs,
coefficient growth; the charged units against the measured time and the peak memory of each
operation.

## Result: no blocker found at the review bar.

Across ~95 distinct inputs spanning the whole lens, every budgeted call finished in at most ~1.3 s
and held at most ~1.3 bytes per unit; no call aborted or overflowed a stack; and every coefficient
checked matched gp. The cost model is conservative on the series axis: a series operation pays a
series-level a-priori pre-charge *and then* the per-pair / per-coefficient charged arithmetic under
it (§2.7, "The double count is deliberate"), so the real work is bounded by a charge that is
typically a large multiple of it. I could not drive the rate below ~6.5e7 units/s for any series
family, nor the memory above ~1.3 bytes/unit; the bar needs < 5e6 units/s (for 20 s at 1e8) or
> 5.1 bytes/unit (for 512 MB at 1e8).

### Worst measured, by family (release, budget 1e8, through taylor_with_budget)

| family | slowest units/s (worst input) | peak bytes/unit |
|---|---|---|
| exp/log/trig/erf recurrences | 6.8e7 (`(1+x)^(1/3)` @0 n2500), 7.6e7 (`sin(sqrt(1+x)-1)` n2000) | 1.23 (`exp(x)` n3500) |
| division / inversion (rational, Fibonacci, poles) | 7.6e7 (`exp(x/(1-x))` n1200) | 0.6 (`1/(1-2x)` n1e5) |
| composition `f^g` (`exp(g log f)`) | 7.7e7 (`sqrt(sqrt(1+x))` n2000) | 0.08 |
| powers: Miller / binomial / rational exponent | 9.0e7 (`(5/7+x)^(-3/11)` n2000) | 0.11 |
| high order (TAYLOR_ORDER 20000), rational functions | 1e8-class, all `Ok`/refused < 0.12 s | 0.78 |
| high poles `1/sin^k`, `1/(exp-1)^k`, `1/(1-x)^3000` | refused < 0.12 s, or `Ok` | 0.03 |
| transcendental-point, kernel-heavy coefficients | 2.5e8 (`1/(2+sin(x))` @1) — unit-heavy, fast | 1.06 |
| symbolic-coefficient blowup (`1/(1+a x+b x^2)`) | 5.3e8 — unit-heavy, fast | 1.06 |
| dense*dense, sparse*sparse (heap merge) | refused < 0.12 s | 1.2 (tiny) |

The engine never came within 5x of either bar on any series input I could build.

### Correctness (all agreed with gp, or correct-but-symbolic)

Verified every coefficient against gp for: `exp(x/(1-x))`, `atan(x)`, `exp(tan(x))`, `log(cos(x))`,
`sqrt(1-x)exp(x)`, `asin(x)`, `erf(x)` (symbolic `2 pi^(-1/2)` matches gp's float),
`(1+x+x^2)^(-7/3)`, `sqrt(cos(x))`, `atan(x)/(1-x)^2`, `atan(x)^3`, `asin(x)/sqrt(1-x^2)`,
`x/(exp(x)-1)` (Bernoulli), `1/(cos(x)-1)`, high poles `1/sin(x)^5,^10`, `1/(exp(x)-1)^3,^5`,
`1/(x^3(1+x))`, `(x+x^2)/sin(x)^3`; rational-point expansions `1/(1+x^2)@2`, `(x^2+1)/(x^3-x+3)@3`,
`1/(1-x)^4@5`; identities `sin^2+cos^2 = 1` (exact), `cos(2x)-(1-2sin^2) = 0` (exact),
`exp(x)exp(-x)=1`, `log(exp(x))=x`, `exp(log(1+x))=1+x`; and the asymptotic (infinity) expansions of
`1/(1-x)`, `(x^2+1)/(x^2-1)`, `x/(x-1)`. All correct.

## Families that held (no hole)

- **Dense Cauchy products** at high order (`sin(x)*cos(x)` n40000, `(1+x)^50000*(1-x)^50000`): the
  dense-buffer path is charged 16/slot by `alloc` and each pair by `q_mul`/`q_add`; refused cleanly,
  memory <= 1 byte/unit.
- **Sparse products (Johnson heap merge)**: `fz`-style and `(1+x^100)(1+x^200)(1+x^300)` take the
  heap path; charged per pair + per row before it runs (§2.4). No unbounded container.
- **Division / inversion** with coefficient growth (Fibonacci `1/(1-x-x^2)` to n25000, `1/(1-2x)`
  with `2^k` coefficients to n1e5, `1/(1-x)^3000`): the `inv` step pre-charge
  `sum bj.words * g[k-j].words` plus the inner `q_mul`/`q_add` bound the real work; <= 0.6 byte/unit.
- **exp/log/trig/sinh/cosh/tan/erf recurrences** (`exp(x/(1-x))`, `exp(tan(x))`, `log(cos(x))`,
  `tanh`, `asin^2`, `erf`): every step is charged `sum of word-products` *and* the per-coefficient
  Konst ops; rate >= 6.8e7 u/s.
- **Composition `f^g = exp(g log f)`** (`(1+x)^sin(x)`, `(2+x)^(exp(x)-1)`, `exp(exp(exp(x))-e)`):
  each sub-op charged; >= 9.5e7 u/s.
- **Powers**: Miller (`(2+sin(1)+x)^5000`, `(1+x+x^2)^500`), negative/inverse (`((1/(1-x)))^(-500)`,
  `1/(1-x)^3000`), rational exponents (`(5/7+x)^(-3/11)`, `(1+x+x^2+x^3)^(-7/3)`,
  `sqrt(sqrt(1+x))`), and the **binomial path** (undecided leading coefficient, n > 64:
  `(x+sin(1)^2+cos(1)^2-1)^2000` to n3000, `^100000`): all `Ok` or refused < 0.2 s, budget enforced.
- **High order = TAYLOR_ORDER (20000)**: twelve rational functions, all `Ok` or refused < 0.12 s,
  <= 0.78 byte/unit.
- **High pole orders / wide exponent ranges**: `1/sin(x)^k`, `1/(exp(x)-1)^k`, `1/(x^k(1+x))`,
  `1/(sin(x)^50-x^50)` (NeedPrecision doubling), all fast and correct; the bounded precision search
  returns `PrecisionExhausted` (honest) rather than a wrong value where the valuation lies beyond the
  window at low order.
- **Symbolic (indeterminate) coefficient blowup** (`1/(1+a x+b x^2)` to n450, `exp(a x+b x^2)`,
  `log(1+a x+b x^2)`): the term cap and `search_units` make these unit-heavy, so they are fast
  (>= 5.3e8 u/s) and <= 1.06 byte/unit.
- **Transcendental-point, kernel-heavy coefficients** (`1/(2+sin(x))@1`, `log(2+sin(x))@1`,
  `tan(x)/(2+sin(x))@1`, `(2+sin(x))^(1/2)@1`): the BTreeMap-heavy `Konst::mul` is charged
  `ACC_ENTRY`+`search_units`+`q_mul` per pair; rate >= 1.4e8 u/s, <= 1.1 byte/unit.
- **`limit_via_series`**: `series_expansion` with `LIMIT_SERIES_WORK = 1e8` and order 5; same budget
  protection, no separate exposure.

## Minor observations (not blockers; not reproduced as holes at the bar)

1. **Prototype plumbing of the binomial `pow_int` sub-context (`ser.rs` ~line 770).** The undecided-
   `c0`, `n > 64` branch builds `cx2 = Ctx { .., work: Work { used: 0, max: 0 } }`. In the prototype
   this is harmless because `Work::charge` ignores `used`/`max` and uses the thread-local `COST`/
   `MAXCOST`, so the global budget still binds (confirmed: `(x+sin(1)^2+cos(1)^2-1)^2000` at order
   1200/3000 and `^100000` at 1500 all refuse at ~1e8 in < 0.2 s). But the real design threads
   `&mut Work`; the implementer must pass the *parent* `Work` into this inner loop (so both the
   budget and the kernel table are shared), not a fresh `max: 0` one. A literal port that keeps a
   `max: 0` sub-Work would not enforce the budget inside the binomial expansion (an O(S^3) loop) and
   would be a budget hole. §2.16's `pow_int` row ("every operation charged") already requires the
   shared Work; worth calling out in §6 as a port trap next to the existing ones.

2. **HEAD-parser artifacts, not engine defects (the prototype links HEAD's parser, which S1
   rewrites).** `atanh(x)`/`asinh(x)` return `Unsupported("atanh(...) of an expression in the
   variable")` because HEAD's parser maps them to a generic `Function`, not `Unary(Arctanh/Arcsinh)`;
   the engine's `expand_unary`/`atan_like` support both, so S1's rewritten parser routes them
   correctly. `x^3/6` parses as `x^(3/6) = x^(1/2)` under HEAD's parser (the exact bug §1.6 fixes),
   which surfaced as a spurious `NotLaurent` on `(x^5)/(sin(x)-x+x^3/6)^2`. Both are honest (`Err`,
   no facade) and match HEAD's own behaviour, so neither is a regression.

## What I could not break

- No units/s below ~6.5e7 for any series family (bar: < 5e6 for 20 s).
- No peak above ~1.3 bytes/unit for any series family (bar: > 5.1 for 512 MB).
- No abort, no stack overflow in the engine on any API-built or parsed input within the lens.
- No wrong, guessed, or fabricated coefficient; no decided answer that should be undecided, nor the
  reverse, on ~40 gp cross-checks and the identity set.

Reproduction scripts and inputs: `EXP/redteam1-series/tools/{run,rate,rate2,rate3,rate4,scan,ord,memb,gpcheck}.py`.

# S1-symbolic, revision 7rt1: red team 2 (series arithmetic and charged cost vs measured time)

Design attacked: `fu2/design/S1-symbolic.md` revision 7rt1 (all sections, with §2.1, §2.4, §2.7,
§2.15, §2.16). Prototype: `EXP/rev7rt1/proto9/` (EXP = `fu2/design/S1-symbolic`), release binaries
`bin/proto9_rel` and `bin/proto9m_rel`, against the HEAD rlibs of `EXP/rev5/rlibs_rel.txt`. HEAD is
2e15d1c. Every run went through a public entry point (`taylor_with_budget`, `series_expansion` +
`to_expr`, `limit_via_series`) with budget 1e8 (`LIMIT_SERIES_WORK` for the limit path), a 60 s
timeout and a 4 GiB address-space cap (`EXP/redteam2-series/tools/run.py`, `exps.py`). Coefficients
checked against gp 2.15 (`default(seriesprecision, .)`). Scripts and inputs in
`EXP/redteam2-series/{tools,inputs}`.

Lens: products (dense and sparse paths), division and inversion, composition, powers with large /
negative / rational exponents, the exp/log/trig/erf recurrences, high orders up to TAYLOR_ORDER,
high pole orders and wide exponent ranges, very sparse and very dense inputs, coefficient growth; the
charged cost against the measured time of each operation.

## Result: three holes over the review bar, each reproduced.

- **S-B3 (wrong, decided result — bar (a)):** the undecided-leading-coefficient branch of `pow_int`
  (the binomial path, `ser.rs:765-790`) marks its truncated result **EXACT**, so a power of a base
  with an undecided constant term, divided by a pole, returns a finite series that silently drops the
  nonzero terms the caller asked for and reports no `O(.)`. `taylor((x+1+sin(y))^100/x^5, x, 0, 3)`
  returns only exponents -5..-2 with `prec = i64::MAX`; gp has nonzero coefficients at every exponent
  -5..3. The decided analogue (Miller path) is correct.
- **S-B2 (crash / regression vs HEAD — bar (b)):** the dense-path span of `mul` (`ser.rs:312-313`,
  panic at `:324`) subtracts two `i64` exponents without a check. A product whose exponent span
  exceeds `i64::MAX` wraps, `alloc` returns a zero-length buffer, and the engine panics
  (`index out of bounds`). A 97-byte input that HEAD expands in 0.5 ms aborts the release prototype.
- **S-B1 (budget hole — bar (c)):** the first loop of the `log` recurrence step (`ser.rs:585-592`)
  scans **every** offset of the argument each step, with no early break, while charging only for the
  offsets below `k`. A sparse argument whose offsets cluster at a high exponent makes the real work
  `n * |offsets|` while the charge grows at ~17 units/step, so the budget 1e8 is reached only after
  tens of seconds. `log(1 + x^1000000*(1+x)*(1+x^2)*...*(1+x^65536)) / x^1131072` at REPL order (203
  bytes) runs 78 s in release through `taylor_with_budget` and 70 s through `limit_via_series`.

---

## S-B3 (over the bar, bar (a)): the binomial power path marks a truncated result EXACT

Design sections: §2.4 (`pow_int`, item 7 "c0 Unknown: the binomial theorem"), §2.1
(`EXACT_KEEP_TERMS`), §2.16 (`pow_int` row). Prototype: `ser.rs:760-790`.

**What happens.** For `b^n` with n > 64, `vb = 0`, and a leading coefficient `c0` whose zero status is
`Unknown`, `pow_int` takes the binomial branch: it accumulates `sum_{m<slots} C(n,m) c0^(n-m) u'^m`
into `acc` with `add`, having set `cx.p = slots`. Every summand is an exact constant series, and `add`
truncates at `cx.p` through `trunc`, which keeps an exact series exact whenever its terms are below
`p` (or it has at most `EXACT_KEEP_TERMS = 64` terms). So `acc` — a truncation of the degree-`n`
polynomial `(c0 + u')^n` to its first `slots` terms — comes out with `prec = EXACT`. The Miller branch
(decided `c0`) instead builds `Ser::from_dense(0, g, slots)`, which carries `prec = slots`
(inexact). The two branches disagree on the precision of the same kind of object; the binomial one is
wrong.

Because the result is `EXACT`, its precision is `i64::MAX`, so after a pole shift the driver never
sees a precision below the target and never runs the loss-driven retry that would raise the working
precision. The dropped high-order terms of the base, which the pole shifts into the requested output
range, are simply absent, and the result claims to be the complete finite value.

**Reproduction.** `taylor((x+1+sin(y))^100/x^5, x, 0, 3)` (30 bytes, budget 1e8, release):

```
prototype stored terms : val -5 prec 9223372036854775807 exponents [-5, -4, -3, -2]
Miller control (exp(1)): val -5 prec 4                   exponents [-5, -4, -3, -2, -1, 0, 1, 2, 3]
gp (c=1+sin(y)=1+7/5)  : exponents -5..3 all nonzero
```

gp: `c=1+s; f=(x+c)^100/x^5; for(e=-5,3, print(e," ",subst(polcoef(f,e,x),s,7/5)!=0))` prints
nonzero at every e in -5..3. The prototype omits -1, 0, 1, 2, 3 and marks the series exact
(`prec = i64::MAX`, no `O(.)`): it asserts those coefficients are zero. They are not.

No-pole asymmetry that pinpoints the defect (order 3):

```
(x+1+sin(y))^100  undecided base -> prec EXACT  (wrong; should be 4 = O(x^4))
(x+1+exp(1))^100  decided base   -> prec 4      (correct)
```

Same four terms, opposite precision. Persists for `slots > 64`: at order 70 the prototype returns
exponents -5..65 marked EXACT, omitting -5..70's upper terms (x^66..x^95 of the true value) and
claiming them zero. Reached through `taylor_with_budget` (the REPL path) and `limit_via_series`.

**Over the bar?** Yes: a wrong, decided result reported as decided (an exact finite series missing
nonzero terms the caller requested). It is not a facade in the design's sense (no fabricated value is
printed), but it is a silent numerical error with no `Err` and no `O(.)`.

**Fix (closes the class).** The binomial branch must carry the working precision of a truncated
series, exactly as the Miller branch does: build the result through `Ser::from_dense(0, g, slots)`
(or set `prec = slots` and `trunc_hard`), so the result is `O(h^slots)` unless `n * deg(u')` is below
`slots` (then it is genuinely exact). An `EXACT` power may be claimed only when every term of
`(c0+u')^n` was computed. §2.4 item 7 and the §2.16 `pow_int` row should state that the binomial path
returns an inexact series of precision `slots`, like Miller.

---

## S-B2 (over the bar, bar (b)): unchecked exponent subtraction in `mul` panics in release

Design sections: §2.4 (`mul`; §2.1 "All exponent arithmetic ... uses checked_add / checked_mul on
i64; an overflow is Err(Unsupported ...)"). Prototype: `ser.rs:312-324`.

**What happens.** After the pair count, `mul` computes
`let hi = ...saturating_add(...).min(cutoff.saturating_sub(1)); let span = hi - val + 1;`. The
`hi - val` is a **plain** subtraction, not checked or saturating. When the product's exponent span
(top exponent minus valuation) exceeds `i64::MAX`, it wraps (release has no overflow checks), `span`
becomes negative, the dense branch is chosen (`span <= 2*pairs + 64` is true), `cx.alloc(span)`
returns a zero-length buffer (`n.max(0) as u64 = 0`), and the write `c[i] = c[i].add(...)` at
`ser.rs:324` indexes an empty vector and panics. The §2.1 invariant that all exponent arithmetic is
checked is violated at this line.

**Reproduction.** `((1)/x^4000000000000000000 + x^4000000000000000000)^2` written as a product (97
bytes), `taylor_with_budget`, budget 1e8, release:

```
prototype : rc=101  panicked at proto9/ser.rs:324:25: index out of bounds: the len is 0 but index is 0
HEAD taylor: head 526us depth 20 VmHWM 3316 kB   (handles it)
```

Also panics through `series_expansion` and `limit_via_series`. The two operands are exact, so
`cutoff = EXACT`; `hi = 8e18`, `val = -8e18`, `span = 1.6e19 + 1 > i64::MAX`.

**Over the bar?** Yes: a crash on a 97-byte input that HEAD handles without error — a regression
against HEAD (the bar names a crash/abort as a regression).

**Fix (closes the class).** Compute the span with checked arithmetic:
`hi.checked_sub(val).and_then(|d| d.checked_add(1))`; on `None`, return
`Err(Unsupported("a series exponent span overflows i64"))` (or route to the heap-merge path, whose
output size is the pair count, not the span). More broadly, audit `mul` for the other plain `+`/`-`
on exponents (`e = ea + eb` in the dense loop at `:319` is also unchecked, reached only after the
span check, but the counting pass at `:294` already uses `cadd`).

---

## S-B1 (over the bar, bar (c)): the `log` recurrence rescans every offset each step

Design sections: §2.4 (recurrence step charge "sum_j words(u_j)*words(g_{k-j}) plus a base of 1"),
§2.16 (the `log` row: "per step a loop over j <= k (in log, over every offset) ... the summing loop
is O(k), at most the charge"). Prototype: `ser.rs:581-605`.

**What happens.** The `log` step over `k in 1..n` has two inner loops over the offset list `iw`. The
**second** (the actual sum) breaks at `*i >= k`, so it is O(number of offsets below k). The **first**
(finding `wk` where `i == k`, and summing the charge over offsets `i < k`) has **no break**: it scans
all of `iw` every step. For a sparse argument whose offsets all lie at a high exponent `>= K1`, the
steps `k = 1 .. K1` each scan `|iw|` offsets while charging only `17` units (base 16 + 1; nothing is
below `k`). The real work is `~ n * |iw|`; the charge accrues at ~17/step, so the budget 1e8 is
reached only after `n * |iw|` has grown into the 10^11 range. `inv`, `exp_u`, `sincos_u` and `miller`
all break their charge loops at `*i > k` (sorted offsets), so only `log` has this hole. The §2.16
claim "the summing loop is O(k), at most the charge" is false for `log` with a high-offset sparse
argument.

**Reproduction.** `log(1 + x^K1 * (1+x)(1+x^2)(1+x^4)...(1+x^(2^(j-1)))) / x^(K1+2^j)` (the pole
makes the REPL order 0 reach the offsets). Budget 1e8, release, `taylor_with_budget`:

| input (bytes) | K1 | j (offsets 2^j) | time | units | VmHWM |
|---|---|---|---|---|---|
| 189 | 300 000 | 16 (65 536) | 8.4 s | 1e8 | 12 MB |
| 189 | 600 000 | 16 | 18.0 s | 1e8 | 14 MB |
| 189 | 800 000 | 16 | 24.3 s | 1e8 | 16 MB |
| 191 | 1 500 000 | 16 | 40.4 s | 1e8 | 21 MB |
| 203 | 1 000 000 | 17 (131 072) | **78.0 s** | 1e8 | 21 MB |

The 78 s input also takes 70 s through `limit_via_series` (order 5, `LIMIT_SERIES_WORK = 1e8`). Time
is linear in K1 (the count of low steps that rescan all offsets) at fixed budget; the effective rate
falls to 1.3e6 units/s, far below the 5e6 units/s the 20 s bar implies. The outcome is an honest
`BudgetExceeded`, and the `log` result is correct when it finishes (checked:
`log(1+x^3*(1+x)*(1+x^2))` matches gp to O(x^10)); the defect is purely the time to refuse.

**Over the bar?** Yes: a <= 100 KB input (203 bytes) runs 78 s > 20 s in release through two public
entry points.

**Fix (closes the class).** Break the first loop when `*i > k` (as the second loop and every other
recurrence does) and obtain `wk` without a full scan (the offsets are sorted; `wk` is the unique
offset equal to `k`, found by one binary search, or carried from the second loop). Then the step is
O(offsets below k), which the `sum_j words` charge bounds. Correct the §2.16 `log` row: the first
loop must not be "over every offset".

---

## Families tried that held (no hole)

All release, budget 1e8, through `taylor_with_budget` unless noted.

- **The other recurrences on the same sparse high-offset argument.**
  `exp(1 + x^1000000*prod_17)/x^..` at REPL order is refused in 0.18 s (exp_u breaks at `j > k`); the
  same shape under `sin`, `cos`, `sinh`, `cosh`, inverse trig and `inv` (division) breaks early too.
  Only `log` rescans.
- **Dense Cauchy products and the heap-merge path** (from red team 1, rerun): `sin(x)*cos(x)` at high
  order, `(1+x)^N*(1-x)^N`, `fz`-style sparse products: refused cleanly, the pair weights charged
  before the pass; no panic unless the exponent span overflows (S-B2).
- **Powers, Miller path** (decided c0): `(x+exp(1))^100/x^5`, `(1+x)^10000`, rational exponents
  `(5/7+x)^(-3/11)`: correct, precision carried, refused or `Ok` within the bar.
- **Powers, binomial path, when the budget refuses it:** `(x+sin(1)^2+cos(1)^2-1)^2000`, `^100000`:
  refused at 1e8 in <= 0.17 s (as red team 1 series m3 and the design note). The S-B3 defect is the
  precision marking of the *admitted* binomial result, not its budget.
- **Division / inversion with coefficient growth, high poles:** `1/(1-x-x^2)`, `1/(1-2x)` to 1e5,
  `1/sin(x)^k`, `1/(exp(x)-1)^k`: the `inv` and the per-step charges bound the work; no hole.
- **Exact products of small operands** (the design's `EXACT_KEEP_PAIRS` path) stay fast, except that
  a wide exponent span triggers S-B2.
- **Correctness cross-checks against gp** on the admitted results of exp, log (dense and the small
  sparse case), sin/cos, atan, erf, Fibonacci division, Bernoulli `x/(exp(x)-1)`: all matched.

## Notes

- S-B3 and S-B2 are both specific to the binomial-power and the wide-exponent-product corners of the
  lens; the `redteam1-series` report did not reach them (it exercised decided-c0 powers and dense
  products, and its binomial inputs were all budget-refused, so the EXACT marking never surfaced).
- S-B1 is a new instance of the "charge must bound the real loop" class that reviews 3 to 6 closed
  for `mul`, `sum_series` and the other recurrences; `log`'s first loop was missed because its charge
  loop and its summing loop are separate, and only the summing one was given the early break.

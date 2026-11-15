# S1-symbolic design: adversarial review 4

Design reviewed: `fu2/design/S1-symbolic.md` (1864 lines, with "Revision 3", "Revision 2", "Revision 1").
HEAD dc367a2 (working tree clean). My experiments: `fu2/design/S1-symbolic/review4/` (below `R4/`):
`build.sh` (links a program against the HEAD rlibs; release rlibs from
`cargo build --release -p rustmath-symbolic -p rustmath-reals`, list in `rlibs_rel.txt`), `runp.py` (one timed
prototype call), `repro4.py` (every new input of this review), outputs `clone_hole.txt` and `key_hole.txt`,
micro-benchmarks `src/trialdiv.rs`, `src/decimal.rs`, `src/headdeep.rs`, gp references `gp/c*.gp` and `gp/o*.txt`.
`R4/p4/` is the revision 3 prototype `rev3/proto4/` with only `main.rs` changed (a `FASTEXIT` mode that
prints the engine result and exits, because `measure` walks the result tree without sharing); `R4/p5/` and
`R4/p6/` are `p4/` with the two key fixes of B2 below. All timings are release, budget 1e8 (the provisional
`TAYLOR_WORK`), unless stated.

Verdict: **not approved**. Two blockers, both budget holes of the class of the earlier reviews (a short REPL
line runs 13 to 33 s, or takes gigabytes, under `TAYLOR_WORK`). They are new: revision 3 fixed every
blocker and minor of review 3, and its charged arithmetic layer holds on every input I tried against it.
The holes are in operations that are not big-number arithmetic: copies of coefficients (B1) and kernel keys
of shared expressions (B2).

## What I verified (holds)

- Review 3 B1 and B2 are fixed. Rerun on the revision 3 prototype (release): `x + (3/5)^200000` `Ok` in
  0.16 s (3.7e7 units); `x + (3^330000)/(5^200000)*2*3` refused at 1.2e8 after 0.36 s (unbounded: 1.1 s,
  1.8e8 units); the `*2*3*...*29` variant refused after 0.36 s; `(2/3+x)^(1/7)` and `sqrt(1+x/3)` at 2000
  `Ok` in 0.54 s and 0.22 s; the 67-character zero-multiplied sum refused after 1.0 s; the 100-radical inputs
  (k = 1..100 and 3..102) refused in 10 ms and 70 ms; one radical `x + (3^340000+1)^(1/2)` `Ok` in 6 ms
  (7.2e7 units). Every number matches §5.3 and the Revision 3 table.
- Review 3 minors: m1 (`1/(2+sin(x))` at 1: order 30 `Ok`, 2.1e6 units; order 50 `Ok` in 0.15 s), m2
  (200 radical folds refused after 0.35 s), m4 (`1/(x + erf(1) - 1/2)` is `Ok` with c0 =
  `-2/(1 - 2 erf(1))`, c1 = `-4/(1 - 2 erf(1))^2`; balls of `2 + erf(1)`, `erf(1) - 1/2` exclude 0, `erf(0)` is
  [0, 0]), m3, m5 and m6 as text. `budget_kernel_keys`: 1.89e7 refused (1.911e7), 1.95e7 `Ok` (1.914e7).
  `(1+x)^(10^100)` at 1400: 9.99e7 units, 104 MB; `(x + sin(2^100000))^1000` at 1000: 6.2e5 units.
- Charged layer: I read `q_mul` to `i_exact_root` (`rev3/proto4/konst.rs:231-331`) and the formulas of §2.7
  against `Rational::new`/`simplify` (`rustmath-rationals/src/rational.rs:19-73`) and Lehmer's gcd
  (`rustmath-integers/src/lehmer.rs:103`: an unbalanced pair costs one division, so `N D` is the right
  shape). The unit-test values of §4.1 are right (`q_add(1/3, 1/5)` = 73, `q_mul` of two 2-word integers =
  73, `pow_bits(2, 340000)` = 340 003). `Integer::exact_root` (`integer.rs:458`,
  `perfect_power.rs:230`) is the sieve, the 2-adic odd root and at most log2 d square roots, so `w^2` per
  root holds for d <= 64.
- gp (PARI/GP 2.15, `R4/gp/`): tan, csc, cot, Bernoulli, H49E3 (`== 1`), log cos, exp sin, triple pole,
  `(8+x)^(1/3)`, `x^x` at 1, `1/(x+sin(1)+I)`, `1/(2+sin(x))` at 1, acosh at 2, `x^2` at 10^30, the
  asymptotic case, `#binary(3^340000)` = 538 888, `1/(x+x^2)^200` ([-200, 203, [1, -200, 20100]] and the last
  coefficient equal to `binomial(401, 199)`), `1/sin(x)^140` ([-140, 71, [1, 0, 70/3]]),
  `1/(exp(x)-1)^150` ([-150, 153, [1, -75, 11225/4]]), erf, `-1/tan(x)`, `exp(I*x)`, `1/(x^2+1)` at I. Every
  value equals the design's tables.
- Callers (grep of every `*.rs` outside `target` at HEAD): `rustmath-jupyter/src/repl.rs:1981`,
  `rustmath-symbolic/src/limits.rs:430`, `:554`, and the `series.rs`/`lib.rs` tests. `rustmath-py` calls
  none. Complete, as §3.1 says. HEAD line references of `series.rs`, `limits.rs`, `simplify.rs`,
  `expression.rs`, `repl.rs`, `parser.rs`, `integrate_exact.rs:96`, `:148` and `plane_curve_ops.rs:492` are
  correct.
- HANDOFF coverage: F1 (algorithm, simplify and Drop recursion, REPL bound) and S7's last part (the literal
  parser) are covered; I3 already did S7's point part, and the design keeps its test.
- Huge literals: the new parser reads any decimal literal (`from_decimal_str`, quadratic: 0.34 s for
  631 291 digits). The REPL already parses unbounded `BigInt` literals at `repl.rs:728`, so this adds no new
  exposure. Decimal Display of the largest admitted integers is fast (0.12 s for 2^21 bits,
  `R4/src/decimal.rs`).
- Deep inputs: HEAD's `taylor` aborts with a stack overflow on `x+x+...+x` at 20 000 terms (8 MB, release)
  and 5 000 (2 MB) (`R4/src/headdeep.rs`); the prototype reaches about 30 000. Not a regression (minor m3).

## Blockers

### B1. Copies of coefficients are charged one unit per term, whatever their size (§2.7, §2.4)

§2.7 charges a series `add` "16 plus the number of terms of both operands, plus the constant adds". Only the
common exponents reach `SeriesConst::add`, which charges `words(a) + words(b)`. Every other term is copied
into the result (`rev3/proto4/ser.rs:147-172`, `a.t[i - 1].1.clone()`) for one unit, although the copy costs its
words: a `BTreeMap` with one node, one `Mono` vector and two big integers per monomial. Series `neg`
(`ser.rs:175`, used by every `Sub` and unary minus) has no charge at all and copies every coefficient. The
variable's series is copied at every occurrence of the variable (`ser.rs:769`, `cx.xser.clone()`), and its
constant term is the point, which may be any constant. `shift`, `split0`, `coef` and `offsets` copy too.
The design's claim "every coefficient word in memory was charged when it was produced, so the budget bounds
the work" holds, but the copies are work that the model does not count.

A constant with many monomials is cheap to build: `(1+pi)^511*(1+e)^511` (20 characters) is an exact
product of two 512-term powers, 262 144 monomials, charged 6.7e7 units in 0.23 s. Then each further `+x`
copies it for about 90 units. Reproduction (`R4/clone_hole.txt`, `R4/repro4.py`, point 0, order 1):

| input | length | wall | units |
|---|---|---|---|
| `((1+pi)^511*(1+e)^511)*0` | 24 | 0.26 s | 6.702e7 |
| `((1+pi)^511*(1+e)^511 +x+x+...+x)*0`, 160 `+x` | 344 | 15.0 s, `Ok` | 6.703e7 |
| the same with 400 `+x` | 824 | **33.1 s**, `Ok` | 6.706e7 |
| the same with 160 `-x` | 344 | 13.2 s, `Ok` | 6.703e7 |
| `x^0+x^0+...+x^0` (100 terms) at the point `(1+pi)^511*(1+e)^511` | 399 | 8.4 s, `Ok` | 6.703e7 |
| control: `(1+pi)^511*(1+e)^511+x+y+...+y`, 160 `+y` (the constant term is added, so charged) | 344 | refused after 1.2 s | 1.03e8 |

The 400 additions are charged 3.6e4 units for 33 s, about 1 100 units/s, 10 000 times below the floor of
1e7 units/s that §5.3 assumes. The output is 0, so no display is involved. The budget admits about 370 000
such additions, so the time is bounded only by the input length. No recalibration can close this: a smaller
constant makes each copy cheaper and lets more of them pass.

Required change:

1. Charge every copy of a coefficient by its words before it is made (`words(c)`, as `SeriesConst::add`
   already does): every term that a series `add`, `sub` or `neg` copies, common or not; `shift`, `split0`,
   `coef`, `offsets`; and the variable's series at each occurrence (charge `words` of the point once per
   occurrence, or expand the variable once per attempt and share it). Alternatively, hold coefficients behind
   `Arc` so a copy is O(1), and charge only what a mutation copies. Add a row for each to the table of §2.7,
   and extend the grep rule of §6 to `.clone()` of a `SeriesConst` or a `Laurent` outside the charged code.
2. Add to the §5.3 calibration an input at the slowest end of each copy path (the four rows above), and
   recompute the slowest rate.
3. Budget tests (debug, outcome asserted, "returns within 20 s"): the 400-`+x` input, the 160-`-x` input and
   the 100-`x^0` input at the point `(1+pi)^511*(1+e)^511`, all at order 1 with budget 1e8:
   `BudgetExceeded` (or `Ok`) within 20 s. Each runs 8 to 33 s in release on the revision 3 prototype, longer
   in debug.

### B2. Kernel keys: the charge stops at the remaining budget in bytes, and keys and walks grow with the tree, not the shared expression (§2.2, §2.7)

§2.2: "Creation is charged before the key is built: `key_size(expr)` (a walk that adds the bytes each node
writes, stopping at the remaining budget), then `key_size / 8 + 1` units." The prototype
(`rev3/proto4/konst.rs:990-998`) passes `remaining()` units as the byte limit. When the key has more bytes
than the remaining units, the walk stops just above `remaining` bytes, the charge is about `remaining / 8`,
which passes, and `write_key` then writes the whole key. Before that, `mk_kernel` computes `has_indet` with
`Expr::symbols()` (`konst.rs:999-1004`, `substitute.rs:337`), an uncharged walk of the whole tree.

The key size is unbounded because a kernel's expression shares subtrees: `to_expr` of a constant reuses each
kernel's `Expr` through `Arc`, so `exp(1 + 2 E + E^2)` holds E twice. Nesting doubles the tree at every level
while the shared expression grows linearly. Every tree walk (`key_size`, `write_key`, `symbols`, the
`HashMap` hash of the key, `ball_cost` and `enclose_real`) is linear in the tree, so exponential in the input.
The prototype also stores each key twice (the table key and `KData.key`).

Reproduction (`R4/key_hole.txt`, `R4/repro4.py`; N_k is `(1+exp(...(1+exp(sin(1)))^2...))^2` nested k times;
point 0, order 1; address space capped at 6 GiB):

| input | length | wall | peak memory | units |
|---|---|---|---|---|
| `exp((1+exp(N_14))^511 + x)` | 182 | 3.1 s, `Ok` | 644 MB | 2.84e7 |
| `exp((1+exp(N_16))^511 + x)` | 204 | **11.4 s**, `Ok` | **2.56 GB** | 2.90e7 |
| `exp((1+exp(N_6))^511*(1+exp(N_6'))^511 + x)` (N_6' with `sin(2)`) | 180 | **12.9 s**, `Ok` | **3.09 GB** | 7.11e7 |
| the same with N_8 and N_10 | 224, 268 | aborted after 27 s and 33 s: allocation failure at 6 GiB | | |

Time and memory double per level while the units stay at about 2.9e7. Two fixes, each tested on a copy of
the prototype:

- `R4/p5/`: the key walk stops at `8 * remaining` bytes, so an oversized key is refused before it is written.
  The N_14 case is then charged 7.9e7 units (3.1 s, 644 MB) and N_16 is refused after 1.0 s, but the product
  case with N_10 still runs 9.5 s before its refusal: the uncharged `symbols()` walk of the 262 144-term
  argument.
- `R4/p6/`: `p5` without the `symbols()` walk. Every case above is refused in 0.5 to 0.9 s.

Even with both fixes a key costs one unit per 8 bytes and is stored twice, so the keys held under
`TAYLOR_WORK` may reach about 1.6 GB (644 MB at 7.9e7 units in the N_14 case).

Required change:

1. Bound the key walk by the budget in the unit it charges: stop once `bytes / 8 + 1` would exceed the
   remaining budget, and refuse before writing. Store one copy of each key.
2. Compute `has_indet` from the normal form of the argument (an OR over its kernels' flags and its
   indeterminates), never by walking the kernel's expression.
3. Make kernel creation linear in the shared expression. The preferred fix is to key a kernel whose argument
   is a constant by (kind, class, operator, the argument's normal form), with each sub-kernel written as its
   interning id. Kernels are interned once per expansion, so this is injective, and it is more canonical than
   the expression key. The other fix is to charge one unit per key byte, so that keys hold at most about
   `TAYLOR_WORK` bytes. Either way, say in §2.2 that kernel expressions are shared DAGs, that tree walks
   over them are exponential, and which walks remain: the numeric test is charged by `ball_cost` and refuses
   at the budget, which is honest. In §3.4, add that J1b's size estimate must be memoized per shared node:
   counting each occurrence by a plain walk is exponential on these results.
4. Budget tests (debug, outcome asserted, within 20 s, and peak memory checked where the test harness
   allows): `exp((1+exp(N_16))^511 + x)` and the N_10 product case, at 0, order 1, budget 1e8: refused (or
   `Ok`) within 20 s. Both fail on the revision 3 prototype (11.4 s and 2.6 GB; abort).

## Minor issues (fix in the revision)

- **m1. Two paths skip the exact-power caps, so the unbounded `taylor` hangs.** §2.7 says the caps "apply with
  or without a budget; they keep `taylor` (unbounded) from expanding huge constants". (a) `inverse` turns a
  monomial with `(1/P0)^e` into `P0^e` by `pow_u` without the multinomial caps
  (`konst.rs:703`). `x + ((2+sin(1))^(-1099511627776))^(-1)` (38 characters) does not return within
  20 s unbounded. `((2+sin(1))^(-5000))^(-1)` takes 15.8 s and 1.65e10 units, while
  `x + (2+sin(1))^5000` gives a kernel in 0.1 ms. (b) The rational-power path computes `(m r^(1/d)/den)^p`
  for p <= 4096 with no bit cap (`konst.rs:1138`). The design states the caps only for r = 1.
  `x + (2*3^(-300000))^(4095/2)` (28 characters) does not return within 20 s unbounded, while `^5000` gives a
  kernel in 4 ms. Both are refused at once under the budget, so the REPL is safe. Apply the caps of §2.2 on
  both paths, with the kernel fallback (`P0^e`, `c^(p/d)`), and test both inputs unbounded.
- **m2. The trial-division charge has no per-prime base.** Using the table loop of §2.2
  (`R4/src/trialdiv.rs`: 6542 primes, `p^d`, a clone and a remainder each), a 63-bit radicand with d = 3 takes
  0.60 ms for 6 542 units (1.1e7 units/s), at the floor of §5.3. The d = 2 case takes 0.33 ms (2.0e7). Add a
  base per candidate prime, for example `(w(n) + 8)` units each. Note also that the prototype's
  `dth_power_part` loops over every p < 2^16 with `is_small_prime`, about 10^7 operations per call. The port
  must use the `prime_range` table, as §2.2 says.
- **m3. Recursion depth of `expand` and quadratic `contains_symbol`.** On an 8 MB stack in release, the
  prototype aborts with a stack overflow on a left-deep sum of about 33 000 terms. HEAD aborts at 20 000 terms,
  and at 5 000 on 2 MB. `expand` also calls `contains_symbol` at every node, which is quadratic on left-deep
  chains. Neither is a regression, and the item names simplify and Drop. §7 item 5, however, lists only
  Display, PartialEq and substitute. List `expand` there and in the new HANDOFF row, or make it iterate along
  the left spine of `Add`/`Sub`/`Mul` as §2.10 does for `simplify`, with variable dependence computed once
  per node.
- **m4. §3.3 says "The line numbers are at HEAD".** That holds for the library sites, but the test sites are
  at c79a582, as the header says. In the files I3 edited they moved: `orthogonal_polys.rs:629` at HEAD is
  `use rustmath_integers::Integer;`, while c79a582 has the `matches!` there. Fix the sentence. The
  recompile already finds the sites.
- **m5. Port traps worth a line in §6.** The binomial path builds a second context with a dummy
  `Work { used: 0, max: 0 }` (`ser.rs:640`). That is harmless with the prototype's thread-local counter, but a
  literal port with an explicit `&mut Work` would fault at once or charge nothing, so the same `Work` must be
  used. The series `mul` pre-charge loop recomputes `kb.words()` (O(terms)) for every pair before it charges
  (`ser.rs:203-211`). Compute each term's words once.
- **m6. Say what "bounds the memory" means in numbers.** In the `TAYLOR_WORK` doc comment, record the
  measured bytes per unit: about 2.8 bytes per unit for a 262 144-term constant (185 MB at 6.7e7), and 8 to
  16 per unit for kernel keys until B2 is fixed.

## Summary for the orchestrator

Revision 3 fixes review 3: the charged big-number layer is correct, and every review 3 input and every gp
value I reran matches the design. The algorithm, the zero test, the ball code, the API, the callers, the
parser and Display, Drop, and HANDOFF coverage all hold. Two new budget holes block approval.

- B1: copies of coefficients are charged one unit per term. A sum `+x` copies a 262 144-term constant
  for 90 units, so an 824-character line runs 33 s, and 100 `x^0` at a large point run 8.4 s.
- B2: the kernel-key charge stops at the remaining units counted as bytes, and keys and the `symbols()`
  walk grow with the expanded tree of shared kernel expressions. A 204-character line runs 11 s and uses
  2.6 GB, and a 224-character line exhausts 6 GiB.

Both are local fixes in the cost model and the key construction (§2.2, §2.7), each with budget tests. The
minors are: cap bypasses that hang the unbounded `taylor` (m1), a per-prime base for trial division (m2),
the expand recursion and quadratic `contains_symbol` missing from §7 (m3), a line-number sentence (m4), port
traps (m5), and the memory figure (m6).

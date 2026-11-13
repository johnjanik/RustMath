# Review 2 of J1a-repl-parser-and-small (design with Revision 1)

Reviewer scratch: `$W = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1a-repl-parser-and-small/review2`.

| Path | Content |
|---|---|
| `$W/head_repl.rs` | HEAD's `repl.rs` (md5 equal to the tree at c79a582) |
| `$W/proto/src/repl.rs` | the revision prototype (md5 equal to `$S/r1/proto/src/repl.rs`), built with `$W/proto/build.sh` (test binary) and `./build.sh lib` (driver `proto/drive`) |
| `$W/drive_head`, `$W/drive_rel` | `../drive.rs` on HEAD's debug and release rlibs |
| `$W/r2_*.txt`, `$W/r2_*_head.out`, `$W/r2_*_proto.out` | this review's inputs and outputs (HEAD, prototype) |
| `$W/r2_primes.txt`, `$W/r2_primes_head_rel.out`, `$W/est.py`, `$W/r2_est.out` | HEAD release `primes` above 2^64 and the design's time estimate for the same calls |
| `$W/c1.gp`, `$W/s.gp` | gp checks of the `primes`, `count_bound` and `sqrt_mod` values (`gp -q c1.gp < /dev/null`) |
| `$W/f64edge.rs` | num-rational 0.4.2 `Ratio<BigInt>::to_f64` and num-bigint `to_f64` at the overflow and subnormal edges |
| `$W/gb_small.rs` | the checked graph model (debug build) |

The large binaries (`drive_head`, `proto/drive`, `proto/proto_test`) are deleted after this review; `proto/build.sh`
and the rustc line in the design's scratch table rebuild them.

Verdict: **not approved**. Two blockers. Both are local and have small fixes. Review 1's three blockers are
fixed (checked below), and most of the design reproduces exactly.

## What I reproduced (and agree with)

- **Prototype.** `proto_test`: 68 of 68 pass (63 existing, 5 revision tests), 7.4 s debug.
- **S9.** Every row of 2.1 and 2.4 that I ran (`$W/t1.txt`, `$W/r2_a.txt`, `$W/r2_c.txt`) gives the designed
  value on the prototype and the stated error or panic on HEAD: `2 * -3` -6, `-2**2` -4, `2^-3^2` 1/512,
  `2 * -3 ** 2` -18, `-3 ** -2` -1/9, `(2+3)*4` 20, `x1e-3` 4, `"[" + "]"` `"[]"` (HEAD returns the text
  `"[" + "]"` as one literal), `π*2` Ok, `h / z0` ZeroDivisionError (HEAD panics), `1e400` ValueError.
  Review 1's regression rows (`review1/t3.txt`: `1/2 * 1/3`, `r * 3 / 2`, `e1 * 2/3`, ...) are identical on HEAD
  and the prototype. The Clifford rows of `$S/r1/t_cl.txt` give equal elements (only the `HashMap` order of
  the Debug text differs). I enumerated HEAD's `apply_binary_op` arms (`:9666-9968`): the only type with a
  `* Rational` arm is CliffordElem, and it also has `/ Integer` (`:9829`), so the left-associative product
  keeps every value HEAD computed. `group_inner`: I re-derived the rule j = min(k, r, D) (the t-th leading
  `(` stays open across the stretch iff the depth there stays above t; the trailing run then closes the pairs
  in order); it is correct, including `(f(")"))` and `((x)+f(y))`.
- **S10.** HEAD `pochhammer(3, 4)` = 0. The ch. 49 digest line 292 defines the rising Pochhammer symbol; the
  ch. 144 digest line 37 has only `NumberOfPermutations` (falling). The gp products are right.
- **S11.** The library fix is correct: num-rational 0.4.2 agrees with Python `float(Fraction(n, d))` at
  2^1024 - 2^970 - 1 (f64::MAX), at 2^1024 - 2^970 (the tie, inf), at 1/2^1075 (tie to 0), 3/2^1076,
  1/2^1074, (2^1075+1)/2^2149 (all 5e-324), and with a negative denominator (`$W/f64edge.rs`). On the
  prototype `mean([r2])` = 1, `variance([r2, 1])` = 0, `mean([2^1100])` ValueError. Also fixed and not listed:
  `median([r2, 1, 2])` **panics** on HEAD (`rustmath-stats/src/statistics.rs:60:44`, `partial_cmp` of NaN
  unwrapped; `$W/r2_median.txt`); the prototype gives 1 (minor m6).
- **S14.** All rows of 5.4 that the prototype implements reproduce (`$W/t3_proto.out`); `chart(M, ...)` with
  100000 names and the following `coordinate_form` and `exterior_derivative` limit message run in 0.05 s,
  44 MB peak (debug driver).
- **N5.** All rows of 1.4 on the prototype (`$W/t6.txt`), plus `plot_points=10^30`, `=-3`, `="5"` refused.
- **S15 graphs.** The checked model in a debug build: `complete_graph(2^62)` and `(2^64-1)` give None, no
  panic; 3585 admitted with 132444240, 3586 refused with 264675488; `path_graph(1187500)` 133000000 exactly.
  `wheel_graph(n)` has n + 1 vertices (`generators/mod.rs:52`), so its degree groups are right.
- **gp values.** `$W/c1.gp` and `$W/s.gp` reproduce every value of 6.4 and 7.4: the `count_bound` row
  `[6, 21, 43, 526, 749, 2171, 4038, 144764, 1000000, 1000001, 1006608, 959374]`, `primepi(6*10^9)` =
  279545368, `primes([2^64, 2^64+99])`, `primes([10^30, 10^30+299])`, the roots mod 2^127+1, mod p20·q21, mod
  psi_12 (4 and 73), the eight roots of 9 mod 1009·1013·(2^89-1), and p40·q40.
- **Citations.** ch. 18 `:61`, `:125`, `:334`, `:417`; ch. 19 `:188-189`; ch. 20 `:44-46`, `:88`, `:157`;
  ch. 49 `:292`; ch. 144 `:37`: each says what the design quotes.
- **I1 and I2 interfaces** (read in their designs): `mpqs_limit: Some(0)` disables the sieve and `Some(60)`
  also takes composites below 25 digits; `proof: false` puts BPSW passes >= psi_12 in `probable_primes`;
  `try_sqrt_mod_prime` returns Ok(None) only for Jacobi -1, a proof for any odd modulus; `try_is_prime`
  caches proven primes, so the finitefields contract checks cost nothing. So the arithmetic of
  `test_sqrt_mod_factor_phases` (T = 2000, 1010, 500) holds, and `crt_two` exists (`crt.rs:93`).
- **Review 1.** B1 (rational arms), B3 (checked model) and m1-m10 are done as stated.

## Blockers

### B1. S9 item 10: `float_arith` gives wrong range bounds where HEAD's were right

`eval_to_f64` now computes in doubles first and uses the exact path only when the double is inf or NaN. Two
things escape that check.

(a) **Cancellation in texts that HEAD evaluated exactly.** Section 2.2 item 10 says: "For a text with
`+ - * /`, literals and constants that HEAD read correctly this is HEAD's value bit for bit". That holds only
for texts that HEAD's `eval_simple_arithmetic` could parse. A text with `^`, `**` or a variable name made it
fail, and HEAD then evaluated the **whole text exactly** (`eval_expr`, `:5174-5185`) and rounded once. The
prototype evaluates those texts in doubles (`$W/r2_g.txt`):

```
w = var("w")
find_root(2*w == 1, 10^20 + 1 - 10^20, 3)    HEAD: No root found (bound 1; the root 0.5 is outside [1, 3])
                                             proto: 0.49999999997089617  (bound read as 0.0: 1e20 + 1 - 1e20)
a = 10^20;  find_root(2*w == 1, a + 1 - a, 3)          same: HEAD right, proto 0.4999...
find_root(2*w == 1, 2^53 + 1 - 2^53, 3)                same
```

So `find_root` returns a "root" outside the interval the user gave. This changes a value HEAD computed
correctly. The policy of this unit forbids that, and S9 was meant to remove this kind of silent error.

(b) **An inf or an underflow absorbed into a finite result.** x / inf = 0 and an underflow to 0 are finite, so
the "not finite" check does not catch them (`$W/r2_d.txt`):

```
find_root(w^2 == 4, 1, 3 * 10^300 / 10^400 * 10^100)          proto: No root found (bound 0.0); HEAD: TypeError
find_root(w^2 == 4, 1, 10^-200 * 10^-200 * 10^300 * 10^100 * 3)  proto: No root found (bound 0.0); HEAD: SyntaxError
3 * 10^300 / 10^400 * 10^100                                     proto: 3 (the same text, exact)
```

Python agrees that the doubles give 0.0 (`3*1e300/float('inf')*1e100`), and the exact value is 3.

Required change:
1. `eval_to_f64(s)`: evaluate exactly first. If `eval_expr(s)` gives an Integer or a Rational, return
   `exact_to_f64` of it (one rounding: HEAD's value where HEAD was right, and correct where HEAD failed). If
   it gives a Float, return that. Only when the value is not an exact number (the text uses pi, e, tau or
   float arithmetic, so `eval_expr` gives a Symbol, an Expr or a TypeError) use `float_arith`.
2. `float_arith` returns None (then the error of step 1 stands) as soon as an atom or an intermediate value
   is not finite, or an operation on nonzero finite operands gives 0 or a subnormal (an underflow). Then
   `pi * 10^-400 * 10^400` is an error, not 0.
3. Correct the "bit for bit" sentence in 2.2 item 10 and 2.3. State in the doc of `float_arith` that a text
   with constants is computed in IEEE doubles, so cancellation (`10^20 + pi - 10^20`) is possible there.
   Those texts were errors at HEAD. Alternatively evaluate them with a guard or at a higher precision.
4. Tests (fail on the prototype): `find_root(2*w == 1, 10^20 + 1 - 10^20, 3)` and the `a + 1 - a` form are
   "No root found" (pass on HEAD: regression rows); `find_root(w^2 == 4, 1, 3 * 10^300 / 10^400 * 10^100)` = 2
   and the `10^-200 * 10^-200 * ...` form = 2 (exact value 3; HEAD errors); keep every row of
   `test_range_bounds_float_arith`.

### B2. S15 `primes`: the time estimate refuses calls that HEAD answered correctly in under 0.1 s

Section 6.3 step 3 refuses the part above 2^64 when
`n_odd · t_test(B) + count_bound(y_hi) · t_proof(B) > 10 s`. `count_bound` is the Montgomery-Vaughan bound,
which holds for every x and so is far above the real count at large heights (2y/ln y against y/ln x). The
estimate then charges an ECPP proof for each of those primes, most of which do not exist. Measured
(`$W/r2_primes_head_rel.out`; estimate `$W/est.py`, the design's formulas; truth from gp, `$W/c1.gp`):

| Call | HEAD release | Correct answer | Design estimate |
|---|---|---|---|
| `primes(10^400, 10^400 + 10)` | `[]`, under 0.1 s | `[]` (gp `nextprime(10^400) - 10^400` = 69) | 47.5 s, refused |
| `primes(2^1000, 2^1000 + 200)` | `[]`, 0.07 s | `[]` (next prime at +297) | 273 s, refused |
| `primes(2^512, 2^512 + 70)` | `[]`, under 0.1 s | `[]` (next prime at +75) | 18.7 s, refused |
| `primes(2^512, 2^512 + 100)` | one prime, under 0.1 s | one prime | 25.1 s, refused |
| `primes(10^30, 10^30 + 20000)` | 258 primes, under 0.7 s | 258 primes (gp) | 26.8 s, refused |
| `primes(2^512, 2^512 + 2000)` | 7 primes, under 0.1 s | 7 primes (gp) | 307 s, refused |

An empty answer is decided at HEAD: every candidate failed a Miller-Rabin test, which is a witness. So the
first three rows are new refusals of calls the old code computed correctly in well under 10 s. That is what
the S15 policy (and the `repl_limits` doc: "a refused call would take more than about 10 s") excludes. The
design's argument ("HEAD's answers there were not proven") does not cover them. The test of 6.4 even pins one
of them (`primes(10^400, 10^400 + 10)` refused). For the other rows the real cost after I1 is small: 258 proofs
of 100 bits at about 7 ms, or 7 proofs of 512 bits at about 0.41 s (I1 means), while the estimate says
27 s and 307 s. The largest admitted span is 6635 at 10^30, 32 at 2^512 and 1 at 10^400.

Required change: two passes. First run `is_probable_prime` (BPSW, I1) on the odd candidates, refusing
beforehand only on `n_odd · t_test(B) > 10 s`. Keep the probable primes; there are at most `count_bound(y)`, so
memory stays bounded by the existing `PRIMES_COUNT` check. Then refuse if `k · t_proof(B) > 10 s`, with k the
actual number of probable primes, and name k in the message. Otherwise prove each with `try_is_prime` as
designed (Err gives the ValueError). The answer is unchanged: a BPSW failure is a proof of compositeness, and
every listed prime is proven. Tests: `primes(10^400, 10^400 + 10)` = `[]` and `primes(2^1000, 2^1000 + 200)` = `[]`
(gp values above; F on the design, P on HEAD: regression rows). `primes(2^1000, 2^1000 + 10^5)` stays refused,
since the BPSW pass alone is about 47 s by the model. The release acceptance check of 8.2 should add
`primes(10^30, 10^30 + 20000)` (258 primes, admitted, within 10 s) and `primes(2^512, 2^512 + 2000)` (7 primes).
Correct the last paragraph of 6.3.

## Minor

- m1. The prototype has `check_chart` only in `coordinate_form` (`proto/src/repl.rs:4630`). On it
  `exterior_derivative(a, C4)` and `wedge(a, a, C4)` (a on R^3, C4 a 4-name chart) still give
  `DiffForm(degree=2)` (`$W/t3_proto.out`). The design (5.2) specifies the check in both arms, so this is only
  a note: these two rows of 5.4 are unverified, and the implementer must add both checks.
- m2. Found on the way, not in this unit: rustmath-symbolic `distribute_negation` (`expand.rs:250-263`) is
  wrong in both arms. -(a + b) becomes (-a) - (-b), and -(a - b) becomes (-b) - (-a). So `expand(-(x + 1))` =
  `(-x - -1)` = 1 - x, and `expand(-(x - 1)*(x + 1))` = x^2 - 1 (`$W/r2_b.txt`; HEAD and prototype alike).
  `PROJECT_TRACKER.md:2159` rates `expand` 🟢. S9's unary minus makes `g = -(x + 1); expand(g)` reachable from
  `eval_expr`, and the symbolic parser already reached it at HEAD. Add it to section 9 and to HANDOFF (S1's
  area), and downgrade the tracker row.
- m3. Section 9 says HEAD's abort at a 732-term sum is gone. That holds for numeric chains. A symbolic chain
  builds a left-nested `Expr`, whose display and drop recurse per term: `w + w + ... + w` with 3000 terms still
  aborts the debug driver (8 MB main thread), and 2000 terms pass. Say so in section 9.
- m4. `2 ** -10001` says "exponent must be an integer in 0..=10000 (REPL limit), got 10001". It checks
  |b| but prints it as the value given. Name the exponent as written (-10001), or say that |exponent| is
  limited.
- m5. `GRAPH_BYTES` is sized for an assignment (three copies). A bare `Graph(3000000)` needs about 288 MB at
  HEAD (two copies) and is refused. That is still within "a few hundred MB", but the doc should say that
  refused bare expressions start at about 266 MB.
- m6. Add `median([r2, 1, 2])` = `1` to `test_large_rational_to_float` (F: HEAD panics at
  `statistics.rs:60`, a kernel kill), and mention it in 4.1.
- m7. `value_type_name` formats the whole value with `Debug` to read the variant name. `-L` for a list of
  10^6 entries builds a string of tens of MB for a one-word message. A `match` on the variant costs nothing.
  HEAD's TypeError messages already print whole values, so this is optional.

## Coverage, callers, plan

- Every item is covered and none is declared out of scope; none needs to be.
- Callers (my own grep over the workspace): `Plot3DOptions`, `parse_from_args`, `eval_simple_arithmetic`,
  `sqrt_mod_report` and the graph, `prime_pi` and `primes` arms occur only in `rustmath-jupyter/src/repl.rs`.
  No crate depends on rustmath-jupyter (it is only a workspace member). `Rational::to_f64` has callers in
  rustmath-manifolds (`integration.rs:528`, `transition.rs:311`, `tensor_field.rs:329`, `vector_field.rs:148`,
  `:242`), rustmath-modular and others. The plan's full `cargo test --workspace` against the 14,173 baseline
  covers them. `prime::factor` stays trial division in I2, so the sqrt_mod rewrite is still needed after I2.
- Plan order is implementable. S11 comes before S9 (the new `eval_to_f64` uses `exact_to_f64`), and S9 comes
  before S14 and N5 (they use `top_level_chars`). With B1's change, `eval_to_f64` calls `eval_expr` first.
  `eval_expr` is already its fallback at HEAD, so no new side effect appears. B2's change replaces one
  estimate by two and needs I1's `is_probable_prime`, which is public (I1 §2.1).
- Cost claims checked: the graph model (re-run), the chart with 100000 names in debug (0.05 s), the
  `count_bound` values, HEAD's `primes` timings (above). The `t_test` model is low for large B (release
  modpow 39 ms at 4096 bits, 2.37 s at 16384 bits, `$W/modpow`), but 8.2 re-times it.

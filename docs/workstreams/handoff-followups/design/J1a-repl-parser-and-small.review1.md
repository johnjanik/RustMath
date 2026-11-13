# Review 1 of J1a-repl-parser-and-small

Reviewer scratch: `$V = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1a-repl-parser-and-small/review1`.

| Path | Content |
|---|---|
| `$V/drive_dbg` | `drive.rs` linked against the HEAD debug rlib `librustmath_jupyter-5152ef5a4115ff6f.rlib` (the rlib `cargo build -p rustmath-jupyter --lib` names at c79a582) |
| `$V/drive_rel` | the same driver against the release rlib `target/release/deps/librustmath_jupyter-abf61e484246aa63.rlib` (HEAD sources) |
| `$V/proto/` | a copy of the designer's prototype `repl.rs` (all five patches), rebuilt here: `proto_test` (63 of 63 pass, 6.4 s) and `proto/drive` |
| `$V/t1.txt` .. `t9.txt`, `head2.out`, `proto2.out`, `head6.out`, `proto6.out` | REPL inputs and outputs, HEAD against prototype |
| `$V/gb.rs` | the design's `hash_set_bytes` / `graph_bytes` code, debug build |
| `$V/chk.gp` | gp cross-checks |
| `$V/sieve_rel` | the I1 sieve prototype (`../sieve_i1.rs`) built with `-O` |

Verdict: **not approved**. Three blockers, all small to fix. The rest of the design holds up well; most
claims reproduce exactly.

## What I reproduced (and agree with)

- **N5.** All eight rows of 1.1 on HEAD. On the prototype every row gives the designed result, plus
  `plot_points=2` (4 arrows), `=1`, `=-3`, `=10^30`, a repeated option and an empty name (all refused
  with clear messages). Every grid divides by `n - 1` (`:5496`, `:5741`, `:5973`, `:6251`, `:6481`,
  `:6805`), so 2 is the right lower bound. `Plot3DOptions` and `parse_from_args` have no caller outside
  `repl.rs`, and no crate depends on rustmath-jupyter.
- **S9.** Every row of 2.1 on HEAD, including the panic `π*2` at `repl.rs:9597`. On the prototype every
  expected value of 2.4 holds (`2 * -3` = -6, `-2^2` = -4, `2 ** -1` = 1/2, `-2**-2` = -1/4,
  `(-2)**-3` = -1/8, `0 ** -1` ZeroDivisionError, `2**-2**2` = 1/16, `(2+3)*4` = 20, strings with commas
  and operators, `"é"*2`, `Polynomial([gcd(4, 6), 1])` = x + 2). gp agrees on the numeric rows
  (`$V/chk.gp`; note gp prints `32` for `print(3 - -2)`, so Python is the right reference there, as the
  design says).
- **S10.** HEAD `pochhammer(3, 4)` = 0, `pochhammer(5, 3)` = 60. The ch. 49 digest line 292 defines the
  rising Pochhammer symbol; ch. 144 line 37 has only the falling factorial as `NumberOfPermutations`.
  The values 360, 210, 240240 are right.
- **S11.** Every row of 4.1 on HEAD (`mean([r2])` NaN, `fft` `[NaN + 0i]`, `mean([1/10^310])` 0,
  `mean([2^1100])` inf, `find_root(...)` 0, and also `integrate_num(x*1, x, 0, r2)` NaN). num-rational
  0.4.2 `ratio_to_f64` is correctly rounded with a sticky remainder and returns ±inf / ±0 on overflow /
  underflow (read in `~/.cargo/registry/.../num-rational-0.4.2/src/lib.rs:1557-1647`). All seven
  `to_f64` sites in `repl.rs` are covered (`:985` already refuses non-finite values).
- **S14.** `chart(E3, "t")` gives `Chart(t)` on HEAD, `chart(M0, "t")` too. `Chart::new` checks the count
  against the dimension it is given (`chart.rs:53-58`), so passing dim(M) is the right fix. The test
  rewrites match MF1 §4.3 / §4.4 item 5.
- **S15.** `star_graph(18446744073709551615)` panics on HEAD (`basic.rs:663`). The memory model
  reproduces: `G = complete_graph(3000)` VmHWM 331136 kB (model 332.5 MB), `G = star_graph(10^6)`
  388992 kB (model 392.6 MB). HEAD release: `primes(0, 4·10^5)` 9.0 s, `prime_pi(4·10^5)` 9.2 s, so the
  new limits refuse nothing HEAD did in 10 s. I1's sieve prototype with `-O`: `prime_pi(10^9)` 1.39 s,
  `prime_pi(6·10^9)` = 279545368 in 10.43 s (matches gp `primepi`). The count-bound values (144764,
  959374, 4/25/1229) match gp.
- **sqrt_mod.** All values of 7.4 match gp (`$V/chk.gp`: the eight roots of 9 mod
  1009·1013·(2^89-1), 1013·(2^89-1) = 627016629898045109236406418443, the p40·q40 product). The phase
  arithmetic of `test_sqrt_mod_factor_phases` is right (T = 128000 / 64 = 2000; T = 64640 / 64 = 1010).
  I checked the "no new refusal" claim for a prime n, which the design does not measure: HEAD release
  `sqrt_mod(4, 2^2203-1)` 2.7 s, `2^4423-1` 20.8 s, `2^9689-1` 210 s. So HEAD's 10 s reach for a prime
  is about 3400 bits, below `SQRT_MOD_I2_MAX_BITS` = 4096: the claim holds. The CRT, Jacobi and
  per-piece arguments are correct, and the API separates proven answers, caveated answers and refusals.

## Blockers

### B1. S9: the new left-associative `*` / `/` split breaks rational products that HEAD computed

The design says (2.3) "Values are equal; only symbolic display and the evaluation order change". That
is false, because the REPL has no Integer/Rational cross arms in `apply_binary_op` (the design's own
section 9 notes this). HEAD splits `a * b / c` as `a * (b / c)`, so `b / c` with two integer literals
became a Rational literal (`eval_expr` `:737-748`) and met an existing `Rational * Rational` arm. The
prototype splits `(a * b) / c` and gets `Rational * Integer`, which has no arm.

HEAD against prototype (`$V/t3.txt`, `$V/t2.txt`):

```
1/2 * 1/3          HEAD 1/6     prototype TypeError: Cannot apply * to Rational(1/2) and Integer(1)
2/3 * 3/4          HEAD 1/2     prototype TypeError
1/2 + 1/3 * 3/4    HEAD 3/4     prototype TypeError
r = Rational(1, 3); r * 3 / 2      HEAD 1/2     prototype TypeError
e1 = 1/2; e1 * 2/3                 HEAD 1/3     prototype TypeError
```

gp: `1/2 * 1/3` = 1/6, `2/3 * 3/4` = 1/2. These are ordinary inputs, and no existing test covers them
(which is why the prototype stays 63 of 63). Note that HEAD's `*`-before-`/` order was never a value
defect: `a * (b / c)` equals `(a * b) / c` for every type the REPL divides; only the display differs.
The new negative-exponent arm makes this worse: `2**-2 * 4` is a TypeError (Rational times Integer).

Required change: either (a) add the Integer/Rational coercions to `apply_binary_op` (Integer op Rational
and Rational op Integer for `+ - * /`, and Integer / Integer giving a Rational with a ZeroDivisionError
for 0, as MAGMA, Sage and PARI do), or (b) keep HEAD's grouping for `*` and `/` (split at the rightmost
`*` first, then at the rightmost `/`) and drop the "`a * b / c`" fix. (a) is better and is small. Either
way, add regression tests that pass on HEAD and must keep passing: `1/2 * 1/3` = `1/6`,
`2/3 * 3/4` = `1/2`, `r * 3 / 2` = `1/2` (gp), and with (a) also `2**-2 * 4` = `1`, `8 / 4 * 2` = `4`,
`8 * 2 / 4` = `4`. Correct the sentence in 2.3 and move the mixed-arithmetic item out of section 9
if (a) is chosen.

### B2. S9: `eval_simple_arithmetic` (`repl.rs:5189-5253`) gives wrong values and is not covered

The design says one scanner is "used by every splitter", but this float evaluator of plot and
`find_root` ranges (called first by `eval_to_f64`, `:5168`, for every range bound) is a seventh
operator splitter that the design does not mention. It applies a leading `-` to the whole rest before it
splits at `+` and `-` (`:5197-5202`), so `-a + b` becomes `-(a + b)`:

```
find_root(w^2 == 4, -3+2, 3)     HEAD and prototype: "No root found" (the bound became -5; gp: -3+2 = -1, root 2)
find_root(w^2 == 9, -pi-1, 0)    HEAD and prototype: "No root found" (the bound became -(pi-1) = -2.14)
find_root(w^2 == 9, -1-3, 0)     HEAD and prototype: "No root found" (the bound became 2)
find_root(w^2 == 9, -4, 0)       -3 (control)
```

The same bounds go to every plot range (`plot(x, (x, -pi-1, pi+1))` silently plots [-2.14, 4.14]).
This is a silent wrong value, the kind of sign-precedence parser defect S9 names. The function also has
the `-` before `*` defect (`2*-1` fails there and only the `eval_expr` fallback saves it) and compares a
byte index with `chars().nth(i - 1)` (`:5217`).

Required change: rewrite `eval_simple_arithmetic` on the same rules as `try_eval_binary_op` (binary
`+`/`-` from `top_level_chars` with `is_binary_sign`, then `*`/`/`, then a leading sign, then grouping
parentheses), or remove its leading-sign shortcut and its `-` split so that every signed or mixed form
falls back to `eval_expr`. Add tests that fail on HEAD: `find_root(w^2 == 4, -3+2, 3)` = 2,
`find_root(w^2 == 9, -pi-1, 0)` within 1e-9 of -3, `find_root(w^2 == 9, -1-3, 0)` = -3, and
`find_root(w^2 == 4, 0, 2*-1+5)` = 2.

### B3. S15: `graph_bytes` overflows u128 for `complete_graph(n)` with n >= about 2^62

Section 6.1 says "Each arm computes graph_bytes in u128 (no overflow for any usize n)". For the complete
graph the model is v · (48 + 9 · next_power_of_two(ceil(8(v-1)/7)) + 32), about 9 · 2^65 · 2^64 = 2^132
for v = 2^64 - 1. `$V/gb.rs` (the design's code, debug): `complete_graph(3585)` 132444240,
`complete_graph(3586)` 264675488, and a panic "attempt to multiply with overflow" at
n = 4611686018427387904 (2^62) and at n = 18446744073709551615. `eval_as_usize` admits both. In a
debug build (the test profile) that is a panic, and S15 exists to prevent exactly that.

Required change: compute with `checked_mul` / `checked_add` and treat overflow as "above the limit", or
refuse n above a safe bound (for example n > 2^32 for `complete_graph`) before the model. Add
`complete_graph(18446744073709551615)` and `complete_graph(4611686018427387904)` to `test_graph_limits`
(refused with the limit message), and correct the sentence in 6.1.

## Minor

- m1. `count_bound(y)` (6.3): the design does not say how it is computed. With `y` as f64, a span above
  about 1.8·10^308 gives inf / inf = NaN, and `NaN > 10^6` is false, so the call would be admitted and
  run forever. Specify an exact Integer pre-check (refuse when y > 8·10^6, where the bound already
  exceeds 10^6) before any float, and add `primes(0, 10^400)` refused to the tests.
- m2. 4.2 says "No other unit touches rustmath-rationals". I3 edits `rational.rs:530` and
  `sage_wrapper.rs:457` (I3 design, row B1 and §9). I3 lands first, so this is only a merge note; fix the
  sentence.
- m3. S11: a Float passes through `exact_to_f64`, and float literals parse with `str::parse::<f64>`,
  which turns `1e400` into inf (`eval_expr` `:730-735`, `eval_to_f64` `:5159`). So
  `find_root(x - 1, x, 0, 1e400)` still gets an infinite bound. Either refuse non-finite float literals
  (one line in both places) or list it in section 9.
- m4. S11: state in the `exact_to_f64` doc that a nonzero rational below the least subnormal rounds to
  0.0 (the IEEE result, as Python's `float(Fraction(1, 10**400))`), and test
  `mean([Rational(1, 10^400)])` = 0, so that "outside the range of a double" is not read as including
  underflow.
- m5. sqrt_mod phase 3: the second `factorization_with_limits` call runs the sieve again on a cofactor
  that phase 2 already sieved when phase 3 found nothing (up to 2.2 s wasted at 60 digits). Pass
  `mpqs_limit: Some(0)` when c' equals the phase-2 composite.
- m6. sqrt_mod step 4: the Jacobi shortcut is restricted to odd exponents e, but a square mod q^e is a
  square mod q for every e >= 1, so (b mod q / q) = -1 proves "no root" for any e. Harmless; the
  restriction can go.
- m7. `SqrtModLimits` is called "test-only", but `sqrt_mod_report` uses it through
  `sqrt_mod_report_with(.., &SQRT_MOD_LIMITS)`. Fix the wording (it is an ordinary private struct).
- m8. `test_graph_limits` builds `G = path_graph(1000000)` in the debug test binary (334 MB peak, in
  parallel with the other tests). Use the `graph_bytes` unit check for the boundary and a smaller graph
  (for example 10^5) for the end-to-end case.
- m9. I1 dropped the `sqrt_mod(4, psi_12)` and `sqrt_mod(73, psi_12)` rows because trial division could
  not factor psi_12 = 399165290221 · 798330580441. After J1a the sieve factors it at once, so these rows
  can come back as tests (gp `factor(318665857834031151167461)`).
- m10. Unary minus makes `-(-x)` reachable; rustmath-symbolic displays it as `--x` (prototype output),
  which reads like a decrement. A library display issue; list it in section 9.

## Coverage, callers, plan

- Every item is covered; nothing is declared out of scope, and nothing needs to be.
- Callers: I grepped the workspace for `Plot3DOptions`, `parse_from_args`, `plot_points`, `pochhammer`,
  `chart(`, `sqrt_mod`, `prime_pi`, `primes(`, and the graph constructors. All uses are inside
  `rustmath-jupyter/src/repl.rs`, as the design says. `Rational::to_f64` is used across the workspace;
  the plan's full `cargo test --workspace` against the 14,173 baseline is the right check.
- `parse_symbolic_arg` (`:4928`) is the only `if let Ok(..) = self.eval_expr(..)` fallback in the file,
  and the design handles it. The broad HEAD-against-prototype diff (`$V/head6.out`, `$V/proto6.out`, 44
  symbolic and numeric commands) shows only improvements apart from B1.
- The plan order works (libraries first, then S9 before S14 and N5, which use `top_level_chars`). The
  release timing re-checks in 8.2 are needed, since the `primes` and sqrt_mod constants are provisional
  until I1 and I2 land.

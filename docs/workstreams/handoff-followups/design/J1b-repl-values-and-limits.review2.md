# Review 2 of J1b-repl-values-and-limits (independent, adversarial)

Reviewer scratch: `$R = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1b-repl-values-and-limits/review2/`.
No repository file was touched. I read the whole design (revision 1 included), review 1, the HEAD sources
of `rustmath-jupyter`, the r1 prototype, and the designs of J1a, C1, S1, MF1, I1, I2, I3, P1, M1a and M1b
where they meet `repl.rs`.

Builds in `$R`: `r1_test` (the r1 prototype's test harness, debug, rustc against the workspace rlibs),
`libr1.rlib`, `kernel_r1` (the prototype kernel binary, debug) and `client` (the designer's ZeroMQ probe).
Release runs use the designer's `r1/replmem_head_rel` (HEAD) and `r1/replmem_r1` (r1) harnesses. gp
scripts: `$R/ref_rev2.gp`, `$R/ref_rev2b.gp`.

Verdict: **approved**, no blockers. The review-1 blockers (B1 loop cost, B2 merge with J1a, B3 the C1 arms)
are fixed and I reproduced the fixes. The minor issues below should be folded in by the implementer; none
of them changes the architecture or a correctness argument.

## What I reproduced and holds

- **Provenance.** `proto/src_head` is byte-identical to HEAD's `rustmath-jupyter/src`; r1 differs from the
  original prototype only in `repl.rs`. `cargo` metadata: no crate depends on `rustmath-jupyter`, and no
  file outside the crate names `RustMathValue` or `rustmath_jupyter::`. rug 1.30.0 is in `Cargo.lock`.
  No profile sets `panic = "abort"`.
- **r1 test harness (debug):** 70 passed, 3 failed: exactly `test_int_arg_ranges_named`,
  `test_int_args_no_panic`, `test_int_args_truncation_fixed` (table 2.5). The suite takes 4.8 s.
- **F2 at HEAD and after (release, real `ReplContext`).** HEAD: `G = complete_graph(2000)` 221 MB, then
  `H = G` 364 MB; `A = identity(1000)` 118 MB, `B = A` 168 MB. r1: 77 MB and 77 MB; 42 MB and 46 MB. The
  first line drops too (HEAD holds the value, the variable and `last_result` as three copies). The HEAD
  clones are at `repl.rs:649-651`, `:625-627`, `:683`, `:706`, `:710` as stated. `grep get_mut|iter_mut|ref
  mut|&mut RustMathValue` in `repl.rs` finds nothing, so `Arc` keeps value semantics; the prototype moves
  `ReplContext` into a thread, so `Send` holds after F2.
- **S12 at HEAD.** `matrix_power` takes a u32 and calls `Matrix::pow` (`rustmath-matrix/src/matrix.rs:410-436`,
  which squares once after the last bit); `"ab" * 10000` then `* 10000` peaks at 785 MB and the next
  `* 10000` aborts with "memory allocation of 2000000000000 bytes failed" (reproduced). `pow` accepts only
  Integer/Integer with `POW_EXP` (`:1011-1023`).
- **S12 after (r1, release).** `x = 2 ** 33554431` 1.63 s; `x = 3 ** 21170489` 4.32 s; `3 ** 21170490`
  refused at once; Fibonacci matrix at 1.2·10^7, 11999999, 8388607, 6000001, 4194303: 3.6, 3.5, 2.0, 1.3,
  0.8 s (the odd exponents with many products by M are not refused by the time model); `"ab" * 10000`
  then `* 838` admitted, `* 839` refused with the new message; `t + t` refused at 33520000 bytes.
- **The trace bound (a)**, re-derived: r = M^t with t = floor(e/2^(i+1)), |tr r| <= d·ρ(M)^t,
  ρ(M)^e = ρ(M^e) <= ||M^e||_∞ <= d·max|entry| (Horn and Johnson Thm. 5.6.9), so some entry has
  log2|entry| >= 2^(i+1)·g - log2 d when g > 0. (a) refuses only results whose largest entry already
  exceeds RESULT_BITS, so it never refuses what (b) would admit. Checked by hand: [[2]] at 2^32 - 1 refuses
  at i = 30; the companion matrix of x^3 - 8 at 10^9 refuses at i = 27 (t = 3, tr = 24). (b)'s entry bound
  and (c)'s floor (products · estimate(d, 1 word)) are correct; the floor numbers in 2.3 recompute exactly
  (15.2 s for the 2x2 permutation matrix at 2^33554431, 4.8 s for [[1]] at 2^33554431 + 1).
- **D (release).** HEAD `x = sigma(2, 55000000)` 23.3 s here (design: 18.5 s), r1 3.5 s;
  `x = sigma(2, 268435456)` r1 2.1 s, VmHWM 446 MB. `Expr`'s Display (`rustmath-symbolic/src/expression.rs:437`)
  prints `Binary` as "(l op r)" (5 bytes) and unary ops as at most 9 + 2 bytes, so the per-node bounds of
  `expr_display_size` hold at HEAD.
- **S13 (debug kernels, designer's probe).** HEAD kernel: SIGINT while idle kills it. Prototype: alive
  after an idle SIGINT and `2+2` runs; queued `y = 1`, `y` are answered `aborted`, `aborted`, then the long
  cell `KeyboardInterrupt` (1.02 s after the interrupt request); `factor(<200-digit semiprime>)`: heartbeat
  echoed in 1 ms, interrupt reply in 1 ms, stderr notice at 3.0 s (2 s after the request), shutdown reply
  in 1 ms, process gone 50 ms later. HEAD's loop (`kernel.rs:51-100`) polls the sockets in turn with 100 ms
  timeouts and `handle_execute` calls `ctx.eval` inline, as stated.
- **gp references** (`$R/ref_rev2.gp`, `ref_rev2b.gp`, with `parisizemax` 2 GB): `numbpart(417)`,
  `numbpart(10^4)`, `numbpart(416)`, the falling and rising products of 5.4 (including
  `prod(i=0,2,-(2^64+5)+i)`), `#digits(2^10001)` = 3011 and `2^10001 % 10^20`, `2^1001 % 10^20`,
  `#binary(3^21170489)` = 33554432, `#binary(3^21170490)` = 33554433, `#binary(2^33554432)` = 33554433,
  `10000*#binary(3^10000)` = 158500000, `2^-1`, `(-2)^-3`, `2^-100`, `(-1)^-(10^30+1)`, `(2/3)^-2`, `(2/3)^3`,
  54724922, every matrix power of 2.6, `divrem(x^2-1, x+1)` = `[x - 1, 0]`, `#binary(sigma(2, 2^26))` =
  67108865, `#digits(2^16777215)` = 5050445. All match the design.
- **Citations.** The MAGMA digest lines quoted (ch. 1 §1.2 :24-31, ch. 3 :400, ch. 18 :125 and :270,
  ch. 20 :157, ch. 26 :204, ch. 144 :37, ch. 145 :55, ch. 49 :292) say what the design says. Knuth §4.6.3,
  Horn and Johnson 5.6.9 and Brent and Zimmermann §1.7 are the right sections.
- **Cross-unit names.** C1: `number_of_partitions(n: u64) -> Integer`, `NUMBER_OF_PARTITIONS_MAX_N` = 10^16,
  `falling_factorial_integer(x: &Integer, k: u64)`, `rising_factorial_integer` (C1 §2.2, §4.2). MF1:
  `tensor_field::component_count(n, rank) -> Result<usize>`, `DegreeOverflow`, and its §4.4
  recommendations 1 to 4 are all in plan step 8. J1a: the negative-exponent arm with `OP_POW_EXP` (its §2.2
  item 8), the Rational power arm (item 9), S14 charts with dim(M) names, `GRAPH_BYTES` (S15); J1b's merged
  semantics keep J1a's values. P1, M1a and M1b do not change `Matrix::mul` or the polynomial `Mul`, so the
  product time models stay valid.

## Minor issues (required changes)

### m1. `plot_function` / `plot_expr` has a long sampling loop and no checkpoint

Section 4.2 puts checkpoints in the 13 plot functions and says the REPL's other loops "do not need
checkpoints: their cost is bounded by a limit or by the length of the line". The `plot_function` /
`plot_expr` arm (`repl.rs:3761-3815`, loop at `:3799`) samples up to `PLOT_POINTS` = 10000 points with
`substitute` and `try_eval_to_f64` per point and is not in the list. HEAD release, `e = x + x` doubled 11
times (2^12 leaves): `plot_function(e, x, 0, 1, 10000)` takes 3.74 s, and each further `e = e + e` doubles
it (about 30 s at 2^15 leaves), with no cancellation point. `nintegrate` (`:2990-3035`, `simpson` with 101
samples) and `find_root` (`:3080-3160`, `bisection` up to 1000 steps) sample through closures passed to
rustmath-numerical, so a checkpoint cannot go there without the closure returning a fake value.
Required: a checkpoint per sample in the `plot_function` loop (and a test like `test_plot_checkpoint`);
name `nintegrate` and `find_root` among the library calls of 4.2 and section 7; correct the sentence about
the other loops.

### m2. `display_size` counts zero polynomial coefficients, so a short text is summarised as huge

`display_size` adds `int_text_bytes(c) + 24` for every coefficient, zero ones included, but the text has no
term for a zero coefficient (`poly_text` skips them). r1, release: `P = Polynomial([0, ..., 0, 1])` with
700000 zeros (text `x^700000`, 8 bytes) is shown as `<polynomial with 700001 coefficients of up to 1 bits:
its text has more than 16777216 bytes, not printed (REPL display limit)>`. The value is kept and the summary
says it is not printed, but its stated reason is false. Required: skip zero coefficients in the bound (add
nothing for them) and add this case to `test_output_budget`.

### m3. `Integer * Integer` is unguarded while `**` refuses the same result

r1, release: `x = 2 ** 33554431`, then `x ** 2` is refused at once ("more than 33554432"), while `y = x * x`
computes the 2^26-bit value in 3.2 s. Two more lines give 2^27 bits (about 12 s) and 2^28 bits (about 33 s,
section 7), each one uninterruptible library call. Section 7 lists this, but it is the same growth path that
S12 closes for `**`, and the design already bounds `Matrix * Matrix`, `Polynomial * Polynomial` and scalar
products. Required: bound `Integer * Integer` (and the Rational products J1a adds) by the result size,
bits(a) + bits(b) <= POW_BITS (or a time model like `check_scalar_product`), with a test; or, if the
designer keeps it open, say in the help text and the `POW_BITS` doc comment that `*` is not bounded.

### m4. Existing tests with hard-coded limit messages are not all listed

Section 6.3 says the tests that read these constants "format their expected messages from `repl_limits`, so
they follow the constants". Several rows of `test_int_arg_ranges_named` are literal text that the new models
make false: `:12148-12150` (`stirling1(1250, 1249)`, `stirling2(1300, 1299)`, `eulerian(1100, 1099)`
refused by the `TABLE_BYTES` table model; with C1's rolling rows and the time model, w = 2 and they are
admitted), `:12154-12155` (`multinomial(300000, [300000])` "exceeds 2500000000000", changes with the
`FACTORIAL_WORK` refit), `:12159-12160` and `:12164` (`narayana` and `delannoy` "exceeds 400000000000", if
the `binom` ladder moves `BINOMIAL_WORK`). Required: list these rows with their new inputs and expectations
(refusals at the fitted boundaries, values for the now-admitted calls with gp references, for example
`stirling(1250, 1249, 1)`), next to `test_table_limits` in 6.3.

### m5. The provisional Stirling constants contradict the evidence in the same row

With c0 = 150 ns, c1 = 1.3 ns and B = log2(n!) (gp `lngamma(n+1)/log(2)`), `stirling_seconds` gives 15.6 s
for stirling1(5000, 2500) (C1 measured 4.0 s), 128 s for (10000, 5000) (C1: 34 s) and 10.8 s for
(20000, 19900) (C1: 0.33 s): all three would be refused. B = log2(n!) is loose away from the peak
(`#binary(abs(stirling(5000, 2500, 1)))` = 32846 against 54233). The protocol refits the constants, so
nothing is wrong once it runs, but the provisional row should not suggest values that refuse 0.33 s calls.
Required: either give constants fitted on C1's three points per kind, or say that c0 and c1 are
placeholders that the `stirling` ladder replaces, and state that one constant per kind cannot fit both
the centre and the edge of a row with this B (so the fit should use the worst admitted region, the centre,
and accept over-estimates at the edges).

### m6. The `print(...)` function returns the display text as a String value

`print` as a function (`repl.rs:823-846`) builds `RustMathValue::String` from `other.to_display().text`.
After section 3, `s = print(x)` for a large x stores the summary text as a String value. This is a caller
of `to_display` whose result is data, not only output, and section 3 does not list it. Required: list it in
3.2 or 7 and choose: keep it (the returned String is what was printed, summary included) and say so in the
help text, or return the full text through `to_display_unchecked` with its own size check.

### m7. The checkpoint order in the tuple and expression-statement paths is not specified

1.2 gives value, checkpoint, display, checkpoint, store for a single assignment, and says the other two
paths "keep their structure, with the same two checkpoints before their stores". In both paths HEAD stores
first and displays after (`:625-633`, `:683-684`), and the prototype's expression path has one checkpoint
and stores before displaying. So "an interrupt during the display discards the line" holds only for single
assignments. Required: state the order for all three paths (moving the display before the stores in the
tuple path is easy) or narrow the claim.

## Nits

- `checked_matrix_power` (b) adds bit_length(k) = 1 for k = 1, so `matrix_power(Matrix([[2]]), 2^25 - 1)`
  is refused (bound 2^25 + 1 bits) while `2 ** (2^25 - 1)` (2^25 bits) is admitted. ceil(log2 k)
  (that is, bit_length(k - 1)) is the exact slack.
- `matrix_product_seconds` charges w_max^1.465 per inner step also when one operand has one-word entries
  (r·M in `matrix_power`, `A * identity(n)`), which over-estimates those products by a large factor. It
  errs towards refusing, and the size check binds first for single products, so only say so in the doc
  comment.
- After S1 §2.11, Display adds parentheses around negative Integer, Rational and Real bases of `^` and around
  Rational operands of `^` and `/`, so `((-2) ^ 2)` prints 10 bytes for an estimate of 9. Add 2 bytes per
  numeric leaf so `expr_display_size` stays an upper bound.
- `parse_symbolic_arg` (`:4928`) swallows the KeyboardInterrupt of `eval_expr`, falls back to `parse_expr`,
  and the arm then runs its library call (`taylor`, `simplify`, ...) before the next checkpoint stops the
  line. Propagating errors named KeyboardInterrupt or TimeoutError at the three swallowing sites stops
  before the library call.
- Section 5: specify k = 0 (the empty product, 1) before forming |x| + k - 1, so no u64 subtraction
  underflows.
- `bar_chart` integer labels (`:3858`) and the `matrix_inverse` text (`:1219` ff.) convert with the
  library Display outside the budget; "one helper converts every integer the REPL formats itself" should
  name them (or route the labels through `decimal_string`).
- 4.2 says the abort order "is ipykernel's behaviour"; ipykernel sends the error reply first and aborts the
  queue after it. The design's order (abort first) is better; say that it differs.
- 2.2 "So `pow` and `**` agree for every input": `**` also has IntegerMod, Symbol and Expr arms that `pow`
  does not; say "for integer and rational bases".
- 6.2 step 3 rounds limits down to 1-2-2.5-5 values, which can leave a refused call at 5 to 9 s (for
  example FIBONACCI_N = 5·10^7 refuses 6·10^7 at about 9 s), while the X2 policy says a refused call
  takes more than about 10 s. Say which rule wins.
- `remeasure_r1.sh` header still says "sections 1 to 4 and 4b" and "Policy (section 5.1)" (now sections
  1 to 5 and 6.1).
- STRING_BYTES = OUTPUT_BYTES, so `"a" * 16777216` is admitted and then summarised (its quoted text has
  2^24 + 2 bytes). Harmless; the 2.6 test should expect the summary.

## Checklist

| Check | Result |
|---|---|
| Defects confirmed at HEAD | yes: F2 (memory), S12 (`matrix_power` u32, string abort), D (sigma display), S13 (idle SIGINT kills, no replies while busy) reproduced |
| Items covered | F2, S12, S13, the limits protocol and the C1 arms all covered; nothing declared out of scope; section 7 items are outside the four items (m3 is the closest) |
| Algorithms correct | trace bound, entry bound and floor re-derived and correct; time models are fits, refit by the protocol (m5) |
| Decided / Unresolved | not applicable (no bounded searches); (a) is a certain bound and (b) says "may" |
| No facade or silent truncation | yes, except the false size reason in m2; summaries say "not printed" and give the size |
| Callers found | yes inside the crate (no dependents); `print(...)` as a value (m6) and the hard-coded test rows (m4) missing |
| Tests fail on HEAD, external references | yes; gp sample reproduced |
| Cost claims | sampled (F2, S12, D, S13); hold within machine noise |
| Plan implementable in order, crate green | yes; each step compiles on its own (step 2's loop gets its checkpoint in step 4) |

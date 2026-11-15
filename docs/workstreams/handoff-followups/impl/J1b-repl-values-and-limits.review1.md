# Review 1 of the J1b implementation (repl values and limits)

Reviewer scratch: `$R = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/J1b-repl-values-and-limits.rev1/`.
I edited no repository file and made no commit, stash or checkout. I read the whole design (revision 1),
both design reviews, the implementation report, the reports of the earlier units (handoffs to J1b from
C1, I2, I3 and J1a), the whole working-tree diff, and the new file `tests/kernel_protocol.rs` in full.
The release build for the timing samples is deleted (`target/release`), and so are my scratch binaries.

Verdict: **approved**. I found no blocker. The tests pass. Four minor issues are listed below. Each fix
is cheap, and none of them changes a value the REPL returns.

## Commands run

| Command | Result |
|---|---|
| `cargo test -p rustmath-jupyter --no-fail-fast` | lib 109 passed, 0 failed, 1 ignored (7.1 s); main 0; `kernel_protocol` 4 passed; doc 0 (`$R/test1.log`) |
| `cargo test -p rustmath-jupyter --test kernel_protocol`, twice more | 4 passed each time (4.2 s, 3.8 s); no `rustmath-kernel` process left behind |
| `cargo check --workspace --tests` | ok (`$R/check.log`) |
| `cargo clippy -p rustmath-jupyter --no-deps --all-targets` | 71 warnings. The only ones on changed lines are the pre-existing `needless_return` on `return Ok(Plot {..})` lines that only gained `.into()` (as the report says) |
| design `mem/scen_r1.sh` with a release `replmem` built from the final crate | one copy at every step: chart 192 MB, form 232 MB, `v = w` and `u = w` add nothing; `identity(1000)` and `B = A` 42 / 46 MB; `complete_graph(2000)` and `H = G` 77 MB both (`$R/scen_out.txt`) |
| design `ref_r1.gp`, `ref_r1b.gp` and the report's `ref_j1b.gp` | same values as the stored outputs (the stored files differ only in captured stderr) |
| my gp script `$R/gp/rev.gp` | every reference value of the new and changed tests (the Stirling bit counts, 780625, 844350, A(1100,1099) = 1, 19995333683325000, the matrix powers, the falling/rising products, numbpart, the powers and bit lengths) matches |
| grep for `unsafe`, em-dashes, `rustmath-misc/output.txt`, `.tmp_dir_*` | none added or changed |

## What I checked and holds

- **F2.** The eight heavy variants are behind `Arc`. All three store paths now run value, checkpoint,
  display, checkpoint, store (review 2 m7). `add_edge` copies through `Graph::clone`. Measured above.
- **S12.** I worked through `checked_int_pow`, `int_power_value`, `rational_power_value`,
  `checked_int_product`, `check_rational_op`, `check_scalar_product`, `check_poly_product` and the
  string arms, input class by input class. The classes: b = 0, ±1 and |b| >= 2; e < 0 and huge e;
  a rational base with a zero numerator; the empty string; negative counts. Each class gives either
  the gp value or an honest refusal. `checked_matrix_power`: I re-derived the trace bound (a) with
  t = floor(e/2^(i+1)), the loop invariant, the size bound with ceil(log2 k), and the floor (c). The
  property test against `Matrix::pow` is in place.
- **New rational gcd guard (deviation 1).** It is justified. `Rational::new` calls `Integer::gcd`,
  which is Lehmer's algorithm (`rustmath-integers/src/integer.rs:121`) and quadratic. Measured here:
  `gcd(a, b)` on two numbers of 2^21 bits takes 6.9 s, and `Rational(a, b)` takes 7.1 s (admitted;
  the estimate is 7.5 s).
- **D.** `display_size` is iterative and stops past the limit. The per-node bounds hold. Zero
  coefficients are skipped. `poly_text` equals the library Display (tested). The 40-doubling DAG is
  summarised in 0.01 s (release).
- **S13.** I read the select loop. Each branch is cancel safe: zeromq 0.4.1 `RouterSocket::recv` and
  `RepSocket::recv` await only `fair_queue.next()`, and tokio's `Signal::recv` is cancel safe. The
  clear-before-submit argument holds, because interrupts are set only on the kernel task and only
  while busy. The abort-before-reply order holds. Shutdown while busy works: `run` returns, then
  `main` returns, and the evaluation thread is not joined. `kernel_info_request` on control gets the
  reply only. The message types match HEAD's.
- **C1 arms.** The zero shortcuts and k = 0 are handled first. The top factor is right for each sign
  of x. The static assertion `PARTITIONS_N <= NUMBER_OF_PARTITIONS_MAX_N` is there. Edge probes in
  release: `eulerian(4294967295, 4294967294)` = 1 at once, `stirling2(0, 1)` = 0,
  `falling_factorial(-1, 3)` = -6 (gp `prod(i=0,2,-1-i)`).
- **Existing tests.** Each changed row has a gp reference or replaces a removed function (`table_bytes`,
  `product_work`). No assertion was weakened without a reference value.
- **Handoffs.** C1's two arm switches are done. So is J1a's slow `test_table_limits`: the lib suite now
  takes 7 s instead of about 7 minutes. J1a's `pochhammer` note and the MF1 §4.4 item 4 comments are
  done too. S1 and MF1 did not land (no commits, no reports), so leaving TAYLOR_ORDER,
  FORM_COMPONENTS and DIFF_FORM_DEGREE alone is correct.
- **Cost claims, release samples** (`$R/samples*.txt`). All are within machine noise of the report:

| Input | Report | Measured |
|---|---|---|
| `x = 3 ** 21170489` | 3.2 s | 3.10 s |
| `x = sigma(2, 55700000)` (printed) / `sigma(2, 268435456)` (summary) | 2.40 / 1.56 s | 2.48 / 1.67 s, 446 MB |
| `fibonacci(70000000)` | 8.6 s | 8.43 s (a first run took 10.33 s on a busier machine) |
| `partitions(19999999999)`, `(19500000001)`, `(16123456789)` | 8.4 s | 8.59, 7.85, 8.01 s |
| `schroder(250000)`, `derangements(2000000)`, `motzkin(300000)`, `bell(5000)` | 9.4, 8.1-8.7, 8.6, 5.5 s | 9.89, 9.15, 8.44, 5.44 s |
| `stirling1(6000, 3000)`, `eulerian(5000, 2500)`, `stirling2(6800, 3400)` | 5.9, 6.9, 7.3 s | 5.65, 7.37, 7.74 s |
| `falling_factorial(4294967295, 2000000)`, `rising_factorial(1, 3000000)` | 7.3, 6.9 s | 7.33, 6.90 s |
| `identity(563) * identity(563)` / `564` | admitted / refused | 7.30 s / refused at once |
| Fibonacci matrix ^ 1.2·10^7; `[[-1]]` ^ (2^33554431 + 1); `identity(8)` ^ 2^300000 | 2.1 s; 2.6 s; 4.6 s | 2.28 s; 2.71 s; 4.45 s |
| `P * P`, 15000 unit coefficients | 6.5 s | 6.11 s |
| `pow(Rational(2, 3), 1900000)` (gcd guard admits) | | 5.55 s |

Report accuracy nit: the confirmation run `J1b-work/meas/confirm/confirm.txt` was made before the last
constant change. It shows `identity(584)^2` admitted and `585` refused at "about 10.0 s", which means
MUL_STEP_NS = 40. The "563 8.5 s" row in the report's table is in no log. My run at the final constant
(46) admits 563 in 7.30 s and refuses 564, so the code and its doc comment are right.

## Minor issues

### m1. `print(...)` concatenates strings past STRING_BYTES (repl.rs:958-981)

S12 bounds string repetition and `+` at 16 MiB. The `print` function, used as a value, joins its String
arguments unchecked. Release run:

```
s = "ab" * 8388608        36 MB
t = print(s, s)           135 MB
u = print(t, t, t, t)     512 MB
v = print(u, u, u, u)     1.9 GB
```

The next line would ask for about 8 GB and abort the process with an allocation failure. That is the
exact failure that S12's string fix was meant to remove (an OOM abort is not caught by `eval_guarded`).
This behaviour is not new (HEAD's `print` joined its arguments the same way), but it leaves the S12 string
class open. Required: refuse in the `print` arm when the joined length exceeds `STRING_BYTES` (the
same message style as `+`), and add a row to `test_string_growth_guard`.

### m2. `eval_depth` leaks after a panic that `eval_guarded` catches (repl.rs:664-671, :815-817; kernel.rs:117)

J1a added `eval_depth`. `eval_expr` increments it, recurses, then decrements it, with no guard. If a
library call panics inside a nested `eval_expr`, the unwind skips the decrement. `eval_guarded` catches
the panic, and the `ReplContext` keeps a raised depth for the rest of the session. `eval` never resets
it. Each such panic lowers the remaining nesting budget by the depth at which it happened. Once the
total reaches `EVAL_DEPTH` = 200, every expression fails with "nested more than 200 levels deep". The
soundness argument in the `eval_guarded` doc (kernel.rs:117) lists the state the REPL restores, but
not this field. The design was written before J1a's change, so its argument could not cover it. I probed
about 90 commands in release with the real `eval_guarded` (`$R/panics.rs`) and found no reachable
panic today, so this is latent. Required: set `self.eval_depth = 0` at the start of
`ReplContext::eval`, and name it in the `eval_guarded` doc.

### m3. Stale timing sentences in the binomial limits (repl.rs:11037-11043, :12015-12020)

The `BINOMIAL_WORK` doc says "C(524288, 262144) (work 2^38, 9.8 s) is admitted". The
`checked_binomial` doc says "C(524288, 262143) ... takes 10.5 s". Since I3, release takes 0.03 s for
`binomial(524288, 262144)`. This is I3's handoff to "J1a/J1b", which J1a's review assigned to J1b. The
`repl_limits` doc now says that the binomial family was not re-measured. The per-constant sentences
still state the old times as if they were current. Required: mark them as measured before I3
(`c79a582`), giving the new time, or drop the times. Leaving the constants unmeasured is fine: they only
over-refuse.

### m4. `gcd`, `lcm` and `xgcd` run the same quadratic gcd unguarded (repl.rs:1021, :1035, :1054)

Deviation 1 bounds every rational reduction by `check_rational_gcd`. The three arms that call the same
Lehmer gcd directly are not bounded. `gcd(a, b)` takes 6.9 s on two numbers of 2^21 bits (measured), so
on two numbers of 2^25 bits (each admitted in a few seconds) it runs for about half an hour. It is one
uninterruptible library call. This is not new, but J1b now has the model. Required (cheap): call
`check_rational_gcd(cmd, bits(a), bits(b))` in these arms, or add them to the report's list of
unbounded library calls for the HANDOFF (design §7 names `det`, `factor` and `nth_prime`, but not gcd).

## Nits (no change required)

- `checkpoint` (repl.rs:624) formats the TimeoutError from the current `time_limit`, not from the limit
  that set the running cell's deadline. After `time_limit(100)` inside a cell that started with a limit
  of 1 s, the message says "100 s". The same happens after `time_limit(0)`, where it says "0 s".
- `charpoly` (repl.rs:1630 ff.) still formats its polynomial with the library Display (num-bigint).
  The call itself is an unbounded library call (design §7), so this matters little.
- The `EVAL_DEPTH` doc still says debug nesting can overflow below the bound. In the kernel the REPL
  now runs on a 64 MiB thread (200 levels at about 240 KB is 48 MB). The sentence remains true for
  test threads only.
- `multinomial(4294967295, [4294967295])` (which is 1) and `stirling1(n, n - 1)` (which is C(n, 2)) are
  refused by the time models although they are trivial. These are honest over-refusals.

## Review-2 notes and deviations

The report folds in all of review 2's minors and nits, except the header of `remeasure_r1.sh`. That
file is a design scratch file, so skipping it is reasonable. The deviations are all supported by
evidence: the multinomial model, MUL_STEP_NS 46, the rounding rule, POW_BITS kept at 2^25, and the
stirling2 k = 1 shortcut. I spot-checked the multinomial model against C1's code
(`rustmath-combinatorics/src/lib.rs:416`: one binomial per k_i, then a product tree), and it is a
sound upper bound for that structure.

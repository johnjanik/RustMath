# Review 1 of J1b-repl-values-and-limits (adversarial)

Reviewer scratch: `$R = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1b-repl-values-and-limits/review1/`.
No repository file was touched. Builds: the designer's rustc scripts against the HEAD debug and release
rlibs (`libhead.rlib`, `libproto.rlib`, `libproto_rel.rlib`), the kernel binaries `kernel_head` and
`kernel_proto`, the probe client, a copy of J1a's prototype (`$R/j1a/`), and a patched copy of the J1b
prototype (`$R/fixsrc/`, see B1).

Verdict: **not approved** (3 blockers). F2, D and the S13 kernel work are sound and reproduce; the
blockers are a quadratic loop in the new `checked_matrix_power` that defeats its own time guard, and
two integration gaps with units that land before J1b (J1a, C1).

## What was reproduced and holds

- HEAD = 9c24389 code, `proto/src_head` is byte-identical to HEAD's `rustmath-jupyter/src`.
- F2 defect and fix (debug, real `ReplContext`): HEAD `w = coordinate_form(M, C, 0)` (dim 1.5·10^6)
  287/194 MB, `v = w` 475/288 MB, `u = w` 569/381 MB; prototype 100/100 MB at every step.
  `complete_graph(2000)`, `H = G`: HEAD 223 then 367 MB, prototype 79 MB both. `RustMathValue` and
  `rustmath_jupyter` have no users outside the crate (grep). `Arc` gives no `&mut`, so value semantics are
  kept by construction; `Send` holds (the prototype moves the context into a thread).
- Prototype test harness: 67 passed, 3 failed, exactly the three tests the design lists in 2.5.
- S12 at HEAD: `2 ** 10001`, `pow(2, 1001)`, `matrix_power(Matrix([[1,1],[0,1]]), 10 ** 30)`,
  `(-1) ** 10^30` (via variables), `0 ** 10^30` refused; prototype gives the gp values.
- gp references (`$R/ref.gp`, `ref2.gp`): `#digits(2^10001)` 3011, `2^10001 % 10^20` = 63548609585193418752;
  `2^1001 % 10^20` = 48773674411336138752; `#binary(3^21170489)` 33554432, `#binary(3^21170490)` 33554433,
  `#binary(2^33554432)` 33554433; `[1,1;0,1]^(10^30)`, `^4294967296`, `^(2^63)`; `[2,-1,0;1,3,1;0,1,-2]^7`;
  Fibonacci^1000 residues; `[0,0,8;1,0,0;0,1,0]^30 == 8^10*matid(3)`; `[3,1;1,2]^20`;
  `#binary(sigma(2, 2^25))` 33554433; `10000*#binary(3^10000)` 158500000; the `setrand(20261112)` semiprime
  equals the test constant. All match the design.
- D: num-bigint 0.4.8 `to_string` in release: 3.36 s at 2^24 bits, 9.32 s at 2^25 (`$R/rug/nbconv.rs`).
  Release prototype: `sigma(2, 268435456)` 1.61 s with the summary; `3 ** 21170490` refused at once;
  `identity(600) * identity(600)` refused (10.4 s estimate); `identity(400)^2` 2.1 s.
- S13: HEAD kernel dies on SIGINT while idle (`client kernel_head sigint_idle`: alive false); prototype
  survives, answers `interrupt_request` in under 1 ms, `KeyboardInterrupt` 34 ms later, keeps `a`; a
  `divmod(Polynomial([1]), Polynomial([0]))` panic becomes `InternalError` and `1+2` then works.
  zeromq 0.4.1 `RouterSocket::recv`/`RepSocket::recv` are `fair_queue.next().await` and change state only
  on completion (cancel safe). No `panic = "abort"` in any profile. The worker is a `std::thread`, so the
  tokio runtime drop does not wait for it.
- MAGMA digests: ch. 1 §1.2 (`<Ctrl>-C` "interrupts at a convenient point", twice within half a second
  exits, `<Ctrl>-\`), ch. 3 §3.12 `Alarm(s)`, ch. 18 `n ^ k` (:125), ch. 26 `A ^ n` (:204) as cited.
- The trace bound of 2.2 (a) is correct: |tr M^t| <= d·ρ(M)^t and ρ(M)^e <= ||M^e||_∞ <= d·max|entry|
  (Horn and Johnson 5.6.9), and bit_length(tr) - 1 <= log2|tr| keeps it a lower bound. The entry bound of
  (b) (bA + bB + bit_length(k)) is correct.

## Blockers

### B1. `checked_matrix_power` is quadratic in bits(e); its time guard does not bound it

The loop keeps `t` (the exponent read so far) as an `Integer` and does `t = t.clone() + t`,
`t = t + 1` and `t.to_f64()` every iteration. Each is O(bits(t)), so the loop costs O(bits(e)^2), and none
of it is in `matrix_product_seconds` (which also has no per-product overhead: it was fitted on d = 2..200,
where the d^3 inner steps dominate). Release prototype (`$R/replmem_proto_rel`):

| Command | Time | Model estimate |
|---|---|---|
| `matrix_power(Matrix([[1]]), 2 ** 200000)` | 0.98 s | about 0.02 s |
| `matrix_power(Matrix([[1]]), 2 ** 1000000)` | 23.7 s | about 0.1 s |
| `matrix_power(Matrix([[0]]), 2 ** 1000000)` | 7.9 s | (trace 0, `t` only) |
| `matrix_power(Matrix([[-1]]), 2 ** 1000000)` | 23.6 s | about 0.1 s |

Five times the bits gives 24 times the time. `2 ** 33554431` is admitted by POW_BITS, so
`matrix_power(Matrix([[1]]), 2 ** 33554431)` would run about 7 hours (extrapolated) with no refusal. The
checkpoints make it interruptible, but the design's claims "(b) is a rigorous ... time check" and "everything
before it is at most the time budget" (2.2) are false, and S12 asks for exactly this guard.

Required change:
1. Drop `t`. At the start of iteration i (bits ebits-1 .. i+1 read), t = floor(e / 2^(i+1)), so
   e/t >= 2^(i+1). Use lb = 2^(i+1)·(bit_length(tr) - 1 - log2 d) - log2 d (f64; `powi` saturates to inf,
   and a non-positive bracket gives no refusal, as now). This is still a certain lower bound. Patched copy
   (`$R/fixsrc`, release): `[[1]]` at `2 ** 1000000` 0.10 s, at `2 ** 33554431 + 1` 3.2 s; `[[0,1],[1,0]]`
   at `2 ** 1000000` 0.35 s; `[[2]]`, 4294967295 still refused at once; Fibonacci 1.2·10^7 unchanged (6.2 s).
2. Add a per-product term to `matrix_product_seconds` (fit it on 1x1 and 2x2 products, including the trace,
   the two `max_entry_bits` scans and the allocation), so that MATRIX_SECONDS bounds the number of
   iterations for matrices whose entries do not grow (identity, permutation, nilpotent, [[±1]]).
3. Tests: `matrix_power(Matrix([[1]]), 2 ** 1000000)` is `[[1]]` and, with `f = 2 ** 1000001 + 1`,
   `matrix_power(Matrix([[0,1],[1,0]]), f)` is `[[0, 1], [1, 0]]` (gp `[0,1;1,0]^(2^1000001+1)`), each under
   a few seconds in debug (fails on the prototype: minutes); the `mpow` ladder of `remeasure.sh` gets
   `matrix_power(Matrix([[1]]), 2 ** 33554431)` and the permutation matrix at the same exponent.

### B2. Not reconciled with J1a, which is implemented before J1b

J1b says it "names functions and match arms" because J1a rewrites `repl.rs`, but three J1a changes break
J1b as written (checked on a copy of J1a's prototype, `$R/j1a/`):

1. **Negative exponents.** J1a adds an arm before `(Integer, "**" | "^", Integer)` that returns the
   rational 1/a^|b| (`2 ** -1` gives `1/2` in J1a's prototype; J1a §2.2 item 8 and its review), bounded by
   `int_arg(.., 0..=OP_POW_EXP, ..)`. J1b deletes `OP_POW_EXP` (2.4) and states that a negative exponent
   "stays a ValueError (the REPL does not promote)" (2.2, section 6). That is a compile break and a
   contradictory spec. Required: J1b specifies the merged semantics: J1a's rational result kept, its bound
   replaced by the POW_BITS check on the denominator (`int_pow_bits(a, |b|)`), and `pow(b, e)` with e < 0
   giving the same rational (MAGMA ch. 18 `n ^ k`, gp `2^-1` = 1/2), or an explicit reason why `pow` differs;
   the 2.6 row `pow(2, -1)` and the section 6 bullet change accordingly.
2. **One-name charts.** J1a's S14 refuses `chart(M, "t")` unless dim(M) = 1 (J1a prototype:
   `M = Manifold(3, "M")`, `chart(M, "t")` gives "chart: the manifold has dimension 3, so the chart needs 3
   coordinate names, got 1"), and MF1 already makes `exterior_derivative` refuse a one-name chart. J1b's
   `test_values_shared_not_copied` (Manifold(3), `chart(M, "t")`), the F2 harness `mem/scen.sh` named in
   the acceptance list, and the `forms` and `coords` ladders of `remeasure.sh` all use one-name charts on
   large manifolds, so they fail or measure refusals.
3. **The chart now dominates MANIFOLD_COORDS.** With J1a, a form on a 1.5·10^6-dimensional manifold needs a
   1.5·10^6-name chart: J1a's prototype takes 4.4 s and 268 MB resident (317 MB peak) for the `chart` line
   alone (debug, 12 MB command via `@file`). The provisional MANIFOLD_COORDS = 6·10^6 (section 5.3) was
   measured without it and is not credible (four times that chart plus a 380 MB form). Required: the
   ladders build the chart with dim(M) names (generated into a file, read with `@file`), and the
   MANIFOLD_COORDS and FORM_COMPONENTS memory models count the chart (after F2, one copy behind `Arc`).

### B3. The REPL halves of C1's S3/S10 have no owner; J1b's plan rests on a false condition

C1 hands to J1b, by name, the REPL arm changes its library work enables (C1 §4.2 "J1b moves the REPL to
the Integer versions"; §7 "REPL arm changes that C1 makes possible (J1b's, not done here)"): `partitions`
through `number_of_partitions(n: u64)` with a time bound (dropping the machine-word `PARTITIONS_N = 416`),
and `falling_factorial` / `rising_factorial` on any integer n through `falling_factorial_integer` /
`rising_factorial_integer`. J1a does not do them (J1a §3 only moves the `pochhammer` name; grep of J1a's
design finds neither function). J1b's section 5.3 makes the partitions switch conditional ("if J1a made
it") and never mentions the factorial functions; C1's review (m3) already flagged the gap. PARTITIONS_N
and the u32 domain of the factorial arms are exactly "repl_limits they [C1] allow" (item 4). Required:
J1b takes both arm changes unconditionally (new step in 7.1, with tests: `partitions(417)` =
18987964267331664557 and `partitions(10^4)` from gp `numbpart` (both fail on HEAD: refused above 416);
`falling_factorial(2^40, 3)` = 1329227995781289095444965371779481600 (gp `prod(i=0,2,2^40-i)`),
`rising_factorial(-5, 3)` = -60 (gp `prod(i=0,2,-5+i)`)), and the `partitions` ladder runs on them; or it declares them
out of scope with a reason and a HANDOFF line.

## Minor issues

- **m1. Section 6 rug claim.** "rug ... converts 2^27 bits in well under a second" is false here: rug 1.30
  `Display` takes 1.39 s at 2^25 bits, 8.0 s at 2^27, 18.3 s at 2^28 (release, `$R/rug/rugconv.rs`). rug is
  already in rustmath-jupyter's graph at HEAD (through rustmath-complex, -matrix, -rationals; `cargo tree -i
  rug`), not "through C1". Still worth taking: BigInt to rug costs 2 ms at 2^25 bits, so a rug display
  would cut `3 ** 21170489` from 12.5 s to about 4 s and let INT_DISPLAY_BITS rise to about 2^26 under the
  10 s policy.
- **m2. D is a per-integer cap.** Composite outputs stay unbounded: `[a, a, a, ...]` with a 2^25-bit `a`
  costs 9 s per element; an `Expr` with a large Integer leaf is printed twice (text and `expr_to_latex`);
  `Polynomial([a])` and `Polynomial * Polynomial` (`:9746`) exist, so "coefficients do not grow without a
  product command" is wrong. Either a per-output digit budget or an honest line in section 6; "Lists ...
  are covered" should say "per element".
- **m3. `divmod` by the zero polynomial is left panicking to serve as a test fixture.** Fix the arm
  (ZeroDivisionError) and drive `kernel_reports_panics` with another trigger, or rely on the
  `eval_guarded` unit test.
- **m4. Prototype and design disagree.** The prototype refuses on `lb > LIMIT` (design: `lb - 1.0 >`); its
  `checkpoint` doc says "consuming it" (design: sticky); its assignment still clones twice and displays
  after storing. With the design's order (display before store) an interrupt during a long display still
  stores the line, so "an interrupted line has no effect" holds only for interrupts before the display.
  Say so.
- **m5. "Every REPL loop other than matrix_power is bounded to well under a second"** is asserted, not
  measured: plot and grid sampling cost scales with the expression size. Checkpoints in the sampling loops
  are cheap and in the crate; add them or measure.
- **m6. GRAPH_BYTES** (J1a S15) is derived from "an assignment holds three copies"; after F2 it should be
  re-derived in section 5 like FORM_COMPONENTS.
- **m7. FIBONACCI_N parenthetical** ("up to about 8·10^7 if the summarised range is under 10 s")
  contradicts the protocol's "worst admitted input": n just below the display cutoff (4.83·10^7) costs
  about 5 s plus 9.3 s of display, so any limit above about 4·10^7 admits calls over 10 s. Same shape for
  DERANGEMENTS_N. The protocol (step 3) is right; fix the provisional text.
- **m8. Adjacent growth path.** `Matrix * Integer` (`scalar_mul`) has no size check: `ones(100) * x` with a
  2^25-bit x asks for 10^4 times 4 MiB (an out-of-memory abort, which S13 cannot catch). The same entry
  bound as `check_matrix_product` would cover it. The string-repetition message nests parentheses
  ("(the result may have at most 16777216 bytes (REPL limit))").
- **m9.** MF1 §4.4 item 4 (stale `diff_form.rs` / `tensor_field.rs` line references in the arms) is not in
  the plan.

## Checklist

| Check | Result |
|---|---|
| Defects confirmed at HEAD | yes (F2, S12, D, S13 all reproduced) |
| Items covered | F2, S12, S13 yes; item 4 protocol yes, but the forms/coords ladders and the C1 arm switches are broken or conditional (B2, B3) |
| Algorithms correct | trace and entry bounds correct; loop cost wrong (B1) |
| Decided/Unresolved | not applicable (no bounded searches); the trace refusal is a certain bound |
| No facade or silent truncation | yes (summaries say "not printed"; refusals name the limit) |
| Callers found | yes inside the crate; cross-unit callers of the same arms (J1a, C1) not reconciled (B2, B3) |
| Tests fail on HEAD, external references | yes; gp values checked |
| Cost claims | F2 and D hold; matrix_power fails (B1); rug claim false (m1) |
| Plan implementable in order | not as written: J1a's arm uses a removed constant, and the specified test and harnesses use refused charts (B2) |

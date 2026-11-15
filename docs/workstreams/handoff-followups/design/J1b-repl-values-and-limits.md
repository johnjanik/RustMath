# J1b-repl-values-and-limits: shared REPL values, growth guards, a real interrupt, the C1 arms and the limit re-measurement

Unit J1b of the HANDOFF "Open follow-ups" (`docs/HANDOFF.md:283-311` at `9c24389`; HEAD `c79a582` has the
same code). Crate: `rustmath-jupyter` only (`Cargo.toml`, `src/repl.rs`, `src/kernel.rs`, `src/handlers.rs`,
`src/protocol.rs`, `src/install.rs`, new `tests/kernel_protocol.rs`). J1b is implemented last, after I3, I1,
I2, P1, F1, R1, M1a, M1b, S1, MF1, C1 and J1a. J1a rewrites parts of `repl.rs` (parser S9, S10, S11, S14,
S15), so this document names functions and match arms, and gives HEAD line numbers only as a guide.
Section 0.1 lists what J1b takes from J1a and C1 and how it merges with them.

## Revision 1 (after review 1)

| Review issue | Change |
|---|---|
| B1: `checked_matrix_power` keeps an `Integer` counter `t`; the loop is O(bits(e)^2) and the time model does not count it (`[[1]]` at `2 ** 1000000`: 23.7 s) | The counter is gone. At the start of iteration i, r = M^t with t = floor(e/2^(i+1)), so e/t >= 2^(i+1); the trace bound uses that power of two (2.2 (a)). `matrix_product_seconds` gets a per-product term (100 ns, fitted on d = 1..32 in release, 2.3), and an up-front floor refuses at once when the number of products times the smallest product estimate exceeds the budget (2.2 (c)). Revised prototype `$J/r1/` (release): `[[1]]` at `2 ** 1000000` 0.12 s, at `2 ** 33554431 + 1` 3.58 s; `[[-1]]` 3.83 s; `[[0,1],[1,0]]` at `2 ** 1000000` 0.43 s and at `2 ** 33554431 + 1` refused at once (floor 15.2 s); `[[2]]`, 4294967295 refused at once; Fibonacci 1.2·10^7 2.64 s. New tests and ladder rows (2.6, 6.2). |
| B2 (1): J1a's `Integer ** negative` arm returns 1/a^\|b\| bounded by `OP_POW_EXP`, which J1b deleted while saying negative exponents stay errors | Merged semantics (2.2): J1a's rational result is kept; one helper `int_power_value` serves `**`, `^` and `pow`; for e < 0 the denominator b^\|e\| is bounded by `POW_BITS` through `int_pow_bits`. `pow(2, -1)` is `1/2` (MAGMA ch. 18 and ch. 20, gp `2^-1`). J1a's revision-1 arm `Rational ** Integer` goes through `rational_power_value` with the same bound. No use of `OP_POW_EXP` or `POW_EXP` remains (plan step 0 greps for them). The 2.5 row for `pow(2, -1)` and the section 7 bullet changed. |
| B2 (2): J1a's S14 refuses one-name charts on dim(M) > 1, so J1b's F2 test, `scen.sh` and the forms/coords ladders measured refusals | Every chart in J1b's tests, harnesses and ladders has dim(M) names: `chart(M, "a,b,c")` in `test_values_shared_not_copied`; `$J/mem/scen_r1.sh` and `$J/remeasure/remeasure_r1.sh` write `C = chart(M, "x0,x1,...")` to a file and read it with `@file`. |
| B2 (3): the chart dominates MANIFOLD_COORDS; 6·10^6 was measured without it | Measured with a dim(M)-name chart behind `Arc` (r1 prototype, release): chart line 0.76-0.85 s, 163 MB peak, 149 MB resident at n = 1.5·10^6; then `w = coordinate_form(M, C, 0)` 231 MB; at 2.5·10^6: 273 MB and 339 MB; at 3·10^6: 313 MB and 406 MB. Provisional `MANIFOLD_COORDS` = 2.5·10^6 (6.3). The FORM_COMPONENTS model counts the chart too (negligible there: n <= 2236 names). |
| B3: C1's REPL arm switches (partitions through `number_of_partitions`, falling/rising factorial of any integer) have no owner | J1b takes both unconditionally: new section 5 and plan step 7, with tests from gp (`numbpart(417)`, `numbpart(10^4)`, `prod(i=0,2,2^40-i)`, `prod(i=0,2,-5+i)`; these fail on HEAD), a time model for the factorial products, and the `partitions` and `falling` ladders. |
| m1: the rug claim was false; rug is already in the graph | Corrected (7). And taken: `decimal_string` converts integers above 2^15 bits through rug (GMP `mpz_get_str`). Measured (release): `x = 2 ** 33554431` 1.48 s end to end (10.1 s with num-bigint's conversion in the original prototype; HEAD refuses it), `3 ** 21170489` 3.82 s (12.5 s), `sigma(2, 55000000)` 3.02 s (HEAD 18.5 s). rug becomes a direct dependency of rustmath-jupyter, with the same declaration as `rustmath-rationals/Cargo.toml:16` (already in `Cargo.lock`). |
| m2: the display cap was per integer only | One budget per output: `display_size` bounds the bytes of a value's text before anything is formatted (integers, rationals, lists, matrices, polynomials, symbolic expressions, charts), and a value above `OUTPUT_BYTES` = 16 MiB is shown as a one-line summary (3.2). Found on the way: a symbolic DAG prints exponentially: `e = e + e` 23 times takes 1.6 s and 135 MB to print at HEAD and doubles per line; with the budget the summary of 40 doublings takes under 2 s in debug. Integers inside a symbolic expression are printed by rustmath-symbolic with num-bigint, so they get their own cap (`LIBRARY_INT_DISPLAY_BITS` = 2^22). Polynomials are now formatted by the REPL (`poly_text`, equal to the library Display, tested), each coefficient converted once. |
| m3: `divmod` by the zero polynomial was kept as a panic fixture | The arm calls `div_rem` and maps `DivisionByZero` to ZeroDivisionError (4.2). The integration test `kernel_reports_panics` is replaced by a unit test of the real evaluation loop with a stub evaluator that panics (4.5). |
| m4: prototype and design disagreed | The prototype is aligned: `lb - 1.0 >` in both; `checkpoint` is sticky and documented so; the assignment computes the value, checkpoints, displays, checkpoints again, then stores it with one clone (O(1) for the shared variants) and moves it into `last_result`. With the second checkpoint an interrupt during the display also discards the line, so "an interrupted line has no effect" holds up to the store (4.2). |
| m5: "other REPL loops are bounded well under a second" was not measured | Measured: `plot(e, (x, 0, 1))` with e a DAG of 2^18 leaves takes 8.9 s at HEAD, `plot3d` 48 s, doubling per `e = e + e`. Every sampling loop of the 13 plot functions gets a checkpoint per sample (4.2). One sample is still a library call (`substitute` and `eval_float`); that gap is listed in section 7. |
| m6: GRAPH_BYTES was sized for three copies | Added to 6.3 with ladder `graph`. Measured (r1, release): a graph exists once after F2 (`complete_graph(5000)`: 366 MB, model 369 MB), and `add_edge` copies it (727 MB). Provisional GRAPH_BYTES = 200 MB, so `add_edge` stays under 400 MB. |
| m7: the FIBONACCI_N / DERANGEMENTS_N text contradicted protocol step 3 | Rewritten: the provisional target is the largest n whose compute plus display is at most 10 s (6.3), 5·10^7 and 10^6. |
| m8: `Matrix * Integer` had no size check; nested parentheses in a message; MF1 §4.4 item 4 not planned | `check_scalar_product` bounds `Matrix * Integer`, `Integer * Matrix`, `scalar_mul` and `Polynomial * Integer`; `Polynomial * Polynomial` gets a size and time check like matrices; `String + String` is bounded by `STRING_BYTES` (2.2). The repetition message reads "... in 0..=8388608 (REPL limit: the result has at most 16777216 bytes)". MF1 §4.4 item 4 (stale line references in the form arms) is plan step 8. |

Revised prototype: `$J/r1/src/` (HEAD to r1: `$J/r1/r1_repl.rs.diff`; original prototype to r1:
`$J/r1/r1_vs_proto_repl.rs.diff`; `kernel.rs`, `handlers.rs`, `protocol.rs` as in `$J/proto/src/`). Debug
test run: 70 of 73 pass; the three failures are exactly the HEAD tests of table 2.5.

---

Items (quoted from the task):

1. **F2 (REPL part).** "the REPL clones a value on every assignment and lookup (repl.rs:649, :706)".
2. **S12.** "a growth-based size guard for matrix_power and ** (repl.rs:1530, :9675)".
3. **S13.** "no evaluation timeout or interrupt" (the kernel runs eval on the main thread, handlers.rs:81).
4. **Limits.** After S1, C1 and MF1 land: re-measure and raise the `repl_limits` they allow
   (TAYLOR_ORDER and its stack note, FIBONACCI_N, BELL_N, MOTZKIN_N, DERANGEMENTS_N, the TABLE_BYTES users,
   FORM_COMPONENTS / DIFF_FORM_DEGREE) under the X2 policy, with a measurement protocol specified now.
   C1 hands two REPL arm changes to J1b as part of this (C1 §4.2, §7): `partitions` through
   `number_of_partitions(u64)` (dropping the machine-word `PARTITIONS_N = 416`), and `falling_factorial` /
   `rising_factorial` on any integer through `falling_factorial_integer` / `rising_factorial_integer`.

One more defect was found while calibrating S12 and is fixed here because S12 and item 4 depend on it:

5. **D (found).** Display of large values. `sigma(2, 268435456)` is admitted by the X2 limits and takes 207 s
   in release, almost all of it in `Integer::to_string`; a symbolic DAG prints exponentially.

Nothing is declared out of scope. Section 4.2 states exactly what an interrupt stops and what it does not:
a single library call is never stopped (a thread cannot be killed in Rust), and the only hard stop is the
process exit that a kernel restart or shutdown performs.

Scratch: `$J = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/J1b-repl-values-and-limits`.
The crate sources were not modified. All experiments compile a copy of the crate with `rustc` against
the workspace rlibs (debug: `target/debug/deps`; release: `target/release/deps`, built with
`cargo build --release -p rustmath-jupyter --lib`).

| Path | Content |
|---|---|
| `$J/proto/src_head/` | the crate sources at HEAD |
| `$J/proto/src/` | the original prototype (F2, S12, D, S13); `proto_*.rs.diff`: HEAD to it, file by file |
| `$J/r1/src/` | the revised prototype (revision 1; `repl.rs` only differs from `proto/src`). `r1_repl.rs.diff`: HEAD to r1; `r1_vs_proto_repl.rs.diff`: proto to r1; `patch_r1.py`, `patch_r1_display.py`: the scripted part of the revision (the test hunk and the one-pass matrix display were edited in place) |
| `$J/r1/buildtest.sh`, `buildlib_rel.sh`, `buildbin_rel.sh` | rustc builds of the crate copy with `rug` added (debug test harness, release lib, release harness) |
| `$J/r1/mpow_r1.txt`, `disp_r1.txt`, `coords_rel.txt`, `graph_r1.txt`, `scen_r1_rel.txt` | revision-1 measurements (release): matrix powers, display, chart plus form, graphs, F2 scenarios |
| `$J/r1/gp/ref_r1.gp`, `ref_r1b.gp`, `ref_r1c.gp` and `.out` | gp references added in revision 1 |
| `$J/mem/replmem.rs`, `scen.sh` (HEAD scenarios, one-name charts), `scen_r1.sh` (revision 1: dim(M)-name charts via `@file`) | the X2 review-3 harness (real `ReplContext`, one REPL line per argument, VmHWM and VmRSS after each line; `@file` reads a long line from a file) |
| `$J/mem/head_scen*.txt`, `p_scen*.txt`, `sigma_head_rel.txt`, `str_head.txt` | HEAD and original-prototype measurements |
| `$J/s12/powbench.rs`, `mulfit.rs`, `disp27.txt` | release timings of `Integer::pow`, num-bigint decimal display, `Matrix::mul`, `Matrix::pow` |
| `$J/s13/client.rs`, `run*.txt` | a ZeroMQ client that drives the kernel binary like Jupyter (shell, control, iopub, heartbeat, SIGINT); results on HEAD and on the prototype |
| `$J/remeasure/remeasure_r1.sh`, `replmem.rs` | the limit re-measurement protocol of section 6, runnable (`remeasure.sh` and `head_run/`: the original version and its check run at HEAD) |

Spec references. MAGMA Handbook digests: ch. 1 §1.2 `<Ctrl>-C` ("Magma interrupts at a convenient point.
If typed twice within half a second, Magma exits immediately"; `<Ctrl>-\` quits at once) and ch. 3 §3.12
`Alarm(s)` (`rustmath-lava/docs/Chapter_01_Statements_and_Expressions.md:24-31`,
`Chapter_03_Input_and_Output.md:400`); ch. 18 integer `n ^ k` (`Chapter_18_Ring_of_Integers.md:125`) and
`NumberOfPartitions(n)` ("n must be small", `:270`); ch. 20 rational `a ^ k`
(`Chapter_20_Rational_Field.md:157`); ch. 26 `A ^ n` ("A^0 is the identity. If n < 0, A must be
invertible", `Chapter_26_Matrices.md:204`); ch. 144 `NumberOfPermutations(n, k)` (the falling factorial,
`Chapter_144_Enumerative_Combinatorics.md:37`); ch. 145 `NumberOfPartitions(n)`
(`Chapter_145_Partitions_Words_and_Young_Tableaux.md:55`); ch. 49 Pochhammer symbol
(`Chapter_49_Power_Laurent_and_Puiseux_Series.md:292`). Jupyter: the messaging protocol (5.3 and later:
`interrupt_request` on the control channel; `execute_request` field `stop_on_error`, default true;
`execute_reply` status `aborted`) and the kernel spec field `interrupt_mode` (`signal` or `message`),
jupyter-client documentation. Algorithms: left-to-right binary powering, D. E. Knuth, *TAOCP* vol. 2, 3rd ed.,
§4.6.3; spectral radius bounds ρ(A) <= |||A||| and ρ(A^k) = ρ(A)^k, R. A. Horn and C. R. Johnson, *Matrix
Analysis*, 2nd ed., Thm. 5.6.9; Toom-3 cost exponent log 5 / log 3, Knuth §4.3.3; radix conversion cost,
R. P. Brent and P. Zimmermann, *Modern Computer Arithmetic*, §1.7; balanced product trees, J. von zur Gathen
and J. Gerhard, *Modern Computer Algebra*, 3rd ed., §10.1.

---

## 0. Summary

| Item | At HEAD (measured) | Fix |
|---|---|---|
| F2 | `w = coordinate_form(M, C, 0)` on a 1.5·10^6-dimensional manifold peaks at 285 MB for a 94 MB form (3 copies); `v = w` and `u = w` raise the resident memory to 379 MB; `wedge(w, w, C)` copies `w` twice; `H = G` for `complete_graph(2000)` goes from 221 to 365 MB | The heavy payloads of `RustMathValue` move behind `Arc` (Matrix, Polynomial, List, Graph, the Plot SVG, Chart, DiffForm, VectorField). Every clone of such a value is O(1). Measured (r1, release, a 1.5·10^6-name chart): chart 149 MB, chart plus form 231 MB, `v = w` and `u = w` free; the graph 77 MB before and after `H = G` |
| S12 | `2 ** 10001` and `pow(2, 1001)` are refused (exponent bounds), while `a ** 10000` with `a = 3 ** 10000` (1.6·10^8 bits, minutes) is admitted; `matrix_power(Matrix([[2]]), 4294967295)` runs until memory is gone; `matrix_power(M, 10**30)` is refused although M = [[1,1],[0,1]] gives [[1, 10^30],[0,1]]; `identity(1000) * identity(1000)` runs about 35 s; `s * 10000` with a 20 kB string aborts with "memory allocation of 2000000000000 bytes failed"; `ones(100, 100) * x` with a 2^25-bit x asks for 40 GiB | Guards on the size of the result instead of on the exponent: powers of integers and rationals refuse more than `POW_BITS` = 2^25 bits (exact before the power is taken; negative exponents give J1a's rational); `matrix_power` is computed by the REPL with left-to-right binary powering, a certain early refusal from the trace, an up-front floor on the number of products, and a size and time check before every product, any non-negative exponent; `Matrix * Matrix`, `Polynomial * Polynomial` and scalar products get size checks (and time checks for the products); strings refuse results above 16 MiB |
| D (found) | `sigma(2, 268435456)` (admitted, 2^28 bits) takes 207 s in release, 205 s of it printing; num-bigint's decimal conversion takes 9.3 s at 2^25 bits, 26 s at 2^26, 76 s at 2^27; a symbolic DAG of depth 23 takes 1.6 s and 135 MB to print and doubles per level | Integers above 2^15 bits are converted through GMP (rug): 1.4 s at 2^25 bits. One budget per output (`OUTPUT_BYTES` = 16 MiB of text, estimated before formatting): above it the value is kept and shown as a summary that gives its size. `sigma(2, 268435456)` takes 1.76 s |
| S13 | `interrupt_request` gets no reply while a cell runs (none within 10 s; the cell runs to its end); the heartbeat is not answered; `shutdown_request` gets no reply; SIGINT, which Jupyter sends by default (`interrupt_mode` absent means `signal`), **kills the kernel, even when idle**; a library panic kills the kernel | The REPL runs on a dedicated thread with a 64 MiB stack; the async loop serves shell, control, heartbeat and SIGINT at once (`tokio::select!`). Interrupts and an optional time limit are cooperative: they stop a cell at the next checkpoint (between lines, between sub-expressions, before storing a result, between the products of `matrix_power`, between plot samples). Measured: interrupt reply in 1 ms, `KeyboardInterrupt` reply 10 to 60 ms later, variables kept; queued cells answered `aborted` (`stop_on_error`); a library call that ignores the interrupt gets a stderr notice after 2 s, and shutdown or restart ends the process at once; a panic becomes an `InternalError` |
| C1 arms | `partitions(417)` refused (machine word); `falling_factorial(2^40, 3)`, `rising_factorial(-5, 3)` refused (u32 n) | `partitions` through `number_of_partitions(u64)` with a time limit; the factorial arms take any integer n and a u64 k, with zero products answered at once and a product-tree time model |
| Limits | Sized for the old callees and with three copies of every value | Protocol and script in section 6; provisional targets from the S1, C1 and MF1 measurements; the Stirling/Eulerian memory model becomes a time model; `DIFF_FORM_DEGREE` goes; `FORM_COMPONENTS`, `MANIFOLD_COORDS` and `GRAPH_BYTES` are re-derived with one copy and the chart counted |

### 0.1 What J1b takes from J1a and C1

* **J1a (implemented before J1b).** The S9 parser (string literals with commas, signed operands, grouping
  parentheses), so tests write `(-1) ** e`, `2 ** -1` and `chart(M, "a,b,c")` directly. S14: `chart`
  takes exactly dim(M) comma-separated names. S15: `GRAPH_BYTES` and `graph_bytes` (J1b re-derives the
  constant, 6.3). The power arms: J1a adds `(Integer, "**" | "^", Integer)` for a negative exponent
  (the rational 1/a^|b|, bounded by `OP_POW_EXP`), and its revision 1 adds `(Rational, "**" | "^", Integer)`
  and mixed Integer/Rational arithmetic. J1b replaces the bounds of both power arms (2.2) and deletes
  `OP_POW_EXP` and `POW_EXP`; the values J1a's tests expect (`2 ** -1` = 1/2, `-2**-2` = -1/4,
  `(-2)**-3` = -1/8, `0 ** -1` ZeroDivisionError, `2**-2**2` = 1/16) are unchanged.
* **C1 (implemented before J1b).** `number_of_partitions(n: u64) -> Integer` with
  `NUMBER_OF_PARTITIONS_MAX_N` = 10^16, `falling_factorial_integer(x: &Integer, k: u64)`,
  `rising_factorial_integer(x: &Integer, k: u64)` (C1 §2.2, §4.2), and the faster callees whose costs C1 §7
  lists. If C1 landed under other names, map them; the arms of section 5 need only these three functions.

---

## 1. F2 (REPL part): values behind `Arc`

### 1.1 The defect at HEAD

`repl.rs` (HEAD):

* `:649-651` (single assignment): `self.variables.insert(var_name, value.clone()); self.last_result = Some(value.clone()); return Ok(value.to_display());` Two deep copies; the evaluated value is the third copy at the peak.
* `:625-627` (tuple unpacking): a deep copy per element and one of the list.
* `:683` (expression statement): `self.last_result = Some(value.clone())`.
* `:706` (variable lookup in `eval_expr`): `return Ok(value.clone())`. Every argument that names a variable is a deep copy, because `parse_args` and the single-argument arms call `eval_expr`.
* `:710` (`_`): the same for `last_result`.

`RustMathValue` (`:142-186`) holds its payloads inline, so `clone` copies a whole matrix, list, graph, chart
or form. The X2 limits already price this in: `MANIFOLD_COORDS` (`:10535-10545`) cites "three copies at
the peak" and a copy "on every argument lookup".

Measured with the real `ReplContext` at HEAD (`$J/mem/replmem.rs`, one process per scenario; HEAD allows a
one-name chart; release):

| Scenario (line) | HEAD time | HEAD VmHWM / VmRSS |
|---|---|---|
| `w = coordinate_form(M, C, 0)`, dim 1.5·10^6 (94 MB a copy) | 0.18 s | 285 / 191 MB |
| `v = w` | 0.20 s | 473 / 285 MB |
| `u = w` | 0.20 s | 566 / 379 MB |
| `coordinate_form(M, C, 0)` (not assigned) | 0.12 s | 566 / 379 MB |
| `wedge(w, w, C)` (refused by FORM_COMPONENTS after the lookups) | 0.14 s | peak 285 MB from RSS 98 MB |
| `L = [10^5 integers of 101 bits]` | 0.14 s | 63 / 53 MB |
| `K = L` | 0.04 s | 74 / 74 MB |
| `P = [L, L, L, L]` | 0.18 s | 255 / 206 MB |
| `A = identity(1000)` | 0.17 s | 118 / 76 MB |
| `B = A` | 0.18 s | 168 / 150 MB |
| `G = complete_graph(2000)` | 0.43 s | 221 / 221 MB |
| `H = G` | 0.09 s | 365 / 365 MB |

(`$J/mem/head_scen_rel.txt`, `graph_rel_replmem_head_rel.txt`, `r1/replmem_head_rel` for the identity rows.
`EuclideanSpace` already holds its manifold in an `Arc`: 387 MB before and after.)

### 1.2 Fix

Change the heavy variants of `RustMathValue` to hold an `Arc`:

```rust
pub enum RustMathValue {
    ...
    Matrix(Arc<Matrix<Integer>>),
    Polynomial(Arc<UnivariatePolynomial<Integer>>),
    List(Arc<Vec<RustMathValue>>),
    Graph(Arc<Graph>),
    /// A plot with SVG content and description
    Plot { description: String, svg: Arc<str> },
    ChartVal(Arc<Chart>),
    DiffFormVal(Arc<DiffForm>),
    VectorFieldVal(Arc<VectorField>),
    ...   // the other variants unchanged
}
```

Why `Arc` and not `Rc`: `ReplContext` is `Send + Sync` at HEAD (checked with `fn is_send<T: Send>()`,
`$J/sendcheck.rs`), and S13 moves the context into its worker thread. `Arc` keeps that. The atomic count
costs nothing measurable here.

Why these variants: they are the ones whose size the REPL lets grow beyond a few MB. Measured at HEAD: a
chart or form of 1.5·10^6 coordinates (about 90 MB each), `identity(1000)` (37 MB), `complete_graph(2000)`
(72 MB), lists of 10^5 to 10^6 items, plot SVGs of several MB, polynomials from `charpoly` of large
matrices. The other variants stay inline: `CliffordAlgebra(10)`, its volume form, `JordanAlgebra(20)` and
`SpecialJordanAlgebra(10)` change resident memory by less than 0.1 MB (`$J/mem/clifford_head.txt`); `Expr`
already shares its children (`Binary(op, Arc<Expr>, Arc<Expr>)`); `Symbol`, `Bool`, `Float`, `Complex`, the
points and `IntegerMod` are small; `Manifold` is already `Arc`. `Integer`, `Rational` and `String` stay
inline although their clone is a memcpy: a single command makes at most a 2^28-bit `sigma` result (32 MiB,
cloned in about 5 ms) or a 16 MiB string (S12), and wrapping `Integer` would touch about 470 sites. The cost
of that choice (a list literal of k copies of a large integer holds k copies) is listed in section 7.

Value semantics are kept by construction: `Arc` gives no `&mut`, so any code that modified a shared payload
would stop compiling. The only arm that modifies a payload is `add_edge` (`:3388-3409`); it already works on
a copy (`let mut g = match &args[0] { Graph(g) => g.clone(), .. }`), which becomes `Graph::clone(g)` and
`Ok(RustMathValue::Graph(Arc::new(g)))`. No other arm mutates a value (`grep get_mut` finds nothing).

The edits, found by the compiler (prototype `$J/proto/proto_repl.rs.diff`; `$J/autofix.py` applied rustc's
suggestions):

| Kind | Count | Edit |
|---|---|---|
| A construction `RustMathValue::X(v)` of a wrapped variant | 88 | `RustMathValue::X(v.into())` (`Arc<T>: From<T>`) |
| `Plot { description, svg }` field shorthand | 28 | `Plot { description, svg: svg.into() }` |
| `to_display` of `Plot` (`:444`) | 1 | `result.svg = Some(svg.to_string())` |
| `multinomial` iterates the list (`:2311`) | 1 | `for k in ks.iter()` |
| Polynomial `+`, `-`, `*` (`:9740-9748`) | 3 | `Polynomial(Arc::new((**p).clone() + (**q).clone()))` |
| `add_edge` | 2 | as above |

Pattern matches need no change: `RustMathValue::Matrix(m)` binds `&Arc<Matrix<_>>`, method calls and
`&T` parameters auto-deref. The tests module compiles unchanged and all 63 HEAD tests pass with F2 alone.

The assignment path computes the value once and moves it:

```rust
// eval_line, single assignment (HEAD :646-651)
let value = self.eval_expr(expr)?;
self.checkpoint()?;                                   // S13: an interrupt before the display
let display = value.to_display();                     // D: bounded by OUTPUT_BYTES
self.checkpoint()?;                                   // S13: an interrupt during the display
self.variables.insert(var_name, value.clone());       // O(1) for the Arc variants
self.last_result = Some(value);                       // moved, not cloned
return Ok(display);
```

Tuple unpacking (`:614-635`) and the expression statement (`:682-684`) keep their structure, with the same
two checkpoints before their stores; their clones are O(1) for the wrapped variants. The lookup at `:706`
stays `Ok(value.clone())`, which is now O(1) for the wrapped variants.

Invariant: after the change, `RustMathValue::clone` allocates at most O(1) memory for every wrapped
variant, and at most the size of one inline payload (bounded by the limits above) otherwise. Correctness:
the payload types are immutable behind `Arc`, so sharing cannot be observed except through `Arc::ptr_eq`.

### 1.3 Callers and behaviour changes

`RustMathValue` is `pub`, but no crate outside `rustmath-jupyter` uses it (`grep -rn RustMathValue` over
the workspace, excluding `target/`). No output text changes. `{:?}` of a wrapped payload prints the same
text (`Arc<T>: Debug` forwards). The limits whose doc comments count copies (`MANIFOLD_COORDS`,
`FORM_COMPONENTS`, J1a's `GRAPH_BYTES`) are re-derived in section 6.

### 1.4 Tests (new, in `repl.rs` tests; prototype `j1b_values_shared_not_copied` passes)

* `test_values_shared_not_copied`: `M = Manifold(3, "M")`, `C = chart(M, "a,b,c")` (J1a's S14 needs
  dim(M) names), `w = coordinate_form(M, C, 0)`, `v = w`: the `DiffFormVal` payloads of `w`, `v` and
  `last_result` are `Arc::ptr_eq`. `L = [1, 2, 3]`, `P = [L, L]`: both elements of `P` are `ptr_eq` to
  `L`'s list. `A = identity(3)`, `B = A`: `ptr_eq`. Fails on HEAD (does not compile: the payloads are not
  `Arc`). (The r1 prototype writes the names blank-separated, `"a b c"`, because it does not contain J1a's
  parser; see the note on its chart arm in 1.6.)
* `test_value_semantics_kept`: `G = Graph(3)`, `H = add_edge(G, 0, 1)`: `num_edges(G)` is `0`,
  `num_edges(H)` is `1`. Passes on HEAD and after (guards copy-on-write).

No memory test in `cargo test`: the test threads share one process, so VmHWM would be unreliable. The
measurement is `$J/mem/scen_r1.sh <replmem built from the final crate>` (section 8.2).

### 1.5 Measured after the change (r1 prototype, release, `$J/r1/scen_r1_rel.txt`)

| Scenario (line) | Time | VmHWM / VmRSS |
|---|---|---|
| `C = chart(M, <1.5·10^6 names>)` (a 12 MB line, `@file`) | 0.84 s | 163 / 149 MB |
| `w = coordinate_form(M, C, 0)` | 0.05 s | 231 / 231 MB |
| `v = w`, then `u = w` | 0.00 s | 231 / 231 MB |
| `coordinate_form(M, C, 0)` (a second form, not assigned) | 0.05 s | 325 / 325 MB |
| `wedge(w, w, C)` (refused by FORM_COMPONENTS) | 0.00 s | 231 / 231 MB |
| `L = [10^5 integers of 101 bits]`, `K = L` | 0.10 s, 0.02 s | 35 / 35 MB, 35 / 32 MB |
| `P = [L, L, L, L]` | 0.08 s | 80 / 23 MB (the peak is the display text) |
| `A = identity(1000)`, `B = A` | 0.07 s, 0.06 s | 42 / 35 MB, 46 / 35 MB |
| `G = complete_graph(2000)`, `H = G` | 0.25 s, 0.00 s | 77 / 77 MB |

### 1.6 Cost

Construction of a wrapped value costs one extra allocation (the `Arc` header). Lookups and assignments of
wrapped values become O(1): 0.18-0.20 s at HEAD to 0.00 s for the form.

Note on the r1 prototype's `chart` arm: HEAD's `parse_args` splits a string literal at its commas (J1a/S9),
so for measurement only the prototype accepts names separated by commas or blanks and checks their number
against dim(M) when there is more than one name (J1a/S14 semantics; one name keeps HEAD's behaviour so the
HEAD tests still run). The implementation uses J1a's arm, not this one.

---

## 2. S12: growth guards for powers, products and strings

### 2.1 The defects at HEAD

The guards bound an argument, not the size of the result.

* `**` and `^` (`apply_binary_op`, `:9674-9677`): `int_arg::<u32>(op, "exponent", b, 0..=OP_POW_EXP)` with
  `OP_POW_EXP = 10000`. `pow(b, e)` (`:1011-1023`) uses `POW_EXP = 1000`. So `2 ** 10001` and
  `pow(2, 1001)` are refused (the results have 10002 and 1002 bits), while `a = 3 ** 10000` then
  `a ** 10000` is admitted: 1.58·10^8 bits, about 25 s for the power and 150 s for the display in
  release (from the timings in 2.3), and `b ** 10000` after that asks for 1.6·10^12 bits. With
  `ulimit -v 3000000`, `a = 2 ** 10000`, `b = a ** 10000` did not finish in 600 s in debug.
* `pow(2, -1)`, `2 ** 10001`, `b ** e` with b = -1 and e = 10^30: all refused, although the results are
  1/2, 2^10001 and 1. (After J1a, `2 ** -1` is 1/2 but `pow(2, -1)` is still refused.)
* `matrix_power` (`:1530-1545`): `int_arg::<u32>(.., 0..=u32::MAX, "")` then `Matrix::pow(u32)`
  (`rustmath-matrix/src/matrix.rs:410-436`). `matrix_power(Matrix([[2]]), 4294967295)` squares until
  memory is gone; `matrix_power(Matrix([[1,1],[0,1]]), 10**30)` is refused although the result is
  [[1, 10^30],[0, 1]] (gp `[1,1;0,1]^(10^30)`). `Matrix::pow` also squares the base once more than needed
  (`base = base*base` after the last bit), so it computes M^(2^(k+1)) for e = 2^k: twice the result size.
  Release, Fibonacci matrix [[1,1],[1,0]]: e = 10^7, 3.8 s; e = 3·10^7, 14.6 s (`$J/s12/powbench.rs`).
* `Matrix * Matrix` (`:9727-9732`): no check. `identity(1000) * identity(1000)` is 10^9 inner steps of
  `Matrix::mul` (`matrix.rs:380-400`, two clones and one multiplication per step): about 35 s in release;
  `identity(600) * identity(600)` took 9.1 s end to end.
* Scalar products (`Matrix * Integer`, `Integer * Matrix` `:9733-9735`, `scalar_mul` `:1622-1636`,
  `Polynomial * Integer` `:9749-9751`): no check. `ones(100, 100) * x` with x of 2^25 bits asks for
  10^4 · 4 MiB = 40 GiB: an allocation failure, which aborts the process.
* `Polynomial * Polynomial` (`:9746-9748`): no check. The library product is schoolbook
  (`rustmath-polynomials/src/univariate.rs:915-933`): `Polynomial([1]*10000)` squared takes 3.06 s
  (release, r1), and `p = p * p` doubles the degree and the coefficient size per line.
* String repetition (`:9852-9858`): `count <= STRING_REPEAT = 10000`, whatever the length. Measured at
  HEAD (`$J/mem/str_head.txt`): `s = "ab" * 10000`; `t = s * 10000` (200 MB, peak 786 MB);
  `u = t * 10000` aborts the process with "memory allocation of 2000000000000 bytes failed". String
  concatenation (`:9860-9862`) has no bound either: `s = s + s` doubles per line.

### 2.2 Fix

New private functions in `repl.rs`, next to `show_int`, and new limits (2.4). Prototype:
`$J/r1/src/repl.rs`.

**Integer and rational powers (merged with J1a).** The size of b^e is known before the power is taken.

```rust
/// The bit length of |b|^k for |b| >= 2, k >= 0: floor(k·log2|b|) + 1, as an f64 (+inf when k does
/// not fit an f64). log2_abs (the existing helper, :10715) reads the top 64 bits, so the relative
/// error is about 1e-16; for a power of two it is exact.
fn int_pow_bits(b: &Integer, k: &Integer) -> f64 {
    (k.to_f64().unwrap_or(f64::INFINITY) * log2_abs(b)).floor() + 1.0
}

/// b^k for k >= 0 (callers pass |e|; `e` is the exponent as written, for the message), refused when
/// the result would have more than POW_BITS bits (S12). |b| <= 1 needs no power, so k may be any
/// non-negative integer there.
fn checked_int_pow(cmd: &str, b: &Integer, k: &Integer, e: &Integer) -> Result<Integer, EvalError> {
    debug_assert!(k.signum() >= 0);
    if b.is_zero() {
        return Ok(if k.is_zero() { Integer::one() } else { Integer::zero() });
    }
    if b.bit_length() == 1 {                       // b = 1 or b = -1
        let odd = k.as_bigint().bit(0);
        return Ok(if b.signum() < 0 && odd { b.clone() } else { Integer::one() });
    }
    let bits = int_pow_bits(b, k);
    if !(bits <= repl_limits::POW_BITS) {
        return Err(EvalError::new("ValueError", format!(
            "{}: b^e for b = {}, e = {} has {}about {:.3e} bits, more than {} (REPL limit)",
            cmd, show_int(b), show_int(e), if e.signum() < 0 { "a denominator of " } else { "" },
            bits, repl_limits::POW_BITS)));
    }
    // |b| >= 2, so k <= bits <= POW_BITS < 2^32.
    let k32 = u32::try_from(k.as_bigint()).expect("k <= POW_BITS");
    Ok(b.pow(k32))
}

/// b^e for an integer b and any integer e (MAGMA ch. 18 `n ^ k`): an Integer for e >= 0; for e < 0
/// the rational 1/b^|e| (J1a's semantics; MAGMA, Sage and PARI give 2^-1 = 1/2), with b^|e| bounded
/// by POW_BITS; 0^e for e < 0 is a ZeroDivisionError ("{cmd}: 0 has no inverse (exponent {e})", J1a's
/// message).
fn int_power_value(cmd: &str, b: &Integer, e: &Integer) -> Result<RustMathValue, EvalError>;

/// r^e for a rational r = n/d and any integer e (MAGMA ch. 20 `a ^ k`): n^|e| / d^|e|, inverted for
/// e < 0. Refused when int_pow_bits of the two powers together exceeds POW_BITS ("{cmd}: r^e for
/// r = n/d, e = .. has about .. bits in its numerator and denominator, more than .. (REPL limit)");
/// 0^e for e < 0 is a ZeroDivisionError.
fn rational_power_value(cmd: &str, r: &Rational, e: &Integer) -> Result<RustMathValue, EvalError>;
```

Arms after J1b:

* `apply_binary_op`: one arm `(Integer(a), "**" | "^", Integer(b)) => int_power_value(op, a, b)` replaces
  J1a's two Integer power arms (the `if b.signum() < 0` arm with `OP_POW_EXP` and the non-negative arm).
  `(Rational(a), "**" | "^", Integer(b)) => rational_power_value(op, a, b)` replaces the arm of J1a's
  revision 1 (it bounds |b| by `OP_POW_EXP`); if J1a's final version has no such arm, J1b adds it.
* `pow(b, e)`: `(Integer, Integer) => int_power_value("pow", ..)`, `(Rational, Integer) =>
  rational_power_value("pow", ..)`; anything else is the TypeError "pow requires an integer or rational
  base and an integer exponent". So `pow` and `**` agree for every input (HEAD's `pow` refused e < 0).

Correctness: for |b| >= 2 the bit length of |b|^k is floor(k·log2|b|) + 1, so the check refuses exactly the
results above POW_BITS bits up to the f64 rounding of k·log2|b| (relative 1e-16; at the boundary
k = 21170489 for b = 3, gp gives `#binary(3^21170489)` = 33554432 and `#binary(3^21170490)` = 33554433, and
the function returns the same). For e < 0 the denominator is b^|e| and the numerator is 1, so the result
has the bits of the denominator (plus one); for a rational base the two powers are checked together.

**Matrix powers.** The REPL computes M^e itself, by left-to-right binary powering (Knuth §4.6.3: square,
then multiply by M when the next bit is 1). Products by M are cheaper than right-to-left's products of two
large powers, and there is no extra final squaring. The loop keeps no exponent counter.

```rust
/// M^e (MAGMA `A ^ n`, ch. 26), e any non-negative integer, with a growth guard (S12).
fn checked_matrix_power(&self, m: &Matrix<Integer>, e: &Integer) -> Result<Matrix<Integer>, EvalError> {
    // non-square: MatrixError "Matrix power is only defined for square matrices" (the library text)
    // e < 0: ValueError "matrix_power: the exponent must be a non-negative integer, got ..
    //        (MAGMA allows n < 0 for an invertible A; the REPL does not compute integer inverses)"
    // e = 0 or d = 0: Matrix::identity(d)
    let (d, ebits) = (m.rows(), e.bit_length());
    // (c) up-front floor: ebits - 1 squarings and popcount(e) - 1 products by M
    let products = (ebits - 1) + (e.as_bigint().magnitude().count_ones() - 1);
    let floor_secs = products as f64 * matrix_product_seconds(d, d, d, 1);
    if floor_secs > repl_limits::PRODUCT_SECONDS {
        return Err(.. "matrix_power: an exponent of {ebits} bits needs {products} products of {d}x{d} \
                       matrices, about {floor_secs:.1} s even if the entries stay small (estimate), \
                       more than 10 s (REPL limit)");
    }
    let log2_d = (d as f64).log2();
    let (mut r, mut spent) = (m.clone(), 0.0);
    for i in (0..ebits - 1).rev() {
        self.checkpoint()?;
        // (a) r = M^t with t = floor(e / 2^(i+1)), so e/t >= 2^(i+1)
        let g = (trace(&r).bit_length() as f64 - 1.0) - log2_d;
        if g > 0.0 {
            let lb = ((i + 1) as f64).exp2() * g - log2_d;          // +inf when 2^(i+1) overflows
            if lb - 1.0 > repl_limits::RESULT_BITS {
                return Err(.. "matrix_power: M^e for e = .. has an entry of more than {lb - 1} bits \
                               (from the trace), more than 33554432 (REPL limit)");
            }
        }
        // (b) size and time check before every product
        spent += check_matrix_product("matrix_power", &r, &r, spent)?;
        r = r.mul(&r)?;
        if e.as_bigint().bit(i) {
            self.checkpoint()?;
            spent += check_matrix_product("matrix_power", &r, m, spent)?;
            r = r.mul(m)?;
        }
    }
    Ok(r)   // invariant: at the end of iteration i, r = M^floor(e / 2^i)
}
```

(a) is a certain lower bound. At the start of iteration i the bits ebits-1 .. i+1 of e have been read, so
r = M^t with t = floor(e/2^(i+1)) >= 1 and e/t >= 2^(i+1). The eigenvalues of r are the λ_j^t, so
\|tr r\| <= d·ρ(M)^t and ρ(M) >= (\|tr r\|/d)^(1/t). For M^e, ρ(M)^e = ρ(M^e) <= \|\|M^e\|\|_∞ <= d·max\|entry\|
(Horn and Johnson Thm. 5.6.9 for the induced ∞-norm). So some entry of M^e has
log2\|entry\| >= (e/t)·(log2\|tr r\| - log2 d) - log2 d. With log2\|tr r\| >= bit_length(tr r) - 1, the
bracket is at least g; when g > 0 the right side is at least 2^(i+1)·g - log2 d = lb. When lb - 1 (one bit
of margin for the f64 rounding) exceeds RESULT_BITS, M^e certainly has more bits than the limit, and the
call is refused before any large product. For tr r = 0 the bit length is 0 and g < 0, so (a) does not
apply. Against the original bound (e/t instead of 2^(i+1)) it is weaker by less than a factor 2, so the
refusal can come one iteration later. For [[2]] and e = 2^32 - 1 it still refuses at i = 30 (the first
iteration, lb = 2^31). (a) does not apply when every tr(M^t) is 0 (nilpotent M, or eigenvalues whose
powers cancel, e.g. the companion matrix of x^3 - 8 at t not divisible by 3); then (b) protects.

(b) is a rigorous size bound and an estimated time bound. With bA, bB the largest bit lengths of the entries,
\|a\| < 2^bA and \|b\| < 2^bB, so an entry of A·B (A: r x k) is below k·2^(bA+bB) <= 2^(bA+bB+bit_length(k)).
The product is refused before it is computed when r·c·(bA + bB + bit_length(k)) exceeds RESULT_BITS, or when
its estimated time, added to the time already estimated for this call, exceeds PRODUCT_SECONDS:

```rust
/// Estimated seconds of Matrix::mul on an (r x k)·(k x c) product whose operands have entries of at
/// most `bits` bits: (100 + r·k·c·(34 + 10·w^1.465)) ns, w = ceil(bits/64) words. The 100 ns cover
/// one product's trace, its two entry scans and its allocation; the step term is the release fit of
/// 2.3 (1.465 = log 5 / log 3, the Toom-3 exponent of num-bigint). Constants MUL_PER_PRODUCT_NS,
/// MUL_STEP_NS, MUL_WORD_NS.
fn matrix_product_seconds(r: usize, k: usize, c: usize, bits: u64) -> f64

/// Refuses A·B when its entries may exceed RESULT_BITS in total, or when `spent` plus its estimated
/// time exceeds PRODUCT_SECONDS. Returns the estimated seconds of the product.
fn check_matrix_product(cmd: &str, a: &Matrix<Integer>, b: &Matrix<Integer>, spent: f64) -> Result<f64, EvalError>
```

(c) is the loop's own cost. Each iteration does O(1) big-integer work besides its products: `bit(i)` indexes
one digit of e, the trace is d additions of entries, the two entry scans are O(d^2); no quantity grows with
the iteration count. The per-product term of `matrix_product_seconds` covers that work (fitted on 1x1 to
32x32 products, 2.3), so the sum of the estimates bounds the loop, and the floor refuses at once when even
the smallest possible estimate per product, `matrix_product_seconds(d, d, d, 1)` (entries of at most one
word), times the number of products exceeds the budget. A refused `matrix_power` therefore stops before
the product that would break a limit, after at most about PRODUCT_SECONDS of estimated work.

Messages of (b): "`{cmd}`: the product of a r x k and a k x c matrix with entries of up to bA and bB bits may
have entries of N bits, T bits in all, more than 33554432 (REPL limit)" and "`{cmd}`: the products would
take about S s (estimate), more than 10 s (REPL limit)". "may" because the bound can be above the true size
(cancellation).

**Other products.** The `Matrix * Matrix` arm calls `check_matrix_product("*", a, b, 0.0)?` before
`a.mul(b)` when `a.cols() == b.rows()` (otherwise the library error stays as it is). `matrix_power` no
longer calls `Matrix::pow`.

```rust
/// Refuses p·q (schoolbook, (m+1)(n+1) coefficient products) when the result may have more than
/// RESULT_BITS bits in all (a coefficient of p·q is a sum of at most min(m, n) + 1 products, so it has
/// at most bp + bq + bits(min(m, n) + 1) bits; there are m + n + 1 of them), or when its estimated time
/// matrix_product_seconds(1, m+1, n+1, max(bp, bq)) · POLY_STEP_FACTOR exceeds PRODUCT_SECONDS.
fn check_poly_product(cmd: &str, p: &UnivariatePolynomial<Integer>, q: &UnivariatePolynomial<Integer>) -> Result<(), EvalError>

/// Refuses k·A for a matrix or polynomial with `count` entries of at most `bits` bits when the
/// result may have more than RESULT_BITS bits in all (an entry of k·a has at most bits(k) + bits(a)
/// bits). One multiplication per entry, so the size bounds the time too.
fn check_scalar_product(cmd: &str, what: &str, count: usize, bits: u64, k: &Integer) -> Result<(), EvalError>
```

`Polynomial * Polynomial` calls `check_poly_product(op, p, q)?`; `Matrix * Integer`, `Integer * Matrix` and
`scalar_mul(M, k)` call `check_scalar_product(cmd, "a matrix", rows·cols, max_entry_bits(m), k)?`;
`Polynomial * Integer` and `Integer * Polynomial` call it with "a polynomial", the coefficient count and
`max_coeff_bits(p)`. Messages: "`*`: a matrix with 10000 entries of up to 1 bits times an integer of
3355444 bits may have 3.355e10 bits in all, more than 33554432 (REPL limit)"; "`*`: the product of
polynomials with 16000 and 16000 coefficients would take about 11.3 s (estimate), more than 10 s (REPL
limit)".

**Strings.** `(String(s), "*", Integer(n))`: the count range becomes `0..=STRING_BYTES / s.len()` (any count
for an empty string), with `why` = "REPL limit: the result has at most 16777216 bytes", so the message reads
"*: repetition count must be an integer in 0..=8388608 (REPL limit: the result has at most 16777216 bytes),
got ..". A negative count still gives `""`. `(String(a), "+", String(b))`: refused when
`a.len() + b.len() > STRING_BYTES` ("+: the concatenation would have N bytes, more than 16777216 (REPL
limit)").

### 2.3 Calibration (release)

Exact callees (`$J/s12/powbench.rs`, `mulfit.rs`; display columns from section 3):

| Integer result | `Integer::pow` (3^e) | num-bigint decimal display (HEAD) | rug decimal display (r1) |
|---|---|---|---|
| 2^20 bits | 0.017 s | 0.06 s | |
| 2^22 bits | 0.15 s | 0.44 s | |
| 2^24 bits | 0.89 s | 3.6 s | |
| 2^25 bits | 2.4 s | 9.3 s | 1.4 s |
| 2^26 bits | 6.4 s | 26 s | (summarised above 5.57·10^7 bits) |
| 2^27 bits | 18.4 s | 76 s (`disp27.txt`) | 8.0 s (review 1) |
| 2^28 bits | 52 s | (about 200 s) | 18.3 s (review 1) |

`Matrix::mul`, ns per inner step (d^3 steps), entries of b bits (w words), large d:

| d, b | 200, 8 | 100, 64 | 100, 256 | 60, 1024 | 40, 4096 | 20, 16384 | 12, 65536 | 6, 262144 | 4, 1048576 | 2, 4194304 |
|---|---|---|---|---|---|---|---|---|---|---|
| ns/step | 33.7 | 45 | 92 | 297 | 2843 | 29204 | 2.5e5 | 1.9e6 | 2.0e7 | 1.3e8 |
| model 34 + 10·w^1.465 | 44 | 44 | 110 | 614 | 4460 | 33860 | 2.6e5 | 2.0e6 | 1.5e7 | 1.2e8 |

The model is within a factor 0.75 to 2.1 of the measurement; it over-estimates for small entries (errs
towards refusing) and under-estimates by up to 25 % at w = 16384.

Per product, small d (revision 1, `matrix_power` of a matrix whose entries do not grow, e = 2^k, release):

| Matrix | d | k (products) | time | ns per product | minus d^3 · 35 ns |
|---|---|---|---|---|---|
| `[[1]]`, `[[-1]]`, `[[0]]` | 1 | 2·10^6 | 0.20, 0.20, 0.18 s | 100, 100, 90 | 65 |
| `[[0,1],[1,0]]` | 2 | 2·10^6 | 0.78 s | 390 | 110 |
| `identity(3)` | 3 | 10^6 | 1.07 s | 1070 | 125 |
| `identity(4)` | 4 | 10^6 | 2.34 s | 2340 | 100 |
| `identity(8)` | 8 | 2·10^5 | 3.84 s | 19200 | 1300 (37.5 ns a step) |
| `identity(16)` | 16 | 3·10^4 | 4.30 s | 143000 | (35 ns a step) |
| `identity(32)` | 32 | 4000 | 4.45 s | 1.11·10^6 | (34 ns a step) |

So one product costs about 100 ns plus 35 ns per inner step for one-word entries: `MUL_PER_PRODUCT_NS` = 100.
The step constant of the model (44 ns at w = 1) over-estimates the measured 34-37 ns.

`Polynomial * Polynomial` (r1, release, all coefficients 1): 2000 terms 0.12 s, 5000 terms 0.79 s, 10000
terms 3.06 s: 30-32 ns per coefficient product against 44 ns in the model, so `POLY_STEP_FACTOR` = 1.0 errs
towards refusing by a factor 1.4 (the protocol refits it, ladder `pmul`).

End to end with the real `ReplContext` (r1 prototype, release, `$J/r1/mpow_r1.txt`, `disp_r1.txt`):

| Command | Result | Time | VmHWM |
|---|---|---|---|
| `x = 3 ** 21170489` (2^25 bits) | admitted | 3.82 s (2.4 s power, 1.4 s display) | 50 MB |
| `3 ** 21170490` | refused | 0.00 s | 4 MB |
| `x = 2 ** 33554431` | admitted | 1.48 s (display) | 37 MB |
| `x = matrix_power(Matrix([[1,1],[1,0]]), 12000000)` | admitted | 2.64 s | 38 MB |
| `matrix_power(Matrix([[1,1],[1,0]]), 12100000)` | refused by (b), size | 0.70 s | |
| `matrix_power(Matrix([[1,1],[1,0]]), 30000000)` | refused by (b), size | 0.82 s | |
| `e = 2 ** 1000000`; `[[1]]`, `[[-1]]`, `[[0,1],[1,0]]` to the e | `[[1]]`, `[[1]]`, `[[1, 0], [0, 1]]` | 0.12, 0.12, 0.43 s | 5 MB |
| `f = 2 ** 33554431 + 1`; `[[1]]`, `[[-1]]` to the f | `[[1]]`, `[[-1]]` | 3.58, 3.83 s | 46 MB |
| `[[0,1],[1,0]]` to the f | refused by (c), floor 15.2 s | 0.00 s | |
| `matrix_power(Matrix([[2]]), 4294967295)` | refused by (a) | 0.00 s | |
| `x = identity(600) * identity(600)` | admitted (estimate 9.50 s; HEAD took 9.1 s end to end) | | |
| `x = "ab" * 8388608` | admitted (16 MiB) | 0.03 s | 69 MB |

The review's prototype run (with the counter t) took 23.7 s for `[[1]]` at `2 ** 1000000`; at
`2 ** 33554431` it would have run for hours.

So POW_BITS = 2^25 meets the X2 policy as a lower bound: every admitted power takes at most about 4 s end
to end. With the rug display, 2^26 bits would still meet it (6.4 s power, the value then summarised), so
the protocol's ladder `pow` may raise POW_BITS to 2^26 (6.3). The constant 34 ns in the model makes
`identity(600)^2` admitted (9.5 s) or refused (10.4 s with 48 ns): the implementer re-fits the constants on
the final build (ladders `mmul`, `mpow`, `pmul`) and keeps the fit table in the doc comment.

### 2.4 API and limits

`mod repl_limits`: remove `POW_EXP`, `OP_POW_EXP` (J1a's arms use it, 2.2) and `STRING_REPEAT`; add

```rust
/// b ** e, b ^ e and pow(b, e) for integer and rational b: the bits of the result (for e < 0, of
/// its denominator). 3^e at 2^25 bits takes 2.4 s for the power and 1.4 s for the display (release).
pub const POW_BITS: f64 = 33_554_432.0;            // 2^25 (ladder `pow`: possibly 2^26)
/// Products of matrices and polynomials, matrix powers and scalar products: a bound on the total
/// bits of the entries of each product.
pub const RESULT_BITS: f64 = 33_554_432.0;         // 2^25
/// matrix_power, Matrix * Matrix and Polynomial * Polynomial: the estimated seconds of the products.
pub const PRODUCT_SECONDS: f64 = 10.0;
/// String repetition and concatenation: the bytes of the result.
pub const STRING_BYTES: usize = 1 << 24;
```

and the private model constants `MUL_PER_PRODUCT_NS = 100.0`, `MUL_STEP_NS = 34.0`, `MUL_WORD_NS = 10.0`,
`POLY_STEP_FACTOR = 1.0` next to `matrix_product_seconds`. The display constants are in section 3.

The `repl_limits` doc comment: replace "The REPL has no timeout or interrupt, and a panic or an
out-of-memory abort kills the kernel, so these limits are its only protection" by the S13 situation
(section 4.2: interrupts and the time limit act between library calls, so the limits still bound single
calls) and remove `matrix_power` from the "do not bound the time" list. Help text: the `pow`, `**` and
`matrix_power` lines state the size limits and that negative exponents give rationals; the limits paragraph
(`:10303-10313` region of `show_help`) gets the new bounds.

### 2.5 Callers and behaviour changes

Only `repl.rs` changes. Every result that was admitted before and has at most 2^25 bits is unchanged.
Changes: `**`/`pow` admit larger exponents with small results, refuse large results with small exponents,
and `pow` with a negative exponent gives the rational (as `**` after J1a); `pow` accepts a rational base;
`matrix_power` accepts any non-negative Integer exponent and refuses by size and time; `Matrix * Matrix`,
`Polynomial * Polynomial` and scalar products refuse results estimated above 10 s or 2^25 bits; string
repetition admits counts above 10000 for short strings and refuses results above 16 MiB; concatenation
refuses results above 16 MiB.

Existing HEAD tests whose expectations change (HEAD line numbers; r1 run: these three tests fail, the
other 70 pass):

| Test | Line | Old expectation | New expectation |
|---|---|---|---|
| `test_int_args_truncation_fixed` | `:11985-11986` | `matrix_power(Matrix([[1, 1], [0, 1]]), 4294967296)` gives "exponent must be an integer in 0..=4294967295" | value `[[1, 4294967296], [0, 1]]` (gp `[1,1;0,1]^4294967296`) |
| same | `:11987-11990` | the `falling_factorial` / `rising_factorial` / `partitions(417)` rows | section 5.4 |
| `test_int_args_no_panic` | `:12006` | `pow(2, 2^63)`: "pow: exponent must be an integer in 0..=1000" | "pow: b^e for b = 2, e = 9223372036854775808 has about 9.223e18 bits, more than 33554432 (REPL limit)" |
| same | `:12007` | `2 ** 2^64`: "**: exponent must be an integer in 0..=10000" | "**: b^e for b = 2, e = 18446744073709551616 has about" |
| same | `:12011` | `matrix_power(Matrix([[1, 1], [0, 1]]), 2^63)` refused | value `[[1, 9223372036854775808], [0, 1]]` (gp `[1,1;0,1]^(2^63)`) |
| same | `:12036` | `"ab" * c` (c = 2^1100): "repetition count must be an integer in 0..=10000" | "*: repetition count must be an integer in 0..=8388608 (REPL limit: the result has at most 16777216 bytes)" |
| `test_int_arg_ranges_named` | `:12177` | `pow(2, -1)`: "pow: exponent must be an integer in 0..=1000" | moves to the value tests: `pow(2, -1)` is `1/2` (gp `2^-1`) |
| same | `:12178` | `2 ** 10001` refused | value: 3011 digits ending `63548609585193418752` (gp `#digits(2^10001)`, `2^10001 % 10^20`) |

`:11884` (`b = t ** 420`, 2^4200000) stays admitted. After J1a, any J1a test that formats `OP_POW_EXP` in an
expected message (none in J1a's design; check with the grep of plan step 0) is rewritten to the POW_BITS
message.

### 2.6 Tests (new; the r1 prototype versions `j1b_int_pow_guard`, `j1b_r1_powers_scalar_divmod`, `j1b_matrix_power_checkpoint_and_guards` and `j1b_r1_matrix_power_loop` pass in debug)

`test_int_pow_growth_guard`:

| Input | Expected | Reference | HEAD |
|---|---|---|---|
| `2 ** 10001` | 3011 digits, ends `63548609585193418752` | gp `#digits(2^10001)`, `2^10001 % 10^20` | refused |
| `pow(2, 1001)` | ends `48773674411336138752`, 302 digits | gp | refused |
| `int_pow_bits(3, 21170489)`, `(3, 21170490)`, `(2, 33554431)` | 33554432, 33554433, 33554432 | gp `#binary(3^21170489)` = 33554432, `#binary(3^21170490)` = 33554433 | n/a (new fn) |
| `3 ** 21170490` | ValueError containing "more than 33554432" | | computes (minutes in debug) |
| `2 ** 33554432` | ValueError | `#binary(2^33554432)` = 33554433 | refused (other message) |
| `(-1) ** (10 ** 30)`, `(-1) ** (10 ** 30 + 1)` | `1`, `-1` | gp `(-1)^(10^30)` | refused (and "Cannot parse" before J1a) |
| `0 ** 0`, `0 ** (10 ** 30)`, `1 ** (10 ** 30)` | `1`, `0`, `1` | gp `0^0` = 1 | refused |
| `a = 3 ** 10000`, `a ** 10000` | ValueError | 1.58·10^8 bits (gp `10000*#binary(3^10000)` = 158500000) | runs for minutes |

`test_negative_and_rational_powers` (merged semantics, 2.2):

| Input | Expected | Reference | HEAD |
|---|---|---|---|
| `2 ** -1`, `pow(2, -1)` | `1/2`, `1/2` | gp `2^-1` | refused, refused (after J1a: `1/2`, refused) |
| `pow(-2, -3)`, `(-2) ** -3` | `-1/8` | gp `(-2)^-3` | refused |
| `pow(2, -100)` | `1/1267650600228229401496703205376` | gp `2^-100` | refused |
| `(-1) ** -(10 ** 30 + 1)` | `-1` | gp `(-1)^-(10^30+1)` | refused |
| `0 ** -1`, `pow(0, -1)` | ZeroDivisionError | gp `0^-1` error | refused |
| `3 ** -21170490` | ValueError "has a denominator of about 3.355e7 bits, more than 33554432" | | refused (other message) |
| `pow(Rational(2, 3), -2)`, `(2/3) ** 3` | `9/4`, `8/27` | gp `(2/3)^-2`, `(2/3)^3` | TypeError, "Cannot parse" |
| `pow(Rational(2, 3), 21170489)` | ValueError "more than 33554432" (21170490 + 33554432 bits; gp `21170490 + ceil(21170489*log(3)/log(2))` = 54724922) | | TypeError |

`test_matrix_power_growth_guard`:

| Input | Expected | Reference | HEAD |
|---|---|---|---|
| `matrix_power(Matrix([[1,1],[0,1]]), 10 ** 9)` | `[[1, 1000000000], [0, 1]]` | gp | same |
| `matrix_power(Matrix([[1,1],[0,1]]), 10 ** 30)` | `[[1, 1000000000000000000000000000000], [0, 1]]` | gp `[1,1;0,1]^(10^30)` | refused |
| `matrix_power(Matrix([[2,-1,0],[1,3,1],[0,1,-2]]), 7)` | `[[-1215, -1171, -259], [1171, 215, 135], [259, 135, -201]]` | gp `[2,-1,0;1,3,1;0,1,-2]^7` | same |
| `matrix_power(Matrix([[1,1],[1,0]]), 1000)` | entry (0,1) has 209 digits and ends `76137795166849228875`; (0,0) ends `91902245245323403501`; (1,1) ends `15764450078474174626` | gp `([1,1;1,0]^1000)[1,2] % 10^20` etc. | same |
| `matrix_power(Matrix([[0,0,8],[1,0,0],[0,1,0]]), 30)` | `8^10` times the identity (`1073741824` on the diagonal) | gp `[0,0,8;1,0,0;0,1,0]^30 == 8^10*matid(3)` | same |
| `matrix_power(Matrix([[2]]), 4294967295)` | ValueError "from the trace", in under 1 s | | never ends |
| `matrix_power(Matrix([[0,0,8],[1,0,0],[0,1,0]]), 10 ** 9)` | ValueError "from the trace" | | refused (u32) |
| `matrix_power(Matrix([[3,1],[1,2]]), 20)` | `[[106894531250, 66064453125], [66064453125, 40830078125]]` | gp `[3,1;1,2]^20` | same |
| `e = 2 ** 1000000`, `matrix_power(Matrix([[1]]), e)` | `[[1]]`, in a few seconds in debug | gp `Mat(1)^(2^1000000)` | refused (u32) |
| `f = 2 ** 1000001 + 1`, `matrix_power(Matrix([[0,1],[1,0]]), f)` | `[[0, 1], [1, 0]]` (both together 2.1 s in debug, r1) | gp `[0,1;1,0]^(2^1000001+1)` | refused (u32) |
| `g = 2 ** 33554431`, `matrix_power(Matrix([[0,1],[1,0]]), g)` | ValueError "even if the entries stay small", in under 1 s | | refused (u32) |
| `identity(1000) * identity(1000)`; `matrix_power(identity(1000), 2)` | ValueError "s (estimate)" (44 s); ValueError "even if the entries stay small" (the floor: one product, 44 s) | | about 35 s release, many minutes debug |
| `matrix_power(Matrix([[1,2],[3,4]]), 0)`, `(.., -1)`, non-square | identity; ValueError "non-negative"; MatrixError | | `-1` has another message |
| property: `checked_matrix_power(&m, e) == m.pow(e)` for m = [[1,1],[1,0]] and three 3x3 matrices with entries in -3..3, e in 0..40 | equal | the library function | n/a |
| `matrix_product_seconds(600, 600, 600, 64)` = (100 + 2.16·10^8 · 44) ns = 9.50 s with the provisional constants (update with the refit); increasing in d and in bits; `check_matrix_product` refuses r·c·(bA + bB + bit_length(k)) > RESULT_BITS (2x2 with entries of 2^23 bits: 4·(2^24 + 2) > 2^25) | | | n/a |

`test_product_size_guards`: `x = 2 ** 3355443`: `ones(100, 100) * x` and `scalar_mul(ones(100, 100), x)`
are ValueErrors containing "may have" (HEAD: 4.2 GB, an abort); `Matrix([[1]]) * x` is admitted;
`y = 2 ** 20000000`, `P = Polynomial([y, y])`, `P * P` is a ValueError containing "may have";
`Polynomial([1, 1]) * Polynomial([1, 1])` is `x^2 + 2*x + 1`; `check_poly_product` of two 15000-term
polynomials with unit coefficients refuses by time (estimate 9.9 s with `POLY_STEP_FACTOR` 1.0 is just
under; use 16000 terms: 11.3 s) without building them in the REPL (unit test on the function).

`test_string_growth_guard`: `x = "ab" * 10001` admitted (HEAD refused); `"a" * 16777216` admitted,
`"a" * 16777217` refused; `x = "ab" * 10000`, `x * 838` admitted, `x * 839` refused with the new message;
`"" * c` for c = 2^1100 is `""`; a negative count is `""` (unchanged); `s = "ab" * 8388608`, `s + "c"` is a
ValueError "more than 16777216" (HEAD: admitted).

### 2.7 Cost

`int_pow_bits` is O(1). `checked_matrix_power` adds one trace (d additions) and one scan of the entry bit
lengths per product (O(d^2), against O(d^3) for the product), and O(bits(e)) for the popcount of the floor.
For the Fibonacci matrix at e = 1.2·10^7 it is faster than `Matrix::pow` (no final squaring, products by M
are small). `check_poly_product` and `check_scalar_product` scan the coefficients or entries once.

---

## 3. D (found): display of large values

### 3.1 The defect at HEAD

`to_display` prints every Integer in full (`RustMathValue::Integer(n) => n.to_string()`, `:300-303`; also
`Rational`, the entries of `Matrix` twice, once for text and once for LaTeX, `:330-356`, and the polynomial
coefficients twice). num-bigint 0.4.8 converts to decimal by divide and conquer with schoolbook division
(`biguint/convert.rs:677-800`), which is quadratic (Brent and Zimmermann §1.7: a subquadratic conversion
needs fast division). Measured (release, 2.3): 9.3 s at 2^25 bits, 26 s at 2^26, 76 s at 2^27. The X2 limits
did not include the display: `SIGMA_BITS = 2^28` admits `sigma(2, 268435456)`, whose end-to-end time is
**207 s** in release (`$J/mem/sigma_head_rel.txt`; the sum itself takes about 1.7 s). `x = sigma(2,
55000000)` takes 18.5 s (`$J/r1/replmem_head_rel`).

The display is also unbounded in size. A symbolic expression is a DAG (`Binary(op, Arc<Expr>, Arc<Expr>)`),
and its Display walks it as a tree. `x = var("x")`, `e = x + x`, then `e = e + e` repeated: each line
doubles the printed text; at 23 doublings the line takes 1.57 s and 135 MB (HEAD, release), at 30 it would
take about 3 minutes and 17 GB (an allocation failure). Lists print every element: `[a, a, a, a]` with a
2^25-bit `a` costs four full conversions.

### 3.2 Fix

**Faster conversion.** One helper converts every integer the REPL formats itself:

```rust
/// The decimal text of n, the same text as n.to_string(). num-bigint 0.4.8 converts by divide and
/// conquer with schoolbook division, which is quadratic (9.3 s at 2^25 bits, release); above 2^15
/// bits the conversion goes through GMP's mpz_get_str (rug), which is subquadratic (Brent and
/// Zimmermann, Modern Computer Arithmetic, 1.7): 1.4 s at 2^25 bits.
fn decimal_string(n: &Integer) -> String {
    if n.bit_length() <= 1 << 15 { return n.to_string(); }
    let mag = rug::Integer::from_digits(&n.as_bigint().magnitude().to_u64_digits(), rug::integer::Order::Lsf);
    if n.signum() < 0 { (-mag).to_string() } else { mag.to_string() }
}
```

`Cargo.toml` of rustmath-jupyter gains `rug = { version = "1.24", default-features = false, features =
["integer"] }` (the declaration of `rustmath-rationals/Cargo.toml:16`, with only the feature used; rug 1.30
is already in `Cargo.lock` and already linked into rustmath-jupyter through rustmath-complex,
rustmath-matrix and rustmath-rationals, so nothing new is built). Converting the digits costs 2 ms at 2^25
bits (review 1). A faster `Display for Integer` in rustmath-integers would serve every crate; that is I3's
crate and is listed in section 7.

**One budget per output.** Before formatting, `to_display` bounds the size of the text:

```rust
/// The text of one REPL output, in bytes: a value whose text (estimated before formatting) would be
/// larger is shown as a one-line summary; the value itself is kept and can be used.
pub const OUTPUT_BYTES: u64 = 1 << 24;                     // 16 MiB
/// Symbolic expressions are printed by rustmath-symbolic, whose integers convert with num-bigint
/// (0.44 s at 2^22 bits, 3.6 s at 2^24, release): an expression with a larger integer is summarised.
pub const LIBRARY_INT_DISPLAY_BITS: u64 = 1 << 22;

/// An upper bound on the bytes of the text form of a value, and the largest Integer a library
/// formatter would print in it.
struct DisplaySize { bytes: u64, library_int_bits: u64 }

/// display_size of a value: computed without formatting, iteratively (no recursion), and stopped once
/// `bytes` passes `limit`, so it costs O(limit) at most whatever the DAG.
fn display_size(v: &RustMathValue, limit: u64) -> DisplaySize;
/// The one-line summary of a value that to_display does not print.
fn display_summary(v: &RustMathValue, size: &DisplaySize) -> String;
```

`to_display(&self)` computes `display_size(self, OUTPUT_BYTES)`; above `OUTPUT_BYTES`, or with an integer in
a symbolic expression above `LIBRARY_INT_DISPLAY_BITS`, it returns `display_summary`; otherwise
`to_display_unchecked` formats as before (a list formats its elements with `to_display_unchecked`, since
the whole list was measured). Per-node bounds of `display_size`: an Integer floor(bits·log10 2) + 2 bytes
(digits and sign: an upper bound), a Rational the two plus 1, an IntegerMod the two plus 8, a String its
length plus 2, a matrix the entries plus 2 per entry and per row, a polynomial the coefficients plus 24 per
term (" + ", "*x^", the exponent), a list the elements plus 2 per element, a chart the names plus 2 each, a
symbolic expression: Integer and Rational leaves as above (and their bits into `library_int_bits`), `Real`
330 bytes (the longest f64 Display is about 320 characters), a Symbol its name, `Binary` 5, `Unary` 16,
`Function` its name plus 2 per argument; every other variant 256 bytes (fixed-size texts: Bool, Float,
Complex, points, graphs, manifolds, the algebra headers, the Plot description). The LaTeX form has the same
leaves with a constant factor more markup.

Summaries: `<integer of N bits, about D decimal digits: not printed (REPL display limit 16777216 bytes)>`
(with "negative" for a negative value); for the other values `<{what}: its text has more than 16777216 bytes,
not printed (REPL display limit)>` with `what` one of "list of k elements", "r x c matrix with entries of
up to B bits", "polynomial with k coefficients of up to B bits", "rational number with a numerator of N bits
and a denominator of M bits", "integer modulo an integer of N bits", "string of N bytes", "chart with N
coordinates", "symbolic expression"; for the library cap `<symbolic expression: it contains an integer of
N bits, more than 4194304 (REPL display limit for symbolic expressions)>`. Not a facade: each summary says
that the value is not printed, and gives its size.

**Formatting each integer once.** `Integer`, `Rational` (text `n` or `n/d`, as rustmath-rationals'
Display, `rational.rs:222-230`; LaTeX `\frac{n}{d}`), `IntegerMod`, matrix entries, polynomial coefficients
and the leaves of `expr_to_latex` (the REPL's own LaTeX of an expression) go through `decimal_string`. The
`Matrix` arm builds text and LaTeX in one pass (each entry string is appended to both and dropped, so no
table of entry strings is kept: `A = identity(1000)` peaks at 42 MB, HEAD 118 MB). The `Polynomial` arm
converts each coefficient once and builds the text with `poly_text(coeffs, strs)`, which reproduces
rustmath-polynomials' Display (`univariate.rs:709-743`: terms from the highest degree, " + ", `c*x^i`, `x`
for a unit coefficient, `0` for the zero polynomial); a unit test checks `poly_text` against `p.to_string()`.
The text of a symbolic expression stays rustmath-symbolic's Display (library-formatted, hence the
`LIBRARY_INT_DISPLAY_BITS` cap).

Time bound: rug's conversion is superadditive in the bits, so the REPL-formatted integers of one output cost
at most the conversion of one integer with all their bits, at most about 5.57·10^7 bits under the budget,
about 3 s (`sigma(2, 55700000)` 2.93 s end to end). The library-formatted integers of an expression cost at
most (bits / 2^22) conversions of 2^22 bits, 13 × 0.44 s = 5.8 s in the worst case. `print(a, b, ..)` and
`vars()` apply the budget to each value (section 7).

Measured (r1 prototype, release): `sigma(2, 268435456)` 207 s to 1.76 s (VmHWM 446 MB, the value itself;
X2 measured 445 MB); against the original prototype's num-bigint display (HEAD refuses these powers):
`x = 2 ** 33554431` 10.1 s to 1.48 s, `x = 3 ** 21170489` 12.5 s to 3.82 s; against HEAD:
`sigma(2, 55000000)` 18.5 s to 3.02 s; the `e = e + e` DAG after 40 doublings is summarised in under 2 s in
debug (HEAD: about 2^17 times the 1.6 s of 23 doublings, and an allocation failure long before).

### 3.3 Tests (r1 prototype `j1b_display_guard`, `j1b_r1_output_budget` pass in debug)

* `test_integer_display_limit`: `RustMathValue::Integer(1 << 67108864).to_display()` starts with
  `<integer of 67108865 bits`; the negative value starts with `<negative integer`; a 100-bit value prints in
  full. REPL: `sigma(2, 67108864)` displays the summary (gp `#binary(sigma(2, 2^26))` = 67108865; 2^26 bits
  are about 2.02·10^7 digits, more than OUTPUT_BYTES; fails on HEAD by taking minutes in debug).
  `decimal_string(a) == a.to_string()` for a = ±(3·2^k - 12345), k in {32767, 32768, 32769, 40000, 100003}
  (both sides of the 2^15-bit switch).
* `test_output_budget`: `x = var("x")`, `e = x + x`, 40 times `e = e + e`, then `e`: the summary
  `<symbolic expression: its text has more than 16777216 bytes` in under 2 s (HEAD: does not end);
  `a = 2 ** 16777215`, `[a, a, a, a]` (four times 5.05·10^6 digits): `<list of 4 elements: its text has more
  than` (HEAD prints 20 MB); `Matrix([[1,2],[3,4]])` text `[[1, 2], [3, 4]]` and LaTeX
  `$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$` (unchanged); `poly_text` equals `p.to_string()` for
  [0], [1], [0, 1], [3, 0, 1], [-5, 2, 0, 7], [1, 1, 1, -1]; `Rational(6, 4)` displays `3/2` with LaTeX
  `$\frac{3}{2}$` (unchanged); `y = 2 ** 4194304`, `x * y` (an expression with a 4194305-bit leaf) is
  summarised with "contains an integer of 4194305 bits", while `x * (2 ** 4194303)` prints (0.49 s in
  release with the library's num-bigint conversion; r1).

---

## 4. S13: a real interrupt, a time limit, and a kernel that answers while it computes

### 4.1 The defects at HEAD

`kernel.rs:43-101` polls shell, control and heartbeat in turn with 100 ms timeouts, and
`handle_shell_message` awaits `handle_execute` (`handlers.rs:53-155`), which calls `ctx.eval` (`:81`)
synchronously on the runtime's main thread. While a cell runs, nothing else is read. `install.rs:37-51`
writes no `interrupt_mode`, so Jupyter uses `signal` and sends SIGINT, whose default action ends the
process. Probe (`$J/s13/client.rs`, the HEAD kernel binary, a cell of 300 lines `x = factorial(10000)`,
about 28 s; `run1.txt`, `run_final_head.txt`):

| Request during the cell | HEAD |
|---|---|
| heartbeat ping | not echoed within 5 s |
| `interrupt_request` (control) | no reply within 10 s; the cell runs to its end (27.9 s), status ok |
| SIGINT | the kernel process dies |
| SIGINT while idle | the kernel process dies |
| three requests queued, interrupt the first | no interrupt; all three run, status ok |
| `factor(<200-digit semiprime>)`, then interrupt and `shutdown_request` | no reply to either within 10 s; the process is still alive 10 s later |
| `divmod(Polynomial([1]), Polynomial([0]))` | panic in `rustmath-polynomials/src/univariate.rs:553`, the kernel dies, no reply |

The REPL's own long loops have no cancellation point either. Measured at HEAD (release): with
`e = x + x` doubled 17 times (2^18 leaves as a tree), `plot(e, (x, 0, 1))` takes 8.9 s and
`plot3d(e, (x, 0, 1), (y, 0, 1))` 47.7 s; each further `e = e + e` doubles both.

### 4.2 Design

**What stops and what does not.** Rust cannot stop a thread from outside, and the REPL calls into 16
crates that do not check any flag. So:

* An interrupt or the time limit stops a cell at its next **checkpoint**: before each line, at the start
  of every `eval_expr` (that is, between the sub-expressions and arguments of a line), after the
  right-hand side of an assignment, after its display and before it is stored (so an interrupt that arrives
  before the store discards the line, as a Python `KeyboardInterrupt`; one that arrives after the store
  lets the line complete and stops the cell at the next line), the same for an expression statement,
  between the products of `matrix_power`, and once per sample in the sampling loops of the 13 plot
  functions (`plot_function_internal`, `eval_slope_field`, `eval_parametric_plot`, `eval_contour_plot`,
  `eval_vector_field`, `eval_implicit_plot`, `eval_region_plot`, `eval_plot3d`, `eval_parametric_plot3d`,
  `eval_implicit_plot3d`, `eval_revolution_plot3d`, `eval_spherical_plot3d`, `eval_cylindrical_plot3d`;
  per point of a curve, per grid row of a surface or grid). The error is `KeyboardInterrupt: interrupted`
  or `TimeoutError: the evaluation reached its time limit of S s (set with time_limit)`.
* A **single library call** (`factor`, `nth_prime`, `det`, `simplify`, `taylor`, one big multiplication,
  one plot sample of a huge expression tree, a display) runs to completion. The kernel stays responsive
  meanwhile (heartbeat, control); after `INTERRUPT_NOTICE_AFTER` = 2 s it publishes on stderr: "Interrupt
  requested 2 s ago. The running command is inside a library call that does not check for interrupts; it
  stops when that call returns. Restart the kernel to stop it now (all variables are lost)." When the call
  returns, the next checkpoint raises the interrupt and the result is discarded.
* The **hard stop** is the process exit: `shutdown_request` (Jupyter's restart sends it with
  `restart: true`) is answered at once, `run()` returns, and the process ends with the evaluation thread.
  This mirrors MAGMA ch. 1 §1.2: `<Ctrl>-C` "interrupts at a convenient point", and `<Ctrl>-\` (or a
  second `<Ctrl>-C` within half a second) quits. The kernel does not exit on a second interrupt, because
  Jupyter users expect the restart button for that.
* The REPL's other loops (`ones`/`zeros`/`identity` up to 10^6 entries, histogram binning up to 1000 bins,
  the list and matrix literal parsers, which are bounded by the input) do not need checkpoints: their cost
  is bounded by a limit or by the length of the line.

**`ReplContext` (repl.rs).**

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Interrupts the evaluation running in a ReplContext from another thread (S13). The evaluation
/// stops at its next checkpoint with a KeyboardInterrupt. A request made while nothing runs applies
/// to the next evaluation, unless `clear` discards it.
#[derive(Clone, Debug, Default)]
pub struct InterruptHandle(Arc<AtomicBool>);
impl InterruptHandle {
    pub fn interrupt(&self);      // store(true, SeqCst)
    pub fn clear(&self);          // store(false, SeqCst)
    pub fn is_pending(&self) -> bool;
}

// new fields of ReplContext
interrupt: InterruptHandle,
time_limit: Option<Duration>,   // per eval call; None: no limit (the default)
deadline: Option<Instant>,      // of the running eval call

pub fn interrupt_handle(&self) -> InterruptHandle;
pub fn set_time_limit(&mut self, limit: Option<Duration>);
pub fn time_limit(&self) -> Option<Duration>;

/// A cancellation point: Err while an interrupt is pending or after the deadline. The flag is
/// sticky: it is not cleared here, so a caller that swallows the error fails again at the next
/// checkpoint; `eval` clears it when the cell ends.
fn checkpoint(&self) -> Result<(), EvalError>;

pub fn eval(&mut self, code: &str) -> Result<EvalResult, EvalError> {
    self.deadline = self.time_limit.map(|d| Instant::now() + d);
    let result = self.eval_cell(code);        // the HEAD body of eval, plus a checkpoint per line
    self.deadline = None;
    self.interrupt.clear();                   // it stopped this cell, or came after its last checkpoint
    result
}
```

The flag is **sticky** during an evaluation. Three places swallow evaluation errors (`parse_symbolic_arg`
`if let Ok(result) = self.eval_expr(s)`, `:4928`; the `eval_simple_arithmetic` fallbacks, `:5168`, `:5199`;
J1a may rewrite the last two); a consuming checkpoint there would lose the interrupt, a sticky one fails
again at the next checkpoint. The checkpoint costs one atomic load, plus `Instant::now()` (about 25 ns) when
a time limit is set; `eval_expr` does string work of microseconds per call, and a plot sample costs at least
microseconds.

New REPL command `time_limit(s)`: `s` a non-negative Integer or Float (seconds, at most one year);
0 removes the limit; `time_limit()` shows it. It applies to later cells (the running cell keeps its
deadline). The output is a String: `"time limit: 60 s for each later cell"` or `"no time limit"`. It is
the cooperative analogue of MAGMA's `Alarm(s)` (ch. 3 §3.12), which ends the process. Help text: a line for
`time_limit` and a sentence on interrupts (what stops, what does not).

**`divmod` (repl.rs).** The arm calls `p.div_rem(q)` (`univariate.rs`, a `Result`) instead of `quo_rem`
(which panics): `MathError::DivisionByZero` becomes `ZeroDivisionError: divmod: division by the zero
polynomial`, any other error `MathError: divmod: {e}`.

**Kernel (kernel.rs, handlers.rs, protocol.rs, install.rs).** Prototype: `$J/proto/src/kernel.rs`,
`handlers.rs`.

* `EvalWorker`: a thread named `rustmath-eval` with `EVAL_STACK_BYTES = 64 << 20` (8 times the 8 MiB of a
  main thread; only touched pages are committed). It owns the evaluator and loops on a
  `std::sync::mpsc::Receiver<EvalJob>`; `EvalJob { code: String, reply: tokio::sync::oneshot::Sender<EvalOutcome> }`;
  `pub struct EvalOutcome { pub result: Result<EvalResult, EvalError>, pub stdout: String, pub stderr: String }`
  (in handlers.rs). The kernel keeps the `InterruptHandle`. The loop is generic over a private trait so that
  the tests can drive it with a stub:

  ```rust
  /// What the evaluation thread runs: ReplContext in the kernel, a stub in the tests.
  trait Evaluator: Send + 'static {
      fn eval(&mut self, code: &str) -> Result<EvalResult, EvalError>;
      fn output(&self) -> (String, String);       // stdout and stderr of the last eval
  }
  impl Evaluator for ReplContext { .. }
  impl EvalWorker {
      fn spawn() -> std::io::Result<Self>;                                  // spawn_with(ReplContext::new(), ..)
      fn spawn_with<E: Evaluator>(ev: E, interrupt: InterruptHandle) -> std::io::Result<Self>;
  }
  ```

  The loop body is `let result = eval_guarded(|| ev.eval(&job.code)); let (stdout, stderr) = ev.output();`
  then the reply.
* `pub fn eval_guarded(f) -> Result<EvalResult, EvalError>`: `catch_unwind(AssertUnwindSafe(f))`; a panic
  becomes `InternalError: a library call panicked: <message>. The variables defined before this cell are
  kept.` `AssertUnwindSafe` is sound here: the REPL stores a variable only after its value is computed,
  `eval_cell` clears the output buffers at its start, and `eval` sets the deadline at its start; the
  kernel clears the interrupt flag before each job. What a panicking library leaves behind (a poisoned
  mutex, a half-updated cache inside that library) is not repaired; the message does not claim otherwise.
  The workspace profiles do not set `panic = "abort"` (checked).
* The execution count moves to the kernel (`RustMathKernel::execution_count`), because `execute_input` must
  carry it before the job starts. `ReplContext::execution_count` and `increment_count` stay (public API),
  unused by the kernel.
* `handle_execute` splits into `begin_execute(msg, count, sockets) -> ExecuteRequest` (busy,
  `execute_input`) and `finish_execute(msg, &request, count, outcome, sockets) -> JupyterMessage` (stdout,
  stderr, `execute_result` or `error`, idle, the reply). New `aborted_reply(msg, count)` (`{"status":
  "aborted", "execution_count": n}`) and `publish_stream(sockets, parent, name, text)`.
  `handle_complete(msg)` loses its unused `&ReplContext` parameter.
* The main loop is one `tokio::select!` over: `sockets.shell.recv()` (only while idle; a busy kernel leaves
  shell requests queued, which keeps the protocol's sequential semantics), `sockets.control.recv()`,
  `sockets.heartbeat.recv()` (echo), the job's oneshot (while busy), SIGINT
  (`tokio::signal::unix::signal(SignalKind::interrupt())`, `#[cfg(unix)]`; tokio's `full` feature already
  includes `signal`), and the 2 s notice timer (after an interrupt, once). zeromq 0.4.1's `recv` is
  `fair_queue.next().await` (cancel safe), so dropping the losing branches loses no message; HEAD's
  timeouts already relied on that.
* `interrupt_request` (control) and SIGINT: if a job runs, `interrupt.interrupt()` and remember the time;
  the control reply is sent at once either way. While idle they do nothing (an idle SIGINT no longer ends
  the process).
* Before each job the kernel calls `interrupt.clear()`, then sends the job. Ordering argument: interrupts
  are requested only on the kernel's own task (control and SIGINT branches of the same loop), so no
  request can fall between the clear and the send; a request after the send is seen by the job's
  checkpoints; a request after the job's last checkpoint is cleared by `eval` or by the next dispatch.
* `stop_on_error`: when a job fails and the request has `stop_on_error` and is not `silent`, the kernel
  first answers every `execute_request` already queued on the shell socket with `aborted` (polled with
  `tokio::time::timeout(Duration::ZERO, ..)`; other shell messages are handled normally), and only then
  publishes the failed cell's outputs, idle and reply. So a request sent after the client saw the error
  is never aborted (the prototype first aborted after the reply, and a fast client's next cell was
  aborted: `run2.txt`). This is ipykernel's behaviour with its default `stop_on_error_timeout = 0`.
  `ExecuteRequest::stop_on_error` gets `#[serde(default = "default_true")]` (the protocol default; HEAD
  defaulted to false).
* `shutdown_request` (control): reply, `running = false`; `run()` returns and `main` returns, which ends
  the process and the evaluation thread. `kernel_info_request` on control: reply only (no busy/idle
  status, which would make a front end show an idle kernel during a cell).
* If the worker thread is gone (only a stack overflow, which aborts the process anyway, or a bug), the
  job's oneshot errs: the kernel replies `InternalError: the evaluation thread stopped` and exits `run()`,
  so that Jupyter restarts it.
* `install.rs`: kernel.json gains `"interrupt_mode": "message"`, so Jupyter sends `interrupt_request`.
  Kernels installed before keep `signal`; SIGINT is handled the same way.

### 4.3 Measured with the prototype (`$J/s13/run_final.txt`)

| Scenario | Prototype |
|---|---|
| heartbeat during the cell | echoed in 1 ms |
| `interrupt_request` during the cell | reply in 1 ms; `execute_reply` error `KeyboardInterrupt` 45 ms later; then `a` (set before) still gives 5 |
| SIGINT during the cell | `KeyboardInterrupt` 55 ms later; the process is alive |
| SIGINT while idle | alive; the next cell runs |
| three queued requests, interrupt the first | `aborted`, `aborted`, then `error KeyboardInterrupt` (the order of section 4.2) |
| `factor(<200-digit semiprime>)`, interrupt, wait, shutdown | interrupt reply 1 ms; stderr notice at 2.0 s; shutdown reply 1 ms; process gone 50 ms later |
| `divmod(Polynomial([1]), Polynomial([0]))`, then `1+2` (original prototype, before the divmod fix) | `InternalError` in 0.1 s; `1+2` gives 3 |

### 4.4 Callers and behaviour changes

Public API of the crate: `RustMathKernel` keeps `from_connection_file` and `run`; `handlers::handle_execute`
is replaced by `begin_execute`/`finish_execute`, `handle_complete` loses a parameter; new
`repl::InterruptHandle`, `ReplContext::{interrupt_handle, set_time_limit, time_limit}`,
`kernel::{eval_guarded, EVAL_STACK_BYTES, INTERRUPT_NOTICE_AFTER}`, `handlers::{EvalOutcome,
aborted_reply, publish_stream}`. No other crate uses `rustmath_jupyter` (workspace grep). User-visible:
queued cells are aborted after an error (as in every ipykernel front end); a panic is an `InternalError`
instead of a dead kernel; `divmod` by the zero polynomial is a ZeroDivisionError; SIGINT interrupts instead
of killing; the REPL runs on a 64 MiB stack instead of the 8 MiB main thread.

### 4.5 Tests

Unit tests in `repl.rs` (the prototype versions of the first four pass in debug; the divmod case is in the r1 test
`j1b_r1_powers_scalar_divmod`; the plot checkpoint is not prototyped):

* `test_pending_interrupt_stops_next_eval`: `h = ctx.interrupt_handle(); h.interrupt();`
  `ctx.eval("a = 1")` is `KeyboardInterrupt`; `a` is not in `ctx.variables`; `h.is_pending()` is false;
  `ctx.eval("a = 1")` then gives `1`. Fails on HEAD (no such API).
* `test_interrupt_from_another_thread`: a cell `first = 1`, 5000 lines `x = factorial(3000)`, `last = 2`;
  another thread interrupts after 300 ms; `KeyboardInterrupt` within 5 s; `first` is set, `last` is not.
* `test_time_limit`: `set_time_limit(Some(Duration::ZERO))`: `eval("1 + 1")` is `TimeoutError`;
  `None`: `2`. `time_limit(1)` reports "time limit: 1 s"; a cell of 10^5 factorial lines stops with
  `TimeoutError` within 4 s; `time_limit(0)` reports "no time limit". `time_limit(-1)`, `time_limit("x")`:
  ValueError.
* `test_matrix_power_checkpoint`: with a pending interrupt, `ctx.checked_matrix_power(&m, &1000)` is
  `KeyboardInterrupt` (the loop's own checkpoint).
* `test_plot_checkpoint`: `x = var("x")`, `e = x + x`, 12 times `e = e + e`; then from another thread
  interrupt after 200 ms while `plot3d(e, (x, 0, 1), (y, 0, 1))` runs (about 1.5 s in release at 2^13
  leaves, several times that in debug): `KeyboardInterrupt` within 2 s of the request. Fails on HEAD (no
  API; HEAD runs the plot to its end).
* `test_divmod_by_zero_polynomial`: `divmod(Polynomial([1]), Polynomial([0]))` is the ZeroDivisionError
  "divmod: division by the zero polynomial" (HEAD: panic); `divmod(Polynomial([-1, 0, 1]),
  Polynomial([1, 1]))` displays `[x + -1, 0]` as at HEAD (the quotient x - 1 and remainder 0 of gp
  `divrem(x^2-1, x+1)`; rustmath-polynomials prints a negative coefficient as `+ -1`).

Unit tests in `kernel.rs`:

* `test_eval_guarded_turns_panics_into_errors`: `eval_guarded(|| panic!("boom"))` is `InternalError`
  containing "boom"; `eval_guarded(|| Ok(EvalResult::text("x")))` is Ok.
* `test_eval_worker_survives_panic` (replaces the integration test `kernel_reports_panics` of the original
  design): a stub `Evaluator` that keeps a counter, panics on the code `"boom"` and returns the counter
  otherwise; `EvalWorker::spawn_with(stub, handle)`; jobs `"a"`, `"boom"`, `"b"` through the real channel:
  replies `1`, `InternalError` containing "boom", `2` (the state survived and the thread is alive).

Integration tests, new file `rustmath-jupyter/tests/kernel_protocol.rs` (header: MAGMA ch. 1 §1.2, ch. 3
§3.12, the Jupyter messaging protocol). Each test starts the kernel binary
(`env!("CARGO_BIN_EXE_rustmath-kernel")`) with a connection file in a `tempfile::tempdir()` (five free
ports, empty key, so no HMAC) and drives it with zeromq `DealerSocket` (shell, control), `SubSocket`
(iopub) and `ReqSocket` (heartbeat), as `$J/s13/client.rs` does; it waits for a `kernel_info_reply` instead
of sleeping, and kills the child on drop. The long cooperative cell is 2000 lines `x = factorial(3000)`
(over 20 s in debug); bounds are generous (5 s) to stay stable on a loaded machine. All fail on HEAD.

| Test | Steps | Expected |
|---|---|---|
| `kernel_answers_while_busy` | `a = 5`; long cell; after 300 ms: heartbeat ping, control `kernel_info_request`, `interrupt_request` | echo, info reply, interrupt reply each within 1 s; `KeyboardInterrupt` reply within 5 s; then `a` gives `5` |
| `kernel_survives_sigint` (`#[cfg(unix)]`, `kill -INT`) | SIGINT while idle; `2+2`; long cell; SIGINT | alive, `4`; `KeyboardInterrupt` within 5 s; alive |
| `kernel_aborts_queued_cells` | long cell, `y = 1`, `y` sent together; interrupt | replies `aborted`, `aborted`, `error` (`KeyboardInterrupt`); a later `y` is the Symbol `y` (never assigned) |
| `kernel_stops_on_shutdown_during_library_call` | `factor(56265256014966897789815307676575655213155187188640868279434630869656784436560427178262558957748157589423366406142190461229644562799099646663196635340923360377107699471706747773488052056948343747546823)` (gp: `setrand(20261112); randomprime([10^99,10^100])` twice, product); interrupt; read iopub for 4 s; `shutdown_request` | interrupt reply within 1 s; a stderr stream containing "does not check for interrupts"; shutdown reply within 1 s; the process exits within 5 s |
| `kernel_json_interrupt_mode` (unit, install.rs) | `generate_kernel_json(..)` | `["interrupt_mode"] == "message"` |

### 4.6 Cost

One extra thread (64 MiB of address space, committed on touch). A job crosses two channels (microseconds).
Checkpoints: one atomic load each.

---

## 5. The C1 hand-off: `partitions` and the factorial arms

C1 adds the library functions and leaves the REPL arms to J1b (C1 §4.2, §7). J1b switches them.

### 5.1 The defects at HEAD

* `partitions(n)` (`:2532-2546`): `int_arg::<usize>(.., 0..=PARTITIONS_N, "p(n) must fit in a machine
  word")` with `PARTITIONS_N = 416`, because `partition_count` is usize-valued. `partitions(417)` is
  refused although p(417) = 18987964267331664557 (gp `numbpart(417)`); MAGMA's `NumberOfPartitions(n)`
  (ch. 18, ch. 145) returns an integer for any small n.
* `falling_factorial(n, k)` and `rising_factorial(n, k)` (`:2348-2398`): n is a u32 (rising: n + k - 1 too),
  so `falling_factorial(2^40, 3)` and `rising_factorial(-5, 3)` are refused although the products are
  1329227995781289095444965371779481600 and -60 (gp `prod(i=0,2,2^40-i)`, `prod(i=0,2,-5+i)`). Sage's
  `falling_factorial(x, a)` and `rising_factorial(x, a)` take any ring element x; MAGMA ch. 144
  `NumberOfPermutations(n, k)` is the falling factorial and ch. 49 uses the rising one as the Pochhammer
  symbol. Their work model `product_work(k, top) = k·(k·bits(top))` was fitted to the old k-step loop; C1's
  product tree is about 15 times faster at k = 10^5 (C1 §7) and its cost is about M(B), the multiplication
  time of the result size B, which the k^2 model under-estimates for few large factors
  (`falling_factorial(x, 2)` with x of 2^27 bits: work 2^29, admitted, one product of 33 s).

### 5.2 Fix

```rust
"partitions" | "partition_count" => {
    // MAGMA NumberOfPartitions(n) (ch. 18, ch. 145): C1's number_of_partitions (Hardy-Ramanujan-
    // Rademacher with a rigorous enclosure; exact for every n <= 10^16).
    let n = int_arg::<u64>("partitions", "n", n, 0..=repl_limits::PARTITIONS_N, "REPL limit, about 10 s")?;
    Ok(RustMathValue::Integer(number_of_partitions(n)))
}
```

`PARTITIONS_N: u64` (provisional 10^10, C1 §2.7: 5.4 s; section 6) with
`const _: () = assert!(repl_limits::PARTITIONS_N <= NUMBER_OF_PARTITIONS_MAX_N);` so C1's domain panic is
unreachable. A negative n stays a ValueError (MAGMA ch. 18: "non-negative integer n").

```rust
"falling_factorial" => {
    // x(x-1)...(x-k+1) for any integer x (Sage falling_factorial; MAGMA ch. 144 NumberOfPermutations).
    let k = int_arg::<u64>("falling_factorial", "k", k, 0..=u64::MAX, "")?;
    if x.signum() >= 0 && Integer::from(k) > *x { return Ok(Integer(0)); }    // the factor 0 occurs
    // largest factor: x for x >= 0; |x| + k - 1 for x < 0
    check_factorial_product("falling_factorial", k, top_bits)?;
    Ok(RustMathValue::Integer(falling_factorial_integer(x, k)))
}
"rising_factorial" | "pochhammer" => {   // the pochhammer name after J1a's S10
    // x(x+1)...(x+k-1), the Pochhammer symbol of MAGMA ch. 49 and Sage rising_factorial.
    if x.signum() <= 0 && Integer::from(k) > -x.clone() { return Ok(Integer(0)); }  // the factor 0 occurs
    // largest factor: x + k - 1 for x > 0; |x| for x <= 0
    check_factorial_product("rising_factorial", k, top_bits)?;
    Ok(RustMathValue::Integer(rising_factorial_integer(x, k)))
}

/// Estimated seconds of a balanced product of k factors of at most `top_bits` bits (C1's
/// product_range_u64 / product_tree; von zur Gathen and Gerhard §10.1): FACT_FACTOR_NS·k +
/// FACT_WORD_NS·W^1.465 ns with W = k·top_bits/64 words, the size of the result; the top levels of
/// the tree cost about M(W) and dominate.
fn factorial_product_seconds(k: u64, top_bits: u64) -> f64
/// Refuses when factorial_product_seconds exceeds PRODUCT_SECONDS: "{cmd}: the product of {k}
/// factors of up to {top_bits} bits would take about {t:.1} s (estimate), more than 10 s (REPL limit)".
fn check_factorial_product(cmd: &str, k: u64, top_bits: u64) -> Result<(), EvalError>
```

Provisional constants from C1's release measurements: `falling(2^32-1, 10^6)` 3.3 s and `10^5` 0.14 s give
`FACT_WORD_NS` = 15 (model 3.3 s and 0.12 s), `FACT_FACTOR_NS` = 20; ladder `falling` refits both. The
model also covers few large factors: k = 2 with x of 2^25 bits estimates 9.9 s (one product of two 2^25-bit
numbers, about 4 s), k = 3 estimates 18 s and is refused (about 12 s). The result has at most k·top_bits
bits, so the time bound also bounds its memory (W^1.465·15 ns <= 10 s gives at most about 7·10^7 bits). All
arithmetic of the model is in f64 (no overflow for k up to u64::MAX). `FACTORIAL_WORK` and `product_work`
remain for `multinomial` only (its constant is re-fitted in section 6).

Correctness of the zero shortcuts: for x >= 0 and k > x the falling product contains the factor x - x = 0;
for x <= 0 and k > -x the rising product contains x + (-x) = 0. C1's functions return 0 there too (C1
§4.2), but only after building the factors, so the REPL answers first (as HEAD did for
`falling_factorial(5, 1000000)`).

### 5.3 Callers and behaviour changes

Only the three arms and `repl_limits` change. `partitions(n)` admits n up to `PARTITIONS_N`; values for
n <= 416 are unchanged (both functions are exact). `falling_factorial` and `rising_factorial` admit any
integer n and any k up to u64::MAX, and refuse by the time model instead of the work model. The values HEAD
computed are unchanged (C1 checks its functions against the u32 loops). `pochhammer` is the rising factorial
after J1a (its §3).

### 5.4 Tests

`test_c1_arms` (all values from gp, `$J/r1/gp/ref_r1.out`, `ref_r1c.gp`):

| Input | Expected | Reference | HEAD |
|---|---|---|---|
| `partitions(417)` | `18987964267331664557` | gp `numbpart(417)` | refused |
| `partitions(10000)` | `36167251325636293988820471890953695495016030339315650422081868605887952568754066420592310556052906916435144` (107 digits) | gp `numbpart(10^4)` | refused |
| `partitions(416)`, `partitions(0)`, `partitions(100)` | `17873792969689876004`, `1`, `190569292` | gp | same |
| `partitions(-1)`, `partitions(PARTITIONS_N + 1)` | ValueError "partitions: n must be an integer in 0..=10000000000" (formatted from the constant) | | other range |
| `falling_factorial(2^40, 3)` | `1329227995781289095444965371779481600` | gp `prod(i=0,2,2^40-i)` | refused |
| `falling_factorial(4294967297, 2)` | `18446744078004518912` | gp `prod(i=0,1,4294967297-i)` | refused |
| `falling_factorial(-3, 4)`, `falling_factorial(-4294967296, 2)` | `360`, `18446744078004518912` | gp `prod(i=0,3,-3-i)` | refused |
| `falling_factorial(10^30, 3)` | `999999999999999999999999999997000000000000000000000000000002000000000000000000000000000000` | gp | refused |
| `falling_factorial(5, 6)`, `falling_factorial(5, 10^12)` | `0`, `0` (at once) | gp `prod(i=0,5,5-i)` | `0`, refused (k > u32) |
| `rising_factorial(-5, 3)`, `pochhammer(-5, 3)` | `-60`, `-60` | gp `prod(i=0,2,-5+i)` | refused |
| `rising_factorial(4294967295, 2)` | `18446744069414584320` | gp | refused ("n + k - 1 must fit in 32 bits") |
| `rising_factorial(2^64, 2)`, `rising_factorial(-(2^64 + 5), 3)` | `340282366920938463481821351505477763072`, `-6277101735386680767919177826258927978529847705109601976380` | gp | refused |
| `rising_factorial(-5, 5)`, `rising_factorial(-5, 6)`, `rising_factorial(-5, 10^12)` | `-120`, `0`, `0` (at once) | gp | refused |
| `falling_factorial(4294967295, 3000000)` | ValueError "the product of 3000000 factors of up to 32 bits would take about" | model 16.8 s | other message |
| `rising_factorial(1, 10000000)` | ValueError (model 65 s) | | other message |
| `falling_factorial(3, -1)` | ValueError "k must be an integer in 0..=18446744073709551615" | | other range |

Existing HEAD tests that change: in `test_int_args_truncation_fixed` (`:11987-11990`) the rows
`falling_factorial(4294967297, 2)`, `rising_factorial(4294967295, 2)` and `partitions(417)` move to value
checks (the values above). In `test_int_arg_ranges_named` the rows `falling_factorial(4294967295, 300000)`
(now admitted: model 0.57 s) and `rising_factorial(1, 400000)` (admitted: 0.42 s) are replaced by the two
refusals above; `partitions(-1)` keeps its prefix (the range is formatted from `PARTITIONS_N`). The value
rows `:12079-12088` (`falling_factorial(10, 3)` = 720, `(5, 1000000)` = 0, `(4294967295, 2)`,
`rising_factorial(3, 4)` = 360, `(0, 4000000000)` = 0, ...) keep their values.

### 5.5 Cost

`partitions` costs what C1 measured (8 ms at 10^6, 0.94 s at 10^9, 5.4 s at 10^10, release). The factorial
arms add one bit-length computation and the model (O(1)).

---

## 6. Limit re-measurement after S1, C1 and MF1 (protocol)

### 6.1 Policy (X2, as written in the `repl_limits` doc comment at HEAD)

A limit is set so that a refused call would take more than about 10 s (release build, this container) or
more than about 400 MB, and an admitted call needs at most a few hundred MB. Two changes in how it is
measured, both forced by findings of this unit:

1. **End to end, with the real `ReplContext`**, not the callee alone: the X2 limits were sized from rustc
   mirrors of the callees and missed the decimal display (section 3; `bell(600)` measured 11.2 s for the
   callee and takes 12.9 s end to end at HEAD, `$J/remeasure/head_run/remeasure.txt`).
2. **No copies, the chart counted**: after F2 a value exists once; the memory of a command is its working
   set plus its result, and the resident values it needs (a form needs a chart of dim(M) names).

Display is part of the time. After section 3 a result up to `OUTPUT_BYTES` of text (about 5.57·10^7 bits
for one integer) prints in at most about 3 s and a larger one is summarised, so the worst admitted input is
either just below the display budget (compute plus about 3 s) or the largest admitted one (compute only).
The ladders sample both.

### 6.2 Protocol

Run after S1, C1, MF1, J1a and J1b's sections 1 to 5 are in the tree.

1. For a limit L: temporarily set L (and any model it feeds) to the top of its ladder in the working
   tree (local edit, never committed).
2. `$J/remeasure/remeasure_r1.sh <outdir> <ladder>`: builds `cargo build --release -p rustmath-jupyter
   --lib` (the shared target dir), links `replmem.rs` against it, and runs every input in its own process
   (`timeout 120`), printing the end-to-end time and VmHWM. Charts are written with dim(M) comma-separated
   names to a file and read with `@file`. Ladders: `fib bell motzkin derange schroder stirling partitions
   falling catalan binom taylor forms coords graph pow mpow mmul pmul scalar str display`.
3. Choose the largest ladder value whose worst admitted input takes at most about 10 s and 400 MB (compute
   plus display, section 6.1); round down to a 1-2-2.5-5 value. For multi-argument models, fit the
   constants (section 6.3) and check the worst admitted inputs the model allows.
4. Set the constant; rerun the ladder; the first refused input must be refused instantly with the
   documented message, and the largest admitted must meet the policy.
5. Write the measured row (input, time, VmHWM) into the constant's doc comment, the help text, and the
   PROJECT_TRACKER row.
6. Debug check: run the tests at the new limits (`cargo test -p rustmath-jupyter`); no test may run an
   admitted input near a limit in debug (the limit tests use the refusals and small values, as today).

### 6.3 Per limit (provisional targets from the designers' measurements; the protocol decides)

| Limit (HEAD) | Arm | Callee after | Evidence | Provisional target | Change besides the constant |
|---|---|---|---|---|---|
| `TAYLOR_ORDER = 20000` (S1 sets 2000 and adds `TAYLOR_WORK`) | `taylor` | S1 engine, balanced output | S1 §5.1: exp(x) Display 83 ms at 2000, 0.94 s at 5000, 39 s at 20000; peak 17 / 86 MB / 1.43 GB standalone; the text is 5.3 MB at order 2000 and 38 MB at 5000 (S1). After section 3, a series above 16 MiB of text (exp(x) above about order 3000) is kept and summarised, so the order bounds the computation only | 5000 if `taylor("exp(x)", x, 0, 5000)` and `sin(x)`, `exp(sin(x))`, `tan(x)` (budget-refused or admitted) stay within 10 s and 400 MB end to end; else 2000 | **Stack note**: S1 measured output depth 18 at order 20000, and J1b runs the REPL on a 64 MiB thread. Check in debug: `taylor("exp(x)", x, 0, TAYLOR_ORDER)` in a `std::thread::Builder::new().stack_size(2 << 20)` thread (the cargo test stack) succeeds; then delete the stack sentences of the doc comment and say the series is balanced |
| `FIBONACCI_N = 1_300_000` | `fibonacci`, `lucas` | doubling | C1 §7: 3·10^7 2.9 s, 5·10^7 5.4 s, 10^8 14.3 s (callee). Result 0.694n bits; rug display 1.5 s at 3.5·10^7 bits, about 3 s at the budget (n = 8.03·10^7) | 5·10^7: about 5.4 + 1.5 = 7 s. 6·10^7 is about 9 s and 7·10^7 about 11 s; above 8.03·10^7 the value is summarised but the compute alone exceeds 10 s (10^8: 14.3 s). Protocol step 3 takes the largest n with compute plus display <= 10 s, rounded down: 5·10^7 | none |
| `BELL_N = 600` | `bell` | Bell triangle | C1: 4000 3.25 s, 5000 6.96 s; results below 6·10^4 bits | 5000 | none |
| `MOTZKIN_N = 4_500` | `motzkin` | three-term recurrence | C1: 2·10^5 4.26 s, 3·10^5 10.0 s; results about 1.6n bits (display negligible) | 250000 (about 7 s) | none |
| `DERANGEMENTS_N = 1_400` | `derangements` | binary splitting | C1: 10^6 3.4 s, 2·10^6 10.7 s; D(n) has log2(n!) bits: 1.85·10^7 at 10^6 (rug display about 0.7 s), 3.9·10^7 at 2·10^6 | 10^6: about 4.1 s. 1.5·10^6 (about 7.5 s) is not a 1-2-2.5-5 value; 2·10^6 is over (12 s) | none |
| `TABLE_BYTES = 4·10^8` with `table_bytes` / `check_table` | `stirling1`, `stirling2`, `eulerian` | rolling rows with band trimming, O(w) memory | C1 §3.3: (5000, 2500) 4.0 / 3.9 / 8.2 s; (10000, 5000) 34 / 31 / 73 s; (20000, 19900) 0.33 / 0.28 / 5.3 s; memory (w + 1)(B/8 + 32) bytes | a time model, refused above 10 s | Replace the memory model by `stirling_seconds(kind, n, k) = (c0·n·w + c1·n·w·B/64)·1e-9` with w the band width (`min(k, n - k + 1)`, Eulerian `min(k', n - k')` after `k' = min(k, n - 1 - k)`) and B an a-priori bound on the result: B = log2(n!) for stirling1 and eulerian (both rows sum to n!), B = n·log2(k) for stirling2 (S(n,k) <= k^n/k!). Provisional c0 = 150 ns, c1 = 1.3 ns; fit per kind on the ladder `stirling`. Keep a memory check (w + 1)·(B/8 + 32) bytes <= TABLE_BYTES. Rename `check_table` to `check_stirling`; `test_table_limits` becomes `test_stirling_limits` with the boundary cases the fitted model gives |
| `SCHRODER_N = 50_000` (sized by the TABLE_BYTES budget) | `schroder`, `schroder_small` | rolling | C1: 10^5 1.83 s, 2·10^5 7.2 s, 3·10^5 16.4 s, 3 MB | 200000 | doc: time, not memory |
| `PARTITIONS_N = 416` (section 5) | `partitions` | `number_of_partitions` | C1 §2.7: 10^9 0.94 s, 3·10^9 1.97 s, 10^10 5.4 s; about 5·10^-10·n s above 10^9; results of about 3.7·n^(1/2) bits | 10^10 (2·10^10 is about 11 s) | u64 domain (section 5) |
| `FACTORIAL_WORK = 2.5e12` with `product_work` | `falling_factorial`, `rising_factorial` (section 5: time model); `multinomial` | product trees | C1 §7: falling(2^32-1, 10^5) 0.14 s, 10^6 3.3 s; rising 3.5 s; multinomial (10^6, [2]·5·10^5) 1.6 s | falling/rising: `factorial_product_seconds` <= 10 s, constants fitted on ladder `falling`; multinomial: FACTORIAL_WORK re-fitted (the k^2 model over-estimates about 15 times at k = 10^5) | section 5 |
| `FORM_COMPONENTS = 10^6` with `check_form_components` | `exterior_derivative`, `wedge` | MF1: 64 B a slot, no tree growth; F2: no copies | MF1 prototype 66 MB for d(dx_0), n = 1000 (10^6 slots); the chart of n <= 2236 names is under 0.2 MB, so it does not count here | 5·10^6 (about 320 MB for the result plus the input) | MF1 §4.4: compute the count with `rustmath_manifolds::tensor_field::component_count` (map its Err to "too many components to store"); delete the arms' `p.checked_add(1)` / `p.checked_add(q)` and let the library's "The degree ... does not fit in a machine word" through as ComputationError; rewrite the doc comment (no "each further d enlarges the expressions", no copies); keep `missing_components_error` |
| `DIFF_FORM_DEGREE = usize::MAX / 2` | `diff_form` | MF1 returns `DegreeOverflow` | MF1 §4.4 | removed | `diff_form` takes any usize degree (a negative one is still refused by the usize conversion); update the test lines `:12204-12207` and `:12379-12388` |
| `MANIFOLD_COORDS = 1_500_000` | `coordinate_form` | F2; J1a's S14 chart | r1 release (`$J/r1/coords_rel.txt`), chart then form: 1.5·10^6 163 / 231 MB; 2.5·10^6 273 / 339 MB; 3·10^6 313 / 406 MB (VmHWM after the chart line / after the form line); a second form adds 64 B per coordinate | 2.5·10^6 | doc: one copy of the form (64 B per component) plus the chart it needs (about 56 B per name plus the vector), measured with the chart |
| `GRAPH_BYTES = 133_000_000` (J1a S15, "one copy; an assignment holds three") | graph constructors | F2 | r1 release (`$J/r1/graph_r1.txt`): `complete_graph(3585)` 134 MB, `add_edge` on it 264 MB; `complete_graph(5000)` 366 MB (J1a's model 369 MB), `add_edge` 727 MB; `path_graph(3·10^6)`, `Graph(7·10^6)` 332 MB, `add_edge` 660 MB | 200·10^6: a graph is held once, `add_edge` copies it (two at its peak, 400 MB). `complete_graph` stays at 3585 (the next size doubles its tables to 264.7 MB); `path_graph` about 1.78·10^6, `Graph(n)` about 4.1·10^6 | J1a's message loses "an assignment holds three copies" and says "add_edge copies the graph"; J1a's `test_graph_limits` boundary rows are recomputed from `graph_bytes` and the new constant |
| `POW_BITS = 2^25` (2.4) | `**`, `^`, `pow` | rug display | 3^e at 2^25 bits 3.82 s end to end, at 2^26 bits about 6.4 s (summarised display), at 2^27 18.4 s | 2^26 | none |
| `RESULT_BITS = 2^25`, `PRODUCT_SECONDS = 10`, `MUL_*`, `POLY_STEP_FACTOR` (2.4) | `matrix_power`, `*` | | 2.3 | RESULT_BITS 2^25 (or 2^26 with POW_BITS); constants refitted on ladders `mpow`, `mmul`, `pmul` | none |
| `OUTPUT_BYTES = 2^24`, `LIBRARY_INT_DISPLAY_BITS = 2^22` (3.2) | display | rug | `sigma(2, 55700000)` 2.93 s (the largest printed integer) | as set | ladder `display` confirms |

The same protocol applies to `CATALAN_N`, `BINOMIAL_BITS`/`BINOMIAL_WORK` with `narayana`/`delannoy`,
`DELANNOY_SUM_CAP`, J1a's `PRIMES_COUNT` (sized at 375 bytes a prime with the copies of an assignment;
after F2 the list is held once) and `STRING_BYTES`, whose callees C1, I3 and F2 changed (C1 §7, I3 §7.3),
with ladders `catalan`, `binom` (and `primes(0, 10^7)` .. `primes(0, 2·10^7)` for PRIMES_COUNT).
`SIGMA_BITS` stays a memory bound; section 3 removes its display time. `EUCLIDEAN_DIM` (387 MB, already an
`Arc`) is unchanged.

Existing tests that read these constants (`test_int_arg_ranges_named` `:12135`, `test_taylor_order_checked`
`:12228`, `test_table_limits` `:12242`, `test_form_component_limits` `:12346`, the help-text checks)
format their expected messages from `repl_limits`, so they follow the constants; the ones listed in the
table get the stated edits.

---

## 7. Found on the way, not fixed here (for HANDOFF)

* `Matrix::pow` (`rustmath-matrix/src/matrix.rs:410-436`) squares the base once more than needed (M^(2e)
  for e = 2^k); the REPL no longer calls it.
* Negative matrix exponents: MAGMA allows `A ^ n` for n < 0 when A is invertible (ch. 26; over Z, det A =
  ±1). The REPL refuses them with a ValueError that says so; it would need an integer inverse (adjugate)
  with a cost bound on the determinant (`det` itself is unbounded, below).
* Library calls cannot be interrupted. A cancellation hook in `rustmath-core` (a thread-local or global
  flag that long loops in integers, matrix, symbolic and combinatorics check) would extend S13 into them.
* Decimal conversion in rustmath-integers (`Display for Integer`, num-bigint) is quadratic: 9.3 s at 2^25
  bits. The REPL converts through rug (1.4 s at 2^25, 8.0 s at 2^27, 18.3 s at 2^28 bits, release); a fast
  `Display` in rustmath-integers (I3's crate) would serve every crate and would let the REPL drop
  `LIBRARY_INT_DISPLAY_BITS`.
* `Integer * Integer` has no size or time check. A product at most doubles the size, so reaching a slow one
  takes many lines, but one product of two 2^27-bit integers takes about 33 s (release) and is not
  interruptible.
* Clifford and Jordan elements are displayed through rustmath-algebras' Debug/Display without the output
  budget (their coefficients grow only through products of elements). `print(a, b, ..)` and `vars()` apply
  the budget to each value, so one call can print several times 16 MiB.
* A plot samples the expression as a tree, and a symbolic expression is a DAG: `e = e + e` repeated makes
  the tree exponentially larger than the value (2^18 leaves: plot 8.9 s, plot3d 47.7 s, release). The
  checkpoints stop a plot between samples, but one sample (`substitute` and `eval_float` in
  rustmath-symbolic) is a library call. A tree-size budget for sampling, like `display_size`, would bound it.
* List literals copy `Integer` elements inline (`[a, a, a]` holds three copies of a), because `Integer`
  is not behind `Arc` (1.2): a list of k copies of a 2^25-bit integer needs k·4 MiB.
* Unbounded library calls that remain: `det`, `rank`, `matrix_inverse`, `charpoly` and `rref` on matrices up
  to `MATRIX_DIM` = 1000; `factor` and `sigma`'s factoring; `nth_prime` (until I1's sieve and a limit).

---

## 8. Plan

### 8.1 Files, in implementation order (each step builds and tests green)

0. Merge check (J1a and C1 are in the tree). `grep -n "OP_POW_EXP\|POW_EXP\|STRING_REPEAT" rustmath-jupyter/src/repl.rs`
   lists every arm and test that uses the constants J1b removes (expected: J1a's Integer and Rational power
   arms, the `pow` arm, string repetition, their tests). `grep -n "chart(" rustmath-jupyter/src/repl.rs`
   confirms J1a's tests use dim(M) names. Confirm C1's `number_of_partitions`, `NUMBER_OF_PARTITIONS_MAX_N`,
   `falling_factorial_integer`, `rising_factorial_integer` are exported from rustmath-combinatorics.
1. `rustmath-jupyter/src/repl.rs`, F2: the enum change, then `cargo build -p rustmath-jupyter
   --message-format=json` and apply the `.into()` suggestions (or by hand, about 120 sites, table 1.2);
   the hand edits; the assignment path of 1.2 (with the two checkpoints added in step 4); tests
   `test_values_shared_not_copied`, `test_value_semantics_kept`.
2. `repl.rs`, S12: `int_pow_bits`, `checked_int_pow`, `int_power_value`, `rational_power_value`,
   `max_entry_bits`, `max_coeff_bits`, `trace` (a small helper), `matrix_product_seconds` with its constants,
   `check_matrix_product`, `check_poly_product`, `check_scalar_product`, `ReplContext::checked_matrix_power`;
   the `**`/`^` arms (replacing J1a's), the `pow`, `matrix_power`, `Matrix * Matrix`, `Polynomial *
   Polynomial`, scalar product (`*` arms and `scalar_mul`), string repetition and concatenation arms;
   `repl_limits` (remove `POW_EXP`, `OP_POW_EXP`, `STRING_REPEAT`; add `POW_BITS`, `RESULT_BITS`,
   `PRODUCT_SECONDS`, `STRING_BYTES` with their measured doc comments); the section-header comment above
   `repl_limits` (cite ch. 18 `^`, ch. 20 `^`, ch. 26 `A ^ n`, Knuth §4.6.3, Horn and Johnson 5.6.9); the
   existing tests of 2.5; new tests of 2.6.
3. `Cargo.toml` (rug) and `repl.rs`, D: `decimal_string`, `poly_text`, `DisplaySize`, `int_text_bytes`,
   `display_size`, `expr_display_size`, `display_summary`, `to_display` / `to_display_unchecked`, the
   Integer, Rational, IntegerMod, Matrix (one pass), Polynomial and List arms, `expr_to_latex` leaves;
   `OUTPUT_BYTES`, `LIBRARY_INT_DISPLAY_BITS` (cite Brent and Zimmermann §1.7); tests of 3.3.
4. `repl.rs`, S13: `InterruptHandle`, the three fields, `interrupt_handle`, `set_time_limit`,
   `time_limit`, `checkpoint`, `eval`/`eval_cell`, the checkpoints of 4.2 (lines, `eval_expr`, the two in
   each store path, `matrix_power`, the 13 plot sampling loops), the `time_limit` command, the `divmod`
   arm, the help text; tests of 4.5 (repl part).
5. `src/protocol.rs` (`stop_on_error` default), `src/handlers.rs` (`EvalOutcome`, `begin_execute`,
   `finish_execute`, `aborted_reply`, `publish_stream`, `handle_complete`), `src/kernel.rs` (module doc with
   the MAGMA ch. 1 §1.2 and ch. 3 §3.12 citations and the "what stops" paragraph; `Evaluator`,
   `EvalWorker::{spawn, spawn_with}`, `eval_guarded`, `Running`, the `select!` loop, abort, control),
   `src/install.rs` (`interrupt_mode`); unit tests `test_eval_guarded_turns_panics_into_errors`,
   `test_eval_worker_survives_panic`, `kernel_json_interrupt_mode`.
6. `rustmath-jupyter/tests/kernel_protocol.rs` (new): the four integration tests of 4.5.
7. `repl.rs`, section 5: the `partitions`, `falling_factorial`, `rising_factorial` / `pochhammer` arms,
   `factorial_product_seconds`, `check_factorial_product` and their constants, `PARTITIONS_N: u64` with the
   static assertion; imports from rustmath-combinatorics; tests of 5.4 and the existing-test edits of 5.4.
8. Section 6, once S1, C1 and MF1 are in: run `remeasure_r1.sh`, set the constants and models, rewrite the
   doc comments and help paragraph, adjust the tests named in 6.3. With it, MF1 §4.4 items 2 to 4: the arms'
   own degree-overflow checks go, `check_form_components` uses `component_count`, and the stale line
   references in the comments of the `exterior_derivative` and `wedge` arms ("diff_form.rs:139",
   "tensor_field.rs:268-292") and of `check_form_components` ("tensor_field.rs:162-166") cite the functions
   instead.
9. Docs: PROJECT_TRACKER.md, jupyter section: rewrite `:2898` (REPL integer arguments and limits: growth
   guards, display budget, the re-measured limits, the C1 arms, the new tests; drop "matrix_power" and "the
   REPL has no timeout" from its not-bounded list), `:2896` (wire protocol: control and heartbeat served
   while busy, `interrupt_mode`, `stop_on_error`), new 🟢 rows "REPL values shared through Arc (F2)", "REPL
   interrupt, time limit and panic isolation (S13)", "REPL display budget and GMP decimal conversion".
   HANDOFF: remove F2 (REPL part), S12, S13 and C1's "REPL falling_factorial / rising_factorial still take
   u32 n" and partitions notes from the split-outs, replace the TAYLOR/C1/MF1 limit notes by the new values,
   add section 7.

### 8.2 Acceptance

```bash
cargo test -p rustmath-jupyter --no-fail-fast            # HEAD: 63 passed (lib); expect 63 + new, and tests/kernel_protocol.rs
cargo test -p rustmath-jupyter --test kernel_protocol    # the four integration tests
cargo check --workspace --tests                          # rustmath-jupyter is a leaf: no other crate depends on it
cargo clippy -p rustmath-jupyter
$J/mem/scen_r1.sh <release replmem built from the final crate>   # F2 table of 1.5 (one copy at every step)
$J/remeasure/remeasure_r1.sh /tmp/<scratch>/final pow mpow mmul pmul scalar str display   # S12 and D boundaries
$J/remeasure/remeasure_r1.sh /tmp/<scratch>/final                                           # section 6, all ladders
gp -q $J/r1/gp/ref_r1.gp < /dev/null; gp -q $J/r1/gp/ref_r1b.gp < /dev/null; gp -q $J/r1/gp/ref_r1c.gp < /dev/null
```

No downstream crate can be affected (`rustmath-jupyter` has no dependents). Every test named "fails on
HEAD" must be checked on HEAD once (apply only the test hunk). The Cargo.toml change adds no crate to
`Cargo.lock` (rug 1.30.0 is already there); `cargo check --workspace` confirms.

### 8.3 Risks

* **Merge with J1a.** J1a edits the same file (parser, `parse_args`, `split_at_depth_zero`, S10/S11/S14/S15
  arms, the power arms, `GRAPH_BYTES`). The F2 edits are mechanical and compiler-guided, so apply them after
  J1a; the line numbers here are HEAD's. Step 0 lists every site of the removed constants.
* **The time models** (matrix products, polynomial products, factorial products) are fits (factor 0.75 to
  2.1 for matrices); refit on the final build. They assume `Matrix::mul` and the polynomial `Mul` as at
  HEAD (M1a, M1b and P1 do not change them) and C1's product tree.
* **Display budget** changes what users see for outputs above 16 MiB of text (about 1.68·10^7 digits).
  Jupyter front ends struggle with outputs of many MB anyway; the summary gives the exact size. `poly_text`
  duplicates rustmath-polynomials' Display format; its unit test catches a change of the library format.
* **rug as a direct dependency.** It is already linked; only the declaration is new. The conversion
  depends on GMP's `mpz_get_str`, whose output is the standard decimal form (the equality test of 3.3).
* **stop_on_error** now aborts queued cells after any error (Jupyter's documented default; HEAD ignored the
  field). A client that sends `stop_on_error: false` keeps the old behaviour.
* **Timing-based tests** (interrupt from a thread, the plot checkpoint, the kernel tests) use bounds of
  several seconds against cells of 20 s or more; a very slow machine could still fail them.
* **The library call of `kernel_stops_on_shutdown_during_library_call`** must outlast the 2 s notice. If a
  later unit bounds `factor` so that the 200-digit semiprime is refused at once, use another single long
  library call (for example `det` of a 1000x1000 matrix with entries 1..10^6, which section 7 lists as
  unbounded).
* **catch_unwind** keeps the kernel alive after a library panic, but cannot repair state inside the
  library that panicked; the message says the variables are kept, nothing more. A stack overflow still
  aborts the process (64 MiB makes it rarer).
* **Integration tests** spawn processes and bind five local TCP ports each; they need `kill` for the SIGINT
  test (`#[cfg(unix)]`).

# Code review 1 of M1a-matrix-rcf-bound (implementation)

Reviewer scratch: `fu2/impl/M1a-matrix-rcf-bound.rev1/` (logs, gp script `refs.gp`, harness sources
`old/main.rs` and `new/main.rs`; the harness binaries were deleted after use). No repository file was
touched. The tree is HEAD `fd081fa` plus this unit's uncommitted changes (6 files, one of them new).

Verdict: **approve**. There are no blockers. Six minor notes follow. None changes a value or the
algorithm.

## 1. Acceptance commands (run by me)

| Command | Result |
|---|---|
| `cargo test -p rustmath-matrix --no-fail-fast` | 270 lib + 3 + 3 + 35 doc = 311 passed, 0 failed, 1 ignored (`acc_matrix.log`) |
| `cargo test -p rustmath-matrix --lib peeled -- --nocapture` | 9 passed (`acc_peeled.log`) |
| `cargo test -p rustmath-finitefields --test matrix_forms_gfp --no-fail-fast` | 19 passed, 0 failed |
| `cargo test -p rustmath-numberfields --no-fail-fast` | 167 passed, 0 failed, 1 ignored |
| `cargo check --workspace --tests` | exit 0 |
| `cargo clippy -p rustmath-matrix --tests` | exit 101 from the 2 `erasing_op` errors in `integer_forms.rs:645,670`. These are old and not in this diff. No lint in `frobenius.rs`, `krylov.rs`, `companion.rs` or `polynomial_matrix.rs` |
| `cargo clippy -p rustmath-matrix` | exit 0, no lint in the touched files |
| `rustfmt --check` | `frobenius.rs`, `krylov.rs`, `matrix_forms_gfp.rs` are clean. `companion.rs` has 13 hunks before and after this change, so the change adds none |
| `grep unsafe frobenius.rs` / added lines | nothing |
| em-dashes in added prose | none |

Per-test debug times (one thread, idle machine): conjdiag20_s7 0.48 s (bound 20 s), wbw20 0.38 s (20 s),
repeated eigenvalues 0.97 s (20 s), arrow 0.89 s (10 s), multilevel families 4.1 s (60 s per call),
GF(101) n = 60 0.10 s (30 s), exact fields 0.08 s, Z-semantics test 0.16 s, H26E9 0.004 s.

## 2. Reference values (re-run by me with gp 2.15, `rev1/refs.gp`)

All of these print 1 or the expected value:

* The CONJDIAG20_S7 constant (I extracted it from the test source): `matfrobenius(A,1) == [prod(x-i)]`,
  `minpoly == charpoly`, `matdet == 20!`.
* The B matrices of wbw20, the repeated-eigenvalue case, arrow20, derog28, nilp28, `3·I_40`, GF(101)
  n = 60 and n = 40.
* H26E9: `[x^2+4, x^2+4, x+4]`, and `t*a*t^-1 == r`.
* The 4×4 sentinel example: `[x^4 - x^3]`, and mod 2, 3, 5, 7 `[x^4 + (p-1)x^3]`.
* `matdiagonal([2,2,3])`, the merge example (`minpoly == (x^2+1)(x-1)^2`), and the lcm of the
  coprime-split example.

## 3. Algorithm and honesty check

I read `frobenius.rs` in full, and the changed parts of `krylov.rs` and `companion.rs`.

* `maximal_vector` matches design 1.2.3. It uses one span with `reduce_prefix(y, m)`, the S1' test
  `f(A)·y ∈ U` (or `len == n`), then the S2 spin of y modulo U. Then come the coprime split, the merge,
  and the respin with an exact check that `deg = deg αβ` and that the relation polynomial equals αβ.
* Every live candidate that is not dropped pushes exactly one h with `h(A)·y ∈ U`, and each h comes
  from a verified fact. A candidate is dropped only before a current vector exists (k = 0, so y ∈ U),
  and then it lies in every later U. Each h divides the final f, and that is checked. Earlier f's
  divide the merged lcm. So the certificate gives `f = exp(V/U)`.
* `reduce_prefix` decides membership in the span of the first m rows. Those rows are an echelon basis
  that later pushes never change, and a nonzero combination of them is nonzero at the pivot of its
  first row. `truncate` keeps `rows/piv/lam/alpha` and `Span::vecs` in step. `Echelon::push` inverts
  before it stores, so an `Err` leaves no half-pushed row (and the span is dropped anyway).
* The r = 1 path returns R[0][0] after checking that it has degree n. For r = 1 the presentation's
  generator is w_0, so T is the Krylov basis of w_0, as the docs say.
* `certified_form` keeps the old certificate (`A·T == T·C`, independent columns, n columns). It adds
  three guards: `factors.len() == generators.len()`, `deg > 0`, and `columns.len() + k <= n`. The last
  one stops pushing early but gives the same `Err` as before. `from_invariant_factors` checks the
  factors are monic and form a divisibility chain. `rcf_from_presentation` still skips degree-0
  diagonal entries before it calls `certified_form`.
* Termination for any `Field` impl: a span holds at most n vectors; each level adds at least one
  vector; a merge must raise `deg f`; `div_rem_field` leaves `r.len() <= deg b`, so the Euclid degree
  drops; the coprime split is capped. Nothing unbounded is reachable from the public API. Memory is
  O(n²).
* Coprime-split cap (F4, review m6). I simulated the per-prime exponent dynamics for all
  e_f, e_a < 400. The gcd count never exceeds `log2(1 + e_f) + 2`, the bound is tight, and it is
  always at most `deg f + 2`.
* Exactness argument (m1). I re-derived both bullets. Full depth: the stored-vector matrix is L·E,
  with L integer lower triangular with diagonal ±1 (alpha) and E integer echelon with unit pivots, so
  it is unimodular. The coordinates are integers. The (F6) quotients of integer polynomials by monic
  ones lie in Z[x]. MinimalPolynomial depth: the D×D pivot minor is ±1, and `f_1(A) = 0` holds over
  Z. The argument is correct, and it also covers the r = 1 path through `krylov.rs` "Exactness".
* No path returns an undecided value. Every failed check is `Err(InvalidOperation("internal: ..."))`,
  and the public chain then falls back to the old certified stages.

## 4. Independent fuzz (harness: working-tree sources by `#[path]`, debug rlibs)

I ran `peel(Full)`, `peel(MinimalPolynomial)` and `rational_canonical_form_peeled` against
`invariant_factors_classical` (Smith form of xI - A). The inputs were W·B·W⁻¹ (r = n), with B random
upper triangular with few distinct diagonal values, or a Jordan matrix (also conjugated by a sparse
unit triangular Q). I counted merges with an atomic counter in the scratch copy only.

| Field | Cases | Wrong | Err | Multi-factor | Merges |
|---|---|---|---|---|---|
| word-size GF(2), GF(3), GF(5), GF(7), n ≤ 12 | 4 × 400 | 0 | 0 | 1287 | 6245 |
| Q, n ≤ 9 | 200 | 0 | 0 | 133 | 620 |
| `FiniteFieldElement` GF(4), GF(9), GF(8), n ≤ 9 | 3 × 150 | 0 | 0 | 378 | 1352 |
| bound `PrimeField` p = 2, 3, 5, 7, 101, n ≤ 11 | 5 × 120 | 0 | 0 | 436 | 2326 |

The guard branch (a | f after S2) never fired, as the design predicts. So over every exact field type
in the workspace the first stage answered every case, and the fallbacks are only reached in Z
semantics or for rounding fields.

## 5. Cost claims (sampled)

* Scalar inputs, new path, against the Smith form of R on this tree (shown as "old"):

  | Input | New IF | New RCF | Old IF |
  |---|---|---|---|
  | word-size GF(7), 2·I, n = 40 | 0.02 s | 0.03 s | 0.004 s |
  | word-size GF(7), 2·I, n = 60 | 0.10 s | 0.13 s | 0.016 s |
  | word-size GF(7), 2·I, n = 100 | 0.79 s | 0.91 s | 0.05 s |
  | Q, 3·I, n = 40 | 0.10 s | 0.18 s | 0.011 s |

  This matches the module docs (0.71 s against 0.05 s, and 0.10 s against 0.01 s).
* Review m5 family (Q, `W·Q·J·Q⁻¹·W⁻¹`, eigenvalues 1, 2, 5, n = 30, 10 seeds): IF 0.66 to 1.05 s and
  RCF 0.87 to 1.51 s, with 6 to 16 merges. This is within the documented "up to 1.8 s and 2.6 s".
* HEAD's public functions on CONJDIAG20_S7, built against the current rlibs (`old/main.rs`,
  `old_cd7.log`): IF 3.26 s, MP 3.50 s, RCF 5.94 s. This confirms the report's 3.2 s / 5.7 s "on this
  tree". The earlier 81 to 102 s did not survive I3's integer changes.

## 6. Tests and assertions

* No assertion was weakened. The only changed assertion is the T comparison in `check_forms`. It now
  compares the public T with `rational_canonical_form_peeled` (the design's behaviour change 2), and a
  separate check certifies the Krylov T (`A·T == T·C`) and its factors.
* `krylov_vs_classical_gfp` and `invariant_factors_agree_with_smith_of_xi_minus_a` keep every old
  check and add the peeled ones. Their matrix lists moved into helpers without change (the count 69 is
  asserted; the GF(p) generator is unchanged).
* `forms_sentinel_natural_families` keeps `answered >= 43`; only its comment changed.
* Design 1.5 tests are all present (`peeled_generators_are_clean` sits in `companion.rs` because of
  the private constants, a documented deviation). Review notes m1 to m8 are all folded in, and I
  checked each in the code.
* The handoffs from earlier units were read. R1's note is informational. F1's container note (binding
  of `Matrix<R>`, `Vector<R>`, `SparseMatrix<F>`) is outside this design, and the report says so under
  "Left undone". The new path itself is not exposed to it: `field_constants` binds every accumulator,
  and an unbound nonzero-looking entry only disables a skip in `mat_vec`.

## 7. Minor notes

* **n1.** `peeled_forms_conjdiag20_s7_was_81_s` no longer detects a regression to the Smith path on
  this tree. HEAD's three public calls take 12.7 s here, under the 20 s bound (section 5). The arrow
  test got a comment for the same situation (m8), but this one did not.
  Required change: add the same caveat to the comment ("on this tree the Smith form of R takes
  3.3 + 3.5 + 5.9 s, so a pass alone does not show the fix"). Optionally, lower the bound to about
  5 s (measured 0.48 s for the whole test). The r = n tests (wbw20: 31.5 + 31.5 + 101 s on the old
  path) still discriminate.
* **n2.** Stale doc. `companion.rs:443` reads "The invariant factors of a square `A` by the Krylov path
  (fast path)." That function is now the first fallback.
  Required change: write "(first fallback)", as was done for `rational_canonical_form_krylov`.
* **n3.** In `release_timing_dense_q` (ignored), `companion.rs:1852` labels the time of the public
  `invariant_factors` as "Krylov path".
  Required change: label it "public path" or "main path".
* **n4.** The `invariant_factors` docs use r both for the number of factors (`f_1 | … | f_r`, an old
  line) and for the number of Krylov blocks (the new Complexity paragraph). `frobenius.rs` uses s for
  the number of factors.
  Required change: use s in the first line, or name r explicitly in the Complexity paragraph.
* **n5.** Optional. `matrix_forms_gfp.rs:345` (`forms_sentinel_non_unit_pivot_is_err`) still names two
  stages ("the Krylov path and the fallback"). The text stays true, because the peeling also needs the
  presentation, but "every stage" would match the new comments. One added comment line in
  `forms_fixed_start_basis_worst_case_gf101` is over 100 characters.
* **n6.** For the orchestrator, not a defect of this unit: F1's container handoff (`Matrix<R>`,
  `Vector<R>`, `SparseMatrix<F>` binding of unbound constants) has no owner among the matrix units.
  M1b binds only its own inputs. It should stay listed as open in HANDOFF.

## 8. Review notes of the design that were skipped

The implementation skipped only the optional parts of m3: a shortcut for a diagonal R, and peeling in
Krylov coordinates. I agree. Neither is required, and the slowdown on scalar-like input stays within
the bound and is documented with measured figures.

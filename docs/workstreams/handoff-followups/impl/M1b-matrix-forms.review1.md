# Review 1 of the M1b-matrix-forms implementation

Reviewer scratch: `$R = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/impl/M1b-matrix-forms.rev1`.
Base HEAD `d3a133b` (M1a committed). Uncommitted tree: 6 modified files, 3 new files
(`rustmath-polynomials/src/factorization_field.rs`, `rustmath-finitefields/src/prime_field_factor.rs`,
`rustmath-matrix/src/primary_forms.rs`), all read in full. The full diff was read too.

Verdict: **approve**. No blockers. Tests pass. Three minor notes below, none of which needs a change
before the commit.

## 1. Acceptance commands (run by me, logs in `$R`)

| Command | Result |
|---|---|
| `cargo test -p rustmath-polynomials --no-fail-fast` | 438 passed, 0 failed, 4 ignored (`test_polynomials.log`) |
| `cargo test -p rustmath-finitefields --no-fail-fast` | 329 passed, 0 failed (`matrix_forms_gfp` 27, lib 201 incl. the 3 `prime_field_factor` tests) |
| `cargo test -p rustmath-matrix --no-fail-fast` | 331 passed, 0 failed, 1 ignored (lib 289, doc 36 incl. `primary_forms::jordan_form`) |
| `cargo test -p rustmath-rings --lib dedekind --no-fail-fast` | 26 passed (the `Matrix::solve` caller in `nfctx.rs`) |
| `cargo check --workspace --tests` | exit 0; no warning in a touched file except the pre-existing unused imports of the commented-out tests in `eigenvalues.rs:259-260` (HEAD `:396-397`) |
| `cargo clippy -p rustmath-matrix -p rustmath-polynomials -p rustmath-finitefields` | exit 0; no hit in a touched file |
| `rustfmt --check` on the three new files, `linear_solve.rs`, `matrix_forms_gfp.rs` | clean |

The counts match the implementation report. No `unsafe`, no em-dash in the new prose, no change to
`rustmath-misc/output.txt` or `.tmp_dir_*`, no tracker or docs file edited.

## 2. Reference values (re-derived with PARI/GP 2.15, `$R/ref.gp`, `$R/ref.log`)

Every expected value in the new tests was recomputed independently, not copied from the design logs:
the four `factor`/`factormod` orders of the factorization tests (`fac1`-`fm3`), all multi-RHS values
(`matinverseimage(A~, W~)~`, `matker(A~)`, `W*A^-1`, `matinverseimage(A, B)`, the GF(3) V, N, `[;]`
and the brute-force counts `[9, 9]`), `U*C*U^-1 == G` with `matdet(U) = 1`, `pif` of G, Sage `m`,
Sage `a`, the ranks `[2, 1, 1]`, `matfrobenius` of A3/B3, the Sage example 1 transform `S*A*S^-1 == B`,
the charpolys of A2/B2, `[x^2, x^2]` against `[x^2, x, x]`, H26E9 (`t*a*t^-1 == r`, `matfrobenius`,
the PIF multiset, the ranks `[2,2,2]`/`[3,3,3]` of the test comment), the GF(3) 7×7 and 4×4 cases,
`nextprime(2^64) = 2^64 + 13` with its `factormod` order, `[x^2]` over GF(2), and `[x+4, x+4]` against
`[x^2+3x+1]` over GF(5). For every expected J or F I checked `matfrobenius(J) == matfrobenius(A)`.
All agree.

## 3. Algorithm and honesty checks

* **Primary core** (`primary_forms.rs:333-430`). Re-derived the action of x on both bases. Companion:
  `x·x^{size-1}q = f_i - Σ c_s x^s q`, so the last column holds `-c_s`, subdiagonal 1s. Jordan:
  `x·x^{d-1}p^j q = p^{j+1}q - Σ c_s x^s p^j q`; the code puts `-c_s` at rows `pos+jd+s`, column
  `pos+jd+d-1`, and the subdiagonal 1 at column `jd+d-1` lands on row `(j+1)d` (the next p-power).
  The `-c` entries and the subdiagonal never overlap. The certificate `M·P == P·G` plus
  `linearly_independent(cols)` proves `P⁻¹MP = G` whatever steps 1-3 did, and failures are
  `Err(InvalidOperation("internal: ..."))`. Index bounds: `deg h < k_i` is checked before the
  column read, `pos + size <= n` before each block.
* **Oracle validation** (`primary_blocks`, `:218-296`): coefficients bound by `c ↦ c + z` first
  (review M2), monic, degree >= 1, multiplicity >= 1, pairwise distinct, each `f_i / Π p^e = 1`,
  valuations of `f_r` equal to the multiplicities. The valuation loop terminates (degree drop is
  checked). Irreducibility of the oracle's factors is trusted (Zassenhaus exact, Berlekamp
  certified by `dim ker(Q - I)`); a broken oracle can only give `Err` or a certified similar matrix.
* **Block order**: keys `(k, e, i)` sorted, factor order stable-sorted by degree. This gives MAGMA's
  printed H26E9 list exactly.
* **is_similar** (`:576-625`): `None` only from unequal certified invariant factors (decided). T from
  `P_A.solve_left_matrix(P_B)`; `T·P_A == P_B` and the certified RCFs with the checked-equal C give
  `T = P_B·P_A⁻¹` invertible and `T·A·T⁻¹ = B`. Correct.
* **Probe and binding**: sentinel-only input is `Err(InvalidArgument)` from all four functions;
  even composite moduli are rejected by the probe; odd composite moduli are documented as
  unspecified (review M3). Measured: mod 9, `jordan_form` gives berlekamp's `InvalidArgument("the
  modulus 9 is not a prime")`, `is_similar(a, a)` answers `Some` (documented); mod 4 both are
  `InvalidArgument`. An unbound 7 next to bound mod 5 entries binds to 2 (`J₂(2)`, pairs `(x+3, 2)`).
* **Multi-RHS** (`linear_solve.rs:175-368`): one RREF of `[A | B]`, `None` iff the last pivot is in a
  B column; free variables `F::zero()` exactly as HEAD, so `solve` is unchanged (the 9 old `solve_*`
  tests and `matrix_forms_gfp::solve_gf7` pass). The left nullspace is `Aᵀ.kernel()`. Shapes with
  0 rows or 0 columns work (tests).
* **Loops and allocations**: `poly_pow` and `check_product` loop at most `Σ e <= n` times; the
  valuation loop at most `deg f_i` times; allocations are O(n²) or `m·(n+k)`. No new unbounded path
  from a public API. The only unbounded costs are M1a's RCF fallbacks and Zassenhaus/Berlekamp,
  which pass through as `Err` at their caps.

### Stress test (`$R/adv.rs`, `$R/adv.log`; compiled with rustc against the workspace rlibs, binary deleted)

27 cases, all pass: direct sums of `C(p^e)` with the input block order shuffled, conjugated by random
unimodular U. Over Q: shared primes with exponents 1-3 across three invariant factors, `x²-2` with
`x³-2`, a cubic before a quadratic in the input (`x²+x+1`, `(x³-x-1)²`), the 5×5 zero matrix,
`x⁴+x+1`, two degree-2 factors whose order needs the second coefficient, and denominators
(`x-3, x+3, x-1/2, (x-1/2)², x+1/3`). Over GF(2), GF(3), GF(5), GF(7): linear, quadratic and cubic
irreducibles with e up to 3. For each: the pairs equal the expected list (I derived the order myself
and checked it with gp `factor`/`factormod`), J and F equal the expected MAGMA-layout matrices exactly,
both transforms satisfy `T·A = F·T` with rank n, `is_similar(A, X)` returns a certified T for X =
the input sum, J and F, and a trace-changing perturbation is consistent with the invariant factors.
Also 200 random GF(2) multi-RHS systems against brute force: 0 mismatches, matrix and sequence forms
agree.

## 4. Cost claims (sample, debug libraries, `$R/tim.rs`, `$R/tim.log`)

| Input | Docs | Measured here |
|---|---|---|
| Q dense n = 8 / 12 / 16, `jordan_form` | 4 / 19 / 72 ms | 4 / 28 / 65 ms |
| Q dense n = 8 / 12 / 16, `is_similar(A, J)` | 16 / 137 / 610 ms | 19 / 116 / 523 ms |
| GF(10²⁴ + 7) dense n = 20, `jordan_form` / `is_similar` | 0.73 / 0.34 s | 0.60 / 0.31 s |
| Q multi-RHS n = k = 30, one elimination against k `solve_left` | 0.35 s / 1.6 s | 0.37 s / 1.86 s (same V) |

Also measured: scalar 20×20 over Q, `jordan_form` 39 ms, `is_similar` 90 ms; `U·diag(1,…,2,…)·U⁻¹`
n = 16, 32 ms / 69 ms. The claims hold. `10²⁴ + 7` is prime (gp `isprime`).

## 5. Design coverage, review notes, handoffs

* Every item of the design (A-D, sections 1-6, Plan steps 1-5) is implemented as specified. Plan step 6
  (trackers, HANDOFF) is left to the orchestrator by instruction and is listed in the report's
  Handoffs with current line numbers (I checked `primary_forms.rs:333/538`,
  `factorization_field.rs:28/99`, `prime_field_factor.rs:30`, `linear_solve.rs:175/236/255/275/295/319/337/360`).
* Design review notes M1-M5: all five are folded in, and I checked each (primitive-integer sort key
  with the `x-1/2` third test; `bind_poly` before the oracle; the composite-modulus sentences in the
  module docs and `is_similar`; the "about 60% (3.4 s on 5.6 s)" comment, labelled as the prototype's
  measurement; each shape case names its A).
* Handoffs from earlier units: M1a's contracts are used unchanged; I1 makes `berlekamp::check_prime`
  a proof for every size, which the `prime_field_factor` docs state correctly; P1's PARI-ordered
  `factor_over_integers` is re-sorted anyway (stable, same key). F1's container binding stays open
  (correctly listed as such).
* Deviations (public `check_monic`/`check_product` helpers, extra validations, new cost tables) are
  reasonable and documented in the report.
* Callers: workspace grep for `JordanForm`, `jordan_form`, `is_similar(`, `primary_rational_form`,
  `FactorizationField`, `solve_matrix`, `solve_left_matrix`, `solve_left_sequence` finds only the new
  code and the re-exports. The old `JordanForm { j, p }` had no constructor and no user.
* Tests: all new tests use new API, so they cannot compile on HEAD. No existing assertion was
  changed or removed. The `linear_solve.rs` diff outside the new code is rustfmt whitespace only
  (three joined lines), and the `eigenvalues.rs` deletions are commented-out code and the unused
  struct only.

## 6. Minor notes (no change required for this commit)

1. `jordan_form`, `primary_rational_form` and `primary_invariant_factors` do not say what
   `is_similar` says: entries with two different `PrimeField` moduli panic in the arithmetic. This
   is the crate-wide behaviour (`invariant_factors` and the RCF do the same), so it is only a doc
   gap. A one-line "the entries must lie in one field" in the module docs, section "Entries and
   fields", would cover all four.
2. `matrix_forms_gfp.rs`, comment of `h26e9_magma_handbook_example`: "factor of each
   matfrobenius(a,1) entry: x+1, x+4 (twice), x+4" reads oddly; gp gives
   `[x^2 + 4, x^2 + 4, x + 4]` with `x^2 + 4 = (x + 1)(x + 4)`. Wording only; the asserted values
   are right.
3. The orchestrator should carry the stated limits into the tracker rows: the generalised Jordan
   layout for `deg p >= 2` is unverified against MAGMA, the factor order over Q is PARI's,
   `FactorizationField` exists for Q and GF(p) only, and `is_similar` is unspecified for an odd
   composite `PrimeField` modulus.

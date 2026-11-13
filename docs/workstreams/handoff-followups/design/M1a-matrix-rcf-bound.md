# M1a-matrix-rcf-bound: a bounded algorithm for invariant factors, minimal polynomial and rational form

Unit M1a of the HANDOFF follow-ups. Crate: rustmath-matrix (one test file in rustmath-finitefields
is touched only for comments and one new test). HEAD `c79a582` (same code as `9c24389`).

Items (docs/HANDOFF.md, "Open follow-ups", `:312-315`):

1. "Rational form and invariant factors: no complexity bound (the fixed start basis has a worst case with
   r = n, and one 2x2 relation matrix over Q took 81 s in debug); [Ste97] or Storjohann would bound it."
2. "Doc nits from the last review: the rational-form "Fallback" paragraph names only the sentinel
   trigger, the `polynomial_matrix` header names only `PrimeField` sentinels."

Scratch: `$S = /tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/M1a-matrix-rcf-bound`.

| Path | Content |
|---|---|
| `$S/exp/frob.rs` | Prototype of the new algorithm (crate-internal API, generic over `Field`). It has a `static mut STATS` counter for measurements: do not copy it (the repo forbids `unsafe`). |
| `$S/exp/common.rs`, `build.sh` | Harness: includes `polynomial_matrix.rs`, `krylov.rs`, `companion.rs` of HEAD by `#[path]` and links the debug rlibs (`$S/rlibs.txt`) |
| `$S/exp/t2.rs` | Fuzz: 600 matrices over Q, 1200 over `PrimeField` GF(2,3,5,7), the 45 sentinel families (all agree with HEAD; the new path never errs on bound input) |
| `$S/exp/t10.rs` | Fuzz: 180 matrices over `FiniteFieldElement` GF(4), GF(9), GF(8) (all agree with HEAD; no error) |
| `$S/exp/t3.rs` | 3000 random sentinel-only matrices: new path against the HEAD Krylov path |
| `$S/exp/t4.rs` | Timings over Q: `./t4 old <case>`, `./t4 new <case>`, `./t4 gp <case>` (dumps `case_<c>.gp` and our factors `ours_<c>.gp`) |
| `$S/exp/t7.rs`, `t8.rs` | Timings over GF(p): `PrimeField` (t7) and a word-size `Gf<101>` (t8) |
| `$S/exp/treal.rs` | `Real` (rounding) inputs: the fallback trigger of item 2 |
| `$S/exp/conjdiag20_s7.rs.txt` | The 81 s matrix as a Rust constant, ready to paste (`CONJDIAG20_S7`) |
| `$S/exp/cmp.gp`, `bref.gp`, `refs.gp`, `cd7chk.gp`, `gf101.gp`, `arrow.gp` | PARI/GP 2.15 reference checks (all print 1) |
| `$S/exp/old_times.log`, `new_times.log`, `t4_v3.log` | Measured times (debug, this container, other jobs running) |
| `$S/dl/diss2up.pdf`, `L76.txt` .. `R86.txt` | A. Storjohann, *Algorithms for Matrix Canonical Forms*, PhD thesis, ETH Zürich 2000 (Diss. ETH 13922), ch. 9 text |

## 0. Summary

| Item | At HEAD | Fix |
|---|---|---|
| 1 | The Smith step on the r×r Krylov relation matrix `R` has no bound. Measured in debug: the 2×2 `R` of `CONJDIAG20_S7` takes `invariant_factors` 101.6 s and `rational_canonical_form` 177.7 s; the r = n input `W·B·W⁻¹` (n = 14) 95 s for `invariant_factors` (n = 20: over 300 s); over GF(101) with `PrimeField`, r = n = 60: `rational_canonical_form` 99 s | New crate-internal module `frobenius.rs`: cyclic decomposition by maximal vectors (peeling). Deterministic, no factorization, O((r + 1)·n³) field operations (so O(n⁴) worst case, the order of Steel's algorithm [Ste97]). Measured in debug: the 81 s case 0.18 s (`invariant_factors`) and 0.31 s (`rational_canonical_form`); r = n, n = 20: 0.31 s and 0.52 s; GF(101) n = 60: 0.75 s and 1.8 s. The old paths stay as fallbacks for Z semantics (sentinels) and rounding fields only |
| 2 | Confirmed: `companion.rs:540-545` says the fallback "only happens" for sentinel matrices, but a `Real` input reaches it too (measured); `polynomial_matrix.rs:24` names only `PrimeField` sentinels, not `FiniteFieldElement` ones | Rewritten paragraphs (section 2.2); they also describe the new three-stage chain |

Nothing is declared out of scope. Storjohann's O(n³) algorithm (thesis ch. 9) is not implemented; the
item asks for a bound, and section 1.2.8 says why O(n⁴) without factorization was chosen. Over Q a
bit-complexity bound is not proved (section 1.2.7 states what is and is not claimed).

## 1. Item 1: a complexity bound

### 1.1 The defect at HEAD

Code: `invariant_factors` (`companion.rs:466`) runs `invariant_factors_krylov` (`:428`): the Krylov
presentation `F^n ≅ F[x]^r / im(R)` (`krylov.rs:265`), then `smith_form_diagonal` (`polynomial_matrix.rs:203`)
of the r×r `R`. `rational_canonical_form` (`companion.rs:564`) runs `rcf_from_presentation` (`:737`) with
`smith_form_with_scaled_left_inverse` (`polynomial_matrix.rs:223`). Both Smith steps are the classical
reduction `smith_reduce` (`polynomial_matrix.rs:238`): least-degree pivot, Euclid by repeated division in
place, row additions for divisibility, no reduction of the other entries modulo anything. The module docs say
so: "No bound is claimed for the Smith step on `R`" (`krylov.rs:29`, `companion.rs:25-36`, `:450-452`, `:559`).

Reproduction (debug build of HEAD, this container, with other jobs running; `$S/exp/t4.rs`, the old path is
the public function at HEAD):

| Input | n | r | Factors | HEAD `invariant_factors` | HEAD `rational_canonical_form` | gp `matfrobenius(A,1)` / `(A,2)` |
|---|---|---|---|---|---|---|
| `CONJDIAG20_S7` = U·diag(1..20)·U⁻¹ (the 81 s case; R diagonal degrees [13, 7]) | 20 | 2 | 1 | **101.6 s** | **177.7 s** | 14 ms / 40 ms |
| `W·B·W⁻¹`, W the start basis, B upper triangular, diagonal 1..14 | 14 | 14 | 1 | **95 s** (HANDOFF; here IF + RCF over 300 s) | | 3 ms / 6 ms |
| the same, n = 20 | 20 | 20 | 1 | (see `old_times.log`, cap 300 s) | | 22 ms / 46 ms |
| CONJDIAG seed 6 | 20 | 3 | 1 | 0.66 s | 6.2 s | 2.0 s / 38 s |
| `PrimeField` GF(101): W·B·W⁻¹, n = 60 | 60 | 60 | 1 | 5.7 s | **99.1 s** | |
| word-size `Gf<101>`: the same, n = 100 | 100 | 100 | 1 | 1.75 s | 29.4 s | |

Why the 2×2 case is slow: for `CONJDIAG20_S7`, `R = [d0 c; 0 d1]` with `deg d0 = 13`, `deg d1 = 7`,
`deg c = 12`, and `c` has coefficients of up to 33 characters (`$S/exp/R_20_7.gp`). gp: `gcd(R[1,1],R[2,2])
= 1`, so the Smith form is `diag(1, d0·d1)`, a trivial answer. `smith_reduce` instead runs Euclid on
`(d1, c)` in column 0 while column 1 accumulates the cofactors times `d0`, with unnormalised rational
remainders, then repairs divisibility with more Euclid steps on those products. Why r = n is slow: then
`R = xI - B` (test `fixed_start_basis_worst_case_has_r_equal_n`, `krylov.rs:486`), so the Smith step is the
classical reduction of an n×n characteristic matrix, which was the slow path before `06945db`. Over GF(p)
there is no coefficient growth, but the operation count of the classical reduction grows fast (Gf<101>:
n = 60 to n = 100, 1.9 s to 29.4 s for the rational form, about n^5.3).

### 1.2 The fix: cyclic decomposition by maximal vectors ("peeling")

#### 1.2.1 Sources

* MAGMA Handbook ch. 26 §26.11 (`MinimalPolynomial`), §26.12.2 (`InvariantFactors`, `RationalForm`); digest
  `rustmath-lava/docs/Chapter_26_Matrices.md:271-312`. MAGMA uses [Ste97] (A. Steel, J. Symbolic Comput. 24
  (1997) 409-432), which factors the characteristic polynomial (digest `:299`). The digest gives no matrix
  for the only worked example H26E9 (M1b ports it from the V2.x page); no new handbook example applies here.
* K. Hoffman and R. Kunze, *Linear Algebra*, 2nd ed., 1971, §7.2, Cyclic Decomposition Theorem (Theorem 3):
  a vector whose annihilator modulo an invariant subspace has maximal degree generates a cyclic subspace
  with an invariant complement. N. Jacobson, *Basic Algebra I*, 2nd ed., §3.8-3.10 (structure of torsion
  modules over F[x]).
* W. Keller-Gehrig, Theoret. Comput. Sci. 36 (1985) 309-317 [KG85]: spinning (already cited in `krylov.rs`).
* A. Storjohann, PhD thesis (2000), ch. 9: Lemma 9.2 (a Krylov basis gives a block companion form, block
  upper triangular when the degree sequence is greedy), Theorem 9.5 (shifted Hessenberg form = right
  Hermite form of xI - A), §9.4 step 5 (in a well chosen spin order the diagonal polynomial s_j divides the
  off-diagonal ones in its column), Fig. 9.6 (complexity table: Steel 1997 is O(n⁴), Storjohann O(n³)).
* E. Bach and J. Shallit, *Algorithmic Number Theory* I, MIT Press 1996, §4.8 (gcd-free bases; the coprime
  split below is its two-element case).

#### 1.2.2 Notation and facts

V = F^n is an F[x]-module with x·v = A·v. U ⊆ V is A-invariant. For y ∈ V, `ann_U(y)` is the monic
generator of {h : h(A)·y ∈ U} (the relative minimal polynomial), and `exp(V/U)` the monic generator of the
annihilator of V/U.

* (F1) Spinning y modulo U (reduce y, A·y, A²·y, ... against an echelon basis of U and of the vectors
  already taken, until the first one that reduces to zero) gives k = deg ann_U(y) and the relation
  `A^k·y = Σ_s c_s·u_s + Σ_{t<k} c'_t·A^t·y` with `ann_U(y) = x^k - Σ_t c'_t·x^t`. The coordinates `c_s`
  are on the stored basis u_s of U.
* (F2) If y_1, ..., y_m generate V/U as an F[x]-module, then `exp(V/U) = lcm_j ann_U(y_j)`.
* (F3) Coprime merge. Let `ann_U(v) = f`, `ann_U(y) = a`, and α | f, β | a monic with gcd(α, β) = 1 and
  αβ = lcm(f, a). Then `v' = (f/α)(A)·v + (a/β)(A)·y` has `ann_U(v') = αβ`. (The two summands have
  annihilators α and β modulo U; for coprime annihilators the annihilator of the sum is the product.)
* (F4) Coprime split (gcd only): `g = gcd(f, a)`, `α = f`, `β = a/g`; while `h = gcd(α, β)` is not 1:
  `α = α/h`, `β = β·h`. Invariant αβ = lcm(f, a), α | f, β | a. Per irreducible π with exponents e_f, e_a:
  if e_a ≤ e_f, β never holds π and α keeps e_f. Otherwise β starts at e_a - e_f, each step moves
  min(e_α, e_β) from α to β, so e_β at least doubles while e_α > 0; at the end α has 0 and β has e_a. So
  the loop ends after at most log2(n) + 2 rounds with gcd(α, β) = 1.
* (F5) Cyclic decomposition (Hoffman-Kunze §7.2): if `ann_U(v) = exp(V/U)` (v is *maximal* modulo U), then
  the cyclic submodule of V/U generated by the image of v has dimension deg exp(V/U), it is a direct
  summand of V/U, and the invariant factors of V/U are exp(V/U) together with those of V/(U + K(v)), where
  `K(v) = span(v, A·v, A²·v, ...)`. Applied level by level, U_ℓ has an A-invariant complement in V.
* (F6) Divisibility of the relation coordinates. Let v_1, ..., v_s be the maximal vectors found level by
  level (U_0 = 0, U_ℓ = U_{ℓ-1} + K(v_ℓ), f_ℓ = ann_{U_{ℓ-1}}(v_ℓ), so f_1 = μ_A and f_{ℓ+1} | f_ℓ). Write
  the level-ℓ relation on the stored basis {A^t·v_i : i < ℓ, t < deg f_i} as
  `f_ℓ(A)·v_ℓ = Σ_{i<ℓ} c_{ℓi}(A)·v_i` with `deg c_{ℓi} < deg f_i`. Then f_ℓ divides every c_{ℓi}.
  Proof: by (F5) V = U_{ℓ-1} ⊕ W with W invariant and f_ℓ(A)·W = 0, so `f_ℓ(A)·v_ℓ = f_ℓ(A)·u` for some
  u ∈ U_{ℓ-1}. By induction every earlier column has the property. Reduce `f_ℓ·(canonical coordinates of
  u)` to canonical form from the last component down: each remainder modulo f_i is a multiple of f_ℓ
  (f_ℓ | f_i), and each quotient multiplies an earlier column whose entries are multiples of f_i, hence of
  f_ℓ. Canonical coordinates are unique, so they are the c_{ℓi}. (Storjohann §9.4 step 5 states the same
  divisibility for his spin order.)
* (F7) Clean generators. With `τ_{ℓi} = c_{ℓi}/f_ℓ`, `g_ℓ = v_ℓ - Σ_{i<ℓ} τ_{ℓi}(A)·v_i` (the **uncorrected**
  v_i) satisfies `f_ℓ(A)·g_ℓ = 0` and `V = ⊕_ℓ K(g_ℓ)`. Proof: the relation matrix of the generators v_ℓ
  is R' = Y·D with D = diag(f_ℓ) and Y unit upper triangular, `Y[i][ℓ] = -τ_{ℓi}`; φ: e_ℓ ↦ v_ℓ is onto
  with kernel im R' (the Krylov presentation theorem, as in `krylov.rs:13-18`), so ψ = φ∘Y has kernel
  im D, and ψ(e_ℓ) = g_ℓ. No compounding: each g_ℓ is built from the v's and the c's only.

#### 1.2.3 Algorithm

```
peel(A, depth)                       // depth ∈ {MinimalPolynomial, Full}; A square n×n, n ≥ 1
  (z, one, _) := field_constants(A)
  P := krylov_presentation(A)?       // generators w_{s_1..s_r}, R, Krylov basis (O(n³), verified)
  if r = 1: return one level (generator w_{s_1}, factor R[0][0], no relation)   // cyclic: done
  live := P.generators               // they generate V as an F[x]-module
  U := empty span                    // echelon form plus the stored vectors, in insertion order
  levels := []
  while dim U < n:
      level := maximal_vector(A, U, live)?          // Err "internal" if no live candidate leaves U
      if levels non-empty and last.factor mod level.factor ≠ 0: Err "internal: not a divisibility chain"
      push level                                    // U now holds U_ℓ
      if depth = MinimalPolynomial: stop
  return levels, U.vectors

maximal_vector(A, S, live)           // S = one span; U = U_{ℓ-1} = its first m vectors (m = dim U)
  cur := none                        // (v, f, coordinates c of A^{deg f}·v); while cur exists, S = U + K(v)
  hs := []                           // certificate: polynomials h with h(A)·y ∈ U, one per candidate
  for y in live, last to first:
    if cur = none:
      (k, c) := spin(S, y)?                            // (F1); pushes K(y) on top of U
      if k = 0: drop y from live for good (y ∈ U ⊆ every later U; nothing was pushed); continue
      f := x^k - Σ_t c[m+t]·x^t; cur := (y, f, c); push f to hs; continue
    (v, f, c_v) := cur
    if len S = n:                              push f to hs; continue   // y ∈ U + K(v) = V, so f(A)·y ∈ U
    if f(A)·y reduces to 0 against the first m vectors of S:             // S1': h = f
                                               push f to hs; continue
    truncate S to m; (k, c) := spin(S, y)?; truncate S to m              // S2: a := ann_U(y)
    a := x^k - Σ_t c[m+t]·x^t; push a to hs
    if f mod a = 0: spin(S, v)?; continue      // guard (unreachable after S1'): restore U + K(v)
    (α, β) := coprime_split(f, a)?                                       // (F4)
    v' := (f/α)(A)·v + (a/β)(A)·y;  f' := α·β                            // (F3)
    if deg f' ≤ deg f: Err "internal: merge did not grow"
    (k', c') := spin(S, v')?                                             // S = U + K(v')
    if k' ≠ deg f' or x^k' - Σ c'[m+t]·x^t ≠ f': Err "internal: merged vector has the wrong annihilator"
    cur := (v', f', c')
  if cur = none: return none (caller turns this into Err)
  for h in hs: if f mod h ≠ 0: Err "internal: exponent certificate failed"
  return Level { generator: v, factor: f, offset: m, relation: c_v[0..m] }   // S now holds U_ℓ = U + K(v)

generators(levels, basis)            // RCF only; increasing order f_s | ... | f_1
  for ℓ: g_ℓ := v_ℓ; for i < ℓ: c := Σ_t relation_ℓ[off_i + t]·x^t (t < deg f_i)
            τ := exact_quotient(c, f_ℓ)?   // Err "internal: relation not divisible" if a remainder is left (F6)
            g_ℓ -= Σ_t τ_t·basis[off_i + t]
  return [g_s, ..., g_1]
```

`spin(span, y)` pushes `y, A·y, ..., A^{k-1}·y` while they stay independent, verifies the final relation
exactly (`Σ_s c_s·u_s == A^k·y` with `F`'s `==`, as `krylov.rs:318-333` does), errs if a push would exceed
n vectors, and returns `(k, c)` with `c` over all stored vectors.

Choices that do not affect correctness, only speed: candidates last to first (the last Krylov generator
has the largest annihilator for the r = n family, so no merge happens there); a candidate found in U while
no current vector exists is in every later U and is dropped; one span serves both U and U + K(v) (the
rows of K(v) sit on top of the first m rows, and "modulo U" means reducing against the first m rows only),
so nothing is cloned; S1' costs deg f matrix-vector products and one reduction; S2 runs only when a merge
follows (if f(A)·y ∉ U then a ∤ f). A cheaper first test "y ∈ U + K(v)" (one reduction against all rows)
is implied by S1' and never fired on the multi-level families, so it is left out. Peeling in Krylov
coordinates (the block companion matrix of R instead of A) was measured and not adopted: faster when R has
small coefficients (the 28×28 derogatory family: 0.05 s against 0.45 s), slower when they are large
(`CONJDIAG20_S7`: 2.2 s against 0.18 s).

#### 1.2.4 What is verified (the result is certified, not only proved)

Over an exact field each claim below is checked by exact computation in the run:

1. Every spin relation `A^k·y = Σ c_s·u_s` (also the relations of the presentation, `krylov.rs:318-333`).
2. After a merge, the respun relative minimal polynomial equals the predicted `αβ`.
3. Exponent certificate: for each live candidate y a polynomial h with `h(A)·y ∈ U` is known from a verified
   fact (y ∈ U + K(v) = V with v's verified relation; or `f(A)·y` reduced to 0 modulo U; or a verified
   spin of y modulo U), and
   `f mod h = 0` is checked at the end of the level. Since the candidates generate V/U, f annihilates V/U;
   v has relative minimal polynomial exactly f (verified spin), so `f = exp(V/U)` and v is maximal.
4. Divisibility chain `f_{ℓ+1} | f_ℓ`, and `dim U` reaches exactly n.
   For r = 1 the presentation is the certificate: its one verified relation has degree n, so the
   generator is cyclic and `R[0][0] = χ_A = μ_A` is the only invariant factor.
5. Rational form: exact divisibility in (F6), then the existing certificate (`A·T == T·C` and independence
   of the columns, `companion.rs:720`, `:782-799`).

Then (F5), applied level by level, proves that `f_s | ... | f_1` are the invariant factors and f_1 = μ_A.
A failed check is `Err(InvalidOperation("internal: ..."))`, never a value.

#### 1.2.5 API (all crate-internal; the public API does not change)

New file `rustmath-matrix/src/frobenius.rs`, `pub(crate) mod frobenius;` in `lib.rs` (next to `krylov`).

```rust
/// How far `peel` goes: the first level only (minimal polynomial) or all levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeelDepth { MinimalPolynomial, Full }

/// One cyclic summand found by `peel` (level ℓ): the maximal vector v_ℓ (uncorrected), its relative
/// minimal polynomial f_ℓ, the offset m = dim U_{ℓ-1} of its Krylov vectors in `Peeling::basis`, and the
/// coordinates of f_ℓ(A)·v_ℓ on basis[0..m].
struct PeelLevel<F: Field> { generator: Vec<F>, factor: UnivariatePolynomial<F>, offset: usize, relation: Vec<F> }

/// The levels in the order found (f_1 = μ_A first, f_{ℓ+1} | f_ℓ) and the stored basis
/// v_1, A·v_1, ..., v_2, A·v_2, ... (complete only for PeelDepth::Full).
pub(crate) struct Peeling<F: Field> { levels: Vec<PeelLevel<F>>, basis: Vec<Vec<F>> }

/// The decomposition of F^n (x acting as `a`), section 1.2.3. `a` must be square (the callers check);
/// n = 0 gives no levels. Err(InvalidOperation("internal: ...")) on a failed check (section 1.2.4),
/// errors of Field::inverse propagate (Z semantics), errors of krylov_presentation propagate.
pub(crate) fn peel<F: Field>(a: &Matrix<F>, depth: PeelDepth) -> Result<Peeling<F>>;

impl<F: Field> Peeling<F> {
    /// f_s | ... | f_1, increasing (the order of `invariant_factors`).
    pub(crate) fn invariant_factors(&self) -> Vec<UnivariatePolynomial<F>>;
    /// f_1 (= μ_A); `None` when there are no levels (n = 0).
    pub(crate) fn minimal_polynomial(&self) -> Option<UnivariatePolynomial<F>>;
    /// The clean generators g_s, ..., g_1 (same order as `invariant_factors`), (F7). Requires Full depth
    /// (Err(InvalidOperation) if `basis.len() != n`). Err(InvalidOperation("internal: ...")) if a division in (F6) is
    /// not exact. `a` is the matrix given to `peel` (its `field_constants` give the bound zero).
    pub(crate) fn generators(&self, a: &Matrix<F>) -> Result<Vec<Vec<F>>>;
}
```

Private helpers in `frobenius.rs`: `Span<F>` (an `Echelon<F>` plus the stored vectors; `len`, `truncate`,
`reduces_to_zero_mod_prefix(y, m)`, `spin`), `relation_polynomial(c_tail, z, one)` (`x^k - Σ c_t x^t`),
`coefficient_polynomial(c, z)`, `monic`, `gcd_monic` (Euclid with `div_rem_field`, every remainder made
monic with `Field::inverse`), `exact_quotient` (Err internal on a remainder), `coprime_split`
(with an iteration cap of deg f + 2, Err internal beyond it).

Changes to `krylov.rs`: `Echelon` becomes `pub(crate)`; `new`, `reduce`, `push`, `coords` become
`pub(crate)`; new `pub(crate) fn len(&self) -> usize`, `pub(crate) fn truncate(&mut self, k: usize)` (drops
rows `k..`; valid because `lam[i]` refers only to rows `< i` and later rows never change earlier ones) and
`pub(crate) fn reduce_prefix(&self, y: &[F], m: usize) -> Vec<F>` (the reduction of `reduce` against the
first m rows only; the result is zero iff y lies in the span of the first m stored vectors, because the
first m rows are an echelon basis of that span). `poly_at_matrix_times_vector`
moves from `companion.rs:491` to `krylov.rs` as `pub(crate) fn poly_mat_vec(p, a, w, z) -> Vec<F>` (Horner,
starts from `z`), used by `companion.rs::image_by_horner` and `frobenius.rs`. A `#[cfg(test)] pub(crate) fn
start_basis<F: Field>(n, k: &(F, F, F)) -> Vec<Vec<F>>` (the n start vectors, as the existing test at
`krylov.rs:486` builds them) for the companion tests. `krylov_presentation` and `linearly_independent`
keep their signatures and behaviour (M1b relies on `field_constants`, `linearly_independent`, `mat_vec`,
`div_rem_field`; all stay).

Changes to `companion.rs`:

```rust
fn invariant_factors_peeled<F: Field>(m: &Matrix<F>) -> Result<Vec<UnivariatePolynomial<F>>>;   // peel(Full)
fn minimal_polynomial_peeled<F: Field>(m: &Matrix<F>) -> Result<UnivariatePolynomial<F>>;       // peel(MinimalPolynomial)
fn rational_canonical_form_peeled<F: Field>(m: &Matrix<F>) -> Result<RationalCanonicalForm<F>>; // peel + certified_form
/// The fallback chain behind the public functions (Krylov Smith form, then xI - A).
fn invariant_factors_fallback<F: Field>(m: &Matrix<F>) -> Result<Vec<UnivariatePolynomial<F>>>;
/// T from the factors and their generators (block by block g, A·g, ...), C from the factors, and the
/// certificate (A·T == T·C, independent_columns); Err(InvalidOperation("internal: rational form
/// certificate failed")) otherwise. Factored out of rcf_from_presentation, which now calls it.
fn certified_form<F: Field>(m: &Matrix<F>, factors: Vec<UnivariatePolynomial<F>>, generators: Vec<Vec<F>>) -> Result<RationalCanonicalForm<F>>;
```

Public functions (signatures unchanged):

* `invariant_factors`: square check; `invariant_factors_peeled(m).or_else(|_| invariant_factors_krylov(m))
  .or_else(|_| invariant_factors_classical(m))`.
* `minimal_polynomial`: square check; n = 0 gives 1 as now; `minimal_polynomial_peeled(m).or_else(|_|
  invariant_factors_fallback(m) last factor)`.
* `rational_canonical_form`: square check and the n = 0 case as now; `rational_canonical_form_peeled(m)
  .or_else(|_| rational_canonical_form_krylov(m)).or_else(|_| rational_canonical_form_classical(m))`.

Invariant for all three: over an exact field the first stage returns Ok (section 1.2.4 never fails on a
lawful exact `Field`; measured: no error on the 1980 bound matrices of `$S/exp/t2.rs` and `t10.rs`); the
fallbacks run only in Z semantics (sentinels) or for a rounding `Field`.

#### 1.2.6 Cost (field operations)

r ≤ n is the number of Krylov blocks, s the number of invariant factors, D_ℓ = deg f_ℓ (Σ D_ℓ = n). A
reduction and a matrix-vector product are O(n²).

* Presentation: O(n³). If r = 1 nothing else runs (most random matrices).
* Level ℓ: first candidate spin O(D_ℓ·n²); each other live candidate O(D_ℓ·n²) (S1'); each merge
  O(D_ℓ·n²) (S2 spin, the two polynomial applications, the respin) plus O(n² log n) (coprime split:
  O(log n) gcds of degree ≤ n). Merges per level ≤ min(D_ℓ, r) (each merge raises deg f, and each
  candidate merges at most once). Each dropped candidate costs one reduction, once.
* Sum: O(n³ + r·Σ_ℓ D_ℓ·n² + r·n² + Σ_ℓ min(D_ℓ, r)·D_ℓ·n²) = **O((r + 1)·n³)**, so O(n⁴) worst case
  and O(n³) when r is bounded.
* Rational form: generators O(s·n²) (exact quotients of degree < n), T by n matrix-vector products O(n³),
  certificate O(n³) (unchanged).
* Memory: O(n²) (one span of at most n vectors of length n, plus its echelon data).

#### 1.2.7 Over Q

The operation count above is a proof. No bit-complexity bound is proved. What can be said (docs must say
it this way, as a description, not a theorem): every vector is built from A, the ±1 start vectors and
divisors of χ_A applied as polynomials in A; every echelon entry and every coordinate is a quotient of
minors of the matrix of stored vectors (Cramer); every factor f_ℓ, a and α, β is a monic divisor of χ_A;
gcds are taken only between such divisors. The generators of (F7) are built from the uncorrected v's, so
no size compounds over levels. Measured: section 1.4.

#### 1.2.8 Why not Storjohann's O(n³) algorithm

Thesis ch. 9 (`$S/dl/L76.txt` .. `R86.txt`): Zigzag form by elementary similarity transforms (§9.2), block
diagonal to Frobenius by a gcd-free basis (§9.3), and a divide and conquer Zigzag to Frobenius step (§9.4)
whose conditioning step needs the `Stab` operation (§1.1, Lemma 1.1) and the machinery of Prop. 7.12. That
is several times the code of this design. Over Q the similarity eliminations of §9.2 work on arbitrary
entries with no Cramer structure, so their sizes are not controlled either. The chosen algorithm reuses
the verified Krylov machinery, needs no factorization (unlike [Ste97]), and has the bound Storjohann lists
for [Ste97] (Fig. 9.6: n⁴). The tracker keeps an "absent" bullet for an O(n³) Frobenius form (section 3).

### 1.3 Callers and behaviour changes

`grep -rn "invariant_factors\|minimal_polynomial(\|rational_canonical_form\|RationalCanonicalForm\|
krylov_presentation\|smith_form_diagonal\|smith_form_with_scaled_left_inverse" --include=*.rs` over the
workspace (excluding target):

* Production callers outside rustmath-matrix: none. `rustmath-numberfields/src/classgroup.rs:386` calls its own
  local `invariant_factors` (integer relations, `:391`); `rustmath-numberfields/src/rnfeq.rs:197` calls
  `companion::characteristic_polynomial` (unchanged); `rustmath-jupyter/src/repl.rs:1480` calls
  `characteristic_polynomial` (unchanged). `minimal_polynomial` hits in algebraic, numberfields and
  finitefields are other functions.
* Test callers: `rustmath-finitefields/tests/matrix_forms_gfp.rs` (`:19`, `:119-145`, `:365`, `:413`, `:525`,
  `:616`, `:707`, `:768`) and the tests in `companion.rs`, `krylov.rs`, `polynomial_matrix.rs`.
* M1b (implemented next) uses only the documented contracts of `invariant_factors`,
  `rational_canonical_form`, `field_constants`, `linearly_independent`, `div_rem_field` (M1b design §0.3).
  All are kept; `mat_vec` and `div_rem_field` keep their names.

Behaviour changes:

1. Values: none. Invariant factors and the minimal polynomial are unique; `canonical_matrix` is unchanged.
2. `change_of_basis` T: for r = 1 the same matrix as at HEAD (the Krylov basis of the first generator). For
   r ≥ 2 a different, equally certified T. Docs already say T is not unique (`companion.rs:154-155`). No test
   pins T's entries; one test compares T between the public function and the Krylov path
   (`companion.rs:1114-1117`, in `check_forms`): it must compare with the peeled path instead (section 1.5).
3. Speed: section 1.4. The pathological inputs drop from minutes (or no result in 300 s) to under 2 s.
   Where HEAD's Smith step on R happens to be cheap, the new path is slower for `invariant_factors`: about
   2× on the multi-level families (28×28 derogatory: 0.95 s against 0.47 s), about 10× on scalar matrices
   (40×40 `3·I`: 0.16 s against 0.016 s; `PrimeField` GF(7), 60×60 `2·I`: 0.65 s against 0.05 s), and
   on the merge-heavy arrow family (n = 20: 0.27 s against 0.04 s). For `rational_canonical_form` the
   new path is within 1.5× of HEAD on those inputs, and far faster wherever HEAD's transform step blows up
   (arrow n = 20: 0.31 s against 21.4 s). All within O((r+1)·n³); the docs state these numbers.
4. Sentinel (Z semantics) input: the chain new → Krylov → classical answers every input HEAD answered, with
   the same values (`$S/exp/t2.rs`: of the 45 families, 19 are answered by the new path and by HEAD's Krylov
   path with equal values, 24 only by the classical stage (at HEAD and after), and `upperones5`,
   `upperones6` stay Err;
   `$S/exp/t3.rs`: of 3000 random sentinel matrices, 926 have a presentation, the new path answers 838, the
   Krylov Smith path 891, and in 53 cases only the Krylov Smith path answers, which is why it stays as the
   middle stage). The count in `forms_sentinel_natural_families` stays 43.
5. `Real` input: whichever stage passes its checks first answers, as before no accuracy claim
   (`$S/exp/treal.rs`: 5×5 Ok on the new path; 8×8 all stages Err, as at HEAD).
6. Errors: same types and messages for non-square input; internal messages of the new path are new
   strings, all `InvalidOperation("internal: ...")`.

### 1.4 Measurements of the prototype (debug, this container, with other jobs running)

All factor lists equal gp `matfrobenius(A,1)` (`$S/exp/cmp.gp`, `arrow.gp`: each prints 1). "new RCF" is
the peeling plus the generators, T and the existing certificate (the time of the new
`rational_canonical_form`). HEAD times from `$S/exp/old_times.log`, `old_times_b.log`, `old_times_c.log`
(300 s cap per case); new times from `$S/exp/new_times_v4.log`.

| Input | n | r | s | HEAD IF | HEAD RCF | new IF | new RCF (peel + T + certificate) | gp IF / RCF |
|---|---|---|---|---|---|---|---|---|
| CONJDIAG20_S7 (the 81 s case) | 20 | 2 | 1 | 101.6 s | 177.7 s | 0.18 s | 0.31 s | 14 / 40 ms |
| CONJDIAG seed 2 (2 merges) | 20 | 2 | 1 | 0.33 s | 1.14 s | 0.93 s | 1.31 s | 10 / 46 ms |
| CONJDIAG seed 6 (2 merges) | 20 | 3 | 1 | 0.66 s | 6.2 s | 0.97 s | 1.36 s | 2016 / 38211 ms |
| W·B·W⁻¹ (r = n), n = 14 | 14 | 14 | 1 | 95 s (HANDOFF); IF + RCF > 300 s | | 0.09 s | 0.14 s | 3 / 6 ms |
| W·B·W⁻¹, n = 20 | 20 | 20 | 1 | > 300 s | | 0.31 s | 0.52 s | 22 / 46 ms |
| W·B·W⁻¹, n = 30 | 30 | 30 | 1 | (not run) | | 3.1 s | 5.8 s | 169 / 288 ms |
| W·B_rep·W⁻¹ (diagonal (i mod 3)+1), n = 20 | 20 | 20 | 2 | > 300 s | | 0.47 s | 1.40 s | 20 / 29 ms |
| arrow (merge stress), n = 20 | 20 | 20 | 1 | 0.04 s | 21.4 s | 0.27 s | 0.31 s | 5 ms |
| arrow, n = 30 (28 merges) | 30 | 30 | 1 | (not run) | | 1.61 s | 1.77 s | 35 ms |
| 3·I, n = 40 | 40 | 40 | 40 | 0.016 s | 0.12 s | 0.16 s | 0.27 s | 0 / 1 ms |
| U·(C(x³+x+1) ⊕ C(x-2))^{⊕7}·U⁻¹ | 28 | 7 | 7 | 0.47 s | 0.95 s | 0.95 s | 1.37 s | 21 / 17 ms |
| U·(J₁(0) ⊕ ... ⊕ J₇(0))·U⁻¹ | 28 | 7 | 7 | 0.37 s | 4.08 s | 0.99 s | 4.54 s | 13 / 22 ms |
| U·J₅(0)^{⊕6}·U⁻¹ | 30 | 6 | 6 | 0.39 s | 0.72 s | 0.72 s | 1.19 s | 12 / 15 ms |
| U·diag(i mod 4)·U⁻¹ | 30 | 8 | 8 | 0.59 s | 2.59 s | 1.21 s | 3.05 s | 11 / 28 ms |
| random dense in [-5, 5] | 30 | 1 | 1 | 4.98 s | 9.47 s | 4.81 s (presentation) | 9.2 s | 156 / 352 ms |
| `PrimeField` GF(101) W·B·W⁻¹ | 60 | 60 | 1 | 5.7 s | 99.1 s | 0.75 s | 1.8 s | |
| `PrimeField` GF(101) W·B·W⁻¹ | 40 | 40 | 1 | 1.40 s | 10.4 s | 0.25 s | | |
| `Gf<101>` W·B·W⁻¹ | 100 | 100 | 1 | 1.75 s | 29.4 s | 0.17 s | | |
| `PrimeField` GF(7) dense | 60 | 1 | 1 | 0.50 s | 1.31 s | 0.31 s | 1.14 s | |
| `PrimeField` GF(7) 2·I | 60 | 60 | 60 | 0.05 s | 0.74 s | 0.65 s | 1.14 s | |
| `PrimeField` GF(2) U·(3×3 block)^{⊕15}·U⁻¹ | 45 | 30 | 15 | 0.34 s | 0.85 s | 0.51 s | 0.73 s | |

The dense rows show the floor of every Krylov method with this `Rational`: one spin of a cyclic vector
(echelon entries are quotients of minors of the Krylov matrix; 200-350 characters at n = 20). That cost is
the same at HEAD and is not part of this item.

### 1.5 Tests

Reference commands: PARI/GP 2.15 (`/usr/bin/gp`). Each expected value is a similarity invariant of a matrix
B similar to the test input, so the gp command uses B (reproducible without our start vectors). All of
`$S/exp/bref.gp`, `cmp.gp`, `cd7chk.gp`, `gf101.gp` print 1.

New tests in `companion.rs` (module `tests`):

| Test | Input | Expected | gp | HEAD |
|---|---|---|---|---|
| `peeled_forms_conjdiag20_s7_was_81_s` | `CONJDIAG20_S7` (paste `$S/exp/conjdiag20_s7.rs.txt`) | `krylov_presentation(a).relations` is 2×2 with diagonal degrees [13, 7]; `invariant_factors_peeled == [Π_{i=1}^{20}(x-i)]`; the three public functions agree; RCF re-checked as in `check_forms_timed` (C, `A·T == T·C`, `rank T == 20`, product == `charpoly_berkowitz`); the three public calls together < 20 s in debug (measured 0.8 s) | `v=[...]; A=matrix(20,20,i,j,v[(i-1)*20+j]); matfrobenius(A,1) == [prod(i=1,20,x-i)]`, `minpoly(A) == charpoly(A)`, `matdet(A) == 20!` (all 1, `cd7chk.gp`) | fails the bound (IF 101.6 s, RCF 177.7 s) |
| `peeled_forms_fixed_start_basis_worst_case_n20` | `A = W·B·W⁻¹`, W = `start_basis(20)`, B upper triangular, `B[i][i] = i+1`, `B[i][j] = ((3i+5j) mod 7) - 3` (j > i, 0-indexed; the B of `krylov.rs:486` at n = 20) | r = 20 and `R == xI - B` (as in the n = 6 test); factors `[Π_{i=1}^{20}(x-i)]`; RCF certified; IF + RCF < 20 s (measured 0.8 s) | `B = matrix(20,20,i,j, if(i==j, i, if(j>i, ((3*(i-1)+5*(j-1))%7)-3, 0))); matfrobenius(B,1) == [prod(i=1,20,x-i)]` (1) | does not finish in 300 s |
| `peeled_forms_r_equals_n_repeated_eigenvalues` | as above with `B[i][i] = (i mod 3) + 1` | `[x - 3, (x-1)^7·(x-2)^7·(x-3)^5]`; RCF certified; IF + RCF < 20 s (measured 1.9 s) | `B = matrix(20,20,i,j, if(i==j, (i-1)%3+1, if(j>i, ((3*(i-1)+5*(j-1))%7)-3, 0))); matfrobenius(B,1) == [(x-1)^7*(x-2)^7*(x-3)^5, x-3]` (1) | does not finish |
| `peeled_merges_arrow_n20` | `A = W·B·W⁻¹`, n = 20, `B[i][i] = i+1`, `B[0][j] = (5j mod 7) - 3` (j ≥ 1), other entries 0 | the last generator has a minimal polynomial of degree 2 (spin it: so merges must run); factors `[Π(x-i)]`; RCF certified; IF + RCF < 10 s (measured 0.6 s) | `B = matrix(20,20,i,j, if(i==j, i, if(i==1 && j>1, ((5*(j-1))%7)-3, 0))); matfrobenius(B,1) == [prod(i=1,20,x-i)]` (1) | fails the bound (IF 0.04 s, RCF 21.4 s) |
| `peeled_multilevel_families` | U = `lcg_unimodular(n, seed)` (4n elementary row operations, coefficients in [-2, 2], a deterministic test helper); `U·(C(x³+x+1) ⊕ C(x-2))^{⊕7}·U⁻¹` (n = 28); `U·(J₁(0) ⊕ ... ⊕ J₇(0))·U⁻¹` (n = 28); `3·I₄₀` | `7 × [x⁴ - 2x³ + x² - x - 2]`; `[x, x², ..., x⁷]`; `40 × [x - 3]`; the peeled path directly, then the public RCF certified | `D = matconcat(matdiagonal(vector(14,k, if(k%2, matcompanion(x^3+x+1), matcompanion(x-2))))); matfrobenius(D,1) == vector(7,k,x^4-2*x^3+x^2-x-2)`; `N = matconcat(matdiagonal(vector(7,k,matcompanion(x^k)))); matfrobenius(N,1) == vector(7,k,x^(8-k))`; `matfrobenius(3*matid(40),1) == vector(40,k,x-3)` (all 1) | the public part passes on HEAD (HEAD is faster here, section 1.4); the peeled part is new API. The test pins the multi-level peeling, 7 to 40 levels |
| `peeled_path_over_prime_field_r_equals_n` | `Gf<101>`: W·B·W⁻¹, n = 60, B as in the n = 20 test | `[Π_{i=1}^{60}(x-i) mod 101]`; peeled IF and RCF certified; r = 60 asserted | `B = matrix(60,60,i,j, if(i==j, i, if(j>i, ((3*(i-1)+5*(j-1))%7)-3, 0)))*Mod(1,101); matfrobenius(B,1) == [Mod(1,101)*prod(i=1,60,x-i)]` (1) | passes (old path 1.9 s with `Gf`) |
| `peel_exact_fields_never_fall_back` | the 69 matrices of `invariant_factors_agree_with_smith_of_xi_minus_a` and the 4×64 of `krylov_vs_classical_gfp` | `peel(Full)` is Ok and equals the classical Smith oracle | (oracle) | API missing |
| `coprime_split_examples` | (f, a) = (x²(x-1)³, x³(x-1)(x+1)); (x³, x⁵); (x-1, x+1); (f, f) | α | f, β | a, gcd(α, β) = 1, αβ = lcm: (α, β) = ((x-1)³, x³(x+1)); (1, x⁵); (x-1, x+1); (f, 1) | `lcm(x^2*(x-1)^3, x^3*(x-1)*(x+1)) == (x-1)^3*x^3*(x+1)` (1), `lcm(x^3,x^5) == x^5` | API missing |
| `merged_vector_has_lcm_annihilator` | A = `C(x²+1) ⊕ C((x-1)²)` over Q (4×4); v = e_1 (ann x²+1), y = e_3 (ann (x-1)²) | `coprime_split`, merge, spin: ann = (x²+1)(x-1)² | `minpoly(matconcat(matdiagonal([matcompanion(x^2+1), matcompanion((x-1)^2)]))) == (x^2+1)*(x-1)^2` (1) | API missing |
| `peeled_generators_are_clean` | DEROG12, NILP15, the derogatory 5×5 of `forms_r8_derogatory_irreducible_quadratic` | for each g_ℓ from `Peeling::generators`: `f_ℓ(A)·g_ℓ == 0` (Horner), and the columns g, A·g, ... are independent | (structural) | API missing |
| `rounding_field_peel_terminates` | the two `Real` matrices of `krylov.rs::rounding_field_terminates` | `peel` returns Ok with total degree n, or `Err(InvalidOperation)`; never hangs | (no reference: no accuracy claim) | API missing |

Changes to existing tests:

* `check_forms` (`companion.rs:1101`): add `invariant_factors_peeled(a) == expected`,
  `minimal_polynomial_peeled(a) == last`, and compare `rcf.change_of_basis` with
  `rational_canonical_form_peeled(a)` instead of `rational_canonical_form_krylov(a)` (`:1114-1117`); keep a
  separate check that `rational_canonical_form_krylov(a)` is Ok with a certified T (`A·T == T·C`).
* `check_forms_timed` (`:1408`): assert `invariant_factors_peeled(a) == expected` (the fast path is the one
  timed).
* `invariant_factors_agree_with_smith_of_xi_minus_a` (`:1591`): add the peeled factors and its certified T.
* `check_krylov_gfp` (`:2134`) and `krylov_vs_classical_gfp` (`:2236`): add the peeled path.
* `forms_conjdiag20_q_small_r_rcf_timing` (`:1858`): keep the 15 s bound, update the comment ("measured with
  the peeled path: about 0.5 s in debug; the Krylov Smith path took 2.1 s, before `702d0a5` 30 s").
* `release_timing_dense_q` (`:1669`, ignored): add `CONJDIAG20_S7` and the n = 20 worst case.

`rustmath-finitefields/tests/matrix_forms_gfp.rs`: no expected value changes. Update the comment of
`forms_sentinel_natural_families` (`:589-594`) and `forms_sentinel_permutations_and_jordan` (`:457-459`) to
name the three stages. Optionally add `forms_fixed_start_basis_worst_case_gf101` (`PrimeField`, n = 40, the
start vectors rebuilt in the test with the LCG of `krylov.rs:84-99`; expected `[Π_{i=1}^{40}(x-i)]` mod 101,
gp as above with 40; bound 30 s: HEAD RCF 10.4 s, new 0.4 s; this documents the PrimeField speed-up, it does
not fail on HEAD).

Test time added (debug, measured parts): about 10 s in total, the largest part the multi-level families.

### 1.6 Docs inside the crate (item 1)

* `frobenius.rs` header: MAGMA ch. 26 §26.11 and §26.12.2 with the digest path; [Ste97] (what MAGMA uses,
  and that it factors); the algorithm (1.2.3), the facts with their sources (1.2.1-1.2.2), what is verified
  (1.2.4), cost (1.2.6), the Q statement (1.2.7), and that Storjohann's O(n³) algorithm is not implemented.
* `companion.rs` module docs (`:13-41`): Algorithm (peeling first; the Krylov Smith path and `xI - A` as
  fallbacks), Complexity (O((r+1)·n³) field operations; no bit bound over Q; the 81 s case and the r = n
  case with their new times), Fallback (section 2.2). Remove "No bound is claimed for the Smith step".
* `invariant_factors` (`:439-465`), `minimal_polynomial` (`:477-482`), `rational_canonical_form` (`:520-563`):
  "Computed from" paragraphs, the certificate list (1.2.4), Complexity, Fallback.
* `krylov.rs` "Cost" (`:27-35`): the presentation stays O(n³); the bound statement moves to `frobenius.rs`;
  the r = n remark stays as a fact about the presentation, with a pointer that the peeling handles it in
  O(n⁴).
* `polynomial_matrix.rs` header (`:12-25`): its Smith reduction now only serves the fallbacks (section 2.2).

## 2. Item 2: doc nits

### 2.1 Confirmed at HEAD

* `companion.rs:540-545`, the "Fallback" paragraph of `rational_canonical_form`: "As for
  [`invariant_factors`], this only happens for matrices built only from unbound `Ring::zero()`/`one()`
  sentinels". It omits the rounding-field trigger, which the module docs (`:37-42`) and `invariant_factors`
  (`:463-465`) name. Measured (`$S/exp/treal.rs`): for the 8×8 `Real` matrix of
  `krylov.rs::rounding_field_terminates`, `krylov_presentation` errs, so `rational_canonical_form` runs the
  classical fallback, which returns `Err(InvalidOperation("internal: rational form certificate failed"))`.
  The sentence "this only happens for [sentinels]" is false.
* `polynomial_matrix.rs:24`: "which happens for matrices of unbound `PrimeField` sentinels computed in Z". The
  GF(p^n) type `FiniteFieldElement` has the same unbound sentinels and reaches the same fallback
  (`rustmath-finitefields/tests/matrix_forms_gfp.rs:489-534`, the 3-cycle and the all-ones 3×3). The same
  narrowing is in `polynomial_matrix.rs:176-178` ("For `PrimeField`'s modulus-0 sentinels ..."): fix it too.

### 2.2 Fix (the new texts also describe the three-stage chain)

`rational_canonical_form`, "Fallback" paragraph:

> Fallback: if the peeled construction returns `Err`, the Krylov construction runs (the Smith form of the
> relation matrix R with P⁻¹, as before), and if that returns `Err` too, the same construction on the
> classical presentation by `xI - A` with the unit vectors as generators; the result or error of the last
> stage is returned. Over an exact field (Q, bound GF(p) or GF(p^n)) the first stage does not fail. Two kinds
> of input reach the fallbacks. First, matrices built only from the unbound `Ring::zero()`/`one()`
> sentinels of `PrimeField` or `FiniteFieldElement`, computed in Z, where only ±1 are invertible: a stage
> that meets another pivot returns `Err`. There a result is returned only when every inverted pivot is ±1
> and `det T = ±1`, so it is the result over Z and, reduced, over GF(p) for every p. Second, a rounding
> `Field` impl (`Real`) whose relation checks or certificate fail; the result is then whatever stage passes
> its checks, with no accuracy claim, and it can be `Err` (the 8×8 matrix of the `krylov` tests).

`polynomial_matrix.rs` header, last paragraph:

> `companion.rs` applies the reduction only in its fallbacks: to the r×r relation matrix of a Krylov
> presentation (`krylov.rs`), and then to the n×n characteristic matrix `xI - A`. Its main path
> (`frobenius.rs`) needs no Smith form. The fallbacks run for matrices built only from the unbound
> `Ring::zero()`/`one()` sentinels of `PrimeField` or of the GF(p^n) type `FiniteFieldElement` (computed
> in Z), and for a rounding `Field` impl (`Real`) whose checks fail. No complexity bound is claimed for
> this reduction: coefficient and degree growth over Q are not controlled (one 2×2 relation matrix took
> 101 s in debug).

`polynomial_matrix.rs:176-178`: "For the modulus-0 `Ring::zero()`/`Ring::one()` sentinels of `PrimeField`
and `FiniteFieldElement`, `is_zero()` can only answer ..." (the rest unchanged).

The same two triggers, named once each, in the `invariant_factors` and `minimal_polynomial` docs and the
module docs (they already name both; reword for the three stages).

## 3. Trackers and HANDOFF

* `PROJECT_TRACKER.md`: rows `:954` (invariant factors and minimal polynomial), `:956` (Krylov
  presentation: now also the candidate source of the peeling), `:964` (rational form), `:967` (F[x] Smith
  form: fallback only); a new row for `frobenius.rs` (🟢, the peeling, its tests). Absent bullets `:978`
  (minimal polynomial with a bound: now present, O((r+1)n³) field operations; remove or reword to "no
  bit-complexity bound over Q") and `:980` (reword: "Frobenius form in O(n³) (Storjohann 1998/2000)": still
  absent; an O(n⁴) bound is in place). Change Log entry.
* `docs/magma/status_overrides.csv`: rows `ch026.InvariantFactors` (`:3192`), `ch026.MinimalPolynomial`
  (`:3178`), `ch059.MinimalPolynomial` (`:493`), `ch026.RationalForm` (`:3190`): new path and `file:line`,
  the bound, the new timings. Statuses unchanged (verified / partial / partial / partial; the "partial"
  reasons are unrelated: Z via Q, no Proof parameter, MAGMA's row convention). Then
  `python3 tools/magma_tracker/build.py`.
* `docs/HANDOFF.md`: remove the item at `:312-315`; add a line to the "fixed" list with the commit.

## 4. Plan

Implementation order (one crate, one commit or two):

1. `krylov.rs`: `Echelon` pub(crate) with `len`, `truncate`, `reduce_prefix`; `poly_mat_vec` moved here;
   `#[cfg(test)] start_basis`; a unit test `echelon_truncate_and_prefix` (push 4 vectors, truncate to 2,
   `reduce_prefix` decides membership in the span of the first 1 and 2 vectors, `coords` still recovers
   coordinates after truncation). `cargo test -p rustmath-matrix --lib krylov` (unchanged results).
2. `frobenius.rs` (new) + `lib.rs` line: `Span`, polynomial helpers, `peel`, `maximal_vector`,
   `Peeling::{invariant_factors, minimal_polynomial, generators}`, and its own unit tests
   (`coprime_split_examples`, `merged_vector_has_lcm_annihilator`, `peeled_generators_are_clean`,
   `rounding_field_peel_terminates`). Port from `$S/exp/frob.rs` (drop `STATS`/`bump`/`spin_timed`, keep the
   logic: the single span with `reduce_prefix`, the S1' then S2 order, the certificate list, candidate
   dropping, the r = 1 fast path).
3. `companion.rs`: `certified_form` factored out of `rcf_from_presentation`; the `*_peeled` functions;
   the three-stage chains in the public functions; `image_by_horner` uses `krylov::poly_mat_vec`; docs
   (1.6, 2.2); tests (1.5).
4. `polynomial_matrix.rs`: header and `:176-178` docs only.
5. `rustmath-finitefields/tests/matrix_forms_gfp.rs`: comments; optional GF(101) test.
6. Trackers and HANDOFF (section 3); `python3 tools/magma_tracker/build.py`.

Acceptance:

```bash
cargo fmt -p rustmath-matrix
cargo clippy -p rustmath-matrix --tests          # no new warnings in the touched files
cargo test -p rustmath-matrix --no-fail-fast      # all green; the new timing tests well under their bounds
cargo test -p rustmath-matrix --lib peeled -- --nocapture
cargo test -p rustmath-finitefields --test matrix_forms_gfp --no-fail-fast
cargo test -p rustmath-numberfields --no-fail-fast   # uses companion::characteristic_polynomial only
cargo check --workspace --tests
grep -n "unsafe" rustmath-matrix/src/frobenius.rs    # nothing
```

Risks:

* Z semantics: the new path may err where the Krylov Smith path answers (53 of 926 random sentinel cases);
  the middle stage is required, do not drop it. Keep `forms_sentinel_natural_families` at 43.
* `PrimeField` and `FiniteFieldElement` sentinels: every constant must come from `field_constants` (start
  vectors, accumulators of `poly_mat_vec` and `mat_vec`, the `one` of `relation_polynomial`). The
  `forms_sparse_bound_gf2_gf3_gf5` test checks that T stays bound.
* The echelon handling relies on `truncate` keeping `lam`/`alpha` consistent (rows only refer to earlier
  rows) and on `reduce_prefix` reducing against the first m rows only; `Span` must truncate its stored
  vectors in step. `echelon_truncate_and_prefix` pins both.
* Debug timings vary with load; bounds are 20 s or more against measured times under 2 s.
* Scalar-like input (r = s = n) is slower than at HEAD (0.78 s against 0.05 s at n = 60 over `PrimeField`);
  documented in 1.3; within the bound.
* Rounding fields: termination relies on the caps (n stored vectors per span, deg f must grow on a merge,
  the coprime split cap); `rounding_field_peel_terminates` covers it.

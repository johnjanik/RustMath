//! Cyclic decomposition of `F^n` (x acting as a square matrix `A`) by maximal vectors
//! ("peeling"): the invariant factors, the minimal polynomial and the generators of the
//! rational form in `O((r + 1)·n³)` field operations, with no factorization.
//!
//! MAGMA Handbook ch. 26 §26.11 (`MinimalPolynomial`) and §26.12.2 (`InvariantFactors`,
//! `RationalForm`), digest `rustmath-lava/docs/Chapter_26_Matrices.md`. MAGMA uses Steel's
//! algorithm \[Ste97\] (A. Steel, J. Symbolic Comput. 24 (1997) 409–432), which factors the
//! characteristic polynomial. This module does not factor.
//!
//! Sources.
//! * K. Hoffman and R. Kunze, *Linear Algebra*, 2nd ed., 1971, §7.2, Cyclic Decomposition
//!   Theorem (Theorem 3): a vector whose annihilator modulo an invariant subspace has maximal
//!   degree generates a cyclic subspace with an invariant complement. N. Jacobson, *Basic
//!   Algebra I*, 2nd ed., §3.8–3.10 (torsion modules over F\[x\]).
//! * W. Keller-Gehrig, Theoret. Comput. Sci. 36 (1985) 309–317 \[KG85\]: spinning (`krylov.rs`).
//! * A. Storjohann, *Algorithms for Matrix Canonical Forms*, PhD thesis, ETH Zürich 2000
//!   (Diss. ETH 13922), ch. 9: §9.4 step 5 states the divisibility (F6) for his spin order;
//!   Fig. 9.6 lists Steel's algorithm as `O(n⁴)`, the bound reached here. Storjohann's `O(n³)`
//!   algorithm (zigzag form, §9.2–9.4; ISSAC 1998) is not implemented.
//! * E. Bach and J. Shallit, *Algorithmic Number Theory* I, MIT Press 1996, §4.8 (gcd-free
//!   bases; the coprime split (F4) is its two-element case).
//!
//! Notation. `V = F^n` is an `F[x]`-module with `x·v = A·v`, and `U ⊆ V` is A-invariant. For
//! `y ∈ V`, `ann_U(y)` is the monic generator of `{h : h(A)·y ∈ U}`, and `exp(V/U)` the monic
//! generator of the annihilator of `V/U`. `K(v) = span(v, A·v, A²·v, …)`.
//!
//! Facts.
//! * (F1) Spinning `y` modulo `U` (reduce `y, A·y, A²·y, …` against an echelon basis of `U` and
//!   of the vectors already taken, up to the first one that reduces to zero) gives
//!   `k = deg ann_U(y)` and the relation `A^k·y = Σ_s c_s·u_s + Σ_{t<k} c'_t·A^t·y`, with
//!   `ann_U(y) = x^k - Σ_t c'_t·x^t` and `u_s` the stored basis of `U`.
//! * (F2) If `y_1, …, y_m` generate `V/U`, then `exp(V/U) = lcm_j ann_U(y_j)`.
//! * (F3) Coprime merge: let `ann_U(v) = f`, `ann_U(y) = a`, and `α | f`, `β | a` monic with
//!   `gcd(α, β) = 1` and `αβ = lcm(f, a)`. Then `v' = (f/α)(A)·v + (a/β)(A)·y` has
//!   `ann_U(v') = αβ` (the summands have the coprime annihilators α and β modulo `U`).
//! * (F4) Coprime split (gcds only): `g = gcd(f, a)`, `α = f`, `β = a/g`; while
//!   `h = gcd(α, β) ≠ 1`: `α ← α/h`, `β ← β·h`. Invariant: `αβ = lcm(f, a)`, `α | f`, `β | a`.
//!   Per irreducible π with exponents `e_f`, `e_a`: if `e_a ≤ e_f`, β never holds π. Otherwise
//!   β starts with `e_a - e_f ≥ 1`, each round moves `min(e_α, e_β)` from α to β, so `e_β` at
//!   least doubles while `e_α > e_β`, and one more round empties α. So the loop ends after at
//!   most `log2(1 + e_f) + 2` gcds (the last one, which returns 1, included), and
//!   `log2(1 + e_f) + 2 ≤ deg f + 2`. [`coprime_split`] caps its gcds at `deg f + 2`.
//! * (F5) Cyclic decomposition (Hoffman-Kunze §7.2): if `ann_U(v) = exp(V/U)` (v is *maximal*
//!   modulo `U`), the image of `K(v)` in `V/U` has dimension `deg exp(V/U)` and is a direct
//!   summand, and the invariant factors of `V/U` are `exp(V/U)` together with those of
//!   `V/(U + K(v))`.
//! * (F6) Let `v_1, …, v_s` be the maximal vectors found level by level (`U_0 = 0`,
//!   `U_ℓ = U_{ℓ-1} + K(v_ℓ)`, `f_ℓ = ann_{U_{ℓ-1}}(v_ℓ)`, so `f_1 = μ_A` and `f_{ℓ+1} | f_ℓ`),
//!   and write the level-ℓ relation on the stored basis `{A^t·v_i : i < ℓ, t < deg f_i}` as
//!   `f_ℓ(A)·v_ℓ = Σ_{i<ℓ} c_{ℓi}(A)·v_i` with `deg c_{ℓi} < deg f_i`. Then `f_ℓ | c_{ℓi}`.
//!   Proof: by (F5), `V = U_{ℓ-1} ⊕ W` with W invariant and `f_ℓ(A)·W = 0`, so
//!   `f_ℓ(A)·v_ℓ = f_ℓ(A)·u` for some `u ∈ U_{ℓ-1}`. Reduce `f_ℓ·(coordinates of u)` to the
//!   canonical form from the last component down: each remainder modulo `f_i` is a multiple of
//!   `f_ℓ` (as `f_ℓ | f_i`), and each quotient multiplies an earlier column whose entries are
//!   multiples of `f_i` (induction). Canonical coordinates are unique.
//! * (F7) Clean generators: with `τ_{ℓi} = c_{ℓi}/f_ℓ`, `g_ℓ = v_ℓ - Σ_{i<ℓ} τ_{ℓi}(A)·v_i`
//!   (the *uncorrected* `v_i`) has `f_ℓ(A)·g_ℓ = 0`, and `V = ⊕_ℓ K(g_ℓ)`. Proof: the relation
//!   matrix of the generators `v_ℓ` is `R' = Y·D` with `D = diag(f_ℓ)` and Y unit upper
//!   triangular (`Y[i][ℓ] = -τ_{ℓi}`); `φ: e_ℓ ↦ v_ℓ` is onto with kernel `im R'` (the
//!   presentation theorem of `krylov.rs`), so `ψ = φ∘Y` has kernel `im D`, and `ψ(e_ℓ) = g_ℓ`.
//!
//! Algorithm ([`peel`]).
//! 1. The Krylov presentation (`krylov.rs`, `O(n³)`, verified) gives r generators of V. If
//!    r = 1, its verified relation has degree n: V is cyclic and `R[0][0] = χ_A = μ_A`.
//! 2. Otherwise, level by level: a maximal vector `v_ℓ` modulo `U_{ℓ-1}` is found among the
//!    live candidates (the generators), with `f_ℓ = exp(V/U_{ℓ-1})`, and `U_ℓ = U_{ℓ-1} + K(v_ℓ)`.
//!    The loop stops at `dim U = n`, or after level 1 for the minimal polynomial.
//! 3. A level takes the candidates last to first. The first one not in U is spun (F1) and
//!    becomes the current vector v with `f = ann_U(v)`. For each later candidate y: if
//!    `f(A)·y ∈ U`, nothing changes. Otherwise `a = ann_U(y)` is spun, `(α, β)` is the coprime
//!    split (F4) of `(f, a)`, and v becomes `v' = (f/α)(A)·v + (a/β)(A)·y` with
//!    `f' = αβ = lcm(f, a)` (F3), respun and checked. A candidate found in U before any current
//!    vector exists lies in every later U and is dropped for good.
//! 4. The rational form uses the clean generators (F7) ([`Peeling::generators`]).
//!
//! One span (an `Echelon` of `krylov.rs` plus the stored vectors) holds U and, on top of it,
//! `K(v)`: "modulo U" means reducing against its first `m = dim U` rows only, so nothing is
//! cloned.
//!
//! What is verified. Over an exact field every claim is checked by exact computation in the
//! run; a failed check is `Err(InvalidOperation("internal: …"))`, never a value.
//! 1. Every spin relation `A^k·y = Σ c_s·u_s` (and every relation of the presentation).
//! 2. After a merge, the respun relative minimal polynomial equals the predicted `αβ`.
//! 3. Exponent certificate: for each live candidate y a polynomial h with `h(A)·y ∈ U` is known
//!    from a verified fact (`y ∈ U + K(v) = V`; or `f(A)·y` reduced to 0 modulo U; or a
//!    verified spin of y modulo U), and `f mod h = 0` is checked at the end of the level.
//!    Since the candidates generate `V/U`, f annihilates `V/U`; v has relative minimal
//!    polynomial exactly f, so `f = exp(V/U)` and v is maximal.
//! 4. The divisibility chain `f_{ℓ+1} | f_ℓ`, and `dim U` reaches exactly n.
//! 5. Rational form: the divisions of (F6) are exact, then the certificate of
//!    `companion.rs` (`A·T == T·C`, independent columns).
//!
//! Then (F5), applied level by level, proves that `f_s | … | f_1` are the invariant factors and
//! `f_1 = μ_A`.
//!
//! Cost (field operations). r ≤ n is the number of Krylov blocks, s the number of invariant
//! factors, `D_ℓ = deg f_ℓ` (`Σ D_ℓ = n`); a reduction or a matrix-vector product is `O(n²)`.
//! * The presentation is `O(n³)`. For r = 1 nothing else runs (most random matrices).
//! * Level ℓ: the first spin `O(D_ℓ·n²)`; each other live candidate `O(D_ℓ·n²)` (the test
//!   `f(A)·y ∈ U`); each merge `O(D_ℓ·n²)` (spin of y, two polynomial applications, respin)
//!   plus the coprime split (`O(log n)` gcds of degree at most `D_ℓ`). A merge raises `deg f`
//!   and a candidate merges at most once, so a level has at most `min(D_ℓ, r)` merges.
//! * Sum: `O(n³ + r·Σ_ℓ D_ℓ·n² + r·n²·log n + Σ_ℓ min(D_ℓ, r)·D_ℓ·n²) = O((r + 1)·n³)`, so
//!   `O(n⁴)` in the worst case and `O(n³)` for bounded r. The clean generators cost `O(s·n²)`.
//!   Memory `O(n²)`.
//! * The bound is tight for inputs with many small blocks and many invariant factors (scalar
//!   or diagonal with few distinct values: `r ≈ s ≈ n`, `Θ(n⁴)`), where the Smith form of the
//!   diagonal relation matrix is cheaper (measured in `companion.rs`).
//!
//! Over Q. The operation count is a proof; no bit-complexity bound is proved. What can be said
//! (a description, not a theorem): every vector is built from A and the ±1 start vectors by
//! polynomials in A; every echelon entry and every coordinate is a quotient of minors of the
//! matrix of stored vectors (Cramer); every `f_ℓ`, a, α, β is a monic divisor of `χ_A`, and
//! gcds are taken only between such divisors. Within a level each merge applies the cofactors
//! `(f/α)(A)` and `(a/β)(A)` to the current vector, so after several merges that vector is a
//! polynomial in A of the summed cofactor degrees applied to the candidates. Per prime the
//! absorbed exponents strictly increase, so the summed degree is `O(n·max e)`, at most
//! `O(n²)`. The clean generators (F7) are built from the uncorrected `v_ℓ`, so nothing
//! compounds across levels.
//!
//! Exactness. Every constant comes from [`field_constants`] of A (`z`, `one`, the start vectors
//! of the presentation, the accumulators of `mat_vec` and `poly_mat_vec`, the leading 1 of
//! each relation polynomial), so for `PrimeField` and the GF(p^n) types the results are bound
//! as soon as A has a bound entry. A matrix of only unbound `Ring::zero()`/`one()` sentinels is
//! computed in Z, where `Field::inverse` succeeds only on ±1, and a non-unit pivot is `Err`.
//! When [`peel`] succeeds there, its result is the one over Z and, reduced, over GF(p) for
//! every p:
//! * Full depth: every push of the final span inverted a pivot ±1, so the matrix of the stored
//!   vectors `A^t·v_ℓ` is a lower triangular matrix with diagonal ±1 times an integer matrix
//!   that is unit upper triangular in pivot order: it is unimodular over Z. The verified
//!   relations hold over Z, the `f_ℓ` are monic, and the quotients of (F6) are exact divisions
//!   by monic polynomials, so they lie in Z\[x\]. Hence `R' = Y·D` over Z\[x\] with Y
//!   unimodular, the presentation `Z^n ≅ Z[x]^s / im R'` holds over Z and modulo every p, and
//!   the invariant factors modulo p are the `f_ℓ` modulo p.
//! * Minimal polynomial depth: the spin of `v_1` had pivots ±1, so `v_1, …, A^{D-1}·v_1` stay
//!   independent modulo p (a D×D minor is ±1), and `deg μ_p ≥ D`. And `f_1(A) = 0` over Z
//!   (`f_1 = μ_A` over Q, certified), so `μ_p | f_1 mod p`. Hence `μ_p = f_1 mod p`.
//!
//! For a rounding `Field` impl (`Real`) the checks use that type's `==` and `is_zero`, and no
//! accuracy is claimed. Termination holds for any `Field` impl: a span stores at most n
//! vectors, a merge must raise `deg f`, Euclid degrees strictly drop, and the coprime split is
//! capped.

use crate::krylov::{field_constants, krylov_presentation, mat_vec, poly_mat_vec, Echelon};
use crate::polynomial_matrix::div_rem_field;
use crate::Matrix;
use rustmath_core::{Field, MathError, Result};
use rustmath_polynomials::UnivariatePolynomial;

type Poly<F> = UnivariatePolynomial<F>;

fn internal(msg: &str) -> MathError {
    MathError::InvalidOperation(format!("internal: {msg}"))
}

/// How far [`peel`] goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeelDepth {
    /// The first level only: `f_1 = μ_A`.
    MinimalPolynomial,
    /// All levels: every invariant factor, and the full basis for [`Peeling::generators`].
    Full,
}

/// One cyclic summand found by [`peel`] (level ℓ): the maximal vector `v_ℓ` (uncorrected), its
/// relative minimal polynomial `f_ℓ`, the offset `m = dim U_{ℓ-1}` of its Krylov vectors in
/// [`Peeling::basis`], and the coordinates of `f_ℓ(A)·v_ℓ` on `basis[0..m]`.
struct PeelLevel<F: Field> {
    generator: Vec<F>,
    factor: Poly<F>,
    offset: usize,
    relation: Vec<F>,
}

/// The result of [`peel`]: the levels in the order found (`f_1 = μ_A` first, `f_{ℓ+1} | f_ℓ`)
/// and the stored basis `v_1, A·v_1, …, v_2, A·v_2, …` (complete only for [`PeelDepth::Full`]).
pub(crate) struct Peeling<F: Field> {
    levels: Vec<PeelLevel<F>>,
    basis: Vec<Vec<F>>,
}

/// An echelon basis together with the stored vectors `u_0, u_1, …` (in insertion order).
struct Span<F: Field> {
    ech: Echelon<F>,
    vecs: Vec<Vec<F>>,
    n: usize,
    z: F,
}

impl<F: Field> Span<F> {
    fn new(n: usize, z: F, one: F) -> Self {
        Span {
            ech: Echelon::new(n, z.clone(), one),
            vecs: Vec::new(),
            n,
            z,
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.ech.len(), self.vecs.len());
        self.vecs.len()
    }

    /// Keeps the first k stored vectors (and their echelon rows).
    fn truncate(&mut self, k: usize) {
        self.ech.truncate(k);
        self.vecs.truncate(k);
    }

    /// Whether `y` lies in the span of the first m stored vectors (exact over an exact field).
    fn reduces_to_zero_mod_prefix(&self, y: &[F], m: usize) -> bool {
        self.ech.reduce_prefix(y, m).iter().all(|x| x.is_zero())
    }

    /// Spins `y` modulo the stored vectors (F1): pushes `y, A·y, …, A^{k-1}·y` while they stay
    /// independent, verifies the final relation `A^k·y = Σ_s c_s·u_s` with `F`'s `==`, and
    /// returns `(k, c)` with `c` over all stored vectors (length `len()` after the call).
    /// `Err(InvalidOperation)` if the relation check fails or a push would exceed n vectors;
    /// errors from `Field::inverse` propagate.
    fn spin(&mut self, a: &Matrix<F>, y: Vec<F>) -> Result<(usize, Vec<F>)> {
        let start = self.len();
        let mut cur = y;
        loop {
            let (yr, lam) = self.ech.reduce(&cur);
            if yr.iter().all(|x| x.is_zero()) {
                let c = self.ech.coords(&lam)?;
                let mut s = vec![self.z.clone(); self.n];
                for (cs, u) in c.iter().zip(&self.vecs) {
                    if cs.is_zero() {
                        continue;
                    }
                    for (st, ut) in s.iter_mut().zip(u) {
                        *st = st.clone() + cs.clone() * ut.clone();
                    }
                }
                if s != cur {
                    return Err(internal("spin relation check failed"));
                }
                return Ok((self.len() - start, c));
            }
            if self.len() == self.n {
                // Impossible over an exact field; the cap keeps a rounding impl from looping.
                return Err(internal("spinning found more than n independent vectors"));
            }
            if !self.ech.push(yr, lam)? {
                return Err(internal("spinning stored a zero vector"));
            }
            let next = mat_vec(a, &cur, &self.z);
            self.vecs.push(cur);
            cur = next;
        }
    }
}

/// `x^k - Σ_{t<k} c_t·x^t` for `c_tail = (c_0, …, c_{k-1})`.
fn relation_polynomial<F: Field>(c_tail: &[F], z: &F, one: &F) -> Poly<F> {
    let mut v: Vec<F> = c_tail.iter().map(|x| z.clone() - x.clone()).collect();
    v.push(one.clone());
    Poly::new(v)
}

/// `Σ_t c_t·x^t` (the zero polynomial, built from `z`, for an empty `c`).
fn coefficient_polynomial<F: Field>(c: &[F], z: &F) -> Poly<F> {
    if c.is_empty() {
        Poly::new(vec![z.clone()])
    } else {
        Poly::new(c.to_vec())
    }
}

/// `p` divided by its leading coefficient (`p` itself if it is zero or monic).
fn monic<F: Field>(p: Poly<F>) -> Result<Poly<F>> {
    let Some(lc) = p.leading_coefficient() else {
        return Ok(p);
    };
    if lc.is_one() {
        return Ok(p);
    }
    let inv = lc.inverse()?;
    Ok(p.scalar_mul(&inv))
}

/// The monic gcd of `a` and `b` (not both zero), by Euclid with [`div_rem_field`], every
/// remainder made monic with `Field::inverse`. The degrees strictly drop, so it terminates for
/// any `Field` impl.
fn gcd_monic<F: Field>(a: &Poly<F>, b: &Poly<F>) -> Result<Poly<F>> {
    let (mut a, mut b) = (a.clone(), b.clone());
    while !b.is_zero() {
        let (_, r) = div_rem_field(&a, &b)?;
        a = b;
        b = monic(r)?;
    }
    monic(a)
}

/// `a / b` when the division is exact; `Err(InvalidOperation("internal: {what}"))` if a
/// remainder is left.
fn exact_quotient<F: Field>(a: &Poly<F>, b: &Poly<F>, what: &str) -> Result<Poly<F>> {
    let (q, r) = div_rem_field(a, b)?;
    if !r.is_zero() {
        return Err(internal(what));
    }
    Ok(q)
}

/// (F4): `(α, β)` with `α | f`, `β | a`, `gcd(α, β) = 1` and `αβ = lcm(f, a)`, for monic f
/// and a. At most `deg f + 2` gcds in the loop, the final one included (enough over an exact
/// field, see (F4)); `Err(InvalidOperation)` beyond that.
fn coprime_split<F: Field>(f: &Poly<F>, a: &Poly<F>) -> Result<(Poly<F>, Poly<F>)> {
    let g = gcd_monic(f, a)?;
    let mut alpha = f.clone();
    let mut beta = exact_quotient(a, &g, "gcd does not divide")?;
    let cap = f.degree().unwrap_or(0) + 2;
    for _ in 0..cap {
        let h = gcd_monic(&alpha, &beta)?;
        if h.degree() == Some(0) {
            return Ok((alpha, beta));
        }
        alpha = exact_quotient(&alpha, &h, "gcd does not divide")?;
        beta = beta * h;
    }
    Err(internal("coprime split did not converge"))
}

/// The current vector of a level: `(v, f = ann_U(v), coordinates of A^{deg f}·v on all stored
/// vectors)`; while it exists, the span holds `U + K(v)`.
type Current<F> = (Vec<F>, Poly<F>, Vec<F>);

/// One level of [`peel`]: a maximal vector modulo `U` = the first `m = span.len()` stored
/// vectors, among the `live` candidates (last to first). On `Some`, the span holds
/// `U + K(v)`. Candidates found in `U` before a current vector exists are removed from `live`.
/// `None` if every candidate lies in `U`.
fn maximal_vector<F: Field>(
    span: &mut Span<F>,
    a: &Matrix<F>,
    live: &mut Vec<Vec<F>>,
    z: &F,
    one: &F,
) -> Result<Option<PeelLevel<F>>> {
    let m = span.len();
    let n = span.n;
    let mut cur: Option<Current<F>> = None;
    // The exponent certificate: one h with h(A)·y ∈ U per live candidate y.
    let mut hs: Vec<Poly<F>> = Vec::new();
    let mut keep = vec![true; live.len()];
    for idx in (0..live.len()).rev() {
        let y = &live[idx];
        let Some((v, f, cv)) = cur.take() else {
            let (k, c) = span.spin(a, y.clone())?;
            if k == 0 {
                // y ∈ U ⊆ every later U; nothing was pushed.
                keep[idx] = false;
                continue;
            }
            let f = relation_polynomial(&c[m..], z, one);
            hs.push(f.clone());
            cur = Some((y.clone(), f, c));
            continue;
        };
        // y ∈ U + K(v) = V, or f(A)·y ∈ U: h = f.
        if span.len() == n || span.reduces_to_zero_mod_prefix(&poly_mat_vec(&f, a, y, z), m) {
            hs.push(f.clone());
            cur = Some((v, f, cv));
            continue;
        }
        // a = ann_U(y).
        span.truncate(m);
        let (ka, ca) = span.spin(a, y.clone())?;
        span.truncate(m);
        let ay = relation_polynomial(&ca[m..m + ka], z, one);
        hs.push(ay.clone());
        if div_rem_field(&f, &ay)?.1.is_zero() {
            // Unreachable over an exact field (a | f would put f(A)·y in U): restore U + K(v).
            let (kv, _) = span.spin(a, v.clone())?;
            if Some(kv) != f.degree() {
                return Err(internal("respun vector has the wrong annihilator"));
            }
            cur = Some((v, f, cv));
            continue;
        }
        // (F4), (F3).
        let (alpha, beta) = coprime_split(&f, &ay)?;
        let mut v2 = poly_mat_vec(
            &exact_quotient(&f, &alpha, "split factor does not divide")?,
            a,
            &v,
            z,
        );
        let w2 = poly_mat_vec(
            &exact_quotient(&ay, &beta, "split factor does not divide")?,
            a,
            y,
            z,
        );
        for (s, t) in v2.iter_mut().zip(w2) {
            *s = s.clone() + t;
        }
        let f2 = alpha * beta;
        if f2.degree() <= f.degree() {
            return Err(internal("merge did not grow"));
        }
        let (k2, c2) = span.spin(a, v2.clone())?;
        if Some(k2) != f2.degree() || relation_polynomial(&c2[m..], z, one) != f2 {
            return Err(internal("merged vector has the wrong annihilator"));
        }
        cur = Some((v2, f2, c2));
    }
    let mut idx = 0;
    live.retain(|_| {
        idx += 1;
        keep[idx - 1]
    });
    let Some((v, f, cv)) = cur else {
        return Ok(None);
    };
    for h in &hs {
        if !div_rem_field(&f, h)?.1.is_zero() {
            return Err(internal("exponent certificate failed"));
        }
    }
    Ok(Some(PeelLevel {
        generator: v,
        factor: f,
        offset: m,
        relation: cv[..m].to_vec(),
    }))
}

/// The cyclic decomposition of `F^n` (x acting as `a`), see the module docs. `a` must be
/// square (`Err(InvalidArgument)` from the presentation otherwise); n = 0 gives no levels.
/// `Err(InvalidOperation("internal: …"))` on a failed check; errors of `Field::inverse` (Z
/// semantics) and of the Krylov presentation propagate.
pub(crate) fn peel<F: Field>(a: &Matrix<F>, depth: PeelDepth) -> Result<Peeling<F>> {
    let n = a.rows();
    let presentation = krylov_presentation(a)?;
    if n == 0 {
        return Ok(Peeling {
            levels: Vec::new(),
            basis: Vec::new(),
        });
    }
    if presentation.generators.len() == 1 {
        // Cyclic: the verified relation of degree n is the certificate.
        let factor = presentation.relations.get(0, 0).clone();
        if factor.degree() != Some(n) {
            return Err(internal("cyclic presentation of the wrong degree"));
        }
        let generator = presentation.generators[0].clone();
        return Ok(Peeling {
            levels: vec![PeelLevel {
                generator,
                factor,
                offset: 0,
                relation: Vec::new(),
            }],
            basis: presentation.basis,
        });
    }
    let (z, one, _) = field_constants(a.data().iter());
    let mut live = presentation.generators;
    let mut span = Span::new(n, z.clone(), one.clone());
    let mut levels: Vec<PeelLevel<F>> = Vec::new();
    while span.len() < n {
        let Some(level) = maximal_vector(&mut span, a, &mut live, &z, &one)? else {
            return Err(internal("the candidates do not generate F^n"));
        };
        if let Some(last) = levels.last() {
            if !div_rem_field(&last.factor, &level.factor)?.1.is_zero() {
                return Err(internal("the factors do not form a divisibility chain"));
            }
        }
        levels.push(level);
        if depth == PeelDepth::MinimalPolynomial {
            break;
        }
    }
    Ok(Peeling {
        levels,
        basis: span.vecs,
    })
}

impl<F: Field> Peeling<F> {
    /// `f_s | … | f_1`, increasing (the order of `companion::invariant_factors`). Only `f_1`
    /// after [`PeelDepth::MinimalPolynomial`].
    pub(crate) fn invariant_factors(&self) -> Vec<Poly<F>> {
        self.levels.iter().rev().map(|l| l.factor.clone()).collect()
    }

    /// `f_1 = μ_A`; `None` when there are no levels (n = 0).
    pub(crate) fn minimal_polynomial(&self) -> Option<Poly<F>> {
        self.levels.first().map(|l| l.factor.clone())
    }

    /// The clean generators `g_s, …, g_1` of (F7), in the order of [`Self::invariant_factors`]:
    /// `f_ℓ(A)·g_ℓ = 0` and `F^n = ⊕_ℓ K(g_ℓ)`. `a` is the matrix given to [`peel`] (its
    /// [`field_constants`] give the bound zero). `Err(InvalidOperation)` if the peeling was not
    /// [`PeelDepth::Full`] (`basis.len() != n`), or `"internal: …"` if a division of (F6) is not
    /// exact.
    pub(crate) fn generators(&self, a: &Matrix<F>) -> Result<Vec<Vec<F>>> {
        let n = a.rows();
        if self.basis.len() != n {
            return Err(MathError::InvalidOperation(
                "the generators need the full peeling (PeelDepth::Full)".to_string(),
            ));
        }
        let (z, _, _) = field_constants(a.data().iter());
        let mut gs = Vec::with_capacity(self.levels.len());
        for (l, level) in self.levels.iter().enumerate() {
            let mut g = level.generator.clone();
            for earlier in &self.levels[..l] {
                let d = earlier
                    .factor
                    .degree()
                    .ok_or_else(|| internal("zero invariant factor"))?;
                let end = earlier.offset + d;
                if end > level.relation.len() {
                    return Err(internal("relation of the wrong length"));
                }
                let c = coefficient_polynomial(&level.relation[earlier.offset..end], &z);
                let tau = exact_quotient(&c, &level.factor, "relation not divisible")?;
                // deg τ < d, so A^t·v_i = basis[offset + t] is stored.
                for (t, coef) in tau.coefficients().iter().enumerate() {
                    if coef.is_zero() {
                        continue;
                    }
                    let u = &self.basis[earlier.offset + t];
                    for (gt, ut) in g.iter_mut().zip(u) {
                        *gt = gt.clone() - coef.clone() * ut.clone();
                    }
                }
            }
            gs.push(g);
        }
        gs.reverse();
        Ok(gs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::companion::companion_matrix;
    use rustmath_rationals::Rational;

    fn px(c: &[i64]) -> Poly<Rational> {
        Poly::new(c.iter().map(|&v| Rational::from_integer(v)).collect())
    }

    #[test]
    fn coprime_split_examples() {
        // gp: lcm(x^2*(x-1)^3, x^3*(x-1)*(x+1)) == (x-1)^3*x^3*(x+1) (1),
        //     lcm(x^3, x^5) == x^5 (1), lcm(x-1, x+1) == x^2-1 (1).
        let x = px(&[0, 1]);
        let xm1 = px(&[-1, 1]);
        let xp1 = px(&[1, 1]);
        let pow = |p: &Poly<Rational>, e: usize| (0..e).fold(px(&[1]), |acc, _| acc * p.clone());
        let f1 = pow(&x, 2) * pow(&xm1, 3);
        let cases = [
            (
                f1.clone(),
                pow(&x, 3) * xm1.clone() * xp1.clone(),
                pow(&xm1, 3),
                pow(&x, 3) * xp1.clone(),
            ),
            (pow(&x, 3), pow(&x, 5), px(&[1]), pow(&x, 5)),
            (xm1.clone(), xp1.clone(), xm1.clone(), xp1.clone()),
            (f1.clone(), f1.clone(), f1.clone(), px(&[1])),
        ];
        for (f, a, alpha_expected, beta_expected) in cases {
            let (alpha, beta) = coprime_split(&f, &a).unwrap();
            assert_eq!(alpha, alpha_expected, "f = {f}, a = {a}");
            assert_eq!(beta, beta_expected, "f = {f}, a = {a}");
            assert!(div_rem_field(&f, &alpha).unwrap().1.is_zero());
            assert!(div_rem_field(&a, &beta).unwrap().1.is_zero());
            assert_eq!(gcd_monic(&alpha, &beta).unwrap(), px(&[1]));
            // αβ = lcm(f, a) = f·a / gcd(f, a).
            let g = gcd_monic(&f, &a).unwrap();
            let lcm = exact_quotient(&(f.clone() * a.clone()), &g, "lcm").unwrap();
            assert_eq!(alpha * beta, lcm);
        }
    }

    #[test]
    fn merged_vector_has_lcm_annihilator() {
        // A = C(x^2+1) ⊕ C((x-1)^2) over Q; v = e_1 has annihilator x^2 + 1, y = e_3 has
        // (x - 1)^2; the merge (F3) has the product. gp:
        // minpoly(matconcat(matdiagonal([matcompanion(x^2+1), matcompanion((x-1)^2)])))
        //     == (x^2+1)*(x-1)^2 (1)
        let f = px(&[1, 0, 1]);
        let a_y = px(&[1, -2, 1]);
        let c1 = companion_matrix(&f).unwrap();
        let c2 = companion_matrix(&a_y).unwrap();
        let mut a = Matrix::<Rational>::zeros(4, 4);
        for i in 0..2 {
            for j in 0..2 {
                a.set(i, j, c1.get(i, j).unwrap().clone()).unwrap();
                a.set(i + 2, j + 2, c2.get(i, j).unwrap().clone()).unwrap();
            }
        }
        let (z, one, _) = field_constants(a.data().iter());
        let e = |i: usize| {
            let mut v = vec![z.clone(); 4];
            v[i] = one.clone();
            v
        };
        // The annihilators of v and y, by spinning over U = 0.
        for (v, ann) in [(e(0), &f), (e(2), &a_y)] {
            let mut span = Span::new(4, z.clone(), one.clone());
            let (k, c) = span.spin(&a, v).unwrap();
            assert_eq!(k, 2);
            assert_eq!(&relation_polynomial(&c, &z, &one), ann);
        }
        let (alpha, beta) = coprime_split(&f, &a_y).unwrap();
        assert_eq!((alpha.clone(), beta.clone()), (f.clone(), a_y.clone()));
        let mut v2 = poly_mat_vec(&exact_quotient(&f, &alpha, "q").unwrap(), &a, &e(0), &z);
        let w2 = poly_mat_vec(&exact_quotient(&a_y, &beta, "q").unwrap(), &a, &e(2), &z);
        for (s, t) in v2.iter_mut().zip(w2) {
            *s = s.clone() + t;
        }
        let mut span = Span::new(4, z.clone(), one.clone());
        let (k, c) = span.spin(&a, v2).unwrap();
        assert_eq!(k, 4);
        assert_eq!(relation_polynomial(&c, &z, &one), f * a_y);
    }

    #[test]
    fn peel_small_cases() {
        // A = diag(2, 2, 3): gp matfrobenius(matdiagonal([2,2,3]),1) = [x^2 - 5*x + 6, x - 2];
        // the zero 2×2: [x, x]; [5]: [x - 5]; the 0×0 matrix has no levels.
        let qm = |n: usize, v: &[i64]| {
            Matrix::from_vec(n, n, v.iter().map(|&x| Rational::from_integer(x)).collect()).unwrap()
        };
        let p = peel(&qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3]), PeelDepth::Full).unwrap();
        assert_eq!(p.invariant_factors(), vec![px(&[-2, 1]), px(&[6, -5, 1])]);
        assert_eq!(p.minimal_polynomial(), Some(px(&[6, -5, 1])));
        let p = peel(
            &qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3]),
            PeelDepth::MinimalPolynomial,
        )
        .unwrap();
        assert_eq!(p.minimal_polynomial(), Some(px(&[6, -5, 1])));
        assert!(matches!(
            p.generators(&qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3])),
            Err(MathError::InvalidOperation(_))
        ));
        let p = peel(&qm(2, &[0, 0, 0, 0]), PeelDepth::Full).unwrap();
        assert_eq!(p.invariant_factors(), vec![px(&[0, 1]), px(&[0, 1])]);
        let p = peel(&qm(1, &[5]), PeelDepth::Full).unwrap();
        assert_eq!(p.invariant_factors(), vec![px(&[-5, 1])]);
        let e = Matrix::<Rational>::from_vec(0, 0, vec![]).unwrap();
        let p = peel(&e, PeelDepth::Full).unwrap();
        assert!(p.invariant_factors().is_empty());
        assert_eq!(p.minimal_polynomial(), None);
        assert!(p.generators(&e).unwrap().is_empty());
        let r = Matrix::from_vec(2, 3, vec![Rational::from_integer(1); 6]).unwrap();
        assert!(matches!(
            peel(&r, PeelDepth::Full),
            Err(MathError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rounding_field_peel_terminates() {
        // The two Real matrices of krylov::rounding_field_terminates: peel returns Ok with
        // total degree n, or Err(InvalidOperation); it never hangs (no accuracy claim).
        use rustmath_reals::Real;
        for (n, mult) in [(5usize, 7usize), (8, 7919)] {
            let a = Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|i| Real::from(((i * mult) % 11) as f64 - 5.0))
                    .collect(),
            )
            .unwrap();
            for depth in [PeelDepth::Full, PeelDepth::MinimalPolynomial] {
                match peel(&a, depth) {
                    Ok(p) => {
                        let total: usize = p
                            .invariant_factors()
                            .iter()
                            .map(|f| f.degree().unwrap())
                            .sum();
                        if depth == PeelDepth::Full {
                            assert_eq!(total, n);
                        } else {
                            assert!(total <= n);
                        }
                    }
                    Err(e) => assert!(matches!(e, MathError::InvalidOperation(_)), "{e:?}"),
                }
            }
        }
    }
}

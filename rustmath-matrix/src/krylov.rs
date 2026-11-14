//! Krylov presentation of `F^n` as an `F[x]`-module, `x` acting as a square matrix `A`.
//!
//! MAGMA Handbook ch. 26 §26.11 (`MinimalPolynomial`; also ch. 59 §59.4.3) and §26.12.2
//! (`InvariantFactors`, `RationalForm`), digest `rustmath-lava/docs/Chapter_26_Matrices.md`.
//!
//! Algorithm.
//! * Spinning (Krylov iteration): start vectors `w_0, w_1, …` are spun to `w, A·w, A²·w, …`
//!   modulo the span of the vectors already collected, giving a block upper-triangular matrix
//!   with companion blocks on the diagonal. W. Keller-Gehrig, *Fast algorithms for the
//!   characteristic polynomial*, Theoret. Comput. Sci. 36 (1985) 309–317 \[KG85\]. The spinning
//!   step is also the basis of Steel's algorithm \[Ste97\] (A. Steel, J. Symbolic Comp. 24 (1997)
//!   409–432), which MAGMA uses; the combination step here is not \[Ste97\]'s.
//! * Combination: block `j` ends with a relation `A^{k_j}·w_{s_j} = Σ_{i ≤ j} c_i(A)·w_{s_i}`;
//!   the relations form the columns of an r×r upper-triangular relation matrix `R` over `F[x]`
//!   with `R[j][j]` monic of degree `k_j` and `Σ k_j = n`. Then `F^n ≅ F[x]^r / im(R)`:
//!   `φ: F[x]^r → F^n`, `e_i ↦ w_{s_i}`, is onto, `im R ⊆ ker φ`, and
//!   `dim_F F[x]^r / im R = deg det R = n`. So the invariant factors of `A` are the non-constant
//!   diagonal entries of the Smith form of `R` (presentation theorem: D. S. Dummit and
//!   R. M. Foote, *Abstract Algebra*, 3rd ed., §12.1–12.2; N. Jacobson, *Basic Algebra I*, 2nd
//!   ed., §3.8–3.10). `companion.rs` uses the presentation in two ways. Its main path
//!   (`frobenius.rs`) needs no Smith form: it peels maximal vectors off the generators `w_{s_j}`
//!   (cyclic decomposition), and for r = 1 the presentation already is the answer. Its first
//!   fallback takes the Smith form of `R`, without transforms for the invariant factors, and
//!   with `P⁻¹` (up to a unit per column) for the rational form, whose generators it computes
//!   from the reduction of the columns modulo `im R` and the Krylov basis `u_0, …, u_{n-1}`
//!   kept in [`KrylovPresentation::basis`].
//! * Not implemented: A. Storjohann, *An O(n³) algorithm for the Frobenius normal form*, ISSAC
//!   1998, 101–105 \[Sto98\] (zigzag form).
//!
//! Cost: spinning is `O(n³)` field operations (at most n matrix-vector products and n
//! reductions), coordinates and self-checks `O(r·n²)`. Over Q every reduced-row entry is a
//! quotient of minors of the Krylov matrix (Cramer), so its size is polynomial in n. Usually
//! `r` is the number of invariant factors (1 for a random matrix), but the start basis `W`
//! (columns `w_j`) is fixed and unimodular, and spinning `w_j` under `A` is spinning `e_j` under
//! `B = W⁻¹·A·W`. So every `A = W·B·W⁻¹` with `B` upper triangular gives `r = n` and
//! `R = xI - B`. For such an `R` the Smith step is as slow as the classical reduction of the
//! n×n `xI - A`, and no bound is claimed for it; it only runs in the fallback. The peeling of
//! `frobenius.rs` handles r = n in `O(n⁴)` field operations (its bound is `O((r + 1)·n³)`, see
//! there).
//!
//! Exactness. Every constant the algorithm creates (start-vector entries, the stored pivot
//! entries, the monic leading coefficients of `R`, the accumulators of [`mat_vec`]) is built in
//! `A`'s own field: `z = Σ a_ij·0`, `one = z + 1`, `neg_one = z - 1`. For `PrimeField` and the
//! GF(p^n) element types the bare `Ring::zero()`/`Ring::one()` are modulus-0 sentinels, and
//! sentinel-with-sentinel arithmetic stays in Z (`1 + 1 = 2`, a hidden zero in GF(2) and never
//! invertible in Z); built from `z` they are bound as soon as `A` has a bound entry.
//!
//! A matrix with only sentinel entries is computed in Z semantics, where only ±1 are
//! invertible. The ±1 start vectors then create pivots (±2, ±3, …) that are not entries of `A`,
//! for example for every cyclic shift or `J_n(1)` with n ≥ 3, and `Field::inverse` rejects
//! them: [`krylov_presentation`] returns that `Err`, never a wrong value. The public functions
//! in `companion.rs` then fall back (past the peeling and the Smith form of `R`, which both
//! need this presentation) to the classical Smith form of `xI - A`, whose constant pivots are
//! entries of `A`. Whenever a computation in Z semantics succeeds (any path), every pivot it
//! inverted is ±1, so every step is unimodular over Z (Krylov basis, peeling) and Z[x] (Smith
//! steps), and the change of basis `T` of the rational form has `det T = ±1` (its certificate
//! falls back to the division-free determinant when its elimination meets a non-unit pivot):
//! the result is the one over Z, and its reduction is the result over GF(p) for every p
//! (`frobenius.rs`, "Exactness", gives the argument for the peeling).

use crate::polynomial_matrix::PolynomialMatrix;
use crate::Matrix;
use rustmath_core::{Field, MathError, Result};
use rustmath_polynomials::UnivariatePolynomial;

/// A presentation of `F^n` (x acting as `A`) by r generators and an r×r relation matrix.
pub(crate) struct KrylovPresentation<F: Field> {
    /// The generators `w_{s_1}, …, w_{s_r}` (the start vectors of the blocks), length n each.
    pub(crate) generators: Vec<Vec<F>>,
    /// r×r over `F[x]`, upper triangular, column j = relation j; `R[j][j]` monic of degree
    /// `k_j ≥ 1` with `Σ k_j = n`; `F^n ≅ F[x]^r / im(R)` via `e_j ↦ generators[j]`.
    pub(crate) relations: PolynomialMatrix<F>,
    /// The Krylov basis `u_0, …, u_{n-1}` of `F^n`, block by block: block j contributes
    /// `w, A·w, …, A^{k_j-1}·w` (`w = generators[j]`, `k_j = deg R[j][j]`), so
    /// `u_{off_j+t} = A^t·w` with `off_j = k_0 + … + k_{j-1}`.
    pub(crate) basis: Vec<Vec<F>>,
}

/// The constants `(z, one, neg_one)` of the field of the entries `xs`: `z = Σ x·0`, bound to
/// that field when any entry is bound (`PrimeField`, GF(p^n)), plain 0 otherwise (Q).
pub(crate) fn field_constants<'a, F: Field + 'a>(xs: impl Iterator<Item = &'a F>) -> (F, F, F) {
    let z = xs.fold(F::zero(), |s, x| s + x.clone() * F::zero());
    let one = z.clone() + F::one();
    let neg_one = z.clone() - F::one();
    (z, one, neg_one)
}

/// The sign sequence of the start vectors: a 64-bit LCG (Knuth's MMIX constants), sign = top
/// bit. Local and fixed so that results are reproducible across platforms; correctness never
/// depends on it (it only affects speed and which valid change of basis is returned).
struct SignSeq(u64);

impl SignSeq {
    fn new() -> Self {
        SignSeq(0x5DEE_CE66D)
    }

    fn next_is_plus(&mut self) -> bool {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 63 == 0
    }
}

/// `w_j`: `w_j[i] = 0` for `i < j`, `w_j[j] = 1`, `w_j[i] = ±1` for `i > j` (consumes
/// `n - j - 1` signs). The `w_j` form a unit-lower-triangular, hence unimodular, basis.
fn start_vector<F: Field>(n: usize, j: usize, k: &(F, F, F), seq: &mut SignSeq) -> Vec<F> {
    let (z, one, neg_one) = k;
    let mut w = vec![z.clone(); n];
    w[j] = one.clone();
    for wi in w.iter_mut().skip(j + 1) {
        *wi = if seq.next_is_plus() {
            one.clone()
        } else {
            neg_one.clone()
        };
    }
    w
}

/// `A·v` (A square), reading `a.data()` directly and skipping zero entries. Each entry starts
/// from `z` (the `z` of [`field_constants`] of `A`), so the result is bound to `A`'s field even
/// where every product is skipped.
pub(crate) fn mat_vec<F: Field>(a: &Matrix<F>, v: &[F], z: &F) -> Vec<F> {
    let n = a.cols();
    let d = a.data();
    (0..a.rows())
        .map(|i| {
            let mut acc = z.clone();
            for (aik, vk) in d[i * n..(i + 1) * n].iter().zip(v) {
                if !aik.is_zero() && !vk.is_zero() {
                    acc = acc + aik.clone() * vk.clone();
                }
            }
            acc
        })
        .collect()
}

/// `p(A)·w`, by Horner's rule on vectors: `v ← A·v + c·w` for the coefficients `c` of `p` from
/// the top down. Every entry starts from `z` (as in [`mat_vec`]), so the result is bound to
/// `A`'s field. `deg p + 1` matrix-vector products.
pub(crate) fn poly_mat_vec<F: Field>(
    p: &UnivariatePolynomial<F>,
    a: &Matrix<F>,
    w: &[F],
    z: &F,
) -> Vec<F> {
    let mut v = vec![z.clone(); a.rows()];
    for c in p.coefficients().iter().rev() {
        v = mat_vec(a, &v, z);
        if !c.is_zero() {
            for (vt, wt) in v.iter_mut().zip(w) {
                *vt = vt.clone() + c.clone() * wt.clone();
            }
        }
    }
    v
}

/// The n start vectors `w_0, …, w_{n-1}` that [`krylov_presentation`] spins, for tests that
/// build its worst case `W·B·W⁻¹` (`k` from [`field_constants`]).
#[cfg(test)]
pub(crate) fn start_basis<F: Field>(n: usize, k: &(F, F, F)) -> Vec<Vec<F>> {
    let mut seq = SignSeq::new();
    (0..n).map(|j| start_vector(n, j, k, &mut seq)).collect()
}

/// An incremental echelon basis of the span of the vectors `u_0, u_1, …` added so far.
///
/// * Row `i` is `u_i` reduced against rows `0..i`, scaled so that `rows[i][piv[i]] = one`
///   (set explicitly).
/// * [`Echelon::reduce`] runs over the stored rows **in insertion order**; reducing by row `i`
///   with `c = y[piv_i]` updates `y[t] -= c·rows[i][t]` for `t ≠ piv_i` (when `c ≠ 0`) and
///   sets `y[piv_i] = z` explicitly. Since `rows[s][piv_i] = 0` for `s > i` (row `s` was itself
///   reduced by row `i`), later rows never disturb an earlier pivot, so a reduced vector is zero
///   at every pivot position, and its first non-zero entry (the pivot of a new row) is a new
///   position. Over an exact field this gives at most n rows, and every reduced vector is zero
///   once n rows are stored. (A rounding impl can break `rows[s][piv_i] = 0` through the
///   scaling, so [`krylov_presentation`] also caps the number of stored rows at n.)
/// * `lam[i]` and `alpha[i]` record `u_i = Σ_{s<i} lam[i][s]·rows[s] + alpha[i]·rows[i]`, from
///   which [`Echelon::coords`] recovers coordinates in the `u` basis.
/// * Rows only refer to earlier rows (`lam[i]` has length i, and pushing row i changes no row
///   `< i`), so [`Echelon::truncate`] to k rows leaves an echelon basis of the span of
///   `u_0, …, u_{k-1}`, and [`Echelon::reduce_prefix`] against the first m rows decides
///   membership in the span of `u_0, …, u_{m-1}` while more rows are stored (`frobenius.rs`).
pub(crate) struct Echelon<F: Field> {
    n: usize,
    z: F,
    one: F,
    rows: Vec<Vec<F>>,
    piv: Vec<usize>,
    lam: Vec<Vec<F>>,
    alpha: Vec<F>,
}

impl<F: Field> Echelon<F> {
    pub(crate) fn new(n: usize, z: F, one: F) -> Self {
        Echelon {
            n,
            z,
            one,
            rows: Vec::new(),
            piv: Vec::new(),
            lam: Vec::new(),
            alpha: Vec::new(),
        }
    }

    /// The number of stored rows.
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Drops the rows `k..` (nothing if `k >= len`). The remaining rows, pivots and records
    /// are unchanged and consistent, because row i refers only to rows `< i`.
    pub(crate) fn truncate(&mut self, k: usize) {
        self.rows.truncate(k);
        self.piv.truncate(k);
        self.lam.truncate(k);
        self.alpha.truncate(k);
    }

    /// `y` reduced against the first `m` stored rows (in insertion order), and the multipliers
    /// used (one per row).
    fn reduce_upto(&self, y: &[F], m: usize) -> (Vec<F>, Vec<F>) {
        let mut y = y.to_vec();
        let mut lam = Vec::with_capacity(m);
        for (r, &p) in self.rows.iter().zip(&self.piv).take(m) {
            let c = y[p].clone();
            if !c.is_zero() {
                for t in 0..self.n {
                    if t != p && !r[t].is_zero() {
                        y[t] = y[t].clone() - c.clone() * r[t].clone();
                    }
                }
            }
            y[p] = self.z.clone();
            lam.push(c);
        }
        (y, lam)
    }

    /// `y` reduced against the stored rows, and the multipliers used (one per stored row).
    pub(crate) fn reduce(&self, y: &[F]) -> (Vec<F>, Vec<F>) {
        self.reduce_upto(y, self.rows.len())
    }

    /// `y` reduced against the first `m` stored rows only (all rows if `m >= len`). The first m
    /// rows are an echelon basis of the span of the first m stored vectors (later rows never
    /// change them), so over an exact field the result is zero iff `y` lies in that span.
    pub(crate) fn reduce_prefix(&self, y: &[F], m: usize) -> Vec<F> {
        self.reduce_upto(y, m).0
    }

    /// Stores the reduced vector `y` (of some `u`) with its multipliers `lam` from
    /// [`Echelon::reduce`]. `Ok(false)` (nothing stored) if `y` is zero; errors from
    /// `Field::inverse` of the pivot propagate.
    pub(crate) fn push(&mut self, y: Vec<F>, lam: Vec<F>) -> Result<bool> {
        let Some(p) = y.iter().position(|x| !x.is_zero()) else {
            return Ok(false);
        };
        let alpha = y[p].clone();
        let inv = alpha.inverse()?;
        let mut row: Vec<F> = y.into_iter().map(|x| x * inv.clone()).collect();
        row[p] = self.one.clone();
        self.rows.push(row);
        self.piv.push(p);
        self.lam.push(lam);
        self.alpha.push(alpha);
        Ok(true)
    }

    /// The coordinates `c` with `y = Σ_s c_s·u_s`, for a `y` that reduced to zero with
    /// multipliers `lam` (so `y = Σ_i lam_i·rows[i]`), by back substitution from the top row.
    pub(crate) fn coords(&self, lam: &[F]) -> Result<Vec<F>> {
        let m = lam.len();
        let mut beta = lam.to_vec();
        let mut c = vec![self.z.clone(); m];
        for i in (0..m).rev() {
            if beta[i].is_zero() {
                continue;
            }
            let ci = beta[i].clone() * self.alpha[i].inverse()?;
            for (s, l) in self.lam[i].iter().enumerate() {
                if !l.is_zero() {
                    beta[s] = beta[s].clone() - ci.clone() * l.clone();
                }
            }
            c[i] = ci;
        }
        Ok(c)
    }
}

/// Whether the vectors `vs` (length n each) are linearly independent, by the exact incremental
/// elimination of [`Echelon`] (pivots inverted with `Field::inverse`, whose errors propagate;
/// no `/`).
pub(crate) fn linearly_independent<F: Field>(vs: &[Vec<F>]) -> Result<bool> {
    let n = vs.first().map_or(0, |v| v.len());
    let (z, one, _) = field_constants(vs.iter().flatten());
    let mut e = Echelon::new(n, z, one);
    for v in vs {
        if v.len() != n {
            return Err(MathError::DimensionMismatch(
                "vectors of different lengths".to_string(),
            ));
        }
        let (y, lam) = e.reduce(v);
        if !e.push(y, lam)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The Krylov presentation of `A` (see the module docs).
///
/// * `Err(InvalidArgument)` if `A` is not square; `r = 0` for the 0×0 matrix.
/// * Each relation is verified before it is used: at the end of block `j`, `Σ_s c_s·u_s` is
///   recomputed and compared with `A^{k_j}·w_{s_j}` with `F`'s `==` (exact over Q, GF(p),
///   number fields); a mismatch gives
///   `Err(InvalidOperation("internal: Krylov relation check failed"))`. Over an exact field a
///   mismatch means a broken `Field` impl. For a rounding impl the comparison is that type's own
///   (`Real`: absolute tolerance 1e-10): small well-conditioned inputs can pass, and beyond
///   that the check fails in practice (measured: `Real`, 8×8, entries in [-5, 5]); no accuracy
///   claim either way. Termination holds for any `Field` impl: at most n vectors are stored,
///   and an (n+1)-th one is `Err(InvalidOperation)`.
/// * Errors from `Field::inverse` propagate.
pub(crate) fn krylov_presentation<F: Field>(a: &Matrix<F>) -> Result<KrylovPresentation<F>> {
    if !a.is_square() {
        return Err(MathError::InvalidArgument(
            "Krylov presentation needs a square matrix".to_string(),
        ));
    }
    let n = a.rows();
    let k = field_constants(a.data().iter());
    let (z, one) = (k.0.clone(), k.1.clone());
    let mut seq = SignSeq::new();
    let mut ech = Echelon::new(n, z.clone(), one.clone());
    // The collected vectors u_0, u_1, …: per block w, A·w, …, A^{k-1}·w.
    let mut us: Vec<Vec<F>> = Vec::with_capacity(n);
    let mut generators: Vec<Vec<F>> = Vec::new();
    // Per block: (offset of its first u, degree k, coordinates of A^k·w in the u basis).
    let mut blocks: Vec<(usize, usize, Vec<F>)> = Vec::new();

    for j in 0..n {
        if us.len() == n {
            break;
        }
        let w = start_vector(n, j, &k, &mut seq);
        let (mut yr, mut lam) = ech.reduce(&w);
        let off = us.len();
        let mut y = w.clone();
        loop {
            if yr.iter().all(|x| x.is_zero()) {
                // y lies in the span: w itself (skip it) or A^k·w (end of the block).
                if us.len() == off {
                    break;
                }
                let c = ech.coords(&lam)?;
                let mut s = vec![z.clone(); n];
                for (cs, u) in c.iter().zip(&us) {
                    if cs.is_zero() {
                        continue;
                    }
                    for (st, ut) in s.iter_mut().zip(u) {
                        *st = st.clone() + cs.clone() * ut.clone();
                    }
                }
                if s != y {
                    return Err(MathError::InvalidOperation(
                        "internal: Krylov relation check failed".to_string(),
                    ));
                }
                blocks.push((off, us.len() - off, c));
                generators.push(w);
                break;
            }
            if us.len() == n {
                // Impossible over an exact field (the w_j are a basis and every stored row has
                // a new pivot position); the cap keeps a rounding impl from looping.
                return Err(MathError::InvalidOperation(
                    "internal: Krylov spinning found more than n independent vectors".to_string(),
                ));
            }
            ech.push(yr, lam)?;
            us.push(y.clone());
            y = mat_vec(a, &y, &z);
            (yr, lam) = ech.reduce(&y);
        }
    }
    if us.len() != n {
        // The w_j are a basis and every stored row has a new pivot position.
        return Err(MathError::InvalidOperation(
            "internal: Krylov spinning did not span F^n".to_string(),
        ));
    }

    let r = blocks.len();
    let mut data = Vec::with_capacity(r * r);
    for i in 0..r {
        for (j, (_, _, c)) in blocks.iter().enumerate() {
            let entry = if i > j {
                UnivariatePolynomial::new(vec![z.clone()])
            } else {
                let (off_i, k_i, _) = &blocks[i];
                let mut coeffs: Vec<F> = (0..*k_i)
                    .map(|t| z.clone() - c[off_i + t].clone())
                    .collect();
                if i == j {
                    coeffs.push(one.clone());
                }
                UnivariatePolynomial::new(coeffs)
            };
            data.push(entry);
        }
    }
    Ok(KrylovPresentation {
        generators,
        relations: PolynomialMatrix::new(r, r, data)?,
        basis: us,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;
    use rustmath_rationals::Rational;

    fn px(c: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(c.iter().map(|&v| Rational::from_integer(v)).collect())
    }

    fn qm(n: usize, v: &[i64]) -> Matrix<Rational> {
        Matrix::from_vec(n, n, v.iter().map(|&x| Rational::from_integer(x)).collect()).unwrap()
    }

    /// `p(A)·w`, by Horner's rule.
    fn eval(
        p: &UnivariatePolynomial<Rational>,
        a: &Matrix<Rational>,
        w: &[Rational],
    ) -> Vec<Rational> {
        let z = Rational::from_integer(0);
        let mut v = vec![z.clone(); w.len()];
        for c in p.coefficients().iter().rev() {
            v = mat_vec(a, &v, &z);
            for (vt, wt) in v.iter_mut().zip(w) {
                *vt = vt.clone() + c.clone() * wt.clone();
            }
        }
        v
    }

    /// Checks the presentation independently of the internal self-check: R is upper
    /// triangular with monic diagonal entries of total degree n, and every column of R is a
    /// relation, `Σ_i R[i][j](A)·w_i = 0`.
    fn check_presentation(a: &Matrix<Rational>) -> usize {
        let n = a.rows();
        let p = krylov_presentation(a).unwrap();
        let r = p.relations.rows();
        assert_eq!(p.relations.cols(), r);
        assert_eq!(p.generators.len(), r);
        let mut total = 0;
        for j in 0..r {
            let d = p.relations.get(j, j);
            assert_eq!(d.leading_coefficient(), Some(&Rational::from_integer(1)));
            total += d.degree().unwrap();
            for i in (j + 1)..r {
                assert!(p.relations.get(i, j).is_zero(), "R not upper triangular");
            }
            let mut s = vec![Rational::from_integer(0); n];
            for i in 0..=j {
                for (st, vt) in s
                    .iter_mut()
                    .zip(eval(p.relations.get(i, j), a, &p.generators[i]))
                {
                    *st = st.clone() + vt;
                }
            }
            assert!(
                s.iter().all(|x| x.is_zero()),
                "column {j} of R is not a relation"
            );
        }
        assert_eq!(total, n);
        r
    }

    #[test]
    fn presentation_relations_hold() {
        // Dense, diagonal with repeated entries, nilpotent, scalar, zero.
        assert_eq!(check_presentation(&qm(3, &[1, 2, 0, 0, 1, 3, 4, 0, 1])), 1);
        assert!(check_presentation(&qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3])) >= 2);
        assert!(check_presentation(&qm(3, &[0, 1, 0, 0, 0, 0, 0, 0, 0])) >= 2);
        assert_eq!(check_presentation(&qm(3, &[5, 0, 0, 0, 5, 0, 0, 0, 5])), 3);
        assert_eq!(check_presentation(&qm(2, &[0, 0, 0, 0])), 2);
        assert_eq!(check_presentation(&qm(1, &[7])), 1);
    }

    #[test]
    fn presentation_empty_and_non_square() {
        let e = Matrix::<Rational>::from_vec(0, 0, vec![]).unwrap();
        let p = krylov_presentation(&e).unwrap();
        assert_eq!(p.relations.rows(), 0);
        assert!(p.generators.is_empty());
        let r = Matrix::from_vec(2, 3, vec![Rational::from_integer(1); 6]).unwrap();
        assert!(matches!(
            krylov_presentation(&r),
            Err(MathError::InvalidArgument(_))
        ));
    }

    #[test]
    fn start_vectors_depend_only_on_n_and_j() {
        // w_j consumes n - j - 1 signs, so the basis is fixed by n; it is unit lower
        // triangular with ±1 below the diagonal.
        let k = field_constants::<Rational>(std::iter::empty());
        let mut seq = SignSeq::new();
        let ws: Vec<Vec<Rational>> = (0..5).map(|j| start_vector(5, j, &k, &mut seq)).collect();
        for (j, w) in ws.iter().enumerate() {
            for (i, x) in w.iter().enumerate() {
                if i < j {
                    assert!(x.is_zero());
                } else if i == j {
                    assert_eq!(x, &Rational::from_integer(1));
                } else {
                    assert!(*x == Rational::from_integer(1) || *x == Rational::from_integer(-1));
                }
            }
        }
        assert!(linearly_independent(&ws).unwrap());
    }

    #[test]
    fn linearly_independent_detects_dependence() {
        let v = |c: &[i64]| {
            c.iter()
                .map(|&x| Rational::from_integer(x))
                .collect::<Vec<_>>()
        };
        assert!(linearly_independent(&[v(&[1, 2, 3]), v(&[0, 1, 1]), v(&[1, 0, 0])]).unwrap());
        assert!(!linearly_independent(&[v(&[1, 2, 3]), v(&[0, 1, 1]), v(&[1, 3, 4])]).unwrap());
        assert!(!linearly_independent(&[v(&[1, 2]), v(&[0, 0])]).unwrap());
        assert!(linearly_independent::<Rational>(&[]).unwrap());
    }

    #[test]
    fn fixed_start_basis_worst_case_has_r_equal_n() {
        // The documented worst case: with W the start basis (columns w_j) and B upper
        // triangular, A = W·B·W⁻¹ spins like B on the unit vectors, so r = n and R = xI - B.
        // B has the distinct eigenvalues 1..6, so its only invariant factor is
        // prod(x - i) (gp: Vecrev(prod(i=1,6,x-i)) = [720, -1764, 1624, -735, 175, -21, 1]).
        let n = 6;
        let k = field_constants::<Rational>(std::iter::empty());
        let mut seq = SignSeq::new();
        let ws: Vec<Vec<Rational>> = (0..n).map(|j| start_vector(n, j, &k, &mut seq)).collect();
        let w = Matrix::from_vec(
            n,
            n,
            (0..n * n).map(|idx| ws[idx % n][idx / n].clone()).collect(),
        )
        .unwrap();
        let mut b = vec![0i64; n * n];
        for i in 0..n {
            b[i * n + i] = i as i64 + 1;
            for j in (i + 1)..n {
                b[i * n + j] = ((3 * i + 5 * j) % 7) as i64 - 3;
            }
        }
        let bm = qm(n, &b);
        let a = w
            .mul(&bm)
            .unwrap()
            .mul(&w.inverse().unwrap().unwrap())
            .unwrap();
        assert_eq!(check_presentation(&a), n);
        let p = krylov_presentation(&a).unwrap();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    px(&[-b[i * n + j], 1])
                } else {
                    px(&[-b[i * n + j]])
                };
                assert_eq!(
                    p.relations.get(i, j),
                    &expected,
                    "R != xI - B at ({i}, {j})"
                );
            }
        }
        assert_eq!(
            crate::companion::invariant_factors(&a).unwrap(),
            vec![px(&[720, -1764, 1624, -735, 175, -21, 1])]
        );
    }

    #[test]
    fn echelon_truncate_and_prefix() {
        // Four vectors u0..u3 in Q^4, pushed in order; then reduce_prefix decides membership
        // in span(u0) and span(u0, u1) while all four rows are stored, truncate(2) keeps an
        // echelon basis of span(u0, u1), and coords still recovers coordinates.
        let v = |c: &[i64]| {
            c.iter()
                .map(|&x| Rational::from_integer(x))
                .collect::<Vec<_>>()
        };
        let us = [
            v(&[2, 1, 0, 3]),
            v(&[0, 1, 1, 1]),
            v(&[1, 0, 2, 0]),
            v(&[0, 0, 1, 5]),
        ];
        let (z, one, _) = field_constants::<Rational>(std::iter::empty());
        let mut e = Echelon::new(4, z.clone(), one);
        for u in &us {
            let (y, lam) = e.reduce(u);
            assert!(e.push(y, lam).unwrap());
        }
        assert_eq!(e.len(), 4);
        let is_zero = |y: Vec<Rational>| y.iter().all(|x| x.is_zero());
        // 3·u0 is in span(u0); u0 + 2·u1 = (2, 3, 2, 5) is in span(u0, u1) but not span(u0).
        let w = v(&[2, 3, 2, 5]);
        assert!(is_zero(e.reduce_prefix(&v(&[6, 3, 0, 9]), 1)));
        assert!(!is_zero(e.reduce_prefix(&w, 1)));
        assert!(is_zero(e.reduce_prefix(&w, 2)));
        // u2 lies in the full span but not in span(u0, u1).
        assert!(!is_zero(e.reduce_prefix(&us[2], 2)));
        assert!(is_zero(e.reduce_prefix(&us[2], 4)));
        e.truncate(2);
        assert_eq!(e.len(), 2);
        let (y, lam) = e.reduce(&w);
        assert!(is_zero(y));
        assert_eq!(e.coords(&lam).unwrap(), v(&[1, 2]));
        assert!(!is_zero(e.reduce(&us[2]).0));
        // Pushing after a truncation extends the basis again.
        let (y, lam) = e.reduce(&us[3]);
        assert!(e.push(y, lam).unwrap());
        let (y, lam) = e.reduce(&v(&[2, 3, 3, 10]));
        assert!(is_zero(y));
        assert_eq!(e.coords(&lam).unwrap(), v(&[1, 2, 1]));
    }

    #[test]
    fn rounding_field_terminates() {
        // Real wraps f64 (is_zero and == use a 1e-10 absolute tolerance): the spinning
        // terminates (at most n stored rows), and the relation self-check either passes or
        // gives Err(InvalidOperation); no accuracy claim. Measured: the 5×5 case passes, the
        // 8×8 case (entries in [-5, 5]) fails the self-check.
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
            match krylov_presentation(&a) {
                Ok(p) => {
                    let total: usize = (0..p.relations.rows())
                        .map(|j| p.relations.get(j, j).degree().unwrap())
                        .sum();
                    assert_eq!(total, n);
                }
                Err(e) => assert!(matches!(e, MathError::InvalidOperation(_)), "{e:?}"),
            }
        }
    }
}

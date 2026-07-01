//! Exact characteristic polynomials, eigenspaces/eigenvalues, and matrix
//! multiplicative order (MAGMA ch. 26).
//!
//! MAGMA source: Handbook chapter 26 (Matrices), §26.11 (`CharacteristicPolynomial`,
//! `Eigenvalues`, `Eigenspace`) and §26.13 (`Order`, `ProjectiveOrder` of an
//! invertible matrix).
//!
//! This module is **purely additive**. It provides an *exact*, division-free
//! path that does not go through the `f64` QR iteration in `eigenvalues.rs`:
//!
//! * [`charpoly_berkowitz`] — the Samuelson–Berkowitz algorithm, which computes
//!   `det(xI − A)` using only ring `+`, `−`, `×` (no divisions). It is therefore
//!   correct over **any commutative ring**: finite fields of *any* characteristic
//!   (where the trace/Faddeev–LeVerrier method in `companion.rs` divides by the
//!   step index `k` and breaks when `p | k`), the integers `ℤ`, and `ℚ`.
//! * [`Matrix::eigenspace`] — `Nullspace(A − eI)` over a field (§26.11).
//! * [`Matrix::rational_eigenvalues`] — exact eigenvalues of a rational matrix as
//!   the rational roots of its characteristic polynomial, with multiplicities.
//! * matrix multiplicative order: [`Matrix::multiplicative_order_bounded`],
//!   [`Matrix::order_dividing`], [`Matrix::projective_order_dividing`] and the
//!   [`Matrix::pow_bigint`]/[`Matrix::is_identity`]/[`Matrix::is_scalar`] helpers
//!   (§26.13).

use crate::Matrix;
use rustmath_core::{CommutativeRing, EuclideanDomain, Field, MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// The characteristic polynomial `det(xI − A) ∈ R[x]` of a square matrix over a
/// commutative ring, via the division-free Samuelson–Berkowitz algorithm.
///
/// The result is monic of degree `n` with coefficients in increasing degree
/// order (`coeff[0]` constant, `coeff[n] = 1`).
pub fn charpoly_berkowitz<R: CommutativeRing>(a: &Matrix<R>) -> Result<UnivariatePolynomial<R>> {
    if !a.is_square() {
        return Err(MathError::InvalidArgument(
            "characteristic polynomial requires a square matrix".to_string(),
        ));
    }
    let n = a.rows();
    if n == 0 {
        return Ok(UnivariatePolynomial::new(vec![R::one()]));
    }

    // `p` holds the char-poly coefficients of the leading k×k principal
    // submatrix, ordered high→low degree. Start with the 0×0 matrix: p = [1].
    let mut p: Vec<R> = vec![R::one()];

    for k in 1..=n {
        let akk = a.get(k - 1, k - 1)?.clone();

        // Toeplitz first column `col` of length k+1:
        //   col[0] = 1, col[1] = -A_kk, col[t] = -(S · M^{t-2} · R) for t ≥ 2,
        // where M = A[0..k-1, 0..k-1], S = A[k-1, 0..k-1], R = A[0..k-1, k-1].
        let mut col = vec![R::zero(); k + 1];
        col[0] = R::one();
        col[1] = R::zero() - akk;

        if k >= 2 {
            let s: Vec<R> = (0..k - 1).map(|j| a.get(k - 1, j).unwrap().clone()).collect();
            let rcol: Vec<R> = (0..k - 1).map(|i| a.get(i, k - 1).unwrap().clone()).collect();

            // w = M^t · R for t = 0, 1, …; col[t+2] = -(S · w).
            let mut w = rcol;
            for t in 0..(k - 1) {
                let mut dotv = R::zero();
                for (sj, wj) in s.iter().zip(w.iter()) {
                    dotv = dotv + sj.clone() * wj.clone();
                }
                col[t + 2] = R::zero() - dotv;

                if t + 1 < k - 1 {
                    let mut nw = vec![R::zero(); k - 1];
                    for (i, nwi) in nw.iter_mut().enumerate() {
                        let mut acc = R::zero();
                        for (j, wj) in w.iter().enumerate() {
                            acc = acc + a.get(i, j).unwrap().clone() * wj.clone();
                        }
                        *nwi = acc;
                    }
                    w = nw;
                }
            }
        }

        // Multiply the (k+1)×k lower-triangular Toeplitz matrix (first column
        // `col`) by the length-k vector `p`.
        let mut np = vec![R::zero(); k + 1];
        for (i, npi) in np.iter_mut().enumerate() {
            let mut acc = R::zero();
            for (j, pj) in p.iter().enumerate() {
                if i >= j {
                    let idx = i - j;
                    if idx < col.len() {
                        acc = acc + col[idx].clone() * pj.clone();
                    }
                }
            }
            *npi = acc;
        }
        p = np;
    }

    // `p` is high→low; UnivariatePolynomial wants low→high.
    p.reverse();
    Ok(UnivariatePolynomial::new(p))
}

impl<R: CommutativeRing> Matrix<R> {
    /// The characteristic polynomial `det(xI − A)` via [`charpoly_berkowitz`].
    pub fn charpoly_exact(&self) -> Result<UnivariatePolynomial<R>> {
        charpoly_berkowitz(self)
    }
}

impl<F: Field> Matrix<F> {
    /// The eigenspace `Nullspace(A − eI)` for a scalar `e` (MAGMA `Eigenspace`).
    /// Returns a basis of the (right) eigenvectors; empty when `e` is not an
    /// eigenvalue.
    pub fn eigenspace(&self, e: &F) -> Result<Vec<Vec<F>>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "eigenspace requires a square matrix".to_string(),
            ));
        }
        let n = self.rows();
        let mut shifted = self.clone();
        for i in 0..n {
            let d = shifted[(i, i)].clone() - e.clone();
            shifted[(i, i)] = d;
        }
        shifted.kernel()
    }
}

impl Matrix<Rational> {
    /// The exact eigenvalues of a rational matrix, as `(value, multiplicity)`
    /// pairs (MAGMA `Eigenvalues` over ℚ). Found as the rational roots of the
    /// characteristic polynomial via the rational-root theorem, so only the
    /// eigenvalues lying in ℚ are returned.
    pub fn rational_eigenvalues(&self) -> Result<Vec<(Rational, usize)>> {
        let cp = self.charpoly_exact()?;
        // Coefficients low→high over ℚ; clear denominators to an integer poly.
        let qcoeffs: Vec<Rational> = cp.coefficients().to_vec();
        let mut denom = Integer::one();
        for c in &qcoeffs {
            denom = lcm_int(&denom, c.denominator());
        }
        let mut icoeffs: Vec<Integer> = qcoeffs
            .iter()
            .map(|c| {
                let (q, _r) = (c.numerator().clone() * denom.clone())
                    .div_rem(c.denominator())
                    .unwrap();
                q
            })
            .collect();

        // Work high→low for synthetic division.
        icoeffs.reverse();
        // Strip leading zeros defensively.
        while icoeffs.len() > 1 && icoeffs[0].is_zero() {
            icoeffs.remove(0);
        }

        let mut roots: Vec<(Rational, usize)> = Vec::new();

        // Handle the root 0 first (rational-root theorem needs a nonzero
        // constant term): its multiplicity is the number of trailing zeros.
        {
            let mut mult0 = 0usize;
            while icoeffs.len() > 1 && icoeffs.last().unwrap().is_zero() {
                icoeffs.pop();
                mult0 += 1;
            }
            if mult0 > 0 {
                roots.push((Rational::from_integer(0), mult0));
            }
        }

        if icoeffs.len() >= 2 {
            let leading = icoeffs[0].clone();
            let constant = icoeffs.last().unwrap().clone();
            let ps = integer_divisors(&constant); // p | a0
            let qs = integer_divisors(&leading); // q | an (positive)

            let mut candidates: Vec<Rational> = Vec::new();
            for p in &ps {
                for q in &qs {
                    if q.signum() <= 0 {
                        continue;
                    }
                    if let Ok(r) = Rational::new(p.clone(), q.clone()) {
                        candidates.push(r);
                    }
                }
            }
            candidates.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap());
            candidates.dedup();

            for cand in candidates {
                // Deflate as many times as (x - cand) divides the polynomial.
                let mut mult = 0usize;
                loop {
                    match synthetic_divide(&icoeffs, &cand) {
                        Some(quot) => {
                            icoeffs = quot;
                            mult += 1;
                        }
                        None => break,
                    }
                    if icoeffs.len() < 2 {
                        break;
                    }
                }
                if mult > 0 {
                    roots.push((cand, mult));
                }
            }
        }

        Ok(roots)
    }
}

impl<R: Ring> Matrix<R> {
    /// `self^e` for a non-negative big-integer exponent, by binary exponentiation.
    pub fn pow_bigint(&self, e: &Integer) -> Result<Self> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "matrix power requires a square matrix".to_string(),
            ));
        }
        if e.signum() < 0 {
            return Err(MathError::InvalidArgument(
                "pow_bigint requires a non-negative exponent".to_string(),
            ));
        }
        let n = self.rows();
        let mut result = Matrix::identity(n);
        let mut base = self.clone();
        let mut ee = e.clone();
        let two = Integer::from(2);
        while ee.signum() > 0 {
            let (q, r) = ee.div_rem(&two)?;
            if r.is_one() {
                result = result.mul(&base)?;
            }
            base = base.mul(&base)?;
            ee = q;
        }
        Ok(result)
    }

    /// If `self = s·I` for some scalar `s`, return `Some(s)`; otherwise `None`.
    pub fn is_scalar(&self) -> Option<R> {
        if !self.is_square() || self.rows() == 0 {
            return None;
        }
        let n = self.rows();
        let s = self[(0, 0)].clone();
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { s.clone() } else { R::zero() };
                if self[(i, j)] != want {
                    return None;
                }
            }
        }
        Some(s)
    }

    /// Whether `self` is the identity matrix.
    pub fn is_identity(&self) -> bool {
        match self.is_scalar() {
            Some(s) => s.is_one(),
            None => false,
        }
    }

    /// The multiplicative order of `self` (smallest `k > 0` with `Aᵏ = I`) by
    /// direct powering, searching `k ∈ [1, bound]`. Returns `None` if no such
    /// `k ≤ bound` exists — this is an honest "UNRESOLVED within the bound",
    /// never a claim that the order is infinite.
    pub fn multiplicative_order_bounded(&self, bound: usize) -> Result<Option<Integer>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "order requires a square matrix".to_string(),
            ));
        }
        let mut cur = self.clone();
        for k in 1..=bound {
            if cur.is_identity() {
                return Ok(Some(Integer::from(k as i64)));
            }
            cur = cur.mul(self)?;
        }
        Ok(None)
    }

    /// Given a known multiple `m` of the order (e.g. the exponent of the ambient
    /// finite group), return the exact multiplicative order by stripping prime
    /// factors. Returns `None` when `Aᵐ ≠ I` (so `m` is not actually a multiple
    /// of the order). This is the efficient route for matrices over finite
    /// fields, where `m` can be taken to be a divisor of `|GLₙ(q)|`.
    pub fn order_dividing(&self, m: &Integer) -> Result<Option<Integer>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "order requires a square matrix".to_string(),
            ));
        }
        if m.signum() <= 0 {
            return Err(MathError::InvalidArgument(
                "the multiple must be positive".to_string(),
            ));
        }
        if !self.pow_bigint(m)?.is_identity() {
            return Ok(None);
        }
        let mut e = m.clone();
        for (p, _mult) in rustmath_integers::prime::factor(m) {
            loop {
                let (q, r) = e.div_rem(&p)?;
                if !r.is_zero() {
                    break;
                }
                if self.pow_bigint(&q)?.is_identity() {
                    e = q;
                } else {
                    break;
                }
            }
        }
        Ok(Some(e))
    }

    /// Given a known multiple `m` of the projective order, return the exact
    /// projective order `n` (smallest `n | m` with `Aⁿ = s·I`) and the scalar
    /// `s` (MAGMA `ProjectiveOrder`). Returns `None` when `Aᵐ` is not scalar.
    pub fn projective_order_dividing(&self, m: &Integer) -> Result<Option<(Integer, R)>> {
        if !self.is_square() {
            return Err(MathError::InvalidArgument(
                "projective order requires a square matrix".to_string(),
            ));
        }
        if m.signum() <= 0 {
            return Err(MathError::InvalidArgument(
                "the multiple must be positive".to_string(),
            ));
        }
        if self.pow_bigint(m)?.is_scalar().is_none() {
            return Ok(None);
        }
        let mut e = m.clone();
        for (p, _mult) in rustmath_integers::prime::factor(m) {
            loop {
                let (q, r) = e.div_rem(&p)?;
                if !r.is_zero() {
                    break;
                }
                if self.pow_bigint(&q)?.is_scalar().is_some() {
                    e = q;
                } else {
                    break;
                }
            }
        }
        let s = self.pow_bigint(&e)?.is_scalar().unwrap();
        Ok(Some((e, s)))
    }
}

// -------------------------- integer helpers ------------------------------- //

fn lcm_int(a: &Integer, b: &Integer) -> Integer {
    if a.is_zero() || b.is_zero() {
        return Integer::zero();
    }
    let g = a.gcd(b);
    let (q, _) = (a.clone() * b.clone()).div_rem(&g).unwrap();
    q.abs()
}

/// All (positive and negative) divisors of a nonzero integer.
fn integer_divisors(n: &Integer) -> Vec<Integer> {
    if n.is_zero() {
        return Vec::new();
    }
    let mut divisors = vec![Integer::one()];
    for (p, mult) in rustmath_integers::prime::factor(&n.abs()) {
        let mut new_divs = Vec::new();
        for d in &divisors {
            let mut pk = Integer::one();
            for _ in 0..=mult {
                new_divs.push(d.clone() * pk.clone());
                pk = pk * p.clone();
            }
        }
        divisors = new_divs;
    }
    // include negatives
    let mut with_neg = Vec::with_capacity(divisors.len() * 2);
    for d in divisors {
        with_neg.push(d.clone());
        with_neg.push(-d);
    }
    with_neg
}

/// Synthetic division of an integer polynomial (coefficients high→low) by
/// `(x − r)` for a rational `r`. Returns the integer-coefficient quotient
/// (scaled to stay integral) if the division is exact, else `None`.
///
/// We test the root exactly over ℚ; on success we return `q·denominator`-cleared
/// integer coefficients of the quotient so that further deflation stays in ℤ[x].
fn synthetic_divide(coeffs_hi_lo: &[Integer], r: &Rational) -> Option<Vec<Integer>> {
    // Evaluate p(r) exactly; bail if nonzero.
    // p(r) = Σ c_i r^{deg-i}. Use Horner over ℚ.
    let mut acc = Rational::from_integer(0);
    for c in coeffs_hi_lo {
        acc = acc * r.clone() + Rational::from_integer(c.clone());
    }
    if !acc.numerator().is_zero() {
        return None;
    }

    // Horner quotient over ℚ.
    let mut qcoeffs: Vec<Rational> = Vec::with_capacity(coeffs_hi_lo.len().saturating_sub(1));
    let mut running = Rational::from_integer(0);
    for (idx, c) in coeffs_hi_lo.iter().enumerate() {
        running = running * r.clone() + Rational::from_integer(c.clone());
        if idx < coeffs_hi_lo.len() - 1 {
            qcoeffs.push(running.clone());
        }
        // last `running` is the remainder (== 0 by the check above)
    }

    // Clear denominators so the quotient is integral again.
    let mut denom = Integer::one();
    for c in &qcoeffs {
        denom = lcm_int(&denom, c.denominator());
    }
    let out: Vec<Integer> = qcoeffs
        .iter()
        .map(|c| {
            let (q, _r) = (c.numerator().clone() * denom.clone())
                .div_rem(c.denominator())
                .unwrap();
            q
        })
        .collect();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::companion::characteristic_polynomial;
    use rustmath_rationals::Rational;
    use std::fmt;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    fn q(n: i64) -> Rational {
        Rational::from_integer(n)
    }
    fn z(n: i64) -> Integer {
        Integer::from(n)
    }

    #[test]
    fn berkowitz_matches_faddeev_leverrier_over_Q() {
        // A few random-ish rational matrices; cross-check against companion.rs.
        let mats = vec![
            Matrix::from_vec(2, 2, vec![q(1), q(2), q(3), q(4)]).unwrap(),
            Matrix::from_vec(3, 3, vec![q(2), q(0), q(1), q(1), q(3), q(0), q(0), q(1), q(4)]).unwrap(),
            Matrix::from_vec(3, 3, vec![q(0), q(1), q(0), q(0), q(0), q(1), q(1), q(0), q(0)]).unwrap(),
        ];
        for a in mats {
            let b = charpoly_berkowitz(&a).unwrap();
            let f = characteristic_polynomial(&a).unwrap();
            assert_eq!(b.coefficients(), f.coefficients(), "charpoly mismatch");
        }
    }

    #[test]
    fn berkowitz_works_over_Z() {
        // Faddeev–LeVerrier cannot run over ℤ (needs field division); Berkowitz can.
        // A = [[1,2],[3,4]] -> det(xI-A) = x^2 - 5x - 2.
        let a = Matrix::from_vec(2, 2, vec![z(1), z(2), z(3), z(4)]).unwrap();
        let cp = charpoly_berkowitz(&a).unwrap();
        assert_eq!(cp.coefficients(), &[z(-2), z(-5), z(1)]);
    }

    // ---- a self-contained GF(5) to exercise the characteristic-p path -------
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct F5(u8);
    impl F5 {
        fn n(v: i64) -> F5 {
            F5(v.rem_euclid(5) as u8)
        }
    }
    impl fmt::Display for F5 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Add for F5 {
        type Output = F5;
        fn add(self, o: F5) -> F5 {
            F5((self.0 + o.0) % 5)
        }
    }
    impl Sub for F5 {
        type Output = F5;
        fn sub(self, o: F5) -> F5 {
            F5((self.0 + 5 - o.0) % 5)
        }
    }
    impl Mul for F5 {
        type Output = F5;
        fn mul(self, o: F5) -> F5 {
            F5((self.0 * o.0) % 5)
        }
    }
    impl Neg for F5 {
        type Output = F5;
        fn neg(self) -> F5 {
            F5((5 - self.0) % 5)
        }
    }
    impl Div for F5 {
        type Output = F5;
        fn div(self, o: F5) -> F5 {
            self * o.inverse().unwrap()
        }
    }
    impl Ring for F5 {
        fn zero() -> F5 {
            F5(0)
        }
        fn one() -> F5 {
            F5(1)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_one(&self) -> bool {
            self.0 == 1
        }
    }
    impl CommutativeRing for F5 {}
    impl rustmath_core::IntegralDomain for F5 {}
    impl Field for F5 {
        fn inverse(&self) -> Result<F5> {
            if self.0 == 0 {
                return Err(MathError::DivisionByZero);
            }
            // Fermat: a^(p-2)
            let mut r = F5(1);
            for _ in 0..3 {
                r = r * *self;
            }
            Ok(r)
        }
    }

    #[test]
    fn berkowitz_works_in_characteristic_5() {
        // n = 5 forces the trace/Faddeev method to divide by k = 5 ≡ 0 (mod 5).
        // Berkowitz needs no division and returns the correct char poly.
        // Take the identity on GF(5)^5: char poly should be (x-1)^5.
        let a = Matrix::<F5>::identity(5);
        let cp = charpoly_berkowitz(&a).unwrap();
        // (x-1)^5 over GF(5) = x^5 - 1 (freshman's dream: (x-1)^5 = x^5 - 1^5).
        // coefficients low->high: [-1, 0,0,0,0, 1] == [4,0,0,0,0,1] in GF(5).
        assert_eq!(
            cp.coefficients(),
            &[F5::n(-1), F5(0), F5(0), F5(0), F5(0), F5(1)]
        );
    }

    #[test]
    fn eigenspace_and_rational_eigenvalues() {
        // diag(2,3): eigenvalues 2 (mult 1) and 3 (mult 1).
        let a = Matrix::from_vec(2, 2, vec![q(2), q(0), q(0), q(3)]).unwrap();
        let evals = a.rational_eigenvalues().unwrap();
        let set: Vec<Rational> = evals.iter().map(|(v, _)| v.clone()).collect();
        assert!(set.contains(&q(2)));
        assert!(set.contains(&q(3)));
        for (_, m) in &evals {
            assert_eq!(*m, 1);
        }

        // eigenspace for e=2 is spanned by (1,0).
        let es = a.eigenspace(&q(2)).unwrap();
        assert_eq!(es.len(), 1);

        // A non-eigenvalue yields an empty eigenspace.
        assert!(a.eigenspace(&q(5)).unwrap().is_empty());
    }

    #[test]
    fn rational_eigenvalues_with_multiplicity_and_zero() {
        // Nilpotent [[0,1],[0,0]] -> char poly x^2 -> eigenvalue 0 mult 2.
        let a = Matrix::from_vec(2, 2, vec![q(0), q(1), q(0), q(0)]).unwrap();
        let evals = a.rational_eigenvalues().unwrap();
        assert_eq!(evals, vec![(q(0), 2)]);
    }

    #[test]
    fn multiplicative_order_of_rotation() {
        // 90° rotation over ℚ: order 4.
        let a = Matrix::from_vec(2, 2, vec![q(0), q(-1), q(1), q(0)]).unwrap();
        assert_eq!(
            a.multiplicative_order_bounded(20).unwrap(),
            Some(Integer::from(4))
        );
        // With a known multiple 12, prime-strip to the exact order 4.
        assert_eq!(a.order_dividing(&Integer::from(12)).unwrap(), Some(Integer::from(4)));

        // Projective order: A^2 = -I is scalar, so proj order = 2, s = -1.
        let (n, s) = a.projective_order_dividing(&Integer::from(12)).unwrap().unwrap();
        assert_eq!(n, Integer::from(2));
        assert_eq!(s, q(-1));
    }

    #[test]
    fn order_reports_non_multiple() {
        // A shear has infinite order; a wrong "multiple" gives None (honest).
        let a = Matrix::from_vec(2, 2, vec![q(1), q(1), q(0), q(1)]).unwrap();
        assert_eq!(a.order_dividing(&Integer::from(6)).unwrap(), None);
        assert_eq!(a.multiplicative_order_bounded(50).unwrap(), None);
    }
}

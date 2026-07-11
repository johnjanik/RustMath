//! Eisenstein (totally ramified) extensions of `Q_p` of degree `e`:
//! `Z_p[pi]/(E(pi))` with `E` monic Eisenstein.
//!
//! # Model
//!
//! `E(x) = x^e + c_{e-1} x^{e-1} + ... + c_0` with `v_p(c_i) >= 1` for
//! `i < e` and `v_p(c_0) = 1` exactly (checked at construction; monicity is
//! required so that `1, pi, ..., pi^{e-1}` is a `Z_p`-basis of `Z_p[pi]`).
//! Elements are vectors `[a_0, ..., a_{e-1}]` over `Z/p^N` (the fixed-modulus
//! model, coefficientwise): integral elements only; the field of fractions
//! (negative valuations) is deliberately out of scope for this stage.
//!
//! # Valuation (pinned convention, sympy-verified)
//!
//! `v_L` is normalized by `v_L(pi) = 1`, so `v_L(p) = e` and the unique
//! extension `w` of `v_p` with `w(p) = 1` is `w = v_L / e`, `w(pi) = 1/e`.
//! For an integral element the values `e*v_p(a_i) + i` are pairwise distinct
//! (they differ mod `e`), so there is no cancellation and
//!
//! ```text
//! v_L(sum a_i pi^i) = min_i ( e * v_p(a_i) + i ).
//! ```
//!
//! # Norm law (sympy-verified before implementation)
//!
//! With `f = 1` the uniform law `v_p(N_{L/Q_p}(x)) = f * v_L(x)` reads
//!
//! ```text
//! v_p(N(x)) = v_L(x)        (equivalently n * w(x), n = e)
//! ```
//!
//! e.g. `N(pi) = (-1)^e E(0)` has `v_p = 1 = v_L(pi)`.
//!
//! # Self-certification
//!
//! [`EisensteinElement::norm`] computes the multiplication-matrix determinant
//! AND the Sylvester resultant `Res(E, a)` (two independent algorithms) and
//! errors on mismatch; [`EisensteinElement::trace`] cross-checks the matrix
//! trace against `sum_i a_i s_i` with the power sums `s_i` from Newton's
//! identities. Never a silently wrong value.

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use super::poly_arith::{canon, det_bareiss, polymul_mod};

/// A totally ramified extension `Q_p(pi)`, `E(pi) = 0`, `E` monic Eisenstein.
#[derive(Clone, Debug)]
pub struct EisensteinExtension {
    prime: Integer,
    /// Ramification index `e` = degree (`f` = 1).
    degree: usize,
    /// Elements are known mod `p^N` coefficientwise (pi-adic precision `e*N`).
    precision: u32,
    /// Monic Eisenstein modulus, little-endian, EXACT integer coefficients
    /// (kept unreduced so that norms/traces of canonical lifts are exact).
    modulus: Vec<Integer>,
}

impl PartialEq for EisensteinExtension {
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime
            && self.degree == other.degree
            && self.precision == other.precision
            && self.modulus == other.modulus
    }
}

impl EisensteinExtension {
    /// Build `Z_p[pi]/(E)` after verifying that `E` is monic Eisenstein at
    /// `p`: leading coefficient 1, `v_p(c_i) >= 1` for `0 < i < e`, and
    /// `v_p(c_0) = 1` exactly.
    pub fn new(
        prime: Integer,
        eisenstein_poly: &UnivariatePolynomial<Integer>,
        precision: u32,
    ) -> Result<Arc<Self>> {
        if !prime.is_prime() {
            return Err(MathError::InvalidArgument(
                "EisensteinExtension: p must be prime".to_string(),
            ));
        }
        if precision == 0 {
            return Err(MathError::InvalidArgument(
                "EisensteinExtension: precision must be >= 1".to_string(),
            ));
        }
        let degree = eisenstein_poly.degree().ok_or_else(|| {
            MathError::InvalidArgument("EisensteinExtension: zero modulus".to_string())
        })?;
        if degree == 0 {
            return Err(MathError::InvalidArgument(
                "EisensteinExtension: modulus must have degree >= 1".to_string(),
            ));
        }
        let coeffs = eisenstein_poly.coefficients();
        if !coeffs[degree].is_one() {
            return Err(MathError::InvalidArgument(
                "EisensteinExtension: modulus must be monic".to_string(),
            ));
        }
        if coeffs[0].valuation(&prime) != 1 {
            return Err(MathError::InvalidArgument(
                "EisensteinExtension: constant term must have v_p exactly 1".to_string(),
            ));
        }
        for c in coeffs.iter().take(degree).skip(1) {
            if !c.is_zero() && c.valuation(&prime) == 0 {
                return Err(MathError::InvalidArgument(
                    "EisensteinExtension: middle coefficients must be divisible by p"
                        .to_string(),
                ));
            }
        }
        Ok(Arc::new(EisensteinExtension {
            prime,
            degree,
            precision,
            modulus: coeffs.to_vec(),
        }))
    }

    /// The prime `p`.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// The degree `e` (= ramification index; residue degree f = 1).
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Coefficient precision `N` (elements known mod `p^N` coefficientwise,
    /// i.e. pi-adic absolute precision `e*N`).
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// The exact monic Eisenstein modulus (little-endian).
    pub fn modulus(&self) -> &[Integer] {
        &self.modulus
    }

    /// Zero element.
    pub fn zero(self: &Arc<Self>) -> EisensteinElement {
        EisensteinElement {
            ext: self.clone(),
            coeffs: vec![Integer::zero(); self.degree],
        }
    }

    /// One element.
    pub fn one(self: &Arc<Self>) -> EisensteinElement {
        let mut coeffs = vec![Integer::zero(); self.degree];
        coeffs[0] = Integer::one();
        EisensteinElement {
            ext: self.clone(),
            coeffs,
        }
    }

    /// The uniformizer `pi` (class of `x`); `v_L(pi) = 1`.
    pub fn uniformizer(self: &Arc<Self>) -> EisensteinElement {
        let mut coeffs = vec![Integer::zero(); self.degree];
        if self.degree >= 2 {
            coeffs[1] = Integer::one();
        } else {
            // degree 1: pi = -c_0 (E = x + c_0), the base field itself
            let pn = self.prime.pow(self.precision);
            coeffs[0] = canon(-self.modulus[0].clone(), &pn);
        }
        EisensteinElement {
            ext: self.clone(),
            coeffs,
        }
    }

    /// Element from little-endian `Z`-coefficients (length <= e; padded,
    /// reduced mod `p^N`).
    pub fn element(self: &Arc<Self>, coeffs: &[Integer]) -> Result<EisensteinElement> {
        if coeffs.len() > self.degree {
            return Err(MathError::InvalidArgument(format!(
                "EisensteinExtension::element: {} coefficients for degree {}",
                coeffs.len(),
                self.degree
            )));
        }
        let pn = self.prime.pow(self.precision);
        let mut v: Vec<Integer> = coeffs.iter().map(|c| canon(c.clone(), &pn)).collect();
        v.resize(self.degree, Integer::zero());
        Ok(EisensteinElement {
            ext: self.clone(),
            coeffs: v,
        })
    }

    /// Power sums `s_0, ..., s_{e-1}` of the roots of `E` via Newton's
    /// identities (`s_0 = e`; `s_k = -k c_{e-k} - sum_{i=1}^{k-1} c_{e-i}
    /// s_{k-i}`), exact integers; convention verified against companion
    /// matrices in sympy.
    fn power_sums(&self) -> Vec<Integer> {
        let e = self.degree;
        let c = &self.modulus; // c[i], i < e, plus c[e] = 1
        let mut s: Vec<Integer> = vec![Integer::from(e as i64)];
        for k in 1..e {
            let mut tot = -(Integer::from(k as i64) * c[e - k].clone());
            for i in 1..k {
                tot = tot - c[e - i].clone() * s[k - i].clone();
            }
            s.push(tot);
        }
        s
    }

    fn pn(&self) -> Integer {
        self.prime.pow(self.precision)
    }
}

impl fmt::Display for EisensteinExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Eisenstein extension of Q_{} of degree {} (v(pi) = 1/{})",
            self.prime, self.degree, self.degree
        )
    }
}

/// An integral element `sum a_i pi^i` of an Eisenstein extension.
#[derive(Clone, Debug)]
pub struct EisensteinElement {
    ext: Arc<EisensteinExtension>,
    /// length `e`, canonical mod `p^N`
    coeffs: Vec<Integer>,
}

impl PartialEq for EisensteinElement {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.ext, &other.ext) || *self.ext == *other.ext)
            && self.coeffs == other.coeffs
    }
}

impl EisensteinElement {
    /// The parent extension.
    pub fn extension(&self) -> &Arc<EisensteinExtension> {
        &self.ext
    }

    /// Coefficients `[a_0, ..., a_{e-1}]`, canonical mod `p^N`.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coeffs
    }

    fn assert_same_ext(&self, other: &Self) {
        assert!(
            Arc::ptr_eq(&self.ext, &other.ext) || *self.ext == *other.ext,
            "EisensteinElement: elements from different extensions"
        );
    }

    /// Normalized valuation `v_L` (`v_L(pi) = 1`, `v_L(p) = e`):
    /// `min_i (e * v_p(a_i) + i)` on the canonical representative — exact
    /// because the candidate values are pairwise distinct mod `e`.
    /// `None` means zero to precision (`v_L >= e*N`).
    pub fn valuation_pi(&self) -> Option<u64> {
        let e = self.ext.degree as u64;
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| e * c.valuation(&self.ext.prime) as u64 + i as u64)
            .min()
    }

    /// Multiplicative inverse (units only, i.e. `v_L = 0` i.e. `a_0` a
    /// p-unit). Newton `z -> z(2 - xz)` from `a_0^{-1} mod p`, then
    /// certified `x*z = 1` exactly mod `(E, p^N)`.
    pub fn inverse(&self) -> Result<Self> {
        let e = self.ext.degree;
        let p = &self.ext.prime;
        let n = self.ext.precision;
        // starting inverse in the residue field GF(p): 1/a_0 mod p
        let a0_inv = canon(self.coeffs[0].clone(), p)
            .mod_inverse(p)
            .ok_or(MathError::NotInvertible)?;
        let mut z = vec![Integer::zero(); e];
        z[0] = a0_inv;
        // v_L(xz - 1) >= 1 and doubles each step; pi-adic target is e*N,
        // so ceil(log2(e*N)) + 1 steps suffice; we certify exactly anyway.
        let pn = self.ext.pn();
        let mut steps = 0u32;
        // v_L(xz - 1) doubles per step, target e*N: log2(e*N) + 2 steps
        // suffice; 64 + log2 bound is far beyond any representable target.
        let max_steps = 66 - ((e as u64) * (n as u64)).leading_zeros().min(60);
        loop {
            let xz = polymul_mod(&self.coeffs, &z, &self.ext.modulus, &pn);
            let mut is_one = xz[0].is_one();
            for c in &xz[1..] {
                is_one &= c.is_zero();
            }
            if is_one {
                break;
            }
            steps += 1;
            if steps > max_steps {
                return Err(MathError::NumericalError(
                    "EisensteinElement::inverse: Newton iteration failed to \
                     converge (element is not a unit?)"
                        .to_string(),
                ));
            }
            let mut two_minus: Vec<Integer> =
                xz.iter().map(|c| canon(-c.clone(), &pn)).collect();
            two_minus[0] = canon(two_minus[0].clone() + Integer::from(2), &pn);
            z = polymul_mod(&z, &two_minus, &self.ext.modulus, &pn);
        }
        Ok(EisensteinElement {
            ext: self.ext.clone(),
            coeffs: z,
        })
    }

    fn multiplication_matrix(&self) -> Vec<Vec<Integer>> {
        let e = self.ext.degree;
        let pn = self.ext.pn();
        let mut cols = Vec::with_capacity(e);
        for j in 0..e {
            let mut basis = vec![Integer::zero(); e];
            basis[j] = Integer::one();
            cols.push(polymul_mod(&self.coeffs, &basis, &self.ext.modulus, &pn));
        }
        (0..e)
            .map(|i| (0..e).map(|j| cols[j][i].clone()).collect())
            .collect()
    }

    /// Norm via `det` of the multiplication matrix (Bareiss, exact on the
    /// canonical lifts, reduced mod `p^N`).
    pub fn norm_via_matrix(&self) -> Integer {
        canon(det_bareiss(self.multiplication_matrix()), &self.ext.pn())
    }

    /// Norm via the Sylvester resultant `Res(E, a)` (`= prod a(theta_i)` for
    /// monic `E`); layout verified against sympy's `resultant`.
    pub fn norm_via_resultant(&self) -> Integer {
        let pn = self.ext.pn();
        // trim a to its true degree on the canonical lift
        let mut a = self.coeffs.clone();
        while a.last().is_some_and(|c| c.is_zero()) {
            a.pop();
        }
        if a.is_empty() {
            return Integer::zero(); // representative is exactly 0
        }
        let e = self.ext.degree; // deg E
        let d = a.len() - 1; // deg a
        if d == 0 {
            // Res(E, const c) = c^e
            let mut r = Integer::one();
            for _ in 0..e {
                r = canon(r * a[0].clone(), &pn);
            }
            return r;
        }
        let size = e + d;
        let mut m = vec![vec![Integer::zero(); size]; size];
        // d rows of E (big-endian), then e rows of a (big-endian)
        for i in 0..d {
            for (k, c) in self.ext.modulus.iter().rev().enumerate() {
                m[i][i + k] = c.clone();
            }
        }
        for j in 0..e {
            for (k, c) in a.iter().rev().enumerate() {
                m[d + j][j + k] = c.clone();
            }
        }
        canon(det_bareiss(m), &pn)
    }

    /// Trace via the multiplication matrix diagonal.
    pub fn trace_via_matrix(&self) -> Integer {
        let pn = self.ext.pn();
        let m = self.multiplication_matrix();
        let mut t = Integer::zero();
        for (i, row) in m.iter().enumerate() {
            t = canon(t + row[i].clone(), &pn);
        }
        t
    }

    /// Trace via Newton power sums: `Tr(sum a_i pi^i) = sum a_i s_i`
    /// (independent of the matrix path; formula sympy-verified).
    pub fn trace_via_power_sums(&self) -> Integer {
        let pn = self.ext.pn();
        let s = self.ext.power_sums();
        let mut t = Integer::zero();
        for (a, si) in self.coeffs.iter().zip(s.iter()) {
            t = canon(t + a.clone() * si.clone(), &pn);
        }
        t
    }

    /// Norm `N_{L/Q_p}(x)` mod `p^N`, computed by **two independent
    /// algorithms** (multiplication-matrix determinant and Sylvester
    /// resultant) and cross-certified.
    pub fn norm(&self) -> Result<Integer> {
        let via_matrix = self.norm_via_matrix();
        let via_res = self.norm_via_resultant();
        if via_matrix != via_res {
            return Err(MathError::NumericalError(format!(
                "Eisenstein norm self-certification failed: matrix={} resultant={}",
                via_matrix, via_res
            )));
        }
        Ok(via_matrix)
    }

    /// Trace `Tr_{L/Q_p}(x)` mod `p^N`, cross-certified between the matrix
    /// diagonal and Newton power sums.
    pub fn trace(&self) -> Result<Integer> {
        let via_matrix = self.trace_via_matrix();
        let via_ps = self.trace_via_power_sums();
        if via_matrix != via_ps {
            return Err(MathError::NumericalError(format!(
                "Eisenstein trace self-certification failed: matrix={} power-sums={}",
                via_matrix, via_ps
            )));
        }
        Ok(via_matrix)
    }
}

impl fmt::Display for EisensteinElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            match i {
                0 => write!(f, "{}", c)?,
                1 => write!(f, "{}*pi", c)?,
                _ => write!(f, "{}*pi^{}", c, i)?,
            }
        }
        if first {
            write!(f, "0")?;
        }
        write!(f, " + O({}^{})", self.ext.prime, self.ext.precision)
    }
}

impl Add for EisensteinElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.assert_same_ext(&other);
        let pn = self.ext.pn();
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| canon(a.clone() + b.clone(), &pn))
            .collect();
        EisensteinElement {
            ext: self.ext,
            coeffs,
        }
    }
}

impl Sub for EisensteinElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.assert_same_ext(&other);
        let pn = self.ext.pn();
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| canon(a.clone() - b.clone(), &pn))
            .collect();
        EisensteinElement {
            ext: self.ext,
            coeffs,
        }
    }
}

impl Mul for EisensteinElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.assert_same_ext(&other);
        let pn = self.ext.pn();
        let coeffs = polymul_mod(&self.coeffs, &other.coeffs, &self.ext.modulus, &pn);
        EisensteinElement {
            ext: self.ext,
            coeffs,
        }
    }
}

impl Neg for EisensteinElement {
    type Output = Self;
    fn neg(self) -> Self {
        let pn = self.ext.pn();
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| canon(-c.clone(), &pn))
            .collect();
        EisensteinElement {
            ext: self.ext,
            coeffs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ints(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&c| Integer::from(c)).collect()
    }

    /// x^3 - 2 over Q_2 (Eisenstein; all expected values sympy-verified
    /// via resultants/companion matrices before this Rust existed).
    fn q2_cbrt2() -> Arc<EisensteinExtension> {
        EisensteinExtension::new(
            Integer::from(2),
            &UnivariatePolynomial::new(ints(&[-2, 0, 0, 1])),
            20,
        )
        .unwrap()
    }

    #[test]
    fn test_eisenstein_verification() {
        let p = Integer::from(5);
        // x^2 - 5 is Eisenstein at 5
        assert!(EisensteinExtension::new(
            p.clone(),
            &UnivariatePolynomial::new(ints(&[-5, 0, 1])),
            8
        )
        .is_ok());
        // x^2 - 25 is not (v(25) = 2)
        assert!(EisensteinExtension::new(
            p.clone(),
            &UnivariatePolynomial::new(ints(&[-25, 0, 1])),
            8
        )
        .is_err());
        // x^2 - x - 5 is not (middle coefficient a unit)
        assert!(EisensteinExtension::new(
            p.clone(),
            &UnivariatePolynomial::new(ints(&[-5, -1, 1])),
            8
        )
        .is_err());
        // 2x^2 - 5 is not monic
        assert!(EisensteinExtension::new(
            p,
            &UnivariatePolynomial::new(ints(&[-5, 0, 2])),
            8
        )
        .is_err());
        // x^3 + 2x + 2 IS Eisenstein at 2 (the Newton-polygon gate poly)
        assert!(EisensteinExtension::new(
            Integer::from(2),
            &UnivariatePolynomial::new(ints(&[2, 2, 0, 1])),
            8
        )
        .is_ok());
    }

    #[test]
    fn test_pi_powers_and_p() {
        // pi^3 = 2 in Q_2(2^{1/3})
        let ext = q2_cbrt2();
        let pi = ext.uniformizer();
        let pi3 = pi.clone() * pi.clone() * pi.clone();
        assert_eq!(pi3, ext.element(&ints(&[2])).unwrap());
        assert_eq!(pi.valuation_pi(), Some(1));
        assert_eq!(pi3.valuation_pi(), Some(3)); // v_L(p) = e = 3
        let pi2 = pi.clone() * pi.clone();
        assert_eq!(pi2.valuation_pi(), Some(2));
    }

    #[test]
    fn test_norm_trace_sympy_values() {
        let ext = q2_cbrt2();
        let pi = ext.uniformizer();
        // sympy: N(pi) = 2, Tr(pi) = 0
        assert_eq!(pi.norm().unwrap(), Integer::from(2));
        assert_eq!(pi.trace().unwrap(), Integer::zero());
        // sympy: N(4 + 2 pi + pi^2) = 36, Tr = 12
        let a = ext.element(&ints(&[4, 2, 1])).unwrap();
        assert_eq!(a.norm().unwrap(), Integer::from(36));
        assert_eq!(a.trace().unwrap(), Integer::from(12));
    }

    #[test]
    fn test_degree2_norm_formula() {
        // Q_2(sqrt2): N(a + b*pi) = a^2 - 2b^2 (sympy-verified table)
        let ext = EisensteinExtension::new(
            Integer::from(2),
            &UnivariatePolynomial::new(ints(&[-2, 0, 1])),
            20,
        )
        .unwrap();
        let pn = Integer::from(2).pow(20);
        for (a, b) in [(3i64, 2i64), (1, 1), (0, 5), (7, 0)] {
            let el = ext.element(&ints(&[a, b])).unwrap();
            let expect = canon(Integer::from(a * a - 2 * b * b), &pn);
            assert_eq!(el.norm().unwrap(), expect, "N({} + {} pi)", a, b);
            let expect_tr = canon(Integer::from(2 * a), &pn);
            assert_eq!(el.trace().unwrap(), expect_tr);
        }
    }

    #[test]
    fn test_norm_valuation_law() {
        // sympy-verified: v_p(N(x)) = v_L(x) (= f * v_L with f = 1)
        let ext = q2_cbrt2();
        for coeffs in [
            [0i64, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [4, 2, 1],
            [6, 0, 4],
            [1, 3, 5],
            [0, 4, 2],
        ] {
            let a = ext.element(&ints(&coeffs)).unwrap();
            let v_l = a.valuation_pi().unwrap();
            let n = a.norm().unwrap();
            assert_eq!(
                n.valuation(&Integer::from(2)) as u64,
                v_l,
                "v_p(N(x)) != v_L(x) for {:?}",
                coeffs
            );
        }
    }

    #[test]
    fn test_law_battery_random() {
        // N(xy) = N(x)N(y), Tr(x+y) = Tr(x)+Tr(y) — deterministic LCG
        let ext = q2_cbrt2();
        let pn = Integer::from(2).pow(20);
        let mut state: u64 = 0x5EED5EED;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..12 {
            let a = ext
                .element(&ints(&[next() % 500, next() % 500, next() % 500]))
                .unwrap();
            let b = ext
                .element(&ints(&[next() % 500, next() % 500, next() % 500]))
                .unwrap();
            let n_ab = (a.clone() * b.clone()).norm().unwrap();
            let na_nb = canon(a.norm().unwrap() * b.norm().unwrap(), &pn);
            assert_eq!(n_ab, na_nb, "N(xy) != N(x)N(y)");
            let t_ab = (a.clone() + b.clone()).trace().unwrap();
            let ta_tb = canon(a.trace().unwrap() + b.trace().unwrap(), &pn);
            assert_eq!(t_ab, ta_tb, "Tr(x+y) != Tr(x)+Tr(y)");
        }
    }

    #[test]
    fn test_inverse() {
        let ext = q2_cbrt2();
        // 1 + pi is a unit (v_L = 0)
        let a = ext.element(&ints(&[1, 1, 0])).unwrap();
        let inv = a.inverse().unwrap();
        assert_eq!(a * inv, ext.one());
        // pi is not a unit
        assert!(ext.uniformizer().inverse().is_err());
        // 2 is not a unit
        assert!(ext.element(&ints(&[2])).unwrap().inverse().is_err());
        // 3 IS a unit (odd)
        let three = ext.element(&ints(&[3])).unwrap();
        assert_eq!(three.clone() * three.inverse().unwrap(), ext.one());
    }

    #[test]
    fn test_degree1_norm_trace_identity() {
        // E = x - 5 over Q_5: the trivial "extension"; N = Tr = id
        let ext = EisensteinExtension::new(
            Integer::from(5),
            &UnivariatePolynomial::new(ints(&[-5, 1])),
            8,
        )
        .unwrap();
        let a = ext.element(&ints(&[7])).unwrap();
        assert_eq!(a.norm().unwrap(), Integer::from(7));
        assert_eq!(a.trace().unwrap(), Integer::from(7));
    }

    #[test]
    fn test_x3_plus_2x_plus_2() {
        // the Newton-polygon gate polynomial as an actual extension:
        // sympy power sums for x^3+2x+2: s = [3, 0, -4]
        let ext = EisensteinExtension::new(
            Integer::from(2),
            &UnivariatePolynomial::new(ints(&[2, 2, 0, 1])),
            16,
        )
        .unwrap();
        let pn = Integer::from(2).pow(16);
        // Tr(pi^2) = s_2 = -4; N(pi) = (-1)^3 * 2 = -2
        let pi = ext.uniformizer();
        let pi2 = pi.clone() * pi.clone();
        assert_eq!(pi2.trace().unwrap(), canon(Integer::from(-4), &pn));
        assert_eq!(pi.norm().unwrap(), canon(Integer::from(-2), &pn));
        assert_eq!(pi.norm().unwrap().valuation(&Integer::from(2)), 1);
    }
}

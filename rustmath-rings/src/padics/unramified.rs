//! Unramified extensions `Q_p^unr` of degree `f`: `Z_q = Z_p[x]/(u(x))`.
//!
//! `u` is a monic lift of an irreducible polynomial over `GF(p)`. When the
//! verified Conway table of `rustmath-finitefields` covers `(p, f)` the
//! **Conway polynomial lift** is used (coefficients in `[0, p)`), which makes
//! extensions of the same `p` compatible; otherwise the lexicographically
//! first monic irreducible found by enumeration is used. **In both cases the
//! modulus is self-certified** irreducible mod `p` at construction — nothing
//! is trusted blindly (campaign discipline after the `fp_factor` bug).
//!
//! # Model
//!
//! Elements are vectors `[a_0, ..., a_{f-1}]` over `Z/p^N` representing
//! `sum a_i g^i` where `g` is the class of `x` — the fixed-modulus model of
//! [`super::padic_integer::PadicInteger`], extended coefficientwise. All
//! results are exact functions of the canonical representatives, correct
//! modulo `p^N`.
//!
//! # Valuation (pinned convention)
//!
//! The unique extension `w` of `v_p` to `Q_q` has `w(p) = 1` and value group
//! `Z` (e = 1). For an integral element, `w(sum a_i g^i) = min_i v_p(a_i)`
//! (the basis `1, g, ..., g^{f-1}` reduces to a basis of the residue field).
//!
//! # Norm and trace laws (sympy-verified before implementation)
//!
//! - `N(x) = prod_{k<f} sigma^k(x)` and `Tr(x) = sum_k sigma^k(x)` where
//!   `sigma` is Frobenius, the unique automorphism with
//!   `sigma(a) = a^p mod p`; equal to `det`/`tr` of multiplication-by-`x`.
//!   [`UnramifiedElement::norm`]/[`trace`](UnramifiedElement::trace) compute
//!   **both and cross-certify** they agree.
//! - `v_p(N(x)) = f * w(x)` (with the `w(p) = 1` normalization above).

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use super::poly_arith::{
    canon, det_bareiss, eval_poly_ext, ext_inverse, is_irreducible_mod_p, polymul_mod,
    polypow_mod,
};

/// The unramified extension of `Q_p` of degree `f` at precision `p^N`.
#[derive(Clone, Debug)]
pub struct UnramifiedExtension {
    prime: Integer,
    degree: usize,
    precision: u32,
    /// Monic modulus `u`, little-endian, length `degree + 1`; irreducible mod
    /// `p` (certified at construction).
    modulus: Vec<Integer>,
    /// Whether `modulus` is the Conway polynomial lift.
    is_conway: bool,
    /// `sigma(g)` as a coefficient vector mod `p^N`: the Hensel-lifted root
    /// of `u` congruent to `g^p` mod `p` (certified at construction).
    frobenius_image: Vec<Integer>,
}

impl PartialEq for UnramifiedExtension {
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime
            && self.degree == other.degree
            && self.precision == other.precision
            && self.modulus == other.modulus
    }
}

/// The canonical monic irreducible-mod-`p` lift used for the degree-`f`
/// unramified extension: the Conway polynomial when the verified table covers
/// `(p, f)`, otherwise the first monic irreducible in the base-`p`
/// enumeration of constant-through-leading coefficients. Either way the
/// result is re-certified irreducible mod `p` here.
pub fn unramified_modulus(prime: &Integer, degree: usize) -> Result<UnivariatePolynomial<Integer>> {
    if degree == 0 {
        return Err(MathError::InvalidArgument(
            "unramified_modulus: degree must be >= 1".to_string(),
        ));
    }
    if !prime.is_prime() {
        return Err(MathError::InvalidArgument(
            "unramified_modulus: p must be prime".to_string(),
        ));
    }
    // Conway lookup (p must fit in u32)
    let p_small: Option<u32> = {
        let v = prime.to_i64();
        // to_i64 saturates for huge values; re-verify round-trip
        if Integer::from(v) == *prime && v > 0 && v <= u32::MAX as i64 {
            Some(v as u32)
        } else {
            None
        }
    };
    if let Some(ps) = p_small {
        if let Some(conway) = rustmath_finitefields::conway_polynomial(ps, degree) {
            let coeffs = conway.coefficients().to_vec();
            // never trust blindly: re-certify
            if is_irreducible_mod_p(&coeffs, prime)? {
                return Ok(conway);
            }
            return Err(MathError::NumericalError(
                "unramified_modulus: Conway table entry failed irreducibility \
                 certification — table corrupt?"
                    .to_string(),
            ));
        }
    }
    // Fallback: enumerate monic lifts x^f + a_{f-1} x^{f-1} + ... + a_0 with
    // a_i in [0, p), in base-p counter order, and take the first that
    // certifies irreducible mod p.
    let mut counter = Integer::zero();
    let bound = {
        let mut b = Integer::one();
        for _ in 0..degree {
            b = b * prime.clone();
        }
        b
    };
    while counter < bound {
        let mut coeffs = Vec::with_capacity(degree + 1);
        let mut c = counter.clone();
        for _ in 0..degree {
            coeffs.push(canon(c.clone(), prime));
            c = c / prime.clone();
        }
        coeffs.push(Integer::one());
        if is_irreducible_mod_p(&coeffs, prime)? {
            return Ok(UnivariatePolynomial::new(coeffs));
        }
        counter = counter + Integer::one();
    }
    // unreachable for a true prime: irreducibles of every degree exist
    Err(MathError::NumericalError(
        "unramified_modulus: no monic irreducible of the requested degree found \
         (impossible for a prime p)"
            .to_string(),
    ))
}

impl UnramifiedExtension {
    /// Degree-`f` unramified extension of `Q_p` at precision `p^N`, with the
    /// canonical modulus of [`unramified_modulus`].
    pub fn new(prime: Integer, degree: usize, precision: u32) -> Result<Arc<Self>> {
        let modulus = unramified_modulus(&prime, degree)?;
        Self::with_modulus(prime, &modulus, precision)
    }

    /// Unramified extension with a caller-supplied **monic** modulus, which
    /// is certified irreducible mod `p` (an error otherwise). The Frobenius
    /// image is Hensel-lifted and certified at construction.
    pub fn with_modulus(
        prime: Integer,
        modulus: &UnivariatePolynomial<Integer>,
        precision: u32,
    ) -> Result<Arc<Self>> {
        if !prime.is_prime() {
            return Err(MathError::InvalidArgument(
                "UnramifiedExtension: p must be prime".to_string(),
            ));
        }
        if precision == 0 {
            return Err(MathError::InvalidArgument(
                "UnramifiedExtension: precision must be >= 1".to_string(),
            ));
        }
        let degree = modulus.degree().ok_or_else(|| {
            MathError::InvalidArgument("UnramifiedExtension: zero modulus".to_string())
        })?;
        if degree == 0 {
            return Err(MathError::InvalidArgument(
                "UnramifiedExtension: modulus must have degree >= 1".to_string(),
            ));
        }
        let raw = modulus.coefficients();
        if !raw[degree].is_one() {
            return Err(MathError::InvalidArgument(
                "UnramifiedExtension: modulus must be monic (leading coefficient 1)"
                    .to_string(),
            ));
        }
        if !is_irreducible_mod_p(raw, &prime)? {
            return Err(MathError::InvalidArgument(
                "UnramifiedExtension: modulus is not irreducible mod p — the \
                 quotient is not a field (this rejects e.g. the old placeholder \
                 x^f - (1+p), which factors mod p)"
                    .to_string(),
            ));
        }
        let pn = prime.pow(precision);
        let modulus_vec: Vec<Integer> = raw.iter().map(|c| canon(c.clone(), &pn)).collect();
        // the reduction may not be monic anymore if precision changed values;
        // canon of 1 mod p^N (N>=1) is 1, so monicity survives.
        let frobenius_image =
            compute_frobenius_image(&modulus_vec, &prime, precision, degree)?;
        let ext = UnramifiedExtension {
            prime,
            degree,
            precision,
            modulus: modulus_vec,
            is_conway: false, // recomputed by tag_conway below
            frobenius_image,
        };
        Ok(Arc::new(ext.tag_conway()))
    }

    fn tag_conway(self) -> Self {
        // is_conway is informational; recompute from the table when possible
        let p_small = {
            let v = self.prime.to_i64();
            if Integer::from(v) == self.prime && v > 0 && v <= u32::MAX as i64 {
                Some(v as u32)
            } else {
                None
            }
        };
        let is_conway = p_small
            .and_then(|ps| rustmath_finitefields::conway_polynomial(ps, self.degree))
            .map(|c| {
                let pn = self.prime.pow(self.precision);
                let cv: Vec<Integer> =
                    c.coefficients().iter().map(|x| canon(x.clone(), &pn)).collect();
                cv == self.modulus
            })
            .unwrap_or(false);
        UnramifiedExtension { is_conway, ..self }
    }

    /// The prime `p`.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// The residue degree `f` (= extension degree; e = 1).
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Absolute precision `N`: elements are known mod `p^N`.
    pub fn precision(&self) -> u32 {
        self.precision
    }

    /// The monic modulus `u` (little-endian, reduced mod `p^N`).
    pub fn modulus(&self) -> &[Integer] {
        &self.modulus
    }

    /// Whether the modulus is the Conway polynomial lift.
    pub fn is_conway(&self) -> bool {
        self.is_conway
    }

    /// `sigma(g)` where `sigma` is Frobenius: the unique root of `u`
    /// congruent to `g^p` mod `p` (Hensel-lifted, certified).
    pub fn frobenius_image(&self) -> &[Integer] {
        &self.frobenius_image
    }

    /// Zero element.
    pub fn zero(self: &Arc<Self>) -> UnramifiedElement {
        UnramifiedElement {
            ext: self.clone(),
            coeffs: vec![Integer::zero(); self.degree],
        }
    }

    /// One element.
    pub fn one(self: &Arc<Self>) -> UnramifiedElement {
        let mut coeffs = vec![Integer::zero(); self.degree];
        coeffs[0] = Integer::one();
        UnramifiedElement {
            ext: self.clone(),
            coeffs,
        }
    }

    /// The generator `g` (class of `x`).
    pub fn generator(self: &Arc<Self>) -> UnramifiedElement {
        let mut coeffs = vec![Integer::zero(); self.degree];
        if self.degree >= 2 {
            coeffs[1] = Integer::one();
        } else {
            // degree 1: x = -u_0 (u = x + u_0 monic)
            let pn = self.prime.pow(self.precision);
            coeffs[0] = canon(-self.modulus[0].clone(), &pn);
        }
        UnramifiedElement {
            ext: self.clone(),
            coeffs,
        }
    }

    /// Element from little-endian `Z`-coefficients (length <= f; padded,
    /// reduced mod `p^N`).
    pub fn element(self: &Arc<Self>, coeffs: &[Integer]) -> Result<UnramifiedElement> {
        if coeffs.len() > self.degree {
            return Err(MathError::InvalidArgument(format!(
                "UnramifiedExtension::element: {} coefficients for degree {}",
                coeffs.len(),
                self.degree
            )));
        }
        let pn = self.prime.pow(self.precision);
        let mut v: Vec<Integer> = coeffs.iter().map(|c| canon(c.clone(), &pn)).collect();
        v.resize(self.degree, Integer::zero());
        Ok(UnramifiedElement {
            ext: self.clone(),
            coeffs: v,
        })
    }

    fn pn(&self) -> Integer {
        self.prime.pow(self.precision)
    }
}

/// Hensel-lift the root of `u` starting from `x^p mod (u, p)`; certified:
/// `u(r) = 0 mod p^N` and `r = x^p mod p`.
fn compute_frobenius_image(
    modulus: &[Integer],
    p: &Integer,
    precision: u32,
    degree: usize,
) -> Result<Vec<Integer>> {
    let x_vec = {
        let mut v = vec![Integer::zero(); degree.max(1)];
        if degree >= 2 {
            v[1] = Integer::one();
        } else {
            v[0] = canon(-modulus[0].clone(), p);
        }
        v
    };
    let r0 = if degree >= 2 {
        polypow_mod(&x_vec, p, modulus, p)
    } else {
        x_vec.clone()
    };
    let uprime: Vec<Integer> = (1..modulus.len())
        .map(|i| Integer::from(i as i64) * modulus[i].clone())
        .collect();
    let mut r = r0.clone();
    let mut k: u32 = 1;
    while k < precision {
        k = (2 * k).min(precision);
        let mk = p.pow(k);
        let ur = eval_poly_ext(modulus, &r, modulus, &mk);
        let upr = eval_poly_ext(&uprime, &r, modulus, &mk);
        // u is separable mod p (irreducible over a finite field), so u'(r)
        // is a unit; ext_inverse self-certifies.
        let upr_inv = ext_inverse(&upr, modulus, p, k)?;
        let corr = polymul_mod(&ur, &upr_inv, modulus, &mk);
        for i in 0..degree {
            r[i] = canon(r[i].clone() - corr[i].clone(), &mk);
        }
    }
    // certify the lift
    let pn = p.pow(precision);
    let ur = eval_poly_ext(modulus, &r, modulus, &pn);
    if ur.iter().any(|c| !c.is_zero()) {
        return Err(MathError::NumericalError(
            "Frobenius certification failed: u(r) != 0 mod p^N".to_string(),
        ));
    }
    for (a, b) in r.iter().zip(r0.iter()) {
        if canon(a.clone() - b.clone(), p) != Integer::zero() {
            return Err(MathError::NumericalError(
                "Frobenius certification failed: r != x^p mod p".to_string(),
            ));
        }
    }
    Ok(r)
}

/// An element of `Z_q = Z_p[x]/(u)` at precision `p^N` (integral elements;
/// negative-valuation field elements are deliberately out of scope for this
/// stage — see module docs).
#[derive(Clone, Debug)]
pub struct UnramifiedElement {
    ext: Arc<UnramifiedExtension>,
    /// length `f`, canonical mod `p^N`
    coeffs: Vec<Integer>,
}

impl PartialEq for UnramifiedElement {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.ext, &other.ext) || *self.ext == *other.ext)
            && self.coeffs == other.coeffs
    }
}

impl UnramifiedElement {
    /// The parent extension.
    pub fn extension(&self) -> &Arc<UnramifiedExtension> {
        &self.ext
    }

    /// Coefficients `[a_0, ..., a_{f-1}]`, canonical mod `p^N`.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coeffs
    }

    fn assert_same_ext(&self, other: &Self) {
        assert!(
            Arc::ptr_eq(&self.ext, &other.ext) || *self.ext == *other.ext,
            "UnramifiedElement: elements from different extensions"
        );
    }

    /// Valuation `w` with `w(p) = 1`: `min_i v_p(a_i)`, computed on the
    /// canonical representative. `None` means zero to precision (`w >= N`).
    pub fn valuation(&self) -> Option<u32> {
        self.coeffs
            .iter()
            .filter(|c| !c.is_zero())
            .map(|c| c.valuation(&self.ext.prime))
            .min()
    }

    /// Frobenius `sigma` (the unique lift of `a -> a^p`), applied by
    /// evaluating the coefficient polynomial at the certified root
    /// `sigma(g)`: `sigma(sum a_i g^i) = sum a_i sigma(g)^i`.
    pub fn frobenius(&self) -> Self {
        let pn = self.ext.pn();
        let coeffs = eval_poly_ext(
            &self.coeffs,
            &self.ext.frobenius_image,
            &self.ext.modulus,
            &pn,
        );
        UnramifiedElement {
            ext: self.ext.clone(),
            coeffs,
        }
    }

    /// Multiplicative inverse (units only, i.e. `w(x) = 0`); self-certified
    /// by `ext_inverse`.
    pub fn inverse(&self) -> Result<Self> {
        let z = ext_inverse(
            &self.coeffs,
            &self.ext.modulus,
            &self.ext.prime,
            self.ext.precision,
        )?;
        Ok(UnramifiedElement {
            ext: self.ext.clone(),
            coeffs: z,
        })
    }

    /// The `f x f` multiplication-by-`self` matrix over `Z/p^N` in the basis
    /// `1, g, ..., g^{f-1}` (column `j` = coefficients of `self * g^j`).
    fn multiplication_matrix(&self) -> Vec<Vec<Integer>> {
        let f = self.ext.degree;
        let pn = self.ext.pn();
        let mut cols = Vec::with_capacity(f);
        for j in 0..f {
            let mut basis = vec![Integer::zero(); f];
            basis[j] = Integer::one();
            cols.push(polymul_mod(&self.coeffs, &basis, &self.ext.modulus, &pn));
        }
        (0..f)
            .map(|i| (0..f).map(|j| cols[j][i].clone()).collect())
            .collect()
    }

    /// Norm via `det` of the multiplication matrix (exact integer Bareiss
    /// determinant of the canonical lifts, reduced mod `p^N`).
    pub fn norm_via_matrix(&self) -> Integer {
        let pn = self.ext.pn();
        canon(det_bareiss(self.multiplication_matrix()), &pn)
    }

    /// Trace via the multiplication matrix (sum of the diagonal mod `p^N`).
    pub fn trace_via_matrix(&self) -> Integer {
        let pn = self.ext.pn();
        let m = self.multiplication_matrix();
        let mut t = Integer::zero();
        for (i, row) in m.iter().enumerate() {
            t = canon(t + row[i].clone(), &pn);
        }
        t
    }

    /// Norm as the product of the Frobenius orbit
    /// `prod_{k=0}^{f-1} sigma^k(x)`. Errors if the product does not land in
    /// `Z_p` (which would mean broken Frobenius arithmetic).
    pub fn norm_via_frobenius(&self) -> Result<Integer> {
        let (prod, _) = self.frobenius_orbit_product_sum();
        if prod.coeffs[1..].iter().any(|c| !c.is_zero()) {
            return Err(MathError::NumericalError(
                "norm_via_frobenius: orbit product did not land in Z_p".to_string(),
            ));
        }
        Ok(prod.coeffs[0].clone())
    }

    /// Trace as the sum of the Frobenius orbit. Errors if the sum does not
    /// land in `Z_p`.
    pub fn trace_via_frobenius(&self) -> Result<Integer> {
        let (_, sum) = self.frobenius_orbit_product_sum();
        if sum.coeffs[1..].iter().any(|c| !c.is_zero()) {
            return Err(MathError::NumericalError(
                "trace_via_frobenius: orbit sum did not land in Z_p".to_string(),
            ));
        }
        Ok(sum.coeffs[0].clone())
    }

    fn frobenius_orbit_product_sum(&self) -> (Self, Self) {
        let f = self.ext.degree;
        let mut prod = UnramifiedExtension::one(&self.ext);
        let mut sum = UnramifiedExtension::zero(&self.ext);
        let mut conj = self.clone();
        for k in 0..f {
            prod = prod * conj.clone();
            sum = sum + conj.clone();
            if k + 1 < f {
                conj = conj.frobenius();
            }
        }
        (prod, sum)
    }

    /// Galois conjugates `[x, sigma(x), ..., sigma^{f-1}(x)]`.
    pub fn conjugates(&self) -> Vec<Self> {
        let f = self.ext.degree;
        let mut out = Vec::with_capacity(f);
        let mut c = self.clone();
        for _ in 0..f {
            out.push(c.clone());
            c = c.frobenius();
        }
        out
    }

    /// Norm `N_{Q_q/Q_p}(x)` mod `p^N`, computed **both** as the Frobenius
    /// orbit product and as the multiplication-matrix determinant, and
    /// cross-certified equal (an error means an arithmetic bug, never a
    /// silently wrong value).
    pub fn norm(&self) -> Result<Integer> {
        let via_matrix = self.norm_via_matrix();
        let via_frob = self.norm_via_frobenius()?;
        if via_matrix != via_frob {
            return Err(MathError::NumericalError(format!(
                "norm self-certification failed: matrix={} frobenius={}",
                via_matrix, via_frob
            )));
        }
        Ok(via_matrix)
    }

    /// Trace `Tr_{Q_q/Q_p}(x)` mod `p^N`, cross-certified between the orbit
    /// sum and the matrix trace.
    pub fn trace(&self) -> Result<Integer> {
        let via_matrix = self.trace_via_matrix();
        let via_frob = self.trace_via_frobenius()?;
        if via_matrix != via_frob {
            return Err(MathError::NumericalError(format!(
                "trace self-certification failed: matrix={} frobenius={}",
                via_matrix, via_frob
            )));
        }
        Ok(via_matrix)
    }
}

impl fmt::Display for UnramifiedElement {
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
                1 => write!(f, "{}*g", c)?,
                _ => write!(f, "{}*g^{}", c, i)?,
            }
        }
        if first {
            write!(f, "0")?;
        }
        write!(f, " + O({}^{})", self.ext.prime, self.ext.precision)
    }
}

impl Add for UnramifiedElement {
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
        UnramifiedElement {
            ext: self.ext,
            coeffs,
        }
    }
}

impl Sub for UnramifiedElement {
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
        UnramifiedElement {
            ext: self.ext,
            coeffs,
        }
    }
}

impl Mul for UnramifiedElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.assert_same_ext(&other);
        let pn = self.ext.pn();
        let coeffs = polymul_mod(&self.coeffs, &other.coeffs, &self.ext.modulus, &pn);
        UnramifiedElement {
            ext: self.ext,
            coeffs,
        }
    }
}

impl Neg for UnramifiedElement {
    type Output = Self;
    fn neg(self) -> Self {
        let pn = self.ext.pn();
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| canon(-c.clone(), &pn))
            .collect();
        UnramifiedElement {
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

    /// Conway(5,3) = x^3 + 3x + 3. Every expected value below was verified
    /// independently in python/sympy (resultants + companion matrices +
    /// an independent Hensel implementation) BEFORE this Rust existed.
    fn q5_cubed() -> Arc<UnramifiedExtension> {
        UnramifiedExtension::new(Integer::from(5), 3, 12).unwrap()
    }

    #[test]
    fn test_construction_uses_conway_and_certifies() {
        let ext = q5_cubed();
        assert_eq!(ext.modulus(), &ints(&[3, 3, 0, 1])[..]); // Conway(5,3)
        assert!(ext.is_conway());
        assert_eq!(ext.degree(), 3);
        // frobenius image certified at construction (sympy got
        // [3523149, 58392794, 123831887] mod 5^12 with the same setup)
        assert_eq!(
            ext.frobenius_image(),
            &ints(&[3523149, 58392794, 123831887])[..]
        );
    }

    #[test]
    fn test_frobenius_order_f() {
        let ext = q5_cubed();
        let g = ext.generator();
        let s1 = g.frobenius();
        let s2 = s1.frobenius();
        let s3 = s2.frobenius();
        assert_ne!(s1, g, "Frobenius must move the generator");
        assert_ne!(s2, g);
        assert_ne!(s2, s1);
        assert_eq!(s3, g, "sigma^f = identity");
    }

    #[test]
    fn test_frobenius_is_ring_hom_and_reduces_to_pth_power() {
        let ext = q5_cubed();
        let a = ext.element(&ints(&[2, 3, 1])).unwrap();
        let b = ext.element(&ints(&[4, 0, 2])).unwrap();
        // sigma(ab) = sigma(a)sigma(b), sigma(a+b) = sigma(a)+sigma(b)
        assert_eq!(
            (a.clone() * b.clone()).frobenius(),
            a.frobenius() * b.frobenius()
        );
        assert_eq!(
            (a.clone() + b.clone()).frobenius(),
            a.frobenius() + b.frobenius()
        );
        // sigma(a) = a^5 mod 5
        let a5 = a.clone() * a.clone() * a.clone() * a.clone() * a.clone();
        let sa = a.frobenius();
        let p = Integer::from(5);
        for (x, y) in sa.coefficients().iter().zip(a5.coefficients().iter()) {
            assert_eq!(canon(x.clone() - y.clone(), &p), Integer::zero());
        }
    }

    #[test]
    fn test_norm_trace_generator_sympy_values() {
        // sympy: N(g) = Res(u, x) = -3, Tr(g) = 0 for u = x^3 + 3x + 3
        let ext = q5_cubed();
        let g = ext.generator();
        let pn = Integer::from(5).pow(12);
        assert_eq!(g.norm().unwrap(), canon(Integer::from(-3), &pn));
        assert_eq!(g.trace().unwrap(), Integer::zero());
    }

    #[test]
    fn test_norm_trace_element_sympy_values() {
        // sympy: N(2 + 3g + g^2) = 11, Tr(2 + 3g + g^2) = 0
        let ext = q5_cubed();
        let a = ext.element(&ints(&[2, 3, 1])).unwrap();
        assert_eq!(a.norm().unwrap(), Integer::from(11));
        assert_eq!(a.trace().unwrap(), Integer::zero());
        // sympy: Tr(5 + 10g) = 15
        let b = ext.element(&ints(&[5, 10, 0])).unwrap();
        assert_eq!(b.trace().unwrap(), Integer::from(15));
    }

    #[test]
    fn test_conway52_norm_trace() {
        // sympy: for Conway(5,2) = x^2 + 4x + 2: N(g) = 2, Tr(g) = -4
        let ext = UnramifiedExtension::new(Integer::from(5), 2, 12).unwrap();
        assert_eq!(ext.modulus(), &ints(&[2, 4, 1])[..]);
        let g = ext.generator();
        let pn = Integer::from(5).pow(12);
        assert_eq!(g.norm().unwrap(), Integer::from(2));
        assert_eq!(g.trace().unwrap(), canon(Integer::from(-4), &pn));
    }

    #[test]
    fn test_degree_one_norm_trace_identity() {
        // degree-1 sanity: N = Tr = identity
        let ext =
            UnramifiedExtension::with_modulus(Integer::from(7), &UnivariatePolynomial::new(
                ints(&[3, 1]), // x + 3, "generator" = -3
            ), 8)
            .unwrap();
        let a = ext.element(&ints(&[10])).unwrap();
        assert_eq!(a.norm().unwrap(), Integer::from(10));
        assert_eq!(a.trace().unwrap(), Integer::from(10));
        let g = ext.generator();
        let pn = Integer::from(7).pow(8);
        assert_eq!(g.norm().unwrap(), canon(Integer::from(-3), &pn));
    }

    #[test]
    fn test_law_battery_random() {
        // N(xy) = N(x)N(y); Tr(x+y) = Tr(x)+Tr(y); N = prod of conjugates;
        // Tr = sum of conjugates. Deterministic LCG "random" elements.
        let ext = q5_cubed();
        let pn = Integer::from(5).pow(12);
        let mut state: u64 = 0xDEADBEEF;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };
        for _ in 0..12 {
            let a = ext
                .element(&ints(&[next() % 1000, next() % 1000, next() % 1000]))
                .unwrap();
            let b = ext
                .element(&ints(&[next() % 1000, next() % 1000, next() % 1000]))
                .unwrap();
            let n_ab = (a.clone() * b.clone()).norm().unwrap();
            let na_nb = canon(a.norm().unwrap() * b.norm().unwrap(), &pn);
            assert_eq!(n_ab, na_nb, "N(xy) != N(x)N(y)");
            let t_ab = (a.clone() + b.clone()).trace().unwrap();
            let ta_tb = canon(a.trace().unwrap() + b.trace().unwrap(), &pn);
            assert_eq!(t_ab, ta_tb, "Tr(x+y) != Tr(x)+Tr(y)");
            // conjugate product/sum (this is what norm()/trace() certify
            // against the matrix internally, but assert explicitly too)
            let conjs = a.conjugates();
            let mut prod = ext.one();
            let mut sum = ext.zero();
            for c in conjs {
                prod = prod * c.clone();
                sum = sum + c;
            }
            assert_eq!(prod.coefficients()[0], a.norm().unwrap());
            assert!(prod.coefficients()[1..].iter().all(|c| c.is_zero()));
            assert_eq!(sum.coefficients()[0], a.trace().unwrap());
            assert!(sum.coefficients()[1..].iter().all(|c| c.is_zero()));
        }
    }

    #[test]
    fn test_norm_valuation_law() {
        // sympy-verified law: v_p(N(x)) = f * w(x), w = min_i v_p(a_i)
        let ext = q5_cubed();
        for coeffs in [[5i64, 10, 0], [25, 0, 50], [5, 1, 0], [0, 0, 75]] {
            let a = ext.element(&ints(&coeffs)).unwrap();
            let w = a.valuation().unwrap();
            let n = a.norm().unwrap();
            assert_eq!(
                n.valuation(&Integer::from(5)),
                3 * w,
                "v_p(N(x)) != f*w(x) for {:?}",
                coeffs
            );
        }
    }

    #[test]
    fn test_inverse_units() {
        let ext = q5_cubed();
        let a = ext.element(&ints(&[2, 3, 1])).unwrap();
        let inv = a.inverse().unwrap();
        assert_eq!(a.clone() * inv, ext.one());
        // p is not a unit
        let p_elem = ext.element(&ints(&[5, 0, 0])).unwrap();
        assert!(p_elem.inverse().is_err());
    }

    #[test]
    fn test_placeholder_polynomial_rejected() {
        // x^2 - 6 is NOT irreducible mod 5 — the bug in the old
        // PadicExtension::unramified placeholder; must be rejected here.
        let bad = UnivariatePolynomial::new(ints(&[-6, 0, 1]));
        assert!(UnramifiedExtension::with_modulus(Integer::from(5), &bad, 8).is_err());
    }

    #[test]
    fn test_fallback_search_when_no_conway_entry() {
        // (p, f) = (37, 2) is outside the Conway table: fallback enumeration
        // must produce a certified irreducible monic quadratic mod 37.
        let ext = UnramifiedExtension::new(Integer::from(37), 2, 6).unwrap();
        assert!(!ext.is_conway());
        assert!(is_irreducible_mod_p(ext.modulus(), &Integer::from(37)).unwrap());
        // laws still hold
        let g = ext.generator();
        let pn = Integer::from(37).pow(6);
        // N(g) = constant term of modulus (degree 2: g * sigma(g) = u_0)
        let u0 = ext.modulus()[0].clone();
        assert_eq!(g.norm().unwrap(), canon(u0, &pn));
    }

    #[test]
    fn test_valuation() {
        let ext = q5_cubed();
        assert_eq!(ext.element(&ints(&[5, 10, 0])).unwrap().valuation(), Some(1));
        assert_eq!(ext.element(&ints(&[25, 0, 50])).unwrap().valuation(), Some(2));
        assert_eq!(ext.element(&ints(&[5, 1, 0])).unwrap().valuation(), Some(0));
        assert_eq!(ext.zero().valuation(), None);
    }
}

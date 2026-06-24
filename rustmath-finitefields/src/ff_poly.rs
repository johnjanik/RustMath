//! Polynomial arithmetic over finite fields, plus a self-contained GF(p^n) type.
//!
//! This module provides the computational substrate used by the
//! Cantor–Zassenhaus factorization and irreducible-polynomial search:
//!
//! * [`FiniteFieldElement`] — a trait abstracting a finite field with a known
//!   order `q = p^k`. Implemented for [`crate::prime_field::PrimeField`] (GF(p))
//!   and for [`Gfpn`] (GF(p^n)).
//! * [`FFPoly`] — a dense univariate polynomial whose coefficients live in a
//!   `FiniteFieldElement`. We do **not** reuse
//!   `rustmath_polynomials::UnivariatePolynomial<PrimeField>` because that type
//!   calls `R::zero()` / `R::one()`, which panic for `PrimeField` (the modulus
//!   is not known statically). `FFPoly` instead carries a sample element of the
//!   field so it can synthesise zero/one without a static constructor.
//! * [`Gfpn`] — a fully working GF(p^n) element type (multiplication reduced
//!   modulo a defining irreducible polynomial, inverse via the extended
//!   Euclidean algorithm). This also supplies the reduction helpers needed for
//!   GF(p^n) element arithmetic.

use crate::prime_field::PrimeField;
use rustmath_core::{EuclideanDomain, Field, MathError, Result, Ring};
use rustmath_integers::Integer;
use std::fmt;

/// A finite field whose elements support exact arithmetic and equality.
///
/// Implementors describe a field GF(q) with `q = characteristic^degree`.
/// The trait deliberately works with owned values (`Clone`) to keep the
/// factorization code readable; the fields in question are small.
pub trait FiniteFieldElement: Clone + PartialEq + fmt::Debug {
    /// Additive identity of the field.
    fn zero(&self) -> Self;
    /// Multiplicative identity of the field.
    fn one(&self) -> Self;
    /// True if this element is the additive identity.
    fn is_zero(&self) -> bool;
    /// True if this element is the multiplicative identity.
    fn is_one(&self) -> bool;

    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    /// Multiplicative inverse. Errors on zero.
    fn invert(&self) -> Result<Self>;

    /// Characteristic `p` of the field.
    fn characteristic(&self) -> Integer;
    /// Order `q = p^k` of the field.
    fn order(&self) -> Integer;

    /// Construct the field element corresponding to the integer `n`
    /// (i.e. `n` copies of the multiplicative identity), in the same field.
    fn from_int(&self, n: &Integer) -> Self {
        let mut result = self.zero();
        let one = self.one();
        let mut k = Integer::zero();
        let target = {
            let q = self.characteristic();
            // reduce n modulo characteristic for the prime-subfield image
            let (_, r) = n.div_rem(&q).unwrap();
            if r.signum() < 0 {
                r + q
            } else {
                r
            }
        };
        while k < target {
            result = result.add(&one);
            k = k + Integer::one();
        }
        result
    }

    /// Raise to an integer power using square-and-multiply.
    fn pow(&self, exp: &Integer) -> Self {
        let mut result = self.one();
        let mut base = self.clone();
        let mut e = exp.clone();
        let two = Integer::from(2);
        while e > Integer::zero() {
            let (q, r) = e.div_rem(&two).unwrap();
            if r.is_one() {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            e = q;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// GF(p) via PrimeField
// ---------------------------------------------------------------------------

impl FiniteFieldElement for PrimeField {
    fn zero(&self) -> Self {
        PrimeField::new(Integer::zero(), self.modulus().clone()).unwrap()
    }
    fn one(&self) -> Self {
        PrimeField::new(Integer::one(), self.modulus().clone()).unwrap()
    }
    fn is_zero(&self) -> bool {
        Ring::is_zero(self)
    }
    fn is_one(&self) -> bool {
        Ring::is_one(self)
    }
    fn add(&self, other: &Self) -> Self {
        self.clone() + other.clone()
    }
    fn sub(&self, other: &Self) -> Self {
        self.clone() - other.clone()
    }
    fn mul(&self, other: &Self) -> Self {
        self.clone() * other.clone()
    }
    fn neg(&self) -> Self {
        -self.clone()
    }
    fn invert(&self) -> Result<Self> {
        Field::inverse(self)
    }
    fn characteristic(&self) -> Integer {
        self.modulus().clone()
    }
    fn order(&self) -> Integer {
        self.modulus().clone()
    }
}

// ---------------------------------------------------------------------------
// Dense polynomials over a finite field
// ---------------------------------------------------------------------------

/// A dense univariate polynomial with coefficients in a [`FiniteFieldElement`].
///
/// Coefficients are stored in increasing-degree order with no trailing zeros
/// (except that the zero polynomial is represented by an empty coefficient
/// vector). A `sample` field element carries the field context so the type can
/// produce zero/one of the coefficient field on demand.
#[derive(Clone, Debug)]
pub struct FFPoly<F: FiniteFieldElement> {
    coeffs: Vec<F>,
    sample: F,
}

impl<F: FiniteFieldElement> FFPoly<F> {
    /// Build a polynomial from coefficients (low degree first), trimming
    /// trailing zeros. `sample` must be any element of the coefficient field.
    pub fn new(coeffs: Vec<F>, sample: F) -> Self {
        let mut p = FFPoly { coeffs, sample };
        p.normalize();
        p
    }

    /// The zero polynomial over the field of `sample`.
    pub fn zero(sample: F) -> Self {
        FFPoly {
            coeffs: Vec::new(),
            sample,
        }
    }

    /// The constant polynomial `1`.
    pub fn one(sample: F) -> Self {
        let one = sample.one();
        FFPoly {
            coeffs: vec![one],
            sample,
        }
    }

    /// The monomial `x` (degree 1).
    pub fn x(sample: F) -> Self {
        let zero = sample.zero();
        let one = sample.one();
        FFPoly::new(vec![zero, one], sample)
    }

    /// Constant polynomial equal to the given field element.
    pub fn constant(c: F) -> Self {
        let sample = c.clone();
        FFPoly::new(vec![c], sample)
    }

    fn normalize(&mut self) {
        while let Some(last) = self.coeffs.last() {
            if last.is_zero() {
                self.coeffs.pop();
            } else {
                break;
            }
        }
    }

    /// True if this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Degree of the polynomial, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Coefficient slice (low degree first).
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// A sample element of the coefficient field.
    pub fn sample(&self) -> &F {
        &self.sample
    }

    /// Leading coefficient, or `None` for the zero polynomial.
    pub fn leading(&self) -> Option<&F> {
        self.coeffs.last()
    }

    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i);
            let b = other.coeffs.get(i);
            let v = match (a, b) {
                (Some(a), Some(b)) => a.add(b),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => self.sample.zero(),
            };
            out.push(v);
        }
        FFPoly::new(out, self.sample.clone())
    }

    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i);
            let b = other.coeffs.get(i);
            let v = match (a, b) {
                (Some(a), Some(b)) => a.sub(b),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.neg(),
                (None, None) => self.sample.zero(),
            };
            out.push(v);
        }
        FFPoly::new(out, self.sample.clone())
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return FFPoly::zero(self.sample.clone());
        }
        let mut out = vec![self.sample.zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                if b.is_zero() {
                    continue;
                }
                let prod = a.mul(b);
                out[i + j] = out[i + j].add(&prod);
            }
        }
        FFPoly::new(out, self.sample.clone())
    }

    /// Multiply every coefficient by a field scalar.
    pub fn scalar_mul(&self, scalar: &F) -> Self {
        let out: Vec<F> = self.coeffs.iter().map(|c| c.mul(scalar)).collect();
        FFPoly::new(out, self.sample.clone())
    }

    /// Return the monic associate (divide by the leading coefficient).
    /// The zero polynomial is returned unchanged.
    pub fn make_monic(&self) -> Self {
        match self.leading() {
            None => self.clone(),
            Some(lead) => {
                if lead.is_one() {
                    return self.clone();
                }
                let inv = lead.invert().expect("leading coeff nonzero");
                self.scalar_mul(&inv)
            }
        }
    }

    /// True if monic (leading coefficient is 1).
    pub fn is_monic(&self) -> bool {
        self.leading().map(|c| c.is_one()).unwrap_or(false)
    }

    /// Polynomial long division. Returns `(quotient, remainder)` with
    /// `self == quotient * divisor + remainder` and `deg(remainder) < deg(divisor)`.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self)> {
        if divisor.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let zero = self.sample.zero();
        let ddeg = divisor.degree().unwrap();
        let dlead_inv = divisor.leading().unwrap().invert()?;

        // If the dividend has smaller degree, the quotient is zero.
        let ndeg = match self.degree() {
            None => return Ok((FFPoly::zero(self.sample.clone()), self.clone())),
            Some(d) => d,
        };
        if ndeg < ddeg {
            return Ok((FFPoly::zero(self.sample.clone()), self.clone()));
        }

        // `rem` holds the running remainder coefficients (low degree first);
        // it is exactly the dividend's length and we zero out high terms as we
        // cancel them. `quot` has length ndeg - ddeg + 1.
        let mut rem = self.coeffs.clone();
        let mut quot = vec![zero.clone(); ndeg - ddeg + 1];

        // Process from the highest degree of the dividend down to ddeg.
        for rdeg in (ddeg..=ndeg).rev() {
            if rem[rdeg].is_zero() {
                continue;
            }
            let shift = rdeg - ddeg;
            let factor = rem[rdeg].mul(&dlead_inv);
            quot[shift] = factor.clone();
            // Subtract factor * x^shift * divisor from rem.
            for (j, dc) in divisor.coeffs.iter().enumerate() {
                let prod = factor.mul(dc);
                rem[shift + j] = rem[shift + j].sub(&prod);
            }
        }

        let q = FFPoly::new(quot, self.sample.clone());
        let r = FFPoly::new(rem, self.sample.clone());
        Ok((q, r))
    }

    /// Remainder of `self` divided by `divisor`.
    pub fn rem(&self, divisor: &Self) -> Result<Self> {
        Ok(self.div_rem(divisor)?.1)
    }

    /// Monic greatest common divisor via the Euclidean algorithm.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let r = a.rem(&b).expect("nonzero divisor");
            a = b;
            b = r;
        }
        if a.is_zero() {
            a
        } else {
            a.make_monic()
        }
    }

    /// Formal derivative.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return FFPoly::zero(self.sample.clone());
        }
        let mut out = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            let mult = self.sample.from_int(&Integer::from(i as i64));
            out.push(c.mul(&mult));
        }
        FFPoly::new(out, self.sample.clone())
    }

    /// Compute `self^exp mod modulus`, by repeated squaring with reduction.
    pub fn pow_mod(&self, exp: &Integer, modulus: &Self) -> Result<Self> {
        let mut result = FFPoly::one(self.sample.clone()).rem(modulus)?;
        let mut base = self.rem(modulus)?;
        let mut e = exp.clone();
        let two = Integer::from(2);
        while e > Integer::zero() {
            let (q, r) = e.div_rem(&two).unwrap();
            if r.is_one() {
                result = result.mul(&base).rem(modulus)?;
            }
            base = base.mul(&base).rem(modulus)?;
            e = q;
        }
        Ok(result)
    }

    /// Evaluate at a field point.
    pub fn evaluate(&self, point: &F) -> F {
        let mut acc = self.sample.zero();
        for c in self.coeffs.iter().rev() {
            acc = acc.mul(point).add(c);
        }
        acc
    }
}

impl<F: FiniteFieldElement> PartialEq for FFPoly<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<F: FiniteFieldElement + fmt::Display> fmt::Display for FFPoly<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
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
                1 => write!(f, "({})*x", c)?,
                _ => write!(f, "({})*x^{}", c, i)?,
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GF(p^n) element type
// ---------------------------------------------------------------------------

/// An element of GF(p^n), represented as a polynomial in `x` of degree < n,
/// reduced modulo a defining irreducible polynomial over GF(p).
///
/// Unlike [`crate::extension_field::ExtensionField`], this type performs full
/// modular reduction on multiplication and supports inversion via the extended
/// Euclidean algorithm, so it forms a genuine field and can serve as the
/// coefficient field for [`FFPoly`].
#[derive(Clone, Debug)]
pub struct Gfpn {
    /// Coefficients in GF(p), low degree first, length exactly `n` (the
    /// extension degree). Stored as integers in `[0, p)`.
    coeffs: Vec<Integer>,
    /// Characteristic p.
    p: Integer,
    /// Defining irreducible polynomial over GF(p), monic, degree n, length
    /// `n + 1`. Stored as integers in `[0, p)`.
    modulus: Vec<Integer>,
}

impl Gfpn {
    fn redp(v: Integer, p: &Integer) -> Integer {
        let (_, r) = v.div_rem(p).unwrap();
        if r.signum() < 0 {
            r + p.clone()
        } else {
            r
        }
    }

    /// Extension degree `n`.
    pub fn extension_degree(&self) -> usize {
        self.modulus.len() - 1
    }

    /// Build an element from coefficients (low degree first) modulo the field
    /// defined by `p` and the monic irreducible `modulus` (low degree first).
    pub fn new(coeffs: Vec<Integer>, p: Integer, modulus: Vec<Integer>) -> Self {
        let n = modulus.len() - 1;
        let mut c = vec![Integer::zero(); n];
        for (i, v) in coeffs.into_iter().enumerate() {
            if i < n {
                c[i] = Gfpn::redp(v, &p);
            } else {
                // higher-degree terms must be reduced; do it via a temporary poly
                // (rare path: callers usually pass reduced inputs)
                let mut tmp = Gfpn {
                    coeffs: vec![Integer::zero(); n],
                    p: p.clone(),
                    modulus: modulus.clone(),
                };
                tmp.coeffs[0] = Gfpn::redp(v, &p);
                // multiply by x^i
                for _ in 0..i {
                    tmp = tmp.mul_by_x();
                }
                for k in 0..n {
                    c[k] = Gfpn::redp(c[k].clone() + tmp.coeffs[k].clone(), &p);
                }
            }
        }
        Gfpn {
            coeffs: c,
            p,
            modulus,
        }
    }

    fn mul_by_x(&self) -> Gfpn {
        let n = self.extension_degree();
        let mut shifted = vec![Integer::zero(); n + 1];
        for i in 0..n {
            shifted[i + 1] = self.coeffs[i].clone();
        }
        // reduce: shifted has degree up to n; subtract leading * modulus
        if !shifted[n].is_zero() {
            let lead = shifted[n].clone();
            // modulus is monic, so subtract lead * modulus (degree n)
            for i in 0..=n {
                shifted[i] = Gfpn::redp(
                    shifted[i].clone() - lead.clone() * self.modulus[i].clone(),
                    &self.p,
                );
            }
        }
        shifted.truncate(n);
        Gfpn {
            coeffs: shifted,
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        }
    }

    /// Coefficient vector (length n, low degree first).
    pub fn coeffs(&self) -> &[Integer] {
        &self.coeffs
    }
    /// The defining modulus polynomial.
    pub fn modulus(&self) -> &[Integer] {
        &self.modulus
    }
}

impl PartialEq for Gfpn {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs && self.p == other.p && self.modulus == other.modulus
    }
}

impl fmt::Display for Gfpn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} in GF({}^{})", self.coeffs, self.p, self.extension_degree())
    }
}

impl FiniteFieldElement for Gfpn {
    fn zero(&self) -> Self {
        Gfpn {
            coeffs: vec![Integer::zero(); self.extension_degree()],
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        }
    }
    fn one(&self) -> Self {
        let n = self.extension_degree();
        let mut c = vec![Integer::zero(); n];
        c[0] = Integer::one();
        Gfpn {
            coeffs: c,
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        }
    }
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }
    fn is_one(&self) -> bool {
        self.coeffs[0].is_one() && self.coeffs.iter().skip(1).all(|c| c.is_zero())
    }
    fn add(&self, other: &Self) -> Self {
        let n = self.extension_degree();
        let mut c = vec![Integer::zero(); n];
        for i in 0..n {
            c[i] = Gfpn::redp(self.coeffs[i].clone() + other.coeffs[i].clone(), &self.p);
        }
        Gfpn {
            coeffs: c,
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        }
    }
    fn sub(&self, other: &Self) -> Self {
        let n = self.extension_degree();
        let mut c = vec![Integer::zero(); n];
        for i in 0..n {
            c[i] = Gfpn::redp(self.coeffs[i].clone() - other.coeffs[i].clone(), &self.p);
        }
        Gfpn {
            coeffs: c,
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        }
    }
    fn mul(&self, other: &Self) -> Self {
        // Schoolbook multiply then reduce modulo the defining polynomial.
        let n = self.extension_degree();
        let mut acc = vec![Integer::zero(); 2 * n - 1];
        for i in 0..n {
            if self.coeffs[i].is_zero() {
                continue;
            }
            for j in 0..n {
                if other.coeffs[j].is_zero() {
                    continue;
                }
                acc[i + j] = acc[i + j].clone() + self.coeffs[i].clone() * other.coeffs[j].clone();
            }
        }
        for v in acc.iter_mut() {
            *v = Gfpn::redp(v.clone(), &self.p);
        }
        // reduce from the top down
        for deg in (n..acc.len()).rev() {
            if acc[deg].is_zero() {
                continue;
            }
            let lead = acc[deg].clone();
            let base = deg - n;
            for i in 0..=n {
                acc[base + i] = Gfpn::redp(
                    acc[base + i].clone() - lead.clone() * self.modulus[i].clone(),
                    &self.p,
                );
            }
        }
        let mut c = vec![Integer::zero(); n];
        for i in 0..n {
            c[i] = acc[i].clone();
        }
        Gfpn {
            coeffs: c,
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        }
    }
    fn neg(&self) -> Self {
        self.zero().sub(self)
    }
    fn invert(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        // Extended Euclidean algorithm in GF(p)[x] between `self` and modulus.
        // Work with FFPoly over PrimeField to reuse polynomial machinery.
        let pf = |v: &Integer| PrimeField::new(v.clone(), self.p.clone()).unwrap();
        let sample = pf(&Integer::zero());

        let a = FFPoly::new(self.coeffs.iter().map(pf).collect(), sample.clone());
        let m = FFPoly::new(self.modulus.iter().map(pf).collect(), sample.clone());

        // Extended Euclid: find s with s*a ≡ 1 (mod m)
        let mut old_r = m.clone();
        let mut r = a.clone();
        let mut old_s = FFPoly::zero(sample.clone());
        let mut s = FFPoly::one(sample.clone());
        while !r.is_zero() {
            let (q, rem) = old_r.div_rem(&r)?;
            old_r = r;
            r = rem;
            let new_s = old_s.sub(&q.mul(&s));
            old_s = s;
            s = new_s;
        }
        // old_r is gcd (should be a nonzero constant), old_s is the cofactor of a
        let g = old_r.make_monic();
        if g.degree() != Some(0) {
            return Err(MathError::InvalidArgument(
                "element not invertible (modulus not irreducible?)".to_string(),
            ));
        }
        // scale old_s by 1/leading(old_r) so that s*a ≡ 1
        let lead_inv = old_r.leading().unwrap().invert()?;
        let inv_poly = old_s.scalar_mul(&lead_inv);
        let coeffs: Vec<Integer> = {
            let n = self.extension_degree();
            let mut out = vec![Integer::zero(); n];
            for (i, c) in inv_poly.coeffs().iter().enumerate() {
                if i < n {
                    out[i] = c.value().clone();
                }
            }
            out
        };
        Ok(Gfpn {
            coeffs,
            p: self.p.clone(),
            modulus: self.modulus.clone(),
        })
    }
    fn characteristic(&self) -> Integer {
        self.p.clone()
    }
    fn order(&self) -> Integer {
        self.p.pow(self.extension_degree() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pf(v: i64, p: i64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
    }

    fn poly(vals: &[i64], p: i64) -> FFPoly<PrimeField> {
        let sample = pf(0, p);
        FFPoly::new(vals.iter().map(|&v| pf(v, p)).collect(), sample)
    }

    #[test]
    fn test_poly_div_rem() {
        // (x^2 + 1) over GF(7) divided by (x + 1): quotient x-1, remainder 2
        let a = poly(&[1, 0, 1], 7);
        let b = poly(&[1, 1], 7);
        let (q, r) = a.div_rem(&b).unwrap();
        // q = x - 1 = x + 6
        assert_eq!(q, poly(&[6, 1], 7));
        // r = 2
        assert_eq!(r, poly(&[2], 7));
        // verify a == q*b + r
        let recon = q.mul(&b).add(&r);
        assert_eq!(recon, a);
    }

    #[test]
    fn test_poly_gcd() {
        // gcd of (x^2-1) and (x^2 + 2x + 1) over GF(7) is (x+1)
        let a = poly(&[6, 0, 1], 7); // x^2 - 1
        let b = poly(&[1, 2, 1], 7); // x^2 + 2x + 1
        let g = a.gcd(&b);
        assert_eq!(g, poly(&[1, 1], 7)); // x + 1
    }

    #[test]
    fn test_pow_mod() {
        // x^7 mod (x^2+1) over GF(7)
        let base = poly(&[0, 1], 7);
        let modu = poly(&[1, 0, 1], 7);
        let r = base.pow_mod(&Integer::from(7), &modu).unwrap();
        // x^2 = -1, so x^7 = x*(x^2)^3 = x*(-1)^3 = -x = 6x
        assert_eq!(r, poly(&[0, 6], 7));
    }

    #[test]
    fn test_gfpn_field_ops() {
        // GF(2^2) with modulus x^2 + x + 1
        let p = Integer::from(2);
        let modulus = vec![Integer::from(1), Integer::from(1), Integer::from(1)];
        let x = Gfpn::new(vec![Integer::from(0), Integer::from(1)], p.clone(), modulus.clone());
        // x^2 = x + 1
        let x2 = x.mul(&x);
        assert_eq!(x2.coeffs(), &[Integer::from(1), Integer::from(1)]);
        // x^3 = 1 (x has order 3 in GF(4)*)
        let x3 = x2.mul(&x);
        assert!(x3.is_one());
        // inverse of x is x^2 = x+1
        let xinv = x.invert().unwrap();
        assert!(x.mul(&xinv).is_one());
    }

    #[test]
    fn test_gfpn_gf9() {
        // GF(3^2) with modulus x^2 + 1 (irreducible over GF(3))
        let p = Integer::from(3);
        let modulus = vec![Integer::from(1), Integer::from(0), Integer::from(1)];
        let x = Gfpn::new(vec![Integer::from(0), Integer::from(1)], p.clone(), modulus.clone());
        // x^2 = -1 = 2
        let x2 = x.mul(&x);
        assert_eq!(x2.coeffs(), &[Integer::from(2), Integer::from(0)]);
        // every nonzero element invertible
        let a = Gfpn::new(vec![Integer::from(1), Integer::from(2)], p.clone(), modulus.clone());
        let ainv = a.invert().unwrap();
        assert!(a.mul(&ainv).is_one());
        assert_eq!(a.order(), Integer::from(9));
    }
}

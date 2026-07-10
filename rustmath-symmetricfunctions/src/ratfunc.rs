//! Exact univariate polynomials and rational functions over an arbitrary field.
//!
//! This is the coefficient machinery for the Hall-Littlewood and Macdonald bases:
//!
//! * `Poly<F>` — dense univariate polynomials over a [`Field`], with Euclidean
//!   division and monic GCD.
//! * `RatFunc<F>` — the fraction field `F(x)` in *canonical form* (numerator and
//!   denominator coprime, denominator monic), so `PartialEq` is exact equality of
//!   rational functions. `RatFunc<F>` itself implements
//!   [`rustmath_core::Field`], so towers are available by nesting:
//!
//!   * `Q(t)   = RatFunc<Rational>` — Hall-Littlewood coefficients,
//!   * `Q(t)(q) = RatFunc<RatFunc<Rational>>` — Macdonald coefficients, with `q`
//!     as the *outer* variable. Substituting the outer variable (e.g. `q = 0` or
//!     `q = t`) is then just polynomial evaluation over the inner field, which is
//!     how the Macdonald specialization maps are implemented.
//!
//! Nothing here is approximate: all arithmetic is exact field arithmetic, and
//! every fraction is reduced by a monic GCD at construction time.

use rustmath_core::{CommutativeRing, Field, IntegralDomain, MathError, Result, Ring};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Poly
// ---------------------------------------------------------------------------

/// A dense univariate polynomial over the field `F`, stored low-degree-first.
///
/// Invariant: no trailing zero coefficients; the zero polynomial has an empty
/// coefficient vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Poly<F: Field> {
    coeffs: Vec<F>,
}

impl<F: Field> Poly<F> {
    fn trim(mut coeffs: Vec<F>) -> Vec<F> {
        while coeffs.last().map(|c| c.is_zero()).unwrap_or(false) {
            coeffs.pop();
        }
        coeffs
    }

    /// Build from coefficients, low degree first (trailing zeros are trimmed).
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Poly {
            coeffs: Self::trim(coeffs),
        }
    }

    /// The zero polynomial.
    pub fn zero() -> Self {
        Poly { coeffs: Vec::new() }
    }

    /// The constant polynomial `1`.
    pub fn one() -> Self {
        Self::constant(F::one())
    }

    /// A constant polynomial.
    pub fn constant(c: F) -> Self {
        Self::from_coeffs(vec![c])
    }

    /// The variable `x` itself.
    pub fn var() -> Self {
        Self::from_coeffs(vec![F::zero(), F::one()])
    }

    /// `x^n`.
    pub fn var_pow(n: usize) -> Self {
        let mut coeffs = vec![F::zero(); n + 1];
        coeffs[n] = F::one();
        Poly { coeffs }
    }

    /// Coefficients, low degree first (empty for the zero polynomial).
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Coefficient of `x^i` (zero if beyond the degree).
    pub fn coeff(&self, i: usize) -> F {
        self.coeffs.get(i).cloned().unwrap_or_else(F::zero)
    }

    /// Degree, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Is this the constant polynomial `1`?
    pub fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }

    fn leading(&self) -> Option<&F> {
        self.coeffs.last()
    }

    /// Sum of two polynomials.
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.coeff(i) + other.coeff(i));
        }
        Self::from_coeffs(out)
    }

    /// Difference of two polynomials.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Negation.
    pub fn neg(&self) -> Self {
        Poly {
            coeffs: self.coeffs.iter().map(|c| -c.clone()).collect(),
        }
    }

    /// Product of two polynomials.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![F::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                out[i + j] = out[i + j].clone() + a.clone() * b.clone();
            }
        }
        Self::from_coeffs(out)
    }

    /// Multiply by a scalar.
    pub fn scale(&self, c: &F) -> Self {
        Self::from_coeffs(self.coeffs.iter().map(|a| a.clone() * c.clone()).collect())
    }

    /// Euclidean division: `self = q * d + r` with `deg r < deg d`.
    /// Errors with `DivisionByZero` if `d` is zero.
    pub fn divrem(&self, d: &Self) -> Result<(Self, Self)> {
        if d.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let dd = d.degree().expect("nonzero");
        let dl = d.leading().expect("nonzero").clone();
        let mut rem = self.clone();
        let mut quo = vec![F::zero(); self.coeffs.len().saturating_sub(dd)];
        while let Some(rd) = rem.degree() {
            if rd < dd {
                break;
            }
            let factor = rem.leading().expect("nonzero").clone() / dl.clone();
            let shift = rd - dd;
            quo[shift] = quo[shift].clone() + factor.clone();
            // rem -= factor * x^shift * d
            let mut sub = vec![F::zero(); shift + dd + 1];
            for (i, c) in d.coeffs.iter().enumerate() {
                sub[shift + i] = factor.clone() * c.clone();
            }
            rem = rem.sub(&Poly::from_coeffs(sub));
        }
        Ok((Self::from_coeffs(quo), rem))
    }

    /// Exact division; errors if the division leaves a remainder.
    pub fn div_exact(&self, d: &Self) -> Result<Self> {
        let (q, r) = self.divrem(d)?;
        if r.is_zero() {
            Ok(q)
        } else {
            Err(MathError::InvalidArgument(
                "polynomial division not exact".to_string(),
            ))
        }
    }

    /// Rescale so the leading coefficient is `1` (identity on the zero polynomial).
    pub fn monic(&self) -> Self {
        match self.leading() {
            None => Self::zero(),
            Some(l) => {
                let inv = l
                    .inverse()
                    .expect("leading coefficient of a nonzero polynomial is nonzero");
                self.scale(&inv)
            }
        }
    }

    /// Monic greatest common divisor (Euclid's algorithm over the field `F`).
    ///
    /// Each remainder is made monic before the next step: multiplying by a
    /// unit does not change the gcd, and over nested fraction fields (e.g.
    /// `F = Q(t)`) this re-reduces every coefficient and prevents the
    /// multiplicative coefficient swell of the naive remainder sequence.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.monic();
        let mut b = other.monic();
        while !b.is_zero() {
            let (_, r) = a.divrem(&b).expect("b nonzero");
            a = b;
            b = r.monic();
        }
        a
    }

    /// Evaluate at `x` (Horner's rule).
    pub fn eval(&self, x: &F) -> F {
        let mut acc = F::zero();
        for c in self.coeffs.iter().rev() {
            acc = acc * x.clone() + c.clone();
        }
        acc
    }

    /// `self^n`.
    pub fn pow(&self, n: usize) -> Self {
        let mut acc = Self::one();
        for _ in 0..n {
            acc = acc.mul(self);
        }
        acc
    }
}

impl<F: Field> fmt::Display for Poly<F> {
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
                0 => write!(f, "({})", c)?,
                1 => write!(f, "({})*x", c)?,
                _ => write!(f, "({})*x^{}", c, i)?,
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RatFunc
// ---------------------------------------------------------------------------

/// A rational function `num/den` over the field `F`, kept in canonical form:
/// `gcd(num, den) = 1` and `den` monic. Canonical form makes derived
/// `PartialEq` exact mathematical equality.
#[derive(Clone, Debug, PartialEq)]
pub struct RatFunc<F: Field> {
    num: Poly<F>,
    den: Poly<F>,
}

impl<F: Field> RatFunc<F> {
    /// Build `num/den` in canonical form. Errors if `den` is zero.
    pub fn new(num: Poly<F>, den: Poly<F>) -> Result<Self> {
        if den.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok(Self::reduced(num, den))
    }

    /// Internal: canonicalize `num/den` assuming `den != 0`.
    fn reduced(num: Poly<F>, den: Poly<F>) -> Self {
        if num.is_zero() {
            return RatFunc {
                num: Poly::zero(),
                den: Poly::one(),
            };
        }
        let g = num.gcd(&den);
        let num = num.div_exact(&g).expect("gcd divides numerator");
        let den = den.div_exact(&g).expect("gcd divides denominator");
        // Make the denominator monic, folding the scalar into the numerator.
        let lc = den
            .leading()
            .expect("denominator nonzero")
            .clone();
        let lc_inv = lc.inverse().expect("nonzero leading coefficient");
        RatFunc {
            num: num.scale(&lc_inv),
            den: den.scale(&lc_inv),
        }
    }

    /// Embed a polynomial.
    pub fn from_poly(p: Poly<F>) -> Self {
        RatFunc {
            num: p,
            den: Poly::one(),
        }
    }

    /// Embed a constant.
    pub fn constant(c: F) -> Self {
        Self::from_poly(Poly::constant(c))
    }

    /// The variable `x` as a rational function.
    pub fn var() -> Self {
        Self::from_poly(Poly::var())
    }

    /// Numerator (canonical form).
    pub fn numerator(&self) -> &Poly<F> {
        &self.num
    }

    /// Denominator (canonical form: monic, coprime to the numerator).
    pub fn denominator(&self) -> &Poly<F> {
        &self.den
    }

    /// If the denominator is `1`, view this as a polynomial.
    pub fn as_polynomial(&self) -> Option<&Poly<F>> {
        if self.den.is_one() {
            Some(&self.num)
        } else {
            None
        }
    }

    /// Evaluate at `x`; errors with `DivisionByZero` when `x` is a pole.
    pub fn eval(&self, x: &F) -> Result<F> {
        let d = self.den.eval(x);
        if d.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok(self.num.eval(x) / d)
    }
}

impl<F: Field> fmt::Display for RatFunc<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.is_one() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "[{}] / [{}]", self.num, self.den)
        }
    }
}

impl<F: Field> Add for RatFunc<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let num = self.num.mul(&other.den).add(&other.num.mul(&self.den));
        let den = self.den.mul(&other.den);
        Self::reduced(num, den)
    }
}

impl<F: Field> Sub for RatFunc<F> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<F: Field> Mul for RatFunc<F> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let num = self.num.mul(&other.num);
        let den = self.den.mul(&other.den);
        Self::reduced(num, den)
    }
}

impl<F: Field> Neg for RatFunc<F> {
    type Output = Self;
    fn neg(self) -> Self {
        RatFunc {
            num: self.num.neg(),
            den: self.den,
        }
    }
}

impl<F: Field> Div for RatFunc<F> {
    type Output = Self;
    /// Panics on division by zero (mirrors `Rational`'s `Div`); use
    /// [`Field::divide`] for the checked version.
    fn div(self, other: Self) -> Self {
        if other.num.is_zero() {
            panic!("RatFunc: division by zero");
        }
        let num = self.num.mul(&other.den);
        let den = self.den.mul(&other.num);
        Self::reduced(num, den)
    }
}

impl<F: Field> Ring for RatFunc<F> {
    fn zero() -> Self {
        Self::from_poly(Poly::zero())
    }

    fn one() -> Self {
        Self::from_poly(Poly::one())
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    fn is_one(&self) -> bool {
        self.num.is_one() && self.den.is_one()
    }
}

impl<F: Field> CommutativeRing for RatFunc<F> {}
impl<F: Field> IntegralDomain for RatFunc<F> {}

impl<F: Field> Field for RatFunc<F> {
    fn inverse(&self) -> Result<Self> {
        if self.num.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Self::new(self.den.clone(), self.num.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn rp(coeffs: &[i64]) -> Poly<Rational> {
        Poly::from_coeffs(coeffs.iter().map(|&c| Rational::from(c)).collect())
    }

    #[test]
    fn test_poly_arith_and_divrem() {
        // (x^2 - 1) = (x - 1)(x + 1)
        let a = rp(&[-1, 0, 1]);
        let b = rp(&[-1, 1]);
        let c = rp(&[1, 1]);
        assert_eq!(b.mul(&c), a);
        let (q, r) = a.divrem(&b).unwrap();
        assert_eq!(q, c);
        assert!(r.is_zero());
        // gcd(x^2 - 1, x^2 + 2x + 1) = x + 1 (monic)
        let d = rp(&[1, 2, 1]);
        assert_eq!(a.gcd(&d), c);
        assert!(a.divrem(&Poly::zero()).is_err());
    }

    #[test]
    fn test_poly_eval() {
        // p(x) = 2x^2 - 3x + 1 at x = 3 -> 10
        let p = rp(&[1, -3, 2]);
        assert_eq!(p.eval(&Rational::from(3)), Rational::from(10));
    }

    #[test]
    fn test_ratfunc_canonical_form() {
        // (x^2 - 1)/(2x - 2) reduces to (x + 1)/2 with monic denominator:
        // num = x/2 + 1/2, den = 1.
        let r = RatFunc::new(rp(&[-1, 0, 1]), rp(&[-2, 2])).unwrap();
        let expected = RatFunc::new(rp(&[1, 1]), rp(&[2])).unwrap();
        assert_eq!(r, expected);
        assert!(r.as_polynomial().is_some());
        // Different unreduced representatives compare equal.
        let a = RatFunc::new(rp(&[0, 1]), rp(&[1, 1])).unwrap(); // x/(x+1)
        let b = RatFunc::new(rp(&[0, 2, 2]), rp(&[2, 4, 2])).unwrap(); // 2x(x+1)/2(x+1)^2
        assert_eq!(a, b);
    }

    #[test]
    fn test_ratfunc_field_ops() {
        type F = RatFunc<Rational>;
        let x = F::var();
        let one = <F as Ring>::one();
        // x/(x+1) + 1/(x+1) = 1
        let a = RatFunc::new(rp(&[0, 1]), rp(&[1, 1])).unwrap();
        let b = RatFunc::new(rp(&[1]), rp(&[1, 1])).unwrap();
        assert_eq!(a.clone() + b, one.clone());
        // x * x^{-1} = 1
        assert_eq!(x.clone() * x.inverse().unwrap(), one);
        assert!(<F as Ring>::zero().inverse().is_err());
    }

    #[test]
    fn test_ratfunc_eval_and_poles() {
        // f = 1/(1 - t): value at t=2 is -1, pole at t=1.
        let f = RatFunc::new(rp(&[1]), rp(&[1, -1])).unwrap();
        assert_eq!(f.eval(&Rational::from(2)).unwrap(), Rational::from(-1));
        assert!(f.eval(&Rational::from(1)).is_err());
    }

    #[test]
    fn test_nested_tower_q_over_qt() {
        // Q(t)(q): check (q*t - q + t - 1)/(q*t - 1) at q=0 gives 1 - t ... wait:
        // at q=0: (t - 1)/(-1) = 1 - t. This is the Macdonald P_(2) coefficient
        // specializing to the Hall-Littlewood one.
        type T = RatFunc<Rational>;
        type QT = RatFunc<T>;
        let t = |c: &[i64]| RatFunc::from_poly(rp(c)); // poly in t as T
        let num = Poly::from_coeffs(vec![t(&[-1, 1]), t(&[-1, 1])]); // (t-1) + (t-1) q
        let den = Poly::from_coeffs(vec![t(&[-1]), t(&[0, 1])]); // -1 + t q
        let f = QT::new(num, den).unwrap();
        let at_q0 = f.eval(&<T as Ring>::zero()).unwrap();
        assert_eq!(at_q0, t(&[1, -1])); // 1 - t
        // at q = t: (t^2 - t + t - 1)/(t^2 - 1) = 1.
        let at_qt = f.eval(&T::var()).unwrap();
        assert!(at_qt.is_one());
    }
}

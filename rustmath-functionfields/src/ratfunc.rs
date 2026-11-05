//! The rational function field `ℚ(t)`.
//!
//! An element of `ℚ(t)` is a fraction `p(t) / q(t)` of two univariate
//! polynomials over `ℚ`, kept in lowest terms with a monic denominator. This
//! type is a [`Field`] (and therefore a [`Ring`]/[`EuclideanDomain`]), so it can
//! serve as the coefficient ring of `UnivariatePolynomial<RationalFunction>`,
//! i.e. of `ℚ(t)[x]` — the ambient ring of all function-field constructions.
//!
//! Polynomial arithmetic over `ℚ` is reused verbatim from
//! [`rustmath_polynomials::UnivariatePolynomial`]; nothing is reimplemented here
//! beyond the fraction bookkeeping.

use rustmath_core::{
    CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, NumericConversion, Result,
    Ring,
};
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A univariate polynomial over `ℚ` in the transcendental `t`.
pub type QtPoly = UnivariatePolynomial<Rational>;

/// An element of the rational function field `ℚ(t) = Frac(ℚ[t])`.
///
/// Stored as `num / den` in canonical form: `gcd(num, den) = 1`, `den` monic,
/// and the zero element represented as `0 / 1`.
#[derive(Clone)]
pub struct RationalFunction {
    num: QtPoly,
    den: QtPoly,
}

/// Monic gcd of two `ℚ[t]` polynomials (the Euclidean gcd, normalised so the
/// leading coefficient is `1`). Returns the constant `1` for two zero inputs.
fn monic_gcd(a: &QtPoly, b: &QtPoly) -> QtPoly {
    if a.is_zero() && b.is_zero() {
        return QtPoly::one();
    }
    let g = a.gcd(b);
    if g.is_zero() {
        QtPoly::one()
    } else {
        g.make_monic()
    }
}

impl RationalFunction {
    /// Build `num / den` in canonical form. Errors if `den` is the zero polynomial.
    pub fn new(num: QtPoly, den: QtPoly) -> Result<Self> {
        if den.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let mut rf = RationalFunction { num, den };
        rf.normalize();
        Ok(rf)
    }

    /// Reduce to lowest terms with a monic denominator.
    fn normalize(&mut self) {
        if self.num.is_zero() {
            self.den = QtPoly::one();
            return;
        }
        let g = monic_gcd(&self.num, &self.den);
        let (n, _) = self.num.div_rem(&g).unwrap();
        let (d, _) = self.den.div_rem(&g).unwrap();
        // Make the denominator monic, folding its leading coefficient into the
        // numerator so the value is unchanged.
        let lc = d.leading_coefficient().cloned().unwrap_or_else(Rational::one);
        let lc_inv = lc.inverse().unwrap();
        self.num = n.scalar_mul(&lc_inv);
        self.den = d.scalar_mul(&lc_inv);
    }

    /// The rational function `t` (the transcendental generator).
    pub fn t() -> Self {
        RationalFunction {
            num: QtPoly::var(),
            den: QtPoly::one(),
        }
    }

    /// Embed a constant `c ∈ ℚ`.
    pub fn from_rational(c: Rational) -> Self {
        RationalFunction {
            num: QtPoly::constant(c),
            den: QtPoly::one(),
        }
    }

    /// Embed an integer constant.
    pub fn from_i64(n: i64) -> Self {
        Self::from_rational(Rational::from_i64(n))
    }

    /// Embed a polynomial `p(t) ∈ ℚ[t]` as `p(t) / 1`.
    pub fn from_poly(p: QtPoly) -> Self {
        RationalFunction {
            num: p,
            den: QtPoly::one(),
        }
    }

    /// The numerator (in lowest terms, denominator monic).
    pub fn numerator(&self) -> &QtPoly {
        &self.num
    }

    /// The (monic) denominator (in lowest terms).
    pub fn denominator(&self) -> &QtPoly {
        &self.den
    }

    /// `true` iff this element lies in `ℚ[t]` (denominator is a constant).
    pub fn is_polynomial(&self) -> bool {
        self.den.degree() == Some(0)
    }

    /// Evaluate at `t = a ∈ ℚ`. Returns `None` if the denominator vanishes at `a`
    /// (a pole of this rational function).
    pub fn evaluate(&self, a: &Rational) -> Option<Rational> {
        let d = self.den.evaluate(a);
        if d.is_zero() {
            return None;
        }
        let n = self.num.evaluate(a);
        Some(n / d)
    }
}

impl PartialEq for RationalFunction {
    fn eq(&self, other: &Self) -> bool {
        // Both sides are canonical (lowest terms, monic denominator), so a
        // component-wise comparison is exact.
        self.num == other.num && self.den == other.den
    }
}

impl Eq for RationalFunction {}

impl fmt::Display for RationalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_polynomial() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({}) / ({})", self.num, self.den)
        }
    }
}

impl fmt::Debug for RationalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RatFunc({:?} / {:?})", self.num, self.den)
    }
}

impl Add for RationalFunction {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let num = self.num.clone() * other.den.clone() + other.num.clone() * self.den.clone();
        let den = self.den * other.den;
        RationalFunction::new(num, den).unwrap()
    }
}

impl Sub for RationalFunction {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let num = self.num.clone() * other.den.clone() - other.num.clone() * self.den.clone();
        let den = self.den * other.den;
        RationalFunction::new(num, den).unwrap()
    }
}

impl Mul for RationalFunction {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let num = self.num * other.num;
        let den = self.den * other.den;
        RationalFunction::new(num, den).unwrap()
    }
}

impl Neg for RationalFunction {
    type Output = Self;
    fn neg(self) -> Self {
        RationalFunction {
            num: -self.num,
            den: self.den,
        }
    }
}

impl Div for RationalFunction {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        if other.is_zero() {
            panic!("division by zero in ℚ(t)");
        }
        // (a/b) / (c/d) = (a*d) / (b*c)
        let num = self.num * other.den;
        let den = self.den * other.num;
        RationalFunction::new(num, den).unwrap()
    }
}

impl Ring for RationalFunction {
    fn zero() -> Self {
        RationalFunction {
            num: QtPoly::zero(),
            den: QtPoly::one(),
        }
    }

    fn one() -> Self {
        RationalFunction {
            num: QtPoly::one(),
            den: QtPoly::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    fn is_one(&self) -> bool {
        self.den.degree() == Some(0)
            && self.den.coefficients()[0].is_one()
            && self.num.degree() == Some(0)
            && self.num.coefficients()[0].is_one()
    }
}

impl CommutativeRing for RationalFunction {}
impl IntegralDomain for RationalFunction {}

impl EuclideanDomain for RationalFunction {
    fn norm(&self) -> u64 {
        // ℚ(t) is a field: a trivial Euclidean norm (0 for zero, 1 otherwise)
        // makes `UnivariatePolynomial<RationalFunction>` a Euclidean domain so
        // its `div_rem`/`gcd` are usable for factorization recombination.
        if self.is_zero() {
            0
        } else {
            1
        }
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok((self.clone() / other.clone(), RationalFunction::zero()))
    }
}

impl Field for RationalFunction {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        RationalFunction::new(self.den.clone(), self.num.clone())
    }
}

impl NumericConversion for RationalFunction {
    fn from_i64(n: i64) -> Self {
        RationalFunction::from_i64(n)
    }
    fn from_u64(n: u64) -> Self {
        RationalFunction::from_i64(n as i64)
    }
    fn to_i64(&self) -> Option<i64> {
        if self.is_polynomial() {
            self.num.coefficients().first().and_then(|c| {
                if self.num.degree() == Some(0) {
                    c.to_i64()
                } else {
                    None
                }
            })
        } else {
            None
        }
    }
    fn to_u64(&self) -> Option<u64> {
        self.to_i64().and_then(|n| if n >= 0 { Some(n as u64) } else { None })
    }
    fn to_usize(&self) -> Option<usize> {
        self.to_i64().and_then(|n| if n >= 0 { Some(n as usize) } else { None })
    }
    fn to_f64(&self) -> Option<f64> {
        if self.is_polynomial() && self.num.degree() == Some(0) {
            self.num.coefficients()[0].to_f64()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn t_plus_one_over_t() {
        // (t+1)/t  +  1/t  = (t+2)/t
        let t = RationalFunction::t();
        let one_over_t =
            RationalFunction::new(QtPoly::one(), QtPoly::var()).unwrap();
        let lhs = (t.clone() + RationalFunction::one()) * one_over_t.clone();
        let sum = lhs + one_over_t;
        // expected (t+2)/t
        let expected = RationalFunction::new(
            UnivariatePolynomial::new(vec![q(2), q(1)]),
            QtPoly::var(),
        )
        .unwrap();
        assert_eq!(sum, expected);
    }

    #[test]
    fn reduces_to_lowest_terms() {
        // (t^2 - 1)/(t - 1) = t + 1
        let num = UnivariatePolynomial::new(vec![q(-1), q(0), q(1)]);
        let den = UnivariatePolynomial::new(vec![q(-1), q(1)]);
        let rf = RationalFunction::new(num, den).unwrap();
        assert!(rf.is_polynomial());
        let expected = RationalFunction::from_poly(UnivariatePolynomial::new(vec![q(1), q(1)]));
        assert_eq!(rf, expected);
    }

    #[test]
    fn inverse_and_division() {
        let t = RationalFunction::t();
        let inv = t.inverse().unwrap();
        assert_eq!(t.clone() * inv.clone(), RationalFunction::one());
        assert_eq!(RationalFunction::one() / t.clone(), inv);
    }

    #[test]
    fn evaluate_with_pole() {
        // 1/(t-2): pole at t=2, value 1 at t=3
        let den = UnivariatePolynomial::new(vec![q(-2), q(1)]);
        let rf = RationalFunction::new(QtPoly::one(), den).unwrap();
        assert_eq!(rf.evaluate(&q(2)), None);
        assert_eq!(rf.evaluate(&q(3)), Some(q(1)));
    }

    #[test]
    fn field_axioms_small() {
        let t = RationalFunction::t();
        let a = t.clone() + RationalFunction::from_i64(3);
        assert_eq!(a.clone() - a.clone(), RationalFunction::zero());
        assert_eq!(a.clone() * RationalFunction::one(), a);
    }
}

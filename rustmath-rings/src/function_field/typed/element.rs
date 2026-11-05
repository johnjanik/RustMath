//! Elements of the rational function field K(x), in canonical form.
//!
//! A [`RationalFunction`] is a pair (num, den) of `UnivariatePolynomial<K>`
//! with the invariants:
//!
//! - `den` is nonzero and **monic**;
//! - `gcd(num, den) = 1`;
//! - zero is represented as `0/1`.
//!
//! Because the form is canonical, equality is plain structural equality and
//! `PartialEq` is exact.

use rustmath_core::{CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, Result, Ring};
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Exact division of polynomials over a field; panics if not exact.
///
/// Used only where divisibility is a mathematical invariant (dividing by a
/// gcd we just computed).
pub(crate) fn exact_div<K: Field + EuclideanDomain>(
    a: &UnivariatePolynomial<K>,
    b: &UnivariatePolynomial<K>,
) -> UnivariatePolynomial<K> {
    let (q, r) = a
        .div_rem(b)
        .expect("exact_div: division by zero polynomial");
    assert!(
        r.is_zero(),
        "exact_div: division was not exact (broken invariant)"
    );
    q
}

/// An element of K(x) = Frac(K[x]) in canonical form (gcd 1, monic den).
#[derive(Clone, Debug, PartialEq)]
pub struct RationalFunction<K: Field + EuclideanDomain> {
    num: UnivariatePolynomial<K>,
    den: UnivariatePolynomial<K>,
}

impl<K: Field + EuclideanDomain> RationalFunction<K> {
    /// Create `num/den`, normalizing to canonical form.
    ///
    /// Returns `Err(MathError::DivisionByZero)` if `den` is zero.
    pub fn new(num: UnivariatePolynomial<K>, den: UnivariatePolynomial<K>) -> Result<Self> {
        if den.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        if num.is_zero() {
            return Ok(RationalFunction {
                num: UnivariatePolynomial::zero(),
                den: UnivariatePolynomial::one(),
            });
        }
        // Cancel the gcd (Euclidean gcd over the field K, arbitrary scaling).
        let g = num.gcd(&den);
        let mut num = exact_div(&num, &g);
        let mut den = exact_div(&den, &g);
        // Make the denominator monic, dividing the numerator by the same unit.
        let lc = den
            .leading_coefficient()
            .expect("nonzero denominator has a leading coefficient")
            .clone();
        if !lc.is_one() {
            let lc_inv = lc.inverse()?;
            num = num.scalar_mul(&lc_inv);
            den = den.scalar_mul(&lc_inv);
        }
        Ok(RationalFunction { num, den })
    }

    /// Embed a polynomial into K(x).
    pub fn from_polynomial(p: UnivariatePolynomial<K>) -> Self {
        RationalFunction {
            num: p,
            den: UnivariatePolynomial::one(),
        }
    }

    /// Embed a constant into K(x).
    pub fn constant(c: K) -> Self {
        RationalFunction::from_polynomial(UnivariatePolynomial::constant(c))
    }

    /// The generator x of K(x).
    pub fn gen() -> Self {
        RationalFunction::from_polynomial(UnivariatePolynomial::new(vec![K::zero(), K::one()]))
    }

    /// The (canonical) numerator; coprime to the denominator.
    pub fn numerator(&self) -> &UnivariatePolynomial<K> {
        &self.num
    }

    /// The (canonical) denominator; monic and coprime to the numerator.
    pub fn denominator(&self) -> &UnivariatePolynomial<K> {
        &self.den
    }

    /// Is this element a polynomial (denominator 1)?
    pub fn is_polynomial(&self) -> bool {
        self.den.is_one()
    }

    /// Evaluate at a point of K. `Err` if the point is a pole.
    pub fn evaluate(&self, point: &K) -> Result<K> {
        let d = self.den.evaluate(point);
        if d.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok(self.num.evaluate(point) * d.inverse()?)
    }

    /// Formal derivative d/dx, by the quotient rule.
    pub fn derivative(&self) -> Self {
        // (n/d)' = (n' d - n d') / d^2
        let n = &self.num;
        let d = &self.den;
        let num = n.derivative() * d.clone() - n.clone() * d.derivative();
        let den = d.clone() * d.clone();
        RationalFunction::new(num, den).expect("denominator square is nonzero")
    }
}

impl<K: Field + EuclideanDomain> fmt::Display for RationalFunction<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.is_one() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({})/({})", self.num, self.den)
        }
    }
}

impl<K: Field + EuclideanDomain> Add for RationalFunction<K> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let num = self.num * rhs.den.clone() + rhs.num * self.den.clone();
        let den = self.den * rhs.den;
        RationalFunction::new(num, den).expect("product of nonzero denominators is nonzero")
    }
}

impl<K: Field + EuclideanDomain> Sub for RationalFunction<K> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<K: Field + EuclideanDomain> Mul for RationalFunction<K> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let num = self.num * rhs.num;
        let den = self.den * rhs.den;
        RationalFunction::new(num, den).expect("product of nonzero denominators is nonzero")
    }
}

impl<K: Field + EuclideanDomain> Neg for RationalFunction<K> {
    type Output = Self;
    fn neg(self) -> Self {
        RationalFunction {
            num: self.num.negate(),
            den: self.den,
        }
    }
}

impl<K: Field + EuclideanDomain> Div for RationalFunction<K> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs.num.is_zero() {
            panic!("division by zero in K(x)");
        }
        let num = self.num * rhs.den;
        let den = self.den * rhs.num;
        RationalFunction::new(num, den).expect("nonzero denominator")
    }
}

impl<K: Field + EuclideanDomain> Ring for RationalFunction<K> {
    fn zero() -> Self {
        RationalFunction {
            num: UnivariatePolynomial::zero(),
            den: UnivariatePolynomial::one(),
        }
    }

    fn one() -> Self {
        RationalFunction {
            num: UnivariatePolynomial::one(),
            den: UnivariatePolynomial::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    fn is_one(&self) -> bool {
        self.num.is_one() && self.den.is_one()
    }
}

impl<K: Field + EuclideanDomain> CommutativeRing for RationalFunction<K> {}
impl<K: Field + EuclideanDomain> IntegralDomain for RationalFunction<K> {}

impl<K: Field + EuclideanDomain> EuclideanDomain for RationalFunction<K> {
    fn norm(&self) -> u64 {
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
        Ok((self.clone() * other.inverse()?, Self::zero()))
    }
}

impl<K: Field + EuclideanDomain> Field for RationalFunction<K> {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        // num and den are coprime; only re-normalize the leading unit.
        RationalFunction::new(self.den.clone(), self.num.clone())
    }
}

/// The parent object: the rational function field K(x).
///
/// K(x) is genus 0; its places and divisors live in
/// [`super::place`] and [`super::divisor`].
#[derive(Clone, Debug, Default)]
pub struct RationalFunctionField<K: Field + EuclideanDomain> {
    _marker: PhantomData<K>,
}

impl<K: Field + EuclideanDomain> RationalFunctionField<K> {
    /// Create the field K(x).
    pub fn new() -> Self {
        RationalFunctionField {
            _marker: PhantomData,
        }
    }

    /// The zero element.
    pub fn zero(&self) -> RationalFunction<K> {
        RationalFunction::zero()
    }

    /// The one element.
    pub fn one(&self) -> RationalFunction<K> {
        RationalFunction::one()
    }

    /// The generator x.
    pub fn gen(&self) -> RationalFunction<K> {
        RationalFunction::gen()
    }

    /// Build `num/den` (normalized). `Err` on zero denominator.
    pub fn element(
        &self,
        num: UnivariatePolynomial<K>,
        den: UnivariatePolynomial<K>,
    ) -> Result<RationalFunction<K>> {
        RationalFunction::new(num, den)
    }

    /// Embed a polynomial.
    pub fn from_polynomial(&self, p: UnivariatePolynomial<K>) -> RationalFunction<K> {
        RationalFunction::from_polynomial(p)
    }

    /// Embed a constant of K.
    pub fn constant(&self, c: K) -> RationalFunction<K> {
        RationalFunction::constant(c)
    }

    /// The genus of K(x) (always 0: the projective line).
    pub fn genus(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::gfp::GFp;
    use super::*;
    use rustmath_rationals::Rational;

    fn qpoly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| Rational::from_i64(c)).collect())
    }

    fn fpoly<const P: u64>(coeffs: &[i64]) -> UnivariatePolynomial<GFp<P>> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| GFp::<P>::new(c)).collect())
    }

    fn qrf(num: &[i64], den: &[i64]) -> RationalFunction<Rational> {
        RationalFunction::new(qpoly(num), qpoly(den)).unwrap()
    }

    #[test]
    fn test_normalization_q() {
        // (2x+2)/(4x): gcd cancelled, monic denominator =>
        // ((1/2)x + 1/2)/x. Cross-check: (2x+2)*x == 4x*((1/2)x+1/2). Verified
        // by hand: both are 2x^2 + 2x.
        let f = qrf(&[2, 2], &[0, 4]);
        let half = Rational::new(1, 2).unwrap();
        assert_eq!(
            f.numerator(),
            &UnivariatePolynomial::new(vec![half.clone(), half])
        );
        assert_eq!(f.denominator(), &qpoly(&[0, 1]));
        assert!(f.denominator().is_monic());
        // gcd(num, den) is a unit.
        assert_eq!(f.numerator().gcd(f.denominator()).degree(), Some(0));
    }

    #[test]
    fn test_normalization_gf5() {
        // Over GF(5): (2x+2)/(4x) = (3x+3)/x. sympy/hand-verified:
        // (2x+2)*x = 2x^2+2x = (4x)(3x+3) mod 5.
        let f = RationalFunction::new(fpoly::<5>(&[2, 2]), fpoly::<5>(&[0, 4])).unwrap();
        assert_eq!(f.numerator(), &fpoly::<5>(&[3, 3]));
        assert_eq!(f.denominator(), &fpoly::<5>(&[0, 1]));
    }

    #[test]
    fn test_zero_denominator_is_err() {
        assert!(RationalFunction::new(qpoly(&[1]), UnivariatePolynomial::zero()).is_err());
    }

    #[test]
    fn test_add_to_one() {
        // x/(x+1) + 1/(x+1) = 1.
        let a = qrf(&[0, 1], &[1, 1]);
        let b = qrf(&[1], &[1, 1]);
        assert!((a + b).is_one());
    }

    #[test]
    fn test_field_axioms_samples_q() {
        let f = qrf(&[-1, 0, 1], &[0, 2, 0, 1]); // (x^2-1)/(x^3+2x)
        let g = qrf(&[3, 1], &[-2, 0, 1]); // (x+3)/(x^2-2)
        // a - a = 0, a * a^{-1} = 1, (f*g)/g = f, distributivity sample.
        assert!((f.clone() - f.clone()).is_zero());
        assert!((f.clone() * f.inverse().unwrap()).is_one());
        assert_eq!(f.clone() * g.clone() / g.clone(), f);
        let h = qrf(&[1, 1], &[1]);
        assert_eq!(
            f.clone() * (g.clone() + h.clone()),
            f.clone() * g.clone() + f.clone() * h.clone()
        );
    }

    #[test]
    fn test_field_axioms_samples_gf5() {
        let f = RationalFunction::new(fpoly::<5>(&[1, 0, 1]), fpoly::<5>(&[1, 4, 0, 0, 0, 1]))
            .unwrap();
        let g = RationalFunction::new(fpoly::<5>(&[2, 3]), fpoly::<5>(&[0, 1])).unwrap();
        assert!((f.clone() - f.clone()).is_zero());
        assert!((f.clone() * f.inverse().unwrap()).is_one());
        assert_eq!(f.clone() * g.clone() / g.clone(), f);
        // (x+2)(x+3) = x^2 + 1 over GF(5) (sympy-verified factorization).
        let a = RationalFunction::from_polynomial(fpoly::<5>(&[2, 1]));
        let b = RationalFunction::from_polynomial(fpoly::<5>(&[3, 1]));
        assert_eq!(a * b, RationalFunction::from_polynomial(fpoly::<5>(&[1, 0, 1])));
    }

    #[test]
    fn test_inverse_of_zero_is_err() {
        assert!(RationalFunction::<Rational>::zero().inverse().is_err());
    }

    #[test]
    fn test_evaluate() {
        // f = (x^2+1)/(x+1): f(2) = 5/3; pole at x = -1.
        let f = qrf(&[1, 0, 1], &[1, 1]);
        assert_eq!(
            f.evaluate(&Rational::from_i64(2)).unwrap(),
            Rational::new(5, 3).unwrap()
        );
        assert!(f.evaluate(&Rational::from_i64(-1)).is_err());
    }

    #[test]
    fn test_derivative() {
        // (1/x)' = -1/x^2 and (x^2)' = 2x (calculus, exact).
        let inv_x = qrf(&[1], &[0, 1]);
        assert_eq!(inv_x.derivative(), qrf(&[-1], &[0, 0, 1]));
        let x2 = qrf(&[0, 0, 1], &[1]);
        assert_eq!(x2.derivative(), qrf(&[0, 2], &[1]));
    }

    #[test]
    fn test_parent_field_constructors() {
        let kx = RationalFunctionField::<Rational>::new();
        assert!(kx.zero().is_zero());
        assert!(kx.one().is_one());
        assert_eq!(kx.genus(), 0);
        let x = kx.gen();
        assert_eq!(x, RationalFunction::from_polynomial(qpoly(&[0, 1])));
        assert_eq!(
            kx.element(qpoly(&[0, 2]), qpoly(&[2])).unwrap(),
            x
        );
        assert_eq!(kx.constant(Rational::from_i64(3)), qrf(&[3], &[1]));
    }

    #[test]
    fn test_canonical_form_equality() {
        // Same element built two ways compares equal structurally.
        assert_eq!(qrf(&[2, 2], &[0, 4]), qrf(&[1, 1], &[0, 2]));
        assert_ne!(qrf(&[1, 1], &[0, 2]), qrf(&[1], &[0, 1]));
    }
}

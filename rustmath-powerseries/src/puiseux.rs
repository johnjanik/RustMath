//! Puiseux series `R⟨⟨x⟩⟩` — rational exponents.
//!
//! MAGMA source: Handbook Chapter 49 "Power, Laurent and Puiseux Series",
//! category `RngSerPuis`/`RngSerPuisElt`.  Covers §49.1.2 (a Puiseux series is
//! internally a Laurent series in `y = x^{1/e}` for a minimal exponent
//! denominator `e`), §49.4.5–49.4.6 (Valuation / AbsolutePrecision /
//! ExponentDenominator, which may be non-integral rationals), and the ring/field
//! structure (`R⟨⟨x⟩⟩` is a field when `R` is — indeed it is the algebraic
//! closure of `R((x))` when `R` is algebraically closed of characteristic 0).
//!
//! A [`PuiseuxSeries`] wraps an internal [`LaurentSeries`] in the variable
//! `y = x^{1/e}` together with the ramification index `e ≥ 1`; the exponent of a
//! term stored at `y`-exponent `w` is the rational `w/e`.  Binary operations
//! bring the two operands to a common denominator `E = lcm(e₁, e₂)` by rescaling
//! (inserting zero coefficients on the finer grid), operate on the Laurent
//! parts, and then minimise the denominator again.

use crate::laurent::LaurentSeries;
use crate::precision::{Precision, DEFAULT_PRECISION};
use crate::series::PowerSeries;
use rustmath_core::{CommutativeRing, Field, IntegralDomain, Result, Ring};
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn lcm_u64(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd_u64(a, b) * b
    }
}

/// A Puiseux series: a Laurent series in `y = x^{1/e}`.
#[derive(Clone, Debug)]
pub struct PuiseuxSeries<R: Ring> {
    /// Exponent denominator (ramification index), `≥ 1`.
    e: u64,
    /// The underlying Laurent series in `y = x^{1/e}`.
    inner: LaurentSeries<R>,
}

impl<R: Ring> PuiseuxSeries<R> {
    /// Build from an explicit denominator `e` and a Laurent series in `x^{1/e}`.
    pub fn from_laurent(e: u64, inner: LaurentSeries<R>) -> Self {
        PuiseuxSeries {
            e: e.max(1),
            inner,
        }
        .reduce()
    }

    /// The zero series to relative precision `prec`.
    pub fn zero(prec: usize) -> Self {
        PuiseuxSeries {
            e: 1,
            inner: LaurentSeries::zero(prec),
        }
    }

    /// The constant `1` to relative precision `prec`.
    pub fn one(prec: usize) -> Self {
        PuiseuxSeries {
            e: 1,
            inner: LaurentSeries::one(prec),
        }
    }

    /// Embed a Laurent series (denominator 1).
    pub fn from_laurent_series(inner: LaurentSeries<R>) -> Self {
        PuiseuxSeries { e: 1, inner }.reduce()
    }

    /// Embed a power series (denominator 1, valuation `≥ 0`).
    pub fn from_power_series(f: &PowerSeries<R>) -> Self {
        PuiseuxSeries {
            e: 1,
            inner: LaurentSeries::from_power_series(f),
        }
    }

    /// Build from an exponent denominator `e` and an exact coefficient window
    /// (starting at `y`-exponent `val`, `y = x^{1/e}`) under a [`Precision`]
    /// regime; the capped vs exact-polynomial semantics of the window are
    /// those of [`LaurentSeries::with_precision`].
    pub fn with_precision(e: u64, val: i64, coeffs: Vec<R>, prec: Precision) -> Self {
        Self::from_laurent(e, LaurentSeries::with_precision(val, coeffs, prec))
    }

    /// The monomial `coeff · x^{num/den}` under a [`Precision`] regime (the
    /// window carries the regime's default term count).
    pub fn monomial_with(num: i64, den: u64, coeff: R, prec: Precision) -> Self {
        Self::monomial(num, den, coeff, prec.default_terms())
    }

    /// The monomial `coeff · x^{num/den}` to relative precision `prec`.
    pub fn monomial(num: i64, den: u64, coeff: R, prec: usize) -> Self {
        let den = den.max(1);
        PuiseuxSeries {
            e: den,
            inner: LaurentSeries::monomial(num, coeff, prec),
        }
        .reduce()
    }

    /// The exponent denominator `e` (`ExponentDenominator`).
    pub fn exponent_denominator(&self) -> u64 {
        self.e
    }

    /// The valuation as a rational `w/e` (may be non-integral).  Returns `None`
    /// for a series with no known non-zero term.
    pub fn valuation(&self) -> Option<Rational> {
        self.inner.leading_coefficient()?;
        let w = self.inner.valuation();
        Some(Rational::new(w, self.e as i64).unwrap())
    }

    /// The absolute precision as a rational `p/e`.
    pub fn absolute_precision(&self) -> Rational {
        Rational::new(self.inner.absolute_precision(), self.e as i64).unwrap()
    }

    /// The coefficient of `x^{num/den}` (zero if that exponent is not on this
    /// series' grid).
    pub fn coefficient(&self, num: i64, den: u64) -> R {
        let den = den.max(1);
        // exponent num/den == w/e  <=>  w = num * e / den, must be integral
        let scaled = num as i128 * self.e as i128;
        if scaled % den as i128 != 0 {
            return R::zero();
        }
        self.inner.coefficient((scaled / den as i128) as i64)
    }

    /// The underlying Laurent series in `x^{1/e}`.
    pub fn laurent_part(&self) -> &LaurentSeries<R> {
        &self.inner
    }

    /// Whether every known coefficient is zero.
    pub fn is_weakly_zero(&self) -> bool {
        self.inner.is_weakly_zero()
    }

    /// Rescale the internal Laurent series onto a finer grid `y' = x^{1/(e·f)}`
    /// (used to bring two operands to a common denominator).
    fn rescale(&self, f: u64) -> LaurentSeries<R> {
        if f <= 1 {
            return self.inner.clone();
        }
        let base = self.inner.lowest_exponent();
        let len = self.inner.relative_precision();
        let f = f as usize;
        // Interleave each coefficient with (f-1) zeros on the finer grid.
        let mut new_coeffs = vec![R::zero(); len * f];
        for i in 0..len {
            new_coeffs[i * f] = self.inner.coefficient(base + i as i64);
        }
        LaurentSeries::new(base * f as i64, new_coeffs)
    }

    /// Minimise the exponent denominator (`e` and the internal grid) losslessly.
    fn reduce(mut self) -> Self {
        if self.e <= 1 {
            return self;
        }
        let len = self.inner.relative_precision();
        if len == 0 {
            self.e = 1;
            return self;
        }
        let base = self.inner.lowest_exponent();
        let mut c = self.e;
        c = gcd_u64(c, base.unsigned_abs());
        c = gcd_u64(c, len as u64);
        for i in 0..len {
            if !self.inner.coefficient(base + i as i64).is_zero() {
                c = gcd_u64(c, i as u64);
            }
        }
        if c <= 1 {
            return self;
        }
        let cf = c as usize;
        let new_len = len / cf;
        let new_coeffs: Vec<R> = (0..new_len)
            .map(|j| self.inner.coefficient(base + (j * cf) as i64))
            .collect();
        self.inner = LaurentSeries::new(base / c as i64, new_coeffs);
        self.e /= c;
        self
    }

    /// Bring `self` and `other` to a common denominator, returning the shared
    /// `E` and the two rescaled Laurent parts.
    fn align(&self, other: &Self) -> (u64, LaurentSeries<R>, LaurentSeries<R>) {
        let big = lcm_u64(self.e, other.e);
        (big, self.rescale(big / self.e), other.rescale(big / other.e))
    }
}

impl<R: Ring> PartialEq for PuiseuxSeries<R> {
    fn eq(&self, other: &Self) -> bool {
        let (_, a, b) = self.align(other);
        a == b
    }
}

impl<R: Ring> Add for PuiseuxSeries<R> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let (big, a, b) = self.align(&other);
        PuiseuxSeries::from_laurent(big, a + b)
    }
}

impl<R: Ring> Sub for PuiseuxSeries<R> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let (big, a, b) = self.align(&other);
        PuiseuxSeries::from_laurent(big, a - b)
    }
}

impl<R: Ring> Mul for PuiseuxSeries<R> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (big, a, b) = self.align(&other);
        PuiseuxSeries::from_laurent(big, a * b)
    }
}

impl<R: Ring> Neg for PuiseuxSeries<R> {
    type Output = Self;
    fn neg(self) -> Self {
        PuiseuxSeries {
            e: self.e,
            inner: -self.inner,
        }
    }
}

impl<R: Ring> Ring for PuiseuxSeries<R> {
    fn zero() -> Self {
        PuiseuxSeries::zero(DEFAULT_PRECISION)
    }
    fn one() -> Self {
        PuiseuxSeries::one(DEFAULT_PRECISION)
    }
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }
    fn is_one(&self) -> bool {
        self.e == 1 && self.inner.is_one()
    }
}

impl<R: CommutativeRing> CommutativeRing for PuiseuxSeries<R> {}
impl<R: IntegralDomain> IntegralDomain for PuiseuxSeries<R> {}

impl<R: Field> PuiseuxSeries<R> {
    /// Multiplicative inverse: invert the internal Laurent series over the
    /// field `R`, keeping the same exponent denominator.
    pub fn try_inverse(&self) -> Result<Self> {
        Ok(PuiseuxSeries {
            e: self.e,
            inner: self.inner.try_inverse()?,
        }
        .reduce())
    }
}

impl<R: Field> Div for PuiseuxSeries<R> {
    type Output = Self;
    // Division in a field is multiplication by the inverse.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        let inv = other
            .try_inverse()
            .expect("division by a zero Puiseux series");
        self * inv
    }
}

impl<R: Field> Field for PuiseuxSeries<R> {
    fn inverse(&self) -> Result<Self> {
        self.try_inverse()
    }
}

impl<R: Ring> fmt::Display for PuiseuxSeries<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.e == 1 {
            return write!(f, "{}", self.inner);
        }
        let mut first = true;
        let base = self.inner.lowest_exponent();
        for i in 0..self.inner.relative_precision() {
            let c = self.inner.coefficient(base + i as i64);
            if c.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            let exp = Rational::new(base + i as i64, self.e as i64).unwrap();
            if exp.is_zero() {
                write!(f, "{c}")?;
            } else if c.is_one() {
                write!(f, "x^({exp})")?;
            } else {
                write!(f, "{c}*x^({exp})")?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        write!(f, " + O(x^({}))", self.absolute_precision())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn fractional_valuation() {
        // x^{1/2} + x^{3/2}
        let f = PuiseuxSeries::from_laurent(2, LaurentSeries::new(1, vec![q(1), q(0), q(1), q(0)]));
        assert_eq!(f.exponent_denominator(), 2);
        assert_eq!(f.valuation(), Some(Rational::new(1, 2).unwrap()));
        assert_eq!(f.coefficient(1, 2), q(1));
        assert_eq!(f.coefficient(3, 2), q(1));
        assert_eq!(f.coefficient(1, 1), q(0)); // x^1 not on the grid pattern here
    }

    #[test]
    fn denominator_reduction() {
        // Only integral exponents present => denominator collapses to 1.
        let f = PuiseuxSeries::from_laurent(2, LaurentSeries::new(0, vec![q(1), q(0), q(3), q(0)]));
        assert_eq!(f.exponent_denominator(), 1);
        assert_eq!(f.coefficient(0, 1), q(1));
        assert_eq!(f.coefficient(1, 1), q(3));
    }

    #[test]
    fn precision_regime_constructors() {
        use crate::precision::Precision;
        // Monomial under a fixed regime carries the regime's term count.
        let m = PuiseuxSeries::monomial_with(1, 2, q(1), Precision::Fixed(4));
        assert_eq!(m.laurent_part().relative_precision(), 4);
        assert_eq!(m.valuation(), Some(Rational::new(1, 2).unwrap()));
        // with_precision delegates window capping to the Laurent layer:
        // x^{1/2} + x^{3/2} + x^{5/2} capped at a 2-term window keeps only
        // the coefficients at y-exponents 1 and 2 (x^{1/2} and the zero at x).
        let f = PuiseuxSeries::with_precision(
            2,
            1,
            vec![q(1), q(0), q(1), q(0), q(1)],
            Precision::Fixed(2),
        );
        assert_eq!(f.coefficient(1, 2), q(1));
        // absolute precision (in x) is (1 + 2)/2 = 3/2: x^{3/2} already unknown
        assert_eq!(f.absolute_precision(), Rational::new(3, 2).unwrap());
    }

    #[test]
    fn arithmetic_propagates_min_precision() {
        // On a common grid, the sum's absolute precision is the operands' min.
        let a = PuiseuxSeries::with_precision(2, 1, vec![q(1), q(1)], Precision::Fixed(2)); // O(x^{3/2})
        let b = PuiseuxSeries::with_precision(2, 1, vec![q(1), q(1), q(1), q(1)], Precision::Fixed(4)); // O(x^{5/2})
        let s = a + b;
        assert_eq!(s.absolute_precision(), Rational::new(3, 2).unwrap());
    }

    #[test]
    fn common_denominator_arithmetic() {
        // x^{1/2} * x^{1/3} = x^{5/6}
        let a = PuiseuxSeries::monomial(1, 2, q(1), 4);
        let b = PuiseuxSeries::monomial(1, 3, q(1), 4);
        let p = a * b;
        assert_eq!(p.valuation(), Some(Rational::new(5, 6).unwrap()));
        assert_eq!(p.coefficient(5, 6), q(1));

        // x^{1/2} + x^{1/2} = 2 x^{1/2}
        let c = PuiseuxSeries::monomial(1, 2, q(1), 4);
        let d = PuiseuxSeries::monomial(1, 2, q(1), 4);
        let s = c + d;
        assert_eq!(s.exponent_denominator(), 2);
        assert_eq!(s.coefficient(1, 2), q(2));
    }

    #[test]
    fn puiseux_field_inverse() {
        // 1/(1 + x^{1/2}) has denominator 2 and starts 1 - x^{1/2} + x - ...
        let f = PuiseuxSeries::from_laurent(2, LaurentSeries::new(0, vec![q(1), q(1), q(0), q(0), q(0), q(0)]));
        let inv = f.try_inverse().unwrap();
        assert_eq!(inv.coefficient(0, 1), q(1));
        assert_eq!(inv.coefficient(1, 2), q(-1));
        assert_eq!(inv.coefficient(2, 2), q(1)); // x^1
        // f * (1/f) = 1
        let prod = f * inv;
        assert!(prod.is_one());
    }

    #[test]
    fn ring_trait_object_usage() {
        fn cube<T: Ring>(t: T) -> T {
            t.clone() * t.clone() * t
        }
        // (x^{1/3})^3 = x
        let g = cube(PuiseuxSeries::monomial(1, 3, q(1), 6));
        assert_eq!(g.valuation(), Some(q(1)));
        assert_eq!(g.exponent_denominator(), 1);
    }
}

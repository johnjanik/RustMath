//! Rational numbers (fractions)

use rustmath_core::ordering::{OrderedField, OrderedRing};
use rustmath_core::{CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Rational number (fraction)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Rational {
    numerator: Integer,
    denominator: Integer,
}

impl Rational {
    /// Create a new rational number from integers, automatically simplifying
    pub fn new<T: Into<Integer>>(numerator: T, denominator: T) -> Result<Self> {
        let num = numerator.into();
        let den = denominator.into();

        if den.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let mut rational = Rational {
            numerator: num,
            denominator: den,
        };

        rational.simplify();
        Ok(rational)
    }

    /// Create a rational from an integer
    pub fn from_integer<T: Into<Integer>>(n: T) -> Self {
        Rational {
            numerator: n.into(),
            denominator: Integer::one(),
        }
    }

    /// Create a rational from an i64
    pub fn from_i64(n: i64) -> Self {
        Rational::from_integer(n)
    }

    /// Get the numerator
    pub fn numerator(&self) -> &Integer {
        &self.numerator
    }

    /// Get the denominator
    pub fn denominator(&self) -> &Integer {
        &self.denominator
    }

    /// Simplify to lowest terms
    fn simplify(&mut self) {
        let gcd = self.numerator.gcd(&self.denominator);

        if !gcd.is_one() {
            self.numerator = self.numerator.clone() / gcd.clone();
            self.denominator = self.denominator.clone() / gcd;
        }

        // Ensure denominator is positive
        if self.denominator.signum() < 0 {
            self.numerator = -self.numerator.clone();
            self.denominator = -self.denominator.clone();
        }
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Rational {
            numerator: self.numerator.abs(),
            denominator: self.denominator.clone(),
        }
    }

    /// Get the reciprocal
    pub fn reciprocal(&self) -> Result<Self> {
        if self.numerator.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        Ok(Rational {
            numerator: self.denominator.clone(),
            denominator: self.numerator.clone(),
        })
    }

    /// Compute the floor (largest integer <= self)
    pub fn floor(&self) -> Integer {
        if self.numerator.signum() >= 0 {
            self.numerator.clone() / self.denominator.clone()
        } else {
            // For negative numbers, need to round down
            let (q, r) = self.numerator.div_rem(&self.denominator).unwrap();
            if r.is_zero() {
                q
            } else {
                q - Integer::one()
            }
        }
    }

    /// Compute the ceiling (smallest integer >= self)
    pub fn ceil(&self) -> Integer {
        -(-self.clone()).floor()
    }

    /// Round to the nearest integer
    ///
    /// Uses the "round half up" rule: 0.5 rounds to 1, -0.5 rounds to -1
    pub fn round(&self) -> Integer {
        // Add 1/2 and take floor for positive, subtract 1/2 and take ceil for negative
        if self.numerator.signum() >= 0 {
            let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
            (self.clone() + half).floor()
        } else {
            let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
            (self.clone() - half).ceil()
        }
    }

    /// Compute the p-adic valuation of this rational number
    ///
    /// Returns v_p(a/b) = v_p(a) - v_p(b) where v_p(n) is the p-adic valuation of n.
    /// This is the exponent of p in the prime factorization when written in lowest terms.
    pub fn valuation(&self, p: &Integer) -> i32 {
        let num_val = self.numerator.valuation(p) as i32;
        let den_val = self.denominator.valuation(p) as i32;
        num_val - den_val
    }

    /// Compute the absolute value (norm) of this rational number
    ///
    /// Returns |a/b| as a rational number
    pub fn norm(&self) -> Self {
        self.abs()
    }

    /// Convert to float (may lose precision)
    pub fn to_f64(&self) -> Option<f64> {
        let num = self.numerator.to_f64()?;
        let den = self.denominator.to_f64()?;
        Some(num / den)
    }

    /// Create a rational from an f64
    ///
    /// Attempts to convert a floating-point number to a rational approximation.
    /// This uses a continued fraction algorithm with a maximum denominator.
    pub fn from_f64(f: f64) -> Result<Self> {
        if f.is_nan() || f.is_infinite() {
            return Err(MathError::InvalidArgument("Cannot convert NaN or infinity to rational".to_string()));
        }

        if f == 0.0 {
            return Ok(Rational::zero());
        }

        // Extract sign
        let sign = if f < 0.0 { -1 } else { 1 };
        let f = f.abs();

        // Simple algorithm: convert to fraction with limited precision
        // For better accuracy, we'd use continued fractions
        let max_denominator = 1_000_000;
        let mut best_num = 1i64;
        let mut best_den = 1i64;
        let mut best_error = (f - best_num as f64 / best_den as f64).abs();

        for den in 1..=max_denominator {
            let num = (f * den as f64).round() as i64;
            let error = (f - num as f64 / den as f64).abs();
            if error < best_error {
                best_num = num;
                best_den = den;
                best_error = error;
                if error < 1e-10 {
                    break;
                }
            }
        }

        Rational::new(sign * best_num, best_den)
    }

    /// Check if this is an integer
    pub fn is_integer(&self) -> bool {
        self.denominator.is_one()
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        let left = self.numerator.clone() * other.denominator.clone();
        let right = other.numerator.clone() * self.denominator.clone();
        left.cmp(&right)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rational({}/{})", self.numerator, self.denominator)
    }
}

// Typesetting implementation
impl rustmath_typesetting::MathDisplay for Rational {
    fn math_format(&self, options: &rustmath_typesetting::FormatOptions) -> String {
        use rustmath_typesetting::OutputFormat;

        // If denominator is 1, just display as an integer
        if self.denominator.is_one() {
            return self.numerator.math_format(options);
        }

        let num_str = self.numerator.to_string();
        let den_str = self.denominator.to_string();

        match options.format {
            OutputFormat::LaTeX => {
                rustmath_typesetting::latex::fraction(&num_str, &den_str, options)
            }
            OutputFormat::Unicode => {
                rustmath_typesetting::unicode::fraction(&num_str, &den_str)
            }
            OutputFormat::Ascii => {
                rustmath_typesetting::ascii::fraction(&num_str, &den_str, options.mode)
            }
            OutputFormat::Html => {
                rustmath_typesetting::html::fraction(&num_str, &den_str)
            }
            OutputFormat::Plain => format!("{}/{}", num_str, den_str),
        }
    }

    fn precedence(&self) -> i32 {
        if self.denominator.is_one() {
            rustmath_typesetting::utils::precedence::ATOMIC
        } else {
            // Fractions should be treated as division
            rustmath_typesetting::utils::precedence::MULTIPLY
        }
    }
}

// Arithmetic operations
impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Add<&'b Rational> for &Rational {
    type Output = Rational;

    fn add(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.denominator + &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;

        Rational::new(num, den).unwrap()
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Sub<&'b Rational> for &Rational {
    type Output = Rational;

    fn sub(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.denominator - &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;

        Rational::new(num, den).unwrap()
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.numerator.clone();
        let den = self.denominator.clone() * other.denominator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Mul<&'b Rational> for &Rational {
    type Output = Rational;

    fn mul(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.numerator;
        let den = &self.denominator * &other.denominator;

        Rational::new(num, den).unwrap()
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let num = self.numerator.clone() * other.denominator.clone();
        let den = self.denominator.clone() * other.numerator.clone();

        Rational::new(num, den).unwrap()
    }
}

impl<'b> Div<&'b Rational> for &Rational {
    type Output = Rational;

    fn div(self, other: &'b Rational) -> Rational {
        let num = &self.numerator * &other.denominator;
        let den = &self.denominator * &other.numerator;

        Rational::new(num, den).unwrap()
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Rational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl Neg for &Rational {
    type Output = Rational;

    fn neg(self) -> Rational {
        Rational {
            numerator: -&self.numerator,
            denominator: self.denominator.clone(),
        }
    }
}

// Ring trait implementation
impl Ring for Rational {
    fn zero() -> Self {
        Rational {
            numerator: Integer::zero(),
            denominator: Integer::one(),
        }
    }

    fn one() -> Self {
        Rational {
            numerator: Integer::one(),
            denominator: Integer::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    fn is_one(&self) -> bool {
        self.numerator.is_one() && self.denominator.is_one()
    }
}

impl CommutativeRing for Rational {}
impl IntegralDomain for Rational {}

impl EuclideanDomain for Rational {
    fn norm(&self) -> u64 {
        // For a field, we can use a trivial norm: 0 for zero, 1 for non-zero
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
        // In a field, division is always exact (no remainder)
        Ok((self.clone() / other.clone(), Rational::zero()))
    }
}

impl Field for Rational {
    fn inverse(&self) -> Result<Self> {
        self.reciprocal()
    }

    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(self.clone() / other.clone())
        }
    }
}

// Ordered-field structure: the order is the standard total order on the
// rationals (already exposed via `PartialOrd`/`Ord` above), which is
// translation-invariant under `+` and compatible with `*` on non-negative
// elements (MAGMA Handbook ch. 25 comparison-aware field requirement).
impl OrderedRing for Rational {
    fn sign(&self) -> i32 {
        // `denominator` is always kept strictly positive by `simplify`, so
        // the sign of the fraction is exactly the sign of the numerator.
        self.numerator.signum() as i32
    }

    fn abs(&self) -> Self {
        Rational::abs(self)
    }
}

impl OrderedField for Rational {}

impl NumericConversion for Rational {
    fn from_i64(n: i64) -> Self {
        Rational::from_integer(n)
    }

    fn from_u64(n: u64) -> Self {
        Rational::from_integer(n)
    }

    fn to_i64(&self) -> Option<i64> {
        if self.is_integer() {
            Some(self.numerator.to_i64())
        } else {
            None
        }
    }

    fn to_u64(&self) -> Option<u64> {
        if self.is_integer() {
            self.numerator.to_u64()
        } else {
            None
        }
    }

    fn to_usize(&self) -> Option<usize> {
        if self.is_integer() {
            self.numerator.to_usize()
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        self.to_f64()
    }
}

// Implement From<i64> for Rational
impl From<i64> for Rational {
    fn from(n: i64) -> Self {
        Rational::from_integer(n)
    }
}

// Implement From<i32> for convenience
impl From<i32> for Rational {
    fn from(n: i32) -> Self {
        Rational::from_integer(n as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation_and_simplification() {
        let r = Rational::new(4, 6).unwrap();
        assert_eq!(r.numerator(), &Integer::from(2));
        assert_eq!(r.denominator(), &Integer::from(3));
    }

    #[test]
    fn test_arithmetic() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(1, 3).unwrap();

        let sum = a.clone() + b.clone();
        assert_eq!(sum, Rational::new(5, 6).unwrap());

        let diff = a.clone() - b.clone();
        assert_eq!(diff, Rational::new(1, 6).unwrap());

        let prod = a.clone() * b.clone();
        assert_eq!(prod, Rational::new(1, 6).unwrap());

        let quot = a.clone() / b.clone();
        assert_eq!(quot, Rational::new(3, 2).unwrap());
    }

    #[test]
    fn test_comparison() {
        let a = Rational::new(1, 2).unwrap();
        let b = Rational::new(2, 3).unwrap();

        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn test_floor_ceil() {
        let r = Rational::new(7, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(2));
        assert_eq!(r.ceil(), Integer::from(3));

        let r = Rational::new(-7, 3).unwrap();
        assert_eq!(r.floor(), Integer::from(-3));
        assert_eq!(r.ceil(), Integer::from(-2));
    }

    // -- OrderedField / OrderedRing laws --------------------------------
    //
    // Compile-time proof that `Rational` actually satisfies the
    // `OrderedField` bound (which in turn requires `Field + OrderedRing`,
    // i.e. `Ring + PartialOrd + Field`).
    fn _assert_rational_is_ordered_field<F: OrderedField>() {}
    const _: fn() = || _assert_rational_is_ordered_field::<Rational>();

    fn r(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    #[test]
    fn test_ordered_ring_sign() {
        assert_eq!(OrderedRing::sign(&r(5, 7)), 1);
        assert_eq!(OrderedRing::sign(&r(-5, 7)), -1);
        assert_eq!(OrderedRing::sign(&r(0, 1)), 0);
        // Sign must be invariant under the sign of numerator/denominator
        // individually (only the overall sign of the fraction matters).
        assert_eq!(OrderedRing::sign(&r(-3, -4)), 1);
        assert_eq!(OrderedRing::sign(&r(3, -4)), -1);
    }

    #[test]
    fn test_ordered_ring_abs() {
        assert_eq!(OrderedRing::abs(&r(-3, 4)), r(3, 4));
        assert_eq!(OrderedRing::abs(&r(3, 4)), r(3, 4));
        assert_eq!(OrderedRing::abs(&r(0, 1)), r(0, 1));
        // The trait method must agree with the pre-existing inherent `abs`.
        let x = r(-11, 5);
        assert_eq!(OrderedRing::abs(&x), x.abs());
    }

    #[test]
    fn test_ordered_ring_is_positive_negative() {
        assert!(r(1, 3).is_positive());
        assert!(!r(1, 3).is_negative());
        assert!(r(-1, 3).is_negative());
        assert!(!r(-1, 3).is_positive());
        assert!(!r(0, 1).is_positive());
        assert!(!r(0, 1).is_negative());
    }

    #[test]
    fn test_ordered_ring_min_max() {
        let a = r(1, 2);
        let b = r(2, 3);
        assert_eq!(a.max_with(&b), b.clone());
        assert_eq!(a.min_with(&b), a.clone());
        assert_eq!(a.max_with(&a), a.clone());
        assert_eq!(a.min_with(&a), a.clone());
    }

    #[test]
    fn test_ordered_field_total_order_trichotomy() {
        let values = [r(-3, 2), r(-1, 1), r(0, 1), r(1, 4), r(2, 3), r(5, 1)];
        for a in &values {
            for b in &values {
                // Exactly one of <, ==, > holds (trichotomy of a total order).
                let lt = a < b;
                let eq = a == b;
                let gt = a > b;
                assert_eq!(lt as u8 + eq as u8 + gt as u8, 1);
                // `PartialOrd` agrees with the direct comparison operators.
                assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
            }
        }
    }

    #[test]
    fn test_ordered_field_translation_invariance() {
        // a <= b  =>  a + c <= b + c, for every c (positive, negative, zero,
        // integral or fractional).
        let pairs = [(r(1, 3), r(2, 3)), (r(-5, 2), r(-1, 2)), (r(0, 1), r(0, 1))];
        let shifts = [r(0, 1), r(1, 1), r(-1, 1), r(7, 4), r(-7, 4)];

        for (a, b) in &pairs {
            assert!(a <= b);
            for c in &shifts {
                assert!(
                    (a.clone() + c.clone()) <= (b.clone() + c.clone()),
                    "translation invariance failed for a={a}, b={b}, c={c}"
                );
            }
        }
    }

    #[test]
    fn test_ordered_field_multiplicative_compatibility() {
        // Product of two non-negative elements is non-negative.
        let nonneg = [r(0, 1), r(1, 5), r(3, 2), r(9, 1)];
        for a in &nonneg {
            for b in &nonneg {
                assert!((a.clone() * b.clone()).sign() >= 0);
            }
        }

        // Multiplying an inequality by a strictly positive constant
        // preserves the order: a <= b && c > 0 => a*c <= b*c.
        let a = r(1, 3);
        let b = r(5, 6);
        assert!(a <= b);
        for c in [r(1, 7), r(2, 1), r(11, 3)] {
            assert!(c.is_positive());
            assert!(a.clone() * c.clone() <= b.clone() * c.clone());
        }

        // Multiplying by a strictly negative constant reverses the order.
        for c in [r(-1, 7), r(-2, 1), r(-11, 3)] {
            assert!(c.is_negative());
            assert!(a.clone() * c.clone() >= b.clone() * c.clone());
        }
    }

    #[test]
    fn test_ordered_field_abs_is_multiplicative_and_triangle_inequality() {
        let values = [r(-7, 3), r(0, 1), r(4, 5), r(-1, 1), r(11, 2)];
        for x in &values {
            for y in &values {
                // |x*y| = |x|*|y|
                let lhs = (x.clone() * y.clone()).abs();
                let rhs = x.abs() * y.abs();
                assert_eq!(lhs, rhs);
                // Triangle inequality: |x + y| <= |x| + |y|
                let sum_abs = (x.clone() + y.clone()).abs();
                let abs_sum = x.abs() + y.abs();
                assert!(sum_abs <= abs_sum);
            }
        }
    }
}

//! Interval arithmetic for real numbers

use crate::Real;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Interval [a, b] representing all real numbers x such that a ≤ x ≤ b
///
/// Interval arithmetic provides guaranteed bounds on computations,
/// useful for verified numerical analysis and error bounds.
#[derive(Clone, Debug)]
pub struct Interval {
    lower: Real,
    upper: Real,
}

impl Interval {
    /// Create a new interval [lower, upper]
    ///
    /// Panics if lower > upper
    pub fn new(lower: Real, upper: Real) -> Self {
        assert!(
            lower.to_f64() <= upper.to_f64(),
            "Invalid interval: lower bound must be <= upper bound"
        );
        Interval { lower, upper }
    }

    /// Create an interval from f64 bounds
    pub fn from_f64(lower: f64, upper: f64) -> Self {
        Self::new(Real::from(lower), Real::from(upper))
    }

    /// Create a point interval [a, a]
    pub fn point(value: Real) -> Self {
        Interval {
            lower: value.clone(),
            upper: value,
        }
    }

    /// Get the lower bound
    pub fn lower(&self) -> &Real {
        &self.lower
    }

    /// Get the upper bound
    pub fn upper(&self) -> &Real {
        &self.upper
    }

    /// Get the midpoint of the interval
    pub fn midpoint(&self) -> Real {
        let two = Real::from(2.0);
        (self.lower.clone() + self.upper.clone()) / two
    }

    /// Get the width (diameter) of the interval
    pub fn width(&self) -> Real {
        self.upper.clone() - self.lower.clone()
    }

    /// Get the radius (half-width) of the interval
    pub fn radius(&self) -> Real {
        let two = Real::from(2.0);
        self.width() / two
    }

    /// Check if this interval contains a value
    pub fn contains(&self, value: &Real) -> bool {
        let v = value.to_f64();
        let l = self.lower.to_f64();
        let u = self.upper.to_f64();
        l <= v && v <= u
    }

    /// Check if this interval contains another interval
    pub fn contains_interval(&self, other: &Interval) -> bool {
        let l1 = self.lower.to_f64();
        let u1 = self.upper.to_f64();
        let l2 = other.lower.to_f64();
        let u2 = other.upper.to_f64();
        l1 <= l2 && u2 <= u1
    }

    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &Interval) -> bool {
        let l1 = self.lower.to_f64();
        let u1 = self.upper.to_f64();
        let l2 = other.lower.to_f64();
        let u2 = other.upper.to_f64();
        !(u1 < l2 || u2 < l1)
    }

    /// Compute the intersection of two intervals
    ///
    /// Returns None if the intervals don't overlap
    pub fn intersection(&self, other: &Interval) -> Option<Interval> {
        let l1 = self.lower.to_f64();
        let u1 = self.upper.to_f64();
        let l2 = other.lower.to_f64();
        let u2 = other.upper.to_f64();

        let lower = l1.max(l2);
        let upper = u1.min(u2);

        if lower <= upper {
            Some(Interval::from_f64(lower, upper))
        } else {
            None
        }
    }

    /// Compute the hull (smallest interval containing both)
    pub fn hull(&self, other: &Interval) -> Interval {
        let l1 = self.lower.to_f64();
        let u1 = self.upper.to_f64();
        let l2 = other.lower.to_f64();
        let u2 = other.upper.to_f64();

        Interval::from_f64(l1.min(l2), u1.max(u2))
    }

    /// Check if this is a point interval (zero width)
    pub fn is_point(&self) -> bool {
        self.width().to_f64() == 0.0
    }

    /// Split the interval at its midpoint
    pub fn split(&self) -> (Interval, Interval) {
        let mid = self.midpoint();
        (
            Interval::new(self.lower.clone(), mid.clone()),
            Interval::new(mid, self.upper.clone()),
        )
    }

    /// Absolute value of an interval
    pub fn abs(&self) -> Interval {
        let l = self.lower.to_f64();
        let u = self.upper.to_f64();

        if l >= 0.0 {
            // [a, b] with a >= 0 => [a, b]
            self.clone()
        } else if u <= 0.0 {
            // [a, b] with b <= 0 => [-b, -a]
            Interval::from_f64(-u, -l)
        } else {
            // [a, b] with a < 0 < b => [0, max(|a|, |b|)]
            Interval::from_f64(0.0, l.abs().max(u.abs()))
        }
    }

    /// Square of an interval
    pub fn square(&self) -> Interval {
        self.clone() * self.clone()
    }

    /// Square root of an interval (requires non-negative interval)
    pub fn sqrt(&self) -> Option<Interval> {
        let l = self.lower.to_f64();
        let u = self.upper.to_f64();

        if l < 0.0 {
            None // Cannot take sqrt of negative numbers
        } else {
            Some(Interval::from_f64(l.sqrt(), u.sqrt()))
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

impl Add for Interval {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // [a, b] + [c, d] = [a+c, b+d]
        Interval {
            lower: self.lower + other.lower,
            upper: self.upper + other.upper,
        }
    }
}

impl Sub for Interval {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // [a, b] - [c, d] = [a-d, b-c]
        Interval {
            lower: self.lower - other.upper,
            upper: self.upper - other.lower,
        }
    }
}

impl Mul for Interval {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // [a, b] * [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
        let l1 = self.lower.to_f64();
        let u1 = self.upper.to_f64();
        let l2 = other.lower.to_f64();
        let u2 = other.upper.to_f64();

        let products = vec![l1 * l2, l1 * u2, u1 * l2, u1 * u2];
        let min = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Interval::from_f64(min, max)
    }
}

impl Div for Interval {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // Division is more complex - we need to handle division by intervals containing zero
        let l2 = other.lower.to_f64();
        let u2 = other.upper.to_f64();

        if l2 <= 0.0 && 0.0 <= u2 {
            // Divisor contains zero - result is entire real line (or large interval)
            return Interval::from_f64(f64::NEG_INFINITY, f64::INFINITY);
        }

        // [a, b] / [c, d] = [a, b] * [1/d, 1/c] (when 0 not in [c, d])
        let recip_lower = Real::from(1.0 / u2);
        let recip_upper = Real::from(1.0 / l2);

        self * Interval::new(recip_lower, recip_upper)
    }
}

impl Neg for Interval {
    type Output = Self;

    fn neg(self) -> Self {
        // -[a, b] = [-b, -a]
        Interval {
            lower: -self.upper,
            upper: -self.lower,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_interval() {
        let i = Interval::from_f64(1.0, 3.0);
        assert_eq!(i.lower().to_f64(), 1.0);
        assert_eq!(i.upper().to_f64(), 3.0);
        assert_eq!(i.midpoint().to_f64(), 2.0);
        assert_eq!(i.width().to_f64(), 2.0);
    }

    #[test]
    fn test_addition() {
        let i1 = Interval::from_f64(1.0, 2.0);
        let i2 = Interval::from_f64(3.0, 4.0);
        let sum = i1 + i2;

        assert_eq!(sum.lower().to_f64(), 4.0);
        assert_eq!(sum.upper().to_f64(), 6.0);
    }

    #[test]
    fn test_multiplication() {
        let i1 = Interval::from_f64(2.0, 3.0);
        let i2 = Interval::from_f64(4.0, 5.0);
        let prod = i1 * i2;

        assert_eq!(prod.lower().to_f64(), 8.0);
        assert_eq!(prod.upper().to_f64(), 15.0);
    }

    #[test]
    fn test_contains() {
        let i = Interval::from_f64(1.0, 5.0);
        assert!(i.contains(&Real::from(3.0)));
        assert!(!i.contains(&Real::from(6.0)));
    }

    #[test]
    fn test_intersection() {
        let i1 = Interval::from_f64(1.0, 5.0);
        let i2 = Interval::from_f64(3.0, 7.0);

        let intersection = i1.intersection(&i2).unwrap();
        assert_eq!(intersection.lower().to_f64(), 3.0);
        assert_eq!(intersection.upper().to_f64(), 5.0);
    }

    #[test]
    fn test_abs() {
        let i = Interval::from_f64(-3.0, 2.0);
        let abs_i = i.abs();

        assert_eq!(abs_i.lower().to_f64(), 0.0);
        assert_eq!(abs_i.upper().to_f64(), 3.0);
    }

    #[test]
    fn test_sqrt() {
        let i = Interval::from_f64(4.0, 9.0);
        let sqrt_i = i.sqrt().unwrap();

        assert_eq!(sqrt_i.lower().to_f64(), 2.0);
        assert_eq!(sqrt_i.upper().to_f64(), 3.0);
    }
}

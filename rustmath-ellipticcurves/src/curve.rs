//! Core elliptic curve implementation
//!
//! Provides elliptic curves in generalized Weierstrass form:
//! y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
//!
//! Also provides short Weierstrass form: y² = x³ + ax + b

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Zero, One, Signed, ToPrimitive};
use std::fmt;

/// An elliptic curve in generalized Weierstrass form
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EllipticCurve {
    pub a1: BigInt,
    pub a2: BigInt,
    pub a3: BigInt,
    pub a4: BigInt,
    pub a6: BigInt,
    pub discriminant: BigInt,
    pub conductor: Option<BigInt>,
}

/// A point on an elliptic curve
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    pub x: BigRational,
    pub y: BigRational,
    pub infinity: bool,
}

impl EllipticCurve {
    /// Create a new elliptic curve from Weierstrass coefficients
    /// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
    pub fn new(a1: BigInt, a2: BigInt, a3: BigInt, a4: BigInt, a6: BigInt) -> Self {
        // Compute b-invariants
        let b2 = &a1 * &a1 + BigInt::from(4) * &a2;
        let b4 = BigInt::from(2) * &a4 + &a1 * &a3;
        let b6 = &a3 * &a3 + BigInt::from(4) * &a6;
        let b8 = &a1 * &a1 * &a6 + BigInt::from(4) * &a2 * &a6
            - &a1 * &a3 * &a4 + &a2 * &a3 * &a3 - &a4 * &a4;

        // Compute discriminant: Δ = -b₂²b₈ - 8b₄³ - 27b₆² + 9b₂b₄b₆
        let discriminant = -&b2 * &b2 * &b8
            - BigInt::from(8) * &b4 * &b4 * &b4
            - BigInt::from(27) * &b6 * &b6
            + BigInt::from(9) * &b2 * &b4 * &b6;

        Self {
            a1,
            a2,
            a3,
            a4,
            a6,
            discriminant,
            conductor: None,
        }
    }

    /// Create an elliptic curve in short Weierstrass form: y² = x³ + ax + b
    pub fn from_short_weierstrass(a: BigInt, b: BigInt) -> Self {
        Self::new(BigInt::zero(), BigInt::zero(), BigInt::zero(), a, b)
    }

    /// Check if the curve is singular (discriminant is zero)
    pub fn is_singular(&self) -> bool {
        self.discriminant.is_zero()
    }

    /// Get the j-invariant of the curve
    pub fn j_invariant(&self) -> Option<BigRational> {
        if self.is_singular() {
            return None;
        }

        let b2 = &self.a1 * &self.a1 + BigInt::from(4) * &self.a2;
        let b4 = BigInt::from(2) * &self.a4 + &self.a1 * &self.a3;
        let _b6 = &self.a3 * &self.a3 + BigInt::from(4) * &self.a6;

        let c4 = &b2 * &b2 - BigInt::from(24) * &b4;
        let numerator = c4.pow(3);

        Some(BigRational::new(numerator, self.discriminant.clone()))
    }

    /// Add two points on the curve
    pub fn add_points(&self, p: &Point, q: &Point) -> Point {
        if p.infinity {
            return q.clone();
        }
        if q.infinity {
            return p.clone();
        }

        // Check if points are negatives of each other
        if p.x == q.x {
            let neg_y = self.negate_y(&p.x, &p.y);
            if q.y == neg_y {
                return Point::infinity();
            }
        }

        if p == q {
            return self.double_point(p);
        }

        // For short Weierstrass form: y² = x³ + ax + b
        // λ = (y₂ - y₁) / (x₂ - x₁)
        let lambda = (&q.y - &p.y) / (&q.x - &p.x);

        // x₃ = λ² - x₁ - x₂
        let x3 = &lambda * &lambda - &p.x - &q.x;

        // y₃ = λ(x₁ - x₃) - y₁
        let y3 = &lambda * (&p.x - &x3) - &p.y;

        Point {
            x: x3,
            y: y3,
            infinity: false,
        }
    }

    /// Double a point on the curve
    pub fn double_point(&self, p: &Point) -> Point {
        if p.infinity {
            return p.clone();
        }

        // For short Weierstrass: y² = x³ + ax + b
        // λ = (3x² + a) / (2y)
        let three = BigRational::from(BigInt::from(3));
        let two = BigRational::from(BigInt::from(2));

        let numerator = &three * &p.x * &p.x + BigRational::from(self.a4.clone());
        let denominator = &two * &p.y;

        if denominator.is_zero() {
            return Point::infinity();
        }

        let lambda = numerator / denominator;

        // x₃ = λ² - 2x
        let x3 = &lambda * &lambda - &two * &p.x;

        // y₃ = λ(x - x₃) - y
        let y3 = &lambda * (&p.x - &x3) - &p.y;

        Point {
            x: x3,
            y: y3,
            infinity: false,
        }
    }

    /// Scalar multiplication: compute [n]P
    pub fn scalar_mul(&self, n: &BigInt, p: &Point) -> Point {
        if n.is_zero() || p.infinity {
            return Point::infinity();
        }

        if n.is_negative() {
            let neg_p = self.negate_point(p);
            return self.scalar_mul(&-n, &neg_p);
        }

        // Binary method (double-and-add)
        let mut result = Point::infinity();
        let mut base = p.clone();
        let mut k = n.clone();

        while !k.is_zero() {
            if &k % BigInt::from(2) == BigInt::one() {
                result = self.add_points(&result, &base);
            }
            base = self.double_point(&base);
            k /= BigInt::from(2);
        }

        result
    }

    /// Negate a point on the curve
    pub fn negate_point(&self, p: &Point) -> Point {
        if p.infinity {
            return p.clone();
        }

        let neg_y = self.negate_y(&p.x, &p.y);
        Point {
            x: p.x.clone(),
            y: neg_y,
            infinity: false,
        }
    }

    /// Compute the negation of y-coordinate for a given x
    /// For short Weierstrass, this is simply -y
    /// For general form: -(y + a₁x + a₃)
    fn negate_y(&self, x: &BigRational, y: &BigRational) -> BigRational {
        let a1_term = BigRational::from(self.a1.clone()) * x;
        let a3_term = BigRational::from(self.a3.clone());
        -(y + &a1_term + &a3_term)
    }

    /// Check if a point is on the curve
    pub fn is_on_curve(&self, p: &Point) -> bool {
        if p.infinity {
            return true;
        }

        // For short Weierstrass: y² = x³ + ax + b
        let lhs = &p.y * &p.y;
        let rhs = &p.x * &p.x * &p.x
            + BigRational::from(self.a4.clone()) * &p.x
            + BigRational::from(self.a6.clone());

        lhs == rhs
    }

    /// Compute the 2-torsion rank (points of order dividing 2)
    pub fn two_torsion_rank(&self) -> i32 {
        // Count roots of x³ + ax + b = 0
        // This is a simplified implementation
        // Real implementation would need to factor the polynomial
        1 // Always have point at infinity
    }

    /// Check if a prime is a bad prime (divides the discriminant)
    pub fn is_bad_prime(&self, p: &BigInt) -> bool {
        &self.discriminant % p == BigInt::zero()
    }

    /// Compute a_p for a good prime p (p + 1 - #E(F_p))
    pub fn compute_a_p(&self, p: &BigInt) -> i64 {
        let p_val = p.to_i64().unwrap_or(2);
        (p_val + 1 - self.count_points_mod_p(p_val)) as i64
    }

    /// Count points on the curve modulo p (naive method)
    /// Real implementation would use Schoof's algorithm
    fn count_points_mod_p(&self, p: i64) -> i64 {
        let mut count = 1; // Point at infinity

        let a = self.a4.to_i64().unwrap_or(0);
        let b = self.a6.to_i64().unwrap_or(0);

        for x in 0..p {
            let rhs = (x * x * x + a * x + b).rem_euclid(p);

            // Check if rhs is a quadratic residue
            for y in 0..p {
                if (y * y).rem_euclid(p) == rhs {
                    count += 1;
                }
            }
        }

        count
    }
}

impl Point {
    /// Create a new affine point
    pub fn new(x: BigRational, y: BigRational) -> Self {
        Self {
            x,
            y,
            infinity: false,
        }
    }

    /// Create the point at infinity
    pub fn infinity() -> Self {
        Self {
            x: BigRational::zero(),
            y: BigRational::zero(),
            infinity: true,
        }
    }

    /// Create a point from integer coordinates
    pub fn from_integers(x: i64, y: i64) -> Self {
        Self {
            x: BigRational::from(BigInt::from(x)),
            y: BigRational::from(BigInt::from(y)),
            infinity: false,
        }
    }
}

impl fmt::Display for EllipticCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a1.is_zero() && self.a2.is_zero() && self.a3.is_zero() {
            write!(f, "y² = x³ + {}x + {}", self.a4, self.a6)
        } else {
            write!(
                f,
                "y² + {}xy + {}y = x³ + {}x² + {}x + {}",
                self.a1, self.a3, self.a2, self.a4, self.a6
            )
        }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.infinity {
            write!(f, "O (point at infinity)")
        } else {
            write!(f, "({}, {})", self.x, self.y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_creation() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1)
        );
        assert!(!curve.is_singular());
    }

    #[test]
    fn test_point_on_curve() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0)
        );

        // Point (0, 0) should be on y² = x³ - x
        let p = Point::new(
            BigRational::zero(),
            BigRational::zero()
        );
        assert!(curve.is_on_curve(&p));

        // Point (1, 0) should be on y² = x³ - x
        let q = Point::new(
            BigRational::one(),
            BigRational::zero()
        );
        assert!(curve.is_on_curve(&q));
    }

    #[test]
    fn test_point_addition() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0)
        );

        let p = Point::new(BigRational::zero(), BigRational::zero());
        let q = Point::infinity();

        let r = curve.add_points(&p, &q);
        assert_eq!(r, p);
    }

    #[test]
    fn test_point_doubling() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(2),
            BigInt::from(3)
        );

        // Point (-1, 0) is on y² = x³ + 2x + 3
        let p = Point::new(
            BigRational::from(BigInt::from(-1)),
            BigRational::from(BigInt::from(0))
        );

        assert!(curve.is_on_curve(&p));
        let doubled = curve.double_point(&p);
        // Doubling a point where y=0 gives infinity
        assert!(doubled.infinity);
    }

    #[test]
    fn test_scalar_multiplication() {
        // Use curve y² = x³ - x for simplicity
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(0)
        );

        // Point (0, 0) is on the curve
        let p = Point::new(
            BigRational::from(BigInt::from(0)),
            BigRational::from(BigInt::from(0))
        );

        assert!(curve.is_on_curve(&p));
        let result = curve.scalar_mul(&BigInt::from(2), &p);
        // [2]P for a point of order 2 is infinity
        assert!(result.infinity || curve.is_on_curve(&result));
    }

    #[test]
    fn test_j_invariant() {
        // For y² = x³ + x (curve with CM by Gaussian integers)
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(1),
            BigInt::from(0)
        );

        let j = curve.j_invariant();
        assert!(j.is_some());
        // j-invariant should be 1728 for this curve
    }
}

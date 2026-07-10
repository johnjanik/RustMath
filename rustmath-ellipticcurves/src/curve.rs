//! Core elliptic curve implementation
//!
//! Provides elliptic curves in generalized Weierstrass form:
//! y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
//!
//! Also provides short Weierstrass form: y² = x³ + ax + b
//!
//! # Canonicalization note (B3)
//!
//! This is the over-Q implementation, now backed by
//! `rustmath_integers::Integer` / `rustmath_rationals::Rational` (Phase-2
//! num-*-to-core normalization). It still overlaps with
//! `rustmath_schemes::elliptic_curves::rational::EllipticCurveRational`
//! (also over Q); merging the two over-Q implementations remains DEFERRED
//! until that one is normalized as well. The generic-field curve lives at
//! `rustmath_ellipticcurves::generic::EllipticCurve` and is unaffected.

use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;

/// An elliptic curve in generalized Weierstrass form
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EllipticCurve {
    pub a1: Integer,
    pub a2: Integer,
    pub a3: Integer,
    pub a4: Integer,
    pub a6: Integer,
    pub discriminant: Integer,
    pub conductor: Option<Integer>,
}

/// A point on an elliptic curve
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Point {
    pub x: Rational,
    pub y: Rational,
    pub infinity: bool,
}

impl EllipticCurve {
    /// Create a new elliptic curve from Weierstrass coefficients
    /// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
    pub fn new(a1: Integer, a2: Integer, a3: Integer, a4: Integer, a6: Integer) -> Self {
        // Compute b-invariants
        let b2 = &a1 * &a1 + Integer::from(4) * a2.clone();
        let b4 = Integer::from(2) * a4.clone() + &a1 * &a3;
        let b6 = &a3 * &a3 + Integer::from(4) * a6.clone();
        let b8 = a1.clone() * a1.clone() * a6.clone() + Integer::from(4) * a2.clone() * a6.clone()
            - a1.clone() * a3.clone() * a4.clone()
            + a2.clone() * a3.clone() * a3.clone()
            - a4.clone() * a4.clone();

        // Compute discriminant: Δ = -b₂²b₈ - 8b₄³ - 27b₆² + 9b₂b₄b₆
        let discriminant = -(b2.clone() * b2.clone() * b8)
            - Integer::from(8) * b4.clone() * b4.clone() * b4.clone()
            - Integer::from(27) * b6.clone() * b6.clone()
            + Integer::from(9) * b2 * b4 * b6;

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
    pub fn from_short_weierstrass(a: Integer, b: Integer) -> Self {
        Self::new(Integer::zero(), Integer::zero(), Integer::zero(), a, b)
    }

    /// Check if the curve is singular (discriminant is zero)
    pub fn is_singular(&self) -> bool {
        self.discriminant.is_zero()
    }

    /// Get the j-invariant of the curve
    pub fn j_invariant(&self) -> Option<Rational> {
        if self.is_singular() {
            return None;
        }

        let b2 = &self.a1 * &self.a1 + Integer::from(4) * self.a2.clone();
        let b4 = Integer::from(2) * self.a4.clone() + &self.a1 * &self.a3;
        let _b6 = &self.a3 * &self.a3 + Integer::from(4) * self.a6.clone();

        let c4 = &b2 * &b2 - Integer::from(24) * b4;
        let numerator = c4.pow(3);

        Some(
            Rational::new(numerator, self.discriminant.clone())
                .expect("non-singular curve has nonzero discriminant"),
        )
    }

    /// Add two points on the curve (general Weierstrass form).
    ///
    /// For the chord through P ≠ ±Q with slope λ and intercept ν
    /// (y = λx + ν), the sum is
    /// x₃ = λ² + a₁λ − a₂ − x₁ − x₂,  y₃ = −(λ + a₁)x₃ − ν − a₃.
    /// Both inputs are assumed to lie on the curve.
    pub fn add_points(&self, p: &Point, q: &Point) -> Point {
        if p.infinity {
            return q.clone();
        }
        if q.infinity {
            return p.clone();
        }

        if p.x == q.x {
            // Same x-coordinate: either negatives of each other or equal
            // (two points on the curve with equal x are equal or negatives).
            let neg_y = self.negate_y(&p.x, &p.y);
            if q.y == neg_y {
                return Point::infinity();
            }
            return self.double_point(p);
        }

        // λ = (y₂ - y₁) / (x₂ - x₁), ν = y₁ - λx₁
        let lambda = (&q.y - &p.y) / (&q.x - &p.x);
        let nu = p.y.clone() - lambda.clone() * p.x.clone();

        let a1 = Rational::from_integer(self.a1.clone());
        let a2 = Rational::from_integer(self.a2.clone());
        let a3 = Rational::from_integer(self.a3.clone());

        // x₃ = λ² + a₁λ − a₂ − x₁ − x₂
        let x3 = &lambda * &lambda + a1.clone() * lambda.clone() - a2 - p.x.clone() - q.x.clone();

        // y₃ = −(λ + a₁)x₃ − ν − a₃
        let y3 = -((lambda + a1) * x3.clone()) - nu - a3;

        Point {
            x: x3,
            y: y3,
            infinity: false,
        }
    }

    /// Double a point on the curve (general Weierstrass form).
    ///
    /// The tangent slope is λ = (3x² + 2a₂x + a₄ − a₁y) / (2y + a₁x + a₃);
    /// a vanishing denominator means P is 2-torsion and [2]P = O.
    pub fn double_point(&self, p: &Point) -> Point {
        if p.infinity {
            return p.clone();
        }

        let a1 = Rational::from_integer(self.a1.clone());
        let a2 = Rational::from_integer(self.a2.clone());
        let a3 = Rational::from_integer(self.a3.clone());
        let a4 = Rational::from_integer(self.a4.clone());
        let two = Rational::from_i64(2);
        let three = Rational::from_i64(3);

        let numerator =
            three * p.x.clone() * p.x.clone() + two.clone() * a2.clone() * p.x.clone() + a4
                - a1.clone() * p.y.clone();
        let denominator = two.clone() * p.y.clone() + a1.clone() * p.x.clone() + a3.clone();

        if denominator.is_zero() {
            return Point::infinity();
        }

        let lambda = numerator / denominator;
        let nu = p.y.clone() - lambda.clone() * p.x.clone();

        // x₃ = λ² + a₁λ − a₂ − 2x
        let x3 = &lambda * &lambda + a1.clone() * lambda.clone() - a2 - two * p.x.clone();

        // y₃ = −(λ + a₁)x₃ − ν − a₃
        let y3 = -((lambda + a1) * x3.clone()) - nu - a3;

        Point {
            x: x3,
            y: y3,
            infinity: false,
        }
    }

    /// Scalar multiplication: compute [n]P
    pub fn scalar_mul(&self, n: &Integer, p: &Point) -> Point {
        if n.is_zero() || p.infinity {
            return Point::infinity();
        }

        if n.signum() < 0 {
            let neg_p = self.negate_point(p);
            return self.scalar_mul(&-n, &neg_p);
        }

        // Binary method (double-and-add)
        let mut result = Point::infinity();
        let mut base = p.clone();
        let mut k = n.clone();
        let two = Integer::from(2);

        while !k.is_zero() {
            if (&k % &two).is_one() {
                result = self.add_points(&result, &base);
            }
            base = self.double_point(&base);
            k = k / two.clone();
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
    fn negate_y(&self, x: &Rational, y: &Rational) -> Rational {
        let a1_term = Rational::from_integer(self.a1.clone()) * x.clone();
        let a3_term = Rational::from_integer(self.a3.clone());
        -(y.clone() + a1_term + a3_term)
    }

    /// Check if a point is on the curve (general Weierstrass form):
    /// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆.
    pub fn is_on_curve(&self, p: &Point) -> bool {
        if p.infinity {
            return true;
        }

        let a1 = Rational::from_integer(self.a1.clone());
        let a2 = Rational::from_integer(self.a2.clone());
        let a3 = Rational::from_integer(self.a3.clone());
        let a4 = Rational::from_integer(self.a4.clone());
        let a6 = Rational::from_integer(self.a6.clone());

        let lhs = &p.y * &p.y + a1 * p.x.clone() * p.y.clone() + a3 * p.y.clone();
        let rhs = p.x.clone() * p.x.clone() * p.x.clone()
            + a2 * p.x.clone() * p.x.clone()
            + a4 * p.x.clone()
            + a6;

        lhs == rhs
    }

    /// The 2-rank of E(Q)[2]: r ∈ {0, 1, 2} with E(Q)[2] ≅ (Z/2)^r.
    ///
    /// The x-coordinates of 2-torsion points are the roots of
    /// 4x³ + b₂x² + 2b₄x + b₆. Under X = 36x + 3b₂ these correspond
    /// exactly to the rational (hence integral) roots of the monic cubic
    /// X³ − 27c₄X − 54c₆.
    pub fn two_torsion_rank(&self) -> i32 {
        let (c4, c6) = self.c_invariants();
        let a = Integer::from(-27) * c4;
        let b = Integer::from(-54) * c6;
        let roots = crate::torsion::integer_cubic_roots(&a, &b);
        match roots.len() {
            0 => 0,
            1 => 1,
            3 => 2,
            n => unreachable!(
                "cubic with {} rational roots (nonsingular curve has distinct roots)",
                n
            ),
        }
    }

    /// Check if a prime is a bad prime (divides the discriminant)
    pub fn is_bad_prime(&self, p: &Integer) -> bool {
        (&self.discriminant % p).is_zero()
    }

    /// Compute a_p for a good prime p (p + 1 - #E(F_p))
    pub fn compute_a_p(&self, p: &Integer) -> i64 {
        let p_val = <Integer as NumericConversion>::to_i64(p).unwrap_or(2);
        (p_val + 1 - self.count_points_mod_p(p_val)) as i64
    }

    /// Count points on the curve modulo p (naive method)
    /// Real implementation would use Schoof's algorithm
    fn count_points_mod_p(&self, p: i64) -> i64 {
        let mut count = 1; // Point at infinity

        let a = <Integer as NumericConversion>::to_i64(&self.a4).unwrap_or(0);
        let b = <Integer as NumericConversion>::to_i64(&self.a6).unwrap_or(0);

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
    pub fn new(x: Rational, y: Rational) -> Self {
        Self {
            x,
            y,
            infinity: false,
        }
    }

    /// Create the point at infinity
    pub fn infinity() -> Self {
        Self {
            x: Rational::zero(),
            y: Rational::zero(),
            infinity: true,
        }
    }

    /// Create a point from integer coordinates
    pub fn from_integers(x: i64, y: i64) -> Self {
        Self {
            x: Rational::from_i64(x),
            y: Rational::from_i64(y),
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
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));
        assert!(!curve.is_singular());
    }

    #[test]
    fn test_point_on_curve() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        // Point (0, 0) should be on y² = x³ - x
        let p = Point::new(Rational::zero(), Rational::zero());
        assert!(curve.is_on_curve(&p));

        // Point (1, 0) should be on y² = x³ - x
        let q = Point::new(Rational::one(), Rational::zero());
        assert!(curve.is_on_curve(&q));
    }

    #[test]
    fn test_point_addition() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let p = Point::new(Rational::zero(), Rational::zero());
        let q = Point::infinity();

        let r = curve.add_points(&p, &q);
        assert_eq!(r, p);
    }

    #[test]
    fn test_point_doubling() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(2), Integer::from(3));

        // Point (-1, 0) is on y² = x³ + 2x + 3
        let p = Point::new(Rational::from_i64(-1), Rational::from_i64(0));

        assert!(curve.is_on_curve(&p));
        let doubled = curve.double_point(&p);
        // Doubling a point where y=0 gives infinity
        assert!(doubled.infinity);
    }

    #[test]
    fn test_scalar_multiplication() {
        // Use curve y² = x³ - x for simplicity
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        // Point (0, 0) is on the curve
        let p = Point::new(Rational::from_i64(0), Rational::from_i64(0));

        assert!(curve.is_on_curve(&p));
        let result = curve.scalar_mul(&Integer::from(2), &p);
        // [2]P for a point of order 2 is infinity
        assert!(result.infinity || curve.is_on_curve(&result));
    }

    #[test]
    fn test_j_invariant() {
        // For y² = x³ + x (curve with CM by Gaussian integers)
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(1), Integer::from(0));

        let j = curve.j_invariant();
        assert!(j.is_some());
        // j-invariant should be 1728 for this curve
    }
}

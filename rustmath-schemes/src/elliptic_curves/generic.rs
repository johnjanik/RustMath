//! Generic Elliptic Curves over Arbitrary Fields
//!
//! This module implements the generic `EllipticCurve_generic` base class from
//! `sage.schemes.elliptic_curves.ell_generic`, providing elliptic curve arithmetic
//! over any field.
//!
//! # Weierstrass Form
//!
//! An elliptic curve in generalized Weierstrass form is given by:
//!
//! ```text
//! y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
//! ```
//!
//! In short Weierstrass form (used when the characteristic is not 2 or 3):
//!
//! ```text
//! y² = x³ + ax + b
//! ```
//!
//! # Point Arithmetic
//!
//! Points on elliptic curves form an abelian group with:
//! - Identity element: Point at infinity (O)
//! - Addition: Geometric chord-and-tangent process
//! - Doubling: Tangent line at a point
//! - Scalar multiplication: Repeated addition using double-and-add
//!
//! # Examples
//!
//! ## Curve over Rationals
//!
//! ```
//! use rustmath_schemes::elliptic_curves::generic::{EllipticCurve, Point};
//! use rustmath_rationals::Rational;
//! use rustmath_integers::Integer;
//!
//! // Create y² = x³ - x over Q
//! let curve = EllipticCurve::<Rational>::short_weierstrass(
//!     Rational::from_integer(-1),
//!     Rational::from_integer(0),
//! );
//!
//! // Create point (0, 0)
//! let p = Point::new(
//!     Rational::from_integer(0),
//!     Rational::from_integer(0),
//! );
//!
//! // Verify it's on the curve
//! assert!(curve.is_on_curve(&p));
//!
//! // Double the point
//! let doubled = curve.double_point(&p).unwrap();
//! ```
//!
//! ## Curve over Finite Field
//!
//! Note: Currently, the `short_weierstrass` constructor doesn't work with
//! `PrimeField` because its `Ring::zero()` and `Ring::one()` methods require
//! parameters (the modulus). Use the full `new()` constructor instead, or
//! use fields like `Rational` that have proper zero/one implementations.
//!
//! ```ignore
//! use rustmath_schemes::elliptic_curves::generic::{EllipticCurve, Point};
//! use rustmath_finitefields::PrimeField;
//! use rustmath_integers::Integer;
//!
//! // Work over GF(7)
//! let p7 = Integer::from(7);
//!
//! // y² = x³ + 2x + 3 over GF(7)
//! // Note: Would need to provide zeros manually
//! let zero = PrimeField::new(Integer::from(0), p7.clone()).unwrap();
//! let a = PrimeField::new(Integer::from(2), p7.clone()).unwrap();
//! let b = PrimeField::new(Integer::from(3), p7.clone()).unwrap();
//! let curve = EllipticCurve::new(zero.clone(), zero.clone(), zero, a, b);
//! ```

use rustmath_core::{Field, MathError, Result};
use num_traits::{Zero, One};
use std::fmt::{self, Debug, Display};

/// A point on an elliptic curve over a field F
///
/// Points are either affine points (x, y) or the point at infinity.
#[derive(Clone, Debug)]
pub struct Point<F: Field> {
    /// x-coordinate (None for point at infinity)
    x: Option<F>,
    /// y-coordinate (None for point at infinity)
    y: Option<F>,
}

impl<F: Field> Point<F> {
    /// Create a new affine point (x, y)
    pub fn new(x: F, y: F) -> Self {
        Point {
            x: Some(x),
            y: Some(y),
        }
    }

    /// Create the point at infinity (identity element)
    pub fn infinity() -> Self {
        Point { x: None, y: None }
    }

    /// Check if this is the point at infinity
    pub fn is_infinity(&self) -> bool {
        self.x.is_none() && self.y.is_none()
    }

    /// Get the x-coordinate (returns None for point at infinity)
    pub fn x(&self) -> Option<&F> {
        self.x.as_ref()
    }

    /// Get the y-coordinate (returns None for point at infinity)
    pub fn y(&self) -> Option<&F> {
        self.y.as_ref()
    }

    /// Take ownership of coordinates
    pub fn into_coords(self) -> (Option<F>, Option<F>) {
        (self.x, self.y)
    }
}

impl<F: Field> PartialEq for Point<F> {
    fn eq(&self, other: &Self) -> bool {
        match (&self.x, &self.y, &other.x, &other.y) {
            (None, None, None, None) => true,
            (Some(x1), Some(y1), Some(x2), Some(y2)) => x1 == x2 && y1 == y2,
            _ => false,
        }
    }
}

impl<F: Field> Eq for Point<F> {}

impl<F: Field + Display> Display for Point<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "O")
        } else {
            write!(
                f,
                "({}, {})",
                self.x.as_ref().unwrap(),
                self.y.as_ref().unwrap()
            )
        }
    }
}

/// Generic elliptic curve over a field F
///
/// Represents an elliptic curve in generalized Weierstrass form:
/// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
///
/// For short Weierstrass form (y² = x³ + ax + b), use a₁ = a₂ = a₃ = 0.
#[derive(Clone, Debug)]
pub struct EllipticCurve<F: Field> {
    /// Coefficient a₁ in generalized Weierstrass equation
    a1: F,
    /// Coefficient a₂ in generalized Weierstrass equation
    a2: F,
    /// Coefficient a₃ in generalized Weierstrass equation
    a3: F,
    /// Coefficient a₄ in generalized Weierstrass equation (a in short form)
    a4: F,
    /// Coefficient a₆ in generalized Weierstrass equation (b in short form)
    a6: F,
    /// Cached discriminant
    discriminant: Option<F>,
    /// Cached b-invariants
    b2: Option<F>,
    b4: Option<F>,
    b6: Option<F>,
    b8: Option<F>,
}

impl<F: Field + Zero + One> EllipticCurve<F> {
    /// Create a new elliptic curve from Weierstrass coefficients
    ///
    /// Creates the curve: y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
    ///
    /// # Arguments
    ///
    /// * `a1`, `a2`, `a3`, `a4`, `a6` - Weierstrass coefficients
    ///
    /// # Returns
    ///
    /// The elliptic curve with the given coefficients.
    pub fn new(a1: F, a2: F, a3: F, a4: F, a6: F) -> Self {
        EllipticCurve {
            a1,
            a2,
            a3,
            a4,
            a6,
            discriminant: None,
            b2: None,
            b4: None,
            b6: None,
            b8: None,
        }
    }

    /// Create an elliptic curve in short Weierstrass form: y² = x³ + ax + b
    ///
    /// This is the most common form, used when the characteristic is not 2 or 3.
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient of x
    /// * `b` - Constant term
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_schemes::elliptic_curves::generic::EllipticCurve;
    /// use rustmath_rationals::Rational;
    ///
    /// // y² = x³ - x (congruent number curve with n=1)
    /// let curve = EllipticCurve::short_weierstrass(
    ///     Rational::from_integer(-1),
    ///     Rational::from_integer(0),
    /// );
    /// ```
    pub fn short_weierstrass(a: F, b: F) -> Self {
        Self::new(F::zero(), F::zero(), F::zero(), a, b)
    }

    /// Get coefficient a₁
    pub fn a1(&self) -> &F {
        &self.a1
    }

    /// Get coefficient a₂
    pub fn a2(&self) -> &F {
        &self.a2
    }

    /// Get coefficient a₃
    pub fn a3(&self) -> &F {
        &self.a3
    }

    /// Get coefficient a₄ (a in short form)
    pub fn a4(&self) -> &F {
        &self.a4
    }

    /// Get coefficient a₆ (b in short form)
    pub fn a6(&self) -> &F {
        &self.a6
    }

    /// Check if this is in short Weierstrass form (y² = x³ + ax + b)
    pub fn is_short_weierstrass(&self) -> bool {
        self.a1.is_zero() && self.a2.is_zero() && self.a3.is_zero()
    }

    /// Compute the b₂ invariant: b₂ = a₁² + 4a₂
    pub fn b2(&mut self) -> F {
        if let Some(ref b2) = self.b2 {
            return b2.clone();
        }

        let four = F::one() + F::one() + F::one() + F::one();
        let b2 = self.a1.clone() * self.a1.clone() + four * self.a2.clone();
        self.b2 = Some(b2.clone());
        b2
    }

    /// Compute the b₄ invariant: b₄ = 2a₄ + a₁a₃
    pub fn b4(&mut self) -> F {
        if let Some(ref b4) = self.b4 {
            return b4.clone();
        }

        let two = F::one() + F::one();
        let b4 = two * self.a4.clone() + self.a1.clone() * self.a3.clone();
        self.b4 = Some(b4.clone());
        b4
    }

    /// Compute the b₆ invariant: b₆ = a₃² + 4a₆
    pub fn b6(&mut self) -> F {
        if let Some(ref b6) = self.b6 {
            return b6.clone();
        }

        let four = F::one() + F::one() + F::one() + F::one();
        let b6 = self.a3.clone() * self.a3.clone() + four * self.a6.clone();
        self.b6 = Some(b6.clone());
        b6
    }

    /// Compute the b₈ invariant
    ///
    /// b₈ = a₁²a₆ + 4a₂a₆ - a₁a₃a₄ + a₂a₃² - a₄²
    pub fn b8(&mut self) -> F {
        if let Some(ref b8) = self.b8 {
            return b8.clone();
        }

        let four = F::one() + F::one() + F::one() + F::one();

        let term1 = self.a1.clone() * self.a1.clone() * self.a6.clone();
        let term2 = four * self.a2.clone() * self.a6.clone();
        let term3 = self.a1.clone() * self.a3.clone() * self.a4.clone();
        let term4 = self.a2.clone() * self.a3.clone() * self.a3.clone();
        let term5 = self.a4.clone() * self.a4.clone();

        let b8 = term1 + term2 - term3 + term4 - term5;
        self.b8 = Some(b8.clone());
        b8
    }

    /// Compute the discriminant Δ
    ///
    /// Δ = -b₂²b₈ - 8b₄³ - 27b₆² + 9b₂b₄b₆
    ///
    /// The curve is singular if and only if Δ = 0.
    pub fn discriminant(&mut self) -> F {
        if let Some(ref disc) = self.discriminant {
            return disc.clone();
        }

        let b2 = self.b2();
        let b4 = self.b4();
        let b6 = self.b6();
        let b8 = self.b8();

        let eight = F::one() + F::one() + F::one() + F::one()
            + F::one() + F::one() + F::one() + F::one();
        let nine = eight.clone() + F::one();
        let twenty_seven = nine.clone() + nine.clone() + nine.clone();

        let term1 = -(b2.clone() * b2.clone() * b8);
        let term2 = -(eight * b4.clone() * b4.clone() * b4.clone());
        let term3 = -(twenty_seven * b6.clone() * b6.clone());
        let term4 = nine * b2 * b4 * b6;

        let disc = term1 + term2 + term3 + term4;
        self.discriminant = Some(disc.clone());
        disc
    }

    /// Check if the curve is singular (discriminant is zero)
    pub fn is_singular(&mut self) -> bool {
        self.discriminant().is_zero()
    }

    /// Compute the c₄ invariant: c₄ = b₂² - 24b₄
    pub fn c4(&mut self) -> F {
        let b2 = self.b2();
        let b4 = self.b4();

        let twenty_four = {
            let four = F::one() + F::one() + F::one() + F::one();
            let six = four.clone() + F::one() + F::one();
            four * six
        };

        b2.clone() * b2 - twenty_four * b4
    }

    /// Compute the j-invariant
    ///
    /// j = c₄³ / Δ
    ///
    /// The j-invariant is a fundamental isomorphism invariant of elliptic curves.
    /// Two curves are isomorphic over an algebraically closed field if and only if
    /// they have the same j-invariant.
    pub fn j_invariant(&mut self) -> Result<F> {
        let disc = self.discriminant();
        if disc.is_zero() {
            return Err(MathError::InvalidArgument(
                "j-invariant undefined for singular curve".to_string(),
            ));
        }

        let c4 = self.c4();
        let c4_cubed = c4.clone() * c4.clone() * c4;

        c4_cubed.divide(&disc)
    }

    /// Check if a point is on the curve
    ///
    /// Verifies that the point satisfies the Weierstrass equation:
    /// y² + a₁xy + a₃y = x³ + a₂x² + a₄x + a₆
    pub fn is_on_curve(&self, point: &Point<F>) -> bool {
        if point.is_infinity() {
            return true;
        }

        let x = point.x().unwrap();
        let y = point.y().unwrap();

        // LHS: y² + a₁xy + a₃y
        let lhs = y.clone() * y.clone()
            + self.a1.clone() * x.clone() * y.clone()
            + self.a3.clone() * y.clone();

        // RHS: x³ + a₂x² + a₄x + a₆
        let rhs = x.clone() * x.clone() * x.clone()
            + self.a2.clone() * x.clone() * x.clone()
            + self.a4.clone() * x.clone()
            + self.a6.clone();

        lhs == rhs
    }

    /// Negate a point: compute -P
    ///
    /// For the point (x, y), the negative is (x, -y - a₁x - a₃) in general form,
    /// or simply (x, -y) in short Weierstrass form.
    pub fn negate_point(&self, point: &Point<F>) -> Point<F> {
        if point.is_infinity() {
            return Point::infinity();
        }

        let x = point.x().unwrap();
        let y = point.y().unwrap();

        // In general form: -(y + a₁x + a₃)
        let neg_y = -(y.clone() + self.a1.clone() * x.clone() + self.a3.clone());

        Point::new(x.clone(), neg_y)
    }

    /// Add two points on the curve: compute P + Q
    ///
    /// Implements the chord-and-tangent group law.
    ///
    /// # Special cases
    ///
    /// - P + O = P
    /// - O + Q = Q
    /// - P + (-P) = O
    /// - P + P = 2P (uses doubling)
    ///
    /// # Algorithm
    ///
    /// For distinct points P and Q:
    /// 1. Compute the slope λ of the line through P and Q
    /// 2. Find where the line intersects the curve again
    /// 3. Negate the y-coordinate to get P + Q
    pub fn add_points(&self, p: &Point<F>, q: &Point<F>) -> Result<Point<F>> {
        // Handle point at infinity cases
        if p.is_infinity() {
            return Ok(q.clone());
        }
        if q.is_infinity() {
            return Ok(p.clone());
        }

        let x1 = p.x().unwrap();
        let y1 = p.y().unwrap();
        let x2 = q.x().unwrap();
        let y2 = q.y().unwrap();

        // Check if points are negatives of each other
        if x1 == x2 {
            let neg_q = self.negate_point(q);
            if p == &neg_q {
                return Ok(Point::infinity());
            }
        }

        // If points are the same, use doubling
        if p == q {
            return self.double_point(p);
        }

        // For short Weierstrass form: y² = x³ + ax + b
        if self.is_short_weierstrass() {
            // Slope: λ = (y₂ - y₁) / (x₂ - x₁)
            let dy = y2.clone() - y1.clone();
            let dx = x2.clone() - x1.clone();
            let lambda = dy.divide(&dx)?;

            // x₃ = λ² - x₁ - x₂
            let x3 = lambda.clone() * lambda.clone() - x1.clone() - x2.clone();

            // y₃ = λ(x₁ - x₃) - y₁
            let y3 = lambda * (x1.clone() - x3.clone()) - y1.clone();

            return Ok(Point::new(x3, y3));
        }

        // General Weierstrass form
        // λ = (y₂ - y₁) / (x₂ - x₁)
        let dy = y2.clone() - y1.clone();
        let dx = x2.clone() - x1.clone();
        let lambda = dy.divide(&dx)?;

        // ν = (y₁x₂ - y₂x₁) / (x₂ - x₁)
        let nu_num = y1.clone() * x2.clone() - y2.clone() * x1.clone();
        let nu = nu_num.divide(&dx)?;

        // x₃ = λ² + a₁λ - a₂ - x₁ - x₂
        let x3 = lambda.clone() * lambda.clone()
            + self.a1.clone() * lambda.clone()
            - self.a2.clone()
            - x1.clone()
            - x2.clone();

        // y₃ = -(λ + a₁)x₃ - ν - a₃
        let y3 = -(lambda.clone() + self.a1.clone()) * x3.clone() - nu - self.a3.clone();

        Ok(Point::new(x3, y3))
    }

    /// Double a point: compute 2P
    ///
    /// Uses the tangent line at P to find where it intersects the curve again.
    ///
    /// # Algorithm
    ///
    /// For point P = (x, y):
    /// 1. Compute the slope λ of the tangent line at P
    /// 2. Find where the tangent intersects the curve
    /// 3. Negate to get 2P
    pub fn double_point(&self, point: &Point<F>) -> Result<Point<F>> {
        if point.is_infinity() {
            return Ok(Point::infinity());
        }

        let x = point.x().unwrap();
        let y = point.y().unwrap();

        // For short Weierstrass form: y² = x³ + ax + b
        if self.is_short_weierstrass() {
            let two = F::one() + F::one();
            let three = two.clone() + F::one();

            // Check if y = 0 (tangent is vertical)
            if y.is_zero() {
                return Ok(Point::infinity());
            }

            // Slope: λ = (3x² + a) / (2y)
            let numerator = three * x.clone() * x.clone() + self.a4.clone();
            let denominator = two.clone() * y.clone();
            let lambda = numerator.divide(&denominator)?;

            // x₃ = λ² - 2x
            let x3 = lambda.clone() * lambda.clone() - two * x.clone();

            // y₃ = λ(x - x₃) - y
            let y3 = lambda * (x.clone() - x3.clone()) - y.clone();

            return Ok(Point::new(x3, y3));
        }

        // General Weierstrass form
        let two = F::one() + F::one();
        let three = two.clone() + F::one();

        // Check if tangent is vertical: 2y + a₁x + a₃ = 0
        let denom_check = two.clone() * y.clone() + self.a1.clone() * x.clone() + self.a3.clone();
        if denom_check.is_zero() {
            return Ok(Point::infinity());
        }

        // λ = (3x² + 2a₂x + a₄ - a₁y) / (2y + a₁x + a₃)
        let numerator = three * x.clone() * x.clone()
            + two.clone() * self.a2.clone() * x.clone()
            + self.a4.clone()
            - self.a1.clone() * y.clone();
        let lambda = numerator.divide(&denom_check)?;

        // ν = (-x³ + a₄x + 2a₆ - a₃y) / (2y + a₁x + a₃)
        let nu_num = -(x.clone() * x.clone() * x.clone())
            + self.a4.clone() * x.clone()
            + two.clone() * self.a6.clone()
            - self.a3.clone() * y.clone();
        let nu = nu_num.divide(&denom_check)?;

        // x₃ = λ² + a₁λ - a₂ - 2x
        let x3 = lambda.clone() * lambda.clone()
            + self.a1.clone() * lambda.clone()
            - self.a2.clone()
            - two * x.clone();

        // y₃ = -(λ + a₁)x₃ - ν - a₃
        let y3 = -(lambda + self.a1.clone()) * x3.clone() - nu - self.a3.clone();

        Ok(Point::new(x3, y3))
    }

    /// Scalar multiplication: compute [n]P
    ///
    /// Computes n times P using the double-and-add algorithm.
    ///
    /// # Arguments
    ///
    /// * `n` - The scalar (as i64)
    /// * `point` - The point to multiply
    ///
    /// # Algorithm
    ///
    /// Uses binary representation of n:
    /// - For each bit, double the accumulated result
    /// - If bit is 1, add P to the result
    ///
    /// # Complexity
    ///
    /// O(log n) point operations
    pub fn scalar_mul(&self, n: i64, point: &Point<F>) -> Result<Point<F>> {
        // Handle special cases
        if n == 0 || point.is_infinity() {
            return Ok(Point::infinity());
        }

        if n == 1 {
            return Ok(point.clone());
        }

        if n < 0 {
            let neg_point = self.negate_point(point);
            return self.scalar_mul(-n, &neg_point);
        }

        // Double-and-add algorithm
        let mut result = Point::infinity();
        let mut addend = point.clone();
        let mut k = n;

        while k > 0 {
            if k % 2 == 1 {
                result = self.add_points(&result, &addend)?;
            }
            addend = self.double_point(&addend)?;
            k /= 2;
        }

        Ok(result)
    }

    /// Compute the order of a point (if finite)
    ///
    /// Returns the smallest positive integer n such that [n]P = O.
    /// Returns None if the point has infinite order or computation fails.
    ///
    /// # Warning
    ///
    /// This is a naive implementation that may be very slow for points of large order.
    /// In practice, better algorithms (e.g., using torsion structure) should be used.
    pub fn point_order(&self, point: &Point<F>, max_order: i64) -> Option<i64> {
        if point.is_infinity() {
            return Some(1);
        }

        let mut current = point.clone();
        for i in 1..=max_order {
            if current.is_infinity() {
                return Some(i);
            }
            current = self.add_points(&current, point).ok()?;
        }

        None // Order exceeds max_order or is infinite
    }
}

impl<F: Field + Zero + One + Display> Display for EllipticCurve<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_short_weierstrass() {
            write!(f, "y² = x³ + ({})x + ({})", self.a4, self.a6)
        } else {
            write!(
                f,
                "y² + ({})xy + ({})y = x³ + ({})x² + ({})x + ({})",
                self.a1, self.a3, self.a2, self.a4, self.a6
            )
        }
    }
}

// Tests are in the tests.rs module

//! Elliptic curve arithmetic over rationals
//!
//! This module implements elliptic curves in Weierstrass form: y^2 = x^3 + ax + b

use rustmath_rationals::Rational;
use std::fmt;

/// A point on an elliptic curve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipticCurvePoint<T> {
    /// The point at infinity (identity element)
    Infinity,
    /// An affine point (x, y)
    Affine { x: T, y: T },
}

impl<T: fmt::Display> fmt::Display for EllipticCurvePoint<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EllipticCurvePoint::Infinity => write!(f, "O (point at infinity)"),
            EllipticCurvePoint::Affine { x, y } => write!(f, "({}, {})", x, y),
        }
    }
}

/// An elliptic curve in Weierstrass form: y^2 = x^3 + ax + b
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EllipticCurve<T> {
    /// Coefficient a
    pub a: T,
    /// Coefficient b
    pub b: T,
}

impl EllipticCurve<Rational> {
    /// Create a new elliptic curve over the rationals
    ///
    /// Returns an error if the curve is singular (discriminant = 0)
    pub fn new(a: Rational, b: Rational) -> Result<Self, String> {
        // Check discriminant: Δ = -16(4a³ + 27b²)
        let a_cubed = a.clone() * a.clone() * a.clone();
        let b_squared = b.clone() * b.clone();

        let discriminant = (Rational::new(4, 1).unwrap() * a_cubed + Rational::new(27, 1).unwrap() * b_squared)
                         * Rational::new(-16, 1).unwrap();

        if discriminant == Rational::new(0, 1).unwrap() {
            return Err("Curve is singular (discriminant is zero)".to_string());
        }

        Ok(EllipticCurve { a, b })
    }

    /// Check if a point is on the curve
    pub fn contains_point(&self, point: &EllipticCurvePoint<Rational>) -> bool {
        match point {
            EllipticCurvePoint::Infinity => true,
            EllipticCurvePoint::Affine { x, y } => {
                // Check if y^2 = x^3 + ax + b
                let lhs = y.clone() * y.clone();
                let rhs = x.clone() * x.clone() * x.clone()
                        + self.a.clone() * x.clone()
                        + self.b.clone();
                lhs == rhs
            }
        }
    }

    /// Add two points on the elliptic curve
    pub fn add(&self, p: &EllipticCurvePoint<Rational>, q: &EllipticCurvePoint<Rational>)
        -> Result<EllipticCurvePoint<Rational>, String> {

        // Check that points are on the curve
        if !self.contains_point(p) {
            return Err("Point p is not on the curve".to_string());
        }
        if !self.contains_point(q) {
            return Err("Point q is not on the curve".to_string());
        }

        match (p, q) {
            // O + Q = Q
            (EllipticCurvePoint::Infinity, _) => Ok(q.clone()),
            // P + O = P
            (_, EllipticCurvePoint::Infinity) => Ok(p.clone()),

            (EllipticCurvePoint::Affine { x: x1, y: y1 },
             EllipticCurvePoint::Affine { x: x2, y: y2 }) => {

                // Check if points are inverses: P + (-P) = O
                if x1 == x2 && y1 != y2 {
                    return Ok(EllipticCurvePoint::Infinity);
                }

                // Calculate slope
                let slope = if x1 == x2 && y1 == y2 {
                    // Point doubling: slope = (3x₁² + a) / (2y₁)
                    if y1 == &Rational::new(0, 1).unwrap() {
                        return Ok(EllipticCurvePoint::Infinity);
                    }
                    let numerator = Rational::new(3, 1).unwrap() * x1.clone() * x1.clone() + self.a.clone();
                    let denominator = Rational::new(2, 1).unwrap() * y1.clone();
                    numerator / denominator
                } else {
                    // Point addition: slope = (y₂ - y₁) / (x₂ - x₁)
                    let numerator = y2.clone() - y1.clone();
                    let denominator = x2.clone() - x1.clone();

                    if denominator == Rational::new(0, 1).unwrap() {
                        return Ok(EllipticCurvePoint::Infinity);
                    }
                    numerator / denominator
                };

                // Calculate x₃ = slope² - x₁ - x₂
                let x3 = slope.clone() * slope.clone() - x1.clone() - x2.clone();

                // Calculate y₃ = slope(x₁ - x₃) - y₁
                let y3 = slope * (x1.clone() - x3.clone()) - y1.clone();

                Ok(EllipticCurvePoint::Affine { x: x3, y: y3 })
            }
        }
    }

    /// Double a point on the curve (more efficient than add(p, p))
    pub fn double(&self, p: &EllipticCurvePoint<Rational>)
        -> Result<EllipticCurvePoint<Rational>, String> {
        self.add(p, p)
    }

    /// Negate a point on the curve
    pub fn negate(&self, p: &EllipticCurvePoint<Rational>) -> EllipticCurvePoint<Rational> {
        match p {
            EllipticCurvePoint::Infinity => EllipticCurvePoint::Infinity,
            EllipticCurvePoint::Affine { x, y } => {
                EllipticCurvePoint::Affine {
                    x: x.clone(),
                    y: -y.clone()
                }
            }
        }
    }

    /// Scalar multiplication: compute n * P using double-and-add
    pub fn scalar_mul(&self, n: i64, p: &EllipticCurvePoint<Rational>)
        -> Result<EllipticCurvePoint<Rational>, String> {

        if n == 0 {
            return Ok(EllipticCurvePoint::Infinity);
        }

        if n < 0 {
            let neg_p = self.negate(p);
            return self.scalar_mul(-n, &neg_p);
        }

        // Double-and-add algorithm
        let mut result = EllipticCurvePoint::Infinity;
        let mut temp = p.clone();
        let mut k = n;

        while k > 0 {
            if k & 1 == 1 {
                result = self.add(&result, &temp)?;
            }
            temp = self.double(&temp)?;
            k >>= 1;
        }

        Ok(result)
    }

    /// Get the discriminant of the curve
    pub fn discriminant(&self) -> Rational {
        let a_cubed = self.a.clone() * self.a.clone() * self.a.clone();
        let b_squared = self.b.clone() * self.b.clone();

        (Rational::new(4, 1).unwrap() * a_cubed + Rational::new(27, 1).unwrap() * b_squared) * Rational::new(-16, 1).unwrap()
    }

    /// Get the j-invariant of the curve
    ///
    /// The j-invariant is given by: j = c₄³ / Δ
    /// where c₄ = -48a and Δ is the discriminant
    pub fn j_invariant(&self) -> Rational {
        let discriminant = self.discriminant();

        if discriminant == Rational::new(0, 1).unwrap() {
            panic!("Cannot compute j-invariant for singular curve");
        }

        // c₄ = -48a for Weierstrass form y² = x³ + ax + b
        let c4 = Rational::new(-48, 1).unwrap() * self.a.clone();
        let c4_cubed = c4.clone() * c4.clone() * c4.clone();

        // j = c₄³ / Δ
        c4_cubed / discriminant
    }

    /// Compute the order of a point (smallest n > 0 such that nP = O)
    ///
    /// Returns None if the point has infinite order or if the order exceeds max_order
    pub fn point_order(&self, p: &EllipticCurvePoint<Rational>, max_order: i64)
        -> Result<Option<i64>, String> {

        if !self.contains_point(p) {
            return Err("Point is not on the curve".to_string());
        }

        if p == &EllipticCurvePoint::Infinity {
            return Ok(Some(1)); // Order of identity is 1
        }

        let mut current = p.clone();
        for n in 1..=max_order {
            if current == EllipticCurvePoint::Infinity {
                return Ok(Some(n));
            }
            current = self.add(&current, p)?;
        }

        Ok(None) // Order exceeds max_order or point has infinite order
    }

    /// Check if a point is a torsion point (has finite order)
    ///
    /// Checks up to max_order. Returns true if the point has finite order <= max_order.
    pub fn is_torsion(&self, p: &EllipticCurvePoint<Rational>, max_order: i64)
        -> Result<bool, String> {
        let order = self.point_order(p, max_order)?;
        Ok(order.is_some())
    }

    /// Find all torsion points of a specific order
    ///
    /// This is a naive implementation that only works for small orders.
    /// For n-torsion, we need points P such that nP = O but kP ≠ O for 0 < k < n.
    ///
    /// Note: This returns only rational torsion points found by checking multiples.
    pub fn torsion_points_of_order(&self, order: i64, max_search: i64)
        -> Result<Vec<EllipticCurvePoint<Rational>>, String> {

        if order <= 0 {
            return Err("Order must be positive".to_string());
        }

        let mut torsion_points = Vec::new();

        // Always include the identity for order 1
        if order == 1 {
            torsion_points.push(EllipticCurvePoint::Infinity);
            return Ok(torsion_points);
        }

        // Search for rational points by trying x values
        // For each rational x, solve y^2 = x^3 + ax + b
        for num in -max_search..=max_search {
            for denom in 1..=max_search {
                let x = Rational::new(num, denom).unwrap();

                // Compute x^3 + ax + b
                let rhs = x.clone() * x.clone() * x.clone()
                        + self.a.clone() * x.clone()
                        + self.b.clone();

                // Check if rhs is a perfect square (approximate check for rationals)
                // For rational torsion points, we need y to be rational
                // This is a simplified check - try y values
                for y_num in -max_search..=max_search {
                    for y_denom in 1..=max_search {
                        let y = Rational::new(y_num, y_denom).unwrap();

                        if y.clone() * y.clone() == rhs {
                            let point = EllipticCurvePoint::Affine {
                                x: x.clone(),
                                y: y.clone(),
                            };

                            if let Ok(Some(pt_order)) = self.point_order(&point, order) {
                                if pt_order == order && !torsion_points.contains(&point) {
                                    torsion_points.push(point);
                                }
                            }

                            // Also check the negation
                            let neg_point = EllipticCurvePoint::Affine {
                                x: x.clone(),
                                y: -y.clone(),
                            };

                            if let Ok(Some(pt_order)) = self.point_order(&neg_point, order) {
                                if pt_order == order && !torsion_points.contains(&neg_point) {
                                    torsion_points.push(neg_point);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(torsion_points)
    }

    /// Get all 2-torsion points (points of order dividing 2)
    ///
    /// These are points (x, 0) where x^3 + ax + b = 0, plus the identity.
    /// Over the rationals, there can be 0, 1, or 3 such rational points.
    pub fn two_torsion_points(&self) -> Vec<EllipticCurvePoint<Rational>> {
        let mut points = vec![EllipticCurvePoint::Infinity];

        // Find roots of x^3 + ax + b = 0
        // This is a cubic equation, which is complex to solve in general
        // For now, try small integer values
        for x_num in -100..=100 {
            let x = Rational::new(x_num, 1).unwrap();
            let value = x.clone() * x.clone() * x.clone()
                      + self.a.clone() * x.clone()
                      + self.b.clone();

            if value == Rational::new(0, 1).unwrap() {
                points.push(EllipticCurvePoint::Affine {
                    x,
                    y: Rational::new(0, 1).unwrap(),
                });
            }
        }

        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elliptic_curve_creation() {
        // y^2 = x^3 - x (discriminant = 64)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        );
        assert!(curve.is_ok());

        // Singular curve: y^2 = x^3 (discriminant = 0)
        let singular = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        );
        assert!(singular.is_err());
    }

    #[test]
    fn test_point_on_curve() {
        // y^2 = x^3 - x
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // (0, 0) is on the curve
        let p1 = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };
        assert!(curve.contains_point(&p1));

        // (1, 0) is on the curve: 0 = 1 - 1 + 0 = 0
        let p2 = EllipticCurvePoint::Affine {
            x: Rational::new(1, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };
        assert!(curve.contains_point(&p2));

        // (1, 1) is not on the curve: 1 ≠ 1 - 1 + 0 = 0
        let p3 = EllipticCurvePoint::Affine {
            x: Rational::new(1, 1).unwrap(),
            y: Rational::new(1, 1).unwrap(),
        };
        assert!(!curve.contains_point(&p3));
    }

    #[test]
    fn test_point_addition_identity() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        // P + O = P
        let result = curve.add(&p, &EllipticCurvePoint::Infinity).unwrap();
        assert_eq!(result, p);

        // O + P = P
        let result = curve.add(&EllipticCurvePoint::Infinity, &p).unwrap();
        assert_eq!(result, p);
    }

    #[test]
    fn test_point_addition_inverse() {
        let curve = EllipticCurve::new(
            Rational::new(2, 1).unwrap(),
            Rational::new(3, 1).unwrap()
        ).unwrap();

        // Point and its inverse
        // For y^2 = x^3 + 2x + 3, point (-1, 0) is on the curve
        let p = EllipticCurvePoint::Affine {
            x: Rational::new(-1, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        let neg_p = curve.negate(&p);

        // P + (-P) = O (in this case p = -p since y = 0)
        let result = curve.add(&p, &neg_p).unwrap();
        assert_eq!(result, EllipticCurvePoint::Infinity);
    }

    #[test]
    fn test_scalar_multiplication() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        // 0 * P = O
        let result = curve.scalar_mul(0, &p).unwrap();
        assert_eq!(result, EllipticCurvePoint::Infinity);

        // 1 * P = P
        let result = curve.scalar_mul(1, &p).unwrap();
        assert_eq!(result, p);

        // 2 * P = P + P
        let double = curve.double(&p).unwrap();
        let scalar_2 = curve.scalar_mul(2, &p).unwrap();
        assert_eq!(double, scalar_2);
    }

    #[test]
    fn test_discriminant() {
        // y^2 = x^3 - x has discriminant 64
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let disc = curve.discriminant();
        assert_eq!(disc, Rational::new(64, 1).unwrap());
    }

    #[test]
    fn test_point_order_identity() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let order = curve.point_order(&EllipticCurvePoint::Infinity, 100).unwrap();
        assert_eq!(order, Some(1));
    }

    #[test]
    fn test_point_order_two_torsion() {
        // y^2 = x^3 - x has 2-torsion points at (0,0), (1,0), (-1,0)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        let order = curve.point_order(&p, 10).unwrap();
        assert_eq!(order, Some(2));
    }

    #[test]
    fn test_is_torsion() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        // (0, 0) is a 2-torsion point
        let p = EllipticCurvePoint::Affine {
            x: Rational::new(0, 1).unwrap(),
            y: Rational::new(0, 1).unwrap(),
        };

        assert!(curve.is_torsion(&p, 10).unwrap());
    }

    #[test]
    fn test_two_torsion_points() {
        // y^2 = x^3 - x has three 2-torsion points: (0,0), (1,0), (-1,0)
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let torsion = curve.two_torsion_points();

        // Should find 4 points total: O and the three points with y=0
        assert_eq!(torsion.len(), 4);
        assert!(torsion.contains(&EllipticCurvePoint::Infinity));
    }

    #[test]
    fn test_torsion_points_of_order_one() {
        let curve = EllipticCurve::new(
            Rational::new(-1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let order_1 = curve.torsion_points_of_order(1, 5).unwrap();
        assert_eq!(order_1.len(), 1);
        assert_eq!(order_1[0], EllipticCurvePoint::Infinity);
    }

    #[test]
    fn test_j_invariant() {
        // y^2 = x^3 + x has j-invariant 1728 (curve with CM by i)
        let curve = EllipticCurve::new(
            Rational::new(1, 1).unwrap(),
            Rational::new(0, 1).unwrap()
        ).unwrap();

        let j = curve.j_invariant();
        assert_eq!(j, Rational::new(1728, 1).unwrap());
    }

    #[test]
    fn test_j_invariant_other() {
        // y^2 = x^3 + 1 has j-invariant 0 (curve with CM by ω)
        let curve = EllipticCurve::new(
            Rational::new(0, 1).unwrap(),
            Rational::new(1, 1).unwrap()
        ).unwrap();

        let j = curve.j_invariant();
        assert_eq!(j, Rational::new(0, 1).unwrap());
    }
}

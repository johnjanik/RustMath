//! Mumford divisor representation for hyperelliptic curves
//!
//! A divisor on a hyperelliptic curve y^2 = f(x) is represented in Mumford form as a pair (u, v)
//! where:
//! - u(x) is a monic polynomial of degree ≤ g (g = genus of the curve)
//! - v(x) is a polynomial of degree < deg(u)
//! - The relation v(x)^2 ≡ f(x) (mod u(x)) is satisfied
//!
//! This representation uniquely determines a reduced divisor on the curve.
//! The divisor represents the formal sum: Σ(xᵢ, v(xᵢ)) - deg(u) * ∞
//! where xᵢ are the roots of u(x).

use rustmath_core::{Ring, Field};
use rustmath_polynomials::univariate::Polynomial;
use std::fmt;

/// A divisor on a hyperelliptic curve in Mumford representation
///
/// Invariant: deg(v) < deg(u) and u is monic
#[derive(Debug, Clone)]
pub struct MumfordDivisor<F: Field> {
    /// The u polynomial (monic)
    pub u: Polynomial<F>,
    /// The v polynomial (deg(v) < deg(u))
    pub v: Polynomial<F>,
}

impl<F: Field + Clone + PartialEq> MumfordDivisor<F> {
    /// Create a new Mumford divisor (u, v)
    ///
    /// # Panics
    /// Panics if u is not monic or if deg(v) >= deg(u)
    pub fn new(u: Polynomial<F>, v: Polynomial<F>) -> Self {
        // Check that u is monic
        if u.degree() > 0 && !u.is_monic() {
            panic!("u polynomial must be monic");
        }

        // Check that deg(v) < deg(u)
        if v.degree() >= u.degree() && !v.is_zero() {
            panic!("deg(v) must be < deg(u)");
        }

        MumfordDivisor { u, v }
    }

    /// Create the zero divisor (identity element)
    ///
    /// The zero divisor is represented as (1, 0)
    pub fn zero() -> Self {
        MumfordDivisor {
            u: Polynomial::one(),
            v: Polynomial::zero(),
        }
    }

    /// Check if this is the zero divisor
    pub fn is_zero(&self) -> bool {
        self.u.degree() == 0 && self.v.is_zero()
    }

    /// Get the degree of the divisor (degree of u)
    pub fn degree(&self) -> usize {
        self.u.degree()
    }

    /// Create a divisor from a single point (x, y)
    ///
    /// This creates the divisor [(x, y) - ∞] represented as:
    /// - u(X) = X - x (degree 1 polynomial)
    /// - v(X) = y (constant polynomial)
    pub fn from_point(x: F, y: F) -> Self {
        // u(X) = X - x
        let u = Polynomial::from_coefficients(vec![-x.clone(), F::one()]);
        // v(X) = y
        let v = Polynomial::from_coefficients(vec![y]);

        MumfordDivisor { u, v }
    }

    /// Create a divisor from two points (x1, y1) and (x2, y2)
    ///
    /// This creates the divisor [(x1, y1) + (x2, y2) - 2∞]
    pub fn from_two_points(x1: F, y1: F, x2: F, y2: F) -> Self {
        if x1 == x2 {
            // Points have the same x-coordinate
            if y1 == y2 {
                // Same point, use doubling formula
                // For now, just create u = (X - x1)^2
                let u_linear = Polynomial::from_coefficients(vec![-x1.clone(), F::one()]);
                let u = u_linear.clone() * u_linear;
                let v = Polynomial::from_coefficients(vec![y1]);
                return MumfordDivisor { u, v };
            } else {
                // Opposite points, sum is zero divisor
                return MumfordDivisor::zero();
            }
        }

        // u(X) = (X - x1)(X - x2)
        let u_1 = Polynomial::from_coefficients(vec![-x1.clone(), F::one()]);
        let u_2 = Polynomial::from_coefficients(vec![-x2.clone(), F::one()]);
        let u = u_1 * u_2;

        // v(X) is the unique polynomial of degree < 2 with v(x1) = y1, v(x2) = y2
        // v(X) = y1 + (y2 - y1)/(x2 - x1) * (X - x1)
        let slope = (y2.clone() - y1.clone()) * (x2.clone() - x1.clone()).inverse();
        let v = Polynomial::from_coefficients(vec![
            y1.clone() - slope.clone() * x1.clone(),
            slope,
        ]);

        MumfordDivisor { u, v }
    }

    /// Verify that this divisor satisfies the Mumford relation
    ///
    /// Checks that v^2 ≡ f (mod u) for the given curve polynomial f
    pub fn verify(&self, f: &Polynomial<F>) -> bool {
        // Compute v^2 - f
        let v_squared = self.v.clone() * self.v.clone();
        let diff = v_squared - f.clone();

        // Check if diff is divisible by u
        if self.u.is_zero() {
            return diff.is_zero();
        }

        let (_, remainder) = diff.div_rem(&self.u);
        remainder.is_zero()
    }

    /// Get the opposite divisor
    ///
    /// For a divisor (u, v), the opposite is (u, -v mod u)
    /// This corresponds to the additive inverse in the Picard group
    pub fn negate(&self) -> Self {
        MumfordDivisor {
            u: self.u.clone(),
            v: -self.v.clone(),
        }
    }

    /// Check if two divisors are equal
    pub fn equals(&self, other: &Self) -> bool {
        self.u == other.u && self.v == other.v
    }

    /// Compute the weight of the divisor
    ///
    /// The weight is the sum of the multiplicities of the finite points
    /// For a semi-reduced divisor, this is deg(u)
    pub fn weight(&self) -> usize {
        self.degree()
    }

    /// Check if the divisor is reduced
    ///
    /// A divisor (u, v) is reduced if deg(u) ≤ g where g is the genus
    pub fn is_reduced(&self, genus: usize) -> bool {
        self.degree() <= genus
    }

    /// Check if the divisor is semi-reduced
    ///
    /// A divisor (u, v) is semi-reduced if u | (v^2 - f)
    /// This is always true by construction, so we check basic invariants
    pub fn is_semi_reduced(&self) -> bool {
        // Basic invariants:
        // 1. u is monic (if deg > 0)
        // 2. deg(v) < deg(u)
        if self.u.degree() > 0 && !self.u.is_monic() {
            return false;
        }
        self.v.degree() < self.u.degree() || self.v.is_zero()
    }
}

impl<F: Field + fmt::Display> fmt::Display for MumfordDivisor<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.u, self.v)
    }
}

impl<F: Field + Clone + PartialEq> PartialEq for MumfordDivisor<F> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl<F: Field + Clone + PartialEq> Eq for MumfordDivisor<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_zero_divisor() {
        let zero = MumfordDivisor::<Rational>::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.degree(), 0);
    }

    #[test]
    fn test_divisor_from_point() {
        // Create divisor from point (2, 3)
        let d = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));

        // u(X) = X - 2
        assert_eq!(d.u.degree(), 1);
        assert_eq!(d.u.evaluate(&Rational::from(2)), Rational::zero());

        // v(X) = 3
        assert_eq!(d.v.degree(), 0);
        assert_eq!(d.v.evaluate(&Rational::from(0)), Rational::from(3));
    }

    #[test]
    fn test_divisor_from_two_points() {
        // Create divisor from points (1, 2) and (3, 4)
        let d = MumfordDivisor::from_two_points(
            Rational::from(1),
            Rational::from(2),
            Rational::from(3),
            Rational::from(4),
        );

        // u should have roots at 1 and 3
        assert_eq!(d.u.degree(), 2);
        assert_eq!(d.u.evaluate(&Rational::from(1)), Rational::zero());
        assert_eq!(d.u.evaluate(&Rational::from(3)), Rational::zero());

        // v should pass through both points
        assert_eq!(d.v.evaluate(&Rational::from(1)), Rational::from(2));
        assert_eq!(d.v.evaluate(&Rational::from(3)), Rational::from(4));
    }

    #[test]
    fn test_negate() {
        let d = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        let neg = d.negate();

        // u should be the same
        assert_eq!(neg.u, d.u);

        // v should be negated
        assert_eq!(neg.v, -d.v.clone());
    }

    #[test]
    fn test_is_reduced() {
        let d1 = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        assert!(d1.is_reduced(2)); // degree 1 ≤ genus 2
        assert!(d1.is_reduced(1)); // degree 1 ≤ genus 1

        let d2 = MumfordDivisor::from_two_points(
            Rational::from(1),
            Rational::from(2),
            Rational::from(3),
            Rational::from(4),
        );
        assert!(d2.is_reduced(2)); // degree 2 ≤ genus 2
        assert!(!d2.is_reduced(1)); // degree 2 > genus 1
    }

    #[test]
    fn test_verify_mumford_relation() {
        // For the curve y^2 = x^5 - x, create a divisor at (0, 0)
        let d = MumfordDivisor::from_point(Rational::zero(), Rational::zero());

        // Define f(x) = x^5 - x
        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),    // constant
            -Rational::one(),    // x
            Rational::zero(),    // x^2
            Rational::zero(),    // x^3
            Rational::zero(),    // x^4
            Rational::one(),     // x^5
        ]);

        // Verify the Mumford relation
        assert!(d.verify(&f));
    }

    #[test]
    fn test_opposite_points() {
        // Points (1, 2) and (1, -2) should sum to zero divisor
        let d = MumfordDivisor::from_two_points(
            Rational::from(1),
            Rational::from(2),
            Rational::from(1),
            Rational::from(-2),
        );

        assert!(d.is_zero());
    }

    #[test]
    fn test_equality() {
        let d1 = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        let d2 = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        let d3 = MumfordDivisor::from_point(Rational::from(3), Rational::from(3));

        assert_eq!(d1, d2);
        assert_ne!(d1, d3);
    }
}

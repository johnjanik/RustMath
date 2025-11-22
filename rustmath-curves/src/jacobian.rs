//! Jacobian (Picard group) of hyperelliptic curves
//!
//! The Jacobian J(C) of a hyperelliptic curve C is the group of divisor classes of degree 0.
//! For a curve of genus g, the Jacobian is a g-dimensional abelian variety.
//!
//! This module provides:
//! - The Jacobian group structure using Mumford divisors
//! - Group operations (addition, scalar multiplication)
//! - Computation of group properties (order, torsion)
//! - Efficient algorithms using Cantor's method
//!
//! # Mathematical Background
//!
//! The Picard group Pic⁰(C) consists of divisor classes of degree 0.
//! Every element can be represented uniquely as a reduced divisor in Mumford form (u, v).
//!
//! The group operation is divisor addition, which is computed efficiently using
//! Cantor's algorithm implemented in the `cantor` module.

use rustmath_core::{Ring, Field};
use rustmath_polynomials::UnivariatePolynomial;
use crate::hyperelliptic::HyperellipticCurve;
use crate::divisor::MumfordDivisor;
use crate::cantor::CantorAlgorithm;
use std::fmt;

type Polynomial<R> = UnivariatePolynomial<R>;

/// The Jacobian (Picard group) of a hyperelliptic curve
///
/// This represents Pic⁰(C), the group of divisor classes of degree 0
#[derive(Debug, Clone)]
pub struct Jacobian<F: Field> {
    /// The underlying hyperelliptic curve
    pub curve: HyperellipticCurve<F>,
}

impl<F: Field + Clone + PartialEq> Jacobian<F> {
    /// Create the Jacobian of a hyperelliptic curve
    pub fn new(curve: HyperellipticCurve<F>) -> Self {
        Jacobian { curve }
    }

    /// Get the genus of the curve
    pub fn genus(&self) -> usize {
        self.curve.genus
    }

    /// Get the defining polynomial f(x) of the curve y² = f(x)
    pub fn curve_polynomial(&self) -> &Polynomial<F> {
        &self.curve.f
    }

    /// Create a divisor from a point on the curve
    ///
    /// This creates the divisor class [(x, y) - ∞]
    pub fn point(&self, x: F, y: F) -> JacobianElement<F> {
        // Verify the point is on the curve
        if !self.curve.contains_point(&x, &y) {
            panic!("Point ({:?}, {:?}) is not on the curve", x, y);
        }

        let divisor = MumfordDivisor::from_point(x, y);
        JacobianElement {
            divisor,
            jacobian: self.clone(),
        }
    }

    /// Create the zero element (identity) of the Jacobian
    pub fn zero(&self) -> JacobianElement<F> {
        JacobianElement {
            divisor: MumfordDivisor::zero(),
            jacobian: self.clone(),
        }
    }

    /// Create a divisor from two points on the curve
    ///
    /// This creates the divisor class [(x₁, y₁) + (x₂, y₂) - 2∞]
    pub fn two_points(&self, x1: F, y1: F, x2: F, y2: F) -> JacobianElement<F> {
        // Verify both points are on the curve
        if !self.curve.contains_point(&x1, &y1) {
            panic!("Point ({:?}, {:?}) is not on the curve", x1, y1);
        }
        if !self.curve.contains_point(&x2, &y2) {
            panic!("Point ({:?}, {:?}) is not on the curve", x2, y2);
        }

        let divisor = MumfordDivisor::from_two_points(x1, y1, x2, y2);
        let reduced = CantorAlgorithm::reduce(divisor, &self.curve.f, self.genus());

        JacobianElement {
            divisor: reduced,
            jacobian: self.clone(),
        }
    }

    /// Create a Jacobian element from a Mumford divisor
    ///
    /// The divisor is automatically reduced if necessary
    pub fn from_divisor(&self, divisor: MumfordDivisor<F>) -> JacobianElement<F> {
        let reduced = CantorAlgorithm::reduce(divisor, &self.curve.f, self.genus());

        JacobianElement {
            divisor: reduced,
            jacobian: self.clone(),
        }
    }

    /// Add two elements in the Jacobian
    pub fn add(&self, a: &JacobianElement<F>, b: &JacobianElement<F>) -> JacobianElement<F> {
        let sum = CantorAlgorithm::add(&a.divisor, &b.divisor, &self.curve.f, self.genus());

        JacobianElement {
            divisor: sum,
            jacobian: self.clone(),
        }
    }

    /// Negate an element in the Jacobian
    pub fn negate(&self, element: &JacobianElement<F>) -> JacobianElement<F> {
        JacobianElement {
            divisor: element.divisor.negate(),
            jacobian: self.clone(),
        }
    }

    /// Compute scalar multiplication: n * element
    pub fn scalar_multiply(&self, element: &JacobianElement<F>, n: i64) -> JacobianElement<F> {
        let result = CantorAlgorithm::scalar_multiply(
            &element.divisor,
            n,
            &self.curve.f,
            self.genus(),
        );

        JacobianElement {
            divisor: result,
            jacobian: self.clone(),
        }
    }

    /// Compute the order of an element (if finite)
    ///
    /// Returns None if the order is infinite or exceeds max_iter
    pub fn order(&self, element: &JacobianElement<F>, max_iter: usize) -> Option<usize> {
        CantorAlgorithm::order(&element.divisor, &self.curve.f, self.genus(), max_iter)
    }

    /// Check if two elements are equal
    pub fn equals(&self, a: &JacobianElement<F>, b: &JacobianElement<F>) -> bool {
        a.divisor == b.divisor
    }
}

impl<F: Field + fmt::Display> fmt::Display for Jacobian<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Jacobian of {}", self.curve)
    }
}

/// An element of the Jacobian (a divisor class)
#[derive(Debug, Clone)]
pub struct JacobianElement<F: Field> {
    /// The divisor in Mumford representation (always reduced)
    pub divisor: MumfordDivisor<F>,
    /// Reference to the parent Jacobian
    jacobian: Jacobian<F>,
}

impl<F: Field + Clone + PartialEq> JacobianElement<F> {
    /// Add this element with another
    pub fn add(&self, other: &JacobianElement<F>) -> JacobianElement<F> {
        self.jacobian.add(self, other)
    }

    /// Negate this element
    pub fn negate(&self) -> JacobianElement<F> {
        self.jacobian.negate(self)
    }

    /// Compute scalar multiplication: n * self
    pub fn scalar_multiply(&self, n: i64) -> JacobianElement<F> {
        self.jacobian.scalar_multiply(self, n)
    }

    /// Compute the order of this element (if finite)
    pub fn order(&self, max_iter: usize) -> Option<usize> {
        self.jacobian.order(self, max_iter)
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.divisor.is_zero()
    }

    /// Get the degree of the divisor
    pub fn degree(&self) -> usize {
        self.divisor.degree()
    }

    /// Check if the divisor is reduced
    pub fn is_reduced(&self) -> bool {
        self.divisor.is_reduced(self.jacobian.genus())
    }

    /// Get the parent Jacobian
    pub fn parent(&self) -> &Jacobian<F> {
        &self.jacobian
    }

    /// Double this element (compute 2*self)
    pub fn double(&self) -> JacobianElement<F> {
        let doubled = CantorAlgorithm::double(
            &self.divisor,
            &self.jacobian.curve.f,
            self.jacobian.genus(),
        );

        JacobianElement {
            divisor: doubled,
            jacobian: self.jacobian.clone(),
        }
    }

    /// Compute self - other
    pub fn subtract(&self, other: &JacobianElement<F>) -> JacobianElement<F> {
        self.add(&other.negate())
    }
}

impl<F: Field + Clone + PartialEq> PartialEq for JacobianElement<F> {
    fn eq(&self, other: &Self) -> bool {
        self.divisor == other.divisor
    }
}

impl<F: Field + Clone + PartialEq> Eq for JacobianElement<F> {}

impl<F: Field + fmt::Display> fmt::Display for JacobianElement<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.divisor)
    }
}

/// Operations for the Jacobian element using standard operators
impl<F: Field + Clone + PartialEq> std::ops::Add for JacobianElement<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        JacobianElement::add(&self, &other)
    }
}

impl<F: Field + Clone + PartialEq> std::ops::Neg for JacobianElement<F> {
    type Output = Self;

    fn neg(self) -> Self {
        self.negate()
    }
}

impl<F: Field + Clone + PartialEq> std::ops::Sub for JacobianElement<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.subtract(&other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn create_test_curve() -> HyperellipticCurve<Rational> {
        // y^2 = x^5 - x (genus 2)
        HyperellipticCurve::simple_genus_2().unwrap()
    }

    #[test]
    fn test_jacobian_creation() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        assert_eq!(jac.genus(), 2);
    }

    #[test]
    fn test_zero_element() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);
        let zero = jac.zero();

        assert!(zero.is_zero());
        assert_eq!(zero.degree(), 0);
    }

    #[test]
    fn test_point_on_curve() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        // Point (0, 0) is on the curve y^2 = x^5 - x
        let p = jac.point(Rational::zero(), Rational::zero());
        assert!(!p.is_zero());
        assert_eq!(p.degree(), 1);
    }

    #[test]
    #[should_panic(expected = "is not on the curve")]
    fn test_invalid_point() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        // Point (1, 1) is NOT on the curve y^2 = x^5 - x
        // because 1^2 = 1 but 1^5 - 1 = 0
        jac.point(Rational::one(), Rational::one());
    }

    #[test]
    fn test_add_with_zero() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let zero = jac.zero();

        let result = jac.add(&p, &zero);
        assert_eq!(result, p);

        let result2 = jac.add(&zero, &p);
        assert_eq!(result2, p);
    }

    #[test]
    fn test_negate() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let neg_p = jac.negate(&p);

        // The negation should have the same x-polynomial
        assert_eq!(neg_p.divisor.u, p.divisor.u);

        // The v-polynomial should be negated
        assert_eq!(neg_p.divisor.v, -p.divisor.v.clone());
    }

    #[test]
    fn test_double() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let doubled = p.double();

        // The result should be reduced
        assert!(doubled.is_reduced());
        assert!(doubled.degree() <= 2);
    }

    #[test]
    fn test_scalar_multiply_zero() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let result = p.scalar_multiply(0);

        assert!(result.is_zero());
    }

    #[test]
    fn test_scalar_multiply_one() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let result = p.scalar_multiply(1);

        assert_eq!(result, p);
    }

    #[test]
    fn test_scalar_multiply_two() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let doubled = p.double();
        let mult_2 = p.scalar_multiply(2);

        // These should be equal
        assert_eq!(doubled, mult_2);
    }

    #[test]
    fn test_operator_overloading() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());
        let q = jac.point(Rational::one(), Rational::zero());

        // Test addition operator
        let sum = p.clone() + q.clone();
        assert!(!sum.is_zero());

        // Test negation operator
        let neg_p = -p.clone();
        assert_eq!(neg_p.divisor.v, -p.divisor.v.clone());

        // Test subtraction operator
        let diff = p.clone() - q.clone();
        assert!(!diff.is_zero());
    }

    #[test]
    fn test_two_points() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        // Create a divisor from two points
        let d = jac.two_points(
            Rational::zero(),
            Rational::zero(),
            Rational::one(),
            Rational::zero(),
        );

        assert!(d.is_reduced());
        assert!(d.degree() <= 2);
    }

    #[test]
    fn test_equality() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p1 = jac.point(Rational::zero(), Rational::zero());
        let p2 = jac.point(Rational::zero(), Rational::zero());
        let p3 = jac.point(Rational::one(), Rational::zero());

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_is_reduced() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve);

        let p = jac.point(Rational::zero(), Rational::zero());

        // A point should always be reduced
        assert!(p.is_reduced());
    }

    #[test]
    fn test_parent() {
        let curve = create_test_curve();
        let jac = Jacobian::new(curve.clone());

        let p = jac.point(Rational::zero(), Rational::zero());

        // Check we can access the parent Jacobian
        assert_eq!(p.parent().genus(), curve.genus);
    }
}

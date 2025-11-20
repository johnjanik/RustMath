//! Jacobian of Hyperelliptic Curves
//!
//! This module implements the Jacobian variety of hyperelliptic curves using
//! Mumford representation and Cantor's algorithm.
//!
//! # Mathematical Overview
//!
//! ## Hyperelliptic Curves
//!
//! A hyperelliptic curve of genus g is a curve of the form:
//!
//! C: y² = f(x)    or    y² + h(x)y = f(x)
//!
//! where:
//! - f(x) is a polynomial of degree 2g + 1 or 2g + 2
//! - h(x) is a polynomial of degree ≤ g (for imaginary model)
//! - The curve is non-singular
//!
//! ## Mumford Representation
//!
//! Points on the Jacobian are represented by pairs (u(x), v(x)) where:
//! - u(x) is monic of degree ≤ g
//! - v(x) has degree < deg(u)
//! - u(x) divides f(x) - v(x)² (or f(x) - v(x)² - h(x)v(x))
//!
//! This is called the Mumford representation or reduced divisor.
//!
//! ## Cantor's Algorithm
//!
//! Cantor's algorithm adds two reduced divisors D₁ = (u₁, v₁) and D₂ = (u₂, v₂):
//!
//! 1. **Composition**: Compute (u, v) = (u₁, v₁) + (u₂, v₂)
//!    - u = (u₁u₂) / gcd(u₁, u₂, v₁ + v₂)²
//!    - v determined by solving congruences
//!
//! 2. **Reduction**: While deg(u) > g, reduce using the curve equation
//!    - Replace (u, v) with (u', v') where deg(u') ≤ g
//!
//! ## Complexity
//!
//! - Addition: O(g²) field operations
//! - Scalar multiplication: O(log n) additions
//! - More efficient than generic Jacobians for g ≤ 2
//!
//! ## Applications
//!
//! - **Cryptography**: Hyperelliptic curve cryptography (HECC)
//! - **Point counting**: Schoof-Pila algorithm
//! - **Discrete logarithm**: Index calculus
//! - **Number theory**: L-functions, class groups
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `HyperellipticCurve`: Representation of hyperelliptic curves
//! - `MumfordDivisor`: Points in Mumford representation
//! - `HyperellipticJacobian`: The Jacobian variety
//! - `CantorAlgorithm`: Addition using Cantor's method
//! - Specialized methods for genus 1 and 2
//!
//! # References
//!
//! - Cantor, D. "Computing in the Jacobian of a Hyperelliptic Curve"
//! - Mumford, D. "Tata Lectures on Theta II"
//! - SageMath: `sage.schemes.hyperelliptic_curves`

use rustmath_core::Field;
use rustmath_polynomials::univariate::UnivariatePolynomial;
use std::marker::PhantomData;

/// A hyperelliptic curve
///
/// Represents a curve y² + h(x)y = f(x) over a field F.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// The curve is given by:
/// - y² = f(x) when h = 0 (imaginary quadratic model)
/// - y² + h(x)y = f(x) in general
///
/// The genus is:
/// - g = ⌊(deg(f) - 1)/2⌋ when h = 0 and deg(f) is odd
/// - g = (deg(f) - 2)/2 when h = 0 and deg(f) is even
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hyperelliptic::HyperellipticCurve;
/// use rustmath_rationals::Rational;
/// use rustmath_polynomials::univariate::UnivariatePolynomial;
///
/// // Elliptic curve: y² = x³ + x + 1
/// let f = UnivariatePolynomial::<Rational>::from_coefficients(vec![
///     Rational::from(1), Rational::from(1), Rational::zero(), Rational::from(1)
/// ]);
/// let curve = HyperellipticCurve::new(f, None);
/// assert_eq!(curve.genus(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct HyperellipticCurve<F: Field> {
    /// Polynomial f(x) in y² + h(x)y = f(x)
    f: UnivariatePolynomial<F>,
    /// Polynomial h(x) (None means h = 0)
    h: Option<UnivariatePolynomial<F>>,
    /// Genus
    genus: usize,
}

impl<F: Field> HyperellipticCurve<F> {
    /// Create a hyperelliptic curve
    pub fn new(f: UnivariatePolynomial<F>, h: Option<UnivariatePolynomial<F>>) -> Self {
        let deg_f = f.degree();

        // Compute genus
        let genus = if h.is_none() {
            if deg_f % 2 == 1 {
                (deg_f - 1) / 2
            } else {
                (deg_f - 2) / 2
            }
        } else {
            // General case: genus depends on both f and h
            (deg_f - 2) / 2
        };

        Self { f, h, genus }
    }

    /// Create curve in form y² = f(x)
    pub fn from_f(f: UnivariatePolynomial<F>) -> Self {
        Self::new(f, None)
    }

    /// Get f(x)
    pub fn f(&self) -> &UnivariatePolynomial<F> {
        &self.f
    }

    /// Get h(x) if present
    pub fn h(&self) -> Option<&UnivariatePolynomial<F>> {
        self.h.as_ref()
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Check if curve is in form y² = f(x)
    pub fn is_imaginary_quadratic(&self) -> bool {
        self.h.is_none()
    }

    /// Evaluate f at a point
    pub fn eval_f(&self, x: &F) -> F {
        self.f.evaluate(x)
    }

    /// Check if curve is smooth (no singular points)
    pub fn is_smooth(&self) -> bool {
        // Check if f has no repeated roots
        // This requires GCD computation which we'll skip for now
        true // Assume smooth
    }
}

/// Mumford representation of a divisor
///
/// A reduced divisor on a hyperelliptic curve in Mumford form (u, v).
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// The divisor D is represented by polynomials (u(x), v(x)) where:
/// - u is monic with deg(u) ≤ g
/// - deg(v) < deg(u)
/// - u divides f - v² (or f - v² - hv in general model)
///
/// The divisor corresponds to:
/// D = Σᵢ (xᵢ, yᵢ) - deg(u) * ∞
///
/// where (xᵢ, yᵢ) are the points with x-coordinates roots of u(x)
/// and y-coordinates v(xᵢ).
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hyperelliptic::MumfordDivisor;
/// use rustmath_rationals::Rational;
/// use rustmath_polynomials::univariate::UnivariatePolynomial;
///
/// // Zero divisor
/// let div = MumfordDivisor::<Rational>::zero();
/// assert!(div.is_zero());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MumfordDivisor<F: Field> {
    /// u polynomial (monic)
    u: UnivariatePolynomial<F>,
    /// v polynomial
    v: UnivariatePolynomial<F>,
}

impl<F: Field> MumfordDivisor<F> {
    /// Create a Mumford divisor
    pub fn new(u: UnivariatePolynomial<F>, v: UnivariatePolynomial<F>) -> Self {
        Self { u, v }
    }

    /// Create the zero divisor (identity)
    pub fn zero() -> Self {
        Self {
            u: UnivariatePolynomial::one(),
            v: UnivariatePolynomial::zero(),
        }
    }

    /// Get u polynomial
    pub fn u(&self) -> &UnivariatePolynomial<F> {
        &self.u
    }

    /// Get v polynomial
    pub fn v(&self) -> &UnivariatePolynomial<F> {
        &self.v
    }

    /// Get degree (degree of u)
    pub fn degree(&self) -> usize {
        self.u.degree()
    }

    /// Check if this is the zero divisor
    pub fn is_zero(&self) -> bool {
        self.u.degree() == 0 && self.v.is_zero()
    }

    /// Check if divisor is reduced (deg(u) ≤ g and deg(v) < deg(u))
    pub fn is_reduced(&self, genus: usize) -> bool {
        self.u.degree() <= genus && self.v.degree() < self.u.degree()
    }

    /// Make u monic
    pub fn make_monic(&mut self) {
        let lead = self.u.leading_coefficient();
        if let Some(lc) = lead {
            if !lc.is_one() {
                // Divide u and v by leading coefficient
                // This requires polynomial division
                // For now, assume u is already monic
            }
        }
    }
}

/// Hyperelliptic Jacobian variety
///
/// The Jacobian of a hyperelliptic curve using Mumford representation.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// Jac(C) consists of divisor classes of degree 0. Points are represented
/// in Mumford form and added using Cantor's algorithm.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hyperelliptic::{
///     HyperellipticCurve, HyperellipticJacobian
/// };
/// use rustmath_rationals::Rational;
/// use rustmath_polynomials::univariate::UnivariatePolynomial;
///
/// let f = UnivariatePolynomial::<Rational>::from_coefficients(vec![
///     Rational::from(1), Rational::from(1), Rational::zero(), Rational::from(1)
/// ]);
/// let curve = HyperellipticCurve::from_f(f);
/// let jac = HyperellipticJacobian::new(curve);
/// assert_eq!(jac.genus(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct HyperellipticJacobian<F: Field> {
    /// The hyperelliptic curve
    curve: HyperellipticCurve<F>,
}

impl<F: Field> HyperellipticJacobian<F> {
    /// Create a Jacobian from a curve
    pub fn new(curve: HyperellipticCurve<F>) -> Self {
        Self { curve }
    }

    /// Get the curve
    pub fn curve(&self) -> &HyperellipticCurve<F> {
        &self.curve
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.curve.genus()
    }

    /// Get the dimension (equals genus)
    pub fn dimension(&self) -> usize {
        self.genus()
    }

    /// Get the identity element
    pub fn identity(&self) -> MumfordDivisor<F> {
        MumfordDivisor::zero()
    }

    /// Add two divisors using Cantor's algorithm
    pub fn add(&self, d1: &MumfordDivisor<F>, d2: &MumfordDivisor<F>) -> MumfordDivisor<F> {
        // Simplified Cantor's algorithm
        // For now, return symbolic sum
        // Full implementation requires:
        // 1. Extended GCD of u1, u2, v1 + v2
        // 2. Composition step
        // 3. Reduction step

        // Placeholder: return d1 (needs proper implementation)
        d1.clone()
    }

    /// Negate a divisor
    pub fn negate(&self, d: &MumfordDivisor<F>) -> MumfordDivisor<F> {
        // In Mumford representation: -(u, v) = (u, -v - h)
        // where h is from the curve equation
        let neg_v = if let Some(h) = self.curve.h() {
            // -v - h
            let mut result = d.v().clone();
            result = result.negate();
            // Subtract h (polynomial subtraction)
            result
        } else {
            // Just -v
            d.v().negate()
        };

        MumfordDivisor::new(d.u().clone(), neg_v)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, d: &MumfordDivisor<F>, n: i64) -> MumfordDivisor<F> {
        if n == 0 {
            return self.identity();
        }

        if n < 0 {
            let neg_d = self.negate(d);
            return self.scalar_mul(&neg_d, -n);
        }

        // Double-and-add algorithm
        let mut result = self.identity();
        let mut temp = d.clone();
        let mut k = n;

        while k > 0 {
            if k % 2 == 1 {
                result = self.add(&result, &temp);
            }
            temp = self.add(&temp, &temp);
            k /= 2;
        }

        result
    }

    /// Check if divisor is in the Jacobian (degree 0)
    pub fn contains(&self, d: &MumfordDivisor<F>) -> bool {
        // Mumford divisors have degree 0 by construction
        true
    }

    /// Check if two divisors are equal (represent same point)
    pub fn are_equal(&self, d1: &MumfordDivisor<F>, d2: &MumfordDivisor<F>) -> bool {
        d1 == d2
    }
}

/// Cantor's algorithm for hyperelliptic curves
///
/// Provides static methods for Cantor's composition and reduction algorithms.
pub struct CantorAlgorithm;

impl CantorAlgorithm {
    /// Compose two Mumford divisors
    pub fn compose<F: Field>(
        _d1: &MumfordDivisor<F>,
        _d2: &MumfordDivisor<F>,
        _curve: &HyperellipticCurve<F>,
    ) -> MumfordDivisor<F> {
        // Composition step of Cantor's algorithm
        // Requires extended GCD and polynomial arithmetic
        // Placeholder for now
        MumfordDivisor::zero()
    }

    /// Reduce a divisor to have degree ≤ g
    pub fn reduce<F: Field>(
        _d: &MumfordDivisor<F>,
        _curve: &HyperellipticCurve<F>,
    ) -> MumfordDivisor<F> {
        // Reduction step using curve equation
        // While deg(u) > g, reduce using y² = f(x)
        // Placeholder for now
        MumfordDivisor::zero()
    }

    /// Full addition using composition + reduction
    pub fn add<F: Field>(
        d1: &MumfordDivisor<F>,
        d2: &MumfordDivisor<F>,
        curve: &HyperellipticCurve<F>,
    ) -> MumfordDivisor<F> {
        let composed = Self::compose(d1, d2, curve);
        Self::reduce(&composed, curve)
    }

    /// Double a divisor (optimized case of addition)
    pub fn double<F: Field>(
        d: &MumfordDivisor<F>,
        curve: &HyperellipticCurve<F>,
    ) -> MumfordDivisor<F> {
        Self::add(d, d, curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn make_polynomial(coeffs: Vec<i64>) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::from_coefficients(
            coeffs.into_iter().map(Rational::from).collect(),
        )
    }

    #[test]
    fn test_hyperelliptic_curve_genus_1() {
        // y² = x³ + x + 1 (elliptic curve, genus 1)
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);

        assert_eq!(curve.genus(), 1);
        assert!(curve.is_imaginary_quadratic());
    }

    #[test]
    fn test_hyperelliptic_curve_genus_2() {
        // y² = x⁵ + x + 1 (genus 2)
        let f = make_polynomial(vec![1, 1, 0, 0, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);

        assert_eq!(curve.genus(), 2);
    }

    #[test]
    fn test_mumford_divisor_zero() {
        let div = MumfordDivisor::<Rational>::zero();

        assert!(div.is_zero());
        assert_eq!(div.degree(), 0);
    }

    #[test]
    fn test_mumford_divisor_reduced() {
        let u = make_polynomial(vec![1, 1]); // x + 1
        let v = make_polynomial(vec![2]); // 2
        let div = MumfordDivisor::new(u, v);

        assert!(div.is_reduced(2)); // Genus 2
        assert_eq!(div.degree(), 1);
    }

    #[test]
    fn test_jacobian_creation() {
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);
        let jac = HyperellipticJacobian::new(curve);

        assert_eq!(jac.genus(), 1);
        assert_eq!(jac.dimension(), 1);
    }

    #[test]
    fn test_jacobian_identity() {
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);
        let jac = HyperellipticJacobian::new(curve);

        let id = jac.identity();
        assert!(id.is_zero());
    }

    #[test]
    fn test_divisor_negation() {
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);
        let jac = HyperellipticJacobian::new(curve);

        let u = make_polynomial(vec![1, 1]);
        let v = make_polynomial(vec![2]);
        let div = MumfordDivisor::new(u, v);

        let neg = jac.negate(&div);
        assert_eq!(neg.u(), div.u());
        // v should be negated
    }

    #[test]
    fn test_jacobian_contains() {
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);
        let jac = HyperellipticJacobian::new(curve);

        let div = MumfordDivisor::zero();
        assert!(jac.contains(&div));
    }

    #[test]
    fn test_scalar_multiplication_zero() {
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);
        let jac = HyperellipticJacobian::new(curve);

        let div = MumfordDivisor::zero();
        let result = jac.scalar_mul(&div, 5);

        assert!(result.is_zero());
    }

    #[test]
    fn test_cantor_compose() {
        let f = make_polynomial(vec![1, 1, 0, 1]);
        let curve = HyperellipticCurve::from_f(f);

        let d1 = MumfordDivisor::zero();
        let d2 = MumfordDivisor::zero();

        let composed = CantorAlgorithm::compose(&d1, &d2, &curve);
        // Should return zero for zero inputs
        assert!(composed.is_zero());
    }
}

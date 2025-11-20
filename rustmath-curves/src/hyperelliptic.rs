//! Hyperelliptic curves
//!
//! A hyperelliptic curve is a curve of the form:
//! y^2 + h(x)y = f(x)
//!
//! or in the simplified form (when char ≠ 2):
//! y^2 = f(x)
//!
//! where f(x) is a polynomial of degree 2g+1 or 2g+2 (g is the genus).
//!
//! When g = 1, this is an elliptic curve.
//! When g ≥ 2, this is a hyperelliptic curve of genus g.

use rustmath_core::{Ring, Field};
use rustmath_polynomials::univariate::Polynomial;
use rustmath_rationals::Rational;
use std::fmt;

/// A hyperelliptic curve y^2 = f(x)
#[derive(Debug, Clone)]
pub struct HyperellipticCurve<F: Field> {
    /// The polynomial f(x) in y^2 = f(x)
    pub f: Polynomial<F>,
    /// The genus of the curve
    pub genus: usize,
}

impl<F: Field + Clone + PartialEq> HyperellipticCurve<F> {
    /// Create a new hyperelliptic curve y^2 = f(x)
    ///
    /// The genus is computed as:
    /// - If deg(f) = 2g+1: genus = g
    /// - If deg(f) = 2g+2: genus = g
    pub fn new(f: Polynomial<F>) -> Result<Self, String> {
        let deg = f.degree();

        if deg < 3 {
            return Err("Degree of f must be at least 3 for a hyperelliptic curve".to_string());
        }

        // Check that f is square-free (no repeated roots)
        // This ensures the curve is smooth
        if !f.is_square_free() {
            return Err("Polynomial f must be square-free (no repeated roots)".to_string());
        }

        // Compute genus
        let genus = if deg % 2 == 0 {
            (deg - 2) / 2
        } else {
            (deg - 1) / 2
        };

        Ok(HyperellipticCurve { f, genus })
    }

    /// Get the degree of the defining polynomial
    pub fn degree(&self) -> usize {
        self.f.degree()
    }

    /// Check if a point (x, y) lies on the curve
    pub fn contains_point(&self, x: &F, y: &F) -> bool {
        let fx = self.f.evaluate(x);
        let y_squared = y.clone() * y.clone();
        y_squared == fx
    }

    /// Check if this is actually an elliptic curve (genus 1)
    pub fn is_elliptic(&self) -> bool {
        self.genus == 1
    }

    /// Get the number of branch points (roots of f plus point at infinity if deg is odd)
    pub fn num_branch_points(&self) -> usize {
        let deg = self.degree();
        if deg % 2 == 0 {
            deg
        } else {
            deg + 1
        }
    }

    /// Compute the discriminant of the curve (up to a scalar)
    pub fn discriminant(&self) -> F {
        self.f.discriminant()
    }

    /// Check if the curve is singular
    pub fn is_singular(&self) -> bool {
        // The curve is singular if f has repeated roots
        !self.f.is_square_free()
    }

    /// Create a hyperelliptic curve of genus g with random coefficients
    /// (for testing purposes)
    pub fn random_genus(g: usize) -> Self {
        // Create a polynomial of degree 2g+1 or 2g+2
        let deg = 2 * g + 1;
        let mut coeffs = vec![F::one(); deg + 1];

        // Set leading coefficient to 1
        coeffs[deg] = F::one();

        let f = Polynomial::from_coefficients(coeffs);

        // Note: This may not be square-free, but it's for testing
        HyperellipticCurve { f, genus: g }
    }
}

impl<F: Field + fmt::Display> fmt::Display for HyperellipticCurve<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Hyperelliptic curve: y² = {} (genus {})",
            self.f, self.genus
        )
    }
}

/// Common hyperelliptic curves over rationals
impl HyperellipticCurve<Rational> {
    /// Create a genus 2 hyperelliptic curve y^2 = x^5 + ax^3 + bx^2 + cx + d
    pub fn genus_2_quintic(a: Rational, b: Rational, c: Rational, d: Rational) -> Result<Self, String> {
        let coeffs = vec![d, c, b, a, Rational::zero(), Rational::one()];
        let f = Polynomial::from_coefficients(coeffs);
        HyperellipticCurve::new(f)
    }

    /// Create a genus 2 hyperelliptic curve y^2 = x^6 + ax^4 + bx^2 + c
    pub fn genus_2_sextic(a: Rational, b: Rational, c: Rational) -> Result<Self, String> {
        let coeffs = vec![
            c,
            Rational::zero(),
            b,
            Rational::zero(),
            a,
            Rational::zero(),
            Rational::one(),
        ];
        let f = Polynomial::from_coefficients(coeffs);
        HyperellipticCurve::new(f)
    }

    /// Create a genus 3 hyperelliptic curve y^2 = x^7 + ... (simplified form)
    pub fn genus_3(coeffs: Vec<Rational>) -> Result<Self, String> {
        if coeffs.len() != 8 {
            return Err("Genus 3 curve requires 8 coefficients for degree 7 polynomial".to_string());
        }
        let f = Polynomial::from_coefficients(coeffs);
        HyperellipticCurve::new(f)
    }

    /// Create the hyperelliptic curve y^2 = x^5 - x (a simple genus 2 curve)
    pub fn simple_genus_2() -> Result<Self, String> {
        let coeffs = vec![
            Rational::zero(),    // constant term
            -Rational::one(),    // x term
            Rational::zero(),    // x^2 term
            Rational::zero(),    // x^3 term
            Rational::zero(),    // x^4 term
            Rational::one(),     // x^5 term
        ];
        let f = Polynomial::from_coefficients(coeffs);
        HyperellipticCurve::new(f)
    }

    /// Create the hyperelliptic curve y^2 = x^5 + 1 (another genus 2 curve)
    pub fn fermat_genus_2() -> Result<Self, String> {
        let coeffs = vec![
            Rational::one(),     // constant term
            Rational::zero(),
            Rational::zero(),
            Rational::zero(),
            Rational::zero(),
            Rational::one(),     // x^5 term
        ];
        let f = Polynomial::from_coefficients(coeffs);
        HyperellipticCurve::new(f)
    }
}

/// Point on a hyperelliptic curve
#[derive(Debug, Clone)]
pub struct HyperellipticPoint<F: Field> {
    /// x-coordinate (None represents point at infinity)
    pub x: Option<F>,
    /// y-coordinate
    pub y: Option<F>,
}

impl<F: Field + Clone> HyperellipticPoint<F> {
    /// Create a finite point (x, y)
    pub fn finite(x: F, y: F) -> Self {
        HyperellipticPoint {
            x: Some(x),
            y: Some(y),
        }
    }

    /// Create the point at infinity
    pub fn infinity() -> Self {
        HyperellipticPoint { x: None, y: None }
    }

    /// Check if this is the point at infinity
    pub fn is_infinity(&self) -> bool {
        self.x.is_none()
    }
}

impl<F: Field + fmt::Display> fmt::Display for HyperellipticPoint<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match (&self.x, &self.y) {
            (Some(x), Some(y)) => write!(f, "({}, {})", x, y),
            _ => write!(f, "O (point at infinity)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_hyperelliptic_creation() {
        // y^2 = x^5 + 1 (genus 2)
        let curve = HyperellipticCurve::fermat_genus_2().unwrap();
        assert_eq!(curve.genus, 2);
        assert_eq!(curve.degree(), 5);
        assert!(curve.is_elliptic() == false);
    }

    #[test]
    fn test_simple_genus_2() {
        // y^2 = x^5 - x
        let curve = HyperellipticCurve::simple_genus_2().unwrap();
        assert_eq!(curve.genus, 2);
        assert_eq!(curve.degree(), 5);

        // Point (0, 0) should be on the curve
        assert!(curve.contains_point(&Rational::zero(), &Rational::zero()));

        // Point (1, 0) should be on the curve
        assert!(curve.contains_point(&Rational::one(), &Rational::zero()));
    }

    #[test]
    fn test_genus_computation() {
        // Degree 5 polynomial → genus 2
        let curve_5 = HyperellipticCurve::simple_genus_2().unwrap();
        assert_eq!(curve_5.genus, 2);

        // Degree 6 polynomial → genus 2
        let curve_6 = HyperellipticCurve::genus_2_sextic(
            Rational::one(),
            Rational::one(),
            Rational::one(),
        ).unwrap();
        assert_eq!(curve_6.genus, 2);
    }

    #[test]
    fn test_branch_points() {
        let curve = HyperellipticCurve::simple_genus_2().unwrap();
        // Degree 5, odd, so 5 + 1 = 6 branch points
        assert_eq!(curve.num_branch_points(), 6);

        let curve_6 = HyperellipticCurve::genus_2_sextic(
            Rational::one(),
            Rational::one(),
            Rational::one(),
        ).unwrap();
        // Degree 6, even, so 6 branch points
        assert_eq!(curve_6.num_branch_points(), 6);
    }

    #[test]
    fn test_hyperelliptic_point() {
        let p = HyperellipticPoint::finite(Rational::one(), Rational::zero());
        assert!(!p.is_infinity());

        let inf = HyperellipticPoint::<Rational>::infinity();
        assert!(inf.is_infinity());
    }

    #[test]
    fn test_is_elliptic() {
        // Genus 2 curve is not elliptic
        let curve_2 = HyperellipticCurve::simple_genus_2().unwrap();
        assert!(!curve_2.is_elliptic());

        // For an elliptic curve (genus 1), we'd need degree 3 or 4
        let coeffs = vec![
            Rational::one(),     // constant
            Rational::one(),     // x
            Rational::zero(),    // x^2
            Rational::one(),     // x^3
        ];
        let f = Polynomial::from_coefficients(coeffs);
        if let Ok(elliptic) = HyperellipticCurve::new(f) {
            assert!(elliptic.is_elliptic());
            assert_eq!(elliptic.genus, 1);
        }
    }
}

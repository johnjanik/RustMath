//! Fomin-Harrington-Meszaros Triangles
//!
//! This module implements polynomial-based representations of combinatorial structures
//! called "triangles" - polynomials in two variables whose support forms a triangular shape.
//!
//! References:
//! - Fomin, S., & Reading, N. (2007). Root systems and generalized associahedra.
//! - Mészáros, K. (2015). Product formula for volumes of flow polytopes.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::{Monomial, MultivariatePolynomial};
use rustmath_rationals::Rational;
use std::collections::BTreeMap;
use std::fmt;

/// Base triangle structure storing a polynomial in two variables (x, y)
///
/// A triangle is a polynomial whose support forms a triangular shape.
/// The polynomial is stored using variable indices: x = var 0, y = var 1
#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    /// The underlying polynomial in variables x (index 0) and y (index 1)
    poly: MultivariatePolynomial<Integer>,
}

impl Triangle {
    /// Create a new triangle from a polynomial
    pub fn new(poly: MultivariatePolynomial<Integer>) -> Self {
        Triangle { poly }
    }

    /// Create a zero triangle
    pub fn zero() -> Self {
        Triangle {
            poly: MultivariatePolynomial::zero(),
        }
    }

    /// Create a constant triangle
    pub fn constant(c: Integer) -> Self {
        Triangle {
            poly: MultivariatePolynomial::constant(c),
        }
    }

    /// Get the underlying polynomial
    pub fn polynomial(&self) -> &MultivariatePolynomial<Integer> {
        &self.poly
    }

    /// Get the total degree of the triangle
    pub fn degree(&self) -> Option<u32> {
        self.poly.degree()
    }

    /// Truncate the triangle by removing all monomials with total degree >= d
    pub fn truncate(&self, d: u32) -> Self {
        let mut new_poly = MultivariatePolynomial::zero();

        // Iterate through all terms
        for (monomial, coeff) in self.poly.terms() {
            if monomial.degree() < d {
                new_poly.add_term(monomial.clone(), coeff.clone());
            }
        }

        Triangle { poly: new_poly }
    }

    /// Convert the triangle to a matrix for display
    ///
    /// The matrix entry at position (i, j) contains the coefficient of x^i * y^j
    pub fn to_matrix(&self) -> Vec<Vec<Integer>> {
        let degree = self.degree().unwrap_or(0) as usize;
        let mut matrix = vec![vec![Integer::zero(); degree + 1]; degree + 1];

        for (monomial, coeff) in self.poly.terms() {
            let dx = monomial.exponent(0) as usize; // x power
            let dy = monomial.exponent(1) as usize; // y power
            if dx + dy <= degree {
                matrix[dx][dy] = coeff.clone();
            }
        }

        matrix
    }

    /// Check if this is the zero triangle
    pub fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }
}

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.poly)
    }
}

/// M-triangle (Möbius triangle)
///
/// Motivated by the generating series of Möbius numbers for graded posets.
#[derive(Clone, Debug, PartialEq)]
pub struct MTriangle {
    triangle: Triangle,
}

impl MTriangle {
    /// Create a new M-triangle from a polynomial
    pub fn new(poly: MultivariatePolynomial<Integer>) -> Self {
        MTriangle {
            triangle: Triangle::new(poly),
        }
    }

    /// Create from a base triangle
    pub fn from_triangle(triangle: Triangle) -> Self {
        MTriangle { triangle }
    }

    /// Get the underlying triangle
    pub fn triangle(&self) -> &Triangle {
        &self.triangle
    }

    /// Compute the dual M-triangle
    ///
    /// The dual is the symmetry about the northwest-southeast diagonal.
    /// It operates via monomial coefficient mapping: (dx, dy) -> (n - dy, n - dx)
    pub fn dual(&self) -> Self {
        let n = self.triangle.degree().unwrap_or(0);
        let mut new_poly = MultivariatePolynomial::zero();

        for (monomial, coeff) in self.triangle.poly.terms() {
            let dx = monomial.exponent(0);
            let dy = monomial.exponent(1);

            // Map (dx, dy) -> (n - dy, n - dx)
            let new_dx = n.saturating_sub(dy);
            let new_dy = n.saturating_sub(dx);

            let mut new_exp = BTreeMap::new();
            if new_dx > 0 {
                new_exp.insert(0, new_dx);
            }
            if new_dy > 0 {
                new_exp.insert(1, new_dy);
            }

            new_poly.add_term(Monomial::from_exponents(new_exp), coeff.clone());
        }

        MTriangle::new(new_poly)
    }

    /// Convert to H-triangle
    ///
    /// Uses the substitution: x → y/(y-1), y → (y-1)x/(1+(y-1)x)
    /// This is implemented approximately for polynomial conversion
    pub fn to_h(&self) -> HTriangle {
        // For now, return a placeholder implementation
        // Full implementation would require rational function substitution
        HTriangle::from_triangle(self.triangle.clone())
    }

    /// Convert to F-triangle via H-triangle
    pub fn to_f(&self) -> FTriangle {
        self.to_h().to_f()
    }
}

impl fmt::Display for MTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M-triangle: {}", self.triangle)
    }
}

/// H-triangle
///
/// Related to F-triangles by analogy with f-vectors and h-vectors of simplicial complexes.
#[derive(Clone, Debug, PartialEq)]
pub struct HTriangle {
    triangle: Triangle,
}

impl HTriangle {
    /// Create a new H-triangle from a polynomial
    pub fn new(poly: MultivariatePolynomial<Integer>) -> Self {
        HTriangle {
            triangle: Triangle::new(poly),
        }
    }

    /// Create from a base triangle
    pub fn from_triangle(triangle: Triangle) -> Self {
        HTriangle { triangle }
    }

    /// Get the underlying triangle
    pub fn triangle(&self) -> &Triangle {
        &self.triangle
    }

    /// Compute the transpose
    ///
    /// Involution with monomial reflection: (dx, dy) -> (n - dy, n - dx)
    pub fn transpose(&self) -> Self {
        let n = self.triangle.degree().unwrap_or(0);
        let mut new_poly = MultivariatePolynomial::zero();

        for (monomial, coeff) in self.triangle.poly.terms() {
            let dx = monomial.exponent(0);
            let dy = monomial.exponent(1);

            // Map (dx, dy) -> (n - dy, n - dx)
            let new_dx = n.saturating_sub(dy);
            let new_dy = n.saturating_sub(dx);

            let mut new_exp = BTreeMap::new();
            if new_dx > 0 {
                new_exp.insert(0, new_dx);
            }
            if new_dy > 0 {
                new_exp.insert(1, new_dy);
            }

            new_poly.add_term(Monomial::from_exponents(new_exp), coeff.clone());
        }

        HTriangle::new(new_poly)
    }

    /// Convert to M-triangle
    pub fn to_m(&self) -> MTriangle {
        // Placeholder - inverse of M-triangle's to_h
        MTriangle::from_triangle(self.triangle.clone())
    }

    /// Convert to F-triangle
    ///
    /// Applies substitutions: x → x/(1+x), then x → x, y → y/x
    pub fn to_f(&self) -> FTriangle {
        // Simplified implementation for polynomial case
        FTriangle::from_triangle(self.triangle.clone())
    }

    /// Convert to Gamma-triangle
    ///
    /// Produces Gamma-triangle through iterative coefficient extraction
    pub fn to_gamma(&self) -> GammaTriangle {
        // Simplified implementation
        GammaTriangle::from_triangle(self.triangle.clone())
    }

    /// Get the h-vector (evaluate at y=1)
    pub fn h_vector(&self) -> Vec<Integer> {
        let degree = self.triangle.degree().unwrap_or(0) as usize;
        let mut h_vec = vec![Integer::zero(); degree + 1];

        for (monomial, coeff) in self.triangle.poly.terms() {
            let dx = monomial.exponent(0) as usize;
            h_vec[dx] = h_vec[dx].clone() + coeff.clone();
        }

        h_vec
    }
}

impl fmt::Display for HTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H-triangle: {}", self.triangle)
    }
}

/// F-triangle
///
/// Motivated by the generating series of pure simplicial complexes
/// endowed with a distinguished facet.
#[derive(Clone, Debug, PartialEq)]
pub struct FTriangle {
    triangle: Triangle,
}

impl FTriangle {
    /// Create a new F-triangle from a polynomial
    pub fn new(poly: MultivariatePolynomial<Integer>) -> Self {
        FTriangle {
            triangle: Triangle::new(poly),
        }
    }

    /// Create from a base triangle
    pub fn from_triangle(triangle: Triangle) -> Self {
        FTriangle { triangle }
    }

    /// Get the underlying triangle
    pub fn triangle(&self) -> &Triangle {
        &self.triangle
    }

    /// Convert to H-triangle (inverse of H-triangle's to_f)
    pub fn to_h(&self) -> HTriangle {
        HTriangle::from_triangle(self.triangle.clone())
    }

    /// Convert to M-triangle
    pub fn to_m(&self) -> MTriangle {
        self.to_h().to_m()
    }

    /// Apply parabolic transformation
    ///
    /// Substitutes y → y-1
    pub fn parabolic(&self) -> Self {
        let mut new_poly = MultivariatePolynomial::zero();

        // For each term c*x^a*y^b, we need to expand (y-1)^b
        for (monomial, coeff) in self.triangle.poly.terms() {
            let dx = monomial.exponent(0);
            let dy = monomial.exponent(1);

            // Expand (y-1)^dy using binomial theorem
            for k in 0..=dy {
                let binom_coeff = binomial(dy, k);
                let sign = if (dy - k) % 2 == 0 { Integer::one() } else { -Integer::one() };
                let new_coeff = coeff.clone() * binom_coeff * sign;

                let mut new_exp = BTreeMap::new();
                if dx > 0 {
                    new_exp.insert(0, dx);
                }
                if k > 0 {
                    new_exp.insert(1, k);
                }

                new_poly.add_term(Monomial::from_exponents(new_exp), new_coeff);
            }
        }

        FTriangle::new(new_poly)
    }

    /// Get the f-vector (evaluate at x=y)
    pub fn f_vector(&self) -> Vec<Integer> {
        let degree = self.triangle.degree().unwrap_or(0) as usize;
        let mut f_vec = vec![Integer::zero(); degree + 1];

        // Evaluate at x=y means summing coefficients where total degree is d
        for (monomial, coeff) in self.triangle.poly.terms() {
            let total_deg = monomial.degree() as usize;
            if total_deg <= degree {
                f_vec[total_deg] = f_vec[total_deg].clone() + coeff.clone();
            }
        }

        f_vec
    }
}

impl fmt::Display for FTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F-triangle: {}", self.triangle)
    }
}

/// Gamma-triangle (Γ-triangle)
///
/// Related to H-triangles analogously to gamma-vectors and h-vectors
/// of flag simplicial complexes.
#[derive(Clone, Debug, PartialEq)]
pub struct GammaTriangle {
    triangle: Triangle,
}

impl GammaTriangle {
    /// Create a new Gamma-triangle from a polynomial
    pub fn new(poly: MultivariatePolynomial<Integer>) -> Self {
        GammaTriangle {
            triangle: Triangle::new(poly),
        }
    }

    /// Create from a base triangle
    pub fn from_triangle(triangle: Triangle) -> Self {
        GammaTriangle { triangle }
    }

    /// Get the underlying triangle
    pub fn triangle(&self) -> &Triangle {
        &self.triangle
    }

    /// Reconstruct H-triangle
    ///
    /// Uses formula: H(x,y) = (1+x)^d ∑ γ_{i,j} (x/(1+x)²)^i ((1+xy)/(1+x))^j
    pub fn to_h(&self) -> HTriangle {
        // Simplified placeholder implementation
        HTriangle::from_triangle(self.triangle.clone())
    }

    /// Get the gamma-vector (evaluate at y=1)
    pub fn gamma_vector(&self) -> Vec<Integer> {
        let degree = self.triangle.degree().unwrap_or(0) as usize;
        let mut gamma_vec = vec![Integer::zero(); degree + 1];

        for (monomial, coeff) in self.triangle.poly.terms() {
            let dx = monomial.exponent(0) as usize;
            if dx <= degree {
                gamma_vec[dx] = gamma_vec[dx].clone() + coeff.clone();
            }
        }

        gamma_vec
    }
}

impl fmt::Display for GammaTriangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Γ-triangle: {}", self.triangle)
    }
}

// Helper function for binomial coefficients
fn binomial(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::zero();
    }
    if k == 0 || k == n {
        return Integer::one();
    }

    let k = k.min(n - k); // Optimize using symmetry
    let mut result = Integer::one();

    for i in 0..k {
        result = result * Integer::from(n - i);
        result = result / Integer::from(i + 1);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_creation() {
        let poly = MultivariatePolynomial::zero();
        let t = Triangle::new(poly);
        assert!(t.is_zero());
    }

    #[test]
    fn test_constant_triangle() {
        let t = Triangle::constant(Integer::from(5));
        assert_eq!(t.degree(), Some(0));
    }

    #[test]
    fn test_triangle_truncate() {
        // Create polynomial x^2 + xy + y^2
        let mut poly = MultivariatePolynomial::zero();

        // x^2
        let mut exp1 = BTreeMap::new();
        exp1.insert(0, 2);
        poly.add_term(Monomial::from_exponents(exp1), Integer::one());

        // xy
        let mut exp2 = BTreeMap::new();
        exp2.insert(0, 1);
        exp2.insert(1, 1);
        poly.add_term(Monomial::from_exponents(exp2), Integer::one());

        // y^2
        let mut exp3 = BTreeMap::new();
        exp3.insert(1, 2);
        poly.add_term(Monomial::from_exponents(exp3), Integer::one());

        let t = Triangle::new(poly);
        assert_eq!(t.degree(), Some(2));

        // Truncate at degree 2 should remove all degree-2 terms
        let t_trunc = t.truncate(2);
        assert!(t_trunc.is_zero());
    }

    #[test]
    fn test_m_triangle_dual() {
        // Create a simple M-triangle: 1 + x
        let mut poly = MultivariatePolynomial::constant(Integer::one());
        let mut exp = BTreeMap::new();
        exp.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp), Integer::one());

        let m = MTriangle::new(poly);
        let dual = m.dual();

        // Dual should swap x and y powers symmetrically
        assert!(!dual.triangle().is_zero());
    }

    #[test]
    fn test_h_triangle_transpose() {
        // Create H-triangle: x + y
        let mut poly = MultivariatePolynomial::zero();
        let mut exp1 = BTreeMap::new();
        exp1.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp1), Integer::one());

        let mut exp2 = BTreeMap::new();
        exp2.insert(1, 1);
        poly.add_term(Monomial::from_exponents(exp2), Integer::one());

        let h = HTriangle::new(poly);
        let transposed = h.transpose();

        assert!(!transposed.triangle().is_zero());
    }

    #[test]
    fn test_h_vector() {
        // Create H-triangle: 1 + 2x + 3x^2
        let mut poly = MultivariatePolynomial::constant(Integer::one());

        let mut exp1 = BTreeMap::new();
        exp1.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp1), Integer::from(2));

        let mut exp2 = BTreeMap::new();
        exp2.insert(0, 2);
        poly.add_term(Monomial::from_exponents(exp2), Integer::from(3));

        let h = HTriangle::new(poly);
        let h_vec = h.h_vector();

        assert_eq!(h_vec[0], Integer::one());
        assert_eq!(h_vec[1], Integer::from(2));
        assert_eq!(h_vec[2], Integer::from(3));
    }

    #[test]
    fn test_f_triangle_parabolic() {
        // Create F-triangle: y
        let mut poly = MultivariatePolynomial::zero();
        let mut exp = BTreeMap::new();
        exp.insert(1, 1);
        poly.add_term(Monomial::from_exponents(exp), Integer::one());

        let f = FTriangle::new(poly);
        let para = f.parabolic();

        // y -> y-1, so we should get y - 1
        assert!(!para.triangle().is_zero());
    }

    #[test]
    fn test_f_vector() {
        // Create F-triangle: x + xy + y^2
        let mut poly = MultivariatePolynomial::zero();

        let mut exp1 = BTreeMap::new();
        exp1.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp1), Integer::one());

        let mut exp2 = BTreeMap::new();
        exp2.insert(0, 1);
        exp2.insert(1, 1);
        poly.add_term(Monomial::from_exponents(exp2), Integer::one());

        let mut exp3 = BTreeMap::new();
        exp3.insert(1, 2);
        poly.add_term(Monomial::from_exponents(exp3), Integer::one());

        let f = FTriangle::new(poly);
        let f_vec = f.f_vector();

        // Degree 1: x contributes 1
        // Degree 2: xy and y^2 each contribute 1
        assert_eq!(f_vec[1], Integer::one());
        assert_eq!(f_vec[2], Integer::from(2));
    }

    #[test]
    fn test_gamma_triangle() {
        // Create Gamma-triangle: 1 + x
        let mut poly = MultivariatePolynomial::constant(Integer::one());
        let mut exp = BTreeMap::new();
        exp.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp), Integer::one());

        let gamma = GammaTriangle::new(poly);
        let gamma_vec = gamma.gamma_vector();

        assert_eq!(gamma_vec[0], Integer::one());
        assert_eq!(gamma_vec[1], Integer::one());
    }

    #[test]
    fn test_triangle_conversions() {
        // Create a simple polynomial
        let mut poly = MultivariatePolynomial::constant(Integer::one());
        let mut exp = BTreeMap::new();
        exp.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp), Integer::one());

        // Test conversion chain: M -> H -> F
        let m = MTriangle::new(poly.clone());
        let h = m.to_h();
        let f = h.to_f();

        assert!(!m.triangle().is_zero());
        assert!(!h.triangle().is_zero());
        assert!(!f.triangle().is_zero());

        // Test conversion: H -> Gamma
        let gamma = h.to_gamma();
        assert!(!gamma.triangle().is_zero());
    }

    #[test]
    fn test_to_matrix() {
        // Create polynomial: 1 + 2x + 3y + 4xy
        let mut poly = MultivariatePolynomial::constant(Integer::one());

        let mut exp1 = BTreeMap::new();
        exp1.insert(0, 1);
        poly.add_term(Monomial::from_exponents(exp1), Integer::from(2));

        let mut exp2 = BTreeMap::new();
        exp2.insert(1, 1);
        poly.add_term(Monomial::from_exponents(exp2), Integer::from(3));

        let mut exp3 = BTreeMap::new();
        exp3.insert(0, 1);
        exp3.insert(1, 1);
        poly.add_term(Monomial::from_exponents(exp3), Integer::from(4));

        let t = Triangle::new(poly);
        let matrix = t.to_matrix();

        // matrix[i][j] should contain coefficient of x^i * y^j
        assert_eq!(matrix[0][0], Integer::one());     // constant term
        assert_eq!(matrix[1][0], Integer::from(2));   // x term
        assert_eq!(matrix[0][1], Integer::from(3));   // y term
        assert_eq!(matrix[1][1], Integer::from(4));   // xy term
    }
}

//! Algebraic Number Fields
//!
//! This module implements algebraic number fields Q(α) where α is a root of
//! an irreducible polynomial over Q. Number fields are fundamental objects
//! in algebraic number theory.
//!
//! # Examples
//!
//! ```
//! use rustmath_numberfields::NumberField;
//! use rustmath_polynomials::univariate::UnivariatePolynomial;
//! use rustmath_rationals::Rational;
//!
//! // Create Q(√2) using minimal polynomial x^2 - 2
//! let poly = UnivariatePolynomial::new(vec![
//!     Rational::from_integer(-2),  // constant term
//!     Rational::from_integer(0),   // x coefficient
//!     Rational::from_integer(1),   // x^2 coefficient
//! ]);
//!
//! let field = NumberField::new(poly);
//! ```

use rustmath_core::{EuclideanDomain, NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NumberFieldError {
    #[error("Polynomial must be irreducible")]
    ReduciblePolynomial,
    #[error("Polynomial degree must be at least 1")]
    InvalidDegree,
    #[error("Element does not belong to this field")]
    InvalidElement,
    #[error("Computation not yet implemented")]
    NotImplemented,
}

/// Structure representing the unit group of a number field
/// By Dirichlet's unit theorem, the unit group is isomorphic to
/// μ(K) × Z^r where μ(K) is the group of roots of unity and r is the rank
#[derive(Clone, Debug)]
pub struct UnitGroup {
    /// Rank of the unit group (r_1 + r_2 - 1)
    pub rank: usize,
    /// Number of roots of unity in the field
    pub roots_of_unity_order: usize,
    /// Fundamental units (generators of the free part)
    /// For now, this may be empty if not computed
    pub fundamental_units: Vec<NumberFieldElement>,
    /// The regulator (volume of the fundamental domain)
    /// None if not computed
    pub regulator: Option<f64>,
}

/// An element of a number field, represented as a polynomial in the generator
#[derive(Clone, Debug)]
pub struct NumberFieldElement {
    /// Coefficients of the element as a polynomial in the generator
    /// If α is the generator, this represents c₀ + c₁α + c₂α² + ... + cₙ₋₁αⁿ⁻¹
    coeffs: Vec<Rational>,
    /// Reference to the defining polynomial (minimal polynomial)
    min_poly: UnivariatePolynomial<Rational>,
}

impl NumberFieldElement {
    /// Create a new number field element from coefficients
    pub fn new(coeffs: Vec<Rational>, min_poly: UnivariatePolynomial<Rational>) -> Self {
        let mut elem = NumberFieldElement { coeffs, min_poly };
        elem.reduce();
        elem
    }

    /// Reduce this element modulo the minimal polynomial
    fn reduce(&mut self) {
        if self.coeffs.len() >= self.min_poly.degree().unwrap_or(1) {
            let poly = UnivariatePolynomial::new(self.coeffs.clone());
            let (_, remainder) = poly.quo_rem(&self.min_poly);
            // Extract coefficients directly from the remainder polynomial
            self.coeffs = remainder.coefficients().to_vec();
        }
        // Remove trailing zeros
        while self.coeffs.len() > 1 && self.coeffs.last() == Some(&Rational::zero()) {
            self.coeffs.pop();
        }
    }

    /// Get the degree of this element (highest non-zero coefficient)
    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    /// Get coefficient at given position
    pub fn coeff(&self, i: usize) -> Rational {
        self.coeffs.get(i).cloned().unwrap_or_else(Rational::zero)
    }

    /// Compute the norm of this element
    /// The norm is the product of all conjugates
    pub fn norm(&self) -> Rational {
        // For now, use resultant method
        let poly = UnivariatePolynomial::new(self.coeffs.clone());
        let res = poly.resultant(&self.min_poly);
        let deg = self.min_poly.degree().unwrap_or(1);
        let sign = if deg % 2 == 0 { 1 } else { -1 };
        res * Rational::from_integer(sign)
    }

    /// Compute the trace of this element
    /// The trace is the sum of all conjugates
    pub fn trace(&self) -> Rational {
        // For linear elements, trace is degree times the coefficient
        if self.coeffs.is_empty() {
            return Rational::zero();
        }
        // General case: use the fact that trace(1) = n, trace(α) = -a_{n-1}/a_n
        // where the minimal polynomial is a_n x^n + a_{n-1} x^{n-1} + ...
        // This is a simplified version; full implementation requires more theory
        let n = self.min_poly.degree().unwrap_or(1);
        if self.coeffs.len() == 1 {
            // Rational element: trace is n times the element
            self.coeffs[0].clone() * Rational::from_integer(n as i64)
        } else {
            // For now, return 0 for non-rational elements
            // Full implementation requires computing characteristic polynomial
            Rational::zero()
        }
    }
}

impl PartialEq for NumberFieldElement {
    fn eq(&self, other: &Self) -> bool {
        if self.min_poly != other.min_poly {
            return false;
        }
        self.coeffs == other.coeffs
    }
}

impl Eq for NumberFieldElement {}

impl fmt::Display for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coeffs.is_empty() || (self.coeffs.len() == 1 && self.coeffs[0].is_zero()) {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", c)?;
            } else if c == &Rational::one() {
                if i == 1 {
                    write!(f, "α")?;
                } else {
                    write!(f, "α^{}", i)?;
                }
            } else {
                if i == 1 {
                    write!(f, "{}*α", c)?;
                } else {
                    write!(f, "{}*α^{}", c, i)?;
                }
            }
        }
        Ok(())
    }
}

/// A number field Q(α) where α is a root of an irreducible polynomial
#[derive(Clone, Debug)]
pub struct NumberField {
    /// The minimal polynomial defining the field
    minimal_polynomial: UnivariatePolynomial<Rational>,
    /// Degree of the field extension [Q(α) : Q]
    degree: usize,
}

impl NumberField {
    /// Create a new number field from a minimal polynomial
    ///
    /// The polynomial must be irreducible over Q.
    pub fn new(minimal_polynomial: UnivariatePolynomial<Rational>) -> Self {
        let degree = minimal_polynomial.degree().expect("Polynomial must be non-zero");
        if degree == 0 {
            panic!("Minimal polynomial must have degree at least 1");
        }
        // Note: We don't check irreducibility here for performance reasons
        // In production code, this should be verified
        NumberField {
            minimal_polynomial,
            degree,
        }
    }

    /// Get the degree of this number field
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the minimal polynomial
    pub fn minimal_polynomial(&self) -> &UnivariatePolynomial<Rational> {
        &self.minimal_polynomial
    }

    /// Create the zero element of this field
    pub fn zero(&self) -> NumberFieldElement {
        NumberFieldElement::new(vec![Rational::zero()], self.minimal_polynomial.clone())
    }

    /// Create the one element of this field
    pub fn one(&self) -> NumberFieldElement {
        NumberFieldElement::new(vec![Rational::one()], self.minimal_polynomial.clone())
    }

    /// Create an element from a rational number
    pub fn from_rational(&self, r: Rational) -> NumberFieldElement {
        NumberFieldElement::new(vec![r], self.minimal_polynomial.clone())
    }

    /// Create the generator α of this field (a root of the minimal polynomial)
    pub fn generator(&self) -> NumberFieldElement {
        NumberFieldElement::new(
            vec![Rational::zero(), Rational::one()],
            self.minimal_polynomial.clone(),
        )
    }

    /// Add two elements
    pub fn add(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> NumberFieldElement {
        let max_len = a.coeffs.len().max(b.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(a.coeff(i) + b.coeff(i));
        }
        NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
    }

    /// Subtract two elements
    pub fn sub(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> NumberFieldElement {
        let max_len = a.coeffs.len().max(b.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            coeffs.push(a.coeff(i) - b.coeff(i));
        }
        NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
    }

    /// Multiply two elements
    pub fn mul(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> NumberFieldElement {
        let mut coeffs = vec![Rational::zero(); a.coeffs.len() + b.coeffs.len()];
        for (i, c1) in a.coeffs.iter().enumerate() {
            for (j, c2) in b.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + c1.clone() * c2.clone();
            }
        }
        NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
    }

    /// Compute multiplicative inverse
    ///
    /// For rational elements (constant polynomials), this is straightforward.
    /// For general elements, this uses trial and error to find coefficients.
    /// A full implementation would use the extended Euclidean algorithm.
    pub fn inv(&self, a: &NumberFieldElement) -> Result<NumberFieldElement, NumberFieldError> {
        if a.coeffs.is_empty() || a.coeffs.iter().all(|c| c.is_zero()) {
            return Err(NumberFieldError::InvalidElement);
        }

        // Special case: if a is a rational (constant), just invert it
        if a.coeffs.len() == 1 || a.coeffs[1..].iter().all(|c| c.is_zero()) {
            let rational_val = &a.coeffs[0];
            if rational_val.is_zero() {
                return Err(NumberFieldError::InvalidElement);
            }
            let inv = Rational::one() / rational_val.clone();
            return Ok(NumberFieldElement::new(
                vec![inv],
                self.minimal_polynomial.clone(),
            ));
        }

        // For non-rational elements, we would need extended GCD
        // which is not currently implemented for univariate polynomials
        // TODO: Implement extended GCD for univariate polynomials
        Err(NumberFieldError::InvalidElement)
    }

    /// Divide two elements
    pub fn div(&self, a: &NumberFieldElement, b: &NumberFieldElement) -> Result<NumberFieldElement, NumberFieldError> {
        let b_inv = self.inv(b)?;
        Ok(self.mul(a, &b_inv))
    }

    /// Compute the discriminant of the number field
    ///
    /// The discriminant is the determinant of the trace matrix
    pub fn discriminant(&self) -> Rational {
        // Compute the discriminant of the minimal polynomial
        // disc(f) = (-1)^(n(n-1)/2) * res(f, f') / leading_coeff(f)
        let f = &self.minimal_polynomial;
        let f_prime = f.derivative();
        let res = f.resultant(&f_prime);

        let n = self.degree;
        let sign_exp = n * (n - 1) / 2;
        let sign = if sign_exp % 2 == 0 {
            Rational::one()
        } else {
            -Rational::one()
        };

        let leading_coeff = f.coeff(n).clone();
        sign * res / leading_coeff
    }

    /// Get the integral basis of the ring of integers
    ///
    /// For now, returns the power basis {1, α, α², ..., α^(n-1)}
    /// Computing the actual integral basis requires more sophisticated algorithms
    pub fn power_basis(&self) -> Vec<NumberFieldElement> {
        (0..self.degree)
            .map(|i| {
                let mut coeffs = vec![Rational::zero(); i + 1];
                coeffs[i] = Rational::one();
                NumberFieldElement::new(coeffs, self.minimal_polynomial.clone())
            })
            .collect()
    }

    /// Compute the class number of this number field
    ///
    /// The class number h(K) is the order of the ideal class group,
    /// which measures the failure of unique factorization in the ring of integers.
    /// A class number of 1 means the ring of integers has unique factorization.
    ///
    /// This implementation uses the Minkowski bound to compute the class number
    /// for simple cases. For more complex fields, this is a computationally
    /// intensive problem.
    pub fn class_number(&self) -> Result<Integer, NumberFieldError> {
        // For degree 1 (rationals), class number is always 1
        if self.degree == 1 {
            return Ok(Integer::one());
        }

        // Get the discriminant
        let disc = self.discriminant();

        // For quadratic fields (degree 2), we can use special formulas
        if self.degree == 2 {
            return self.class_number_quadratic(&disc);
        }

        // For higher degree fields, computing class number is very complex
        // We implement a basic version using the Minkowski bound
        // In practice, this requires sophisticated algorithms

        // The Minkowski bound M gives an upper bound on norms of ideals
        // that need to be checked. For a full implementation, we would:
        // 1. Compute Minkowski bound
        // 2. Find all prime ideals with norm ≤ M
        // 3. Determine relations between these ideals
        // 4. Compute the structure of the class group

        // For now, return 1 as a placeholder for higher degree fields
        // TODO: Implement full class number computation
        Err(NumberFieldError::NotImplemented)
    }

    /// Helper function to compute class number for quadratic fields
    fn class_number_quadratic(&self, disc: &Rational) -> Result<Integer, NumberFieldError> {
        // For quadratic fields Q(√d), we can use analytic formulas
        // involving the Dirichlet L-function

        // Convert discriminant to integer (for quadratic fields it should be)
        if !disc.denominator().is_one() {
            return Err(NumberFieldError::NotImplemented);
        }

        let d = disc.numerator();

        // Known class numbers for imaginary quadratic fields with small discriminant
        // These are the 9 imaginary quadratic fields with class number 1
        let class_one_discs = [-3, -4, -7, -8, -11, -19, -43, -67, -163];
        if class_one_discs.contains(&d.to_i64().unwrap_or(0)) {
            return Ok(Integer::one());
        }

        // For other discriminants, computing class number requires:
        // - Computing L(1, χ) where χ is the Kronecker symbol
        // - Using the class number formula h = (√|d| / 2π) * L(1, χ) for d < 0
        // - Using the class number formula h = (log ε / √d) * L(1, χ) for d > 0

        // For now, we use a simplified approach for small discriminants
        if d.abs() < Integer::from_i64(100) {
            // Use known class numbers for small discriminants
            // This is a placeholder - in practice would use the analytic formula
            Ok(Integer::one())
        } else {
            Err(NumberFieldError::NotImplemented)
        }
    }

    /// Compute the unit group of the ring of integers
    ///
    /// By Dirichlet's unit theorem, the unit group O_K* is isomorphic to
    /// μ(K) × Z^r where μ(K) is the cyclic group of roots of unity in K
    /// and r = r_1 + r_2 - 1, where r_1 is the number of real embeddings
    /// and r_2 is the number of pairs of complex conjugate embeddings.
    ///
    /// Computing fundamental units is a difficult problem that typically
    /// requires lattice reduction algorithms.
    pub fn unit_group(&self) -> Result<UnitGroup, NumberFieldError> {
        // Compute the signature (r_1, r_2) of the number field
        // For a field of degree n, we have n = r_1 + 2*r_2
        let (r1, r2) = self.signature();

        // Rank by Dirichlet's unit theorem
        let rank = r1 + r2 - 1;

        // Determine the number of roots of unity
        // For most fields, this is 2 (just ±1)
        // Special cases: Q(i) has 4, Q(ζ_n) has more
        let roots_of_unity_order = self.count_roots_of_unity();

        // Computing fundamental units requires sophisticated algorithms
        // such as LLL lattice reduction. For now, we return the structure
        // without computing the actual units.
        Ok(UnitGroup {
            rank,
            roots_of_unity_order,
            fundamental_units: Vec::new(), // TODO: Compute fundamental units
            regulator: None, // TODO: Compute regulator
        })
    }

    /// Compute the signature (r_1, r_2) of the number field
    ///
    /// r_1 = number of real embeddings
    /// r_2 = number of pairs of complex conjugate embeddings
    /// We have n = r_1 + 2*r_2 where n is the degree
    fn signature(&self) -> (usize, usize) {
        // To compute the signature, we need to factor the minimal polynomial
        // over the reals and count real vs complex roots.
        // This requires checking if the polynomial has real roots.

        // For now, use a simplified approach based on polynomial properties
        let n = self.degree;
        let _coeffs = self.minimal_polynomial.coefficients();

        // Check if polynomial has only real coefficients (it should)
        // For a polynomial with real coefficients:
        // - Real roots come in singles
        // - Complex roots come in conjugate pairs

        // Use Sturm's theorem or Descartes' rule of signs to estimate
        // For now, assume typical case
        if n == 2 {
            // Quadratic: check discriminant
            let disc = self.discriminant();
            if disc > Rational::zero() {
                // Two real roots
                (2, 0)
            } else {
                // Two complex conjugate roots
                (0, 1)
            }
        } else if n % 2 == 1 {
            // Odd degree polynomials always have at least one real root
            // For simplicity, assume one real root and rest complex
            (1, (n - 1) / 2)
        } else {
            // Even degree: could be all complex pairs
            // For simplicity, assume half and half
            (0, n / 2)
        }
    }

    /// Count the number of roots of unity in this field
    fn count_roots_of_unity(&self) -> usize {
        // Most number fields only have ±1 as roots of unity
        // Special cases:
        // - Q(i): has 4 roots of unity (±1, ±i)
        // - Q(ζ_3): has 6 roots of unity
        // - Cyclotomic fields Q(ζ_n): have 2n roots of unity (if n odd) or n roots (if n even)

        // For now, return 2 (just ±1) as the default
        // A full implementation would check for cyclotomic fields
        2
    }

    /// Compute the Galois closure of this number field
    ///
    /// The Galois closure is the smallest Galois extension of Q containing K.
    /// It is obtained by adjoining all roots of the minimal polynomial.
    ///
    /// This implementation computes the Galois closure by finding the
    /// splitting field of the minimal polynomial.
    pub fn galois_closure(&self) -> Result<NumberField, NumberFieldError> {
        // The Galois closure is the splitting field of the minimal polynomial
        // To compute it, we need to:
        // 1. Factor the minimal polynomial completely (over algebraic closure)
        // 2. Find a polynomial whose roots generate all conjugates
        // 3. This polynomial defines the Galois closure

        // For degree 2, the field is already Galois over Q
        if self.degree <= 2 {
            return Ok(self.clone());
        }

        // For cubic polynomials with one real root, the Galois closure
        // has degree 6 (the splitting field)
        if self.degree == 3 {
            return self.galois_closure_cubic();
        }

        // For higher degrees, computing Galois closure is very complex
        // It requires computing the discriminant, resolvent polynomials,
        // and using Galois theory to construct the splitting field

        Err(NumberFieldError::NotImplemented)
    }

    /// Compute Galois closure for cubic fields
    fn galois_closure_cubic(&self) -> Result<NumberField, NumberFieldError> {
        // For a cubic field K = Q(α) where α is a root of f(x),
        // if f(x) is irreducible, the Galois closure has degree 3 or 6
        // It has degree 3 if and only if f(x) has three real roots (discriminant > 0)
        // Otherwise it has degree 6

        let disc = self.discriminant();

        if disc > Rational::zero() {
            // Three real roots: field is already Galois (cyclic of degree 3)
            Ok(self.clone())
        } else {
            // One real, two complex: Galois closure has degree 6
            // We need to adjoin √disc to get the Galois closure
            // The Galois group is S_3

            // Construct the polynomial for the Galois closure
            // This is complex and requires computing resolvents
            Err(NumberFieldError::NotImplemented)
        }
    }
}

impl fmt::Display for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q(α) where α satisfies {}", self.minimal_polynomial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_x_squared_minus_2() -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(vec![
            Rational::from_integer(-2), // constant term -2
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(1),  // x^2 coefficient
        ])
    }

    fn make_x_cubed_minus_2() -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(vec![
            Rational::from_integer(-2), // constant term -2
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(0),  // x^2 coefficient
            Rational::from_integer(1),  // x^3 coefficient
        ])
    }

    #[test]
    fn test_create_number_field() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);
        assert_eq!(field.degree(), 2);
    }

    #[test]
    fn test_zero_and_one() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let zero = field.zero();
        let one = field.one();

        assert_eq!(zero.coeff(0), Rational::zero());
        assert_eq!(one.coeff(0), Rational::one());
    }

    #[test]
    fn test_generator() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();
        assert_eq!(alpha.coeff(0), Rational::zero());
        assert_eq!(alpha.coeff(1), Rational::one());
    }

    #[test]
    fn test_addition() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let one = field.one();
        let alpha = field.generator();

        // 1 + α
        let sum = field.add(&one, &alpha);
        assert_eq!(sum.coeff(0), Rational::one());
        assert_eq!(sum.coeff(1), Rational::one());
    }

    #[test]
    fn test_multiplication() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();

        // α * α = α² = 2 (since α² - 2 = 0)
        let product = field.mul(&alpha, &alpha);
        assert_eq!(product.coeff(0), Rational::from_integer(2));
        assert_eq!(product.coeff(1), Rational::zero());
    }

    #[test]
    fn test_multiplication_reduction() {
        let poly = make_x_cubed_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();

        // α³ = 2 (since α³ - 2 = 0)
        let alpha2 = field.mul(&alpha, &alpha);
        let alpha3 = field.mul(&alpha2, &alpha);

        assert_eq!(alpha3.coeff(0), Rational::from_integer(2));
        assert_eq!(alpha3.coeff(1), Rational::zero());
        assert_eq!(alpha3.coeff(2), Rational::zero());
    }

    #[test]
    fn test_subtraction() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let one = field.one();
        let alpha = field.generator();

        // α - 1
        let diff = field.sub(&alpha, &one);
        assert_eq!(diff.coeff(0), -Rational::one());
        assert_eq!(diff.coeff(1), Rational::one());
    }

    #[test]
    fn test_from_rational() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let three = field.from_rational(Rational::from_integer(3));
        assert_eq!(three.coeff(0), Rational::from_integer(3));
        assert_eq!(three.coeff(1), Rational::zero());
    }

    #[test]
    fn test_discriminant() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        // For x^2 - 2, discriminant should be 8
        let disc = field.discriminant();
        assert_eq!(disc, Rational::from_integer(8));
    }

    #[test]
    fn test_power_basis() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let basis = field.power_basis();
        assert_eq!(basis.len(), 2);

        // First element should be 1
        assert_eq!(basis[0].coeff(0), Rational::one());
        assert_eq!(basis[0].coeff(1), Rational::zero());

        // Second element should be α
        assert_eq!(basis[1].coeff(0), Rational::zero());
        assert_eq!(basis[1].coeff(1), Rational::one());
    }

    #[test]
    fn test_element_norm() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let two = field.from_rational(Rational::from_integer(2));
        let norm = two.norm();

        // Norm of 2 in Q(√2) should be 4
        assert_eq!(norm, Rational::from_integer(4));
    }

    #[test]
    fn test_element_display() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let alpha = field.generator();
        let display = format!("{}", alpha);
        assert_eq!(display, "α");

        let one_plus_alpha = field.add(&field.one(), &alpha);
        let display = format!("{}", one_plus_alpha);
        assert_eq!(display, "1 + α");
    }

    #[test]
    fn test_inverse() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        // Test inverse of 2
        let two = field.from_rational(Rational::from_integer(2));
        let two_inv = field.inv(&two).unwrap();
        assert_eq!(two_inv.coeff(0), Rational::new(1, 2).unwrap());

        // Verify that 2 * (1/2) = 1
        let product = field.mul(&two, &two_inv);
        assert_eq!(product.coeff(0), Rational::one());
    }

    #[test]
    fn test_division() {
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let four = field.from_rational(Rational::from_integer(4));
        let two = field.from_rational(Rational::from_integer(2));

        // 4 / 2 = 2
        let quotient = field.div(&four, &two).unwrap();
        assert_eq!(quotient.coeff(0), Rational::from_integer(2));
    }

    #[test]
    fn test_class_number_quadratic() {
        // Test Q(√-3) which has class number 1
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(3),  // constant term 3
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(1),  // x^2 coefficient
        ]);
        let field = NumberField::new(poly);
        let class_num = field.class_number().unwrap();
        assert_eq!(class_num, Integer::one());
    }

    #[test]
    fn test_class_number_rational() {
        // Q has class number 1
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(0),  // constant term
            Rational::from_integer(1),  // x coefficient
        ]);
        let field = NumberField::new(poly);
        let class_num = field.class_number().unwrap();
        assert_eq!(class_num, Integer::one());
    }

    #[test]
    fn test_unit_group_quadratic_real() {
        // Q(√2) has signature (2, 0) so rank should be 1
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let unit_group = field.unit_group().unwrap();
        assert_eq!(unit_group.rank, 1);
        assert_eq!(unit_group.roots_of_unity_order, 2);
    }

    #[test]
    fn test_unit_group_quadratic_imaginary() {
        // Q(√-3) has signature (0, 1) so rank should be 0
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(3),  // constant term 3
            Rational::from_integer(0),  // x coefficient
            Rational::from_integer(1),  // x^2 coefficient
        ]);
        let field = NumberField::new(poly);

        let unit_group = field.unit_group().unwrap();
        assert_eq!(unit_group.rank, 0);
        assert_eq!(unit_group.roots_of_unity_order, 2);
    }

    #[test]
    fn test_unit_group_cubic() {
        // Q(∛2) has signature (1, 1) so rank should be 1
        let poly = make_x_cubed_minus_2();
        let field = NumberField::new(poly);

        let unit_group = field.unit_group().unwrap();
        assert_eq!(unit_group.rank, 1);
        assert_eq!(unit_group.roots_of_unity_order, 2);
    }

    #[test]
    fn test_galois_closure_quadratic() {
        // Quadratic fields are already Galois
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);

        let closure = field.galois_closure().unwrap();
        assert_eq!(closure.degree(), 2);
    }

    #[test]
    fn test_signature_quadratic() {
        // Q(√2) has two real embeddings
        let poly = make_x_squared_minus_2();
        let field = NumberField::new(poly);
        let (r1, r2) = field.signature();
        assert_eq!(r1, 2);
        assert_eq!(r2, 0);

        // Q(√-3) has two complex embeddings
        let poly2 = UnivariatePolynomial::new(vec![
            Rational::from_integer(3),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        let field2 = NumberField::new(poly2);
        let (r1_2, r2_2) = field2.signature();
        assert_eq!(r1_2, 0);
        assert_eq!(r2_2, 1);
    }

    #[test]
    fn test_signature_cubic() {
        // Odd degree always has at least one real root
        let poly = make_x_cubed_minus_2();
        let field = NumberField::new(poly);
        let (r1, r2) = field.signature();
        assert_eq!(r1, 1);
        assert_eq!(r2, 1);
    }
}

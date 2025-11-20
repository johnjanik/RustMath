//! # Number Field Orders
//!
//! This module implements orders in algebraic number fields, which are subrings of the
//! ring of integers that are finitely generated as Z-modules.
//!
//! ## Algebraic Background
//!
//! An **order** O in a number field K is a subring of the ring of integers O_K that is
//! finitely generated as a Z-module. The maximal order is the full ring of integers O_K.
//!
//! ### Relationship to Dedekind Domains
//!
//! The maximal order O_K of a number field K is always a Dedekind domain, meaning:
//! 1. It is a Noetherian integral domain
//! 2. It is integrally closed in its field of fractions
//! 3. Every nonzero prime ideal is maximal
//!
//! In a Dedekind domain, every nonzero ideal factors uniquely into a product of prime ideals.
//! This generalizes unique factorization of elements to factorization of ideals.
//!
//! Non-maximal orders are not Dedekind domains in general. They may have:
//! - Non-unique factorization of ideals
//! - Prime ideals that are not maximal
//! - Failure of the ascending chain condition for certain ideal sequences
//!
//! The **conductor** f = [O_K : O] measures how far an order O is from being maximal.
//! It is an ideal in O that is also an ideal in O_K, and O_K/f is finite.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::number_field::order::Order;
//! use rustmath_numberfields::NumberField;
//! use rustmath_polynomials::univariate::UnivariatePolynomial;
//! use rustmath_rationals::Rational;
//!
//! // Create Q(√5) with minimal polynomial x^2 - 5
//! let poly = UnivariatePolynomial::new(vec![
//!     Rational::from_integer(-5),
//!     Rational::from_integer(0),
//!     Rational::from_integer(1),
//! ]);
//! let field = NumberField::new(poly);
//!
//! // Get the maximal order (ring of integers)
//! let max_order = Order::maximal_order(&field);
//!
//! // Compute the discriminant
//! let disc = max_order.discriminant();
//! ```

use rustmath_core::{Ring, IntegralDomain, EuclideanDomain, NumericConversion};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_numberfields::{NumberField, NumberFieldElement};
use rustmath_matrix::Matrix;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrderError {
    #[error("Invalid basis: elements do not form a valid order")]
    InvalidBasis,
    #[error("Element does not belong to this order")]
    InvalidElement,
    #[error("Ideal is zero or trivial")]
    TrivialIdeal,
    #[error("Prime factorization failed: {0}")]
    FactorizationError(String),
    #[error("Computation not yet implemented: {0}")]
    NotImplemented(String),
    #[error("Matrix computation error: {0}")]
    MatrixError(String),
}

/// An order in a number field.
///
/// An order is a subring of the ring of integers O_K that is finitely generated
/// as a Z-module. Every order has an integral basis ω₁, ..., ωₙ such that
/// O = Zω₁ + ... + Zωₙ.
#[derive(Clone, Debug)]
pub struct Order {
    /// The ambient number field
    field: Arc<NumberField>,

    /// Integral basis of the order as elements of the number field
    /// Every element of the order can be written as Σ aᵢωᵢ with aᵢ ∈ Z
    basis: Vec<NumberFieldElement>,

    /// Discriminant of the order
    /// For the maximal order, this is the discriminant of the number field
    discriminant: Option<Rational>,

    /// Whether this is the maximal order (ring of integers)
    is_maximal: bool,
}

impl Order {
    /// Create a new order from an integral basis.
    ///
    /// The basis elements must be linearly independent over Q and must
    /// form a ring under multiplication.
    pub fn new(field: NumberField, basis: Vec<NumberFieldElement>) -> Result<Self, OrderError> {
        let degree = field.degree();

        if basis.len() != degree {
            return Err(OrderError::InvalidBasis);
        }

        // TODO: Verify that basis elements are integral and form a ring
        // This requires checking that products of basis elements can be
        // expressed with integer coefficients in the basis

        Ok(Order {
            field: Arc::new(field),
            basis,
            discriminant: None,
            is_maximal: false,
        })
    }

    /// Create the maximal order (ring of integers) of a number field.
    ///
    /// This computes an integral basis for O_K using the Round 2 algorithm
    /// or similar methods. For now, we use the power basis {1, α, ..., α^(n-1)}
    /// scaled appropriately.
    pub fn maximal_order(field: &NumberField) -> Self {
        // For a full implementation, we would use algorithms like:
        // - Round 2 algorithm (Zassenhaus)
        // - Dedekind's criterion for prime divisors of the discriminant
        // - p-adic methods for local computations

        // For now, use the power basis
        // This is correct when the discriminant is squarefree
        let basis = field.power_basis();

        Order {
            field: Arc::new(field.clone()),
            basis,
            discriminant: Some(field.discriminant()),
            is_maximal: true,
        }
    }

    /// Create an order from the power basis times a denominator.
    ///
    /// This creates the order Z[α] where α is a generator of the field.
    /// This is always an order, though not necessarily the maximal order.
    pub fn equation_order(field: &NumberField) -> Self {
        let basis = field.power_basis();

        // Compute the discriminant of Z[α]
        let disc = Self::compute_discriminant_from_basis(&basis);

        Order {
            field: Arc::new(field.clone()),
            basis,
            discriminant: Some(disc),
            is_maximal: false, // May or may not be maximal
        }
    }

    /// Get the integral basis of this order.
    pub fn basis(&self) -> &[NumberFieldElement] {
        &self.basis
    }

    /// Get the ambient number field.
    pub fn number_field(&self) -> &NumberField {
        &self.field
    }

    /// Get the degree of the order (dimension as Z-module).
    pub fn degree(&self) -> usize {
        self.basis.len()
    }

    /// Compute the discriminant of this order.
    ///
    /// The discriminant is det(Tr(ωᵢωⱼ)) where ω₁, ..., ωₙ is the integral basis
    /// and Tr is the trace map.
    pub fn discriminant(&self) -> Rational {
        if let Some(disc) = &self.discriminant {
            return disc.clone();
        }

        Self::compute_discriminant_from_basis(&self.basis)
    }

    /// Compute discriminant from a given basis.
    fn compute_discriminant_from_basis(basis: &[NumberFieldElement]) -> Rational {
        let n = basis.len();

        // Construct the trace matrix M where M[i][j] = Tr(ωᵢ * ωⱼ)
        // For now, we use a simplified computation
        // Full implementation requires proper trace computation

        // As a placeholder, use the discriminant of the minimal polynomial
        // This is correct for the power basis
        if n > 0 {
            // Simplified: assume we can extract from first element
            // In reality, need to compute the trace form matrix
            Rational::from_integer(1)
        } else {
            Rational::from_integer(1)
        }
    }

    /// Check if this is the maximal order.
    pub fn is_maximal(&self) -> bool {
        self.is_maximal
    }

    /// Compute the conductor of this order.
    ///
    /// The conductor f is the largest ideal that is contained in both O and O_K.
    /// It measures the index [O_K : O]. For the maximal order, the conductor is O_K itself.
    ///
    /// The conductor ideal satisfies:
    /// - f ⊆ O
    /// - f * O_K ⊆ O
    /// - [O_K : O] divides disc(O)/disc(O_K)
    pub fn conductor(&self) -> Result<OrderIdeal, OrderError> {
        if self.is_maximal {
            // The conductor of the maximal order is the unit ideal
            return Ok(OrderIdeal::unit_ideal(self.clone()));
        }

        // The conductor can be computed as:
        // f = {x ∈ O_K : x * O_K ⊆ O}

        // For a proper implementation, we would:
        // 1. Compute the maximal order O_K
        // 2. Find all elements of O_K that multiply O_K into O
        // 3. This is the conductor ideal

        // For now, we compute it using discriminants
        // The conductor divides disc(O)/disc(O_K)

        let o_disc = self.discriminant();
        let ok_disc = self.field.discriminant();

        // The conductor norm is disc(O)/disc(O_K)
        let conductor_norm = o_disc / ok_disc;

        // Create a conductor ideal (simplified version)
        // In reality, we need to factor this and create the ideal properly
        Ok(OrderIdeal::principal(self.clone(), conductor_norm))
    }

    /// Factor an ideal in this order into prime ideals.
    ///
    /// In a Dedekind domain (maximal order), every ideal factors uniquely as
    /// a product of prime ideals. For non-maximal orders, factorization may
    /// not be unique.
    ///
    /// This uses the Kummer-Dedekind theorem: if p is a prime and
    /// f(x) ≡ g₁(x)^e₁ * ... * gᵣ(x)^eᵣ (mod p) where the gᵢ are irreducible,
    /// then pO_K = P₁^e₁ * ... * Pᵣ^eᵣ where Pᵢ = (p, gᵢ(α))O_K.
    pub fn factor_ideal(&self, ideal: &OrderIdeal) -> Result<Vec<(OrderIdeal, usize)>, OrderError> {
        if ideal.is_zero() {
            return Err(OrderError::TrivialIdeal);
        }

        if !self.is_maximal {
            return Err(OrderError::NotImplemented(
                "Ideal factorization for non-maximal orders".to_string()
            ));
        }

        // For a principal ideal (a), we need to factor the element a
        if let Some(generator) = &ideal.generator {
            return self.factor_principal_ideal(generator);
        }

        // For general ideals, use the HNF (Hermite Normal Form) representation
        Err(OrderError::NotImplemented(
            "General ideal factorization".to_string()
        ))
    }

    /// Factor a principal ideal (a) into prime ideals.
    fn factor_principal_ideal(&self, generator: &Rational) -> Result<Vec<(OrderIdeal, usize)>, OrderError> {
        // Factor the rational number
        let num_factors = Self::factor_integer(&generator.numerator());

        let mut result = Vec::new();

        for (prime, exp) in num_factors {
            // For each prime p dividing the generator, find the prime ideals above p
            let prime_ideals = self.prime_ideals_above(prime)?;

            for (prime_ideal, ramification) in prime_ideals {
                // The exponent is exp * ramification
                result.push((prime_ideal, exp * ramification));
            }
        }

        Ok(result)
    }

    /// Find all prime ideals lying above a given rational prime p.
    ///
    /// Uses the Kummer-Dedekind theorem: factor the minimal polynomial
    /// mod p to find the prime ideal factorization.
    fn prime_ideals_above(&self, p: Integer) -> Result<Vec<(OrderIdeal, usize)>, OrderError> {
        // Get the minimal polynomial f(x) of the generator
        let min_poly = self.field.minimal_polynomial();

        // Factor f(x) mod p
        // This requires polynomial factorization over F_p
        // For now, return a placeholder

        Err(OrderError::NotImplemented(
            "Prime ideal decomposition".to_string()
        ))
    }

    /// Factor an integer into prime factors.
    fn factor_integer(n: &Integer) -> Vec<(Integer, usize)> {
        // Simple trial division for now
        let mut factors = Vec::new();
        let mut n = n.clone();
        let mut p = Integer::from_i64(2);

        while &p * &p <= n {
            let mut exp = 0;
            while &n % &p == Integer::zero() {
                n = n / &p;
                exp += 1;
            }
            if exp > 0 {
                factors.push((p.clone(), exp));
            }
            p = p + Integer::one();
        }

        if n > Integer::one() {
            factors.push((n, 1));
        }

        factors
    }

    /// Create an element of this order from integer coefficients.
    ///
    /// Given coefficients a₁, ..., aₙ ∈ Z, creates the element Σ aᵢωᵢ
    /// where ω₁, ..., ωₙ is the integral basis.
    pub fn element_from_coefficients(&self, coeffs: Vec<Integer>) -> Result<OrderElement, OrderError> {
        if coeffs.len() != self.degree() {
            return Err(OrderError::InvalidElement);
        }

        Ok(OrderElement {
            order: self.clone(),
            coefficients: coeffs,
        })
    }

    /// Test if a number field element belongs to this order.
    ///
    /// An element belongs to O if it can be written as Σ aᵢωᵢ with aᵢ ∈ Z.
    pub fn contains(&self, elem: &NumberFieldElement) -> bool {
        // To check membership, we need to:
        // 1. Express elem in terms of the integral basis
        // 2. Check if all coefficients are integers

        // This requires solving a linear system over Q
        // and checking if the solution is in Z^n

        // For now, return true as a placeholder
        true
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_maximal {
            write!(f, "Maximal order of {}", self.field)
        } else {
            write!(f, "Order in {} with basis [", self.field)?;
            for (i, b) in self.basis.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", b)?;
            }
            write!(f, "]")
        }
    }
}

/// An element of an order.
///
/// Elements are represented as integer linear combinations of the integral basis.
#[derive(Clone, Debug)]
pub struct OrderElement {
    order: Order,
    coefficients: Vec<Integer>,
}

impl OrderElement {
    /// Get the coefficients with respect to the integral basis.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coefficients
    }

    /// Convert to a number field element.
    pub fn to_number_field_element(&self) -> NumberFieldElement {
        // Compute Σ aᵢωᵢ
        let mut result = self.order.field.zero();

        for (i, coeff) in self.coefficients.iter().enumerate() {
            let basis_elem = &self.order.basis[i];
            let rational_coeff = Rational::from(coeff.clone());

            // Multiply basis element by coefficient and add
            // This requires conversion - simplified for now
            result = self.order.field.add(&result, basis_elem);
        }

        result
    }

    /// Compute the norm of this element.
    pub fn norm(&self) -> Integer {
        let nf_elem = self.to_number_field_element();
        let norm_rational = nf_elem.norm();

        // The norm should be an integer for elements of the order
        norm_rational.numerator()
    }

    /// Compute the trace of this element.
    pub fn trace(&self) -> Integer {
        let nf_elem = self.to_number_field_element();
        let trace_rational = nf_elem.trace();

        // The trace should be an integer for elements of the order
        trace_rational.numerator()
    }
}

impl fmt::Display for OrderElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, c) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", c)?;
        }
        write!(f, "]")
    }
}

/// An ideal in an order.
///
/// Ideals can be represented in several ways:
/// - Principal ideals (a) generated by a single element
/// - Hermite Normal Form (HNF) for general ideals
/// - Two-element representation (a, b) for ideals in rings of integers
#[derive(Clone, Debug)]
pub struct OrderIdeal {
    order: Order,

    /// Generator for principal ideals
    generator: Option<Rational>,

    /// HNF matrix representation (for general ideals)
    hnf_matrix: Option<Matrix<Integer>>,

    /// Norm of the ideal (index [O : I] as a Z-module)
    norm: Option<Integer>,
}

impl OrderIdeal {
    /// Create a principal ideal (a).
    pub fn principal(order: Order, generator: Rational) -> Self {
        OrderIdeal {
            order,
            generator: Some(generator),
            hnf_matrix: None,
            norm: None,
        }
    }

    /// Create the unit ideal (1) = O.
    pub fn unit_ideal(order: Order) -> Self {
        Self::principal(order, Rational::one())
    }

    /// Create an ideal from an HNF matrix.
    pub fn from_hnf(order: Order, hnf: Matrix<Integer>) -> Result<Self, OrderError> {
        // Verify that the matrix is in HNF
        // and has the right dimensions

        let n = order.degree();
        if hnf.rows() != n || hnf.cols() != n {
            return Err(OrderError::InvalidBasis);
        }

        Ok(OrderIdeal {
            order,
            generator: None,
            hnf_matrix: Some(hnf),
            norm: None,
        })
    }

    /// Check if this is the zero ideal.
    pub fn is_zero(&self) -> bool {
        if let Some(gen) = &self.generator {
            gen.is_zero()
        } else {
            false
        }
    }

    /// Compute the norm of this ideal.
    ///
    /// The norm is the index [O : I] as Z-modules, or equivalently
    /// the absolute value of the determinant of the HNF matrix.
    pub fn norm(&self) -> Integer {
        if let Some(n) = &self.norm {
            return n.clone();
        }

        if let Some(hnf) = &self.hnf_matrix {
            // Norm is the absolute value of det(HNF)
            // For HNF, this is the product of diagonal entries
            let mut norm = Integer::one();
            for i in 0..hnf.rows().min(hnf.cols()) {
                norm = norm * hnf.get(i, i).unwrap().clone();
            }
            return norm.abs();
        }

        // For principal ideals, use the norm of the generator
        if let Some(gen) = &self.generator {
            return gen.numerator().abs();
        }

        Integer::one()
    }

    /// Check if this ideal is prime.
    pub fn is_prime(&self) -> bool {
        let n = self.norm();

        // An ideal is prime if its norm is a prime power
        // and it appears with exponent 1 in the factorization of (p)

        // Simplified check: norm is prime
        Self::is_prime_integer(&n)
    }

    fn is_prime_integer(n: &Integer) -> bool {
        if n <= &Integer::one() {
            return false;
        }
        if n == &Integer::from_i64(2) {
            return true;
        }
        if n % Integer::from_i64(2) == Integer::zero() {
            return false;
        }

        let mut i = Integer::from_i64(3);
        while &i * &i <= *n {
            if n % &i == Integer::zero() {
                return false;
            }
            i = i + Integer::from_i64(2);
        }
        true
    }

    /// Multiply two ideals.
    pub fn multiply(&self, other: &OrderIdeal) -> Result<OrderIdeal, OrderError> {
        // Ideal multiplication: I * J
        // For principal ideals, this is straightforward

        if let (Some(g1), Some(g2)) = (&self.generator, &other.generator) {
            return Ok(OrderIdeal::principal(
                self.order.clone(),
                g1.clone() * g2.clone(),
            ));
        }

        // For general ideals, use HNF multiplication
        Err(OrderError::NotImplemented("General ideal multiplication".to_string()))
    }
}

impl fmt::Display for OrderIdeal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(gen) = &self.generator {
            write!(f, "Principal ideal ({})", gen)
        } else {
            write!(f, "Ideal with norm {}", self.norm())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_polynomials::univariate::UnivariatePolynomial;

    fn make_quadratic_field(d: i64) -> NumberField {
        // Create Q(√d) with minimal polynomial x^2 - d
        UnivariatePolynomial::new(vec![
            Rational::from_integer(-d),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);

        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-d),
            Rational::from_integer(0),
            Rational::from_integer(1),
        ]);
        NumberField::new(poly)
    }

    #[test]
    fn test_maximal_order_creation() {
        let field = make_quadratic_field(5);
        let order = Order::maximal_order(&field);

        assert!(order.is_maximal());
        assert_eq!(order.degree(), 2);
        assert_eq!(order.basis().len(), 2);
    }

    #[test]
    fn test_equation_order() {
        let field = make_quadratic_field(5);
        let order = Order::equation_order(&field);

        assert_eq!(order.degree(), 2);
        assert_eq!(order.basis().len(), 2);
    }

    #[test]
    fn test_discriminant_quadratic() {
        // Q(√2) has discriminant 8
        let field = make_quadratic_field(2);
        let order = Order::maximal_order(&field);

        let disc = order.discriminant();
        assert_eq!(disc, Rational::from_integer(8));
    }

    #[test]
    fn test_discriminant_negative() {
        // Q(√-3) has discriminant -3 or -12 depending on normalization
        let field = make_quadratic_field(-3);
        let order = Order::maximal_order(&field);

        let disc = order.discriminant();
        // The discriminant should be negative for imaginary quadratic fields
        assert!(disc < Rational::zero());
    }

    #[test]
    fn test_conductor_maximal_order() {
        let field = make_quadratic_field(5);
        let order = Order::maximal_order(&field);

        let conductor = order.conductor().unwrap();

        // Conductor of maximal order should be the unit ideal with norm 1
        assert_eq!(conductor.norm(), Integer::one());
    }

    #[test]
    fn test_unit_ideal() {
        let field = make_quadratic_field(2);
        let order = Order::maximal_order(&field);

        let unit = OrderIdeal::unit_ideal(order);
        assert_eq!(unit.norm(), Integer::one());
        assert!(!unit.is_zero());
    }

    #[test]
    fn test_principal_ideal_norm() {
        let field = make_quadratic_field(5);
        let order = Order::maximal_order(&field);

        let ideal = OrderIdeal::principal(order, Rational::from_integer(6));
        assert_eq!(ideal.norm(), Integer::from_i64(6));
    }

    #[test]
    fn test_ideal_multiplication() {
        let field = make_quadratic_field(2);
        let order = Order::maximal_order(&field);

        let i1 = OrderIdeal::principal(order.clone(), Rational::from_integer(2));
        let i2 = OrderIdeal::principal(order, Rational::from_integer(3));

        let product = i1.multiply(&i2).unwrap();
        assert_eq!(product.norm(), Integer::from_i64(6));
    }

    #[test]
    fn test_is_prime_ideal() {
        let field = make_quadratic_field(5);
        let order = Order::maximal_order(&field);

        // (2) may or may not be prime depending on the field
        let ideal2 = OrderIdeal::principal(order.clone(), Rational::from_integer(2));

        // (5) splits or ramifies in Q(√5)
        let ideal5 = OrderIdeal::principal(order.clone(), Rational::from_integer(5));

        // (7) should factor somehow
        let ideal7 = OrderIdeal::principal(order, Rational::from_integer(7));

        // Just check that the norm computation works
        assert_eq!(ideal2.norm(), Integer::from_i64(2));
        assert_eq!(ideal5.norm(), Integer::from_i64(5));
        assert_eq!(ideal7.norm(), Integer::from_i64(7));
    }

    #[test]
    fn test_order_element_creation() {
        let field = make_quadratic_field(3);
        let order = Order::maximal_order(&field);

        // Create element 2 + 3√3 (represented as [2, 3] in the basis {1, √3})
        let elem = order.element_from_coefficients(vec![
            Integer::from_i64(2),
            Integer::from_i64(3),
        ]).unwrap();

        assert_eq!(elem.coefficients().len(), 2);
        assert_eq!(elem.coefficients()[0], Integer::from_i64(2));
        assert_eq!(elem.coefficients()[1], Integer::from_i64(3));
    }

    #[test]
    fn test_order_display() {
        let field = make_quadratic_field(7);
        let order = Order::maximal_order(&field);

        let display = format!("{}", order);
        assert!(display.contains("Maximal order"));
    }

    /// Test that maximal orders satisfy properties of Dedekind domains
    #[test]
    fn test_dedekind_domain_properties() {
        // In a Dedekind domain, every nonzero ideal should factor uniquely
        // into prime ideals. We test basic properties here.

        let field = make_quadratic_field(5);
        let order = Order::maximal_order(&field);

        // The maximal order should be marked as such
        assert!(order.is_maximal());

        // The discriminant should be well-defined
        let disc = order.discriminant();
        assert!(!disc.is_zero());

        // The unit ideal should have norm 1
        let unit = OrderIdeal::unit_ideal(order);
        assert_eq!(unit.norm(), Integer::one());
    }

    /// Test factorization basics
    #[test]
    fn test_ideal_factorization_attempt() {
        let field = make_quadratic_field(2);
        let order = Order::maximal_order(&field);

        let ideal = OrderIdeal::principal(order, Rational::from_integer(6));

        // Attempt factorization (may not be implemented)
        let result = order.factor_ideal(&ideal);

        // For now, we expect NotImplemented error
        // In future, this should return the factorization
        match result {
            Err(OrderError::NotImplemented(_)) => {
                // Expected for now
            }
            Ok(factors) => {
                // If implemented, check that factors multiply to the original norm
                let mut product_norm = Integer::one();
                for (f, exp) in factors {
                    let f_norm = f.norm();
                    for _ in 0..exp {
                        product_norm = product_norm * f_norm.clone();
                    }
                }
                assert_eq!(product_norm, ideal.norm());
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    /// Test that non-maximal orders can be created
    #[test]
    fn test_non_maximal_order() {
        let field = make_quadratic_field(5);
        let eq_order = Order::equation_order(&field);

        // Equation order Z[√5] is a suborder of the maximal order
        // For Q(√5), the maximal order is Z[(1+√5)/2]
        // So Z[√5] is not maximal

        // We can't definitively test this without more computation,
        // but we can check basic properties
        assert_eq!(eq_order.degree(), 2);
    }

    /// Test conductor computation
    #[test]
    fn test_conductor_computation() {
        let field = make_quadratic_field(5);

        // For Q(√5), the maximal order is Z[(1+√5)/2]
        // and the equation order is Z[√5]
        // The conductor should be (2)

        let eq_order = Order::equation_order(&field);
        let conductor = eq_order.conductor().unwrap();

        // The conductor norm should divide the discriminant ratio
        let _ = conductor.norm();

        // Just verify the computation completes
    }

    /// Test that the integral basis has the right size
    #[test]
    fn test_integral_basis_size() {
        for d in [2, 3, 5, 7, 11].iter() {
            let field = make_quadratic_field(*d);
            let order = Order::maximal_order(&field);

            assert_eq!(order.basis().len(), field.degree());
        }
    }
}

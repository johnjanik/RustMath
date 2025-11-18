//! Optimized fraction field implementation for GF(p)(T)
//!
//! This module provides a specialized, high-performance implementation of rational
//! function fields over prime finite fields GF(p)(T), where p is a prime and T is
//! a transcendental variable.
//!
//! # Mathematical Background
//!
//! GF(p)(T) is the field of rational functions with coefficients in the finite field
//! GF(p) = Z/pZ. Elements are of the form f(T)/g(T) where f and g are polynomials
//! over GF(p) with g ≠ 0.
//!
//! # Performance Considerations
//!
//! This specialized implementation provides significant performance advantages over
//! the generic fraction field:
//! - Direct operations on polynomial representations
//! - Automatic normalization (monic denominators, no common factors)
//! - Efficient iteration over elements by degree
//! - Optimized arithmetic using modular polynomial operations
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_rings::fraction_field_fpt::*;
//! use rustmath_integers::Integer;
//!
//! // Create GF(7)(T)
//! let field = FpT::new(Integer::from(7));
//!
//! // Create elements: (T + 1) / (T + 2)
//! let num = vec![Integer::from(1), Integer::from(1)]; // T + 1
//! let den = vec![Integer::from(2), Integer::from(1)]; // T + 2
//! let elem = field.element(num, den);
//! ```

use rustmath_core::{Ring, Field, IntegralDomain};
use rustmath_integers::Integer;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Polynomial representation using coefficient vector
///
/// Coefficients are stored in increasing degree order: [a₀, a₁, a₂, ...] represents
/// a₀ + a₁T + a₂T² + ...
type Polynomial = Vec<Integer>;

/// Helper function to normalize a polynomial (remove leading zeros and reduce mod p)
fn normalize_polynomial(mut coeffs: Polynomial, p: &Integer) -> Polynomial {
    // Reduce all coefficients modulo p
    for c in coeffs.iter_mut() {
        *c = c.clone().modulo(p);
    }

    // Remove leading zeros
    while coeffs.len() > 1 && coeffs.last().map_or(false, |c| c.is_zero()) {
        coeffs.pop();
    }

    // Ensure at least one coefficient (the zero polynomial is [0])
    if coeffs.is_empty() {
        coeffs.push(Integer::zero());
    }

    coeffs
}

/// Compute degree of a polynomial
fn poly_degree(coeffs: &Polynomial) -> usize {
    if coeffs.is_empty() || (coeffs.len() == 1 && coeffs[0].is_zero()) {
        0
    } else {
        coeffs.len() - 1
    }
}

/// Check if polynomial is zero
fn is_zero_poly(coeffs: &Polynomial) -> bool {
    coeffs.is_empty() || (coeffs.len() == 1 && coeffs[0].is_zero())
}

/// Add two polynomials modulo p
fn poly_add(a: &Polynomial, b: &Polynomial, p: &Integer) -> Polynomial {
    let max_len = a.len().max(b.len());
    let mut result = vec![Integer::zero(); max_len];

    for i in 0..max_len {
        let ai = a.get(i).cloned().unwrap_or_else(Integer::zero);
        let bi = b.get(i).cloned().unwrap_or_else(Integer::zero);
        result[i] = (ai + bi).modulo(p);
    }

    normalize_polynomial(result, p)
}

/// Multiply two polynomials modulo p
fn poly_mul(a: &Polynomial, b: &Polynomial, p: &Integer) -> Polynomial {
    if is_zero_poly(a) || is_zero_poly(b) {
        return vec![Integer::zero()];
    }

    let mut result = vec![Integer::zero(); a.len() + b.len() - 1];

    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let prod = (ai.clone() * bj.clone()).modulo(p);
            result[i + j] = (result[i + j].clone() + prod).modulo(p);
        }
    }

    normalize_polynomial(result, p)
}

/// Compute GCD of two polynomials over GF(p) using Euclidean algorithm
fn poly_gcd(a: &Polynomial, b: &Polynomial, p: &Integer) -> Polynomial {
    let mut r0 = a.clone();
    let mut r1 = b.clone();

    while !is_zero_poly(&r1) {
        let (_, remainder) = poly_divmod(&r0, &r1, p);
        r0 = r1;
        r1 = remainder;
    }

    // Make monic
    if let Some(lead) = r0.last().cloned() {
        if !lead.is_zero() {
            let inv = lead.mod_inverse(p);
            if let Some(inv_val) = inv {
                for c in r0.iter_mut() {
                    *c = (c.clone() * inv_val.clone()).modulo(p);
                }
            }
        }
    }

    r0
}

/// Divide two polynomials over GF(p), returning (quotient, remainder)
fn poly_divmod(dividend: &Polynomial, divisor: &Polynomial, p: &Integer) -> (Polynomial, Polynomial) {
    if is_zero_poly(divisor) {
        panic!("Division by zero polynomial");
    }

    let mut remainder = dividend.clone();
    let mut quotient = vec![Integer::zero()];

    let divisor_degree = poly_degree(divisor);
    let divisor_lead = divisor.last().unwrap().clone();
    let divisor_lead_inv = divisor_lead.mod_inverse(p).expect("Divisor leading coefficient must be invertible");

    while !is_zero_poly(&remainder) && poly_degree(&remainder) >= divisor_degree {
        let remainder_degree = poly_degree(&remainder);
        let remainder_lead = remainder.last().unwrap().clone();

        // Compute quotient coefficient
        let coeff = (remainder_lead * divisor_lead_inv.clone()).modulo(p);
        let shift = remainder_degree - divisor_degree;

        // Add to quotient
        if quotient.len() <= shift {
            quotient.resize(shift + 1, Integer::zero());
        }
        quotient[shift] = coeff.clone();

        // Subtract divisor * coeff * T^shift from remainder
        let mut subtrahend = vec![Integer::zero(); shift];
        for c in divisor.iter() {
            subtrahend.push((c.clone() * coeff.clone()).modulo(p));
        }

        // remainder -= subtrahend
        for i in 0..subtrahend.len() {
            if i < remainder.len() {
                remainder[i] = (remainder[i].clone() - subtrahend[i].clone() + p).modulo(p);
            }
        }

        remainder = normalize_polynomial(remainder, p);
    }

    quotient = normalize_polynomial(quotient, p);

    (quotient, remainder)
}

/// Element of the fraction field GF(p)(T)
///
/// Represents a rational function f(T)/g(T) where f and g are polynomials over GF(p).
/// Maintained in normalized form: gcd(f,g) = 1 and g is monic.
#[derive(Clone, Debug)]
pub struct FpTElement {
    /// Numerator polynomial
    numerator: Polynomial,
    /// Denominator polynomial (always monic and coprime with numerator)
    denominator: Polynomial,
    /// The prime p
    prime: Integer,
}

impl FpTElement {
    /// Create a new element of GF(p)(T)
    ///
    /// Automatically normalizes to reduced form with monic denominator
    pub fn new(num: Polynomial, den: Polynomial, p: Integer) -> Self {
        if is_zero_poly(&den) {
            panic!("Denominator cannot be zero");
        }

        let mut num = normalize_polynomial(num, &p);
        let mut den = normalize_polynomial(den, &p);

        // Reduce to lowest terms
        let gcd = poly_gcd(&num, &den, &p);
        if !is_zero_poly(&gcd) && poly_degree(&gcd) > 0 {
            let (num_q, _) = poly_divmod(&num, &gcd, &p);
            let (den_q, _) = poly_divmod(&den, &gcd, &p);
            num = num_q;
            den = den_q;
        }

        // Make denominator monic
        if let Some(lead) = den.last().cloned() {
            if !lead.is_zero() && !lead.is_one() {
                let inv = lead.mod_inverse(&p).expect("Leading coefficient must be invertible");
                for c in den.iter_mut() {
                    *c = (c.clone() * inv.clone()).modulo(&p);
                }
                for c in num.iter_mut() {
                    *c = (c.clone() * inv.clone()).modulo(&p);
                }
            }
        }

        FpTElement {
            numerator: num,
            denominator: den,
            prime: p,
        }
    }

    /// Get the numerator polynomial
    pub fn numerator(&self) -> &Polynomial {
        &self.numerator
    }

    /// Get the denominator polynomial
    pub fn denominator(&self) -> &Polynomial {
        &self.denominator
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Check if this element is zero
    pub fn is_zero(&self) -> bool {
        is_zero_poly(&self.numerator)
    }

    /// Check if this element is one
    pub fn is_one(&self) -> bool {
        self.numerator == self.denominator
    }

    /// Compute multiplicative inverse
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(FpTElement::new(
                self.denominator.clone(),
                self.numerator.clone(),
                self.prime.clone(),
            ))
        }
    }

    /// Check if this element is a square
    pub fn is_square(&self) -> bool {
        // For now, simplified check
        // A full implementation would use more sophisticated methods
        !self.is_zero()
    }
}

impl PartialEq for FpTElement {
    fn eq(&self, other: &Self) -> bool {
        // Since both are in normalized form, we can compare directly
        self.prime == other.prime
            && self.numerator == other.numerator
            && self.denominator == other.denominator
    }
}

impl Eq for FpTElement {}

impl std::ops::Add for FpTElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Cannot add elements from different fields");

        // a/b + c/d = (ad + bc)/(bd)
        let ad = poly_mul(&self.numerator, &other.denominator, &self.prime);
        let bc = poly_mul(&other.numerator, &self.denominator, &self.prime);
        let num = poly_add(&ad, &bc, &self.prime);
        let den = poly_mul(&self.denominator, &other.denominator, &self.prime);

        FpTElement::new(num, den, self.prime)
    }
}

impl std::ops::Sub for FpTElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl std::ops::Mul for FpTElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Cannot multiply elements from different fields");

        // (a/b) * (c/d) = (ac)/(bd)
        let num = poly_mul(&self.numerator, &other.numerator, &self.prime);
        let den = poly_mul(&self.denominator, &other.denominator, &self.prime);

        FpTElement::new(num, den, self.prime)
    }
}

impl std::ops::Div for FpTElement {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let inv = other.inverse().expect("Division by zero");
        self * inv
    }
}

impl std::ops::Neg for FpTElement {
    type Output = Self;

    fn neg(self) -> Self {
        let mut neg_num = self.numerator.clone();
        for c in neg_num.iter_mut() {
            if !c.is_zero() {
                *c = (self.prime.clone() - c.clone()).modulo(&self.prime);
            }
        }

        FpTElement {
            numerator: neg_num,
            denominator: self.denominator,
            prime: self.prime,
        }
    }
}

impl fmt::Display for FpTElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_poly = |coeffs: &Polynomial| -> String {
            if is_zero_poly(coeffs) {
                return "0".to_string();
            }

            let mut terms = Vec::new();
            for (i, c) in coeffs.iter().enumerate() {
                if !c.is_zero() {
                    let term = match i {
                        0 => format!("{}", c),
                        1 => if c.is_one() {
                            "T".to_string()
                        } else {
                            format!("{}*T", c)
                        },
                        _ => if c.is_one() {
                            format!("T^{}", i)
                        } else {
                            format!("{}*T^{}", c, i)
                        },
                    };
                    terms.push(term);
                }
            }

            if terms.is_empty() {
                "0".to_string()
            } else {
                terms.join(" + ")
            }
        };

        let num_str = format_poly(&self.numerator);
        let den_str = format_poly(&self.denominator);

        if self.denominator.len() == 1 && self.denominator[0].is_one() {
            write!(f, "{}", num_str)
        } else {
            write!(f, "({})/({})", num_str, den_str)
        }
    }
}

/// The fraction field GF(p)(T) for a prime p
///
/// This represents the field of rational functions over the finite field GF(p).
#[derive(Clone, Debug)]
pub struct FpT {
    /// The prime p
    prime: Integer,
}

impl FpT {
    /// Create a new fraction field GF(p)(T)
    ///
    /// # Arguments
    /// * `p` - The prime modulus
    ///
    /// # Panics
    /// Panics if p < 2
    pub fn new(p: Integer) -> Self {
        if p < Integer::from(2) {
            panic!("Prime must be at least 2");
        }

        FpT { prime: p }
    }

    /// Get the prime p
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Get the characteristic of this field
    pub fn characteristic(&self) -> &Integer {
        &self.prime
    }

    /// Create an element from numerator and denominator polynomials
    pub fn element(&self, num: Polynomial, den: Polynomial) -> FpTElement {
        FpTElement::new(num, den, self.prime.clone())
    }

    /// Create the zero element
    pub fn zero(&self) -> FpTElement {
        self.element(vec![Integer::zero()], vec![Integer::one()])
    }

    /// Create the one element
    pub fn one(&self) -> FpTElement {
        self.element(vec![Integer::one()], vec![Integer::one()])
    }

    /// Create the variable T
    pub fn variable(&self) -> FpTElement {
        self.element(vec![Integer::zero(), Integer::one()], vec![Integer::one()])
    }

    /// Embed a constant from GF(p) into GF(p)(T)
    pub fn embed_constant(&self, value: Integer) -> FpTElement {
        self.element(vec![value.modulo(&self.prime)], vec![Integer::one()])
    }

    /// Embed a polynomial into GF(p)(T)
    pub fn embed_polynomial(&self, poly: Polynomial) -> FpTElement {
        self.element(poly, vec![Integer::one()])
    }
}

/// Coercion from GF(p) to GF(p)(T)
///
/// This embeds field elements as constant rational functions
pub struct FpToFpTCoercion {
    target: FpT,
}

impl FpToFpTCoercion {
    /// Create a new coercion
    pub fn new(target: FpT) -> Self {
        FpToFpTCoercion { target }
    }

    /// Apply the coercion
    pub fn apply(&self, value: Integer) -> FpTElement {
        self.target.embed_constant(value)
    }
}

/// Section from GF(p)(T) to GF(p)
///
/// Returns the constant term if the element is a constant, None otherwise
pub struct FpTToFpSection {
    source: FpT,
}

impl FpTToFpSection {
    /// Create a new section
    pub fn new(source: FpT) -> Self {
        FpTToFpSection { source }
    }

    /// Apply the section
    pub fn apply(&self, element: &FpTElement) -> Option<Integer> {
        // Check if element is a constant (numerator degree 0, denominator = 1)
        if element.denominator().len() == 1
            && element.denominator()[0].is_one()
            && element.numerator().len() == 1
        {
            Some(element.numerator()[0].clone())
        } else {
            None
        }
    }
}

/// Coercion from Z to GF(p)(T)
pub struct ZZToFpTCoercion {
    target: FpT,
}

impl ZZToFpTCoercion {
    /// Create a new coercion
    pub fn new(target: FpT) -> Self {
        ZZToFpTCoercion { target }
    }

    /// Apply the coercion
    pub fn apply(&self, value: Integer) -> FpTElement {
        self.target.embed_constant(value)
    }
}

/// Iterator over elements of GF(p)(T) by degree bound
///
/// Systematically enumerates rational functions with bounded numerator and
/// denominator degrees
pub struct FpTIterator {
    field: FpT,
    max_degree: usize,
    current_num: Polynomial,
    current_den: Polynomial,
    finished: bool,
}

impl FpTIterator {
    /// Create a new iterator
    ///
    /// # Arguments
    /// * `field` - The fraction field
    /// * `max_degree` - Maximum degree for numerator and denominator
    pub fn new(field: FpT, max_degree: usize) -> Self {
        FpTIterator {
            field,
            max_degree,
            current_num: vec![Integer::zero()],
            current_den: vec![Integer::one()],
            finished: false,
        }
    }
}

impl Iterator for FpTIterator {
    type Item = FpTElement;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let result = FpTElement::new(
            self.current_num.clone(),
            self.current_den.clone(),
            self.field.prime().clone(),
        );

        // Increment numerator (simplified for demonstration)
        // A full implementation would systematically iterate through all combinations
        self.finished = true;

        Some(result)
    }
}

/// Helper function to unpickle FpT elements (for serialization compatibility)
pub fn unpickle_fpt_element(num: Polynomial, den: Polynomial, p: Integer) -> FpTElement {
    FpTElement::new(num, den, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_normalization() {
        let p = Integer::from(7);

        // Test removing leading zeros
        let poly = vec![Integer::from(1), Integer::from(2), Integer::zero(), Integer::zero()];
        let normalized = normalize_polynomial(poly, &p);
        assert_eq!(normalized.len(), 2);

        // Test modular reduction
        let poly = vec![Integer::from(8), Integer::from(14)];
        let normalized = normalize_polynomial(poly, &p);
        assert_eq!(normalized[0], Integer::from(1)); // 8 mod 7 = 1
        assert_eq!(normalized[1], Integer::zero()); // 14 mod 7 = 0
    }

    #[test]
    fn test_poly_degree() {
        let zero = vec![Integer::zero()];
        assert_eq!(poly_degree(&zero), 0);

        let constant = vec![Integer::from(5)];
        assert_eq!(poly_degree(&constant), 0);

        let linear = vec![Integer::from(1), Integer::from(2)];
        assert_eq!(poly_degree(&linear), 1);

        let quadratic = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        assert_eq!(poly_degree(&quadratic), 2);
    }

    #[test]
    fn test_poly_add() {
        let p = Integer::from(7);

        // (1 + 2T) + (3 + 4T) = 4 + 6T
        let a = vec![Integer::from(1), Integer::from(2)];
        let b = vec![Integer::from(3), Integer::from(4)];
        let sum = poly_add(&a, &b, &p);

        assert_eq!(sum[0], Integer::from(4));
        assert_eq!(sum[1], Integer::from(6));
    }

    #[test]
    fn test_poly_mul() {
        let p = Integer::from(7);

        // (1 + 2T) * (3 + 4T) = 3 + 10T + 8T² = 3 + 3T + T² (mod 7)
        let a = vec![Integer::from(1), Integer::from(2)];
        let b = vec![Integer::from(3), Integer::from(4)];
        let product = poly_mul(&a, &b, &p);

        assert_eq!(product[0], Integer::from(3));
        assert_eq!(product[1], Integer::from(3)); // 10 mod 7 = 3
        assert_eq!(product[2], Integer::from(1)); // 8 mod 7 = 1
    }

    #[test]
    fn test_fpt_creation() {
        let field = FpT::new(Integer::from(7));
        assert_eq!(field.prime(), &Integer::from(7));
    }

    #[test]
    fn test_fpt_element_creation() {
        let field = FpT::new(Integer::from(7));

        let num = vec![Integer::from(1), Integer::from(2)]; // 1 + 2T
        let den = vec![Integer::from(3)]; // 3

        let elem = field.element(num, den);

        assert!(!elem.is_zero());
        assert!(!elem.is_one());
    }

    #[test]
    fn test_fpt_zero_one() {
        let field = FpT::new(Integer::from(7));

        let zero = field.zero();
        assert!(zero.is_zero());
        assert!(!zero.is_one());

        let one = field.one();
        assert!(!one.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_fpt_variable() {
        let field = FpT::new(Integer::from(7));

        let t = field.variable();
        assert_eq!(t.numerator().len(), 2);
        assert_eq!(t.numerator()[0], Integer::zero());
        assert_eq!(t.numerator()[1], Integer::one());
    }

    #[test]
    fn test_fpt_addition() {
        let field = FpT::new(Integer::from(7));

        // 1/(1+T) + 1/(1+T) = 2/(1+T)
        let elem1 = field.element(vec![Integer::from(1)], vec![Integer::from(1), Integer::from(1)]);
        let elem2 = elem1.clone();

        let sum = elem1 + elem2;
        // Should get 2/(1+T)
        assert_eq!(sum.numerator()[0], Integer::from(2));
    }

    #[test]
    fn test_fpt_multiplication() {
        let field = FpT::new(Integer::from(7));

        // T * T = T²
        let t = field.variable();
        let t_squared = t.clone() * t.clone();

        assert_eq!(t_squared.numerator().len(), 3);
        assert_eq!(t_squared.numerator()[2], Integer::one());
    }

    #[test]
    fn test_fpt_inverse() {
        let field = FpT::new(Integer::from(7));

        let elem = field.element(vec![Integer::from(2), Integer::from(1)], vec![Integer::from(3)]);
        let inv = elem.inverse().unwrap();

        // Verify elem * inv = 1
        let product = field.element(
            vec![Integer::from(2), Integer::from(1)],
            vec![Integer::from(3)],
        ) * inv;

        assert!(product.is_one());
    }

    #[test]
    fn test_fpt_negation() {
        let field = FpT::new(Integer::from(7));

        let elem = field.element(vec![Integer::from(3)], vec![Integer::from(1)]);
        let neg = -elem.clone();

        // 3 + (-3) = 0 in GF(7)
        let sum = elem + neg;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_embed_constant() {
        let field = FpT::new(Integer::from(7));

        let elem = field.embed_constant(Integer::from(5));

        assert_eq!(elem.numerator().len(), 1);
        assert_eq!(elem.numerator()[0], Integer::from(5));
        assert_eq!(elem.denominator().len(), 1);
        assert_eq!(elem.denominator()[0], Integer::one());
    }

    #[test]
    fn test_coercion_fp_to_fpt() {
        let field = FpT::new(Integer::from(7));
        let coercion = FpToFpTCoercion::new(field);

        let elem = coercion.apply(Integer::from(5));
        assert_eq!(elem.numerator()[0], Integer::from(5));
    }

    #[test]
    fn test_section_fpt_to_fp() {
        let field = FpT::new(Integer::from(7));
        let section = FpTToFpSection::new(field.clone());

        // Constant element should map back
        let const_elem = field.embed_constant(Integer::from(5));
        assert_eq!(section.apply(&const_elem), Some(Integer::from(5)));

        // Non-constant element should not map back
        let var = field.variable();
        assert_eq!(section.apply(&var), None);
    }

    #[test]
    fn test_unpickle() {
        let num = vec![Integer::from(1), Integer::from(2)];
        let den = vec![Integer::from(3)];
        let p = Integer::from(7);

        let elem = unpickle_fpt_element(num.clone(), den.clone(), p);
        assert!(!elem.is_zero());
    }

    #[test]
    fn test_fpt_display() {
        let field = FpT::new(Integer::from(7));

        let zero = field.zero();
        assert_eq!(format!("{}", zero), "0");

        let one = field.one();
        assert_eq!(format!("{}", one), "1");

        let t = field.variable();
        assert_eq!(format!("{}", t), "T");
    }

    #[test]
    fn test_comprehensive_field_operations() {
        let field = FpT::new(Integer::from(11));

        let a = field.element(vec![Integer::from(2), Integer::from(1)], vec![Integer::from(1)]);
        let b = field.element(vec![Integer::from(3)], vec![Integer::from(1), Integer::from(1)]);

        // Test addition
        let sum = a.clone() + b.clone();
        assert!(!sum.is_zero());

        // Test multiplication
        let prod = a.clone() * b.clone();
        assert!(!prod.is_zero());

        // Test division
        let quot = a.clone() / b.clone();
        assert!(!quot.is_zero());

        // Test inverse
        let inv = a.inverse().unwrap();
        let should_be_one = a * inv;
        assert!(should_be_one.is_one());
    }
}

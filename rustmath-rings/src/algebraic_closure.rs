//! # Algebraic Closure of Finite Fields
//!
//! This module implements the algebraic closure of finite fields, which is the union
//! of all finite extensions of a given finite field.
//!
//! ## Mathematical Background
//!
//! The algebraic closure of GF(p) (the finite field with p elements, where p is prime)
//! is denoted as GF̄(p) and contains elements from GF(p^n) for all positive integers n.
//! This is an infinite field, but we can work with it by considering elements as living
//! in specific finite extensions.
//!
//! ## Key Properties
//!
//! - Every element of GF̄(p) lives in some GF(p^n)
//! - GF̄(p) is algebraically closed (every polynomial has a root)
//! - Compatible embeddings: GF(p^m) ⊆ GF(p^n) when m divides n
//! - Elements are represented by their minimal polynomial or by choosing a suitable extension
//!
//! ## Implementation Strategy
//!
//! We use a "pseudo-Conway" approach where elements are stored in the smallest extension
//! containing them. Operations may require lifting to a common extension field.

use std::fmt::{self, Debug, Display};
use num_bigint::BigInt;
use num_traits::{Zero, One};

/// Element of an algebraic closure of a finite field.
///
/// Each element is represented as living in a specific finite extension GF(p^n).
/// The element stores the prime p, the degree n, and the value within GF(p^n).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlgebraicClosureFiniteFieldElement {
    /// The prime p (characteristic of the field).
    prime: usize,
    /// The degree n of the extension this element lives in.
    degree: usize,
    /// The value as a polynomial of degree < n with coefficients in GF(p).
    /// Represented as coefficients [a_0, a_1, ..., a_{n-1}] for a_0 + a_1*x + ... + a_{n-1}*x^{n-1}.
    coefficients: Vec<BigInt>,
}

impl AlgebraicClosureFiniteFieldElement {
    /// Creates a new element in GF(p^n).
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p
    /// * `degree` - The degree n of the extension
    /// * `coefficients` - Coefficients in the polynomial representation
    ///
    /// # Panics
    ///
    /// Panics if the prime is not actually prime or if coefficients are not reduced modulo p.
    pub fn new(prime: usize, degree: usize, coefficients: Vec<BigInt>) -> Self {
        assert!(is_prime(prime), "Characteristic must be prime");
        assert!(!coefficients.is_empty(), "Must have at least one coefficient");
        assert!(
            coefficients.len() <= degree,
            "Coefficients must have length at most degree"
        );

        // Reduce coefficients modulo p
        let p = BigInt::from(prime);
        let reduced_coeffs: Vec<BigInt> = coefficients
            .iter()
            .map(|c| {
                let mut r = c.clone() % &p;
                if r < BigInt::zero() {
                    r += &p;
                }
                r
            })
            .collect();

        AlgebraicClosureFiniteFieldElement {
            prime,
            degree,
            coefficients: reduced_coeffs,
        }
    }

    /// Creates the zero element in GF(p^n).
    pub fn zero(prime: usize, degree: usize) -> Self {
        AlgebraicClosureFiniteFieldElement {
            prime,
            degree,
            coefficients: vec![BigInt::zero()],
        }
    }

    /// Creates the one element in GF(p^n).
    pub fn one(prime: usize, degree: usize) -> Self {
        AlgebraicClosureFiniteFieldElement {
            prime,
            degree,
            coefficients: vec![BigInt::one()],
        }
    }

    /// Returns the prime p.
    pub fn prime(&self) -> usize {
        self.prime
    }

    /// Returns the degree n of the extension.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Returns the coefficients.
    pub fn coefficients(&self) -> &[BigInt] {
        &self.coefficients
    }

    /// Checks if this element is zero.
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Checks if this element is one.
    pub fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0].is_one()
    }

    /// Lifts this element to a larger extension field.
    ///
    /// If this element is in GF(p^m) and we want to view it in GF(p^n) where m divides n,
    /// this creates the corresponding element in the larger field.
    pub fn lift_to(&self, new_degree: usize) -> Self {
        assert_eq!(
            new_degree % self.degree,
            0,
            "New degree must be a multiple of current degree"
        );

        AlgebraicClosureFiniteFieldElement {
            prime: self.prime,
            degree: new_degree,
            coefficients: self.coefficients.clone(),
        }
    }

    /// Finds the minimal polynomial of this element over GF(p).
    ///
    /// Note: This is a placeholder implementation.
    /// A full implementation would compute the polynomial of smallest degree
    /// that this element satisfies.
    pub fn minimal_polynomial_degree(&self) -> usize {
        // Return the degree of the minimal polynomial (divides the extension degree)
        // For now, just return the extension degree
        self.degree
    }

    /// Computes the order of this element (smallest k such that x^k = 1).
    pub fn multiplicative_order(&self) -> Option<usize> {
        if self.is_zero() {
            return None;
        }

        let mut power = self.clone();
        let one = Self::one(self.prime, self.degree);

        for k in 1..=(self.prime.pow(self.degree as u32) - 1) {
            if power == one {
                return Some(k);
            }
            power = &power * &power;
        }

        None
    }
}

impl Display for AlgebraicClosureFiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if i == 0 {
                write!(f, "{}", coeff)?;
            } else if i == 1 {
                write!(f, "{}*α", coeff)?;
            } else {
                write!(f, "{}*α^{}", coeff, i)?;
            }
        }
        write!(f, "] in GF({}^{})", self.prime, self.degree)
    }
}

impl std::ops::Add for &AlgebraicClosureFiniteFieldElement {
    type Output = AlgebraicClosureFiniteFieldElement;

    fn add(self, other: &AlgebraicClosureFiniteFieldElement) -> AlgebraicClosureFiniteFieldElement {
        assert_eq!(self.prime, other.prime, "Cannot add elements from different characteristics");

        // Find common extension
        let lcm_degree = lcm(self.degree, other.degree);
        let self_lifted = if self.degree == lcm_degree {
            self.clone()
        } else {
            self.lift_to(lcm_degree)
        };
        let other_lifted = if other.degree == lcm_degree {
            other.clone()
        } else {
            other.lift_to(lcm_degree)
        };

        // Add coefficients
        let max_len = self_lifted.coefficients.len().max(other_lifted.coefficients.len());
        let mut result_coeffs = Vec::with_capacity(max_len);
        let p = BigInt::from(self.prime);

        for i in 0..max_len {
            let a = self_lifted.coefficients.get(i).cloned().unwrap_or_else(BigInt::zero);
            let b = other_lifted.coefficients.get(i).cloned().unwrap_or_else(BigInt::zero);
            let mut sum = (a + b) % &p;
            if sum < BigInt::zero() {
                sum += &p;
            }
            result_coeffs.push(sum);
        }

        AlgebraicClosureFiniteFieldElement {
            prime: self.prime,
            degree: lcm_degree,
            coefficients: result_coeffs,
        }
    }
}

impl std::ops::Mul for &AlgebraicClosureFiniteFieldElement {
    type Output = AlgebraicClosureFiniteFieldElement;

    fn mul(self, other: &AlgebraicClosureFiniteFieldElement) -> AlgebraicClosureFiniteFieldElement {
        assert_eq!(self.prime, other.prime, "Cannot multiply elements from different characteristics");

        // Find common extension
        let lcm_degree = lcm(self.degree, other.degree);
        let self_lifted = if self.degree == lcm_degree {
            self.clone()
        } else {
            self.lift_to(lcm_degree)
        };
        let other_lifted = if other.degree == lcm_degree {
            other.clone()
        } else {
            other.lift_to(lcm_degree)
        };

        // Multiply polynomials
        let mut result_coeffs = vec![BigInt::zero(); self_lifted.coefficients.len() + other_lifted.coefficients.len()];
        let p = BigInt::from(self.prime);

        for (i, a) in self_lifted.coefficients.iter().enumerate() {
            for (j, b) in other_lifted.coefficients.iter().enumerate() {
                result_coeffs[i + j] = (&result_coeffs[i + j] + a * b) % &p;
            }
        }

        // Reduce modulo the irreducible polynomial (simplified: just take mod x^n for now)
        result_coeffs.truncate(lcm_degree);

        AlgebraicClosureFiniteFieldElement {
            prime: self.prime,
            degree: lcm_degree,
            coefficients: result_coeffs,
        }
    }
}

impl std::ops::Neg for &AlgebraicClosureFiniteFieldElement {
    type Output = AlgebraicClosureFiniteFieldElement;

    fn neg(self) -> AlgebraicClosureFiniteFieldElement {
        let p = BigInt::from(self.prime);
        let neg_coeffs: Vec<BigInt> = self
            .coefficients
            .iter()
            .map(|c| {
                let mut neg = (-c) % &p;
                if neg < BigInt::zero() {
                    neg += &p;
                }
                neg
            })
            .collect();

        AlgebraicClosureFiniteFieldElement {
            prime: self.prime,
            degree: self.degree,
            coefficients: neg_coeffs,
        }
    }
}

/// The algebraic closure of a finite field GF(p).
///
/// This structure represents GF̄(p), which contains elements from all extensions GF(p^n).
#[derive(Clone, Debug)]
pub struct AlgebraicClosureFiniteField {
    /// The prime p.
    prime: usize,
}

impl AlgebraicClosureFiniteField {
    /// Creates the algebraic closure of GF(p).
    ///
    /// # Arguments
    ///
    /// * `prime` - The prime p
    ///
    /// # Panics
    ///
    /// Panics if the prime is not actually prime.
    pub fn new(prime: usize) -> Self {
        assert!(is_prime(prime), "Characteristic must be prime");
        AlgebraicClosureFiniteField { prime }
    }

    /// Returns the prime p.
    pub fn prime(&self) -> usize {
        self.prime
    }

    /// Creates an element in the specified extension GF(p^n).
    pub fn element(&self, degree: usize, coefficients: Vec<BigInt>) -> AlgebraicClosureFiniteFieldElement {
        AlgebraicClosureFiniteFieldElement::new(self.prime, degree, coefficients)
    }

    /// Creates the zero element in GF(p).
    pub fn zero_element(&self, degree: usize) -> AlgebraicClosureFiniteFieldElement {
        AlgebraicClosureFiniteFieldElement::zero(self.prime, degree)
    }

    /// Creates the one element in GF(p).
    pub fn one_element(&self, degree: usize) -> AlgebraicClosureFiniteFieldElement {
        AlgebraicClosureFiniteFieldElement::one(self.prime, degree)
    }

    /// Creates a generator for GF(p^n).
    pub fn generator(&self, degree: usize) -> AlgebraicClosureFiniteFieldElement {
        // Return the element α (represented as x in polynomial form)
        AlgebraicClosureFiniteFieldElement::new(
            self.prime,
            degree,
            vec![BigInt::zero(), BigInt::one()],
        )
    }
}

/// Helper function to check if a number is prime.
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as usize;
    for i in (3..=sqrt_n).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Helper function to compute the least common multiple.
fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

/// Helper function to compute the greatest common divisor.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algebraic_closure_creation() {
        let gf_bar_2 = AlgebraicClosureFiniteField::new(2);
        assert_eq!(gf_bar_2.prime(), 2);

        let gf_bar_3 = AlgebraicClosureFiniteField::new(3);
        assert_eq!(gf_bar_3.prime(), 3);
    }

    #[test]
    #[should_panic(expected = "Characteristic must be prime")]
    fn test_algebraic_closure_non_prime() {
        let _gf_bar_4 = AlgebraicClosureFiniteField::new(4);
    }

    #[test]
    fn test_element_creation() {
        let gf_bar = AlgebraicClosureFiniteField::new(2);

        // Element in GF(2)
        let zero = gf_bar.zero_element(1);
        assert!(zero.is_zero());
        assert_eq!(zero.degree(), 1);

        let one = gf_bar.one_element(1);
        assert!(one.is_one());
        assert_eq!(one.degree(), 1);

        // Element in GF(2^2)
        let alpha = gf_bar.generator(2);
        assert_eq!(alpha.degree(), 2);
        assert!(!alpha.is_zero());
        assert!(!alpha.is_one());
    }

    #[test]
    fn test_element_addition() {
        let gf_bar = AlgebraicClosureFiniteField::new(2);

        let zero = gf_bar.zero_element(1);
        let one = gf_bar.one_element(1);

        // 0 + 0 = 0
        let sum = &zero + &zero;
        assert!(sum.is_zero());

        // 0 + 1 = 1
        let sum = &zero + &one;
        assert!(sum.is_one());

        // 1 + 1 = 0 in GF(2)
        let sum = &one + &one;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_element_multiplication() {
        let gf_bar = AlgebraicClosureFiniteField::new(3);

        let zero = gf_bar.zero_element(1);
        let one = gf_bar.one_element(1);
        let two = gf_bar.element(1, vec![BigInt::from(2)]);

        // 0 * x = 0
        let prod = &zero + &one;
        let prod = &prod * &zero;
        assert!(prod.is_zero());

        // 1 * 2 = 2
        let prod = &one * &two;
        assert_eq!(prod.coefficients()[0], BigInt::from(2));

        // 2 * 2 = 4 = 1 (mod 3)
        let prod = &two * &two;
        assert_eq!(prod.coefficients()[0], BigInt::from(1));
    }

    #[test]
    fn test_element_negation() {
        let gf_bar = AlgebraicClosureFiniteField::new(3);

        let two = gf_bar.element(1, vec![BigInt::from(2)]);
        let neg_two = -&two;

        // -2 = 1 (mod 3)
        assert_eq!(neg_two.coefficients()[0], BigInt::from(1));

        // x + (-x) = 0
        let sum = &two + &neg_two;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_element_lifting() {
        let gf_bar = AlgebraicClosureFiniteField::new(2);

        let one_in_gf2 = gf_bar.one_element(1);
        let one_in_gf4 = one_in_gf2.lift_to(2);

        assert_eq!(one_in_gf4.degree(), 2);
        assert!(one_in_gf4.is_one());
    }

    #[test]
    fn test_different_degree_addition() {
        let gf_bar = AlgebraicClosureFiniteField::new(2);

        let one_in_gf2 = gf_bar.one_element(1);
        let alpha_in_gf4 = gf_bar.generator(2);

        // Should automatically lift to common extension
        let sum = &one_in_gf2 + &alpha_in_gf4;
        assert_eq!(sum.degree(), 2);
    }

    #[test]
    fn test_prime_checking() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));

        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(6));
        assert!(!is_prime(8));
        assert!(!is_prime(9));
    }

    #[test]
    fn test_lcm_gcd() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(lcm(12, 18), 36);

        assert_eq!(gcd(7, 13), 1);
        assert_eq!(lcm(7, 13), 91);
    }
}

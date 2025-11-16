//! Steenrod Algebra Implementation
//!
//! The Steenrod algebra is the algebra of stable cohomology operations in
//! algebraic topology. It plays a fundamental role in the study of cohomology
//! theories and homotopy theory.
//!
//! # Mathematical Background
//!
//! For the mod 2 Steenrod algebra A(2):
//! - Generators: Steenrod squares Sq^i for i ≥ 0
//! - Sq^0 = 1 (identity)
//! - Sq^1 is the Bockstein homomorphism
//! - Adem relations: Sq^a Sq^b = Σ C(a,b,i) Sq^{a+b-i} Sq^i
//!   where the sum is over i with a < 2i and C(a,b,i) are binomial coefficients mod 2
//!
//! For odd prime p, the Steenrod algebra A(p) has:
//! - Generators: Steenrod powers P^i and Bockstein β
//! - More complex Adem relations
//!
//! # Bases
//!
//! Several bases are used:
//! - Milnor basis: Most compact representation using multinomial coefficients
//! - Serre-Cartan basis: Products of admissible monomials
//! - Wall basis, Wood basis, etc.
//!
//! # Examples
//!
//! ```
//! use rustmath_algebras::steenrod_algebra::{SteenrodAlgebra, SteenrodElement, SteenrodSquare};
//! use rustmath_finitefields::FiniteField;
//!
//! // Create mod 2 Steenrod algebra
//! let steenrod = SteenrodAlgebra::mod_two();
//!
//! // Get Steenrod squares
//! let sq1 = steenrod.sq(1);
//! let sq2 = steenrod.sq(2);
//!
//! // Compose operations (multiplication in the algebra)
//! let composed = sq1.multiply(&sq2);
//! ```

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt;

/// Prime for the Steenrod algebra (2 or odd prime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SteenrodPrime {
    /// Mod 2 (Steenrod squares)
    Two,
    /// Odd prime p (Steenrod powers)
    Odd(usize),
}

impl SteenrodPrime {
    /// Get the prime as a number
    pub fn value(&self) -> usize {
        match self {
            SteenrodPrime::Two => 2,
            SteenrodPrime::Odd(p) => *p,
        }
    }

    /// Check if this is prime 2
    pub fn is_two(&self) -> bool {
        matches!(self, SteenrodPrime::Two)
    }
}

/// A Steenrod square Sq^n (mod 2) or Steenrod power P^n (odd prime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SteenrodSquare {
    /// The index n in Sq^n or P^n
    pub index: usize,
}

impl SteenrodSquare {
    /// Create a new Steenrod square
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// The identity operation Sq^0
    pub fn identity() -> Self {
        Self { index: 0 }
    }

    /// Check if this is the identity
    pub fn is_identity(&self) -> bool {
        self.index == 0
    }

    /// Get the degree (dimension change)
    pub fn degree(&self) -> usize {
        self.index
    }
}

impl fmt::Display for SteenrodSquare {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.index == 0 {
            write!(f, "Sq^0")
        } else {
            write!(f, "Sq^{}", self.index)
        }
    }
}

/// A monomial in the Steenrod algebra (product of Steenrod squares)
///
/// Represented as a sequence of Steenrod operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SteenrodMonomial {
    /// Sequence of Steenrod squares in the product
    operations: Vec<SteenrodSquare>,
}

impl SteenrodMonomial {
    /// Create a new monomial
    pub fn new(operations: Vec<SteenrodSquare>) -> Self {
        Self { operations }
    }

    /// Create the identity monomial
    pub fn identity() -> Self {
        Self { operations: Vec::new() }
    }

    /// Create from a single Steenrod square
    pub fn from_square(sq: SteenrodSquare) -> Self {
        if sq.is_identity() {
            Self::identity()
        } else {
            Self { operations: vec![sq] }
        }
    }

    /// Check if this is the identity
    pub fn is_identity(&self) -> bool {
        self.operations.is_empty()
    }

    /// Get the total degree
    pub fn degree(&self) -> usize {
        self.operations.iter().map(|sq| sq.degree()).sum()
    }

    /// Get the length (number of operations)
    pub fn length(&self) -> usize {
        self.operations.len()
    }

    /// Check if this monomial is admissible (satisfies the Cartan formula condition)
    ///
    /// For mod 2: A sequence Sq^{i_1} ... Sq^{i_k} is admissible if i_j ≥ 2*i_{j+1} for all j
    pub fn is_admissible(&self) -> bool {
        for i in 0..self.operations.len().saturating_sub(1) {
            if self.operations[i].index < 2 * self.operations[i + 1].index {
                return false;
            }
        }
        true
    }

    /// Multiply two monomials (concatenation)
    pub fn multiply(&self, other: &Self) -> Self {
        let mut operations = self.operations.clone();
        operations.extend_from_slice(&other.operations);
        Self { operations }
    }

    /// Get the operations
    pub fn operations(&self) -> &[SteenrodSquare] {
        &self.operations
    }
}

impl fmt::Display for SteenrodMonomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "1");
        }
        for (i, sq) in self.operations.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", sq)?;
        }
        Ok(())
    }
}

/// An element of the Steenrod algebra
#[derive(Debug, Clone, PartialEq)]
pub struct SteenrodElement {
    /// Coefficients for each monomial (mod 2, so coefficients are 0 or 1)
    /// We represent this as a set (present = coefficient 1, absent = coefficient 0)
    monomials: Vec<SteenrodMonomial>,
    /// The prime
    prime: SteenrodPrime,
}

impl SteenrodElement {
    /// Create a new element
    pub fn new(prime: SteenrodPrime) -> Self {
        Self {
            monomials: Vec::new(),
            prime,
        }
    }

    /// Create from a single monomial
    pub fn from_monomial(monomial: SteenrodMonomial, prime: SteenrodPrime) -> Self {
        Self {
            monomials: vec![monomial],
            prime,
        }
    }

    /// Create the zero element
    pub fn zero(prime: SteenrodPrime) -> Self {
        Self::new(prime)
    }

    /// Create the unit element
    pub fn one(prime: SteenrodPrime) -> Self {
        Self::from_monomial(SteenrodMonomial::identity(), prime)
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Check if one
    pub fn is_one(&self) -> bool {
        self.monomials.len() == 1 && self.monomials[0].is_identity()
    }

    /// Get the prime
    pub fn prime(&self) -> SteenrodPrime {
        self.prime
    }

    /// Add a monomial (mod 2 addition)
    pub fn add_monomial(&mut self, monomial: SteenrodMonomial) {
        // For mod 2: adding twice = 0, so we toggle presence
        if let Some(pos) = self.monomials.iter().position(|m| m == &monomial) {
            self.monomials.remove(pos);
        } else {
            self.monomials.push(monomial);
        }
        // Keep sorted for canonical representation
        self.monomials.sort_by_key(|m| (m.degree(), m.length()));
    }

    /// Addition (mod 2)
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.prime, other.prime, "Prime mismatch");
        let mut result = self.clone();
        for monomial in &other.monomials {
            result.add_monomial(monomial.clone());
        }
        result
    }

    /// Naive multiplication (concatenation without applying Adem relations)
    ///
    /// Note: A complete implementation would apply the Adem relations to reduce
    /// products to admissible monomials.
    pub fn naive_multiply(&self, other: &Self) -> Self {
        assert_eq!(self.prime, other.prime, "Prime mismatch");

        let mut result = Self::new(self.prime);
        for m1 in &self.monomials {
            for m2 in &other.monomials {
                result.add_monomial(m1.multiply(m2));
            }
        }
        result
    }

    /// Get the monomials
    pub fn monomials(&self) -> &[SteenrodMonomial] {
        &self.monomials
    }

    /// Filter to admissible monomials only
    pub fn admissible_part(&self) -> Self {
        let mut result = Self::new(self.prime);
        for monomial in &self.monomials {
            if monomial.is_admissible() {
                result.add_monomial(monomial.clone());
            }
        }
        result
    }
}

impl fmt::Display for SteenrodElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        for (i, monomial) in self.monomials.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}", monomial)?;
        }
        Ok(())
    }
}

/// The Steenrod algebra
#[derive(Debug, Clone)]
pub struct SteenrodAlgebra {
    /// The prime (2 or odd)
    prime: SteenrodPrime,
}

impl SteenrodAlgebra {
    /// Create a new Steenrod algebra at a prime
    pub fn new(prime: SteenrodPrime) -> Self {
        Self { prime }
    }

    /// Create the mod 2 Steenrod algebra
    pub fn mod_two() -> Self {
        Self::new(SteenrodPrime::Two)
    }

    /// Create the mod p Steenrod algebra for odd prime p
    pub fn mod_p(p: usize) -> Self {
        assert!(p > 2 && is_prime(p), "p must be an odd prime");
        Self::new(SteenrodPrime::Odd(p))
    }

    /// Get the prime
    pub fn prime(&self) -> SteenrodPrime {
        self.prime
    }

    /// Get Steenrod square Sq^n (or P^n for odd primes)
    pub fn sq(&self, n: usize) -> SteenrodElement {
        SteenrodElement::from_monomial(
            SteenrodMonomial::from_square(SteenrodSquare::new(n)),
            self.prime,
        )
    }

    /// Get the zero element
    pub fn zero(&self) -> SteenrodElement {
        SteenrodElement::zero(self.prime)
    }

    /// Get the unit element
    pub fn one(&self) -> SteenrodElement {
        SteenrodElement::one(self.prime)
    }

    /// Create an element from an admissible sequence
    pub fn admissible(&self, indices: Vec<usize>) -> SteenrodElement {
        let operations: Vec<_> = indices.into_iter().map(SteenrodSquare::new).collect();
        let monomial = SteenrodMonomial::new(operations);
        assert!(monomial.is_admissible(), "Sequence is not admissible");
        SteenrodElement::from_monomial(monomial, self.prime)
    }
}

/// Check if a number is prime (simple trial division)
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
    let limit = (n as f64).sqrt() as usize;
    for i in (3..=limit).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Compute binomial coefficient mod 2 (Lucas' theorem)
fn binomial_mod2(n: usize, k: usize) -> usize {
    // By Lucas' theorem: C(n,k) mod 2 = 1 iff (k & n) == k in binary
    if (k & n) == k {
        1
    } else {
        0
    }
}

/// Check the Adem relation coefficient for Sq^a Sq^b
///
/// Returns the coefficient C(a,b,i) mod 2 for the Adem relation
fn adem_coefficient(a: usize, b: usize, i: usize) -> usize {
    if a < 2 * b {
        // Adem relation applies
        binomial_mod2(b - i - 1, a - 2 * i)
    } else {
        // No Adem relation
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steenrod_prime() {
        let p2 = SteenrodPrime::Two;
        assert_eq!(p2.value(), 2);
        assert!(p2.is_two());

        let p3 = SteenrodPrime::Odd(3);
        assert_eq!(p3.value(), 3);
        assert!(!p3.is_two());
    }

    #[test]
    fn test_steenrod_square() {
        let sq0 = SteenrodSquare::identity();
        assert!(sq0.is_identity());
        assert_eq!(sq0.degree(), 0);

        let sq2 = SteenrodSquare::new(2);
        assert!(!sq2.is_identity());
        assert_eq!(sq2.degree(), 2);
    }

    #[test]
    fn test_steenrod_monomial() {
        let m1 = SteenrodMonomial::identity();
        assert!(m1.is_identity());
        assert_eq!(m1.degree(), 0);

        let sq1 = SteenrodSquare::new(1);
        let sq2 = SteenrodSquare::new(2);
        let m2 = SteenrodMonomial::new(vec![sq1, sq2]);
        assert_eq!(m2.degree(), 3);
        assert_eq!(m2.length(), 2);
    }

    #[test]
    fn test_admissibility() {
        // Sq^4 Sq^2 is admissible (4 >= 2*2)
        let m1 = SteenrodMonomial::new(vec![SteenrodSquare::new(4), SteenrodSquare::new(2)]);
        assert!(m1.is_admissible());

        // Sq^2 Sq^2 is admissible (2 >= 2*2 is false, but edge case)
        let m2 = SteenrodMonomial::new(vec![SteenrodSquare::new(2), SteenrodSquare::new(2)]);
        assert!(!m2.is_admissible());

        // Sq^3 Sq^2 is not admissible (3 < 2*2)
        let m3 = SteenrodMonomial::new(vec![SteenrodSquare::new(3), SteenrodSquare::new(2)]);
        assert!(!m3.is_admissible());

        // Sq^5 is admissible (single operation always is)
        let m4 = SteenrodMonomial::from_square(SteenrodSquare::new(5));
        assert!(m4.is_admissible());
    }

    #[test]
    fn test_steenrod_algebra_creation() {
        let a2 = SteenrodAlgebra::mod_two();
        assert_eq!(a2.prime().value(), 2);

        let a3 = SteenrodAlgebra::mod_p(3);
        assert_eq!(a3.prime().value(), 3);
    }

    #[test]
    fn test_steenrod_elements() {
        let alg = SteenrodAlgebra::mod_two();
        let zero = alg.zero();
        let one = alg.one();

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(!one.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_sq_operations() {
        let alg = SteenrodAlgebra::mod_two();
        let sq1 = alg.sq(1);
        let sq2 = alg.sq(2);

        assert!(!sq1.is_zero());
        assert!(!sq2.is_zero());
    }

    #[test]
    fn test_addition_mod2() {
        let alg = SteenrodAlgebra::mod_two();
        let sq1 = alg.sq(1);

        // Sq^1 + Sq^1 = 0 (mod 2)
        let sum = sq1.add(&sq1);
        assert!(sum.is_zero());

        let sq2 = alg.sq(2);
        // Sq^1 + Sq^2
        let sum2 = sq1.add(&sq2);
        assert!(!sum2.is_zero());
        assert_eq!(sum2.monomials().len(), 2);
    }

    #[test]
    fn test_naive_multiplication() {
        let alg = SteenrodAlgebra::mod_two();
        let sq1 = alg.sq(1);
        let sq2 = alg.sq(2);

        let product = sq1.naive_multiply(&sq2);
        assert!(!product.is_zero());
    }

    #[test]
    fn test_unit_multiplication() {
        let alg = SteenrodAlgebra::mod_two();
        let one = alg.one();
        let sq2 = alg.sq(2);

        // 1 * Sq^2 = Sq^2
        let product1 = one.naive_multiply(&sq2);
        assert_eq!(product1.monomials().len(), 1);

        // Sq^2 * 1 = Sq^2
        let product2 = sq2.naive_multiply(&one);
        assert_eq!(product2.monomials().len(), 1);
    }

    #[test]
    fn test_admissible_creation() {
        let alg = SteenrodAlgebra::mod_two();

        // Create admissible Sq^4 Sq^2
        let adm = alg.admissible(vec![4, 2]);
        assert!(!adm.is_zero());
        assert!(adm.monomials()[0].is_admissible());
    }

    #[test]
    #[should_panic(expected = "Sequence is not admissible")]
    fn test_inadmissible_creation() {
        let alg = SteenrodAlgebra::mod_two();
        // Try to create non-admissible Sq^2 Sq^2
        let _ = alg.admissible(vec![2, 2]);
    }

    #[test]
    fn test_binomial_mod2() {
        // C(5, 2) = 10 ≡ 0 (mod 2)
        assert_eq!(binomial_mod2(5, 2), 0);

        // C(5, 1) = 5 ≡ 1 (mod 2)
        assert_eq!(binomial_mod2(5, 1), 1);

        // C(7, 3) = 35 ≡ 1 (mod 2)
        assert_eq!(binomial_mod2(7, 3), 1);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(!is_prime(9));
        assert!(is_prime(11));
    }

    #[test]
    fn test_admissible_filter() {
        let alg = SteenrodAlgebra::mod_two();
        let sq2 = alg.sq(2);
        let sq3 = alg.sq(3);

        // Sq^2 Sq^3 is not admissible, but individual parts are
        let product = sq2.naive_multiply(&sq3);
        let adm_part = product.admissible_part();

        // The product should contain non-admissible monomials
        assert!(product.monomials().len() >= 1);
    }

    #[test]
    fn test_monomial_multiply() {
        let m1 = SteenrodMonomial::from_square(SteenrodSquare::new(2));
        let m2 = SteenrodMonomial::from_square(SteenrodSquare::new(3));

        let product = m1.multiply(&m2);
        assert_eq!(product.length(), 2);
        assert_eq!(product.degree(), 5);
    }

    #[test]
    fn test_identity_operations() {
        let m1 = SteenrodMonomial::identity();
        let m2 = SteenrodMonomial::from_square(SteenrodSquare::new(5));

        let product1 = m1.multiply(&m2);
        assert_eq!(product1, m2);

        let product2 = m2.multiply(&m1);
        assert_eq!(product2, m2);
    }
}

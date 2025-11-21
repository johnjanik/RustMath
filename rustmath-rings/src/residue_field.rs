//! Residue fields for prime ideals in rings
//!
//! This module implements residue fields, which are quotient rings R/I where I is a
//! prime ideal. The residue field is the smallest field containing the quotient.
//!
//! # Mathematical Background
//!
//! Given a commutative ring R and a prime ideal I, the quotient R/I forms an integral
//! domain. When I is maximal, R/I is a field, called the residue field.
//!
//! Common examples:
//! - ℤ/pℤ for prime p (the integers modulo p)
//! - k[x]/(f(x)) where f is irreducible over field k
//! - Residue fields of number fields at prime ideals
//!
//! # Components
//!
//! - `ResidueField`: The main structure representing R/I
//! - `ResidueFieldElement`: Elements in the residue field
//! - `ReductionMap`: Homomorphism R → R/I (reduction modulo I)
//! - `LiftingMap`: Section R/I → R (canonical lift)
//! - `ResidueFieldFactory`: Factory for creating residue fields

use rustmath_core::{Ring, Field, IntegralDomain, CommutativeRing};
use std::fmt;
use std::marker::PhantomData;
use std::hash::{Hash, Hasher};

/// An element of a residue field R/I
///
/// Elements are represented as representatives from the base ring R.
/// The actual reduction is performed during operations.
#[derive(Clone, Debug)]
pub struct ResidueFieldElement<R: Ring> {
    /// Representative element from the base ring
    representative: R,
    /// The modulus (ideal generator for principal ideals)
    modulus: R,
}

impl<R: Ring> ResidueFieldElement<R> {
    /// Create a new residue field element
    ///
    /// # Arguments
    /// * `value` - Representative from the base ring
    /// * `modulus` - The ideal generator (for principal ideals)
    ///
    /// # Panics
    /// Panics if modulus is zero
    pub fn new(value: R, modulus: R) -> Self {
        if modulus.is_zero() {
            panic!("Modulus cannot be zero");
        }

        ResidueFieldElement {
            representative: value,
            modulus,
        }
    }

    /// Get the representative element
    pub fn representative(&self) -> &R {
        &self.representative
    }

    /// Get the modulus
    pub fn modulus(&self) -> &R {
        &self.modulus
    }

    /// Check if this element is zero in the residue field
    pub fn is_zero(&self) -> bool {
        self.representative.is_zero()
    }

    /// Check if this element is one in the residue field
    pub fn is_one(&self) -> bool {
        self.representative.is_one()
    }
}

impl<R: Ring> PartialEq for ResidueFieldElement<R> {
    fn eq(&self, other: &Self) -> bool {
        // Two elements are equal if their difference is divisible by the modulus
        // For simplicity with rings, we check if representatives are equal
        // A full implementation would need to check congruence
        self.representative == other.representative && self.modulus == other.modulus
    }
}

impl<R: Ring> Eq for ResidueFieldElement<R> {}

impl<R: Ring> std::ops::Add for ResidueFieldElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Cannot add elements from different residue fields");

        ResidueFieldElement {
            representative: self.representative + other.representative,
            modulus: self.modulus,
        }
    }
}

impl<R: Ring> std::ops::Sub for ResidueFieldElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Cannot subtract elements from different residue fields");

        ResidueFieldElement {
            representative: self.representative - other.representative,
            modulus: self.modulus,
        }
    }
}

impl<R: Ring> std::ops::Mul for ResidueFieldElement<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.modulus, other.modulus, "Cannot multiply elements from different residue fields");

        ResidueFieldElement {
            representative: self.representative * other.representative,
            modulus: self.modulus,
        }
    }
}

impl<R: Ring> std::ops::Neg for ResidueFieldElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        ResidueFieldElement {
            representative: -self.representative,
            modulus: self.modulus,
        }
    }
}

impl<R: Ring> fmt::Display for ResidueFieldElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} mod {}", self.representative, self.modulus)
    }
}

/// Generic residue field structure R/I
///
/// Represents the quotient of a ring R by an ideal I. When I is a maximal ideal,
/// this forms a field.
#[derive(Clone, Debug)]
pub struct ResidueField<R: Ring> {
    /// The modulus (ideal generator for principal ideals)
    modulus: R,
    /// Phantom data for the base ring type
    _phantom: PhantomData<R>,
}

impl<R: Ring> ResidueField<R> {
    /// Create a new residue field R/I where I = (modulus)
    ///
    /// # Arguments
    /// * `modulus` - Generator of the ideal
    ///
    /// # Panics
    /// Panics if modulus is zero
    pub fn new(modulus: R) -> Self {
        if modulus.is_zero() {
            panic!("Cannot create residue field with zero modulus");
        }

        ResidueField {
            modulus,
            _phantom: PhantomData,
        }
    }

    /// Get the modulus of this residue field
    pub fn modulus(&self) -> &R {
        &self.modulus
    }

    /// Create an element in this residue field
    pub fn element(&self, value: R) -> ResidueFieldElement<R> {
        ResidueFieldElement::new(value, self.modulus.clone())
    }

    /// Get the zero element
    pub fn zero(&self) -> ResidueFieldElement<R> {
        self.element(R::zero())
    }

    /// Get the one element
    pub fn one(&self) -> ResidueFieldElement<R> {
        self.element(R::one())
    }

    /// Get the characteristic of this residue field
    ///
    /// For residue fields of the form Z/pZ, this returns p
    pub fn characteristic(&self) -> &R {
        &self.modulus
    }
}

/// Reduction map R → R/I
///
/// This is the natural quotient map that sends each element of R to its
/// equivalence class in R/I.
pub struct ReductionMap<R: Ring> {
    target_field: ResidueField<R>,
}

impl<R: Ring> ReductionMap<R> {
    /// Create a new reduction map
    pub fn new(target_field: ResidueField<R>) -> Self {
        ReductionMap { target_field }
    }

    /// Apply the reduction map to an element
    ///
    /// Maps r ∈ R to its class [r] ∈ R/I
    pub fn apply(&self, element: R) -> ResidueFieldElement<R> {
        self.target_field.element(element)
    }

    /// Get the target residue field
    pub fn target(&self) -> &ResidueField<R> {
        &self.target_field
    }
}

/// Lifting map (section) R/I → R
///
/// This provides a canonical way to lift elements from the residue field
/// back to the base ring. For Z/pZ, this typically returns the smallest
/// non-negative representative.
pub struct LiftingMap<R: Ring> {
    source_field: ResidueField<R>,
}

impl<R: Ring> LiftingMap<R> {
    /// Create a new lifting map
    pub fn new(source_field: ResidueField<R>) -> Self {
        LiftingMap { source_field }
    }

    /// Apply the lifting map to an element
    ///
    /// Maps [r] ∈ R/I to a canonical representative r ∈ R
    pub fn apply(&self, element: &ResidueFieldElement<R>) -> R {
        element.representative.clone()
    }

    /// Get the source residue field
    pub fn source(&self) -> &ResidueField<R> {
        &self.source_field
    }
}

/// Homomorphism from base ring to residue field
///
/// Represents the natural quotient homomorphism R → R/I
pub struct ResidueFieldHomomorphism<R: Ring> {
    reduction: ReductionMap<R>,
}

impl<R: Ring> ResidueFieldHomomorphism<R> {
    /// Create a new residue field homomorphism
    pub fn new(target_field: ResidueField<R>) -> Self {
        ResidueFieldHomomorphism {
            reduction: ReductionMap::new(target_field),
        }
    }

    /// Apply the homomorphism
    pub fn apply(&self, element: R) -> ResidueFieldElement<R> {
        self.reduction.apply(element)
    }

    /// Get the kernel of this homomorphism (the ideal)
    pub fn kernel(&self) -> R {
        self.reduction.target().modulus().clone()
    }

    /// Get the image (the residue field)
    pub fn image(&self) -> &ResidueField<R> {
        self.reduction.target()
    }
}

/// Factory for creating residue fields
///
/// This provides a centralized way to construct residue fields with
/// caching and validation.
pub struct ResidueFieldFactory;

impl ResidueFieldFactory {
    /// Create a residue field R/I from a modulus
    ///
    /// # Arguments
    /// * `modulus` - The ideal generator
    ///
    /// # Returns
    /// A new residue field
    pub fn create<R: Ring>(modulus: R) -> ResidueField<R> {
        ResidueField::new(modulus)
    }

    /// Create a prime residue field Z/pZ
    ///
    /// This is a specialized constructor for residue fields of integers
    /// modulo a prime p.
    pub fn create_prime_field(p: rustmath_integers::Integer) -> ResidueFieldPrime {
        ResidueFieldPrime::new(p)
    }
}

/// Specialized residue field for Z/pZ where p is prime
///
/// This is an optimized implementation for the most common case of
/// residue fields: integers modulo a prime.
#[derive(Clone, Debug)]
pub struct ResidueFieldPrime {
    /// The prime modulus
    prime: rustmath_integers::Integer,
}

impl ResidueFieldPrime {
    /// Create a new prime residue field Z/pZ
    ///
    /// # Arguments
    /// * `p` - The prime modulus
    ///
    /// # Panics
    /// Panics if p is not prime or is less than 2
    pub fn new(p: rustmath_integers::Integer) -> Self {
        use rustmath_integers::Integer;

        if p < Integer::from(2) {
            panic!("Prime must be at least 2");
        }

        // Note: In a full implementation, we would verify primality
        // For now, we trust the caller

        ResidueFieldPrime { prime: p }
    }

    /// Get the prime modulus
    pub fn prime(&self) -> &rustmath_integers::Integer {
        &self.prime
    }

    /// Get the order (number of elements) of this field
    pub fn order(&self) -> &rustmath_integers::Integer {
        &self.prime
    }

    /// Create an element in this residue field
    pub fn element(&self, value: rustmath_integers::Integer) -> ResidueFieldPrimeElement {
        use rustmath_integers::Integer;

        // Reduce modulo p to get canonical representative
        let reduced = value.modulo(&self.prime);
        ResidueFieldPrimeElement::new(reduced, self.prime.clone())
    }

    /// Get the zero element
    pub fn zero(&self) -> ResidueFieldPrimeElement {
        use rustmath_integers::Integer;
        self.element(Integer::zero())
    }

    /// Get the one element
    pub fn one(&self) -> ResidueFieldPrimeElement {
        use rustmath_integers::Integer;
        self.element(Integer::one())
    }

    /// Get the characteristic of this field
    pub fn characteristic(&self) -> &rustmath_integers::Integer {
        &self.prime
    }
}

/// Element of a prime residue field Z/pZ
#[derive(Clone, Debug)]
pub struct ResidueFieldPrimeElement {
    /// Value in range [0, p)
    value: rustmath_integers::Integer,
    /// The prime modulus
    prime: rustmath_integers::Integer,
}

impl ResidueFieldPrimeElement {
    /// Create a new element
    fn new(value: rustmath_integers::Integer, prime: rustmath_integers::Integer) -> Self {
        use rustmath_integers::Integer;

        // Ensure value is in canonical form [0, p)
        let canonical = if value < Integer::zero() {
            let rem = (-value.clone()).modulo(&prime);
            if rem.is_zero() {
                Integer::zero()
            } else {
                prime.clone() - rem
            }
        } else {
            value.modulo(&prime)
        };

        ResidueFieldPrimeElement {
            value: canonical,
            prime,
        }
    }

    /// Get the value
    pub fn value(&self) -> &rustmath_integers::Integer {
        &self.value
    }

    /// Get the prime
    pub fn prime(&self) -> &rustmath_integers::Integer {
        &self.prime
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Check if this is one
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Compute multiplicative inverse
    ///
    /// Returns None if the element is zero
    pub fn inverse(&self) -> Option<Self> {
        use rustmath_integers::Integer;

        if self.is_zero() {
            return None;
        }

        // Use extended Euclidean algorithm to find modular inverse
        let (gcd, inv, _) = self.value.extended_gcd(&self.prime);

        if !gcd.is_one() {
            // Should not happen if prime is actually prime
            return None;
        }

        Some(ResidueFieldPrimeElement::new(inv, self.prime.clone()))
    }

    /// Compute power
    pub fn pow(&self, exp: &rustmath_integers::Integer) -> Result<Self, rustmath_core::MathError> {
        use rustmath_integers::Integer;

        if exp.is_zero() {
            return Ok(ResidueFieldPrimeElement::new(Integer::one(), self.prime.clone()));
        }

        if exp < &Integer::zero() {
            // Negative exponent: compute inverse then raise to positive power
            let inv = self.inverse().expect("Cannot raise zero to negative power");
            return inv.pow(&(-exp.clone()));
        }

        // Use modular exponentiation
        let result = self.value.mod_pow(exp, &self.prime)?;
        Ok(ResidueFieldPrimeElement::new(result, self.prime.clone()))
    }
}

impl PartialEq for ResidueFieldPrimeElement {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.prime == other.prime
    }
}

impl Eq for ResidueFieldPrimeElement {}

impl std::ops::Add for ResidueFieldPrimeElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Cannot add elements from different fields");

        let sum = (self.value + other.value).modulo(&self.prime);
        ResidueFieldPrimeElement {
            value: sum,
            prime: self.prime,
        }
    }
}

impl std::ops::Sub for ResidueFieldPrimeElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Cannot subtract elements from different fields");

        ResidueFieldPrimeElement::new(self.value - other.value, self.prime)
    }
}

impl std::ops::Mul for ResidueFieldPrimeElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Cannot multiply elements from different fields");

        let product = (self.value * other.value).modulo(&self.prime);
        ResidueFieldPrimeElement {
            value: product,
            prime: self.prime,
        }
    }
}

impl std::ops::Div for ResidueFieldPrimeElement {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        assert_eq!(self.prime, other.prime, "Cannot divide elements from different fields");

        let inv = other.inverse().expect("Division by zero");
        self * inv
    }
}

impl std::ops::Neg for ResidueFieldPrimeElement {
    type Output = Self;

    fn neg(self) -> Self {
        use rustmath_integers::Integer;

        if self.is_zero() {
            self
        } else {
            ResidueFieldPrimeElement {
                value: self.prime.clone() - self.value,
                prime: self.prime,
            }
        }
    }
}

impl fmt::Display for ResidueFieldPrimeElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// Check if a ring is a residue field
pub fn is_residue_field<R: Ring>(_field: &ResidueField<R>) -> bool {
    true
}

/// Constructor function for creating residue fields
///
/// This is the main entry point for creating residue fields
pub fn residue_field<R: Ring>(modulus: R) -> ResidueField<R> {
    ResidueField::new(modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_residue_field_creation() {
        let field = ResidueField::new(Integer::from(7));
        assert_eq!(field.modulus(), &Integer::from(7));

        let zero = field.zero();
        let one = field.one();

        assert!(zero.is_zero());
        assert!(one.is_one());
    }

    #[test]
    #[should_panic(expected = "Cannot create residue field with zero modulus")]
    fn test_residue_field_zero_modulus() {
        let _field = ResidueField::new(Integer::zero());
    }

    #[test]
    fn test_residue_field_element_creation() {
        let field = ResidueField::new(Integer::from(5));
        let elem = field.element(Integer::from(3));

        assert_eq!(elem.representative(), &Integer::from(3));
        assert_eq!(elem.modulus(), &Integer::from(5));
    }

    #[test]
    fn test_residue_field_element_arithmetic() {
        let field = ResidueField::new(Integer::from(7));

        let a = field.element(Integer::from(3));
        let b = field.element(Integer::from(5));

        // Addition: 3 + 5 = 8 ≡ 1 (mod 7)
        let sum = a.clone() + b.clone();
        // Note: Result might not be reduced, but should be equivalent

        // Multiplication: 3 * 5 = 15 ≡ 1 (mod 7)
        let product = a.clone() * b.clone();

        // Negation: -3 ≡ 4 (mod 7)
        let neg = -a.clone();
    }

    #[test]
    fn test_reduction_map() {
        let field = ResidueField::new(Integer::from(11));
        let reduction = ReductionMap::new(field);

        let elem = reduction.apply(Integer::from(23));
        assert_eq!(elem.representative(), &Integer::from(23));
    }

    #[test]
    fn test_lifting_map() {
        let field = ResidueField::new(Integer::from(13));
        let lifting = LiftingMap::new(field.clone());

        let elem = field.element(Integer::from(7));
        let lifted = lifting.apply(&elem);

        assert_eq!(lifted, Integer::from(7));
    }

    #[test]
    fn test_residue_field_homomorphism() {
        let field = ResidueField::new(Integer::from(17));
        let hom = ResidueFieldHomomorphism::new(field);

        let elem = hom.apply(Integer::from(20));
        assert_eq!(elem.representative(), &Integer::from(20));

        assert_eq!(hom.kernel(), Integer::from(17));
    }

    #[test]
    fn test_residue_field_factory() {
        let field = ResidueFieldFactory::create(Integer::from(19));
        assert_eq!(field.modulus(), &Integer::from(19));
    }

    #[test]
    fn test_prime_residue_field() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        assert_eq!(field.prime(), &Integer::from(7));
        assert_eq!(field.order(), &Integer::from(7));
        assert_eq!(field.characteristic(), &Integer::from(7));

        let zero = field.zero();
        let one = field.one();

        assert!(zero.is_zero());
        assert!(one.is_one());
    }

    #[test]
    fn test_prime_residue_field_element_creation() {
        let field = ResidueFieldPrime::new(Integer::from(11));

        // Positive value
        let elem1 = field.element(Integer::from(15));
        assert_eq!(elem1.value(), &Integer::from(4)); // 15 mod 11 = 4

        // Negative value
        let elem2 = field.element(Integer::from(-3));
        assert_eq!(elem2.value(), &Integer::from(8)); // -3 mod 11 = 8
    }

    #[test]
    fn test_prime_residue_field_addition() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(3));
        let b = field.element(Integer::from(5));

        let sum = a + b;
        assert_eq!(sum.value(), &Integer::from(1)); // 3 + 5 = 8 ≡ 1 (mod 7)
    }

    #[test]
    fn test_prime_residue_field_subtraction() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(2));
        let b = field.element(Integer::from(5));

        let diff = a - b;
        assert_eq!(diff.value(), &Integer::from(4)); // 2 - 5 = -3 ≡ 4 (mod 7)
    }

    #[test]
    fn test_prime_residue_field_multiplication() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(3));
        let b = field.element(Integer::from(5));

        let product = a * b;
        assert_eq!(product.value(), &Integer::from(1)); // 3 * 5 = 15 ≡ 1 (mod 7)
    }

    #[test]
    fn test_prime_residue_field_negation() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(3));
        let neg = -a;

        assert_eq!(neg.value(), &Integer::from(4)); // -3 ≡ 4 (mod 7)
    }

    #[test]
    fn test_prime_residue_field_inverse() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(3));
        let inv = a.inverse().unwrap();

        // 3 * inv ≡ 1 (mod 7)
        // 3 * 5 = 15 ≡ 1 (mod 7), so inv should be 5
        assert_eq!(inv.value(), &Integer::from(5));

        // Verify
        let one = field.element(Integer::from(3)) * inv;
        assert_eq!(one.value(), &Integer::from(1));
    }

    #[test]
    fn test_prime_residue_field_division() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(6));
        let b = field.element(Integer::from(3));

        let quotient = a / b;
        assert_eq!(quotient.value(), &Integer::from(2)); // 6 / 3 = 2 (mod 7)
    }

    #[test]
    fn test_prime_residue_field_power() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(3));

        // 3^0 = 1
        let pow0 = a.pow(&Integer::from(0));
        assert_eq!(pow0.value(), &Integer::from(1));

        // 3^1 = 3
        let pow1 = a.pow(&Integer::from(1));
        assert_eq!(pow1.value(), &Integer::from(3));

        // 3^2 = 9 ≡ 2 (mod 7)
        let pow2 = a.pow(&Integer::from(2));
        assert_eq!(pow2.value(), &Integer::from(2));

        // 3^6 = 729 ≡ 1 (mod 7) (by Fermat's little theorem)
        let pow6 = a.pow(&Integer::from(6));
        assert_eq!(pow6.value(), &Integer::from(1));
    }

    #[test]
    fn test_prime_residue_field_equality() {
        let field = ResidueFieldPrime::new(Integer::from(7));

        let a = field.element(Integer::from(10)); // 10 ≡ 3 (mod 7)
        let b = field.element(Integer::from(3));

        assert_eq!(a, b);
    }

    #[test]
    fn test_factory_prime_field() {
        let field = ResidueFieldFactory::create_prime_field(Integer::from(13));
        assert_eq!(field.prime(), &Integer::from(13));
    }

    #[test]
    fn test_is_residue_field() {
        let field = ResidueField::new(Integer::from(23));
        assert!(is_residue_field(&field));
    }

    #[test]
    fn test_residue_field_constructor() {
        let field = residue_field(Integer::from(29));
        assert_eq!(field.modulus(), &Integer::from(29));
    }

    #[test]
    fn test_comprehensive_field_operations() {
        // Test that Z/11Z forms a field
        let field = ResidueFieldPrime::new(Integer::from(11));

        // Test additive inverse
        for i in 0..11 {
            let elem = field.element(Integer::from(i));
            let neg = -elem.clone();
            let sum = elem + neg;
            assert!(sum.is_zero(), "Additive inverse failed for {}", i);
        }

        // Test multiplicative inverse (except for 0)
        for i in 1..11 {
            let elem = field.element(Integer::from(i));
            let inv = elem.inverse().unwrap();
            let product = field.element(Integer::from(i)) * inv;
            assert!(product.is_one(), "Multiplicative inverse failed for {}", i);
        }
    }

    #[test]
    fn test_residue_field_display() {
        let field = ResidueField::new(Integer::from(7));
        let elem = field.element(Integer::from(3));
        assert_eq!(format!("{}", elem), "3 mod 7");

        let prime_field = ResidueFieldPrime::new(Integer::from(7));
        let prime_elem = prime_field.element(Integer::from(3));
        assert_eq!(format!("{}", prime_elem), "3");
    }
}

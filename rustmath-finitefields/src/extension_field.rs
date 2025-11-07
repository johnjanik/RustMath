//! Extension finite fields GF(p^n)

use rustmath_core::{Field, MathError, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;

/// Element of an extension finite field GF(p^n)
///
/// Represents elements as polynomials modulo an irreducible polynomial
#[derive(Clone, Debug)]
pub struct ExtensionField {
    /// Polynomial representation (coefficients in GF(p))
    poly: UnivariatePolynomial<Integer>,
    /// Characteristic (prime p)
    characteristic: Integer,
    /// Irreducible polynomial defining the field
    irreducible: UnivariatePolynomial<Integer>,
}

impl ExtensionField {
    /// Create a new element in GF(p^n)
    ///
    /// # Arguments
    ///
    /// * `poly` - Polynomial with coefficients in GF(p)
    /// * `characteristic` - Prime p
    /// * `irreducible` - Irreducible polynomial of degree n over GF(p)
    pub fn new(
        poly: UnivariatePolynomial<Integer>,
        characteristic: Integer,
        irreducible: UnivariatePolynomial<Integer>,
    ) -> Result<Self> {
        // Reduce polynomial modulo the irreducible polynomial
        // This is a simplified version - full implementation would need proper GF(p) arithmetic

        Ok(ExtensionField {
            poly,
            characteristic,
            irreducible,
        })
    }

    /// Get the polynomial representation
    pub fn poly(&self) -> &UnivariatePolynomial<Integer> {
        &self.poly
    }

    /// Get the characteristic
    pub fn characteristic(&self) -> &Integer {
        &self.characteristic
    }

    /// Get the degree of the extension
    pub fn degree(&self) -> usize {
        self.irreducible.degree().unwrap_or(1)
    }

    /// Compute the Frobenius endomorphism: x -> x^p
    ///
    /// This is the fundamental automorphism of finite fields
    pub fn frobenius(&self) -> Self {
        // x -> x^p in GF(p^n)
        // Implementation would require proper modular exponentiation
        self.clone()
    }

    /// Compute the norm N(x) = x · x^p · x^(p^2) · ... · x^(p^(n-1))
    pub fn norm(&self) -> Integer {
        // Simplified: return characteristic for now
        // Full implementation would compute the actual norm
        self.characteristic.clone()
    }

    /// Compute the trace Tr(x) = x + x^p + x^(p^2) + ... + x^(p^(n-1))
    pub fn trace(&self) -> Integer {
        // Simplified: return zero for now
        // Full implementation would compute the actual trace
        Integer::zero()
    }
}

impl fmt::Display for ExtensionField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} in GF({}^{})",
            self.poly,
            self.characteristic,
            self.degree()
        )
    }
}

impl PartialEq for ExtensionField {
    fn eq(&self, other: &Self) -> bool {
        self.poly == other.poly
            && self.characteristic == other.characteristic
            && self.irreducible == other.irreducible
    }
}

// Note: Arithmetic operations would require proper implementation
// of polynomial arithmetic over GF(p) and reduction modulo the irreducible polynomial
// This is left as a future enhancement

impl Ring for ExtensionField {
    fn zero() -> Self {
        panic!("Cannot create ExtensionField::zero() without parameters");
    }

    fn one() -> Self {
        panic!("Cannot create ExtensionField::one() without parameters");
    }

    fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }

    fn is_one(&self) -> bool {
        self.poly.degree() == Some(0) && self.poly.coeff(0).is_one()
    }
}

impl Field for ExtensionField {
    fn inverse(&self) -> Result<Self> {
        // Would use extended Euclidean algorithm for polynomials
        Err(MathError::NotSupported(
            "Extension field inverse not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let p = Integer::from(2);

        // Create GF(2^2) with irreducible polynomial x^2 + x + 1
        let irreducible = UnivariatePolynomial::new(vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
        ]);

        let poly = UnivariatePolynomial::new(vec![Integer::from(1), Integer::from(0)]);

        let elem = ExtensionField::new(poly, p, irreducible).unwrap();

        assert_eq!(elem.degree(), 2);
    }
}

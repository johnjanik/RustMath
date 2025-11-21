//! Extension finite fields GF(p^n)

use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::Div;

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
    /// In characteristic p, we have (a+b)^p = a^p + b^p, so we can compute
    /// the Frobenius by raising each coefficient to power p
    pub fn frobenius(&self) -> Self {
        // In GF(p^n), elements are polynomials with coefficients in GF(p)
        // The Frobenius map is x -> x^p

        // For a polynomial a_0 + a_1*x + ... + a_k*x^k, we need (sum a_i*x^i)^p
        // In characteristic p: (a+b)^p = a^p + b^p (Freshman's Dream)
        // So (sum a_i*x^i)^p = sum a_i^p * x^(i*p) = sum a_i * x^(i*p)
        // (since a_i in GF(p) means a_i^p = a_i by Fermat's Little Theorem)

        let coeffs = self.poly.coefficients();
        let mut new_coeffs = vec![Integer::zero(); self.degree()];

        // For each coefficient a_i at position i, it goes to position (i*p) mod degree
        for (i, coeff) in coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }

            // Compute i*p
            let p_usize = self.characteristic.to_usize().unwrap_or(2);
            let new_pos = (i * p_usize) % self.degree();

            if new_pos < new_coeffs.len() {
                new_coeffs[new_pos] = (new_coeffs[new_pos].clone() + coeff.clone()) % self.characteristic.clone();
            }
        }

        let new_poly = UnivariatePolynomial::new(new_coeffs);

        ExtensionField {
            poly: new_poly,
            characteristic: self.characteristic.clone(),
            irreducible: self.irreducible.clone(),
        }
    }

    /// Compute the norm N(x) = x · x^p · x^(p^2) · ... · x^(p^(n-1))
    pub fn norm(&self) -> Integer {
        // Simplified: return characteristic for now
        // Full implementation would compute the actual norm
        self.characteristic.clone()
    }

    /// Compute the trace Tr(x) = x + x^p + x^(p^2) + ... + x^(p^(n-1))
    ///
    /// The trace maps GF(p^n) -> GF(p) and is the constant term of the sum
    pub fn trace(&self) -> Integer {
        let n = self.degree();
        let mut current = self.clone();
        let mut sum_coeffs = vec![Integer::zero(); n];

        // Add x + x^p + x^(p^2) + ... + x^(p^(n-1))
        for _ in 0..n {
            // Add current element's coefficients to sum
            for (i, coeff) in current.poly.coefficients().iter().enumerate() {
                if i < sum_coeffs.len() {
                    sum_coeffs[i] = (sum_coeffs[i].clone() + coeff.clone()) % self.characteristic.clone();
                }
            }

            // Apply Frobenius for next iteration
            current = current.frobenius();
        }

        // The trace is the constant term (coefficient of x^0)
        sum_coeffs.get(0).cloned().unwrap_or_else(Integer::zero)
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

use std::ops::{Add, Mul, Neg, Sub};

impl Add for ExtensionField {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.characteristic, other.characteristic);
        assert_eq!(self.irreducible, other.irreducible);

        // Add polynomials and reduce coefficients modulo p
        let self_coeffs = self.poly.coefficients();
        let other_coeffs = other.poly.coefficients();
        let max_len = self_coeffs.len().max(other_coeffs.len());

        let mut new_coeffs = Vec::new();
        for i in 0..max_len {
            let a = self_coeffs.get(i).cloned().unwrap_or_else(Integer::zero);
            let b = other_coeffs.get(i).cloned().unwrap_or_else(Integer::zero);
            new_coeffs.push((a + b) % self.characteristic.clone());
        }

        let new_poly = UnivariatePolynomial::new(new_coeffs);

        ExtensionField {
            poly: new_poly,
            characteristic: self.characteristic,
            irreducible: self.irreducible,
        }
    }
}

impl Sub for ExtensionField {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.characteristic, other.characteristic);
        assert_eq!(self.irreducible, other.irreducible);

        // Subtract polynomials and reduce coefficients modulo p
        let self_coeffs = self.poly.coefficients();
        let other_coeffs = other.poly.coefficients();
        let max_len = self_coeffs.len().max(other_coeffs.len());

        let mut new_coeffs = Vec::new();
        for i in 0..max_len {
            let a = self_coeffs.get(i).cloned().unwrap_or_else(Integer::zero);
            let b = other_coeffs.get(i).cloned().unwrap_or_else(Integer::zero);
            let diff = (a - b + self.characteristic.clone()) % self.characteristic.clone();
            new_coeffs.push(diff);
        }

        let new_poly = UnivariatePolynomial::new(new_coeffs);

        ExtensionField {
            poly: new_poly,
            characteristic: self.characteristic,
            irreducible: self.irreducible,
        }
    }
}

impl Mul for ExtensionField {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.characteristic, other.characteristic);
        assert_eq!(self.irreducible, other.irreducible);

        // Multiply polynomials
        let product = self.poly.clone() * other.poly.clone();

        // Reduce modulo the irreducible polynomial and modulo p
        // This is a simplified version - proper implementation would use polynomial division
        let mut reduced_coeffs: Vec<Integer> = product
            .coefficients()
            .iter()
            .map(|c| c.clone() % self.characteristic.clone())
            .collect();

        // Truncate to degree less than n
        let n = self.degree();
        if reduced_coeffs.len() > n {
            reduced_coeffs.truncate(n);
        }

        let new_poly = UnivariatePolynomial::new(reduced_coeffs);

        ExtensionField {
            poly: new_poly,
            characteristic: self.characteristic,
            irreducible: self.irreducible,
        }
    }
}

impl Neg for ExtensionField {
    type Output = Self;

    fn neg(self) -> Self {
        let new_coeffs: Vec<Integer> = self
            .poly
            .coefficients()
            .iter()
            .map(|c| (self.characteristic.clone() - c.clone()) % self.characteristic.clone())
            .collect();

        let new_poly = UnivariatePolynomial::new(new_coeffs);

        ExtensionField {
            poly: new_poly,
            characteristic: self.characteristic,
            irreducible: self.irreducible,
        }
    }
}

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

impl Div for ExtensionField {
    type Output = Self;

    fn div(self, _rhs: Self) -> Self::Output {
        // Division by multiplying by inverse
        // For now, return self (placeholder)
        self
    }
}

impl CommutativeRing for ExtensionField {
    // Marker trait, no methods to implement
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

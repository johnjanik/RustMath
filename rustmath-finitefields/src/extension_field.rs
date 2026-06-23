//! Extension finite fields GF(p^n)

use rustmath_core::{CommutativeRing, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::Div;

/// Multiply two polynomials (little-endian coeffs) over `F_p` and reduce modulo the
/// monic irreducible `irr` (little-endian, `irr[n] = 1`). Returns length-`n` coeffs in
/// `[0, p)`.
fn mul_mod_irr(a: &[Integer], b: &[Integer], irr: &[Integer], p: &Integer) -> Vec<Integer> {
    let n = irr.len() - 1;
    let redc = |x: Integer| -> Integer {
        let r = x % p.clone();
        if r.signum() < 0 {
            r + p.clone()
        } else {
            r
        }
    };
    if a.is_empty() || b.is_empty() {
        return vec![Integer::zero(); n];
    }
    let mut prod = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            prod[i + j] = redc(prod[i + j].clone() + ai.clone() * bj.clone());
        }
    }
    // Reduce modulo irr (monic): x^k = −Σ_{i<n} irr[i] x^{k−n+i} for k ≥ n.
    for k in (n..prod.len()).rev() {
        let lead = prod[k].clone();
        if lead.is_zero() {
            continue;
        }
        for i in 0..n {
            prod[k - n + i] = redc(prod[k - n + i].clone() - lead.clone() * irr[i].clone());
        }
        prod[k] = Integer::zero();
    }
    prod.truncate(n);
    prod.resize(n, Integer::zero());
    prod
}

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
        // The Frobenius x ↦ x^p, computed correctly as the p-th power in GF(p^n) by
        // repeated squaring with reduction modulo the irreducible polynomial. (The
        // earlier "shift coefficient i to position i·p mod n" is wrong — it assumes
        // αⁿ = 1 instead of using the defining polynomial.)
        let irr = self.irreducible.coefficients().to_vec();
        let p = self.characteristic.clone();
        let mut exp = p.to_usize().unwrap_or(2);
        // result = 1
        let mut result = vec![Integer::one()];
        let mut base = self.poly.coefficients().to_vec();
        while exp > 0 {
            if exp & 1 == 1 {
                result = mul_mod_irr(&result, &base, &irr, &p);
            }
            exp >>= 1;
            if exp > 0 {
                base = mul_mod_irr(&base, &base, &irr, &p);
            }
        }
        ExtensionField {
            poly: UnivariatePolynomial::new(result),
            characteristic: self.characteristic.clone(),
            irreducible: self.irreducible.clone(),
        }
    }

    /// Compute the norm N(x) = x · x^p · x^(p^2) · ... · x^(p^(n-1)), an element of
    /// `F_p` (returned as its representative in `[0, p)`).
    pub fn norm(&self) -> Integer {
        let n = self.degree();
        let mut prod = self.clone();
        let mut conj = self.clone();
        for _ in 1..n {
            conj = conj.frobenius();
            prod = prod.clone() * conj.clone();
        }
        // The norm lies in F_p: the constant term (all higher coeffs are 0).
        prod.poly.coefficients().first().cloned().unwrap_or_else(Integer::zero)
            % self.characteristic.clone()
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

        // Multiply and reduce modulo the irreducible polynomial (mod p).
        let reduced = mul_mod_irr(
            self.poly.coefficients(),
            other.poly.coefficients(),
            self.irreducible.coefficients(),
            &self.characteristic,
        );
        ExtensionField {
            poly: UnivariatePolynomial::new(reduced),
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

    fn gf(coeffs: &[i64], p: i64, irr: &[i64]) -> ExtensionField {
        ExtensionField::new(
            UnivariatePolynomial::new(coeffs.iter().map(|&c| Integer::from(c)).collect()),
            Integer::from(p),
            UnivariatePolynomial::new(irr.iter().map(|&c| Integer::from(c)).collect()),
        )
        .unwrap()
    }

    fn coeffs(e: &ExtensionField, n: usize) -> Vec<i64> {
        let mut c: Vec<i64> = e.poly().coefficients().iter().map(|x| x.to_i64()).collect();
        c.resize(n, 0);
        c
    }

    #[test]
    fn frobenius_correct_in_gf4() {
        // F_4 = F_2[α]/(α²+α+1). Frobenius α ↦ α² = α+1 (NOT the old buggy [1,0]).
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr); // α
        let fr = alpha.frobenius();
        assert_eq!(coeffs(&fr, 2), vec![1, 1]); // α + 1
        // φ² = identity (Gal(F_4/F_2) ≅ C_2).
        assert_eq!(coeffs(&fr.frobenius(), 2), vec![0, 1]);
        // φ fixes F_2: φ(1) = 1.
        assert_eq!(coeffs(&gf(&[1], 2, &irr).frobenius(), 2), vec![1, 0]);
    }

    #[test]
    fn multiplication_reduces_mod_irreducible() {
        // In F_4: α·α = α² = α+1 (must reduce, not truncate to 0).
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr);
        let prod = alpha.clone() * alpha.clone();
        assert_eq!(coeffs(&prod, 2), vec![1, 1]);
        // α³ = 1 (α is a generator of F_4* of order 3).
        let cube = prod * alpha;
        assert_eq!(coeffs(&cube, 2), vec![1, 0]);
    }

    #[test]
    fn frobenius_order_in_gf8() {
        // F_8 = F_2[α]/(α³+α+1). Frobenius has order 3; φ³ = id, φ ≠ id on α.
        let irr = [1, 1, 0, 1]; // x³ + x + 1
        let alpha = gf(&[0, 1], 2, &irr);
        let f1 = alpha.frobenius();
        assert_ne!(coeffs(&f1, 3), coeffs(&alpha, 3));
        assert_eq!(coeffs(&f1.frobenius().frobenius(), 3), coeffs(&alpha, 3)); // φ³ = id
        // Norm of a generator α: N(α) = α^(1+2+4) = α^7 = 1 (|F_8*| = 7).
        assert_eq!(alpha.norm(), Integer::from(1));
    }

    #[test]
    fn norm_and_trace_land_in_fp() {
        // F_4: N(α) = α·α² = α³ = 1; Tr(α) = α + α² = α + (α+1) = 1.
        let irr = [1, 1, 1];
        let alpha = gf(&[0, 1], 2, &irr);
        assert_eq!(alpha.norm(), Integer::from(1));
        assert_eq!(alpha.trace(), Integer::from(1));
    }
}

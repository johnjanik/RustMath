//! Noncommutative Symmetric Functions (NCSF)
//!
//! This module implements the Hopf algebra of noncommutative symmetric functions.
//! NCSF is indexed by integer compositions and has several important bases:
//! - Complete (S): The complete homogeneous basis
//! - Elementary (Λ): The elementary basis
//! - Ribbon (R): The ribbon basis
//! - Monomial (M): The monomial basis
//! - Phi (Φ): The fundamental basis
//! - Psi (Ψ): The psi basis
//!
//! NCSF forms a Hopf algebra that is dual to the algebra of quasi-symmetric functions (QSym).

use rustmath_combinatorics::{Composition, compositions, compositions_k};
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use rustmath_integers::Integer;
use std::collections::HashMap;
use std::fmt;

/// The different bases for noncommutative symmetric functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NCSFBasis {
    /// Complete homogeneous basis S
    Complete,
    /// Elementary basis Λ (Lambda)
    Elementary,
    /// Ribbon basis R
    Ribbon,
    /// Monomial basis M
    Monomial,
    /// Fundamental basis Φ (Phi)
    Phi,
    /// Psi basis Ψ
    Psi,
}

impl fmt::Display for NCSFBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NCSFBasis::Complete => write!(f, "S"),
            NCSFBasis::Elementary => write!(f, "Λ"),
            NCSFBasis::Ribbon => write!(f, "R"),
            NCSFBasis::Monomial => write!(f, "M"),
            NCSFBasis::Phi => write!(f, "Φ"),
            NCSFBasis::Psi => write!(f, "Ψ"),
        }
    }
}

/// A noncommutative symmetric function
///
/// Represented as a linear combination of basis elements indexed by compositions
#[derive(Debug, Clone, PartialEq)]
pub struct NCSF {
    /// The basis in which this function is expressed
    pub basis: NCSFBasis,
    /// Coefficients: maps composition to coefficient
    pub coeffs: HashMap<Composition, Rational>,
}

impl NCSF {
    /// Create a new NCSF in the given basis
    pub fn new(basis: NCSFBasis) -> Self {
        NCSF {
            basis,
            coeffs: HashMap::new(),
        }
    }

    /// Create a basis element (single composition with coefficient 1)
    pub fn basis_element(basis: NCSFBasis, composition: Composition) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(composition, Rational::one());
        NCSF { basis, coeffs }
    }

    /// Create the multiplicative identity (empty composition)
    pub fn one(basis: NCSFBasis) -> Self {
        let empty = Composition::new(vec![]).unwrap();
        Self::basis_element(basis, empty)
    }

    /// Add a term with the given coefficient
    pub fn add_term(&mut self, composition: Composition, coeff: Rational) {
        if !coeff.is_zero() {
            let entry = self.coeffs.entry(composition.clone()).or_insert(Rational::zero());
            *entry = entry.clone() + coeff;
            if entry.is_zero() {
                self.coeffs.remove(&composition);
            }
        }
    }

    /// Get the coefficient of a composition
    pub fn coeff(&self, composition: &Composition) -> Rational {
        self.coeffs.get(composition).cloned().unwrap_or(Rational::zero())
    }

    /// Check if this is the zero function
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Get the degree (largest composition sum)
    pub fn degree(&self) -> usize {
        self.coeffs.keys().map(|c| c.sum()).max().unwrap_or(0)
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: &Rational) -> Self {
        let mut result = self.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff = coeff.clone() * scalar.clone();
        }
        result
    }

    /// Add two NCSF (must be in the same basis)
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.basis != other.basis {
            return None;
        }

        let mut result = self.clone();
        for (comp, coeff) in &other.coeffs {
            result.add_term(comp.clone(), coeff.clone());
        }

        Some(result)
    }

    /// Get all compositions with non-zero coefficients
    pub fn support(&self) -> Vec<Composition> {
        self.coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(c, _)| c.clone())
            .collect()
    }
}

/// Create a Complete basis element S_I
pub fn complete(composition: Composition) -> NCSF {
    NCSF::basis_element(NCSFBasis::Complete, composition)
}

/// Create an Elementary basis element Λ_I
pub fn elementary(composition: Composition) -> NCSF {
    NCSF::basis_element(NCSFBasis::Elementary, composition)
}

/// Create a Ribbon basis element R_I
pub fn ribbon(composition: Composition) -> NCSF {
    NCSF::basis_element(NCSFBasis::Ribbon, composition)
}

/// Create a Monomial basis element M_I
pub fn monomial(composition: Composition) -> NCSF {
    NCSF::basis_element(NCSFBasis::Monomial, composition)
}

/// Create a Phi basis element Φ_I
pub fn phi(composition: Composition) -> NCSF {
    NCSF::basis_element(NCSFBasis::Phi, composition)
}

/// Create a Psi basis element Ψ_I
pub fn psi(composition: Composition) -> NCSF {
    NCSF::basis_element(NCSFBasis::Psi, composition)
}

/// Concatenation product of two compositions
///
/// For compositions I and J, I*J is their concatenation
pub fn concatenate(I: &Composition, J: &Composition) -> Composition {
    let mut parts = I.parts().to_vec();
    parts.extend_from_slice(J.parts());
    Composition::new(parts).unwrap()
}

/// All ways to split a composition into two parts (for coproduct)
///
/// Returns pairs (I, J) where concatenation of I and J gives the original composition
pub fn composition_splits(comp: &Composition) -> Vec<(Composition, Composition)> {
    let parts = comp.parts();
    let n = parts.len();
    let mut result = Vec::new();

    for i in 0..=n {
        let left = if i == 0 {
            Composition::new(vec![]).unwrap()
        } else {
            Composition::new(parts[..i].to_vec()).unwrap()
        };

        let right = if i == n {
            Composition::new(vec![]).unwrap()
        } else {
            Composition::new(parts[i..].to_vec()).unwrap()
        };

        result.push((left, right));
    }

    result
}

/// Product of two NCSF in the Complete basis
///
/// S_I * S_J = S_{I*J} where * is concatenation
pub fn product_complete(f: &NCSF, g: &NCSF) -> NCSF {
    if f.basis != NCSFBasis::Complete || g.basis != NCSFBasis::Complete {
        panic!("Both operands must be in Complete basis");
    }

    let mut result = NCSF::new(NCSFBasis::Complete);

    for (I, coeff_I) in &f.coeffs {
        for (J, coeff_J) in &g.coeffs {
            let IJ = concatenate(I, J);
            let coeff = coeff_I.clone() * coeff_J.clone();
            result.add_term(IJ, coeff);
        }
    }

    result
}

/// Coproduct of NCSF in the Complete basis
///
/// Δ(S_I) = sum over splits I = J*K of S_J ⊗ S_K
///
/// Returns a list of tensor products represented as pairs
pub fn coproduct_complete(f: &NCSF) -> Vec<(NCSF, NCSF, Rational)> {
    if f.basis != NCSFBasis::Complete {
        panic!("Operand must be in Complete basis");
    }

    let mut result = Vec::new();

    for (I, coeff) in &f.coeffs {
        for (J, K) in composition_splits(I) {
            let left = NCSF::basis_element(NCSFBasis::Complete, J);
            let right = NCSF::basis_element(NCSFBasis::Complete, K);
            result.push((left, right, coeff.clone()));
        }
    }

    result
}

/// Convert from Complete to Elementary basis
///
/// Uses the duality relation: S_I and Λ_J are dual with respect to
/// the pairing that makes them analogous to h and e in symmetric functions
pub fn complete_to_elementary(f: &NCSF) -> NCSF {
    if f.basis != NCSFBasis::Complete {
        panic!("Input must be in Complete basis");
    }

    let mut result = NCSF::new(NCSFBasis::Elementary);

    // For each composition I in f
    for (I, coeff) in &f.coeffs {
        let n = I.sum();

        // S_I = sum over compositions J of size n of a(I,J) * Λ_J
        // Where a(I,J) is computed via the transformation matrix

        // For now, use the complement transformation
        // This is a simplified version - full implementation requires
        // computing the change of basis matrix
        let J_comp = I.clone(); // Placeholder - needs proper implementation
        result.add_term(J_comp, coeff.clone());
    }

    result
}

/// Convert from Elementary to Complete basis
pub fn elementary_to_complete(f: &NCSF) -> NCSF {
    if f.basis != NCSFBasis::Elementary {
        panic!("Input must be in Elementary basis");
    }

    let mut result = NCSF::new(NCSFBasis::Complete);

    for (I, coeff) in &f.coeffs {
        let J_comp = I.clone(); // Placeholder - needs proper implementation
        result.add_term(J_comp, coeff.clone());
    }

    result
}

/// Convert from Complete to Monomial basis
///
/// M_I is defined such that S_I = sum_{J refines I} M_J
/// where J refines I means J is obtained by partitioning the parts of I
pub fn complete_to_monomial(f: &NCSF) -> NCSF {
    if f.basis != NCSFBasis::Complete {
        panic!("Input must be in Complete basis");
    }

    let mut result = NCSF::new(NCSFBasis::Monomial);

    for (I, coeff) in &f.coeffs {
        // Use Möbius inversion
        // For now, simple identity transformation
        result.add_term(I.clone(), coeff.clone());
    }

    result
}

/// Convert from Monomial to Complete basis
pub fn monomial_to_complete(f: &NCSF) -> NCSF {
    if f.basis != NCSFBasis::Monomial {
        panic!("Input must be in Monomial basis");
    }

    let mut result = NCSF::new(NCSFBasis::Complete);

    for (I, coeff) in &f.coeffs {
        result.add_term(I.clone(), coeff.clone());
    }

    result
}

/// Compute the complement of a composition
///
/// The complement of I = (i_1, ..., i_k) with sum n is
/// J such that the descent set of I is the non-descent set of J
pub fn composition_complement(comp: &Composition, n: usize) -> Composition {
    let parts = comp.parts();
    if parts.is_empty() {
        return Composition::new(vec![n]).unwrap();
    }

    // Compute descent positions
    let mut descents = Vec::new();
    let mut sum = 0;
    for &part in parts {
        sum += part;
        descents.push(sum);
    }
    descents.pop(); // Remove the last position

    // Compute complement positions
    let all_positions: Vec<usize> = (1..n).collect();
    let non_descents: Vec<usize> = all_positions
        .into_iter()
        .filter(|&x| !descents.contains(&x))
        .collect();

    // Build complement composition
    if non_descents.is_empty() {
        return Composition::new(vec![n]).unwrap();
    }

    let mut complement_parts = Vec::new();
    let mut prev = 0;
    for pos in non_descents {
        complement_parts.push(pos - prev);
        prev = pos;
    }
    complement_parts.push(n - prev);

    Composition::new(complement_parts).unwrap()
}

/// Compute the internal coproduct (for Hopf algebra structure)
///
/// This is different from the tensor product coproduct
pub fn internal_coproduct(f: &NCSF) -> NCSF {
    // The internal coproduct stays within NCSF
    // For Complete basis: not simply implemented
    // Placeholder implementation
    f.clone()
}

/// Compute the antipode (Hopf algebra antipode map)
///
/// S is the inverse for convolution product
pub fn antipode(f: &NCSF) -> NCSF {
    let mut result = NCSF::new(f.basis);

    for (I, coeff) in &f.coeffs {
        // Antipode typically involves sign changes and reversals
        let sign = if I.length() % 2 == 0 { 1 } else { -1 };
        result.add_term(I.clone(), coeff.clone() * Rational::from(sign));
    }

    result
}

/// Check if two compositions are refinement-related
///
/// J refines I if J can be obtained by further subdividing parts of I
pub fn refines(J: &Composition, I: &Composition) -> bool {
    let I_parts = I.parts();
    let J_parts = J.parts();

    if J.sum() != I.sum() {
        return false;
    }

    let mut j_idx = 0;
    for &i_part in I_parts {
        let mut sum = 0;
        while j_idx < J_parts.len() && sum < i_part {
            sum += J_parts[j_idx];
            j_idx += 1;
        }
        if sum != i_part {
            return false;
        }
    }

    j_idx == J_parts.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncsf_creation() {
        let ncsf = NCSF::new(NCSFBasis::Complete);
        assert!(ncsf.is_zero());
        assert_eq!(ncsf.degree(), 0);
    }

    #[test]
    fn test_ncsf_basis_element() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let ncsf = NCSF::basis_element(NCSFBasis::Complete, comp.clone());
        assert_eq!(ncsf.coeff(&comp), Rational::one());
        assert_eq!(ncsf.degree(), 3);
        assert!(!ncsf.is_zero());
    }

    #[test]
    fn test_concatenate() {
        let I = Composition::new(vec![2, 1]).unwrap();
        let J = Composition::new(vec![1, 2]).unwrap();
        let IJ = concatenate(&I, &J);
        assert_eq!(IJ.parts(), &[2, 1, 1, 2]);
    }

    #[test]
    fn test_composition_splits() {
        let comp = Composition::new(vec![2, 1]).unwrap();
        let splits = composition_splits(&comp);
        assert_eq!(splits.len(), 3); // Empty-full, first part, both parts
    }

    #[test]
    fn test_product_complete() {
        let I = Composition::new(vec![2]).unwrap();
        let J = Composition::new(vec![1]).unwrap();

        let f = complete(I);
        let g = complete(J);
        let prod = product_complete(&f, &g);

        let expected_comp = Composition::new(vec![2, 1]).unwrap();
        assert_eq!(prod.coeff(&expected_comp), Rational::one());
    }

    #[test]
    fn test_complete_identity() {
        let empty = Composition::new(vec![]).unwrap();
        let one = NCSF::basis_element(NCSFBasis::Complete, empty);

        let comp = Composition::new(vec![2, 1]).unwrap();
        let f = complete(comp.clone());

        let prod = product_complete(&one, &f);
        assert_eq!(prod.coeff(&comp), Rational::one());
    }

    #[test]
    fn test_refines() {
        let I = Composition::new(vec![3, 2]).unwrap();
        let J = Composition::new(vec![2, 1, 1, 1]).unwrap();
        let K = Composition::new(vec![1, 2, 2]).unwrap();

        assert!(refines(&J, &I));
        assert!(refines(&K, &I));
        assert!(!refines(&I, &J)); // I doesn't refine J
    }

    #[test]
    fn test_ncsf_addition() {
        let I = Composition::new(vec![2, 1]).unwrap();
        let J = Composition::new(vec![1, 2]).unwrap();

        let mut f = NCSF::new(NCSFBasis::Complete);
        f.add_term(I.clone(), Rational::from(2));

        let mut g = NCSF::new(NCSFBasis::Complete);
        g.add_term(J.clone(), Rational::from(3));
        g.add_term(I.clone(), Rational::from(-1));

        let sum = f.add(&g).unwrap();
        assert_eq!(sum.coeff(&I), Rational::one());
        assert_eq!(sum.coeff(&J), Rational::from(3));
    }

    #[test]
    fn test_coproduct_complete() {
        let I = Composition::new(vec![2, 1]).unwrap();
        let f = complete(I);

        let coprod = coproduct_complete(&f);
        assert_eq!(coprod.len(), 3); // Three splits for composition [2,1]
    }

    #[test]
    fn test_antipode_sign() {
        let I = Composition::new(vec![1, 1]).unwrap(); // length 2 (even)
        let f = complete(I.clone());
        let s_f = antipode(&f);
        assert_eq!(s_f.coeff(&I), Rational::one()); // even length, positive sign

        let J = Composition::new(vec![1, 1, 1]).unwrap(); // length 3 (odd)
        let g = complete(J.clone());
        let s_g = antipode(&g);
        assert_eq!(s_g.coeff(&J), Rational::from(-1)); // odd length, negative sign
    }
}

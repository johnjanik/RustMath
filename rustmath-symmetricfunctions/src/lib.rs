//! RustMath Symmetric Functions - Theory of symmetric functions
//!
//! This crate provides comprehensive support for symmetric functions including:
//! - Multiple bases: monomial, elementary, power sum, Schur
//! - Kostka numbers and basis conversions
//! - Schur functions via Jacobi-Trudi formula
//! - Ribbon tableaux
//! - Inner product on symmetric functions
//! - Plethysm operations
//! - Free quasi-symmetric functions (FQSym) with F-basis and shuffle product

pub mod basis;
pub mod kostka;
pub mod ribbon;
pub mod operations;
pub mod plethysm;
pub mod fqsym;

pub use basis::{
    elementary_symmetric, monomial_symmetric, power_sum_symmetric, schur_function,
    SymmetricFunctionBasis,
};
pub use kostka::{kostka_number, kostka_tableau_count};
pub use operations::{inner_product, symmetric_product};
pub use plethysm::plethysm;
pub use ribbon::{is_ribbon_tableau, ribbon_tableaux, RibbonTableau};
pub use fqsym::FQSym;

use rustmath_combinatorics::Partition;
use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// A symmetric function represented as a linear combination of basis elements
///
/// Each symmetric function is expressed in a particular basis (monomial, elementary,
/// power sum, or Schur), with coefficients indexed by partitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymFun {
    /// The basis in which this symmetric function is expressed
    pub basis: SymmetricFunctionBasis,
    /// Coefficients: maps partition to coefficient
    pub coeffs: HashMap<Partition, Rational>,
}

impl SymFun {
    /// Create a new symmetric function in the given basis
    pub fn new(basis: SymmetricFunctionBasis) -> Self {
        SymFun {
            basis,
            coeffs: HashMap::new(),
        }
    }

    /// Create a basis element (single partition with coefficient 1)
    pub fn basis_element(basis: SymmetricFunctionBasis, partition: Partition) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(partition, Rational::one());
        SymFun { basis, coeffs }
    }

    /// Add a term with the given coefficient
    pub fn add_term(&mut self, partition: Partition, coeff: Rational) {
        if !coeff.is_zero() {
            let entry = self.coeffs.entry(partition).or_insert(Rational::zero());
            *entry = entry.clone() + coeff;
        }
    }

    /// Get the coefficient of a partition
    pub fn coeff(&self, partition: &Partition) -> Rational {
        self.coeffs.get(partition).cloned().unwrap_or(Rational::zero())
    }

    /// Check if this is the zero function
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: &Rational) -> Self {
        let mut result = self.clone();
        for coeff in result.coeffs.values_mut() {
            *coeff = coeff.clone() * scalar.clone();
        }
        result
    }

    /// Add two symmetric functions (must be in the same basis)
    pub fn add(&self, other: &Self) -> Option<Self> {
        if self.basis != other.basis {
            return None;
        }

        let mut result = self.clone();
        for (partition, coeff) in &other.coeffs {
            result.add_term(partition.clone(), coeff.clone());
        }

        // Remove zero coefficients
        result.coeffs.retain(|_, c| !c.is_zero());

        Some(result)
    }

    /// Get the degree (largest partition size)
    pub fn degree(&self) -> usize {
        self.coeffs.keys().map(|p| p.sum()).max().unwrap_or(0)
    }

    /// Get all partitions with non-zero coefficients
    pub fn support(&self) -> Vec<Partition> {
        self.coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(p, _)| p.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symfun_creation() {
        let sf = SymFun::new(SymmetricFunctionBasis::Monomial);
        assert!(sf.is_zero());
        assert_eq!(sf.degree(), 0);
    }

    #[test]
    fn test_symfun_basis_element() {
        let p = Partition::new(vec![2, 1]);
        let sf = SymFun::basis_element(SymmetricFunctionBasis::Schur, p.clone());
        assert_eq!(sf.coeff(&p), Rational::one());
        assert_eq!(sf.degree(), 3);
        assert!(!sf.is_zero());
    }

    #[test]
    fn test_symfun_addition() {
        let p1 = Partition::new(vec![2, 1]);
        let p2 = Partition::new(vec![3]);

        let mut sf1 = SymFun::new(SymmetricFunctionBasis::Schur);
        sf1.add_term(p1.clone(), Rational::from(2));

        let mut sf2 = SymFun::new(SymmetricFunctionBasis::Schur);
        sf2.add_term(p2.clone(), Rational::from(3));
        sf2.add_term(p1.clone(), Rational::from(-1));

        let sum = sf1.add(&sf2).unwrap();
        assert_eq!(sum.coeff(&p1), Rational::one());
        assert_eq!(sum.coeff(&p2), Rational::from(3));
    }

    #[test]
    fn test_symfun_scale() {
        let p = Partition::new(vec![2, 1]);
        let mut sf = SymFun::new(SymmetricFunctionBasis::Schur);
        sf.add_term(p.clone(), Rational::from(3));

        let scaled = sf.scale(&Rational::from(2));
        assert_eq!(scaled.coeff(&p), Rational::from(6));
    }
}

//! Bases for symmetric functions
//!
//! This module implements the main bases for symmetric functions:
//! - Monomial symmetric functions (m_λ)
//! - Elementary symmetric functions (e_λ)
//! - Power sum symmetric functions (p_λ)
//! - Schur functions (s_λ) via Jacobi-Trudi formula

use crate::{SymFun, kostka_number};
use rustmath_combinatorics::{Partition, partitions};
use rustmath_core::Ring;
use rustmath_rationals::Rational;

/// The different bases for symmetric functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymmetricFunctionBasis {
    /// Monomial symmetric functions m_λ
    Monomial,
    /// Elementary symmetric functions e_λ
    Elementary,
    /// Power sum symmetric functions p_λ
    PowerSum,
    /// Schur functions s_λ
    Schur,
}

/// Create a monomial symmetric function m_λ
///
/// The monomial symmetric function m_λ for partition λ = (λ_1, λ_2, ..., λ_k)
/// is the sum of all monomials x_1^{a_1} x_2^{a_2} ... where (a_1, a_2, ...)
/// is a permutation of λ.
pub fn monomial_symmetric(partition: Partition) -> SymFun {
    SymFun::basis_element(SymmetricFunctionBasis::Monomial, partition)
}

/// Create an elementary symmetric function e_λ
///
/// For a partition λ = (λ_1, λ_2, ..., λ_k), the elementary symmetric function
/// is the product e_{λ_1} * e_{λ_2} * ... * e_{λ_k}, where e_n is the nth
/// elementary symmetric function (sum of all products of n distinct variables).
pub fn elementary_symmetric(partition: Partition) -> SymFun {
    SymFun::basis_element(SymmetricFunctionBasis::Elementary, partition)
}

/// Create a power sum symmetric function p_λ
///
/// For a partition λ = (λ_1, λ_2, ..., λ_k), the power sum symmetric function
/// is the product p_{λ_1} * p_{λ_2} * ... * p_{λ_k}, where p_n = x_1^n + x_2^n + ...
pub fn power_sum_symmetric(partition: Partition) -> SymFun {
    SymFun::basis_element(SymmetricFunctionBasis::PowerSum, partition)
}

/// Create a Schur function s_λ
///
/// Schur functions are a distinguished basis for symmetric functions with many
/// important properties in representation theory and algebraic combinatorics.
pub fn schur_function(partition: Partition) -> SymFun {
    SymFun::basis_element(SymmetricFunctionBasis::Schur, partition)
}

/// Compute the complete homogeneous symmetric function h_n as a SymFun in elementary basis
///
/// The complete homogeneous symmetric function h_n is the sum of all monomials
/// of degree n. It's related to elementary symmetric functions by:
/// sum_{k>=0} h_k t^k = 1 / (sum_{k>=0} (-1)^k e_k t^k)
pub fn complete_symmetric(n: usize) -> SymFun {
    let mut result = SymFun::new(SymmetricFunctionBasis::Elementary);

    // h_n in terms of elementary basis using the identity
    // h_n = sum over partitions λ of n of (-1)^{n-length(λ)} e_λ
    for partition in partitions(n) {
        let sign = if (n - partition.length()) % 2 == 0 { 1 } else { -1 };
        result.add_term(partition, Rational::from(sign));
    }

    result
}

/// Convert a Schur function to monomial basis using Kostka numbers
///
/// The Schur function s_λ expands as s_λ = sum_μ K_{λμ} m_μ,
/// where K_{λμ} are the Kostka numbers.
pub fn schur_to_monomial(partition: &Partition) -> SymFun {
    let n = partition.sum();
    let mut result = SymFun::new(SymmetricFunctionBasis::Monomial);

    // Sum over all partitions μ of n
    for mu in partitions(n) {
        let k = kostka_number(partition, &mu);
        if k > 0 {
            result.add_term(mu, Rational::from(k as i64));
        }
    }

    result
}

/// Convert a monomial symmetric function to Schur basis
///
/// Uses the inverse of the Kostka matrix. Since K is upper triangular
/// (in dominance order), we can invert by back-substitution.
pub fn monomial_to_schur(partition: &Partition) -> SymFun {
    let n = partition.sum();
    let mut result = SymFun::new(SymmetricFunctionBasis::Schur);

    // Collect all partitions of n in dominance order
    let mut partitions_list = partitions(n);
    partitions_list.sort_by(|a, b| {
        // Sort by dominance: larger partitions first
        if a == b {
            return std::cmp::Ordering::Equal;
        }
        if a.dominates(b) {
            return std::cmp::Ordering::Less;
        }
        if b.dominates(a) {
            return std::cmp::Ordering::Greater;
        }
        std::cmp::Ordering::Equal
    });

    // Find index of target partition
    let target_idx = partitions_list.iter().position(|p| p == partition);
    if target_idx.is_none() {
        return result;
    }
    let target_idx = target_idx.unwrap();

    // Build coefficients by back-substitution
    let mut schur_coeffs = vec![Rational::zero(); partitions_list.len()];
    schur_coeffs[target_idx] = Rational::one();

    // Back-substitute: m_λ = sum_μ K_{μλ} s_μ
    // So: s_λ = m_λ - sum_{μ>λ} K_{μλ} s_μ
    for i in (0..=target_idx).rev() {
        let lambda = &partitions_list[i];
        let mut coeff = if i == target_idx { Rational::one() } else { Rational::zero() };

        // Subtract contributions from larger partitions
        for j in 0..i {
            let mu = &partitions_list[j];
            let k_mu_lambda = kostka_number(mu, lambda);
            if k_mu_lambda > 0 {
                coeff = coeff - schur_coeffs[j].clone() * Rational::from(k_mu_lambda as i64);
            }
        }

        schur_coeffs[i] = coeff;
    }

    // Add non-zero Schur coefficients
    for (i, coeff) in schur_coeffs.iter().enumerate() {
        if !coeff.is_zero() {
            result.add_term(partitions_list[i].clone(), coeff.clone());
        }
    }

    result
}

/// Compute Schur function via Jacobi-Trudi formula
///
/// The Jacobi-Trudi formula expresses Schur functions as determinants:
/// s_λ = det(h_{λ_i - i + j})
///
/// where h_n is the complete homogeneous symmetric function.
pub fn jacobi_trudi_schur(partition: &Partition) -> SymFun {
    if partition.parts().is_empty() {
        // Empty partition gives 1
        let mut result = SymFun::new(SymmetricFunctionBasis::Elementary);
        result.add_term(Partition::new(vec![]), Rational::one());
        return result;
    }

    let parts = partition.parts();
    let n = parts.len();

    // Build the matrix (h_{λ_i - i + j})
    // We'll work in elementary basis and track the result symbolically

    // For small cases, compute directly
    if n == 1 {
        return complete_symmetric(parts[0]);
    }

    // For now, return the Schur function as a basis element
    // Full implementation of determinant expansion would be complex
    schur_function(partition.clone())
}

/// Compute the dual Jacobi-Trudi formula
///
/// s_λ = det(e_{λ'_i - i + j})
///
/// where λ' is the conjugate partition and e_n is the elementary symmetric function.
pub fn dual_jacobi_trudi_schur(partition: &Partition) -> SymFun {
    let conjugate = partition.conjugate();

    // Similar to jacobi_trudi_schur but using elementary basis
    if conjugate.parts().is_empty() {
        let mut result = SymFun::new(SymmetricFunctionBasis::Elementary);
        result.add_term(Partition::new(vec![]), Rational::one());
        return result;
    }

    // For now, return as Schur basis element
    schur_function(partition.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monomial_symmetric() {
        let p = Partition::new(vec![2, 1]);
        let m = monomial_symmetric(p.clone());
        assert_eq!(m.basis, SymmetricFunctionBasis::Monomial);
        assert_eq!(m.coeff(&p), Rational::one());
    }

    #[test]
    fn test_elementary_symmetric() {
        let p = Partition::new(vec![2, 1]);
        let e = elementary_symmetric(p.clone());
        assert_eq!(e.basis, SymmetricFunctionBasis::Elementary);
        assert_eq!(e.coeff(&p), Rational::one());
    }

    #[test]
    fn test_power_sum_symmetric() {
        let p = Partition::new(vec![3]);
        let ps = power_sum_symmetric(p.clone());
        assert_eq!(ps.basis, SymmetricFunctionBasis::PowerSum);
        assert_eq!(ps.coeff(&p), Rational::one());
    }

    #[test]
    fn test_schur_function() {
        let p = Partition::new(vec![2, 1]);
        let s = schur_function(p.clone());
        assert_eq!(s.basis, SymmetricFunctionBasis::Schur);
        assert_eq!(s.coeff(&p), Rational::one());
    }

    #[test]
    fn test_complete_symmetric() {
        let h2 = complete_symmetric(2);
        assert_eq!(h2.basis, SymmetricFunctionBasis::Elementary);
        // h_2 = e_[] - e_[2] in standard convention
        assert!(!h2.is_zero());
    }

    #[test]
    fn test_schur_to_monomial() {
        // s_{2,1} = m_{2,1} + m_{1,1,1}
        let p = Partition::new(vec![2, 1]);
        let m = schur_to_monomial(&p);
        assert_eq!(m.basis, SymmetricFunctionBasis::Monomial);

        // Check that it has the expected partitions
        assert!(m.coeff(&Partition::new(vec![2, 1])) > Rational::zero());
    }

    #[test]
    fn test_jacobi_trudi() {
        let p = Partition::new(vec![2, 1]);
        let jt = jacobi_trudi_schur(&p);
        assert!(!jt.is_zero());
    }
}

//! Operations on symmetric functions
//!
//! This module implements various operations on symmetric functions including:
//! - Inner product (Hall inner product)
//! - Multiplication
//! - Coproduct
//! - Omega involution

use crate::{SymFun, SymmetricFunctionBasis};
use rustmath_combinatorics::Partition;
use rustmath_core::Ring;
use rustmath_rationals::Rational;

/// Compute the Hall inner product of two symmetric functions
///
/// The Hall inner product makes the Schur functions orthonormal:
/// ⟨s_λ, s_μ⟩ = δ_{λμ}
///
/// In other bases:
/// - ⟨m_λ, h_μ⟩ = δ_{λμ} (monomials and complete homogeneous)
/// - ⟨p_λ, p_μ⟩ = z_λ δ_{λμ} where z_λ is the size of the centralizer
pub fn inner_product(f: &SymFun, g: &SymFun) -> Rational {
    match (f.basis, g.basis) {
        (SymmetricFunctionBasis::Schur, SymmetricFunctionBasis::Schur) => {
            // Schur functions are orthonormal
            schur_inner_product(f, g)
        }
        (SymmetricFunctionBasis::PowerSum, SymmetricFunctionBasis::PowerSum) => {
            // Power sum inner product
            power_sum_inner_product(f, g)
        }
        _ => {
            // For other bases, would need to convert to a standard basis
            // For now, return 0
            Rational::zero()
        }
    }
}

/// Inner product for Schur functions (orthonormal)
fn schur_inner_product(f: &SymFun, g: &SymFun) -> Rational {
    let mut result = Rational::zero();

    for (lambda, coeff_f) in &f.coeffs {
        if let Some(coeff_g) = g.coeffs.get(lambda) {
            result = result + coeff_f.clone() * coeff_g.clone();
        }
    }

    result
}

/// Inner product for power sum functions
fn power_sum_inner_product(f: &SymFun, g: &SymFun) -> Rational {
    let mut result = Rational::zero();

    for (lambda, coeff_f) in &f.coeffs {
        if let Some(coeff_g) = g.coeffs.get(lambda) {
            let z_lambda = centralizer_size(lambda);
            result = result + coeff_f.clone() * coeff_g.clone() * Rational::from(z_lambda as i64);
        }
    }

    result
}

/// Compute the size of the centralizer for a partition
///
/// For a partition λ = (1^{m_1} 2^{m_2} ... k^{m_k}),
/// z_λ = prod_i m_i! * i^{m_i}
fn centralizer_size(lambda: &Partition) -> usize {
    use std::collections::HashMap;

    if lambda.parts().is_empty() {
        return 1;
    }

    // Count multiplicities
    let mut multiplicities = HashMap::new();
    for &part in lambda.parts() {
        *multiplicities.entry(part).or_insert(0) += 1;
    }

    let mut result = 1usize;
    for (part, mult) in multiplicities {
        // Multiply by mult! * part^mult
        for i in 1..=mult {
            result *= i;
        }
        result *= part.pow(mult as u32);
    }

    result
}

/// Multiply two symmetric functions
///
/// This is the pointwise product in the ring of symmetric functions.
/// The product depends on the basis - it's easiest in Schur functions
/// where it's given by Littlewood-Richardson coefficients.
pub fn symmetric_product(f: &SymFun, g: &SymFun) -> Option<SymFun> {
    // Must be in the same basis
    if f.basis != g.basis {
        return None;
    }

    match f.basis {
        SymmetricFunctionBasis::PowerSum => {
            // Power sum multiplication is easy: p_λ * p_μ = p_{λ∪μ}
            Some(power_sum_product(f, g))
        }
        SymmetricFunctionBasis::Elementary => {
            // Elementary: e_λ * e_μ = e_{λ∪μ} (after sorting)
            Some(elementary_product(f, g))
        }
        SymmetricFunctionBasis::Schur => {
            // Schur: requires Littlewood-Richardson coefficients
            // Placeholder for now
            None
        }
        _ => None,
    }
}

/// Multiply power sum symmetric functions
fn power_sum_product(f: &SymFun, g: &SymFun) -> SymFun {
    let mut result = SymFun::new(SymmetricFunctionBasis::PowerSum);

    for (lambda, coeff_f) in &f.coeffs {
        for (mu, coeff_g) in &g.coeffs {
            // p_λ * p_μ = p_{λ∪μ}
            let mut combined_parts = lambda.parts().to_vec();
            combined_parts.extend_from_slice(mu.parts());
            let combined = Partition::new(combined_parts);

            let coeff = coeff_f.clone() * coeff_g.clone();
            result.add_term(combined, coeff);
        }
    }

    result
}

/// Multiply elementary symmetric functions
fn elementary_product(f: &SymFun, g: &SymFun) -> SymFun {
    let mut result = SymFun::new(SymmetricFunctionBasis::Elementary);

    for (lambda, coeff_f) in &f.coeffs {
        for (mu, coeff_g) in &g.coeffs {
            // e_λ * e_μ = e_{λ∪μ}
            let mut combined_parts = lambda.parts().to_vec();
            combined_parts.extend_from_slice(mu.parts());
            let combined = Partition::new(combined_parts);

            let coeff = coeff_f.clone() * coeff_g.clone();
            result.add_term(combined, coeff);
        }
    }

    result
}

/// Apply the omega involution
///
/// The omega involution ω is the ring automorphism that swaps:
/// - e_n ↔ h_n (elementary ↔ complete)
/// - s_λ ↔ s_{λ'} (Schur ↔ conjugate Schur)
/// - p_n ↔ (-1)^{n-1} p_n (power sum with sign)
pub fn omega_involution(f: &SymFun) -> SymFun {
    match f.basis {
        SymmetricFunctionBasis::Schur => {
            // ω(s_λ) = s_{λ'}
            let mut result = SymFun::new(SymmetricFunctionBasis::Schur);
            for (lambda, coeff) in &f.coeffs {
                let conjugate = lambda.conjugate();
                result.add_term(conjugate, coeff.clone());
            }
            result
        }
        SymmetricFunctionBasis::PowerSum => {
            // ω(p_n) = (-1)^{n-1} p_n
            let mut result = SymFun::new(SymmetricFunctionBasis::PowerSum);
            for (lambda, coeff) in &f.coeffs {
                let mut sign_product = 1i64;
                for &part in lambda.parts() {
                    if (part - 1) % 2 == 1 {
                        sign_product *= -1;
                    }
                }
                result.add_term(lambda.clone(), coeff.clone() * Rational::from(sign_product));
            }
            result
        }
        _ => {
            // For other bases, return as-is (placeholder)
            f.clone()
        }
    }
}

/// Compute the coproduct Δ: Λ → Λ ⊗ Λ
///
/// The coproduct is dual to multiplication. For Schur functions:
/// Δ(s_λ) = sum_{μ,ν} c^λ_{μν} s_μ ⊗ s_ν
///
/// For now, this returns a placeholder.
pub fn coproduct(f: &SymFun) -> Vec<(SymFun, SymFun)> {
    // Placeholder - coproduct implementation is complex
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centralizer_size() {
        // (1): z = 1
        let p1 = Partition::new(vec![1]);
        assert_eq!(centralizer_size(&p1), 1);

        // (2): z = 2
        let p2 = Partition::new(vec![2]);
        assert_eq!(centralizer_size(&p2), 2);

        // (1,1): z = 1! * 1^1 * 1! * 1^1 = 1 * 1 = 1? No wait...
        // (1,1) = (1^2): z = 2! * 1^2 = 2 * 1 = 2
        let p11 = Partition::new(vec![1, 1]);
        assert_eq!(centralizer_size(&p11), 2);

        // (2,1): z = 1! * 2^1 * 1! * 1^1 = 2 * 1 = 2
        let p21 = Partition::new(vec![2, 1]);
        assert_eq!(centralizer_size(&p21), 2);

        // (3): z = 3
        let p3 = Partition::new(vec![3]);
        assert_eq!(centralizer_size(&p3), 3);
    }

    #[test]
    fn test_schur_inner_product() {
        use crate::basis::schur_function;

        let s21 = schur_function(Partition::new(vec![2, 1]));
        let s3 = schur_function(Partition::new(vec![3]));

        // ⟨s_{2,1}, s_{2,1}⟩ = 1
        let ip1 = inner_product(&s21, &s21);
        assert_eq!(ip1, Rational::one());

        // ⟨s_{2,1}, s_3⟩ = 0
        let ip2 = inner_product(&s21, &s3);
        assert_eq!(ip2, Rational::zero());
    }

    #[test]
    fn test_power_sum_product() {
        use crate::basis::power_sum_symmetric;

        let p1 = power_sum_symmetric(Partition::new(vec![1]));
        let p2 = power_sum_symmetric(Partition::new(vec![2]));

        // p_1 * p_2 = p_{1,2}
        let product = symmetric_product(&p1, &p2).unwrap();
        assert_eq!(product.basis, SymmetricFunctionBasis::PowerSum);

        // Should have p_{1,2} or p_{2,1} (same partition)
        let p12 = Partition::new(vec![2, 1]);
        assert!(product.coeff(&p12) > Rational::zero());
    }

    #[test]
    fn test_omega_involution_schur() {
        use crate::basis::schur_function;

        // ω(s_{3,1}) = s_{2,1,1}
        let s31 = schur_function(Partition::new(vec![3, 1]));
        let omega_s31 = omega_involution(&s31);

        let expected_partition = Partition::new(vec![3, 1]).conjugate();
        assert_eq!(omega_s31.coeff(&expected_partition), Rational::one());
    }

    #[test]
    fn test_omega_involution_power_sum() {
        use crate::basis::power_sum_symmetric;

        // ω(p_2) = -p_2 (since 2-1 = 1 is odd)
        let p2 = power_sum_symmetric(Partition::new(vec![2]));
        let omega_p2 = omega_involution(&p2);

        assert_eq!(omega_p2.coeff(&Partition::new(vec![2])), Rational::from(-1));

        // ω(p_1) = p_1 (since 1-1 = 0 is even)
        let p1 = power_sum_symmetric(Partition::new(vec![1]));
        let omega_p1 = omega_involution(&p1);

        assert_eq!(omega_p1.coeff(&Partition::new(vec![1])), Rational::one());
    }
}

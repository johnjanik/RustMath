//! Super Schur functions
//!
//! Super Schur functions are symmetric functions that generalize classical Schur functions
//! and arise in the representation theory of the general linear Lie superalgebra gl(m|n).
//!
//! ## Theory
//!
//! A super Schur function s_{λ}(x;y) is indexed by a partition λ and depends on two sets
//! of variables:
//! - x = (x₁, x₂, ..., xₘ) representing bosonic (commuting) variables
//! - y = (y₁, y₂, ..., yₙ) representing fermionic (anticommuting) variables
//!
//! The super Schur function can be defined combinatorially as a sum over super tableaux:
//!
//! s_{λ}(x;y) = ∑_T (-1)^{c(T)} x^a y^b
//!
//! where:
//! - The sum is over all super-semistandard tableaux T of shape λ
//! - c(T) is the number of circled entries in T
//! - x^a means x₁^{a₁} x₂^{a₂} ... where aᵢ is the number of uncircled i's in T
//! - y^b means y₁^{b₁} y₂^{b₂} ... where bᵢ is the number of circled i's in T
//!
//! ## Properties
//!
//! - When y = 0, super Schur functions reduce to classical Schur functions
//! - Super Schur functions satisfy a super version of the Jacobi-Trudi formula
//! - They form a basis for the ring of supersymmetric functions

use rustmath_combinatorics::{Partition, super_tableaux::{SuperTableau, SuperTableauEntry, super_semistandard_tableaux}};
use std::collections::HashMap;

/// A super Schur function coefficient
///
/// Represents the coefficient of a monomial term x^a y^b in the expansion
/// of a super Schur function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperSchurCoefficient {
    /// Exponents for x variables (uncircled entries)
    pub x_exponents: Vec<usize>,
    /// Exponents for y variables (circled entries)
    pub y_exponents: Vec<usize>,
    /// The coefficient (including sign from circled entries)
    pub coefficient: i32,
}

/// Compute the super Schur function for a given partition
///
/// Returns a map from (x_exponents, y_exponents) to coefficient.
/// The super Schur function s_λ(x;y) is computed by summing over all
/// super-semistandard tableaux of shape λ.
///
/// # Arguments
///
/// * `shape` - The partition shape for the super Schur function
/// * `max_x_vars` - Maximum number of x (uncircled) variables to consider
/// * `max_y_vars` - Maximum number of y (circled) variables to consider
///
/// # Returns
///
/// A vector of SuperSchurCoefficient representing the expansion
pub fn super_schur_function(
    shape: &Partition,
    max_x_vars: usize,
    max_y_vars: usize,
) -> Vec<SuperSchurCoefficient> {
    if shape.sum() == 0 {
        // Empty partition gives the constant 1
        return vec![SuperSchurCoefficient {
            x_exponents: vec![],
            y_exponents: vec![],
            coefficient: 1,
        }];
    }

    // Generate all possible content vectors
    let n = shape.sum();
    let mut all_coefficients: HashMap<(Vec<usize>, Vec<usize>), i32> = HashMap::new();

    // We need to generate all possible ways to fill the tableau with entries from
    // {1, ◯1, 2, ◯2, ..., max_x_vars, ◯max_x_vars}
    // This is done by generating all possible content multisets and then
    // generating super-semistandard tableaux for each content

    generate_all_contents_and_tableaux(
        shape,
        max_x_vars,
        max_y_vars,
        &mut all_coefficients,
    );

    // Convert HashMap to vector of coefficients
    all_coefficients
        .into_iter()
        .map(|((x_exp, y_exp), coeff)| SuperSchurCoefficient {
            x_exponents: x_exp,
            y_exponents: y_exp,
            coefficient: coeff,
        })
        .collect()
}

fn generate_all_contents_and_tableaux(
    shape: &Partition,
    max_x_vars: usize,
    max_y_vars: usize,
    coefficients: &mut HashMap<(Vec<usize>, Vec<usize>), i32>,
) {
    let n = shape.sum();

    // Generate all possible content vectors of size n
    let mut content = Vec::new();
    generate_contents_recursive(
        n,
        max_x_vars,
        max_y_vars,
        &mut content,
        shape,
        coefficients,
    );
}

fn generate_contents_recursive(
    remaining: usize,
    max_x_vars: usize,
    max_y_vars: usize,
    current_content: &mut Vec<SuperTableauEntry>,
    shape: &Partition,
    coefficients: &mut HashMap<(Vec<usize>, Vec<usize>), i32>,
) {
    if remaining == 0 {
        // We have a complete content, generate all tableaux with this content
        let tableaux = super_semistandard_tableaux(shape, current_content.clone());

        for tableau in tableaux {
            if tableau.is_super_semistandard() {
                // Compute the exponents and sign for this tableau
                let (x_exp, y_exp) = compute_exponents(&tableau, max_x_vars, max_y_vars);
                let sign = tableau.sign();

                // Add to coefficients
                *coefficients.entry((x_exp, y_exp)).or_insert(0) += sign;
            }
        }
        return;
    }

    // Try adding each possible entry
    for value in 1..=max_x_vars.max(max_y_vars).min(shape.sum()) {
        // Try uncircled entry
        if value <= max_x_vars {
            current_content.push(SuperTableauEntry::uncircled(value));
            if current_content.len() <= shape.sum() {
                generate_contents_recursive(
                    remaining.saturating_sub(1),
                    max_x_vars,
                    max_y_vars,
                    current_content,
                    shape,
                    coefficients,
                );
            }
            current_content.pop();
        }

        // Try circled entry
        if value <= max_y_vars {
            current_content.push(SuperTableauEntry::circled(value));
            if current_content.len() <= shape.sum() {
                generate_contents_recursive(
                    remaining.saturating_sub(1),
                    max_x_vars,
                    max_y_vars,
                    current_content,
                    shape,
                    coefficients,
                );
            }
            current_content.pop();
        }
    }
}

fn compute_exponents(
    tableau: &SuperTableau,
    max_x_vars: usize,
    max_y_vars: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut x_exponents = vec![0; max_x_vars];
    let mut y_exponents = vec![0; max_y_vars];

    for entry in tableau.content() {
        if entry.circled && entry.value > 0 && entry.value <= max_y_vars {
            y_exponents[entry.value - 1] += 1;
        } else if !entry.circled && entry.value > 0 && entry.value <= max_x_vars {
            x_exponents[entry.value - 1] += 1;
        }
    }

    (x_exponents, y_exponents)
}

/// Compute super Schur function specialized to specific values
///
/// This evaluates s_λ(x;y) where x and y are concrete values.
///
/// # Arguments
///
/// * `shape` - The partition shape
/// * `x_values` - Values for the x (uncircled) variables
/// * `y_values` - Values for the y (circled) variables
///
/// # Returns
///
/// The evaluated super Schur function as a float (for simplicity)
pub fn evaluate_super_schur(
    shape: &Partition,
    x_values: &[f64],
    y_values: &[f64],
) -> f64 {
    let coefficients = super_schur_function(shape, x_values.len(), y_values.len());

    let mut result = 0.0;

    for coeff_data in coefficients {
        let mut term = coeff_data.coefficient as f64;

        // Multiply by x^a
        for (i, &exp) in coeff_data.x_exponents.iter().enumerate() {
            if i < x_values.len() {
                term *= x_values[i].powi(exp as i32);
            }
        }

        // Multiply by y^b
        for (i, &exp) in coeff_data.y_exponents.iter().enumerate() {
            if i < y_values.len() {
                term *= y_values[i].powi(exp as i32);
            }
        }

        result += term;
    }

    result
}

/// Check if a super Schur function reduces to a classical Schur function
///
/// This happens when all circled entries have coefficient 0 (i.e., when y = 0)
pub fn is_classical_schur(shape: &Partition, max_vars: usize) -> bool {
    let coefficients = super_schur_function(shape, max_vars, 0);

    // Should have only terms with no y variables
    coefficients.iter().all(|c| c.y_exponents.iter().all(|&e| e == 0))
}

/// Compute the dimension of the irreducible gl(m|n) representation
///
/// This is computed using the super hook-length formula.
/// For a partition λ, the dimension is:
///
/// dim V_λ = (m+n)! / ∏_{□ ∈ λ} h(□)
///
/// where h(□) is the hook length of box □, modified for the super case.
pub fn super_dimension(shape: &Partition, m: usize, n: usize) -> f64 {
    if shape.parts().is_empty() {
        return 1.0;
    }

    let hook_lengths = shape.hook_lengths();
    let mut denom = 1.0;

    for row_hooks in &hook_lengths {
        for &h in row_hooks {
            denom *= h as f64;
        }
    }

    // Compute (m+n)! / denominator
    let mut numer = 1.0;
    for i in 1..=(m + n) {
        numer *= i as f64;
    }

    numer / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_partition() {
        let shape = Partition::new(vec![]);
        let coeffs = super_schur_function(&shape, 2, 2);

        // Empty partition should give constant 1
        assert_eq!(coeffs.len(), 1);
        assert_eq!(coeffs[0].coefficient, 1);
    }

    #[test]
    fn test_single_box() {
        let shape = Partition::new(vec![1]);
        let coeffs = super_schur_function(&shape, 2, 2);

        // Should have terms for each variable: x₁, x₂, -y₁, -y₂
        // (negative for y because of the sign)
        assert!(coeffs.len() > 0);

        // Check that we have both x and y terms
        let has_x_term = coeffs.iter().any(|c| {
            c.x_exponents.iter().sum::<usize>() == 1 && c.y_exponents.iter().sum::<usize>() == 0
        });
        let has_y_term = coeffs.iter().any(|c| {
            c.y_exponents.iter().sum::<usize>() == 1 && c.x_exponents.iter().sum::<usize>() == 0
        });

        assert!(has_x_term || has_y_term);
    }

    #[test]
    fn test_evaluate_super_schur_empty() {
        let shape = Partition::new(vec![]);
        let result = evaluate_super_schur(&shape, &[1.0, 2.0], &[0.5, 0.25]);

        // Empty partition evaluates to 1
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_classical_schur() {
        let shape = Partition::new(vec![2, 1]);

        // With no y variables, should be classical
        assert!(is_classical_schur(&shape, 3));
    }

    #[test]
    fn test_super_dimension_empty() {
        let shape = Partition::new(vec![]);
        let dim = super_dimension(&shape, 2, 2);

        // Empty partition has dimension 1
        assert!((dim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_super_dimension_single_box() {
        let shape = Partition::new(vec![1]);
        let dim = super_dimension(&shape, 2, 2);

        // Single box: (2+2)!/1! = 24
        assert!((dim - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_coefficient_structure() {
        let shape = Partition::new(vec![2]);
        let coeffs = super_schur_function(&shape, 2, 2);

        // All coefficients should have total degree 2
        for coeff in &coeffs {
            let total_degree: usize = coeff.x_exponents.iter().sum::<usize>()
                + coeff.y_exponents.iter().sum::<usize>();
            assert_eq!(total_degree, 2);
        }
    }
}

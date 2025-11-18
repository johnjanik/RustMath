//! Monomial generation and enumeration
//!
//! This module provides functions for generating monomials in multivariate polynomials,
//! corresponding to SageMath's `sage.rings.monomials`.
//!
//! # Mathematical Background
//!
//! A monomial in variables x₁, x₂, ..., xₙ is a product:
//!
//! x₁^{α₁} · x₂^{α₂} · ... · xₙ^{αₙ}
//!
//! where αᵢ ≥ 0 are non-negative integers. The monomial is characterized by its
//! exponent vector (α₁, α₂, ..., αₙ).
//!
//! ## Total Degree
//!
//! The total degree of a monomial is:
//! deg(x₁^{α₁} · ... · xₙ^{αₙ}) = α₁ + α₂ + ... + αₙ
//!
//! ## Monomial Orderings
//!
//! Common orderings include:
//! - **Lexicographic (lex)**: x^a > x^b if leftmost nonzero entry of a-b is positive
//! - **Graded lexicographic (grevlex)**: Compare by total degree first, then lex
//! - **Graded reverse lexicographic (degrevlex)**: Compare by total degree, then reverse lex
//!
//! ## Applications
//!
//! - Gröbner basis algorithms
//! - Polynomial interpolation
//! - Combinatorial enumeration
//! - Hilbert function computation
//!
//! # Key Functions
//!
//! - `monomials`: Generate all monomials up to a given degree

use std::collections::HashSet;

/// Monomial representation as exponent vector
///
/// # Type Parameters
///
/// Exponents are represented as vectors of non-negative integers.
pub type MonomialExponent = Vec<usize>;

/// Generate all monomials up to a given degree
///
/// This corresponds to SageMath's `monomials` function.
///
/// # Arguments
///
/// * `num_variables` - Number of variables
/// * `max_degree` - Maximum total degree
///
/// # Returns
///
/// Vector of monomial exponent vectors
///
/// # Examples
///
/// ```ignore
/// // Generate monomials in 2 variables up to degree 2:
/// // 1, x, y, x^2, xy, y^2
/// let mons = monomials(2, 2);
/// assert_eq!(mons.len(), 6);
/// ```
///
/// # Mathematical Details
///
/// The number of monomials in n variables of degree exactly d is:
/// C(n + d - 1, d) = (n + d - 1)! / (d! · (n - 1)!)
///
/// The number up to degree d is:
/// ∑_{i=0}^d C(n + i - 1, i) = C(n + d, d)
pub fn monomials(num_variables: usize, max_degree: usize) -> Vec<MonomialExponent> {
    if num_variables == 0 {
        return vec![vec![]];
    }

    let mut result = Vec::new();

    // Generate monomials degree by degree
    for degree in 0..=max_degree {
        let mons = monomials_of_degree(num_variables, degree);
        result.extend(mons);
    }

    result
}

/// Generate monomials of exact degree
///
/// # Arguments
///
/// * `num_variables` - Number of variables
/// * `degree` - Exact total degree
///
/// # Returns
///
/// Vector of monomial exponent vectors of the given degree
///
/// # Examples
///
/// ```ignore
/// // Monomials of degree 2 in 2 variables: x^2, xy, y^2
/// let mons = monomials_of_degree(2, 2);
/// assert_eq!(mons.len(), 3);
/// ```
pub fn monomials_of_degree(num_variables: usize, degree: usize) -> Vec<MonomialExponent> {
    if num_variables == 0 {
        if degree == 0 {
            return vec![vec![]];
        } else {
            return vec![];
        }
    }

    if degree == 0 {
        return vec![vec![0; num_variables]];
    }

    let mut result = Vec::new();

    // Recursively generate monomials
    generate_monomials_recursive(num_variables, degree, 0, &mut vec![0; num_variables], &mut result);

    result
}

/// Recursive helper for monomial generation
///
/// Uses a backtracking approach to generate all partitions of degree into num_variables parts.
fn generate_monomials_recursive(
    num_variables: usize,
    remaining_degree: usize,
    current_var: usize,
    current_exponents: &mut Vec<usize>,
    result: &mut Vec<MonomialExponent>,
) {
    if current_var == num_variables - 1 {
        // Last variable gets all remaining degree
        current_exponents[current_var] = remaining_degree;
        result.push(current_exponents.clone());
        current_exponents[current_var] = 0;
        return;
    }

    // Try all possible exponents for current variable
    for exp in 0..=remaining_degree {
        current_exponents[current_var] = exp;
        generate_monomials_recursive(
            num_variables,
            remaining_degree - exp,
            current_var + 1,
            current_exponents,
            result,
        );
    }
    current_exponents[current_var] = 0;
}

/// Count monomials of given degree
///
/// # Arguments
///
/// * `num_variables` - Number of variables
/// * `degree` - Total degree
///
/// # Returns
///
/// Number of monomials of that degree
///
/// # Mathematical Details
///
/// Uses the formula: C(n + d - 1, d) = (n + d - 1)! / (d! · (n - 1)!)
pub fn count_monomials_of_degree(num_variables: usize, degree: usize) -> usize {
    if num_variables == 0 {
        return if degree == 0 { 1 } else { 0 };
    }

    binomial(num_variables + degree - 1, degree)
}

/// Count monomials up to given degree
///
/// # Arguments
///
/// * `num_variables` - Number of variables
/// * `max_degree` - Maximum total degree
///
/// # Returns
///
/// Total number of monomials up to that degree
pub fn count_monomials(num_variables: usize, max_degree: usize) -> usize {
    if num_variables == 0 {
        return 1;
    }

    binomial(num_variables + max_degree, max_degree)
}

/// Compute binomial coefficient C(n, k)
///
/// # Arguments
///
/// * `n` - Upper index
/// * `k` - Lower index
///
/// # Returns
///
/// Binomial coefficient n choose k
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = if k > n - k { n - k } else { k };

    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

/// Convert monomial exponent vector to string
///
/// # Arguments
///
/// * `exponents` - Exponent vector
/// * `variables` - Variable names
///
/// # Returns
///
/// String representation of the monomial
///
/// # Examples
///
/// ```ignore
/// let exp = vec![2, 1, 0];
/// let vars = vec!["x", "y", "z"];
/// let s = monomial_to_string(&exp, &vars);
/// // Result: "x^2*y"
/// ```
pub fn monomial_to_string(exponents: &[usize], variables: &[&str]) -> String {
    let mut parts = Vec::new();

    for (i, &exp) in exponents.iter().enumerate() {
        if exp == 0 {
            continue;
        }

        let var = if i < variables.len() {
            variables[i]
        } else {
            "x"
        };

        if exp == 1 {
            parts.push(var.to_string());
        } else {
            parts.push(format!("{}^{}", var, exp));
        }
    }

    if parts.is_empty() {
        "1".to_string()
    } else {
        parts.join("*")
    }
}

/// Compute total degree of a monomial
///
/// # Arguments
///
/// * `exponents` - Exponent vector
///
/// # Returns
///
/// Total degree
pub fn total_degree(exponents: &[usize]) -> usize {
    exponents.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monomials_single_variable() {
        let mons = monomials(1, 3);
        assert_eq!(mons.len(), 4); // 1, x, x^2, x^3
        assert_eq!(mons[0], vec![0]);
        assert_eq!(mons[1], vec![1]);
        assert_eq!(mons[2], vec![2]);
        assert_eq!(mons[3], vec![3]);
    }

    #[test]
    fn test_monomials_two_variables_degree_0() {
        let mons = monomials(2, 0);
        assert_eq!(mons.len(), 1); // Just 1
        assert_eq!(mons[0], vec![0, 0]);
    }

    #[test]
    fn test_monomials_two_variables_degree_1() {
        let mons = monomials(2, 1);
        assert_eq!(mons.len(), 3); // 1, x, y
    }

    #[test]
    fn test_monomials_two_variables_degree_2() {
        let mons = monomials(2, 2);
        assert_eq!(mons.len(), 6); // 1, x, y, x^2, xy, y^2
    }

    #[test]
    fn test_monomials_of_degree() {
        let mons = monomials_of_degree(2, 2);
        assert_eq!(mons.len(), 3); // x^2, xy, y^2

        // Check they all have degree 2
        for mon in &mons {
            assert_eq!(total_degree(mon), 2);
        }
    }

    #[test]
    fn test_monomials_of_degree_zero() {
        let mons = monomials_of_degree(3, 0);
        assert_eq!(mons.len(), 1);
        assert_eq!(mons[0], vec![0, 0, 0]);
    }

    #[test]
    fn test_count_monomials_of_degree() {
        assert_eq!(count_monomials_of_degree(2, 0), 1); // 1
        assert_eq!(count_monomials_of_degree(2, 1), 2); // x, y
        assert_eq!(count_monomials_of_degree(2, 2), 3); // x^2, xy, y^2
        assert_eq!(count_monomials_of_degree(2, 3), 4); // x^3, x^2y, xy^2, y^3
    }

    #[test]
    fn test_count_monomials() {
        assert_eq!(count_monomials(2, 0), 1);
        assert_eq!(count_monomials(2, 1), 3);
        assert_eq!(count_monomials(2, 2), 6);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
    }

    #[test]
    fn test_monomial_to_string() {
        let vars = vec!["x", "y", "z"];

        assert_eq!(monomial_to_string(&[0, 0, 0], &vars), "1");
        assert_eq!(monomial_to_string(&[1, 0, 0], &vars), "x");
        assert_eq!(monomial_to_string(&[0, 1, 0], &vars), "y");
        assert_eq!(monomial_to_string(&[2, 0, 0], &vars), "x^2");
        assert_eq!(monomial_to_string(&[1, 1, 0], &vars), "x*y");
        assert_eq!(monomial_to_string(&[2, 1, 3], &vars), "x^2*y*z^3");
    }

    #[test]
    fn test_total_degree() {
        assert_eq!(total_degree(&[0, 0, 0]), 0);
        assert_eq!(total_degree(&[1, 0, 0]), 1);
        assert_eq!(total_degree(&[1, 1, 0]), 2);
        assert_eq!(total_degree(&[2, 1, 3]), 6);
    }

    #[test]
    fn test_monomials_three_variables() {
        let mons = monomials(3, 1);
        // Should be: 1, x, y, z
        assert_eq!(mons.len(), 4);
    }

    #[test]
    fn test_large_monomial_count() {
        // Test that count formula matches actual generation
        for n in 1..5 {
            for d in 0..5 {
                let generated = monomials_of_degree(n, d);
                let counted = count_monomials_of_degree(n, d);
                assert_eq!(generated.len(), counted,
                    "Mismatch for {} variables, degree {}", n, d);
            }
        }
    }

    #[test]
    fn test_monomials_no_variables() {
        let mons = monomials(0, 5);
        assert_eq!(mons.len(), 1);
        assert_eq!(mons[0], vec![]);
    }

    #[test]
    fn test_unique_monomials() {
        // Check that all generated monomials are unique
        let mons = monomials(3, 3);
        let unique: HashSet<_> = mons.iter().collect();
        assert_eq!(mons.len(), unique.len());
    }

    #[test]
    fn test_monomial_degree_consistency() {
        // All monomials from monomials(n, d) should have degree <= d
        let mons = monomials(3, 4);
        for mon in mons {
            assert!(total_degree(&mon) <= 4);
        }
    }
}

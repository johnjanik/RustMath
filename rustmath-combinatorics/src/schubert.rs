//! Schubert Polynomials
//!
//! This module provides functionality for computing Schubert polynomials,
//! which are fundamental objects in the cohomology of flag varieties.
//!
//! Schubert polynomials are indexed by permutations and computed using
//! divided difference operators. They satisfy important properties like
//! Monk's rule for multiplication.

use crate::permutations::Permutation;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::multivariate::{Monomial, MultivariatePolynomial};
use std::collections::BTreeMap;

/// A divided difference operator acting on multivariate polynomials
///
/// The divided difference operator ∂_i acts on polynomials by:
/// ∂_i(f) = (f - s_i(f)) / (x_i - x_{i+1})
///
/// where s_i swaps variables x_i and x_{i+1}.
pub struct DividedDifference {
    /// The index i of the operator ∂_i (0-indexed)
    index: usize,
}

impl DividedDifference {
    /// Create a new divided difference operator ∂_i
    ///
    /// # Arguments
    /// * `index` - The index i (0-indexed, so ∂_0 swaps x_0 and x_1)
    pub fn new(index: usize) -> Self {
        DividedDifference { index }
    }

    /// Apply the divided difference operator to a polynomial
    ///
    /// Computes ∂_i(f) = (f - s_i(f)) / (x_i - x_{i+1})
    pub fn apply<R: Ring>(&self, poly: &MultivariatePolynomial<R>) -> MultivariatePolynomial<R> {
        let i = self.index;

        // Compute s_i(f) by swapping variables x_i and x_{i+1}
        let swapped = swap_variables(poly, i, i + 1);

        // Compute f - s_i(f)
        let diff = poly.clone() - swapped;

        // Divide by (x_i - x_{i+1})
        divide_by_difference(&diff, i, i + 1)
    }
}

/// Swap two variables in a polynomial
fn swap_variables<R: Ring>(
    poly: &MultivariatePolynomial<R>,
    i: usize,
    j: usize,
) -> MultivariatePolynomial<R> {
    let mut result = MultivariatePolynomial::zero();

    // Iterate over all terms
    for (monomial, coeff) in poly.iter_terms() {
        // Create a new monomial with swapped variables
        let mut new_exponents = BTreeMap::new();

        for (var, exp) in monomial.iter_exponents() {
            let new_var = if *var == i {
                j
            } else if *var == j {
                i
            } else {
                *var
            };
            new_exponents.insert(new_var, *exp);
        }

        let new_monomial = Monomial::from_exponents(new_exponents);
        result.add_term(new_monomial, coeff.clone());
    }

    result
}

/// Divide a polynomial by (x_i - x_j)
///
/// For a polynomial that is divisible by (x_i - x_j), compute the quotient.
/// Uses the formula: for base monomial m, the term m*x_i^a*x_j^b / (x_i - x_j)
/// contributes m * sum_{k=min(a,b)}^{max(a,b)-1} x_i^k * x_j^{a+b-1-k} to the quotient
fn divide_by_difference<R: Ring>(
    poly: &MultivariatePolynomial<R>,
    i: usize,
    j: usize,
) -> MultivariatePolynomial<R> {
    let mut result = MultivariatePolynomial::zero();
    let mut processed = std::collections::HashSet::new();

    // Process pairs of terms (to avoid double-counting)
    for (monomial, coeff) in poly.iter_terms() {
        let exp_i = monomial.exponent(i);
        let exp_j = monomial.exponent(j);

        // Create a canonical key to identify processed term pairs
        let mut key_exps = BTreeMap::new();
        for (var, exp) in monomial.iter_exponents() {
            key_exps.insert(*var, *exp);
        }
        let key = (key_exps.clone(), exp_i.min(exp_j), exp_i.max(exp_j));

        if processed.contains(&key) {
            continue;
        }
        processed.insert(key.clone());

        // Get base exponents
        let mut base_exponents = BTreeMap::new();
        for (var, exp) in monomial.iter_exponents() {
            if *var != i && *var != j {
                base_exponents.insert(*var, *exp);
            }
        }

        if exp_i == exp_j {
            // Terms with equal exponents cancel in f - s_i(f)
            continue;
        }

        // For the pair (x_i^a * x_j^b, x_i^b * x_j^a) in f - s_i(f):
        // The quotient is: base * sum_{k=b}^{a-1} x_i^k * x_j^{a+b-1-k}  [when a > b]
        // Process only if exp_i > exp_j (to avoid duplication)
        if exp_i > exp_j {
            for k in exp_j..exp_i {
                let mut quot_exp = base_exponents.clone();
                if k > 0 {
                    quot_exp.insert(i, k);
                }
                let j_exp = exp_i + exp_j - 1 - k;
                if j_exp > 0 {
                    quot_exp.insert(j, j_exp);
                }
                let quot_monomial = Monomial::from_exponents(quot_exp);
                result.add_term(quot_monomial, coeff.clone());
            }
        }
    }

    result
}

/// Compute the Schubert polynomial for a given permutation
///
/// Schubert polynomials S_w are indexed by permutations w in S_n.
/// They are computed using divided difference operators:
///
/// 1. Start with the longest permutation w_0 which has S_{w_0} = product x_i^{n-i}
/// 2. Apply divided differences to reduce to the target permutation
///
/// # Arguments
/// * `perm` - The permutation (0-indexed)
///
/// # Returns
/// The Schubert polynomial S_w
pub fn schubert_polynomial(perm: &Permutation) -> MultivariatePolynomial<Integer> {
    let n = perm.size();

    if n == 0 {
        return MultivariatePolynomial::constant(Integer::one());
    }

    // For identity permutation, S_id = 1
    if perm.inversions() == 0 {
        return MultivariatePolynomial::constant(Integer::one());
    }

    // Create the longest permutation w_0 = (n-1, n-2, ..., 1, 0) in 0-indexed
    let longest = Permutation::from_vec((0..n).rev().collect()).unwrap();

    // Start with the Schubert polynomial for w_0
    // S_{w_0} = x_0^{n-1} * x_1^{n-2} * ... * x_{n-2}^1
    let mut poly = MultivariatePolynomial::constant(Integer::one());
    for i in 0..(n - 1) {
        let power = (n - 1 - i) as u32;
        let var_poly = MultivariatePolynomial::variable(i);
        for _ in 0..power {
            poly = poly.clone() * var_poly.clone();
        }
    }

    // If perm is already the longest permutation, return
    if perm == &longest {
        return poly;
    }

    // Apply divided differences to reduce to the target permutation
    let mut current_perm = (0..n).rev().collect::<Vec<_>>();
    let target = perm.as_slice();

    // Bubble-sort style reduction
    while &current_perm[..] != target {
        let mut found = false;

        for i in 0..(n - 1) {
            if current_perm[i] > current_perm[i + 1] {
                let val_i = current_perm[i];
                let val_j = current_perm[i + 1];

                let pos_i_in_target = target.iter().position(|&x| x == val_i).unwrap();
                let pos_j_in_target = target.iter().position(|&x| x == val_j).unwrap();

                // If they're in the right order in target, remove this inversion
                if pos_i_in_target < pos_j_in_target {
                    let dd = DividedDifference::new(i);
                    poly = dd.apply(&poly);
                    current_perm.swap(i, i + 1);
                    found = true;
                    break;
                }
            }
        }

        if !found {
            break;
        }
    }

    poly
}

/// Monk's rule for multiplying Schubert polynomials
///
/// Computes S_w * x_r by direct polynomial multiplication.
///
/// # Arguments
/// * `perm` - The permutation w
/// * `variable` - The variable index r (0-indexed)
///
/// # Returns
/// The product S_w * x_r as a multivariate polynomial
pub fn monk_rule(perm: &Permutation, variable: usize) -> MultivariatePolynomial<Integer> {
    let s_w = schubert_polynomial(perm);
    let x_r = MultivariatePolynomial::variable(variable);
    s_w * x_r
}

/// Monk's rule expansion returning the permutations in the expansion
///
/// Returns a list of permutations u such that S_w * x_r appears in the expansion.
///
/// # Arguments
/// * `perm` - The permutation w
/// * `variable` - The variable index r (0-indexed)
///
/// # Returns
/// A vector of permutations u that appear in the Monk's rule expansion
pub fn monk_rule_expansion(perm: &Permutation, variable: usize) -> Vec<Permutation> {
    let n = perm.size();
    let r = variable;

    if r >= n {
        return vec![];
    }

    let mut result = Vec::new();
    let w = perm.as_slice();

    // Find the position where r appears in the inverse permutation
    let inv_perm = perm.inverse();
    let pos_r = inv_perm.apply(r).unwrap();

    // Try all adjacent transpositions
    for i in 0..(n - 1) {
        // Check if this is an ascent (can increase length)
        if w[i] < w[i + 1] {
            // Check Monk's rule conditions
            if i >= pos_r || (w[i] <= r && r < w[i + 1]) {
                let mut new_w = w.to_vec();
                new_w.swap(i, i + 1);

                if let Some(new_perm) = Permutation::from_vec(new_w) {
                    // Verify that this increases the inversion count
                    if new_perm.inversions() == perm.inversions() + 1 {
                        result.push(new_perm);
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divided_difference_simple() {
        // Test ∂_0 on x_0
        let x0 = MultivariatePolynomial::variable(0);
        let dd = DividedDifference::new(0);
        let result = dd.apply(&x0);

        // ∂_0(x_0) = (x_0 - x_1) / (x_0 - x_1) = 1
        assert_eq!(result, MultivariatePolynomial::constant(Integer::one()));
    }

    #[test]
    fn test_divided_difference_quadratic() {
        // Test ∂_0 on x_0^2
        let x0: MultivariatePolynomial<Integer> = MultivariatePolynomial::variable(0);
        let x0_squared = x0.clone() * x0;

        let dd = DividedDifference::new(0);
        let result = dd.apply(&x0_squared);

        // ∂_0(x_0^2) = (x_0^2 - x_1^2) / (x_0 - x_1) = x_0 + x_1
        let expected: MultivariatePolynomial<Integer> = MultivariatePolynomial::variable(0) + MultivariatePolynomial::variable(1);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_schubert_identity() {
        // S_id = 1
        let id = Permutation::identity(3);
        let s_id = schubert_polynomial(&id);
        assert_eq!(s_id, MultivariatePolynomial::constant(Integer::one()));
    }

    #[test]
    fn test_schubert_longest() {
        // S_(2,1,0) for S_3 should be x_0^2 * x_1
        let perm = Permutation::from_vec(vec![2, 1, 0]).unwrap();
        let s_w = schubert_polynomial(&perm);

        // Should be x_0^2 * x_1
        let x0 = MultivariatePolynomial::variable(0);
        let x1 = MultivariatePolynomial::variable(1);
        let expected = x0.clone() * x0.clone() * x1;

        assert_eq!(s_w, expected);
    }

    #[test]
    fn test_schubert_simple_transposition() {
        // S_(1,0) = x_0
        let perm = Permutation::from_vec(vec![1, 0]).unwrap();
        let s_w = schubert_polynomial(&perm);

        let expected = MultivariatePolynomial::variable(0);
        assert_eq!(s_w, expected);
    }

    #[test]
    fn test_monk_rule_identity() {
        // S_id * x_0 = S_(1,0) = x_0
        let id = Permutation::identity(2);
        let result = monk_rule(&id, 0);

        let expected = MultivariatePolynomial::variable(0);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_swap_variables() {
        // Swap x_0 and x_1 in x_0^2 * x_1
        let x0: MultivariatePolynomial<Integer> = MultivariatePolynomial::variable(0);
        let x1: MultivariatePolynomial<Integer> = MultivariatePolynomial::variable(1);
        let poly = x0.clone() * x0 * x1.clone();

        let swapped = swap_variables(&poly, 0, 1);

        // Should be x_1^2 * x_0
        let expected = x1.clone() * x1 * MultivariatePolynomial::variable(0);
        assert_eq!(swapped, expected);
    }
}

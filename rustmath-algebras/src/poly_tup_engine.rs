//! Polynomial Tuple Engine
//!
//! This module provides utility functions for working with polynomials in a tuple representation.
//! Polynomials are represented as tuples of (exponent, coefficient) pairs for efficient operations.
//!
//! Corresponds to sage.algebras.fusion_rings.poly_tup_engine

use rustmath_core::{Ring, Result};
use std::collections::HashMap;

/// Represents a polynomial as a tuple of (exponents, coefficient) pairs
///
/// The exponents are represented as a Vec<usize> for multivariate polynomials.
/// Each element in the outer Vec represents one monomial term.
pub type PolyTuple<R> = Vec<(Vec<usize>, R)>;

/// Convert a polynomial from monomial coefficient map to tuple representation
///
/// Takes a mapping from exponent vectors to coefficients and converts it
/// to a vector of (exponent, coefficient) pairs, filtering out zero coefficients.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.poly_to_tup
///
/// # Arguments
/// * `coeffs` - HashMap mapping exponent vectors to coefficients
///
/// # Returns
/// A PolyTuple representation with non-zero coefficients
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::poly_to_tup;
/// use rustmath_integers::Integer;
/// use std::collections::HashMap;
///
/// let mut coeffs = HashMap::new();
/// coeffs.insert(vec![2, 0], Integer::from(1));  // x^2
/// coeffs.insert(vec![0, 0], Integer::from(1));  // constant term
///
/// let tup = poly_to_tup(coeffs);
/// assert_eq!(tup.len(), 2);
/// ```
pub fn poly_to_tup<R: Ring>(coeffs: HashMap<Vec<usize>, R>) -> PolyTuple<R> {
    coeffs
        .into_iter()
        .filter(|(_, coeff)| !coeff.is_zero())
        .collect()
}

/// Get the constant coefficient from a polynomial in tuple form
///
/// Returns the coefficient of the zero-exponent term (the constant term).
/// If no constant term exists, returns zero.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.constant_coeff
///
/// # Arguments
/// * `poly_tup` - Polynomial in tuple representation
///
/// # Returns
/// The constant coefficient
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::constant_coeff;
/// use rustmath_integers::Integer;
///
/// let poly = vec![
///     (vec![2, 0], Integer::from(3)),  // 3x^2
///     (vec![0, 0], Integer::from(5)),  // 5
/// ];
/// assert_eq!(constant_coeff(&poly), Integer::from(5));
/// ```
pub fn constant_coeff<R: Ring>(poly_tup: &PolyTuple<R>) -> R {
    for (exp, coeff) in poly_tup {
        if exp.iter().all(|&e| e == 0) {
            return coeff.clone();
        }
    }
    R::zero()
}

/// Get the indices of variables that appear in the polynomial
///
/// Returns a vector of variable indices that have non-zero exponents
/// in at least one term of the polynomial.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.variables
///
/// # Arguments
/// * `poly_tup` - Polynomial in tuple representation
///
/// # Returns
/// Sorted vector of variable indices
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::variables;
/// use rustmath_integers::Integer;
///
/// let poly = vec![
///     (vec![2, 0, 1], Integer::from(1)),  // x_0^2 * x_2
///     (vec![0, 3, 0], Integer::from(2)),  // x_1^3
/// ];
/// assert_eq!(variables(&poly), vec![0, 1, 2]);
/// ```
pub fn variables<R: Ring>(poly_tup: &PolyTuple<R>) -> Vec<usize> {
    let mut var_set = std::collections::HashSet::new();

    for (exp, _) in poly_tup {
        for (idx, &e) in exp.iter().enumerate() {
            if e > 0 {
                var_set.insert(idx);
            }
        }
    }

    let mut vars: Vec<usize> = var_set.into_iter().collect();
    vars.sort_unstable();
    vars
}

/// Get the degrees of each variable in the polynomial
///
/// Returns a mapping from variable index to its maximum degree
/// across all terms in the polynomial.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.get_variables_degrees
///
/// # Arguments
/// * `poly_tup` - Polynomial in tuple representation
///
/// # Returns
/// HashMap mapping variable indices to their maximum degrees
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::get_variables_degrees;
/// use rustmath_integers::Integer;
///
/// let poly = vec![
///     (vec![2, 0], Integer::from(1)),  // x_0^2
///     (vec![1, 3], Integer::from(1)),  // x_0 * x_1^3
/// ];
/// let degrees = get_variables_degrees(&poly);
/// assert_eq!(degrees.get(&0), Some(&2));
/// assert_eq!(degrees.get(&1), Some(&3));
/// ```
pub fn get_variables_degrees<R: Ring>(poly_tup: &PolyTuple<R>) -> HashMap<usize, usize> {
    let mut degrees: HashMap<usize, usize> = HashMap::new();

    for (exp, _) in poly_tup {
        for (idx, &e) in exp.iter().enumerate() {
            if e > 0 {
                degrees.entry(idx)
                    .and_modify(|max_deg| *max_deg = (*max_deg).max(e))
                    .or_insert(e);
            }
        }
    }

    degrees
}

/// Resize exponent vectors to a new dimension
///
/// Adjusts all exponent vectors in the polynomial to have exactly `new_dim` elements,
/// either by truncating or padding with zeros.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.resize
///
/// # Arguments
/// * `poly_tup` - Polynomial in tuple representation
/// * `new_dim` - Desired dimension for exponent vectors
///
/// # Returns
/// New polynomial with resized exponent vectors
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::resize;
/// use rustmath_integers::Integer;
///
/// let poly = vec![
///     (vec![2, 1], Integer::from(1)),
/// ];
/// let resized = resize(&poly, 4);
/// assert_eq!(resized[0].0, vec![2, 1, 0, 0]);
/// ```
pub fn resize<R: Ring>(poly_tup: &PolyTuple<R>, new_dim: usize) -> PolyTuple<R> {
    poly_tup
        .iter()
        .map(|(exp, coeff)| {
            let mut new_exp = exp.clone();
            new_exp.resize(new_dim, 0);
            (new_exp, coeff.clone())
        })
        .collect()
}

/// Apply a coefficient mapping function to all coefficients
///
/// Transforms each coefficient in the polynomial using the provided function.
/// This is useful for operations like changing the base ring or normalizing coefficients.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.apply_coeff_map
///
/// # Arguments
/// * `poly_tup` - Polynomial in tuple representation
/// * `f` - Function to apply to each coefficient
///
/// # Returns
/// New polynomial with transformed coefficients
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::apply_coeff_map;
/// use rustmath_integers::Integer;
///
/// let poly = vec![
///     (vec![1], Integer::from(2)),
///     (vec![2], Integer::from(3)),
/// ];
/// let doubled = apply_coeff_map(&poly, |c| c.clone() + c.clone());
/// assert_eq!(doubled[0].1, Integer::from(4));
/// assert_eq!(doubled[1].1, Integer::from(6));
/// ```
pub fn apply_coeff_map<R: Ring, F>(poly_tup: &PolyTuple<R>, f: F) -> PolyTuple<R>
where
    F: Fn(&R) -> R,
{
    poly_tup
        .iter()
        .map(|(exp, coeff)| (exp.clone(), f(coeff)))
        .filter(|(_, coeff)| !coeff.is_zero())
        .collect()
}

/// Create a sortkey function for polynomial tuples
///
/// Returns a closure that can be used to compare and sort polynomial terms.
/// Terms are ordered lexicographically by their exponent vectors.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.poly_tup_sortkey
///
/// # Returns
/// A comparison function for sorting polynomial terms
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::poly_tup_sortkey;
/// use rustmath_integers::Integer;
///
/// let mut poly = vec![
///     (vec![0, 2], Integer::from(1)),
///     (vec![2, 0], Integer::from(1)),
///     (vec![1, 1], Integer::from(1)),
/// ];
///
/// let sortkey = poly_tup_sortkey();
/// poly.sort_by(|a, b| sortkey(&a.0).cmp(&sortkey(&b.0)));
///
/// assert_eq!(poly[0].0, vec![0, 2]);
/// assert_eq!(poly[1].0, vec![1, 1]);
/// assert_eq!(poly[2].0, vec![2, 0]);
/// ```
pub fn poly_tup_sortkey() -> impl Fn(&Vec<usize>) -> Vec<usize> {
    |exp: &Vec<usize>| exp.clone()
}

/// Compute known powers of variables in a polynomial
///
/// Given a set of variable indices, computes all powers of those variables
/// that appear in the polynomial, organized by variable.
///
/// Corresponds to sage.algebras.fusion_rings.poly_tup_engine.compute_known_powers
///
/// # Arguments
/// * `poly_tup` - Polynomial in tuple representation
/// * `variables` - Indices of variables to track
///
/// # Returns
/// HashMap mapping each variable to a sorted vector of its appearing powers
///
/// # Examples
/// ```
/// use rustmath_algebras::poly_tup_engine::compute_known_powers;
/// use rustmath_integers::Integer;
///
/// let poly = vec![
///     (vec![1, 0], Integer::from(1)),  // x_0
///     (vec![2, 0], Integer::from(1)),  // x_0^2
///     (vec![0, 3], Integer::from(1)),  // x_1^3
/// ];
///
/// let powers = compute_known_powers(&poly, &[0, 1]);
/// assert_eq!(powers.get(&0), Some(&vec![1, 2]));
/// assert_eq!(powers.get(&1), Some(&vec![3]));
/// ```
pub fn compute_known_powers<R: Ring>(
    poly_tup: &PolyTuple<R>,
    variables: &[usize],
) -> HashMap<usize, Vec<usize>> {
    let mut powers: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();

    for var in variables {
        powers.insert(*var, std::collections::HashSet::new());
    }

    for (exp, _) in poly_tup {
        for &var in variables {
            if var < exp.len() && exp[var] > 0 {
                powers.get_mut(&var).unwrap().insert(exp[var]);
            }
        }
    }

    powers
        .into_iter()
        .map(|(var, power_set)| {
            let mut power_vec: Vec<usize> = power_set.into_iter().collect();
            power_vec.sort_unstable();
            (var, power_vec)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_poly_to_tup() {
        let mut coeffs = HashMap::new();
        coeffs.insert(vec![2, 0], Integer::from(1));
        coeffs.insert(vec![0, 0], Integer::from(3));
        coeffs.insert(vec![1, 1], Integer::zero()); // Should be filtered out

        let tup = poly_to_tup(coeffs);
        assert_eq!(tup.len(), 2); // Zero coefficient filtered out
    }

    #[test]
    fn test_constant_coeff() {
        let poly = vec![
            (vec![2, 0], Integer::from(3)),
            (vec![0, 0], Integer::from(7)),
            (vec![1, 1], Integer::from(2)),
        ];

        assert_eq!(constant_coeff(&poly), Integer::from(7));
    }

    #[test]
    fn test_constant_coeff_missing() {
        let poly = vec![
            (vec![2, 0], Integer::from(3)),
            (vec![1, 1], Integer::from(2)),
        ];

        assert_eq!(constant_coeff(&poly), Integer::zero());
    }

    #[test]
    fn test_variables() {
        let poly = vec![
            (vec![2, 0, 1], Integer::from(1)),
            (vec![0, 3, 0], Integer::from(2)),
        ];

        assert_eq!(variables(&poly), vec![0, 1, 2]);
    }

    #[test]
    fn test_get_variables_degrees() {
        let poly = vec![
            (vec![2, 0, 1], Integer::from(1)),
            (vec![3, 2, 0], Integer::from(1)),
            (vec![1, 1, 2], Integer::from(1)),
        ];

        let degrees = get_variables_degrees(&poly);
        assert_eq!(degrees.get(&0), Some(&3));
        assert_eq!(degrees.get(&1), Some(&2));
        assert_eq!(degrees.get(&2), Some(&2));
    }

    #[test]
    fn test_resize() {
        let poly = vec![
            (vec![2, 1], Integer::from(5)),
        ];

        // Expand
        let expanded = resize(&poly, 4);
        assert_eq!(expanded[0].0, vec![2, 1, 0, 0]);
        assert_eq!(expanded[0].1, Integer::from(5));

        // Truncate
        let truncated = resize(&poly, 1);
        assert_eq!(truncated[0].0, vec![2]);
    }

    #[test]
    fn test_apply_coeff_map() {
        let poly = vec![
            (vec![1], Integer::from(2)),
            (vec![2], Integer::from(3)),
        ];

        let doubled = apply_coeff_map(&poly, |c| c.clone() + c.clone());
        assert_eq!(doubled[0].1, Integer::from(4));
        assert_eq!(doubled[1].1, Integer::from(6));
    }

    #[test]
    fn test_apply_coeff_map_filters_zeros() {
        let poly = vec![
            (vec![1], Integer::from(2)),
            (vec![2], Integer::from(3)),
        ];

        // Map all to zero - should result in empty polynomial
        let zeroed = apply_coeff_map(&poly, |_| Integer::zero());
        assert_eq!(zeroed.len(), 0);
    }

    #[test]
    fn test_poly_tup_sortkey() {
        let mut poly = vec![
            (vec![2, 0], Integer::from(1)),
            (vec![0, 2], Integer::from(1)),
            (vec![1, 1], Integer::from(1)),
        ];

        let sortkey = poly_tup_sortkey();
        poly.sort_by(|a, b| sortkey(&a.0).cmp(&sortkey(&b.0)));

        assert_eq!(poly[0].0, vec![0, 2]);
        assert_eq!(poly[1].0, vec![1, 1]);
        assert_eq!(poly[2].0, vec![2, 0]);
    }

    #[test]
    fn test_compute_known_powers() {
        let poly = vec![
            (vec![1, 0, 0], Integer::from(1)),
            (vec![2, 0, 0], Integer::from(1)),
            (vec![3, 0, 0], Integer::from(1)),
            (vec![0, 2, 0], Integer::from(1)),
            (vec![0, 4, 0], Integer::from(1)),
        ];

        let powers = compute_known_powers(&poly, &[0, 1, 2]);
        assert_eq!(powers.get(&0), Some(&vec![1, 2, 3]));
        assert_eq!(powers.get(&1), Some(&vec![2, 4]));
        assert_eq!(powers.get(&2), Some(&vec![]));
    }
}

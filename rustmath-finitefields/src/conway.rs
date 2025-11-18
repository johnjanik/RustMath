//! Conway polynomials for finite field construction
//!
//! Conway polynomials are a standard choice of irreducible polynomials
//! for constructing finite field extensions GF(p^n). They ensure
//! compatibility between different extensions of the same prime field.

use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::collections::HashMap;

/// Get the Conway polynomial for GF(p^n)
///
/// Returns the standard irreducible polynomial of degree n over GF(p).
/// This is a lookup table of pre-computed Conway polynomials for small p and n.
///
/// # Arguments
///
/// * `p` - Prime characteristic
/// * `n` - Degree of the extension
///
/// # Returns
///
/// The Conway polynomial as a `UnivariatePolynomial<Integer>` with coefficients in [0, p).
/// Returns `None` if the Conway polynomial is not in the lookup table.
///
/// # Format
///
/// Polynomials are given as coefficient vectors [a_0, a_1, ..., a_n]
/// representing a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
pub fn conway_polynomial(p: u32, n: usize) -> Option<UnivariatePolynomial<Integer>> {
    let table = get_conway_table();

    table.get(&(p, n)).map(|coeffs| {
        let int_coeffs: Vec<Integer> = coeffs.iter().map(|&c| Integer::from(c as i64)).collect();
        UnivariatePolynomial::new(int_coeffs)
    })
}

/// Check if a Conway polynomial is available for GF(p^n)
pub fn has_conway_polynomial(p: u32, n: usize) -> bool {
    let table = get_conway_table();
    table.contains_key(&(p, n))
}

/// Get the lookup table of Conway polynomials
///
/// Returns a HashMap mapping (p, n) to coefficient vectors.
///
/// # Format
///
/// Each entry (p, n) -> [a_0, a_1, ..., a_n] represents the polynomial:
/// a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n over GF(p)
fn get_conway_table() -> HashMap<(u32, usize), Vec<u32>> {
    let mut table = HashMap::new();

    // Conway polynomials for GF(2^n)
    // Source: Frank Lübeck's Conway polynomial database
    table.insert((2, 1), vec![1, 1]); // x + 1
    table.insert((2, 2), vec![1, 1, 1]); // x^2 + x + 1
    table.insert((2, 3), vec![1, 1, 0, 1]); // x^3 + x + 1
    table.insert((2, 4), vec![1, 1, 0, 0, 1]); // x^4 + x + 1
    table.insert((2, 5), vec![1, 0, 1, 0, 0, 1]); // x^5 + x^2 + 1
    table.insert((2, 6), vec![1, 1, 0, 0, 0, 0, 1]); // x^6 + x + 1
    table.insert((2, 7), vec![1, 1, 0, 0, 0, 0, 0, 1]); // x^7 + x + 1
    table.insert((2, 8), vec![1, 0, 1, 1, 1, 0, 0, 0, 1]); // x^8 + x^4 + x^3 + x^2 + 1

    // Conway polynomials for GF(3^n)
    table.insert((3, 1), vec![2, 1]); // x + 2
    table.insert((3, 2), vec![2, 1, 1]); // x^2 + x + 2
    table.insert((3, 3), vec![1, 2, 0, 1]); // x^3 + 2*x + 1
    table.insert((3, 4), vec![2, 1, 0, 0, 1]); // x^4 + x + 2
    table.insert((3, 5), vec![1, 2, 0, 0, 0, 1]); // x^5 + 2*x + 1

    // Conway polynomials for GF(5^n)
    table.insert((5, 1), vec![3, 1]); // x + 3
    table.insert((5, 2), vec![3, 1, 1]); // x^2 + x + 3
    table.insert((5, 3), vec![3, 3, 0, 1]); // x^3 + 3*x + 3
    table.insert((5, 4), vec![2, 4, 0, 0, 1]); // x^4 + 4*x + 2

    // Conway polynomials for GF(7^n)
    table.insert((7, 1), vec![4, 1]); // x + 4
    table.insert((7, 2), vec![6, 1, 1]); // x^2 + x + 6
    table.insert((7, 3), vec![4, 2, 0, 1]); // x^3 + 2*x + 4

    // Conway polynomials for GF(11^n)
    table.insert((11, 1), vec![9, 1]); // x + 9
    table.insert((11, 2), vec![2, 1, 1]); // x^2 + x + 2

    // Conway polynomials for GF(13^n)
    table.insert((13, 1), vec![11, 1]); // x + 11
    table.insert((13, 2), vec![2, 1, 1]); // x^2 + x + 2

    // Conway polynomials for GF(17^n)
    table.insert((17, 1), vec![14, 1]); // x + 14
    table.insert((17, 2), vec![3, 1, 1]); // x^2 + x + 3

    // Conway polynomials for GF(19^n)
    table.insert((19, 1), vec![17, 1]); // x + 17
    table.insert((19, 2), vec![2, 1, 1]); // x^2 + x + 2

    // Conway polynomials for GF(23^n)
    table.insert((23, 1), vec![18, 1]); // x + 18
    table.insert((23, 2), vec![5, 1, 1]); // x^2 + x + 5

    // Conway polynomials for GF(29^n)
    table.insert((29, 1), vec![27, 1]); // x + 27
    table.insert((29, 2), vec![2, 1, 1]); // x^2 + x + 2

    // Conway polynomials for GF(31^n)
    table.insert((31, 1), vec![28, 1]); // x + 28
    table.insert((31, 2), vec![3, 1, 1]); // x^2 + x + 3

    table
}

/// Get all available Conway polynomials
///
/// Returns a vector of (p, n) pairs for which Conway polynomials are available
pub fn available_conway_polynomials() -> Vec<(u32, usize)> {
    let table = get_conway_table();
    let mut keys: Vec<(u32, usize)> = table.keys().copied().collect();
    keys.sort();
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conway_gf2() {
        // GF(2^2): x^2 + x + 1
        let poly = conway_polynomial(2, 2).unwrap();
        assert_eq!(poly.degree(), Some(2));
        assert_eq!(*poly.coeff(0), Integer::from(1));
        assert_eq!(*poly.coeff(1), Integer::from(1));
        assert_eq!(*poly.coeff(2), Integer::from(1));
    }

    #[test]
    fn test_conway_gf3() {
        // GF(3^2): x^2 + x + 2
        let poly = conway_polynomial(3, 2).unwrap();
        assert_eq!(poly.degree(), Some(2));
        assert_eq!(*poly.coeff(0), Integer::from(2));
        assert_eq!(*poly.coeff(1), Integer::from(1));
        assert_eq!(*poly.coeff(2), Integer::from(1));
    }

    #[test]
    fn test_has_conway() {
        assert!(has_conway_polynomial(2, 4));
        assert!(has_conway_polynomial(3, 3));
        assert!(!has_conway_polynomial(2, 100)); // Not in table
        assert!(!has_conway_polynomial(1000, 1)); // Prime not in table
    }

    #[test]
    fn test_available_polynomials() {
        let available = available_conway_polynomials();

        // Should have at least the ones we defined
        assert!(available.contains(&(2, 1)));
        assert!(available.contains(&(2, 8)));
        assert!(available.contains(&(3, 5)));
        assert!(available.contains(&(31, 2)));

        // Should be sorted
        for i in 1..available.len() {
            let (p1, n1) = available[i - 1];
            let (p2, n2) = available[i];
            assert!(p1 < p2 || (p1 == p2 && n1 < n2));
        }
    }

    #[test]
    fn test_conway_polynomial_format() {
        // Verify that all polynomials are monic (leading coefficient is 1)
        let table = get_conway_table();
        for ((p, n), coeffs) in table.iter() {
            assert_eq!(coeffs.len(), n + 1, "Polynomial for GF({}^{}) has wrong length", p, n);
            assert_eq!(coeffs[*n], 1, "Polynomial for GF({}^{}) is not monic", p, n);

            // All coefficients should be in [0, p)
            for &coeff in coeffs.iter() {
                assert!(coeff < *p, "Coefficient {} >= {} in polynomial for GF({}^{})", coeff, p, p, n);
            }
        }
    }
}

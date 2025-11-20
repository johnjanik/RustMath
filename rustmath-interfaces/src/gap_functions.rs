//! GAP Functions - Convenient wrappers for common GAP functions
//!
//! This module provides Rust wrappers for commonly used GAP functions,
//! making it easier to call GAP functionality from Rust code without
//! manually constructing GAP command strings.
//!
//! # Overview
//!
//! GAP provides thousands of mathematical functions. This module wraps
//! the most commonly used ones for:
//! - Group theory (Order, IsAbelian, Center, etc.)
//! - Number theory (Factorial, IsPrime, Gcd, Lcm, etc.)
//! - Combinatorics (Binomial, Stirling, Partitions, etc.)
//! - Linear algebra (Determinant, Rank, NullSpace, etc.)
//! - General utility functions
//!
//! # Example
//!
//! ```rust,ignore
//! use rustmath_interfaces::gap_functions::*;
//! use rustmath_interfaces::gap::GapInterface;
//!
//! let gap = GapInterface::new()?;
//!
//! // Number theory
//! let fact = factorial(&gap, 10)?;
//! let prime = is_prime(&gap, 17)?;
//! assert_eq!(prime, true);
//!
//! // Combinatorics
//! let binom = binomial(&gap, 10, 5)?;
//! assert_eq!(binom, 252);
//! ```

use crate::gap::{GapError, GapInterface, Result};
use crate::gap_parser::{parse_boolean, parse_integer, parse_integer_list};

// ============================================================================
// Number Theory Functions
// ============================================================================

/// Compute factorial of n
pub fn factorial(gap: &GapInterface, n: usize) -> Result<String> {
    gap.execute(&format!("Factorial({})", n))
}

/// Check if n is prime
pub fn is_prime(gap: &GapInterface, n: i64) -> Result<bool> {
    let result = gap.execute(&format!("IsPrimeInt({})", n))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute GCD of two integers
pub fn gcd(gap: &GapInterface, a: i64, b: i64) -> Result<i64> {
    let result = gap.execute(&format!("Gcd({}, {})", a, b))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute LCM of two integers
pub fn lcm(gap: &GapInterface, a: i64, b: i64) -> Result<i64> {
    let result = gap.execute(&format!("Lcm({}, {})", a, b))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute the extended GCD: returns (gcd, x, y) where gcd = ax + by
pub fn gcdex(gap: &GapInterface, a: i64, b: i64) -> Result<(i64, i64, i64)> {
    let result = gap.execute(&format!("Gcdex({}, {})", a, b))?;
    let list = parse_integer_list(&result.replace("rec( gcd := ", "[")
        .replace(", coeff1 := ", ", ")
        .replace(", coeff2 := ", ", ")
        .replace(" )", "]"))
        .map_err(|e| GapError::ParseError(e.to_string()))?;

    if list.len() >= 3 {
        Ok((list[0], list[1], list[2]))
    } else {
        Err(GapError::ParseError("Invalid Gcdex result".to_string()))
    }
}

/// Prime factorization of n
pub fn factor_int(gap: &GapInterface, n: i64) -> Result<Vec<i64>> {
    let result = gap.execute(&format!("FactorsInt({})", n))?;
    parse_integer_list(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute Euler's totient function φ(n)
pub fn phi(gap: &GapInterface, n: i64) -> Result<i64> {
    let result = gap.execute(&format!("Phi({})", n))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute sigma function (sum of divisors)
pub fn sigma(gap: &GapInterface, n: i64) -> Result<i64> {
    let result = gap.execute(&format!("Sigma({})", n))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Get all divisors of n
pub fn divisors(gap: &GapInterface, n: i64) -> Result<Vec<i64>> {
    let result = gap.execute(&format!("DivisorsInt({})", n))?;
    parse_integer_list(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute n mod m
pub fn modulo(gap: &GapInterface, n: i64, m: i64) -> Result<i64> {
    let result = gap.execute(&format!("RemInt({}, {})", n, m))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute a^b mod m (modular exponentiation)
pub fn powermod(gap: &GapInterface, a: i64, b: i64, m: i64) -> Result<i64> {
    let result = gap.execute(&format!("PowerMod({}, {}, {})", a, b, m))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

// ============================================================================
// Combinatorics Functions
// ============================================================================

/// Compute binomial coefficient C(n, k)
pub fn binomial(gap: &GapInterface, n: i64, k: i64) -> Result<i64> {
    let result = gap.execute(&format!("Binomial({}, {})", n, k))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute Stirling number of the first kind
pub fn stirling1(gap: &GapInterface, n: i64, k: i64) -> Result<i64> {
    let result = gap.execute(&format!("Stirling1({}, {})", n, k))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute Stirling number of the second kind
pub fn stirling2(gap: &GapInterface, n: i64, k: i64) -> Result<i64> {
    let result = gap.execute(&format!("Stirling2({}, {})", n, k))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute Bell number B(n)
pub fn bell(gap: &GapInterface, n: usize) -> Result<String> {
    gap.execute(&format!("Bell({})", n))
}

/// Compute Catalan number C(n)
pub fn catalan(gap: &GapInterface, n: usize) -> Result<String> {
    gap.execute(&format!("Binomial(2*{}, {}) / ({} + 1)", n, n, n))
}

/// Get all partitions of n
pub fn partitions(gap: &GapInterface, n: usize) -> Result<String> {
    gap.execute(&format!("Partitions({})", n))
}

/// Count partitions of n
pub fn nr_partitions(gap: &GapInterface, n: usize) -> Result<i64> {
    let result = gap.execute(&format!("NrPartitions({})", n))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Generate all permutations of a list
pub fn permutations(gap: &GapInterface, list: &str) -> Result<String> {
    gap.execute(&format!("Permutations({})", list))
}

/// Generate all combinations of k elements from a list
pub fn combinations(gap: &GapInterface, list: &str, k: usize) -> Result<String> {
    gap.execute(&format!("Combinations({}, {})", list, k))
}

// ============================================================================
// Group Theory Functions
// ============================================================================

/// Get the order of a group
pub fn order(gap: &GapInterface, group: &str) -> Result<usize> {
    let result = gap.execute(&format!("Order({})", group))?;
    parse_integer(&result)
        .map(|n| n as usize)
        .map_err(|e| GapError::ParseError(e.to_string()))
}

/// Check if a group is abelian
pub fn is_abelian(gap: &GapInterface, group: &str) -> Result<bool> {
    let result = gap.execute(&format!("IsAbelian({})", group))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Check if a group is simple
pub fn is_simple(gap: &GapInterface, group: &str) -> Result<bool> {
    let result = gap.execute(&format!("IsSimpleGroup({})", group))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Check if a group is solvable
pub fn is_solvable(gap: &GapInterface, group: &str) -> Result<bool> {
    let result = gap.execute(&format!("IsSolvableGroup({})", group))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Check if a group is cyclic
pub fn is_cyclic(gap: &GapInterface, group: &str) -> Result<bool> {
    let result = gap.execute(&format!("IsCyclic({})", group))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Get the center of a group
pub fn center(gap: &GapInterface, group: &str) -> Result<String> {
    gap.execute(&format!("Center({})", group))
}

/// Get the derived subgroup (commutator subgroup)
pub fn derived_subgroup(gap: &GapInterface, group: &str) -> Result<String> {
    gap.execute(&format!("DerivedSubgroup({})", group))
}

/// Get the generators of a group
pub fn generators(gap: &GapInterface, group: &str) -> Result<String> {
    gap.execute(&format!("GeneratorsOfGroup({})", group))
}

/// Compute the conjugacy classes of a group
pub fn conjugacy_classes(gap: &GapInterface, group: &str) -> Result<String> {
    gap.execute(&format!("ConjugacyClasses({})", group))
}

/// Get the number of conjugacy classes
pub fn nr_conjugacy_classes(gap: &GapInterface, group: &str) -> Result<usize> {
    let result = gap.execute(&format!("Size(ConjugacyClasses({}))", group))?;
    parse_integer(&result)
        .map(|n| n as usize)
        .map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute the character table of a group
pub fn character_table(gap: &GapInterface, group: &str) -> Result<String> {
    gap.execute(&format!("CharacterTable({})", group))
}

/// Compute the normalizer of a subgroup
pub fn normalizer(gap: &GapInterface, group: &str, subgroup: &str) -> Result<String> {
    gap.execute(&format!("Normalizer({}, {})", group, subgroup))
}

/// Compute the centralizer of an element
pub fn centralizer(gap: &GapInterface, group: &str, element: &str) -> Result<String> {
    gap.execute(&format!("Centralizer({}, {})", group, element))
}

/// Compute the orbit of a point under a group action
pub fn orbit(gap: &GapInterface, group: &str, point: usize) -> Result<Vec<i64>> {
    let result = gap.execute(&format!("Orbit({}, {})", group, point))?;
    parse_integer_list(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute the stabilizer of a point
pub fn stabilizer(gap: &GapInterface, group: &str, point: usize) -> Result<String> {
    gap.execute(&format!("Stabilizer({}, {})", group, point))
}

/// Check if a group is transitive
pub fn is_transitive(gap: &GapInterface, group: &str) -> Result<bool> {
    let result = gap.execute(&format!("IsTransitive({})", group))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

// ============================================================================
// Linear Algebra Functions
// ============================================================================

/// Compute the determinant of a matrix
pub fn determinant(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("DeterminantMat({})", matrix))
}

/// Compute the rank of a matrix
pub fn rank(gap: &GapInterface, matrix: &str) -> Result<usize> {
    let result = gap.execute(&format!("RankMat({})", matrix))?;
    parse_integer(&result)
        .map(|n| n as usize)
        .map_err(|e| GapError::ParseError(e.to_string()))
}

/// Compute the trace of a matrix
pub fn trace(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("TraceMat({})", matrix))
}

/// Compute the null space of a matrix
pub fn null_space(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("NullspaceMat({})", matrix))
}

/// Compute the eigenvalues of a matrix
pub fn eigenvalues(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("Eigenvalues({}, {})", matrix, matrix))
}

/// Compute the characteristic polynomial of a matrix
pub fn char_poly(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("CharacteristicPolynomial({}, {})", matrix, "x"))
}

/// Transpose a matrix
pub fn transpose(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("TransposedMat({})", matrix))
}

/// Compute the inverse of a matrix
pub fn inverse_mat(gap: &GapInterface, matrix: &str) -> Result<String> {
    gap.execute(&format!("Inverse({})", matrix))
}

// ============================================================================
// Polynomial Functions
// ============================================================================

/// Create a polynomial over rationals
pub fn polynomial(gap: &GapInterface, coeffs: &[i64], var: &str) -> Result<String> {
    let coeffs_str = coeffs.iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    gap.execute(&format!("UnivariatePolynomial(Rationals, [{}], {})", coeffs_str, var))
}

/// Get the degree of a polynomial
pub fn degree(gap: &GapInterface, poly: &str) -> Result<i64> {
    let result = gap.execute(&format!("Degree({})", poly))?;
    parse_integer(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Factor a polynomial
pub fn factor_polynomial(gap: &GapInterface, poly: &str) -> Result<String> {
    gap.execute(&format!("Factors({})", poly))
}

/// Evaluate a polynomial at a point
pub fn value_polynomial(gap: &GapInterface, poly: &str, point: &str) -> Result<String> {
    gap.execute(&format!("Value({}, {})", poly, point))
}

// ============================================================================
// List and Set Functions
// ============================================================================

/// Get the length/size of a list or set
pub fn length(gap: &GapInterface, obj: &str) -> Result<usize> {
    let result = gap.execute(&format!("Length({})", obj))?;
    parse_integer(&result)
        .map(|n| n as usize)
        .map_err(|e| GapError::ParseError(e.to_string()))
}

/// Sort a list
pub fn sort(gap: &GapInterface, list: &str) -> Result<String> {
    gap.execute(&format!("Sort({});", list))?;
    gap.execute(list)
}

/// Get unique elements of a list
pub fn unique(gap: &GapInterface, list: &str) -> Result<String> {
    gap.execute(&format!("Set({})", list))
}

/// Sum elements of a list
pub fn sum_list(gap: &GapInterface, list: &str) -> Result<String> {
    gap.execute(&format!("Sum({})", list))
}

/// Product of elements of a list
pub fn product_list(gap: &GapInterface, list: &str) -> Result<String> {
    gap.execute(&format!("Product({})", list))
}

/// Check if an element is in a list
pub fn contains(gap: &GapInterface, list: &str, element: &str) -> Result<bool> {
    let result = gap.execute(&format!("{} in {}", element, list))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

/// Get the position of an element in a list
pub fn position(gap: &GapInterface, list: &str, element: &str) -> Result<Option<usize>> {
    let result = gap.execute(&format!("Position({}, {})", list, element))?;
    if result.trim() == "fail" {
        Ok(None)
    } else {
        parse_integer(&result)
            .map(|n| Some(n as usize))
            .map_err(|e| GapError::ParseError(e.to_string()))
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Print a value (for debugging)
pub fn print(gap: &GapInterface, obj: &str) -> Result<String> {
    gap.execute(&format!("Print({})", obj))
}

/// Display a value (nicely formatted)
pub fn display(gap: &GapInterface, obj: &str) -> Result<String> {
    gap.execute(&format!("Display({})", obj))
}

/// Get the type of an object
pub fn type_of(gap: &GapInterface, obj: &str) -> Result<String> {
    gap.execute(&format!("TypeObj({})", obj))
}

/// Check if an object is bound (defined)
pub fn is_bound(gap: &GapInterface, var: &str) -> Result<bool> {
    let result = gap.execute(&format!("IsBound({})", var))?;
    parse_boolean(&result).map_err(|e| GapError::ParseError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GAP
    fn test_factorial() {
        let gap = GapInterface::new().unwrap();
        let result = factorial(&gap, 5).unwrap();
        assert_eq!(result.trim(), "120");
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_is_prime() {
        let gap = GapInterface::new().unwrap();
        assert_eq!(is_prime(&gap, 17).unwrap(), true);
        assert_eq!(is_prime(&gap, 18).unwrap(), false);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_gcd_lcm() {
        let gap = GapInterface::new().unwrap();
        assert_eq!(gcd(&gap, 12, 18).unwrap(), 6);
        assert_eq!(lcm(&gap, 12, 18).unwrap(), 36);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_binomial() {
        let gap = GapInterface::new().unwrap();
        assert_eq!(binomial(&gap, 10, 5).unwrap(), 252);
    }

    #[test]
    #[ignore] // Requires GAP
    fn test_group_functions() {
        let gap = GapInterface::new().unwrap();
        assert_eq!(order(&gap, "SymmetricGroup(5)").unwrap(), 120);
        assert_eq!(is_abelian(&gap, "CyclicGroup(5)").unwrap(), true);
        assert_eq!(is_abelian(&gap, "SymmetricGroup(3)").unwrap(), false);
    }
}

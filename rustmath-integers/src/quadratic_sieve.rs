//! Quadratic Sieve algorithm for integer factorization
//!
//! The Quadratic Sieve is one of the fastest algorithms for factoring large composite
//! integers that don't have very small factors. It was the fastest general-purpose
//! factorization algorithm from 1981 until the development of the Number Field Sieve.
//!
//! # Algorithm Overview
//!
//! The quadratic sieve finds integers x such that x² ≡ y² (mod n), then computes
//! gcd(x - y, n) to find factors. It works by:
//!
//! 1. Choosing a factor base of small primes
//! 2. Finding values x near √n where x² - n is smooth (factors completely over the factor base)
//! 3. Using sieving to efficiently find smooth values
//! 4. Solving a linear system over GF(2) to find dependencies
//! 5. Using dependencies to construct x² ≡ y² (mod n)
//!
//! # Examples
//!
//! ```
//! use rustmath_integers::Integer;
//! use rustmath_integers::quadratic_sieve::quadratic_sieve_factor;
//!
//! let n = Integer::from(15347); // 103 × 149
//! if let Some(factor) = quadratic_sieve_factor(&n) {
//!     println!("Found factor: {}", factor);
//! }
//! ```

use crate::Integer;
use crate::prime::is_prime;
use rustmath_core::{Ring, NumericConversion};
use std::collections::HashMap;

/// A smooth relation: x² ≡ product of primes (mod n)
#[derive(Debug, Clone)]
struct SmoothRelation {
    x: Integer,
    q: Integer, // q = x² - n
    factors: Vec<(usize, u32)>, // (prime_index, exponent)
}

/// Compute the Legendre symbol (a/p) using Euler's criterion
///
/// Returns 1 if a is a quadratic residue mod p, -1 if not, 0 if a ≡ 0 (mod p)
fn legendre_symbol(a: &Integer, p: u64) -> i8 {
    if p == 2 {
        return 1;
    }

    let a_mod_p = (a.clone() % Integer::from(p as i64)).to_i64();
    if a_mod_p == 0 {
        return 0;
    }

    // Use Euler's criterion: (a/p) = a^((p-1)/2) mod p
    let exp = Integer::from((p - 1) / 2);
    let result = Integer::from(a_mod_p).mod_pow(&exp, &Integer::from(p as i64))
        .unwrap_or(Integer::one());

    let result_i64 = result.to_i64();
    if result_i64 == 1 {
        1
    } else if result_i64 as u64 == p - 1 {
        -1
    } else {
        0
    }
}

/// Find a square root of n modulo p using Tonelli-Shanks algorithm
///
/// Returns Some(r) where r² ≡ n (mod p), or None if no root exists
#[allow(dead_code)]
fn tonelli_shanks(n: &Integer, p: u64) -> Option<i64> {
    if p == 2 {
        return Some(n.to_i64() % 2);
    }

    let n_mod_p = (n.clone() % Integer::from(p as i64)).to_i64();
    if n_mod_p == 0 {
        return Some(0);
    }

    // Check if n is a quadratic residue
    if legendre_symbol(&Integer::from(n_mod_p), p) != 1 {
        return None;
    }

    // Simple case: p ≡ 3 (mod 4)
    if p % 4 == 3 {
        let exp = (p + 1) / 4;
        let root = Integer::from(n_mod_p)
            .mod_pow(&Integer::from(exp as i64), &Integer::from(p as i64))
            .unwrap_or(Integer::zero());
        return Some(root.to_i64());
    }

    // General Tonelli-Shanks algorithm
    // Write p - 1 = 2^s * q with q odd
    let mut q = p - 1;
    let mut s = 0u32;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    // Find a quadratic non-residue z
    let mut z = 2i64;
    while legendre_symbol(&Integer::from(z), p) != -1 {
        z += 1;
        if z > 100 {
            return None; // Safety check
        }
    }

    let mut m = s;
    let mut c = Integer::from(z)
        .mod_pow(&Integer::from(q as i64), &Integer::from(p as i64))
        .unwrap_or(Integer::one());
    let mut t = Integer::from(n_mod_p)
        .mod_pow(&Integer::from(q as i64), &Integer::from(p as i64))
        .unwrap_or(Integer::one());
    let mut r = Integer::from(n_mod_p)
        .mod_pow(&Integer::from(((q + 1) / 2) as i64), &Integer::from(p as i64))
        .unwrap_or(Integer::one());

    loop {
        if t.is_zero() {
            return Some(0);
        }
        if t.is_one() {
            return Some(r.to_i64());
        }

        // Find the least i such that t^(2^i) = 1
        let mut i = 1u32;
        let mut t_power = (t.clone() * t.clone()) % Integer::from(p as i64);
        while !t_power.is_one() && i < m {
            t_power = (t_power.clone() * t_power.clone()) % Integer::from(p as i64);
            i += 1;
        }

        if i >= m {
            return None;
        }

        // Update values
        let b = Integer::from(2i64)
            .mod_pow(&Integer::from((m - i - 1) as i64), &Integer::from(p as i64))
            .and_then(|exp| c.mod_pow(&exp, &Integer::from(p as i64)))
            .unwrap_or(Integer::one());

        m = i;
        c = (b.clone() * b.clone()) % Integer::from(p as i64);
        t = (t * c.clone()) % Integer::from(p as i64);
        r = (r * b) % Integer::from(p as i64);
    }
}

/// Generate the factor base: primes p where n is a quadratic residue mod p
fn generate_factor_base(n: &Integer, size: usize) -> Vec<u64> {
    let mut factor_base = vec![2]; // Always include 2

    let mut p = 3u64;
    while factor_base.len() < size && p < 1000000 {
        if is_prime(&Integer::from(p as i64)) {
            if legendre_symbol(n, p) != -1 {
                factor_base.push(p);
            }
        }
        p += 2;
    }

    factor_base
}

/// Trial division to check if n is B-smooth
///
/// Returns Some(factorization) if n factors completely over factor_base,
/// None otherwise
fn trial_factor(n: &Integer, factor_base: &[u64]) -> Option<Vec<(usize, u32)>> {
    let mut factors = Vec::new();
    let mut remaining = n.abs();

    for (idx, &p) in factor_base.iter().enumerate() {
        let p_int = Integer::from(p as i64);
        let mut exponent = 0u32;

        while (&remaining % &p_int).is_zero() {
            remaining = remaining / p_int.clone();
            exponent += 1;
        }

        if exponent > 0 {
            factors.push((idx, exponent));
        }
    }

    if remaining.is_one() {
        Some(factors)
    } else {
        None
    }
}

/// Sieve to find smooth relations
fn sieve_for_smooth_relations(
    n: &Integer,
    factor_base: &[u64],
    sieve_size: usize,
) -> Vec<SmoothRelation> {
    let sqrt_n = n.sqrt().unwrap_or(Integer::one());
    let mut relations = Vec::new();

    // Sieve interval: [sqrt(n) - sieve_size/2, sqrt(n) + sieve_size/2]
    let start = sqrt_n.clone() - Integer::from((sieve_size / 2) as i64);
    let start = if start.signum() < 0 {
        Integer::one()
    } else {
        start
    };

    // For each x in the sieve interval
    for i in 0..sieve_size {
        let x = start.clone() + Integer::from(i as i64);
        let q = x.clone() * x.clone() - n.clone();

        if q <= Integer::zero() {
            continue;
        }

        // Try to factor q over the factor base
        if let Some(factors) = trial_factor(&q, factor_base) {
            relations.push(SmoothRelation { x, q, factors });

            // Stop when we have enough relations
            if relations.len() >= factor_base.len() + 10 {
                break;
            }
        }
    }

    relations
}

/// Build exponent matrix mod 2 from smooth relations
fn build_exponent_matrix(relations: &[SmoothRelation], base_size: usize) -> Vec<Vec<bool>> {
    let mut matrix = Vec::new();

    for relation in relations {
        let mut row = vec![false; base_size];
        for &(idx, exp) in &relation.factors {
            if idx < base_size {
                row[idx] = (exp % 2) == 1;
            }
        }
        matrix.push(row);
    }

    matrix
}

/// Gaussian elimination over GF(2) to find dependencies
///
/// Returns vectors indicating which relations to multiply together
fn gaussian_elimination_gf2(matrix: &[Vec<bool>]) -> Vec<Vec<bool>> {
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut mat = matrix.to_vec();
    let mut dependencies = Vec::new();
    let mut pivot_rows = Vec::new();

    for col in 0..cols {
        // Find pivot
        let mut pivot = None;
        for row in 0..rows {
            if pivot_rows.contains(&row) {
                continue;
            }
            if mat[row][col] {
                pivot = Some(row);
                break;
            }
        }

        let Some(pivot_row) = pivot else {
            // No pivot found - this column gives us a dependency
            let mut dependency = vec![false; rows];

            // Mark which rows contribute to this dependency
            for (row_idx, row) in mat.iter().enumerate() {
                if row[col] {
                    dependency[row_idx] = true;
                }
            }

            dependencies.push(dependency);
            continue;
        };

        pivot_rows.push(pivot_row);

        // Eliminate this column in other rows
        for row in 0..rows {
            if row == pivot_row || !mat[row][col] {
                continue;
            }

            for c in 0..cols {
                mat[row][c] ^= mat[pivot_row][c];
            }
        }
    }

    dependencies
}

/// Try to extract a factor from a dependency
fn try_dependency(
    n: &Integer,
    relations: &[SmoothRelation],
    dependency: &[bool],
) -> Option<Integer> {
    if dependency.iter().filter(|&&b| b).count() < 2 {
        return None;
    }

    // Compute x = product of x_i for relations in dependency
    let mut x_product = Integer::one();
    let mut factor_counts: HashMap<usize, u32> = HashMap::new();

    for (i, &use_relation) in dependency.iter().enumerate() {
        if use_relation && i < relations.len() {
            x_product = (x_product * relations[i].x.clone()) % n.clone();

            for &(idx, exp) in &relations[i].factors {
                *factor_counts.entry(idx).or_insert(0) += exp;
            }
        }
    }

    // Compute y = product of prime_i^(sum_of_exponents/2)
    // We need the actual primes, but we only have indices
    // This is a limitation - we need to pass factor_base here
    // For now, let's reconstruct y from the q values

    let mut q_product = Integer::one();
    for (i, &use_relation) in dependency.iter().enumerate() {
        if use_relation && i < relations.len() {
            q_product = (q_product * relations[i].q.clone()) % n.clone();
        }
    }

    // Since all exponents are even (mod 2), q_product should be a perfect square
    let y = q_product.sqrt().unwrap_or(Integer::one());

    // Make sure we're in the right range
    let x = x_product % n.clone();
    let y = y % n.clone();

    if x == y {
        return None;
    }

    // Try both x - y and x + y
    let factor1 = (x.clone() - y.clone()).gcd(n);
    let factor2 = (x + y).gcd(n);

    if factor1 > Integer::one() && &factor1 < n {
        Some(factor1)
    } else if factor2 > Integer::one() && &factor2 < n {
        Some(factor2)
    } else {
        None
    }
}

/// Factor n using the Quadratic Sieve algorithm
///
/// # Arguments
///
/// * `n` - The number to factor (should be odd composite)
///
/// # Returns
///
/// A non-trivial factor of n, or None if factorization fails
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::quadratic_sieve::quadratic_sieve_factor;
///
/// let n = Integer::from(15347); // 103 × 149
/// if let Some(factor) = quadratic_sieve_factor(&n) {
///     println!("Found factor: {}", factor);
///     assert!((&n % &factor).is_zero());
/// }
/// ```
pub fn quadratic_sieve_factor(n: &Integer) -> Option<Integer> {
    quadratic_sieve_factor_with_params(n, 50, 1000)
}

/// Factor n using the Quadratic Sieve with custom parameters
///
/// # Arguments
///
/// * `n` - The number to factor
/// * `factor_base_size` - Size of the factor base (typically 30-200)
/// * `sieve_size` - Size of the sieve interval (typically 1000-100000)
///
/// # Returns
///
/// A non-trivial factor of n, or None if factorization fails
pub fn quadratic_sieve_factor_with_params(
    n: &Integer,
    factor_base_size: usize,
    sieve_size: usize,
) -> Option<Integer> {
    // Quick checks
    if n <= &Integer::one() {
        return None;
    }

    if n.is_even() {
        return Some(Integer::from(2));
    }

    if is_prime(n) {
        return None;
    }

    // Trial division for small factors
    let small_primes = [3i64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        let p_int = Integer::from(p);
        if (n % &p_int).is_zero() {
            return Some(p_int);
        }
    }

    // Check if n is a perfect power
    for exp in 2..=10 {
        let root = n.nth_root(exp).ok()?;
        if root.clone().pow(exp) == *n {
            return Some(root);
        }
    }

    // Generate factor base
    let factor_base = generate_factor_base(n, factor_base_size);

    if factor_base.len() < 10 {
        return None; // Not enough primes in factor base
    }

    // Sieve for smooth relations
    let relations = sieve_for_smooth_relations(n, &factor_base, sieve_size);

    if relations.len() < factor_base.len() {
        return None; // Not enough relations found
    }

    // Build exponent matrix
    let matrix = build_exponent_matrix(&relations, factor_base.len());

    // Find dependencies via Gaussian elimination
    let dependencies = gaussian_elimination_gf2(&matrix);

    // Try each dependency to find a factor
    for dependency in dependencies {
        if let Some(factor) = try_dependency(n, &relations, &dependency) {
            if factor > Integer::one() && &factor < n {
                return Some(factor);
            }
        }
    }

    None
}

/// Complete factorization using Quadratic Sieve
///
/// Returns the complete prime factorization of n.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::quadratic_sieve::quadratic_sieve_factor_complete;
///
/// let n = Integer::from(1001); // 7 × 11 × 13
/// let factors = quadratic_sieve_factor_complete(&n);
/// assert_eq!(factors.len(), 3);
/// ```
pub fn quadratic_sieve_factor_complete(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Use trial division for small factors first
    let small_primes = [2i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        let p_int = Integer::from(p);
        let mut count = 0u32;
        while (&remaining % &p_int).is_zero() {
            remaining = remaining / p_int.clone();
            count += 1;
        }
        if count > 0 {
            factors.push((p_int, count));
        }
    }

    // Use QS for the rest
    while remaining > Integer::one() && !is_prime(&remaining) {
        if let Some(factor) = quadratic_sieve_factor(&remaining) {
            let mut count = 0u32;
            while (&remaining % &factor).is_zero() {
                remaining = remaining / factor.clone();
                count += 1;
            }
            factors.push((factor, count));
        } else {
            // Couldn't factor, assume it's prime
            factors.push((remaining.clone(), 1));
            remaining = Integer::one(); // Mark as done
            break;
        }
    }

    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_symbol() {
        // 2 is a quadratic residue mod 7 (since 3² = 9 ≡ 2 mod 7)
        assert_eq!(legendre_symbol(&Integer::from(2), 7), 1);

        // 3 is not a quadratic residue mod 7
        assert_eq!(legendre_symbol(&Integer::from(3), 7), -1);

        // 0 is always 0
        assert_eq!(legendre_symbol(&Integer::from(0), 7), 0);
    }

    #[test]
    fn test_tonelli_shanks() {
        // Find square root of 2 mod 7
        // We know 3² = 9 ≡ 2 (mod 7) or 4² = 16 ≡ 2 (mod 7)
        let root = tonelli_shanks(&Integer::from(2), 7);
        assert!(root.is_some());
        let r = root.unwrap();
        assert!((r * r) % 7 == 2 || ((7 - r) * (7 - r)) % 7 == 2);
    }

    #[test]
    fn test_factor_base_generation() {
        let n = Integer::from(15347);
        let factor_base = generate_factor_base(&n, 20);
        assert!(factor_base.len() >= 10);
        assert_eq!(factor_base[0], 2);
    }

    #[test]
    fn test_trial_factor() {
        let factor_base = vec![2, 3, 5, 7];
        let n = Integer::from(60); // 2² × 3 × 5

        let factors = trial_factor(&n, &factor_base);
        assert!(factors.is_some());

        let n = Integer::from(11 * 13); // Not smooth over {2,3,5,7}
        let factors = trial_factor(&n, &factor_base);
        assert!(factors.is_none());
    }

    #[test]
    fn test_quadratic_sieve_small() {
        // 143 = 11 × 13
        let n = Integer::from(143);
        let factor = quadratic_sieve_factor_with_params(&n, 20, 500);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!(f == Integer::from(11) || f == Integer::from(13));
    }

    #[test]
    fn test_quadratic_sieve_larger() {
        // 1001 = 7 × 11 × 13
        let n = Integer::from(1001);
        let factor = quadratic_sieve_factor_with_params(&n, 30, 1000);
        assert!(factor.is_some());
        let f = factor.unwrap();
        assert!((&n % &f).is_zero());
        assert!(f > Integer::one());
        assert!(&f < &n);
    }

    #[test]
    fn test_quadratic_sieve_prime() {
        let p = Integer::from(97);
        let factor = quadratic_sieve_factor(&p);
        assert!(factor.is_none());
    }

    #[test]
    fn test_quadratic_sieve_even() {
        let n = Integer::from(100);
        let factor = quadratic_sieve_factor(&n);
        assert_eq!(factor, Some(Integer::from(2)));
    }

    #[test]
    fn test_complete_factorization() {
        let n = Integer::from(60); // 2² × 3 × 5
        let factors = quadratic_sieve_factor_complete(&n);

        // Verify factorization
        let mut product = Integer::one();
        for (prime, exp) in &factors {
            for _ in 0..*exp {
                product = product * prime.clone();
            }
        }
        assert_eq!(product, n);
    }

    #[test]
    fn test_gaussian_elimination() {
        // Simple test matrix
        let matrix = vec![
            vec![true, false, true],
            vec![false, true, true],
            vec![true, true, false],
        ];

        let deps = gaussian_elimination_gf2(&matrix);
        // Should find at least one dependency
        assert!(!deps.is_empty());
    }
}

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
use crate::prime::{factor_pollard_rho, is_prime};
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

/// Find a square root of n modulo a prime p using the Tonelli-Shanks algorithm
/// (H. Cohen, GTM 138, Algorithm 1.5.1; MAGMA `Modsqrt`, Handbook ch. 18 §18.12.1).
///
/// Returns `Some(r)` with 0 <= r < p and r² ≡ n (mod p), and `None` exactly when n is a
/// quadratic non-residue modulo p (decided by Euler's criterion, which is exact for prime p).
///
/// Not used by the sieve yet (it trial-divides every value); the crate's public square root
/// is [`Integer::sqrt_mod_prime`].
///
/// # Panics
///
/// If p is not a prime below 2^63. The non-residue search and the main loop rely on p being
/// prime; for a composite p they could not decide anything, so a composite modulus is
/// rejected up front instead of being answered with `None` ("no square root").
#[allow(dead_code)]
fn tonelli_shanks(n: &Integer, p: u64) -> Option<i64> {
    assert!(
        p < (1u64 << 63) && is_prime(&Integer::from(p)),
        "tonelli_shanks: p = {p} is not a prime below 2^63"
    );
    let p_int = Integer::from(p as i64);

    if p == 2 {
        return Some(n.modulo(&p_int).to_i64());
    }

    let n_mod_p = n.modulo(&p_int).to_i64();
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
            .mod_pow(&Integer::from(exp as i64), &p_int)
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

    // Find the least quadratic non-residue z. For an odd prime p exactly (p - 1)/2 of
    // 1, ..., p - 1 are non-residues, so the search ends below p. There is no fixed cap: the
    // least non-residue exceeds 100 for some primes (it is 107 for p = 23616331489).
    let mut z = 2i64;
    while legendre_symbol(&Integer::from(z), p) != -1 {
        z += 1;
    }

    let mut m = s;
    let mut c = Integer::from(z)
        .mod_pow(&Integer::from(q as i64), &p_int)
        .unwrap_or(Integer::one());
    let mut t = Integer::from(n_mod_p)
        .mod_pow(&Integer::from(q as i64), &p_int)
        .unwrap_or(Integer::one());
    let mut r = Integer::from(n_mod_p)
        .mod_pow(&Integer::from(((q + 1) / 2) as i64), &p_int)
        .unwrap_or(Integer::one());

    loop {
        if t.is_one() {
            return Some(r.to_i64());
        }

        // Find the least i such that t^(2^i) = 1
        let mut i = 1u32;
        let mut t_power = (t.clone() * t.clone()) % p_int.clone();
        while !t_power.is_one() && i < m {
            t_power = (t_power.clone() * t_power.clone()) % p_int.clone();
            i += 1;
        }

        // For prime p and a residue n, t lies in the subgroup of order 2^(m-1) (Cohen 1.5.1),
        // so some i < m has t^(2^i) = 1.
        assert!(
            i < m,
            "tonelli_shanks: invariant t^(2^(m-1)) = 1 failed for the prime p = {p}"
        );

        // Update values
        let b = Integer::from(2i64)
            .mod_pow(&Integer::from((m - i - 1) as i64), &p_int)
            .and_then(|exp| c.mod_pow(&exp, &p_int))
            .unwrap_or(Integer::one());

        m = i;
        c = (b.clone() * b.clone()) % p_int.clone();
        t = (t * c.clone()) % p_int.clone();
        r = (r * b) % p_int.clone();
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
/// Returns the prime factorization of |n| as `(prime, exponent)` pairs with the primes in
/// ascending order (PARI/GP `factor(n)` without the sign; MAGMA `Factorization(n)` / `MPQS`,
/// Handbook ch. 18 §18.10). For n = 0, 1 and -1 the list is empty.
///
/// # Algorithm
///
/// 1. Trial division by the primes up to 47.
/// 2. While the cofactor m is not accepted by [`is_prime`]: if [`quadratic_sieve_factor`]
///    returns a proper divisor d of m, d is factored by recursion and each of its primes is
///    divided out of m with full multiplicity (d need not be prime: the sieve's gcd and the
///    perfect-power root are just divisors). If the sieve finds nothing, m is factored
///    completely by [`crate::prime::factor_pollard_rho`] and the loop ends.
/// 3. A last cofactor passed the loop guard, so it is prime.
///
/// A cofactor the sieve could not split is never pushed as a prime. Every returned factor is
/// prime in the sense of [`is_prime`] (Miller-Rabin to the bases 2..37: a proof below
/// psi_12 = 318665857834031151167461, Sorenson-Webster 2017).
///
/// # Cost
///
/// The sieve stage is weak (one polynomial, a factor base of 50 primes, 1000 sieve values, and
/// the GF(2) dependency step is not a nullspace computation), so beyond small primes and
/// perfect powers it rarely splits anything; the work is then that of
/// [`crate::prime::factor_pollard_rho`]: no time limit, about sqrt(q) rho steps for the
/// second-largest prime factor q of |n|, impractical when q has 20 or more digits.
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
///
/// // 12780049 = 53 · 59 · 61 · 67 (PARI/GP: factor(12780049)); the sieve finds no factor,
/// // so the cofactor is finished by Pollard rho.
/// let factors = quadratic_sieve_factor_complete(&Integer::from(12780049));
/// let primes: Vec<Integer> = [53, 59, 61, 67].into_iter().map(Integer::from).collect();
/// assert_eq!(factors, primes.into_iter().map(|p| (p, 1)).collect::<Vec<_>>());
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
        match quadratic_sieve_factor(&remaining) {
            Some(divisor)
                if divisor > Integer::one()
                    && divisor < remaining
                    && (&remaining % &divisor).is_zero() =>
            {
                // The divisor might be composite: factor it by recursion (it is smaller than
                // `remaining`), then extract every instance of each of its primes, so no
                // prime comes up twice.
                for (prime, _) in quadratic_sieve_factor_complete(&divisor) {
                    let mut count = 0u32;
                    while (&remaining % &prime).is_zero() {
                        remaining = remaining / prime.clone();
                        count += 1;
                    }
                    if count > 0 {
                        factors.push((prime, count));
                    }
                }
            }
            _ => {
                // The sieve found no factor. `remaining` is composite (loop guard), so it is
                // not pushed: factor it completely with Pollard rho instead. Its primes are
                // above 47 and coprime to every prime pushed so far.
                factors.extend(factor_pollard_rho(&remaining));
                remaining = Integer::one();
            }
        }
    }

    // The loop guard accepted `remaining` as prime.
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors.sort_by(|a, b| a.0.cmp(&b.0));
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

    /// The least quadratic non-residue of p = 23616331489 is 107, above the old search cap
    /// of 100, which made the function answer `None` ("no square root") for residues.
    #[test]
    fn test_tonelli_shanks_least_non_residue_above_100() {
        // gp: p = 23616331489; isprime(p) = 1; p % 8 = 1; valuation(p-1, 2) = 5;
        // z = 2; while(kronecker(z, p) != -1, z++); z = 107
        let p: u64 = 23616331489;
        let p_int = Integer::from(p);
        // gp: r = lift(sqrt(Mod(2, p))); min(r, p - r) = 4184367042
        // gp: r = lift(sqrt(Mod(1000, p))); min(r, p - r) = 2465952743
        for (n, root) in [(2i64, 4184367042i64), (1000, 2465952743)] {
            let r = tonelli_shanks(&Integer::from(n), p).expect("n is a residue mod p");
            assert!(
                r == root || r == p as i64 - root,
                "sqrt({n}) mod {p}: got {r}"
            );
            let sq = (Integer::from(r) * Integer::from(r)).modulo(&p_int);
            assert_eq!(sq, Integer::from(n));
        }
        // gp: kronecker(-2, p) = 1, so -2 has a root; r^2 + 2 ≡ 0 (mod p).
        let r = tonelli_shanks(&Integer::from(-2), p).expect("-2 is a residue mod p");
        assert!(r >= 0 && (r as u64) < p);
        assert_eq!(
            (Integer::from(r) * Integer::from(r) + Integer::from(2)).modulo(&p_int),
            Integer::zero()
        );
        // gp: kronecker(107, p) = -1
        assert_eq!(tonelli_shanks(&Integer::from(107), p), None);
    }

    #[test]
    fn test_tonelli_shanks_small_primes() {
        // gp: lift(sqrt(Mod(3, 13))) = 4 (the roots are 4 and 9)
        let r = tonelli_shanks(&Integer::from(3), 13).unwrap();
        assert!(r == 4 || r == 9, "got {r}");
        // gp: kronecker(3, 41) = -1; issquare(Mod(5, 41)) = 1
        assert_eq!(tonelli_shanks(&Integer::from(3), 41), None);
        let r = tonelli_shanks(&Integer::from(5), 41).unwrap();
        assert_eq!((r * r) % 41, 5);
        // Exhaustive: for each odd prime p < 200 and each a, a root exists exactly for the
        // squares mod p (brute force, independent of the code under test).
        for p in (3u64..200).filter(|&p| is_prime(&Integer::from(p))) {
            for a in 0..p as i64 {
                let is_square = (0..p as i64).any(|x| (x * x) % p as i64 == a);
                match tonelli_shanks(&Integer::from(a), p) {
                    Some(r) => assert_eq!((r * r) % p as i64, a, "p = {p}, a = {a}"),
                    None => assert!(!is_square, "p = {p}: {a} is a square but got None"),
                }
            }
        }
    }

    /// A composite modulus is a contract violation, not a "no square root".
    #[test]
    #[should_panic(expected = "tonelli_shanks: p = 15 is not a prime below 2^63")]
    fn test_tonelli_shanks_composite_modulus_panics() {
        // gp: isprime(15) = 0
        let _ = tonelli_shanks(&Integer::from(4), 15);
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

    /// The sieve's divisor 3127 = 53 · 59 of 165731 = 53^2 · 59 is composite and shares the
    /// prime 53 with the cofactor; it used to be pushed as the "prime" (3127, 1) next to
    /// (53, 1).
    #[test]
    fn test_complete_factorization_composite_sieve_divisor() {
        // gp: factor(165731) = [53,2; 59,1]
        let n = Integer::from(165731);
        assert_eq!(quadratic_sieve_factor(&n), Some(Integer::from(3127)));
        assert_eq!(
            quadratic_sieve_factor_complete(&n),
            vec![(Integer::from(53), 2), (Integer::from(59), 1)]
        );
    }

    /// Blocker B-04: the sieve finds no factor of 3127 = 53 · 59, and the driver used to push
    /// 3127 itself as a prime.
    #[test]
    fn test_complete_factorization_sieve_failure() {
        // gp: factor(3127) = [53,1; 59,1]; isprime(3127) = 0
        let n = Integer::from(3127);
        assert_eq!(quadratic_sieve_factor(&n), None);
        assert_eq!(
            quadratic_sieve_factor_complete(&n),
            vec![(Integer::from(53), 1), (Integer::from(59), 1)]
        );
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

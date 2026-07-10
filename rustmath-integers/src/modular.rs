//! Primitive roots modulo n
//!
//! The `ModularInteger` element type that used to live here was an unused
//! duplicate of `rustmath_finitefields::IntegerMod` (the canonical Z/nZ
//! element, with the `Zmod` parent) and has been deleted. Only the
//! pure-`Integer` free function [`primitive_roots`] remains, because it is
//! independent of any element type (used by `rustmath-constants` primality
//! certificates).

use crate::Integer;
use rustmath_core::NumericConversion;

/// Compute the multiplicative order of `g` modulo `n`, i.e. the smallest
/// k > 0 with g^k ≡ 1 (mod n), or `None` if `g` is not a unit mod n.
///
/// Iterates at most n steps (the order divides φ(n) < n when it exists).
fn multiplicative_order_mod(g: &Integer, n: &Integer) -> Option<Integer> {
    if !g.gcd(n).is_one() {
        return None;
    }

    let mut power = g.clone() % n.clone();
    let mut k = Integer::one();
    while !power.is_one() && k < *n {
        power = (power * g.clone()) % n.clone();
        k = k + Integer::one();
    }

    if power.is_one() {
        Some(k)
    } else {
        None
    }
}

/// Find primitive roots modulo n
///
/// A primitive root modulo n is an integer g such that every integer coprime
/// to n is congruent to a power of g modulo n. Equivalently, g has
/// multiplicative order φ(n).
///
/// Primitive roots exist if and only if n is 1, 2, 4, p^k, or 2p^k for odd
/// prime p.
///
/// # Search bound (honest limitation)
///
/// Candidates are searched in `1..min(n, 1000)` — the same bound the previous
/// implementation applied silently. For n ≤ 1000 the returned list is
/// complete; for larger n it contains exactly the primitive roots below
/// 1000 (possibly none, even when primitive roots exist).
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_integers::modular::primitive_roots;
///
/// let roots = primitive_roots(&Integer::from(7));
/// // 7 is prime, so it has primitive roots: [3, 5]
/// assert_eq!(roots, vec![Integer::from(3), Integer::from(5)]);
/// ```
pub fn primitive_roots(n: &Integer) -> Vec<Integer> {
    if *n <= Integer::one() {
        return vec![];
    }

    // Calculate φ(n)
    let phi_n = match n.euler_phi() {
        Ok(val) => val,
        Err(_) => return vec![],
    };

    let mut roots = Vec::new();

    // Search bound preserved from the previous implementation (see docs).
    for candidate_val in 1..n.to_usize().unwrap_or(1000).min(1000) {
        let candidate = Integer::from(candidate_val as i64);

        if multiplicative_order_mod(&candidate, n).as_ref() == Some(&phi_n) {
            roots.push(candidate);
        }
    }

    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplicative_order_mod() {
        // Order of 2 mod 7: 2^3 = 8 ≡ 1 (mod 7)
        assert_eq!(
            multiplicative_order_mod(&Integer::from(2), &Integer::from(7)),
            Some(Integer::from(3))
        );

        // Order of 3 mod 7 (primitive root): φ(7) = 6
        assert_eq!(
            multiplicative_order_mod(&Integer::from(3), &Integer::from(7)),
            Some(Integer::from(6))
        );

        // Non-unit has no order: gcd(6, 9) = 3
        assert_eq!(
            multiplicative_order_mod(&Integer::from(6), &Integer::from(9)),
            None
        );

        // 1 has order 1
        assert_eq!(
            multiplicative_order_mod(&Integer::one(), &Integer::from(11)),
            Some(Integer::one())
        );
    }

    #[test]
    fn test_primitive_roots() {
        // Primitive roots of 7 (prime)
        let roots = primitive_roots(&Integer::from(7));
        assert_eq!(roots.len(), 2); // φ(φ(7)) = φ(6) = 2
        assert!(roots.contains(&Integer::from(3)));
        assert!(roots.contains(&Integer::from(5)));

        // Primitive roots of 14 = 2 * 7
        let roots = primitive_roots(&Integer::from(14));
        assert!(!roots.is_empty());

        // 8 has no primitive roots (not 1, 2, 4, p^k, or 2p^k)
        let roots = primitive_roots(&Integer::from(8));
        assert!(roots.is_empty());
    }
}

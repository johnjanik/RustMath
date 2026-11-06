//! Generic group operations
//!
//! This module provides algorithms that work with any group element,
//! including discrete logarithm, order computation, and related operations.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::generic;
//!
//! // Compute powers using binary exponentiation
//! let base = 5;
//! let result = generic::power(base, 10, |a, b| a * b, 1);
//! assert_eq!(result, 5_i32.pow(10));
//! ```

use std::collections::HashMap;

/// Compute a^n using binary exponentiation
///
/// # Arguments
/// * `base` - The base element
/// * `exp` - The exponent
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
///
/// # Examples
/// ```
/// use rustmath_groups::generic::power;
///
/// let result = power(3, 5, |a, b| a * b, 1);
/// assert_eq!(result, 243);
/// ```
pub fn power<T: Clone>(base: T, exp: usize, mul: fn(T, T) -> T, identity: T) -> T {
    if exp == 0 {
        return identity;
    }

    let mut result = identity;
    let mut b = base;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = mul(result, b.clone());
        }
        e >>= 1;
        // Square only while bits remain: a final, unused squaring would compute
        // base^(2^(k+1)) for a k-bit exponent, which can overflow even when
        // base^exp itself fits (e.g. 5^10 in i32).
        if e > 0 {
            b = mul(b.clone(), b);
        }
    }

    result
}

/// Compute the inverse power a^(-n)
///
/// # Arguments
/// * `base` - The base element
/// * `exp` - The exponent
/// * `mul` - Multiplication operation
/// * `inv` - Inverse operation
/// * `identity` - The identity element
pub fn power_signed<T: Clone>(
    base: T,
    exp: isize,
    mul: fn(T, T) -> T,
    inv: fn(T) -> T,
    identity: T,
) -> T {
    if exp >= 0 {
        power(base, exp as usize, mul, identity)
    } else {
        let inv_base = inv(base);
        power(inv_base, (-exp) as usize, mul, identity)
    }
}

/// Baby-step giant-step algorithm for discrete logarithm
///
/// Solves base^x = target for x in the range [lower, upper]
///
/// # Arguments
/// * `base` - The base element
/// * `target` - The target element to reach
/// * `lower` - Lower bound on the logarithm
/// * `upper` - Upper bound on the logarithm
/// * `mul` - Multiplication operation
/// * `inv` - Inverse operation
/// * `identity` - The identity element
/// * `eq` - Equality test
/// * `hash` - Hash function for storing elements
///
/// # Returns
/// Some(x) if base^x = target for some x in [lower, upper], None otherwise
pub fn bsgs<T: Clone + Eq + std::hash::Hash>(
    base: T,
    target: T,
    lower: usize,
    upper: usize,
    mul: fn(T, T) -> T,
    inv: fn(T) -> T,
    identity: T,
) -> Option<usize> {
    if upper < lower {
        return None;
    }

    let range = upper - lower + 1;
    let m = (range as f64).sqrt().ceil() as usize;

    // Baby step: compute and store base^j for j = 0..m
    let mut table: HashMap<T, usize> = HashMap::new();

    let mut gamma = identity.clone();
    for j in 0..m {
        table.insert(gamma.clone(), j);
        gamma = mul(gamma, base.clone());
    }

    // Giant step: compute target * (base^m)^(-i) for i = 0..ceil(range/m)
    let base_m = power(base.clone(), m, mul, identity.clone());
    let base_m_inv = inv(base_m);

    // Adjust target by base^(-lower)
    let adjustment = power(base.clone(), lower, mul, identity.clone());
    let adjusted_target = mul(target, inv(adjustment));

    let mut gamma = adjusted_target;
    for i in 0..((range + m - 1) / m) {
        if let Some(&j) = table.get(&gamma) {
            let x = lower + i * m + j;
            if x <= upper {
                return Some(x);
            }
        }
        gamma = mul(gamma.clone(), base_m_inv.clone());
    }

    None
}

/// Discrete logarithm using Pollard's rho algorithm
///
/// Solves base^x = target assuming the group has known order
///
/// # Arguments
/// * `base` - The base element
/// * `target` - The target element
/// * `order` - The order of the group (or subgroup)
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
/// * `partition` - Function to partition elements into 3 classes (returns 0, 1, or 2)
///
/// # Returns
/// `Some(x)` only if `base^x = target` has been verified. `None` means no
/// verified logarithm was found within `order` iterations; it does **not** prove
/// that `target ∉ ⟨base⟩` (the walk can fail to produce a usable collision).
///
/// The collision solve `x = (a − c)/(d − b) mod order` is only valid when
/// `target ∈ ⟨base⟩` and the orders of `base` and `target` divide `order`; for
/// other inputs (e.g. `order = |⟨base⟩|` and `target ∉ ⟨base⟩`) it yields a
/// spurious exponent, so every candidate is checked before it is returned.
pub fn discrete_log_rho<T: Clone + Eq>(
    base: T,
    target: T,
    order: usize,
    mul: fn(T, T) -> T,
    identity: T,
    partition: fn(&T) -> usize,
) -> Option<usize> {
    // Using Floyd's cycle detection
    let mut x = identity.clone();
    let mut a = 0usize;
    let mut b = 0usize;

    let mut y = x.clone();
    let mut c = a;
    let mut d = b;

    let iterate = |elem: T, a: usize, b: usize| -> (T, usize, usize) {
        match partition(&elem) {
            0 => {
                // Multiply by target
                (mul(elem, target.clone()), a, (b + 1) % order)
            }
            1 => {
                // Multiply by base
                (mul(elem, base.clone()), (a + 1) % order, b)
            }
            _ => {
                // Square
                (mul(elem.clone(), elem), (2 * a) % order, (2 * b) % order)
            }
        }
    };

    for _ in 0..order {
        // Tortoise moves one step
        let (x_new, a_new, b_new) = iterate(x, a, b);
        x = x_new;
        a = a_new;
        b = b_new;

        // Hare moves two steps
        let (y_new, c_new, d_new) = iterate(y.clone(), c, d);
        let (y_new2, c_new2, d_new2) = iterate(y_new, c_new, d_new);
        y = y_new2;
        c = c_new2;
        d = d_new2;

        if x == y {
            // Found collision: base^a * target^b = base^c * target^d
            // So: base^(a-c) = target^(d-b)
            // Therefore: target = base^((a-c)/(d-b)) mod order

            let num = if a >= c {
                a - c
            } else {
                order + a - c
            };

            let den = if d >= b {
                d - b
            } else {
                order + d - b
            };

            if den == 0 {
                continue;
            }

            // Solve num ≡ x * den (mod order), then verify: the congruence is
            // only meaningful when target ∈ <base>.
            if let Some(inv_den) = mod_inverse(den, order) {
                let candidate = ((num as u128 * inv_den as u128) % order as u128) as usize;
                if power(base.clone(), candidate, mul, identity.clone()) == target {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

/// Compute the order of a group element
///
/// # Arguments
/// * `elem` - The element whose order to compute
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
/// * `max_order` - Maximum order to check (prevents infinite loops)
///
/// # Returns
/// Some(n) where n is the smallest positive integer with elem^n = identity,
/// or None if no such n <= max_order exists
pub fn order<T: Clone + Eq>(
    elem: T,
    mul: fn(T, T) -> T,
    identity: T,
    max_order: usize,
) -> Option<usize> {
    if elem == identity {
        return Some(1);
    }

    let mut current = elem.clone();
    for n in 1..=max_order {
        if current == identity {
            return Some(n);
        }
        current = mul(current, elem.clone());
    }

    None
}

/// Determine the exact order from a known multiple
///
/// If we know elem^m = identity, this finds the smallest n dividing m
/// such that elem^n = identity.
///
/// # Arguments
/// * `elem` - The element
/// * `multiple` - A known multiple of the order
/// * `factors` - Prime factorization of the multiple as (prime, exponent) pairs
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
///
/// # Returns
/// The exact order of the element
pub fn order_from_multiple<T: Clone + Eq>(
    elem: T,
    multiple: usize,
    factors: &[(usize, usize)],
    mul: fn(T, T) -> T,
    identity: T,
) -> usize {
    let mut order = multiple;

    for &(p, k) in factors {
        let mut pk = 1usize;
        for _ in 0..k {
            pk *= p;
        }

        // Try dividing out powers of p
        for _ in 0..k {
            let test_order = order / p;
            let test_elem = power(elem.clone(), test_order, mul, identity.clone());

            if test_elem == identity {
                order = test_order;
            } else {
                break;
            }
        }
    }

    order
}

/// Test if an element has a specific order
///
/// # Arguments
/// * `elem` - The element to test
/// * `n` - The purported order
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
///
/// # Returns
/// true if elem has order exactly n, false otherwise
pub fn has_order<T: Clone + Eq>(
    elem: T,
    n: usize,
    mul: fn(T, T) -> T,
    identity: T,
) -> bool {
    // Check that elem^n = identity
    let power_n = power(elem.clone(), n, mul, identity.clone());
    if power_n != identity {
        return false;
    }

    // Check that elem^k != identity for all proper divisors k of n
    for k in 1..n {
        if n % k == 0 {
            let power_k = power(elem.clone(), k, mul, identity.clone());
            if power_k == identity {
                return false;
            }
        }
    }

    true
}

/// Compute the commutator [a, b] = a * b * a^(-1) * b^(-1)
///
/// # Arguments
/// * `a` - First element
/// * `b` - Second element
/// * `mul` - Multiplication operation
/// * `inv` - Inverse operation
pub fn commutator<T: Clone>(a: T, b: T, mul: fn(T, T) -> T, inv: fn(T) -> T) -> T {
    let ab = mul(a.clone(), b.clone());
    let a_inv = inv(a);
    let b_inv = inv(b);
    mul(ab, mul(a_inv, b_inv))
}

/// Compute the conjugate b * a * b^(-1)
///
/// # Arguments
/// * `a` - Element to conjugate
/// * `b` - Element to conjugate by
/// * `mul` - Multiplication operation
/// * `inv` - Inverse operation
pub fn conjugate<T: Clone>(a: T, b: T, mul: fn(T, T) -> T, inv: fn(T) -> T) -> T {
    let ba = mul(b.clone(), a);
    let b_inv = inv(b);
    mul(ba, b_inv)
}

/// Compute modular inverse using extended Euclidean algorithm
///
/// Returns Some(x) where (a * x) % m == 1, or None if no inverse exists
fn mod_inverse(a: usize, m: usize) -> Option<usize> {
    let (mut old_r, mut r) = (a as isize, m as isize);
    let (mut old_s, mut s) = (1isize, 0isize);

    while r != 0 {
        let quotient = old_r / r;
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;
    }

    if old_r != 1 {
        return None; // No inverse exists
    }

    // Make sure result is positive
    let result = if old_s < 0 {
        (old_s + m as isize) as usize
    } else {
        old_s as usize
    };

    Some(result)
}

/// Compute GCD of two numbers
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Compute LCM of two numbers
fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    (a * b) / gcd(a, b)
}

/// Split `lcm(m, n)` into coprime factors: returns `(l, m', n')` with
/// `l = lcm(m, n)`, `m' | m`, `n' | n`, `gcd(m', n') = 1` and `m'·n' = l`.
///
/// Each prime power `p^e ∥ l` goes to `m'` if `p^e | m`, otherwise to `n'`.
/// Computed with gcds only (no factorization), as in Sage's `arith.xlcm`.
fn xlcm(m: usize, n: usize) -> (usize, usize, usize) {
    let l = lcm(m, n);
    let mut m_part = m;
    // `g` collects the primes dividing `n` to a strictly higher power than `m`;
    // strip them from `m`.
    let mut g = gcd(m_part, n / gcd(m, n));
    while g != 1 {
        m_part /= g;
        g = gcd(m_part, g);
    }
    (l, m_part, l / m_part)
}

/// Merge two commuting elements into one whose order is the LCM of their orders
///
/// Given commuting elements `a`, `b` of orders `m`, `n` (e.g. any two elements of
/// an abelian group), splits `lcm(m, n) = m'·n'` with `m' | m`, `n' | n`,
/// `gcd(m', n') = 1` and returns `a^(m/m') · b^(n/n')`. The two factors have
/// coprime orders `m'` and `n'` and commute, so the product has order
/// `m'·n' = lcm(m, n)`. This is the standard "merge points" step of generic
/// group-structure algorithms (cf. Sage `sage.groups.generic.merge_points`).
///
/// The result is only guaranteed when `a` and `b` commute and `order_a`,
/// `order_b` are their exact orders; the function does not check either.
///
/// # Arguments
/// * `a` - First element
/// * `b` - Second element (must commute with `a`)
/// * `order_a` - Exact order of element a (≥ 1)
/// * `order_b` - Exact order of element b (≥ 1)
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
///
/// # Returns
/// An element with order lcm(order_a, order_b)
///
/// # Panics
/// If `order_a` or `order_b` is 0.
pub fn merge_elements<T: Clone + Eq>(
    a: T,
    b: T,
    order_a: usize,
    order_b: usize,
    mul: fn(T, T) -> T,
    identity: T,
) -> T {
    assert!(
        order_a >= 1 && order_b >= 1,
        "merge_elements: element orders must be positive"
    );
    let (_l, m_part, n_part) = xlcm(order_a, order_b);

    let a_part = power(a, order_a / m_part, mul, identity.clone());
    let b_part = power(b, order_b / n_part, mul, identity);

    mul(a_part, b_part)
}

/// Structure description helper - computes a string describing group structure
///
/// This is a simplified version that describes cyclic groups and direct products
pub fn structure_description_cyclic(order: usize) -> String {
    if order == 1 {
        return "Trivial group".to_string();
    }

    // Factor the order
    let factors = prime_factorization(order);

    if factors.len() == 1 && factors[0].1 == 1 {
        return format!("C{}", order);
    }

    // For composite orders, describe as product if appropriate
    let parts: Vec<String> = factors
        .iter()
        .map(|(p, k)| {
            let pk = p.pow(*k as u32);
            format!("C{}", pk)
        })
        .collect();

    if parts.len() == 1 {
        parts[0].clone()
    } else {
        parts.join(" x ")
    }
}

/// Simple prime factorization helper
fn prime_factorization(mut n: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();
    let mut d = 2;

    while d * d <= n {
        let mut exp = 0;
        while n % d == 0 {
            n /= d;
            exp += 1;
        }
        if exp > 0 {
            factors.push((d, exp));
        }
        d += 1;
    }

    if n > 1 {
        factors.push((n, 1));
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power() {
        let mul = |a: i32, b: i32| a * b;
        let result = power(2, 10, mul, 1);
        assert_eq!(result, 1024);

        let result = power(3, 5, mul, 1);
        assert_eq!(result, 243);

        let result = power(5, 0, mul, 1);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_power_signed() {
        let mul = |a: f64, b: f64| a * b;
        let inv = |a: f64| 1.0 / a;

        let result = power_signed(2.0, 3, mul, inv, 1.0);
        assert_eq!(result, 8.0);

        let result = power_signed(2.0, -2, mul, inv, 1.0);
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_bsgs_simple() {
        // Test in Z/11Z multiplicative group
        let mul = |a: usize, b: usize| (a * b) % 11;
        let inv = |a: usize| {
            // Find multiplicative inverse mod 11
            for i in 1..11 {
                if (a * i) % 11 == 1 {
                    return i;
                }
            }
            1
        };

        // Solve 2^x ≡ 3 (mod 11)
        // We know 2^8 ≡ 3 (mod 11)
        let result = bsgs(2, 3, 0, 10, mul, inv, 1);
        assert!(result.is_some());

        let x = result.unwrap();
        let check = power(2, x, mul, 1);
        assert_eq!(check, 3);
    }

    #[test]
    fn test_order() {
        let mul = |a: i32, b: i32| (a * b) % 7;

        // Test order of 3 in Z/7Z (should be 6)
        let ord = order(3, mul, 1, 10);
        assert_eq!(ord, Some(6));

        // Test order of 1 (should be 1)
        let ord = order(1, mul, 1, 10);
        assert_eq!(ord, Some(1));
    }

    #[test]
    fn test_has_order() {
        let mul = |a: i32, b: i32| (a * b) % 7;

        assert!(has_order(3, 6, mul, 1));
        assert!(!has_order(3, 3, mul, 1));
        assert!(has_order(1, 1, mul, 1));
    }

    #[test]
    fn test_order_from_multiple() {
        let mul = |a: i32, b: i32| (a * b) % 13;

        // 2 has order 12 in Z/13Z
        // We know 2^12 = 1, so we can find exact order from multiple 12
        let factors = vec![(2, 2), (3, 1)]; // 12 = 2^2 * 3
        let exact_order = order_from_multiple(2, 12, &factors, mul, 1);
        assert_eq!(exact_order, 12);
    }

    #[test]
    fn test_commutator() {
        // In an abelian group, all commutators are identity
        let mul = |a: i32, b: i32| (a + b) % 10;
        let inv = |a: i32| (10 - a) % 10;

        let comm = commutator(3, 5, mul, inv);
        assert_eq!(comm, 0); // 0 is identity in Z/10Z under addition
    }

    #[test]
    fn test_conjugate() {
        // In an abelian group, conjugate equals the element
        let mul = |a: i32, b: i32| (a + b) % 10;
        let inv = |a: i32| (10 - a) % 10;

        let conj = conjugate(3, 5, mul, inv);
        assert_eq!(conj, 3);
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 11), Some(4)); // 3 * 4 ≡ 1 (mod 11)
        assert_eq!(mod_inverse(2, 7), Some(4)); // 2 * 4 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(2, 4), None); // 2 has no inverse mod 4
    }

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(7, 13), 91);
    }

    #[test]
    fn test_merge_elements() {
        let mul = |a: i32, b: i32| (a + b) % 12;

        // In Z/12Z: element 3 has order 4, element 4 has order 3
        // Their LCM should be 12
        let merged = merge_elements(3, 4, 4, 3, mul, 0);

        // Verify the merged element has order 12
        let ord = order(merged, mul, 0, 20);
        assert_eq!(ord, Some(12));
    }

    #[test]
    fn test_merge_elements_shared_primes() {
        // Z/24Z: 6 has order 4, 4 has order 6; lcm = 12. The orders share the
        // prime 2, so exponents must come from the xlcm split (4, 3), not from
        // order/gcd.
        let mul = |a: i32, b: i32| (a + b) % 24;
        let merged = merge_elements(6, 4, 4, 6, mul, 0);
        assert_eq!(order(merged, mul, 0, 30), Some(12));
        // Orders where one divides the other: result has the larger order.
        let merged = merge_elements(8, 2, 3, 12, mul, 0);
        assert_eq!(order(merged, mul, 0, 30), Some(12));
    }

    #[test]
    fn test_xlcm() {
        for m in 1..=60usize {
            for n in 1..=60usize {
                let (l, mm, nn) = xlcm(m, n);
                assert_eq!(l, lcm(m, n));
                assert_eq!(m % mm, 0);
                assert_eq!(n % nn, 0);
                assert_eq!(gcd(mm, nn), 1);
                assert_eq!(mm * nn, l);
            }
        }
    }

    #[test]
    fn test_power_no_overflow_at_top_bit() {
        let mul = |a: i32, b: i32| a * b;
        // 5^10 = 9765625 fits in i32; the old loop squared once more (5^16) and
        // overflowed.
        assert_eq!(power(5, 10, mul, 1), 9_765_625);
        assert_eq!(power(2, 30, mul, 1), 1 << 30);
    }

    #[test]
    fn test_discrete_log_rho_results_are_verified() {
        // Z/12Z additive, base 4 has order 3 but the caller passes the group
        // order 12. Any returned exponent must actually satisfy base^x = target;
        // targets outside <4> = {0, 4, 8} must give None.
        let mul = |a: usize, b: usize| (a + b) % 12;
        let partition = |x: &usize| x % 3;
        for target in 0..12usize {
            match discrete_log_rho(4, target, 12, mul, 0, partition) {
                Some(x) => assert_eq!(power(4, x, mul, 0), target, "target {target}"),
                None => {}
            }
        }
        // Z/36Z additive, base 4 generates the subgroup {0, 4, ..., 32} of order 9,
        // and the docs allow passing that subgroup order. Target 1 is not in <4>;
        // the unverified collision solve used to return x = 7 (4*7 = 28 != 1).
        let mul36 = |a: usize, b: usize| (a + b) % 36;
        let part36 = |x: &usize| x % 3;
        assert_eq!(discrete_log_rho(4, 1, 9, mul36, 0, part36), None);
        for target in 0..36usize {
            match discrete_log_rho(4, target, 9, mul36, 0, part36) {
                Some(x) => assert_eq!(power(4, x, mul36, 0), target, "target {target}"),
                // None means "not found"; rho is not complete, so it decides nothing.
                None => {}
            }
        }
        // Z/11Z additive, prime order: every target t is 3x for the unique
        // x = 3^{-1} t = 4t mod 11, and rho must find it for every t != 0 (a
        // line-by-line Python port of this function finds all 10), so None
        // here is a failure, not "not found".
        let mul11 = |a: usize, b: usize| (a + b) % 11;
        let part11 = |x: &usize| x % 3;
        for target in 1..11usize {
            let x = discrete_log_rho(3, target, 11, mul11, 0, part11);
            assert_eq!(x, Some((4 * target) % 11), "target {target}");
            assert_eq!(power(3, x.unwrap(), mul11, 0), target);
        }
    }

    #[test]
    fn test_prime_factorization() {
        assert_eq!(prime_factorization(12), vec![(2, 2), (3, 1)]);
        assert_eq!(prime_factorization(17), vec![(17, 1)]);
        assert_eq!(prime_factorization(100), vec![(2, 2), (5, 2)]);
    }

    #[test]
    fn test_structure_description() {
        assert_eq!(structure_description_cyclic(1), "Trivial group");
        assert_eq!(structure_description_cyclic(7), "C7");
        assert_eq!(structure_description_cyclic(6), "C2 x C3");
    }
}

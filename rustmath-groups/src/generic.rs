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
        b = mul(b.clone(), b.clone());
        e >>= 1;
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
/// Some(x) if base^x = target, None if not found
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

            // Solve num ≡ x * den (mod order)
            if let Some(inv_den) = mod_inverse(den, order) {
                return Some((num * inv_den) % order);
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

/// Merge two elements to get an element whose order is the LCM of their orders
///
/// Given elements a and b with orders m and n respectively, returns an element
/// with order lcm(m, n).
///
/// # Arguments
/// * `a` - First element
/// * `b` - Second element
/// * `order_a` - Order of element a
/// * `order_b` - Order of element b
/// * `mul` - Multiplication operation
/// * `identity` - The identity element
///
/// # Returns
/// An element with order lcm(order_a, order_b)
pub fn merge_elements<T: Clone + Eq>(
    a: T,
    b: T,
    order_a: usize,
    order_b: usize,
    mul: fn(T, T) -> T,
    identity: T,
) -> T {
    let target_order = lcm(order_a, order_b);

    // Compute a^(order_a / gcd(order_a, order_b))
    let g = gcd(order_a, order_b);
    let exp_a = order_a / g;
    let exp_b = order_b / g;

    let a_part = power(a, exp_a, mul, identity.clone());
    let b_part = power(b, exp_b, mul, identity.clone());

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

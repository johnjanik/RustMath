//! Prime number algorithms

use crate::Integer;
use rustmath_core::{NumericConversion, Ring};

/// Check if a number is prime using trial division (for small numbers)
pub fn is_prime_trial_division(n: &Integer) -> bool {
    if *n <= Integer::one() {
        return false;
    }
    if *n <= Integer::from(3) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    let mut i = Integer::from(3);
    let sqrt_n = isqrt(n);

    while i <= sqrt_n {
        if (n.clone() % i.clone()).is_zero() {
            return false;
        }
        i = i + Integer::from(2);
    }

    true
}

/// Miller-Rabin primality test
pub fn is_prime_miller_rabin(n: &Integer, rounds: u32) -> bool {
    if *n <= Integer::one() {
        return false;
    }
    if *n <= Integer::from(3) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Write n-1 as d * 2^r
    let mut d = n.clone() - Integer::one();
    let mut r = 0u32;

    while d.is_even() {
        d = d / Integer::from(2);
        r += 1;
    }

    // Witness loop
    let n_minus_one = n.clone() - Integer::one();

    for _ in 0..rounds {
        // In a real implementation, we'd use random witnesses
        // For now, use deterministic witnesses
        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        for &a_val in &witnesses {
            let a = Integer::from(a_val);
            if a >= *n {
                continue;
            }

            let mut x = a.mod_pow(&d, n).unwrap();

            if x.is_one() || x == n_minus_one {
                continue;
            }

            let mut is_composite = true;
            for _ in 0..r - 1 {
                x = x.mod_pow(&Integer::from(2), n).unwrap();
                if x == n_minus_one {
                    is_composite = false;
                    break;
                }
            }

            if is_composite {
                return false;
            }
        }
    }

    true
}

/// Check if a number is prime (combines methods)
pub fn is_prime(n: &Integer) -> bool {
    // Use trial division for small numbers
    if let Some(n_small) = n.to_u64() {
        if n_small < 1000 {
            return is_prime_trial_division(n);
        }
    }

    // Use Miller-Rabin for larger numbers
    is_prime_miller_rabin(n, 20)
}

/// Generate the next prime after n
pub fn next_prime(n: &Integer) -> Integer {
    let mut candidate = if n.is_even() {
        n.clone() + Integer::one()
    } else {
        n.clone() + Integer::from(2)
    };

    while !is_prime(&candidate) {
        candidate = candidate + Integer::from(2);
    }

    candidate
}

/// Generate the previous prime before n
///
/// Returns None if n <= 2
pub fn previous_prime(n: &Integer) -> Option<Integer> {
    if *n <= Integer::from(2) {
        return None;
    }

    if *n <= Integer::from(3) {
        return Some(Integer::from(2));
    }

    let mut candidate = if n.is_even() {
        n.clone() - Integer::one()
    } else {
        n.clone() - Integer::from(2)
    };

    while candidate > Integer::from(2) {
        if is_prime(&candidate) {
            return Some(candidate);
        }
        candidate = candidate - Integer::from(2);
    }

    Some(Integer::from(2))
}

/// Integer square root (floor)
fn isqrt(n: &Integer) -> Integer {
    if n.is_zero() || n.is_one() {
        return n.clone();
    }

    // Newton's method
    let mut x = n.clone();
    let mut y = (n.clone() + Integer::one()) / Integer::from(2);

    while y < x {
        x = y.clone();
        y = (x.clone() + n.clone() / x.clone()) / Integer::from(2);
    }

    x
}

/// Compute all prime factors of n
pub fn factor(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle factor of 2
    if remaining.is_even() {
        let mut count = 0;
        while remaining.is_even() {
            remaining = remaining / Integer::from(2);
            count += 1;
        }
        factors.push((Integer::from(2), count));
    }

    // Try odd factors
    let mut factor = Integer::from(3);
    while factor.clone() * factor.clone() <= remaining {
        let mut count = 0;
        while (&remaining % &factor).is_zero() {
            remaining = remaining / factor.clone();
            count += 1;
        }
        if count > 0 {
            factors.push((factor.clone(), count));
        }
        factor = factor + Integer::from(2);
    }

    // If remaining > 1, it's a prime factor
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors
}

/// Pollard's Rho algorithm for finding a non-trivial factor
///
/// This is a probabilistic algorithm that's much faster than trial division
/// for large composite numbers with small factors.
///
/// Returns None if n is prime or if the algorithm fails to find a factor
pub fn pollard_rho(n: &Integer) -> Option<Integer> {
    if n <= &Integer::from(1) {
        return None;
    }

    if n.is_even() {
        return Some(Integer::from(2));
    }

    // Check if n is prime first (for small n)
    if is_prime(n) {
        return None;
    }

    // Pollard's Rho with Floyd's cycle detection
    let mut x = Integer::from(2);
    let mut y = Integer::from(2);

    // f(x) = (x² + 1) mod n
    let f = |val: &Integer| -> Integer {
        let result = (val.clone() * val.clone() + Integer::one()) % n.clone();
        if result.signum() < 0 {
            result + n.clone()
        } else {
            result
        }
    };

    // Try to find a factor
    for _ in 0..1000000 {
        x = f(&x);
        y = f(&f(&y));

        let diff = if x > y {
            x.clone() - y.clone()
        } else {
            y.clone() - x.clone()
        };

        let d = diff.gcd(n);

        if d > Integer::one() && &d < n {
            return Some(d);
        }

        if d == *n {
            // Cycle detected, restart with different starting point
            x = Integer::from(rand_int() % 100 + 2);
            y = x.clone();
        }
    }

    None
}

// Simple deterministic "random" number for Pollard's Rho
fn rand_int() -> i64 {
    use std::cell::Cell;
    thread_local!(static COUNTER: Cell<i64> = Cell::new(42));
    COUNTER.with(|c| {
        let val = c.get();
        c.set(val.wrapping_mul(1103515245).wrapping_add(12345));
        (val / 65536) % 32768
    })
}

/// Factor using Pollard's Rho algorithm
///
/// More efficient than trial division for numbers with small factors
pub fn factor_pollard_rho(n: &Integer) -> Vec<(Integer, u32)> {
    if n.is_zero() || n.is_one() {
        return vec![];
    }

    let mut factors = Vec::new();
    let mut remaining = n.abs();

    // Handle factor of 2
    if remaining.is_even() {
        let mut count = 0;
        while remaining.is_even() {
            remaining = remaining / Integer::from(2);
            count += 1;
        }
        factors.push((Integer::from(2), count));
    }

    // Use Pollard's Rho for the rest
    while remaining > Integer::one() && !is_prime(&remaining) {
        if let Some(factor) = pollard_rho(&remaining) {
            let mut count = 0;
            while (&remaining % &factor).is_zero() {
                remaining = remaining / factor.clone();
                count += 1;
            }
            factors.push((factor, count));
        } else {
            // Fallback to trial division if Pollard's Rho fails
            let mut d = Integer::from(3);
            while d.clone() * d.clone() <= remaining {
                if (&remaining % &d).is_zero() {
                    let mut count = 0;
                    while (&remaining % &d).is_zero() {
                        remaining = remaining / d.clone();
                        count += 1;
                    }
                    factors.push((d.clone(), count));
                    break;
                }
                d = d + Integer::from(2);
            }

            if remaining > Integer::one() {
                break;
            }
        }
    }

    // If remaining > 1, it's a prime factor
    if remaining > Integer::one() {
        factors.push((remaining, 1));
    }

    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(is_prime(&Integer::from(2)));
        assert!(is_prime(&Integer::from(3)));
        assert!(is_prime(&Integer::from(5)));
        assert!(is_prime(&Integer::from(7)));
        assert!(is_prime(&Integer::from(11)));
        assert!(is_prime(&Integer::from(13)));

        assert!(!is_prime(&Integer::from(1)));
        assert!(!is_prime(&Integer::from(4)));
        assert!(!is_prime(&Integer::from(6)));
        assert!(!is_prime(&Integer::from(8)));
        assert!(!is_prime(&Integer::from(9)));
        assert!(!is_prime(&Integer::from(10)));
    }

    #[test]
    fn test_next_prime() {
        assert_eq!(next_prime(&Integer::from(2)), Integer::from(3));
        assert_eq!(next_prime(&Integer::from(7)), Integer::from(11));
        assert_eq!(next_prime(&Integer::from(20)), Integer::from(23));
    }

    #[test]
    fn test_previous_prime() {
        assert_eq!(previous_prime(&Integer::from(3)), Some(Integer::from(2)));
        assert_eq!(previous_prime(&Integer::from(10)), Some(Integer::from(7)));
        assert_eq!(previous_prime(&Integer::from(20)), Some(Integer::from(19)));
        assert_eq!(previous_prime(&Integer::from(100)), Some(Integer::from(97)));

        // Edge cases
        assert_eq!(previous_prime(&Integer::from(2)), None);
        assert_eq!(previous_prime(&Integer::from(1)), None);
    }

    #[test]
    fn test_factor() {
        let factors = factor(&Integer::from(60));
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0], (Integer::from(2), 2));
        assert_eq!(factors[1], (Integer::from(3), 1));
        assert_eq!(factors[2], (Integer::from(5), 1));

        let factors = factor(&Integer::from(17));
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0], (Integer::from(17), 1));
    }
}

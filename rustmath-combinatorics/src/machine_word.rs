//! Checked machine-word helpers for counts, ranks and loop bounds.
//!
//! MAGMA Handbook ch. 144 §144.2 (the counting functions) and §144.3 (subsets and their ranks),
//! digest `rustmath-lava/docs/Chapter_144_Enumerative_Combinatorics.md`. Several public
//! functions of this crate return a count or a rank as a `usize`. These helpers compute such a
//! value exactly when it fits, and report `None` (or panic with a stated message) when it does
//! not, instead of wrapping, saturating or truncating (design C1 §1.6).
//!
//! Every helper does at most about `usize::BITS` steps, whatever its arguments are.

/// base^exp, or None when it exceeds usize::MAX. Any exp for base 0 and 1.
pub(crate) fn checked_pow_usize(base: usize, exp: usize) -> Option<usize> {
    match base {
        0 => Some(if exp == 0 { 1 } else { 0 }),
        1 => Some(1),
        // base >= 2: an exponent of 2^32 or more overflows every usize.
        _ => u32::try_from(exp).ok().and_then(|e| base.checked_pow(e)),
    }
}

/// C(m, j), or None when it exceeds usize::MAX; 0 when j > m. m is a u128 so that n + k - 1 of
/// two usize values fits. At most min(j, m - j, usize::BITS + 1) steps.
///
/// With j <- min(j, m - j), c_i = c_(i-1) (m - j + i) / i = C(m - j + i, i) is an exact
/// division, and the sequence does not decrease, so once c_i exceeds usize::MAX the result does
/// too. Since m - j >= j >= i, c_i >= C(2i, i) >= 2^i, so the loop ends within usize::BITS + 1
/// steps. If c_(i-1) (m - j + i) overflows u128, then c_i >= 2^128 / i > usize::MAX.
pub(crate) fn binomial_usize_checked(m: u128, j: u128) -> Option<usize> {
    if j > m {
        return Some(0);
    }
    let j = j.min(m - j);
    let max = usize::MAX as u128;
    let mut c: u128 = 1;
    let mut i: u128 = 1;
    while i <= j {
        c = c.checked_mul(m - j + i)? / i;
        if c > max {
            return None;
        }
        i += 1;
    }
    usize::try_from(c).ok()
}

/// n (n-1) ... (n-k+1), or None when it exceeds usize::MAX. 0 when k > n.
///
/// Every factor except possibly the last is at least 2, so the product at least doubles per
/// step and the loop ends within usize::BITS + 1 steps.
pub(crate) fn falling_factorial_usize_checked(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let mut result: usize = 1;
    for i in 0..k {
        result = result.checked_mul(n - i)?;
    }
    Some(result)
}

/// x as a u32 argument.
///
/// # Panics
///
/// With "{what}: {x} exceeds u32::MAX" when x does not fit a u32.
pub(crate) fn u32_arg(x: usize, what: &str) -> u32 {
    match u32::try_from(x) {
        Ok(v) => v,
        Err(_) => panic!("{what}: {x} exceeds u32::MAX"),
    }
}

/// 2^n, the number of subsets (masks) of an n-element set, as a usize.
///
/// # Panics
///
/// With "{what}: 2^{n} subsets exceed usize::MAX" when n >= usize::BITS: the subsets cannot be
/// listed with a usize mask.
pub(crate) fn subset_mask_count(n: usize, what: &str) -> usize {
    match u32::try_from(n).ok().and_then(|s| 1usize.checked_shl(s)) {
        Some(c) => c,
        None => panic!("{what}: 2^{n} subsets exceed usize::MAX"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::NumericConversion;
    use rustmath_integers::{Integer, SageInteger};

    #[test]
    fn checked_pow_usize_values() {
        // gp: 2^63, 3^40 = 12157665459056928801, 17^15 = 2862423051509815793; 3^41 and 18^16
        // exceed 2^64 - 1.
        assert_eq!(checked_pow_usize(0, 0), Some(1));
        assert_eq!(checked_pow_usize(0, 1 << 40), Some(0));
        assert_eq!(checked_pow_usize(1, 1 << 40), Some(1));
        assert_eq!(checked_pow_usize(5, 0), Some(1));
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(checked_pow_usize(2, 63), Some(1 << 63));
            assert_eq!(checked_pow_usize(2, 64), None);
            assert_eq!(checked_pow_usize(3, 40), Some(12157665459056928801));
            assert_eq!(checked_pow_usize(3, 41), None);
            assert_eq!(checked_pow_usize(2, (1 << 32) + 1), None);
            assert_eq!(checked_pow_usize(17, 15), Some(2862423051509815793));
            assert_eq!(checked_pow_usize(18, 16), None);
        }
    }

    #[test]
    fn binomial_usize_checked_matches_exact_binomials() {
        for m in 0..=100u32 {
            for j in 0..=m + 2 {
                let exact = <Integer as SageInteger>::binomial(&Integer::from(m), &Integer::from(j));
                assert_eq!(
                    binomial_usize_checked(u128::from(m), u128::from(j)),
                    exact.to_usize(),
                    "C({m}, {j})"
                );
            }
        }
        // j > m (review 2 m1).
        assert_eq!(binomial_usize_checked(3, 5), Some(0));
        assert_eq!(binomial_usize_checked(0, 1), Some(0));
        #[cfg(target_pointer_width = "64")]
        {
            // gp binomial(67, 33) = 14226520737620288370; binomial(68, 34) > 2^64.
            assert_eq!(binomial_usize_checked(67, 33), Some(14226520737620288370));
            assert_eq!(binomial_usize_checked(68, 34), None);
            // gp binomial(2^32 + 2, 2) = 9223372043297226753.
            assert_eq!(binomial_usize_checked((1 << 32) + 2, 2), Some(9223372043297226753));
            let big = 2 * (usize::MAX as u128) - 1; // n + k - 1 at n = k = usize::MAX
            assert_eq!(binomial_usize_checked(big, usize::MAX as u128 - 1), None);
            assert_eq!(binomial_usize_checked(usize::MAX as u128, 1), Some(usize::MAX));
            assert_eq!(binomial_usize_checked(usize::MAX as u128 + 1, 1), None);
        }
    }

    #[test]
    fn falling_factorial_usize_checked_values() {
        // Moved from debruijn.rs (its private helpers are gone).
        assert_eq!(falling_factorial_usize_checked(5, 3), Some(60)); // 5 * 4 * 3
        assert_eq!(falling_factorial_usize_checked(10, 2), Some(90)); // 10 * 9
        assert_eq!(falling_factorial_usize_checked(5, 0), Some(1));
        assert_eq!(falling_factorial_usize_checked(5, 6), Some(0));
        assert_eq!(binomial_usize_checked(4, 2), Some(6));
        assert_eq!(binomial_usize_checked(5, 3), Some(10));
        assert_eq!(binomial_usize_checked(10, 0), Some(1));
        assert_eq!(binomial_usize_checked(10, 10), Some(1));
        assert_eq!(binomial_usize_checked(5, 6), Some(0));
        #[cfg(target_pointer_width = "64")]
        {
            // gp 20! = 2432902008176640000; 21! and 70! exceed 2^64.
            assert_eq!(falling_factorial_usize_checked(20, 20), Some(2432902008176640000));
            assert_eq!(falling_factorial_usize_checked(21, 21), None);
        }
        assert_eq!(falling_factorial_usize_checked(70, 70), None);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "exceeds u32::MAX")]
    fn u32_arg_refuses_2_pow_32() {
        u32_arg(1 << 32, "test");
    }

    #[test]
    fn u32_arg_passes_small_values() {
        assert_eq!(u32_arg(7, "test"), 7);
        assert_eq!(u32_arg(u32::MAX as usize, "test"), u32::MAX);
    }

    #[test]
    fn subset_mask_count_values() {
        assert_eq!(subset_mask_count(0, "test"), 1);
        assert_eq!(subset_mask_count(10, "test"), 1024);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(subset_mask_count(63, "test"), 1 << 63);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "test: 2^64 subsets exceed usize::MAX")]
    fn subset_mask_count_refuses_the_word_size() {
        subset_mask_count(64, "test");
    }
}

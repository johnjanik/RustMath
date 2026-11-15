//! Factorials, falling and rising factorials, and balanced products.
//!
//! MAGMA Handbook ch. 144 §144.2 `Factorial(n)` and `NumberOfPermutations(n, k)` (digest
//! `rustmath-lava/docs/Chapter_144_Enumerative_Combinatorics.md:36-37`), and the Pochhammer
//! symbol (a)_n = a (a+1) ... (a+n-1) of ch. 49 `HypergeometricSeries`
//! (`Chapter_49_Power_Laurent_and_Puiseux_Series.md:292`).
//!
//! - Falling and rising factorial powers: Graham, Knuth, Patashnik, *Concrete Mathematics*
//!   (2nd ed.), §2.6. Sage's `falling_factorial(x, a)` and `rising_factorial(x, a)`
//!   (sage.arith.misc, x any ring element, a >= 0) are the secondary reference.
//! - Balanced (product-tree) products: J. von zur Gathen, J. Gerhard, *Modern Computer
//!   Algebra* (3rd ed., 2013), §10.1. Consecutive word-sized factors are packed into one u64
//!   before the tree, so n! is about n / 4 leaves.
//!
//! | function | value |
//! |---|---|
//! | [`falling_factorial_integer`] | x (x-1) ... (x-k+1) for any integer x |
//! | [`rising_factorial_integer`] | x (x+1) ... (x+k-1) for any integer x (Pochhammer (x)_k) |
//! | [`falling_factorial_in`] | the falling product in any ring |
//! | [`rising_factorial_in`] | the rising product in any ring |

use num_bigint::BigInt;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;

/// The product of a list by pairwise products, level by level (vzGG §10.1). 1 for the empty
/// list.
pub(crate) fn product_tree(v: Vec<Integer>) -> Integer {
    product_tree_in(v).unwrap_or_else(Integer::one)
}

/// The balanced product of a list over any ring, or None for the empty list.
fn product_tree_in<R: Ring>(mut v: Vec<R>) -> Option<R> {
    while v.len() > 1 {
        let mut next = Vec::with_capacity(v.len().div_ceil(2));
        let mut it = v.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(a * b),
                None => next.push(a),
            }
        }
        v = next;
    }
    v.pop()
}

/// The product of word-sized factors: consecutive factors are packed into one u64 while the
/// product fits, then the leaves are multiplied by [`product_tree`]. 1 for no factors.
pub(crate) fn product_u64s(factors: impl Iterator<Item = u64>) -> Integer {
    let mut leaves = Vec::new();
    let mut acc: u64 = 1;
    for f in factors {
        match acc.checked_mul(f) {
            Some(x) => acc = x,
            None => {
                leaves.push(Integer::from(acc));
                acc = f;
            }
        }
    }
    leaves.push(Integer::from(acc));
    product_tree(leaves)
}

/// The product of u128 factors: factors below 2^64 are packed as in [`product_u64s`], a
/// larger factor becomes its own leaf. 1 for no factors.
pub(crate) fn product_u128s(factors: impl Iterator<Item = u128>) -> Integer {
    let mut leaves = Vec::new();
    let mut acc: u64 = 1;
    for f in factors {
        match u64::try_from(f) {
            Ok(f) => match acc.checked_mul(f) {
                Some(x) => acc = x,
                None => {
                    leaves.push(Integer::from(acc));
                    acc = f;
                }
            },
            Err(_) => leaves.push(Integer::new(BigInt::from(f))),
        }
    }
    leaves.push(Integer::from(acc));
    product_tree(leaves)
}

/// lo (lo+1) ... hi for 1 <= lo; 1 when lo > hi. Packs consecutive factors into one u64 while
/// the product fits.
pub(crate) fn product_range_u64(lo: u64, hi: u64) -> Integer {
    if lo > hi {
        return Integer::one();
    }
    product_u64s(lo..=hi)
}

/// a (a+1) ... (a+k-1) for an integer a >= 1: [`product_range_u64`] when a + k - 1 < 2^64,
/// else one leaf per factor. 1 when k = 0.
pub(crate) fn rising_run(a: &Integer, k: u64) -> Integer {
    debug_assert!(a.signum() > 0, "rising_run needs a >= 1");
    if k == 0 {
        return Integer::one();
    }
    if let Some(lo) = a.to_u64() {
        if let Some(hi) = lo.checked_add(k - 1) {
            return product_range_u64(lo, hi);
        }
    }
    let mut leaves = Vec::new();
    let mut t = a.clone();
    for _ in 0..k {
        let next = t.clone() + Integer::one();
        leaves.push(std::mem::replace(&mut t, next));
    }
    product_tree(leaves)
}

fn negate_if(v: Integer, odd: bool) -> Integer {
    if odd {
        -v
    } else {
        v
    }
}

/// x (x-1) ... (x-k+1) for any integer x; 1 when k = 0 (Sage `falling_factorial(x, k)`).
///
/// MAGMA `NumberOfPermutations(n, k)` (ch. 144 §144.2) for x = n >= 0. For x >= 0 the
/// product is 0 when k > x (the factor 0 occurs); for x < 0 it is (-1)^k (-x)(-x+1)...(-x+k-1).
/// Cost: one balanced product of k factors (O(M(B) log k) for a B-bit result).
pub fn falling_factorial_integer(x: &Integer, k: u64) -> Integer {
    if k == 0 {
        return Integer::one();
    }
    if x.signum() >= 0 {
        if Integer::from(k) > *x {
            return Integer::zero();
        }
        rising_run(&(x.clone() - Integer::from(k) + Integer::one()), k)
    } else {
        negate_if(rising_run(&(-x.clone()), k), k % 2 == 1)
    }
}

/// x (x+1) ... (x+k-1) for any integer x; 1 when k = 0. The Pochhammer symbol (x)_k of MAGMA
/// ch. 49 (`HypergeometricSeries`) and Sage `rising_factorial(x, k)`.
///
/// For x <= 0 the product is 0 when k > -x (the factor 0 occurs), and otherwise
/// (-1)^k (-x)(-x-1)...(-x-k+1). Cost: one balanced product of k factors.
pub fn rising_factorial_integer(x: &Integer, k: u64) -> Integer {
    if k == 0 {
        return Integer::one();
    }
    if x.signum() > 0 {
        return rising_run(x, k);
    }
    let m = -x.clone();
    if Integer::from(k) > m {
        return Integer::zero();
    }
    negate_if(falling_factorial_integer(&m, k), k % 2 == 1)
}

/// The factors x, x - one, x - 2 one, ... (`step = -1`) or x, x + one, ... (`step = +1`),
/// multiplied by a balanced product over R. A zero factor ends the product early and is
/// returned (0 times anything is 0).
fn product_of_steps_in<R: Ring>(x: &R, k: u64, one: &R, down: bool) -> R {
    if k == 0 {
        return one.clone();
    }
    let mut factors = Vec::new();
    let mut f = x.clone();
    for i in 0..k {
        if f.is_zero() {
            return f;
        }
        if i + 1 < k {
            let next = if down {
                f.clone() - one.clone()
            } else {
                f.clone() + one.clone()
            };
            factors.push(std::mem::replace(&mut f, next));
        } else {
            factors.push(f.clone());
        }
    }
    // The factors x + i one commute with each other, so the order of the product does not
    // matter even for a non-commutative R.
    product_tree_in(factors).unwrap_or_else(|| one.clone())
}

/// x (x - 1) ... (x - (k-1)) in any ring (Sage `falling_factorial(x, k)` for x a ring
/// element); `one.clone()` when k = 0.
///
/// `one` is the identity of the ring x belongs to. It is passed in because `R::one()` is an
/// unbound integer constant for the parent-carrying element types and may panic for a type
/// without such a sentinel (rustmath-core `Ring` docs). Factor i is x - i one, built by
/// repeated subtraction; a zero factor ends the product early.
pub fn falling_factorial_in<R: Ring>(x: &R, k: u64, one: &R) -> R {
    product_of_steps_in(x, k, one, true)
}

/// x (x + 1) ... (x + (k-1)) in any ring, the Pochhammer symbol (x)_k (Sage
/// `rising_factorial(x, k)`); `one.clone()` when k = 0. `one` is the identity of the ring x
/// belongs to (see [`falling_factorial_in`]). Factor i is x + i one; a zero factor ends the
/// product early.
pub fn rising_factorial_in<R: Ring>(x: &R, k: u64, one: &R) -> R {
    product_of_steps_in(x, k, one, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_product(x: i64, k: u64, down: bool) -> Integer {
        let mut r = Integer::one();
        for i in 0..k as i64 {
            r = r * Integer::from(if down { x - i } else { x + i });
        }
        r
    }

    #[test]
    fn product_range_u64_edges() {
        assert_eq!(product_range_u64(5, 4), Integer::one());
        assert_eq!(product_range_u64(1, 1), Integer::one());
        // 20! = 2432902008176640000 fits a u64, 21! does not (gp 21! = 51090942171709440000):
        // the range crosses the packing boundary.
        assert_eq!(product_range_u64(1, 20), Integer::from(2432902008176640000u64));
        assert_eq!(
            product_range_u64(1, 21),
            Integer::from_decimal_str("51090942171709440000").unwrap()
        );
        // lo near u64::MAX: gp (2^64-2)*(2^64-1) = 340282366920938463408034375210639556610.
        assert_eq!(
            product_range_u64(u64::MAX - 1, u64::MAX),
            Integer::from_decimal_str("340282366920938463408034375210639556610").unwrap()
        );
        assert_eq!(product_range_u64(u64::MAX, u64::MAX), Integer::from(u64::MAX));
        // A long range against the plain loop.
        let mut r = Integer::one();
        for i in 7..=300u64 {
            r = r * Integer::from(i);
        }
        assert_eq!(product_range_u64(7, 300), r);
    }

    #[test]
    fn product_u64s_and_u128s() {
        assert_eq!(product_u64s(std::iter::empty()), Integer::one());
        assert_eq!(product_u64s([3u64, 0, 5].into_iter()), Integer::zero());
        let f = [u64::MAX, u64::MAX, 2, 3];
        let mut r = Integer::one();
        for &x in &f {
            r = r * Integer::from(x);
        }
        assert_eq!(product_u64s(f.into_iter()), r);
        let big = (1u128 << 64) + 5;
        let g = [7u128, big, 11, big];
        let mut r = Integer::one();
        for &x in &g {
            r = r * Integer::new(BigInt::from(x));
        }
        assert_eq!(product_u128s(g.into_iter()), r);
    }

    #[test]
    fn rising_run_with_big_start() {
        // a = 2^64 - 2: the factors leave u64, so each one becomes a leaf.
        let a = Integer::from(u64::MAX - 1);
        let mut r = Integer::one();
        let mut t = a.clone();
        for _ in 0..5 {
            r = r * t.clone();
            t = t + Integer::one();
        }
        assert_eq!(rising_run(&a, 5), r);
        assert_eq!(rising_run(&a, 0), Integer::one());
        assert_eq!(rising_run(&a, 1), a);
    }

    #[test]
    fn falling_and_rising_match_the_direct_product() {
        for x in -30i64..=30 {
            for k in 0..=40u64 {
                let xi = Integer::from(x);
                assert_eq!(falling_factorial_integer(&xi, k), direct_product(x, k, true), "falling({x}, {k})");
                assert_eq!(rising_factorial_integer(&xi, k), direct_product(x, k, false), "rising({x}, {k})");
            }
        }
    }

    #[test]
    fn generic_versions_match_the_integer_versions() {
        let one = Integer::one();
        for x in -12i64..=12 {
            for k in 0..=15u64 {
                let xi = Integer::from(x);
                assert_eq!(falling_factorial_in(&xi, k, &one), falling_factorial_integer(&xi, k));
                assert_eq!(rising_factorial_in(&xi, k, &one), rising_factorial_integer(&xi, k));
            }
        }
    }
}

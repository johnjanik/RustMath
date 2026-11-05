//! Exact comparison of algebraic numbers
//!
//! Implement exact equality and ordering for algebraic numbers.

use crate::algebraic_number::AlgebraicNumber;
use crate::algebraic_real::AlgebraicReal;
use std::cmp::Ordering;

/// Compare two algebraic numbers for exact equality
///
/// # Arguments
/// * `a` - First algebraic number
/// * `b` - Second algebraic number
///
/// # Returns
/// true if a and b represent the same algebraic number
///
/// # Caveat
/// See [`AlgebraicNumber`]'s `PartialEq` impl: this is conservative for
/// non-rational values (may return `false` for two values that are actually
/// equal), which is the opposite risk profile from
/// [`AlgebraicReal`]'s `sign`/`cmp`/`eq` (which can, in the worst case,
/// return `true` for two distinct-but-close values). `true` here can be
/// trusted; `false` cannot be taken as a certified proof of inequality for
/// non-rational values.
pub fn algebraic_eq(a: &AlgebraicNumber, b: &AlgebraicNumber) -> bool {
    a == b
}

/// Compare two algebraic real numbers
///
/// # Arguments
/// * `a` - First algebraic real
/// * `b` - Second algebraic real
///
/// # Returns
/// Ordering::Less if a < b, Ordering::Equal if a = b, Ordering::Greater if a > b
///
/// # Caveat
/// Thin wrapper around [`AlgebraicReal::cmp`], which is a **best-effort**
/// decision procedure, not a certified one: for a genuinely-undecidable-equal
/// pair of *distinct* values whose difference cannot be separated from zero
/// within its iteration budget, it reports `Ordering::Equal`. See
/// `AlgebraicReal::sign`/`Ord::cmp` for the full caveat.
pub fn algebraic_compare(a: &AlgebraicReal, b: &AlgebraicReal) -> Ordering {
    a.cmp(b)
}

/// Check if an algebraic real is positive
///
/// # Caveat
/// See [`AlgebraicReal::sign`]: for values indistinguishable from zero
/// within the refinement iteration budget, this is a best-effort `false`,
/// not a certified proof of non-positivity.
pub fn is_positive(a: &AlgebraicReal) -> bool {
    a.sign() > 0
}

/// Check if an algebraic real is negative
///
/// # Caveat
/// See [`AlgebraicReal::sign`]: for values indistinguishable from zero
/// within the refinement iteration budget, this is a best-effort `false`,
/// not a certified proof of non-negativity.
pub fn is_negative(a: &AlgebraicReal) -> bool {
    a.sign() < 0
}

/// Check if an algebraic real is zero
///
/// # Caveat
/// Unlike `sign()`, this delegates to `AlgebraicReal::is_zero`, which only
/// recognizes zero when the value simplifies to the exact rational `0` (it
/// does *not* run interval refinement). It is therefore conservative in the
/// opposite direction from the `sign()`/`cmp()` best-effort-zero caveat: it
/// can return `false` for a composite expression that `sign()` would
/// (correctly, or via the best-effort fallback) report as zero. It never
/// returns `true` for a nonzero value.
pub fn is_zero(a: &AlgebraicReal) -> bool {
    a.is_zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_equality() {
        let a = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let b = AlgebraicNumber::from_rational(Rational::new(3, 2).unwrap());
        let c = AlgebraicNumber::from_rational(Rational::new(5, 2).unwrap());

        assert!(algebraic_eq(&a, &b));
        assert!(!algebraic_eq(&a, &c));
    }

    #[test]
    fn test_comparison() {
        let a = AlgebraicReal::from_i64(3);
        let b = AlgebraicReal::from_i64(5);

        assert_eq!(algebraic_compare(&a, &b), Ordering::Less);
        assert_eq!(algebraic_compare(&b, &a), Ordering::Greater);
        assert_eq!(algebraic_compare(&a, &a), Ordering::Equal);
    }

    #[test]
    fn test_sign_tests() {
        let positive = AlgebraicReal::from_i64(5);
        let negative = AlgebraicReal::from_i64(-3);
        let zero = AlgebraicReal::from_i64(0);

        assert!(is_positive(&positive));
        assert!(!is_positive(&negative));
        assert!(!is_positive(&zero));

        assert!(is_negative(&negative));
        assert!(!is_negative(&positive));
        assert!(!is_negative(&zero));

        assert!(is_zero(&zero));
        assert!(!is_zero(&positive));
        assert!(!is_zero(&negative));
    }
}

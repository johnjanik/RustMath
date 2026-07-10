//! Ordered algebraic structures.
//!
//! MAGMA source: Handbook chapters 25 (real/complex fields), 143 (exact
//! polytopes), 159 (linear programming) — all need a comparison-aware ring/field
//! that the base [`Ring`]/[`Field`] tower deliberately omits (it only requires
//! [`PartialEq`]).
//!
//! This module is purely additive: it defines [`OrderedRing`] and
//! [`OrderedField`] on top of the existing trait tower and implements them for
//! the primitive integer types `i32`/`i64` (mirroring the `Ring` impls that
//! already live in [`crate::traits`]).

use crate::{Field, Ring};

/// A ring equipped with an order compatible with the ring operations.
///
/// The order is exposed through [`PartialOrd`] (so `<`, `<=`, … work directly)
/// together with a fast [`sign`](OrderedRing::sign) and
/// [`abs`](OrderedRing::abs). Implementors must guarantee that the order is
/// translation-invariant (`a <= b` ⇒ `a + c <= b + c`) and that products of
/// non-negative elements are non-negative.
pub trait OrderedRing: Ring + PartialOrd {
    /// The sign of the element: `-1`, `0`, or `+1`.
    fn sign(&self) -> i32;

    /// The absolute value `|self|`.
    fn abs(&self) -> Self;

    /// Whether the element is strictly greater than zero.
    fn is_positive(&self) -> bool {
        self.sign() > 0
    }

    /// Whether the element is strictly less than zero.
    fn is_negative(&self) -> bool {
        self.sign() < 0
    }

    /// The larger of `self` and `other`.
    fn max_with(&self, other: &Self) -> Self {
        if self >= other {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// The smaller of `self` and `other`.
    fn min_with(&self, other: &Self) -> Self {
        if self <= other {
            self.clone()
        } else {
            other.clone()
        }
    }
}

/// An ordered field: a [`Field`] whose order makes it an [`OrderedRing`].
///
/// Examples are the rationals and the (arbitrary-precision) reals. Finite fields
/// and the complex numbers are **not** ordered fields.
pub trait OrderedField: Field + OrderedRing {}

macro_rules! impl_ordered_primitive {
    ($($t:ty),*) => {
        $(
            impl OrderedRing for $t {
                fn sign(&self) -> i32 {
                    match (*self).cmp(&0) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    }
                }

                fn abs(&self) -> Self {
                    // `i32::MIN`/`i64::MIN` have no positive counterpart; wrap in
                    // that pathological case rather than panic, matching the
                    // existing `EuclideanDomain::norm` behaviour for these types.
                    (*self).wrapping_abs()
                }
            }
        )*
    };
}

impl_ordered_primitive!(i32, i64);

// `i32`/`i64` are not `Field`s, so `OrderedField` is intentionally not
// implemented for them here; ordered-field impls live with the field types
// (`Rational`, `BigFloat`, …) in their own crates.
#[allow(dead_code)]
fn _assert_ordered_field<F: OrderedField>() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign() {
        assert_eq!(OrderedRing::sign(&5i32), 1);
        assert_eq!(OrderedRing::sign(&0i32), 0);
        assert_eq!(OrderedRing::sign(&(-7i64)), -1);
    }

    #[test]
    fn test_abs() {
        assert_eq!(OrderedRing::abs(&(-7i32)), 7);
        assert_eq!(OrderedRing::abs(&7i64), 7);
    }

    #[test]
    fn test_min_max_positive_negative() {
        assert_eq!((3i32).max_with(&8), 8);
        assert_eq!((3i32).min_with(&8), 3);
        assert!((5i64).is_positive());
        assert!((-5i64).is_negative());
        assert!(!(0i64).is_positive());
    }

    #[test]
    fn test_order_is_translation_invariant_on_i64() {
        let (a, b, c) = (3i64, 8i64, 100i64);
        assert!(a < b);
        assert!(a + c < b + c);
    }
}

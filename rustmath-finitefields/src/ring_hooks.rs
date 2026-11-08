//! Shared pieces of the three-valued `Ring` element predicates
//! (`unit_status`, `nilpotent_status`, `unit_inverse`, `multiplicative_order_status`)
//! for the finite-ring element types of this crate.
//!
//! MAGMA source: Handbook Ch. 17 §17.5.5 (`IsUnit`, `IsNilpotent`), Ch. 19 (`Order`,
//! `IsUnit` in Z/mZ), Ch. 21 (`Order`, `IsUnit` in GF(q)), Ch. 48 (`IsUnit`,
//! `IsNilpotent` in Galois rings).
//!
//! # Deferred integer constants (the sentinels)
//!
//! `IntegerMod`, `PrimeField` (modulus 0) and `FiniteFieldElement` (unbound) all document
//! their static `Ring::zero()`/`Ring::one()` values, and everything built from them, as an
//! integer constant `c` awaiting the canonical map `Z -> R` into the ring of whatever bound
//! operand it meets. Generic code reaches these hooks with exactly such values
//! (`R::one() + R::one()`), so a hook may only state what holds for the image of `c` in
//! **every** target ring (all of them nonzero):
//!
//! | `c` | unit | nilpotent | order | `unit_inverse` |
//! |---|---|---|---|---|
//! | `±1` | `Some(true)` | `Some(false)` | `1`: `Finite(1)`; `-1`: `Undecided` (it is 1 in characteristic 2) | `Ok(self)` |
//! | `0` | `Some(false)` | `Some(true)` | `NotAUnit` | `Err(NotInvertible)` |
//! | other | `None` | `None` | `Undecided` | `Err(InvalidArgument)` |
//!
//! PARI/GP for the "other" row: `[gcd(2,5), gcd(2,4), Mod(2,4)^2, Mod(-1,2)==1]` gives
//! `[1, 2, 0, 1]` (2 is a unit mod 5, a nilpotent mod 4; -1 = 1 mod 2).

use rustmath_core::{MathError, MultiplicativeOrder, NumericConversion, Result};
use rustmath_integers::Integer;

fn is_plus_minus_one(c: &Integer) -> bool {
    c.is_one() || (-c.clone()).is_one()
}

/// `IsUnit` of a deferred integer constant `c` (see the module docs).
pub(crate) fn sentinel_unit_status(c: &Integer) -> Option<bool> {
    if is_plus_minus_one(c) {
        Some(true)
    } else if c.is_zero() {
        Some(false)
    } else {
        None
    }
}

/// `IsNilpotent` of a deferred integer constant `c` (see the module docs).
pub(crate) fn sentinel_nilpotent_status(c: &Integer) -> Option<bool> {
    if c.is_zero() {
        Some(true)
    } else if is_plus_minus_one(c) {
        Some(false)
    } else {
        None
    }
}

/// `Order` of a deferred integer constant `c` (see the module docs).
pub(crate) fn sentinel_order(c: &Integer) -> MultiplicativeOrder {
    if c.is_one() {
        MultiplicativeOrder::Finite(1)
    } else if c.is_zero() {
        MultiplicativeOrder::NotAUnit
    } else {
        MultiplicativeOrder::Undecided
    }
}

/// Inverse of a deferred integer constant `c`, where `this` is the element holding it.
pub(crate) fn sentinel_unit_inverse<T: Clone>(c: &Integer, this: &T) -> Result<T> {
    if is_plus_minus_one(c) {
        Ok(this.clone())
    } else if c.is_zero() {
        Err(MathError::NotInvertible)
    } else {
        Err(MathError::InvalidArgument(format!(
            "whether the integer constant {c} is invertible depends on the ring it binds to; \
             bind it to a modulus or field first"
        )))
    }
}

/// Map an exact `Integer` order to [`MultiplicativeOrder`].
pub(crate) fn order_from_integer(k: &Integer) -> MultiplicativeOrder {
    match k.to_u64() {
        Some(v) => MultiplicativeOrder::Finite(v),
        None => MultiplicativeOrder::FiniteExceedsU64,
    }
}

/// Whether `value` is nilpotent in Z/`modulus` (`modulus > 1`): every prime dividing the
/// modulus divides the value (Z/m is a product of the local rings Z/p^e).
pub(crate) fn is_nilpotent_mod(value: &Integer, modulus: &Integer) -> bool {
    if value.is_zero() {
        return true;
    }
    rustmath_integers::prime::factor(modulus)
        .into_iter()
        .all(|(prime, _)| (value.clone() % prime).is_zero())
}

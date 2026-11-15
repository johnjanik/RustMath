//! `Ring` element predicates of `Integer` (MAGMA Handbook Ch. 17 §17.5.5 `IsUnit`,
//! `IsNilpotent`; Ch. 19 `Order`). Z is a domain with unit group {±1}.
//!
//! GAP: `IsUnit(Integers, 2)` -> false, `IsUnit(Integers, -1)` -> true, `Order(-1)` -> 2.

use rustmath_core::{MathError, MultiplicativeOrder, Ring};
use rustmath_integers::Integer;

#[test]
fn integer_units_nilpotents_orders() {
    let two = Integer::from(2);
    assert_eq!(two.unit_status(), Some(false));
    assert_eq!(two.nilpotent_status(), Some(false));
    assert_eq!(two.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(two.unit_inverse(), Err(MathError::NotInvertible));

    let minus_one = Integer::from(-1);
    assert_eq!(minus_one.unit_status(), Some(true));
    assert_eq!(minus_one.nilpotent_status(), Some(false));
    assert_eq!(minus_one.multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(minus_one.unit_inverse(), Ok(Integer::from(-1)));

    let one = Integer::from(1);
    assert_eq!(one.multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    assert_eq!(one.unit_inverse(), Ok(Integer::from(1)));

    let zero = Integer::from(0);
    assert_eq!(zero.unit_status(), Some(false));
    assert_eq!(zero.nilpotent_status(), Some(true));
    assert_eq!(zero.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));

    // A large non-unit.
    let big = Integer::from(2).pow(100u32) + Integer::from(1);
    assert_eq!(big.unit_status(), Some(false));
    assert_eq!(<Integer as Ring>::static_characteristic(), Some(0));
    // MAGMA IsCommutative(Integers()) and IsCommutative(Rationals()) are true.
    assert_eq!(<Integer as Ring>::static_is_commutative(), Some(true));
}

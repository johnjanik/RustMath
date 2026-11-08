//! `Ring` element predicates of `Rational` (MAGMA Handbook Ch. 20 `IsUnit`; Ch. 17 §17.5.5
//! `IsNilpotent`; Ch. 19 `Order`). Q is a field of characteristic 0 whose roots of unity
//! are ±1.
//!
//! References: GAP `IsUnit(Rationals, 2/3)` -> true, `Inverse(2/3)` -> 3/2,
//! `Order(2/3)` -> infinity, `Order(-1)` -> 2; PARI `1/(2/3)` -> 3/2.

use rustmath_core::{MathError, MultiplicativeOrder, Ring};
use rustmath_rationals::Rational;

fn q(n: i64, d: i64) -> Rational {
    Rational::new(n, d).unwrap()
}

#[test]
fn rational_units_nilpotents_orders() {
    let x = q(2, 3);
    assert_eq!(x.unit_status(), Some(true));
    assert_eq!(x.nilpotent_status(), Some(false));
    assert_eq!(x.unit_inverse(), Ok(q(3, 2)));
    assert_eq!(x.multiplicative_order_status(), MultiplicativeOrder::Infinite);

    assert_eq!(q(-1, 1).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(q(-1, 1).unit_inverse(), Ok(q(-1, 1)));
    assert_eq!(q(1, 1).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    assert_eq!(q(-2, 1).multiplicative_order_status(), MultiplicativeOrder::Infinite);

    let zero = q(0, 1);
    assert_eq!(zero.unit_status(), Some(false));
    assert_eq!(zero.nilpotent_status(), Some(true));
    assert_eq!(zero.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));

    assert_eq!(<Rational as Ring>::static_characteristic(), Some(0));
}

// `Rational::reciprocal` (hence `Field::inverse`) of a negative rational used to return a
// negative denominator, e.g. 1/(-1), which compared unequal to -1 and was not `is_one`
// after negation. PARI: [1/(-1), 1/(-2/3)] -> [-1, -3/2].
#[test]
fn reciprocal_of_negative_is_canonical() {
    assert_eq!(q(-1, 1).reciprocal(), Ok(q(-1, 1)));
    assert_eq!(q(-2, 3).reciprocal(), Ok(q(-3, 2)));
    assert!((-q(-1, 1).reciprocal().unwrap()).is_one());
    assert_eq!(q(-2, 3).unit_inverse(), Ok(q(-3, 2)));
}

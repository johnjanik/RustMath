//! `Ring` element predicates and parent finiteness of the real types (MAGMA Handbook
//! Ch. 25: `IsUnit` is true for the nonzero elements of a real field, `IsFinite` is false;
//! Ch. 17 §17.5.5 `IsNilpotent`; Ch. 19 `Order`). The only real roots of unity are ±1.
//! GAP: `Order(-1)` -> 2.

use rustmath_core::{MathError, MultiplicativeOrder, Parent, Ring};
use rustmath_reals::{BigFloat, BigFloatField, Real, RealMPFR};

#[test]
fn real_mpfr_predicates() {
    let two = RealMPFR::with_val(53, 2.0);
    assert_eq!(two.unit_status(), Some(true));
    assert_eq!(two.nilpotent_status(), Some(false));
    assert_eq!(two.unit_inverse().unwrap(), RealMPFR::with_val(53, 0.5));
    assert_eq!(two.multiplicative_order_status(), MultiplicativeOrder::Infinite);

    let minus_one = RealMPFR::with_val(53, -1.0);
    assert_eq!(minus_one.multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(RealMPFR::with_val(53, 1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    // Close to 1 but not 1: infinite order.
    let near_one = RealMPFR::with_val(53, 1.0 + f64::EPSILON);
    assert_eq!(near_one.multiplicative_order_status(), MultiplicativeOrder::Infinite);

    let zero = RealMPFR::with_val(53, 0.0);
    assert_eq!(zero.unit_status(), Some(false));
    assert_eq!(zero.nilpotent_status(), Some(true));
    assert_eq!(zero.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let x = RealMPFR::with_val(53, bad);
        assert_eq!(x.unit_status(), None);
        assert_eq!(x.nilpotent_status(), None);
        assert_eq!(x.multiplicative_order_status(), MultiplicativeOrder::Undecided);
        assert!(matches!(x.unit_inverse(), Err(MathError::DomainError(_))));
    }
    assert_eq!(<RealMPFR as Ring>::static_characteristic(), Some(0));
}

#[test]
fn bigfloat_predicates() {
    let two = BigFloat::from(2.0);
    assert_eq!(two.unit_status(), Some(true));
    assert_eq!(two.nilpotent_status(), Some(false));
    assert_eq!(two.unit_inverse().unwrap(), BigFloat::from(0.5));
    assert_eq!(two.multiplicative_order_status(), MultiplicativeOrder::Infinite);
    assert_eq!(BigFloat::from(-1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(BigFloat::from(1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    let zero = BigFloat::from(0.0);
    assert_eq!(zero.unit_status(), Some(false));
    assert_eq!(zero.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));
}

#[test]
fn real_f64_predicates() {
    let two = Real::from(2.0);
    assert_eq!(two.unit_status(), Some(true));
    assert_eq!(two.unit_inverse().unwrap(), Real::from(0.5));
    // Tolerance-based type: no Infinite claim.
    assert_eq!(two.multiplicative_order_status(), MultiplicativeOrder::Undecided);
    assert_eq!(Real::from(-1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    let zero = Real::from(0.0);
    assert_eq!(zero.unit_status(), Some(false));
    assert_eq!(zero.nilpotent_status(), Some(true));
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));
    let nan = Real::from(f64::NAN);
    assert_eq!(nan.unit_status(), None);
    assert_eq!(nan.multiplicative_order_status(), MultiplicativeOrder::Undecided);
    assert!(matches!(nan.unit_inverse(), Err(MathError::DomainError(_))));
}

// MAGMA Ch. 25: IsFinite is false for real fields.
#[test]
fn bigfloat_field_is_infinite() {
    let r = BigFloatField::new(128);
    assert_eq!(r.cardinality(), None);
    assert_eq!(r.is_finite(), Some(false));
}

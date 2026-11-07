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

// The f64 `Real` has a 1e-10 tolerance in `is_zero`/`is_one`/`==`. A nonzero double inside
// it is a unit of R (it is a nonzero real number), but the type cannot tell it from 0, so the
// hooks must not claim "non-unit"/"nilpotent" (review follow-up C1.2). Likewise a double
// within the tolerance of ±1 but not ±1 is not provably of order 1 or 2 (the real roots of
// unity are exactly ±1).
#[test]
fn real_f64_tolerance_band_is_undecided() {
    for v in [1e-11, -1e-11, 5e-324, -9.99e-11] {
        let x = Real::from(v);
        assert!(x.is_zero(), "{v} is inside the tolerance");
        assert_eq!(x.unit_status(), None, "{v}");
        assert_eq!(x.nilpotent_status(), None, "{v}");
        assert_eq!(x.multiplicative_order_status(), MultiplicativeOrder::Undecided, "{v}");
        assert!(matches!(x.unit_inverse(), Err(MathError::NumericalError(_))), "{v}");
    }
    // Exactly zero (either sign) stays decided.
    for v in [0.0, -0.0] {
        let x = Real::from(v);
        assert_eq!(x.unit_status(), Some(false));
        assert_eq!(x.nilpotent_status(), Some(true));
        assert_eq!(x.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
        assert_eq!(x.unit_inverse(), Err(MathError::NotInvertible));
    }
    // At or beyond the tolerance: a nonzero real, decided.
    for v in [1e-10, -1e-10, 2.5e-10, -3.0, 1e300] {
        let x = Real::from(v);
        assert_eq!(x.unit_status(), Some(true), "{v}");
        assert_eq!(x.nilpotent_status(), Some(false), "{v}");
        assert_eq!(x.unit_inverse().unwrap().to_f64(), 1.0 / v, "{v}");
    }
    // Near ±1 but not ±1: undecided (the trait default would trust `is_one`).
    for v in [1.0 + 5e-11, 1.0 - 5e-11, -1.0 + 5e-11, -1.0 - 5e-11] {
        let x = Real::from(v);
        assert!(v != 1.0 && v != -1.0);
        assert_eq!(x.multiplicative_order_status(), MultiplicativeOrder::Undecided, "{v}");
        assert_eq!(x.unit_status(), Some(true), "{v}");
    }
    assert_eq!(Real::from(1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    assert_eq!(Real::from(-1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    for v in [f64::INFINITY, f64::NEG_INFINITY] {
        let x = Real::from(v);
        assert_eq!(x.unit_status(), None);
        assert_eq!(x.nilpotent_status(), None);
        assert!(matches!(x.unit_inverse(), Err(MathError::DomainError(_))));
    }
}

// MAGMA Ch. 25: IsFinite is false for real fields.
#[test]
fn bigfloat_field_is_infinite() {
    let r = BigFloatField::new(128);
    assert_eq!(r.cardinality(), None);
    assert_eq!(r.is_finite(), Some(false));
}

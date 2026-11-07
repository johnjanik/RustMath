//! `Ring` element predicates of the complex types (MAGMA Handbook Ch. 25: `IsUnit` is true
//! for the nonzero elements of a complex field; Ch. 17 §17.5.5 `IsNilpotent`; Ch. 19
//! `Order`).
//!
//! Exact values (MPC, BigComplex) lie in Q(i), whose roots of unity are mu_4. GAP:
//! `Order(E(4))` -> 4; `Filtered([1..50], n -> Phi(n) <= 2)` -> [ 1, 2, 3, 4, 6 ];
//! `E(4) in GaussianRationals` -> true; `E(3) in GaussianRationals` -> false.

use rustmath_complex::bigcomplex::BigComplex;
use rustmath_complex::{Complex, ComplexMPFR};
use rustmath_core::{MathError, MultiplicativeOrder, Ring};
use rustmath_reals::BigFloat;

fn mpc(re: f64, im: f64) -> ComplexMPFR {
    ComplexMPFR::with_val(53, (re, im))
}

#[test]
fn complex_mpfr_predicates() {
    let two_i = mpc(0.0, 2.0);
    assert_eq!(two_i.unit_status(), Some(true));
    assert_eq!(two_i.nilpotent_status(), Some(false));
    assert_eq!(two_i.unit_inverse().unwrap(), mpc(0.0, -0.5));
    assert_eq!(two_i.multiplicative_order_status(), MultiplicativeOrder::Infinite);

    assert_eq!(mpc(0.0, 1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    assert_eq!(mpc(0.0, -1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    assert_eq!(mpc(-1.0, 0.0).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(mpc(1.0, 0.0).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    assert_eq!(mpc(2.0, 0.0).multiplicative_order_status(), MultiplicativeOrder::Infinite);
    // Near the unit circle, but the exact binary value is in Q(i) \ mu_4.
    assert_eq!(mpc(0.6, 0.8).multiplicative_order_status(), MultiplicativeOrder::Infinite);

    let zero = mpc(0.0, 0.0);
    assert_eq!(zero.unit_status(), Some(false));
    assert_eq!(zero.nilpotent_status(), Some(true));
    assert_eq!(zero.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));

    let nan = mpc(f64::NAN, 0.0);
    assert_eq!(nan.unit_status(), None);
    assert_eq!(nan.multiplicative_order_status(), MultiplicativeOrder::Undecided);
    assert!(matches!(nan.unit_inverse(), Err(MathError::DomainError(_))));
    assert_eq!(<ComplexMPFR as Ring>::static_characteristic(), Some(0));
}

#[test]
fn bigcomplex_predicates() {
    let c = |re: f64, im: f64| BigComplex::new(BigFloat::from(re), BigFloat::from(im));
    assert_eq!(c(0.0, 2.0).unit_status(), Some(true));
    assert_eq!(c(0.0, 1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    assert_eq!(c(0.0, -1.0).multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    assert_eq!(c(-1.0, 0.0).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(c(1.0, 0.0).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
    assert_eq!(c(0.6, 0.8).multiplicative_order_status(), MultiplicativeOrder::Infinite);
    assert_eq!(c(0.0, 0.0).multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
    assert_eq!(c(0.0, 0.0).unit_inverse(), Err(MathError::NotInvertible));
    let inv = c(0.0, 2.0).unit_inverse().unwrap();
    assert!((inv * c(0.0, 2.0)).is_one());
}

#[test]
fn complex_f64_predicates() {
    let two_i = Complex::new(0.0, 2.0);
    assert_eq!(two_i.unit_status(), Some(true));
    assert_eq!(two_i.nilpotent_status(), Some(false));
    // Tolerance-based parts: no Infinite claim.
    assert_eq!(two_i.multiplicative_order_status(), MultiplicativeOrder::Undecided);
    assert_eq!(Complex::new(0.0, 0.0).unit_inverse(), Err(MathError::NotInvertible));
    assert_eq!(Complex::new(f64::NAN, 0.0).unit_status(), None);
}

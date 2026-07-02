//! U1 verification: exercises the (now-compiling) genus-≥2 hyperelliptic/Jacobian
//! stack over ℚ. Requires the `genus2` feature; compiles against the library, so
//! it is independent of the still-broken in-crate test modules (U2).
//!
//! Run: `cargo test -p rustmath-curves --features genus2 --test genus2_jacobian`

#![cfg(feature = "genus2")]

use rustmath_core::{Field, Ring};
use rustmath_curves::hyperelliptic::HyperellipticCurve;
use rustmath_curves::jacobian::Jacobian;
use rustmath_rationals::Rational;

#[test]
fn genus2_jacobian_group_law_over_q() {
    // y^2 = x^5 - x  (genus 2): a=b=d=0, c=-1 in x^5 + a x^3 + b x^2 + c x + d.
    let curve = HyperellipticCurve::<Rational>::genus_2_quintic(
        Rational::zero(),
        Rational::zero(),
        -Rational::one(),
        Rational::zero(),
    )
    .expect("y^2 = x^5 - x is a valid genus-2 curve");
    assert_eq!(curve.genus, 2, "genus of x^5 - x");

    // (0,0) is on the curve: 0^2 = 0^5 - 0.
    assert!(curve.contains_point(&Rational::zero(), &Rational::zero()));

    let jac = Jacobian::new(curve);
    let p = jac.point(Rational::zero(), Rational::zero());
    assert!(!p.is_zero());

    // Identity law: P + 0 = P.
    let zero = jac.zero();
    assert_eq!(jac.add(&p, &zero), p, "P + 0 = P");

    // Doubling stays in reduced Mumford form with deg(u) <= genus.
    let doubled = p.double();
    assert!(doubled.is_reduced(), "2P reduced");
    assert!(doubled.degree() <= 2, "deg(2P) <= genus");

    // Cantor scalar-mul agrees with doubling: 2*P == P.double().
    assert_eq!(p.scalar_multiply(2), doubled, "2·P == double(P)");
}

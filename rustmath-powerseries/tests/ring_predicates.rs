//! Ring predicates of the series types (MAGMA Handbook Ch. 49 §49.4.4 `IsUnit`,
//! `IsNilpotent`; Ch. 50 for lazy series): `unit_status`, `unit_inverse`,
//! `nilpotent_status`, `multiplicative_order_status`, `static_characteristic`.
//!
//! Reference values from PARI/GP 2.15 (`echo '...' | gp -q`):
//!   1/(1+x+O(x^5))                  -> 1 - x + x^2 - x^3 + x^4 + O(x^5)
//!   1/(1+2*x+3*x^2+O(x^6))          -> 1 - 2*x + x^2 + 4*x^3 - 11*x^4 + 10*x^5 + O(x^6)
//!   1/(-1+x^2+O(x^6))               -> -1 - x^2 - x^4 + O(x^6)
//!   lift(1/(Mod(1,4)+Mod(2,4)*x+O(x^6))) -> 1 + 2*x + O(x^6)
//!   1/(Mod(2,4)+x+O(x^6))           -> impossible inverse in Fl_inv: Mod(2, 4)
//!   (Mod(2,4)+x+O(x^6))^4           -> x^4 + O(x^8)
//!   (Mod(2,4)+Mod(2,4)*x)^2         -> Mod(0, 4)
//!   1/(x^-2*(1+3*x)+O(x^3))         -> x^2 - 3*x^3 + 9*x^4 - 27*x^5 + 81*x^6 + O(x^7)
//!   1/(3*x^-1+x+O(x^4))             -> 1/3*x - 1/9*x^3 + 1/27*x^5 + O(x^6)
//!   1/(1+y+O(y^6))                  -> 1 - y + y^2 - y^3 + y^4 - y^5 + O(y^6)
//!   1/(1+x+O(x^10))                 -> 1 - x + ... - x^9 + O(x^10)
//!   polcoeff(1/(1+x^25+O(x^60)), 25) -> -1;  polcoeff(1/(1+x^25+O(x^60)), 50) -> 1
//! At c79a582 every hook kept its default: `None`/`NotImplemented` for these series, and
//! three wrong answers (`O(x^0)` a non-unit and nilpotent, lazy `1 + x^25` of order 1 with
//! itself as inverse, lazy `x^30` nilpotent).

use rustmath_core::{MathError, MultiplicativeOrder, Ring};
use rustmath_finitefields::IntegerMod;
use rustmath_integers::Integer;
use rustmath_powerseries::{LaurentSeries, LazyPowerSeries, PowerSeries, PuiseuxSeries};
use rustmath_rationals::Rational;

fn zi(cs: &[i64]) -> Vec<Integer> {
    cs.iter().map(|&c| Integer::from(c)).collect()
}
fn coeffs<R: Ring>(f: &PowerSeries<R>) -> Vec<R> {
    (0..f.precision()).map(|i| f.coeff(i).clone()).collect()
}
fn z4(cs: &[i64]) -> Vec<IntegerMod> {
    cs.iter()
        .map(|&c| IntegerMod::new(Integer::from(c), Integer::from(4)).unwrap())
        .collect()
}

#[test]
fn power_series_units_over_z() {
    let f = PowerSeries::new(zi(&[1, 1]), 5);
    assert_eq!(f.unit_status(), Some(true));
    assert_eq!(coeffs(&f.unit_inverse().unwrap()), zi(&[1, -1, 1, -1, 1]));
    let g = PowerSeries::new(zi(&[1, 2, 3]), 6);
    assert_eq!(
        coeffs(&g.unit_inverse().unwrap()),
        zi(&[1, -2, 1, 4, -11, 10])
    );
    let h = PowerSeries::new(zi(&[-1, 0, 1]), 6);
    assert_eq!(
        coeffs(&h.unit_inverse().unwrap()),
        zi(&[-1, 0, -1, 0, -1, 0])
    );
    // The units of Z[[x]] have constant term ±1.
    let k = PowerSeries::new(zi(&[2, 1]), 5);
    assert_eq!(k.unit_status(), Some(false));
    assert_eq!(k.unit_inverse(), Err(MathError::NotInvertible));
    assert_eq!(
        k.multiplicative_order_status(),
        MultiplicativeOrder::NotAUnit
    );
    assert_eq!(k.nilpotent_status(), Some(false));
    assert_eq!(
        <PowerSeries<Integer> as Ring>::static_characteristic(),
        Some(0)
    );
    assert_eq!(
        <PowerSeries<Integer> as Ring>::static_is_commutative(),
        Some(true)
    );
}

#[test]
fn power_series_over_z4() {
    let f = PowerSeries::new(z4(&[1, 2]), 6);
    assert_eq!(f.unit_status(), Some(true));
    let inv = f.unit_inverse().unwrap();
    assert_eq!(inv, PowerSeries::new(z4(&[1, 2]), 6));
    let g = PowerSeries::new(z4(&[2, 1]), 6);
    assert_eq!(g.unit_status(), Some(false));
    assert_eq!(g.unit_inverse(), Err(MathError::NotInvertible));
    // 2 + x is not nilpotent ((2 + x)^4 = x^4 + ...): its coefficient 1 is not nilpotent
    // and Z/4 is commutative.
    let expected = if <IntegerMod as Ring>::static_is_commutative() == Some(true) {
        Some(false)
    } else {
        None
    };
    assert_eq!(g.nilpotent_status(), expected);
    assert_eq!(expected, Some(false));
    // 2 + 2x is nilpotent as a polynomial, but the unknown tail decides nothing.
    let h = PowerSeries::new(z4(&[2, 2]), 6);
    assert_eq!(h.nilpotent_status(), None);
}

#[test]
fn power_series_precision_zero() {
    let f = PowerSeries::<Integer>::new(vec![], 0);
    assert_eq!(f.unit_status(), None);
    assert_eq!(f.nilpotent_status(), None);
    assert!(!matches!(
        f.unit_inverse(),
        Ok(_) | Err(MathError::NotInvertible)
    ));
    assert_eq!(
        f.multiplicative_order_status(),
        MultiplicativeOrder::Undecided
    );
    // Known precision: O(x^5) is weakly zero (the convention), 1 + O(x^5) weakly one.
    let z = PowerSeries::new(zi(&[0]), 5);
    assert_eq!(z.nilpotent_status(), Some(true));
    let one = PowerSeries::new(zi(&[1]), 5);
    assert_eq!(
        one.multiplicative_order_status(),
        MultiplicativeOrder::Finite(1)
    );
    let minus = PowerSeries::new(zi(&[-1]), 5);
    assert_eq!(
        minus.multiplicative_order_status(),
        MultiplicativeOrder::Finite(2)
    );
}

#[test]
fn laurent_hooks() {
    let f = LaurentSeries::new(-2, zi(&[1, 3, 0, 0, 0]));
    assert_eq!(f.unit_status(), Some(true));
    let inv = f.unit_inverse().unwrap();
    assert_eq!(inv.lowest_exponent(), 2);
    assert_eq!(inv.relative_precision(), 5);
    let got: Vec<Integer> = (2..7).map(|e| inv.coefficient(e)).collect();
    assert_eq!(got, zi(&[1, -3, 9, -27, 81]));
    assert_eq!(f.nilpotent_status(), Some(false));
    assert_eq!(
        <LaurentSeries<Integer> as Ring>::static_characteristic(),
        Some(0)
    );

    let q = |n: i64, d: i64| Rational::new(n, d).unwrap();
    let g = LaurentSeries::new(-1, vec![q(3, 1), q(0, 1), q(1, 1), q(0, 1), q(0, 1)]);
    let ginv = g.unit_inverse().unwrap();
    assert_eq!(ginv.lowest_exponent(), 1);
    let got: Vec<Rational> = (1..6).map(|e| ginv.coefficient(e)).collect();
    assert_eq!(got, vec![q(1, 3), q(0, 1), q(-1, 9), q(0, 1), q(1, 27)]);
    assert_eq!(ginv, g.try_inverse().unwrap());

    // Over Z the leading coefficient 3 is not a unit; whether the series is a unit would
    // need "Z is a domain", which no hook states (documented limit).
    let gz = LaurentSeries::new(-1, zi(&[3, 0, 1, 0, 0]));
    assert_eq!(gz.unit_status(), None);
    assert!(matches!(
        gz.unit_inverse(),
        Err(MathError::NotImplemented(_))
    ));
    assert_eq!(gz.nilpotent_status(), Some(false));
    // The empty window knows nothing.
    let e = LaurentSeries::<Integer>::new(3, vec![]);
    assert_eq!(e.unit_status(), None);
    assert_eq!(e.nilpotent_status(), None);
    assert_eq!(
        e.multiplicative_order_status(),
        MultiplicativeOrder::Undecided
    );
    let w = LaurentSeries::new(0, zi(&[0, 0]));
    assert_eq!(w.unit_status(), Some(false));
    assert_eq!(w.unit_inverse(), Err(MathError::NotInvertible));
}

#[test]
fn puiseux_hooks() {
    let f = PuiseuxSeries::from_laurent(2, LaurentSeries::new(0, zi(&[1, 1, 0, 0, 0, 0])));
    assert_eq!(f.unit_status(), Some(true));
    let inv = f.unit_inverse().unwrap();
    let expected =
        PuiseuxSeries::from_laurent(2, LaurentSeries::new(0, zi(&[1, -1, 1, -1, 1, -1])));
    assert_eq!(inv, expected);
    assert_eq!(f.nilpotent_status(), Some(false));
    assert_eq!(
        <PuiseuxSeries<Integer> as Ring>::static_characteristic(),
        Some(0)
    );
}

#[test]
fn lazy_hooks() {
    let f = LazyPowerSeries::from_coeffs(zi(&[1, 1]));
    assert_eq!(f.unit_status(), Some(true));
    let inv = f.unit_inverse().unwrap();
    assert_eq!(
        inv.coefficients(10),
        zi(&[1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    );

    let mut c = vec![0i64; 26];
    c[0] = 1;
    c[25] = 1;
    let g = LazyPowerSeries::from_coeffs(zi(&c));
    assert_eq!(
        g.multiplicative_order_status(),
        MultiplicativeOrder::Undecided
    );
    let ginv = g.unit_inverse().unwrap();
    assert_eq!(ginv.coeff(25), Integer::from(-1));
    assert_eq!(ginv.coeff(50), Integer::from(1));

    let mut c30 = vec![0i64; 31];
    c30[30] = 1;
    let x30 = LazyPowerSeries::from_coeffs(zi(&c30));
    assert_eq!(x30.nilpotent_status(), None);
    assert_eq!(x30.unit_status(), Some(false));
    assert_eq!(
        x30.multiplicative_order_status(),
        MultiplicativeOrder::NotAUnit
    );

    // 2x^3 + x^5 over Z: the lowest coefficient 2 is not nilpotent.
    let h = LazyPowerSeries::from_coeffs(zi(&[0, 0, 0, 2, 0, 1]));
    assert_eq!(h.nilpotent_status(), Some(false));
    assert_eq!(
        <LazyPowerSeries<Integer> as Ring>::static_characteristic(),
        Some(0)
    );
}

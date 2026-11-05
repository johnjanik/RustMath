//! THROWAWAY adversarial-verification tests — DELETE THIS FILE AFTER RUNNING.
//! Expected values derived independently with sympy before writing.

use rustmath_powerseries::PowerSeries;
use rustmath_rationals::Rational;

fn q(n: i64) -> Rational {
    Rational::from_i64(n)
}
fn qq(n: i64, d: i64) -> Rational {
    Rational::new(n, d).unwrap()
}

#[test]
fn adv_exp_log_one_plus_x() {
    // sympy: exp(log(1+x)) = 1 + x exactly.
    let prec = 12;
    let one_plus_x: PowerSeries<Rational> = PowerSeries::new(vec![q(1), q(1)], prec);
    let l = one_plus_x.log().unwrap();
    let e = l.exp().unwrap();
    assert_eq!(e.coeff(0), &q(1));
    assert_eq!(e.coeff(1), &q(1));
    for i in 2..prec {
        assert_eq!(e.coeff(i), &q(0), "exp(log(1+x)) coeff {}", i);
    }
    // sanity: log(1+x) itself alternates: x - x^2/2 + x^3/3 - ...
    assert_eq!(l.coeff(1), &q(1));
    assert_eq!(l.coeff(2), &qq(-1, 2));
    assert_eq!(l.coeff(3), &qq(1, 3));
    assert_eq!(l.coeff(4), &qq(-1, 4));
}

#[test]
fn adv_reversion_signed_catalan() {
    // sympy: series of (-1+sqrt(1+4x))/2, the inverse of t+t^2:
    // [0, 1, -1, 2, -5, 14, -42, 132, -429, 1430]
    let prec = 10;
    let f: PowerSeries<Rational> = PowerSeries::new(vec![q(0), q(1), q(1)], prec);
    let g = f.reversion().unwrap();
    let expect: [i64; 10] = [0, 1, -1, 2, -5, 14, -42, 132, -429, 1430];
    for (i, e) in expect.iter().enumerate() {
        assert_eq!(g.coeff(i), &q(*e), "reversion coeff {}", i);
    }
    // round trips: f(g) = x and g(f) = x
    let fg = f.try_compose(&g).unwrap();
    let gf = g.try_compose(&f).unwrap();
    for i in 0..prec {
        let want = if i == 1 { q(1) } else { q(0) };
        assert_eq!(fg.coeff(i), &want, "f(g) coeff {}", i);
        assert_eq!(gf.coeff(i), &want, "g(f) coeff {}", i);
    }
}

#[test]
fn adv_integral_then_derivative_roundtrip() {
    // f = 3 - x + 5x^2 + 7x^3 - 2x^5 at precision 9.
    let prec = 9;
    let f: PowerSeries<Rational> =
        PowerSeries::new(vec![q(3), q(-1), q(5), q(7), q(0), q(-2)], prec);
    let fi = f.integral().unwrap();
    // hand-checked antiderivative: [0, 3, -1/2, 5/3, 7/4, 0, -1/3, 0, 0]
    assert_eq!(fi.coeff(0), &q(0));
    assert_eq!(fi.coeff(1), &q(3));
    assert_eq!(fi.coeff(2), &qq(-1, 2));
    assert_eq!(fi.coeff(3), &qq(5, 3));
    assert_eq!(fi.coeff(4), &qq(7, 4));
    assert_eq!(fi.coeff(5), &q(0));
    assert_eq!(fi.coeff(6), &qq(-1, 3));
    let fd = fi.derivative();
    for i in 0..prec - 1 {
        assert_eq!(fd.coeff(i), f.coeff(i), "d/dx integral(f) coeff {}", i);
    }
}

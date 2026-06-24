//! Smoke check exercising the public API (a runnable stand-in for `cargo test`,
//! which is blocked in this sandbox). Mirrors the unit tests' assertions.

use rustmath_core::{Field, Ring};
use rustmath_functionfields::{
    factor_over_qt, ff_poly_from_coeffs, is_irreducible_over_qt, specialize_poly, FunctionField,
    RationalFunction, Specialization,
};
use rustmath_rationals::Rational;

fn q(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn x2_minus_t2() -> rustmath_functionfields::FfPoly {
    let t = RationalFunction::t();
    let minus_t2 = RationalFunction::zero() - (t.clone() * t);
    ff_poly_from_coeffs(vec![
        minus_t2,
        RationalFunction::zero(),
        RationalFunction::one(),
    ])
}

fn main() {
    let mut pass = 0;
    let mut fail = 0;
    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond {
                pass += 1;
                println!("ok   - {}", $name);
            } else {
                fail += 1;
                println!("FAIL - {}", $name);
            }
        };
    }

    // --- ratfunc ---
    let t = RationalFunction::t();
    let inv = t.inverse().unwrap();
    check!("t * (1/t) == 1", t.clone() * inv == RationalFunction::one());

    let num = rustmath_functionfields::QtPoly::new(vec![q(-1), q(0), q(1)]);
    let den = rustmath_functionfields::QtPoly::new(vec![q(-1), q(1)]);
    let rf = RationalFunction::new(num, den).unwrap();
    check!("(t^2-1)/(t-1) reduces to a polynomial", rf.is_polynomial());

    // --- specialization ---
    let f = x2_minus_t2();
    check!(
        "specialize x^2-t^2 at t=3 is Good x^2-9",
        matches!(specialize_poly(&f, &q(3)),
            Specialization::Good(ref p) if p.coefficients() == [q(-9), q(0), q(1)])
    );
    check!(
        "specialize x^2-t^2 at t=0 is NotSeparable",
        matches!(specialize_poly(&f, &q(0)), Specialization::NotSeparable(_))
    );

    // --- factorization ---
    let (content, factors) = factor_over_qt(&f);
    let total_deg: usize = factors.iter().map(|(g, m)| g.degree().unwrap() * m).sum();
    check!("factor x^2-t^2: content == 1", content == RationalFunction::one());
    check!("factor x^2-t^2: two distinct factors", factors.len() == 2);
    check!("factor x^2-t^2: total degree 2", total_deg == 2);
    // Reconstruct product.
    let mut prod = ff_poly_from_coeffs(vec![content]);
    for (g, m) in &factors {
        for _ in 0..*m {
            prod = prod * g.clone();
        }
    }
    check!("factor x^2-t^2: product reconstructs F", prod == f);

    // x^2 - t irreducible.
    let f_irr = ff_poly_from_coeffs(vec![
        RationalFunction::zero() - t.clone(),
        RationalFunction::zero(),
        RationalFunction::one(),
    ]);
    check!("x^2 - t is irreducible over Q(t)", is_irreducible_over_qt(&f_irr).unwrap());

    // x^3 - t irreducible.
    let f_cub = ff_poly_from_coeffs(vec![
        RationalFunction::zero() - t.clone(),
        RationalFunction::zero(),
        RationalFunction::zero(),
        RationalFunction::one(),
    ]);
    check!("x^3 - t is irreducible over Q(t)", is_irreducible_over_qt(&f_cub).unwrap());

    // (x-t)^2 -> one factor multiplicity 2.
    let lin = ff_poly_from_coeffs(vec![
        RationalFunction::zero() - t.clone(),
        RationalFunction::one(),
    ]);
    let sq = lin.clone() * lin.clone();
    let (_, fac2) = factor_over_qt(&sq);
    check!("(x-t)^2 -> single factor", fac2.len() == 1);
    check!("(x-t)^2 -> multiplicity 2", fac2.get(0).map(|(_, m)| *m) == Some(2));

    // (x^2-t)(x^2-(t+1)) reducible into two quadratics.
    let q1 = ff_poly_from_coeffs(vec![
        RationalFunction::zero() - t.clone(),
        RationalFunction::zero(),
        RationalFunction::one(),
    ]);
    let q2 = ff_poly_from_coeffs(vec![
        RationalFunction::zero() - (t.clone() + RationalFunction::one()),
        RationalFunction::zero(),
        RationalFunction::one(),
    ]);
    let quartic = q1 * q2;
    check!(
        "quartic (x^2-t)(x^2-(t+1)) is reducible",
        !is_irreducible_over_qt(&quartic).unwrap()
    );
    let (_, fq) = factor_over_qt(&quartic);
    let qt_total: usize = fq.iter().map(|(g, m)| g.degree().unwrap() * m).sum();
    check!("quartic factors total degree 4", qt_total == 4);

    // --- FunctionField ---
    let c = t.clone() * t.clone() * t.clone() + RationalFunction::one();
    let fdef = ff_poly_from_coeffs(vec![
        RationalFunction::zero() - c,
        RationalFunction::zero(),
        RationalFunction::one(),
    ]);
    let k = FunctionField::new(fdef).unwrap();
    check!("FunctionField degree == 2", k.degree() == 2);
    check!("reducible F rejected by FunctionField::new", FunctionField::new(x2_minus_t2()).is_err());
    check!(
        "K specialized at t=2 is x^2-9 (good)",
        matches!(k.specialize(&q(2)),
            Specialization::Good(ref p) if p.coefficients() == [q(-9), q(0), q(1)])
    );

    println!("\n{} passed, {} failed", pass, fail);
    if fail > 0 {
        std::process::exit(1);
    }
}

//! MPFR-oracle conformance test for the pure-Rust [`BigFloat`] transcendentals.
//!
//! `bigfloat.rs` documents a guaranteed accuracy for `sqrt`/`exp`/`ln`/`sin`/
//! `cos`/`atan`/`pi`/`e`: relative error at most 4 ulp at the value's
//! precision `p` (`|err| <= |exact| * 2^(2-p)`), except that `sin`/`cos`
//! results with `|exact| < 2^-24` (argument almost exactly on a zero of the
//! function) are guaranteed to **absolute** error `2^-(p+8)` instead. This
//! test enforces those bounds against MPFR via `rug` (already a regular
//! dependency of this crate — `RealMPFR` wraps it — so no new dependency).
//!
//! Coverage deliberately includes the hard cases: large arguments (argument
//! reduction mod 2π for `sin`/`cos`, re-squaring for `exp`) and values near
//! branch points (`ln` near 1, `sqrt` near 0).

use rug::{Float, Integer as RugInteger};
use rustmath_core::analytic::RealField;
use rustmath_reals::bigfloat::BigFloat;
use std::str::FromStr;

/// Guard bits used by the BigFloat implementation (`bigfloat.rs::GUARD`);
/// the tiny-`sin`/`cos` absolute-error branch keys off `2^-GUARD`.
const GUARD: u32 = 24;

/// Extra oracle bits beyond the value under test.
const WIDE_EXTRA: u32 = 96;

/// Exact conversion of a `BigFloat` (dyadic by construction) to a
/// `rug::Float` with `prec` bits. Asserts exactness.
fn to_rug(x: &BigFloat, prec: u32) -> Float {
    let (m, e) = x.mantissa_exponent();
    assert!(
        (m.bit_length() as u32) <= prec,
        "oracle precision too small for exact conversion"
    );
    let mi = RugInteger::from_str(&m.to_string()).expect("Integer decimal digits");
    let f = Float::with_val(prec, mi);
    assert!(e >= i32::MIN as i64 && e <= i32::MAX as i64);
    f << (e as i32)
}

/// Check `got` (computed by BigFloat at `x`'s precision) against
/// `oracle(x)` evaluated by MPFR at much higher precision.
///
/// `tiny_abs` enables the documented `sin`/`cos` near-zero absolute bound.
fn check(
    name: &str,
    x: &BigFloat,
    got: &BigFloat,
    oracle: impl Fn(Float) -> Float,
    tiny_abs: bool,
) {
    let p = RealField::precision(x);
    let wide = p as u32 + WIDE_EXTRA;
    let want = oracle(to_rug(x, wide));
    let err = Float::with_val(wide, to_rug(got, wide) - &want).abs();
    // relative bound: |want| * 2^(2-p)  (4 ulp at precision p)
    let mut bound: Float = (want.clone().abs() << 2u32) >> (p as u32);
    if tiny_abs {
        let thresh = Float::with_val(wide, 1) >> GUARD;
        if want.clone().abs() < thresh {
            let abs_bound = Float::with_val(wide, 1) >> (p as u32 + 8);
            if bound < abs_bound {
                bound = abs_bound;
            }
        }
    }
    assert!(
        err <= bound,
        "{name}(x) beyond documented accuracy at prec {p}:\n  x     = {x}\n  got   = {got}\n  want  = {want}\n  err   = {err}\n  bound = {bound}"
    );
}

const PRECISIONS: [u64; 4] = [64, 128, 256, 640];

/// Parse a decimal literal to a BigFloat at precision `p` (the rounded dyadic
/// becomes the *exact* test input — the oracle sees the same dyadic).
fn bf(s: &str, p: u64) -> BigFloat {
    BigFloat::from_decimal_str(s, p).expect("test literal parses")
}

#[test]
fn oracle_sqrt() {
    for &p in &PRECISIONS {
        for s in ["2", "3", "0.5", "123456789.123456789", "1e30"] {
            let x = bf(s, p);
            check("sqrt", &x, &x.sqrt(), |f| f.sqrt(), false);
        }
        // near the branch point 0
        for s in ["1e-30", "1e-200", "4e-290"] {
            let x = bf(s, p);
            check("sqrt", &x, &x.sqrt(), |f| f.sqrt(), false);
        }
        // 1 + 2^-40 (exactly representable at every tested precision)
        let x = bf("1", p) + BigFloat::from_f64((2.0f64).powi(-40), p);
        check("sqrt", &x, &x.sqrt(), |f| f.sqrt(), false);
    }
}

#[test]
fn oracle_exp() {
    for &p in &PRECISIONS {
        for s in ["-700", "-50", "-1", "0.125", "0.5", "1", "10", "100", "700"] {
            let x = bf(s, p);
            check("exp", &x, &x.exp(), |f| f.exp(), false);
        }
    }
}

#[test]
fn oracle_ln() {
    for &p in &PRECISIONS {
        for s in ["1e-300", "1e-30", "0.5", "2", "10", "6.02214076e23", "1e300"] {
            let x = bf(s, p);
            check("ln", &x, &x.ln(), |f| f.ln(), false);
        }
        // near the branch point 1: 1 ± 2^-40 (exact dyadics; ln ≈ ±2^-40,
        // where naive series/cancellation would lose ~40 bits)
        let tiny = BigFloat::from_f64((2.0f64).powi(-40), p);
        let xp = bf("1", p) + tiny.clone();
        check("ln", &xp, &xp.ln(), |f| f.ln(), false);
        let xm = bf("1", p) - tiny;
        check("ln", &xm, &xm.ln(), |f| f.ln(), false);
    }
}

#[test]
fn oracle_sin_cos() {
    for &p in &PRECISIONS {
        for s in ["0.5", "1", "-2.5", "3", "12345.6789", "1e10", "1e15"] {
            let x = bf(s, p);
            check("sin", &x, &x.sin(), |f| f.sin(), true);
            check("cos", &x, &x.cos(), |f| f.cos(), true);
        }
    }
}

#[test]
fn oracle_sin_near_pi_tiny_result() {
    // Argument almost exactly on a zero of sin: 64 decimal digits of π,
    // rounded to p bits. The exact sin is ~max(2^-p, 10^-65) — far below
    // 2^-GUARD for every tested precision — so the documented *absolute*
    // bound 2^-(p+8) applies. This stresses the reduction mod 2π: the
    // implementation must internally carry π well beyond p bits.
    let pi64 = "3.141592653589793238462643383279502884197169399375105820974944592";
    for &p in &PRECISIONS {
        let x = bf(pi64, p);
        check("sin", &x, &x.sin(), |f| f.sin(), true);
        // and cos near its zero at π/2
        let half_pi = "1.570796326794896619231321691639751442098584699687552910487472296";
        let y = bf(half_pi, p);
        check("cos", &y, &y.cos(), |f| f.cos(), true);
    }
}

#[test]
fn oracle_atan() {
    for &p in &PRECISIONS {
        for s in ["1e-30", "0.125", "0.5", "1", "-3", "1e10", "-1e15"] {
            let x = bf(s, p);
            check("atan", &x, &x.atan(), |f| f.atan(), false);
        }
    }
}

#[test]
fn oracle_constants_pi_e() {
    for &p in &PRECISIONS {
        let wide = p as u32 + WIDE_EXTRA;
        let pi = <BigFloat as RealField>::pi(p);
        let want = Float::with_val(wide, rug::float::Constant::Pi);
        let err = Float::with_val(wide, to_rug(&pi, wide) - &want).abs();
        let bound: Float = (want.clone().abs() << 2u32) >> (p as u32);
        assert!(err <= bound, "pi at prec {p}: err {err} > bound {bound}");

        let e = <BigFloat as RealField>::e(p);
        let want = Float::with_val(wide, 1u32).exp();
        let err = Float::with_val(wide, to_rug(&e, wide) - &want).abs();
        let bound: Float = (want.clone().abs() << 2u32) >> (p as u32);
        assert!(err <= bound, "e at prec {p}: err {err} > bound {bound}");
    }
}

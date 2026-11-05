//! Real and complex analytic fields, plus a certified-arithmetic [`Ball`] trait.
//!
//! MAGMA source: Handbook chapter 25 (real and complex fields), chapter 46
//! (Newton polygons), 127 (L-functions), 132 (modular forms) — everything that
//! needs an arbitrary-precision real or complex field with the elementary and
//! transcendental function surface.
//!
//! ## Precision model
//!
//! MAGMA precision is a **per-field / per-value** property. Rather than push a
//! precision parameter into the whole trait tower (which would poison `Field`
//! for exact rings), precision is exposed two ways:
//!
//! * a [`precision`](RealField::precision) accessor on each *value*, and
//! * precision-carrying *constructors* (`pi(prec)`, `e(prec)`, `from_f64(x,
//!   prec)`), with a precision-carrying [`Parent`](crate::parent::Parent)
//!   (`BigFloatField(prec)` in `rustmath-reals`) built on top for
//!   construction/coercion.
//!
//! Binary operations combine two values at the larger of their precisions; this
//! keeps precision from silently degrading. All of this is documented on the
//! concrete `BigFloat`/`BigComplex` backends.
//!
//! ## Rounding
//!
//! Every operation returns a result rounded to the result value's precision.
//! The pure-Rust backend (`rustmath-reals::bigfloat::BigFloat`) rounds
//! half-to-even; the rug/MPFR backend uses MPFR's round-to-nearest-even.
//! Transcendental functions are *not* guaranteed correctly rounded, but each
//! backend documents an accuracy bound (the pure-Rust backend targets a few
//! ulps at the value's precision; see `bigfloat.rs`).
//!
//! ## Finiteness
//!
//! Backends differ on non-finite values and document their model:
//!
//! * `BigFloat`/`BigComplex` have **no** `±∞`/`NaN`; every value is a finite
//!   dyadic number. Constructors map non-finite `f64` input to zero
//!   (documented on [`RealField::from_f64`]), and domain errors in the
//!   transcendentals are implementation-defined per method doc.
//! * `RealMPFR`/`ComplexMPFR` inherit the full MPFR special-value model
//!   (`±∞`, `NaN`) and propagate them IEEE-style.
//!
//! ## Naming
//!
//! The trait names `RealField`/`ComplexField` deliberately collide with the
//! pre-existing *structs* `rustmath_reals::RealField` and
//! `rustmath_complex::ComplexField`; none of the traits in this module are
//! re-exported from the crate root. Import them path-qualified, e.g.
//! `use rustmath_core::analytic::RealField;` (aliasing if both names are
//! needed in one scope).
//!
//! Purely additive.

use crate::ordering::OrderedField;
use crate::{Field, Result};

/// A real, ordered field with elementary and transcendental functions.
///
/// This is the trait the `rustmath-reals::BigFloat` (pure-Rust arbitrary
/// precision) and `rustmath-reals::RealMPFR` (rug/MPFR) backends both implement.
pub trait RealField: OrderedField {
    /// The working precision (in bits) carried by this value.
    fn precision(&self) -> u64;

    /// Construct a value approximating `x` to `precision` bits.
    fn from_f64(x: f64, precision: u64) -> Self;

    /// A best-effort `f64` approximation (may lose precision).
    fn to_f64(&self) -> f64;

    /// `π` computed to `precision` bits.
    fn pi(precision: u64) -> Self;

    /// Euler's number `e` computed to `precision` bits.
    fn e(precision: u64) -> Self;

    /// The (principal, non-negative) square root. `sqrt` of a negative value is
    /// implementation-defined (typically returns zero or `NaN`-like).
    fn sqrt(&self) -> Self;

    /// The exponential `exp(self)`.
    fn exp(&self) -> Self;

    /// The natural logarithm `ln(self)` (of a positive value).
    fn ln(&self) -> Self;

    /// The sine.
    fn sin(&self) -> Self;

    /// The cosine.
    fn cos(&self) -> Self;

    /// The principal arctangent, in `(-π/2, π/2)`.
    fn atan(&self) -> Self;

    // ---- Wave 0 contract delta (additive) -----------------------------------

    /// Return this value re-rounded to `precision` bits.
    ///
    /// Widening never invents information (the value is unchanged; only the
    /// carried precision grows); narrowing rounds to the new precision using
    /// the backend's rounding rule.
    fn with_precision(&self, precision: u64) -> Self;

    /// The largest integer-valued element `<= self` (at `self`'s precision).
    ///
    /// If the integer needs more bits than `self`'s precision it is rounded to
    /// that precision, like every other result.
    fn floor(&self) -> Self;

    /// The smallest integer-valued element `>= self` (at `self`'s precision).
    fn ceil(&self) -> Self;

    /// The nearest integer-valued element; ties round **away from zero**
    /// (`round(2.5) = 3`, `round(-2.5) = -3`).
    fn round(&self) -> Self;

    /// Four-quadrant arctangent: the argument of the point `(x, y)` with
    /// `y = self`, in `(-π, π]`. By convention `atan2(0, 0) = 0`, and
    /// `atan2(0, x) = π` for `x < 0`.
    ///
    /// The default derives from [`atan`](RealField::atan), [`pi`](RealField::pi)
    /// and the sign, at the larger of the two operands' precisions; backends
    /// with a native implementation should override it.
    fn atan2(&self, x: &Self) -> Self {
        let prec = RealField::precision(self).max(RealField::precision(x));
        let sy = self.sign();
        let sx = x.sign();
        if sx == 0 {
            let pi = Self::pi(prec);
            let two = Self::from_f64(2.0, prec);
            return match sy {
                0 => Self::from_f64(0.0, prec),
                s if s > 0 => pi / two,
                _ => -(pi / two),
            };
        }
        let base = (self.clone() / x.clone()).atan();
        if sx > 0 {
            base
        } else if sy >= 0 {
            base + Self::pi(prec)
        } else {
            base - Self::pi(prec)
        }
    }

    /// Render as a decimal string with `digits` significant digits.
    ///
    /// Guarantees: the printed value differs from `self` by at most one unit
    /// in the last printed digit, and round-trips through
    /// [`from_decimal_str`](RealField::from_decimal_str) at `self`'s precision
    /// whenever `digits` covers that precision (`digits >= ceil(prec·log10 2) + 1`).
    /// The exact formatting (plain vs. scientific, trailing zeros) is
    /// backend-defined.
    fn to_decimal_string(&self, digits: usize) -> String;

    /// Parse a decimal string (optional sign, optional fraction, optional
    /// `e`/`E` decimal exponent) and round it to `precision` bits.
    ///
    /// Returns `MathError::ParseError` on malformed input.
    fn from_decimal_str(s: &str, precision: u64) -> Result<Self>;

    /// `self` raised to a real power, defaulted as `exp(exponent · ln(self))`.
    ///
    /// Domain (default implementation):
    /// * base `> 0`: the principal real power;
    /// * base `== 0`: `0^0 = 1` and `0^y = 0` for `y != 0` (this model has no
    ///   infinities, so `y < 0` also yields zero — documented, not IEEE);
    /// * base `< 0`: **panics** — there is no real value in general and the
    ///   default refuses to fabricate one. Backends that can do better (e.g.
    ///   integral exponents, or MPFR's NaN model) override this method and
    ///   document their extension.
    ///
    /// NB: [`Ring::pow`](crate::Ring::pow)`(u32)` also exists; where both
    /// traits are in scope, call this one as `RealField::pow(&x, &y)`.
    fn pow(&self, exponent: &Self) -> Self {
        let prec = RealField::precision(self).max(RealField::precision(exponent));
        if self.is_zero() {
            if exponent.is_zero() {
                return Self::from_f64(1.0, prec);
            }
            return Self::from_f64(0.0, prec);
        }
        if self.sign() < 0 {
            panic!(
                "RealField::pow (default impl): negative base has no principal real power; \
                 use a backend override that supports it"
            );
        }
        (exponent.clone() * self.ln()).exp()
    }
}

/// A complex field over a real subfield, with the standard transcendental
/// surface.
pub trait ComplexField: Field {
    /// The underlying real field.
    type Real: RealField;

    /// The real part.
    fn re(&self) -> Self::Real;

    /// The imaginary part.
    fn im(&self) -> Self::Real;

    /// The complex conjugate.
    fn conj(&self) -> Self;

    /// The modulus `|self|` (a real number).
    fn abs(&self) -> Self::Real;

    /// The argument `arg(self) ∈ (-π, π]` (a real number).
    fn arg(&self) -> Self::Real;

    /// Build a complex number from real and imaginary parts.
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;

    /// The imaginary unit `i` at the given precision.
    fn i(precision: u64) -> Self;

    /// The working precision (in bits).
    fn precision(&self) -> u64;

    /// A principal square root.
    fn sqrt(&self) -> Self;

    /// The complex exponential.
    fn exp(&self) -> Self;

    /// A principal branch of the complex logarithm.
    fn ln(&self) -> Self;

    /// The complex sine.
    fn sin(&self) -> Self;

    /// The complex cosine.
    fn cos(&self) -> Self;

    // ---- Wave 0 contract delta (additive) -----------------------------------

    /// Return this value with both parts re-rounded to `precision` bits (same
    /// semantics as [`RealField::with_precision`]).
    fn with_precision(&self, precision: u64) -> Self;

    /// `self` raised to a complex power, defaulted as the principal branch
    /// `exp(exponent · ln(self))`.
    ///
    /// Domain (default implementation): `0^0 = 1`; `0^w = 0` for `w != 0`
    /// (no infinities in the pure-Rust model); nonzero base uses the
    /// principal logarithm (`arg ∈ (-π, π]`).
    ///
    /// NB: [`Ring::pow`](crate::Ring::pow)`(u32)` also exists; where both
    /// traits are in scope, call this one as `ComplexField::pow(&z, &w)`.
    fn pow(&self, exponent: &Self) -> Self {
        if self.is_zero() {
            let prec =
                ComplexField::precision(self).max(ComplexField::precision(exponent));
            let re = if exponent.is_zero() { 1.0 } else { 0.0 };
            return Self::from_real_imag(
                <Self::Real as RealField>::from_f64(re, prec),
                <Self::Real as RealField>::from_f64(0.0, prec),
            );
        }
        (exponent.clone() * self.ln()).exp()
    }
}

/// A ball (center + non-negative radius) for certified / interval arithmetic.
///
/// A ball rigorously encloses a set of points: `contains(p)` returns `true` only
/// when `p` is *guaranteed* to lie within `radius` of `center`. The center is a
/// real or complex number; the radius is always a (non-negative) real.
pub trait Ball: Clone {
    /// The center type (a [`RealField`] or [`ComplexField`] element).
    type Center;
    /// The radius type (a non-negative real).
    type Radius;

    /// The center of the ball.
    fn center(&self) -> Self::Center;

    /// The (non-negative) radius of the ball.
    fn radius(&self) -> Self::Radius;

    /// Construct a ball from a center and a radius.
    fn from_center_radius(center: Self::Center, radius: Self::Radius) -> Self;

    /// Whether the ball is certified to contain `point`.
    fn contains(&self, point: &Self::Center) -> bool;

    /// Whether the ball is certified to contain zero.
    fn contains_zero(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ordering::OrderedRing;
    use crate::{CommutativeRing, MathError, Result, Ring};
    use std::fmt;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    // ---- An `f64`-backed RealField test double -------------------------------

    #[derive(Clone, Copy, Debug)]
    struct Dbl(f64);

    impl fmt::Display for Dbl {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl PartialEq for Dbl {
        fn eq(&self, o: &Self) -> bool {
            (self.0 - o.0).abs() < 1e-9
        }
    }
    impl PartialOrd for Dbl {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&o.0)
        }
    }
    impl Add for Dbl {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Dbl(self.0 + o.0)
        }
    }
    impl Sub for Dbl {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Dbl(self.0 - o.0)
        }
    }
    impl Mul for Dbl {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            Dbl(self.0 * o.0)
        }
    }
    impl Div for Dbl {
        type Output = Self;
        fn div(self, o: Self) -> Self {
            Dbl(self.0 / o.0)
        }
    }
    impl Neg for Dbl {
        type Output = Self;
        fn neg(self) -> Self {
            Dbl(-self.0)
        }
    }
    impl Ring for Dbl {
        fn zero() -> Self {
            Dbl(0.0)
        }
        fn one() -> Self {
            Dbl(1.0)
        }
        fn is_zero(&self) -> bool {
            self.0.abs() < 1e-9
        }
        fn is_one(&self) -> bool {
            (self.0 - 1.0).abs() < 1e-9
        }
    }
    impl CommutativeRing for Dbl {}
    impl Field for Dbl {
        fn inverse(&self) -> Result<Self> {
            if self.is_zero() {
                Err(MathError::DivisionByZero)
            } else {
                Ok(Dbl(1.0 / self.0))
            }
        }
    }
    impl OrderedRing for Dbl {
        fn sign(&self) -> i32 {
            if self.0 > 0.0 {
                1
            } else if self.0 < 0.0 {
                -1
            } else {
                0
            }
        }
        fn abs(&self) -> Self {
            Dbl(self.0.abs())
        }
    }
    impl OrderedField for Dbl {}
    impl RealField for Dbl {
        fn precision(&self) -> u64 {
            53
        }
        fn from_f64(x: f64, _p: u64) -> Self {
            Dbl(x)
        }
        fn to_f64(&self) -> f64 {
            self.0
        }
        fn pi(_p: u64) -> Self {
            Dbl(std::f64::consts::PI)
        }
        fn e(_p: u64) -> Self {
            Dbl(std::f64::consts::E)
        }
        fn sqrt(&self) -> Self {
            Dbl(self.0.sqrt())
        }
        fn exp(&self) -> Self {
            Dbl(self.0.exp())
        }
        fn ln(&self) -> Self {
            Dbl(self.0.ln())
        }
        fn sin(&self) -> Self {
            Dbl(self.0.sin())
        }
        fn cos(&self) -> Self {
            Dbl(self.0.cos())
        }
        fn atan(&self) -> Self {
            Dbl(self.0.atan())
        }
        fn with_precision(&self, _precision: u64) -> Self {
            *self
        }
        fn floor(&self) -> Self {
            Dbl(self.0.floor())
        }
        fn ceil(&self) -> Self {
            Dbl(self.0.ceil())
        }
        fn round(&self) -> Self {
            Dbl(self.0.round()) // f64::round ties away from zero
        }
        fn to_decimal_string(&self, digits: usize) -> String {
            format!("{:.*e}", digits.saturating_sub(1), self.0)
        }
        fn from_decimal_str(s: &str, _precision: u64) -> Result<Self> {
            s.parse::<f64>()
                .map(Dbl)
                .map_err(|e| MathError::ParseError(e.to_string()))
        }
        // atan2 and pow deliberately NOT overridden: the tests below exercise
        // the trait's default implementations.
    }

    // ---- An `f64`-backed ComplexField test double ----------------------------

    #[derive(Clone, Copy, Debug)]
    struct Cpx {
        re: f64,
        im: f64,
    }
    impl Cpx {
        fn new(re: f64, im: f64) -> Self {
            Cpx { re, im }
        }
    }
    impl fmt::Display for Cpx {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
    impl PartialEq for Cpx {
        fn eq(&self, o: &Self) -> bool {
            (self.re - o.re).abs() < 1e-9 && (self.im - o.im).abs() < 1e-9
        }
    }
    impl Add for Cpx {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Cpx::new(self.re + o.re, self.im + o.im)
        }
    }
    impl Sub for Cpx {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Cpx::new(self.re - o.re, self.im - o.im)
        }
    }
    impl Mul for Cpx {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            Cpx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
        }
    }
    impl Div for Cpx {
        type Output = Self;
        fn div(self, o: Self) -> Self {
            let d = o.re * o.re + o.im * o.im;
            Cpx::new(
                (self.re * o.re + self.im * o.im) / d,
                (self.im * o.re - self.re * o.im) / d,
            )
        }
    }
    impl Neg for Cpx {
        type Output = Self;
        fn neg(self) -> Self {
            Cpx::new(-self.re, -self.im)
        }
    }
    impl Ring for Cpx {
        fn zero() -> Self {
            Cpx::new(0.0, 0.0)
        }
        fn one() -> Self {
            Cpx::new(1.0, 0.0)
        }
        fn is_zero(&self) -> bool {
            self.re.abs() < 1e-9 && self.im.abs() < 1e-9
        }
        fn is_one(&self) -> bool {
            (self.re - 1.0).abs() < 1e-9 && self.im.abs() < 1e-9
        }
    }
    impl CommutativeRing for Cpx {}
    impl Field for Cpx {
        fn inverse(&self) -> Result<Self> {
            if self.is_zero() {
                Err(MathError::DivisionByZero)
            } else {
                Ok(Cpx::one() / *self)
            }
        }
    }
    impl ComplexField for Cpx {
        type Real = Dbl;
        fn re(&self) -> Dbl {
            Dbl(self.re)
        }
        fn im(&self) -> Dbl {
            Dbl(self.im)
        }
        fn conj(&self) -> Self {
            Cpx::new(self.re, -self.im)
        }
        fn abs(&self) -> Dbl {
            Dbl((self.re * self.re + self.im * self.im).sqrt())
        }
        fn arg(&self) -> Dbl {
            Dbl(self.im.atan2(self.re))
        }
        fn from_real_imag(re: Dbl, im: Dbl) -> Self {
            Cpx::new(re.0, im.0)
        }
        fn i(_p: u64) -> Self {
            Cpx::new(0.0, 1.0)
        }
        fn precision(&self) -> u64 {
            53
        }
        fn sqrt(&self) -> Self {
            let r = (self.re * self.re + self.im * self.im).sqrt();
            let re = ((r + self.re) / 2.0).sqrt();
            let im = ((r - self.re) / 2.0).sqrt() * self.im.signum().max(0.0).mul_add(2.0, -1.0);
            Cpx::new(re, im)
        }
        fn exp(&self) -> Self {
            let e = self.re.exp();
            Cpx::new(e * self.im.cos(), e * self.im.sin())
        }
        fn ln(&self) -> Self {
            let r = (self.re * self.re + self.im * self.im).sqrt();
            Cpx::new(r.ln(), self.im.atan2(self.re))
        }
        fn sin(&self) -> Self {
            Cpx::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
        }
        fn cos(&self) -> Self {
            Cpx::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
        }
        fn with_precision(&self, _precision: u64) -> Self {
            *self
        }
        // pow deliberately NOT overridden: tests exercise the default.
    }

    // ---- A concrete real Ball ------------------------------------------------

    #[derive(Clone, Copy, Debug)]
    struct RealBallF64 {
        c: f64,
        r: f64,
    }
    impl Ball for RealBallF64 {
        type Center = f64;
        type Radius = f64;
        fn center(&self) -> f64 {
            self.c
        }
        fn radius(&self) -> f64 {
            self.r
        }
        fn from_center_radius(c: f64, r: f64) -> Self {
            RealBallF64 { c, r: r.abs() }
        }
        fn contains(&self, point: &f64) -> bool {
            (point - self.c).abs() <= self.r
        }
        fn contains_zero(&self) -> bool {
            self.c.abs() <= self.r
        }
    }

    #[test]
    fn test_realfield_double() {
        let two = Dbl::from_f64(2.0, 53);
        assert!((RealField::sqrt(&two).to_f64() - std::f64::consts::SQRT_2).abs() < 1e-12);
        let pi = Dbl::pi(53);
        assert!((RealField::sin(&pi).to_f64()).abs() < 1e-9);
        assert!((RealField::exp(&Dbl::from_f64(0.0, 53)).to_f64() - 1.0).abs() < 1e-12);
        assert_eq!(RealField::precision(&two), 53);
    }

    #[test]
    fn test_complexfield_double() {
        let z = Cpx::new(3.0, 4.0);
        assert!((z.abs().to_f64() - 5.0).abs() < 1e-12);
        assert_eq!(z.conj(), Cpx::new(3.0, -4.0));
        // (i)^2 = -1
        let i = Cpx::i(53);
        assert_eq!(i * i, -Cpx::one());
        // exp(0) = 1
        assert_eq!(ComplexField::exp(&Cpx::zero()), Cpx::one());
    }

    #[test]
    fn test_ball_contains() {
        let b = RealBallF64::from_center_radius(1.0, 0.5);
        assert_eq!(b.center(), 1.0);
        assert_eq!(b.radius(), 0.5);
        assert!(b.contains(&1.25));
        assert!(!b.contains(&2.0));
        assert!(!b.contains_zero());

        let around0 = RealBallF64::from_center_radius(0.1, 0.2);
        assert!(around0.contains_zero());
    }

    #[test]
    fn test_default_atan2_quadrants() {
        let f = |v: f64| Dbl::from_f64(v, 53);
        // reference: f64's atan2 (same conventions except atan2(0,-x) sign of zero,
        // which Dbl cannot represent — covered by the explicit π case below)
        for &(y, x) in &[
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (0.5, 2.0),
            (2.0, -0.5),
        ] {
            let got = f(y).atan2(&f(x)).to_f64();
            assert!(
                (got - y.atan2(x)).abs() < 1e-12,
                "atan2({y}, {x}): got {got}, want {}",
                y.atan2(x)
            );
        }
        // axes and origin
        assert_eq!(f(0.0).atan2(&f(0.0)).to_f64(), 0.0);
        assert!((f(0.0).atan2(&f(-2.0)).to_f64() - std::f64::consts::PI).abs() < 1e-12);
        assert!((f(3.0).atan2(&f(0.0)).to_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((f(-3.0).atan2(&f(0.0)).to_f64() + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(f(0.0).atan2(&f(2.0)).to_f64(), 0.0);
    }

    #[test]
    fn test_default_real_pow() {
        let f = |v: f64| Dbl::from_f64(v, 53);
        // NB: UFCS because `Ring::pow(u32)` is also in scope (see method docs).
        assert!((RealField::pow(&f(2.0), &f(10.0)).to_f64() - 1024.0).abs() < 1e-9);
        assert!((RealField::pow(&f(9.0), &f(0.5)).to_f64() - 3.0).abs() < 1e-12);
        assert!((RealField::pow(&f(2.0), &f(-1.0)).to_f64() - 0.5).abs() < 1e-12);
        // 0^0 = 1, 0^y = 0
        assert_eq!(RealField::pow(&f(0.0), &f(0.0)).to_f64(), 1.0);
        assert_eq!(RealField::pow(&f(0.0), &f(3.0)).to_f64(), 0.0);
    }

    #[test]
    #[should_panic(expected = "negative base")]
    fn test_default_real_pow_negative_base_panics() {
        let f = |v: f64| Dbl::from_f64(v, 53);
        let _ = RealField::pow(&f(-2.0), &f(0.5));
    }

    #[test]
    fn test_floor_ceil_round_ties() {
        let f = |v: f64| Dbl::from_f64(v, 53);
        assert_eq!(RealField::floor(&f(2.7)).to_f64(), 2.0);
        assert_eq!(RealField::floor(&f(-2.1)).to_f64(), -3.0);
        assert_eq!(RealField::ceil(&f(2.1)).to_f64(), 3.0);
        assert_eq!(RealField::ceil(&f(-2.7)).to_f64(), -2.0);
        // ties away from zero
        assert_eq!(RealField::round(&f(2.5)).to_f64(), 3.0);
        assert_eq!(RealField::round(&f(-2.5)).to_f64(), -3.0);
        assert_eq!(RealField::round(&f(2.4)).to_f64(), 2.0);
    }

    #[test]
    fn test_decimal_string_roundtrip() {
        let x = Dbl::from_f64(3.14159265358979, 53);
        let s = x.to_decimal_string(15);
        let back = Dbl::from_decimal_str(&s, 53).unwrap();
        assert!((back.to_f64() - x.to_f64()).abs() < 1e-13);
        assert!(Dbl::from_decimal_str("not a number", 53).is_err());
    }

    #[test]
    fn test_default_complex_pow() {
        // i^2 = -1 via the default exp(w ln z); UFCS because Ring::pow is in scope
        let i = Cpx::i(53);
        let two = Cpx::new(2.0, 0.0);
        let m1 = ComplexField::pow(&i, &two);
        assert!((m1.re().to_f64() + 1.0).abs() < 1e-12);
        assert!(m1.im().to_f64().abs() < 1e-12);
        // 0^0 = 1
        let z0 = Cpx::zero();
        assert_eq!(ComplexField::pow(&z0, &Cpx::zero()), Cpx::one());
        assert_eq!(ComplexField::pow(&z0, &two), Cpx::zero());
        // with_precision is identity on the double
        assert_eq!(ComplexField::with_precision(&i, 100), i);
    }
}

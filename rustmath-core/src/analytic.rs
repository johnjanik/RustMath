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
//! Purely additive.

use crate::ordering::OrderedField;
use crate::Field;

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
}

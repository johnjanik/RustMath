//! `BigComplex`: arbitrary-precision complex numbers over [`BigFloat`].
//!
//! MAGMA source: Handbook chapter 25 (real and complex fields).
//!
//! `BigComplex` pairs two pure-Rust [`BigFloat`] components. It implements the
//! full `Ring`/`Field` tower plus the core
//! [`ComplexField`](rustmath_core::analytic::ComplexField) surface (re, im,
//! conj, abs, arg, and the transcendental functions), all with **no** external
//! float backend. The rug/MPC-backed [`ComplexMPFR`](crate::ComplexMPFR)
//! remains available for callers who want it.

use rustmath_core::analytic::{ComplexField, RealField};
use rustmath_core::ordering::OrderedRing;
use rustmath_core::{
    CommutativeRing, Field, IntegralDomain, MathError, NumericConversion, Result, Ring,
};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::{BigFloat, DEFAULT_PRECISION};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An arbitrary-precision complex number `re + im·i`.
#[derive(Clone, Debug)]
pub struct BigComplex {
    re: BigFloat,
    im: BigFloat,
}

fn bf_int(n: i64, prec: u64) -> BigFloat {
    BigFloat::from_integer(&Integer::from(n), prec)
}

/// `cosh(x) = (e^x + e^{-x})/2`.
fn cosh(x: &BigFloat) -> BigFloat {
    let prec = x.prec();
    let ex = x.clone().exp();
    let enx = (-x.clone()).exp();
    (ex + enx) / bf_int(2, prec)
}

/// `sinh(x) = (e^x - e^{-x})/2`.
fn sinh(x: &BigFloat) -> BigFloat {
    let prec = x.prec();
    let ex = x.clone().exp();
    let enx = (-x.clone()).exp();
    (ex - enx) / bf_int(2, prec)
}

/// `atan2(y, x)`, the argument of `x + y·i`.
fn atan2(y: &BigFloat, x: &BigFloat) -> BigFloat {
    let prec = x.prec().max(y.prec());
    let pi = <BigFloat as RealField>::pi(prec);
    match (OrderedRing::sign(x), OrderedRing::sign(y)) {
        (0, 0) => BigFloat::zero_prec(prec),
        (0, s) if s > 0 => pi / bf_int(2, prec),
        (0, _) => -(pi / bf_int(2, prec)),
        (sx, sy) => {
            let base = (y.clone() / x.clone()).atan();
            if sx > 0 {
                base
            } else if sy >= 0 {
                base + pi
            } else {
                base - pi
            }
        }
    }
}

impl BigComplex {
    /// Construct from real and imaginary [`BigFloat`] parts.
    pub fn new(re: BigFloat, im: BigFloat) -> Self {
        BigComplex { re, im }
    }

    /// Zero at the given precision.
    pub fn zero_prec(prec: u64) -> Self {
        BigComplex {
            re: BigFloat::zero_prec(prec),
            im: BigFloat::zero_prec(prec),
        }
    }

    /// One at the given precision.
    pub fn one_prec(prec: u64) -> Self {
        BigComplex {
            re: BigFloat::one_prec(prec),
            im: BigFloat::zero_prec(prec),
        }
    }

    /// The imaginary unit `i` at the given precision.
    pub fn i_prec(prec: u64) -> Self {
        BigComplex {
            re: BigFloat::zero_prec(prec),
            im: BigFloat::one_prec(prec),
        }
    }

    /// Construct from an integer at the given precision.
    pub fn from_integer(n: &Integer, prec: u64) -> Self {
        BigComplex {
            re: BigFloat::from_integer(n, prec),
            im: BigFloat::zero_prec(prec),
        }
    }

    /// Construct from a rational at the given precision.
    pub fn from_rational(r: &Rational, prec: u64) -> Self {
        BigComplex {
            re: BigFloat::from_rational(r, prec),
            im: BigFloat::zero_prec(prec),
        }
    }

    /// The working precision (bits).
    pub fn prec(&self) -> u64 {
        self.re.prec().max(self.im.prec())
    }

    /// The real part.
    pub fn real(&self) -> BigFloat {
        self.re.clone()
    }

    /// The imaginary part.
    pub fn imag(&self) -> BigFloat {
        self.im.clone()
    }

    fn norm_sq(&self) -> BigFloat {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }
}

impl Add for BigComplex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        BigComplex {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}
impl Sub for BigComplex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        BigComplex {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}
impl Mul for BigComplex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        BigComplex {
            re: self.re.clone() * o.re.clone() - self.im.clone() * o.im.clone(),
            im: self.re * o.im + self.im * o.re,
        }
    }
}
impl Div for BigComplex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let d = o.norm_sq();
        BigComplex {
            re: (self.re.clone() * o.re.clone() + self.im.clone() * o.im.clone()) / d.clone(),
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }
}
impl Neg for BigComplex {
    type Output = Self;
    fn neg(self) -> Self {
        BigComplex {
            re: -self.re,
            im: -self.im,
        }
    }
}
impl PartialEq for BigComplex {
    fn eq(&self, o: &Self) -> bool {
        self.re == o.re && self.im == o.im
    }
}
impl fmt::Display for BigComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*i", self.re, self.im)
    }
}

impl From<i64> for BigComplex {
    fn from(n: i64) -> Self {
        Self::from_integer(&Integer::from(n), DEFAULT_PRECISION)
    }
}

impl Ring for BigComplex {
    fn zero() -> Self {
        Self::zero_prec(DEFAULT_PRECISION)
    }
    fn one() -> Self {
        Self::one_prec(DEFAULT_PRECISION)
    }
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
    fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }
}
impl CommutativeRing for BigComplex {}
impl IntegralDomain for BigComplex {}
impl Field for BigComplex {
    fn inverse(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let d = self.norm_sq();
        Ok(BigComplex {
            re: self.re.clone() / d.clone(),
            im: -self.im.clone() / d,
        })
    }
}

impl ComplexField for BigComplex {
    type Real = BigFloat;

    fn re(&self) -> BigFloat {
        self.re.clone()
    }
    fn im(&self) -> BigFloat {
        self.im.clone()
    }
    fn conj(&self) -> Self {
        BigComplex {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }
    fn abs(&self) -> BigFloat {
        self.norm_sq().sqrt()
    }
    fn arg(&self) -> BigFloat {
        atan2(&self.im, &self.re)
    }
    fn from_real_imag(re: BigFloat, im: BigFloat) -> Self {
        BigComplex { re, im }
    }
    fn i(precision: u64) -> Self {
        Self::i_prec(precision)
    }
    fn precision(&self) -> u64 {
        self.prec()
    }
    fn sqrt(&self) -> Self {
        let prec = self.prec();
        let two = bf_int(2, prec);
        let r = self.abs();
        if self.im.is_zero() && OrderedRing::sign(&self.re) < 0 {
            // pure negative real: sqrt = i*sqrt(|re|)
            return BigComplex {
                re: BigFloat::zero_prec(prec),
                im: OrderedRing::abs(&self.re).sqrt(),
            };
        }
        let re = ((r.clone() + self.re.clone()) / two.clone()).sqrt();
        let mut im = ((r - self.re.clone()) / two).sqrt();
        if OrderedRing::sign(&self.im) < 0 {
            im = -im;
        }
        BigComplex { re, im }
    }
    fn exp(&self) -> Self {
        let e = self.re.clone().exp();
        BigComplex {
            re: e.clone() * self.im.clone().cos(),
            im: e * self.im.clone().sin(),
        }
    }
    fn ln(&self) -> Self {
        BigComplex {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }
    fn sin(&self) -> Self {
        BigComplex {
            re: self.re.clone().sin() * cosh(&self.im),
            im: self.re.clone().cos() * sinh(&self.im),
        }
    }
    fn cos(&self) -> Self {
        BigComplex {
            re: self.re.clone().cos() * cosh(&self.im),
            im: -(self.re.clone().sin() * sinh(&self.im)),
        }
    }
}

impl NumericConversion for BigComplex {
    fn from_i64(n: i64) -> Self {
        Self::from_integer(&Integer::from(n), DEFAULT_PRECISION)
    }
    fn from_u64(n: u64) -> Self {
        Self::from_integer(&Integer::from(n), DEFAULT_PRECISION)
    }
    fn to_i64(&self) -> Option<i64> {
        if self.im.is_zero() {
            NumericConversion::to_i64(&self.re)
        } else {
            None
        }
    }
    fn to_u64(&self) -> Option<u64> {
        if self.im.is_zero() {
            NumericConversion::to_u64(&self.re)
        } else {
            None
        }
    }
    fn to_usize(&self) -> Option<usize> {
        self.to_u64().map(|n| n as usize)
    }
    fn to_f64(&self) -> Option<f64> {
        if self.im.is_zero() {
            Some(self.re.to_f64())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn test_arithmetic() {
        let p = 128;
        let z = BigComplex::new(BigFloat::from_integer(&Integer::from(3), p), BigFloat::from_integer(&Integer::from(4), p));
        let w = BigComplex::new(BigFloat::from_integer(&Integer::from(1), p), BigFloat::from_integer(&Integer::from(2), p));
        // (3+4i)(1+2i) = -5 + 10i
        let prod = z.clone() * w.clone();
        assert!(approx(prod.real().to_f64(), -5.0, 1e-30));
        assert!(approx(prod.imag().to_f64(), 10.0, 1e-30));
        // |3+4i| = 5
        assert!(approx(z.abs().to_f64(), 5.0, 1e-30));
    }

    #[test]
    fn test_inverse_and_div() {
        let p = 128;
        let z = BigComplex::new(bf_int(1, p), bf_int(1, p)); // 1+i
        let inv = z.inverse().unwrap();
        let one = z.clone() * inv;
        assert!(one.is_one());
    }

    #[test]
    fn test_i_squared_is_minus_one() {
        let p = 100;
        let i = BigComplex::i_prec(p);
        let m1 = i.clone() * i;
        assert_eq!(m1, -BigComplex::one_prec(p));
    }

    #[test]
    fn test_sqrt_of_minus_one_is_i() {
        let p = 120;
        let neg1 = -BigComplex::one_prec(p);
        let s = ComplexField::sqrt(&neg1);
        assert!(approx(s.real().to_f64(), 0.0, 1e-25));
        assert!(approx(s.imag().to_f64(), 1.0, 1e-25));
    }

    #[test]
    fn test_exp_i_pi_is_minus_one() {
        // Euler's identity: exp(i*pi) = -1
        let p = 160;
        let pi = <BigFloat as RealField>::pi(p);
        let ipi = BigComplex::new(BigFloat::zero_prec(p), pi);
        let val = ComplexField::exp(&ipi);
        assert!(approx(val.real().to_f64(), -1.0, 1e-14));
        assert!(approx(val.imag().to_f64(), 0.0, 1e-14));
    }

    #[test]
    fn test_ln_exp_roundtrip() {
        let p = 140;
        let z = BigComplex::new(bf_int(2, p), bf_int(3, p));
        let back = ComplexField::exp(&ComplexField::ln(&z));
        assert!(approx(back.real().to_f64(), 2.0, 1e-12));
        assert!(approx(back.imag().to_f64(), 3.0, 1e-12));
    }

    #[test]
    fn test_conj_and_arg() {
        let p = 120;
        let z = BigComplex::new(bf_int(1, p), bf_int(1, p)); // 1+i
        assert_eq!(z.conj(), BigComplex::new(bf_int(1, p), bf_int(-1, p)));
        // arg(1+i) = pi/4
        let expected = std::f64::consts::FRAC_PI_4;
        assert!(approx(z.arg().to_f64(), expected, 1e-14));
    }
}

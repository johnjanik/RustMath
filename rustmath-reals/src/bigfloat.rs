//! `BigFloat`: a self-contained, pure-Rust arbitrary-precision real number.
//!
//! MAGMA source: Handbook chapter 25 (real and complex fields).
//!
//! # Design
//!
//! A `BigFloat` is a *binary* floating-point value
//!
//! ```text
//!     value = mantissa · 2^exponent
//! ```
//!
//! where `mantissa` is a signed [`rustmath_integers::Integer`] and `exponent`
//! is an `i64`, together with a target `precision` (in bits). Every result is
//! rounded to `precision` significant bits with round-half-to-even, and stored
//! in a canonical form (odd mantissa / trailing zero bits stripped) so that
//! equal dyadic values have identical representations.
//!
//! This is a **native** implementation built only on `rustmath-integers`: it
//! pulls in **no** external float backend, is 100% safe Rust, and builds
//! offline. It is the portable default for `RealField`; the rug/MPFR-backed
//! [`RealMPFR`](crate::RealMPFR) remains available for callers that want it.
//!
//! Binary operations combine at the **larger** of the two operands' precisions
//! so precision never silently degrades.

use rustmath_core::analytic::RealField;
use rustmath_core::ordering::{OrderedField, OrderedRing};
use rustmath_core::parent::Parent;
use rustmath_core::{
    CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, NumericConversion, Result,
    Ring,
};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Default precision (bits) used by precision-free constructors such as
/// [`Ring::zero`]/[`Ring::one`] and [`NumericConversion::from_i64`].
pub const DEFAULT_PRECISION: u64 = 53;

/// Extra guard bits used internally for transcendental evaluation.
const GUARD: u64 = 24;

/// A pure-Rust arbitrary-precision binary floating-point real number.
#[derive(Clone, Debug)]
pub struct BigFloat {
    /// Signed significand.
    mantissa: Integer,
    /// Binary exponent: `value = mantissa * 2^exponent`.
    exponent: i64,
    /// Target precision in bits.
    precision: u64,
}

// ------------------------------------------------------------------ helpers --

fn two_pow(k: u64) -> Integer {
    Integer::from(2).pow(k as u32)
}

impl BigFloat {
    /// Build a value `mantissa · 2^exponent` rounded to `precision` bits.
    fn from_parts(mantissa: Integer, exponent: i64, precision: u64) -> Self {
        let prec = precision.max(1);
        let (m, e) = normalize(mantissa, exponent, prec);
        BigFloat {
            mantissa: m,
            exponent: e,
            precision: prec,
        }
    }

    /// The zero value at a given precision.
    pub fn zero_prec(precision: u64) -> Self {
        BigFloat {
            mantissa: Integer::zero(),
            exponent: 0,
            precision: precision.max(1),
        }
    }

    /// The one value at a given precision.
    pub fn one_prec(precision: u64) -> Self {
        BigFloat {
            mantissa: Integer::one(),
            exponent: 0,
            precision: precision.max(1),
        }
    }

    /// Construct from an [`Integer`] at the given precision.
    pub fn from_integer(n: &Integer, precision: u64) -> Self {
        Self::from_parts(n.clone(), 0, precision)
    }

    /// Construct the nearest `precision`-bit value to a [`Rational`].
    pub fn from_rational(r: &Rational, precision: u64) -> Self {
        let prec = precision.max(1);
        let num = r.numerator().clone();
        let den = r.denominator().clone();
        if num.is_zero() {
            return Self::zero_prec(prec);
        }
        let sign = num.signum() as i64 * den.signum() as i64;
        let anum = num.abs();
        let aden = den.abs();
        let target = (prec + GUARD) as i64;
        // Choose k so that anum·2^k / aden has ~target bits.
        let k = (target + aden.bit_length() as i64 - anum.bit_length() as i64).max(0) as u64;
        let scaled = anum * two_pow(k);
        let (q, rem) = scaled.div_rem(&aden).unwrap();
        // round-half-even against aden
        let mut q = q;
        let two_r = rem.clone() + rem;
        match two_r.cmp(&aden) {
            Ordering::Greater => q = q + Integer::one(),
            Ordering::Less => {}
            Ordering::Equal => {
                if q.is_odd() {
                    q = q + Integer::one();
                }
            }
        }
        let signed = if sign < 0 { -q } else { q };
        Self::from_parts(signed, -(k as i64), prec)
    }

    /// The precision (bits) carried by this value.
    pub fn prec(&self) -> u64 {
        self.precision
    }

    /// Re-round this value to a new precision.
    pub fn with_precision(&self, precision: u64) -> Self {
        Self::from_parts(self.mantissa.clone(), self.exponent, precision)
    }

    /// Position of the most significant bit (`floor(log2|self|)`), or `None`
    /// for zero.
    fn magnitude_bits(&self) -> Option<i64> {
        if self.mantissa.is_zero() {
            None
        } else {
            Some(self.exponent + self.mantissa.bit_length() as i64 - 1)
        }
    }

    fn combined_prec(&self, other: &Self) -> u64 {
        self.precision.max(other.precision)
    }

    /// Signed comparison of exact values.
    fn cmp_value(&self, other: &Self) -> Ordering {
        let sa = self.mantissa.signum();
        let sb = other.mantissa.signum();
        if sa != sb {
            return sa.cmp(&sb);
        }
        if sa == 0 {
            return Ordering::Equal;
        }
        // same sign, both non-zero: compare magnitudes by aligning exponents
        let e = self.exponent.min(other.exponent);
        let a = self.mantissa.abs() * two_pow((self.exponent - e) as u64);
        let b = other.mantissa.abs() * two_pow((other.exponent - e) as u64);
        let mag = a.cmp(&b);
        if sa < 0 {
            mag.reverse()
        } else {
            mag
        }
    }

    fn add_impl(&self, other: &Self) -> Self {
        let prec = self.combined_prec(other);
        if self.mantissa.is_zero() {
            return other.with_precision(prec);
        }
        if other.mantissa.is_zero() {
            return self.with_precision(prec);
        }
        // Negligibility: if magnitudes differ by more than the working window,
        // the smaller operand cannot affect the rounded result.
        let ma = self.magnitude_bits().unwrap();
        let mb = other.magnitude_bits().unwrap();
        let window = (prec + GUARD + 4) as i64;
        if ma - mb > window {
            return self.with_precision(prec);
        }
        if mb - ma > window {
            return other.with_precision(prec);
        }
        let e = self.exponent.min(other.exponent);
        let a = self.mantissa.clone() * two_pow((self.exponent - e) as u64);
        let b = other.mantissa.clone() * two_pow((other.exponent - e) as u64);
        Self::from_parts(a + b, e, prec)
    }

    fn mul_impl(&self, other: &Self) -> Self {
        let prec = self.combined_prec(other);
        Self::from_parts(
            self.mantissa.clone() * other.mantissa.clone(),
            self.exponent + other.exponent,
            prec,
        )
    }

    fn div_impl(&self, other: &Self) -> Result<Self> {
        if other.mantissa.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let prec = self.combined_prec(other);
        if self.mantissa.is_zero() {
            return Ok(Self::zero_prec(prec));
        }
        let sign = self.mantissa.signum() as i64 * other.mantissa.signum() as i64;
        let anum = self.mantissa.abs();
        let aden = other.mantissa.abs();
        let target = (prec + GUARD) as i64;
        let k = (target + aden.bit_length() as i64 - anum.bit_length() as i64).max(0) as u64;
        let scaled = anum * two_pow(k);
        let (q, rem) = scaled.div_rem(&aden).unwrap();
        let mut q = q;
        let two_r = rem.clone() + rem;
        match two_r.cmp(&aden) {
            Ordering::Greater => q = q + Integer::one(),
            Ordering::Less => {}
            Ordering::Equal => {
                if q.is_odd() {
                    q = q + Integer::one();
                }
            }
        }
        let signed = if sign < 0 { -q } else { q };
        Ok(Self::from_parts(
            signed,
            self.exponent - other.exponent - k as i64,
            prec,
        ))
    }

    /// Negation that borrows (does not consume `self`).
    fn negated(&self) -> Self {
        BigFloat {
            mantissa: -self.mantissa.clone(),
            exponent: self.exponent,
            precision: self.precision,
        }
    }

    /// Best-effort `f64` value (inherent, so `.to_f64()` is unambiguous even
    /// though both [`RealField`] and [`NumericConversion`] also define it).
    pub fn to_f64(&self) -> f64 {
        <Self as RealField>::to_f64(self)
    }

    /// Multiply/divide by `2^n` exactly (no rounding).
    fn scale2(&self, n: i64) -> Self {
        if self.mantissa.is_zero() {
            return self.clone();
        }
        BigFloat {
            mantissa: self.mantissa.clone(),
            exponent: self.exponent + n,
            precision: self.precision,
        }
    }

    fn is_negative_val(&self) -> bool {
        self.mantissa.signum() < 0
    }
}

/// Reduce `mantissa · 2^exp` to at most `prec` significant bits with
/// round-half-to-even, then strip trailing zero bits into a canonical form.
fn normalize(mantissa: Integer, exp: i64, prec: u64) -> (Integer, i64) {
    if mantissa.is_zero() {
        return (Integer::zero(), 0);
    }
    let sign = mantissa.signum();
    let mut m = mantissa.abs();
    let mut e = exp;
    let bits = m.bit_length();
    if bits > prec {
        let drop = bits - prec;
        let divisor = two_pow(drop);
        let (q, r) = m.div_rem(&divisor).unwrap();
        m = q;
        let two_r = r.clone() + r;
        match two_r.cmp(&divisor) {
            Ordering::Greater => m = m + Integer::one(),
            Ordering::Less => {}
            Ordering::Equal => {
                if m.is_odd() {
                    m = m + Integer::one();
                }
            }
        }
        e += drop as i64;
    }
    // strip trailing zero bits
    let two = Integer::from(2);
    while !m.is_zero() && m.is_even() {
        m = m / two.clone();
        e += 1;
    }
    let signed = if sign < 0 { -m } else { m };
    (signed, e)
}

// -------------------------------------------------------- transcendentals ----

/// `atanh(x) = x + x^3/3 + x^5/5 + …`, converges for `|x| < 1`.
fn atanh_series(x: &BigFloat, wp: u64) -> BigFloat {
    let x2 = x.mul_impl(x);
    let mut term = x.with_precision(wp);
    let mut sum = term.clone();
    let mut k: i64 = 1;
    let stop = -(wp as i64 + 8);
    loop {
        term = term.mul_impl(&x2);
        k += 2;
        let t = term.div_impl(&BigFloat::from_i64_prec(k, wp)).unwrap();
        sum = sum.add_impl(&t);
        match term.magnitude_bits() {
            None => break,
            Some(mb) if mb < stop => break,
            _ => {}
        }
    }
    sum
}

/// `atan(x) = x - x^3/3 + x^5/5 - …`, for small `|x|`.
fn atan_series(x: &BigFloat, wp: u64) -> BigFloat {
    let x2 = x.mul_impl(x);
    let mut term = x.with_precision(wp);
    let mut sum = term.clone();
    let mut k: i64 = 1;
    let stop = -(wp as i64 + 8);
    loop {
        term = term.mul_impl(&x2);
        k += 2;
        let t = term.div_impl(&BigFloat::from_i64_prec(k, wp)).unwrap();
        if k % 4 == 1 {
            sum = sum.add_impl(&t);
        } else {
            sum = sum.add_impl(&t.neg());
        }
        match term.magnitude_bits() {
            None => break,
            Some(mb) if mb < stop => break,
            _ => {}
        }
    }
    sum
}

fn ln2_at(wp: u64) -> BigFloat {
    // ln 2 = 2 * atanh(1/3)
    let third = BigFloat::from_rational(&Rational::new(Integer::one(), Integer::from(3)).unwrap(), wp);
    atanh_series(&third, wp).scale2(1)
}

fn pi_at(wp: u64) -> BigFloat {
    // Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    let fifth =
        BigFloat::from_rational(&Rational::new(Integer::one(), Integer::from(5)).unwrap(), wp);
    let r239 =
        BigFloat::from_rational(&Rational::new(Integer::one(), Integer::from(239)).unwrap(), wp);
    let a = atan_series(&fifth, wp).mul_impl(&BigFloat::from_i64_prec(16, wp));
    let b = atan_series(&r239, wp).mul_impl(&BigFloat::from_i64_prec(4, wp));
    a.add_impl(&b.neg())
}

/// `exp(x)` at working precision `wp`.
fn exp_at(x: &BigFloat, wp: u64) -> BigFloat {
    if x.mantissa.is_zero() {
        return BigFloat::one_prec(wp);
    }
    // range reduce: y = x / 2^s so that |y| < 1/2
    let mag = x.magnitude_bits().unwrap();
    let s = if mag >= -1 { (mag + 2) as u64 } else { 0 };
    let y = x.with_precision(wp).scale2(-(s as i64));
    // Taylor sum of exp(y)
    let mut term = BigFloat::one_prec(wp);
    let mut sum = term.clone();
    let mut k: i64 = 0;
    let stop = -(wp as i64 + 8);
    loop {
        k += 1;
        term = term.mul_impl(&y).div_impl(&BigFloat::from_i64_prec(k, wp)).unwrap();
        sum = sum.add_impl(&term);
        match term.magnitude_bits() {
            None => break,
            Some(mb) if mb < stop => break,
            _ => {}
        }
    }
    // square s times
    for _ in 0..s {
        sum = sum.mul_impl(&sum);
    }
    sum
}

/// `ln(x)` for `x > 0` at working precision `wp`.
fn ln_at(x: &BigFloat, wp: u64) -> BigFloat {
    // x = 2^msb * y with y in [1, 2)
    let msb = x.magnitude_bits().unwrap();
    let y = x.with_precision(wp).scale2(-msb); // in [1, 2)
    // t = (y - 1)/(y + 1), |t| <= 1/3
    let one = BigFloat::one_prec(wp);
    let t = y.add_impl(&one.negated()).div_impl(&y.add_impl(&one)).unwrap();
    let ln_y = atanh_series(&t, wp).scale2(1);
    let e_ln2 = BigFloat::from_i64_prec(msb, wp).mul_impl(&ln2_at(wp));
    e_ln2.add_impl(&ln_y)
}

/// `sqrt(x)` for `x >= 0` at working precision `wp`.
fn sqrt_at(x: &BigFloat, wp: u64) -> BigFloat {
    if x.mantissa.is_zero() {
        return BigFloat::zero_prec(wp);
    }
    // ensure even exponent
    let (mut m, mut e) = (x.mantissa.abs(), x.exponent);
    if e.rem_euclid(2) != 0 {
        m = m * Integer::from(2);
        e -= 1;
    }
    // scale up so isqrt has enough bits
    let need = 2 * (wp + GUARD) as i64;
    let curbits = m.bit_length() as i64;
    let mut shift2 = 0i64;
    if curbits < need {
        // make (need - curbits) even
        let mut d = need - curbits;
        if d % 2 != 0 {
            d += 1;
        }
        shift2 = d;
        m = m * two_pow(d as u64);
    }
    let r = m.sqrt().unwrap(); // floor sqrt
    let exp = e / 2 - shift2 / 2;
    BigFloat::from_parts(r, exp, wp)
}

/// `(sin x, cos x)` at working precision `wp`.
fn sincos_at(x: &BigFloat, wp: u64) -> (BigFloat, BigFloat) {
    let pi = pi_at(wp);
    let twopi = pi.scale2(1);
    // k = round(x / twopi); r = x - k*twopi in [-pi, pi]
    let ratio = x.with_precision(wp).div_impl(&twopi).unwrap();
    let k = ratio.round_to_integer();
    let r = x
        .with_precision(wp)
        .add_impl(&BigFloat::from_integer(&k, wp).mul_impl(&twopi).neg());
    // halve until small
    let mag = r.magnitude_bits().unwrap_or(i64::MIN);
    let m = if mag >= -1 { (mag + 2) as u64 } else { 0 };
    let theta = r.scale2(-(m as i64));
    // Taylor for sin, cos of theta
    let mut sin = theta.clone();
    let mut cos = BigFloat::one_prec(wp);
    let theta2 = theta.mul_impl(&theta);
    let mut term_sin = theta.clone();
    let mut term_cos = BigFloat::one_prec(wp);
    let stop = -(wp as i64 + 8);
    let mut n: i64 = 1;
    loop {
        // sin next term: multiply by -theta^2 / ((2n)(2n+1))
        let denom_s = BigFloat::from_i64_prec(2 * n, wp).mul_impl(&BigFloat::from_i64_prec(2 * n + 1, wp));
        term_sin = term_sin.mul_impl(&theta2).div_impl(&denom_s).unwrap().neg();
        sin = sin.add_impl(&term_sin);
        // cos next term: multiply by -theta^2 / ((2n-1)(2n))
        let denom_c =
            BigFloat::from_i64_prec(2 * n - 1, wp).mul_impl(&BigFloat::from_i64_prec(2 * n, wp));
        term_cos = term_cos.mul_impl(&theta2).div_impl(&denom_c).unwrap().neg();
        cos = cos.add_impl(&term_cos);
        let done_sin = matches!(term_sin.magnitude_bits(), None) || term_sin.magnitude_bits().map_or(true, |mb| mb < stop);
        let done_cos = matches!(term_cos.magnitude_bits(), None) || term_cos.magnitude_bits().map_or(true, |mb| mb < stop);
        if done_sin && done_cos {
            break;
        }
        n += 1;
    }
    // double-angle m times
    for _ in 0..m {
        let s2 = sin.mul_impl(&cos).scale2(1); // 2 s c
        let c2 = cos.mul_impl(&cos).add_impl(&sin.mul_impl(&sin).neg()); // c^2 - s^2
        sin = s2;
        cos = c2;
    }
    (sin, cos)
}

impl BigFloat {
    fn from_i64_prec(n: i64, prec: u64) -> Self {
        Self::from_parts(Integer::from(n), 0, prec)
    }

    /// Round to the nearest integer (round-half-up on ties), returning an
    /// [`Integer`].
    fn round_to_integer(&self) -> Integer {
        if self.mantissa.is_zero() {
            return Integer::zero();
        }
        if self.exponent >= 0 {
            return self.mantissa.clone() * two_pow(self.exponent as u64);
        }
        let shift = (-self.exponent) as u64;
        let divisor = two_pow(shift);
        let (q, r) = self.mantissa.div_rem(&divisor).unwrap();
        // r has the sign of the mantissa (truncated division)
        let two_r = (r.clone() + r).abs();
        if two_r.cmp(&divisor) != Ordering::Less {
            if self.mantissa.signum() < 0 {
                q - Integer::one()
            } else {
                q + Integer::one()
            }
        } else {
            q
        }
    }
}

// -------------------------------------------------------------- operators ----

impl Add for BigFloat {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.add_impl(&other)
    }
}
impl Sub for BigFloat {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.add_impl(&other.neg())
    }
}
impl Mul for BigFloat {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.mul_impl(&other)
    }
}
impl Div for BigFloat {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.div_impl(&other)
            .expect("BigFloat division by zero")
    }
}
impl Neg for BigFloat {
    type Output = Self;
    fn neg(self) -> Self {
        BigFloat {
            mantissa: -self.mantissa,
            exponent: self.exponent,
            precision: self.precision,
        }
    }
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_value(other) == Ordering::Equal
    }
}
impl PartialOrd for BigFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp_value(other))
    }
}

impl fmt::Display for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

impl From<i64> for BigFloat {
    fn from(n: i64) -> Self {
        Self::from_parts(Integer::from(n), 0, DEFAULT_PRECISION)
    }
}
impl From<i32> for BigFloat {
    fn from(n: i32) -> Self {
        Self::from_parts(Integer::from(n), 0, DEFAULT_PRECISION)
    }
}
impl From<f64> for BigFloat {
    fn from(x: f64) -> Self {
        <Self as RealField>::from_f64(x, DEFAULT_PRECISION)
    }
}

// ------------------------------------------------------------- trait tower ---

impl Ring for BigFloat {
    fn zero() -> Self {
        Self::zero_prec(DEFAULT_PRECISION)
    }
    fn one() -> Self {
        Self::one_prec(DEFAULT_PRECISION)
    }
    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }
    fn is_one(&self) -> bool {
        self.mantissa.is_one() && self.exponent == 0
    }
}
impl CommutativeRing for BigFloat {}
impl IntegralDomain for BigFloat {}
impl Field for BigFloat {
    fn inverse(&self) -> Result<Self> {
        BigFloat::one_prec(self.precision).div_impl(self)
    }
    fn divide(&self, other: &Self) -> Result<Self> {
        self.div_impl(other)
    }
}

impl OrderedRing for BigFloat {
    fn sign(&self) -> i32 {
        self.mantissa.signum() as i32
    }
    fn abs(&self) -> Self {
        BigFloat {
            mantissa: self.mantissa.abs(),
            exponent: self.exponent,
            precision: self.precision,
        }
    }
}
impl OrderedField for BigFloat {}

impl RealField for BigFloat {
    fn precision(&self) -> u64 {
        self.precision
    }

    fn from_f64(x: f64, precision: u64) -> Self {
        let prec = precision.max(1);
        if x == 0.0 {
            return Self::zero_prec(prec);
        }
        if !x.is_finite() {
            // No infinities in this model; saturate to zero mantissa.
            return Self::zero_prec(prec);
        }
        let bits = x.to_bits();
        let sign = if (bits >> 63) & 1 == 1 { -1i64 } else { 1 };
        let exp_field = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & 0x000f_ffff_ffff_ffff;
        let (mant, e) = if exp_field == 0 {
            // subnormal
            (frac, -1074i64)
        } else {
            (frac | 0x0010_0000_0000_0000, exp_field - 1075)
        };
        let m = Integer::from(mant) * Integer::from(sign);
        Self::from_parts(m, e, prec)
    }

    fn to_f64(&self) -> f64 {
        if self.mantissa.is_zero() {
            return 0.0;
        }
        // Reduce mantissa to ~60 bits to avoid intermediate overflow.
        let bits = self.mantissa.bit_length() as i64;
        let shift = if bits > 60 { (bits - 60) as u64 } else { 0 };
        let reduced = if shift > 0 {
            self.mantissa.clone() / two_pow(shift)
        } else {
            self.mantissa.clone()
        };
        let m = reduced.to_f64().unwrap_or(f64::NAN);
        let e = self.exponent + shift as i64;
        m * (e as f64).exp2()
    }

    fn pi(precision: u64) -> Self {
        let wp = precision + GUARD;
        pi_at(wp).with_precision(precision)
    }

    fn e(precision: u64) -> Self {
        let wp = precision + GUARD;
        exp_at(&BigFloat::one_prec(wp), wp).with_precision(precision)
    }

    fn sqrt(&self) -> Self {
        if self.is_negative_val() {
            // domain error: return zero for the real model
            return Self::zero_prec(self.precision);
        }
        let wp = self.precision + GUARD;
        sqrt_at(&self.with_precision(wp), wp).with_precision(self.precision)
    }

    fn exp(&self) -> Self {
        let wp = self.precision + GUARD;
        exp_at(&self.with_precision(wp), wp).with_precision(self.precision)
    }

    fn ln(&self) -> Self {
        let wp = self.precision + GUARD;
        ln_at(&self.with_precision(wp), wp).with_precision(self.precision)
    }

    fn sin(&self) -> Self {
        let wp = self.precision + GUARD;
        sincos_at(&self.with_precision(wp), wp).0.with_precision(self.precision)
    }

    fn cos(&self) -> Self {
        let wp = self.precision + GUARD;
        sincos_at(&self.with_precision(wp), wp).1.with_precision(self.precision)
    }

    fn atan(&self) -> Self {
        let wp = self.precision + GUARD;
        let prec = self.precision;
        let neg = self.is_negative_val();
        let a = self.with_precision(wp).abs();
        let one = BigFloat::one_prec(wp);
        // reduce a <= 1 via atan(a) = pi/2 - atan(1/a)
        let (a, complement) = if a.cmp_value(&one) == Ordering::Greater {
            (one.clone().div_impl(&a).unwrap(), true)
        } else {
            (a, false)
        };
        // shrink argument: atan(a) = 2 atan(a / (1 + sqrt(1+a^2)))
        let mut a = a;
        let mut r = 0u32;
        let half = BigFloat::from_rational(
            &Rational::new(Integer::one(), Integer::from(2)).unwrap(),
            wp,
        );
        while a.cmp_value(&half) == Ordering::Greater {
            let denom = one.add_impl(&sqrt_at(&one.add_impl(&a.mul_impl(&a)), wp));
            a = a.div_impl(&denom).unwrap();
            r += 1;
        }
        let mut result = atan_series(&a, wp);
        for _ in 0..r {
            result = result.scale2(1);
        }
        if complement {
            let pi_half = pi_at(wp).scale2(-1);
            result = pi_half.add_impl(&result.neg());
        }
        if neg {
            result = result.neg();
        }
        result.with_precision(prec)
    }
}

impl NumericConversion for BigFloat {
    fn from_i64(n: i64) -> Self {
        Self::from_parts(Integer::from(n), 0, DEFAULT_PRECISION)
    }
    fn from_u64(n: u64) -> Self {
        Self::from_parts(Integer::from(n), 0, DEFAULT_PRECISION)
    }
    fn to_i64(&self) -> Option<i64> {
        let i = self.round_to_integer();
        i.to_f64().and_then(|f| {
            if f.is_finite() && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                Some(i.to_i64())
            } else {
                None
            }
        })
    }
    fn to_u64(&self) -> Option<u64> {
        let v = self.to_f64();
        if v.is_finite() && v >= 0.0 && v <= u64::MAX as f64 {
            Some(v as u64)
        } else {
            None
        }
    }
    fn to_usize(&self) -> Option<usize> {
        self.to_u64().map(|n| n as usize)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(<Self as RealField>::to_f64(self))
    }
}

// ---------------------------------------------------------- parent (field) ---

/// A precision-carrying parent for [`BigFloat`], mirroring MAGMA's
/// `RealField(prec)`.
///
/// Named `BigFloatField` (not `RealField`) to avoid clashing with the
/// pre-existing f64 [`RealField`](crate::real::RealField) factory struct in this
/// crate.
#[derive(Clone, Debug, PartialEq)]
pub struct BigFloatField {
    precision: u64,
}

impl BigFloatField {
    /// A real field of the given precision (bits).
    pub fn new(precision: u64) -> Self {
        BigFloatField {
            precision: precision.max(1),
        }
    }
    /// The precision (bits).
    pub fn precision(&self) -> u64 {
        self.precision
    }
    /// Zero in this field.
    pub fn zero(&self) -> BigFloat {
        BigFloat::zero_prec(self.precision)
    }
    /// One in this field.
    pub fn one(&self) -> BigFloat {
        BigFloat::one_prec(self.precision)
    }
    /// Coerce an integer into this field.
    pub fn from_integer(&self, n: &Integer) -> BigFloat {
        BigFloat::from_integer(n, self.precision)
    }
    /// Coerce a rational into this field.
    pub fn from_rational(&self, r: &Rational) -> BigFloat {
        BigFloat::from_rational(r, self.precision)
    }
    /// Approximate an `f64` in this field.
    pub fn from_f64(&self, x: f64) -> BigFloat {
        <BigFloat as RealField>::from_f64(x, self.precision)
    }
    /// `π` at this field's precision.
    pub fn pi(&self) -> BigFloat {
        <BigFloat as RealField>::pi(self.precision)
    }
    /// `e` at this field's precision.
    pub fn e(&self) -> BigFloat {
        <BigFloat as RealField>::e(self.precision)
    }
}

impl Parent for BigFloatField {
    type Element = BigFloat;
    fn contains(&self, _element: &BigFloat) -> bool {
        true
    }
    fn zero(&self) -> Option<BigFloat> {
        Some(BigFloat::zero_prec(self.precision))
    }
    fn one(&self) -> Option<BigFloat> {
        Some(BigFloat::one_prec(self.precision))
    }
    fn name(&self) -> String {
        format!("RealField({})", self.precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn test_basic_arithmetic() {
        let p = 128;
        let a = BigFloat::from_integer(&Integer::from(3), p);
        let b = BigFloat::from_integer(&Integer::from(4), p);
        assert!(approx((a.clone() + b.clone()).to_f64(), 7.0, 1e-30));
        assert!(approx((a.clone() * b.clone()).to_f64(), 12.0, 1e-30));
        assert!(approx((a.clone() - b.clone()).to_f64(), -1.0, 1e-30));
        assert!(approx((b / a).to_f64(), 4.0 / 3.0, 1e-30));
    }

    #[test]
    fn test_from_rational_exact_third() {
        let p = 200;
        let third = BigFloat::from_rational(
            &Rational::new(Integer::from(1), Integer::from(3)).unwrap(),
            p,
        );
        // 3 * (1/3) should round-trip very close to 1
        let one = third.clone() * BigFloat::from_integer(&Integer::from(3), p);
        assert!(approx(one.to_f64(), 1.0, 1e-40));
    }

    #[test]
    fn test_sqrt2() {
        let p = 200;
        let two = BigFloat::from_integer(&Integer::from(2), p);
        let s = RealField::sqrt(&two);
        // s*s == 2 to high precision
        let sq = s.clone() * s.clone();
        assert!(approx(sq.to_f64(), 2.0, 1e-40));
        assert!(approx(s.to_f64(), std::f64::consts::SQRT_2, 1e-14));
    }

    #[test]
    fn test_exp_ln_roundtrip() {
        let p = 160;
        let x = BigFloat::from_rational(
            &Rational::new(Integer::from(7), Integer::from(5)).unwrap(),
            p,
        );
        let back = RealField::ln(&RealField::exp(&x));
        assert!(approx(back.to_f64(), 1.4, 1e-14));
        // e = exp(1)
        let e = <BigFloat as RealField>::e(p);
        assert!(approx(e.to_f64(), std::f64::consts::E, 1e-14));
    }

    #[test]
    fn test_pi_and_trig() {
        let p = 160;
        let pi = <BigFloat as RealField>::pi(p);
        assert!(approx(pi.to_f64(), std::f64::consts::PI, 1e-14));
        // sin(pi/6) = 1/2, cos(pi/3) = 1/2
        let sixth = pi.clone().div_impl(&BigFloat::from_integer(&Integer::from(6), p)).unwrap();
        assert!(approx(RealField::sin(&sixth).to_f64(), 0.5, 1e-14));
        let third = pi.div_impl(&BigFloat::from_integer(&Integer::from(3), p)).unwrap();
        assert!(approx(RealField::cos(&third).to_f64(), 0.5, 1e-14));
    }

    #[test]
    fn test_atan_tan_of_one() {
        let p = 160;
        let one = BigFloat::one_prec(p);
        // atan(1) = pi/4
        let a = RealField::atan(&one);
        let pi4 = <BigFloat as RealField>::pi(p).div_impl(&BigFloat::from_integer(&Integer::from(4), p)).unwrap();
        assert!(approx(a.to_f64(), pi4.to_f64(), 1e-14));
        // atan of a large value
        let big = BigFloat::from_integer(&Integer::from(1000), p);
        let ab = RealField::atan(&big);
        assert!(approx(ab.to_f64(), (1000.0f64).atan(), 1e-12));
    }

    #[test]
    fn test_ordering_and_sign() {
        let p = 64;
        let a = BigFloat::from_rational(&Rational::new(Integer::from(1), Integer::from(3)).unwrap(), p);
        let b = BigFloat::from_rational(&Rational::new(Integer::from(1), Integer::from(2)).unwrap(), p);
        assert!(a < b);
        assert_eq!(OrderedRing::sign(&(a.clone() - b.clone())), -1);
        assert_eq!(OrderedRing::sign(&BigFloat::zero_prec(p)), 0);
        assert!(approx(OrderedRing::abs(&(a - b)).to_f64(), 1.0 / 6.0, 1e-15));
    }

    #[test]
    fn test_parent_field() {
        let rf = BigFloatField::new(100);
        assert_eq!(rf.precision(), 100);
        assert_eq!(rf.name(), "RealField(100)");
        assert!(approx(rf.pi().to_f64(), std::f64::consts::PI, 1e-14));
        let half = rf.from_rational(&Rational::new(Integer::from(1), Integer::from(2)).unwrap());
        assert!(approx(half.to_f64(), 0.5, 1e-30));
        assert!(rf.contains(&half));
    }

    #[test]
    fn test_from_f64_roundtrip() {
        let x = <BigFloat as RealField>::from_f64(3.14159265358979, 60);
        assert!(approx(x.to_f64(), 3.14159265358979, 1e-15));
        let n = <BigFloat as RealField>::from_f64(-42.0, 60);
        assert_eq!(NumericConversion::to_i64(&n), Some(-42));
    }
}

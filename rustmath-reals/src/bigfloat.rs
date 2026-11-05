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
//!
//! # Accuracy
//!
//! The transcendental functions (`sqrt`, `exp`, `ln`, `sin`, `cos`, `atan`,
//! `pi`, `e`, `pow`) evaluate with [`GUARD`] extra bits — plus
//! argument-dependent extra bits where the algorithm demands it (`exp`'s
//! re-squaring stage, `sin`/`cos` argument reduction mod 2π) — and round back
//! to the value's precision. Guaranteed accuracy, enforced against an
//! MPFR oracle in `tests/bigfloat_rug_oracle.rs`:
//!
//! * all listed functions: **relative error ≤ 4 ulp** at the value's
//!   precision `p` (i.e. `|err| ≤ |exact| · 2^(2-p)`), with one exception:
//! * `sin`/`cos` whose exact result is tiny because the argument sits almost
//!   exactly on a zero of the function: when `|result| < 2^-GUARD`, the
//!   guarantee is **absolute** error `≤ 2^-(p+8)` instead.
//!
//! # Finiteness
//!
//! There are no `±∞`/`NaN` values: every `BigFloat` is a finite dyadic
//! number. `from_f64` maps non-finite input to zero (documented on the
//! trait); `ln`/`sqrt` domain violations are per-method documented.

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
    // range reduce: y = x / 2^s so that |y| < 1/2. Each of the s squarings
    // that undo the reduction can double the relative error, so the whole
    // computation runs with s extra guard bits (then rounds back to wp).
    let mag = x.magnitude_bits().unwrap();
    let s = if mag >= -1 { (mag + 2) as u64 } else { 0 };
    let wpe = wp + s + 8;
    let y = x.with_precision(wpe).scale2(-(s as i64));
    // Taylor sum of exp(y)
    let mut term = BigFloat::one_prec(wpe);
    let mut sum = term.clone();
    let mut k: i64 = 0;
    let stop = -(wpe as i64 + 8);
    loop {
        k += 1;
        term = term.mul_impl(&y).div_impl(&BigFloat::from_i64_prec(k, wpe)).unwrap();
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
    sum.with_precision(wp)
}

/// `ln(x)` for `x > 0` at working precision `wp`.
fn ln_at(x: &BigFloat, wp: u64) -> BigFloat {
    let msb = x.magnitude_bits().unwrap();
    if msb == 0 || msb == -1 {
        // x in [1/2, 2): evaluate directly as 2·atanh((x-1)/(x+1)), |t| < 1/3.
        // The split below would compute ln(x) = ±ln2 + ln_y with catastrophic
        // cancellation for x near 1 (e.g. ln(1 - 2^-40) would lose 40 bits);
        // this branch keeps the result *relatively* accurate however small.
        let one = BigFloat::one_prec(wp);
        let xw = x.with_precision(wp);
        let t = xw.add_impl(&one.negated()).div_impl(&xw.add_impl(&one)).unwrap();
        return atanh_series(&t, wp).scale2(1);
    }
    // x = 2^msb * y with y in [1, 2); |msb| >= 1 so msb·ln2 dominates and the
    // final addition cancels at most one bit.
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
    // Argument reduction mod 2π cancels ~mag(x) leading bits (r = x - k·2π
    // with |k·2π| ≈ |x|), so the reduction runs with mag(x) extra bits; the
    // reduced argument r then carries an absolute error ≤ ~2^-(wp+8)
    // regardless of how large x was.
    let xmag = x.magnitude_bits().unwrap_or(0).max(0) as u64;
    let wpr = wp + xmag + 8;
    let pi = pi_at(wpr);
    let twopi = pi.scale2(1);
    // k = round(x / twopi); r = x - k*twopi in [-pi, pi]
    let ratio = x.with_precision(wpr).div_impl(&twopi).unwrap();
    let k = ratio.round_to_integer();
    let r = x
        .with_precision(wpr)
        .add_impl(&BigFloat::from_integer(&k, wpr).mul_impl(&twopi).neg())
        .with_precision(wp);
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

/// Round-half-even integer division `num / den` (both must be positive-`den`).
fn div_round_half_even(num: Integer, den: &Integer) -> Integer {
    let (q, r) = num.div_rem(den).unwrap();
    let mut q = q;
    let two_r = r.clone() + r;
    match two_r.cmp(den) {
        Ordering::Greater => q = q + Integer::one(),
        Ordering::Less => {}
        Ordering::Equal => {
            if q.is_odd() {
                q = q + Integer::one();
            }
        }
    }
    q
}

impl BigFloat {
    fn from_i64_prec(n: i64, prec: u64) -> Self {
        Self::from_parts(Integer::from(n), 0, prec)
    }

    /// The exact `(mantissa, exponent)` pair with `self = mantissa · 2^exponent`
    /// (canonical: odd mantissa, or `(0, 0)` for zero). Lossless export, e.g.
    /// for serialization or cross-checking against another float backend.
    pub fn mantissa_exponent(&self) -> (Integer, i64) {
        (self.mantissa.clone(), self.exponent)
    }

    /// The largest [`Integer`] `<= self` (exact).
    pub fn floor_int(&self) -> Integer {
        if self.mantissa.is_zero() {
            return Integer::zero();
        }
        if self.exponent >= 0 {
            return self.mantissa.clone() * two_pow(self.exponent as u64);
        }
        let divisor = two_pow((-self.exponent) as u64);
        let (q, r) = self.mantissa.div_rem(&divisor).unwrap();
        // div_rem truncates toward zero; floor must go down for negatives.
        if self.mantissa.signum() < 0 && !r.is_zero() {
            q - Integer::one()
        } else {
            q
        }
    }

    /// The smallest [`Integer`] `>= self` (exact).
    pub fn ceil_int(&self) -> Integer {
        if self.mantissa.is_zero() {
            return Integer::zero();
        }
        if self.exponent >= 0 {
            return self.mantissa.clone() * two_pow(self.exponent as u64);
        }
        let divisor = two_pow((-self.exponent) as u64);
        let (q, r) = self.mantissa.div_rem(&divisor).unwrap();
        if self.mantissa.signum() > 0 && !r.is_zero() {
            q + Integer::one()
        } else {
            q
        }
    }

    /// The nearest [`Integer`]; ties round away from zero (exact).
    pub fn round_int(&self) -> Integer {
        self.round_to_integer()
    }

    /// `round(|self| · 10^p)` as an [`Integer`] (round-half-even).
    fn scaled_decimal(&self, p: i64) -> Integer {
        let mut num = self.mantissa.abs();
        let mut den = Integer::one();
        if p >= 0 {
            num = num * Integer::from(10).pow(p as u32);
        } else {
            den = den * Integer::from(10).pow((-p) as u32);
        }
        if self.exponent >= 0 {
            num = num * two_pow(self.exponent as u64);
        } else {
            den = den * two_pow((-self.exponent) as u64);
        }
        div_round_half_even(num, &den)
    }

    /// Render as a decimal string with `digits` significant digits, in
    /// scientific notation `[-]d[.ddd…]e<exp>` (zero prints as `"0"`).
    ///
    /// The printed value is the round-half-even nearest `digits`-digit decimal,
    /// so it differs from `self` by at most half a unit in the last printed
    /// digit, and round-trips through [`Self::from_decimal_str`] when `digits`
    /// covers the precision (`digits >= ceil(prec·log10 2) + 1`).
    pub fn to_decimal_string(&self, digits: usize) -> String {
        let digits = digits.max(1) as i64;
        if self.mantissa.is_zero() {
            return "0".to_string();
        }
        let neg = self.mantissa.signum() < 0;
        let mag = self.magnitude_bits().unwrap();
        // k_est ≈ floor(log10 |self|); p scales to `digits` integer digits.
        let k_est = ((mag as f64) * std::f64::consts::LOG10_2).floor() as i64;
        let mut p = digits - 1 - k_est;
        // The estimate can be off by one (and rounding can carry a digit);
        // adjust until the scaled integer has exactly `digits` digits.
        for _ in 0..4 {
            let n = self.scaled_decimal(p);
            let s = n.to_string();
            let len = s.len() as i64;
            if len == digits {
                let dec_exp = (digits - 1) - p;
                let mantissa_str = if s.len() == 1 {
                    s
                } else {
                    format!("{}.{}", &s[..1], &s[1..])
                };
                return format!("{}{}e{}", if neg { "-" } else { "" }, mantissa_str, dec_exp);
            }
            p += digits - len;
        }
        unreachable!("BigFloat::to_decimal_string: digit scaling failed to converge");
    }

    /// Parse a decimal string — `[+|-] digits [. digits] [(e|E) [+|-] digits]`
    /// — rounding to `precision` bits. Whitespace is trimmed. Errors with
    /// [`MathError::ParseError`] on malformed input (including `inf`/`nan`,
    /// which this model cannot represent).
    pub fn from_decimal_str(s: &str, precision: u64) -> Result<Self> {
        let prec = precision.max(1);
        let t = s.trim();
        let err = || MathError::ParseError(format!("invalid decimal number: {s:?}"));
        let bytes = t.as_bytes();
        let mut i = 0usize;
        let mut negative = false;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            negative = bytes[i] == b'-';
            i += 1;
        }
        let ten = Integer::from(10);
        let mut mant = Integer::zero();
        let mut any_digit = false;
        let mut frac_digits: i64 = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            mant = mant * ten.clone() + Integer::from((bytes[i] - b'0') as i64);
            any_digit = true;
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                mant = mant * ten.clone() + Integer::from((bytes[i] - b'0') as i64);
                frac_digits += 1;
                any_digit = true;
                i += 1;
            }
        }
        if !any_digit {
            return Err(err());
        }
        let mut exp10: i64 = 0;
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            i += 1;
            let exp_str = &t[i..];
            if exp_str.is_empty() {
                return Err(err());
            }
            exp10 = exp_str.parse::<i64>().map_err(|_| err())?;
            i = bytes.len();
        }
        if i != bytes.len() {
            return Err(err());
        }
        if negative {
            mant = -mant;
        }
        let d = exp10 - frac_digits;
        if d >= 0 {
            let scaled = mant * ten.pow(d as u32);
            Ok(Self::from_integer(&scaled, prec))
        } else {
            let den = ten.pow((-d) as u32);
            let r = Rational::new(mant, den)
                .map_err(|e| MathError::ParseError(format!("{s:?}: {e}")))?;
            Ok(Self::from_rational(&r, prec))
        }
    }

    /// Round to the nearest integer (ties away from zero), returning an
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

    fn with_precision(&self, precision: u64) -> Self {
        BigFloat::with_precision(self, precision)
    }

    fn floor(&self) -> Self {
        Self::from_integer(&self.floor_int(), self.precision)
    }

    fn ceil(&self) -> Self {
        Self::from_integer(&self.ceil_int(), self.precision)
    }

    fn round(&self) -> Self {
        Self::from_integer(&self.round_to_integer(), self.precision)
    }

    fn to_decimal_string(&self, digits: usize) -> String {
        BigFloat::to_decimal_string(self, digits)
    }

    fn from_decimal_str(s: &str, precision: u64) -> Result<Self> {
        BigFloat::from_decimal_str(s, precision)
    }

    // atan2: the trait default (atan/pi/sign at combined precision) is used.

    /// Extends the trait default for negative bases with an **integral**
    /// exponent: `x^n = (-1)^n · |x|^n`. A negative base with a non-integral
    /// exponent still panics (no real value; this model has no NaN).
    fn pow(&self, exponent: &Self) -> Self {
        let prec = self.precision.max(exponent.precision);
        if self.mantissa.is_zero() {
            if exponent.is_zero() {
                return Self::one_prec(prec);
            }
            // 0^y := 0 for y != 0 (no infinities in this model, so y < 0 too).
            return Self::zero_prec(prec);
        }
        if self.is_negative_val() {
            let n = exponent.round_to_integer();
            if BigFloat::from_integer(&n, exponent.precision) != *exponent {
                panic!(
                    "BigFloat::pow: negative base with non-integral exponent has no \
                     real value (and BigFloat has no NaN)"
                );
            }
            let magnitude = OrderedRing::abs(self);
            let val = RealField::exp(
                &(exponent.clone() * RealField::ln(&magnitude)),
            );
            return if n.is_odd() { -val } else { val };
        }
        RealField::exp(&(exponent.clone() * RealField::ln(self)))
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
        // Exact: convert the rounded Integer directly, returning None on
        // overflow. (A previous version range-checked through f64, so a
        // value of exactly 2^63 rounded to `i64::MAX as f64`, passed the
        // guard, and then panicked inside `Integer::to_i64()`.)
        use num_traits::ToPrimitive;
        self.round_to_integer().as_bigint().to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        // Exact: convert the rounded Integer directly, same pattern as
        // `to_i64`. (The previous version range-checked through f64, so a
        // value like `u64::MAX` - which is not exactly representable in
        // f64 - could round up to `2^64` as f64, fail the `<= u64::MAX as
        // f64` guard by a hair, and incorrectly return `None` for a value
        // that does fit; other boundary values could round the other way
        // and silently return a truncated/wrong `u64`.)
        use num_traits::ToPrimitive;
        self.round_to_integer().as_bigint().to_u64()
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
    fn test_to_i64_pow63_boundary() {
        // 2^63 does not fit in i64 but converts (via f64) to exactly
        // `i64::MAX as f64` = 2^63, so the old f64-guarded range check
        // passed and Integer::to_i64() panicked. Must be None.
        let x = BigFloat::from_parts(Integer::from(1), 63, 128);
        assert_eq!(x.to_i64(), None);
        // -(2^63) = i64::MIN is exactly representable and fits.
        let y = BigFloat::from_parts(Integer::from(-1), 63, 128);
        assert_eq!(y.to_i64(), Some(i64::MIN));
        // from_i64(i64::MAX) rounds the 63-bit mantissa UP to 2^63 at the
        // 53-bit default precision, so the stored value genuinely does not
        // fit an i64: the honest exact answer is None. (The old code
        // panicked on precisely this round-trip.)
        assert_eq!(BigFloat::from_i64(i64::MAX).to_i64(), None);
        assert_eq!(BigFloat::from_i64(i64::MIN).to_i64(), Some(i64::MIN));
        // A value exact at 53 bits round-trips.
        let v = 1i64 << 62;
        assert_eq!(BigFloat::from_i64(v).to_i64(), Some(v));
    }

    #[test]
    fn test_to_u64_boundaries() {
        // Around 2^60: exact at ample precision, and negative values must
        // be rejected rather than silently wrapped/truncated.
        let two_pow_60 = BigFloat::from_parts(Integer::from(1), 60, 128);
        assert_eq!(two_pow_60.to_u64(), Some(1u64 << 60));

        let neg = BigFloat::from_parts(Integer::from(-1), 60, 128);
        assert_eq!(neg.to_u64(), None);

        // u64::MAX is not exactly representable in f64 (it rounds up to
        // 2^64 as f64), so the old `v <= u64::MAX as f64` guard could
        // reject it, or a nearby value could round the wrong way through
        // f64 and produce a truncated/wrong result. At sufficient
        // precision this must round-trip exactly.
        let max = BigFloat::from_integer(&Integer::from(u64::MAX), 128);
        assert_eq!(max.to_u64(), Some(u64::MAX));

        // One past u64::MAX must not fit.
        let too_big =
            BigFloat::from_integer(&(Integer::from(u64::MAX) + Integer::from(1)), 128);
        assert_eq!(too_big.to_u64(), None);

        // Zero and small values round-trip.
        assert_eq!(BigFloat::from_u64(0).to_u64(), Some(0));
        assert_eq!(BigFloat::from_u64(42).to_u64(), Some(42));

        // `from_u64` at DEFAULT_PRECISION (53 bits) genuinely cannot
        // represent all 64-bit values exactly: rounding u64::MAX (all 64
        // bits set) to 53 significant bits rounds the mantissa up and
        // overflows to exactly 2^64, which honestly no longer fits in a
        // u64 - None is the correct, honest answer here, not a fabricated
        // truncated value.
        let big_53 = BigFloat::from_u64(u64::MAX);
        assert_eq!(big_53.round_to_integer(), Integer::from(2u64).pow(64));
        assert_eq!(big_53.to_u64(), None);
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

    #[test]
    fn test_floor_ceil_round() {
        let p = 96;
        let f = |v: f64| <BigFloat as RealField>::from_f64(v, p);
        assert_eq!(RealField::floor(&f(2.7)).to_f64(), 2.0);
        assert_eq!(RealField::floor(&f(-2.1)).to_f64(), -3.0);
        assert_eq!(RealField::floor(&f(5.0)).to_f64(), 5.0);
        assert_eq!(RealField::ceil(&f(2.1)).to_f64(), 3.0);
        assert_eq!(RealField::ceil(&f(-2.7)).to_f64(), -2.0);
        // ties away from zero
        assert_eq!(RealField::round(&f(2.5)).to_f64(), 3.0);
        assert_eq!(RealField::round(&f(-2.5)).to_f64(), -3.0);
        assert_eq!(RealField::round(&f(2.4)).to_f64(), 2.0);
        // exact Integer variants
        assert_eq!(f(-2.5).floor_int(), Integer::from(-3));
        assert_eq!(f(-2.5).ceil_int(), Integer::from(-2));
        assert_eq!(f(-2.5).round_int(), Integer::from(-3));
    }

    #[test]
    fn test_decimal_string_io() {
        let p = 200;
        let pi = <BigFloat as RealField>::pi(p);
        let s = pi.to_decimal_string(65); // > 200·log10(2) ≈ 60.2 digits
        // 200 bits carry ~60 correct decimal digits; check the first 55.
        assert!(s.starts_with("3.141592653589793238462643383279502884197169399375105820"),
            "got {s}");
        let back = BigFloat::from_decimal_str(&s, p).unwrap();
        // round-trip: |pi - back| < 2^-190
        let diff = OrderedRing::abs(&(pi.clone() - back));
        let bound = BigFloat::one_prec(p).scale2(-190);
        assert!(diff < bound, "diff {}", diff.to_f64());
        // fixed-point, negative, exponent forms parse
        assert!(approx(BigFloat::from_decimal_str("-12.5e-2", 80).unwrap().to_f64(), -0.125, 1e-18));
        assert!(approx(BigFloat::from_decimal_str("+.5", 80).unwrap().to_f64(), 0.5, 1e-18));
        assert_eq!(BigFloat::from_decimal_str("0", 80).unwrap().to_f64(), 0.0);
        // zero prints as "0"
        assert_eq!(BigFloat::zero_prec(80).to_decimal_string(10), "0");
        // malformed inputs error
        for bad in ["", ".", "1e", "1.2.3", "nan", "inf", "12x"] {
            assert!(BigFloat::from_decimal_str(bad, 80).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn test_atan2_default_quadrants() {
        let p = 120;
        let f = |v: f64| <BigFloat as RealField>::from_f64(v, p);
        for &(y, x) in &[
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (0.0, -2.0),
            (3.0, 0.0),
            (-3.0, 0.0),
        ] {
            let got = RealField::atan2(&f(y), &f(x)).to_f64();
            assert!(approx(got, y.atan2(x), 1e-14), "atan2({y},{x}) = {got}");
        }
        assert_eq!(RealField::atan2(&f(0.0), &f(0.0)).to_f64(), 0.0);
    }

    #[test]
    fn test_pow() {
        let p = 128;
        let f = |v: f64| <BigFloat as RealField>::from_f64(v, p);
        assert!(approx(RealField::pow(&f(2.0), &f(10.0)).to_f64(), 1024.0, 1e-30));
        assert!(approx(RealField::pow(&f(9.0), &f(0.5)).to_f64(), 3.0, 1e-30));
        assert!(approx(RealField::pow(&f(2.0), &f(-2.0)).to_f64(), 0.25, 1e-30));
        // negative base, integral exponent (BigFloat extension over the default)
        assert!(approx(RealField::pow(&f(-2.0), &f(3.0)).to_f64(), -8.0, 1e-30));
        assert!(approx(RealField::pow(&f(-2.0), &f(4.0)).to_f64(), 16.0, 1e-30));
        // 0^0 = 1, 0^y = 0
        assert!(RealField::pow(&f(0.0), &f(0.0)).is_one());
        assert!(RealField::pow(&f(0.0), &f(2.0)).is_zero());
    }

    #[test]
    #[should_panic(expected = "non-integral exponent")]
    fn test_pow_negative_base_non_integral_panics() {
        let f = |v: f64| <BigFloat as RealField>::from_f64(v, 64);
        let _ = RealField::pow(&f(-2.0), &f(0.5));
    }

    #[test]
    fn test_with_precision_and_mantissa_exponent() {
        let p = 200;
        let third = BigFloat::from_rational(
            &Rational::new(Integer::from(1), Integer::from(3)).unwrap(),
            p,
        );
        let narrow = RealField::with_precision(&third, 50);
        assert_eq!(narrow.prec(), 50);
        assert!(approx(narrow.to_f64(), 1.0 / 3.0, 1e-14));
        // mantissa_exponent is an exact export
        let (m, e) = narrow.mantissa_exponent();
        let rebuilt = BigFloat::from_integer(&m, 50) * BigFloat::one_prec(50).scale2(e);
        assert_eq!(rebuilt, narrow);
    }
}

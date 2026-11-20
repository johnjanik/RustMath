//! Rigorous midpoint-radius enclosures of real constants for the zero test of the series engine
//! (S1 design §2.3): MPFR values (through rustmath-reals) with a checked exponent window and exact
//! zeros, used only to prove a constant nonzero when the symbolic test is undecided.
//!
//! Specification: MAGMA Handbook ch. 49, "Power, Laurent and Puiseux Series" (§49.1.3, §49.1.4,
//! §49.5), digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`; the S1
//! design (frozen at revision 7rt4 with revision 8), ported from its prototype `proto13`.
//!
//! References: L. Fousse, G. Hanrot, V. Lefevre, P. Pelissier and P. Zimmermann, "MPFR: A
//! multiple-precision binary floating-point library with correct rounding", ACM Trans. Math.
//! Softw. 33(2), 2007, article 13; F. Johansson, "Arb: efficient arbitrary-precision
//! midpoint-radius interval arithmetic", IEEE Trans. Comput. 66(8), 2017, 1281-1292.

// Prototype (revision 2): rigorous midpoint-radius enclosure of a real constant, used only to
// prove a constant nonzero when the symbolic zero test is undecided.
// MPFR (through rustmath-reals RealMPFR) rounds every operation correctly to nearest, so each
// result r is within |r| 2^-p of the exact value of the operation on its (rounded) inputs, PROVIDED
// that no underflow or overflow happens. Revision 2 (review 2, B1) makes that proviso checked:
//   R1 (window): every MPFR value computed here (midpoints, radii, Lipschitz bounds) must be 0 or
//      have its exponent in [-W, W], W = 2^28; otherwise there is no enclosure. MPFR's exponent
//      range is [1 - 2^30, 2^30 - 1], so a product or a sum of values in the window cannot
//      underflow or overflow, and a result outside the window is refused before it is used.
//   R2 (exact zeros): a computed midpoint equal to 0 is accepted only when the exact result is 0
//      (a factor is 0, the summands are exact negatives, the argument is a zero of f). A nonzero
//      exact value that MPFR flushed to 0 is therefore refused.
//   exp, sinh, cosh need |m| + r <= 2^27, so that their results stay in the window.
// Revision 2 (review 2, B2): MPFR sin and cos reduce the argument with as many bits of pi as the
// argument's exponent. They are called only when r < 1 and exponent(m) <= p + 64; otherwise the
// trivial enclosure [-1, 1] is used (tan: no enclosure). The a-priori charge (ball_cost) counts
// every node of the expression evaluated, with sin/cos/tan at precision 2p + 64.

use crate::series_const::Konst;
use crate::{BinaryOp, Expr, UnaryOp};
use rug::Float;
use rustmath_reals::RealMPFR;

/// exponent window of every MPFR value computed by the ball code
pub const WINDOW: i32 = 1 << 28;

/// revision 7rt2 (red team 2 numeric, m1): why a constant has no enclosure at one precision.
/// `Never`: no precision gives one (a non-real or opaque node, a function, an integer exponent
/// above 2^32, a value outside the window, a domain check failed by the whole ball, the budget).
/// `TooWide`: a domain or guard check failed on a ball that still contains a point of the
/// domain (`lo <= 0 < hi` for log, sqrt and real powers; `|m| <= r` for 1/x and tan's cosine;
/// tan's guard; `|m| - r < 1 <= |m| + r` for asin and acos; the exp guard failed with `|m| - r` below it;
/// `lo <= 1 < hi` for acosh), so a higher precision may give one. The zero test and the integer
/// test go on to the next precision on `TooWide` and stop on `Never`. (The prototype records the
/// reason in a thread-local flag set where the check fails; the port returns it.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NoBall {
    Never,
    TooWide,
}
thread_local! {
    static TOO_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
fn too_wide<T>() -> Option<T> {
    TOO_WIDE.with(|c| c.set(true));
    None
}
/// revision 7 (review 7, B1): the erf threshold X_p, the least multiple of 1/2 with
/// X_p^2 >= (p + 16) ln 2: 7.5, 14 and 27 at 64, 256 and 1024 bits (exact in binary)
pub fn erf_threshold(p: u32) -> f64 {
    let t = ((p as f64 + 16.0) * std::f64::consts::LN_2).sqrt();
    (2.0 * t + 1e-9).ceil() / 2.0
}
/// exp, sinh, cosh: |m| + r must not exceed this (exp(2^27) has exponent about 1.94e8 < WINDOW)
pub const EXP_ARG_MAX: f64 = (1u64 << 27) as f64;

#[derive(Clone)]
pub struct Ball {
    pub m: RealMPFR,
    pub r: RealMPFR,
    p: u32,
}

/// R1: x is 0 or has its exponent in [-WINDOW, WINDOW] (NaN and infinities fail)
fn fits(x: &RealMPFR) -> bool {
    let f = x.as_float();
    if f.is_zero() {
        return true;
    }
    match f.get_exp() {
        Some(e) => (-WINDOW..=WINDOW).contains(&e),
        None => false,
    }
}
fn chk(x: RealMPFR) -> Option<RealMPFR> {
    if fits(&x) {
        Some(x)
    } else {
        None
    }
}
fn is_zero(x: &RealMPFR) -> bool {
    x.as_float().is_zero()
}
/// revision 7rt4: d = 2^k (k >= 0), so n/d is exact at p bits when n has at most p bits
fn is_pow2(d: &rustmath_integers::Integer) -> bool {
    // d == 2^(b-1): its trailing zeros are all its bits but the top one
    d.signum() > 0 && d.as_bigint().magnitude().trailing_zeros() == Some(d.bit_length() - 1)
}
/// 2^e exactly (no loop)
fn two_pow(p: u32, e: i32) -> RealMPFR {
    let one = Float::with_val(p.max(2), 1u32);
    RealMPFR::from_float(if e >= 0 {
        one << (e as u32)
    } else {
        one >> ((-e) as u32)
    })
}
fn rz(p: u32) -> RealMPFR {
    RealMPFR::with_val(p, 0.0)
}

impl Ball {
    fn eps(&self) -> RealMPFR {
        two_pow(self.p, -(self.p as i32))
    }
    fn up(&self, x: RealMPFR) -> Option<RealMPFR> {
        // inflate a computed radius to an upper bound
        let f = RealMPFR::with_val(self.p, 1.0) + two_pow(self.p, 4 - self.p as i32);
        chk(&x.abs() * &f)
    }
    /// an exact value (no rounding error), checked against the window
    pub fn exact(p: u32, m: RealMPFR) -> Option<Ball> {
        Some(Ball {
            m: chk(m)?,
            r: rz(p),
            p,
        })
    }
    /// a correctly rounded value of the exact number `exact_is_zero` tells about (R2)
    pub fn rounded(p: u32, m: RealMPFR, exact_is_zero: bool) -> Option<Ball> {
        let m = chk(m)?;
        if is_zero(&m) && !exact_is_zero {
            return None; // R2: a nonzero number flushed to 0
        }
        let b = Ball {
            m: m.clone(),
            r: rz(p),
            p,
        };
        let r = b.up(&m.abs() * &b.eps())?;
        Some(Ball { m, r, p })
    }
    /// result midpoint m with propagated radius prop; `zero_ok` says whether the exact result is 0
    fn with_rounding(&self, m: RealMPFR, prop: RealMPFR, zero_ok: bool) -> Option<Ball> {
        let m = chk(m)?;
        if is_zero(&m) && !zero_ok {
            return None; // R2
        }
        let prop = chk(prop)?;
        let r = self.up(chk(&prop + &(&m.abs() * &self.eps()))?)?;
        Some(Ball { m, r, p: self.p })
    }
    pub fn excludes_zero(&self) -> bool {
        self.m.abs() > self.r
    }
    /// revision 7rt4: the exact 0 (midpoint and radius 0)
    pub fn is_exact_zero(&self) -> bool {
        is_zero(&self.m) && is_zero(&self.r)
    }
    fn lo(&self) -> Option<RealMPFR> {
        chk(&self.m - &self.r)
    }
    fn hi(&self) -> Option<RealMPFR> {
        chk(&self.m + &self.r)
    }
    pub fn add(&self, o: &Ball) -> Option<Ball> {
        let m = &self.m + &o.m;
        let exact_zero = self.m == -o.m.clone();
        self.with_rounding(m, chk(&self.r + &o.r)?, exact_zero)
    }
    pub fn neg(&self) -> Ball {
        Ball {
            m: -self.m.clone(),
            r: self.r.clone(),
            p: self.p,
        }
    }
    pub fn mul(&self, o: &Ball) -> Option<Ball> {
        let m = &self.m * &o.m;
        let zero_ok = is_zero(&self.m) || is_zero(&o.m);
        let a = chk(&self.m.abs() * &o.r)?;
        let b = chk(&o.m.abs() * &self.r)?;
        let c = chk(&self.r * &o.r)?;
        let prop = chk(&chk(&a + &b)? + &c)?;
        self.with_rounding(m, prop, zero_ok)
    }
    pub fn inv(&self) -> Option<Ball> {
        if !self.excludes_zero() {
            // revision 7rt2: an exact 0 has no inverse at any precision; a ball across 0 is too wide
            return if is_zero(&self.m) && is_zero(&self.r) {
                None
            } else {
                too_wide()
            };
        }
        let a = self.m.abs();
        let m = RealMPFR::with_val(self.p, 1.0) / self.m.clone();
        // |1/x - 1/m| <= r / (|m| (|m| - r)); the denominator is rounded down by inflating r
        let gap = chk(&a - &self.up(self.r.clone())?)?;
        if gap <= rz(self.p) {
            return too_wide();
        }
        let prop = self.up(chk(&self.r / &chk(&a * &gap)?)?)?;
        self.with_rounding(m, prop, false)
    }
    fn lip(&self, m: RealMPFR, l: RealMPFR, zero_ok: bool) -> Option<Ball> {
        let l = chk(l)?;
        let prop = self.up(chk(&l * &self.r)?)?;
        self.with_rounding(m, prop, zero_ok)
    }
    /// [-1, 1]: the enclosure of sin or cos of any real number
    fn unit(p: u32) -> Ball {
        Ball {
            m: rz(p),
            r: RealMPFR::with_val(p, 1.0),
            p,
        }
    }
    /// B2: may MPFR sin/cos be called on this ball? (cost grows with the exponent of m)
    fn trig_ok(&self) -> bool {
        let small_r = self.r < RealMPFR::with_val(self.p, 1.0);
        let e = self.m.as_float().get_exp().unwrap_or(i32::MIN);
        small_r && (is_zero(&self.m) || (e as i64) <= self.p as i64 + 64)
    }
    /// revision 7rt2: the exp guard failed: too wide when |m| alone passes it
    fn exp_guard_fail(&self) -> Option<Ball> {
        if self.m.abs() - self.r.clone() <= RealMPFR::with_val(self.p, EXP_ARG_MAX) {
            too_wide()
        } else {
            None
        }
    }
    fn exp_arg_ok(&self) -> bool {
        let a = &self.m.abs() + &self.r;
        a.is_finite() && a <= RealMPFR::with_val(self.p, EXP_ARG_MAX)
    }
    pub fn unary(&self, op: UnaryOp) -> Option<Ball> {
        use UnaryOp::*;
        let p = self.p;
        let one = RealMPFR::with_val(p, 1.0);
        let zero = rz(p);
        let mz = is_zero(&self.m);
        let m_is_one = self.m == one;
        let f = |g: fn(&RealMPFR) -> RealMPFR| g(&self.m);
        match op {
            Neg => Some(self.neg()),
            Exp => {
                if !self.exp_arg_ok() {
                    return self.exp_guard_fail();
                }
                let l = self.up(self.hi()?.exp())?;
                self.lip(f(RealMPFR::exp), l, false)
            }
            Log => {
                let lo = self.lo()?;
                if lo <= zero {
                    return if self.hi()? > zero { too_wide() } else { None };
                }
                let l = self.up(chk(one.clone() / lo)?)?;
                self.lip(f(RealMPFR::ln), l, m_is_one)
            }
            Sqrt => {
                // revision 7rt4 (red team 4 numeric, m3): sqrt(0) = 0 for the exact 0
                if self.is_exact_zero() {
                    return Some(self.clone());
                }
                let lo = self.lo()?;
                if lo <= zero {
                    return if self.hi()? > zero { too_wide() } else { None };
                }
                let l = self.up(chk(
                    one.clone() / chk(RealMPFR::with_val(p, 2.0) * lo.sqrt())?
                )?)?;
                self.lip(f(RealMPFR::sqrt), l, mz)
            }
            Sin => {
                if !self.trig_ok() {
                    return Some(Ball::unit(p));
                }
                self.lip(f(RealMPFR::sin), one, mz)
            }
            Cos => {
                if !self.trig_ok() {
                    return Some(Ball::unit(p));
                }
                self.lip(f(RealMPFR::cos), one, false)
            }
            Tan => {
                if !self.trig_ok() {
                    // r >= 1, or an exponent above p + 64: both may pass at a higher precision
                    return too_wide();
                }
                let s = self.unary(Sin)?;
                let c = self.unary(Cos)?;
                s.mul(&c.inv()?)
            }
            Arctan => self.lip(f(RealMPFR::atan), one, mz),
            // revision 3 (review 3, m4): erf, L = 2 (the true bound is 2/sqrt(pi) = 1.128..)
            // revision 7 (review 7, B1): MPFR's erf sums the Taylor series at 0 with about
            // x^2/ln 2 extra bits and at least x^2 terms (mpfr 4.2.2, erf.c, mpfr_erf_0), so its
            // cost grows like x^2 until erf(x) rounds to +-1. For |x| > X_p, where
            // X_p^2 >= (p + 16) ln 2, the enclosure is built without MPFR: for x >= X >= 1,
            // 0 < 1 - erf(x) = erfc(x) < exp(-x^2)/(x sqrt(pi)) <= exp(-X^2) <= 2^-(p+16)
            // (Abramowitz and Stegun 7.1.13), so erf(x) lies in [1 - 2^-(p+16), 1]. MPFR is
            // called only when |m| < X_p; ball_cost charges each erf node by erf_units.
            Erf => {
                let xp = RealMPFR::with_val(p, erf_threshold(p));
                // m - r and m + r are rounded to nearest, which is monotone: lo > X_p implies
                // that the exact m - r exceeds X_p (X_p is representable)
                let lo = self.lo()?;
                let hi = self.hi()?;
                if lo > xp {
                    return Some(Ball {
                        m: one,
                        r: two_pow(p, -(p as i32) - 16),
                        p,
                    });
                }
                if hi < -xp.clone() {
                    return Some(Ball {
                        m: -one,
                        r: two_pow(p, -(p as i32) - 16),
                        p,
                    });
                }
                if self.m.abs() >= xp {
                    // a ball across +-X_p: the range of erf, [-1, 1]
                    return Some(Ball::unit(p));
                }
                let two = RealMPFR::with_val(p, 2.0);
                let m = RealMPFR::from_float(self.m.as_float().clone().erf());
                self.lip(m, two, mz)
            }
            Tanh => self.lip(f(RealMPFR::tanh), one, mz),
            Sinh | Cosh => {
                if !self.exp_arg_ok() {
                    return self.exp_guard_fail();
                }
                let a = chk(&self.m.abs() + &self.r)?;
                let l = self.up(a.cosh())?;
                if op == Sinh {
                    self.lip(f(RealMPFR::sinh), l, mz)
                } else {
                    self.lip(f(RealMPFR::cosh), l, false)
                }
            }
            Arcsin | Arccos => {
                let a = self.up(&self.m.abs() + &self.r)?;
                if a >= one {
                    // too wide when the ball meets (-1, 1): |m| - r < 1 (the midpoint of 1 - 2^-120
                    // rounds to 1 at 64 bits)
                    return if self.m.abs() - self.r.clone() < one {
                        too_wide()
                    } else {
                        None
                    };
                }
                let l = self.up(chk(one.clone() / chk((one.clone() - &a * &a).sqrt())?)?)?;
                if op == Arcsin {
                    self.lip(f(RealMPFR::asin), l, mz)
                } else {
                    self.lip(f(RealMPFR::acos), l, m_is_one)
                }
            }
            // revision 7rt3 (red team 3 numeric, m1): MPFR's own asinh, acosh and atanh with
            // Lipschitz bounds. Revision 7rt2 used the log formulas, which lose the argument's
            // relative precision: x + sqrt(x^2 + 1) cancels for x << -1, 1 + x rounds to 1 for
            // |x| < 2^-p, and x^2 leaves the window past 2^(2^27), at every precision
            Arcsinh => {
                // |asinh'(x)| = 1/sqrt(1 + x^2) <= min(1, 1/d) with d = |m| - r (shrunk by a
                // few ulps) the least |x| on the ball
                let d = chk(&self.m.abs() - &self.r)?;
                let shrink = RealMPFR::with_val(p, 1.0) - two_pow(p, 4 - p as i32);
                let l = if d > one {
                    self.up(chk(one.clone() / chk(&d * &shrink)?)?)?
                } else {
                    one.clone()
                };
                let m = RealMPFR::from_float(self.m.as_float().clone().asinh());
                self.lip(m, l, mz)
            }
            Arccosh => {
                // x >= lo > 1: |acosh'(x)| <= 1/sqrt(lo^2 - 1), with lo shrunk by a few ulps and
                // lo^2 - 1 as (lo - 1)(lo + 1) (lo - 1 exact for lo <= 2, Sterbenz)
                let lo = self.lo()?;
                let shrink = RealMPFR::with_val(p, 1.0) - two_pow(p, 4 - p as i32);
                let lod = chk(&lo * &shrink)?;
                if lod <= one {
                    return if self.hi()? > one { too_wide() } else { None };
                }
                let g = chk(&chk(&lod - &one)? * &chk(&lod + &one)?)?;
                let l = self.up(chk(one.clone() / chk(g.sqrt())?)?)?;
                let m = RealMPFR::from_float(self.m.as_float().clone().acosh());
                self.lip(m, l, false)
            }
            Arctanh => {
                // |x| <= a < 1: |atanh'(x)| <= 1/(1 - a^2) = 1/((1 - a)(1 + a))
                let a = self.up(&self.m.abs() + &self.r)?;
                if a >= one {
                    return if self.m.abs() - self.r.clone() < one {
                        too_wide()
                    } else {
                        None
                    };
                }
                let g = chk(&chk(&one - &a)? * &chk(&one + &a)?)?;
                let l = self.up(chk(one.clone() / g)?)?;
                let m = RealMPFR::from_float(self.m.as_float().clone().atanh());
                self.lip(m, l, mz)
            }
            Abs => Some(Ball {
                m: self.m.abs(),
                r: self.r.clone(),
                p,
            }),
            _ => None,
        }
    }
}

/// revision 4 (review 4, B2): kernel expressions share subtrees through Arc, so the tree can be
/// exponentially larger than the DAG. A node reached through an Arc is evaluated once per call
/// (memoized by its address; the expression is borrowed for the whole call, so addresses are
/// stable and distinct), which makes the evaluation linear in the DAG.
/// revision 7rt2: ball_expr with the reason when there is no enclosure
pub fn ball_expr_why(e: &Expr, p: u32) -> Result<Ball, NoBall> {
    TOO_WIDE.with(|c| c.set(false));
    match ball_expr(e, p) {
        Some(b) => Ok(b),
        None => Err(if TOO_WIDE.with(|c| c.get()) {
            NoBall::TooWide
        } else {
            NoBall::Never
        }),
    }
}
pub fn ball_expr(e: &Expr, p: u32) -> Option<Ball> {
    // revision 5 (review 5, B1): iterative post-order over the DAG with an explicit stack, so the
    // evaluation uses O(1) native stack whatever the depth of the expression. Each node is
    // evaluated once (memoized by its address; the expression is borrowed for the whole call). Every
    // child of a node is needed by it, so the first child without an enclosure ends the call.
    use std::collections::HashMap;
    fn ad(x: &Expr) -> usize {
        std::ptr::from_ref(x).addr()
    }
    let mut memo: HashMap<usize, Ball> = HashMap::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if memo.contains_key(&ad(x)) {
            continue;
        }
        if !expanded {
            stack.push((x, true));
            match x {
                Expr::Binary(op, l, r) => {
                    // an integer exponent is read, not evaluated
                    let int_pow = matches!((op, r.as_ref()), (BinaryOp::Pow, Expr::Integer(_)));
                    if !int_pow && !memo.contains_key(&ad(r)) {
                        stack.push((r, false));
                    }
                    if !memo.contains_key(&ad(l)) {
                        stack.push((l, false));
                    }
                }
                Expr::Unary(op, i) => {
                    // revision 7rt4: exp(k log a) with an integer-valued k evaluates a, not log a
                    let c: &Expr = match (op, exp_int_pow(i)) {
                        (UnaryOp::Exp, Some((_, a))) => a,
                        _ => i,
                    };
                    if !memo.contains_key(&ad(c)) {
                        stack.push((c, false));
                    }
                }
                _ => {}
            }
            continue;
        }
        let get = |y: &Expr| memo.get(&ad(y));
        let v: Option<Ball> = match x {
            // revision 7rt4 (red team 4 numeric, m3): a literal whose value is exact at p bits (an
            // integer of at most p bits, a dyadic rational with such a numerator) is an exact ball,
            // so `1 - 1` in a hint or a kernel's source is the exact 0 (revision 8: 0 +- 2^-63, and
            // sqrt, log and the powers of it were TooWide at every precision)
            Expr::Integer(n) if n.bit_length() <= p as u64 => {
                Ball::exact(p, RealMPFR::with_val_integer(p, n))
            }
            Expr::Integer(n) => Ball::rounded(p, RealMPFR::with_val_integer(p, n), n.is_zero()),
            Expr::Rational(q)
                if q.numerator().bit_length() <= p as u64 && is_pow2(q.denominator()) =>
            {
                Ball::exact(p, RealMPFR::with_val_rational(p, q))
            }
            Expr::Rational(q) => Ball::rounded(
                p,
                RealMPFR::with_val_rational(p, q),
                q.numerator().is_zero(),
            ),
            Expr::Real(f) => {
                if f.is_finite() {
                    Ball::exact(p, RealMPFR::with_val(p, *f))
                } else {
                    None
                }
            }
            Expr::Symbol(s) => match s.name() {
                "pi" => Ball::rounded(p, RealMPFR::pi(p), false),
                "e" => Ball::rounded(p, RealMPFR::e(p), false),
                _ => None,
            },
            Expr::Binary(op, l, r) => {
                let a = get(l)?;
                match op {
                    BinaryOp::Add => a.add(get(r)?),
                    BinaryOp::Sub => a.add(&get(r)?.neg()),
                    BinaryOp::Mul => a.mul(get(r)?),
                    BinaryOp::Div => a.mul(&get(r)?.inv()?),
                    BinaryOp::Pow => {
                        // integer exponent: binary powering (each product checked by R1/R2);
                        // otherwise exp(b log a) for a > 0
                        if let Expr::Integer(n) = r.as_ref() {
                            // revision 7rt3 (red team 3 numeric, m2): every exponent that fits in
                            // i64 by binary powering (at most 64 squarings and 64 products);
                            // beyond i64, |a|^k = exp(k log |a|) with the sign (-1)^k when a < 0.
                            // Revision 7rt2 refused |k| > 2^32 at every precision
                            match rustmath_core::NumericConversion::to_i64(n) {
                                None => big_int_pow(a, n, p),
                                Some(k) => bin_pow(a, k, p),
                            }
                        } else if let Some(k) = small_int_value(r) {
                            // revision 8 (review 8, m1): an exponent with an exact integer value k:
                            // binary powering for |k| <= 3 (at most 3 products and an inverse), else
                            // |a|^k = exp(k log |a|) with the sign; both within the 4 operations that
                            // ball_cost charges for a non-literal exponent
                            if k == 0 {
                                Ball::exact(p, RealMPFR::with_val(p, 1.0))
                            } else if k.unsigned_abs() <= 3 {
                                bin_pow(a, k, p)
                            } else {
                                big_int_pow(a, &rustmath_integers::Integer::from(k), p)
                            }
                        } else {
                            let b = get(r)?;
                            // revision 7rt4 (red team 4 numeric, m3): 0^b = 0 for b decided > 0;
                            // no value for b decided <= 0; else too wide
                            if a.is_exact_zero() {
                                let z = rz(p);
                                if b.lo()? > z {
                                    Ball::exact(p, z)
                                } else if b.hi()? <= z {
                                    None
                                } else {
                                    too_wide()
                                }
                            } else {
                                a.unary(UnaryOp::Log)?.mul(b)?.unary(UnaryOp::Exp)
                            }
                        }
                    }
                    BinaryOp::Mod => None,
                }
            }
            Expr::Unary(UnaryOp::Exp, i) if exp_int_pow(i).is_some() => {
                let (k, a) = exp_int_pow(i).unwrap();
                int_pow_ball(get(a)?, k, p)
            }
            Expr::Unary(op, i) => get(i)?.unary(*op),
            Expr::Function(_, _) => None,
        };
        memo.insert(ad(x), v?);
    }
    memo.remove(&ad(e))
}

/// revision 8 (review 8, m1): the exact value in i64 of an exponent that is not an integer literal: a
/// literal under negations, sums, differences and products (`-2`, `(-(-(5)))`, `3 - 1`, `2*3 - 3`),
/// each operation checked, read by a walk of at most SMALL_INT_NODES nodes and SMALL_INT_DEPTH levels
/// (no allocation, a constant amount of work per power node, covered by its charge). An exponent with
/// such a value is an integer power, so `(-3 + 1)^(-2)`, which a hint or a kernel made from an input
/// node can render, is enclosed as the normal form's `c^-2` is (revision 7rt3 sent it to
/// exp(r log a), which needs a > 0). Anything else (a quotient, a power, a deeper or larger shape) is
/// None, and the power takes exp(r log a) as before.
pub const SMALL_INT_NODES: u32 = 64;
pub const SMALL_INT_DEPTH: u32 = 8;
fn small_int_value(e: &Expr) -> Option<i64> {
    fn go(e: &Expr, depth: u32, left: &mut u32) -> Option<i64> {
        if *left == 0 || depth > SMALL_INT_DEPTH {
            return None;
        }
        *left -= 1;
        match e {
            Expr::Integer(n) => rustmath_core::NumericConversion::to_i64(n),
            Expr::Unary(UnaryOp::Neg, i) => go(i, depth + 1, left)?.checked_neg(),
            Expr::Binary(op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul), l, r) => {
                let a = go(l, depth + 1, left)?;
                let b = go(r, depth + 1, left)?;
                match op {
                    BinaryOp::Add => a.checked_add(b),
                    BinaryOp::Sub => a.checked_sub(b),
                    _ => a.checked_mul(b),
                }
            }
            _ => None,
        }
    }
    let mut left = SMALL_INT_NODES;
    go(e, 0, &mut left)
}

/// revision 7rt4: `exp(k log a)` with k an exact integer value (`small_int_value`) is a^k for every
/// nonzero a: the branches of log differ by 2 pi i m, and k times that is a multiple of 2 pi i. The
/// normal form renders a constant base to a series exponent as `exp(r log l)` (section 2.5), so at a
/// point where r is the integer k the constant term `exp(k log c)` of a negative c had no enclosure
/// (log of a negative ball), while its input syntax `c^k` had one (red team 4's review of revision
/// 7rt4: a random tower `((-3) + tanh(..))^(x+1)` at -2). Returns (k, a).
fn exp_int_pow(i: &Expr) -> Option<(i64, &Expr)> {
    if let Expr::Binary(BinaryOp::Mul, l, r) = i {
        if let Expr::Unary(UnaryOp::Log, a) = r.as_ref() {
            if let Some(k) = small_int_value(l) {
                return Some((k, a));
            }
        }
        if let Expr::Unary(UnaryOp::Log, a) = l.as_ref() {
            if let Some(k) = small_int_value(r) {
                return Some((k, a));
            }
        }
    }
    None
}
/// a^k for an exact integer k as the power paths do (a^0 = 1, binary powering for |k| <= 3, else
/// exp(k log |a|) with the sign); a must be nonzero (it is the argument of a log)
fn int_pow_ball(a: &Ball, k: i64, p: u32) -> Option<Ball> {
    if !a.excludes_zero() {
        return if a.is_exact_zero() { None } else { too_wide() };
    }
    if k == 0 {
        Ball::exact(p, RealMPFR::with_val(p, 1.0))
    } else if k.unsigned_abs() <= 3 {
        bin_pow(a, k, p)
    } else {
        big_int_pow(a, &rustmath_integers::Integer::from(k), p)
    }
}

/// a^k by binary powering (each product checked by R1/R2): at most 64 squarings and 64 products
fn bin_pow(a: &Ball, k: i64, p: u32) -> Option<Ball> {
    let mut acc = Ball::exact(p, RealMPFR::with_val(p, 1.0))?;
    let mut base = a.clone();
    let mut e = k.unsigned_abs();
    while e > 0 {
        if e & 1 == 1 {
            acc = acc.mul(&base)?;
        }
        e >>= 1;
        if e > 0 {
            base = base.mul(&base)?;
        }
    }
    if k < 0 {
        acc.inv()
    } else {
        Some(acc)
    }
}

/// revision 7rt3 (red team 3 numeric, m2): a^n for an integer n beyond i64: |a|^n = exp(n log |a|),
/// negated when a < 0 and n is odd, inverted when n < 0 (a ball across 0 is too wide)
fn big_int_pow(a: &Ball, n: &rustmath_integers::Integer, p: u32) -> Option<Ball> {
    // revision 7rt4 (red team 4 numeric, m3): 0^n = 0 for n > 0 (no value for n < 0)
    if a.is_exact_zero() {
        return if n.signum() > 0 {
            Ball::exact(p, rz(p))
        } else {
            None
        };
    }
    if !a.excludes_zero() {
        return too_wide();
    }
    let kb = Ball::rounded(p, RealMPFR::with_val_integer(p, &n.abs()), false)?;
    let neg = a.m < rz(p);
    let v = a
        .unary(UnaryOp::Abs)?
        .unary(UnaryOp::Log)?
        .mul(&kb)?
        .unary(UnaryOp::Exp)?;
    let v = if neg && n.is_odd() { v.neg() } else { v };
    if n.signum() < 0 {
        v.inv()
    } else {
        Some(v)
    }
}

/// A-priori cost of ball_expr(e, p) in work units, counted by walking the expression that will
/// be evaluated (every distinct node once, as ball_expr evaluates it).
/// The walk stops as soon as the sum passes `limit` (the caller's remaining budget).
/// revision 6: units per distinct node for the memory of the walk (memo, stacks, seen set: about
/// 280 bytes per node at p = 64 with the hash tables at their lowest load)
pub const BALL_NODE: u64 = 64;
/// u = (p/64 + 1)^2 * 8: the units of one MPFR operation at precision p
pub fn unit_units(p: u32) -> u64 {
    (p as u64 / 64 + 1).pow(2) * 8
}
/// revision 7 (review 7, B1): the units of a unary node at precision p (without BALL_NODE).
/// sin, cos: an operation at 2p + 64 (the argument reduction at exponent at most p + 64), tan
/// three of them; asinh, acosh, atanh 6 operations (their log formulas); erf `erf_units`; the
/// others 2.
pub fn unary_units(op: UnaryOp, p: u32, arg: &Expr) -> u64 {
    let unit = unit_units(p);
    let trig = ((2 * p as u64 + 64) / 64 + 1).pow(2) * 8;
    // revision 7rt1 (red team 1, m1): MPFR's sin computes sqrt(1 - cos^2) and doubles its working
    // precision when the result rounds to +-1 (sin.c, "near 1"), so a midpoint within an ulp of
    // (k + 1/2) pi costs an operation at about 3p + 64 bits; charged so (cos has no such path)
    let sin_units = ((3 * p as u64 + 64) / 64 + 1).pow(2) * 8;
    match op {
        UnaryOp::Sin => sin_units,
        UnaryOp::Cos => trig,
        UnaryOp::Tan => 3 * trig,
        UnaryOp::Arcsinh | UnaryOp::Arccosh | UnaryOp::Arctanh => 6 * unit,
        UnaryOp::Erf => erf_units(p, literal_abs_bound(arg)),
        _ => 2 * unit,
    }
}
/// revision 7 (review 7, B1): MPFR's erf costs about (terms) x (an operation at p + x^2/ln 2 bits),
/// with at least x^2 terms, and it is called only for |m| < X_p. Charged
/// ERF_FACTOR u (ERF_BASE + ceil(a^2)) with a = min(the argument's bound, X_p): the bound is known a priori
/// when the argument is a literal (through at most 4 negations), else a = X_p. An argument literal
/// at or above X_p + 1 takes the cheap enclosure: 2u.
pub const ERF_FACTOR: u64 = 2;
pub const ERF_BASE: u64 = 8;
pub fn erf_units(p: u32, arg_bound: Option<(f64, f64)>) -> u64 {
    let x = erf_threshold(p);
    let a = match arg_bound {
        // revision 7rt1: the cheap regime is decided by a lower bound of |v|, the charge by an upper one
        Some((lo, _)) if lo >= x + 1.0 => return 2 * unit_units(p),
        Some((_, hi)) => hi.min(x),
        None => x,
    };
    unit_units(p) * ERF_FACTOR * (ERF_BASE + (a * a).ceil() as u64)
}
/// revision 7rt1 (red team 1, m2): bounds lo <= m < hi of a magnitude m as (lo, hi, e) scaled by
/// 2^e, from the top 64 bits only: O(1) whatever the size (revision 7 converted the whole literal
/// with to_f64, an uncharged O(n) copy and division per erf node)
fn mag_bounds(m: &num_bigint::BigUint) -> (f64, f64, i64) {
    let b = m.bits();
    if b <= 64 {
        let v = num_traits::ToPrimitive::to_u64(m).unwrap() as f64;
        (v * (1.0 - 1e-15), v * (1.0 + 1e-15), 0)
    } else {
        let sh = b - 64;
        let t = num_traits::ToPrimitive::to_u64(&(m >> sh)).unwrap() as f64;
        (t * (1.0 - 1e-15), (t + 1.0) * (1.0 + 1e-15), sh as i64)
    }
}
fn scaled(v: f64, e: i64) -> f64 {
    if e > 2000 {
        f64::INFINITY
    } else if e < -2000 {
        0.0
    } else {
        v * 2f64.powi(e as i32)
    }
}
/// an upper bound stays positive when the value is (a scaled bound below the f64 range is 0)
fn up(h: f64, v: f64) -> f64 {
    if v > 0.0 {
        h.max(f64::MIN_POSITIVE)
    } else {
        h
    }
}
/// bounds lo <= |v| <= hi of a literal v (an integer, a rational or a float, under at most 4
/// negations), in O(1); None when the argument is anything else (revision 7rt1: a pair of bounds,
/// computed from bit lengths and top words)
pub fn literal_abs_bound(e: &Expr) -> Option<(f64, f64)> {
    let mut x = e;
    for _ in 0..4 {
        match x {
            Expr::Unary(UnaryOp::Neg, i) => x = i,
            _ => break,
        }
    }
    match x {
        Expr::Integer(n) => {
            let (lo, hi, e) = mag_bounds(n.as_bigint().magnitude());
            Some((scaled(lo, e), up(scaled(hi, e), hi)))
        }
        Expr::Rational(q) => {
            let (nl, nh, ne) = mag_bounds(q.numerator().as_bigint().magnitude());
            let (dl, dh, de) = mag_bounds(q.denominator().as_bigint().magnitude());
            Some((
                scaled(nl / dh * (1.0 - 1e-15), ne - de),
                up(scaled(nh / dl * (1.0 + 1e-15), ne - de), nh),
            ))
        }
        Expr::Real(f) => {
            let v = f.abs();
            if v.is_finite() {
                Some((v, v))
            } else {
                None
            }
        }
        _ => None,
    }
}
pub fn ball_cost(e: &Expr, p: u32, limit: u64) -> u64 {
    let unit = unit_units(p);
    let mut total: u64 = 0;
    let mut stack: Vec<&Expr> = vec![e];
    // revision 4 (review 4, B2): each node reached through an Arc is counted once, as ball_expr
    // evaluates it once
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    while let Some(x) = stack.pop() {
        let c = match x {
            Expr::Integer(n) => unit + n.bit_length() / 64 + 1,
            Expr::Rational(q) => {
                unit + (q.numerator().bit_length() + q.denominator().bit_length()) / 64 + 1
            }
            Expr::Real(_) | Expr::Symbol(_) => unit,
            Expr::Binary(op, l, r) => {
                if seen.insert(std::sync::Arc::as_ptr(l).addr()) {
                    stack.push(l);
                }
                if seen.insert(std::sync::Arc::as_ptr(r).addr()) {
                    stack.push(r);
                }
                match (op, r.as_ref()) {
                    // binary powering: at most 2 * bits(k) products
                    // revision 7rt3: up to 64 bits (revision 7rt2: 33); beyond i64 exp(k log |a|), 3u
                    (BinaryOp::Pow, Expr::Integer(k)) => unit * (2 * k.bit_length().min(64) + 1),
                    // revision 8 (review 8, m1): a non-literal exponent: exp(b log a) (3 operations)
                    // or, for an exact integer value, at most 4 (binary powering for |k| <= 3 with an
                    // inverse, or log, product, exp and inverse)
                    (BinaryOp::Pow, _) => unit * 4,
                    _ => unit,
                }
            }
            Expr::Unary(op, i) => {
                if seen.insert(std::sync::Arc::as_ptr(i).addr()) {
                    stack.push(i);
                }
                unary_units(*op, p, i)
            }
            Expr::Function(_, args) => {
                for a in args.iter() {
                    if seen.insert(std::sync::Arc::as_ptr(a).addr()) {
                        stack.push(a);
                    }
                }
                unit
            }
        };
        // revision 6: plus BALL_NODE per distinct node for its memo entry, its stack entry and
        // its `seen` entry (two MPFR values at p <= 64 are about 160 bytes with the hash slot)
        total = total.saturating_add(c).saturating_add(BALL_NODE);
        if total > limit {
            return total;
        }
    }
    total
}

fn real_free(c: &Konst) -> bool {
    // revision 5: cached when the constant is built
    c.t.sz.real_free
}

/// a real ball enclosure of a constant free of I and of indeterminates (None: no enclosure).
/// Charged before it runs: ball_cost of the expression evaluated.
fn enclose(c: &Konst, p: u32) -> Result<Ball, NoBall> {
    if c.is_zero() {
        return Ball::exact(p, rz(p)).ok_or(NoBall::Never);
    }
    if !real_free(c) {
        return Err(NoBall::Never);
    }
    let e = c.to_expr();
    let remaining = crate::series_const::remaining();
    let cost = ball_cost(&e, p, remaining);
    if !crate::series_const::charge(cost) {
        return Err(NoBall::Never);
    }
    ball_expr_why(&e, p)
}

/// c = A + B*I, A and B free of I: NonZero iff A and B both have real enclosures and one of
/// them excludes 0 (then A and B are real numbers, and A + B*I = 0 iff A = B = 0)
/// revision 7rt2 (red team 2 numeric, m1): a precision at which an enclosure is too wide goes on
/// to the next precision, as section 2.2 rule 5 says (revision 7rt1 stopped at the first missing
/// enclosure, so constants that 256 bits decide stayed undecided); `Never` stops
pub fn proves_nonzero_ab(a: &Konst, b: &Konst) -> bool {
    for p in [64u32, 256, 1024] {
        let ea = enclose(a, p);
        if matches!(ea, Err(NoBall::Never)) {
            return false;
        }
        let eb = enclose(b, p);
        match (ea, eb) {
            (Ok(x), Ok(y)) => {
                if x.excludes_zero() || y.excludes_zero() {
                    return true;
                }
            }
            (_, Err(NoBall::Never)) => return false,
            _ => {} // too wide at this precision
        }
    }
    false
}

/// revision 7rt3 (red team 3 structure, H1): the sign of a constant A free of I, decided by its
/// real enclosures at 64, 256 and 1024 bits as rule 5 (a too-wide precision goes on, `Never` stops):
/// Some(0) when A is exactly 0, Some(1) or Some(-1) when a ball excludes 0, None otherwise.
pub fn real_sign(a: &Konst) -> Option<i32> {
    if let Some(q) = a.as_rational() {
        return Some(q.numerator().signum() as i32);
    }
    for p in [64u32, 256, 1024] {
        match enclose(a, p) {
            Ok(x) if x.excludes_zero() => return Some(if x.m > rz(p) { 1 } else { -1 }),
            Ok(_) | Err(NoBall::TooWide) => continue,
            Err(NoBall::Never) => return None,
        }
    }
    None
}

/// floor of a finite float with |x| < 2^p, converted through binary digits (never decimal)
fn floor_small(x: &RealMPFR, p: u32) -> Option<rustmath_integers::Integer> {
    let f = x.as_float();
    if !f.is_zero() && f.get_exp().is_none_or(|e| e > p as i32) {
        return None;
    }
    let z = f.clone().floor().to_integer()?;
    let neg = z < 0;
    let za = z.clone().abs();
    let mut digits = vec![0u8; za.significant_digits::<u8>()];
    za.write_digits(&mut digits, rug::integer::Order::Lsf);
    let mut v = rustmath_integers::Integer::from(0);
    let mut sh = rustmath_integers::Integer::from(1);
    for d in digits {
        v = v + sh.clone() * rustmath_integers::Integer::from(d as i64);
        sh = sh * rustmath_integers::Integer::from(256);
    }
    Some(if neg { -v } else { v })
}

/// alpha = A + B*I: decided non-integer when B is enclosed away from 0, or B = 0 exactly and the
/// enclosure of A lies strictly between two consecutive integers. Unknown carries the integer
/// nearest to mid(A) when |mid(A)| < 2^p (else None: the caller reports alpha itself).
pub fn non_integer(
    a: &Konst,
    b: &Konst,
) -> (crate::series_const::ZS, Option<rustmath_integers::Integer>) {
    use crate::series_const::ZS;
    let mut nearest = None;
    for p in [64u32, 256, 1024] {
        // revision 7rt2 (red team 2 numeric, m1): too wide goes on to the next precision
        let ea = enclose(a, p);
        if matches!(ea, Err(NoBall::Never)) {
            return (ZS::Unknown, nearest);
        }
        let (ea, eb) = match (ea, enclose(b, p)) {
            (Ok(x), Ok(y)) => (x, y),
            (_, Err(NoBall::Never)) => return (ZS::Unknown, nearest),
            _ => continue,
        };
        if eb.excludes_zero() {
            return (ZS::NonZero, None);
        }
        if b.is_zero() {
            // k = floor(midpoint); alpha lies in (k, k+1) when the balls of alpha - k and
            // alpha - (k+1) both exclude 0 (the enclosure is an interval that contains the midpoint)
            let fl = match floor_small(&ea.m, p) {
                Some(z) => z,
                None => continue, // |mid| >= 2^p: the ball contains several integers at this p
            };
            let k1 = fl.clone() + rustmath_integers::Integer::from(1);
            let bk = match Ball::rounded(p, RealMPFR::with_val_integer(p, &fl), fl.is_zero()) {
                Some(x) => x,
                None => continue,
            };
            let bk1 = match Ball::rounded(p, RealMPFR::with_val_integer(p, &k1), k1.is_zero()) {
                Some(x) => x,
                None => continue,
            };
            let (d0, d1) = match (ea.add(&bk.neg()), ea.add(&bk1.neg())) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            let half = RealMPFR::with_val(p, 0.5);
            nearest = Some(if &ea.m - &RealMPFR::with_val_integer(p, &fl) < half {
                fl.clone()
            } else {
                k1.clone()
            });
            if d0.excludes_zero() && d1.excludes_zero() {
                return (ZS::NonZero, None);
            }
        }
    }
    (ZS::Unknown, nearest)
}

// ---------------------------------------------------------------------------------------------
// S1 request C-1 (docs/workstreams/m23_golay/REQUESTS.md, section 2): complex enclosures of a
// constant, for the branch-cut rule of the series engine (`series_engine::cut_check`). The real
// balls above enclose a constant only when every step of its expression is real (log of a
// positive ball, and so on), so a constant term such as log(-2), (-4)^(1/2) or arcsin(2) has no
// real enclosure. A complex enclosure is a rectangle re + im i of two real balls, computed with
// the principal branches of the engine's conventions (`series.rs`, module header): log z =
// log|z| + i arg z with arg in (-pi, pi], sqrt and powers through exp(a log z), arcsin z =
// -i log(iz + sqrt(1 - z^2)), arccos z = pi/2 - arcsin z, arctan z = (i/2)(log(1 - iz) -
// log(1 + iz)), arcsinh z = log(z + sqrt(1 + z^2)), arccosh z = log(z + sqrt(z + 1) sqrt(z - 1)),
// arctanh z = (log(1 + z) - log(1 - z))/2. A log whose argument's rectangle meets the cut
// (-oo, 0] without lying on it exactly has no enclosure at that precision (too wide); on the cut
// exactly (an imaginary part that is the exact 0) it is log|z| + i pi. A real step keeps the real
// ball's result and an exact 0 imaginary part, so a real constant stays exactly real. The engine
// uses these enclosures only to decide that a quantity is nonzero (a rectangle that excludes 0
// from one coordinate) or exactly 0 (an exact 0 coordinate); a coordinate whose ball contains 0
// decides nothing. Charged before it runs: COMPLEX_COST_FACTOR times ball_cost.
// ---------------------------------------------------------------------------------------------

/// a complex operation costs at most this many real ones (a product: 4 products and 2 sums)
pub const COMPLEX_COST_FACTOR: u64 = 8;

/// a rectangle re + im i enclosing a complex constant
#[derive(Clone)]
pub struct CBall {
    pub re: Ball,
    pub im: Ball,
}

impl Ball {
    /// Some(1) or Some(-1) when the ball excludes 0, Some(0) for the exact 0, else None
    pub fn sign(&self) -> Option<i32> {
        if self.is_exact_zero() {
            Some(0)
        } else if self.excludes_zero() {
            Some(if self.m > rz(self.p) { 1 } else { -1 })
        } else {
            None
        }
    }
    /// the sign of (self - q) for a small integer q
    pub fn sign_minus(&self, q: i64) -> Option<i32> {
        let b = Ball::exact(self.p, RealMPFR::with_val(self.p, q as f64))?;
        self.add(&b.neg())?.sign()
    }
}

fn cexact_zero(p: u32) -> Option<Ball> {
    Ball::exact(p, rz(p))
}

impl CBall {
    fn real(b: Ball) -> Option<CBall> {
        let p = b.p;
        Some(CBall {
            re: b,
            im: cexact_zero(p)?,
        })
    }
    fn i_unit(p: u32) -> Option<CBall> {
        Some(CBall {
            re: cexact_zero(p)?,
            im: Ball::exact(p, RealMPFR::with_val(p, 1.0))?,
        })
    }
    fn int(p: u32, q: i64) -> Option<CBall> {
        CBall::real(Ball::exact(p, RealMPFR::with_val(p, q as f64))?)
    }
    /// the imaginary part is the exact 0: the constant is real
    pub fn is_real(&self) -> bool {
        self.im.is_exact_zero()
    }
    fn add(&self, o: &CBall) -> Option<CBall> {
        Some(CBall {
            re: self.re.add(&o.re)?,
            im: self.im.add(&o.im)?,
        })
    }
    fn neg(&self) -> CBall {
        CBall {
            re: self.re.neg(),
            im: self.im.neg(),
        }
    }
    fn sub(&self, o: &CBall) -> Option<CBall> {
        self.add(&o.neg())
    }
    fn mul(&self, o: &CBall) -> Option<CBall> {
        if self.is_real() && o.is_real() {
            return CBall::real(self.re.mul(&o.re)?);
        }
        // (a + bi)(c + di) = (ac - bd) + (ad + bc) i; exact zeros stay exact (Ball::mul)
        let ac = self.re.mul(&o.re)?;
        let bd = self.im.mul(&o.im)?;
        let ad = self.re.mul(&o.im)?;
        let bc = self.im.mul(&o.re)?;
        Some(CBall {
            re: ac.add(&bd.neg())?,
            im: ad.add(&bc)?,
        })
    }
    /// times i
    fn mul_i(&self) -> CBall {
        CBall {
            re: self.im.neg(),
            im: self.re.clone(),
        }
    }
    fn half(&self) -> Option<CBall> {
        let h = CBall::real(Ball::exact(self.re.p, RealMPFR::with_val(self.re.p, 0.5))?)?;
        self.mul(&h)
    }
    fn inv(&self) -> Option<CBall> {
        if self.is_real() {
            return CBall::real(self.re.inv()?);
        }
        // 1/(a + bi) = (a - bi)/(a^2 + b^2)
        let d = self.re.mul(&self.re)?.add(&self.im.mul(&self.im)?)?;
        let di = d.inv()?;
        Some(CBall {
            re: self.re.mul(&di)?,
            im: self.im.neg().mul(&di)?,
        })
    }
    fn exp(&self) -> Option<CBall> {
        let e = self.re.unary(UnaryOp::Exp)?;
        if self.is_real() {
            return CBall::real(e);
        }
        Some(CBall {
            re: e.mul(&self.im.unary(UnaryOp::Cos)?)?,
            im: e.mul(&self.im.unary(UnaryOp::Sin)?)?,
        })
    }
    /// the principal log: arg in (-pi, pi]; a rectangle that meets the cut (-oo, 0] without lying
    /// on it exactly is too wide, and 0 has no log
    fn log(&self) -> Option<CBall> {
        let p = self.re.p;
        if self.is_real() {
            return match self.re.sign() {
                Some(1) => CBall::real(self.re.unary(UnaryOp::Log)?),
                Some(-1) => Some(CBall {
                    re: self.re.neg().unary(UnaryOp::Log)?,
                    im: Ball::rounded(p, RealMPFR::pi(p), false)?,
                }),
                Some(_) => None,
                None => too_wide(),
            };
        }
        let m2 = self.re.mul(&self.re)?.add(&self.im.mul(&self.im)?)?;
        let lm = m2
            .unary(UnaryOp::Log)?
            .mul(&Ball::exact(p, RealMPFR::with_val(p, 0.5))?)?;
        let half_pi = Ball::rounded(p, &RealMPFR::pi(p) * &RealMPFR::with_val(p, 0.5), false)?;
        let arg = if self.re.sign() == Some(1) {
            self.im.mul(&self.re.inv()?)?.unary(UnaryOp::Arctan)?
        } else if self.im.sign() == Some(1) {
            half_pi.add(&self.re.mul(&self.im.inv()?)?.unary(UnaryOp::Arctan)?.neg())?
        } else if self.im.sign() == Some(-1) {
            half_pi
                .neg()
                .add(&self.re.mul(&self.im.inv()?)?.unary(UnaryOp::Arctan)?.neg())?
        } else {
            // the rectangle meets the half-line (-oo, 0]
            return too_wide();
        };
        Some(CBall { re: lm, im: arg })
    }
    fn sqrt(&self) -> Option<CBall> {
        if self.is_real() {
            return match self.re.sign() {
                Some(1) | Some(0) => CBall::real(self.re.unary(UnaryOp::Sqrt)?),
                Some(_) => Some(CBall {
                    re: cexact_zero(self.re.p)?,
                    im: self.re.neg().unary(UnaryOp::Sqrt)?,
                }),
                None => too_wide(),
            };
        }
        self.log()?.half()?.exp()
    }
    /// z^w = exp(w log z) (principal), z nonzero
    fn pow(&self, w: &CBall) -> Option<CBall> {
        w.mul(&self.log()?)?.exp()
    }
    fn powi(&self, k: i64) -> Option<CBall> {
        let p = self.re.p;
        let mut acc = CBall::int(p, 1)?;
        let mut base = self.clone();
        let mut e = k.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base)?;
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base)?;
            }
        }
        if k < 0 {
            acc.inv()
        } else {
            Some(acc)
        }
    }
    fn unary(&self, op: UnaryOp) -> Option<CBall> {
        use UnaryOp::*;
        let p = self.re.p;
        if self.is_real() {
            // a real step keeps the real ball (an exact 0 imaginary part); outside the real
            // domain of op (log of a negative ball, arcsin of 2) the complex formulas below
            if let Some(b) = self.re.unary(op) {
                return CBall::real(b);
            }
            if matches!(
                op,
                Exp | Sin | Cos | Tan | Sinh | Cosh | Tanh | Arctan | Arcsinh | Erf | Abs | Neg
            ) {
                // total on the reals: a failure is a window or guard failure, not a domain one
                return None;
            }
        }
        let one = CBall::int(p, 1)?;
        match op {
            Neg => Some(self.neg()),
            Exp => self.exp(),
            Log => self.log(),
            Sqrt => self.sqrt(),
            Abs => {
                let m2 = self.re.mul(&self.re)?.add(&self.im.mul(&self.im)?)?;
                CBall::real(m2.unary(Sqrt)?)
            }
            Sin | Cos | Sinh | Cosh => {
                // sin(a + bi) = sin a cosh b + i cos a sinh b, cos(a + bi) = cos a cosh b -
                // i sin a sinh b; sinh(a + bi) = sinh a cos b + i cosh a sin b, cosh(a + bi) =
                // cosh a cos b + i sinh a sin b
                let (a, b) = (&self.re, &self.im);
                match op {
                    Sin => Some(CBall {
                        re: a.unary(Sin)?.mul(&b.unary(Cosh)?)?,
                        im: a.unary(Cos)?.mul(&b.unary(Sinh)?)?,
                    }),
                    Cos => Some(CBall {
                        re: a.unary(Cos)?.mul(&b.unary(Cosh)?)?,
                        im: a.unary(Sin)?.mul(&b.unary(Sinh)?)?.neg(),
                    }),
                    Sinh => Some(CBall {
                        re: a.unary(Sinh)?.mul(&b.unary(Cos)?)?,
                        im: a.unary(Cosh)?.mul(&b.unary(Sin)?)?,
                    }),
                    _ => Some(CBall {
                        re: a.unary(Cosh)?.mul(&b.unary(Cos)?)?,
                        im: a.unary(Sinh)?.mul(&b.unary(Sin)?)?,
                    }),
                }
            }
            Tan => self.unary(Sin)?.mul(&self.unary(Cos)?.inv()?),
            Tanh => self.unary(Sinh)?.mul(&self.unary(Cosh)?.inv()?),
            Arcsin => {
                // -i log(iz + sqrt(1 - z^2))
                let s = one.sub(&self.mul(self)?)?.sqrt()?;
                let l = self.mul_i().add(&s)?.log()?;
                Some(l.mul_i().neg())
            }
            Arccos => {
                let half_pi = CBall::real(Ball::rounded(
                    p,
                    &RealMPFR::pi(p) * &RealMPFR::with_val(p, 0.5),
                    false,
                )?)?;
                half_pi.sub(&self.unary(Arcsin)?)
            }
            Arctan => {
                // (i/2)(log(1 - iz) - log(1 + iz))
                let iz = self.mul_i();
                let d = one.sub(&iz)?.log()?.sub(&one.add(&iz)?.log()?)?;
                d.mul_i().half()
            }
            Arcsinh => {
                let s = one.add(&self.mul(self)?)?.sqrt()?;
                self.add(&s)?.log()
            }
            Arccosh => {
                let s = self.add(&one)?.sqrt()?.mul(&self.sub(&one)?.sqrt()?)?;
                self.add(&s)?.log()
            }
            Arctanh => one.add(self)?.log()?.sub(&one.sub(self)?.log()?)?.half(),
            _ => None,
        }
    }
}

/// a complex enclosure of the constant expression e at precision p (memoized by node address, an
/// explicit stack, as `ball_expr`); `None` for a node it does not evaluate (a free symbol, a
/// function call, mod, erf, gamma, ... of a non-real argument) or a failed guard
pub fn cball_expr(e: &Expr, p: u32) -> Option<CBall> {
    use std::collections::HashMap;
    fn ad(x: &Expr) -> usize {
        std::ptr::from_ref(x).addr()
    }
    let named_i = !crate::series_const::var_is_named_i();
    let mut memo: HashMap<usize, CBall> = HashMap::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if memo.contains_key(&ad(x)) {
            continue;
        }
        if !expanded {
            stack.push((x, true));
            match x {
                Expr::Binary(op, l, r) => {
                    let int_pow = matches!((op, r.as_ref()), (BinaryOp::Pow, Expr::Integer(_)));
                    if !int_pow && !memo.contains_key(&ad(r)) {
                        stack.push((r, false));
                    }
                    if !memo.contains_key(&ad(l)) {
                        stack.push((l, false));
                    }
                }
                Expr::Unary(_, i) if !memo.contains_key(&ad(i)) => stack.push((i, false)),
                _ => {}
            }
            continue;
        }
        let get = |y: &Expr| memo.get(&ad(y));
        let v: Option<CBall> = match x {
            Expr::Symbol(s) if s.name() == "I" && named_i => CBall::i_unit(p),
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => {
                CBall::real(ball_expr(x, p)?)
            }
            Expr::Binary(op, l, r) => {
                let a = get(l)?;
                match op {
                    BinaryOp::Add => a.add(get(r)?),
                    BinaryOp::Sub => a.sub(get(r)?),
                    BinaryOp::Mul => a.mul(get(r)?),
                    BinaryOp::Div => a.mul(&get(r)?.inv()?),
                    BinaryOp::Pow => {
                        let k = match r.as_ref() {
                            Expr::Integer(n) => rustmath_core::NumericConversion::to_i64(n),
                            _ => small_int_value(r),
                        };
                        match k {
                            Some(k) => {
                                if k < 0 && a.re.is_exact_zero() && a.im.is_exact_zero() {
                                    None
                                } else {
                                    a.powi(k)
                                }
                            }
                            None if matches!(r.as_ref(), Expr::Integer(_)) => None,
                            None => {
                                if a.re.is_exact_zero() && a.im.is_exact_zero() {
                                    None
                                } else {
                                    a.pow(get(r)?)
                                }
                            }
                        }
                    }
                    BinaryOp::Mod => None,
                }
            }
            Expr::Unary(op, i) => get(i)?.unary(*op),
            Expr::Function(_, _) => None,
        };
        memo.insert(ad(x), v?);
    }
    memo.remove(&ad(e))
}

/// a complex enclosure of a constant free of indeterminates at precision p, charged
/// COMPLEX_COST_FACTOR times `ball_cost` before it runs; `TooWide` when a guard failed on a
/// rectangle that a higher precision may shrink
pub fn cball_konst(c: &Konst, p: u32) -> Result<CBall, NoBall> {
    if c.is_zero() {
        return CBall::real(Ball::exact(p, rz(p)).ok_or(NoBall::Never)?).ok_or(NoBall::Never);
    }
    if c.t.sz.any_indet {
        return Err(NoBall::Never);
    }
    let e = c.to_expr();
    let remaining = crate::series_const::remaining();
    let cost = ball_cost(&e, p, remaining).saturating_mul(COMPLEX_COST_FACTOR);
    if !crate::series_const::charge(cost) {
        return Err(NoBall::Never);
    }
    TOO_WIDE.with(|w| w.set(false));
    match cball_expr(&e, p) {
        Some(b) => Ok(b),
        None => Err(if TOO_WIDE.with(|w| w.get()) {
            NoBall::TooWide
        } else {
            NoBall::Never
        }),
    }
}

/// S1 request C-1: decide a question about a constant by its complex enclosures at 64, 256 and
/// 1024 bits: `f` returns `Some(answer)` when the rectangle decides, `None` when it does not; a
/// precision without an enclosure that is too wide goes on to the next one, and `Never` stops.
pub fn cdecide<T>(c: &Konst, f: impl Fn(&CBall) -> Option<T>) -> Option<T> {
    for p in [64u32, 256, 1024] {
        match cball_konst(c, p) {
            Ok(b) => {
                if let Some(t) = f(&b) {
                    return Some(t);
                }
            }
            Err(NoBall::TooWide) => continue,
            Err(NoBall::Never) => return None,
        }
    }
    None
}

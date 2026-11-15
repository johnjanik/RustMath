//! p-adic numbers Q_p, as u·p^v with u a truncated element of Z_p.
//!
//! MAGMA Handbook Ch. 47: §47.7.1 (`L ! u` for an integer u), §47.8.1 (`x / y`, `x ^ -1`),
//! §47.8.2 (equality), §47.8.3 (`IsZero`, `IsOne`, `IsUnit`); Ch. 17 §17.5.2 (`R ! a`) and
//! §17.5.5 (`IsUnit`, `IsNilpotent`). Precision conventions follow Sage
//! (doc.sagemath.org, `padic_capped_relative_element`: `is_zero`, `is_equal_to`).

use super::padic_integer::PadicInteger;
use crate::category_methods::{cheap_primality, minimal_root};
use rustmath_core::{CommutativeRing, Field, MathError, MultiplicativeOrder, Result, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An element of Q_p (any integer p > 1), stored as `u · p^v` with `u` a [`PadicInteger`]
/// known to relative precision n: it stands for every element of `u p^v + p^(v+n) Z_p`.
/// The stored `u` need not be a unit (sums are not normalised); [`PadicRational::valuation`]
/// is the stored exponent v.
///
/// The model and its departures from MAGMA are those of [`PadicInteger`] (equality to the
/// common precision, `is_zero`/`is_one` "to the known precision", `IsUnit` of Ch. 17, and
/// the fixed-modulus model). As there, the ring predicates are proofs under the convention
/// that `is_zero` and `is_one` are exact: an element indistinguishable from 0 (from 1) is
/// treated as 0 (as 1), so `1 + O(5^10)` has order `Finite(1)`. In Q_p every nonzero
/// element is a unit of the field; the
/// "valuation 0" test of §47.8.3 is `x.normalized().valuation() == 0`. Two consequences of
/// weak equality: an element known to absolute precision <= 0 (for example `a - a` with
/// `a = 1/5 + O(5^0)`) is weakly zero and also compares equal to `Ring::one()`; it is
/// reported as zero only (`is_one()` is false), so `is_zero` and `is_one` never both hold.
///
/// # Unbound constants (the static `zero()`/`one()`)
///
/// As for [`PadicInteger`]: an unbound element is an exact integer c (MAGMA `L ! c`,
/// §47.7.1) whose unit part is an unbound `PadicInteger` and whose valuation is 0.
/// * `x + c`, `x - c`, `c - x`, `x * c`, `x / c`, `c / x` bind c into x's field at x's
///   relative precision ([`PadicRational::bind_integer`], normalised), except that `x ± 0`
///   is `x`, `0 - x` is `-x`, and `0 * x`, `x * 0`, `0 / x` (x not weakly zero) are the
///   unbound 0.
/// * Two unbound operands combine in Z. `c / d` is the unbound c/d when d divides c; for
///   another nonzero d, `/` panics and [`Field::divide`] returns `Err(InvalidArgument)` ("bind
///   an operand first"); d = 0 is "Division by zero".
/// * `prime()`, `precision()`, `valuation()`, `abs()` panic on an unbound element; `lift`
///   returns `Err(InvalidArgument)`; `Display` prints c.
#[derive(Clone, Debug)]
pub struct PadicRational {
    /// p-adic integer part (unit in zp); an unbound PadicInteger for an unbound element
    unit: PadicInteger,
    /// Valuation (power of p); 0 for an unbound element
    valuation: i32,
}

/// `p^k` as an integer (k >= 0).
fn ipow(p: &Integer, k: i64) -> Integer {
    p.pow(u32::try_from(k).expect("p-adic exponent fits in u32"))
}

/// Whether `p^e` divides the integer `a·p^s - b·p^t` (shifts `s, t >= 0`); `true` when
/// `e <= 0`. A term whose shift is at least e is divisible by p^e and is dropped, so no power
/// of p above p^e is formed.
///
/// This is the membership test of value equality, exact for every p > 1 (review 2, N1): for
/// an integer D and `e >= 0`, `D ∈ p^e Z_p` iff `p^e | D` in Z (`Z/p^e Z = Z_p/p^e Z_p`), and
/// multiplication by `p^w` maps `p^j Z_p` onto `p^(j+w) Z_p` (p is a unit of Q_p = Z_p[1/p]).
/// No reduced fraction is formed, whose denominator could hide part of a composite p.
fn divides_shifted_difference(
    p: &Integer,
    e: i64,
    (a, s): (&Integer, i64),
    (b, t): (&Integer, i64),
) -> bool {
    if e <= 0 {
        return true;
    }
    let term = |c: &Integer, k: i64| {
        if k >= e {
            Integer::zero()
        } else {
            c.clone() * ipow(p, k)
        }
    };
    ((term(a, s) - term(b, t)) % ipow(p, e)).is_zero()
}

impl PadicRational {
    /// Create a p-adic rational from a rational number
    ///
    /// # Arguments
    ///
    /// * `rational` - Rational number to convert
    /// * `prime` - Prime p
    /// * `precision` - Precision for the p-adic integer part
    ///
    /// The rational 0 gives `0 + O(p^precision)` at valuation 0.
    pub fn from_rational(rational: Rational, prime: Integer, precision: usize) -> Result<Self> {
        if rational.is_zero() {
            return Ok(PadicRational {
                unit: PadicInteger::zero(prime, precision)?,
                valuation: 0,
            });
        }
        // Compute p-adic valuation of numerator and denominator
        let num_val = rational.numerator().valuation(&prime) as i32;
        let den_val = rational.denominator().valuation(&prime) as i32;
        let valuation = num_val - den_val;

        // Extract the unit part (remove all factors of p)
        let mut num = rational.numerator().clone();
        let mut den = rational.denominator().clone();

        // Remove factors of p from numerator
        for _ in 0..num_val {
            num = num / prime.clone();
        }

        // Remove factors of p from denominator
        for _ in 0..den_val {
            den = den / prime.clone();
        }

        // Now num and den are coprime to p
        // Compute num/den in zp
        let num_padic = PadicInteger::from_integer(num, prime.clone(), precision)?;
        let den_padic = PadicInteger::from_integer(den, prime, precision)?;

        let unit = num_padic * den_padic.inverse()?;

        Ok(PadicRational { unit, valuation })
    }

    /// Create from a p-adic integer (an unbound one gives the unbound constant).
    pub fn from_padic_integer(padic: PadicInteger) -> Self {
        PadicRational {
            unit: padic,
            valuation: 0,
        }
    }

    /// The unbound integer constant `c`.
    pub(crate) fn unbound(c: Integer) -> Self {
        PadicRational {
            unit: PadicInteger::unbound(c),
            valuation: 0,
        }
    }

    /// Whether this element has a prime and a precision (see the type docs).
    pub fn is_bound(&self) -> bool {
        self.unit.is_bound()
    }

    /// `L ! c` (MAGMA Ch. 47 §47.7.1) at relative precision `precision`: `from_rational(c)`
    /// (normalised: c = m p^k, m to precision n, valuation k) for c != 0; the zero
    /// `0 + O(p^n)` at valuation 0 for c = 0.
    pub(crate) fn bind_integer(c: &Integer, prime: &Integer, precision: usize) -> Self {
        PadicRational::from_rational(Rational::from_integer(c.clone()), prime.clone(), precision)
            .expect("an integer constant binds for every p > 1 and precision > 0")
    }

    /// The integer constant of an unbound element.
    fn constant(&self) -> &Integer {
        debug_assert!(!self.is_bound());
        self.unit.value()
    }

    /// `self` (unbound) bound into `model`'s field at `model`'s relative precision.
    fn bind_like(&self, model: &PadicRational) -> PadicRational {
        PadicRational::bind_integer(self.constant(), model.unit.prime(), model.unit.precision())
    }

    fn require_bound(&self, method: &str) {
        if !self.is_bound() {
            panic!(
                "PadicRational::{method}: the unbound constant {} (from Ring::zero()/one()) has \
                 no prime or precision; combine it with a bound element, or use \
                 PadicField::zero()/one()/from_rational",
                self.unit.value()
            );
        }
    }

    /// x = u' p^k with p not dividing u', u' known to relative precision n - j where
    /// j = v_p(stored unit value); `None` for an element that is zero to its precision.
    /// Bound elements only. (For a composite p, u' need not be a unit of Z_p.)
    pub(crate) fn normalized(&self) -> Option<PadicRational> {
        self.require_bound("normalized");
        if self.unit.is_zero() {
            return None;
        }
        let p = self.unit.prime();
        let n = self.unit.precision();
        let j = self.unit.value().valuation(p);
        let u = self.unit.value().clone() / ipow(p, i64::from(j));
        let unit = PadicInteger::from_integer(u, p.clone(), n - j as usize)
            .expect("j < n because 0 < u < p^n");
        Some(PadicRational {
            unit,
            valuation: self.valuation + j as i32,
        })
    }

    /// The exact rational `u · p^v` (the canonical representative); the constant c for an
    /// unbound element.
    pub(crate) fn representative(&self) -> Rational {
        if !self.is_bound() {
            return Rational::from_integer(self.constant().clone());
        }
        let unit = self.unit.value().clone();
        let p = self.unit.prime();
        let v = i64::from(self.valuation);
        if v >= 0 {
            Rational::from_integer(unit * ipow(p, v))
        } else {
            Rational::new(unit, ipow(p, -v)).expect("p^k is nonzero")
        }
    }

    /// `valuation + precision`: x is known modulo `p^absolute_precision`.
    ///
    /// # Panics
    ///
    /// On an unbound element.
    pub fn absolute_precision(&self) -> i64 {
        self.require_bound("absolute_precision");
        i64::from(self.valuation) + self.unit.precision() as i64
    }

    /// The exact rational `r` known to absolute precision `absolute`, relative precision at
    /// most `cap`: `from_rational(r, p, min(cap, absolute - v_p(r)))` when `r != 0` and
    /// `absolute > v_p(r)`; otherwise the zero `0 + O(p^cap)` at valuation `absolute - cap`
    /// (absolute precision `absolute`). For a prime `p` (the E.6 inverse; `v_p` of a
    /// rational is read from its reduced fraction).
    pub(crate) fn from_rational_to_absolute_precision(
        r: &Rational,
        prime: &Integer,
        absolute: i64,
        cap: usize,
    ) -> Result<Self> {
        if !r.is_zero() {
            let w = i64::from(r.numerator().valuation(prime))
                - i64::from(r.denominator().valuation(prime));
            if absolute > w {
                let rel = usize::try_from(absolute - w).unwrap_or(usize::MAX).min(cap);
                return PadicRational::from_rational(r.clone(), prime.clone(), rel);
            }
        }
        let valuation = i32::try_from(absolute - cap as i64).map_err(|_| {
            MathError::NumericalError("p-adic valuation does not fit in i32".to_string())
        })?;
        Ok(PadicRational {
            unit: PadicInteger::zero(prime.clone(), cap)?,
            valuation,
        })
    }

    /// Get the unit part (an unbound `PadicInteger` for an unbound element).
    pub fn unit(&self) -> &PadicInteger {
        &self.unit
    }

    /// Get the valuation
    ///
    /// # Panics
    ///
    /// On an unbound element.
    pub fn valuation(&self) -> i32 {
        self.require_bound("valuation");
        self.valuation
    }

    /// Get the prime
    ///
    /// # Panics
    ///
    /// On an unbound element.
    pub fn prime(&self) -> &Integer {
        self.require_bound("prime");
        self.unit.prime()
    }

    /// Get the precision
    ///
    /// # Panics
    ///
    /// On an unbound element.
    pub fn precision(&self) -> usize {
        self.require_bound("precision");
        self.unit.precision()
    }

    /// Compute absolute value (p-adic norm): |x|_p = p^(-v) where v is the valuation
    ///
    /// # Panics
    ///
    /// On an unbound element.
    pub fn abs(&self) -> f64 {
        self.require_bound("abs");
        let p = self.prime().to_f64().unwrap();
        p.powi(-self.valuation)
    }

    /// Lift to higher precision
    pub fn lift(&self, new_precision: usize) -> Result<Self> {
        if !self.is_bound() {
            return Err(MathError::InvalidArgument(format!(
                "PadicRational::lift: the unbound constant {} has no precision; bind it first",
                self.unit.value()
            )));
        }
        Ok(PadicRational {
            unit: self.unit.lift(new_precision)?,
            valuation: self.valuation,
        })
    }

    /// Whether `self` (bound) agrees with the exact integer `c` to its absolute precision,
    /// that is `u p^v - c ∈ p^(v+n) Z_p` (u the stored unit value, n the relative precision).
    /// Every power of p formed is at most `p^n` or at most about `c^2`, whatever v is:
    /// * `v <= 0`: `u p^v - c = p^v (u - c p^-v)`, so test `p^n | u - c p^-v` (the c term is
    ///   dropped when `-v >= n`);
    /// * `v > 0`: `u p^v ∈ p^v Z_p`, so `p^v | c` is necessary (false at once when
    ///   `p^v > |c| > 0`, read from bit lengths); then `c = p^v c''` and test `p^n | u - c''`.
    fn value_equals_integer(&self, c: &Integer) -> bool {
        let p = self.unit.prime();
        let u = self.unit.value();
        let n = self.unit.precision() as i64;
        let v = i64::from(self.valuation);
        if v <= 0 {
            return divides_shifted_difference(p, n, (u, 0), (c, -v));
        }
        if c.is_zero() {
            return divides_shifted_difference(p, n, (u, 0), (c, 0));
        }
        // p^v >= 2^(v·floor(log2 p)) > |c| when v·floor(log2 p) >= bits(c).
        let floor_log2_p = p.bit_length().saturating_sub(1);
        if (v as u64).saturating_mul(floor_log2_p) >= c.bit_length() {
            return false;
        }
        let pv = ipow(p, v);
        if !(c.clone() % pv.clone()).is_zero() {
            return false;
        }
        divides_shifted_difference(p, n, (u, 0), (&(c.clone() / pv), 0))
    }

    /// The inverse of a bound element that is not weakly zero, through [`Self::normalized`]:
    /// x = u' p^k gives x^-1 = (u'^-1 mod p^(n-j)) · p^-k. `Err(NotImplemented)` when
    /// `gcd(u', p) != 1` (only for a composite p: undecided, never `NotInvertible`).
    fn bound_inverse(&self) -> Result<Self> {
        let x = self.normalized().ok_or(MathError::DivisionByZero)?;
        if !x.unit.value().gcd(x.unit.prime()).is_one() {
            return Err(MathError::NotImplemented(format!(
                "PadicRational inverse: the normalised unit {} is not coprime to the composite \
                 p = {}",
                x.unit.value(),
                x.unit.prime()
            )));
        }
        Ok(PadicRational {
            unit: x.unit.inverse()?,
            valuation: -x.valuation,
        })
    }

    /// Division of two unbound constants: the unbound c/d when d divides c.
    fn unbound_quotient(c: &Integer, d: &Integer) -> Result<Self> {
        if d.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        if (c.clone() % d.clone()).is_zero() {
            Ok(PadicRational::unbound(c.clone() / d.clone()))
        } else {
            Err(MathError::InvalidArgument(format!(
                "PadicRational: {c} / {d} is not an integer constant; bind an operand first \
                 (combine it with a bound element or use PadicField::from_rational)"
            )))
        }
    }

    /// The rules of `/` (type docs) as a `Result`.
    fn checked_div(&self, other: &Self) -> Result<Self> {
        match (self.is_bound(), other.is_bound()) {
            (false, false) => PadicRational::unbound_quotient(self.constant(), other.constant()),
            (false, true) => {
                if other.is_zero() {
                    return Err(MathError::DivisionByZero);
                }
                if self.constant().is_zero() {
                    return Ok(PadicRational::unbound(Integer::zero()));
                }
                self.bind_like(other).checked_div(other)
            }
            (true, false) => {
                if other.constant().is_zero() {
                    return Err(MathError::DivisionByZero);
                }
                self.checked_div(&other.bind_like(self))
            }
            (true, true) => {
                if other.is_zero() {
                    return Err(MathError::DivisionByZero);
                }
                Ok(self.clone() * other.bound_inverse()?)
            }
        }
    }
}

impl fmt::Display for PadicRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.is_bound() || self.valuation == 0 {
            write!(f, "{}", self.unit)
        } else if self.valuation > 0 {
            write!(f, "{}*{}^{}", self.unit, self.prime(), self.valuation)
        } else {
            write!(f, "{}/{}^{}", self.unit, self.prime(), -self.valuation)
        }
    }
}

/// Value equality to the common absolute precision (see [`PadicInteger`], point 1): same
/// prime, and the exact representatives differ by an element of `p^m Z_p` with m the
/// smaller absolute precision. Not computed as `(x - y).is_zero()`: the `Add` below aligns
/// the higher-valuation operand at the other's relative precision and can drop known
/// digits (`0 + O(5^10)` vs `5^5 + O(5^8)` would compare equal that way). An unbound
/// operand is exact.
///
/// Cost: with `v0 = min(v_x, v_y)` the difference is `p^v0 (u_x p^(v_x - v0) - u_y p^(v_y -
/// v0))`, and the test is `p^(m - v0) | ...` on integers (`divides_shifted_difference`).
/// Since `m - v0` is at most the relative precision of the operand of valuation v0, every
/// power of p formed is at most `p^n` with n the larger relative precision: the cost does
/// not grow with the valuations. Against an unbound c see `value_equals_integer`.
impl PartialEq for PadicRational {
    fn eq(&self, other: &Self) -> bool {
        match (self.is_bound(), other.is_bound()) {
            (false, false) => self.constant() == other.constant(),
            (true, false) => self.value_equals_integer(other.constant()),
            (false, true) => other.value_equals_integer(self.constant()),
            (true, true) => {
                if self.unit.prime() != other.unit.prime() {
                    return false;
                }
                let (vx, vy) = (i64::from(self.valuation), i64::from(other.valuation));
                let v0 = vx.min(vy);
                let m = self.absolute_precision().min(other.absolute_precision());
                divides_shifted_difference(
                    self.unit.prime(),
                    m - v0,
                    (self.unit.value(), vx - v0),
                    (other.unit.value(), vy - v0),
                )
            }
        }
    }
}

impl Ring for PadicRational {
    /// The unbound constant 0 (see the type docs).
    fn zero() -> Self {
        PadicRational::unbound(Integer::zero())
    }

    /// The unbound constant 1 (see the type docs).
    fn one() -> Self {
        PadicRational::unbound(Integer::one())
    }

    fn is_zero(&self) -> bool {
        self.unit.is_zero()
    }

    /// Value comparison with 1 to the absolute precision, false for a weakly zero element:
    /// `(5 + O(5^10)) · 5^-1`, stored as (5, -1), is one.
    fn is_one(&self) -> bool {
        if !self.is_bound() {
            return self.constant().is_one();
        }
        !self.is_zero() && self.value_equals_integer(&Integer::from(1))
    }

    /// Square-and-multiply; `x.pow(0)` of a bound `x` is the bound 1 at `x`'s relative
    /// precision; an unbound c gives the unbound c^n.
    fn pow(&self, n: u32) -> Self {
        if !self.is_bound() {
            return PadicRational::unbound(self.constant().pow(n));
        }
        if n == 0 {
            return PadicRational::from_padic_integer(
                PadicInteger::one(self.unit.prime().clone(), self.unit.precision())
                    .expect("a bound element has p > 1 and precision > 0"),
            );
        }
        let mut result = self.clone();
        for bit in (0..(31 - n.leading_zeros())).rev() {
            result = result.clone() * result;
            if (n >> bit) & 1 == 1 {
                result = result * self.clone();
            }
        }
        result
    }

    /// * weakly zero: `Some(false)` (the convention);
    /// * x = u' p^k with `gcd(u', p) = 1`: `Some(true)` (u' ∈ Z_p^×, p^k ∈ Q_p^×; for a
    ///   prime p this is every nonzero x);
    /// * p a prime power q^a (q prime, checked when q < psi_12): `Some(true)` (Q_p = Q_q is a
    ///   field, and every lift of a non-weakly-zero x is nonzero);
    /// * otherwise `None` (a composite p with two prime factors: Q_p ≅ ∏ Q_q).
    ///
    /// Unbound c: `Some(c != 0)` (a nonzero integer is a unit of Q_p = Z_p[1/p]).
    fn unit_status(&self) -> Option<bool> {
        if !self.is_bound() {
            return Some(!self.constant().is_zero());
        }
        let Some(x) = self.normalized() else {
            return Some(false);
        };
        if x.unit.value().gcd(x.unit.prime()).is_one() {
            return Some(true);
        }
        let (q, _) = minimal_root(x.unit.prime());
        (cheap_primality(&q) == Some(true)).then_some(true)
    }

    /// The normalised inverse of a bound element. Weakly zero: `Err(NotInvertible)`;
    /// `gcd(u', p) != 1` (composite p only): `Err(NotImplemented)` (undecided). Unbound c:
    /// `±1` give themselves, `0` gives `Err(NotInvertible)`, any other c
    /// `Err(InvalidArgument)` (1/c is not an integer constant; bind it first).
    fn unit_inverse(&self) -> Result<Self> {
        if !self.is_bound() {
            let c = self.constant();
            if c.is_one() || (-c.clone()).is_one() {
                return Ok(self.clone());
            }
            if c.is_zero() {
                return Err(MathError::NotInvertible);
            }
            return Err(MathError::InvalidArgument(format!(
                "PadicRational::unit_inverse: 1/{c} is not an integer constant; bind it first"
            )));
        }
        if self.is_zero() {
            return Err(MathError::NotInvertible);
        }
        self.bound_inverse()
    }

    /// Weakly zero: `Some(true)` (the convention); otherwise `Some(false)` (Q_p ≅ ∏ Q_q is
    /// reduced for every p > 1). Unbound c: `Some(c == 0)`.
    fn nilpotent_status(&self) -> Option<bool> {
        Some(self.is_zero())
    }

    /// * weakly zero: `NotAUnit`; `1`: `Finite(1)`; `-1`: `Finite(2)`;
    /// * x = u' p^k with `gcd(u', p) = 1` and `k != 0`: `Infinite` for every p > 1 (if
    ///   x^m = 1 then p^(mk) = u'^(-m) would be a unit of Z_p, but p^|mk| ∈ pZ_p);
    /// * k = 0: the [`PadicInteger`] rule on u';
    /// * otherwise `Undecided`.
    ///
    /// Unbound c: `0` `NotAUnit`, `±1` `Finite(1)`/`Finite(2)`, `|c| >= 2` `Infinite`
    /// (Q ⊂ Q_p, and c^m = 1 in Q forces c = ±1). Sage: `Qp(5)(5)`, `Qp(5)(1/5)`
    /// `multiplicative_order()` = +Infinity.
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        if self.is_zero() {
            return MultiplicativeOrder::NotAUnit;
        }
        if self.is_one() {
            return MultiplicativeOrder::Finite(1);
        }
        if (-self.clone()).is_one() {
            return MultiplicativeOrder::Finite(2);
        }
        if !self.is_bound() {
            return MultiplicativeOrder::Infinite;
        }
        let x = self.normalized().expect("not weakly zero");
        if !x.unit.value().gcd(x.unit.prime()).is_one() {
            return MultiplicativeOrder::Undecided;
        }
        if x.valuation != 0 {
            MultiplicativeOrder::Infinite
        } else {
            x.unit.multiplicative_order_status()
        }
    }

    /// `Some(0)`: Q ⊂ Q_p.
    fn static_characteristic() -> Option<u64> {
        Some(0)
    }

    fn static_is_commutative() -> Option<bool> {
        Some(true)
    }

    const HAS_UNBOUND_CONSTANTS: bool = true;

    fn is_unbound_constant(&self) -> bool {
        !self.is_bound()
    }

    /// MAGMA `L ! c` at the model's prime and relative precision.
    fn bind_to_parent_of(&self, model: &Self) -> Option<Self> {
        (!self.is_bound() && model.is_bound()).then(|| self.bind_like(model))
    }
}

impl CommutativeRing for PadicRational {}

impl Field for PadicRational {
    /// Bound: the normalised inverse (x = u' p^k gives (u'^-1 mod p^(n-j)) p^-k); weakly
    /// zero: `Err(DivisionByZero)`; `gcd(u', p) != 1` (composite p): `Err(NotImplemented)`.
    /// Unbound c: `±1` give themselves, `0` gives `Err(DivisionByZero)`, any other c
    /// `Err(InvalidArgument)` (1/c is not an integer constant; bind it first).
    fn inverse(&self) -> Result<Self> {
        if !self.is_bound() {
            return match self.unit_inverse() {
                Err(MathError::NotInvertible) => Err(MathError::DivisionByZero),
                r => r,
            };
        }
        if self.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        self.bound_inverse()
    }

    /// MAGMA `x / y` (Ch. 47 §47.8.1) with the rules of `/` as `Result`s: an unbound operand
    /// is bound into the other operand's field first (so `x.divide(&two)` equals `x / two`);
    /// a zero divisor gives `Err(DivisionByZero)`; the unbound 0 over a nonzero y is the
    /// unbound 0; unbound c / unbound d is the unbound c/d when d divides c, else
    /// `Err(InvalidArgument)` ("bind an operand first").
    fn divide(&self, other: &Self) -> Result<Self> {
        self.checked_div(other)
    }
}

impl Neg for PadicRational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        PadicRational {
            unit: -self.unit,
            valuation: self.valuation,
        }
    }
}

impl Neg for &PadicRational {
    type Output = PadicRational;

    fn neg(self) -> Self::Output {
        PadicRational {
            unit: -self.unit.clone(),
            valuation: self.valuation,
        }
    }
}

impl Add for PadicRational {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        // The exact zero must not lower the other operand's precision.
        if !other.is_bound() && other.constant().is_zero() {
            return self;
        }
        if !self.is_bound() && self.constant().is_zero() {
            return other;
        }
        match (self.is_bound(), other.is_bound()) {
            (false, false) => {
                return PadicRational::unbound(self.constant().clone() + other.constant().clone())
            }
            (false, true) => return self.bind_like(&other) + other,
            (true, false) => {
                let o = other.bind_like(&self);
                return self + o;
            }
            (true, true) => {}
        }
        if self.valuation == other.valuation {
            // Same valuation: just add the units
            PadicRational {
                unit: self.unit + other.unit,
                valuation: self.valuation,
            }
        } else if self.valuation < other.valuation {
            // self has smaller valuation (larger p-adic norm)
            let diff = (other.valuation - self.valuation) as usize;
            let p = self.prime().clone();
            let mut shifted_other = other.unit;
            for _ in 0..diff {
                shifted_other = shifted_other * PadicInteger::from_integer(p.clone(), p.clone(), self.precision()).unwrap();
            }
            PadicRational {
                unit: self.unit + shifted_other,
                valuation: self.valuation,
            }
        } else {
            // other has smaller valuation
            let diff = (self.valuation - other.valuation) as usize;
            let p = self.prime().clone();
            let precision = self.precision();
            let mut shifted_self = self.unit;
            for _ in 0..diff {
                shifted_self = shifted_self * PadicInteger::from_integer(p.clone(), p.clone(), precision).unwrap();
            }
            PadicRational {
                unit: shifted_self + other.unit,
                valuation: other.valuation,
            }
        }
    }
}

impl Sub for PadicRational {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Mul for PadicRational {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        // The exact zero times anything is the exact zero.
        if (!self.is_bound() && self.constant().is_zero())
            || (!other.is_bound() && other.constant().is_zero())
        {
            return PadicRational::unbound(Integer::zero());
        }
        match (self.is_bound(), other.is_bound()) {
            (false, false) => {
                PadicRational::unbound(self.constant().clone() * other.constant().clone())
            }
            (false, true) => self.bind_like(&other) * other,
            (true, false) => {
                let o = other.bind_like(&self);
                self * o
            }
            (true, true) => PadicRational {
                unit: self.unit * other.unit,
                valuation: self.valuation + other.valuation,
            },
        }
    }
}

impl Div for PadicRational {
    type Output = Self;

    /// The rules of [`Field::divide`], panicking where it returns `Err`.
    fn div(self, other: Self) -> Self::Output {
        match self.checked_div(&other) {
            Ok(q) => q,
            Err(MathError::DivisionByZero) => panic!("Division by zero"),
            Err(e) => panic!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64, d: i64) -> Rational {
        Rational::new(Integer::from(n), Integer::from(d)).unwrap()
    }
    fn qp(n: i64, d: i64, p: i64, prec: usize) -> PadicRational {
        PadicRational::from_rational(q(n, d), Integer::from(p), prec).unwrap()
    }
    fn from_int(v: i64, p: i64, prec: usize) -> PadicRational {
        PadicRational::from_padic_integer(
            PadicInteger::from_integer(Integer::from(v), Integer::from(p), prec).unwrap(),
        )
    }
    fn one() -> PadicRational {
        <PadicRational as Ring>::one()
    }

    #[test]
    fn test_from_rational() {
        let p = Integer::from(5);
        let prec = 10;

        // 3/5 in Q_5
        let rat = Rational::new(Integer::from(3), Integer::from(5)).unwrap();
        let padic = PadicRational::from_rational(rat, p, prec).unwrap();

        // Valuation should be -1 (one factor of 5 in denominator)
        assert_eq!(padic.valuation(), -1);
    }

    #[test]
    fn test_padic_norm() {
        let p = Integer::from(5);
        let prec = 10;

        // 25 = 5^2 has 5-adic norm 1/25
        let rat = Rational::new(Integer::from(25), Integer::from(1)).unwrap();
        let padic = PadicRational::from_rational(rat, p, prec).unwrap();

        assert_eq!(padic.valuation(), 2);
        assert!((padic.abs() - 0.04).abs() < 1e-10);
    }

    // gp: (1+O(5^10)) + (4+O(5^10)) == 5+O(5^11) -> 1. At c79a582 the sum, stored as
    // (5, 0), compared unequal to from_rational(5) = (1, 1).
    #[test]
    fn sum_one_plus_four_equals_five() {
        let s = from_int(1, 5, 10) + from_int(4, 5, 10);
        assert_eq!(s, qp(5, 1, 5, 10));
        assert_eq!(qp(5, 1, 5, 10), s);
    }

    #[test]
    fn zero_at_valuation_two_equals_ring_zero() {
        let z = (from_int(3, 5, 10) - from_int(3, 5, 10)) * qp(25, 1, 5, 10);
        assert_eq!(z.valuation(), 2);
        assert_eq!(z, <PadicRational as Ring>::zero());
        assert_eq!(<PadicRational as Ring>::zero(), z);
        let x = <PadicRational as Ring>::zero() + qp(25, 1, 5, 10);
        assert_eq!(x.valuation(), 2);
        assert_eq!(x.precision(), 10);
    }

    #[test]
    fn ring_one_binds_with_full_precision() {
        let s = one() + qp(7, 25, 5, 10);
        assert_eq!(s, qp(32, 25, 5, 10));
        assert_eq!(s.absolute_precision(), 8);
    }

    // gp: (1/5)*(5+O(5^10)) == 1 -> 1. At c79a582 is_one of (5, -1) was false.
    #[test]
    fn is_one_unnormalised() {
        let x = from_int(5, 5, 10) * qp(1, 5, 5, 10);
        assert_eq!(x.valuation(), -1);
        assert!(x.is_one());
    }

    // An element of absolute precision 0 is weakly zero, and is not one (review M6).
    // Composite p (review 2, N1): 3 * 6^-1 + O(6^0) has representative 1/2; every lift is
    // 1/2 + Z_6, whose Q_2 component has valuation -1, so it is never 1, although the
    // reduced valuation v_6(1/2 - 1) = 0 >= 0 would say so.
    #[test]
    fn weakly_zero_is_not_one() {
        let a = qp(1, 5, 5, 1);
        let x = a.clone() - a;
        assert_eq!((x.valuation(), x.precision(), x.absolute_precision()), (-1, 1, 0));
        assert!(x.is_zero());
        assert!(!x.is_one());
        assert_eq!(x.unit_status(), Some(false));
        assert_eq!(x.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
        assert!(x == one()); // weak equality: vacuous at absolute precision 0 (documented)

        let y = from_int(3, 6, 1) * qp(1, 6, 6, 1);
        assert_eq!((y.valuation(), y.absolute_precision()), (-1, 0));
        assert!(!y.is_zero());
        assert!(!y.is_one());
        assert_ne!(y, one());
        assert_ne!(y.multiplicative_order_status(), MultiplicativeOrder::Finite(1));
        assert!(!matches!(y.unit_inverse(), Ok(ref v) if v == &y));
    }

    // 0 + O(5^10) vs 5^5 + O(5^8): they differ at 5^5, within both precisions. Through
    // `Sub`, the difference is (0, 0, n = 3) and looks zero.
    #[test]
    fn equality_is_not_fooled_by_add_precision_loss() {
        let a = from_int(0, 5, 10);
        let b = PadicRational {
            unit: PadicInteger::from_integer(Integer::from(1), Integer::from(5), 3).unwrap(),
            valuation: 5,
        };
        assert!((a.clone() - b.clone()).is_zero());
        assert_ne!(a, b);
    }

    /// The definition of `d ∈ p^m Z_p` for `p = ∏ q^a` (Z_p ≅ ∏ Z_q): `v_q(d) >= a·m` for
    /// every q | p.
    fn in_power_by_definition(d: &Rational, factors: &[(i64, i64)], m: i64) -> bool {
        d.is_zero()
            || factors.iter().all(|&(q, a)| {
                let q = Integer::from(q);
                i64::from(d.numerator().valuation(&q)) - i64::from(d.denominator().valuation(&q))
                    >= a * m
            })
    }

    fn exact_value(x: &PadicRational) -> Rational {
        let u = Rational::from_integer(x.unit.value().clone());
        let p = x.unit.prime().clone();
        let v = x.valuation;
        if v >= 0 {
            u * Rational::from_integer(p.pow(v as u32))
        } else {
            u / Rational::from_integer(p.pow((-v) as u32))
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 33) % ((hi - lo + 1) as u64)) as i64
        }
    }

    // `==` (bound and unbound operands) and `is_one` against the definition, for a prime p
    // and the composite p = 6, 12, 4 (review 2, N1), on unnormalised storage: the integer
    // test of the fix-up round (bounded by the relative precisions) gives the same answers.
    #[test]
    fn equality_matches_the_definition() {
        let mut rng = Lcg(0x2545_f491_4f6c_dd1d);
        let mut equal_pairs = 0;
        for (p, factors) in [
            (5i64, &[(5i64, 1i64)][..]),
            (6, &[(2, 1), (3, 1)][..]),
            (12, &[(2, 2), (3, 1)][..]),
            (4, &[(2, 2)][..]),
        ] {
            let pz = Integer::from(p);
            let elt = |u: i64, n: i64, v: i64| PadicRational {
                unit: PadicInteger::from_integer(Integer::from(u), pz.clone(), n as usize).unwrap(),
                valuation: v as i32,
            };
            for _ in 0..600 {
                let n = rng.range(1, 4);
                let x = elt(rng.range(0, p.pow(n as u32) - 1), n, rng.range(-4, 4));
                let y = match rng.range(0, 2) {
                    // A random element.
                    0 => {
                        let n = rng.range(1, 4);
                        elt(rng.range(0, p.pow(n as u32) - 1), n, rng.range(-4, 4))
                    }
                    // x plus a multiple of p^k, at another precision.
                    1 => {
                        let k = rng.range(0, 4);
                        let u =
                            x.unit.value().to_i64().unwrap() + p.pow(k as u32) * rng.range(0, 3);
                        elt(u, rng.range(1, 5), i64::from(x.valuation))
                    }
                    // the value of x stored as (u p^j, v - j).
                    _ => {
                        let j = rng.range(0, 3);
                        let u = x.unit.value().to_i64().unwrap() * p.pow(j as u32);
                        elt(u, n + j, i64::from(x.valuation) - j)
                    }
                };
                let m = x.absolute_precision().min(y.absolute_precision());
                let expected =
                    in_power_by_definition(&(exact_value(&x) - exact_value(&y)), factors, m);
                assert_eq!(x == y, expected, "p = {p}: {x:?} vs {y:?}");
                assert_eq!(y == x, expected);
                equal_pairs += usize::from(expected);

                let c = rng.range(-60, 60) * p.pow(rng.range(0, 3) as u32);
                let cr = Rational::from_integer(Integer::from(c));
                let expected = in_power_by_definition(
                    &(exact_value(&x) - cr),
                    factors,
                    x.absolute_precision(),
                );
                let unbound = PadicRational::unbound(Integer::from(c));
                assert_eq!(x == unbound, expected, "p = {p}: {x:?} vs {c}");
                assert_eq!(unbound == x, expected);

                let d1 = exact_value(&x) - Rational::from_integer(Integer::from(1));
                let expected =
                    !x.is_zero() && in_power_by_definition(&d1, factors, x.absolute_precision());
                assert_eq!(x.is_one(), expected, "p = {p}: is_one({x:?})");
            }
        }
        assert!(equal_pairs > 300, "only {equal_pairs} equal pairs");
    }

    // `==` and `is_one` form no power of p above the relative precisions, so a valuation of
    // 10^9 costs nothing (fix-up round 1: the first R1 version built p^|v|, 48 s in debug at
    // v = 10^7).
    #[test]
    fn equality_cost_does_not_grow_with_the_valuation() {
        let p = Integer::from(5);
        let unit =
            |u: i64, n: usize| PadicInteger::from_integer(Integer::from(u), p.clone(), n).unwrap();
        let big = 1_000_000_000;
        let x = PadicRational {
            unit: unit(1, 10),
            valuation: big,
        };
        let five = qp(5, 1, 5, 10);
        assert_ne!(x, five);
        assert!(!x.is_one());
        assert_eq!(x, x.clone());
        // The same value stored as (5^2, big - 2) at relative precision 12.
        assert_eq!(
            x,
            PadicRational {
                unit: unit(25, 12),
                valuation: big - 2
            }
        );
        // 1 + 5^12 agrees with 1 to relative precision 10; 1 + 5^9 does not.
        assert_eq!(
            x,
            PadicRational {
                unit: unit(1 + 5i64.pow(12), 15),
                valuation: big
            }
        );
        assert_ne!(
            x,
            PadicRational {
                unit: unit(1 + 5i64.pow(9), 15),
                valuation: big
            }
        );
        assert_ne!(
            x,
            PadicRational {
                unit: unit(1, 10),
                valuation: big + 3
            }
        );
        assert_ne!(x, one());
        assert_ne!(x, PadicRational::unbound(Integer::from(250)));
        assert_eq!(
            x.multiplicative_order_status(),
            MultiplicativeOrder::Infinite
        );
        let y = PadicRational {
            unit: unit(1, 10),
            valuation: -big,
        };
        assert!(!y.is_one());
        assert_ne!(y, one());
        assert_ne!(y, five);
        assert_eq!(
            y.multiplicative_order_status(),
            MultiplicativeOrder::Infinite
        );
        // 250 = 2·5^3.
        let z = PadicRational {
            unit: unit(2, 10),
            valuation: 3,
        };
        assert_eq!(z, PadicRational::unbound(Integer::from(250)));
        assert_ne!(z, PadicRational::unbound(Integer::from(250 + 5i64.pow(12))));
        assert_eq!(z, PadicRational::unbound(Integer::from(250 + 5i64.pow(13))));
        assert_ne!(z, PadicRational::unbound(Integer::from(2)));
    }

    // gp: 1/(5+O(5^10)) -> 5^-1 + O(5^8). At c79a582 Field::inverse returned
    // Err(NotInvertible) and `/` panicked.
    #[test]
    fn inverse_of_unnormalised_element() {
        let x = from_int(5, 5, 10);
        let inv = Field::inverse(&x).unwrap();
        assert_eq!(inv.unit().value(), &Integer::from(1));
        assert_eq!(inv.valuation(), -1);
        assert_eq!(inv.precision(), 9);
        let y = from_int(3, 5, 10) / (from_int(1, 5, 10) + from_int(4, 5, 10));
        assert_eq!(y, qp(3, 5, 5, 10));
    }

    // gp: 3/2 in Q_5 is a unit (lift(Mod(3,5^10)/2) -> 4882814).
    #[test]
    fn divide_binds_unbound() {
        let two = one() + one();
        let x = qp(3, 1, 5, 10);
        let h = x.divide(&two).unwrap();
        assert_eq!(h, qp(3, 2, 5, 10));
        assert_eq!(h, x.clone() / two.clone());
        assert_eq!(h.unit().value(), &Integer::from(4_882_814));
        assert_eq!(x.divide(&<PadicRational as Ring>::zero()), Err(MathError::DivisionByZero));
        let z = <PadicRational as Ring>::zero().divide(&x).unwrap();
        assert!(!z.is_bound() && z.is_zero());
        let four_over_two = (two.clone() * two.clone()).divide(&two).unwrap();
        assert!(!four_over_two.is_bound());
        assert_eq!(four_over_two, two);
        assert!(matches!(one().divide(&two), Err(MathError::InvalidArgument(_))));
    }

    #[test]
    #[should_panic(expected = "bind an operand")]
    fn unbound_quotient_panics() {
        let two = one() + one();
        let _ = one() / two;
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn division_by_unbound_zero_panics() {
        let _ = qp(3, 1, 5, 10) / <PadicRational as Ring>::zero();
    }

    // At c79a582 the valuation was u32::MAX as i32 = -1.
    #[test]
    fn from_rational_zero_has_valuation_zero() {
        let z = qp(0, 1, 5, 10);
        assert_eq!(z.valuation(), 0);
        assert_eq!(z.absolute_precision(), 10);
        assert!(z.is_zero());
    }

    #[test]
    #[should_panic(expected = "PadicRational::valuation")]
    fn unbound_valuation_refuses() {
        let _ = one().valuation();
    }

    #[test]
    #[should_panic(expected = "PadicRational::prime")]
    fn unbound_prime_refuses() {
        let _ = one().prime();
    }

    #[test]
    #[should_panic(expected = "PadicRational::abs")]
    fn unbound_abs_refuses() {
        let _ = one().abs();
    }

    #[test]
    fn unbound_rational_refuses() {
        assert!(matches!(one().lift(5), Err(MathError::InvalidArgument(_))));
        assert_eq!(format!("{}", one() + one()), "2");
    }

    #[test]
    fn pow_and_hooks_for_containers() {
        let x = qp(2, 5, 5, 6);
        let p0 = x.pow(0);
        assert!(p0.is_bound() && p0.is_one() && p0.precision() == 6);
        assert_eq!(x.pow(3), qp(8, 125, 5, 6));
        const { assert!(<PadicRational as Ring>::HAS_UNBOUND_CONSTANTS) };
        let b = one().bind_to_parent_of(&x).unwrap();
        assert!(b.is_bound() && b.is_one());
        assert_eq!(x.bind_to_parent_of(&x), None);
    }

    // gp: valuation(5, 5) = 1, so 5 and 1/5 have infinite order (Sage +Infinity);
    // Mod(2,6^4)^8 -> 256 (2 is not nilpotent in Q_6 ≅ Q_2 × Q_3).
    #[test]
    fn hooks_in_q5() {
        use MultiplicativeOrder::*;
        let five = from_int(1, 5, 10) + from_int(4, 5, 10); // stored (5, 0)
        assert_eq!(five.unit_status(), Some(true));
        assert_eq!(five.unit_inverse().unwrap(), qp(1, 5, 5, 10));
        assert_eq!(five.multiplicative_order_status(), Infinite);
        assert_eq!(qp(1, 5, 5, 10).multiplicative_order_status(), Infinite);
        assert_eq!(qp(-1, 1, 5, 10).multiplicative_order_status(), Finite(2));
        assert_eq!(qp(0, 1, 5, 10).multiplicative_order_status(), NotAUnit);
        assert_eq!(qp(2, 1, 5, 10).multiplicative_order_status(), Infinite);
        assert_eq!(qp(0, 1, 5, 10).unit_inverse(), Err(MathError::NotInvertible));
        assert_eq!(from_int(2, 6, 4).nilpotent_status(), Some(false));
        assert_eq!(from_int(2, 6, 4).unit_status(), None);
        assert!(matches!(from_int(2, 6, 4).unit_inverse(), Err(MathError::NotImplemented(_))));
        // p = 4 = 2^2: Q_4 = Q_2 is a field, so 2 + O(4^3) is a unit.
        assert_eq!(from_int(2, 4, 3).unit_status(), Some(true));

        let two = one() + one();
        assert_eq!(two.unit_status(), Some(true));
        assert!(matches!(two.unit_inverse(), Err(e) if e != MathError::NotInvertible));
        assert_eq!(two.multiplicative_order_status(), Infinite);
        assert_eq!(two.nilpotent_status(), Some(false));
        assert_eq!((-one()).multiplicative_order_status(), Finite(2));
        assert_eq!(<PadicRational as Ring>::static_characteristic(), Some(0));
    }
}

//! p-adic integers Z_p, truncated to a finite absolute precision.
//!
//! MAGMA Handbook Ch. 47 (p-adic rings): §47.7.1 (`L ! u`, coercion of an integer "with
//! maximum possible precision"), §47.8.1 (arithmetic), §47.8.2 (equality), §47.8.3
//! (`IsZero`, `IsOne`, `IsUnit`); Ch. 17 §17.5.2 (`R ! a` for an integer a is a·1_R) and
//! §17.5.5 (`IsUnit`, `IsNilpotent`). The precision conventions follow Sage
//! (doc.sagemath.org, `padic_capped_relative_element`, `padic_generic_element`): `is_zero`
//! means "zero to the known precision", and equality compares to the smaller precision.
//!
//! # Unbound constants
//!
//! The static `Ring::zero()` and `Ring::one()` cannot know p or the precision. They return
//! an **unbound** element: an exact integer constant c awaiting the canonical map Z -> Z_p
//! (MAGMA `R ! c`). It binds on contact with a bound element (see [`PadicInteger`]). The
//! map Z -> Z_p is injective (characteristic 0), so arithmetic on two unbound constants in
//! Z is exact.

use crate::category_methods::{cheap_primality, minimal_root};
use rustmath_core::{MathError, MultiplicativeOrder, Result, Ring};
use rustmath_integers::Integer;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// An element of Z_p (any integer p > 1) known modulo p^n: the value `u` in `[0, p^n)`
/// stands for every element of `u + p^n Z_p`.
///
/// # The model, and where it departs from MAGMA (Ch. 47)
///
/// The element approximates an element of Z_p. `is_zero`/`is_one` mean "indistinguishable
/// from 0/1 at the known precision" (Sage's `is_zero()`), and the ring predicates of
/// [`Ring`] are proofs for every element of Z_p that the approximation represents, under
/// the convention that `is_zero` and `is_one` are exact: an element indistinguishable from
/// 0 (from 1) is treated as 0 (as 1). So `5 + O(5^10)` is not nilpotent (Z_5 is a domain),
/// although its residue in Z/5^10 is; and `1 + O(5^10)` has order `Finite(1)` and
/// `-1 + O(5^10)` order `Finite(2)`, although the lift `1 + 5^10` has infinite order. Four
/// departures from the digest:
///
/// 1. **Equality.** MAGMA disables `eq` in free-precision rings (§47.8.2). Here `Ring:
///    PartialEq` is used by generic code, so `x == y` means that the values agree modulo
///    p^min(n_x, n_y) (Sage `is_equal_to`). It is not transitive: `1 + O(5^2)` equals both
///    `1 + O(5^10)` and `26 + O(5^10)`, which differ. The workspace has the same convention
///    for `PowerSeries`, and for the unbound corner of `GaloisRingElement`,
///    `FiniteFieldElement` and `IntegerMod`. No p-adic value is hashed.
/// 2. **`IsZero`, `IsOne`.** MAGMA answers false in free-precision rings (§47.8.3). Here
///    they answer "indistinguishable"; always false would make `R::zero().is_zero()` false
///    and break every generic zero test.
/// 3. **`IsUnit`.** [`Ring::unit_status`] is Ch. 17 §17.5.5 `IsUnit` ("a unit of its
///    parent ring"). In Z_p it agrees with §47.8.3 ("valuation 0").
/// 4. **Fixed precision.** MAGMA's fixed-precision ring is the quotient
///    `pAdicQuotientRing(p, k)` of characteristic p^k (§47.3.1). `PrecisionModel::
///    FixedModulus` follows Sage's fixed-mod Z_p instead (characteristic 0); the quotient
///    ring Z/p^k is `Zmod::new(p^k)`.
///
/// # Unbound constants (the static `zero()`/`one()`)
///
/// An unbound element holds an exact integer c (`prime == 0`, `precision == 0`), the image
/// of c under Z -> Z_p for a p not yet known (MAGMA `R ! c`, Ch. 17 §17.5.2, §47.7.1).
/// * `+`, `-`, `*` with one unbound operand c bind c into the other operand's ring at its
///   precision first, except that the exact zero stays exact: `x + 0` and `x - 0` are `x`,
///   `0 - x` is `-x`, and `0 * x`, `x * 0` are the unbound 0 (binding would give a zero of
///   finite precision). Two unbound operands combine in Z.
/// * `==` treats an unbound c as exact: `c == x` iff c ≡ x mod p^n_x.
/// * `is_zero`/`is_one` of an unbound c: `c == 0` / `c == 1`.
/// * [`PadicInteger::prime`], [`PadicInteger::precision`], [`PadicInteger::valuation`],
///   [`PadicInteger::residue`] panic on an unbound element, naming how to bind it;
///   `lift`, `truncate`, `inverse` (except of ±1 and 0) and `hensel_lift_linear` return
///   `Err(InvalidArgument)`. [`PadicInteger::value`] returns c, unreduced. `Display` prints c.
/// * Binding fixes the precision: `unbound(5^10)` bound at precision 10 is `0 + O(5^10)`.
#[derive(Clone, Debug)]
pub struct PadicInteger {
    /// Value modulo p^precision, in [0, p^precision); the integer c when unbound.
    value: Integer,
    /// p > 1; 0 for an unbound element.
    prime: Integer,
    /// Number of p-adic digits, > 0; 0 for an unbound element.
    precision: usize,
}

impl PadicInteger {
    /// Create a new p-adic integer from an integer
    ///
    /// # Arguments
    ///
    /// * `value` - Integer to convert to p-adic form
    /// * `prime` - Prime p
    /// * `precision` - Number of p-adic digits to keep
    pub fn from_integer(value: Integer, prime: Integer, precision: usize) -> Result<Self> {
        if prime <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Prime must be > 1".to_string(),
            ));
        }

        if precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        // Compute p^precision
        let modulus = prime.pow(precision as u32);

        // Reduce value modulo p^precision
        let reduced = value.clone() % modulus.clone();
        let value = if reduced.signum() < 0 {
            reduced + modulus
        } else {
            reduced
        };

        Ok(PadicInteger {
            value,
            prime,
            precision,
        })
    }

    /// Create zero
    pub fn zero(prime: Integer, precision: usize) -> Result<Self> {
        Self::from_integer(Integer::zero(), prime, precision)
    }

    /// Create one
    pub fn one(prime: Integer, precision: usize) -> Result<Self> {
        Self::from_integer(Integer::one(), prime, precision)
    }

    /// The unbound integer constant `c` (the static `Ring::zero()`/`one()` and everything
    /// computed from such values alone).
    pub(crate) fn unbound(c: Integer) -> Self {
        PadicInteger {
            value: c,
            prime: Integer::zero(),
            precision: 0,
        }
    }

    /// Whether this element has a prime and a precision. `false` only for the unbound
    /// constants of `Ring::zero()`/`Ring::one()` and values computed from them alone.
    pub fn is_bound(&self) -> bool {
        !self.prime.is_zero()
    }

    /// `other.parent ! self` for an unbound `self` (MAGMA `R ! c`): `self`'s constant
    /// reduced to `other`'s prime and precision. `other` must be bound.
    fn bind_like(&self, other: &PadicInteger) -> PadicInteger {
        debug_assert!(!self.is_bound() && other.is_bound());
        PadicInteger::from_integer(self.value.clone(), other.prime.clone(), other.precision)
            .expect("a bound element has p > 1 and precision > 0")
    }

    /// Bring two operands, at least one of them bound, to a common ring: an unbound one is
    /// bound into the other's ring.
    fn coerce_pair(self, other: Self) -> (Self, Self) {
        match (self.is_bound(), other.is_bound()) {
            (true, false) => {
                let o = other.bind_like(&self);
                (self, o)
            }
            (false, true) => {
                let s = self.bind_like(&other);
                (s, other)
            }
            _ => (self, other),
        }
    }

    fn require_bound(&self, method: &str) {
        if !self.is_bound() {
            panic!(
                "PadicInteger::{method}: the unbound constant {} (from Ring::zero()/one()) has \
                 no prime or precision; combine it with a bound element, or use \
                 PadicIntegerRing::zero()/one() or PadicInteger::zero(p, n)",
                self.value
            );
        }
    }

    fn unbound_error(&self, method: &str) -> MathError {
        MathError::InvalidArgument(format!(
            "PadicInteger::{method}: the unbound constant {} has no prime or precision; bind \
             it first (combine it with a bound element, or use PadicInteger::from_integer)",
            self.value
        ))
    }

    /// The value modulo p^precision, in [0, p^precision). For an unbound element, the
    /// unreduced integer constant.
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Get the prime p
    ///
    /// # Panics
    ///
    /// On an unbound element (see the type docs).
    pub fn prime(&self) -> &Integer {
        self.require_bound("prime");
        &self.prime
    }

    /// Get the precision
    ///
    /// # Panics
    ///
    /// On an unbound element (see the type docs).
    pub fn precision(&self) -> usize {
        self.require_bound("precision");
        self.precision
    }

    /// Get the p-adic valuation (largest k such that p^k divides the value)
    ///
    /// # Panics
    ///
    /// On an unbound element (see the type docs).
    pub fn valuation(&self) -> u32 {
        self.require_bound("valuation");
        if self.value.is_zero() {
            return u32::MAX; // Infinite valuation
        }

        self.value.valuation(&self.prime)
    }

    /// Get the residue (value modulo p)
    ///
    /// # Panics
    ///
    /// On an unbound element (see the type docs).
    pub fn residue(&self) -> Integer {
        self.require_bound("residue");
        let residue = self.value.clone() % self.prime.clone();
        if residue.signum() < 0 {
            residue + self.prime.clone()
        } else {
            residue
        }
    }

    /// Lift precision to a higher value
    pub fn lift(&self, new_precision: usize) -> Result<Self> {
        if !self.is_bound() {
            return Err(self.unbound_error("lift"));
        }
        if new_precision < self.precision {
            return Err(MathError::InvalidArgument(
                "New precision must be >= current precision".to_string(),
            ));
        }

        Ok(PadicInteger {
            value: self.value.clone(),
            prime: self.prime.clone(),
            precision: new_precision,
        })
    }

    /// Truncate to lower precision
    pub fn truncate(&self, new_precision: usize) -> Result<Self> {
        if !self.is_bound() {
            return Err(self.unbound_error("truncate"));
        }
        if new_precision > self.precision {
            return Err(MathError::InvalidArgument(
                "New precision must be <= current precision".to_string(),
            ));
        }

        if new_precision == 0 {
            return Err(MathError::InvalidArgument(
                "Precision must be > 0".to_string(),
            ));
        }

        let modulus = self.prime.pow(new_precision as u32);
        let reduced = self.value.clone() % modulus.clone();

        Ok(PadicInteger {
            value: reduced,
            prime: self.prime.clone(),
            precision: new_precision,
        })
    }

    /// Compute multiplicative inverse (if it exists)
    ///
    /// Inverse exists if and only if gcd(value, p) = 1. An unbound ±1 is its own inverse,
    /// the unbound 0 gives `Err(NotInvertible)`, any other unbound constant
    /// `Err(InvalidArgument)` (whether it is a unit depends on p).
    pub fn inverse(&self) -> Result<Self> {
        if !self.is_bound() {
            return if self.value.is_one() || (-self.value.clone()).is_one() {
                Ok(self.clone())
            } else if self.value.is_zero() {
                Err(MathError::NotInvertible)
            } else {
                Err(self.unbound_error("inverse"))
            };
        }
        let modulus = self.prime.pow(self.precision as u32);

        // Check if inverse exists
        let gcd = self.value.gcd(&self.prime);
        if !gcd.is_one() {
            return Err(MathError::NotInvertible);
        }

        // Use extended GCD to find inverse
        let (gcd_check, s, _) = self.value.extended_gcd(&modulus);
        if !gcd_check.is_one() {
            return Err(MathError::NotInvertible);
        }

        // s is the inverse, but might be negative
        let inv = if s.signum() < 0 {
            s + modulus.clone()
        } else {
            s % modulus.clone()
        };

        Ok(PadicInteger {
            value: inv,
            prime: self.prime.clone(),
            precision: self.precision,
        })
    }

    /// Hensel lifting for linear congruences: ax ≡ b (mod p^k)
    ///
    /// Lifts a solution from precision n to precision n+1
    /// This is a simplified version that works for the inverse operation.
    /// `self` must be bound; unbound `a`, `b` are bound into its ring.
    pub fn hensel_lift_linear(&self, a: &Self, b: &Self) -> Result<Self> {
        if !self.is_bound() {
            return Err(self.unbound_error("hensel_lift_linear"));
        }
        let bind = |y: &Self| if y.is_bound() { y.clone() } else { y.bind_like(self) };
        let (a, b) = (bind(a), bind(b));
        // Newton's method: x_{n+1} = x_n - f(x_n)/f'(x_n)
        // For f(x) = ax - b, we have f'(x) = a
        // So x_{n+1} = x_n - (ax_n - b)/a = x_n(2 - ax_n/b)

        if self.precision >= 50 {
            // Already at high precision
            return Ok(self.clone());
        }

        let new_precision = self.precision + 1;

        // Compute correction term
        let ax = a.clone() * self.clone();
        let residual = b - ax;

        // Lift: x' = x + residual * a^(-1)
        let a_inv = a.inverse()?;
        let correction = residual * a_inv;
        let lifted = self.clone() + correction;

        // Convert to higher precision
        PadicInteger::from_integer(lifted.value, self.prime.clone(), new_precision)
    }

    /// `u^(q-1) mod p^n`, for the order rule (bound elements only).
    fn power_mod_modulus(&self, e: &Integer) -> Integer {
        let modulus = self.prime.pow(self.precision as u32);
        self.value
            .mod_pow(e, &modulus)
            .expect("the modulus p^n is at least 2")
    }
}

/// Hensel lifting for polynomial roots
///
/// Given a polynomial f and an approximate root r such that f(r) ≡ 0 (mod p^k),
/// lift r to a root modulo p^(k+1).
///
/// This uses Newton's method: r' = r - f(r)/f'(r) mod p^(k+1)
///
/// Note: This is a simplified version. Full implementation would work with
/// polynomial types from rustmath-polynomials. `root` must be bound
/// (`Err(InvalidArgument)` otherwise); unbound values are bound into its ring.
pub fn hensel_lift_root(
    root: &PadicInteger,
    f_value: &PadicInteger,
    f_prime_value: &PadicInteger,
) -> Result<PadicInteger> {
    if !root.is_bound() {
        return Err(root.unbound_error("hensel_lift_root"));
    }
    let bind = |y: &PadicInteger| if y.is_bound() { y.clone() } else { y.bind_like(root) };
    let (f_value, f_prime_value) = (bind(f_value), bind(f_prime_value));
    if f_prime_value.is_zero() {
        return Err(MathError::InvalidArgument(
            "Derivative is zero at root".to_string(),
        ));
    }

    // Newton step: r' = r - f(r)/f'(r)
    let f_prime_inv = f_prime_value.inverse()?;
    let correction = f_value * f_prime_inv;
    let new_root = root.clone() - correction;

    // Lift precision
    let new_precision = root.precision() + 1;
    PadicInteger::from_integer(new_root.value, root.prime().clone(), new_precision)
}

impl fmt::Display for PadicInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.is_bound() {
            return write!(f, "{}", self.value);
        }
        write!(
            f,
            "{} + O({}^{})",
            self.value, self.prime, self.precision
        )
    }
}

/// Value equality to the common precision (see the type docs, point 1).
impl PartialEq for PadicInteger {
    fn eq(&self, other: &Self) -> bool {
        match (self.is_bound(), other.is_bound()) {
            (false, false) => self.value == other.value,
            (true, true) => {
                if self.prime != other.prime {
                    return false;
                }
                let modulus = self.prime.pow(self.precision.min(other.precision) as u32);
                (self.value.clone() - other.value.clone()).modulo(&modulus).is_zero()
            }
            (true, false) | (false, true) => {
                let (b, u) = if self.is_bound() { (self, other) } else { (other, self) };
                let modulus = b.prime.pow(b.precision as u32);
                (b.value.clone() - u.value.clone()).modulo(&modulus).is_zero()
            }
        }
    }
}

impl Add for PadicInteger {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // The exact zero leaves the other operand (and its precision) unchanged.
        if !other.is_bound() && other.value.is_zero() {
            return self;
        }
        if !self.is_bound() && self.value.is_zero() {
            return other;
        }
        if !self.is_bound() && !other.is_bound() {
            return PadicInteger::unbound(self.value + other.value);
        }
        let (a, b) = self.coerce_pair(other);
        assert_eq!(a.prime, b.prime, "Primes must match");
        let precision = a.precision.min(b.precision);
        let sum = a.value + b.value;
        PadicInteger::from_integer(sum, a.prime, precision).unwrap()
    }
}

impl Sub for PadicInteger {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        if !other.is_bound() && other.value.is_zero() {
            return self;
        }
        if !self.is_bound() && self.value.is_zero() {
            return -other;
        }
        if !self.is_bound() && !other.is_bound() {
            return PadicInteger::unbound(self.value - other.value);
        }
        let (a, b) = self.coerce_pair(other);
        assert_eq!(a.prime, b.prime, "Primes must match");
        let precision = a.precision.min(b.precision);
        let diff = a.value - b.value;
        PadicInteger::from_integer(diff, a.prime, precision).unwrap()
    }
}

impl Mul for PadicInteger {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // The exact zero times anything is the exact zero.
        if (!self.is_bound() && self.value.is_zero()) || (!other.is_bound() && other.value.is_zero())
        {
            return PadicInteger::unbound(Integer::zero());
        }
        if !self.is_bound() && !other.is_bound() {
            return PadicInteger::unbound(self.value * other.value);
        }
        let (a, b) = self.coerce_pair(other);
        assert_eq!(a.prime, b.prime, "Primes must match");
        let precision = a.precision.min(b.precision);
        let prod = a.value * b.value;
        PadicInteger::from_integer(prod, a.prime, precision).unwrap()
    }
}

impl Neg for PadicInteger {
    type Output = Self;

    fn neg(self) -> Self {
        if !self.is_bound() {
            return PadicInteger::unbound(-self.value);
        }
        let modulus = self.prime.pow(self.precision as u32);
        let neg_val = modulus - self.value;

        PadicInteger::from_integer(neg_val, self.prime, self.precision).unwrap()
    }
}

impl Ring for PadicInteger {
    /// The unbound constant 0 (see the type docs).
    fn zero() -> Self {
        PadicInteger::unbound(Integer::zero())
    }

    /// The unbound constant 1 (see the type docs).
    fn one() -> Self {
        PadicInteger::unbound(Integer::one())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Square-and-multiply. `x.pow(0)` of a bound `x` is the bound 1 at `x`'s precision; an
    /// unbound c gives the unbound c^n.
    fn pow(&self, n: u32) -> Self {
        if !self.is_bound() {
            return PadicInteger::unbound(self.value.pow(n));
        }
        if n == 0 {
            return PadicInteger::one(self.prime.clone(), self.precision)
                .expect("a bound element has p > 1 and precision > 0");
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

    /// `x = u + O(p^n)` is a unit of Z_p iff `gcd(u, p) = 1` (its image in Z/p^n is a unit,
    /// for every lift), any p > 1. Unbound c: `±1` units, `0` not, other c undecided (it
    /// depends on p).
    fn unit_status(&self) -> Option<bool> {
        if !self.is_bound() {
            return if self.value.is_zero() {
                Some(false)
            } else if self.value.is_one() || (-self.value.clone()).is_one() {
                Some(true)
            } else {
                None
            };
        }
        Some(self.value.gcd(&self.prime).is_one())
    }

    /// Zero (to its precision) is nilpotent (the convention); every other element is not:
    /// each lift is nonzero, and Z_p ≅ ∏_{q | p} Z_q is reduced for every p > 1. An
    /// unbound c is nilpotent iff c = 0 (Z ⊂ Z_p).
    fn nilpotent_status(&self) -> Option<bool> {
        Some(self.value.is_zero())
    }

    /// [`PadicInteger::inverse`]: the truncation of the Z_p inverse to precision n;
    /// `Err(NotInvertible)` for a proven non-unit; `Err(InvalidArgument)` for an unbound
    /// constant other than 0, ±1 (undecided).
    fn unit_inverse(&self) -> Result<Self> {
        self.inverse()
    }

    /// * zero or a non-unit: `NotAUnit`; `1`: `Finite(1)`; `-1`: `Finite(2)` (`-1 != 1` in
    ///   characteristic 0);
    /// * p = q^a with q a prime (checked when q < psi_12): for q odd, `u^(q-1) mod p^n != 1`
    ///   gives `Infinite` (the roots of unity of Q_q are μ_(q-1), and every lift y ≡ u has
    ///   y^(q-1) ≡ u^(q-1)); for q = 2 every unit other than ±1 gives `Infinite`
    ///   (μ(Q_2) = {±1}, and here x ≢ ±1 mod p^n);
    /// * otherwise `Undecided` (x is congruent to a root of unity to the working
    ///   precision; it may be that root or a nearby element of infinite order).
    ///
    /// Unbound c: `0` `NotAUnit`, `1` `Finite(1)`, `-1` `Finite(2)`, else `Undecided`.
    /// References: Sage `Qp(5)(2).multiplicative_order()` = +Infinity, `Qp(5)(-1)` = 2.
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
            return MultiplicativeOrder::Undecided;
        }
        if self.unit_status() != Some(true) {
            return MultiplicativeOrder::NotAUnit;
        }
        let (q, _) = minimal_root(&self.prime);
        if cheap_primality(&q) != Some(true) {
            return MultiplicativeOrder::Undecided;
        }
        if q == Integer::from(2) {
            return MultiplicativeOrder::Infinite;
        }
        if !self.power_mod_modulus(&(q - Integer::one())).is_one() {
            MultiplicativeOrder::Infinite
        } else {
            MultiplicativeOrder::Undecided
        }
    }

    /// `Some(0)`: Z embeds in Z_p for every p > 1 (Sage `characteristic()` "always 0").
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

    /// MAGMA `R ! c` at the model's prime and precision.
    fn bind_to_parent_of(&self, model: &Self) -> Option<Self> {
        (!self.is_bound() && model.is_bound()).then(|| self.bind_like(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64, p: i64, n: usize) -> PadicInteger {
        PadicInteger::from_integer(Integer::from(v), Integer::from(p), n).unwrap()
    }

    #[test]
    fn test_basic_ops() {
        let p = Integer::from(5);
        let prec = 3;

        let a = PadicInteger::from_integer(Integer::from(7), p.clone(), prec).unwrap();
        let b = PadicInteger::from_integer(Integer::from(11), p.clone(), prec).unwrap();

        // 7 + 11 = 18 in Z_5
        let sum = a.clone() + b.clone();
        assert_eq!(sum.value(), &Integer::from(18));

        // 7 * 11 = 77 in Z_5
        let prod = a.clone() * b.clone();
        assert_eq!(prod.value(), &Integer::from(77));
    }

    #[test]
    fn test_valuation() {
        let p = Integer::from(5);
        let prec = 10;

        // 25 = 5^2
        let a = PadicInteger::from_integer(Integer::from(25), p.clone(), prec).unwrap();
        assert_eq!(a.valuation(), 2);

        // 7 has valuation 0 (not divisible by 5)
        let b = PadicInteger::from_integer(Integer::from(7), p.clone(), prec).unwrap();
        assert_eq!(b.valuation(), 0);
    }

    #[test]
    fn test_inverse() {
        let p = Integer::from(5);
        let prec = 3;

        let a = PadicInteger::from_integer(Integer::from(2), p.clone(), prec).unwrap();
        let inv = a.inverse().unwrap();

        // Verify: a * a^(-1) ≡ 1 (mod 5^3)
        let prod = a * inv;
        assert!(prod.is_one());
    }

    #[test]
    fn test_truncate() {
        let p = Integer::from(5);
        let a = PadicInteger::from_integer(Integer::from(123), p.clone(), 5).unwrap();

        let truncated = a.truncate(2).unwrap();
        assert_eq!(truncated.precision(), 2);
    }

    // The static constructors no longer panic (at c79a582: "Cannot create
    // PadicInteger::zero() without parameters"); they are unbound and compare as exact.
    #[test]
    fn ring_zero_one_are_unbound() {
        let zero = <PadicInteger as Ring>::zero();
        let one = <PadicInteger as Ring>::one();
        assert!(!zero.is_bound() && !one.is_bound());
        assert!(zero.is_zero() && one.is_one());
        assert!(!zero.is_one() && !one.is_zero());
        let z54 = PadicInteger::zero(Integer::from(5), 4).unwrap();
        assert_eq!(zero, z54);
        assert_eq!(z54, zero);
        assert_eq!(one, z(1, 5, 4));
        assert_eq!(format!("{one}"), "1");
    }

    // Horner through the sentinels. gp: (1 + 2*7 + 3*7^2) % 5^4 -> 162.
    #[test]
    fn generic_horner_through_sentinels() {
        fn horner<R: Ring>(cs: &[R], x: &R) -> R {
            let mut acc = R::zero();
            for c in cs.iter().rev() {
                acc = acc * x.clone() + c.clone();
            }
            acc
        }
        let cs = [
            <PadicInteger as Ring>::one(),
            <PadicInteger as Ring>::one() + <PadicInteger as Ring>::one(),
            z(3, 5, 4),
        ];
        let r = horner(&cs, &z(7, 5, 4));
        assert!(r.is_bound());
        assert_eq!(r.value(), &Integer::from(162));
        assert_eq!(r.precision(), 4);
    }

    // At c79a582 `x.pow(0)` panicked in the static `one()`.
    #[test]
    fn pow_zero_is_bound_one() {
        let x = z(7, 5, 4);
        let one = x.pow(0);
        assert!(one.is_bound());
        assert_eq!(one, PadicInteger::one(Integer::from(5), 4).unwrap());
        assert_eq!(one.precision(), 4);
        // gp: Mod(7, 5^4)^5 -> Mod(557, 625).
        assert_eq!(x.pow(5).value(), &Integer::from(557));
        let two = <PadicInteger as Ring>::one() + <PadicInteger as Ring>::one();
        assert_eq!(two.pow(10).value(), &Integer::from(1024));
        assert!(!two.pow(10).is_bound());
    }

    // gp: Mod(26, 5^5) == Mod(1, 5^5) -> 0.
    #[test]
    fn equality_common_precision() {
        assert_eq!(z(1, 5, 10), z(1, 5, 5));
        assert_ne!(z(1, 5, 10), z(26, 5, 5));
        assert_eq!(z(26, 5, 10), z(1, 5, 2));
        assert_ne!(z(1, 5, 4), z(1, 7, 4));
    }

    #[test]
    fn zero_is_exact_in_sums_and_products() {
        let x = z(7, 5, 4);
        let zero = <PadicInteger as Ring>::zero();
        let p0 = zero.clone() * x.clone();
        let p1 = x.clone() * zero.clone();
        assert!(!p0.is_bound() && p0.is_zero());
        assert!(!p1.is_bound() && p1.is_zero());
        let s = x.clone() + zero.clone();
        assert_eq!(s, x);
        assert_eq!(s.precision(), 4);
        assert_eq!((zero.clone() - x.clone()).value(), &Integer::from(625 - 7));
        assert_eq!((x.clone() - zero).value(), &Integer::from(7));
    }

    #[test]
    #[should_panic(expected = "PadicInteger::prime")]
    fn unbound_prime_refuses() {
        let _ = <PadicInteger as Ring>::one().prime();
    }

    #[test]
    #[should_panic(expected = "PadicInteger::precision")]
    fn unbound_precision_refuses() {
        let _ = <PadicInteger as Ring>::one().precision();
    }

    #[test]
    #[should_panic(expected = "PadicInteger::valuation")]
    fn unbound_valuation_refuses() {
        let _ = <PadicInteger as Ring>::one().valuation();
    }

    #[test]
    #[should_panic(expected = "PadicInteger::residue")]
    fn unbound_residue_refuses() {
        let _ = <PadicInteger as Ring>::one().residue();
    }

    #[test]
    fn unbound_results_refuse() {
        let one = <PadicInteger as Ring>::one();
        assert!(matches!(one.lift(5), Err(MathError::InvalidArgument(_))));
        assert!(matches!(one.truncate(1), Err(MathError::InvalidArgument(_))));
        assert_eq!(one.inverse(), Ok(one.clone()));
        assert_eq!(<PadicInteger as Ring>::zero().inverse(), Err(MathError::NotInvertible));
        let two = one.clone() + one.clone();
        assert!(matches!(two.inverse(), Err(MathError::InvalidArgument(_))));
        assert!(matches!(
            hensel_lift_root(&one, &one, &one),
            Err(MathError::InvalidArgument(_))
        ));
    }

    // The three hooks for containers (F1): an unbound constant binds to a model's ring.
    #[test]
    fn unbound_constant_hooks() {
        const { assert!(<PadicInteger as Ring>::HAS_UNBOUND_CONSTANTS) };
        let one = <PadicInteger as Ring>::one();
        let x = z(7, 5, 4);
        assert!(one.is_unbound_constant() && !x.is_unbound_constant());
        let b = one.bind_to_parent_of(&x).unwrap();
        assert!(b.is_bound() && b.is_one() && b.precision() == 4);
        assert_eq!(x.bind_to_parent_of(&x), None);
        assert_eq!(one.bind_to_parent_of(&one), None);
        // 5^4 binds to 0 + O(5^4).
        let c = PadicInteger::unbound(Integer::from(625));
        assert!(c.bind_to_parent_of(&x).unwrap().is_zero());
    }

    // gp: lift(Mod(7,5^4)^-1) -> 268; Mod(2,5^10)^4 -> Mod(16, 9765625);
    // lift(teichmuller(2+O(5^10))) -> 6139557, Mod(6139557,5^10)^4 == 1 -> 1;
    // Mod(3, 2^5) is neither 1 nor -1; lift(Mod(5,6^4)^-1) -> 1037; Mod(2,6^4)^8 -> 256.
    // p = 4 and 25: Mod(3,4^3) -> Mod(3, 64) (not ±1); lift(Mod(2,25^2)^4) -> 16 (not 1).
    // Sage: Qp(5)(2).multiplicative_order() -> +Infinity, Qp(5)(-1) -> 2.
    #[test]
    fn hooks_in_z5() {
        let seven = z(7, 5, 4);
        assert_eq!(seven.unit_status(), Some(true));
        assert_eq!(seven.unit_inverse().unwrap().value(), &Integer::from(268));
        let ten = z(10, 5, 4);
        assert_eq!(ten.unit_status(), Some(false));
        assert_eq!(ten.nilpotent_status(), Some(false));
        assert_eq!(ten.unit_inverse(), Err(MathError::NotInvertible));
        assert_eq!(ten.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
        assert_eq!(z(0, 5, 4).nilpotent_status(), Some(true));

        use MultiplicativeOrder::*;
        assert_eq!(z(2, 5, 10).multiplicative_order_status(), Infinite);
        assert_eq!(z(6139557, 5, 10).multiplicative_order_status(), Undecided);
        assert_eq!(z(-1, 5, 10).multiplicative_order_status(), Finite(2));
        assert_eq!(z(1, 5, 10).multiplicative_order_status(), Finite(1));
        assert_eq!(z(5, 5, 10).multiplicative_order_status(), NotAUnit);
        assert_eq!(z(3, 2, 5).multiplicative_order_status(), Infinite);

        // Composite p = 6: Z_6 ≅ Z_2 × Z_3 is reduced.
        let two6 = z(2, 6, 4);
        assert_eq!(two6.unit_status(), Some(false));
        assert_eq!(two6.nilpotent_status(), Some(false));
        let five6 = z(5, 6, 4);
        assert_eq!(five6.unit_status(), Some(true));
        assert_eq!(five6.unit_inverse().unwrap().value(), &Integer::from(1037));
        assert_eq!(five6.multiplicative_order_status(), Undecided);

        // p = 4 = 2^2: Z_4 = Z_2, so 3 + O(4^3) (3 mod 64, not ±1) has infinite order.
        assert_eq!(z(3, 4, 3).multiplicative_order_status(), Infinite);
        // p = 25 = 5^2: 2^4 = 16 != 1 mod 25^2, so 2 has infinite order.
        assert_eq!(z(2, 25, 2).multiplicative_order_status(), Infinite);

        let one = <PadicInteger as Ring>::one();
        let two = one.clone() + one.clone();
        assert_eq!((-one.clone()).multiplicative_order_status(), Finite(2));
        assert_eq!(<PadicInteger as Ring>::zero().nilpotent_status(), Some(true));
        assert_eq!(two.unit_status(), None);
        assert_eq!(two.nilpotent_status(), Some(false));
        assert_eq!(two.multiplicative_order_status(), Undecided);
        assert_eq!(<PadicInteger as Ring>::static_characteristic(), Some(0));
        assert_eq!(<PadicInteger as Ring>::static_is_commutative(), Some(true));
    }
}

//! Core algebraic traits
//!
//! This module defines the fundamental algebraic structures used in mathematics:
//! - Magma: Set with binary operation
//! - Semigroup: Magma with associativity
//! - Monoid: Semigroup with identity
//! - Group: Monoid with inverses
//! - Ring: Abelian group with multiplication
//! - Field: Ring with multiplicative inverses
//! - EuclideanDomain: Ring with division algorithm
//! - Module: Generalization of vector spaces

use crate::error::{MathError, Result};
use std::any::Any;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A magma is a set with a binary operation
pub trait Magma: Sized {
    /// The binary operation
    fn op(&self, other: &Self) -> Self;
}

/// A semigroup is a magma where the operation is associative
pub trait Semigroup: Magma {}

/// A monoid is a semigroup with an identity element
pub trait Monoid: Semigroup {
    /// The identity element
    fn identity() -> Self;

    /// Check if this is the identity
    fn is_identity(&self) -> bool;
}

/// A group is a monoid where every element has an inverse
pub trait Group: Monoid {
    /// The inverse element
    fn inverse(&self) -> Self;
}

/// An abelian (commutative) group
pub trait AbelianGroup: Group {}

/// A ring is an abelian group under addition with an associative multiplication
pub trait Ring:
    Sized
    + Clone
    + Debug
    + Display
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// The additive identity (zero)
    fn zero() -> Self;

    /// The multiplicative identity (one)
    fn one() -> Self;

    /// Check if this element is zero
    fn is_zero(&self) -> bool;

    /// Check if this element is one
    fn is_one(&self) -> bool;

    /// Negate this element (additive inverse)
    fn negate(&self) -> Self {
        -self.clone()
    }

    /// Compute self raised to the power n (for non-negative n)
    fn pow(&self, n: u32) -> Self {
        if n == 0 {
            Self::one()
        } else if n == 1 {
            self.clone()
        } else {
            let half = self.pow(n / 2);
            let result = half.clone() * half;
            if n.is_multiple_of(2) {
                result
            } else {
                result * self.clone()
            }
        }
    }

    // --- Element predicates (MAGMA Handbook Ch. 17 §17.5.5 `IsUnit`, `IsNilpotent`;
    // Ch. 19 `Order`; Ch. 17 §17.4.2 `Characteristic`) ---------------------------------
    //
    // These are three-valued: `Some(true)` / `Some(false)` (or a decided
    // `MultiplicativeOrder`) are proofs; `None` / `MultiplicativeOrder::Undecided` mean
    // that this element type cannot decide (or, for a hook with a cost cap, did not decide
    // within it). A concrete ring that can decide overrides them
    // in its own `impl Ring`. The defaults use only the ring axioms, never a search, and
    // never a static constructor (`Self::one()` of a parent-carrying type is an unbound
    // integer constant: the Galois-ring, finite-field and p-adic element types; it may panic
    // for a type without such a sentinel), so they are total for every implementor.

    /// MAGMA `IsUnit(a)` (Ch. 17 §17.5.5): whether `self` has a two-sided inverse.
    ///
    /// Default (ring axioms only): `1` and `-1` are units in every ring; `0` is a non-unit
    /// (reached only when `0 != 1`, i.e. in a nonzero ring); anything else is `None`.
    fn unit_status(&self) -> Option<bool> {
        if self.is_one() || (-self.clone()).is_one() {
            Some(true)
        } else if self.is_zero() {
            Some(false)
        } else {
            None
        }
    }

    /// MAGMA `IsNilpotent(x)` (Ch. 17 §17.5.5): whether `self^k = 0` for some `k >= 1`.
    ///
    /// Default: `0` gives `Some(true)`; a proven unit gives `Some(false)` (if `u^k = 0` then
    /// `1 = u^-k u^k = 0`, and `0` was handled first); anything else is `None`.
    fn nilpotent_status(&self) -> Option<bool> {
        if self.is_zero() {
            Some(true)
        } else if self.unit_status() == Some(true) {
            Some(false)
        } else {
            None
        }
    }

    /// The inverse of a unit.
    ///
    /// `Ok(v)`: `v` is the inverse of `self` (for inexact types such as floating-point
    /// reals, `v` is the type's own `Field::inverse` at working precision).
    /// `Err(MathError::NotInvertible)`: `self` is a proven non-unit. Any other `Err` means
    /// undecided and is never a proof of non-invertibility.
    ///
    /// Default: `±1` are self-inverse (returned as `self.clone()`, so a parent-carrying
    /// element keeps its parent); a proven non-unit gives `NotInvertible`; anything else
    /// gives `NotImplemented`.
    fn unit_inverse(&self) -> Result<Self> {
        if self.is_one() || (-self.clone()).is_one() {
            Ok(self.clone())
        } else if self.unit_status() == Some(false) {
            Err(MathError::NotInvertible)
        } else {
            Err(MathError::NotImplemented(
                "unit_inverse: this ring element type does not compute this inverse".into(),
            ))
        }
    }

    /// MAGMA `Order(a)` of a ring element (Ch. 19: the multiplicative order of a unit, and
    /// `0` for a non-unit; Ch. 21 for finite fields). See [`MultiplicativeOrder`].
    ///
    /// Default (no search: a search that finds nothing never decides):
    /// * `1` gives `Finite(1)`;
    /// * `-1` (here `!= 1`) gives `Finite(2)` only when [`Ring::static_characteristic`]
    ///   is known and is neither 1 nor 2. Otherwise it is `Undecided`: `-1` may be a
    ///   deferred integer constant (the `IntegerMod`/`FiniteFieldElement` sentinels, or a
    ///   container of them) that later binds into characteristic 2, where its order is 1;
    /// * a proven non-unit gives `NotAUnit`;
    /// * anything else is `Undecided`.
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        if self.is_one() {
            MultiplicativeOrder::Finite(1)
        } else if (-self.clone()).is_one() {
            match Self::static_characteristic() {
                Some(c) if c != 1 && c != 2 => MultiplicativeOrder::Finite(2),
                _ => MultiplicativeOrder::Undecided,
            }
        } else if self.unit_status() == Some(false) {
            MultiplicativeOrder::NotAUnit
        } else {
            MultiplicativeOrder::Undecided
        }
    }

    /// The characteristic (MAGMA `Characteristic(R)`, Ch. 17 §17.4.2) shared by every ring
    /// this element type represents, when the type alone determines it: `Some(0)` for the
    /// Z, Q, R and C types.
    ///
    /// `None` (default) when it depends on runtime data (Z/mZ, GF(q), GR(p^a, d): ask the
    /// parent) or is unknown. A type returns `Some(c)` only if every value it can hold,
    /// including any deferred constant from `zero()`/`one()`, lies in a ring of
    /// characteristic `c`. A nonzero ring of characteristic 0 contains a copy of Z, so it is
    /// infinite.
    fn static_characteristic() -> Option<u64> {
        None
    }

    /// MAGMA `IsCommutative(R)` (Ch. 17 §17.4.3) for every ring this element type
    /// represents, when the type alone determines it.
    ///
    /// `Some(true)` only if every value the type can hold, including any deferred constant
    /// from `zero()`/`one()`, lies in a commutative ring. `None` (default) when it is unknown
    /// or depends on runtime data. Containers use it to apply results that hold only over
    /// commutative rings (for `UnivariatePolynomial`: Atiyah-Macdonald, *Introduction to
    /// Commutative Algebra*, Ch. 1, Exercise 2). It is a separate hook because a generic
    /// `impl<R: Ring>` cannot see the marker trait [`CommutativeRing`].
    fn static_is_commutative() -> Option<bool> {
        None
    }

    // --- Unbound integer constants (RustMath; MAGMA has none) -----------------------------
    //
    // The parent-carrying element types of rustmath-finitefields (`IntegerMod` and
    // `PrimeField` with modulus 0, `FiniteFieldElement`, `GaloisRingElement`,
    // `ExtensionField`) and of rustmath-rings (the p-adic `PadicInteger`, `PadicRational`,
    // `PadicExtensionElement`) return from the static `zero()`/`one()` an integer constant c
    // that awaits the canonical map Z -> R (MAGMA ch. 48 §48.2.3 and ch. 47 §47.7.1:
    // `R ! a`). Arithmetic binds such a constant on contact with a bound element. Containers
    // bind their unbound entries to the parent of a bound sibling with these three hooks.

    /// Whether values of this type can be unbound constants. Default `false`; containers skip
    /// all binding work for such types (a compile-time constant). A container over `R` sets it
    /// to `R::HAS_UNBOUND_CONSTANTS`.
    const HAS_UNBOUND_CONSTANTS: bool = false;

    /// Whether `self` is an unbound constant: a value without a parent that stands for the
    /// image of an integer under the canonical map Z -> R (the static `zero()`/`one()` of a
    /// parent-carrying type, and everything computed from such values alone).
    ///
    /// For a container over such a type (a polynomial in R\[x\] or R\[x, x^-1\]) it means that
    /// no coefficient is bound: the value is the image of an element of Z\[x\] (or
    /// Z\[x, x^-1\]). A zero container with no stored coefficient counts as unbound, since it
    /// is 0 in every ring. Default `false`.
    fn is_unbound_constant(&self) -> bool {
        false
    }

    /// The image of `self` under the canonical map into the parent of `model`.
    ///
    /// Contract: for a type with `HAS_UNBOUND_CONSTANTS`, this is `Some(..)` exactly when
    /// `self.is_unbound_constant()` and `!model.is_unbound_constant()`; it is `None`
    /// otherwise, and always `None` for types without unbound constants (the default). A
    /// container maps coefficient by coefficient (to the parent of one of `model`'s bound
    /// coefficients) and drops the coefficients that become zero.
    fn bind_to_parent_of(&self, model: &Self) -> Option<Self> {
        let _ = model;
        None
    }
}

/// The multiplicative order of a ring element, see [`Ring::multiplicative_order_status`]
/// (MAGMA `Order(a)`, Handbook Ch. 19 and Ch. 21).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiplicativeOrder {
    /// A unit; the least `n >= 1` with `x^n = 1`.
    Finite(u64),
    /// A unit of finite order at least `2^64` (only in rings with more than `2^64`
    /// elements). The exact value comes from the type's own `Integer`-valued inherent
    /// `multiplicative_order()`, which every type of `rustmath-finitefields` that can return
    /// this variant provides (`IntegerMod`, `PrimeField`, `FiniteFieldElement`,
    /// `GaloisRingElement`; `ExtensionField` through `as_element()`).
    FiniteExceedsU64,
    /// A unit with `x^n != 1` for every `n >= 1` (proven).
    Infinite,
    /// Not a unit, so the order is undefined (MAGMA `Order` returns 0 in Z/mZ, Ch. 19).
    NotAUnit,
    /// Not decided. Either the element type cannot decide, or deciding needs work beyond the
    /// hook's cost cap (the finite-ring types of rustmath-finitefields: a factorization of the
    /// group exponent with proven prime factors; their exact inherent `multiplicative_order()`
    /// decides). A capped `Undecided` can become a decided answer later in the same process,
    /// when cached work makes it cheap; both answers are honest.
    Undecided,
}

/// A commutative ring (multiplication is commutative)
pub trait CommutativeRing: Ring {}

/// An integral domain is a commutative ring with no zero divisors
pub trait IntegralDomain: CommutativeRing {}

/// A Euclidean domain is an integral domain with a division algorithm
pub trait EuclideanDomain: IntegralDomain {
    /// The Euclidean function (e.g., absolute value for integers, degree for polynomials)
    fn norm(&self) -> u64;

    /// Division with remainder: returns (quotient, remainder)
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)>;

    /// Greatest common divisor using Euclidean algorithm
    fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, r) = a.div_rem(&b).unwrap();
            a = b;
            b = r;
        }

        a
    }

    /// Least common multiple
    fn lcm(&self, other: &Self) -> Self {
        if self.is_zero() && other.is_zero() {
            return Self::zero();
        }
        let gcd = self.gcd(other);
        let (quotient, _) = (self.clone() * other.clone()).div_rem(&gcd).unwrap();
        quotient
    }

    /// Extended Euclidean algorithm: returns (gcd, s, t) where gcd = s*a + t*b
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let mut old_r = self.clone();
        let mut r = other.clone();
        let mut old_s = Self::one();
        let mut s = Self::zero();
        let mut old_t = Self::zero();
        let mut t = Self::one();

        while !r.is_zero() {
            let (quotient, remainder) = old_r.div_rem(&r).unwrap();
            old_r = r;
            r = remainder;

            let temp_s = s.clone();
            s = old_s.clone() - quotient.clone() * s;
            old_s = temp_s;

            let temp_t = t.clone();
            t = old_t.clone() - quotient * t;
            old_t = temp_t;
        }

        (old_r, old_s, old_t)
    }

    /// The unit `u` with `self = u · n`, where `n` is the canonical (normalized) associate
    /// of `self` (MAGMA `Normalize`, Handbook ch. 23 §23.5.1): `sign(self)` in Z, `self`
    /// itself in a field (whose canonical associate of a nonzero element is 1). `0` is its
    /// own canonical associate, so `0` gives `Some(one)`.
    ///
    /// `None` means this ring does not single out a canonical associate of this non-unit;
    /// callers must then leave the element as it is rather than guess.
    ///
    /// The default knows only the units (a unit's canonical associate is 1). A unit is
    /// recognised by an exact division `1 = q · self`, checked by multiplying back, so a
    /// lawless `div_rem` cannot make the default claim a non-unit is a unit. Rings with
    /// non-unit elements that have non-trivial associates (Z, ...) override it.
    fn normalization_unit(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::one());
        }
        match Self::one().div_rem(self) {
            Ok((q, r)) if r.is_zero() && (q.clone() * self.clone()).is_one() => Some(self.clone()),
            _ => None,
        }
    }

    /// Conversion hook for exact algorithms that generic code over `R: EuclideanDomain` may
    /// run only for one concrete coefficient type (stable Rust has no specialization). For
    /// example, `UnivariatePolynomial::gcd` in `rustmath-polynomials` computes a gcd over Q
    /// through Z[x] by Gauss' lemma when its coefficients are `rustmath_rationals::Rational`.
    ///
    /// The default `None` means "no such path": generic code then runs its generic algorithm,
    /// so a missing override costs speed only. An implementor that overrides it must return
    /// `Some(self)`, the element itself (so `downcast_ref::<Self>()` succeeds and reads this
    /// very value), and must override [`EuclideanDomain::from_any`] to accept a `Box<Self>`.
    /// A consumer uses the pair only for a type-specific algorithm that returns the same value
    /// as its generic one, and falls back to the generic algorithm when a conversion fails.
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    /// The inverse of [`EuclideanDomain::as_any`]: `Some(x)` when `value` holds a `Self`
    /// (`Box<Self>`), `None` otherwise. The default (no fast path) is always `None`.
    fn from_any(value: Box<dyn Any>) -> Option<Self> {
        drop(value);
        None
    }
}

/// A field is a commutative ring where every non-zero element has a multiplicative inverse
pub trait Field: CommutativeRing + Div<Output = Self> {
    /// Multiplicative inverse (reciprocal)
    fn inverse(&self) -> Result<Self>;

    /// Division
    fn divide(&self, other: &Self) -> Result<Self> {
        if other.is_zero() {
            Err(MathError::DivisionByZero)
        } else {
            Ok(self.clone() * other.inverse()?)
        }
    }
}

/// A module over a ring (generalization of vector space)
pub trait Module<R: Ring>:
    Sized
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
{
    /// Scalar multiplication
    fn scalar_mul(&self, scalar: &R) -> Self;

    /// The zero element
    fn zero() -> Self;

    /// Check if this is zero
    fn is_zero(&self) -> bool;
}

/// A vector space is a module over a field
pub trait VectorSpace<F: Field>: Module<F> {
    /// The dimension of the vector space (if finite)
    fn dimension() -> Option<usize>;
}

/// An algebra over a base ring `R`.
///
/// An algebra is a ring `A` equipped with a compatible `R`-module structure
/// (scalar multiplication by elements of the base ring), such that
/// `(r · a) * b = r · (a * b) = a * (r · b)`.
///
/// This is the canonical `Algebra` trait for the whole workspace
/// (`rustmath-algebras` re-exports it; the former duplicates in
/// `rustmath-algebras::traits` and `rustmath-rings::ring` are gone).
///
/// In RustMath's element-typed trait system the base ring is identified by
/// the type parameter `R`; `base_ring` returns a canonical representative
/// element of it (by default `R::zero()`).
pub trait Algebra<R: Ring>: Ring {
    /// A canonical representative element of the base ring.
    ///
    /// Defaults to `R::zero()`; parent-aware implementations that carry a
    /// runtime base ring should override this.
    fn base_ring(&self) -> R {
        R::zero()
    }

    /// Scalar multiplication by an element of the base ring
    fn scalar_mul(&self, scalar: &R) -> Self;

    /// The dimension of the algebra as an `R`-module, if finite and known
    fn dimension(&self) -> Option<usize> {
        None
    }
}

/// Trait for types that can be converted to/from standard Rust numeric types
pub trait NumericConversion {
    /// Convert from i64
    fn from_i64(n: i64) -> Self;

    /// Convert from u64
    fn from_u64(n: u64) -> Self;

    /// Try to convert to i64
    fn to_i64(&self) -> Option<i64>;

    /// Try to convert to u64
    fn to_u64(&self) -> Option<u64>;

    /// Try to convert to usize
    fn to_usize(&self) -> Option<usize>;

    /// Try to convert to f64 (may lose precision)
    fn to_f64(&self) -> Option<f64>;
}

// Implement Ring for standard integer types for convenience in tests
impl Ring for i32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn is_zero(&self) -> bool { *self == 0 }
    fn is_one(&self) -> bool { *self == 1 }

    // `i32` models Z, whose unit group is {±1} and which is a domain (MAGMA Ch. 17 §17.5.5,
    // Ch. 19). No negation here, so `i32::MIN` cannot overflow.
    fn unit_status(&self) -> Option<bool> { Some(*self == 1 || *self == -1) }
    fn nilpotent_status(&self) -> Option<bool> { Some(*self == 0) }
    fn unit_inverse(&self) -> Result<Self> {
        if *self == 1 || *self == -1 { Ok(*self) } else { Err(MathError::NotInvertible) }
    }
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        match *self {
            1 => MultiplicativeOrder::Finite(1),
            -1 => MultiplicativeOrder::Finite(2),
            _ => MultiplicativeOrder::NotAUnit,
        }
    }
    fn static_characteristic() -> Option<u64> { Some(0) }
    fn static_is_commutative() -> Option<bool> { Some(true) }
}

impl CommutativeRing for i32 {}
impl IntegralDomain for i32 {}

/// Euclid on magnitudes (Knuth, TAOCP vol. 2, 3rd ed., §4.5.2 Algorithm A); gcd(0, 0) = 0.
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Extended Euclid with truncated quotients in i128: the recurrence of
/// `num_integer::Integer::extended_gcd` and of `Integer::extended_gcd` (rustmath-integers),
/// with the triple negated when the last remainder is negative, so `g >= 0`.
/// `(0, 0)` gives `(0, 0, 0)`. Inputs are i64 values, so no step overflows i128.
fn xgcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 && b == 0 {
        return (0, 0, 0);
    }
    let (mut r0, mut r1, mut s0, mut s1, mut t0, mut t1) = (a, b, 1i128, 0i128, 0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if r0 < 0 {
        (-r0, -s0, -t0)
    } else {
        (r0, s0, t0)
    }
}

impl EuclideanDomain for i32 {
    /// `|self|`; total (`norm(i32::MIN)` = 2^31).
    fn norm(&self) -> u64 {
        u64::from(self.unsigned_abs())
    }

    /// Truncated division. `Err(DivisionByZero)` for 0 and `Err(NumericalError)` for
    /// `i32::MIN / -1`, whose quotient 2^31 does not fit in i32.
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        match (self.checked_div(*other), self.checked_rem(*other)) {
            (Some(q), Some(r)) => Ok((q, r)),
            _ => Err(MathError::NumericalError("i32::MIN / -1 overflows i32".to_string())),
        }
    }

    /// MAGMA `Gcd` (Handbook ch. 18 §18.6): non-negative, Euclid on the magnitudes.
    ///
    /// # Panics
    /// Only when the gcd is 2^31, which does not fit in i32: both inputs in {0, i32::MIN},
    /// not both 0.
    fn gcd(&self, other: &Self) -> Self {
        let g = gcd_u64(u64::from(self.unsigned_abs()), u64::from(other.unsigned_abs()));
        i32::try_from(g).expect("gcd(i32::MIN, 0) = 2^31 does not fit in i32")
    }

    /// MAGMA `Lcm` (§18.6): non-negative, `|a| / g · |b|` formed in u64; 0 when an input is 0.
    ///
    /// # Panics
    /// When the lcm exceeds i32::MAX.
    fn lcm(&self, other: &Self) -> Self {
        if *self == 0 || *other == 0 {
            return 0;
        }
        let (a, b) = (u64::from(self.unsigned_abs()), u64::from(other.unsigned_abs()));
        let l = (a / gcd_u64(a, b)) * b;
        i32::try_from(l).expect("lcm does not fit in i32")
    }

    /// MAGMA `Xgcd` (§18.6): `(g, x, y)` with `g = x·a + y·b >= 0`; the multipliers of
    /// `Integer::extended_gcd` (equal to PARI `gcdext`). The cofactors satisfy
    /// `|x| <= |b| / g` and `|y| <= |a| / g` (Knuth §4.5.2), so only `g` can leave i32.
    ///
    /// # Panics
    /// Only when `g` = 2^31: inputs `(MIN, 0)`, `(0, MIN)`, `(MIN, MIN)`.
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let (g, x, y) = xgcd_i128(i128::from(*self), i128::from(*other));
        let f = |v: i128| {
            i32::try_from(v).expect("extended_gcd: gcd = 2^31 does not fit in i32")
        };
        (f(g), f(x), f(y))
    }

    /// The sign (MAGMA `Normalize` over Z: the canonical associate is `|self|`).
    fn normalization_unit(&self) -> Option<Self> {
        Some(if *self < 0 { -1 } else { 1 })
    }
}

impl Ring for i64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn is_zero(&self) -> bool { *self == 0 }
    fn is_one(&self) -> bool { *self == 1 }

    // `i64` models Z, whose unit group is {±1} and which is a domain (MAGMA Ch. 17 §17.5.5,
    // Ch. 19). No negation here, so `i64::MIN` cannot overflow.
    fn unit_status(&self) -> Option<bool> { Some(*self == 1 || *self == -1) }
    fn nilpotent_status(&self) -> Option<bool> { Some(*self == 0) }
    fn unit_inverse(&self) -> Result<Self> {
        if *self == 1 || *self == -1 { Ok(*self) } else { Err(MathError::NotInvertible) }
    }
    fn multiplicative_order_status(&self) -> MultiplicativeOrder {
        match *self {
            1 => MultiplicativeOrder::Finite(1),
            -1 => MultiplicativeOrder::Finite(2),
            _ => MultiplicativeOrder::NotAUnit,
        }
    }
    fn static_characteristic() -> Option<u64> { Some(0) }
    fn static_is_commutative() -> Option<bool> { Some(true) }
}

impl CommutativeRing for i64 {}
impl IntegralDomain for i64 {}

impl EuclideanDomain for i64 {
    /// `|self|`; total (`norm(i64::MIN)` = 2^63).
    fn norm(&self) -> u64 {
        self.unsigned_abs()
    }

    /// Truncated division. `Err(DivisionByZero)` for 0 and `Err(NumericalError)` for
    /// `i64::MIN / -1`, whose quotient 2^63 does not fit in i64.
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        match (self.checked_div(*other), self.checked_rem(*other)) {
            (Some(q), Some(r)) => Ok((q, r)),
            _ => Err(MathError::NumericalError("i64::MIN / -1 overflows i64".to_string())),
        }
    }

    /// MAGMA `Gcd` (Handbook ch. 18 §18.6): non-negative, Euclid on the magnitudes.
    ///
    /// # Panics
    /// Only when the gcd is 2^63, which does not fit in i64: both inputs in {0, i64::MIN},
    /// not both 0.
    fn gcd(&self, other: &Self) -> Self {
        let g = gcd_u64(self.unsigned_abs(), other.unsigned_abs());
        i64::try_from(g).expect("gcd(i64::MIN, 0) = 2^63 does not fit in i64")
    }

    /// MAGMA `Lcm` (§18.6): non-negative, `|a| / g · |b|` formed in u128; 0 when an input is 0.
    ///
    /// # Panics
    /// When the lcm exceeds i64::MAX.
    fn lcm(&self, other: &Self) -> Self {
        if *self == 0 || *other == 0 {
            return 0;
        }
        let (a, b) = (self.unsigned_abs(), other.unsigned_abs());
        let l = u128::from(a / gcd_u64(a, b)) * u128::from(b);
        i64::try_from(l).expect("lcm does not fit in i64")
    }

    /// MAGMA `Xgcd` (§18.6): `(g, x, y)` with `g = x·a + y·b >= 0`; the multipliers of
    /// `Integer::extended_gcd` (equal to PARI `gcdext`). The cofactors satisfy
    /// `|x| <= |b| / g` and `|y| <= |a| / g` (Knuth §4.5.2), so only `g` can leave i64.
    ///
    /// # Panics
    /// Only when `g` = 2^63: inputs `(MIN, 0)`, `(0, MIN)`, `(MIN, MIN)`.
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let (g, x, y) = xgcd_i128(i128::from(*self), i128::from(*other));
        let f = |v: i128| {
            i64::try_from(v).expect("extended_gcd: gcd = 2^63 does not fit in i64")
        };
        (f(g), f(x), f(y))
    }

    /// The sign (MAGMA `Normalize` over Z: the canonical associate is `|self|`).
    fn normalization_unit(&self) -> Option<Self> {
        Some(if *self < 0 { -1 } else { 1 })
    }
}

impl NumericConversion for i32 {
    fn from_i64(n: i64) -> Self {
        n as i32
    }

    fn from_u64(n: u64) -> Self {
        n as i32
    }

    fn to_i64(&self) -> Option<i64> {
        Some(*self as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        if *self >= 0 {
            Some(*self as u64)
        } else {
            None
        }
    }

    fn to_usize(&self) -> Option<usize> {
        if *self >= 0 {
            Some(*self as usize)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl NumericConversion for i64 {
    fn from_i64(n: i64) -> Self {
        n
    }

    fn from_u64(n: u64) -> Self {
        n as i64
    }

    fn to_i64(&self) -> Option<i64> {
        Some(*self)
    }

    fn to_u64(&self) -> Option<u64> {
        if *self >= 0 {
            Some(*self as u64)
        } else {
            None
        }
    }

    fn to_usize(&self) -> Option<usize> {
        if *self >= 0 {
            Some(*self as usize)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::{MathError, MultiplicativeOrder, Ring};
    use std::fmt;
    use std::ops::{Add, Mul, Neg, Sub};

    // The unbound-constant hooks (F1 item 7) are inert for every type that keeps the
    // defaults: no unbound constants, and nothing to bind.
    #[test]
    fn default_hooks_are_inert() {
        const { assert!(!<i32 as Ring>::HAS_UNBOUND_CONSTANTS) };
        const { assert!(!<i64 as Ring>::HAS_UNBOUND_CONSTANTS) };
        assert!(!5i32.is_unbound_constant());
        assert!(!0i64.is_unbound_constant());
        assert_eq!(5i32.bind_to_parent_of(&3), None);
        assert_eq!(1i64.bind_to_parent_of(&-1), None);
    }

    #[test]
    fn test_pow() {
        assert_eq!(2i32.pow(0), 1);
        assert_eq!(2i32.pow(1), 2);
        assert_eq!(2i32.pow(10), 1024);
        assert_eq!(3i32.pow(4), 81);
    }

    // i32/i64 model Z: units {±1} (GAP: IsUnit(Integers, 2) = false, IsUnit(Integers, -1) =
    // true, Order(-1) = 2).
    #[test]
    fn test_i32_i64_element_predicates() {
        assert_eq!(1i32.unit_status(), Some(true));
        assert_eq!((-1i32).unit_status(), Some(true));
        assert_eq!(2i32.unit_status(), Some(false));
        assert_eq!(0i32.unit_status(), Some(false));
        assert_eq!(0i32.nilpotent_status(), Some(true));
        assert_eq!(2i32.nilpotent_status(), Some(false));
        assert_eq!(1i32.multiplicative_order_status(), MultiplicativeOrder::Finite(1));
        assert_eq!((-1i32).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
        assert_eq!(2i32.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
        assert_eq!(2i32.unit_inverse(), Err(MathError::NotInvertible));
        assert_eq!((-1i32).unit_inverse(), Ok(-1));
        assert_eq!(<i32 as Ring>::static_characteristic(), Some(0));
        // No overflow on i32::MIN (the defaults would negate it).
        assert_eq!(i32::MIN.unit_status(), Some(false));
        assert_eq!(i32::MIN.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);

        assert_eq!((-1i64).unit_status(), Some(true));
        assert_eq!(7i64.unit_status(), Some(false));
        assert_eq!((-1i64).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
        assert_eq!(7i64.unit_inverse(), Err(MathError::NotInvertible));
        assert_eq!(<i64 as Ring>::static_characteristic(), Some(0));
        // Z is commutative (MAGMA IsCommutative(Integers()) = true).
        assert_eq!(<i32 as Ring>::static_is_commutative(), Some(true));
        assert_eq!(<i64 as Ring>::static_is_commutative(), Some(true));
    }

    /// Test-local `Z/6Z` rings that pin the trait defaults. `$panicky` makes the static
    /// constructors panic, like a parent-carrying type without a sentinel, to prove the
    /// defaults never call them.
    macro_rules! z6_ring {
        ($name:ident, $panicky:expr, $chr:expr) => {
            #[derive(Clone, Copy, Debug, PartialEq)]
            struct $name(u8);
            impl fmt::Display for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{}", self.0)
                }
            }
            impl Add for $name {
                type Output = Self;
                fn add(self, o: Self) -> Self {
                    $name((self.0 + o.0) % 6)
                }
            }
            impl Sub for $name {
                type Output = Self;
                fn sub(self, o: Self) -> Self {
                    $name((self.0 + 6 - o.0) % 6)
                }
            }
            impl Mul for $name {
                type Output = Self;
                fn mul(self, o: Self) -> Self {
                    $name((self.0 * o.0) % 6)
                }
            }
            impl Neg for $name {
                type Output = Self;
                fn neg(self) -> Self {
                    $name((6 - self.0) % 6)
                }
            }
            impl Ring for $name {
                fn zero() -> Self {
                    if $panicky {
                        panic!("static zero() called");
                    }
                    $name(0)
                }
                fn one() -> Self {
                    if $panicky {
                        panic!("static one() called");
                    }
                    $name(1)
                }
                fn is_zero(&self) -> bool {
                    self.0 == 0
                }
                fn is_one(&self) -> bool {
                    self.0 == 1
                }
                fn static_characteristic() -> Option<u64> {
                    $chr
                }
            }
        };
    }
    z6_ring!(Z6, false, None);
    z6_ring!(Z6Char, false, Some(6));
    z6_ring!(Z6Panicky, true, None);

    // Reference (PARI/GP, `echo '...' | gp -q`):
    //   [x | x <- [0..5], gcd(x,6)==1]      -> [1, 5]
    //   znorder(Mod(5,6))                   -> 2
    //   [x | x <- [0..5], Mod(x,6)^6 == 0]  -> [0]
    // The defaults decide only what the ring axioms prove; 2, 3, 4 stay undecided.
    #[test]
    fn test_default_predicates_on_z6() {
        assert_eq!(Z6(1).unit_status(), Some(true));
        assert_eq!(Z6(5).unit_status(), Some(true)); // 5 = -1
        assert_eq!(Z6(0).unit_status(), Some(false));
        assert_eq!(Z6(0).nilpotent_status(), Some(true));
        assert_eq!(Z6(5).nilpotent_status(), Some(false));
        assert_eq!(Z6(1).multiplicative_order_status(), MultiplicativeOrder::Finite(1));
        assert_eq!(Z6(0).multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
        assert_eq!(Z6(0).unit_inverse(), Err(MathError::NotInvertible));
        assert_eq!(Z6(5).unit_inverse(), Ok(Z6(5)));
        assert_eq!(Z6(1).unit_inverse(), Ok(Z6(1)));
        for x in [2u8, 3, 4] {
            assert_eq!(Z6(x).unit_status(), None);
            assert_eq!(Z6(x).nilpotent_status(), None);
            assert_eq!(Z6(x).multiplicative_order_status(), MultiplicativeOrder::Undecided);
            assert!(matches!(Z6(x).unit_inverse(), Err(MathError::NotImplemented(_))));
        }
        assert_eq!(<Z6 as Ring>::static_characteristic(), None);
        // -1 with unknown characteristic: order 1 or 2, so undecided.
        assert_eq!(Z6(5).multiplicative_order_status(), MultiplicativeOrder::Undecided);
        // With the characteristic known (6, not 1 or 2) the default proves order 2.
        assert_eq!(Z6Char(5).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
        assert_eq!(Z6Char(2).multiplicative_order_status(), MultiplicativeOrder::Undecided);
    }

    // The defaults must not call `Self::zero()`/`Self::one()` (they may panic for a
    // parent-carrying type without a sentinel).
    #[test]
    fn test_defaults_never_call_static_constructors() {
        for x in 0u8..6 {
            let e = Z6Panicky(x);
            let _ = e.unit_status();
            let _ = e.nilpotent_status();
            let _ = e.unit_inverse();
            let _ = e.multiplicative_order_status();
        }
        assert_eq!(Z6Panicky(5).unit_status(), Some(true));
        assert_eq!(Z6Panicky(3).nilpotent_status(), None);
    }

    // --- EuclideanDomain::normalization_unit (MAGMA `Normalize`, Handbook ch. 23 §23.5.1) ---
    //
    // Reference (PARI/GP): `sign(-6)` -> -1, `sign(6)` -> 1, `sign(0)` -> 0 (the canonical
    // associate of 0 is 0 itself, so its unit is 1); `Mod(3,7)^-1` -> Mod(5, 7).

    use super::{CommutativeRing, EuclideanDomain, IntegralDomain, Result as MathResult};

    #[test]
    fn test_normalization_unit_i32_i64_is_the_sign() {
        assert_eq!((-6i32).normalization_unit(), Some(-1));
        assert_eq!(6i32.normalization_unit(), Some(1));
        assert_eq!(0i32.normalization_unit(), Some(1));
        assert_eq!((-1i32).normalization_unit(), Some(-1));
        assert_eq!(i32::MIN.normalization_unit(), Some(-1));
        assert_eq!((-6i64).normalization_unit(), Some(-1));
        assert_eq!(6i64.normalization_unit(), Some(1));
        assert_eq!(0i64.normalization_unit(), Some(1));
        assert_eq!((-1i64).normalization_unit(), Some(-1));
        assert_eq!(i64::MIN.normalization_unit(), Some(-1));
    }

    /// GF(7) with `div_rem(a, b) = (a·b⁻¹, 0)`, using only the trait default.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct F7(u8);
    impl fmt::Display for F7 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Add for F7 {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            F7((self.0 + o.0) % 7)
        }
    }
    impl Sub for F7 {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            F7((self.0 + 7 - o.0) % 7)
        }
    }
    impl Mul for F7 {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            F7((self.0 * o.0) % 7)
        }
    }
    impl Neg for F7 {
        type Output = Self;
        fn neg(self) -> Self {
            F7((7 - self.0) % 7)
        }
    }
    impl Ring for F7 {
        fn zero() -> Self {
            F7(0)
        }
        fn one() -> Self {
            F7(1)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_one(&self) -> bool {
            self.0 == 1
        }
    }
    impl CommutativeRing for F7 {}
    impl IntegralDomain for F7 {}
    impl EuclideanDomain for F7 {
        fn norm(&self) -> u64 {
            u64::from(self.0 != 0)
        }
        fn div_rem(&self, other: &Self) -> MathResult<(Self, Self)> {
            if other.is_zero() {
                return Err(MathError::DivisionByZero);
            }
            let inv = (1u8..7).find(|&k| (k * other.0) % 7 == 1).unwrap();
            Ok((F7((self.0 * inv) % 7), F7(0)))
        }
    }

    /// Z through i64 arithmetic but *without* the sign override: only the trait default.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ZDefault(i64);
    impl fmt::Display for ZDefault {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Add for ZDefault {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            ZDefault(self.0 + o.0)
        }
    }
    impl Sub for ZDefault {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            ZDefault(self.0 - o.0)
        }
    }
    impl Mul for ZDefault {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            ZDefault(self.0 * o.0)
        }
    }
    impl Neg for ZDefault {
        type Output = Self;
        fn neg(self) -> Self {
            ZDefault(-self.0)
        }
    }
    impl Ring for ZDefault {
        fn zero() -> Self {
            ZDefault(0)
        }
        fn one() -> Self {
            ZDefault(1)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_one(&self) -> bool {
            self.0 == 1
        }
    }
    impl CommutativeRing for ZDefault {}
    impl IntegralDomain for ZDefault {}
    impl EuclideanDomain for ZDefault {
        fn norm(&self) -> u64 {
            self.0.unsigned_abs()
        }
        fn div_rem(&self, other: &Self) -> MathResult<(Self, Self)> {
            if other.is_zero() {
                return Err(MathError::DivisionByZero);
            }
            Ok((ZDefault(self.0 / other.0), ZDefault(self.0 % other.0)))
        }
    }

    #[test]
    fn test_normalization_unit_default_on_a_field() {
        // Every nonzero element of a field is a unit: its canonical associate is 1.
        assert_eq!(F7(3).normalization_unit(), Some(F7(3)));
        assert_eq!(F7(6).normalization_unit(), Some(F7(6)));
        assert_eq!(F7(1).normalization_unit(), Some(F7(1)));
        assert_eq!(F7(0).normalization_unit(), Some(F7(1)));
    }

    #[test]
    fn test_normalization_unit_default_knows_only_units() {
        // Units of Z are found; a non-unit gets the honest `None` (the default has no
        // notion of sign), never a guessed unit.
        assert_eq!(ZDefault(1).normalization_unit(), Some(ZDefault(1)));
        assert_eq!(ZDefault(-1).normalization_unit(), Some(ZDefault(-1)));
        assert_eq!(ZDefault(0).normalization_unit(), Some(ZDefault(1)));
        assert_eq!(ZDefault(2).normalization_unit(), None);
        assert_eq!(ZDefault(-6).normalization_unit(), None);
    }

    #[test]
    fn test_as_any_default_offers_no_fast_path() {
        // The default: no type-erased view, and no conversion back, even from a `Box<Self>`.
        assert!(F7(3).as_any().is_none());
        assert!(ZDefault(5).as_any().is_none());
        assert!(6i64.as_any().is_none());
        assert!(6i32.as_any().is_none());
        assert_eq!(<F7 as EuclideanDomain>::from_any(Box::new(F7(3))), None);
        assert_eq!(<i64 as EuclideanDomain>::from_any(Box::new(6i64)), None);
    }

    // MAGMA Gcd / Lcm / Xgcd on machine words (Handbook ch. 18 §18.6): non-negative gcd and lcm,
    // g >= 0 in Xgcd. Reference values from PARI/GP 2.15.4:
    //   gp -q: [gcd(4,-6),gcd(0,-5),gcd(-5,0),gcd(0,0)] -> [2, 5, 5, 0]
    //          [lcm(-4,6),lcm(-4,-6),lcm(0,-5)] -> [12, 12, 0]
    //          [gcdext(3,-7),gcdext(4,-6),gcdext(-4,-6),gcdext(0,-5),gcdext(5,-5),gcdext(-5,5)]
    //            -> [[-2,-1,1],[-1,-1,2],[1,-1,2],[0,-1,5],[0,-1,5],[0,1,5]]
    // (gcdext(a, b) = [u, v, d] with d = u·a + v·b, so extended_gcd(a, b) = (d, u, v).)
    fn check_signs<T>()
    where
        T: EuclideanDomain + Copy + fmt::Debug + From<i8>,
    {
        let t = |v: i8| T::from(v);
        assert_eq!(t(4).gcd(&t(-6)), t(2));
        assert_eq!(t(0).gcd(&t(-5)), t(5));
        assert_eq!(t(-5).gcd(&t(0)), t(5));
        assert_eq!(t(0).gcd(&t(0)), t(0));
        assert_eq!(t(-4).lcm(&t(6)), t(12));
        assert_eq!(t(-4).lcm(&t(-6)), t(12));
        assert_eq!(t(0).lcm(&t(-5)), t(0));
        assert_eq!(t(3).extended_gcd(&t(-7)), (t(1), t(-2), t(-1)));
        assert_eq!(t(4).extended_gcd(&t(-6)), (t(2), t(-1), t(-1)));
        assert_eq!(t(-4).extended_gcd(&t(-6)), (t(2), t(1), t(-1)));
        assert_eq!(t(0).extended_gcd(&t(-5)), (t(5), t(0), t(-1)));
        assert_eq!(t(5).extended_gcd(&t(-5)), (t(5), t(0), t(-1)));
        assert_eq!(t(-5).extended_gcd(&t(5)), (t(5), t(0), t(1)));
        assert_eq!(t(0).extended_gcd(&t(0)), (t(0), t(0), t(0)));
    }

    #[test]
    fn test_i32_i64_gcd_lcm_xgcd_signs_match_pari() {
        check_signs::<i32>();
        check_signs::<i64>();
    }

    #[test]
    fn test_i64_euclidean_extremes() {
        // gp -q: gcd(-2^63,6) -> 2; gcdext(-2^63,3) -> [1, 3074457345618258603, 1];
        //        gcdext(-2^63,-1) -> [0,-1,1]; gcdext(2^63-1,-2^63) -> [-1,-1,1];
        //        lcm(-2^61,3) -> 6917529027641081856
        let m = i64::MIN;
        assert_eq!(m.gcd(&6), 2);
        assert_eq!(m.gcd(&-1), 1);
        assert_eq!(m.extended_gcd(&3), (1, 1, 3_074_457_345_618_258_603));
        assert_eq!(m.extended_gcd(&-1), (1, 0, -1));
        assert_eq!(i64::MAX.extended_gcd(&m), (1, -1, -1));
        assert_eq!((-(1i64 << 61)).lcm(&3), 6_917_529_027_641_081_856);
        assert_eq!((1i64 << 40).lcm(&(1i64 << 40)), 1i64 << 40);
        assert_eq!(m.norm(), 1u64 << 63);
        assert!(matches!(m.div_rem(&-1), Err(MathError::NumericalError(_))));
        assert_eq!(m.div_rem(&2).unwrap(), (-(1i64 << 62), 0));
        assert!(matches!(5i64.div_rem(&0), Err(MathError::DivisionByZero)));
    }

    #[test]
    fn test_i32_euclidean_extremes() {
        // gp -q: gcdext(-2^31,3) -> [1, 715827883, 1]; gcdext(2^31-1,-2^31) -> [-1,-1,1];
        //        lcm(-2^29,3) -> 1610612736
        let m = i32::MIN;
        assert_eq!(m.gcd(&6), 2);
        assert_eq!(m.gcd(&-1), 1);
        assert_eq!(m.extended_gcd(&3), (1, 1, 715_827_883));
        assert_eq!(i32::MAX.extended_gcd(&m), (1, -1, -1));
        assert_eq!(65536i32.lcm(&65536), 65536);
        assert_eq!((-(1i32 << 29)).lcm(&3), 1_610_612_736);
        assert_eq!(m.norm(), 1u64 << 31);
        assert!(matches!(m.div_rem(&-1), Err(MathError::NumericalError(_))));
    }

    // gp -q: gcd(-2^63,0) -> 9223372036854775808 > i64::MAX
    #[test]
    #[should_panic(expected = "does not fit in i64")]
    fn test_i64_gcd_min_zero_panics() {
        let _ = i64::MIN.gcd(&0);
    }

    // gp -q: lcm(-2^62,3) -> 13835058055282163712 > i64::MAX
    #[test]
    #[should_panic(expected = "does not fit in i64")]
    fn test_i64_lcm_overflow_panics() {
        let _ = (-(1i64 << 62)).lcm(&3);
    }

    // gp -q: gcdext(-2^63,0)[3] -> 9223372036854775808 > i64::MAX
    #[test]
    #[should_panic(expected = "does not fit in i64")]
    fn test_i64_xgcd_min_zero_panics() {
        let _ = i64::MIN.extended_gcd(&0);
    }

    // gp -q: gcd(-2^31,0) -> 2147483648 > i32::MAX
    #[test]
    #[should_panic(expected = "does not fit in i32")]
    fn test_i32_gcd_min_zero_panics() {
        let _ = i32::MIN.gcd(&0);
    }

    // gp -q: lcm(-2^30,3) -> 3221225472 > i32::MAX
    #[test]
    #[should_panic(expected = "does not fit in i32")]
    fn test_i32_lcm_overflow_panics() {
        let _ = (-(1i32 << 30)).lcm(&3);
    }
}

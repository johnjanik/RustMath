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
    // that this element type cannot decide. A concrete ring that can decide overrides them
    // in its own `impl Ring`. The defaults use only the ring axioms, never a search, and
    // never a static constructor (`Self::one()` panics for parent-carrying types such as
    // Galois-ring and p-adic elements), so they are total for every implementor.

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
    /// This element type cannot decide.
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
}

impl CommutativeRing for i32 {}
impl IntegralDomain for i32 {}

impl EuclideanDomain for i32 {
    fn norm(&self) -> u64 {
        self.abs() as u64
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok((self / other, self % other))
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
}

impl CommutativeRing for i64 {}
impl IntegralDomain for i64 {}

impl EuclideanDomain for i64 {
    fn norm(&self) -> u64 {
        self.abs() as u64
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok((self / other, self % other))
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
    }

    /// Test-local `Z/6Z` rings that pin the trait defaults. `$panicky` makes the static
    /// constructors panic, like `GaloisRingElement` and the p-adic types, to prove the
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

    // The defaults must not call `Self::zero()`/`Self::one()` (they panic for
    // parent-carrying types).
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
}

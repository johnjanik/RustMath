//! # Ring Category Methods
//!
//! This module provides category-theoretic methods for rings, mirroring SageMath's
//! `sage.categories.rings` module structure. It organizes methods into four main categories:
//!
//! - **ElementMethods**: Methods for ring elements (units, divisibility, inverses)
//! - **ParentMethods**: Methods for ring structures (properties, constructions, ideals)
//! - **MorphismMethods**: Methods for ring homomorphisms (injectivity, extensions)
//! - **SubcategoryMethods**: Methods for subcategories (domains, fields, commutative rings)
//!
//! ## Overview
//!
//! SageMath's category system organizes methods based on their mathematical context.
//! This module provides the Rust equivalent using traits that can be implemented by
//! various ring types.
//!
//! ## Honest defaults (MAGMA Handbook Ch. 17, Ch. 16, Ch. 54)
//!
//! The parent predicates of [`RingParentMethods`] are three-valued like MAGMA's generic ring
//! predicates (Ch. 17 §17.4.3 `IsCommutative`, `IsDomain`, `IsField`, `IsFinite`: "true if
//! known; false if known not; error if unknown"), with `Characteristic` (§17.4.2),
//! `PrimeField` (§17.4.1) and `Random` (§17.5.2). The morphism predicates of
//! [`RingMorphismMethods`] are MAGMA `IsInjective` / `IsSurjective` (Ch. 54 §54.10.7; Ch. 16
//! §16.3.2 `Kernel`). `None` and `Err` make no claim; a default never fabricates an answer.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::category_methods::{RingElementMethods, RingParentMethods};
//! ```

use rustmath_core::{Ring, EuclideanDomain, MathError, Result};
use rustmath_integers::Integer;

// ============================================================================
// ElementMethods - Methods for ring elements
// ============================================================================

/// Methods for ring elements, corresponding to SageMath's `Rings.ElementMethods`
///
/// This trait provides operations on ring elements including:
/// - Testing if an element is a unit
/// - Computing multiplicative inverses
/// - Exact division operations
///
/// Reference: https://doc.sagemath.org/html/en/reference/categories/sage/categories/rings.html#sage.categories.rings.Rings.ElementMethods
pub trait RingElementMethods: Ring {
    /// Check if this element is a unit (has a multiplicative inverse); MAGMA `IsUnit`
    /// (Handbook Ch. 17 §17.5.5).
    ///
    /// An element `a` in a ring `R` is a unit if there exists `b` in `R`
    /// such that `a * b = b * a = 1`.
    ///
    /// Three-valued: `Some(true)` / `Some(false)` are decided, `None` means this element
    /// type cannot decide. It is [`Ring::unit_status`], which every concrete ring decides
    /// in its own `impl Ring` (every nonzero element of Q, R, C; gcd with the modulus in
    /// Z/mZ; valuation zero in a Galois ring).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::category_methods::RingElementMethods;
    ///
    /// // In integers, only ±1 are units
    /// assert_eq!(1i32.is_unit_element(), Some(true));
    /// assert_eq!((-1i32).is_unit_element(), Some(true));
    /// assert_eq!(2i32.is_unit_element(), Some(false));
    /// ```
    fn is_unit_element(&self) -> Option<bool> {
        self.unit_status()
    }

    /// Return the multiplicative inverse of this element if it is a unit
    ///
    /// It is [`Ring::unit_inverse`]: each concrete ring computes the inverse in its own
    /// `impl Ring`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::category_methods::RingElementMethods;
    ///
    /// // In integers, only ±1 have inverses
    /// assert_eq!(1i32.inverse_of_unit().unwrap(), 1);
    /// assert_eq!((-1i32).inverse_of_unit().unwrap(), -1);
    /// assert!(2i32.inverse_of_unit().is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// `MathError::NotInvertible` means the element is a proven non-unit. Any other error
    /// (`NotImplemented`, `InvalidArgument`, `DomainError`) means undecided: it is never a
    /// proof of non-invertibility.
    fn inverse_of_unit(&self) -> Result<Self> {
        self.unit_inverse()
    }

    /// Divide `self` by `other` if the division is exact (no remainder)
    ///
    /// This method attempts to compute `self / other` and returns an error
    /// if the division is not exact (i.e., if there would be a remainder).
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_rings::category_methods::RingElementMethods;
    ///
    /// assert_eq!(6i32.divide_if_possible(&2).unwrap(), 3);
    /// assert_eq!(6i32.divide_if_possible(&3).unwrap(), 2);
    /// assert!(7i32.divide_if_possible(&2).is_err());  // Not exact
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `MathError::NotDivisible` if the division is not exact or if
    /// attempting to divide by zero.
    fn divide_if_possible(&self, other: &Self) -> Result<Self>
    where
        Self: EuclideanDomain,
    {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let (quotient, remainder) = self.div_rem(other)?;

        if remainder.is_zero() {
            Ok(quotient)
        } else {
            Err(MathError::NotDivisible("Division not possible".to_string()))
        }
    }
}

// ============================================================================
// ParentMethods - Methods for ring structures
// ============================================================================

/// Whether a ring of characteristic `c` is provably not an integral domain.
///
/// `c = 1` is the zero ring (`1 = 0`). For composite `c = ab` with `1 < a, b < c`,
/// `(a·1)(b·1) = c·1 = 0` while `a·1` and `b·1` are nonzero (both are below the
/// characteristic). `false` for 0 and for primes. [`Integer::is_prime`] is the deterministic
/// word test on `u64` (strong tests to the 12 prime bases 2..37; no composite below
/// ψ12 ≈ 3.19·10^23 > 2^64 passes them, J. Sorenson and J. Webster, "Strong pseudoprimes to
/// twelve prime bases", Math. Comp. 86 (2017)), so both verdicts are proofs.
fn characteristic_rules_out_domain(c: u64) -> bool {
    c == 1 || (c >= 4 && !Integer::from(c).is_prime())
}

/// `Some(c)` when the parent's own data prove `R = Z·1 ≅ Z/cZ`, else `None`.
///
/// * A known characteristic `c` equal to a known order: the additive subgroup `Z·1` already
///   has `c` elements, so it is all of `R`.
/// * Characteristic unknown and a known prime order `p`: `1 != 0`, and its additive order
///   divides `p` (Lagrange), so it is `p` and again `Z·1 = R`.
///
/// Such a ring is commutative, and it is an integral domain (indeed the prime field `F_c`)
/// iff `c` is prime.
fn residue_ring_modulus<P: RingParentMethods + ?Sized>(parent: &P) -> Option<u64> {
    let n = u64::try_from(parent.order()?).ok()?;
    if n == 0 {
        return None;
    }
    match parent.characteristic() {
        Some(c) => (c == n).then_some(c),
        None => Integer::from(n).is_prime().then_some(n),
    }
}

/// Methods for ring structures (parents): SageMath `Rings.ParentMethods`; MAGMA Handbook
/// Ch. 17 §17.4 (generic ring functions).
///
/// # Honest defaults
///
/// The predicates are three-valued like MAGMA's (Ch. 17 §17.4.3: "true if known; false if
/// known not; error if unknown"): `Some(_)` is a proof, `None` makes no claim. A default
/// derives only what the ring axioms and this parent's own [`Self::characteristic`] and
/// [`Self::order`] prove. It never guesses or searches, and it never reads the static
/// `Self::Element::zero()`/`one()`: they panic for some parent-carrying element types (the
/// p-adic types), and for the sentinel types (`IntegerMod`, `FiniteFieldElement`,
/// `GaloisRingElement`) they are unbound placeholders that are the same for every parent, so
/// they say nothing about this parent. A parent that knows more overrides.
///
/// The defaults assume the parent is a unital subring of the values of `Self::Element`, with
/// that type's `0`, `1` and operations. A parent whose elements are representatives with
/// other arithmetic (for example `Integer` residues standing for Z/nZ) must override
/// `characteristic`.
///
/// Default dependency order (an override may call only methods earlier in it): `order` →
/// `characteristic` → `is_zero_ring`, `is_finite` → `is_commutative`, `is_integral_domain` →
/// `is_field` → `is_prime_field`, `is_integrally_closed`, `is_noetherian`. Because
/// `is_integral_domain` comes before `is_field`, a parent that overrides `is_field` with
/// `Some(true)` should also override `is_integral_domain` (a field is a domain).
///
/// Reference: https://doc.sagemath.org/html/en/reference/categories/sage/categories/rings.html#sage.categories.rings.Rings.ParentMethods
pub trait RingParentMethods {
    /// The type of elements in this ring
    type Element: Ring;

    /// Returns true (this is always a ring)
    ///
    /// This method exists for consistency with SageMath's category system.
    fn is_ring(&self) -> bool {
        true
    }

    /// MAGMA `IsCommutative(R)` (Ch. 17 §17.4.3): whether `a * b = b * a` for all elements.
    ///
    /// Default: `Some(true)` for the zero ring and for a parent proven to be `Z/cZ` (a known
    /// characteristic equal to a known order, or a known prime order); otherwise `None`.
    fn is_commutative(&self) -> Option<bool> {
        if self.is_zero_ring() == Some(true) || residue_ring_modulus(self).is_some() {
            Some(true)
        } else {
            None
        }
    }

    /// Whether this is the zero ring (`1 = 0`), decided from the parent (MAGMA Ch. 17
    /// §17.4.2: characteristic 1, equivalently `#R = 1`).
    ///
    /// A known [`Self::characteristic`] `c` gives `Some(c == 1)`; otherwise a known
    /// [`Self::order`] `n` gives `Some(n == 1)`; otherwise `None`. (An implementor whose
    /// characteristic and order contradict each other has a bug; the characteristic is read
    /// first.) It never compares the static `Self::Element::zero()` and `one()`: they panic
    /// for the p-adic element types, and for the sentinel types (`IntegerMod`,
    /// `FiniteFieldElement`, `GaloisRingElement`) they are parent-independent unbound
    /// placeholders, so `zero() == one()` says nothing about this parent.
    fn is_zero_ring(&self) -> Option<bool> {
        match (self.characteristic(), self.order()) {
            (Some(c), _) => Some(c == 1),
            (None, Some(n)) => Some(n == 1),
            (None, None) => None,
        }
    }

    /// MAGMA `IsField(R)` (Ch. 17 §17.4.3): every nonzero element is invertible.
    ///
    /// Default: a proven non-domain is not a field; a finite integral domain is a field
    /// (`x ↦ ax` is injective for `a != 0`, hence onto); otherwise `None`.
    fn is_field(&self) -> Option<bool> {
        match self.is_integral_domain() {
            Some(false) => Some(false),
            Some(true) if self.is_finite() == Some(true) => Some(true),
            _ => None,
        }
    }

    /// MAGMA `IsDomain(R)` (Ch. 17 §17.4.3): commutative, `0 != 1`, no zero divisors.
    ///
    /// Default:
    /// * the zero ring is not one;
    /// * a composite characteristic `c = ab` is not one (`(a·1)(b·1) = 0`, see the module
    ///   helper `characteristic_rules_out_domain`);
    /// * a parent proven to be `Z/cZ` (see [`Self::is_commutative`]) is one iff `c` is prime;
    /// * otherwise `None` (characteristic 0 or a prime does not decide it).
    fn is_integral_domain(&self) -> Option<bool> {
        if self.is_zero_ring() == Some(true) {
            return Some(false);
        }
        if let Some(c) = self.characteristic() {
            if characteristic_rules_out_domain(c) {
                return Some(false);
            }
        }
        residue_ring_modulus(self).map(|c| Integer::from(c).is_prime())
    }

    /// Whether this ring is integrally closed in its field of fractions.
    ///
    /// Default: a field is (it is its own fraction field), so `Some(true)` when
    /// [`Self::is_field`] is `Some(true)`; otherwise `None`. The notion is defined for
    /// domains, so no default answers `Some(false)`.
    fn is_integrally_closed(&self) -> Option<bool> {
        if self.is_field() == Some(true) {
            Some(true)
        } else {
            None
        }
    }

    /// Whether every ascending chain of ideals stabilises.
    ///
    /// Default: a finite ring (finitely many ideals) and a field (ideals `0` and `R`) are
    /// Noetherian (Atiyah–Macdonald, *Introduction to Commutative Algebra*, Ch. 6);
    /// otherwise `None`.
    fn is_noetherian(&self) -> Option<bool> {
        if self.is_finite() == Some(true) || self.is_field() == Some(true) {
            Some(true)
        } else {
            None
        }
    }

    /// MAGMA `IsPrimeField(F)` (Ch. 21 §21.2.1; `PrimeField`, Ch. 17 §17.4.1): a field that
    /// is Q or F_p.
    ///
    /// Default: a proven non-field is not one; a field of known order `n = p^k` is its prime
    /// field iff `k = 1` iff `n` is prime; otherwise `None` (a characteristic-0 field may or
    /// may not be Q).
    fn is_prime_field(&self) -> Option<bool> {
        match self.is_field() {
            Some(false) => Some(false),
            Some(true) => self
                .order()
                .and_then(|n| u64::try_from(n).ok())
                .map(|n| Integer::from(n).is_prime()),
            None => None,
        }
    }

    /// MAGMA `Characteristic(R)` (Ch. 17 §17.4.2): the least `m >= 1` with `m·1 = 0`, or 0.
    ///
    /// `Some(c)` is exact (`Some(1)`: the zero ring); `None` means unknown or `>= 2^64`. The
    /// exact `Integer` value of a large finite ring comes from its own parent type
    /// (`FiniteField::characteristic`, `GaloisRing::characteristic`, `Zmod::characteristic`
    /// in `rustmath-finitefields`).
    ///
    /// Default: [`Ring::static_characteristic`] of the element type (`Some(0)` for the Z, Q,
    /// R and C types); else `Some(1)` when [`Self::order`] is 1; else `None`.
    fn characteristic(&self) -> Option<u64> {
        Self::Element::static_characteristic().or_else(|| (self.order() == Some(1)).then_some(1))
    }

    /// Compute the Lie bracket [x, y] = xy - yx
    ///
    /// The Lie bracket measures the non-commutativity of ring multiplication.
    ///
    /// # Arguments
    ///
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// # Returns
    ///
    /// The element `x * y - y * x`
    fn bracket(&self, x: &Self::Element, y: &Self::Element) -> Self::Element {
        x.clone() * y.clone() - y.clone() * x.clone()
    }

    /// The order (number of elements) of this ring, when it is known and fits in `usize`.
    ///
    /// `None` makes **no claim**: the ring may be infinite, finite with more than
    /// `usize::MAX` elements, or of uncomputed size. Use [`Self::is_finite`]. An override
    /// must not call any other method of this trait (it is first in the default dependency
    /// order).
    fn order(&self) -> Option<usize> {
        None
    }

    /// MAGMA `IsFinite` (Handbook Ch. 17 §17.4.3): `Some(true)` known finite, `Some(false)`
    /// known infinite, `None` undecided.
    ///
    /// Default: a known `order()` proves finiteness; characteristic 0 proves infiniteness (a
    /// ring of characteristic 0 is nonzero and contains a copy of Z); characteristic 1 is the
    /// zero ring, which is finite; otherwise `None` (a positive characteristic does not
    /// decide it: F_p and F_p[x] share it).
    fn is_finite(&self) -> Option<bool> {
        if self.order().is_some() {
            return Some(true);
        }
        match self.characteristic() {
            Some(0) => Some(false),
            Some(1) => Some(true),
            _ => None,
        }
    }

    /// The negation of [`Self::is_finite`]; `None` when that is undecided.
    fn is_infinite(&self) -> Option<bool> {
        self.is_finite().map(|finite| !finite)
    }

    /// Return the Krull dimension of this ring
    ///
    /// The Krull dimension is the supremum of lengths of chains of prime ideals.
    ///
    /// # Default Implementation
    ///
    /// `None`: unknown. (The zero ring, of dimension −∞ by convention, also gives `None`.)
    /// Override for specific ring types.
    fn krull_dimension(&self) -> Option<usize> {
        None
    }

    /// A random element of this ring (MAGMA `Random(R)`, Ch. 17 §17.5.2).
    ///
    /// # Errors
    ///
    /// Default: `MathError::NotImplemented`. A parent that can sample overrides this; the
    /// default never returns a fixed element as if it were random.
    fn random_element(&self) -> Result<Self::Element> {
        Err(MathError::NotImplemented(
            "random_element: this parent does not implement Random (MAGMA Ch. 17 §17.5.2)".into(),
        ))
    }

    /// A random nonzero element of this ring.
    ///
    /// # Errors
    ///
    /// Default: `MathError::InvalidArgument` for the zero ring (it has no nonzero element),
    /// otherwise `MathError::NotImplemented` as for [`Self::random_element`].
    fn random_nonzero_element(&self) -> Result<Self::Element> {
        if self.is_zero_ring() == Some(true) {
            return Err(MathError::InvalidArgument(
                "random_nonzero_element: the zero ring has no nonzero element".into(),
            ));
        }
        Err(MathError::NotImplemented(
            "random_nonzero_element: this parent does not implement Random (MAGMA Ch. 17 §17.5.2)"
                .into(),
        ))
    }
}

// ============================================================================
// MorphismMethods - Methods for ring homomorphisms
// ============================================================================

/// Methods for ring homomorphisms, corresponding to SageMath's `Rings.MorphismMethods`
///
/// This trait provides operations on ring homomorphisms including:
/// - Testing injectivity
/// - Testing if morphism is nonzero
/// - Extending to fraction fields
///
/// The predicates are three-valued: `Some(_)` is a proof, `None` makes no claim (MAGMA
/// `IsInjective`, `IsSurjective`, Ch. 54 §54.10.7; `Kernel`, Ch. 16 §16.3.2).
///
/// Reference: https://doc.sagemath.org/html/en/reference/categories/sage/categories/rings.html#sage.categories.rings.Rings.MorphismMethods
pub trait RingMorphismMethods {
    /// The domain ring type
    type Domain: Ring;

    /// The codomain ring type
    type Codomain: Ring;

    /// Apply the morphism to an element of the domain
    fn apply(&self, x: &Self::Domain) -> Self::Codomain;

    /// MAGMA `IsInjective` (Ch. 54 §54.10.7; Ch. 16 §16.3.2 `Kernel`): `Some(true)` proven
    /// injective (kernel 0), `Some(false)` proven not, `None` undecided.
    ///
    /// Default `None`: this trait sees only the element map, not the kernel.
    fn is_injective(&self) -> Option<bool> {
        None
    }

    /// Whether this morphism is nonzero.
    ///
    /// A ring homomorphism φ is zero iff φ(1) = 0 (φ(x) = φ(x)·φ(1)), so the default
    /// checks the image of 1; the answer is a proof if `apply` is a ring homomorphism.
    ///
    /// # Panics
    ///
    /// Reads the static `Self::Domain::one()`, which panics for the p-adic element types
    /// (`PadicInteger`, `PadicRational`, `PadicExtensionElement`); morphisms from those must
    /// override this method. For the sentinel domain types (`IntegerMod`,
    /// `FiniteFieldElement`, `GaloisRingElement`), `apply` receives the parent-independent
    /// unbound 1, and the answer is right only if `apply` maps it as the domain's 1.
    fn is_nonzero(&self) -> bool {
        !self.apply(&Self::Domain::one()).is_zero()
    }

    /// MAGMA `IsSurjective` (Ch. 54 §54.10.7): `Some(true)` proven onto, `Some(false)` proven
    /// not, `None` undecided. Default `None`.
    fn is_surjective(&self) -> Option<bool> {
        None
    }

    /// Whether this morphism is bijective (a bijective ring homomorphism is an isomorphism):
    /// the Kleene "and" of [`Self::is_injective`] and [`Self::is_surjective`]. `Some(false)`
    /// if either is `Some(false)`, `Some(true)` if both are `Some(true)`, else `None`.
    fn is_isomorphism(&self) -> Option<bool> {
        match (self.is_injective(), self.is_surjective()) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        }
    }
}

// ============================================================================
// SubcategoryMethods - Methods for ring subcategories
// ============================================================================

/// Marker trait for rings with no zero divisors (domains)
///
/// A ring has no zero divisors if `a * b = 0` implies `a = 0` or `b = 0`.
/// This corresponds to SageMath's `Rings.NoZeroDivisors()` subcategory.
pub trait NoZeroDivisors: Ring {}

/// Marker trait for rings where all nonzero elements are invertible (division rings/fields)
///
/// This corresponds to SageMath's `Rings.Division()` subcategory.
pub trait Division: Ring {}

/// Marker trait for commutative rings
///
/// A ring is commutative if `a * b = b * a` for all elements.
pub trait Commutative: Ring {}

// ============================================================================
// Automatic implementations
// ============================================================================

// Implement RingElementMethods for all Ring types
impl<R: Ring> RingElementMethods for R {}

// Implement marker traits for standard integer types
impl NoZeroDivisors for i32 {}
impl NoZeroDivisors for i64 {}
impl Commutative for i32 {}
impl Commutative for i64 {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_element_is_unit() {
        // In integers, only ±1 are units
        assert_eq!(1i32.is_unit_element(), Some(true));
        assert_eq!((-1i32).is_unit_element(), Some(true));
        assert_eq!(0i32.is_unit_element(), Some(false));
        assert_eq!(2i32.is_unit_element(), Some(false));
        assert_eq!((-2i32).is_unit_element(), Some(false));
    }

    // The defect's generic path for the concrete rings (MAGMA Ch. 20 / Ch. 25 IsUnit: every
    // nonzero element of Q, R is a unit). GAP: IsUnit(Rationals, 2/3) -> true,
    // Inverse(2/3) -> 3/2. PARI: lift(Mod(3,8)^-1) -> 3; lift(Mod(2,5)^-1) -> 3.
    #[test]
    fn test_ring_element_methods_on_concrete_rings() {
        use rustmath_finitefields::{FiniteField, IntegerMod};
        use rustmath_integers::Integer;
        use rustmath_rationals::Rational;
        use rustmath_reals::RealMPFR;

        let two_thirds = Rational::new(2, 3).unwrap();
        assert_eq!(two_thirds.is_unit_element(), Some(true));
        assert_eq!(two_thirds.inverse_of_unit(), Ok(Rational::new(3, 2).unwrap()));
        assert_eq!(Rational::new(0, 1).unwrap().inverse_of_unit(), Err(MathError::NotInvertible));

        assert_eq!(RealMPFR::with_val(53, 2.0).is_unit_element(), Some(true));

        let three = IntegerMod::new(Integer::from(3), Integer::from(8)).unwrap();
        assert_eq!(RingElementMethods::is_unit_element(&three), Some(true));
        assert_eq!(three.inverse_of_unit(), Ok(three.clone()));

        let gf5 = FiniteField::new(Integer::from(5), 1).unwrap();
        let two = gf5.from_int(Integer::from(2));
        assert_eq!(two.inverse_of_unit(), Ok(gf5.from_int(Integer::from(3))));
    }

    // Defaults on a parent-carrying element type whose static `one()` panics: nonzero,
    // non-±1 p-adic integers must not panic. 6 in Z_5 is a unit and not nilpotent, but the
    // p-adic types do not override the hooks (yet), so the honest answer is "undecided".
    // (`rustmath_category::RingElementMethods::is_nilpotent` is exactly
    // `Ring::nilpotent_status`.)
    #[test]
    fn test_defaults_on_padic_integer_do_not_panic() {
        use crate::padics::PadicInteger;
        use rustmath_integers::Integer;
        let six = PadicInteger::from_integer(Integer::from(6), Integer::from(5), 10).unwrap();
        let nil = Ring::nilpotent_status(&six);
        assert_ne!(nil, Some(true));
        assert_eq!(nil, None);
        assert_eq!(six.is_unit_element(), None);
        assert!(!matches!(six.inverse_of_unit(), Err(MathError::NotInvertible)));
    }

    #[test]
    fn test_ring_element_inverse_of_unit() {
        // In integers, only ±1 have inverses
        assert_eq!(1i32.inverse_of_unit().unwrap(), 1);
        assert_eq!((-1i32).inverse_of_unit().unwrap(), -1);
        assert!(0i32.inverse_of_unit().is_err());
        assert!(2i32.inverse_of_unit().is_err());
    }

    #[test]
    fn test_ring_element_divide_if_possible() {
        // Exact divisions
        assert_eq!(6i32.divide_if_possible(&2).unwrap(), 3);
        assert_eq!(6i32.divide_if_possible(&3).unwrap(), 2);
        assert_eq!(6i32.divide_if_possible(&6).unwrap(), 1);

        // Non-exact divisions
        assert!(7i32.divide_if_possible(&2).is_err());
        assert!(5i32.divide_if_possible(&3).is_err());

        // Division by zero
        assert!(6i32.divide_if_possible(&0).is_err());
    }

    #[test]
    fn test_ring_element_divide_negative() {
        assert_eq!((-6i32).divide_if_possible(&2).unwrap(), -3);
        assert_eq!(6i32.divide_if_possible(&(-2)).unwrap(), -3);
        assert_eq!((-6i32).divide_if_possible(&(-2)).unwrap(), 3);
    }

    // Example: Simple ring parent for testing
    #[derive(Debug, Clone)]
    struct IntegerRing;

    // Z: `characteristic` comes from the default (`i32::static_characteristic() = Some(0)`).
    impl RingParentMethods for IntegerRing {
        type Element = i32;

        fn is_commutative(&self) -> Option<bool> {
            Some(true)
        }

        fn is_integral_domain(&self) -> Option<bool> {
            Some(true)
        }

        // GAP: IsField(Integers) -> false.
        fn is_field(&self) -> Option<bool> {
            Some(false)
        }
    }

    // GAP: [Characteristic, IsField, IsIntegralRing, IsCommutative, IsFinite](Integers)
    // -> [0, false, true, true, false].
    #[test]
    fn test_ring_parent_properties() {
        let z = IntegerRing;
        assert!(z.is_ring());
        assert_eq!(z.is_commutative(), Some(true));
        assert_eq!(z.is_integral_domain(), Some(true));
        assert_eq!(z.is_zero_ring(), Some(false));
        assert_eq!(z.characteristic(), Some(0));
        assert_eq!(z.is_infinite(), Some(true));
        assert_eq!(z.is_finite(), Some(false));
        assert_eq!(z.is_field(), Some(false));
        assert_eq!(z.is_prime_field(), Some(false));
    }

    #[test]
    fn test_ring_parent_bracket() {
        let z = IntegerRing;

        // For commutative rings, bracket is always zero
        assert_eq!(z.bracket(&2, &3), 0);
        assert_eq!(z.bracket(&5, &7), 0);
        assert_eq!(z.bracket(&1, &1), 0);
    }

    // ---- Test parents for the honest defaults (MAGMA Ch. 17 §17.4) ----

    use rustmath_core::NumericConversion;
    use rustmath_finitefields::{
        FiniteField, FiniteFieldElement, GaloisRing, GaloisRingElement, IntegerMod, Zmod,
    };
    use rustmath_rationals::Rational;

    /// A parent that overrides nothing: what every new implementor inherits.
    struct Bare<E>(std::marker::PhantomData<E>);
    impl<E: Ring> RingParentMethods for Bare<E> {
        type Element = E;
    }
    fn bare<E: Ring>() -> Bare<E> {
        Bare(std::marker::PhantomData)
    }

    /// GR(p^a, d) with its exact characteristic and order; nothing else overridden.
    struct GrParent(GaloisRing);
    impl RingParentMethods for GrParent {
        type Element = GaloisRingElement;
        fn characteristic(&self) -> Option<u64> {
            GaloisRing::characteristic(&self.0).to_u64()
        }
        fn order(&self) -> Option<usize> {
            self.0.cardinality().to_usize()
        }
        // Galois rings are commutative (MAGMA Ch. 48).
        fn is_commutative(&self) -> Option<bool> {
            Some(true)
        }
    }

    /// Z/nZ with its exact characteristic and order only.
    struct ZmodParent(Zmod);
    impl RingParentMethods for ZmodParent {
        type Element = IntegerMod;
        fn characteristic(&self) -> Option<u64> {
            self.0.characteristic().to_u64()
        }
        fn order(&self) -> Option<usize> {
            self.0.order().to_usize()
        }
    }
    fn zmod(n: u64) -> ZmodParent {
        ZmodParent(Zmod::new(Integer::from(n)).unwrap())
    }

    /// GF(q) with its exact characteristic and order, and `is_field = Some(true)` only.
    struct FqParent(FiniteField);
    impl RingParentMethods for FqParent {
        type Element = FiniteFieldElement;
        fn characteristic(&self) -> Option<u64> {
            self.0.characteristic().to_u64()
        }
        fn order(&self) -> Option<usize> {
            self.0.order().to_usize()
        }
        fn is_field(&self) -> Option<bool> {
            Some(true)
        }
    }
    fn gf(p: u64, n: usize) -> FqParent {
        FqParent(FiniteField::new(Integer::from(p), n).unwrap())
    }

    /// GF(q) known only to be an integral domain (plus characteristic and order).
    struct FqDomain(FiniteField);
    impl RingParentMethods for FqDomain {
        type Element = FiniteFieldElement;
        fn characteristic(&self) -> Option<u64> {
            self.0.characteristic().to_u64()
        }
        fn order(&self) -> Option<usize> {
            self.0.order().to_usize()
        }
        fn is_integral_domain(&self) -> Option<bool> {
            Some(true)
        }
    }

    /// The zero ring, known from its characteristic only.
    struct ZeroRing;
    impl RingParentMethods for ZeroRing {
        type Element = IntegerMod;
        fn characteristic(&self) -> Option<u64> {
            Some(1)
        }
    }

    /// The zero ring, known from its order only.
    struct ZeroRingByOrder;
    impl RingParentMethods for ZeroRingByOrder {
        type Element = IntegerMod;
        fn order(&self) -> Option<usize> {
            Some(1)
        }
    }

    /// Q, known only to be a field.
    struct QField;
    impl RingParentMethods for QField {
        type Element = Rational;
        fn is_field(&self) -> Option<bool> {
            Some(true)
        }
    }

    // GAP: [Characteristic, IsField, IsPrimeField, IsFinite, IsCommutative](Rationals)
    // -> [0, true, true, false, true]. The default knows only the characteristic (from the
    // element type), so it decides char 0, infinite and nonzero, and claims nothing else.
    // At a46a6a8 the defaults answered `false` for every predicate (a fabricated "not a
    // field") and `random_element` returned the constant 0.
    #[test]
    fn test_parent_defaults_do_not_guess() {
        let q = bare::<Rational>();
        assert_eq!(q.characteristic(), Some(0));
        assert_eq!(q.is_zero_ring(), Some(false));
        assert_eq!(q.is_finite(), Some(false));
        assert_eq!(q.is_infinite(), Some(true));
        assert_eq!(q.is_commutative(), None);
        assert_eq!(q.is_integral_domain(), None);
        assert_eq!(q.is_field(), None);
        assert_eq!(q.is_prime_field(), None);
        assert_eq!(q.is_noetherian(), None);
        assert_eq!(q.is_integrally_closed(), None);
        assert_eq!(q.krull_dimension(), None);
        assert!(matches!(
            q.random_element(),
            Err(MathError::NotImplemented(_))
        ));
        assert!(matches!(
            q.random_nonzero_element(),
            Err(MathError::NotImplemented(_))
        ));
    }

    // Q given as a field: a field is integrally closed (its own fraction field) and
    // Noetherian (ideals 0 and Q). Prime-field status is not derivable from char 0 (GAP
    // IsPrimeField(Rationals) -> true, but CF(3) has char 0 too), and `is_integral_domain`
    // precedes `is_field` in the dependency order, so it stays `None` unless overridden.
    #[test]
    fn test_field_parent_derivations() {
        let q = QField;
        assert_eq!(q.is_integrally_closed(), Some(true));
        assert_eq!(q.is_noetherian(), Some(true));
        assert_eq!(q.is_prime_field(), None);
        assert_eq!(q.is_integral_domain(), None);
    }

    // Item (b). `PadicInteger::zero()/one()` panic ("Cannot create PadicInteger::zero()
    // without parameters"); the defaults must not call them. `GaloisRingElement::zero()`
    // panicked at a46a6a8 and is an unbound sentinel since 029f7ab; the default must give
    // `None` either way (neither type has a static characteristic).
    #[test]
    fn test_parent_carrying_elements_do_not_panic() {
        use crate::padics::PadicInteger;

        let zp = bare::<PadicInteger>();
        assert_eq!(zp.characteristic(), None);
        assert_eq!(zp.is_zero_ring(), None);
        assert_eq!(zp.is_field(), None);
        assert_eq!(zp.is_finite(), None);
        assert!(matches!(
            zp.random_element(),
            Err(MathError::NotImplemented(_))
        ));
        assert!(matches!(
            zp.random_nonzero_element(),
            Err(MathError::NotImplemented(_))
        ));

        let gr = bare::<GaloisRingElement>();
        assert_eq!(gr.characteristic(), None);
        assert_eq!(gr.is_zero_ring(), None);
        assert_eq!(gr.is_field(), None);
        assert_eq!(gr.is_finite(), None);
        assert!(matches!(
            gr.random_element(),
            Err(MathError::NotImplemented(_))
        ));
        assert!(matches!(
            gr.random_nonzero_element(),
            Err(MathError::NotImplemented(_))
        ));
    }

    // Item (b), second half: `IntegerMod::zero()` and `one()` are unbound sentinels, the
    // same for every parent, so `zero() == one()` (false) said "not the zero ring" even for
    // a parent of characteristic 1. GAP: ZmodnZ(1): Zero = One -> true;
    // PARI: Mod(1,1) == Mod(0,1) -> 1.
    #[test]
    fn test_is_zero_ring_is_decided_from_the_parent() {
        assert_eq!(bare::<IntegerMod>().is_zero_ring(), None);
        assert_eq!(ZeroRing.is_zero_ring(), Some(true));
        assert_eq!(ZeroRingByOrder.is_zero_ring(), Some(true));
        assert_eq!(ZeroRingByOrder.characteristic(), Some(1));
        assert_eq!(zmod(6).is_zero_ring(), Some(false));
    }

    // GR(2^3, 2) (MAGMA H48E1). PARI: char 8 (8*1 = 0 mod 8, least such), 2^3 = Mod(0, 8)
    // (2 is nilpotent, so not a domain), order 8^2 = 64.
    #[test]
    fn test_galois_ring_parent() {
        let gr = GrParent(GaloisRing::new(Integer::from(2), 3, 2).unwrap());
        assert_eq!(gr.characteristic(), Some(8));
        assert_eq!(gr.order(), Some(64));
        assert_eq!(gr.is_zero_ring(), Some(false));
        assert_eq!(gr.is_integral_domain(), Some(false));
        assert_eq!(gr.is_field(), Some(false));
        assert_eq!(gr.is_prime_field(), Some(false));
        assert_eq!(gr.is_finite(), Some(true));
        assert_eq!(gr.is_noetherian(), Some(true));
        assert_eq!(gr.is_commutative(), Some(true));
        assert_eq!(gr.is_integrally_closed(), None);
    }

    // GAP: [Characteristic, Size, IsField, IsCommutative](ZmodnZ(n)):
    //   n = 2: [2, 2, true, true]   n = 3: [3, 3, true, true]   n = 4: [4, 4, false, true]
    //   n = 6: [6, 6, false, true]  n = 7: [7, 7, true, true]   n = 8: [8, 8, false, true].
    // PARI: Mod(2,6)*Mod(3,6) = Mod(0,6), Mod(2,4)^2 = Mod(0,4), Mod(2,8)*Mod(4,8) = Mod(0,8)
    // (zero divisors); isprime([2,3,4,6,7,8]) = [1,1,0,0,1,0]. (GAP's IsIntegralRing is
    // true for ZmodnZ(6) and ZmodnZ(8), a GAP quirk, so it is not used as a reference.)
    // Characteristic = order, so each parent is Z·1 = Z/nZ: commutative, and a domain (the
    // prime field F_n) iff n is prime.
    #[test]
    fn test_zmod_parents() {
        let z6 = zmod(6);
        assert_eq!(
            (
                z6.characteristic(),
                z6.is_zero_ring(),
                z6.is_commutative(),
                z6.is_integral_domain(),
                z6.is_field(),
                z6.is_prime_field(),
                z6.is_finite(),
                z6.is_noetherian()
            ),
            (
                Some(6),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(true)
            )
        );
        assert_eq!(zmod(4).is_integral_domain(), Some(false));
        assert_eq!(zmod(8).is_field(), Some(false));
        for p in [2u64, 3] {
            assert_eq!(zmod(p).is_integral_domain(), Some(true), "Z/{p}");
            assert_eq!(zmod(p).is_field(), Some(true), "Z/{p}");
        }
        let z7 = zmod(7);
        assert_eq!(
            (
                z7.characteristic(),
                z7.is_commutative(),
                z7.is_integral_domain(),
                z7.is_field(),
                z7.is_prime_field(),
                z7.is_integrally_closed()
            ),
            (
                Some(7),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true)
            )
        );
    }

    // A known prime order alone proves Z/p (1 != 0 has additive order dividing p).
    // GAP: IsField(ZmodnZ(7)) -> true, IsCommutative(ZmodnZ(7)) -> true.
    #[test]
    fn test_prime_order_alone_decides() {
        struct OrderSeven;
        impl RingParentMethods for OrderSeven {
            type Element = IntegerMod;
            fn order(&self) -> Option<usize> {
                Some(7)
            }
        }
        assert_eq!(OrderSeven.characteristic(), None);
        assert_eq!(OrderSeven.is_commutative(), Some(true));
        assert_eq!(OrderSeven.is_field(), Some(true));
        assert_eq!(OrderSeven.is_prime_field(), Some(true));

        // A composite order with unknown characteristic decides nothing (Z/4 and F_4 share
        // order 4; GAP IsField(ZmodnZ(4)) -> false, IsField(GF(4)) -> true).
        struct OrderFour;
        impl RingParentMethods for OrderFour {
            type Element = IntegerMod;
            fn order(&self) -> Option<usize> {
                Some(4)
            }
        }
        assert_eq!(OrderFour.is_commutative(), None);
        assert_eq!(OrderFour.is_integral_domain(), None);
        assert_eq!(OrderFour.is_field(), None);
    }

    // A finite integral domain is a field (x -> ax is injective, hence onto).
    // GAP: [Characteristic, Size, IsField, IsPrimeField](GF(25)) -> [5, 25, true, false].
    // Z known only as a domain stays undecided as a field (it is infinite).
    #[test]
    fn test_finite_domain_is_field() {
        let f25 = FqDomain(FiniteField::new(Integer::from(5), 2).unwrap());
        assert_eq!(
            (
                f25.is_field(),
                f25.is_prime_field(),
                f25.is_integrally_closed(),
                f25.is_noetherian()
            ),
            (Some(true), Some(false), Some(true), Some(true))
        );

        struct ZDomainOnly;
        impl RingParentMethods for ZDomainOnly {
            type Element = i32;
            fn is_integral_domain(&self) -> Option<bool> {
                Some(true)
            }
        }
        assert_eq!(ZDomainOnly.is_field(), None);
    }

    // GAP: [Characteristic, Size, IsField, IsPrimeField](GF(q)):
    //   q = 25: [5, 25, true, false]   q = 5: [5, 5, true, true]   q = 7: [7, 7, true, true].
    #[test]
    fn test_finite_field_parents_prime_field() {
        let f25 = gf(5, 2);
        assert_eq!(f25.characteristic(), Some(5));
        assert_eq!(f25.order(), Some(25));
        assert_eq!(f25.is_prime_field(), Some(false));
        assert_eq!(f25.is_integrally_closed(), Some(true));
        assert_eq!(f25.is_noetherian(), Some(true));
        assert_eq!(gf(5, 1).is_prime_field(), Some(true));
        assert_eq!(gf(7, 1).is_prime_field(), Some(true));
    }

    // GAP: [Size, Characteristic, IsField, IsCommutative, Zero = One](ZmodnZ(1))
    // -> [1, 1, false, true, true]; PARI: Mod(1,1) == Mod(0,1) -> 1.
    #[test]
    fn test_zero_ring_parent() {
        for (name, values, nonzero) in [
            (
                "char 1",
                (
                    ZeroRing.is_zero_ring(),
                    ZeroRing.is_commutative(),
                    ZeroRing.is_integral_domain(),
                    ZeroRing.is_field(),
                    ZeroRing.is_prime_field(),
                    ZeroRing.is_finite(),
                    ZeroRing.is_noetherian(),
                ),
                ZeroRing.random_nonzero_element(),
            ),
            (
                "order 1",
                (
                    ZeroRingByOrder.is_zero_ring(),
                    ZeroRingByOrder.is_commutative(),
                    ZeroRingByOrder.is_integral_domain(),
                    ZeroRingByOrder.is_field(),
                    ZeroRingByOrder.is_prime_field(),
                    ZeroRingByOrder.is_finite(),
                    ZeroRingByOrder.is_noetherian(),
                ),
                ZeroRingByOrder.random_nonzero_element(),
            ),
        ] {
            assert_eq!(
                values,
                (
                    Some(true),
                    Some(true),
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(true),
                    Some(true)
                ),
                "{name}"
            );
            assert!(
                matches!(nonzero, Err(MathError::InvalidArgument(_))),
                "{name}"
            );
        }
    }

    // A characteristic that does not fit u64 makes no claim. PARI: isprime(2^64+13) -> 1,
    // isprime(2^64-59) -> 1, factor(2^64-61) = 3^2 * 5 * 97 * 197 * 325957 * 65812583.
    #[test]
    fn test_large_characteristic_makes_no_claim() {
        let p = Integer::from(2u64).pow(64) + Integer::from(13);
        let big = FqParent(FiniteField::new(p, 1).unwrap());
        assert_eq!(big.characteristic(), None);
        assert_eq!(big.order(), None);
        assert_eq!(big.is_zero_ring(), None);
        assert_eq!(big.is_prime_field(), None);
        assert_eq!(big.is_finite(), None);

        assert!(!characteristic_rules_out_domain(18_446_744_073_709_551_557)); // 2^64 - 59
        assert!(characteristic_rules_out_domain(18_446_744_073_709_551_555)); // 2^64 - 61

        // Boundary values: 0 and primes decide nothing; 1 and composites do.
        // PARI: isprime([2,3,4,6,7,8,25,561]) = [1,1,0,0,1,0,0,0] (561 = 3*11*17, Carmichael).
        for (c, rules_out) in [
            (0, false),
            (1, true),
            (2, false),
            (3, false),
            (4, true),
            (6, true),
            (7, false),
            (8, true),
            (25, true),
            (561, true),
        ] {
            assert_eq!(characteristic_rules_out_domain(c), rules_out, "c = {c}");
        }
    }

    // Example: Simple morphism for testing. x -> 2x is additive but not multiplicative
    // (2·(xy) != (2x)(2y)) and does not send 1 to 1: it tests the trait plumbing, not a
    // ring homomorphism.
    struct DoubleMap;

    impl RingMorphismMethods for DoubleMap {
        type Domain = i32;
        type Codomain = i32;

        fn apply(&self, x: &i32) -> i32 {
            2 * x
        }

        fn is_injective(&self) -> Option<bool> {
            // 2x = 0 forces x = 0 in Z.
            Some(true)
        }
    }

    // PARI: 1 % 2 == 0 -> 0, so 1 is not in 2Z: the map is not onto, but the trait cannot
    // see that, so `is_surjective` stays undecided.
    #[test]
    fn test_ring_morphism_basic() {
        let f = DoubleMap;
        assert_eq!(f.apply(&3), 6);
        assert_eq!(f.apply(&0), 0);
        assert!(f.is_nonzero()); // f(1) = 2 ≠ 0
        assert_eq!(f.is_injective(), Some(true));
        assert_eq!(f.is_surjective(), None);
        assert_eq!(f.is_isomorphism(), None);
    }

    /// The identity of Z, with no overrides.
    struct IdZ;
    impl RingMorphismMethods for IdZ {
        type Domain = Integer;
        type Codomain = Integer;
        fn apply(&self, x: &Integer) -> Integer {
            x.clone()
        }
    }

    /// Reduction Z -> Z/5Z: onto, with 5 in the kernel.
    struct ReduceMod5;
    impl RingMorphismMethods for ReduceMod5 {
        type Domain = Integer;
        type Codomain = IntegerMod;
        fn apply(&self, x: &Integer) -> IntegerMod {
            IntegerMod::new(x.clone(), Integer::from(5)).unwrap()
        }
        fn is_injective(&self) -> Option<bool> {
            Some(false)
        }
        fn is_surjective(&self) -> Option<bool> {
            Some(true)
        }
    }

    // At a46a6a8 the identity of Z reported is_injective = is_surjective = is_isomorphism =
    // false. PARI: Mod(5,5) == 0 -> 1 (5 is in the kernel of Z -> Z/5).
    #[test]
    fn test_ring_morphism_defaults_and_kleene() {
        assert_eq!(
            (
                IdZ.is_injective(),
                IdZ.is_surjective(),
                IdZ.is_isomorphism()
            ),
            (None, None, None)
        );
        assert!(IdZ.is_nonzero());
        assert!(ReduceMod5.apply(&Integer::from(5)).is_zero());
        assert_eq!(ReduceMod5.is_isomorphism(), Some(false));

        struct Bijective;
        impl RingMorphismMethods for Bijective {
            type Domain = Integer;
            type Codomain = Integer;
            fn apply(&self, x: &Integer) -> Integer {
                x.clone()
            }
            fn is_injective(&self) -> Option<bool> {
                Some(true)
            }
            fn is_surjective(&self) -> Option<bool> {
                Some(true)
            }
        }
        assert_eq!(Bijective.is_isomorphism(), Some(true));
    }
}

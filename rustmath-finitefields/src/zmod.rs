//! The ring Z/mZ as a parent object
//!
//! [`Zmod`] is the parent of [`IntegerMod`] elements: it carries the modulus
//! once, so `zero()`/`one()` are lawful (unlike the static
//! [`Ring::zero`]/[`Ring::one`] on the element type, which must fall back to
//! the modulus-0 sentinel documented on [`IntegerMod`]).
//!
//! The Sage-style constructor is the free function [`Integers`]:
//! `Integers(m)` = `Zmod::new(m)` = Z/mZ.
//!
//! # Naming
//!
//! The type is deliberately named `Zmod`, not `IntegerModRing`: the latter is
//! already a *trait* in `rustmath_rings::abc`, and reusing the name would
//! shadow it under glob imports.
//!
//! # Example
//!
//! ```
//! use rustmath_finitefields::{Integers, Zmod};
//! use rustmath_integers::Integer;
//! use rustmath_core::Ring; // for is_zero()/is_one()
//!
//! let z10 = Integers(Integer::from(10)).unwrap();
//! let a = z10.element(Integer::from(17)).unwrap();
//! assert_eq!(a.value(), &Integer::from(7)); // 17 ≡ 7 (mod 10)
//! assert!(z10.zero().is_zero());
//! assert!(z10.one().is_one());
//! assert!(!z10.is_field()); // 10 is not prime
//! ```

use crate::integer_mod::IntegerMod;
use rustmath_core::{EuclideanDomain, MathError, NumericConversion, Parent, Result};
use rustmath_integers::prime::{factor, is_prime};
use rustmath_integers::Integer;
use std::fmt;

/// The ring of integers modulo m (Z/mZ) as a parent object
///
/// Elements are [`IntegerMod`] values. Unlike the element-level statics
/// `Ring::zero()`/`Ring::one()`, the parent-level [`Zmod::zero`] and
/// [`Zmod::one`] carry the modulus and never involve the modulus-0 sentinel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Zmod {
    modulus: Integer,
}

impl Zmod {
    /// Create the ring Z/mZ.
    ///
    /// The modulus must be > 1 (matching [`IntegerMod::new`]); Z/0Z ≅ Z and
    /// the trivial ring Z/1Z are not representable by the element type.
    pub fn new(modulus: Integer) -> Result<Self> {
        if modulus <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Modulus must be > 1".to_string(),
            ));
        }
        Ok(Zmod { modulus })
    }

    /// The modulus m.
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// The order (cardinality) of the ring, which equals the modulus.
    pub fn order(&self) -> &Integer {
        &self.modulus
    }

    /// The characteristic of the ring, which equals the modulus.
    pub fn characteristic(&self) -> Integer {
        self.modulus.clone()
    }

    /// Whether Z/mZ is a field (true iff m is prime).
    pub fn is_field(&self) -> bool {
        is_prime(&self.modulus)
    }

    /// Construct the element `value mod m`.
    pub fn element(&self, value: Integer) -> Result<IntegerMod> {
        IntegerMod::new(value, self.modulus.clone())
    }

    /// The zero element of Z/mZ (parent-aware; carries the modulus).
    pub fn zero(&self) -> IntegerMod {
        IntegerMod::new(Integer::zero(), self.modulus.clone())
            .expect("modulus > 1 by construction")
    }

    /// The one element of Z/mZ (parent-aware; carries the modulus).
    pub fn one(&self) -> IntegerMod {
        IntegerMod::new(Integer::one(), self.modulus.clone())
            .expect("modulus > 1 by construction")
    }

    /// Canonically map an element into this ring.
    ///
    /// Accepts elements of this same ring (identity), the modulus-0 sentinel
    /// (the canonical map Z → Z/mZ), and elements of Z/nZ when m divides n
    /// (the canonical projection Z/nZ → Z/mZ). Anything else is an error —
    /// there is no canonical map.
    pub fn coerce(&self, x: &IntegerMod) -> Result<IntegerMod> {
        let n = x.modulus();
        if n == &self.modulus || n.is_zero() {
            self.element(x.value().clone())
        } else {
            let (_, rem) = n.div_rem(&self.modulus)?;
            if rem.is_zero() {
                self.element(x.value().clone())
            } else {
                Err(MathError::InvalidArgument(format!(
                    "No canonical map from Z/{}Z to Z/{}Z",
                    n, self.modulus
                )))
            }
        }
    }

    /// The factored modulus `m = prod p_i^{e_i}` (MAGMA `FactoredModulus`).
    pub fn factored_modulus(&self) -> Vec<(Integer, u32)> {
        factor(&self.modulus)
    }

    /// Solve the linear congruence `a*x = b (mod m)` (MAGMA `Solution`).
    ///
    /// Returns one solution `x`, or `None` if no solution exists (i.e. when
    /// `gcd(a, m)` does not divide `b`).
    pub fn solution(&self, a: &Integer, b: &Integer) -> Option<IntegerMod> {
        let m = &self.modulus;
        let (g, s, _) = a.extended_gcd(m);
        // Normalize the gcd sign (it can be negative for negative `a`).
        let (g, s) = if g.signum() < 0 { (-g, -s) } else { (g, s) };
        let (q, r) = b.div_rem(&g).ok()?;
        if !r.is_zero() {
            return None; // g does not divide b
        }
        // x0 = (b/g) * s  (mod m/g) is a solution mod m.
        let mg = m.clone() / g;
        let (_, x0) = (q * s).div_rem(&mg).ok()?;
        let x0 = if x0.signum() < 0 { x0 + mg } else { x0 };
        self.element(x0).ok()
    }
}

impl fmt::Display for Zmod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ring of integers modulo {}", self.modulus)
    }
}

impl Parent for Zmod {
    type Element = IntegerMod;

    fn contains(&self, element: &Self::Element) -> bool {
        element.modulus() == &self.modulus
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(Zmod::zero(self))
    }

    fn one(&self) -> Option<Self::Element> {
        Some(Zmod::one(self))
    }

    /// The cardinality, when it fits in `usize`. `None` here only means
    /// "not representable as usize" — Z/mZ is always finite, see
    /// [`Parent::is_finite`] below.
    fn cardinality(&self) -> Option<usize> {
        self.modulus.to_usize()
    }

    fn is_finite(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        format!("{}", self)
    }
}

/// Sage/MAGMA-style constructor for Z/mZ: `Integers(m)`.
///
/// This is the constructor named in Sage (`Integers(m)` = `IntegerModRing(m)`)
/// and MAGMA (`Integers(m)`); it simply forwards to [`Zmod::new`].
#[allow(non_snake_case)]
pub fn Integers(modulus: Integer) -> Result<Zmod> {
    Zmod::new(modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_core::Ring;

    #[test]
    fn test_zmod_construction() {
        let z10 = Zmod::new(Integer::from(10)).unwrap();
        assert_eq!(z10.modulus(), &Integer::from(10));
        assert_eq!(z10.order(), &Integer::from(10));
        assert_eq!(z10.characteristic(), Integer::from(10));
        assert!(!z10.is_field());

        let z7 = Integers(Integer::from(7)).unwrap();
        assert!(z7.is_field());

        assert!(Zmod::new(Integer::from(1)).is_err());
        assert!(Zmod::new(Integer::from(0)).is_err());
        assert!(Zmod::new(Integer::from(-5)).is_err());
    }

    #[test]
    fn test_parent_aware_zero_one() {
        let z7 = Zmod::new(Integer::from(7)).unwrap();
        let zero = z7.zero();
        let one = z7.one();
        assert!(zero.is_zero());
        assert!(one.is_one());
        // Parent-aware identities carry the modulus (no sentinel involved).
        assert_eq!(zero.modulus(), &Integer::from(7));
        assert_eq!(one.modulus(), &Integer::from(7));

        let x = z7.element(Integer::from(5)).unwrap();
        assert_eq!(x.clone() + zero, x);
        assert_eq!(x.clone() * one, x);
    }

    #[test]
    fn test_parent_trait() {
        let z6 = Zmod::new(Integer::from(6)).unwrap();
        let x = z6.element(Integer::from(4)).unwrap();
        assert!(z6.contains(&x));
        assert_eq!(Parent::cardinality(&z6), Some(6));
        assert!(Parent::is_finite(&z6));
        assert_eq!(Parent::zero(&z6), Some(z6.zero()));
        assert_eq!(Parent::one(&z6), Some(z6.one()));

        let z7 = Zmod::new(Integer::from(7)).unwrap();
        assert!(!z7.contains(&x));
    }

    #[test]
    fn test_coerce() {
        let z6 = Zmod::new(Integer::from(6)).unwrap();

        // Identity coercion.
        let x = z6.element(Integer::from(4)).unwrap();
        assert_eq!(z6.coerce(&x).unwrap(), x);

        // Sentinel coercion: the canonical map Z -> Z/6Z.
        let one = <IntegerMod as Ring>::one();
        let seven = one.clone()
            + one.clone()
            + one.clone()
            + one.clone()
            + one.clone()
            + one.clone()
            + one.clone();
        assert_eq!(
            z6.coerce(&seven).unwrap(),
            z6.element(Integer::from(1)).unwrap()
        );

        // Projection Z/12Z -> Z/6Z (6 | 12).
        let z12 = Zmod::new(Integer::from(12)).unwrap();
        let y = z12.element(Integer::from(10)).unwrap();
        assert_eq!(
            z6.coerce(&y).unwrap(),
            z6.element(Integer::from(4)).unwrap()
        );

        // No canonical map Z/7Z -> Z/6Z.
        let z7 = Zmod::new(Integer::from(7)).unwrap();
        let w = z7.element(Integer::from(3)).unwrap();
        assert!(z6.coerce(&w).is_err());
    }

    #[test]
    fn test_display() {
        let z10 = Zmod::new(Integer::from(10)).unwrap();
        assert_eq!(format!("{}", z10), "Ring of integers modulo 10");
        assert_eq!(Parent::name(&z10), "Ring of integers modulo 10");
    }

    #[test]
    fn test_factored_modulus() {
        let z12 = Zmod::new(Integer::from(12)).unwrap();
        let mut f = z12.factored_modulus();
        f.sort();
        assert_eq!(f, vec![(Integer::from(2), 2), (Integer::from(3), 1)]);
    }

    #[test]
    fn test_linear_congruence_solution() {
        // 3x = 5 (mod 7) -> x = 4 (3*4 = 12 = 5)
        let z7 = Zmod::new(Integer::from(7)).unwrap();
        let x = z7.solution(&Integer::from(3), &Integer::from(5)).unwrap();
        assert_eq!(x.value(), &Integer::from(4));
        // 2x = 3 (mod 4) has no solution (gcd(2,4) = 2 does not divide 3)
        let z4 = Zmod::new(Integer::from(4)).unwrap();
        assert!(z4.solution(&Integer::from(2), &Integer::from(3)).is_none());
        // 2x = 2 (mod 4) -> some x with 2x = 2
        let sol = z4.solution(&Integer::from(2), &Integer::from(2)).unwrap();
        assert_eq!(
            (z4.element(Integer::from(2)).unwrap() * sol).value(),
            &Integer::from(2)
        );
        // Negative a: -3x = 5 (mod 7) must still solve (gcd sign normalized;
        // the un-normalized version could return None or a wrong residue).
        let x = z7.solution(&Integer::from(-3), &Integer::from(5)).unwrap();
        assert_eq!(
            (z7.element(Integer::from(-3)).unwrap() * x).value(),
            &Integer::from(5)
        );
    }
}

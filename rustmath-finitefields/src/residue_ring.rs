//! Canonical integer residue class ring `Integers(m) = Z/mZ` (MAGMA ch. 19).
//!
//! MAGMA source: Chapter 19 — Integer Residue Class Rings (`ResidueClassRing`,
//! `quo< Z | m >`, `Modulus`, `FactoredModulus`, `Characteristic`, `IsField`,
//! `IsUnit`, `IsNilpotent`, `Order`, `Solution`, `Sqrt`, `AllSquareRoots`).
//!
//! This is the **canonical** `Z/mZ` parent for the port: it owns the modulus in
//! a shared [`UniqueRepresentation`] [`Parent`], and its element type unifies the
//! two legacy per-element modulus carriers ([`crate::IntegerMod`] here and
//! `rustmath_integers::modular::ModularInteger`). See the module note at the
//! bottom for the integrator: those two remain and should be reconciled onto
//! this type.
//!
//! `Integers(m)` implements [`CommutativeRing`] for every `m > 1`. It also
//! implements [`Field`], but that is only *mathematically valid when `m` is
//! prime*: [`ResidueClassRingElement::inverse`] returns an error for non-units,
//! and [`Integers::is_field`] reports whether the ring is genuinely a field. As
//! the MAGMA handbook itself notes, when `m` is prime one should prefer the
//! dedicated finite-field machinery ([`crate::FiniteField`], ch. 21).

use once_cell::sync::Lazy;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use rustmath_core::{
    CommutativeRing, EuclideanDomain, Field, MathError, NumericConversion, Parent, Result, Ring,
    UniqueCache, UniqueRepresentation,
};
use rustmath_integers::prime::factor;
use rustmath_integers::{crt_two, Integer};

use crate::integer_mod::square_root_mod_prime_power;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IntegersData {
    modulus: Integer,
    is_field: bool,
}

/// The residue class ring `Z/mZ`, carried as a shared [`Parent`].
#[derive(Clone)]
pub struct Integers {
    data: Arc<IntegersData>,
}

static ZM_CACHE: Lazy<UniqueCache<Integer, IntegersData>> = Lazy::new(UniqueCache::new);

impl UniqueRepresentation for IntegersData {
    type Key = Integer;
    fn key(&self) -> Self::Key {
        self.modulus.clone()
    }
    fn cache() -> &'static UniqueCache<Self::Key, Self> {
        &ZM_CACHE
    }
}

impl Integers {
    /// Construct `Integers(m) = Z/mZ` for `m > 1`.
    pub fn new(modulus: Integer) -> Result<Self> {
        if modulus <= Integer::one() {
            return Err(MathError::InvalidArgument("modulus must be > 1".into()));
        }
        let is_field = modulus.is_prime();
        let data = IntegersData::get_unique(modulus.clone(), || IntegersData { modulus, is_field });
        Ok(Integers { data })
    }

    /// The modulus `m`.
    pub fn modulus(&self) -> &Integer {
        &self.data.modulus
    }

    /// The characteristic (equals the modulus).
    pub fn characteristic(&self) -> &Integer {
        &self.data.modulus
    }

    /// The cardinality `m` as an [`Integer`].
    pub fn order(&self) -> Integer {
        self.data.modulus.clone()
    }

    /// Whether the ring is a field (i.e. the modulus is prime).
    pub fn is_field(&self) -> bool {
        self.data.is_field
    }

    /// The factored modulus `m = prod p_i^{e_i}` (MAGMA `FactoredModulus`).
    pub fn factored_modulus(&self) -> Vec<(Integer, u32)> {
        factor(&self.data.modulus)
    }

    /// Whether two rings are the same (same modulus).
    pub fn same_ring(&self, other: &Integers) -> bool {
        self.data.modulus == other.data.modulus
    }

    // -- element constructors -------------------------------------------------

    /// The element `k mod m`.
    pub fn element(&self, k: Integer) -> ResidueClassRingElement {
        let (_, r) = k.div_rem(&self.data.modulus).unwrap();
        let value = if r.signum() < 0 {
            r + self.data.modulus.clone()
        } else {
            r
        };
        ResidueClassRingElement {
            value,
            ring: self.clone(),
        }
    }

    /// The additive identity `0`.
    pub fn zero(&self) -> ResidueClassRingElement {
        ResidueClassRingElement {
            value: Integer::zero(),
            ring: self.clone(),
        }
    }

    /// The multiplicative identity `1`.
    pub fn one(&self) -> ResidueClassRingElement {
        ResidueClassRingElement {
            value: Integer::one(),
            ring: self.clone(),
        }
    }

    /// Solve the linear congruence `a*x = b (mod m)` (MAGMA `Solution`).
    /// Returns one solution `x`, or `None` if no solution exists.
    pub fn solution(&self, a: &Integer, b: &Integer) -> Option<ResidueClassRingElement> {
        let m = &self.data.modulus;
        let (g, s, _) = a.extended_gcd(m);
        let (q, r) = b.div_rem(&g).ok()?;
        if !r.is_zero() {
            return None; // g does not divide b
        }
        // x0 = (b/g) * s  (mod m/g), lifted to mod m
        let mg = m.clone() / g.clone();
        let (_, x0) = (q * s).div_rem(&mg).ok()?;
        let x0 = if x0.signum() < 0 { x0 + mg.clone() } else { x0 };
        Some(self.element(x0))
    }
}

impl fmt::Debug for Integers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Integers({})", self.data.modulus)
    }
}

impl fmt::Display for Integers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Z/{}Z", self.data.modulus)
    }
}

impl PartialEq for Integers {
    fn eq(&self, other: &Self) -> bool {
        self.data.modulus == other.data.modulus
    }
}
impl Eq for Integers {}

impl Parent for Integers {
    type Element = ResidueClassRingElement;
    fn contains(&self, element: &Self::Element) -> bool {
        element.ring.data.modulus == self.data.modulus
    }
    fn zero(&self) -> Option<Self::Element> {
        Some(Integers::zero(self))
    }
    fn one(&self) -> Option<Self::Element> {
        Some(Integers::one(self))
    }
    fn cardinality(&self) -> Option<usize> {
        self.data.modulus.to_usize()
    }
    fn name(&self) -> String {
        format!("{self}")
    }
}

// ---------------------------------------------------------------------------
// Element
// ---------------------------------------------------------------------------

/// An element of [`Integers`] `= Z/mZ`, carrying its shared parent ring.
#[derive(Clone)]
pub struct ResidueClassRingElement {
    value: Integer,
    ring: Integers,
}

impl ResidueClassRingElement {
    /// The canonical representative in `[0, m)`.
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Alias for [`Self::value`] (MAGMA lifts to `Z`).
    pub fn lift(&self) -> Integer {
        self.value.clone()
    }

    /// The parent ring.
    pub fn ring(&self) -> &Integers {
        &self.ring
    }

    fn modulus(&self) -> &Integer {
        &self.ring.data.modulus
    }

    fn assert_same(&self, other: &Self) {
        assert!(
            self.ring.data.modulus == other.ring.data.modulus,
            "operands live in different residue rings"
        );
    }

    /// Whether the element is a unit, i.e. `gcd(value, m) = 1` (MAGMA `IsUnit`).
    pub fn is_unit(&self) -> bool {
        let (g, _, _) = self.value.extended_gcd(self.modulus());
        g.is_one()
    }

    /// The multiplicative inverse, if the element is a unit.
    ///
    /// This is only guaranteed to succeed for every non-zero element when the
    /// modulus is prime (see the module note about the [`Field`] impl).
    pub fn inverse(&self) -> Result<Self> {
        if self.value.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        let (g, x, _) = self.value.extended_gcd(self.modulus());
        if !g.is_one() {
            return Err(MathError::NotInvertible);
        }
        let inv = if x.signum() < 0 {
            x + self.modulus().clone()
        } else {
            x
        };
        Ok(ResidueClassRingElement {
            value: inv,
            ring: self.ring.clone(),
        })
    }

    /// `self^exp` (modular exponentiation; negative exponents invert first).
    pub fn pow(&self, exp: &Integer) -> Result<Self> {
        if exp.signum() < 0 {
            return self.inverse()?.pow(&(-exp.clone()));
        }
        let v = self.value.mod_pow(exp, self.modulus())?;
        Ok(self.ring.element(v))
    }

    /// The multiplicative order of a unit, or `None` for a non-unit
    /// (MAGMA `Order`).
    pub fn multiplicative_order(&self) -> Option<Integer> {
        if !self.is_unit() {
            return None;
        }
        // order divides euler_phi(m); strip prime factors from phi.
        let phi = self.modulus().euler_phi().ok()?;
        let mut order = phi.clone();
        for (prime, mult) in factor(&phi) {
            for _ in 0..mult {
                let cand = order.clone() / prime.clone();
                if self.value.mod_pow(&cand, self.modulus()).ok()?.is_one() {
                    order = cand;
                } else {
                    break;
                }
            }
        }
        Some(order)
    }

    /// Whether the element is nilpotent (MAGMA `IsNilpotent`): true iff every
    /// prime dividing `m` divides `value`.
    pub fn is_nilpotent(&self) -> bool {
        if self.value.is_zero() {
            return true;
        }
        for (prime, _) in factor(self.modulus()) {
            let (_, r) = self.value.div_rem(&prime).unwrap();
            if !r.is_zero() {
                return false;
            }
        }
        true
    }

    /// Whether `value^2 = value` (MAGMA `IsIdempotent`).
    pub fn is_idempotent(&self) -> bool {
        let sq = self.clone() * self.clone();
        sq == *self
    }

    /// Whether the element is a zero divisor: non-zero and not a unit.
    pub fn is_zero_divisor(&self) -> bool {
        !self.value.is_zero() && !self.is_unit()
    }

    /// A square root of the element modulo `m`, if one exists (MAGMA `Sqrt`).
    /// Computed via the factorization of `m`, prime-power square roots, and CRT.
    pub fn sqrt(&self) -> Option<Self> {
        let m = self.modulus().clone();
        let factors = factor(&m);
        let mut moduli: Vec<Integer> = Vec::new();
        let mut roots: Vec<Integer> = Vec::new();
        for (prime, e) in &factors {
            let pk = prime.pow(*e);
            let r = square_root_mod_prime_power(&self.value, prime, *e as usize)?;
            moduli.push(pk);
            roots.push(r);
        }
        // CRT combine
        let mut cur_mod = Integer::one();
        let mut cur_val = Integer::zero();
        for (r, pk) in roots.iter().zip(moduli.iter()) {
            let combined = crt_two(&cur_val, &cur_mod, r, pk).ok()?;
            cur_val = combined;
            cur_mod = cur_mod * pk.clone();
        }
        Some(self.ring.element(cur_val))
    }
}

impl fmt::Debug for ResidueClassRingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.ring.data.modulus)
    }
}

impl fmt::Display for ResidueClassRingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl PartialEq for ResidueClassRingElement {
    fn eq(&self, other: &Self) -> bool {
        self.ring.data.modulus == other.ring.data.modulus && self.value == other.value
    }
}
impl Eq for ResidueClassRingElement {}

impl Add for ResidueClassRingElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.assert_same(&other);
        self.ring.element(self.value.clone() + other.value)
    }
}
impl Sub for ResidueClassRingElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.assert_same(&other);
        self.ring.element(self.value.clone() - other.value)
    }
}
impl Mul for ResidueClassRingElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.assert_same(&other);
        self.ring.element(self.value.clone() * other.value)
    }
}
impl Neg for ResidueClassRingElement {
    type Output = Self;
    fn neg(self) -> Self {
        if self.value.is_zero() {
            self
        } else {
            let v = self.modulus().clone() - self.value.clone();
            self.ring.element(v)
        }
    }
}
impl Div for ResidueClassRingElement {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.assert_same(&other);
        let inv = other.inverse().expect("division by non-unit in Z/mZ");
        self * inv
    }
}

impl Ring for ResidueClassRingElement {
    fn zero() -> Self {
        panic!("ResidueClassRingElement::zero() needs a parent; use Integers::zero()");
    }
    fn one() -> Self {
        panic!("ResidueClassRingElement::one() needs a parent; use Integers::one()");
    }
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

impl CommutativeRing for ResidueClassRingElement {}

// NOTE (honesty): Field is only mathematically valid when the modulus is prime.
// `inverse()` returns `Err(NotInvertible)` for non-units, and `Integers::is_field`
// reports the true status. See the module-level documentation.
impl Field for ResidueClassRingElement {
    fn inverse(&self) -> Result<Self> {
        ResidueClassRingElement::inverse(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_and_reduction() {
        let r = Integers::new(Integer::from(10)).unwrap();
        let a = r.element(Integer::from(17));
        assert_eq!(a.value(), &Integer::from(7));
        let b = r.element(Integer::from(-3));
        assert_eq!(b.value(), &Integer::from(7));
    }

    #[test]
    fn arithmetic() {
        let r = Integers::new(Integer::from(7)).unwrap();
        let a = r.element(Integer::from(3));
        let b = r.element(Integer::from(5));
        assert_eq!((a.clone() + b.clone()).value(), &Integer::from(1)); // 8=1
        assert_eq!((a.clone() * b.clone()).value(), &Integer::from(1)); // 15=1
        assert_eq!((a.clone() - b.clone()).value(), &Integer::from(5)); // -2=5
        assert_eq!((-a).value(), &Integer::from(4));
    }

    #[test]
    fn is_field_predicate() {
        assert!(Integers::new(Integer::from(7)).unwrap().is_field());
        assert!(!Integers::new(Integer::from(10)).unwrap().is_field());
    }

    #[test]
    fn inverse_and_field_division() {
        let r = Integers::new(Integer::from(7)).unwrap();
        let a = r.element(Integer::from(3));
        assert_eq!(a.inverse().unwrap().value(), &Integer::from(5)); // 3*5=15=1
        // Field division
        let six = r.element(Integer::from(6));
        let two = r.element(Integer::from(2));
        assert_eq!((six / two).value(), &Integer::from(3));
    }

    #[test]
    fn non_unit_inverse_errors() {
        let r = Integers::new(Integer::from(6)).unwrap();
        let a = r.element(Integer::from(2));
        assert!(!a.is_unit());
        assert!(a.inverse().is_err());
        assert!(a.is_zero_divisor());
    }

    #[test]
    fn predicates() {
        // Z/12Z: 6 is nilpotent? 6^2=36=0 mod 12 -> yes nilpotent.
        let r = Integers::new(Integer::from(12)).unwrap();
        assert!(r.element(Integer::from(6)).is_nilpotent());
        // 4 mod 12: 4^k never 0 (4,4,...) -> not nilpotent (2|4 but 3 not).
        assert!(!r.element(Integer::from(4)).is_nilpotent());
        // idempotent: 4 mod 12 -> 16=4 -> idempotent
        assert!(r.element(Integer::from(4)).is_idempotent());
        // Z/8Z: 2 nilpotent (2^3=8=0)
        let r8 = Integers::new(Integer::from(8)).unwrap();
        assert!(r8.element(Integer::from(2)).is_nilpotent());
    }

    #[test]
    fn multiplicative_order() {
        let r = Integers::new(Integer::from(7)).unwrap();
        assert_eq!(
            r.element(Integer::from(3)).multiplicative_order(),
            Some(Integer::from(6))
        );
        assert_eq!(
            r.element(Integer::from(2)).multiplicative_order(),
            Some(Integer::from(3))
        );
        // non-unit has no order
        let r6 = Integers::new(Integer::from(6)).unwrap();
        assert_eq!(r6.element(Integer::from(2)).multiplicative_order(), None);
    }

    #[test]
    fn linear_congruence_solution() {
        // 3x = 5 (mod 7) -> x = 4 (3*4=12=5)
        let r = Integers::new(Integer::from(7)).unwrap();
        let x = r.solution(&Integer::from(3), &Integer::from(5)).unwrap();
        assert_eq!(x.value(), &Integer::from(4));
        // 2x = 3 (mod 4) has no solution (gcd(2,4)=2 does not divide 3)
        let r4 = Integers::new(Integer::from(4)).unwrap();
        assert!(r4.solution(&Integer::from(2), &Integer::from(3)).is_none());
        // 2x = 2 (mod 4) -> x = 1 (a solution)
        let sol = r4.solution(&Integer::from(2), &Integer::from(2)).unwrap();
        assert_eq!(
            (r4.element(Integer::from(2)) * sol.clone()).value(),
            &Integer::from(2)
        );
    }

    #[test]
    fn sqrt_composite() {
        // sqrt of 4 mod 15 (=3*5): should square back to 4.
        let r = Integers::new(Integer::from(15)).unwrap();
        let a = r.element(Integer::from(4));
        let s = a.sqrt().unwrap();
        assert_eq!((s.clone() * s).value(), &Integer::from(4));
    }

    #[test]
    fn shared_parent_uniqueness() {
        let r1 = Integers::new(Integer::from(9)).unwrap();
        let r2 = Integers::new(Integer::from(9)).unwrap();
        assert!(Arc::ptr_eq(&r1.data, &r2.data));
    }
}

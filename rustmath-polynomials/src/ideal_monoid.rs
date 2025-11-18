//! Ideal Monoid
//!
//! This module implements the monoid of ideals under ideal multiplication.
//! The set of ideals in a ring forms a monoid under multiplication, with
//! the unit ideal as the identity element.

use crate::ideal::Ideal;
use crate::groebner::MonomialOrdering;
use rustmath_core::Ring;
use std::fmt;

/// The monoid of ideals in a polynomial ring
///
/// This represents the multiplicative structure of ideals.
/// Ideals form a commutative monoid under multiplication:
/// - Closure: I * J is an ideal
/// - Associativity: (I * J) * K = I * (J * K)
/// - Identity: I * ⟨1⟩ = I
#[derive(Clone, Debug)]
pub struct IdealMonoid<R: Ring> {
    /// The default monomial ordering for ideals
    ordering: MonomialOrdering,
    /// Phantom data for the coefficient ring
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> IdealMonoid<R> {
    /// Create a new ideal monoid with the specified ordering
    ///
    /// # Examples
    /// ```
    /// use rustmath_polynomials::ideal_monoid::IdealMonoid;
    /// use rustmath_polynomials::groebner::MonomialOrdering;
    /// use rustmath_integers::Integer;
    ///
    /// let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
    /// ```
    pub fn new(ordering: MonomialOrdering) -> Self {
        IdealMonoid {
            ordering,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the identity element (unit ideal)
    ///
    /// The unit ideal ⟨1⟩ is the identity for ideal multiplication
    pub fn identity(&self) -> Ideal<R> {
        Ideal::unit(self.ordering)
    }

    /// Get the zero element (zero ideal)
    ///
    /// The zero ideal ⟨0⟩ is the absorbing element for ideal multiplication
    pub fn zero(&self) -> Ideal<R> {
        Ideal::zero(self.ordering)
    }

    /// Multiply two ideals
    ///
    /// Returns I * J = ⟨{ab : a ∈ I, b ∈ J}⟩
    pub fn multiply(&self, i1: &Ideal<R>, i2: &Ideal<R>) -> Ideal<R>
    where
        R: Clone,
    {
        i1.product(i2)
    }

    /// Compute the power of an ideal
    ///
    /// Returns I^n = I * I * ... * I (n times)
    pub fn power(&self, ideal: &Ideal<R>, n: usize) -> Ideal<R>
    where
        R: Clone,
    {
        if n == 0 {
            return self.identity();
        }
        if ideal.is_zero() {
            return self.zero();
        }
        if ideal.is_unit() {
            return self.identity();
        }

        let mut result = ideal.clone();
        for _ in 1..n {
            result = self.multiply(&result, ideal);
        }
        result
    }

    /// Check if an ideal is the identity
    pub fn is_identity(&self, ideal: &Ideal<R>) -> bool {
        ideal.is_unit()
    }

    /// Check if an ideal is the zero
    pub fn is_zero(&self, ideal: &Ideal<R>) -> bool {
        ideal.is_zero()
    }

    /// Get the ordering used by this monoid
    pub fn ordering(&self) -> MonomialOrdering {
        self.ordering
    }
}

impl<R: Ring> fmt::Display for IdealMonoid<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Monoid of ideals with {:?} ordering", self.ordering)
    }
}

/// Check if an object is an IdealMonoid
///
/// This is a type-checking function for compatibility with dynamic typing
pub fn is_ideal_monoid<R: Ring>(_obj: &IdealMonoid<R>) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_create_ideal_monoid() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        assert!(is_ideal_monoid(&monoid));
    }

    #[test]
    fn test_identity_element() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let identity = monoid.identity();
        assert!(identity.is_unit());
        assert!(monoid.is_identity(&identity));
    }

    #[test]
    fn test_zero_element() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let zero = monoid.zero();
        assert!(zero.is_zero());
        assert!(monoid.is_zero(&zero));
    }

    #[test]
    fn test_multiply_ideals() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let i1 = monoid.identity();
        let i2 = monoid.zero();

        let product = monoid.multiply(&i1, &i2);
        assert!(product.is_zero());
    }

    #[test]
    fn test_multiply_identity() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let identity = monoid.identity();

        let product = monoid.multiply(&identity, &identity);
        assert!(product.is_unit());
    }

    #[test]
    fn test_power_zero() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let ideal = monoid.identity();

        let power = monoid.power(&ideal, 0);
        assert!(power.is_unit());
    }

    #[test]
    fn test_power_one() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let ideal = monoid.identity();

        let power = monoid.power(&ideal, 1);
        assert!(power.is_unit());
    }

    #[test]
    fn test_power_multiple() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let ideal = monoid.identity();

        let power = monoid.power(&ideal, 5);
        assert!(power.is_unit());
    }

    #[test]
    fn test_power_zero_ideal() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let ideal = monoid.zero();

        let power = monoid.power(&ideal, 3);
        assert!(power.is_zero());
    }

    #[test]
    fn test_display() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Lex);
        let display = format!("{}", monoid);
        assert!(display.contains("Monoid of ideals"));
    }

    #[test]
    fn test_ordering() {
        let monoid: IdealMonoid<Integer> = IdealMonoid::new(MonomialOrdering::Grlex);
        assert_eq!(monoid.ordering(), MonomialOrdering::Grlex);
    }
}

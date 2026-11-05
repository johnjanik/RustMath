//! Adoption of the `rustmath-core` algebraic tower for the existing free-monoid
//! word type.
//!
//! MAGMA handbook chapter 77 (Finitely Presented Semigroups): a free monoid is
//! the monoid of all finite words over its generators under concatenation, with
//! the empty word as identity. This file supplies the `rustmath_core`
//! `Magma → Semigroup → Monoid` implementations for
//! [`crate::free_monoid::FreeMonoidElement`] without modifying that module,
//! following the port rule that trait impls for existing types live in separate
//! `*_core_impls.rs` files.
//!
//! (The crate also keeps its own local `Monoid` trait in `crate::monoid`; new
//! code should prefer these core-trait impls.)

use crate::free_monoid::FreeMonoidElement;
use rustmath_core::{Magma, Monoid, Semigroup};

impl Magma for FreeMonoidElement {
    /// Word concatenation.
    fn op(&self, other: &Self) -> Self {
        self.mul(other)
    }
}

impl Semigroup for FreeMonoidElement {}

impl Monoid for FreeMonoidElement {
    /// The empty word.
    fn identity() -> Self {
        FreeMonoidElement::identity()
    }

    fn is_identity(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_monoid_element_is_a_core_monoid() {
        let a = FreeMonoidElement::new(vec![0]);
        let b = FreeMonoidElement::new(vec![1]);
        let id = <FreeMonoidElement as Monoid>::identity();

        // associativity of concatenation
        let ab_c = a.op(&b).op(&FreeMonoidElement::new(vec![0]));
        let a_bc = a.op(&b.op(&FreeMonoidElement::new(vec![0])));
        assert_eq!(ab_c, a_bc);
        assert_eq!(ab_c.word(), &[0, 1, 0]);

        // identity laws
        assert_eq!(a.op(&id), a);
        assert_eq!(id.op(&a), a);
        assert!(id.is_identity());
        assert!(!a.is_identity());
    }
}

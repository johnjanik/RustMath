//! Group Semidirect Products
//!
//! This module implements semidirect products of groups. A semidirect product
//! G ⋊_φ H is constructed from two groups G and H along with a homomorphism
//! φ: H → Aut(G).
//!
//! #Theory
//!
//! The semidirect product G ⋊_φ H consists of pairs (g, h) with g ∈ G and h ∈ H,
//! with multiplication given by:
//!   (g₁, h₁)(g₂, h₂) = (g₁ φ(h₁)(g₂), h₁h₂)
//!
//! When φ is trivial, this reduces to the direct product G × H.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::semidirect_product::*;
//!
//! // Create the dihedral group as a semidirect product
//! // D_n ≅ Z_n ⋊ Z_2, where Z_2 acts on Z_n by inversion
//! ```

use std::fmt;
use std::marker::PhantomData;

/// A semidirect product of two groups G ⋊_φ H
///
/// Elements are pairs (g, h) where g ∈ G and h ∈ H.
/// The multiplication is defined by the homomorphism φ: H → Aut(G).
#[derive(Clone, Debug)]
pub struct GroupSemidirectProduct<G, H, F>
where
    F: Fn(&H, &G) -> G,
{
    /// The normal subgroup G
    normal_subgroup: PhantomData<G>,
    /// The quotient group H
    quotient_group: PhantomData<H>,
    /// The action homomorphism φ: H × G → G
    /// φ(h)(g) represents how h ∈ H acts on g ∈ G
    action: F,
}

/// An element of a semidirect product G ⋊ H
///
/// Represented as a pair (g, h) with g ∈ G and h ∈ H.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupSemidirectProductElement<G, H> {
    /// Element from the normal subgroup G
    pub g_component: G,
    /// Element from the quotient group H
    pub h_component: H,
}

impl<G, H, F> GroupSemidirectProduct<G, H, F>
where
    F: Fn(&H, &G) -> G,
{
    /// Create a new semidirect product with the given action
    ///
    /// # Arguments
    ///
    /// * `action` - The homomorphism φ: H × G → G defining how H acts on G
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_groups::semidirect_product::*;
    ///
    /// // Create a semidirect product with a specific action
    /// let action = |h: &i32, g: &i32| {
    ///     if *h % 2 == 0 { *g } else { -g }
    /// };
    /// let semidirect = GroupSemidirectProduct::new(action);
    /// ```
    pub fn new(action: F) -> Self {
        GroupSemidirectProduct {
            normal_subgroup: PhantomData,
            quotient_group: PhantomData,
            action,
        }
    }

    /// Create an element of the semidirect product
    pub fn element(&self, g: G, h: H) -> GroupSemidirectProductElement<G, H> {
        GroupSemidirectProductElement {
            g_component: g,
            h_component: h,
        }
    }

    /// Get the action homomorphism
    pub fn action(&self) -> &F {
        &self.action
    }
}

impl<G, H> GroupSemidirectProductElement<G, H>
where
    G: Clone,
    H: Clone,
{
    /// Create a new semidirect product element
    pub fn new(g: G, h: H) -> Self {
        GroupSemidirectProductElement {
            g_component: g,
            h_component: h,
        }
    }

    /// Get the G component
    pub fn g(&self) -> &G {
        &self.g_component
    }

    /// Get the H component
    pub fn h(&self) -> &H {
        &self.h_component
    }

    /// Decompose into components
    pub fn into_components(self) -> (G, H) {
        (self.g_component, self.h_component)
    }
}

impl<G, H, F> GroupSemidirectProduct<G, H, F>
where
    G: Clone + Mul<Output = G>,
    H: Clone + Mul<Output = H>,
    F: Fn(&H, &G) -> G,
{
    /// Multiply two elements in the semidirect product
    ///
    /// (g₁, h₁)(g₂, h₂) = (g₁ φ(h₁)(g₂), h₁h₂)
    pub fn multiply(
        &self,
        left: &GroupSemidirectProductElement<G, H>,
        right: &GroupSemidirectProductElement<G, H>,
    ) -> GroupSemidirectProductElement<G, H> {
        // Compute φ(h₁)(g₂)
        let action_result = (self.action)(&left.h_component, &right.g_component);

        // Compute g₁ * φ(h₁)(g₂)
        let g_result = left.g_component.clone() * action_result;

        // Compute h₁ * h₂
        let h_result = left.h_component.clone() * right.h_component.clone();

        GroupSemidirectProductElement {
            g_component: g_result,
            h_component: h_result,
        }
    }
}

/// Trait for groups that can form semidirect products
pub trait SemidirectProductCompatible: Sized + Clone {
    /// The identity element
    fn identity() -> Self;

    /// Multiply two elements
    fn mul(&self, other: &Self) -> Self;

    /// Compute the inverse
    fn inverse(&self) -> Self;
}

// Implement for common integer types as additive groups
impl SemidirectProductCompatible for i32 {
    fn identity() -> Self {
        0
    }

    fn mul(&self, other: &Self) -> Self {
        self + other
    }

    fn inverse(&self) -> Self {
        -self
    }
}

impl SemidirectProductCompatible for i64 {
    fn identity() -> Self {
        0
    }

    fn mul(&self, other: &Self) -> Self {
        self + other
    }

    fn inverse(&self) -> Self {
        -self
    }
}

/// Convenience trait for multiplication
pub trait Mul {
    type Output;
    fn mul(self, rhs: Self) -> Self::Output;
}

// Implement Mul for i32 (additive group)
impl Mul for i32 {
    type Output = i32;
    fn mul(self, rhs: Self) -> Self::Output {
        self + rhs
    }
}

// Implement Mul for i64 (additive group)
impl Mul for i64 {
    type Output = i64;
    fn mul(self, rhs: Self) -> Self::Output {
        self + rhs
    }
}

impl<G, H> fmt::Display for GroupSemidirectProductElement<G, H>
where
    G: fmt::Display,
    H: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.g_component, self.h_component)
    }
}

/// Create the direct product G × H (trivial semidirect product)
pub fn direct_product<G, H>() -> GroupSemidirectProduct<G, H, impl Fn(&H, &G) -> G>
where
    G: Clone,
    H: Clone,
{
    GroupSemidirectProduct::new(|_h: &H, g: &G| g.clone())
}

/// Create the dihedral group D_n as a semidirect product Z_n ⋊ Z_2
///
/// The group Z_2 = {0, 1} acts on Z_n by:
/// - 0 acts trivially: φ(0)(k) = k
/// - 1 acts by inversion: φ(1)(k) = -k (mod n)
pub fn dihedral_group(n: usize) -> GroupSemidirectProduct<i32, i32, impl Fn(&i32, &i32) -> i32> {
    GroupSemidirectProduct::new(move |h: &i32, g: &i32| {
        if h % 2 == 0 {
            *g
        } else {
            (n as i32 - g) % n as i32
        }
    })
}

/// Create the holomorph of a group
///
/// The holomorph Hol(G) = G ⋊ Aut(G) where Aut(G) acts on G by automorphisms.
/// This is a placeholder for future implementation.
pub struct Holomorph<G> {
    _phantom: PhantomData<G>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_product() {
        let product: GroupSemidirectProduct<i32, i32, _> = direct_product();

        let e1 = GroupSemidirectProductElement::new(3, 4);
        let e2 = GroupSemidirectProductElement::new(5, 6);

        let result = product.multiply(&e1, &e2);

        // In direct product: (3, 4) * (5, 6) = (3+5, 4+6) = (8, 10)
        assert_eq!(result.g_component, 8);
        assert_eq!(result.h_component, 10);
    }

    #[test]
    fn test_dihedral_group() {
        let d4 = dihedral_group(4); // D_4 has 8 elements

        // Identity in both components
        let e = GroupSemidirectProductElement::new(0, 0);

        // Rotation by 1
        let r = GroupSemidirectProductElement::new(1, 0);

        // Reflection
        let s = GroupSemidirectProductElement::new(0, 1);

        // Test r^4 = e (rotation by 4 is identity mod 4)
        let r2 = d4.multiply(&r, &r);
        let r3 = d4.multiply(&r2, &r);
        let r4 = d4.multiply(&r3, &r);
        assert_eq!(r4.g_component % 4, 0);

        // Test s^2 = e (reflection squared is identity)
        let s2 = d4.multiply(&s, &s);
        assert_eq!(s2.h_component % 2, 0);

        // Test srs = r^{-1} (conjugation reverses rotation)
        let sr = d4.multiply(&s, &r);
        let srs = d4.multiply(&sr, &s);
        // s * r * s should give rotation by -1, which is 3 mod 4
        assert_eq!(srs.g_component % 4, 3);
    }

    #[test]
    fn test_element_creation() {
        let elem = GroupSemidirectProductElement::new(5, 7);

        assert_eq!(elem.g(), &5);
        assert_eq!(elem.h(), &7);

        let (g, h) = elem.into_components();
        assert_eq!(g, 5);
        assert_eq!(h, 7);
    }

    #[test]
    fn test_custom_action() {
        // Create a semidirect product with custom action
        // φ(h)(g) = h * g (scalar multiplication)
        let action = |h: &i32, g: &i32| h * g;
        let sp = GroupSemidirectProduct::new(action);

        let e1 = GroupSemidirectProductElement::new(2, 3);
        let e2 = GroupSemidirectProductElement::new(4, 5);

        // (2, 3) * (4, 5) = (2 + 3*4, 3*5) = (2 + 12, 15) = (14, 15)
        let result = sp.multiply(&e1, &e2);

        assert_eq!(result.g_component, 14);
        assert_eq!(result.h_component, 15);
    }

    #[test]
    fn test_display() {
        let elem = GroupSemidirectProductElement::new(10, 20);
        let display = format!("{}", elem);
        assert_eq!(display, "(10, 20)");
    }

    #[test]
    fn test_identity() {
        assert_eq!(i32::identity(), 0);
        assert_eq!(i64::identity(), 0);
    }

    #[test]
    fn test_inverse() {
        assert_eq!(5i32.inverse(), -5);
        assert_eq!((-10i64).inverse(), 10);
    }

    #[test]
    fn test_semidirect_product_compatible() {
        let a = 7i32;
        let b = 3i32;

        assert_eq!(a.mul(&b), 10);
        assert_eq!(b.mul(&a), 10);
    }

    #[test]
    fn test_associativity() {
        let sp: GroupSemidirectProduct<i32, i32, _> = direct_product();

        let e1 = GroupSemidirectProductElement::new(1, 2);
        let e2 = GroupSemidirectProductElement::new(3, 4);
        let e3 = GroupSemidirectProductElement::new(5, 6);

        // (e1 * e2) * e3
        let left_assoc = sp.multiply(&sp.multiply(&e1, &e2), &e3);

        // e1 * (e2 * e3)
        let right_assoc = sp.multiply(&e1, &sp.multiply(&e2, &e3));

        assert_eq!(left_assoc.g_component, right_assoc.g_component);
        assert_eq!(left_assoc.h_component, right_assoc.h_component);
    }
}

//! Morphisms for algebraic structures
//!
//! This module provides concrete morphism types for algebraic structures
//! like rings, fields, modules, and algebras. These morphisms preserve
//! the algebraic structure and integrate with the category theory framework.
//!
//! # Structure Preservation
//!
//! - **Ring Homomorphism**: φ: R → S such that:
//!   - φ(a + b) = φ(a) + φ(b) (preserves addition)
//!   - φ(a × b) = φ(a) × φ(b) (preserves multiplication)
//!   - φ(1_R) = 1_S (preserves unity)
//!
//! - **Field Homomorphism**: A ring homomorphism between fields
//!
//! - **Module Homomorphism**: φ: M → N such that:
//!   - φ(m + n) = φ(m) + φ(n) (preserves addition)
//!   - φ(r·m) = r·φ(m) (preserves scalar multiplication)

use crate::morphism::Morphism;
use std::fmt;
use std::marker::PhantomData;

/// A ring homomorphism φ: R → S
///
/// Preserves ring structure: addition, multiplication, and unity
///
/// # Properties
/// For all a, b ∈ R:
/// - φ(a + b) = φ(a) + φ(b)
/// - φ(a × b) = φ(a) × φ(b)
/// - φ(0_R) = 0_S
/// - φ(1_R) = 1_S
///
/// # Examples
/// - Embedding Z → Q (integers into rationals)
/// - Evaluation map R[x] → R at a point
/// - Reduction mod n: Z → Z/nZ
#[derive(Clone, Debug)]
pub struct RingMorphism<R, S, F>
where
    F: Fn(&R) -> S + Clone,
{
    source_ring: R,
    target_ring: S,
    map: F,
    _phantom: PhantomData<(R, S)>,
}

impl<R, S, F> RingMorphism<R, S, F>
where
    F: Fn(&R) -> S + Clone,
{
    /// Create a new ring homomorphism
    ///
    /// # Arguments
    /// - `source_ring`: The source ring (domain)
    /// - `target_ring`: The target ring (codomain)
    /// - `map`: The function implementing the homomorphism
    ///
    /// Note: This constructor does not verify that `map` is actually a homomorphism.
    /// The caller is responsible for ensuring structure preservation.
    pub fn new(source_ring: R, target_ring: S, map: F) -> Self {
        RingMorphism {
            source_ring,
            target_ring,
            map,
            _phantom: PhantomData,
        }
    }

    /// Apply the ring homomorphism to an element
    pub fn apply(&self, element: &R) -> S {
        (self.map)(element)
    }

    /// Get the source ring
    pub fn source_ring(&self) -> &R {
        &self.source_ring
    }

    /// Get the target ring
    pub fn target_ring(&self) -> &S {
        &self.target_ring
    }
}

impl<R: Clone, S: Clone, F> Morphism for RingMorphism<R, S, F>
where
    F: Fn(&R) -> S + Clone,
{
    type Object = R;

    fn source(&self) -> &Self::Object {
        &self.source_ring
    }

    fn target(&self) -> &Self::Object {
        &self.source_ring // Simplified: same type
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        // Composition of ring homomorphisms
        // Would require storing composed functions, which has type system challenges
        None
    }
}

impl<R: fmt::Display, S: fmt::Display, F> fmt::Display for RingMorphism<R, S, F>
where
    F: Fn(&R) -> S + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ring Homomorphism: {} → {}",
            self.source_ring, self.target_ring
        )
    }
}

/// A field homomorphism φ: F → K
///
/// A ring homomorphism between fields. Field homomorphisms are always injective
/// (one-to-one) because fields have no proper ideals.
///
/// # Properties
/// - All properties of ring homomorphisms
/// - Injective: ker(φ) = {0}
/// - Either φ is zero map or isomorphism onto image
#[derive(Clone, Debug)]
pub struct FieldMorphism<F, K, M>
where
    M: Fn(&F) -> K + Clone,
{
    source_field: F,
    target_field: K,
    map: M,
    _phantom: PhantomData<(F, K)>,
}

impl<F, K, M> FieldMorphism<F, K, M>
where
    M: Fn(&F) -> K + Clone,
{
    /// Create a new field homomorphism
    pub fn new(source_field: F, target_field: K, map: M) -> Self {
        FieldMorphism {
            source_field,
            target_field,
            map,
            _phantom: PhantomData,
        }
    }

    /// Apply the field homomorphism to an element
    pub fn apply(&self, element: &F) -> K {
        (self.map)(element)
    }

    /// Get the source field
    pub fn source_field(&self) -> &F {
        &self.source_field
    }

    /// Get the target field
    pub fn target_field(&self) -> &K {
        &self.target_field
    }

    /// Check if this is an embedding (injective)
    ///
    /// All non-trivial field homomorphisms are injective
    pub fn is_embedding(&self) -> bool {
        true // Field homomorphisms are always injective
    }
}

impl<F: Clone, K: Clone, M> Morphism for FieldMorphism<F, K, M>
where
    M: Fn(&F) -> K + Clone,
{
    type Object = F;

    fn source(&self) -> &Self::Object {
        &self.source_field
    }

    fn target(&self) -> &Self::Object {
        &self.source_field // Simplified
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        None // Composition not implemented due to type system constraints
    }
}

impl<F: fmt::Display, K: fmt::Display, M> fmt::Display for FieldMorphism<F, K, M>
where
    M: Fn(&F) -> K + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Field Homomorphism: {} → {}",
            self.source_field, self.target_field
        )
    }
}

/// A module homomorphism φ: M → N over a ring R
///
/// Preserves module structure: addition and scalar multiplication
///
/// # Properties
/// For all m, n ∈ M and r ∈ R:
/// - φ(m + n) = φ(m) + φ(n)
/// - φ(r·m) = r·φ(m)
/// - φ(0) = 0
#[derive(Clone, Debug)]
pub struct ModuleMorphism<M, N, F>
where
    F: Fn(&M) -> N + Clone,
{
    source_module: M,
    target_module: N,
    map: F,
    _phantom: PhantomData<(M, N)>,
}

impl<M, N, F> ModuleMorphism<M, N, F>
where
    F: Fn(&M) -> N + Clone,
{
    /// Create a new module homomorphism
    pub fn new(source_module: M, target_module: N, map: F) -> Self {
        ModuleMorphism {
            source_module,
            target_module,
            map,
            _phantom: PhantomData,
        }
    }

    /// Apply the module homomorphism to an element
    pub fn apply(&self, element: &M) -> N {
        (self.map)(element)
    }

    /// Get the source module
    pub fn source_module(&self) -> &M {
        &self.source_module
    }

    /// Get the target module
    pub fn target_module(&self) -> &N {
        &self.target_module
    }
}

impl<M: Clone, N: Clone, F> Morphism for ModuleMorphism<M, N, F>
where
    F: Fn(&M) -> N + Clone,
{
    type Object = M;

    fn source(&self) -> &Self::Object {
        &self.source_module
    }

    fn target(&self) -> &Self::Object {
        &self.source_module // Simplified
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        None
    }
}

/// An algebra homomorphism φ: A → B over a field F
///
/// Preserves both ring structure and vector space structure
///
/// # Properties
/// - Ring homomorphism: φ(a × b) = φ(a) × φ(b)
/// - Module homomorphism: φ(λ·a) = λ·φ(a)
#[derive(Clone, Debug)]
pub struct AlgebraMorphism<A, B, F>
where
    F: Fn(&A) -> B + Clone,
{
    source_algebra: A,
    target_algebra: B,
    map: F,
    _phantom: PhantomData<(A, B)>,
}

impl<A, B, F> AlgebraMorphism<A, B, F>
where
    F: Fn(&A) -> B + Clone,
{
    /// Create a new algebra homomorphism
    pub fn new(source_algebra: A, target_algebra: B, map: F) -> Self {
        AlgebraMorphism {
            source_algebra,
            target_algebra,
            map,
            _phantom: PhantomData,
        }
    }

    /// Apply the algebra homomorphism to an element
    pub fn apply(&self, element: &A) -> B {
        (self.map)(element)
    }
}

/// A group homomorphism φ: G → H
///
/// Preserves group structure
///
/// # Properties
/// For all g, h ∈ G:
/// - φ(g · h) = φ(g) · φ(h)
/// - φ(e_G) = e_H (preserves identity)
/// - φ(g⁻¹) = φ(g)⁻¹ (preserves inverses - follows from above)
#[derive(Clone, Debug)]
pub struct GroupMorphism<G, H, F>
where
    F: Fn(&G) -> H + Clone,
{
    source_group: G,
    target_group: H,
    map: F,
    _phantom: PhantomData<(G, H)>,
}

impl<G, H, F> GroupMorphism<G, H, F>
where
    F: Fn(&G) -> H + Clone,
{
    /// Create a new group homomorphism
    pub fn new(source_group: G, target_group: H, map: F) -> Self {
        GroupMorphism {
            source_group,
            target_group,
            map,
            _phantom: PhantomData,
        }
    }

    /// Apply the group homomorphism to an element
    pub fn apply(&self, element: &G) -> H {
        (self.map)(element)
    }

    /// Get the source group
    pub fn source_group(&self) -> &G {
        &self.source_group
    }

    /// Get the target group
    pub fn target_group(&self) -> &H {
        &self.target_group
    }
}

impl<G: Clone, H: Clone, F> Morphism for GroupMorphism<G, H, F>
where
    F: Fn(&G) -> H + Clone,
{
    type Object = G;

    fn source(&self) -> &Self::Object {
        &self.source_group
    }

    fn target(&self) -> &Self::Object {
        &self.source_group // Simplified
    }

    fn compose(&self, _other: &Self) -> Option<Self> {
        None
    }
}

/// Common ring homomorphisms
pub mod standard {
    use super::*;

    /// Identity ring homomorphism: R → R
    pub fn identity_ring_morphism<R: Clone>(ring: R) -> RingMorphism<R, R, impl Fn(&R) -> R + Clone> {
        let ring_clone = ring.clone();
        RingMorphism::new(ring, ring_clone, |x: &R| x.clone())
    }

    /// Zero ring homomorphism: R → S (maps everything to 0)
    ///
    /// Note: Only a ring homomorphism if R is the zero ring
    pub fn zero_ring_morphism<R: Clone, S: Clone>(
        source: R,
        target: S,
        zero: S,
    ) -> RingMorphism<R, S, impl Fn(&R) -> S + Clone> {
        RingMorphism::new(source, target, move |_: &R| zero.clone())
    }

    /// Inclusion morphism for subrings
    ///
    /// When S is a subring of R, this is the natural inclusion S ↪ R
    pub fn inclusion_morphism<T: Clone>(
        subring: T,
        ring: T,
    ) -> RingMorphism<T, T, impl Fn(&T) -> T + Clone> {
        RingMorphism::new(subring, ring, |x: &T| x.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_morphism_creation() {
        let morph = RingMorphism::new(0, 0, |x: &i32| x * 2);
        assert_eq!(morph.apply(&5), 10);
    }

    #[test]
    fn test_ring_morphism_identity() {
        let morph = standard::identity_ring_morphism(42);
        assert_eq!(morph.apply(&42), 42);
        assert_eq!(morph.source_ring(), &42);
    }

    #[test]
    fn test_ring_morphism_zero() {
        let morph = standard::zero_ring_morphism(0, 0, 0);
        assert_eq!(morph.apply(&42), 0);
        assert_eq!(morph.apply(&100), 0);
    }

    #[test]
    fn test_ring_morphism_display() {
        let morph = RingMorphism::new(5, 10, |x: &i32| x * 2);
        let display = format!("{}", morph);
        assert!(display.contains("Ring Homomorphism"));
    }

    #[test]
    fn test_field_morphism_creation() {
        let morph = FieldMorphism::new(1.0, 1.0, |x: &f64| x.sqrt());
        assert_eq!(morph.apply(&4.0), 2.0);
    }

    #[test]
    fn test_field_morphism_is_embedding() {
        let morph = FieldMorphism::new(1.0, 1.0, |x: &f64| *x);
        assert!(morph.is_embedding());
    }

    #[test]
    fn test_field_morphism_display() {
        let morph = FieldMorphism::new(1.0, 2.0, |x: &f64| *x);
        let display = format!("{}", morph);
        assert!(display.contains("Field Homomorphism"));
    }

    #[test]
    fn test_module_morphism_creation() {
        let morph = ModuleMorphism::new(vec![0], vec![0, 0], |v: &Vec<i32>| {
            vec![v[0], v[0]]
        });
        let result = morph.apply(&vec![5]);
        assert_eq!(result, vec![5, 5]);
    }

    #[test]
    fn test_algebra_morphism_creation() {
        let morph = AlgebraMorphism::new(0, 0, |x: &i32| x * 3);
        assert_eq!(morph.apply(&5), 15);
    }

    #[test]
    fn test_group_morphism_creation() {
        let morph = GroupMorphism::new(0, 0, |x: &i32| x + 10);
        assert_eq!(morph.apply(&5), 15);
        assert_eq!(morph.source_group(), &0);
        assert_eq!(morph.target_group(), &0);
    }

    #[test]
    fn test_inclusion_morphism() {
        let morph = standard::inclusion_morphism(0, 0);
        assert_eq!(morph.apply(&42), 42);
    }

    // Test that morphisms implement the Morphism trait
    #[test]
    fn test_ring_morphism_as_morphism() {
        let morph = RingMorphism::new(0, 0, |x: &i32| *x);
        assert_eq!(morph.source(), &0);
        assert_eq!(morph.target(), &0);
    }

    #[test]
    fn test_field_morphism_as_morphism() {
        let morph = FieldMorphism::new(0.0, 0.0, |x: &f64| *x);
        assert_eq!(morph.source(), &0.0);
    }

    #[test]
    fn test_module_morphism_as_morphism() {
        let morph = ModuleMorphism::new(vec![0], vec![0], |v: &Vec<i32>| v.clone());
        assert_eq!(morph.source(), &vec![0]);
    }

    #[test]
    fn test_group_morphism_as_morphism() {
        let morph = GroupMorphism::new(0, 0, |x: &i32| *x);
        assert_eq!(morph.source(), &0);
    }

    // Example: embedding Z → Q (conceptually)
    #[test]
    fn test_integer_to_rational_concept() {
        // This is a conceptual test showing how you'd use it
        // In practice, you'd use actual Integer and Rational types
        let embed = RingMorphism::new(0i32, (0i32, 1i32), |n: &i32| (*n, 1i32));
        let result = embed.apply(&5);
        assert_eq!(result, (5, 1)); // Represents 5/1
    }

    // Example: evaluation morphism for polynomials
    #[test]
    fn test_polynomial_evaluation_concept() {
        // Conceptual: evaluate polynomial at x = 2
        // In practice, you'd use actual Polynomial type
        let coeffs = vec![1, 2, 3]; // represents 1 + 2x + 3x²
        let eval_at_2 = RingMorphism::new(coeffs.clone(), 0, |poly: &Vec<i32>| {
            // Evaluate at x = 2: 1 + 2(2) + 3(4) = 1 + 4 + 12 = 17
            poly[0] + poly[1] * 2 + poly[2] * 4
        });
        assert_eq!(eval_at_2.apply(&coeffs), 17);
    }
}

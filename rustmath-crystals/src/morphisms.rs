//! Crystal morphisms and isomorphisms
//!
//! This module provides structures for mappings between crystals that
//! preserve the crystal structure (morphisms), including:
//! - Crystal morphisms (ψ: B → B')
//! - Crystal isomorphisms
//! - Embeddings and projections
//! - Virtual crystals

use crate::operators::{Crystal, CrystalElement};
use crate::weight::Weight;
use std::collections::HashMap;
use std::marker::PhantomData;

/// A crystal morphism ψ: B → B'
///
/// A crystal morphism is a map that commutes with crystal operators:
/// ψ(f_i b) = f_i ψ(b) for all i and b
pub trait CrystalMorphism<C1: Crystal, C2: Crystal> {
    /// Apply the morphism to an element
    fn apply(&self, elem: &C1::Element) -> Option<C2::Element>;

    /// Check if the morphism is strict (injective and respects all structure)
    fn is_strict(&self) -> bool {
        true
    }

    /// Check if this is an isomorphism
    fn is_isomorphism(&self) -> bool {
        false
    }

    /// Get the image of the morphism
    fn image(&self, crystal: &C1) -> Vec<C2::Element> {
        crystal
            .elements()
            .iter()
            .filter_map(|elem| self.apply(elem))
            .collect()
    }

    /// Check if morphism commutes with e_i
    fn commutes_ei(&self, crystal1: &C1, crystal2: &C2, elem: &C1::Element, i: usize) -> bool {
        if let Some(mapped) = self.apply(elem) {
            let e_then_map = crystal1.e_i(elem, i).and_then(|e| self.apply(&e));
            let map_then_e = crystal2.e_i(&mapped, i);
            e_then_map == map_then_e
        } else {
            true
        }
    }

    /// Check if morphism commutes with f_i
    fn commutes_fi(&self, crystal1: &C1, crystal2: &C2, elem: &C1::Element, i: usize) -> bool {
        if let Some(mapped) = self.apply(elem) {
            let f_then_map = crystal1.f_i(elem, i).and_then(|f| self.apply(&f));
            let map_then_f = crystal2.f_i(&mapped, i);
            f_then_map == map_then_f
        } else {
            true
        }
    }
}

/// Identity morphism
pub struct IdentityMorphism<C: Crystal> {
    _phantom: PhantomData<C>,
}

impl<C: Crystal> IdentityMorphism<C> {
    /// Create a new identity morphism
    pub fn new() -> Self {
        IdentityMorphism {
            _phantom: PhantomData,
        }
    }
}

impl<C: Crystal> Default for IdentityMorphism<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Crystal> CrystalMorphism<C, C> for IdentityMorphism<C> {
    fn apply(&self, elem: &C::Element) -> Option<C::Element> {
        Some(elem.clone())
    }

    fn is_isomorphism(&self) -> bool {
        true
    }
}

/// Explicit morphism defined by a map
pub struct ExplicitMorphism<C1: Crystal, C2: Crystal> {
    /// Map from elements of C1 to elements of C2
    pub map: HashMap<C1::Element, C2::Element>,
    _phantom1: PhantomData<C1>,
    _phantom2: PhantomData<C2>,
}

impl<C1: Crystal, C2: Crystal> ExplicitMorphism<C1, C2> {
    /// Create a new explicit morphism
    pub fn new() -> Self {
        ExplicitMorphism {
            map: HashMap::new(),
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }

    /// Add a mapping
    pub fn add_mapping(&mut self, from: C1::Element, to: C2::Element) {
        self.map.insert(from, to);
    }

    /// Create from a hash map
    pub fn from_map(map: HashMap<C1::Element, C2::Element>) -> Self {
        ExplicitMorphism {
            map,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
        }
    }
}

impl<C1: Crystal, C2: Crystal> Default for ExplicitMorphism<C1, C2> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C1: Crystal, C2: Crystal> CrystalMorphism<C1, C2> for ExplicitMorphism<C1, C2> {
    fn apply(&self, elem: &C1::Element) -> Option<C2::Element> {
        self.map.get(elem).cloned()
    }
}

/// Crystal isomorphism
///
/// An isomorphism is a bijective morphism with a two-sided inverse.
pub struct CrystalIsomorphism<C1: Crystal, C2: Crystal> {
    /// Forward map
    pub forward: ExplicitMorphism<C1, C2>,
    /// Backward map
    pub backward: ExplicitMorphism<C2, C1>,
}

impl<C1: Crystal, C2: Crystal> CrystalIsomorphism<C1, C2> {
    /// Create a new isomorphism
    pub fn new(
        forward: ExplicitMorphism<C1, C2>,
        backward: ExplicitMorphism<C2, C1>,
    ) -> Self {
        CrystalIsomorphism { forward, backward }
    }

    /// Apply forward direction
    pub fn forward(&self, elem: &C1::Element) -> Option<C2::Element> {
        self.forward.apply(elem)
    }

    /// Apply backward direction
    pub fn backward(&self, elem: &C2::Element) -> Option<C1::Element> {
        self.backward.apply(elem)
    }

    /// Verify that this is actually an isomorphism
    pub fn verify(&self, crystal1: &C1, crystal2: &C2) -> bool {
        // Check that forward ∘ backward = identity on image
        for elem2 in crystal2.elements() {
            if let Some(elem1) = self.backward(&elem2) {
                if let Some(elem2_back) = self.forward(&elem1) {
                    if elem2 != elem2_back {
                        return false;
                    }
                }
            }
        }

        // Check that backward ∘ forward = identity
        for elem1 in crystal1.elements() {
            if let Some(elem2) = self.forward(&elem1) {
                if let Some(elem1_back) = self.backward(&elem2) {
                    if elem1 != elem1_back {
                        return false;
                    }
                }
            }
        }

        true
    }
}

impl<C1: Crystal, C2: Crystal> CrystalMorphism<C1, C2> for CrystalIsomorphism<C1, C2> {
    fn apply(&self, elem: &C1::Element) -> Option<C2::Element> {
        self.forward.apply(elem)
    }

    fn is_isomorphism(&self) -> bool {
        true
    }
}

/// Embedding: injective crystal morphism
pub struct CrystalEmbedding<C1: Crystal, C2: Crystal> {
    /// The underlying morphism
    pub morphism: ExplicitMorphism<C1, C2>,
}

impl<C1: Crystal, C2: Crystal> CrystalEmbedding<C1, C2> {
    /// Create a new embedding
    pub fn new(morphism: ExplicitMorphism<C1, C2>) -> Self {
        CrystalEmbedding { morphism }
    }

    /// Check if this is actually injective
    pub fn is_injective(&self) -> bool {
        let mut seen = std::collections::HashSet::new();
        for value in self.morphism.map.values() {
            if !seen.insert(value.clone()) {
                return false;
            }
        }
        true
    }
}

impl<C1: Crystal, C2: Crystal> CrystalMorphism<C1, C2> for CrystalEmbedding<C1, C2> {
    fn apply(&self, elem: &C1::Element) -> Option<C2::Element> {
        self.morphism.apply(elem)
    }

    fn is_strict(&self) -> bool {
        true
    }
}

/// Virtual crystal
///
/// A virtual crystal is a crystal with a modified Dynkin diagram indexing.
/// It's used for studying the relationship between different types.
pub struct VirtualCrystal<C: Crystal> {
    /// The underlying crystal
    pub underlying: C,
    /// Map from virtual indices to actual indices
    pub index_map: HashMap<usize, usize>,
}

impl<C: Crystal> VirtualCrystal<C> {
    /// Create a new virtual crystal
    pub fn new(underlying: C, index_map: HashMap<usize, usize>) -> Self {
        VirtualCrystal {
            underlying,
            index_map,
        }
    }

    /// Apply virtual e_i (maps to actual e_j)
    pub fn virtual_ei(&self, elem: &C::Element, i: usize) -> Option<C::Element> {
        if let Some(&j) = self.index_map.get(&i) {
            self.underlying.e_i(elem, j)
        } else {
            None
        }
    }

    /// Apply virtual f_i
    pub fn virtual_fi(&self, elem: &C::Element, i: usize) -> Option<C::Element> {
        if let Some(&j) = self.index_map.get(&i) {
            self.underlying.f_i(elem, j)
        } else {
            None
        }
    }
}

/// Composition of morphisms
///
/// Given ψ: B → B' and φ: B' → B'', construct φ ∘ ψ: B → B''
pub struct ComposedMorphism<C1, C2, C3, M1, M2>
where
    C1: Crystal,
    C2: Crystal,
    C3: Crystal,
    M1: CrystalMorphism<C1, C2>,
    M2: CrystalMorphism<C2, C3>,
{
    /// First morphism
    pub first: M1,
    /// Second morphism
    pub second: M2,
    _phantom1: PhantomData<C1>,
    _phantom2: PhantomData<C2>,
    _phantom3: PhantomData<C3>,
}

impl<C1, C2, C3, M1, M2> ComposedMorphism<C1, C2, C3, M1, M2>
where
    C1: Crystal,
    C2: Crystal,
    C3: Crystal,
    M1: CrystalMorphism<C1, C2>,
    M2: CrystalMorphism<C2, C3>,
{
    /// Create a new composed morphism
    pub fn new(first: M1, second: M2) -> Self {
        ComposedMorphism {
            first,
            second,
            _phantom1: PhantomData,
            _phantom2: PhantomData,
            _phantom3: PhantomData,
        }
    }
}

impl<C1, C2, C3, M1, M2> CrystalMorphism<C1, C3> for ComposedMorphism<C1, C2, C3, M1, M2>
where
    C1: Crystal,
    C2: Crystal,
    C3: Crystal,
    M1: CrystalMorphism<C1, C2>,
    M2: CrystalMorphism<C2, C3>,
{
    fn apply(&self, elem: &C1::Element) -> Option<C3::Element> {
        self.first.apply(elem).and_then(|e| self.second.apply(&e))
    }

    fn is_isomorphism(&self) -> bool {
        self.first.is_isomorphism() && self.second.is_isomorphism()
    }
}

/// Verify that a function defines a crystal morphism
pub fn verify_morphism<C1: Crystal, C2: Crystal, F>(
    crystal1: &C1,
    crystal2: &C2,
    func: F,
    rank: usize,
) -> bool
where
    F: Fn(&C1::Element) -> Option<C2::Element>,
{
    for elem in crystal1.elements() {
        if let Some(mapped) = func(&elem) {
            // Check e_i commutation
            for i in 0..rank {
                let e_then_map = crystal1.e_i(&elem, i).and_then(|e| func(&e));
                let map_then_e = crystal2.e_i(&mapped, i);
                if e_then_map != map_then_e {
                    return false;
                }

                // Check f_i commutation
                let f_then_map = crystal1.f_i(&elem, i).and_then(|f| func(&f));
                let map_then_f = crystal2.f_i(&mapped, i);
                if f_then_map != map_then_f {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::SimpleCrystal;
    use crate::weight::Weight;

    #[test]
    fn test_identity_morphism() {
        let crystal = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![0, 0]), Weight::new(vec![1, 0])],
        };

        let id: IdentityMorphism<SimpleCrystal> = IdentityMorphism::new();
        let elem = Weight::new(vec![0, 0]);

        assert_eq!(id.apply(&elem), Some(elem.clone()));
        assert!(id.is_isomorphism());
    }

    #[test]
    fn test_explicit_morphism() {
        let mut morphism: ExplicitMorphism<SimpleCrystal, SimpleCrystal> = ExplicitMorphism::new();
        let w1 = Weight::new(vec![1, 0]);
        let w2 = Weight::new(vec![0, 1]);

        morphism.add_mapping(w1.clone(), w2.clone());

        assert_eq!(morphism.apply(&w1), Some(w2));
        assert_eq!(morphism.apply(&Weight::new(vec![2, 0])), None);
    }

    #[test]
    fn test_crystal_isomorphism() {
        let mut forward: ExplicitMorphism<SimpleCrystal, SimpleCrystal> = ExplicitMorphism::new();
        let mut backward: ExplicitMorphism<SimpleCrystal, SimpleCrystal> = ExplicitMorphism::new();

        let w1 = Weight::new(vec![1, 0]);
        let w2 = Weight::new(vec![0, 1]);

        forward.add_mapping(w1.clone(), w2.clone());
        backward.add_mapping(w2.clone(), w1.clone());

        let iso = CrystalIsomorphism::new(forward, backward);

        assert_eq!(iso.forward(&w1), Some(w2.clone()));
        assert_eq!(iso.backward(&w2), Some(w1));
        assert!(iso.is_isomorphism());
    }

    #[test]
    fn test_virtual_crystal() {
        let crystal = SimpleCrystal {
            rank: 2,
            elements: vec![Weight::new(vec![0, 0])],
        };

        let mut index_map = HashMap::new();
        index_map.insert(0, 1); // Virtual index 0 maps to actual index 1

        let virtual_crystal = VirtualCrystal::new(crystal, index_map);

        let elem = Weight::new(vec![0, 0]);
        // Virtual e_0 should apply actual e_1
        let _ = virtual_crystal.virtual_ei(&elem, 0);
    }
}

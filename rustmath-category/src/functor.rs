//! Functors - Structure-preserving maps between categories
//!
//! A functor F: C → D maps objects and morphisms from category C to category D
//! while preserving composition and identities.

use std::marker::PhantomData;

/// A functor between categories
///
/// A functor F: C → D consists of:
/// - A mapping F: Ob(C) → Ob(D) of objects
/// - A mapping F: Hom_C(A,B) → Hom_D(F(A),F(B)) of morphisms
///
/// Such that:
/// - F(g ∘ f) = F(g) ∘ F(f) (preserves composition)
/// - F(id_A) = id_F(A) (preserves identities)
pub trait Functor {
    /// Source category object type
    type SourceObject;
    /// Target category object type
    type TargetObject;
    /// Source category morphism type
    type SourceMorphism;
    /// Target category morphism type
    type TargetMorphism;

    /// Map an object from the source category to the target category
    fn map_object(&self, obj: &Self::SourceObject) -> Self::TargetObject;

    /// Map a morphism from the source category to the target category
    fn map_morphism(&self, morph: &Self::SourceMorphism) -> Self::TargetMorphism;
}

/// The identity functor Id_C: C → C
///
/// Maps every object and morphism to itself
#[derive(Clone, Debug)]
pub struct IdentityFunctor<Obj, Morph> {
    _phantom_obj: PhantomData<Obj>,
    _phantom_morph: PhantomData<Morph>,
}

impl<Obj, Morph> IdentityFunctor<Obj, Morph> {
    /// Create a new identity functor
    pub fn new() -> Self {
        IdentityFunctor {
            _phantom_obj: PhantomData,
            _phantom_morph: PhantomData,
        }
    }
}

impl<Obj, Morph> Default for IdentityFunctor<Obj, Morph> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Obj: Clone, Morph: Clone> Functor for IdentityFunctor<Obj, Morph> {
    type SourceObject = Obj;
    type TargetObject = Obj;
    type SourceMorphism = Morph;
    type TargetMorphism = Morph;

    fn map_object(&self, obj: &Self::SourceObject) -> Self::TargetObject {
        obj.clone()
    }

    fn map_morphism(&self, morph: &Self::SourceMorphism) -> Self::TargetMorphism {
        morph.clone()
    }
}

/// A forgetful functor that "forgets" structure
///
/// For example:
/// - Grp → Set: forget group structure, remember underlying set
/// - Ring → Ab: forget multiplication, remember additive group
/// - Top → Set: forget topology, remember underlying set
#[derive(Clone, Debug)]
pub struct ForgetfulFunctor<From, To> {
    _phantom_from: PhantomData<From>,
    _phantom_to: PhantomData<To>,
}

impl<From, To> ForgetfulFunctor<From, To> {
    /// Create a new forgetful functor
    pub fn new() -> Self {
        ForgetfulFunctor {
            _phantom_from: PhantomData,
            _phantom_to: PhantomData,
        }
    }
}

impl<From, To> Default for ForgetfulFunctor<From, To> {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker trait for extracting underlying structure
pub trait HasUnderlyingStructure<U> {
    /// Extract the underlying structure
    fn underlying(&self) -> U;
}

/// Functor composition: (G ∘ F)(x) = G(F(x))
///
/// If F: C → D and G: D → E, then G ∘ F: C → E
#[derive(Clone, Debug)]
pub struct ComposedFunctor<F, G> {
    first: F,
    second: G,
}

impl<F, G> ComposedFunctor<F, G> {
    /// Create a new composed functor G ∘ F
    pub fn new(first: F, second: G) -> Self {
        ComposedFunctor { first, second }
    }
}

impl<F, G> Functor for ComposedFunctor<F, G>
where
    F: Functor,
    G: Functor<SourceObject = F::TargetObject, SourceMorphism = F::TargetMorphism>,
{
    type SourceObject = F::SourceObject;
    type TargetObject = G::TargetObject;
    type SourceMorphism = F::SourceMorphism;
    type TargetMorphism = G::TargetMorphism;

    fn map_object(&self, obj: &Self::SourceObject) -> Self::TargetObject {
        let intermediate = self.first.map_object(obj);
        self.second.map_object(&intermediate)
    }

    fn map_morphism(&self, morph: &Self::SourceMorphism) -> Self::TargetMorphism {
        let intermediate = self.first.map_morphism(morph);
        self.second.map_morphism(&intermediate)
    }
}

/// A contravariant functor F: C^op → D
///
/// Reverses the direction of morphisms:
/// f: A → B in C maps to F(f): F(B) → F(A) in D
pub trait ContravariantFunctor {
    /// Source category object type
    type SourceObject;
    /// Target category object type
    type TargetObject;
    /// Source category morphism type
    type SourceMorphism;
    /// Target category morphism type
    type TargetMorphism;

    /// Map an object (same as covariant functor)
    fn map_object(&self, obj: &Self::SourceObject) -> Self::TargetObject;

    /// Map a morphism in the reverse direction
    fn map_morphism_op(&self, morph: &Self::SourceMorphism) -> Self::TargetMorphism;
}

/// The Hom functor Hom(A, -): C → Set
///
/// Maps an object B to the set of morphisms Hom(A, B)
/// Maps a morphism f: B → C to the function Hom(A, B) → Hom(A, C)
/// given by composition with f
#[derive(Clone, Debug)]
pub struct HomFunctor<A> {
    fixed_object: A,
}

impl<A> HomFunctor<A> {
    /// Create a new Hom(A, -) functor
    pub fn new(fixed_object: A) -> Self {
        HomFunctor { fixed_object }
    }

    /// Get the fixed object A
    pub fn fixed_object(&self) -> &A {
        &self.fixed_object
    }
}

/// An endofunctor F: C → C
///
/// A functor from a category to itself
pub trait Endofunctor {
    /// Object type in the category
    type Object;
    /// Morphism type in the category
    type Morphism;

    /// Map an object to another object in the same category
    fn map_object(&self, obj: &Self::Object) -> Self::Object;

    /// Map a morphism to another morphism in the same category
    fn map_morphism(&self, morph: &Self::Morphism) -> Self::Morphism;
}

/// A functor that preserves limits
pub trait LimitPreservingFunctor: Functor {
    // Marker trait - implementors preserve limits
}

/// A functor that preserves colimits
pub trait ColimitPreservingFunctor: Functor {
    // Marker trait - implementors preserve colimits
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test types for demonstration
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestObject(String);

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestMorphism {
        source: String,
        target: String,
        name: String,
    }

    #[test]
    fn test_identity_functor_object() {
        let id_functor = IdentityFunctor::<TestObject, TestMorphism>::new();
        let obj = TestObject("A".to_string());

        let mapped = id_functor.map_object(&obj);
        assert_eq!(mapped, obj);
    }

    #[test]
    fn test_identity_functor_morphism() {
        let id_functor = IdentityFunctor::<TestObject, TestMorphism>::new();
        let morph = TestMorphism {
            source: "A".to_string(),
            target: "B".to_string(),
            name: "f".to_string(),
        };

        let mapped = id_functor.map_morphism(&morph);
        assert_eq!(mapped, morph);
    }

    // Test functor that doubles object names
    struct DoublingFunctor;

    impl Functor for DoublingFunctor {
        type SourceObject = TestObject;
        type TargetObject = TestObject;
        type SourceMorphism = TestMorphism;
        type TargetMorphism = TestMorphism;

        fn map_object(&self, obj: &Self::SourceObject) -> Self::TargetObject {
            TestObject(format!("{}_{}", obj.0, obj.0))
        }

        fn map_morphism(&self, morph: &Self::SourceMorphism) -> Self::TargetMorphism {
            TestMorphism {
                source: format!("{}_{}", morph.source, morph.source),
                target: format!("{}_{}", morph.target, morph.target),
                name: morph.name.clone(),
            }
        }
    }

    #[test]
    fn test_custom_functor() {
        let f = DoublingFunctor;
        let obj = TestObject("A".to_string());

        let mapped = f.map_object(&obj);
        assert_eq!(mapped, TestObject("A_A".to_string()));
    }

    #[test]
    fn test_functor_composition() {
        let f1 = DoublingFunctor;
        let f2 = DoublingFunctor;

        let composed = ComposedFunctor::new(f1, f2);
        let obj = TestObject("A".to_string());

        // First doubling: A → A_A
        // Second doubling: A_A → A_A_A_A
        let mapped = composed.map_object(&obj);
        assert_eq!(mapped, TestObject("A_A_A_A".to_string()));
    }

    #[test]
    fn test_hom_functor_creation() {
        let obj = TestObject("A".to_string());
        let hom_functor = HomFunctor::new(obj.clone());

        assert_eq!(hom_functor.fixed_object(), &obj);
    }

    #[test]
    fn test_forgetful_functor_creation() {
        let _functor = ForgetfulFunctor::<TestObject, String>::new();
        // Just test that it compiles and can be created
    }

    // Test endofunctor
    struct SquaringEndofunctor;

    impl Endofunctor for SquaringEndofunctor {
        type Object = TestObject;
        type Morphism = TestMorphism;

        fn map_object(&self, obj: &Self::Object) -> Self::Object {
            TestObject(format!("({})^2", obj.0))
        }

        fn map_morphism(&self, morph: &Self::Morphism) -> Self::Morphism {
            TestMorphism {
                source: format!("({})^2", morph.source),
                target: format!("({})^2", morph.target),
                name: format!("{}^2", morph.name),
            }
        }
    }

    #[test]
    fn test_endofunctor() {
        let f = SquaringEndofunctor;
        let obj = TestObject("X".to_string());

        let mapped = f.map_object(&obj);
        assert_eq!(mapped, TestObject("(X)^2".to_string()));
    }

    #[test]
    fn test_identity_functor_default() {
        let functor1 = IdentityFunctor::<TestObject, TestMorphism>::new();
        let functor2 = IdentityFunctor::<TestObject, TestMorphism>::default();

        let obj = TestObject("A".to_string());
        assert_eq!(functor1.map_object(&obj), functor2.map_object(&obj));
    }
}

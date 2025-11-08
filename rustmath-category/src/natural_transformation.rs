//! Natural transformations - Morphisms between functors
//!
//! A natural transformation η: F ⇒ G is a family of morphisms
//! that relates two functors while respecting the categorical structure.

use crate::functor::Functor;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// A natural transformation between functors
///
/// Given functors F, G: C → D, a natural transformation η: F ⇒ G
/// consists of a family of morphisms η_A: F(A) → G(A) for each object A
/// such that for every f: A → B, the naturality square commutes:
///
/// ```text
/// F(A) --η_A--> G(A)
///  |              |
/// F(f)          G(f)
///  |              |
///  v              v
/// F(B) --η_B--> G(B)
/// ```
///
/// This means: G(f) ∘ η_A = η_B ∘ F(f)
pub struct NaturalTransformation<Obj, Morph>
where
    Obj: Eq + Hash,
{
    /// Components of the natural transformation
    /// Maps each object A to the morphism η_A: F(A) → G(A)
    components: HashMap<Obj, Morph>,
}

impl<Obj, Morph> NaturalTransformation<Obj, Morph>
where
    Obj: Eq + Hash,
{
    /// Create a new natural transformation from components
    pub fn new(components: HashMap<Obj, Morph>) -> Self {
        NaturalTransformation { components }
    }

    /// Get the component at a specific object
    pub fn component_at(&self, obj: &Obj) -> Option<&Morph> {
        self.components.get(obj)
    }

    /// Get all components
    pub fn components(&self) -> &HashMap<Obj, Morph> {
        &self.components
    }

    /// Number of components
    pub fn num_components(&self) -> usize {
        self.components.len()
    }
}

impl<Obj, Morph> fmt::Display for NaturalTransformation<Obj, Morph>
where
    Obj: Eq + Hash + fmt::Display,
    Morph: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Natural Transformation with {} components:", self.num_components())?;
        for (obj, morph) in &self.components {
            writeln!(f, "  η_{}: {}", obj, morph)?;
        }
        Ok(())
    }
}

/// Vertical composition of natural transformations
///
/// Given η: F ⇒ G and θ: G ⇒ H, we can compose them to get
/// θ ∘ η: F ⇒ H with components (θ ∘ η)_A = θ_A ∘ η_A
pub fn vertical_composition<Obj, Morph, Comp>(
    eta: &NaturalTransformation<Obj, Morph>,
    theta: &NaturalTransformation<Obj, Morph>,
    compose: Comp,
) -> NaturalTransformation<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
    Morph: Clone,
    Comp: Fn(&Morph, &Morph) -> Morph,
{
    let mut composed_components = HashMap::new();

    for (obj, eta_component) in eta.components() {
        if let Some(theta_component) = theta.component_at(obj) {
            // (θ ∘ η)_A = θ_A ∘ η_A
            let composed = compose(theta_component, eta_component);
            composed_components.insert(obj.clone(), composed);
        }
    }

    NaturalTransformation::new(composed_components)
}

/// Horizontal composition of natural transformations
///
/// Given η: F ⇒ G (F,G: C → D) and θ: H ⇒ K (H,K: D → E),
/// we get θ * η: H∘F ⇒ K∘G with components (θ * η)_A = θ_G(A) ∘ H(η_A) = K(η_A) ∘ θ_F(A)
#[allow(dead_code)]
pub struct HorizontalComposition;

/// The identity natural transformation Id_F: F ⇒ F
///
/// Has components (Id_F)_A = id_F(A) for each object A
pub struct IdentityNaturalTransformation<Obj, Morph>
where
    Obj: Eq + Hash,
{
    components: HashMap<Obj, Morph>,
}

impl<Obj, Morph> IdentityNaturalTransformation<Obj, Morph>
where
    Obj: Eq + Hash + Clone,
    Morph: Clone,
{
    /// Create the identity natural transformation for a functor
    ///
    /// Requires providing the identity morphisms for each object
    pub fn new(identity_morphisms: HashMap<Obj, Morph>) -> Self {
        IdentityNaturalTransformation {
            components: identity_morphisms,
        }
    }

    /// Convert to a general natural transformation
    pub fn to_natural_transformation(self) -> NaturalTransformation<Obj, Morph> {
        NaturalTransformation::new(self.components)
    }
}

/// A natural isomorphism is a natural transformation where each component is an isomorphism
///
/// η: F ⇒ G is a natural isomorphism if each η_A is an isomorphism
pub struct NaturalIsomorphism<Obj, Morph>
where
    Obj: Eq + Hash,
{
    transformation: NaturalTransformation<Obj, Morph>,
}

impl<Obj, Morph> NaturalIsomorphism<Obj, Morph>
where
    Obj: Eq + Hash,
{
    /// Create a natural isomorphism
    ///
    /// Note: This doesn't verify that components are actually isomorphisms
    pub fn new(components: HashMap<Obj, Morph>) -> Self {
        NaturalIsomorphism {
            transformation: NaturalTransformation::new(components),
        }
    }

    /// Get the underlying natural transformation
    pub fn as_transformation(&self) -> &NaturalTransformation<Obj, Morph> {
        &self.transformation
    }

    /// Get the component at a specific object
    pub fn component_at(&self, obj: &Obj) -> Option<&Morph> {
        self.transformation.component_at(obj)
    }
}

/// Functor category [C, D]
///
/// Objects are functors F: C → D
/// Morphisms are natural transformations between functors
pub struct FunctorCategory<F> {
    /// Functors in this category
    functors: Vec<F>,
}

impl<F> FunctorCategory<F> {
    /// Create a new functor category
    pub fn new(functors: Vec<F>) -> Self {
        FunctorCategory { functors }
    }

    /// Get all functors
    pub fn functors(&self) -> &[F] {
        &self.functors
    }

    /// Number of functors
    pub fn num_functors(&self) -> usize {
        self.functors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestObject(String);

    impl fmt::Display for TestObject {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestMorphism {
        source: String,
        target: String,
        name: String,
    }

    impl fmt::Display for TestMorphism {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}: {} → {}", self.name, self.source, self.target)
        }
    }

    #[test]
    fn test_natural_transformation_creation() {
        let mut components = HashMap::new();
        components.insert(
            TestObject("A".to_string()),
            TestMorphism {
                source: "F(A)".to_string(),
                target: "G(A)".to_string(),
                name: "η_A".to_string(),
            },
        );

        let nat_trans = NaturalTransformation::new(components);
        assert_eq!(nat_trans.num_components(), 1);
    }

    #[test]
    fn test_component_at() {
        let mut components = HashMap::new();
        let obj_a = TestObject("A".to_string());
        let morph = TestMorphism {
            source: "F(A)".to_string(),
            target: "G(A)".to_string(),
            name: "η_A".to_string(),
        };
        components.insert(obj_a.clone(), morph.clone());

        let nat_trans = NaturalTransformation::new(components);

        assert_eq!(nat_trans.component_at(&obj_a), Some(&morph));
        assert_eq!(nat_trans.component_at(&TestObject("B".to_string())), None);
    }

    #[test]
    fn test_vertical_composition() {
        // Create η: F ⇒ G
        let mut eta_components = HashMap::new();
        let obj = TestObject("A".to_string());
        eta_components.insert(
            obj.clone(),
            TestMorphism {
                source: "F(A)".to_string(),
                target: "G(A)".to_string(),
                name: "η_A".to_string(),
            },
        );
        let eta = NaturalTransformation::new(eta_components);

        // Create θ: G ⇒ H
        let mut theta_components = HashMap::new();
        theta_components.insert(
            obj.clone(),
            TestMorphism {
                source: "G(A)".to_string(),
                target: "H(A)".to_string(),
                name: "θ_A".to_string(),
            },
        );
        let theta = NaturalTransformation::new(theta_components);

        // Compose them
        let composed = vertical_composition(&eta, &theta, |theta_comp, eta_comp| TestMorphism {
            source: eta_comp.source.clone(),
            target: theta_comp.target.clone(),
            name: format!("{} ∘ {}", theta_comp.name, eta_comp.name),
        });

        assert_eq!(composed.num_components(), 1);
        let comp = composed.component_at(&obj).unwrap();
        assert_eq!(comp.source, "F(A)");
        assert_eq!(comp.target, "H(A)");
        assert_eq!(comp.name, "θ_A ∘ η_A");
    }

    #[test]
    fn test_identity_natural_transformation() {
        let mut identity_morphisms = HashMap::new();
        let obj = TestObject("A".to_string());
        identity_morphisms.insert(
            obj.clone(),
            TestMorphism {
                source: "F(A)".to_string(),
                target: "F(A)".to_string(),
                name: "id_F(A)".to_string(),
            },
        );

        let id_trans = IdentityNaturalTransformation::new(identity_morphisms);
        let nat_trans = id_trans.to_natural_transformation();

        assert_eq!(nat_trans.num_components(), 1);
        let comp = nat_trans.component_at(&obj).unwrap();
        assert_eq!(comp.source, comp.target);
    }

    #[test]
    fn test_natural_isomorphism() {
        let mut components = HashMap::new();
        let obj = TestObject("A".to_string());
        components.insert(
            obj.clone(),
            TestMorphism {
                source: "F(A)".to_string(),
                target: "G(A)".to_string(),
                name: "η_A".to_string(),
            },
        );

        let nat_iso = NaturalIsomorphism::new(components);
        assert!(nat_iso.component_at(&obj).is_some());
    }

    #[test]
    fn test_functor_category() {
        // Using () as placeholder functor type for this test
        let functors = vec![(), (), ()];
        let cat = FunctorCategory::new(functors);

        assert_eq!(cat.num_functors(), 3);
    }

    #[test]
    fn test_display_natural_transformation() {
        let mut components = HashMap::new();
        components.insert(
            TestObject("A".to_string()),
            TestMorphism {
                source: "F(A)".to_string(),
                target: "G(A)".to_string(),
                name: "η_A".to_string(),
            },
        );

        let nat_trans = NaturalTransformation::new(components);
        let display = format!("{}", nat_trans);

        assert!(display.contains("Natural Transformation"));
        assert!(display.contains("η_A"));
    }

    #[test]
    fn test_empty_natural_transformation() {
        let components: HashMap<TestObject, TestMorphism> = HashMap::new();
        let nat_trans = NaturalTransformation::new(components);

        assert_eq!(nat_trans.num_components(), 0);
    }

    #[test]
    fn test_vertical_composition_partial_overlap() {
        // η defined on A and B
        let mut eta_components = HashMap::new();
        eta_components.insert(
            TestObject("A".to_string()),
            TestMorphism {
                source: "F(A)".to_string(),
                target: "G(A)".to_string(),
                name: "η_A".to_string(),
            },
        );
        eta_components.insert(
            TestObject("B".to_string()),
            TestMorphism {
                source: "F(B)".to_string(),
                target: "G(B)".to_string(),
                name: "η_B".to_string(),
            },
        );
        let eta = NaturalTransformation::new(eta_components);

        // θ only defined on A
        let mut theta_components = HashMap::new();
        theta_components.insert(
            TestObject("A".to_string()),
            TestMorphism {
                source: "G(A)".to_string(),
                target: "H(A)".to_string(),
                name: "θ_A".to_string(),
            },
        );
        let theta = NaturalTransformation::new(theta_components);

        // Composition should only have component at A
        let composed = vertical_composition(&eta, &theta, |theta_comp, eta_comp| TestMorphism {
            source: eta_comp.source.clone(),
            target: theta_comp.target.clone(),
            name: format!("{} ∘ {}", theta_comp.name, eta_comp.name),
        });

        assert_eq!(composed.num_components(), 1);
        assert!(composed.component_at(&TestObject("A".to_string())).is_some());
        assert!(composed.component_at(&TestObject("B".to_string())).is_none());
    }
}

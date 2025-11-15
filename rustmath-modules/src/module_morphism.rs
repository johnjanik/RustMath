//! Module Morphisms with Linearity
//!
//! Module morphisms (homomorphisms) that are defined by specifying where
//! basis elements map, then extending by linearity.
//!
//! Corresponds to sage.categories.modules_with_basis.ModuleMorphismByLinearity

use rustmath_core::Ring;
use crate::combinatorial_free_module::CombinatorialFreeModuleElement;
use std::hash::Hash;
use std::marker::PhantomData;

/// A module morphism defined by linearity
///
/// Given a function f: I → Y where I is the basis index set and Y is the codomain,
/// this constructs a module morphism g: X → Y where X is the free module with basis I.
///
/// The morphism is defined by: g(Σᵢ λᵢ bᵢ) = Σᵢ λᵢ f(i)
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `I` - The basis index set of the domain
/// * `J` - The basis index set of the codomain
/// * `F` - The function type from I to codomain elements
pub struct ModuleMorphismByLinearity<R, I, J, F>
where
    R: Ring,
    I: Hash + Eq + Clone,
    J: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, J>,
{
    /// The function that defines where each basis element maps
    on_basis: F,
    /// Phantom data for type parameters
    _phantom: PhantomData<(R, I, J)>,
}

impl<R, I, J, F> ModuleMorphismByLinearity<R, I, J, F>
where
    R: Ring,
    I: Hash + Eq + Clone,
    J: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, J>,
{
    /// Create a new module morphism defined by where basis elements map
    ///
    /// # Arguments
    ///
    /// * `on_basis` - A function that takes a basis index and returns its image
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_modules::module_morphism::ModuleMorphismByLinearity;
    /// use rustmath_modules::CombinatorialFreeModuleElement;
    /// use rustmath_integers::Integer;
    ///
    /// // Define a morphism that maps basis element i to 2*basis element i
    /// let morphism = ModuleMorphismByLinearity::new(|i: &usize| {
    ///     CombinatorialFreeModuleElement::monomial(*i, Integer::from(2))
    /// });
    /// ```
    pub fn new(on_basis: F) -> Self {
        ModuleMorphismByLinearity {
            on_basis,
            _phantom: PhantomData,
        }
    }

    /// Apply the morphism to a module element
    ///
    /// Extends the basis mapping linearly: g(Σᵢ λᵢ bᵢ) = Σᵢ λᵢ f(i)
    ///
    /// # Arguments
    ///
    /// * `element` - The element to apply the morphism to
    ///
    /// # Returns
    ///
    /// The image of the element under this morphism
    pub fn apply(&self, element: &CombinatorialFreeModuleElement<R, I>)
        -> CombinatorialFreeModuleElement<R, J>
    {
        let mut result = CombinatorialFreeModuleElement::zero();

        // Apply linearity: for each term λᵢ * bᵢ, compute λᵢ * f(i) and sum
        for (index, coeff) in element.iter() {
            // Get the image of this basis element
            let image = (self.on_basis)(index);

            // Scale by the coefficient
            let scaled_image = image.scalar_mul(coeff);

            // Add to the result
            result = result + scaled_image;
        }

        result
    }

    /// Get the underlying function that maps basis elements
    pub fn on_basis_function(&self) -> &F {
        &self.on_basis
    }
}

/// Compose two module morphisms
///
/// Given morphisms f: X → Y and g: Y → Z, computes g ∘ f: X → Z
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `I` - Basis indices of domain
/// * `J` - Basis indices of intermediate space
/// * `K` - Basis indices of codomain
pub fn compose_morphisms<'a, R, I, J, K, F, G>(
    f: &'a ModuleMorphismByLinearity<R, I, J, F>,
    g: &'a ModuleMorphismByLinearity<R, J, K, G>,
) -> impl Fn(&CombinatorialFreeModuleElement<R, I>) -> CombinatorialFreeModuleElement<R, K> + 'a
where
    R: Ring,
    I: Hash + Eq + Clone,
    J: Hash + Eq + Clone,
    K: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, J>,
    G: Fn(&J) -> CombinatorialFreeModuleElement<R, K>,
{
    move |element| {
        // First apply f, then apply g to the result
        let intermediate = f.apply(element);
        g.apply(&intermediate)
    }
}

/// A module endomorphism (morphism from a module to itself)
///
/// Specialized version of ModuleMorphismByLinearity for the case where
/// domain and codomain have the same basis index set.
pub type ModuleEndomorphism<R, I, F> = ModuleMorphismByLinearity<R, I, I, F>;

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_morphism_creation() {
        // Create a morphism that doubles each basis element
        let morphism = ModuleMorphismByLinearity::new(|i: &usize| {
            CombinatorialFreeModuleElement::monomial(*i, Integer::from(2))
        });

        // Test on a single basis element
        let b0: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::from_basis_index(0);

        let result = morphism.apply(&b0);

        // Should map to 2*b0
        assert_eq!(result.coefficient(&0), Integer::from(2));
    }

    #[test]
    fn test_morphism_linearity() {
        // Create a morphism that maps i to i+1
        let shift = ModuleMorphismByLinearity::new(|i: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(i + 1)
        });

        // Create element 3*b0 + 5*b1
        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (0, Integer::from(3)),
                (1, Integer::from(5)),
            ]);

        let result = shift.apply(&elem);

        // Should map to 3*b1 + 5*b2
        assert_eq!(result.coefficient(&1), Integer::from(3));
        assert_eq!(result.coefficient(&2), Integer::from(5));
        assert_eq!(result.coefficient(&0), Integer::from(0));
    }

    #[test]
    fn test_morphism_on_zero() {
        let morphism = ModuleMorphismByLinearity::new(|i: &usize| {
            CombinatorialFreeModuleElement::monomial(*i, Integer::from(2))
        });

        let zero: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::zero();

        let result = morphism.apply(&zero);
        assert!(result.is_zero());
    }

    #[test]
    fn test_morphism_to_different_basis() {
        // Map from integers to strings
        let morphism = ModuleMorphismByLinearity::new(|i: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(format!("b{}", i))
        });

        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::monomial(5, Integer::from(7));

        let result: CombinatorialFreeModuleElement<Integer, String> = morphism.apply(&elem);

        assert_eq!(result.coefficient(&"b5".to_string()), Integer::from(7));
    }

    #[test]
    fn test_endomorphism() {
        // Create an endomorphism that scales by 3
        let scale: ModuleEndomorphism<Integer, usize, _> =
            ModuleMorphismByLinearity::new(|i: &usize| {
                CombinatorialFreeModuleElement::monomial(*i, Integer::from(3))
            });

        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::monomial(2, Integer::from(5));

        let result = scale.apply(&elem);

        // 5*b2 should map to 15*b2
        assert_eq!(result.coefficient(&2), Integer::from(15));
    }

    #[test]
    fn test_zero_morphism() {
        // Create the zero morphism (maps everything to zero)
        let zero_morphism = ModuleMorphismByLinearity::new(|_i: &usize| {
            CombinatorialFreeModuleElement::<Integer, usize>::zero()
        });

        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (0, Integer::from(1)),
                (1, Integer::from(2)),
                (2, Integer::from(3)),
            ]);

        let result = zero_morphism.apply(&elem);
        assert!(result.is_zero());
    }
}

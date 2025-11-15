//! Algebra Morphisms
//!
//! Morphisms (homomorphisms) between algebras that respect both the
//! module structure and the multiplication operation.
//!
//! Corresponds to sage.categories.algebras.Algebras.Homomorphisms

use rustmath_core::Ring;
use rustmath_modules::{CombinatorialFreeModuleElement, ModuleMorphismByLinearity};
use std::hash::Hash;
use std::marker::PhantomData;

/// An algebra morphism between algebras with basis
///
/// An algebra morphism f: A → B must satisfy:
/// - f(a + b) = f(a) + f(b) (additive)
/// - f(λa) = λf(a) (scalar multiplication)
/// - f(ab) = f(a)f(b) (multiplicative)
/// - f(1_A) = 1_B (preserves identity)
///
/// When algebras have a basis, we can define the morphism by specifying
/// where each basis element maps, then extending by linearity.
///
/// # Type Parameters
///
/// * `R` - The base ring
/// * `I` - Basis index set of the domain algebra
/// * `J` - Basis index set of the codomain algebra
/// * `F` - The function type mapping basis indices
///
/// # Examples
///
/// ```
/// use rustmath_algebras::algebra_morphism::AlgebraMorphism;
/// use rustmath_modules::CombinatorialFreeModuleElement;
/// use rustmath_integers::Integer;
///
/// // Define a morphism that maps basis element i to 2*basis element i
/// let morphism = AlgebraMorphism::from_on_basis(|i: &usize| {
///     CombinatorialFreeModuleElement::monomial(*i, Integer::from(2))
/// });
/// ```
pub struct AlgebraMorphism<R, I, J, F>
where
    R: Ring,
    I: Hash + Eq + Clone,
    J: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, J>,
{
    /// The underlying module morphism
    module_morphism: ModuleMorphismByLinearity<R, I, J, F>,
    /// Phantom data
    _phantom: PhantomData<(R, I, J)>,
}

impl<R, I, J, F> AlgebraMorphism<R, I, J, F>
where
    R: Ring,
    I: Hash + Eq + Clone,
    J: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, J>,
{
    /// Create an algebra morphism from a function on basis elements
    ///
    /// # Arguments
    ///
    /// * `on_basis` - Function specifying where each basis element maps
    ///
    /// # Note
    ///
    /// The caller must ensure this function defines a valid algebra morphism
    /// (respects multiplication and identity). This is not automatically verified.
    pub fn from_on_basis(on_basis: F) -> Self {
        AlgebraMorphism {
            module_morphism: ModuleMorphismByLinearity::new(on_basis),
            _phantom: PhantomData,
        }
    }

    /// Apply the morphism to an algebra element
    ///
    /// Extends the basis mapping linearly to all elements
    pub fn apply(&self, element: &CombinatorialFreeModuleElement<R, I>)
        -> CombinatorialFreeModuleElement<R, J>
    {
        self.module_morphism.apply(element)
    }

    /// Get access to the underlying module morphism
    pub fn as_module_morphism(&self) -> &ModuleMorphismByLinearity<R, I, J, F> {
        &self.module_morphism
    }
}

/// An algebra endomorphism (morphism from an algebra to itself)
pub type AlgebraEndomorphism<R, I, F> = AlgebraMorphism<R, I, I, F>;

/// An automorphism of an algebra (invertible endomorphism)
///
/// This represents a bijective algebra morphism from an algebra to itself.
/// Automorphisms form a group under composition.
pub struct AlgebraAutomorphism<R, I, F, G>
where
    R: Ring,
    I: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, I>,
    G: Fn(&I) -> CombinatorialFreeModuleElement<R, I>,
{
    /// The forward morphism
    forward: AlgebraEndomorphism<R, I, F>,
    /// The inverse morphism
    inverse: AlgebraEndomorphism<R, I, G>,
    /// Phantom data
    _phantom: PhantomData<R>,
}

impl<R, I, F, G> AlgebraAutomorphism<R, I, F, G>
where
    R: Ring,
    I: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, I>,
    G: Fn(&I) -> CombinatorialFreeModuleElement<R, I>,
{
    /// Create an automorphism from forward and inverse functions
    ///
    /// # Arguments
    ///
    /// * `forward` - Function defining the automorphism
    /// * `inverse` - Function defining the inverse automorphism
    ///
    /// # Note
    ///
    /// The caller must ensure that `forward` and `inverse` are actually inverses.
    /// This is not automatically verified.
    pub fn new(forward: F, inverse: G) -> Self {
        AlgebraAutomorphism {
            forward: AlgebraMorphism::from_on_basis(forward),
            inverse: AlgebraMorphism::from_on_basis(inverse),
            _phantom: PhantomData,
        }
    }

    /// Apply the automorphism to an element
    pub fn apply(&self, element: &CombinatorialFreeModuleElement<R, I>)
        -> CombinatorialFreeModuleElement<R, I>
    {
        self.forward.apply(element)
    }

    /// Apply the inverse automorphism to an element
    pub fn apply_inverse(&self, element: &CombinatorialFreeModuleElement<R, I>)
        -> CombinatorialFreeModuleElement<R, I>
    {
        self.inverse.apply(element)
    }

    /// Get the forward morphism
    pub fn forward_morphism(&self) -> &AlgebraEndomorphism<R, I, F> {
        &self.forward
    }

    /// Get the inverse morphism
    pub fn inverse_morphism(&self) -> &AlgebraEndomorphism<R, I, G> {
        &self.inverse
    }
}

/// Helper function to verify a morphism respects multiplication (for testing)
///
/// Checks that f(a*b) = f(a)*f(b) for given elements
pub fn verify_multiplicative<R, I, J, F, Mult1, Mult2>(
    morphism: &AlgebraMorphism<R, I, J, F>,
    a: &CombinatorialFreeModuleElement<R, I>,
    b: &CombinatorialFreeModuleElement<R, I>,
    domain_mult: Mult1,
    codomain_mult: Mult2,
) -> bool
where
    R: Ring,
    I: Hash + Eq + Clone,
    J: Hash + Eq + Clone,
    F: Fn(&I) -> CombinatorialFreeModuleElement<R, J>,
    Mult1: Fn(&CombinatorialFreeModuleElement<R, I>, &CombinatorialFreeModuleElement<R, I>)
        -> CombinatorialFreeModuleElement<R, I>,
    Mult2: Fn(&CombinatorialFreeModuleElement<R, J>, &CombinatorialFreeModuleElement<R, J>)
        -> CombinatorialFreeModuleElement<R, J>,
{
    // Compute f(a*b)
    let ab = domain_mult(a, b);
    let f_ab = morphism.apply(&ab);

    // Compute f(a)*f(b)
    let fa = morphism.apply(a);
    let fb = morphism.apply(b);
    let fa_fb = codomain_mult(&fa, &fb);

    // Check equality
    f_ab == fa_fb
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_algebra_morphism_creation() {
        // Create a morphism that doubles each basis element
        let morphism = AlgebraMorphism::from_on_basis(|i: &usize| {
            CombinatorialFreeModuleElement::monomial(*i, Integer::from(2))
        });

        let b0: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::from_basis_index(0);

        let result = morphism.apply(&b0);
        assert_eq!(result.coefficient(&0), Integer::from(2));
    }

    #[test]
    fn test_algebra_morphism_linearity() {
        // Morphism that shifts indices by 1
        let shift = AlgebraMorphism::from_on_basis(|i: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(i + 1)
        });

        // Test on 3*b0 + 5*b1
        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (0, Integer::from(3)),
                (1, Integer::from(5)),
            ]);

        let result = shift.apply(&elem);

        // Should map to 3*b1 + 5*b2
        assert_eq!(result.coefficient(&1), Integer::from(3));
        assert_eq!(result.coefficient(&2), Integer::from(5));
    }

    #[test]
    fn test_algebra_endomorphism() {
        // Identity endomorphism
        let identity: AlgebraEndomorphism<Integer, usize, _> =
            AlgebraMorphism::from_on_basis(|i: &usize| {
                CombinatorialFreeModuleElement::from_basis_index(*i)
            });

        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::monomial(5, Integer::from(7));

        let result = identity.apply(&elem);
        assert_eq!(result, elem);
    }

    #[test]
    fn test_algebra_automorphism() {
        // Simple automorphism: swap basis 0 and 1
        let auto = AlgebraAutomorphism::new(
            |i: &usize| {
                let new_index = match *i {
                    0 => 1,
                    1 => 0,
                    n => n,
                };
                CombinatorialFreeModuleElement::from_basis_index(new_index)
            },
            |i: &usize| {
                let new_index = match *i {
                    0 => 1,
                    1 => 0,
                    n => n,
                };
                CombinatorialFreeModuleElement::from_basis_index(new_index)
            },
        );

        let b0: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::from_basis_index(0);

        // Apply automorphism
        let result = auto.apply(&b0);
        assert_eq!(result.coefficient(&1), Integer::from(1));
        assert_eq!(result.coefficient(&0), Integer::from(0));

        // Apply inverse - should get back to b0
        let back = auto.apply_inverse(&result);
        assert_eq!(back, b0);
    }

    #[test]
    fn test_verify_multiplicative() {
        // Identity morphism (always respects multiplication)
        let identity = AlgebraMorphism::from_on_basis(|i: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(*i)
        });

        let b0: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::from_basis_index(0);
        let b1: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::from_basis_index(1);

        // Define a simple multiplication (just adds indices for testing)
        let mult = |a: &CombinatorialFreeModuleElement<Integer, usize>,
                    b: &CombinatorialFreeModuleElement<Integer, usize>| {
            // Simplified: just return b for testing
            b.clone()
        };

        // Verify identity morphism respects this multiplication
        assert!(verify_multiplicative(&identity, &b0, &b1, &mult, &mult));
    }
}

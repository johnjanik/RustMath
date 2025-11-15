//! Associated Graded Algebra
//!
//! Given a filtered algebra A with filtration (F_i), construct the associated
//! graded algebra gr(A) = ⊕ᵢ Gᵢ where Gᵢ = Fᵢ / Σⱼ₍ⱼ<ᵢ₎ Fⱼ.
//!
//! For a filtered algebra with a compatible graded basis, there is an R-module
//! isomorphism from A to gr(A). The multiplication in gr(A) is derived from A
//! by taking only the leading terms (those at the expected total degree).
//!
//! Corresponds to sage.algebras.associated_graded

use rustmath_core::Ring;
use rustmath_modules::CombinatorialFreeModuleElement;
use std::hash::Hash;
use std::marker::PhantomData;

/// Associated graded algebra constructed from a filtered algebra
///
/// Given a filtered algebra A, this constructs the associated graded algebra
/// gr(A) whose i-th graded component is Gᵢ = Fᵢ / Σⱼ₍ⱼ<ᵢ₎ Fⱼ.
///
/// When A has a compatible graded basis structure, there is an R-module
/// isomorphism between A and gr(A).
///
/// # Type Parameters
///
/// * `R` - The base ring
/// * `I` - The basis index set
/// * `DegreeFunc` - Function type that computes degree of basis elements
/// * `ProductFunc` - Function type that computes products in the underlying algebra
///
/// # Examples
///
/// ```
/// use rustmath_algebras::associated_graded::AssociatedGradedAlgebra;
/// use rustmath_modules::CombinatorialFreeModuleElement;
/// use rustmath_integers::Integer;
///
/// // Define a degree function
/// let degree_fn = |i: &usize| *i;
///
/// // Define a product function for the underlying algebra
/// let product_fn = |i: &usize, j: &usize| {
///     CombinatorialFreeModuleElement::from_basis_index(i + j)
/// };
///
/// // Create the associated graded algebra
/// let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);
/// ```
pub struct AssociatedGradedAlgebra<R, I, DegreeFunc, ProductFunc>
where
    R: Ring,
    I: Hash + Eq + Clone,
    DegreeFunc: Fn(&I) -> usize,
    ProductFunc: Fn(&I, &I) -> CombinatorialFreeModuleElement<R, I>,
{
    /// Function that computes the degree of a basis element in the underlying algebra
    degree_on_basis: DegreeFunc,
    /// Function that computes products on basis elements in the underlying algebra
    underlying_product: ProductFunc,
    /// Phantom data
    _phantom: PhantomData<(R, I)>,
}

impl<R, I, DegreeFunc, ProductFunc> AssociatedGradedAlgebra<R, I, DegreeFunc, ProductFunc>
where
    R: Ring,
    I: Hash + Eq + Clone,
    DegreeFunc: Fn(&I) -> usize,
    ProductFunc: Fn(&I, &I) -> CombinatorialFreeModuleElement<R, I>,
{
    /// Create a new associated graded algebra
    ///
    /// # Arguments
    ///
    /// * `degree_on_basis` - Function that returns the degree of each basis element
    /// * `underlying_product` - Function that computes products in the underlying algebra
    ///
    /// # Note
    ///
    /// The degree function must be compatible with the product function for
    /// the graded structure to be well-defined.
    pub fn new(degree_on_basis: DegreeFunc, underlying_product: ProductFunc) -> Self {
        AssociatedGradedAlgebra {
            degree_on_basis,
            underlying_product,
            _phantom: PhantomData,
        }
    }

    /// Get the degree of a basis element
    ///
    /// # Arguments
    ///
    /// * `index` - The basis index
    ///
    /// # Returns
    ///
    /// The degree of the basis element
    pub fn degree_of_basis(&self, index: &I) -> usize {
        (self.degree_on_basis)(index)
    }

    /// Compute the product of two basis elements in the associated graded algebra
    ///
    /// The product in gr(A) is obtained by:
    /// 1. Computing the product in the underlying algebra
    /// 2. Keeping only terms whose degree equals deg(x) + deg(y)
    ///
    /// This implements the projection p_{i+j}(u'v') where u' and v' are
    /// representatives of degree i and j respectively.
    ///
    /// # Arguments
    ///
    /// * `left` - First basis index
    /// * `right` - Second basis index
    ///
    /// # Returns
    ///
    /// The product as a linear combination of basis elements
    pub fn product_on_basis(
        &self,
        left: &I,
        right: &I,
    ) -> CombinatorialFreeModuleElement<R, I>
    where
        R: Clone,
    {
        // Compute product in underlying algebra
        let underlying_product = (self.underlying_product)(left, right);

        // Expected degree of result
        let expected_degree = self.degree_of_basis(left) + self.degree_of_basis(right);

        // Filter to keep only terms at the expected degree
        let mut result = CombinatorialFreeModuleElement::zero();

        for (index, coeff) in underlying_product.iter() {
            if self.degree_of_basis(index) == expected_degree {
                let term = CombinatorialFreeModuleElement::monomial(index.clone(), coeff.clone());
                result = result + term;
            }
        }

        result
    }

    /// Compute the product of two elements in the associated graded algebra
    ///
    /// # Arguments
    ///
    /// * `left` - First element
    /// * `right` - Second element
    ///
    /// # Returns
    ///
    /// The product element
    pub fn product(
        &self,
        left: &CombinatorialFreeModuleElement<R, I>,
        right: &CombinatorialFreeModuleElement<R, I>,
    ) -> CombinatorialFreeModuleElement<R, I>
    where
        R: Clone,
    {
        let mut result = CombinatorialFreeModuleElement::zero();

        for (left_index, left_coeff) in left.iter() {
            for (right_index, right_coeff) in right.iter() {
                let product_basis = self.product_on_basis(left_index, right_index);
                let coeff = left_coeff.clone() * right_coeff.clone();
                let term = product_basis.scalar_mul(&coeff);
                result = result + term;
            }
        }

        result
    }

    /// Get the degree of an element
    ///
    /// For a homogeneous element (supported on a single graded component),
    /// returns the degree. For non-homogeneous elements, this returns the
    /// maximum degree of the support.
    ///
    /// # Arguments
    ///
    /// * `element` - The element to compute degree of
    ///
    /// # Returns
    ///
    /// The degree (or maximum degree for non-homogeneous elements)
    pub fn degree_of_element(&self, element: &CombinatorialFreeModuleElement<R, I>) -> Option<usize> {
        let mut degrees = Vec::new();

        for (index, _coeff) in element.iter() {
            degrees.push(self.degree_of_basis(index));
        }

        degrees.into_iter().max()
    }

    /// Check if an element is homogeneous (lives in a single graded component)
    ///
    /// # Arguments
    ///
    /// * `element` - The element to check
    ///
    /// # Returns
    ///
    /// `true` if the element is homogeneous, `false` otherwise
    pub fn is_homogeneous(&self, element: &CombinatorialFreeModuleElement<R, I>) -> bool {
        let mut degrees = Vec::new();

        for (index, _coeff) in element.iter() {
            let deg = self.degree_of_basis(index);
            if !degrees.is_empty() && degrees[0] != deg {
                return false;
            }
            degrees.push(deg);
        }

        true
    }

    /// Extract the homogeneous component of a given degree
    ///
    /// # Arguments
    ///
    /// * `element` - The element to extract from
    /// * `degree` - The degree to extract
    ///
    /// # Returns
    ///
    /// The homogeneous component of the given degree
    pub fn homogeneous_component(
        &self,
        element: &CombinatorialFreeModuleElement<R, I>,
        degree: usize,
    ) -> CombinatorialFreeModuleElement<R, I>
    where
        R: Clone,
    {
        let mut result = CombinatorialFreeModuleElement::zero();

        for (index, coeff) in element.iter() {
            if self.degree_of_basis(index) == degree {
                let term = CombinatorialFreeModuleElement::monomial(index.clone(), coeff.clone());
                result = result + term;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_associated_graded_creation() {
        // Simple degree function: each basis element's degree is its index
        let degree_fn = |i: &usize| *i;

        // Simple product: multiply indices and create basis element
        let product_fn = |i: &usize, j: &usize| {
            CombinatorialFreeModuleElement::<Integer, usize>::from_basis_index(i + j)
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        assert_eq!(gr_algebra.degree_of_basis(&3), 3);
        assert_eq!(gr_algebra.degree_of_basis(&5), 5);
    }

    #[test]
    fn test_product_on_basis() {
        // Degree is the index
        let degree_fn = |i: &usize| *i;

        // Product that preserves degree
        let product_fn = |i: &usize, j: &usize| {
            // Result has degree i + j
            CombinatorialFreeModuleElement::<Integer, usize>::from_basis_index(i + j)
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        // Multiply basis elements of degree 2 and 3
        let product = gr_algebra.product_on_basis(&2, &3);

        // Should get basis element of degree 5
        let expected_idx = 5;
        assert_eq!(product.coefficient(&expected_idx), Integer::from(1));
    }

    #[test]
    fn test_product_filtering() {
        // Degree is the index
        let degree_fn = |i: &usize| *i;

        // Product that generates multiple terms
        let product_fn = |i: &usize, j: &usize| {
            // Generate terms of different degrees (only one should survive)
            let expected = i + j;
            let mut result = CombinatorialFreeModuleElement::from_basis_index(expected);
            // Add a term with wrong degree (should be filtered out)
            let wrong_degree_term = CombinatorialFreeModuleElement::monomial(
                expected + 1,
                Integer::from(2),
            );
            result = result + wrong_degree_term;
            result
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        // Multiply basis elements of degree 1 and 2
        let product = gr_algebra.product_on_basis(&1, &2);

        // Should only have the term of degree 3, not degree 4
        assert_eq!(product.coefficient(&3), Integer::from(1));
        assert_eq!(product.coefficient(&4), Integer::from(0)); // Filtered out
    }

    #[test]
    fn test_homogeneous_check() {
        let degree_fn = |i: &usize| *i;
        let product_fn = |i: &usize, j: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(i + j)
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        // Homogeneous element (all terms have degree 3)
        let homogeneous: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::monomial(3, Integer::from(1));

        assert!(gr_algebra.is_homogeneous(&homogeneous));

        // Non-homogeneous element (terms of degree 2 and 3)
        let non_homogeneous: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (2, Integer::from(1)),
                (3, Integer::from(1)),
            ]);

        assert!(!gr_algebra.is_homogeneous(&non_homogeneous));
    }

    #[test]
    fn test_homogeneous_component_extraction() {
        let degree_fn = |i: &usize| *i;
        let product_fn = |i: &usize, j: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(i + j)
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        // Mixed element with degrees 2, 3, 3
        let element: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (2, Integer::from(5)),
                (3, Integer::from(7)),
            ]);

        // Extract degree 3 component
        let deg3_component = gr_algebra.homogeneous_component(&element, 3);
        assert_eq!(deg3_component.coefficient(&3), Integer::from(7));
        assert_eq!(deg3_component.coefficient(&2), Integer::from(0));

        // Extract degree 2 component
        let deg2_component = gr_algebra.homogeneous_component(&element, 2);
        assert_eq!(deg2_component.coefficient(&2), Integer::from(5));
        assert_eq!(deg2_component.coefficient(&3), Integer::from(0));
    }

    #[test]
    fn test_element_degree() {
        let degree_fn = |i: &usize| *i;
        let product_fn = |i: &usize, j: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(i + j)
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        // Single term of degree 5
        let elem: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::from_basis_index(5);

        assert_eq!(gr_algebra.degree_of_element(&elem), Some(5));

        // Mixed degrees (should return max)
        let mixed: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (2, Integer::from(1)),
                (5, Integer::from(1)),
                (3, Integer::from(1)),
            ]);

        assert_eq!(gr_algebra.degree_of_element(&mixed), Some(5));
    }

    #[test]
    fn test_full_product() {
        let degree_fn = |i: &usize| *i;
        let product_fn = |i: &usize, j: &usize| {
            CombinatorialFreeModuleElement::from_basis_index(i + j)
        };

        let gr_algebra = AssociatedGradedAlgebra::new(degree_fn, product_fn);

        // Product of (b1 + b2) * (b2 + b3)
        let left: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (1, Integer::from(1)),
                (2, Integer::from(1)),
            ]);

        let right: CombinatorialFreeModuleElement<Integer, usize> =
            CombinatorialFreeModuleElement::sum_of_terms(vec![
                (2, Integer::from(1)),
                (3, Integer::from(1)),
            ]);

        let product = gr_algebra.product(&left, &right);

        // Expected: b3 (from 1+2), b4 (from 1+3 and 2+2), b5 (from 2+3)
        assert_eq!(product.coefficient(&3), Integer::from(1));
        assert_eq!(product.coefficient(&4), Integer::from(2));
        assert_eq!(product.coefficient(&5), Integer::from(1));
    }
}

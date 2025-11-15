//! BGG Dual Modules
//!
//! This module implements Bernstein-Gelfand-Gelfand (BGG) dual modules,
//! which are fundamental structures in the representation theory of
//! semisimple Lie algebras.
//!
//! # Mathematical Background
//!
//! For a semisimple Lie algebra 𝔤 with Borel subalgebra 𝔟 and highest weight
//! module M, the BGG dual module M^∨ is defined as the weight space dual:
//!
//! M^∨ = ⊕_λ M_λ*
//!
//! where M_λ is the λ-weight space of M. The module structure on M^∨ is
//! given by:
//!
//! (x · φ)(v) = φ(τ(x) · v)
//!
//! where τ is the transpose anti-automorphism of 𝔤.
//!
//! # Category O
//!
//! BGG dual modules lie in the BGG category O, which consists of:
//! - Finitely generated U(𝔤)-modules
//! - Semisimple under the Cartan subalgebra 𝔥
//! - Locally 𝔫-finite (where 𝔫 is the nilradical)
//!
//! # Simple Modules
//!
//! For a highest weight λ, the simple module L_λ can be realized as a
//! submodule of the dual Verma module M_λ^∨. This provides an efficient
//! computational framework for working with simple modules.
//!
//! Corresponds to sage.algebras.lie_algebras.bgg_dual_module
//!
//! # References
//!
//! - Bernstein, Gelfand, Gelfand "Category of 𝔤-modules" (1976)
//! - Humphreys "Representations of Semisimple Lie Algebras in the BGG Category O" (2008)
//! - Dixmier "Enveloping Algebras" (1996)

use crate::cartan_type::CartanType;
use rustmath_core::Ring;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Index for simple module basis elements
///
/// Basis elements are indexed by sequences representing PBW (Poincaré-Birkhoff-Witt)
/// monomials applied to the highest weight vector.
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bgg_dual_module::SimpleModuleIndices;
/// let indices = SimpleModuleIndices::new(3);
/// assert_eq!(indices.rank(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleModuleIndices {
    /// Rank of the underlying Lie algebra
    rank: usize,
    /// Maximum degree for indexing
    max_degree: Option<usize>,
}

impl SimpleModuleIndices {
    /// Create new simple module indices
    pub fn new(rank: usize) -> Self {
        SimpleModuleIndices {
            rank,
            max_degree: None,
        }
    }

    /// Create with a maximum degree bound
    pub fn with_max_degree(rank: usize, max_degree: usize) -> Self {
        SimpleModuleIndices {
            rank,
            max_degree: Some(max_degree),
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the maximum degree (if set)
    pub fn max_degree(&self) -> Option<usize> {
        self.max_degree
    }

    /// Check if an index is valid
    pub fn is_valid(&self, index: &[usize]) -> bool {
        if let Some(max) = self.max_degree {
            index.len() <= max && index.iter().all(|&i| i < self.rank)
        } else {
            index.iter().all(|&i| i < self.rank)
        }
    }
}

impl Display for SimpleModuleIndices {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(max) = self.max_degree {
            write!(
                f,
                "Simple module indices (rank {}, max degree {})",
                self.rank, max
            )
        } else {
            write!(f, "Simple module indices (rank {})", self.rank)
        }
    }
}

/// Simple Module L_λ
///
/// The simple highest weight module with highest weight λ.
/// Realized as a submodule of the dual Verma module.
///
/// # Type Parameters
///
/// * `R` - The base ring (typically ℚ or ℂ)
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bgg_dual_module::SimpleModule;
/// # use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// # use rustmath_integers::Integer;
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let weight = vec![Integer::from(2), Integer::from(1)];
/// let module: SimpleModule<Integer> = SimpleModule::new(ct, weight);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleModule<R: Ring> {
    /// Cartan type of the underlying Lie algebra
    cartan_type: CartanType,
    /// Highest weight λ
    highest_weight: Vec<R>,
    /// Indexing structure
    indices: SimpleModuleIndices,
    /// Basis elements (indexed by PBW monomials)
    basis_cache: HashMap<Vec<usize>, R>,
}

impl<R: Ring + Clone> SimpleModule<R> {
    /// Create a new simple module with highest weight λ
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type of the Lie algebra
    /// * `highest_weight` - The highest weight λ
    pub fn new(cartan_type: CartanType, highest_weight: Vec<R>) -> Self {
        let rank = cartan_type.rank();
        SimpleModule {
            cartan_type,
            highest_weight,
            indices: SimpleModuleIndices::new(rank),
            basis_cache: HashMap::new(),
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the highest weight
    pub fn highest_weight(&self) -> &[R] {
        &self.highest_weight
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.cartan_type.rank()
    }

    /// Check if the module is finite-dimensional
    ///
    /// The module is finite-dimensional if and only if the highest weight
    /// is dominant integral.
    pub fn is_finite_dimensional(&self) -> bool
    where
        R: PartialOrd + From<i64>,
    {
        // Check if all weights are non-negative (dominant)
        self.highest_weight
            .iter()
            .all(|w| w >= &R::from(0))
    }

    /// Get the dimension (if finite-dimensional)
    pub fn dimension(&self) -> Option<usize>
    where
        R: PartialOrd + From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
    {
        if !self.is_finite_dimensional() {
            return None;
        }

        // Use Weyl dimension formula
        // For now, return None (full implementation would compute this)
        None
    }

    /// Lift an element to the ambient dual Verma module
    ///
    /// Maps from L_λ to M_λ^∨
    pub fn lift<T>(&self, element: T) -> T {
        // In a full implementation, this would embed into the dual Verma module
        element
    }

    /// Retract an element from the ambient dual Verma module
    ///
    /// Projects from M_λ^∨ to L_λ (if the element is in the image)
    pub fn retract<T>(&self, element: T) -> Option<T> {
        // In a full implementation, this would check membership and project
        Some(element)
    }

    /// Get a basis element by index
    pub fn basis_element(&self, index: &[usize]) -> Option<SimpleModuleElement<R>>
    where
        R: From<i64>,
    {
        if !self.indices.is_valid(index) {
            return None;
        }

        Some(SimpleModuleElement {
            index: index.to_vec(),
            coefficient: R::from(1),
        })
    }

    /// Construct basis elements up to a given depth
    ///
    /// Iteratively builds basis elements by applying lowering operators
    /// to the highest weight vector.
    pub fn basis_up_to_depth(&self, depth: usize) -> Vec<SimpleModuleElement<R>>
    where
        R: From<i64>,
    {
        let mut basis = Vec::new();

        // Start with highest weight vector (empty index)
        basis.push(SimpleModuleElement {
            index: vec![],
            coefficient: R::from(1),
        });

        // Iteratively construct basis at each depth
        for d in 1..=depth {
            self.construct_depth_basis(d, &mut basis);
        }

        basis
    }

    /// Construct basis elements at a specific depth
    fn construct_depth_basis(&self, depth: usize, basis: &mut Vec<SimpleModuleElement<R>>)
    where
        R: From<i64>,
    {
        // For each generator, apply it to lower-depth elements
        for gen in 0..self.rank() {
            // This is a placeholder - full implementation would:
            // 1. Apply the lowering operator f_gen
            // 2. Check for linear independence
            // 3. Add new basis elements

            if depth == 1 {
                basis.push(SimpleModuleElement {
                    index: vec![gen],
                    coefficient: R::from(1),
                });
            }
        }
    }
}

impl<R: Ring + Clone> Display for SimpleModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Simple module of type {} with highest weight [{:?}]",
            self.cartan_type,
            self.highest_weight
                .iter()
                .map(|_| "λ")
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Finite-Dimensional Simple Module
///
/// A simple module where the highest weight is dominant integral,
/// guaranteeing finite dimension. This specialization enables
/// additional computational methods.
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bgg_dual_module::FiniteDimensionalSimpleModule;
/// # use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// # use rustmath_integers::Integer;
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let weight = vec![Integer::from(1), Integer::from(0)];
/// let module: FiniteDimensionalSimpleModule<Integer> =
///     FiniteDimensionalSimpleModule::new(ct, weight);
/// assert!(module.is_finite_dimensional());
/// ```
#[derive(Debug, Clone)]
pub struct FiniteDimensionalSimpleModule<R: Ring> {
    /// Base simple module
    base: SimpleModule<R>,
}

impl<R: Ring + Clone + PartialOrd + From<i64>> FiniteDimensionalSimpleModule<R> {
    /// Create a new finite-dimensional simple module
    ///
    /// # Arguments
    ///
    /// * `cartan_type` - The Cartan type
    /// * `highest_weight` - The dominant integral highest weight
    ///
    /// # Panics
    ///
    /// Panics if the highest weight is not dominant (has negative entries)
    pub fn new(cartan_type: CartanType, highest_weight: Vec<R>) -> Self {
        // Verify dominant weight
        assert!(
            highest_weight.iter().all(|w| w >= &R::from(0)),
            "Highest weight must be dominant (non-negative)"
        );

        FiniteDimensionalSimpleModule {
            base: SimpleModule::new(cartan_type, highest_weight),
        }
    }

    /// Get the base simple module
    pub fn base(&self) -> &SimpleModule<R> {
        &self.base
    }

    /// Check if finite-dimensional (always true for this type)
    pub fn is_finite_dimensional(&self) -> bool {
        true
    }

    /// Get the complete basis
    ///
    /// Computes the full basis using the BGG resolution
    pub fn basis(&self) -> Vec<SimpleModuleElement<R>> {
        // Full implementation would compute using BGG resolution
        // For now, return basis up to some reasonable depth
        self.base.basis_up_to_depth(5)
    }
}

impl<R: Ring + Clone> Display for FiniteDimensionalSimpleModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Finite-dimensional {}", self.base)
    }
}

/// BGG Dual Module M^∨
///
/// The dual of a highest weight module M, with the BGG dual structure.
/// For a Verma module M_λ, the dual M_λ^∨ contains the simple module L_λ
/// as a submodule.
///
/// # Type Parameters
///
/// * `R` - The base ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bgg_dual_module::BGGDualModule;
/// # use rustmath_liealgebras::cartan_type::{CartanType, CartanLetter};
/// # use rustmath_integers::Integer;
/// let ct = CartanType::new(CartanLetter::A, 2).unwrap();
/// let weight = vec![Integer::from(2), Integer::from(1)];
/// let dual: BGGDualModule<Integer> = BGGDualModule::new(ct, weight);
/// ```
#[derive(Debug, Clone)]
pub struct BGGDualModule<R: Ring> {
    /// Cartan type
    cartan_type: CartanType,
    /// Highest weight of the original module
    highest_weight: Vec<R>,
    /// Simple module contained in this dual
    simple_submodule: Option<SimpleModule<R>>,
}

impl<R: Ring + Clone> BGGDualModule<R> {
    /// Create a new BGG dual module
    pub fn new(cartan_type: CartanType, highest_weight: Vec<R>) -> Self {
        BGGDualModule {
            cartan_type,
            highest_weight,
            simple_submodule: None,
        }
    }

    /// Get the Cartan type
    pub fn cartan_type(&self) -> &CartanType {
        &self.cartan_type
    }

    /// Get the highest weight
    pub fn highest_weight(&self) -> &[R] {
        &self.highest_weight
    }

    /// Get or construct the simple submodule
    pub fn simple_submodule(&mut self) -> &SimpleModule<R> {
        if self.simple_submodule.is_none() {
            self.simple_submodule = Some(SimpleModule::new(
                self.cartan_type.clone(),
                self.highest_weight.clone(),
            ));
        }
        self.simple_submodule.as_ref().unwrap()
    }

    /// Action of a Lie algebra element on a basis element
    ///
    /// Computes x · φ where x is a Lie algebra element and φ is
    /// a dual basis element.
    pub fn lie_algebra_action(
        &self,
        gen_index: usize,
        basis_index: &[usize],
    ) -> Vec<(Vec<usize>, R)>
    where
        R: From<i64>,
    {
        // Placeholder - full implementation would compute the action
        // using the transpose anti-automorphism
        vec![(basis_index.to_vec(), R::from(1))]
    }

    /// Action of a PBW monomial on a basis element
    pub fn pbw_action(&self, monomial: &[usize], basis_index: &[usize]) -> Vec<(Vec<usize>, R)>
    where
        R: From<i64>,
    {
        // Placeholder - full implementation would compute PBW action
        let mut result_index = basis_index.to_vec();
        result_index.extend_from_slice(monomial);
        vec![(result_index, R::from(1))]
    }
}

impl<R: Ring + Clone> Display for BGGDualModule<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BGG dual module of type {} with highest weight",
            self.cartan_type
        )
    }
}

/// Element of a simple module
///
/// Represents an element in L_λ as a linear combination of basis elements.
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::bgg_dual_module::SimpleModuleElement;
/// # use rustmath_integers::Integer;
/// let elem: SimpleModuleElement<Integer> = SimpleModuleElement::new(vec![0, 1], Integer::from(3));
/// assert_eq!(elem.index(), &[0, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleModuleElement<R: Ring> {
    /// Index in the basis (PBW monomial)
    index: Vec<usize>,
    /// Coefficient
    coefficient: R,
}

impl<R: Ring + Clone> SimpleModuleElement<R> {
    /// Create a new element
    pub fn new(index: Vec<usize>, coefficient: R) -> Self {
        SimpleModuleElement { index, coefficient }
    }

    /// Create the zero element
    pub fn zero() -> Self
    where
        R: From<i64>,
    {
        SimpleModuleElement {
            index: vec![],
            coefficient: R::from(0),
        }
    }

    /// Get the index
    pub fn index(&self) -> &[usize] {
        &self.index
    }

    /// Get the coefficient
    pub fn coefficient(&self) -> &R {
        &self.coefficient
    }

    /// Get the degree (length of the index)
    pub fn degree(&self) -> usize {
        self.index.len()
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coefficient.is_zero()
    }
}

impl<R: Ring + Clone + Display> Display for SimpleModuleElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.index.is_empty() {
            write!(f, "{} · v₀", self.coefficient)
        } else {
            write!(f, "{} · f_{:?} · v₀", self.coefficient, self.index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartan_type::{CartanLetter, CartanType};
    use rustmath_integers::Integer;

    #[test]
    fn test_simple_module_indices() {
        let indices = SimpleModuleIndices::new(3);
        assert_eq!(indices.rank(), 3);

        assert!(indices.is_valid(&[0, 1, 2]));
        assert!(!indices.is_valid(&[0, 3])); // 3 >= rank
    }

    #[test]
    fn test_simple_module_indices_with_degree() {
        let indices = SimpleModuleIndices::with_max_degree(3, 2);
        assert_eq!(indices.max_degree(), Some(2));

        assert!(indices.is_valid(&[0, 1]));
        assert!(!indices.is_valid(&[0, 1, 2])); // degree > max
    }

    #[test]
    fn test_simple_module_creation() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let weight = vec![Integer::from(2), Integer::from(1)];
        let module: SimpleModule<Integer> = SimpleModule::new(ct, weight);

        assert_eq!(module.rank(), 2);
        assert_eq!(module.highest_weight().len(), 2);
    }

    #[test]
    fn test_simple_module_finite_dimensional() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();

        // Dominant weight - finite dimensional
        let weight1 = vec![Integer::from(2), Integer::from(1)];
        let module1: SimpleModule<Integer> = SimpleModule::new(ct.clone(), weight1);
        assert!(module1.is_finite_dimensional());

        // Non-dominant weight - infinite dimensional
        let weight2 = vec![Integer::from(-1), Integer::from(1)];
        let module2: SimpleModule<Integer> = SimpleModule::new(ct, weight2);
        assert!(!module2.is_finite_dimensional());
    }

    #[test]
    fn test_basis_element() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let weight = vec![Integer::from(1), Integer::from(1)];
        let module: SimpleModule<Integer> = SimpleModule::new(ct, weight);

        let elem = module.basis_element(&[0, 1]);
        assert!(elem.is_some());
        assert_eq!(elem.unwrap().index(), &[0, 1]);
    }

    #[test]
    fn test_basis_construction() {
        let ct = CartanType::new(CartanLetter::A, 1).unwrap();
        let weight = vec![Integer::from(2)];
        let module: SimpleModule<Integer> = SimpleModule::new(ct, weight);

        let basis = module.basis_up_to_depth(2);
        assert!(!basis.is_empty());
    }

    #[test]
    fn test_finite_dimensional_simple_module() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let weight = vec![Integer::from(1), Integer::from(0)];
        let module: FiniteDimensionalSimpleModule<Integer> =
            FiniteDimensionalSimpleModule::new(ct, weight);

        assert!(module.is_finite_dimensional());
    }

    #[test]
    #[should_panic(expected = "Highest weight must be dominant")]
    fn test_finite_dimensional_non_dominant() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let weight = vec![Integer::from(-1), Integer::from(1)];
        let _module: FiniteDimensionalSimpleModule<Integer> =
            FiniteDimensionalSimpleModule::new(ct, weight);
    }

    #[test]
    fn test_bgg_dual_module() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let weight = vec![Integer::from(2), Integer::from(1)];
        let dual: BGGDualModule<Integer> = BGGDualModule::new(ct, weight);

        assert_eq!(dual.cartan_type().rank(), 2);
    }

    #[test]
    fn test_bgg_dual_simple_submodule() {
        let ct = CartanType::new(CartanLetter::A, 2).unwrap();
        let weight = vec![Integer::from(1), Integer::from(1)];
        let mut dual: BGGDualModule<Integer> = BGGDualModule::new(ct, weight);

        let simple = dual.simple_submodule();
        assert_eq!(simple.rank(), 2);
    }

    #[test]
    fn test_element_creation() {
        let elem: SimpleModuleElement<Integer> =
            SimpleModuleElement::new(vec![0, 1], Integer::from(5));

        assert_eq!(elem.index(), &[0, 1]);
        assert_eq!(elem.coefficient(), &Integer::from(5));
        assert_eq!(elem.degree(), 2);
    }

    #[test]
    fn test_element_zero() {
        let zero: SimpleModuleElement<Integer> = SimpleModuleElement::zero();
        assert!(zero.is_zero());
    }
}

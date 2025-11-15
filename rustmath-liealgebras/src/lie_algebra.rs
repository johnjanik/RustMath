//! Core Lie Algebra Infrastructure
//!
//! This module provides the fundamental traits and types for working with Lie algebras
//! in RustMath. It defines the abstract interface for Lie algebras and provides
//! concrete implementations for common construction patterns.
//!
//! # Lie Algebra Hierarchy
//!
//! - `LieAlgebraBase`: Core trait defining bracket operations
//! - `LieAlgebraWithGenerators`: Algebras with distinguished generator sets
//! - `FinitelyGeneratedLieAlgebra`: Algebras with finite generator count
//! - `InfinitelyGeneratedLieAlgebra`: Algebras with infinite generators
//! - `LieAlgebraFromAssociative`: Lie algebras constructed via commutator bracket
//! - `MatrixLieAlgebraFromAssociative`: Matrix-specific specialization
//!
//! # Construction Patterns
//!
//! 1. **From Structure Coefficients**: Explicitly specify bracket relations
//! 2. **From Associative Algebras**: Use commutator [a,b] = ab - ba
//! 3. **From Cartan Type**: Classical and exceptional Lie algebras
//! 4. **Free Construction**: Free Lie algebras on generators
//!
//! Corresponds to sage.algebras.lie_algebras.lie_algebra
//!
//! # References
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - Kac, V. "Infinite Dimensional Lie Algebras" (1990)
//! - SageMath: sage.algebras.lie_algebras.lie_algebra

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt::{self, Display, Debug};
use std::hash::Hash;
use std::ops::{Add, Sub, Mul, Neg};
use std::marker::PhantomData;

// ============================================================================
// Core Traits
// ============================================================================

/// Base trait for all Lie algebra elements
///
/// Defines the fundamental Lie bracket operation that must satisfy:
/// - Bilinearity: [ax + by, z] = a[x,z] + b[y,z]
/// - Antisymmetry: [x, y] = -[y, x]
/// - Jacobi identity: [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0
pub trait LieAlgebraElement<R: Ring>: Clone + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> {
    /// Compute the Lie bracket [self, other]
    fn bracket(&self, other: &Self) -> Self;

    /// Scalar multiplication
    fn scalar_mul(&self, scalar: &R) -> Self;

    /// Check if this is the zero element
    fn is_zero(&self) -> bool;
}

/// Trait for Lie algebra structures (the algebra itself, not elements)
pub trait LieAlgebraBase<R: Ring> {
    /// The type of elements in this Lie algebra
    type Element: LieAlgebraElement<R>;

    /// Get the zero element
    fn zero(&self) -> Self::Element;

    /// Create an element from coordinates (if applicable)
    fn element_from_coords(&self, coords: Vec<R>) -> Option<Self::Element>;

    /// Get the dimension (if finite-dimensional)
    fn dimension(&self) -> Option<usize>;

    /// Check if this is an abelian Lie algebra
    fn is_abelian(&self) -> bool {
        false
    }

    /// Check if this is nilpotent
    fn is_nilpotent(&self) -> bool {
        false
    }

    /// Check if this is solvable
    fn is_solvable(&self) -> bool {
        false
    }

    /// Check if this is semisimple
    fn is_semisimple(&self) -> bool {
        false
    }
}

/// Trait for Lie algebras with distinguished generators
///
/// This trait extends LieAlgebraBase with the concept of a generating set.
/// The algebra can be finite or infinite dimensional, but it has a (possibly infinite)
/// set of distinguished generators.
pub trait LieAlgebraWithGenerators<R: Ring>: LieAlgebraBase<R> {
    /// Get the generators as a vector (for finite generation)
    fn generators(&self) -> Vec<Self::Element>;

    /// Get the number of generators (if finite)
    fn num_generators(&self) -> Option<usize>;

    /// Get the i-th generator (0-indexed)
    fn generator(&self, i: usize) -> Option<Self::Element>;

    /// Get generator by name/index key
    fn generator_by_key(&self, key: &str) -> Option<Self::Element>;
}

// ============================================================================
// Finitely Generated Lie Algebras
// ============================================================================

/// A Lie algebra with a finite set of generators
///
/// This represents Lie algebras where the generating set is finite,
/// though the algebra itself may be infinite-dimensional (e.g., free Lie algebras).
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `E` - The element type
#[derive(Clone, Debug)]
pub struct FinitelyGeneratedLieAlgebra<R: Ring + Clone, E: LieAlgebraElement<R>> {
    /// Names/labels for generators
    generator_names: Vec<String>,

    /// The actual generator elements
    generators: Vec<E>,

    /// Dimension (if finite-dimensional)
    dimension: Option<usize>,

    /// Ring marker
    _ring: PhantomData<R>,
}

impl<R: Ring + Clone, E: LieAlgebraElement<R> + Debug> FinitelyGeneratedLieAlgebra<R, E> {
    /// Create a new finitely generated Lie algebra
    ///
    /// # Arguments
    ///
    /// * `generator_names` - Names for the generators
    /// * `generators` - The actual generator elements
    /// * `dimension` - The dimension (None if infinite-dimensional)
    pub fn new(
        generator_names: Vec<String>,
        generators: Vec<E>,
        dimension: Option<usize>,
    ) -> Result<Self, String> {
        if generator_names.len() != generators.len() {
            return Err("Number of names must match number of generators".to_string());
        }

        Ok(FinitelyGeneratedLieAlgebra {
            generator_names,
            generators,
            dimension,
            _ring: PhantomData,
        })
    }

    /// Get the generator names
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// Number of generators
    pub fn num_generators(&self) -> usize {
        self.generators.len()
    }
}

impl<R: Ring + Clone, E: LieAlgebraElement<R> + Debug> Display for FinitelyGeneratedLieAlgebra<R, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dimension {
            Some(d) => write!(
                f,
                "Finitely generated Lie algebra on {} generators (dimension {})",
                self.num_generators(),
                d
            ),
            None => write!(
                f,
                "Finitely generated Lie algebra on {} generators (infinite-dimensional)",
                self.num_generators()
            ),
        }
    }
}

// ============================================================================
// Infinitely Generated Lie Algebras
// ============================================================================

/// A Lie algebra with an infinite set of generators
///
/// This represents Lie algebras where the generating set is infinite,
/// such as the infinite-dimensional Heisenberg algebra or Virasoro algebra.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `E` - The element type
#[derive(Clone, Debug)]
pub struct InfinitelyGeneratedLieAlgebra<R: Ring + Clone, E: LieAlgebraElement<R>> {
    /// Description of the indexing set
    index_description: String,

    /// A function or description for how to construct generators
    /// (We store a description; actual construction happens via methods)
    construction_method: String,

    /// Ring and element markers
    _ring: PhantomData<R>,
    _element: PhantomData<E>,
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> InfinitelyGeneratedLieAlgebra<R, E> {
    /// Create a new infinitely generated Lie algebra
    ///
    /// # Arguments
    ///
    /// * `index_description` - Description of the index set (e.g., "Z" for integers)
    /// * `construction_method` - Description of how generators are constructed
    pub fn new(index_description: String, construction_method: String) -> Self {
        InfinitelyGeneratedLieAlgebra {
            index_description,
            construction_method,
            _ring: PhantomData,
            _element: PhantomData,
        }
    }

    /// Get a description of the indexing set
    pub fn index_description(&self) -> &str {
        &self.index_description
    }
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> Display for InfinitelyGeneratedLieAlgebra<R, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Infinitely generated Lie algebra indexed by {}",
            self.index_description
        )
    }
}

// ============================================================================
// Lie Algebras from Associative Algebras
// ============================================================================

/// A Lie algebra constructed from an associative algebra via the commutator bracket
///
/// Given an associative algebra A with multiplication *, the commutator
/// bracket [a, b] = a*b - b*a makes A into a Lie algebra.
///
/// This is a fundamental construction that produces many important Lie algebras:
/// - gl_n(R): General linear algebra from n×n matrices
/// - sl_n(R): Special linear algebra from traceless matrices
/// - Matrix Lie algebras: Various subalgebras of gl_n
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `A` - The associative algebra element type
///
/// # Examples
///
/// ```ignore
/// use rustmath_liealgebras::LieAlgebraFromAssociative;
/// use rustmath_matrix::Matrix;
/// use rustmath_integers::Integer;
///
/// // Create gl_2(Z) from 2×2 integer matrices
/// let gl2 = LieAlgebraFromAssociative::<Integer, Matrix<Integer>>::from_matrices(2);
/// ```
#[derive(Clone, Debug)]
pub struct LieAlgebraFromAssociative<R: Ring + Clone, A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A>,
{
    /// Description of the associative algebra
    algebra_description: String,

    /// Basis elements of the associative algebra (if finite-dimensional)
    basis: Option<Vec<A>>,

    /// Dimension (if finite-dimensional)
    dimension: Option<usize>,

    /// Ring marker
    _ring: PhantomData<R>,
}

impl<R: Ring + Clone, A> LieAlgebraFromAssociative<R, A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A>,
{
    /// Create a Lie algebra from an associative algebra
    ///
    /// # Arguments
    ///
    /// * `description` - Description of the associative algebra
    /// * `basis` - Optional basis (for finite-dimensional case)
    pub fn new(description: String, basis: Option<Vec<A>>) -> Self {
        let dimension = basis.as_ref().map(|b| b.len());

        LieAlgebraFromAssociative {
            algebra_description: description,
            basis,
            dimension,
            _ring: PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get a description of the associative algebra
    pub fn associative_algebra_description(&self) -> &str {
        &self.algebra_description
    }
}

impl<R: Ring + Clone, A> Display for LieAlgebraFromAssociative<R, A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lie algebra of {}",
            self.algebra_description
        )
    }
}

/// Element of a Lie algebra constructed from an associative algebra
///
/// This wraps an element of the associative algebra and provides
/// the Lie bracket operation via the commutator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssociativeLieElement<A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A> + PartialEq,
{
    /// The underlying associative algebra element
    element: A,
}

impl<A> AssociativeLieElement<A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A> + PartialEq,
{
    /// Create a new element
    pub fn new(element: A) -> Self {
        AssociativeLieElement { element }
    }

    /// Get the underlying associative element
    pub fn associative_element(&self) -> &A {
        &self.element
    }

    /// Lift this element to the associative algebra
    pub fn lift(&self) -> A {
        self.element.clone()
    }

    /// Compute the commutator bracket [self, other] = self*other - other*self
    pub fn commutator(&self, other: &Self) -> Self {
        let product1 = self.element.clone() * other.element.clone();
        let product2 = other.element.clone() * self.element.clone();
        AssociativeLieElement {
            element: product1 - product2,
        }
    }
}

impl<A> Add for AssociativeLieElement<A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A> + PartialEq,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        AssociativeLieElement {
            element: self.element + other.element,
        }
    }
}

impl<A> Sub for AssociativeLieElement<A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A> + PartialEq,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        AssociativeLieElement {
            element: self.element - other.element,
        }
    }
}

impl<A> Neg for AssociativeLieElement<A>
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A> + PartialEq,
{
    type Output = Self;

    fn neg(self) -> Self {
        AssociativeLieElement {
            element: -self.element,
        }
    }
}

impl<R, A> LieAlgebraElement<R> for AssociativeLieElement<A>
where
    R: Ring + Clone,
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A> + PartialEq + Mul<R, Output = A>,
{
    fn bracket(&self, other: &Self) -> Self {
        self.commutator(other)
    }

    fn scalar_mul(&self, scalar: &R) -> Self {
        AssociativeLieElement {
            element: self.element.clone() * scalar.clone(),
        }
    }

    fn is_zero(&self) -> bool {
        // Note: This requires a proper zero check for A
        // In practice, this would need to be implemented per algebra type
        false // Placeholder
    }
}

// ============================================================================
// Matrix Lie Algebras from Associative Algebras
// ============================================================================

/// Specialized Lie algebra for matrix algebras
///
/// This is a specialization of LieAlgebraFromAssociative for the case where
/// the associative algebra is a matrix algebra. It provides additional
/// matrix-specific functionality.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug)]
pub struct MatrixLieAlgebraFromAssociative<R: Ring + Clone> {
    /// Size of matrices (n for n×n matrices)
    matrix_size: usize,

    /// Description (e.g., "gl_3(Z)")
    description: String,

    /// Ring marker
    _ring: PhantomData<R>,
}

impl<R: Ring + Clone> MatrixLieAlgebraFromAssociative<R> {
    /// Create a matrix Lie algebra from n×n matrices
    ///
    /// # Arguments
    ///
    /// * `n` - The matrix size
    /// * `description` - Description of the algebra
    pub fn new(n: usize, description: String) -> Self {
        MatrixLieAlgebraFromAssociative {
            matrix_size: n,
            description,
            _ring: PhantomData,
        }
    }

    /// Get the matrix size
    pub fn matrix_size(&self) -> usize {
        self.matrix_size
    }

    /// Get the dimension (n² for gl_n)
    pub fn dimension(&self) -> usize {
        self.matrix_size * self.matrix_size
    }
}

impl<R: Ring + Clone> Display for MatrixLieAlgebraFromAssociative<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

/// Element of a matrix Lie algebra
///
/// Wraps a matrix and provides Lie bracket via commutator
#[derive(Clone, Debug)]
pub struct MatrixLieElement<R: Ring + Clone> {
    /// The underlying matrix
    matrix: Matrix<R>,
}

impl<R: Ring + Clone> MatrixLieElement<R> {
    /// Create a new matrix Lie element
    pub fn new(matrix: Matrix<R>) -> Self {
        MatrixLieElement { matrix }
    }

    /// Get the underlying matrix
    pub fn matrix(&self) -> &Matrix<R> {
        &self.matrix
    }

    /// Lift to the matrix algebra (returns a copy of the matrix)
    pub fn lift_to_matrix(&self) -> Matrix<R> {
        self.matrix.clone()
    }
}

// ============================================================================
// Morphisms and Lifts
// ============================================================================

/// Morphism that lifts Lie algebra elements to the associative algebra
///
/// For a Lie algebra L constructed from an associative algebra A,
/// this provides the natural inclusion morphism L → A.
pub struct LiftMorphismToAssociative<R, A>
where
    R: Ring + Clone,
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A>,
{
    /// Source Lie algebra
    source_description: String,

    /// Target associative algebra
    target_description: String,

    /// Markers
    _ring: PhantomData<R>,
    _algebra: PhantomData<A>,
}

impl<R, A> LiftMorphismToAssociative<R, A>
where
    R: Ring + Clone,
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A>,
{
    /// Create a new lift morphism
    pub fn new(source: String, target: String) -> Self {
        LiftMorphismToAssociative {
            source_description: source,
            target_description: target,
            _ring: PhantomData,
            _algebra: PhantomData,
        }
    }

    /// Apply the morphism to an element
    pub fn apply(&self, element: &AssociativeLieElement<A>) -> A {
        element.lift()
    }

    /// Get the preimage of an associative element (the inverse map)
    pub fn preimage(&self, element: A) -> AssociativeLieElement<A> {
        AssociativeLieElement::new(element)
    }
}

impl<R, A> Display for LiftMorphismToAssociative<R, A>
where
    R: Ring + Clone,
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Neg<Output = A>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lift morphism: {} -> {}",
            self.source_description, self.target_description
        )
    }
}

// ============================================================================
// Element Wrapper
// ============================================================================

/// Generic wrapper for Lie algebra elements
///
/// This provides a uniform interface for elements from different concrete
/// Lie algebra implementations.
#[derive(Clone, Debug)]
pub struct LieAlgebraElementWrapper<R: Ring + Clone, E: LieAlgebraElement<R>> {
    /// The wrapped element
    element: E,

    /// Description of which algebra this element belongs to
    algebra_description: String,

    /// Ring marker
    _ring: PhantomData<R>,
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> LieAlgebraElementWrapper<R, E> {
    /// Create a new wrapped element
    pub fn new(element: E, algebra_description: String) -> Self {
        LieAlgebraElementWrapper {
            element,
            algebra_description,
            _ring: PhantomData,
        }
    }

    /// Get the underlying element
    pub fn element(&self) -> &E {
        &self.element
    }

    /// Unwrap to get the element
    pub fn unwrap(self) -> E {
        self.element
    }

    /// Get the algebra description
    pub fn algebra(&self) -> &str {
        &self.algebra_description
    }
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> LieAlgebraElement<R> for LieAlgebraElementWrapper<R, E> {
    fn bracket(&self, other: &Self) -> Self {
        LieAlgebraElementWrapper {
            element: self.element.bracket(&other.element),
            algebra_description: self.algebra_description.clone(),
            _ring: PhantomData,
        }
    }

    fn scalar_mul(&self, scalar: &R) -> Self {
        LieAlgebraElementWrapper {
            element: self.element.scalar_mul(scalar),
            algebra_description: self.algebra_description.clone(),
            _ring: PhantomData,
        }
    }

    fn is_zero(&self) -> bool {
        self.element.is_zero()
    }
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> Add for LieAlgebraElementWrapper<R, E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        LieAlgebraElementWrapper {
            element: self.element + other.element,
            algebra_description: self.algebra_description,
            _ring: PhantomData,
        }
    }
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> Sub for LieAlgebraElementWrapper<R, E> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        LieAlgebraElementWrapper {
            element: self.element - other.element,
            algebra_description: self.algebra_description,
            _ring: PhantomData,
        }
    }
}

impl<R: Ring + Clone, E: LieAlgebraElement<R>> Neg for LieAlgebraElementWrapper<R, E> {
    type Output = Self;

    fn neg(self) -> Self {
        LieAlgebraElementWrapper {
            element: -self.element,
            algebra_description: self.algebra_description,
            _ring: PhantomData,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    // Helper: Simple element type for testing
    #[derive(Clone, Debug, PartialEq)]
    struct SimpleElement {
        value: Integer,
    }

    impl Add for SimpleElement {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            SimpleElement { value: self.value + other.value }
        }
    }

    impl Sub for SimpleElement {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            SimpleElement { value: self.value - other.value }
        }
    }

    impl Neg for SimpleElement {
        type Output = Self;
        fn neg(self) -> Self {
            SimpleElement { value: -self.value }
        }
    }

    impl Mul for SimpleElement {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
            SimpleElement { value: self.value * other.value }
        }
    }

    impl Mul<Integer> for SimpleElement {
        type Output = Self;
        fn mul(self, scalar: Integer) -> Self {
            SimpleElement { value: self.value * scalar }
        }
    }

    impl LieAlgebraElement<Integer> for SimpleElement {
        fn bracket(&self, _other: &Self) -> Self {
            // Abelian for testing
            SimpleElement { value: Integer::from(0) }
        }

        fn scalar_mul(&self, scalar: &Integer) -> Self {
            SimpleElement { value: self.value.clone() * scalar.clone() }
        }

        fn is_zero(&self) -> bool {
            self.value == Integer::from(0)
        }
    }

    #[test]
    fn test_finitely_generated_creation() {
        let gen1 = SimpleElement { value: Integer::from(1) };
        let gen2 = SimpleElement { value: Integer::from(2) };

        let alg = FinitelyGeneratedLieAlgebra::new(
            vec!["e".to_string(), "f".to_string()],
            vec![gen1, gen2],
            Some(2),
        ).unwrap();

        assert_eq!(alg.num_generators(), 2);
        assert_eq!(alg.dimension(), Some(2));
    }

    #[test]
    fn test_infinitely_generated_creation() {
        let alg: InfinitelyGeneratedLieAlgebra<Integer, SimpleElement> =
            InfinitelyGeneratedLieAlgebra::new(
                "Z".to_string(),
                "Standard generators".to_string(),
            );

        assert_eq!(alg.index_description(), "Z");
    }

    #[test]
    fn test_associative_lie_element_commutator() {
        let a = AssociativeLieElement::new(SimpleElement { value: Integer::from(2) });
        let b = AssociativeLieElement::new(SimpleElement { value: Integer::from(3) });

        // For our simple element: commutator = 2*3 - 3*2 = 0 (abelian)
        let bracket = a.commutator(&b);
        assert_eq!(bracket.element.value, Integer::from(0));
    }

    #[test]
    fn test_associative_lie_element_operations() {
        let a = AssociativeLieElement::new(SimpleElement { value: Integer::from(5) });
        let b = AssociativeLieElement::new(SimpleElement { value: Integer::from(3) });

        let sum = a.clone() + b.clone();
        assert_eq!(sum.element.value, Integer::from(8));

        let diff = a.clone() - b.clone();
        assert_eq!(diff.element.value, Integer::from(2));

        let neg = -a.clone();
        assert_eq!(neg.element.value, Integer::from(-5));
    }

    #[test]
    fn test_element_wrapper() {
        let elem = SimpleElement { value: Integer::from(7) };
        let wrapped = LieAlgebraElementWrapper::new(elem, "Test algebra".to_string());

        assert_eq!(wrapped.algebra(), "Test algebra");
        assert_eq!(wrapped.element().value, Integer::from(7));
    }

    #[test]
    fn test_lift_morphism() {
        let morphism: LiftMorphismToAssociative<Integer, SimpleElement> =
            LiftMorphismToAssociative::new(
                "Lie algebra".to_string(),
                "Associative algebra".to_string(),
            );

        let elem = SimpleElement { value: Integer::from(10) };
        let lie_elem = AssociativeLieElement::new(elem.clone());

        let lifted = morphism.apply(&lie_elem);
        assert_eq!(lifted.value, Integer::from(10));

        let preimage = morphism.preimage(elem);
        assert_eq!(preimage.element.value, Integer::from(10));
    }

    #[test]
    fn test_matrix_lie_algebra_creation() {
        let gl3: MatrixLieAlgebraFromAssociative<Integer> =
            MatrixLieAlgebraFromAssociative::new(3, "gl_3(Z)".to_string());

        assert_eq!(gl3.matrix_size(), 3);
        assert_eq!(gl3.dimension(), 9);
    }

    #[test]
    fn test_lie_algebra_from_associative() {
        let alg: LieAlgebraFromAssociative<Integer, SimpleElement> =
            LieAlgebraFromAssociative::new(
                "Simple associative algebra".to_string(),
                None,
            );

        assert_eq!(alg.dimension(), None);
        assert_eq!(alg.associative_algebra_description(), "Simple associative algebra");
    }
}

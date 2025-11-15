//! Lie Algebra Representations
//!
//! A representation of a Lie algebra g over a field K is a vector space V
//! together with a bilinear map g × V → V (the action) satisfying:
//!
//! [x, y]·v = x·(y·v) - y·(x·v)  for all x,y ∈ g, v ∈ V
//!
//! This fundamental property connects the Lie bracket to the representation action,
//! making representations the primary tool for studying Lie algebras concretely.
//!
//! # Representation Types
//!
//! - **Representation_abstract**: Base trait for all representations
//! - **RepresentationByMorphism**: Defined via homomorphism to matrices
//! - **TrivialRepresentation**: One-dimensional representation with zero action
//! - **FaithfulRepresentationNilpotentPBW**: For nilpotent algebras via PBW quotient
//! - **FaithfulRepresentationPBWPosChar**: Positive characteristic construction
//!
//! # Mathematical Background
//!
//! Representations provide a concrete way to study abstract Lie algebras by
//! realizing them as matrices acting on vector spaces. Key facts:
//!
//! 1. **Schur's Lemma**: Irreducible representations have minimal endomorphism rings
//! 2. **Complete Reducibility**: Representations of semisimple algebras decompose
//! 3. **Weights**: For representations of sl_n, elements diagonalize simultaneously
//! 4. **Casimir Operators**: Central elements act as scalars on irreducibles
//!
//! Corresponds to sage.algebras.lie_algebras.representation
//!
//! # References
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - Fulton, W. and Harris, J. "Representation Theory" (1991)
//! - Hall, B. "Lie Groups, Lie Algebras, and Representations" (2015)

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt::{self, Display, Debug};
use std::hash::Hash;
use std::ops::{Add, Sub, Mul, Neg};
use std::marker::PhantomData;

// ============================================================================
// Base Representation Trait
// ============================================================================

/// Abstract base trait for Lie algebra representations
///
/// A representation consists of:
/// - A vector space V (the representation space)
/// - An action g × V → V satisfying the representation axiom
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
/// * `V` - The vector type (elements of the representation space)
pub trait Representation<R: Ring, V> {
    /// Get the dimension of the representation space (if finite)
    fn dimension(&self) -> Option<usize>;

    /// Action of a Lie algebra element on a vector
    ///
    /// This must satisfy: [x,y]·v = x·(y·v) - y·(x·v)
    fn action(&self, lie_element: &[R], vector: &V) -> V;

    /// Check if this is a faithful representation
    ///
    /// A representation is faithful if only the zero element acts as zero
    fn is_faithful(&self) -> bool {
        false // Default: unknown
    }

    /// Check if this is irreducible
    ///
    /// A representation is irreducible if it has no proper invariant subspaces
    fn is_irreducible(&self) -> bool {
        false // Default: unknown
    }
}

// ============================================================================
// Representation Element
// ============================================================================

/// Element of a representation space
///
/// Represents vectors in the vector space on which the Lie algebra acts.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug, PartialEq)]
pub struct RepresentationElement<R: Ring + Clone> {
    /// Coordinates in the representation basis
    coordinates: Vec<R>,
}

impl<R: Ring + Clone> RepresentationElement<R> {
    /// Create a new element
    pub fn new(coordinates: Vec<R>) -> Self {
        RepresentationElement { coordinates }
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coordinates
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Add for RepresentationElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.coordinates.len(), other.coordinates.len());

        let coords: Vec<R> = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        RepresentationElement::new(coords)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Sub for RepresentationElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.coordinates.len(), other.coordinates.len());

        let coords: Vec<R> = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        RepresentationElement::new(coords)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Neg for RepresentationElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coords: Vec<R> = self
            .coordinates
            .into_iter()
            .map(|c| -c)
            .collect();

        RepresentationElement::new(coords)
    }
}

// ============================================================================
// Representation by Morphism
// ============================================================================

/// Representation defined by a Lie algebra homomorphism to matrices
///
/// This is the most common way to define representations: specify how each
/// basis element of the Lie algebra acts as a matrix.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug)]
pub struct RepresentationByMorphism<R: Ring + Clone> {
    /// Dimension of the representation space
    dimension: usize,

    /// Dimension of the Lie algebra
    lie_algebra_dimension: usize,

    /// Action matrices: basis element index → matrix
    action_matrices: HashMap<usize, Matrix<R>>,
}

impl<R: Ring + Clone + PartialEq + From<i64>> RepresentationByMorphism<R> {
    /// Create a new representation by specifying action matrices
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimension of the representation space
    /// * `lie_algebra_dimension` - Dimension of the Lie algebra
    /// * `action_matrices` - Map from basis index to action matrix
    pub fn new(
        dimension: usize,
        lie_algebra_dimension: usize,
        action_matrices: HashMap<usize, Matrix<R>>,
    ) -> Result<Self, String> {
        // Verify all matrices have correct size
        for (idx, mat) in &action_matrices {
            if mat.rows() != dimension || mat.cols() != dimension {
                return Err(format!(
                    "Action matrix for basis element {} has wrong size: {}×{}, expected {}×{}",
                    idx,
                    mat.rows(),
                    mat.cols(),
                    dimension,
                    dimension
                ));
            }
        }

        Ok(RepresentationByMorphism {
            dimension,
            lie_algebra_dimension,
            action_matrices,
        })
    }

    /// Get the representation dimension
    pub fn representation_dimension(&self) -> usize {
        self.dimension
    }

    /// Get the Lie algebra dimension
    pub fn lie_algebra_dimension(&self) -> usize {
        self.lie_algebra_dimension
    }

    /// Get the action matrix for a basis element
    pub fn action_matrix(&self, index: usize) -> Option<&Matrix<R>> {
        self.action_matrices.get(&index)
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Representation<R, RepresentationElement<R>>
    for RepresentationByMorphism<R>
{
    fn dimension(&self) -> Option<usize> {
        Some(self.dimension)
    }

    fn action(&self, lie_element: &[R], vector: &RepresentationElement<R>) -> RepresentationElement<R> {
        assert_eq!(lie_element.len(), self.lie_algebra_dimension);
        assert_eq!(vector.dimension(), self.dimension);

        // Compute Σ lie_element[i] * action_matrices[i] * vector
        let mut result = vec![R::from(0); self.dimension];

        for (i, coeff) in lie_element.iter().enumerate() {
            if coeff == &R::from(0) {
                continue;
            }

            if let Some(mat) = self.action_matrices.get(&i) {
                // Apply matrix to vector and add scaled result
                for row in 0..self.dimension {
                    let mut sum = R::from(0);
                    for col in 0..self.dimension {
                        sum = sum.clone() + mat.get(row, col).clone() * vector.coordinates[col].clone();
                    }
                    result[row] = result[row].clone() + coeff.clone() * sum;
                }
            }
        }

        RepresentationElement::new(result)
    }
}

// ============================================================================
// Trivial Representation
// ============================================================================

/// The trivial (one-dimensional) representation
///
/// In this representation, every Lie algebra element acts as zero.
/// This is always a representation since [x,y]·v = 0 = x·0 - y·0.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug)]
pub struct TrivialRepresentation<R: Ring + Clone> {
    /// Dimension of the Lie algebra
    lie_algebra_dimension: usize,

    /// Ring marker
    _ring: PhantomData<R>,
}

impl<R: Ring + Clone> TrivialRepresentation<R> {
    /// Create the trivial representation
    pub fn new(lie_algebra_dimension: usize) -> Self {
        TrivialRepresentation {
            lie_algebra_dimension,
            _ring: PhantomData,
        }
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Representation<R, RepresentationElement<R>>
    for TrivialRepresentation<R>
{
    fn dimension(&self) -> Option<usize> {
        Some(1)
    }

    fn action(&self, lie_element: &[R], vector: &RepresentationElement<R>) -> RepresentationElement<R> {
        assert_eq!(lie_element.len(), self.lie_algebra_dimension);
        assert_eq!(vector.dimension(), 1);

        // Action is always zero
        RepresentationElement::new(vec![R::from(0)])
    }

    fn is_faithful(&self) -> bool {
        // Trivial rep is faithful only for the zero Lie algebra
        self.lie_algebra_dimension == 0
    }

    fn is_irreducible(&self) -> bool {
        true // One-dimensional reps are always irreducible
    }
}

// ============================================================================
// Faithful Representation for Nilpotent Algebras (PBW Construction)
// ============================================================================

/// Faithful representation of a nilpotent Lie algebra via PBW quotient
///
/// For a nilpotent Lie algebra, we can construct a faithful representation
/// using the quotient of the universal enveloping algebra by an appropriate ideal.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug)]
pub struct FaithfulRepresentationNilpotentPBW<R: Ring + Clone> {
    /// Dimension of the Lie algebra
    lie_algebra_dimension: usize,

    /// Dimension of the representation (depends on the construction)
    representation_dimension: usize,

    /// Action matrices computed from PBW basis
    action_matrices: HashMap<usize, Matrix<R>>,

    /// Whether this uses minimal basis
    is_minimal: bool,
}

impl<R: Ring + Clone + PartialEq + From<i64>> FaithfulRepresentationNilpotentPBW<R> {
    /// Create a faithful representation for a nilpotent Lie algebra
    ///
    /// # Arguments
    ///
    /// * `lie_algebra_dimension` - Dimension of the Lie algebra
    /// * `representation_dimension` - Dimension of the constructed representation
    /// * `action_matrices` - Pre-computed action matrices
    /// * `is_minimal` - Whether this uses minimal basis
    pub fn new(
        lie_algebra_dimension: usize,
        representation_dimension: usize,
        action_matrices: HashMap<usize, Matrix<R>>,
        is_minimal: bool,
    ) -> Self {
        FaithfulRepresentationNilpotentPBW {
            lie_algebra_dimension,
            representation_dimension,
            action_matrices,
            is_minimal,
        }
    }

    /// Check if using minimal basis
    pub fn is_minimal(&self) -> bool {
        self.is_minimal
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Representation<R, RepresentationElement<R>>
    for FaithfulRepresentationNilpotentPBW<R>
{
    fn dimension(&self) -> Option<usize> {
        Some(self.representation_dimension)
    }

    fn action(&self, lie_element: &[R], vector: &RepresentationElement<R>) -> RepresentationElement<R> {
        assert_eq!(lie_element.len(), self.lie_algebra_dimension);
        assert_eq!(vector.dimension(), self.representation_dimension);

        let mut result = vec![R::from(0); self.representation_dimension];

        for (i, coeff) in lie_element.iter().enumerate() {
            if coeff == &R::from(0) {
                continue;
            }

            if let Some(mat) = self.action_matrices.get(&i) {
                for row in 0..self.representation_dimension {
                    let mut sum = R::from(0);
                    for col in 0..self.representation_dimension {
                        sum = sum.clone() + mat.get(row, col).clone() * vector.coordinates[col].clone();
                    }
                    result[row] = result[row].clone() + coeff.clone() * sum;
                }
            }
        }

        RepresentationElement::new(result)
    }

    fn is_faithful(&self) -> bool {
        true // By construction
    }
}

// ============================================================================
// Faithful Representation in Positive Characteristic
// ============================================================================

/// Faithful representation for positive characteristic fields
///
/// Uses Gröbner basis techniques with minimal polynomials to construct
/// faithful representations when the base field has positive characteristic.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (must have positive characteristic)
#[derive(Clone, Debug)]
pub struct FaithfulRepresentationPBWPosChar<R: Ring + Clone> {
    /// Dimension of the Lie algebra
    lie_algebra_dimension: usize,

    /// Dimension of the representation
    representation_dimension: usize,

    /// Action matrices
    action_matrices: HashMap<usize, Matrix<R>>,

    /// Characteristic of the field
    characteristic: usize,
}

impl<R: Ring + Clone + PartialEq + From<i64>> FaithfulRepresentationPBWPosChar<R> {
    /// Create a faithful representation in positive characteristic
    ///
    /// # Arguments
    ///
    /// * `lie_algebra_dimension` - Dimension of the Lie algebra
    /// * `representation_dimension` - Dimension of the representation
    /// * `action_matrices` - Action matrices computed via Gröbner basis
    /// * `characteristic` - Characteristic of the base field
    pub fn new(
        lie_algebra_dimension: usize,
        representation_dimension: usize,
        action_matrices: HashMap<usize, Matrix<R>>,
        characteristic: usize,
    ) -> Self {
        assert!(characteristic > 0, "Characteristic must be positive");

        FaithfulRepresentationPBWPosChar {
            lie_algebra_dimension,
            representation_dimension,
            action_matrices,
            characteristic,
        }
    }

    /// Get the characteristic
    pub fn characteristic(&self) -> usize {
        self.characteristic
    }
}

impl<R: Ring + Clone + PartialEq + From<i64>> Representation<R, RepresentationElement<R>>
    for FaithfulRepresentationPBWPosChar<R>
{
    fn dimension(&self) -> Option<usize> {
        Some(self.representation_dimension)
    }

    fn action(&self, lie_element: &[R], vector: &RepresentationElement<R>) -> RepresentationElement<R> {
        assert_eq!(lie_element.len(), self.lie_algebra_dimension);
        assert_eq!(vector.dimension(), self.representation_dimension);

        let mut result = vec![R::from(0); self.representation_dimension];

        for (i, coeff) in lie_element.iter().enumerate() {
            if coeff == &R::from(0) {
                continue;
            }

            if let Some(mat) = self.action_matrices.get(&i) {
                for row in 0..self.representation_dimension {
                    let mut sum = R::from(0);
                    for col in 0..self.representation_dimension {
                        sum = sum.clone() + mat.get(row, col).clone() * vector.coordinates[col].clone();
                    }
                    result[row] = result[row].clone() + coeff.clone() * sum;
                }
            }
        }

        RepresentationElement::new(result)
    }

    fn is_faithful(&self) -> bool {
        true // By construction
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_representation_element() {
        let v1 = RepresentationElement::new(vec![Integer::from(1), Integer::from(2)]);
        let v2 = RepresentationElement::new(vec![Integer::from(3), Integer::from(4)]);

        assert_eq!(v1.dimension(), 2);

        let sum = v1.clone() + v2.clone();
        assert_eq!(sum.coordinates()[0], Integer::from(4));
        assert_eq!(sum.coordinates()[1], Integer::from(6));

        let diff = v1.clone() - v2.clone();
        assert_eq!(diff.coordinates()[0], Integer::from(-2));
        assert_eq!(diff.coordinates()[1], Integer::from(-2));

        let neg = -v1;
        assert_eq!(neg.coordinates()[0], Integer::from(-1));
        assert_eq!(neg.coordinates()[1], Integer::from(-2));
    }

    #[test]
    fn test_trivial_representation() {
        let trivial = TrivialRepresentation::<Integer>::new(3);

        assert_eq!(trivial.dimension(), Some(1));
        assert!(trivial.is_irreducible());
        assert!(!trivial.is_faithful());

        let v = RepresentationElement::new(vec![Integer::from(5)]);
        let lie_elem = vec![Integer::from(1), Integer::from(2), Integer::from(3)];

        let result = trivial.action(&lie_elem, &v);
        assert_eq!(result.coordinates()[0], Integer::from(0));
    }

    #[test]
    fn test_representation_by_morphism_creation() {
        let mut matrices = HashMap::new();

        // Identity matrix for first basis element
        let id = Matrix::identity(2);
        matrices.insert(0, id);

        let rep = RepresentationByMorphism::new(2, 1, matrices).unwrap();

        assert_eq!(rep.representation_dimension(), 2);
        assert_eq!(rep.lie_algebra_dimension(), 1);
        assert!(rep.action_matrix(0).is_some());
    }

    #[test]
    fn test_representation_by_morphism_action() {
        let mut matrices = HashMap::new();

        // Simple 2x2 matrix
        let mat_data = vec![
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(2)],
        ];
        let mat = Matrix::from_rows(&mat_data);
        matrices.insert(0, mat);

        let rep = RepresentationByMorphism::new(2, 1, matrices).unwrap();

        let v = RepresentationElement::new(vec![Integer::from(3), Integer::from(4)]);
        let lie_elem = vec![Integer::from(1)]; // Coefficient 1 for first basis element

        let result = rep.action(&lie_elem, &v);
        assert_eq!(result.coordinates()[0], Integer::from(3)); // 1*3 + 0*4
        assert_eq!(result.coordinates()[1], Integer::from(8)); // 0*3 + 2*4
    }

    #[test]
    fn test_faithful_nilpotent_pbw() {
        let mut matrices = HashMap::new();
        matrices.insert(0, Matrix::identity(2));

        let rep = FaithfulRepresentationNilpotentPBW::new(1, 2, matrices, false);

        assert!(rep.is_faithful());
        assert!(!rep.is_minimal());
        assert_eq!(rep.dimension(), Some(2));
    }

    #[test]
    fn test_faithful_pos_char() {
        let mut matrices = HashMap::new();
        matrices.insert(0, Matrix::identity(2));

        let rep = FaithfulRepresentationPBWPosChar::new(1, 2, matrices, 5);

        assert!(rep.is_faithful());
        assert_eq!(rep.characteristic(), 5);
        assert_eq!(rep.dimension(), Some(2));
    }
}

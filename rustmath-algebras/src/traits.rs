//! Core traits for algebraic structures
//!
//! Defines the fundamental traits used across all algebra implementations.

use rustmath_core::{Ring, MathError, Result};
use std::fmt::{Debug, Display};

/// An algebra over a ring R
///
/// An algebra is a ring that is also a module over a base ring R,
/// with compatibility between the ring structure and the module structure.
pub trait Algebra<R: Ring>: Ring {
    /// The base ring over which this algebra is defined
    fn base_ring() -> R;

    /// Scalar multiplication by an element of the base ring
    fn scalar_mul(&self, scalar: &R) -> Self;

    /// The dimension of the algebra (if finite dimensional)
    fn dimension() -> Option<usize> {
        None
    }
}

/// An algebra with a basis (finite dimensional)
pub trait AlgebraWithBasis<R: Ring>: Algebra<R> {
    /// Type representing basis elements
    type BasisElement: Clone + Debug + PartialEq;

    /// Get a basis for the algebra
    fn basis() -> Vec<Self::BasisElement>;

    /// Express an algebra element as a linear combination of basis elements
    fn to_basis_coords(&self) -> Vec<R>;

    /// Construct an algebra element from basis coordinates
    fn from_basis_coords(coords: Vec<R>) -> Result<Self>;

    /// The structure constants: basis[i] * basis[j] = sum_k c[i][j][k] * basis[k]
    fn structure_constants() -> Vec<Vec<Vec<R>>>;
}

/// A graded algebra
pub trait GradedAlgebra<R: Ring>: Algebra<R> {
    /// Get the degree/grade of this element
    fn degree(&self) -> i32;

    /// Get the homogeneous component of a given degree
    fn homogeneous_component(&self, degree: i32) -> Self;
}

/// A free algebra (non-commutative polynomial algebra)
pub trait FreeAlgebra<R: Ring>: Algebra<R> {
    /// Type representing generators
    type Generator: Clone + Debug + PartialEq;

    /// Get the generators of the algebra
    fn generators() -> Vec<Self::Generator>;

    /// Get the number of generators
    fn rank() -> usize;
}

/// A quotient algebra A/I where I is an ideal
pub trait QuotientAlgebra<A: Ring>: Ring {
    /// The ambient algebra
    type Ambient: Ring;

    /// Lift an element to the ambient algebra
    fn lift(&self) -> Self::Ambient;

    /// Reduce an element of the ambient algebra to the quotient
    fn reduce(element: Self::Ambient) -> Self;
}

/// A Lie algebra
pub trait LieAlgebra<R: Ring>: Clone + Debug {
    /// The Lie bracket [x, y]
    fn bracket(&self, other: &Self) -> Self;

    /// Check if this is zero
    fn is_zero(&self) -> bool;

    /// The zero element
    fn zero() -> Self;

    /// Scalar multiplication
    fn scalar_mul(&self, scalar: &R) -> Self;
}

/// An associative algebra (redundant with Algebra but makes intent clear)
pub trait AssociativeAlgebra<R: Ring>: Algebra<R> {}

/// A commutative algebra
pub trait CommutativeAlgebra<R: Ring>: Algebra<R> {}

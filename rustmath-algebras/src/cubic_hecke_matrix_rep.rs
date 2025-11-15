//! Cubic Hecke Algebra Matrix Representations
//!
//! This module implements matrix representations of cubic Hecke algebras,
//! providing concrete realizations of the abstract algebra elements as
//! matrices over appropriate rings.
//!
//! The matrix representations are essential for:
//! - Computing with cubic Hecke algebra elements efficiently
//! - Studying irreducible representations
//! - Calculating invariants and specializations
//!
//! Corresponds to sage.algebras.hecke_algebras.cubic_hecke_matrix_rep
//!
//! # References
//!
//! - Marin, I. "The cubic Hecke algebra on at most 5 strands" (2015)
//! - Brav, C. and Thomas, H. "Braid groups and Kleinian singularities" (2011)

use rustmath_core::{Ring, Field};
use rustmath_matrix::Matrix;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Generator sign in the braid group
///
/// Represents whether a braid generator appears with positive or negative exponent.
///
/// # Variants
///
/// - `Pos`: Positive generator s_i
/// - `Neg`: Negative generator s_i^{-1}
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_matrix_rep::GenSign;
/// let pos = GenSign::Pos;
/// let neg = GenSign::Neg;
/// assert_eq!(pos.invert(), neg);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenSign {
    /// Positive generator
    Pos,
    /// Negative generator (inverse)
    Neg,
}

impl GenSign {
    /// Invert the sign
    pub fn invert(&self) -> Self {
        match self {
            GenSign::Pos => GenSign::Neg,
            GenSign::Neg => GenSign::Pos,
        }
    }

    /// Convert to integer (1 for positive, -1 for negative)
    pub fn to_int(&self) -> i32 {
        match self {
            GenSign::Pos => 1,
            GenSign::Neg => -1,
        }
    }

    /// Create from integer
    pub fn from_int(i: i32) -> Self {
        if i >= 0 {
            GenSign::Pos
        } else {
            GenSign::Neg
        }
    }
}

impl Display for GenSign {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GenSign::Pos => write!(f, "+"),
            GenSign::Neg => write!(f, "-"),
        }
    }
}

/// Type of representation for the cubic Hecke algebra
///
/// Different representation types correspond to different specializations
/// of the cubic Hecke algebra parameters.
///
/// # Variants
///
/// - `Generic`: Generic representation over the ring of definition
/// - `Split`: Representation over the splitting field
/// - `Regular`: Regular representation
/// - `Irreducible`: Irreducible representation (specific type)
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_matrix_rep::RepresentationType;
/// let rep_type = RepresentationType::Split;
/// assert_eq!(rep_type, RepresentationType::Split);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepresentationType {
    /// Generic representation over ring of definition
    Generic,
    /// Representation over splitting field
    Split,
    /// Regular representation
    Regular,
    /// Irreducible representation
    Irreducible,
}

impl Display for RepresentationType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RepresentationType::Generic => write!(f, "Generic"),
            RepresentationType::Split => write!(f, "Split"),
            RepresentationType::Regular => write!(f, "Regular"),
            RepresentationType::Irreducible => write!(f, "Irreducible"),
        }
    }
}

/// Absolute irreducible representations of the cubic Hecke algebra
///
/// Enumerates the distinct irreducible representations up to isomorphism.
/// For the cubic Hecke algebra on n strands, the number and types of
/// irreducible representations depend on n.
///
/// # Mathematical Background
///
/// The irreducible representations of H(n) are indexed in various ways.
/// This enumeration provides a canonical labeling.
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_matrix_rep::AbsIrreducibleRep;
/// let rep = AbsIrreducibleRep::Trivial;
/// assert_eq!(rep.dimension(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsIrreducibleRep {
    /// Trivial representation (dimension 1)
    Trivial,
    /// Sign representation (dimension 1)
    Sign,
    /// Standard representation (dimension n-1)
    Standard,
    /// Reflection representation
    Reflection,
    /// Other irreducible representations (indexed)
    Other(usize),
}

impl AbsIrreducibleRep {
    /// Get the dimension of this representation
    ///
    /// For indexed representations, returns an estimate
    pub fn dimension(&self) -> usize {
        match self {
            AbsIrreducibleRep::Trivial => 1,
            AbsIrreducibleRep::Sign => 1,
            AbsIrreducibleRep::Standard => 2, // Depends on n
            AbsIrreducibleRep::Reflection => 2,
            AbsIrreducibleRep::Other(i) => *i, // Placeholder
        }
    }

    /// Check if this is a one-dimensional representation
    pub fn is_one_dimensional(&self) -> bool {
        matches!(self, AbsIrreducibleRep::Trivial | AbsIrreducibleRep::Sign)
    }
}

impl Display for AbsIrreducibleRep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AbsIrreducibleRep::Trivial => write!(f, "Trivial"),
            AbsIrreducibleRep::Sign => write!(f, "Sign"),
            AbsIrreducibleRep::Standard => write!(f, "Standard"),
            AbsIrreducibleRep::Reflection => write!(f, "Reflection"),
            AbsIrreducibleRep::Other(i) => write!(f, "Irreducible({})", i),
        }
    }
}

/// Matrix space for cubic Hecke algebra representations
///
/// A specialized matrix space that knows about the cubic Hecke algebra
/// structure. This allows for efficient computation with representation
/// matrices.
///
/// # Type Parameters
///
/// * `R` - The base ring for matrix entries
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_matrix_rep::CubicHeckeMatrixSpace;
/// # use rustmath_integers::Integer;
/// let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(3);
/// assert_eq!(space.dimension(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct CubicHeckeMatrixSpace<R: Ring> {
    /// Dimension of the matrices
    dimension: usize,
    /// Number of strands in the underlying braid group
    strands: usize,
    /// Type of representation
    rep_type: RepresentationType,
    /// Base ring marker
    _phantom: PhantomData<R>,
}

impl<R: Ring> CubicHeckeMatrixSpace<R> {
    /// Create a new matrix space for cubic Hecke representations
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of the matrices
    pub fn new(dimension: usize) -> Self {
        CubicHeckeMatrixSpace {
            dimension,
            strands: dimension + 1, // Heuristic
            rep_type: RepresentationType::Generic,
            _phantom: PhantomData,
        }
    }

    /// Create a matrix space for a specific representation type
    pub fn with_type(dimension: usize, rep_type: RepresentationType) -> Self {
        CubicHeckeMatrixSpace {
            dimension,
            strands: dimension + 1,
            rep_type,
            _phantom: PhantomData,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of strands
    pub fn strands(&self) -> usize {
        self.strands
    }

    /// Get the representation type
    pub fn rep_type(&self) -> RepresentationType {
        self.rep_type
    }

    /// Create the zero matrix
    pub fn zero(&self) -> CubicHeckeMatrixRep<R>
    where
        R: From<i64> + Clone,
    {
        CubicHeckeMatrixRep {
            matrix: Matrix::zeros(self.dimension, self.dimension),
            space: self.clone(),
        }
    }

    /// Create the identity matrix
    pub fn identity(&self) -> CubicHeckeMatrixRep<R>
    where
        R: From<i64> + Clone,
    {
        CubicHeckeMatrixRep {
            matrix: Matrix::identity(self.dimension),
            space: self.clone(),
        }
    }
}

impl<R: Ring> Display for CubicHeckeMatrixSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Matrix space for {} representation of dimension {}",
            self.rep_type, self.dimension
        )
    }
}

/// Matrix representation of a cubic Hecke algebra element
///
/// Represents a cubic Hecke algebra element as a matrix over the base ring.
/// This is a concrete realization that allows for explicit computation.
///
/// # Type Parameters
///
/// * `R` - The base ring for matrix entries
///
/// # Examples
///
/// ```
/// # use rustmath_algebras::cubic_hecke_matrix_rep::{CubicHeckeMatrixSpace, CubicHeckeMatrixRep};
/// # use rustmath_integers::Integer;
/// let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);
/// let identity = space.identity();
/// assert!(identity.is_identity());
/// ```
#[derive(Debug, Clone)]
pub struct CubicHeckeMatrixRep<R: Ring> {
    /// The underlying matrix
    matrix: Matrix<R>,
    /// The matrix space this belongs to
    space: CubicHeckeMatrixSpace<R>,
}

impl<R: Ring + Clone> CubicHeckeMatrixRep<R> {
    /// Create a new matrix representation
    ///
    /// # Arguments
    ///
    /// * `matrix` - The underlying matrix
    /// * `space` - The matrix space
    pub fn new(matrix: Matrix<R>, space: CubicHeckeMatrixSpace<R>) -> Self {
        CubicHeckeMatrixRep { matrix, space }
    }

    /// Get the underlying matrix
    pub fn matrix(&self) -> &Matrix<R> {
        &self.matrix
    }

    /// Get the matrix space
    pub fn space(&self) -> &CubicHeckeMatrixSpace<R> {
        &self.space
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.space.dimension()
    }

    /// Check if this is the zero matrix
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq + From<i64>,
    {
        let zero = R::from(0);
        for i in 0..self.dimension() {
            for j in 0..self.dimension() {
                if let Ok(val) = self.matrix.get(i, j) {
                    if *val != zero {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if this is the identity matrix
    pub fn is_identity(&self) -> bool
    where
        R: PartialEq + From<i64>,
    {
        let zero = R::from(0);
        let one = R::from(1);

        for i in 0..self.dimension() {
            for j in 0..self.dimension() {
                if let Ok(val) = self.matrix.get(i, j) {
                    let expected = if i == j { &one } else { &zero };
                    if val != expected {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Matrix multiplication
    pub fn multiply(&self, other: &Self) -> Result<Self, String>
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64>,
    {
        if self.dimension() != other.dimension() {
            return Err("Matrix dimensions must match".to_string());
        }

        let result = (self.matrix.clone() * other.matrix.clone())
            .map_err(|e| format!("Matrix multiplication failed: {:?}", e))?;
        Ok(CubicHeckeMatrixRep {
            matrix: result,
            space: self.space.clone(),
        })
    }

    /// Matrix addition
    pub fn add(&self, other: &Self) -> Result<Self, String>
    where
        R: std::ops::Add<Output = R>,
    {
        if self.dimension() != other.dimension() {
            return Err("Matrix dimensions must match".to_string());
        }

        let result = (self.matrix.clone() + other.matrix.clone())
            .map_err(|e| format!("Matrix addition failed: {:?}", e))?;
        Ok(CubicHeckeMatrixRep {
            matrix: result,
            space: self.space.clone(),
        })
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R>,
    {
        let result = self.matrix.scalar_mul(scalar);
        CubicHeckeMatrixRep {
            matrix: result,
            space: self.space.clone(),
        }
    }

    /// Compute the trace
    pub fn trace(&self) -> Result<R, String>
    where
        R: std::ops::Add<Output = R> + From<i64>,
    {
        self.matrix.trace()
            .map_err(|e| format!("Trace computation failed: {:?}", e))
    }

    /// Compute the determinant
    pub fn determinant(&self) -> Result<R, String>
    where
        R: std::ops::Add<Output = R>
            + std::ops::Sub<Output = R>
            + std::ops::Mul<Output = R>
            + std::ops::Neg<Output = R>
            + From<i64>
            + PartialEq,
    {
        self.matrix.determinant()
            .map_err(|e| format!("Determinant computation failed: {:?}", e))
    }

    /// Compute matrix power
    pub fn power(&self, n: usize) -> Result<Self, String>
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + From<i64>,
    {
        if n == 0 {
            return Ok(self.space.identity());
        }

        let mut result = self.clone();
        for _ in 1..n {
            result = result.multiply(self)?;
        }
        Ok(result)
    }

    /// Representation of a braid generator
    ///
    /// Returns the matrix representation of the i-th braid generator
    /// (or its inverse if sign is negative).
    pub fn generator_matrix(
        space: &CubicHeckeMatrixSpace<R>,
        i: usize,
        sign: GenSign,
    ) -> Result<Self, String>
    where
        R: From<i64> + Clone,
    {
        if i >= space.strands {
            return Err(format!(
                "Generator index {} out of range for {} strands",
                i, space.strands
            ));
        }

        // For now, return a simple placeholder matrix
        // In a full implementation, this would compute the actual
        // representation matrix based on the representation type

        // Placeholder: identity matrix
        let matrix = Matrix::identity(space.dimension);

        Ok(CubicHeckeMatrixRep {
            matrix,
            space: space.clone(),
        })
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for CubicHeckeMatrixRep<R> {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl<R: Ring + Clone + PartialEq> Eq for CubicHeckeMatrixRep<R> {}

impl<R: Ring + Clone + Display> Display for CubicHeckeMatrixRep<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cubic Hecke matrix representation:\n{}", self.matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_gen_sign() {
        let pos = GenSign::Pos;
        let neg = GenSign::Neg;

        assert_eq!(pos.invert(), neg);
        assert_eq!(neg.invert(), pos);
        assert_eq!(pos.to_int(), 1);
        assert_eq!(neg.to_int(), -1);
    }

    #[test]
    fn test_gen_sign_from_int() {
        assert_eq!(GenSign::from_int(1), GenSign::Pos);
        assert_eq!(GenSign::from_int(-1), GenSign::Neg);
        assert_eq!(GenSign::from_int(5), GenSign::Pos);
        assert_eq!(GenSign::from_int(-10), GenSign::Neg);
    }

    #[test]
    fn test_representation_type() {
        let gen = RepresentationType::Generic;
        let split = RepresentationType::Split;

        assert_eq!(gen, RepresentationType::Generic);
        assert_ne!(gen, split);
    }

    #[test]
    fn test_abs_irreducible_rep() {
        let triv = AbsIrreducibleRep::Trivial;
        let sign = AbsIrreducibleRep::Sign;
        let std = AbsIrreducibleRep::Standard;

        assert_eq!(triv.dimension(), 1);
        assert_eq!(sign.dimension(), 1);
        assert!(triv.is_one_dimensional());
        assert!(sign.is_one_dimensional());
        assert!(!std.is_one_dimensional());
    }

    #[test]
    fn test_matrix_space() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(3);
        assert_eq!(space.dimension(), 3);
        assert_eq!(space.rep_type(), RepresentationType::Generic);
    }

    #[test]
    fn test_matrix_space_with_type() {
        let space: CubicHeckeMatrixSpace<Integer> =
            CubicHeckeMatrixSpace::with_type(2, RepresentationType::Split);
        assert_eq!(space.dimension(), 2);
        assert_eq!(space.rep_type(), RepresentationType::Split);
    }

    #[test]
    fn test_zero_matrix() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);
        let zero = space.zero();
        assert!(zero.is_zero());
        assert!(!zero.is_identity());
    }

    #[test]
    fn test_identity_matrix() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(3);
        let identity = space.identity();
        assert!(identity.is_identity());
        assert!(!identity.is_zero());
    }

    #[test]
    fn test_matrix_operations() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);
        let id = space.identity();
        let zero = space.zero();

        // id + zero = id
        let sum = id.add(&zero).unwrap();
        assert!(sum.is_identity());

        // id * id = id
        let product = id.multiply(&id).unwrap();
        assert!(product.is_identity());
    }

    #[test]
    fn test_scalar_multiplication() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);
        let id = space.identity();

        let scaled = id.scalar_mul(&Integer::from(3));
        assert!(!scaled.is_identity());
        assert!(!scaled.is_zero());
    }

    #[test]
    fn test_trace() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(3);
        let id = space.identity();

        let trace = id.trace().unwrap();
        assert_eq!(trace, Integer::from(3));
    }

    #[test]
    fn test_determinant() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);
        let id = space.identity();

        let det = id.determinant().unwrap();
        assert_eq!(det, Integer::from(1));
    }

    #[test]
    fn test_power() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);
        let id = space.identity();

        let squared = id.power(2).unwrap();
        assert!(squared.is_identity());

        let zero_power = id.power(0).unwrap();
        assert!(zero_power.is_identity());
    }

    #[test]
    fn test_generator_matrix() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);

        let gen = CubicHeckeMatrixRep::generator_matrix(&space, 0, GenSign::Pos);
        assert!(gen.is_ok());

        let inv_gen = CubicHeckeMatrixRep::generator_matrix(&space, 0, GenSign::Neg);
        assert!(inv_gen.is_ok());
    }

    #[test]
    fn test_generator_matrix_out_of_range() {
        let space: CubicHeckeMatrixSpace<Integer> = CubicHeckeMatrixSpace::new(2);

        let result = CubicHeckeMatrixRep::generator_matrix(&space, 10, GenSign::Pos);
        assert!(result.is_err());
    }
}

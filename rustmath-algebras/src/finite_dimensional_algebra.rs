//! Finite Dimensional Algebra Implementation
//!
//! A finite dimensional algebra over a ring R has a finite basis as an R-module.
//! Elements are represented as linear combinations of basis elements.
//! Multiplication is determined by structure constants.
//!
//! Corresponds to sage.algebras.finite_dimensional_algebras.finite_dimensional_algebra

use rustmath_core::{Ring, MathError, Result};
use std::fmt::{self, Display};
use std::ops::{Add, Mul, Neg, Sub};
use crate::traits::Algebra;

/// An element of a finite dimensional algebra
///
/// Represented as a vector of coordinates with respect to a fixed basis
#[derive(Clone, Debug)]
pub struct FiniteDimensionalAlgebraElement<R: Ring> {
    /// Coordinates with respect to the basis
    coords: Vec<R>,
    /// Reference to the structure constants (indices: [basis_i][basis_j][result_k])
    /// Stored as a flattened array for efficiency
    structure_constants: Vec<Vec<Vec<R>>>,
}

impl<R: Ring> FiniteDimensionalAlgebraElement<R> {
    /// Create a new element from coordinates
    pub fn new(coords: Vec<R>, structure_constants: Vec<Vec<Vec<R>>>) -> Result<Self> {
        let dim = structure_constants.len();
        if coords.len() != dim {
            return Err(MathError::InvalidArgument(format!(
                "Dimension mismatch: expected {} but got {}",
                dim,
                coords.len()
            )));
        }
        Ok(Self { coords, structure_constants })
    }

    /// Create a zero element
    pub fn zero(dim: usize, structure_constants: Vec<Vec<Vec<R>>>) -> Self {
        Self {
            coords: vec![R::zero(); dim],
            structure_constants,
        }
    }

    /// Create a basis element (all zeros except one 1)
    pub fn basis_element(
        index: usize,
        dim: usize,
        structure_constants: Vec<Vec<Vec<R>>>,
    ) -> Result<Self> {
        if index >= dim {
            return Err(MathError::InvalidArgument(format!(
                "Index out of bounds: index {} >= size {}",
                index, dim
            )));
        }
        let mut coords = vec![R::zero(); dim];
        coords[index] = R::one();
        Ok(Self { coords, structure_constants })
    }

    /// Get the dimension of the algebra
    pub fn dimension(&self) -> usize {
        self.coords.len()
    }

    /// Get the coordinates
    pub fn coordinates(&self) -> &[R] {
        &self.coords
    }

    /// Get a mutable reference to the coordinates
    pub fn coordinates_mut(&mut self) -> &mut [R] {
        &mut self.coords
    }

    /// Check if all coordinates are zero
    fn is_zero_coords(&self) -> bool {
        self.coords.iter().all(|c| c.is_zero())
    }

    /// Reconstruct an element from serialized data (unpickle)
    ///
    /// This function recreates a FiniteDimensionalAlgebraElement from its component parts,
    /// bypassing the normal validation. This is useful for deserialization where the
    /// element was previously validated.
    ///
    /// # Arguments
    /// * `coords` - The vector representation of the element
    /// * `structure_constants` - The structure constants of the parent algebra
    ///
    /// # Returns
    /// A reconstructed algebra element
    ///
    /// # Note
    /// This function is analogous to SageMath's unpickle_FiniteDimensionalAlgebraElement
    /// which reconstructs elements during Python's unpickling process.
    ///
    /// Corresponds to sage.algebras.finite_dimensional_algebras.finite_dimensional_algebra_element.unpickle_FiniteDimensionalAlgebraElement
    pub fn unpickle(coords: Vec<R>, structure_constants: Vec<Vec<Vec<R>>>) -> Self {
        Self {
            coords,
            structure_constants,
        }
    }
}

impl<R: Ring> PartialEq for FiniteDimensionalAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.coords.len() != other.coords.len() {
            return false;
        }
        self.coords.iter().zip(&other.coords).all(|(a, b)| a == b)
    }
}

impl<R: Ring> Display for FiniteDimensionalAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (i, coord) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, "]")
    }
}

impl<R: Ring> Add for FiniteDimensionalAlgebraElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "Cannot add elements of different dimensions"
        );

        let coords = self
            .coords
            .iter()
            .zip(&other.coords)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Self {
            coords,
            structure_constants: self.structure_constants,
        }
    }
}

impl<R: Ring> Sub for FiniteDimensionalAlgebraElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<R: Ring> Neg for FiniteDimensionalAlgebraElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coords = self.coords.iter().map(|c| -c.clone()).collect();
        Self {
            coords,
            structure_constants: self.structure_constants,
        }
    }
}

impl<R: Ring> Mul for FiniteDimensionalAlgebraElement<R> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "Cannot multiply elements of different dimensions"
        );

        let dim = self.dimension();
        let mut result_coords = vec![R::zero(); dim];

        // Multiply using structure constants:
        // (sum_i a_i * e_i) * (sum_j b_j * e_j) = sum_i sum_j a_i * b_j * (e_i * e_j)
        // where e_i * e_j = sum_k c[i][j][k] * e_k
        for i in 0..dim {
            for j in 0..dim {
                let coeff = self.coords[i].clone() * other.coords[j].clone();
                if !coeff.is_zero() {
                    for k in 0..dim {
                        let structure_coeff = self.structure_constants[i][j][k].clone();
                        result_coords[k] =
                            result_coords[k].clone() + coeff.clone() * structure_coeff;
                    }
                }
            }
        }

        Self {
            coords: result_coords,
            structure_constants: self.structure_constants,
        }
    }
}

impl<R: Ring> Ring for FiniteDimensionalAlgebraElement<R> {
    fn zero() -> Self {
        // This is a bit awkward - we need structure constants
        // In practice, elements should be created through the algebra structure
        Self {
            coords: Vec::new(),
            structure_constants: Vec::new(),
        }
    }

    fn one() -> Self {
        // Similar issue - need proper context
        Self {
            coords: Vec::new(),
            structure_constants: Vec::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.is_zero_coords()
    }

    fn is_one(&self) -> bool {
        // Typically the first basis element is the identity
        // but this depends on the algebra structure
        if self.coords.is_empty() {
            return false;
        }
        self.coords[0].is_one() && self.coords[1..].iter().all(|c| c.is_zero())
    }
}

impl<R: Ring> Algebra<R> for FiniteDimensionalAlgebraElement<R> {
    fn base_ring() -> R {
        R::zero()
    }

    fn scalar_mul(&self, scalar: &R) -> Self {
        let coords = self
            .coords
            .iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();
        Self {
            coords,
            structure_constants: self.structure_constants.clone(),
        }
    }

    fn dimension() -> Option<usize> {
        None // Would need to be stored in the algebra structure
    }
}

/// A finite dimensional algebra structure
///
/// Stores the dimension, basis names, and structure constants
pub struct FiniteDimensionalAlgebra<R: Ring> {
    /// Dimension of the algebra
    dimension: usize,
    /// Names of basis elements
    basis_names: Vec<String>,
    /// Structure constants: basis[i] * basis[j] = sum_k constants[i][j][k] * basis[k]
    structure_constants: Vec<Vec<Vec<R>>>,
}

impl<R: Ring> FiniteDimensionalAlgebra<R> {
    /// Create a new finite dimensional algebra from structure constants
    ///
    /// # Arguments
    /// * `structure_constants` - A 3D array where constants[i][j][k] is the coefficient
    ///   of basis[k] in the product basis[i] * basis[j]
    pub fn new(structure_constants: Vec<Vec<Vec<R>>>) -> Result<Self> {
        let dimension = structure_constants.len();

        // Validate structure constants
        for (_i, row) in structure_constants.iter().enumerate() {
            if row.len() != dimension {
                return Err(MathError::InvalidArgument(format!(
                    "Dimension mismatch: expected {} but got {}",
                    dimension,
                    row.len()
                )));
            }
            for (_j, col) in row.iter().enumerate() {
                if col.len() != dimension {
                    return Err(MathError::InvalidArgument(format!(
                        "Dimension mismatch: expected {} but got {}",
                        dimension,
                        col.len()
                    )));
                }
            }
        }

        let basis_names = (0..dimension).map(|i| format!("e{}", i)).collect();

        Ok(Self {
            dimension,
            basis_names,
            structure_constants,
        })
    }

    /// Create a new finite dimensional algebra with named basis
    pub fn with_basis_names(
        basis_names: Vec<String>,
        structure_constants: Vec<Vec<Vec<R>>>,
    ) -> Result<Self> {
        let dimension = basis_names.len();

        if structure_constants.len() != dimension {
            return Err(MathError::InvalidArgument(format!(
                "Dimension mismatch: expected {} but got {}",
                dimension,
                structure_constants.len()
            )));
        }

        Ok(Self {
            dimension,
            basis_names,
            structure_constants,
        })
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the basis names
    pub fn basis_names(&self) -> &[String] {
        &self.basis_names
    }

    /// Get the structure constants
    pub fn structure_constants(&self) -> &Vec<Vec<Vec<R>>> {
        &self.structure_constants
    }

    /// Create the zero element
    pub fn zero(&self) -> FiniteDimensionalAlgebraElement<R> {
        FiniteDimensionalAlgebraElement::zero(
            self.dimension,
            self.structure_constants.clone(),
        )
    }

    /// Create the identity element (assumes first basis element is identity)
    pub fn one(&self) -> FiniteDimensionalAlgebraElement<R> {
        let mut coords = vec![R::zero(); self.dimension];
        coords[0] = R::one();
        FiniteDimensionalAlgebraElement {
            coords,
            structure_constants: self.structure_constants.clone(),
        }
    }

    /// Get a basis element
    pub fn basis_element(&self, index: usize) -> Result<FiniteDimensionalAlgebraElement<R>> {
        FiniteDimensionalAlgebraElement::basis_element(
            index,
            self.dimension,
            self.structure_constants.clone(),
        )
    }

    /// Create an element from coordinates
    pub fn element(&self, coords: Vec<R>) -> Result<FiniteDimensionalAlgebraElement<R>> {
        FiniteDimensionalAlgebraElement::new(coords, self.structure_constants.clone())
    }

    /// Check if the algebra is associative
    pub fn is_associative(&self) -> bool {
        // Check if (e_i * e_j) * e_k = e_i * (e_j * e_k) for all i, j, k
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    // Compute (e_i * e_j) * e_k
                    let mut lhs = vec![R::zero(); self.dimension];
                    for m in 0..self.dimension {
                        let c_ijm = &self.structure_constants[i][j][m];
                        for n in 0..self.dimension {
                            let c_mkn = &self.structure_constants[m][k][n];
                            lhs[n] = lhs[n].clone() + c_ijm.clone() * c_mkn.clone();
                        }
                    }

                    // Compute e_i * (e_j * e_k)
                    let mut rhs = vec![R::zero(); self.dimension];
                    for m in 0..self.dimension {
                        let c_jkm = &self.structure_constants[j][k][m];
                        for n in 0..self.dimension {
                            let c_imn = &self.structure_constants[i][m][n];
                            rhs[n] = rhs[n].clone() + c_imn.clone() * c_jkm.clone();
                        }
                    }

                    // Check if lhs == rhs
                    if !lhs.iter().zip(&rhs).all(|(a, b)| a == b) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if the algebra has a unit (identity element)
    pub fn has_unit(&self) -> Option<Vec<R>> {
        // Look for an element e such that e * basis[i] = basis[i] * e = basis[i] for all i
        // This means: sum_k e[k] * c[k][i][j] = delta[i][j] for all i,j

        // Try the first basis element as identity (common convention)
        let mut e = vec![R::zero(); self.dimension];
        e[0] = R::one();

        // Check if this is an identity
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                // e * basis[i] should give basis[i]
                let mut result = R::zero();
                for k in 0..self.dimension {
                    result = result + e[k].clone() * self.structure_constants[k][i][j].clone();
                }

                let expected = if i == j { R::one() } else { R::zero() };
                if result != expected {
                    return None; // No unit element found
                }
            }
        }

        Some(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_matrix_algebra_2x2() {
        // Create the algebra of 2x2 matrices represented as a 4-dimensional algebra
        // Basis: E11, E12, E21, E22 (standard matrix units)
        // E_ij * E_kl = delta_jk * E_il

        let dim = 4;
        let mut constants = vec![vec![vec![Integer::zero(); dim]; dim]; dim];

        // E11 * E11 = E11
        constants[0][0][0] = Integer::one();
        // E11 * E12 = E12
        constants[0][1][1] = Integer::one();
        // E12 * E21 = E11
        constants[1][2][0] = Integer::one();
        // E12 * E22 = E12
        constants[1][3][1] = Integer::one();
        // E21 * E11 = E21
        constants[2][0][2] = Integer::one();
        // E21 * E12 = E22
        constants[2][1][3] = Integer::one();
        // E22 * E21 = E21
        constants[3][2][2] = Integer::one();
        // E22 * E22 = E22
        constants[3][3][3] = Integer::one();

        let algebra = FiniteDimensionalAlgebra::new(constants).unwrap();
        assert_eq!(algebra.dimension(), 4);

        // The algebra should be associative
        assert!(algebra.is_associative());
    }

    #[test]
    fn test_algebra_arithmetic() {
        // Simple 2D algebra: e0 * e0 = e0, e0 * e1 = e1, e1 * e0 = e1, e1 * e1 = 0
        let mut constants = vec![vec![vec![Integer::zero(); 2]; 2]; 2];
        constants[0][0][0] = Integer::one(); // e0 * e0 = e0
        constants[0][1][1] = Integer::one(); // e0 * e1 = e1
        constants[1][0][1] = Integer::one(); // e1 * e0 = e1
        // e1 * e1 = 0 (all zeros)

        let algebra = FiniteDimensionalAlgebra::new(constants).unwrap();

        let e0 = algebra.basis_element(0).unwrap();
        let e1 = algebra.basis_element(1).unwrap();

        // Test addition
        let sum = e0.clone() + e1.clone();
        assert_eq!(sum.coords, vec![Integer::one(), Integer::one()]);

        // Test multiplication
        let prod = e0.clone() * e1.clone();
        assert_eq!(prod.coords, vec![Integer::zero(), Integer::one()]);
    }

    #[test]
    fn test_unpickle_element() {
        // Create structure constants for a simple algebra
        let mut constants = vec![vec![vec![Integer::zero(); 2]; 2]; 2];
        constants[0][0][0] = Integer::one();
        constants[0][1][1] = Integer::one();
        constants[1][0][1] = Integer::one();

        // Create element normally
        let algebra = FiniteDimensionalAlgebra::new(constants.clone()).unwrap();
        let elem = algebra.element(vec![Integer::from(3), Integer::from(4)]).unwrap();

        // Reconstruct the same element using unpickle
        let unpickled = FiniteDimensionalAlgebraElement::unpickle(
            vec![Integer::from(3), Integer::from(4)],
            constants.clone(),
        );

        // Both elements should be equal
        assert_eq!(elem, unpickled);
        assert_eq!(unpickled.coordinates(), &[Integer::from(3), Integer::from(4)]);

        // Test that operations work on unpickled elements
        let elem2 = FiniteDimensionalAlgebraElement::unpickle(
            vec![Integer::from(1), Integer::from(2)],
            constants,
        );

        let sum = unpickled + elem2;
        assert_eq!(sum.coordinates(), &[Integer::from(4), Integer::from(6)]);
    }
}

//! Heisenberg Lie Algebras
//!
//! The Heisenberg Lie algebra is a nilpotent Lie algebra with basis elements:
//! - p_0, ..., p_{n-1} (position-like generators)
//! - q_0, ..., q_{n-1} (momentum-like generators)
//! - z (central element)
//!
//! The Lie bracket satisfies the canonical commutation relations:
//! - [p_i, q_j] = δ_ij * z (Kronecker delta)
//! - [p_i, z] = [q_i, z] = 0
//! - [p_i, p_j] = [q_i, q_j] = 0
//!
//! Properties:
//! - Dimension: 2n + 1 (for rank n)
//! - Nilpotent: 2-step nilpotent (lower central series terminates at step 2)
//! - Center: Spanned by z
//!
//! Corresponds to sage.algebras.lie_algebras.heisenberg
//!
//! References:
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - Hall, B. "Lie Groups, Lie Algebras, and Representations" (2015)

use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// Finite-Dimensional Heisenberg Lie Algebra
///
/// The Heisenberg algebra of rank n with basis:
/// - p_i (i = 0, ..., n-1): position generators
/// - q_i (i = 0, ..., n-1): momentum generators
/// - z: central element
///
/// The Lie bracket satisfies [p_i, q_j] = δ_ij * z, with all other brackets zero.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
///
/// # Examples
///
/// ```
/// # use rustmath_liealgebras::HeisenbergAlgebra;
/// let h3: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(3);
/// assert_eq!(h3.rank(), 3);
/// assert_eq!(h3.dimension(), 7); // 2*3 + 1
/// assert!(h3.is_nilpotent());
/// ```
pub struct HeisenbergAlgebra<R: Ring> {
    /// Rank of the algebra (number of p_i and q_i pairs)
    rank: usize,
    /// Coefficient ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> HeisenbergAlgebra<R> {
    /// Create a Heisenberg algebra of rank n
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank (number of p_i and q_i pairs)
    pub fn new(rank: usize) -> Self {
        HeisenbergAlgebra {
            rank,
            coefficient_ring: PhantomData,
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Dimension is 2n + 1 (n p's, n q's, 1 z)
    pub fn dimension(&self) -> usize {
        2 * self.rank + 1
    }

    /// Check if this is nilpotent (always true, 2-step nilpotent)
    pub fn is_nilpotent(&self) -> bool {
        true
    }

    /// Get the nilpotency step (always 2 for Heisenberg algebras)
    pub fn step(&self) -> usize {
        2
    }

    /// Check if this is solvable (always true, nilpotent implies solvable)
    pub fn is_solvable(&self) -> bool {
        true
    }

    /// Get the zero element
    pub fn zero(&self) -> HeisenbergAlgebraElement<R>
    where
        R: From<i64>,
    {
        HeisenbergAlgebraElement::zero(self.dimension())
    }

    /// Get the central element z
    pub fn z(&self) -> HeisenbergAlgebraElement<R>
    where
        R: From<i64>,
    {
        HeisenbergAlgebraElement::basis_element(2 * self.rank, self.dimension())
    }

    /// Get the i-th position generator p_i
    ///
    /// # Arguments
    ///
    /// * `i` - Index (must be < rank)
    pub fn p(&self, i: usize) -> HeisenbergAlgebraElement<R>
    where
        R: From<i64>,
    {
        if i >= self.rank {
            return HeisenbergAlgebraElement::zero(self.dimension());
        }
        HeisenbergAlgebraElement::basis_element(i, self.dimension())
    }

    /// Get the i-th momentum generator q_i
    ///
    /// # Arguments
    ///
    /// * `i` - Index (must be < rank)
    pub fn q(&self, i: usize) -> HeisenbergAlgebraElement<R>
    where
        R: From<i64>,
    {
        if i >= self.rank {
            return HeisenbergAlgebraElement::zero(self.dimension());
        }
        HeisenbergAlgebraElement::basis_element(self.rank + i, self.dimension())
    }

    /// Get all basis elements
    pub fn basis(&self) -> Vec<HeisenbergAlgebraElement<R>>
    where
        R: From<i64>,
    {
        (0..self.dimension())
            .map(|i| HeisenbergAlgebraElement::basis_element(i, self.dimension()))
            .collect()
    }

    /// Compute the Lie bracket of two basis elements
    ///
    /// Returns the structure coefficients as a HashMap
    pub fn bracket_on_basis(&self, i: usize, j: usize) -> HeisenbergAlgebraElement<R>
    where
        R: From<i64>,
    {
        // Basis ordering: p_0, ..., p_{n-1}, q_0, ..., q_{n-1}, z
        let n = self.rank;

        // If either is z (the central element), bracket is zero
        if i == 2 * n || j == 2 * n {
            return HeisenbergAlgebraElement::zero(self.dimension());
        }

        // Check if i is a p and j is a q (or vice versa)
        if i < n && j >= n && j < 2 * n {
            // [p_i, q_j] = δ_ij * z
            let p_idx = i;
            let q_idx = j - n;
            if p_idx == q_idx {
                // Return z (the last basis element)
                return self.z();
            }
        } else if j < n && i >= n && i < 2 * n {
            // [q_j, p_i] = -δ_ij * z
            let p_idx = j;
            let q_idx = i - n;
            if p_idx == q_idx {
                // Return -z
                let mut z = self.z();
                return z.negate();
            }
        }

        // All other brackets are zero
        HeisenbergAlgebraElement::zero(self.dimension())
    }

    /// Get the center of the algebra (spanned by z)
    pub fn center_basis(&self) -> Vec<HeisenbergAlgebraElement<R>>
    where
        R: From<i64>,
    {
        vec![self.z()]
    }

    /// Dimension of the center (always 1)
    pub fn center_dimension(&self) -> usize {
        1
    }
}

impl<R: Ring + Clone> Display for HeisenbergAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Heisenberg algebra of rank {}", self.rank)
    }
}

/// Element of a Heisenberg Lie algebra
///
/// Represented as a linear combination of basis elements:
/// c_0*p_0 + ... + c_{n-1}*p_{n-1} + d_0*q_0 + ... + d_{n-1}*q_{n-1} + e*z
#[derive(Clone)]
pub struct HeisenbergAlgebraElement<R: Ring> {
    /// Coefficients for each basis element
    coefficients: Vec<R>,
}

impl<R: Ring + Clone> HeisenbergAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: Vec<R>) -> Self {
        HeisenbergAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero(dimension: usize) -> Self
    where
        R: From<i64>,
    {
        HeisenbergAlgebraElement {
            coefficients: vec![R::from(0); dimension],
        }
    }

    /// Create a basis element
    pub fn basis_element(index: usize, dimension: usize) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = vec![R::from(0); dimension];
        if index < dimension {
            coefficients[index] = R::from(1);
        }
        HeisenbergAlgebraElement { coefficients }
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Get coefficient at index
    pub fn coefficient(&self, index: usize) -> Option<&R> {
        self.coefficients.get(index)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Dimension of the element
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    /// Add two elements
    pub fn add(&self, other: &Self) -> Self
    where
        R: std::ops::Add<Output = R>,
    {
        assert_eq!(self.dimension(), other.dimension());
        let coefficients = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        HeisenbergAlgebraElement { coefficients }
    }

    /// Negate an element
    pub fn negate(&mut self) -> Self
    where
        R: std::ops::Neg<Output = R>,
    {
        let coefficients = self
            .coefficients
            .iter()
            .map(|c| -c.clone())
            .collect();
        HeisenbergAlgebraElement { coefficients }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> Self
    where
        R: std::ops::Mul<Output = R>,
    {
        let coefficients = self
            .coefficients
            .iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();
        HeisenbergAlgebraElement { coefficients }
    }

    /// Lie bracket with another element
    ///
    /// Requires the parent algebra to compute properly
    pub fn bracket(&self, other: &Self, algebra: &HeisenbergAlgebra<R>) -> Self
    where
        R: From<i64> + std::ops::Add<Output = R> + std::ops::Mul<Output = R> + std::ops::Sub<Output = R>,
    {
        assert_eq!(self.dimension(), other.dimension());
        assert_eq!(self.dimension(), algebra.dimension());

        let mut result = Self::zero(self.dimension());

        // Compute [∑ a_i e_i, ∑ b_j e_j] = ∑ a_i b_j [e_i, e_j]
        for (i, a_i) in self.coefficients.iter().enumerate() {
            if a_i.is_zero() {
                continue;
            }
            for (j, b_j) in other.coefficients.iter().enumerate() {
                if b_j.is_zero() {
                    continue;
                }

                // Compute [e_i, e_j]
                let bracket_ij = algebra.bracket_on_basis(i, j);

                // Multiply by a_i * b_j and add to result
                let term = bracket_ij.scalar_mul(&(a_i.clone() * b_j.clone()));
                result = result.add(&term);
            }
        }

        result
    }

    /// Extract the central component (coefficient of z)
    pub fn central_component(&self) -> Option<&R> {
        // z is the last basis element
        if self.coefficients.is_empty() {
            None
        } else {
            Some(&self.coefficients[self.coefficients.len() - 1])
        }
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for HeisenbergAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<R: Ring + Clone + PartialEq> Eq for HeisenbergAlgebraElement<R> {}

impl<R: Ring + Clone + Display> Display for HeisenbergAlgebraElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = (self.coefficients.len() - 1) / 2;
        let mut terms = Vec::new();

        // p terms
        for i in 0..n {
            if !self.coefficients[i].is_zero() {
                terms.push(format!("{}*p_{}", self.coefficients[i], i));
            }
        }

        // q terms
        for i in 0..n {
            if !self.coefficients[n + i].is_zero() {
                terms.push(format!("{}*q_{}", self.coefficients[n + i], i));
            }
        }

        // z term
        if self.coefficients.len() > 2 * n && !self.coefficients[2 * n].is_zero() {
            terms.push(format!("{}*z", self.coefficients[2 * n]));
        }

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

/// Matrix Representation of Heisenberg Algebra
///
/// Realizes the Heisenberg algebra as (n+2)×(n+2) matrices of the form:
///
/// ⎡0   p^T   k⎤
/// ⎢0   0_n   q⎥
/// ⎣0   0     0⎦
///
/// where p, q ∈ R^n and k ∈ R.
pub struct HeisenbergAlgebraMatrix<R: Ring> {
    /// The rank of the algebra
    rank: usize,
    /// Base algebra reference
    base: HeisenbergAlgebra<R>,
}

impl<R: Ring + Clone> HeisenbergAlgebraMatrix<R> {
    /// Create a new matrix representation
    pub fn new(rank: usize) -> Self {
        HeisenbergAlgebraMatrix {
            rank,
            base: HeisenbergAlgebra::new(rank),
        }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the base algebra
    pub fn base(&self) -> &HeisenbergAlgebra<R> {
        &self.base
    }

    /// Matrix size (n+2)×(n+2)
    pub fn matrix_size(&self) -> usize {
        self.rank + 2
    }

    /// Convert an element to its matrix representation
    pub fn to_matrix(&self, element: &HeisenbergAlgebraElement<R>) -> Matrix<R>
    where
        R: From<i64>,
    {
        let n = self.rank;
        let size = self.matrix_size();
        let mut mat = Matrix::zeros(size, size);

        // Extract coefficients
        let coeffs = element.coefficients();

        // p_i goes into position (0, i+1)
        for i in 0..n {
            if i < coeffs.len() && !coeffs[i].is_zero() {
                let _ = mat.set(0, i + 1, coeffs[i].clone());
            }
        }

        // q_i goes into position (i+1, n+1)
        for i in 0..n {
            if n + i < coeffs.len() && !coeffs[n + i].is_zero() {
                let _ = mat.set(i + 1, n + 1, coeffs[n + i].clone());
            }
        }

        // z goes into position (0, n+1)
        if coeffs.len() > 2 * n && !coeffs[2 * n].is_zero() {
            let _ = mat.set(0, n + 1, coeffs[2 * n].clone());
        }

        mat
    }

    /// Convert a matrix back to an algebra element
    pub fn from_matrix(&self, mat: &Matrix<R>) -> Result<HeisenbergAlgebraElement<R>, String>
    where
        R: From<i64> + PartialEq,
    {
        if mat.rows() != self.matrix_size() || mat.cols() != self.matrix_size() {
            return Err(format!(
                "Matrix must be {}×{} for rank {} Heisenberg algebra",
                self.matrix_size(),
                self.matrix_size(),
                self.rank
            ));
        }

        let n = self.rank;
        let dim = self.base.dimension();
        let mut coeffs = vec![R::from(0); dim];

        // Extract p_i from (0, i+1)
        for i in 0..n {
            coeffs[i] = mat.get(0, i + 1).map_err(|e| e.to_string())?.clone();
        }

        // Extract q_i from (i+1, n+1)
        for i in 0..n {
            coeffs[n + i] = mat.get(i + 1, n + 1).map_err(|e| e.to_string())?.clone();
        }

        // Extract z from (0, n+1)
        coeffs[2 * n] = mat.get(0, n + 1).map_err(|e| e.to_string())?.clone();

        Ok(HeisenbergAlgebraElement::new(coeffs))
    }
}

/// Infinite-Dimensional Heisenberg Algebra
///
/// A Heisenberg algebra with infinitely many generators.
/// The basis consists of p_i, q_i for all i ∈ ℕ, plus the central element z.
pub struct InfiniteHeisenbergAlgebra<R: Ring> {
    /// Base ring marker
    coefficient_ring: PhantomData<R>,
}

impl<R: Ring + Clone> InfiniteHeisenbergAlgebra<R> {
    /// Create a new infinite-dimensional Heisenberg algebra
    pub fn new() -> Self {
        InfiniteHeisenbergAlgebra {
            coefficient_ring: PhantomData,
        }
    }

    /// Check if this is finite dimensional (always false)
    pub fn is_finite_dimensional(&self) -> bool {
        false
    }

    /// Check if this is nilpotent (always true, 2-step nilpotent)
    pub fn is_nilpotent(&self) -> bool {
        true
    }

    /// Get the nilpotency step (always 2)
    pub fn step(&self) -> usize {
        2
    }

    /// Check if this is solvable (always true)
    pub fn is_solvable(&self) -> bool {
        true
    }
}

impl<R: Ring + Clone> Default for InfiniteHeisenbergAlgebra<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ring + Clone> Display for InfiniteHeisenbergAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Infinite-dimensional Heisenberg algebra")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heisenberg_creation() {
        let h2: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(2);
        assert_eq!(h2.rank(), 2);
        assert_eq!(h2.dimension(), 5); // 2*2 + 1
        assert!(h2.is_nilpotent());
        assert!(h2.is_solvable());
        assert_eq!(h2.step(), 2);
    }

    #[test]
    fn test_heisenberg_generators() {
        let h3: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(3);

        // Test p generators
        let p0 = h3.p(0);
        assert_eq!(p0.coefficient(0), Some(&1));
        assert_eq!(p0.coefficient(1), Some(&0));

        // Test q generators
        let q0 = h3.q(0);
        assert_eq!(q0.coefficient(3), Some(&1)); // rank=3, so q_0 is at index 3

        // Test z
        let z = h3.z();
        assert_eq!(z.coefficient(6), Some(&1)); // z is at index 2*3 = 6
    }

    #[test]
    fn test_heisenberg_bracket() {
        let h2: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(2);

        let p0 = h2.p(0);
        let q0 = h2.q(0);
        let z = h2.z();

        // [p_0, q_0] = z
        let bracket = p0.bracket(&q0, &h2);
        assert_eq!(bracket, z);

        // [p_0, p_1] = 0
        let p1 = h2.p(1);
        let bracket2 = p0.bracket(&p1, &h2);
        assert!(bracket2.is_zero());

        // [p_0, z] = 0
        let bracket3 = p0.bracket(&z, &h2);
        assert!(bracket3.is_zero());
    }

    #[test]
    fn test_heisenberg_bracket_different_indices() {
        let h3: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(3);

        let p0 = h3.p(0);
        let q1 = h3.q(1);

        // [p_0, q_1] = 0 (different indices)
        let bracket = p0.bracket(&q1, &h3);
        assert!(bracket.is_zero());

        // [p_1, q_1] = z (same index)
        let p1 = h3.p(1);
        let bracket2 = p1.bracket(&q1, &h3);
        assert_eq!(bracket2, h3.z());
    }

    #[test]
    fn test_heisenberg_center() {
        let h2: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(2);
        assert_eq!(h2.center_dimension(), 1);
        let center = h2.center_basis();
        assert_eq!(center.len(), 1);
        assert_eq!(center[0], h2.z());
    }

    #[test]
    fn test_element_operations() {
        let h2: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(2);

        let p0 = h2.p(0);
        let q0 = h2.q(0);

        // Addition
        let sum = p0.add(&q0);
        assert_eq!(sum.coefficient(0), Some(&1)); // p_0 coefficient
        assert_eq!(sum.coefficient(2), Some(&1)); // q_0 coefficient

        // Scalar multiplication
        let scaled = p0.scalar_mul(&5);
        assert_eq!(scaled.coefficient(0), Some(&5));
    }

    #[test]
    fn test_matrix_representation() {
        let h2_mat: HeisenbergAlgebraMatrix<i64> = HeisenbergAlgebraMatrix::new(2);
        assert_eq!(h2_mat.matrix_size(), 4); // 2 + 2

        let h2 = h2_mat.base();
        let p0 = h2.p(0);

        let mat = h2_mat.to_matrix(&p0);
        assert_eq!(mat.rows(), 4);
        assert_eq!(mat.cols(), 4);

        // p_0 should be at position (0, 1)
        assert_eq!(mat.get(0, 1).unwrap(), &1);
    }

    #[test]
    fn test_infinite_heisenberg() {
        let inf_h: InfiniteHeisenbergAlgebra<i64> = InfiniteHeisenbergAlgebra::new();
        assert!(!inf_h.is_finite_dimensional());
        assert!(inf_h.is_nilpotent());
        assert_eq!(inf_h.step(), 2);
    }

    #[test]
    fn test_central_component() {
        let h1: HeisenbergAlgebra<i64> = HeisenbergAlgebra::new(1);
        let z = h1.z();
        assert_eq!(z.central_component(), Some(&1));

        let p0 = h1.p(0);
        assert_eq!(p0.central_component(), Some(&0));
    }
}

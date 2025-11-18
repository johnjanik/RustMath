//! # Free Module Basis
//!
//! This module provides basis structures for free modules and their duals,
//! corresponding to SageMath's `sage.tensor.modules.free_module_basis`.
//!
//! ## Main Types
//!
//! - `Basis`: Abstract basis trait
//! - `FreeModuleBasis`: A basis of a free module
//! - `FreeModuleCoBasis`: Dual basis (cobasis) of a free module

use std::fmt;
use std::marker::PhantomData;

/// Abstract basis trait
///
/// Represents a basis for a free module or tensor module
pub trait Basis {
    /// The module type this is a basis for
    type Module;

    /// Get the module this basis belongs to
    fn module(&self) -> &Self::Module;

    /// Get the rank (number of basis elements)
    fn rank(&self) -> usize;

    /// Get a name/label for the basis
    fn name(&self) -> &str;
}

/// A basis of a finite-rank free module
///
/// This represents an ordered basis (e_0, e_1, ..., e_{n-1}) of a free module.
pub struct FreeModuleBasis<M> {
    /// The module this basis belongs to
    module: PhantomData<M>,
    /// Rank of the module
    rank: usize,
    /// Symbol/name for basis vectors
    symbol: String,
    /// Indices for basis vectors
    indices: Vec<String>,
    /// LaTeX symbol
    latex_symbol: String,
}

impl<M> FreeModuleBasis<M> {
    /// Create a new basis with default naming
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank of the module
    /// * `symbol` - Symbol to use for basis vectors (e.g., "e")
    pub fn new(rank: usize, symbol: String) -> Self {
        let indices: Vec<String> = (0..rank).map(|i| i.to_string()).collect();
        let latex_symbol = format!("\\{}", symbol);

        Self {
            module: PhantomData,
            rank,
            symbol: symbol.clone(),
            indices,
            latex_symbol,
        }
    }

    /// Create a new basis with custom indices
    pub fn with_indices(rank: usize, symbol: String, indices: Vec<String>) -> Self {
        assert_eq!(
            indices.len(),
            rank,
            "Number of indices must match rank"
        );

        let latex_symbol = format!("\\{}", symbol);

        Self {
            module: PhantomData,
            rank,
            symbol,
            indices,
            latex_symbol,
        }
    }

    /// Get the i-th basis vector name
    pub fn vector_name(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of bounds");
        format!("{}_{}", self.symbol, self.indices[i])
    }

    /// Get the LaTeX representation of the i-th basis vector
    pub fn vector_latex(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of bounds");
        format!("{}_{{{}}}",  self.latex_symbol, self.indices[i])
    }

    /// Get all basis vector names
    pub fn vector_names(&self) -> Vec<String> {
        (0..self.rank).map(|i| self.vector_name(i)).collect()
    }

    /// Get the symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl<M> fmt::Display for FreeModuleBasis<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Basis ({}, ..., {}) on {}-dimensional module",
            self.vector_name(0),
            self.vector_name(self.rank - 1),
            self.rank
        )
    }
}

/// Dual basis (cobasis) of a free module
///
/// For a basis (e_i) of a module M, the dual basis consists of the
/// linear forms e^i : M → R such that e^i(e_j) = δ^i_j
pub struct FreeModuleCoBasis<M> {
    /// The dual module this cobasis belongs to
    dual_module: PhantomData<M>,
    /// Rank of the module
    rank: usize,
    /// Symbol for cobasis vectors (linear forms)
    symbol: String,
    /// Indices
    indices: Vec<String>,
    /// LaTeX symbol
    latex_symbol: String,
}

impl<M> FreeModuleCoBasis<M> {
    /// Create a new cobasis
    pub fn new(rank: usize, symbol: String) -> Self {
        let indices: Vec<String> = (0..rank).map(|i| i.to_string()).collect();
        let latex_symbol = format!("\\{}", symbol);

        Self {
            dual_module: PhantomData,
            rank,
            symbol: symbol.clone(),
            indices,
            latex_symbol,
        }
    }

    /// Create a dual cobasis from a basis
    pub fn from_basis(basis: &FreeModuleBasis<M>) -> Self {
        // Dual uses same symbol but with superscript
        Self::new(basis.rank, basis.symbol.clone())
    }

    /// Get the i-th cobasis vector name (as a superscript)
    pub fn covector_name(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of bounds");
        format!("{}^{}", self.symbol, self.indices[i])
    }

    /// Get the LaTeX representation of the i-th cobasis vector
    pub fn covector_latex(&self, i: usize) -> String {
        assert!(i < self.rank, "Index out of bounds");
        format!("{}^{{{}}}", self.latex_symbol, self.indices[i])
    }

    /// Get all cobasis vector names
    pub fn covector_names(&self) -> Vec<String> {
        (0..self.rank).map(|i| self.covector_name(i)).collect()
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl<M> fmt::Display for FreeModuleCoBasis<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dual basis ({}, ..., {}) on {}-dimensional dual module",
            self.covector_name(0),
            self.covector_name(self.rank - 1),
            self.rank
        )
    }
}

/// Change of basis transformation
///
/// Represents a transformation matrix from one basis to another
pub struct BasisChange<R> {
    /// The transformation matrix
    matrix: Vec<Vec<R>>,
    /// Dimension
    dim: usize,
}

impl<R: Clone> BasisChange<R> {
    /// Create a new basis change from a matrix
    pub fn new(matrix: Vec<Vec<R>>) -> Self {
        let dim = matrix.len();
        assert!(dim > 0, "Matrix must be non-empty");
        assert!(
            matrix.iter().all(|row| row.len() == dim),
            "Matrix must be square"
        );

        Self { matrix, dim }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the transformation matrix
    pub fn matrix(&self) -> &Vec<Vec<R>> {
        &self.matrix
    }

    /// Apply the basis change to a vector
    pub fn apply(&self, vector: &[R]) -> Vec<R>
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + Default + Copy,
    {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        let mut result = vec![R::default(); self.dim];

        for i in 0..self.dim {
            let mut sum = R::default();
            for j in 0..self.dim {
                sum = sum + self.matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Dummy module type for testing
    struct TestModule;

    #[test]
    fn test_free_module_basis() {
        let basis: FreeModuleBasis<TestModule> = FreeModuleBasis::new(3, "e".to_string());

        assert_eq!(basis.rank(), 3);
        assert_eq!(basis.vector_name(0), "e_0");
        assert_eq!(basis.vector_name(1), "e_1");
        assert_eq!(basis.vector_name(2), "e_2");
    }

    #[test]
    fn test_basis_with_custom_indices() {
        let indices = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let basis: FreeModuleBasis<TestModule> =
            FreeModuleBasis::with_indices(3, "e".to_string(), indices);

        assert_eq!(basis.rank(), 3);
        assert_eq!(basis.vector_name(0), "e_x");
        assert_eq!(basis.vector_name(1), "e_y");
        assert_eq!(basis.vector_name(2), "e_z");
    }

    #[test]
    fn test_basis_latex() {
        let basis: FreeModuleBasis<TestModule> = FreeModuleBasis::new(2, "e".to_string());

        assert_eq!(basis.vector_latex(0), "\\e_{0}");
        assert_eq!(basis.vector_latex(1), "\\e_{1}");
    }

    #[test]
    fn test_cobasis() {
        let cobasis: FreeModuleCoBasis<TestModule> = FreeModuleCoBasis::new(3, "e".to_string());

        assert_eq!(cobasis.rank(), 3);
        assert_eq!(cobasis.covector_name(0), "e^0");
        assert_eq!(cobasis.covector_name(1), "e^1");
        assert_eq!(cobasis.covector_name(2), "e^2");
    }

    #[test]
    fn test_cobasis_from_basis() {
        let basis: FreeModuleBasis<TestModule> = FreeModuleBasis::new(3, "e".to_string());
        let cobasis = FreeModuleCoBasis::from_basis(&basis);

        assert_eq!(cobasis.rank(), 3);
        assert_eq!(cobasis.covector_name(0), "e^0");
    }

    #[test]
    fn test_cobasis_latex() {
        let cobasis: FreeModuleCoBasis<TestModule> = FreeModuleCoBasis::new(2, "e".to_string());

        assert_eq!(cobasis.covector_latex(0), "\\e^{0}");
        assert_eq!(cobasis.covector_latex(1), "\\e^{1}");
    }

    #[test]
    fn test_basis_change_identity() {
        let matrix = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        let change = BasisChange::new(matrix);

        let vector = vec![2, 3, 5];
        let result = change.apply(&vector);

        assert_eq!(result, vec![2, 3, 5]);
    }

    #[test]
    fn test_basis_change_transformation() {
        // A simple rotation-like transformation
        let matrix = vec![vec![1, 1, 0], vec![0, 1, 0], vec![0, 0, 1]];
        let change = BasisChange::new(matrix);

        let vector = vec![1, 0, 0];
        let result = change.apply(&vector);

        assert_eq!(result, vec![1, 0, 0]);

        let vector2 = vec![0, 1, 0];
        let result2 = change.apply(&vector2);

        assert_eq!(result2, vec![1, 1, 0]);
    }

    #[test]
    fn test_vector_names() {
        let basis: FreeModuleBasis<TestModule> = FreeModuleBasis::new(3, "v".to_string());
        let names = basis.vector_names();

        assert_eq!(names, vec!["v_0", "v_1", "v_2"]);
    }

    #[test]
    fn test_covector_names() {
        let cobasis: FreeModuleCoBasis<TestModule> = FreeModuleCoBasis::new(3, "f".to_string());
        let names = cobasis.covector_names();

        assert_eq!(names, vec!["f^0", "f^1", "f^2"]);
    }
}

//! RustMath Matrix - Linear algebra operations
//!
//! This crate provides matrix and vector operations for linear algebra.

pub mod action;
pub mod args;
pub mod berlekamp_massey;
pub mod charpoly_exact;
pub mod companion;
pub mod decomposition;
pub mod eigenvalues;
pub mod eigenvalues_approx;
pub mod inner_product;
pub mod integer_forms;
pub mod lattice;
pub mod linear_solve;
pub mod enumerate;
pub mod lll;
pub mod matrix;
pub mod matrix_algebra;
pub mod polynomial_matrix;
pub mod polynomial_ops;
pub mod sparse;
pub mod sparse_forms;
pub mod special;
pub mod strassen;
pub mod vector;
pub mod vector_core_impls;
pub mod vector_space;

pub use action::{
    MatrixAction, Point, PolynomialMap,
    VectorMatrixAction, MatrixVectorAction, MatrixMatrixAction,
    MatrixSchemePointAction, MatrixPolymapAction, PolymapMatrixAction,
};
pub use args::{SparseEntry, MatrixArgs, matrix_args_init};
pub use berlekamp_massey::{berlekamp_massey, berlekamp_massey_verify};
pub use charpoly_exact::charpoly_berkowitz;
pub use companion::{characteristic_polynomial, companion_matrix, rational_canonical_form, RationalCanonicalForm};
pub use decomposition::{
    CholeskyDecomposition, HessenbergDecomposition, LUDecomposition, PLUDecomposition,
    QRDecomposition, SVDDecomposition,
};
pub use eigenvalues::{EigenDecomposition, Eigenvector, JordanForm};
pub use inner_product::InnerProductSpace;
pub use integer_forms::{HermiteNormalForm, SmithNormalForm};
pub use lattice::{
    gram_schmidt_exact, gram_schmidt_real, lll_is_reduced_exact, lll_reduce_real,
    lll_reduce_rf, Lattice, LATTICE_PRECISION,
};
pub use linear_solve::RowEchelonForm;
pub use matrix::Matrix;
pub use matrix_algebra::{MatN, MatrixAlgebra};
pub use sparse::{SparseMatrix, SparseMatrixIterator};
pub use sparse_forms::SparseMatrixED;
pub use special::{
    block_diagonal_matrix, block_matrix, circulant, column_matrix, diagonal_matrix,
    elementary_matrix_add, elementary_matrix_scale, elementary_matrix_swap, hankel, hilbert,
    identity_matrix, jordan_block, lehmer, ones_matrix, random_diagonal_matrix,
    random_integer_matrix, random_lower_triangular, random_unimodular_matrix,
    random_upper_triangular, toeplitz, vandermonde, zero_matrix,
};
pub use strassen::strassen_multiply;
pub use vector::Vector;
pub use vector_core_impls::{FixedVector, LinearMap};
pub use vector_space::{QuotientSpace, VectorSpace};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_matrix() {
        let m = Matrix::from_vec(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }
}

//! RustMath Matrix - Linear algebra operations
//!
//! This crate provides matrix and vector operations for linear algebra.

pub mod action;
pub mod args;
pub mod companion;
pub mod decomposition;
pub mod eigenvalues;
pub mod inner_product;
pub mod integer_forms;
pub mod linear_solve;
pub mod matrix;
pub mod polynomial_matrix;
pub mod polynomial_ops;
pub mod sparse;
pub mod special;
pub mod vector;
pub mod vector_space;

pub use action::{
    MatrixAction, Point, PolynomialMap,
    VectorMatrixAction, MatrixVectorAction, MatrixMatrixAction,
    MatrixSchemePointAction, MatrixPolymapAction, PolymapMatrixAction,
};
pub use args::{SparseEntry, MatrixArgs, matrix_args_init};
pub use companion::{characteristic_polynomial, companion_matrix, rational_canonical_form, RationalCanonicalForm};
pub use decomposition::{
    CholeskyDecomposition, HessenbergDecomposition, LUDecomposition, PLUDecomposition,
    QRDecomposition, SVDDecomposition,
};
pub use eigenvalues::{EigenDecomposition, Eigenvector, JordanForm};
pub use inner_product::InnerProductSpace;
pub use integer_forms::{HermiteNormalForm, SmithNormalForm};
pub use linear_solve::RowEchelonForm;
pub use matrix::Matrix;
pub use sparse::{SparseMatrix, SparseMatrixIterator};
pub use special::{
    block_diagonal_matrix, block_matrix, circulant, column_matrix, diagonal_matrix,
    elementary_matrix_add, elementary_matrix_scale, elementary_matrix_swap, hankel, hilbert,
    identity_matrix, jordan_block, lehmer, ones_matrix, toeplitz, vandermonde, zero_matrix,
};
pub use vector::Vector;
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

//! RustMath Matrix - Linear algebra operations
//!
//! This crate provides matrix and vector operations for linear algebra.

pub mod decomposition;
pub mod linear_solve;
pub mod matrix;
pub mod polynomial_ops;
pub mod vector;
pub mod vector_space;

pub use decomposition::{
    CholeskyDecomposition, HessenbergDecomposition, LUDecomposition, PLUDecomposition,
    QRDecomposition,
};
pub use linear_solve::RowEchelonForm;
pub use matrix::Matrix;
pub use vector::Vector;
pub use vector_space::VectorSpace;

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

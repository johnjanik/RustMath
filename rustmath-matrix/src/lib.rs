//! RustMath Matrix - Linear algebra operations
//!
//! This crate provides matrix and vector operations for linear algebra.

pub mod decomposition;
pub mod linear_solve;
pub mod matrix;
pub mod vector;

pub use decomposition::{LUDecomposition, PLUDecomposition};
pub use linear_solve::RowEchelonForm;
pub use matrix::Matrix;
pub use vector::Vector;

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

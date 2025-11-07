//! RustMath Matrix - Linear algebra operations
//!
//! This crate provides matrix and vector operations for linear algebra.

pub mod matrix;
pub mod vector;

pub use matrix::Matrix;
pub use vector::Vector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_matrix() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }
}

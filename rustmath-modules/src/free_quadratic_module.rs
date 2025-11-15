//! Free quadratic modules with bilinear forms

use rustmath_core::Ring;
use num_bigint::BigInt;

/// A free module with a quadratic form Q: M → R
#[derive(Clone, Debug)]
pub struct FreeQuadraticModule<R: Ring> {
    rank: usize,
    /// Gram matrix of the bilinear form
    gram_matrix: Vec<Vec<R>>,
}

impl<R: Ring> FreeQuadraticModule<R> {
    pub fn new(gram_matrix: Vec<Vec<R>>) -> Self {
        let rank = gram_matrix.len();
        assert!(gram_matrix.iter().all(|row| row.len() == rank));
        Self { rank, gram_matrix }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn gram_matrix(&self) -> &[Vec<R>] {
        &self.gram_matrix
    }

    /// Evaluate quadratic form Q(v) = v^T * G * v
    pub fn evaluate(&self, coords: &[R]) -> R {
        assert_eq!(coords.len(), self.rank);
        let mut result = R::zero();
        for i in 0..self.rank {
            for j in 0..self.rank {
                let term = coords[i].clone() * self.gram_matrix[i][j].clone() * coords[j].clone();
                result = result + term;
            }
        }
        result
    }

    /// Associated bilinear form B(v, w) = v^T * G * w
    pub fn bilinear_form(&self, v: &[R], w: &[R]) -> R {
        assert_eq!(v.len(), self.rank);
        assert_eq!(w.len(), self.rank);
        let mut result = R::zero();
        for i in 0..self.rank {
            for j in 0..self.rank {
                let term = v[i].clone() * self.gram_matrix[i][j].clone() * w[j].clone();
                result = result + term;
            }
        }
        result
    }
}

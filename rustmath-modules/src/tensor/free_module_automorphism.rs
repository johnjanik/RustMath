//! # Free Module Automorphisms
//!
//! This module provides automorphisms of free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_automorphism`.

use std::marker::PhantomData;

/// An automorphism of a free module
///
/// This is an invertible endomorphism, represented by an invertible matrix
pub struct FreeModuleAutomorphism<R> {
    matrix: Vec<Vec<R>>,
    dimension: usize,
    ring: PhantomData<R>,
}

impl<R: Clone> FreeModuleAutomorphism<R> {
    pub fn new(matrix: Vec<Vec<R>>) -> Self {
        let dimension = matrix.len();
        assert!(dimension > 0);
        assert!(matrix.iter().all(|row| row.len() == dimension));

        Self {
            matrix,
            dimension,
            ring: PhantomData,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn matrix(&self) -> &Vec<Vec<R>> {
        &self.matrix
    }

    pub fn apply(&self, vector: &[R]) -> Vec<R>
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + Default + Copy,
    {
        assert_eq!(vector.len(), self.dimension);

        let mut result = vec![R::default(); self.dimension];
        for i in 0..self.dimension {
            let mut sum = R::default();
            for j in 0..self.dimension {
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

    #[test]
    fn test_automorphism_identity() {
        let matrix = vec![vec![1, 0], vec![0, 1]];
        let aut = FreeModuleAutomorphism::new(matrix);

        let v = vec![3, 5];
        let result = aut.apply(&v);

        assert_eq!(result, vec![3, 5]);
    }

    #[test]
    fn test_automorphism_transformation() {
        let matrix = vec![vec![2, 1], vec![1, 1]];
        let aut = FreeModuleAutomorphism::new(matrix);

        let v = vec![1, 0];
        let result = aut.apply(&v);

        assert_eq!(result, vec![2, 1]);
    }
}

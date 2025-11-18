//! # Free Module Morphisms
//!
//! This module provides morphisms between free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_morphism`.

use std::marker::PhantomData;

/// A morphism (homomorphism) between free modules
pub struct FiniteRankFreeModuleMorphism<R> {
    matrix: Vec<Vec<R>>,
    domain_rank: usize,
    codomain_rank: usize,
    ring: PhantomData<R>,
}

impl<R: Clone> FiniteRankFreeModuleMorphism<R> {
    pub fn new(matrix: Vec<Vec<R>>, domain_rank: usize, codomain_rank: usize) -> Self {
        assert_eq!(matrix.len(), codomain_rank);
        assert!(matrix.iter().all(|row| row.len() == domain_rank));

        Self {
            matrix,
            domain_rank,
            codomain_rank,
            ring: PhantomData,
        }
    }

    pub fn domain_rank(&self) -> usize {
        self.domain_rank
    }

    pub fn codomain_rank(&self) -> usize {
        self.codomain_rank
    }

    pub fn matrix(&self) -> &Vec<Vec<R>> {
        &self.matrix
    }

    pub fn apply(&self, vector: &[R]) -> Vec<R>
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + Default + Copy,
    {
        assert_eq!(vector.len(), self.domain_rank);

        let mut result = vec![R::default(); self.codomain_rank];
        for i in 0..self.codomain_rank {
            let mut sum = R::default();
            for j in 0..self.domain_rank {
                sum = sum + self.matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        result
    }
}

/// An endomorphism (morphism from a module to itself)
pub struct FiniteRankFreeModuleEndomorphism<R> {
    inner: FiniteRankFreeModuleMorphism<R>,
}

impl<R: Clone> FiniteRankFreeModuleEndomorphism<R> {
    pub fn new(matrix: Vec<Vec<R>>) -> Self {
        let rank = matrix.len();
        Self {
            inner: FiniteRankFreeModuleMorphism::new(matrix, rank, rank),
        }
    }

    pub fn rank(&self) -> usize {
        self.inner.domain_rank()
    }

    pub fn apply(&self, vector: &[R]) -> Vec<R>
    where
        R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + Default + Copy,
    {
        self.inner.apply(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphism() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let morph = FiniteRankFreeModuleMorphism::new(matrix, 3, 2);

        assert_eq!(morph.domain_rank(), 3);
        assert_eq!(morph.codomain_rank(), 2);
    }

    #[test]
    fn test_morphism_apply() {
        let matrix = vec![vec![1, 0], vec![0, 1], vec![1, 1]];
        let morph = FiniteRankFreeModuleMorphism::new(matrix, 2, 3);

        let v = vec![2, 3];
        let result = morph.apply(&v);

        assert_eq!(result, vec![2, 3, 5]);
    }

    #[test]
    fn test_endomorphism() {
        let matrix = vec![vec![2, 0], vec![0, 3]];
        let endo = FiniteRankFreeModuleEndomorphism::new(matrix);

        assert_eq!(endo.rank(), 2);

        let v = vec![1, 1];
        let result = endo.apply(&v);

        assert_eq!(result, vec![2, 3]);
    }
}

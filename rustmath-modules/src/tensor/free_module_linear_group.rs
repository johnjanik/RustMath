//! # Free Module Linear Groups
//!
//! This module provides linear groups (general linear groups) for free modules,
//! corresponding to SageMath's `sage.tensor.modules.free_module_linear_group`.

use std::marker::PhantomData;

/// General linear group GL(n, R) of a free module
///
/// The group of invertible n×n matrices over ring R
pub struct FreeModuleLinearGroup<R> {
    rank: usize,
    ring: PhantomData<R>,
}

impl<R> FreeModuleLinearGroup<R> {
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            ring: PhantomData,
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Check if a matrix is in the group (simplified - just checks dimensions)
    pub fn contains(&self, matrix: &Vec<Vec<R>>) -> bool {
        matrix.len() == self.rank && matrix.iter().all(|row| row.len() == self.rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_group() {
        let gl: FreeModuleLinearGroup<i32> = FreeModuleLinearGroup::new(3);

        assert_eq!(gl.rank(), 3);
    }

    #[test]
    fn test_contains() {
        let gl: FreeModuleLinearGroup<i32> = FreeModuleLinearGroup::new(2);

        let matrix = vec![vec![1, 0], vec![0, 1]];
        assert!(gl.contains(&matrix));

        let bad_matrix = vec![vec![1, 0, 0], vec![0, 1, 0]];
        assert!(!gl.contains(&bad_matrix));
    }
}

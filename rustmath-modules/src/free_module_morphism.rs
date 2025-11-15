//! Free module morphisms

use rustmath_core::Ring;
use crate::free_module::FreeModule;
use crate::free_module_element::FreeModuleElement;

/// A morphism between free modules (represented as a matrix)
#[derive(Clone, Debug)]
pub struct FreeModuleMorphism<R: Ring> {
    domain_rank: usize,
    codomain_rank: usize,
    /// Matrix representation: rows correspond to codomain, columns to domain
    matrix: Vec<Vec<R>>,
}

impl<R: Ring> FreeModuleMorphism<R> {
    /// Create a new morphism from a matrix
    /// Matrix rows = codomain dimension, columns = domain dimension
    pub fn new(matrix: Vec<Vec<R>>) -> Self {
        assert!(!matrix.is_empty(), "Matrix cannot be empty");
        let codomain_rank = matrix.len();
        let domain_rank = matrix[0].len();

        // Verify all rows have same length
        assert!(matrix.iter().all(|row| row.len() == domain_rank));

        Self {
            domain_rank,
            codomain_rank,
            matrix,
        }
    }

    /// Create zero morphism
    pub fn zero(domain_rank: usize, codomain_rank: usize) -> Self {
        Self {
            domain_rank,
            codomain_rank,
            matrix: vec![vec![R::zero(); domain_rank]; codomain_rank],
        }
    }

    /// Create identity morphism
    pub fn identity(rank: usize) -> Self {
        let mut matrix = vec![vec![R::zero(); rank]; rank];
        for i in 0..rank {
            matrix[i][i] = R::one();
        }
        Self {
            domain_rank: rank,
            codomain_rank: rank,
            matrix,
        }
    }

    /// Get domain rank
    pub fn domain_rank(&self) -> usize {
        self.domain_rank
    }

    /// Get codomain rank
    pub fn codomain_rank(&self) -> usize {
        self.codomain_rank
    }

    /// Get matrix element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> &R {
        &self.matrix[i][j]
    }

    /// Apply morphism to an element
    pub fn apply(&self, element: &FreeModuleElement<R>) -> FreeModuleElement<R> {
        assert_eq!(element.dimension(), self.domain_rank);

        let mut result = vec![R::zero(); self.codomain_rank];
        for i in 0..self.codomain_rank {
            for j in 0..self.domain_rank {
                let prod = self.matrix[i][j].clone() * element.coordinates()[j].clone();
                result[i] = result[i].clone() + prod;
            }
        }

        FreeModuleElement::new(result)
    }

    /// Check if this is the zero morphism
    pub fn is_zero(&self) -> bool {
        self.matrix.iter().all(|row| row.iter().all(|x| x.is_zero()))
    }

    /// Compose this morphism with another: self ∘ other
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.domain_rank, other.codomain_rank);

        let mut result = vec![vec![R::zero(); other.domain_rank]; self.codomain_rank];

        for i in 0..self.codomain_rank {
            for j in 0..other.domain_rank {
                for k in 0..self.domain_rank {
                    let prod = self.matrix[i][k].clone() * other.matrix[k][j].clone();
                    result[i][j] = result[i][j].clone() + prod;
                }
            }
        }

        Self::new(result)
    }

    /// Add two morphisms
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.domain_rank, other.domain_rank);
        assert_eq!(self.codomain_rank, other.codomain_rank);

        let matrix: Vec<Vec<R>> = self.matrix.iter()
            .zip(&other.matrix)
            .map(|(row1, row2)| {
                row1.iter().zip(row2).map(|(a, b)| a.clone() + b.clone()).collect()
            })
            .collect();

        Self::new(matrix)
    }

    /// Scale by a scalar
    pub fn scale(&self, scalar: &R) -> Self {
        let matrix: Vec<Vec<R>> = self.matrix.iter()
            .map(|row| row.iter().map(|x| scalar.clone() * x.clone()).collect())
            .collect();

        Self::new(matrix)
    }

    /// Get the kernel (null space) - returns basis vectors
    /// This is a simplified implementation
    pub fn kernel_basis(&self) -> Vec<FreeModuleElement<R>> {
        // TODO: Implement proper kernel computation using Gaussian elimination
        // For now, return empty vector
        Vec::new()
    }

    /// Get the image (column space) - returns basis vectors
    pub fn image_basis(&self) -> Vec<FreeModuleElement<R>> {
        // TODO: Implement proper image computation
        // For now, return the columns as basis (not reduced)
        (0..self.domain_rank)
            .map(|j| {
                let coords: Vec<R> = (0..self.codomain_rank)
                    .map(|i| self.matrix[i][j].clone())
                    .collect();
                FreeModuleElement::new(coords)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_identity() {
        let id: FreeModuleMorphism<BigInt> = FreeModuleMorphism::identity(3);
        let v = FreeModuleElement::new(vec![
            BigInt::from(1),
            BigInt::from(2),
            BigInt::from(3),
        ]);
        let result = id.apply(&v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_application() {
        // 2x2 matrix [[1, 2], [3, 4]]
        let morphism = FreeModuleMorphism::new(vec![
            vec![BigInt::from(1), BigInt::from(2)],
            vec![BigInt::from(3), BigInt::from(4)],
        ]);

        let v = FreeModuleElement::new(vec![BigInt::from(1), BigInt::from(2)]);
        let result = morphism.apply(&v);

        // [1, 2] * [[1, 2], [3, 4]]^T = [1*1 + 2*2, 1*3 + 2*4] = [5, 11]
        assert_eq!(result.coordinates()[0], BigInt::from(5));
        assert_eq!(result.coordinates()[1], BigInt::from(11));
    }

    #[test]
    fn test_composition() {
        let f = FreeModuleMorphism::new(vec![
            vec![BigInt::from(1), BigInt::from(2)],
        ]);

        let g = FreeModuleMorphism::new(vec![
            vec![BigInt::from(3)],
            vec![BigInt::from(4)],
        ]);

        let composed = f.compose(&g);
        assert_eq!(composed.domain_rank(), 1);
        assert_eq!(composed.codomain_rank(), 1);
    }
}

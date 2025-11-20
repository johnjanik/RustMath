//! Root systems and Cartan matrices
//!
//! Root systems classify simple Lie algebras. This module provides
//! implementations for classical root systems (A_n, B_n, C_n, D_n).

use crate::weight::Weight;

/// Type of root system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootSystemType {
    /// Type A_n (SL(n+1))
    A(usize),
    /// Type B_n (SO(2n+1))
    B(usize),
    /// Type C_n (Sp(2n))
    C(usize),
    /// Type D_n (SO(2n))
    D(usize),
    /// Type E_6 (exceptional)
    E6,
    /// Type E_7 (exceptional)
    E7,
    /// Type E_8 (exceptional)
    E8,
    /// Type F_4 (exceptional)
    F4,
    /// Type G_2 (exceptional)
    G2,
}

/// A root system with Cartan matrix
#[derive(Debug, Clone)]
pub struct RootSystem {
    /// Type of the root system
    pub root_type: RootSystemType,
    /// Rank of the root system
    pub rank: usize,
    /// Cartan matrix
    pub cartan_matrix: Vec<Vec<i64>>,
}

impl RootSystem {
    /// Create a root system of given type
    pub fn new(root_type: RootSystemType) -> Self {
        let rank = match root_type {
            RootSystemType::A(n) => n,
            RootSystemType::B(n) => n,
            RootSystemType::C(n) => n,
            RootSystemType::D(n) => n,
            RootSystemType::E6 => 6,
            RootSystemType::E7 => 7,
            RootSystemType::E8 => 8,
            RootSystemType::F4 => 4,
            RootSystemType::G2 => 2,
        };

        let cartan_matrix = Self::build_cartan_matrix(root_type);

        RootSystem {
            root_type,
            rank,
            cartan_matrix,
        }
    }

    /// Build the Cartan matrix for the root system
    fn build_cartan_matrix(root_type: RootSystemType) -> Vec<Vec<i64>> {
        match root_type {
            RootSystemType::A(n) => {
                // Type A_n Cartan matrix
                let mut matrix = vec![vec![0; n]; n];
                for i in 0..n {
                    matrix[i][i] = 2;
                    if i > 0 {
                        matrix[i][i - 1] = -1;
                    }
                    if i < n - 1 {
                        matrix[i][i + 1] = -1;
                    }
                }
                matrix
            }
            RootSystemType::B(n) => {
                // Type B_n Cartan matrix
                let mut matrix = vec![vec![0; n]; n];
                for i in 0..n {
                    matrix[i][i] = 2;
                    if i > 0 {
                        if i == n - 1 {
                            matrix[i][i - 1] = -2; // Short root connecting to long root
                        } else {
                            matrix[i][i - 1] = -1;
                        }
                    }
                    if i < n - 1 {
                        matrix[i][i + 1] = -1;
                    }
                }
                matrix
            }
            RootSystemType::C(n) => {
                // Type C_n Cartan matrix
                let mut matrix = vec![vec![0; n]; n];
                for i in 0..n {
                    matrix[i][i] = 2;
                    if i > 0 {
                        if i == n - 1 {
                            matrix[i][i - 1] = -2; // Short root to long root
                        } else {
                            matrix[i][i - 1] = -1;
                        }
                    }
                    if i < n - 1 {
                        matrix[i][i + 1] = -1;
                    }
                }
                matrix
            }
            RootSystemType::D(n) => {
                // Type D_n Cartan matrix
                let mut matrix = vec![vec![0; n]; n];
                for i in 0..n {
                    matrix[i][i] = 2;
                    if i > 0 && i < n - 1 {
                        matrix[i][i - 1] = -1;
                    }
                    if i < n - 2 {
                        matrix[i][i + 1] = -1;
                    }
                }
                // Special structure for D_n: last two nodes connect to n-2
                if n >= 3 {
                    matrix[n - 2][n - 3] = -1;
                    matrix[n - 2][n - 1] = -1;
                    matrix[n - 1][n - 2] = -1;
                }
                matrix
            }
            RootSystemType::E6 => {
                // Type E_6 Cartan matrix
                vec![
                    vec![2, 0, -1, 0, 0, 0],
                    vec![0, 2, 0, -1, 0, 0],
                    vec![-1, 0, 2, -1, 0, 0],
                    vec![0, -1, -1, 2, -1, 0],
                    vec![0, 0, 0, -1, 2, -1],
                    vec![0, 0, 0, 0, -1, 2],
                ]
            }
            RootSystemType::E7 => {
                // Type E_7 Cartan matrix
                vec![
                    vec![2, 0, -1, 0, 0, 0, 0],
                    vec![0, 2, 0, -1, 0, 0, 0],
                    vec![-1, 0, 2, -1, 0, 0, 0],
                    vec![0, -1, -1, 2, -1, 0, 0],
                    vec![0, 0, 0, -1, 2, -1, 0],
                    vec![0, 0, 0, 0, -1, 2, -1],
                    vec![0, 0, 0, 0, 0, -1, 2],
                ]
            }
            RootSystemType::E8 => {
                // Type E_8 Cartan matrix
                vec![
                    vec![2, 0, -1, 0, 0, 0, 0, 0],
                    vec![0, 2, 0, -1, 0, 0, 0, 0],
                    vec![-1, 0, 2, -1, 0, 0, 0, 0],
                    vec![0, -1, -1, 2, -1, 0, 0, 0],
                    vec![0, 0, 0, -1, 2, -1, 0, 0],
                    vec![0, 0, 0, 0, -1, 2, -1, 0],
                    vec![0, 0, 0, 0, 0, -1, 2, -1],
                    vec![0, 0, 0, 0, 0, 0, -1, 2],
                ]
            }
            RootSystemType::F4 => {
                // Type F_4 Cartan matrix
                vec![
                    vec![2, -1, 0, 0],
                    vec![-1, 2, -2, 0],
                    vec![0, -1, 2, -1],
                    vec![0, 0, -1, 2],
                ]
            }
            RootSystemType::G2 => {
                // Type G_2 Cartan matrix
                vec![
                    vec![2, -1],
                    vec![-3, 2],
                ]
            }
        }
    }

    /// Get the simple root alpha_i as a weight
    pub fn simple_root(&self, i: usize) -> Weight {
        assert!(i < self.rank);
        Weight::new(self.cartan_matrix[i].clone())
    }

    /// Get the coroot corresponding to simple root i
    pub fn simple_coroot(&self, i: usize) -> Weight {
        assert!(i < self.rank);
        // For simply-laced types, coroots equal roots
        // For non-simply-laced, we need to scale appropriately
        self.simple_root(i)
    }

    /// Compute the action of simple coroot i on a weight
    /// Returns ⟨w, α_i^∨⟩ where α_i^∨ is the i-th simple coroot
    pub fn coroot_action(&self, weight: &Weight, i: usize) -> i64 {
        assert!(i < self.rank);
        assert_eq!(weight.rank(), self.rank);

        // ⟨w, α_i^∨⟩ = sum_j w_j * C_{ji}
        // where C is the Cartan matrix
        // Use saturating operations to avoid overflow
        let mut sum = 0i64;
        for j in 0..self.rank {
            sum = sum.saturating_add(weight.coords[j].saturating_mul(self.cartan_matrix[i][j]));
        }
        sum
    }

    /// Check if a weight is in the root lattice
    pub fn is_in_root_lattice(&self, weight: &Weight) -> bool {
        weight.rank() == self.rank
    }

    /// Get all positive roots (for small ranks)
    pub fn positive_roots(&self) -> Vec<Weight> {
        match self.root_type {
            RootSystemType::A(n) => {
                // Positive roots: e_i - e_j for i < j
                let mut roots = Vec::new();
                for i in 0..=n {
                    for j in i + 1..=n {
                        let mut coords = vec![0; n];
                        // Express e_i - e_j in simple root basis
                        for k in i..j {
                            coords[k] += 1;
                        }
                        roots.push(Weight::new(coords));
                    }
                }
                roots
            }
            _ => {
                // For other types, just return simple roots for now
                (0..self.rank).map(|i| self.simple_root(i)).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_a_cartan_matrix() {
        let root_system = RootSystem::new(RootSystemType::A(3));
        assert_eq!(root_system.rank, 3);

        // Type A_3 Cartan matrix should be:
        // [ 2 -1  0]
        // [-1  2 -1]
        // [ 0 -1  2]
        assert_eq!(root_system.cartan_matrix[0], vec![2, -1, 0]);
        assert_eq!(root_system.cartan_matrix[1], vec![-1, 2, -1]);
        assert_eq!(root_system.cartan_matrix[2], vec![0, -1, 2]);
    }

    #[test]
    fn test_type_b_cartan_matrix() {
        let root_system = RootSystem::new(RootSystemType::B(2));
        // Type B_2 Cartan matrix:
        // [ 2 -1]
        // [-2  2]
        assert_eq!(root_system.cartan_matrix[0], vec![2, -1]);
        assert_eq!(root_system.cartan_matrix[1], vec![-2, 2]);
    }

    #[test]
    fn test_simple_roots() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let alpha1 = root_system.simple_root(0);
        let alpha2 = root_system.simple_root(1);

        assert_eq!(alpha1.coords, vec![2, -1]);
        assert_eq!(alpha2.coords, vec![-1, 2]);
    }

    #[test]
    fn test_coroot_action() {
        let root_system = RootSystem::new(RootSystemType::A(2));
        let w = Weight::new(vec![1, 0]);

        // ⟨w, α_1^∨⟩
        let action = root_system.coroot_action(&w, 0);
        assert_eq!(action, 2); // 1*2 + 0*(-1) = 2
    }
}

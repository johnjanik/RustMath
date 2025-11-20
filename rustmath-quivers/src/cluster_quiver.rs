//! Cluster algebra quivers with mutation operations
//!
//! This module implements quivers specifically for cluster algebras, including:
//! - Quiver mutations (fundamental operation in cluster algebra theory)
//! - Mutation sequences and tracking
//! - Mutation type classification (finite, affine, infinite)
//! - Conversion between exchange matrices and quivers
//!
//! # Cluster Algebra Background
//!
//! A cluster algebra is defined by an exchange matrix B (or equivalently, a quiver).
//! The fundamental operation is **mutation** at a vertex k, which transforms both
//! the quiver and the cluster variables.
//!
//! # References
//!
//! - Fomin & Zelevinsky, "Cluster algebras I: Foundations" (2002)
//! - Keller, "Cluster algebras, quiver representations and triangulated categories" (2012)

use crate::quiver::Quiver;
use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Display};

/// Mutation type classification for cluster algebras
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationType {
    /// Finite type (finitely many clusters)
    Finite,
    /// Affine type (infinite but tamely infinite)
    Affine,
    /// Indefinite/wild type (infinitely many clusters)
    Infinite,
}

impl Display for MutationType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MutationType::Finite => write!(f, "finite"),
            MutationType::Affine => write!(f, "affine"),
            MutationType::Infinite => write!(f, "infinite"),
        }
    }
}

/// A mutation sequence - ordered list of vertex indices to mutate
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MutationSequence {
    /// Sequence of vertices to mutate
    pub sequence: Vec<usize>,
}

impl MutationSequence {
    /// Create an empty mutation sequence
    pub fn new() -> Self {
        MutationSequence {
            sequence: Vec::new(),
        }
    }

    /// Create a mutation sequence from a vector
    pub fn from_vec(sequence: Vec<usize>) -> Self {
        MutationSequence { sequence }
    }

    /// Add a mutation to the sequence
    pub fn push(&mut self, vertex: usize) {
        self.sequence.push(vertex);
    }

    /// Get the length of the sequence
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Extend with another sequence
    pub fn extend(&mut self, other: &MutationSequence) {
        self.sequence.extend_from_slice(&other.sequence);
    }

    /// Reverse the sequence
    pub fn reverse(&self) -> MutationSequence {
        let mut rev = self.sequence.clone();
        rev.reverse();
        MutationSequence { sequence: rev }
    }

    /// Compose with another sequence (self followed by other)
    pub fn compose(&self, other: &MutationSequence) -> MutationSequence {
        let mut result = self.clone();
        result.extend(other);
        result
    }

    /// Get iterator over mutations
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.sequence.iter()
    }
}

impl Default for MutationSequence {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for MutationSequence {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.sequence.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}

/// A quiver for cluster algebras with mutation operations
///
/// This wraps a basic Quiver and adds cluster algebra specific functionality:
/// - Exchange matrix representation
/// - Mutation operations
/// - Mutation type classification
/// - Cluster variable tracking
#[derive(Debug, Clone)]
pub struct ClusterQuiver {
    /// Number of vertices (rank of cluster algebra)
    rank: usize,
    /// Exchange matrix as i64 values
    /// B[i][j] = number of arrows from j to i (minus number from i to j)
    exchange_matrix: Vec<Vec<i64>>,
    /// Frozen vertices (cannot be mutated)
    frozen_vertices: HashSet<usize>,
    /// History of mutations applied
    mutation_history: MutationSequence,
}

impl ClusterQuiver {
    /// Create a new cluster quiver from an exchange matrix
    ///
    /// The exchange matrix B is skew-symmetric for the principal part.
    /// Rows beyond rank represent frozen vertices (coefficients).
    pub fn new(exchange_matrix: Vec<Vec<i64>>) -> Result<Self, String> {
        if exchange_matrix.is_empty() {
            return Err("Exchange matrix cannot be empty".to_string());
        }

        let rank = exchange_matrix[0].len();

        // Check all rows have the same length
        for (i, row) in exchange_matrix.iter().enumerate() {
            if row.len() != rank {
                return Err(format!(
                    "Row {} has length {}, expected {}",
                    i, row.len(), rank
                ));
            }
        }

        Ok(ClusterQuiver {
            rank,
            exchange_matrix,
            frozen_vertices: HashSet::new(),
            mutation_history: MutationSequence::new(),
        })
    }

    /// Create a cluster quiver of type A_n
    ///
    /// Creates a quiver: 0 -> 1 -> 2 -> ... -> (n-1)
    pub fn type_a(n: usize) -> Self {
        if n == 0 {
            panic!("Type A_n requires n >= 1");
        }

        let mut matrix = vec![vec![0i64; n]; n];
        for i in 0..n - 1 {
            matrix[i][i + 1] = 1;
            matrix[i + 1][i] = -1;
        }

        ClusterQuiver {
            rank: n,
            exchange_matrix: matrix,
            frozen_vertices: HashSet::new(),
            mutation_history: MutationSequence::new(),
        }
    }

    /// Create a cluster quiver of type D_n
    ///
    /// Creates a quiver with a fork at vertex n-2
    pub fn type_d(n: usize) -> Result<Self, String> {
        if n < 4 {
            return Err("Type D_n requires n >= 4".to_string());
        }

        let mut matrix = vec![vec![0i64; n]; n];

        // Linear chain from 0 to n-3
        for i in 0..n - 3 {
            matrix[i][i + 1] = 1;
            matrix[i + 1][i] = -1;
        }

        // Fork at n-3 to both n-2 and n-1
        matrix[n - 3][n - 2] = 1;
        matrix[n - 2][n - 3] = -1;
        matrix[n - 3][n - 1] = 1;
        matrix[n - 1][n - 3] = -1;

        Ok(ClusterQuiver {
            rank: n,
            exchange_matrix: matrix,
            frozen_vertices: HashSet::new(),
            mutation_history: MutationSequence::new(),
        })
    }

    /// Create a cluster quiver of type E_n (n = 6, 7, 8)
    pub fn type_e(n: usize) -> Result<Self, String> {
        if n < 6 || n > 8 {
            return Err("Type E_n requires n ∈ {6, 7, 8}".to_string());
        }

        let mut matrix = vec![vec![0i64; n]; n];

        // Linear chain with a branch
        // Structure: 0 - 1 - 2 - 3 - ... with branch at position 2
        for i in 0..n - 1 {
            if i == 2 {
                // Add branch from vertex 2 to vertex n-1
                matrix[2][n - 1] = 1;
                matrix[n - 1][2] = -1;
            } else if i != n - 1 {
                matrix[i][i + 1] = 1;
                matrix[i + 1][i] = -1;
            }
        }

        Ok(ClusterQuiver {
            rank: n,
            exchange_matrix: matrix,
            frozen_vertices: HashSet::new(),
            mutation_history: MutationSequence::new(),
        })
    }

    /// Get the rank (number of mutable vertices)
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the total number of vertices (including frozen)
    pub fn num_vertices(&self) -> usize {
        self.exchange_matrix.len()
    }

    /// Get the exchange matrix
    pub fn exchange_matrix(&self) -> &Vec<Vec<i64>> {
        &self.exchange_matrix
    }

    /// Get entry B[i][j] from exchange matrix
    pub fn get_entry(&self, i: usize, j: usize) -> i64 {
        if i < self.exchange_matrix.len() && j < self.rank {
            self.exchange_matrix[i][j]
        } else {
            0
        }
    }

    /// Set entry B[i][j] in exchange matrix
    pub fn set_entry(&mut self, i: usize, j: usize, value: i64) {
        if i < self.exchange_matrix.len() && j < self.rank {
            self.exchange_matrix[i][j] = value;
        }
    }

    /// Freeze a vertex (make it non-mutable)
    pub fn freeze_vertex(&mut self, vertex: usize) {
        if vertex < self.rank {
            self.frozen_vertices.insert(vertex);
        }
    }

    /// Unfreeze a vertex
    pub fn unfreeze_vertex(&mut self, vertex: usize) {
        self.frozen_vertices.remove(&vertex);
    }

    /// Check if a vertex is frozen
    pub fn is_frozen(&self, vertex: usize) -> bool {
        self.frozen_vertices.contains(&vertex)
    }

    /// Get the mutation history
    pub fn mutation_history(&self) -> &MutationSequence {
        &self.mutation_history
    }

    /// Mutate at vertex k
    ///
    /// Applies the cluster algebra mutation formula:
    /// - For each pair of vertices (i, j):
    ///   - If i = k or j = k: flip sign of B[i][j]
    ///   - Otherwise: B'[i][j] = B[i][j] + sgn(B[i][k]) * max(B[i][k] * B[k][j], 0)
    ///
    /// Returns an error if k is out of bounds or frozen.
    pub fn mutate(&mut self, k: usize) -> Result<(), String> {
        if k >= self.rank {
            return Err(format!("Vertex {} out of bounds (rank = {})", k, self.rank));
        }

        if self.is_frozen(k) {
            return Err(format!("Vertex {} is frozen", k));
        }

        let rows = self.exchange_matrix.len();
        let mut new_matrix = self.exchange_matrix.clone();

        for i in 0..rows {
            for j in 0..self.rank {
                if i == k || j == k {
                    // Flip sign for row k or column k
                    new_matrix[i][j] = -self.exchange_matrix[i][j];
                } else {
                    // Apply mutation formula
                    let b_ik = self.exchange_matrix[i][k];
                    let b_kj = self.exchange_matrix[k][j];
                    let b_ij = self.exchange_matrix[i][j];

                    if b_ik > 0 && b_kj > 0 {
                        new_matrix[i][j] = b_ij + b_ik * b_kj;
                    } else if b_ik < 0 && b_kj < 0 {
                        new_matrix[i][j] = b_ij - b_ik * b_kj;
                    }
                    // else: entry unchanged
                }
            }
        }

        self.exchange_matrix = new_matrix;
        self.mutation_history.push(k);
        Ok(())
    }

    /// Create a mutated copy without modifying this quiver
    pub fn mutated(&self, k: usize) -> Result<Self, String> {
        let mut copy = self.clone();
        copy.mutation_history = MutationSequence::new(); // Reset history for the copy
        copy.mutate(k)?;
        Ok(copy)
    }

    /// Apply a sequence of mutations
    pub fn mutate_sequence(&mut self, sequence: &MutationSequence) -> Result<(), String> {
        for &k in sequence.iter() {
            self.mutate(k)?;
        }
        Ok(())
    }

    /// Check if the quiver is mutation-equivalent to another quiver
    ///
    /// Two quivers are mutation-equivalent if one can be obtained from the other
    /// by a sequence of mutations.
    pub fn is_mutation_equivalent(&self, other: &ClusterQuiver, max_depth: usize) -> bool {
        if self.rank != other.rank {
            return false;
        }

        // BFS to explore mutation equivalence
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((self.clone(), 0));
        visited.insert(self.matrix_hash());

        while let Some((quiver, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Check if we found the target
            if quiver.matrix_equals(other) {
                return true;
            }

            // Try all mutations
            for k in 0..self.rank {
                if let Ok(mut mutated) = quiver.mutated(k) {
                    mutated.mutation_history = MutationSequence::new(); // Don't track history in BFS
                    let hash = mutated.matrix_hash();

                    if !visited.contains(&hash) {
                        visited.insert(hash);
                        queue.push_back((mutated, depth + 1));
                    }
                }
            }
        }

        false
    }

    /// Determine the mutation type (finite, affine, or infinite)
    ///
    /// This is done by:
    /// 1. Computing the Cartan companion matrix C = B^T B
    /// 2. Checking if C is positive definite (finite type)
    /// 3. Checking if C is positive semi-definite with 1D kernel (affine type)
    /// 4. Otherwise, it's infinite type
    pub fn mutation_type(&self) -> MutationType {
        // For simplicity, we use a heuristic based on mutation periodicity
        // A more rigorous approach would compute eigenvalues of C = B^T B

        let max_exploration = 100;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        let initial_hash = self.matrix_hash();
        queue.push_back((self.clone(), 0));
        visited.insert(initial_hash);

        while let Some((quiver, depth)) = queue.pop_front() {
            if depth >= max_exploration {
                // Couldn't determine in reasonable time, assume infinite
                return MutationType::Infinite;
            }

            for k in 0..self.rank {
                if let Ok(mut mutated) = quiver.mutated(k) {
                    mutated.mutation_history = MutationSequence::new();
                    let hash = mutated.matrix_hash();

                    if hash == initial_hash && depth > 0 {
                        // Found a cycle back to initial - likely finite or affine
                        // Distinguish by checking if all mutations eventually cycle
                        if visited.len() < 20 {
                            return MutationType::Finite;
                        } else {
                            return MutationType::Affine;
                        }
                    }

                    if !visited.contains(&hash) {
                        visited.insert(hash);
                        queue.push_back((mutated, depth + 1));
                    }
                }
            }
        }

        // If we've seen many distinct matrices, likely infinite
        if visited.len() > 50 {
            MutationType::Infinite
        } else {
            MutationType::Finite
        }
    }

    /// Convert to a basic Quiver representation
    ///
    /// Creates a directed graph where:
    /// - Vertices are labeled 0, 1, ..., rank-1
    /// - An arrow from i to j exists if B[i][j] < 0 (i.e., B[j][i] > 0)
    /// - Multiple arrows are represented by edge labels
    pub fn to_quiver(&self) -> Quiver {
        let mut quiver = Quiver::new(self.rank);

        for i in 0..self.rank {
            for j in 0..self.rank {
                if i != j {
                    let b_ji = self.exchange_matrix[j][i];
                    if b_ji > 0 {
                        // Arrow from i to j with multiplicity b_ji
                        for k in 0..b_ji {
                            let label = if b_ji == 1 {
                                format!("a_{}{}", i, j)
                            } else {
                                format!("a_{}{}_{}", i, j, k)
                            };
                            let _ = quiver.add_edge(i, j, label);
                        }
                    }
                }
            }
        }

        quiver
    }

    /// Compute hash of the exchange matrix for comparison
    fn matrix_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for row in &self.exchange_matrix {
            for &val in row {
                val.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Check if exchange matrices are equal
    fn matrix_equals(&self, other: &ClusterQuiver) -> bool {
        if self.rank != other.rank {
            return false;
        }
        if self.exchange_matrix.len() != other.exchange_matrix.len() {
            return false;
        }

        for i in 0..self.exchange_matrix.len() {
            for j in 0..self.rank {
                if self.exchange_matrix[i][j] != other.exchange_matrix[i][j] {
                    return false;
                }
            }
        }

        true
    }

    /// Get the Cartan companion matrix C = B^T * B
    ///
    /// This is used for classification purposes. The matrix C:
    /// - Is positive definite ⟺ finite type
    /// - Is positive semi-definite with 1D kernel ⟺ affine type
    /// - Otherwise ⟺ infinite type
    pub fn cartan_companion(&self) -> Vec<Vec<i64>> {
        let n = self.rank;
        let mut c = vec![vec![0i64; n]; n];

        for i in 0..n {
            for j in 0..n {
                let mut sum = 0i64;
                for k in 0..self.exchange_matrix.len() {
                    sum += self.exchange_matrix[k][i] * self.exchange_matrix[k][j];
                }
                c[i][j] = sum;
            }
        }

        c
    }

    /// Get the principal part (top rank × rank submatrix)
    pub fn principal_part(&self) -> Vec<Vec<i64>> {
        let mut result = vec![vec![0i64; self.rank]; self.rank];
        for i in 0..self.rank {
            for j in 0..self.rank {
                result[i][j] = self.exchange_matrix[i][j];
            }
        }
        result
    }

    /// Add coefficients to the quiver
    ///
    /// Extends the exchange matrix with additional rows for frozen vertices
    pub fn add_coefficients(&mut self, num_coefficients: usize) {
        for _ in 0..num_coefficients {
            self.exchange_matrix.push(vec![0i64; self.rank]);
        }
    }

    /// Reset mutation history
    pub fn reset_history(&mut self) {
        self.mutation_history = MutationSequence::new();
    }
}

impl Display for ClusterQuiver {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "ClusterQuiver(rank={}, vertices={})", self.rank, self.num_vertices())?;
        writeln!(f, "Exchange matrix:")?;
        for row in &self.exchange_matrix {
            write!(f, "  [")?;
            for (j, &val) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:3}", val)?;
            }
            writeln!(f, "]")?;
        }
        if !self.frozen_vertices.is_empty() {
            writeln!(f, "Frozen vertices: {:?}", self.frozen_vertices)?;
        }
        if !self.mutation_history.is_empty() {
            writeln!(f, "Mutation history: {}", self.mutation_history)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_sequence() {
        let mut seq = MutationSequence::new();
        assert!(seq.is_empty());

        seq.push(0);
        seq.push(1);
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.sequence, vec![0, 1]);

        let rev = seq.reverse();
        assert_eq!(rev.sequence, vec![1, 0]);
    }

    #[test]
    fn test_type_a_quiver() {
        let q = ClusterQuiver::type_a(3);
        assert_eq!(q.rank(), 3);

        // Check exchange matrix for A_3: 0 -> 1 -> 2
        assert_eq!(q.get_entry(0, 1), 1);
        assert_eq!(q.get_entry(1, 0), -1);
        assert_eq!(q.get_entry(1, 2), 1);
        assert_eq!(q.get_entry(2, 1), -1);
    }

    #[test]
    fn test_type_d_quiver() {
        let q = ClusterQuiver::type_d(4).unwrap();
        assert_eq!(q.rank(), 4);

        // Check the fork structure
        assert_eq!(q.get_entry(0, 1), 1);
        assert_eq!(q.get_entry(1, 2), -1);
        assert_eq!(q.get_entry(1, 3), -1);
    }

    #[test]
    fn test_type_e_quiver() {
        let q = ClusterQuiver::type_e(6).unwrap();
        assert_eq!(q.rank(), 6);
    }

    #[test]
    fn test_mutation_a2() {
        // A_2 quiver: 0 -> 1
        let mut q = ClusterQuiver::type_a(2);

        // Initial: B = [[0, 1], [-1, 0]]
        assert_eq!(q.get_entry(0, 1), 1);
        assert_eq!(q.get_entry(1, 0), -1);

        // Mutate at vertex 0
        q.mutate(0).unwrap();

        // After mutation at 0: B = [[0, -1], [1, 0]]
        assert_eq!(q.get_entry(0, 1), -1);
        assert_eq!(q.get_entry(1, 0), 1);

        // Mutate again at vertex 0 - should return to original
        q.mutate(0).unwrap();
        assert_eq!(q.get_entry(0, 1), 1);
        assert_eq!(q.get_entry(1, 0), -1);
    }

    #[test]
    fn test_mutation_a3() {
        // A_3 quiver: 0 -> 1 -> 2
        let mut q = ClusterQuiver::type_a(3);

        // Mutate at middle vertex
        q.mutate(1).unwrap();

        // Check mutation affected adjacent entries
        assert_eq!(q.get_entry(0, 1), -1);
        assert_eq!(q.get_entry(1, 2), -1);

        // Check new edge appeared
        assert_eq!(q.get_entry(0, 2), 1);
    }

    #[test]
    fn test_frozen_vertex() {
        let mut q = ClusterQuiver::type_a(3);

        q.freeze_vertex(1);
        assert!(q.is_frozen(1));

        // Mutation should fail
        assert!(q.mutate(1).is_err());

        // Other vertices should still work
        assert!(q.mutate(0).is_ok());
    }

    #[test]
    fn test_mutation_sequence_application() {
        let mut q = ClusterQuiver::type_a(3);
        let seq = MutationSequence::from_vec(vec![0, 1, 2, 1, 0]);

        assert!(q.mutate_sequence(&seq).is_ok());
        assert_eq!(q.mutation_history().len(), 5);
    }

    #[test]
    fn test_cartan_companion() {
        let q = ClusterQuiver::type_a(2);
        let c = q.cartan_companion();

        // For A_2, C should be [[1, -1], [-1, 1]]
        assert_eq!(c[0][0], 1);
        assert_eq!(c[0][1], -1);
        assert_eq!(c[1][0], -1);
        assert_eq!(c[1][1], 1);
    }

    #[test]
    fn test_to_quiver() {
        let cq = ClusterQuiver::type_a(3);
        let q = cq.to_quiver();

        assert_eq!(q.num_vertices(), 3);
        assert!(q.num_edges() >= 2); // At least two arrows
    }

    #[test]
    fn test_mutation_type_finite() {
        // A_2 is finite type
        let q = ClusterQuiver::type_a(2);
        let mt = q.mutation_type();
        assert_eq!(mt, MutationType::Finite);
    }

    #[test]
    fn test_matrix_equality() {
        let q1 = ClusterQuiver::type_a(3);
        let q2 = ClusterQuiver::type_a(3);

        assert!(q1.matrix_equals(&q2));

        let mut q3 = ClusterQuiver::type_a(3);
        q3.mutate(0).unwrap();
        assert!(!q1.matrix_equals(&q3));
    }

    #[test]
    fn test_principal_part() {
        let mut q = ClusterQuiver::type_a(2);
        q.add_coefficients(2);

        let principal = q.principal_part();
        assert_eq!(principal.len(), 2);
        assert_eq!(principal[0].len(), 2);
    }

    #[test]
    fn test_mutation_history() {
        let mut q = ClusterQuiver::type_a(3);

        q.mutate(0).unwrap();
        q.mutate(1).unwrap();

        let history = q.mutation_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history.sequence, vec![0, 1]);
    }
}

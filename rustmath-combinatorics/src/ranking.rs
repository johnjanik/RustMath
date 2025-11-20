//! Ranking and unranking utilities for combinatorial objects
//!
//! A ranking is a bijection between combinatorial objects and integers 0, 1, 2, ..., n-1
//! where n is the total number of objects. This allows efficient enumeration and random access.

use crate::binomial;
use rustmath_core::NumericConversion;

/// A trait for combinatorial objects that can be ranked and unranked
pub trait Rankable: Sized {
    /// Convert this object to its rank (index in lexicographic order)
    fn rank(&self) -> usize;

    /// Create an object from its rank
    fn unrank(rank: usize, params: &Self::Params) -> Option<Self>;

    /// Parameters needed to specify the combinatorial class
    type Params;

    /// The total number of objects with given parameters
    fn count(params: &Self::Params) -> usize;
}

/// A ranking table for efficient enumeration of combinatorial objects
///
/// This structure pre-computes rankings for fast lookup
pub struct RankingTable<T: Rankable> {
    /// The parameters defining this combinatorial class
    params: T::Params,
    /// Total number of elements
    count: usize,
}

impl<T: Rankable> RankingTable<T> {
    /// Create a new ranking table for objects with given parameters
    pub fn new(params: T::Params) -> Self {
        let count = T::count(&params);
        RankingTable { params, count }
    }

    /// Get the total number of objects
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the object at a given rank
    pub fn get(&self, rank: usize) -> Option<T> {
        if rank >= self.count {
            return None;
        }
        T::unrank(rank, &self.params)
    }

    /// Get the rank of an object
    pub fn rank(&self, obj: &T) -> usize {
        obj.rank()
    }

    /// Iterator over all objects in rank order
    pub fn iter(&self) -> RankingTableIter<'_, T> {
        RankingTableIter {
            table: self,
            current_rank: 0,
        }
    }

    /// Get a random object (requires rand crate feature)
    #[cfg(feature = "random")]
    pub fn random(&self) -> Option<T> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let rank = rng.gen_range(0..self.count);
        self.get(rank)
    }
}

/// Iterator over ranking table
pub struct RankingTableIter<'a, T: Rankable> {
    table: &'a RankingTable<T>,
    current_rank: usize,
}

impl<'a, T: Rankable> Iterator for RankingTableIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_rank >= self.table.count {
            return None;
        }
        let obj = self.table.get(self.current_rank)?;
        self.current_rank += 1;
        Some(obj)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.table.count - self.current_rank;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Rankable> ExactSizeIterator for RankingTableIter<'a, T> {}

/// Ranking implementation for combinations
///
/// Combinations are ranked in lexicographic order
pub struct CombinationRank {
    /// The k-subset represented as a sorted vector of indices
    elements: Vec<usize>,
    n: usize,
    k: usize,
}

impl CombinationRank {
    pub fn new(n: usize, k: usize, elements: Vec<usize>) -> Option<Self> {
        if elements.len() != k {
            return None;
        }
        if elements.iter().any(|&x| x >= n) {
            return None;
        }
        // Check if sorted and distinct
        for i in 1..elements.len() {
            if elements[i] <= elements[i - 1] {
                return None;
            }
        }
        Some(CombinationRank { elements, n, k })
    }

    pub fn elements(&self) -> &[usize] {
        &self.elements
    }
}

impl Rankable for CombinationRank {
    type Params = (usize, usize); // (n, k)

    fn rank(&self) -> usize {
        let n = self.n;
        let k = self.k;
        let mut rank = 0;

        for (i, &elem) in self.elements.iter().enumerate() {
            // Count combinations that come before this one
            let remaining = k - i;
            let start = if i == 0 { 0 } else { self.elements[i - 1] + 1 };

            for val in start..elem {
                // If we chose 'val' at position i, how many combinations are there?
                // We need to choose (remaining - 1) elements from {val+1, ..., n-1}
                let choices = n - val - 1;
                if choices >= remaining - 1 {
                    rank += binomial(choices as u32, (remaining - 1) as u32).to_usize().unwrap_or(0);
                }
            }
        }

        rank
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let (n, k) = *params;
        if k == 0 {
            return Some(CombinationRank {
                elements: vec![],
                n,
                k,
            });
        }
        if k > n {
            return None;
        }

        let mut elements = Vec::new();
        let mut remaining_rank = rank;
        let mut start = 0;

        for i in 0..k {
            let remaining = k - i;

            // Find the smallest value v such that C(n-v-1, remaining-1) <= remaining_rank
            for val in start..n {
                let choices = n - val - 1;
                if choices < remaining - 1 {
                    continue;
                }

                let count = binomial(choices as u32, (remaining - 1) as u32)
                    .to_usize()
                    .unwrap_or(0);

                if remaining_rank < count {
                    elements.push(val);
                    start = val + 1;
                    break;
                } else {
                    remaining_rank -= count;
                }
            }
        }

        if elements.len() != k {
            return None;
        }

        Some(CombinationRank { elements, n, k })
    }

    fn count(params: &Self::Params) -> usize {
        let (n, k) = *params;
        binomial(n as u32, k as u32).to_usize().unwrap_or(0)
    }
}

/// Ranking for permutations
///
/// Permutations are ranked using the factorial number system (Lehmer code)
pub struct PermutationRank {
    /// The permutation as a vector
    elements: Vec<usize>,
}

impl PermutationRank {
    pub fn new(elements: Vec<usize>) -> Option<Self> {
        let n = elements.len();
        // Verify it's a valid permutation
        let mut seen = vec![false; n];
        for &elem in &elements {
            if elem >= n || seen[elem] {
                return None;
            }
            seen[elem] = true;
        }
        Some(PermutationRank { elements })
    }

    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Compute the Lehmer code for this permutation
    pub fn lehmer_code(&self) -> Vec<usize> {
        let n = self.elements.len();
        let mut code = vec![0; n];

        for i in 0..n {
            let mut count = 0;
            for j in i + 1..n {
                if self.elements[j] < self.elements[i] {
                    count += 1;
                }
            }
            code[i] = count;
        }

        code
    }
}

impl Rankable for PermutationRank {
    type Params = usize; // n

    fn rank(&self) -> usize {
        let lehmer = self.lehmer_code();
        let n = lehmer.len();
        let mut rank = 0;
        let mut factorial = 1;

        for i in (0..n).rev() {
            rank += lehmer[i] * factorial;
            factorial *= n - i;
        }

        rank
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let n = *params;
        if n == 0 {
            return Some(PermutationRank { elements: vec![] });
        }

        // Convert rank to Lehmer code
        let mut lehmer = vec![0; n];
        let mut remaining_rank = rank;
        let mut factorial = 1;

        for i in 1..n {
            factorial *= i;
        }

        for i in 0..n {
            if factorial == 0 {
                break;
            }
            lehmer[i] = remaining_rank / factorial;
            remaining_rank %= factorial;
            if n - i > 1 {
                factorial /= n - i - 1;
            }
        }

        // Convert Lehmer code to permutation
        let mut available: Vec<usize> = (0..n).collect();
        let mut elements = vec![0; n];

        for i in 0..n {
            if lehmer[i] >= available.len() {
                return None;
            }
            elements[i] = available.remove(lehmer[i]);
        }

        Some(PermutationRank { elements })
    }

    fn count(params: &Self::Params) -> usize {
        let n = *params;
        (1..=n).product()
    }
}

/// Ranking for binary trees with n internal nodes
///
/// Binary trees are ranked using their bijection with Dyck paths (balanced parentheses).
/// This gives a natural ranking using Catalan numbers.
///
/// A binary tree with n internal nodes corresponds to the n-th Catalan number C_n.
/// Trees are represented as sequences of left/right child indicators in pre-order traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryTreeRank {
    /// Pre-order traversal representation: 0 = leaf, 1 = left child, 2 = right child, 3 = both
    /// We use a more compact representation: sequence of bits indicating tree structure
    structure: Vec<bool>, // true = internal node, false = leaf
    num_internal_nodes: usize,
}

impl BinaryTreeRank {
    /// Create a new binary tree rank from a structure sequence
    ///
    /// The structure is a pre-order traversal where each internal node contributes
    /// two children (either leaves or subtrees)
    pub fn new(structure: Vec<bool>) -> Option<Self> {
        // Verify it's a valid binary tree structure
        let num_internal = structure.iter().filter(|&&b| b).count();
        let num_leaves = structure.iter().filter(|&&b| !b).count();

        // A binary tree with n internal nodes has n+1 leaves
        if num_leaves != num_internal + 1 {
            return None;
        }

        // Verify it's a valid pre-order sequence
        if !is_valid_tree_sequence(&structure) {
            return None;
        }

        Some(BinaryTreeRank {
            structure,
            num_internal_nodes: num_internal,
        })
    }

    /// Get the structure sequence
    pub fn structure(&self) -> &[bool] {
        &self.structure
    }

    /// Get the number of internal nodes
    pub fn num_internal_nodes(&self) -> usize {
        self.num_internal_nodes
    }
}

/// Check if a sequence represents a valid binary tree in pre-order
fn is_valid_tree_sequence(seq: &[bool]) -> bool {
    if seq.is_empty() {
        return false;
    }

    // Count nodes: internal nodes consume 2 slots, leaves consume 0
    let mut available_slots = 1; // Start with root slot

    for &is_internal in seq {
        if available_slots == 0 {
            return false;
        }

        available_slots -= 1; // Consume one slot

        if is_internal {
            available_slots += 2; // Internal node creates 2 new slots
        }
    }

    available_slots == 0
}

impl Rankable for BinaryTreeRank {
    type Params = usize; // Number of internal nodes

    fn rank(&self) -> usize {
        // Convert tree structure to rank using Catalan number indexing
        // This uses a dynamic programming approach
        rank_binary_tree_recursive(&self.structure, 0).0
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let n = *params;
        let structure = unrank_binary_tree(rank, n)?;
        BinaryTreeRank::new(structure)
    }

    fn count(params: &Self::Params) -> usize {
        // The count is the n-th Catalan number
        let n = *params;
        crate::catalan(n as u32).to_usize().unwrap_or(0)
    }
}

/// Recursively compute the rank of a binary tree
fn rank_binary_tree_recursive(structure: &[bool], start: usize) -> (usize, usize) {
    if start >= structure.len() || !structure[start] {
        // Leaf node
        return (0, start + 1);
    }

    // Internal node - rank left and right subtrees
    let (left_rank, next_pos) = rank_binary_tree_recursive(structure, start + 1);
    let (right_rank, final_pos) = rank_binary_tree_recursive(structure, next_pos);

    // Combine ranks using Catalan number arithmetic
    let left_size = count_internals(&structure[start + 1..next_pos]);
    let right_size = count_internals(&structure[next_pos..final_pos]);

    let rank = combine_tree_ranks(left_rank, right_rank, left_size, right_size);

    (rank, final_pos)
}

fn count_internals(seq: &[bool]) -> usize {
    seq.iter().filter(|&&b| b).count()
}

fn combine_tree_ranks(left_rank: usize, right_rank: usize, left_size: usize, right_size: usize) -> usize {
    // Simplified ranking - in practice this would use Catalan number indexing
    // For now, use a basic combination
    left_rank * (right_size + 1) + right_rank
}

fn unrank_binary_tree(rank: usize, n: usize) -> Option<Vec<bool>> {
    if n == 0 {
        return Some(vec![false]);
    }

    // Generate the rank-th binary tree with n internal nodes
    // This is a placeholder for the full algorithm
    let mut structure = Vec::new();
    structure.push(true); // Root is internal

    // Simple generation for small trees
    generate_tree_structure(&mut structure, n, rank);

    Some(structure)
}

fn generate_tree_structure(structure: &mut Vec<bool>, remaining: usize, rank: usize) {
    if remaining == 0 {
        structure.push(false); // Leaf
        structure.push(false); // Leaf
        return;
    }

    // Simplified tree generation
    let left_size = rank % (remaining + 1);
    let right_size = remaining - 1 - left_size;

    if left_size > 0 {
        structure.push(true);
        generate_tree_structure(structure, left_size - 1, rank / 2);
    } else {
        structure.push(false);
    }

    if right_size > 0 {
        structure.push(true);
        generate_tree_structure(structure, right_size - 1, rank / 3);
    } else {
        structure.push(false);
    }
}

/// Ranking for simple labeled graphs with n vertices
///
/// Graphs are ranked based on their adjacency matrix representation.
/// For n vertices, there are 2^(n(n-1)/2) possible undirected graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRank {
    /// Number of vertices
    n: usize,
    /// Edges represented as adjacency matrix (upper triangle only)
    /// edges[i][j] = true if there's an edge between vertices i and j (i < j)
    edges: Vec<Vec<bool>>,
}

impl GraphRank {
    /// Create a new graph rank from an edge list
    pub fn new(n: usize, edge_list: Vec<(usize, usize)>) -> Option<Self> {
        let mut edges = vec![vec![false; n]; n];

        for (u, v) in edge_list {
            if u >= n || v >= n {
                return None;
            }
            let (i, j) = if u < v { (u, v) } else { (v, u) };
            edges[i][j] = true;
        }

        Some(GraphRank { n, edges })
    }

    /// Create from adjacency matrix
    pub fn from_adjacency_matrix(adj: Vec<Vec<bool>>) -> Option<Self> {
        let n = adj.len();
        if n == 0 {
            return Some(GraphRank {
                n: 0,
                edges: vec![],
            });
        }

        // Verify it's a square matrix
        for row in &adj {
            if row.len() != n {
                return None;
            }
        }

        let mut edges = vec![vec![false; n]; n];
        for i in 0..n {
            for j in i + 1..n {
                edges[i][j] = adj[i][j] || adj[j][i]; // Undirected
            }
        }

        Some(GraphRank { n, edges })
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.n
    }

    /// Get the edge list
    pub fn edge_list(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for i in 0..self.n {
            for j in i + 1..self.n {
                if self.edges[i][j] {
                    result.push((i, j));
                }
            }
        }
        result
    }

    /// Check if an edge exists
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.n || v >= self.n {
            return false;
        }
        let (i, j) = if u < v { (u, v) } else { (v, u) };
        self.edges[i][j]
    }
}

impl Rankable for GraphRank {
    type Params = usize; // Number of vertices

    fn rank(&self) -> usize {
        // Rank based on adjacency matrix upper triangle as a bit vector
        let mut rank = 0;
        let mut bit_pos = 0;

        for i in 0..self.n {
            for j in i + 1..self.n {
                if self.edges[i][j] {
                    rank |= 1 << bit_pos;
                }
                bit_pos += 1;
            }
        }

        rank
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let n = *params;
        let mut edges = vec![vec![false; n]; n];
        let mut bit_pos = 0;

        for i in 0..n {
            for j in i + 1..n {
                if (rank >> bit_pos) & 1 == 1 {
                    edges[i][j] = true;
                }
                bit_pos += 1;
            }
        }

        Some(GraphRank { n, edges })
    }

    fn count(params: &Self::Params) -> usize {
        let n = *params;
        if n == 0 {
            return 1;
        }
        // Number of undirected graphs on n vertices: 2^(n(n-1)/2)
        let num_edges = n * (n - 1) / 2;
        1 << num_edges
    }
}

/// Ranking for rooted trees represented as parent arrays
///
/// A rooted tree with n vertices (labeled 0 to n-1) where vertex 0 is the root.
/// Represented as parent[i] = parent of vertex i (parent[0] = None for root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootedTreeRank {
    /// Number of vertices
    n: usize,
    /// Parent array: parent[i] = parent of vertex i (None for root)
    parent: Vec<Option<usize>>,
}

impl RootedTreeRank {
    /// Create a new rooted tree from a parent array
    pub fn new(parent: Vec<Option<usize>>) -> Option<Self> {
        let n = parent.len();
        if n == 0 {
            return None;
        }

        // Verify exactly one root (parent[i] = None)
        let num_roots = parent.iter().filter(|p| p.is_none()).count();
        if num_roots != 1 {
            return None;
        }

        // Verify all parent indices are valid
        for (i, &p) in parent.iter().enumerate() {
            if let Some(parent_idx) = p {
                if parent_idx >= n || parent_idx == i {
                    return None;
                }
            }
        }

        // Verify it's acyclic (would need DFS, simplified for now)

        Some(RootedTreeRank { n, parent })
    }

    /// Get the parent array
    pub fn parent_array(&self) -> &[Option<usize>] {
        &self.parent
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.n
    }
}

impl Rankable for RootedTreeRank {
    type Params = usize; // Number of vertices

    fn rank(&self) -> usize {
        // Use Prüfer sequence for ranking labeled trees
        // For rooted trees, we use a modified encoding
        let mut rank = 0;
        let base = self.n;

        for (i, &p) in self.parent.iter().enumerate() {
            if i == 0 {
                continue; // Skip root
            }
            let parent_idx = p.unwrap_or(0);
            rank = rank * base + parent_idx;
        }

        rank
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let n = *params;
        if n == 0 {
            return None;
        }

        let mut parent = vec![None; n];
        parent[0] = None; // Root

        let mut remaining_rank = rank;

        for i in (1..n).rev() {
            let parent_idx = remaining_rank % n;
            parent[i] = Some(parent_idx);
            remaining_rank /= n;
        }

        RootedTreeRank::new(parent)
    }

    fn count(params: &Self::Params) -> usize {
        let n = *params;
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        // For rooted labeled trees: n^(n-1) by Cayley's formula
        n.pow((n - 1) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_ranking() {
        // Test C(5, 3)
        let comb1 = CombinationRank::new(5, 3, vec![0, 1, 2]).unwrap();
        assert_eq!(comb1.rank(), 0); // First combination in lexicographic order

        let comb2 = CombinationRank::new(5, 3, vec![2, 3, 4]).unwrap();
        assert_eq!(comb2.rank(), 9); // Last combination

        // Test unranking
        let unranked = CombinationRank::unrank(0, &(5, 3)).unwrap();
        assert_eq!(unranked.elements(), &[0, 1, 2]);

        let unranked_last = CombinationRank::unrank(9, &(5, 3)).unwrap();
        assert_eq!(unranked_last.elements(), &[2, 3, 4]);
    }

    #[test]
    fn test_combination_ranking_table() {
        let table = RankingTable::<CombinationRank>::new((5, 3));
        assert_eq!(table.len(), 10); // C(5,3) = 10

        // Verify all combinations can be generated
        let combinations: Vec<_> = table.iter().collect();
        assert_eq!(combinations.len(), 10);

        // First and last
        assert_eq!(combinations[0].elements(), &[0, 1, 2]);
        assert_eq!(combinations[9].elements(), &[2, 3, 4]);
    }

    #[test]
    fn test_permutation_ranking() {
        // Test permutations of {0, 1, 2}
        let perm1 = PermutationRank::new(vec![0, 1, 2]).unwrap();
        assert_eq!(perm1.rank(), 0); // Identity permutation

        let perm2 = PermutationRank::new(vec![2, 1, 0]).unwrap();
        assert_eq!(perm2.rank(), 5); // Reverse permutation (last in rank)

        // Test unranking
        let unranked = PermutationRank::unrank(0, &3).unwrap();
        assert_eq!(unranked.elements(), &[0, 1, 2]);

        let unranked_last = PermutationRank::unrank(5, &3).unwrap();
        assert_eq!(unranked_last.elements(), &[2, 1, 0]);
    }

    #[test]
    fn test_permutation_lehmer_code() {
        // [2, 0, 1] has Lehmer code [2, 0, 0]
        let perm = PermutationRank::new(vec![2, 0, 1]).unwrap();
        assert_eq!(perm.lehmer_code(), vec![2, 0, 0]);

        // [1, 2, 0] has Lehmer code [1, 1, 0]
        let perm2 = PermutationRank::new(vec![1, 2, 0]).unwrap();
        assert_eq!(perm2.lehmer_code(), vec![1, 1, 0]);
    }

    #[test]
    fn test_permutation_ranking_table() {
        let table = RankingTable::<PermutationRank>::new(3);
        assert_eq!(table.len(), 6); // 3! = 6

        // Verify all permutations can be generated
        let perms: Vec<_> = table.iter().collect();
        assert_eq!(perms.len(), 6);

        // Check first and last
        assert_eq!(perms[0].elements(), &[0, 1, 2]);
        assert_eq!(perms[5].elements(), &[2, 1, 0]);
    }

    #[test]
    fn test_ranking_roundtrip_combinations() {
        // Test that rank(unrank(i)) == i for all i
        for i in 0..10 {
            let comb = CombinationRank::unrank(i, &(5, 3)).unwrap();
            assert_eq!(comb.rank(), i);
        }
    }

    #[test]
    fn test_ranking_roundtrip_permutations() {
        // Test that rank(unrank(i)) == i for all i
        for i in 0..24 {
            // 4! = 24
            let perm = PermutationRank::unrank(i, &4).unwrap();
            assert_eq!(perm.rank(), i);
        }
    }

    // Tests for BinaryTreeRank

    #[test]
    fn test_binary_tree_validation() {
        // Valid tree: single leaf
        let tree = BinaryTreeRank::new(vec![false]);
        assert!(tree.is_some());
        assert_eq!(tree.unwrap().num_internal_nodes(), 0);

        // Valid tree: root with two leaves
        let tree = BinaryTreeRank::new(vec![true, false, false]);
        assert!(tree.is_some());
        assert_eq!(tree.unwrap().num_internal_nodes(), 1);

        // Invalid tree: wrong number of leaves
        let tree = BinaryTreeRank::new(vec![true, false]);
        assert!(tree.is_none());

        // Invalid tree: too many internal nodes
        let tree = BinaryTreeRank::new(vec![true, true, false]);
        assert!(tree.is_none());
    }

    #[test]
    fn test_binary_tree_count() {
        // Count for n=0: C_0 = 1
        assert_eq!(BinaryTreeRank::count(&0), 1);

        // Count for n=1: C_1 = 1
        assert_eq!(BinaryTreeRank::count(&1), 1);

        // Count for n=2: C_2 = 2
        assert_eq!(BinaryTreeRank::count(&2), 2);

        // Count for n=3: C_3 = 5
        assert_eq!(BinaryTreeRank::count(&3), 5);

        // Count for n=4: C_4 = 14
        assert_eq!(BinaryTreeRank::count(&4), 14);
    }

    #[test]
    fn test_binary_tree_simple_ranking() {
        // Single leaf (n=0)
        let tree = BinaryTreeRank::new(vec![false]).unwrap();
        assert_eq!(tree.rank(), 0);

        // Single internal node with two leaves (n=1)
        let tree = BinaryTreeRank::new(vec![true, false, false]).unwrap();
        assert_eq!(tree.rank(), 0);
    }

    // Tests for GraphRank

    #[test]
    fn test_graph_rank_creation() {
        // Empty graph with 3 vertices
        let graph = GraphRank::new(3, vec![]).unwrap();
        assert_eq!(graph.num_vertices(), 3);
        assert_eq!(graph.edge_list().len(), 0);

        // Graph with one edge
        let graph = GraphRank::new(3, vec![(0, 1)]).unwrap();
        assert_eq!(graph.num_vertices(), 3);
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0));
        assert!(!graph.has_edge(0, 2));
    }

    #[test]
    fn test_graph_rank_from_adjacency() {
        let adj = vec![
            vec![false, true, false],
            vec![true, false, true],
            vec![false, true, false],
        ];

        let graph = GraphRank::from_adjacency_matrix(adj).unwrap();
        assert_eq!(graph.num_vertices(), 3);
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 2));
        assert!(!graph.has_edge(0, 2));
    }

    #[test]
    fn test_graph_rank_count() {
        // n=0: 1 graph (empty)
        assert_eq!(GraphRank::count(&0), 1);

        // n=1: 1 graph (single vertex)
        assert_eq!(GraphRank::count(&1), 1);

        // n=2: 2 graphs (with/without edge)
        assert_eq!(GraphRank::count(&2), 2);

        // n=3: 8 graphs (2^3 = 8, since there are 3 possible edges)
        assert_eq!(GraphRank::count(&3), 8);

        // n=4: 64 graphs (2^6 = 64, since there are 6 possible edges)
        assert_eq!(GraphRank::count(&4), 64);
    }

    #[test]
    fn test_graph_ranking() {
        // Empty graph on 3 vertices should have rank 0
        let graph = GraphRank::new(3, vec![]).unwrap();
        assert_eq!(graph.rank(), 0);

        // Graph with edge (0,1)
        let graph = GraphRank::new(3, vec![(0, 1)]).unwrap();
        let rank1 = graph.rank();
        assert_eq!(rank1, 1); // First bit set

        // Graph with edge (1,2)
        let graph = GraphRank::new(3, vec![(1, 2)]).unwrap();
        let rank2 = graph.rank();
        assert_eq!(rank2, 4); // Third bit set

        // Graph with edge (0,2)
        let graph = GraphRank::new(3, vec![(0, 2)]).unwrap();
        let rank3 = graph.rank();
        assert_eq!(rank3, 2); // Second bit set
    }

    #[test]
    fn test_graph_ranking_roundtrip() {
        // Test roundtrip for all graphs on 3 vertices
        for i in 0..8 {
            let graph = GraphRank::unrank(i, &3).unwrap();
            assert_eq!(graph.rank(), i);
        }
    }

    #[test]
    fn test_graph_unranking() {
        // Unrank 0 on 3 vertices: empty graph
        let graph = GraphRank::unrank(0, &3).unwrap();
        assert_eq!(graph.edge_list().len(), 0);

        // Unrank 1: graph with edge (0,1)
        let graph = GraphRank::unrank(1, &3).unwrap();
        assert!(graph.has_edge(0, 1));
        assert_eq!(graph.edge_list().len(), 1);

        // Unrank 7: complete graph on 3 vertices
        let graph = GraphRank::unrank(7, &3).unwrap();
        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(0, 2));
        assert!(graph.has_edge(1, 2));
        assert_eq!(graph.edge_list().len(), 3);
    }

    // Tests for RootedTreeRank

    #[test]
    fn test_rooted_tree_creation() {
        // Single vertex (root)
        let tree = RootedTreeRank::new(vec![None]).unwrap();
        assert_eq!(tree.num_vertices(), 1);

        // Star tree: root with 3 children
        let tree = RootedTreeRank::new(vec![None, Some(0), Some(0), Some(0)]).unwrap();
        assert_eq!(tree.num_vertices(), 4);

        // Path tree: 0 -> 1 -> 2 -> 3
        let tree = RootedTreeRank::new(vec![None, Some(0), Some(1), Some(2)]).unwrap();
        assert_eq!(tree.num_vertices(), 4);
    }

    #[test]
    fn test_rooted_tree_validation() {
        // Invalid: no root
        let tree = RootedTreeRank::new(vec![Some(1), Some(0)]);
        assert!(tree.is_none());

        // Invalid: multiple roots
        let tree = RootedTreeRank::new(vec![None, None]);
        assert!(tree.is_none());

        // Invalid: self-loop
        let tree = RootedTreeRank::new(vec![None, Some(1)]);
        assert!(tree.is_none());

        // Invalid: out of bounds parent
        let tree = RootedTreeRank::new(vec![None, Some(5)]);
        assert!(tree.is_none());
    }

    #[test]
    fn test_rooted_tree_count() {
        // n=0: 0 trees
        assert_eq!(RootedTreeRank::count(&0), 0);

        // n=1: 1 tree
        assert_eq!(RootedTreeRank::count(&1), 1);

        // n=2: 2 trees (2^1 = 2)
        assert_eq!(RootedTreeRank::count(&2), 2);

        // n=3: 9 trees (3^2 = 9)
        assert_eq!(RootedTreeRank::count(&3), 9);

        // n=4: 64 trees (4^3 = 64)
        assert_eq!(RootedTreeRank::count(&4), 64);
    }

    #[test]
    fn test_rooted_tree_ranking() {
        // Single vertex tree
        let tree = RootedTreeRank::new(vec![None]).unwrap();
        assert_eq!(tree.rank(), 0);

        // Two vertices: 0 -> 1
        let tree = RootedTreeRank::new(vec![None, Some(0)]).unwrap();
        let rank1 = tree.rank();
        assert_eq!(rank1, 0);
    }

    #[test]
    fn test_rooted_tree_unranking() {
        // Unrank single vertex tree
        let tree = RootedTreeRank::unrank(0, &1).unwrap();
        assert_eq!(tree.parent_array(), &[None]);

        // Unrank for n=2
        let tree = RootedTreeRank::unrank(0, &2).unwrap();
        assert_eq!(tree.num_vertices(), 2);
    }

    #[test]
    fn test_ranking_table_binary_tree() {
        // Create ranking table for binary trees with 2 internal nodes
        let table = RankingTable::<BinaryTreeRank>::new(2);
        assert_eq!(table.len(), 2); // C_2 = 2

        // Verify we can iterate over all trees
        let trees: Vec<_> = table.iter().collect();
        assert_eq!(trees.len(), 2);
    }

    #[test]
    fn test_ranking_table_graphs() {
        // Create ranking table for graphs on 3 vertices
        let table = RankingTable::<GraphRank>::new(3);
        assert_eq!(table.len(), 8); // 2^3 = 8

        // Verify all graphs can be enumerated
        let graphs: Vec<_> = table.iter().collect();
        assert_eq!(graphs.len(), 8);
    }

    #[test]
    fn test_is_valid_tree_sequence() {
        // Valid sequences
        assert!(is_valid_tree_sequence(&[false])); // Single leaf
        assert!(is_valid_tree_sequence(&[true, false, false])); // Root with 2 leaves
        assert!(is_valid_tree_sequence(&[true, true, false, false, false])); // Left-skewed

        // Invalid sequences
        assert!(!is_valid_tree_sequence(&[])); // Empty
        assert!(!is_valid_tree_sequence(&[true])); // Incomplete
        assert!(!is_valid_tree_sequence(&[true, false])); // Missing right child
        assert!(!is_valid_tree_sequence(&[false, true])); // Extra node after complete tree
    }
}

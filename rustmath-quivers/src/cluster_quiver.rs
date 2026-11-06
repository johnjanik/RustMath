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
//! # Conventions
//!
//! `b_ij` = (number of arrows i -> j) - (number of arrows j -> i), as in Fomin-Zelevinsky
//! and in SageMath's `ClusterQuiver(...).b_matrix()`. The principal part must be
//! skew-symmetrizable for the classification results below to apply.
//!
//! No chapter of the MAGMA Handbook covers cluster algebras, so SageMath
//! (`sage.combinat.cluster_algebra_quiver`) is the reference for API shape and test values.
//!
//! # References
//!
//! - [FZ1] Fomin & Zelevinsky, "Cluster algebras I: Foundations", J. AMS 15 (2002);
//!   matrix mutation, eq. (4.3).
//! - [FZ2] Fomin & Zelevinsky, "Cluster algebras II: Finite type classification",
//!   Invent. Math. 154 (2003); Cartan counterpart (1.6), Thm 1.4 (finite type), Thm 1.8
//!   (finite type iff 2-finite).
//! - [Kac] Kac, "Infinite dimensional Lie algebras", 3rd ed., ch. 4 (finite / affine /
//!   indefinite trichotomy of generalized Cartan matrices, Thm 4.3, Lemma 4.5, Prop. 4.7).
//! - [CK] Caldero & Keller, "From triangulated categories to cluster algebras II",
//!   Ann. Sci. ENS 39 (2006): acyclic quivers in one mutation class are related by
//!   sink/source mutations (so they share their underlying graph).
//! - [FST] Felikson, Shapiro & Tumarkin, "Skew-symmetric cluster algebras of finite mutation
//!   type", J. EMS 14 (2012): the affine quivers are mutation-finite, and a connected quiver
//!   with at least 3 vertices and an arrow of multiplicity >= 3 is mutation-infinite.
//! - Keller, "Cluster algebras, quiver representations and triangulated categories" (2012)

use crate::quiver::Quiver;
use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Display};

/// Mutation type classification for cluster algebras
///
/// For a connected (indecomposable) skew-symmetrizable exchange matrix B:
/// - `Finite`: finitely many clusters, i.e. B is mutation-equivalent to an acyclic matrix
///   whose Cartan companion is a Cartan matrix of finite type ([FZ2] Thm 1.4).
/// - `Affine`: not finite, and B is mutation-equivalent to an acyclic matrix whose Cartan
///   companion is a generalized Cartan matrix of affine type ([Kac] ch. 4), e.g. the
///   Kronecker quiver.
/// - `Infinite`: neither of the above (wild acyclic type, or not mutation-acyclic at all,
///   e.g. the Markov quiver).
///
/// For a disconnected B: `Finite` if every component is finite, `Infinite` if some
/// component is `Infinite`, and `Affine` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationType {
    /// Finite type (finitely many clusters)
    Finite,
    /// Affine type (infinitely many clusters; tame)
    Affine,
    /// Neither finite nor affine (infinitely many clusters): an acyclic member of wild
    /// (indefinite) type, or no acyclic member at all, as for the Markov quiver
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
    /// B[i][j] = number of arrows from i to j minus number of arrows from j to i
    /// (Fomin-Zelevinsky / SageMath convention)
    exchange_matrix: Vec<Vec<i64>>,
    /// Frozen vertices (cannot be mutated)
    frozen_vertices: HashSet<usize>,
    /// History of mutations applied
    mutation_history: MutationSequence,
}

impl ClusterQuiver {
    /// Create a new cluster quiver from an exchange matrix
    ///
    /// The principal part (top `rank x rank` block, `rank` = number of columns) should be
    /// skew-symmetrizable. Rows beyond rank represent frozen vertices (coefficients).
    ///
    /// Returns an error if the rows have different lengths or there are fewer rows than
    /// columns (the principal part would not be square).
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

        if exchange_matrix.len() < rank {
            return Err(format!(
                "Exchange matrix has {} rows but {} columns; need at least as many rows as columns",
                exchange_matrix.len(),
                rank
            ));
        }

        Ok(ClusterQuiver {
            rank,
            exchange_matrix,
            frozen_vertices: HashSet::new(),
            mutation_history: MutationSequence::new(),
        })
    }

    /// Wrap a square (coefficient-free) exchange matrix.
    fn from_square(matrix: Vec<Vec<i64>>) -> Self {
        ClusterQuiver {
            rank: matrix.len(),
            exchange_matrix: matrix,
            frozen_vertices: HashSet::new(),
            mutation_history: MutationSequence::new(),
        }
    }

    /// Create a cluster quiver of type A_n
    ///
    /// Creates the linearly oriented quiver 0 -> 1 -> 2 -> ... -> (n-1), i.e.
    /// `b_{i,i+1} = 1`. (SageMath's standard `['A', n]` quiver uses the bipartite
    /// orientation instead; both are of type A_n.)
    pub fn type_a(n: usize) -> Self {
        if n == 0 {
            panic!("Type A_n requires n >= 1");
        }

        let mut matrix = vec![vec![0i64; n]; n];
        for i in 0..n - 1 {
            matrix[i][i + 1] = 1;
            matrix[i + 1][i] = -1;
        }

        Self::from_square(matrix)
    }

    /// Create a cluster quiver of type D_n (n >= 4)
    ///
    /// Dynkin diagram: the path 0 - 1 - ... - (n-2) plus the edge (n-3) - (n-1), so the
    /// fork is at vertex n-3. The orientation is SageMath's standard one for `['D', n]`
    /// (bipartite: vertices at even distance from 0 are sources). For n = 4 this is
    /// 0 -> 1 <- 2, 3 -> 1, matching Sage's `ClusterQuiver(['D',4]).b_matrix()`.
    pub fn type_d(n: usize) -> Result<Self, String> {
        if n < 4 {
            return Err("Type D_n requires n >= 4".to_string());
        }

        let mut edges: Vec<(usize, usize)> = (0..n - 2).map(|i| (i, i + 1)).collect();
        edges.push((n - 3, n - 1));

        Ok(Self::from_square(bipartite_tree_matrix(n, &edges)))
    }

    /// Create a cluster quiver of type E_n (n = 6, 7, 8)
    ///
    /// Dynkin diagram: the path 0 - 1 - ... - (n-2) plus the branch 2 - (n-1), as in
    /// SageMath's `['E', n]`, with Sage's bipartite orientation (vertices at even distance
    /// from 0 are sources).
    pub fn type_e(n: usize) -> Result<Self, String> {
        if n < 6 || n > 8 {
            return Err("Type E_n requires n ∈ {6, 7, 8}".to_string());
        }

        let mut edges: Vec<(usize, usize)> = (0..n - 2).map(|i| (i, i + 1)).collect();
        edges.push((2, n - 1));

        Ok(Self::from_square(bipartite_tree_matrix(n, &edges)))
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
    /// Returns an error if k is out of bounds or frozen, or if an entry would overflow i64.
    pub fn mutate(&mut self, k: usize) -> Result<(), String> {
        if k >= self.rank {
            return Err(format!("Vertex {} out of bounds (rank = {})", k, self.rank));
        }

        if self.is_frozen(k) {
            return Err(format!("Vertex {} is frozen", k));
        }

        let new_matrix = matrix_mutation(&self.exchange_matrix, k)
            .ok_or_else(|| format!("Mutation at vertex {} overflows i64", k))?;

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

    /// Decide (labelled) mutation equivalence by breadth-first search
    ///
    /// Searches the mutation class of `self` (mutations at non-frozen vertices only, no
    /// relabelling of vertices, frozen rows mutated along) for the full exchange matrix
    /// of `other`, using at most `max_depth` mutations.
    ///
    /// The set of vertices frozen with [`ClusterQuiver::freeze_vertex`] is part of the
    /// data, like the shape: mutation is only defined at non-frozen vertices and never
    /// changes which vertices are frozen, so every member of the mutation class of `self`
    /// has `self`'s frozen set. Quivers with different frozen sets are therefore never
    /// mutation-equivalent. (Searching with `self`'s frozen set alone would make the
    /// answer depend on the argument order.) The mutation history is not compared.
    ///
    /// Returns:
    /// - `Some(true)` if `other` was reached (a proof of equivalence);
    /// - `Some(false)` if the matrices have different shapes or the frozen sets differ, or
    ///   the whole labelled mutation class of `self` lies within `max_depth` mutations and
    ///   was enumerated without meeting `other` (a proof of non-equivalence);
    /// - `None` if the depth bound cut the search off before either happened (undecided).
    pub fn is_mutation_equivalent(&self, other: &ClusterQuiver, max_depth: usize) -> Option<bool> {
        if self.rank != other.rank || self.exchange_matrix.len() != other.exchange_matrix.len() {
            return Some(false);
        }
        if self.frozen_vertices != other.frozen_vertices {
            return Some(false);
        }

        let mutable: Vec<usize> = (0..self.rank).filter(|&k| !self.is_frozen(k)).collect();
        let mut visited: HashSet<Vec<Vec<i64>>> = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(self.exchange_matrix.clone());
        queue.push_back((self.exchange_matrix.clone(), 0usize));
        let mut truncated = false;

        while let Some((matrix, depth)) = queue.pop_front() {
            if matrix == other.exchange_matrix {
                return Some(true);
            }

            for &k in &mutable {
                let next = match matrix_mutation(&matrix, k) {
                    Some(next) => next,
                    None => {
                        // Entry overflow: this branch cannot be explored.
                        truncated = true;
                        continue;
                    }
                };
                if visited.contains(&next) {
                    continue;
                }
                if depth >= max_depth {
                    truncated = true;
                    continue;
                }
                visited.insert(next.clone());
                queue.push_back((next, depth + 1));
            }
        }

        if truncated {
            None
        } else {
            Some(false)
        }
    }

    /// Determine the mutation type (finite, affine, or infinite) of the cluster algebra
    ///
    /// Works on the principal part restricted to the non-frozen vertices (vertices frozen
    /// with [`ClusterQuiver::freeze_vertex`] are dropped, unlike in
    /// [`ClusterQuiver::cartan_companion`]), one connected component at a time (see
    /// [`MutationType`] for how components are combined). For a connected component, the
    /// labelled mutation class is explored breadth-first, storing at most
    /// `max(max_class_size, 1)` matrices for that component: the starting matrix is always
    /// stored and examined, so `max_class_size = 0` behaves like `1`. A type is returned
    /// only when a theorem decides it:
    ///
    /// - an acyclic member whose Cartan companion is of finite type gives `Finite`
    ///   ([FZ2] Thm 1.4);
    /// - if the whole class was enumerated and every member has `|b_ij b_ji| <= 3`,
    ///   B is 2-finite, hence `Finite` ([FZ2] Thm 1.8);
    /// - an acyclic member with affine Cartan companion, together with a member having
    ///   `|b_ij b_ji| >= 4` (so not finite, [FZ2] Thm 1.8), gives `Affine`;
    /// - if the whole class was enumerated, it is not 2-finite and it has no acyclic
    ///   member of affine type, the type is `Infinite`;
    /// - skew-symmetric B only: the first acyclic member decides the type, since all
    ///   acyclic members share their underlying graph [CK]; and an entry `|b_ij| >= 3`
    ///   gives `Infinite` (mutation-infinite, while finite and affine quivers are
    ///   mutation-finite with multiplicities <= 2 [FST]).
    ///
    /// The Cartan companion is classified with the finite / affine / indefinite
    /// trichotomy of [Kac] ch. 4, using exact integer determinants.
    ///
    /// Returns `Err` if the principal part is not skew-symmetrizable, or if the bound was
    /// reached (or an entry overflowed) before the type was decided: a bounded search that
    /// finds nothing never decides.
    pub fn mutation_type(&self, max_class_size: usize) -> Result<MutationType, String> {
        let principal = self.principal_part();
        let mutable: Vec<usize> = (0..self.rank).filter(|&k| !self.is_frozen(k)).collect();
        let b = submatrix(&principal, &mutable);

        if !is_skew_symmetrizable(&b)? {
            return Err(
                "Principal part (on the non-frozen vertices) is not skew-symmetrizable".to_string(),
            );
        }

        let mut any_affine = false;
        let mut unresolved: Option<String> = None;
        for component in connected_components(&b) {
            match classify_connected(&submatrix(&b, &component), max_class_size) {
                Ok(MutationType::Infinite) => return Ok(MutationType::Infinite),
                Ok(MutationType::Affine) => any_affine = true,
                Ok(MutationType::Finite) => {}
                Err(e) => unresolved = Some(e),
            }
        }

        if let Some(e) = unresolved {
            return Err(e);
        }
        Ok(if any_affine {
            MutationType::Affine
        } else {
            MutationType::Finite
        })
    }

    /// Convert the principal part to a basic Quiver representation
    ///
    /// Creates a directed graph where:
    /// - Vertices are labeled 0, 1, ..., rank-1
    /// - There are `b_ij` arrows i -> j whenever `b_ij > 0` (FZ / SageMath convention)
    /// - Arrows are labelled `a_i_j` (single arrow) or `a_i_j_t` (t-th of several)
    ///
    /// Returns an error if the principal part is not skew-symmetric: a skew-symmetrizable
    /// but not skew-symmetric matrix needs valued arrows, which `Quiver` cannot express.
    pub fn to_quiver(&self) -> Result<Quiver, String> {
        let b = self.principal_part();
        for i in 0..self.rank {
            for j in 0..self.rank {
                if b[i][j] != -b[j][i] {
                    return Err(format!(
                        "Principal part is not skew-symmetric (b[{}][{}] = {}, b[{}][{}] = {}); \
                         a plain quiver cannot represent it",
                        i, j, b[i][j], j, i, b[j][i]
                    ));
                }
            }
        }

        let mut quiver = Quiver::new(self.rank);
        for i in 0..self.rank {
            for j in 0..self.rank {
                let b_ij = b[i][j];
                if i != j && b_ij > 0 {
                    for t in 0..b_ij {
                        let label = if b_ij == 1 {
                            format!("a_{}_{}", i, j)
                        } else {
                            format!("a_{}_{}_{}", i, j, t)
                        };
                        quiver.add_edge(i, j, label)?;
                    }
                }
            }
        }

        Ok(quiver)
    }

    /// Check if exchange matrices are equal (only used by the tests since the BFS
    /// routines compare raw matrices)
    #[cfg(test)]
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

    /// Get the Cartan companion A(B) of the principal part B
    ///
    /// Fomin-Zelevinsky's Cartan counterpart [FZ2, eq. (1.6)]: `a_ii = 2` and
    /// `a_ij = -|b_ij|` for `i != j`. For a skew-symmetrizable B this is a symmetrizable
    /// generalized Cartan matrix.
    ///
    /// B is the whole `rank x rank` block returned by [`ClusterQuiver::principal_part`],
    /// and the result is indexed by vertex like it:
    /// - coefficient rows (the rows beyond `rank`, as added by
    ///   [`ClusterQuiver::add_coefficients`]) are ignored;
    /// - vertices frozen with [`ClusterQuiver::freeze_vertex`] are kept.
    ///
    /// [`ClusterQuiver::mutation_type`] instead classifies the principal part restricted
    /// to the non-frozen vertices. The companion it uses is the principal submatrix of
    /// this one on those vertices, since the companion is computed entry by entry.
    ///
    /// The cluster algebra is of finite type iff B is mutation-equivalent to some B' whose
    /// companion A(B') is a Cartan matrix of finite type [FZ2, Thm 1.4]. A(B) of B itself
    /// need not be of finite type: the oriented 3-cycle lies in the A_3 mutation class but
    /// its companion is the affine matrix of type A~_2. Use [`ClusterQuiver::mutation_type`]
    /// for the classification.
    pub fn cartan_companion(&self) -> Vec<Vec<i64>> {
        cartan_companion_of(&self.principal_part())
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

/// Matrix mutation mu_k of an extended exchange matrix (rows >= columns) [FZ1, eq. (4.3)]:
/// `b'_ij = -b_ij` if `k` is `i` or `j`, else `b_ij + sgn(b_ik) [b_ik b_kj]_+`.
///
/// Returns `None` if an entry overflows i64.
fn matrix_mutation(b: &[Vec<i64>], k: usize) -> Option<Vec<Vec<i64>>> {
    let cols = b.first().map_or(0, |row| row.len());
    let mut out = b.to_vec();
    for i in 0..b.len() {
        for j in 0..cols {
            if i == k || j == k {
                out[i][j] = b[i][j].checked_neg()?;
            } else {
                let b_ik = b[i][k];
                let b_kj = b[k][j];
                if b_ik > 0 && b_kj > 0 {
                    out[i][j] = b[i][j].checked_add(b_ik.checked_mul(b_kj)?)?;
                } else if b_ik < 0 && b_kj < 0 {
                    out[i][j] = b[i][j].checked_sub(b_ik.checked_mul(b_kj)?)?;
                }
            }
        }
    }
    Some(out)
}

/// Skew-symmetric exchange matrix of a tree on `n` vertices with SageMath's standard
/// bipartite orientation (`_bipartite_graph_to_digraph`): every edge points away from its
/// endpoint at even distance from vertex 0.
fn bipartite_tree_matrix(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<i64>> {
    let mut adjacency = vec![Vec::new(); n];
    for &(u, v) in edges {
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    let mut even: Vec<Option<bool>> = vec![None; n];
    let mut queue = VecDeque::new();
    even[0] = Some(true);
    queue.push_back(0);
    while let Some(u) = queue.pop_front() {
        let parity = even[u] == Some(true);
        for &v in &adjacency[u] {
            if even[v].is_none() {
                even[v] = Some(!parity);
                queue.push_back(v);
            }
        }
    }

    let mut matrix = vec![vec![0i64; n]; n];
    for &(u, v) in edges {
        let (source, target) = if even[u] == Some(true) { (u, v) } else { (v, u) };
        matrix[source][target] = 1;
        matrix[target][source] = -1;
    }
    matrix
}

/// Cartan companion of a square matrix: `a_ii = 2`, `a_ij = -|b_ij|` [FZ2, eq. (1.6)].
fn cartan_companion_of(b: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let n = b.len();
    let mut a = vec![vec![0i64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = if i == j { 2 } else { -b[i][j].abs() };
        }
    }
    a
}

/// Square submatrix on the given (ordered) indices.
fn submatrix(b: &[Vec<i64>], indices: &[usize]) -> Vec<Vec<i64>> {
    indices
        .iter()
        .map(|&i| indices.iter().map(|&j| b[i][j]).collect())
        .collect()
}

/// Connected components of the graph with an edge i - j whenever `b_ij != 0` or `b_ji != 0`.
fn connected_components(b: &[Vec<i64>]) -> Vec<Vec<usize>> {
    let n = b.len();
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut component = vec![start];
        let mut queue = VecDeque::from([start]);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if !seen[v] && (b[u][v] != 0 || b[v][u] != 0) {
                    seen[v] = true;
                    component.push(v);
                    queue.push_back(v);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Whether a square matrix is skew-symmetrizable: there are `d_i > 0` with
/// `d_i b_ij = -d_j b_ji` for all i, j. The `d_i` are propagated along a spanning forest
/// as reduced fractions and every entry is then checked.
///
/// Returns `Err` only on i128 overflow of the symmetrizer.
fn is_skew_symmetrizable(b: &[Vec<i64>]) -> Result<bool, String> {
    let n = b.len();
    for i in 0..n {
        if b[i][i] != 0 {
            return Ok(false);
        }
        for j in 0..n {
            // sign-skew-symmetry: b_ij and b_ji are both zero or of opposite signs
            if (b[i][j] == 0) != (b[j][i] == 0) || (b[i][j] != 0 && (b[i][j] > 0) == (b[j][i] > 0))
            {
                return Ok(false);
            }
        }
    }

    let overflow = || "Overflow while computing a skew-symmetrizer".to_string();
    // d[i] = (numerator, denominator), both positive and coprime
    let mut d: Vec<Option<(i128, i128)>> = vec![None; n];
    for start in 0..n {
        if d[start].is_some() {
            continue;
        }
        d[start] = Some((1, 1));
        let mut queue = VecDeque::from([start]);
        while let Some(i) = queue.pop_front() {
            let (p_i, q_i) = d[i].expect("assigned before queueing");
            for j in 0..n {
                if b[i][j] == 0 {
                    continue;
                }
                let bij = (b[i][j] as i128).abs();
                let bji = (b[j][i] as i128).abs();
                match d[j] {
                    // d_i |b_ij| = d_j |b_ji|
                    Some((p_j, q_j)) => {
                        let lhs = p_i
                            .checked_mul(q_j)
                            .and_then(|x| x.checked_mul(bij))
                            .ok_or_else(overflow)?;
                        let rhs = p_j
                            .checked_mul(q_i)
                            .and_then(|x| x.checked_mul(bji))
                            .ok_or_else(overflow)?;
                        if lhs != rhs {
                            return Ok(false);
                        }
                    }
                    None => {
                        let p = p_i.checked_mul(bij).ok_or_else(overflow)?;
                        let q = q_i.checked_mul(bji).ok_or_else(overflow)?;
                        let g = gcd_i128(p, q);
                        d[j] = Some((p / g, q / g));
                        queue.push_back(j);
                    }
                }
            }
        }
    }
    Ok(true)
}

/// Whether the quiver of a square matrix (arrow i -> j iff `b_ij > 0`) has no oriented
/// cycle (Kahn's algorithm).
fn is_acyclic_matrix(b: &[Vec<i64>]) -> bool {
    let n = b.len();
    let mut in_degree: Vec<usize> = (0..n)
        .map(|j| (0..n).filter(|&i| b[i][j] > 0).count())
        .collect();
    let mut stack: Vec<usize> = (0..n).filter(|&j| in_degree[j] == 0).collect();
    let mut removed = 0;
    while let Some(i) = stack.pop() {
        removed += 1;
        for j in 0..n {
            if b[i][j] > 0 {
                in_degree[j] -= 1;
                if in_degree[j] == 0 {
                    stack.push(j);
                }
            }
        }
    }
    removed == n
}

/// Kac's trichotomy for an indecomposable generalized Cartan matrix [Kac, Thm 4.3].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GcmKind {
    Finite,
    Affine,
    Indefinite,
}

/// Whether the leading principal minors of orders 1..=m of `a` are all positive.
///
/// Fraction-free (Bareiss) elimination without pivoting: after step k the pivot is the
/// leading principal minor of order k + 1. `None` on i128 overflow.
fn leading_minors_positive(a: &[Vec<i64>], m: usize) -> Option<bool> {
    let mut w: Vec<Vec<i128>> = (0..m)
        .map(|i| (0..m).map(|j| a[i][j] as i128).collect())
        .collect();
    let mut previous: i128 = 1;
    for k in 0..m {
        let pivot = w[k][k];
        if pivot <= 0 {
            return Some(false);
        }
        for i in k + 1..m {
            for j in k + 1..m {
                let value = w[i][j]
                    .checked_mul(pivot)?
                    .checked_sub(w[i][k].checked_mul(w[k][j])?)?;
                w[i][j] = value / previous;
            }
        }
        previous = pivot;
    }
    Some(true)
}

/// Exact determinant by fraction-free (Bareiss) elimination with row pivoting.
/// `None` on i128 overflow.
fn determinant(a: &[Vec<i64>]) -> Option<i128> {
    let n = a.len();
    let mut w: Vec<Vec<i128>> = a
        .iter()
        .map(|row| row.iter().map(|&x| x as i128).collect())
        .collect();
    let mut sign: i128 = 1;
    let mut previous: i128 = 1;
    for k in 0..n {
        if w[k][k] == 0 {
            match (k + 1..n).find(|&r| w[r][k] != 0) {
                Some(r) => {
                    w.swap(k, r);
                    sign = -sign;
                }
                None => return Some(0),
            }
        }
        let pivot = w[k][k];
        for i in k + 1..n {
            for j in k + 1..n {
                let value = w[i][j]
                    .checked_mul(pivot)?
                    .checked_sub(w[i][k].checked_mul(w[k][j])?)?;
                w[i][j] = value / previous;
            }
        }
        previous = pivot;
    }
    if n == 0 {
        Some(1)
    } else {
        sign.checked_mul(w[n - 1][n - 1])
    }
}

/// Classify an indecomposable symmetrizable GCM `a` (n >= 1) as finite, affine or indefinite.
///
/// With S = DA symmetric (D positive diagonal), principal minors of S and A have the same
/// signs. Finite iff S is positive definite (Sylvester) [Kac, Prop. 4.7]. Affine iff S is
/// positive semidefinite of corank 1 [Kac, Prop. 4.7]; if A is affine, the proper principal
/// submatrix on vertices 0..n-1 is of finite type [Kac, Lemma 4.5] and det A = 0.
/// Conversely det A = 0 with that submatrix positive definite gives, by Cauchy interlacing,
/// n - 1 positive eigenvalues and one zero eigenvalue of S.
fn classify_gcm(a: &[Vec<i64>]) -> Result<GcmKind, String> {
    let n = a.len();
    let overflow = || "Overflow while classifying a Cartan companion".to_string();
    if leading_minors_positive(a, n).ok_or_else(overflow)? {
        return Ok(GcmKind::Finite);
    }
    if n >= 2
        && determinant(a).ok_or_else(overflow)? == 0
        && leading_minors_positive(a, n - 1).ok_or_else(overflow)?
    {
        return Ok(GcmKind::Affine);
    }
    Ok(GcmKind::Indefinite)
}

/// Mutation type of a connected skew-symmetrizable square matrix; see
/// [`ClusterQuiver::mutation_type`] for the decision rules and their sources.
fn classify_connected(b: &[Vec<i64>], max_class_size: usize) -> Result<MutationType, String> {
    let n = b.len();
    if n <= 1 {
        // rank 0 (no cluster variables) or A_1
        return Ok(MutationType::Finite);
    }
    let skew_symmetric = (0..n).all(|i| (0..n).all(|j| b[i][j] == -b[j][i]));

    // The starting matrix is always stored, so a bound of 0 acts as 1.
    let limit = max_class_size.max(1);
    let mut visited: HashSet<Vec<Vec<i64>>> = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(b.to_vec());
    queue.push_back(b.to_vec());

    let mut not_finite = false; // some member has |b_ij b_ji| >= 4
    let mut affine_acyclic = false; // some member is acyclic with affine Cartan companion
    let mut truncated = false;

    while let Some(m) = queue.pop_front() {
        for i in 0..n {
            for j in i + 1..n {
                if (m[i][j] as i128 * m[j][i] as i128).abs() >= 4 {
                    not_finite = true;
                }
                if skew_symmetric && m[i][j].abs() >= 3 {
                    // rank 2: |b_12 b_21| >= 9, an indefinite Cartan companion;
                    // rank >= 3: mutation-infinite, but finite and affine types are
                    // mutation-finite [FST].
                    return Ok(MutationType::Infinite);
                }
            }
        }

        if is_acyclic_matrix(&m) {
            match classify_gcm(&cartan_companion_of(&m))? {
                GcmKind::Finite => return Ok(MutationType::Finite),
                GcmKind::Affine => {
                    if skew_symmetric {
                        return Ok(MutationType::Affine);
                    }
                    affine_acyclic = true;
                }
                GcmKind::Indefinite => {
                    if skew_symmetric {
                        return Ok(MutationType::Infinite);
                    }
                }
            }
        }

        if not_finite && affine_acyclic {
            return Ok(MutationType::Affine);
        }

        for k in 0..n {
            let next = matrix_mutation(&m, k).ok_or_else(|| {
                "Mutation type undecided: an exchange-matrix entry overflowed i64".to_string()
            })?;
            if visited.contains(&next) {
                continue;
            }
            if visited.len() >= limit {
                truncated = true;
                continue;
            }
            visited.insert(next.clone());
            queue.push_back(next);
        }
    }

    if truncated {
        return Err(format!(
            "Mutation type undecided: the mutation class has more than {} matrices",
            limit
        ));
    }
    // The whole labelled mutation class has been enumerated.
    if !not_finite {
        // 2-finite [FZ2, Thm 1.8]
        return Ok(MutationType::Finite);
    }
    // Not 2-finite, and no acyclic member of affine type exists.
    Ok(MutationType::Infinite)
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

        // For A_2, C is the Cartan matrix [[2, -1], [-1, 2]] (FZ II eq. (1.6):
        // a_ii = 2, a_ij = -|b_ij|; GAP: CartanMatrix of SimpleLieAlgebra("A",2,Rationals))
        assert_eq!(c[0][0], 2);
        assert_eq!(c[0][1], -1);
        assert_eq!(c[1][0], -1);
        assert_eq!(c[1][1], 2);
    }

    #[test]
    fn test_cartan_companion_skew_symmetrizable_and_frozen_rows() {
        // B_2-type exchange matrix: companion [[2, -1], [-2, 2]] (det 2, finite type)
        let mut q = ClusterQuiver::new(vec![vec![0, 1], vec![-2, 0]]).unwrap();
        assert_eq!(q.cartan_companion(), vec![vec![2, -1], vec![-2, 2]]);

        // Coefficient rows do not enter the companion of the principal part
        q.add_coefficients(2);
        q.set_entry(2, 0, 5);
        q.set_entry(3, 1, -7);
        assert_eq!(q.cartan_companion(), vec![vec![2, -1], vec![-2, 2]]);

        // The companion of the oriented 3-cycle (A_3 class) is the affine A~_2 matrix
        let cycle = ClusterQuiver::new(vec![vec![0, 1, -1], vec![-1, 0, 1], vec![1, -1, 0]])
            .unwrap();
        assert_eq!(
            cycle.cartan_companion(),
            vec![vec![2, -1, -1], vec![-1, 2, -1], vec![-1, -1, 2]]
        );
        assert_eq!(determinant(&cycle.cartan_companion()), Some(0));
    }

    #[test]
    fn test_cartan_companion_dynkin_reference() {
        // GAP 4: CartanMatrix(RootSystem(SimpleLieAlgebra("D",4,Rationals)))
        let gap_d4 = vec![
            vec![2, -1, 0, 0],
            vec![-1, 2, -1, -1],
            vec![0, -1, 2, 0],
            vec![0, -1, 0, 2],
        ];
        assert_eq!(ClusterQuiver::type_d(4).unwrap().cartan_companion(), gap_d4);

        // GAP 4: CartanMatrix(RootSystem(SimpleLieAlgebra("E",6,Rationals))), Bourbaki
        // labelling. Sage's E_6 vertices 0,1,2,3,4,5 are Bourbaki nodes 1,3,4,5,6,2.
        let gap_e6 = vec![
            vec![2, 0, -1, 0, 0, 0],
            vec![0, 2, 0, -1, 0, 0],
            vec![-1, 0, 2, -1, 0, 0],
            vec![0, -1, -1, 2, -1, 0],
            vec![0, 0, 0, -1, 2, -1],
            vec![0, 0, 0, 0, -1, 2],
        ];
        let to_bourbaki = [0usize, 2, 3, 4, 5, 1];
        let e6 = ClusterQuiver::type_e(6).unwrap().cartan_companion();
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(e6[i][j], gap_e6[to_bourbaki[i]][to_bourbaki[j]], "entry ({}, {})", i, j);
            }
        }

        // Determinants of the Cartan matrices (GAP): A_n: n+1, D_n: 4, E_6/7/8: 3/2/1
        assert_eq!(determinant(&ClusterQuiver::type_a(5).cartan_companion()), Some(6));
        assert_eq!(determinant(&ClusterQuiver::type_d(5).unwrap().cartan_companion()), Some(4));
        for (n, det) in [(6usize, 3i128), (7, 2), (8, 1)] {
            let a = ClusterQuiver::type_e(n).unwrap().cartan_companion();
            assert_eq!(determinant(&a), Some(det), "E_{}", n);
        }
    }

    #[test]
    fn test_standard_quivers_match_sage() {
        // Sage: ClusterQuiver(['D',4]).b_matrix()
        let d4 = ClusterQuiver::type_d(4).unwrap();
        assert_eq!(
            d4.exchange_matrix(),
            &vec![
                vec![0, 1, 0, 0],
                vec![-1, 0, -1, -1],
                vec![0, 1, 0, 0],
                vec![0, 1, 0, 0],
            ]
        );

        // Sage's ['D',5] digraph: graph 0-1-2-3, 2-4 oriented with sources {0, 2}
        let d5 = ClusterQuiver::type_d(5).unwrap();
        let mut expected = vec![vec![0i64; 5]; 5];
        for &(s, t) in &[(0, 1), (2, 1), (2, 3), (2, 4)] {
            expected[s][t] = 1;
            expected[t][s] = -1;
        }
        assert_eq!(d5.exchange_matrix(), &expected);

        // Sage's ['E',6] digraph: graph 0-1-2-3-4, 2-5 oriented with sources {0, 2, 4}
        let e6 = ClusterQuiver::type_e(6).unwrap();
        let mut expected = vec![vec![0i64; 6]; 6];
        for &(s, t) in &[(0, 1), (2, 1), (2, 3), (4, 3), (2, 5)] {
            expected[s][t] = 1;
            expected[t][s] = -1;
        }
        assert_eq!(e6.exchange_matrix(), &expected);
    }

    #[test]
    fn test_to_quiver() {
        let cq = ClusterQuiver::type_a(3);
        let q = cq.to_quiver().unwrap();

        assert_eq!(q.num_vertices(), 3);
        assert!(q.num_edges() >= 2); // At least two arrows
    }

    #[test]
    fn test_to_quiver_orientation_and_multiplicity() {
        // A_3: 0 -> 1 -> 2 (b_ij > 0 means arrows i -> j)
        let q = ClusterQuiver::type_a(3).to_quiver().unwrap();
        assert_eq!(q.num_edges(), 2);
        let a01 = q.get_edge("a_0_1").unwrap();
        assert_eq!((a01.source, a01.target), (0, 1));
        let a12 = q.get_edge("a_1_2").unwrap();
        assert_eq!((a12.source, a12.target), (1, 2));

        // Kronecker quiver: two arrows 0 -> 1
        let k = ClusterQuiver::new(vec![vec![0, 2], vec![-2, 0]]).unwrap();
        let q = k.to_quiver().unwrap();
        assert_eq!(q.num_edges(), 2);
        assert!(q.edges().iter().all(|e| e.source == 0 && e.target == 1));

        // Labels stay distinct for two-digit vertices (vertices 1 -> 12 and 11 -> 2)
        let mut m = vec![vec![0i64; 13]; 13];
        m[1][12] = 1;
        m[12][1] = -1;
        m[11][2] = 1;
        m[2][11] = -1;
        assert_eq!(ClusterQuiver::new(m).unwrap().to_quiver().unwrap().num_edges(), 2);

        // Skew-symmetrizable but not skew-symmetric: no plain quiver
        let b2 = ClusterQuiver::new(vec![vec![0, 1], vec![-2, 0]]).unwrap();
        assert!(b2.to_quiver().is_err());
    }

    #[test]
    fn test_new_rejects_non_square_principal_part() {
        assert!(ClusterQuiver::new(vec![vec![0, 1, 0]]).is_err());
        assert!(ClusterQuiver::new(vec![vec![0, 1], vec![-1, 0], vec![1, 1]]).is_ok());
    }

    #[test]
    fn test_mutation_type_finite() {
        // A_2 is finite type
        let q = ClusterQuiver::type_a(2);
        let mt = q.mutation_type(1000);
        assert_eq!(mt, Ok(MutationType::Finite));
    }

    fn quiver(rows: Vec<Vec<i64>>) -> ClusterQuiver {
        ClusterQuiver::new(rows).unwrap()
    }

    #[test]
    fn test_mutation_type_dynkin_finite() {
        for n in 1..=6 {
            assert_eq!(ClusterQuiver::type_a(n).mutation_type(1000), Ok(MutationType::Finite));
        }
        for n in 4..=7 {
            assert_eq!(
                ClusterQuiver::type_d(n).unwrap().mutation_type(1000),
                Ok(MutationType::Finite)
            );
        }
        for n in 6..=8 {
            assert_eq!(
                ClusterQuiver::type_e(n).unwrap().mutation_type(1000),
                Ok(MutationType::Finite)
            );
        }
        // Oriented 3-cycle: in the A_3 mutation class (Sage: mutation_type ['A', 3]),
        // although its own Cartan companion is affine
        let cycle = quiver(vec![vec![0, 1, -1], vec![-1, 0, 1], vec![1, -1, 0]]);
        assert_eq!(cycle.mutation_type(1000), Ok(MutationType::Finite));
        // ... and a bound too small to reach an acyclic member decides nothing
        assert!(cycle.mutation_type(1).is_err());
    }

    #[test]
    fn test_mutation_type_affine_and_wild() {
        // Kronecker quiver (Sage: ['A', [1, 1], 1])
        assert_eq!(
            quiver(vec![vec![0, 2], vec![-2, 0]]).mutation_type(1000),
            Ok(MutationType::Affine)
        );
        // Rank 2 with |b_12 b_21| = 9: wild
        assert_eq!(
            quiver(vec![vec![0, 3], vec![-3, 0]]).mutation_type(1000),
            Ok(MutationType::Infinite)
        );
        // Acyclic triangle 0 -> 1 -> 2, 0 -> 2 (Sage: ['A', [1, 2], 1])
        assert_eq!(
            quiver(vec![vec![0, 1, 1], vec![-1, 0, 1], vec![-1, -1, 0]]).mutation_type(1000),
            Ok(MutationType::Affine)
        );
        // D~_4: four leaves pointing into a centre (Sage: ['D', 4, 1])
        let mut d4_affine = vec![vec![0i64; 5]; 5];
        for leaf in 1..5 {
            d4_affine[leaf][0] = 1;
            d4_affine[0][leaf] = -1;
        }
        assert_eq!(quiver(d4_affine).mutation_type(1000), Ok(MutationType::Affine));
        // Star with five leaves: acyclic of indefinite (wild) type
        let mut star = vec![vec![0i64; 6]; 6];
        for leaf in 1..6 {
            star[leaf][0] = 1;
            star[0][leaf] = -1;
        }
        assert_eq!(quiver(star).mutation_type(1000), Ok(MutationType::Infinite));
        // Markov quiver: mutation class {B, -B}, no acyclic member, not 2-finite
        assert_eq!(
            quiver(vec![vec![0, 2, -2], vec![-2, 0, 2], vec![2, -2, 0]]).mutation_type(1000),
            Ok(MutationType::Infinite)
        );
        // Rank 3 with a triple arrow: mutation-infinite
        assert_eq!(
            quiver(vec![vec![0, 3, 0], vec![-3, 0, 1], vec![0, -1, 0]]).mutation_type(1000),
            Ok(MutationType::Infinite)
        );
    }

    #[test]
    fn test_mutation_type_skew_symmetrizable() {
        // B_2 (|b_12 b_21| = 2) and G_2 (= 3): finite
        assert_eq!(
            quiver(vec![vec![0, 1], vec![-2, 0]]).mutation_type(1000),
            Ok(MutationType::Finite)
        );
        assert_eq!(
            quiver(vec![vec![0, 1], vec![-3, 0]]).mutation_type(1000),
            Ok(MutationType::Finite)
        );
        // |b_12 b_21| = 4: affine A_2^(2); = 5: wild
        assert_eq!(
            quiver(vec![vec![0, 1], vec![-4, 0]]).mutation_type(1000),
            Ok(MutationType::Affine)
        );
        assert_eq!(
            quiver(vec![vec![0, 1], vec![-5, 0]]).mutation_type(1000),
            Ok(MutationType::Infinite)
        );
        // B_3: companion [[2,-1,0],[-1,2,-1],[0,-2,2]], det 2
        assert_eq!(
            quiver(vec![vec![0, 1, 0], vec![-1, 0, 1], vec![0, -2, 0]]).mutation_type(1000),
            Ok(MutationType::Finite)
        );
        // C~_2 = C_2^(1) (Kac, Table Aff 1): companion [[2,-1,0],[-2,2,-2],[0,-1,2]], det 0
        assert_eq!(
            quiver(vec![vec![0, 1, 0], vec![-2, 0, 2], vec![0, -1, 0]]).mutation_type(1000),
            Ok(MutationType::Affine)
        );
        // Acyclic, indefinite companion, infinite mutation class: the bounded search
        // must not claim a type
        assert!(quiver(vec![vec![0, 1, 0], vec![-3, 0, 1], vec![0, -3, 0]])
            .mutation_type(500)
            .is_err());
        // Not skew-symmetrizable (d_0 = d_1 = d_2 but d_0 = 2 d_2)
        assert!(quiver(vec![vec![0, 1, 1], vec![-1, 0, 1], vec![-2, -1, 0]])
            .mutation_type(1000)
            .is_err());
        // Not even sign-skew-symmetric
        assert!(quiver(vec![vec![0, 1], vec![1, 0]]).mutation_type(1000).is_err());
    }

    #[test]
    fn test_mutation_type_components_and_frozen() {
        // A_1 + Kronecker
        assert_eq!(
            quiver(vec![vec![0, 0, 0], vec![0, 0, 2], vec![0, -2, 0]]).mutation_type(1000),
            Ok(MutationType::Affine)
        );
        // A_2 + Markov
        let mut m = vec![vec![0i64; 5]; 5];
        m[0][1] = 1;
        m[1][0] = -1;
        for &(i, j) in &[(2usize, 3usize), (3, 4), (4, 2)] {
            m[i][j] = 2;
            m[j][i] = -2;
        }
        assert_eq!(quiver(m).mutation_type(1000), Ok(MutationType::Infinite));

        // Freezing vertex 2 of the affine triangle leaves the A_2 quiver 0 -> 1
        let mut t = quiver(vec![vec![0, 1, 1], vec![-1, 0, 1], vec![-1, -1, 0]]);
        t.freeze_vertex(2);
        assert_eq!(t.mutation_type(1000), Ok(MutationType::Finite));

        // Coefficient rows do not change the type
        let mut k = quiver(vec![vec![0, 2], vec![-2, 0]]);
        k.add_coefficients(1);
        k.set_entry(2, 0, 1);
        assert_eq!(k.mutation_type(1000), Ok(MutationType::Affine));
    }

    #[test]
    fn test_mutation_type_class_size_boundary() {
        // The starting matrix is always stored and examined, even with max_class_size = 0:
        // an acyclic start of finite or affine type decides without any other matrix.
        assert_eq!(ClusterQuiver::type_a(2).mutation_type(0), Ok(MutationType::Finite));
        assert_eq!(
            quiver(vec![vec![0, 2], vec![-2, 0]]).mutation_type(0),
            Ok(MutationType::Affine)
        );

        // Oriented 3-cycle (cyclic start). mu_0 of it is the acyclic A_3 quiver
        // 1 -> 0 -> 2. Bounds 0 and 1 both store only the start and fail the same way;
        // bound 2 also stores mu_0, which decides.
        let cycle = quiver(vec![vec![0, 1, -1], vec![-1, 0, 1], vec![1, -1, 0]]);
        let undecided =
            Err("Mutation type undecided: the mutation class has more than 1 matrices".to_string());
        assert_eq!(cycle.mutation_type(0), undecided);
        assert_eq!(cycle.mutation_type(1), undecided);
        assert_eq!(cycle.mutation_type(2), Ok(MutationType::Finite));

        // Markov quiver: labelled class {B, -B} (every mu_k gives -B). It fits into a
        // bound of 2 but not of 1 (or 0), and only the complete class decides Infinite.
        let markov = quiver(vec![vec![0, 2, -2], vec![-2, 0, 2], vec![2, -2, 0]]);
        assert_eq!(markov.mutation_type(0), undecided);
        assert_eq!(markov.mutation_type(1), undecided);
        assert_eq!(markov.mutation_type(2), Ok(MutationType::Infinite));
    }

    #[test]
    fn test_cartan_companion_keeps_freeze_vertex_vertices() {
        // Acyclic triangle 0 -> 1 -> 2, 0 -> 2: its companion is the affine A~_2 matrix
        let mut t = quiver(vec![vec![0, 1, 1], vec![-1, 0, 1], vec![-1, -1, 0]]);
        let a_tilde_2 = vec![vec![2, -1, -1], vec![-1, 2, -1], vec![-1, -1, 2]];
        assert_eq!(t.cartan_companion(), a_tilde_2);
        assert_eq!(t.mutation_type(1000), Ok(MutationType::Affine));

        // freeze_vertex(2) does not change cartan_companion: it stays the 3 x 3 companion
        // of principal_part(), indexed by vertex, with det 0 ...
        t.freeze_vertex(2);
        assert_eq!(t.principal_part().len(), 3);
        assert_eq!(t.cartan_companion(), a_tilde_2);
        assert_eq!(determinant(&t.cartan_companion()), Some(0));

        // ... while mutation_type drops vertex 2. The companion it classifies is the
        // submatrix of cartan_companion on {0, 1}, the A_2 Cartan matrix, so it is Finite.
        let a2 = vec![vec![2, -1], vec![-1, 2]];
        assert_eq!(submatrix(&t.cartan_companion(), &[0, 1]), a2);
        assert_eq!(cartan_companion_of(&submatrix(&t.principal_part(), &[0, 1])), a2);
        assert_eq!(t.mutation_type(1000), Ok(MutationType::Finite));
    }

    #[test]
    fn test_is_mutation_equivalent() {
        let a3 = ClusterQuiver::type_a(3);
        let mut target = a3.clone();
        target.mutate_sequence(&MutationSequence::from_vec(vec![1, 0, 2])).unwrap();
        assert_eq!(a3.is_mutation_equivalent(&target, 3), Some(true));
        assert_eq!(a3.is_mutation_equivalent(&a3, 0), Some(true));

        // A_2 class is {B, -B}: enumerated completely, so a negative answer is a proof
        let a2 = ClusterQuiver::type_a(2);
        let kronecker = quiver(vec![vec![0, 2], vec![-2, 0]]);
        assert_eq!(a2.is_mutation_equivalent(&kronecker, 10), Some(false));

        // Infinite mutation class: a bounded search that does not meet the target is
        // undecided
        let wild = quiver(vec![vec![0, 3, 0], vec![-3, 0, 1], vec![0, -1, 0]]);
        let other = quiver(vec![vec![0, 1, 0], vec![-1, 0, 1], vec![0, -1, 0]]);
        assert_eq!(wild.is_mutation_equivalent(&other, 4), None);

        // Different shapes are never equivalent
        assert_eq!(a2.is_mutation_equivalent(&a3, 5), Some(false));
    }

    #[test]
    fn test_is_mutation_equivalent_frozen_sets() {
        // mu_1(A_3) is the oriented 3-cycle 0 -> 2 -> 1 -> 0
        let a3 = ClusterQuiver::type_a(3);
        let h = a3.mutated(1).unwrap();
        assert_eq!(h.exchange_matrix(), &vec![vec![0, -1, 1], vec![1, 0, -1], vec![-1, 1, 0]]);
        assert_eq!(a3.is_mutation_equivalent(&h, 1), Some(true));
        assert_eq!(h.is_mutation_equivalent(&a3, 1), Some(true));

        // The same matrix with vertex 1 frozen is a different object, in either argument
        // order. (Searching with self's frozen set alone gave Some(true) for a3 -> h_frozen
        // but Some(false) for h_frozen -> a3.)
        let mut h_frozen = h.clone();
        h_frozen.freeze_vertex(1);
        assert_eq!(a3.is_mutation_equivalent(&h_frozen, 10), Some(false));
        assert_eq!(h_frozen.is_mutation_equivalent(&a3, 10), Some(false));

        // Equal frozen sets {1}: mutations at 0 and 2 only. mu_0 and mu_2 commute on A_3,
        // so the class of a3_frozen is {B, mu_0 B, mu_2 B, mu_2 mu_0 B}.
        let mut a3_frozen = a3.clone();
        a3_frozen.freeze_vertex(1);
        let mut g = a3_frozen.clone();
        g.mutate_sequence(&MutationSequence::from_vec(vec![0, 2])).unwrap();
        assert_eq!(g.exchange_matrix(), &vec![vec![0, -1, 0], vec![1, 0, -1], vec![0, 1, 0]]);
        assert_eq!(a3_frozen.is_mutation_equivalent(&g, 2), Some(true));
        assert_eq!(g.is_mutation_equivalent(&a3_frozen, 2), Some(true));
        // The oriented 3-cycle needs a mutation at the frozen vertex 1. The class of
        // a3_frozen (4 matrices) and that of h_frozen (10 matrices, checked by an
        // independent script) are enumerated in full, so both answers are proofs.
        assert_eq!(a3_frozen.is_mutation_equivalent(&h_frozen, 10), Some(false));
        assert_eq!(h_frozen.is_mutation_equivalent(&a3_frozen, 10), Some(false));
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

//! Cluster Complexes and Generalized Associahedra
//!
//! This module implements cluster complexes associated with root systems and cluster algebras,
//! including generalized associahedra which are polytopal realizations of cluster complexes.
//!
//! A cluster complex is a simplicial complex whose vertices are cluster variables and whose
//! facets are clusters (maximal compatible sets of cluster variables). For finite-type cluster
//! algebras, cluster complexes correspond to root systems.
//!
//! Generalized associahedra are a family of polytopes associated with finite root systems.
//! The type A_n associahedron is the classical Stasheff associahedron, which has vertices
//! corresponding to triangulations of a convex (n+3)-gon.
//!
//! References:
//! - Fomin & Zelevinsky: "Cluster algebras I: Foundations"
//! - Fomin & Reading: "Root systems and generalized associahedra"
//! - Chapoton, Fomin, Zelevinsky: "Polytopal realizations of generalized associahedra"
//!
//! Corresponds to sage.combinat.cluster_complex

use rustmath_integers::Integer;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// An almost-positive root (root or negative simple root)
///
/// In cluster complex theory, vertices correspond to almost-positive roots:
/// positive roots or negatives of simple roots.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlmostPositiveRoot {
    /// The root vector represented as integer coefficients in the simple root basis
    /// For root = sum_i c_i * alpha_i, this stores [c_1, c_2, ..., c_n]
    pub coefficients: Vec<i32>,
}

impl AlmostPositiveRoot {
    /// Create a new almost-positive root from simple root coefficients
    pub fn new(coefficients: Vec<i32>) -> Self {
        AlmostPositiveRoot { coefficients }
    }

    /// Check if this is a positive root (all coefficients >= 0)
    pub fn is_positive(&self) -> bool {
        self.coefficients.iter().all(|&c| c >= 0)
    }

    /// Check if this is a simple root (exactly one coefficient is 1, rest are 0)
    pub fn is_simple(&self) -> bool {
        self.coefficients.iter().filter(|&&c| c == 1).count() == 1
            && self.coefficients.iter().filter(|&&c| c == 0).count() == self.coefficients.len() - 1
    }

    /// Check if this is a negative simple root (exactly one coefficient is -1, rest are 0)
    pub fn is_negative_simple(&self) -> bool {
        self.coefficients.iter().filter(|&&c| c == -1).count() == 1
            && self.coefficients.iter().filter(|&&c| c == 0).count() == self.coefficients.len() - 1
    }

    /// Get the index if this is a simple or negative simple root
    pub fn simple_root_index(&self) -> Option<usize> {
        if self.is_simple() || self.is_negative_simple() {
            self.coefficients.iter().position(|&c| c != 0)
        } else {
            None
        }
    }

    /// Compute the height of this root (sum of coefficients)
    pub fn height(&self) -> i32 {
        self.coefficients.iter().sum()
    }
}

impl fmt::Display for AlmostPositiveRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coeff)?;
        }
        write!(f, ")")
    }
}

/// A cluster: a maximal compatible set of almost-positive roots
///
/// In type A_n, a cluster corresponds to a triangulation of a convex (n+3)-gon.
/// Clusters have size equal to the rank of the root system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    /// The roots in this cluster
    pub roots: Vec<AlmostPositiveRoot>,
    /// The rank (dimension) of the associated root system
    pub rank: usize,
}

impl Cluster {
    /// Create a new cluster
    pub fn new(roots: Vec<AlmostPositiveRoot>, rank: usize) -> Option<Self> {
        // A cluster must have exactly rank elements
        if roots.len() != rank {
            return None;
        }

        Some(Cluster { roots, rank })
    }

    /// Get the cluster variables (roots) in this cluster
    pub fn variables(&self) -> &[AlmostPositiveRoot] {
        &self.roots
    }

    /// Check if this cluster contains a given root
    pub fn contains(&self, root: &AlmostPositiveRoot) -> bool {
        self.roots.contains(root)
    }

    /// Compute the intersection with another cluster
    pub fn intersection(&self, other: &Cluster) -> Vec<AlmostPositiveRoot> {
        self.roots
            .iter()
            .filter(|r| other.contains(r))
            .cloned()
            .collect()
    }

    /// Check if two clusters are compatible (differ by exactly one element)
    ///
    /// Two clusters are compatible if they can be connected by a mutation,
    /// i.e., they differ in exactly one root.
    pub fn is_compatible(&self, other: &Cluster) -> bool {
        let intersection_size = self.intersection(other).len();
        intersection_size == self.rank - 1
    }
}

/// Cartan type for cluster algebras
///
/// This mirrors the Cartan types used in Lie theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClusterCartanType {
    /// Type A_n (n >= 1)
    A(usize),
    /// Type B_n (n >= 2)
    B(usize),
    /// Type C_n (n >= 2)
    C(usize),
    /// Type D_n (n >= 4)
    D(usize),
    /// Type E_6
    E6,
    /// Type E_7
    E7,
    /// Type E_8
    E8,
    /// Type F_4
    F4,
    /// Type G_2
    G2,
}

impl ClusterCartanType {
    /// Get the rank of this Cartan type
    pub fn rank(&self) -> usize {
        match self {
            ClusterCartanType::A(n) => *n,
            ClusterCartanType::B(n) => *n,
            ClusterCartanType::C(n) => *n,
            ClusterCartanType::D(n) => *n,
            ClusterCartanType::E6 => 6,
            ClusterCartanType::E7 => 7,
            ClusterCartanType::E8 => 8,
            ClusterCartanType::F4 => 4,
            ClusterCartanType::G2 => 2,
        }
    }

    /// Validate that the rank is appropriate for this type
    pub fn validate(&self) -> bool {
        match self {
            ClusterCartanType::A(n) => *n >= 1,
            ClusterCartanType::B(n) => *n >= 2,
            ClusterCartanType::C(n) => *n >= 2,
            ClusterCartanType::D(n) => *n >= 4,
            _ => true, // Exceptional types have fixed ranks
        }
    }
}

impl fmt::Display for ClusterCartanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterCartanType::A(n) => write!(f, "A_{}", n),
            ClusterCartanType::B(n) => write!(f, "B_{}", n),
            ClusterCartanType::C(n) => write!(f, "C_{}", n),
            ClusterCartanType::D(n) => write!(f, "D_{}", n),
            ClusterCartanType::E6 => write!(f, "E_6"),
            ClusterCartanType::E7 => write!(f, "E_7"),
            ClusterCartanType::E8 => write!(f, "E_8"),
            ClusterCartanType::F4 => write!(f, "F_4"),
            ClusterCartanType::G2 => write!(f, "G_2"),
        }
    }
}

/// A cluster complex: the simplicial complex of compatible clusters
///
/// The cluster complex is a pure simplicial complex of dimension (rank - 1),
/// where vertices are almost-positive roots and maximal simplices are clusters.
#[derive(Debug, Clone)]
pub struct ClusterComplex {
    /// The Cartan type
    pub cartan_type: ClusterCartanType,
    /// The rank of the root system
    pub rank: usize,
    /// All almost-positive roots (vertices of the complex)
    pub vertices: Vec<AlmostPositiveRoot>,
    /// All clusters (maximal simplices / facets)
    pub clusters: Vec<Cluster>,
    /// The Cartan matrix
    pub cartan_matrix: Vec<Vec<i32>>,
}

impl ClusterComplex {
    /// Create a new cluster complex for a given Cartan type
    pub fn new(cartan_type: ClusterCartanType) -> Option<Self> {
        if !cartan_type.validate() {
            return None;
        }

        let rank = cartan_type.rank();
        let cartan_matrix = Self::construct_cartan_matrix(&cartan_type);
        let vertices = Self::generate_almost_positive_roots(&cartan_type, &cartan_matrix);
        let clusters = Self::generate_clusters(&cartan_type, &vertices, rank);

        Some(ClusterComplex {
            cartan_type,
            rank,
            vertices,
            clusters,
            cartan_matrix,
        })
    }

    /// Construct the Cartan matrix for a given type
    fn construct_cartan_matrix(cartan_type: &ClusterCartanType) -> Vec<Vec<i32>> {
        match cartan_type {
            ClusterCartanType::A(n) => Self::cartan_matrix_type_a(*n),
            ClusterCartanType::B(n) => Self::cartan_matrix_type_b(*n),
            ClusterCartanType::C(n) => Self::cartan_matrix_type_c(*n),
            ClusterCartanType::D(n) => Self::cartan_matrix_type_d(*n),
            ClusterCartanType::E6 => Self::cartan_matrix_type_e(6),
            ClusterCartanType::E7 => Self::cartan_matrix_type_e(7),
            ClusterCartanType::E8 => Self::cartan_matrix_type_e(8),
            ClusterCartanType::F4 => Self::cartan_matrix_type_f4(),
            ClusterCartanType::G2 => Self::cartan_matrix_type_g2(),
        }
    }

    /// Cartan matrix for type A_n
    fn cartan_matrix_type_a(n: usize) -> Vec<Vec<i32>> {
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

    /// Cartan matrix for type B_n
    fn cartan_matrix_type_b(n: usize) -> Vec<Vec<i32>> {
        let mut matrix = vec![vec![0; n]; n];
        for i in 0..n {
            matrix[i][i] = 2;
            if i > 0 {
                matrix[i][i - 1] = -1;
            }
            if i < n - 1 {
                if i == n - 2 {
                    matrix[i][i + 1] = -2; // Long root to short root
                } else {
                    matrix[i][i + 1] = -1;
                }
            }
        }
        matrix
    }

    /// Cartan matrix for type C_n
    fn cartan_matrix_type_c(n: usize) -> Vec<Vec<i32>> {
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

    /// Cartan matrix for type D_n
    fn cartan_matrix_type_d(n: usize) -> Vec<Vec<i32>> {
        let mut matrix = vec![vec![0; n]; n];
        for i in 0..n {
            matrix[i][i] = 2;
            if i > 0 && i < n - 1 {
                matrix[i][i - 1] = -1;
                matrix[i][i + 1] = -1;
            } else if i == n - 1 {
                // Last two roots branch from the (n-2)-th root
                matrix[i][n - 3] = -1;
            } else if i == n - 2 {
                matrix[i][n - 3] = -1;
            }
        }
        matrix
    }

    /// Cartan matrix for type E_n (n = 6, 7, 8)
    fn cartan_matrix_type_e(n: usize) -> Vec<Vec<i32>> {
        // Placeholder for E type - these require specific constructions
        let mut matrix = vec![vec![0; n]; n];
        for i in 0..n {
            matrix[i][i] = 2;
        }
        // TODO: Implement proper E_6, E_7, E_8 Cartan matrices
        matrix
    }

    /// Cartan matrix for type F_4
    fn cartan_matrix_type_f4() -> Vec<Vec<i32>> {
        vec![
            vec![2, -1, 0, 0],
            vec![-1, 2, -2, 0],
            vec![0, -1, 2, -1],
            vec![0, 0, -1, 2],
        ]
    }

    /// Cartan matrix for type G_2
    fn cartan_matrix_type_g2() -> Vec<Vec<i32>> {
        vec![vec![2, -3], vec![-1, 2]]
    }

    /// Generate all almost-positive roots for a given Cartan type
    fn generate_almost_positive_roots(
        cartan_type: &ClusterCartanType,
        _cartan_matrix: &[Vec<i32>],
    ) -> Vec<AlmostPositiveRoot> {
        let rank = cartan_type.rank();
        let mut roots = Vec::new();

        // Add negative simple roots
        for i in 0..rank {
            let mut coeffs = vec![0; rank];
            coeffs[i] = -1;
            roots.push(AlmostPositiveRoot::new(coeffs));
        }

        // Add positive roots (for type A_n, these are well-known)
        // For a general implementation, we'd use the root system module
        match cartan_type {
            ClusterCartanType::A(n) => {
                // For type A_n, positive roots are α_{i,j} = α_i + α_{i+1} + ... + α_j
                // for 0 <= i <= j < n
                for i in 0..*n {
                    for j in i..*n {
                        let mut coeffs = vec![0; rank];
                        for k in i..=j {
                            coeffs[k] = 1;
                        }
                        roots.push(AlmostPositiveRoot::new(coeffs));
                    }
                }
            }
            _ => {
                // For other types, add simple roots for now
                // TODO: Implement full positive root generation for all types
                for i in 0..rank {
                    let mut coeffs = vec![0; rank];
                    coeffs[i] = 1;
                    roots.push(AlmostPositiveRoot::new(coeffs));
                }
            }
        }

        roots
    }

    /// Generate all clusters for a given Cartan type
    fn generate_clusters(
        cartan_type: &ClusterCartanType,
        vertices: &[AlmostPositiveRoot],
        rank: usize,
    ) -> Vec<Cluster> {
        let mut clusters = Vec::new();

        match cartan_type {
            ClusterCartanType::A(n) => {
                // For type A_n, clusters correspond to triangulations of an (n+3)-gon
                // The initial cluster consists of n negative simple roots
                let initial: Vec<_> = vertices
                    .iter()
                    .filter(|r| r.is_negative_simple())
                    .take(rank)
                    .cloned()
                    .collect();

                if let Some(cluster) = Cluster::new(initial, rank) {
                    clusters.push(cluster);
                }

                // Generate more clusters via mutations
                // For type A_2, we have 5 clusters corresponding to 5 triangulations
                // TODO: Implement full cluster generation via mutation
            }
            _ => {
                // For other types, start with the initial cluster
                let initial: Vec<_> = vertices
                    .iter()
                    .filter(|r| r.is_negative_simple())
                    .take(rank)
                    .cloned()
                    .collect();

                if let Some(cluster) = Cluster::new(initial, rank) {
                    clusters.push(cluster);
                }
            }
        }

        clusters
    }

    /// Get the number of clusters (facets)
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get all clusters
    pub fn get_clusters(&self) -> &[Cluster] {
        &self.clusters
    }

    /// Get all vertices
    pub fn get_vertices(&self) -> &[AlmostPositiveRoot] {
        &self.vertices
    }

    /// Compute the f-vector of the cluster complex
    ///
    /// The f-vector (f_0, f_1, ..., f_d) counts the number of faces of each dimension
    pub fn f_vector(&self) -> Vec<usize> {
        // For now, just return basic statistics
        // TODO: Implement full face enumeration
        vec![self.num_vertices(), 0, self.num_clusters()]
    }
}

/// A generalized associahedron
///
/// The generalized associahedron is a convex polytope whose vertices correspond to clusters
/// and whose edges connect clusters that differ by one root (mutation).
///
/// For type A_n, this is the classical Stasheff associahedron.
#[derive(Debug, Clone)]
pub struct GeneralizedAssociahedron {
    /// The underlying cluster complex
    pub cluster_complex: ClusterComplex,
    /// Vertices of the polytope (labeled by clusters)
    pub vertices: Vec<usize>, // Indices into cluster_complex.clusters
    /// Edges between vertices
    pub edges: Vec<(usize, usize)>,
}

impl GeneralizedAssociahedron {
    /// Create a generalized associahedron for a given Cartan type
    pub fn new(cartan_type: ClusterCartanType) -> Option<Self> {
        let cluster_complex = ClusterComplex::new(cartan_type)?;

        // Vertices are clusters
        let vertices: Vec<usize> = (0..cluster_complex.num_clusters()).collect();

        // Edges connect compatible clusters
        let mut edges = Vec::new();
        for i in 0..cluster_complex.num_clusters() {
            for j in (i + 1)..cluster_complex.num_clusters() {
                if cluster_complex.clusters[i].is_compatible(&cluster_complex.clusters[j]) {
                    edges.push((i, j));
                }
            }
        }

        Some(GeneralizedAssociahedron {
            cluster_complex,
            vertices,
            edges,
        })
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the dimension of the polytope
    pub fn dimension(&self) -> usize {
        self.cluster_complex.rank
    }

    /// Check if this is the classical associahedron (type A)
    pub fn is_classical_associahedron(&self) -> bool {
        matches!(self.cluster_complex.cartan_type, ClusterCartanType::A(_))
    }
}

/// Helper function to create a type A cluster complex
pub fn cluster_complex_type_a(n: usize) -> Option<ClusterComplex> {
    ClusterComplex::new(ClusterCartanType::A(n))
}

/// Helper function to create a generalized associahedron of type A
pub fn associahedron(n: usize) -> Option<GeneralizedAssociahedron> {
    GeneralizedAssociahedron::new(ClusterCartanType::A(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_almost_positive_root() {
        let root1 = AlmostPositiveRoot::new(vec![1, 0, 0]);
        assert!(root1.is_simple());
        assert!(root1.is_positive());
        assert_eq!(root1.simple_root_index(), Some(0));

        let root2 = AlmostPositiveRoot::new(vec![-1, 0, 0]);
        assert!(root2.is_negative_simple());
        assert!(!root2.is_positive());
        assert_eq!(root2.simple_root_index(), Some(0));

        let root3 = AlmostPositiveRoot::new(vec![1, 1, 0]);
        assert!(!root3.is_simple());
        assert!(root3.is_positive());
        assert_eq!(root3.height(), 2);
    }

    #[test]
    fn test_cartan_type() {
        let type_a3 = ClusterCartanType::A(3);
        assert_eq!(type_a3.rank(), 3);
        assert!(type_a3.validate());

        let type_b1 = ClusterCartanType::B(1);
        assert!(!type_b1.validate()); // B_n requires n >= 2

        let type_e6 = ClusterCartanType::E6;
        assert_eq!(type_e6.rank(), 6);
    }

    #[test]
    fn test_cartan_matrix_type_a() {
        let matrix = ClusterComplex::cartan_matrix_type_a(3);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0], vec![2, -1, 0]);
        assert_eq!(matrix[1], vec![-1, 2, -1]);
        assert_eq!(matrix[2], vec![0, -1, 2]);
    }

    #[test]
    fn test_cluster_complex_type_a() {
        let complex = cluster_complex_type_a(2);
        assert!(complex.is_some());

        let complex = complex.unwrap();
        assert_eq!(complex.rank, 2);
        assert_eq!(complex.cartan_type, ClusterCartanType::A(2));
        assert!(complex.num_vertices() > 0);
    }

    #[test]
    fn test_cluster() {
        let root1 = AlmostPositiveRoot::new(vec![-1, 0]);
        let root2 = AlmostPositiveRoot::new(vec![0, -1]);

        let cluster = Cluster::new(vec![root1.clone(), root2.clone()], 2);
        assert!(cluster.is_some());

        let cluster = cluster.unwrap();
        assert_eq!(cluster.variables().len(), 2);
        assert!(cluster.contains(&root1));
        assert!(cluster.contains(&root2));

        // Wrong size should fail
        let bad_cluster = Cluster::new(vec![root1], 2);
        assert!(bad_cluster.is_none());
    }

    #[test]
    fn test_cluster_compatibility() {
        let root1 = AlmostPositiveRoot::new(vec![-1, 0]);
        let root2 = AlmostPositiveRoot::new(vec![0, -1]);
        let root3 = AlmostPositiveRoot::new(vec![1, 0]);

        let cluster1 = Cluster::new(vec![root1.clone(), root2.clone()], 2).unwrap();
        let cluster2 = Cluster::new(vec![root1.clone(), root3.clone()], 2).unwrap();

        // These clusters differ by one element (root2 vs root3)
        assert!(cluster1.is_compatible(&cluster2));

        let intersection = cluster1.intersection(&cluster2);
        assert_eq!(intersection.len(), 1);
        assert!(intersection.contains(&root1));
    }

    #[test]
    fn test_associahedron() {
        let assoc = associahedron(2);
        assert!(assoc.is_some());

        let assoc = assoc.unwrap();
        assert!(assoc.is_classical_associahedron());
        assert_eq!(assoc.dimension(), 2);
        assert!(assoc.num_vertices() > 0);
    }

    #[test]
    fn test_generalized_associahedron_type_b() {
        let gen_assoc = GeneralizedAssociahedron::new(ClusterCartanType::B(2));
        assert!(gen_assoc.is_some());

        let gen_assoc = gen_assoc.unwrap();
        assert!(!gen_assoc.is_classical_associahedron());
        assert_eq!(gen_assoc.dimension(), 2);
    }
}

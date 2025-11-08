//! Chain complexes and homology computation
//!
//! A chain complex is a sequence of abelian groups (or modules) connected by
//! boundary maps such that the composition of consecutive maps is zero.

use rustmath_groups::AbelianGroup;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use std::collections::HashMap;
use std::fmt;

/// A chain complex of free abelian groups
///
/// Represents: ... → C_{n+1} --d_{n+1}--> C_n --d_n--> C_{n-1} → ...
///
/// Where each C_n is a free abelian group Z^{r_n} and d_n are boundary maps
/// satisfying d_n ∘ d_{n+1} = 0
#[derive(Clone, Debug)]
pub struct ChainComplex {
    /// Dimensions of chain groups at each degree
    /// ranks[n] = rank of C_n
    ranks: HashMap<i32, usize>,

    /// Boundary maps d_n: C_n → C_{n-1}
    /// boundary_maps[n] represents d_n
    boundary_maps: HashMap<i32, Matrix<Integer>>,

    /// Minimum and maximum non-zero degrees
    min_degree: i32,
    max_degree: i32,
}

impl ChainComplex {
    /// Create a new chain complex
    ///
    /// # Arguments
    /// * `ranks` - Dimensions of each chain group
    /// * `boundary_maps` - Boundary maps between consecutive groups
    pub fn new(
        ranks: HashMap<i32, usize>,
        boundary_maps: HashMap<i32, Matrix<Integer>>,
    ) -> Result<Self, String> {
        // Find min and max degrees
        let mut min_degree = i32::MAX;
        let mut max_degree = i32::MIN;

        for &deg in ranks.keys() {
            min_degree = min_degree.min(deg);
            max_degree = max_degree.max(deg);
        }

        if ranks.is_empty() {
            return Err("Chain complex must have at least one chain group".to_string());
        }

        // Validate boundary maps
        for (&deg, map) in &boundary_maps {
            let current_rank = ranks.get(&deg).unwrap_or(&0);
            let prev_rank = ranks.get(&(deg - 1)).unwrap_or(&0);

            if map.rows() != *prev_rank {
                return Err(format!(
                    "Boundary map d_{} has {} rows, expected {}",
                    deg,
                    map.rows(),
                    prev_rank
                ));
            }

            if map.cols() != *current_rank {
                return Err(format!(
                    "Boundary map d_{} has {} columns, expected {}",
                    deg,
                    map.cols(),
                    current_rank
                ));
            }
        }

        // Verify d_n ∘ d_{n+1} = 0 where possible
        for deg in (min_degree + 1)..=max_degree {
            if let (Some(d_n), Some(d_next)) = (
                boundary_maps.get(&deg),
                boundary_maps.get(&(deg + 1)),
            ) {
                // Compute d_n * d_{n+1}
                let product = (d_n.clone() * d_next.clone())
                    .map_err(|e| format!("Failed to compose boundary maps: {:?}", e))?;

                // Check if it's the zero matrix
                for i in 0..product.rows() {
                    for j in 0..product.cols() {
                        if let Ok(elem) = product.get(i, j) {
                            if *elem != Integer::from(0) {
                                return Err(format!(
                                    "Boundary maps don't compose to zero at degree {}",
                                    deg
                                ));
                            }
                        }
                    }
                }
            }
        }

        Ok(ChainComplex {
            ranks,
            boundary_maps,
            min_degree,
            max_degree,
        })
    }

    /// Create a trivial chain complex (all zero groups)
    pub fn trivial() -> Self {
        ChainComplex {
            ranks: HashMap::new(),
            boundary_maps: HashMap::new(),
            min_degree: 0,
            max_degree: 0,
        }
    }

    /// Get the rank of the chain group at degree n
    pub fn rank(&self, n: i32) -> usize {
        *self.ranks.get(&n).unwrap_or(&0)
    }

    /// Get the boundary map d_n: C_n → C_{n-1}
    pub fn boundary_map(&self, n: i32) -> Option<&Matrix<Integer>> {
        self.boundary_maps.get(&n)
    }

    /// Get the minimum non-zero degree
    pub fn min_degree(&self) -> i32 {
        self.min_degree
    }

    /// Get the maximum non-zero degree
    pub fn max_degree(&self) -> i32 {
        self.max_degree
    }

    /// Compute the n-th homology group H_n = ker(d_n) / im(d_{n+1})
    pub fn homology(&self, n: i32) -> HomologyGroup {
        // Get d_n and d_{n+1}
        let rank_n = self.rank(n);

        if rank_n == 0 {
            // If C_n is trivial, H_n is trivial
            return HomologyGroup {
                degree: n,
                free_rank: 0,
                torsion: vec![],
            };
        }

        // Compute kernel of d_n
        let kernel_gens = if let Some(d_n) = self.boundary_map(n) {
            compute_kernel(d_n)
        } else {
            // If d_n doesn't exist, ker(d_n) = C_n (all of it)
            rank_n
        };

        // Compute image of d_{n+1}
        let image_rank = if let Some(d_next) = self.boundary_map(n + 1) {
            compute_rank(d_next)
        } else {
            0
        };

        // H_n has free rank = dim(ker(d_n)) - dim(im(d_{n+1}))
        let free_rank = if kernel_gens >= image_rank {
            kernel_gens - image_rank
        } else {
            0
        };

        HomologyGroup {
            degree: n,
            free_rank,
            torsion: vec![], // Simplified - full torsion computation requires Smith normal form
        }
    }

    /// Compute all homology groups
    pub fn all_homology(&self) -> Vec<HomologyGroup> {
        let mut result = Vec::new();

        for deg in self.min_degree..=self.max_degree {
            result.push(self.homology(deg));
        }

        result
    }

    /// Compute the Euler characteristic χ = Σ (-1)^n rank(C_n)
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0i64;

        for deg in self.min_degree..=self.max_degree {
            let rank = self.rank(deg) as i64;
            if deg % 2 == 0 {
                chi += rank;
            } else {
                chi -= rank;
            }
        }

        chi
    }
}

impl fmt::Display for ChainComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Chain Complex:")?;
        for deg in self.min_degree..=self.max_degree {
            let rank = self.rank(deg);
            write!(f, "  C_{} = Z^{}", deg, rank)?;
            if let Some(d) = self.boundary_map(deg) {
                writeln!(f, " --d_{}--> ({}×{} matrix)", deg, d.rows(), d.cols())?;
            } else {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// A homology group H_n
///
/// Represented as H_n ≅ Z^r ⊕ Z/t₁Z ⊕ ... ⊕ Z/tₖZ
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomologyGroup {
    /// Degree n
    pub degree: i32,
    /// Free rank r (number of Z factors)
    pub free_rank: usize,
    /// Torsion coefficients [t₁, t₂, ..., tₖ]
    pub torsion: Vec<usize>,
}

impl HomologyGroup {
    /// Check if the homology group is trivial (zero)
    pub fn is_trivial(&self) -> bool {
        self.free_rank == 0 && self.torsion.is_empty()
    }

    /// Check if the homology group is free (no torsion)
    pub fn is_free(&self) -> bool {
        self.torsion.is_empty()
    }

    /// Get the rank (number of Z factors)
    pub fn rank(&self) -> usize {
        self.free_rank
    }

    /// Convert to an AbelianGroup
    pub fn to_abelian_group(&self) -> AbelianGroup {
        AbelianGroup::new(self.free_rank, self.torsion.clone())
            .unwrap_or_else(|_| AbelianGroup::free(self.free_rank))
    }

    /// Get a string representation
    pub fn structure_string(&self) -> String {
        if self.is_trivial() {
            return "0".to_string();
        }

        let mut parts = Vec::new();

        if self.free_rank > 0 {
            if self.free_rank == 1 {
                parts.push("Z".to_string());
            } else {
                parts.push(format!("Z^{}", self.free_rank));
            }
        }

        for &t in &self.torsion {
            parts.push(format!("Z/{}", t));
        }

        parts.join(" ⊕ ")
    }
}

impl fmt::Display for HomologyGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H_{} = {}", self.degree, self.structure_string())
    }
}

/// Compute the rank (dimension of column space) of a matrix
fn compute_rank(mat: &Matrix<Integer>) -> usize {
    // Use Gaussian elimination to find rank
    let rows = mat.rows();
    let cols = mat.cols();

    if rows == 0 || cols == 0 {
        return 0;
    }

    // Create a mutable copy for row reduction
    let mut temp = Vec::new();
    for i in 0..rows {
        let mut row = Vec::new();
        for j in 0..cols {
            row.push(mat.get(i, j).unwrap().clone());
        }
        temp.push(row);
    }

    let mut rank = 0;
    let mut col = 0;

    while rank < rows && col < cols {
        // Find pivot
        let mut pivot_row = rank;
        for r in (rank + 1)..rows {
            if temp[r][col].abs() > temp[pivot_row][col].abs() {
                pivot_row = r;
            }
        }

        if temp[pivot_row][col] == Integer::from(0) {
            col += 1;
            continue;
        }

        // Swap rows
        temp.swap(rank, pivot_row);

        // Eliminate below
        for r in (rank + 1)..rows {
            if temp[r][col] != Integer::from(0) {
                let factor = temp[r][col].clone();
                let pivot = temp[rank][col].clone();

                for c in col..cols {
                    let new_val = temp[r][c].clone() * pivot.clone()
                        - temp[rank][c].clone() * factor.clone();
                    temp[r][c] = new_val;
                }
            }
        }

        rank += 1;
        col += 1;
    }

    rank
}

/// Compute the dimension of the kernel (nullspace) of a matrix
fn compute_kernel(mat: &Matrix<Integer>) -> usize {
    let cols = mat.cols();
    let rank = compute_rank(mat);

    // By rank-nullity theorem: dim(ker) = dim(domain) - rank
    if cols >= rank {
        cols - rank
    } else {
        0
    }
}

/// Create a chain complex from a simplicial complex (common in algebraic topology)
///
/// This is a helper function to create chain complexes from combinatorial data
pub fn simplicial_chain_complex(
    vertices: usize,
    edges: Vec<(usize, usize)>,
    triangles: Vec<(usize, usize, usize)>,
) -> ChainComplex {
    let mut ranks = HashMap::new();
    let mut boundary_maps = HashMap::new();

    // C_0 = Z^{# vertices}
    ranks.insert(0, vertices);

    // C_1 = Z^{# edges}
    if !edges.is_empty() {
        ranks.insert(1, edges.len());

        // Boundary map d_1: C_1 → C_0
        // Each edge [v_i, v_j] maps to v_j - v_i
        let mut entries = Vec::new();
        for (i, j) in &edges {
            for v in 0..vertices {
                if v == *j {
                    entries.push(Integer::from(1));
                } else if v == *i {
                    entries.push(Integer::from(-1));
                } else {
                    entries.push(Integer::from(0));
                }
            }
        }

        if let Ok(d1) = Matrix::from_vec(vertices, edges.len(), entries) {
            boundary_maps.insert(1, d1);
        }
    }

    // C_2 = Z^{# triangles}
    if !triangles.is_empty() {
        ranks.insert(2, triangles.len());

        // Boundary map d_2: C_2 → C_1
        // Each triangle [v_i, v_j, v_k] maps to [v_j,v_k] - [v_i,v_k] + [v_i,v_j]
        let mut entries = Vec::new();

        for triangle in &triangles {
            for edge in &edges {
                // Check if edge is a face of triangle
                let coeff = triangle_edge_incidence(*triangle, *edge);
                entries.push(Integer::from(coeff));
            }
        }

        if !edges.is_empty() {
            if let Ok(d2) = Matrix::from_vec(edges.len(), triangles.len(), entries) {
                boundary_maps.insert(2, d2);
            }
        }
    }

    ChainComplex::new(ranks, boundary_maps).unwrap_or_else(|_| ChainComplex::trivial())
}

/// Compute the incidence number of an edge in a triangle
fn triangle_edge_incidence(triangle: (usize, usize, usize), edge: (usize, usize)) -> i32 {
    let (v0, v1, v2) = triangle;
    let (e0, e1) = edge;

    // Order matters: we use [v0, v1, v2] with standard orientation
    // Edges are: [v0,v1], [v1,v2], [v2,v0] (with orientation)

    if (e0 == v0 && e1 == v1) || (e1 == v0 && e0 == v1) {
        if e0 == v0 { 1 } else { -1 }
    } else if (e0 == v1 && e1 == v2) || (e1 == v1 && e0 == v2) {
        if e0 == v1 { 1 } else { -1 }
    } else if (e0 == v2 && e1 == v0) || (e1 == v2 && e0 == v0) {
        if e0 == v2 { 1 } else { -1 }
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_complex() {
        let complex = ChainComplex::trivial();
        assert_eq!(complex.rank(0), 0);
        assert_eq!(complex.rank(1), 0);
    }

    #[test]
    fn test_simple_chain_complex() {
        // Create: 0 → Z^2 --d1--> Z^3 → 0
        // where d1 = [1 0; 0 1; -1 -1]

        let mut ranks = HashMap::new();
        ranks.insert(0, 3);
        ranks.insert(1, 2);

        let mut boundary_maps = HashMap::new();
        let d1 = Matrix::from_vec(
            3,
            2,
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
                Integer::from(-1),
                Integer::from(-1),
            ],
        )
        .unwrap();
        boundary_maps.insert(1, d1);

        let complex = ChainComplex::new(ranks, boundary_maps);
        assert!(complex.is_ok());

        let complex = complex.unwrap();
        assert_eq!(complex.rank(0), 3);
        assert_eq!(complex.rank(1), 2);
    }

    #[test]
    fn test_homology_of_circle() {
        // Circle: 1 vertex, 1 edge forming a loop
        // C_0 = Z, C_1 = Z, d_1 = [0] (edge has same start and end)

        let mut ranks = HashMap::new();
        ranks.insert(0, 1);
        ranks.insert(1, 1);

        let mut boundary_maps = HashMap::new();
        let d1 = Matrix::from_vec(1, 1, vec![Integer::from(0)]).unwrap();
        boundary_maps.insert(1, d1);

        let complex = ChainComplex::new(ranks, boundary_maps).unwrap();

        // H_0 should be Z (connected)
        let h0 = complex.homology(0);
        assert_eq!(h0.free_rank, 1);

        // H_1 should be Z (one loop)
        let h1 = complex.homology(1);
        assert_eq!(h1.free_rank, 1);
    }

    #[test]
    fn test_homology_group_trivial() {
        let h = HomologyGroup {
            degree: 0,
            free_rank: 0,
            torsion: vec![],
        };

        assert!(h.is_trivial());
        assert!(h.is_free());
        assert_eq!(h.structure_string(), "0");
    }

    #[test]
    fn test_homology_group_free() {
        let h = HomologyGroup {
            degree: 1,
            free_rank: 2,
            torsion: vec![],
        };

        assert!(!h.is_trivial());
        assert!(h.is_free());
        assert_eq!(h.rank(), 2);
        assert_eq!(h.structure_string(), "Z^2");
    }

    #[test]
    fn test_homology_group_with_torsion() {
        let h = HomologyGroup {
            degree: 2,
            free_rank: 1,
            torsion: vec![2, 3],
        };

        assert!(!h.is_trivial());
        assert!(!h.is_free());
        assert_eq!(h.structure_string(), "Z ⊕ Z/2 ⊕ Z/3");
    }

    #[test]
    fn test_compute_rank() {
        // Rank 2 matrix
        let mat = Matrix::from_vec(
            3,
            3,
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
                Integer::from(0),
                Integer::from(1),
                Integer::from(1),
                Integer::from(0),
            ],
        )
        .unwrap();

        assert_eq!(compute_rank(&mat), 2);
    }

    #[test]
    fn test_compute_kernel() {
        // Matrix with 1-dimensional kernel
        // [1 0]
        // [0 1]
        // [1 1]
        // Kernel is span of (-1, 1)

        let mat = Matrix::from_vec(
            3,
            2,
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
                Integer::from(1),
                Integer::from(1),
            ],
        )
        .unwrap();

        assert_eq!(compute_kernel(&mat), 0); // Full rank, no kernel

        // Zero matrix has full-dimensional kernel
        let zero_mat = Matrix::from_vec(2, 3, vec![Integer::from(0); 6]).unwrap();
        assert_eq!(compute_kernel(&zero_mat), 3);
    }

    #[test]
    fn test_euler_characteristic() {
        // Point: C_0 = Z
        let mut ranks = HashMap::new();
        ranks.insert(0, 1);
        let boundary_maps = HashMap::new();

        let complex = ChainComplex::new(ranks, boundary_maps).unwrap();
        assert_eq!(complex.euler_characteristic(), 1);
    }

    #[test]
    fn test_simplicial_circle() {
        // Circle with 3 vertices and 3 edges
        let complex = simplicial_chain_complex(3, vec![(0, 1), (1, 2), (2, 0)], vec![]);

        assert_eq!(complex.rank(0), 3);
        assert_eq!(complex.rank(1), 3);

        // Euler characteristic = 3 - 3 = 0
        assert_eq!(complex.euler_characteristic(), 0);
    }

    #[test]
    fn test_simplicial_triangle() {
        // Filled triangle (2-simplex)
        let vertices = 3;
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let triangles = vec![(0, 1, 2)];

        let complex = simplicial_chain_complex(vertices, edges, triangles);

        assert_eq!(complex.rank(0), 3);
        assert_eq!(complex.rank(1), 3);
        assert_eq!(complex.rank(2), 1);

        // Euler characteristic = 3 - 3 + 1 = 1
        assert_eq!(complex.euler_characteristic(), 1);
    }

    #[test]
    fn test_homology_to_abelian_group() {
        let h = HomologyGroup {
            degree: 1,
            free_rank: 2,
            torsion: vec![],
        };

        let g = h.to_abelian_group();
        assert_eq!(g.free_rank(), 2);
        assert_eq!(g.torsion_rank(), 0);
    }

    #[test]
    fn test_display_chain_complex() {
        let mut ranks = HashMap::new();
        ranks.insert(0, 2);
        ranks.insert(1, 1);

        let boundary_maps = HashMap::new();
        let complex = ChainComplex::new(ranks, boundary_maps).unwrap();

        let display = format!("{}", complex);
        assert!(display.contains("Chain Complex"));
        assert!(display.contains("C_0"));
    }

    #[test]
    fn test_boundary_composition_validation() {
        // Create an invalid complex where d_n ∘ d_{n+1} ≠ 0
        let mut ranks = HashMap::new();
        ranks.insert(0, 2);
        ranks.insert(1, 2);
        ranks.insert(2, 1);

        let mut boundary_maps = HashMap::new();

        // d_1: Z^2 → Z^2, identity
        let d1 = Matrix::from_vec(
            2,
            2,
            vec![
                Integer::from(1),
                Integer::from(0),
                Integer::from(0),
                Integer::from(1),
            ],
        )
        .unwrap();
        boundary_maps.insert(1, d1);

        // d_2: Z → Z^2, [1; 1]
        let d2 = Matrix::from_vec(2, 1, vec![Integer::from(1), Integer::from(1)]).unwrap();
        boundary_maps.insert(2, d2);

        // d_1 ∘ d_2 = [1; 1], which is not zero
        // This should fail validation
        let complex = ChainComplex::new(ranks, boundary_maps);
        assert!(complex.is_err());
    }
}

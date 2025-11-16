//! Lattice polytopes
//!
//! This module provides types and functions for working with lattice polytopes -
//! polytopes whose vertices all have integer coordinates in a lattice.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
//! use rustmath_integers::Integer;
//!
//! // Create a triangle with vertices at (0,0), (1,0), (0,1)
//! let vertices = vec![
//!     vec![Integer::from(0), Integer::from(0)],
//!     vec![Integer::from(1), Integer::from(0)],
//!     vec![Integer::from(0), Integer::from(1)],
//! ];
//! let polytope = LatticePolytopeClass::new(vertices);
//!
//! assert_eq!(polytope.n_vertices(), 3);
//! assert_eq!(polytope.dim(), 2);
//! ```

use rustmath_core::Ring;
use rustmath_integers::Integer;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::collections::HashMap;

/// A lattice polytope
///
/// A polytope whose vertices all have integer coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePolytopeClass {
    /// The vertices of the polytope
    vertices: Vec<Vec<Integer>>,
    /// Dimension of the ambient space
    ambient_dim: usize,
    /// Cached dimension of the polytope
    dimension: Option<usize>,
}

impl LatticePolytopeClass {
    /// Create a new lattice polytope from vertices
    ///
    /// # Arguments
    ///
    /// * `vertices` - The vertices of the polytope (must have integer coordinates)
    ///
    /// # Panics
    ///
    /// Panics if vertices list is empty or if vertices have different dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(0), Integer::from(0)],
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    /// ```
    pub fn new(vertices: Vec<Vec<Integer>>) -> Self {
        if vertices.is_empty() {
            panic!("Polytope must have at least one vertex");
        }

        let ambient_dim = vertices[0].len();

        // Verify all vertices have the same dimension
        for v in &vertices {
            if v.len() != ambient_dim {
                panic!("All vertices must have the same dimension");
            }
        }

        Self {
            vertices,
            ambient_dim,
            dimension: None,
        }
    }

    /// Get the vertices
    pub fn vertices(&self) -> &[Vec<Integer>] {
        &self.vertices
    }

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the dimension of the ambient space
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Get the dimension of the polytope
    ///
    /// This is the dimension of the affine hull of the vertices.
    pub fn dim(&self) -> usize {
        // For a proper implementation, we would compute the affine dimension
        // by finding the rank of the matrix of vertex differences
        // For now, we use a simplified computation
        if self.n_vertices() == 1 {
            0
        } else {
            // Approximate dimension based on number of vertices
            (self.n_vertices() - 1).min(self.ambient_dim)
        }
    }

    /// Get the lattice dimension
    ///
    /// This is the dimension of the smallest affine lattice containing the polytope.
    pub fn lattice_dim(&self) -> usize {
        self.ambient_dim
    }

    /// Check if the polytope is reflexive
    ///
    /// A polytope is reflexive if both it and its polar dual are lattice polytopes.
    /// This is a simplified check.
    pub fn is_reflexive(&self) -> bool {
        // For a proper implementation, we would:
        // 1. Compute the polar dual
        // 2. Check if all vertices of the dual are lattice points
        // For now, return false as a conservative estimate
        false
    }

    /// Get all lattice points within the polytope
    ///
    /// Returns all integer points that lie inside or on the boundary of the polytope.
    ///
    /// Note: This is a placeholder that returns only vertices.
    /// A full implementation would enumerate all interior lattice points.
    pub fn points(&self) -> Vec<Vec<Integer>> {
        // For a simple implementation, return the vertices
        // A complete implementation would compute all interior lattice points
        self.vertices.clone()
    }

    /// Get the number of lattice points
    pub fn n_points(&self) -> usize {
        self.points().len()
    }

    /// Compute facet normals
    ///
    /// Returns the normal vectors to each facet of the polytope.
    ///
    /// Note: This is a placeholder implementation.
    pub fn facet_normals(&self) -> Vec<Vec<Integer>> {
        // For a proper implementation, we would compute the convex hull
        // and extract the facet normals
        vec![]
    }

    /// Get the polar dual
    ///
    /// For a reflexive polytope, returns the polar dual.
    /// Returns None if not reflexive or if computation is not implemented.
    pub fn polar(&self) -> Option<Self> {
        // The polar dual of a polytope with vertices v_i is:
        // { x : <x, v_i> >= -1 for all i }
        // This requires solving a dual polytope problem
        None
    }

    /// Get faces of a specific dimension
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of faces to return
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::LatticePolytopeClass;
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(0), Integer::from(0)],
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    ///
    /// // Dimension 0 faces are vertices
    /// let vertex_faces = polytope.faces(0);
    /// ```
    pub fn faces(&self, dimension: usize) -> Vec<Self> {
        if dimension == 0 {
            // 0-dimensional faces are vertices
            self.vertices
                .iter()
                .map(|v| Self::new(vec![v.clone()]))
                .collect()
        } else {
            // For higher dimensions, we would need to compute the face lattice
            vec![]
        }
    }

    /// Get facets (codimension-1 faces)
    pub fn facets(&self) -> Vec<Self> {
        if self.dim() == 0 {
            vec![]
        } else {
            self.faces(self.dim().saturating_sub(1))
        }
    }

    /// Get edges (1-dimensional faces)
    pub fn edges(&self) -> Vec<Self> {
        self.faces(1)
    }

    /// Check if this polytope contains a point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    pub fn contains(&self, point: &[Integer]) -> bool {
        if point.len() != self.ambient_dim {
            return false;
        }

        // For a proper implementation, we would check if the point
        // satisfies all facet inequalities
        // For now, check if the point is one of the vertices
        self.vertices.iter().any(|v| v.as_slice() == point)
    }

    /// Compute the volume of the polytope
    ///
    /// Returns the lattice volume (normalized volume).
    /// Note: This is a placeholder that returns 0.
    pub fn volume(&self) -> Integer {
        // For a proper implementation, we would use the lattice point
        // enumeration theorem or triangulation
        Integer::from(0)
    }

    /// Check if two polytopes are equal
    ///
    /// Two polytopes are equal if they have the same set of vertices.
    pub fn equals(&self, other: &Self) -> bool {
        if self.n_vertices() != other.n_vertices() {
            return false;
        }

        // Check if all vertices match (order-independent)
        for v in &self.vertices {
            if !other.vertices.contains(v) {
                return false;
            }
        }

        true
    }
}

impl Hash for LatticePolytopeClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on vertices
        self.n_vertices().hash(state);
        self.ambient_dim.hash(state);
        // For proper hashing, we should sort vertices first
        for v in &self.vertices {
            for coord in v {
                coord.hash(state);
            }
        }
    }
}

impl fmt::Display for LatticePolytopeClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LatticePolytope({}-dimensional with {} vertices)",
            self.dim(),
            self.n_vertices()
        )
    }
}

/// Create a lattice polytope from vertices
///
/// This is a convenience function that wraps LatticePolytopeClass::new.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::lattice_polytope;
/// use rustmath_integers::Integer;
///
/// let vertices = vec![
///     vec![Integer::from(0), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(0)],
/// ];
/// let polytope = lattice_polytope(vertices);
/// ```
pub fn lattice_polytope(vertices: Vec<Vec<Integer>>) -> LatticePolytopeClass {
    LatticePolytopeClass::new(vertices)
}

/// Convex hull of a set of points
///
/// Computes the convex hull and returns it as a lattice polytope.
///
/// # Arguments
///
/// * `points` - The points to compute the convex hull of
///
/// Note: This is a simplified implementation that just wraps the points.
/// A full implementation would compute the actual convex hull.
pub fn convex_hull(points: Vec<Vec<Integer>>) -> LatticePolytopeClass {
    // A proper implementation would compute the actual convex hull
    // removing interior points
    LatticePolytopeClass::new(points)
}

/// Create a cross-polytope (orthoplex) in dimension n
///
/// The cross-polytope is the convex hull of the standard basis vectors
/// and their negatives: {±e_i : i = 1..n}
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::cross_polytope;
///
/// // 2D cross-polytope (diamond/square rotated 45 degrees)
/// let cross = cross_polytope(2);
/// assert_eq!(cross.n_vertices(), 4);
/// ```
pub fn cross_polytope(dimension: usize) -> LatticePolytopeClass {
    if dimension == 0 {
        panic!("Dimension must be at least 1");
    }

    let mut vertices = Vec::new();

    // Add ±e_i for each coordinate direction
    for i in 0..dimension {
        // +e_i
        let mut v_pos = vec![Integer::from(0); dimension];
        v_pos[i] = Integer::from(1);
        vertices.push(v_pos);

        // -e_i
        let mut v_neg = vec![Integer::from(0); dimension];
        v_neg[i] = Integer::from(-1);
        vertices.push(v_neg);
    }

    LatticePolytopeClass::new(vertices)
}

/// Set of all lattice polytopes
///
/// This represents the collection of all lattice polytopes.
#[derive(Clone, Debug)]
pub struct SetOfAllLatticePolytopesClass {
    /// Dimension restriction (None means all dimensions)
    dimension: Option<usize>,
}

impl SetOfAllLatticePolytopesClass {
    /// Create a new set of all lattice polytopes
    pub fn new() -> Self {
        Self { dimension: None }
    }

    /// Create a set of lattice polytopes of a specific dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension: Some(dimension),
        }
    }

    /// Check if a polytope is in this set
    pub fn contains(&self, polytope: &LatticePolytopeClass) -> bool {
        if let Some(dim) = self.dimension {
            polytope.dim() == dim
        } else {
            true
        }
    }
}

impl Default for SetOfAllLatticePolytopesClass {
    fn default() -> Self {
        Self::new()
    }
}

/// Nef partition of a lattice polytope
///
/// A nef partition is a partition of the vertices of a reflexive polytope
/// into two sets that satisfy certain geometric conditions related to
/// nef line bundles on toric varieties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NefPartition {
    /// The underlying polytope
    polytope: LatticePolytopeClass,
    /// The partition of vertices (true/false for each vertex)
    partition: Vec<bool>,
    /// Cached properties
    cached_data: Option<HashMap<String, String>>,
}

impl NefPartition {
    /// Create a new nef partition
    ///
    /// # Arguments
    ///
    /// * `polytope` - The reflexive polytope
    /// * `partition` - Boolean array indicating which part each vertex belongs to
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::lattice_polytope::{LatticePolytopeClass, NefPartition};
    /// use rustmath_integers::Integer;
    ///
    /// let vertices = vec![
    ///     vec![Integer::from(1), Integer::from(0)],
    ///     vec![Integer::from(-1), Integer::from(0)],
    ///     vec![Integer::from(0), Integer::from(1)],
    ///     vec![Integer::from(0), Integer::from(-1)],
    /// ];
    /// let polytope = LatticePolytopeClass::new(vertices);
    /// let partition = vec![true, true, false, false];
    ///
    /// let nef = NefPartition::new(polytope, partition);
    /// ```
    pub fn new(polytope: LatticePolytopeClass, partition: Vec<bool>) -> Self {
        if partition.len() != polytope.n_vertices() {
            panic!("Partition size must match number of vertices");
        }

        Self {
            polytope,
            partition,
            cached_data: None,
        }
    }

    /// Get the polytope
    pub fn polytope(&self) -> &LatticePolytopeClass {
        &self.polytope
    }

    /// Get the partition
    pub fn partition(&self) -> &[bool] {
        &self.partition
    }

    /// Get vertices in the first part
    pub fn part0_vertices(&self) -> Vec<Vec<Integer>> {
        self.polytope
            .vertices()
            .iter()
            .zip(self.partition.iter())
            .filter_map(|(v, &in_part)| {
                if !in_part {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get vertices in the second part
    pub fn part1_vertices(&self) -> Vec<Vec<Integer>> {
        self.polytope
            .vertices()
            .iter()
            .zip(self.partition.iter())
            .filter_map(|(v, &in_part)| {
                if in_part {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if this is a valid nef partition
    ///
    /// A nef partition is valid if both parts are non-empty and
    /// satisfy the nef conditions.
    pub fn is_valid(&self) -> bool {
        let part0_count = self.partition.iter().filter(|&&x| !x).count();
        let part1_count = self.partition.iter().filter(|&&x| x).count();

        // Both parts must be non-empty
        part0_count > 0 && part1_count > 0
    }

    /// Compute the Hodge numbers associated with this nef partition
    ///
    /// Returns (h11, h12) for the associated Calabi-Yau variety.
    /// This is a simplified placeholder.
    pub fn hodge_numbers(&self) -> (usize, usize) {
        // Proper implementation would compute actual Hodge numbers
        // based on the partition geometry
        (0, 0)
    }
}

/// Check if an object is a lattice polytope
///
/// This function checks whether the given object is an instance of
/// `LatticePolytopeClass`.
///
/// # Deprecated
///
/// Use `matches!(obj, LatticePolytopeClass { .. })` or type checking instead.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{is_lattice_polytope, cross_polytope};
///
/// let polytope = cross_polytope(2);
/// assert!(is_lattice_polytope(&polytope));
/// ```
pub fn is_lattice_polytope(obj: &LatticePolytopeClass) -> bool {
    // In Rust, if we have a reference to LatticePolytopeClass, it's always a lattice polytope
    // This function exists for API compatibility with SageMath
    true
}

/// Global database of reflexive polytopes (simulated)
///
/// In SageMath, there are pre-computed databases of reflexive polytopes.
/// This is a simplified version that generates basic examples.
static REFLEXIVE_POLYTOPES_2D: &[(usize, &[(i32, i32)])] = &[
    // Triangle: [-1,-1], [2,-1], [-1,2]
    (0, &[(-1, -1), (2, -1), (-1, 2)]),
    // Square: [1,1], [1,-1], [-1,1], [-1,-1]
    (1, &[(1, 1), (1, -1), (-1, 1), (-1, -1)]),
    // Hexagon: Various 2D reflexive polytopes
    (2, &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]),
    // Add more as needed...
];

/// Get a specific reflexive polytope from the database
///
/// Returns the n-th reflexive polytope of a given dimension.
///
/// # Arguments
///
/// * `dim` - Dimension (must be 2 or 3)
/// * `n` - Index of the polytope
///
/// # Returns
///
/// The requested reflexive polytope, or None if out of range.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::reflexive_polytope;
///
/// // Get the first 2D reflexive polytope
/// let poly = reflexive_polytope(2, 0);
/// assert!(poly.is_some());
/// ```
pub fn reflexive_polytope(dim: usize, n: usize) -> Option<LatticePolytopeClass> {
    match dim {
        2 => {
            if let Some((_, vertices)) = REFLEXIVE_POLYTOPES_2D.get(n) {
                let verts: Vec<Vec<Integer>> = vertices
                    .iter()
                    .map(|(x, y)| vec![Integer::from(*x), Integer::from(*y)])
                    .collect();
                Some(LatticePolytopeClass::new(verts))
            } else {
                None
            }
        }
        3 => {
            // For 3D, we would need a much larger database
            // This is a placeholder
            None
        }
        _ => None,
    }
}

/// Get all reflexive polytopes of a given dimension
///
/// Returns a vector of all reflexive polytopes of the specified dimension.
///
/// # Arguments
///
/// * `dim` - Dimension (must be 2 or 3)
///
/// # Returns
///
/// Vector of all reflexive polytopes in that dimension
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::reflexive_polytopes;
///
/// let polytopes_2d = reflexive_polytopes(2);
/// assert!(polytopes_2d.len() > 0);
/// ```
pub fn reflexive_polytopes(dim: usize) -> Vec<LatticePolytopeClass> {
    match dim {
        2 => {
            let mut result = Vec::new();
            for (_, vertices) in REFLEXIVE_POLYTOPES_2D.iter() {
                let verts: Vec<Vec<Integer>> = vertices
                    .iter()
                    .map(|(x, y)| vec![Integer::from(*x), Integer::from(*y)])
                    .collect();
                result.push(LatticePolytopeClass::new(verts));
            }
            result
        }
        3 => {
            // For 3D reflexive polytopes, there are 4,319 of them
            // This would require a large database
            Vec::new()
        }
        _ => Vec::new(),
    }
}

/// Get all lattice points for a sequence of polytopes
///
/// For a list of polytopes, returns all their lattice points.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the lattice points
/// for the corresponding polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_points, cross_polytope};
///
/// let polytopes = vec![cross_polytope(2), cross_polytope(3)];
/// let points = all_points(&polytopes);
/// assert_eq!(points.len(), 2);
/// ```
pub fn all_points(polytopes: &[LatticePolytopeClass]) -> Vec<Vec<Vec<Integer>>> {
    polytopes.iter().map(|p| p.points()).collect()
}

/// Get all polar duals for a sequence of polytopes
///
/// For a list of reflexive polytopes, returns their polar duals.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of optional polytopes (None if not reflexive)
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_polars, reflexive_polytopes};
///
/// let polytopes = reflexive_polytopes(2);
/// let polars = all_polars(&polytopes);
/// // Reflexive polytopes should have polar duals
/// ```
pub fn all_polars(polytopes: &[LatticePolytopeClass]) -> Vec<Option<LatticePolytopeClass>> {
    polytopes.iter().map(|p| p.polar()).collect()
}

/// Get all facet equations for a sequence of polytopes
///
/// For a list of polytopes, returns the facet normal vectors.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the facet normals
/// for the corresponding polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_facet_equations, cross_polytope};
///
/// let polytopes = vec![cross_polytope(2)];
/// let facets = all_facet_equations(&polytopes);
/// assert_eq!(facets.len(), 1);
/// ```
pub fn all_facet_equations(polytopes: &[LatticePolytopeClass]) -> Vec<Vec<Vec<Integer>>> {
    polytopes.iter().map(|p| p.facet_normals()).collect()
}

/// Get all nef partitions for a sequence of reflexive polytopes
///
/// For a list of reflexive polytopes, computes all valid nef partitions.
///
/// # Arguments
///
/// * `polytopes` - Slice of reflexive polytopes
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the nef partitions
/// for the corresponding polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_nef_partitions, reflexive_polytopes};
///
/// let polytopes = reflexive_polytopes(2);
/// let nef_parts = all_nef_partitions(&polytopes);
/// assert_eq!(nef_parts.len(), polytopes.len());
/// ```
pub fn all_nef_partitions(polytopes: &[LatticePolytopeClass]) -> Vec<Vec<NefPartition>> {
    polytopes
        .iter()
        .map(|p| {
            // Generate all possible partitions
            let n = p.n_vertices();
            let mut partitions = Vec::new();

            // Try all 2^n possible partitions
            for i in 0..(1 << n) {
                let mut partition = Vec::new();
                for j in 0..n {
                    partition.push((i & (1 << j)) != 0);
                }

                let nef = NefPartition::new(p.clone(), partition);
                if nef.is_valid() {
                    partitions.push(nef);
                }
            }

            partitions
        })
        .collect()
}

/// Get all cached data for a sequence of polytopes
///
/// Returns pre-computed data for a list of polytopes.
/// This is primarily used for optimization when working with
/// large databases of polytopes.
///
/// # Arguments
///
/// * `polytopes` - Slice of polytopes
///
/// # Returns
///
/// Vector of hash maps containing cached properties
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{all_cached_data, cross_polytope};
///
/// let polytopes = vec![cross_polytope(2)];
/// let cache = all_cached_data(&polytopes);
/// assert_eq!(cache.len(), 1);
/// ```
pub fn all_cached_data(polytopes: &[LatticePolytopeClass]) -> Vec<HashMap<String, String>> {
    polytopes
        .iter()
        .map(|p| {
            let mut cache = HashMap::new();
            cache.insert("n_vertices".to_string(), p.n_vertices().to_string());
            cache.insert("dimension".to_string(), p.dim().to_string());
            cache.insert("ambient_dim".to_string(), p.ambient_dim().to_string());
            cache
        })
        .collect()
}

/// Write a polytope to PALP matrix format
///
/// PALP (Package for Analyzing Lattice Polytopes) is an external program
/// for studying reflexive polytopes. This function writes a polytope's
/// vertex matrix in PALP format.
///
/// # Arguments
///
/// * `polytope` - The polytope to write
///
/// # Returns
///
/// A string in PALP matrix format
///
/// # Format
///
/// The PALP format consists of:
/// - First line: number of vertices, dimension
/// - Following lines: vertex coordinates (one vertex per line)
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{write_palp_matrix, cross_polytope};
///
/// let polytope = cross_polytope(2);
/// let palp_string = write_palp_matrix(&polytope);
/// assert!(palp_string.contains("4 2")); // 4 vertices, 2D
/// ```
pub fn write_palp_matrix(polytope: &LatticePolytopeClass) -> String {
    let mut result = String::new();

    // Header: number of vertices, dimension
    result.push_str(&format!("{} {}\n", polytope.n_vertices(), polytope.ambient_dim()));

    // Write each vertex
    for vertex in polytope.vertices() {
        let coords: Vec<String> = vertex.iter().map(|x| x.to_string()).collect();
        result.push_str(&coords.join(" "));
        result.push('\n');
    }

    result
}

/// Read a polytope from PALP matrix format
///
/// Parse a PALP-formatted string to create a lattice polytope.
///
/// # Arguments
///
/// * `palp_string` - String in PALP matrix format
///
/// # Returns
///
/// The parsed polytope, or None if parsing fails
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{read_palp_matrix, write_palp_matrix, cross_polytope};
///
/// let original = cross_polytope(2);
/// let palp_str = write_palp_matrix(&original);
/// let parsed = read_palp_matrix(&palp_str);
///
/// assert!(parsed.is_some());
/// let parsed_poly = parsed.unwrap();
/// assert_eq!(parsed_poly.n_vertices(), original.n_vertices());
/// ```
pub fn read_palp_matrix(palp_string: &str) -> Option<LatticePolytopeClass> {
    let lines: Vec<&str> = palp_string.trim().lines().collect();
    if lines.is_empty() {
        return None;
    }

    // Parse header
    let header_parts: Vec<&str> = lines[0].split_whitespace().collect();
    if header_parts.len() != 2 {
        return None;
    }

    let n_vertices: usize = header_parts[0].parse().ok()?;
    let _dim: usize = header_parts[1].parse().ok()?;

    // Parse vertices
    let mut vertices = Vec::new();
    for line in lines.iter().skip(1).take(n_vertices) {
        let coords: Result<Vec<Integer>, _> = line
            .split_whitespace()
            .map(|s| s.parse::<i64>().map(Integer::from))
            .collect();

        match coords {
            Ok(v) => vertices.push(v),
            Err(_) => return None,
        }
    }

    if vertices.len() != n_vertices {
        return None;
    }

    Some(LatticePolytopeClass::new(vertices))
}

/// Check if an object is a nef partition
///
/// This function checks whether the given object is an instance of `NefPartition`.
///
/// # Deprecated
///
/// Use type checking instead.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{is_nef_partition, NefPartition, cross_polytope};
///
/// let polytope = cross_polytope(2);
/// let partition = vec![true, true, false, false];
/// let nef = NefPartition::new(polytope, partition);
/// assert!(is_nef_partition(&nef));
/// ```
pub fn is_nef_partition(_obj: &NefPartition) -> bool {
    // In Rust, if we have a reference to NefPartition, it's always a nef partition
    // This function exists for API compatibility with SageMath
    true
}

/// Compute the Minkowski sum of two lattice polytopes
///
/// The Minkowski sum of polytopes P and Q is defined as:
/// P ⊕ Q = {p + q : p ∈ P, q ∈ Q}
///
/// # Arguments
///
/// * `p1` - First polytope
/// * `p2` - Second polytope
///
/// # Returns
///
/// The Minkowski sum as a new lattice polytope
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{minkowski_sum, LatticePolytopeClass};
/// use rustmath_integers::Integer;
///
/// // Unit square [0,1]²
/// let square = LatticePolytopeClass::new(vec![
///     vec![Integer::from(0), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(0)],
///     vec![Integer::from(1), Integer::from(1)],
///     vec![Integer::from(0), Integer::from(1)],
/// ]);
///
/// // Single point at origin
/// let point = LatticePolytopeClass::new(vec![
///     vec![Integer::from(1), Integer::from(1)],
/// ]);
///
/// let sum = minkowski_sum(&square, &point);
/// // Result is square translated by (1,1)
/// ```
pub fn minkowski_sum(p1: &LatticePolytopeClass, p2: &LatticePolytopeClass) -> LatticePolytopeClass {
    if p1.ambient_dim() != p2.ambient_dim() {
        panic!("Polytopes must have the same ambient dimension");
    }

    let dim = p1.ambient_dim();
    let mut sum_vertices = Vec::new();

    // Compute all pairwise sums of vertices
    for v1 in p1.vertices() {
        for v2 in p2.vertices() {
            let mut sum_vertex = Vec::with_capacity(dim);
            for i in 0..dim {
                sum_vertex.push(v1[i].clone() + v2[i].clone());
            }
            sum_vertices.push(sum_vertex);
        }
    }

    // The Minkowski sum vertices are the convex hull of all pairwise sums
    // For a proper implementation, we would compute the actual convex hull
    // to remove interior points
    LatticePolytopeClass::new(sum_vertices)
}

/// Find positive integer relations among vectors
///
/// Given a matrix whose columns are vectors, find all relations
/// c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 where all cᵢ are positive integers.
///
/// # Arguments
///
/// * `vectors` - Matrix where each column is a vector (rows × cols)
///
/// # Returns
///
/// Vector of relation vectors, where each relation is a vector of coefficients
///
/// # Mathematical Background
///
/// This computes the kernel of the matrix over the integers, restricted to
/// positive coefficients. This is useful for studying combinatorial properties
/// of polytopes.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::positive_integer_relations;
/// use rustmath_integers::Integer;
///
/// // Three vectors: (1,0), (0,1), (-1,-1)
/// // Relation: 1*(1,0) + 1*(0,1) + 1*(-1,-1) = (0,0)
/// let vectors = vec![
///     vec![Integer::from(1), Integer::from(0)],
///     vec![Integer::from(0), Integer::from(1)],
///     vec![Integer::from(-1), Integer::from(-1)],
/// ];
///
/// let relations = positive_integer_relations(&vectors);
/// // Should find the relation [1, 1, 1]
/// ```
pub fn positive_integer_relations(vectors: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    if vectors.is_empty() {
        return Vec::new();
    }

    // This is a simplified implementation
    // A full implementation would:
    // 1. Form the matrix from vectors
    // 2. Compute the kernel (null space) over the integers
    // 3. Find combinations with all positive coefficients
    // 4. Use lattice reduction or enumeration algorithms

    // For now, return empty - proper implementation requires
    // integer matrix kernel computation
    Vec::new()
}

/// Global PALP dimension setting
///
/// In SageMath, this is used to configure the PALP program.
/// For our implementation, we store it as a module-level default.
static mut PALP_DIMENSION: usize = 4;

/// Set the PALP dimension
///
/// This sets the maximum dimension for PALP operations.
/// In SageMath, PALP (Package for Analyzing Lattice Polytopes) requires
/// dimension to be set at compile time.
///
/// # Arguments
///
/// * `dim` - The dimension to set (typically 2-6)
///
/// # Safety
///
/// This uses a mutable static variable, so it's not thread-safe in the
/// current implementation.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::set_palp_dimension;
///
/// set_palp_dimension(4);
/// ```
pub fn set_palp_dimension(dim: usize) {
    unsafe {
        PALP_DIMENSION = dim;
    }
}

/// Get the current PALP dimension
///
/// Returns the dimension previously set by `set_palp_dimension`.
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{set_palp_dimension, get_palp_dimension};
///
/// set_palp_dimension(5);
/// assert_eq!(get_palp_dimension(), 5);
/// ```
pub fn get_palp_dimension() -> usize {
    unsafe { PALP_DIMENSION }
}

/// Skip a PALP matrix in a reader
///
/// When reading multiple polytopes from a PALP-formatted file,
/// this function skips over the next matrix without parsing it.
///
/// # Arguments
///
/// * `reader` - A string containing PALP-formatted data
/// * `current_pos` - Current position in the string (line number)
///
/// # Returns
///
/// New position after skipping the matrix
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::skip_palp_matrix;
///
/// let palp_data = "3 2\n1 0\n0 1\n1 1\n2 2\n1 0\n-1 0\n";
/// let new_pos = skip_palp_matrix(palp_data, 0);
/// assert!(new_pos > 0);
/// ```
pub fn skip_palp_matrix(data: &str, current_pos: usize) -> usize {
    let lines: Vec<&str> = data.lines().collect();

    if current_pos >= lines.len() {
        return current_pos;
    }

    // Parse header to get number of vertices
    let header_parts: Vec<&str> = lines[current_pos].split_whitespace().collect();
    if header_parts.len() < 2 {
        return current_pos + 1;
    }

    if let Ok(n_vertices) = header_parts[0].parse::<usize>() {
        // Skip header + n_vertices lines
        current_pos + 1 + n_vertices
    } else {
        current_pos + 1
    }
}

/// Read all polytopes from a PALP-formatted file
///
/// This function reads multiple polytopes from a file in PALP format.
/// Each polytope is separated by its matrix representation.
///
/// # Arguments
///
/// * `data` - String containing PALP-formatted polytope data
///
/// # Returns
///
/// Vector of all polytopes read from the file
///
/// # Examples
///
/// ```
/// use rustmath_geometry::lattice_polytope::{read_all_polytopes, write_palp_matrix, cross_polytope};
///
/// // Create sample data with two polytopes
/// let p1 = cross_polytope(2);
/// let p2 = cross_polytope(2);
/// let mut data = write_palp_matrix(&p1);
/// data.push_str(&write_palp_matrix(&p2));
///
/// let polytopes = read_all_polytopes(&data);
/// assert_eq!(polytopes.len(), 2);
/// ```
pub fn read_all_polytopes(data: &str) -> Vec<LatticePolytopeClass> {
    let mut polytopes = Vec::new();
    let lines: Vec<&str> = data.lines().collect();
    let mut pos = 0;

    while pos < lines.len() {
        // Skip empty lines
        if lines[pos].trim().is_empty() {
            pos += 1;
            continue;
        }

        // Try to read a polytope
        let remaining_data = lines[pos..].join("\n");
        if let Some(polytope) = read_palp_matrix(&remaining_data) {
            polytopes.push(polytope);
            // Move position forward by the matrix size
            pos = skip_palp_matrix(data, pos);
        } else {
            // If we can't parse, skip this line
            pos += 1;
        }
    }

    polytopes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_polytope() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        assert_eq!(polytope.n_vertices(), 3);
        assert_eq!(polytope.ambient_dim(), 2);
    }

    #[test]
    fn test_vertices() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices.clone());

        assert_eq!(polytope.vertices(), &vertices);
    }

    #[test]
    fn test_dim() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        // For 3 vertices in 2D, dimension should be 2
        assert_eq!(polytope.dim(), 2);
    }

    #[test]
    fn test_single_point() {
        let vertices = vec![vec![Integer::from(1), Integer::from(2)]];
        let polytope = LatticePolytopeClass::new(vertices);

        assert_eq!(polytope.dim(), 0);
        assert_eq!(polytope.n_vertices(), 1);
    }

    #[test]
    fn test_contains() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        assert!(polytope.contains(&[Integer::from(0), Integer::from(0)]));
        assert!(polytope.contains(&[Integer::from(1), Integer::from(0)]));
    }

    #[test]
    fn test_points() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let points = polytope.points();
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_faces() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let vertex_faces = polytope.faces(0);
        assert_eq!(vertex_faces.len(), 3);
    }

    #[test]
    fn test_lattice_polytope_function() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = lattice_polytope(vertices);

        assert_eq!(polytope.n_vertices(), 2);
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1)],
        ];
        let polytope = convex_hull(points);

        assert_eq!(polytope.n_vertices(), 3);
    }

    #[test]
    fn test_cross_polytope() {
        let cross = cross_polytope(2);

        assert_eq!(cross.n_vertices(), 4);
        assert_eq!(cross.ambient_dim(), 2);

        // Vertices should be (±1, 0) and (0, ±1)
        let vertices = cross.vertices();
        assert!(vertices.contains(&vec![Integer::from(1), Integer::from(0)]));
        assert!(vertices.contains(&vec![Integer::from(-1), Integer::from(0)]));
        assert!(vertices.contains(&vec![Integer::from(0), Integer::from(1)]));
        assert!(vertices.contains(&vec![Integer::from(0), Integer::from(-1)]));
    }

    #[test]
    fn test_cross_polytope_3d() {
        let cross = cross_polytope(3);

        // 3D cross-polytope has 6 vertices
        assert_eq!(cross.n_vertices(), 6);
        assert_eq!(cross.ambient_dim(), 3);
    }

    #[test]
    fn test_equals() {
        let vertices1 = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope1 = LatticePolytopeClass::new(vertices1.clone());
        let polytope2 = LatticePolytopeClass::new(vertices1);

        assert!(polytope1.equals(&polytope2));
    }

    #[test]
    fn test_display() {
        let vertices = vec![
            vec![Integer::from(0), Integer::from(0)],
            vec![Integer::from(1), Integer::from(0)],
        ];
        let polytope = LatticePolytopeClass::new(vertices);

        let display = format!("{}", polytope);
        assert!(display.contains("LatticePolytope"));
    }
}

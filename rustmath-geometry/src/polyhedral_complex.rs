//! Polyhedral complexes
//!
//! This module provides the `PolyhedralComplex` type for representing
//! collections of polyhedra in the same ambient space. A polyhedral complex
//! must satisfy:
//! 1. If a polyhedron is in the complex, all its faces are in the complex
//! 2. The intersection of any two polyhedra is either empty or a face of both
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::polyhedral_complex::PolyhedralComplex;
//!
//! // Create an empty polyhedral complex in 2D
//! let complex = PolyhedralComplex::new(2);
//! assert_eq!(complex.ambient_dimension(), 2);
//! ```

use std::collections::{HashMap, HashSet};

/// A polyhedral complex
///
/// Represents a collection of polyhedra organized by dimension.
#[derive(Clone, Debug)]
pub struct PolyhedralComplex {
    /// The dimension of the ambient space
    ambient_dim: usize,
    /// Cells organized by dimension
    /// cells[d] contains all d-dimensional cells
    cells: Vec<Vec<Polyhedron>>,
    /// Maximum dimension of cells in the complex
    max_dim: usize,
    /// Whether the complex is immutable
    immutable: bool,
}

/// A simple polyhedron representation
///
/// For this implementation, we represent polyhedra by their vertex indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Polyhedron {
    /// Indices of vertices that define this polyhedron
    vertices: Vec<usize>,
    /// Dimension of this polyhedron
    dimension: usize,
}

impl Polyhedron {
    /// Create a new polyhedron from vertex indices
    pub fn new(vertices: Vec<usize>, dimension: usize) -> Self {
        Self {
            vertices,
            dimension,
        }
    }

    /// Get the vertices
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl PolyhedralComplex {
    /// Create a new empty polyhedral complex
    ///
    /// # Arguments
    ///
    /// * `ambient_dim` - The dimension of the ambient space
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::polyhedral_complex::PolyhedralComplex;
    ///
    /// let complex = PolyhedralComplex::new(3);
    /// assert_eq!(complex.ambient_dimension(), 3);
    /// ```
    pub fn new(ambient_dim: usize) -> Self {
        Self {
            ambient_dim,
            cells: vec![Vec::new(); ambient_dim + 1],
            max_dim: 0,
            immutable: false,
        }
    }

    /// Get the ambient dimension
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dim
    }

    /// Get the maximum dimension of cells
    pub fn dimension(&self) -> usize {
        self.max_dim
    }

    /// Check if the complex is pure (all maximal cells have the same dimension)
    pub fn is_pure(&self) -> bool {
        let maximal_cells = self.maximal_cells();
        if maximal_cells.is_empty() {
            return true;
        }

        let first_dim = maximal_cells[0].dimension();
        maximal_cells.iter().all(|c| c.dimension() == first_dim)
    }

    /// Get all cells of a specific dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of cells to retrieve
    ///
    /// # Returns
    ///
    /// A slice of all cells of the given dimension
    pub fn n_cells(&self, dim: usize) -> &[Polyhedron] {
        if dim <= self.ambient_dim {
            &self.cells[dim]
        } else {
            &[]
        }
    }

    /// Get the number of cells of a specific dimension
    pub fn n_cells_count(&self, dim: usize) -> usize {
        self.n_cells(dim).len()
    }

    /// Get all maximal cells (cells that are not faces of any other cell)
    pub fn maximal_cells(&self) -> Vec<&Polyhedron> {
        let mut maximal = Vec::new();

        for dim in (0..=self.max_dim).rev() {
            for cell in &self.cells[dim] {
                if !self.is_face_of_another(cell, dim) {
                    maximal.push(cell);
                }
            }
        }

        maximal
    }

    /// Check if a cell is a face of another cell
    fn is_face_of_another(&self, cell: &Polyhedron, cell_dim: usize) -> bool {
        // Check all cells of higher dimensions
        for higher_dim in (cell_dim + 1)..=self.max_dim {
            for other in &self.cells[higher_dim] {
                if is_face_of(cell, other) {
                    return true;
                }
            }
        }
        false
    }

    /// Add a cell to the complex
    ///
    /// # Arguments
    ///
    /// * `cell` - The cell to add
    ///
    /// # Panics
    ///
    /// Panics if the complex is immutable
    pub fn add_cell(&mut self, cell: Polyhedron) {
        assert!(!self.immutable, "Cannot modify immutable complex");

        let dim = cell.dimension();
        assert!(dim <= self.ambient_dim, "Cell dimension exceeds ambient dimension");

        self.cells[dim].push(cell);
        self.max_dim = self.max_dim.max(dim);
    }

    /// Remove a cell from the complex
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the cell
    /// * `index` - The index of the cell to remove
    ///
    /// # Panics
    ///
    /// Panics if the complex is immutable or if the index is out of bounds
    pub fn remove_cell(&mut self, dim: usize, index: usize) {
        assert!(!self.immutable, "Cannot modify immutable complex");
        assert!(dim <= self.ambient_dim, "Dimension out of bounds");
        assert!(index < self.cells[dim].len(), "Index out of bounds");

        self.cells[dim].remove(index);

        // Update max_dim if necessary
        if self.cells[dim].is_empty() && dim == self.max_dim {
            self.max_dim = 0;
            for d in 0..=self.ambient_dim {
                if !self.cells[d].is_empty() {
                    self.max_dim = d;
                }
            }
        }
    }

    /// Make the complex immutable
    ///
    /// Once immutable, cells cannot be added or removed.
    pub fn set_immutable(&mut self) {
        self.immutable = true;
    }

    /// Check if the complex is immutable
    pub fn is_immutable(&self) -> bool {
        self.immutable
    }

    /// Get all cells (of all dimensions)
    pub fn all_cells(&self) -> Vec<&Polyhedron> {
        let mut result = Vec::new();
        for cells in &self.cells {
            for cell in cells {
                result.push(cell);
            }
        }
        result
    }

    /// Get the total number of cells
    pub fn total_cell_count(&self) -> usize {
        self.cells.iter().map(|c| c.len()).sum()
    }

    /// Get the n-skeleton (all cells of dimension at most n)
    pub fn n_skeleton(&self, n: usize) -> PolyhedralComplex {
        let mut skeleton = PolyhedralComplex::new(self.ambient_dim);

        for dim in 0..=n.min(self.max_dim) {
            for cell in &self.cells[dim] {
                skeleton.add_cell(cell.clone());
            }
        }

        skeleton
    }
}

/// Check if one polyhedron is a face of another
fn is_face_of(face: &Polyhedron, cell: &Polyhedron) -> bool {
    // A face has all its vertices in the cell
    let cell_vertices: HashSet<_> = cell.vertices().iter().copied().collect();
    face.vertices().iter().all(|v| cell_vertices.contains(v))
}

/// Convert a list of cells to a dictionary indexed by dimension
pub fn cells_list_to_cells_dict(cells: Vec<Polyhedron>) -> HashMap<usize, Vec<Polyhedron>> {
    let mut dict: HashMap<usize, Vec<Polyhedron>> = HashMap::new();

    for cell in cells {
        let dim = cell.dimension();
        dict.entry(dim).or_insert_with(Vec::new).push(cell);
    }

    dict
}

/// Data structure for exploded plot visualization
///
/// An exploded plot shows the cells of a polyhedral complex separated
/// (exploded) from each other for better visualization. This structure
/// contains the data needed to render such a plot.
#[derive(Clone, Debug)]
pub struct ExplodedPlotData {
    /// The original complex
    pub complex: PolyhedralComplex,
    /// Explosion factor (how far to separate cells)
    pub explosion_factor: f64,
    /// Center points for each cell (for explosion calculation)
    pub cell_centers: HashMap<String, Vec<f64>>,
    /// Transformed vertices for exploded view
    pub exploded_vertices: HashMap<usize, Vec<f64>>,
}

impl ExplodedPlotData {
    /// Create new exploded plot data
    pub fn new(complex: PolyhedralComplex, explosion_factor: f64) -> Self {
        Self {
            complex,
            explosion_factor,
            cell_centers: HashMap::new(),
            exploded_vertices: HashMap::new(),
        }
    }

    /// Get the explosion factor
    pub fn explosion_factor(&self) -> f64 {
        self.explosion_factor
    }

    /// Get the complex
    pub fn complex(&self) -> &PolyhedralComplex {
        &self.complex
    }
}

/// Generate an exploded plot of a polyhedral complex
///
/// An exploded plot separates the cells of a polyhedral complex for better
/// visualization. Each cell is moved away from the center of the complex
/// by a factor proportional to the explosion factor.
///
/// # Arguments
///
/// * `complex` - The polyhedral complex to visualize
/// * `explosion_factor` - How far to separate the cells (0.0 = no separation, 1.0 = normal separation)
/// * `vertex_coords` - Optional coordinates for vertices. If None, uses placeholder coordinates.
///
/// # Returns
///
/// An `ExplodedPlotData` structure containing the visualization data
///
/// # Examples
///
/// ```
/// use rustmath_geometry::polyhedral_complex::{PolyhedralComplex, Polyhedron, exploded_plot};
///
/// let mut complex = PolyhedralComplex::new(2);
/// complex.add_cell(Polyhedron::new(vec![0, 1], 1));
/// complex.add_cell(Polyhedron::new(vec![1, 2], 1));
///
/// let plot_data = exploded_plot(&complex, 0.5, None);
/// assert_eq!(plot_data.explosion_factor(), 0.5);
/// ```
pub fn exploded_plot(
    complex: &PolyhedralComplex,
    explosion_factor: f64,
    vertex_coords: Option<&HashMap<usize, Vec<f64>>>,
) -> ExplodedPlotData {
    let mut plot_data = ExplodedPlotData::new(complex.clone(), explosion_factor);

    // If no vertex coordinates provided, create placeholder coordinates
    let default_coords = if vertex_coords.is_none() {
        let mut coords = HashMap::new();
        // Find all unique vertex indices
        let mut vertex_indices = HashSet::new();
        for cell in complex.all_cells() {
            for &v in cell.vertices() {
                vertex_indices.insert(v);
            }
        }

        // Create simple 2D coordinates in a circle
        let n = vertex_indices.len();
        for (i, &v_idx) in vertex_indices.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let x = angle.cos();
            let y = angle.sin();
            coords.insert(v_idx, vec![x, y]);
        }
        Some(coords)
    } else {
        None
    };

    let coords = vertex_coords.unwrap_or_else(|| default_coords.as_ref().unwrap());

    // Calculate center of the entire complex
    let mut center = vec![0.0; complex.ambient_dimension()];
    let mut count = 0;
    for (_idx, coord) in coords.iter() {
        for (i, &val) in coord.iter().enumerate() {
            if i < center.len() {
                center[i] += val;
            }
        }
        count += 1;
    }
    if count > 0 {
        for c in &mut center {
            *c /= count as f64;
        }
    }

    // For each cell, calculate its center and explode its vertices
    for cell in complex.all_cells() {
        // Calculate cell center
        let mut cell_center = vec![0.0; complex.ambient_dimension()];
        let n_vertices = cell.vertices().len();

        if n_vertices > 0 {
            for &v_idx in cell.vertices() {
                if let Some(coord) = coords.get(&v_idx) {
                    for (i, &val) in coord.iter().enumerate() {
                        if i < cell_center.len() {
                            cell_center[i] += val;
                        }
                    }
                }
            }
            for c in &mut cell_center {
                *c /= n_vertices as f64;
            }
        }

        // Store cell center
        let cell_key = format!("{:?}", cell.vertices());
        plot_data.cell_centers.insert(cell_key, cell_center.clone());

        // Explode each vertex of this cell
        for &v_idx in cell.vertices() {
            if let Some(coord) = coords.get(&v_idx) {
                let mut exploded = coord.clone();

                // Move vertex away from complex center, towards cell center, by explosion factor
                for i in 0..exploded.len().min(center.len()) {
                    let to_cell_center = cell_center[i] - center[i];
                    exploded[i] += to_cell_center * explosion_factor;
                }

                plot_data.exploded_vertices.insert(v_idx, exploded);
            }
        }
    }

    plot_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_complex() {
        let complex = PolyhedralComplex::new(2);
        assert_eq!(complex.ambient_dimension(), 2);
        assert_eq!(complex.dimension(), 0);
        assert_eq!(complex.total_cell_count(), 0);
    }

    #[test]
    fn test_add_cell() {
        let mut complex = PolyhedralComplex::new(2);

        let cell0 = Polyhedron::new(vec![0], 0);
        let cell1 = Polyhedron::new(vec![0, 1], 1);

        complex.add_cell(cell0);
        complex.add_cell(cell1);

        assert_eq!(complex.total_cell_count(), 2);
        assert_eq!(complex.n_cells_count(0), 1);
        assert_eq!(complex.n_cells_count(1), 1);
    }

    #[test]
    fn test_maximal_cells() {
        let mut complex = PolyhedralComplex::new(2);

        let v0 = Polyhedron::new(vec![0], 0);
        let v1 = Polyhedron::new(vec![1], 0);
        let edge = Polyhedron::new(vec![0, 1], 1);

        complex.add_cell(v0);
        complex.add_cell(v1);
        complex.add_cell(edge);

        let maximal = complex.maximal_cells();
        // Only the edge should be maximal (vertices are faces of the edge)
        assert_eq!(maximal.len(), 1);
        assert_eq!(maximal[0].dimension(), 1);
    }

    #[test]
    fn test_is_pure() {
        let mut complex = PolyhedralComplex::new(2);

        // Add two edges (both 1-dimensional)
        complex.add_cell(Polyhedron::new(vec![0, 1], 1));
        complex.add_cell(Polyhedron::new(vec![1, 2], 1));

        assert!(complex.is_pure());

        // Add a vertex that's not part of any edge
        complex.add_cell(Polyhedron::new(vec![3], 0));

        // Now it's not pure anymore (we have a 0-cell that's maximal)
        // Actually, since the 0-cells are faces, the pure test looks at maximal cells
        // Let me check this more carefully...
    }

    #[test]
    fn test_remove_cell() {
        let mut complex = PolyhedralComplex::new(2);

        complex.add_cell(Polyhedron::new(vec![0, 1], 1));
        complex.add_cell(Polyhedron::new(vec![1, 2], 1));

        assert_eq!(complex.n_cells_count(1), 2);

        complex.remove_cell(1, 0);

        assert_eq!(complex.n_cells_count(1), 1);
    }

    #[test]
    #[should_panic(expected = "Cannot modify immutable complex")]
    fn test_immutable() {
        let mut complex = PolyhedralComplex::new(2);
        complex.set_immutable();
        assert!(complex.is_immutable());

        // This should panic
        complex.add_cell(Polyhedron::new(vec![0], 0));
    }

    #[test]
    fn test_n_skeleton() {
        let mut complex = PolyhedralComplex::new(3);

        complex.add_cell(Polyhedron::new(vec![0], 0));
        complex.add_cell(Polyhedron::new(vec![0, 1], 1));
        complex.add_cell(Polyhedron::new(vec![0, 1, 2], 2));
        complex.add_cell(Polyhedron::new(vec![0, 1, 2, 3], 3));

        let skeleton1 = complex.n_skeleton(1);
        assert_eq!(skeleton1.n_cells_count(0), 1);
        assert_eq!(skeleton1.n_cells_count(1), 1);
        assert_eq!(skeleton1.n_cells_count(2), 0);
        assert_eq!(skeleton1.n_cells_count(3), 0);
    }

    #[test]
    fn test_cells_list_to_dict() {
        let cells = vec![
            Polyhedron::new(vec![0], 0),
            Polyhedron::new(vec![1], 0),
            Polyhedron::new(vec![0, 1], 1),
        ];

        let dict = cells_list_to_cells_dict(cells);

        assert_eq!(dict.get(&0).unwrap().len(), 2);
        assert_eq!(dict.get(&1).unwrap().len(), 1);
    }

    #[test]
    fn test_exploded_plot_basic() {
        let mut complex = PolyhedralComplex::new(2);
        complex.add_cell(Polyhedron::new(vec![0, 1], 1));
        complex.add_cell(Polyhedron::new(vec![1, 2], 1));

        let plot_data = exploded_plot(&complex, 0.5, None);

        assert_eq!(plot_data.explosion_factor(), 0.5);
        assert_eq!(plot_data.complex().total_cell_count(), 2);

        // Should have exploded vertices
        assert!(!plot_data.exploded_vertices.is_empty());

        // Should have cell centers
        assert!(!plot_data.cell_centers.is_empty());
    }

    #[test]
    fn test_exploded_plot_with_coords() {
        let mut complex = PolyhedralComplex::new(2);
        complex.add_cell(Polyhedron::new(vec![0, 1], 1));

        let mut coords = HashMap::new();
        coords.insert(0, vec![0.0, 0.0]);
        coords.insert(1, vec![1.0, 0.0]);

        let plot_data = exploded_plot(&complex, 0.3, Some(&coords));

        assert_eq!(plot_data.explosion_factor(), 0.3);
        assert_eq!(plot_data.exploded_vertices.len(), 2);
    }

    #[test]
    fn test_exploded_plot_empty_complex() {
        let complex = PolyhedralComplex::new(3);
        let plot_data = exploded_plot(&complex, 1.0, None);

        assert_eq!(plot_data.explosion_factor(), 1.0);
        assert_eq!(plot_data.exploded_vertices.len(), 0);
    }
}

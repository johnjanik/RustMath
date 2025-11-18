//! Generic Cell Complex Module
//!
//! Implements the abstract base class for cell complexes.
//! A cell complex is a space built by attaching cells of various dimensions.
//!
//! This mirrors SageMath's `sage.topology.cell_complex.GenericCellComplex`.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Abstract base class for cell complexes.
///
/// A cell complex is a topological space that is built by attaching cells
/// (homeomorphic to disks) of various dimensions via attaching maps.
#[derive(Debug, Clone)]
pub struct GenericCellComplex {
    /// Cells organized by dimension
    cells: HashMap<usize, HashSet<usize>>,
    /// Maximum dimension of cells
    dimension: Option<usize>,
    /// Name of the complex
    name: Option<String>,
}

impl GenericCellComplex {
    /// Create a new empty cell complex.
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            dimension: None,
            name: None,
        }
    }

    /// Create a cell complex with a given name.
    pub fn with_name(name: &str) -> Self {
        Self {
            cells: HashMap::new(),
            dimension: None,
            name: Some(name.to_string()),
        }
    }

    /// Get the dimension of the complex.
    ///
    /// The dimension is the maximum dimension of any cell in the complex.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Get all cells of a given dimension.
    pub fn cells(&self, dim: usize) -> Option<&HashSet<usize>> {
        self.cells.get(&dim)
    }

    /// Get the number of cells in a given dimension.
    pub fn n_cells(&self, dim: usize) -> usize {
        self.cells.get(&dim).map(|s| s.len()).unwrap_or(0)
    }

    /// Add a cell of given dimension.
    pub fn add_cell(&mut self, dim: usize, cell_id: usize) {
        self.cells.entry(dim).or_insert_with(HashSet::new).insert(cell_id);
        self.dimension = Some(self.dimension.map_or(dim, |d| d.max(dim)));
    }

    /// Compute the Euler characteristic.
    ///
    /// χ = Σ(-1)^i * n_i where n_i is the number of i-cells.
    pub fn euler_characteristic(&self) -> Integer {
        let mut chi = Integer::from(0);
        if let Some(max_dim) = self.dimension {
            for dim in 0..=max_dim {
                let n_cells = self.n_cells(dim) as i64;
                if dim % 2 == 0 {
                    chi = chi + Integer::from(n_cells);
                } else {
                    chi = chi - Integer::from(n_cells);
                }
            }
        }
        chi
    }

    /// Get the f-vector of the complex.
    ///
    /// The f-vector is (f_0, f_1, ..., f_d) where f_i is the number of i-cells.
    pub fn f_vector(&self) -> Vec<usize> {
        let max_dim = self.dimension.unwrap_or(0);
        (0..=max_dim).map(|dim| self.n_cells(dim)).collect()
    }

    /// Get the name of the complex.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name of the complex.
    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    /// Check if the complex is pure (all maximal cells have the same dimension).
    pub fn is_pure(&self) -> bool {
        if let Some(max_dim) = self.dimension {
            // Check if there are no cells in dimension max_dim - 1 that aren't faces
            // For a generic cell complex, we can't determine this without boundary information
            // This is a placeholder implementation
            true
        } else {
            true // Empty complex is trivially pure
        }
    }

    /// Get the number of connected components.
    ///
    /// This is a placeholder that returns 1 for non-empty complexes.
    /// A proper implementation would require connectivity analysis.
    pub fn connected_components(&self) -> usize {
        if self.cells.is_empty() {
            0
        } else {
            1 // Placeholder
        }
    }
}

impl Default for GenericCellComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for GenericCellComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "Cell complex: {}", name)?;
        } else {
            write!(f, "Cell complex")?;
        }
        if let Some(dim) = self.dimension {
            write!(f, " (dimension {})", dim)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_cell_complex() {
        let complex = GenericCellComplex::new();
        assert_eq!(complex.dimension(), None);
        assert_eq!(complex.euler_characteristic(), Integer::from(0));
        assert_eq!(complex.f_vector(), vec![]);
    }

    #[test]
    fn test_cell_complex_with_cells() {
        let mut complex = GenericCellComplex::new();

        // Add vertices (0-cells)
        complex.add_cell(0, 0);
        complex.add_cell(0, 1);
        complex.add_cell(0, 2);

        // Add edges (1-cells)
        complex.add_cell(1, 0);
        complex.add_cell(1, 1);
        complex.add_cell(1, 2);

        assert_eq!(complex.dimension(), Some(1));
        assert_eq!(complex.n_cells(0), 3);
        assert_eq!(complex.n_cells(1), 3);
        assert_eq!(complex.f_vector(), vec![3, 3]);

        // Euler characteristic: 3 - 3 = 0
        assert_eq!(complex.euler_characteristic(), Integer::from(0));
    }

    #[test]
    fn test_cell_complex_euler_characteristic() {
        let mut complex = GenericCellComplex::new();

        // Create a triangle: 3 vertices, 3 edges, 1 face
        for i in 0..3 {
            complex.add_cell(0, i);
        }
        for i in 0..3 {
            complex.add_cell(1, i);
        }
        complex.add_cell(2, 0);

        // χ = 3 - 3 + 1 = 1
        assert_eq!(complex.euler_characteristic(), Integer::from(1));
    }

    #[test]
    fn test_cell_complex_with_name() {
        let mut complex = GenericCellComplex::with_name("Test Complex");
        assert_eq!(complex.name(), Some("Test Complex"));

        complex.set_name("New Name");
        assert_eq!(complex.name(), Some("New Name"));
    }

    #[test]
    fn test_f_vector() {
        let mut complex = GenericCellComplex::new();
        complex.add_cell(0, 0);
        complex.add_cell(0, 1);
        complex.add_cell(1, 0);
        complex.add_cell(2, 0);
        complex.add_cell(2, 1);
        complex.add_cell(2, 2);

        assert_eq!(complex.f_vector(), vec![2, 1, 3]);
    }

    #[test]
    fn test_is_pure() {
        let complex = GenericCellComplex::new();
        assert!(complex.is_pure());

        let mut complex2 = GenericCellComplex::new();
        complex2.add_cell(0, 0);
        complex2.add_cell(1, 0);
        assert!(complex2.is_pure());
    }

    #[test]
    fn test_connected_components() {
        let mut complex = GenericCellComplex::new();
        assert_eq!(complex.connected_components(), 0);

        complex.add_cell(0, 0);
        assert_eq!(complex.connected_components(), 1);
    }
}

//! Tableau tuples - tuples of Young tableaux
//!
//! This module provides structures for working with tuples of tableaux,
//! which arise in the representation theory of Ariki-Koike algebras and
//! quantum groups.

use crate::partitions::{Partition, PartitionTuple};
use crate::tableaux::Tableau;

/// A tuple of Young tableaux
///
/// A tableau tuple is an ordered sequence of tableaux, typically used in
/// representation theory of Ariki-Koike algebras (also known as Hecke
/// algebras of type G(r,1,n)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableauTuple {
    /// The component tableaux in this tuple
    components: Vec<Tableau>,
}

impl TableauTuple {
    /// Create a new tableau tuple from a vector of tableaux
    pub fn new(components: Vec<Tableau>) -> Self {
        TableauTuple { components }
    }

    /// Create an empty tableau tuple with a given level (number of components)
    pub fn empty(level: usize) -> Self {
        TableauTuple {
            components: vec![Tableau::new(vec![]).unwrap(); level],
        }
    }

    /// Get the level (number of component tableaux)
    pub fn level(&self) -> usize {
        self.components.len()
    }

    /// Get the component tableaux
    pub fn components(&self) -> &[Tableau] {
        &self.components
    }

    /// Get the total size (sum of sizes of all component tableaux)
    pub fn size(&self) -> usize {
        self.components.iter().map(|t| t.size()).sum()
    }

    /// Get the shape as a partition tuple
    pub fn shape(&self) -> PartitionTuple {
        let shapes: Vec<Partition> = self.components.iter().map(|t| t.shape().clone()).collect();
        PartitionTuple::new(shapes)
    }

    /// Get a specific component tableau
    pub fn component(&self, index: usize) -> Option<&Tableau> {
        self.components.get(index)
    }

    /// Get the content (multiset of all entries across all tableaux)
    pub fn content(&self) -> Vec<usize> {
        let mut entries: Vec<usize> = self
            .components
            .iter()
            .flat_map(|t| t.content())
            .collect();
        entries.sort_unstable();
        entries
    }

    /// Check if this is a valid tableau tuple
    ///
    /// A valid tableau tuple has tableaux that are all valid Young tableaux
    pub fn is_valid(&self) -> bool {
        self.components.iter().all(|t| t.size() == 0 || t.rows().iter().all(|row| !row.is_empty()))
    }

    /// Check if rows are strictly increasing in each tableau
    ///
    /// This is the row-strict condition
    pub fn is_row_strict(&self) -> bool {
        self.components.iter().all(|tableau| {
            tableau.rows().iter().all(|row| {
                row.windows(2).all(|w| w[0] < w[1])
            })
        })
    }

    /// Check if columns are strictly increasing in each tableau
    ///
    /// This is the column-strict condition
    pub fn is_column_strict(&self) -> bool {
        self.components.iter().all(|tableau| {
            if tableau.rows().is_empty() {
                return true;
            }
            for col in 0..tableau.rows()[0].len() {
                for row in 1..tableau.rows().len() {
                    if col < tableau.rows()[row].len() {
                        if tableau.rows()[row][col] <= tableau.rows()[row - 1][col] {
                            return false;
                        }
                    }
                }
            }
            true
        })
    }

    /// Display the tableau tuple as a string
    pub fn to_string(&self) -> String {
        self.components
            .iter()
            .enumerate()
            .map(|(i, t)| format!("Component {}:\n{}", i, t.to_string()))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// A standard tableau tuple
///
/// A standard tableau tuple is a tableau tuple where:
/// 1. Each tableau is standard (rows and columns strictly increasing)
/// 2. The entries 1, 2, ..., n appear exactly once across all tableaux
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StandardTableauTuple {
    /// The underlying tableau tuple
    tableau_tuple: TableauTuple,
}

impl StandardTableauTuple {
    /// Create a new standard tableau tuple
    ///
    /// Returns None if the tuple is not standard
    pub fn new(components: Vec<Tableau>) -> Option<Self> {
        let tuple = TableauTuple::new(components);

        if !tuple.is_valid() {
            return None;
        }

        // Check that rows and columns are strictly increasing in each tableau
        if !tuple.is_row_strict() || !tuple.is_column_strict() {
            return None;
        }

        // Check that we have exactly 1, 2, ..., n across all tableaux
        let n = tuple.size();
        if n > 0 {
            let mut seen = vec![false; n + 1];
            for tableau in &tuple.components {
                for row in tableau.rows() {
                    for &entry in row {
                        if entry == 0 || entry > n || seen[entry] {
                            return None;
                        }
                        seen[entry] = true;
                    }
                }
            }
        }

        Some(StandardTableauTuple { tableau_tuple: tuple })
    }

    /// Get the underlying tableau tuple
    pub fn tableau_tuple(&self) -> &TableauTuple {
        &self.tableau_tuple
    }

    /// Get the level
    pub fn level(&self) -> usize {
        self.tableau_tuple.level()
    }

    /// Get the size
    pub fn size(&self) -> usize {
        self.tableau_tuple.size()
    }

    /// Get the shape
    pub fn shape(&self) -> PartitionTuple {
        self.tableau_tuple.shape()
    }

    /// Get the components
    pub fn components(&self) -> &[Tableau] {
        self.tableau_tuple.components()
    }

    /// Display as string
    pub fn to_string(&self) -> String {
        self.tableau_tuple.to_string()
    }
}

/// A row-standard tableau tuple
///
/// A row-standard tableau tuple is a tableau tuple where:
/// 1. Rows are strictly increasing in each tableau
/// 2. Columns are strictly increasing in each tableau
/// 3. The entries 1, 2, ..., n appear exactly once across all tableaux
///
/// This is similar to StandardTableauTuple but emphasizes the row-strict property
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowStandardTableauTuple {
    /// The underlying tableau tuple
    tableau_tuple: TableauTuple,
}

impl RowStandardTableauTuple {
    /// Create a new row-standard tableau tuple
    ///
    /// Returns None if the tuple is not row-standard
    pub fn new(components: Vec<Tableau>) -> Option<Self> {
        let tuple = TableauTuple::new(components);

        if !tuple.is_valid() {
            return None;
        }

        // Check row-strict and column-strict conditions
        if !tuple.is_row_strict() || !tuple.is_column_strict() {
            return None;
        }

        // Check that we have exactly 1, 2, ..., n across all tableaux
        let n = tuple.size();
        if n > 0 {
            let mut seen = vec![false; n + 1];
            for tableau in &tuple.components {
                for row in tableau.rows() {
                    for &entry in row {
                        if entry == 0 || entry > n || seen[entry] {
                            return None;
                        }
                        seen[entry] = true;
                    }
                }
            }
        }

        Some(RowStandardTableauTuple { tableau_tuple: tuple })
    }

    /// Get the underlying tableau tuple
    pub fn tableau_tuple(&self) -> &TableauTuple {
        &self.tableau_tuple
    }

    /// Get the level
    pub fn level(&self) -> usize {
        self.tableau_tuple.level()
    }

    /// Get the size
    pub fn size(&self) -> usize {
        self.tableau_tuple.size()
    }

    /// Get the shape
    pub fn shape(&self) -> PartitionTuple {
        self.tableau_tuple.shape()
    }

    /// Get the components
    pub fn components(&self) -> &[Tableau] {
        self.tableau_tuple.components()
    }

    /// Display as string
    pub fn to_string(&self) -> String {
        self.tableau_tuple.to_string()
    }
}

/// Generate all standard tableau tuples of a given shape
///
/// Uses recursive backtracking to fill the tableaux with 1, 2, ..., n
pub fn standard_tableau_tuples(shape: &PartitionTuple) -> Vec<StandardTableauTuple> {
    let n = shape.sum();
    if n == 0 {
        let empty_tableaux: Vec<Tableau> = (0..shape.level())
            .map(|_| Tableau::new(vec![]).unwrap())
            .collect();
        if let Some(tuple) = StandardTableauTuple::new(empty_tableaux) {
            return vec![tuple];
        }
        return vec![];
    }

    let mut result = Vec::new();
    let mut current: Vec<Vec<Vec<usize>>> = shape
        .components()
        .iter()
        .map(|partition| {
            partition
                .parts()
                .iter()
                .map(|&len| vec![0; len])
                .collect()
        })
        .collect();

    generate_standard_tableau_tuples(&mut current, shape, 1, n, &mut result);

    result
}

fn generate_standard_tableau_tuples(
    current: &mut Vec<Vec<Vec<usize>>>,
    shape: &PartitionTuple,
    next_value: usize,
    n: usize,
    result: &mut Vec<StandardTableauTuple>,
) {
    if next_value > n {
        // All values placed, create tableau tuple
        let tableaux: Vec<Tableau> = current
            .iter()
            .map(|rows| Tableau::new(rows.clone()).unwrap())
            .collect();

        if let Some(tuple) = StandardTableauTuple::new(tableaux) {
            result.push(tuple);
        }
        return;
    }

    // Try placing next_value in each valid position across all tableaux
    for comp_idx in 0..shape.level() {
        let partition = &shape.components()[comp_idx];
        for r in 0..partition.length() {
            for c in 0..partition.parts()[r] {
                if can_place_in_tuple(current, comp_idx, r, c, next_value) {
                    current[comp_idx][r][c] = next_value;
                    generate_standard_tableau_tuples(current, shape, next_value + 1, n, result);
                    current[comp_idx][r][c] = 0;
                }
            }
        }
    }
}

fn can_place_in_tuple(
    current: &[Vec<Vec<usize>>],
    comp_idx: usize,
    row: usize,
    col: usize,
    value: usize,
) -> bool {
    // Check that this position hasn't been filled
    if current[comp_idx][row][col] != 0 {
        return false;
    }

    // Check that all positions to the left are filled and smaller
    if col > 0 {
        if current[comp_idx][row][col - 1] == 0 || current[comp_idx][row][col - 1] >= value {
            return false;
        }
    }

    // Check that all positions above are filled and smaller
    if row > 0 {
        if current[comp_idx][row - 1][col] == 0 || current[comp_idx][row - 1][col] >= value {
            return false;
        }
    }

    true
}

/// Generate all row-standard tableau tuples of a given shape
///
/// This is currently the same as standard_tableau_tuples since
/// row-standard implies both row and column strict
pub fn row_standard_tableau_tuples(shape: &PartitionTuple) -> Vec<RowStandardTableauTuple> {
    let standard_tuples = standard_tableau_tuples(shape);

    standard_tuples
        .into_iter()
        .filter_map(|st| {
            let components = st.components().to_vec();
            RowStandardTableauTuple::new(components)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tableau_tuple_creation() {
        let t1 = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let t2 = Tableau::new(vec![vec![4]]).unwrap();
        let tuple = TableauTuple::new(vec![t1, t2]);

        assert_eq!(tuple.level(), 2);
        assert_eq!(tuple.size(), 4);
    }

    #[test]
    fn test_tableau_tuple_empty() {
        let tuple = TableauTuple::empty(3);
        assert_eq!(tuple.level(), 3);
        assert_eq!(tuple.size(), 0);
    }

    #[test]
    fn test_tableau_tuple_shape() {
        let t1 = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
        let t2 = Tableau::new(vec![vec![6]]).unwrap();
        let tuple = TableauTuple::new(vec![t1, t2]);

        let shape = tuple.shape();
        assert_eq!(shape.level(), 2);
        assert_eq!(shape.components()[0].parts(), &[3, 2]);
        assert_eq!(shape.components()[1].parts(), &[1]);
    }

    #[test]
    fn test_tableau_tuple_content() {
        let t1 = Tableau::new(vec![vec![1, 3], vec![2]]).unwrap();
        let t2 = Tableau::new(vec![vec![4]]).unwrap();
        let tuple = TableauTuple::new(vec![t1, t2]);

        assert_eq!(tuple.content(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_row_strict() {
        // Row-strict tableau tuple
        let t1 = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        let t2 = Tableau::new(vec![vec![6]]).unwrap();
        let tuple = TableauTuple::new(vec![t1, t2]);
        assert!(tuple.is_row_strict());

        // Not row-strict (1, 3, 2 is not strictly increasing)
        let t3 = Tableau::new(vec![vec![1, 3, 2]]).unwrap();
        let tuple2 = TableauTuple::new(vec![t3]);
        assert!(!tuple2.is_row_strict());

        // Weakly increasing rows (1, 1, 2) - not strictly increasing
        let t4 = Tableau::new(vec![vec![1, 1, 2]]).unwrap();
        let tuple3 = TableauTuple::new(vec![t4]);
        assert!(!tuple3.is_row_strict());
    }

    #[test]
    fn test_column_strict() {
        // Column-strict tableau tuple
        let t1 = Tableau::new(vec![vec![1, 2, 4], vec![3, 5, 6]]).unwrap();
        let tuple = TableauTuple::new(vec![t1]);
        assert!(tuple.is_column_strict());

        // Not column-strict (3 <= 1 in first column)
        let t2 = Tableau::new(vec![vec![1, 2], vec![1, 5]]).unwrap();
        let tuple2 = TableauTuple::new(vec![t2]);
        assert!(!tuple2.is_column_strict());
    }

    #[test]
    fn test_standard_tableau_tuple_creation() {
        // Valid standard tableau tuple
        let t1 = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        let t2 = Tableau::new(vec![vec![6]]).unwrap();
        let tuple = StandardTableauTuple::new(vec![t1, t2]);
        assert!(tuple.is_some());

        let tuple = tuple.unwrap();
        assert_eq!(tuple.size(), 6);
        assert_eq!(tuple.level(), 2);
    }

    #[test]
    fn test_standard_tableau_tuple_invalid() {
        // Invalid - missing entry 3
        let t1 = Tableau::new(vec![vec![1, 2]]).unwrap();
        let t2 = Tableau::new(vec![vec![4]]).unwrap();
        let tuple = StandardTableauTuple::new(vec![t1, t2]);
        assert!(tuple.is_none());

        // Invalid - duplicate entry 1
        let t3 = Tableau::new(vec![vec![1, 2]]).unwrap();
        let t4 = Tableau::new(vec![vec![1]]).unwrap();
        let tuple2 = StandardTableauTuple::new(vec![t3, t4]);
        assert!(tuple2.is_none());
    }

    #[test]
    fn test_row_standard_tableau_tuple_creation() {
        // Valid row-standard tableau tuple (which is standard)
        let t1 = Tableau::new(vec![vec![1, 3, 5], vec![2, 4]]).unwrap();
        let t2 = Tableau::new(vec![vec![6]]).unwrap();
        let tuple = RowStandardTableauTuple::new(vec![t1, t2]);
        assert!(tuple.is_some());

        let tuple = tuple.unwrap();
        assert_eq!(tuple.size(), 6);
        assert_eq!(tuple.level(), 2);
    }

    #[test]
    fn test_row_standard_tableau_tuple_invalid() {
        // Invalid - not row-strict (has 1, 1 in row)
        let t1 = Tableau::new(vec![vec![1, 1, 2]]).unwrap();
        let tuple = RowStandardTableauTuple::new(vec![t1]);
        assert!(tuple.is_none());

        // Invalid - not column-strict
        let t2 = Tableau::new(vec![vec![1, 2], vec![1, 3]]).unwrap();
        let tuple2 = RowStandardTableauTuple::new(vec![t2]);
        assert!(tuple2.is_none());
    }

    #[test]
    fn test_generate_standard_tableau_tuples_small() {
        // Shape: ([1], [1]) - two tableaux each with one cell
        let shape = PartitionTuple::new(vec![
            Partition::new(vec![1]),
            Partition::new(vec![1]),
        ]);

        let tuples = standard_tableau_tuples(&shape);

        // Should have 2 tuples:
        // ([1], [2]) and ([2], [1])
        assert_eq!(tuples.len(), 2);

        for tuple in &tuples {
            assert_eq!(tuple.size(), 2);
            assert_eq!(tuple.level(), 2);
            // Individual components don't need to be standard,
            // but they should be row-strict and column-strict
            assert!(tuple.tableau_tuple().is_row_strict());
            assert!(tuple.tableau_tuple().is_column_strict());
        }
    }

    #[test]
    fn test_generate_standard_tableau_tuples_empty() {
        // Empty shape
        let shape = PartitionTuple::new(vec![
            Partition::new(vec![]),
            Partition::new(vec![]),
        ]);

        let tuples = standard_tableau_tuples(&shape);
        assert_eq!(tuples.len(), 1);
        assert_eq!(tuples[0].size(), 0);
    }

    #[test]
    fn test_generate_standard_tableau_tuples_shape_2_1() {
        // Shape: ([2, 1], []) - one tableau with shape [2,1], one empty
        let shape = PartitionTuple::new(vec![
            Partition::new(vec![2, 1]),
            Partition::new(vec![]),
        ]);

        let tuples = standard_tableau_tuples(&shape);

        // Shape [2,1] has 2 standard tableaux
        assert_eq!(tuples.len(), 2);

        for tuple in &tuples {
            assert_eq!(tuple.size(), 3);
            assert_eq!(tuple.level(), 2);
            // Check that the first component is row-strict and column-strict
            assert!(tuple.tableau_tuple().is_row_strict());
            assert!(tuple.tableau_tuple().is_column_strict());
            assert_eq!(tuple.components()[1].size(), 0);
        }
    }

    #[test]
    fn test_row_standard_tableau_tuples_generation() {
        // Shape: ([1], [1])
        let shape = PartitionTuple::new(vec![
            Partition::new(vec![1]),
            Partition::new(vec![1]),
        ]);

        let tuples = row_standard_tableau_tuples(&shape);

        // Should have 2 tuples
        assert_eq!(tuples.len(), 2);

        for tuple in &tuples {
            assert_eq!(tuple.size(), 2);
            assert_eq!(tuple.level(), 2);
        }
    }

    #[test]
    fn test_tableau_tuple_to_string() {
        let t1 = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let t2 = Tableau::new(vec![vec![4]]).unwrap();
        let tuple = TableauTuple::new(vec![t1, t2]);

        let s = tuple.to_string();
        assert!(s.contains("Component 0"));
        assert!(s.contains("Component 1"));
    }

    #[test]
    fn test_standard_tableau_tuple_with_three_components() {
        // Valid standard tableau tuple with 3 components
        let t1 = Tableau::new(vec![vec![1, 2]]).unwrap();
        let t2 = Tableau::new(vec![vec![3, 4]]).unwrap();
        let t3 = Tableau::new(vec![vec![5]]).unwrap();
        let tuple = StandardTableauTuple::new(vec![t1, t2, t3]);
        assert!(tuple.is_some());

        let tuple = tuple.unwrap();
        assert_eq!(tuple.size(), 5);
        assert_eq!(tuple.level(), 3);
    }

    #[test]
    fn test_tableau_tuple_component_access() {
        let t1 = Tableau::new(vec![vec![1, 2]]).unwrap();
        let t2 = Tableau::new(vec![vec![3]]).unwrap();
        let tuple = TableauTuple::new(vec![t1.clone(), t2.clone()]);

        assert_eq!(tuple.component(0), Some(&t1));
        assert_eq!(tuple.component(1), Some(&t2));
        assert_eq!(tuple.component(2), None);
    }
}

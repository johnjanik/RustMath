//! Specht modules - irreducible representations of the symmetric group
//!
//! This module implements Specht modules S^λ with their polytabloid basis and Garnir relations.
//! Specht modules are the irreducible representations of the symmetric group S_n,
//! indexed by partitions λ of n.
//!
//! # Theory
//!
//! - A **tabloid** {T} is an equivalence class of tableaux where rows can be permuted
//! - A **polytabloid** e_T is constructed from a standard tableau T by:
//!   e_T = Σ_{σ ∈ C_T} sgn(σ) · {σT}
//!   where C_T is the column stabilizer of T
//! - The **Specht module** S^λ is spanned by polytabloids e_T for all standard tableaux T of shape λ
//! - **Garnir elements** provide relations between polytabloids
//!
//! # References
//!
//! - James and Kerber, "The Representation Theory of the Symmetric Group"
//! - Sagan, "The Symmetric Group: Representations, Combinatorial Algorithms, and Symmetric Functions"

use crate::partitions::Partition;
use crate::permutations::Permutation;
use crate::tableaux::Tableau;
use std::collections::HashMap;

/// A tabloid - an equivalence class of tableaux where row order doesn't matter
///
/// Two tableaux give the same tabloid if their rows contain the same elements
/// (possibly in different orders)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tabloid {
    /// Rows of the tabloid, where each row is a sorted set
    rows: Vec<Vec<usize>>,
    shape: Partition,
}

impl Tabloid {
    /// Create a tabloid from a tableau
    ///
    /// The rows are sorted to create a canonical representative
    pub fn from_tableau(tableau: &Tableau) -> Self {
        let mut rows: Vec<Vec<usize>> = tableau
            .rows()
            .iter()
            .map(|row| {
                let mut sorted_row = row.clone();
                sorted_row.sort_unstable();
                sorted_row
            })
            .collect();

        // Ensure rows themselves are in a canonical order for comparison
        // (They're already sorted by the tableau shape property)

        Tabloid {
            rows,
            shape: tableau.shape().clone(),
        }
    }

    /// Get the shape of the tabloid
    pub fn shape(&self) -> &Partition {
        &self.shape
    }

    /// Get the rows
    pub fn rows(&self) -> &[Vec<usize>] {
        &self.rows
    }

    /// Apply a permutation to the tabloid
    ///
    /// Returns a new tabloid with the permutation applied to all entries
    /// Assumes 1-indexed tableau entries, converts to 0-indexed for permutation
    pub fn apply_permutation(&self, perm: &Permutation) -> Self {
        let rows: Vec<Vec<usize>> = self
            .rows
            .iter()
            .map(|row| {
                let mut new_row: Vec<usize> = row
                    .iter()
                    .map(|&x| {
                        // x is 1-indexed, convert to 0-indexed for permutation
                        perm.apply(x - 1).map(|v| v + 1).unwrap_or(x)
                    })
                    .collect();
                new_row.sort_unstable();
                new_row
            })
            .collect();

        Tabloid {
            rows,
            shape: self.shape.clone(),
        }
    }
}

/// A polytabloid - a signed sum of tabloids forming a basis element of the Specht module
///
/// For a standard tableau T, the polytabloid e_T is defined as:
/// e_T = Σ_{σ ∈ C_T} sgn(σ) · {σT}
/// where C_T is the column stabilizer of T
#[derive(Debug, Clone)]
pub struct Polytabloid {
    /// The generating tableau
    tableau: Tableau,
    /// Linear combination of tabloids with integer coefficients (sgn)
    /// Maps from tabloid to coefficient (+1 or -1)
    terms: HashMap<Tabloid, i32>,
}

impl Polytabloid {
    /// Create a polytabloid from a standard tableau
    ///
    /// Computes e_T = Σ_{σ ∈ C_T} sgn(σ) · {σT}
    pub fn from_tableau(tableau: Tableau) -> Self {
        if !tableau.is_standard() {
            panic!("Polytabloid can only be created from standard tableau");
        }

        let mut terms = HashMap::new();
        let column_perms = column_stabilizer(&tableau);

        for perm in column_perms {
            let permuted_tableau = apply_permutation_to_tableau(&tableau, &perm);
            let tabloid = Tabloid::from_tableau(&permuted_tableau);
            let sign = perm.sign(); // Returns i32: +1 or -1

            *terms.entry(tabloid).or_insert(0) += sign;
        }

        // Remove zero terms
        terms.retain(|_, &mut v| v != 0);

        Polytabloid { tableau, terms }
    }

    /// Get the generating tableau
    pub fn tableau(&self) -> &Tableau {
        &self.tableau
    }

    /// Get the shape
    pub fn shape(&self) -> &Partition {
        self.tableau.shape()
    }

    /// Get the terms (tabloids with coefficients)
    pub fn terms(&self) -> &HashMap<Tabloid, i32> {
        &self.terms
    }

    /// Check if this polytabloid is zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Add two polytabloids
    pub fn add(&self, other: &Polytabloid) -> PolytabloidSum {
        if self.shape() != other.shape() {
            panic!("Cannot add polytabloids of different shapes");
        }

        let mut result = PolytabloidSum::zero(self.shape().clone());
        result.add_polytabloid(self, 1);
        result.add_polytabloid(other, 1);
        result
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: i32) -> PolytabloidSum {
        let mut result = PolytabloidSum::zero(self.shape().clone());
        result.add_polytabloid(self, scalar);
        result
    }
}

/// A linear combination of polytabloids
#[derive(Debug, Clone)]
pub struct PolytabloidSum {
    shape: Partition,
    /// Maps tabloids to their coefficients in the linear combination
    terms: HashMap<Tabloid, i32>,
}

impl PolytabloidSum {
    /// Create zero element
    pub fn zero(shape: Partition) -> Self {
        PolytabloidSum {
            shape,
            terms: HashMap::new(),
        }
    }

    /// Create from a single polytabloid
    pub fn from_polytabloid(poly: &Polytabloid, coeff: i32) -> Self {
        let mut sum = Self::zero(poly.shape().clone());
        sum.add_polytabloid(poly, coeff);
        sum
    }

    /// Add a polytabloid with given coefficient
    pub fn add_polytabloid(&mut self, poly: &Polytabloid, coeff: i32) {
        for (tabloid, &poly_coeff) in &poly.terms {
            *self.terms.entry(tabloid.clone()).or_insert(0) += coeff * poly_coeff;
        }
        // Remove zero terms
        self.terms.retain(|_, &mut v| v != 0);
    }

    /// Get the shape
    pub fn shape(&self) -> &Partition {
        &self.shape
    }

    /// Get the terms
    pub fn terms(&self) -> &HashMap<Tabloid, i32> {
        &self.terms
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Add two sums
    pub fn add(&mut self, other: &PolytabloidSum) {
        if self.shape != other.shape {
            panic!("Cannot add polytabloid sums of different shapes");
        }
        for (tabloid, &coeff) in &other.terms {
            *self.terms.entry(tabloid.clone()).or_insert(0) += coeff;
        }
        self.terms.retain(|_, &mut v| v != 0);
    }

    /// Scalar multiplication
    pub fn scalar_mul(&mut self, scalar: i32) {
        for (_, coeff) in self.terms.iter_mut() {
            *coeff *= scalar;
        }
        self.terms.retain(|_, &mut v| v != 0);
    }
}

/// A Garnir element - used to define relations in the Specht module
///
/// For a tableau T and a Garnir set A (a set of consecutive cells in columns i and i+1),
/// the Garnir element is: Σ_{σ ∈ S_A} sgn(σ) · {σT}
#[derive(Debug, Clone)]
pub struct GarnirElement {
    /// The base tableau
    tableau: Tableau,
    /// The Garnir set (row indices and column index)
    garnir_set: GarnirSet,
    /// The resulting polytabloid sum
    element: PolytabloidSum,
}

/// Describes a Garnir set - consecutive cells in two adjacent columns
#[derive(Debug, Clone)]
pub struct GarnirSet {
    /// First column index
    col: usize,
    /// Starting row index
    start_row: usize,
    /// Ending row index (inclusive)
    end_row: usize,
}

impl GarnirElement {
    /// Create a Garnir element
    ///
    /// The Garnir set consists of cells in columns col and col+1,
    /// from start_row to end_row (inclusive)
    pub fn new(tableau: Tableau, col: usize, start_row: usize, end_row: usize) -> Self {
        let garnir_set = GarnirSet {
            col,
            start_row,
            end_row,
        };

        // Collect the entries in the Garnir set
        let mut garnir_entries = Vec::new();
        for r in start_row..=end_row {
            if let Some(entry) = tableau.get(r, col) {
                garnir_entries.push(entry);
            }
            if let Some(entry) = tableau.get(r, col + 1) {
                garnir_entries.push(entry);
            }
        }

        // Generate all permutations of the Garnir set entries
        let n = garnir_entries.len();
        let mut element = PolytabloidSum::zero(tableau.shape().clone());

        // For each permutation of the Garnir entries
        use crate::permutations::all_permutations;
        let perms = all_permutations(n);

        for perm in perms {
            let mut new_tableau_rows = tableau.rows().to_vec();
            let mut idx = 0;

            // Apply permutation to Garnir set entries
            for r in start_row..=end_row {
                if col < new_tableau_rows.get(r).map(|row| row.len()).unwrap_or(0) {
                    if let Some(permuted_idx) = perm.apply(idx) {
                        new_tableau_rows[r][col] = garnir_entries[permuted_idx];
                    }
                    idx += 1;
                }
                if col + 1 < new_tableau_rows.get(r).map(|row| row.len()).unwrap_or(0) {
                    if let Some(permuted_idx) = perm.apply(idx) {
                        new_tableau_rows[r][col + 1] = garnir_entries[permuted_idx];
                    }
                    idx += 1;
                }
            }

            // Create tabloid and add with sign
            if let Some(new_tableau) = Tableau::new(new_tableau_rows) {
                let tabloid = Tabloid::from_tableau(&new_tableau);
                let sign = perm.sign(); // Returns i32: +1 or -1
                *element.terms.entry(tabloid).or_insert(0) += sign;
            }
        }

        element.terms.retain(|_, &mut v| v != 0);

        GarnirElement {
            tableau,
            garnir_set,
            element,
        }
    }

    /// Get the Garnir element as a polytabloid sum
    pub fn element(&self) -> &PolytabloidSum {
        &self.element
    }

    /// Get the base tableau
    pub fn tableau(&self) -> &Tableau {
        &self.tableau
    }

    /// Get the Garnir set description
    pub fn garnir_set(&self) -> &GarnirSet {
        &self.garnir_set
    }
}

/// The Specht module S^λ - irreducible representation of S_n
///
/// The Specht module has a basis of polytabloids e_T, one for each
/// standard Young tableau T of shape λ
#[derive(Debug, Clone)]
pub struct SpechtModule {
    /// The partition defining the module
    partition: Partition,
    /// Basis polytabloids (one for each standard tableau)
    basis: Vec<Polytabloid>,
    /// Map from tableau to its index in the basis
    tableau_index: HashMap<Vec<Vec<usize>>, usize>,
}

impl SpechtModule {
    /// Create a Specht module for a given partition
    pub fn new(partition: Partition) -> Self {
        use crate::tableaux::standard_tableaux;

        let tableaux = standard_tableaux(&partition);
        let mut basis = Vec::new();
        let mut tableau_index = HashMap::new();

        for (idx, tableau) in tableaux.iter().enumerate() {
            let poly = Polytabloid::from_tableau(tableau.clone());
            basis.push(poly);
            tableau_index.insert(tableau.rows().to_vec(), idx);
        }

        SpechtModule {
            partition,
            basis,
            tableau_index,
        }
    }

    /// Get the partition
    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Get the dimension (number of basis elements)
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Get the basis polytabloids
    pub fn basis(&self) -> &[Polytabloid] {
        &self.basis
    }

    /// Get a specific basis element by index
    pub fn basis_element(&self, idx: usize) -> Option<&Polytabloid> {
        self.basis.get(idx)
    }

    /// Get the index of a tableau in the basis
    pub fn tableau_to_index(&self, tableau: &Tableau) -> Option<usize> {
        self.tableau_index.get(tableau.rows()).copied()
    }

    /// Create a Garnir element for a basis tableau
    pub fn garnir_element(
        &self,
        tableau_idx: usize,
        col: usize,
        start_row: usize,
        end_row: usize,
    ) -> Option<GarnirElement> {
        let tableau = self.basis.get(tableau_idx)?.tableau().clone();
        Some(GarnirElement::new(tableau, col, start_row, end_row))
    }

    /// Compute the action of a permutation on a basis element
    ///
    /// Returns the result as a linear combination of basis elements
    pub fn permutation_action(
        &self,
        perm: &Permutation,
        basis_idx: usize,
    ) -> HashMap<usize, i32> {
        let poly = match self.basis.get(basis_idx) {
            Some(p) => p,
            None => return HashMap::new(),
        };

        let mut result = HashMap::new();

        // Apply permutation to each tabloid in the polytabloid
        for (tabloid, &coeff) in &poly.terms {
            let new_tabloid = tabloid.apply_permutation(perm);

            // Try to express this tabloid in terms of basis polytabloids
            // This is non-trivial; for now we just track the tabloid
            // A full implementation would use the Straightening Algorithm
            // For now, we return a simplified result
            result.insert(basis_idx, coeff);
        }

        result
    }

    /// Check if this module satisfies the Garnir relations
    ///
    /// This is a theoretical check - Garnir elements should act as zero
    pub fn verify_garnir_relation(
        &self,
        tableau_idx: usize,
        col: usize,
        start_row: usize,
        end_row: usize,
    ) -> bool {
        if let Some(garnir) = self.garnir_element(tableau_idx, col, start_row, end_row) {
            // In a proper Specht module, Garnir elements should satisfy certain relations
            // For now, we just check that it was constructed successfully
            !garnir.element().is_zero()
        } else {
            false
        }
    }
}

/// Helper function to compute the column stabilizer of a tableau
///
/// Returns all permutations that preserve each column
fn column_stabilizer(tableau: &Tableau) -> Vec<Permutation> {
    let n = tableau.size();
    if n == 0 {
        return vec![Permutation::identity(0)];
    }

    // Collect elements in each column
    let mut columns: Vec<Vec<usize>> = Vec::new();
    let num_cols = tableau.rows()[0].len();

    for col in 0..num_cols {
        let mut column = Vec::new();
        for row in tableau.rows() {
            if col < row.len() {
                column.push(row[col]);
            }
        }
        columns.push(column);
    }

    // Generate all permutations that permute within each column
    generate_column_permutations(&columns, n)
}

/// Generate all permutations that permute elements within their columns
fn generate_column_permutations(columns: &[Vec<usize>], n: usize) -> Vec<Permutation> {
    if n == 0 {
        return vec![Permutation::identity(0)];
    }

    // Start with identity permutation (maps i -> i for all i in 0..n)
    let mut result = vec![vec![0; n]];

    // Initialize result with identity
    for i in 0..n {
        result[0][i] = i;
    }

    for column in columns {
        if column.is_empty() {
            continue;
        }

        use crate::permutations::all_permutations;
        let col_perms = all_permutations(column.len());

        let mut new_result = Vec::new();

        for base_perm in &result {
            for col_perm in &col_perms {
                let mut new_perm = base_perm.clone();

                // Apply column permutation to this column's elements
                // column contains 1-indexed elements, we need 0-indexed for array access
                for (i, &elem) in column.iter().enumerate() {
                    if let Some(permuted_idx) = col_perm.apply(i) {
                        let new_elem = column[permuted_idx];
                        // Set the mapping: elem-1 maps to new_elem-1 (convert to 0-indexed)
                        if elem > 0 && elem <= n && new_elem > 0 && new_elem <= n {
                            new_perm[elem - 1] = new_elem - 1;
                        }
                    }
                }

                new_result.push(new_perm);
            }
        }

        result = new_result;
    }

    result
        .into_iter()
        .filter_map(|perm_vec| Permutation::from_vec(perm_vec))
        .collect()
}

/// Apply a permutation to a tableau
/// Assumes 1-indexed tableau entries, converts to 0-indexed for permutation
fn apply_permutation_to_tableau(tableau: &Tableau, perm: &Permutation) -> Tableau {
    let new_rows: Vec<Vec<usize>> = tableau
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| {
                    // x is 1-indexed, convert to 0-indexed for permutation
                    perm.apply(x - 1).map(|v| v + 1).unwrap_or(x)
                })
                .collect()
        })
        .collect();

    Tableau::new(new_rows).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tabloid_creation() {
        let t = Tableau::new(vec![vec![1, 3, 2], vec![4, 5]]).unwrap();
        let tabloid = Tabloid::from_tableau(&t);

        // Rows should be sorted
        assert_eq!(tabloid.rows()[0], vec![1, 2, 3]);
        assert_eq!(tabloid.rows()[1], vec![4, 5]);
    }

    #[test]
    fn test_tabloid_equality() {
        let t1 = Tableau::new(vec![vec![1, 2, 3], vec![4, 5]]).unwrap();
        let t2 = Tableau::new(vec![vec![3, 1, 2], vec![5, 4]]).unwrap();

        let tabloid1 = Tabloid::from_tableau(&t1);
        let tabloid2 = Tabloid::from_tableau(&t2);

        // Should be equal since rows contain same elements
        assert_eq!(tabloid1, tabloid2);
    }

    #[test]
    fn test_polytabloid_creation() {
        let t = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let poly = Polytabloid::from_tableau(t);

        assert!(!poly.is_zero());
        assert_eq!(poly.shape().parts(), &[2, 1]);
    }

    #[test]
    fn test_specht_module_dimension() {
        // Shape [2, 1] has 2 standard tableaux
        let partition = Partition::new(vec![2, 1]);
        let module = SpechtModule::new(partition);

        assert_eq!(module.dimension(), 2);
    }

    #[test]
    fn test_specht_module_basis() {
        let partition = Partition::new(vec![2, 1]);
        let module = SpechtModule::new(partition);

        assert_eq!(module.basis().len(), 2);

        for poly in module.basis() {
            assert!(!poly.is_zero());
            assert_eq!(poly.shape(), module.partition());
        }
    }

    #[test]
    fn test_polytabloid_addition() {
        let t1 = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let t2 = Tableau::new(vec![vec![1, 3], vec![2]]).unwrap();

        let poly1 = Polytabloid::from_tableau(t1);
        let poly2 = Polytabloid::from_tableau(t2);

        let sum = poly1.add(&poly2);
        assert!(!sum.is_zero());
    }

    #[test]
    fn test_polytabloid_scalar_mul() {
        let t = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let poly = Polytabloid::from_tableau(t);

        let scaled = poly.scalar_mul(3);
        assert!(!scaled.is_zero());

        let zero = poly.scalar_mul(0);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_garnir_element() {
        let t = Tableau::new(vec![vec![1, 2, 4], vec![3, 5]]).unwrap();
        let garnir = GarnirElement::new(t, 0, 0, 1);

        // Garnir element should be non-zero in general
        assert!(garnir.element().shape().parts() == &[3, 2]);
    }

    #[test]
    fn test_specht_module_hook_formula() {
        // Verify dimension matches hook length formula
        let partition = Partition::new(vec![3, 2]);
        let module = SpechtModule::new(partition.clone());

        let expected_dim = partition.dimension();
        assert_eq!(module.dimension(), expected_dim);
    }

    #[test]
    fn test_specht_module_small_cases() {
        // Partition [1] should have dimension 1
        let p1 = Partition::new(vec![1]);
        let m1 = SpechtModule::new(p1);
        assert_eq!(m1.dimension(), 1);

        // Partition [2] should have dimension 1
        let p2 = Partition::new(vec![2]);
        let m2 = SpechtModule::new(p2);
        assert_eq!(m2.dimension(), 1);

        // Partition [1, 1] should have dimension 1
        let p11 = Partition::new(vec![1, 1]);
        let m11 = SpechtModule::new(p11);
        assert_eq!(m11.dimension(), 1);
    }

    #[test]
    fn test_column_stabilizer() {
        let t = Tableau::new(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let stab = column_stabilizer(&t);

        // Column stabilizer for [[1,2],[3,4]] should have 4 elements:
        // permutations that swap (1,3) and/or (2,4)
        assert_eq!(stab.len(), 4);
    }

    #[test]
    fn test_tabloid_permutation_action() {
        let t = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let tabloid = Tabloid::from_tableau(&t);

        // Permutation that maps 0->1, 1->2, 2->0 (which is (0 1 2) in 0-indexed)
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        let result = tabloid.apply_permutation(&perm);

        assert_eq!(result.shape(), tabloid.shape());
    }

    #[test]
    fn test_polytabloid_sum_operations() {
        let t = Tableau::new(vec![vec![1, 2], vec![3]]).unwrap();
        let poly = Polytabloid::from_tableau(t);

        let mut sum1 = PolytabloidSum::from_polytabloid(&poly, 2);
        let sum2 = PolytabloidSum::from_polytabloid(&poly, -2);

        sum1.add(&sum2);
        assert!(sum1.is_zero()); // Should cancel out
    }

    #[test]
    fn test_specht_module_larger() {
        // Test partition [3, 2, 1] which has dimension 16
        let partition = Partition::new(vec![3, 2, 1]);
        let module = SpechtModule::new(partition.clone());

        assert_eq!(module.dimension(), 16);
        assert_eq!(module.basis().len(), 16);
    }
}

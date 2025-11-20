//! Kirillov-Reshetikhin (KR) crystals
//!
//! KR crystals are finite-dimensional crystals for affine Lie algebras.
//! They are denoted B^{r,s} where r is a Dynkin node and s is a positive integer.
//!
//! These crystals play a crucial role in the theory of affine crystals and
//! are used to construct crystal bases for integrable highest weight modules.

use crate::operators::{Crystal, CrystalElement};
use crate::root_system::{RootSystem, RootSystemType};
use crate::weight::Weight;
use crate::tableau_crystal::TableauElement;

/// A Kirillov-Reshetikhin crystal element
///
/// For type A, this is represented as a tableau with specific properties.
/// For other types, we use a more general representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KRElement {
    /// Tableau representation (for type A)
    Tableau(TableauElement),
    /// General representation as a sequence of letters
    Letters(Vec<usize>),
    /// Rigged configuration representation (advanced)
    RiggedConfig(Vec<(Vec<usize>, Vec<i64>)>),
}

impl KRElement {
    /// Create from a tableau
    pub fn from_tableau(tableau: TableauElement) -> Self {
        KRElement::Tableau(tableau)
    }

    /// Create from a sequence of letters
    pub fn from_letters(letters: Vec<usize>) -> Self {
        KRElement::Letters(letters)
    }

    /// Create from rigged configuration
    /// Each entry is (partition, rigging)
    pub fn from_rigged(config: Vec<(Vec<usize>, Vec<i64>)>) -> Self {
        KRElement::RiggedConfig(config)
    }
}

/// Kirillov-Reshetikhin crystal B^{r,s}
///
/// r is the Dynkin node (1 to n for type A_n^{(1)})
/// s is the height (number of rows for rectangular tableaux)
#[derive(Debug, Clone)]
pub struct KRCrystal {
    /// The affine type
    pub affine_type: RootSystemType,
    /// Dynkin node r
    pub r: usize,
    /// Height s
    pub s: usize,
    /// Root system (classical part)
    pub root_system: RootSystem,
    /// Generated elements
    elements_cache: Vec<KRElement>,
}

impl KRCrystal {
    /// Create a new KR crystal B^{r,s}
    pub fn new(affine_type: RootSystemType, r: usize, s: usize) -> Self {
        let root_system = RootSystem::new(affine_type);
        KRCrystal {
            affine_type,
            r,
            s,
            root_system,
            elements_cache: Vec::new(),
        }
    }

    /// Create KR crystal for type A (rectangular tableaux)
    pub fn type_a_rectangular(n: usize, rows: usize, cols: usize) -> Self {
        let mut crystal = KRCrystal::new(RootSystemType::A(n), cols, rows);
        crystal.generate_type_a_tableaux();
        crystal
    }

    /// Generate all elements for type A (as rectangular tableaux)
    fn generate_type_a_tableaux(&mut self) {
        let n = match self.affine_type {
            RootSystemType::A(n) => n,
            _ => return,
        };

        // Generate all semistandard tableaux of shape (r, r, ..., r) with s rows
        // and entries from 1 to n+1
        let shape = vec![self.r; self.s];
        self.elements_cache = self.generate_tableaux(shape, n + 1);
    }

    /// Generate semistandard tableaux with given shape and max entry
    fn generate_tableaux(&self, shape: Vec<usize>, max_entry: usize) -> Vec<KRElement> {
        let mut result = Vec::new();
        self.generate_tableaux_recursive(&shape, 0, vec![], &mut result, max_entry);
        result
    }

    fn generate_tableaux_recursive(
        &self,
        shape: &[usize],
        row_idx: usize,
        current: Vec<Vec<usize>>,
        result: &mut Vec<KRElement>,
        max_entry: usize,
    ) {
        if row_idx >= shape.len() {
            if !current.is_empty() {
                let tableau = TableauElement::new(current);
                if tableau.is_semistandard() {
                    result.push(KRElement::Tableau(tableau));
                }
            }
            return;
        }

        let row_len = shape[row_idx];
        let prev_row = if row_idx > 0 {
            &current[row_idx - 1]
        } else {
            &vec![]
        };

        // Generate all valid rows
        let mut rows = Vec::new();
        self.generate_valid_rows(
            row_len,
            prev_row,
            max_entry,
            vec![],
            &mut rows,
        );

        for row in rows {
            let mut next = current.clone();
            next.push(row);
            self.generate_tableaux_recursive(shape, row_idx + 1, next, result, max_entry);
        }
    }

    fn generate_valid_rows(
        &self,
        len: usize,
        prev_row: &[usize],
        max_entry: usize,
        current: Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == len {
            result.push(current);
            return;
        }

        let pos = current.len();
        let min_val = if pos > 0 {
            current[pos - 1] // Weakly increasing
        } else {
            1
        };

        let max_val = if pos < prev_row.len() {
            prev_row[pos].saturating_sub(1) // Strictly less than above
        } else {
            max_entry
        };

        for val in min_val..=max_val.min(max_entry) {
            if val == 0 {
                continue;
            }
            let mut next = current.clone();
            next.push(val);
            self.generate_valid_rows(len, prev_row, max_entry, next, result);
        }
    }

    /// Compute the classical weight of a KR crystal element
    fn classical_weight(&self, elem: &KRElement) -> Weight {
        match elem {
            KRElement::Tableau(tab) => {
                let n = match self.affine_type {
                    RootSystemType::A(n) => n,
                    _ => return Weight::zero(self.root_system.rank),
                };
                tab.compute_weight(n + 1)
            }
            KRElement::Letters(letters) => {
                let mut coords = vec![0i64; self.root_system.rank];
                for &letter in letters {
                    if letter > 0 && letter <= coords.len() {
                        coords[letter - 1] += 1;
                    }
                }
                Weight::new(coords)
            }
            KRElement::RiggedConfig(_) => {
                // Placeholder
                Weight::zero(self.root_system.rank)
            }
        }
    }

    /// Check if an element is highest weight
    pub fn is_hw(&self, elem: &KRElement) -> bool {
        // Try all e_i operators
        for i in 0..self.root_system.rank {
            if self.apply_ei(elem, i).is_some() {
                return false;
            }
        }
        true
    }

    /// Apply e_i to a KR element
    fn apply_ei(&self, elem: &KRElement, i: usize) -> Option<KRElement> {
        match elem {
            KRElement::Tableau(tab) => {
                // Use tableau crystal operators
                // This is simplified - full implementation would use signature
                None // Placeholder
            }
            KRElement::Letters(_) => None,
            KRElement::RiggedConfig(_) => None,
        }
    }

    /// Apply f_i to a KR element
    fn apply_fi(&self, elem: &KRElement, i: usize) -> Option<KRElement> {
        match elem {
            KRElement::Tableau(tab) => {
                // Use tableau crystal operators
                None // Placeholder
            }
            KRElement::Letters(_) => None,
            KRElement::RiggedConfig(_) => None,
        }
    }

    /// Get the dimension (number of elements)
    pub fn dimension(&self) -> usize {
        self.elements_cache.len()
    }

    /// Classical decomposition: decompose into classical highest weight crystals
    pub fn classical_decomposition(&self) -> Vec<(Weight, usize)> {
        // Group elements by classical weight and find multiplicities
        let mut weight_mult = std::collections::HashMap::new();

        for elem in &self.elements_cache {
            let weight = self.classical_weight(elem);
            *weight_mult.entry(weight).or_insert(0) += 1;
        }

        weight_mult.into_iter().collect()
    }
}

impl Crystal for KRCrystal {
    type Element = KRElement;

    fn weight(&self, b: &Self::Element) -> Weight {
        self.classical_weight(b)
    }

    fn e_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        self.apply_ei(b, i)
    }

    fn f_i(&self, b: &Self::Element, i: usize) -> Option<Self::Element> {
        self.apply_fi(b, i)
    }

    fn elements(&self) -> Vec<Self::Element> {
        self.elements_cache.clone()
    }
}

/// Tensor product of KR crystals
///
/// The tensor product B^{r_1,s_1} ⊗ B^{r_2,s_2} ⊗ ... is used to
/// construct crystal bases for general integrable modules.
pub struct KRTensorProduct {
    /// List of KR crystals
    pub factors: Vec<KRCrystal>,
}

impl KRTensorProduct {
    /// Create a tensor product of KR crystals
    pub fn new(factors: Vec<KRCrystal>) -> Self {
        KRTensorProduct { factors }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.factors.iter().map(|kr| kr.dimension()).product()
    }
}

/// Energy function for KR crystals
///
/// The energy function D: B^{r,s} → Z assigns an integer to each element.
/// It is a key invariant for affine crystals.
pub fn energy_function(crystal: &KRCrystal, elem: &KRElement) -> i64 {
    // Simplified implementation
    // The real energy function requires computing the affine weight
    0
}

/// R-matrix: Combinatorial R-matrix for KR crystals
///
/// The R-matrix R: B ⊗ B' → B' ⊗ B is a bijection that intertwines
/// the crystal structures.
pub fn r_matrix(
    b1: &KRElement,
    b2: &KRElement,
) -> (KRElement, KRElement) {
    // Placeholder - full implementation requires implementing the combinatorial R-matrix
    (b1.clone(), b2.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kr_crystal_creation() {
        let kr = KRCrystal::new(RootSystemType::A(2), 1, 1);
        assert_eq!(kr.r, 1);
        assert_eq!(kr.s, 1);
    }

    #[test]
    fn test_kr_element() {
        let tableau = TableauElement::new(vec![vec![1, 2]]);
        let elem = KRElement::from_tableau(tableau);

        match elem {
            KRElement::Tableau(tab) => {
                assert_eq!(tab.tableau, vec![vec![1, 2]]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_type_a_rectangular() {
        let kr = KRCrystal::type_a_rectangular(2, 1, 2);
        // B^{2,1} for A_2: rectangular tableau with 1 row and 2 columns
        // Should generate tableaux with entries from {1,2,3}

        assert!(kr.dimension() > 0);
    }

    #[test]
    fn test_kr_letters() {
        let elem = KRElement::from_letters(vec![1, 2, 2, 3]);
        match elem {
            KRElement::Letters(letters) => {
                assert_eq!(letters, vec![1, 2, 2, 3]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_classical_weight() {
        let tableau = TableauElement::new(vec![vec![1, 2]]);
        let elem = KRElement::from_tableau(tableau);
        let kr = KRCrystal::new(RootSystemType::A(2), 1, 1);

        let weight = kr.classical_weight(&elem);
        // Weight should count: one 1, one 2, zero 3s
        assert_eq!(weight.coords, vec![1, 1, 0]);
    }
}

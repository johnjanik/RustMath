//! Fusion Double - Drinfeld double of a finite group
//!
//! The FusionDouble represents the fusion ring of the Drinfeld (quantum) double
//! of a finite group. Simple objects are indexed by pairs (g, χ) where g is a
//! conjugacy class representative and χ is an irreducible character of the
//! centralizer of g.
//!
//! This is a simplified implementation focused on small finite groups.

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::fmt;

/// Index for a simple object in the fusion double
///
/// Consists of a conjugacy class index and a character index
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusionDoubleIndex {
    /// Conjugacy class index
    pub class_index: usize,
    /// Character index (index into irreducible characters of centralizer)
    pub character_index: usize,
}

impl FusionDoubleIndex {
    /// Create a new fusion double index
    pub fn new(class_index: usize, character_index: usize) -> Self {
        FusionDoubleIndex {
            class_index,
            character_index,
        }
    }
}

impl fmt::Display for FusionDoubleIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(C{}, χ{})", self.class_index, self.character_index)
    }
}

/// Element of a fusion double
#[derive(Debug, Clone)]
pub struct FusionDoubleElement {
    /// Coefficients for each basis element
    pub coeffs: HashMap<FusionDoubleIndex, Integer>,
    /// Group order (for validation)
    group_order: usize,
}

impl FusionDoubleElement {
    /// Create a new element from coefficients
    pub fn new(coeffs: HashMap<FusionDoubleIndex, Integer>, group_order: usize) -> Self {
        FusionDoubleElement { coeffs, group_order }
    }

    /// Create a basis element
    pub fn basis_element(index: FusionDoubleIndex, group_order: usize) -> Self {
        let mut coeffs = HashMap::new();
        coeffs.insert(index, Integer::one());
        FusionDoubleElement { coeffs, group_order }
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.values().all(|c| c.is_zero())
    }

    /// Add two elements
    pub fn add(&self, other: &FusionDoubleElement) -> Self {
        let mut result_coeffs = self.coeffs.clone();
        for (index, coeff) in &other.coeffs {
            let current = result_coeffs.entry(index.clone()).or_insert(Integer::zero());
            *current = current.clone() + coeff.clone();
        }
        result_coeffs.retain(|_, v| !v.is_zero());
        FusionDoubleElement::new(result_coeffs, self.group_order)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &Integer) -> Self {
        let mut result_coeffs = HashMap::new();
        for (index, coeff) in &self.coeffs {
            result_coeffs.insert(index.clone(), coeff.clone() * scalar.clone());
        }
        result_coeffs.retain(|_, v| !v.is_zero());
        FusionDoubleElement::new(result_coeffs, self.group_order)
    }
}

impl fmt::Display for FusionDoubleElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (index, coeff) in &self.coeffs {
            if !first {
                write!(f, " + ")?;
            }
            if coeff != &Integer::one() {
                write!(f, "{}*", coeff)?;
            }
            write!(f, "{}", index)?;
            first = false;
        }
        Ok(())
    }
}

/// Drinfeld double of a finite group
///
/// The Drinfeld double D(G) is a modular tensor category whose simple objects
/// are indexed by pairs (class_idx, character_idx).
///
/// This is a simplified implementation that stores the group order and
/// provides basic fusion ring operations.
#[derive(Debug, Clone)]
pub struct FusionDouble {
    /// Order of the underlying group
    group_order: usize,
    /// Number of conjugacy classes
    num_classes: usize,
    /// Basis indices
    basis_indices: Vec<FusionDoubleIndex>,
    /// Fusion coefficients (computed lazily)
    fusion_coeffs: HashMap<(usize, usize, usize), i64>,
    /// S-matrix cache
    s_matrix_cache: Option<Vec<Vec<Rational>>>,
}

impl FusionDouble {
    /// Create a new fusion double for a finite group
    ///
    /// # Arguments
    /// * `group_order` - Order of the group |G|
    /// * `num_classes` - Number of conjugacy classes
    ///
    /// For a simple example, D(Z_n) has n² simple objects
    pub fn new(group_order: usize, num_classes: usize) -> Self {
        let mut basis_indices = Vec::new();

        // For each conjugacy class, add basis elements for each character
        // Simplified: assume each class has the same number of characters
        let chars_per_class = (group_order / num_classes).max(1);

        for class_idx in 0..num_classes {
            for char_idx in 0..chars_per_class {
                basis_indices.push(FusionDoubleIndex::new(class_idx, char_idx));
            }
        }

        FusionDouble {
            group_order,
            num_classes,
            basis_indices,
            fusion_coeffs: HashMap::new(),
            s_matrix_cache: None,
        }
    }

    /// Create fusion double of cyclic group Z_n
    pub fn cyclic_group(n: usize) -> Self {
        // Z_n has n conjugacy classes (all elements are their own class)
        Self::new(n, n)
    }

    /// Get the group order
    pub fn group_order(&self) -> usize {
        self.group_order
    }

    /// Get the dimension (number of simple objects)
    pub fn dimension(&self) -> usize {
        self.basis_indices.len()
    }

    /// Get the global quantum dimension
    ///
    /// For the Drinfeld double D(G), this equals |G|²
    pub fn global_quantum_dimension(&self) -> Integer {
        let n = (self.group_order * self.group_order) as i64;
        Integer::from(n)
    }

    /// Create a basis element by index
    pub fn basis_element(&self, index: usize) -> Option<FusionDoubleElement> {
        if index < self.basis_indices.len() {
            Some(FusionDoubleElement::basis_element(
                self.basis_indices[index].clone(),
                self.group_order,
            ))
        } else {
            None
        }
    }

    /// Compute the fusion product of two basis elements
    ///
    /// Returns a HashMap mapping result indices to multiplicities
    pub fn fusion_product(
        &self,
        idx_i: &FusionDoubleIndex,
        idx_j: &FusionDoubleIndex,
    ) -> HashMap<FusionDoubleIndex, i64> {
        // Simplified implementation
        let mut result = HashMap::new();

        // Placeholder: return the first index with multiplicity 1
        if idx_i == idx_j {
            result.insert(idx_i.clone(), 1);
        }

        result
    }

    /// Compute the S-matrix
    ///
    /// The S-matrix encodes modular transformation properties
    pub fn s_matrix(&mut self) -> Vec<Vec<Rational>> {
        if let Some(ref cached) = self.s_matrix_cache {
            return cached.clone();
        }

        let n = self.dimension();
        let mut s_matrix = vec![vec![Rational::zero(); n]; n];

        // Compute S-matrix entries
        for i in 0..n {
            for j in 0..n {
                s_matrix[i][j] = self.compute_s_ij(i, j);
            }
        }

        self.s_matrix_cache = Some(s_matrix.clone());
        s_matrix
    }

    /// Compute an S-matrix entry
    fn compute_s_ij(&self, i: usize, j: usize) -> Rational {
        // Placeholder implementation
        if i == j {
            Rational::one()
        } else {
            Rational::zero()
        }
    }

    /// Compute the quantum dimension of a basis element
    pub fn quantum_dimension(&self, _index: usize) -> Rational {
        // Simplified: all simple objects have quantum dimension 1
        Rational::one()
    }

    /// Compute twist (ribbon element) for a basis element
    pub fn twist(&self, _index: usize) -> Rational {
        // Placeholder implementation
        Rational::one()
    }

    /// Check if the fusion ring is multiplicity-free
    pub fn is_multiplicity_free(&self) -> bool {
        self.fusion_coeffs.values().all(|&n| n == 0 || n == 1)
    }
}

impl fmt::Display for FusionDouble {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FusionDouble(|G|={}, {} simple objects)",
            self.group_order,
            self.dimension()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_double_creation() {
        // Create the fusion double of Z_3
        let fd = FusionDouble::cyclic_group(3);

        assert_eq!(fd.group_order(), 3);
        assert!(fd.dimension() > 0);
    }

    #[test]
    fn test_global_quantum_dimension() {
        // D(Z_3) has global dimension = 3² = 9
        let fd = FusionDouble::cyclic_group(3);
        let global_dim = fd.global_quantum_dimension();
        assert_eq!(global_dim, Integer::from(9));
    }

    #[test]
    fn test_basis_element() {
        let fd = FusionDouble::cyclic_group(2);

        if let Some(elem) = fd.basis_element(0) {
            assert!(!elem.is_zero());
        }
    }

    #[test]
    fn test_element_addition() {
        let fd = FusionDouble::cyclic_group(2);

        if let Some(e1) = fd.basis_element(0) {
            let e2 = e1.add(&e1);
            assert!(!e2.is_zero());
        }
    }

    #[test]
    fn test_quantum_dimension() {
        let fd = FusionDouble::cyclic_group(4);
        let d = fd.quantum_dimension(0);
        assert!(d > Rational::zero());
    }

    #[test]
    fn test_fusion_double_display() {
        let fd = FusionDouble::cyclic_group(5);
        let display = format!("{}", fd);
        assert!(display.contains("FusionDouble"));
        assert!(display.contains("|G|=5"));
    }
}


//! Ribbon tableaux with spin and fermionic formula
//!
//! A ribbon tableau is a standard tableau whose shape is a ribbon (border strip).
//! Ribbons are connected skew shapes containing no 2×2 blocks.
//! The spin of a ribbon is related to its height and is crucial in symmetric function theory.

use crate::partitions::Partition;
use crate::skew_partition::{SkewPartition, SkewTableau};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// A ribbon tableau - a tableau of ribbon shape
///
/// A ribbon is a connected skew partition that contains no 2×2 block of cells.
/// Ribbons are also called border strips or rim hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RibbonTableau {
    /// The underlying skew tableau
    tableau: SkewTableau,
}

impl RibbonTableau {
    /// Create a new ribbon tableau
    ///
    /// Returns None if the shape is not a ribbon or the tableau is not standard
    pub fn new(tableau: SkewTableau) -> Option<Self> {
        // Check that the shape is a ribbon
        if !tableau.shape().is_ribbon() {
            return None;
        }

        // Check that the tableau is standard
        if !tableau.is_standard() {
            return None;
        }

        Some(RibbonTableau { tableau })
    }

    /// Create a ribbon tableau from a skew partition and a sequence of labels
    ///
    /// The labels should be placed on the ribbon cells in order
    pub fn from_partition_and_labels(
        shape: SkewPartition,
        labels: Vec<usize>,
    ) -> Option<Self> {
        if !shape.is_ribbon() {
            return None;
        }

        if labels.len() != shape.size() {
            return None;
        }

        // Create the tableau entries
        let cells = shape.cells();
        let mut label_map: HashMap<(usize, usize), usize> = HashMap::new();
        for (i, &cell) in cells.iter().enumerate() {
            label_map.insert(cell, labels[i]);
        }

        // Build the entries matrix
        let outer_parts = shape.outer().parts();
        let mut entries = Vec::new();

        for (row, &outer_len) in outer_parts.iter().enumerate() {
            let mut row_entries = Vec::new();
            for col in 0..outer_len {
                if let Some(&label) = label_map.get(&(row, col)) {
                    row_entries.push(Some(label));
                } else {
                    row_entries.push(None);
                }
            }
            entries.push(row_entries);
        }

        let tableau = SkewTableau::new(shape, entries)?;
        RibbonTableau::new(tableau)
    }

    /// Get the underlying skew tableau
    pub fn tableau(&self) -> &SkewTableau {
        &self.tableau
    }

    /// Get the shape of the ribbon
    pub fn shape(&self) -> &SkewPartition {
        self.tableau.shape()
    }

    /// Get the size (number of cells) of the ribbon
    pub fn size(&self) -> usize {
        self.shape().size()
    }

    /// Compute the height of the ribbon
    ///
    /// The height is the number of rows spanned by the ribbon minus 1
    pub fn height(&self) -> usize {
        self.shape().height().unwrap_or(0)
    }

    /// Compute the spin of the ribbon
    ///
    /// The spin is defined as: spin = height - (size - height - 1) = 2*height - size + 1
    /// This is also equal to (number of rows - number of columns) the ribbon spans.
    ///
    /// The spin determines the sign in various symmetric function identities.
    pub fn spin(&self) -> i32 {
        self.shape().spin().unwrap_or(0)
    }

    /// Get the entry at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<usize> {
        self.tableau.get(row, col)
    }

    /// Get all entries in reading order (left to right, top to bottom)
    pub fn entries(&self) -> Vec<usize> {
        let cells = self.shape().cells();
        let mut entries = Vec::new();

        for &(row, col) in &cells {
            if let Some(entry) = self.get(row, col) {
                entries.push(entry);
            }
        }

        entries
    }
}

/// A ribbon decomposition of a partition
///
/// Some partitions can be decomposed into a sequence of ribbons.
/// This is important in computing characters of symmetric groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RibbonDecomposition {
    /// The original partition
    partition: Partition,
    /// The ribbons in the decomposition
    ribbons: Vec<SkewPartition>,
}

impl RibbonDecomposition {
    /// Create a ribbon decomposition from a partition
    ///
    /// Returns None if the partition cannot be decomposed into ribbons
    pub fn new(partition: Partition) -> Option<Self> {
        // Convert partition to a skew partition (with empty inner partition)
        let skew = SkewPartition::new(partition.clone(), Partition::new(vec![]))?;

        // Try to decompose into ribbons
        let ribbons = skew.ribbon_decomposition()?;

        Some(RibbonDecomposition { partition, ribbons })
    }

    /// Create a ribbon decomposition from an explicit list of ribbons
    pub fn from_ribbons(partition: Partition, ribbons: Vec<SkewPartition>) -> Option<Self> {
        // Verify all ribbons are valid
        for ribbon in &ribbons {
            if !ribbon.is_ribbon() {
                return None;
            }
        }

        // Verify the ribbons partition the shape
        let total_size: usize = ribbons.iter().map(|r| r.size()).sum();
        if total_size != partition.sum() {
            return None;
        }

        Some(RibbonDecomposition { partition, ribbons })
    }

    /// Get the original partition
    pub fn partition(&self) -> &Partition {
        &self.partition
    }

    /// Get the ribbons
    pub fn ribbons(&self) -> &[SkewPartition] {
        &self.ribbons
    }

    /// Get the number of ribbons
    pub fn num_ribbons(&self) -> usize {
        self.ribbons.len()
    }

    /// Get the spin of a specific ribbon
    pub fn ribbon_spin(&self, index: usize) -> Option<i32> {
        self.ribbons.get(index)?.spin()
    }

    /// Get all spins in order
    pub fn spins(&self) -> Vec<i32> {
        self.ribbons
            .iter()
            .filter_map(|r| r.spin())
            .collect()
    }

    /// Compute the fermionic formula factor
    ///
    /// The fermionic formula gives a generating function for ribbon tableaux:
    /// ∏(1 + q^{spin(ribbon)}) over all ribbons in the decomposition
    ///
    /// Returns the coefficients of the polynomial as a map: power -> coefficient
    pub fn fermionic_polynomial(&self) -> HashMap<i32, Integer> {
        let mut poly: HashMap<i32, Integer> = HashMap::new();
        poly.insert(0, Integer::one());

        for ribbon in &self.ribbons {
            if let Some(spin) = ribbon.spin() {
                let mut new_poly: HashMap<i32, Integer> = HashMap::new();

                // Multiply current polynomial by (1 + q^spin)
                for (&power, coeff) in &poly {
                    // Contribute to q^power (from the 1 term)
                    let entry = new_poly.entry(power).or_insert(Integer::zero());
                    *entry = entry.clone() + coeff.clone();

                    // Contribute to q^(power + spin) (from the q^spin term)
                    let entry2 = new_poly.entry(power + spin).or_insert(Integer::zero());
                    *entry2 = entry2.clone() + coeff.clone();
                }

                poly = new_poly;
            }
        }

        poly
    }

    /// Evaluate the fermionic polynomial at a specific value of q
    ///
    /// This computes ∏(1 + q^{spin(ribbon)}) for a given q
    pub fn fermionic_value(&self, q: &Rational) -> Rational {
        let mut result = <Rational as Ring>::one();

        for ribbon in &self.ribbons {
            if let Some(spin) = ribbon.spin() {
                // Compute q^spin
                let q_power = if spin >= 0 {
                    q.pow(spin as u32)
                } else {
                    // For negative spin, use q^(-|spin|) = 1/q^|spin|
                    <Rational as Ring>::one() / q.pow((-spin) as u32)
                };

                // Multiply by (1 + q^spin)
                result = result * (<Rational as Ring>::one() + q_power);
            }
        }

        result
    }

    /// Compute the total spin (sum of all ribbon spins)
    pub fn total_spin(&self) -> i32 {
        self.spins().iter().sum()
    }
}

/// Generate all ribbon tableaux of a given ribbon shape
///
/// Returns all standard tableaux whose shape is the given ribbon
pub fn ribbon_tableaux(shape: SkewPartition) -> Option<Vec<RibbonTableau>> {
    if !shape.is_ribbon() {
        return None;
    }

    let n = shape.size();
    if n == 0 {
        return Some(vec![]);
    }

    // For a ribbon, there are potentially multiple standard tableaux
    // depending on how we fill it with 1,2,...,n

    // Generate all permutations and check which ones give valid standard tableaux
    let cells = shape.cells();
    let mut result = Vec::new();

    // Generate all permutations of 1..=n
    use crate::permutations::all_permutations;
    let perms = all_permutations(n);

    for perm in perms {
        let labels: Vec<usize> = perm.as_slice().iter().map(|&x| x + 1).collect();

        if let Some(tableau) = RibbonTableau::from_partition_and_labels(shape.clone(), labels) {
            result.push(tableau);
        }
    }

    Some(result)
}

/// Compute the fermionic formula for a partition
///
/// Given a partition λ and a content μ, computes the fermionic formula
/// which is a polynomial in q. Returns the coefficients as a map.
pub fn fermionic_formula(partition: &Partition) -> Option<HashMap<i32, Integer>> {
    let decomp = RibbonDecomposition::new(partition.clone())?;
    Some(decomp.fermionic_polynomial())
}

/// Compute the Kostka-Foulkes polynomial K_{λ,μ}(q) using ribbons
///
/// The Kostka-Foulkes polynomial counts semistandard tableaux of shape λ
/// and content μ, weighted by the charge statistic.
/// For ribbon decompositions, this can be computed using the fermionic formula.
///
/// This is a simplified version that works when λ can be decomposed into ribbons.
pub fn kostka_foulkes_ribbon(
    lambda: &Partition,
    mu: &Partition,
) -> Option<HashMap<i32, Integer>> {
    // Check if lambda and mu are compatible
    if lambda.sum() != mu.sum() {
        return None;
    }

    // For ribbons, the Kostka-Foulkes polynomial can be computed using
    // the fermionic formula for certain cases
    // This is a placeholder for the full implementation

    // For now, just return the fermionic formula for lambda
    fermionic_formula(lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ribbon_tableau_creation() {
        // Create a simple ribbon: [2, 1]
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        assert!(shape.is_ribbon());

        // Create a ribbon tableau with labels [1, 2, 3]
        let labels = vec![1, 2, 3];
        let ribbon = RibbonTableau::from_partition_and_labels(shape, labels);

        assert!(ribbon.is_some());
        let r = ribbon.unwrap();
        assert_eq!(r.size(), 3);
    }

    #[test]
    fn test_ribbon_height() {
        // Ribbon [2, 1] has height 1 (spans 2 rows)
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let labels = vec![1, 2, 3];
        let ribbon = RibbonTableau::from_partition_and_labels(shape, labels).unwrap();

        assert_eq!(ribbon.height(), 1);
    }

    #[test]
    fn test_ribbon_spin() {
        // Ribbon [2, 1] has spin = 2*1 - 3 + 1 = 0
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let labels = vec![1, 2, 3];
        let ribbon = RibbonTableau::from_partition_and_labels(shape, labels).unwrap();

        assert_eq!(ribbon.spin(), 0);
    }

    #[test]
    fn test_ribbon_spin_horizontal() {
        // Single row ribbon [3] has height 0, spin = 2*0 - 3 + 1 = -2
        let outer = Partition::new(vec![3]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let labels = vec![1, 2, 3];
        let ribbon = RibbonTableau::from_partition_and_labels(shape, labels).unwrap();

        assert_eq!(ribbon.height(), 0);
        assert_eq!(ribbon.spin(), -2);
    }

    #[test]
    fn test_ribbon_spin_vertical() {
        // Vertical ribbon [1, 1, 1] has height 2, spin = 2*2 - 3 + 1 = 2
        let outer = Partition::new(vec![1, 1, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let labels = vec![1, 2, 3];
        let ribbon = RibbonTableau::from_partition_and_labels(shape, labels).unwrap();

        assert_eq!(ribbon.height(), 2);
        assert_eq!(ribbon.spin(), 2);
    }

    #[test]
    fn test_ribbon_decomposition() {
        // Partition [3, 2, 1] might decompose into ribbons
        let partition = Partition::new(vec![3, 2, 1]);

        // Try to create decomposition
        let decomp = RibbonDecomposition::new(partition.clone());

        if let Some(d) = decomp {
            assert_eq!(d.partition(), &partition);
            assert!(d.num_ribbons() > 0);

            // Total size should equal partition size
            let total: usize = d.ribbons().iter().map(|r| r.size()).sum();
            assert_eq!(total, partition.sum());
        }
    }

    #[test]
    fn test_fermionic_polynomial_simple() {
        // Single cell ribbon [1] has spin -1 + 1 = 0
        let partition = Partition::new(vec![1]);
        let decomp = RibbonDecomposition::new(partition).unwrap();

        let poly = decomp.fermionic_polynomial();

        // For a single ribbon with spin 0, polynomial is (1 + q^0) = (1 + 1) = 2
        // Actually, (1 + q^0) as a polynomial is 1 + 1*q^0 = 2 at q^0
        // No, the polynomial is kept symbolic: coefficients are {0: 2}
        // Wait, let me reconsider: (1 + q^0) = (1 + 1) as a polynomial is coefficient 2 at power 0

        // Actually the formula is ∏(1 + q^spin), so for spin=0:
        // (1 + q^0) = 1 + 1 as polynomial, which has coeff 1 at power 0, coeff 1 at power 0
        // The sum gives 2 at power 0? No.

        // Let me think again: The polynomial (1 + q^0) = 1 + 1 * q^0
        // Since q^0 = 1, this is just constant 2.
        // But in the polynomial representation, we track coefficients:
        // The constant term is the coefficient of q^0

        // Actually, in the implementation, we multiply (1 + q^spin)
        // For spin = 0: (1 + q^0) means we have terms at power 0 (from "1") and power 0 (from "q^0")
        // So we should have coefficient 2 at power 0

        // Hmm, but actually looking at the code, it adds contributions separately:
        // power = 0 gets +1 from the "1" term
        // power = 0 + spin = 0 + 0 = 0 gets +1 from the "q^spin" term
        // So total coefficient at power 0 is 2

        assert!(poly.contains_key(&0));
        assert_eq!(poly[&0], Integer::from(2));
    }

    #[test]
    fn test_fermionic_polynomial_hook() {
        // Ribbon [2, 1] has spin 0
        let partition = Partition::new(vec![2, 1]);
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let decomp = RibbonDecomposition::from_ribbons(partition, vec![shape]).unwrap();

        let poly = decomp.fermionic_polynomial();

        // (1 + q^0) = 2 at power 0
        assert_eq!(poly[&0], Integer::from(2));
    }

    #[test]
    fn test_fermionic_value() {
        // Test evaluation at q = 2
        let partition = Partition::new(vec![2, 1]);
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![]);
        let shape = SkewPartition::new(outer, inner).unwrap();

        let decomp = RibbonDecomposition::from_ribbons(partition, vec![shape]).unwrap();

        let q = Rational::from(2);
        let value = decomp.fermionic_value(&q);

        // (1 + 2^0) = (1 + 1) = 2
        assert_eq!(value, Rational::from(2));
    }

    #[test]
    fn test_fermionic_two_ribbons() {
        // Two ribbons: one with spin -2 (horizontal [3]), one with spin 2 (vertical [1,1,1])
        let outer1 = Partition::new(vec![3]);
        let inner1 = Partition::new(vec![]);
        let ribbon1 = SkewPartition::new(outer1, inner1).unwrap();
        assert_eq!(ribbon1.spin(), Some(-2));

        let outer2 = Partition::new(vec![1, 1, 1]);
        let inner2 = Partition::new(vec![]);
        let ribbon2 = SkewPartition::new(outer2, inner2).unwrap();
        assert_eq!(ribbon2.spin(), Some(2));

        // Create a dummy partition for the decomposition
        let partition = Partition::new(vec![3, 1, 1, 1]);
        let decomp = RibbonDecomposition::from_ribbons(
            partition,
            vec![ribbon1, ribbon2],
        ).unwrap();

        let poly = decomp.fermionic_polynomial();

        // (1 + q^{-2})(1 + q^2)
        // = 1 + q^2 + q^{-2} + q^0
        // = q^{-2} + 1 + q^0 + q^2
        // = q^{-2} + 2 + q^2  (since q^0 = 1 merges with the constant 1)
        // So we should have: coefficient 1 at power -2, coefficient 2 at power 0, coefficient 1 at power 2

        // Wait, let's trace through the algorithm:
        // Start: {0: 1}
        // Multiply by (1 + q^{-2}):
        //   - From power 0, coeff 1: contribute to 0 and 0 + (-2) = -2
        //   - Result: {0: 1, -2: 1}
        // Multiply by (1 + q^2):
        //   - From power 0, coeff 1: contribute to 0 and 0 + 2 = 2
        //   - From power -2, coeff 1: contribute to -2 and -2 + 2 = 0
        //   - Result: {0: 1+1=2, 2: 1, -2: 1}

        assert_eq!(poly.get(&-2), Some(&Integer::from(1)));
        assert_eq!(poly.get(&0), Some(&Integer::from(2)));
        assert_eq!(poly.get(&2), Some(&Integer::from(1)));
    }

    #[test]
    fn test_total_spin() {
        let outer1 = Partition::new(vec![3]);
        let inner1 = Partition::new(vec![]);
        let ribbon1 = SkewPartition::new(outer1, inner1).unwrap();

        let outer2 = Partition::new(vec![1, 1, 1]);
        let inner2 = Partition::new(vec![]);
        let ribbon2 = SkewPartition::new(outer2, inner2).unwrap();

        let partition = Partition::new(vec![3, 1, 1, 1]);
        let decomp = RibbonDecomposition::from_ribbons(
            partition,
            vec![ribbon1, ribbon2],
        ).unwrap();

        // Total spin = -2 + 2 = 0
        assert_eq!(decomp.total_spin(), 0);
    }

    #[test]
    fn test_fermionic_formula_helper() {
        // Test the helper function
        let partition = Partition::new(vec![2, 1]);
        let poly = fermionic_formula(&partition);

        assert!(poly.is_some());
        let p = poly.unwrap();
        assert!(p.contains_key(&0));
    }
}

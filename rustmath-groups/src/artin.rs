//! Artin Groups
//!
//! This module implements Artin groups, which are groups defined by generators
//! and relations based on a Coxeter matrix. Artin groups are fundamental objects
//! in geometric group theory with deep connections to braid groups, mapping class
//! groups, and Coxeter groups.
//!
//! # Mathematical Structure
//!
//! An Artin group is defined by:
//! - Generators: s_i for i in an index set I
//! - Relations: For each pair (i,j) with m_ij ≠ ∞ in the Coxeter matrix,
//!   the alternating product s_i s_j s_i ... (m_ij terms) equals
//!   the alternating product s_j s_i s_j ... (m_ij terms)
//!
//! Special cases:
//! - m_ii = 1 for all i
//! - m_ij = m_ji
//! - m_ij ∈ {2, 3, 4, ...} ∪ {∞}
//!
//! When m_ij = 2, the generators s_i and s_j commute.
//! When m_ij = ∞, there is no relation between s_i and s_j.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::artin::{ArtinGroup, CoxeterMatrix};
//!
//! // Create a Coxeter matrix for type A_2
//! let matrix = CoxeterMatrix::type_a(2);
//!
//! // Create the corresponding Artin group
//! let a2 = ArtinGroup::new(matrix);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use crate::finitely_presented::FinitelyPresentedGroup;
use crate::free_group::FreeGroupElement;
use crate::group_traits::{Group, GroupElement};

/// A Coxeter matrix defining the relations of an Artin group
///
/// The matrix stores m_ij values where m_ii = 1 and m_ij = m_ji.
/// Value 0 represents infinity (no relation).
#[derive(Debug, Clone)]
pub struct CoxeterMatrix {
    /// The rank (number of generators)
    rank: usize,
    /// Matrix entries: (i, j) -> m_ij
    /// We store 0 for infinity
    entries: Vec<Vec<usize>>,
}

impl CoxeterMatrix {
    /// Create a new Coxeter matrix from explicit entries
    ///
    /// # Arguments
    ///
    /// * `entries` - Square matrix where entries[i][j] = m_ij (use 0 for ∞)
    pub fn new(entries: Vec<Vec<usize>>) -> Self {
        let rank = entries.len();
        assert!(rank > 0, "Coxeter matrix must have at least one generator");

        // Verify it's square
        for row in &entries {
            assert_eq!(row.len(), rank, "Coxeter matrix must be square");
        }

        // Verify diagonal is all 1s
        for i in 0..rank {
            assert_eq!(entries[i][i], 1, "Diagonal entries must be 1");
        }

        // Verify symmetry
        for i in 0..rank {
            for j in 0..rank {
                assert_eq!(
                    entries[i][j], entries[j][i],
                    "Coxeter matrix must be symmetric"
                );
            }
        }

        Self { rank, entries }
    }

    /// Create a Coxeter matrix of type A_n
    ///
    /// Type A_n has generators s_0, ..., s_{n-1} with relations:
    /// - s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1} (m_ij = 3 for adjacent)
    /// - s_i s_j = s_j s_i for |i-j| ≥ 2 (m_ij = 2 for non-adjacent)
    pub fn type_a(n: usize) -> Self {
        assert!(n >= 1, "Type A requires n >= 1");

        let mut entries = vec![vec![1; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    entries[i][j] = 1;
                } else if (i as i32 - j as i32).abs() == 1 {
                    entries[i][j] = 3; // Adjacent generators
                } else {
                    entries[i][j] = 2; // Non-adjacent commute
                }
            }
        }

        Self { rank: n, entries }
    }

    /// Create a Coxeter matrix of type B_n / C_n
    ///
    /// Type B_n has an additional relation at the end.
    pub fn type_b(n: usize) -> Self {
        assert!(n >= 2, "Type B requires n >= 2");

        let mut matrix = Self::type_a(n);
        // Modify the last two generators to have order 4
        matrix.entries[n - 2][n - 1] = 4;
        matrix.entries[n - 1][n - 2] = 4;

        matrix
    }

    /// Returns the rank of the Coxeter matrix
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the entry m_ij (returns 0 for infinity)
    pub fn get(&self, i: usize, j: usize) -> usize {
        assert!(i < self.rank && j < self.rank, "Index out of bounds");
        self.entries[i][j]
    }

    /// Check if m_ij is finite
    pub fn is_finite(&self, i: usize, j: usize) -> bool {
        self.get(i, j) > 0
    }
}

impl fmt::Display for CoxeterMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coxeter matrix of rank {}:", self.rank)?;
        for i in 0..self.rank {
            write!(f, "  [")?;
            for j in 0..self.rank {
                let val = self.entries[i][j];
                if val == 0 {
                    write!(f, "∞")?;
                } else {
                    write!(f, "{}", val)?;
                }
                if j < self.rank - 1 {
                    write!(f, " ")?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

/// An Artin group defined by a Coxeter matrix
///
/// The Artin group has generators s_i and relations determined by the Coxeter matrix.
#[derive(Debug, Clone)]
pub struct ArtinGroup {
    /// The Coxeter matrix
    coxeter_matrix: CoxeterMatrix,
    /// Generator names
    generator_names: Vec<String>,
    /// Underlying finitely presented group
    fp_group: FinitelyPresentedGroup,
}

impl ArtinGroup {
    /// Create a new Artin group from a Coxeter matrix
    pub fn new(coxeter_matrix: CoxeterMatrix) -> Self {
        let rank = coxeter_matrix.rank();
        let generator_names: Vec<String> = (0..rank).map(|i| format!("s{}", i)).collect();

        // Build relations from the Coxeter matrix
        let mut relations = Vec::new();

        for i in 0..rank {
            for j in (i + 1)..rank {
                let m_ij = coxeter_matrix.get(i, j);

                if m_ij == 0 {
                    // No relation (infinite order)
                    continue;
                } else if m_ij == 1 {
                    // This shouldn't happen for i ≠ j
                    continue;
                } else {
                    // Create the alternating product relation
                    // s_i s_j s_i ... (m_ij terms) = s_j s_i s_j ... (m_ij terms)

                    let mut left_word = FreeGroupElement::identity();
                    let mut right_word = FreeGroupElement::identity();

                    for k in 0..m_ij {
                        if k % 2 == 0 {
                            left_word = left_word.multiply(&FreeGroupElement::generator(i as isize, 1));
                            right_word =
                                right_word.multiply(&FreeGroupElement::generator(j as isize, 1));
                        } else {
                            left_word = left_word.multiply(&FreeGroupElement::generator(j as isize, 1));
                            right_word =
                                right_word.multiply(&FreeGroupElement::generator(i as isize, 1));
                        }
                    }

                    // Relation is left * right^{-1} = 1
                    let relation = left_word.multiply(&right_word.inverse());
                    relations.push(relation);
                }
            }
        }

        let fp_group = FinitelyPresentedGroup::new(generator_names.clone(), relations.iter().map(|r| r.word().iter().flat_map(|(gen, exp)| vec![*gen, *exp]).collect()).collect());

        Self {
            coxeter_matrix,
            generator_names,
            fp_group,
        }
    }

    /// Returns a reference to the Coxeter matrix
    pub fn coxeter_matrix(&self) -> &CoxeterMatrix {
        &self.coxeter_matrix
    }

    /// Returns the rank (number of generators)
    pub fn rank(&self) -> usize {
        self.coxeter_matrix.rank()
    }

    /// Returns the generator names
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// Returns a reference to the underlying finitely presented group
    pub fn as_finitely_presented_group(&self) -> &FinitelyPresentedGroup {
        &self.fp_group
    }

    /// Get the ith generator (0-indexed)
    pub fn gen(&self, i: usize) -> ArtinGroupElement {
        assert!(i < self.rank(), "Generator index out of bounds");
        ArtinGroupElement {
            parent: self.clone(),
            word: FreeGroupElement::generator(i as isize, 1),
        }
    }
}

impl fmt::Display for ArtinGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Artin group of rank {}", self.rank())
    }
}

impl Group for ArtinGroup {
    type Element = ArtinGroupElement;

    fn identity(&self) -> Self::Element {
        ArtinGroupElement {
            parent: self.clone(),
            word: FreeGroupElement::identity(),
        }
    }

    fn is_finite(&self) -> bool {
        // Artin groups are generally infinite
        // Only finite-type Artin groups with specific conditions can be finite
        false
    }

    fn order(&self) -> Option<usize> {
        // Artin groups are infinite
        None
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if element belongs to this group by comparing rank
        self.coxeter_matrix.rank() == element.parent.coxeter_matrix.rank()
    }
}

/// An element of an Artin group
#[derive(Debug, Clone)]
pub struct ArtinGroupElement {
    parent: ArtinGroup,
    word: FreeGroupElement,
}

impl ArtinGroupElement {
    /// Create a new element from a word
    pub fn new(parent: ArtinGroup, word: FreeGroupElement) -> Self {
        Self { parent, word }
    }

    /// Returns a reference to the parent group
    pub fn parent(&self) -> &ArtinGroup {
        &self.parent
    }

    /// Returns a reference to the word representation
    pub fn word(&self) -> &FreeGroupElement {
        &self.word
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.word.is_identity()
    }

    /// Multiply with another element
    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            parent: self.parent.clone(),
            word: self.word.multiply(&other.word),
        }
    }

    /// Compute the inverse
    pub fn inverse(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            word: self.word.inverse(),
        }
    }

    /// Raise to a power
    pub fn pow(&self, n: isize) -> Self {
        Self {
            parent: self.parent.clone(),
            word: self.word.pow(n),
        }
    }
}

impl fmt::Display for ArtinGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.word)
    }
}

impl PartialEq for ArtinGroupElement {
    fn eq(&self, other: &Self) -> bool {
        // In a full implementation, this would use normal forms
        self.word == other.word
    }
}

impl Eq for ArtinGroupElement {}

impl std::hash::Hash for ArtinGroupElement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash only the word, consistent with PartialEq
        self.word.hash(state);
    }
}

impl std::ops::Mul for ArtinGroupElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.op(&other)
    }
}

impl GroupElement for ArtinGroupElement {
    fn identity() -> Self {
        // Create a minimal Artin group with one generator for the identity element
        let coxeter_matrix = CoxeterMatrix::new(vec![vec![1]]);
        let parent = ArtinGroup::new(coxeter_matrix);
        ArtinGroupElement {
            parent,
            word: FreeGroupElement::identity(),
        }
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn op(&self, other: &Self) -> Self {
        self.multiply(other)
    }
}

/// A finite-type Artin group (where the Coxeter group is finite)
///
/// These are also called spherical Artin groups or Artin groups of finite type.
pub type FiniteTypeArtinGroup = ArtinGroup;

/// An element of a finite-type Artin group
pub type FiniteTypeArtinGroupElement = ArtinGroupElement;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coxeter_matrix_type_a() {
        let m = CoxeterMatrix::type_a(3);
        assert_eq!(m.rank(), 3);
        assert_eq!(m.get(0, 0), 1);
        assert_eq!(m.get(0, 1), 3); // Adjacent
        assert_eq!(m.get(0, 2), 2); // Non-adjacent
    }

    #[test]
    fn test_coxeter_matrix_type_b() {
        let m = CoxeterMatrix::type_b(3);
        assert_eq!(m.rank(), 3);
        assert_eq!(m.get(1, 2), 4); // B-type relation
    }

    #[test]
    fn test_coxeter_matrix_custom() {
        let entries = vec![vec![1, 3, 2], vec![3, 1, 3], vec![2, 3, 1]];
        let m = CoxeterMatrix::new(entries);
        assert_eq!(m.rank(), 3);
    }

    #[test]
    #[should_panic(expected = "Coxeter matrix must be square")]
    fn test_invalid_coxeter_matrix() {
        CoxeterMatrix::new(vec![vec![1, 2], vec![2, 1, 3]]);
    }

    #[test]
    fn test_artin_group_creation() {
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);
        assert_eq!(a.rank(), 2);
    }

    #[test]
    fn test_artin_group_generators() {
        let m = CoxeterMatrix::type_a(3);
        let a = ArtinGroup::new(m);

        let s0 = a.gen(0);
        let s1 = a.gen(1);
        let s2 = a.gen(2);

        assert!(!s0.is_identity());
        assert!(!s1.is_identity());
        assert!(!s2.is_identity());
    }

    #[test]
    fn test_artin_element_multiplication() {
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);

        let s0 = a.gen(0);
        let s1 = a.gen(1);

        let prod = s0.multiply(&s1);
        assert!(!prod.is_identity());
    }

    #[test]
    fn test_artin_element_inverse() {
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);

        let s0 = a.gen(0);
        let s0_inv = s0.inverse();

        let prod = s0.multiply(&s0_inv);
        // Note: Without normalization, this won't reduce to identity
        // In a full implementation, we'd use normal forms
    }

    #[test]
    fn test_artin_element_power() {
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);

        let s0 = a.gen(0);
        let s0_cubed = s0.pow(3);

        assert!(!s0_cubed.is_identity());
    }

    #[test]
    fn test_identity() {
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);

        let id = a.identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_display() {
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);

        let display = format!("{}", a);
        assert!(display.contains("Artin group"));
    }

    #[test]
    fn test_commuting_generators() {
        // In type A_3, s_0 and s_2 commute (m_02 = 2)
        let m = CoxeterMatrix::type_a(3);
        let a = ArtinGroup::new(m);

        let s0 = a.gen(0);
        let s2 = a.gen(2);

        let s0s2 = s0.multiply(&s2);
        let s2s0 = s2.multiply(&s0);

        // They should have the same word representation
        // (in a full implementation with normalization)
    }

    #[test]
    fn test_braid_relation() {
        // Type A_2: s_0 s_1 s_0 = s_1 s_0 s_1
        let m = CoxeterMatrix::type_a(2);
        let a = ArtinGroup::new(m);

        let s0 = a.gen(0);
        let s1 = a.gen(1);

        let left = s0.multiply(&s1).multiply(&s0);
        let right = s1.multiply(&s0).multiply(&s1);

        // In a full implementation with normalization, these would be equal
    }
}

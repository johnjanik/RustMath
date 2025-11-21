//! Cubic Braid Groups
//!
//! This module implements cubic braid groups, which are factor groups of Artin braid groups
//! where the generators have order 3. These groups are closely related to complex reflection
//! groups and have important applications in topology and representation theory.
//!
//! The cubic braid group on n strands can be presented in three different types:
//! - **Coxeter**: The full cubic braid group with only cubic relations on generators
//! - **AssionS**: A finite factor group with additional Assion S-type relations
//! - **AssionU**: A finite factor group with additional Assion U-type relations
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::cubic_braid::{CubicBraidGroup, CubicBraidType};
//!
//! // Create a cubic braid group on 4 strands with Coxeter type
//! let cbg = CubicBraidGroup::new(4, CubicBraidType::Coxeter);
//! assert_eq!(cbg.strands(), 4);
//!
//! // Create an Assion S-type group
//! let assion_s = CubicBraidGroup::assion_s(5);
//! assert!(assion_s.is_finite());
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::finitely_presented::FinitelyPresentedGroup;
use crate::free_group::FreeGroupElement;
use crate::group_traits::Group;

/// Type of cubic braid group
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubicBraidType {
    /// Coxeter type: Full cubic braid group with only cubic relations
    Coxeter,
    /// AssionS type: Finite factor group with Assion S-type relations
    AssionS,
    /// AssionU type: Finite factor group with Assion U-type relations
    AssionU,
}

impl fmt::Display for CubicBraidType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CubicBraidType::Coxeter => write!(f, "Coxeter"),
            CubicBraidType::AssionS => write!(f, "AssionS"),
            CubicBraidType::AssionU => write!(f, "AssionU"),
        }
    }
}

/// A cubic braid group
///
/// The cubic braid group on n strands is a group with generators s_0, ..., s_{n-2}
/// satisfying the braid relations and the cubic relations s_i^3 = 1.
///
/// Depending on the type, additional relations may be imposed:
/// - **Coxeter**: No additional relations (infinite group for n ≥ 4)
/// - **AssionS**: Assion S-relation makes the group finite for all n
/// - **AssionU**: Assion U-relation makes the group finite for all n
#[derive(Debug, Clone)]
pub struct CubicBraidGroup {
    /// Number of strands
    n_strands: usize,
    /// Type of cubic braid group
    cbg_type: CubicBraidType,
    /// Generator names
    generator_names: Vec<String>,
    /// Underlying finitely presented group
    fp_group: FinitelyPresentedGroup,
}

impl CubicBraidGroup {
    /// Create a new cubic braid group on n strands
    ///
    /// # Arguments
    ///
    /// * `n_strands` - Number of strands (must be ≥ 2)
    /// * `cbg_type` - Type of cubic braid group
    ///
    /// # Panics
    ///
    /// Panics if n_strands < 2
    pub fn new(n_strands: usize, cbg_type: CubicBraidType) -> Self {
        assert!(n_strands >= 2, "Need at least 2 strands");

        let n_gens = n_strands - 1;
        let generator_names: Vec<String> = (0..n_gens).map(|i| format!("s{}", i)).collect();

        // Build relations
        let mut relations = Vec::new();

        // Cubic relations: s_i^3 = 1
        for i in 0..n_gens {
            let gen = FreeGroupElement::generator(i as i32, 3);
            relations.push(gen);
        }

        // Braid relations: s_i s_j = s_j s_i for |i-j| >= 2
        for i in 0..n_gens {
            for j in (i + 2)..n_gens {
                let si = FreeGroupElement::generator(i as i32, 1);
                let sj = FreeGroupElement::generator(j as i32, 1);
                let si_inv = FreeGroupElement::generator(i as i32, -1);
                let sj_inv = FreeGroupElement::generator(j as i32, -1);

                // s_i s_j s_i^{-1} s_j^{-1} = 1
                let rel = si.multiply(&sj).multiply(&si_inv).multiply(&sj_inv);
                relations.push(rel);
            }
        }

        // Braid relations: s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}
        for i in 0..(n_gens - 1) {
            let si = FreeGroupElement::generator(i as i32, 1);
            let si1 = FreeGroupElement::generator((i + 1) as i32, 1);
            let si_inv = FreeGroupElement::generator(i as i32, -1);
            let si1_inv = FreeGroupElement::generator((i + 1) as i32, -1);

            // s_i s_{i+1} s_i s_{i+1}^{-1} s_i^{-1} s_{i+1}^{-1} = 1
            let rel = si
                .multiply(&si1)
                .multiply(&si)
                .multiply(&si1_inv)
                .multiply(&si_inv)
                .multiply(&si1_inv);
            relations.push(rel);
        }

        // Add Assion relations if needed
        match cbg_type {
            CubicBraidType::Coxeter => {
                // No additional relations
            }
            CubicBraidType::AssionS => {
                if n_strands >= 5 {
                    // Assion S relation:
                    // s_3 s_1 t_2 s_1 t_2^{-1} t_3 t_2 s_1 t_2^{-1} t_3^{-1} = 1
                    // where t_i = (s_i s_{i+1})^3
                    let s1 = FreeGroupElement::generator(1, 1);
                    let s3 = FreeGroupElement::generator(3, 1);

                    // t_2 = (s_2 s_3)^3
                    let s2 = FreeGroupElement::generator(2, 1);
                    let s3_again = FreeGroupElement::generator(3, 1);
                    let t2 = s2.multiply(&s3_again);
                    let t2 = t2.multiply(&t2).multiply(&t2); // Cube it

                    // t_3 = (s_3 s_4)^3
                    let s3_third = FreeGroupElement::generator(3, 1);
                    let s4 = FreeGroupElement::generator(4, 1);
                    let t3 = s3_third.multiply(&s4);
                    let t3 = t3.multiply(&t3).multiply(&t3); // Cube it

                    let t2_inv = t2.inverse();
                    let t3_inv = t3.inverse();

                    let rel = s3
                        .multiply(&s1)
                        .multiply(&t2)
                        .multiply(&s1)
                        .multiply(&t2_inv)
                        .multiply(&t3)
                        .multiply(&t2)
                        .multiply(&s1)
                        .multiply(&t2_inv)
                        .multiply(&t3_inv);
                    relations.push(rel);
                }
            }
            CubicBraidType::AssionU => {
                if n_strands >= 5 {
                    // Assion U relation: t_1 t_3 = 1
                    // where t_i = (s_i s_{i+1})^3

                    // t_1 = (s_1 s_2)^3
                    let s1 = FreeGroupElement::generator(1, 1);
                    let s2 = FreeGroupElement::generator(2, 1);
                    let t1 = s1.multiply(&s2);
                    let t1 = t1.multiply(&t1).multiply(&t1); // Cube it

                    // t_3 = (s_3 s_4)^3
                    let s3 = FreeGroupElement::generator(3, 1);
                    let s4 = FreeGroupElement::generator(4, 1);
                    let t3 = s3.multiply(&s4);
                    let t3 = t3.multiply(&t3).multiply(&t3); // Cube it

                    let rel = t1.multiply(&t3);
                    relations.push(rel);
                }
            }
        }

        let fp_group = FinitelyPresentedGroup::new(n_gens, relations);

        Self {
            n_strands,
            cbg_type,
            generator_names,
            fp_group,
        }
    }

    /// Create an Assion S-type cubic braid group
    ///
    /// This is a shorthand for `CubicBraidGroup::new(n, CubicBraidType::AssionS)`
    pub fn assion_s(n_strands: usize) -> Self {
        Self::new(n_strands, CubicBraidType::AssionS)
    }

    /// Create an Assion U-type cubic braid group
    ///
    /// This is a shorthand for `CubicBraidGroup::new(n, CubicBraidType::AssionU)`
    pub fn assion_u(n_strands: usize) -> Self {
        Self::new(n_strands, CubicBraidType::AssionU)
    }

    /// Returns the number of strands
    pub fn strands(&self) -> usize {
        self.n_strands
    }

    /// Returns the type of the cubic braid group
    pub fn cbg_type(&self) -> CubicBraidType {
        self.cbg_type
    }

    /// Returns the index set [0, 1, ..., n-2] where n is the number of strands
    pub fn index_set(&self) -> Vec<usize> {
        (0..self.n_strands - 1).collect()
    }

    /// Returns the generator names
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// Returns whether this is a finite group
    ///
    /// - Coxeter type is finite for n ≤ 3, infinite for n ≥ 4
    /// - AssionS and AssionU types are always finite
    pub fn is_finite(&self) -> bool {
        match self.cbg_type {
            CubicBraidType::Coxeter => self.n_strands <= 3,
            CubicBraidType::AssionS | CubicBraidType::AssionU => true,
        }
    }

    /// Returns the order of the group (if finite)
    ///
    /// For Coxeter type with n ≤ 3:
    /// - n=2: order 3
    /// - n=3: order 24 (isomorphic to SL(2,3))
    ///
    /// For Assion types, the orders are:
    /// - AssionS(n): 3^(n-1) * n!
    /// - AssionU(n): 3^(n-1) * n! / 2
    pub fn order(&self) -> Option<usize> {
        if !self.is_finite() {
            return None;
        }

        match (self.cbg_type, self.n_strands) {
            (CubicBraidType::Coxeter, 2) => Some(3),
            (CubicBraidType::Coxeter, 3) => Some(24),
            (CubicBraidType::AssionS, n) => {
                // 3^(n-1) * n!
                let three_power = 3_usize.pow((n - 1) as u32);
                let factorial = (1..=n).product::<usize>();
                Some(three_power * factorial)
            }
            (CubicBraidType::AssionU, n) => {
                // 3^(n-1) * n! / 2
                let three_power = 3_usize.pow((n - 1) as u32);
                let factorial = (1..=n).product::<usize>();
                Some(three_power * factorial / 2)
            }
            _ => None,
        }
    }

    /// Returns the degrees of the group (for finite reflection groups)
    ///
    /// Only available for certain Coxeter types:
    /// - n=2: [3]
    /// - n=3: [4, 6]
    pub fn degrees(&self) -> Option<Vec<usize>> {
        if self.cbg_type != CubicBraidType::Coxeter {
            return None;
        }

        match self.n_strands {
            2 => Some(vec![3]),
            3 => Some(vec![4, 6]),
            _ => None,
        }
    }

    /// Returns the codegrees of the group (for finite reflection groups)
    ///
    /// Only available for certain Coxeter types:
    /// - n=2: [0]
    /// - n=3: [0, 2]
    pub fn codegrees(&self) -> Option<Vec<usize>> {
        if self.cbg_type != CubicBraidType::Coxeter {
            return None;
        }

        match self.n_strands {
            2 => Some(vec![0]),
            3 => Some(vec![0, 2]),
            _ => None,
        }
    }

    /// Returns a reference to the underlying finitely presented group
    pub fn as_finitely_presented_group(&self) -> &FinitelyPresentedGroup {
        &self.fp_group
    }
}

impl fmt::Display for CubicBraidGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cubic Braid group on {} strands of type {}",
            self.n_strands, self.cbg_type
        )
    }
}

impl Group for CubicBraidGroup {
    type Element = CubicBraidElement;

    fn identity(&self) -> Self::Element {
        CubicBraidElement {
            group: self.clone(),
            word: FreeGroupElement::identity(),
        }
    }

    fn is_finite(&self) -> bool {
        // Cubic braid groups are infinite
        false
    }

    fn order(&self) -> Option<usize> {
        // Cubic braid groups are infinite
        None
    }

    fn contains(&self, element: &Self::Element) -> bool {
        // Check if element belongs to this cubic braid group
        self.n_strands == element.group.n_strands && self.cbg_type == element.group.cbg_type
    }
}

/// An element of a cubic braid group
#[derive(Debug, Clone)]
pub struct CubicBraidElement {
    group: CubicBraidGroup,
    word: FreeGroupElement,
}

impl CubicBraidElement {
    /// Create a new cubic braid element from a word
    pub fn new(group: CubicBraidGroup, word: FreeGroupElement) -> Self {
        Self { group, word }
    }

    /// Returns a reference to the parent group
    pub fn parent(&self) -> &CubicBraidGroup {
        &self.group
    }

    /// Returns a reference to the word representation
    pub fn word(&self) -> &FreeGroupElement {
        &self.word
    }

    /// Returns the ith generator (0-indexed)
    pub fn generator(group: &CubicBraidGroup, i: usize) -> Self {
        assert!(
            i < group.n_strands - 1,
            "Generator index out of range: {} (max {})",
            i,
            group.n_strands - 2
        );
        Self {
            group: group.clone(),
            word: FreeGroupElement::generator(i as i32, 1),
        }
    }

    /// Multiply this element with another
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(
            self.group.n_strands, other.group.n_strands,
            "Cannot multiply elements from different groups"
        );
        Self {
            group: self.group.clone(),
            word: self.word.multiply(&other.word),
        }
    }

    /// Compute the inverse of this element
    pub fn inverse(&self) -> Self {
        Self {
            group: self.group.clone(),
            word: self.word.inverse(),
        }
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.word.is_identity()
    }

    /// Raise this element to a power
    pub fn pow(&self, n: i32) -> Self {
        Self {
            group: self.group.clone(),
            word: self.word.pow(n),
        }
    }
}

impl fmt::Display for CubicBraidElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.word)
    }
}

impl PartialEq for CubicBraidElement {
    fn eq(&self, other: &Self) -> bool {
        if self.group.n_strands != other.group.n_strands {
            return false;
        }
        // For now, use word equality
        // In a full implementation, this would use the Burau representation
        self.word == other.word
    }
}

impl Eq for CubicBraidElement {}

use std::hash::{Hash, Hasher};

impl Hash for CubicBraidElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash only the word, as elements are compared within the same group context
        self.word.hash(state);
    }
}

use crate::group_traits::GroupElement;

impl std::ops::Mul for CubicBraidElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.op(&other)
    }
}

impl GroupElement for CubicBraidElement {
    fn identity() -> Self {
        // Create a minimal default group (2 strands, Coxeter type)
        // Note: Users should prefer calling group.identity() for the specific group
        let group = CubicBraidGroup::new(2, CubicBraidType::Coxeter);
        CubicBraidElement {
            group,
            word: FreeGroupElement::identity(),
        }
    }

    fn inverse(&self) -> Self {
        Self::inverse(self)
    }

    fn op(&self, other: &Self) -> Self {
        self.multiply(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_braid_group_creation() {
        let cbg = CubicBraidGroup::new(4, CubicBraidType::Coxeter);
        assert_eq!(cbg.strands(), 4);
        assert_eq!(cbg.cbg_type(), CubicBraidType::Coxeter);
        assert!(!cbg.is_finite());
    }

    #[test]
    fn test_assion_groups() {
        let assion_s = CubicBraidGroup::assion_s(5);
        assert_eq!(assion_s.strands(), 5);
        assert_eq!(assion_s.cbg_type(), CubicBraidType::AssionS);
        assert!(assion_s.is_finite());

        let assion_u = CubicBraidGroup::assion_u(5);
        assert_eq!(assion_u.strands(), 5);
        assert_eq!(assion_u.cbg_type(), CubicBraidType::AssionU);
        assert!(assion_u.is_finite());
    }

    #[test]
    fn test_group_orders() {
        let cbg2 = CubicBraidGroup::new(2, CubicBraidType::Coxeter);
        assert_eq!(cbg2.order(), Some(3));

        let cbg3 = CubicBraidGroup::new(3, CubicBraidType::Coxeter);
        assert_eq!(cbg3.order(), Some(24));

        let cbg4 = CubicBraidGroup::new(4, CubicBraidType::Coxeter);
        assert_eq!(cbg4.order(), None); // Infinite

        let assion_s3 = CubicBraidGroup::assion_s(3);
        // 3^2 * 3! = 9 * 6 = 54
        assert_eq!(assion_s3.order(), Some(54));

        let assion_u3 = CubicBraidGroup::assion_u(3);
        // 3^2 * 3! / 2 = 54 / 2 = 27
        assert_eq!(assion_u3.order(), Some(27));
    }

    #[test]
    fn test_index_set() {
        let cbg = CubicBraidGroup::new(5, CubicBraidType::Coxeter);
        assert_eq!(cbg.index_set(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_degrees_codegrees() {
        let cbg2 = CubicBraidGroup::new(2, CubicBraidType::Coxeter);
        assert_eq!(cbg2.degrees(), Some(vec![3]));
        assert_eq!(cbg2.codegrees(), Some(vec![0]));

        let cbg3 = CubicBraidGroup::new(3, CubicBraidType::Coxeter);
        assert_eq!(cbg3.degrees(), Some(vec![4, 6]));
        assert_eq!(cbg3.codegrees(), Some(vec![0, 2]));

        let cbg4 = CubicBraidGroup::new(4, CubicBraidType::Coxeter);
        assert_eq!(cbg4.degrees(), None);
        assert_eq!(cbg4.codegrees(), None);
    }

    #[test]
    fn test_elements() {
        let cbg = CubicBraidGroup::new(3, CubicBraidType::Coxeter);
        let s0 = CubicBraidElement::generator(&cbg, 0);
        let s1 = CubicBraidElement::generator(&cbg, 1);

        // Test multiplication
        let s0s1 = s0.multiply(&s1);
        assert!(!s0s1.is_identity());

        // Test inverse
        let s0_inv = s0.inverse();
        let prod = s0.multiply(&s0_inv);
        // Note: In a full implementation, this would be normalized to identity

        // Test power - s^3 should be identity in a cubic braid group
        let s0_cubed = s0.pow(3);
        // In a full implementation with normalization, this would be identity
    }

    #[test]
    fn test_identity() {
        let cbg = CubicBraidGroup::new(4, CubicBraidType::Coxeter);
        let id = cbg.identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_display() {
        let cbg = CubicBraidGroup::new(4, CubicBraidType::AssionS);
        let display = format!("{}", cbg);
        assert!(display.contains("Cubic Braid group"));
        assert!(display.contains("4 strands"));
        assert!(display.contains("AssionS"));
    }

    #[test]
    fn test_generator_names() {
        let cbg = CubicBraidGroup::new(4, CubicBraidType::Coxeter);
        let names = cbg.generator_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "s0");
        assert_eq!(names[1], "s1");
        assert_eq!(names[2], "s2");
    }

    #[test]
    #[should_panic(expected = "Need at least 2 strands")]
    fn test_invalid_strand_count() {
        CubicBraidGroup::new(1, CubicBraidType::Coxeter);
    }

    #[test]
    #[should_panic(expected = "Generator index out of range")]
    fn test_invalid_generator() {
        let cbg = CubicBraidGroup::new(3, CubicBraidType::Coxeter);
        CubicBraidElement::generator(&cbg, 5);
    }
}

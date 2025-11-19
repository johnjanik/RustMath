//! PARI Group Wrapper
//!
//! This module provides a wrapper for PARI group objects, particularly useful
//! for Galois groups computed via the PARI computer algebra system.
//!
//! # Note
//!
//! In SageMath, this wraps PARI's group functionality. In RustMath, we provide
//! a simplified implementation that stores group properties without direct PARI
//! integration. Full PARI integration would require FFI bindings to libpari.
//!
//! # Examples
//!
//! ```
//! use rustmath_groups::pari_group::PariGroup;
//!
//! // Create a PARI group representation
//! let g = PariGroup::new(5, "D(5)".to_string(), 10);
//! assert_eq!(g.degree(), 5);
//! assert_eq!(g.order(), 10);
//! ```

use std::fmt;

/// A wrapper for PARI group objects
///
/// This represents a group computed or manipulated via PARI, typically
/// arising from Galois group computations for polynomials.
#[derive(Debug, Clone)]
pub struct PariGroup {
    /// The degree of the group (degree of the associated polynomial)
    degree: usize,
    /// Label/description of the group (e.g., "S5", "A4", "D(10)")
    label: String,
    /// Order (cardinality) of the group
    order: usize,
    /// Transitive number (if applicable)
    transitive_number: Option<usize>,
    /// Signature: 1 if group is in alternating group, -1 otherwise
    signature: i32,
}

impl PariGroup {
    /// Create a new PARI group
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the group
    /// * `label` - Descriptive label for the group
    /// * `order` - Order of the group
    pub fn new(degree: usize, label: String, order: usize) -> Self {
        Self {
            degree,
            label,
            order,
            transitive_number: None,
            signature: -1, // Default: not in alternating group
        }
    }

    /// Create a new PARI group with full data
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the group
    /// * `label` - Descriptive label
    /// * `order` - Order of the group
    /// * `transitive_number` - Transitive group classification number
    /// * `signature` - 1 if in alternating group, -1 otherwise
    pub fn with_data(
        degree: usize,
        label: String,
        order: usize,
        transitive_number: Option<usize>,
        signature: i32,
    ) -> Self {
        Self {
            degree,
            label,
            order,
            transitive_number,
            signature,
        }
    }

    /// Get the degree of the group
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the label/description
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the order (cardinality) of the group
    pub fn order(&self) -> usize {
        self.order
    }

    /// Alias for order()
    pub fn cardinality(&self) -> usize {
        self.order
    }

    /// Get the transitive number
    pub fn transitive_number(&self) -> Option<usize> {
        self.transitive_number
    }

    /// Set the transitive number
    pub fn set_transitive_number(&mut self, n: usize) {
        self.transitive_number = Some(n);
    }

    /// Get the signature
    ///
    /// Returns 1 if the group is contained in the alternating group,
    /// -1 otherwise.
    pub fn signature(&self) -> i32 {
        self.signature
    }

    /// Set the signature
    pub fn set_signature(&mut self, sig: i32) {
        self.signature = sig;
    }

    /// Check if the group is in the alternating group
    pub fn is_in_alternating_group(&self) -> bool {
        self.signature == 1
    }

    /// Get the transitive label (e.g., "5T3" for degree 5, transitive number 3)
    pub fn transitive_label(&self) -> Option<String> {
        self.transitive_number
            .map(|n| format!("{}T{}", self.degree, n))
    }
}

impl fmt::Display for PariGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PARI group {} of degree {} and order {}",
            self.label, self.degree, self.order
        )
    }
}

impl PartialEq for PariGroup {
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree
            && self.order == other.order
            && self.label == other.label
            && self.transitive_number == other.transitive_number
    }
}

impl Eq for PariGroup {}

/// Create a PARI group for the symmetric group S_n
pub fn symmetric_pari_group(n: usize) -> PariGroup {
    let order = factorial(n);
    PariGroup::with_data(n, format!("S({})", n), order, None, -1)
}

/// Create a PARI group for the alternating group A_n
pub fn alternating_pari_group(n: usize) -> PariGroup {
    let order = factorial(n) / 2;
    PariGroup::with_data(n, format!("A({})", n), order, None, 1)
}

/// Create a PARI group for the dihedral group D_n
pub fn dihedral_pari_group(n: usize) -> PariGroup {
    let order = 2 * n;
    PariGroup::with_data(n, format!("D({})", n), order, None, if n % 2 == 0 { 1 } else { -1 })
}

/// Create a PARI group for the cyclic group C_n
pub fn cyclic_pari_group(n: usize) -> PariGroup {
    PariGroup::with_data(n, format!("C({})", n), n, None, 1)
}

/// Compute factorial
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pari_group_creation() {
        let g = PariGroup::new(5, "S5".to_string(), 120);
        assert_eq!(g.degree(), 5);
        assert_eq!(g.label(), "S5");
        assert_eq!(g.order(), 120);
        assert_eq!(g.cardinality(), 120);
    }

    #[test]
    fn test_pari_group_with_data() {
        let g = PariGroup::with_data(4, "A4".to_string(), 12, Some(3), 1);
        assert_eq!(g.degree(), 4);
        assert_eq!(g.order(), 12);
        assert_eq!(g.transitive_number(), Some(3));
        assert_eq!(g.signature(), 1);
        assert!(g.is_in_alternating_group());
    }

    #[test]
    fn test_transitive_label() {
        let mut g = PariGroup::new(5, "D(5)".to_string(), 10);
        g.set_transitive_number(3);
        assert_eq!(g.transitive_label(), Some("5T3".to_string()));
    }

    #[test]
    fn test_symmetric_pari_group() {
        let s5 = symmetric_pari_group(5);
        assert_eq!(s5.degree(), 5);
        assert_eq!(s5.order(), 120);
        assert_eq!(s5.label(), "S(5)");
    }

    #[test]
    fn test_alternating_pari_group() {
        let a4 = alternating_pari_group(4);
        assert_eq!(a4.degree(), 4);
        assert_eq!(a4.order(), 12);
        assert_eq!(a4.label(), "A(4)");
        assert!(a4.is_in_alternating_group());
    }

    #[test]
    fn test_dihedral_pari_group() {
        let d5 = dihedral_pari_group(5);
        assert_eq!(d5.degree(), 5);
        assert_eq!(d5.order(), 10);
        assert_eq!(d5.label(), "D(5)");
    }

    #[test]
    fn test_cyclic_pari_group() {
        let c7 = cyclic_pari_group(7);
        assert_eq!(c7.degree(), 7);
        assert_eq!(c7.order(), 7);
        assert_eq!(c7.label(), "C(7)");
    }

    #[test]
    fn test_equality() {
        let g1 = PariGroup::new(5, "S5".to_string(), 120);
        let g2 = PariGroup::new(5, "S5".to_string(), 120);
        let g3 = PariGroup::new(5, "A5".to_string(), 60);

        assert_eq!(g1, g2);
        assert_ne!(g1, g3);
    }

    #[test]
    fn test_signature() {
        let mut g = PariGroup::new(4, "Test".to_string(), 10);
        assert_eq!(g.signature(), -1);
        assert!(!g.is_in_alternating_group());

        g.set_signature(1);
        assert_eq!(g.signature(), 1);
        assert!(g.is_in_alternating_group());
    }

    #[test]
    fn test_display() {
        let g = PariGroup::new(5, "S5".to_string(), 120);
        let display = format!("{}", g);
        assert!(display.contains("PARI group"));
        assert!(display.contains("S5"));
        assert!(display.contains("degree 5"));
        assert!(display.contains("order 120"));
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(6), 720);
    }
}

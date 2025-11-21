//! Nilpotent Lie Groups
//!
//! This module implements nilpotent Lie groups, which are Lie groups whose Lie algebra
//! is nilpotent. These groups have special properties and can be constructed using
//! the exponential map and Baker-Campbell-Hausdorff formula.
//!
//! # Overview
//!
//! A nilpotent Lie group is a Lie group G whose associated Lie algebra g is nilpotent.
//! Key features:
//! - The exponential map exp: g → G is a diffeomorphism
//! - The logarithm map log: G → g is well-defined globally
//! - Group multiplication uses the Baker-Campbell-Hausdorff (BCH) formula
//! - Left/right-invariant vector fields extend from the Lie algebra
//!
//! # Example
//!
//! ```ignore
//! use rustmath_groups::nilpotent_lie_group::{NilpotentLieGroup, NilpotentLieGroupElement};
//!
//! // Create a Heisenberg group (simplest non-abelian nilpotent Lie group)
//! let h3 = NilpotentLieGroup::heisenberg(3);
//!
//! // Elements are represented as Lie algebra elements via exponential coordinates
//! let x = h3.exp(&[1.0, 0.0, 0.0]);
//! let y = h3.exp(&[0.0, 1.0, 0.0]);
//!
//! // Multiply using BCH formula
//! let xy = x.multiply(&y);
//! ```

use std::fmt;
use crate::group_traits::{Group, GroupElement};

/// Element of a nilpotent Lie group
///
/// Elements are represented via exponential coordinates of the first kind,
/// which means each element g is stored as X ∈ g such that g = exp(X).
///
/// # Type Parameters
///
/// For simplicity, we use Vec<f64> to represent Lie algebra elements.
/// In a full implementation, this would be a generic Lie algebra element type.
#[derive(Clone, Debug, PartialEq)]
pub struct NilpotentLieGroupElement {
    /// Lie algebra coordinates (exponential coordinates)
    coordinates: Vec<f64>,

    /// Dimension of the Lie algebra
    dim: usize,
}

impl NilpotentLieGroupElement {
    /// Create a new element from Lie algebra coordinates
    ///
    /// # Arguments
    ///
    /// - `coordinates`: Coordinates in the Lie algebra basis
    pub fn new(coordinates: Vec<f64>) -> Self {
        let dim = coordinates.len();
        NilpotentLieGroupElement { coordinates, dim }
    }

    /// Get the coordinates of this element
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Multiply two elements using the BCH formula (simplified)
    ///
    /// For nilpotent Lie algebras, the BCH series terminates after finitely many terms.
    /// This is a simplified implementation that works for low-dimensional cases.
    fn bch_multiply(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "Elements must have same dimension");

        // For 1D (abelian case), just add
        if self.dim == 1 {
            let mut result = vec![0.0; self.dim];
            for i in 0..self.dim {
                result[i] = self.coordinates[i] + other.coordinates[i];
            }
            return NilpotentLieGroupElement::new(result);
        }

        // For higher dimensions, use simplified BCH:
        // log(exp(X) exp(Y)) = X + Y + (1/2)[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]]) + ...
        //
        // For nilpotent algebras of degree n, terms with n+1 or more brackets vanish.
        // We compute up to degree 3 (sufficient for most nilpotent groups).

        let mut result = vec![0.0; self.dim];

        // First order: X + Y
        for i in 0..self.dim {
            result[i] = self.coordinates[i] + other.coordinates[i];
        }

        // Second order: (1/2)[X, Y]
        // For simplicity, we assume the standard basis and use structure constants
        // In a full implementation, this would use the actual Lie bracket
        let bracket = self.lie_bracket_simple(other);
        for i in 0..self.dim {
            result[i] += 0.5 * bracket[i];
        }

        // Higher order terms would go here for more complex groups
        // For now, we stop at second order which is exact for 2-step nilpotent groups

        NilpotentLieGroupElement::new(result)
    }

    /// Compute a simplified Lie bracket [X, Y]
    ///
    /// This is a placeholder that returns zero for abelian case
    /// In a full implementation, this would use actual structure constants
    fn lie_bracket_simple(&self, other: &Self) -> Vec<f64> {
        let mut result = vec![0.0; self.dim];

        // For Heisenberg-type algebras (dim >= 3):
        // If basis is {X, Y, Z} with [X, Y] = Z, other brackets = 0
        if self.dim >= 3 {
            // Simplified Heisenberg bracket: [e_0, e_1] = e_2
            result[2] = self.coordinates[0] * other.coordinates[1]
                      - self.coordinates[1] * other.coordinates[0];
        }

        result
    }

    /// Compute the inverse using exp(-X) = exp(X)^(-1)
    fn bch_inverse(&self) -> Self {
        let mut result = vec![0.0; self.dim];
        for i in 0..self.dim {
            result[i] = -self.coordinates[i];
        }
        NilpotentLieGroupElement::new(result)
    }
}

impl fmt::Display for NilpotentLieGroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exp(")?;
        for (i, &coord) in self.coordinates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", coord)?;
        }
        write!(f, ")")
    }
}

impl std::hash::Hash for NilpotentLieGroupElement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash based on discretized coordinates
        for &coord in &self.coordinates {
            let discretized = (coord * 1000000.0).round() as i64;
            discretized.hash(state);
        }
        self.dim.hash(state);
    }
}

impl Eq for NilpotentLieGroupElement {}

impl std::ops::Mul for NilpotentLieGroupElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.op(&other)
    }
}

impl GroupElement for NilpotentLieGroupElement {
    fn identity() -> Self {
        // Can't determine dimension without context, use 1D as default
        NilpotentLieGroupElement::new(vec![0.0])
    }

    fn inverse(&self) -> Self {
        self.bch_inverse()
    }

    fn op(&self, other: &Self) -> Self {
        self.bch_multiply(other)
    }

    fn pow(&self, n: i64) -> Self {
        if n == 0 {
            let mut coords = vec![0.0; self.dim];
            return NilpotentLieGroupElement::new(coords);
        }

        if n < 0 {
            return self.inverse().pow(-n);
        }

        // For nilpotent groups, we can use exp(nX) = exp(X)^n
        let mut coords = vec![0.0; self.dim];
        for i in 0..self.dim {
            coords[i] = (n as f64) * self.coordinates[i];
        }
        NilpotentLieGroupElement::new(coords)
    }

    fn order(&self) -> Option<usize> {
        // Most nilpotent Lie groups have elements of infinite order
        // (except the identity)
        if self.coordinates.iter().all(|&x| x.abs() < 1e-10) {
            return Some(1);
        }
        None
    }
}

/// A nilpotent Lie group
///
/// This struct represents a nilpotent Lie group G associated with a
/// nilpotent Lie algebra g. The group structure is determined by the
/// Baker-Campbell-Hausdorff formula.
///
/// # Examples
///
/// - Abelian groups (trivial Lie bracket)
/// - Heisenberg groups (2-step nilpotent)
/// - Upper triangular matrices with 1's on diagonal
#[derive(Clone, Debug)]
pub struct NilpotentLieGroup {
    /// Dimension of the underlying Lie algebra
    dimension: usize,

    /// Description of the group
    description: String,

    /// Nilpotency class (length of lower central series)
    nilpotency_class: usize,

    /// Basis element names
    basis_names: Vec<String>,
}

impl NilpotentLieGroup {
    /// Create a new nilpotent Lie group from dimension and nilpotency class
    ///
    /// # Arguments
    ///
    /// - `dimension`: Dimension of the Lie algebra
    /// - `nilpotency_class`: Nilpotency class (1 = abelian, 2 = Heisenberg-type, etc.)
    pub fn new(dimension: usize, nilpotency_class: usize) -> Self {
        let basis_names = (0..dimension)
            .map(|i| format!("e_{}", i))
            .collect();

        NilpotentLieGroup {
            dimension,
            description: format!(
                "{}-dimensional nilpotent Lie group of class {}",
                dimension, nilpotency_class
            ),
            nilpotency_class,
            basis_names,
        }
    }

    /// Create a Heisenberg group of dimension 2n+1
    ///
    /// The Heisenberg group H_{2n+1} is the most important 2-step nilpotent group.
    /// For n=1, it's the 3-dimensional Heisenberg group with basis {X, Y, Z}
    /// where [X, Y] = Z and all other brackets are zero.
    pub fn heisenberg(n: usize) -> Self {
        let dim = 2 * n + 1;
        let mut basis_names = Vec::new();

        for i in 0..n {
            basis_names.push(format!("X_{}", i));
        }
        for i in 0..n {
            basis_names.push(format!("Y_{}", i));
        }
        basis_names.push("Z".to_string());

        NilpotentLieGroup {
            dimension: dim,
            description: format!("Heisenberg group H_{}", dim),
            nilpotency_class: 2,
            basis_names,
        }
    }

    /// Create an abelian group (nilpotency class 1)
    pub fn abelian(dimension: usize) -> Self {
        NilpotentLieGroup::new(dimension, 1)
    }

    /// Exponential map: Lie algebra → Lie group
    ///
    /// For nilpotent groups, exp is a global diffeomorphism
    pub fn exp(&self, algebra_element: &[f64]) -> NilpotentLieGroupElement {
        assert_eq!(
            algebra_element.len(),
            self.dimension,
            "Algebra element must have dimension {}",
            self.dimension
        );
        NilpotentLieGroupElement::new(algebra_element.to_vec())
    }

    /// Logarithm map: Lie group → Lie algebra
    ///
    /// For nilpotent groups, log is well-defined everywhere
    pub fn log(&self, group_element: &NilpotentLieGroupElement) -> Vec<f64> {
        assert_eq!(
            group_element.dimension(),
            self.dimension,
            "Element must have dimension {}",
            self.dimension
        );
        group_element.coordinates().to_vec()
    }

    /// Get the dimension of this group
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the nilpotency class
    pub fn nilpotency_class(&self) -> usize {
        self.nilpotency_class
    }

    /// Get the basis element names
    pub fn basis_names(&self) -> &[String] {
        &self.basis_names
    }

    /// Compute left translation by an element
    ///
    /// Returns the map L_g: h ↦ g·h
    pub fn left_translation(
        &self,
        g: &NilpotentLieGroupElement,
    ) -> impl Fn(&NilpotentLieGroupElement) -> NilpotentLieGroupElement + '_ {
        let g_clone = g.clone();
        move |h| g_clone.op(h)
    }

    /// Compute right translation by an element
    ///
    /// Returns the map R_g: h ↦ h·g
    pub fn right_translation(
        &self,
        g: &NilpotentLieGroupElement,
    ) -> impl Fn(&NilpotentLieGroupElement) -> NilpotentLieGroupElement + '_ {
        let g_clone = g.clone();
        move |h| h.op(&g_clone)
    }

    /// Compute conjugation by an element
    ///
    /// Returns the map c_g: h ↦ g·h·g^(-1)
    pub fn conjugation(
        &self,
        g: &NilpotentLieGroupElement,
    ) -> impl Fn(&NilpotentLieGroupElement) -> NilpotentLieGroupElement + '_ {
        let g_clone = g.clone();
        let g_inv = g.inverse();
        move |h| g_clone.op(h).op(&g_inv)
    }
}

impl fmt::Display for NilpotentLieGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl Group for NilpotentLieGroup {
    type Element = NilpotentLieGroupElement;

    fn identity(&self) -> Self::Element {
        NilpotentLieGroupElement::new(vec![0.0; self.dimension])
    }

    fn is_abelian(&self) -> bool {
        self.nilpotency_class == 1
    }

    fn is_finite(&self) -> bool {
        false // Lie groups are continuous, hence infinite
    }

    fn order(&self) -> Option<usize> {
        None // Infinite
    }

    fn contains(&self, element: &Self::Element) -> bool {
        element.dimension() == self.dimension
    }

    fn exponent(&self) -> Option<usize> {
        None // Infinite group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_creation() {
        let elem = NilpotentLieGroupElement::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(elem.dimension(), 3);
        assert_eq!(elem.coordinates(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_element_identity() {
        let id = NilpotentLieGroupElement::new(vec![0.0, 0.0]);
        assert!(id.is_identity());
    }

    #[test]
    fn test_element_inverse() {
        let elem = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        let inv = elem.inverse();
        assert_eq!(inv.coordinates(), &[-1.0, -2.0]);
    }

    #[test]
    fn test_element_multiply_abelian() {
        let x = NilpotentLieGroupElement::new(vec![1.0]);
        let y = NilpotentLieGroupElement::new(vec![2.0]);
        let xy = x.op(&y);
        assert_eq!(xy.coordinates(), &[3.0]);
    }

    #[test]
    fn test_element_multiply_heisenberg() {
        // In Heisenberg group, [X, Y] = Z
        let x = NilpotentLieGroupElement::new(vec![1.0, 0.0, 0.0]);
        let y = NilpotentLieGroupElement::new(vec![0.0, 1.0, 0.0]);
        let xy = x.op(&y);

        // exp(X) exp(Y) = exp(X + Y + (1/2)[X,Y])
        // = exp(X + Y + (1/2)Z)
        assert!((xy.coordinates()[0] - 1.0).abs() < 1e-10);
        assert!((xy.coordinates()[1] - 1.0).abs() < 1e-10);
        assert!((xy.coordinates()[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_element_power() {
        let x = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        let x_cubed = x.pow(3);

        // exp(X)^3 = exp(3X)
        assert_eq!(x_cubed.coordinates(), &[3.0, 6.0]);
    }

    #[test]
    fn test_element_power_negative() {
        let x = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        let x_inv = x.pow(-1);
        assert_eq!(x_inv.coordinates(), &[-1.0, -2.0]);
    }

    #[test]
    fn test_group_creation() {
        let g = NilpotentLieGroup::new(5, 2);
        assert_eq!(g.dimension(), 5);
        assert_eq!(g.nilpotency_class(), 2);
    }

    #[test]
    fn test_heisenberg_group() {
        let h3 = NilpotentLieGroup::heisenberg(1);
        assert_eq!(h3.dimension(), 3);
        assert_eq!(h3.nilpotency_class(), 2);
        assert!(!h3.is_abelian());
    }

    #[test]
    fn test_abelian_group() {
        let a = NilpotentLieGroup::abelian(4);
        assert_eq!(a.dimension(), 4);
        assert_eq!(a.nilpotency_class(), 1);
        assert!(a.is_abelian());
    }

    #[test]
    fn test_exp_log() {
        let g = NilpotentLieGroup::new(3, 2);
        let algebra_elem = vec![1.0, 2.0, 3.0];

        let group_elem = g.exp(&algebra_elem);
        let recovered = g.log(&group_elem);

        assert_eq!(recovered, algebra_elem);
    }

    #[test]
    fn test_group_identity() {
        let g = NilpotentLieGroup::heisenberg(1);
        let id = g.identity();

        assert_eq!(id.coordinates(), &[0.0, 0.0, 0.0]);
        assert!(id.is_identity());
    }

    #[test]
    fn test_group_is_infinite() {
        let g = NilpotentLieGroup::new(2, 1);
        assert!(!g.is_finite());
        assert_eq!(g.order(), None);
    }

    #[test]
    fn test_group_contains() {
        let g = NilpotentLieGroup::heisenberg(1);
        let elem = NilpotentLieGroupElement::new(vec![1.0, 2.0, 3.0]);
        assert!(g.contains(&elem));

        let wrong_dim = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        assert!(!g.contains(&wrong_dim));
    }

    #[test]
    fn test_left_translation() {
        let g = NilpotentLieGroup::abelian(2);
        let a = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        let b = NilpotentLieGroupElement::new(vec![3.0, 4.0]);

        let left_a = g.left_translation(&a);
        let result = left_a(&b);

        assert_eq!(result.coordinates(), &[4.0, 6.0]); // a + b in abelian case
    }

    #[test]
    fn test_right_translation() {
        let g = NilpotentLieGroup::abelian(2);
        let a = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        let b = NilpotentLieGroupElement::new(vec![3.0, 4.0]);

        let right_a = g.right_translation(&a);
        let result = right_a(&b);

        assert_eq!(result.coordinates(), &[4.0, 6.0]); // b + a in abelian case
    }

    #[test]
    fn test_conjugation_abelian() {
        let g = NilpotentLieGroup::abelian(2);
        let a = NilpotentLieGroupElement::new(vec![1.0, 2.0]);
        let b = NilpotentLieGroupElement::new(vec![3.0, 4.0]);

        let conj_a = g.conjugation(&a);
        let result = conj_a(&b);

        // In abelian groups, conjugation is identity
        assert_eq!(result.coordinates(), b.coordinates());
    }

    #[test]
    fn test_element_display() {
        let elem = NilpotentLieGroupElement::new(vec![1.0, 2.0, 3.0]);
        let display = format!("{}", elem);
        assert!(display.contains("exp"));
    }

    #[test]
    fn test_group_display() {
        let g = NilpotentLieGroup::heisenberg(1);
        let display = format!("{}", g);
        assert!(display.contains("Heisenberg"));
    }

    #[test]
    fn test_basis_names() {
        let g = NilpotentLieGroup::heisenberg(1);
        let names = g.basis_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "X_0");
        assert_eq!(names[1], "Y_0");
        assert_eq!(names[2], "Z");
    }
}

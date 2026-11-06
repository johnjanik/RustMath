//! Nilpotent Lie Groups
//!
//! This module implements simply connected nilpotent Lie groups, which are Lie
//! groups whose Lie algebra is nilpotent, in exponential coordinates of the first
//! kind: an element `exp(X)` is stored as the coordinate vector of `X ∈ g`.
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
//! Multiplication needs the Lie bracket of g. It is known exactly for the
//! brackets in [`LieBracket`] (abelian, Heisenberg `h_{2n+1}`); both are at most
//! 2-step nilpotent, where the BCH series `X + Y + ½[X,Y]` is exact. A group whose
//! bracket the constructor's data do not determine ([`NilpotentLieGroup::new`]
//! with class ≥ 2 outside dimension 3) refuses to multiply in dimension ≥ 3
//! rather than guess a bracket.
//!
//! MAGMA: nilpotent Lie algebras are Handbook ch. 100 (Lie Algebras, §100.18,
//! `NilpotentLieAlgebra`; `rustmath-lava/docs/Chapter_100_Lie_Algebras.md`).
//! BCH formula: e.g. Hall, *Lie Groups, Lie Algebras, and Representations*, §5.
//!
//! # Example
//!
//! ```
//! use rustmath_groups::group_traits::GroupElement;
//! use rustmath_groups::nilpotent_lie_group::NilpotentLieGroup;
//!
//! // The 3-dimensional Heisenberg group, basis {X, Y, Z} with [X, Y] = Z
//! let h3 = NilpotentLieGroup::heisenberg(1);
//!
//! let x = h3.exp(&[1.0, 0.0, 0.0]);
//! let y = h3.exp(&[0.0, 1.0, 0.0]);
//!
//! // exp(X) exp(Y) = exp(X + Y + ½[X, Y]) = exp(X + Y + ½Z)
//! assert_eq!(x.op(&y).coordinates(), &[1.0, 1.0, 0.5]);
//! assert_eq!(y.op(&x).coordinates(), &[1.0, 1.0, -0.5]);
//! ```

use std::fmt;
use crate::group_traits::{Group, GroupElement};

/// A Lie bracket on `ℝ^d`, in the basis of the group it belongs to.
///
/// Only brackets whose structure constants are known exactly are representable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LieBracket {
    /// `[x, y] = 0` for all `x, y` (nilpotency class 1).
    Abelian,
    /// The Heisenberg algebra `h_{2n+1}` in the basis
    /// `X_0, …, X_{n-1}, Y_0, …, Y_{n-1}, Z`: `[X_i, Y_i] = Z`, all other basis
    /// brackets zero (nilpotency class 2 for `n ≥ 1`).
    Heisenberg { n: usize },
}

impl LieBracket {
    /// Dimension of the algebra this bracket is defined on, if fixed.
    fn fixed_dimension(&self) -> Option<usize> {
        match self {
            LieBracket::Abelian => None,
            LieBracket::Heisenberg { n } => Some(2 * n + 1),
        }
    }

    /// `[x, y]` for coordinate vectors `x, y` of equal length.
    ///
    /// # Panics
    /// If the lengths differ, or do not match a Heisenberg algebra's dimension.
    pub fn bracket(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), y.len(), "bracket: vectors must have the same dimension");
        let mut out = vec![0.0; x.len()];
        match *self {
            LieBracket::Abelian => {}
            LieBracket::Heisenberg { n } => {
                assert_eq!(
                    x.len(),
                    2 * n + 1,
                    "Heisenberg bracket h_{} needs vectors of dimension {}",
                    2 * n + 1,
                    2 * n + 1
                );
                // [Σ x_i X_i + x'_i Y_i + ..., Σ y_i X_i + y'_i Y_i + ...]
                //   = Σ_i (x_i y'_i − x'_i y_i) Z
                let omega: f64 = (0..n).map(|i| x[i] * y[n + i] - x[n + i] * y[i]).sum();
                out[2 * n] = omega;
            }
        }
        out
    }
}

/// Element of a nilpotent Lie group
///
/// Elements are represented via exponential coordinates of the first kind,
/// which means each element g is stored as X ∈ g such that g = exp(X).
///
/// An element made by [`NilpotentLieGroup::exp`] (or [`NilpotentLieGroup::identity`])
/// carries its group's [`LieBracket`]. An element made by [`new`](Self::new) has
/// no bracket attached: multiplying it with an element that has one uses that
/// bracket; multiplying two such bare elements is only defined in dimension ≤ 2,
/// where every nilpotent Lie algebra is abelian.
///
/// # Equality
///
/// The derived `PartialEq` compares the coordinates **and** the attached
/// bracket, so elements of groups with different known brackets are never
/// equal. In particular a bare `NilpotentLieGroupElement::new(v)` (no bracket)
/// is **not** `==` `G.exp(&v)` for a group `G` with a known bracket, although
/// both have the coordinates `v`; compare [`coordinates`](Self::coordinates)
/// to compare coordinates only. (Treating "no bracket" as equal to every
/// bracket would make `==` intransitive.) `Hash` ignores the bracket, which is
/// consistent with this `==`.
///
/// A group whose bracket is undetermined ([`NilpotentLieGroup::new`] with
/// class ≥ 2 outside dimension 3) attaches no bracket, so its elements compare
/// like bare elements, by coordinates only: `NilpotentLieGroup::new(4, 2).exp(&v)`
/// `==` `NilpotentLieGroup::new(4, 3).exp(&v)` (and `== new(v)`), although the
/// two groups are not isomorphic. `==` therefore does not tell such groups
/// apart.
///
/// Coordinates are compared exactly as stored `f64` values, as are
/// [`is_identity`](GroupElement::is_identity) and
/// [`order`](GroupElement::order): see there for round-off.
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

    /// Lie bracket of the group this element belongs to (`None` = not attached)
    bracket: Option<LieBracket>,
}

impl NilpotentLieGroupElement {
    /// Create a new element from Lie algebra coordinates, with no group (and
    /// hence no Lie bracket) attached.
    ///
    /// # Arguments
    ///
    /// - `coordinates`: Coordinates in the Lie algebra basis
    pub fn new(coordinates: Vec<f64>) -> Self {
        let dim = coordinates.len();
        NilpotentLieGroupElement {
            coordinates,
            dim,
            bracket: None,
        }
    }

    fn with_bracket(coordinates: Vec<f64>, bracket: Option<LieBracket>) -> Self {
        let dim = coordinates.len();
        if let Some(d) = bracket.and_then(|b| b.fixed_dimension()) {
            assert_eq!(dim, d, "element dimension {} does not match its Lie bracket ({})", dim, d);
        }
        NilpotentLieGroupElement {
            coordinates,
            dim,
            bracket,
        }
    }

    /// Get the coordinates of this element
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// The Lie bracket attached to this element, if any.
    pub fn bracket(&self) -> Option<LieBracket> {
        self.bracket
    }

    /// The bracket under which `self · other` is computed.
    ///
    /// # Panics
    /// If the two elements carry different brackets (different groups).
    fn common_bracket(&self, other: &Self) -> Option<LieBracket> {
        match (self.bracket, other.bracket) {
            (Some(a), Some(b)) => {
                assert_eq!(a, b, "cannot multiply elements of different nilpotent Lie groups");
                Some(a)
            }
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Multiply two elements using the BCH formula.
    ///
    /// `log(exp(X) exp(Y)) = X + Y + ½[X,Y] + (1/12)([X,[X,Y]] + [Y,[Y,X]]) + …`.
    /// Every [`LieBracket`] is at most 2-step nilpotent, so all brackets of two
    /// or more brackets vanish and `X + Y + ½[X,Y]` is exact.
    ///
    /// # Panics
    /// If neither element has a bracket attached and the dimension is ≥ 3 (the
    /// product is then not determined), or if the brackets differ.
    fn bch_multiply(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "Elements must have same dimension");

        let attached = self.common_bracket(other);
        let bracket = match attached {
            Some(b) => b,
            // Every nilpotent Lie algebra of dimension ≤ 2 is abelian.
            None if self.dim <= 2 => LieBracket::Abelian,
            None => unimplemented!(
                "product of {}-dimensional nilpotent Lie group elements with no Lie \
                 bracket attached: the bracket is not determined in dimension ≥ 3. \
                 Create the elements with NilpotentLieGroup::exp of a group whose \
                 bracket is known (abelian, heisenberg)",
                self.dim
            ),
        };

        let mut result: Vec<f64> = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| a + b)
            .collect();
        let br = bracket.bracket(&self.coordinates, &other.coordinates);
        for (r, b) in result.iter_mut().zip(br) {
            *r += 0.5 * b;
        }

        NilpotentLieGroupElement::with_bracket(result, attached)
    }

    /// Compute the inverse using exp(-X) = exp(X)^(-1)
    fn bch_inverse(&self) -> Self {
        let result = self.coordinates.iter().map(|&c| -c).collect();
        NilpotentLieGroupElement::with_bracket(result, self.bracket)
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
    /// The trait's context-free identity. Its dimension cannot be known here, so
    /// it is the 1-dimensional `exp(0)`; use [`NilpotentLieGroup::identity`] for
    /// the identity of a specific group. [`is_identity`](Self::is_identity) is
    /// dimension-independent.
    fn identity() -> Self {
        NilpotentLieGroupElement::new(vec![0.0])
    }

    fn inverse(&self) -> Self {
        self.bch_inverse()
    }

    fn op(&self, other: &Self) -> Self {
        self.bch_multiply(other)
    }

    fn pow(&self, n: i64) -> Self {
        // exp(X)^n = exp(nX): powers of one element commute, so BCH reduces to a sum.
        let coords = self.coordinates.iter().map(|&c| (n as f64) * c).collect();
        NilpotentLieGroupElement::with_bracket(coords, self.bracket)
    }

    /// `exp(X)` is the identity iff `X = 0` (exp is a bijection), in any dimension.
    ///
    /// The test is exact on the **stored** `f64` coordinates: every coordinate
    /// must be `0.0` (or `-0.0`). Floating-point round-off is not the identity:
    /// a product that is the identity in exact arithmetic but leaves a residue
    /// such as `1e-17` in some coordinate is reported as not the identity. No
    /// tolerance is applied, because none would be correct for every scale of
    /// input.
    fn is_identity(&self) -> bool {
        self.coordinates.iter().all(|&x| x == 0.0)
    }

    /// A simply connected nilpotent Lie group is torsion-free:
    /// `exp(X)^k = exp(kX) = e` iff `kX = 0` iff `X = 0`. So the order is 1 for
    /// the identity and infinite (`None`) otherwise.
    ///
    /// Like [`is_identity`](Self::is_identity), this is decided on the stored
    /// `f64` coordinates exactly: an element whose coordinates carry round-off
    /// residue (of an exact-arithmetic identity) has order `None` (infinite),
    /// which is true of the stored element, not of the intended one.
    fn order(&self) -> Option<usize> {
        if self.is_identity() {
            Some(1)
        } else {
            None
        }
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

    /// Lie bracket in the basis `basis_names` (`None` = not determined)
    bracket: Option<LieBracket>,
}

impl NilpotentLieGroup {
    /// Create a new nilpotent Lie group from dimension and nilpotency class
    ///
    /// Dimension and class alone determine the Lie bracket only in two cases,
    /// which are used: class ≤ 1 (abelian), and dimension 3 with class 2 (the
    /// Heisenberg algebra `h_3`, the only non-abelian 3-dimensional nilpotent Lie
    /// algebra), with `[e_0, e_1] = e_2`. Otherwise the bracket is unknown and
    /// products of elements of dimension ≥ 3 are refused (`unimplemented!`).
    ///
    /// # Arguments
    ///
    /// - `dimension`: Dimension of the Lie algebra
    /// - `nilpotency_class`: Nilpotency class (1 = abelian, 2 = Heisenberg-type, etc.)
    pub fn new(dimension: usize, nilpotency_class: usize) -> Self {
        let basis_names = (0..dimension)
            .map(|i| format!("e_{}", i))
            .collect();

        let bracket = if nilpotency_class <= 1 {
            Some(LieBracket::Abelian)
        } else if dimension == 3 && nilpotency_class == 2 {
            Some(LieBracket::Heisenberg { n: 1 })
        } else {
            None
        };

        NilpotentLieGroup {
            dimension,
            description: format!(
                "{}-dimensional nilpotent Lie group of class {}",
                dimension, nilpotency_class
            ),
            nilpotency_class,
            basis_names,
            bracket,
        }
    }

    /// Create a Heisenberg group of dimension 2n+1
    ///
    /// The Heisenberg group H_{2n+1} is the most important 2-step nilpotent group.
    /// Its basis is `X_0, …, X_{n-1}, Y_0, …, Y_{n-1}, Z` with `[X_i, Y_i] = Z`
    /// and all other brackets zero. For n=1, it's the 3-dimensional Heisenberg
    /// group with basis {X, Y, Z} where [X, Y] = Z.
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
            bracket: Some(LieBracket::Heisenberg { n }),
        }
    }

    /// Create an abelian group (nilpotency class 1)
    pub fn abelian(dimension: usize) -> Self {
        NilpotentLieGroup::new(dimension, 1)
    }

    /// The Lie bracket of this group's algebra, if determined.
    pub fn bracket(&self) -> Option<LieBracket> {
        self.bracket
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
        NilpotentLieGroupElement::with_bracket(algebra_element.to_vec(), self.bracket)
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

    /// `e` viewed in this group: attaches this group's bracket to a bare element.
    fn adopt(&self, e: &NilpotentLieGroupElement) -> NilpotentLieGroupElement {
        assert_eq!(e.dimension(), self.dimension, "Element must have dimension {}", self.dimension);
        let mut e = e.clone();
        if e.bracket.is_none() {
            e.bracket = self.bracket;
        }
        e
    }

    /// Product `a · b` in this group (BCH with this group's bracket).
    ///
    /// # Panics
    /// If an element belongs to a group with a different bracket, or (via
    /// `unimplemented!`) if this group's bracket is unknown and the dimension is ≥ 3.
    pub fn multiply(
        &self,
        a: &NilpotentLieGroupElement,
        b: &NilpotentLieGroupElement,
    ) -> NilpotentLieGroupElement {
        self.adopt(a).op(&self.adopt(b))
    }

    /// Compute left translation by an element
    ///
    /// Returns the map L_g: h ↦ g·h
    pub fn left_translation(
        &self,
        g: &NilpotentLieGroupElement,
    ) -> impl Fn(&NilpotentLieGroupElement) -> NilpotentLieGroupElement + '_ {
        let g = self.adopt(g);
        move |h| self.multiply(&g, h)
    }

    /// Compute right translation by an element
    ///
    /// Returns the map R_g: h ↦ h·g
    pub fn right_translation(
        &self,
        g: &NilpotentLieGroupElement,
    ) -> impl Fn(&NilpotentLieGroupElement) -> NilpotentLieGroupElement + '_ {
        let g = self.adopt(g);
        move |h| self.multiply(h, &g)
    }

    /// Compute conjugation by an element
    ///
    /// Returns the map c_g: h ↦ g·h·g^(-1)
    pub fn conjugation(
        &self,
        g: &NilpotentLieGroupElement,
    ) -> impl Fn(&NilpotentLieGroupElement) -> NilpotentLieGroupElement + '_ {
        let g = self.adopt(g);
        let g_inv = g.inverse();
        move |h| self.multiply(&self.multiply(&g, h), &g_inv)
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
        NilpotentLieGroupElement::with_bracket(vec![0.0; self.dimension], self.bracket)
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

    /// Elements of the right dimension that are bare or carry this group's bracket.
    fn contains(&self, element: &Self::Element) -> bool {
        element.dimension() == self.dimension
            && (element.bracket.is_none() || element.bracket == self.bracket)
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
        // In Heisenberg group, [X, Y] = Z. The elements must come from the group:
        // a bare 3-dimensional coordinate vector does not determine the bracket.
        let h3 = NilpotentLieGroup::heisenberg(1);
        let x = h3.exp(&[1.0, 0.0, 0.0]);
        let y = h3.exp(&[0.0, 1.0, 0.0]);
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

    #[test]
    fn test_identity_any_dimension_and_order() {
        for d in 0..6 {
            let id = NilpotentLieGroupElement::new(vec![0.0; d]);
            assert!(id.is_identity());
            assert_eq!(id.order(), Some(1));
        }
        // exp(X) with X != 0 is not the identity and has infinite order, however
        // small X is (torsion-free group).
        let tiny = NilpotentLieGroupElement::new(vec![1e-12, 0.0]);
        assert!(!tiny.is_identity());
        assert_eq!(tiny.order(), None);
        // x · x^{-1} is the identity.
        let h3 = NilpotentLieGroup::heisenberg(1);
        let x = h3.exp(&[1.5, -2.0, 0.25]);
        assert!(x.op(&x.inverse()).is_identity());
        assert!(h3.identity().op(&x) == x);
    }

    #[test]
    fn test_identity_is_exact_on_stored_coordinates() {
        // x y y^{-1} x^{-1} is the identity in exact arithmetic, but in f64 the
        // BCH sums leave round-off in the coordinates; the documented test is
        // exact on the stored coordinates, so this element is not the identity.
        let h3 = NilpotentLieGroup::heisenberg(1);
        let x = h3.exp(&[0.1, 0.2, 0.3]);
        let y = h3.exp(&[0.2, 0.7, 0.9]);
        let r = x.op(&y).op(&y.inverse()).op(&x.inverse());
        assert!(r.coordinates().iter().all(|c| c.abs() < 1e-15), "{:?}", r.coordinates());
        assert!(r.coordinates().iter().any(|&c| c != 0.0), "{:?}", r.coordinates());
        assert!(!r.is_identity());
        assert_eq!(r.order(), None);
    }

    #[test]
    fn test_equality_includes_the_bracket() {
        // Same coordinates, but `new` attaches no bracket while `exp` attaches
        // the Heisenberg bracket: the derived `==` tells them apart.
        let h3 = NilpotentLieGroup::heisenberg(1);
        let v = [1.0, 2.0, 3.0];
        let bare = NilpotentLieGroupElement::new(v.to_vec());
        let in_h3 = h3.exp(&v);
        assert_eq!(bare.coordinates(), in_h3.coordinates());
        assert_eq!(bare.bracket(), None);
        assert_eq!(in_h3.bracket(), Some(LieBracket::Heisenberg { n: 1 }));
        assert_ne!(bare, in_h3);
        // Elements of different groups with the same coordinates differ too.
        assert_ne!(NilpotentLieGroup::abelian(3).exp(&v), in_h3);
        assert_eq!(h3.exp(&v), in_h3);
    }

    #[test]
    fn test_equality_without_known_bracket_is_by_coordinates() {
        // Dimension 4 with class 2 (h_3 ⊕ R) and class 3 (the filiform n_4) are
        // non-isomorphic, but `new` determines neither bracket, so it attaches
        // none: their elements compare by coordinates only, as documented.
        let g2 = NilpotentLieGroup::new(4, 2);
        let g3 = NilpotentLieGroup::new(4, 3);
        assert_eq!(g2.bracket(), None);
        assert_eq!(g3.bracket(), None);
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(g2.exp(&v), g3.exp(&v));
        assert_eq!(g2.exp(&v), NilpotentLieGroupElement::new(v.to_vec()));
        assert_ne!(g2.exp(&v), g3.exp(&[1.0, 2.0, 3.0, 5.0]));
    }

    #[test]
    fn test_abelian_group_is_commutative() {
        // Regression: elements of dimension >= 3 used to get a hard-coded
        // Heisenberg bracket [e_0, e_1] = e_2 whatever their group was.
        let a = NilpotentLieGroup::abelian(3);
        assert!(a.is_abelian());
        let x = a.exp(&[1.0, 0.0, 0.0]);
        let y = a.exp(&[0.0, 1.0, 0.0]);
        assert_eq!(x.op(&y).coordinates(), &[1.0, 1.0, 0.0]);
        assert_eq!(x.op(&y), y.op(&x));
    }

    #[test]
    fn test_heisenberg_h5_bracket() {
        // H_5: basis X_0, X_1, Y_0, Y_1, Z with [X_i, Y_i] = Z only.
        let h5 = NilpotentLieGroup::heisenberg(2);
        let e = |i: usize| {
            let mut v = vec![0.0; 5];
            v[i] = 1.0;
            h5.exp(&v)
        };
        let (x0, x1, y0, y1) = (e(0), e(1), e(2), e(3));
        // [X_0, X_1] = 0: they commute.
        assert_eq!(x0.op(&x1).coordinates(), &[1.0, 1.0, 0.0, 0.0, 0.0]);
        // [X_0, Y_0] = Z
        assert_eq!(x0.op(&y0).coordinates(), &[1.0, 0.0, 1.0, 0.0, 0.5]);
        // [X_1, Y_1] = Z, [X_0, Y_1] = 0
        assert_eq!(x1.op(&y1).coordinates(), &[0.0, 1.0, 0.0, 1.0, 0.5]);
        assert_eq!(x0.op(&y1).coordinates(), &[1.0, 0.0, 0.0, 1.0, 0.0]);
        // Group commutator x y x^{-1} y^{-1} = exp([X, Y]) in a 2-step group.
        let c = x0.commutator(&y0);
        assert_eq!(c.coordinates(), &[0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// Independent check of the BCH product against the matrix model of
    /// H_{2n+1}: X = (x, y, z) ↦ the (n+2)×(n+2) nilpotent matrix with first row
    /// (0, x, z), last column (z, y, 0), so that [X, Y] = XY − YX = ω(X, Y)·E.
    /// exp(N) = I + N + N²/2 and log(I + N) = N − N²/2 exactly (N³ = 0).
    #[test]
    fn test_heisenberg_bch_matches_matrix_model() {
        fn to_matrix(n: usize, v: &[f64]) -> Vec<Vec<f64>> {
            let m = n + 2;
            let mut a = vec![vec![0.0; m]; m];
            for i in 0..n {
                a[0][1 + i] = v[i];
                a[1 + i][m - 1] = v[n + i];
            }
            a[0][m - 1] = v[2 * n];
            a
        }
        fn from_matrix(n: usize, a: &[Vec<f64>]) -> Vec<f64> {
            let m = n + 2;
            let mut v = vec![0.0; 2 * n + 1];
            for i in 0..n {
                v[i] = a[0][1 + i];
                v[n + i] = a[1 + i][m - 1];
            }
            v[2 * n] = a[0][m - 1];
            v
        }
        fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let m = a.len();
            let mut c = vec![vec![0.0; m]; m];
            for i in 0..m {
                for k in 0..m {
                    for j in 0..m {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            c
        }
        fn lin(a: &[Vec<f64>], sa: f64, b: &[Vec<f64>], sb: f64) -> Vec<Vec<f64>> {
            a.iter()
                .zip(b)
                .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| sa * x + sb * y).collect())
                .collect()
        }
        fn mat_exp(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let m = a.len();
            let mut id = vec![vec![0.0; m]; m];
            for i in 0..m {
                id[i][i] = 1.0;
            }
            let a2 = mat_mul(a, a);
            lin(&lin(&id, 1.0, a, 1.0), 1.0, &a2, 0.5)
        }
        fn mat_log(g: &[Vec<f64>]) -> Vec<Vec<f64>> {
            let m = g.len();
            let mut nmat: Vec<Vec<f64>> = g.to_vec();
            for i in 0..m {
                nmat[i][i] -= 1.0;
            }
            let n2 = mat_mul(&nmat, &nmat);
            lin(&nmat, 1.0, &n2, -0.5)
        }

        for n in 1..=3usize {
            let h = NilpotentLieGroup::heisenberg(n);
            let d = 2 * n + 1;
            let xs: Vec<f64> = (0..d).map(|i| 0.7 * (i as f64) - 1.3).collect();
            let ys: Vec<f64> = (0..d).map(|i| 2.1 - 0.4 * (i as f64) * (i as f64)).collect();
            let got = h.exp(&xs).op(&h.exp(&ys));
            let expected = from_matrix(
                n,
                &mat_log(&mat_mul(&mat_exp(&to_matrix(n, &xs)), &mat_exp(&to_matrix(n, &ys)))),
            );
            for (g, e) in got.coordinates().iter().zip(&expected) {
                assert!((g - e).abs() < 1e-12, "n={n}: got {:?}, expected {:?}", got.coordinates(), expected);
            }
        }
    }

    #[test]
    fn test_heisenberg_conjugation() {
        // Ad_{exp X}(Y) = Y + [X, Y] in a 2-step nilpotent group.
        let h3 = NilpotentLieGroup::heisenberg(1);
        let g = h3.exp(&[2.0, 0.0, 0.0]);
        let conj = h3.conjugation(&g);
        let r = conj(&h3.exp(&[0.0, 3.0, 1.0]));
        assert_eq!(r.coordinates(), &[0.0, 3.0, 7.0]); // 1 + [2X, 3Y] = 1 + 6
    }

    #[test]
    fn test_new_dimension3_class2_is_heisenberg() {
        let g = NilpotentLieGroup::new(3, 2);
        assert_eq!(g.bracket(), Some(LieBracket::Heisenberg { n: 1 }));
        let x = g.exp(&[1.0, 0.0, 0.0]);
        let y = g.exp(&[0.0, 1.0, 0.0]);
        assert_eq!(x.op(&y).coordinates(), &[1.0, 1.0, 0.5]);
    }

    #[test]
    #[should_panic(expected = "not determined")]
    fn test_new_undetermined_bracket_refuses_to_multiply() {
        // Dimension 5 and class 2 do not determine the algebra (h_5, h_3 ⊕ R^2, …).
        let g = NilpotentLieGroup::new(5, 2);
        assert_eq!(g.bracket(), None);
        let x = g.exp(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let y = g.exp(&[0.0, 1.0, 0.0, 0.0, 0.0]);
        let _ = x.op(&y);
    }

    #[test]
    #[should_panic(expected = "different nilpotent Lie groups")]
    fn test_elements_of_different_groups_do_not_multiply() {
        let a = NilpotentLieGroup::abelian(3).exp(&[1.0, 0.0, 0.0]);
        let h = NilpotentLieGroup::heisenberg(1).exp(&[0.0, 1.0, 0.0]);
        let _ = a.op(&h);
    }

    #[test]
    fn test_contains_checks_bracket() {
        let h3 = NilpotentLieGroup::heisenberg(1);
        let a3 = NilpotentLieGroup::abelian(3);
        assert!(h3.contains(&h3.exp(&[1.0, 2.0, 3.0])));
        assert!(!h3.contains(&a3.exp(&[1.0, 2.0, 3.0])));
    }
}

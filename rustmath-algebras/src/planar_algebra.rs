//! Planar Algebra
//!
//! A planar algebra is a mathematical structure that formalizes operations on planar diagrams.
//! It consists of:
//! - A collection of vector spaces P_n (planar diagrams with n boundary points)
//! - Composition operations (inserting diagrams into disks)
//! - Planarity constraints (no crossings)
//!
//! Planar algebras were introduced by Vaughan Jones to study subfactors and
//! have connections to:
//! - Knot theory and the Jones polynomial
//! - Quantum groups and representation theory
//! - Conformal field theory
//! - Statistical mechanics
//!
//! This module provides a framework for working with planar diagram algebras,
//! with the Temperley-Lieb algebra as the primary example.
//!
//! References:
//! - Jones, V. F. R. "Planar algebras I" (1999)
//! - Jones, V. F. R. "The planar algebra of a bipartite graph" (2000)

use rustmath_core::{Ring, MathError, Result};
use rustmath_modules::CombinatorialFreeModuleElement;
use crate::diagram::{PartitionDiagram, TemperleyLiebDiagram};
use std::fmt::{self, Display};
use std::marker::PhantomData;

/// A planar diagram with marked boundary points
///
/// This is a wrapper around partition diagrams that enforces planarity
/// and provides operations specific to planar algebras.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlanarDiagram {
    /// The underlying diagram
    diagram: TemperleyLiebDiagram,
    /// Number of boundary points
    boundary_points: usize,
    /// Shading (for bipartite planar algebras)
    shading: Option<bool>,
}

impl PlanarDiagram {
    /// Create a new planar diagram
    ///
    /// # Arguments
    ///
    /// * `diagram` - The underlying Temperley-Lieb diagram (guaranteed planar)
    /// * `shading` - Optional shading for bipartite planar algebras
    pub fn new(diagram: TemperleyLiebDiagram, shading: Option<bool>) -> Self {
        let boundary_points = diagram.order() * 2;
        PlanarDiagram {
            diagram,
            boundary_points,
            shading,
        }
    }

    /// Create the identity planar diagram
    pub fn identity(order: usize) -> Self {
        let diagram = TemperleyLiebDiagram::identity(order);
        PlanarDiagram::new(diagram, None)
    }

    /// Get the number of boundary points
    pub fn boundary_points(&self) -> usize {
        self.boundary_points
    }

    /// Get the underlying diagram
    pub fn diagram(&self) -> &TemperleyLiebDiagram {
        &self.diagram
    }

    /// Get the shading (if any)
    pub fn shading(&self) -> Option<bool> {
        self.shading
    }

    /// Compose two planar diagrams
    ///
    /// This performs vertical composition (stacking)
    pub fn compose(&self, other: &PlanarDiagram) -> Result<PlanarDiagram> {
        let composed = self.diagram.compose(other.diagram())?;

        // Handle shading (for bipartite planar algebras)
        let new_shading = match (self.shading, other.shading) {
            (Some(s1), Some(s2)) if s1 == s2 => Some(s1),
            (Some(s), None) | (None, Some(s)) => Some(s),
            (None, None) => None,
            _ => return Err(MathError::InvalidArgument(
                "Incompatible shadings in planar diagram composition".to_string()
            )),
        };

        Ok(PlanarDiagram::new(composed, new_shading))
    }

    /// Check if diagram has the rotation property (invariant under rotation)
    pub fn is_rotation_invariant(&self) -> bool {
        // This would require checking if the diagram looks the same
        // when rotated. For now, return false (not implemented)
        false
    }

    /// Count the Euler characteristic contribution
    ///
    /// For planar diagrams: χ = boundary_components - internal_loops
    pub fn euler_characteristic(&self) -> i32 {
        let boundary_components = self.diagram.diagram().propagating_count();
        // Count internal regions (simplified version)
        boundary_components as i32
    }
}

impl Display for PlanarDiagram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Planar[{}]", self.diagram)
    }
}

/// A planar algebra structure
///
/// This provides the framework for operations on planar diagrams.
///
/// # Type Parameters
///
/// * `R` - The base ring
#[derive(Clone, Debug)]
pub struct PlanarAlgebra<R: Ring> {
    /// The parameter (often loop value δ)
    delta: R,
    /// Maximum order we're working with
    max_order: usize,
    /// Phantom data for ring type
    _phantom: PhantomData<R>,
}

impl<R: Ring> PlanarAlgebra<R> {
    /// Create a new planar algebra
    ///
    /// # Arguments
    ///
    /// * `delta` - The loop parameter δ
    /// * `max_order` - Maximum number of strands to work with
    pub fn new(delta: R, max_order: usize) -> Self {
        PlanarAlgebra {
            delta,
            max_order,
            _phantom: PhantomData,
        }
    }

    /// Get the delta parameter
    pub fn delta(&self) -> &R {
        &self.delta
    }

    /// Get the identity element for n strands
    pub fn identity(&self, n: usize) -> PlanarElement<R> {
        if n > self.max_order {
            panic!("Order {} exceeds max_order {}", n, self.max_order);
        }

        let diagram = PlanarDiagram::identity(n);
        PlanarElement {
            module_element: CombinatorialFreeModuleElement::from_basis_index(diagram),
        }
    }

    /// Get the zero element
    pub fn zero(&self) -> PlanarElement<R> {
        PlanarElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Multiply two planar diagrams (compose with loop evaluation)
    pub fn product_on_basis(&self, d1: &PlanarDiagram, d2: &PlanarDiagram) -> PlanarElement<R> {
        // Compose diagrams
        let composed = match d1.compose(d2) {
            Ok(d) => d,
            Err(_) => return self.zero(),
        };

        // Count loops (simplified - actual counting requires tracking middle strands)
        let loops_count = self.estimate_loops(d1, d2);

        // Compute coefficient
        let mut coefficient = R::one();
        for _ in 0..loops_count {
            coefficient = coefficient * self.delta.clone();
        }

        PlanarElement {
            module_element: CombinatorialFreeModuleElement::monomial(composed, coefficient),
        }
    }

    /// Estimate number of loops created in composition
    fn estimate_loops(&self, _d1: &PlanarDiagram, _d2: &PlanarDiagram) -> usize {
        // Simplified implementation - actual loop counting requires
        // analyzing the middle region connectivity
        0
    }

    /// Create tensor product of two planar diagrams (horizontal composition)
    pub fn tensor_product(&self, e1: &PlanarElement<R>, e2: &PlanarElement<R>) -> PlanarElement<R> {
        // This would implement horizontal composition (placing diagrams side-by-side)
        // For now, return zero as placeholder
        self.zero()
    }

    /// Apply a rotation to a planar diagram
    pub fn rotate(&self, element: &PlanarElement<R>, _angle: f64) -> PlanarElement<R> {
        // Rotation in planar algebras
        // For now, return the element unchanged (identity rotation)
        element.clone()
    }
}

/// An element of a planar algebra
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanarElement<R: Ring> {
    /// The underlying combinatorial free module element
    pub module_element: CombinatorialFreeModuleElement<R, PlanarDiagram>,
}

impl<R: Ring> PlanarElement<R> {
    /// Create the zero element
    pub fn zero() -> Self {
        PlanarElement {
            module_element: CombinatorialFreeModuleElement::zero(),
        }
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool {
        self.module_element.is_zero()
    }

    /// Add two elements
    pub fn add(&self, other: &PlanarElement<R>) -> PlanarElement<R> {
        PlanarElement {
            module_element: self.module_element.clone() + other.module_element.clone(),
        }
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &R) -> PlanarElement<R> {
        PlanarElement {
            module_element: self.module_element.scalar_mul(scalar),
        }
    }

    /// Multiply two elements
    pub fn multiply(&self, other: &PlanarElement<R>, algebra: &PlanarAlgebra<R>) -> PlanarElement<R> {
        let mut result = PlanarElement::zero();

        for (d1, c1) in self.module_element.iter() {
            for (d2, c2) in other.module_element.iter() {
                let basis_product = algebra.product_on_basis(d1, d2);
                let coeff_product = c1.clone() * c2.clone();
                let term = basis_product.scalar_mul(&coeff_product);

                result = result.add(&term);
            }
        }

        result
    }
}

impl<R: Ring + Display> Display for PlanarElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let terms: Vec<_> = self.module_element.iter().collect();

        for (i, (diagram, coeff)) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }

            if coeff.is_one() {
                write!(f, "{}", diagram)?;
            } else {
                write!(f, "{}*{}", coeff, diagram)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_planar_diagram_creation() {
        let diagram = PlanarDiagram::identity(3);
        assert_eq!(diagram.boundary_points(), 6);
        assert_eq!(diagram.shading(), None);
    }

    #[test]
    fn test_planar_algebra_creation() {
        let algebra = PlanarAlgebra::<Integer>::new(Integer::from(2), 10);
        assert_eq!(*algebra.delta(), Integer::from(2));
    }

    #[test]
    fn test_identity_element() {
        let algebra = PlanarAlgebra::<Integer>::new(Integer::from(2), 10);
        let id = algebra.identity(3);
        assert!(!id.is_zero());
    }

    #[test]
    fn test_composition() {
        let d1 = PlanarDiagram::identity(2);
        let d2 = PlanarDiagram::identity(2);

        let composed = d1.compose(&d2).unwrap();
        assert_eq!(composed.boundary_points(), 4);
    }

    #[test]
    fn test_shading() {
        let diagram = TemperleyLiebDiagram::identity(2);
        let planar_shaded = PlanarDiagram::new(diagram.clone(), Some(true));
        let planar_unshaded = PlanarDiagram::new(diagram, Some(false));

        assert_eq!(planar_shaded.shading(), Some(true));
        assert_eq!(planar_unshaded.shading(), Some(false));
    }

    #[test]
    fn test_addition() {
        let algebra = PlanarAlgebra::<Integer>::new(Integer::from(2), 10);
        let e1 = algebra.identity(2);
        let e2 = algebra.identity(2);

        let sum = e1.add(&e2);
        assert!(!sum.is_zero());
    }

    #[test]
    fn test_scalar_multiplication() {
        let algebra = PlanarAlgebra::<Integer>::new(Integer::from(2), 10);
        let e = algebra.identity(2);
        let scalar = Integer::from(3);

        let scaled = e.scalar_mul(&scalar);
        assert!(!scaled.is_zero());
    }
}

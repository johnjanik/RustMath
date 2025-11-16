//! Library of standard hyperplane arrangements
//!
//! This module provides a collection of commonly-used hyperplane arrangements
//! including braid arrangements, coordinate arrangements, and others from
//! combinatorial geometry.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
//! use rustmath_integers::Integer;
//!
//! let lib = HyperplaneArrangementLibrary;
//! let braid = lib.braid::<Integer>(3);
//!
//! // The braid arrangement in 3D has 3 choose 2 = 3 hyperplanes
//! assert_eq!(braid.num_hyperplanes(), 3);
//! ```

use crate::hyperplane_arrangement::arrangement::{HyperplaneArrangementElement, HyperplaneArrangements};
use crate::hyperplane_arrangement::hyperplane::Hyperplane;
use rustmath_core::Ring;

/// Library of standard hyperplane arrangements
///
/// Provides methods to construct commonly-used hyperplane arrangements
/// from combinatorial geometry and representation theory.
pub struct HyperplaneArrangementLibrary;

impl HyperplaneArrangementLibrary {
    /// Create the braid arrangement in dimension n
    ///
    /// The braid arrangement consists of hyperplanes x_i - x_j = 0
    /// for all i < j. This has C(n, 2) = n(n-1)/2 hyperplanes.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The dimension of the ambient space
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let braid = lib.braid::<Integer>(3);
    ///
    /// assert_eq!(braid.num_hyperplanes(), 3); // C(3,2) = 3
    /// assert!(braid.is_central()); // All pass through origin
    /// ```
    pub fn braid<R: Ring>(&self, dimension: usize) -> HyperplaneArrangementElement<R> {
        if dimension == 0 {
            panic!("Dimension must be at least 1");
        }

        let mut hyperplanes = Vec::new();

        // Create hyperplanes x_i - x_j = 0 for all i < j
        for i in 0..dimension {
            for j in (i + 1)..dimension {
                let mut coeffs = vec![R::zero(); dimension];
                coeffs[i] = R::one();
                coeffs[j] = -R::one();
                hyperplanes.push(Hyperplane::new(coeffs, R::zero()));
            }
        }

        if hyperplanes.is_empty() {
            // For dimension 1, create a single hyperplane x = 0
            hyperplanes.push(Hyperplane::new(vec![R::one()], R::zero()));
        }

        HyperplaneArrangementElement::new(hyperplanes)
    }

    /// Create the coordinate arrangement in dimension n
    ///
    /// The coordinate arrangement consists of hyperplanes x_i = 0
    /// for i = 0, 1, ..., n-1. This has n hyperplanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let coord = lib.coordinate::<Integer>(3);
    ///
    /// assert_eq!(coord.num_hyperplanes(), 3);
    /// assert!(coord.is_central());
    /// ```
    pub fn coordinate<R: Ring>(&self, dimension: usize) -> HyperplaneArrangementElement<R> {
        if dimension == 0 {
            panic!("Dimension must be at least 1");
        }

        let mut hyperplanes = Vec::new();

        // Create hyperplanes x_i = 0 for all i
        for i in 0..dimension {
            let mut coeffs = vec![R::zero(); dimension];
            coeffs[i] = R::one();
            hyperplanes.push(Hyperplane::new(coeffs, R::zero()));
        }

        HyperplaneArrangementElement::new(hyperplanes)
    }

    /// Create the Boolean arrangement in dimension n
    ///
    /// The Boolean arrangement consists of hyperplanes x_i = 0 and x_i = 1
    /// for i = 0, 1, ..., n-1. This has 2n hyperplanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let boolean = lib.boolean::<Integer>(2);
    ///
    /// assert_eq!(boolean.num_hyperplanes(), 4); // 2 * 2
    /// ```
    pub fn boolean<R: Ring>(&self, dimension: usize) -> HyperplaneArrangementElement<R> {
        if dimension == 0 {
            panic!("Dimension must be at least 1");
        }

        let mut hyperplanes = Vec::new();

        // Create hyperplanes x_i = 0 and x_i = 1 for all i
        for i in 0..dimension {
            let mut coeffs = vec![R::zero(); dimension];
            coeffs[i] = R::one();

            // x_i = 0
            hyperplanes.push(Hyperplane::new(coeffs.clone(), R::zero()));

            // x_i = 1
            hyperplanes.push(Hyperplane::new(coeffs, R::one()));
        }

        HyperplaneArrangementElement::new(hyperplanes)
    }

    /// Create the graphical arrangement for a complete graph K_n
    ///
    /// For a complete graph on n vertices, this creates hyperplanes
    /// x_i - x_j = 0 for all pairs of vertices (same as braid arrangement).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let graphical = lib.graphical::<Integer>(4);
    ///
    /// assert_eq!(graphical.num_hyperplanes(), 6); // C(4,2) = 6
    /// ```
    pub fn graphical<R: Ring>(&self, num_vertices: usize) -> HyperplaneArrangementElement<R> {
        // For a complete graph, this is the same as the braid arrangement
        self.braid(num_vertices)
    }

    /// Create the Shi arrangement in dimension n
    ///
    /// The Shi arrangement consists of hyperplanes x_i - x_j = 0 and x_i - x_j = 1
    /// for all i < j. This has n(n-1) hyperplanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let shi = lib.shi::<Integer>(3);
    ///
    /// assert_eq!(shi.num_hyperplanes(), 6); // 3 * 2 = 6
    /// ```
    pub fn shi<R: Ring>(&self, dimension: usize) -> HyperplaneArrangementElement<R> {
        if dimension == 0 {
            panic!("Dimension must be at least 1");
        }

        let mut hyperplanes = Vec::new();

        // Create hyperplanes x_i - x_j = 0 and x_i - x_j = 1 for all i < j
        for i in 0..dimension {
            for j in (i + 1)..dimension {
                let mut coeffs = vec![R::zero(); dimension];
                coeffs[i] = R::one();
                coeffs[j] = -R::one();

                // x_i - x_j = 0
                hyperplanes.push(Hyperplane::new(coeffs.clone(), R::zero()));

                // x_i - x_j = 1
                hyperplanes.push(Hyperplane::new(coeffs, R::one()));
            }
        }

        if hyperplanes.is_empty() {
            // For dimension 1, create a simple arrangement
            hyperplanes.push(Hyperplane::new(vec![R::one()], R::zero()));
        }

        HyperplaneArrangementElement::new(hyperplanes)
    }

    /// Create the Linial arrangement in dimension n
    ///
    /// The Linial arrangement consists of hyperplanes x_i - x_j = 1
    /// for all i < j. This has C(n, 2) = n(n-1)/2 hyperplanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let linial = lib.linial::<Integer>(3);
    ///
    /// assert_eq!(linial.num_hyperplanes(), 3); // C(3,2) = 3
    /// assert!(!linial.is_central()); // None pass through origin
    /// ```
    pub fn linial<R: Ring>(&self, dimension: usize) -> HyperplaneArrangementElement<R> {
        if dimension == 0 {
            panic!("Dimension must be at least 1");
        }

        let mut hyperplanes = Vec::new();

        // Create hyperplanes x_i - x_j = 1 for all i < j
        for i in 0..dimension {
            for j in (i + 1)..dimension {
                let mut coeffs = vec![R::zero(); dimension];
                coeffs[i] = R::one();
                coeffs[j] = -R::one();
                hyperplanes.push(Hyperplane::new(coeffs, R::one()));
            }
        }

        if hyperplanes.is_empty() {
            // For dimension 1, create a single hyperplane x = 1
            hyperplanes.push(Hyperplane::new(vec![R::one()], R::one()));
        }

        HyperplaneArrangementElement::new(hyperplanes)
    }

    /// Create the semiorder arrangement in dimension n
    ///
    /// The semiorder arrangement consists of hyperplanes x_i - x_j = 1
    /// and x_i - x_j = -1 for all i < j.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::hyperplane_arrangement::library::HyperplaneArrangementLibrary;
    /// use rustmath_integers::Integer;
    ///
    /// let lib = HyperplaneArrangementLibrary;
    /// let semiorder = lib.semiorder::<Integer>(3);
    ///
    /// assert_eq!(semiorder.num_hyperplanes(), 6); // 2 * C(3,2) = 6
    /// ```
    pub fn semiorder<R: Ring>(&self, dimension: usize) -> HyperplaneArrangementElement<R> {
        if dimension == 0 {
            panic!("Dimension must be at least 1");
        }

        let mut hyperplanes = Vec::new();

        // Create hyperplanes x_i - x_j = 1 and x_i - x_j = -1 for all i < j
        for i in 0..dimension {
            for j in (i + 1)..dimension {
                let mut coeffs = vec![R::zero(); dimension];
                coeffs[i] = R::one();
                coeffs[j] = -R::one();

                // x_i - x_j = 1
                hyperplanes.push(Hyperplane::new(coeffs.clone(), R::one()));

                // x_i - x_j = -1
                hyperplanes.push(Hyperplane::new(coeffs, -R::one()));
            }
        }

        if hyperplanes.is_empty() {
            // For dimension 1, create simple hyperplanes
            hyperplanes.push(Hyperplane::new(vec![R::one()], R::one()));
        }

        HyperplaneArrangementElement::new(hyperplanes)
    }
}

/// Helper function to create a parent for arrangements
///
/// # Arguments
///
/// * `dimension` - The dimension of the ambient space
pub fn make_parent(dimension: usize) -> HyperplaneArrangements {
    HyperplaneArrangements::new(dimension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_braid_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let braid = lib.braid::<Integer>(3);

        // Braid in 3D has C(3,2) = 3 hyperplanes
        assert_eq!(braid.num_hyperplanes(), 3);
        assert_eq!(braid.ambient_dimension(), 3);
        assert!(braid.is_central());
    }

    #[test]
    fn test_braid_arrangement_larger() {
        let lib = HyperplaneArrangementLibrary;
        let braid = lib.braid::<Integer>(4);

        // Braid in 4D has C(4,2) = 6 hyperplanes
        assert_eq!(braid.num_hyperplanes(), 6);
    }

    #[test]
    fn test_coordinate_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let coord = lib.coordinate::<Integer>(3);

        assert_eq!(coord.num_hyperplanes(), 3);
        assert!(coord.is_central());
        assert!(coord.is_essential());
    }

    #[test]
    fn test_boolean_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let boolean = lib.boolean::<Integer>(2);

        // Boolean in 2D has 2*2 = 4 hyperplanes
        assert_eq!(boolean.num_hyperplanes(), 4);
        assert_eq!(boolean.ambient_dimension(), 2);
        assert!(!boolean.is_central()); // Not all pass through origin
    }

    #[test]
    fn test_graphical_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let graphical = lib.graphical::<Integer>(4);

        // Complete graph K_4 has C(4,2) = 6 edges
        assert_eq!(graphical.num_hyperplanes(), 6);
    }

    #[test]
    fn test_shi_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let shi = lib.shi::<Integer>(3);

        // Shi has 2 * C(3,2) = 6 hyperplanes
        assert_eq!(shi.num_hyperplanes(), 6);
    }

    #[test]
    fn test_linial_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let linial = lib.linial::<Integer>(3);

        assert_eq!(linial.num_hyperplanes(), 3);
        assert!(!linial.is_central()); // x_i - x_j = 1 doesn't pass through origin
    }

    #[test]
    fn test_semiorder_arrangement() {
        let lib = HyperplaneArrangementLibrary;
        let semiorder = lib.semiorder::<Integer>(3);

        // 2 * C(3,2) = 6
        assert_eq!(semiorder.num_hyperplanes(), 6);
    }

    #[test]
    fn test_with_rationals() {
        let lib = HyperplaneArrangementLibrary;
        let braid = lib.braid::<Rational>(3);

        assert_eq!(braid.num_hyperplanes(), 3);
        assert!(braid.is_central());
    }

    #[test]
    fn test_make_parent() {
        let parent = make_parent(3);
        assert_eq!(parent.dimension(), 3);
    }
}

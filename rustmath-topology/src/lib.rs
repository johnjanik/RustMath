//! # RustMath Topology Module
//!
//! This module provides comprehensive topological structures and computations,
//! mirroring SageMath's `sage.topology` module.
//!
//! ## Overview
//!
//! Topology studies properties of spaces that are preserved under continuous deformations.
//! This module provides:
//!
//! - **Cell Complexes**: Abstract cell complex structures with boundary operations
//! - **Cubical Complexes**: Topological spaces built from cubes
//! - **Delta Complexes**: Simplicial-style complexes with identification rules
//! - **Simplicial Complexes**: Fundamental structures built from simplices
//! - **Simplicial Sets**: Category-theoretic approach to homotopy theory
//! - **Filtered Complexes**: Complexes with filtration for persistent homology
//! - **Moment Angle Complexes**: Toric topology structures
//! - **Morphisms and Homsets**: Maps between topological spaces
//! - **Knot Theory**: Knots, links, braids, polynomial invariants, and Reidemeister moves
//!
//! ## Module Structure
//!
//! - `cell_complex`: Generic cell complex base class
//! - `cubical_complex`: Cubical complexes and cubes
//! - `delta_complex`: Delta complexes with quotient structure
//! - `filtered_simplicial_complex`: Filtered simplicial complexes for TDA
//! - `moment_angle_complex`: Moment-angle complexes from toric topology
//! - `simplicial_complex`: Core simplicial complex functionality
//! - `simplicial_complex_examples`: Pre-built simplicial complex examples
//! - `simplicial_complex_homset`: Homomorphism sets between simplicial complexes
//! - `simplicial_complex_morphism`: Maps between simplicial complexes
//! - `simplicial_set`: Simplicial sets for homotopy theory
//! - `simplicial_set_constructions`: Constructions (cones, suspensions, products)
//! - `simplicial_set_examples`: Pre-built simplicial set examples
//! - `simplicial_set_morphism`: Maps between simplicial sets
//! - `knot`: Core knot and link representations (PD codes, Gauss codes)
//! - `braid`: Braid words and braid group operations
//! - `reidemeister`: Reidemeister moves for knot diagrams
//! - `knot_polynomials`: Jones and HOMFLY polynomial invariants
//! - `knot_invariants`: Crossing number, unknotting number, genus, etc.
//! - `link`: Link operations (connected sum, satellite, torus knots)
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_topology::simplicial_complex::{SimplicialComplex, Simplex};
//! use rustmath_topology::simplicial_complex_examples;
//!
//! // Create a torus as a simplicial complex
//! let torus = simplicial_complex_examples::torus();
//!
//! // Compute its Euler characteristic
//! let euler_char = torus.euler_characteristic();
//! assert_eq!(euler_char, 0); // Torus has Euler characteristic 0
//! ```

pub mod cell_complex;
pub mod cubical_complex;
pub mod delta_complex;
pub mod filtered_simplicial_complex;
pub mod moment_angle_complex;
pub mod simplicial_complex;
pub mod simplicial_complex_examples;
pub mod simplicial_complex_homset;
pub mod simplicial_complex_morphism;
pub mod simplicial_set;
pub mod simplicial_set_constructions;
pub mod simplicial_set_examples;
pub mod simplicial_set_morphism;

// Knot theory modules
pub mod knot;
pub mod braid;
pub mod reidemeister;
pub mod knot_polynomials;
pub mod knot_invariants;
pub mod link;

// Re-export commonly used types
pub use cell_complex::GenericCellComplex;
pub use simplicial_complex::{SimplicialComplex, Simplex};
pub use simplicial_set::{SimplicialSet, AbstractSimplex};

// Re-export knot theory types
pub use knot::{Knot, Crossing, CrossingType};
pub use braid::{Braid, BraidGenerator};
pub use link::Link;
pub use knot_polynomials::LaurentPolynomial;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Test that modules are accessible
        use crate::simplicial_complex::*;
        use crate::simplicial_set::*;
    }
}

//! RustMath Affine Schemes - Algebraic geometry via schemes
//!
//! This crate provides a comprehensive implementation of affine schemes, the
//! fundamental objects in modern algebraic geometry. An affine scheme is the
//! spectrum Spec(R) of a commutative ring R, equipped with a structure sheaf.
//!
//! # Mathematical Background
//!
//! An affine scheme Spec(R) consists of:
//! - The set of all prime ideals of R (as topological space with Zariski topology)
//! - A structure sheaf O_Spec(R) of rings on this space
//!
//! This generalizes classical algebraic varieties by allowing nilpotent elements
//! and working over arbitrary commutative rings (not just algebraically closed fields).
//!
//! # Key Features
//!
//! - **Affine schemes**: Spec(R) construction for commutative rings
//! - **Prime ideals**: Representation and manipulation of prime ideals
//! - **Zariski topology**: Open/closed sets defined by ideals
//! - **Structure sheaves**: Localization and sections over open sets
//! - **Morphisms**: Ring homomorphisms induce scheme morphisms
//! - **Fiber products**: Construct X ×_S Y for affine schemes
//! - **Base change**: Change the base ring/scheme
//! - **Dimension theory**: Krull dimension and height of primes
//! - **Affine varieties**: Classical varieties as special cases over fields
//!
//! # Examples
//!
//! ```ignore
//! use rustmath_affineschemes::*;
//! use rustmath_integers::Integer;
//!
//! // Create Spec(Z) - the affine scheme of integers
//! let spec_z = AffineScheme::spec_integers();
//!
//! // Create Spec(k[x, y]) - affine 2-space over a field k
//! let affine_2 = AffineScheme::affine_space(2);
//!
//! // Compute Krull dimension
//! let dim = spec_z.dimension();
//! ```

pub mod prime_ideal;
pub mod spec;
pub mod structure_sheaf;
pub mod morphism;
pub mod fiber_product;
pub mod dimension;
pub mod varieties;

pub use prime_ideal::{Ideal, PrimeIdeal, is_prime, radical};
pub use spec::{AffineScheme, SpecPoint, ZariskiOpen, ZariskiClosed};
pub use structure_sheaf::{StructureSheaf, LocalRing, Section};
pub use morphism::{SchemeMorphism, induced_morphism};
pub use fiber_product::{FiberProduct, fiber_product, base_change};
pub use dimension::{krull_dimension, height, transcendence_degree};
pub use varieties::{AffineVariety, coordinate_ring, function_field};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_scheme_basic() {
        // Basic smoke test - will be expanded with actual implementation
        assert!(true);
    }
}

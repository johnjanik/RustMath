//! # Quotient Ring Element Module
//!
//! This module provides the `QuotientRingElement` type, representing elements
//! in a quotient ring R/I.
//!
//! ## Overview
//!
//! Elements of a quotient ring are equivalence classes (cosets) under the
//! equivalence relation r ~ s ⟺ r - s ∈ I.
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_rings::quotient_ring_element::QuotientRingElement;
//! ```

// Re-export from quotient_ring module
pub use crate::quotient_ring::{QuotientRingElement, QuotientRingError};

//! p-adic numbers and rings
//!
//! This module provides implementations of p-adic numbers with various precision models.
//!
//! # Precision Models
//!
//! - **Capped Relative Precision**: Elements track precision relative to their valuation.
//!   This is the most commonly used model and corresponds to Sage's default p-adic implementation.
//!
//! # Examples
//!
//! ```rust
//! use rustmath_integers::Integer;
//! use rustmath_rings::padics::capped_relative::CappedRelativePadicElement;
//!
//! // Create 7 + O(5^10) in Q_5
//! let x = CappedRelativePadicElement::new(
//!     Integer::from(7),
//!     Integer::from(5),
//!     10
//! ).unwrap();
//!
//! // Arithmetic operations track precision correctly
//! let y = CappedRelativePadicElement::new(
//!     Integer::from(3),
//!     Integer::from(5),
//!     10
//! ).unwrap();
//!
//! let sum = x.clone() + y.clone();
//! let prod = x * y;
//! ```

pub mod capped_relative;

pub use capped_relative::CappedRelativePadicElement;

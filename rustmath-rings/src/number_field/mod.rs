//! Number Field Orders
//!
//! This module implements orders (rings of integers) in algebraic number fields,
//! corresponding to sage.rings.number_field.order.

pub mod order;

pub use order::{Order, OrderElement, OrderIdeal, OrderError};

//! Number Field Module
//!
//! This module provides functionality for algebraic number fields, including:
//! - S-unit equation solving
//! - Fundamental units computation
//! - Regulator calculation
//! - S-unit group structure

pub mod s_unit_solver;

// Re-export main types and functions
pub use s_unit_solver::{
    SUnitError,
    SUnitGroup,
    compute_fundamental_units,
    compute_regulator,
    compute_s_unit_group,
    solve_s_unit_equation,
    height,
    is_s_unit,
};

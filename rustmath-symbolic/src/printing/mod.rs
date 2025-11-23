//! Pretty printing for symbolic expressions
//!
//! This module provides various output formats for symbolic expressions:
//! - Unicode: For terminal display with mathematical symbols
//! - LaTeX: For Jupyter notebooks and documentation
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::{Expr, printing};
//!
//! let x = Expr::symbol("x");
//! let expr = x.clone().pow(Expr::from(2)) + x;
//!
//! // Unicode output: x² + x
//! let unicode_str = printing::to_unicode(&expr);
//!
//! // LaTeX output: {x}^{2} + x
//! let latex_str = printing::to_latex(&expr);
//! ```

pub mod unicode;
pub mod latex;

pub use unicode::to_unicode;
pub use latex::to_latex;

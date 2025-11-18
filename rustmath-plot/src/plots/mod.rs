//! Advanced 2D plotting functions (Phase 4)
//!
//! This module provides function plotting capabilities including:
//! - Basic function plots: `plot()`, `parametric_plot()`
//! - Data visualization: `list_plot()`, `scatter_plot()`
//! - Contour and density plots: `contour_plot()`, `density_plot()`
//! - Statistical plots: `histogram()`, `bar_chart()`
//! - Specialized plots: `matrix_plot()`
//!
//! Based on SageMath's sage.plot module.

mod plot;
mod parametric_plot;
mod list_plot;
mod scatter_plot;
mod contour_plot;
mod density_plot;
mod histogram;
mod bar_chart;
mod matrix_plot;

// Re-export all plot functions
pub use plot::plot;
pub use parametric_plot::parametric_plot;
pub use list_plot::list_plot;
pub use scatter_plot::scatter_plot;
pub use contour_plot::contour_plot;
pub use density_plot::density_plot;
pub use histogram::histogram;
pub use bar_chart::bar_chart;
pub use matrix_plot::matrix_plot;

//! 3D plotting functions
//!
//! This module provides high-level functions for creating 3D plots,
//! including surface plots, parametric plots, and more.

pub mod plot3d;
pub mod parametric_plot3d;
pub mod list_plot3d;
pub mod revolution_plot3d;

pub use plot3d::plot3d;
pub use parametric_plot3d::parametric_plot3d;
pub use list_plot3d::list_plot3d;
pub use revolution_plot3d::revolution_plot3d;

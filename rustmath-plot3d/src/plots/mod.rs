//! 3D plotting functions
//!
//! This module provides high-level functions for creating 3D plots,
//! including surface plots, parametric plots, scatter plots, implicit surfaces,
//! vector fields, and wireframe visualizations.

pub mod plot3d;
pub mod parametric_plot3d;
pub mod list_plot3d;
pub mod revolution_plot3d;
pub mod wireframe_plot3d;
pub mod scatter_plot3d;
pub mod implicit_plot3d;
pub mod vector_field_plot3d;

pub use plot3d::plot3d;
pub use parametric_plot3d::parametric_plot3d;
pub use list_plot3d::{list_plot3d, scatter_plot3d as list_scatter_plot3d};
pub use revolution_plot3d::revolution_plot3d;
pub use wireframe_plot3d::{wireframe_plot3d, wireframe_parametric_plot3d};
pub use scatter_plot3d::{scatter_plot3d, scatter_plot3d_colored};
pub use implicit_plot3d::implicit_plot3d;
pub use vector_field_plot3d::vector_field_plot3d;

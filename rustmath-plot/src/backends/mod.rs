//! Rendering backends for different output formats
//!
//! This module contains implementations of the RenderBackend trait
//! for various output formats like SVG and PNG.

mod svg_backend;
mod raster_backend;

pub use svg_backend::SvgBackend;
pub use raster_backend::RasterBackend;

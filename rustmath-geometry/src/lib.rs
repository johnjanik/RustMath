//! RustMath Geometry - Geometric objects and algorithms
//!
//! This crate provides geometric primitives and computational geometry algorithms.

pub mod point;
pub mod line;
pub mod polygon;
pub mod polyhedron;

pub use point::{Point2D, Point3D};
pub use line::{Line2D, LineSegment2D};
pub use polygon::{Polygon, convex_hull};
pub use polyhedron::{Polyhedron, Face};

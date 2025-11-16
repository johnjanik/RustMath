//! RustMath Geometry - Geometric objects and algorithms
//!
//! This crate provides geometric primitives and computational geometry algorithms.

pub mod point;
pub mod line;
pub mod polygon;
pub mod polyhedron;
pub mod convex_hull_3d;
pub mod triangulation;
pub mod face_lattice;
pub mod toric;
pub mod voronoi;
pub mod convex_set;
pub mod relative_interior;
pub mod hasse_diagram;
pub mod abc;
pub mod linear_expression;
pub mod point_collection;
pub mod newton_polygon;
pub mod integral_points;
pub mod polyhedral_complex;
pub mod hyperplane_arrangement;
pub mod hyperbolic_space;
pub mod lattice_polytope;
pub mod pseudolines;
pub mod ribbon_graph;
pub mod riemannian_manifolds;

pub use point::{Point2D, Point3D};
pub use line::{Line2D, LineSegment2D};
pub use polygon::{Polygon, convex_hull};
pub use polyhedron::{Polyhedron, Face};
pub use convex_hull_3d::{convex_hull_3d, convex_hull_3d_simple};
pub use triangulation::{Triangle, delaunay_triangulation};
pub use face_lattice::{FaceLattice, LatticeFace};
pub use toric::{Cone, Fan, ToricVariety, projective_space_fan};
pub use voronoi::{VoronoiDiagram, VoronoiCell, voronoi_brute_force};
pub use lattice_polytope::{LatticePolytopeClass, lattice_polytope, cross_polytope};
pub use pseudolines::{PseudolineArrangement, Encoding};
pub use ribbon_graph::{RibbonGraph, make_ribbon, bipartite_ribbon_graph};
pub use riemannian_manifolds::{ParametrizedSurface3D, SurfaceGenerators};

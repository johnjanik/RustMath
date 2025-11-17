//! RustMath Quivers - Quiver theory and representation theory
//!
//! This crate provides data structures and algorithms for working with quivers
//! (directed graphs with labeled edges), which are fundamental in representation theory.
//!
//! # Overview
//!
//! A **quiver** is a directed graph where:
//! - Vertices are labeled by integers
//! - Edges are labeled by unique strings
//!
//! A **path** in a quiver is a sequence of edges where the terminal vertex of each
//! edge is the initial vertex of the next edge.
//!
//! # Examples
//!
//! ```
//! use rustmath_quivers::Quiver;
//!
//! // Create a quiver with 3 vertices
//! let mut q = Quiver::new(3);
//!
//! // Add labeled edges
//! q.add_edge(0, 1, "a").unwrap();
//! q.add_edge(1, 2, "b").unwrap();
//! q.add_edge(0, 2, "c").unwrap();
//!
//! assert_eq!(q.num_vertices(), 3);
//! assert_eq!(q.num_edges(), 3);
//! ```

pub mod quiver;
pub mod path;
pub mod path_semigroup;

pub use quiver::Quiver;
pub use path::QuiverPath;
pub use path_semigroup::PathSemigroup;

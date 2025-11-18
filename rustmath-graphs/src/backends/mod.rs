//! Graph backend implementations
//!
//! This module provides different graph storage backends similar to SageMath's graph backends.
//! Different backends are optimized for different use cases:
//! - `GenericGraphBackend`: Base trait for all backends
//! - `DenseGraphBackend`: Dense adjacency matrix (good for dense graphs)
//! - `SparseGraphBackend`: Sparse adjacency list (good for sparse graphs)
//! - `StaticSparseBackend`: Immutable sparse representation (optimized for queries)

pub mod generic_backend;
pub mod c_graph;
pub mod dense_graph;
pub mod sparse_graph;
pub mod static_dense_graph;
pub mod static_sparse_graph;
pub mod static_sparse_backend;

pub use generic_backend::GenericGraphBackend;
pub use c_graph::{CGraphBackend, SearchIterator};
pub use dense_graph::{DenseGraph, DenseGraphBackend};
pub use sparse_graph::{SparseGraph, SparseGraphBackend};
pub use static_dense_graph::*;
pub use static_sparse_graph::*;
pub use static_sparse_backend::{StaticSparseBackend, StaticSparseCGraph};

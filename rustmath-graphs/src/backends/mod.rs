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
pub use static_dense_graph::{
    is_strongly_regular, is_triangle_free, triangles_count as triangles_count_dense,
    connected_full_subgraphs, connected_subgraph_iterator, ConnectedSubgraphIterator
};
pub use static_sparse_graph::{
    spectral_radius, tarjan_strongly_connected_components,
    strongly_connected_components_digraph, triangles_count as triangles_count_sparse
};
pub use static_sparse_backend::{StaticSparseBackend, StaticSparseCGraph};

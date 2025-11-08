//! RustMath Graphs - Graph theory data structures and algorithms
//!
//! This crate provides graph data structures, algorithms for traversal,
//! shortest paths, and other graph-theoretic computations.

pub mod graph;
pub mod generators;
pub mod weighted_graph;
pub mod digraph;
pub mod multigraph;

pub use graph::Graph;
pub use generators::*;
pub use weighted_graph::WeightedGraph;
pub use digraph::DiGraph;
pub use multigraph::MultiGraph;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_graph() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert_eq!(g.num_vertices(), 3);
        assert_eq!(g.num_edges(), 2);
    }
}

//! RustMath Graphs - Graph theory data structures and algorithms
//!
//! This crate provides graph data structures, algorithms for traversal,
//! shortest paths, and other graph-theoretic computations.

pub mod graph;
pub mod generators;
pub mod weighted_graph;
pub mod digraph;
pub mod multigraph;
pub mod asteroidal_triples;
pub mod boost_graph;
pub mod backends;
pub mod centrality;
pub mod cliquer;
pub mod cographs;
pub mod comparability;
pub mod connectivity;
pub mod convexity_properties;
pub mod distances_all_pairs;
pub mod domination;
pub mod edge_connectivity;
pub mod planarity;
pub mod spanning_tree;
pub mod weakly_chordal;
pub mod traversals;
pub mod views;
pub mod trees;
pub mod tutte_polynomial;
pub mod strongly_regular_db;
pub mod degree_sequences;
pub mod graph_path;

// Advanced graph theory (tracker 07)
pub mod homomorphisms;
pub mod automorphisms;
pub mod cayley;
pub mod spectra;
pub mod ramsey;

// MAGMA Handbook port — Wave 1 (chapters 149, 150, 151).
// Exact algebraic OUTPUT objects (Matrix<Integer>, polynomials, Integer flows)
// added *beside* the existing f64/Vec<i64> encodings; see each module header.
pub mod magma_matrices; // ch149 §149.16: adjacency/incidence/distance/Laplacian over Z
pub mod exact_spectra; // ch149 §149.10: characteristic polynomial over Z + integer spectrum
pub mod graph_polynomials; // ch149 §149.19: chromatic/matching/Tutte as rustmath-polynomials
pub mod magma_automorphisms; // ch149 §149.22: automorphism group (perm-list; groups wiring deferred)
pub mod multigraph_algorithms; // ch150: handshake/loops/Euler tours/multigraph traversal
pub mod network; // ch151: capacitated networks, max-flow/min-cut/min-cost-flow over Integer

pub use graph::Graph;
pub use generators::*;
pub use weighted_graph::WeightedGraph;
pub use digraph::DiGraph;
pub use multigraph::MultiGraph;
pub use asteroidal_triples::is_asteroidal_triple_free;
pub use degree_sequences::DegreeSequence;
pub use graph_path::GraphPath;

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

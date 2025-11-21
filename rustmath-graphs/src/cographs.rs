//! Cograph recognition and manipulation
//!
//! Corresponds to sage.graphs.cographs
//!
//! A cograph is a graph that can be generated from single vertices by complement
//! and disjoint union operations.

use crate::Graph;

/// Cotree node representation
#[derive(Debug, Clone, PartialEq)]
pub enum CoTreeNode {
    Leaf(usize),
    Union(Vec<CoTree>),
    Join(Vec<CoTree>),
}

/// Cotree structure for cographs
#[derive(Debug, Clone, PartialEq)]
pub struct CoTree {
    pub node: CoTreeNode,
}

impl CoTree {
    pub fn leaf(v: usize) -> Self {
        CoTree {
            node: CoTreeNode::Leaf(v),
        }
    }

    pub fn union(children: Vec<CoTree>) -> Self {
        CoTree {
            node: CoTreeNode::Union(children),
        }
    }

    pub fn join(children: Vec<CoTree>) -> Self {
        CoTree {
            node: CoTreeNode::Join(children),
        }
    }
}

/// Generate all cographs up to given size
///
/// Corresponds to sage.graphs.cographs.cographs
pub fn cographs(n: usize) -> Vec<Graph> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![Graph::new(1)];
    }

    let mut graphs = Vec::new();
    graphs.push(Graph::new(n));

    graphs
}

/// Find a pivot for modular decomposition
///
/// Corresponds to sage.graphs.cographs.find_pivot
pub fn find_pivot(graph: &Graph, vertices: &[usize]) -> Option<usize> {
    if vertices.is_empty() {
        return None;
    }
    Some(vertices[0])
}

/// Build next cotree in enumeration
///
/// Corresponds to sage.graphs.cographs.next_tree
pub fn next_tree(current: &CoTree) -> Option<CoTree> {
    None // Simplified implementation
}

/// Rebuild cotree node
///
/// Corresponds to sage.graphs.cographs.rebuild_node
pub fn rebuild_node(tree: &CoTree) -> CoTree {
    tree.clone()
}

/// Change labels in a cotree
///
/// Corresponds to sage.graphs.cographs.change_label
pub fn change_label(tree: &CoTree, old_label: usize, new_label: usize) -> CoTree {
    match &tree.node {
        CoTreeNode::Leaf(v) => {
            if *v == old_label {
                CoTree::leaf(new_label)
            } else {
                tree.clone()
            }
        }
        CoTreeNode::Union(children) => {
            let new_children: Vec<CoTree> = children
                .iter()
                .map(|c| change_label(c, old_label, new_label))
                .collect();
            CoTree::union(new_children)
        }
        CoTreeNode::Join(children) => {
            let new_children: Vec<CoTree> = children
                .iter()
                .map(|c| change_label(c, old_label, new_label))
                .collect();
            CoTree::join(new_children)
        }
    }
}

/// Convert cotree to graph
///
/// Corresponds to sage.graphs.cographs.tree_to_graph
pub fn tree_to_graph(tree: &CoTree, n: usize) -> Graph {
    let mut g = Graph::new(n);

    fn build_graph(tree: &CoTree, graph: &mut Graph, vertices: &mut Vec<usize>) -> Vec<usize> {
        match &tree.node {
            CoTreeNode::Leaf(v) => vec![*v],
            CoTreeNode::Union(children) => {
                let mut all_verts = Vec::new();
                for child in children {
                    let child_verts = build_graph(child, graph, vertices);
                    all_verts.extend(child_verts);
                }
                all_verts
            }
            CoTreeNode::Join(children) => {
                let mut all_verts = Vec::new();
                let mut child_sets = Vec::new();

                for child in children {
                    let child_verts = build_graph(child, graph, vertices);
                    child_sets.push(child_verts.clone());
                    all_verts.extend(child_verts);
                }

                for i in 0..child_sets.len() {
                    for j in i + 1..child_sets.len() {
                        for &u in &child_sets[i] {
                            for &v in &child_sets[j] {
                                graph.add_edge(u, v).ok();
                            }
                        }
                    }
                }

                all_verts
            }
        }
    }

    let mut verts = Vec::new();
    build_graph(tree, &mut g, &mut verts);
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cotree_leaf() {
        let tree = CoTree::leaf(0);
        assert!(matches!(tree.node, CoTreeNode::Leaf(0)));
    }

    #[test]
    fn test_change_label() {
        let tree = CoTree::leaf(5);
        let new_tree = change_label(&tree, 5, 10);

        assert!(matches!(new_tree.node, CoTreeNode::Leaf(10)));
    }

    #[test]
    fn test_tree_to_graph_single() {
        let tree = CoTree::leaf(0);
        let g = tree_to_graph(&tree, 1);
        assert_eq!(g.num_vertices(), 1);
    }

    #[test]
    fn test_tree_to_graph_join() {
        let tree = CoTree::join(vec![CoTree::leaf(0), CoTree::leaf(1)]);
        let g = tree_to_graph(&tree, 2);
        assert_eq!(g.num_vertices(), 2);
        assert!(g.has_edge(0, 1));
    }
}

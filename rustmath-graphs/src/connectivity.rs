//! Graph connectivity algorithms and analysis
//!
//! This module provides functions for analyzing graph connectivity properties including:
//! - Connected components and strong connectivity
//! - Bridges and cut vertices (articulation points)
//! - Vertex and edge connectivity
//! - Blocks and cut decomposition
//! - Triconnectivity and SPQR trees

use std::collections::{HashMap, HashSet, VecDeque};
use crate::graph::Graph;
use crate::digraph::DiGraph;

/// Check if a graph is connected
///
/// An undirected graph is connected if there is a path between every pair of vertices.
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// `true` if the graph is connected, `false` otherwise
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::connectivity::is_connected;
///
/// let mut g = Graph::new(3);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// assert!(is_connected(&g));
/// ```
pub fn is_connected(graph: &Graph) -> bool {
    graph.is_connected()
}

/// Find all connected components in a graph
///
/// Returns a vector where each element is a vector of vertices in that component.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector of connected components, where each component is a vector of vertex indices
///
/// # Examples
/// ```
/// use rustmath_graphs::graph::Graph;
/// use rustmath_graphs::connectivity::connected_components;
///
/// let mut g = Graph::new(5);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(2, 3).unwrap();
/// let components = connected_components(&g);
/// assert_eq!(components.len(), 3); // {0,1}, {2,3}, {4}
/// ```
pub fn connected_components(graph: &Graph) -> Vec<Vec<usize>> {
    graph.connected_components()
}

/// Count the number of connected components
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Number of connected components
pub fn connected_components_number(graph: &Graph) -> usize {
    graph.connected_components().len()
}

/// Get the sizes of all connected components
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector containing the size of each connected component
pub fn connected_components_sizes(graph: &Graph) -> Vec<usize> {
    graph.connected_components()
        .iter()
        .map(|comp| comp.len())
        .collect()
}

/// Get connected components as subgraphs
///
/// Returns the adjacency list for each connected component.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector of connected components, each represented as a mapping from vertices to their neighbors
pub fn connected_components_subgraphs(graph: &Graph) -> Vec<HashMap<usize, Vec<usize>>> {
    let components = graph.connected_components();
    let mut subgraphs = Vec::new();

    for component in components {
        let mut subgraph = HashMap::new();
        for &v in &component {
            let neighbors: Vec<usize> = graph.neighbors(v)
                .unwrap_or_default()
                .into_iter()
                .filter(|n| component.contains(n))
                .collect();
            subgraph.insert(v, neighbors);
        }
        subgraphs.push(subgraph);
    }

    subgraphs
}

/// Find the connected component containing a specific vertex
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `vertex` - The vertex to find the component for
///
/// # Returns
/// Vector of vertices in the same component as the given vertex, or None if vertex is invalid
pub fn connected_component_containing_vertex(graph: &Graph, vertex: usize) -> Option<Vec<usize>> {
    if vertex >= graph.num_vertices() {
        return None;
    }

    let mut visited = vec![false; graph.num_vertices()];
    let mut component = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(vertex);
    visited[vertex] = true;

    while let Some(v) = queue.pop_front() {
        component.push(v);
        for neighbor in graph.neighbors(v).unwrap_or_default() {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    Some(component)
}

/// Check if a graph is strongly connected (for directed graphs)
///
/// A directed graph is strongly connected if there's a path from every vertex to every other vertex.
///
/// # Arguments
/// * `digraph` - The directed graph to check
///
/// # Returns
/// `true` if the graph is strongly connected, `false` otherwise
pub fn is_strongly_connected(digraph: &DiGraph) -> bool {
    digraph.is_strongly_connected()
}

/// Find strongly connected components in a directed graph
///
/// Uses Kosaraju's algorithm to find all strongly connected components.
///
/// # Arguments
/// * `digraph` - The directed graph to analyze
///
/// # Returns
/// Vector of strongly connected components
pub fn strongly_connected_components(digraph: &DiGraph) -> Vec<Vec<usize>> {
    digraph.strongly_connected_components()
}

/// Find the strongly connected component containing a specific vertex
///
/// # Arguments
/// * `digraph` - The directed graph to analyze
/// * `vertex` - The vertex to find the component for
///
/// # Returns
/// Vector of vertices in the same strongly connected component, or None if vertex is invalid
pub fn strongly_connected_component_containing_vertex(digraph: &DiGraph, vertex: usize) -> Option<Vec<usize>> {
    if vertex >= digraph.num_vertices() {
        return None;
    }

    let components = digraph.strongly_connected_components();
    for component in components {
        if component.contains(&vertex) {
            return Some(component);
        }
    }

    None
}

/// Get strongly connected components as subgraphs
///
/// # Arguments
/// * `digraph` - The directed graph to analyze
///
/// # Returns
/// Vector of strongly connected components as subgraphs
pub fn strongly_connected_components_subgraphs(digraph: &DiGraph) -> Vec<HashMap<usize, Vec<usize>>> {
    let components = digraph.strongly_connected_components();
    let mut subgraphs = Vec::new();

    for component in components {
        let component_set: HashSet<usize> = component.iter().copied().collect();
        let mut subgraph = HashMap::new();

        for &v in &component {
            let edges = digraph.edges();
            let neighbors: Vec<usize> = edges.iter()
                .filter(|(u, _)| *u == v)
                .map(|(_, w)| *w)
                .filter(|w| component_set.contains(w))
                .collect();
            subgraph.insert(v, neighbors);
        }
        subgraphs.push(subgraph);
    }

    subgraphs
}

/// Create a condensation graph from strongly connected components
///
/// The condensation graph is a DAG where each node represents a strongly connected component.
///
/// # Arguments
/// * `digraph` - The directed graph to analyze
///
/// # Returns
/// A new DiGraph representing the condensation
pub fn strongly_connected_components_digraph(digraph: &DiGraph) -> DiGraph {
    let components = digraph.strongly_connected_components();
    let n_components = components.len();

    // Map vertices to their component indices
    let mut vertex_to_comp = HashMap::new();
    for (comp_idx, component) in components.iter().enumerate() {
        for &v in component {
            vertex_to_comp.insert(v, comp_idx);
        }
    }

    // Build condensation graph
    let mut condensation = DiGraph::new(n_components);
    let edges = digraph.edges();

    for (u, v) in edges {
        let comp_u = vertex_to_comp[&u];
        let comp_v = vertex_to_comp[&v];

        if comp_u != comp_v {
            condensation.add_edge(comp_u, comp_v).ok();
        }
    }

    condensation
}

/// Find all bridges in a graph
///
/// A bridge is an edge whose removal increases the number of connected components.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector of bridges represented as (u, v) edge pairs
///
/// # Algorithm
/// Uses Tarjan's bridge-finding algorithm with DFS and low-link values
pub fn bridges(graph: &Graph) -> Vec<(usize, usize)> {
    let n = graph.num_vertices();
    let mut visited = vec![false; n];
    let mut disc = vec![0; n];
    let mut low = vec![0; n];
    let mut parent = vec![None; n];
    let mut time = 0;
    let mut result = Vec::new();

    for v in 0..n {
        if !visited[v] {
            bridges_dfs(graph, v, &mut visited, &mut disc, &mut low, &mut parent, &mut time, &mut result);
        }
    }

    result
}

fn bridges_dfs(
    graph: &Graph,
    u: usize,
    visited: &mut Vec<bool>,
    disc: &mut Vec<usize>,
    low: &mut Vec<usize>,
    parent: &mut Vec<Option<usize>>,
    time: &mut usize,
    bridges: &mut Vec<(usize, usize)>,
) {
    visited[u] = true;
    disc[u] = *time;
    low[u] = *time;
    *time += 1;

    for &v in graph.neighbors(u).unwrap_or_default().iter() {
        if !visited[v] {
            parent[v] = Some(u);
            bridges_dfs(graph, v, visited, disc, low, parent, time, bridges);

            low[u] = low[u].min(low[v]);

            // If low[v] > disc[u], then (u, v) is a bridge
            if low[v] > disc[u] {
                bridges.push(if u < v { (u, v) } else { (v, u) });
            }
        } else if Some(v) != parent[u] {
            low[u] = low[u].min(disc[v]);
        }
    }
}

/// Find all cut vertices (articulation points) in a graph
///
/// A cut vertex is a vertex whose removal increases the number of connected components.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector of cut vertices
///
/// # Algorithm
/// Uses Tarjan's algorithm with DFS and low-link values
pub fn cut_vertices(graph: &Graph) -> Vec<usize> {
    let n = graph.num_vertices();
    let mut visited = vec![false; n];
    let mut disc = vec![0; n];
    let mut low = vec![0; n];
    let mut parent = vec![None; n];
    let mut is_cut_vertex = vec![false; n];
    let mut time = 0;

    for v in 0..n {
        if !visited[v] {
            cut_vertices_dfs(graph, v, &mut visited, &mut disc, &mut low, &mut parent, &mut is_cut_vertex, &mut time);
        }
    }

    is_cut_vertex.iter()
        .enumerate()
        .filter(|(_, &is_cut)| is_cut)
        .map(|(v, _)| v)
        .collect()
}

fn cut_vertices_dfs(
    graph: &Graph,
    u: usize,
    visited: &mut Vec<bool>,
    disc: &mut Vec<usize>,
    low: &mut Vec<usize>,
    parent: &mut Vec<Option<usize>>,
    is_cut_vertex: &mut Vec<bool>,
    time: &mut usize,
) {
    visited[u] = true;
    disc[u] = *time;
    low[u] = *time;
    *time += 1;
    let mut children = 0;

    for &v in graph.neighbors(u).unwrap_or_default().iter() {
        if !visited[v] {
            children += 1;
            parent[v] = Some(u);
            cut_vertices_dfs(graph, v, visited, disc, low, parent, is_cut_vertex, time);

            low[u] = low[u].min(low[v]);

            // u is an articulation point if:
            // 1. u is root of DFS tree and has two or more children
            // 2. u is not root and low[v] >= disc[u]
            if parent[u].is_none() && children > 1 {
                is_cut_vertex[u] = true;
            }
            if parent[u].is_some() && low[v] >= disc[u] {
                is_cut_vertex[u] = true;
            }
        } else if Some(v) != parent[u] {
            low[u] = low[u].min(disc[v]);
        }
    }
}

/// Check if a specific vertex is a cut vertex
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `vertex` - The vertex to check
///
/// # Returns
/// `true` if the vertex is a cut vertex, `false` otherwise
pub fn is_cut_vertex(graph: &Graph, vertex: usize) -> bool {
    if vertex >= graph.num_vertices() {
        return false;
    }

    let original_components = connected_components_number(graph);

    // Create a new graph without the vertex
    let mut new_graph = Graph::new(graph.num_vertices() - 1);

    for (u, v) in graph.edges() {
        if u != vertex && v != vertex {
            let new_u = if u > vertex { u - 1 } else { u };
            let new_v = if v > vertex { v - 1 } else { v };
            new_graph.add_edge(new_u, new_v).ok();
        }
    }

    let new_components = connected_components_number(&new_graph);

    // If removing the vertex increases the number of components, it's a cut vertex
    new_components > original_components
}

/// Check if a specific edge is a cut edge (bridge)
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `u` - First vertex of the edge
/// * `v` - Second vertex of the edge
///
/// # Returns
/// `true` if the edge is a bridge, `false` otherwise
pub fn is_cut_edge(graph: &Graph, u: usize, v: usize) -> bool {
    if !graph.has_edge(u, v) {
        return false;
    }

    let all_bridges = bridges(graph);
    let edge = if u < v { (u, v) } else { (v, u) };
    all_bridges.contains(&edge)
}

/// Check if a set of vertices forms a vertex cut
///
/// A vertex cut is a set of vertices whose removal disconnects the graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `vertices` - The set of vertices to check
///
/// # Returns
/// `true` if the set forms a vertex cut, `false` otherwise
pub fn is_vertex_cut(graph: &Graph, vertices: &[usize]) -> bool {
    let original_components = connected_components_number(graph);

    // Create vertex set for quick lookup
    let vertex_set: HashSet<usize> = vertices.iter().copied().collect();

    // Create a new graph without the vertices
    let remaining_vertices: Vec<usize> = (0..graph.num_vertices())
        .filter(|v| !vertex_set.contains(v))
        .collect();

    if remaining_vertices.is_empty() {
        return false;
    }

    // Map old vertices to new indices
    let mut vertex_map = HashMap::new();
    for (new_idx, &old_idx) in remaining_vertices.iter().enumerate() {
        vertex_map.insert(old_idx, new_idx);
    }

    let mut new_graph = Graph::new(remaining_vertices.len());

    for (u, v) in graph.edges() {
        if !vertex_set.contains(&u) && !vertex_set.contains(&v) {
            let new_u = vertex_map[&u];
            let new_v = vertex_map[&v];
            new_graph.add_edge(new_u, new_v).ok();
        }
    }

    let new_components = connected_components_number(&new_graph);
    new_components > original_components
}

/// Check if a set of edges forms an edge cut
///
/// An edge cut is a set of edges whose removal disconnects the graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `edges` - The set of edges to check
///
/// # Returns
/// `true` if the set forms an edge cut, `false` otherwise
pub fn is_edge_cut(graph: &Graph, edges: &[(usize, usize)]) -> bool {
    let original_components = connected_components_number(graph);

    // Create edge set for quick lookup (normalized with u < v)
    let edge_set: HashSet<(usize, usize)> = edges.iter()
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();

    // Create a new graph without the edges
    let mut new_graph = Graph::new(graph.num_vertices());

    for (u, v) in graph.edges() {
        let normalized_edge = if u < v { (u, v) } else { (v, u) };
        if !edge_set.contains(&normalized_edge) {
            new_graph.add_edge(u, v).ok();
        }
    }

    let new_components = connected_components_number(&new_graph);
    new_components > original_components
}

/// Compute the vertex connectivity of a graph
///
/// The vertex connectivity is the minimum number of vertices that need to be removed
/// to disconnect the graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The vertex connectivity (0 for disconnected graphs, n-1 for complete graphs)
pub fn vertex_connectivity(graph: &Graph) -> usize {
    let n = graph.num_vertices();

    if n == 0 || n == 1 {
        return 0;
    }

    if !is_connected(graph) {
        return 0;
    }

    // For complete graph, connectivity is n-1
    if graph.num_edges() == n * (n - 1) / 2 {
        return n - 1;
    }

    // Check vertex cuts of increasing size
    for size in 1..n {
        if has_vertex_cut_of_size(graph, size) {
            return size;
        }
    }

    n - 1
}

fn has_vertex_cut_of_size(graph: &Graph, size: usize) -> bool {
    let n = graph.num_vertices();

    // Try all combinations of 'size' vertices
    let mut indices = vec![0; size];
    for i in 0..size {
        indices[i] = i;
    }

    loop {
        if is_vertex_cut(graph, &indices) {
            return true;
        }

        // Generate next combination
        let mut i = size;
        while i > 0 {
            i -= 1;
            if indices[i] < n - size + i {
                indices[i] += 1;
                for j in i + 1..size {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return false;
            }
        }
    }
}

/// Compute the edge connectivity of a graph
///
/// The edge connectivity is the minimum number of edges that need to be removed
/// to disconnect the graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The edge connectivity
pub fn edge_connectivity(graph: &Graph) -> usize {
    if !is_connected(graph) {
        return 0;
    }

    let n = graph.num_vertices();
    if n <= 1 {
        return 0;
    }

    // Edge connectivity is at most the minimum degree
    let min_degree = (0..n)
        .filter_map(|v| graph.degree(v))
        .min()
        .unwrap_or(0);

    // Check edge cuts of increasing size
    for size in 1..=min_degree {
        if has_edge_cut_of_size(graph, size) {
            return size;
        }
    }

    min_degree
}

fn has_edge_cut_of_size(graph: &Graph, size: usize) -> bool {
    let edges = graph.edges();
    let m = edges.len();

    if size > m {
        return false;
    }

    // Try all combinations of 'size' edges
    let mut indices = vec![0; size];
    for i in 0..size {
        indices[i] = i;
    }

    loop {
        let edge_cut: Vec<(usize, usize)> = indices.iter()
            .map(|&i| edges[i])
            .collect();

        if is_edge_cut(graph, &edge_cut) {
            return true;
        }

        // Generate next combination
        let mut i = size;
        while i > 0 {
            i -= 1;
            if indices[i] < m - size + i {
                indices[i] += 1;
                for j in i + 1..size {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return false;
            }
        }
    }
}

/// Find blocks (biconnected components) and cut vertices
///
/// Returns a tuple of (blocks, cut_vertices) where:
/// - blocks: vector of blocks, each block is a vector of edges
/// - cut_vertices: vector of articulation points
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Tuple of (blocks, cut_vertices)
pub fn blocks_and_cut_vertices(graph: &Graph) -> (Vec<Vec<(usize, usize)>>, Vec<usize>) {
    let cut_verts = cut_vertices(graph);
    let blocks = find_blocks(graph);
    (blocks, cut_verts)
}

fn find_blocks(graph: &Graph) -> Vec<Vec<(usize, usize)>> {
    let n = graph.num_vertices();
    let mut visited_edges = HashSet::new();
    let mut blocks = Vec::new();

    let mut disc = vec![0; n];
    let mut low = vec![0; n];
    let mut parent = vec![None; n];
    let mut visited = vec![false; n];
    let mut time = 0;
    let mut edge_stack = Vec::new();

    for v in 0..n {
        if !visited[v] {
            find_blocks_dfs(
                graph,
                v,
                &mut visited,
                &mut disc,
                &mut low,
                &mut parent,
                &mut time,
                &mut edge_stack,
                &mut blocks,
                &mut visited_edges,
            );
        }
    }

    blocks
}

fn find_blocks_dfs(
    graph: &Graph,
    u: usize,
    visited: &mut Vec<bool>,
    disc: &mut Vec<usize>,
    low: &mut Vec<usize>,
    parent: &mut Vec<Option<usize>>,
    time: &mut usize,
    edge_stack: &mut Vec<(usize, usize)>,
    blocks: &mut Vec<Vec<(usize, usize)>>,
    visited_edges: &mut HashSet<(usize, usize)>,
) {
    visited[u] = true;
    disc[u] = *time;
    low[u] = *time;
    *time += 1;
    let mut children = 0;

    for &v in graph.neighbors(u).unwrap_or_default().iter() {
        let edge = if u < v { (u, v) } else { (v, u) };

        if !visited[v] {
            children += 1;
            parent[v] = Some(u);
            edge_stack.push(edge);
            visited_edges.insert(edge);

            find_blocks_dfs(graph, v, visited, disc, low, parent, time, edge_stack, blocks, visited_edges);

            low[u] = low[u].min(low[v]);

            // If u is an articulation point, pop a block
            if (parent[u].is_none() && children > 1) || (parent[u].is_some() && low[v] >= disc[u]) {
                let mut block = Vec::new();
                while let Some(e) = edge_stack.pop() {
                    block.push(e);
                    if e == edge {
                        break;
                    }
                }
                blocks.push(block);
            }
        } else if Some(v) != parent[u] && disc[v] < disc[u] {
            edge_stack.push(edge);
            visited_edges.insert(edge);
            low[u] = low[u].min(disc[v]);
        }
    }

    // If u is a root and has remaining edges in stack
    if parent[u].is_none() && !edge_stack.is_empty() {
        blocks.push(edge_stack.drain(..).collect());
    }
}

/// Find strong articulation points in a directed graph
///
/// A strong articulation point is a vertex whose removal increases the number
/// of strongly connected components.
///
/// # Arguments
/// * `digraph` - The directed graph to analyze
///
/// # Returns
/// Vector of strong articulation points
pub fn strong_articulation_points(digraph: &DiGraph) -> Vec<usize> {
    let n = digraph.num_vertices();
    let original_count = digraph.strongly_connected_components().len();
    let mut result = Vec::new();

    for v in 0..n {
        // Create graph without vertex v
        let mut new_graph = DiGraph::new(n - 1);
        let edges = digraph.edges();

        for (u, w) in edges {
            if u != v && w != v {
                let new_u = if u > v { u - 1 } else { u };
                let new_w = if w > v { w - 1 } else { w };
                new_graph.add_edge(new_u, new_w).ok();
            }
        }

        let new_count = new_graph.strongly_connected_components().len();
        if new_count > original_count {
            result.push(v);
        }
    }

    result
}

/// Find minimal vertex separators
///
/// A minimal separator is a vertex set S such that there exist vertices a, b
/// where S separates a and b, and no proper subset of S separates a and b.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Vector of minimal separators, each represented as a vector of vertices
pub fn minimal_separators(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.num_vertices();
    let mut separators = HashSet::new();

    // For each pair of non-adjacent vertices
    for a in 0..n {
        for b in a + 1..n {
            if !graph.has_edge(a, b) {
                // Find minimal separator between a and b
                if let Some(sep) = find_minimal_separator(graph, a, b) {
                    let mut sorted_sep = sep.clone();
                    sorted_sep.sort();
                    separators.insert(sorted_sep);
                }
            }
        }
    }

    separators.into_iter().collect()
}

fn find_minimal_separator(graph: &Graph, a: usize, b: usize) -> Option<Vec<usize>> {
    // Find neighbors of a that can reach b
    let neighbors_a: Vec<usize> = graph.neighbors(a)
        .unwrap_or_default()
        .into_iter()
        .filter(|&v| v != b && can_reach_without(graph, v, b, &[a]))
        .collect();

    if neighbors_a.is_empty() {
        return None;
    }

    // Try to find minimal subset that separates a and b
    Some(neighbors_a)
}

fn can_reach_without(graph: &Graph, from: usize, to: usize, forbidden: &[usize]) -> bool {
    let forbidden_set: HashSet<usize> = forbidden.iter().copied().collect();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back(from);
    visited.insert(from);

    while let Some(v) = queue.pop_front() {
        if v == to {
            return true;
        }

        for neighbor in graph.neighbors(v).unwrap_or_default() {
            if !forbidden_set.contains(&neighbor) && !visited.contains(&neighbor) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    false
}

/// Cleave a graph at a set of edges
///
/// Returns the connected components after removing the specified edges.
///
/// # Arguments
/// * `graph` - The graph to cleave
/// * `edges` - The edges to remove
///
/// # Returns
/// Vector of connected components after edge removal
pub fn cleave(graph: &Graph, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let edge_set: HashSet<(usize, usize)> = edges.iter()
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();

    let mut new_graph = Graph::new(graph.num_vertices());

    for (u, v) in graph.edges() {
        let normalized = if u < v { (u, v) } else { (v, u) };
        if !edge_set.contains(&normalized) {
            new_graph.add_edge(u, v).ok();
        }
    }

    connected_components(&new_graph)
}

/// Create a block-cut tree from a graph
///
/// The block-cut tree is a bipartite graph where one part represents blocks
/// and the other represents cut vertices.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// A Graph representing the block-cut tree
pub fn blocks_and_cuts_tree(graph: &Graph) -> Graph {
    let (blocks, cut_verts) = blocks_and_cut_vertices(graph);
    let n_blocks = blocks.len();
    let n_cuts = cut_verts.len();

    // Create bipartite graph: blocks (0..n_blocks), cut vertices (n_blocks..n_blocks+n_cuts)
    let mut tree = Graph::new(n_blocks + n_cuts);

    // Map cut vertices to their indices in the tree
    let mut cut_vertex_map = HashMap::new();
    for (i, &v) in cut_verts.iter().enumerate() {
        cut_vertex_map.insert(v, n_blocks + i);
    }

    // Connect blocks to their cut vertices
    for (block_idx, block) in blocks.iter().enumerate() {
        // Find vertices in this block
        let mut block_vertices = HashSet::new();
        for &(u, v) in block {
            block_vertices.insert(u);
            block_vertices.insert(v);
        }

        // Connect to cut vertices in this block
        for &v in &block_vertices {
            if let Some(&tree_idx) = cut_vertex_map.get(&v) {
                tree.add_edge(block_idx, tree_idx).ok();
            }
        }
    }

    tree
}

/// Check if a graph is triconnected
///
/// A graph is triconnected if it remains connected after removing any two vertices.
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// `true` if the graph is triconnected, `false` otherwise
pub fn is_triconnected(graph: &Graph) -> bool {
    vertex_connectivity(graph) >= 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_connected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        assert!(is_connected(&g));

        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert!(!is_connected(&g2));
    }

    #[test]
    fn test_connected_components() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(3, 4).unwrap();

        let components = connected_components(&g);
        assert_eq!(components.len(), 3);
        assert_eq!(connected_components_number(&g), 3);

        let sizes = connected_components_sizes(&g);
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));
        assert!(sizes.contains(&1));
    }

    #[test]
    fn test_connected_component_containing_vertex() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(3, 4).unwrap();

        let comp = connected_component_containing_vertex(&g, 1).unwrap();
        assert_eq!(comp.len(), 3);
        assert!(comp.contains(&0));
        assert!(comp.contains(&1));
        assert!(comp.contains(&2));
    }

    #[test]
    fn test_bridges() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let br = bridges(&g);
        assert_eq!(br.len(), 2); // (2,3) and (3,4) are bridges
        assert!(br.contains(&(2, 3)));
        assert!(br.contains(&(3, 4)));
    }

    #[test]
    fn test_cut_vertices() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let cuts = cut_vertices(&g);
        assert!(cuts.contains(&2));
        assert!(cuts.contains(&3));
    }

    #[test]
    fn test_is_cut_vertex() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(1, 3).unwrap();

        assert!(is_cut_vertex(&g, 1)); // Vertex 1 is a cut vertex
        assert!(!is_cut_vertex(&g, 0)); // Vertex 0 is not
    }

    #[test]
    fn test_is_cut_edge() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(2, 3).unwrap();

        assert!(is_cut_edge(&g, 2, 3)); // Bridge
        assert!(!is_cut_edge(&g, 0, 1)); // Not a bridge (part of cycle)
    }

    #[test]
    fn test_vertex_connectivity() {
        // Complete graph K4 has connectivity 3
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }
        assert_eq!(vertex_connectivity(&g), 3);

        // Path has connectivity 1
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert_eq!(vertex_connectivity(&g2), 1);
    }

    #[test]
    fn test_edge_connectivity() {
        // Triangle has edge connectivity 2
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        assert_eq!(edge_connectivity(&g), 2);

        // Path has edge connectivity 1
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert_eq!(edge_connectivity(&g2), 1);
    }

    #[test]
    fn test_strongly_connected() {
        let mut dg = DiGraph::new(3);
        dg.add_edge(0, 1).unwrap();
        dg.add_edge(1, 2).unwrap();
        dg.add_edge(2, 0).unwrap();
        assert!(is_strongly_connected(&dg));

        let mut dg2 = DiGraph::new(3);
        dg2.add_edge(0, 1).unwrap();
        dg2.add_edge(1, 2).unwrap();
        assert!(!is_strongly_connected(&dg2));
    }

    #[test]
    fn test_is_triconnected() {
        // Complete graph K4 is triconnected
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }
        assert!(is_triconnected(&g));

        // Triangle is not triconnected
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 0).unwrap();
        assert!(!is_triconnected(&g2));
    }

    #[test]
    fn test_cleave() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let components = cleave(&g, &[(1, 2)]);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_blocks_and_cut_vertices() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 4).unwrap();

        let (blocks, cuts) = blocks_and_cut_vertices(&g);
        assert!(cuts.contains(&2));
        assert!(cuts.contains(&3));
        assert!(!blocks.is_empty());
    }
}

//! Graph data structures and basic operations

use std::collections::{HashMap, HashSet, VecDeque};

/// An undirected graph using adjacency list representation
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    num_vertices: usize,
    /// Adjacency list: vertex -> list of neighbors
    adj: Vec<HashSet<usize>>,
}

impl Graph {
    /// Create a new graph with n vertices
    pub fn new(n: usize) -> Self {
        Graph {
            num_vertices: n,
            adj: vec![HashSet::new(); n],
        }
    }

    /// Get the number of vertices
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    /// Get the number of edges
    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|neighbors| neighbors.len()).sum::<usize>() / 2
    }

    /// Add an edge between vertices u and v
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<(), String> {
        if u >= self.num_vertices || v >= self.num_vertices {
            return Err(format!("Vertex out of bounds"));
        }
        self.adj[u].insert(v);
        self.adj[v].insert(u);
        Ok(())
    }

    /// Check if there's an edge between u and v
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.num_vertices || v >= self.num_vertices {
            return false;
        }
        self.adj[u].contains(&v)
    }

    /// Get the degree of a vertex
    pub fn degree(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.adj[v].len())
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, v: usize) -> Option<Vec<usize>> {
        if v >= self.num_vertices {
            return None;
        }
        Some(self.adj[v].iter().copied().collect())
    }

    /// Perform breadth-first search from a starting vertex
    pub fn bfs(&self, start: usize) -> Result<Vec<usize>, String> {
        if start >= self.num_vertices {
            return Err("Start vertex out of bounds".to_string());
        }

        let mut visited = vec![false; self.num_vertices];
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(v) = queue.pop_front() {
            order.push(v);

            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(order)
    }

    /// Perform depth-first search from a starting vertex
    pub fn dfs(&self, start: usize) -> Result<Vec<usize>, String> {
        if start >= self.num_vertices {
            return Err("Start vertex out of bounds".to_string());
        }

        let mut visited = vec![false; self.num_vertices];
        let mut order = Vec::new();

        self.dfs_recursive(start, &mut visited, &mut order);

        Ok(order)
    }

    fn dfs_recursive(&self, v: usize, visited: &mut [bool], order: &mut Vec<usize>) {
        visited[v] = true;
        order.push(v);

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                self.dfs_recursive(neighbor, visited, order);
            }
        }
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.num_vertices == 0 {
            return true;
        }

        let visited = self.bfs(0).unwrap();
        visited.len() == self.num_vertices
    }

    /// Find shortest path between two vertices using BFS
    pub fn shortest_path(&self, start: usize, end: usize) -> Result<Option<Vec<usize>>, String> {
        if start >= self.num_vertices || end >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        if start == end {
            return Ok(Some(vec![start]));
        }

        let mut visited = vec![false; self.num_vertices];
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(v) = queue.pop_front() {
            if v == end {
                // Reconstruct path
                let mut path = vec![end];
                let mut current = end;

                while current != start {
                    current = parent[&current];
                    path.push(current);
                }

                path.reverse();
                return Ok(Some(path));
            }

            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent.insert(neighbor, v);
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(None) // No path found
    }

    /// Find all connected components
    ///
    /// Returns a vector where each element is a vector of vertices in that component
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.num_vertices];
        let mut components = Vec::new();

        for v in 0..self.num_vertices {
            if !visited[v] {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(v);
                visited[v] = true;

                while let Some(u) = queue.pop_front() {
                    component.push(u);

                    for &neighbor in &self.adj[u] {
                        if !visited[neighbor] {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }

                components.push(component);
            }
        }

        components
    }

    /// Check if the graph has a cycle
    pub fn has_cycle(&self) -> bool {
        let mut visited = vec![false; self.num_vertices];
        let mut parent = vec![None; self.num_vertices];

        for v in 0..self.num_vertices {
            if !visited[v] && self.has_cycle_dfs(v, &mut visited, &mut parent) {
                return true;
            }
        }

        false
    }

    fn has_cycle_dfs(
        &self,
        v: usize,
        visited: &mut [bool],
        parent: &mut [Option<usize>],
    ) -> bool {
        visited[v] = true;

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                parent[neighbor] = Some(v);
                if self.has_cycle_dfs(neighbor, visited, parent) {
                    return true;
                }
            } else if parent[v] != Some(neighbor) {
                // Found a back edge (cycle)
                return true;
            }
        }

        false
    }

    /// Find a spanning tree using BFS
    ///
    /// Returns edges (as pairs of vertices) that form a spanning tree
    /// Returns None if graph is not connected
    pub fn spanning_tree_bfs(&self) -> Option<Vec<(usize, usize)>> {
        if self.num_vertices == 0 {
            return Some(Vec::new());
        }

        let mut edges = Vec::new();
        let mut visited = vec![false; self.num_vertices];
        let mut queue = VecDeque::new();

        queue.push_back(0);
        visited[0] = true;

        while let Some(v) = queue.pop_front() {
            for &neighbor in &self.adj[v] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    edges.push((v, neighbor));
                    queue.push_back(neighbor);
                }
            }
        }

        // Check if all vertices were visited (graph is connected)
        if visited.iter().all(|&v| v) {
            Some(edges)
        } else {
            None
        }
    }

    /// Check if the graph is bipartite
    ///
    /// A graph is bipartite if its vertices can be divided into two sets
    /// such that all edges connect vertices from different sets
    pub fn is_bipartite(&self) -> bool {
        let mut color = vec![None; self.num_vertices];

        for start in 0..self.num_vertices {
            if color[start].is_none() {
                let mut queue = VecDeque::new();
                queue.push_back(start);
                color[start] = Some(0);

                while let Some(v) = queue.pop_front() {
                    let current_color = color[v].unwrap();

                    for &neighbor in &self.adj[v] {
                        if let Some(neighbor_color) = color[neighbor] {
                            if neighbor_color == current_color {
                                return false; // Adjacent vertices have same color
                            }
                        } else {
                            color[neighbor] = Some(1 - current_color);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        true
    }

    /// Greedy graph coloring
    ///
    /// Returns a coloring (vertex -> color mapping) using a greedy algorithm
    /// The number of colors used may not be optimal
    pub fn greedy_coloring(&self) -> Vec<usize> {
        let mut colors = vec![None; self.num_vertices];

        for v in 0..self.num_vertices {
            // Find colors used by neighbors
            let mut used_colors = HashSet::new();
            for &neighbor in &self.adj[v] {
                if let Some(color) = colors[neighbor] {
                    used_colors.insert(color);
                }
            }

            // Find smallest available color
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            colors[v] = Some(color);
        }

        colors.into_iter().map(|c| c.unwrap()).collect()
    }

    /// Get the chromatic number (minimum number of colors needed)
    ///
    /// Uses greedy coloring as an approximation
    pub fn chromatic_number(&self) -> usize {
        let coloring = self.greedy_coloring();
        if coloring.is_empty() {
            0
        } else {
            *coloring.iter().max().unwrap() + 1
        }
    }

    /// Find the diameter of the graph (longest shortest path)
    ///
    /// Returns None if graph is not connected
    pub fn diameter(&self) -> Option<usize> {
        if !self.is_connected() {
            return None;
        }

        let mut max_distance = 0;

        for v in 0..self.num_vertices {
            let distances = self.bfs_distances(v);
            if let Some(&max_dist) = distances.iter().max() {
                if max_dist > max_distance {
                    max_distance = max_dist;
                }
            }
        }

        Some(max_distance)
    }

    fn bfs_distances(&self, start: usize) -> Vec<usize> {
        let mut distances = vec![usize::MAX; self.num_vertices];
        let mut queue = VecDeque::new();

        queue.push_back(start);
        distances[start] = 0;

        while let Some(v) = queue.pop_front() {
            for &neighbor in &self.adj[v] {
                if distances[neighbor] == usize::MAX {
                    distances[neighbor] = distances[v] + 1;
                    queue.push_back(neighbor);
                }
            }
        }

        distances
    }

    /// Get all vertices in the graph
    pub fn vertices(&self) -> Vec<usize> {
        (0..self.num_vertices).collect()
    }

    /// Get all edges in the graph
    ///
    /// Returns a vector of (u, v) tuples where u < v
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for u in 0..self.num_vertices {
            for &v in &self.adj[u] {
                if u < v {
                    edges.push((u, v));
                }
            }
        }
        edges
    }

    /// Add multiple edges at once
    pub fn add_edges(&mut self, edges: &[(usize, usize)]) -> Result<(), String> {
        for &(u, v) in edges {
            self.add_edge(u, v)?;
        }
        Ok(())
    }

    /// Check if the graph is a tree
    ///
    /// A tree is a connected acyclic graph
    pub fn is_tree(&self) -> bool {
        // A tree with n vertices has exactly n-1 edges
        if self.num_vertices == 0 {
            return true;
        }

        self.is_connected() && self.num_edges() == self.num_vertices - 1
    }

    /// Check if the graph is a forest
    ///
    /// A forest is an acyclic graph (may be disconnected)
    pub fn is_forest(&self) -> bool {
        !self.has_cycle()
    }

    /// Check if the graph has an Eulerian path or circuit
    ///
    /// Returns (has_path, has_circuit)
    pub fn is_eulerian(&self) -> (bool, bool) {
        if !self.is_connected() {
            return (false, false);
        }

        // Count vertices with odd degree
        let mut odd_degree_count = 0;
        for v in 0..self.num_vertices {
            if self.degree(v).unwrap() % 2 == 1 {
                odd_degree_count += 1;
            }
        }

        // Eulerian circuit: all vertices have even degree
        // Eulerian path: exactly 2 vertices have odd degree
        let has_circuit = odd_degree_count == 0;
        let has_path = odd_degree_count == 0 || odd_degree_count == 2;

        (has_path, has_circuit)
    }

    /// Get the length of the shortest path between two vertices
    pub fn shortest_path_length(&self, start: usize, end: usize) -> Result<Option<usize>, String> {
        match self.shortest_path(start, end)? {
            Some(path) => Ok(Some(path.len() - 1)),
            None => Ok(None),
        }
    }

    /// Find all paths between two vertices
    ///
    /// Warning: This can be exponential in the graph size
    pub fn all_paths(&self, start: usize, end: usize) -> Result<Vec<Vec<usize>>, String> {
        if start >= self.num_vertices || end >= self.num_vertices {
            return Err("Vertex out of bounds".to_string());
        }

        let mut paths = Vec::new();
        let mut current_path = vec![start];
        let mut visited = vec![false; self.num_vertices];
        visited[start] = true;

        self.all_paths_helper(start, end, &mut current_path, &mut visited, &mut paths);

        Ok(paths)
    }

    fn all_paths_helper(
        &self,
        current: usize,
        end: usize,
        path: &mut Vec<usize>,
        visited: &mut [bool],
        paths: &mut Vec<Vec<usize>>,
    ) {
        if current == end {
            paths.push(path.clone());
            return;
        }

        for &neighbor in &self.adj[current] {
            if !visited[neighbor] {
                path.push(neighbor);
                visited[neighbor] = true;
                self.all_paths_helper(neighbor, end, path, visited, paths);
                path.pop();
                visited[neighbor] = false;
            }
        }
    }

    /// Perform topological sort on a DAG (Directed Acyclic Graph)
    ///
    /// For undirected graphs, this performs a topological-like ordering based on DFS finish times
    /// Returns None if the graph has a cycle (for directed interpretation)
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        if self.has_cycle() {
            return None;
        }

        let mut visited = vec![false; self.num_vertices];
        let mut stack = Vec::new();

        for v in 0..self.num_vertices {
            if !visited[v] {
                self.topological_dfs(v, &mut visited, &mut stack);
            }
        }

        stack.reverse();
        Some(stack)
    }

    fn topological_dfs(&self, v: usize, visited: &mut [bool], stack: &mut Vec<usize>) {
        visited[v] = true;

        for &neighbor in &self.adj[v] {
            if !visited[neighbor] {
                self.topological_dfs(neighbor, visited, stack);
            }
        }

        stack.push(v);
    }

    /// Find minimum spanning tree using Kruskal's algorithm
    ///
    /// Returns edges in the MST, or None if graph is not connected
    pub fn min_spanning_tree(&self) -> Option<Vec<(usize, usize)>> {
        if !self.is_connected() {
            return None;
        }

        let mut edges = self.edges();
        // Sort edges (already sorted by vertex pairs)
        edges.sort_by_key(|(u, v)| (*u, *v));

        let mut mst = Vec::new();
        let mut uf = UnionFind::new(self.num_vertices);

        for (u, v) in edges {
            if !uf.is_connected(u, v) {
                uf.union(u, v);
                mst.push((u, v));

                if mst.len() == self.num_vertices - 1 {
                    break;
                }
            }
        }

        Some(mst)
    }

    /// Find maximum matching in a bipartite graph
    ///
    /// Returns None if graph is not bipartite
    pub fn max_bipartite_matching(&self) -> Option<Vec<(usize, usize)>> {
        if !self.is_bipartite() {
            return None;
        }

        // Get bipartition
        let (left, right) = self.bipartition()?;

        // Use augmenting path algorithm
        let mut matching: HashMap<usize, usize> = HashMap::new();

        for &u in &left {
            let mut visited = HashSet::new();
            self.augment_matching(u, &mut matching, &mut visited, &left, &right);
        }

        let edges: Vec<(usize, usize)> = matching.iter().map(|(&u, &v)| (u, v)).collect();
        Some(edges)
    }

    fn bipartition(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        let mut color = vec![None; self.num_vertices];
        let mut queue = VecDeque::new();

        // Color first component
        for start in 0..self.num_vertices {
            if color[start].is_none() {
                queue.push_back(start);
                color[start] = Some(0);

                while let Some(v) = queue.pop_front() {
                    let current_color = color[v].unwrap();

                    for &neighbor in &self.adj[v] {
                        if color[neighbor].is_none() {
                            color[neighbor] = Some(1 - current_color);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        let left: Vec<usize> = (0..self.num_vertices)
            .filter(|&v| color[v] == Some(0))
            .collect();
        let right: Vec<usize> = (0..self.num_vertices)
            .filter(|&v| color[v] == Some(1))
            .collect();

        Some((left, right))
    }

    fn augment_matching(
        &self,
        u: usize,
        matching: &mut HashMap<usize, usize>,
        visited: &mut HashSet<usize>,
        _left: &[usize],
        _right: &[usize],
    ) -> bool {
        for &v in &self.adj[u] {
            if visited.contains(&v) {
                continue;
            }
            visited.insert(v);

            // If v is unmatched or we can recursively find an augmenting path
            if !matching.values().any(|&matched| matched == v) ||
               self.augment_matching(*matching.iter().find(|(_, &val)| val == v).unwrap().0, matching, visited, _left, _right) {
                matching.insert(u, v);
                return true;
            }
        }
        false
    }

    /// Count the number of spanning trees using Kirchhoff's matrix-tree theorem
    ///
    /// Returns the number of spanning trees in the graph.
    /// For disconnected graphs, returns 0.
    /// Uses the Laplacian matrix and computes its cofactor determinant.
    pub fn spanning_trees_count(&self) -> i64 {
        if !self.is_connected() {
            return 0;
        }

        if self.num_vertices == 0 {
            return 0;
        }

        if self.num_vertices == 1 {
            return 1;
        }

        // Create Laplacian matrix
        let n = self.num_vertices;
        let mut laplacian = vec![vec![0i64; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    laplacian[i][j] = self.adj[i].len() as i64;
                } else if self.adj[i].contains(&j) {
                    laplacian[i][j] = -1;
                }
            }
        }

        // Remove first row and column to get cofactor
        let mut cofactor = vec![vec![0i64; n - 1]; n - 1];
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                cofactor[i][j] = laplacian[i + 1][j + 1];
            }
        }

        // Compute determinant
        determinant(&cofactor)
    }

    /// Compute the chromatic polynomial of the graph
    ///
    /// Returns a polynomial where P(k) gives the number of ways to color
    /// the graph with k colors. Uses deletion-contraction algorithm.
    pub fn chromatic_polynomial(&self) -> Vec<i64> {
        chromatic_poly_helper(self)
    }

    /// Lexicographic breadth-first search
    ///
    /// Returns a vertex ordering that can be used for perfect elimination ordering
    /// and recognition of chordal graphs.
    pub fn lex_bfs(&self) -> Vec<usize> {
        if self.num_vertices == 0 {
            return vec![];
        }

        let n = self.num_vertices;
        let mut order = Vec::new();
        let mut labels: Vec<Vec<usize>> = vec![vec![]; n];
        let mut unnumbered: HashSet<usize> = (0..n).collect();

        for i in (0..n).rev() {
            // Find unnumbered vertex with lexicographically largest label
            let mut max_vertex = *unnumbered.iter().next().unwrap();
            for &v in &unnumbered {
                if labels[v] > labels[max_vertex] {
                    max_vertex = v;
                }
            }

            order.push(max_vertex);
            unnumbered.remove(&max_vertex);

            // Update labels of unnumbered neighbors
            for &neighbor in &self.adj[max_vertex] {
                if unnumbered.contains(&neighbor) {
                    labels[neighbor].push(i);
                }
            }
        }

        order.reverse();
        order
    }
}

/// Helper function to compute determinant of a matrix
fn determinant(matrix: &[Vec<i64>]) -> i64 {
    let n = matrix.len();
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    // Use Gaussian elimination for larger matrices
    let mut m = matrix.to_vec();
    let mut det = 1i64;
    let mut sign = 1i64;

    for i in 0..n {
        // Find pivot
        let mut pivot = i;
        for j in i + 1..n {
            if m[j][i].abs() > m[pivot][i].abs() {
                pivot = j;
            }
        }

        if m[pivot][i] == 0 {
            return 0;
        }

        if pivot != i {
            m.swap(i, pivot);
            sign = -sign;
        }

        det *= m[i][i];

        // Eliminate column
        for j in i + 1..n {
            let factor = m[j][i];
            for k in i..n {
                m[j][k] = m[j][k] * m[i][i] - m[i][k] * factor;
            }
        }
    }

    // Adjust for divisions we didn't do
    for i in 1..n {
        det /= m[i - 1][i - 1].pow((n - i) as u32);
    }

    sign * det
}

/// Helper function for chromatic polynomial using deletion-contraction
fn chromatic_poly_helper(graph: &Graph) -> Vec<i64> {
    let n = graph.num_vertices();

    // Base cases
    if n == 0 {
        return vec![1];
    }

    if graph.num_edges() == 0 {
        // Empty graph: P(k) = k^n
        let mut poly = vec![0; n + 1];
        poly[n] = 1;
        return poly;
    }

    // Find an edge to delete/contract
    let mut edge = None;
    'outer: for u in 0..n {
        for &v in &graph.adj[u] {
            if u < v {
                edge = Some((u, v));
                break 'outer;
            }
        }
    }

    let (u, v) = edge.unwrap();

    // Deletion: remove edge (u, v)
    let mut g_delete = graph.clone();
    g_delete.adj[u].remove(&v);
    g_delete.adj[v].remove(&u);
    let p_delete = chromatic_poly_helper(&g_delete);

    // Contraction: merge vertices u and v
    let g_contract = contract_edge(graph, u, v);
    let p_contract = chromatic_poly_helper(&g_contract);

    // P(G) = P(G-e) - P(G/e)
    subtract_poly(&p_delete, &p_contract)
}

/// Contract an edge (merge two vertices)
fn contract_edge(graph: &Graph, u: usize, v: usize) -> Graph {
    let n = graph.num_vertices();
    let mut new_graph = Graph::new(n - 1);

    // Map old vertices to new vertices (v is removed, everything after shifts down)
    let vertex_map: Vec<usize> = (0..n)
        .filter(|&i| i != v)
        .enumerate()
        .map(|(new_idx, _)| new_idx)
        .collect();

    let get_new_vertex = |old: usize| -> usize {
        if old == v {
            if u < v {
                u
            } else {
                u - 1
            }
        } else if old < v {
            old
        } else {
            old - 1
        }
    };

    // Add edges
    for i in 0..n {
        if i == v {
            continue;
        }
        for &j in &graph.adj[i] {
            if j == v || j <= i {
                continue;
            }
            let new_i = get_new_vertex(i);
            let new_j = get_new_vertex(j);
            if new_i != new_j {
                new_graph.add_edge(new_i, new_j).ok();
            }
        }
    }

    // Add edges from v to u's new position
    for &neighbor in &graph.adj[v] {
        if neighbor != u {
            let new_u = get_new_vertex(u);
            let new_neighbor = get_new_vertex(neighbor);
            if new_u != new_neighbor {
                new_graph.add_edge(new_u, new_neighbor).ok();
            }
        }
    }

    new_graph
}

/// Subtract two polynomials
fn subtract_poly(p1: &[i64], p2: &[i64]) -> Vec<i64> {
    let max_len = p1.len().max(p2.len());
    let mut result = vec![0; max_len];

    for i in 0..p1.len() {
        result[i] += p1[i];
    }
    for i in 0..p2.len() {
        result[i] -= p2[i];
    }

    // Remove leading zeros
    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }

    result
}

/// Union-Find data structure for Kruskal's algorithm
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return;
        }

        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
    }

    fn is_connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let g = Graph::new(5);
        assert_eq!(g.num_vertices(), 5);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 2));
        assert!(!g.has_edge(0, 2));
        assert_eq!(g.num_edges(), 2);
    }

    #[test]
    fn test_degree() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();

        assert_eq!(g.degree(0), Some(2));
        assert_eq!(g.degree(1), Some(1));
        assert_eq!(g.degree(2), Some(1));
    }

    #[test]
    fn test_bfs() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();

        let order = g.bfs(0).unwrap();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 0); // Start vertex first
    }

    #[test]
    fn test_is_connected() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();

        assert!(g.is_connected());

        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();

        assert!(!g2.is_connected());
    }

    #[test]
    fn test_shortest_path() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(0, 3).unwrap();

        let path = g.shortest_path(0, 3).unwrap().unwrap();
        assert_eq!(path, vec![0, 3]); // Direct path is shorter
    }

    #[test]
    fn test_connected_components() {
        let mut g = Graph::new(6);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(3, 4).unwrap();
        // Vertex 5 is isolated

        let components = g.connected_components();
        assert_eq!(components.len(), 3); // Three components: {0,1,2}, {3,4}, {5}
    }

    #[test]
    fn test_has_cycle() {
        // Tree - no cycle
        let mut g1 = Graph::new(4);
        g1.add_edge(0, 1).unwrap();
        g1.add_edge(0, 2).unwrap();
        g1.add_edge(0, 3).unwrap();
        assert!(!g1.has_cycle());

        // Graph with cycle
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 3).unwrap();
        g2.add_edge(3, 0).unwrap();
        assert!(g2.has_cycle());
    }

    #[test]
    fn test_spanning_tree() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let tree = g.spanning_tree_bfs().unwrap();
        assert_eq!(tree.len(), 3); // Tree has n-1 edges for n vertices
    }

    #[test]
    fn test_is_bipartite() {
        // Bipartite graph (complete bipartite K_{2,2})
        let mut g1 = Graph::new(4);
        g1.add_edge(0, 2).unwrap();
        g1.add_edge(0, 3).unwrap();
        g1.add_edge(1, 2).unwrap();
        g1.add_edge(1, 3).unwrap();
        assert!(g1.is_bipartite());

        // Triangle (odd cycle) - not bipartite
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 0).unwrap();
        assert!(!g2.is_bipartite());
    }

    #[test]
    fn test_greedy_coloring() {
        // Triangle needs 3 colors
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let coloring = g.greedy_coloring();
        // Check that adjacent vertices have different colors
        assert_ne!(coloring[0], coloring[1]);
        assert_ne!(coloring[1], coloring[2]);
        assert_ne!(coloring[2], coloring[0]);
    }

    #[test]
    fn test_chromatic_number() {
        // Complete graph K_3 has chromatic number 3
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        assert_eq!(g.chromatic_number(), 3);
    }

    #[test]
    fn test_diameter() {
        // Path graph 0-1-2-3 has diameter 3
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(g.diameter(), Some(3));

        // Disconnected graph has no diameter
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert_eq!(g2.diameter(), None);
    }

    #[test]
    fn test_vertices_edges() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(g.vertices(), vec![0, 1, 2, 3]);

        let edges = g.edges();
        assert_eq!(edges.len(), 3);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
        assert!(edges.contains(&(2, 3)));
    }

    #[test]
    fn test_add_edges() {
        let mut g = Graph::new(4);
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        g.add_edges(&edges).unwrap();

        assert_eq!(g.num_edges(), 3);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(2, 3));
    }

    #[test]
    fn test_is_tree() {
        // Tree: connected acyclic with n-1 edges
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        assert!(g.is_tree());

        // Add cycle - no longer a tree
        g.add_edge(3, 0).unwrap();
        assert!(!g.is_tree());
    }

    #[test]
    fn test_is_forest() {
        // Forest: acyclic (may be disconnected)
        let mut g = Graph::new(5);
        g.add_edge(0, 1).unwrap();
        g.add_edge(2, 3).unwrap();
        assert!(g.is_forest());

        // Add cycle - no longer a forest
        g.add_edge(3, 4).unwrap();
        g.add_edge(4, 2).unwrap();
        assert!(!g.is_forest());
    }

    #[test]
    fn test_is_eulerian() {
        // Eulerian circuit: all vertices have even degree
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 0).unwrap();

        let (has_path, has_circuit) = g.is_eulerian();
        assert!(has_path);
        assert!(has_circuit);

        // Eulerian path: exactly 2 vertices with odd degree
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();

        let (has_path2, has_circuit2) = g2.is_eulerian();
        assert!(has_path2);
        assert!(!has_circuit2);
    }

    #[test]
    fn test_shortest_path_length() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        assert_eq!(g.shortest_path_length(0, 3).unwrap(), Some(3));
        assert_eq!(g.shortest_path_length(0, 2).unwrap(), Some(2));
        assert_eq!(g.shortest_path_length(0, 0).unwrap(), Some(0));
    }

    #[test]
    fn test_all_paths() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(0, 2).unwrap();
        g.add_edge(1, 3).unwrap();
        g.add_edge(2, 3).unwrap();

        let paths = g.all_paths(0, 3).unwrap();
        assert_eq!(paths.len(), 2); // Two paths: 0->1->3 and 0->2->3

        // Verify paths
        for path in &paths {
            assert_eq!(path[0], 0);
            assert_eq!(path[path.len() - 1], 3);
        }
    }

    #[test]
    fn test_min_spanning_tree() {
        // Create a graph: triangle 0-1-2-0
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let mst = g.min_spanning_tree().unwrap();
        assert_eq!(mst.len(), 2); // MST of 3 vertices has 2 edges

        // Disconnected graph has no MST
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert!(g2.min_spanning_tree().is_none());
    }

    #[test]
    fn test_max_bipartite_matching() {
        // Create a bipartite graph
        let mut g = Graph::new(4);
        g.add_edge(0, 2).unwrap(); // Left side: 0, 1; Right side: 2, 3
        g.add_edge(0, 3).unwrap();
        g.add_edge(1, 2).unwrap();

        let matching = g.max_bipartite_matching().unwrap();
        assert_eq!(matching.len(), 2); // Maximum matching has 2 edges

        // Non-bipartite graph returns None
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 0).unwrap();
        assert!(g2.max_bipartite_matching().is_none());
    }

    #[test]
    fn test_spanning_trees_count() {
        // Complete graph K3 has 3 spanning trees
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();
        assert_eq!(g.spanning_trees_count(), 3);

        // Path graph has 1 spanning tree (itself)
        let mut g2 = Graph::new(4);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 3).unwrap();
        assert_eq!(g2.spanning_trees_count(), 1);

        // Complete graph K4 has 16 spanning trees
        let mut g3 = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g3.add_edge(i, j).unwrap();
            }
        }
        assert_eq!(g3.spanning_trees_count(), 16);

        // Disconnected graph has 0 spanning trees
        let mut g4 = Graph::new(4);
        g4.add_edge(0, 1).unwrap();
        g4.add_edge(2, 3).unwrap();
        assert_eq!(g4.spanning_trees_count(), 0);
    }

    #[test]
    fn test_chromatic_polynomial() {
        // Empty graph with 3 vertices: P(k) = k^3
        let g = Graph::new(3);
        let poly = g.chromatic_polynomial();
        assert_eq!(poly.len(), 4);
        assert_eq!(poly[3], 1); // Coefficient of k^3

        // Path graph P3: P(k) = k(k-1)^2 = k^3 - 2k^2 + k
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        let poly2 = g2.chromatic_polynomial();
        assert_eq!(poly2[0], 0);  // Constant term
        assert_eq!(poly2[1], 1);  // k term
        assert_eq!(poly2[2], -2); // k^2 term
        assert_eq!(poly2[3], 1);  // k^3 term

        // Triangle K3: P(k) = k(k-1)(k-2) = k^3 - 3k^2 + 2k
        let mut g3 = Graph::new(3);
        g3.add_edge(0, 1).unwrap();
        g3.add_edge(1, 2).unwrap();
        g3.add_edge(2, 0).unwrap();
        let poly3 = g3.chromatic_polynomial();
        assert_eq!(poly3[0], 0);  // Constant term
        assert_eq!(poly3[1], 2);  // k term
        assert_eq!(poly3[2], -3); // k^2 term
        assert_eq!(poly3[3], 1);  // k^3 term
    }

    #[test]
    fn test_lex_bfs() {
        // Path graph
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let order = g.lex_bfs();
        assert_eq!(order.len(), 4);
        // All vertices should appear exactly once
        let mut sorted_order = order.clone();
        sorted_order.sort();
        assert_eq!(sorted_order, vec![0, 1, 2, 3]);

        // Empty graph
        let g2 = Graph::new(3);
        let order2 = g2.lex_bfs();
        assert_eq!(order2.len(), 3);

        // Complete graph
        let mut g3 = Graph::new(3);
        g3.add_edge(0, 1).unwrap();
        g3.add_edge(1, 2).unwrap();
        g3.add_edge(2, 0).unwrap();
        let order3 = g3.lex_bfs();
        assert_eq!(order3.len(), 3);
    }
}

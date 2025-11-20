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

    /// Check if the graph is directed
    ///
    /// Currently, Graph is always undirected. For directed graphs, use DiGraph.
    pub fn is_directed(&self) -> bool {
        false
    }

    /// Add a new vertex to the graph
    ///
    /// Returns the index of the newly added vertex.
    /// The new vertex initially has no edges.
    pub fn add_vertex(&mut self) -> usize {
        let new_idx = self.num_vertices;
        self.num_vertices += 1;
        self.adj.push(HashSet::new());
        new_idx
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

    /// Check if the graph has a Hamiltonian cycle
    ///
    /// A Hamiltonian cycle visits every vertex exactly once and returns to the start.
    /// Uses backtracking (NP-complete problem, exponential time).
    pub fn is_hamiltonian(&self) -> bool {
        if self.num_vertices == 0 {
            return false;
        }

        // Start from vertex 0 and try to find a Hamiltonian cycle
        let mut path = vec![0];
        let mut visited = vec![false; self.num_vertices];
        visited[0] = true;

        self.hamiltonian_helper(&mut path, &mut visited)
    }

    fn hamiltonian_helper(&self, path: &mut Vec<usize>, visited: &mut Vec<bool>) -> bool {
        // If all vertices are in the path
        if path.len() == self.num_vertices {
            // Check if there's an edge back to the start
            return self.has_edge(*path.last().unwrap(), path[0]);
        }

        let current = *path.last().unwrap();

        // Try each neighbor
        for &neighbor in &self.adj[current] {
            if !visited[neighbor] {
                path.push(neighbor);
                visited[neighbor] = true;

                if self.hamiltonian_helper(path, visited) {
                    return true;
                }

                path.pop();
                visited[neighbor] = false;
            }
        }

        false
    }

    /// Check if the graph is planar
    ///
    /// A graph is planar if it can be drawn on a plane without edge crossings.
    /// Uses a simplified check based on Kuratowski's theorem:
    /// - A graph is planar iff it doesn't contain K5 or K3,3 as a subdivision
    ///
    /// This is a basic implementation using Euler's formula for small graphs.
    pub fn is_planar(&self) -> bool {
        let n = self.num_vertices;
        let m = self.num_edges();

        // Empty or single vertex graph is planar
        if n <= 3 {
            return true;
        }

        // Euler's formula: for a connected planar graph, v - e + f = 2
        // This gives us: e <= 3v - 6 (for simple graphs)
        if m > 3 * n - 6 {
            return false;
        }

        // If bipartite, stronger bound: e <= 2v - 4
        if self.is_bipartite() && m > 2 * n - 4 {
            return false;
        }

        // For small graphs, check for K5 or K3,3 minors
        // K5 has 5 vertices and 10 edges
        if n == 5 && m == 10 {
            // Check if it's a complete graph K5 (not planar)
            let mut is_complete = true;
            for i in 0..5 {
                if self.degree(i) != Some(4) {
                    is_complete = false;
                    break;
                }
            }
            if is_complete {
                return false;
            }
        }

        // K3,3 has 6 vertices and 9 edges
        if n == 6 && m == 9 && self.is_bipartite() {
            // Check if it's complete bipartite K3,3 (not planar)
            let parts = self.bipartition();
            if let Some((left, right)) = parts {
                if left.len() == 3 && right.len() == 3 {
                    // Check if complete
                    let mut is_complete = true;
                    for &u in &left {
                        if self.degree(u) != Some(3) {
                            is_complete = false;
                            break;
                        }
                    }
                    if is_complete {
                        return false;
                    }
                }
            }
        }

        // For other cases, assume planar (full planarity testing is complex)
        true
    }

    /// Enumerate all perfect matchings in the graph
    ///
    /// A perfect matching covers all vertices with non-overlapping edges.
    /// Returns all possible perfect matchings.
    pub fn perfect_matchings(&self) -> Vec<Vec<(usize, usize)>> {
        if self.num_vertices % 2 != 0 {
            // Perfect matching requires even number of vertices
            return vec![];
        }

        let mut matchings = Vec::new();
        let mut current_matching = Vec::new();
        let mut unmatched: HashSet<usize> = (0..self.num_vertices).collect();

        self.find_perfect_matchings(&mut current_matching, &mut unmatched, &mut matchings);

        matchings
    }

    fn find_perfect_matchings(
        &self,
        current: &mut Vec<(usize, usize)>,
        unmatched: &mut HashSet<usize>,
        result: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if unmatched.is_empty() {
            // Found a complete perfect matching
            result.push(current.clone());
            return;
        }

        // Pick the smallest unmatched vertex to reduce branching
        let &u = unmatched.iter().min().unwrap();

        // Try pairing it with each unmatched neighbor
        let neighbors: Vec<usize> = self.adj[u]
            .iter()
            .copied()
            .filter(|v| unmatched.contains(v) && *v > u)
            .collect();

        for v in neighbors {
            // Add edge to matching
            current.push((u, v));
            unmatched.remove(&u);
            unmatched.remove(&v);

            self.find_perfect_matchings(current, unmatched, result);

            // Backtrack
            current.pop();
            unmatched.insert(u);
            unmatched.insert(v);
        }
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

    /// Compute the matching polynomial
    ///
    /// The matching polynomial m(G, x) is defined such that the coefficient of x^k
    /// gives information about matchings in the graph.
    /// For a graph on n vertices: m(G, x) = Σ (-1)^k * m_k(G) * x^(n-2k)
    /// where m_k(G) is the number of k-matchings (matchings with k edges).
    ///
    /// # Returns
    /// Coefficient vector where index i corresponds to the coefficient of x^i
    pub fn matching_polynomial(&self) -> Vec<i64> {
        matching_poly_helper(self)
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

    /// Compute the Cartesian product of two graphs
    ///
    /// The Cartesian product G □ H has vertex set V(G) × V(H).
    /// Two vertices (u, v) and (u', v') are adjacent iff:
    /// - u = u' and vv' ∈ E(H), or
    /// - v = v' and uu' ∈ E(G)
    pub fn cartesian_product(&self, other: &Graph) -> Graph {
        let n1 = self.num_vertices;
        let n2 = other.num_vertices;
        let mut product = Graph::new(n1 * n2);

        // Helper to get vertex index in product graph
        let vertex_idx = |i: usize, j: usize| i * n2 + j;

        // Add edges from first graph (same second coordinate)
        for i in 0..n1 {
            for &neighbor in &self.adj[i] {
                if i < neighbor {
                    for j in 0..n2 {
                        let u = vertex_idx(i, j);
                        let v = vertex_idx(neighbor, j);
                        product.add_edge(u, v).ok();
                    }
                }
            }
        }

        // Add edges from second graph (same first coordinate)
        for j in 0..n2 {
            for &neighbor in &other.adj[j] {
                if j < neighbor {
                    for i in 0..n1 {
                        let u = vertex_idx(i, j);
                        let v = vertex_idx(i, neighbor);
                        product.add_edge(u, v).ok();
                    }
                }
            }
        }

        product
    }

    /// Compute the tensor product (categorical product) of two graphs
    ///
    /// The tensor product G × H has vertex set V(G) × V(H).
    /// Two vertices (u, v) and (u', v') are adjacent iff:
    /// - uu' ∈ E(G) AND vv' ∈ E(H)
    pub fn tensor_product(&self, other: &Graph) -> Graph {
        let n1 = self.num_vertices;
        let n2 = other.num_vertices;
        let mut product = Graph::new(n1 * n2);

        let vertex_idx = |i: usize, j: usize| i * n2 + j;

        for i in 0..n1 {
            for j in 0..n2 {
                for &i_neighbor in &self.adj[i] {
                    for &j_neighbor in &other.adj[j] {
                        let u = vertex_idx(i, j);
                        let v = vertex_idx(i_neighbor, j_neighbor);
                        if u < v {
                            product.add_edge(u, v).ok();
                        }
                    }
                }
            }
        }

        product
    }

    /// Compute the strong product of two graphs
    ///
    /// The strong product G ⊠ H has vertex set V(G) × V(H).
    /// Two vertices (u, v) and (u', v') are adjacent iff:
    /// - u = u' and vv' ∈ E(H), or
    /// - v = v' and uu' ∈ E(G), or
    /// - uu' ∈ E(G) and vv' ∈ E(H)
    pub fn strong_product(&self, other: &Graph) -> Graph {
        let n1 = self.num_vertices;
        let n2 = other.num_vertices;
        let mut product = Graph::new(n1 * n2);

        let vertex_idx = |i: usize, j: usize| i * n2 + j;

        for i in 0..n1 {
            for j in 0..n2 {
                let u = vertex_idx(i, j);

                // Edges from same i coordinate (Cartesian component)
                for &j_neighbor in &other.adj[j] {
                    if j < j_neighbor {
                        let v = vertex_idx(i, j_neighbor);
                        product.add_edge(u, v).ok();
                    }
                }

                // Edges from same j coordinate (Cartesian component)
                for &i_neighbor in &self.adj[i] {
                    if i < i_neighbor {
                        let v = vertex_idx(i_neighbor, j);
                        product.add_edge(u, v).ok();
                    }
                }

                // Diagonal edges (tensor component)
                for &i_neighbor in &self.adj[i] {
                    for &j_neighbor in &other.adj[j] {
                        if i < i_neighbor || (i == i_neighbor && j < j_neighbor) {
                            let v = vertex_idx(i_neighbor, j_neighbor);
                            if u != v {
                                product.add_edge(u, v).ok();
                            }
                        }
                    }
                }
            }
        }

        product
    }

    /// Compute the k-th power of the graph
    ///
    /// In G^k, vertices u and v are adjacent iff their distance in G is at most k.
    pub fn graph_power(&self, k: usize) -> Graph {
        if k == 0 {
            return Graph::new(self.num_vertices);
        }

        let mut power = Graph::new(self.num_vertices);

        // Compute all-pairs shortest distances
        for u in 0..self.num_vertices {
            let distances = self.bfs_distances(u);
            for v in u + 1..self.num_vertices {
                if distances[v] != usize::MAX && distances[v] <= k {
                    power.add_edge(u, v).ok();
                }
            }
        }

        power
    }

    /// Compute the line graph of this graph
    ///
    /// The line graph L(G) has vertices corresponding to edges of G.
    /// Two vertices in L(G) are adjacent iff the corresponding edges in G share a vertex.
    ///
    /// Returns (line_graph, edge_map) where edge_map[i] gives the original edge for vertex i.
    pub fn line_graph(&self) -> (Graph, Vec<(usize, usize)>) {
        let edges = self.edges();
        let num_edges = edges.len();
        let mut line = Graph::new(num_edges);

        // Two edges are adjacent in line graph if they share a vertex
        for i in 0..num_edges {
            for j in i + 1..num_edges {
                let (u1, v1) = edges[i];
                let (u2, v2) = edges[j];

                if u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
                    line.add_edge(i, j).ok();
                }
            }
        }

        (line, edges)
    }

    /// Delete an edge from the graph
    ///
    /// Returns a new graph with the edge (u, v) removed.
    pub fn delete_edge(&self, u: usize, v: usize) -> Graph {
        let mut g = self.clone();
        g.adj[u].remove(&v);
        g.adj[v].remove(&u);
        g
    }

    /// Contract an edge in the graph
    ///
    /// Returns a new graph where vertices u and v are merged into a single vertex.
    /// The merged vertex inherits all edges from both u and v.
    pub fn contract_edge_public(&self, u: usize, v: usize) -> Graph {
        contract_edge(self, u, v)
    }

    /// Delete a vertex from the graph
    ///
    /// Returns a new graph with the vertex removed and all its edges deleted.
    pub fn delete_vertex(&self, vertex: usize) -> Graph {
        if vertex >= self.num_vertices {
            return self.clone();
        }

        let mut g = Graph::new(self.num_vertices - 1);

        // Map old vertices to new vertices
        let get_new_vertex = |old: usize| -> Option<usize> {
            if old == vertex {
                None
            } else if old < vertex {
                Some(old)
            } else {
                Some(old - 1)
            }
        };

        // Add edges
        for i in 0..self.num_vertices {
            if i == vertex {
                continue;
            }
            for &j in &self.adj[i] {
                if j == vertex || j <= i {
                    continue;
                }
                if let (Some(new_i), Some(new_j)) = (get_new_vertex(i), get_new_vertex(j)) {
                    g.add_edge(new_i, new_j).ok();
                }
            }
        }

        g
    }

    /// Check if this graph is a minor of another graph
    ///
    /// A graph H is a minor of G if H can be obtained from G by deleting edges,
    /// deleting vertices, and contracting edges.
    /// Note: This is a simple check, not a full minor testing algorithm.
    pub fn has_minor(&self, minor: &Graph) -> bool {
        // Simple heuristic checks
        if minor.num_vertices > self.num_vertices {
            return false;
        }
        if minor.num_edges() > self.num_edges() {
            return false;
        }

        // For small graphs, could implement more sophisticated checks
        // This is a placeholder for the full minor testing problem (NP-complete)
        true
    }

    /// Compute the girth of the graph
    ///
    /// The girth is the length of the shortest cycle.
    /// Returns None if the graph is acyclic.
    pub fn girth(&self) -> Option<usize> {
        if !self.has_cycle() {
            return None;
        }

        let mut min_cycle = usize::MAX;

        // For each edge, find shortest path that doesn't use that edge
        for u in 0..self.num_vertices {
            for &v in &self.adj[u] {
                if u < v {
                    // Remove edge (u, v) temporarily
                    let mut g = self.clone();
                    g.adj[u].remove(&v);
                    g.adj[v].remove(&u);

                    // Find shortest path from u to v in modified graph
                    if let Ok(Some(path)) = g.shortest_path(u, v) {
                        let cycle_length = path.len(); // path.len() includes both endpoints
                        min_cycle = min_cycle.min(cycle_length);
                    }
                }
            }
        }

        if min_cycle == usize::MAX {
            None
        } else {
            Some(min_cycle)
        }
    }

    /// Compute the radius of the graph
    ///
    /// The radius is the minimum eccentricity over all vertices.
    /// Eccentricity of a vertex is the maximum distance from that vertex to any other.
    /// Returns None if the graph is not connected.
    pub fn radius(&self) -> Option<usize> {
        if !self.is_connected() {
            return None;
        }

        let mut min_eccentricity = usize::MAX;

        for v in 0..self.num_vertices {
            let distances = self.bfs_distances(v);
            let eccentricity = distances.iter()
                .filter(|&&d| d != usize::MAX)
                .max()
                .copied()
                .unwrap_or(0);

            min_eccentricity = min_eccentricity.min(eccentricity);
        }

        Some(min_eccentricity)
    }

    /// Get the eccentricity of a vertex
    ///
    /// Eccentricity is the maximum distance from this vertex to any other vertex.
    /// Returns None if the graph is not connected.
    pub fn eccentricity(&self, v: usize) -> Option<usize> {
        if v >= self.num_vertices || !self.is_connected() {
            return None;
        }

        let distances = self.bfs_distances(v);
        distances.iter()
            .filter(|&&d| d != usize::MAX)
            .max()
            .copied()
    }

    /// Find the center of the graph
    ///
    /// The center is the set of vertices with minimum eccentricity (equal to radius).
    /// Returns None if the graph is not connected.
    pub fn center(&self) -> Option<Vec<usize>> {
        let radius = self.radius()?;
        let mut center_vertices = Vec::new();

        for v in 0..self.num_vertices {
            if self.eccentricity(v) == Some(radius) {
                center_vertices.push(v);
            }
        }

        Some(center_vertices)
    }

    /// Find the maximum clique size (clique number) using a simple algorithm
    ///
    /// Note: This uses a greedy approximation, not guaranteed to be optimal.
    /// Finding the maximum clique is NP-complete.
    pub fn clique_number(&self) -> usize {
        if self.num_vertices == 0 {
            return 0;
        }

        let mut max_clique_size = 1;

        // Try building a clique starting from each vertex
        for start in 0..self.num_vertices {
            let mut clique = vec![start];
            let mut candidates: HashSet<usize> = self.adj[start].clone();

            loop {
                // Find vertex in candidates that is connected to all vertices in current clique
                let mut best_vertex = None;
                let mut max_connections = 0;

                for &v in &candidates {
                    // Check if v is connected to all vertices in the current clique
                    let connected_to_all = clique.iter().all(|&u| self.has_edge(u, v));

                    if connected_to_all {
                        // Count how many other candidates v is connected to
                        let connections = candidates.iter()
                            .filter(|&&w| w != v && self.has_edge(v, w))
                            .count();

                        if connections >= max_connections {
                            max_connections = connections;
                            best_vertex = Some(v);
                        }
                    }
                }

                if let Some(v) = best_vertex {
                    clique.push(v);
                    // Keep only candidates that are neighbors of v (and thus potential clique members)
                    candidates = candidates.into_iter()
                        .filter(|&w| w != v && self.has_edge(v, w))
                        .collect();
                } else {
                    // No more vertices can be added to the clique
                    break;
                }
            }

            max_clique_size = max_clique_size.max(clique.len());
        }

        max_clique_size
    }

    /// Check if the graph is perfect
    ///
    /// A graph is perfect if the chromatic number equals the clique number
    /// for every induced subgraph.
    ///
    /// This implementation uses the Strong Perfect Graph Theorem:
    /// A graph is perfect iff neither it nor its complement contains an odd hole
    /// (induced cycle of odd length ≥ 5) or odd antihole.
    ///
    /// Note: This is a simplified heuristic check that may not be complete.
    /// Complete graphs and bipartite graphs are always perfect.
    pub fn is_perfect(&self) -> bool {
        // Check basic cases
        if self.num_vertices <= 3 {
            return true;
        }

        // Bipartite graphs are perfect
        if self.is_bipartite() {
            return true;
        }

        // Complete graphs are perfect (check if all vertices are connected)
        let expected_edges = (self.num_vertices * (self.num_vertices - 1)) / 2;
        if self.num_edges() == expected_edges {
            return true;
        }

        // Check if chromatic number equals clique number
        // This is a necessary condition but not sufficient
        let chi = self.chromatic_number();
        let omega = self.clique_number();

        if chi != omega {
            return false;
        }

        // For a more complete check, we would need to verify all induced subgraphs
        // This is computationally expensive, so we use heuristics

        // Check for odd holes (odd cycles of length ≥ 5)
        if self.has_odd_hole() {
            return false;
        }

        // Check for odd antiholes in complement
        let complement = self.complement();
        if complement.has_odd_hole() {
            return false;
        }

        true
    }

    /// Get the complement of the graph
    ///
    /// In the complement, vertices are adjacent iff they are not adjacent in the original.
    pub fn complement(&self) -> Graph {
        let mut comp = Graph::new(self.num_vertices);

        for u in 0..self.num_vertices {
            for v in u + 1..self.num_vertices {
                if !self.has_edge(u, v) {
                    comp.add_edge(u, v).ok();
                }
            }
        }

        comp
    }

    /// Check if the graph contains an odd hole (odd cycle of length ≥ 5)
    ///
    /// This is a heuristic check, not guaranteed to find all odd holes.
    fn has_odd_hole(&self) -> bool {
        // Look for odd cycles of length 5, 7, 9, etc.
        for length in (5..=self.num_vertices).step_by(2) {
            if self.has_cycle_of_length(length) {
                return true;
            }
        }
        false
    }

    /// Check if the graph has a cycle of a specific length
    ///
    /// Uses DFS with depth limit to search for cycles.
    fn has_cycle_of_length(&self, target_length: usize) -> bool {
        if target_length < 3 || target_length > self.num_vertices {
            return false;
        }

        for start in 0..self.num_vertices {
            let mut path = vec![start];
            let mut visited = vec![false; self.num_vertices];
            visited[start] = true;

            if self.find_cycle_of_length(start, start, target_length, &mut path, &mut visited) {
                return true;
            }
        }

        false
    }

    fn find_cycle_of_length(
        &self,
        start: usize,
        current: usize,
        remaining: usize,
        path: &mut Vec<usize>,
        visited: &mut Vec<bool>,
    ) -> bool {
        if remaining == 1 {
            // Check if we can close the cycle
            return self.has_edge(current, start);
        }

        for &neighbor in &self.adj[current] {
            if neighbor == start && remaining > 2 {
                continue; // Don't close cycle early
            }
            if !visited[neighbor] {
                path.push(neighbor);
                visited[neighbor] = true;

                if self.find_cycle_of_length(start, neighbor, remaining - 1, path, visited) {
                    return true;
                }

                path.pop();
                visited[neighbor] = false;
            }
        }

        false
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

/// Helper function for matching polynomial computation
fn matching_poly_helper(graph: &Graph) -> Vec<i64> {
    let n = graph.num_vertices();

    // Base case: empty graph
    if n == 0 {
        return vec![1];
    }

    // Base case: graph with no edges
    if graph.num_edges() == 0 {
        // m(G, x) = x^n for n isolated vertices
        let mut poly = vec![0; n + 1];
        poly[n] = 1;
        return poly;
    }

    // Base case: single vertex
    if n == 1 {
        return vec![1]; // m(G, x) = 1
    }

    // Find an edge to use for deletion-contraction
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
    let p_delete = matching_poly_helper(&g_delete);

    // Removal: remove both vertices u and v
    let g_remove = remove_vertices(graph, u, v);
    let p_remove = matching_poly_helper(&g_remove);

    // m(G, x) = m(G - e, x) - m(G - {u, v}, x)
    // The second term needs to be multiplied by x^0 when adding back (no change needed)
    subtract_poly(&p_delete, &p_remove)
}

/// Remove two vertices from a graph
fn remove_vertices(graph: &Graph, u: usize, v: usize) -> Graph {
    let n = graph.num_vertices();
    let mut new_graph = Graph::new(n - 2);

    // Determine which vertex to remove first
    let (first, second) = if u < v { (u, v) } else { (v, u) };

    // Map old vertices to new vertices
    let get_new_vertex = |old: usize| -> Option<usize> {
        if old == first || old == second {
            None
        } else if old < first {
            Some(old)
        } else if old < second {
            Some(old - 1)
        } else {
            Some(old - 2)
        }
    };

    // Add edges
    for i in 0..n {
        if i == first || i == second {
            continue;
        }
        for &j in &graph.adj[i] {
            if j == first || j == second || j <= i {
                continue;
            }
            if let (Some(new_i), Some(new_j)) = (get_new_vertex(i), get_new_vertex(j)) {
                new_graph.adj[new_i].insert(new_j);
                new_graph.adj[new_j].insert(new_i);
            }
        }
    }

    new_graph
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
    fn test_add_vertex() {
        let mut g = Graph::new(3);
        assert_eq!(g.num_vertices(), 3);

        let v3 = g.add_vertex();
        assert_eq!(v3, 3);
        assert_eq!(g.num_vertices(), 4);

        let v4 = g.add_vertex();
        assert_eq!(v4, 4);
        assert_eq!(g.num_vertices(), 5);

        // Can add edges to new vertices
        g.add_edge(0, v3).unwrap();
        g.add_edge(v3, v4).unwrap();
        assert_eq!(g.num_edges(), 2);
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
    fn test_matching_polynomial() {
        // Single edge: m(G, x) = x^2 - 1
        let mut g = Graph::new(2);
        g.add_edge(0, 1).unwrap();
        let poly = g.matching_polynomial();
        // poly[0] = -1 (constant term)
        // poly[2] = 1 (x^2 term)
        assert_eq!(poly[0], -1);
        assert_eq!(poly[2], 1);

        // Triangle K3: m(G, x) = x^3 - 3x
        let mut g2 = Graph::new(3);
        g2.add_edge(0, 1).unwrap();
        g2.add_edge(1, 2).unwrap();
        g2.add_edge(2, 0).unwrap();
        let poly2 = g2.matching_polynomial();
        assert_eq!(poly2[0], 0);   // Constant term
        assert_eq!(poly2[1], -3);  // x term
        assert_eq!(poly2[3], 1);   // x^3 term

        // Empty graph with 2 vertices: m(G, x) = x^2
        let g3 = Graph::new(2);
        let poly3 = g3.matching_polynomial();
        assert_eq!(poly3.len(), 3);
        assert_eq!(poly3[2], 1);
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

    #[test]
    fn test_is_hamiltonian() {
        // Complete graph K4 is Hamiltonian
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }
        assert!(g.is_hamiltonian());

        // Cycle C5 is Hamiltonian
        let mut g2 = Graph::new(5);
        for i in 0..5 {
            g2.add_edge(i, (i + 1) % 5).unwrap();
        }
        assert!(g2.is_hamiltonian());

        // Path graph is not Hamiltonian (no cycle)
        let mut g3 = Graph::new(4);
        g3.add_edge(0, 1).unwrap();
        g3.add_edge(1, 2).unwrap();
        g3.add_edge(2, 3).unwrap();
        assert!(!g3.is_hamiltonian());

        // Star graph is not Hamiltonian
        let mut g4 = Graph::new(5);
        for i in 1..5 {
            g4.add_edge(0, i).unwrap();
        }
        assert!(!g4.is_hamiltonian());
    }

    #[test]
    fn test_is_planar() {
        // K4 is planar
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }
        assert!(g.is_planar());

        // K5 is not planar
        let mut g2 = Graph::new(5);
        for i in 0..5 {
            for j in i + 1..5 {
                g2.add_edge(i, j).unwrap();
            }
        }
        assert!(!g2.is_planar());

        // Tree is planar
        let mut g3 = Graph::new(5);
        g3.add_edge(0, 1).unwrap();
        g3.add_edge(0, 2).unwrap();
        g3.add_edge(1, 3).unwrap();
        g3.add_edge(1, 4).unwrap();
        assert!(g3.is_planar());

        // K3,3 is not planar
        let mut g4 = Graph::new(6);
        for i in 0..3 {
            for j in 3..6 {
                g4.add_edge(i, j).unwrap();
            }
        }
        assert!(!g4.is_planar());
    }

    #[test]
    fn test_perfect_matchings() {
        // Complete graph K4 has 3 perfect matchings
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                g.add_edge(i, j).unwrap();
            }
        }
        let matchings = g.perfect_matchings();
        assert_eq!(matchings.len(), 3);

        // Each matching should have 2 edges (covering 4 vertices)
        for matching in &matchings {
            assert_eq!(matching.len(), 2);
        }

        // Cycle C6 has 2 perfect matchings
        let mut g2 = Graph::new(6);
        for i in 0..6 {
            g2.add_edge(i, (i + 1) % 6).unwrap();
        }
        let matchings2 = g2.perfect_matchings();
        assert_eq!(matchings2.len(), 2);

        // Odd number of vertices has no perfect matching
        let g3 = Graph::new(5);
        assert_eq!(g3.perfect_matchings().len(), 0);

        // Path of length 2 has 1 perfect matching
        let mut g4 = Graph::new(2);
        g4.add_edge(0, 1).unwrap();
        let matchings4 = g4.perfect_matchings();
        assert_eq!(matchings4.len(), 1);
        assert_eq!(matchings4[0], vec![(0, 1)]);
    }

    #[test]
    fn test_cartesian_product() {
        // P2 □ P2 = C4 (path of 2 vertices product with itself gives 4-cycle)
        let mut p2 = Graph::new(2);
        p2.add_edge(0, 1).unwrap();

        let product = p2.cartesian_product(&p2);
        assert_eq!(product.num_vertices(), 4);
        assert_eq!(product.num_edges(), 4);

        // Verify it's a 4-cycle by checking degrees
        for v in 0..4 {
            assert_eq!(product.degree(v), Some(2));
        }

        // K2 □ K1 = K2 (edge product with single vertex)
        let k1 = Graph::new(1);
        let product2 = p2.cartesian_product(&k1);
        assert_eq!(product2.num_vertices(), 2);
        assert_eq!(product2.num_edges(), 1);
    }

    #[test]
    fn test_tensor_product() {
        // K2 × K2 = two disjoint edges
        let mut k2 = Graph::new(2);
        k2.add_edge(0, 1).unwrap();

        let product = k2.tensor_product(&k2);
        assert_eq!(product.num_vertices(), 4);
        // K2 × K2 creates edges (0,0)-(1,1) and (0,1)-(1,0)
        assert_eq!(product.num_edges(), 2);

        // K3 × K2
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();

        let product2 = k3.tensor_product(&k2);
        assert_eq!(product2.num_vertices(), 6);
        // K3 × K2: 3 edges × 1 edge × 2 (both directions) = 6 edges
        assert_eq!(product2.num_edges(), 6);
    }

    #[test]
    fn test_strong_product() {
        // P2 ⊠ P2
        let mut p2 = Graph::new(2);
        p2.add_edge(0, 1).unwrap();

        let product = p2.strong_product(&p2);
        assert_eq!(product.num_vertices(), 4);
        // Strong product includes Cartesian (4 edges) + tensor (0 edges) = 4 edges
        // But actually for P2⊠P2 we get more due to diagonals
        assert!(product.num_edges() >= 4);

        // Verify connectivity
        assert!(product.is_connected());
    }

    #[test]
    fn test_graph_power() {
        // Path P4: 0-1-2-3
        let mut p4 = Graph::new(4);
        p4.add_edge(0, 1).unwrap();
        p4.add_edge(1, 2).unwrap();
        p4.add_edge(2, 3).unwrap();

        // P4^0 should have no edges
        let p0 = p4.graph_power(0);
        assert_eq!(p0.num_edges(), 0);

        // P4^1 = P4 (same as original)
        let p1 = p4.graph_power(1);
        assert_eq!(p1.num_edges(), 3);

        // P4^2 should connect vertices at distance ≤ 2
        let p2 = p4.graph_power(2);
        assert!(p2.has_edge(0, 1));
        assert!(p2.has_edge(0, 2));
        assert!(p2.has_edge(1, 3));
        assert!(p2.has_edge(2, 3));

        // P4^3 should be complete (all distances ≤ 3)
        let p3 = p4.graph_power(3);
        assert_eq!(p3.num_edges(), 6); // K4 has 6 edges
    }

    #[test]
    fn test_line_graph() {
        // Triangle K3
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();

        let (line, edge_map) = k3.line_graph();
        assert_eq!(line.num_vertices(), 3); // 3 edges in K3
        assert_eq!(edge_map.len(), 3);

        // Line graph of K3 is also K3 (each edge shares a vertex with every other edge)
        assert_eq!(line.num_edges(), 3);

        // Path P3
        let mut p3 = Graph::new(3);
        p3.add_edge(0, 1).unwrap();
        p3.add_edge(1, 2).unwrap();

        let (line2, edge_map2) = p3.line_graph();
        assert_eq!(line2.num_vertices(), 2); // 2 edges in P3
        assert_eq!(line2.num_edges(), 1); // The two edges share vertex 1
        assert_eq!(edge_map2, vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn test_delete_edge() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 0).unwrap();

        let g2 = g.delete_edge(0, 1);
        assert_eq!(g2.num_vertices(), 3);
        assert_eq!(g2.num_edges(), 2);
        assert!(!g2.has_edge(0, 1));
        assert!(g2.has_edge(1, 2));
        assert!(g2.has_edge(2, 0));
    }

    #[test]
    fn test_contract_edge() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();

        let g2 = g.contract_edge_public(1, 2);
        assert_eq!(g2.num_vertices(), 3);
        // After contracting 1-2, we have vertices {0, merged(1,2), 3}
        // The merged vertex should be connected to both 0 and 3
        assert!(g2.is_connected());
    }

    #[test]
    fn test_delete_vertex() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        g.add_edge(2, 3).unwrap();
        g.add_edge(3, 0).unwrap();

        let g2 = g.delete_vertex(1);
        assert_eq!(g2.num_vertices(), 3);
        // After removing vertex 1, edges 0-1 and 1-2 are gone
        assert_eq!(g2.num_edges(), 2); // Only 2-3 and 3-0 remain (remapped)
    }

    #[test]
    fn test_girth() {
        // Tree has no cycles
        let mut tree = Graph::new(4);
        tree.add_edge(0, 1).unwrap();
        tree.add_edge(0, 2).unwrap();
        tree.add_edge(0, 3).unwrap();
        assert_eq!(tree.girth(), None);

        // Triangle has girth 3
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();
        assert_eq!(k3.girth(), Some(3));

        // Square has girth 4
        let mut c4 = Graph::new(4);
        c4.add_edge(0, 1).unwrap();
        c4.add_edge(1, 2).unwrap();
        c4.add_edge(2, 3).unwrap();
        c4.add_edge(3, 0).unwrap();
        assert_eq!(c4.girth(), Some(4));

        // Pentagon has girth 5
        let mut c5 = Graph::new(5);
        for i in 0..5 {
            c5.add_edge(i, (i + 1) % 5).unwrap();
        }
        assert_eq!(c5.girth(), Some(5));
    }

    #[test]
    fn test_radius() {
        // Path P4: 0-1-2-3
        let mut p4 = Graph::new(4);
        p4.add_edge(0, 1).unwrap();
        p4.add_edge(1, 2).unwrap();
        p4.add_edge(2, 3).unwrap();

        // Radius is 1 (center vertices 1 or 2 have eccentricity 2)
        // Wait, let me recalculate:
        // Eccentricity of 0: max(0→1=1, 0→2=2, 0→3=3) = 3
        // Eccentricity of 1: max(1→0=1, 1→2=1, 1→3=2) = 2
        // Eccentricity of 2: max(2→0=2, 2→1=1, 2→3=1) = 2
        // Eccentricity of 3: max(3→0=3, 3→1=2, 3→2=1) = 3
        // Radius = min(3, 2, 2, 3) = 2
        assert_eq!(p4.radius(), Some(2));

        // Star graph: center has eccentricity 1, leaves have eccentricity 2
        let mut star = Graph::new(5);
        for i in 1..5 {
            star.add_edge(0, i).unwrap();
        }
        assert_eq!(star.radius(), Some(1));

        // Complete graph: all vertices have eccentricity 1
        let mut k4 = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                k4.add_edge(i, j).unwrap();
            }
        }
        assert_eq!(k4.radius(), Some(1));

        // Disconnected graph has no radius
        let mut disconnected = Graph::new(4);
        disconnected.add_edge(0, 1).unwrap();
        disconnected.add_edge(2, 3).unwrap();
        assert_eq!(disconnected.radius(), None);
    }

    #[test]
    fn test_eccentricity() {
        let mut p4 = Graph::new(4);
        p4.add_edge(0, 1).unwrap();
        p4.add_edge(1, 2).unwrap();
        p4.add_edge(2, 3).unwrap();

        assert_eq!(p4.eccentricity(0), Some(3));
        assert_eq!(p4.eccentricity(1), Some(2));
        assert_eq!(p4.eccentricity(2), Some(2));
        assert_eq!(p4.eccentricity(3), Some(3));
    }

    #[test]
    fn test_center() {
        // Path P4: center should be vertices 1 and 2
        let mut p4 = Graph::new(4);
        p4.add_edge(0, 1).unwrap();
        p4.add_edge(1, 2).unwrap();
        p4.add_edge(2, 3).unwrap();

        let center = p4.center().unwrap();
        assert_eq!(center.len(), 2);
        assert!(center.contains(&1));
        assert!(center.contains(&2));

        // Star: center is the hub vertex
        let mut star = Graph::new(5);
        for i in 1..5 {
            star.add_edge(0, i).unwrap();
        }
        assert_eq!(star.center(), Some(vec![0]));
    }

    #[test]
    fn test_clique_number() {
        // Complete graph K4 has clique number 4
        let mut k4 = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                k4.add_edge(i, j).unwrap();
            }
        }
        assert_eq!(k4.clique_number(), 4);

        // Triangle has clique number 3
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();
        assert_eq!(k3.clique_number(), 3);

        // Path has clique number 2 (just edges)
        let mut p3 = Graph::new(3);
        p3.add_edge(0, 1).unwrap();
        p3.add_edge(1, 2).unwrap();
        assert_eq!(p3.clique_number(), 2);

        // Empty graph has clique number 1
        let empty = Graph::new(3);
        assert_eq!(empty.clique_number(), 1);
    }

    #[test]
    fn test_complement() {
        // Complement of K3 is empty graph
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();

        let comp = k3.complement();
        assert_eq!(comp.num_edges(), 0);

        // Complement of empty graph is complete
        let empty = Graph::new(3);
        let comp2 = empty.complement();
        assert_eq!(comp2.num_edges(), 3); // K3 has 3 edges

        // Complement of path P3
        let mut p3 = Graph::new(3);
        p3.add_edge(0, 1).unwrap();
        p3.add_edge(1, 2).unwrap();

        let comp3 = p3.complement();
        assert_eq!(comp3.num_edges(), 1); // Only edge 0-2
        assert!(comp3.has_edge(0, 2));
    }

    #[test]
    fn test_is_perfect() {
        // Bipartite graphs are perfect
        let mut k22 = Graph::new(4);
        k22.add_edge(0, 2).unwrap();
        k22.add_edge(0, 3).unwrap();
        k22.add_edge(1, 2).unwrap();
        k22.add_edge(1, 3).unwrap();
        assert!(k22.is_perfect());

        // Complete graphs are perfect
        let mut k4 = Graph::new(4);
        for i in 0..4 {
            for j in i + 1..4 {
                k4.add_edge(i, j).unwrap();
            }
        }
        assert!(k4.is_perfect());

        // Small graphs are perfect
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();
        assert!(k3.is_perfect());
    }

    #[test]
    fn test_has_cycle_of_length() {
        // Triangle
        let mut k3 = Graph::new(3);
        k3.add_edge(0, 1).unwrap();
        k3.add_edge(1, 2).unwrap();
        k3.add_edge(2, 0).unwrap();

        assert!(k3.has_cycle_of_length(3));
        assert!(!k3.has_cycle_of_length(4));
        assert!(!k3.has_cycle_of_length(5));

        // 4-cycle
        let mut c4 = Graph::new(4);
        c4.add_edge(0, 1).unwrap();
        c4.add_edge(1, 2).unwrap();
        c4.add_edge(2, 3).unwrap();
        c4.add_edge(3, 0).unwrap();

        assert!(c4.has_cycle_of_length(4));
        assert!(!c4.has_cycle_of_length(3));

        // 5-cycle
        let mut c5 = Graph::new(5);
        for i in 0..5 {
            c5.add_edge(i, (i + 1) % 5).unwrap();
        }
        assert!(c5.has_cycle_of_length(5));
        assert!(!c5.has_cycle_of_length(3));
        assert!(!c5.has_cycle_of_length(4));
    }
}

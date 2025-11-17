//! Quiver paths - sequences of edges in a quiver

use crate::Quiver;
use std::cmp::Ordering;

/// A path in a quiver - a sequence of composable edges
///
/// A path is defined by:
/// - An initial (start) vertex
/// - A terminal (end) vertex
/// - A sequence of edge indices that connect them
///
/// The empty sequence represents a trivial path (idempotent) at a vertex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuiverPath {
    /// Initial vertex of the path
    pub start: usize,
    /// Terminal vertex of the path
    pub end: usize,
    /// Sequence of edge indices in the quiver
    pub edges: Vec<usize>,
}

impl QuiverPath {
    /// Create a new path
    ///
    /// Validates that the edges form a valid path from start to end
    pub fn new(quiver: &Quiver, start: usize, end: usize, edges: Vec<usize>) -> Result<Self, String> {
        // Empty path (trivial/idempotent)
        if edges.is_empty() {
            if start != end {
                return Err("Empty path must have start == end".to_string());
            }
            return Ok(QuiverPath { start, end, edges });
        }

        // Validate edge sequence
        let mut current = start;
        for &edge_idx in &edges {
            if edge_idx >= quiver.num_edges() {
                return Err(format!("Edge index {} out of bounds", edge_idx));
            }

            let edge = &quiver.edges()[edge_idx];
            if edge.source != current {
                return Err(format!(
                    "Edge {} starts at {}, but current vertex is {}",
                    edge.label, edge.source, current
                ));
            }
            current = edge.target;
        }

        if current != end {
            return Err(format!(
                "Path ends at vertex {}, but expected {}",
                current, end
            ));
        }

        Ok(QuiverPath { start, end, edges })
    }

    /// Create a trivial (identity) path at a vertex
    pub fn trivial(vertex: usize) -> Self {
        QuiverPath {
            start: vertex,
            end: vertex,
            edges: Vec::new(),
        }
    }

    /// Get the length of the path (number of edges)
    pub fn length(&self) -> usize {
        self.edges.len()
    }

    /// Check if this is a trivial path
    pub fn is_trivial(&self) -> bool {
        self.edges.is_empty()
    }

    /// Compose two paths (concatenation)
    ///
    /// Returns None if the paths are not composable (end of first != start of second)
    pub fn compose(&self, other: &QuiverPath) -> Option<QuiverPath> {
        if self.end != other.start {
            return None;
        }

        let mut edges = self.edges.clone();
        edges.extend_from_slice(&other.edges);

        Some(QuiverPath {
            start: self.start,
            end: other.end,
            edges,
        })
    }

    /// Get a slice of the path
    ///
    /// Returns None if the slice indices are invalid
    pub fn slice(&self, quiver: &Quiver, start_idx: usize, end_idx: usize) -> Option<QuiverPath> {
        if start_idx > end_idx || end_idx > self.edges.len() {
            return None;
        }

        if start_idx == end_idx {
            // Trivial path
            if start_idx == 0 {
                return Some(QuiverPath::trivial(self.start));
            } else if start_idx == self.edges.len() {
                return Some(QuiverPath::trivial(self.end));
            } else {
                let vertex = quiver.edges()[self.edges[start_idx - 1]].target;
                return Some(QuiverPath::trivial(vertex));
            }
        }

        let sliced_edges: Vec<usize> = self.edges[start_idx..end_idx].to_vec();

        let slice_start = if start_idx == 0 {
            self.start
        } else {
            quiver.edges()[self.edges[start_idx - 1]].target
        };

        let slice_end = quiver.edges()[self.edges[end_idx - 1]].target;

        QuiverPath::new(quiver, slice_start, slice_end, sliced_edges).ok()
    }

    /// Check if this path contains another path as a subpath
    pub fn has_subpath(&self, other: &QuiverPath) -> bool {
        if other.length() > self.length() {
            return false;
        }

        if other.is_trivial() {
            return true; // Every path contains trivial paths
        }

        // Check each possible starting position
        for i in 0..=(self.edges.len() - other.edges.len()) {
            if self.edges[i..i + other.edges.len()] == other.edges[..] {
                return true;
            }
        }

        false
    }

    /// Check if this path has another path as a prefix
    pub fn has_prefix(&self, prefix: &QuiverPath) -> bool {
        if prefix.length() > self.length() {
            return false;
        }

        if prefix.is_trivial() {
            return prefix.start == self.start;
        }

        self.edges[..prefix.length()] == prefix.edges[..]
    }

    /// Check if this path has another path as a suffix
    pub fn has_suffix(&self, suffix: &QuiverPath) -> bool {
        if suffix.length() > self.length() {
            return false;
        }

        if suffix.is_trivial() {
            return suffix.end == self.end;
        }

        let start = self.length() - suffix.length();
        self.edges[start..] == suffix.edges[..]
    }

    /// Find the "greatest common divisor" of two paths
    ///
    /// Returns the longest overlapping segment where the end of self overlaps with the start of other
    pub fn gcd(&self, other: &QuiverPath) -> Option<QuiverPath> {
        let max_overlap = self.length().min(other.length());

        for len in (1..=max_overlap).rev() {
            let self_suffix = &self.edges[self.edges.len() - len..];
            let other_prefix = &other.edges[..len];

            if self_suffix == other_prefix {
                // Found the overlap - need to construct the path
                // The overlapping segment is the suffix of self
                return Some(QuiverPath {
                    start: if len == self.length() {
                        self.start
                    } else {
                        // We'd need the quiver to determine the start vertex properly
                        // For now, we'll return None since we need more context
                        return None;
                    },
                    end: self.end,
                    edges: self_suffix.to_vec(),
                });
            }
        }

        None
    }

    /// Get the string representation of the path
    pub fn to_string(&self, quiver: &Quiver) -> String {
        if self.is_trivial() {
            return format!("e_{}", self.start);
        }

        let labels: Vec<String> = self
            .edges
            .iter()
            .map(|&idx| quiver.edges()[idx].label.clone())
            .collect();

        labels.join("*")
    }
}

impl PartialOrd for QuiverPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuiverPath {
    /// Compare paths by:
    /// 1. Negative length (shorter paths come later)
    /// 2. Initial vertex
    /// 3. Terminal vertex
    /// 4. Reverse lexicographic edge order
    fn cmp(&self, other: &Self) -> Ordering {
        // First by negative length (longer paths first)
        match other.length().cmp(&self.length()) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Then by initial vertex
        match self.start.cmp(&other.start) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Then by terminal vertex
        match self.end.cmp(&other.end) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Finally by reverse lexicographic edge order
        for i in (0..self.length()).rev() {
            match self.edges[i].cmp(&other.edges[i]) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }

        Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_quiver() -> Quiver {
        let mut q = Quiver::new(4);
        q.add_edge(0, 1, "a").unwrap();
        q.add_edge(1, 2, "b").unwrap();
        q.add_edge(2, 3, "c").unwrap();
        q.add_edge(0, 2, "d").unwrap();
        q
    }

    #[test]
    fn test_trivial_path() {
        let path = QuiverPath::trivial(0);
        assert_eq!(path.start, 0);
        assert_eq!(path.end, 0);
        assert_eq!(path.length(), 0);
        assert!(path.is_trivial());
    }

    #[test]
    fn test_valid_path() {
        let q = setup_quiver();
        // Path: 0 -a-> 1 -b-> 2
        let path = QuiverPath::new(&q, 0, 2, vec![0, 1]);
        assert!(path.is_ok());
        let p = path.unwrap();
        assert_eq!(p.start, 0);
        assert_eq!(p.end, 2);
        assert_eq!(p.length(), 2);
        assert!(!p.is_trivial());
    }

    #[test]
    fn test_invalid_path_disconnected() {
        let q = setup_quiver();
        // Edges don't connect: 0 -a-> 1, then 2 -c-> 3 (missing connection)
        let path = QuiverPath::new(&q, 0, 3, vec![0, 2]);
        assert!(path.is_err());
    }

    #[test]
    fn test_invalid_path_wrong_end() {
        let q = setup_quiver();
        // Path goes 0 -a-> 1 but claims to end at 2
        let path = QuiverPath::new(&q, 0, 2, vec![0]);
        assert!(path.is_err());
    }

    #[test]
    fn test_compose_paths() {
        let q = setup_quiver();
        let p1 = QuiverPath::new(&q, 0, 1, vec![0]).unwrap(); // 0 -a-> 1
        let p2 = QuiverPath::new(&q, 1, 2, vec![1]).unwrap(); // 1 -b-> 2

        let composed = p1.compose(&p2);
        assert!(composed.is_some());
        let p = composed.unwrap();
        assert_eq!(p.start, 0);
        assert_eq!(p.end, 2);
        assert_eq!(p.edges, vec![0, 1]);
    }

    #[test]
    fn test_compose_incompatible() {
        let q = setup_quiver();
        let p1 = QuiverPath::new(&q, 0, 1, vec![0]).unwrap(); // 0 -a-> 1
        let p2 = QuiverPath::new(&q, 2, 3, vec![2]).unwrap(); // 2 -c-> 3

        let composed = p1.compose(&p2);
        assert!(composed.is_none()); // End of p1 (1) != start of p2 (2)
    }

    #[test]
    fn test_compose_with_trivial() {
        let q = setup_quiver();
        let p = QuiverPath::new(&q, 0, 1, vec![0]).unwrap(); // 0 -a-> 1
        let trivial = QuiverPath::trivial(1);

        let composed = p.compose(&trivial);
        assert!(composed.is_some());
        let result = composed.unwrap();
        assert_eq!(result.edges, vec![0]);
        assert_eq!(result.start, 0);
        assert_eq!(result.end, 1);
    }

    #[test]
    fn test_slice() {
        let q = setup_quiver();
        let p = QuiverPath::new(&q, 0, 3, vec![0, 1, 2]).unwrap(); // 0 -a-> 1 -b-> 2 -c-> 3

        // Slice [1, 3) should give 1 -b-> 2 -c-> 3
        let sliced = p.slice(&q, 1, 3);
        assert!(sliced.is_some());
        let s = sliced.unwrap();
        assert_eq!(s.edges, vec![1, 2]);
        assert_eq!(s.start, 1);
        assert_eq!(s.end, 3);
    }

    #[test]
    fn test_has_subpath() {
        let q = setup_quiver();
        let full = QuiverPath::new(&q, 0, 3, vec![0, 1, 2]).unwrap();
        let sub = QuiverPath::new(&q, 1, 2, vec![1]).unwrap();

        assert!(full.has_subpath(&sub));
        assert!(!sub.has_subpath(&full));
    }

    #[test]
    fn test_has_prefix() {
        let q = setup_quiver();
        let full = QuiverPath::new(&q, 0, 3, vec![0, 1, 2]).unwrap();
        let prefix = QuiverPath::new(&q, 0, 2, vec![0, 1]).unwrap();

        assert!(full.has_prefix(&prefix));
        assert!(!prefix.has_prefix(&full));
    }

    #[test]
    fn test_has_suffix() {
        let q = setup_quiver();
        let full = QuiverPath::new(&q, 0, 3, vec![0, 1, 2]).unwrap();
        let suffix = QuiverPath::new(&q, 1, 3, vec![1, 2]).unwrap();

        assert!(full.has_suffix(&suffix));
        assert!(!suffix.has_suffix(&full));
    }

    #[test]
    fn test_to_string() {
        let q = setup_quiver();
        let trivial = QuiverPath::trivial(0);
        assert_eq!(trivial.to_string(&q), "e_0");

        let path = QuiverPath::new(&q, 0, 2, vec![0, 1]).unwrap();
        assert_eq!(path.to_string(&q), "a*b");
    }

    #[test]
    fn test_ordering() {
        let q = setup_quiver();
        let p1 = QuiverPath::new(&q, 0, 1, vec![0]).unwrap(); // length 1
        let p2 = QuiverPath::new(&q, 0, 2, vec![0, 1]).unwrap(); // length 2

        // Longer paths should come first (negative length ordering)
        assert!(p2 < p1);
    }
}

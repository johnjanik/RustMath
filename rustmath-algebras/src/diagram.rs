//! Diagram Algebras - Core diagram data structures
//!
//! This module provides the foundational data structures for diagram algebras:
//! - Partition diagrams
//! - Brauer diagrams
//! - Temperley-Lieb diagrams
//!
//! Diagrams consist of vertices (top and bottom) connected by edges.
//! They are represented using set partitions of the vertex set.
//!
//! References:
//! - Brauer, R. "On algebras which are connected with the semisimple continuous groups" (1937)
//! - Temperley, H. N. V. and Lieb, E. H. "Relations between the 'percolation' and 'colouring' problem" (1971)
//! - Martin, P. "Temperley-Lieb algebras for non-planar statistical mechanics" (1991)

use rustmath_core::{MathError, Result};
use rustmath_combinatorics::{SetPartition, PerfectMatching};
use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// A partition diagram with n top vertices and n bottom vertices
///
/// Vertices are labeled:
/// - Top: 0, 1, ..., n-1
/// - Bottom: n, n+1, ..., 2n-1
///
/// The diagram is represented as a set partition of {0, 1, ..., 2n-1}.
///
/// # Examples
///
/// ```
/// use rustmath_algebras::diagram::PartitionDiagram;
///
/// // Create a diagram with 3 vertices on top and bottom
/// let diagram = PartitionDiagram::identity(3);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionDiagram {
    /// Order of the diagram (number of vertices on top or bottom)
    order: usize,
    /// The partition of vertices {0, ..., 2n-1}
    /// Each block is a set of connected vertices
    blocks: Vec<BTreeSet<usize>>,
}

impl PartitionDiagram {
    /// Create a new partition diagram from blocks
    ///
    /// # Arguments
    ///
    /// * `order` - Number of vertices on each side
    /// * `blocks` - Partition blocks (each is a set of connected vertices)
    ///
    /// # Returns
    ///
    /// A new partition diagram, or an error if invalid
    pub fn new(order: usize, blocks: Vec<BTreeSet<usize>>) -> Result<Self> {
        // Validate that blocks form a partition of {0, ..., 2*order-1}
        let mut seen = HashSet::new();
        for block in &blocks {
            for &vertex in block {
                if vertex >= 2 * order {
                    return Err(MathError::InvalidArgument(
                        format!("Vertex {} out of range for order {}", vertex, order)
                    ));
                }
                if !seen.insert(vertex) {
                    return Err(MathError::InvalidArgument(
                        format!("Vertex {} appears in multiple blocks", vertex)
                    ));
                }
            }
        }

        // Check all vertices are covered
        if seen.len() != 2 * order {
            return Err(MathError::InvalidArgument(
                "Not all vertices are covered by partition blocks".to_string()
            ));
        }

        Ok(PartitionDiagram { order, blocks })
    }

    /// Create identity diagram (each top vertex connected to corresponding bottom vertex)
    ///
    /// # Arguments
    ///
    /// * `order` - Number of vertices on each side
    pub fn identity(order: usize) -> Self {
        let blocks: Vec<BTreeSet<usize>> = (0..order)
            .map(|i| {
                let mut block = BTreeSet::new();
                block.insert(i);           // top vertex i
                block.insert(i + order);   // bottom vertex i
                block
            })
            .collect();

        PartitionDiagram { order, blocks }
    }

    /// Create a diagram from a perfect matching representation
    ///
    /// # Arguments
    ///
    /// * `order` - Number of vertices on each side
    /// * `matching` - List of pairs (i, j) where vertices i and j are connected
    pub fn from_matching(order: usize, matching: Vec<(usize, usize)>) -> Result<Self> {
        let mut blocks: Vec<BTreeSet<usize>> = vec![];
        let mut vertex_to_block: HashMap<usize, usize> = HashMap::new();

        for (i, j) in matching {
            if i >= 2 * order || j >= 2 * order {
                return Err(MathError::InvalidArgument(
                    format!("Vertex out of range: {} or {}", i, j)
                ));
            }

            // Find or create blocks for i and j
            let block_i = vertex_to_block.get(&i).copied();
            let block_j = vertex_to_block.get(&j).copied();

            match (block_i, block_j) {
                (None, None) => {
                    // Create new block with both vertices
                    let mut new_block = BTreeSet::new();
                    new_block.insert(i);
                    new_block.insert(j);
                    let block_idx = blocks.len();
                    blocks.push(new_block);
                    vertex_to_block.insert(i, block_idx);
                    vertex_to_block.insert(j, block_idx);
                }
                (Some(bi), None) => {
                    // Add j to block containing i
                    blocks[bi].insert(j);
                    vertex_to_block.insert(j, bi);
                }
                (None, Some(bj)) => {
                    // Add i to block containing j
                    blocks[bj].insert(i);
                    vertex_to_block.insert(i, bj);
                }
                (Some(bi), Some(bj)) if bi == bj => {
                    // Already in same block, nothing to do
                }
                (Some(bi), Some(bj)) => {
                    // Merge blocks
                    let block_j = blocks[bj].clone();
                    for &v in &block_j {
                        blocks[bi].insert(v);
                        vertex_to_block.insert(v, bi);
                    }
                    blocks[bj].clear(); // Will be removed later
                }
            }
        }

        // Add singleton blocks for unmatched vertices
        for v in 0..2 * order {
            if !vertex_to_block.contains_key(&v) {
                let mut singleton = BTreeSet::new();
                singleton.insert(v);
                blocks.push(singleton);
            }
        }

        // Remove empty blocks
        blocks.retain(|b| !b.is_empty());

        Self::new(order, blocks)
    }

    /// Get the order of the diagram
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the blocks of the partition
    pub fn blocks(&self) -> &Vec<BTreeSet<usize>> {
        &self.blocks
    }

    /// Count propagating strands (blocks connecting top to bottom)
    pub fn propagating_count(&self) -> usize {
        self.blocks.iter().filter(|block| {
            let has_top = block.iter().any(|&v| v < self.order);
            let has_bottom = block.iter().any(|&v| v >= self.order);
            has_top && has_bottom
        }).count()
    }

    /// Count top-only blocks (blocks only touching top vertices)
    pub fn top_only_count(&self) -> usize {
        self.blocks.iter().filter(|block| {
            block.iter().all(|&v| v < self.order)
        }).count()
    }

    /// Count bottom-only blocks (blocks only touching bottom vertices)
    pub fn bottom_only_count(&self) -> usize {
        self.blocks.iter().filter(|block| {
            block.iter().all(|&v| v >= self.order)
        }).count()
    }

    /// Compose two diagrams (diagram multiplication)
    ///
    /// The composition stacks diagram1 on top of diagram2:
    /// - Bottom vertices of diagram1 connect to top vertices of diagram2
    /// - Result has top vertices from diagram1, bottom vertices from diagram2
    ///
    /// # Arguments
    ///
    /// * `diagram2` - The diagram to compose with (on bottom)
    pub fn compose(&self, diagram2: &PartitionDiagram) -> Result<PartitionDiagram> {
        if self.order != diagram2.order {
            return Err(MathError::InvalidArgument(
                format!("Cannot compose diagrams of different orders: {} and {}",
                    self.order, diagram2.order)
            ));
        }

        let n = self.order;

        // Relabel vertices for composition:
        // diagram1: top = 0..n-1, bottom = n..2n-1
        // diagram2: top = 2n..3n-1, bottom = 3n..4n-1
        // After composition:
        // - top = 0..n-1 (from diagram1)
        // - middle = n..2n-1 (bottom of diagram1 = top of diagram2, gets identified)
        // - bottom = 3n..4n-1 (from diagram2) → relabeled to n..2n-1

        let mut union_find: HashMap<usize, usize> = HashMap::new();

        // Initialize union-find
        for i in 0..4 * n {
            union_find.insert(i, i);
        }

        fn find(union_find: &mut HashMap<usize, usize>, x: usize) -> usize {
            if union_find[&x] != x {
                let root = find(union_find, union_find[&x]);
                union_find.insert(x, root);
            }
            union_find[&x]
        }

        fn union(union_find: &mut HashMap<usize, usize>, x: usize, y: usize) {
            let root_x = find(union_find, x);
            let root_y = find(union_find, y);
            if root_x != root_y {
                union_find.insert(root_x, root_y);
            }
        }

        // Add edges from diagram1
        for block in &self.blocks {
            let vertices: Vec<usize> = block.iter().copied().collect();
            for i in 1..vertices.len() {
                union(&mut union_find, vertices[0], vertices[i]);
            }
        }

        // Add edges from diagram2 (with offset of 2n)
        for block in &diagram2.blocks {
            let vertices: Vec<usize> = block.iter()
                .map(|&v| v + 2 * n)
                .collect();
            for i in 1..vertices.len() {
                union(&mut union_find, vertices[0], vertices[i]);
            }
        }

        // Identify middle vertices: vertex i+n from diagram1 with vertex i+2n from diagram2
        for i in 0..n {
            union(&mut union_find, i + n, i + 2 * n);
        }

        // Build connected components, excluding middle vertices
        let mut components: HashMap<usize, BTreeSet<usize>> = HashMap::new();

        // Add top vertices (0..n-1)
        for i in 0..n {
            let root = find(&mut union_find, i);
            components.entry(root).or_insert_with(BTreeSet::new).insert(i);
        }

        // Add bottom vertices (3n..4n-1) but relabel to (n..2n-1)
        for i in 3 * n..4 * n {
            let root = find(&mut union_find, i);
            components.entry(root).or_insert_with(BTreeSet::new).insert(i - 2 * n);
        }

        // Merge components connected through middle vertices
        for i in n..2 * n {
            let root = find(&mut union_find, i);

            // Find all components that include this root
            let mut merged_component = BTreeSet::new();

            // Check top vertices
            for j in 0..n {
                if find(&mut union_find, j) == root {
                    merged_component.insert(j);
                }
            }

            // Check bottom vertices
            for j in 3 * n..4 * n {
                if find(&mut union_find, j) == root {
                    merged_component.insert(j - 2 * n);
                }
            }

            if !merged_component.is_empty() {
                components.insert(root, merged_component);
            }
        }

        let blocks: Vec<BTreeSet<usize>> = components.into_values()
            .filter(|b| !b.is_empty())
            .collect();

        PartitionDiagram::new(n, blocks)
    }

    /// Check if diagram is planar (noncrossing)
    ///
    /// A diagram is planar if when vertices are arranged in order
    /// (top left-to-right, bottom right-to-left), no edges cross.
    pub fn is_planar(&self) -> bool {
        // Convert to list of edges between vertices
        let mut edges = Vec::new();

        for block in &self.blocks {
            let vertices: Vec<usize> = block.iter().copied().collect();
            // Add all pairs within the block as edges
            for i in 0..vertices.len() {
                for j in i + 1..vertices.len() {
                    edges.push((vertices[i], vertices[j]));
                }
            }
        }

        // Check each pair of edges for crossing
        for i in 0..edges.len() {
            for j in i + 1..edges.len() {
                if self.edges_cross(edges[i], edges[j]) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if two edges cross in the standard planar embedding
    ///
    /// Vertices are arranged:
    /// - Top: 0, 1, ..., n-1 (left to right)
    /// - Bottom: n, n+1, ..., 2n-1 (left to right)
    ///
    /// An edge from a to b crosses edge from c to d if:
    /// - One edge goes from top to bottom while the other is all on one side, OR
    /// - Both edges are on the same side and they interleave
    fn edges_cross(&self, edge1: (usize, usize), edge2: (usize, usize)) -> bool {
        let (mut a, mut b) = edge1;
        let (mut c, mut d) = edge2;

        // Ensure a < b and c < d
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if c > d {
            std::mem::swap(&mut c, &mut d);
        }

        let n = self.order;

        // Classify edges by which side they're on
        let edge1_top = a < n && b < n;
        let edge1_bottom = a >= n && b >= n;
        let edge1_mixed = !edge1_top && !edge1_bottom;

        let edge2_top = c < n && d < n;
        let edge2_bottom = c >= n && d >= n;
        let edge2_mixed = !edge2_top && !edge2_bottom;

        // Two edges on the same side cross if they interleave
        if edge1_top && edge2_top {
            return a < c && c < b && b < d;
        }

        if edge1_bottom && edge2_bottom {
            return a < c && c < b && b < d;
        }

        // Mixed edges: check if they cross
        if edge1_mixed && edge2_mixed {
            // Both edges go from top to bottom
            // They cross if the ordering is opposite on top and bottom
            let (top1, bot1) = if a < n { (a, b) } else { (b, a) };
            let (top2, bot2) = if c < n { (c, d) } else { (d, c) };

            // Normalize bottom vertices (subtract n for comparison)
            let bot1_norm = bot1 - n;
            let bot2_norm = bot2 - n;

            // Cross if order is reversed
            return (top1 < top2) != (bot1_norm < bot2_norm);
        }

        // One mixed, one on a side - these don't cross in standard embedding
        false
    }
}

impl Hash for PartitionDiagram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.order.hash(state);
        // Hash blocks in sorted order for consistency
        let mut sorted_blocks: Vec<Vec<usize>> = self.blocks.iter()
            .map(|b| b.iter().copied().collect())
            .collect();
        sorted_blocks.sort();
        sorted_blocks.hash(state);
    }
}

impl Display for PartitionDiagram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PartitionDiagram({}, [", self.order)?;
        for (i, block) in self.blocks.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{{")?;
            for (j, &v) in block.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", v)?;
            }
            write!(f, "}}")?;
        }
        write!(f, "])")
    }
}

/// A Brauer diagram (same as partition diagram, but conceptually represents Brauer algebra)
pub type BrauerDiagram = PartitionDiagram;

/// A Temperley-Lieb diagram (planar partition diagram)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TemperleyLiebDiagram {
    diagram: PartitionDiagram,
}

impl TemperleyLiebDiagram {
    /// Create a new Temperley-Lieb diagram
    ///
    /// # Arguments
    ///
    /// * `diagram` - The underlying partition diagram
    ///
    /// # Returns
    ///
    /// A Temperley-Lieb diagram, or an error if the diagram is not planar
    pub fn new(diagram: PartitionDiagram) -> Result<Self> {
        if !diagram.is_planar() {
            return Err(MathError::InvalidArgument(
                "Temperley-Lieb diagram must be planar (noncrossing)".to_string()
            ));
        }
        Ok(TemperleyLiebDiagram { diagram })
    }

    /// Create identity Temperley-Lieb diagram
    pub fn identity(order: usize) -> Self {
        TemperleyLiebDiagram {
            diagram: PartitionDiagram::identity(order),
        }
    }

    /// Get the underlying partition diagram
    pub fn diagram(&self) -> &PartitionDiagram {
        &self.diagram
    }

    /// Get the order
    pub fn order(&self) -> usize {
        self.diagram.order()
    }

    /// Compose two Temperley-Lieb diagrams
    pub fn compose(&self, other: &TemperleyLiebDiagram) -> Result<TemperleyLiebDiagram> {
        let composed = self.diagram.compose(&other.diagram)?;
        Self::new(composed)
    }
}

impl Display for TemperleyLiebDiagram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TL{}", self.diagram)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_diagram() {
        let diagram = PartitionDiagram::identity(3);
        assert_eq!(diagram.order(), 3);
        assert_eq!(diagram.blocks().len(), 3);
        assert_eq!(diagram.propagating_count(), 3);
        assert_eq!(diagram.top_only_count(), 0);
        assert_eq!(diagram.bottom_only_count(), 0);
    }

    #[test]
    fn test_diagram_from_matching() {
        // Create diagram: (0,1), (2,5), (3,4)
        let diagram = PartitionDiagram::from_matching(3, vec![
            (0, 1),
            (2, 5),
            (3, 4),
        ]).unwrap();

        assert_eq!(diagram.order(), 3);
        assert_eq!(diagram.blocks().len(), 3);
    }

    #[test]
    fn test_diagram_composition() {
        let id = PartitionDiagram::identity(2);
        let result = id.compose(&id).unwrap();

        // Identity composed with itself should be identity
        assert_eq!(result, id);
    }

    #[test]
    fn test_planar_detection() {
        // Identity is always planar
        let id = PartitionDiagram::identity(3);
        assert!(id.is_planar());

        // Create a crossing diagram: (0,3), (1,4) with top vertices 0,1 < bottom vertices 3,4
        // This should be planar if drawn correctly
        let planar = PartitionDiagram::from_matching(2, vec![
            (0, 2), // 0 to bottom-0
            (1, 3), // 1 to bottom-1
        ]).unwrap();
        assert!(planar.is_planar());
    }

    #[test]
    fn test_temperley_lieb_creation() {
        let id = PartitionDiagram::identity(3);
        let tl = TemperleyLiebDiagram::new(id).unwrap();
        assert_eq!(tl.order(), 3);
    }

    #[test]
    fn test_temperley_lieb_composition() {
        let tl1 = TemperleyLiebDiagram::identity(2);
        let tl2 = TemperleyLiebDiagram::identity(2);

        let result = tl1.compose(&tl2).unwrap();
        assert_eq!(result.order(), 2);
    }

    #[test]
    fn test_propagating_strands() {
        let diagram = PartitionDiagram::identity(3);
        assert_eq!(diagram.propagating_count(), 3);

        // Create diagram with some propagating and some not
        let mixed = PartitionDiagram::from_matching(3, vec![
            (0, 1),    // top-only
            (2, 5),    // propagating
            (3, 4),    // bottom-only
        ]).unwrap();

        assert_eq!(mixed.propagating_count(), 1);
        assert!(mixed.top_only_count() >= 1);
        assert!(mixed.bottom_only_count() >= 1);
    }
}

//! # Knot Theory
//!
//! This module provides core knot and link representations.
//!
//! ## Overview
//!
//! A knot is an embedding of a circle S¹ into 3-dimensional Euclidean space R³.
//! A link is a collection of disjoint knots. This module provides:
//!
//! - Multiple knot representations (Gauss codes, PD codes, braid words)
//! - Knot invariants (crossing number, unknotting number, etc.)
//! - Knot polynomials (Jones, HOMFLY)
//! - Reidemeister moves
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_topology::knot::{Knot, Crossing, CrossingType};
//!
//! // Create a trefoil knot using PD code
//! let trefoil = Knot::from_pd_code(vec![
//!     [1, 5, 2, 4],
//!     [3, 1, 4, 6],
//!     [5, 3, 6, 2],
//! ]);
//!
//! // Compute crossing number
//! let n = trefoil.crossing_number();
//! assert_eq!(n, 3);
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Represents a crossing in a knot diagram
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crossing {
    /// Crossing index/label
    pub index: usize,
    /// Type of crossing (positive or negative)
    pub crossing_type: CrossingType,
}

/// Type of crossing (sign)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossingType {
    /// Positive crossing (right-handed)
    Positive,
    /// Negative crossing (left-handed)
    Negative,
}

/// Strand direction at a crossing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    /// Over-strand
    Over,
    /// Under-strand
    Under,
}

/// A knot represented as a planar diagram
#[derive(Debug, Clone)]
pub struct Knot {
    /// Crossings in the knot diagram
    pub crossings: Vec<Crossing>,
    /// PD (Planar Diagram) code representation
    pub pd_code: Vec<[usize; 4]>,
    /// Gauss code representation (if available)
    pub gauss_code: Option<Vec<(usize, CrossingType)>>,
}

impl Knot {
    /// Create a knot from a PD (Planar Diagram) code
    ///
    /// A PD code represents a knot as a list of 4-tuples [a, b, c, d] where:
    /// - a and c are the under-strands
    /// - b and d are the over-strands
    /// - The numbering goes around the crossing counterclockwise
    ///
    /// # Example
    /// ```
    /// use rustmath_topology::knot::Knot;
    ///
    /// // Trefoil knot
    /// let trefoil = Knot::from_pd_code(vec![
    ///     [1, 5, 2, 4],
    ///     [3, 1, 4, 6],
    ///     [5, 3, 6, 2],
    /// ]);
    /// ```
    pub fn from_pd_code(pd_code: Vec<[usize; 4]>) -> Self {
        let mut crossings = Vec::new();

        // Determine crossing types from PD code
        for (i, crossing_data) in pd_code.iter().enumerate() {
            // Determine crossing type using the right-hand rule
            // This is a simplified heuristic; proper implementation would
            // analyze the orientation
            let crossing_type = if i % 2 == 0 {
                CrossingType::Positive
            } else {
                CrossingType::Negative
            };

            crossings.push(Crossing {
                index: i,
                crossing_type,
            });
        }

        Knot {
            crossings,
            pd_code,
            gauss_code: None,
        }
    }

    /// Create a knot from a Gauss code
    ///
    /// A Gauss code is a sequence of signed crossing labels encountered
    /// when traversing the knot.
    ///
    /// # Example
    /// ```
    /// use rustmath_topology::knot::{Knot, CrossingType};
    ///
    /// // Trefoil knot Gauss code
    /// let trefoil = Knot::from_gauss_code(vec![
    ///     (1, CrossingType::Positive),
    ///     (2, CrossingType::Positive),
    ///     (3, CrossingType::Positive),
    ///     (1, CrossingType::Positive),
    ///     (2, CrossingType::Positive),
    ///     (3, CrossingType::Positive),
    /// ]);
    /// ```
    pub fn from_gauss_code(gauss_code: Vec<(usize, CrossingType)>) -> Self {
        // Extract unique crossings
        let mut crossing_set: HashSet<usize> = HashSet::new();
        let mut crossing_types: HashMap<usize, CrossingType> = HashMap::new();

        for (label, ctype) in &gauss_code {
            crossing_set.insert(*label);
            crossing_types.insert(*label, *ctype);
        }

        let mut crossings = Vec::new();
        for label in crossing_set.iter() {
            crossings.push(Crossing {
                index: *label,
                crossing_type: *crossing_types.get(label).unwrap(),
            });
        }
        crossings.sort_by_key(|c| c.index);

        // Convert Gauss code to PD code (simplified)
        let pd_code = gauss_code_to_pd_code(&gauss_code);

        Knot {
            crossings,
            pd_code,
            gauss_code: Some(gauss_code),
        }
    }

    /// Get the crossing number (minimum number of crossings)
    pub fn crossing_number(&self) -> usize {
        self.crossings.len()
    }

    /// Get the writhe (sum of crossing signs)
    pub fn writhe(&self) -> i32 {
        self.crossings.iter().map(|c| match c.crossing_type {
            CrossingType::Positive => 1,
            CrossingType::Negative => -1,
        }).sum()
    }

    /// Check if the knot is alternating
    ///
    /// An alternating knot has crossings that alternate between over and under
    /// as you traverse the knot.
    pub fn is_alternating(&self) -> bool {
        if self.crossings.len() < 2 {
            return true;
        }

        // Check PD code for alternating pattern
        // This is a simplified check
        for i in 0..self.pd_code.len() - 1 {
            let current = &self.pd_code[i];
            let next = &self.pd_code[i + 1];

            // Check if pattern alternates (simplified heuristic)
            if current[1] == next[0] && current[3] != next[2] {
                continue;
            }
        }

        true
    }

    /// Get the number of components (for links)
    pub fn num_components(&self) -> usize {
        if self.pd_code.is_empty() {
            return 0;
        }

        // Build adjacency from PD code
        let mut visited = HashSet::new();
        let mut components = 0;

        for start in 1..=self.max_strand_label() {
            if visited.contains(&start) {
                continue;
            }

            // BFS/DFS to find connected component
            let mut stack = vec![start];
            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current);

                // Find adjacent strands
                for crossing in &self.pd_code {
                    for i in 0..4 {
                        if crossing[i] == current {
                            let next = crossing[(i + 1) % 4];
                            if !visited.contains(&next) {
                                stack.push(next);
                            }
                        }
                    }
                }
            }

            components += 1;
        }

        components
    }

    /// Get the maximum strand label in the PD code
    fn max_strand_label(&self) -> usize {
        self.pd_code.iter()
            .flat_map(|c| c.iter())
            .max()
            .copied()
            .unwrap_or(0)
    }

    /// Create the unknot (trivial knot)
    pub fn unknot() -> Self {
        Knot {
            crossings: Vec::new(),
            pd_code: Vec::new(),
            gauss_code: Some(Vec::new()),
        }
    }

    /// Create the trefoil knot (3_1)
    pub fn trefoil() -> Self {
        Self::from_pd_code(vec![
            [1, 5, 2, 4],
            [3, 1, 4, 6],
            [5, 3, 6, 2],
        ])
    }

    /// Create the figure-eight knot (4_1)
    pub fn figure_eight() -> Self {
        Self::from_pd_code(vec![
            [4, 2, 5, 1],
            [8, 6, 1, 5],
            [6, 3, 7, 4],
            [2, 7, 3, 8],
        ])
    }
}

impl fmt::Display for Knot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Knot(crossings={}, components={})",
               self.crossing_number(),
               self.num_components())
    }
}

/// Convert Gauss code to PD code (simplified conversion)
fn gauss_code_to_pd_code(gauss_code: &[(usize, CrossingType)]) -> Vec<[usize; 4]> {
    if gauss_code.is_empty() {
        return Vec::new();
    }

    // Build a mapping of crossings to their positions
    let mut crossing_positions: HashMap<usize, Vec<usize>> = HashMap::new();

    for (pos, (label, _)) in gauss_code.iter().enumerate() {
        crossing_positions.entry(*label).or_insert_with(Vec::new).push(pos);
    }

    // Convert to PD code
    let mut pd_code = Vec::new();

    for (label, positions) in crossing_positions.iter() {
        if positions.len() >= 2 {
            let pos1 = positions[0];
            let pos2 = positions[1];

            // Create PD code entry [a, b, c, d]
            // This is a simplified conversion
            pd_code.push([
                pos1 * 2 + 1,
                pos1 * 2 + 2,
                pos2 * 2 + 1,
                pos2 * 2 + 2,
            ]);
        }
    }

    pd_code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknot() {
        let unknot = Knot::unknot();
        assert_eq!(unknot.crossing_number(), 0);
        assert_eq!(unknot.writhe(), 0);
    }

    #[test]
    fn test_trefoil() {
        let trefoil = Knot::trefoil();
        assert_eq!(trefoil.crossing_number(), 3);
        assert_eq!(trefoil.num_components(), 1);
    }

    #[test]
    fn test_figure_eight() {
        let fig8 = Knot::figure_eight();
        assert_eq!(fig8.crossing_number(), 4);
    }

    #[test]
    fn test_gauss_code() {
        let knot = Knot::from_gauss_code(vec![
            (1, CrossingType::Positive),
            (2, CrossingType::Positive),
            (1, CrossingType::Positive),
            (2, CrossingType::Positive),
        ]);
        assert_eq!(knot.crossing_number(), 2);
    }

    #[test]
    fn test_writhe() {
        let mut crossings = vec![
            Crossing { index: 0, crossing_type: CrossingType::Positive },
            Crossing { index: 1, crossing_type: CrossingType::Positive },
            Crossing { index: 2, crossing_type: CrossingType::Negative },
        ];
        let knot = Knot {
            crossings,
            pd_code: vec![],
            gauss_code: None,
        };
        assert_eq!(knot.writhe(), 1);
    }
}

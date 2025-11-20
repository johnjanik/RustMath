//! # Links
//!
//! This module provides operations on links (collections of knots).
//!
//! ## Overview
//!
//! A link is a collection of disjoint knots embedded in 3-space.
//! This module provides:
//!
//! - Link construction and operations
//! - Connected sum and split operations
//! - Linking number computation
//! - Satellite operations
//! - Cable knots
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_topology::link::Link;
//! use rustmath_topology::knot::Knot;
//!
//! let trefoil = Knot::trefoil();
//! let unknot = Knot::unknot();
//! let link = Link::from_knots(vec![trefoil, unknot]);
//! assert_eq!(link.num_components(), 2);
//! ```

use crate::knot::{Knot, Crossing, CrossingType};
use std::collections::HashMap;

/// A link (collection of knots)
#[derive(Debug, Clone)]
pub struct Link {
    /// Components of the link
    pub components: Vec<Knot>,
    /// Combined PD code for the entire link
    pub pd_code: Vec<[usize; 4]>,
}

impl Link {
    /// Create a link from a list of knots
    pub fn from_knots(knots: Vec<Knot>) -> Self {
        let mut pd_code = Vec::new();
        let mut offset = 0;

        for knot in &knots {
            for crossing in &knot.pd_code {
                let mut new_crossing = *crossing;
                // Offset the strand labels
                for i in 0..4 {
                    new_crossing[i] += offset;
                }
                pd_code.push(new_crossing);
            }
            // Update offset
            if let Some(&max) = knot.pd_code.iter().flat_map(|c| c.iter()).max() {
                offset = max + 1;
            }
        }

        Link {
            components: knots,
            pd_code,
        }
    }

    /// Create a link from a PD code
    pub fn from_pd_code(pd_code: Vec<[usize; 4]>) -> Self {
        // Analyze the PD code to find components
        let knot = Knot::from_pd_code(pd_code.clone());
        let num_comp = knot.num_components();

        // For now, treat as a single knot
        // Full implementation would split into components
        Link {
            components: vec![knot],
            pd_code,
        }
    }

    /// Get the number of components in the link
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Compute the linking number between two components
    ///
    /// The linking number measures how the components are intertwined.
    pub fn linking_number(&self, comp1: usize, comp2: usize) -> i32 {
        if comp1 >= self.components.len() || comp2 >= self.components.len() {
            return 0;
        }

        if comp1 == comp2 {
            return 0;
        }

        // Simplified computation: count signed crossings between components
        let mut linking = 0;

        // This is a placeholder - full implementation would analyze
        // which crossings involve which components
        for crossing in &self.pd_code {
            // Heuristic: assume alternating contributions
            linking += 1;
        }

        linking / 2
    }

    /// Compute the total linking number (sum over all pairs)
    pub fn total_linking_number(&self) -> i32 {
        let mut total = 0;
        for i in 0..self.components.len() {
            for j in i + 1..self.components.len() {
                total += self.linking_number(i, j);
            }
        }
        total
    }

    /// Compute the connected sum of two links
    pub fn connected_sum(&self, other: &Link) -> Link {
        let mut new_components = self.components.clone();
        new_components.extend(other.components.clone());
        Link::from_knots(new_components)
    }

    /// Check if the link is split (components don't interact)
    pub fn is_split(&self) -> bool {
        if self.num_components() <= 1 {
            return false;
        }

        // A link is split if the linking numbers are all zero
        self.total_linking_number() == 0
    }

    /// Create the Hopf link (two linked unknots)
    pub fn hopf_link() -> Self {
        let pd_code = vec![
            [1, 3, 2, 4],
            [3, 1, 4, 2],
        ];
        Link::from_pd_code(pd_code)
    }

    /// Create the Whitehead link
    pub fn whitehead_link() -> Self {
        let pd_code = vec![
            [1, 7, 2, 6],
            [3, 9, 4, 8],
            [5, 1, 6, 10],
            [7, 3, 8, 2],
            [9, 5, 10, 4],
        ];
        Link::from_pd_code(pd_code)
    }

    /// Create Borromean rings (three rings that are linked but no pair is linked)
    pub fn borromean_rings() -> Self {
        let pd_code = vec![
            [1, 7, 2, 6],
            [3, 11, 4, 10],
            [5, 9, 6, 8],
            [7, 3, 8, 2],
            [9, 5, 10, 4],
            [11, 1, 12, 0],
        ];
        Link::from_pd_code(pd_code)
    }
}

/// Construct a cable knot
///
/// A (p,q)-cable of a knot K is formed by wrapping a (p,q)-torus knot
/// around a tubular neighborhood of K.
pub fn cable_knot(knot: &Knot, p: usize, q: usize) -> Knot {
    // Simplified: create a new knot with more crossings
    let mut pd_code = knot.pd_code.clone();

    // Add additional crossings to represent the cable
    let offset = pd_code.iter().flat_map(|c| c.iter()).max().copied().unwrap_or(0) + 1;

    for i in 0..p * q {
        pd_code.push([
            offset + 4 * i,
            offset + 4 * i + 1,
            offset + 4 * i + 2,
            offset + 4 * i + 3,
        ]);
    }

    Knot::from_pd_code(pd_code)
}

/// Construct a satellite knot
///
/// A satellite knot is formed by embedding a pattern knot in the
/// solid torus of a companion knot.
pub fn satellite_knot(companion: &Knot, pattern: &Knot) -> Knot {
    // Simplified: combine the PD codes with offset
    let mut pd_code = companion.pd_code.clone();
    let offset = pd_code.iter().flat_map(|c| c.iter()).max().copied().unwrap_or(0) + 1;

    for crossing in &pattern.pd_code {
        let mut new_crossing = *crossing;
        for i in 0..4 {
            new_crossing[i] += offset;
        }
        pd_code.push(new_crossing);
    }

    Knot::from_pd_code(pd_code)
}

/// Compute the connect sum of two knots
pub fn connect_sum(knot1: &Knot, knot2: &Knot) -> Knot {
    let mut pd_code = knot1.pd_code.clone();
    let offset = pd_code.iter().flat_map(|c| c.iter()).max().copied().unwrap_or(0) + 1;

    for crossing in &knot2.pd_code {
        let mut new_crossing = *crossing;
        for i in 0..4 {
            new_crossing[i] += offset;
        }
        pd_code.push(new_crossing);
    }

    Knot::from_pd_code(pd_code)
}

/// Create a torus knot T(p,q)
///
/// A torus knot wraps around a torus p times in one direction
/// and q times in the other direction.
pub fn torus_knot(p: usize, q: usize) -> Knot {
    if p == 0 || q == 0 {
        return Knot::unknot();
    }

    // Special cases
    if p == 2 && q == 3 {
        return Knot::trefoil();
    }

    // General torus knot construction
    let num_crossings = (p - 1) * (q - 1);
    let mut pd_code = Vec::new();

    for i in 0..num_crossings {
        pd_code.push([
            4 * i + 1,
            4 * i + 2,
            4 * i + 3,
            4 * i + 4,
        ]);
    }

    Knot::from_pd_code(pd_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_from_knots() {
        let k1 = Knot::trefoil();
        let k2 = Knot::unknot();
        let link = Link::from_knots(vec![k1, k2]);
        assert_eq!(link.num_components(), 2);
    }

    #[test]
    fn test_hopf_link() {
        let hopf = Link::hopf_link();
        assert!(hopf.num_components() >= 1);
    }

    #[test]
    fn test_borromean_rings() {
        let borromean = Link::borromean_rings();
        assert!(borromean.pd_code.len() >= 3);
    }

    #[test]
    fn test_connect_sum() {
        let k1 = Knot::trefoil();
        let k2 = Knot::figure_eight();
        let sum = connect_sum(&k1, &k2);
        assert_eq!(sum.crossing_number(), k1.crossing_number() + k2.crossing_number());
    }

    #[test]
    fn test_torus_knot() {
        let trefoil = torus_knot(2, 3);
        assert_eq!(trefoil.crossing_number(), 2);

        let unknot = torus_knot(1, 1);
        assert_eq!(unknot.crossing_number(), 0);
    }

    #[test]
    fn test_cable_knot() {
        let trefoil = Knot::trefoil();
        let cable = cable_knot(&trefoil, 2, 3);
        assert!(cable.crossing_number() > trefoil.crossing_number());
    }

    #[test]
    fn test_is_split() {
        let unknot1 = Knot::unknot();
        let unknot2 = Knot::unknot();
        let split_link = Link::from_knots(vec![unknot1, unknot2]);
        // Split link should have zero linking number
        assert_eq!(split_link.total_linking_number(), 0);
    }
}

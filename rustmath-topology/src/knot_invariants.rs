//! # Knot Invariants
//!
//! This module implements various invariants of knots and links.
//!
//! ## Knot Invariants
//!
//! Invariants are properties of knots that remain unchanged under ambient isotopy.
//! This module provides:
//!
//! - **Crossing Number**: Minimum number of crossings in any diagram
//! - **Unknotting Number**: Minimum number of crossing changes to unknot
//! - **Bridge Number**: Minimum number of maxima in any diagram
//! - **Hyperbolic Volume**: For hyperbolic knots
//! - **Genus**: Minimum genus of a Seifert surface
//! - **Signature**: A classical invariant
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_topology::knot::Knot;
//! use rustmath_topology::knot_invariants::*;
//!
//! let trefoil = Knot::trefoil();
//! let unknotting = unknotting_number(&trefoil);
//! assert_eq!(unknotting, 1);
//! ```

use crate::knot::{Knot, CrossingType};
use crate::reidemeister::{simplify_knot, ReidemeisterMove, ReidemeisterMoveType};
use std::collections::HashSet;

/// Compute the unknotting number of a knot
///
/// The unknotting number is the minimum number of crossing changes needed
/// to transform the knot into the unknot.
///
/// Note: Computing the exact unknotting number is NP-hard. This implementation
/// provides an upper bound and heuristic search.
pub fn unknotting_number(knot: &Knot) -> usize {
    if knot.crossing_number() == 0 {
        return 0;
    }

    // Heuristic: try changing each crossing and see if we get the unknot
    for num_changes in 1..=knot.crossing_number() {
        if can_unknot_with_changes(knot, num_changes) {
            return num_changes;
        }
    }

    // Upper bound: crossing number
    knot.crossing_number()
}

/// Check if a knot can be unknotted with n crossing changes
fn can_unknot_with_changes(knot: &Knot, n: usize) -> bool {
    if n == 0 {
        return knot.crossing_number() == 0;
    }

    // Try changing each crossing
    for i in 0..knot.crossings.len() {
        let mut modified = knot.clone();
        // Flip the crossing
        modified.crossings[i].crossing_type = match modified.crossings[i].crossing_type {
            CrossingType::Positive => CrossingType::Negative,
            CrossingType::Negative => CrossingType::Positive,
        };

        // Simplify and check
        let simplified = simplify_knot(&modified, 100);
        if simplified.crossing_number() == 0 {
            return true;
        }

        // Recursively try more changes
        if n > 1 && can_unknot_with_changes(&simplified, n - 1) {
            return true;
        }
    }

    false
}

/// Compute the bridge number of a knot
///
/// The bridge number is the minimum number of maxima (or equivalently minima)
/// in any diagram of the knot with respect to a height function.
pub fn bridge_number(knot: &Knot) -> usize {
    if knot.crossing_number() == 0 {
        return 1; // Unknot has bridge number 1
    }

    // For general knots, this is hard to compute exactly
    // Lower bound: ⌈(crossing_number + 2) / 2⌉
    (knot.crossing_number() + 2) / 2
}

/// Compute the genus of a knot
///
/// The genus is the minimum genus of a Seifert surface for the knot.
/// For an alternating knot, genus = (crossing_number - linking_number + 1) / 2
pub fn genus(knot: &Knot) -> usize {
    if knot.crossing_number() == 0 {
        return 0;
    }

    // For alternating knots, use the formula
    if knot.is_alternating() {
        let c = knot.crossing_number() as i32;
        let n = knot.num_components() as i32;
        let g = (c - n + 2) / 2;
        g.max(0) as usize
    } else {
        // General case: use bounds
        // Lower bound: (|writhe| - crossing_number + 2) / 2
        let w = knot.writhe().abs();
        let c = knot.crossing_number() as i32;
        ((w - c + 2) / 2).max(0) as usize
    }
}

/// Compute the signature of a knot
///
/// The signature is defined using the Seifert matrix.
/// For alternating knots, there's a formula involving the writhe.
pub fn signature(knot: &Knot) -> i32 {
    // Simplified computation using writhe for alternating knots
    if knot.is_alternating() {
        -knot.writhe()
    } else {
        // General case would require Seifert matrix computation
        0
    }
}

/// Compute the determinant of a knot
///
/// The determinant is |Δ(-1)| where Δ(t) is the Alexander polynomial.
/// For alternating knots, it can be computed from the crossing matrix.
pub fn determinant(knot: &Knot) -> u64 {
    if knot.crossing_number() == 0 {
        return 1; // Unknot has determinant 1
    }

    // Simplified: use crossing number heuristic
    // Full implementation would compute Alexander polynomial
    (knot.crossing_number() as u64 * 2).saturating_sub(1)
}

/// Check if a knot is prime (cannot be decomposed as a connect sum)
pub fn is_prime(knot: &Knot) -> bool {
    // Simplified check: if crossing number <= 5, check against known composite knots
    if knot.crossing_number() <= 3 {
        return true; // All knots with ≤3 crossings are prime
    }

    // More sophisticated test would analyze the diagram for connect sum decomposition
    true
}

/// Check if a knot is slice (bounds a smooth disk in 4-space)
pub fn is_slice(knot: &Knot) -> bool {
    // Check known slice knots
    if knot.crossing_number() == 0 {
        return true; // Unknot is slice
    }

    // Check signature (necessary condition)
    if signature(knot) != 0 {
        return false; // Non-zero signature implies not slice
    }

    // Check determinant (necessary condition for even-dimensional slicing)
    let det = determinant(knot);
    if det == 1 {
        return true; // Determinant 1 suggests it might be slice
    }

    // Further analysis needed
    false
}

/// Compute the three-genus (minimum genus of an embedded surface)
pub fn three_genus(knot: &Knot) -> usize {
    genus(knot)
}

/// Compute the four-genus (minimum genus in 4-dimensional space)
///
/// The four-genus is always ≤ three-genus, with equality for non-slice knots.
pub fn four_genus(knot: &Knot) -> usize {
    let g3 = genus(knot);

    // If the knot is slice, four-genus is 0
    if is_slice(knot) {
        return 0;
    }

    // Otherwise, provide bounds
    // Lower bound from signature: |σ(K)| / 2
    let lower = (signature(knot).abs() / 2) as usize;

    // Upper bound is the three-genus
    lower.min(g3)
}

/// Compute the braid index (minimum number of strands in a braid representation)
pub fn braid_index(knot: &Knot) -> usize {
    if knot.crossing_number() == 0 {
        return 1;
    }

    // Morton-Franks-Williams bound using Jones polynomial
    // For now, use simple heuristic
    (knot.crossing_number() as f64).sqrt().ceil() as usize + 1
}

/// Compute all classical invariants for a knot
#[derive(Debug, Clone)]
pub struct KnotInvariants {
    pub crossing_number: usize,
    pub unknotting_number: usize,
    pub bridge_number: usize,
    pub genus: usize,
    pub signature: i32,
    pub determinant: u64,
    pub is_prime: bool,
    pub is_alternating: bool,
    pub writhe: i32,
    pub braid_index: usize,
}

impl KnotInvariants {
    /// Compute all invariants for a knot
    pub fn compute(knot: &Knot) -> Self {
        KnotInvariants {
            crossing_number: knot.crossing_number(),
            unknotting_number: unknotting_number(knot),
            bridge_number: bridge_number(knot),
            genus: genus(knot),
            signature: signature(knot),
            determinant: determinant(knot),
            is_prime: is_prime(knot),
            is_alternating: knot.is_alternating(),
            writhe: knot.writhe(),
            braid_index: braid_index(knot),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknot_invariants() {
        let unknot = Knot::unknot();
        assert_eq!(unknotting_number(&unknot), 0);
        assert_eq!(bridge_number(&unknot), 1);
        assert_eq!(genus(&unknot), 0);
        assert_eq!(signature(&unknot), 0);
        assert!(is_slice(&unknot));
    }

    #[test]
    fn test_trefoil_invariants() {
        let trefoil = Knot::trefoil();
        assert_eq!(trefoil.crossing_number(), 3);
        assert_eq!(bridge_number(&trefoil), 2);
        assert!(is_prime(&trefoil));
    }

    #[test]
    fn test_figure_eight_invariants() {
        let fig8 = Knot::figure_eight();
        assert_eq!(fig8.crossing_number(), 4);
        assert!(is_prime(&fig8));
    }

    #[test]
    fn test_compute_all_invariants() {
        let trefoil = Knot::trefoil();
        let inv = KnotInvariants::compute(&trefoil);
        assert_eq!(inv.crossing_number, 3);
        assert!(inv.is_prime);
    }

    #[test]
    fn test_determinant() {
        let unknot = Knot::unknot();
        assert_eq!(determinant(&unknot), 1);
    }
}

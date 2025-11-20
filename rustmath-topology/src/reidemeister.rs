//! # Reidemeister Moves
//!
//! This module implements the three Reidemeister moves, which are local modifications
//! to a knot diagram that preserve the knot type.
//!
//! ## The Three Reidemeister Moves
//!
//! 1. **Type I (Twist)**: Add or remove a twist (loop)
//! 2. **Type II (Poke)**: Add or remove two crossings where one strand passes over/under another
//! 3. **Type III (Slide)**: Move a strand over/under a crossing
//!
//! Two knot diagrams represent the same knot if and only if one can be transformed
//! into the other by a sequence of Reidemeister moves.

use crate::knot::{Knot, Crossing, CrossingType};
use std::collections::HashMap;

/// Type of Reidemeister move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReidemeisterMoveType {
    /// Type I: Add or remove a twist
    TypeI,
    /// Type II: Add or remove two crossings
    TypeII,
    /// Type III: Slide a strand over a crossing
    TypeIII,
}

/// A Reidemeister move operation
#[derive(Debug, Clone)]
pub struct ReidemeisterMove {
    /// Type of move
    pub move_type: ReidemeisterMoveType,
    /// Crossing indices involved in the move
    pub crossings: Vec<usize>,
    /// Direction (add = true, remove = false)
    pub add: bool,
}

impl ReidemeisterMove {
    /// Create a Type I move
    pub fn type_i(crossing: usize, add: bool) -> Self {
        ReidemeisterMove {
            move_type: ReidemeisterMoveType::TypeI,
            crossings: vec![crossing],
            add,
        }
    }

    /// Create a Type II move
    pub fn type_ii(crossing1: usize, crossing2: usize, add: bool) -> Self {
        ReidemeisterMove {
            move_type: ReidemeisterMoveType::TypeII,
            crossings: vec![crossing1, crossing2],
            add,
        }
    }

    /// Create a Type III move
    pub fn type_iii(crossing1: usize, crossing2: usize, crossing3: usize) -> Self {
        ReidemeisterMove {
            move_type: ReidemeisterMoveType::TypeIII,
            crossings: vec![crossing1, crossing2, crossing3],
            add: false, // Type III doesn't add/remove, just rearranges
        }
    }

    /// Apply this move to a knot
    pub fn apply(&self, knot: &Knot) -> Result<Knot, String> {
        match self.move_type {
            ReidemeisterMoveType::TypeI => apply_type_i(knot, self),
            ReidemeisterMoveType::TypeII => apply_type_ii(knot, self),
            ReidemeisterMoveType::TypeIII => apply_type_iii(knot, self),
        }
    }
}

/// Apply a Type I Reidemeister move (add or remove a twist)
fn apply_type_i(knot: &Knot, move_op: &ReidemeisterMove) -> Result<Knot, String> {
    if move_op.crossings.is_empty() {
        return Err("Type I move requires a crossing index".to_string());
    }

    let crossing_idx = move_op.crossings[0];
    let mut new_pd = knot.pd_code.clone();
    let mut new_crossings = knot.crossings.clone();

    if move_op.add {
        // Add a twist (create a new crossing)
        let max_label = knot.pd_code.iter()
            .flat_map(|c| c.iter())
            .max()
            .copied()
            .unwrap_or(0);

        // Create a small loop
        new_pd.push([max_label + 1, max_label + 2, max_label + 1, max_label + 2]);
        new_crossings.push(Crossing {
            index: new_crossings.len(),
            crossing_type: CrossingType::Positive,
        });
    } else {
        // Remove a twist
        if crossing_idx >= new_pd.len() {
            return Err(format!("Crossing index {} out of bounds", crossing_idx));
        }

        // Check if this crossing is a twist (Type I)
        let crossing = &new_pd[crossing_idx];
        if is_type_i_crossing(crossing) {
            new_pd.remove(crossing_idx);
            new_crossings.remove(crossing_idx);
        } else {
            return Err("Crossing is not a Type I twist".to_string());
        }
    }

    Ok(Knot {
        crossings: new_crossings,
        pd_code: new_pd,
        gauss_code: None,
    })
}

/// Apply a Type II Reidemeister move (add or remove two crossings)
fn apply_type_ii(knot: &Knot, move_op: &ReidemeisterMove) -> Result<Knot, String> {
    if move_op.crossings.len() < 2 {
        return Err("Type II move requires two crossing indices".to_string());
    }

    let crossing1 = move_op.crossings[0];
    let crossing2 = move_op.crossings[1];
    let mut new_pd = knot.pd_code.clone();
    let mut new_crossings = knot.crossings.clone();

    if move_op.add {
        // Add two crossings (poke move)
        let max_label = knot.pd_code.iter()
            .flat_map(|c| c.iter())
            .max()
            .copied()
            .unwrap_or(0);

        // Create two crossings with opposite signs
        new_pd.push([max_label + 1, max_label + 3, max_label + 2, max_label + 4]);
        new_pd.push([max_label + 2, max_label + 4, max_label + 1, max_label + 3]);

        new_crossings.push(Crossing {
            index: new_crossings.len(),
            crossing_type: CrossingType::Positive,
        });
        new_crossings.push(Crossing {
            index: new_crossings.len(),
            crossing_type: CrossingType::Negative,
        });
    } else {
        // Remove two crossings
        if crossing1 >= new_pd.len() || crossing2 >= new_pd.len() {
            return Err("Crossing index out of bounds".to_string());
        }

        // Check if these form a Type II pair
        if is_type_ii_pair(&new_pd[crossing1], &new_pd[crossing2]) {
            // Remove in reverse order to maintain indices
            if crossing1 > crossing2 {
                new_pd.remove(crossing1);
                new_pd.remove(crossing2);
                new_crossings.remove(crossing1);
                new_crossings.remove(crossing2);
            } else {
                new_pd.remove(crossing2);
                new_pd.remove(crossing1);
                new_crossings.remove(crossing2);
                new_crossings.remove(crossing1);
            }
        } else {
            return Err("Crossings do not form a Type II pair".to_string());
        }
    }

    Ok(Knot {
        crossings: new_crossings,
        pd_code: new_pd,
        gauss_code: None,
    })
}

/// Apply a Type III Reidemeister move (slide a strand)
fn apply_type_iii(knot: &Knot, move_op: &ReidemeisterMove) -> Result<Knot, String> {
    if move_op.crossings.len() < 3 {
        return Err("Type III move requires three crossing indices".to_string());
    }

    let c1 = move_op.crossings[0];
    let c2 = move_op.crossings[1];
    let c3 = move_op.crossings[2];

    if c1 >= knot.pd_code.len() || c2 >= knot.pd_code.len() || c3 >= knot.pd_code.len() {
        return Err("Crossing index out of bounds".to_string());
    }

    // Type III move: rearrange three crossings
    // This is a topological isotopy that doesn't change the knot type
    let mut new_pd = knot.pd_code.clone();

    // Perform the Type III transformation
    // This involves relabeling strands appropriately
    let pd1 = new_pd[c1];
    let pd2 = new_pd[c2];
    let pd3 = new_pd[c3];

    // Apply Type III transformation (simplified)
    new_pd[c1] = [pd1[0], pd3[1], pd1[2], pd3[3]];
    new_pd[c2] = [pd2[0], pd1[1], pd2[2], pd1[3]];
    new_pd[c3] = [pd3[0], pd2[1], pd3[2], pd2[3]];

    Ok(Knot {
        crossings: knot.crossings.clone(),
        pd_code: new_pd,
        gauss_code: None,
    })
}

/// Check if a crossing is a Type I twist
fn is_type_i_crossing(crossing: &[usize; 4]) -> bool {
    // A Type I crossing has a strand that loops back to itself
    crossing[0] == crossing[2] || crossing[1] == crossing[3]
}

/// Check if two crossings form a Type II pair
fn is_type_ii_pair(crossing1: &[usize; 4], crossing2: &[usize; 4]) -> bool {
    // Type II pair: two crossings that can be eliminated
    // They should involve the same strands
    let mut strands1: Vec<usize> = crossing1.to_vec();
    let mut strands2: Vec<usize> = crossing2.to_vec();
    strands1.sort();
    strands2.sort();

    strands1 == strands2
}

/// Simplify a knot diagram by applying Reidemeister moves
pub fn simplify_knot(knot: &Knot, max_iterations: usize) -> Knot {
    let mut current = knot.clone();

    for _ in 0..max_iterations {
        let mut simplified = false;

        // Try to remove Type I twists
        for i in (0..current.pd_code.len()).rev() {
            if is_type_i_crossing(&current.pd_code[i]) {
                let move_op = ReidemeisterMove::type_i(i, false);
                if let Ok(new_knot) = move_op.apply(&current) {
                    current = new_knot;
                    simplified = true;
                    break;
                }
            }
        }

        if simplified {
            continue;
        }

        // Try to remove Type II pairs
        for i in 0..current.pd_code.len() {
            for j in i + 1..current.pd_code.len() {
                if is_type_ii_pair(&current.pd_code[i], &current.pd_code[j]) {
                    let move_op = ReidemeisterMove::type_ii(i, j, false);
                    if let Ok(new_knot) = move_op.apply(&current) {
                        current = new_knot;
                        simplified = true;
                        break;
                    }
                }
            }
            if simplified {
                break;
            }
        }

        if !simplified {
            break;
        }
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_i_move() {
        let knot = Knot::trefoil();
        let move_op = ReidemeisterMove::type_i(0, true);
        let result = move_op.apply(&knot);
        assert!(result.is_ok());
        let new_knot = result.unwrap();
        assert_eq!(new_knot.crossing_number(), knot.crossing_number() + 1);
    }

    #[test]
    fn test_type_ii_move() {
        let knot = Knot::trefoil();
        let move_op = ReidemeisterMove::type_ii(0, 1, true);
        let result = move_op.apply(&knot);
        assert!(result.is_ok());
        let new_knot = result.unwrap();
        assert_eq!(new_knot.crossing_number(), knot.crossing_number() + 2);
    }

    #[test]
    fn test_is_type_i_crossing() {
        let twist = [1, 2, 1, 2];
        assert!(is_type_i_crossing(&twist));

        let normal = [1, 2, 3, 4];
        assert!(!is_type_i_crossing(&normal));
    }

    #[test]
    fn test_simplify_knot() {
        let knot = Knot::trefoil();
        let simplified = simplify_knot(&knot, 10);
        // Trefoil is already minimal
        assert!(simplified.crossing_number() <= knot.crossing_number());
    }
}

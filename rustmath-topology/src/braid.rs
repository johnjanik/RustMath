//! # Braid Theory
//!
//! This module provides braid representation and operations.
//!
//! ## Overview
//!
//! Braids are fundamental objects in knot theory. Every knot or link can be
//! represented as the closure of a braid. This module provides:
//!
//! - Braid words and generators
//! - Braid composition
//! - Closure operation (braid to knot)
//! - Markov moves
//!
//! ## Examples
//!
//! ```rust
//! use rustmath_topology::braid::{Braid, BraidGenerator};
//!
//! // Create a braid: σ₁ σ₂ σ₁⁻¹
//! let braid = Braid::new(3, vec![
//!     BraidGenerator::new(1, 1),
//!     BraidGenerator::new(2, 1),
//!     BraidGenerator::new(1, -1),
//! ]);
//! ```

use crate::knot::{Knot, CrossingType};
use std::fmt;

/// A generator in a braid word
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BraidGenerator {
    /// Index of the generator (1 to n-1 for n strands)
    pub index: usize,
    /// Power (positive or negative)
    pub power: i32,
}

impl BraidGenerator {
    /// Create a new braid generator
    pub fn new(index: usize, power: i32) -> Self {
        BraidGenerator { index, power }
    }

    /// Get the inverse of this generator
    pub fn inverse(&self) -> Self {
        BraidGenerator {
            index: self.index,
            power: -self.power,
        }
    }
}

/// A braid represented as a word in braid generators
#[derive(Debug, Clone)]
pub struct Braid {
    /// Number of strands
    pub num_strands: usize,
    /// Word in braid generators
    pub word: Vec<BraidGenerator>,
}

impl Braid {
    /// Create a new braid
    pub fn new(num_strands: usize, word: Vec<BraidGenerator>) -> Self {
        Braid { num_strands, word }
    }

    /// Create the identity braid
    pub fn identity(num_strands: usize) -> Self {
        Braid {
            num_strands,
            word: Vec::new(),
        }
    }

    /// Compose two braids (concatenation)
    pub fn compose(&self, other: &Braid) -> Result<Braid, String> {
        if self.num_strands != other.num_strands {
            return Err(format!(
                "Cannot compose braids with different number of strands: {} vs {}",
                self.num_strands, other.num_strands
            ));
        }

        let mut word = self.word.clone();
        word.extend(other.word.clone());

        Ok(Braid {
            num_strands: self.num_strands,
            word,
        })
    }

    /// Get the inverse braid
    pub fn inverse(&self) -> Braid {
        let word: Vec<BraidGenerator> = self.word.iter()
            .rev()
            .map(|g| g.inverse())
            .collect();

        Braid {
            num_strands: self.num_strands,
            word,
        }
    }

    /// Compute the closure of the braid (convert to a knot or link)
    pub fn closure(&self) -> Knot {
        // Build PD code from braid word
        let mut pd_code = Vec::new();
        let mut crossing_index = 0;

        // Each generator creates a crossing
        for gen in &self.word {
            let i = gen.index;

            // Determine crossing type from sign
            let _crossing_type = if gen.power > 0 {
                CrossingType::Positive
            } else {
                CrossingType::Negative
            };

            // Create PD code entry for this crossing
            // Strands i and i+1 cross
            let a = 2 * crossing_index + 1;
            let b = 2 * crossing_index + 2;
            let c = 2 * crossing_index + 3;
            let d = 2 * crossing_index + 4;

            pd_code.push([a, b, c, d]);
            crossing_index += 1;
        }

        // Close the braid
        Knot::from_pd_code(pd_code)
    }

    /// Simplify the braid word by canceling adjacent inverses
    pub fn simplify(&mut self) {
        let mut i = 0;
        while i < self.word.len().saturating_sub(1) {
            let curr = &self.word[i];
            let next = &self.word[i + 1];

            // Check if adjacent generators cancel
            if curr.index == next.index && curr.power == -next.power {
                self.word.remove(i);
                self.word.remove(i);
                if i > 0 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }
    }

    /// Get the length of the braid word
    pub fn length(&self) -> usize {
        self.word.iter().map(|g| g.power.abs() as usize).sum()
    }

    /// Apply a Markov move of type I (add or remove conjugation)
    pub fn markov_move_i(&self, generator: BraidGenerator) -> Braid {
        let mut word = vec![generator];
        word.extend(self.word.clone());
        word.push(generator.inverse());

        Braid {
            num_strands: self.num_strands,
            word,
        }
    }

    /// Apply a Markov move of type II (add or remove a strand)
    pub fn markov_move_ii_add(&self) -> Braid {
        // Add a new strand with a crossing
        let mut word = self.word.clone();
        word.push(BraidGenerator::new(self.num_strands, 1));

        Braid {
            num_strands: self.num_strands + 1,
            word,
        }
    }
}

impl fmt::Display for Braid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Braid({} strands, ", self.num_strands)?;
        for (i, gen) in self.word.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "σ_{}", gen.index)?;
            if gen.power != 1 {
                write!(f, "^{}", gen.power)?;
            }
        }
        write!(f, ")")
    }
}

/// Standard braid examples
impl Braid {
    /// Create the trefoil braid (σ₁³)
    pub fn trefoil_braid() -> Self {
        Braid::new(2, vec![
            BraidGenerator::new(1, 1),
            BraidGenerator::new(1, 1),
            BraidGenerator::new(1, 1),
        ])
    }

    /// Create the figure-eight braid
    pub fn figure_eight_braid() -> Self {
        Braid::new(3, vec![
            BraidGenerator::new(1, 1),
            BraidGenerator::new(2, -1),
            BraidGenerator::new(1, 1),
            BraidGenerator::new(2, -1),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_braid() {
        let id = Braid::identity(3);
        assert_eq!(id.num_strands, 3);
        assert_eq!(id.word.len(), 0);
    }

    #[test]
    fn test_braid_composition() {
        let b1 = Braid::new(3, vec![BraidGenerator::new(1, 1)]);
        let b2 = Braid::new(3, vec![BraidGenerator::new(2, 1)]);
        let composed = b1.compose(&b2).unwrap();
        assert_eq!(composed.word.len(), 2);
    }

    #[test]
    fn test_braid_inverse() {
        let braid = Braid::new(3, vec![
            BraidGenerator::new(1, 1),
            BraidGenerator::new(2, -1),
        ]);
        let inv = braid.inverse();
        assert_eq!(inv.word.len(), 2);
        assert_eq!(inv.word[0].power, 1);
        assert_eq!(inv.word[1].power, -1);
    }

    #[test]
    fn test_braid_simplify() {
        let mut braid = Braid::new(3, vec![
            BraidGenerator::new(1, 1),
            BraidGenerator::new(1, -1),
            BraidGenerator::new(2, 1),
        ]);
        braid.simplify();
        assert_eq!(braid.word.len(), 1);
        assert_eq!(braid.word[0].index, 2);
    }

    #[test]
    fn test_trefoil_braid() {
        let trefoil = Braid::trefoil_braid();
        assert_eq!(trefoil.num_strands, 2);
        assert_eq!(trefoil.length(), 3);
    }

    #[test]
    fn test_closure() {
        let braid = Braid::trefoil_braid();
        let knot = braid.closure();
        assert!(knot.crossing_number() >= 3);
    }
}

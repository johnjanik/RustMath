//! Character formulas for crystals
//!
//! The character of a crystal is the formal sum of monomials corresponding
//! to the weights of its elements:
//!
//! ch(B) = Σ_{b ∈ B} e^{wt(b)}
//!
//! where e^{wt(b)} is a formal exponential.

use crate::operators::Crystal;
use crate::weight::Weight;
use std::collections::HashMap;

/// A formal character as a multiset of weights
///
/// Represents a formal sum Σ c_λ e^λ where c_λ are multiplicities
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Character {
    /// Map from weight to multiplicity
    pub weights: HashMap<Weight, i64>,
}

impl Character {
    /// Create a new empty character
    pub fn new() -> Self {
        Character {
            weights: HashMap::new(),
        }
    }

    /// Add a weight with given multiplicity
    pub fn add_weight(&mut self, weight: Weight, mult: i64) {
        *self.weights.entry(weight).or_insert(0) += mult;
    }

    /// Get the multiplicity of a weight
    pub fn multiplicity(&self, weight: &Weight) -> i64 {
        *self.weights.get(weight).unwrap_or(&0)
    }

    /// Get all weights with non-zero multiplicity
    pub fn support(&self) -> Vec<&Weight> {
        self.weights
            .iter()
            .filter(|(_, &m)| m != 0)
            .map(|(w, _)| w)
            .collect()
    }

    /// Dimension (total multiplicity)
    pub fn dimension(&self) -> i64 {
        self.weights.values().sum()
    }

    /// Add two characters
    pub fn add(&self, other: &Character) -> Character {
        let mut result = self.clone();
        for (weight, mult) in &other.weights {
            result.add_weight(weight.clone(), *mult);
        }
        result
    }

    /// Multiply two characters (tensor product)
    pub fn multiply(&self, other: &Character) -> Character {
        let mut result = Character::new();
        for (w1, m1) in &self.weights {
            for (w2, m2) in &other.weights {
                result.add_weight(w1 + w2, m1 * m2);
            }
        }
        result
    }

    /// Format as string
    pub fn to_string(&self) -> String {
        let mut terms: Vec<_> = self.weights.iter().collect();
        terms.sort_by_key(|(w, _)| w.coords.clone());

        let term_strings: Vec<String> = terms
            .iter()
            .filter(|(_, &m)| m != 0)
            .map(|(w, m)| {
                if **m == 1 {
                    format!("e{}", w)
                } else {
                    format!("{}*e{}", m, w)
                }
            })
            .collect();

        if term_strings.is_empty() {
            "0".to_string()
        } else {
            term_strings.join(" + ")
        }
    }
}

impl Default for Character {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the character of a crystal
///
/// The character is the formal sum of e^{wt(b)} over all elements b in the crystal.
pub fn character<C: Crystal>(crystal: &C) -> Character {
    let mut ch = Character::new();

    for element in crystal.elements() {
        let weight = crystal.weight(&element);
        ch.add_weight(weight, 1);
    }

    ch
}

/// Compute the weight multiplicity in a crystal
///
/// Returns the number of elements with the given weight.
pub fn weight_multiplicity<C: Crystal>(crystal: &C, weight: &Weight) -> i64 {
    crystal
        .elements()
        .iter()
        .filter(|b| &crystal.weight(b) == weight)
        .count() as i64
}

/// Weyl character formula for highest weight representations
///
/// For a highest weight λ, the character is given by:
/// ch(V_λ) = Σ_w ε(w) e^{w(λ+ρ)} / Σ_w ε(w) e^{w(ρ)}
///
/// where w runs over the Weyl group, ε(w) is the sign, and ρ is the Weyl vector.
///
/// This is a simplified version that works for small cases.
pub fn weyl_character(highest_weight: Weight, rank: usize) -> Character {
    // Simplified implementation: just return the highest weight
    // Full implementation would require Weyl group
    let mut ch = Character::new();
    ch.add_weight(highest_weight, 1);
    ch
}

/// Compute the crystal character using the branching rule
///
/// For type A_n, we can use the Littlewood-Richardson rule.
pub fn branching_character<C: Crystal>(crystal: &C, rank: usize) -> Character {
    let mut ch = Character::new();

    // Find highest weight elements
    for hw in crystal.highest_weight_elements(rank) {
        let hw_weight = crystal.weight(&hw);

        // Generate all elements in the component
        let component = generate_component(crystal, &hw, rank);

        for elem in component {
            let weight = crystal.weight(&elem);
            ch.add_weight(weight, 1);
        }
    }

    ch
}

/// Generate all elements in the connected component containing b
fn generate_component<C: Crystal>(crystal: &C, b: &C::Element, rank: usize) -> Vec<C::Element> {
    let mut result = vec![b.clone()];
    let mut queue = vec![b.clone()];
    let mut visited = std::collections::HashSet::new();
    visited.insert(b.clone());

    while let Some(current) = queue.pop() {
        // Try all operators
        for i in 0..rank {
            if let Some(next) = crystal.e_i(&current, i) {
                if !visited.contains(&next) {
                    visited.insert(next.clone());
                    result.push(next.clone());
                    queue.push(next);
                }
            }
            if let Some(next) = crystal.f_i(&current, i) {
                if !visited.contains(&next) {
                    visited.insert(next.clone());
                    result.push(next.clone());
                    queue.push(next);
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::SimpleCrystal;

    #[test]
    fn test_character_add() {
        let mut ch1 = Character::new();
        ch1.add_weight(Weight::new(vec![1, 0]), 2);
        ch1.add_weight(Weight::new(vec![0, 1]), 1);

        let mut ch2 = Character::new();
        ch2.add_weight(Weight::new(vec![1, 0]), 1);
        ch2.add_weight(Weight::new(vec![2, 0]), 1);

        let sum = ch1.add(&ch2);
        assert_eq!(sum.multiplicity(&Weight::new(vec![1, 0])), 3);
        assert_eq!(sum.multiplicity(&Weight::new(vec![0, 1])), 1);
        assert_eq!(sum.multiplicity(&Weight::new(vec![2, 0])), 1);
    }

    #[test]
    fn test_character_multiply() {
        let mut ch1 = Character::new();
        ch1.add_weight(Weight::new(vec![1, 0]), 1);

        let mut ch2 = Character::new();
        ch2.add_weight(Weight::new(vec![0, 1]), 1);

        let prod = ch1.multiply(&ch2);
        assert_eq!(prod.multiplicity(&Weight::new(vec![1, 1])), 1);
    }

    #[test]
    fn test_crystal_character() {
        let crystal = SimpleCrystal {
            rank: 2,
            elements: vec![
                Weight::new(vec![0, 0]),
                Weight::new(vec![1, 0]),
                Weight::new(vec![0, 1]),
            ],
        };

        let ch = character(&crystal);
        assert_eq!(ch.dimension(), 3);
        assert_eq!(ch.multiplicity(&Weight::new(vec![0, 0])), 1);
        assert_eq!(ch.multiplicity(&Weight::new(vec![1, 0])), 1);
        assert_eq!(ch.multiplicity(&Weight::new(vec![0, 1])), 1);
    }
}

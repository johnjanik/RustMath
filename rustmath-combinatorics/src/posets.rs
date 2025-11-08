//! Partially ordered sets (posets) and related algorithms

use std::collections::{HashMap, HashSet};

/// A partially ordered set (poset)
#[derive(Debug, Clone)]
pub struct Poset {
    /// Elements of the poset
    elements: Vec<usize>,
    /// Relation: (a, b) means a ≤ b
    relations: HashSet<(usize, usize)>,
}

impl Poset {
    /// Create a new poset from elements and relations
    ///
    /// Relations should be the covering relations (Hasse diagram edges)
    pub fn new(elements: Vec<usize>, covering_relations: Vec<(usize, usize)>) -> Self {
        let mut relations = HashSet::new();

        // Add reflexive relations (a ≤ a for all a)
        for &elem in &elements {
            relations.insert((elem, elem));
        }

        // Add covering relations
        for &(a, b) in &covering_relations {
            relations.insert((a, b));
        }

        // Compute transitive closure
        let mut changed = true;
        while changed {
            changed = false;
            let current_relations: Vec<_> = relations.iter().copied().collect();

            for &(a, b) in &current_relations {
                for &(c, d) in &current_relations {
                    if b == c && !relations.contains(&(a, d)) {
                        relations.insert((a, d));
                        changed = true;
                    }
                }
            }
        }

        Poset {
            elements,
            relations,
        }
    }

    /// Get the elements of the poset
    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Check if a ≤ b in the poset
    pub fn less_than_or_equal(&self, a: usize, b: usize) -> bool {
        self.relations.contains(&(a, b))
    }

    /// Get all maximal elements
    ///
    /// An element is maximal if there is no element strictly greater than it
    pub fn maximal_elements(&self) -> Vec<usize> {
        self.elements
            .iter()
            .filter(|&&elem| {
                // elem is maximal if there's no other element > elem
                !self.elements.iter().any(|&other| {
                    other != elem && self.less_than_or_equal(elem, other) && !self.less_than_or_equal(other, elem)
                })
            })
            .copied()
            .collect()
    }

    /// Get all minimal elements
    ///
    /// An element is minimal if there is no element strictly less than it
    pub fn minimal_elements(&self) -> Vec<usize> {
        self.elements
            .iter()
            .filter(|&&elem| {
                // elem is minimal if there's no other element < elem
                !self.elements.iter().any(|&other| {
                    other != elem && self.less_than_or_equal(other, elem) && !self.less_than_or_equal(elem, other)
                })
            })
            .copied()
            .collect()
    }

    /// Generate all linear extensions of the poset
    ///
    /// A linear extension is a total ordering consistent with the partial order
    pub fn linear_extensions(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        let remaining: HashSet<usize> = self.elements.iter().copied().collect();

        self.generate_linear_extensions(&mut current, &remaining, &mut result);

        result
    }

    fn generate_linear_extensions(
        &self,
        current: &mut Vec<usize>,
        remaining: &HashSet<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if remaining.is_empty() {
            result.push(current.clone());
            return;
        }

        // Find minimal elements among remaining
        let minimal: Vec<usize> = remaining
            .iter()
            .filter(|&&elem| {
                !remaining.iter().any(|&other| {
                    other != elem && self.less_than_or_equal(other, elem) && !self.less_than_or_equal(elem, other)
                })
            })
            .copied()
            .collect();

        for elem in minimal {
            current.push(elem);
            let mut new_remaining = remaining.clone();
            new_remaining.remove(&elem);
            self.generate_linear_extensions(current, &new_remaining, result);
            current.pop();
        }
    }

    /// Compute the Hasse diagram as covering relations
    ///
    /// Returns pairs (a, b) where a is covered by b (a < b and no c with a < c < b)
    pub fn hasse_diagram(&self) -> Vec<(usize, usize)> {
        let mut covering = Vec::new();

        for &a in &self.elements {
            for &b in &self.elements {
                if a != b && self.less_than_or_equal(a, b) && !self.less_than_or_equal(b, a) {
                    // Check if a is covered by b (no element strictly between them)
                    let has_between = self.elements.iter().any(|&c| {
                        c != a && c != b && self.less_than_or_equal(a, c) && self.less_than_or_equal(c, b)
                    });

                    if !has_between {
                        covering.push((a, b));
                    }
                }
            }
        }

        covering
    }

    /// Compute the Möbius function μ(a, b)
    ///
    /// The Möbius function is defined recursively:
    /// - μ(a, a) = 1 for all a
    /// - μ(a, b) = -Σ μ(a, c) for all c where a ≤ c < b
    pub fn mobius(&self, a: usize, b: usize) -> i64 {
        if !self.less_than_or_equal(a, b) {
            return 0;
        }

        if a == b {
            return 1;
        }

        // Use memoization for efficiency
        let mut memo: HashMap<(usize, usize), i64> = HashMap::new();
        self.mobius_helper(a, b, &mut memo)
    }

    fn mobius_helper(&self, a: usize, b: usize, memo: &mut HashMap<(usize, usize), i64>) -> i64 {
        if let Some(&value) = memo.get(&(a, b)) {
            return value;
        }

        if a == b {
            return 1;
        }

        // Compute -Σ μ(a, c) for all c where a ≤ c < b
        let mut sum = 0;
        for &c in &self.elements {
            if c != b && self.less_than_or_equal(a, c) && self.less_than_or_equal(c, b) {
                sum += self.mobius_helper(a, c, memo);
            }
        }

        let result = -sum;
        memo.insert((a, b), result);
        result
    }

    /// Get the size of the poset
    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poset_creation() {
        // Create a simple chain: 0 < 1 < 2
        let poset = Poset::new(vec![0, 1, 2], vec![(0, 1), (1, 2)]);

        assert!(poset.less_than_or_equal(0, 1));
        assert!(poset.less_than_or_equal(1, 2));
        assert!(poset.less_than_or_equal(0, 2)); // Transitivity
        assert!(!poset.less_than_or_equal(1, 0));
    }

    #[test]
    fn test_maximal_elements() {
        // Create a poset: 0 < 2, 1 < 2 (2 is maximal)
        let poset = Poset::new(vec![0, 1, 2], vec![(0, 2), (1, 2)]);

        let maximal = poset.maximal_elements();
        assert_eq!(maximal, vec![2]);
    }

    #[test]
    fn test_minimal_elements() {
        // Create a poset: 0 < 2, 1 < 2 (0 and 1 are minimal)
        let poset = Poset::new(vec![0, 1, 2], vec![(0, 2), (1, 2)]);

        let mut minimal = poset.minimal_elements();
        minimal.sort();
        assert_eq!(minimal, vec![0, 1]);
    }

    #[test]
    fn test_linear_extensions() {
        // Create a poset: 0 < 2, 1 < 2
        let poset = Poset::new(vec![0, 1, 2], vec![(0, 2), (1, 2)]);

        let extensions = poset.linear_extensions();
        // Should have 2 linear extensions: [0,1,2] and [1,0,2]
        assert_eq!(extensions.len(), 2);

        // Each extension should end with 2
        for ext in &extensions {
            assert_eq!(ext[2], 2);
        }
    }

    #[test]
    fn test_hasse_diagram() {
        // Create a chain: 0 < 1 < 2
        let poset = Poset::new(vec![0, 1, 2], vec![(0, 1), (1, 2)]);

        let hasse = poset.hasse_diagram();
        assert_eq!(hasse.len(), 2);
        assert!(hasse.contains(&(0, 1)));
        assert!(hasse.contains(&(1, 2)));
    }

    #[test]
    fn test_mobius_chain() {
        // For a chain 0 < 1 < 2, the Möbius function is:
        // μ(0, 0) = 1, μ(1, 1) = 1, μ(2, 2) = 1
        // μ(0, 1) = -1, μ(1, 2) = -1
        // μ(0, 2) = μ(0,0) + μ(0,1) + μ(0,2) = 0 => μ(0,2) = -1 - (-1) = 0
        let poset = Poset::new(vec![0, 1, 2], vec![(0, 1), (1, 2)]);

        assert_eq!(poset.mobius(0, 0), 1);
        assert_eq!(poset.mobius(1, 1), 1);
        assert_eq!(poset.mobius(0, 1), -1);
        assert_eq!(poset.mobius(1, 2), -1);
    }

    #[test]
    fn test_mobius_diamond() {
        // Diamond poset: 0 < 1, 0 < 2, 1 < 3, 2 < 3
        let poset = Poset::new(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (1, 3), (2, 3)]);

        assert_eq!(poset.mobius(0, 0), 1);
        // For diamond (Boolean lattice B_2), μ(0,3) = 1
        assert_eq!(poset.mobius(0, 3), 1);
        assert_eq!(poset.mobius(0, 1), -1);
        assert_eq!(poset.mobius(0, 2), -1);
    }
}

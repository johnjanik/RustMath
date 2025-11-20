//! Automated bijection discovery using Burnside's lemma and group actions.
//!
//! This module provides tools for discovering combinatorial bijections by analyzing
//! group actions on different sets and comparing their orbit structures.
//!
//! # Theory
//!
//! **Burnside's Lemma** states that the number of orbits of a group G acting on a set X is:
//!
//! |X/G| = (1/|G|) × Σ_{g∈G} |X^g|
//!
//! where X^g is the set of elements fixed by g.
//!
//! # Bijection Discovery
//!
//! Two sets X and Y with group actions from G may have a natural bijection if:
//! 1. They have the same number of orbits under G
//! 2. Their orbit structures match (orbit sizes are equal after sorting)
//! 3. Fixed point counts match for each group element
//!
//! # Examples
//!
//! ```
//! use rustmath_combinatorics::bijectionist::*;
//! use rustmath_combinatorics::permutations::Permutation;
//!
//! // Action of S_3 on 3-element subsets of {1,2,3,4,5}
//! // Compare with action on other combinatorial objects
//! ```

use crate::permutations::Permutation;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// A group action of group G on set X.
///
/// This trait defines how a group element acts on elements of a set.
pub trait GroupAction<Element, GroupElement> {
    /// Apply the group element to transform the set element.
    fn act(&self, g: &GroupElement, x: &Element) -> Element;

    /// Check if an element is fixed by a group element.
    fn is_fixed(&self, g: &GroupElement, x: &Element) -> bool
    where
        Element: PartialEq,
    {
        self.act(g, x) == *x
    }

    /// Count the number of fixed points of a group element.
    fn count_fixed_points(&self, g: &GroupElement, elements: &[Element]) -> usize
    where
        Element: PartialEq,
    {
        elements.iter().filter(|x| self.is_fixed(g, x)).count()
    }

    /// Compute the orbit of an element under the group.
    fn orbit(&self, element: &Element, group: &[GroupElement]) -> Vec<Element>
    where
        Element: Clone + Eq + Hash,
    {
        let mut orbit = HashSet::new();
        let mut to_explore = vec![element.clone()];

        while let Some(current) = to_explore.pop() {
            if orbit.insert(current.clone()) {
                for g in group {
                    let transformed = self.act(g, &current);
                    if !orbit.contains(&transformed) {
                        to_explore.push(transformed);
                    }
                }
            }
        }

        orbit.into_iter().collect()
    }

    /// Compute all orbits partitioning the set.
    fn orbits(&self, elements: &[Element], group: &[GroupElement]) -> Vec<Vec<Element>>
    where
        Element: Clone + Eq + Hash,
    {
        let mut remaining: HashSet<Element> = elements.iter().cloned().collect();
        let mut orbits = Vec::new();

        while let Some(element) = remaining.iter().next().cloned() {
            let orbit = self.orbit(&element, group);
            for x in &orbit {
                remaining.remove(x);
            }
            orbits.push(orbit);
        }

        orbits
    }
}

/// Compute the number of orbits using Burnside's lemma.
///
/// Returns (number of orbits, detailed fixed point counts).
pub fn burnside_count<Element, GroupElement, Action>(
    action: &Action,
    elements: &[Element],
    group: &[GroupElement],
) -> (usize, Vec<usize>)
where
    Element: Clone + PartialEq,
    Action: GroupAction<Element, GroupElement>,
{
    if group.is_empty() {
        return (0, vec![]);
    }

    let fixed_counts: Vec<usize> = group
        .iter()
        .map(|g| action.count_fixed_points(g, elements))
        .collect();

    let total_fixed: usize = fixed_counts.iter().sum();
    let num_orbits = total_fixed / group.len();

    (num_orbits, fixed_counts)
}

/// Structure representing orbit decomposition information.
#[derive(Debug, Clone)]
pub struct OrbitStructure {
    /// Number of orbits
    pub num_orbits: usize,
    /// Sizes of each orbit (sorted)
    pub orbit_sizes: Vec<usize>,
    /// Fixed point counts for each group element
    pub fixed_counts: Vec<usize>,
    /// Total number of elements
    pub total_elements: usize,
}

impl OrbitStructure {
    /// Create orbit structure from explicit orbit computation.
    pub fn from_orbits<T>(orbits: &[Vec<T>], fixed_counts: Vec<usize>) -> Self {
        let mut orbit_sizes: Vec<usize> = orbits.iter().map(|o| o.len()).collect();
        orbit_sizes.sort_unstable();

        let total_elements = orbit_sizes.iter().sum();

        OrbitStructure {
            num_orbits: orbits.len(),
            orbit_sizes,
            fixed_counts,
            total_elements,
        }
    }

    /// Create orbit structure using Burnside's lemma (without explicit orbits).
    pub fn from_burnside(
        num_orbits: usize,
        total_elements: usize,
        fixed_counts: Vec<usize>,
    ) -> Self {
        OrbitStructure {
            num_orbits,
            orbit_sizes: vec![],
            fixed_counts,
            total_elements,
        }
    }

    /// Check if this structure is compatible with another for bijection.
    pub fn is_compatible_with(&self, other: &OrbitStructure) -> bool {
        // Must have same number of orbits
        if self.num_orbits != other.num_orbits {
            return false;
        }

        // Must have same total elements
        if self.total_elements != other.total_elements {
            return false;
        }

        // If orbit sizes are known, they must match
        if !self.orbit_sizes.is_empty() && !other.orbit_sizes.is_empty() {
            if self.orbit_sizes != other.orbit_sizes {
                return false;
            }
        }

        // Fixed point counts should match (strong condition)
        if self.fixed_counts.len() == other.fixed_counts.len() {
            if self.fixed_counts != other.fixed_counts {
                return false;
            }
        }

        true
    }

    /// Compute a compatibility score (0.0 = incompatible, 1.0 = perfect match).
    pub fn compatibility_score(&self, other: &OrbitStructure) -> f64 {
        let mut score = 0.0;
        let mut checks = 0.0;

        // Check orbit count (weight: 3.0)
        checks += 3.0;
        if self.num_orbits == other.num_orbits {
            score += 3.0;
        }

        // Check total elements (weight: 2.0)
        checks += 2.0;
        if self.total_elements == other.total_elements {
            score += 2.0;
        }

        // Check orbit sizes if available (weight: 3.0)
        if !self.orbit_sizes.is_empty() && !other.orbit_sizes.is_empty() {
            checks += 3.0;
            if self.orbit_sizes == other.orbit_sizes {
                score += 3.0;
            }
        }

        // Check fixed point counts (weight: 2.0)
        if self.fixed_counts.len() == other.fixed_counts.len() {
            checks += 2.0;
            if self.fixed_counts == other.fixed_counts {
                score += 2.0;
            } else {
                // Partial credit for partial match
                let matches = self
                    .fixed_counts
                    .iter()
                    .zip(&other.fixed_counts)
                    .filter(|(a, b)| a == b)
                    .count();
                score += 2.0 * (matches as f64 / self.fixed_counts.len() as f64);
            }
        }

        if checks > 0.0 {
            score / checks
        } else {
            0.0
        }
    }
}

/// A potential bijective correspondence between two sets.
#[derive(Debug)]
pub struct BijectiveCorrespondence<X, Y> {
    pub set_x_name: String,
    pub set_y_name: String,
    pub structure_x: OrbitStructure,
    pub structure_y: OrbitStructure,
    pub compatibility_score: f64,
    pub suggested_bijection: Option<HashMap<X, Y>>,
}

impl<X, Y> BijectiveCorrespondence<X, Y> {
    /// Check if this is a plausible bijection (score >= threshold).
    pub fn is_plausible(&self, threshold: f64) -> bool {
        self.compatibility_score >= threshold
    }
}

/// The main bijection finder that discovers potential correspondences.
pub struct BijectionFinder {
    /// Minimum compatibility score to report (0.0 to 1.0)
    pub min_score: f64,
}

impl BijectionFinder {
    /// Create a new bijection finder with default settings.
    pub fn new() -> Self {
        BijectionFinder { min_score: 0.8 }
    }

    /// Create a finder with a custom minimum score threshold.
    pub fn with_threshold(min_score: f64) -> Self {
        BijectionFinder { min_score }
    }

    /// Analyze if two group actions suggest a bijection.
    pub fn find_bijection<X, Y, G, ActionX, ActionY>(
        &self,
        name_x: &str,
        elements_x: &[X],
        action_x: &ActionX,
        name_y: &str,
        elements_y: &[Y],
        action_y: &ActionY,
        group: &[G],
    ) -> Option<BijectiveCorrespondence<X, Y>>
    where
        X: Clone + PartialEq + Eq + Hash,
        Y: Clone + PartialEq + Eq + Hash,
        ActionX: GroupAction<X, G>,
        ActionY: GroupAction<Y, G>,
    {
        // Compute orbit structures
        let orbits_x = action_x.orbits(elements_x, group);
        let (num_orbits_x, fixed_x) = burnside_count(action_x, elements_x, group);
        let structure_x = OrbitStructure::from_orbits(&orbits_x, fixed_x);

        let orbits_y = action_y.orbits(elements_y, group);
        let (num_orbits_y, fixed_y) = burnside_count(action_y, elements_y, group);
        let structure_y = OrbitStructure::from_orbits(&orbits_y, fixed_y);

        // Compute compatibility
        let score = structure_x.compatibility_score(&structure_y);

        if score < self.min_score {
            return None;
        }

        // Try to construct explicit bijection if orbit structures match
        let suggested_bijection = if structure_x.is_compatible_with(&structure_y) {
            self.construct_bijection(&orbits_x, &orbits_y)
        } else {
            None
        };

        Some(BijectiveCorrespondence {
            set_x_name: name_x.to_string(),
            set_y_name: name_y.to_string(),
            structure_x,
            structure_y,
            compatibility_score: score,
            suggested_bijection,
        })
    }

    /// Attempt to construct an explicit bijection by matching orbits.
    fn construct_bijection<X, Y>(
        &self,
        orbits_x: &[Vec<X>],
        orbits_y: &[Vec<Y>],
    ) -> Option<HashMap<X, Y>>
    where
        X: Clone + Eq + Hash,
        Y: Clone + Eq + Hash,
    {
        if orbits_x.len() != orbits_y.len() {
            return None;
        }

        // Sort orbits by size for matching
        let mut sorted_x: Vec<&Vec<X>> = orbits_x.iter().collect();
        let mut sorted_y: Vec<&Vec<Y>> = orbits_y.iter().collect();
        sorted_x.sort_by_key(|o| o.len());
        sorted_y.sort_by_key(|o| o.len());

        // Check if orbit sizes match
        for (ox, oy) in sorted_x.iter().zip(sorted_y.iter()) {
            if ox.len() != oy.len() {
                return None;
            }
        }

        // Construct bijection by pairing elements from matching orbits
        let mut bijection = HashMap::new();
        for (ox, oy) in sorted_x.iter().zip(sorted_y.iter()) {
            for (x, y) in ox.iter().zip(oy.iter()) {
                bijection.insert(x.clone(), y.clone());
            }
        }

        Some(bijection)
    }
}

impl Default for BijectionFinder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONCRETE IMPLEMENTATIONS FOR COMMON ACTIONS
// ============================================================================

/// Action of a permutation on a sequence by permuting indices.
pub struct PermutationActionOnSequence;

impl<T: Clone> GroupAction<Vec<T>, Permutation> for PermutationActionOnSequence {
    fn act(&self, perm: &Permutation, seq: &Vec<T>) -> Vec<T> {
        let n = seq.len();
        let mut result = vec![];

        for i in 0..n {
            // Apply permutation: result[i] = seq[perm(i)]
            if let Some(target_idx) = perm.apply(i) {
                if target_idx < n {
                    result.push(seq[target_idx].clone());
                }
            }
        }

        result
    }
}

/// Action of a permutation on a set (represented as sorted vector).
pub struct PermutationActionOnSet;

impl GroupAction<Vec<usize>, Permutation> for PermutationActionOnSet {
    fn act(&self, perm: &Permutation, set: &Vec<usize>) -> Vec<usize> {
        let mut result: Vec<usize> = set
            .iter()
            .filter_map(|&x| perm.apply(x))
            .collect();
        result.sort_unstable();
        result
    }
}

/// Action of a permutation on pairs.
pub struct PermutationActionOnPairs;

impl GroupAction<(usize, usize), Permutation> for PermutationActionOnPairs {
    fn act(&self, perm: &Permutation, pair: &(usize, usize)) -> (usize, usize) {
        let a = perm.apply(pair.0).unwrap_or(pair.0);
        let b = perm.apply(pair.1).unwrap_or(pair.1);
        if a <= b {
            (a, b)
        } else {
            (b, a)
        }
    }
}

/// Action on multisets (represented as sorted vectors).
pub struct PermutationActionOnMultiset;

impl<T: Clone + Ord> GroupAction<Vec<T>, Permutation> for PermutationActionOnMultiset {
    fn act(&self, perm: &Permutation, multiset: &Vec<T>) -> Vec<T> {
        let n = multiset.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            if let Some(target_idx) = perm.apply(i) {
                if target_idx < n {
                    result.push(multiset[target_idx].clone());
                }
            }
        }

        result.sort();
        result
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate all subsets of size k from n elements.
pub fn all_k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();
    generate_subsets(0, n, k, &mut current, &mut result);
    result
}

fn generate_subsets(
    start: usize,
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }

    for i in start..n {
        current.push(i);
        generate_subsets(i + 1, n, k, current, result);
        current.pop();
    }
}

/// Generate all ordered pairs from n elements.
pub fn all_ordered_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Generate all unordered pairs from n elements.
pub fn all_unordered_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burnside_trivial_action() {
        // Trivial action on 3 elements
        struct TrivialAction;
        impl GroupAction<usize, ()> for TrivialAction {
            fn act(&self, _g: &(), x: &usize) -> usize {
                *x
            }
        }

        let action = TrivialAction;
        let elements = vec![0, 1, 2];
        let group = vec![()];

        let (num_orbits, _) = burnside_count(&action, &elements, &group);
        assert_eq!(num_orbits, 3); // Each element in its own orbit
    }

    #[test]
    fn test_subset_action() {
        // S_3 acting on 2-subsets of {0,1,2}
        let subsets = all_k_subsets(3, 2); // [[0,1], [0,2], [1,2]]
        let s3 = vec![
            Permutation::identity(3),
            Permutation::from_vec(vec![1, 0, 2]).unwrap(), // (0 1)
            Permutation::from_vec(vec![0, 2, 1]).unwrap(), // (1 2)
            Permutation::from_vec(vec![2, 1, 0]).unwrap(), // (0 2)
            Permutation::from_vec(vec![1, 2, 0]).unwrap(), // (0 1 2)
            Permutation::from_vec(vec![2, 0, 1]).unwrap(), // (0 2 1)
        ];

        let action = PermutationActionOnSet;
        let (num_orbits, fixed) = burnside_count(&action, &subsets, &s3);

        // All 2-subsets are in the same orbit under S_3
        assert_eq!(num_orbits, 1);
        assert_eq!(fixed.len(), 6);
    }

    #[test]
    fn test_orbit_computation() {
        let action = PermutationActionOnSet;
        let subset = vec![0, 1];
        let s3 = vec![
            Permutation::identity(3),
            Permutation::from_vec(vec![1, 0, 2]).unwrap(),
            Permutation::from_vec(vec![0, 2, 1]).unwrap(),
        ];

        let orbit = action.orbit(&subset, &s3);
        assert!(orbit.len() > 0);
        assert!(orbit.contains(&vec![0, 1]));
    }

    #[test]
    fn test_orbits_partition() {
        let action = PermutationActionOnSet;
        let elements = vec![vec![0, 1], vec![0, 2], vec![1, 2]];
        let group = vec![Permutation::identity(3)];

        let orbits = action.orbits(&elements, &group);
        assert_eq!(orbits.len(), 3); // Identity keeps everything separate
    }

    #[test]
    fn test_orbit_structure_compatibility() {
        let struct1 = OrbitStructure {
            num_orbits: 2,
            orbit_sizes: vec![2, 3],
            fixed_counts: vec![5, 5],
            total_elements: 5,
        };

        let struct2 = OrbitStructure {
            num_orbits: 2,
            orbit_sizes: vec![2, 3],
            fixed_counts: vec![5, 5],
            total_elements: 5,
        };

        assert!(struct1.is_compatible_with(&struct2));
        assert_eq!(struct1.compatibility_score(&struct2), 1.0);
    }

    #[test]
    fn test_orbit_structure_incompatibility() {
        let struct1 = OrbitStructure {
            num_orbits: 2,
            orbit_sizes: vec![2, 3],
            fixed_counts: vec![5],
            total_elements: 5,
        };

        let struct2 = OrbitStructure {
            num_orbits: 3,
            orbit_sizes: vec![1, 2, 2],
            fixed_counts: vec![5],
            total_elements: 5,
        };

        assert!(!struct1.is_compatible_with(&struct2));
        assert!(struct1.compatibility_score(&struct2) < 1.0);
    }

    #[test]
    fn test_bijection_finder() {
        let finder = BijectionFinder::new();

        // S_2 acting on 2-element sequences vs 2-element sets
        let sequences = vec![vec![0, 1], vec![1, 0]];
        let sets = vec![vec![0, 1]];

        let s2 = vec![
            Permutation::identity(2),
            Permutation::from_vec(vec![1, 0]).unwrap(),
        ];

        let seq_action = PermutationActionOnSequence;
        let set_action = PermutationActionOnSet;

        let result = finder.find_bijection(
            "sequences",
            &sequences,
            &seq_action,
            "sets",
            &sets,
            &set_action,
            &s2,
        );

        // These should NOT match (different number of elements)
        assert!(result.is_none() || !result.unwrap().is_plausible(0.8));
    }

    #[test]
    fn test_k_subsets_generation() {
        let subsets = all_k_subsets(4, 2);
        assert_eq!(subsets.len(), 6); // C(4,2) = 6

        let subsets = all_k_subsets(3, 3);
        assert_eq!(subsets.len(), 1); // C(3,3) = 1
        assert_eq!(subsets[0], vec![0, 1, 2]);

        let subsets = all_k_subsets(3, 0);
        assert_eq!(subsets.len(), 1); // C(3,0) = 1
        assert_eq!(subsets[0], vec![]);
    }

    #[test]
    fn test_pairs_generation() {
        let pairs = all_unordered_pairs(3);
        assert_eq!(pairs.len(), 3); // C(3,2) = 3
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(0, 2)));
        assert!(pairs.contains(&(1, 2)));

        let pairs = all_ordered_pairs(3);
        assert_eq!(pairs.len(), 6); // 3 * 2 = 6
    }

    #[test]
    fn test_permutation_action_on_pairs() {
        let action = PermutationActionOnPairs;
        let perm = Permutation::from_vec(vec![1, 0, 2]).unwrap(); // Swap 0 and 1
        let pair = (0, 1);

        let result = action.act(&perm, &pair);
        assert_eq!(result, (0, 1)); // Normalized to (min, max)
    }

    #[test]
    fn test_fixed_points_count() {
        let action = PermutationActionOnSet;
        let identity = Permutation::identity(3);
        let elements = vec![vec![0, 1], vec![0, 2], vec![1, 2]];

        let count = action.count_fixed_points(&identity, &elements);
        assert_eq!(count, 3); // Identity fixes all elements
    }

    #[test]
    fn test_classic_bijection_example() {
        // Test orbit counting with different subset sizes

        let n = 3;

        // For n=3, we have 3-subsets of 6 elements
        let subsets_3 = all_k_subsets(2 * n, n);
        let group_6 = vec![Permutation::identity(2 * n)]; // Identity on 6 elements

        // For comparison, 2-subsets of 3 elements
        let subsets_2 = all_k_subsets(n, 2);
        let group_3 = vec![Permutation::identity(n)]; // Identity on 3 elements

        // These have different sizes, so won't bijectively correspond
        // but we can still analyze their orbit structures

        let action = PermutationActionOnSet;
        let (orbits_3, _) = burnside_count(&action, &subsets_3, &group_6);
        let (orbits_2, _) = burnside_count(&action, &subsets_2, &group_3);

        // With trivial group, each subset is its own orbit
        assert_eq!(orbits_3, subsets_3.len());
        assert_eq!(orbits_2, subsets_2.len());
    }
}

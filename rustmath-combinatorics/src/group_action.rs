//! Group actions, Burnside's lemma, and Pólya enumeration theorem
//!
//! This module provides tools for counting objects under group actions using:
//! - Burnside's lemma (Cauchy-Frobenius lemma): counts distinct colorings
//! - Pólya enumeration theorem: counts colorings by weight using cycle indices
//!
//! # Burnside's Lemma
//!
//! Given a group G acting on a set X, the number of distinct orbits is:
//! |X/G| = (1/|G|) * Σ_{g∈G} |Fix(g)|
//!
//! where Fix(g) is the number of elements of X fixed by permutation g.
//!
//! # Pólya Enumeration Theorem
//!
//! The cycle index of a permutation group G is:
//! Z(G; x₁, x₂, ..., xₙ) = (1/|G|) * Σ_{g∈G} ∏_{i=1}^{n} xᵢ^{cᵢ(g)}
//!
//! where cᵢ(g) is the number of cycles of length i in permutation g.
//!
//! # Examples
//!
//! ```rust,ignore
//! use rustmath_combinatorics::group_action::*;
//! use rustmath_combinatorics::Permutation;
//!
//! // Count distinct colorings of a square with 2 colors under rotation
//! let rotations = vec![
//!     Permutation::identity(4),  // no rotation
//!     Permutation::from_vec(vec![1, 2, 3, 0]).unwrap(),  // 90° rotation
//!     Permutation::from_vec(vec![2, 3, 0, 1]).unwrap(),  // 180° rotation
//!     Permutation::from_vec(vec![3, 0, 1, 2]).unwrap(),  // 270° rotation
//! ];
//!
//! let group = PermutationGroup::new(rotations);
//! let count = burnside_lemma(&group, 2);  // 2 colors
//! assert_eq!(count, 6);  // 6 distinct colorings
//! ```

use crate::permutations::Permutation;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// A permutation group represented by its elements
#[derive(Debug, Clone)]
pub struct PermutationGroup {
    /// The permutations in the group
    elements: Vec<Permutation>,
    /// The degree (number of points acted upon)
    degree: usize,
}

impl PermutationGroup {
    /// Create a new permutation group from a list of permutations
    pub fn new(elements: Vec<Permutation>) -> Self {
        let degree = elements.first().map(|p| p.size()).unwrap_or(0);

        // Verify all permutations have the same size
        for perm in &elements {
            assert_eq!(perm.size(), degree, "All permutations must have the same size");
        }

        PermutationGroup { elements, degree }
    }

    /// Create the symmetric group S_n
    pub fn symmetric(n: usize) -> Self {
        use crate::permutations::all_permutations;
        PermutationGroup::new(all_permutations(n))
    }

    /// Create the cyclic group Z_n (rotations)
    pub fn cyclic(n: usize) -> Self {
        let mut elements = Vec::new();

        // Generate rotations
        for k in 0..n {
            let perm: Vec<usize> = (0..n).map(|i| (i + k) % n).collect();
            elements.push(Permutation::from_vec(perm).unwrap());
        }

        PermutationGroup::new(elements)
    }

    /// Create the dihedral group D_n (rotations and reflections)
    pub fn dihedral(n: usize) -> Self {
        let mut elements = Vec::new();

        // Generate rotations
        for k in 0..n {
            let perm: Vec<usize> = (0..n).map(|i| (i + k) % n).collect();
            elements.push(Permutation::from_vec(perm).unwrap());
        }

        // Generate reflections
        for k in 0..n {
            let perm: Vec<usize> = (0..n)
                .map(|i| {
                    // Reflect across axis through vertex k
                    let reflected = (2 * k + n - i) % n;
                    reflected
                })
                .collect();
            elements.push(Permutation::from_vec(perm).unwrap());
        }

        PermutationGroup::new(elements)
    }

    /// Get the order (number of elements) of the group
    pub fn order(&self) -> usize {
        self.elements.len()
    }

    /// Get the degree (number of points acted upon)
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the elements of the group
    pub fn elements(&self) -> &[Permutation] {
        &self.elements
    }

    /// Compute the cycle type of a permutation
    ///
    /// Returns a map from cycle length to count
    pub fn cycle_type(perm: &Permutation) -> HashMap<usize, usize> {
        let cycles = perm.cycles();
        let mut cycle_type = HashMap::new();

        // Count cycles by length
        for cycle in &cycles {
            *cycle_type.entry(cycle.len()).or_insert(0) += 1;
        }

        // Add fixed points (1-cycles)
        let total_in_cycles: usize = cycles.iter().map(|c| c.len()).sum();
        let fixed_points = perm.size() - total_in_cycles;
        if fixed_points > 0 {
            *cycle_type.entry(1).or_insert(0) += fixed_points;
        }

        cycle_type
    }
}

/// Apply Burnside's lemma to count distinct colorings
///
/// Given a permutation group acting on n positions and k colors,
/// this counts the number of distinct colorings under the group action.
///
/// # Arguments
///
/// * `group` - The permutation group representing the symmetries
/// * `num_colors` - The number of colors available
///
/// # Returns
///
/// The number of distinct colorings
///
/// # Example
///
/// ```rust,ignore
/// // Count colorings of a square with 2 colors under rotations
/// let rotations = PermutationGroup::cyclic(4);
/// let count = burnside_lemma(&rotations, 2);
/// assert_eq!(count, 6);
/// ```
pub fn burnside_lemma(group: &PermutationGroup, num_colors: usize) -> Integer {
    let mut sum = Integer::zero();

    for perm in group.elements() {
        // Count the number of cycles (including fixed points)
        let cycle_type = PermutationGroup::cycle_type(perm);
        let num_cycles: usize = cycle_type.values().sum();

        // Each cycle can be colored independently
        // Number of fixed colorings is num_colors^(number of cycles)
        let fixed_count = Integer::from(num_colors as u32).pow(num_cycles as u32);
        sum = sum + fixed_count;
    }

    // Divide by the order of the group
    sum / Integer::from(group.order() as u32)
}

/// Compute the cycle index polynomial of a permutation group
///
/// The cycle index is represented as a sum of monomials, where each
/// monomial corresponds to a permutation and its cycle structure.
///
/// Returns a vector of cycle type counts, where each entry is a map
/// from cycle length to count, along with the coefficient (count of
/// permutations with that cycle type).
///
/// # Example
///
/// ```rust,ignore
/// let group = PermutationGroup::cyclic(4);
/// let ci = cycle_index(&group);
/// ```
pub fn cycle_index(group: &PermutationGroup) -> Vec<(HashMap<usize, usize>, usize)> {
    let mut cycle_types: HashMap<Vec<(usize, usize)>, usize> = HashMap::new();

    // Count how many permutations have each cycle type
    for perm in group.elements() {
        let cycle_type = PermutationGroup::cycle_type(perm);

        // Convert to sorted vector for use as hash key
        let mut ct_vec: Vec<(usize, usize)> = cycle_type.iter()
            .map(|(&len, &count)| (len, count))
            .collect();
        ct_vec.sort();

        *cycle_types.entry(ct_vec).or_insert(0) += 1;
    }

    // Convert back to the result format
    cycle_types.into_iter()
        .map(|(ct_vec, count)| {
            let ct_map: HashMap<usize, usize> = ct_vec.into_iter().collect();
            (ct_map, count)
        })
        .collect()
}

/// Apply Pólya enumeration theorem with weighted colors
///
/// This is a generalization of Burnside's lemma that counts colorings
/// where each color has a weight.
///
/// # Arguments
///
/// * `group` - The permutation group representing the symmetries
/// * `color_weights` - The weights of each color (as integers)
///
/// # Returns
///
/// A polynomial (represented as coefficients) counting colorings by total weight
///
/// For example, with 2 colors of weight 1 each, this gives the same result
/// as Burnside's lemma.
///
/// # Example
///
/// ```rust,ignore
/// let group = PermutationGroup::cyclic(3);
/// // Two colors, each with weight 1
/// let weights = vec![Integer::one(), Integer::one()];
/// let result = polya_enumeration(&group, &weights);
/// ```
pub fn polya_enumeration(
    group: &PermutationGroup,
    color_weights: &[Integer],
) -> HashMap<usize, Rational> {
    let mut result: HashMap<usize, Rational> = HashMap::new();

    let group_order = Rational::from_integer(Integer::from(group.order() as u32));

    for perm in group.elements() {
        let cycle_type = PermutationGroup::cycle_type(perm);

        // For each cycle of length k, we can assign any color to it,
        // and all k elements in the cycle get that color.
        // So the contribution is Σ(wᵢᵏ) for colors i

        // We need to multiply contributions from all cycles
        // Start with polynomial "1"
        let mut perm_contribution: HashMap<usize, Rational> = HashMap::new();
        perm_contribution.insert(0, Rational::one());

        for (&cycle_len, &cycle_count) in &cycle_type {
            // For a cycle of length cycle_len, each color i contributes weight wᵢ^cycle_len
            let mut cycle_poly: HashMap<usize, Rational> = HashMap::new();

            for weight in color_weights {
                // Weight^cycle_len
                let w = weight.pow(cycle_len as u32);
                let weight_val = w.to_usize().unwrap_or(0);
                let current = cycle_poly.entry(weight_val).or_insert(Rational::zero()).clone();
                cycle_poly.insert(weight_val, current + Rational::one());
            }

            // Multiply this contribution cycle_count times
            for _ in 0..cycle_count {
                perm_contribution = multiply_polynomials(&perm_contribution, &cycle_poly);
            }
        }

        // Add this permutation's contribution to the result
        for (weight, coeff) in perm_contribution {
            let current = result.entry(weight).or_insert(Rational::zero()).clone();
            result.insert(weight, current + coeff);
        }
    }

    // Divide by |G|
    for (weight, coeff) in result.clone() {
        result.insert(weight, coeff / group_order.clone());
    }

    result
}

/// Multiply two polynomials represented as maps from exponent to coefficient
fn multiply_polynomials(
    p1: &HashMap<usize, Rational>,
    p2: &HashMap<usize, Rational>,
) -> HashMap<usize, Rational> {
    let mut result = HashMap::new();

    for (&exp1, coeff1) in p1 {
        for (&exp2, coeff2) in p2 {
            let exp = exp1 + exp2;
            let coeff = coeff1.clone() * coeff2.clone();
            let current = result.entry(exp).or_insert(Rational::zero()).clone();
            result.insert(exp, current + coeff);
        }
    }

    result
}

/// Count the number of elements fixed by a permutation under k-coloring
///
/// This is a helper function that computes |Fix(g)| for Burnside's lemma.
///
/// # Arguments
///
/// * `perm` - The permutation
/// * `num_colors` - The number of colors
///
/// # Returns
///
/// The number of colorings fixed by the permutation
pub fn count_fixed_colorings(perm: &Permutation, num_colors: usize) -> Integer {
    let cycle_type = PermutationGroup::cycle_type(perm);
    let num_cycles: usize = cycle_type.values().sum();
    Integer::from(num_colors as u32).pow(num_cycles as u32)
}

/// Enumerate all distinct colorings under a group action
///
/// This generates all distinct colorings (one representative from each orbit).
///
/// # Arguments
///
/// * `group` - The permutation group
/// * `num_colors` - The number of colors (0 to num_colors-1)
///
/// # Returns
///
/// A vector of colorings, where each coloring is a vector of color indices
///
/// # Note
///
/// This can be expensive for large groups or many colors. Use `burnside_lemma`
/// to just count without enumerating.
pub fn enumerate_distinct_colorings(
    group: &PermutationGroup,
    num_colors: usize,
) -> Vec<Vec<usize>> {
    let n = group.degree();
    let mut orbits: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    // Generate all possible colorings
    enumerate_colorings_helper(n, num_colors, &mut vec![], &mut |coloring| {
        // Check if we've seen this coloring or any in its orbit
        if !seen.contains(coloring) {
            // This is a new orbit - add it
            orbits.push(coloring.clone());

            // Mark all elements in this orbit as seen
            for perm in group.elements() {
                let transformed = apply_permutation_to_coloring(coloring, perm);
                seen.insert(transformed);
            }
        }
    });

    orbits
}

/// Helper function to enumerate all colorings recursively
fn enumerate_colorings_helper<F>(
    n: usize,
    num_colors: usize,
    current: &mut Vec<usize>,
    callback: &mut F,
) where
    F: FnMut(&Vec<usize>),
{
    if current.len() == n {
        callback(current);
        return;
    }

    for color in 0..num_colors {
        current.push(color);
        enumerate_colorings_helper(n, num_colors, current, callback);
        current.pop();
    }
}

/// Apply a permutation to a coloring
fn apply_permutation_to_coloring(coloring: &[usize], perm: &Permutation) -> Vec<usize> {
    let mut result = vec![0; coloring.len()];
    for i in 0..coloring.len() {
        if let Some(j) = perm.apply(i) {
            result[j] = coloring[i];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_type() {
        // Identity permutation: all fixed points
        let id = Permutation::identity(4);
        let ct = PermutationGroup::cycle_type(&id);
        assert_eq!(ct.get(&1), Some(&4)); // 4 fixed points

        // Single 3-cycle and 1 fixed point
        let perm = Permutation::from_vec(vec![1, 2, 0, 3]).unwrap();
        let ct = PermutationGroup::cycle_type(&perm);
        assert_eq!(ct.get(&3), Some(&1)); // one 3-cycle
        assert_eq!(ct.get(&1), Some(&1)); // one fixed point

        // Two 2-cycles
        let perm = Permutation::from_vec(vec![1, 0, 3, 2]).unwrap();
        let ct = PermutationGroup::cycle_type(&perm);
        assert_eq!(ct.get(&2), Some(&2)); // two 2-cycles
    }

    #[test]
    fn test_cyclic_group() {
        let c3 = PermutationGroup::cyclic(3);
        assert_eq!(c3.order(), 3);
        assert_eq!(c3.degree(), 3);
    }

    #[test]
    fn test_dihedral_group() {
        let d3 = PermutationGroup::dihedral(3);
        assert_eq!(d3.order(), 6); // D_3 has 6 elements (3 rotations + 3 reflections)
        assert_eq!(d3.degree(), 3);
    }

    #[test]
    fn test_burnside_square_rotations() {
        // Square with 4 vertices under rotation group (cyclic group of order 4)
        let rotations = PermutationGroup::cyclic(4);

        // With 2 colors: should have 6 distinct colorings
        let count = burnside_lemma(&rotations, 2);
        assert_eq!(count, Integer::from(6));
    }

    #[test]
    fn test_burnside_triangle_rotations() {
        // Equilateral triangle under rotation group (cyclic group of order 3)
        let rotations = PermutationGroup::cyclic(3);

        // With 2 colors
        // - Identity fixes all 2^3 = 8 colorings
        // - Each 120° rotation fixes colorings where all vertices have same color: 2
        // Total: (8 + 2 + 2) / 3 = 12 / 3 = 4
        let count = burnside_lemma(&rotations, 2);
        assert_eq!(count, Integer::from(4));
    }

    #[test]
    fn test_burnside_triangle_full_symmetry() {
        // Equilateral triangle under full symmetry group (dihedral group D_3)
        let d3 = PermutationGroup::dihedral(3);

        // With 2 colors
        // This should give fewer distinct colorings than just rotations
        let count = burnside_lemma(&d3, 2);
        assert_eq!(count, Integer::from(4));
    }

    #[test]
    fn test_burnside_necklace() {
        // Necklace problem: circular arrangement of n beads with k colors
        // For n=4, k=2 (4 beads, 2 colors, rotations only)
        let rotations = PermutationGroup::cyclic(4);
        let count = burnside_lemma(&rotations, 2);
        assert_eq!(count, Integer::from(6));
    }

    #[test]
    fn test_cycle_index_c3() {
        let c3 = PermutationGroup::cyclic(3);
        let ci = cycle_index(&c3);

        // C_3 has:
        // - 1 identity (3 fixed points)
        // - 2 rotations (each is a single 3-cycle)
        assert_eq!(ci.len(), 2);

        // Check that we have the right cycle types
        let has_identity = ci.iter().any(|(ct, count)| {
            ct.get(&1) == Some(&3) && *count == 1
        });
        assert!(has_identity);

        let has_rotations = ci.iter().any(|(ct, count)| {
            ct.get(&3) == Some(&1) && *count == 2
        });
        assert!(has_rotations);
    }

    #[test]
    fn test_count_fixed_colorings() {
        // Identity fixes all colorings
        let id = Permutation::identity(3);
        let fixed = count_fixed_colorings(&id, 2);
        assert_eq!(fixed, Integer::from(8)); // 2^3

        // A 3-cycle fixes only uniform colorings
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        let fixed = count_fixed_colorings(&perm, 2);
        assert_eq!(fixed, Integer::from(2)); // 2 colors
    }

    #[test]
    fn test_enumerate_distinct_colorings_small() {
        // Two positions with 2 colors and trivial group (identity only)
        let trivial = PermutationGroup::new(vec![Permutation::identity(2)]);
        let colorings = enumerate_distinct_colorings(&trivial, 2);
        assert_eq!(colorings.len(), 4); // All 4 colorings are distinct

        // Two positions with 2 colors and swap symmetry
        let swap = PermutationGroup::new(vec![
            Permutation::identity(2),
            Permutation::from_vec(vec![1, 0]).unwrap(),
        ]);
        let colorings = enumerate_distinct_colorings(&swap, 2);
        assert_eq!(colorings.len(), 3); // (0,0), (1,1), and (0,1)~(1,0)
    }

    #[test]
    fn test_polya_enumeration_simple() {
        // Two positions with swap symmetry, each color weight 1
        let swap = PermutationGroup::new(vec![
            Permutation::identity(2),
            Permutation::from_vec(vec![1, 0]).unwrap(),
        ]);

        let weights = vec![Integer::one(), Integer::one()];
        let result = polya_enumeration(&swap, &weights);

        // Total count (sum of all coefficients) should equal Burnside count
        let total: Rational = result.values().fold(Rational::zero(), |acc, x| acc + x.clone());
        assert_eq!(total, Rational::from_integer(Integer::from(3)));
    }
}

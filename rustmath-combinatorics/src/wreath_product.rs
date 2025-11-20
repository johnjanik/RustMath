//! Wreath products and colored permutations
//!
//! This module implements colored permutations, which are elements of the wreath product
//! Z_r ≀ S_n (the wreath product of the cyclic group Z_r with the symmetric group S_n).
//!
//! A colored permutation consists of:
//! - A base permutation σ ∈ S_n
//! - A coloring function c: {0,...,n-1} → {0,...,r-1} assigning a color to each position
//!
//! The group operation is: (c1, σ1) * (c2, σ2) = (c3, σ1 ∘ σ2)
//! where c3[i] = (c1[i] + c2[σ1^{-1}[i]]) mod r

use crate::permutations::Permutation;
use std::collections::HashMap;

/// A colored permutation representing an element of Z_r ≀ S_n
///
/// The wreath product Z_r ≀ S_n consists of permutations where each element
/// has an associated "color" from the cyclic group Z_r = {0, 1, ..., r-1}.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColoredPermutation {
    /// The base permutation σ ∈ S_n
    permutation: Permutation,
    /// Colors assigned to each position: colors[i] ∈ {0, ..., r-1}
    colors: Vec<usize>,
    /// The modulus r for the cyclic group Z_r
    r: usize,
}

impl ColoredPermutation {
    /// Create a colored permutation from a permutation and color vector
    ///
    /// Returns None if:
    /// - The permutation and colors have different lengths
    /// - Any color is >= r
    /// - r is 0
    pub fn new(permutation: Permutation, colors: Vec<usize>, r: usize) -> Option<Self> {
        if r == 0 {
            return None;
        }

        if permutation.size() != colors.len() {
            return None;
        }

        // Validate all colors are in {0, ..., r-1}
        if colors.iter().any(|&c| c >= r) {
            return None;
        }

        Some(ColoredPermutation {
            permutation,
            colors,
            r,
        })
    }

    /// Create the identity colored permutation in Z_r ≀ S_n
    ///
    /// All positions map to themselves with color 0
    pub fn identity(n: usize, r: usize) -> Option<Self> {
        if r == 0 {
            return None;
        }

        Some(ColoredPermutation {
            permutation: Permutation::identity(n),
            colors: vec![0; n],
            r,
        })
    }

    /// Get the size n of the permutation (for Z_r ≀ S_n)
    pub fn size(&self) -> usize {
        self.permutation.size()
    }

    /// Get the modulus r (for Z_r ≀ S_n)
    pub fn modulus(&self) -> usize {
        self.r
    }

    /// Get the underlying permutation
    pub fn permutation(&self) -> &Permutation {
        &self.permutation
    }

    /// Get the color vector
    pub fn colors(&self) -> &[usize] {
        &self.colors
    }

    /// Get the color at a specific position
    pub fn color_at(&self, i: usize) -> Option<usize> {
        self.colors.get(i).copied()
    }

    /// Compose two colored permutations: self * other
    ///
    /// The composition is defined as:
    /// (c1, σ1) * (c2, σ2) = (c3, σ1 ∘ σ2)
    /// where c3[i] = (c1[i] + c2[σ1^{-1}[i]]) mod r
    ///
    /// Returns None if the permutations have incompatible sizes or moduli
    pub fn compose(&self, other: &ColoredPermutation) -> Option<Self> {
        if self.size() != other.size() || self.r != other.r {
            return None;
        }

        // Compose the underlying permutations
        let composed_perm = self.permutation.compose(&other.permutation)?;

        // Compute the new color vector
        // c3[i] = (c1[i] + c2[σ1^{-1}[i]]) mod r
        let inv_perm = self.permutation.inverse();
        let mut new_colors = Vec::with_capacity(self.size());

        for i in 0..self.size() {
            let inv_i = inv_perm.apply(i).unwrap();
            let new_color = (self.colors[i] + other.colors[inv_i]) % self.r;
            new_colors.push(new_color);
        }

        Some(ColoredPermutation {
            permutation: composed_perm,
            colors: new_colors,
            r: self.r,
        })
    }

    /// Compute the inverse of this colored permutation
    ///
    /// For (c, σ), the inverse is (c', σ^{-1}) where
    /// c'[i] = -c[σ[i]] mod r
    pub fn inverse(&self) -> Self {
        let inv_perm = self.permutation.inverse();
        let mut inv_colors = vec![0; self.size()];

        for i in 0..self.size() {
            let sigma_i = self.permutation.apply(i).unwrap();
            // c'[i] = -c[σ[i]] mod r = (r - c[σ[i]]) mod r
            inv_colors[i] = if self.colors[sigma_i] == 0 {
                0
            } else {
                self.r - self.colors[sigma_i]
            };
        }

        ColoredPermutation {
            permutation: inv_perm,
            colors: inv_colors,
            r: self.r,
        }
    }

    /// Compute the order of this colored permutation
    ///
    /// Returns the smallest positive integer k such that self^k = identity
    pub fn order(&self) -> usize {
        let mut current = self.clone();
        let identity = ColoredPermutation::identity(self.size(), self.r).unwrap();
        let mut order = 1;

        while current != identity {
            current = current.compose(&self).unwrap();
            order += 1;

            // Safety check to prevent infinite loops
            if order > 1_000_000 {
                return order;
            }
        }

        order
    }

    /// Compute the colored cycle decomposition
    ///
    /// Returns a vector of colored cycles. Each cycle is represented as:
    /// - A vector of positions forming the cycle
    /// - The sum of colors along the cycle (mod r)
    ///
    /// For example, if we have a cycle (0 1 2) with colors [1, 2, 3],
    /// the colored cycle is ([0, 1, 2], (1+2+3) mod r)
    pub fn colored_cycles(&self) -> Vec<(Vec<usize>, usize)> {
        let n = self.size();
        let mut visited = vec![false; n];
        let mut colored_cycles = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }

            let mut cycle = vec![start];
            let mut color_sum = self.colors[start];
            visited[start] = true;

            let mut current = self.permutation.apply(start).unwrap();

            while current != start {
                cycle.push(current);
                color_sum = (color_sum + self.colors[current]) % self.r;
                visited[current] = true;
                current = self.permutation.apply(current).unwrap();
            }

            // Include all cycles (even fixed points with color 0)
            // to maintain full information about the colored permutation
            colored_cycles.push((cycle, color_sum));
        }

        colored_cycles
    }

    /// Compute the cycle type of this colored permutation
    ///
    /// The cycle type is a multiset of (length, color_sum) pairs,
    /// represented as a HashMap where:
    /// - Key: (cycle_length, color_sum mod r)
    /// - Value: number of cycles with that (length, color_sum) signature
    ///
    /// This is a complete invariant for conjugacy classes in Z_r ≀ S_n
    pub fn cycle_type(&self) -> HashMap<(usize, usize), usize> {
        let colored_cycles = self.colored_cycles();
        let mut cycle_type = HashMap::new();

        for (cycle, color_sum) in colored_cycles {
            let signature = (cycle.len(), color_sum);
            *cycle_type.entry(signature).or_insert(0) += 1;
        }

        cycle_type
    }

    /// Check if two colored permutations are conjugate
    ///
    /// Two elements are conjugate in Z_r ≀ S_n if and only if
    /// they have the same cycle type
    pub fn is_conjugate_to(&self, other: &ColoredPermutation) -> bool {
        if self.size() != other.size() || self.r != other.r {
            return false;
        }

        self.cycle_type() == other.cycle_type()
    }

    /// Convert to a readable string representation showing cycles with colors
    ///
    /// Format: each cycle is shown as (i1^c1 i2^c2 ... ik^ck)
    /// where i_j are positions and c_j are colors
    pub fn to_colored_cycle_string(&self) -> String {
        let colored_cycles = self.colored_cycles();

        if colored_cycles.is_empty() {
            return "()".to_string();
        }

        let mut parts = Vec::new();

        for (cycle, _color_sum) in colored_cycles {
            if cycle.len() == 1 && self.colors[cycle[0]] == 0 {
                // Skip fixed points with color 0 for cleaner display
                continue;
            }

            let cycle_str = cycle
                .iter()
                .map(|&i| {
                    if self.colors[i] == 0 {
                        format!("{}", i)
                    } else {
                        format!("{}^{}", i, self.colors[i])
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");

            parts.push(format!("({})", cycle_str));
        }

        if parts.is_empty() {
            "()".to_string()
        } else {
            parts.join("")
        }
    }
}

/// Generate all colored permutations in Z_r ≀ S_n
///
/// This generates all r^n * n! elements of the wreath product
///
/// Warning: The size grows very quickly! For n=3, r=2: 48 elements
/// For n=4, r=2: 384 elements
pub fn all_colored_permutations(n: usize, r: usize) -> Option<Vec<ColoredPermutation>> {
    if r == 0 {
        return None;
    }

    // Generate all permutations
    let perms = crate::permutations::all_permutations(n);

    // Generate all color vectors
    let mut all_colors = Vec::new();
    generate_all_colorings(n, r, &mut vec![], &mut all_colors);

    let mut result = Vec::new();

    for perm in perms {
        for colors in &all_colors {
            if let Some(cp) = ColoredPermutation::new(perm.clone(), colors.clone(), r) {
                result.push(cp);
            }
        }
    }

    Some(result)
}

/// Helper function to generate all possible colorings
fn generate_all_colorings(
    n: usize,
    r: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == n {
        result.push(current.clone());
        return;
    }

    for color in 0..r {
        current.push(color);
        generate_all_colorings(n, r, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = ColoredPermutation::identity(3, 2).unwrap();
        assert_eq!(id.size(), 3);
        assert_eq!(id.modulus(), 2);
        assert_eq!(id.colors(), &[0, 0, 0]);
    }

    #[test]
    fn test_new_validation() {
        let perm = Permutation::identity(3);

        // Valid creation
        assert!(ColoredPermutation::new(perm.clone(), vec![0, 1, 0], 2).is_some());

        // Invalid: color >= r
        assert!(ColoredPermutation::new(perm.clone(), vec![0, 2, 0], 2).is_none());

        // Invalid: r = 0
        assert!(ColoredPermutation::new(perm.clone(), vec![0, 0, 0], 0).is_none());

        // Invalid: mismatched sizes
        assert!(ColoredPermutation::new(perm, vec![0, 1], 2).is_none());
    }

    #[test]
    fn test_compose() {
        // Create two simple colored permutations in Z_2 ≀ S_3
        let perm1 = Permutation::from_vec(vec![1, 0, 2]).unwrap(); // Swap 0 and 1
        let cp1 = ColoredPermutation::new(perm1, vec![1, 0, 0], 2).unwrap();

        let perm2 = Permutation::from_vec(vec![0, 2, 1]).unwrap(); // Swap 1 and 2
        let cp2 = ColoredPermutation::new(perm2, vec![0, 1, 0], 2).unwrap();

        let result = cp1.compose(&cp2).unwrap();

        // Verify the result
        assert_eq!(result.size(), 3);
        assert_eq!(result.modulus(), 2);
    }

    #[test]
    fn test_inverse() {
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap(); // 3-cycle
        let cp = ColoredPermutation::new(perm, vec![1, 2, 1], 3).unwrap();

        let inv = cp.inverse();
        let product = cp.compose(&inv).unwrap();
        let identity = ColoredPermutation::identity(3, 3).unwrap();

        assert_eq!(product, identity);
    }

    #[test]
    fn test_colored_cycles() {
        // Create a colored permutation with a 3-cycle
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap(); // (0 1 2)
        let cp = ColoredPermutation::new(perm, vec![1, 2, 3], 5).unwrap();

        let colored_cycles = cp.colored_cycles();

        // Should have one cycle
        assert_eq!(colored_cycles.len(), 1);
        assert_eq!(colored_cycles[0].0, vec![0, 1, 2]);
        assert_eq!(colored_cycles[0].1, (1 + 2 + 3) % 5); // Color sum = 6 mod 5 = 1
    }

    #[test]
    fn test_cycle_type() {
        // Create a colored permutation: (0 1) with colors [1, 1], fixed point 2 with color 0
        let perm = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        let cp = ColoredPermutation::new(perm, vec![1, 1, 0], 2).unwrap();

        let cycle_type = cp.cycle_type();

        // Should have one 2-cycle with color sum (1+1) mod 2 = 0
        // and one 1-cycle (fixed point) with color sum 0
        assert_eq!(cycle_type.get(&(2, 0)), Some(&1)); // One 2-cycle with color sum 0
        assert_eq!(cycle_type.get(&(1, 0)), Some(&1)); // One fixed point with color 0
    }

    #[test]
    fn test_conjugacy() {
        // Two permutations with the same cycle type should be conjugate
        let perm1 = Permutation::from_vec(vec![1, 0, 2]).unwrap(); // (0 1)
        let cp1 = ColoredPermutation::new(perm1, vec![1, 1, 0], 2).unwrap();

        let perm2 = Permutation::from_vec(vec![0, 2, 1]).unwrap(); // (1 2)
        let cp2 = ColoredPermutation::new(perm2, vec![0, 1, 1], 2).unwrap();

        assert!(cp1.is_conjugate_to(&cp2));
    }

    #[test]
    fn test_order() {
        // A 3-cycle with colors [1, 1, 1] in Z_3
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        let cp = ColoredPermutation::new(perm, vec![1, 1, 1], 3).unwrap();

        // The order should be 3 (since color sum is 3 ≡ 0 mod 3)
        assert_eq!(cp.order(), 3);
    }

    #[test]
    fn test_order_identity() {
        let id = ColoredPermutation::identity(4, 2).unwrap();
        assert_eq!(id.order(), 1);
    }

    #[test]
    fn test_to_colored_cycle_string() {
        let perm = Permutation::from_vec(vec![1, 2, 0, 3]).unwrap();
        let cp = ColoredPermutation::new(perm, vec![1, 0, 2, 0], 3).unwrap();

        let cycle_str = cp.to_colored_cycle_string();

        // Should show the 3-cycle with colors
        // Format: (0^1 1 2^2) where positions with color 0 don't show the exponent
        assert!(cycle_str.contains("0^1"));
        assert!(cycle_str.contains("2^2"));
    }

    #[test]
    fn test_all_colored_permutations_small() {
        // Generate all elements of Z_2 ≀ S_2
        let all = all_colored_permutations(2, 2).unwrap();

        // Should have 2^2 * 2! = 4 * 2 = 8 elements
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_cycle_type_invariant() {
        // The cycle type should be preserved under conjugation
        let perm = Permutation::from_vec(vec![1, 0, 3, 2]).unwrap(); // (0 1)(2 3)
        let cp = ColoredPermutation::new(perm, vec![1, 1, 0, 0], 2).unwrap();

        let cycle_type = cp.cycle_type();

        // Should have two 2-cycles
        // One with color sum (1+1) mod 2 = 0
        // One with color sum (0+0) mod 2 = 0
        assert_eq!(cycle_type.get(&(2, 0)), Some(&2));
    }

    #[test]
    fn test_different_moduli_incompatible() {
        let perm = Permutation::identity(2);
        let cp1 = ColoredPermutation::new(perm.clone(), vec![0, 0], 2).unwrap();
        let cp2 = ColoredPermutation::new(perm, vec![0, 0], 3).unwrap();

        // Different moduli should not compose
        assert!(cp1.compose(&cp2).is_none());
        assert!(!cp1.is_conjugate_to(&cp2));
    }
}

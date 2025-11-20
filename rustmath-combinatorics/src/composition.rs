//! Integer compositions (ordered partitions)
//!
//! A composition of n is an ordered sequence of positive integers that sum to n.

/// An integer composition (ordered partition)
///
/// A composition of n is an ordered sequence of positive integers that sum to n
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Composition {
    parts: Vec<usize>,
}

impl Composition {
    /// Create a composition from a vector of parts
    pub fn new(parts: Vec<usize>) -> Option<Self> {
        if parts.iter().any(|&p| p == 0) {
            return None; // All parts must be positive
        }
        Some(Composition { parts })
    }

    /// Get the sum of the composition
    pub fn sum(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }
}

/// Generate all compositions of n
///
/// A composition is an ordered way of writing n as a sum of positive integers
pub fn compositions(n: usize) -> Vec<Composition> {
    if n == 0 {
        return vec![Composition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_compositions(n, &mut current, &mut result);

    result
}

fn generate_compositions(n: usize, current: &mut Vec<usize>, result: &mut Vec<Composition>) {
    if n == 0 {
        result.push(Composition {
            parts: current.clone(),
        });
        return;
    }

    for i in 1..=n {
        current.push(i);
        generate_compositions(n - i, current, result);
        current.pop();
    }
}

/// Generate all compositions of n into exactly k parts
pub fn compositions_k(n: usize, k: usize) -> Vec<Composition> {
    if k == 0 {
        if n == 0 {
            return vec![Composition { parts: vec![] }];
        } else {
            return vec![];
        }
    }

    if k > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_compositions_k(n, k, &mut current, &mut result);

    result
}

fn generate_compositions_k(
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Composition>,
) {
    if current.len() == k {
        if n == 0 {
            result.push(Composition {
                parts: current.clone(),
            });
        }
        return;
    }

    let remaining_parts = k - current.len();
    let min_value = 1;
    let max_value = n.saturating_sub(remaining_parts - 1);

    for i in min_value..=max_value {
        current.push(i);
        generate_compositions_k(n - i, k, current, result);
        current.pop();
    }
}

/// A signed integer composition
///
/// A signed composition is an ordered sequence of non-zero integers (positive or negative)
/// that sum to some value. Each part has both a magnitude and a sign.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignedComposition {
    parts: Vec<isize>,
}

impl SignedComposition {
    /// Create a signed composition from a vector of parts
    ///
    /// Returns None if any part is zero (all parts must be non-zero)
    pub fn new(parts: Vec<isize>) -> Option<Self> {
        if parts.iter().any(|&p| p == 0) {
            return None; // All parts must be non-zero
        }
        Some(SignedComposition { parts })
    }

    /// Get the sum of the composition (considering signs)
    pub fn sum(&self) -> isize {
        self.parts.iter().sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts
    pub fn parts(&self) -> &[isize] {
        &self.parts
    }

    /// Reverse the order of the parts
    ///
    /// Example: [1, -2, 3] becomes [3, -2, 1]
    pub fn reverse(&self) -> Self {
        let mut reversed_parts = self.parts.clone();
        reversed_parts.reverse();
        SignedComposition {
            parts: reversed_parts,
        }
    }

    /// Reverse the signs of all parts
    ///
    /// Example: [1, -2, 3] becomes [-1, 2, -3]
    pub fn reverse_signs(&self) -> Self {
        SignedComposition {
            parts: self.parts.iter().map(|&p| -p).collect(),
        }
    }

    /// Create a signed composition from an unsigned composition with explicit signs
    ///
    /// The signs vector should have the same length as the composition's parts.
    /// True indicates positive, false indicates negative.
    ///
    /// Returns None if the lengths don't match.
    pub fn from_composition(comp: &Composition, signs: Vec<bool>) -> Option<Self> {
        if comp.length() != signs.len() {
            return None;
        }

        let parts: Vec<isize> = comp
            .parts()
            .iter()
            .zip(signs.iter())
            .map(|(&part, &is_positive)| {
                if is_positive {
                    part as isize
                } else {
                    -(part as isize)
                }
            })
            .collect();

        Some(SignedComposition { parts })
    }

    /// Get the absolute value composition (all parts made positive)
    pub fn abs_composition(&self) -> Composition {
        let abs_parts: Vec<usize> = self.parts.iter().map(|&p| p.abs() as usize).collect();
        // Safe unwrap: we know all parts are non-zero, so abs won't be zero
        Composition::new(abs_parts).unwrap()
    }

    /// Get the signs of the parts
    ///
    /// Returns true for positive parts, false for negative parts
    pub fn signs(&self) -> Vec<bool> {
        self.parts.iter().map(|&p| p > 0).collect()
    }
}

/// Generate all signed compositions of n
///
/// A signed composition is an ordered way of writing n as a sum of non-zero integers
/// (which can be positive or negative)
pub fn signed_compositions(n: isize) -> Vec<SignedComposition> {
    // For signed compositions, we first generate all unsigned compositions of |n|,
    // then generate all possible sign assignments
    let abs_n = n.abs() as usize;

    if abs_n == 0 {
        return vec![SignedComposition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let unsigned_comps = compositions(abs_n);

    for comp in unsigned_comps {
        // Generate all 2^k sign assignments for k parts
        let k = comp.length();
        let num_sign_patterns = 1 << k; // 2^k

        for i in 0..num_sign_patterns {
            let signs: Vec<bool> = (0..k).map(|j| (i >> j) & 1 == 1).collect();

            if let Some(signed_comp) = SignedComposition::from_composition(&comp, signs) {
                // Only include compositions that sum to n
                if signed_comp.sum() == n {
                    result.push(signed_comp);
                }
            }
        }
    }

    result
}

/// Generate all signed compositions of n into exactly k parts
pub fn signed_compositions_k(n: isize, k: usize) -> Vec<SignedComposition> {
    if k == 0 {
        if n == 0 {
            return vec![SignedComposition { parts: vec![] }];
        } else {
            return vec![];
        }
    }

    let abs_n = n.abs() as usize;
    let mut result = Vec::new();

    // Generate all unsigned compositions of abs_n into k parts
    let unsigned_comps = compositions_k(abs_n, k);

    for comp in unsigned_comps {
        // Generate all 2^k sign assignments
        let num_sign_patterns = 1 << k;

        for i in 0..num_sign_patterns {
            let signs: Vec<bool> = (0..k).map(|j| (i >> j) & 1 == 1).collect();

            if let Some(signed_comp) = SignedComposition::from_composition(&comp, signs) {
                // Only include compositions that sum to n
                if signed_comp.sum() == n {
                    result.push(signed_comp);
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositions() {
        // Compositions of 4: [4], [1,3], [2,2], [3,1], [1,1,2], [1,2,1], [2,1,1], [1,1,1,1]
        let comps = compositions(4);
        assert_eq!(comps.len(), 8); // 2^(n-1) = 2^3 = 8

        // All compositions should sum to 4
        for comp in &comps {
            assert_eq!(comp.sum(), 4);
        }
    }

    #[test]
    fn test_compositions_k() {
        // Compositions of 5 into 3 parts
        let comps = compositions_k(5, 3);
        // Should be: [1,1,3], [1,2,2], [1,3,1], [2,1,2], [2,2,1], [3,1,1]
        assert_eq!(comps.len(), 6);

        for comp in &comps {
            assert_eq!(comp.sum(), 5);
            assert_eq!(comp.length(), 3);
        }
    }

    #[test]
    fn test_composition_ordering() {
        // Compositions are ordered (unlike partitions)
        let comp1 = Composition::new(vec![1, 3]).unwrap();
        let comp2 = Composition::new(vec![3, 1]).unwrap();

        // These should be different
        assert_ne!(comp1, comp2);
        assert_eq!(comp1.sum(), comp2.sum());
    }

    #[test]
    fn test_signed_composition_new() {
        // Valid signed composition
        let sc = SignedComposition::new(vec![1, -2, 3]).unwrap();
        assert_eq!(sc.parts(), &[1, -2, 3]);
        assert_eq!(sc.sum(), 2);
        assert_eq!(sc.length(), 3);

        // Zero parts should be rejected
        assert!(SignedComposition::new(vec![1, 0, 3]).is_none());
        assert!(SignedComposition::new(vec![0]).is_none());

        // Empty composition is valid
        let empty = SignedComposition::new(vec![]).unwrap();
        assert_eq!(empty.sum(), 0);
        assert_eq!(empty.length(), 0);
    }

    #[test]
    fn test_signed_composition_reverse() {
        let sc = SignedComposition::new(vec![1, -2, 3, -4]).unwrap();
        let reversed = sc.reverse();

        assert_eq!(reversed.parts(), &[-4, 3, -2, 1]);
        assert_eq!(reversed.sum(), sc.sum()); // Sum should be preserved

        // Reversing twice should give original
        let double_reversed = reversed.reverse();
        assert_eq!(double_reversed, sc);
    }

    #[test]
    fn test_signed_composition_reverse_signs() {
        let sc = SignedComposition::new(vec![1, -2, 3, -4]).unwrap();
        let reversed_signs = sc.reverse_signs();

        assert_eq!(reversed_signs.parts(), &[-1, 2, -3, 4]);
        assert_eq!(reversed_signs.sum(), -sc.sum()); // Sum should be negated

        // Reversing signs twice should give original
        let double_reversed = reversed_signs.reverse_signs();
        assert_eq!(double_reversed, sc);
    }

    #[test]
    fn test_signed_composition_from_composition() {
        let comp = Composition::new(vec![1, 2, 3]).unwrap();

        // All positive
        let sc1 = SignedComposition::from_composition(&comp, vec![true, true, true]).unwrap();
        assert_eq!(sc1.parts(), &[1, 2, 3]);
        assert_eq!(sc1.sum(), 6);

        // Mixed signs
        let sc2 = SignedComposition::from_composition(&comp, vec![true, false, true]).unwrap();
        assert_eq!(sc2.parts(), &[1, -2, 3]);
        assert_eq!(sc2.sum(), 2);

        // All negative
        let sc3 = SignedComposition::from_composition(&comp, vec![false, false, false]).unwrap();
        assert_eq!(sc3.parts(), &[-1, -2, -3]);
        assert_eq!(sc3.sum(), -6);

        // Mismatched lengths should fail
        assert!(SignedComposition::from_composition(&comp, vec![true, false]).is_none());
    }

    #[test]
    fn test_signed_composition_abs_and_signs() {
        let sc = SignedComposition::new(vec![1, -2, 3, -4]).unwrap();

        // abs_composition should give unsigned version
        let abs_comp = sc.abs_composition();
        assert_eq!(abs_comp.parts(), &[1, 2, 3, 4]);

        // signs should track positive/negative
        let signs = sc.signs();
        assert_eq!(signs, vec![true, false, true, false]);

        // Round-trip test
        let reconstructed = SignedComposition::from_composition(&abs_comp, signs).unwrap();
        assert_eq!(reconstructed, sc);
    }

    #[test]
    fn test_signed_compositions_generation() {
        // Generate all signed compositions of 2
        let scs = signed_compositions(2);

        // Should include compositions like [2], [1,1], [-1,3], [3,-1], etc.
        // that sum to 2
        assert!(scs.iter().all(|sc| sc.sum() == 2));

        // Check that we have various compositions
        assert!(scs
            .iter()
            .any(|sc| sc.parts() == &[2] || sc.parts() == &[2]));
        assert!(scs.iter().any(|sc| sc.parts() == &[1, 1]));

        // Verify no duplicates
        for i in 0..scs.len() {
            for j in i + 1..scs.len() {
                assert_ne!(scs[i], scs[j]);
            }
        }
    }

    #[test]
    fn test_signed_compositions_negative() {
        // Generate all signed compositions of -2
        let scs = signed_compositions(-2);

        // All should sum to -2
        assert!(scs.iter().all(|sc| sc.sum() == -2));

        // Should include [-2], [-1,-1], etc.
        assert!(scs.iter().any(|sc| sc.parts() == &[-2]));
        assert!(scs.iter().any(|sc| sc.parts() == &[-1, -1]));
    }

    #[test]
    fn test_signed_compositions_k() {
        // Generate all signed compositions of 3 into 2 parts
        let scs = signed_compositions_k(3, 2);

        // All should sum to 3 and have exactly 2 parts
        assert!(scs.iter().all(|sc| sc.sum() == 3));
        assert!(scs.iter().all(|sc| sc.length() == 2));

        // Should include compositions like [1,2], [2,1], [-1,4], [4,-1], etc.
        assert!(scs.iter().any(|sc| sc.parts() == &[1, 2]));
        assert!(scs.iter().any(|sc| sc.parts() == &[2, 1]));

        // Verify no duplicates
        for i in 0..scs.len() {
            for j in i + 1..scs.len() {
                assert_ne!(scs[i], scs[j]);
            }
        }
    }

    #[test]
    fn test_signed_compositions_zero() {
        // Compositions of 0 should only be empty composition
        let scs = signed_compositions(0);
        assert_eq!(scs.len(), 1);
        assert_eq!(scs[0].parts(), &[]);
        assert_eq!(scs[0].sum(), 0);
    }

    #[test]
    fn test_signed_compositions_k_edge_cases() {
        // k=0 with n=0 should give empty composition
        let scs = signed_compositions_k(0, 0);
        assert_eq!(scs.len(), 1);
        assert_eq!(scs[0].parts(), &[]);

        // k=0 with n!=0 should give no compositions
        let scs = signed_compositions_k(5, 0);
        assert_eq!(scs.len(), 0);

        // k > |n| with all same sign should give no compositions
        // (can't make k parts sum to n if k > |n| with all positive)
        let scs = signed_compositions_k(2, 10);
        // This might still have results with mixed signs
        assert!(scs.iter().all(|sc| sc.sum() == 2));
        assert!(scs.iter().all(|sc| sc.length() == 10));
    }

    #[test]
    fn test_signed_composition_symmetries() {
        let sc = SignedComposition::new(vec![1, -2, 3]).unwrap();

        // Test that reverse and reverse_signs commute
        let rev_then_signs = sc.reverse().reverse_signs();
        let signs_then_rev = sc.reverse_signs().reverse();

        // These should be the same
        assert_eq!(rev_then_signs, signs_then_rev);

        // Test sum properties
        assert_eq!(sc.sum(), -sc.reverse_signs().sum());
    }
}

//! Perfect matchings
//!
//! A perfect matching is a partition of vertices into pairs, where each vertex
//! appears in exactly one pair.

/// A perfect matching on 2n vertices
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfectMatching {
    /// The matching as pairs of vertices
    pairs: Vec<(usize, usize)>,
    n: usize,
}

impl PerfectMatching {
    /// Create a new perfect matching
    pub fn new(pairs: Vec<(usize, usize)>, n: usize) -> Option<Self> {
        // Verify it's a valid perfect matching
        let mut seen = vec![false; 2 * n];

        for &(a, b) in &pairs {
            if a >= 2 * n || b >= 2 * n || seen[a] || seen[b] {
                return None;
            }
            seen[a] = true;
            seen[b] = true;
        }

        // Check all vertices are matched
        if !seen.iter().all(|&x| x) {
            return None;
        }

        Some(PerfectMatching { pairs, n })
    }

    /// Get the pairs in the matching
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Get the number of pairs
    pub fn size(&self) -> usize {
        self.n
    }
}

/// Generate all perfect matchings on 2n vertices
pub fn perfect_matchings(n: usize) -> Vec<PerfectMatching> {
    if n == 0 {
        return vec![PerfectMatching {
            pairs: vec![],
            n: 0,
        }];
    }

    let mut result = Vec::new();
    let mut current_pairs = Vec::new();
    let mut available: Vec<usize> = (0..2 * n).collect();

    generate_perfect_matchings(&mut current_pairs, &mut available, n, &mut result);

    result
}

fn generate_perfect_matchings(
    current: &mut Vec<(usize, usize)>,
    available: &mut Vec<usize>,
    n: usize,
    result: &mut Vec<PerfectMatching>,
) {
    if available.is_empty() {
        result.push(PerfectMatching {
            pairs: current.clone(),
            n,
        });
        return;
    }

    // Take the first available vertex and try matching it with all others
    let first = available[0];

    for i in 1..available.len() {
        let second = available[i];

        // Create the pair
        current.push((first, second));

        // Remove both from available
        let mut new_available = available.clone();
        new_available.retain(|&x| x != first && x != second);

        generate_perfect_matchings(current, &mut new_available, n, result);

        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_matchings() {
        // Perfect matchings on 2 vertices: just one matching {(0,1)}
        let matchings1 = perfect_matchings(1);
        assert_eq!(matchings1.len(), 1);

        // Perfect matchings on 4 vertices: should be 3 matchings
        // {(0,1),(2,3)}, {(0,2),(1,3)}, {(0,3),(1,2)}
        let matchings2 = perfect_matchings(2);
        assert_eq!(matchings2.len(), 3);

        // Verify each is a valid matching
        for matching in &matchings2 {
            assert_eq!(matching.pairs().len(), 2);
        }
    }

    #[test]
    fn test_perfect_matching_validation() {
        // Valid matching on 4 vertices
        let matching = PerfectMatching::new(vec![(0, 1), (2, 3)], 2);
        assert!(matching.is_some());

        // Invalid - duplicate vertex
        let invalid = PerfectMatching::new(vec![(0, 1), (1, 2)], 2);
        assert!(invalid.is_none());

        // Invalid - missing vertex
        let invalid2 = PerfectMatching::new(vec![(0, 1)], 2);
        assert!(invalid2.is_none());
    }
}

//! Combinations and combination generation

/// A combination (k-subset of n elements)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combination {
    /// The selected indices (in sorted order)
    indices: Vec<usize>,
    /// Total number of elements
    n: usize,
}

impl Combination {
    /// Create a combination from a vector of indices
    ///
    /// Returns None if indices are invalid
    pub fn from_vec(indices: Vec<usize>, n: usize) -> Option<Self> {
        let k = indices.len();
        if k > n {
            return None;
        }

        // Check that indices are valid and sorted
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable();
        sorted_indices.dedup();

        if sorted_indices.len() != k {
            return None; // Duplicate indices
        }

        for &idx in &sorted_indices {
            if idx >= n {
                return None;
            }
        }

        Some(Combination {
            indices: sorted_indices,
            n,
        })
    }

    /// Get the size of the combination
    pub fn k(&self) -> usize {
        self.indices.len()
    }

    /// Get the total number of elements
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Convert to rank (lexicographic position)
    ///
    /// This is the position of this combination in the lexicographically ordered
    /// list of all C(n, k) combinations
    pub fn rank(&self) -> usize {
        let k = self.k();
        let mut rank = 0;

        for (i, &idx) in self.indices.iter().enumerate() {
            if i == 0 {
                // Count all combinations that come before this one
                for j in 0..idx {
                    rank += binomial_usize(self.n - 1 - j, k - 1);
                }
            } else {
                let prev_idx = self.indices[i - 1];
                for j in (prev_idx + 1)..idx {
                    rank += binomial_usize(self.n - 1 - j, k - 1 - i);
                }
            }
        }

        rank
    }

    /// Create combination from rank (inverse of rank())
    pub fn unrank(rank: usize, n: usize, k: usize) -> Option<Self> {
        if k > n {
            return None;
        }

        let mut indices = Vec::with_capacity(k);
        let mut remaining_rank = rank;
        let mut start = 0;

        for i in 0..k {
            for j in start..n {
                let count = binomial_usize(n - 1 - j, k - 1 - i);
                if remaining_rank < count {
                    indices.push(j);
                    start = j + 1;
                    break;
                }
                remaining_rank -= count;
            }
        }

        if indices.len() == k {
            Some(Combination { indices, n })
        } else {
            None
        }
    }
}

/// Generate all k-combinations of n elements
///
/// Returns all ways to choose k elements from n, in lexicographic order
pub fn combinations(n: usize, k: usize) -> Vec<Combination> {
    if k > n {
        return vec![];
    }

    if k == 0 {
        return vec![Combination {
            indices: vec![],
            n,
        }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_combinations(n, k, 0, &mut current, &mut result);

    result
}

fn generate_combinations(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Combination>,
) {
    if current.len() == k {
        result.push(Combination {
            indices: current.clone(),
            n,
        });
        return;
    }

    for i in start..n {
        current.push(i);
        generate_combinations(n, k, i + 1, current, result);
        current.pop();
    }
}

/// Compute binomial coefficient using usize (for internal use)
fn binomial_usize(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    let k = k.min(n - k);
    let mut result = 1usize;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_creation() {
        let comb = Combination::from_vec(vec![0, 2, 3], 5).unwrap();
        assert_eq!(comb.k(), 3);
        assert_eq!(comb.n(), 5);
        assert_eq!(comb.indices(), &[0, 2, 3]);
    }

    #[test]
    fn test_combinations_generation() {
        // C(4, 2) = 6 combinations
        let combs = combinations(4, 2);
        assert_eq!(combs.len(), 6);

        // Should be: [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]
        assert_eq!(combs[0].indices(), &[0, 1]);
        assert_eq!(combs[1].indices(), &[0, 2]);
        assert_eq!(combs[2].indices(), &[0, 3]);
        assert_eq!(combs[3].indices(), &[1, 2]);
        assert_eq!(combs[4].indices(), &[1, 3]);
        assert_eq!(combs[5].indices(), &[2, 3]);
    }

    #[test]
    fn test_rank_unrank() {
        // Test rank/unrank roundtrip
        let comb = Combination::from_vec(vec![1, 3, 4], 6).unwrap();
        let rank = comb.rank();

        let comb2 = Combination::unrank(rank, 6, 3).unwrap();
        assert_eq!(comb.indices(), comb2.indices());
    }

    #[test]
    fn test_rank_all_combinations() {
        // Verify that all C(5, 3) combinations have unique ranks from 0 to C(5,3)-1
        let combs = combinations(5, 3);
        let mut ranks: Vec<usize> = combs.iter().map(|c| c.rank()).collect();
        ranks.sort();

        for (i, &rank) in ranks.iter().enumerate() {
            assert_eq!(rank, i);
        }
    }

    #[test]
    fn test_edge_cases() {
        // C(n, 0) = 1
        assert_eq!(combinations(5, 0).len(), 1);

        // C(n, n) = 1
        let combs_nn = combinations(3, 3);
        assert_eq!(combs_nn.len(), 1);
        assert_eq!(combs_nn[0].indices(), &[0, 1, 2]);

        // C(n, k) where k > n should be empty
        assert_eq!(combinations(3, 5).len(), 0);
    }
}

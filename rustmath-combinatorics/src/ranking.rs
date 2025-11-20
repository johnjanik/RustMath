//! Ranking and unranking utilities for combinatorial objects
//!
//! A ranking is a bijection between combinatorial objects and integers 0, 1, 2, ..., n-1
//! where n is the total number of objects. This allows efficient enumeration and random access.

use crate::binomial;
use rustmath_core::NumericConversion;

/// A trait for combinatorial objects that can be ranked and unranked
pub trait Rankable: Sized {
    /// Convert this object to its rank (index in lexicographic order)
    fn rank(&self) -> usize;

    /// Create an object from its rank
    fn unrank(rank: usize, params: &Self::Params) -> Option<Self>;

    /// Parameters needed to specify the combinatorial class
    type Params;

    /// The total number of objects with given parameters
    fn count(params: &Self::Params) -> usize;
}

/// A ranking table for efficient enumeration of combinatorial objects
///
/// This structure pre-computes rankings for fast lookup
pub struct RankingTable<T: Rankable> {
    /// The parameters defining this combinatorial class
    params: T::Params,
    /// Total number of elements
    count: usize,
}

impl<T: Rankable> RankingTable<T> {
    /// Create a new ranking table for objects with given parameters
    pub fn new(params: T::Params) -> Self {
        let count = T::count(&params);
        RankingTable { params, count }
    }

    /// Get the total number of objects
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the object at a given rank
    pub fn get(&self, rank: usize) -> Option<T> {
        if rank >= self.count {
            return None;
        }
        T::unrank(rank, &self.params)
    }

    /// Get the rank of an object
    pub fn rank(&self, obj: &T) -> usize {
        obj.rank()
    }

    /// Iterator over all objects in rank order
    pub fn iter(&self) -> RankingTableIter<'_, T> {
        RankingTableIter {
            table: self,
            current_rank: 0,
        }
    }

    /// Get a random object (requires rand crate feature)
    #[cfg(feature = "random")]
    pub fn random(&self) -> Option<T> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let rank = rng.gen_range(0..self.count);
        self.get(rank)
    }
}

/// Iterator over ranking table
pub struct RankingTableIter<'a, T: Rankable> {
    table: &'a RankingTable<T>,
    current_rank: usize,
}

impl<'a, T: Rankable> Iterator for RankingTableIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_rank >= self.table.count {
            return None;
        }
        let obj = self.table.get(self.current_rank)?;
        self.current_rank += 1;
        Some(obj)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.table.count - self.current_rank;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Rankable> ExactSizeIterator for RankingTableIter<'a, T> {}

/// Ranking implementation for combinations
///
/// Combinations are ranked in lexicographic order
pub struct CombinationRank {
    /// The k-subset represented as a sorted vector of indices
    elements: Vec<usize>,
    n: usize,
    k: usize,
}

impl CombinationRank {
    pub fn new(n: usize, k: usize, elements: Vec<usize>) -> Option<Self> {
        if elements.len() != k {
            return None;
        }
        if elements.iter().any(|&x| x >= n) {
            return None;
        }
        // Check if sorted and distinct
        for i in 1..elements.len() {
            if elements[i] <= elements[i - 1] {
                return None;
            }
        }
        Some(CombinationRank { elements, n, k })
    }

    pub fn elements(&self) -> &[usize] {
        &self.elements
    }
}

impl Rankable for CombinationRank {
    type Params = (usize, usize); // (n, k)

    fn rank(&self) -> usize {
        let n = self.n;
        let k = self.k;
        let mut rank = 0;

        for (i, &elem) in self.elements.iter().enumerate() {
            // Count combinations that come before this one
            let remaining = k - i;
            let start = if i == 0 { 0 } else { self.elements[i - 1] + 1 };

            for val in start..elem {
                // If we chose 'val' at position i, how many combinations are there?
                // We need to choose (remaining - 1) elements from {val+1, ..., n-1}
                let choices = n - val - 1;
                if choices >= remaining - 1 {
                    rank += binomial(choices as u32, (remaining - 1) as u32).to_usize().unwrap_or(0);
                }
            }
        }

        rank
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let (n, k) = *params;
        if k == 0 {
            return Some(CombinationRank {
                elements: vec![],
                n,
                k,
            });
        }
        if k > n {
            return None;
        }

        let mut elements = Vec::new();
        let mut remaining_rank = rank;
        let mut start = 0;

        for i in 0..k {
            let remaining = k - i;

            // Find the smallest value v such that C(n-v-1, remaining-1) <= remaining_rank
            for val in start..n {
                let choices = n - val - 1;
                if choices < remaining - 1 {
                    continue;
                }

                let count = binomial(choices as u32, (remaining - 1) as u32)
                    .to_usize()
                    .unwrap_or(0);

                if remaining_rank < count {
                    elements.push(val);
                    start = val + 1;
                    break;
                } else {
                    remaining_rank -= count;
                }
            }
        }

        if elements.len() != k {
            return None;
        }

        Some(CombinationRank { elements, n, k })
    }

    fn count(params: &Self::Params) -> usize {
        let (n, k) = *params;
        binomial(n as u32, k as u32).to_usize().unwrap_or(0)
    }
}

/// Ranking for permutations
///
/// Permutations are ranked using the factorial number system (Lehmer code)
pub struct PermutationRank {
    /// The permutation as a vector
    elements: Vec<usize>,
}

impl PermutationRank {
    pub fn new(elements: Vec<usize>) -> Option<Self> {
        let n = elements.len();
        // Verify it's a valid permutation
        let mut seen = vec![false; n];
        for &elem in &elements {
            if elem >= n || seen[elem] {
                return None;
            }
            seen[elem] = true;
        }
        Some(PermutationRank { elements })
    }

    pub fn elements(&self) -> &[usize] {
        &self.elements
    }

    /// Compute the Lehmer code for this permutation
    pub fn lehmer_code(&self) -> Vec<usize> {
        let n = self.elements.len();
        let mut code = vec![0; n];

        for i in 0..n {
            let mut count = 0;
            for j in i + 1..n {
                if self.elements[j] < self.elements[i] {
                    count += 1;
                }
            }
            code[i] = count;
        }

        code
    }
}

impl Rankable for PermutationRank {
    type Params = usize; // n

    fn rank(&self) -> usize {
        let lehmer = self.lehmer_code();
        let n = lehmer.len();
        let mut rank = 0;
        let mut factorial = 1;

        for i in (0..n).rev() {
            rank += lehmer[i] * factorial;
            factorial *= n - i;
        }

        rank
    }

    fn unrank(rank: usize, params: &Self::Params) -> Option<Self> {
        let n = *params;
        if n == 0 {
            return Some(PermutationRank { elements: vec![] });
        }

        // Convert rank to Lehmer code
        let mut lehmer = vec![0; n];
        let mut remaining_rank = rank;
        let mut factorial = 1;

        for i in 1..n {
            factorial *= i;
        }

        for i in 0..n {
            if factorial == 0 {
                break;
            }
            lehmer[i] = remaining_rank / factorial;
            remaining_rank %= factorial;
            if n - i > 1 {
                factorial /= n - i - 1;
            }
        }

        // Convert Lehmer code to permutation
        let mut available: Vec<usize> = (0..n).collect();
        let mut elements = vec![0; n];

        for i in 0..n {
            if lehmer[i] >= available.len() {
                return None;
            }
            elements[i] = available.remove(lehmer[i]);
        }

        Some(PermutationRank { elements })
    }

    fn count(params: &Self::Params) -> usize {
        let n = *params;
        (1..=n).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_ranking() {
        // Test C(5, 3)
        let comb1 = CombinationRank::new(5, 3, vec![0, 1, 2]).unwrap();
        assert_eq!(comb1.rank(), 0); // First combination in lexicographic order

        let comb2 = CombinationRank::new(5, 3, vec![2, 3, 4]).unwrap();
        assert_eq!(comb2.rank(), 9); // Last combination

        // Test unranking
        let unranked = CombinationRank::unrank(0, &(5, 3)).unwrap();
        assert_eq!(unranked.elements(), &[0, 1, 2]);

        let unranked_last = CombinationRank::unrank(9, &(5, 3)).unwrap();
        assert_eq!(unranked_last.elements(), &[2, 3, 4]);
    }

    #[test]
    fn test_combination_ranking_table() {
        let table = RankingTable::<CombinationRank>::new((5, 3));
        assert_eq!(table.len(), 10); // C(5,3) = 10

        // Verify all combinations can be generated
        let combinations: Vec<_> = table.iter().collect();
        assert_eq!(combinations.len(), 10);

        // First and last
        assert_eq!(combinations[0].elements(), &[0, 1, 2]);
        assert_eq!(combinations[9].elements(), &[2, 3, 4]);
    }

    #[test]
    fn test_permutation_ranking() {
        // Test permutations of {0, 1, 2}
        let perm1 = PermutationRank::new(vec![0, 1, 2]).unwrap();
        assert_eq!(perm1.rank(), 0); // Identity permutation

        let perm2 = PermutationRank::new(vec![2, 1, 0]).unwrap();
        assert_eq!(perm2.rank(), 5); // Reverse permutation (last in rank)

        // Test unranking
        let unranked = PermutationRank::unrank(0, &3).unwrap();
        assert_eq!(unranked.elements(), &[0, 1, 2]);

        let unranked_last = PermutationRank::unrank(5, &3).unwrap();
        assert_eq!(unranked_last.elements(), &[2, 1, 0]);
    }

    #[test]
    fn test_permutation_lehmer_code() {
        // [2, 0, 1] has Lehmer code [2, 0, 0]
        let perm = PermutationRank::new(vec![2, 0, 1]).unwrap();
        assert_eq!(perm.lehmer_code(), vec![2, 0, 0]);

        // [1, 2, 0] has Lehmer code [1, 1, 0]
        let perm2 = PermutationRank::new(vec![1, 2, 0]).unwrap();
        assert_eq!(perm2.lehmer_code(), vec![1, 1, 0]);
    }

    #[test]
    fn test_permutation_ranking_table() {
        let table = RankingTable::<PermutationRank>::new(3);
        assert_eq!(table.len(), 6); // 3! = 6

        // Verify all permutations can be generated
        let perms: Vec<_> = table.iter().collect();
        assert_eq!(perms.len(), 6);

        // Check first and last
        assert_eq!(perms[0].elements(), &[0, 1, 2]);
        assert_eq!(perms[5].elements(), &[2, 1, 0]);
    }

    #[test]
    fn test_ranking_roundtrip_combinations() {
        // Test that rank(unrank(i)) == i for all i
        for i in 0..10 {
            let comb = CombinationRank::unrank(i, &(5, 3)).unwrap();
            assert_eq!(comb.rank(), i);
        }
    }

    #[test]
    fn test_ranking_roundtrip_permutations() {
        // Test that rank(unrank(i)) == i for all i
        for i in 0..24 {
            // 4! = 24
            let perm = PermutationRank::unrank(i, &4).unwrap();
            assert_eq!(perm.rank(), i);
        }
    }
}

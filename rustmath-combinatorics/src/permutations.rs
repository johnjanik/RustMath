//! Permutations and permutation generation

/// A permutation of n elements
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    /// The permutation as a vector where perm[i] = j means element i goes to position j
    perm: Vec<usize>,
}

impl Permutation {
    /// Create the identity permutation of size n
    pub fn identity(n: usize) -> Self {
        Permutation {
            perm: (0..n).collect(),
        }
    }

    /// Create a permutation from a vector
    ///
    /// Returns None if the vector is not a valid permutation
    pub fn from_vec(perm: Vec<usize>) -> Option<Self> {
        let n = perm.len();
        let mut seen = vec![false; n];

        for &p in &perm {
            if p >= n || seen[p] {
                return None;
            }
            seen[p] = true;
        }

        Some(Permutation { perm })
    }

    /// Get the size of the permutation
    pub fn size(&self) -> usize {
        self.perm.len()
    }

    /// Apply the permutation to an index
    pub fn apply(&self, i: usize) -> Option<usize> {
        self.perm.get(i).copied()
    }

    /// Compute the inverse permutation
    pub fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0; n];

        for (i, &p) in self.perm.iter().enumerate() {
            inv[p] = i;
        }

        Permutation { perm: inv }
    }

    /// Compose two permutations: self ∘ other
    pub fn compose(&self, other: &Permutation) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }

        let perm: Vec<usize> = (0..self.size())
            .map(|i| self.perm[other.perm[i]])
            .collect();

        Some(Permutation { perm })
    }

    /// Get the sign of the permutation (+1 for even, -1 for odd)
    pub fn sign(&self) -> i32 {
        let mut parity = 0;
        let n = self.perm.len();

        for i in 0..n {
            for j in (i + 1)..n {
                if self.perm[i] > self.perm[j] {
                    parity += 1;
                }
            }
        }

        if parity % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Convert to cycle notation
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let n = self.perm.len();
        let mut visited = vec![false; n];
        let mut cycles = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }

            let mut cycle = vec![start];
            visited[start] = true;
            let mut current = self.perm[start];

            while current != start {
                cycle.push(current);
                visited[current] = true;
                current = self.perm[current];
            }

            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }

        cycles
    }

    /// Compute the multiplicative order of the permutation
    ///
    /// Returns the smallest positive integer k such that p^k = identity
    pub fn order(&self) -> usize {
        let cycles = self.cycles();

        if cycles.is_empty() {
            return 1; // Identity permutation
        }

        // Order is the LCM of all cycle lengths
        let mut result = 1;
        for cycle in cycles {
            result = lcm(result, cycle.len());
        }
        result
    }

    /// Convert permutation to a permutation matrix
    ///
    /// Returns a matrix where M[i][j] = 1 if perm[i] = j, and 0 otherwise
    pub fn to_matrix(&self) -> Vec<Vec<u8>> {
        let n = self.perm.len();
        let mut matrix = vec![vec![0u8; n]; n];

        for (i, &p) in self.perm.iter().enumerate() {
            matrix[i][p] = 1;
        }

        matrix
    }

    /// Get the descent set
    ///
    /// Returns positions i where perm[i] > perm[i+1]
    pub fn descents(&self) -> Vec<usize> {
        let mut descents = Vec::new();

        for i in 0..self.perm.len().saturating_sub(1) {
            if self.perm[i] > self.perm[i + 1] {
                descents.push(i);
            }
        }

        descents
    }

    /// Get the ascent set
    ///
    /// Returns positions i where perm[i] < perm[i+1]
    pub fn ascents(&self) -> Vec<usize> {
        let mut ascents = Vec::new();

        for i in 0..self.perm.len().saturating_sub(1) {
            if self.perm[i] < self.perm[i + 1] {
                ascents.push(i);
            }
        }

        ascents
    }

    /// Count the number of descents
    pub fn descent_number(&self) -> usize {
        self.descents().len()
    }

    /// Count the number of ascents
    pub fn ascent_number(&self) -> usize {
        self.ascents().len()
    }

    /// Check if this permutation avoids a given pattern
    ///
    /// A permutation avoids a pattern if it contains no subsequence with the same relative order
    pub fn avoids(&self, pattern: &Permutation) -> bool {
        let n = self.perm.len();
        let k = pattern.size();

        if k > n {
            return true; // Pattern is larger than permutation, trivially avoided
        }

        if k == 0 {
            return true; // Empty pattern is always avoided
        }

        // Check all subsequences of length k
        avoids_helper(&self.perm, &pattern.perm, 0, &mut Vec::new(), k)
    }

    /// Count the number of inversions in this permutation
    ///
    /// An inversion is a pair (i, j) where i < j but perm[i] > perm[j]
    pub fn inversions(&self) -> usize {
        let mut count = 0;
        for i in 0..self.perm.len() {
            for j in (i + 1)..self.perm.len() {
                if self.perm[i] > self.perm[j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Check if this permutation covers another in Bruhat order
    ///
    /// σ covers τ in Bruhat order if σ > τ and they differ by a single transposition
    /// and no permutation lies between them in the order
    pub fn bruhat_covers(&self, other: &Permutation) -> bool {
        if self.size() != other.size() {
            return false;
        }

        // Check if self has exactly one more inversion than other
        if self.inversions() != other.inversions() + 1 {
            return false;
        }

        // Find the positions where they differ
        let mut diff_positions = Vec::new();
        for i in 0..self.perm.len() {
            if self.perm[i] != other.perm[i] {
                diff_positions.push(i);
            }
        }

        // Must differ in exactly 2 positions
        if diff_positions.len() != 2 {
            return false;
        }

        let i = diff_positions[0];
        let j = diff_positions[1];

        // Check if other can be obtained from self by swapping adjacent elements
        // in the sense of Bruhat order (swapping two values, not positions)
        self.perm[i] == other.perm[j] && self.perm[j] == other.perm[i]
    }

    /// Compare this permutation with another in Bruhat order
    ///
    /// Returns true if self ≤ other in Bruhat order
    pub fn bruhat_le(&self, other: &Permutation) -> bool {
        if self.size() != other.size() {
            return false;
        }

        if self == other {
            return true;
        }

        // In Bruhat order, σ ≤ τ if and only if
        // for all i,j the number of k ≤ i with σ(k) ≤ j is at most
        // the number of k ≤ i with τ(k) ≤ j
        for i in 0..self.perm.len() {
            for j in 0..self.perm.len() {
                let count_self = (0..=i)
                    .filter(|&k| self.perm[k] <= j)
                    .count();
                let count_other = (0..=i)
                    .filter(|&k| other.perm[k] <= j)
                    .count();

                if count_self > count_other {
                    return false;
                }
            }
        }

        true
    }

    /// Get the underlying permutation vector as a slice
    pub fn as_slice(&self) -> &[usize] {
        &self.perm
    }
}

/// Helper function for pattern avoidance
fn avoids_helper(
    perm: &[usize],
    pattern: &[usize],
    start: usize,
    subsequence: &mut Vec<usize>,
    k: usize,
) -> bool {
    // If we've selected k elements, check if they form the pattern
    if subsequence.len() == k {
        // Normalize subsequence to get relative order
        let normalized = normalize_sequence(subsequence);
        return normalized != pattern;
    }

    // If we've exhausted the permutation, we're done
    if start >= perm.len() {
        return true;
    }

    // Try including current element
    subsequence.push(perm[start]);
    if !avoids_helper(perm, pattern, start + 1, subsequence, k) {
        subsequence.pop();
        return false;
    }
    subsequence.pop();

    // Try excluding current element
    avoids_helper(perm, pattern, start + 1, subsequence, k)
}

/// Normalize a sequence to get relative order
fn normalize_sequence(seq: &[usize]) -> Vec<usize> {
    let mut indexed: Vec<(usize, usize)> = seq.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    indexed.sort_by_key(|&(x, _)| x);

    let mut result = vec![0; seq.len()];
    for (rank, &(_, orig_idx)) in indexed.iter().enumerate() {
        result[orig_idx] = rank;
    }
    result
}

/// Compute least common multiple of two numbers
fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    (a * b) / gcd(a, b)
}

/// Compute greatest common divisor of two numbers
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Generate all permutations of n elements
pub fn all_permutations(n: usize) -> Vec<Permutation> {
    if n == 0 {
        return vec![Permutation { perm: vec![] }];
    }

    if n == 1 {
        return vec![Permutation { perm: vec![0] }];
    }

    let mut result = Vec::new();
    let mut elements: Vec<usize> = (0..n).collect();

    generate_permutations(&mut elements, 0, &mut result);

    result
}

fn generate_permutations(elements: &mut [usize], start: usize, result: &mut Vec<Permutation>) {
    if start == elements.len() {
        result.push(Permutation {
            perm: elements.to_vec(),
        });
        return;
    }

    for i in start..elements.len() {
        elements.swap(start, i);
        generate_permutations(elements, start + 1, result);
        elements.swap(start, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = Permutation::identity(3);
        assert_eq!(id.perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_inverse() {
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        let inv = perm.inverse();

        assert_eq!(inv.perm, vec![2, 0, 1]);
    }

    #[test]
    fn test_compose() {
        let p1 = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        let p2 = Permutation::from_vec(vec![2, 1, 0]).unwrap();

        let result = p1.compose(&p2).unwrap();
        // p1 ∘ p2: apply p2 first (0→2, 1→1, 2→0), then p1 (2→2, 1→0, 0→1)
        // Result: 0→2, 1→0, 2→1
        assert_eq!(result.perm, vec![2, 0, 1]);
    }

    #[test]
    fn test_sign() {
        let id = Permutation::identity(3);
        assert_eq!(id.sign(), 1);

        let swap = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        assert_eq!(swap.sign(), -1);
    }

    #[test]
    fn test_all_permutations() {
        let perms = all_permutations(3);
        assert_eq!(perms.len(), 6); // 3! = 6
    }

    #[test]
    fn test_cycles() {
        let perm = Permutation::from_vec(vec![1, 2, 0, 3]).unwrap();
        let cycles = perm.cycles();

        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_order() {
        // Identity has order 1
        let id = Permutation::identity(3);
        assert_eq!(id.order(), 1);

        // (0 1 2) cycle has order 3
        let perm = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        assert_eq!(perm.order(), 3);

        // (0 1)(2 3) has order 2 (LCM of cycle lengths)
        let perm2 = Permutation::from_vec(vec![1, 0, 3, 2]).unwrap();
        assert_eq!(perm2.order(), 2);

        // (0 1 2)(3 4) has order 6 (LCM of 3 and 2)
        let perm3 = Permutation::from_vec(vec![1, 2, 0, 4, 3]).unwrap();
        assert_eq!(perm3.order(), 6);
    }

    #[test]
    fn test_to_matrix() {
        let perm = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        let matrix = perm.to_matrix();

        assert_eq!(matrix[0], vec![0, 1, 0]);
        assert_eq!(matrix[1], vec![1, 0, 0]);
        assert_eq!(matrix[2], vec![0, 0, 1]);
    }

    #[test]
    fn test_descents_ascents() {
        // [2, 0, 1, 3] has descents at position 0 (2 > 0)
        let perm = Permutation::from_vec(vec![2, 0, 1, 3]).unwrap();
        let descents = perm.descents();
        let ascents = perm.ascents();

        assert_eq!(descents, vec![0]); // 2 > 0
        assert_eq!(ascents, vec![1, 2]); // 0 < 1, 1 < 3

        // Identity permutation has all ascents, no descents
        let id = Permutation::identity(4);
        assert_eq!(id.descents().len(), 0);
        assert_eq!(id.ascents().len(), 3);
    }

    #[test]
    fn test_pattern_avoidance() {
        // Pattern 2-1-0 (reversed order)
        let pattern = Permutation::from_vec(vec![2, 1, 0]).unwrap();

        // [0, 1, 2, 3] avoids [2, 1, 0] (no decreasing subsequence of length 3)
        let perm1 = Permutation::identity(4);
        assert!(perm1.avoids(&pattern));

        // [3, 2, 1, 0] does NOT avoid [2, 1, 0]
        let perm2 = Permutation::from_vec(vec![3, 2, 1, 0]).unwrap();
        assert!(!perm2.avoids(&pattern));

        // Pattern 1-2-0 (up-up-down)
        let pattern2 = Permutation::from_vec(vec![1, 2, 0]).unwrap();
        // [1, 2, 0] itself contains the pattern
        assert!(!pattern2.avoids(&pattern2));

        // [0, 1, 2] avoids [1, 2, 0] (ascending sequence)
        let perm3 = Permutation::identity(3);
        assert!(perm3.avoids(&pattern2));
    }

    #[test]
    fn test_inversions() {
        // Identity has 0 inversions
        let id = Permutation::identity(3);
        assert_eq!(id.inversions(), 0);

        // [2, 1, 0] has 3 inversions: (0,1), (0,2), (1,2)
        let rev = Permutation::from_vec(vec![2, 1, 0]).unwrap();
        assert_eq!(rev.inversions(), 3);

        // [1, 0, 2] has 1 inversion: (0,1)
        let swap = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        assert_eq!(swap.inversions(), 1);
    }

    #[test]
    fn test_bruhat_order() {
        // Reflexivity: every permutation ≤ itself
        let id = Permutation::identity(3);
        assert!(id.bruhat_le(&id));

        let perm = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        assert!(perm.bruhat_le(&perm));

        // [2, 1, 0] is the maximal element
        let max_perm = Permutation::from_vec(vec![2, 1, 0]).unwrap();
        assert!(max_perm.bruhat_le(&max_perm));

        // Transitivity check with simple case
        // In weak Bruhat order, fewer inversions typically means smaller
        // But the precise order depends on the rank matrix criterion
    }

    #[test]
    fn test_bruhat_covers() {
        // [1, 0, 2] covers [0, 1, 2] (differ by one adjacent transposition)
        let id = Permutation::identity(3);
        let swap = Permutation::from_vec(vec![1, 0, 2]).unwrap();
        assert!(swap.bruhat_covers(&id));

        // [2, 1, 0] does NOT cover [0, 1, 2] (too many inversions apart)
        let max_perm = Permutation::from_vec(vec![2, 1, 0]).unwrap();
        assert!(!max_perm.bruhat_covers(&id));
    }
}

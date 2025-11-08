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
}

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
        assert_eq!(result.perm, vec![1, 2, 0]);
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
}

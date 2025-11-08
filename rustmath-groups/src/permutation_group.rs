//! Permutation groups - finite groups of permutations
//!
//! This module implements permutation groups including the symmetric
//! and alternating groups.

use rustmath_combinatorics::Permutation;
use std::fmt;

/// A permutation group - a subgroup of the symmetric group
#[derive(Clone, Debug)]
pub struct PermutationGroup {
    /// Degree of the group (permutations act on {0, 1, ..., n-1})
    degree: usize,
    /// Generators of the group
    generators: Vec<Permutation>,
}

impl PermutationGroup {
    /// Create a new permutation group from generators
    pub fn new(degree: usize, generators: Vec<Permutation>) -> Result<Self, String> {
        // Validate that all generators have the correct degree
        for gen in &generators {
            if gen.size() != degree {
                return Err(format!(
                    "Generator has degree {}, expected {}",
                    gen.size(),
                    degree
                ));
            }
        }

        Ok(PermutationGroup { degree, generators })
    }

    /// Get the degree of the group
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Get the generators
    pub fn generators(&self) -> &[Permutation] {
        &self.generators
    }
}

impl fmt::Display for PermutationGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PermutationGroup(degree={}, generators={})",
            self.degree,
            self.generators.len()
        )
    }
}

/// The symmetric group S_n
pub struct SymmetricGroup {
    n: usize,
    group: PermutationGroup,
}

impl SymmetricGroup {
    /// Create the symmetric group S_n
    pub fn new(n: usize) -> Self {
        // Generate S_n using adjacent transpositions
        let mut generators = Vec::new();

        if n >= 2 {
            for i in 0..n - 1 {
                let mut perm = Vec::new();
                for j in 0..n {
                    if j == i {
                        perm.push(i + 1);
                    } else if j == i + 1 {
                        perm.push(i);
                    } else {
                        perm.push(j);
                    }
                }
                if let Some(p) = Permutation::from_vec(perm) {
                    generators.push(p);
                }
            }
        }

        let group = PermutationGroup::new(n, generators).unwrap();

        SymmetricGroup { n, group }
    }

    /// Get the degree n
    pub fn degree(&self) -> usize {
        self.n
    }

    /// Get the order of S_n (= n!)
    pub fn order(&self) -> usize {
        factorial(self.n)
    }

    /// Get the underlying permutation group
    pub fn group(&self) -> &PermutationGroup {
        &self.group
    }
}

/// The alternating group A_n (even permutations)
pub struct AlternatingGroup {
    n: usize,
    group: PermutationGroup,
}

impl AlternatingGroup {
    /// Create the alternating group A_n
    pub fn new(n: usize) -> Self {
        if n < 3 {
            // A_1 and A_2 are trivial
            let identity = Permutation::identity(n);
            let group = PermutationGroup::new(n, vec![identity]).unwrap();
            return AlternatingGroup { n, group };
        }

        // Generate A_n using 3-cycles
        let mut generators = Vec::new();

        // Add the 3-cycle (0, 1, 2) -> (1, 2, 0)
        if n >= 3 {
            let mut perm = vec![1, 2, 0];
            perm.extend(3..n);
            if let Some(p) = Permutation::from_vec(perm) {
                generators.push(p);
            }
        }

        // Add more 3-cycles for higher degrees
        if n >= 4 {
            for i in 3..n {
                let mut perm: Vec<usize> = (0..n).collect();
                // Create 3-cycle (0, 1, i)
                perm[0] = 1;
                perm[1] = i;
                perm[i] = 0;
                if let Some(p) = Permutation::from_vec(perm) {
                    generators.push(p);
                }
            }
        }

        let group = PermutationGroup::new(n, generators).unwrap();

        AlternatingGroup { n, group }
    }

    /// Get the degree n
    pub fn degree(&self) -> usize {
        self.n
    }

    /// Get the order of A_n (= n!/2)
    pub fn order(&self) -> usize {
        if self.n < 2 {
            1
        } else {
            factorial(self.n) / 2
        }
    }

    /// Get the underlying permutation group
    pub fn group(&self) -> &PermutationGroup {
        &self.group
    }
}

/// Compute factorial
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_group_creation() {
        // Create Z/2Z as a permutation group
        let swap = Permutation::from_vec(vec![1, 0]).unwrap();
        let group = PermutationGroup::new(2, vec![swap]);
        assert!(group.is_ok());
    }

    #[test]
    fn test_permutation_group_degree() {
        let swap = Permutation::from_vec(vec![1, 0]).unwrap();
        let group = PermutationGroup::new(2, vec![swap]).unwrap();
        assert_eq!(group.degree(), 2);
    }

    #[test]
    fn test_symmetric_group_s3() {
        let s3 = SymmetricGroup::new(3);
        assert_eq!(s3.degree(), 3);
        assert_eq!(s3.order(), 6); // 3! = 6
    }

    #[test]
    fn test_symmetric_group_s4() {
        let s4 = SymmetricGroup::new(4);
        assert_eq!(s4.degree(), 4);
        assert_eq!(s4.order(), 24); // 4! = 24
    }

    #[test]
    fn test_alternating_group_a3() {
        let a3 = AlternatingGroup::new(3);
        assert_eq!(a3.degree(), 3);
        assert_eq!(a3.order(), 3); // 3!/2 = 3
    }

    #[test]
    fn test_alternating_group_a4() {
        let a4 = AlternatingGroup::new(4);
        assert_eq!(a4.degree(), 4);
        assert_eq!(a4.order(), 12); // 4!/2 = 12
    }

    #[test]
    fn test_alternating_group_a5() {
        let a5 = AlternatingGroup::new(5);
        assert_eq!(a5.degree(), 5);
        assert_eq!(a5.order(), 60); // 5!/2 = 60
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(2), 2);
        assert_eq!(factorial(3), 6);
        assert_eq!(factorial(4), 24);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_group_generators() {
        let swap = Permutation::from_vec(vec![1, 0]).unwrap();
        let group = PermutationGroup::new(2, vec![swap.clone()]).unwrap();

        let gens = group.generators();
        assert_eq!(gens.len(), 1);
    }

    #[test]
    fn test_s2_generators() {
        let s2 = SymmetricGroup::new(2);
        // S_2 generated by one transposition
        assert_eq!(s2.group().generators().len(), 1);
    }

    #[test]
    fn test_s3_generators() {
        let s3 = SymmetricGroup::new(3);
        // S_3 generated by two adjacent transpositions
        assert_eq!(s3.group().generators().len(), 2);
    }
}

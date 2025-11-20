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
}

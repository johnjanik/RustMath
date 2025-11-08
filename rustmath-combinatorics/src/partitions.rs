//! Integer partitions and partition generation

/// A partition of an integer n is a way of writing n as a sum of positive integers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    /// The parts of the partition in non-increasing order
    parts: Vec<usize>,
}

impl Partition {
    /// Create a partition from a vector of parts
    ///
    /// The parts will be sorted in non-increasing order
    pub fn new(mut parts: Vec<usize>) -> Self {
        parts.retain(|&p| p > 0);
        parts.sort_by(|a, b| b.cmp(a)); // Sort in decreasing order
        Partition { parts }
    }

    /// Get the number being partitioned
    pub fn sum(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts as a slice
    pub fn parts(&self) -> &[usize] {
        &self.parts
    }

    /// Get the largest part
    pub fn largest_part(&self) -> Option<usize> {
        self.parts.first().copied()
    }

    /// Check if this is a partition of n
    pub fn is_partition_of(&self, n: usize) -> bool {
        self.sum() == n
    }

    /// Convert to Ferrers diagram representation
    pub fn ferrers_diagram(&self) -> String {
        self.parts
            .iter()
            .map(|&p| "*".repeat(p))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the conjugate (transpose) partition
    pub fn conjugate(&self) -> Self {
        if self.parts.is_empty() {
            return Partition { parts: vec![] };
        }

        let max_part = self.largest_part().unwrap_or(0);
        let mut conjugate_parts = vec![0; max_part];

        for &part in &self.parts {
            for i in 0..part {
                conjugate_parts[i] += 1;
            }
        }

        Partition {
            parts: conjugate_parts,
        }
    }

    /// Compute hook lengths for the Young diagram
    ///
    /// Returns a vector of vectors where hook_lengths[i][j] is the hook length
    /// at position (i, j) in the Young diagram
    pub fn hook_lengths(&self) -> Vec<Vec<usize>> {
        if self.parts.is_empty() {
            return vec![];
        }

        let mut hooks = Vec::new();
        let conjugate = self.conjugate();

        for (i, &row_len) in self.parts.iter().enumerate() {
            let mut row_hooks = Vec::new();
            for j in 0..row_len {
                // Hook length = cells to right + cells below + 1
                let cells_right = row_len - j - 1;
                let cells_below = conjugate.parts[j] - i - 1;
                row_hooks.push(cells_right + cells_below + 1);
            }
            hooks.push(row_hooks);
        }

        hooks
    }

    /// Compute the dimension (number of standard Young tableaux)
    ///
    /// Uses the hook length formula: n! / product(hook lengths)
    pub fn dimension(&self) -> usize {
        if self.parts.is_empty() {
            return 1;
        }

        let n = self.sum();
        let hooks = self.hook_lengths();

        // Compute n!
        let mut numerator = 1usize;
        for i in 2..=n {
            numerator *= i;
        }

        // Compute product of hook lengths
        let mut denominator = 1usize;
        for row in hooks {
            for hook in row {
                denominator *= hook;
            }
        }

        numerator / denominator
    }

    /// Check if this partition dominates another in the dominance ordering
    ///
    /// λ dominates μ if the sum of the first k parts of λ is >= the sum of
    /// the first k parts of μ for all k
    pub fn dominates(&self, other: &Partition) -> bool {
        if self.sum() != other.sum() {
            return false; // Can only compare partitions of the same number
        }

        let max_len = self.length().max(other.length());
        let mut sum_self = 0;
        let mut sum_other = 0;

        for i in 0..max_len {
            sum_self += self.parts.get(i).copied().unwrap_or(0);
            sum_other += other.parts.get(i).copied().unwrap_or(0);

            if sum_self < sum_other {
                return false;
            }
        }

        true
    }
}

/// Generate all partitions of n
pub fn partitions(n: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_partitions(n, n, &mut current, &mut result);

    result
}

fn generate_partitions(
    n: usize,
    max_value: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Partition>,
) {
    if n == 0 {
        result.push(Partition {
            parts: current.clone(),
        });
        return;
    }

    for i in (1..=max_value.min(n)).rev() {
        current.push(i);
        generate_partitions(n - i, i, current, result);
        current.pop();
    }
}

/// Count the number of partitions of n (partition function p(n))
///
/// Uses a simple recurrence relation. For large n, more sophisticated
/// methods like Hardy-Ramanujan or Rademacher formulas would be needed.
pub fn partition_count(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    // Use dynamic programming
    let mut dp = vec![0; n + 1];
    dp[0] = 1;

    for part in 1..=n {
        for sum in part..=n {
            dp[sum] += dp[sum - part];
        }
    }

    dp[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let p = Partition::new(vec![3, 1, 2]);
        assert_eq!(p.parts(), &[3, 2, 1]); // Should be sorted
        assert_eq!(p.sum(), 6);
        assert_eq!(p.length(), 3);
    }

    #[test]
    fn test_conjugate() {
        let p = Partition::new(vec![3, 2, 1]);
        let conj = p.conjugate();

        assert_eq!(conj.parts(), &[3, 2, 1]); // Self-conjugate
    }

    #[test]
    fn test_partitions_4() {
        let parts = partitions(4);
        // Partitions of 4: [4], [3,1], [2,2], [2,1,1], [1,1,1,1]
        assert_eq!(parts.len(), 5);
    }

    #[test]
    fn test_partition_count() {
        assert_eq!(partition_count(0), 1);
        assert_eq!(partition_count(1), 1);
        assert_eq!(partition_count(2), 2); // [2], [1,1]
        assert_eq!(partition_count(3), 3); // [3], [2,1], [1,1,1]
        assert_eq!(partition_count(4), 5);
        assert_eq!(partition_count(5), 7);
    }

    #[test]
    fn test_ferrers_diagram() {
        let p = Partition::new(vec![3, 2, 1]);
        let diagram = p.ferrers_diagram();
        assert_eq!(diagram, "***\n**\n*");
    }

    #[test]
    fn test_hook_lengths() {
        // Partition [3, 2, 1]
        let p = Partition::new(vec![3, 2, 1]);
        let hooks = p.hook_lengths();

        // Expected hook lengths:
        // 5 3 1
        // 3 1
        // 1
        assert_eq!(hooks, vec![vec![5, 3, 1], vec![3, 1], vec![1]]);
    }

    #[test]
    fn test_dimension() {
        // Partition [2, 1] has dimension 2
        // (2 standard Young tableaux: 1 2  and  1 3)
        //                              3      2
        let p = Partition::new(vec![2, 1]);
        assert_eq!(p.dimension(), 2);

        // Partition [3, 2, 1] has dimension 16
        let p2 = Partition::new(vec![3, 2, 1]);
        assert_eq!(p2.dimension(), 16);

        // Partition [n] (single row) has dimension 1
        let p3 = Partition::new(vec![5]);
        assert_eq!(p3.dimension(), 1);
    }

    #[test]
    fn test_dominates() {
        // [3, 2] dominates [2, 2, 1]
        let p1 = Partition::new(vec![3, 2]);
        let p2 = Partition::new(vec![2, 2, 1]);
        assert!(p1.dominates(&p2));
        assert!(!p2.dominates(&p1));

        // [3, 1, 1] dominates [2, 2, 1]
        let p3 = Partition::new(vec![3, 1, 1]);
        assert!(p3.dominates(&p2));

        // Every partition dominates itself
        assert!(p1.dominates(&p1));

        // [5] dominates everything
        let p5 = Partition::new(vec![5]);
        assert!(p5.dominates(&p1));
        assert!(p5.dominates(&p2));

        // [1,1,1,1,1] is dominated by everything
        let p_min = Partition::new(vec![1, 1, 1, 1, 1]);
        assert!(p1.dominates(&p_min));
        assert!(p2.dominates(&p_min));
    }
}

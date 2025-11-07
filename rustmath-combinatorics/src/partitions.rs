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
}

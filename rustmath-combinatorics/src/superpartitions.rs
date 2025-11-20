//! Superpartitions - partitions with circled and uncircled parts
//!
//! A superpartition is a partition where each part can be either circled or uncircled.
//! They arise in the study of symmetric functions, representation theory, and
//! combinatorics of tableaux.

use std::fmt;

/// Represents a single part in a superpartition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuperPart {
    /// The value of the part
    pub value: usize,
    /// Whether this part is circled
    pub circled: bool,
}

impl SuperPart {
    /// Create a new superpartition part
    pub fn new(value: usize, circled: bool) -> Self {
        SuperPart { value, circled }
    }

    /// Create an uncircled part
    pub fn uncircled(value: usize) -> Self {
        SuperPart {
            value,
            circled: false,
        }
    }

    /// Create a circled part
    pub fn circled(value: usize) -> Self {
        SuperPart {
            value,
            circled: true,
        }
    }
}

impl fmt::Display for SuperPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.circled {
            write!(f, "◯{}", self.value)
        } else {
            write!(f, "{}", self.value)
        }
    }
}

/// A superpartition with circled and uncircled parts
///
/// A superpartition is a partition where each part can be either circled or uncircled.
/// Parts are stored in non-increasing order, with the convention that circled parts
/// come before uncircled parts of the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SuperPartition {
    /// The parts of the superpartition, ordered by value (decreasing)
    /// Circled parts come before uncircled parts of the same value
    parts: Vec<SuperPart>,
}

impl SuperPartition {
    /// Create a new superpartition from a vector of parts
    ///
    /// The parts will be sorted in the canonical order:
    /// - By value (decreasing)
    /// - For equal values, circled parts come first
    pub fn new(mut parts: Vec<SuperPart>) -> Self {
        // Remove zero parts
        parts.retain(|p| p.value > 0);

        // Sort: first by value (descending), then by circled status (circled first)
        parts.sort_by(|a, b| {
            match b.value.cmp(&a.value) {
                std::cmp::Ordering::Equal => b.circled.cmp(&a.circled),
                other => other,
            }
        });

        SuperPartition { parts }
    }

    /// Create an empty superpartition
    pub fn empty() -> Self {
        SuperPartition { parts: vec![] }
    }

    /// Get the sum of all parts (ignoring circled status)
    pub fn sum(&self) -> usize {
        self.parts.iter().map(|p| p.value).sum()
    }

    /// Get the number of parts
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Get the parts as a slice
    pub fn parts(&self) -> &[SuperPart] {
        &self.parts
    }

    /// Get the number of circled parts
    pub fn num_circled(&self) -> usize {
        self.parts.iter().filter(|p| p.circled).count()
    }

    /// Get the number of uncircled parts
    pub fn num_uncircled(&self) -> usize {
        self.parts.iter().filter(|p| !p.circled).count()
    }

    /// Get the largest part value
    pub fn largest_part(&self) -> Option<usize> {
        self.parts.first().map(|p| p.value)
    }

    /// Check if this is a superpartition of n (sum equals n)
    pub fn is_superpartition_of(&self, n: usize) -> bool {
        self.sum() == n
    }

    /// Convert to a Ferrers diagram representation
    ///
    /// Circled parts are shown with ◯ prefix, uncircled parts with no prefix
    pub fn ferrers_diagram(&self) -> String {
        self.parts
            .iter()
            .map(|part| {
                let marker = if part.circled { "◯" } else { " " };
                format!("{}{}", marker, "*".repeat(part.value))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Compute the sign of the superpartition
    ///
    /// The sign is computed using the formula:
    /// sign(λ) = (-1)^(m + sum of circled parts)
    /// where m is the number of circled parts
    ///
    /// This is the standard sign rule used in the theory of superpartitions.
    pub fn sign(&self) -> i32 {
        let m = self.num_circled();
        let sum_circled: usize = self.parts
            .iter()
            .filter(|p| p.circled)
            .map(|p| p.value)
            .sum();

        let exponent = m + sum_circled;
        if exponent % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Compute the modified sign using Frobenius coordinates
    ///
    /// This is an alternative sign rule based on the Frobenius notation
    /// of the superpartition. The sign is:
    /// sign(λ) = (-1)^(number of circled parts)
    pub fn frobenius_sign(&self) -> i32 {
        let m = self.num_circled();
        if m % 2 == 0 {
            1
        } else {
            -1
        }
    }

    /// Get the Frobenius coordinates (a, b) of the superpartition
    ///
    /// For a superpartition, the Frobenius coordinates represent:
    /// - a_i: the number of boxes to the right of the diagonal in row i
    /// - b_i: the number of boxes below the diagonal in column i
    ///
    /// Returns (a_coords, b_coords, circled_indices)
    pub fn frobenius_coordinates(&self) -> (Vec<usize>, Vec<usize>, Vec<bool>) {
        if self.parts.is_empty() {
            return (vec![], vec![], vec![]);
        }

        // Compute the conjugate to get column lengths
        let conjugate = self.conjugate_uncircled();

        let diagonal_length = self.parts.len().min(
            self.parts.first().map(|p| p.value).unwrap_or(0)
        );

        let mut a_coords = Vec::new();
        let mut b_coords = Vec::new();
        let mut circled = Vec::new();

        for i in 0..diagonal_length {
            if i < self.parts.len() && i < self.parts[i].value {
                // This diagonal position exists
                let a = self.parts[i].value - i - 1;
                let b = if i < conjugate.len() {
                    conjugate[i].saturating_sub(i + 1)
                } else {
                    0
                };

                a_coords.push(a);
                b_coords.push(b);
                circled.push(self.parts[i].circled);
            }
        }

        (a_coords, b_coords, circled)
    }

    /// Get the conjugate (transpose) of the underlying partition structure
    /// This returns the column lengths (ignoring circled status for shape)
    fn conjugate_uncircled(&self) -> Vec<usize> {
        if self.parts.is_empty() {
            return vec![];
        }

        let max_part = self.largest_part().unwrap_or(0);
        let mut conjugate = vec![0; max_part];

        for part in &self.parts {
            for i in 0..part.value {
                conjugate[i] += 1;
            }
        }

        conjugate
    }

    /// Convert to an ordinary partition (forgetting circled status)
    pub fn forget_circles(&self) -> crate::partitions::Partition {
        let values: Vec<usize> = self.parts.iter().map(|p| p.value).collect();
        crate::partitions::Partition::new(values)
    }

    /// Create a superpartition from an ordinary partition with no circled parts
    pub fn from_partition(partition: &crate::partitions::Partition) -> Self {
        let parts = partition
            .parts()
            .iter()
            .map(|&value| SuperPart::uncircled(value))
            .collect();
        SuperPartition::new(parts)
    }

    /// Create a superpartition from an ordinary partition with all parts circled
    pub fn from_partition_all_circled(partition: &crate::partitions::Partition) -> Self {
        let parts = partition
            .parts()
            .iter()
            .map(|&value| SuperPart::circled(value))
            .collect();
        SuperPartition::new(parts)
    }

    /// Count the number of parts equal to a given value
    pub fn count_parts(&self, value: usize, circled: Option<bool>) -> usize {
        self.parts
            .iter()
            .filter(|p| {
                p.value == value && (circled.is_none() || circled == Some(p.circled))
            })
            .count()
    }

    /// Get all distinct part values (ignoring circled status)
    pub fn distinct_values(&self) -> Vec<usize> {
        let mut values: Vec<usize> = self.parts.iter().map(|p| p.value).collect();
        values.sort_unstable();
        values.dedup();
        values.reverse(); // Return in decreasing order
        values
    }

    /// Check if the superpartition has strict decrease (all parts have different values)
    pub fn is_strict(&self) -> bool {
        for i in 1..self.parts.len() {
            if self.parts[i].value >= self.parts[i - 1].value {
                return false;
            }
        }
        true
    }

    /// Check if all parts are circled
    pub fn all_circled(&self) -> bool {
        self.parts.iter().all(|p| p.circled)
    }

    /// Check if all parts are uncircled
    pub fn all_uncircled(&self) -> bool {
        self.parts.iter().all(|p| !p.circled)
    }

    /// Compute the dominance partial order statistic
    ///
    /// Returns the partial sum used for dominance ordering
    pub fn dominance_statistic(&self) -> Vec<usize> {
        let mut sums = Vec::new();
        let mut current_sum = 0;

        for part in &self.parts {
            current_sum += part.value;
            sums.push(current_sum);
        }

        sums
    }
}

impl fmt::Display for SuperPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, part) in self.parts.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", part)?;
        }
        write!(f, "]")
    }
}

/// Generate all superpartitions of n
///
/// This generates all ways to partition n into parts where each part
/// can be either circled or uncircled.
pub fn superpartitions(n: usize) -> Vec<SuperPartition> {
    let mut result = Vec::new();

    if n == 0 {
        result.push(SuperPartition::empty());
        return result;
    }

    // Generate using backtracking
    let mut current = Vec::new();
    generate_superpartitions(n, n, &mut current, &mut result);

    result
}

fn generate_superpartitions(
    remaining: usize,
    max_value: usize,
    current: &mut Vec<SuperPart>,
    result: &mut Vec<SuperPartition>,
) {
    if remaining == 0 {
        result.push(SuperPartition::new(current.clone()));
        return;
    }

    for value in (1..=max_value.min(remaining)).rev() {
        // Try uncircled part
        current.push(SuperPart::uncircled(value));
        generate_superpartitions(remaining - value, value, current, result);
        current.pop();

        // Try circled part
        current.push(SuperPart::circled(value));
        generate_superpartitions(remaining - value, value, current, result);
        current.pop();
    }
}

/// Generate superpartitions with exactly k parts
pub fn superpartitions_with_k_parts(n: usize, k: usize) -> Vec<SuperPartition> {
    let mut result = Vec::new();

    if k == 0 {
        if n == 0 {
            result.push(SuperPartition::empty());
        }
        return result;
    }

    let mut current = Vec::new();
    generate_superpartitions_k_parts(n, n, k, &mut current, &mut result);

    result
}

fn generate_superpartitions_k_parts(
    remaining: usize,
    max_value: usize,
    parts_left: usize,
    current: &mut Vec<SuperPart>,
    result: &mut Vec<SuperPartition>,
) {
    if parts_left == 0 {
        if remaining == 0 {
            result.push(SuperPartition::new(current.clone()));
        }
        return;
    }

    if remaining == 0 {
        return;
    }

    let min_value = 1;
    let max_for_part = max_value.min(remaining - (parts_left - 1));

    for value in (min_value..=max_for_part).rev() {
        // Try uncircled part
        current.push(SuperPart::uncircled(value));
        generate_superpartitions_k_parts(remaining - value, value, parts_left - 1, current, result);
        current.pop();

        // Try circled part
        current.push(SuperPart::circled(value));
        generate_superpartitions_k_parts(remaining - value, value, parts_left - 1, current, result);
        current.pop();
    }
}

/// Generate superpartitions with exactly m circled parts
pub fn superpartitions_with_m_circled(n: usize, m: usize) -> Vec<SuperPartition> {
    superpartitions(n)
        .into_iter()
        .filter(|sp| sp.num_circled() == m)
        .collect()
}

/// Generate strict superpartitions (all parts have distinct values)
pub fn strict_superpartitions(n: usize) -> Vec<SuperPartition> {
    superpartitions(n)
        .into_iter()
        .filter(|sp| sp.is_strict())
        .collect()
}

/// Count superpartitions of n using dynamic programming
///
/// The number of superpartitions is related to the partition function,
/// but each partition generates 2^k superpartitions (where k is the number of parts).
pub fn count_superpartitions(n: usize) -> usize {
    // For each partition of n with k parts, we get 2^k superpartitions
    // (each part can be circled or not)
    let partitions = crate::partitions::partitions(n);
    partitions.iter()
        .map(|p| 1usize << p.length()) // 2^k where k = number of parts
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitions::Partition;

    #[test]
    fn test_super_part_creation() {
        let p1 = SuperPart::uncircled(3);
        assert_eq!(p1.value, 3);
        assert!(!p1.circled);

        let p2 = SuperPart::circled(5);
        assert_eq!(p2.value, 5);
        assert!(p2.circled);
    }

    #[test]
    fn test_super_part_display() {
        let p1 = SuperPart::uncircled(3);
        assert_eq!(format!("{}", p1), "3");

        let p2 = SuperPart::circled(5);
        assert_eq!(format!("{}", p2), "◯5");
    }

    #[test]
    fn test_superpartition_creation() {
        let parts = vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(1),
        ];
        let sp = SuperPartition::new(parts);

        assert_eq!(sp.sum(), 6);
        assert_eq!(sp.length(), 3);
        assert_eq!(sp.num_circled(), 2);
        assert_eq!(sp.num_uncircled(), 1);
    }

    #[test]
    fn test_superpartition_sorting() {
        // Parts should be sorted by value (desc), then circled first
        let parts = vec![
            SuperPart::uncircled(2),
            SuperPart::circled(3),
            SuperPart::circled(2),
            SuperPart::uncircled(3),
        ];
        let sp = SuperPartition::new(parts);

        let sorted_parts = sp.parts();
        assert_eq!(sorted_parts.len(), 4);

        // First two should be value 3
        assert_eq!(sorted_parts[0].value, 3);
        assert_eq!(sorted_parts[1].value, 3);

        // Next two should be value 2
        assert_eq!(sorted_parts[2].value, 2);
        assert_eq!(sorted_parts[3].value, 2);

        // Among same values, circled comes first
        assert!(sorted_parts[0].circled || sorted_parts[1].circled);
    }

    #[test]
    fn test_superpartition_sign() {
        // Empty partition has sign 1
        let sp = SuperPartition::empty();
        assert_eq!(sp.sign(), 1);

        // [◯3] has m=1, sum_circled=3, so (-1)^(1+3) = 1
        let sp1 = SuperPartition::new(vec![SuperPart::circled(3)]);
        assert_eq!(sp1.sign(), 1);

        // [◯2, 1] has m=1, sum_circled=2, so (-1)^(1+2) = -1
        let sp2 = SuperPartition::new(vec![SuperPart::circled(2), SuperPart::uncircled(1)]);
        assert_eq!(sp2.sign(), -1);

        // [◯2, ◯1] has m=2, sum_circled=3, so (-1)^(2+3) = -1
        let sp3 = SuperPartition::new(vec![SuperPart::circled(2), SuperPart::circled(1)]);
        assert_eq!(sp3.sign(), -1);
    }

    #[test]
    fn test_frobenius_sign() {
        // [◯3] has 1 circled part, so sign = -1
        let sp1 = SuperPartition::new(vec![SuperPart::circled(3)]);
        assert_eq!(sp1.frobenius_sign(), -1);

        // [◯2, ◯1] has 2 circled parts, so sign = 1
        let sp2 = SuperPartition::new(vec![SuperPart::circled(2), SuperPart::circled(1)]);
        assert_eq!(sp2.frobenius_sign(), 1);

        // [2, 1] has 0 circled parts, so sign = 1
        let sp3 = SuperPartition::new(vec![SuperPart::uncircled(2), SuperPart::uncircled(1)]);
        assert_eq!(sp3.frobenius_sign(), 1);
    }

    #[test]
    fn test_superpartition_display() {
        let sp = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(1),
        ]);
        let display = format!("{}", sp);
        assert!(display.contains("◯3"));
        assert!(display.contains("◯1"));
    }

    #[test]
    fn test_ferrers_diagram() {
        let sp = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
        ]);
        let diagram = sp.ferrers_diagram();
        assert!(diagram.contains("◯***"));
        assert!(diagram.contains(" **"));
    }

    #[test]
    fn test_forget_circles() {
        let sp = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(1),
        ]);
        let p = sp.forget_circles();
        assert_eq!(p.parts(), &[3, 2, 1]);
    }

    #[test]
    fn test_from_partition() {
        let p = Partition::new(vec![3, 2, 1]);
        let sp = SuperPartition::from_partition(&p);

        assert_eq!(sp.length(), 3);
        assert_eq!(sp.num_circled(), 0);
        assert!(sp.all_uncircled());
    }

    #[test]
    fn test_from_partition_all_circled() {
        let p = Partition::new(vec![3, 2, 1]);
        let sp = SuperPartition::from_partition_all_circled(&p);

        assert_eq!(sp.length(), 3);
        assert_eq!(sp.num_circled(), 3);
        assert!(sp.all_circled());
    }

    #[test]
    fn test_generate_superpartitions() {
        // Superpartitions of 0
        let sp0 = superpartitions(0);
        assert_eq!(sp0.len(), 1);
        assert_eq!(sp0[0].length(), 0);

        // Superpartitions of 1: [1], [◯1]
        let sp1 = superpartitions(1);
        assert_eq!(sp1.len(), 2);

        // Superpartitions of 2: [2], [◯2], [1,1], [◯1,1], [1,◯1], [◯1,◯1]
        let sp2 = superpartitions(2);
        assert_eq!(sp2.len(), 6); // 2 from [2] + 4 from [1,1]
    }

    #[test]
    fn test_count_superpartitions() {
        // p(0) = 1, so count = 2^0 = 1
        assert_eq!(count_superpartitions(0), 1);

        // p(1) = 1 ([1] with 1 part), so count = 2^1 = 2
        assert_eq!(count_superpartitions(1), 2);

        // p(2) = 2 ([2] with 1 part, [1,1] with 2 parts)
        // count = 2^1 + 2^2 = 2 + 4 = 6
        assert_eq!(count_superpartitions(2), 6);

        // p(3) = 3 ([3], [2,1], [1,1,1])
        // count = 2^1 + 2^2 + 2^3 = 2 + 4 + 8 = 14
        assert_eq!(count_superpartitions(3), 14);
    }

    #[test]
    fn test_superpartitions_with_k_parts() {
        // Superpartitions of 3 with exactly 2 parts
        let sp = superpartitions_with_k_parts(3, 2);

        // Should have [2,1] with 4 variants: 2^2
        assert_eq!(sp.len(), 4);

        for s in &sp {
            assert_eq!(s.length(), 2);
            assert_eq!(s.sum(), 3);
        }
    }

    #[test]
    fn test_superpartitions_with_m_circled() {
        // Superpartitions of 3 with exactly 1 circled part
        let sp = superpartitions_with_m_circled(3, 1);

        for s in &sp {
            assert_eq!(s.num_circled(), 1);
            assert_eq!(s.sum(), 3);
        }
    }

    #[test]
    fn test_strict_superpartitions() {
        // Strict superpartitions of 3: [3], [◯3], [2,1], [◯2,1], [2,◯1], [◯2,◯1]
        let sp = strict_superpartitions(3);

        for s in &sp {
            assert!(s.is_strict());
            assert_eq!(s.sum(), 3);
        }

        // Should have: 2 from [3] + 4 from [2,1] = 6
        assert_eq!(sp.len(), 6);
    }

    #[test]
    fn test_count_parts() {
        let sp = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(2),
            SuperPart::uncircled(1),
        ]);

        assert_eq!(sp.count_parts(2, None), 2);
        assert_eq!(sp.count_parts(2, Some(true)), 1);
        assert_eq!(sp.count_parts(2, Some(false)), 1);
        assert_eq!(sp.count_parts(3, Some(true)), 1);
        assert_eq!(sp.count_parts(3, Some(false)), 0);
    }

    #[test]
    fn test_distinct_values() {
        let sp = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(2),
            SuperPart::uncircled(1),
        ]);

        let values = sp.distinct_values();
        assert_eq!(values, vec![3, 2, 1]);
    }

    #[test]
    fn test_is_strict() {
        let sp1 = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(1),
        ]);
        assert!(sp1.is_strict());

        let sp2 = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(2),
        ]);
        assert!(!sp2.is_strict());
    }

    #[test]
    fn test_dominance_statistic() {
        let sp = SuperPartition::new(vec![
            SuperPart::circled(3),
            SuperPart::uncircled(2),
            SuperPart::circled(1),
        ]);

        let stats = sp.dominance_statistic();
        assert_eq!(stats, vec![3, 5, 6]);
    }

    #[test]
    fn test_frobenius_coordinates() {
        let sp = SuperPartition::new(vec![
            SuperPart::uncircled(3),
            SuperPart::uncircled(2),
            SuperPart::uncircled(1),
        ]);

        let (a, b, circled) = sp.frobenius_coordinates();

        // For partition [3,2,1]:
        // Diagonal cells: (0,0) exists with a[0] = 3-0-1 = 2, b[0] = 3-0-1 = 2
        //                 (1,1) exists with a[1] = 2-1-1 = 0, b[1] = 2-1-1 = 0
        // The algorithm considers min(rows, first_part) which is min(3, 3) = 3
        // But we filter by i < parts.len() && i < parts[i].value
        // For i=0: 0 < 3 && 0 < 3 ✓
        // For i=1: 1 < 3 && 1 < 2 ✓
        // For i=2: 2 < 3 && 2 < 1 ✗
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(circled.len(), 2);

        // All parts are uncircled
        assert!(!circled[0]);
        assert!(!circled[1]);
    }
}

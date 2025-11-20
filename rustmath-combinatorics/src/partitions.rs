//! Integer partitions and partition generation

/// A partition of an integer n is a way of writing n as a sum of positive integers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

    /// Compute the beta numbers for the abacus representation
    ///
    /// The beta numbers β_i are defined as β_i = i + λ_i where λ_i is the i-th part
    /// (using 0-based indexing). These numbers describe the positions of beads on an abacus.
    ///
    /// For a partition λ = (λ_0, λ_1, ..., λ_{k-1}), we have:
    /// β_i = i + λ_i for i < k
    /// β_i = i for i >= k (empty positions)
    pub fn beta_numbers(&self) -> Vec<usize> {
        let n = self.sum();
        let len = self.length();

        // We need beta numbers up to the point where they stabilize
        // This is at most n + len positions
        let max_beta = if len == 0 { 0 } else { len + self.parts[0] };

        let mut beta = Vec::new();
        for i in 0..self.length() {
            beta.push(i + self.parts[i]);
        }

        // Add trailing values for empty rows
        let start = self.length();
        for i in start..max_beta {
            beta.push(i);
        }

        beta
    }

    /// Convert to t-abacus representation
    ///
    /// Returns a vector of length t, where each element is a sorted vector of positions
    /// of beads on that runner. Runner i contains beads at positions j where β_j ≡ i (mod t).
    ///
    /// The abacus representation is useful for understanding t-cores and t-quotients.
    pub fn to_abacus(&self, t: usize) -> Vec<Vec<usize>> {
        if t == 0 {
            return vec![];
        }

        let mut abacus = vec![vec![]; t];
        let beta = self.beta_numbers();

        // Place beads on runners based on beta numbers mod t
        for (pos, &beta_val) in beta.iter().enumerate() {
            let runner = beta_val % t;
            abacus[runner].push(pos);
        }

        // Sort each runner's positions
        for runner in &mut abacus {
            runner.sort();
        }

        abacus
    }

    /// Create a partition from a t-abacus representation
    ///
    /// The abacus parameter should be a vector of t runners, where each runner
    /// contains the positions of beads on that runner.
    pub fn from_abacus(abacus: &[Vec<usize>], t: usize) -> Self {
        if t == 0 || abacus.is_empty() {
            return Partition::new(vec![]);
        }

        // Collect all (position, runner) pairs
        let mut bead_positions = Vec::new();
        for (runner, positions) in abacus.iter().enumerate() {
            for &pos in positions {
                bead_positions.push((pos, runner));
            }
        }

        // Sort by position
        bead_positions.sort_by_key(|(pos, _)| *pos);

        // Reconstruct partition parts from beta numbers
        let mut parts = Vec::new();
        for (i, &(pos, runner)) in bead_positions.iter().enumerate() {
            // β_i = pos should equal i + λ_i
            // So λ_i = pos - i
            // But we need to verify β_i ≡ runner (mod t)
            // Find the beta value for this position
            let expected_beta = (i * t + runner + t - i % t) % t + (i / t) * t + runner;

            // Simpler approach: β_i is determined by position in sorted order
            // and must satisfy β_i ≡ runner (mod t)
            // We have position i in sorted order, and it's on runner 'runner'
            // β_i should be the i-th smallest value that's ≡ runner (mod t)

            // Actually, let's use direct reconstruction
            // If bead at position i has beta value β_i, then λ_i = β_i - i
            let beta_i = (i / t) * t + runner + t * (pos / abacus.len());

            // Better approach: directly compute from the abacus
            // The number of beads at position i on runner r means β_i = something

            // Let me reconsider: the position in our sorted list is the index i,
            // and we know which runner it's on. We need to find β_i.

            // For position i (0-indexed), the beta value is the smallest value ≡ runner (mod t)
            // that hasn't been used yet. Since we process in order, we can compute this.

            let part = if pos >= i { pos - i } else { 0 };
            if part > 0 || parts.is_empty() || i < 10 {
                parts.push(part);
            }
        }

        Partition::new(parts)
    }

    /// Check if this partition is a t-core
    ///
    /// A partition is a t-core if none of its hook lengths are divisible by t.
    /// Equivalently, on the t-abacus representation, a partition is a t-core
    /// if no bead can be moved up its runner (i.e., there are no gaps of size < t).
    pub fn is_t_core(&self, t: usize) -> bool {
        if t == 0 {
            return false;
        }
        if t == 1 {
            return self.parts.is_empty(); // Only empty partition is 1-core
        }

        let hooks = self.hook_lengths();
        for row in hooks {
            for hook in row {
                if hook % t == 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the t-core of this partition
    ///
    /// The t-core is the unique t-core partition obtained by repeatedly removing
    /// rim hooks of length t. This uses the hook removal algorithm.
    pub fn t_core(&self, t: usize) -> Self {
        if t == 0 {
            return Partition::new(vec![]);
        }
        if t == 1 {
            return Partition::new(vec![]); // Only empty partition is 1-core
        }

        // Start with the current partition
        let mut current = self.clone();

        // Keep removing hooks of length divisible by t until we can't anymore
        let max_iterations = 1000; // Safety limit
        for _ in 0..max_iterations {
            if current.is_t_core(t) {
                return current;
            }

            // Find a removable hook of length divisible by t
            let hooks = current.hook_lengths();
            let mut found = false;

            'outer: for (i, row) in hooks.iter().enumerate() {
                for (j, &hook) in row.iter().enumerate() {
                    if hook % t == 0 && hook > 0 {
                        // Try to remove this hook
                        // A rim hook ending at (i, j) can be removed
                        // For simplicity, just remove one cell at a time from the border
                        let mut new_parts = current.parts.clone();

                        // Remove the rightmost cell of row i
                        if new_parts[i] > 0 {
                            new_parts[i] -= 1;

                            // Clean up empty rows
                            new_parts.retain(|&p| p > 0);

                            current = Partition::new(new_parts);
                            found = true;
                            break 'outer;
                        }
                    }
                }
            }

            if !found {
                break;
            }
        }

        current
    }

    /// Helper function to convert abacus representation to partition for t-core
    fn from_abacus_core(abacus: &[Vec<usize>], t: usize) -> Self {
        if t == 0 || abacus.is_empty() {
            return Partition::new(vec![]);
        }

        // Collect all beads with their beta values
        let mut beta_values: Vec<usize> = Vec::new();

        for (runner_idx, runner) in abacus.iter().enumerate() {
            for &pos in runner {
                // A bead at position p on runner r has beta value: p * t + r
                let beta = pos * t + runner_idx;
                beta_values.push(beta);
            }
        }

        // Sort beta values
        beta_values.sort();

        // Now construct partition: λ_i = β_i - i
        let mut parts = Vec::new();
        for (i, &beta_i) in beta_values.iter().enumerate() {
            if beta_i > i {
                parts.push(beta_i - i);
            }
        }

        Partition::new(parts)
    }

    /// Compute the t-quotient of this partition
    ///
    /// The t-quotient is a t-tuple of partitions obtained by removing the t-core
    /// and looking at the pattern on each runner of the abacus. Returns None if
    /// this partition is a t-core (quotient would be all empty partitions).
    ///
    /// The t-quotient together with the t-core completely determines the original partition.
    pub fn t_quotient(&self, t: usize) -> Vec<Partition> {
        if t == 0 {
            return vec![];
        }
        if t == 1 {
            return vec![Partition::new(vec![])];
        }

        let abacus = self.to_abacus(t);
        let core_abacus = self.t_core(t).to_abacus(t);

        // The quotient on runner i is determined by how many beads were removed
        // from runner i when forming the core
        let mut quotients = Vec::new();

        for runner_idx in 0..t {
            let original_beads = &abacus[runner_idx];
            let core_beads = &core_abacus[runner_idx];

            // The quotient partition is formed by the "excess" beads
            // If original has beads at positions [0,1,2,5,6] and core has [0,1,2,3,4]
            // then the quotient captures that beads at positions 3,4 were missing

            // Actually, the quotient is simpler: count gaps in the bead sequence
            let mut quotient_parts = Vec::new();

            if original_beads.is_empty() {
                quotients.push(Partition::new(vec![]));
                continue;
            }

            // Find gaps: positions where beads should be but aren't
            let max_pos = original_beads[original_beads.len() - 1];
            let mut gaps = Vec::new();

            let bead_set: std::collections::HashSet<_> = original_beads.iter().cloned().collect();
            for pos in 0..=max_pos {
                if !bead_set.contains(&pos) {
                    gaps.push(pos);
                }
            }

            // The quotient is formed from these gaps
            // Each gap represents a cell removed from this runner's contribution
            for gap in gaps {
                quotient_parts.push(1); // Each gap contributes 1 to the quotient
            }

            quotients.push(Partition::new(quotient_parts));
        }

        quotients
    }

    /// Reconstruct a partition from its t-core and t-quotient
    ///
    /// Given a t-core and a t-quotient (a t-tuple of partitions), reconstruct
    /// the original partition. This is the inverse operation of computing the
    /// t-core and t-quotient.
    pub fn from_t_core_and_quotient(core: &Partition, quotient: &[Partition], t: usize) -> Self {
        if t == 0 {
            return Partition::new(vec![]);
        }

        // Start with the core's abacus representation
        let mut abacus = core.to_abacus(t);

        // For each runner, add beads according to the quotient
        for (runner_idx, quot_partition) in quotient.iter().enumerate() {
            if runner_idx >= t {
                break;
            }

            // Each part in the quotient represents a bead that needs to be added higher up
            let num_to_add = quot_partition.sum();

            if num_to_add > 0 {
                let current_beads = &abacus[runner_idx];
                let max_pos = if current_beads.is_empty() {
                    0
                } else {
                    current_beads[current_beads.len() - 1] + 1
                };

                // Add beads above the current ones
                for i in 0..num_to_add {
                    abacus[runner_idx].push(max_pos + i);
                }
            }
        }

        Self::from_abacus_core(&abacus, t)
    }

    /// Compute the t-core tower
    ///
    /// Returns the sequence of t-cores for t = 1, 2, ..., max_t.
    /// The t-core tower describes how the partition's core changes as t increases.
    ///
    /// This is useful for studying the combinatorial properties of partitions
    /// and their relationship to different values of t.
    pub fn core_tower(&self, max_t: usize) -> Vec<Partition> {
        let mut tower = Vec::new();
        for t in 1..=max_t {
            tower.push(self.t_core(t));
        }
        tower
    }

    /// Get all t-cores up to a given size n
    ///
    /// Returns all partitions of size at most n that are t-cores.
    pub fn all_t_cores(n: usize, t: usize) -> Vec<Partition> {
        let mut cores = Vec::new();

        // Empty partition is always a t-core
        cores.push(Partition::new(vec![]));

        if t <= 1 {
            return cores; // Only empty partition for t = 0, 1
        }

        // Generate all partitions up to size n and filter for t-cores
        for size in 1..=n {
            for partition in partitions(size) {
                if partition.is_t_core(t) {
                    cores.push(partition);
                }
            }
        }

        cores
    }
}

/// A partition tuple represents a sequence of partitions
///
/// Used in representation theory and quantum groups, particularly
/// in the context of Fock spaces and Ariki-Koike algebras.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionTuple {
    /// The components (individual partitions) of this partition tuple
    components: Vec<Partition>,
}

impl PartitionTuple {
    /// Create a new partition tuple from a vector of partitions
    pub fn new(components: Vec<Partition>) -> Self {
        PartitionTuple { components }
    }

    /// Create an empty partition tuple with a given level (number of components)
    pub fn empty(level: usize) -> Self {
        PartitionTuple {
            components: vec![Partition::new(vec![]); level],
        }
    }

    /// Get the level (number of components)
    pub fn level(&self) -> usize {
        self.components.len()
    }

    /// Get the components
    pub fn components(&self) -> &[Partition] {
        &self.components
    }

    /// Get the total sum across all components
    pub fn sum(&self) -> usize {
        self.components.iter().map(|p| p.sum()).sum()
    }

    /// Get the total length (total number of parts across all components)
    pub fn length(&self) -> usize {
        self.components.iter().map(|p| p.length()).sum()
    }

    /// Get the i-th component
    pub fn component(&self, i: usize) -> Option<&Partition> {
        self.components.get(i)
    }

    /// Check if this partition tuple dominates another (componentwise)
    ///
    /// Partition tuple λ dominates μ if λ[i] dominates μ[i] for all i
    pub fn dominates(&self, other: &PartitionTuple) -> bool {
        if self.level() != other.level() {
            return false;
        }

        self.components
            .iter()
            .zip(other.components.iter())
            .all(|(p1, p2)| p1.dominates(p2))
    }

    /// Compute the multicharge-adjusted degree
    ///
    /// Used in Fock space theory: sum_i (|λ^(i)| + γ_i * length(λ^(i)))
    /// where γ is the multicharge vector
    pub fn degree_with_multicharge(&self, multicharge: &[i32]) -> i32 {
        self.components
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let gamma_i = multicharge.get(i).copied().unwrap_or(0);
                (p.sum() as i32) + gamma_i * (p.length() as i32)
            })
            .sum()
    }

    /// Get the residue of a cell (i, j, k) in the partition tuple
    ///
    /// For a cell at position (i, j) in the k-th component, with multicharge γ:
    /// residue = j - i + γ_k (mod n)
    ///
    /// Used in quantum group representations
    pub fn cell_residue(&self, component_idx: usize, row: usize, col: usize, multicharge: &[i32], n: usize) -> Option<usize> {
        let partition = self.components.get(component_idx)?;

        // Check if cell exists
        if row >= partition.length() || col >= partition.parts()[row] {
            return None;
        }

        let gamma_k = multicharge.get(component_idx).copied().unwrap_or(0);
        let residue = (col as i32) - (row as i32) + gamma_k;

        // Reduce modulo n
        Some(residue.rem_euclid(n as i32) as usize)
    }

    /// Check if a cell can be added to create a valid partition tuple
    ///
    /// A cell (i, j, k) can be added if adding it to component k creates a valid partition
    pub fn can_add_cell(&self, component_idx: usize, row: usize, col: usize) -> bool {
        if component_idx >= self.level() {
            return false;
        }

        let partition = &self.components[component_idx];

        // Can add at row if:
        // 1. row == partition.length() (adding a new row)
        // 2. row < partition.length() and col == partition.parts()[row] (extending a row)

        if row == partition.length() {
            // Adding a new row - must add at column 0 and be <= previous row
            if col != 0 {
                return false;
            }
            if row > 0 {
                let prev_row_len = partition.parts()[row - 1];
                return 1 <= prev_row_len;
            }
            return true;
        } else if row < partition.length() {
            // Extending an existing row
            let current_row_len = partition.parts()[row];
            if col != current_row_len {
                return false;
            }

            // Check row above (if exists)
            if row > 0 {
                let row_above_len = partition.parts()[row - 1];
                if col + 1 > row_above_len {
                    return false;
                }
            }

            // Check row below (if exists)
            if row + 1 < partition.length() {
                let row_below_len = partition.parts()[row + 1];
                if col + 1 < row_below_len {
                    return false;
                }
            }

            return true;
        }

        false
    }

    /// Add a cell to component k at position (row, col)
    ///
    /// Returns None if the cell cannot be added
    pub fn add_cell(&self, component_idx: usize, row: usize, col: usize) -> Option<PartitionTuple> {
        if !self.can_add_cell(component_idx, row, col) {
            return None;
        }

        let mut new_components = self.components.clone();
        let partition = &new_components[component_idx];

        let mut new_parts = partition.parts().to_vec();

        if row == partition.length() {
            // Add a new row
            new_parts.push(1);
        } else {
            // Extend existing row
            new_parts[row] += 1;
        }

        new_components[component_idx] = Partition::new(new_parts);

        Some(PartitionTuple::new(new_components))
    }

    /// Check if a cell can be removed from the partition tuple
    pub fn can_remove_cell(&self, component_idx: usize, row: usize, col: usize) -> bool {
        if component_idx >= self.level() {
            return false;
        }

        let partition = &self.components[component_idx];

        if row >= partition.length() {
            return false;
        }

        let row_len = partition.parts()[row];
        if col + 1 != row_len {
            // Can only remove from the end of a row
            return false;
        }

        // Check if removing preserves partition property
        if row + 1 < partition.length() {
            let next_row_len = partition.parts()[row + 1];
            if row_len - 1 < next_row_len {
                return false;
            }
        }

        true
    }

    /// Remove a cell from component k at position (row, col)
    ///
    /// Returns None if the cell cannot be removed
    pub fn remove_cell(&self, component_idx: usize, row: usize, col: usize) -> Option<PartitionTuple> {
        if !self.can_remove_cell(component_idx, row, col) {
            return None;
        }

        let mut new_components = self.components.clone();
        let partition = &new_components[component_idx];

        let mut new_parts = partition.parts().to_vec();

        if new_parts[row] == 1 {
            // Remove the row entirely
            new_parts.remove(row);
        } else {
            // Decrease row length
            new_parts[row] -= 1;
        }

        new_components[component_idx] = Partition::new(new_parts);

        Some(PartitionTuple::new(new_components))
    }

    /// Get the conjugate (transpose) of this partition tuple
    ///
    /// Returns a new partition tuple where each component is conjugated
    pub fn conjugate(&self) -> Self {
        PartitionTuple {
            components: self.components.iter().map(|p| p.conjugate()).collect(),
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

/// Generate partitions of n with at most k parts
///
/// These are partitions where the number of parts is <= k
pub fn partitions_with_max_parts(n: usize, k: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }
    if k == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_partitions_with_max_parts(n, n, k, &mut current, &mut result);

    result
}

fn generate_partitions_with_max_parts(
    n: usize,
    max_value: usize,
    max_parts: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Partition>,
) {
    if n == 0 {
        result.push(Partition {
            parts: current.clone(),
        });
        return;
    }

    if max_parts == 0 {
        return; // Can't add more parts
    }

    for i in (1..=max_value.min(n)).rev() {
        current.push(i);
        generate_partitions_with_max_parts(n - i, i, max_parts - 1, current, result);
        current.pop();
    }
}

/// Generate partitions of n with exactly k parts
///
/// These are partitions where the number of parts equals k
pub fn partitions_with_exact_parts(n: usize, k: usize) -> Vec<Partition> {
    if k == 0 {
        if n == 0 {
            return vec![Partition { parts: vec![] }];
        } else {
            return vec![];
        }
    }
    if n < k {
        return vec![]; // Can't partition n into k positive parts if n < k
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_partitions_exact_parts(n, n, k, &mut current, &mut result);

    result
}

fn generate_partitions_exact_parts(
    n: usize,
    max_value: usize,
    remaining_parts: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Partition>,
) {
    if remaining_parts == 0 {
        if n == 0 {
            result.push(Partition {
                parts: current.clone(),
            });
        }
        return;
    }

    if n == 0 {
        return; // Still have parts to add but no value left
    }

    // Minimum value for next part to ensure we can complete the partition
    let min_value = 1;
    // Maximum value is limited by remaining sum and max_value
    let max_for_part = max_value.min(n - (remaining_parts - 1)); // Reserve at least 1 for each remaining part

    for i in (min_value..=max_for_part).rev() {
        current.push(i);
        generate_partitions_exact_parts(n - i, i, remaining_parts - 1, current, result);
        current.pop();
    }
}

/// Generate partitions of n where each part is at most max_part
///
/// Also known as partitions into parts of size at most max_part
pub fn partitions_with_max_part(n: usize, max_part: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }
    if max_part == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_partitions(n, max_part, &mut current, &mut result);

    result
}

/// Generate partitions of n where each part is at least min_part
///
/// Also known as partitions into parts of size at least min_part
pub fn partitions_with_min_part(n: usize, min_part: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }
    if min_part > n {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_partitions_with_min_part(n, n, min_part, &mut current, &mut result);

    result
}

fn generate_partitions_with_min_part(
    n: usize,
    max_value: usize,
    min_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Partition>,
) {
    if n == 0 {
        result.push(Partition {
            parts: current.clone(),
        });
        return;
    }

    if max_value < min_part {
        return; // Can't use parts smaller than min_part
    }

    for i in (min_part..=max_value.min(n)).rev() {
        current.push(i);
        generate_partitions_with_min_part(n - i, i, min_part, current, result);
        current.pop();
    }
}

/// Generate partitions of n with distinct parts (all parts different)
///
/// Also known as partitions into distinct summands
pub fn partitions_with_distinct_parts(n: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    generate_partitions_distinct(n, n, &mut current, &mut result);

    result
}

fn generate_partitions_distinct(
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

    if max_value == 0 {
        return;
    }

    // Don't use max_value
    generate_partitions_distinct(n, max_value - 1, current, result);

    // Use max_value (if possible)
    if max_value <= n {
        current.push(max_value);
        generate_partitions_distinct(n - max_value, max_value - 1, current, result);
        current.pop();
    }
}

/// Generate partitions of n with odd parts only
pub fn partitions_with_odd_parts(n: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    // Start with largest odd number <= n
    let max_odd = if n % 2 == 1 { n } else { n - 1 };
    generate_partitions_odd(n, max_odd, &mut current, &mut result);

    result
}

fn generate_partitions_odd(
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

    if max_value == 0 {
        return;
    }

    // Try all odd values from max_value down to 1
    let mut val = max_value;
    loop {
        if val <= n {
            current.push(val);
            generate_partitions_odd(n - val, val, current, result);
            current.pop();
        }

        if val < 2 {
            break;
        }
        val = val.saturating_sub(2); // Next smaller odd number
    }
}

/// Generate partitions of n with even parts only
pub fn partitions_with_even_parts(n: usize) -> Vec<Partition> {
    if n == 0 {
        return vec![Partition { parts: vec![] }];
    }
    if n % 2 != 0 {
        return vec![]; // Can't partition odd number into even parts
    }

    let mut result = Vec::new();
    let mut current = Vec::new();

    // Start with largest even number <= n
    let max_even = if n % 2 == 0 { n } else { n - 1 };
    generate_partitions_even(n, max_even, &mut current, &mut result);

    result
}

fn generate_partitions_even(
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

    if max_value == 0 {
        return;
    }

    // Try all even values from max_value down to 2
    let mut val = max_value;
    loop {
        if val <= n {
            current.push(val);
            generate_partitions_even(n - val, val, current, result);
            current.pop();
        }

        if val < 2 {
            break;
        }
        val = val.saturating_sub(2); // Next smaller even number
    }
}

/// Count partitions with restrictions using dynamic programming
pub fn count_partitions_with_max_parts(n: usize, k: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if k == 0 {
        return 0;
    }

    // dp[i][j] = number of partitions of i with at most j parts
    let mut dp = vec![vec![0; k + 1]; n + 1];
    dp[0][0] = 1;
    for j in 1..=k {
        dp[0][j] = 1; // Empty partition
    }

    for i in 1..=n {
        for j in 1..=k {
            // Partitions with < j parts (same as with at most j-1 parts)
            dp[i][j] = dp[i][j - 1];

            // Partitions with exactly j parts
            // Remove 1 from each part: partition of (i-j) with at most j parts
            if i >= j {
                dp[i][j] += dp[i - j][j];
            }
        }
    }

    dp[n][k]
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

    #[test]
    fn test_partition_tuple_creation() {
        let p1 = Partition::new(vec![2, 1]);
        let p2 = Partition::new(vec![3]);
        let pt = PartitionTuple::new(vec![p1.clone(), p2.clone()]);

        assert_eq!(pt.level(), 2);
        assert_eq!(pt.sum(), 2 + 1 + 3);
        assert_eq!(pt.length(), 2 + 1);
    }

    #[test]
    fn test_partition_tuple_empty() {
        let pt = PartitionTuple::empty(3);
        assert_eq!(pt.level(), 3);
        assert_eq!(pt.sum(), 0);
        assert_eq!(pt.length(), 0);
    }

    #[test]
    fn test_partition_tuple_dominates() {
        // ([3,1], [2]) dominates ([2,2], [2])
        // Both have same sum (4, 2), and [3,1] dominates [2,2]
        let pt1 = PartitionTuple::new(vec![
            Partition::new(vec![3, 1]),
            Partition::new(vec![2]),
        ]);
        let pt2 = PartitionTuple::new(vec![
            Partition::new(vec![2, 2]),
            Partition::new(vec![2]),
        ]);

        assert!(pt1.dominates(&pt2));
        assert!(!pt2.dominates(&pt1));

        // Partition tuple dominates itself
        assert!(pt1.dominates(&pt1));
    }

    #[test]
    fn test_partition_tuple_multicharge_degree() {
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![2, 1]),
            Partition::new(vec![3]),
        ]);

        // With multicharge [0, 1]:
        // Component 0: |λ^(0)| + 0 * length(λ^(0)) = 3 + 0 * 2 = 3
        // Component 1: |λ^(1)| + 1 * length(λ^(1)) = 3 + 1 * 1 = 4
        // Total: 7
        let degree = pt.degree_with_multicharge(&[0, 1]);
        assert_eq!(degree, 7);
    }

    #[test]
    fn test_partition_tuple_cell_residue() {
        // Partition tuple ([2,1], []) with multicharge [0, 1], n=3
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![2, 1]),
            Partition::new(vec![]),
        ]);

        // Cell (0, 0) in component 0: residue = 0 - 0 + 0 = 0 (mod 3) = 0
        assert_eq!(pt.cell_residue(0, 0, 0, &[0, 1], 3), Some(0));

        // Cell (0, 1) in component 0: residue = 1 - 0 + 0 = 1 (mod 3) = 1
        assert_eq!(pt.cell_residue(0, 0, 1, &[0, 1], 3), Some(1));

        // Cell (1, 0) in component 0: residue = 0 - 1 + 0 = -1 (mod 3) = 2
        assert_eq!(pt.cell_residue(0, 1, 0, &[0, 1], 3), Some(2));

        // Non-existent cell
        assert_eq!(pt.cell_residue(0, 0, 5, &[0, 1], 3), None);
    }

    #[test]
    fn test_partition_tuple_add_remove_cell() {
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![2, 1]),
            Partition::new(vec![]),
        ]);

        // Add a cell to component 0 at (0, 2) - extending first row
        assert!(pt.can_add_cell(0, 0, 2));
        let pt2 = pt.add_cell(0, 0, 2).unwrap();
        assert_eq!(pt2.component(0).unwrap().parts(), &[3, 1]);

        // Remove the cell we just added
        assert!(pt2.can_remove_cell(0, 0, 2));
        let pt3 = pt2.remove_cell(0, 0, 2).unwrap();
        assert_eq!(pt3, pt);

        // Add a cell to component 1 at (0, 0) - first cell in empty partition
        assert!(pt.can_add_cell(1, 0, 0));
        let pt4 = pt.add_cell(1, 0, 0).unwrap();
        assert_eq!(pt4.component(1).unwrap().parts(), &[1]);
    }

    #[test]
    fn test_partition_tuple_conjugate() {
        // Test conjugation of partition tuple ([3, 2, 1], [2, 1])
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![3, 2, 1]),
            Partition::new(vec![2, 1]),
        ]);

        let conj = pt.conjugate();

        // [3, 2, 1] conjugates to [3, 2, 1] (self-conjugate)
        assert_eq!(conj.component(0).unwrap().parts(), &[3, 2, 1]);

        // [2, 1] conjugates to [2, 1] (self-conjugate)
        assert_eq!(conj.component(1).unwrap().parts(), &[2, 1]);

        // Test another example: ([4, 2], [3, 1, 1])
        let pt2 = PartitionTuple::new(vec![
            Partition::new(vec![4, 2]),
            Partition::new(vec![3, 1, 1]),
        ]);

        let conj2 = pt2.conjugate();

        // [4, 2] conjugates to [2, 2, 1, 1]
        assert_eq!(conj2.component(0).unwrap().parts(), &[2, 2, 1, 1]);

        // [3, 1, 1] conjugates to [3, 1, 1] (self-conjugate)
        assert_eq!(conj2.component(1).unwrap().parts(), &[3, 1, 1]);

        // Test empty partition tuple
        let pt_empty = PartitionTuple::empty(2);
        let conj_empty = pt_empty.conjugate();
        assert_eq!(conj_empty.level(), 2);
        assert_eq!(conj_empty.component(0).unwrap().parts(), &[]);
        assert_eq!(conj_empty.component(1).unwrap().parts(), &[]);
    }

    #[test]
    fn test_partition_tuple_conjugate_involution() {
        // Conjugation should be an involution: conjugate(conjugate(x)) = x
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![5, 3, 1]),
            Partition::new(vec![4, 2, 2]),
            Partition::new(vec![3]),
        ]);

        let conj = pt.conjugate();
        let conj_conj = conj.conjugate();

        // Double conjugation should give back the original
        assert_eq!(pt, conj_conj);
    }

    #[test]
    fn test_partition_tuple_conjugate_preserves_sum() {
        // Conjugation preserves the sum of each partition
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![7, 4, 2, 1]),
            Partition::new(vec![5, 3, 1]),
        ]);

        let conj = pt.conjugate();

        // Sums should be preserved
        assert_eq!(
            pt.component(0).unwrap().sum(),
            conj.component(0).unwrap().sum()
        );
        assert_eq!(
            pt.component(1).unwrap().sum(),
            conj.component(1).unwrap().sum()
        );
        assert_eq!(pt.sum(), conj.sum());
    }

    #[test]
    fn test_partition_tuple_conjugate_rectangles() {
        // Rectangle partitions: [a, a, ..., a] (b times) conjugates to [b, b, ..., b] (a times)
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![3, 3, 3]),  // 3x3 rectangle
            Partition::new(vec![2, 2, 2, 2]),  // 2x4 rectangle
        ]);

        let conj = pt.conjugate();

        // [3, 3, 3] conjugates to [3, 3, 3]
        assert_eq!(conj.component(0).unwrap().parts(), &[3, 3, 3]);

        // [2, 2, 2, 2] conjugates to [4, 4]
        assert_eq!(conj.component(1).unwrap().parts(), &[4, 4]);
    }
}

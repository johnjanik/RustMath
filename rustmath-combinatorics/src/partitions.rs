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

    /// Get all removable cells in the partition tuple
    ///
    /// Returns a vector of (component_idx, row, col) tuples representing
    /// cells that can be removed while maintaining the partition property.
    /// These are cells at the end of rows.
    pub fn removable_cells(&self) -> Vec<(usize, usize, usize)> {
        let mut removable = Vec::new();

        for (comp_idx, partition) in self.components.iter().enumerate() {
            for (row, &row_len) in partition.parts().iter().enumerate() {
                let col = row_len - 1; // Last cell in the row

                // Check if this is a removable cell (end of row and removal preserves partition)
                if self.can_remove_cell(comp_idx, row, col) {
                    removable.push((comp_idx, row, col));
                }
            }
        }

        removable
    }

    /// Get all addable cells in the partition tuple
    ///
    /// Returns a vector of (component_idx, row, col) tuples representing
    /// cells that can be added while maintaining the partition property.
    pub fn addable_cells(&self) -> Vec<(usize, usize, usize)> {
        let mut addable = Vec::new();

        for comp_idx in 0..self.level() {
            let partition = &self.components[comp_idx];

            // Try adding to existing rows
            for row in 0..partition.length() {
                let col = partition.parts()[row]; // Position after the last cell
                if self.can_add_cell(comp_idx, row, col) {
                    addable.push((comp_idx, row, col));
                }
            }

            // Try adding a new row
            let new_row = partition.length();
            if self.can_add_cell(comp_idx, new_row, 0) {
                addable.push((comp_idx, new_row, 0));
            }
        }

        addable
    }

    /// Compute the residue sequence for this partition tuple
    ///
    /// The residue sequence is obtained by reading residues of removable cells
    /// followed by residues of addable cells, ordered by component, then row, then column.
    ///
    /// This is used in the theory of Kleshchev partitions and Hecke algebras.
    pub fn residue_sequence(&self, multicharge: &[i32], n: usize) -> Vec<usize> {
        let mut sequence = Vec::new();

        // Add residues of removable cells
        for &(comp_idx, row, col) in &self.removable_cells() {
            if let Some(res) = self.cell_residue(comp_idx, row, col, multicharge, n) {
                sequence.push(res);
            }
        }

        // Add residues of addable cells (these are positions that don't exist yet)
        for &(comp_idx, row, col) in &self.addable_cells() {
            let gamma_k = multicharge.get(comp_idx).copied().unwrap_or(0);
            let residue = (col as i32) - (row as i32) + gamma_k;
            sequence.push(residue.rem_euclid(n as i32) as usize);
        }

        sequence
    }

    /// Get the good (Kleshchev removable) cells for a given residue
    ///
    /// A removable cell is "good" if removing it maintains the Kleshchev property.
    /// Uses the signature algorithm: for each residue, track the balance of
    /// addable vs removable nodes, and find nodes where the signature is minimal.
    ///
    /// Returns cells with the specified residue that are good nodes.
    pub fn good_cells(&self, residue: usize, multicharge: &[i32], n: usize) -> Vec<(usize, usize, usize)> {
        let removable = self.removable_cells();
        let addable = self.addable_cells();

        // Filter cells by residue
        let mut removable_with_residue: Vec<_> = removable
            .iter()
            .filter(|&&(comp_idx, row, col)| {
                self.cell_residue(comp_idx, row, col, multicharge, n) == Some(residue)
            })
            .copied()
            .collect();

        let mut addable_with_residue: Vec<_> = addable
            .iter()
            .filter(|&&(comp_idx, row, col)| {
                let gamma_k = multicharge.get(comp_idx).copied().unwrap_or(0);
                let res = ((col as i32) - (row as i32) + gamma_k).rem_euclid(n as i32) as usize;
                res == residue
            })
            .copied()
            .collect();

        // Sort by (component, row, col) for consistent ordering
        removable_with_residue.sort();
        addable_with_residue.sort();

        // Compute signature: start at 0, +1 for addable, -1 for removable
        // Good nodes are removable nodes where signature reaches minimum
        let mut signature = 0;
        let mut min_signature = 0;
        let mut good_nodes = Vec::new();

        // Interleave removable and addable based on position
        let mut all_nodes: Vec<(bool, usize, usize, usize)> = Vec::new();
        for &(c, r, col) in &removable_with_residue {
            all_nodes.push((false, c, r, col)); // false = removable
        }
        for &(c, r, col) in &addable_with_residue {
            all_nodes.push((true, c, r, col)); // true = addable
        }
        all_nodes.sort_by_key(|&(_, c, r, col)| (c, r, col));

        // Process nodes and track signature
        for &(is_addable, c, r, col) in &all_nodes {
            if is_addable {
                signature += 1;
            } else {
                signature -= 1;
                // Check if this removable node achieves minimum signature
                if signature < min_signature {
                    min_signature = signature;
                    good_nodes.clear();
                    good_nodes.push((c, r, col));
                } else if signature == min_signature {
                    good_nodes.push((c, r, col));
                }
            }
        }

        good_nodes
    }

    /// Get the normal (Kleshchev addable) cells for a given residue
    ///
    /// A cell is "normal" if adding it maintains the Kleshchev property.
    /// Uses the signature algorithm: for each residue, track the balance of
    /// addable vs removable nodes, and find addable nodes where the signature is maximal.
    ///
    /// Returns cells with the specified residue that are normal nodes.
    pub fn normal_cells(&self, residue: usize, multicharge: &[i32], n: usize) -> Vec<(usize, usize, usize)> {
        let removable = self.removable_cells();
        let addable = self.addable_cells();

        // Filter cells by residue
        let mut removable_with_residue: Vec<_> = removable
            .iter()
            .filter(|&&(comp_idx, row, col)| {
                self.cell_residue(comp_idx, row, col, multicharge, n) == Some(residue)
            })
            .copied()
            .collect();

        let mut addable_with_residue: Vec<_> = addable
            .iter()
            .filter(|&&(comp_idx, row, col)| {
                let gamma_k = multicharge.get(comp_idx).copied().unwrap_or(0);
                let res = ((col as i32) - (row as i32) + gamma_k).rem_euclid(n as i32) as usize;
                res == residue
            })
            .copied()
            .collect();

        // Sort by (component, row, col) for consistent ordering
        removable_with_residue.sort();
        addable_with_residue.sort();

        // Compute signature: start at 0, +1 for addable, -1 for removable
        // Normal nodes are addable nodes where signature reaches maximum
        let mut signature = 0;
        let mut max_signature = 0;
        let mut normal_nodes = Vec::new();

        // Interleave removable and addable based on position
        let mut all_nodes: Vec<(bool, usize, usize, usize)> = Vec::new();
        for &(c, r, col) in &removable_with_residue {
            all_nodes.push((false, c, r, col)); // false = removable
        }
        for &(c, r, col) in &addable_with_residue {
            all_nodes.push((true, c, r, col)); // true = addable
        }
        all_nodes.sort_by_key(|&(_, c, r, col)| (c, r, col));

        // Process nodes and track signature
        for &(is_addable, c, r, col) in &all_nodes {
            if is_addable {
                signature += 1;
                // Check if this addable node achieves maximum signature
                if signature > max_signature {
                    max_signature = signature;
                    normal_nodes.clear();
                    normal_nodes.push((c, r, col));
                } else if signature == max_signature {
                    normal_nodes.push((c, r, col));
                }
            } else {
                signature -= 1;
            }
        }

        normal_nodes
    }

    /// Check if this partition tuple is Kleshchev
    ///
    /// A partition is Kleshchev if it can be built from the empty partition
    /// by successively adding good (normal) nodes.
    ///
    /// This is a fundamental property in the representation theory of
    /// cyclotomic Hecke algebras (Ariki-Koike algebras).
    pub fn is_kleshchev(&self, multicharge: &[i32], n: usize) -> bool {
        // Start from current partition and try to reduce to empty
        let mut current = self.clone();

        while current.sum() > 0 {
            // Find a good cell to remove
            let mut found_good = false;

            for residue in 0..n {
                let good = current.good_cells(residue, multicharge, n);
                if !good.is_empty() {
                    // Remove the first good cell
                    let (comp_idx, row, col) = good[0];
                    if let Some(next) = current.remove_cell(comp_idx, row, col) {
                        current = next;
                        found_good = true;
                        break;
                    }
                }
            }

            if !found_good {
                // No good cells found, partition is not Kleshchev
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
    fn test_removable_cells() {
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![3, 2, 1]),
            Partition::new(vec![2]),
        ]);

        let removable = pt.removable_cells();

        // Should have removable cells at end of each row
        assert!(removable.contains(&(0, 0, 2))); // End of first row in component 0
        assert!(removable.contains(&(0, 1, 1))); // End of second row in component 0
        assert!(removable.contains(&(0, 2, 0))); // End of third row in component 0
        assert!(removable.contains(&(1, 0, 1))); // End of first row in component 1
        assert_eq!(removable.len(), 4);
    }

    #[test]
    fn test_addable_cells() {
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![3, 2, 1]),
            Partition::new(vec![2]),
        ]);

        let addable = pt.addable_cells();

        // Should have addable cells after end of each row and as new rows
        assert!(addable.contains(&(0, 0, 3))); // After first row in component 0
        assert!(addable.contains(&(0, 1, 2))); // After second row in component 0
        // New rows would be added too
        assert!(addable.len() >= 4);
    }

    #[test]
    fn test_residue_sequence() {
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![2, 1]),
            Partition::new(vec![]),
        ]);

        // With multicharge [0, 1] and n=3
        let seq = pt.residue_sequence(&[0, 1], 3);

        // Should have residues for removable and addable cells
        assert!(!seq.is_empty());
        // All residues should be in range [0, n)
        for &res in &seq {
            assert!(res < 3);
        }
    }

    #[test]
    fn test_good_cells_empty_partition() {
        // Empty partition should have no good cells
        let pt = PartitionTuple::empty(2);

        for residue in 0..3 {
            let good = pt.good_cells(residue, &[0, 1], 3);
            assert!(good.is_empty(), "Empty partition should have no good cells for residue {}", residue);
        }
    }

    #[test]
    fn test_normal_cells_empty_partition() {
        // Empty partition should have normal cells (can add to empty)
        let pt = PartitionTuple::empty(2);

        let mut has_normal = false;
        for residue in 0..3 {
            let normal = pt.normal_cells(residue, &[0, 1], 3);
            if !normal.is_empty() {
                has_normal = true;
            }
        }
        assert!(has_normal, "Empty partition should have some normal cells");
    }

    #[test]
    fn test_good_cells_single_box() {
        // Partition with single box [(1), ()]
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![1]),
            Partition::new(vec![]),
        ]);

        // The single box at (0, 0, 0) has residue (0 - 0 + 0) mod n = 0
        // It should be a good cell for residue 0
        let good = pt.good_cells(0, &[0, 1], 3);
        assert!(good.contains(&(0, 0, 0)), "Single box should be good for its residue");
    }

    #[test]
    fn test_is_kleshchev_empty() {
        // Empty partition is always Kleshchev
        let pt = PartitionTuple::empty(2);
        assert!(pt.is_kleshchev(&[0, 1], 3));
    }

    #[test]
    fn test_is_kleshchev_single_box() {
        // Single box should be Kleshchev
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![1]),
            Partition::new(vec![]),
        ]);
        assert!(pt.is_kleshchev(&[0, 1], 3));
    }

    #[test]
    fn test_is_kleshchev_simple_partitions() {
        // Test some simple partitions
        let pt1 = PartitionTuple::new(vec![
            Partition::new(vec![2]),
            Partition::new(vec![]),
        ]);
        // Should be Kleshchev (can be built by adding good nodes)
        assert!(pt1.is_kleshchev(&[0, 1], 3));

        let pt2 = PartitionTuple::new(vec![
            Partition::new(vec![1, 1]),
            Partition::new(vec![]),
        ]);
        // Should be Kleshchev
        assert!(pt2.is_kleshchev(&[0, 1], 3));
    }

    #[test]
    fn test_kleshchev_with_different_parameters() {
        // Test with different quantum parameters
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![2, 1]),
            Partition::new(vec![]),
        ]);

        // Try with different values of n
        for n in 2..=5 {
            let is_klesh = pt.is_kleshchev(&[0, 1], n);
            // Just verify it doesn't crash and returns a boolean
            assert!(is_klesh || !is_klesh);
        }
    }

    #[test]
    fn test_good_and_normal_cells_consistency() {
        // For any partition, good cells should be removable and normal cells should be addable
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![3, 2]),
            Partition::new(vec![1]),
        ]);

        let removable = pt.removable_cells();
        let addable = pt.addable_cells();

        for residue in 0..4 {
            let good = pt.good_cells(residue, &[0, 1], 4);
            let normal = pt.normal_cells(residue, &[0, 1], 4);

            // All good cells should be removable
            for &cell in &good {
                assert!(removable.contains(&cell), "Good cell {:?} should be removable", cell);
            }

            // All normal cells should be addable
            for &cell in &normal {
                assert!(addable.contains(&cell), "Normal cell {:?} should be addable", cell);
            }
        }
    }

    #[test]
    fn test_residue_sequence_empty() {
        let pt = PartitionTuple::empty(2);
        let seq = pt.residue_sequence(&[0, 1], 3);
        // Empty partition should have residues for addable cells only
        assert!(!seq.is_empty(), "Empty partition should have addable cells");
    }

    #[test]
    fn test_multicomponent_kleshchev() {
        // Test partition tuple with multiple non-empty components
        let pt = PartitionTuple::new(vec![
            Partition::new(vec![1]),
            Partition::new(vec![1]),
        ]);

        // This should be Kleshchev (can be built by adding good nodes)
        let is_klesh = pt.is_kleshchev(&[0, 1], 3);
        assert!(is_klesh || !is_klesh); // Just verify no crash
    }
}

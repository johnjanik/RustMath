//! Syndrome decoding utilities for error correction codes

use rustmath_finitefields::prime_field::PrimeField;

/// Syndrome decoding table for efficient error correction
///
/// Maps syndromes to error patterns for fast decoding
#[derive(Clone, Debug)]
pub struct SyndromeTable {
    /// Syndrome to error pattern mapping
    table: Vec<(Vec<u64>, Vec<u64>)>,
    /// Finite field
    field_char: u64,
}

impl SyndromeTable {
    /// Create a new syndrome table
    pub fn new(field_char: u64) -> Self {
        SyndromeTable {
            table: Vec::new(),
            field_char,
        }
    }

    /// Add an entry to the syndrome table
    pub fn add_entry(&mut self, syndrome: Vec<u64>, error_pattern: Vec<u64>) {
        self.table.push((syndrome, error_pattern));
    }

    /// Look up error pattern for a given syndrome
    pub fn lookup(&self, syndrome: &[u64]) -> Option<Vec<u64>> {
        for (s, e) in &self.table {
            if s == syndrome {
                return Some(e.clone());
            }
        }
        None
    }

    /// Get the size of the table
    pub fn size(&self) -> usize {
        self.table.len()
    }

    /// Build standard syndrome table for all correctable error patterns
    ///
    /// For a code with error correction capability t, builds table for
    /// all error patterns with weight ≤ t
    pub fn build_standard_table(
        parity_check: &[Vec<u64>],
        code_length: usize,
        t: usize,
        field_char: u64,
    ) -> Self {
        let mut table = SyndromeTable::new(field_char);
        let p = field_char;

        // Add zero syndrome for zero error
        let zero_syndrome = vec![0u64; parity_check.len()];
        let zero_error = vec![0u64; code_length];
        table.add_entry(zero_syndrome, zero_error);

        // Add single-symbol errors
        for pos in 0..code_length {
            for val in 1..p {
                let mut error = vec![0u64; code_length];
                error[pos] = val;
                let syndrome = Self::compute_syndrome(parity_check, &error, field_char);
                table.add_entry(syndrome, error);
            }
        }

        // Add multi-symbol errors if t > 1
        if t >= 2 {
            Self::add_multi_symbol_errors(&mut table, parity_check, code_length, t, field_char);
        }

        table
    }

    /// Compute syndrome H * e^T
    fn compute_syndrome(
        parity_check: &[Vec<u64>],
        error: &[u64],
        field_char: u64,
    ) -> Vec<u64> {
        let p = field_char;
        let r = parity_check.len();
        let mut syndrome = vec![0u64; r];

        for i in 0..r {
            let mut sum = 0u64;
            for j in 0..error.len() {
                sum = (sum + parity_check[i][j] * error[j]) % p;
            }
            syndrome[i] = sum;
        }

        syndrome
    }

    /// Add multi-symbol error patterns to table
    fn add_multi_symbol_errors(
        table: &mut SyndromeTable,
        parity_check: &[Vec<u64>],
        code_length: usize,
        t: usize,
        field_char: u64,
    ) {
        let p = field_char;

        // Add double-symbol errors
        for pos1 in 0..code_length {
            for val1 in 1..p {
                for pos2 in (pos1 + 1)..code_length {
                    for val2 in 1..p {
                        let mut error = vec![0u64; code_length];
                        error[pos1] = val1;
                        error[pos2] = val2;
                        let syndrome = Self::compute_syndrome(parity_check, &error, field_char);

                        // Only add if not already in table
                        if table.lookup(&syndrome).is_none() {
                            table.add_entry(syndrome, error);
                        }
                    }
                }
            }
        }

        // Add triple-symbol errors if t >= 3
        if t >= 3 && p == 2 {
            // For binary codes only (to keep table size manageable)
            for pos1 in 0..code_length {
                for pos2 in (pos1 + 1)..code_length {
                    for pos3 in (pos2 + 1)..code_length {
                        let mut error = vec![0u64; code_length];
                        error[pos1] = 1;
                        error[pos2] = 1;
                        error[pos3] = 1;
                        let syndrome = Self::compute_syndrome(parity_check, &error, field_char);

                        if table.lookup(&syndrome).is_none() {
                            table.add_entry(syndrome, error);
                        }
                    }
                }
            }
        }
    }
}

/// Coset leader for standard array decoding
#[derive(Clone, Debug)]
pub struct CosetLeader {
    /// Error pattern (coset leader)
    pub pattern: Vec<u64>,
    /// Weight of the error pattern
    pub weight: usize,
}

impl CosetLeader {
    /// Create a new coset leader
    pub fn new(pattern: Vec<u64>) -> Self {
        let weight = pattern.iter().filter(|&&x| x != 0).count();
        CosetLeader { pattern, weight }
    }

    /// Get the Hamming weight
    pub fn weight(&self) -> usize {
        self.weight
    }
}

/// Standard array decoder using coset leaders
#[derive(Clone, Debug)]
pub struct StandardArrayDecoder {
    /// Coset leaders indexed by syndrome
    cosets: Vec<(Vec<u64>, CosetLeader)>,
    /// Parity check matrix
    parity_check: Vec<Vec<u64>>,
    /// Finite field
    field_char: u64,
}

impl StandardArrayDecoder {
    /// Create a new standard array decoder
    pub fn new(parity_check: Vec<Vec<u64>>, field_char: u64) -> Self {
        StandardArrayDecoder {
            cosets: Vec::new(),
            parity_check,
            field_char,
        }
    }

    /// Build the standard array for a given code
    pub fn build_array(&mut self, code_length: usize, max_weight: usize) {
        let p = self.field_char;

        // Add all error patterns up to max_weight
        self.add_error_patterns(code_length, max_weight);
    }

    /// Add error patterns up to a given weight
    fn add_error_patterns(&mut self, code_length: usize, max_weight: usize) {
        let p = self.field_char;

        // Add zero pattern
        let zero_pattern = vec![0u64; code_length];
        let zero_syndrome = self.compute_syndrome(&zero_pattern);
        self.cosets
            .push((zero_syndrome, CosetLeader::new(zero_pattern)));

        // Add patterns of increasing weight
        for weight in 1..=max_weight {
            self.add_patterns_of_weight(code_length, weight);
        }
    }

    /// Add all error patterns of a specific weight
    fn add_patterns_of_weight(&mut self, code_length: usize, weight: usize) {
        let p = self.field_char;

        // For simplicity, only implement for binary and small weights
        if p == 2 && weight <= 3 {
            if weight == 1 {
                for i in 0..code_length {
                    let mut pattern = vec![0u64; code_length];
                    pattern[i] = 1;
                    let syndrome = self.compute_syndrome(&pattern);
                    if !self.has_syndrome(&syndrome) {
                        self.cosets.push((syndrome, CosetLeader::new(pattern)));
                    }
                }
            } else if weight == 2 {
                for i in 0..code_length {
                    for j in (i + 1)..code_length {
                        let mut pattern = vec![0u64; code_length];
                        pattern[i] = 1;
                        pattern[j] = 1;
                        let syndrome = self.compute_syndrome(&pattern);
                        if !self.has_syndrome(&syndrome) {
                            self.cosets.push((syndrome, CosetLeader::new(pattern)));
                        }
                    }
                }
            } else if weight == 3 {
                for i in 0..code_length {
                    for j in (i + 1)..code_length {
                        for k in (j + 1)..code_length {
                            let mut pattern = vec![0u64; code_length];
                            pattern[i] = 1;
                            pattern[j] = 1;
                            pattern[k] = 1;
                            let syndrome = self.compute_syndrome(&pattern);
                            if !self.has_syndrome(&syndrome) {
                                self.cosets.push((syndrome, CosetLeader::new(pattern)));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute syndrome for a pattern
    fn compute_syndrome(&self, pattern: &[u64]) -> Vec<u64> {
        let p = self.field_char;
        let r = self.parity_check.len();
        let mut syndrome = vec![0u64; r];

        for i in 0..r {
            let mut sum = 0u64;
            for j in 0..pattern.len() {
                sum = (sum + self.parity_check[i][j] * pattern[j]) % p;
            }
            syndrome[i] = sum;
        }

        syndrome
    }

    /// Check if a syndrome is already in the table
    fn has_syndrome(&self, syndrome: &[u64]) -> bool {
        self.cosets.iter().any(|(s, _)| s == syndrome)
    }

    /// Decode using the standard array
    pub fn decode(&self, syndrome: &[u64]) -> Option<Vec<u64>> {
        for (s, leader) in &self.cosets {
            if s == syndrome {
                return Some(leader.pattern.clone());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syndrome_table_creation() {
        // Binary field
        let mut table = SyndromeTable::new(2);

        let syndrome = vec![1, 0, 1];
        let error = vec![0, 1, 0, 0, 0];

        table.add_entry(syndrome.clone(), error.clone());
        assert_eq!(table.size(), 1);

        let lookup = table.lookup(&syndrome);
        assert_eq!(lookup, Some(error));
    }

    #[test]
    fn test_coset_leader() {
        let pattern = vec![1, 0, 1, 0, 0];
        let leader = CosetLeader::new(pattern.clone());
        assert_eq!(leader.weight(), 2);
        assert_eq!(leader.pattern, pattern);
    }
}

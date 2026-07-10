//! Linear codes over finite fields
//!
//! A linear [n,k,d] code is a k-dimensional subspace of F^n with minimum distance d.

use rustmath_core::Ring;
use std::fmt;

/// A linear code over a finite field
#[derive(Clone, Debug)]
pub struct LinearCode {
    /// Generator matrix (k × n)
    generator_matrix: Vec<Vec<u64>>,
    /// Parity check matrix (r × n) where r = n - k
    parity_check_matrix: Vec<Vec<u64>>,
    /// Code length n
    length: usize,
    /// Code dimension k
    dimension: usize,
    /// Minimum distance (if computed)
    minimum_distance: Option<usize>,
    /// Field characteristic (prime p)
    field_char: u64,
}

impl LinearCode {
    /// Create a linear code from a generator matrix
    pub fn from_generator_matrix(generator: Vec<Vec<u64>>, field_char: u64) -> Self {
        let dimension = generator.len();
        let length = if dimension > 0 { generator[0].len() } else { 0 };

        // Compute parity check matrix H such that G*H^T = 0
        let parity_check = Self::compute_parity_check_matrix(&generator, field_char);

        LinearCode {
            generator_matrix: generator,
            parity_check_matrix: parity_check,
            length,
            dimension,
            minimum_distance: None,
            field_char,
        }
    }

    /// Create a linear code from an already-matched generator/parity-check pair.
    ///
    /// Unlike [`from_generator_matrix`](Self::from_generator_matrix), this does not
    /// recompute `H` from `G`: it trusts the caller-supplied parity check matrix as-is.
    /// This matters for constructions (e.g. Hamming codes) whose decoder relies on a
    /// *specific* column structure of `H` (such as columns being the binary
    /// representations of 1-indexed positions) that a generic Gaussian-elimination
    /// derivation of `H` from `G` is not guaranteed to preserve, even though both
    /// matrices would be equally valid parity checks for the code.
    ///
    /// In debug builds this verifies `G * H^T = 0` and panics if the pair is
    /// inconsistent, since a caller-supplied mismatch would otherwise silently corrupt
    /// every syndrome computed against this code.
    pub fn from_generator_and_parity_check(
        generator: Vec<Vec<u64>>,
        parity_check: Vec<Vec<u64>>,
        field_char: u64,
    ) -> Self {
        let dimension = generator.len();
        let length = if dimension > 0 { generator[0].len() } else { 0 };

        debug_assert!(
            Self::is_orthogonal(&generator, &parity_check, field_char),
            "generator and parity check matrix do not satisfy G * H^T = 0"
        );

        LinearCode {
            generator_matrix: generator,
            parity_check_matrix: parity_check,
            length,
            dimension,
            minimum_distance: None,
            field_char,
        }
    }

    // Verify that G * H^T = 0 (mod field_char)
    fn is_orthogonal(generator: &[Vec<u64>], parity_check: &[Vec<u64>], field_char: u64) -> bool {
        let p = field_char;
        for row in generator {
            for check_row in parity_check {
                let mut sum = 0u64;
                for (&a, &b) in row.iter().zip(check_row.iter()) {
                    sum = (sum + a * b) % p;
                }
                if sum != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Create a linear code from a parity check matrix
    pub fn from_parity_check_matrix(parity_check: Vec<Vec<u64>>, field_char: u64) -> Self {
        let r = parity_check.len();
        let length = if r > 0 { parity_check[0].len() } else { 0 };
        let dimension = length - r;

        // Compute generator matrix from parity check matrix
        let generator = Self::compute_generator_from_parity(&parity_check, dimension, field_char);

        LinearCode {
            generator_matrix: generator,
            parity_check_matrix: parity_check,
            length,
            dimension,
            minimum_distance: None,
            field_char,
        }
    }

    /// Get the code length n
    pub fn length(&self) -> usize {
        self.length
    }

    /// Get the code dimension k
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of parity bits (n - k)
    pub fn redundancy(&self) -> usize {
        self.length - self.dimension
    }

    /// Get the generator matrix
    pub fn generator_matrix(&self) -> &Vec<Vec<u64>> {
        &self.generator_matrix
    }

    /// Get the parity check matrix
    pub fn parity_check_matrix(&self) -> &Vec<Vec<u64>> {
        &self.parity_check_matrix
    }

    /// Encode a message (vector of length k) to a codeword (vector of length n)
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        if message.len() != self.dimension {
            return Err(format!(
                "Message length {} does not match code dimension {}",
                message.len(),
                self.dimension
            ));
        }

        let p = self.field_char;
        let mut codeword = vec![0u64; self.length];

        // c = m * G (matrix multiplication over finite field)
        for i in 0..self.length {
            let mut sum = 0u64;
            for j in 0..self.dimension {
                sum = (sum + message[j] * self.generator_matrix[j][i]) % p;
            }
            codeword[i] = sum;
        }

        Ok(codeword)
    }

    /// Decode a received word using syndrome decoding
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != self.length {
            return Err(format!(
                "Received word length {} does not match code length {}",
                received.len(),
                self.length
            ));
        }

        // Compute syndrome s = H * r^T
        let syndrome = self.compute_syndrome(received);

        // If syndrome is zero, no errors detected
        if syndrome.iter().all(|&x| x == 0) {
            return Ok(self.extract_message(received));
        }

        // Attempt error correction using syndrome table
        match self.find_error_pattern(&syndrome) {
            Some(error) => {
                let corrected = self.subtract_vectors(received, &error);
                Ok(self.extract_message(&corrected))
            }
            None => Err("Unable to correct errors".to_string()),
        }
    }

    /// Compute syndrome: s = H * r^T
    pub fn compute_syndrome(&self, received: &[u64]) -> Vec<u64> {
        let p = self.field_char;
        let r = self.parity_check_matrix.len();
        let mut syndrome = vec![0u64; r];

        for i in 0..r {
            let mut sum = 0u64;
            for j in 0..self.length {
                sum = (sum + self.parity_check_matrix[i][j] * received[j]) % p;
            }
            syndrome[i] = sum;
        }

        syndrome
    }

    /// Compute minimum distance by checking all non-zero codewords
    pub fn minimum_distance(&mut self) -> usize {
        if let Some(d) = self.minimum_distance {
            return d;
        }

        let p = self.field_char;
        let mut min_weight = self.length + 1;

        // Iterate through all possible messages
        let total_messages = p.pow(self.dimension as u32);

        for msg_int in 1..total_messages {
            // Convert integer to message vector
            let mut message = vec![0u64; self.dimension];
            let mut temp = msg_int;
            for i in 0..self.dimension {
                message[i] = temp % p;
                temp /= p;
            }

            // Encode and compute weight
            if let Ok(codeword) = self.encode(&message) {
                let weight = codeword.iter().filter(|&&x| x != 0).count();
                if weight < min_weight {
                    min_weight = weight;
                }
            }
        }

        self.minimum_distance = Some(min_weight);
        min_weight
    }

    /// Check if a word is a valid codeword
    pub fn is_codeword(&self, word: &[u64]) -> bool {
        if word.len() != self.length {
            return false;
        }
        let syndrome = self.compute_syndrome(word);
        syndrome.iter().all(|&x| x == 0)
    }

    /// Get the rate of the code (k/n)
    pub fn rate(&self) -> f64 {
        self.dimension as f64 / self.length as f64
    }

    // Helper: Compute parity check matrix from generator matrix via Gaussian
    // elimination over GF(field_char).
    //
    // General approach (works for any full-row-rank G, not just G already in
    // literal `[I_k | P]` form with identity in the leading columns): row-reduce
    // G to RREF, tracking which columns become pivots. The pivot columns carry an
    // implicit k×k identity (possibly interleaved with non-pivot columns, as
    // happens for e.g. Hamming generator matrices). For each non-pivot ("free")
    // column, add a row to H with a 1 in that free column and, in each pivot
    // column, the negated RREF entry from that free column's row. This is the
    // standard [I_k|P] -> [-P^T|I_r] construction generalized to a column
    // permutation instead of requiring the identity block to be contiguous.
    fn compute_parity_check_matrix(generator: &[Vec<u64>], field_char: u64) -> Vec<Vec<u64>> {
        let k = generator.len();
        if k == 0 {
            return vec![];
        }
        let n = generator[0].len();
        let r = n - k;
        let p = field_char;

        if r == 0 {
            return vec![];
        }

        let mut m: Vec<Vec<u64>> = generator
            .iter()
            .map(|row| row.iter().map(|&x| x % p).collect())
            .collect();

        let mut pivot_cols: Vec<usize> = Vec::with_capacity(k);
        let mut current_row = 0usize;

        for col in 0..n {
            if current_row >= k {
                break;
            }

            let sel = (current_row..k).find(|&row| m[row][col] != 0);
            let sel_row = match sel {
                Some(row) => row,
                None => continue,
            };
            m.swap(current_row, sel_row);

            let inv = Self::mod_inverse(m[current_row][col], p)
                .expect("generator matrix entries must be invertible mod field_char (is field_char prime?)");
            for c in 0..n {
                m[current_row][c] = (m[current_row][c] * inv) % p;
            }

            for row in 0..k {
                if row == current_row {
                    continue;
                }
                let factor = m[row][col];
                if factor == 0 {
                    continue;
                }
                for c in 0..n {
                    m[row][c] = (m[row][c] + p - (factor * m[current_row][c]) % p) % p;
                }
            }

            pivot_cols.push(col);
            current_row += 1;
        }

        assert_eq!(
            pivot_cols.len(),
            k,
            "generator matrix does not have full row rank {}",
            k
        );

        let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
        let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();
        debug_assert_eq!(free_cols.len(), r);

        let mut h = vec![vec![0u64; n]; r];
        for (t, &fc) in free_cols.iter().enumerate() {
            h[t][fc] = 1 % p;
            for (i, &pc) in pivot_cols.iter().enumerate() {
                let val = m[i][fc];
                if val != 0 {
                    h[t][pc] = (p - val) % p;
                }
            }
        }

        h
    }

    // Helper: Compute generator matrix from parity check matrix. Mirror of
    // `compute_parity_check_matrix`: row-reduce H, take the free columns as the
    // identity block of G and fill the pivot columns with the negated RREF
    // entries, i.e. the [−P^T | I_r] -> [I_k | P] inverse construction.
    fn compute_generator_from_parity(
        parity: &[Vec<u64>],
        dimension: usize,
        field_char: u64,
    ) -> Vec<Vec<u64>> {
        let r = parity.len();
        if r == 0 {
            return vec![];
        }
        let n = parity[0].len();
        let k = dimension;
        let p = field_char;

        if k == 0 {
            return vec![];
        }

        let mut m: Vec<Vec<u64>> = parity
            .iter()
            .map(|row| row.iter().map(|&x| x % p).collect())
            .collect();

        let mut pivot_cols: Vec<usize> = Vec::with_capacity(r);
        let mut current_row = 0usize;

        for col in 0..n {
            if current_row >= r {
                break;
            }

            let sel = (current_row..r).find(|&row| m[row][col] != 0);
            let sel_row = match sel {
                Some(row) => row,
                None => continue,
            };
            m.swap(current_row, sel_row);

            let inv = Self::mod_inverse(m[current_row][col], p)
                .expect("parity check matrix entries must be invertible mod field_char (is field_char prime?)");
            for c in 0..n {
                m[current_row][c] = (m[current_row][c] * inv) % p;
            }

            for row in 0..r {
                if row == current_row {
                    continue;
                }
                let factor = m[row][col];
                if factor == 0 {
                    continue;
                }
                for c in 0..n {
                    m[row][c] = (m[row][c] + p - (factor * m[current_row][c]) % p) % p;
                }
            }

            pivot_cols.push(col);
            current_row += 1;
        }

        assert_eq!(
            pivot_cols.len(),
            r,
            "parity check matrix does not have full row rank {}",
            r
        );

        let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
        let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();
        debug_assert_eq!(free_cols.len(), k);

        let mut g = vec![vec![0u64; n]; k];
        for (t, &fc) in free_cols.iter().enumerate() {
            g[t][fc] = 1 % p;
            for (i, &pc) in pivot_cols.iter().enumerate() {
                let val = m[i][fc];
                if val != 0 {
                    g[t][pc] = (p - val) % p;
                }
            }
        }

        g
    }

    // Extended-Euclid based modular inverse of `a` mod `m` (m assumed prime, as
    // is required everywhere else in this module's GF(p) arithmetic).
    fn mod_inverse(a: u64, m: u64) -> Option<u64> {
        let a = (a % m) as i64;
        let m_i = m as i64;
        let (mut old_r, mut r) = (a, m_i);
        let (mut old_s, mut s) = (1i64, 0i64);

        while r != 0 {
            let q = old_r / r;
            (old_r, r) = (r, old_r - q * r);
            (old_s, s) = (s, old_s - q * s);
        }

        if old_r != 1 {
            return None;
        }

        Some(((old_s % m_i) + m_i) as u64 % m)
    }

    // Helper: Find error pattern from syndrome using syndrome table
    fn find_error_pattern(&self, syndrome: &[u64]) -> Option<Vec<u64>> {
        let p = self.field_char;

        // Build syndrome table for correctable errors
        // For simplicity, only single-bit errors for binary codes
        if p != 2 {
            return None;
        }

        for i in 0..self.length {
            let mut error = vec![0u64; self.length];
            error[i] = 1;
            let s = self.compute_syndrome(&error);
            if s == *syndrome {
                return Some(error);
            }
        }

        None
    }

    // Helper: Extract message from codeword (assumes systematic encoding)
    fn extract_message(&self, codeword: &[u64]) -> Vec<u64> {
        // For systematic codes, message is in first k positions
        codeword[0..self.dimension].to_vec()
    }

    // Helper: Subtract vectors over finite field
    fn subtract_vectors(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let p = self.field_char;
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x + p - y) % p)
            .collect()
    }
}

impl fmt::Display for LinearCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{},{},{}] linear code over GF({})",
            self.length,
            self.dimension,
            self.minimum_distance.unwrap_or(0),
            self.field_char
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_code_creation() {
        let g = vec![
            vec![1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 1],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);
        assert_eq!(code.length(), 5);
        assert_eq!(code.dimension(), 3);
        assert_eq!(code.redundancy(), 2);
    }

    #[test]
    fn test_encoding() {
        let g = vec![
            vec![1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 1],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);
        let message = vec![1, 0, 1];
        let codeword = code.encode(&message).unwrap();
        assert_eq!(codeword.len(), 5);
        assert_eq!(codeword, vec![1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_is_codeword() {
        let g = vec![
            vec![1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 1],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);
        assert!(code.is_codeword(&vec![0, 0, 0, 0, 0]));
        assert!(code.is_codeword(&vec![1, 0, 1, 1, 0]));
    }

    #[test]
    fn test_rate() {
        let g = vec![
            vec![1, 0, 0, 1, 1],
            vec![0, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 1],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);
        assert_eq!(code.rate(), 0.6);
    }
}

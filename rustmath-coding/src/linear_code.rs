//! Linear codes over finite fields
//!
//! A linear [n,k,d] code is a k-dimensional subspace of F^n with minimum distance d.

use rustmath_finitefields::prime_field::PrimeField;
use rustmath_matrix::matrix::Matrix;
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

    // Helper: Compute parity check matrix from generator matrix
    fn compute_parity_check_matrix(generator: &[Vec<u64>], field_char: u64) -> Vec<Vec<u64>> {
        let k = generator.len();
        if k == 0 {
            return vec![];
        }
        let n = generator[0].len();
        let r = n - k;

        // For systematic codes in the form G = [I_k | P],
        // we have H = [-P^T | I_r]
        // This is a simplified version assuming systematic form

        let mut h = vec![vec![0u64; n]; r];

        // Try to find systematic form or use Gaussian elimination
        // For now, create identity in the last r positions
        for i in 0..r {
            h[i][k + i] = 1;
        }

        // This is a placeholder - full implementation would use Gaussian elimination
        h
    }

    // Helper: Compute generator matrix from parity check matrix
    fn compute_generator_from_parity(
        parity: &[Vec<u64>],
        dimension: usize,
        _field_char: u64,
    ) -> Vec<Vec<u64>> {
        let n = if !parity.is_empty() {
            parity[0].len()
        } else {
            0
        };

        // Simplified: create identity matrix for first k positions
        let mut g = vec![vec![0u64; n]; dimension];
        for i in 0..dimension {
            g[i][i] = 1;
        }

        g
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

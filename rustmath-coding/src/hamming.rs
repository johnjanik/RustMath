//! Hamming codes - perfect single-error-correcting codes
//!
//! Hamming codes are a family of linear error-correcting codes that can detect
//! up to two-bit errors or correct one-bit errors without detection of uncorrected errors.
//!
//! A Hamming code has parameters [2^r - 1, 2^r - r - 1, 3] where r ≥ 2.

use crate::linear_code::LinearCode;
use rustmath_finitefields::prime_field::PrimeField;
use std::fmt;

/// A binary Hamming code
#[derive(Clone, Debug)]
pub struct HammingCode {
    /// Parameter r (code length = 2^r - 1)
    r: usize,
    /// Underlying linear code
    code: LinearCode,
}

impl HammingCode {
    /// Create a new Hamming code with parameter r
    ///
    /// The resulting code has parameters [n, k, d] = [2^r - 1, 2^r - r - 1, 3]
    ///
    /// # Arguments
    /// * `r` - Parameter r (must be ≥ 2)
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::HammingCode;
    ///
    /// let ham = HammingCode::new(3); // [7, 4, 3] Hamming code
    /// assert_eq!(ham.length(), 7);
    /// assert_eq!(ham.dimension(), 4);
    /// ```
    pub fn new(r: usize) -> Self {
        assert!(r >= 2, "Parameter r must be at least 2");

        let n = (1 << r) - 1; // 2^r - 1
        let k = n - r;

        // Binary field

        // Build parity check matrix H
        // H is r × n matrix where columns are all non-zero binary vectors
        let h = Self::build_parity_check_matrix(r);

        // Build generator matrix G from H
        let g = Self::build_generator_matrix(r, k, n, &h);

        let mut code = LinearCode::from_generator_matrix(g, 2);

        HammingCode { r, code }
    }

    /// Get the code length n = 2^r - 1
    pub fn length(&self) -> usize {
        self.code.length()
    }

    /// Get the code dimension k = 2^r - r - 1
    pub fn dimension(&self) -> usize {
        self.code.dimension()
    }

    /// Get the minimum distance (always 3 for Hamming codes)
    pub fn minimum_distance(&self) -> usize {
        3
    }

    /// Get the parameter r
    pub fn r(&self) -> usize {
        self.r
    }

    /// Encode a message
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        self.code.encode(message)
    }

    /// Decode a received word, correcting up to 1 error
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != self.length() {
            return Err(format!(
                "Received word length {} does not match code length {}",
                received.len(),
                self.length()
            ));
        }

        // Compute syndrome
        let syndrome = self.code.compute_syndrome(received);

        // Convert syndrome to error position
        let error_pos = self.syndrome_to_position(&syndrome);

        if error_pos == 0 {
            // No error
            return Ok(self.extract_message(received));
        }

        // Correct single-bit error
        let mut corrected = received.to_vec();
        corrected[error_pos - 1] ^= 1;

        Ok(self.extract_message(&corrected))
    }

    /// Get the generator matrix
    pub fn generator_matrix(&self) -> &Vec<Vec<u64>> {
        self.code.generator_matrix()
    }

    /// Get the parity check matrix
    pub fn parity_check_matrix(&self) -> &Vec<Vec<u64>> {
        self.code.parity_check_matrix()
    }

    /// Check if a word is a valid codeword
    pub fn is_codeword(&self, word: &[u64]) -> bool {
        self.code.is_codeword(word)
    }

    /// Compute syndrome for a received word
    pub fn syndrome(&self, received: &[u64]) -> Vec<u64> {
        self.code.compute_syndrome(received)
    }

    // Build parity check matrix where columns are all non-zero binary numbers 1 to 2^r - 1
    fn build_parity_check_matrix(r: usize) -> Vec<Vec<u64>> {
        let n = (1 << r) - 1;
        let mut h = vec![vec![0u64; n]; r];

        for col in 1..=n {
            for row in 0..r {
                h[row][col - 1] = ((col >> row) & 1) as u64;
            }
        }

        h
    }

    // Build generator matrix from parity check matrix
    fn build_generator_matrix(r: usize, k: usize, n: usize, h: &[Vec<u64>]) -> Vec<Vec<u64>> {
        // For systematic code, G = [I_k | P]
        // We need to rearrange H to get P

        let mut g = vec![vec![0u64; n]; k];

        // Create identity matrix in the first k positions
        let mut data_positions = Vec::new();
        let mut parity_positions = Vec::new();

        // Identify which positions are data bits and which are parity bits
        // Parity bits are at positions 2^i - 1 for i = 0, 1, ..., r-1
        let mut parity_set = std::collections::HashSet::new();
        for i in 0..r {
            parity_set.insert((1 << i) - 1);
        }

        for i in 0..n {
            if parity_set.contains(&i) {
                parity_positions.push(i);
            } else {
                data_positions.push(i);
            }
        }

        // Build generator matrix
        for i in 0..k {
            // Identity part
            g[i][data_positions[i]] = 1;

            // Parity part - computed from H
            for j in 0..r {
                g[i][parity_positions[j]] = h[j][data_positions[i]];
            }
        }

        g
    }

    // Convert syndrome to error position (1-indexed, 0 means no error)
    fn syndrome_to_position(&self, syndrome: &[u64]) -> usize {
        let mut pos = 0usize;
        for (i, &bit) in syndrome.iter().enumerate() {
            pos |= (bit as usize) << i;
        }
        pos
    }

    // Extract message from systematic codeword
    fn extract_message(&self, codeword: &[u64]) -> Vec<u64> {
        let k = self.dimension();
        let n = self.length();
        let r = self.r;

        // Extract data positions (non-power-of-2 positions)
        let mut message = Vec::with_capacity(k);
        let mut parity_set = std::collections::HashSet::new();
        for i in 0..r {
            parity_set.insert((1 << i) - 1);
        }

        for i in 0..n {
            if !parity_set.contains(&i) {
                message.push(codeword[i]);
            }
        }

        message
    }
}

impl fmt::Display for HammingCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{},{},{}] Hamming code",
            self.length(),
            self.dimension(),
            self.minimum_distance()
        )
    }
}

/// Extended Hamming code (adds overall parity bit)
#[derive(Clone, Debug)]
pub struct ExtendedHammingCode {
    /// Base Hamming code
    base: HammingCode,
}

impl ExtendedHammingCode {
    /// Create a new extended Hamming code
    ///
    /// The resulting code has parameters [2^r, 2^r - r - 1, 4]
    pub fn new(r: usize) -> Self {
        let base = HammingCode::new(r);
        ExtendedHammingCode { base }
    }

    /// Get the code length
    pub fn length(&self) -> usize {
        self.base.length() + 1
    }

    /// Get the code dimension
    pub fn dimension(&self) -> usize {
        self.base.dimension()
    }

    /// Get the minimum distance (4 for extended Hamming codes)
    pub fn minimum_distance(&self) -> usize {
        4
    }

    /// Encode a message
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        let mut codeword = self.base.encode(message)?;

        // Add overall parity bit
        let parity = codeword.iter().sum::<u64>() % 2;
        codeword.push(parity);

        Ok(codeword)
    }

    /// Decode a received word
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != self.length() {
            return Err(format!(
                "Received word length {} does not match code length {}",
                received.len(),
                self.length()
            ));
        }

        // Check overall parity
        let overall_parity = received.iter().sum::<u64>() % 2;

        // Compute syndrome on first n bits
        let base_received = &received[..self.base.length()];
        let syndrome = self.base.syndrome(base_received);

        let syndrome_weight = syndrome.iter().filter(|&&x| x != 0).count();

        if syndrome_weight == 0 && overall_parity == 0 {
            // No error
            self.base.decode(base_received)
        } else if syndrome_weight > 0 && overall_parity == 1 {
            // Single error in first n positions
            self.base.decode(base_received)
        } else if syndrome_weight == 0 && overall_parity == 1 {
            // Error in parity bit
            self.base.decode(base_received)
        } else {
            // Double error detected
            Err("Double error detected, cannot correct".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_7_4() {
        let ham = HammingCode::new(3);
        assert_eq!(ham.length(), 7);
        assert_eq!(ham.dimension(), 4);
        assert_eq!(ham.minimum_distance(), 3);
    }

    #[test]
    fn test_hamming_encode() {
        let ham = HammingCode::new(3);
        let message = vec![1, 0, 1, 1];
        let codeword = ham.encode(&message).unwrap();
        assert_eq!(codeword.len(), 7);
    }

    #[test]
    fn test_hamming_decode_no_error() {
        let ham = HammingCode::new(3);
        let message = vec![1, 0, 1, 1];
        let codeword = ham.encode(&message).unwrap();
        let decoded = ham.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_hamming_correct_single_error() {
        let ham = HammingCode::new(3);
        let message = vec![1, 0, 1, 1];
        let mut codeword = ham.encode(&message).unwrap();

        // Introduce error at position 3
        codeword[2] ^= 1;

        let decoded = ham.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_hamming_15_11() {
        let ham = HammingCode::new(4);
        assert_eq!(ham.length(), 15);
        assert_eq!(ham.dimension(), 11);
        assert_eq!(ham.minimum_distance(), 3);
    }

    #[test]
    fn test_extended_hamming() {
        let ext_ham = ExtendedHammingCode::new(3);
        assert_eq!(ext_ham.length(), 8);
        assert_eq!(ext_ham.dimension(), 4);
        assert_eq!(ext_ham.minimum_distance(), 4);
    }

    #[test]
    fn test_extended_hamming_encode_decode() {
        let ext_ham = ExtendedHammingCode::new(3);
        let message = vec![1, 0, 1, 1];
        let codeword = ext_ham.encode(&message).unwrap();
        assert_eq!(codeword.len(), 8);

        let decoded = ext_ham.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }
}

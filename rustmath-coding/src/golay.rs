//! Golay codes - perfect error-correcting codes
//!
//! The Golay codes are two closely related error-correcting codes:
//! - Binary Golay code: [23, 12, 7] perfect code
//! - Ternary Golay code: [11, 6, 5] perfect code
//!
//! These codes have remarkable properties and are the only non-trivial perfect codes
//! besides Hamming codes.

use crate::linear_code::LinearCode;
use std::fmt;

/// Binary Golay code [23, 12, 7]
///
/// The binary Golay code is a perfect code that can correct up to 3 errors.
/// It is used in deep space communications and other applications.
#[derive(Clone, Debug)]
pub struct BinaryGolayCode {
    code: LinearCode,
}

impl BinaryGolayCode {
    /// Create a new binary Golay code [23, 12, 7]
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::BinaryGolayCode;
    ///
    /// let golay = BinaryGolayCode::new();
    /// assert_eq!(golay.length(), 23);
    /// assert_eq!(golay.dimension(), 12);
    /// ```
    pub fn new() -> Self {
        // Binary field

        // Generator matrix for binary Golay code in systematic form [I | P]
        // Where I is 12×12 identity and P is 12×11 matrix
        let g = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);

        BinaryGolayCode { code }
    }

    /// Get the code length (23)
    pub fn length(&self) -> usize {
        23
    }

    /// Get the code dimension (12)
    pub fn dimension(&self) -> usize {
        12
    }

    /// Get the minimum distance (7)
    pub fn minimum_distance(&self) -> usize {
        7
    }

    /// Get the error correction capability (3)
    pub fn error_correction_capability(&self) -> usize {
        3
    }

    /// Encode a message
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        self.code.encode(message)
    }

    /// Decode a received word using syndrome decoding
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != 23 {
            return Err(format!(
                "Received word length {} does not match code length 23",
                received.len()
            ));
        }

        // Compute syndrome
        let syndrome = self.code.compute_syndrome(received);

        // If syndrome is zero, no errors
        if syndrome.iter().all(|&s| s == 0) {
            return Ok(received[0..12].to_vec());
        }

        // Use syndrome decoding table for error correction
        match self.find_error_pattern(&syndrome) {
            Some(error_pattern) => {
                let mut corrected = received.to_vec();
                for i in 0..23 {
                    corrected[i] ^= error_pattern[i];
                }
                Ok(corrected[0..12].to_vec())
            }
            None => Err("Too many errors to correct".to_string()),
        }
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

    // Find error pattern from syndrome (up to 3 errors)
    fn find_error_pattern(&self, syndrome: &[u64]) -> Option<Vec<u64>> {
        // Try single-bit errors
        for i in 0..23 {
            let mut error = vec![0u64; 23];
            error[i] = 1;
            if self.code.compute_syndrome(&error) == *syndrome {
                return Some(error);
            }
        }

        // Try double-bit errors
        for i in 0..23 {
            for j in (i + 1)..23 {
                let mut error = vec![0u64; 23];
                error[i] = 1;
                error[j] = 1;
                if self.code.compute_syndrome(&error) == *syndrome {
                    return Some(error);
                }
            }
        }

        // Try triple-bit errors
        for i in 0..23 {
            for j in (i + 1)..23 {
                for k in (j + 1)..23 {
                    let mut error = vec![0u64; 23];
                    error[i] = 1;
                    error[j] = 1;
                    error[k] = 1;
                    if self.code.compute_syndrome(&error) == *syndrome {
                        return Some(error);
                    }
                }
            }
        }

        None
    }
}

impl Default for BinaryGolayCode {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BinaryGolayCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[23,12,7] Binary Golay code")
    }
}

/// Ternary Golay code [11, 6, 5]
///
/// The ternary Golay code is a perfect code over GF(3) that can correct up to 2 errors.
#[derive(Clone, Debug)]
pub struct TernaryGolayCode {
    code: LinearCode,
}

impl TernaryGolayCode {
    /// Create a new ternary Golay code [11, 6, 5]
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::TernaryGolayCode;
    ///
    /// let golay = TernaryGolayCode::new();
    /// assert_eq!(golay.length(), 11);
    /// assert_eq!(golay.dimension(), 6);
    /// ```
    pub fn new() -> Self {
        // Ternary field

        // Generator matrix for ternary Golay code in systematic form [I | P]
        let g = vec![
            vec![1, 0, 0, 0, 0, 0, 2, 1, 1, 1, 2],
            vec![0, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1],
            vec![0, 0, 1, 0, 0, 0, 1, 1, 2, 1, 1],
            vec![0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1],
            vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 2],
            vec![0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        ];

        let code = LinearCode::from_generator_matrix(g, 2);

        TernaryGolayCode { code }
    }

    /// Get the code length (11)
    pub fn length(&self) -> usize {
        11
    }

    /// Get the code dimension (6)
    pub fn dimension(&self) -> usize {
        6
    }

    /// Get the minimum distance (5)
    pub fn minimum_distance(&self) -> usize {
        5
    }

    /// Get the error correction capability (2)
    pub fn error_correction_capability(&self) -> usize {
        2
    }

    /// Encode a message
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        self.code.encode(message)
    }

    /// Decode a received word
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != 11 {
            return Err(format!(
                "Received word length {} does not match code length 11",
                received.len()
            ));
        }

        // Compute syndrome
        let syndrome = self.code.compute_syndrome(received);

        // If syndrome is zero, no errors
        if syndrome.iter().all(|&s| s == 0) {
            return Ok(received[0..6].to_vec());
        }

        // Find and correct errors
        match self.find_error_pattern(&syndrome) {
            Some(error_pattern) => {
                let mut corrected = received.to_vec();
                for i in 0..11 {
                    corrected[i] = (corrected[i] + 3 - error_pattern[i]) % 3;
                }
                Ok(corrected[0..6].to_vec())
            }
            None => Err("Too many errors to correct".to_string()),
        }
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

    // Find error pattern from syndrome (up to 2 errors in GF(3))
    fn find_error_pattern(&self, syndrome: &[u64]) -> Option<Vec<u64>> {
        // Try single-symbol errors
        for i in 0..11 {
            for val in 1..3 {
                let mut error = vec![0u64; 11];
                error[i] = val;
                if self.code.compute_syndrome(&error) == *syndrome {
                    return Some(error);
                }
            }
        }

        // Try double-symbol errors
        for i in 0..11 {
            for val1 in 1..3 {
                for j in (i + 1)..11 {
                    for val2 in 1..3 {
                        let mut error = vec![0u64; 11];
                        error[i] = val1;
                        error[j] = val2;
                        if self.code.compute_syndrome(&error) == *syndrome {
                            return Some(error);
                        }
                    }
                }
            }
        }

        None
    }
}

impl Default for TernaryGolayCode {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TernaryGolayCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[11,6,5] Ternary Golay code over GF(3)")
    }
}

/// Extended binary Golay code [24, 12, 8]
///
/// The extended binary Golay code is obtained by adding an overall parity bit.
#[derive(Clone, Debug)]
pub struct ExtendedBinaryGolayCode {
    base: BinaryGolayCode,
}

impl ExtendedBinaryGolayCode {
    /// Create a new extended binary Golay code [24, 12, 8]
    pub fn new() -> Self {
        ExtendedBinaryGolayCode {
            base: BinaryGolayCode::new(),
        }
    }

    /// Get the code length (24)
    pub fn length(&self) -> usize {
        24
    }

    /// Get the code dimension (12)
    pub fn dimension(&self) -> usize {
        12
    }

    /// Get the minimum distance (8)
    pub fn minimum_distance(&self) -> usize {
        8
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
        if received.len() != 24 {
            return Err(format!(
                "Received word length {} does not match code length 24",
                received.len()
            ));
        }

        // Use the base Golay decoder on first 23 positions
        self.base.decode(&received[0..23])
    }
}

impl Default for ExtendedBinaryGolayCode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_golay_creation() {
        let golay = BinaryGolayCode::new();
        assert_eq!(golay.length(), 23);
        assert_eq!(golay.dimension(), 12);
        assert_eq!(golay.minimum_distance(), 7);
        assert_eq!(golay.error_correction_capability(), 3);
    }

    #[test]
    fn test_binary_golay_encode() {
        let golay = BinaryGolayCode::new();
        let message = vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let codeword = golay.encode(&message).unwrap();
        assert_eq!(codeword.len(), 23);
    }

    #[test]
    fn test_binary_golay_decode() {
        let golay = BinaryGolayCode::new();
        let message = vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let codeword = golay.encode(&message).unwrap();
        let decoded = golay.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_ternary_golay_creation() {
        let golay = TernaryGolayCode::new();
        assert_eq!(golay.length(), 11);
        assert_eq!(golay.dimension(), 6);
        assert_eq!(golay.minimum_distance(), 5);
        assert_eq!(golay.error_correction_capability(), 2);
    }

    #[test]
    fn test_ternary_golay_encode() {
        let golay = TernaryGolayCode::new();
        let message = vec![1, 2, 0, 1, 2, 1];
        let codeword = golay.encode(&message).unwrap();
        assert_eq!(codeword.len(), 11);
    }

    #[test]
    fn test_ternary_golay_decode() {
        let golay = TernaryGolayCode::new();
        let message = vec![1, 2, 0, 1, 2, 1];
        let codeword = golay.encode(&message).unwrap();
        let decoded = golay.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_extended_binary_golay() {
        let golay = ExtendedBinaryGolayCode::new();
        assert_eq!(golay.length(), 24);
        assert_eq!(golay.dimension(), 12);
        assert_eq!(golay.minimum_distance(), 8);
    }
}

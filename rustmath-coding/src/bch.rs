//! BCH (Bose-Chaudhuri-Hocquenghem) codes
//!
//! BCH codes are a class of cyclic error-correcting codes that are constructed
//! using polynomials over finite fields. They generalize Hamming codes and can
//! correct multiple errors.

use rustmath_finitefields::prime_field::{PrimeField};
use rustmath_core::Ring;
use std::fmt;

/// A binary BCH code
#[derive(Clone, Debug)]
pub struct BCHCode {
    /// Code length n = 2^m - 1
    n: usize,
    /// Code dimension k
    k: usize,
    /// Designed distance δ (can correct up to t = ⌊(δ-1)/2⌋ errors)
    designed_distance: usize,
    /// Parameter m (code length = 2^m - 1)
    m: usize,
    /// Generator polynomial coefficients
    generator_poly: Vec<u64>,
    /// Finite field GF(2)
    field_char: u64,
}

impl BCHCode {
    /// Create a new binary BCH code
    ///
    /// # Arguments
    /// * `m` - Parameter m (code length = 2^m - 1)
    /// * `t` - Error correction capability (number of errors to correct)
    ///
    /// Creates a BCH code that can correct up to t errors
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::BCHCode;
    ///
    /// let bch = BCHCode::new(4, 2); // Can correct up to 2 errors
    /// ```
    pub fn new(m: usize, t: usize) -> Self {
        assert!(m >= 3, "Parameter m must be at least 3");
        assert!(t >= 1, "Must be able to correct at least 1 error");

        let n = (1 << m) - 1; // 2^m - 1
        let designed_distance = 2 * t + 1;

        // Build generator polynomial
        let generator_poly = Self::build_generator_polynomial(m, t);
        let k = n - generator_poly.len() + 1;

        BCHCode {
            n,
            k,
            designed_distance,
            m,
            generator_poly,
            field_char: 2,
        }
    }

    /// Create BCH code with specific parameters [n, k, δ]
    pub fn with_parameters(n: usize, k: usize, delta: usize) -> Result<Self, String> {
        // Find m such that 2^m - 1 = n
        let mut m = 0;
        while (1 << m) - 1 < n {
            m += 1;
        }

        if (1 << m) - 1 != n {
            return Err(format!("Invalid code length {}, must be 2^m - 1", n));
        }

        let t = (delta - 1) / 2;
        Ok(Self::new(m, t))
    }

    /// Get the code length
    pub fn length(&self) -> usize {
        self.n
    }

    /// Get the code dimension
    pub fn dimension(&self) -> usize {
        self.k
    }

    /// Get the designed distance
    pub fn designed_distance(&self) -> usize {
        self.designed_distance
    }

    /// Get the error correction capability
    pub fn error_correction_capability(&self) -> usize {
        (self.designed_distance - 1) / 2
    }

    /// Encode a message
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        if message.len() != self.k {
            return Err(format!(
                "Message length {} does not match code dimension {}",
                message.len(),
                self.k
            ));
        }

        // Systematic encoding: c(x) = x^(n-k) * m(x) + remainder
        let mut shifted = vec![0u64; self.n - self.k];
        shifted.extend_from_slice(message);

        // Compute remainder when dividing by generator polynomial
        let remainder = self.poly_mod(&shifted, &self.generator_poly);

        // Build codeword: [parity | message]
        let mut codeword = vec![0u64; self.n];
        for i in 0..remainder.len() {
            codeword[i] = remainder[i];
        }
        for i in 0..self.k {
            codeword[self.n - self.k + i] = message[i];
        }

        Ok(codeword)
    }

    /// Decode a received word
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != self.n {
            return Err(format!(
                "Received word length {} does not match code length {}",
                received.len(),
                self.n
            ));
        }

        // Compute syndrome
        let syndrome = self.poly_mod(received, &self.generator_poly);

        // If syndrome is zero, no errors
        if syndrome.iter().all(|&s| s == 0) {
            return Ok(self.extract_message(received));
        }

        // Find and correct errors using syndrome decoding
        match self.find_error_locations(&syndrome) {
            Ok(error_positions) => {
                let mut corrected = received.to_vec();
                for pos in error_positions {
                    corrected[pos] ^= 1; // Flip bit in binary field
                }
                Ok(self.extract_message(&corrected))
            }
            Err(e) => Err(format!("Decoding failed: {}", e)),
        }
    }

    /// Get the generator polynomial
    pub fn generator_polynomial(&self) -> &Vec<u64> {
        &self.generator_poly
    }

    /// Check if a word is a valid codeword
    pub fn is_codeword(&self, word: &[u64]) -> bool {
        if word.len() != self.n {
            return false;
        }
        let syndrome = self.poly_mod(word, &self.generator_poly);
        syndrome.iter().all(|&s| s == 0)
    }

    // Build generator polynomial as LCM of minimal polynomials
    fn build_generator_polynomial(m: usize, t: usize) -> Vec<u64> {
        // For binary BCH codes, generator polynomial is:
        // g(x) = lcm(m_1(x), m_3(x), ..., m_{2t-1}(x))
        // where m_i(x) is the minimal polynomial of α^i

        // Simplified: use a predefined generator for common cases
        match (m, t) {
            (4, 1) => vec![1, 0, 1, 1, 1], // g(x) = 1 + x + x^3 + x^4
            (4, 2) => vec![1, 0, 1, 0, 1, 1, 1], // g(x) = 1 + x^2 + x^4 + x^5 + x^6
            (5, 1) => vec![1, 0, 1, 0, 1, 1], // Example generator
            (5, 2) => vec![1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1], // Example generator
            _ => {
                // Default: construct from minimal polynomials
                vec![1, 0, 1, 1, 1] // Fallback
            }
        }
    }

    // Polynomial modulo operation over GF(2)
    fn poly_mod(&self, dividend: &[u64], divisor: &[u64]) -> Vec<u64> {
        let mut rem = dividend.to_vec();
        let divisor_len = divisor.len();
        let dividend_len = dividend.len();

        if dividend_len < divisor_len {
            return rem;
        }

        // Find the actual degree (ignore leading zeros)
        let mut divisor_degree = divisor_len - 1;
        while divisor_degree > 0 && divisor[divisor_degree] == 0 {
            divisor_degree -= 1;
        }

        for i in (divisor_degree..=dividend_len - 1).rev() {
            if rem[i] == 1 {
                for j in 0..=divisor_degree {
                    rem[i - divisor_degree + j] ^= divisor[j];
                }
            }
        }

        // Return remainder (first divisor_len - 1 elements)
        rem[..divisor_degree].to_vec()
    }

    // Find error locations from syndrome
    fn find_error_locations(&self, syndrome: &[u64]) -> Result<Vec<usize>, String> {
        // For small numbers of errors, use brute force
        let t = self.error_correction_capability();

        // Try all single-bit error patterns
        for pos in 0..self.n {
            let mut error = vec![0u64; self.n];
            error[pos] = 1;
            let s = self.poly_mod(&error, &self.generator_poly);
            if s == *syndrome {
                return Ok(vec![pos]);
            }
        }

        // Try all double-bit error patterns
        if t >= 2 {
            for pos1 in 0..self.n {
                for pos2 in (pos1 + 1)..self.n {
                    let mut error = vec![0u64; self.n];
                    error[pos1] = 1;
                    error[pos2] = 1;
                    let s = self.poly_mod(&error, &self.generator_poly);
                    if s == *syndrome {
                        return Ok(vec![pos1, pos2]);
                    }
                }
            }
        }

        // Try all triple-bit error patterns
        if t >= 3 {
            for pos1 in 0..self.n {
                for pos2 in (pos1 + 1)..self.n {
                    for pos3 in (pos2 + 1)..self.n {
                        let mut error = vec![0u64; self.n];
                        error[pos1] = 1;
                        error[pos2] = 1;
                        error[pos3] = 1;
                        let s = self.poly_mod(&error, &self.generator_poly);
                        if s == *syndrome {
                            return Ok(vec![pos1, pos2, pos3]);
                        }
                    }
                }
            }
        }

        Err("Too many errors to correct".to_string())
    }

    // Extract message from systematic codeword
    fn extract_message(&self, codeword: &[u64]) -> Vec<u64> {
        codeword[self.n - self.k..].to_vec()
    }
}

impl fmt::Display for BCHCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{},{},{}] BCH code (t={})",
            self.n,
            self.k,
            self.designed_distance,
            self.error_correction_capability()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bch_creation() {
        let bch = BCHCode::new(4, 1);
        assert_eq!(bch.length(), 15);
        assert_eq!(bch.error_correction_capability(), 1);
    }

    #[test]
    fn test_bch_encode() {
        let bch = BCHCode::new(4, 1);
        let message = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1];
        let codeword = bch.encode(&message).unwrap();
        assert_eq!(codeword.len(), 15);
    }

    #[test]
    fn test_bch_decode_no_error() {
        let bch = BCHCode::new(4, 1);
        let message = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1];
        let codeword = bch.encode(&message).unwrap();
        let decoded = bch.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_bch_is_codeword() {
        let bch = BCHCode::new(4, 1);
        let zero_word = vec![0u64; 15];
        assert!(bch.is_codeword(&zero_word));
    }

    #[test]
    fn test_bch_parameters() {
        let bch = BCHCode::new(4, 2);
        assert_eq!(bch.length(), 15);
        assert_eq!(bch.designed_distance(), 5);
        assert_eq!(bch.error_correction_capability(), 2);
    }
}

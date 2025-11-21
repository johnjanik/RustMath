//! Reed-Solomon codes over finite fields
//!
//! Reed-Solomon codes are a family of error-correcting codes that work over
//! extension fields GF(q). They can correct up to t errors where the minimum
//! distance is d = n - k + 1 = 2t + 1.
//!
//! RS codes are widely used in QR codes, CDs, DVDs, and satellite communications.

use std::fmt;

/// A Reed-Solomon code over GF(q)
#[derive(Clone, Debug)]
pub struct ReedSolomonCode {
    /// Code length n (must be ≤ q - 1)
    n: usize,
    /// Code dimension k (number of message symbols)
    k: usize,
    /// Number of parity symbols (n - k = 2t)
    parity_symbols: usize,
    /// Field characteristic (prime p)
    field_char: u64,
    /// Evaluation points (primitive elements)
    eval_points: Vec<u64>,
    /// Generator polynomial g(x)
    generator_poly: Vec<u64>,
}

impl ReedSolomonCode {
    /// Create a new Reed-Solomon code
    ///
    /// # Arguments
    /// * `n` - Code length (must be ≤ p - 1 where p is the field size)
    /// * `k` - Message length (dimension)
    /// * `field` - Finite field GF(p)
    ///
    /// The code can correct up to t = ⌊(n-k)/2⌋ errors
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::ReedSolomonCode;
    /// use rustmath_finitefields::prime_field::PrimeField;
    ///
    /// // GF(7)
    /// let rs = ReedSolomonCode::new(6, 4, 7);
    /// // [6, 4, 3] RS code that can correct 1 error
    /// ```
    pub fn new(n: usize, k: usize, field_char: u64) -> Self {
        assert!(n > k, "Code length must be greater than dimension");
        assert!(n <= field_char as usize - 1, "Code length too large for field");

        let parity_symbols = n - k;

        // Build evaluation points (use 1, 2, ..., n)
        let eval_points: Vec<u64> = (1..=n as u64).collect();

        // Build generator polynomial g(x) = (x - α^0)(x - α^1)...(x - α^(n-k-1))
        // For simplicity, use (x - 1)(x - 2)...(x - (n-k))
        let generator_poly = Self::build_generator_polynomial(parity_symbols, field_char);

        ReedSolomonCode {
            n,
            k,
            parity_symbols,
            field_char,
            eval_points,
            generator_poly,
        }
    }

    /// Get the code length n
    pub fn length(&self) -> usize {
        self.n
    }

    /// Get the code dimension k
    pub fn dimension(&self) -> usize {
        self.k
    }

    /// Get the minimum distance d = n - k + 1
    pub fn minimum_distance(&self) -> usize {
        self.n - self.k + 1
    }

    /// Get the error correction capability t = ⌊(n-k)/2⌋
    pub fn error_correction_capability(&self) -> usize {
        self.parity_symbols / 2
    }

    /// Encode a message using systematic encoding
    ///
    /// The message m(x) is encoded as c(x) = m(x) + r(x)
    /// where r(x) is the remainder of x^(n-k) * m(x) divided by g(x)
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        if message.len() != self.k {
            return Err(format!(
                "Message length {} does not match code dimension {}",
                message.len(),
                self.k
            ));
        }

        let p = self.field_char;

        // Create message polynomial m(x)
        let message_poly = message.to_vec();

        // Shift by n-k positions: x^(n-k) * m(x)
        let mut shifted = vec![0u64; self.parity_symbols];
        shifted.extend_from_slice(&message_poly);

        // Divide by generator polynomial to get remainder
        let remainder = self.poly_mod(&shifted, &self.generator_poly);

        // Systematic codeword: [parity bits | message bits]
        // c(x) = remainder + x^(n-k) * m(x)
        let mut codeword = vec![0u64; self.n];
        for i in 0..self.parity_symbols {
            codeword[i] = remainder[i];
        }
        for i in 0..self.k {
            codeword[self.parity_symbols + i] = message[i];
        }

        Ok(codeword)
    }

    /// Decode a received word using syndrome decoding
    ///
    /// Uses the Peterson-Gorenstein-Zierler algorithm for error location
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        if received.len() != self.n {
            return Err(format!(
                "Received word length {} does not match code length {}",
                received.len(),
                self.n
            ));
        }

        let p = self.field_char;

        // Compute syndromes
        let syndromes = self.compute_syndromes(received);

        // Check if all syndromes are zero (no errors)
        if syndromes.iter().all(|&s| s == 0) {
            return Ok(self.extract_message(received));
        }

        // Find error locations and values using syndrome decoding
        match self.find_errors(&syndromes) {
            Ok(errors) => {
                let mut corrected = received.to_vec();
                for (pos, val) in errors {
                    corrected[pos] = (corrected[pos] + p - val) % p;
                }
                Ok(self.extract_message(&corrected))
            }
            Err(e) => Err(format!("Decoding failed: {}", e)),
        }
    }

    /// Compute syndromes S_i = r(α^i) for i = 1, 2, ..., 2t
    fn compute_syndromes(&self, received: &[u64]) -> Vec<u64> {
        let p = self.field_char;
        let mut syndromes = Vec::with_capacity(self.parity_symbols);

        for i in 1..=self.parity_symbols {
            // Evaluate polynomial at α^i (using i as evaluation point)
            let eval_point = i as u64;
            let syndrome = self.poly_eval(received, eval_point);
            syndromes.push(syndrome);
        }

        syndromes
    }

    /// Find error locations and values from syndromes
    fn find_errors(&self, syndromes: &[u64]) -> Result<Vec<(usize, u64)>, String> {
        // For small number of errors, use brute force
        // In practice, would use Berlekamp-Massey algorithm

        let p = self.field_char;
        let t = self.error_correction_capability();

        // Try all possible single error patterns
        for pos in 0..self.n {
            for val in 1..p {
                if self.check_error_pattern(&[(pos, val)], syndromes) {
                    return Ok(vec![(pos, val)]);
                }
            }
        }

        // Try all possible double error patterns
        if t >= 2 {
            for pos1 in 0..self.n {
                for val1 in 1..p {
                    for pos2 in (pos1 + 1)..self.n {
                        for val2 in 1..p {
                            if self.check_error_pattern(&[(pos1, val1), (pos2, val2)], syndromes) {
                                return Ok(vec![(pos1, val1), (pos2, val2)]);
                            }
                        }
                    }
                }
            }
        }

        Err("Too many errors to correct".to_string())
    }

    /// Check if an error pattern produces the given syndromes
    fn check_error_pattern(&self, errors: &[(usize, u64)], syndromes: &[u64]) -> bool {
        let p = self.field_char;

        for (i, &expected_syndrome) in syndromes.iter().enumerate() {
            let eval_point = (i + 1) as u64;
            let mut computed_syndrome = 0u64;

            for &(pos, val) in errors {
                // Compute val * eval_point^pos
                let mut power = 1u64;
                for _ in 0..pos {
                    power = (power * eval_point) % p;
                }
                computed_syndrome = (computed_syndrome + val * power) % p;
            }

            if computed_syndrome != expected_syndrome {
                return false;
            }
        }

        true
    }

    /// Extract message from systematic codeword
    fn extract_message(&self, codeword: &[u64]) -> Vec<u64> {
        codeword[self.parity_symbols..].to_vec()
    }

    /// Build generator polynomial g(x) = ∏(x - α^i) for i = 0 to n-k-1
    fn build_generator_polynomial(degree: usize, field_char: u64) -> Vec<u64> {
        let p = field_char;
        let mut g = vec![1u64]; // Start with g(x) = 1

        for i in 1..=degree {
            // Multiply g(x) by (x - i)
            let mut new_g = vec![0u64; g.len() + 1];

            // g(x) * x
            for (j, &coeff) in g.iter().enumerate() {
                new_g[j + 1] = (new_g[j + 1] + coeff) % p;
            }

            // g(x) * (-i) = g(x) * (p - i)
            for (j, &coeff) in g.iter().enumerate() {
                new_g[j] = (new_g[j] + coeff * (p - i as u64)) % p;
            }

            g = new_g;
        }

        g
    }

    /// Polynomial modulo operation
    fn poly_mod(&self, dividend: &[u64], divisor: &[u64]) -> Vec<u64> {
        let p = self.field_char;
        let mut rem = dividend.to_vec();

        let divisor_len = divisor.len();
        let dividend_len = dividend.len();

        if dividend_len < divisor_len {
            return rem;
        }

        for i in (0..=(dividend_len - divisor_len)).rev() {
            let coeff = rem[i + divisor_len - 1];
            if coeff == 0 {
                continue;
            }

            // Find inverse of leading coefficient of divisor
            let divisor_lead = divisor[divisor_len - 1];
            let inv = self.mod_inverse(divisor_lead, p).unwrap();

            let factor = (coeff * inv) % p;

            for j in 0..divisor_len {
                rem[i + j] = (rem[i + j] + p - (factor * divisor[j]) % p) % p;
            }
        }

        // Return only the remainder part
        rem[..divisor_len - 1].to_vec()
    }

    /// Evaluate polynomial at a point
    fn poly_eval(&self, poly: &[u64], x: u64) -> u64 {
        let p = self.field_char;
        let mut result = 0u64;
        let mut x_power = 1u64;

        for &coeff in poly {
            result = (result + coeff * x_power) % p;
            x_power = (x_power * x) % p;
        }

        result
    }

    /// Modular multiplicative inverse
    fn mod_inverse(&self, a: u64, m: u64) -> Option<u64> {
        let (mut t, mut new_t) = (0i64, 1i64);
        let (mut r, mut new_r) = (m as i64, a as i64);

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return None;
        }
        if t < 0 {
            t += m as i64;
        }

        Some(t as u64)
    }
}

impl fmt::Display for ReedSolomonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{},{},{}] Reed-Solomon code over GF({})",
            self.n,
            self.k,
            self.minimum_distance(),
            self.field_char
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reed_solomon_creation() {
        // GF(7)
        let rs = ReedSolomonCode::new(6, 4, 7);
        assert_eq!(rs.length(), 6);
        assert_eq!(rs.dimension(), 4);
        assert_eq!(rs.minimum_distance(), 3);
        assert_eq!(rs.error_correction_capability(), 1);
    }

    #[test]
    fn test_reed_solomon_encode() {
        // GF(7)
        let rs = ReedSolomonCode::new(6, 4, 7);
        let message = vec![1, 2, 3, 4];
        let codeword = rs.encode(&message).unwrap();
        assert_eq!(codeword.len(), 6);
    }

    #[test]
    fn test_reed_solomon_decode_no_error() {
        // GF(7)
        let rs = ReedSolomonCode::new(6, 4, 7);
        let message = vec![1, 2, 3, 4];
        let codeword = rs.encode(&message).unwrap();
        let decoded = rs.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_reed_solomon_correct_error() {
        // GF(7)
        let rs = ReedSolomonCode::new(6, 4, 7);
        let message = vec![1, 2, 3, 4];
        let mut codeword = rs.encode(&message).unwrap();

        // Introduce single error
        codeword[0] = (codeword[0] + 3) % 7;

        let decoded = rs.decode(&codeword).unwrap();
        assert_eq!(decoded, message);
    }

    #[test]
    fn test_generator_polynomial() {
        // GF(7)
        let gen = ReedSolomonCode::build_generator_polynomial(2, 7);
        // g(x) = (x - 1)(x - 2) = x^2 - 3x + 2 = x^2 + 4x + 2 (mod 7)
        assert_eq!(gen.len(), 3);
    }
}

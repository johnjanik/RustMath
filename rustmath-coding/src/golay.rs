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

        // Generator matrix for binary Golay code in systematic form [I | P],
        // where I is the 12×12 identity and P is a 12×11 matrix.
        //
        // This is the row-reduced (systematic) form of the cyclic code with
        // generator polynomial g(x) = x^11+x^10+x^6+x^5+x^4+x^2+1, the standard
        // generator of the [23,12,7] binary Golay code (a divisor of x^23-1 over
        // GF(2)). Verified independently (python3) against the known weight
        // enumerator A0=1, A7=253, A8=506, A11=1288, A12=1288, A15=506, A16=253,
        // A23=1, and against the extended [24,12,8] code's weight enumerator
        // A0=1, A8=759, A12=2576, A16=759, A24=1 (759 octads = the 759 blocks of
        // the Steiner system S(5,8,24)).
        //
        // NOTE: an earlier hand-typed matrix here was NOT the true Golay code
        // (its weight enumerator had nonzero A3, A4, ... terms and its extended
        // form had only 367 weight-8 words instead of 759) even though the
        // methods below still claimed minimum_distance()==7 and
        // error_correction_capability()==3 for it. That was a real bug
        // (fabricated invariants that didn't match the actual matrix), not a
        // test bug.
        let g = vec![
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            vec![0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
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

        // Generator matrix for ternary Golay code in systematic form [I | P].
        //
        // This is the row-reduced (systematic) form of the cyclic code over
        // GF(3) with generator polynomial g(x) = x^5-x^3+x^2-x-1 (a degree-5
        // divisor of x^11-1 over GF(3)), the standard generator of the
        // [11,6,5] ternary Golay code. Verified independently (python3)
        // against the known weight enumerator A0=1, A5=132, A6=132, A8=330,
        // A9=110, A11=24 (summing to 3^6=729).
        //
        // NOTE: an earlier hand-typed matrix here was NOT the true ternary
        // Golay code (its weight enumerator had nonzero A3/A4 terms, i.e. true
        // minimum distance 3, not 5) even though minimum_distance() and
        // error_correction_capability() below still claimed 5 and 2. That was
        // a real bug (fabricated invariants not matching the actual matrix),
        // not a test bug.
        let g = vec![
            vec![1, 0, 0, 0, 0, 0, 2, 2, 1, 2, 0],
            vec![0, 1, 0, 0, 0, 0, 0, 2, 2, 1, 2],
            vec![0, 0, 1, 0, 0, 0, 2, 2, 0, 1, 1],
            vec![0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
            vec![0, 0, 0, 0, 1, 0, 1, 2, 2, 2, 1],
            vec![0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 2],
        ];

        // GF(3): entries above include the value 2, which is only meaningful
        // mod 3 (mod 2 it would collapse to 0 and corrupt the code).
        let code = LinearCode::from_generator_matrix(g, 3);

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

    // Enumerate every codeword of a k x n generator matrix over GF(p) and
    // return a weight -> count histogram. k is small (<=12) in every caller
    // here, so brute force (p^k codewords) is cheap.
    fn weight_enumerator(g: &[Vec<u64>], p: u64) -> std::collections::BTreeMap<usize, usize> {
        let k = g.len();
        let n = g[0].len();
        let total: u64 = p.pow(k as u32);
        let mut hist = std::collections::BTreeMap::new();

        for msg_int in 0..total {
            let mut msg = vec![0u64; k];
            let mut t = msg_int;
            for m in msg.iter_mut() {
                *m = t % p;
                t /= p;
            }
            let mut weight = 0usize;
            for col in 0..n {
                let mut sum = 0u64;
                for row in 0..k {
                    sum = (sum + msg[row] * g[row][col]) % p;
                }
                if sum != 0 {
                    weight += 1;
                }
            }
            *hist.entry(weight).or_insert(0) += 1;
        }

        hist
    }

    #[test]
    fn test_binary_golay_weight_enumerator() {
        // Ground truth (verified independently in python3 against the
        // standard cyclic generator polynomial x^11+x^10+x^6+x^5+x^4+x^2+1,
        // a divisor of x^23-1 over GF(2)): the [23,12,7] binary Golay code
        // has weight enumerator A0=1, A7=253, A8=506, A11=1288, A12=1288,
        // A15=506, A16=253, A23=1 (sums to 2^12=4096). In particular the
        // *true* minimum nonzero weight is 7, matching what
        // `minimum_distance()` claims.
        let golay = BinaryGolayCode::new();
        let hist = weight_enumerator(golay.generator_matrix(), 2);
        let expected: std::collections::BTreeMap<usize, usize> = [
            (0, 1),
            (7, 253),
            (8, 506),
            (11, 1288),
            (12, 1288),
            (15, 506),
            (16, 253),
            (23, 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(hist, expected);
        assert_eq!(hist.values().sum::<usize>(), 1 << 12);
    }

    #[test]
    fn test_extended_binary_golay_weight_enumerator_759_octads() {
        // Ground truth: the extended [24,12,8] binary Golay code's nonzero
        // codeword weights are exactly {8,12,16,24} (it is self-dual and
        // doubly-even) with A8=759, A12=2576, A16=759, A24=1. The 759
        // weight-8 codewords ("octads") are precisely the blocks of the
        // Steiner system S(5,8,24).
        let golay = ExtendedBinaryGolayCode::new();
        let base_g = golay.base.generator_matrix();
        let k = base_g.len();
        let n = base_g[0].len();
        let mut extended_g = base_g.clone();
        for row in extended_g.iter_mut() {
            let parity = row.iter().sum::<u64>() % 2;
            row.push(parity);
        }
        let hist = weight_enumerator(&extended_g, 2);
        let expected: std::collections::BTreeMap<usize, usize> =
            [(0, 1), (8, 759), (12, 2576), (16, 759), (24, 1)]
                .into_iter()
                .collect();
        assert_eq!(hist, expected);
        assert_eq!(hist[&8], 759, "extended binary Golay code must have exactly 759 octads");
        assert_eq!(k, 12);
        assert_eq!(n, 23);
    }

    #[test]
    fn test_ternary_golay_weight_enumerator() {
        // Ground truth (verified independently in python3 against the
        // standard cyclic generator polynomial x^5-x^3+x^2-x-1, a divisor of
        // x^11-1 over GF(3)): the [11,6,5] ternary Golay code has weight
        // enumerator A0=1, A5=132, A6=132, A8=330, A9=110, A11=24 (sums to
        // 3^6=729). In particular the true minimum nonzero weight is 5,
        // matching what `minimum_distance()` claims.
        let golay = TernaryGolayCode::new();
        let hist = weight_enumerator(golay.generator_matrix(), 3);
        let expected: std::collections::BTreeMap<usize, usize> = [
            (0, 1),
            (5, 132),
            (6, 132),
            (8, 330),
            (9, 110),
            (11, 24),
        ]
        .into_iter()
        .collect();
        assert_eq!(hist, expected);
        assert_eq!(hist.values().sum::<usize>(), 3usize.pow(6));
    }

    #[test]
    fn test_binary_golay_exhaustive_up_to_3_error_correction() {
        // Ground truth: a [23,12,7] perfect code corrects every error pattern
        // of weight <= 3 (t = floor((7-1)/2) = 3), for every codeword. Fixing
        // one nonzero message and exhaustively trying every weight-1, -2 and
        // -3 error pattern (23 + 253 + 1771 = 2047 patterns) exercises
        // `find_error_pattern`'s brute-force search across all three tiers.
        let golay = BinaryGolayCode::new();
        let message = vec![1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let codeword = golay.encode(&message).unwrap();
        let n = 23;

        for i in 0..n {
            let mut received = codeword.clone();
            received[i] ^= 1;
            assert_eq!(golay.decode(&received).unwrap(), message, "single error at {}", i);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let mut received = codeword.clone();
                received[i] ^= 1;
                received[j] ^= 1;
                assert_eq!(
                    golay.decode(&received).unwrap(),
                    message,
                    "double error at {},{}",
                    i,
                    j
                );
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                for l in (j + 1)..n {
                    let mut received = codeword.clone();
                    received[i] ^= 1;
                    received[j] ^= 1;
                    received[l] ^= 1;
                    assert_eq!(
                        golay.decode(&received).unwrap(),
                        message,
                        "triple error at {},{},{}",
                        i,
                        j,
                        l
                    );
                }
            }
        }
    }
}

//! BCH (Bose-Chaudhuri-Hocquenghem) codes
//!
//! Binary BCH codes of length `n = 2^m - 1`. The generator polynomial is
//! derived honestly for every `(m, t)`:
//! `g(x) = lcm(m_1(x), m_2(x), ..., m_{2t}(x))` — the product of the
//! *distinct* minimal polynomials over GF(2) of `alpha^1..alpha^{2t}`, where
//! `alpha` is a primitive element of GF(2^m) found by factoring `2^m - 1`
//! and checking orders (nothing is hardcoded). The `2t` consecutive roots
//! give the BCH bound `d >= 2t + 1`, so the designed distance is genuine.
//!
//! Honest decoder limitation: `decode` locates errors by *exhaustive search*
//! over error patterns of weight `<= min(t, 3)` (it is not an algebraic
//! Berlekamp-Massey decoder); correctable patterns of weight above that
//! bound return `Err` rather than a wrong answer. For an algebraic
//! BM + Chien + Forney decoder see
//! [`crate::reed_solomon_generic::GenericReedSolomonCode`].

use crate::reed_solomon_generic::find_primitive_element;
use rustmath_core::NumericConversion;
use rustmath_finitefields::FiniteField;
use rustmath_integers::Integer;
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
    /// The generator polynomial is the product of the distinct minimal
    /// polynomials of `alpha^1..alpha^{2t}` for a primitive `alpha` of
    /// GF(2^m), so the designed distance `2t + 1` is guaranteed by the BCH
    /// bound. The dimension `k = n - deg g` follows from the construction
    /// (e.g. `m = 4, t = 2` gives the [15, 7] code).
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::BCHCode;
    ///
    /// let bch = BCHCode::new(4, 2); // [15, 7] code, corrects 2 errors
    /// ```
    pub fn new(m: usize, t: usize) -> Self {
        assert!(m >= 3, "Parameter m must be at least 3");
        assert!(t >= 1, "Must be able to correct at least 1 error");

        let n = (1 << m) - 1; // 2^m - 1
        let designed_distance = 2 * t + 1;
        assert!(
            designed_distance <= n,
            "designed distance 2t + 1 = {designed_distance} exceeds code length {n}"
        );

        // Build generator polynomial (honest derivation over GF(2^m)).
        let generator_poly = Self::build_generator_polynomial(m, t);
        assert!(
            generator_poly.len() <= n,
            "generator polynomial degree {} leaves no message symbols (t too large for m)",
            generator_poly.len() - 1
        );
        let k = n - (generator_poly.len() - 1);

        BCHCode {
            n,
            k,
            designed_distance,
            m,
            generator_poly,
            field_char: 2,
        }
    }

    /// Create BCH code with specific parameters [n, k, δ].
    ///
    /// Returns `Err` if `n` is not of the form `2^m - 1`, or if the BCH code
    /// with designed distance δ does not actually have dimension `k` (an
    /// honest mismatch report — the old behavior silently returned a code of
    /// a different dimension).
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
        let code = Self::new(m, t);
        if code.k != k {
            return Err(format!(
                "BCH code of length {n} with designed distance {delta} has dimension {}, \
                 not the requested {k}",
                code.k
            ));
        }
        Ok(code)
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

    /// Decode a received word.
    ///
    /// Honest limitations, both of them:
    /// * error LOCATION is by exhaustive search over patterns of weight
    ///   `<= min(t, 3)` (see the module docs), not an algebraic decoder;
    /// * this is BOUNDED-DISTANCE decoding, so like every such decoder it
    ///   can only guarantee correctness for `<= t` actual errors. A heavier
    ///   error that happens to land within distance `t` of a DIFFERENT
    ///   codeword decodes silently to that wrong codeword — and for a
    ///   PERFECT code such as BCH(4,1) = [15,11] every word is within
    ///   distance t of some codeword, so a `> t` error is NEVER detected.
    ///   `Err` is returned only when no `<= min(t, 3)` pattern matches the
    ///   syndrome; it is not, and cannot be, a guarantee against
    ///   miscorrection beyond capacity.
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

    // Build the generator polynomial as the LCM of minimal polynomials:
    // g(x) = lcm(m_1(x), m_2(x), ..., m_{2t}(x)), where m_i(x) is the
    // minimal polynomial over GF(2) of alpha^i for a primitive alpha of
    // GF(2^m). Distinct minimal polynomials are distinct irreducibles, so
    // the lcm is their product. (This replaced a hardcoded table whose
    // entries were partly wrong — e.g. the old (4,1) "generator"
    // 1 + x^2 + x^3 + x^4 is divisible by x + 1 and does not divide
    // x^15 - 1 — and whose fallback for unlisted (m,t) silently returned
    // the (4,1) polynomial.)
    fn build_generator_polynomial(m: usize, t: usize) -> Vec<u64> {
        let field = FiniteField::new(Integer::from(2u32), m)
            .expect("GF(2^m) construction cannot fail for m >= 3");
        let alpha = find_primitive_element(&field)
            .unwrap_or_else(|e| panic!("primitive element search in GF(2^{m}) failed: {e}"));

        // Distinct minimal polynomials of alpha^1 .. alpha^{2t}.
        let mut minimal_polys: Vec<Vec<u64>> = Vec::new();
        let mut apow = alpha.clone();
        for _ in 1..=(2 * t) {
            let mp: Vec<u64> = apow
                .minimal_polynomial()
                .iter()
                .map(|c| c.to_u64().expect("GF(2) coefficient") % 2)
                .collect();
            if !minimal_polys.contains(&mp) {
                minimal_polys.push(mp);
            }
            apow = apow * alpha.clone();
        }

        // Product over GF(2), little-endian.
        let mut g = vec![1u64];
        for mp in &minimal_polys {
            let mut next = vec![0u64; g.len() + mp.len() - 1];
            for (i, &x) in g.iter().enumerate() {
                if x == 1 {
                    for (j, &y) in mp.iter().enumerate() {
                        next[i + j] ^= y & 1;
                    }
                }
            }
            g = next;
        }
        g
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

    // Find error locations from syndrome by exhaustive search over error
    // patterns of weight <= min(t, 3). NOT an algebraic decoder; see the
    // module docs for the honest limitation.
    fn find_error_locations(&self, syndrome: &[u64]) -> Result<Vec<usize>, String> {
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

    /// Generator polynomials pinned by an independent Python derivation
    /// (minimal polynomials over the same Conway moduli x^4+x+1 and
    /// x^5+x^2+1 the finitefields crate uses), little-endian:
    ///   (4,1): x^4+x+1                       -> [15,11]
    ///   (4,2): x^8+x^7+x^6+x^4+1             -> [15,7]
    ///   (5,1): x^5+x^2+1                     -> [31,26]
    ///   (5,2): x^10+x^9+x^8+x^6+x^5+x^3+1    -> [31,21]
    #[test]
    fn test_bch_generator_polynomials_pinned() {
        let cases: [(usize, usize, &[u64], usize); 4] = [
            (4, 1, &[1, 1, 0, 0, 1], 11),
            (4, 2, &[1, 0, 0, 0, 1, 0, 1, 1, 1], 7),
            (5, 1, &[1, 0, 1, 0, 0, 1], 26),
            (5, 2, &[1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1], 21),
        ];
        for (m, t, g, k) in cases {
            let bch = BCHCode::new(m, t);
            assert_eq!(bch.generator_polynomial().as_slice(), g, "(m,t)=({m},{t})");
            assert_eq!(bch.dimension(), k, "(m,t)=({m},{t})");
        }
    }

    /// The BCH-bound certificate, checked inside GF(2^m) itself:
    /// g(alpha^j) = 0 for j = 1..2t, where alpha is the (order-verified)
    /// primitive element the construction used. Together with Singleton-free
    /// BCH theory this certifies designed distance >= 2t + 1.
    #[test]
    fn test_bch_bound_certificate() {
        use crate::reed_solomon_generic::find_primitive_element;
        use rustmath_core::Ring;
        use rustmath_finitefields::FiniteField;
        use rustmath_integers::Integer;

        for (m, t) in [(4usize, 1usize), (4, 2), (5, 2)] {
            let bch = BCHCode::new(m, t);
            let field = FiniteField::new(Integer::from(2u32), m).unwrap();
            let alpha = find_primitive_element(&field).unwrap();
            // alpha really has order n = 2^m - 1: alpha^n = 1 and
            // alpha^(n/r) != 1 for the prime divisors of n (15 = 3*5,
            // 31 prime).
            let n = (1usize << m) - 1;
            assert!(alpha.pow_int(&Integer::from(n as u64)).is_one());
            for r in [3usize, 5, 31] {
                if n % r == 0 {
                    // n/r = 1 for prime n = r, i.e. alpha != 1 — still valid.
                    assert!(!alpha.pow_int(&Integer::from((n / r) as u64)).is_one());
                }
            }
            for j in 1..=(2 * t) {
                let point = alpha.pow_int(&Integer::from(j as u64));
                // evaluate g at alpha^j by Horner over GF(2^m)
                let mut acc = field.zero();
                for &c in bch.generator_polynomial().iter().rev() {
                    acc = acc * point.clone() + field.from_int(Integer::from(c));
                }
                assert!(
                    acc.is_zero(),
                    "(m,t)=({m},{t}): g(alpha^{j}) != 0 — BCH bound violated"
                );
            }
        }
    }

    /// Real error correction round trips (the old suite never flipped a
    /// bit): t=1 corrects any single flip, t=2 corrects double flips.
    #[test]
    fn test_bch_error_correction_roundtrip() {
        let bch1 = BCHCode::new(4, 1); // [15,11]
        let message: Vec<u64> = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1];
        let codeword = bch1.encode(&message).unwrap();
        for pos in 0..15 {
            let mut received = codeword.clone();
            received[pos] ^= 1;
            assert_eq!(
                bch1.decode(&received).unwrap(),
                message,
                "single flip at {pos} not corrected"
            );
        }

        let bch2 = BCHCode::new(4, 2); // [15,7]
        let message2: Vec<u64> = vec![1, 0, 1, 1, 0, 0, 1];
        let codeword2 = bch2.encode(&message2).unwrap();
        for (p1, p2) in [(0usize, 7usize), (3, 4), (10, 14)] {
            let mut received = codeword2.clone();
            received[p1] ^= 1;
            received[p2] ^= 1;
            assert_eq!(
                bch2.decode(&received).unwrap(),
                message2,
                "double flip at ({p1},{p2}) not corrected"
            );
        }
    }

    /// with_parameters reports dimension mismatches honestly.
    #[test]
    fn test_with_parameters_honest_dimension() {
        assert!(BCHCode::with_parameters(15, 7, 5).is_ok());
        let err = BCHCode::with_parameters(15, 9, 5).unwrap_err();
        assert!(err.contains("dimension 7"), "unexpected error: {err}");
    }
}

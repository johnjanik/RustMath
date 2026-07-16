//! Reed-Solomon codes over prime fields GF(p) (u64 convenience API)
//!
//! [`ReedSolomonCode`] is a thin `u64` wrapper around
//! [`GenericReedSolomonCode`](crate::reed_solomon_generic::GenericReedSolomonCode)
//! instantiated at [`PrimeField`]. The code is the cyclic MDS code of length
//! `n | p - 1` with roots `alpha^1..alpha^{n-k}`, where `alpha = g^{(p-1)/n}`
//! for the smallest primitive root `g` of `GF(p)` (found honestly: the order
//! checks factor `p - 1`; nothing is hardcoded).
//!
//! * **Encoding** is systematic in the cyclic-code view: the codeword is the
//!   coefficient vector `[parity | message]` of `c(x) = x^{n-k} m(x) -
//!   (x^{n-k} m(x) mod g(x))`, `g(x) = prod_{j=1}^{n-k} (x - alpha^j)`.
//! * **Decoding** is Berlekamp-Massey + Chien search + Forney's formula,
//!   correcting up to `t = floor((n-k)/2)` symbol errors, with a
//!   post-correction syndrome re-check so beyond-capacity patterns fail
//!   honestly whenever they are detectable. (This replaced an earlier
//!   decoder that brute-forced 1- and 2-error patterns while claiming to be
//!   Peterson-Gorenstein-Zierler, over evaluation points `1..n` that made
//!   `d = n - k + 1` false in general.)
//!
//! The minimum distance really is `n - k + 1`: the `n - k` consecutive
//! powers of an order-`n` element give the BCH bound `d >= n - k + 1`, and
//! Singleton gives the reverse inequality.

use crate::reed_solomon_generic::{find_primitive_root_gfp, GenericReedSolomonCode};
use rustmath_core::NumericConversion;
use rustmath_finitefields::PrimeField;
use rustmath_integers::Integer;
use std::fmt;

/// A Reed-Solomon `[n, k, n-k+1]` code over GF(p), `n | p - 1`.
#[derive(Clone, Debug)]
pub struct ReedSolomonCode {
    /// Code length n (must divide p - 1)
    n: usize,
    /// Code dimension k
    k: usize,
    /// Field characteristic (prime p)
    field_char: u64,
    /// alpha = g^{(p-1)/n} for the smallest primitive root g of GF(p)
    alpha: u64,
    /// The generic engine doing the real work
    engine: GenericReedSolomonCode<PrimeField>,
}

impl ReedSolomonCode {
    /// Create a new Reed-Solomon code over GF(p).
    ///
    /// # Arguments
    /// * `n` - Code length; must satisfy `n | p - 1` so that GF(p) contains
    ///   an element of multiplicative order exactly `n`
    /// * `k` - Message length (dimension), `1 <= k < n`
    /// * `field_char` - The prime p
    ///
    /// The code corrects up to `t = floor((n-k)/2)` symbol errors.
    ///
    /// # Panics
    /// Panics if `field_char` is not prime, `k` is not in `1..n`, or
    /// `n` does not divide `p - 1` (each with a precise message).
    ///
    /// # Examples
    /// ```
    /// use rustmath_coding::ReedSolomonCode;
    ///
    /// // [6, 4, 3] RS code over GF(7) (6 divides 7 - 1); corrects 1 error.
    /// let rs = ReedSolomonCode::new(6, 4, 7);
    /// assert_eq!(rs.minimum_distance(), 3);
    /// ```
    pub fn new(n: usize, k: usize, field_char: u64) -> Self {
        assert!(n > k, "Code length must be greater than dimension");
        assert!(k >= 1, "Dimension must be at least 1");
        assert!(
            field_char >= 3 && (field_char as usize - 1).is_multiple_of(n),
            "Code length {n} must divide p - 1 = {} (GF({field_char}) has no element of order {n})",
            field_char.saturating_sub(1),
        );

        let g = find_primitive_root_gfp(field_char)
            .unwrap_or_else(|e| panic!("primitive root search failed: {e}"));
        let exponent = (field_char - 1) / n as u64;
        let alpha_u64 = {
            // g^((p-1)/n) mod p with u128 intermediates
            let (mut result, mut base, mut e) =
                (1u128, g as u128 % field_char as u128, exponent);
            while e > 0 {
                if e & 1 == 1 {
                    result = result * base % field_char as u128;
                }
                base = base * base % field_char as u128;
                e >>= 1;
            }
            result as u64
        };
        let alpha = PrimeField::new(Integer::from(alpha_u64), Integer::from(field_char))
            .expect("alpha construction cannot fail for p >= 3");
        let engine = GenericReedSolomonCode::new(n, k, alpha)
            .unwrap_or_else(|e| panic!("Reed-Solomon construction failed: {e}"));

        ReedSolomonCode {
            n,
            k,
            field_char,
            alpha: alpha_u64,
            engine,
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

    /// Get the minimum distance d = n - k + 1 (MDS; see the module docs)
    pub fn minimum_distance(&self) -> usize {
        self.n - self.k + 1
    }

    /// Get the error correction capability t = ⌊(n-k)/2⌋
    pub fn error_correction_capability(&self) -> usize {
        (self.n - self.k) / 2
    }

    /// The element `alpha` of order exactly `n` defining the code.
    pub fn alpha(&self) -> u64 {
        self.alpha
    }

    /// The generator polynomial `g(x) = prod_{j=1}^{n-k} (x - alpha^j)`,
    /// little-endian coefficients in `[0, p)`.
    pub fn generator_polynomial(&self) -> Vec<u64> {
        self.engine
            .generator_polynomial()
            .iter()
            .map(|c| c.value().to_u64().expect("coefficient fits in u64"))
            .collect()
    }

    /// Encode a message systematically: the codeword is `[parity | message]`
    /// with `c(x)` divisible by the generator polynomial.
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>, String> {
        let msg = self.to_field(message, self.k, "message")?;
        let codeword = self.engine.encode_systematic(&msg)?;
        Ok(Self::to_u64s(&codeword))
    }

    /// Decode a received word: Berlekamp-Massey + Chien + Forney via the
    /// generic engine (see the module docs), returning the message symbols.
    /// Fails honestly (`Err`) on every detectable beyond-capacity pattern.
    pub fn decode(&self, received: &[u64]) -> Result<Vec<u64>, String> {
        let word = self.to_field(received, self.n, "received word")?;
        let (message, _nerrors) = self
            .engine
            .decode_systematic(&word)
            .map_err(|e| format!("Decoding failed: {e}"))?;
        Ok(Self::to_u64s(&message))
    }

    fn to_field(&self, values: &[u64], expect_len: usize, what: &str) -> Result<Vec<PrimeField>, String> {
        if values.len() != expect_len {
            return Err(format!(
                "{what} length {} does not match expected length {}",
                values.len(),
                expect_len
            ));
        }
        values
            .iter()
            .map(|&v| {
                PrimeField::new(Integer::from(v), Integer::from(self.field_char))
                    .map_err(|e| format!("bad symbol {v}: {e:?}"))
            })
            .collect()
    }

    fn to_u64s(word: &[PrimeField]) -> Vec<u64> {
        word.iter()
            .map(|x| x.value().to_u64().expect("symbol fits in u64"))
            .collect()
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
        // Python-pinned: smallest primitive root of 7 is 3, and n = p - 1
        // makes alpha = g.
        assert_eq!(rs.alpha(), 3);
    }

    #[test]
    fn test_reed_solomon_encode() {
        // GF(7); Python-pinned systematic codeword for g = 6 + 2x + x^2.
        let rs = ReedSolomonCode::new(6, 4, 7);
        let message = vec![1, 2, 3, 4];
        let codeword = rs.encode(&message).unwrap();
        assert_eq!(codeword.len(), 6);
        assert_eq!(codeword, vec![1, 3, 1, 2, 3, 4]);
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

    /// Generator polynomial over GF(7): alpha = 3, so
    /// g(x) = (x - 3)(x - 3^2) = (x - 3)(x - 2) = 6 + 2x + x^2 (mod 7),
    /// derived independently in Python.
    #[test]
    fn test_generator_polynomial() {
        let rs = ReedSolomonCode::new(6, 4, 7);
        assert_eq!(rs.generator_polynomial(), vec![6, 2, 1]);
    }

    /// RS(16, 10) over GF(17) through the u64 API: 3 errors corrected
    /// exactly, 4 errors rejected honestly (the same patterns are pinned by
    /// the Python reference pipeline in reed_solomon_generic).
    #[test]
    fn test_reed_solomon_16_10_gf17() {
        let rs = ReedSolomonCode::new(16, 10, 17);
        assert_eq!(rs.error_correction_capability(), 3);
        let message: Vec<u64> = (1..=10).collect();
        let codeword = rs.encode(&message).unwrap();

        let mut received = codeword.clone();
        for (pos, val) in [(2usize, 5u64), (7, 9), (11, 1)] {
            received[pos] = (received[pos] + val) % 17;
        }
        assert_eq!(rs.decode(&received).unwrap(), message);

        let mut received4 = codeword.clone();
        for (pos, val) in [(2usize, 5u64), (7, 9), (11, 1), (14, 3)] {
            received4[pos] = (received4[pos] + val) % 17;
        }
        assert!(rs.decode(&received4).is_err(), "4 > t errors must not decode silently");
    }

    /// n that does not divide p - 1 is rejected up front (GF(7) has no
    /// element of order 5).
    #[test]
    #[should_panic(expected = "must divide")]
    fn test_invalid_length_rejected() {
        ReedSolomonCode::new(5, 3, 7);
    }
}

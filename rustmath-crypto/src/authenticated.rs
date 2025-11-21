//! Authenticated encryption modes
//!
//! Provides encryption with authentication (AEAD - Authenticated Encryption with Associated Data)
//! Currently implements GCM (Galois/Counter Mode)


/// GCM (Galois/Counter Mode) - Authenticated Encryption with Associated Data
///
/// GCM provides both confidentiality and authenticity. It uses CTR mode for encryption
/// and GHASH (based on Galois field multiplication) for authentication.
///
/// This is a simplified implementation using a placeholder block cipher.
/// In production, use with AES or another strong block cipher.
pub struct GCM {
    /// Authentication key (H)
    auth_key: [u8; 16],
    /// Encryption key
    key: Vec<u8>,
}

impl GCM {
    /// Create a new GCM instance
    ///
    /// # Arguments
    /// * `key` - Encryption key (16, 24, or 32 bytes for AES)
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::authenticated::GCM;
    ///
    /// let key = [0u8; 16];
    /// let gcm = GCM::new(&key);
    /// ```
    pub fn new(key: &[u8]) -> Self {
        assert!(
            key.len() == 16 || key.len() == 24 || key.len() == 32,
            "Key must be 16, 24, or 32 bytes"
        );

        // Generate authentication key H = E(K, 0^128)
        let zero_block = [0u8; 16];
        let auth_key = Self::simple_block_encrypt(key, &zero_block);

        GCM {
            auth_key,
            key: key.to_vec(),
        }
    }

    /// Simplified block cipher (placeholder - in production use AES)
    ///
    /// This is a simple substitution-permutation network for demonstration.
    /// Real implementations should use AES.
    fn simple_block_encrypt(key: &[u8], block: &[u8; 16]) -> [u8; 16] {
        let mut result = *block;

        // Simple rounds (this is NOT secure, just for demonstration)
        for round in 0..10 {
            // Add round key
            for i in 0..16 {
                result[i] ^= key[i % key.len()] ^ (round as u8);
            }

            // Simple S-box (byte substitution)
            for byte in result.iter_mut() {
                *byte = Self::simple_sbox(*byte);
            }

            // Simple permutation
            result = Self::simple_permute(&result);
        }

        result
    }

    /// Simple S-box (for demonstration only)
    fn simple_sbox(byte: u8) -> u8 {
        // A simple non-linear transformation
        byte.wrapping_mul(251).wrapping_add(179).rotate_left(3)
    }

    /// Simple permutation (for demonstration only)
    fn simple_permute(block: &[u8; 16]) -> [u8; 16] {
        let mut result = [0u8; 16];
        let perm = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3];
        for i in 0..16 {
            result[i] = block[perm[i]];
        }
        result
    }

    /// Increment a 128-bit counter (little-endian)
    fn increment_counter(counter: &mut [u8; 16]) {
        for i in (0..16).rev() {
            counter[i] = counter[i].wrapping_add(1);
            if counter[i] != 0 {
                break;
            }
        }
    }

    /// GHASH function - authentication using Galois field multiplication
    ///
    /// Computes GHASH(H, A, C) where:
    /// - H is the authentication key
    /// - A is additional authenticated data
    /// - C is the ciphertext
    fn ghash(h: &[u8; 16], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
        let mut result = [0u8; 16];

        // Process AAD
        for chunk in aad.chunks(16) {
            let mut block = [0u8; 16];
            block[..chunk.len()].copy_from_slice(chunk);

            // XOR with result
            for i in 0..16 {
                result[i] ^= block[i];
            }

            // Multiply in GF(2^128)
            result = Self::gf128_multiply(&result, h);
        }

        // Process ciphertext
        for chunk in ciphertext.chunks(16) {
            let mut block = [0u8; 16];
            block[..chunk.len()].copy_from_slice(chunk);

            // XOR with result
            for i in 0..16 {
                result[i] ^= block[i];
            }

            // Multiply in GF(2^128)
            result = Self::gf128_multiply(&result, h);
        }

        // Process lengths
        let aad_bits = (aad.len() as u64) * 8;
        let ct_bits = (ciphertext.len() as u64) * 8;

        let mut len_block = [0u8; 16];
        len_block[0..8].copy_from_slice(&aad_bits.to_be_bytes());
        len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());

        for i in 0..16 {
            result[i] ^= len_block[i];
        }

        result = Self::gf128_multiply(&result, h);

        result
    }

    /// Multiply two elements in GF(2^128)
    ///
    /// Uses the irreducible polynomial: x^128 + x^7 + x^2 + x + 1
    fn gf128_multiply(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
        let mut result = [0u8; 16];
        let mut v = *y;

        for i in 0..128 {
            // If the i-th bit of x is set
            let byte_idx = i / 8;
            let bit_idx = 7 - (i % 8);

            if (x[byte_idx] >> bit_idx) & 1 == 1 {
                // result ^= v
                for j in 0..16 {
                    result[j] ^= v[j];
                }
            }

            // Check if the leftmost bit of v is set
            let leftmost_bit = (v[0] >> 7) & 1;

            // v >>= 1 (right shift)
            for j in (1..16).rev() {
                v[j] = (v[j] >> 1) | (v[j - 1] << 7);
            }
            v[0] >>= 1;

            // If leftmost bit was set, XOR with R = 0xE1000...
            if leftmost_bit == 1 {
                v[15] ^= 0xE1;
            }
        }

        result
    }

    /// Encrypt with authentication
    ///
    /// # Arguments
    /// * `nonce` - Initialization vector (12 bytes recommended)
    /// * `plaintext` - Data to encrypt
    /// * `aad` - Additional authenticated data (not encrypted, but authenticated)
    ///
    /// # Returns
    /// Tuple of (ciphertext, authentication_tag)
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::authenticated::GCM;
    ///
    /// let key = [0u8; 16];
    /// let nonce = [0u8; 12];
    /// let plaintext = b"Hello, World!";
    /// let aad = b"metadata";
    ///
    /// let gcm = GCM::new(&key);
    /// let (ciphertext, tag) = gcm.encrypt(&nonce, plaintext, aad);
    /// ```
    pub fn encrypt(&self, nonce: &[u8], plaintext: &[u8], aad: &[u8]) -> (Vec<u8>, [u8; 16]) {
        assert_eq!(nonce.len(), 12, "Nonce should be 12 bytes");

        // Construct initial counter block (J0)
        let mut counter = [0u8; 16];
        counter[..12].copy_from_slice(nonce);
        counter[15] = 1; // Start counter at 1

        // Generate keystream and encrypt
        let mut ciphertext = Vec::with_capacity(plaintext.len());
        let mut current_counter = counter;

        for chunk in plaintext.chunks(16) {
            Self::increment_counter(&mut current_counter);
            let keystream = Self::simple_block_encrypt(&self.key, &current_counter);

            for (i, &byte) in chunk.iter().enumerate() {
                ciphertext.push(byte ^ keystream[i]);
            }
        }

        // Compute authentication tag
        let ghash_result = Self::ghash(&self.auth_key, aad, &ciphertext);

        // Encrypt GHASH result with counter = 1
        let tag_mask = Self::simple_block_encrypt(&self.key, &counter);
        let mut tag = [0u8; 16];
        for i in 0..16 {
            tag[i] = ghash_result[i] ^ tag_mask[i];
        }

        (ciphertext, tag)
    }

    /// Decrypt and verify authentication
    ///
    /// # Arguments
    /// * `nonce` - Initialization vector (same as used for encryption)
    /// * `ciphertext` - Encrypted data
    /// * `aad` - Additional authenticated data (same as used for encryption)
    /// * `tag` - Authentication tag from encryption
    ///
    /// # Returns
    /// Some(plaintext) if authentication succeeds, None if it fails
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::authenticated::GCM;
    ///
    /// let key = [0u8; 16];
    /// let nonce = [0u8; 12];
    /// let plaintext = b"Hello, World!";
    /// let aad = b"metadata";
    ///
    /// let gcm = GCM::new(&key);
    /// let (ciphertext, tag) = gcm.encrypt(&nonce, plaintext, aad);
    /// let decrypted = gcm.decrypt(&nonce, &ciphertext, aad, &tag);
    ///
    /// assert_eq!(decrypted, Some(plaintext.to_vec()));
    /// ```
    pub fn decrypt(
        &self,
        nonce: &[u8],
        ciphertext: &[u8],
        aad: &[u8],
        tag: &[u8; 16],
    ) -> Option<Vec<u8>> {
        assert_eq!(nonce.len(), 12, "Nonce should be 12 bytes");

        // Verify authentication tag
        let ghash_result = Self::ghash(&self.auth_key, aad, ciphertext);

        // Construct initial counter block
        let mut counter = [0u8; 16];
        counter[..12].copy_from_slice(nonce);
        counter[15] = 1;

        let tag_mask = Self::simple_block_encrypt(&self.key, &counter);
        let mut expected_tag = [0u8; 16];
        for i in 0..16 {
            expected_tag[i] = ghash_result[i] ^ tag_mask[i];
        }

        // Constant-time comparison
        let mut diff = 0u8;
        for i in 0..16 {
            diff |= tag[i] ^ expected_tag[i];
        }

        if diff != 0 {
            return None; // Authentication failed
        }

        // Decrypt
        let mut plaintext = Vec::with_capacity(ciphertext.len());
        let mut current_counter = counter;

        for chunk in ciphertext.chunks(16) {
            Self::increment_counter(&mut current_counter);
            let keystream = Self::simple_block_encrypt(&self.key, &current_counter);

            for (i, &byte) in chunk.iter().enumerate() {
                plaintext.push(byte ^ keystream[i]);
            }
        }

        Some(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcm_encrypt_decrypt() {
        let key = [0x42u8; 16];
        let nonce = [0x24u8; 12];
        let plaintext = b"Hello, GCM!";
        let aad = b"additional data";

        let gcm = GCM::new(&key);
        let (ciphertext, tag) = gcm.encrypt(&nonce, plaintext, aad);

        let decrypted = gcm.decrypt(&nonce, &ciphertext, aad, &tag);
        assert_eq!(decrypted, Some(plaintext.to_vec()));
    }

    #[test]
    fn test_gcm_wrong_tag() {
        let key = [0x42u8; 16];
        let nonce = [0x24u8; 12];
        let plaintext = b"Hello, GCM!";
        let aad = b"additional data";

        let gcm = GCM::new(&key);
        let (ciphertext, mut tag) = gcm.encrypt(&nonce, plaintext, aad);

        // Corrupt the tag
        tag[0] ^= 1;

        let decrypted = gcm.decrypt(&nonce, &ciphertext, aad, &tag);
        assert_eq!(decrypted, None); // Should fail authentication
    }

    #[test]
    fn test_gcm_wrong_aad() {
        let key = [0x42u8; 16];
        let nonce = [0x24u8; 12];
        let plaintext = b"Hello, GCM!";
        let aad = b"additional data";

        let gcm = GCM::new(&key);
        let (ciphertext, tag) = gcm.encrypt(&nonce, plaintext, aad);

        // Try to decrypt with different AAD
        let wrong_aad = b"wrong data";
        let decrypted = gcm.decrypt(&nonce, &ciphertext, wrong_aad, &tag);
        assert_eq!(decrypted, None); // Should fail authentication
    }

    #[test]
    fn test_gcm_empty_plaintext() {
        let key = [0x42u8; 16];
        let nonce = [0x24u8; 12];
        let plaintext = b"";
        let aad = b"only aad";

        let gcm = GCM::new(&key);
        let (ciphertext, tag) = gcm.encrypt(&nonce, plaintext, aad);

        assert_eq!(ciphertext.len(), 0);

        let decrypted = gcm.decrypt(&nonce, &ciphertext, aad, &tag);
        assert_eq!(decrypted, Some(vec![]));
    }

    #[test]
    fn test_gf128_multiply() {
        // Test that multiplication is commutative
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let ab = GCM::gf128_multiply(&a, &b);
        let ba = GCM::gf128_multiply(&b, &a);

        assert_eq!(ab, ba);
    }

    #[test]
    fn test_gf128_zero() {
        // Multiplying by zero should give zero
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let zero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let result = GCM::gf128_multiply(&a, &zero);
        assert_eq!(result, zero);
    }
}

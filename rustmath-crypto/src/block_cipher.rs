//! Block ciphers and S-boxes
//!
//! Simplified implementations for educational purposes

/// An S-box (Substitution box) maps input bytes to output bytes
#[derive(Debug, Clone)]
pub struct SBox {
    /// Forward substitution table (256 entries)
    pub forward: Vec<u8>,
    /// Inverse substitution table
    pub inverse: Vec<u8>,
}

impl SBox {
    /// Create a new S-box from a forward substitution table
    pub fn new(forward: Vec<u8>) -> Self {
        assert_eq!(forward.len(), 256);

        // Generate inverse table
        let mut inverse = vec![0u8; 256];
        for (input, &output) in forward.iter().enumerate() {
            inverse[output as usize] = input as u8;
        }

        SBox { forward, inverse }
    }

    /// Apply forward substitution
    pub fn substitute(&self, byte: u8) -> u8 {
        self.forward[byte as usize]
    }

    /// Apply inverse substitution
    pub fn inverse_substitute(&self, byte: u8) -> u8 {
        self.inverse[byte as usize]
    }

    /// Substitute an entire block
    pub fn substitute_block(&self, block: &[u8]) -> Vec<u8> {
        block.iter().map(|&b| self.substitute(b)).collect()
    }

    /// Inverse substitute an entire block
    pub fn inverse_substitute_block(&self, block: &[u8]) -> Vec<u8> {
        block.iter().map(|&b| self.inverse_substitute(b)).collect()
    }
}

/// Generate a simple (insecure) S-box for demonstration
pub fn generate_simple_sbox() -> SBox {
    let mut forward = Vec::with_capacity(256);

    // Simple permutation: byte -> (byte * 7 + 13) mod 256
    // NOT CRYPTOGRAPHICALLY SECURE - for educational purposes only
    for i in 0..256 {
        forward.push(((i * 7 + 13) % 256) as u8);
    }

    SBox::new(forward)
}

/// Simplified Feistel network (like DES structure)
pub struct FeistelCipher {
    /// Number of rounds
    pub rounds: usize,
    /// Round keys
    pub keys: Vec<u64>,
}

impl FeistelCipher {
    /// Create a new Feistel cipher with given number of rounds
    pub fn new(key: u64, rounds: usize) -> Self {
        // Generate round keys from master key (simplified)
        let mut keys = Vec::new();
        let mut k = key;

        for _ in 0..rounds {
            keys.push(k);
            // Simple key schedule: rotate and XOR
            k = k.rotate_left(8) ^ 0x0123456789ABCDEF;
        }

        FeistelCipher { rounds, keys }
    }

    /// Simple round function
    fn round_function(&self, half: u32, round_key: u64) -> u32 {
        // XOR with round key, then apply some mixing
        let mixed = (half as u64 ^ round_key) as u32;

        // Simple mixing: multiply and rotate
        mixed.wrapping_mul(0x9E3779B9).rotate_left(13)
    }

    /// Encrypt a 64-bit block
    pub fn encrypt(&self, block: u64) -> u64 {
        let mut left = (block >> 32) as u32;
        let mut right = block as u32;

        for &key in &self.keys {
            let temp = right;
            right = left ^ self.round_function(right, key);
            left = temp;
        }

        // Combine halves (no final swap in last round)
        ((right as u64) << 32) | (left as u64)
    }

    /// Decrypt a 64-bit block
    pub fn decrypt(&self, block: u64) -> u64 {
        let mut left = (block >> 32) as u32;
        let mut right = block as u32;

        // Apply rounds in reverse
        for &key in self.keys.iter().rev() {
            let temp = right;
            right = left ^ self.round_function(right, key);
            left = temp;
        }

        ((right as u64) << 32) | (left as u64)
    }
}

/// Simple block cipher mode: Electronic Codebook (ECB)
/// WARNING: ECB is not secure for real use!
pub fn ecb_encrypt(plaintext: &[u8], cipher: &FeistelCipher) -> Vec<u8> {
    let mut ciphertext = Vec::new();

    for chunk in plaintext.chunks(8) {
        let mut block = [0u8; 8];
        block[..chunk.len()].copy_from_slice(chunk);

        let block_u64 = u64::from_be_bytes(block);
        let encrypted = cipher.encrypt(block_u64);

        ciphertext.extend_from_slice(&encrypted.to_be_bytes());
    }

    ciphertext
}

/// Simple block cipher mode: Electronic Codebook (ECB) decryption
pub fn ecb_decrypt(ciphertext: &[u8], cipher: &FeistelCipher) -> Vec<u8> {
    let mut plaintext = Vec::new();

    for chunk in ciphertext.chunks(8) {
        if chunk.len() == 8 {
            let block = u64::from_be_bytes(chunk.try_into().unwrap());
            let decrypted = cipher.decrypt(block);

            plaintext.extend_from_slice(&decrypted.to_be_bytes());
        }
    }

    plaintext
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox_basic() {
        let sbox = generate_simple_sbox();

        // Test that substitution is reversible
        for i in 0..256 {
            let substituted = sbox.substitute(i as u8);
            let recovered = sbox.inverse_substitute(substituted);
            assert_eq!(recovered, i as u8);
        }
    }

    #[test]
    fn test_sbox_block() {
        let sbox = generate_simple_sbox();
        let block = vec![0x01, 0x02, 0x03, 0x04];

        let substituted = sbox.substitute_block(&block);
        let recovered = sbox.inverse_substitute_block(&substituted);

        assert_eq!(block, recovered);
    }

    #[test]
    fn test_feistel_encrypt_decrypt() {
        let key = 0x0123456789ABCDEF;
        let cipher = FeistelCipher::new(key, 8);

        let plaintext = 0x0011223344556677u64;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
        assert_ne!(plaintext, ciphertext); // Should be different
    }

    #[test]
    fn test_feistel_different_keys() {
        let plaintext = 0x123456789ABCDEFu64;

        let cipher1 = FeistelCipher::new(0x1111111111111111, 4);
        let cipher2 = FeistelCipher::new(0x2222222222222222, 4);

        let ct1 = cipher1.encrypt(plaintext);
        let ct2 = cipher2.encrypt(plaintext);

        // Different keys should give different ciphertexts
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_ecb_mode() {
        let key = 0xFEDCBA9876543210;
        let cipher = FeistelCipher::new(key, 8);

        let plaintext = b"Hello World!!!!!"; // 16 bytes
        let ciphertext = ecb_encrypt(plaintext, &cipher);
        let decrypted = ecb_decrypt(&ciphertext, &cipher);

        assert_eq!(&decrypted[..plaintext.len()], plaintext);
    }

    #[test]
    fn test_sbox_all_unique() {
        let sbox = generate_simple_sbox();

        // Check that all outputs are unique (it's a permutation)
        let mut seen = std::collections::HashSet::new();
        for &output in &sbox.forward {
            assert!(seen.insert(output), "Duplicate value in S-box");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn test_feistel_avalanche() {
        let key = 0x0123456789ABCDEF;
        let cipher = FeistelCipher::new(key, 8);

        let plaintext1 = 0x0000000000000000u64;
        let plaintext2 = 0x0000000000000001u64; // Differs by 1 bit

        let ct1 = cipher.encrypt(plaintext1);
        let ct2 = cipher.encrypt(plaintext2);

        // Should differ in multiple bits (avalanche effect)
        let diff = ct1 ^ ct2;
        let bit_count = diff.count_ones();

        // At least some bits should differ
        assert!(bit_count > 0);
    }
}

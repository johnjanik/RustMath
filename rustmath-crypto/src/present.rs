//! PRESENT Block Cipher
//!
//! PRESENT is an ultra-lightweight block cipher designed for resource-constrained devices.
//! It has a 64-bit block size and supports 80-bit and 128-bit keys.
//! This implementation supports the 80-bit key variant with 31 rounds.
//!
//! Reference: "PRESENT: An Ultra-Lightweight Block Cipher" by Bogdanov et al. (2007)

/// PRESENT cipher with 64-bit blocks and 80-bit keys
#[derive(Debug, Clone)]
pub struct Present {
    /// 32 round keys (64 bits each) for 31 rounds + initial whitening
    round_keys: [u64; 32],
}

impl Present {
    /// S-box for 4-bit substitution
    const SBOX: [u8; 16] = [
        0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
        0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2,
    ];

    /// Inverse S-box
    const INV_SBOX: [u8; 16] = [
        0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD,
        0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA,
    ];

    /// Create a new PRESENT cipher with an 80-bit key
    /// The key is represented as a u128, but only the lower 80 bits are used
    pub fn new(key: u128) -> Self {
        assert!(key < (1u128 << 80), "Key must be 80 bits");

        let round_keys = Self::key_schedule(key);
        Present { round_keys }
    }

    /// Generate 32 round keys from the 80-bit master key
    fn key_schedule(key: u128) -> [u64; 32] {
        let mut round_keys = [0u64; 32];
        let mut key_state = key; // 80-bit key state

        for i in 0..32 {
            // Extract leftmost 64 bits as round key
            round_keys[i] = (key_state >> 16) as u64;

            // Update key state
            // 1. Rotate left by 61 bits
            let rotated = ((key_state << 61) | (key_state >> 19)) & ((1u128 << 80) - 1);

            // 2. Pass leftmost 4 bits through S-box
            let leftmost = (rotated >> 76) as u8;
            let s_output = Self::SBOX[leftmost as usize];
            let key_temp = (rotated & !((0xFu128) << 76)) | ((s_output as u128) << 76);

            // 3. XOR round counter (5 bits) with bits [19:15]
            let round_counter = (i + 1) as u128;
            let xor_mask = round_counter << 15;
            key_state = key_temp ^ xor_mask;
        }

        round_keys
    }

    /// Apply S-box layer to 64-bit state (16 nibbles)
    fn sbox_layer(state: u64) -> u64 {
        let mut result = 0u64;
        for i in 0..16 {
            let nibble = ((state >> (i * 4)) & 0xF) as u8;
            let s_nibble = Self::SBOX[nibble as usize];
            result |= (s_nibble as u64) << (i * 4);
        }
        result
    }

    /// Apply inverse S-box layer
    fn inv_sbox_layer(state: u64) -> u64 {
        let mut result = 0u64;
        for i in 0..16 {
            let nibble = ((state >> (i * 4)) & 0xF) as u8;
            let inv_nibble = Self::INV_SBOX[nibble as usize];
            result |= (inv_nibble as u64) << (i * 4);
        }
        result
    }

    /// Apply P-layer (bit permutation)
    /// Bit i moves to position P(i) where P(i) = (16*i) mod 63 (or 63 if i=63)
    fn player(state: u64) -> u64 {
        let mut result = 0u64;
        for i in 0..64 {
            let bit = (state >> i) & 1;
            let new_pos = if i == 63 { 63 } else { (16 * i) % 63 };
            result |= bit << new_pos;
        }
        result
    }

    /// Apply inverse P-layer
    fn inv_player(state: u64) -> u64 {
        let mut result = 0u64;
        for i in 0..64 {
            let new_pos = if i == 63 { 63 } else { (16 * i) % 63 };
            let bit = (state >> new_pos) & 1;
            result |= bit << i;
        }
        result
    }

    /// Encrypt a 64-bit block
    pub fn encrypt(&self, plaintext: u64) -> u64 {
        let mut state = plaintext;

        // 31 rounds
        for i in 0..31 {
            // Add round key
            state ^= self.round_keys[i];

            // S-box layer
            state = Self::sbox_layer(state);

            // P-layer
            state = Self::player(state);
        }

        // Final round key addition
        state ^= self.round_keys[31];

        state
    }

    /// Decrypt a 64-bit block
    pub fn decrypt(&self, ciphertext: u64) -> u64 {
        let mut state = ciphertext;

        // Final round key
        state ^= self.round_keys[31];

        // 31 rounds in reverse
        for i in (0..31).rev() {
            // Inverse P-layer
            state = Self::inv_player(state);

            // Inverse S-box layer
            state = Self::inv_sbox_layer(state);

            // Add round key
            state ^= self.round_keys[i];
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_present_encrypt_decrypt() {
        let key = 0x00000000000000000000u128; // 80-bit zero key
        let cipher = Present::new(key);

        let plaintext = 0x0000000000000000u64;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_present_known_vector() {
        // Test vector from PRESENT specification
        let key = 0x00000000000000000000u128;
        let cipher = Present::new(key);

        let plaintext = 0x0000000000000000u64;
        let ciphertext = cipher.encrypt(plaintext);

        // Expected ciphertext for all-zero plaintext and key
        let expected = 0x5579C1387B228445u64;

        assert_eq!(ciphertext, expected);
    }

    #[test]
    fn test_present_known_vector_2() {
        // Another test vector
        let key = 0xFFFFFFFFFFFFFFFFFFFFu128;
        let cipher = Present::new(key);

        let plaintext = 0x0000000000000000u64;
        let ciphertext = cipher.encrypt(plaintext);

        // Expected ciphertext
        let expected = 0xE72C46C0F5945049u64;

        assert_eq!(ciphertext, expected);
    }

    #[test]
    fn test_present_known_vector_3() {
        // Test vector with non-zero plaintext
        let key = 0x00000000000000000000u128;
        let cipher = Present::new(key);

        let plaintext = 0xFFFFFFFFFFFFFFFFu64;
        let ciphertext = cipher.encrypt(plaintext);

        // Expected ciphertext
        let expected = 0xA112FFC72F68417Bu64;

        assert_eq!(ciphertext, expected);
    }

    #[test]
    fn test_present_different_keys() {
        let plaintext = 0x123456789ABCDEFu64;

        let cipher1 = Present::new(0x12345678901234567890u128);
        let cipher2 = Present::new(0xABCDEF0123456789ABCDu128);

        let ct1 = cipher1.encrypt(plaintext);
        let ct2 = cipher2.encrypt(plaintext);

        // Different keys should produce different ciphertexts
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_sbox_inverse() {
        // Verify S-box and inverse S-box are correct
        for i in 0..16 {
            let s = Present::SBOX[i];
            let inv_s = Present::INV_SBOX[s as usize];
            assert_eq!(inv_s, i as u8);
        }
    }

    #[test]
    fn test_player_inverse() {
        let state = 0x123456789ABCDEFu64;
        let permuted = Present::player(state);
        let unpermuted = Present::inv_player(permuted);
        assert_eq!(state, unpermuted);
    }

    #[test]
    fn test_sbox_layer_inverse() {
        let state = 0xFEDCBA9876543210u64;
        let substituted = Present::sbox_layer(state);
        let unsubstituted = Present::inv_sbox_layer(substituted);
        assert_eq!(state, unsubstituted);
    }
}

//! Simplified DES (S-DES)
//!
//! This is an educational cipher designed to illustrate the principles of DES
//! with a much smaller block size (8 bits instead of 64 bits) and key size (10 bits).
//!
//! Reference: "Cryptography and Network Security" by William Stallings

/// Simplified DES cipher with 8-bit blocks and 10-bit keys
#[derive(Debug, Clone)]
pub struct SimplifiedDES {
    /// Two 8-bit subkeys generated from the master key
    k1: u8,
    k2: u8,
}

impl SimplifiedDES {
    /// Create a new Simplified DES cipher with a 10-bit key
    pub fn new(key: u16) -> Self {
        assert!(key < 1024, "Key must be 10 bits (0-1023)");

        let (k1, k2) = Self::key_schedule(key);
        SimplifiedDES { k1, k2 }
    }

    /// Generate two 8-bit subkeys from a 10-bit master key
    fn key_schedule(key: u16) -> (u8, u8) {
        // P10 permutation
        let p10 = Self::permute_10(key, &[3, 5, 2, 7, 4, 10, 1, 9, 8, 6]);

        // Split into two 5-bit halves
        let mut left = ((p10 >> 5) & 0x1F) as u8;
        let mut right = (p10 & 0x1F) as u8;

        // LS-1: Left shift each half by 1
        left = Self::left_shift_5(left, 1);
        right = Self::left_shift_5(right, 1);

        // P8 permutation to generate K1
        let combined1 = ((left as u16) << 5) | (right as u16);
        let k1 = Self::permute_8(combined1, &[6, 3, 7, 4, 8, 5, 10, 9]);

        // LS-2: Left shift each half by 2 more (total 3)
        left = Self::left_shift_5(left, 2);
        right = Self::left_shift_5(right, 2);

        // P8 permutation to generate K2
        let combined2 = ((left as u16) << 5) | (right as u16);
        let k2 = Self::permute_8(combined2, &[6, 3, 7, 4, 8, 5, 10, 9]);

        (k1, k2)
    }

    /// Permute 10 bits according to the given permutation table
    fn permute_10(input: u16, table: &[usize; 10]) -> u16 {
        let mut output = 0u16;
        for (i, &pos) in table.iter().enumerate() {
            let bit = (input >> (10 - pos)) & 1;
            output |= bit << (10 - i - 1);
        }
        output
    }

    /// Permute to 8 bits from 10 bits according to the given table
    fn permute_8(input: u16, table: &[usize; 8]) -> u8 {
        let mut output = 0u8;
        for (i, &pos) in table.iter().enumerate() {
            let bit = (input >> (10 - pos)) & 1;
            output |= (bit << (8 - i - 1)) as u8;
        }
        output
    }

    /// Left shift a 5-bit value
    fn left_shift_5(value: u8, shift: u8) -> u8 {
        ((value << shift) | (value >> (5 - shift))) & 0x1F
    }

    /// Initial permutation (IP)
    fn initial_permutation(input: u8) -> u8 {
        Self::permute_8_to_8(input, &[2, 6, 3, 1, 4, 8, 5, 7])
    }

    /// Inverse initial permutation (IP^-1)
    fn inverse_initial_permutation(input: u8) -> u8 {
        Self::permute_8_to_8(input, &[4, 1, 3, 5, 7, 2, 8, 6])
    }

    /// Permute 8 bits to 8 bits
    fn permute_8_to_8(input: u8, table: &[usize; 8]) -> u8 {
        let mut output = 0u8;
        for (i, &pos) in table.iter().enumerate() {
            let bit = (input >> (8 - pos)) & 1;
            output |= bit << (8 - i - 1);
        }
        output
    }

    /// Expansion/Permutation (E/P): expand 4 bits to 8 bits
    fn expand_permute(input: u8) -> u8 {
        let table = [4, 1, 2, 3, 2, 3, 4, 1];
        let mut output = 0u8;
        for (i, &pos) in table.iter().enumerate() {
            let bit = (input >> (4 - pos)) & 1;
            output |= bit << (8 - i - 1);
        }
        output
    }

    /// S-box 0
    fn sbox0(input: u8) -> u8 {
        let table = [
            [1, 0, 3, 2],
            [3, 2, 1, 0],
            [0, 2, 1, 3],
            [3, 1, 3, 2],
        ];
        Self::sbox_lookup(input, &table)
    }

    /// S-box 1
    fn sbox1(input: u8) -> u8 {
        let table = [
            [0, 1, 2, 3],
            [2, 0, 1, 3],
            [3, 0, 1, 0],
            [2, 1, 0, 3],
        ];
        Self::sbox_lookup(input, &table)
    }

    /// Generic S-box lookup
    fn sbox_lookup(input: u8, table: &[[u8; 4]; 4]) -> u8 {
        let row = ((input & 0x80) >> 6) | ((input & 0x08) >> 3);
        let col = (input & 0x60) >> 5;
        table[row as usize][col as usize]
    }

    /// P4 permutation
    fn p4(input: u8) -> u8 {
        let table = [2, 4, 3, 1];
        let mut output = 0u8;
        for (i, &pos) in table.iter().enumerate() {
            let bit = (input >> (4 - pos)) & 1;
            output |= bit << (4 - i - 1);
        }
        output
    }

    /// F-function (Feistel function)
    fn f_function(right: u8, subkey: u8) -> u8 {
        // Expand and permute
        let expanded = Self::expand_permute(right);

        // XOR with subkey
        let xored = expanded ^ subkey;

        // Split into two 4-bit halves and apply S-boxes
        let left_half = (xored >> 4) & 0x0F;
        let right_half = xored & 0x0F;

        let s0_out = Self::sbox0(left_half << 4);
        let s1_out = Self::sbox1(right_half << 4);

        // Combine S-box outputs
        let combined = (s0_out << 2) | s1_out;

        // P4 permutation
        Self::p4(combined)
    }

    /// Encrypt an 8-bit block
    pub fn encrypt(&self, plaintext: u8) -> u8 {
        // Initial permutation
        let ip = Self::initial_permutation(plaintext);

        // Split into left and right halves (4 bits each)
        let mut left = (ip >> 4) & 0x0F;
        let mut right = ip & 0x0F;

        // Round 1
        let f_out1 = Self::f_function(right, self.k1);
        let new_left = right;
        let new_right = left ^ f_out1;
        left = new_left;
        right = new_right;

        // Round 2
        let f_out2 = Self::f_function(right, self.k2);
        let new_left = right;
        let new_right = left ^ f_out2;

        // Final output: combine without final swap
        let combined = (new_right << 4) | new_left;

        // Inverse initial permutation
        Self::inverse_initial_permutation(combined)
    }

    /// Decrypt an 8-bit block
    pub fn decrypt(&self, ciphertext: u8) -> u8 {
        // Initial permutation
        let ip = Self::initial_permutation(ciphertext);

        // Split into left and right halves (4 bits each)
        let mut left = (ip >> 4) & 0x0F;
        let mut right = ip & 0x0F;

        // Reverse Round 2 (apply k2 first in decryption)
        let f_out2 = Self::f_function(right, self.k2);
        let new_left = right;
        let new_right = left ^ f_out2;
        left = new_left;
        right = new_right;

        // Reverse Round 1
        let f_out1 = Self::f_function(right, self.k1);
        let new_left = right;
        let new_right = left ^ f_out1;

        // Final output: combine without final swap
        let combined = (new_right << 4) | new_left;

        // Inverse initial permutation
        Self::inverse_initial_permutation(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplified_des_encrypt_decrypt() {
        let key = 0b0111111101; // 10-bit key
        let cipher = SimplifiedDES::new(key);

        let plaintext = 0b10101010; // 8-bit plaintext
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_simplified_des_known_vector() {
        // Test vector from Stallings textbook
        let key = 0b0111111101; // 10-bit key (0x1FD)
        let cipher = SimplifiedDES::new(key);

        let plaintext = 0b00000001;
        let ciphertext = cipher.encrypt(plaintext);

        // Decrypt to verify
        let decrypted = cipher.decrypt(ciphertext);
        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_simplified_des_all_zeros() {
        let key = 0b1010101010;
        let cipher = SimplifiedDES::new(key);

        let plaintext = 0b00000000;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_simplified_des_all_ones() {
        let key = 0b1111111111;
        let cipher = SimplifiedDES::new(key);

        let plaintext = 0b11111111;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_simplified_des_different_keys() {
        let plaintext = 0b11001100;

        let cipher1 = SimplifiedDES::new(0b0111111101);
        let cipher2 = SimplifiedDES::new(0b1010101010);

        let ct1 = cipher1.encrypt(plaintext);
        let ct2 = cipher2.encrypt(plaintext);

        // Different keys should produce different ciphertexts
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_key_schedule() {
        let key = 0b0111111101;
        let (k1, k2) = SimplifiedDES::key_schedule(key);

        // Keys should be different
        assert_ne!(k1, k2);

        // Both k1 and k2 are u8, so they're inherently 8-bit values
        // No need to assert bounds
    }
}

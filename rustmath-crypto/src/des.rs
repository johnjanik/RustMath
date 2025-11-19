//! DES (Data Encryption Standard)
//!
//! The Data Encryption Standard is a symmetric-key block cipher with a 64-bit block size
//! and a 56-bit key (stored in 64 bits with 8 parity bits). It uses 16 rounds of Feistel
//! network with a complex key schedule.
//!
//! NOTE: DES is considered cryptographically broken and should NOT be used for security.
//! This implementation is for educational purposes only.
//!
//! Reference: FIPS PUB 46-3

/// DES cipher with 64-bit blocks and 56-bit keys (64 bits with parity)
#[derive(Debug, Clone)]
pub struct DES {
    /// 16 round keys (48 bits each, stored in 64-bit values)
    round_keys: [u64; 16],
}

impl DES {
    /// Initial Permutation (IP)
    const IP: [usize; 64] = [
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6,
        64, 56, 48, 40, 32, 24, 16, 8,
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7,
    ];

    /// Final Permutation (IP^-1)
    const FP: [usize; 64] = [
        40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25,
    ];

    /// Expansion (E) - expands 32 bits to 48 bits
    const E: [usize; 48] = [
        32, 1, 2, 3, 4, 5,
        4, 5, 6, 7, 8, 9,
        8, 9, 10, 11, 12, 13,
        12, 13, 14, 15, 16, 17,
        16, 17, 18, 19, 20, 21,
        20, 21, 22, 23, 24, 25,
        24, 25, 26, 27, 28, 29,
        28, 29, 30, 31, 32, 1,
    ];

    /// Permutation (P) - permutes 32 bits after S-boxes
    const P: [usize; 32] = [
        16, 7, 20, 21, 29, 12, 28, 17,
        1, 15, 23, 26, 5, 18, 31, 10,
        2, 8, 24, 14, 32, 27, 3, 9,
        19, 13, 30, 6, 22, 11, 4, 25,
    ];

    /// Permuted Choice 1 (PC-1) - selects 56 bits from 64-bit key
    const PC1: [usize; 56] = [
        57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4,
    ];

    /// Permuted Choice 2 (PC-2) - selects 48 bits from 56-bit key state
    const PC2: [usize; 48] = [
        14, 17, 11, 24, 1, 5,
        3, 28, 15, 6, 21, 10,
        23, 19, 12, 4, 26, 8,
        16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55,
        30, 40, 51, 45, 33, 48,
        44, 49, 39, 56, 34, 53,
        46, 42, 50, 36, 29, 32,
    ];

    /// Number of left rotations for each round
    const SHIFTS: [usize; 16] = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1];

    /// S-boxes (8 S-boxes, each 4 rows x 16 columns)
    const SBOXES: [[[u8; 16]; 4]; 8] = [
        // S1
        [
            [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
            [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
            [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
            [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
        ],
        // S2
        [
            [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
            [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
        ],
        // S3
        [
            [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
            [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
            [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
        ],
        // S4
        [
            [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
            [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
            [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
        ],
        // S5
        [
            [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
            [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
            [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
        ],
        // S6
        [
            [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
            [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
        ],
        // S7
        [
            [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
            [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
        ],
        // S8
        [
            [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
        ],
    ];

    /// Create a new DES cipher with a 64-bit key (56 bits used, 8 parity bits)
    pub fn new(key: u64) -> Self {
        let round_keys = Self::key_schedule(key);
        DES { round_keys }
    }

    /// Generate 16 round keys from the master key
    fn key_schedule(key: u64) -> [u64; 16] {
        let mut round_keys = [0u64; 16];

        // Apply PC-1 to get 56-bit key
        let mut key_state = Self::permute(key, &Self::PC1, 64);

        // Split into two 28-bit halves
        let mut c = (key_state >> 28) & 0x0FFFFFFF;
        let mut d = key_state & 0x0FFFFFFF;

        for i in 0..16 {
            // Left rotate both halves
            let shift = Self::SHIFTS[i];
            c = Self::rotate_left_28(c, shift);
            d = Self::rotate_left_28(d, shift);

            // Combine and apply PC-2 to get 48-bit round key
            let combined = (c << 28) | d;
            round_keys[i] = Self::permute(combined, &Self::PC2, 56);
        }

        round_keys
    }

    /// Rotate a 28-bit value left
    fn rotate_left_28(value: u64, shift: usize) -> u64 {
        ((value << shift) | (value >> (28 - shift))) & 0x0FFFFFFF
    }

    /// Generic permutation function
    fn permute(input: u64, table: &[usize], input_bits: usize) -> u64 {
        let mut output = 0u64;
        for (i, &pos) in table.iter().enumerate() {
            let bit = (input >> (input_bits - pos)) & 1;
            output |= bit << (table.len() - i - 1);
        }
        output
    }

    /// Apply S-boxes to 48-bit input, producing 32-bit output
    fn apply_sboxes(input: u64) -> u64 {
        let mut output = 0u64;

        for i in 0..8 {
            // Extract 6 bits for this S-box
            let six_bits = ((input >> (42 - i * 6)) & 0x3F) as u8;

            // Row is determined by outer two bits
            let row = (((six_bits & 0x20) >> 4) | (six_bits & 0x01)) as usize;

            // Column is determined by middle four bits
            let col = ((six_bits >> 1) & 0x0F) as usize;

            // Lookup in S-box
            let s_output = Self::SBOXES[i][row][col];

            // Place 4-bit output in result
            output |= (s_output as u64) << (28 - i * 4);
        }

        output
    }

    /// F-function (Feistel function)
    fn f_function(right: u64, subkey: u64) -> u64 {
        // Expand 32 bits to 48 bits
        let expanded = Self::permute(right, &Self::E, 32);

        // XOR with subkey
        let xored = expanded ^ subkey;

        // Apply S-boxes (48 bits -> 32 bits)
        let s_output = Self::apply_sboxes(xored);

        // Apply P permutation
        Self::permute(s_output, &Self::P, 32)
    }

    /// Encrypt a 64-bit block
    pub fn encrypt(&self, plaintext: u64) -> u64 {
        // Initial permutation
        let mut state = Self::permute(plaintext, &Self::IP, 64);

        // Split into left and right halves
        let mut left = (state >> 32) as u32;
        let mut right = state as u32;

        // 16 rounds
        for i in 0..16 {
            let temp = right;
            let f_out = Self::f_function(right as u64, self.round_keys[i]) as u32;
            right = left ^ f_out;
            left = temp;
        }

        // Combine halves (swap for final)
        let combined = ((right as u64) << 32) | (left as u64);

        // Final permutation
        Self::permute(combined, &Self::FP, 64)
    }

    /// Decrypt a 64-bit block
    pub fn decrypt(&self, ciphertext: u64) -> u64 {
        // Initial permutation
        let mut state = Self::permute(ciphertext, &Self::IP, 64);

        // Split into left and right halves
        let mut left = (state >> 32) as u32;
        let mut right = state as u32;

        // 16 rounds in reverse order
        for i in (0..16).rev() {
            let temp = right;
            let f_out = Self::f_function(right as u64, self.round_keys[i]) as u32;
            right = left ^ f_out;
            left = temp;
        }

        // Combine halves (swap for final)
        let combined = ((right as u64) << 32) | (left as u64);

        // Final permutation
        Self::permute(combined, &Self::FP, 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_des_encrypt_decrypt() {
        let key = 0x133457799BBCDFF1;
        let cipher = DES::new(key);

        let plaintext = 0x0123456789ABCDEF;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_des_known_vector() {
        // Standard DES test vector
        let key = 0x133457799BBCDFF1;
        let cipher = DES::new(key);

        let plaintext = 0x0123456789ABCDEF;
        let ciphertext = cipher.encrypt(plaintext);

        // Expected ciphertext
        let expected = 0x85E813540F0AB405;

        assert_eq!(ciphertext, expected);
    }

    #[test]
    fn test_des_all_zeros() {
        let key = 0x0000000000000000;
        let cipher = DES::new(key);

        let plaintext = 0x0000000000000000;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_des_different_keys() {
        let plaintext = 0x123456789ABCDEF0;

        let cipher1 = DES::new(0x0123456789ABCDEF);
        let cipher2 = DES::new(0xFEDCBA9876543210);

        let ct1 = cipher1.encrypt(plaintext);
        let ct2 = cipher2.encrypt(plaintext);

        // Different keys should produce different ciphertexts
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_permute() {
        // Test permutation with a simple case
        let input = 0x0123456789ABCDEFu64;
        let table = [1, 2, 3, 4, 5, 6, 7, 8];
        let output = DES::permute(input, &table, 64);

        // Should extract the top 8 bits
        assert_eq!(output, 0x01);
    }

    #[test]
    fn test_rotate_left_28() {
        let value = 0x0ABCDEF0;
        let rotated = DES::rotate_left_28(value, 1);
        let expected = 0x0579BDE1; // One left rotation of 28-bit value

        assert_eq!(rotated, expected);
    }
}

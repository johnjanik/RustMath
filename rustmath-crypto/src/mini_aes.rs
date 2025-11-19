//! Mini-AES (Simplified AES)
//!
//! This is an educational cipher designed to illustrate the principles of AES
//! with a much smaller block size (16 bits instead of 128 bits) and key size (16 bits).
//!
//! Mini-AES uses a 4x4 nibble (4-bit) state instead of AES's 4x4 byte state.
//! It performs similar operations: SubNibbles, ShiftRows, MixColumns, and AddRoundKey.

/// Mini-AES cipher with 16-bit blocks and 16-bit keys
#[derive(Debug, Clone)]
pub struct MiniAES {
    /// Round keys (3 round keys for 2 rounds)
    round_keys: [u16; 3],
}

impl MiniAES {
    /// S-box for nibble substitution (16 entries)
    const SBOX: [u8; 16] = [
        0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
        0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7,
    ];

    /// Inverse S-box
    const INV_SBOX: [u8; 16] = [
        0xE, 0x3, 0x4, 0x8, 0x1, 0xC, 0xA, 0xF,
        0x7, 0xD, 0x9, 0x6, 0xB, 0x2, 0x0, 0x5,
    ];

    /// Create a new Mini-AES cipher with a 16-bit key
    pub fn new(key: u16) -> Self {
        let round_keys = Self::key_schedule(key);
        MiniAES { round_keys }
    }

    /// Generate round keys from the master key
    fn key_schedule(key: u16) -> [u16; 3] {
        let mut keys = [0u16; 3];
        keys[0] = key;

        // Round constants
        let rcon = [0x80, 0x30]; // In GF(2^4)

        for i in 0..2 {
            let prev = keys[i];

            // Extract nibbles: w0 = high byte high nibble, w1 = high byte low nibble,
            // w2 = low byte high nibble, w3 = low byte low nibble
            let w0 = ((prev >> 12) & 0xF) as u8;
            let w1 = ((prev >> 8) & 0xF) as u8;
            let w2 = ((prev >> 4) & 0xF) as u8;
            let w3 = (prev & 0xF) as u8;

            // Rotate w2, w3
            let rotated = [w3, w2];

            // SubNibbles on rotated
            let sub0 = Self::SBOX[rotated[0] as usize];
            let sub1 = Self::SBOX[rotated[1] as usize];

            // XOR with round constant
            let new_w0 = w0 ^ sub0 ^ ((rcon[i] >> 4) & 0xF);
            let new_w1 = w1 ^ sub1 ^ (rcon[i] & 0xF);
            let new_w2 = w2 ^ new_w0;
            let new_w3 = w3 ^ new_w1;

            keys[i + 1] = ((new_w0 as u16) << 12)
                | ((new_w1 as u16) << 8)
                | ((new_w2 as u16) << 4)
                | (new_w3 as u16);
        }

        keys
    }

    /// SubNibbles transformation
    fn sub_nibbles(state: u16) -> u16 {
        let n0 = Self::SBOX[((state >> 12) & 0xF) as usize];
        let n1 = Self::SBOX[((state >> 8) & 0xF) as usize];
        let n2 = Self::SBOX[((state >> 4) & 0xF) as usize];
        let n3 = Self::SBOX[(state & 0xF) as usize];

        ((n0 as u16) << 12) | ((n1 as u16) << 8) | ((n2 as u16) << 4) | (n3 as u16)
    }

    /// Inverse SubNibbles transformation
    fn inv_sub_nibbles(state: u16) -> u16 {
        let n0 = Self::INV_SBOX[((state >> 12) & 0xF) as usize];
        let n1 = Self::INV_SBOX[((state >> 8) & 0xF) as usize];
        let n2 = Self::INV_SBOX[((state >> 4) & 0xF) as usize];
        let n3 = Self::INV_SBOX[(state & 0xF) as usize];

        ((n0 as u16) << 12) | ((n1 as u16) << 8) | ((n2 as u16) << 4) | (n3 as u16)
    }

    /// ShiftRows transformation (swap the two right nibbles)
    fn shift_rows(state: u16) -> u16 {
        let n0 = (state >> 12) & 0xF;
        let n1 = (state >> 8) & 0xF;
        let n2 = (state >> 4) & 0xF;
        let n3 = state & 0xF;

        // In a 2x2 matrix: [n0, n1]
        //                  [n2, n3]
        // ShiftRows: swap n1 and n2
        (n0 << 12) | (n2 << 8) | (n1 << 4) | n3
    }

    /// Inverse ShiftRows (same as ShiftRows for 2x2)
    fn inv_shift_rows(state: u16) -> u16 {
        Self::shift_rows(state) // Swap is its own inverse
    }

    /// Multiply in GF(2^4) with irreducible polynomial x^4 + x + 1
    fn gf_mult(a: u8, b: u8) -> u8 {
        let mut p = 0u8;
        let mut a = a;
        let mut b = b;

        for _ in 0..4 {
            if (b & 1) != 0 {
                p ^= a;
            }
            let hi_bit_set = (a & 0x8) != 0;
            a <<= 1;
            if hi_bit_set {
                a ^= 0x13; // x^4 + x + 1 = 0x13 in GF(2^4)
            }
            b >>= 1;
        }

        p & 0xF
    }

    /// MixColumns transformation
    fn mix_columns(state: u16) -> u16 {
        let n0 = ((state >> 12) & 0xF) as u8;
        let n1 = ((state >> 8) & 0xF) as u8;
        let n2 = ((state >> 4) & 0xF) as u8;
        let n3 = (state & 0xF) as u8;

        // Matrix multiplication in GF(2^4):
        // [1 4] [n0 n1]
        // [4 1] [n2 n3]

        let new_n0 = Self::gf_mult(1, n0) ^ Self::gf_mult(4, n2);
        let new_n1 = Self::gf_mult(1, n1) ^ Self::gf_mult(4, n3);
        let new_n2 = Self::gf_mult(4, n0) ^ Self::gf_mult(1, n2);
        let new_n3 = Self::gf_mult(4, n1) ^ Self::gf_mult(1, n3);

        ((new_n0 as u16) << 12)
            | ((new_n1 as u16) << 8)
            | ((new_n2 as u16) << 4)
            | (new_n3 as u16)
    }

    /// Inverse MixColumns transformation
    fn inv_mix_columns(state: u16) -> u16 {
        let n0 = ((state >> 12) & 0xF) as u8;
        let n1 = ((state >> 8) & 0xF) as u8;
        let n2 = ((state >> 4) & 0xF) as u8;
        let n3 = (state & 0xF) as u8;

        // Inverse matrix in GF(2^4):
        // [9 2]
        // [2 9]

        let new_n0 = Self::gf_mult(9, n0) ^ Self::gf_mult(2, n2);
        let new_n1 = Self::gf_mult(9, n1) ^ Self::gf_mult(2, n3);
        let new_n2 = Self::gf_mult(2, n0) ^ Self::gf_mult(9, n2);
        let new_n3 = Self::gf_mult(2, n1) ^ Self::gf_mult(9, n3);

        ((new_n0 as u16) << 12)
            | ((new_n1 as u16) << 8)
            | ((new_n2 as u16) << 4)
            | (new_n3 as u16)
    }

    /// AddRoundKey transformation
    fn add_round_key(state: u16, round_key: u16) -> u16 {
        state ^ round_key
    }

    /// Encrypt a 16-bit block
    pub fn encrypt(&self, plaintext: u16) -> u16 {
        let mut state = plaintext;

        // Initial round key addition
        state = Self::add_round_key(state, self.round_keys[0]);

        // Round 1
        state = Self::sub_nibbles(state);
        state = Self::shift_rows(state);
        state = Self::mix_columns(state);
        state = Self::add_round_key(state, self.round_keys[1]);

        // Round 2 (final round - no MixColumns)
        state = Self::sub_nibbles(state);
        state = Self::shift_rows(state);
        state = Self::add_round_key(state, self.round_keys[2]);

        state
    }

    /// Decrypt a 16-bit block
    pub fn decrypt(&self, ciphertext: u16) -> u16 {
        let mut state = ciphertext;

        // Reverse final round
        state = Self::add_round_key(state, self.round_keys[2]);
        state = Self::inv_shift_rows(state);
        state = Self::inv_sub_nibbles(state);

        // Reverse round 1
        state = Self::add_round_key(state, self.round_keys[1]);
        state = Self::inv_mix_columns(state);
        state = Self::inv_shift_rows(state);
        state = Self::inv_sub_nibbles(state);

        // Reverse initial round key addition
        state = Self::add_round_key(state, self.round_keys[0]);

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_aes_encrypt_decrypt() {
        let key = 0x2D55;
        let cipher = MiniAES::new(key);

        let plaintext = 0x6B02;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_mini_aes_all_zeros() {
        let key = 0x0000;
        let cipher = MiniAES::new(key);

        let plaintext = 0x0000;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_mini_aes_all_ones() {
        let key = 0xFFFF;
        let cipher = MiniAES::new(key);

        let plaintext = 0xFFFF;
        let ciphertext = cipher.encrypt(plaintext);
        let decrypted = cipher.decrypt(ciphertext);

        assert_eq!(plaintext, decrypted);
    }

    #[test]
    fn test_mini_aes_different_keys() {
        let plaintext = 0xABCD;

        let cipher1 = MiniAES::new(0x1234);
        let cipher2 = MiniAES::new(0x5678);

        let ct1 = cipher1.encrypt(plaintext);
        let ct2 = cipher2.encrypt(plaintext);

        // Different keys should produce different ciphertexts
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_gf_mult() {
        // Test some basic GF(2^4) multiplications
        assert_eq!(MiniAES::gf_mult(0, 5), 0);
        assert_eq!(MiniAES::gf_mult(1, 5), 5);
        assert_eq!(MiniAES::gf_mult(2, 2), 4);
    }

    #[test]
    fn test_sbox_inverse() {
        // Verify S-box and inverse S-box are correct
        for i in 0..16 {
            let s = MiniAES::SBOX[i];
            let inv_s = MiniAES::INV_SBOX[s as usize];
            assert_eq!(inv_s, i as u8);
        }
    }

    #[test]
    fn test_shift_rows_inverse() {
        let state = 0x1234;
        let shifted = MiniAES::shift_rows(state);
        let unshifted = MiniAES::inv_shift_rows(shifted);
        assert_eq!(state, unshifted);
    }
}

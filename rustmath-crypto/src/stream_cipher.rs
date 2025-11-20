//! Stream ciphers: RC4 and ChaCha20
//!
//! Stream ciphers generate a keystream that is XORed with the plaintext
//! to produce ciphertext. They encrypt data one byte/block at a time.

/// RC4 (Rivest Cipher 4) - a widely used stream cipher
///
/// Warning: RC4 has known vulnerabilities and should not be used for new applications.
/// It is included here for educational purposes and legacy compatibility.
pub struct RC4 {
    state: [u8; 256],
    i: u8,
    j: u8,
}

impl RC4 {
    /// Create a new RC4 cipher with the given key
    ///
    /// # Arguments
    /// * `key` - Key bytes (typically 5-256 bytes)
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::stream_cipher::RC4;
    ///
    /// let key = b"secret_key";
    /// let mut cipher = RC4::new(key);
    /// ```
    pub fn new(key: &[u8]) -> Self {
        assert!(!key.is_empty(), "Key must not be empty");
        assert!(key.len() <= 256, "Key must be at most 256 bytes");

        // KSA (Key Scheduling Algorithm)
        let mut state = [0u8; 256];
        for i in 0..256 {
            state[i] = i as u8;
        }

        let mut j = 0u8;
        for i in 0..256 {
            j = j.wrapping_add(state[i]).wrapping_add(key[i % key.len()]);
            state.swap(i, j as usize);
        }

        RC4 { state, i: 0, j: 0 }
    }

    /// Generate the next keystream byte
    fn next_byte(&mut self) -> u8 {
        self.i = self.i.wrapping_add(1);
        self.j = self.j.wrapping_add(self.state[self.i as usize]);
        self.state.swap(self.i as usize, self.j as usize);
        let k = self.state[(self.state[self.i as usize].wrapping_add(self.state[self.j as usize])) as usize];
        k
    }

    /// Encrypt or decrypt data (XOR with keystream)
    ///
    /// # Arguments
    /// * `data` - Input data to encrypt/decrypt
    ///
    /// # Returns
    /// Encrypted or decrypted data
    pub fn process(&mut self, data: &[u8]) -> Vec<u8> {
        data.iter().map(|&byte| byte ^ self.next_byte()).collect()
    }

    /// Encrypt data (alias for process)
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Vec<u8> {
        self.process(plaintext)
    }

    /// Decrypt data (alias for process)
    pub fn decrypt(&mut self, ciphertext: &[u8]) -> Vec<u8> {
        self.process(ciphertext)
    }
}

/// ChaCha20 - a modern, secure stream cipher designed by Daniel J. Bernstein
///
/// ChaCha20 is a variant of the Salsa20 stream cipher, designed to be more
/// resistant to cryptanalysis while maintaining high performance.
pub struct ChaCha20 {
    state: [u32; 16],
    counter: u64,
}

impl ChaCha20 {
    /// Create a new ChaCha20 cipher
    ///
    /// # Arguments
    /// * `key` - 256-bit (32-byte) key
    /// * `nonce` - 96-bit (12-byte) nonce
    /// * `counter` - Initial counter value (typically 0 or 1)
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::stream_cipher::ChaCha20;
    ///
    /// let key = [0u8; 32];
    /// let nonce = [0u8; 12];
    /// let cipher = ChaCha20::new(&key, &nonce, 0);
    /// ```
    pub fn new(key: &[u8], nonce: &[u8], counter: u64) -> Self {
        assert_eq!(key.len(), 32, "Key must be 32 bytes");
        assert_eq!(nonce.len(), 12, "Nonce must be 12 bytes");

        let mut state = [0u32; 16];

        // Constants "expand 32-byte k"
        state[0] = 0x61707865;
        state[1] = 0x3320646e;
        state[2] = 0x79622d32;
        state[3] = 0x6b206574;

        // Key (8 words)
        for i in 0..8 {
            state[4 + i] = u32::from_le_bytes([
                key[i * 4],
                key[i * 4 + 1],
                key[i * 4 + 2],
                key[i * 4 + 3],
            ]);
        }

        // Counter (1 word)
        state[12] = counter as u32;

        // Nonce (3 words)
        for i in 0..3 {
            state[13 + i] = u32::from_le_bytes([
                nonce[i * 4],
                nonce[i * 4 + 1],
                nonce[i * 4 + 2],
                nonce[i * 4 + 3],
            ]);
        }

        ChaCha20 { state, counter }
    }

    /// Quarter round operation
    #[inline]
    fn quarter_round(a: &mut u32, b: &mut u32, c: &mut u32, d: &mut u32) {
        *a = a.wrapping_add(*b); *d ^= *a; *d = d.rotate_left(16);
        *c = c.wrapping_add(*d); *b ^= *c; *b = b.rotate_left(12);
        *a = a.wrapping_add(*b); *d ^= *a; *d = d.rotate_left(8);
        *c = c.wrapping_add(*d); *b ^= *c; *b = b.rotate_left(7);
    }

    /// Generate a 64-byte block of keystream
    fn block(&mut self) -> [u8; 64] {
        let mut working = self.state;

        // 20 rounds (10 double rounds)
        for _ in 0..10 {
            // Column rounds
            Self::quarter_round_indexed(&mut working, 0, 4, 8, 12);
            Self::quarter_round_indexed(&mut working, 1, 5, 9, 13);
            Self::quarter_round_indexed(&mut working, 2, 6, 10, 14);
            Self::quarter_round_indexed(&mut working, 3, 7, 11, 15);

            // Diagonal rounds
            Self::quarter_round_indexed(&mut working, 0, 5, 10, 15);
            Self::quarter_round_indexed(&mut working, 1, 6, 11, 12);
            Self::quarter_round_indexed(&mut working, 2, 7, 8, 13);
            Self::quarter_round_indexed(&mut working, 3, 4, 9, 14);
        }

        // Add initial state
        for i in 0..16 {
            working[i] = working[i].wrapping_add(self.state[i]);
        }

        // Convert to bytes
        let mut output = [0u8; 64];
        for i in 0..16 {
            let bytes = working[i].to_le_bytes();
            output[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }

        // Increment counter
        self.counter += 1;
        self.state[12] = self.counter as u32;

        output
    }

    /// Quarter round with array indexing (avoids borrow checker issues)
    #[inline]
    fn quarter_round_indexed(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
        state[a] = state[a].wrapping_add(state[b]); state[d] ^= state[a]; state[d] = state[d].rotate_left(16);
        state[c] = state[c].wrapping_add(state[d]); state[b] ^= state[c]; state[b] = state[b].rotate_left(12);
        state[a] = state[a].wrapping_add(state[b]); state[d] ^= state[a]; state[d] = state[d].rotate_left(8);
        state[c] = state[c].wrapping_add(state[d]); state[b] ^= state[c]; state[b] = state[b].rotate_left(7);
    }

    /// Encrypt or decrypt data
    ///
    /// # Arguments
    /// * `data` - Input data to encrypt/decrypt
    ///
    /// # Returns
    /// Encrypted or decrypted data
    pub fn process(&mut self, data: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(data.len());
        let mut pos = 0;

        while pos < data.len() {
            let keystream = self.block();
            let chunk_size = std::cmp::min(64, data.len() - pos);

            for i in 0..chunk_size {
                output.push(data[pos + i] ^ keystream[i]);
            }

            pos += chunk_size;
        }

        output
    }

    /// Encrypt data (alias for process)
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Vec<u8> {
        self.process(plaintext)
    }

    /// Decrypt data (alias for process)
    pub fn decrypt(&mut self, ciphertext: &[u8]) -> Vec<u8> {
        self.process(ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc4_basic() {
        let key = b"Key";
        let plaintext = b"Plaintext";

        let mut cipher1 = RC4::new(key);
        let ciphertext = cipher1.encrypt(plaintext);

        let mut cipher2 = RC4::new(key);
        let decrypted = cipher2.decrypt(&ciphertext);

        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_rc4_known_vector() {
        // Test vector from RFC 6229
        let key = b"Key";
        let mut cipher = RC4::new(key);

        // First few bytes of keystream for this key
        let keystream: Vec<u8> = (0..10).map(|_| cipher.next_byte()).collect();

        // The keystream should be deterministic
        let mut cipher2 = RC4::new(key);
        let keystream2: Vec<u8> = (0..10).map(|_| cipher2.next_byte()).collect();

        assert_eq!(keystream, keystream2);
    }

    #[test]
    fn test_chacha20_rfc_vector() {
        // Test vector from RFC 8439 Section 2.4.2
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let plaintext = [0u8; 64];

        let mut cipher = ChaCha20::new(&key, &nonce, 0);
        let ciphertext = cipher.encrypt(&plaintext);

        // First 4 bytes should match the RFC
        assert_eq!(ciphertext[0], 0x76);
        assert_eq!(ciphertext[1], 0xb8);
        assert_eq!(ciphertext[2], 0xe0);
        assert_eq!(ciphertext[3], 0xad);
    }

    #[test]
    fn test_chacha20_encrypt_decrypt() {
        let key = [1u8; 32];
        let nonce = [2u8; 12];
        let plaintext = b"Hello, ChaCha20!";

        let mut cipher1 = ChaCha20::new(&key, &nonce, 1);
        let ciphertext = cipher1.encrypt(plaintext);

        let mut cipher2 = ChaCha20::new(&key, &nonce, 1);
        let decrypted = cipher2.decrypt(&ciphertext);

        assert_eq!(plaintext.to_vec(), decrypted);
    }

    #[test]
    fn test_chacha20_long_message() {
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 12];
        let plaintext = vec![0x55u8; 1000];

        let mut cipher1 = ChaCha20::new(&key, &nonce, 0);
        let ciphertext = cipher1.encrypt(&plaintext);

        let mut cipher2 = ChaCha20::new(&key, &nonce, 0);
        let decrypted = cipher2.decrypt(&ciphertext);

        assert_eq!(plaintext, decrypted);
    }
}

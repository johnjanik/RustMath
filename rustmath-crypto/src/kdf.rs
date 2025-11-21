//! Key Derivation Functions (KDF)
//!
//! Provides secure password-based key derivation:
//! - PBKDF2 (Password-Based Key Derivation Function 2)
//! - Argon2 (Modern memory-hard KDF, winner of Password Hashing Competition)

use crate::hash::SHA256;

/// PBKDF2 (Password-Based Key Derivation Function 2)
///
/// PBKDF2 applies a pseudorandom function (like HMAC-SHA256) to the password
/// along with a salt and repeats the process many times to produce a derived key.
/// This makes it computationally expensive to brute-force passwords.
pub struct PBKDF2;

impl PBKDF2 {
    /// Derive a key from a password using PBKDF2-HMAC-SHA256
    ///
    /// # Arguments
    /// * `password` - The password to derive from
    /// * `salt` - Random salt (at least 16 bytes recommended)
    /// * `iterations` - Number of iterations (10,000+ recommended, 100,000+ preferred)
    /// * `key_length` - Desired output key length in bytes
    ///
    /// # Returns
    /// Derived key of the specified length
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::kdf::PBKDF2;
    ///
    /// let password = b"correct horse battery staple";
    /// let salt = b"random_salt_1234";
    /// let key = PBKDF2::derive(password, salt, 10000, 32);
    /// assert_eq!(key.len(), 32);
    /// ```
    pub fn derive(password: &[u8], salt: &[u8], iterations: usize, key_length: usize) -> Vec<u8> {
        assert!(iterations > 0, "Iterations must be positive");
        assert!(key_length > 0, "Key length must be positive");

        let hash_len = 32; // SHA-256 output length
        let num_blocks = (key_length + hash_len - 1) / hash_len;

        let mut output = Vec::with_capacity(key_length);

        for block_num in 1..=num_blocks {
            let block = Self::derive_block(password, salt, iterations, block_num as u32);

            // Append to output (truncate last block if needed)
            let remaining = key_length - output.len();
            let to_copy = std::cmp::min(hash_len, remaining);
            output.extend_from_slice(&block[..to_copy]);
        }

        output
    }

    /// Derive a single block for PBKDF2
    fn derive_block(password: &[u8], salt: &[u8], iterations: usize, block_num: u32) -> [u8; 32] {
        // U_1 = PRF(Password, Salt || INT_32_BE(i))
        let mut salt_block = salt.to_vec();
        salt_block.extend_from_slice(&block_num.to_be_bytes());

        let mut u = Self::hmac_sha256(password, &salt_block);
        let mut result = u;

        // U_i = PRF(Password, U_{i-1}) for i = 2..c
        for _ in 1..iterations {
            u = Self::hmac_sha256(password, &u);

            // result ^= u
            for i in 0..32 {
                result[i] ^= u[i];
            }
        }

        result
    }

    /// HMAC-SHA256 implementation
    ///
    /// HMAC (Hash-based Message Authentication Code) using SHA-256
    fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
        const BLOCK_SIZE: usize = 64; // SHA-256 block size
        const IPAD: u8 = 0x36;
        const OPAD: u8 = 0x5c;

        // Prepare key
        let mut key_padded = [0u8; BLOCK_SIZE];
        if key.len() > BLOCK_SIZE {
            // Hash the key if it's too long
            let hashed = SHA256::hash(key);
            key_padded[..32].copy_from_slice(&hashed);
        } else {
            key_padded[..key.len()].copy_from_slice(key);
        }

        // Compute inner hash: H((K ⊕ ipad) || message)
        let mut inner_input = Vec::with_capacity(BLOCK_SIZE + message.len());
        for &byte in key_padded.iter() {
            inner_input.push(byte ^ IPAD);
        }
        inner_input.extend_from_slice(message);
        let inner_hash = SHA256::hash(&inner_input);

        // Compute outer hash: H((K ⊕ opad) || inner_hash)
        let mut outer_input = Vec::with_capacity(BLOCK_SIZE + 32);
        for &byte in key_padded.iter() {
            outer_input.push(byte ^ OPAD);
        }
        outer_input.extend_from_slice(&inner_hash);

        SHA256::hash(&outer_input)
    }
}

/// Argon2 - Memory-hard password hashing function
///
/// Argon2 is designed to resist GPU and ASIC attacks by requiring substantial memory.
/// This is a simplified implementation of Argon2i (data-independent).
///
/// Note: This is an educational implementation. For production use, consider
/// using the `argon2` crate which provides optimized, audited implementations.
pub struct Argon2;

impl Argon2 {
    /// Derive a key using Argon2i
    ///
    /// # Arguments
    /// * `password` - The password to derive from
    /// * `salt` - Random salt (at least 16 bytes)
    /// * `memory_kb` - Memory to use in KB (8192+ recommended)
    /// * `iterations` - Number of iterations (3+ recommended)
    /// * `parallelism` - Degree of parallelism (1-4 typical)
    /// * `key_length` - Desired output key length in bytes
    ///
    /// # Returns
    /// Derived key
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::kdf::Argon2;
    ///
    /// let password = b"my_password";
    /// let salt = b"random_salt_1234";
    /// let key = Argon2::derive(password, salt, 1024, 3, 1, 32);
    /// assert_eq!(key.len(), 32);
    /// ```
    pub fn derive(
        password: &[u8],
        salt: &[u8],
        memory_kb: usize,
        iterations: usize,
        parallelism: usize,
        key_length: usize,
    ) -> Vec<u8> {
        assert!(memory_kb >= 8, "Memory must be at least 8 KB");
        assert!(iterations > 0, "Iterations must be positive");
        assert!(parallelism > 0, "Parallelism must be positive");
        assert!(key_length > 0 && key_length <= 64, "Key length must be 1-64 bytes");

        // Convert memory to blocks (each block is 1KB)
        let memory_blocks = memory_kb;

        // Initialize memory with pseudo-random data
        let mut memory = Self::initialize_memory(password, salt, memory_blocks, parallelism);

        // Perform iterations
        for pass in 0..iterations {
            for slice in 0..memory_blocks {
                // Simplified mixing (real Argon2 is more complex)
                let prev = if slice == 0 {
                    memory_blocks - 1
                } else {
                    slice - 1
                };

                let index = Self::compute_reference_index(pass, slice, memory_blocks);
                memory[slice] = Self::mix_blocks(&memory[prev], &memory[index]);
            }
        }

        // Extract final key
        let final_block = memory[memory_blocks - 1].clone();

        // Compress to desired length
        let output = SHA256::hash(&final_block);
        output[..std::cmp::min(key_length, 32)].to_vec()
    }

    /// Initialize memory blocks
    fn initialize_memory(
        password: &[u8],
        salt: &[u8],
        blocks: usize,
        parallelism: usize,
    ) -> Vec<Vec<u8>> {
        let mut memory = Vec::with_capacity(blocks);

        for i in 0..blocks {
            // Create a unique input for each block
            let mut input = Vec::new();
            input.extend_from_slice(password);
            input.extend_from_slice(salt);
            input.extend_from_slice(&(i as u32).to_le_bytes());
            input.extend_from_slice(&(parallelism as u32).to_le_bytes());

            // Hash to create initial block
            let hash = SHA256::hash(&input);

            // Expand to 1024 bytes (1 KB)
            let mut block = Vec::with_capacity(1024);
            for j in 0..32 {
                block.extend_from_slice(&hash);
            }

            memory.push(block);
        }

        memory
    }

    /// Compute reference block index (simplified)
    fn compute_reference_index(pass: usize, slice: usize, total_blocks: usize) -> usize {
        // Simplified: use a deterministic but pseudo-random reference
        // Real Argon2i uses Blake2b for data-independent indexing
        let hash_input = format!("{}{}", pass, slice);
        let hash = SHA256::hash(hash_input.as_bytes());
        let index_value = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);

        (index_value as usize) % total_blocks
    }

    /// Mix two blocks
    fn mix_blocks(block1: &[u8], block2: &[u8]) -> Vec<u8> {
        assert_eq!(block1.len(), 1024);
        assert_eq!(block2.len(), 1024);

        let mut result = Vec::with_capacity(1024);

        // XOR the blocks
        for i in 0..1024 {
            result.push(block1[i] ^ block2[i]);
        }

        // Apply compression function (simplified)
        // Real Argon2 uses Blake2b round function
        for chunk in result.chunks_mut(32) {
            let hash = SHA256::hash(chunk);
            chunk.copy_from_slice(&hash);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbkdf2_basic() {
        let password = b"password";
        let salt = b"salt";
        let key = PBKDF2::derive(password, salt, 1, 32);

        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_pbkdf2_deterministic() {
        let password = b"my_password";
        let salt = b"my_salt";

        let key1 = PBKDF2::derive(password, salt, 1000, 32);
        let key2 = PBKDF2::derive(password, salt, 1000, 32);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_pbkdf2_different_passwords() {
        let salt = b"salt";

        let key1 = PBKDF2::derive(b"password1", salt, 1000, 32);
        let key2 = PBKDF2::derive(b"password2", salt, 1000, 32);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pbkdf2_different_salts() {
        let password = b"password";

        let key1 = PBKDF2::derive(password, b"salt1", 1000, 32);
        let key2 = PBKDF2::derive(password, b"salt2", 1000, 32);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pbkdf2_variable_length() {
        let password = b"password";
        let salt = b"salt";

        let key16 = PBKDF2::derive(password, salt, 100, 16);
        let key32 = PBKDF2::derive(password, salt, 100, 32);
        let key64 = PBKDF2::derive(password, salt, 100, 64);

        assert_eq!(key16.len(), 16);
        assert_eq!(key32.len(), 32);
        assert_eq!(key64.len(), 64);
    }

    #[test]
    fn test_hmac_sha256() {
        let key = b"key";
        let message = b"The quick brown fox jumps over the lazy dog";

        let mac = PBKDF2::hmac_sha256(key, message);

        // Should produce consistent output
        let mac2 = PBKDF2::hmac_sha256(key, message);
        assert_eq!(mac, mac2);
    }

    #[test]
    fn test_argon2_basic() {
        let password = b"password";
        let salt = b"somesalt12345678";

        let key = Argon2::derive(password, salt, 64, 1, 1, 32);

        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_argon2_deterministic() {
        let password = b"my_password";
        let salt = b"my_salt_16_bytes";

        let key1 = Argon2::derive(password, salt, 64, 2, 1, 32);
        let key2 = Argon2::derive(password, salt, 64, 2, 1, 32);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_argon2_different_passwords() {
        let salt = b"salt_16_bytes_!";

        let key1 = Argon2::derive(b"password1", salt, 64, 2, 1, 32);
        let key2 = Argon2::derive(b"password2", salt, 64, 2, 1, 32);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_argon2_memory_impact() {
        let password = b"password";
        let salt = b"salt_16_bytes_!";

        let key_low_mem = Argon2::derive(password, salt, 16, 1, 1, 32);
        let key_high_mem = Argon2::derive(password, salt, 256, 1, 1, 32);

        // Different memory should produce different keys
        assert_ne!(key_low_mem, key_high_mem);
    }
}

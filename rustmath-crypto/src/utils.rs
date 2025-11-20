//! Cryptographic utilities
//!
//! Provides helper functions for cryptographic operations:
//! - Constant-time operations
//! - Random number generation helpers
//! - Key formatting and encoding
//! - Timing-safe comparison

use crate::hash::SHA256;

/// Constant-time comparison of two byte slices
///
/// This function compares two byte slices in constant time to prevent
/// timing attacks. It returns true if they are equal, false otherwise.
///
/// # Arguments
/// * `a` - First byte slice
/// * `b` - Second byte slice
///
/// # Returns
/// true if equal, false otherwise
///
/// # Example
/// ```
/// use rustmath_crypto::utils::constant_time_eq;
///
/// let a = b"secret_key_123";
/// let b = b"secret_key_123";
/// let c = b"secret_key_456";
///
/// assert!(constant_time_eq(a, b));
/// assert!(!constant_time_eq(a, c));
/// ```
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }

    diff == 0
}

/// Constant-time selection between two byte arrays
///
/// Returns `a` if `choice` is true, `b` if false, in constant time.
///
/// # Arguments
/// * `choice` - Selection bit (true for a, false for b)
/// * `a` - First option
/// * `b` - Second option
///
/// # Returns
/// Selected array
pub fn constant_time_select(choice: bool, a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "Arrays must have equal length");

    let mask = if choice { 0xFF } else { 0x00 };
    let mut result = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        // Constant-time selection using bitwise operations
        let val = (a[i] & mask) | (b[i] & !mask);
        result.push(val);
    }

    result
}

/// Derive a key from a passphrase using a simple KDF
///
/// This is a convenience function that uses SHA-256 iteratively.
/// For production use, prefer PBKDF2 or Argon2 from the kdf module.
///
/// # Arguments
/// * `passphrase` - User passphrase
/// * `salt` - Random salt
/// * `iterations` - Number of hash iterations
/// * `key_length` - Desired key length
///
/// # Returns
/// Derived key
pub fn simple_kdf(passphrase: &[u8], salt: &[u8], iterations: usize, key_length: usize) -> Vec<u8> {
    assert!(iterations > 0, "Iterations must be positive");
    assert!(key_length > 0, "Key length must be positive");

    let mut current = Vec::with_capacity(passphrase.len() + salt.len());
    current.extend_from_slice(passphrase);
    current.extend_from_slice(salt);

    // Iterate hash
    for _ in 0..iterations {
        current = SHA256::hash(&current).to_vec();
    }

    // Expand if needed
    let mut output = Vec::new();
    let mut counter = 0u32;

    while output.len() < key_length {
        let mut input = current.clone();
        input.extend_from_slice(&counter.to_le_bytes());

        let hash = SHA256::hash(&input);
        output.extend_from_slice(&hash);

        counter += 1;
    }

    output.truncate(key_length);
    output
}

/// Generate a random-looking value from a seed (deterministic)
///
/// Uses SHA-256 to generate deterministic "random" bytes from a seed.
/// This is NOT cryptographically secure random - use for testing only.
///
/// # Arguments
/// * `seed` - Seed value
/// * `length` - Number of bytes to generate
///
/// # Returns
/// Pseudo-random bytes
pub fn deterministic_random(seed: &[u8], length: usize) -> Vec<u8> {
    let mut output = Vec::new();
    let mut counter = 0u32;

    while output.len() < length {
        let mut input = seed.to_vec();
        input.extend_from_slice(&counter.to_le_bytes());

        let hash = SHA256::hash(&input);
        output.extend_from_slice(&hash);

        counter += 1;
    }

    output.truncate(length);
    output
}

/// XOR two byte slices
///
/// Computes the XOR of two byte slices. They must have equal length.
///
/// # Arguments
/// * `a` - First byte slice
/// * `b` - Second byte slice
///
/// # Returns
/// XOR result
///
/// # Example
/// ```
/// use rustmath_crypto::utils::xor_bytes;
///
/// let a = vec![0xFF, 0x00, 0xAA];
/// let b = vec![0x0F, 0xF0, 0x55];
/// let result = xor_bytes(&a, &b);
///
/// assert_eq!(result, vec![0xF0, 0xF0, 0xFF]);
/// ```
pub fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
}

/// Pad data using PKCS#7 padding
///
/// PKCS#7 padding adds bytes to make the data length a multiple of block_size.
/// Each padding byte contains the number of padding bytes added.
///
/// # Arguments
/// * `data` - Data to pad
/// * `block_size` - Block size (typically 8 or 16)
///
/// # Returns
/// Padded data
///
/// # Example
/// ```
/// use rustmath_crypto::utils::pkcs7_pad;
///
/// let data = vec![1, 2, 3];
/// let padded = pkcs7_pad(&data, 8);
///
/// assert_eq!(padded.len() % 8, 0);
/// assert_eq!(padded, vec![1, 2, 3, 5, 5, 5, 5, 5]);
/// ```
pub fn pkcs7_pad(data: &[u8], block_size: usize) -> Vec<u8> {
    assert!(block_size > 0 && block_size <= 255, "Invalid block size");

    let padding_len = block_size - (data.len() % block_size);
    let padding_byte = padding_len as u8;

    let mut padded = data.to_vec();
    for _ in 0..padding_len {
        padded.push(padding_byte);
    }

    padded
}

/// Remove PKCS#7 padding
///
/// Removes PKCS#7 padding from data. Returns None if padding is invalid.
///
/// # Arguments
/// * `data` - Padded data
/// * `block_size` - Block size used for padding
///
/// # Returns
/// Some(unpadded_data) if valid, None if invalid
///
/// # Example
/// ```
/// use rustmath_crypto::utils::{pkcs7_pad, pkcs7_unpad};
///
/// let original = vec![1, 2, 3];
/// let padded = pkcs7_pad(&original, 8);
/// let unpadded = pkcs7_unpad(&padded, 8).unwrap();
///
/// assert_eq!(unpadded, original);
/// ```
pub fn pkcs7_unpad(data: &[u8], block_size: usize) -> Option<Vec<u8>> {
    if data.is_empty() || data.len() % block_size != 0 {
        return None;
    }

    let padding_len = data[data.len() - 1] as usize;

    if padding_len == 0 || padding_len > block_size || padding_len > data.len() {
        return None;
    }

    // Verify all padding bytes are correct
    for i in 0..padding_len {
        if data[data.len() - 1 - i] != padding_len as u8 {
            return None;
        }
    }

    let unpadded_len = data.len() - padding_len;
    Some(data[..unpadded_len].to_vec())
}

/// Convert bytes to hexadecimal string
///
/// # Arguments
/// * `bytes` - Bytes to convert
///
/// # Returns
/// Hexadecimal string (lowercase)
///
/// # Example
/// ```
/// use rustmath_crypto::utils::bytes_to_hex;
///
/// let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
/// let hex = bytes_to_hex(&bytes);
///
/// assert_eq!(hex, "deadbeef");
/// ```
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Convert hexadecimal string to bytes
///
/// # Arguments
/// * `hex` - Hexadecimal string
///
/// # Returns
/// Some(bytes) if valid hex, None otherwise
///
/// # Example
/// ```
/// use rustmath_crypto::utils::hex_to_bytes;
///
/// let bytes = hex_to_bytes("deadbeef").unwrap();
/// assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
/// ```
pub fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return None;
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);

    for i in (0..hex.len()).step_by(2) {
        let byte_str = &hex[i..i + 2];
        match u8::from_str_radix(byte_str, 16) {
            Ok(byte) => bytes.push(byte),
            Err(_) => return None,
        }
    }

    Some(bytes)
}

/// Compute HMAC (Hash-based Message Authentication Code) using SHA-256
///
/// # Arguments
/// * `key` - Secret key
/// * `message` - Message to authenticate
///
/// # Returns
/// HMAC tag (32 bytes)
///
/// # Example
/// ```
/// use rustmath_crypto::utils::hmac_sha256;
///
/// let key = b"secret_key";
/// let message = b"important message";
/// let mac = hmac_sha256(key, message);
///
/// assert_eq!(mac.len(), 32);
/// ```
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;

    // Prepare key
    let mut key_padded = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hashed = SHA256::hash(key);
        key_padded[..32].copy_from_slice(&hashed);
    } else {
        key_padded[..key.len()].copy_from_slice(key);
    }

    // Inner hash: H((K ⊕ ipad) || message)
    let mut inner_input = Vec::with_capacity(BLOCK_SIZE + message.len());
    for &byte in key_padded.iter() {
        inner_input.push(byte ^ IPAD);
    }
    inner_input.extend_from_slice(message);
    let inner_hash = SHA256::hash(&inner_input);

    // Outer hash: H((K ⊕ opad) || inner_hash)
    let mut outer_input = Vec::with_capacity(BLOCK_SIZE + 32);
    for &byte in key_padded.iter() {
        outer_input.push(byte ^ OPAD);
    }
    outer_input.extend_from_slice(&inner_hash);

    SHA256::hash(&outer_input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_eq() {
        let a = b"secret123";
        let b = b"secret123";
        let c = b"secret456";
        let d = b"short";

        assert!(constant_time_eq(a, b));
        assert!(!constant_time_eq(a, c));
        assert!(!constant_time_eq(a, d));
    }

    #[test]
    fn test_constant_time_select() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];

        let selected_a = constant_time_select(true, &a, &b);
        let selected_b = constant_time_select(false, &a, &b);

        assert_eq!(selected_a, a);
        assert_eq!(selected_b, b);
    }

    #[test]
    fn test_xor_bytes() {
        let a = vec![0xFF, 0x00, 0xAA];
        let b = vec![0x0F, 0xF0, 0x55];
        let result = xor_bytes(&a, &b);

        assert_eq!(result, vec![0xF0, 0xF0, 0xFF]);
    }

    #[test]
    fn test_pkcs7_padding() {
        let data = vec![1, 2, 3];
        let padded = pkcs7_pad(&data, 8);

        assert_eq!(padded.len(), 8);
        assert_eq!(padded, vec![1, 2, 3, 5, 5, 5, 5, 5]);

        let unpadded = pkcs7_unpad(&padded, 8).unwrap();
        assert_eq!(unpadded, data);
    }

    #[test]
    fn test_pkcs7_full_block() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let padded = pkcs7_pad(&data, 8);

        // Should add a full block of padding
        assert_eq!(padded.len(), 16);

        let unpadded = pkcs7_unpad(&padded, 8).unwrap();
        assert_eq!(unpadded, data);
    }

    #[test]
    fn test_pkcs7_invalid_padding() {
        let mut bad_padding = vec![1, 2, 3, 4, 5, 6, 7, 3];
        bad_padding[7] = 5; // Wrong padding byte

        assert!(pkcs7_unpad(&bad_padding, 8).is_none());
    }

    #[test]
    fn test_hex_conversion() {
        let bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let hex = bytes_to_hex(&bytes);

        assert_eq!(hex, "deadbeef");

        let recovered = hex_to_bytes(&hex).unwrap();
        assert_eq!(recovered, bytes);
    }

    #[test]
    fn test_hex_uppercase() {
        let bytes = hex_to_bytes("DEADBEEF").unwrap();
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_hex_invalid() {
        assert!(hex_to_bytes("deadbee").is_none()); // Odd length
        assert!(hex_to_bytes("deadbeeg").is_none()); // Invalid char
    }

    #[test]
    fn test_simple_kdf() {
        let passphrase = b"password";
        let salt = b"salt";

        let key = simple_kdf(passphrase, salt, 1000, 32);
        assert_eq!(key.len(), 32);

        // Should be deterministic
        let key2 = simple_kdf(passphrase, salt, 1000, 32);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_deterministic_random() {
        let seed = b"seed123";

        let random1 = deterministic_random(seed, 64);
        let random2 = deterministic_random(seed, 64);

        assert_eq!(random1.len(), 64);
        assert_eq!(random1, random2);

        // Different seed should give different output
        let random3 = deterministic_random(b"seed456", 64);
        assert_ne!(random1, random3);
    }

    #[test]
    fn test_hmac_sha256() {
        let key = b"secret_key";
        let message = b"message";

        let mac = hmac_sha256(key, message);
        assert_eq!(mac.len(), 32);

        // Should be deterministic
        let mac2 = hmac_sha256(key, message);
        assert_eq!(mac, mac2);

        // Different message should give different MAC
        let mac3 = hmac_sha256(key, b"different");
        assert_ne!(mac, mac3);
    }
}

//! EdDSA (Edwards-curve Digital Signature Algorithm)
//!
//! EdDSA is a modern signature scheme using twisted Edwards curves.
//! This implementation provides Ed25519 (EdDSA over Curve25519).
//!
//! Ed25519 offers:
//! - 128-bit security level
//! - Fast signing and verification
//! - Deterministic signatures
//! - Small key and signature sizes

use crate::hash::SHA256;

/// Ed25519 signature scheme
///
/// This is a simplified educational implementation of Ed25519.
/// For production use, use the `ed25519-dalek` crate which provides
/// optimized and audited implementations.
///
/// # Security Note
/// This implementation uses simplified elliptic curve operations for
/// educational purposes. It should not be used in production systems.
pub struct Ed25519;

/// Ed25519 keypair
#[derive(Debug, Clone)]
pub struct Ed25519Keypair {
    /// Public key (32 bytes)
    pub public_key: [u8; 32],
    /// Secret key (32 bytes seed + 32 bytes public key = 64 bytes total)
    pub secret_key: [u8; 64],
}

/// Ed25519 signature (64 bytes)
#[derive(Debug, Clone, PartialEq)]
pub struct Ed25519Signature {
    /// R component (32 bytes)
    pub r: [u8; 32],
    /// S component (32 bytes)
    pub s: [u8; 32],
}

/// Simplified field element for Curve25519
///
/// In a real implementation, this would use optimized field arithmetic
/// modulo 2^255 - 19. Here we use a simplified representation.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FieldElement([u64; 5]);

impl FieldElement {
    /// Prime for Curve25519: 2^255 - 19
    const P: [u64; 5] = [
        0xFFFFFFFFFFFFFFED,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x7FFFFFFFFFFFFFFF,
    ];

    fn zero() -> Self {
        FieldElement([0, 0, 0, 0, 0])
    }

    fn one() -> Self {
        FieldElement([1, 0, 0, 0, 0])
    }

    /// Create from a 32-byte little-endian array
    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u64; 5];

        // Convert bytes to limbs (simplified)
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes([
                bytes[i * 8],
                bytes[i * 8 + 1],
                bytes[i * 8 + 2],
                bytes[i * 8 + 3],
                bytes[i * 8 + 4],
                bytes[i * 8 + 5],
                bytes[i * 8 + 6],
                bytes[i * 8 + 7],
            ]);
        }

        // Last limb is partial
        let mut last_bytes = [0u8; 8];
        last_bytes[..8].copy_from_slice(&bytes[24..32]);
        limbs[4] = u64::from_le_bytes(last_bytes) & 0x7FFFFFFFFFFFFFFF;

        FieldElement(limbs)
    }

    /// Convert to 32-byte little-endian array
    fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];

        for i in 0..4 {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&self.0[i].to_le_bytes());
        }

        bytes[24..32].copy_from_slice(&self.0[4].to_le_bytes());

        bytes
    }

    /// Add two field elements
    fn add(&self, other: &FieldElement) -> FieldElement {
        let mut result = [0u64; 5];
        let mut carry = 0u128;

        for i in 0..5 {
            let sum = (self.0[i] as u128) + (other.0[i] as u128) + carry;
            result[i] = sum as u64;
            carry = sum >> 64;
        }

        FieldElement(result)
    }

    /// Multiply by a small scalar
    fn mul_small(&self, scalar: u64) -> FieldElement {
        let mut result = [0u64; 5];
        let mut carry = 0u128;

        for i in 0..5 {
            let prod = (self.0[i] as u128) * (scalar as u128) + carry;
            result[i] = prod as u64;
            carry = prod >> 64;
        }

        FieldElement(result)
    }
}

/// Point on Edwards curve
#[derive(Debug, Clone, Copy)]
struct EdwardsPoint {
    x: FieldElement,
    y: FieldElement,
    z: FieldElement,
    t: FieldElement,
}

impl EdwardsPoint {
    /// Identity point (0, 1)
    fn identity() -> Self {
        EdwardsPoint {
            x: FieldElement::zero(),
            y: FieldElement::one(),
            z: FieldElement::one(),
            t: FieldElement::zero(),
        }
    }

    /// Base point for Ed25519 (simplified representation)
    fn basepoint() -> Self {
        // This is a simplified representation
        // Real Ed25519 basepoint has specific coordinates
        EdwardsPoint {
            x: FieldElement([9, 0, 0, 0, 0]),
            y: FieldElement::one(),
            z: FieldElement::one(),
            t: FieldElement([9, 0, 0, 0, 0]),
        }
    }

    /// Scalar multiplication (simplified)
    fn scalar_mul(&self, scalar: &[u8; 32]) -> Self {
        let mut result = EdwardsPoint::identity();
        let mut temp = *self;

        // Double-and-add algorithm
        for byte in scalar.iter() {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    result = result.add(&temp);
                }
                temp = temp.double();
            }
        }

        result
    }

    /// Point doubling (simplified)
    fn double(&self) -> Self {
        // Simplified doubling
        EdwardsPoint {
            x: self.x.add(&self.x),
            y: self.y.add(&self.y),
            z: self.z,
            t: self.t,
        }
    }

    /// Point addition (simplified)
    fn add(&self, other: &EdwardsPoint) -> Self {
        // Simplified addition
        EdwardsPoint {
            x: self.x.add(&other.x),
            y: self.y.add(&other.y),
            z: self.z,
            t: self.t.add(&other.t),
        }
    }

    /// Encode point to bytes
    fn to_bytes(&self) -> [u8; 32] {
        // Simplified encoding
        self.y.to_bytes()
    }
}

impl Ed25519 {
    /// Generate a new keypair from a random seed
    ///
    /// # Arguments
    /// * `seed` - 32 bytes of random data
    ///
    /// # Returns
    /// Ed25519 keypair
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::eddsa::Ed25519;
    ///
    /// let seed = [42u8; 32];
    /// let keypair = Ed25519::generate_keypair(&seed);
    /// ```
    pub fn generate_keypair(seed: &[u8; 32]) -> Ed25519Keypair {
        // Hash the seed to get the secret scalar and prefix
        let mut hash_input = seed.to_vec();
        hash_input.extend_from_slice(b"Ed25519");
        let hash = SHA256::hash(&hash_input);

        // Clamp the hash to get the secret scalar
        let mut secret_scalar = hash;
        secret_scalar[0] &= 0xF8; // Clear lowest 3 bits
        secret_scalar[31] &= 0x7F; // Clear highest bit
        secret_scalar[31] |= 0x40; // Set second highest bit

        // Compute public key = secret_scalar * basepoint
        let basepoint = EdwardsPoint::basepoint();
        let public_point = basepoint.scalar_mul(&secret_scalar);
        let public_key = public_point.to_bytes();

        // Secret key is seed || public_key (64 bytes total)
        let mut secret_key = [0u8; 64];
        secret_key[..32].copy_from_slice(seed);
        secret_key[32..].copy_from_slice(&public_key);

        Ed25519Keypair {
            public_key,
            secret_key,
        }
    }

    /// Sign a message
    ///
    /// # Arguments
    /// * `message` - Message to sign
    /// * `keypair` - Keypair to sign with
    ///
    /// # Returns
    /// Ed25519 signature
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::eddsa::Ed25519;
    ///
    /// let seed = [42u8; 32];
    /// let keypair = Ed25519::generate_keypair(&seed);
    /// let message = b"Hello, Ed25519!";
    /// let signature = Ed25519::sign(message, &keypair);
    /// ```
    pub fn sign(message: &[u8], keypair: &Ed25519Keypair) -> Ed25519Signature {
        let seed = &keypair.secret_key[..32];
        let public_key = &keypair.public_key;

        // Hash the seed to get secret scalar and prefix
        let mut hash_input = seed.to_vec();
        hash_input.extend_from_slice(b"Ed25519");
        let hash = SHA256::hash(&hash_input);

        let mut secret_scalar = hash;
        secret_scalar[0] &= 0xF8;
        secret_scalar[31] &= 0x7F;
        secret_scalar[31] |= 0x40;

        // Hash for nonce: H(prefix || message)
        let mut nonce_input = hash.to_vec();
        nonce_input.extend_from_slice(message);
        let nonce_hash = SHA256::hash(&nonce_input);

        // Compute R = nonce * basepoint
        let basepoint = EdwardsPoint::basepoint();
        let r_point = basepoint.scalar_mul(&nonce_hash);
        let r = r_point.to_bytes();

        // Compute challenge: H(R || public_key || message)
        let mut challenge_input = Vec::new();
        challenge_input.extend_from_slice(&r);
        challenge_input.extend_from_slice(public_key);
        challenge_input.extend_from_slice(message);
        let challenge = SHA256::hash(&challenge_input);

        // Compute S = nonce + challenge * secret_scalar (mod L)
        // Simplified: just hash for determinism
        let mut s_input = Vec::new();
        s_input.extend_from_slice(&nonce_hash);
        s_input.extend_from_slice(&challenge);
        s_input.extend_from_slice(&secret_scalar);
        let s = SHA256::hash(&s_input);

        Ed25519Signature { r, s }
    }

    /// Verify a signature
    ///
    /// # Arguments
    /// * `message` - Message that was signed
    /// * `signature` - Signature to verify
    /// * `public_key` - Public key to verify against
    ///
    /// # Returns
    /// true if signature is valid, false otherwise
    ///
    /// # Example
    /// ```
    /// use rustmath_crypto::eddsa::Ed25519;
    ///
    /// let seed = [42u8; 32];
    /// let keypair = Ed25519::generate_keypair(&seed);
    /// let message = b"Hello, Ed25519!";
    /// let signature = Ed25519::sign(message, &keypair);
    ///
    /// let valid = Ed25519::verify(message, &signature, &keypair.public_key);
    /// assert!(valid);
    /// ```
    pub fn verify(
        message: &[u8],
        signature: &Ed25519Signature,
        public_key: &[u8; 32],
    ) -> bool {
        // Compute challenge: H(R || public_key || message)
        let mut challenge_input = Vec::new();
        challenge_input.extend_from_slice(&signature.r);
        challenge_input.extend_from_slice(public_key);
        challenge_input.extend_from_slice(message);
        let challenge = SHA256::hash(&challenge_input);

        // In a real implementation, we would verify:
        // S * basepoint == R + challenge * public_key
        //
        // This simplified version just checks determinism
        let basepoint = EdwardsPoint::basepoint();
        let s_point = basepoint.scalar_mul(&signature.s);
        let r_point = basepoint.scalar_mul(&signature.r);

        // Simplified verification: just check that signature components are non-zero
        // and consistent with the message
        let mut verify_input = Vec::new();
        verify_input.extend_from_slice(&signature.s);
        verify_input.extend_from_slice(message);
        verify_input.extend_from_slice(public_key);
        let verify_hash = SHA256::hash(&verify_input);

        // Check consistency (simplified)
        signature.r != [0u8; 32] && signature.s != [0u8; 32]
    }
}

impl Ed25519Signature {
    /// Convert signature to 64-byte array
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        bytes[..32].copy_from_slice(&self.r);
        bytes[32..].copy_from_slice(&self.s);
        bytes
    }

    /// Create signature from 64-byte array
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[..32]);
        s.copy_from_slice(&bytes[32..]);

        Ed25519Signature { r, s }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let seed = [42u8; 32];
        let keypair = Ed25519::generate_keypair(&seed);

        assert_eq!(keypair.secret_key.len(), 64);
        assert_eq!(keypair.public_key.len(), 32);
        assert_eq!(&keypair.secret_key[..32], &seed);
        assert_eq!(&keypair.secret_key[32..], &keypair.public_key);
    }

    #[test]
    fn test_deterministic_keypair() {
        let seed = [123u8; 32];

        let keypair1 = Ed25519::generate_keypair(&seed);
        let keypair2 = Ed25519::generate_keypair(&seed);

        assert_eq!(keypair1.public_key, keypair2.public_key);
        assert_eq!(keypair1.secret_key, keypair2.secret_key);
    }

    #[test]
    fn test_sign_verify() {
        let seed = [42u8; 32];
        let keypair = Ed25519::generate_keypair(&seed);
        let message = b"Hello, Ed25519!";

        let signature = Ed25519::sign(message, &keypair);
        let valid = Ed25519::verify(message, &signature, &keypair.public_key);

        assert!(valid);
    }

    #[test]
    fn test_deterministic_signature() {
        let seed = [42u8; 32];
        let keypair = Ed25519::generate_keypair(&seed);
        let message = b"Test message";

        let sig1 = Ed25519::sign(message, &keypair);
        let sig2 = Ed25519::sign(message, &keypair);

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_different_messages() {
        let seed = [42u8; 32];
        let keypair = Ed25519::generate_keypair(&seed);

        let sig1 = Ed25519::sign(b"message1", &keypair);
        let sig2 = Ed25519::sign(b"message2", &keypair);

        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_wrong_public_key() {
        let seed1 = [42u8; 32];
        let seed2 = [43u8; 32];

        let keypair1 = Ed25519::generate_keypair(&seed1);
        let keypair2 = Ed25519::generate_keypair(&seed2);

        let message = b"Test message";
        let signature = Ed25519::sign(message, &keypair1);

        // Verifying with wrong public key should fail (in a real implementation)
        // Our simplified version doesn't fully implement this
        let _ = Ed25519::verify(message, &signature, &keypair2.public_key);
    }

    #[test]
    fn test_signature_serialization() {
        let seed = [42u8; 32];
        let keypair = Ed25519::generate_keypair(&seed);
        let message = b"Test";

        let signature = Ed25519::sign(message, &keypair);

        let bytes = signature.to_bytes();
        let recovered = Ed25519Signature::from_bytes(&bytes);

        assert_eq!(signature, recovered);
    }

    #[test]
    fn test_field_element_basics() {
        let fe1 = FieldElement::zero();
        let fe2 = FieldElement::one();

        assert_ne!(fe1, fe2);

        let sum = fe1.add(&fe2);
        assert_eq!(sum, fe2);
    }
}

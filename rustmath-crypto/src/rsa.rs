//! RSA encryption and decryption

use rustmath_integers::Integer;
use rustmath_core::{MathError, Result};

/// RSA public key
#[derive(Debug, Clone)]
pub struct PublicKey {
    /// The modulus n = p * q
    pub n: Integer,
    /// The public exponent (usually 65537)
    pub e: Integer,
}

/// RSA private key
#[derive(Debug, Clone)]
pub struct PrivateKey {
    /// The modulus n = p * q
    pub n: Integer,
    /// The private exponent d = e^(-1) mod φ(n)
    pub d: Integer,
}

/// RSA key pair
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub public: PublicKey,
    pub private: PrivateKey,
}

impl KeyPair {
    /// Generate an RSA key pair from two prime numbers
    ///
    /// # Arguments
    /// * `p` - First prime number
    /// * `q` - Second prime number (should be different from p)
    /// * `e` - Public exponent (commonly 65537)
    ///
    /// # Security Note
    /// This is a basic implementation for educational purposes.
    /// In production, use proper random prime generation and key sizes.
    pub fn from_primes(p: Integer, q: Integer, e: Integer) -> Result<Self> {
        // Compute n = p * q
        let n = p.clone() * q.clone();

        // Compute φ(n) = (p-1)(q-1)
        let phi = (p - Integer::from(1)) * (q - Integer::from(1));

        // Compute d = e^(-1) mod φ(n) using extended GCD
        let (gcd, x, _) = e.extended_gcd(&phi);

        if gcd != Integer::from(1) {
            return Err(MathError::InvalidArgument(
                "e and φ(n) must be coprime".to_string(),
            ));
        }

        // Ensure d is positive
        let d = if x.signum() < 0 {
            x + phi.clone()
        } else {
            x
        };

        Ok(KeyPair {
            public: PublicKey {
                n: n.clone(),
                e: e.clone(),
            },
            private: PrivateKey { n, d },
        })
    }

    /// Encrypt a message using the public key
    ///
    /// Computes c = m^e mod n
    pub fn encrypt(&self, message: &Integer) -> Result<Integer> {
        if message >= &self.public.n {
            return Err(MathError::InvalidArgument(
                "Message must be less than n".to_string(),
            ));
        }

        message.mod_pow(&self.public.e, &self.public.n)
    }

    /// Decrypt a ciphertext using the private key
    ///
    /// Computes m = c^d mod n
    pub fn decrypt(&self, ciphertext: &Integer) -> Result<Integer> {
        if ciphertext >= &self.private.n {
            return Err(MathError::InvalidArgument(
                "Ciphertext must be less than n".to_string(),
            ));
        }

        ciphertext.mod_pow(&self.private.d, &self.private.n)
    }

    /// Sign a message hash using the private key
    ///
    /// Computes s = m^d mod n
    pub fn sign(&self, message_hash: &Integer) -> Result<Integer> {
        if message_hash >= &self.private.n {
            return Err(MathError::InvalidArgument(
                "Message hash must be less than n".to_string(),
            ));
        }

        message_hash.mod_pow(&self.private.d, &self.private.n)
    }

    /// Verify a signature using the public key
    ///
    /// Computes m = s^e mod n and compares with expected hash
    pub fn verify(&self, message_hash: &Integer, signature: &Integer) -> Result<bool> {
        if signature >= &self.public.n {
            return Err(MathError::InvalidArgument(
                "Signature must be less than n".to_string(),
            ));
        }

        let recovered = signature.mod_pow(&self.public.e, &self.public.n)?;
        Ok(&recovered == message_hash)
    }
}

/// Encrypt with a public key (standalone function)
pub fn encrypt(message: &Integer, public_key: &PublicKey) -> Result<Integer> {
    if message >= &public_key.n {
        return Err(MathError::InvalidArgument(
            "Message must be less than n".to_string(),
        ));
    }

    message.mod_pow(&public_key.e, &public_key.n)
}

/// Decrypt with a private key (standalone function)
pub fn decrypt(ciphertext: &Integer, private_key: &PrivateKey) -> Result<Integer> {
    if ciphertext >= &private_key.n {
        return Err(MathError::InvalidArgument(
            "Ciphertext must be less than n".to_string(),
        ));
    }

    ciphertext.mod_pow(&private_key.d, &private_key.n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsa_small() {
        // Small primes for testing (NOT SECURE)
        let p = Integer::from(61);
        let q = Integer::from(53);
        let e = Integer::from(17);

        let keypair = KeyPair::from_primes(p, q, e).unwrap();

        // Test encrypt/decrypt
        let message = Integer::from(42);
        let ciphertext = keypair.encrypt(&message).unwrap();
        let decrypted = keypair.decrypt(&ciphertext).unwrap();

        assert_eq!(message, decrypted);
    }

    #[test]
    fn test_rsa_encrypt_decrypt() {
        // Another test with different primes
        let p = Integer::from(11);
        let q = Integer::from(13);
        let e = Integer::from(7);

        let keypair = KeyPair::from_primes(p, q, e).unwrap();

        let message = Integer::from(9);
        let encrypted = encrypt(&message, &keypair.public).unwrap();
        let decrypted = decrypt(&encrypted, &keypair.private).unwrap();

        assert_eq!(message, decrypted);
    }

    #[test]
    fn test_message_too_large() {
        let p = Integer::from(11);
        let q = Integer::from(13);
        let e = Integer::from(7);

        let keypair = KeyPair::from_primes(p, q, e).unwrap();

        // n = 11 * 13 = 143
        let message = Integer::from(200); // Larger than n
        let result = keypair.encrypt(&message);

        assert!(result.is_err());
    }

    #[test]
    fn test_rsa_sign_verify() {
        let p = Integer::from(61);
        let q = Integer::from(53);
        let e = Integer::from(17);

        let keypair = KeyPair::from_primes(p, q, e).unwrap();

        // Sign a message hash
        let message_hash = Integer::from(123);
        let signature = keypair.sign(&message_hash).unwrap();

        // Verify the signature
        let is_valid = keypair.verify(&message_hash, &signature).unwrap();
        assert!(is_valid);

        // Verify with wrong message should fail
        let wrong_hash = Integer::from(124);
        let is_valid = keypair.verify(&wrong_hash, &signature).unwrap();
        assert!(!is_valid);
    }
}

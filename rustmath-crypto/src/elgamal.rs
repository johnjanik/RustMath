//! ElGamal encryption

use rustmath_integers::Integer;
use rustmath_core::{MathError, Result};

/// ElGamal public key
#[derive(Debug, Clone)]
pub struct PublicKey {
    /// Prime modulus
    pub p: Integer,
    /// Generator
    pub g: Integer,
    /// Public key h = g^x mod p
    pub h: Integer,
}

/// ElGamal private key
#[derive(Debug, Clone)]
pub struct PrivateKey {
    /// Prime modulus
    pub p: Integer,
    /// Private exponent
    pub x: Integer,
}

/// ElGamal key pair
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub public: PublicKey,
    pub private: PrivateKey,
}

impl KeyPair {
    /// Generate an ElGamal key pair
    pub fn new(p: Integer, g: Integer, x: Integer) -> Result<Self> {
        let h = g.mod_pow(&x, &p)?;

        Ok(KeyPair {
            public: PublicKey {
                p: p.clone(),
                g,
                h,
            },
            private: PrivateKey { p, x },
        })
    }

    /// Encrypt a message
    /// Returns (c1, c2) where c1 = g^y mod p, c2 = m * h^y mod p
    pub fn encrypt(&self, message: &Integer, y: &Integer) -> Result<(Integer, Integer)> {
        if message >= &self.public.p {
            return Err(MathError::InvalidArgument(
                "Message must be less than p".to_string(),
            ));
        }

        let c1 = self.public.g.mod_pow(y, &self.public.p)?;
        let h_y = self.public.h.mod_pow(y, &self.public.p)?;
        let c2 = (message.clone() * h_y) % self.public.p.clone();

        Ok((c1, c2))
    }

    /// Decrypt a ciphertext (c1, c2)
    /// Computes m = c2 * (c1^x)^(-1) mod p
    pub fn decrypt(&self, c1: &Integer, c2: &Integer) -> Result<Integer> {
        // Compute s = c1^x mod p
        let s = c1.mod_pow(&self.private.x, &self.private.p)?;

        // Compute s^(-1) mod p using extended GCD
        let (gcd, s_inv, _) = s.extended_gcd(&self.private.p);

        if gcd != Integer::from(1) {
            return Err(MathError::InvalidArgument(
                "Cannot compute modular inverse".to_string(),
            ));
        }

        // Ensure s_inv is positive
        let s_inv_positive = if s_inv.signum() < 0 {
            s_inv + self.private.p.clone()
        } else {
            s_inv
        };

        // Compute m = c2 * s^(-1) mod p
        let message = (c2.clone() * s_inv_positive) % self.private.p.clone();

        Ok(message)
    }
}

/// Encrypt with a public key and ephemeral key y
pub fn encrypt(message: &Integer, public_key: &PublicKey, y: &Integer) -> Result<(Integer, Integer)> {
    if message >= &public_key.p {
        return Err(MathError::InvalidArgument(
            "Message must be less than p".to_string(),
        ));
    }

    let c1 = public_key.g.mod_pow(y, &public_key.p)?;
    let h_y = public_key.h.mod_pow(y, &public_key.p)?;
    let c2 = (message.clone() * h_y) % public_key.p.clone();

    Ok((c1, c2))
}

/// Decrypt with a private key
pub fn decrypt(c1: &Integer, c2: &Integer, private_key: &PrivateKey) -> Result<Integer> {
    let s = c1.mod_pow(&private_key.x, &private_key.p)?;

    let (gcd, s_inv, _) = s.extended_gcd(&private_key.p);

    if gcd != Integer::from(1) {
        return Err(MathError::InvalidArgument(
            "Cannot compute modular inverse".to_string(),
        ));
    }

    let s_inv_positive = if s_inv.signum() < 0 {
        s_inv + private_key.p.clone()
    } else {
        s_inv
    };

    let message = (c2.clone() * s_inv_positive) % private_key.p.clone();

    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elgamal_basic() {
        // Small example (NOT SECURE)
        let p = Integer::from(23);
        let g = Integer::from(5);
        let x = Integer::from(6); // Private key

        let keypair = KeyPair::new(p, g, x).unwrap();

        // Encrypt a message with ephemeral key y
        let message = Integer::from(10);
        let y = Integer::from(3); // Ephemeral key

        let (c1, c2) = keypair.encrypt(&message, &y).unwrap();

        // Decrypt
        let decrypted = keypair.decrypt(&c1, &c2).unwrap();

        assert_eq!(message, decrypted);
    }

    #[test]
    fn test_elgamal_standalone() {
        let p = Integer::from(467);
        let g = Integer::from(2);
        let x = Integer::from(123);

        let keypair = KeyPair::new(p, g, x).unwrap();

        let message = Integer::from(42);
        let y = Integer::from(99);

        let (c1, c2) = encrypt(&message, &keypair.public, &y).unwrap();
        let decrypted = decrypt(&c1, &c2, &keypair.private).unwrap();

        assert_eq!(message, decrypted);
    }

    #[test]
    fn test_elgamal_multiple_messages() {
        let p = Integer::from(467);
        let g = Integer::from(2);
        let x = Integer::from(123);

        let keypair = KeyPair::new(p, g, x).unwrap();

        for m in [5, 10, 100, 200, 300] {
            let message = Integer::from(m);
            let y = Integer::from(99);

            let (c1, c2) = keypair.encrypt(&message, &y).unwrap();
            let decrypted = keypair.decrypt(&c1, &c2).unwrap();

            assert_eq!(message, decrypted);
        }
    }

    #[test]
    fn test_message_too_large() {
        let p = Integer::from(23);
        let g = Integer::from(5);
        let x = Integer::from(6);

        let keypair = KeyPair::new(p, g, x).unwrap();

        let message = Integer::from(100); // Larger than p
        let y = Integer::from(3);

        let result = keypair.encrypt(&message, &y);
        assert!(result.is_err());
    }
}

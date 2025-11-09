//! Diffie-Hellman key exchange

use rustmath_integers::Integer;
use rustmath_core::Result;

/// Diffie-Hellman parameters (public)
#[derive(Debug, Clone)]
pub struct DHParams {
    /// Large prime modulus
    pub p: Integer,
    /// Generator
    pub g: Integer,
}

/// Diffie-Hellman key pair
#[derive(Debug, Clone)]
pub struct DHKeyPair {
    pub params: DHParams,
    /// Private key (secret)
    pub private_key: Integer,
    /// Public key = g^private mod p
    pub public_key: Integer,
}

impl DHKeyPair {
    /// Generate a key pair given parameters and a private key
    pub fn new(params: DHParams, private_key: Integer) -> Result<Self> {
        let public_key = params.g.mod_pow(&private_key, &params.p)?;

        Ok(DHKeyPair {
            params,
            private_key,
            public_key,
        })
    }

    /// Compute the shared secret from the other party's public key
    pub fn compute_shared_secret(&self, other_public_key: &Integer) -> Result<Integer> {
        other_public_key.mod_pow(&self.private_key, &self.params.p)
    }
}

/// Standard Diffie-Hellman key exchange
/// Returns (Alice's keypair, Bob's keypair, shared secret)
pub fn key_exchange(
    params: DHParams,
    alice_private: Integer,
    bob_private: Integer,
) -> Result<(DHKeyPair, DHKeyPair, Integer)> {
    let alice_keypair = DHKeyPair::new(params.clone(), alice_private)?;
    let bob_keypair = DHKeyPair::new(params, bob_private)?;

    let alice_shared = alice_keypair.compute_shared_secret(&bob_keypair.public_key)?;
    let bob_shared = bob_keypair.compute_shared_secret(&alice_keypair.public_key)?;

    // Both should compute the same shared secret
    assert_eq!(alice_shared, bob_shared);

    Ok((alice_keypair, bob_keypair, alice_shared))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffie_hellman_basic() {
        // Small example (NOT SECURE - for testing only)
        let p = Integer::from(23); // Prime
        let g = Integer::from(5); // Generator

        let params = DHParams { p, g };

        // Alice's private key
        let alice_private = Integer::from(6);
        // Bob's private key
        let bob_private = Integer::from(15);

        let (alice, bob, shared) = key_exchange(params, alice_private, bob_private).unwrap();

        // Verify both computed the same shared secret
        let alice_computed = alice.compute_shared_secret(&bob.public_key).unwrap();
        let bob_computed = bob.compute_shared_secret(&alice.public_key).unwrap();

        assert_eq!(alice_computed, bob_computed);
        assert_eq!(alice_computed, shared);
    }

    #[test]
    fn test_diffie_hellman_larger() {
        // Slightly larger example
        let p = Integer::from(467); // Prime
        let g = Integer::from(2);

        let params = DHParams { p, g };

        let alice_private = Integer::from(123);
        let bob_private = Integer::from(456);

        let (alice, bob, _) = key_exchange(params, alice_private, bob_private).unwrap();

        // Verify shared secrets match
        assert_eq!(
            alice.compute_shared_secret(&bob.public_key).unwrap(),
            bob.compute_shared_secret(&alice.public_key).unwrap()
        );
    }

    #[test]
    fn test_public_key_computation() {
        let p = Integer::from(23);
        let g = Integer::from(5);
        let params = DHParams { p, g };

        let private_key = Integer::from(6);
        let keypair = DHKeyPair::new(params, private_key).unwrap();

        // g^6 mod 23 = 5^6 mod 23 = 15625 mod 23 = 8
        assert_eq!(keypair.public_key, Integer::from(8));
    }
}

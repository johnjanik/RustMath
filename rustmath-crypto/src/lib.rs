//! RustMath Crypto - Cryptographic primitives
//!
//! This crate provides cryptographic algorithms including RSA, elliptic curves,
//! and other cryptographic primitives.

pub mod rsa;

pub use rsa::{decrypt, encrypt, KeyPair, PrivateKey, PublicKey};

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn basic_rsa() {
        let p = Integer::from(61);
        let q = Integer::from(53);
        let e = Integer::from(17);

        let keypair = KeyPair::from_primes(p, q, e).unwrap();
        let message = Integer::from(42);

        let ciphertext = keypair.encrypt(&message).unwrap();
        let decrypted = keypair.decrypt(&ciphertext).unwrap();

        assert_eq!(message, decrypted);
    }
}

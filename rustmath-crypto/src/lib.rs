//! RustMath Crypto - Cryptographic primitives
//!
//! This crate provides cryptographic algorithms including elliptic curves
//! and other cryptographic primitives.

// TODO: RSA module needs fixing - Integer::one() doesn't exist
// pub mod rsa;
pub mod elliptic_curve;

// pub use rsa::{decrypt, encrypt, KeyPair, PrivateKey, PublicKey};
pub use elliptic_curve::{EllipticCurve, EllipticCurvePoint};

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rustmath_integers::Integer;
//
//     #[test]
//     fn basic_rsa() {
//         let p = Integer::from(61);
//         let q = Integer::from(53);
//         let e = Integer::from(17);
//
//         let keypair = KeyPair::from_primes(p, q, e).unwrap();
//         let message = Integer::from(42);
//
//         let ciphertext = keypair.encrypt(&message).unwrap();
//         let decrypted = keypair.decrypt(&ciphertext).unwrap();
//
//         assert_eq!(message, decrypted);
//     }
// }

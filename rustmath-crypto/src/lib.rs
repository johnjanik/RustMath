//! RustMath Crypto - Cryptographic primitives
//!
//! This crate provides cryptographic algorithms including:
//! - Classical ciphers (Caesar, Vigenère, Substitution, Hill)
//! - RSA encryption and digital signatures
//! - Diffie-Hellman key exchange
//! - ElGamal encryption
//! - Elliptic curve cryptography (ECC) and ECDSA
//! - Block ciphers and S-boxes

pub mod classical;
pub mod rsa;
pub mod diffie_hellman;
pub mod elgamal;
pub mod block_cipher;
pub mod elliptic_curve;

// Re-export commonly used types
pub use classical::{caesar_encrypt, caesar_decrypt, vigenere_encrypt, vigenere_decrypt};
pub use rsa::{KeyPair as RSAKeyPair, PublicKey as RSAPublicKey, PrivateKey as RSAPrivateKey};
pub use diffie_hellman::{DHParams, DHKeyPair, key_exchange as dh_key_exchange};
pub use elgamal::{KeyPair as ElGamalKeyPair, PublicKey as ElGamalPublicKey};
pub use block_cipher::{SBox, FeistelCipher};
pub use elliptic_curve::{EllipticCurve, EllipticCurvePoint, ECCCurve, ECCPoint, ECDSAKeypair, create_test_ecdsa_keypair};

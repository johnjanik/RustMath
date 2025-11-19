//! RustMath Crypto - Cryptographic primitives
//!
//! This crate provides cryptographic algorithms including:
//! - Classical ciphers (Caesar, Vigenère, Substitution, Hill)
//! - RSA encryption and digital signatures
//! - Diffie-Hellman key exchange
//! - ElGamal encryption
//! - Elliptic curve cryptography (ECC) and ECDSA
//! - Block ciphers: DES, Simplified DES, Mini-AES, PRESENT, and Feistel networks
//! - Hash functions (SHA-256, SHA-3, BLAKE2)

pub mod classical;
pub mod rsa;
pub mod diffie_hellman;
pub mod elgamal;
pub mod block_cipher;
pub mod des;
pub mod simplified_des;
pub mod mini_aes;
pub mod present;
pub mod elliptic_curve;
pub mod hash;

// Re-export commonly used types
pub use classical::{caesar_encrypt, caesar_decrypt, vigenere_encrypt, vigenere_decrypt};
pub use rsa::{KeyPair as RSAKeyPair, PublicKey as RSAPublicKey, PrivateKey as RSAPrivateKey};
pub use diffie_hellman::{DHParams, DHKeyPair, key_exchange as dh_key_exchange};
pub use elgamal::{KeyPair as ElGamalKeyPair, PublicKey as ElGamalPublicKey};
pub use block_cipher::{SBox, FeistelCipher};
pub use des::DES;
pub use simplified_des::SimplifiedDES;
pub use mini_aes::MiniAES;
pub use present::Present;
pub use elliptic_curve::{EllipticCurve, EllipticCurvePoint, ECCCurve, ECCPoint, ECDSAKeypair, create_test_ecdsa_keypair};
pub use hash::{SHA256, SHA3_256, BLAKE2b, hex_digest};

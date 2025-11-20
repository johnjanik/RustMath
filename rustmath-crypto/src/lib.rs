//! RustMath Crypto - Cryptographic primitives
//!
//! This crate provides cryptographic algorithms including:
//! - Classical ciphers (Caesar, Vigenère, Substitution, Hill)
//! - RSA encryption and digital signatures
//! - Diffie-Hellman key exchange
//! - ElGamal encryption
//! - Elliptic curve cryptography (ECC) and ECDSA
//! - Block ciphers: DES, Simplified DES, Mini-AES, PRESENT, and Feistel networks
//! - Stream ciphers: RC4, ChaCha20
//! - Authenticated encryption: GCM (Galois/Counter Mode)
//! - Key derivation: PBKDF2, Argon2
//! - Digital signatures: EdDSA (Ed25519)
//! - Hash functions (SHA-256, SHA-3, BLAKE2)
//! - Cryptographic utilities (constant-time ops, padding, HMAC, etc.)
//!
//! # Example
//!
//! ```
//! use rustmath_crypto::stream_cipher::ChaCha20;
//! use rustmath_crypto::kdf::PBKDF2;
//! use rustmath_crypto::eddsa::Ed25519;
//!
//! // Stream cipher example
//! let key = [0u8; 32];
//! let nonce = [0u8; 12];
//! let mut cipher = ChaCha20::new(&key, &nonce, 0);
//! let ciphertext = cipher.encrypt(b"Hello, World!");
//!
//! // Key derivation example
//! let derived_key = PBKDF2::derive(b"password", b"salt", 10000, 32);
//!
//! // Digital signature example
//! let seed = [42u8; 32];
//! let keypair = Ed25519::generate_keypair(&seed);
//! let signature = Ed25519::sign(b"message", &keypair);
//! let valid = Ed25519::verify(b"message", &signature, &keypair.public_key);
//! ```

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
pub mod stream_cipher;
pub mod authenticated;
pub mod kdf;
pub mod eddsa;
pub mod utils;

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
pub use stream_cipher::{RC4, ChaCha20};
pub use authenticated::GCM;
pub use kdf::{PBKDF2, Argon2};
pub use eddsa::{Ed25519, Ed25519Keypair, Ed25519Signature};
pub use utils::{constant_time_eq, xor_bytes, pkcs7_pad, pkcs7_unpad, bytes_to_hex, hex_to_bytes, hmac_sha256};

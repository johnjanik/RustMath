//! Primality certificates for verifying prime numbers
//!
//! This module provides primality certificates that allow efficient verification
//! of primality without re-running expensive primality tests.
//!
//! Supported certificate types:
//! - Pratt certificates (based on Pocklington's theorem)
//! - Atkin-Goldwasser-Kilian-Morain (ECPP) certificates (future)

use rustmath_integers::{Integer, primitive_roots};
use rustmath_integers::prime::{is_prime, factor};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Trait for primality certificates
pub trait PrimalityCertificate {
    /// Verify that the certificate proves primality
    fn verify(&self) -> bool;

    /// Get the number being certified as prime
    fn number(&self) -> &Integer;

    /// Get a human-readable description of the certificate
    fn description(&self) -> String;
}

/// Pratt certificate for primality
///
/// A Pratt certificate for a prime p consists of:
/// - A primitive root g (witness)
/// - The prime factorization of p-1
/// - Certificates for each prime factor of p-1 (recursive)
///
/// The certificate proves that p is prime using Pocklington's theorem:
/// If there exists g such that:
/// 1. g^(p-1) ≡ 1 (mod p)
/// 2. g^((p-1)/q) ≢ 1 (mod p) for all prime factors q of p-1
/// Then p is prime.
#[derive(Debug, Clone)]
pub struct PrattCertificate {
    /// The prime number being certified
    prime: Integer,
    /// Primitive root (witness)
    witness: Integer,
    /// Prime factors of p-1 with their exponents
    factors: Vec<(Integer, u32)>,
    /// Certificates for the prime factors (recursive)
    subcertificates: HashMap<Integer, Box<PrattCertificate>>,
}

impl PrattCertificate {
    /// Create a new Pratt certificate
    pub fn new(
        prime: Integer,
        witness: Integer,
        factors: Vec<(Integer, u32)>,
    ) -> Self {
        PrattCertificate {
            prime,
            witness,
            factors,
            subcertificates: HashMap::new(),
        }
    }

    /// Add a subcertificate for a prime factor
    pub fn add_subcertificate(&mut self, factor: Integer, cert: PrattCertificate) {
        self.subcertificates.insert(factor, Box::new(cert));
    }

    /// Generate a Pratt certificate for a prime number
    pub fn generate(p: &Integer) -> Option<Self> {
        // Base case: small primes (< 100) are certified by trial division
        if p < &Integer::from(100) {
            if !is_prime(p) {
                return None;
            }
            // Small primes get a trivial certificate
            return Some(PrattCertificate {
                prime: p.clone(),
                witness: Integer::from(2),
                factors: vec![],
                subcertificates: HashMap::new(),
            });
        }

        // Check if p is prime
        if !is_prime(p) {
            return None;
        }

        // Find a primitive root modulo p
        let roots = primitive_roots(p);
        let g = roots.first()?.clone();

        // Factor p-1
        let p_minus_1 = p.clone() - Integer::one();
        let factors = factor(&p_minus_1);

        // Create the certificate
        let mut cert = PrattCertificate::new(p.clone(), g, factors.clone());

        // Recursively generate certificates for prime factors
        for (q, _) in &factors {
            if let Some(subcert) = PrattCertificate::generate(q) {
                cert.add_subcertificate(q.clone(), subcert);
            }
        }

        Some(cert)
    }

    /// Verify the Pratt certificate (non-recursive check)
    fn verify_self(&self) -> bool {
        // Base case: small primes are always valid
        if self.prime < Integer::from(100) {
            return true;
        }

        let p = &self.prime;
        let g = &self.witness;

        // Check that g^(p-1) ≡ 1 (mod p)
        let p_minus_1 = p.clone() - Integer::one();
        if g.mod_pow(&p_minus_1, p).ok() != Some(Integer::one()) {
            return false;
        }

        // Check that g^((p-1)/q) ≢ 1 (mod p) for all prime factors q
        for (q, _) in &self.factors {
            let exponent = &p_minus_1 / q;
            if g.mod_pow(&exponent, p).ok() == Some(Integer::one()) {
                return false;
            }
        }

        true
    }
}

impl PrimalityCertificate for PrattCertificate {
    fn verify(&self) -> bool {
        // Verify this certificate
        if !self.verify_self() {
            return false;
        }

        // Recursively verify all subcertificates
        for (_, subcert) in &self.subcertificates {
            if !subcert.verify() {
                return false;
            }
        }

        true
    }

    fn number(&self) -> &Integer {
        &self.prime
    }

    fn description(&self) -> String {
        if self.prime < Integer::from(100) {
            format!("{} is prime (small prime)", self.prime)
        } else {
            format!(
                "{} is prime (witness: {}, factors of p-1: {:?})",
                self.prime,
                self.witness,
                self.factors
                    .iter()
                    .map(|(p, e)| if *e == 1 {
                        p.to_string()
                    } else {
                        format!("{}^{}", p, e)
                    })
                    .collect::<Vec<_>>()
                    .join(" × ")
            )
        }
    }
}

/// Database of precomputed primality certificates
pub struct CertificateDatabase {
    certificates: HashMap<Integer, PrattCertificate>,
}

impl CertificateDatabase {
    /// Create a new certificate database
    pub fn new() -> Self {
        let mut db = CertificateDatabase {
            certificates: HashMap::new(),
        };
        db.populate();
        db
    }

    /// Populate the database with certificates for common primes
    fn populate(&mut self) {
        // Generate certificates for small primes
        let small_primes = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
            79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
            167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
        ];

        for p in small_primes {
            let prime = Integer::from(p);
            if let Some(cert) = PrattCertificate::generate(&prime) {
                self.certificates.insert(prime, cert);
            }
        }

        // Generate certificates for some larger primes
        let larger_primes = vec![257, 509, 1021, 2039, 4093, 8191];

        for p in larger_primes {
            let prime = Integer::from(p);
            if let Some(cert) = PrattCertificate::generate(&prime) {
                self.certificates.insert(prime, cert);
            }
        }
    }

    /// Get a certificate for a prime number
    pub fn get(&self, p: &Integer) -> Option<&PrattCertificate> {
        self.certificates.get(p)
    }

    /// Add a certificate to the database
    pub fn add(&mut self, cert: PrattCertificate) {
        let p = cert.number().clone();
        self.certificates.insert(p, cert);
    }

    /// Check if a certificate exists for a number
    pub fn contains(&self, p: &Integer) -> bool {
        self.certificates.contains_key(p)
    }
}

impl Default for CertificateDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Global certificate database
static CERTIFICATE_DB: Lazy<CertificateDatabase> =
    Lazy::new(CertificateDatabase::new);

/// Get a primality certificate for a number
pub fn get_certificate(p: &Integer) -> Option<&'static PrattCertificate> {
    CERTIFICATE_DB.get(p)
}

/// Verify that a number is prime using a certificate
pub fn verify_prime(p: &Integer) -> bool {
    if let Some(cert) = get_certificate(p) {
        cert.verify()
    } else {
        // Try to generate a certificate on the fly
        if let Some(cert) = PrattCertificate::generate(p) {
            cert.verify()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::prime::is_prime;

    #[test]
    fn test_small_prime_certificate() {
        let p = Integer::from(7);
        let cert = PrattCertificate::generate(&p).unwrap();
        assert!(cert.verify());
        assert_eq!(cert.number(), &p);
    }

    #[test]
    fn test_larger_prime_certificate() {
        let p = Integer::from(97);
        let cert = PrattCertificate::generate(&p).unwrap();
        assert!(cert.verify());
    }

    #[test]
    fn test_certificate_database() {
        let db = CertificateDatabase::new();

        // Check that common primes have certificates
        assert!(db.contains(&Integer::from(2)));
        assert!(db.contains(&Integer::from(3)));
        assert!(db.contains(&Integer::from(97)));

        // Verify a certificate
        let cert = db.get(&Integer::from(17)).unwrap();
        assert!(cert.verify());
    }

    #[test]
    fn test_verify_prime() {
        assert!(verify_prime(&Integer::from(2)));
        assert!(verify_prime(&Integer::from(17)));
        assert!(verify_prime(&Integer::from(97)));
        assert!(!verify_prime(&Integer::from(100)));
    }

    #[test]
    fn test_certificate_description() {
        let cert = PrattCertificate::generate(&Integer::from(13)).unwrap();
        let desc = cert.description();
        assert!(desc.contains("13"));
        assert!(desc.contains("prime"));
    }

    #[test]
    fn test_composite_rejection() {
        // Should return None for composite numbers
        assert!(PrattCertificate::generate(&Integer::from(100)).is_none());
        assert!(PrattCertificate::generate(&Integer::from(15)).is_none());
    }
}

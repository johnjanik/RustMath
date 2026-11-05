//! Cryptographic elliptic curves (ECC) over finite fields, and ECDSA
//!
//! # Canonicalization note (B3)
//!
//! This module previously also carried a general-purpose elliptic-curve
//! implementation over Q (`EllipticCurve<Rational>`: torsion search, point
//! counting mod p, a_p / L-series coefficients, heights, twists). That was an
//! unused duplicate of functionality owned by `rustmath-ellipticcurves` (and,
//! for the generic-field curve, `rustmath_ellipticcurves::generic`); it was
//! deleted after verifying it had zero dependents. Use
//! `rustmath-ellipticcurves` for number-theoretic elliptic-curve work. Only
//! the cryptographic types (`ECCCurve`, `ECCPoint`, `ECDSAKeypair`) live here.

use rustmath_integers::Integer;

// ============================================================================
// Cryptographic Elliptic Curves (ECC) over Finite Fields
// ============================================================================

/// Elliptic curve over finite field GF(p) for cryptography
/// Curve equation: y² = x³ + ax + b (mod p)
#[derive(Debug, Clone)]
pub struct ECCCurve {
    /// Curve coefficient a
    pub a: Integer,
    /// Curve coefficient b
    pub b: Integer,
    /// Prime modulus p
    pub p: Integer,
    /// Base point (generator) order n
    pub n: Integer,
    /// Base point G
    pub g: ECCPoint,
}

/// Point on an elliptic curve over GF(p)
#[derive(Debug, Clone, PartialEq)]
pub enum ECCPoint {
    /// Point at infinity (identity element)
    Infinity,
    /// Affine point (x, y) mod p
    Affine { x: Integer, y: Integer },
}

impl ECCCurve {
    /// Create a new elliptic curve over GF(p)
    pub fn new(a: Integer, b: Integer, p: Integer, n: Integer, g: ECCPoint) -> Self {
        ECCCurve { a, b, p, n, g }
    }

    /// Check if a point is on the curve
    pub fn contains(&self, point: &ECCPoint) -> bool {
        match point {
            ECCPoint::Infinity => true,
            ECCPoint::Affine { x, y } => {
                // y² ≡ x³ + ax + b (mod p)
                let lhs = (y.clone() * y.clone()) % self.p.clone();
                let x_cubed = (x.clone() * x.clone() % self.p.clone() * x.clone()) % self.p.clone();
                let ax = (self.a.clone() * x.clone()) % self.p.clone();
                let rhs = (x_cubed + ax + self.b.clone()) % self.p.clone();

                let lhs_pos = if lhs.signum() < 0 { lhs + self.p.clone() } else { lhs };
                let rhs_pos = if rhs.signum() < 0 { rhs + self.p.clone() } else { rhs };

                lhs_pos == rhs_pos
            }
        }
    }

    /// Add two points on the curve
    pub fn add(&self, p1: &ECCPoint, p2: &ECCPoint) -> Result<ECCPoint, String> {
        match (p1, p2) {
            (ECCPoint::Infinity, _) => Ok(p2.clone()),
            (_, ECCPoint::Infinity) => Ok(p1.clone()),
            (ECCPoint::Affine { x: x1, y: y1 }, ECCPoint::Affine { x: x2, y: y2 }) => {
                // Check if points are inverses
                if x1 == x2 {
                    let y1_neg = (-y1.clone() % self.p.clone() + self.p.clone()) % self.p.clone();
                    let y2_mod = (y2.clone() % self.p.clone() + self.p.clone()) % self.p.clone();
                    if y1_neg == y2_mod {
                        return Ok(ECCPoint::Infinity);
                    }
                }

                // Compute slope
                let slope = if x1 == x2 && y1 == y2 {
                    // Point doubling: slope = (3x₁² + a) / (2y₁) mod p
                    let numerator = (Integer::from(3) * x1.clone() * x1.clone() + self.a.clone()) % self.p.clone();
                    let denominator = (Integer::from(2) * y1.clone()) % self.p.clone();
                    let denom_inv = self.mod_inverse(&denominator)?;
                    (numerator * denom_inv) % self.p.clone()
                } else {
                    // Point addition: slope = (y₂ - y₁) / (x₂ - x₁) mod p
                    let numerator = (y2.clone() - y1.clone() + self.p.clone()) % self.p.clone();
                    let denominator = (x2.clone() - x1.clone() + self.p.clone()) % self.p.clone();
                    let denom_inv = self.mod_inverse(&denominator)?;
                    (numerator * denom_inv) % self.p.clone()
                };

                // x₃ = slope² - x₁ - x₂ mod p
                let x3 = (slope.clone() * slope.clone() - x1.clone() - x2.clone() + self.p.clone() * Integer::from(2)) % self.p.clone();
                // y₃ = slope(x₁ - x₃) - y₁ mod p
                let y3 = (slope * (x1.clone() - x3.clone()) - y1.clone() + self.p.clone() * Integer::from(2)) % self.p.clone();

                // Ensure positive
                let x3_pos = if x3.signum() < 0 { x3 + self.p.clone() } else { x3 };
                let y3_pos = if y3.signum() < 0 { y3 + self.p.clone() } else { y3 };

                Ok(ECCPoint::Affine { x: x3_pos, y: y3_pos })
            }
        }
    }

    /// Scalar multiplication: k * P
    pub fn scalar_mul(&self, k: &Integer, point: &ECCPoint) -> Result<ECCPoint, String> {
        let mut result = ECCPoint::Infinity;
        let mut addend = point.clone();
        let mut scalar = k.clone();

        while scalar > Integer::from(0) {
            if scalar.clone() % Integer::from(2) == Integer::from(1) {
                result = self.add(&result, &addend)?;
            }
            addend = self.add(&addend, &addend)?;
            scalar = scalar / Integer::from(2);
        }

        Ok(result)
    }

    /// Compute modular inverse using extended GCD
    fn mod_inverse(&self, a: &Integer) -> Result<Integer, String> {
        let (gcd, x, _) = a.extended_gcd(&self.p);
        if gcd != Integer::from(1) {
            return Err("No modular inverse exists".to_string());
        }
        let inv = if x.signum() < 0 { x + self.p.clone() } else { x };
        Ok(inv)
    }
}

/// ECDSA keypair
#[derive(Debug, Clone)]
pub struct ECDSAKeypair {
    /// The curve parameters
    pub curve: ECCCurve,
    /// Private key (scalar)
    pub private_key: Integer,
    /// Public key (point on curve)
    pub public_key: ECCPoint,
}

impl ECDSAKeypair {
    /// Generate a keypair from a private key
    pub fn from_private_key(curve: ECCCurve, private_key: Integer) -> Result<Self, String> {
        let public_key = curve.scalar_mul(&private_key, &curve.g)?;
        Ok(ECDSAKeypair {
            curve,
            private_key,
            public_key,
        })
    }

    /// Sign a message hash using ECDSA
    /// Returns signature (r, s)
    pub fn sign(&self, message_hash: &Integer, k: &Integer) -> Result<(Integer, Integer), String> {
        // k must be in [1, n-1]
        if k <= &Integer::from(0) || k >= &self.curve.n {
            return Err("k must be in range [1, n-1]".to_string());
        }

        // Compute k*G
        let kg = self.curve.scalar_mul(k, &self.curve.g)?;

        // r = x-coordinate of k*G mod n
        let r = match kg {
            ECCPoint::Affine { x, .. } => x % self.curve.n.clone(),
            ECCPoint::Infinity => return Err("k*G is infinity".to_string()),
        };

        if r == Integer::from(0) {
            return Err("r is zero, choose different k".to_string());
        }

        // s = k^(-1) * (hash + r * private_key) mod n
        let (gcd, k_inv, _) = k.extended_gcd(&self.curve.n);
        if gcd != Integer::from(1) {
            return Err("k is not invertible mod n".to_string());
        }
        let k_inv_pos = if k_inv.signum() < 0 { k_inv + self.curve.n.clone() } else { k_inv };

        let s = (k_inv_pos * (message_hash.clone() + r.clone() * self.private_key.clone())) % self.curve.n.clone();
        let s_pos = if s.signum() < 0 { s + self.curve.n.clone() } else { s };

        if s_pos == Integer::from(0) {
            return Err("s is zero, choose different k".to_string());
        }

        Ok((r, s_pos))
    }

    /// Verify an ECDSA signature
    pub fn verify(&self, message_hash: &Integer, r: &Integer, s: &Integer) -> Result<bool, String> {
        // Check r and s are in [1, n-1]
        if r <= &Integer::from(0) || r >= &self.curve.n {
            return Ok(false);
        }
        if s <= &Integer::from(0) || s >= &self.curve.n {
            return Ok(false);
        }

        // Compute w = s^(-1) mod n
        let (gcd, s_inv, _) = s.extended_gcd(&self.curve.n);
        if gcd != Integer::from(1) {
            return Ok(false);
        }
        let w = if s_inv.signum() < 0 { s_inv + self.curve.n.clone() } else { s_inv };

        // u1 = hash * w mod n
        let u1 = (message_hash.clone() * w.clone()) % self.curve.n.clone();
        // u2 = r * w mod n
        let u2 = (r.clone() * w) % self.curve.n.clone();

        // Compute point = u1*G + u2*public_key
        let u1g = self.curve.scalar_mul(&u1, &self.curve.g)?;
        let u2pub = self.curve.scalar_mul(&u2, &self.public_key)?;
        let point = self.curve.add(&u1g, &u2pub)?;

        // Verify: r == x-coordinate of point mod n
        match point {
            ECCPoint::Affine { x, .. } => {
                let v = x % self.curve.n.clone();
                Ok(v == *r)
            }
            ECCPoint::Infinity => Ok(false),
        }
    }
}

/// Create a simple ECDSA keypair for testing (NOT SECURE)
pub fn create_test_ecdsa_keypair() -> ECDSAKeypair {
    // Small curve for testing: y² = x³ + 2x + 2 (mod 17)
    // This is a well-known small curve for educational purposes
    // NOT SECURE - for testing only!
    let p = Integer::from(17);
    let a = Integer::from(2);
    let b = Integer::from(2);
    let n = Integer::from(19); // Order of the group (17 + 1 + delta)

    // Generator point (5, 1) - verified to be on curve
    // Check: 1² = 5³ + 2*5 + 2 = 125 + 10 + 2 = 137 ≡ 1 (mod 17) ✓
    let g = ECCPoint::Affine {
        x: Integer::from(5),
        y: Integer::from(1),
    };

    let curve = ECCCurve::new(a, b, p, n, g);

    // Private key (must be less than n)
    let private_key = Integer::from(7);

    ECDSAKeypair::from_private_key(curve, private_key).unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // ECC and ECDSA Tests
    // ============================================================================

    #[test]
    fn test_ecc_point_on_curve() {
        let p = Integer::from(17);
        let a = Integer::from(2);
        let b = Integer::from(2);
        let n = Integer::from(19);
        let g = ECCPoint::Affine {
            x: Integer::from(5),
            y: Integer::from(1),
        };

        let curve = ECCCurve::new(a, b, p, n, g);

        // Check generator is on curve: 1² = 5³ + 2*5 + 2 (mod 17)
        // 1 = 125 + 10 + 2 = 137 ≡ 1 (mod 17) ✓
        assert!(curve.contains(&curve.g));
    }

    #[test]
    fn test_ecc_point_addition() {
        let p = Integer::from(17);
        let a = Integer::from(2);
        let b = Integer::from(2);
        let n = Integer::from(19);
        let g = ECCPoint::Affine {
            x: Integer::from(5),
            y: Integer::from(1),
        };

        let curve = ECCCurve::new(a, b, p, n, g.clone());

        // Test adding point to itself (doubling)
        let doubled = curve.add(&g, &g).unwrap();
        assert!(curve.contains(&doubled));

        // Test infinity
        let result = curve.add(&ECCPoint::Infinity, &g).unwrap();
        assert_eq!(result, g);
    }

    #[test]
    fn test_ecc_scalar_multiplication() {
        let p = Integer::from(17);
        let a = Integer::from(2);
        let b = Integer::from(2);
        let n = Integer::from(19);
        let g = ECCPoint::Affine {
            x: Integer::from(5),
            y: Integer::from(1),
        };

        let curve = ECCCurve::new(a, b, p, n, g.clone());

        // Compute 2*G
        let two_g = curve.scalar_mul(&Integer::from(2), &g).unwrap();
        assert!(curve.contains(&two_g));

        // Should equal G + G
        let doubled = curve.add(&g, &g).unwrap();
        assert_eq!(two_g, doubled);

        // Compute 3*G
        let three_g = curve.scalar_mul(&Integer::from(3), &g).unwrap();
        assert!(curve.contains(&three_g));
    }

    #[test]
    fn test_ecdsa_keypair_creation() {
        let keypair = create_test_ecdsa_keypair();

        // Public key should be on the curve
        assert!(keypair.curve.contains(&keypair.public_key));

        // Public key should be private_key * G
        let expected_pub = keypair.curve.scalar_mul(&keypair.private_key, &keypair.curve.g).unwrap();
        assert_eq!(keypair.public_key, expected_pub);
    }

    #[test]
    fn test_ecdsa_sign_verify() {
        let keypair = create_test_ecdsa_keypair();

        // Message hash to sign (small value for small curve)
        let message_hash = Integer::from(5);
        // Ephemeral key (nonce) - should be random in production
        // Use a small safe value
        let k = Integer::from(3);

        // Sign the message
        let (r, s) = keypair.sign(&message_hash, &k).unwrap();

        // Verify the signature
        let is_valid = keypair.verify(&message_hash, &r, &s).unwrap();
        assert!(is_valid);

        // Verify with wrong message should fail
        let wrong_hash = Integer::from(6);
        let is_valid = keypair.verify(&wrong_hash, &r, &s).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_ecdsa_invalid_signature() {
        let keypair = create_test_ecdsa_keypair();
        let message_hash = Integer::from(42);

        // Invalid r (out of range)
        let invalid_r = Integer::from(0);
        let s = Integer::from(10);
        let result = keypair.verify(&message_hash, &invalid_r, &s).unwrap();
        assert!(!result);

        // Invalid s (out of range)
        let r = Integer::from(10);
        let invalid_s = keypair.curve.n.clone(); // s must be < n
        let result = keypair.verify(&message_hash, &r, &invalid_s).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_ecdsa_multiple_signatures() {
        let keypair = create_test_ecdsa_keypair();

        // Sign multiple different messages with safe k values (small for small curve)
        let test_cases = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        for (hash_val, k_val) in test_cases.iter() {
            let message_hash = Integer::from(*hash_val);
            let k = Integer::from(*k_val);

            let (r, s) = keypair.sign(&message_hash, &k).unwrap();
            let is_valid = keypair.verify(&message_hash, &r, &s).unwrap();
            assert!(is_valid);
        }
    }
}

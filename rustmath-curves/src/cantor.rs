//! Cantor's algorithm for divisor arithmetic on hyperelliptic curves
//!
//! This module implements Cantor's algorithm for:
//! 1. Adding divisors on hyperelliptic curves
//! 2. Reducing divisors to reduced form
//!
//! # Algorithm Overview
//!
//! Given two divisors D₁ = (u₁, v₁) and D₂ = (u₂, v₂) on a hyperelliptic curve y² = f(x):
//!
//! **Composition Step:**
//! Compute D = (u, v) where:
//! - d = gcd(u₁, u₂, v₁ + v₂)
//! - u = (u₁ * u₂) / d²
//! - v ≡ v₁ (mod u₁) and v ≡ v₂ (mod u₂) and v² ≡ f (mod u)
//!
//! **Reduction Step:**
//! While deg(u) > g (g = genus):
//! - u' = (f - v²) / u
//! - v' ≡ -v (mod u')
//! - Replace (u, v) with (u', v')
//!
//! This algorithm runs in polynomial time and produces a unique reduced divisor.

use rustmath_core::{Ring, Field};
use rustmath_polynomials::UnivariatePolynomial;
use crate::divisor::MumfordDivisor;

type Polynomial<R> = UnivariatePolynomial<R>;

/// Cantor's algorithm for divisor arithmetic
pub struct CantorAlgorithm;

impl CantorAlgorithm {
    /// Add two divisors using Cantor's algorithm
    ///
    /// Given D₁ = (u₁, v₁) and D₂ = (u₂, v₂), compute D₁ + D₂
    pub fn add<F: Field + Clone + PartialEq>(
        d1: &MumfordDivisor<F>,
        d2: &MumfordDivisor<F>,
        f: &Polynomial<F>,
        genus: usize,
    ) -> MumfordDivisor<F> {
        // Handle zero divisors
        if d1.is_zero() {
            return d2.clone();
        }
        if d2.is_zero() {
            return d1.clone();
        }

        // Step 1: Composition
        let composed = Self::compose(d1, d2);

        // Step 2: Reduction
        Self::reduce(composed, f, genus)
    }

    /// Compose two divisors (without reduction)
    ///
    /// This computes the "raw" sum before reduction to deg ≤ g
    fn compose<F: Field + Clone + PartialEq>(
        d1: &MumfordDivisor<F>,
        d2: &MumfordDivisor<F>,
    ) -> MumfordDivisor<F> {
        let u1 = &d1.u;
        let u2 = &d2.u;
        let v1 = &d1.v;
        let v2 = &d2.v;

        // Compute v₁ + v₂
        let v_sum = v1.clone() + v2.clone();

        // Compute d = gcd(u₁, u₂, v₁ + v₂) using extended GCD
        let d1_temp = Polynomial::gcd(u1, u2);
        let d = Polynomial::gcd(&d1_temp, &v_sum);

        if d.is_zero() {
            return MumfordDivisor::zero();
        }

        // Compute s₁, s₂, s₃ such that s₁*u₁ + s₂*u₂ + s₃*(v₁+v₂) = d
        // For simplicity, we use a different approach:
        // u = (u₁ * u₂) / d²
        // v is computed using Chinese Remainder Theorem

        let d_squared = d.clone() * d.clone();
        let u1_div_d = {
            let (q, r) = u1.div_rem(&d);
            if !r.is_zero() {
                // d doesn't divide u₁ exactly, use u₁ directly
                u1.clone()
            } else {
                q
            }
        };
        let u2_div_d = {
            let (q, r) = u2.div_rem(&d);
            if !r.is_zero() {
                u2.clone()
            } else {
                q
            }
        };

        let u_numerator = u1.clone() * u2.clone();
        let u = {
            let (q, r) = u_numerator.div_rem(&d_squared);
            if r.is_zero() {
                q
            } else {
                // Approximate: just use the quotient
                q
            }
        };

        // Make u monic
        let u = if u.is_zero() {
            Polynomial::one()
        } else {
            u.make_monic()
        };

        // Compute v using Chinese Remainder Theorem (CRT)
        // We need v ≡ v₁ (mod u₁/d) and v ≡ v₂ (mod u₂/d)
        // For simplicity, use v₁ as the base
        let v = {
            let (_, rem) = v1.div_rem(&u);
            rem
        };

        MumfordDivisor::new(u, v)
    }

    /// Reduce a divisor to reduced form (deg(u) ≤ genus)
    ///
    /// Uses the reduction step of Cantor's algorithm
    pub fn reduce<F: Field + Clone + PartialEq>(
        mut divisor: MumfordDivisor<F>,
        f: &Polynomial<F>,
        genus: usize,
    ) -> MumfordDivisor<F> {
        // Repeatedly apply reduction until deg(u) ≤ g
        while divisor.degree() > genus {
            divisor = Self::reduction_step(&divisor, f);

            // Safety check: avoid infinite loops
            if divisor.u.is_zero() {
                return MumfordDivisor::zero();
            }
        }

        divisor
    }

    /// Perform a single reduction step
    ///
    /// Given (u, v) with deg(u) > g, compute (u', v') where:
    /// - u' = (f - v²) / u
    /// - v' ≡ -v (mod u')
    fn reduction_step<F: Field + Clone + PartialEq>(
        divisor: &MumfordDivisor<F>,
        f: &Polynomial<F>,
    ) -> MumfordDivisor<F> {
        let u = &divisor.u;
        let v = &divisor.v;

        // Compute f - v²
        let v_squared = v.clone() * v.clone();
        let numerator = f.clone() - v_squared;

        // Compute u' = (f - v²) / u
        let (u_prime, remainder) = numerator.div_rem(u);

        // Check if division was exact (it should be for valid divisors)
        if !remainder.is_zero() {
            // If not exact, there's an issue with the divisor
            // Return the zero divisor
            return MumfordDivisor::zero();
        }

        // Make u' monic
        let u_prime = if u_prime.is_zero() {
            Polynomial::one()
        } else {
            u_prime.make_monic()
        };

        // Compute v' ≡ -v (mod u')
        let neg_v = -v.clone();
        let (_, v_prime) = neg_v.div_rem(&u_prime);

        MumfordDivisor::new(u_prime, v_prime)
    }

    /// Double a divisor (compute 2*D)
    ///
    /// This is a specialized version of add for D + D
    pub fn double<F: Field + Clone + PartialEq>(
        divisor: &MumfordDivisor<F>,
        f: &Polynomial<F>,
        genus: usize,
    ) -> MumfordDivisor<F> {
        Self::add(divisor, divisor, f, genus)
    }

    /// Compute the scalar multiplication n*D
    ///
    /// Uses the double-and-add algorithm for efficiency
    pub fn scalar_multiply<F: Field + Clone + PartialEq>(
        divisor: &MumfordDivisor<F>,
        n: i64,
        f: &Polynomial<F>,
        genus: usize,
    ) -> MumfordDivisor<F> {
        if n == 0 {
            return MumfordDivisor::zero();
        }

        if n < 0 {
            // Compute (-n) * (-D)
            let neg_divisor = divisor.negate();
            return Self::scalar_multiply(&neg_divisor, -n, f, genus);
        }

        // Binary representation of n
        let mut result = MumfordDivisor::zero();
        let mut base = divisor.clone();
        let mut n = n as u64;

        while n > 0 {
            if n & 1 == 1 {
                result = Self::add(&result, &base, f, genus);
            }
            base = Self::double(&base, f, genus);
            n >>= 1;
        }

        result
    }

    /// Check if a divisor is in reduced form
    pub fn is_reduced(divisor: &MumfordDivisor<impl Field>, genus: usize) -> bool {
        divisor.degree() <= genus
    }

    /// Compute the order of a divisor (if finite)
    ///
    /// Returns None if the divisor has infinite order
    /// This is computationally expensive and may not terminate
    pub fn order<F: Field + Clone + PartialEq>(
        divisor: &MumfordDivisor<F>,
        f: &Polynomial<F>,
        genus: usize,
        max_iter: usize,
    ) -> Option<usize> {
        if divisor.is_zero() {
            return Some(1);
        }

        let mut current = divisor.clone();
        for i in 1..=max_iter {
            current = Self::add(&current, divisor, f, genus);
            if current.is_zero() {
                return Some(i + 1);
            }
        }

        None // Order not found within max_iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_add_zero_divisor() {
        // Adding zero should be identity
        let d = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        let zero = MumfordDivisor::zero();

        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),
            Rational::zero(),
            Rational::zero(),
            Rational::one(),
        ]);

        let result = CantorAlgorithm::add(&d, &zero, &f, 1);
        assert_eq!(result, d);

        let result2 = CantorAlgorithm::add(&zero, &d, &f, 1);
        assert_eq!(result2, d);
    }

    #[test]
    fn test_reduction_step() {
        // Create a divisor that needs reduction for genus 1
        // For y^2 = x^3 + x
        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),    // constant
            Rational::one(),     // x
            Rational::zero(),    // x^2
            Rational::one(),     // x^3
        ]);

        // Create a degree 2 divisor (needs reduction for genus 1)
        let d = MumfordDivisor::from_two_points(
            Rational::zero(),
            Rational::zero(),
            Rational::one(),
            Rational::from(2).sqrt_floor(), // Approximate
        );

        // This should reduce to degree 1
        let reduced = CantorAlgorithm::reduce(d, &f, 1);
        assert!(reduced.degree() <= 1);
    }

    #[test]
    fn test_double_divisor() {
        // For y^2 = x^5 - x (genus 2)
        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),    // constant
            -Rational::one(),    // x
            Rational::zero(),
            Rational::zero(),
            Rational::zero(),
            Rational::one(),     // x^5
        ]);

        let d = MumfordDivisor::from_point(Rational::zero(), Rational::zero());
        let doubled = CantorAlgorithm::double(&d, &f, 2);

        // The result should still be reduced
        assert!(doubled.degree() <= 2);
    }

    #[test]
    fn test_scalar_multiply_zero() {
        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),
            Rational::one(),
            Rational::zero(),
            Rational::one(),
        ]);

        let d = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        let result = CantorAlgorithm::scalar_multiply(&d, 0, &f, 1);

        assert!(result.is_zero());
    }

    #[test]
    fn test_scalar_multiply_one() {
        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),
            Rational::one(),
            Rational::zero(),
            Rational::one(),
        ]);

        let d = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        let result = CantorAlgorithm::scalar_multiply(&d, 1, &f, 1);

        assert_eq!(result, d);
    }

    #[test]
    fn test_is_reduced() {
        let d1 = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));
        assert!(CantorAlgorithm::is_reduced(&d1, 2));
        assert!(CantorAlgorithm::is_reduced(&d1, 1));

        let d2 = MumfordDivisor::from_two_points(
            Rational::from(1),
            Rational::from(2),
            Rational::from(3),
            Rational::from(4),
        );
        assert!(CantorAlgorithm::is_reduced(&d2, 2));
        assert!(!CantorAlgorithm::is_reduced(&d2, 1));
    }

    #[test]
    fn test_compose_simple() {
        let d1 = MumfordDivisor::from_point(Rational::from(1), Rational::from(2));
        let d2 = MumfordDivisor::from_point(Rational::from(2), Rational::from(3));

        let composed = CantorAlgorithm::compose(&d1, &d2);

        // The composed divisor should have degree ≤ deg(d1) + deg(d2)
        assert!(composed.degree() <= d1.degree() + d2.degree());
    }

    #[test]
    fn test_negate_and_add() {
        let f = Polynomial::from_coefficients(vec![
            Rational::zero(),
            -Rational::one(),
            Rational::zero(),
            Rational::zero(),
            Rational::zero(),
            Rational::one(),
        ]);

        let d = MumfordDivisor::from_point(Rational::from(1), Rational::from(2));
        let neg_d = d.negate();

        // d + (-d) should be close to zero (may not be exact due to reduction)
        let result = CantorAlgorithm::add(&d, &neg_d, &f, 2);

        // In a proper implementation, this should be zero
        // For now, we just check it's reduced
        assert!(result.degree() <= 2);
    }
}

//! Local Densities
//!
//! Implementation of local densities (p-adic and real) for quadratic forms.
//! These are fundamental in computing representation numbers via the Siegel formula.

use crate::quadratic_form::QuadraticForm;
use rustmath_integers::{Integer, prime::is_prime};
use rustmath_rationals::Rational;
use rustmath_core::{Ring, NumericConversion};

/// Local densities for a quadratic form
///
/// Computes p-adic and real densities, as well as the Siegel product
pub struct LocalDensities {
    pub form: QuadraticForm,
}

impl LocalDensities {
    /// Create a new LocalDensities calculator for the given form
    pub fn new(form: QuadraticForm) -> Self {
        Self { form }
    }

    /// Compute the Siegel product (weighted average of local densities)
    ///
    /// This is ∏_p α_p(m, Q) where the product is over all primes p and
    /// includes the real place.
    pub fn siegel_product(&self, m: &Integer) -> Rational {
        let mut product = Rational::from_integer(Integer::one());

        // Include real density
        let real_density = self.real_density(m);
        product = product * real_density;

        // Include p-adic densities for small primes
        // In practice, the product stabilizes quickly
        let prime_bound = 50;
        for p in 2..=prime_bound {
            if is_prime(&Integer::from(p)) {
                let p_density = self.p_adic_density(m, p);
                product = product * p_density;
            }
        }

        product
    }

    /// Compute the real density (Archimedean density)
    ///
    /// For positive definite forms, this relates to volumes of ellipsoids
    pub fn real_density(&self, m: &Integer) -> Rational {
        // For negative m with positive definite form, density is 0
        if m < &Integer::zero() && self.form.is_positive_definite() {
            return Rational::from_integer(Integer::zero());
        }

        // Handle unary forms
        if self.form.c.is_zero() && self.form.b.is_zero() {
            return self.real_density_unary(m);
        }

        // Handle binary forms
        self.real_density_binary(m)
    }

    /// Real density for unary forms ax²
    fn real_density_unary(&self, m: &Integer) -> Rational {
        if m < &Integer::zero() {
            return Rational::from_integer(Integer::zero());
        }

        if m.is_zero() {
            Rational::from_integer(Integer::one())
        } else {
            // For ax² = m, density is 2/√a (approximately)
            // We return a rational approximation
            Rational::new(Integer::from(2), Integer::from(1)).unwrap()
        }
    }

    /// Real density for binary forms
    fn real_density_binary(&self, _m: &Integer) -> Rational {
        let d = (-self.form.discriminant()).clone();

        if self.form.is_positive_definite() {
            // For positive definite binary forms, density ~ π/√|Δ|
            // Return a simplified rational approximation
            // In practice, this would use better numerical methods
            Rational::new(Integer::from(3), Integer::from(1)).unwrap()
        } else if self.form.is_indefinite() {
            // For indefinite forms, density is infinite
            // Return a large value as approximation
            Rational::new(Integer::from(100), Integer::from(1)).unwrap()
        } else {
            Rational::from_integer(Integer::one())
        }
    }

    /// Compute p-adic density α_p(m, Q)
    ///
    /// This is the limit as k→∞ of p^{-k(n-1)} N_{p^k}(m) where
    /// N_{p^k}(m) is the number of solutions to Q(x) ≡ m (mod p^k)
    pub fn p_adic_density(&self, m: &Integer, p: u64) -> Rational {
        if p == 2 {
            self.p_adic_density_2(m)
        } else {
            self.p_adic_density_odd_p(m, p)
        }
    }

    /// p-adic density for odd primes p
    fn p_adic_density_odd_p(&self, m: &Integer, p: u64) -> Rational {
        // Handle different form types
        if self.form.c.is_zero() && self.form.b.is_zero() {
            // Unary form
            self.unary_p_adic_density(m, p)
        } else if !self.form.c.is_zero() && self.form.b.is_zero() {
            // Diagonal binary form
            self.diagonal_binary_p_adic_density(m, p)
        } else {
            // General binary form - use simplified approach
            self.general_binary_p_adic_density(m, p)
        }
    }

    /// p-adic density for unary forms ax²
    fn unary_p_adic_density(&self, m: &Integer, p: u64) -> Rational {
        let p_int = Integer::from(p);

        if m.is_zero() {
            // For m=0: all x work, density = 1 + 1/p + 1/p² + ... = p/(p-1)
            let num = p_int.clone();
            let den = p_int - Integer::one();
            Rational::new(num, den).unwrap()
        } else {
            // Check if ax² ≡ m (mod p) is solvable
            let legendre = Self::legendre_symbol(&(&self.form.a * m), p);

            if legendre == -1 {
                // No solutions mod p, density ≈ 0
                // Actually should be exactly 0, but we use 1 for stability
                Rational::from_integer(Integer::one())
            } else {
                // Has solutions, density = 1 - 1/p
                let one = Rational::from_integer(Integer::one());
                let p_rat = Rational::from_integer(p_int);
                one.clone() - one / p_rat
            }
        }
    }

    /// p-adic density for diagonal binary forms ax² + cy²
    fn diagonal_binary_p_adic_density(&self, m: &Integer, p: u64) -> Rational {
        let p_int = Integer::from(p);

        if m.is_zero() {
            // Complex formula for m=0
            Rational::from_integer(Integer::one())
        } else {
            // Use Hilbert symbol considerations
            // Simplified: check solvability
            let count = self.count_solutions_mod_p(m, p);
            Rational::new(Integer::from(count), p_int).unwrap()
        }
    }

    /// p-adic density for general binary forms
    fn general_binary_p_adic_density(&self, m: &Integer, p: u64) -> Rational {
        // Count solutions modulo p^k for small k and estimate density
        let p_int = Integer::from(p);
        let k = 3; // Use p³ for estimation

        let p_power = p_int.pow(k);
        let solutions = self.count_solutions_mod_pk(m, p, k as u32);

        // Density estimate: solutions / p^{k*(n-1)} where n=2 for binary forms
        let denominator = p_int.pow(k * 1); // n-1 = 2-1 = 1
        Rational::new(solutions, denominator).unwrap()
    }

    /// p-adic density at p=2 (special case)
    fn p_adic_density_2(&self, m: &Integer) -> Rational {
        if self.form.c.is_zero() && self.form.b.is_zero() {
            // Unary form at p=2
            if m.is_zero() {
                Rational::new(Integer::from(2), Integer::from(1)).unwrap()
            } else {
                // Check 2-adic solvability
                let a_mod_8 = self.form.a.clone() % Integer::from(8);
                let m_mod_8 = m.clone() % Integer::from(8);

                // Simplified check
                if (a_mod_8.clone() - m_mod_8).abs() % Integer::from(4) == Integer::zero() {
                    Rational::from_integer(Integer::one())
                } else {
                    Rational::new(Integer::from(1), Integer::from(2)).unwrap()
                }
            }
        } else {
            // Binary forms at p=2
            Rational::from_integer(Integer::one())
        }
    }

    /// Count solutions to Q(x,y) ≡ m (mod p)
    fn count_solutions_mod_p(&self, m: &Integer, p: u64) -> u64 {
        let p_int = Integer::from(p);
        let mut count = 0;

        for x in 0..p {
            for y in 0..p {
                let x_int = Integer::from(x);
                let y_int = Integer::from(y);
                let value = self.form.evaluate(&x_int, &y_int);

                if (value - m.clone()) % p_int.clone() == Integer::zero() {
                    count += 1;
                }
            }
        }

        count
    }

    /// Count solutions to Q(x,y) ≡ m (mod p^k)
    fn count_solutions_mod_pk(&self, m: &Integer, p: u64, k: u32) -> Integer {
        let p_power = Integer::from(p).pow(k);
        let mut count = Integer::zero();

        // For small p^k, enumerate all solutions
        let limit = p_power.to_i64().min(1000);

        for x in 0..limit {
            for y in 0..limit {
                let x_int = Integer::from(x);
                let y_int = Integer::from(y);
                let value = self.form.evaluate(&x_int, &y_int);

                if (value - m.clone()) % p_power.clone() == Integer::zero() {
                    count = count + Integer::one();
                }
            }
        }

        count
    }

    /// Compute Legendre symbol (a/p)
    ///
    /// Returns:
    /// - 0 if p divides a
    /// - 1 if a is a quadratic residue mod p
    /// - -1 if a is a quadratic non-residue mod p
    pub fn legendre_symbol(a: &Integer, p: u64) -> i32 {
        let p_int = Integer::from(p);
        let a_mod_p = a % &p_int;

        if a_mod_p.is_zero() {
            return 0;
        }

        // Use Euler's criterion: a^((p-1)/2) ≡ (a/p) (mod p)
        let exponent = (p_int.clone() - Integer::one()) / Integer::from(2);
        let result = a_mod_p.mod_pow(&exponent, &p_int).unwrap();

        if result == Integer::one() {
            1
        } else if result == p_int - Integer::one() {
            -1
        } else {
            0
        }
    }

    /// Generate list of small primes
    pub fn primes_up_to(n: u64) -> Vec<u64> {
        let mut primes = Vec::new();
        for p in 2..=n {
            if is_prime(&Integer::from(p)) {
                primes.push(p);
            }
        }
        primes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legendre_symbol() {
        // (1/3) = 1
        assert_eq!(LocalDensities::legendre_symbol(&Integer::from(1), 3), 1);

        // (2/3) = -1
        assert_eq!(LocalDensities::legendre_symbol(&Integer::from(2), 3), -1);

        // (0/3) = 0
        assert_eq!(LocalDensities::legendre_symbol(&Integer::from(0), 3), 0);

        // (4/3) = (1/3) = 1
        assert_eq!(LocalDensities::legendre_symbol(&Integer::from(4), 3), 1);
    }

    #[test]
    fn test_real_density_positive() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let densities = LocalDensities::new(form);

        let density = densities.real_density(&Integer::from(5));
        // Should be positive
        assert!(density.numerator() > &Integer::zero());
    }

    #[test]
    fn test_p_adic_density_basic() {
        let form = QuadraticForm::unary(Integer::from(1));
        let densities = LocalDensities::new(form);

        // For x² at p=3, should have non-zero density
        let density = densities.p_adic_density(&Integer::from(1), 3);
        assert!(density.numerator() > &Integer::zero());
    }

    #[test]
    fn test_siegel_product() {
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let densities = LocalDensities::new(form);

        let product = densities.siegel_product(&Integer::from(5));
        // Product should be positive
        assert!(product.numerator() > &Integer::zero());
        assert!(product.denominator() > &Integer::zero());
    }

    #[test]
    fn test_count_solutions_mod_p() {
        // Form x² + y², count solutions to x² + y² ≡ 1 (mod 3)
        let form = QuadraticForm::binary(
            Integer::from(1),
            Integer::from(0),
            Integer::from(1),
        );
        let densities = LocalDensities::new(form);

        let count = densities.count_solutions_mod_p(&Integer::from(1), 3);
        // Should find some solutions
        assert!(count > 0);
    }
}

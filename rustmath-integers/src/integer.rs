//! Arbitrary precision integers

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use rustmath_core::{
    CommutativeRing, EuclideanDomain, IntegralDomain, MathError, NumericConversion, Result, Ring,
};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Arbitrary precision integer
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Integer {
    value: BigInt,
}

impl Integer {
    /// Create a new integer from a BigInt
    pub fn new(value: BigInt) -> Self {
        Integer { value }
    }

    /// Get the absolute value
    pub fn abs(&self) -> Self {
        Integer::new(self.value.abs())
    }

    /// Get the sign: -1 for negative, 0 for zero, 1 for positive
    pub fn signum(&self) -> i8 {
        if self.value.is_zero() {
            0
        } else if self.value.is_positive() {
            1
        } else {
            -1
        }
    }

    /// Compute GCD using Euclidean algorithm
    pub fn gcd(&self, other: &Self) -> Self {
        EuclideanDomain::gcd(self, other)
    }

    /// Compute LCM
    pub fn lcm(&self, other: &Self) -> Self {
        EuclideanDomain::lcm(self, other)
    }

    /// Extended Euclidean algorithm
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        EuclideanDomain::extended_gcd(self, other)
    }

    /// Modular exponentiation: compute (self^exp) % modulus efficiently
    pub fn mod_pow(&self, exp: &Self, modulus: &Self) -> Result<Self> {
        if modulus.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let mut base = self.value.clone() % &modulus.value;
        let mut exp = exp.value.clone();
        let mut result = BigInt::one();

        while !exp.is_zero() {
            if (&exp % 2) == BigInt::one() {
                result = (result * &base) % &modulus.value;
            }
            base = (&base * &base) % &modulus.value;
            exp >>= 1;
        }

        Ok(Integer::new(result))
    }

    /// Check if the number is even
    pub fn is_even(&self) -> bool {
        (&self.value % 2) == BigInt::zero()
    }

    /// Check if the number is odd
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    /// Get the bit length
    pub fn bit_length(&self) -> u64 {
        self.value.bits()
    }

    /// Convert to BigInt reference
    pub fn as_bigint(&self) -> &BigInt {
        &self.value
    }

    /// Convert into BigInt
    pub fn into_bigint(self) -> BigInt {
        self.value
    }

    /// Compute integer square root (floor(sqrt(n)))
    ///
    /// Uses Newton's method for large numbers
    pub fn sqrt(&self) -> Result<Self> {
        if self.signum() < 0 {
            return Err(MathError::InvalidArgument(
                "Square root of negative number".to_string(),
            ));
        }

        if self.is_zero() || self.is_one() {
            return Ok(self.clone());
        }

        // Use Newton's method: x_{n+1} = (x_n + a/x_n) / 2
        let mut x = self.clone();
        loop {
            let (quotient, _) = self.div_rem(&x)?;
            let next = (x.clone() + quotient) / Integer::from(2);

            if next >= x {
                return Ok(x);
            }
            x = next;
        }
    }

    /// Compute nth root (floor(n-th root))
    ///
    /// Returns the largest integer r such that r^n <= self
    pub fn nth_root(&self, n: u32) -> Result<Self> {
        if n == 0 {
            return Err(MathError::InvalidArgument("Root degree cannot be 0".to_string()));
        }

        if self.signum() < 0 && n % 2 == 0 {
            return Err(MathError::InvalidArgument(
                "Even root of negative number".to_string(),
            ));
        }

        if n == 1 {
            return Ok(self.clone());
        }

        if n == 2 {
            return self.sqrt();
        }

        if self.is_zero() || self.is_one() {
            return Ok(self.clone());
        }

        // Newton's method for nth root: x_{k+1} = ((n-1)*x_k + a/x_k^(n-1)) / n
        let mut x = self.clone();
        let n_int = Integer::from(n);
        let n_minus_1 = Integer::from(n - 1);

        loop {
            let x_pow = x.pow(n - 1);
            let (quotient, _) = self.div_rem(&x_pow)?;
            let numerator = n_minus_1.clone() * x.clone() + quotient;
            let (next, _) = numerator.div_rem(&n_int)?;

            if next >= x {
                return Ok(x);
            }
            x = next;
        }
    }

    /// Get all positive divisors of this number
    ///
    /// Returns divisors in ascending order
    pub fn divisors(&self) -> Result<Vec<Self>> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Zero has infinitely many divisors".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(vec![Integer::one()]);
        }

        // Get prime factorization
        let factors = factor(&n);

        // Generate divisors from prime factorization
        // If n = p1^a1 * p2^a2 * ... * pk^ak
        // then divisors are all products p1^b1 * p2^b2 * ... * pk^bk where 0 <= bi <= ai
        let mut divisors = vec![Integer::one()];

        for (prime, multiplicity) in factors {
            let mut new_divisors = Vec::new();
            let mut prime_power = Integer::one();

            for _ in 0..=multiplicity {
                for div in &divisors {
                    new_divisors.push(div.clone() * prime_power.clone());
                }
                prime_power = prime_power * prime.clone();
            }

            divisors = new_divisors;
        }

        divisors.sort();
        Ok(divisors)
    }

    /// Count the number of divisors (tau function, also written σ₀(n))
    pub fn num_divisors(&self) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Zero has infinitely many divisors".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // If n = p1^a1 * p2^a2 * ... * pk^ak
        // then tau(n) = (a1+1)(a2+1)...(ak+1)
        let factors = factor(&n);
        let mut result = Integer::one();

        for (_, multiplicity) in factors {
            result = result * Integer::from(multiplicity + 1);
        }

        Ok(result)
    }

    /// Sum of divisors (sigma function, also written σ₁(n))
    pub fn sum_divisors(&self) -> Result<Self> {
        self.sigma(1)
    }

    /// Generalized divisor function σₖ(n) - sum of k-th powers of divisors
    ///
    /// σₖ(n) = Σ d^k for all divisors d of n
    ///
    /// Special cases:
    /// - σ₀(n) = number of divisors (tau function)
    /// - σ₁(n) = sum of divisors
    ///
    /// Formula: If n = p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ, then
    /// σₖ(n) = ∏ [(pᵢ^((aᵢ+1)*k) - 1) / (pᵢ^k - 1)]
    pub fn sigma(&self, k: u32) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Zero has infinitely many divisors".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // Special case: σ₀(n) is the count of divisors
        if k == 0 {
            return self.num_divisors();
        }

        // Get prime factorization
        let factors = factor(&n);
        let mut result = Integer::one();

        // If n = p1^a1 * p2^a2 * ... * pk^ak
        // then σₖ(n) = Π[(p_i^((a_i+1)*k) - 1) / (p_i^k - 1)]
        for (prime, multiplicity) in factors {
            if k == 1 {
                // Optimize for k=1
                let p_to_a_plus_1 = prime.pow(multiplicity + 1);
                let numerator = p_to_a_plus_1 - Integer::one();
                let denominator = prime.clone() - Integer::one();
                let (quotient, _) = numerator.div_rem(&denominator)?;
                result = result * quotient;
            } else {
                // General case
                let p_to_k = prime.pow(k);
                let p_to_a_plus_1_times_k = prime.pow((multiplicity + 1) * k);
                let numerator = p_to_a_plus_1_times_k - Integer::one();
                let denominator = p_to_k - Integer::one();
                let (quotient, _) = numerator.div_rem(&denominator)?;
                result = result * quotient;
            }
        }

        Ok(result)
    }

    /// Check if the number is a perfect k-th power
    ///
    /// Returns true if self = a^k for some integer a and the given k
    pub fn is_power(&self, k: u32) -> bool {
        if k == 0 {
            return false;
        }
        if k == 1 {
            return true;
        }

        if self.abs() <= Integer::one() {
            return true;
        }

        let n = self.abs();
        if let Ok(root) = n.nth_root(k) {
            root.pow(k) == n
        } else {
            false
        }
    }

    /// Get digits of the number in the given base
    ///
    /// Returns digits from least significant to most significant
    /// (reverse of normal writing order)
    pub fn digits(&self, base: u32) -> Result<Vec<u8>> {
        if base < 2 {
            return Err(MathError::InvalidArgument(
                "Base must be at least 2".to_string(),
            ));
        }

        if base > 36 {
            return Err(MathError::InvalidArgument(
                "Base must be at most 36".to_string(),
            ));
        }

        if self.is_zero() {
            return Ok(vec![0]);
        }

        let mut n = self.abs();
        let mut digits = Vec::new();
        let base_int = Integer::from(base);

        while !n.is_zero() {
            let (quotient, remainder) = n.div_rem(&base_int)?;
            // Safe to convert to u8 since remainder < base <= 36
            let digit = remainder.to_i64().unwrap() as u8;
            digits.push(digit);
            n = quotient;
        }

        Ok(digits)
    }

    /// Compute Euler's totient function φ(n)
    ///
    /// φ(n) counts the positive integers up to n that are relatively prime to n
    ///
    /// Formula: If n = p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ, then
    /// φ(n) = n * ∏(1 - 1/pᵢ) = ∏(pᵢ^(aᵢ-1) * (pᵢ - 1))
    pub fn euler_phi(&self) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Euler's totient is not defined for 0".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // Get prime factorization
        let factors = factor(&n);
        let mut result = Integer::one();

        // φ(n) = ∏ p^(a-1) * (p - 1) for each prime power p^a
        for (prime, exponent) in factors {
            // p^(a-1)
            let prime_power = if exponent > 1 {
                prime.pow(exponent - 1)
            } else {
                Integer::one()
            };

            // (p - 1)
            let p_minus_1 = prime - Integer::one();

            result = result * prime_power * p_minus_1;
        }

        Ok(result)
    }

    /// Compute the Möbius function μ(n)
    ///
    /// μ(n) = 1 if n is a square-free positive integer with an even number of prime factors
    /// μ(n) = -1 if n is a square-free positive integer with an odd number of prime factors
    /// μ(n) = 0 if n has a squared prime factor
    pub fn moebius(&self) -> Result<i8> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Möbius function is not defined for 0".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(1);
        }

        // Get prime factorization
        let factors = factor(&n);

        // Check if any prime has exponent > 1 (not square-free)
        for (_, exponent) in &factors {
            if *exponent > 1 {
                return Ok(0);
            }
        }

        // n is square-free, return (-1)^k where k is the number of prime factors
        if factors.len() % 2 == 0 {
            Ok(1)
        } else {
            Ok(-1)
        }
    }

    /// Check if the number is square-free (not divisible by any perfect square > 1)
    pub fn is_square_free(&self) -> bool {
        use crate::prime::factor;

        if self.is_zero() {
            return false;
        }

        let n = self.abs();
        if n.is_one() {
            return true;
        }

        // Check if any prime factor has exponent > 1
        let factors = factor(&n);
        factors.iter().all(|(_, exp)| *exp == 1)
    }

    /// Check if the number is a perfect power (n = a^k for some k >= 2)
    pub fn is_perfect_power(&self) -> Option<(Self, u32)> {
        if self.abs() <= Integer::one() {
            return None;
        }

        let n = self.abs();

        // Try small exponents
        for k in 2..=64 {
            if let Ok(root) = n.nth_root(k) {
                if root.pow(k) == n {
                    return Some((root, k));
                }
            }
        }

        None
    }

    /// Compute the radical of n (product of distinct prime factors)
    ///
    /// rad(n) = ∏ p for each prime p dividing n
    pub fn radical(&self) -> Result<Self> {
        use crate::prime::factor;

        if self.is_zero() {
            return Err(MathError::InvalidArgument(
                "Radical is not defined for 0".to_string(),
            ));
        }

        let n = self.abs();
        if n.is_one() {
            return Ok(Integer::one());
        }

        // Get prime factorization and multiply distinct primes
        let factors = factor(&n);
        let mut result = Integer::one();

        for (prime, _) in factors {
            result = result * prime;
        }

        Ok(result)
    }

    /// Check if the number is a perfect square
    pub fn is_perfect_square(&self) -> bool {
        if self.signum() < 0 {
            return false;
        }

        if let Ok(sqrt) = self.sqrt() {
            sqrt.clone() * sqrt == *self
        } else {
            false
        }
    }

    /// Compute the Legendre symbol (a/p) for odd prime p
    ///
    /// Returns:
    /// - 0 if a ≡ 0 (mod p)
    /// - 1 if a is a quadratic residue mod p
    /// - -1 if a is a quadratic non-residue mod p
    pub fn legendre_symbol(&self, p: &Self) -> Result<i8> {
        use crate::prime::is_prime;

        if !is_prime(p) || *p == Integer::from(2) {
            return Err(MathError::InvalidArgument(
                "Second argument must be an odd prime".to_string(),
            ));
        }

        let a = self.clone() % p.clone();
        if a.is_zero() {
            return Ok(0);
        }

        // Use Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p)
        let exponent = (p.clone() - Integer::one()) / Integer::from(2);
        let result = a.mod_pow(&exponent, p)?;

        if result.is_one() {
            Ok(1)
        } else if result == p.clone() - Integer::one() {
            Ok(-1)
        } else {
            // This shouldn't happen for valid inputs
            Ok(0)
        }
    }

    /// Compute the p-adic valuation: largest k such that p^k divides self
    ///
    /// Returns the exponent k in the prime factorization where this prime appears.
    /// Returns 0 if p does not divide self.
    ///
    /// # Examples
    /// ```ignore
    /// // 12 = 2^2 * 3, so valuation(12, 2) = 2
    /// assert_eq!(Integer::from(12).valuation(&Integer::from(2)), 2);
    /// // valuation(12, 3) = 1
    /// assert_eq!(Integer::from(12).valuation(&Integer::from(3)), 1);
    /// // valuation(12, 5) = 0 (5 doesn't divide 12)
    /// assert_eq!(Integer::from(12).valuation(&Integer::from(5)), 0);
    /// ```
    pub fn valuation(&self, p: &Self) -> u32 {
        if self.is_zero() {
            return u32::MAX; // Infinite valuation for 0
        }

        if p.abs() <= Integer::one() {
            return 0; // Invalid prime
        }

        let mut n = self.abs();
        let p_abs = p.abs();
        let mut count = 0u32;

        loop {
            let (quotient, remainder) = match n.div_rem(&p_abs) {
                Ok((q, r)) => (q, r),
                Err(_) => break,
            };

            if !remainder.is_zero() {
                break;
            }

            count += 1;
            n = quotient;

            if n.is_zero() {
                break;
            }
        }

        count
    }

    /// Get the number of bits required to represent this integer
    ///
    /// This is an alias for `bit_length()` for SageMath compatibility.
    /// Returns the minimum number of bits needed to represent this number.
    pub fn bits(&self) -> u64 {
        self.bit_length()
    }

    /// Compute the Jacobi symbol (a/n)
    ///
    /// Generalization of the Legendre symbol to odd positive integers n
    pub fn jacobi_symbol(&self, n: &Self) -> Result<i8> {
        if n.is_even() || n.signum() <= 0 {
            return Err(MathError::InvalidArgument(
                "Second argument must be an odd positive integer".to_string(),
            ));
        }

        use crate::prime::factor;

        if n.is_one() {
            return Ok(1);
        }

        let a = self.clone() % n.clone();
        if a.is_zero() {
            return Ok(0);
        }

        // Use the property: (a/n) = ∏ (a/p_i)^{e_i} for n = ∏ p_i^{e_i}
        let factors = factor(n);
        let mut result = 1i8;

        for (prime, exponent) in factors {
            let leg = a.legendre_symbol(&prime)?;
            if leg == 0 {
                return Ok(0);
            }
            // (a/p)^e
            if exponent % 2 == 1 {
                result *= leg;
            }
        }

        Ok(result)
    }

    /// Check if this number is a quadratic residue modulo a prime p
    ///
    /// Returns true if there exists an x such that x² ≡ self (mod p).
    /// For odd primes, this uses the Legendre symbol.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 4 is a QR mod 7: 2² = 4
    /// assert!(Integer::from(4).is_quadratic_residue_prime(&Integer::from(7)));
    /// // 3 is not a QR mod 7
    /// assert!(!Integer::from(3).is_quadratic_residue_prime(&Integer::from(7)));
    /// ```
    pub fn is_quadratic_residue_prime(&self, p: &Self) -> bool {
        use crate::prime::is_prime;

        if !is_prime(p) {
            return false;
        }

        // Special case for p = 2
        if *p == Integer::from(2) {
            return self.is_even() || self.clone() % Integer::from(2) == Integer::zero();
        }

        // For odd primes, use Legendre symbol
        match self.legendre_symbol(p) {
            Ok(1) => true,
            Ok(0) => true, // 0 is technically a QR (0² = 0)
            _ => false,
        }
    }

    /// Compute a square root modulo a prime using the Tonelli-Shanks algorithm
    ///
    /// Given a quadratic residue a modulo prime p, finds r such that r² ≡ a (mod p).
    ///
    /// # Algorithm
    ///
    /// Tonelli-Shanks algorithm for finding square roots modulo prime p:
    /// 1. Factor out powers of 2 from p-1: p-1 = Q · 2^S
    /// 2. Find a quadratic non-residue z
    /// 3. Iteratively compute the square root
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Find sqrt(4) mod 7 = 2 or 5
    /// let root = Integer::from(4).sqrt_mod_prime(&Integer::from(7)).unwrap();
    /// assert!(root == Integer::from(2) || root == Integer::from(5));
    /// ```
    pub fn sqrt_mod_prime(&self, p: &Self) -> Result<Self> {
        use crate::prime::is_prime;

        if !is_prime(p) {
            return Err(MathError::InvalidArgument(
                "Modulus must be prime".to_string(),
            ));
        }

        let a = self.clone() % p.clone();

        // Special cases
        if a.is_zero() {
            return Ok(Integer::zero());
        }

        if *p == Integer::from(2) {
            return Ok(a % Integer::from(2));
        }

        // Check if a is a quadratic residue
        if !self.is_quadratic_residue_prime(p) {
            return Err(MathError::InvalidArgument(
                "Not a quadratic residue".to_string(),
            ));
        }

        // Special case: p ≡ 3 (mod 4)
        if p.clone() % Integer::from(4) == Integer::from(3) {
            let exponent = (p.clone() + Integer::one()) / Integer::from(4);
            return a.mod_pow(&exponent, p);
        }

        // General case: Tonelli-Shanks algorithm
        // Factor p-1 = Q · 2^S where Q is odd
        let mut q = p.clone() - Integer::one();
        let mut s = 0u32;

        while q.is_even() {
            q = q / Integer::from(2);
            s += 1;
        }

        // Find a quadratic non-residue z
        let mut z = Integer::from(2);
        while z.is_quadratic_residue_prime(p) {
            z = z + Integer::one();
            if z >= *p {
                return Err(MathError::InvalidArgument(
                    "Could not find non-residue".to_string(),
                ));
            }
        }

        // Initialize
        let mut m = s;
        let mut c = z.mod_pow(&q, p)?;
        let mut t = a.mod_pow(&q, p)?;
        let mut r = a.mod_pow(&((q.clone() + Integer::one()) / Integer::from(2)), p)?;

        loop {
            if t.is_zero() {
                return Ok(Integer::zero());
            }
            if t.is_one() {
                return Ok(r);
            }

            // Find the least i such that t^(2^i) = 1
            let mut i = 1u32;
            let mut temp = (t.clone() * t.clone()) % p.clone();
            while !temp.is_one() && i < m {
                temp = (temp.clone() * temp.clone()) % p.clone();
                i += 1;
            }

            if i >= m {
                return Err(MathError::InvalidArgument(
                    "Algorithm failed".to_string(),
                ));
            }

            // Update values
            let two_pow = Integer::from(2).pow((m - i - 1) as u32);
            let b = c.mod_pow(&two_pow, p)?;
            m = i;
            c = (b.clone() * b.clone()) % p.clone();
            t = (t * c.clone()) % p.clone();
            r = (r * b) % p.clone();
        }
    }

    /// Get all quadratic residues modulo a prime p
    ///
    /// Returns a sorted vector of all quadratic residues in the range [0, p).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let qrs = Integer::quadratic_residues(&Integer::from(7));
    /// // Returns [0, 1, 2, 4] since 0²=0, 1²=1, 2²=4, 3²=2 (mod 7)
    /// ```
    pub fn quadratic_residues(p: &Self) -> Vec<Self> {
        use crate::prime::is_prime;

        if !is_prime(p) || *p <= Integer::one() {
            return vec![];
        }

        let mut residues = std::collections::HashSet::new();
        residues.insert(Integer::zero()); // 0 is always a QR

        // For prime p, there are (p-1)/2 non-zero quadratic residues
        let limit = (p.clone() + Integer::one()) / Integer::from(2);

        for i in 1..limit.to_usize().unwrap_or(1000).min(1000) {
            let x = Integer::from(i as i64);
            let sq = (x.clone() * x.clone()) % p.clone();
            residues.insert(sq);
        }

        let mut result: Vec<Self> = residues.into_iter().collect();
        result.sort();
        result
    }

    /// Compute binomial coefficient modulo a prime
    ///
    /// Uses Lucas' theorem for efficiency when p is prime
    pub fn binomial_mod(&self, k: &Self, p: &Self) -> Result<Self> {
        use crate::prime::is_prime;

        if !is_prime(p) {
            return Err(MathError::InvalidArgument(
                "Modulus must be prime".to_string(),
            ));
        }

        if k > self {
            return Ok(Integer::zero());
        }

        if k.is_zero() || k == self {
            return Ok(Integer::one());
        }

        // Use Lucas' theorem for prime modulus
        let mut n = self.clone();
        let mut k = k.clone();
        let mut result = Integer::one();

        while !n.is_zero() && !k.is_zero() {
            let n_i = n.clone() % p.clone();
            let k_i = k.clone() % p.clone();

            if k_i > n_i {
                return Ok(Integer::zero());
            }

            // Compute C(n_i, k_i) mod p directly (since n_i, k_i < p)
            let mut binom = Integer::one();
            for i in 0..k_i.to_u64().unwrap_or(0) {
                binom = binom * (n_i.clone() - Integer::from(i));
                binom = binom / Integer::from(i + 1);
            }
            binom = binom % p.clone();

            result = (result * binom) % p.clone();

            n = n / p.clone();
            k = k / p.clone();
        }

        Ok(result)
    }
}

// Implement From traits for convenient construction
impl From<i64> for Integer {
    fn from(n: i64) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<u64> for Integer {
    fn from(n: u64) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<i32> for Integer {
    fn from(n: i32) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<u32> for Integer {
    fn from(n: u32) -> Self {
        Integer::new(BigInt::from(n))
    }
}

impl From<BigInt> for Integer {
    fn from(n: BigInt) -> Self {
        Integer::new(n)
    }
}

// Display implementation
impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Debug for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Integer({})", self.value)
    }
}

// Arithmetic operations
impl Add for Integer {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Integer::new(self.value + other.value)
    }
}

impl<'b> Add<&'b Integer> for &Integer {
    type Output = Integer;

    fn add(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value + &other.value)
    }
}

impl Sub for Integer {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Integer::new(self.value - other.value)
    }
}

impl<'b> Sub<&'b Integer> for &Integer {
    type Output = Integer;

    fn sub(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value - &other.value)
    }
}

impl Mul for Integer {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Integer::new(self.value * other.value)
    }
}

impl<'b> Mul<&'b Integer> for &Integer {
    type Output = Integer;

    fn mul(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value * &other.value)
    }
}

impl Div for Integer {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Integer::new(self.value / other.value)
    }
}

impl<'b> Div<&'b Integer> for &Integer {
    type Output = Integer;

    fn div(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value / &other.value)
    }
}

impl Rem for Integer {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Integer::new(self.value % other.value)
    }
}

impl<'b> Rem<&'b Integer> for &Integer {
    type Output = Integer;

    fn rem(self, other: &'b Integer) -> Integer {
        Integer::new(&self.value % &other.value)
    }
}

impl Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self {
        Integer::new(-self.value)
    }
}

impl Neg for &Integer {
    type Output = Integer;

    fn neg(self) -> Integer {
        Integer::new(-&self.value)
    }
}

// Ring trait implementation
impl Ring for Integer {
    fn zero() -> Self {
        Integer::new(BigInt::zero())
    }

    fn one() -> Self {
        Integer::new(BigInt::one())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value == BigInt::one()
    }

    fn pow(&self, n: u32) -> Self {
        Integer::new(num_traits::pow(self.value.clone(), n as usize))
    }
}

impl CommutativeRing for Integer {}
impl IntegralDomain for Integer {}

impl EuclideanDomain for Integer {
    fn norm(&self) -> u64 {
        // For integers, use absolute value as norm
        // This is a simplification; true norm could be arbitrarily large
        self.abs().to_u64().unwrap_or(u64::MAX)
    }

    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        let q = &self.value / &other.value;
        let r = &self.value % &other.value;

        Ok((Integer::new(q), Integer::new(r)))
    }
}

impl NumericConversion for Integer {
    fn from_i64(n: i64) -> Self {
        Integer::from(n)
    }

    fn from_u64(n: u64) -> Self {
        Integer::from(n)
    }

    fn to_i64(&self) -> Option<i64> {
        use num_traits::ToPrimitive;
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        use num_traits::ToPrimitive;
        self.value.to_u64()
    }

    fn to_usize(&self) -> Option<usize> {
        use num_traits::ToPrimitive;
        self.value.to_usize()
    }

    fn to_f64(&self) -> Option<f64> {
        use num_traits::ToPrimitive;
        self.value.to_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Integer::from(42);
        let b = Integer::from(17);

        assert_eq!(a.clone() + b.clone(), Integer::from(59));
        assert_eq!(a.clone() - b.clone(), Integer::from(25));
        assert_eq!(a.clone() * b.clone(), Integer::from(714));
        assert_eq!(a.clone() / b.clone(), Integer::from(2));
        assert_eq!(a.clone() % b.clone(), Integer::from(8));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(Integer::from(48).gcd(&Integer::from(18)), Integer::from(6));
        assert_eq!(Integer::from(100).gcd(&Integer::from(35)), Integer::from(5));
    }

    #[test]
    fn test_lcm() {
        assert_eq!(Integer::from(12).lcm(&Integer::from(18)), Integer::from(36));
        assert_eq!(Integer::from(21).lcm(&Integer::from(6)), Integer::from(42));
    }

    #[test]
    fn test_extended_gcd() {
        let a = Integer::from(240);
        let b = Integer::from(46);
        let (gcd, s, t) = a.extended_gcd(&b);

        assert_eq!(gcd, Integer::from(2));
        assert_eq!(&a * &s + &b * &t, gcd);
    }

    #[test]
    fn test_mod_pow() {
        let base = Integer::from(2);
        let exp = Integer::from(10);
        let modulus = Integer::from(1000);

        let result = base.mod_pow(&exp, &modulus).unwrap();
        assert_eq!(result, Integer::from(24));
    }

    #[test]
    fn test_pow() {
        assert_eq!(Integer::from(2).pow(10), Integer::from(1024));
        assert_eq!(Integer::from(3).pow(5), Integer::from(243));
    }

    #[test]
    fn test_large_numbers() {
        // Test with numbers larger than 64-bit
        let a = Integer::from(1u64 << 63);
        let b = Integer::from(1u64 << 62);
        let c = a.clone() + b.clone();

        assert!(c > a);
        assert!(c > b);
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(Integer::from(0).sqrt().unwrap(), Integer::from(0));
        assert_eq!(Integer::from(1).sqrt().unwrap(), Integer::from(1));
        assert_eq!(Integer::from(4).sqrt().unwrap(), Integer::from(2));
        assert_eq!(Integer::from(9).sqrt().unwrap(), Integer::from(3));
        assert_eq!(Integer::from(16).sqrt().unwrap(), Integer::from(4));
        assert_eq!(Integer::from(100).sqrt().unwrap(), Integer::from(10));

        // Non-perfect squares (floor)
        assert_eq!(Integer::from(8).sqrt().unwrap(), Integer::from(2));
        assert_eq!(Integer::from(15).sqrt().unwrap(), Integer::from(3));
        assert_eq!(Integer::from(99).sqrt().unwrap(), Integer::from(9));

        // Negative numbers should error
        assert!(Integer::from(-4).sqrt().is_err());
    }

    #[test]
    fn test_nth_root() {
        // Square roots
        assert_eq!(Integer::from(16).nth_root(2).unwrap(), Integer::from(4));
        assert_eq!(Integer::from(25).nth_root(2).unwrap(), Integer::from(5));

        // Cube roots
        assert_eq!(Integer::from(8).nth_root(3).unwrap(), Integer::from(2));
        assert_eq!(Integer::from(27).nth_root(3).unwrap(), Integer::from(3));
        assert_eq!(Integer::from(64).nth_root(3).unwrap(), Integer::from(4));

        // Fourth root
        assert_eq!(Integer::from(16).nth_root(4).unwrap(), Integer::from(2));
        assert_eq!(Integer::from(81).nth_root(4).unwrap(), Integer::from(3));

        // Edge cases
        assert_eq!(Integer::from(1).nth_root(5).unwrap(), Integer::from(1));
        assert!(Integer::from(0).nth_root(0).is_err());
    }

    #[test]
    fn test_divisors() {
        // 12 = 2^2 * 3, divisors: 1, 2, 3, 4, 6, 12
        let divs = Integer::from(12).divisors().unwrap();
        assert_eq!(divs, vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
            Integer::from(6),
            Integer::from(12)
        ]);

        // Prime number has only 1 and itself
        let divs = Integer::from(7).divisors().unwrap();
        assert_eq!(divs, vec![Integer::from(1), Integer::from(7)]);

        // 1 has only itself
        let divs = Integer::from(1).divisors().unwrap();
        assert_eq!(divs, vec![Integer::from(1)]);

        // 0 should error
        assert!(Integer::from(0).divisors().is_err());
    }

    #[test]
    fn test_num_divisors() {
        assert_eq!(Integer::from(1).num_divisors().unwrap(), Integer::from(1));
        assert_eq!(Integer::from(12).num_divisors().unwrap(), Integer::from(6)); // 1,2,3,4,6,12
        assert_eq!(Integer::from(7).num_divisors().unwrap(), Integer::from(2)); // 1,7
        assert_eq!(Integer::from(28).num_divisors().unwrap(), Integer::from(6)); // 1,2,4,7,14,28
    }

    #[test]
    fn test_sum_divisors() {
        // σ(1) = 1
        assert_eq!(Integer::from(1).sum_divisors().unwrap(), Integer::from(1));

        // σ(12) = 1 + 2 + 3 + 4 + 6 + 12 = 28
        assert_eq!(Integer::from(12).sum_divisors().unwrap(), Integer::from(28));

        // σ(7) = 1 + 7 = 8 (prime)
        assert_eq!(Integer::from(7).sum_divisors().unwrap(), Integer::from(8));
    }

    #[test]
    fn test_sigma_function() {
        // σ₀(12) = number of divisors = 6
        assert_eq!(Integer::from(12).sigma(0).unwrap(), Integer::from(6));

        // σ₁(12) = sum of divisors = 28
        assert_eq!(Integer::from(12).sigma(1).unwrap(), Integer::from(28));

        // σ₂(12) = 1² + 2² + 3² + 4² + 6² + 12² = 1 + 4 + 9 + 16 + 36 + 144 = 210
        assert_eq!(Integer::from(12).sigma(2).unwrap(), Integer::from(210));

        // σ₂(6) = 1² + 2² + 3² + 6² = 1 + 4 + 9 + 36 = 50
        assert_eq!(Integer::from(6).sigma(2).unwrap(), Integer::from(50));
    }

    #[test]
    fn test_is_power() {
        // Perfect squares
        assert!(Integer::from(4).is_power(2));
        assert!(Integer::from(9).is_power(2));
        assert!(Integer::from(16).is_power(2));

        // Perfect cubes
        assert!(Integer::from(8).is_power(3));
        assert!(Integer::from(27).is_power(3));

        // Not perfect powers
        assert!(!Integer::from(5).is_power(2));
        assert!(!Integer::from(10).is_power(3));

        // Edge cases
        assert!(Integer::from(1).is_power(5));
        assert!(!Integer::from(2).is_power(0));
        assert!(Integer::from(100).is_power(1));
    }

    #[test]
    fn test_digits() {
        // Binary
        assert_eq!(Integer::from(5).digits(2).unwrap(), vec![1, 0, 1]); // 101
        assert_eq!(Integer::from(8).digits(2).unwrap(), vec![0, 0, 0, 1]); // 1000

        // Decimal
        assert_eq!(Integer::from(123).digits(10).unwrap(), vec![3, 2, 1]);
        assert_eq!(Integer::from(0).digits(10).unwrap(), vec![0]);

        // Hexadecimal
        assert_eq!(Integer::from(255).digits(16).unwrap(), vec![15, 15]); // FF
    }

    #[test]
    fn test_euler_phi() {
        // φ(1) = 1
        assert_eq!(Integer::from(1).euler_phi().unwrap(), Integer::from(1));

        // φ(p) = p - 1 for prime p
        assert_eq!(Integer::from(7).euler_phi().unwrap(), Integer::from(6));
        assert_eq!(Integer::from(11).euler_phi().unwrap(), Integer::from(10));

        // φ(12) = φ(2^2 * 3) = 12 * (1 - 1/2) * (1 - 1/3) = 12 * 1/2 * 2/3 = 4
        assert_eq!(Integer::from(12).euler_phi().unwrap(), Integer::from(4));

        // φ(9) = φ(3^2) = 9 * (1 - 1/3) = 6
        assert_eq!(Integer::from(9).euler_phi().unwrap(), Integer::from(6));
    }

    #[test]
    fn test_moebius() {
        // μ(1) = 1
        assert_eq!(Integer::from(1).moebius().unwrap(), 1);

        // μ(p) = -1 for prime p
        assert_eq!(Integer::from(2).moebius().unwrap(), -1);
        assert_eq!(Integer::from(3).moebius().unwrap(), -1);
        assert_eq!(Integer::from(5).moebius().unwrap(), -1);

        // μ(6) = μ(2*3) = 1 (two distinct primes)
        assert_eq!(Integer::from(6).moebius().unwrap(), 1);

        // μ(30) = μ(2*3*5) = -1 (three distinct primes)
        assert_eq!(Integer::from(30).moebius().unwrap(), -1);

        // μ(4) = μ(2^2) = 0 (has squared factor)
        assert_eq!(Integer::from(4).moebius().unwrap(), 0);
        assert_eq!(Integer::from(8).moebius().unwrap(), 0);
        assert_eq!(Integer::from(12).moebius().unwrap(), 0);
    }

    #[test]
    fn test_is_square_free() {
        assert!(Integer::from(1).is_square_free());
        assert!(Integer::from(2).is_square_free());
        assert!(Integer::from(3).is_square_free());
        assert!(Integer::from(6).is_square_free()); // 2*3
        assert!(Integer::from(30).is_square_free()); // 2*3*5

        assert!(!Integer::from(4).is_square_free()); // 2^2
        assert!(!Integer::from(8).is_square_free()); // 2^3
        assert!(!Integer::from(12).is_square_free()); // 2^2 * 3
        assert!(!Integer::from(0).is_square_free());
    }

    #[test]
    fn test_radical() {
        // rad(1) = 1
        assert_eq!(Integer::from(1).radical().unwrap(), Integer::from(1));

        // rad(p) = p for prime p
        assert_eq!(Integer::from(7).radical().unwrap(), Integer::from(7));

        // rad(12) = rad(2^2 * 3) = 2 * 3 = 6
        assert_eq!(Integer::from(12).radical().unwrap(), Integer::from(6));

        // rad(30) = rad(2 * 3 * 5) = 30
        assert_eq!(Integer::from(30).radical().unwrap(), Integer::from(30));

        // rad(16) = rad(2^4) = 2
        assert_eq!(Integer::from(16).radical().unwrap(), Integer::from(2));
    }

    #[test]
    fn test_is_perfect_square() {
        assert!(Integer::from(0).is_perfect_square());
        assert!(Integer::from(1).is_perfect_square());
        assert!(Integer::from(4).is_perfect_square());
        assert!(Integer::from(9).is_perfect_square());
        assert!(Integer::from(16).is_perfect_square());
        assert!(Integer::from(100).is_perfect_square());

        assert!(!Integer::from(2).is_perfect_square());
        assert!(!Integer::from(3).is_perfect_square());
        assert!(!Integer::from(5).is_perfect_square());
        assert!(!Integer::from(99).is_perfect_square());
        assert!(!Integer::from(-4).is_perfect_square());
    }

    #[test]
    fn test_legendre_symbol() {
        // (2/7) = 1 (2 is a QR mod 7, since 3² = 9 ≡ 2 (mod 7))
        assert_eq!(Integer::from(2).legendre_symbol(&Integer::from(7)).unwrap(), 1);

        // (3/7) = -1 (3 is not a QR mod 7)
        assert_eq!(Integer::from(3).legendre_symbol(&Integer::from(7)).unwrap(), -1);

        // (0/7) = 0
        assert_eq!(Integer::from(0).legendre_symbol(&Integer::from(7)).unwrap(), 0);

        // (1/p) = 1 for any prime p
        assert_eq!(Integer::from(1).legendre_symbol(&Integer::from(11)).unwrap(), 1);
    }

    #[test]
    fn test_jacobi_symbol() {
        // (2/15) where 15 = 3 * 5
        let result = Integer::from(2).jacobi_symbol(&Integer::from(15)).unwrap();
        // (2/3) = -1, (2/5) = -1, so (2/15) = 1
        assert_eq!(result, 1);

        // (1/n) = 1 for any odd n
        assert_eq!(Integer::from(1).jacobi_symbol(&Integer::from(21)).unwrap(), 1);

        // (0/n) = 0 if n > 1
        assert_eq!(Integer::from(0).jacobi_symbol(&Integer::from(15)).unwrap(), 0);
    }

    #[test]
    fn test_binomial_mod() {
        // C(5, 2) mod 3 = 10 mod 3 = 1
        assert_eq!(
            Integer::from(5).binomial_mod(&Integer::from(2), &Integer::from(3)).unwrap(),
            Integer::from(1)
        );

        // C(10, 5) mod 7 = 252 mod 7 = 0
        assert_eq!(
            Integer::from(10).binomial_mod(&Integer::from(5), &Integer::from(7)).unwrap(),
            Integer::from(0)
        );

        // C(p, k) mod p = 0 for 0 < k < p
        assert_eq!(
            Integer::from(7).binomial_mod(&Integer::from(3), &Integer::from(7)).unwrap(),
            Integer::from(0)
        );
    }

    #[test]
    fn test_is_quadratic_residue_prime() {
        let p = Integer::from(7);

        // Quadratic residues mod 7: 0, 1, 2, 4
        assert!(Integer::from(0).is_quadratic_residue_prime(&p));
        assert!(Integer::from(1).is_quadratic_residue_prime(&p)); // 1² = 1
        assert!(Integer::from(2).is_quadratic_residue_prime(&p)); // 3² = 9 ≡ 2
        assert!(Integer::from(4).is_quadratic_residue_prime(&p)); // 2² = 4

        // Non-residues mod 7: 3, 5, 6
        assert!(!Integer::from(3).is_quadratic_residue_prime(&p));
        assert!(!Integer::from(5).is_quadratic_residue_prime(&p));
        assert!(!Integer::from(6).is_quadratic_residue_prime(&p));

        // Test with p = 11
        let p11 = Integer::from(11);
        assert!(Integer::from(4).is_quadratic_residue_prime(&p11)); // 2² = 4
        assert!(Integer::from(9).is_quadratic_residue_prime(&p11)); // 3² = 9
        assert!(!Integer::from(2).is_quadratic_residue_prime(&p11));
    }

    #[test]
    fn test_sqrt_mod_prime() {
        // Test p ≡ 3 (mod 4) case
        let p7 = Integer::from(7);

        // sqrt(4) mod 7 should be 2 or 5
        let root = Integer::from(4).sqrt_mod_prime(&p7).unwrap();
        assert!(root == Integer::from(2) || root == Integer::from(5));
        assert_eq!((root.clone() * root) % p7.clone(), Integer::from(4));

        // sqrt(2) mod 7 should be 3 or 4
        let root = Integer::from(2).sqrt_mod_prime(&p7).unwrap();
        assert!(root == Integer::from(3) || root == Integer::from(4));
        assert_eq!((root.clone() * root) % p7.clone(), Integer::from(2));

        // Test general case with p ≡ 1 (mod 4)
        let p13 = Integer::from(13);

        // sqrt(3) mod 13 should be 4 or 9
        let root = Integer::from(3).sqrt_mod_prime(&p13).unwrap();
        assert_eq!((root.clone() * root) % p13.clone(), Integer::from(3));

        // sqrt(1) mod any prime should be 1
        assert_eq!(Integer::from(1).sqrt_mod_prime(&p7).unwrap(), Integer::from(1));
        assert_eq!(Integer::from(1).sqrt_mod_prime(&p13).unwrap(), Integer::from(1));

        // sqrt(0) mod any prime should be 0
        assert_eq!(Integer::from(0).sqrt_mod_prime(&p7).unwrap(), Integer::from(0));

        // Non-residue should error
        assert!(Integer::from(3).sqrt_mod_prime(&p7).is_err());
    }

    #[test]
    fn test_quadratic_residues() {
        // QRs mod 7: {0, 1, 2, 4}
        let qrs = Integer::quadratic_residues(&Integer::from(7));
        assert_eq!(qrs.len(), 4);
        assert_eq!(qrs[0], Integer::from(0));
        assert_eq!(qrs[1], Integer::from(1));
        assert_eq!(qrs[2], Integer::from(2));
        assert_eq!(qrs[3], Integer::from(4));

        // QRs mod 11: {0, 1, 3, 4, 5, 9}
        let qrs = Integer::quadratic_residues(&Integer::from(11));
        assert_eq!(qrs.len(), 6); // (11-1)/2 + 1 = 6
        assert!(qrs.contains(&Integer::from(0)));
        assert!(qrs.contains(&Integer::from(1)));
        assert!(qrs.contains(&Integer::from(3)));
        assert!(qrs.contains(&Integer::from(4)));
        assert!(qrs.contains(&Integer::from(5)));
        assert!(qrs.contains(&Integer::from(9)));

        // Verify they're actually quadratic residues
        let p = Integer::from(11);
        for qr in &qrs {
            assert!(qr.is_quadratic_residue_prime(&p));
        }
    }
}

//! Integer modulo n arithmetic
//!
//! Provides comprehensive modular arithmetic operations for integers modulo n,
//! where n can be any positive integer (not necessarily prime).
//!
//! This module supports:
//! - Basic arithmetic (addition, subtraction, multiplication)
//! - Modular inverse (when gcd(a, n) = 1)
//! - Square roots modulo prime and prime powers
//! - Lucas sequences for primality testing
//! - Conversions and homomorphisms

use rustmath_core::{CommutativeRing, EuclideanDomain, MathError, Result, Ring};
use rustmath_integers::Integer;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Element of the ring Z/nZ (integers modulo n)
///
/// Represents an integer modulo n, with efficient operations.
/// Unlike PrimeField, this works for any modulus n (not just primes),
/// but division may not always be defined.
///
/// # The modulus-0 sentinel (`Ring::zero`/`Ring::one`)
///
/// The static [`Ring::zero`]/[`Ring::one`] constructors cannot know a modulus,
/// so they return elements with `modulus == 0`. Mathematically this is lawful:
/// Z/0Z ≅ Z, and such an element is the *unreduced integer* awaiting the
/// canonical map Z → Z/nZ. Arithmetic coerces it on contact: an operation
/// between a modulus-0 element and a modulus-n element yields a modulus-n
/// element. Equality remains strict (both value *and* modulus must match), so
/// compare through arithmetic or reduce first. Use [`Zmod`](crate::Zmod) for a
/// parent whose `zero()`/`one()` carry the modulus from the start.
#[derive(Clone, Debug)]
pub struct IntegerMod {
    value: Integer,
    modulus: Integer,
}

impl IntegerMod {
    /// Create a new element in Z/nZ
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value
    /// * `modulus` - The modulus n (must be > 1)
    ///
    /// # Returns
    ///
    /// An IntegerMod element with value reduced modulo n
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_finitefields::IntegerMod;
    /// use rustmath_integers::Integer;
    ///
    /// let a = IntegerMod::new(Integer::from(17), Integer::from(10)).unwrap();
    /// assert_eq!(a.value(), &Integer::from(7)); // 17 ≡ 7 (mod 10)
    /// ```
    pub fn new(value: Integer, modulus: Integer) -> Result<Self> {
        if modulus <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Modulus must be > 1".to_string(),
            ));
        }

        // Reduce value modulo n, ensuring result is in [0, n)
        let (_, reduced) = value.div_rem(&modulus)?;
        let value = if reduced.signum() < 0 {
            reduced + modulus.clone()
        } else {
            reduced
        };

        Ok(IntegerMod { value, modulus })
    }

    /// Internal: build an element with the given modulus. Modulus 0 is the
    /// unreduced-integer sentinel (Z/0Z ≅ Z, see the type docs); any other
    /// modulus accepted here is > 1 by construction.
    fn make(value: Integer, modulus: Integer) -> Self {
        if modulus.is_zero() {
            IntegerMod { value, modulus }
        } else {
            IntegerMod::new(value, modulus)
                .expect("internal invariant: nonzero moduli are always > 1")
        }
    }

    /// Internal: the modulus of the result of a binary operation, coercing the
    /// modulus-0 sentinel to the other operand's modulus. Panics (like the
    /// historical `assert_eq!`) on two different nonzero moduli.
    fn coerced_modulus(&self, other: &Self, op: &str) -> Integer {
        if self.modulus == other.modulus || other.modulus.is_zero() {
            self.modulus.clone()
        } else if self.modulus.is_zero() {
            other.modulus.clone()
        } else {
            panic!("Cannot {op} elements with different moduli");
        }
    }

    /// Get the value (in range [0, modulus))
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Get the modulus
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Compute modular inverse if it exists
    ///
    /// Returns the multiplicative inverse if gcd(value, modulus) = 1.
    /// Otherwise returns an error.
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_finitefields::IntegerMod;
    /// use rustmath_integers::Integer;
    ///
    /// let a = IntegerMod::new(Integer::from(3), Integer::from(7)).unwrap();
    /// let inv = a.inverse().unwrap();
    /// assert_eq!(inv.value(), &Integer::from(5)); // 3 * 5 ≡ 1 (mod 7)
    /// ```
    pub fn inverse(&self) -> Result<Self> {
        if self.value.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        // Use extended GCD to find multiplicative inverse
        let (gcd, x, _) = self.value.extended_gcd(&self.modulus);
        // Normalize the gcd sign: it can be negative for the modulus-0
        // sentinel (negative values never occur with a reduced modulus > 1).
        let (gcd, x) = if gcd.signum() < 0 { (-gcd, -x) } else { (gcd, x) };
        if !gcd.is_one() {
            return Err(MathError::InvalidArgument(format!(
                "No inverse exists: gcd({}, {}) = {}",
                self.value, self.modulus, gcd
            )));
        }

        // x is the inverse, but may be negative - normalize to [0, modulus)
        let inv = if x < Integer::zero() {
            x + self.modulus.clone()
        } else {
            x
        };

        Ok(IntegerMod {
            value: inv,
            modulus: self.modulus.clone(),
        })
    }

    /// Compute modular exponentiation: self^exp (mod modulus)
    ///
    /// Uses binary exponentiation for efficiency.
    pub fn pow(&self, exp: &Integer) -> Result<Self> {
        if exp < &Integer::zero() {
            // Negative exponent requires computing inverse first
            let inv = self.inverse()?;
            return inv.pow(&(-exp.clone()));
        }

        let mut result = IntegerMod::make(Integer::one(), self.modulus.clone());
        let mut base = self.clone();
        let mut e = exp.clone();

        while e > Integer::zero() {
            if e.clone() % Integer::from(2) == Integer::one() {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            e = e / Integer::from(2);
        }

        Ok(result)
    }

    /// Check if this element is a unit (has a multiplicative inverse)
    pub fn is_unit(&self) -> bool {
        let (gcd, _, _) = self.value.extended_gcd(&self.modulus);
        // |gcd| = 1; the gcd sign can be negative for the modulus-0 sentinel.
        gcd.is_one() || (-gcd).is_one()
    }

    /// Compute the multiplicative order of this element
    ///
    /// Returns the smallest k > 0 such that self^k ≡ 1 (mod n),
    /// or None if the element is not a unit.
    pub fn multiplicative_order(&self) -> Option<Integer> {
        if !self.is_unit() {
            return None;
        }

        if self.modulus.is_zero() {
            // Modulus-0 sentinel = element of Z: only ±1 have finite order.
            return if self.value.is_one() {
                Some(Integer::one())
            } else if self.value == -Integer::one() {
                Some(Integer::from(2))
            } else {
                None
            };
        }

        let mut power = self.clone();
        let mut k = Integer::one();
        let one = IntegerMod::new(Integer::one(), self.modulus.clone()).unwrap();

        // Limit iterations to prevent infinite loops
        let max_order = self.modulus.clone();

        while power != one && k < max_order {
            power = power * self.clone();
            k = k + Integer::one();
        }

        if power == one {
            Some(k)
        } else {
            None
        }
    }

    /// Lift to `Z`: the canonical representative in `[0, modulus)`
    /// (MAGMA lifts residue classes to the integers).
    pub fn lift(&self) -> Integer {
        self.value.clone()
    }

    /// Whether the element is nilpotent (MAGMA `IsNilpotent`): true iff every
    /// prime dividing the modulus divides the value. For the modulus-0
    /// sentinel (an element of `Z`), only `0` is nilpotent.
    pub fn is_nilpotent(&self) -> bool {
        if self.value.is_zero() {
            return true;
        }
        if self.modulus.is_zero() {
            // Z is a domain: no nonzero nilpotents.
            return false;
        }
        for (prime, _) in rustmath_integers::prime::factor(&self.modulus) {
            let (_, r) = self
                .value
                .div_rem(&prime)
                .expect("prime factor is nonzero");
            if !r.is_zero() {
                return false;
            }
        }
        true
    }

    /// Whether `self^2 == self` (MAGMA `IsIdempotent`).
    pub fn is_idempotent(&self) -> bool {
        self.clone() * self.clone() == *self
    }

    /// Whether the element is a zero divisor: non-zero and not a unit.
    /// For the modulus-0 sentinel (an element of `Z`, a domain), always false.
    pub fn is_zero_divisor(&self) -> bool {
        if self.modulus.is_zero() {
            return false;
        }
        !self.value.is_zero() && !self.is_unit()
    }

    /// A square root of the element, if one exists (MAGMA `Sqrt`).
    ///
    /// Computed via the factorization of the modulus, prime-power square
    /// roots ([`square_root_mod_prime_power`]), and CRT recombination. For
    /// the modulus-0 sentinel (an element of `Z`), a root exists iff the
    /// value is a perfect square.
    pub fn sqrt(&self) -> Option<Self> {
        if self.modulus.is_zero() {
            // Z: perfect squares only.
            if self.value.signum() < 0 {
                return None;
            }
            let r = self.value.sqrt().ok()?;
            return if r.clone() * r.clone() == self.value {
                Some(IntegerMod::make(r, Integer::zero()))
            } else {
                None
            };
        }
        let mut cur_mod = Integer::one();
        let mut cur_val = Integer::zero();
        for (prime, e) in rustmath_integers::prime::factor(&self.modulus) {
            let pk = prime.pow(e);
            let r = square_root_mod_prime_power(&self.value, &prime, e as usize)?;
            let combined = rustmath_integers::crt_two(&cur_val, &cur_mod, &r, &pk).ok()?;
            cur_val = combined;
            cur_mod = cur_mod * pk;
        }
        Some(IntegerMod::make(cur_val, self.modulus.clone()))
    }
}

impl fmt::Display for IntegerMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.modulus)
    }
}

impl PartialEq for IntegerMod {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.modulus == other.modulus
    }
}

impl Eq for IntegerMod {}

impl Add for IntegerMod {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "add");
        IntegerMod::make(self.value + other.value, modulus)
    }
}

impl Sub for IntegerMod {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "subtract");
        IntegerMod::make(self.value - other.value, modulus)
    }
}

impl Mul for IntegerMod {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "multiply");
        IntegerMod::make(self.value * other.value, modulus)
    }
}

impl Neg for IntegerMod {
    type Output = Self;

    fn neg(self) -> Self {
        if self.modulus.is_zero() {
            // Modulus-0 sentinel: negate in Z.
            IntegerMod {
                value: -self.value,
                modulus: self.modulus,
            }
        } else if self.value.is_zero() {
            self
        } else {
            IntegerMod::new(self.modulus.clone() - self.value, self.modulus).unwrap()
        }
    }
}

impl Ring for IntegerMod {
    /// The additive identity, as the modulus-0 sentinel (see the type docs):
    /// the canonical image of `0 ∈ Z` in any Z/nZ, coerced on first contact
    /// with a modulus-carrying element. Prefer [`Zmod::zero`](crate::Zmod)
    /// when the parent is known.
    fn zero() -> Self {
        IntegerMod {
            value: Integer::zero(),
            modulus: Integer::zero(),
        }
    }

    /// The multiplicative identity, as the modulus-0 sentinel (see
    /// [`Ring::zero`] above and the type docs).
    fn one() -> Self {
        IntegerMod {
            value: Integer::one(),
            modulus: Integer::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }
}

// Multiplication in Z/nZ (and in the modulus-0 sentinel ring Z) commutes.
impl CommutativeRing for IntegerMod {}

/// Compute square root modulo a prime p
///
/// Given a and prime p, find x such that x^2 ≡ a (mod p).
/// Uses the Tonelli-Shanks algorithm for efficiency.
///
/// # Arguments
///
/// * `a` - The value to find the square root of
/// * `p` - A prime number
///
/// # Returns
///
/// A square root x such that x^2 ≡ a (mod p), or None if a is not a quadratic residue.
///
/// # Algorithm
///
/// Uses Tonelli-Shanks algorithm:
/// 1. Check if a is a quadratic residue using Legendre symbol
/// 2. Handle special cases (a = 0, p = 2, p ≡ 3 (mod 4))
/// 3. For general case, use Tonelli-Shanks to find the root
///
/// # Example
///
/// ```
/// use rustmath_finitefields::square_root_mod_prime;
/// use rustmath_integers::Integer;
///
/// // Find sqrt(4) mod 7 = 2 or 5
/// let sqrt = square_root_mod_prime(&Integer::from(4), &Integer::from(7)).unwrap();
/// let check = (sqrt.clone() * sqrt.clone()) % Integer::from(7);
/// assert_eq!(check, Integer::from(4));
/// ```
pub fn square_root_mod_prime(a: &Integer, p: &Integer) -> Option<Integer> {
    // Reduce a modulo p
    let (_, a_reduced) = a.div_rem(p).ok()?;
    let a = if a_reduced.signum() < 0 {
        a_reduced + p.clone()
    } else {
        a_reduced
    };

    // Special case: a = 0
    if a.is_zero() {
        return Some(Integer::zero());
    }

    // Special case: p = 2
    if p == &Integer::from(2) {
        return Some(a);
    }

    // Check if a is a quadratic residue using Legendre symbol
    let legendre = a.legendre_symbol(p).ok()?;
    if legendre != 1 {
        return None; // Not a quadratic residue
    }

    // Special case: p ≡ 3 (mod 4)
    // In this case, x = a^((p+1)/4) is a square root
    if p.clone() % Integer::from(4) == Integer::from(3) {
        let exp = (p.clone() + Integer::one()) / Integer::from(4);
        let x = a.mod_pow(&exp, p).ok()?;
        return Some(x);
    }

    // General case: use Tonelli-Shanks algorithm
    // Write p - 1 = 2^s * q where q is odd
    let mut q = p.clone() - Integer::one();
    let mut s = 0;
    while q.clone() % Integer::from(2) == Integer::zero() {
        q = q / Integer::from(2);
        s += 1;
    }

    // Find a quadratic non-residue n
    let mut n = Integer::from(2);
    while n.legendre_symbol(p).ok()? == 1 {
        n = n + Integer::one();
    }

    // Initialize
    let mut x = a.mod_pow(&((q.clone() + Integer::one()) / Integer::from(2)), p).ok()?;
    let mut b = a.mod_pow(&q, p).ok()?;
    let mut g = n.mod_pow(&q, p).ok()?;
    let mut r = s;

    loop {
        // Find the least m such that b^(2^m) = 1
        let mut t = b.clone();
        let mut m = 0;
        while m < r && !t.is_one() {
            t = (t.clone() * t.clone()) % p.clone();
            m += 1;
        }

        if m == 0 {
            return Some(x);
        }

        // Update values
        let exp = Integer::from(2).pow((r - m - 1).try_into().ok()?);
        let gs = g.mod_pow(&exp, p).ok()?;
        g = (gs.clone() * gs.clone()) % p.clone();
        x = (x * gs) % p.clone();
        b = (b * g.clone()) % p.clone();
        r = m;
    }
}

/// Compute square root modulo a prime power p^k
///
/// Given a and prime p with exponent k, find x such that x^2 ≡ a (mod p^k).
/// Uses Hensel lifting to lift a solution from mod p to mod p^k.
///
/// # Arguments
///
/// * `a` - The value to find the square root of
/// * `p` - A prime number
/// * `k` - The exponent (k ≥ 1)
///
/// # Returns
///
/// A square root x such that x^2 ≡ a (mod p^k), or None if no such root exists.
pub fn square_root_mod_prime_power(a: &Integer, p: &Integer, k: usize) -> Option<Integer> {
    if k == 0 {
        return None;
    }

    if k == 1 {
        return square_root_mod_prime(a, p);
    }

    // Start with solution modulo p
    let mut x = square_root_mod_prime(a, p)?;

    // Hensel lift from p^i to p^(i+1)
    let mut modulus = p.clone();
    for _ in 1..k {
        let next_modulus = modulus.clone() * p.clone();

        // Compute f(x) = x^2 - a
        let fx = (x.clone() * x.clone() - a.clone()) % next_modulus.clone();

        if fx.is_zero() {
            modulus = next_modulus;
            continue;
        }

        // Compute f'(x) = 2x
        let fpx = (Integer::from(2) * x.clone()) % next_modulus.clone();

        // Compute inverse of f'(x) mod p
        let (gcd, inv, _) = fpx.extended_gcd(p);
        if !gcd.is_one() {
            return None; // Cannot lift
        }

        let inv = if inv < Integer::zero() {
            inv + p.clone()
        } else {
            inv
        };

        // Newton update: x_new = x - f(x) / f'(x) mod p^(i+1)
        // Which simplifies to: x_new = x - (fx / modulus) * inv * p
        let correction = ((fx / modulus.clone()) * inv * p.clone()) % next_modulus.clone();
        x = (x - correction + next_modulus.clone()) % next_modulus.clone();

        modulus = next_modulus;
    }

    Some(x)
}

/// Lucas sequence U_n(P, Q) computation
///
/// Computes the Lucas U sequence: U_0 = 0, U_1 = 1, U_n = P*U_{n-1} - Q*U_{n-2}
/// Used in primality testing and other number-theoretic algorithms.
///
/// # Arguments
///
/// * `n` - Index of the sequence
/// * `p` - Parameter P
/// * `q` - Parameter Q
/// * `modulus` - Compute modulo this value
///
/// # Returns
///
/// U_n(P, Q) mod modulus
pub fn lucas(n: &Integer, p: &Integer, q: &Integer, modulus: &Integer) -> Result<Integer> {
    if n.is_zero() {
        return Ok(Integer::zero());
    }
    if n.is_one() {
        return Ok(Integer::one());
    }

    // Use binary method for efficient computation
    let mut u = Integer::one();
    let mut v = p.clone();
    let mut q_k = q.clone();

    // Get binary representation
    let mut temp = n.clone();
    let mut bits_vec = Vec::new();
    while temp > Integer::zero() {
        bits_vec.push(if temp.clone() % Integer::from(2) == Integer::one() { '1' } else { '0' });
        temp = temp / Integer::from(2);
    }
    bits_vec.reverse();
    let bits_chars = bits_vec;

    for i in 1..bits_chars.len() {
        // Double: U_2n = U_n * V_n
        u = (u.clone() * v.clone()) % modulus.clone();

        // V_2n = V_n^2 - 2*Q^n
        v = (v.clone() * v.clone() - Integer::from(2) * q_k.clone()) % modulus.clone();
        if v.signum() < 0 {
            v = v + modulus.clone();
        }

        q_k = (q_k.clone() * q_k.clone()) % modulus.clone();

        if bits_chars[i] == '1' {
            // Add 1: U_{2n+1} = (P*U_2n + V_2n) / 2
            let mut u_next = (p.clone() * u.clone() + v.clone()) % modulus.clone();
            if u_next.clone() % Integer::from(2) != Integer::zero() {
                u_next = u_next + modulus.clone();
            }
            u = (u_next / Integer::from(2)) % modulus.clone();

            // V_{2n+1} = (D*U_2n + P*V_2n) / 2
            let d = p.clone() * p.clone() - Integer::from(4) * q.clone();
            let mut v_next = (d * u.clone() + p.clone() * v.clone()) % modulus.clone();
            if v_next.clone() % Integer::from(2) != Integer::zero() {
                v_next = v_next + modulus.clone();
            }
            v = (v_next / Integer::from(2)) % modulus.clone();

            q_k = (q_k * q.clone()) % modulus.clone();
        }
    }

    Ok(u)
}

/// Lucas sequence V_n(P, 1) computation (special case with Q=1)
///
/// Computes the Lucas V sequence with Q=1: V_0 = 2, V_1 = P, V_n = P*V_{n-1} - V_{n-2}
/// This is the "lucas_q1" function from SageMath, optimized for Q=1 case.
///
/// # Arguments
///
/// * `n` - Index of the sequence
/// * `p` - Parameter P
/// * `modulus` - Compute modulo this value
///
/// # Returns
///
/// V_n(P, 1) mod modulus
pub fn lucas_q1(n: &Integer, p: &Integer, modulus: &Integer) -> Result<Integer> {
    if n.is_zero() {
        return Ok(Integer::from(2));
    }
    if n.is_one() {
        return Ok(p.clone() % modulus.clone());
    }

    let mut v_prev = Integer::from(2);
    let mut v_curr = p.clone() % modulus.clone();

    let mut k = Integer::one();
    while k < *n {
        let v_next = (p.clone() * v_curr.clone() - v_prev) % modulus.clone();
        v_prev = v_curr;
        v_curr = if v_next.signum() < 0 {
            v_next + modulus.clone()
        } else {
            v_next
        };
        k = k + Integer::one();
    }

    Ok(v_curr)
}

/// Check if a value is an IntegerMod element
///
/// This is a type-checking function, useful for generic code.
#[inline]
pub fn is_integer_mod<T: 'static>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<IntegerMod>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = IntegerMod::new(Integer::from(7), Integer::from(10)).unwrap();
        let b = IntegerMod::new(Integer::from(8), Integer::from(10)).unwrap();

        // Addition: 7 + 8 = 15 ≡ 5 (mod 10)
        let sum = a.clone() + b.clone();
        assert_eq!(sum.value(), &Integer::from(5));

        // Multiplication: 7 * 8 = 56 ≡ 6 (mod 10)
        let prod = a.clone() * b.clone();
        assert_eq!(prod.value(), &Integer::from(6));

        // Subtraction: 7 - 8 = -1 ≡ 9 (mod 10)
        let diff = a.clone() - b.clone();
        assert_eq!(diff.value(), &Integer::from(9));

        // Negation: -7 ≡ 3 (mod 10)
        let neg = -a;
        assert_eq!(neg.value(), &Integer::from(3));
    }

    #[test]
    fn test_inverse() {
        let a = IntegerMod::new(Integer::from(3), Integer::from(7)).unwrap();
        let inv = a.inverse().unwrap();

        // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(inv.value(), &Integer::from(5));

        // Verify
        let prod = a * inv;
        assert!(prod.is_one());
    }

    #[test]
    fn test_inverse_no_exist() {
        // gcd(4, 6) = 2, so no inverse exists
        let a = IntegerMod::new(Integer::from(4), Integer::from(6)).unwrap();
        assert!(a.inverse().is_err());
    }

    #[test]
    fn test_pow() {
        let a = IntegerMod::new(Integer::from(2), Integer::from(10)).unwrap();

        // 2^3 = 8
        let result = a.pow(&Integer::from(3)).unwrap();
        assert_eq!(result.value(), &Integer::from(8));

        // 2^4 = 16 ≡ 6 (mod 10)
        let result = a.pow(&Integer::from(4)).unwrap();
        assert_eq!(result.value(), &Integer::from(6));
    }

    #[test]
    fn test_is_unit() {
        // 3 is a unit mod 7 (coprime)
        let a = IntegerMod::new(Integer::from(3), Integer::from(7)).unwrap();
        assert!(a.is_unit());

        // 4 is not a unit mod 6 (gcd = 2)
        let b = IntegerMod::new(Integer::from(4), Integer::from(6)).unwrap();
        assert!(!b.is_unit());
    }

    #[test]
    fn test_multiplicative_order() {
        // Order of 2 mod 7: 2^1=2, 2^2=4, 2^3=1 (order = 3)
        let a = IntegerMod::new(Integer::from(2), Integer::from(7)).unwrap();
        assert_eq!(a.multiplicative_order(), Some(Integer::from(3)));

        // Order of 3 mod 7: 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1 (order = 6)
        let b = IntegerMod::new(Integer::from(3), Integer::from(7)).unwrap();
        assert_eq!(b.multiplicative_order(), Some(Integer::from(6)));
    }

    #[test]
    fn test_square_root_mod_prime() {
        // sqrt(4) mod 7 = 2 or 5
        let sqrt = square_root_mod_prime(&Integer::from(4), &Integer::from(7)).unwrap();
        let check = (sqrt.clone() * sqrt.clone()) % Integer::from(7);
        assert_eq!(check, Integer::from(4));

        // sqrt(3) mod 7 doesn't exist (3 is not a QR mod 7)
        let no_sqrt = square_root_mod_prime(&Integer::from(3), &Integer::from(7));
        assert!(no_sqrt.is_none());

        // sqrt(1) mod 7 = 1 or 6
        let sqrt = square_root_mod_prime(&Integer::from(1), &Integer::from(7)).unwrap();
        let check = (sqrt.clone() * sqrt.clone()) % Integer::from(7);
        assert_eq!(check, Integer::from(1));
    }

    #[test]
    fn test_square_root_mod_prime_special_cases() {
        // p ≡ 3 (mod 4) case
        let sqrt = square_root_mod_prime(&Integer::from(4), &Integer::from(11)).unwrap();
        let check = (sqrt.clone() * sqrt.clone()) % Integer::from(11);
        assert_eq!(check, Integer::from(4));

        // sqrt(0) = 0
        let sqrt = square_root_mod_prime(&Integer::from(0), &Integer::from(7)).unwrap();
        assert_eq!(sqrt, Integer::zero());
    }

    #[test]
    fn test_square_root_mod_prime_power() {
        // sqrt(4) mod 3^2 = 2 or 7
        let sqrt = square_root_mod_prime_power(&Integer::from(4), &Integer::from(3), 2).unwrap();
        let check = (sqrt.clone() * sqrt.clone()) % Integer::from(9);
        assert_eq!(check, Integer::from(4));

        // sqrt(1) mod 5^3 = 1 or 124
        let sqrt = square_root_mod_prime_power(&Integer::from(1), &Integer::from(5), 3).unwrap();
        let check = (sqrt.clone() * sqrt.clone()) % Integer::from(125);
        assert_eq!(check, Integer::from(1));
    }

    #[test]
    fn test_lucas() {
        // Lucas sequence U_n(1, -1) is the Fibonacci sequence
        // U_0 = 0, U_1 = 1, U_2 = 1, U_3 = 2, U_4 = 3, U_5 = 5
        let modulus = Integer::from(1000);

        assert_eq!(
            lucas(&Integer::from(0), &Integer::one(), &(-Integer::one()), &modulus).unwrap(),
            Integer::zero()
        );
        assert_eq!(
            lucas(&Integer::from(1), &Integer::one(), &(-Integer::one()), &modulus).unwrap(),
            Integer::one()
        );
        assert_eq!(
            lucas(&Integer::from(5), &Integer::one(), &(-Integer::one()), &modulus).unwrap(),
            Integer::from(5)
        );
    }

    #[test]
    fn test_lucas_q1() {
        // V_n(2, 1): V_0 = 2, V_1 = 2, V_2 = 2*2 - 2 = 2, ...
        let modulus = Integer::from(1000);

        assert_eq!(lucas_q1(&Integer::from(0), &Integer::from(2), &modulus).unwrap(), Integer::from(2));
        assert_eq!(lucas_q1(&Integer::from(1), &Integer::from(2), &modulus).unwrap(), Integer::from(2));

        // V_n(3, 1): V_0 = 2, V_1 = 3, V_2 = 3*3 - 2 = 7
        assert_eq!(lucas_q1(&Integer::from(0), &Integer::from(3), &modulus).unwrap(), Integer::from(2));
        assert_eq!(lucas_q1(&Integer::from(1), &Integer::from(3), &modulus).unwrap(), Integer::from(3));
        assert_eq!(lucas_q1(&Integer::from(2), &Integer::from(3), &modulus).unwrap(), Integer::from(7));
    }

    #[test]
    fn test_reduction() {
        // Test that negative values are properly reduced
        let a = IntegerMod::new(Integer::from(-3), Integer::from(7)).unwrap();
        assert_eq!(a.value(), &Integer::from(4)); // -3 ≡ 4 (mod 7)

        // Test that values >= modulus are reduced
        let b = IntegerMod::new(Integer::from(10), Integer::from(7)).unwrap();
        assert_eq!(b.value(), &Integer::from(3)); // 10 ≡ 3 (mod 7)
    }

    #[test]
    fn test_equality() {
        let a = IntegerMod::new(Integer::from(3), Integer::from(7)).unwrap();
        let b = IntegerMod::new(Integer::from(10), Integer::from(7)).unwrap();
        let c = IntegerMod::new(Integer::from(3), Integer::from(11)).unwrap();

        assert_eq!(a, b); // 3 ≡ 10 (mod 7)
        assert_ne!(a, c); // Different moduli
    }

    #[test]
    fn test_display() {
        let a = IntegerMod::new(Integer::from(3), Integer::from(7)).unwrap();
        assert_eq!(format!("{}", a), "3 (mod 7)");
    }

    #[test]
    fn test_ring_zero_one_lawful() {
        // Ring::zero()/one() no longer panic: they are modulus-0 sentinels.
        let zero = <IntegerMod as Ring>::zero();
        let one = <IntegerMod as Ring>::one();
        assert!(zero.is_zero());
        assert!(one.is_one());
        assert_eq!(zero.modulus(), &Integer::zero());

        // Sentinels coerce on contact: x + 0 == x, x * 1 == x, in Z/7Z.
        let x = IntegerMod::new(Integer::from(5), Integer::from(7)).unwrap();
        assert_eq!(x.clone() + zero.clone(), x);
        assert_eq!(zero.clone() + x.clone(), x);
        assert_eq!(x.clone() * one.clone(), x);
        assert_eq!(one.clone() * x.clone(), x);
        assert_eq!(x.clone() - zero.clone(), x);

        // Sentinel arithmetic among sentinels stays in Z (= Z/0Z).
        let two = one.clone() + one.clone();
        assert_eq!(two.value(), &Integer::from(2));
        assert_eq!(two.modulus(), &Integer::zero());
        // ... and reduces once it meets a modulus: (1+1) + 6 ≡ 1 (mod 7).
        let y = IntegerMod::new(Integer::from(6), Integer::from(7)).unwrap();
        assert_eq!((two + y).value(), &Integer::from(1));

        // x + (-x) is zero in Z/7Z.
        let sum = x.clone() + (-x.clone());
        assert!(sum.is_zero());
        assert_eq!(sum.modulus(), &Integer::from(7));
    }

    #[test]
    fn test_sentinel_units_in_z() {
        // The modulus-0 sentinel models Z: only ±1 are units.
        let one = <IntegerMod as Ring>::one();
        let minus_one = -one.clone();
        assert!(one.is_unit());
        assert!(minus_one.is_unit());
        assert_eq!(one.multiplicative_order(), Some(Integer::one()));
        assert_eq!(minus_one.multiplicative_order(), Some(Integer::from(2)));
        assert_eq!(minus_one.inverse().unwrap().value(), &Integer::from(-1));

        let two = one.clone() + one.clone();
        assert!(!two.is_unit());
        assert_eq!(two.multiplicative_order(), None);
        assert!(two.inverse().is_err());

        // Powers of a sentinel are computed in Z.
        let eight = two.pow(&Integer::from(3)).unwrap();
        assert_eq!(eight.value(), &Integer::from(8));
        assert_eq!(eight.modulus(), &Integer::zero());
    }

    #[test]
    #[should_panic(expected = "different moduli")]
    fn test_mismatched_nonzero_moduli_still_panic() {
        let a = IntegerMod::new(Integer::from(1), Integer::from(7)).unwrap();
        let b = IntegerMod::new(Integer::from(1), Integer::from(11)).unwrap();
        let _ = a + b;
    }

    #[test]
    fn test_nilpotent_idempotent_zero_divisor() {
        // Z/12Z: 6 is nilpotent (6^2 = 36 = 0 mod 12).
        let m12 = Integer::from(12);
        assert!(IntegerMod::new(Integer::from(6), m12.clone())
            .unwrap()
            .is_nilpotent());
        // 4 mod 12: 2 | 4 but 3 does not divide 4 -> not nilpotent.
        let four = IntegerMod::new(Integer::from(4), m12.clone()).unwrap();
        assert!(!four.is_nilpotent());
        // ... but idempotent: 16 = 4 mod 12.
        assert!(four.is_idempotent());
        // ... and a zero divisor (nonzero, non-unit).
        assert!(four.is_zero_divisor());
        // Z/8Z: 2 nilpotent (2^3 = 8 = 0).
        assert!(IntegerMod::new(Integer::from(2), Integer::from(8))
            .unwrap()
            .is_nilpotent());
        // Units are not zero divisors.
        let five = IntegerMod::new(Integer::from(5), m12).unwrap();
        assert!(!five.is_zero_divisor());
        assert!(!five.is_nilpotent());
    }

    #[test]
    fn test_sentinel_predicates_model_z() {
        // The modulus-0 sentinel models Z (a domain).
        let one = <IntegerMod as Ring>::one();
        let zero = <IntegerMod as Ring>::zero();
        let two = one.clone() + one.clone();
        assert!(zero.is_nilpotent());
        assert!(!two.is_nilpotent());
        assert!(!two.is_zero_divisor());
        assert!(zero.is_idempotent());
        assert!(one.is_idempotent());
        assert!(!two.is_idempotent());
        // sqrt in Z: perfect squares only.
        let four = two.clone() * two.clone();
        assert_eq!(four.sqrt().unwrap().value(), &Integer::from(2));
        assert!(two.sqrt().is_none());
        assert!((-four).sqrt().is_none());
    }

    #[test]
    fn test_sqrt_composite_modulus() {
        // sqrt of 4 mod 15 (= 3 * 5): must square back to 4.
        let a = IntegerMod::new(Integer::from(4), Integer::from(15)).unwrap();
        let s = a.sqrt().unwrap();
        assert_eq!((s.clone() * s).value(), &Integer::from(4));
        // 2 is a non-residue mod 15 (non-residue mod 3): no root.
        let b = IntegerMod::new(Integer::from(2), Integer::from(15)).unwrap();
        assert!(b.sqrt().is_none());
        // Prime-power modulus: sqrt of 2 mod 49 (3^2 = 9, 2 is a QR mod 7: 4^2 = 16 = 2).
        let c = IntegerMod::new(Integer::from(2), Integer::from(49)).unwrap();
        let r = c.sqrt().unwrap();
        assert_eq!((r.clone() * r).value(), &Integer::from(2));
    }
}

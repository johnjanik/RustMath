//! Integer modulo n arithmetic
//!
//! Provides comprehensive modular arithmetic operations for integers modulo n,
//! where n can be any positive integer (not necessarily prime).
//!
//! This module supports:
//! - Basic arithmetic (addition, subtraction, multiplication)
//! - Modular inverse (when gcd(a, n) = 1)
//! - Square roots modulo primes, prime powers (including p = 2 and p | a) and arbitrary m
//!   (CRT); all roots and their number
//! - Lucas sequences for primality testing
//! - Conversions and homomorphisms
//!
//! # Square roots: MAGMA specification and algorithms
//!
//! MAGMA Handbook (digests in `rustmath-lava/docs/`): ch. 18 §18.12.1 `Modsqrt`; ch. 19 §19.5.6
//! `IsSquare`, `Sqrt` / `SquareRoot`, `AllSquareRoots` / `AllSqrts` (worked example H19E4: the 12
//! square roots of 1404 in Z/2340Z); ch. 47 §47.10 `SquareRoot` ("Hensel lifting from
//! residue-field approximation"), here only for Z/p^kZ (there is no p-adic element `sqrt` yet).
//!
//! Algorithms (no bounded search anywhere; every "no root" answer is a proof):
//! - Tonelli–Shanks modulo an odd prime: H. Cohen, *A Course in Computational Algebraic Number
//!   Theory*, GTM 138, Algorithm 1.5.1 ([`square_root_mod_prime`], which delegates to
//!   `rustmath_integers::Integer::sqrt_mod_prime`; a `None` comes only from a Jacobi symbol −1,
//!   ibid. Algorithm 1.4.10).
//! - Reduction a ≡ p^v·u (mod p^k), p ∤ u: for 0 < v < k an odd v means no root (v_p(x^2) is
//!   even); for v = 2e every root is x = p^e·y with y^2 ≡ u (mod p^(k-2e)) (elementary).
//! - Odd p, unit u: p-adic Newton iteration with precision doubling, J. von zur Gathen and
//!   J. Gerhard, *Modern Computer Algebra* \[vzGG99\], ch. 9 (Newton iteration), §9.4.
//! - p = 2, unit u: u is a square mod 2^j iff j = 1, or j = 2 and u ≡ 1 (mod 4), or j ≥ 3 and
//!   u ≡ 1 (mod 8) (K. Ireland and M. Rosen, *A Classical Introduction to Modern Number Theory*,
//!   2nd ed., GTM 84, Prop. 5.1.1); the root is then lifted one bit at a time.
//! - Composite m: Chinese remaindering over the prime-power factors of m (a root exists iff one
//!   exists modulo every prime power).

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
/// element. Equality coerces the same way: a modulus-0 operand is bound into
/// the other operand's modulus before comparing, so
/// `IntegerMod::zero() == IntegerMod::new(0, n)?`; two bound elements are
/// equal iff value *and* modulus match (unchanged). Use [`Zmod`](crate::Zmod)
/// for a parent whose `zero()`/`one()` carry the modulus from the start.
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

    /// A square root of the element, if one exists (MAGMA `Sqrt` / `SquareRoot`, ch. 19 §19.5.6;
    /// `Modsqrt`, ch. 18 §18.12.1).
    ///
    /// `None` means that the element is provably not a square. This holds unconditionally: the
    /// modulus is factored with [`rustmath_integers::prime::factor`] (trial division), so every
    /// prime factor is proven prime, and the per-prime-power decision is complete (see the
    /// module docs).
    ///
    /// # Which root
    ///
    /// MAGMA only promises *a* root. RustMath fixes a deterministic choice:
    /// - for a prime-power modulus p^k, the **least non-negative root**;
    /// - for a composite modulus m = ∏ p_i^k_i, the CRT combination of the least roots modulo
    ///   each p_i^k_i. It lies in [0, m) but is in general **not** the least root mod m: for
    ///   1404 in Z/2340Z it is 1872, while the least root is 78. Use
    ///   [`all_sqrts`](Self::all_sqrts)`()[0]` for the least root.
    ///
    /// For the modulus-0 sentinel (an element of `Z`, see the type docs), a root exists iff the
    /// value is a perfect square, and the non-negative root is returned.
    ///
    /// The modulus is factored by trial division, so a modulus with a large prime factor is
    /// slow; [`square_root_mod_prime_power`] takes the prime directly.
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_finitefields::IntegerMod;
    /// use rustmath_integers::Integer;
    ///
    /// // 9 mod 16 has the roots 3, 5, 11, 13; the least one is returned.
    /// let a = IntegerMod::new(Integer::from(9), Integer::from(16)).unwrap();
    /// assert_eq!(a.sqrt().unwrap().value(), &Integer::from(3));
    /// // 2 is not a square mod 16.
    /// assert!(IntegerMod::new(Integer::from(2), Integer::from(16)).unwrap().sqrt().is_none());
    /// ```
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
            // `None` here proves that there is no root mod p^e, hence none mod m.
            let r = prime_power_sqrts(&self.value, &prime, e).least()?;
            cur_val = rustmath_integers::crt_two(&cur_val, &cur_mod, &r, &pk)
                .expect("distinct prime powers are coprime");
            cur_mod = cur_mod * pk;
        }
        Some(IntegerMod::make(cur_val, self.modulus.clone()))
    }

    /// Whether the element is a square (MAGMA `IsSquare`, ch. 19 §19.5.6). Equivalent to
    /// `self.sqrt().is_some()`, with the same unconditional guarantee; use [`sqrt`](Self::sqrt)
    /// to get a root as well.
    pub fn is_square(&self) -> bool {
        self.sqrt().is_some()
    }

    /// Every square root of the element, ascending by value (MAGMA `AllSquareRoots` /
    /// `AllSqrts`, ch. 19 §19.5.6; worked example H19E4: the 12 roots of 1404 in Z/2340Z).
    ///
    /// The result is empty iff the element is not a square, and has exactly
    /// [`num_sqrts`](Self::num_sqrts) elements. **That number can be astronomically large**
    /// (0 mod 2^64 has 2^32 roots), so check `num_sqrts()` before calling this.
    ///
    /// For the modulus-0 sentinel (an element of `Z`): `[]`, `[0]`, or `[-r, r]` for a perfect
    /// square r^2 > 0.
    ///
    /// # Example
    ///
    /// ```
    /// use rustmath_finitefields::IntegerMod;
    /// use rustmath_integers::Integer;
    ///
    /// let a = IntegerMod::new(Integer::from(4), Integer::from(15)).unwrap();
    /// let roots: Vec<Integer> = a.all_sqrts().iter().map(|r| r.value().clone()).collect();
    /// assert_eq!(roots, [2, 7, 8, 13].map(Integer::from));
    /// ```
    pub fn all_sqrts(&self) -> Vec<Self> {
        if self.modulus.is_zero() {
            return match self.sqrt() {
                None => Vec::new(),
                Some(r) if r.value.is_zero() => vec![r],
                Some(r) => vec![-r.clone(), r],
            };
        }
        let mut cur: Vec<Integer> = vec![Integer::zero()];
        let mut cur_mod = Integer::one();
        for (prime, e) in rustmath_integers::prime::factor(&self.modulus) {
            let pk = prime.pow(e);
            let roots = prime_power_sqrts(&self.value, &prime, e).all();
            if roots.is_empty() {
                return Vec::new();
            }
            // x ≡ r (mod M), x ≡ s (mod p^e):  x = r + M·((s − r)·M^(-1) mod p^e).
            let m_inv = cur_mod
                .mod_inverse(&pk)
                .expect("distinct prime powers are coprime");
            let mut next = Vec::with_capacity(cur.len() * roots.len());
            for r in &cur {
                for s in &roots {
                    let t = ((s.clone() - r.clone()) * m_inv.clone()).modulo(&pk);
                    next.push(r.clone() + cur_mod.clone() * t);
                }
            }
            cur = next;
            cur_mod = cur_mod * pk;
        }
        cur.sort();
        cur.into_iter()
            .map(|v| IntegerMod::make(v, self.modulus.clone()))
            .collect()
    }

    /// The number of square roots of the element, in closed form (no roots are listed).
    ///
    /// For m = ∏ p_i^k_i it is the product of the per-prime-power counts: for a ≡ 0 (mod p^k)
    /// there are p^⌊k/2⌋ roots; for a ≡ p^(2e)·u, p ∤ u, j = k − 2e, there are 2·p^e roots for
    /// odd p and 2^e, 2·2^e or 4·2^e roots for p = 2 and j = 1, 2 or ≥ 3 (when u is a square).
    /// For the modulus-0 sentinel (an element of `Z`): 0, 1 or 2.
    pub fn num_sqrts(&self) -> Integer {
        if self.modulus.is_zero() {
            return Integer::from(self.all_sqrts().len() as u64);
        }
        let mut n = Integer::one();
        for (prime, e) in rustmath_integers::prime::factor(&self.modulus) {
            n = n * prime_power_sqrts(&self.value, &prime, e).count();
        }
        n
    }
}

impl fmt::Display for IntegerMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.modulus)
    }
}

impl PartialEq for IntegerMod {
    /// Coercing (binding) equality, matching the arithmetic's coercion rules
    /// (see the type docs):
    ///
    /// * **bound vs bound**: equal iff same value *and* same modulus
    ///   (unchanged); different moduli are simply unequal — never a panic.
    /// * **unbound (modulus-0 sentinel) vs bound**: the unbound value is
    ///   bound into the other operand's modulus first, so
    ///   `unbound(v) == bound(w mod n)` iff `v ≡ w (mod n)`; in particular
    ///   `IntegerMod::zero() == IntegerMod::new(0, n)?`. Symmetric.
    /// * **unbound vs unbound**: equality in Z.
    ///
    /// Transitivity caveat (the `PartialEq` law): `unbound(0)` equals the
    /// bound zero of *every* modulus while those bound zeros differ pairwise
    /// (`unbound(0) == bound(0 mod 7)`, `unbound(0) == bound(0 mod 5)`, but
    /// `bound(0 mod 7) != bound(0 mod 5)`). Transitivity can fail only in
    /// that cross-modulus corner — a zone whose *arithmetic* already panics.
    fn eq(&self, other: &Self) -> bool {
        match (self.modulus.is_zero(), other.modulus.is_zero()) {
            // Both bound: strict, exactly as before.
            (false, false) => self.modulus == other.modulus && self.value == other.value,
            // Both unbound: compare in Z (= Z/0Z).
            (true, true) => self.value == other.value,
            // One unbound: bind it into the other's modulus, compare there.
            (true, false) => {
                IntegerMod::make(self.value.clone(), other.modulus.clone()).value == other.value
            }
            (false, true) => {
                IntegerMod::make(other.value.clone(), self.modulus.clone()).value == self.value
            }
        }
    }
}

/// `Eq` marker with a documented caveat: equality is reflexive and symmetric,
/// but transitivity can fail in exactly one corner — the modulus-0 sentinel
/// equals the bound zero of every modulus while bound zeros of different
/// moduli stay unequal (`unbound(0) == bound(0 mod 7)`,
/// `unbound(0) == bound(0 mod 5)`, `bound(0 mod 7) != bound(0 mod 5)`).
/// That corner's arithmetic already panics, and `IntegerMod` implements no
/// `Hash`, so no map/set invariant can be broken by it.
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

    // Element predicates on the generic `Ring` path (MAGMA Ch. 19 `IsUnit`, `IsNilpotent`,
    // `Order`). A bound element delegates to the inherent methods (which concrete call
    // sites resolve to first). The modulus-0 sentinel is an integer constant awaiting the
    // map Z -> Z/nZ, so only what holds in every Z/nZ is decided (`crate::ring_hooks`).

    fn unit_status(&self) -> Option<bool> {
        if self.modulus.is_zero() {
            return crate::ring_hooks::sentinel_unit_status(&self.value);
        }
        Some(IntegerMod::is_unit(self))
    }

    fn nilpotent_status(&self) -> Option<bool> {
        if self.modulus.is_zero() {
            return crate::ring_hooks::sentinel_nilpotent_status(&self.value);
        }
        Some(IntegerMod::is_nilpotent(self))
    }

    fn unit_inverse(&self) -> Result<Self> {
        if self.modulus.is_zero() {
            return crate::ring_hooks::sentinel_unit_inverse(&self.value, self);
        }
        if !IntegerMod::is_unit(self) {
            return Err(MathError::NotInvertible);
        }
        IntegerMod::inverse(self)
    }

    /// The order of a unit of Z/nZ by the inherent `multiplicative_order`, a complete
    /// search (the order of a unit is at most phi(n) < n).
    fn multiplicative_order_status(&self) -> rustmath_core::MultiplicativeOrder {
        if self.modulus.is_zero() {
            return crate::ring_hooks::sentinel_order(&self.value);
        }
        if !IntegerMod::is_unit(self) {
            return rustmath_core::MultiplicativeOrder::NotAUnit;
        }
        match IntegerMod::multiplicative_order(self) {
            Some(k) => crate::ring_hooks::order_from_integer(&k),
            None => rustmath_core::MultiplicativeOrder::Undecided,
        }
    }
}

// Multiplication in Z/nZ (and in the modulus-0 sentinel ring Z) commutes.
impl CommutativeRing for IntegerMod {}

/// A square root of `a` modulo a prime p (MAGMA `Modsqrt(n, m)` for a prime m, Handbook ch. 18
/// §18.12.1; `SquareRoot` / `Sqrt` in GF(p), ch. 21 §21.4.7 "Order and Roots").
///
/// Returns `Some(x)` with 0 <= x < p and x^2 ≡ a (mod p), or `None` if no such x exists. Any
/// integer `a` is accepted (it is reduced mod p first). Which of the two roots x, p - x is
/// returned is unspecified; [`square_root_mod_prime_power`]`(a, p, 1)` returns the least one,
/// and [`IntegerMod::sqrt`] handles any modulus (prime or not).
///
/// # Algorithm
///
/// With b = a mod p: if the Jacobi symbol (b/p) is -1 the result is `None`. This proves that b
/// is not a square modulo p, for every odd p, prime or not (H. Cohen, *A Course in
/// Computational Algebraic Number Theory*, GTM 138, Algorithm 1.4.10). Otherwise the root comes
/// from [`Integer::sqrt_mod_prime`]: b^((p+1)/4) for p ≡ 3 (mod 4), Tonelli–Shanks (ibid.
/// Algorithm 1.5.1) otherwise, and the root is squared back before it is returned. For p = 2
/// the root is b. So `None` is always a proof, and a returned root always squares back.
///
/// # Panics
///
/// - if `p` is rejected by [`rustmath_integers::prime::is_prime`], with the message
///   "square_root_mod_prime: p = 15 is not prime": a composite p, p <= 1, or a negative p. A
///   composite modulus is refused rather than answered (the old code returned a false `None`
///   for 4 mod 15); use [`IntegerMod::sqrt`] / [`IntegerMod::all_sqrts`] for it.
/// - if p passes `is_prime` but the root computation proves p composite. `is_prime`
///   (Miller–Rabin to the bases 2, …, 37) is a proof only below ψ12 = 318665857834031151167461
///   (Sorenson–Webster 2017); above it a strong pseudoprime can pass.
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
/// // 3 is not a square mod 7: the Jacobi symbol (3/7) is -1.
/// assert_eq!(square_root_mod_prime(&Integer::from(3), &Integer::from(7)), None);
/// ```
pub fn square_root_mod_prime(a: &Integer, p: &Integer) -> Option<Integer> {
    assert_prime("square_root_mod_prime", p);
    // Reduce first: for p = 2 the Kronecker symbol of an unreduced odd a is (-1)^((a^2-1)/8),
    // e.g. (3/2) = -1, although 3 ≡ 1 is a square mod 2.
    let b = a.modulo(p);
    if b.kronecker_symbol(p) == -1 {
        // The Jacobi symbol -1 proves that b is not a square modulo p.
        return None;
    }
    match b.sqrt_mod_prime(p) {
        Ok(r) => Some(r),
        // `is_prime(p)` held and (b/p) != -1, so every error here is a proof that p is
        // composite (gcd(b, p) > 1, or Tonelli-Shanks / the square-back check failed).
        Err(e) => panic!("square_root_mod_prime: p = {p} is not prime ({e})"),
    }
}

/// Solution set of x^2 ≡ a (mod p^k) for a prime p and k ≥ 1 (see [`prime_power_sqrts`]).
#[derive(Debug)]
enum SqrtKind {
    /// a ≡ 0 (mod p^k): the roots are the multiples of p^⌈k/2⌉, p^⌊k/2⌋ of them.
    Zero,
    /// No root: v_p(a) is odd, or the unit part is not a square mod p^j.
    NoRoot,
    /// a ≡ p^(2e)·u (mod p^k) with p ∤ u and j = k − 2e ≥ 1. `unit_roots` holds every root of
    /// y^2 ≡ u (mod p^j) in [0, p^j), ascending: 2 of them for odd p; 1, 2 or 4 for p = 2 and
    /// j = 1, 2 or ≥ 3. The roots mod p^k are p^e·(s + t·p^j), s in `unit_roots`, 0 ≤ t < p^e.
    Unit {
        e: u32,
        j: u32,
        unit_roots: Vec<Integer>,
    },
}

/// The complete root structure of x^2 ≡ a (mod p^k).
#[derive(Debug)]
struct PrimePowerSqrts {
    p: Integer,
    k: u32,
    kind: SqrtKind,
}

impl PrimePowerSqrts {
    /// The least non-negative root, in [0, p^k); `None` iff there is no root.
    ///
    /// The roots are p^e·(s + t·p^j) with t ≥ 0, so the least one is p^e·min(unit_roots).
    fn least(&self) -> Option<Integer> {
        match &self.kind {
            SqrtKind::Zero => Some(Integer::zero()),
            SqrtKind::NoRoot => None,
            SqrtKind::Unit { e, unit_roots, .. } => Some(self.p.pow(*e) * unit_roots[0].clone()),
        }
    }

    /// The number of roots in [0, p^k).
    fn count(&self) -> Integer {
        match &self.kind {
            SqrtKind::Zero => self.p.pow(self.k / 2),
            SqrtKind::NoRoot => Integer::zero(),
            SqrtKind::Unit { e, unit_roots, .. } => {
                Integer::from(unit_roots.len() as u64) * self.p.pow(*e)
            }
        }
    }

    /// Every root in [0, p^k), ascending. Has [`count`](Self::count) elements.
    fn all(&self) -> Vec<Integer> {
        let mut out = Vec::new();
        match &self.kind {
            SqrtKind::NoRoot => {}
            SqrtKind::Zero => {
                // t·p^⌈k/2⌉ for 0 ≤ t < p^⌊k/2⌋, already ascending.
                let step = self.p.pow(self.k.div_ceil(2));
                let n = self.p.pow(self.k / 2);
                let mut t = Integer::zero();
                while t < n {
                    out.push(t.clone() * step.clone());
                    t = t + Integer::one();
                }
            }
            SqrtKind::Unit { e, j, unit_roots } => {
                let pe = self.p.pow(*e);
                let pj = self.p.pow(*j);
                for s in unit_roots {
                    let mut t = Integer::zero();
                    while t < pe {
                        out.push(pe.clone() * (s.clone() + t.clone() * pj.clone()));
                        t = t + Integer::one();
                    }
                }
                out.sort();
            }
        }
        out
    }
}

/// The root structure of x^2 ≡ a (mod p^k), for a prime `p` (not checked here) and k ≥ 1.
/// Any integer `a` is accepted; it is reduced into [0, p^k) first.
///
/// Decision procedure (complete; see the module docs for the references): with b = a mod p^k,
/// b = 0 gives [`SqrtKind::Zero`]; otherwise v = v_p(b) < k, and an odd v means no root, since
/// v_p(x^2) = v would be even. For v = 2e the unit u = b/p^v is handled by [`unit_sqrts_odd`]
/// or [`unit_sqrts_two`] modulo p^j, j = k − v.
fn prime_power_sqrts(a: &Integer, p: &Integer, k: u32) -> PrimePowerSqrts {
    debug_assert!(k >= 1, "prime_power_sqrts: k must be >= 1");
    let pk = p.pow(k);
    let b = a.modulo(&pk); // in [0, p^k), also for negative a
    let kind = if b.is_zero() {
        SqrtKind::Zero
    } else {
        let v = b.valuation(p); // 0 <= v < k because 0 < b < p^k
        if v % 2 == 1 {
            SqrtKind::NoRoot
        } else {
            let u = b / p.pow(v); // exact; p does not divide u, 0 < u < p^(k-v)
            let j = k - v;
            let unit_roots = if *p == Integer::from(2) {
                unit_sqrts_two(&u, j)
            } else {
                unit_sqrts_odd(&u, p, j)
            };
            if unit_roots.is_empty() {
                SqrtKind::NoRoot
            } else {
                // Always-on safety net (one multiplication): a returned root must square back.
                // For a prime p this cannot fail; a failure means p is not prime.
                let y = &unit_roots[0];
                assert!(
                    (y.clone() * y.clone() - u.clone())
                        .modulo(&p.pow(j))
                        .is_zero(),
                    "square root mod {p}^{j}: internal check failed, {y}^2 is not {u} \
                     (p must be prime)"
                );
                SqrtKind::Unit {
                    e: v / 2,
                    j,
                    unit_roots,
                }
            }
        }
    };
    PrimePowerSqrts {
        p: p.clone(),
        k,
        kind,
    }
}

/// Every root of y^2 ≡ u (mod p^j) for an odd prime p, p ∤ u and j ≥ 1, ascending.
///
/// Existence: a root exists iff (u/p) = 1 (Ireland–Rosen Prop. 5.1.1), so the result is empty
/// iff (u/p) = −1. Count: otherwise there are exactly two roots, y and p^j − y. The two roots
/// ±y_0 mod p are distinct and simple (the derivative 2y_0 is a unit), so each lifts to exactly
/// one root mod p^j (uniqueness of Hensel lifting, \[vzGG99\] §9.4), and every root mod p^j
/// reduces to one of them. Equivalently, (Z/p^jZ)^* is cyclic of even order (Ireland–Rosen
/// ch. 4 §1), so y^2 ≡ 1 has only the solutions ±1.
///
/// A root mod p from Tonelli–Shanks is lifted by p-adic Newton iteration with precision
/// doubling (\[vzGG99\] ch. 9 §9.4): with prec ← min(2·prec, j) and m = p^prec,
/// y ← y − (y^2 − u)·(2y)^(-1) mod m. Here 2y is a unit (p odd, p ∤ y), and if y^2 ≡ u mod
/// p^prec_old then the new y satisfies y^2 ≡ u mod p^(2·prec_old), hence mod m.
fn unit_sqrts_odd(u: &Integer, p: &Integer, j: u32) -> Vec<Integer> {
    let Some(mut y) = square_root_mod_prime(u, p) else {
        return Vec::new();
    };
    let mut prec = 1u32;
    while prec < j {
        prec = prec.saturating_mul(2).min(j);
        let m = p.pow(prec);
        let inv = (Integer::from(2) * y.clone())
            .mod_inverse(&m)
            .expect("2y is a unit mod p^prec: p is odd and does not divide y");
        let fy = y.clone() * y.clone() - u.clone();
        y = (y - fy * inv).modulo(&m);
    }
    let other = p.pow(j) - y.clone();
    let mut roots = vec![y, other];
    roots.sort();
    roots
}

/// Every root of y^2 ≡ u (mod 2^j) for odd u and j ≥ 1, ascending. Empty iff u is not a square
/// mod 2^j (Ireland–Rosen Prop. 5.1.1: for j = 2 the condition is u ≡ 1 (mod 4), for j ≥ 3 it
/// is u ≡ 1 (mod 8)).
///
/// For j ≥ 3 the lift starts at y = 1 and, for i = 3, …, j − 1, replaces y by y + 2^(i−1) when
/// y^2 ≢ u (mod 2^(i+1)): (y + 2^(i−1))^2 = y^2 + 2^i·y + 2^(2i−2) and 2i − 2 ≥ i + 1, so the
/// step flips bit i of y^2 (y is odd) and keeps the lower bits. The result has 0 < y < 2^(j−1),
/// and the four roots are y, 2^(j−1) − y, y + 2^(j−1) and 2^j − y.
fn unit_sqrts_two(u: &Integer, j: u32) -> Vec<Integer> {
    let two = Integer::from(2);
    match j {
        1 => return vec![Integer::one()],
        2 => {
            return if u.modulo(&Integer::from(4)).is_one() {
                vec![Integer::one(), Integer::from(3)]
            } else {
                Vec::new()
            };
        }
        _ => {}
    }
    if !u.modulo(&Integer::from(8)).is_one() {
        return Vec::new();
    }
    let mut y = Integer::one(); // y^2 ≡ u (mod 8)
    for i in 3..j {
        // Invariant: y odd, 0 < y < 2^(i-1), y^2 ≡ u (mod 2^i).
        if !(y.clone() * y.clone() - u.clone())
            .modulo(&two.pow(i + 1))
            .is_zero()
        {
            y = y + two.pow(i - 1);
        }
    }
    let m = two.pow(j);
    let h = two.pow(j - 1);
    let mut roots = vec![y.clone(), m - y.clone(), y.clone() + h.clone(), h - y];
    roots.sort();
    roots
}

/// Panic with "{name}: p = {p} is not prime" unless `p` passes
/// [`rustmath_integers::prime::is_prime`]. Shared by the public square-root functions that take
/// a prime (see their `# Panics`).
fn assert_prime(name: &str, p: &Integer) {
    assert!(
        rustmath_integers::prime::is_prime(p),
        "{name}: p = {p} is not prime"
    );
}

/// Check the prime of the public prime-power functions and convert k (see their `# Panics`).
fn checked_prime_power_args(name: &str, p: &Integer, k: usize) -> u32 {
    assert_prime(name, p);
    u32::try_from(k).unwrap_or_else(|_| panic!("{name}: k = {k} exceeds u32::MAX"))
}

/// The least non-negative square root of `a` modulo p^k (MAGMA `Modsqrt`, ch. 18 §18.12.1, for
/// a prime-power modulus; ch. 47 §47.10 `SquareRoot` on Z/p^kZ).
///
/// Returns the least x in [0, p^k) with x^2 ≡ a (mod p^k), or `None` if there is no such x.
/// Any integer `a` is accepted (negative or unreduced values are reduced mod p^k first). For
/// k = 0 the ring is Z/1Z = {0} and the result is `Some(0)`. Every root is available from
/// [`all_square_roots_mod_prime_power`].
///
/// # Algorithm
///
/// Complete, with no search (references in the [`integer_mod`](crate::integer_mod) module
/// docs). With b = a mod p^k: b = 0 gives 0. Otherwise v = v_p(b) < k; odd v means no root.
/// For v = 2e the roots are p^e·y with y^2 ≡ u = b/p^v (mod p^(k−v)):
/// - odd p: Tonelli–Shanks mod p (none iff (u/p) = −1), then p-adic Newton lifting;
/// - p = 2: the classical criterion (u ≡ 1 mod 4 for k − v = 2, u ≡ 1 mod 8 for k − v ≥ 3) and
///   a bit-by-bit lift.
///
/// The least root is p^e times the least root of the unit part.
///
/// # Contract: p must be prime
///
/// `p` is checked with [`rustmath_integers::prime::is_prime`], which is a proof for
/// p < ψ12 = 318665857834031151167461 (Miller–Rabin to the bases 2, …, 37; Sorenson–Webster
/// 2017) and a probable-prime test above it. For a prime p, `None` proves that there is no root.
/// A composite p that passes the test is outside the contract and the result is unspecified: it
/// can be a false `None`, or a panic from [`square_root_mod_prime`] or from the internal
/// square-back check (a returned root always squares back). [`IntegerMod::sqrt`] is unaffected: it gets its primes from trial division.
///
/// # Panics
///
/// - if `p` is rejected by [`rustmath_integers::prime::is_prime`] ("p = ... is not prime");
/// - if `k > u32::MAX` (p^k would not fit in memory anyway).
///
/// # Example
///
/// ```
/// use rustmath_finitefields::square_root_mod_prime_power;
/// use rustmath_integers::Integer;
///
/// // 9 mod 2^4: the roots are 3, 5, 11, 13.
/// let r = square_root_mod_prime_power(&Integer::from(9), &Integer::from(2), 4);
/// assert_eq!(r, Some(Integer::from(3)));
/// // 2 has odd 2-adic valuation: no root mod 2^3.
/// assert_eq!(square_root_mod_prime_power(&Integer::from(2), &Integer::from(2), 3), None);
/// ```
pub fn square_root_mod_prime_power(a: &Integer, p: &Integer, k: usize) -> Option<Integer> {
    let k = checked_prime_power_args("square_root_mod_prime_power", p, k);
    if k == 0 {
        return Some(Integer::zero()); // Z/1Z: the residue 0 is a root
    }
    prime_power_sqrts(a, p, k).least()
}

/// Every square root of `a` modulo p^k, ascending (MAGMA `AllSquareRoots` / `AllSqrts`,
/// ch. 19 §19.5.6, for a prime-power modulus).
///
/// The result is empty iff there is no root; its first element is
/// [`square_root_mod_prime_power`]`(a, p, k)`. For k = 0 it is `[0]`. Any integer `a` is
/// accepted (reduced mod p^k first).
///
/// **The list can be astronomically long**: 0 mod 2^64 has 2^32 roots, and p^2·13 mod p^3 has
/// 2p roots. For a ≡ 0 there are p^⌊k/2⌋ roots; for a ≡ p^(2e)·u, p ∤ u, j = k − 2e, there are
/// 2·p^e (odd p) or 2^e, 2·2^e, 4·2^e (p = 2, j = 1, 2, ≥ 3) when u is a square.
/// [`IntegerMod::num_sqrts`] gives the count without listing.
///
/// # Contract and panics
///
/// As for [`square_root_mod_prime_power`]: `p` is checked with
/// [`rustmath_integers::prime::is_prime`] (a proof for p < ψ12 = 318665857834031151167461, a
/// probable-prime test above it). A composite p that passes is outside the contract and the
/// result is unspecified (it can be an incomplete list, or a panic). Panics if `p` is rejected
/// by `is_prime` or if `k > u32::MAX`.
pub fn all_square_roots_mod_prime_power(a: &Integer, p: &Integer, k: usize) -> Vec<Integer> {
    let k = checked_prime_power_args("all_square_roots_mod_prime_power", p, k);
    if k == 0 {
        return vec![Integer::zero()];
    }
    prime_power_sqrts(a, p, k).all()
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

    /// Root counts modulo P^3 for the prime P = 1000000000000000177 (gp: `isprime(P)` = 1,
    /// P ≡ 17 mod 32). `IntegerMod::num_sqrts` cannot reach these (trial division of P^3 is
    /// infeasible), so the private core is tested directly. Reference (PARI/GP 2.15.4, the
    /// `chk(a, P, 3)` structure check defined in tests/square_roots.rs): existence from
    /// `issquare(Mod(a, P^3))`; #roots = 2·P^e for a = P^(2e)·u, and P^floor(3/2) = P for
    /// a = 0; the least roots square back and agree up to sign with the p-adic
    /// `sqrt(13 + O(P^3))` and `P*sqrt(13 + O(P))`.
    #[test]
    fn test_prime_power_sqrts_counts_mod_large_prime_cube() {
        let p = Integer::from_decimal_str("1000000000000000177").unwrap();
        let p2 = p.clone() * p.clone();
        let cases: Vec<(Integer, Integer, Option<Integer>)> = vec![
            (
                Integer::from(13),
                Integer::from(2),
                Integer::from_decimal_str("265840322487973224909748129798363644736247129518228023"),
            ),
            (
                p2.clone() * Integer::from(13),
                Integer::from(2) * p.clone(),
                Integer::from_decimal_str("222291380919552034345574422760703115"),
            ),
            (Integer::from(5), Integer::zero(), None),
            (p.clone() * Integer::from(5), Integer::zero(), None),
            (p2 * Integer::from(5), Integer::zero(), None),
            (Integer::zero(), p.clone(), Some(Integer::zero())),
        ];
        for (a, count, least) in cases {
            let s = prime_power_sqrts(&a, &p, 3);
            assert_eq!(s.count(), count, "count of roots of {a} mod P^3");
            assert_eq!(s.least(), least, "least root of {a} mod P^3");
        }
    }
}

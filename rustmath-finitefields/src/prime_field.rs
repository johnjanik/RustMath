//! Prime finite fields GF(p)

use rustmath_core::{CommutativeRing, EuclideanDomain, Field, MathError, NumericConversion, Result, Ring};
use rustmath_integers::Integer;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Element of a prime finite field GF(p)
///
/// Represents integers modulo a prime p
///
/// # The modulus-0 sentinel (`Ring::zero`/`Ring::one`)
///
/// The static [`Ring::zero`]/[`Ring::one`] constructors cannot know a modulus,
/// so — following the same convention as [`crate::IntegerMod`] — they return
/// elements with `modulus == 0`. Mathematically this is lawful: Z/0Z ≅ Z, and
/// such an element is the *unreduced integer* awaiting the canonical map
/// Z → GF(p). The precise algebra of sentinel elements:
///
/// * **binary ops** (`+`, `-`, `*`, `/`): a modulus-0 operand is coerced on
///   contact — the result carries the other operand's modulus, with the
///   sentinel's value reduced mod p (so `PrimeField::zero() + x == x` and
///   `PrimeField::zero() * x` is the zero of `x`'s field). Two modulus-0
///   operands combine in Z (still modulus 0). Two *different nonzero* moduli
///   panic, as before.
/// * **`Neg`**: negates in Z for the sentinel.
/// * **`inverse`/`Div`**: in Z only `±1` are invertible; anything else is an
///   error (never a wrong value).
/// * **`==`** coerces exactly like the arithmetic: a modulus-0 operand is
///   bound into the other operand's modulus before comparing, so
///   `PrimeField::zero() == PrimeField::new(0, p)?` and, in general,
///   `unbound(v) == bound(w mod p)` iff `v ≡ w (mod p)`. Two unbound
///   elements compare in Z; two bound elements are equal iff value *and*
///   modulus match (cross-modulus stays `false`, as before).
/// * operations that *need* a modulus and cannot get one
///   ([`Self::legendre_symbol`], [`Self::discrete_log`], ...) return an error
///   or panic with a precise message rather than fabricate an answer.
#[derive(Clone, Debug)]
pub struct PrimeField {
    value: Integer,
    modulus: Integer,
}

impl PrimeField {
    /// Create a new element in GF(p)
    ///
    /// Requires p to be prime (not checked here for performance)
    pub fn new(value: Integer, modulus: Integer) -> Result<Self> {
        if modulus <= Integer::one() {
            return Err(MathError::InvalidArgument(
                "Modulus must be > 1".to_string(),
            ));
        }

        // Reduce value modulo p
        let (_, reduced) = value.div_rem(&modulus)?;
        let value = if reduced.signum() < 0 {
            reduced + modulus.clone()
        } else {
            reduced
        };

        Ok(PrimeField { value, modulus })
    }

    /// Internal: build an element with the given modulus. Modulus 0 is the
    /// unreduced-integer sentinel (Z/0Z ≅ Z, see the type docs); any other
    /// modulus accepted here is > 1 by construction.
    fn make(value: Integer, modulus: Integer) -> Self {
        if modulus.is_zero() {
            PrimeField { value, modulus }
        } else {
            PrimeField::new(value, modulus)
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

    /// Get the value
    pub fn value(&self) -> &Integer {
        &self.value
    }

    /// Get the modulus (characteristic of the field)
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Compute the multiplicative order of this element
    ///
    /// Returns the smallest k > 0 such that self^k = 1
    pub fn multiplicative_order(&self) -> Option<Integer> {
        if self.value.is_zero() {
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

        let one = PrimeField::new(Integer::one(), self.modulus.clone()).unwrap();

        while power != one {
            power = power * self.clone();
            k = k + Integer::one();

            // Safety check to prevent infinite loops
            if k > self.modulus.clone() {
                return None;
            }
        }

        Some(k)
    }

    /// Check if this is a generator (primitive element) of the multiplicative group
    ///
    /// An element g is a generator if its order equals p-1
    pub fn is_generator(&self) -> bool {
        if let Some(order) = self.multiplicative_order() {
            order == self.modulus.clone() - Integer::one()
        } else {
            false
        }
    }

    /// Compute the Legendre symbol (a/p)
    ///
    /// Returns 0 if a ≡ 0 (mod p), 1 if a is a quadratic residue, -1 otherwise.
    /// Errors for the modulus-0 sentinel (the symbol needs a concrete prime).
    pub fn legendre_symbol(&self) -> Result<Integer> {
        if self.modulus.is_zero() {
            return Err(MathError::InvalidArgument(
                "Legendre symbol needs a concrete modulus; this is an unbound (modulus-0) element"
                    .to_string(),
            ));
        }
        let symbol = self.value.legendre_symbol(&self.modulus)?;
        Ok(Integer::from(symbol as i32))
    }

    /// Check if this element is a quadratic residue (perfect square in the field).
    /// For the modulus-0 sentinel (an element of Z) this means: a perfect square.
    pub fn is_quadratic_residue(&self) -> bool {
        if self.value.is_zero() {
            return true;
        }

        if self.modulus.is_zero() {
            // Z: perfect squares only.
            if self.value.signum() < 0 {
                return false;
            }
            return match self.value.sqrt() {
                Ok(r) => r.clone() * r == self.value,
                Err(_) => false,
            };
        }

        let leg = self.legendre_symbol().unwrap_or(Integer::zero());
        leg == Integer::one()
    }

    /// Compute discrete logarithm: given g^x = h, find x
    ///
    /// Uses the baby-step giant-step algorithm, which runs in O(√p) time and space.
    ///
    /// # Arguments
    ///
    /// * `base` - The base g (should be a generator for guaranteed solution)
    /// * `target` - The target value h
    ///
    /// # Returns
    ///
    /// The discrete logarithm x such that base^x = target, if it exists.
    ///
    /// # Algorithm
    ///
    /// Baby-step giant-step:
    /// 1. Compute m = ceil(sqrt(p-1))
    /// 2. Baby steps: Store g^j for j = 0, 1, ..., m-1
    /// 3. Giant steps: Compute h * g^(-im) for i = 0, 1, 2, ... until match found
    pub fn discrete_log(base: &PrimeField, target: &PrimeField) -> Result<Integer> {
        assert_eq!(base.modulus, target.modulus, "Moduli must match");
        if base.modulus.is_zero() {
            return Err(MathError::InvalidArgument(
                "discrete_log needs a concrete modulus; these are unbound (modulus-0) elements"
                    .to_string(),
            ));
        }

        // Handle special cases
        if base.value.is_zero() || base.value.is_one() {
            return Err(MathError::InvalidArgument(
                "Base must be neither 0 nor 1".to_string(),
            ));
        }

        if target.value.is_one() {
            return Ok(Integer::zero());
        }

        if target.value.is_zero() {
            return Err(MathError::InvalidArgument(
                "Logarithm of zero does not exist".to_string(),
            ));
        }

        // Order of the multiplicative group is p - 1
        let group_order = base.modulus.clone() - Integer::one();

        // Compute m = ceil(sqrt(p-1))
        let m = group_order.sqrt()? + Integer::one();
        let m_usize = m.to_usize().ok_or_else(|| {
            MathError::NumericalError("Group order too large for discrete log".to_string())
        })?;

        // Baby steps: compute and store g^j for j = 0, 1, ..., m-1
        use std::collections::HashMap;
        let mut baby_steps: HashMap<Integer, usize> = HashMap::new();

        let mut power = PrimeField::new(Integer::one(), base.modulus.clone())?;
        for j in 0..m_usize {
            baby_steps.insert(power.value.clone(), j);
            power = power * base.clone();
        }

        // Compute g^(-m) for giant steps
        let base_power_m = {
            let mut result = PrimeField::new(Integer::one(), base.modulus.clone())?;
            let mut b = base.clone();
            let mut exp = m.clone();

            // Fast exponentiation
            while exp > Integer::zero() {
                if exp.clone() % Integer::from(2) == Integer::one() {
                    result = result * b.clone();
                }
                b = b.clone() * b.clone();
                exp = exp / Integer::from(2);
            }
            result
        };

        let giant_step_multiplier = base_power_m.inverse()?;

        // Giant steps: compute h * g^(-im) for i = 0, 1, 2, ...
        let mut gamma = target.clone();
        for i in 0..m_usize {
            if let Some(&j) = baby_steps.get(&gamma.value) {
                // Found! h = g^(im + j)
                let result = Integer::from(i as i64) * m.clone() + Integer::from(j as i64);
                return Ok(result % group_order);
            }

            gamma = gamma * giant_step_multiplier.clone();
        }

        Err(MathError::InvalidArgument(
            "Discrete logarithm not found (target may not be in the subgroup generated by base)"
                .to_string(),
        ))
    }
}

impl fmt::Display for PrimeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.modulus)
    }
}

impl PartialEq for PrimeField {
    /// Coercing (binding) equality, matching the arithmetic's coercion rules
    /// (see the type docs):
    ///
    /// * **bound vs bound**: equal iff same value *and* same modulus
    ///   (unchanged); different moduli are simply unequal — never a panic.
    /// * **unbound (modulus-0 sentinel) vs bound**: the unbound value is
    ///   bound into the other operand's modulus first, so
    ///   `unbound(v) == bound(w mod p)` iff `v ≡ w (mod p)`; in particular
    ///   `PrimeField::zero() == PrimeField::new(0, p)?`. Symmetric.
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
                PrimeField::make(self.value.clone(), other.modulus.clone()).value == other.value
            }
            (false, true) => {
                PrimeField::make(other.value.clone(), self.modulus.clone()).value == self.value
            }
        }
    }
}

impl Add for PrimeField {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "add");
        PrimeField::make(self.value + other.value, modulus)
    }
}

impl Sub for PrimeField {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "subtract");
        PrimeField::make(self.value - other.value, modulus)
    }
}

impl Mul for PrimeField {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "multiply");
        PrimeField::make(self.value * other.value, modulus)
    }
}

impl Div for PrimeField {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let modulus = self.coerced_modulus(&other, "divide");
        // Coerce the divisor first so its inverse is taken in the right ring
        // (in Z only ±1 are invertible; in GF(p) everything nonzero is).
        let other = PrimeField::make(other.value, modulus.clone());
        let inv = other.inverse().unwrap();
        PrimeField::make(self.value, modulus) * inv
    }
}

impl Neg for PrimeField {
    type Output = Self;

    fn neg(self) -> Self {
        if self.modulus.is_zero() {
            // Modulus-0 sentinel: negate in Z.
            PrimeField {
                value: -self.value,
                modulus: self.modulus,
            }
        } else {
            let neg_val = self.modulus.clone() - self.value;
            PrimeField::new(neg_val, self.modulus).unwrap()
        }
    }
}

impl Ring for PrimeField {
    /// The additive identity, as the modulus-0 sentinel (see the type docs):
    /// the canonical image of `0 ∈ Z` in any GF(p), coerced on first contact
    /// with a modulus-carrying element.
    fn zero() -> Self {
        PrimeField {
            value: Integer::zero(),
            modulus: Integer::zero(),
        }
    }

    /// The multiplicative identity, as the modulus-0 sentinel (see
    /// [`Ring::zero`] above and the type docs).
    fn one() -> Self {
        PrimeField {
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

impl CommutativeRing for PrimeField {
    // Marker trait, no methods to implement
}

impl Field for PrimeField {
    fn inverse(&self) -> Result<Self> {
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

        Ok(PrimeField {
            value: inv,
            modulus: self.modulus.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let p = Integer::from(7);

        let a = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(5), p.clone()).unwrap();

        // Addition: 3 + 5 = 8 ≡ 1 (mod 7)
        let sum = a.clone() + b.clone();
        assert_eq!(sum.value(), &Integer::from(1));

        // Multiplication: 3 * 5 = 15 ≡ 1 (mod 7)
        let prod = a.clone() * b.clone();
        assert_eq!(prod.value(), &Integer::from(1));

        // Subtraction: 3 - 5 = -2 ≡ 5 (mod 7)
        let diff = a.clone() - b.clone();
        assert_eq!(diff.value(), &Integer::from(5));
    }

    #[test]
    fn test_inverse() {
        let p = Integer::from(7);
        let a = PrimeField::new(Integer::from(3), p.clone()).unwrap();

        let inv = a.inverse().unwrap();
        // 3 * 5 = 15 ≡ 1 (mod 7), so inverse of 3 is 5
        assert_eq!(inv.value(), &Integer::from(5));

        // Verify: a * a^(-1) = 1
        let prod = a * inv;
        assert!(prod.is_one());
    }

    #[test]
    fn test_division() {
        let p = Integer::from(7);
        let a = PrimeField::new(Integer::from(6), p.clone()).unwrap();
        let b = PrimeField::new(Integer::from(2), p.clone()).unwrap();

        // 6 / 2 = 3 (mod 7)
        let quot = a / b;
        assert_eq!(quot.value(), &Integer::from(3));
    }

    #[test]
    fn test_multiplicative_order() {
        let p = Integer::from(7);

        // Order of 2 in GF(7)
        let a = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let order = a.multiplicative_order().unwrap();

        // 2^1 = 2, 2^2 = 4, 2^3 = 1 (mod 7)
        assert_eq!(order, Integer::from(3));
    }

    #[test]
    fn test_generator() {
        let p = Integer::from(7);

        // 3 is a generator of GF(7)*
        let g = PrimeField::new(Integer::from(3), p.clone()).unwrap();
        assert!(g.is_generator());

        // 2 is not a generator
        let not_g = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        assert!(!not_g.is_generator());
    }

    #[test]
    fn test_discrete_log() {
        let p = Integer::from(11);
        let g = PrimeField::new(Integer::from(2), p.clone()).unwrap(); // 2 is a generator of GF(11)*

        // Test: 2^7 = 128 ≡ 7 (mod 11)
        let target = PrimeField::new(Integer::from(7), p.clone()).unwrap();
        let log = PrimeField::discrete_log(&g, &target).unwrap();

        // Verify: g^log = target
        let mut verification = PrimeField::new(Integer::one(), p.clone()).unwrap();
        let mut temp_g = g.clone();
        let mut exp = log.clone();

        while exp > Integer::zero() {
            if exp.clone() % Integer::from(2) == Integer::one() {
                verification = verification * temp_g.clone();
            }
            temp_g = temp_g.clone() * temp_g.clone();
            exp = exp / Integer::from(2);
        }

        assert_eq!(verification.value(), target.value());
    }

    // ---- modulus-0 sentinel (Ring::zero()/one()) gates -------------------

    /// The generic-code gate that used to be impossible: seed a fold with
    /// `F::zero()` over any Ring, instantiate at PrimeField.
    fn generic_sum<F: Ring>(v: &[F]) -> F {
        v.iter().fold(F::zero(), |acc, x| acc + x.clone())
    }

    fn generic_dot<F: Ring>(a: &[F], b: &[F]) -> F {
        a.iter()
            .zip(b.iter())
            .fold(F::zero(), |acc, (x, y)| acc + x.clone() * y.clone())
    }

    fn gf7(v: i64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(7)).unwrap()
    }

    #[test]
    fn test_generic_sum_and_dot_over_gf7() {
        // sum(3, 5, 6) = 14 = 0 (mod 7)
        let v = vec![gf7(3), gf7(5), gf7(6)];
        let s = generic_sum(&v);
        assert!(s.is_zero());
        assert_eq!(s.modulus(), &Integer::from(7)); // bound on first contact
        // dot([1,2,3],[4,5,6]) = 4 + 10 + 18 = 32 = 4 (mod 7)
        let a = vec![gf7(1), gf7(2), gf7(3)];
        let b = vec![gf7(4), gf7(5), gf7(6)];
        assert_eq!(generic_dot(&a, &b), gf7(4));
        // empty sum: stays the unbound sentinel, but is honestly zero
        let empty: Vec<PrimeField> = vec![];
        assert!(generic_sum(&empty).is_zero());
    }

    #[test]
    fn test_sentinel_algebra() {
        // unbound zero + bound x = x
        assert_eq!(PrimeField::zero() + gf7(3), gf7(3));
        assert_eq!(gf7(3) + PrimeField::zero(), gf7(3));
        // unbound zero * bound x = bound zero of x's field
        let z = PrimeField::zero() * gf7(3);
        assert!(z.is_zero());
        assert_eq!(z.modulus(), &Integer::from(7));
        // unbound one * x = x
        assert_eq!(PrimeField::one() * gf7(5), gf7(5));
        // sentinel values reduce on contact: (1+1+1+1+1+1+1+1) bound = 1 mod 7
        let mut eight = PrimeField::zero();
        for _ in 0..8 {
            eight = eight + PrimeField::one();
        }
        assert_eq!(eight.value(), &Integer::from(8)); // still in Z
        assert_eq!(eight * gf7(1), gf7(1)); // 8 = 1 (mod 7)
        // Neg in Z; -1 is its own inverse in Z
        let minus_one = -PrimeField::one();
        assert_eq!(minus_one.value(), &Integer::from(-1));
        assert_eq!(minus_one.inverse().unwrap().value(), &Integer::from(-1));
        // unbound 2 is not invertible in Z (error, never a wrong value)
        let two = PrimeField::one() + PrimeField::one();
        assert!(two.inverse().is_err());
        // ... but bound into GF(7) it is: 2 * 4 = 8 = 1
        assert_eq!(gf7(1) / (gf7(1) * two), gf7(4));
        // == binds on compare, like the arithmetic: the sentinel zero equals
        // every bound zero (and agrees with is_zero).
        assert_eq!(PrimeField::zero(), gf7(0));
        assert_eq!(gf7(0), PrimeField::zero());
        assert_ne!(PrimeField::zero(), gf7(3));
        assert!(PrimeField::zero().is_zero());
        assert!(PrimeField::one().is_one());
        // modulus-dependent queries refuse the sentinel honestly
        assert!(PrimeField::one().legendre_symbol().is_err());
        assert_eq!((PrimeField::one() + PrimeField::one()).multiplicative_order(), None);
        assert_eq!(PrimeField::one().multiplicative_order(), Some(Integer::one()));
    }

    /// Tracker gate B-01: Matrix<PrimeField> now works with generic Field
    /// code (rank via Gaussian elimination seeds with F::zero()/F::one()).
    #[test]
    fn test_matrix_rank_over_gf7() {
        use rustmath_matrix::Matrix;
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                gf7(1), gf7(2), gf7(3),
                gf7(2), gf7(4), gf7(6), // 2 * row 1
                gf7(1), gf7(0), gf7(1),
            ],
        )
        .unwrap();
        assert_eq!(m.rank().unwrap(), 2);
        // diag(1, 1, 7): rank 3 over Q but 7 = 0 in GF(7) => rank 2.
        let d = Matrix::from_vec(
            3,
            3,
            vec![
                gf7(1), gf7(0), gf7(0),
                gf7(0), gf7(1), gf7(0),
                gf7(0), gf7(0), gf7(7),
            ],
        )
        .unwrap();
        assert_eq!(d.rank().unwrap(), 2);
    }

    #[test]
    fn test_discrete_log_small() {
        let p = Integer::from(7);
        let g = PrimeField::new(Integer::from(3), p.clone()).unwrap(); // 3 is a generator of GF(7)*

        // 3^0 = 1, 3^1 = 3, 3^2 = 2, 3^3 = 6, 3^4 = 4, 3^5 = 5, 3^6 = 1 (mod 7)
        // Test: log_3(2) = 2
        let target = PrimeField::new(Integer::from(2), p.clone()).unwrap();
        let log = PrimeField::discrete_log(&g, &target).unwrap();
        assert_eq!(log, Integer::from(2));

        // Test: log_3(6) = 3
        let target = PrimeField::new(Integer::from(6), p.clone()).unwrap();
        let log = PrimeField::discrete_log(&g, &target).unwrap();
        assert_eq!(log, Integer::from(3));
    }
}

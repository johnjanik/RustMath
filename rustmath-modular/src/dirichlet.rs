//! # Dirichlet Characters
//!
//! This module provides Dirichlet characters and character groups,
//! corresponding to SageMath's sage.modular.dirichlet module.
//!
//! A Dirichlet character modulo N is a group homomorphism from
//! (Z/NZ)* to the multiplicative group of complex numbers.

use rustmath_integers::Integer;
use std::collections::HashMap;

/// A Dirichlet character modulo N
#[derive(Debug, Clone)]
pub struct DirichletCharacter {
    /// The modulus
    modulus: Integer,
    /// Values of the character on generators of (Z/NZ)*
    values: HashMap<Integer, i32>,
    /// Order of the character
    order: Option<usize>,
}

impl DirichletCharacter {
    /// Create a new Dirichlet character
    ///
    /// # Arguments
    /// * `modulus` - The modulus N
    /// * `values` - Character values on generators
    pub fn new(modulus: Integer, values: HashMap<Integer, i32>) -> Self {
        DirichletCharacter {
            modulus,
            values,
            order: None,
        }
    }

    /// Get the modulus of this character
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Evaluate the character at a given integer.
    ///
    /// Correct, and exact, for the TRIVIAL character: chi_0(n) = 1 when
    /// gcd(n, N) = 1 and 0 otherwise.
    ///
    /// PANICS for any non-trivial character.  `values` stores the character only
    /// on GENERATORS of (Z/NZ)*, so evaluating at an arbitrary n requires writing
    /// n as a word in those generators -- a discrete logarithm in (Z/NZ)*, via the
    /// CRT decomposition into the cyclic factors of the (Z/p^e Z)*.  That is not
    /// implemented.  What the code used to do instead was
    /// `values.get(&n).unwrap_or(&1)`: it returned the stored value if n happened
    /// to be one of the generators, and silently returned 1 -- the trivial value
    /// -- for every other n.  That is wrong for essentially every argument, which
    /// is why this now refuses.
    pub fn eval(&self, n: &Integer) -> i32 {
        // Reduce n modulo the modulus (mod_floor semantics for the positive
        // modulus case: always a non-negative representative)
        let n_mod = n.modulo(&self.modulus);

        // chi(n) = 0 for n not coprime to N, for every character.
        if !n_mod.gcd(&self.modulus).is_one() {
            return 0;
        }

        if self.is_trivial() {
            return 1;
        }

        unimplemented!(
            "DirichletCharacter::eval at {n_mod} mod {}: not implemented for a non-trivial \
             character. The character is stored only by its values on generators of \
             (Z/NZ)*, so evaluating it at an arbitrary n needs a discrete logarithm in \
             (Z/NZ)* (CRT down to the cyclic (Z/p^e Z)*, then a dlog in each), which is \
             not implemented. Previously this returned the stored value when n happened \
             to BE a generator, and silently returned 1 for every other n.",
            self.modulus
        )
    }

    /// Check if this is the trivial character
    pub fn is_trivial(&self) -> bool {
        self.values.is_empty() || self.values.values().all(|&v| v == 1)
    }

    /// Get the order of the character
    pub fn order(&self) -> usize {
        if let Some(ord) = self.order {
            return ord;
        }

        if self.is_trivial() {
            return 1;
        }

        // Find lcm of orders of values
        fn gcd_usize(a: usize, b: usize) -> usize {
            if b == 0 { a } else { gcd_usize(b, a % b) }
        }
        fn lcm_usize(a: usize, b: usize) -> usize {
            if a == 0 || b == 0 { 0 } else { a / gcd_usize(a, b) * b }
        }
        let mut ord = 1;
        for &v in self.values.values() {
            ord = lcm_usize(ord, v.abs() as usize);
        }

        ord
    }

    /// The conductor: the smallest modulus f | N such that this character is
    /// induced from a character mod f.
    ///
    /// Correct, and exact, for the TRIVIAL character: chi_0 mod N is induced from
    /// the (unique) character mod 1, so its conductor is 1 -- for EVERY N. Note
    /// this is not N: the old code returned the modulus, so it reported
    /// conductor 12 for the trivial character mod 12, and hence declared it
    /// primitive (`is_primitive` compares the two).  The trivial character is
    /// primitive only at N = 1.
    ///
    /// PANICS for any non-trivial character: finding the conductor means testing,
    /// for each f | N, whether chi factors through (Z/NZ)* -> (Z/fZ)*, which needs
    /// [`Self::eval`] at arbitrary arguments -- and that is not implemented.
    pub fn conductor(&self) -> Integer {
        if self.is_trivial() {
            return Integer::one();
        }
        unimplemented!(
            "DirichletCharacter::conductor mod {}: not implemented for a non-trivial \
             character. It requires testing, for each f | N, whether chi is induced from \
             a character mod f, which needs evaluation at arbitrary arguments (see \
             `eval`). Previously returned the modulus itself, which also made \
             `is_primitive` report true for every character.",
            self.modulus
        )
    }

    /// Check if the character is primitive (conductor equal to the modulus).
    ///
    /// PANICS for a non-trivial character; see [`Self::conductor`].
    pub fn is_primitive(&self) -> bool {
        self.conductor() == self.modulus
    }

    /// Check if the character is even
    pub fn is_even(&self) -> bool {
        // A character is even if χ(-1) = 1
        self.eval(&(-Integer::one())) == 1
    }

    /// Check if the character is odd
    pub fn is_odd(&self) -> bool {
        // A character is odd if χ(-1) = -1
        self.eval(&(-Integer::one())) == -1
    }

    /// The magnitude |G(chi)| of the Gauss sum
    /// `G(chi) = sum_{a mod N} chi(a) e^{2 pi i a / N}`.
    ///
    /// Correct, and exact, for the TRIVIAL character: G(chi_0 mod N) is
    /// Ramanujan's sum c_N(1) = mu(N), so |G(chi_0)| = |mu(N)|.  (In particular
    /// it is 0 whenever N is not squarefree -- the old code returned a flat 1.0
    /// there.)
    ///
    /// PANICS for any non-trivial character; see the message for why.
    pub fn gauss_sum_magnitude(&self) -> f64 {
        if self.is_trivial() {
            return mobius(&self.modulus).abs() as f64;
        }
        unimplemented!(
            "DirichletCharacter::gauss_sum_magnitude mod {}: not implemented for a \
             non-trivial character. For a PRIMITIVE chi the magnitude is sqrt(N), but \
             deciding primitivity needs the conductor (see `conductor`); for an \
             IMPRIMITIVE chi of conductor f it is |mu(N/f)| sqrt(f), which needs f as \
             well. Previously returned sqrt(N) on the strength of an `is_primitive` that \
             was true for every character, and a flat 1.0 otherwise.",
            self.modulus
        )
    }
}

/// The group of Dirichlet characters modulo N
#[derive(Debug, Clone)]
pub struct DirichletGroup {
    /// The modulus
    modulus: Integer,
    /// List of characters in the group
    characters: Vec<DirichletCharacter>,
}

impl DirichletGroup {
    /// Create the group of Dirichlet characters modulo N
    ///
    /// # Arguments
    /// * `modulus` - The modulus N
    pub fn new(modulus: Integer) -> Self {
        let mut characters = Vec::new();

        // Add the trivial character (always correct: the trivial character
        // mod N always exists).
        characters.push(DirichletCharacter::new(
            modulus.clone(),
            HashMap::new(),
        ));

        // Generating the full group of Dirichlet characters mod N requires
        // factoring (Z/NZ)* and constructing all homomorphisms to C*. That is
        // not implemented, so be honest about it rather than silently
        // returning a "group" containing only the trivial character whenever
        // more characters actually exist (i.e. phi(N) > 1).
        if euler_phi(&modulus) > Integer::one() {
            unimplemented!(
                "DirichletGroup::new: generation of non-trivial Dirichlet characters not yet implemented (facade); only the trivial character (phi(N) = 1) is supported"
            );
        }

        DirichletGroup {
            modulus,
            characters,
        }
    }

    /// Get the modulus
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Get the number of characters
    pub fn len(&self) -> usize {
        self.characters.len()
    }

    /// Check if the group is empty
    pub fn is_empty(&self) -> bool {
        self.characters.is_empty()
    }

    /// Get a character by index
    pub fn get(&self, index: usize) -> Option<&DirichletCharacter> {
        self.characters.get(index)
    }

    /// Get the trivial character
    pub fn trivial_character(&self) -> DirichletCharacter {
        DirichletCharacter::new(self.modulus.clone(), HashMap::new())
    }

    /// Get all characters
    pub fn all_characters(&self) -> &[DirichletCharacter] {
        &self.characters
    }

    /// Order of the group (Euler phi function)
    pub fn order(&self) -> Integer {
        euler_phi(&self.modulus)
    }
}

/// Create a Dirichlet group modulo N
///
/// # Arguments
/// * `N` - The modulus
///
/// # Returns
/// The group of Dirichlet characters modulo N
pub fn dirichlet_group_class(N: Integer) -> DirichletGroup {
    DirichletGroup::new(N)
}

/// Check if an object is a Dirichlet character
pub fn is_dirichlet_character(obj: &DirichletCharacter) -> bool {
    // In Rust, this is always true if we have the object
    true
}

/// Check if an object is a Dirichlet group
pub fn is_dirichlet_group(obj: &DirichletGroup) -> bool {
    // In Rust, this is always true if we have the object
    true
}

/// Get the trivial character modulo N
///
/// # Arguments
/// * `N` - The modulus
///
/// # Returns
/// The trivial character modulo N
pub fn trivial_character(N: Integer) -> DirichletCharacter {
    DirichletCharacter::new(N, HashMap::new())
}


/// Compute the Kronecker character (d/.)
///
/// # Arguments
/// * `d` - The discriminant
///
/// # Returns
/// The Kronecker character
pub fn kronecker_character(d: Integer) -> DirichletCharacter {
    // The Kronecker character is a Dirichlet character defined by
    // the Kronecker symbol (d/n). Computing its actual values requires
    // evaluating the Kronecker symbol, which is not implemented here;
    // previously this silently returned the trivial character instead.
    let _ = d;
    unimplemented!(
        "kronecker_character not yet implemented (facade): previously returned the trivial character"
    )
}

/// Compute the Kronecker character (./d)
///
/// # Arguments
/// * `d` - The discriminant
///
/// # Returns
/// The Kronecker character (upside down)
pub fn kronecker_character_upside_down(d: Integer) -> DirichletCharacter {
    // Similar to kronecker_character but with arguments flipped
    kronecker_character(d)
}

/// Create a principal Dirichlet character modulo N
///
/// This is the character χ that is 1 on all units
pub fn principal_character(N: Integer) -> DirichletCharacter {
    trivial_character(N)
}

/// Create a quadratic Dirichlet character from a discriminant
///
/// # Arguments
/// * `d` - The discriminant (must be 0 or 1 mod 4)
///
/// # Returns
/// The quadratic character (d/·)
pub fn quadratic_character(d: i64) -> DirichletCharacter {
    // The quadratic character is defined by the Kronecker symbol (d/.).
    // Computing the actual character values is not implemented here;
    // previously this silently returned the trivial character instead.
    let _ = d;
    unimplemented!(
        "quadratic_character not yet implemented (facade): previously returned the trivial character"
    )
}

/// The Mobius function mu(n) for n >= 1: 0 if n is not squarefree, else
/// (-1)^(number of prime factors).
fn mobius(n: &Integer) -> i32 {
    if n <= &Integer::zero() {
        return 0;
    }
    let mut m = n.clone();
    let mut p = Integer::from(2);
    let mut primes = 0u32;
    while &p * &p <= m {
        if (&m % &p).is_zero() {
            m = &m / &p;
            if (&m % &p).is_zero() {
                return 0; // p^2 | n
            }
            primes += 1;
        }
        p = p + Integer::one();
    }
    if m > Integer::one() {
        primes += 1;
    }
    if primes.is_multiple_of(2) { 1 } else { -1 }
}

/// Compute Euler's phi function (totient)
fn euler_phi(n: &Integer) -> Integer {
    if n <= &Integer::one() {
        return Integer::one();
    }

    let mut result = n.clone();
    let mut n_copy = n.clone();
    let mut p = Integer::from(2);

    while &p * &p <= n_copy {
        if (&n_copy % &p).is_zero() {
            while (&n_copy % &p).is_zero() {
                n_copy = &n_copy / &p;
            }
            result = result * (&p - &Integer::one()) / p.clone();
        }
        p = p + Integer::one();
    }

    if n_copy > Integer::one() {
        result = result * (&n_copy - &Integer::one()) / n_copy.clone();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-trivial character, stored the way `DirichletCharacter::new` invites:
    /// by its value on a generator of (Z/5Z)* (2 is a generator; chi(2) = -1
    /// gives the quadratic character mod 5).
    fn nontrivial_mod5() -> DirichletCharacter {
        let mut values = HashMap::new();
        values.insert(Integer::from(2), -1);
        DirichletCharacter::new(Integer::from(5), values)
    }

    #[test]
    fn test_trivial_character() {
        let chi = trivial_character(Integer::from(12));
        assert_eq!(chi.modulus(), &Integer::from(12));
        assert!(chi.is_trivial());
        assert_eq!(chi.order(), 1);
    }

    #[test]
    fn test_TrivialCharacter() {
        let chi = trivial_character(Integer::from(12));
        assert!(chi.is_trivial());
    }

    /// The conductor of the TRIVIAL character is 1, for every modulus -- it is
    /// induced from the character mod 1.  The old code returned the modulus, so
    /// it called the trivial character mod 12 primitive.  It is primitive only at
    /// N = 1.
    #[test]
    fn test_trivial_character_conductor_is_one_not_the_modulus() {
        for n in [1u64, 2, 5, 12, 30] {
            let chi = trivial_character(Integer::from(n));
            assert_eq!(chi.conductor(), Integer::one(), "conductor of chi_0 mod {n}");
            assert_eq!(chi.is_primitive(), n == 1, "primitivity of chi_0 mod {n}");
        }
    }

    /// |G(chi_0 mod N)| = |mu(N)| (Ramanujan's sum c_N(1) = mu(N)).  In
    /// particular it is 0 for non-squarefree N, where the old code returned 1.0.
    #[test]
    fn test_trivial_character_gauss_sum_magnitude_is_mobius() {
        for (n, g) in [(1u64, 1.0), (2, 1.0), (5, 1.0), (6, 1.0), (12, 0.0), (30, 1.0), (4, 0.0)] {
            let chi = trivial_character(Integer::from(n));
            assert_eq!(chi.gauss_sum_magnitude(), g, "|G(chi_0 mod {n})|");
        }
    }

    #[test]
    fn test_is_DirichletCharacter() {
        let chi = trivial_character(Integer::from(7));
        assert!(is_dirichlet_character(&chi));
    }

    /// chi(n) = 0 for gcd(n, N) > 1 and 1 otherwise: the trivial character is the
    /// one case `eval` genuinely computes.
    #[test]
    fn test_character_eval() {
        let chi = trivial_character(Integer::from(5));
        assert_eq!(chi.eval(&Integer::from(3)), 1);
        assert_eq!(chi.eval(&Integer::from(5)), 0); // Not coprime to modulus
        assert_eq!(chi.eval(&Integer::from(-1)), 1);
        assert!(chi.is_even());
    }

    /// The facades now REFUSE instead of returning a wrong value.  These three
    /// used to be `#[ignore]`d because they pinned facades; the refusal itself is
    /// testable, so they are ignored no longer.
    #[test]
    #[should_panic(expected = "DirichletGroup::new")]
    fn test_dirichlet_group_is_refused_not_faked() {
        // phi(5) = 4 > 1, so there are non-trivial characters mod 5 and a
        // "group" containing only the trivial one would be a lie.
        let _ = DirichletGroup::new(Integer::from(5));
    }

    #[test]
    #[should_panic(expected = "DirichletGroup::new")]
    fn test_is_DirichletGroup_is_refused_not_faked() {
        let _ = is_dirichlet_group(&DirichletGroup::new(Integer::from(11)));
    }

    #[test]
    #[should_panic(expected = "kronecker_character")]
    fn test_kronecker_character_is_refused_not_faked() {
        let _ = kronecker_character(Integer::from(5));
    }

    #[test]
    #[should_panic(expected = "quadratic_character")]
    fn test_quadratic_character_is_refused_not_faked() {
        let _ = quadratic_character(5);
    }

    /// Evaluating a NON-trivial character is refused.  The old code returned the
    /// stored value at a generator and silently returned 1 -- the trivial value --
    /// at every other argument: here chi(2) = -1 is stored, but chi(3) would have
    /// come back as 1 when the true quadratic character mod 5 has chi(3) = -1.
    #[test]
    #[should_panic(expected = "DirichletCharacter::eval")]
    fn test_nontrivial_eval_is_refused_not_faked() {
        let _ = nontrivial_mod5().eval(&Integer::from(3));
    }

    /// ... but chi(n) = 0 for gcd(n, N) > 1 holds for EVERY character, so that
    /// branch still answers.
    #[test]
    fn test_nontrivial_eval_still_knows_the_noncoprime_zero() {
        assert_eq!(nontrivial_mod5().eval(&Integer::from(10)), 0);
    }

    #[test]
    #[should_panic(expected = "DirichletCharacter::conductor")]
    fn test_nontrivial_conductor_is_refused_not_faked() {
        let _ = nontrivial_mod5().conductor();
    }

    #[test]
    #[should_panic(expected = "DirichletCharacter::gauss_sum_magnitude")]
    fn test_nontrivial_gauss_sum_is_refused_not_faked() {
        let _ = nontrivial_mod5().gauss_sum_magnitude();
    }

    #[test]
    fn test_euler_phi() {
        assert_eq!(euler_phi(&Integer::from(1)), Integer::one());
        assert_eq!(euler_phi(&Integer::from(2)), Integer::one());
        assert_eq!(euler_phi(&Integer::from(5)), Integer::from(4));
        assert_eq!(euler_phi(&Integer::from(12)), Integer::from(4));
    }

    #[test]
    fn test_mobius() {
        for (n, m) in [
            (1i64, 1i32), (2, -1), (3, -1), (4, 0), (5, -1), (6, 1), (7, -1),
            (8, 0), (9, 0), (10, 1), (12, 0), (30, -1), (31, -1),
        ] {
            assert_eq!(mobius(&Integer::from(n)), m, "mu({n})");
        }
    }
}

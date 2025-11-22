//! # Dirichlet Characters
//!
//! This module provides Dirichlet characters and character groups,
//! corresponding to SageMath's sage.modular.dirichlet module.
//!
//! A Dirichlet character modulo N is a group homomorphism from
//! (Z/NZ)* to the multiplicative group of complex numbers.

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Signed};
use num_integer::Integer;
use std::collections::HashMap;

/// A Dirichlet character modulo N
#[derive(Debug, Clone)]
pub struct DirichletCharacter {
    /// The modulus
    modulus: BigInt,
    /// Values of the character on generators of (Z/NZ)*
    values: HashMap<BigInt, i32>,
    /// Order of the character
    order: Option<usize>,
}

impl DirichletCharacter {
    /// Create a new Dirichlet character
    ///
    /// # Arguments
    /// * `modulus` - The modulus N
    /// * `values` - Character values on generators
    pub fn new(modulus: BigInt, values: HashMap<BigInt, i32>) -> Self {
        DirichletCharacter {
            modulus,
            values,
            order: None,
        }
    }

    /// Get the modulus of this character
    pub fn modulus(&self) -> &BigInt {
        &self.modulus
    }

    /// Evaluate the character at a given integer
    ///
    /// # Arguments
    /// * `n` - The integer to evaluate at
    ///
    /// # Returns
    /// The value of the character (as root of unity represented by integer power)
    pub fn eval(&self, n: &BigInt) -> i32 {
        // Reduce n modulo the modulus
        let n_mod = n.mod_floor(&self.modulus);

        // Check if n is coprime to modulus
        if !n_mod.gcd(&self.modulus).is_one() {
            return 0;
        }

        // For trivial character
        if self.values.is_empty() {
            return 1;
        }

        // Compute character value
        // This is simplified; proper implementation would decompose n
        *self.values.get(&n_mod).unwrap_or(&1)
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
        let mut ord = 1;
        for &v in self.values.values() {
            ord = num_integer::lcm(ord, v.abs() as usize);
        }

        ord
    }

    /// Compute the conductor of the character
    pub fn conductor(&self) -> BigInt {
        // Conductor is the smallest modulus for which this character is defined
        // For now, return the modulus (simplified)
        self.modulus.clone()
    }

    /// Check if the character is primitive
    pub fn is_primitive(&self) -> bool {
        self.conductor() == self.modulus
    }

    /// Check if the character is even
    pub fn is_even(&self) -> bool {
        // A character is even if χ(-1) = 1
        self.eval(&(-BigInt::one())) == 1
    }

    /// Check if the character is odd
    pub fn is_odd(&self) -> bool {
        // A character is odd if χ(-1) = -1
        self.eval(&(-BigInt::one())) == -1
    }

    /// Gauss sum of the character
    ///
    /// G(χ) = Σ_{a mod N} χ(a) * e^(2πia/N)
    ///
    /// For primitive characters, |G(χ)| = √N
    pub fn gauss_sum_magnitude(&self) -> f64 {
        if self.is_primitive() {
            (self.modulus.to_f64().unwrap_or(1.0)).sqrt()
        } else {
            // For imprimitive characters, more complex
            1.0
        }
    }
}

/// The group of Dirichlet characters modulo N
#[derive(Debug, Clone)]
pub struct DirichletGroup {
    /// The modulus
    modulus: BigInt,
    /// List of characters in the group
    characters: Vec<DirichletCharacter>,
}

impl DirichletGroup {
    /// Create the group of Dirichlet characters modulo N
    ///
    /// # Arguments
    /// * `modulus` - The modulus N
    pub fn new(modulus: BigInt) -> Self {
        let mut characters = Vec::new();

        // Add the trivial character
        characters.push(DirichletCharacter::new(
            modulus.clone(),
            HashMap::new(),
        ));

        // TODO: Generate all characters
        // This requires factoring (Z/NZ)* and creating all homomorphisms

        DirichletGroup {
            modulus,
            characters,
        }
    }

    /// Get the modulus
    pub fn modulus(&self) -> &BigInt {
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
    pub fn order(&self) -> BigInt {
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
pub fn dirichlet_group_class(N: BigInt) -> DirichletGroup {
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
pub fn trivial_character(N: BigInt) -> DirichletCharacter {
    DirichletCharacter::new(N, HashMap::new())
}


/// Compute the Kronecker character (d/.)
///
/// # Arguments
/// * `d` - The discriminant
///
/// # Returns
/// The Kronecker character
pub fn kronecker_character(d: BigInt) -> DirichletCharacter {
    // The Kronecker character is a Dirichlet character defined by
    // the Kronecker symbol (d/n)
    // For now, return a trivial implementation
    DirichletCharacter::new(d.abs(), HashMap::new())
}

/// Compute the Kronecker character (./d)
///
/// # Arguments
/// * `d` - The discriminant
///
/// # Returns
/// The Kronecker character (upside down)
pub fn kronecker_character_upside_down(d: BigInt) -> DirichletCharacter {
    // Similar to kronecker_character but with arguments flipped
    kronecker_character(d)
}

/// Create a principal Dirichlet character modulo N
///
/// This is the character χ that is 1 on all units
pub fn principal_character(N: BigInt) -> DirichletCharacter {
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
    // The quadratic character is defined by the Kronecker symbol
    let modulus = BigInt::from(d.abs());

    // For a proper implementation, we'd need to compute the actual values
    // For now, return a basic character
    DirichletCharacter::new(modulus, HashMap::new())
}

/// Compute Euler's phi function (totient)
fn euler_phi(n: &BigInt) -> BigInt {
    if n <= &BigInt::one() {
        return BigInt::one();
    }

    let mut result = n.clone();
    let mut n_copy = n.clone();
    let mut p = BigInt::from(2);

    while &p * &p <= n_copy {
        if n_copy.is_multiple_of(&p) {
            while n_copy.is_multiple_of(&p) {
                n_copy /= &p;
            }
            result = result * (&p - BigInt::one()) / &p;
        }
        p += BigInt::one();
    }

    if n_copy > BigInt::one() {
        result = result * (&n_copy - BigInt::one()) / &n_copy;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_character() {
        let chi = trivial_character(BigInt::from(12));
        assert_eq!(chi.modulus(), &BigInt::from(12));
        assert!(chi.is_trivial());
        assert_eq!(chi.order(), 1);
    }

    #[test]
    fn test_TrivialCharacter() {
        let chi = TrivialCharacter(BigInt::from(12));
        assert!(chi.is_trivial());
    }

    #[test]
    fn test_dirichlet_group() {
        let G = DirichletGroup::new(BigInt::from(5));
        assert_eq!(G.modulus(), &BigInt::from(5));
        assert!(!G.is_empty());
    }

    #[test]
    fn test_is_DirichletCharacter() {
        let chi = trivial_character(BigInt::from(7));
        assert!(is_DirichletCharacter(&chi));
    }

    #[test]
    fn test_is_DirichletGroup() {
        let G = DirichletGroup::new(BigInt::from(11));
        assert!(is_DirichletGroup(&G));
    }

    #[test]
    fn test_character_eval() {
        let chi = trivial_character(BigInt::from(5));
        assert_eq!(chi.eval(&BigInt::from(3)), 1);
        assert_eq!(chi.eval(&BigInt::from(5)), 0); // Not coprime to modulus
    }

    #[test]
    fn test_kronecker_character() {
        let chi = kronecker_character(BigInt::from(5));
        assert_eq!(chi.modulus(), &BigInt::from(5));
    }

    #[test]
    fn test_euler_phi() {
        assert_eq!(euler_phi(&BigInt::from(1)), BigInt::one());
        assert_eq!(euler_phi(&BigInt::from(2)), BigInt::one());
        assert_eq!(euler_phi(&BigInt::from(5)), BigInt::from(4));
        assert_eq!(euler_phi(&BigInt::from(12)), BigInt::from(4));
    }
}

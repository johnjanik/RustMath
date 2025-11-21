//! Noncommutative ideals and ideal monoids
//!
//! This module provides ideal structures for noncommutative rings,
//! corresponding to SageMath's `sage.rings.noncommutative_ideals`.
//!
//! # Mathematical Background
//!
//! For a noncommutative ring R, ideals come in three flavors:
//!
//! ## Left Ideals
//!
//! A left ideal I of R satisfies:
//! - I is an additive subgroup of R
//! - r · i ∈ I for all r ∈ R, i ∈ I (left multiplication by R)
//!
//! ## Right Ideals
//!
//! A right ideal I of R satisfies:
//! - I is an additive subgroup of R
//! - i · r ∈ I for all r ∈ R, i ∈ I (right multiplication by R)
//!
//! ## Two-Sided Ideals
//!
//! A two-sided ideal I is both a left and right ideal:
//! - r · i ∈ I and i · r ∈ I for all r ∈ R, i ∈ I
//! - These are the "proper" ideals for noncommutative rings
//!
//! # Key Differences from Commutative Case
//!
//! - Not all left/right ideals are two-sided
//! - Cannot always form quotient rings (need two-sided ideals)
//! - Principal ideal theorem doesn't apply generally
//! - Krull dimension may not be well-defined
//!
//! # Examples
//!
//! ## Matrix Rings
//!
//! In M_n(F), the matrices with first column zero form a left ideal but not
//! a right ideal (for n > 1).
//!
//! ## Weyl Algebra
//!
//! The Weyl algebra A_1 = k⟨x, ∂⟩/(∂x - x∂ - 1) is a noncommutative ring
//! where ideals have special properties related to differential operators.

use std::fmt;
use std::collections::HashSet;

/// Monoid of ideals in a noncommutative ring
///
/// This corresponds to SageMath's `IdealMonoid_nc` class.
///
/// The ideal monoid consists of all ideals of a ring with the operation
/// of ideal multiplication.
#[derive(Clone, Debug)]
pub struct IdealMonoidNc {
    /// Name of the base ring
    ring_name: String,
    /// Type of ideals (left, right, or two-sided)
    ideal_type: IdealType,
    /// Known prime ideals
    prime_ideals: HashSet<String>,
}

/// Type of ideal in a noncommutative ring
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdealType {
    /// Left ideal: R · I ⊆ I
    Left,
    /// Right ideal: I · R ⊆ I
    Right,
    /// Two-sided ideal: R · I ⊆ I and I · R ⊆ I
    TwoSided,
}

impl IdealMonoid_nc {
    /// Create a new ideal monoid
    ///
    /// # Arguments
    ///
    /// * `ring_name` - Name of the base ring
    /// * `ideal_type` - Type of ideals in the monoid
    ///
    /// # Returns
    ///
    /// A new IdealMonoid_nc instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let monoid = IdealMonoid_nc::new(
    ///     "M_2(Q)".to_string(),
    ///     IdealType::Left
    /// );
    /// ```
    pub fn new(ring_name: String, ideal_type: IdealType) -> Self {
        IdealMonoid_nc {
            ring_name,
            ideal_type,
            prime_ideals: HashSet::new(),
        }
    }

    /// Get the ring name
    ///
    /// # Returns
    ///
    /// Name of the base ring
    pub fn ring_name(&self) -> &str {
        &self.ring_name
    }

    /// Get the ideal type
    ///
    /// # Returns
    ///
    /// Type of ideals
    pub fn ideal_type(&self) -> &IdealType {
        &self.ideal_type
    }

    /// Add a prime ideal
    ///
    /// # Arguments
    ///
    /// * `ideal` - String representation of the prime ideal
    pub fn add_prime(&mut self, ideal: String) {
        self.prime_ideals.insert(ideal);
    }

    /// Get number of known primes
    ///
    /// # Returns
    ///
    /// Count of prime ideals
    pub fn num_primes(&self) -> usize {
        self.prime_ideals.len()
    }

    /// Get the unit ideal
    ///
    /// # Returns
    ///
    /// The ring itself as an ideal
    pub fn unit_ideal(&self) -> Ideal_nc {
        Ideal_nc::unit(self.ring_name.clone(), self.ideal_type.clone())
    }

    /// Get the zero ideal
    ///
    /// # Returns
    ///
    /// The zero ideal {0}
    pub fn zero_ideal(&self) -> Ideal_nc {
        Ideal_nc::zero(self.ring_name.clone(), self.ideal_type.clone())
    }
}

impl fmt::Display for IdealMonoid_nc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match self.ideal_type {
            IdealType::Left => "left",
            IdealType::Right => "right",
            IdealType::TwoSided => "two-sided",
        };
        write!(
            f,
            "Monoid of {} ideals of {}",
            type_str, self.ring_name
        )
    }
}

/// Ideal in a noncommutative ring
///
/// This corresponds to SageMath's `Ideal_nc` class.
#[derive(Clone, Debug)]
pub struct IdealNc {
    /// Name/description of the ideal
    name: String,
    /// Ring the ideal belongs to
    ring_name: String,
    /// Type of ideal
    ideal_type: IdealType,
    /// Generators of the ideal
    generators: Vec<String>,
    /// Whether this is a prime ideal
    is_prime: bool,
}

impl Ideal_nc {
    /// Create a new noncommutative ideal
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the ideal
    /// * `ring_name` - Name of the ring
    /// * `ideal_type` - Type of ideal
    /// * `generators` - Generating set
    ///
    /// # Returns
    ///
    /// A new Ideal_nc instance
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let ideal = Ideal_nc::new(
    ///     "I".to_string(),
    ///     "A_1".to_string(),
    ///     IdealType::TwoSided,
    ///     vec!["x".to_string(), "∂".to_string()]
    /// );
    /// ```
    pub fn new(
        name: String,
        ring_name: String,
        ideal_type: IdealType,
        generators: Vec<String>,
    ) -> Self {
        Ideal_nc {
            name,
            ring_name,
            ideal_type,
            generators,
            is_prime: false,
        }
    }

    /// Create the unit ideal (the whole ring)
    ///
    /// # Arguments
    ///
    /// * `ring_name` - Name of the ring
    /// * `ideal_type` - Type of ideal
    ///
    /// # Returns
    ///
    /// The unit ideal
    pub fn unit(ring_name: String, ideal_type: IdealType) -> Self {
        Ideal_nc {
            name: ring_name.clone(),
            ring_name,
            ideal_type,
            generators: vec!["1".to_string()],
            is_prime: false,
        }
    }

    /// Create the zero ideal
    ///
    /// # Arguments
    ///
    /// * `ring_name` - Name of the ring
    /// * `ideal_type` - Type of ideal
    ///
    /// # Returns
    ///
    /// The zero ideal
    pub fn zero(ring_name: String, ideal_type: IdealType) -> Self {
        Ideal_nc {
            name: "(0)".to_string(),
            ring_name,
            ideal_type,
            generators: vec!["0".to_string()],
            is_prime: false,
        }
    }

    /// Get the name
    ///
    /// # Returns
    ///
    /// Name of the ideal
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the ring
    ///
    /// # Returns
    ///
    /// Name of the containing ring
    pub fn ring(&self) -> &str {
        &self.ring_name
    }

    /// Get ideal type
    ///
    /// # Returns
    ///
    /// Type of ideal
    pub fn ideal_type(&self) -> &IdealType {
        &self.ideal_type
    }

    /// Get generators
    ///
    /// # Returns
    ///
    /// Generating set
    pub fn generators(&self) -> &[String] {
        &self.generators
    }

    /// Check if prime
    ///
    /// # Returns
    ///
    /// True if this is a prime ideal
    pub fn is_prime(&self) -> bool {
        self.is_prime
    }

    /// Mark as prime ideal
    pub fn set_prime(&mut self) {
        self.is_prime = true;
    }

    /// Check if this is the unit ideal
    ///
    /// # Returns
    ///
    /// True if this is the whole ring
    pub fn is_unit(&self) -> bool {
        self.generators.contains(&"1".to_string())
    }

    /// Check if this is the zero ideal
    ///
    /// # Returns
    ///
    /// True if this is {0}
    pub fn is_zero(&self) -> bool {
        self.generators.len() == 1 && self.generators[0] == "0"
    }
}

impl fmt::Display for Ideal_nc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match self.ideal_type {
            IdealType::Left => "Left",
            IdealType::Right => "Right",
            IdealType::TwoSided => "Two-sided",
        };
        write!(
            f,
            "{} ideal {} of {}",
            type_str, self.name, self.ring_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_monoid_creation() {
        let monoid = IdealMonoid_nc::new(
            "M_2(Q)".to_string(),
            IdealType::Left,
        );

        assert_eq!(monoid.ring_name(), "M_2(Q)");
        assert_eq!(monoid.ideal_type(), &IdealType::Left);
    }

    #[test]
    fn test_ideal_monoid_primes() {
        let mut monoid = IdealMonoid_nc::new(
            "R".to_string(),
            IdealType::TwoSided,
        );

        monoid.add_prime("P1".to_string());
        monoid.add_prime("P2".to_string());

        assert_eq!(monoid.num_primes(), 2);
    }

    #[test]
    fn test_ideal_monoid_unit() {
        let monoid = IdealMonoid_nc::new(
            "R".to_string(),
            IdealType::TwoSided,
        );

        let unit = monoid.unit_ideal();
        assert!(unit.is_unit());
    }

    #[test]
    fn test_ideal_monoid_zero() {
        let monoid = IdealMonoid_nc::new(
            "R".to_string(),
            IdealType::TwoSided,
        );

        let zero = monoid.zero_ideal();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_ideal_creation() {
        let ideal = Ideal_nc::new(
            "I".to_string(),
            "R".to_string(),
            IdealType::Left,
            vec!["a".to_string(), "b".to_string()],
        );

        assert_eq!(ideal.name(), "I");
        assert_eq!(ideal.ring(), "R");
        assert_eq!(ideal.generators().len(), 2);
    }

    #[test]
    fn test_ideal_types() {
        let left = Ideal_nc::new(
            "L".to_string(),
            "R".to_string(),
            IdealType::Left,
            vec![],
        );

        let right = Ideal_nc::new(
            "R".to_string(),
            "R".to_string(),
            IdealType::Right,
            vec![],
        );

        let two_sided = Ideal_nc::new(
            "T".to_string(),
            "R".to_string(),
            IdealType::TwoSided,
            vec![],
        );

        assert_eq!(left.ideal_type(), &IdealType::Left);
        assert_eq!(right.ideal_type(), &IdealType::Right);
        assert_eq!(two_sided.ideal_type(), &IdealType::TwoSided);
    }

    #[test]
    fn test_ideal_prime() {
        let mut ideal = Ideal_nc::new(
            "P".to_string(),
            "R".to_string(),
            IdealType::TwoSided,
            vec!["x".to_string()],
        );

        assert!(!ideal.is_prime());
        ideal.set_prime();
        assert!(ideal.is_prime());
    }

    #[test]
    fn test_ideal_unit() {
        let ideal = Ideal_nc::unit("R".to_string(), IdealType::TwoSided);

        assert!(ideal.is_unit());
        assert!(!ideal.is_zero());
    }

    #[test]
    fn test_ideal_zero() {
        let ideal = Ideal_nc::zero("R".to_string(), IdealType::TwoSided);

        assert!(ideal.is_zero());
        assert!(!ideal.is_unit());
    }

    #[test]
    fn test_ideal_display() {
        let ideal = Ideal_nc::new(
            "I".to_string(),
            "R".to_string(),
            IdealType::Left,
            vec!["a".to_string()],
        );

        let display = format!("{}", ideal);
        assert!(display.contains("Left"));
        assert!(display.contains("ideal I"));
    }

    #[test]
    fn test_monoid_display() {
        let monoid = IdealMonoid_nc::new(
            "R".to_string(),
            IdealType::Right,
        );

        let display = format!("{}", monoid);
        assert!(display.contains("right ideals"));
    }

    #[test]
    fn test_ideal_clone() {
        let ideal1 = Ideal_nc::new(
            "I".to_string(),
            "R".to_string(),
            IdealType::TwoSided,
            vec!["x".to_string()],
        );
        let ideal2 = ideal1.clone();

        assert_eq!(ideal1.name(), ideal2.name());
        assert_eq!(ideal1.ring(), ideal2.ring());
    }
}

//! # Value Groups
//!
//! The image of a valuation (typically ℤ or a subgroup)

use std::fmt;

/// Codomain of a discrete valuation
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteValuationCodomain {
    /// Whether infinity is included
    includes_infinity: bool,
}

impl DiscreteValuationCodomain {
    pub fn new() -> Self {
        Self { includes_infinity: true }
    }
}

impl Default for DiscreteValuationCodomain {
    fn default() -> Self {
        Self::new()
    }
}

/// Discrete value group (additive group of values)
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteValueGroup {
    /// Generator (typically 1 for ℤ)
    generator: i64,
}

impl DiscreteValueGroup {
    pub fn new(generator: i64) -> Self {
        Self { generator }
    }

    pub fn integers() -> Self {
        Self { generator: 1 }
    }

    pub fn generator(&self) -> i64 {
        self.generator
    }

    pub fn contains(&self, value: i64) -> bool {
        value % self.generator == 0
    }
}

impl Default for DiscreteValueGroup {
    fn default() -> Self {
        Self::integers()
    }
}

/// Discrete value semigroup
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteValueSemigroup {
    /// Generators
    generators: Vec<i64>,
}

impl DiscreteValueSemigroup {
    pub fn new(generators: Vec<i64>) -> Self {
        Self { generators }
    }

    pub fn non_negative_integers() -> Self {
        Self { generators: vec![1] }
    }

    pub fn generators(&self) -> &[i64] {
        &self.generators
    }
}

impl Default for DiscreteValueSemigroup {
    fn default() -> Self {
        Self::non_negative_integers()
    }
}

impl fmt::Display for DiscreteValueGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.generator == 1 {
            write!(f, "ℤ")
        } else {
            write!(f, "{}ℤ", self.generator)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_group() {
        let group = DiscreteValueGroup::integers();
        assert_eq!(group.generator(), 1);
        assert!(group.contains(5));
        assert!(group.contains(-3));

        let scaled = DiscreteValueGroup::new(2);
        assert!(scaled.contains(4));
        assert!(!scaled.contains(3));
    }

    #[test]
    fn test_value_semigroup() {
        let semigroup = DiscreteValueSemigroup::non_negative_integers();
        assert_eq!(semigroup.generators(), &[1]);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", DiscreteValueGroup::integers()), "ℤ");
        assert_eq!(format!("{}", DiscreteValueGroup::new(3)), "3ℤ");
    }
}

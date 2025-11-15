//! Graded Lie Conformal Algebras
//!
//! Provides infrastructure for graded Lie conformal algebras, where elements
//! have associated degrees or weights.
//!
//! A graded Lie conformal algebra has a grading L = ⊕ᵢ Lᵢ such that
//! [Lᵢ_λ Lⱼ] ⊆ R[λ] ⊗ L_{i+j}
//!
//! Corresponds to sage.algebras.lie_conformal_algebras.graded_lie_conformal_algebra

use rustmath_core::Ring;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::fmt::{self, Display};
use crate::lie_conformal_algebra::{LieConformalAlgebra, GeneratorIndex};
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Degree type for graded Lie conformal algebras
///
/// Can be integral or rational depending on the grading
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Degree {
    /// Integer degree
    Integer(i64),
    /// Rational degree
    Rational(i64, i64), // numerator, denominator
}

impl Degree {
    /// Create an integer degree
    pub fn int(n: i64) -> Self {
        Degree::Integer(n)
    }

    /// Create a rational degree
    pub fn rational(num: i64, den: i64) -> Self {
        if den == 1 {
            Degree::Integer(num)
        } else {
            Degree::Rational(num, den)
        }
    }

    /// Add two degrees
    pub fn add(&self, other: &Degree) -> Degree {
        match (self, other) {
            (Degree::Integer(a), Degree::Integer(b)) => Degree::Integer(a + b),
            (Degree::Integer(a), Degree::Rational(num, den)) => {
                Degree::Rational(a * den + num, *den)
            }
            (Degree::Rational(num, den), Degree::Integer(b)) => {
                Degree::Rational(num + b * den, *den)
            }
            (Degree::Rational(num1, den1), Degree::Rational(num2, den2)) => {
                let num = num1 * den2 + num2 * den1;
                let den = den1 * den2;
                Degree::Rational(num, den)
            }
        }
    }

    /// Convert to rational
    pub fn as_rational(&self) -> Rational {
        match self {
            Degree::Integer(n) => Rational::from(*n),
            Degree::Rational(num, den) => Rational::new(*num, *den),
        }
    }
}

impl Display for Degree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Degree::Integer(n) => write!(f, "{}", n),
            Degree::Rational(num, den) => write!(f, "{}/{}", num, den),
        }
    }
}

impl From<i64> for Degree {
    fn from(n: i64) -> Self {
        Degree::Integer(n)
    }
}

/// Weight (conformal weight) for graded structures
///
/// In conformal field theory, elements often have a conformal weight
pub type Weight = Degree;

/// Trait for graded Lie conformal algebras
///
/// A graded Lie conformal algebra has a grading compatible with
/// the λ-bracket operation.
pub trait GradedLieConformalAlgebra<R: Ring>: LieConformalAlgebra<R> {
    /// Get the degree of a generator
    fn generator_degree(&self, index: usize) -> Option<Degree>;

    /// Get all generator degrees
    fn generator_degrees(&self) -> Vec<Degree> {
        if let Some(n) = self.ngens() {
            (0..n)
                .filter_map(|i| self.generator_degree(i))
                .collect()
        } else {
            vec![]
        }
    }

    /// Get the degree of an element
    fn degree(&self, element: &Self::Element) -> Option<Degree>;

    /// Check if the grading is consistent
    fn is_grading_consistent(&self) -> bool {
        // In a properly graded algebra, [Lᵢ_λ Lⱼ] ⊆ R[λ] ⊗ L_{i+j}
        // This would require computing λ-brackets, which we defer to implementations
        true
    }

    /// Get the homogeneous component of a given degree
    fn homogeneous_component(&self, degree: Degree) -> Vec<Self::Element> {
        // Returns all elements of the given degree
        // Default implementation returns empty; override in concrete implementations
        vec![]
    }
}

/// A generic graded Lie conformal algebra structure
///
/// Stores generator names, degrees, and structure information.
#[derive(Clone)]
pub struct GradedLCA<R: Ring> {
    /// Base ring
    base_ring: R,
    /// Number of generators
    ngens: usize,
    /// Generator names
    names: Vec<String>,
    /// Generator degrees
    degrees: Vec<Degree>,
    /// Generator parities (for super-algebras: 0 = even, 1 = odd)
    parities: Option<Vec<u8>>,
}

impl<R: Ring + Clone> GradedLCA<R> {
    /// Create a new graded Lie conformal algebra
    ///
    /// # Arguments
    ///
    /// * `base_ring` - The base ring
    /// * `names` - Generator names
    /// * `degrees` - Generator degrees
    /// * `parities` - Optional generator parities for super-algebras
    pub fn new(
        base_ring: R,
        names: Vec<String>,
        degrees: Vec<Degree>,
        parities: Option<Vec<u8>>,
    ) -> Self {
        assert_eq!(names.len(), degrees.len(), "Names and degrees must have same length");
        if let Some(ref p) = parities {
            assert_eq!(names.len(), p.len(), "Names and parities must have same length");
        }

        GradedLCA {
            base_ring,
            ngens: names.len(),
            names,
            degrees,
            parities,
        }
    }

    /// Get the base ring
    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    /// Get the number of generators
    pub fn ngens(&self) -> usize {
        self.ngens
    }

    /// Get generator name
    pub fn generator_name(&self, i: usize) -> Option<&str> {
        self.names.get(i).map(|s| s.as_str())
    }

    /// Get generator degree
    pub fn generator_degree(&self, i: usize) -> Option<Degree> {
        self.degrees.get(i).copied()
    }

    /// Get generator parity
    pub fn generator_parity(&self, i: usize) -> Option<u8> {
        self.parities.as_ref().and_then(|p| p.get(i).copied())
    }

    /// Check if this is a super-algebra
    pub fn is_super(&self) -> bool {
        self.parities.is_some()
    }

    /// Get all generator names
    pub fn generator_names(&self) -> &[String] {
        &self.names
    }

    /// Get all generator degrees
    pub fn generator_degrees(&self) -> &[Degree] {
        &self.degrees
    }
}

impl<R: Ring + Clone> Display for GradedLCA<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graded Lie conformal algebra with {} generator(s)", self.ngens)?;
        if self.ngens > 0 {
            write!(f, ": ")?;
            for (i, name) in self.names.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}[deg={}]", name, self.degrees[i])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_operations() {
        let d1 = Degree::int(2);
        let d2 = Degree::int(3);
        assert_eq!(d1.add(&d2), Degree::int(5));

        let d3 = Degree::rational(1, 2);
        let d4 = Degree::rational(1, 3);
        let sum = d3.add(&d4);
        assert_eq!(sum, Degree::Rational(5, 6)); // 1/2 + 1/3 = 5/6
    }

    #[test]
    fn test_degree_display() {
        assert_eq!(format!("{}", Degree::int(5)), "5");
        assert_eq!(format!("{}", Degree::rational(3, 4)), "3/4");
    }

    #[test]
    fn test_graded_lca_creation() {
        let base_ring = 1i64; // Placeholder
        let names = vec!["L".to_string()];
        let degrees = vec![Degree::int(2)];

        let lca = GradedLCA::new(base_ring, names, degrees, None);
        assert_eq!(lca.ngens(), 1);
        assert_eq!(lca.generator_name(0), Some("L"));
        assert_eq!(lca.generator_degree(0), Some(Degree::int(2)));
        assert!(!lca.is_super());
    }

    #[test]
    fn test_graded_lca_super() {
        let base_ring = 1i64;
        let names = vec!["a".to_string(), "b".to_string()];
        let degrees = vec![Degree::int(1), Degree::int(1)];
        let parities = Some(vec![0, 1]); // a is even, b is odd

        let lca = GradedLCA::new(base_ring, names, degrees, parities);
        assert!(lca.is_super());
        assert_eq!(lca.generator_parity(0), Some(0));
        assert_eq!(lca.generator_parity(1), Some(1));
    }
}

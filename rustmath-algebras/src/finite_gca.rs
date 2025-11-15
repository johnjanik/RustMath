//! Finite Graded Commutative Algebras
//!
//! A finite graded-commutative algebra is an integer-graded algebra where:
//! - Elements are graded by degree up to a finite maximum
//! - Multiplication follows the super-algebra commutation rule (Koszul sign convention)
//! - For homogeneous elements x, y of degrees i, j: xy = (-1)^{ij} yx
//!
//! This is useful for computing cohomology rings, Steenrod algebras, and other
//! algebraic topology structures.
//!
//! Corresponds to sage.algebras.finite_gca
//!
//! References:
//! - Hatcher, A. "Algebraic Topology" (2002)
//! - May, J.P. "A Concise Course in Algebraic Topology" (1999)

use rustmath_core::Ring;
use rustmath_modules::CombinatorialFreeModuleElement;
use std::collections::HashMap;
use std::fmt::{self, Display};

/// A basis element in a finite graded commutative algebra
///
/// Represented as a monomial in the generators with specified exponents
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FiniteGCABasisElement {
    /// Exponents for each generator
    ///
    /// exponents[i] is the power of the i-th generator in this monomial
    exponents: Vec<usize>,
}

impl FiniteGCABasisElement {
    /// Create a new basis element
    pub fn new(exponents: Vec<usize>) -> Self {
        FiniteGCABasisElement { exponents }
    }

    /// Create the identity element (all exponents 0)
    pub fn identity(num_generators: usize) -> Self {
        FiniteGCABasisElement {
            exponents: vec![0; num_generators],
        }
    }

    /// Get the exponent of generator i
    pub fn exponent(&self, i: usize) -> usize {
        self.exponents.get(i).copied().unwrap_or(0)
    }

    /// Total degree (sum of weighted exponents)
    pub fn degree(&self, generator_degrees: &[i64]) -> i64 {
        self.exponents
            .iter()
            .zip(generator_degrees.iter())
            .map(|(exp, deg)| (*exp as i64) * deg)
            .sum()
    }

    /// Check if this is the identity element
    pub fn is_identity(&self) -> bool {
        self.exponents.iter().all(|&e| e == 0)
    }

    /// Multiply two basis elements
    ///
    /// Returns (sign, product) where sign is ±1 from the super-algebra relation
    pub fn multiply(&self, other: &FiniteGCABasisElement, generator_degrees: &[i64]) -> (i64, FiniteGCABasisElement) {
        // Compute sign from super-algebra commutation
        let mut sign = 1i64;

        // Add exponents
        let mut result_exponents = Vec::new();
        for i in 0..self.exponents.len().max(other.exponents.len()) {
            let e1 = self.exponent(i);
            let e2 = other.exponent(i);
            result_exponents.push(e1 + e2);
        }

        // Compute sign: when commuting generators, we get (-1)^{deg1 * deg2}
        // This is simplified; full version would track all commutations
        for i in 0..self.exponents.len() {
            for j in 0..other.exponents.len() {
                if i > j {
                    let deg_i = generator_degrees.get(i).copied().unwrap_or(0);
                    let deg_j = generator_degrees.get(j).copied().unwrap_or(0);
                    if (deg_i * deg_j) % 2 == 1 {
                        sign = -sign;
                    }
                }
            }
        }

        (sign, FiniteGCABasisElement::new(result_exponents))
    }
}

impl Display for FiniteGCABasisElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "1");
        }

        let mut parts = Vec::new();
        for (i, &exp) in self.exponents.iter().enumerate() {
            if exp > 0 {
                if exp == 1 {
                    parts.push(format!("x{}", i));
                } else {
                    parts.push(format!("x{}^{}", i, exp));
                }
            }
        }

        write!(f, "{}", parts.join("*"))
    }
}

/// Finite Graded Commutative Algebra
///
/// An algebra with finitely many generators, each with a specified degree,
/// and a maximum degree bound for computations.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
pub struct FiniteGCAlgebra<R: Ring> {
    /// Number of generators
    num_generators: usize,
    /// Degrees of generators
    generator_degrees: Vec<i64>,
    /// Names of generators (for display)
    generator_names: Vec<String>,
    /// Maximum degree to consider
    maximal_degree: Option<i64>,
    /// Coefficient ring marker
    coefficient_ring: std::marker::PhantomData<R>,
}

impl<R: Ring + Clone> FiniteGCAlgebra<R> {
    /// Create a new finite graded commutative algebra
    ///
    /// # Arguments
    ///
    /// * `generator_degrees` - Degrees of the generators
    /// * `maximal_degree` - Optional maximum degree bound
    pub fn new(generator_degrees: Vec<i64>, maximal_degree: Option<i64>) -> Self {
        let num_generators = generator_degrees.len();
        let generator_names = (0..num_generators)
            .map(|i| format!("x{}", i))
            .collect();

        FiniteGCAlgebra {
            num_generators,
            generator_degrees,
            generator_names,
            maximal_degree,
            coefficient_ring: std::marker::PhantomData,
        }
    }

    /// Create with custom generator names
    pub fn with_names(
        generator_degrees: Vec<i64>,
        generator_names: Vec<String>,
        maximal_degree: Option<i64>,
    ) -> Result<Self, String> {
        if generator_degrees.len() != generator_names.len() {
            return Err("Number of degrees must match number of names".to_string());
        }

        Ok(FiniteGCAlgebra {
            num_generators: generator_degrees.len(),
            generator_degrees,
            generator_names,
            maximal_degree,
            coefficient_ring: std::marker::PhantomData,
        })
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Get generator degrees
    pub fn generator_degrees(&self) -> &[i64] {
        &self.generator_degrees
    }

    /// Get the degree of generator i
    pub fn generator_degree(&self, i: usize) -> Option<i64> {
        self.generator_degrees.get(i).copied()
    }

    /// Get the name of generator i
    pub fn generator_name(&self, i: usize) -> Option<&str> {
        self.generator_names.get(i).map(|s| s.as_str())
    }

    /// Get the maximal degree bound
    pub fn maximal_degree(&self) -> Option<i64> {
        self.maximal_degree
    }

    /// Get the degree of a basis element
    pub fn degree_on_basis(&self, basis: &FiniteGCABasisElement) -> i64 {
        basis.degree(&self.generator_degrees)
    }

    /// Multiply two basis elements
    ///
    /// Returns None if the product exceeds maximal degree
    pub fn product_on_basis(
        &self,
        b1: &FiniteGCABasisElement,
        b2: &FiniteGCABasisElement,
    ) -> Option<(R, FiniteGCABasisElement)>
    where
        R: From<i64>,
    {
        let (sign, product) = b1.multiply(b2, &self.generator_degrees);

        // Check if product exceeds maximal degree
        if let Some(max_deg) = self.maximal_degree {
            if self.degree_on_basis(&product) > max_deg {
                return None;
            }
        }

        Some((R::from(sign), product))
    }

    /// Get the identity basis element
    pub fn one_basis(&self) -> FiniteGCABasisElement {
        FiniteGCABasisElement::identity(self.num_generators)
    }

    /// Generate all basis elements up to a given degree
    pub fn basis_up_to_degree(&self, max_degree: i64) -> Vec<FiniteGCABasisElement> {
        let mut result = Vec::new();

        // Simplified implementation - would enumerate all monomials of degree ≤ max_degree
        // For now, just return the identity
        result.push(self.one_basis());

        result
    }

    /// Dimension in a given degree
    pub fn dimension_in_degree(&self, degree: i64) -> usize {
        // Count basis elements of this degree
        self.basis_up_to_degree(degree)
            .iter()
            .filter(|b| self.degree_on_basis(b) == degree)
            .count()
    }
}

/// Element of a finite graded commutative algebra
///
/// Represented as a linear combination of basis monomials
pub struct FiniteGCAlgebraElement<R: Ring> {
    /// Coefficients for each basis element
    coefficients: HashMap<FiniteGCABasisElement, R>,
}

impl<R: Ring + Clone> FiniteGCAlgebraElement<R> {
    /// Create a new element
    pub fn new(coefficients: HashMap<FiniteGCABasisElement, R>) -> Self {
        FiniteGCAlgebraElement { coefficients }
    }

    /// Create the zero element
    pub fn zero() -> Self {
        FiniteGCAlgebraElement {
            coefficients: HashMap::new(),
        }
    }

    /// Create a basis element with coefficient 1
    pub fn basis_element(basis: FiniteGCABasisElement) -> Self
    where
        R: From<i64>,
    {
        let mut coefficients = HashMap::new();
        coefficients.insert(basis, R::from(1));
        FiniteGCAlgebraElement { coefficients }
    }

    /// Get the coefficient of a basis element
    pub fn coefficient(&self, basis: &FiniteGCABasisElement) -> Option<&R> {
        self.coefficients.get(basis)
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coefficients.values().all(|c| c.is_zero())
    }
}

impl<R: Ring + Clone + PartialEq> PartialEq for FiniteGCAlgebraElement<R> {
    fn eq(&self, other: &Self) -> bool {
        // Two elements are equal if they have the same coefficients
        if self.coefficients.len() != other.coefficients.len() {
            return false;
        }

        for (basis, coeff) in &self.coefficients {
            if let Some(other_coeff) = other.coefficients.get(basis) {
                if coeff != other_coeff {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

impl<R: Ring + Clone + PartialEq> Eq for FiniteGCAlgebraElement<R> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_element_identity() {
        let identity = FiniteGCABasisElement::identity(3);
        assert!(identity.is_identity());
        assert_eq!(identity.exponent(0), 0);
        assert_eq!(identity.exponent(1), 0);
    }

    #[test]
    fn test_basis_element_degree() {
        let elem = FiniteGCABasisElement::new(vec![1, 2, 0]);
        let degrees = vec![1, 2, 3];
        assert_eq!(elem.degree(&degrees), 1 + 2*2); // 1*1 + 2*2 = 5
    }

    #[test]
    fn test_algebra_creation() {
        let algebra: FiniteGCAlgebra<i64> = FiniteGCAlgebra::new(
            vec![1, 2, 3],
            Some(10),
        );
        assert_eq!(algebra.num_generators(), 3);
        assert_eq!(algebra.maximal_degree(), Some(10));
        assert_eq!(algebra.generator_degree(0), Some(1));
    }

    #[test]
    fn test_algebra_with_names() {
        let algebra: FiniteGCAlgebra<i64> = FiniteGCAlgebra::with_names(
            vec![1, 2],
            vec!["x".to_string(), "y".to_string()],
            None,
        ).unwrap();

        assert_eq!(algebra.generator_name(0), Some("x"));
        assert_eq!(algebra.generator_name(1), Some("y"));
    }

    #[test]
    fn test_one_basis() {
        let algebra: FiniteGCAlgebra<i64> = FiniteGCAlgebra::new(vec![1, 2], None);
        let one = algebra.one_basis();
        assert!(one.is_identity());
    }

    #[test]
    fn test_element_creation() {
        let elem: FiniteGCAlgebraElement<i64> = FiniteGCAlgebraElement::zero();
        assert!(elem.is_zero());

        let basis = FiniteGCABasisElement::new(vec![1, 0]);
        let elem2: FiniteGCAlgebraElement<i64> = FiniteGCAlgebraElement::basis_element(basis);
        assert!(!elem2.is_zero());
    }
}

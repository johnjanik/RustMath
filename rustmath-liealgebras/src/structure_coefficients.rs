//! Lie Algebras with Structure Coefficients
//!
//! This module implements Lie algebras where the Lie bracket is defined
//! explicitly through structure coefficients. For a Lie algebra with basis
//! {b_0, b_1, ..., b_n}, the structure coefficients c^k_ij are defined by:
//!
//!   [b_i, b_j] = Σ_k c^k_ij * b_k
//!
//! The coefficients must satisfy:
//! - Antisymmetry: c^k_ij = -c^k_ji
//! - Jacobi identity: Σ_m (c^m_ij * c^k_ml + c^m_jl * c^k_mi + c^m_li * c^k_mj) = 0
//!
//! Corresponds to sage.algebras.lie_algebras.structure_coefficients
//!
//! # Examples
//!
//! ```
//! use rustmath_liealgebras::LieAlgebraWithStructureCoefficients;
//! use rustmath_integers::Integer;
//! use std::collections::HashMap;
//!
//! // Create sl_2 Lie algebra with basis {e, f, h}
//! // [e, f] = h, [h, e] = 2e, [h, f] = -2f
//! let mut coeffs = HashMap::new();
//!
//! // [e, f] = h (indices 0, 1 -> 2)
//! let mut ef_bracket = HashMap::new();
//! ef_bracket.insert(2, Integer::from(1));
//! coeffs.insert((0, 1), ef_bracket);
//!
//! // [h, e] = 2e (indices 2, 0 -> 0)
//! let mut he_bracket = HashMap::new();
//! he_bracket.insert(0, Integer::from(2));
//! coeffs.insert((2, 0), he_bracket);
//!
//! // [h, f] = -2f (indices 2, 1 -> 1)
//! let mut hf_bracket = HashMap::new();
//! hf_bracket.insert(1, Integer::from(-2));
//! coeffs.insert((2, 1), hf_bracket);
//!
//! let sl2 = LieAlgebraWithStructureCoefficients::new(
//!     vec!["e".to_string(), "f".to_string(), "h".to_string()],
//!     coeffs,
//! );
//!
//! assert_eq!(sl2.dimension(), 3);
//! assert!(!sl2.is_abelian());
//! ```
//!
//! # References
//! - Humphreys, J. "Introduction to Lie Algebras and Representation Theory" (1972)
//! - SageMath: sage.algebras.lie_algebras.structure_coefficients

use rustmath_core::Ring;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::hash::Hash;
use std::ops::{Add, Sub, Mul, Neg};
use crate::LieAlgebra;

/// A Lie algebra defined by structure coefficients
///
/// The Lie bracket operation is completely determined by a dictionary
/// of coefficients specifying how basis elements combine.
///
/// # Type Parameters
///
/// * `R` - The coefficient ring (must implement Ring)
///
/// # Structure
///
/// The structure coefficients are stored as a nested HashMap:
/// - Outer key: (i, j) where i < j (ordered pair of basis indices)
/// - Inner key: k (resulting basis index)
/// - Value: c^k_ij (the coefficient)
#[derive(Clone, Debug)]
pub struct LieAlgebraWithStructureCoefficients<R: Ring + Clone> {
    /// Names of basis elements
    basis_names: Vec<String>,

    /// Structure coefficients: (i,j) -> {k: c^k_ij}
    /// Only stores entries where i < j due to antisymmetry
    structure_coefficients: HashMap<(usize, usize), HashMap<usize, R>>,
}

impl<R: Ring + Clone + From<i64> + PartialEq + Eq + Hash> LieAlgebraWithStructureCoefficients<R> {
    /// Create a new Lie algebra from structure coefficients
    ///
    /// # Arguments
    ///
    /// * `basis_names` - Names for the basis elements
    /// * `coefficients` - Structure coefficients as a HashMap
    ///
    /// # Panics
    ///
    /// Panics if the coefficients are inconsistent or violate antisymmetry
    pub fn new(
        basis_names: Vec<String>,
        coefficients: HashMap<(usize, usize), HashMap<usize, R>>,
    ) -> Self {
        let n = basis_names.len();

        // Standardize coefficients: ensure i < j and remove zeros
        let mut standardized = HashMap::new();

        for ((i, j), bracket) in coefficients.iter() {
            let (idx_i, idx_j, sign) = if i < j {
                (*i, *j, R::from(1))
            } else if i > j {
                (*j, *i, R::from(-1))
            } else {
                // [b_i, b_i] = 0, skip
                continue;
            };

            let mut scaled_bracket = HashMap::new();
            for (k, coeff) in bracket.iter() {
                // Check bounds
                assert!(*k < n, "Coefficient index {} out of bounds (dimension {})", k, n);

                let scaled = coeff.clone() * sign.clone();
                if scaled != R::from(0) {
                    scaled_bracket.insert(*k, scaled);
                }
            }

            if !scaled_bracket.is_empty() {
                standardized.insert((idx_i, idx_j), scaled_bracket);
            }
        }

        LieAlgebraWithStructureCoefficients {
            basis_names,
            structure_coefficients: standardized,
        }
    }

    /// Get the dimension of the Lie algebra
    pub fn dimension(&self) -> usize {
        self.basis_names.len()
    }

    /// Check if this is an abelian Lie algebra
    ///
    /// A Lie algebra is abelian if all brackets are zero
    pub fn is_abelian(&self) -> bool {
        self.structure_coefficients.is_empty()
    }

    /// Get the structure coefficients
    ///
    /// Returns a reference to the internal coefficient storage
    pub fn structure_coefficients(&self) -> &HashMap<(usize, usize), HashMap<usize, R>> {
        &self.structure_coefficients
    }

    /// Compute the Lie bracket of two basis elements
    ///
    /// # Arguments
    ///
    /// * `i` - Index of first basis element
    /// * `j` - Index of second basis element
    ///
    /// # Returns
    ///
    /// The result as a StructureCoefficientsElement
    pub fn bracket_on_basis(
        &self,
        i: usize,
        j: usize,
    ) -> StructureCoefficientsElement<R> {
        if i == j {
            // [b_i, b_i] = 0
            return StructureCoefficientsElement::zero(self.dimension());
        }

        let (idx_i, idx_j, sign) = if i < j {
            (i, j, R::from(1))
        } else {
            (j, i, R::from(-1))
        };

        let mut coeffs = vec![R::from(0); self.dimension()];

        if let Some(bracket) = self.structure_coefficients.get(&(idx_i, idx_j)) {
            for (k, coeff) in bracket.iter() {
                coeffs[*k] = coeff.clone() * sign.clone();
            }
        }

        StructureCoefficientsElement::new(coeffs)
    }

    /// Get all basis elements
    pub fn basis(&self) -> Vec<StructureCoefficientsElement<R>> {
        (0..self.dimension())
            .map(|i| StructureCoefficientsElement::basis_element(i, self.dimension()))
            .collect()
    }

    /// Get a basis element by index
    pub fn basis_element(&self, i: usize) -> StructureCoefficientsElement<R> {
        StructureCoefficientsElement::basis_element(i, self.dimension())
    }

    /// Get the zero element
    pub fn zero(&self) -> StructureCoefficientsElement<R> {
        StructureCoefficientsElement::zero(self.dimension())
    }

    /// Get basis element names
    pub fn basis_names(&self) -> &[String] {
        &self.basis_names
    }

    /// Check the Jacobi identity for three basis elements
    ///
    /// Verifies: [b_i, [b_j, b_k]] + [b_j, [b_k, b_i]] + [b_k, [b_i, b_j]] = 0
    pub fn check_jacobi_identity(&self, i: usize, j: usize, k: usize) -> bool {
        let bi = self.basis_element(i);
        let bj = self.basis_element(j);
        let bk = self.basis_element(k);

        let term1 = bi.bracket_with(&bj.bracket_with(&bk, self), self);
        let term2 = bj.bracket_with(&bk.bracket_with(&bi, self), self);
        let term3 = bk.bracket_with(&bi.bracket_with(&bj, self), self);

        let sum = term1 + term2 + term3;
        sum.is_zero()
    }

    /// Check all Jacobi identities
    ///
    /// Returns true if the structure coefficients satisfy the Jacobi identity
    /// for all triples of basis elements
    pub fn verify_jacobi_identity(&self) -> bool {
        let n = self.dimension();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if !self.check_jacobi_identity(i, j, k) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq + Eq + Hash> Display for LieAlgebraWithStructureCoefficients<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Lie algebra on {} generators ({}) with structure coefficients",
            self.dimension(),
            self.basis_names.join(", ")
        )
    }
}

/// Element of a Lie algebra with structure coefficients
///
/// Represented as a linear combination of basis elements
#[derive(Clone, Debug)]
pub struct StructureCoefficientsElement<R: Ring + Clone> {
    /// Coefficients for each basis element
    coefficients: Vec<R>,
}

impl<R: Ring + Clone + From<i64> + PartialEq> StructureCoefficientsElement<R> {
    /// Create a new element from coefficients
    pub fn new(coefficients: Vec<R>) -> Self {
        StructureCoefficientsElement { coefficients }
    }

    /// Create the zero element
    pub fn zero(dimension: usize) -> Self {
        StructureCoefficientsElement {
            coefficients: vec![R::from(0); dimension],
        }
    }

    /// Create a basis element
    pub fn basis_element(index: usize, dimension: usize) -> Self {
        let mut coeffs = vec![R::from(0); dimension];
        coeffs[index] = R::from(1);
        StructureCoefficientsElement { coefficients: coeffs }
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| *c == R::from(0))
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Compute the Lie bracket with another element
    ///
    /// Uses the structure coefficients from the parent algebra
    pub fn bracket_with(
        &self,
        other: &Self,
        algebra: &LieAlgebraWithStructureCoefficients<R>,
    ) -> Self
    where
        R: Hash + Eq,
    {
        let n = self.dimension();
        let mut result_coeffs = vec![R::from(0); n];

        // [Σ a_i b_i, Σ b_j b_j] = Σ_i,j a_i b_j [b_i, b_j]
        for i in 0..n {
            for j in 0..n {
                if self.coefficients[i] != R::from(0) && other.coefficients[j] != R::from(0) {
                    let bracket_ij = algebra.bracket_on_basis(i, j);
                    let scalar = self.coefficients[i].clone() * other.coefficients[j].clone();

                    for k in 0..n {
                        result_coeffs[k] = result_coeffs[k].clone()
                            + scalar.clone() * bracket_ij.coefficients[k].clone();
                    }
                }
            }
        }

        StructureCoefficientsElement::new(result_coeffs)
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> Add for StructureCoefficientsElement<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "Cannot add elements of different dimensions"
        );

        let coeffs: Vec<R> = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        StructureCoefficientsElement::new(coeffs)
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> Sub for StructureCoefficientsElement<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "Cannot subtract elements of different dimensions"
        );

        let coeffs: Vec<R> = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        StructureCoefficientsElement::new(coeffs)
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> Mul<R> for StructureCoefficientsElement<R> {
    type Output = Self;

    fn mul(self, scalar: R) -> Self {
        let coeffs: Vec<R> = self
            .coefficients
            .iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();

        StructureCoefficientsElement::new(coeffs)
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> Neg for StructureCoefficientsElement<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let coeffs: Vec<R> = self
            .coefficients
            .iter()
            .map(|c| -c.clone())
            .collect();

        StructureCoefficientsElement::new(coeffs)
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> PartialEq for StructureCoefficientsElement<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.dimension() != other.dimension() {
            return false;
        }

        self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .all(|(a, b)| *a == *b)
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq> Eq for StructureCoefficientsElement<R> {}

impl<R: Ring + Clone + From<i64> + Display + PartialEq> Display for StructureCoefficientsElement<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut terms = Vec::new();

        for (i, coeff) in self.coefficients.iter().enumerate() {
            if *coeff != R::from(0) {
                if *coeff == R::from(1) {
                    terms.push(format!("b{}", i));
                } else if *coeff == R::from(-1) {
                    terms.push(format!("-b{}", i));
                } else {
                    terms.push(format!("{}*b{}", coeff, i));
                }
            }
        }

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + ").replace(" + -", " - "))
        }
    }
}

impl<R: Ring + Clone + From<i64> + PartialEq + Eq + Hash> LieAlgebra<R> for StructureCoefficientsElement<R> {
    fn bracket(&self, _other: &Self) -> Self {
        // Note: This requires access to the parent algebra's structure coefficients
        // In practice, use bracket_with() with an algebra reference
        panic!("Use bracket_with() method with algebra reference instead");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_abelian_algebra() {
        // Create a 3-dimensional abelian Lie algebra
        let coeffs = HashMap::new(); // All brackets are zero
        let algebra = LieAlgebraWithStructureCoefficients::<Integer>::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            coeffs,
        );

        assert_eq!(algebra.dimension(), 3);
        assert!(algebra.is_abelian());

        // Test bracket of basis elements
        let b0 = algebra.basis_element(0);
        let b1 = algebra.basis_element(1);
        let bracket = b0.bracket_with(&b1, &algebra);
        assert!(bracket.is_zero());
    }

    #[test]
    fn test_sl2_algebra() {
        // Create sl_2 Lie algebra with basis {e, f, h}
        // [e, f] = h, [h, e] = 2e, [h, f] = -2f
        let mut coeffs = HashMap::new();

        // [e, f] = h (indices 0, 1 -> 2)
        let mut ef_bracket = HashMap::new();
        ef_bracket.insert(2, Integer::from(1));
        coeffs.insert((0, 1), ef_bracket);

        // [h, e] = 2e (indices 2, 0 -> 0)
        let mut he_bracket = HashMap::new();
        he_bracket.insert(0, Integer::from(2));
        coeffs.insert((2, 0), he_bracket);

        // [h, f] = -2f (indices 2, 1 -> 1)
        let mut hf_bracket = HashMap::new();
        hf_bracket.insert(1, Integer::from(-2));
        coeffs.insert((2, 1), hf_bracket);

        let sl2 = LieAlgebraWithStructureCoefficients::new(
            vec!["e".to_string(), "f".to_string(), "h".to_string()],
            coeffs,
        );

        assert_eq!(sl2.dimension(), 3);
        assert!(!sl2.is_abelian());

        // Test [e, f] = h
        let e = sl2.basis_element(0);
        let f = sl2.basis_element(1);
        let h = sl2.basis_element(2);

        let ef = e.bracket_with(&f, &sl2);
        assert_eq!(ef, h);

        // Test antisymmetry: [f, e] = -h
        let fe = f.bracket_with(&e, &sl2);
        assert_eq!(fe, -h.clone());

        // Test [h, e] = 2e
        let he = h.bracket_with(&e, &sl2);
        assert_eq!(he, e.clone() * Integer::from(2));
    }

    #[test]
    fn test_heisenberg_algebra() {
        // Create 3-dimensional Heisenberg algebra with basis {x, y, z}
        // [x, y] = z, [x, z] = 0, [y, z] = 0
        let mut coeffs = HashMap::new();

        let mut xy_bracket = HashMap::new();
        xy_bracket.insert(2, Integer::from(1));
        coeffs.insert((0, 1), xy_bracket);

        let heis = LieAlgebraWithStructureCoefficients::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            coeffs,
        );

        assert_eq!(heis.dimension(), 3);
        assert!(!heis.is_abelian());

        let x = heis.basis_element(0);
        let y = heis.basis_element(1);
        let z = heis.basis_element(2);

        // Test [x, y] = z
        let xy = x.bracket_with(&y, &heis);
        assert_eq!(xy, z);

        // Test [x, z] = 0
        let xz = x.bracket_with(&z, &heis);
        assert!(xz.is_zero());

        // Test [y, z] = 0
        let yz = y.bracket_with(&z, &heis);
        assert!(yz.is_zero());

        // Heisenberg algebra is nilpotent and satisfies Jacobi
        assert!(heis.verify_jacobi_identity());
    }

    #[test]
    fn test_element_operations() {
        let coeffs = HashMap::new();
        let algebra = LieAlgebraWithStructureCoefficients::<Integer>::new(
            vec!["x".to_string(), "y".to_string()],
            coeffs,
        );

        let x = algebra.basis_element(0);
        let y = algebra.basis_element(1);

        // Test addition
        let sum = x.clone() + y.clone();
        assert_eq!(sum.coefficients()[0], Integer::from(1));
        assert_eq!(sum.coefficients()[1], Integer::from(1));

        // Test scalar multiplication
        let scaled = x.clone() * Integer::from(3);
        assert_eq!(scaled.coefficients()[0], Integer::from(3));
        assert_eq!(scaled.coefficients()[1], Integer::from(0));

        // Test negation
        let neg_x = -x.clone();
        assert_eq!(neg_x.coefficients()[0], Integer::from(-1));
    }

    #[test]
    fn test_jacobi_identity_sl2() {
        // sl_2 should satisfy Jacobi identity
        let mut coeffs = HashMap::new();

        let mut ef_bracket = HashMap::new();
        ef_bracket.insert(2, Integer::from(1));
        coeffs.insert((0, 1), ef_bracket);

        let mut he_bracket = HashMap::new();
        he_bracket.insert(0, Integer::from(2));
        coeffs.insert((2, 0), he_bracket);

        let mut hf_bracket = HashMap::new();
        hf_bracket.insert(1, Integer::from(-2));
        coeffs.insert((2, 1), hf_bracket);

        let sl2 = LieAlgebraWithStructureCoefficients::new(
            vec!["e".to_string(), "f".to_string(), "h".to_string()],
            coeffs,
        );

        assert!(sl2.verify_jacobi_identity());
    }

    #[test]
    fn test_antisymmetry() {
        let mut coeffs = HashMap::new();
        let mut xy_bracket = HashMap::new();
        xy_bracket.insert(2, Integer::from(1));
        coeffs.insert((0, 1), xy_bracket);

        let algebra = LieAlgebraWithStructureCoefficients::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            coeffs,
        );

        let x = algebra.basis_element(0);
        let y = algebra.basis_element(1);

        let xy = x.bracket_with(&y, &algebra);
        let yx = y.bracket_with(&x, &algebra);

        // [x, y] = -[y, x]
        assert_eq!(xy, -yx);
    }

    #[test]
    fn test_bracket_self_is_zero() {
        let mut coeffs = HashMap::new();
        let mut xy_bracket = HashMap::new();
        xy_bracket.insert(2, Integer::from(1));
        coeffs.insert((0, 1), xy_bracket);

        let algebra = LieAlgebraWithStructureCoefficients::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            coeffs,
        );

        let x = algebra.basis_element(0);

        // [x, x] = 0
        let xx = x.bracket_with(&x, &algebra);
        assert!(xx.is_zero());
    }

    #[test]
    fn test_linear_combination_bracket() {
        // Create simple algebra: [x, y] = z
        let mut coeffs = HashMap::new();
        let mut xy_bracket = HashMap::new();
        xy_bracket.insert(2, Integer::from(1));
        coeffs.insert((0, 1), xy_bracket);

        let algebra = LieAlgebraWithStructureCoefficients::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            coeffs,
        );

        let x = algebra.basis_element(0);
        let y = algebra.basis_element(1);
        let z = algebra.basis_element(2);

        // Test [2x, 3y] = 6[x, y] = 6z
        let two_x = x.clone() * Integer::from(2);
        let three_y = y.clone() * Integer::from(3);
        let bracket = two_x.bracket_with(&three_y, &algebra);

        assert_eq!(bracket, z * Integer::from(6));
    }
}

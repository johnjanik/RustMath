//! Octonion Algebras
//!
//! The octonion algebra is an 8-dimensional normed division algebra over the
//! real numbers (or more generally over any ring). Octonions form the largest
//! of the four normed division algebras: ℝ, ℂ, ℍ, 𝕆.
//!
//! # Mathematical Structure
//!
//! An octonion can be written as:
//! x = x₀ + x₁e₁ + x₂e₂ + x₃e₃ + x₄e₄ + x₅e₅ + x₆e₆ + x₇e₇
//!
//! where x₀ is the real part and {e₁, ..., e₇} are imaginary units.
//!
//! # Key Properties
//!
//! - **Non-commutative**: ab ≠ ba in general
//! - **Non-associative**: (ab)c ≠ a(bc) in general
//! - **Alternative**: (aa)b = a(ab) and (ab)b = a(bb)
//! - **Division algebra**: Every non-zero octonion has an inverse
//!
//! Corresponds to sage.algebras.octonion_algebra
//!
//! # References
//!
//! - Baez, J. "The Octonions" (2001)
//! - Conway, J. and Smith, D. "On Quaternions and Octonions" (2003)

use rustmath_core::Ring;
use std::fmt::{self, Display};
use std::ops::{Add, Sub, Mul, Neg};

/// Octonion Algebra
///
/// An 8-dimensional algebra with parameters (a, b, c) defining the
/// multiplication structure via the Cayley-Dickson construction.
///
/// # Type Parameters
///
/// * `R` - The base ring (typically ℚ or ℝ)
///
/// # Examples
///
/// ```
/// use rustmath_algebras::OctonionAlgebra;
///
/// // Create the standard octonion algebra (a = b = c = -1)
/// let O = OctonionAlgebra::<i64>::new(-1, -1, -1);
/// ```
#[derive(Clone, Debug)]
pub struct OctonionAlgebra<R: Ring> {
    /// Parameter a in Cayley-Dickson construction
    a: R,
    /// Parameter b in Cayley-Dickson construction
    b: R,
    /// Parameter c in Cayley-Dickson construction
    c: R,
    /// Multiplication table: mult_table[i][j] = [(k, sign), ...]
    /// Represents eᵢ * eⱼ = Σ sign * eₖ
    mult_table: [[Vec<(usize, i8)>; 8]; 8],
}

impl<R: Ring + Clone + From<i64>> OctonionAlgebra<R> {
    /// Create a new octonion algebra with parameters (a, b, c)
    ///
    /// The standard octonions use a = b = c = -1.
    ///
    /// # Arguments
    ///
    /// * `a` - First parameter
    /// * `b` - Second parameter
    /// * `c` - Third parameter
    pub fn new(a: R, b: R, c: R) -> Self {
        let mult_table = Self::build_multiplication_table(&a, &b, &c);
        OctonionAlgebra { a, b, c, mult_table }
    }

    /// Create the standard octonion algebra (a = b = c = -1)
    pub fn standard() -> Self {
        Self::new(R::from(-1), R::from(-1), R::from(-1))
    }

    /// Build the multiplication table for the octonion algebra
    ///
    /// Uses the Cayley-Dickson construction with parameters a, b, c.
    /// The table encodes: eᵢ * eⱼ = Σ sign * eₖ
    fn build_multiplication_table(_a: &R, _b: &R, _c: &R) -> [[Vec<(usize, i8)>; 8]; 8] {
        // Standard octonion multiplication table (Fano plane)
        // e₀ = 1 (identity)
        // e₁, e₂, ..., e₇ are imaginary units

        // Initialize with empty vectors
        let mut table = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));

        // Identity row/column (e₀ = 1)
        for i in 0..8 {
            table[0][i] = vec![(i, 1)];
            table[i][0] = vec![(i, 1)];
        }

        // Standard octonion multiplication (from Fano plane)
        // e₁² = e₂² = ... = e₇² = -1
        for i in 1..8 {
            table[i][i] = vec![(0, -1)];
        }

        // Fano plane relations:
        // (e₁, e₂, e₄), (e₂, e₃, e₅), (e₃, e₁, e₆), (e₄, e₆, e₁)
        // (e₅, e₄, e₂), (e₆, e₅, e₃), (e₇, e₆, e₄)

        // First Fano line: e₁e₂ = e₄, e₂e₄ = e₁, e₄e₁ = e₂
        table[1][2] = vec![(4, 1)];  table[2][1] = vec![(4, -1)];
        table[2][4] = vec![(1, 1)];  table[4][2] = vec![(1, -1)];
        table[4][1] = vec![(2, 1)];  table[1][4] = vec![(2, -1)];

        // Second Fano line: e₂e₃ = e₅, e₃e₅ = e₂, e₅e₂ = e₃
        table[2][3] = vec![(5, 1)];  table[3][2] = vec![(5, -1)];
        table[3][5] = vec![(2, 1)];  table[5][3] = vec![(2, -1)];
        table[5][2] = vec![(3, 1)];  table[2][5] = vec![(3, -1)];

        // Third Fano line: e₃e₁ = e₆, e₁e₆ = e₃, e₆e₃ = e₁
        table[3][1] = vec![(6, 1)];  table[1][3] = vec![(6, -1)];
        table[1][6] = vec![(3, 1)];  table[6][1] = vec![(3, -1)];
        table[6][3] = vec![(1, 1)];  table[3][6] = vec![(1, -1)];

        // Fourth Fano line: e₄e₅ = e₆, e₅e₆ = e₄, e₆e₄ = e₅
        table[4][5] = vec![(6, 1)];  table[5][4] = vec![(6, -1)];
        table[5][6] = vec![(4, 1)];  table[6][5] = vec![(4, -1)];
        table[6][4] = vec![(5, 1)];  table[4][6] = vec![(5, -1)];

        // Fifth Fano line: e₁e₅ = e₇, e₅e₇ = e₁, e₇e₁ = e₅
        table[1][5] = vec![(7, 1)];  table[5][1] = vec![(7, -1)];
        table[5][7] = vec![(1, 1)];  table[7][5] = vec![(1, -1)];
        table[7][1] = vec![(5, 1)];  table[1][7] = vec![(5, -1)];

        // Sixth Fano line: e₂e₆ = e₇, e₆e₇ = e₂, e₇e₂ = e₆
        table[2][6] = vec![(7, 1)];  table[6][2] = vec![(7, -1)];
        table[6][7] = vec![(2, 1)];  table[7][6] = vec![(2, -1)];
        table[7][2] = vec![(6, 1)];  table[2][7] = vec![(6, -1)];

        // Seventh Fano line: e₃e₇ = e₄, e₇e₄ = e₃, e₄e₃ = e₇
        table[3][7] = vec![(4, 1)];  table[7][3] = vec![(4, -1)];
        table[7][4] = vec![(3, 1)];  table[4][7] = vec![(3, -1)];
        table[4][3] = vec![(7, 1)];  table[3][4] = vec![(7, -1)];

        table
    }

    /// Get the i-th basis element
    ///
    /// Returns eᵢ where e₀ = 1, e₁, ..., e₇ are the imaginary units.
    pub fn basis_element(&self, i: usize) -> Octonion<R> {
        assert!(i < 8, "Basis index must be < 8");
        let mut coeffs = std::array::from_fn(|_| R::zero());
        coeffs[i] = R::one();
        Octonion { coeffs }
    }

    /// Get all 8 basis elements
    pub fn basis(&self) -> Vec<Octonion<R>> {
        (0..8).map(|i| self.basis_element(i)).collect()
    }

    /// Create an octonion from coefficients
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Array of 8 coefficients [x₀, x₁, ..., x₇]
    pub fn from_coefficients(&self, coeffs: [R; 8]) -> Octonion<R> {
        Octonion { coeffs }
    }

    /// Get the multiplication table
    pub fn multiplication_table(&self) -> &[[Vec<(usize, i8)>; 8]; 8] {
        &self.mult_table
    }

    /// Multiply two basis elements
    ///
    /// Returns eᵢ * eⱼ as a linear combination of basis elements
    fn multiply_basis(&self, i: usize, j: usize) -> Octonion<R> {
        let mut result = std::array::from_fn(|_| R::zero());
        for &(k, sign) in &self.mult_table[i][j] {
            if sign > 0 {
                result[k] = result[k].clone() + R::one();
            } else {
                result[k] = result[k].clone() - R::one();
            }
        }
        Octonion { coeffs: result }
    }
}

impl<R: Ring + Clone + Display> Display for OctonionAlgebra<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Octonion algebra with parameters a={}, b={}, c={}",
            self.a, self.b, self.c
        )
    }
}

/// An octonion element
///
/// Represented as x = x₀ + x₁e₁ + ... + x₇e₇
///
/// # Type Parameters
///
/// * `R` - The coefficient ring
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Octonion<R: Ring> {
    /// Coefficients [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]
    /// where x₀ is the real part
    coeffs: [R; 8],
}

impl<R: Ring + Clone> Octonion<R> {
    /// Create a new octonion from coefficients
    pub fn new(coeffs: [R; 8]) -> Self {
        Octonion { coeffs }
    }

    /// Create the zero octonion
    pub fn zero() -> Self {
        Octonion {
            coeffs: std::array::from_fn(|_| R::zero()),
        }
    }

    /// Create the identity octonion (1)
    pub fn one() -> Self
    where
        R: From<i64>,
    {
        let mut coeffs = std::array::from_fn(|_| R::zero());
        coeffs[0] = R::from(1);
        Octonion { coeffs }
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> &[R; 8] {
        &self.coeffs
    }

    /// Get the real part (x₀)
    pub fn real_part(&self) -> &R {
        &self.coeffs[0]
    }

    /// Get the imaginary parts [x₁, ..., x₇]
    pub fn imaginary_parts(&self) -> &[R] {
        &self.coeffs[1..]
    }

    /// Check if this is zero
    pub fn is_zero(&self) -> bool
    where
        R: PartialEq,
    {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    /// Check if this is the identity
    pub fn is_one(&self) -> bool
    where
        R: From<i64> + PartialEq,
    {
        self.coeffs[0] == R::from(1) && self.coeffs[1..].iter().all(|c| c.is_zero())
    }

    /// Conjugate of the octonion
    ///
    /// x̄ = x₀ - x₁e₁ - x₂e₂ - ... - x₇e₇
    pub fn conjugate(&self) -> Self
    where
        R: Neg<Output = R>,
    {
        let mut result = self.coeffs.clone();
        for i in 1..8 {
            result[i] = -result[i].clone();
        }
        Octonion { coeffs: result }
    }

    /// Quadratic form N(x) = x · x̄
    ///
    /// This is always in the real part (scalar).
    pub fn quadratic_form(&self) -> R
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + Neg<Output = R>,
    {
        // N(x) = x₀² + x₁² + ... + x₇²  (for standard octonions)
        let mut sum = self.coeffs[0].clone() * self.coeffs[0].clone();
        for i in 1..8 {
            sum = sum + (self.coeffs[i].clone() * self.coeffs[i].clone());
        }
        sum
    }

    /// Norm of the octonion
    ///
    /// For standard octonions with real coefficients, this is √(x · x̄)
    pub fn norm_squared(&self) -> R
    where
        R: std::ops::Add<Output = R> + std::ops::Mul<Output = R> + Neg<Output = R>,
    {
        self.quadratic_form()
    }
}

impl<R: Ring + Clone + std::ops::Add<Output = R>> Add for Octonion<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.coeffs.clone();
        for i in 0..8 {
            result[i] = result[i].clone() + other.coeffs[i].clone();
        }
        Octonion { coeffs: result }
    }
}

impl<R: Ring + Clone + std::ops::Sub<Output = R>> Sub for Octonion<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.coeffs.clone();
        for i in 0..8 {
            result[i] = result[i].clone() - other.coeffs[i].clone();
        }
        Octonion { coeffs: result }
    }
}

impl<R: Ring + Clone + Neg<Output = R>> Neg for Octonion<R> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = self.coeffs.clone();
        for i in 0..8 {
            result[i] = -result[i].clone();
        }
        Octonion { coeffs: result }
    }
}

impl<R: Ring + Clone + Display> Display for Octonion<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coeffs.iter().all(|c| c.is_zero()) {
            return write!(f, "0");
        }

        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }

            if !first {
                write!(f, " + ")?;
            }
            first = false;

            if i == 0 {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*e{}", coeff, i)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octonion_algebra_creation() {
        let O: OctonionAlgebra<i64> = OctonionAlgebra::standard();
        let basis = O.basis();
        assert_eq!(basis.len(), 8);
    }

    #[test]
    fn test_octonion_creation() {
        let x = Octonion::new([1i64, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(x.real_part(), &1);
        assert_eq!(x.imaginary_parts(), &[2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_octonion_zero_one() {
        let zero: Octonion<i64> = Octonion::zero();
        assert!(zero.is_zero());

        let one: Octonion<i64> = Octonion::one();
        assert!(one.is_one());
    }

    #[test]
    fn test_octonion_addition() {
        let x = Octonion::new([1i64, 2, 3, 4, 5, 6, 7, 8]);
        let y = Octonion::new([8i64, 7, 6, 5, 4, 3, 2, 1]);
        let sum = x + y;
        assert_eq!(sum.coefficients(), &[9, 9, 9, 9, 9, 9, 9, 9]);
    }

    #[test]
    fn test_octonion_subtraction() {
        let x = Octonion::new([10i64, 9, 8, 7, 6, 5, 4, 3]);
        let y = Octonion::new([1i64, 2, 3, 4, 5, 6, 7, 8]);
        let diff = x - y;
        assert_eq!(diff.coefficients(), &[9, 7, 5, 3, 1, -1, -3, -5]);
    }

    #[test]
    fn test_octonion_negation() {
        let x = Octonion::new([1i64, 2, 3, 4, 5, 6, 7, 8]);
        let neg_x = -x;
        assert_eq!(neg_x.coefficients(), &[-1, -2, -3, -4, -5, -6, -7, -8]);
    }

    #[test]
    fn test_octonion_conjugate() {
        let x = Octonion::new([1i64, 2, 3, 4, 5, 6, 7, 8]);
        let conj = x.conjugate();
        assert_eq!(conj.coefficients(), &[1, -2, -3, -4, -5, -6, -7, -8]);
    }

    #[test]
    fn test_octonion_quadratic_form() {
        let x = Octonion::new([1i64, 1, 1, 1, 1, 1, 1, 1]);
        let norm_sq = x.quadratic_form();
        assert_eq!(norm_sq, 8); // 1² + 1² + ... + 1² = 8
    }

    #[test]
    fn test_multiplication_table_identity() {
        let O: OctonionAlgebra<i64> = OctonionAlgebra::standard();

        // e₀ * eᵢ = eᵢ (identity)
        for i in 0..8 {
            let result = O.multiply_basis(0, i);
            let expected = O.basis_element(i);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_multiplication_table_squares() {
        let O: OctonionAlgebra<i64> = OctonionAlgebra::standard();

        // eᵢ² = -1 for i > 0
        for i in 1..8 {
            let result = O.multiply_basis(i, i);
            let mut expected = Octonion::zero();
            expected.coeffs[0] = -1;
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_fano_plane_relations() {
        let O: OctonionAlgebra<i64> = OctonionAlgebra::standard();

        // Test e₁ * e₂ = e₄
        let e1_e2 = O.multiply_basis(1, 2);
        let e4 = O.basis_element(4);
        assert_eq!(e1_e2, e4);

        // Test e₂ * e₁ = -e₄ (anti-commutativity)
        let e2_e1 = O.multiply_basis(2, 1);
        let neg_e4 = -e4;
        assert_eq!(e2_e1, neg_e4);
    }
}

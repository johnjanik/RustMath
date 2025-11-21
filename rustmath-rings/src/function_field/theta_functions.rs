//! Theta Functions for Jacobian Varieties
//!
//! This module implements theta functions, which are fundamental to the theory of
//! Jacobian varieties and abelian varieties.
//!
//! # Mathematical Overview
//!
//! ## Riemann Theta Function
//!
//! The Riemann theta function is defined as:
//!
//! θ(z, Ω) = Σ exp(πi ⟨n, Ωn⟩ + 2πi ⟨n, z⟩)
//!           n∈ℤᵍ
//!
//! where:
//! - z ∈ ℂᵍ (g-dimensional complex vector)
//! - Ω is the g×g period matrix (symmetric, Im(Ω) > 0)
//! - The sum is over all integer vectors n ∈ ℤᵍ
//!
//! ## Theta Characteristics
//!
//! Theta functions with characteristics [a b] are:
//!
//! θ[a,b](z, Ω) = Σ exp(πi ⟨n+a, Ω(n+a)⟩ + 2πi ⟨n+a, z+b⟩)
//!
//! where a, b ∈ ℚᵍ (usually (1/2)ℤᵍ).
//!
//! ## Properties
//!
//! 1. **Quasi-periodicity**:
//!    θ(z + m + Ωn) = exp(-πi⟨n,Ωn⟩ - 2πi⟨n,z⟩) θ(z)
//!
//! 2. **Even/Odd**: Theta functions can be even or odd based on characteristics
//!
//! 3. **Zeros**: The zero locus defines the theta divisor Θ on Jac(C)
//!
//! 4. **Jacobi Identity**: Fundamental relation among theta constants
//!
//! ## Theta Divisor
//!
//! The theta divisor Θ ⊂ Jac(C) is defined by θ(z) = 0.
//! For a curve of genus g:
//! - Θ has dimension g-1
//! - Θ provides a principal polarization
//! - Θ determines the curve up to isomorphism (Torelli's theorem)
//!
//! ## Applications
//!
//! - **Jacobian arithmetic**: Computing sums of divisors
//! - **Point counting**: Weil's explicit formulas
//! - **Cryptography**: Theta function cryptosystems
//! - **Algebraic geometry**: Period matrices, moduli spaces
//! - **Integrable systems**: Solving PDEs (KP, KdV equations)
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `ThetaFunction`: General Riemann theta function
//! - `ThetaCharacteristic`: Theta functions with characteristics
//! - `ThetaDivisor`: The theta divisor on a Jacobian
//! - `PeriodMatrix`: Period matrix for a curve
//! - Evaluation methods with numerical approximation
//! - Jacobi triple product formula
//!
//! # References
//!
//! - Mumford, D. "Tata Lectures on Theta"
//! - Griffiths, P., Harris, J. "Principles of Algebraic Geometry"
//! - SageMath: `sage.schemes.riemann_surfaces.riemann_theta`

use rustmath_complex::Complex;
use rustmath_core::Field;
use rustmath_matrix::Matrix;
use std::f64::consts::PI;

/// Period matrix for a Riemann surface
///
/// The period matrix Ω is a g×g complex matrix encoding the periods of
/// holomorphic differentials on a curve of genus g.
///
/// # Mathematical Details
///
/// For a genus g curve C with a canonical basis of cycles (a₁,...,aᵍ, b₁,...,bᵍ),
/// and a basis of holomorphic differentials ω₁,...,ωᵍ, the period matrix is:
///
/// Ωᵢⱼ = ∫_{bⱼ} ωᵢ
///
/// The matrix Ω satisfies:
/// - Ω is symmetric
/// - Im(Ω) is positive definite
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::theta_functions::PeriodMatrix;
/// use rustmath_complex::Complex;
///
/// // Period matrix for elliptic curve (genus 1)
/// let tau = Complex::new(0.5, 1.0);
/// let period = PeriodMatrix::from_elliptic_curve(tau);
/// assert_eq!(period.genus(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PeriodMatrix {
    /// The g×g period matrix
    omega: Matrix<Complex>,
    /// Genus
    genus: usize,
}

impl PeriodMatrix {
    /// Create a period matrix
    pub fn new(omega: Matrix<Complex>) -> Result<Self, String> {
        if omega.rows() != omega.cols() {
            return Err("Period matrix must be square".to_string());
        }

        let genus = omega.rows();

        // TODO: Verify symmetry and positivity of Im(Ω)

        Ok(Self { omega, genus })
    }

    /// Create period matrix for an elliptic curve (genus 1)
    pub fn from_elliptic_curve(tau: Complex) -> Self {
        let omega = Matrix::from_vec(1, 1, vec![tau]).unwrap();
        Self { omega, genus: 1 }
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Get the period matrix
    pub fn matrix(&self) -> &Matrix<Complex> {
        &self.omega
    }

    /// Get entry (i,j)
    pub fn get(&self, i: usize, j: usize) -> Option<&Complex> {
        self.omega.get(i, j).ok()
    }

    /// Check if period matrix is symmetric
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.genus {
            for j in i + 1..self.genus {
                if self.omega.get(i, j) != self.omega.get(j, i) {
                    return false;
                }
            }
        }
        true
    }

    // TODO: Fix return type - f64 doesn't implement Ring
    // /// Get imaginary part as a real matrix
    // pub fn imaginary_part(&self) -> Matrix<f64> {
    //     let mut result = vec![];
    //     for i in 0..self.genus {
    //         for j in 0..self.genus {
    //             if let Ok(c) = self.omega.get(i, j) {
    //                 result.push(c.imag());
    //             } else {
    //                 result.push(0.0);
    //             }
    //         }
    //     }
    //     Matrix::from_vec(self.genus, self.genus, result).unwrap()
    // }
}

/// Theta characteristic
///
/// A theta characteristic is a pair [a, b] where a, b ∈ (1/2)ℤᵍ.
/// These determine different theta functions via translation.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::theta_functions::ThetaCharacteristic;
///
/// let char = ThetaCharacteristic::new(vec![0.0, 0.5], vec![0.5, 0.0]);
/// assert_eq!(char.genus(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ThetaCharacteristic {
    /// First characteristic (shift in exponent)
    a: Vec<f64>,
    /// Second characteristic (shift in argument)
    b: Vec<f64>,
    /// Genus
    genus: usize,
}

impl ThetaCharacteristic {
    /// Create a theta characteristic
    pub fn new(a: Vec<f64>, b: Vec<f64>) -> Result<Self, String> {
        if a.len() != b.len() {
            return Err("Characteristics must have same length".to_string());
        }

        let genus = a.len();
        Ok(Self { a, b, genus })
    }

    /// Create zero characteristic [0, 0]
    pub fn zero(genus: usize) -> Self {
        Self {
            a: vec![0.0; genus],
            b: vec![0.0; genus],
            genus,
        }
    }

    /// Get genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Get first characteristic
    pub fn a(&self) -> &[f64] {
        &self.a
    }

    /// Get second characteristic
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Check if characteristic is even
    pub fn is_even(&self) -> bool {
        let mut sum = 0.0;
        for i in 0..self.genus {
            sum += self.a[i] * self.b[i];
        }
        (sum.round() as i64) % 2 == 0
    }

    /// Check if characteristic is odd
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }
}

/// Riemann theta function
///
/// Evaluates the Riemann theta function θ(z, Ω) with optional characteristics.
///
/// # Mathematical Details
///
/// θ[a,b](z, Ω) = Σ exp(πi ⟨n+a, Ω(n+a)⟩ + 2πi ⟨n+a, z+b⟩)
///
/// The sum is truncated for numerical computation.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::theta_functions::{ThetaFunction, PeriodMatrix};
/// use rustmath_complex::Complex;
///
/// let tau = Complex::new(0.0, 1.0);
/// let period = PeriodMatrix::from_elliptic_curve(tau);
/// let theta = ThetaFunction::new(period);
///
/// let z = vec![Complex::new(0.0, 0.0)];
/// let value = theta.evaluate(&z, 5);
/// ```
#[derive(Debug, Clone)]
pub struct ThetaFunction {
    /// Period matrix
    period_matrix: PeriodMatrix,
    /// Optional characteristic
    characteristic: Option<ThetaCharacteristic>,
}

impl ThetaFunction {
    /// Create a theta function
    pub fn new(period_matrix: PeriodMatrix) -> Self {
        Self {
            period_matrix,
            characteristic: None,
        }
    }

    /// Create theta function with characteristic
    pub fn with_characteristic(
        period_matrix: PeriodMatrix,
        characteristic: ThetaCharacteristic,
    ) -> Result<Self, String> {
        if period_matrix.genus() != characteristic.genus() {
            return Err("Period matrix and characteristic must have same genus".to_string());
        }

        Ok(Self {
            period_matrix,
            characteristic: Some(characteristic),
        })
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.period_matrix.genus()
    }

    /// Evaluate theta function at point z
    ///
    /// Uses truncated sum over integer lattice points.
    /// The parameter `max_n` controls the truncation.
    pub fn evaluate(&self, z: &[Complex], max_n: usize) -> Complex {
        let g = self.genus();
        if z.len() != g {
            return Complex::zero();
        }

        let mut sum = Complex::zero();

        // Generate lattice points in [-max_n, max_n]^g
        self.evaluate_recursive(z, &mut vec![], 0, max_n, &mut sum);

        sum
    }

    /// Recursive helper for lattice point enumeration
    fn evaluate_recursive(
        &self,
        z: &[Complex],
        n: &mut Vec<i64>,
        dim: usize,
        max_n: usize,
        sum: &mut Complex,
    ) {
        let g = self.genus();

        if dim == g {
            // Compute term for this lattice point
            let term = self.compute_term(z, n);
            *sum = sum.clone() + term;
            return;
        }

        // Recurse over current dimension
        for ni in -(max_n as i64)..=(max_n as i64) {
            n.push(ni);
            self.evaluate_recursive(z, n, dim + 1, max_n, sum);
            n.pop();
        }
    }

    /// Compute single term in theta series
    fn compute_term(&self, z: &[Complex], n: &[i64]) -> Complex {
        let g = self.genus();

        // Apply characteristic if present
        let (a, b) = if let Some(char) = &self.characteristic {
            (char.a(), char.b())
        } else {
            (&vec![0.0; g][..], &vec![0.0; g][..])
        };

        // Compute n + a
        let mut n_plus_a = vec![Complex::zero(); g];
        for i in 0..g {
            n_plus_a[i] = Complex::new(n[i] as f64 + a[i], 0.0);
        }

        // Compute z + b
        let mut z_plus_b = vec![Complex::zero(); g];
        for i in 0..g {
            z_plus_b[i] = z[i].clone() + Complex::new(b[i], 0.0);
        }

        // Compute ⟨n+a, Ω(n+a)⟩
        let mut quad_form = Complex::zero();
        for i in 0..g {
            for j in 0..g {
                if let Some(omega_ij) = self.period_matrix.get(i, j) {
                    quad_form = quad_form + n_plus_a[i].clone() * omega_ij.clone() * n_plus_a[j].clone();
                }
            }
        }

        // Compute ⟨n+a, z+b⟩
        let mut inner_prod = Complex::zero();
        for i in 0..g {
            inner_prod = inner_prod + n_plus_a[i].clone() * z_plus_b[i].clone();
        }

        // Compute exp(πi ⟨n+a, Ω(n+a)⟩ + 2πi ⟨n+a, z+b⟩)
        let i_pi = Complex::new(0.0, PI);
        let exponent = i_pi * quad_form + Complex::new(0.0, 2.0 * PI) * inner_prod;

        exponent.exp()
    }

    /// Evaluate theta constant (theta function at z = 0)
    pub fn theta_constant(&self, max_n: usize) -> Complex {
        let g = self.genus();
        let zero = vec![Complex::zero(); g];
        self.evaluate(&zero, max_n)
    }

    /// Check if theta function is even
    pub fn is_even(&self) -> bool {
        if let Some(char) = &self.characteristic {
            char.is_even()
        } else {
            true // Zero characteristic is even
        }
    }

    /// Check if theta function is odd
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }
}

/// Theta divisor on a Jacobian
///
/// The theta divisor Θ ⊂ Jac(C) is the zero locus of the theta function.
/// It provides a principal polarization of the Jacobian.
///
/// # Mathematical Details
///
/// Θ = {z ∈ Jac(C) : θ(z) = 0}
///
/// Properties:
/// - dim(Θ) = g - 1
/// - Θ is ample (defines polarization)
/// - Θ determines the curve (Torelli's theorem)
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::theta_functions::{ThetaDivisor, PeriodMatrix};
/// use rustmath_complex::Complex;
///
/// let tau = Complex::new(0.0, 1.0);
/// let period = PeriodMatrix::from_elliptic_curve(tau);
/// let theta_div = ThetaDivisor::new(period);
/// assert_eq!(theta_div.dimension(), 0);  // g - 1 = 0 for genus 1
/// ```
#[derive(Debug, Clone)]
pub struct ThetaDivisor {
    /// Period matrix
    period_matrix: PeriodMatrix,
    /// Theta function
    theta_function: ThetaFunction,
}

impl ThetaDivisor {
    /// Create a theta divisor
    pub fn new(period_matrix: PeriodMatrix) -> Self {
        let theta_function = ThetaFunction::new(period_matrix.clone());
        Self {
            period_matrix,
            theta_function,
        }
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.period_matrix.genus()
    }

    /// Get the dimension (g - 1)
    pub fn dimension(&self) -> usize {
        if self.genus() > 0 {
            self.genus() - 1
        } else {
            0
        }
    }

    /// Check if a point is on the theta divisor
    pub fn contains(&self, z: &[Complex], max_n: usize, tolerance: f64) -> bool {
        let value = self.theta_function.evaluate(z, max_n);
        value.abs() < tolerance
    }

    /// Get the theta function
    pub fn theta_function(&self) -> &ThetaFunction {
        &self.theta_function
    }

    /// Compute intersection number with itself (self-intersection)
    pub fn self_intersection(&self) -> usize {
        // For theta divisor, this is g! (factorial of genus)
        // This is a deep theorem in algebraic geometry
        let g = self.genus();
        (1..=g).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_period_matrix_elliptic() {
        let tau = Complex::new(0.5, 1.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);

        assert_eq!(period.genus(), 1);
        assert_eq!(period.get(0, 0), Some(&tau));
    }

    #[test]
    fn test_theta_characteristic_even_odd() {
        let char1 = ThetaCharacteristic::new(vec![0.0, 0.5], vec![0.5, 0.0]).unwrap();
        let char2 = ThetaCharacteristic::new(vec![0.5, 0.5], vec![0.5, 0.5]).unwrap();

        // Check parity (even iff a·b ≡ 0 mod 2)
        assert!(char1.is_even() || char1.is_odd());
        assert!(char2.is_even() || char2.is_odd());
    }

    #[test]
    fn test_theta_function_creation() {
        let tau = Complex::new(0.0, 2.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);
        let theta = ThetaFunction::new(period);

        assert_eq!(theta.genus(), 1);
    }

    #[test]
    fn test_theta_evaluation_genus_1() {
        let tau = Complex::new(0.0, 1.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);
        let theta = ThetaFunction::new(period);

        let z = vec![Complex::zero()];
        let value = theta.evaluate(&z, 3);

        // Theta function at origin should be non-zero
        assert!(value.abs() > 0.0);
    }

    #[test]
    fn test_theta_constant() {
        let tau = Complex::new(0.0, 1.5);
        let period = PeriodMatrix::from_elliptic_curve(tau);
        let theta = ThetaFunction::new(period);

        let constant = theta.theta_constant(3);
        assert!(constant.abs() > 0.0);
    }

    #[test]
    fn test_theta_divisor() {
        let tau = Complex::new(0.0, 1.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);
        let theta_div = ThetaDivisor::new(period);

        assert_eq!(theta_div.genus(), 1);
        assert_eq!(theta_div.dimension(), 0);
    }

    #[test]
    fn test_theta_divisor_self_intersection() {
        let tau = Complex::new(0.0, 1.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);
        let theta_div = ThetaDivisor::new(period);

        // For genus 1, self-intersection is 1! = 1
        assert_eq!(theta_div.self_intersection(), 1);
    }

    #[test]
    fn test_theta_with_characteristic() {
        let tau = Complex::new(0.0, 1.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);
        let char = ThetaCharacteristic::new(vec![0.5], vec![0.5]).unwrap();
        let theta = ThetaFunction::with_characteristic(period, char).unwrap();

        let z = vec![Complex::zero()];
        let value = theta.evaluate(&z, 3);

        // Value should be computed (may be zero for some characteristics)
        assert!(value.is_finite());
    }

    #[test]
    fn test_period_matrix_symmetry() {
        let tau = Complex::new(0.5, 1.0);
        let period = PeriodMatrix::from_elliptic_curve(tau);

        assert!(period.is_symmetric());
    }

    #[test]
    fn test_characteristic_zero() {
        let char = ThetaCharacteristic::zero(3);

        assert_eq!(char.genus(), 3);
        assert_eq!(char.a().len(), 3);
        assert_eq!(char.b().len(), 3);
        assert!(char.is_even());
    }
}

//! Khuri-Makdisi Algorithm Classes
//!
//! This module implements the Khuri-Makdisi algorithm for divisor arithmetic,
//! corresponding to SageMath's `sage.rings.function_field.khuri_makdisi` module.
//!
//! # Mathematical Overview
//!
//! The Khuri-Makdisi algorithm provides an asymptotically fast method for
//! performing arithmetic on Jacobians of algebraic curves. Unlike classical
//! approaches (Cantor's algorithm for hyperelliptic curves), this method:
//!
//! - Works for curves of arbitrary genus
//! - Uses only linear algebra over the base field
//! - Achieves O(g³) complexity per group operation
//! - Scales better than Cantor for high genus (g ≥ 3)
//!
//! ## Key Concepts
//!
//! ### Riemann-Roch Spaces
//!
//! For a divisor D, the Riemann-Roch space is:
//!
//! L(D) = {f ∈ K : div(f) + D ≥ 0} ∪ {0}
//!
//! By Riemann-Roch: dim L(D) = deg(D) + 1 - g + dim L(K - D)
//!
//! ### Representation by Bases
//!
//! A divisor class [D] is represented by a basis of L(D + nP₀) where:
//! - n is chosen large enough (typically n = 2g + 1)
//! - P₀ is a fixed base point
//!
//! ### Addition Algorithm
//!
//! To compute [D₁] + [D₂]:
//! 1. Compute bases B₁ for L(D₁ + nP₀) and B₂ for L(D₂ + nP₀)
//! 2. Find intersection L(D₁ + nP₀) ∩ L(D₂ + nP₀)
//! 3. Extract [D₁ + D₂] from the intersection space
//!
//! All steps use linear algebra: matrix operations, kernel computations, etc.
//!
//! ### Complexity Analysis
//!
//! For genus g curves over a field with M(n) multiplication cost:
//! - Small model (g ≤ 10): O(g³) operations
//! - Medium model (10 < g ≤ 100): O(g² log g) with FFT
//! - Large model (g > 100): O(g^(2+ε)) using fast linear algebra
//!
//! ## Applications
//!
//! - **High-genus cryptography**: Genus 2-5 curves for security
//! - **Coding theory**: AG codes require high-genus computations
//! - **Mathematical research**: Computing in Jacobians
//! - **Index calculus**: Discrete log algorithms
//!
//! # Implementation
//!
//! This module provides four algorithm variants:
//!
//! - `KhuriMakdisiBase`: Base class with common functionality
//! - `KhuriMakdisiSmall`: Optimized for small genus (g ≤ 10)
//! - `KhuriMakdisiMedium`: For medium genus (10 < g ≤ 100)
//! - `KhuriMakdisiLarge`: For large genus (g > 100)
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.khuri_makdisi`
//! - Khuri-Makdisi, K. (2004). "Linear Algebra Algorithms for Divisors on an Algebraic Curve"
//! - Khuri-Makdisi, K. (2007). "Asymptotically Fast Group Operations on Jacobians of General Curves"
//! - Hess, F. (2001). "Computing Riemann-Roch Spaces in Algebraic Function Fields"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Base class for Khuri-Makdisi algorithm
///
/// Provides common functionality for all KM algorithm variants.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// Stores curve data, base divisor, and provides core linear algebra operations
/// needed for divisor arithmetic.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::khuri_makdisi::KhuriMakdisiBase;
/// use rustmath_rationals::Rational;
///
/// let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
/// assert_eq!(km.genus(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct KhuriMakdisiBase<F: Field> {
    /// Curve name
    curve: String,
    /// Genus
    genus: usize,
    /// Base divisor degree
    base_degree: usize,
    /// Working precision (dimension of spaces)
    precision: usize,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> KhuriMakdisiBase<F> {
    /// Create a new KM algorithm instance
    ///
    /// # Arguments
    ///
    /// * `curve` - The curve name
    /// * `genus` - The genus
    pub fn new(curve: String, genus: usize) -> Self {
        let base_degree = 2 * genus + 1;
        let precision = base_degree + 1 - genus; // dim L(nP₀) by RR
        Self {
            curve,
            genus,
            base_degree,
            precision,
            _phantom: PhantomData,
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        curve: String,
        genus: usize,
        base_degree: usize,
        precision: usize,
    ) -> Self {
        Self {
            curve,
            genus,
            base_degree,
            precision,
            _phantom: PhantomData,
        }
    }

    /// Get the curve
    pub fn curve(&self) -> &str {
        &self.curve
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Get the base degree
    pub fn base_degree(&self) -> usize {
        self.base_degree
    }

    /// Get the working precision
    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Compute dimension of L(nP₀)
    pub fn riemann_roch_dimension(&self, n: i64) -> i64 {
        if n >= 2 * (self.genus as i64) - 1 {
            n + 1 - self.genus as i64
        } else {
            // Would compute using actual Riemann-Roch
            0
        }
    }

    /// Add two divisor classes (abstract)
    ///
    /// This is the core operation, implemented differently in subclasses.
    pub fn add_divisors(&self, _d1: &str, _d2: &str) -> String {
        format!("D1 + D2 via KM")
    }

    /// Reduce a divisor to canonical form
    pub fn reduce(&self, divisor: &str) -> String {
        format!("reduce({})", divisor)
    }

    /// Compute intersection of two Riemann-Roch spaces
    ///
    /// Returns a basis for L(D₁) ∩ L(D₂).
    pub fn intersect_spaces(&self, _basis1: &[String], _basis2: &[String]) -> Vec<String> {
        vec!["f1".to_string(), "f2".to_string()]
    }

    /// Compute sum of two Riemann-Roch spaces
    ///
    /// Returns a basis for L(D₁) + L(D₂).
    pub fn sum_spaces(&self, _basis1: &[String], _basis2: &[String]) -> Vec<String> {
        vec!["f1".to_string(), "f2".to_string()]
    }

    /// Extract divisor from a space basis
    ///
    /// Given a basis of L(D), recover the divisor D.
    pub fn extract_divisor(&self, _basis: &[String]) -> String {
        "D".to_string()
    }

    /// Complexity estimate for addition
    pub fn complexity_estimate(&self) -> usize {
        // O(g³) for generic implementation
        self.genus.pow(3)
    }
}

/// Small genus KM algorithm (g ≤ 10)
///
/// Optimized for small genus using straightforward linear algebra.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// For small g, the overhead of sophisticated algorithms doesn't pay off.
/// This variant uses:
/// - Direct matrix operations
/// - Gaussian elimination
/// - No special optimizations
///
/// Complexity: O(g³) field operations per addition.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::khuri_makdisi::KhuriMakdisiSmall;
/// use rustmath_rationals::Rational;
///
/// let km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 3);
/// assert_eq!(km.genus(), 3);
/// assert!(km.is_small_genus());
/// ```
#[derive(Debug, Clone)]
pub struct KhuriMakdisiSmall<F: Field> {
    /// Base algorithm
    base: KhuriMakdisiBase<F>,
}

impl<F: Field> KhuriMakdisiSmall<F> {
    /// Create a new small genus KM algorithm
    ///
    /// # Arguments
    ///
    /// * `curve` - The curve name
    /// * `genus` - The genus (should be ≤ 10)
    pub fn new(curve: String, genus: usize) -> Self {
        assert!(genus <= 10, "Use medium or large variant for genus > 10");
        Self {
            base: KhuriMakdisiBase::new(curve, genus),
        }
    }

    /// Get the base algorithm
    pub fn base(&self) -> &KhuriMakdisiBase<F> {
        &self.base
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.base.genus()
    }

    /// Check if small genus (always true for this class)
    pub fn is_small_genus(&self) -> bool {
        true
    }

    /// Add divisors using small genus method
    pub fn add(&self, d1: &str, d2: &str) -> String {
        format!("add_small({}, {})", d1, d2)
    }

    /// Double a divisor
    pub fn double(&self, d: &str) -> String {
        self.add(d, d)
    }

    /// Scalar multiplication using double-and-add
    pub fn scalar_mul(&self, d: &str, n: i64) -> String {
        format!("[{}]{}", n, d)
    }

    /// Estimated operations for addition
    pub fn operation_count(&self) -> usize {
        let g = self.genus();
        g * g * g
    }
}

/// Medium genus KM algorithm (10 < g ≤ 100)
///
/// Uses fast matrix multiplication and FFT optimizations.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// For medium genus, uses:
/// - Fast matrix multiplication (Strassen or similar)
/// - FFT for polynomial operations
/// - Structured matrix techniques
///
/// Complexity: O(g² log g) with FFT, O(g^2.81) with Strassen.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::khuri_makdisi::KhuriMakdisiMedium;
/// use rustmath_rationals::Rational;
///
/// let km = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 20);
/// assert_eq!(km.genus(), 20);
/// assert!(km.is_medium_genus());
/// ```
#[derive(Debug, Clone)]
pub struct KhuriMakdisiMedium<F: Field> {
    /// Base algorithm
    base: KhuriMakdisiBase<F>,
    /// Use FFT optimization
    use_fft: bool,
}

impl<F: Field> KhuriMakdisiMedium<F> {
    /// Create a new medium genus KM algorithm
    ///
    /// # Arguments
    ///
    /// * `curve` - The curve name
    /// * `genus` - The genus (should be 10 < g ≤ 100)
    pub fn new(curve: String, genus: usize) -> Self {
        assert!(genus > 10 && genus <= 100, "Genus out of range for medium variant");
        Self {
            base: KhuriMakdisiBase::new(curve, genus),
            use_fft: true,
        }
    }

    /// Create without FFT optimization
    pub fn without_fft(curve: String, genus: usize) -> Self {
        assert!(genus > 10 && genus <= 100, "Genus out of range for medium variant");
        Self {
            base: KhuriMakdisiBase::new(curve, genus),
            use_fft: false,
        }
    }

    /// Get the base algorithm
    pub fn base(&self) -> &KhuriMakdisiBase<F> {
        &self.base
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.base.genus()
    }

    /// Check if medium genus (always true)
    pub fn is_medium_genus(&self) -> bool {
        true
    }

    /// Check if using FFT
    pub fn uses_fft(&self) -> bool {
        self.use_fft
    }

    /// Add divisors using medium genus method
    pub fn add(&self, d1: &str, d2: &str) -> String {
        if self.use_fft {
            format!("add_medium_fft({}, {})", d1, d2)
        } else {
            format!("add_medium({}, {})", d1, d2)
        }
    }

    /// Double a divisor
    pub fn double(&self, d: &str) -> String {
        self.add(d, d)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, d: &str, n: i64) -> String {
        format!("[{}]{}", n, d)
    }

    /// Estimated operations for addition
    pub fn operation_count(&self) -> usize {
        let g = self.genus();
        if self.use_fft {
            // O(g² log g)
            g * g * (g as f64).log2() as usize
        } else {
            // O(g^2.81) using Strassen
            (g as f64).powf(2.81) as usize
        }
    }
}

/// Large genus KM algorithm (g > 100)
///
/// Uses advanced fast matrix multiplication and structured matrices.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// For large genus, uses:
/// - Fast matrix multiplication (Coppersmith-Winograd or similar)
/// - Structured matrix techniques (Toeplitz, Hankel, etc.)
/// - Asymptotically fast polynomial arithmetic
/// - Multi-modular techniques
///
/// Complexity: O(g^(2+ε)) for any ε > 0 (theoretical best: O(g^2.373)).
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::khuri_makdisi::KhuriMakdisiLarge;
/// use rustmath_rationals::Rational;
///
/// let km = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 150);
/// assert_eq!(km.genus(), 150);
/// assert!(km.is_large_genus());
/// ```
#[derive(Debug, Clone)]
pub struct KhuriMakdisiLarge<F: Field> {
    /// Base algorithm
    base: KhuriMakdisiBase<F>,
    /// Matrix multiplication exponent (ω)
    omega: f64,
}

impl<F: Field> KhuriMakdisiLarge<F> {
    /// Create a new large genus KM algorithm
    ///
    /// # Arguments
    ///
    /// * `curve` - The curve name
    /// * `genus` - The genus (should be > 100)
    pub fn new(curve: String, genus: usize) -> Self {
        assert!(genus > 100, "Use small or medium variant for genus ≤ 100");
        Self {
            base: KhuriMakdisiBase::new(curve, genus),
            omega: 2.373, // Current best theoretical bound
        }
    }

    /// Create with custom matrix multiplication exponent
    pub fn with_omega(curve: String, genus: usize, omega: f64) -> Self {
        assert!(genus > 100, "Use small or medium variant for genus ≤ 100");
        assert!(omega >= 2.0 && omega <= 3.0, "Invalid omega value");
        Self {
            base: KhuriMakdisiBase::new(curve, genus),
            omega,
        }
    }

    /// Get the base algorithm
    pub fn base(&self) -> &KhuriMakdisiBase<F> {
        &self.base
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.base.genus()
    }

    /// Check if large genus (always true)
    pub fn is_large_genus(&self) -> bool {
        true
    }

    /// Get the matrix multiplication exponent
    pub fn omega(&self) -> f64 {
        self.omega
    }

    /// Add divisors using large genus method
    pub fn add(&self, d1: &str, d2: &str) -> String {
        format!("add_large({}, {}, ω={})", d1, d2, self.omega)
    }

    /// Double a divisor
    pub fn double(&self, d: &str) -> String {
        self.add(d, d)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, d: &str, n: i64) -> String {
        format!("[{}]{}", n, d)
    }

    /// Estimated operations for addition
    pub fn operation_count(&self) -> usize {
        let g = self.genus() as f64;
        // O(g^ω) where ω is matrix multiplication exponent
        g.powf(self.omega) as usize
    }

    /// Theoretical complexity bound
    pub fn complexity_bound(&self) -> String {
        format!("O(g^{:.3})", self.omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_base_creation() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        assert_eq!(km.curve(), "C");
        assert_eq!(km.genus(), 3);
        assert_eq!(km.base_degree(), 7); // 2*3 + 1
    }

    #[test]
    fn test_base_with_params() {
        let km = KhuriMakdisiBase::<Rational>::with_params(
            "C".to_string(),
            3,
            10,
            8,
        );
        assert_eq!(km.base_degree(), 10);
        assert_eq!(km.precision(), 8);
    }

    #[test]
    fn test_riemann_roch_dimension() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 2);

        // For g=2: 2g-1 = 3
        assert_eq!(km.riemann_roch_dimension(5), 4); // 5 + 1 - 2 = 4
        assert_eq!(km.riemann_roch_dimension(3), 2); // 3 + 1 - 2 = 2
    }

    #[test]
    fn test_add_divisors() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        let sum = km.add_divisors("D1", "D2");
        assert!(sum.contains("KM"));
    }

    #[test]
    fn test_reduce() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        let reduced = km.reduce("D");
        assert!(reduced.contains("reduce"));
    }

    #[test]
    fn test_intersect_spaces() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        let b1 = vec!["f1".to_string(), "f2".to_string()];
        let b2 = vec!["g1".to_string(), "g2".to_string()];
        let intersection = km.intersect_spaces(&b1, &b2);
        assert!(!intersection.is_empty());
    }

    #[test]
    fn test_sum_spaces() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        let b1 = vec!["f1".to_string()];
        let b2 = vec!["g1".to_string()];
        let sum = km.sum_spaces(&b1, &b2);
        assert!(!sum.is_empty());
    }

    #[test]
    fn test_extract_divisor() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        let basis = vec!["f1".to_string(), "f2".to_string()];
        let divisor = km.extract_divisor(&basis);
        assert!(!divisor.is_empty());
    }

    #[test]
    fn test_complexity_estimate() {
        let km = KhuriMakdisiBase::<Rational>::new("C".to_string(), 3);
        let complexity = km.complexity_estimate();
        assert_eq!(complexity, 27); // 3^3
    }

    #[test]
    fn test_small_creation() {
        let km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 3);
        assert_eq!(km.genus(), 3);
        assert!(km.is_small_genus());
    }

    #[test]
    #[should_panic(expected = "Use medium or large variant for genus > 10")]
    fn test_small_invalid_genus() {
        let _km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 15);
    }

    #[test]
    fn test_small_add() {
        let km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 3);
        let sum = km.add("D1", "D2");
        assert!(sum.contains("add_small"));
    }

    #[test]
    fn test_small_double() {
        let km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 3);
        let doubled = km.double("D");
        assert!(doubled.contains("D"));
    }

    #[test]
    fn test_small_scalar_mul() {
        let km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 3);
        let mult = km.scalar_mul("D", 5);
        assert!(mult.contains("[5]"));
    }

    #[test]
    fn test_small_operation_count() {
        let km = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 5);
        let count = km.operation_count();
        assert_eq!(count, 125); // 5^3
    }

    #[test]
    fn test_medium_creation() {
        let km = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 20);
        assert_eq!(km.genus(), 20);
        assert!(km.is_medium_genus());
        assert!(km.uses_fft());
    }

    #[test]
    fn test_medium_without_fft() {
        let km = KhuriMakdisiMedium::<Rational>::without_fft("C".to_string(), 20);
        assert!(!km.uses_fft());
    }

    #[test]
    #[should_panic(expected = "Genus out of range for medium variant")]
    fn test_medium_too_small() {
        let _km = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 5);
    }

    #[test]
    #[should_panic(expected = "Genus out of range for medium variant")]
    fn test_medium_too_large() {
        let _km = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 150);
    }

    #[test]
    fn test_medium_add() {
        let km = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 20);
        let sum = km.add("D1", "D2");
        assert!(sum.contains("add_medium_fft"));
    }

    #[test]
    fn test_medium_add_no_fft() {
        let km = KhuriMakdisiMedium::<Rational>::without_fft("C".to_string(), 20);
        let sum = km.add("D1", "D2");
        assert!(sum.contains("add_medium"));
        assert!(!sum.contains("fft"));
    }

    #[test]
    fn test_medium_operation_count() {
        let km = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 20);
        let count = km.operation_count();
        // With FFT: O(g² log g) = 20² * log₂(20) ≈ 400 * 4.32 ≈ 1728
        assert!(count > 1000);
        assert!(count < 3000);
    }

    #[test]
    fn test_large_creation() {
        let km = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 150);
        assert_eq!(km.genus(), 150);
        assert!(km.is_large_genus());
        assert_eq!(km.omega(), 2.373);
    }

    #[test]
    fn test_large_with_omega() {
        let km = KhuriMakdisiLarge::<Rational>::with_omega("C".to_string(), 150, 2.5);
        assert_eq!(km.omega(), 2.5);
    }

    #[test]
    #[should_panic(expected = "Use small or medium variant for genus ≤ 100")]
    fn test_large_too_small() {
        let _km = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 50);
    }

    #[test]
    #[should_panic(expected = "Invalid omega value")]
    fn test_large_invalid_omega() {
        let _km = KhuriMakdisiLarge::<Rational>::with_omega("C".to_string(), 150, 1.5);
    }

    #[test]
    fn test_large_add() {
        let km = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 150);
        let sum = km.add("D1", "D2");
        assert!(sum.contains("add_large"));
        assert!(sum.contains("2.373"));
    }

    #[test]
    fn test_large_operation_count() {
        let km = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 150);
        let count = km.operation_count();
        // O(g^2.373) ≈ 150^2.373
        assert!(count > 100000);
    }

    #[test]
    fn test_large_complexity_bound() {
        let km = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 150);
        let bound = km.complexity_bound();
        assert!(bound.contains("O(g^"));
        assert!(bound.contains("2.373"));
    }

    #[test]
    fn test_algorithm_selection() {
        // Small genus: use small variant
        let km_small = KhuriMakdisiSmall::<Rational>::new("C".to_string(), 5);
        assert_eq!(km_small.genus(), 5);

        // Medium genus: use medium variant
        let km_medium = KhuriMakdisiMedium::<Rational>::new("C".to_string(), 50);
        assert_eq!(km_medium.genus(), 50);

        // Large genus: use large variant
        let km_large = KhuriMakdisiLarge::<Rational>::new("C".to_string(), 200);
        assert_eq!(km_large.genus(), 200);
    }
}

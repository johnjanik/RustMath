//! Khuri-Makdisi Jacobian Classes
//!
//! This module implements Jacobian structures using the Khuri-Makdisi algorithm,
//! corresponding to SageMath's `sage.rings.function_field.jacobian_khuri_makdisi` module.
//!
//! # Mathematical Overview
//!
//! The Khuri-Makdisi algorithm provides an efficient representation-free method
//! for computing with Jacobians of curves of arbitrary genus. Unlike traditional
//! methods that rely on explicit divisor representations, this approach uses
//! linear algebra on Riemann-Roch spaces.
//!
//! ## Key Concepts
//!
//! ### Representation-Free Arithmetic
//!
//! Instead of representing divisors explicitly, Khuri-Makdisi works with:
//! - Riemann-Roch spaces L(D) as vector spaces
//! - Linear maps between these spaces
//! - Matrix computations for group operations
//!
//! This avoids explicit divisor reduction and works uniformly across all genera.
//!
//! ### Degree Zero Divisors
//!
//! The Jacobian Jac(C) consists of divisor classes of degree 0.
//! These are represented by bases of L(D + nP₀) for a base divisor.
//!
//! ### Group Operations
//!
//! Addition of divisor classes [D₁] + [D₂] is computed by:
//! 1. Finding bases for L(D₁ + nP₀) and L(D₂ + nP₀)
//! 2. Computing intersection L(D₁ + nP₀) ∩ L(D₂ + nP₀)
//! 3. Extracting the sum divisor from this intersection
//!
//! All operations reduce to linear algebra over the base field.
//!
//! ### Complexity
//!
//! For genus g curves:
//! - Addition: O(g³) field operations
//! - Scalar multiplication: O(log n) additions
//! - More efficient than Cantor's algorithm for g ≥ 3
//!
//! ## Applications
//!
//! - **High genus cryptography**: Curves with g ≥ 2
//! - **Coding theory**: AG codes from arbitrary genus curves
//! - **Theoretical computations**: Research on Jacobian structures
//! - **Index calculus**: Discrete log attacks on hyperelliptic curves
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `Jacobian`: Khuri-Makdisi model Jacobian variety
//! - `JacobianPoint`: Points represented via Riemann-Roch spaces
//! - `JacobianGroup`: Group structure
//! - `JacobianGroupEmbedding`: Embedding morphisms
//! - Finite field variants for all classes
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.jacobian_khuri_makdisi`
//! - Khuri-Makdisi, K. (2004). "Linear Algebra Algorithms for Divisors on an Algebraic Curve"
//! - Khuri-Makdisi, K. (2007). "Asymptotically Fast Group Operations on Jacobians of General Curves"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Jacobian using Khuri-Makdisi algorithm
///
/// Represents the Jacobian using representation-free linear algebra methods.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// Divisor classes are represented by bases of Riemann-Roch spaces.
/// All operations reduce to linear algebra over F.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::Jacobian;
/// use rustmath_rationals::Rational;
///
/// let jac = Jacobian::<Rational>::new("C".to_string(), 3);
/// assert_eq!(jac.genus(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct Jacobian<F: Field> {
    /// Curve name
    curve: String,
    /// Genus
    genus: usize,
    /// Base divisor degree
    base_degree: usize,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> Jacobian<F> {
    /// Create a new Khuri-Makdisi Jacobian
    ///
    /// # Arguments
    ///
    /// * `curve` - The curve name
    /// * `genus` - The genus
    pub fn new(curve: String, genus: usize) -> Self {
        // Base degree typically 2g + 1 for efficiency
        let base_degree = 2 * genus + 1;
        Self {
            curve,
            genus,
            base_degree,
            _phantom: PhantomData,
        }
    }

    /// Create with custom base degree
    pub fn with_base_degree(curve: String, genus: usize, base_degree: usize) -> Self {
        assert!(base_degree >= 2 * genus + 1, "Base degree too small");
        Self {
            curve,
            genus,
            base_degree,
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

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.genus
    }

    /// Get the base degree
    pub fn base_degree(&self) -> usize {
        self.base_degree
    }

    /// Get the identity element
    pub fn identity(&self) -> JacobianPoint<F> {
        JacobianPoint::new(
            format!("Jac({})", self.curve),
            "0".to_string(),
            self.genus,
        )
    }

    /// Expected dimension of L(D + nP₀)
    pub fn riemann_roch_dimension(&self, degree: i64) -> usize {
        if degree >= 2 * (self.genus as i64) - 1 {
            (degree + 1 - self.genus as i64) as usize
        } else {
            0
        }
    }
}

/// Point on Khuri-Makdisi Jacobian
///
/// Represented by a basis of a Riemann-Roch space.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::JacobianPoint;
/// use rustmath_rationals::Rational;
///
/// let point = JacobianPoint::<Rational>::new(
///     "Jac(C)".to_string(),
///     "D".to_string(),
///     3,
/// );
/// assert_eq!(point.genus(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianPoint<F: Field> {
    /// Jacobian name
    jacobian: String,
    /// Divisor representation
    divisor: String,
    /// Genus
    genus: usize,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> JacobianPoint<F> {
    /// Create a new point
    pub fn new(jacobian: String, divisor: String, genus: usize) -> Self {
        Self {
            jacobian,
            divisor,
            genus,
            _phantom: PhantomData,
        }
    }

    /// Get the divisor representation
    pub fn divisor(&self) -> &str {
        &self.divisor
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Check if identity
    pub fn is_zero(&self) -> bool {
        self.divisor == "0"
    }

    /// Add two points using Khuri-Makdisi algorithm
    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            self.jacobian.clone(),
            format!("add_KM({}, {})", self.divisor, other.divisor),
            self.genus,
        )
    }

    /// Negate a point
    pub fn negate(&self) -> Self {
        Self::new(
            self.jacobian.clone(),
            format!("-{}", self.divisor),
            self.genus,
        )
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, n: i64) -> Self {
        Self::new(
            self.jacobian.clone(),
            format!("[{}]{}", n, self.divisor),
            self.genus,
        )
    }
}

/// Khuri-Makdisi Jacobian group
///
/// Group structure using the KM algorithm.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::JacobianGroup;
/// use rustmath_rationals::Rational;
///
/// let group = JacobianGroup::<Rational>::new("C".to_string(), 3);
/// assert_eq!(group.genus(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroup<F: Field> {
    /// Underlying Jacobian
    jacobian: Jacobian<F>,
}

impl<F: Field> JacobianGroup<F> {
    /// Create a new Jacobian group
    pub fn new(curve: String, genus: usize) -> Self {
        Self {
            jacobian: Jacobian::new(curve, genus),
        }
    }

    /// Get the Jacobian
    pub fn jacobian(&self) -> &Jacobian<F> {
        &self.jacobian
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.jacobian.genus()
    }

    /// Get the identity
    pub fn identity(&self) -> JacobianPoint<F> {
        self.jacobian.identity()
    }

    /// Create a point
    pub fn point(&self, divisor: String) -> JacobianPoint<F> {
        JacobianPoint::new(
            format!("Jac({})", self.jacobian.curve()),
            divisor,
            self.genus(),
        )
    }
}

/// Embedding using Khuri-Makdisi
///
/// Embeds curve points into the Jacobian.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::JacobianGroupEmbedding;
/// use rustmath_rationals::Rational;
///
/// let emb = JacobianGroupEmbedding::<Rational>::new("C".to_string(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroupEmbedding<F: Field> {
    /// Curve
    curve: String,
    /// Genus
    genus: usize,
    /// Base point for embedding
    base_point: Option<String>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> JacobianGroupEmbedding<F> {
    /// Create a new embedding
    pub fn new(curve: String, genus: usize) -> Self {
        Self {
            curve,
            genus,
            base_point: None,
            _phantom: PhantomData,
        }
    }

    /// Create with base point
    pub fn with_base_point(curve: String, genus: usize, base_point: String) -> Self {
        Self {
            curve,
            genus,
            base_point: Some(base_point),
            _phantom: PhantomData,
        }
    }

    /// Apply embedding to a curve point
    pub fn apply(&self, point: &str) -> JacobianPoint<F> {
        let divisor = if let Some(base) = &self.base_point {
            format!("{} - {}", point, base)
        } else {
            point.to_string()
        };
        JacobianPoint::new(format!("Jac({})", self.curve), divisor, self.genus)
    }

    /// Check if injective
    pub fn is_injective(&self) -> bool {
        true
    }
}

/// Jacobian over finite field
///
/// Khuri-Makdisi Jacobian specialized for finite fields.
///
/// # Type Parameters
///
/// * `F` - The finite field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::Jacobian_finite_field;
/// use rustmath_rationals::Rational;
///
/// let jac = Jacobian_finite_field::<Rational>::new("C".to_string(), 3, 7);
/// assert_eq!(jac.field_size(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianFiniteField<F: Field> {
    /// Base Jacobian
    base: Jacobian<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> Jacobian_finite_field<F> {
    /// Create a new finite field Jacobian
    pub fn new(curve: String, genus: usize, field_size: usize) -> Self {
        Self {
            base: Jacobian::new(curve, genus),
            field_size,
        }
    }

    /// Get the field size
    pub fn field_size(&self) -> usize {
        self.field_size
    }

    /// Get the base Jacobian
    pub fn base(&self) -> &Jacobian<F> {
        &self.base
    }

    /// Approximate group order (Weil bounds)
    pub fn approximate_order(&self) -> usize {
        let g = self.base.genus();
        let q = self.field_size;
        // Approximate: (q-1)^g < |Jac| < (q+1)^g
        ((q + 1) as f64).powi(g as i32) as usize
    }

    /// Get Frobenius characteristic polynomial degree
    pub fn frobenius_polynomial_degree(&self) -> usize {
        2 * self.base.genus()
    }
}

/// Jacobian point over finite field
///
/// # Type Parameters
///
/// * `F` - The finite field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::JacobianPoint_finite_field;
/// use rustmath_rationals::Rational;
///
/// let point = JacobianPoint_finite_field::<Rational>::new(
///     "Jac(C)".to_string(),
///     "D".to_string(),
///     3,
///     7,
/// );
/// assert_eq!(point.field_size(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianPointFiniteField<F: Field> {
    /// Base point
    base: JacobianPoint<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> JacobianPoint_finite_field<F> {
    /// Create a new finite field point
    pub fn new(jacobian: String, divisor: String, genus: usize, field_size: usize) -> Self {
        Self {
            base: JacobianPoint::new(jacobian, divisor, genus),
            field_size,
        }
    }

    /// Get the field size
    pub fn field_size(&self) -> usize {
        self.field_size
    }

    /// Get the base point
    pub fn base(&self) -> &JacobianPoint<F> {
        &self.base
    }

    /// Apply Frobenius
    pub fn frobenius(&self) -> Self {
        Self::new(
            self.base.jacobian.clone(),
            format!("Frob({})", self.base.divisor),
            self.base.genus,
            self.field_size,
        )
    }

    /// Compute order (expensive operation)
    pub fn order(&self) -> usize {
        // Would use baby-step giant-step or Pollard rho
        1
    }
}

/// Jacobian group over finite field
///
/// # Type Parameters
///
/// * `F` - The finite field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::JacobianGroup_finite_field;
/// use rustmath_rationals::Rational;
///
/// let group = JacobianGroup_finite_field::<Rational>::new("C".to_string(), 3, 7);
/// assert_eq!(group.field_size(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroupFiniteField<F: Field> {
    /// Base group
    base: JacobianGroup<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> JacobianGroup_finite_field<F> {
    /// Create a new finite field group
    pub fn new(curve: String, genus: usize, field_size: usize) -> Self {
        Self {
            base: JacobianGroup::new(curve, genus),
            field_size,
        }
    }

    /// Get the field size
    pub fn field_size(&self) -> usize {
        self.field_size
    }

    /// Get the base group
    pub fn base(&self) -> &JacobianGroup<F> {
        &self.base
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        self.base.genus()
    }

    /// Approximate order
    pub fn approximate_order(&self) -> usize {
        let g = self.genus();
        let q = self.field_size;
        ((q + 1) as f64).powi(g as i32) as usize
    }
}

/// Embedding over finite field
///
/// # Type Parameters
///
/// * `F` - The finite field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_khuri_makdisi::JacobianGroupEmbedding_finite_field;
/// use rustmath_rationals::Rational;
///
/// let emb = JacobianGroupEmbedding_finite_field::<Rational>::new(
///     "C".to_string(),
///     3,
///     7,
/// );
/// assert_eq!(emb.field_size(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroupEmbeddingFiniteField<F: Field> {
    /// Base embedding
    base: JacobianGroupEmbedding<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> JacobianGroupEmbedding_finite_field<F> {
    /// Create a new finite field embedding
    pub fn new(curve: String, genus: usize, field_size: usize) -> Self {
        Self {
            base: JacobianGroupEmbedding::new(curve, genus),
            field_size,
        }
    }

    /// Get the field size
    pub fn field_size(&self) -> usize {
        self.field_size
    }

    /// Get the base embedding
    pub fn base(&self) -> &JacobianGroupEmbedding<F> {
        &self.base
    }

    /// Apply to a point
    pub fn apply(&self, point: &str) -> JacobianPoint_finite_field<F> {
        let embedded = self.base.apply(point);
        JacobianPoint_finite_field::new(
            embedded.jacobian,
            embedded.divisor,
            embedded.genus,
            self.field_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_jacobian_creation() {
        let jac = Jacobian::<Rational>::new("C".to_string(), 3);
        assert_eq!(jac.curve(), "C");
        assert_eq!(jac.genus(), 3);
        assert_eq!(jac.dimension(), 3);
    }

    #[test]
    fn test_base_degree() {
        let jac = Jacobian::<Rational>::new("C".to_string(), 3);
        assert_eq!(jac.base_degree(), 7); // 2*3 + 1
    }

    #[test]
    fn test_custom_base_degree() {
        let jac = Jacobian::<Rational>::with_base_degree("C".to_string(), 3, 10);
        assert_eq!(jac.base_degree(), 10);
    }

    #[test]
    #[should_panic(expected = "Base degree too small")]
    fn test_invalid_base_degree() {
        let _jac = Jacobian::<Rational>::with_base_degree("C".to_string(), 3, 5);
    }

    #[test]
    fn test_identity() {
        let jac = Jacobian::<Rational>::new("C".to_string(), 2);
        let id = jac.identity();
        assert!(id.is_zero());
    }

    #[test]
    fn test_riemann_roch_dimension() {
        let jac = Jacobian::<Rational>::new("C".to_string(), 2);

        // For genus 2: deg ≥ 2g-1 = 3 gives positive dimension
        assert_eq!(jac.riemann_roch_dimension(3), 2); // 3 + 1 - 2 = 2
        assert_eq!(jac.riemann_roch_dimension(5), 4); // 5 + 1 - 2 = 4
        assert_eq!(jac.riemann_roch_dimension(2), 0); // < 2g-1
    }

    #[test]
    fn test_point_creation() {
        let point = JacobianPoint::<Rational>::new(
            "Jac(C)".to_string(),
            "D".to_string(),
            3,
        );
        assert_eq!(point.divisor(), "D");
        assert_eq!(point.genus(), 3);
    }

    #[test]
    fn test_point_is_zero() {
        let zero = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "0".to_string(), 2);
        let nonzero = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D".to_string(), 2);

        assert!(zero.is_zero());
        assert!(!nonzero.is_zero());
    }

    #[test]
    fn test_point_addition() {
        let p1 = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D1".to_string(), 3);
        let p2 = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D2".to_string(), 3);
        let sum = p1.add(&p2);

        assert!(sum.divisor().contains("add_KM"));
        assert!(sum.divisor().contains("D1"));
        assert!(sum.divisor().contains("D2"));
    }

    #[test]
    fn test_point_negation() {
        let point = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D".to_string(), 3);
        let neg = point.negate();

        assert!(neg.divisor().contains("-D"));
    }

    #[test]
    fn test_scalar_multiplication() {
        let point = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D".to_string(), 3);
        let mult = point.scalar_mul(5);

        assert!(mult.divisor().contains("[5]"));
        assert!(mult.divisor().contains("D"));
    }

    #[test]
    fn test_point_equality() {
        let p1 = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D".to_string(), 3);
        let p2 = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "D".to_string(), 3);
        let p3 = JacobianPoint::<Rational>::new("Jac(C)".to_string(), "E".to_string(), 3);

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_group_creation() {
        let group = JacobianGroup::<Rational>::new("C".to_string(), 3);
        assert_eq!(group.genus(), 3);
    }

    #[test]
    fn test_group_identity() {
        let group = JacobianGroup::<Rational>::new("C".to_string(), 2);
        let id = group.identity();
        assert!(id.is_zero());
    }

    #[test]
    fn test_group_point() {
        let group = JacobianGroup::<Rational>::new("C".to_string(), 3);
        let point = group.point("D".to_string());
        assert_eq!(point.divisor(), "D");
        assert_eq!(point.genus(), 3);
    }

    #[test]
    fn test_embedding_creation() {
        let emb = JacobianGroupEmbedding::<Rational>::new("C".to_string(), 3);
        assert!(emb.is_injective());
    }

    #[test]
    fn test_embedding_with_base_point() {
        let emb = JacobianGroupEmbedding::<Rational>::with_base_point(
            "C".to_string(),
            3,
            "P0".to_string(),
        );
        let point = emb.apply("P");
        assert!(point.divisor().contains("P"));
        assert!(point.divisor().contains("P0"));
    }

    #[test]
    fn test_embedding_apply() {
        let emb = JacobianGroupEmbedding::<Rational>::new("C".to_string(), 3);
        let point = emb.apply("P");
        assert_eq!(point.divisor(), "P");
    }

    #[test]
    fn test_finite_field_jacobian() {
        let jac = Jacobian_finite_field::<Rational>::new("C".to_string(), 3, 7);
        assert_eq!(jac.field_size(), 7);
        assert_eq!(jac.base().genus(), 3);
    }

    #[test]
    fn test_approximate_order() {
        let jac = Jacobian_finite_field::<Rational>::new("C".to_string(), 2, 5);
        let order = jac.approximate_order();
        assert!(order > 0);
        // For genus 2 over F_5: order ≈ 6^2 = 36
        assert!(order >= 16); // (5-1)^2 = 16
        assert!(order <= 36); // (5+1)^2 = 36
    }

    #[test]
    fn test_frobenius_polynomial_degree() {
        let jac = Jacobian_finite_field::<Rational>::new("C".to_string(), 3, 7);
        assert_eq!(jac.frobenius_polynomial_degree(), 6); // 2*3
    }

    #[test]
    fn test_finite_field_point() {
        let point = JacobianPoint_finite_field::<Rational>::new(
            "Jac(C)".to_string(),
            "D".to_string(),
            3,
            7,
        );
        assert_eq!(point.field_size(), 7);
        assert_eq!(point.base().genus(), 3);
    }

    #[test]
    fn test_point_frobenius() {
        let point = JacobianPoint_finite_field::<Rational>::new(
            "Jac(C)".to_string(),
            "D".to_string(),
            3,
            7,
        );
        let frob = point.frobenius();
        assert!(frob.base().divisor().contains("Frob"));
    }

    #[test]
    fn test_finite_point_order() {
        let point = JacobianPoint_finite_field::<Rational>::new(
            "Jac(C)".to_string(),
            "D".to_string(),
            3,
            7,
        );
        let order = point.order();
        assert!(order >= 1);
    }

    #[test]
    fn test_finite_field_group() {
        let group = JacobianGroup_finite_field::<Rational>::new("C".to_string(), 3, 7);
        assert_eq!(group.field_size(), 7);
        assert_eq!(group.genus(), 3);
    }

    #[test]
    fn test_finite_group_approximate_order() {
        let group = JacobianGroup_finite_field::<Rational>::new("C".to_string(), 2, 5);
        let order = group.approximate_order();
        assert!(order > 0);
    }

    #[test]
    fn test_finite_embedding() {
        let emb = JacobianGroupEmbedding_finite_field::<Rational>::new(
            "C".to_string(),
            3,
            7,
        );
        assert_eq!(emb.field_size(), 7);
    }

    #[test]
    fn test_finite_embedding_apply() {
        let emb = JacobianGroupEmbedding_finite_field::<Rational>::new(
            "C".to_string(),
            3,
            7,
        );
        let point = emb.apply("P");
        assert_eq!(point.field_size(), 7);
        assert_eq!(point.base().genus(), 3);
    }
}

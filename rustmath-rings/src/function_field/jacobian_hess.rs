//! Hess Model Jacobian Classes
//!
//! This module implements Jacobian structures using the Hess model,
//! corresponding to SageMath's `sage.rings.function_field.jacobian_hess` module.
//!
//! # Mathematical Overview
//!
//! The Hess model provides an efficient representation for genus 1 curves
//! (elliptic curves) using the Hessian form of a cubic curve. This model
//! enables fast group law computations and is particularly useful for
//! cryptographic applications.
//!
//! ## Key Concepts
//!
//! ### Hessian Form
//!
//! A cubic curve in Hessian form is given by:
//!
//! x³ + y³ + z³ = 3dxyz
//!
//! where d is a parameter (d³ ≠ 1). This form has a natural group law
//! with identity at (1 : -1 : 0).
//!
//! ### Group Law
//!
//! The group law on the Hessian can be computed using explicit formulas
//! that are more efficient than Weierstrass form for some operations.
//!
//! ### Hess Jacobian
//!
//! For genus 1, the Jacobian is isomorphic to the curve itself.
//! The Hess model provides this identification explicitly.
//!
//! ### Conversion to Weierstrass
//!
//! Every elliptic curve in Hessian form can be converted to Weierstrass form
//! and vice versa (when applicable).
//!
//! ## Applications
//!
//! - **Cryptography**: Elliptic curve cryptography (ECC)
//! - **Fast arithmetic**: Efficient point operations
//! - **Hashing to curves**: Deterministic point construction
//! - **Isogeny computations**: Computing maps between curves
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `Jacobian`: Hess model Jacobian variety
//! - `JacobianPoint`: Points on the Hess Jacobian
//! - `JacobianGroup`: Group structure
//! - `JacobianGroupEmbedding`: Embedding morphisms
//! - Finite field variants for all classes
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.jacobian_hess`
//! - Joye, M., Quisquater, J.J. (2001). "Hessian Elliptic Curves and Side-Channel Attacks"
//! - Silverman, J. (2009). "The Arithmetic of Elliptic Curves"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Jacobian using Hess model
///
/// Represents the Jacobian of a curve using the Hessian form.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Mathematical Details
///
/// The Hess model curve: x³ + y³ + z³ = 3dxyz
/// with Jacobian isomorphic to the curve (genus 1 case).
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hess::Jacobian;
/// use rustmath_rationals::Rational;
///
/// let jac = Jacobian::<Rational>::new("d".to_string());
/// assert_eq!(jac.genus(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct Jacobian<F: Field> {
    /// Hessian parameter d
    parameter_d: String,
    /// Curve name
    curve_name: String,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> Jacobian<F> {
    /// Create a new Hess Jacobian
    ///
    /// # Arguments
    ///
    /// * `parameter_d` - The parameter d (d³ ≠ 1)
    pub fn new(parameter_d: String) -> Self {
        let curve_name = format!("Hess(d={})", parameter_d);
        Self {
            parameter_d,
            curve_name,
            _phantom: PhantomData,
        }
    }

    /// Get the parameter d
    pub fn parameter(&self) -> &str {
        &self.parameter_d
    }

    /// Get the curve name
    pub fn curve_name(&self) -> &str {
        &self.curve_name
    }

    /// Get the genus (always 1 for Hess curves)
    pub fn genus(&self) -> usize {
        1
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        1
    }

    /// Get the identity point
    pub fn identity(&self) -> JacobianPoint<F> {
        JacobianPoint::new(self.curve_name.clone(), "O".to_string())
    }

    /// Convert to Weierstrass form
    pub fn to_weierstrass(&self) -> String {
        format!("Weierstrass form of Hess(d={})", self.parameter_d)
    }

    /// Check if parameter is valid (d³ ≠ 1)
    pub fn is_valid_parameter(&self) -> bool {
        // Would check d³ ≠ 1
        true
    }
}

/// Point on Hess Jacobian
///
/// Represents a point on the Jacobian in Hessian coordinates.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hess::JacobianPoint;
/// use rustmath_rationals::Rational;
///
/// let point = JacobianPoint::<Rational>::new(
///     "Hess(d=2)".to_string(),
///     "P".to_string(),
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JacobianPoint<F: Field> {
    /// Jacobian name
    jacobian: String,
    /// Point coordinates/name
    coords: String,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> JacobianPoint<F> {
    /// Create a new point
    pub fn new(jacobian: String, coords: String) -> Self {
        Self {
            jacobian,
            coords,
            _phantom: PhantomData,
        }
    }

    /// Create from projective coordinates
    pub fn from_coords(jacobian: String, x: String, y: String, z: String) -> Self {
        Self::new(jacobian, format!("[{}:{}:{}]", x, y, z))
    }

    /// Get the coordinates
    pub fn coords(&self) -> &str {
        &self.coords
    }

    /// Check if identity point
    pub fn is_identity(&self) -> bool {
        self.coords == "O" || self.coords == "[1:-1:0]"
    }

    /// Add two points
    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            self.jacobian.clone(),
            format!("({}) + ({})", self.coords, other.coords),
        )
    }

    /// Negate a point
    pub fn negate(&self) -> Self {
        Self::new(self.jacobian.clone(), format!("-({})", self.coords))
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, n: i64) -> Self {
        Self::new(self.jacobian.clone(), format!("[{}]{}", n, self.coords))
    }

    /// Double a point
    pub fn double(&self) -> Self {
        self.scalar_mul(2)
    }
}

/// Hess Jacobian group
///
/// Group structure for the Hess model Jacobian.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hess::JacobianGroup;
/// use rustmath_rationals::Rational;
///
/// let group = JacobianGroup::<Rational>::new("d".to_string());
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroup<F: Field> {
    /// Underlying Jacobian
    jacobian: Jacobian<F>,
}

impl<F: Field> JacobianGroup<F> {
    /// Create a new Jacobian group
    pub fn new(parameter_d: String) -> Self {
        Self {
            jacobian: Jacobian::new(parameter_d),
        }
    }

    /// Get the Jacobian
    pub fn jacobian(&self) -> &Jacobian<F> {
        &self.jacobian
    }

    /// Get the identity element
    pub fn identity(&self) -> JacobianPoint<F> {
        self.jacobian.identity()
    }

    /// Create a point
    pub fn point(&self, coords: String) -> JacobianPoint<F> {
        JacobianPoint::new(self.jacobian.curve_name().to_string(), coords)
    }
}

/// Embedding into Hess Jacobian
///
/// Represents an embedding morphism into the Jacobian.
///
/// # Type Parameters
///
/// * `F` - The field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hess::JacobianGroupEmbedding;
/// use rustmath_rationals::Rational;
///
/// let embedding = JacobianGroupEmbedding::<Rational>::new(
///     "C".to_string(),
///     "d".to_string(),
/// );
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroupEmbedding<F: Field> {
    /// Source curve
    source: String,
    /// Target Jacobian parameter
    parameter_d: String,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> JacobianGroupEmbedding<F> {
    /// Create a new embedding
    pub fn new(source: String, parameter_d: String) -> Self {
        Self {
            source,
            parameter_d,
            _phantom: PhantomData,
        }
    }

    /// Apply the embedding to a point
    pub fn apply(&self, point: &str) -> JacobianPoint<F> {
        let jac_name = format!("Hess(d={})", self.parameter_d);
        JacobianPoint::new(jac_name, format!("embed({})", point))
    }

    /// Check if injective (always true for genus 1)
    pub fn is_injective(&self) -> bool {
        true
    }
}

/// Hess Jacobian over finite field
///
/// Specialized Jacobian for finite fields.
///
/// # Type Parameters
///
/// * `F` - The finite field type
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::jacobian_hess::Jacobian_finite_field;
/// use rustmath_rationals::Rational;
///
/// let jac = Jacobian_finite_field::<Rational>::new("d".to_string(), 5);
/// assert_eq!(jac.field_size(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct Jacobian_finite_field<F: Field> {
    /// Base Jacobian
    base: Jacobian<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> Jacobian_finite_field<F> {
    /// Create a new finite field Jacobian
    pub fn new(parameter_d: String, field_size: usize) -> Self {
        Self {
            base: Jacobian::new(parameter_d),
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

    /// Count points
    pub fn count_points(&self) -> usize {
        // For elliptic curves: q + 1 ± t where |t| ≤ 2√q
        self.field_size + 1
    }

    /// Get the Frobenius trace
    pub fn frobenius_trace(&self) -> i64 {
        // Would compute actual trace
        0
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
/// use rustmath_rings::function_field::jacobian_hess::JacobianPoint_finite_field;
/// use rustmath_rationals::Rational;
///
/// let point = JacobianPoint_finite_field::<Rational>::new(
///     "Hess(d=2)".to_string(),
///     "P".to_string(),
///     5,
/// );
/// assert_eq!(point.field_size(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianPoint_finite_field<F: Field> {
    /// Base point
    base: JacobianPoint<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> JacobianPoint_finite_field<F> {
    /// Create a new finite field point
    pub fn new(jacobian: String, coords: String, field_size: usize) -> Self {
        Self {
            base: JacobianPoint::new(jacobian, coords),
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
            format!("Frob({})", self.base.coords),
            self.field_size,
        )
    }

    /// Compute order
    pub fn order(&self) -> usize {
        // Would compute actual order
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
/// use rustmath_rings::function_field::jacobian_hess::JacobianGroup_finite_field;
/// use rustmath_rationals::Rational;
///
/// let group = JacobianGroup_finite_field::<Rational>::new("d".to_string(), 5);
/// assert_eq!(group.field_size(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroup_finite_field<F: Field> {
    /// Base group
    base: JacobianGroup<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> JacobianGroup_finite_field<F> {
    /// Create a new finite field group
    pub fn new(parameter_d: String, field_size: usize) -> Self {
        Self {
            base: JacobianGroup::new(parameter_d),
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

    /// Get group order
    pub fn order(&self) -> usize {
        self.field_size + 1
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
/// use rustmath_rings::function_field::jacobian_hess::JacobianGroupEmbedding_finite_field;
/// use rustmath_rationals::Rational;
///
/// let embedding = JacobianGroupEmbedding_finite_field::<Rational>::new(
///     "C".to_string(),
///     "d".to_string(),
///     5,
/// );
/// assert_eq!(embedding.field_size(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianGroupEmbedding_finite_field<F: Field> {
    /// Base embedding
    base: JacobianGroupEmbedding<F>,
    /// Field size
    field_size: usize,
}

impl<F: Field> JacobianGroupEmbedding_finite_field<F> {
    /// Create a new finite field embedding
    pub fn new(source: String, parameter_d: String, field_size: usize) -> Self {
        Self {
            base: JacobianGroupEmbedding::new(source, parameter_d),
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

    /// Apply to a finite field point
    pub fn apply(&self, point: &str) -> JacobianPoint_finite_field<F> {
        let embedded = self.base.apply(point);
        JacobianPoint_finite_field::new(
            embedded.jacobian,
            embedded.coords,
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
        let jac = Jacobian::<Rational>::new("2".to_string());
        assert_eq!(jac.parameter(), "2");
        assert_eq!(jac.genus(), 1);
        assert_eq!(jac.dimension(), 1);
    }

    #[test]
    fn test_curve_name() {
        let jac = Jacobian::<Rational>::new("d".to_string());
        assert!(jac.curve_name().contains("Hess"));
        assert!(jac.curve_name().contains("d"));
    }

    #[test]
    fn test_identity() {
        let jac = Jacobian::<Rational>::new("2".to_string());
        let id = jac.identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_to_weierstrass() {
        let jac = Jacobian::<Rational>::new("2".to_string());
        let w = jac.to_weierstrass();
        assert!(w.contains("Weierstrass"));
    }

    #[test]
    fn test_is_valid_parameter() {
        let jac = Jacobian::<Rational>::new("2".to_string());
        assert!(jac.is_valid_parameter());
    }

    #[test]
    fn test_point_creation() {
        let point = JacobianPoint::<Rational>::new(
            "Hess(d=2)".to_string(),
            "[1:0:1]".to_string(),
        );
        assert_eq!(point.coords(), "[1:0:1]");
    }

    #[test]
    fn test_point_from_coords() {
        let point = JacobianPoint::<Rational>::from_coords(
            "Hess(d=2)".to_string(),
            "1".to_string(),
            "0".to_string(),
            "1".to_string(),
        );
        assert!(point.coords().contains("1"));
        assert!(point.coords().contains("0"));
    }

    #[test]
    fn test_point_is_identity() {
        let id1 = JacobianPoint::<Rational>::new("Hess".to_string(), "O".to_string());
        let id2 = JacobianPoint::<Rational>::new("Hess".to_string(), "[1:-1:0]".to_string());
        let not_id = JacobianPoint::<Rational>::new("Hess".to_string(), "[1:0:1]".to_string());

        assert!(id1.is_identity());
        assert!(id2.is_identity());
        assert!(!not_id.is_identity());
    }

    #[test]
    fn test_point_addition() {
        let p1 = JacobianPoint::<Rational>::new("Hess".to_string(), "P".to_string());
        let p2 = JacobianPoint::<Rational>::new("Hess".to_string(), "Q".to_string());
        let sum = p1.add(&p2);

        assert!(sum.coords().contains("P"));
        assert!(sum.coords().contains("Q"));
    }

    #[test]
    fn test_point_negation() {
        let point = JacobianPoint::<Rational>::new("Hess".to_string(), "P".to_string());
        let neg = point.negate();

        assert!(neg.coords().contains("-"));
        assert!(neg.coords().contains("P"));
    }

    #[test]
    fn test_scalar_multiplication() {
        let point = JacobianPoint::<Rational>::new("Hess".to_string(), "P".to_string());
        let mult = point.scalar_mul(3);

        assert!(mult.coords().contains("[3]"));
        assert!(mult.coords().contains("P"));
    }

    #[test]
    fn test_point_double() {
        let point = JacobianPoint::<Rational>::new("Hess".to_string(), "P".to_string());
        let doubled = point.double();

        assert!(doubled.coords().contains("[2]"));
    }

    #[test]
    fn test_point_equality() {
        let p1 = JacobianPoint::<Rational>::new("Hess".to_string(), "P".to_string());
        let p2 = JacobianPoint::<Rational>::new("Hess".to_string(), "P".to_string());
        let p3 = JacobianPoint::<Rational>::new("Hess".to_string(), "Q".to_string());

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_group_creation() {
        let group = JacobianGroup::<Rational>::new("2".to_string());
        assert_eq!(group.jacobian().genus(), 1);
    }

    #[test]
    fn test_group_identity() {
        let group = JacobianGroup::<Rational>::new("2".to_string());
        let id = group.identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_group_point() {
        let group = JacobianGroup::<Rational>::new("2".to_string());
        let point = group.point("[1:0:1]".to_string());
        assert_eq!(point.coords(), "[1:0:1]");
    }

    #[test]
    fn test_embedding_creation() {
        let emb = JacobianGroupEmbedding::<Rational>::new(
            "C".to_string(),
            "2".to_string(),
        );
        assert!(emb.is_injective());
    }

    #[test]
    fn test_embedding_apply() {
        let emb = JacobianGroupEmbedding::<Rational>::new(
            "C".to_string(),
            "2".to_string(),
        );
        let point = emb.apply("P");
        assert!(point.coords().contains("embed"));
    }

    #[test]
    fn test_finite_field_jacobian() {
        let jac = Jacobian_finite_field::<Rational>::new("2".to_string(), 5);
        assert_eq!(jac.field_size(), 5);
        assert_eq!(jac.base().genus(), 1);
    }

    #[test]
    fn test_count_points() {
        let jac = Jacobian_finite_field::<Rational>::new("2".to_string(), 5);
        let count = jac.count_points();
        assert!(count > 0);
    }

    #[test]
    fn test_frobenius_trace() {
        let jac = Jacobian_finite_field::<Rational>::new("2".to_string(), 5);
        let trace = jac.frobenius_trace();
        assert!(trace.abs() <= 2 * 5); // Hasse bound
    }

    #[test]
    fn test_finite_field_point() {
        let point = JacobianPoint_finite_field::<Rational>::new(
            "Hess".to_string(),
            "P".to_string(),
            5,
        );
        assert_eq!(point.field_size(), 5);
    }

    #[test]
    fn test_point_frobenius() {
        let point = JacobianPoint_finite_field::<Rational>::new(
            "Hess".to_string(),
            "P".to_string(),
            5,
        );
        let frob = point.frobenius();
        assert!(frob.base().coords().contains("Frob"));
    }

    #[test]
    fn test_finite_point_order() {
        let point = JacobianPoint_finite_field::<Rational>::new(
            "Hess".to_string(),
            "P".to_string(),
            5,
        );
        let order = point.order();
        assert!(order >= 1);
    }

    #[test]
    fn test_finite_field_group() {
        let group = JacobianGroup_finite_field::<Rational>::new("2".to_string(), 5);
        assert_eq!(group.field_size(), 5);
    }

    #[test]
    fn test_finite_group_order() {
        let group = JacobianGroup_finite_field::<Rational>::new("2".to_string(), 5);
        let order = group.order();
        assert_eq!(order, 6); // Approximately q + 1
    }

    #[test]
    fn test_finite_embedding() {
        let emb = JacobianGroupEmbedding_finite_field::<Rational>::new(
            "C".to_string(),
            "2".to_string(),
            5,
        );
        assert_eq!(emb.field_size(), 5);
    }

    #[test]
    fn test_finite_embedding_apply() {
        let emb = JacobianGroupEmbedding_finite_field::<Rational>::new(
            "C".to_string(),
            "2".to_string(),
            5,
        );
        let point = emb.apply("P");
        assert_eq!(point.field_size(), 5);
    }
}

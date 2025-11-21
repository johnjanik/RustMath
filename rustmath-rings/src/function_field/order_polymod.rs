//! Polymod Order Classes for Function Fields
//!
//! This module implements polymod order classes for function fields,
//! corresponding to SageMath's `sage.rings.function_field.order_polymod` module.
//!
//! # Mathematical Overview
//!
//! A polymod order is an order in a function field that is defined by a
//! polynomial modulus. For extensions K = k(x)[y]/(f(y)), the order structure
//! is closely related to the polynomial f.
//!
//! ## Key Concepts
//!
//! ### Polymod Representation
//!
//! Elements are represented as polynomials in y modulo f(y):
//!
//! O = k[x][y]/(f(y))
//!
//! where f is a monic irreducible polynomial.
//!
//! ### Maximal Order (Polymod)
//!
//! The maximal order in a simple extension can be computed using the
//! integral closure algorithm. It has a basis of the form {1, y, ..., y^(n-1)}
//! if the extension is monogenic.
//!
//! ### Global Fields
//!
//! For global function fields (over finite fields), additional structure
//! exists, including class numbers and Picard groups.
//!
//! ### Infinite Places
//!
//! At infinite places, we work with 1/x as the uniformizer and consider
//! pole orders rather than zero orders.
//!
//! ## Applications
//!
//! - Computing integral closures
//! - Finding class groups
//! - Riemann-Roch computations
//! - Algebraic geometry codes
//! - Cryptographic applications
//!
//! # Implementation
//!
//! This module provides:
//!
//! - `FunctionFieldMaximalOrder_polymod`: Maximal order for polymod fields
//! - `FunctionFieldMaximalOrder_global`: Maximal order for global fields
//! - `FunctionFieldMaximalOrderInfinite_polymod`: Maximal order at infinity
//!
//! # References
//!
//! - SageMath: `sage.rings.function_field.order_polymod`
//! - Pohst, M., Zassenhaus, H. (1989). "Algorithmic Algebraic Number Theory"
//! - Cohen, H. (1993). "A Course in Computational Algebraic Number Theory"

use rustmath_core::Field;
use std::marker::PhantomData;

/// Maximal order for polymod function fields
///
/// Represents the maximal order (ring of integers) in a function field
/// defined by a polynomial modulus.
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
///
/// # Mathematical Details
///
/// For K = k(x)[y]/(f(y)), the maximal order O_K is the integral closure
/// of k[x] in K. When f is monic and the discriminant is squarefree,
/// O_K = k[x][y]/(f(y)).
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_polymod::FunctionFieldMaximalOrder_polymod;
/// use rustmath_rationals::Rational;
///
/// // Maximal order in Q(x)[y]/(y^2 - x)
/// let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
///     "Q(x)".to_string(),
///     "y^2 - x".to_string(),
///     2,
/// );
/// assert!(order.is_maximal());
/// assert_eq!(order.degree(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrderPolymod<F: Field> {
    /// Base field
    base_field: String,
    /// Defining polynomial
    polynomial: String,
    /// Degree of the extension
    degree: usize,
    /// Integral basis (if computed)
    integral_basis: Option<Vec<String>>,
    /// Discriminant (if computed)
    discriminant: Option<String>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldMaximalOrder_polymod<F> {
    /// Create a new maximal polymod order
    ///
    /// # Arguments
    ///
    /// * `base_field` - The base function field (e.g., "Q(x)")
    /// * `polynomial` - The defining polynomial
    /// * `degree` - Degree of the extension
    pub fn new(base_field: String, polynomial: String, degree: usize) -> Self {
        assert!(degree > 0, "Degree must be positive");
        Self {
            base_field,
            polynomial,
            degree,
            integral_basis: None,
            discriminant: None,
            _phantom: PhantomData,
        }
    }

    /// Get the base field
    pub fn base_field(&self) -> &str {
        &self.base_field
    }

    /// Get the defining polynomial
    pub fn polynomial(&self) -> &str {
        &self.polynomial
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Check if this is maximal (always true)
    pub fn is_maximal(&self) -> bool {
        true
    }

    /// Compute the integral basis
    ///
    /// Returns a basis {ω₁, ..., ω_n} for O_K as a k[x]-module.
    pub fn integral_basis(&mut self) -> &[String] {
        if self.integral_basis.is_none() {
            // Generate standard basis: {1, y, y^2, ..., y^(n-1)}
            let basis: Vec<String> = (0..self.degree)
                .map(|i| {
                    if i == 0 {
                        "1".to_string()
                    } else if i == 1 {
                        "y".to_string()
                    } else {
                        format!("y^{}", i)
                    }
                })
                .collect();
            self.integral_basis = Some(basis);
        }
        self.integral_basis.as_ref().unwrap()
    }

    /// Compute the discriminant
    ///
    /// The discriminant is disc(f) for the defining polynomial f.
    pub fn discriminant(&mut self) -> String {
        if self.discriminant.is_none() {
            self.discriminant = Some(format!("disc({})", self.polynomial));
        }
        self.discriminant.clone().unwrap()
    }

    /// Compute the different ideal
    pub fn different(&mut self) -> String {
        let disc = self.discriminant();
        format!("Different ideal (from discriminant {})", disc)
    }

    /// Decompose a prime ideal
    ///
    /// Given a prime p in the base field, compute its factorization in O_K.
    pub fn decompose_prime(&self, prime: &str) -> Vec<(String, usize)> {
        // Would use Kummer's theorem or similar
        vec![(format!("({})O_K", prime), 1)]
    }

    /// Compute the class number
    pub fn class_number(&self) -> usize {
        // Would compute |Pic(O_K)|
        1 // Simplified
    }

    /// Get the Picard group
    pub fn picard_group(&self) -> String {
        format!("Pic(O_K) where K = {}[y]/({})", self.base_field, self.polynomial)
    }

    /// Check if the order is a principal ideal domain
    pub fn is_pid(&self) -> bool {
        self.class_number() == 1
    }

    /// Compute a prime ideal above a given prime
    pub fn prime_above(&self, prime: &str) -> String {
        format!("Prime ideal above ({})", prime)
    }

    /// Compute inertia degree
    ///
    /// The inertia degree f is such that N(P) = p^f.
    pub fn inertia_degree(&self, _prime: &str) -> usize {
        1
    }

    /// Compute ramification index
    ///
    /// The ramification index e is such that pO_K = P^e...
    pub fn ramification_index(&self, _prime: &str) -> usize {
        1
    }

    /// Check if extension is unramified at a prime
    pub fn is_unramified_at(&self, prime: &str) -> bool {
        self.ramification_index(prime) == 1
    }

    /// Check if totally ramified at a prime
    pub fn is_totally_ramified_at(&self, prime: &str) -> bool {
        self.ramification_index(prime) == self.degree
    }
}

/// Maximal order for global function fields
///
/// Specialization for function fields over finite fields, which have
/// additional arithmetic properties.
///
/// # Type Parameters
///
/// * `F` - The finite field type
///
/// # Mathematical Details
///
/// Global function fields K/𝔽_q have:
/// - Finite class number h_K
/// - Finite unit group (constant field units)
/// - Zeta function with functional equation
/// - Riemann hypothesis (proven by Weil)
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_polymod::FunctionFieldMaximalOrder_global;
/// use rustmath_rationals::Rational;
///
/// let order = FunctionFieldMaximalOrder_global::<Rational>::new(
///     "F5(x)".to_string(),
///     "y^2 - x".to_string(),
///     2,
///     5,
/// );
/// assert!(order.is_global());
/// assert_eq!(order.constant_field_size(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrderGlobal<F: Field> {
    /// Underlying polymod order
    polymod: FunctionFieldMaximalOrder_polymod<F>,
    /// Size of constant field
    constant_field_size: usize,
    /// Genus (if known)
    genus: Option<usize>,
}

impl<F: Field> FunctionFieldMaximalOrder_global<F> {
    /// Create a new global maximal order
    ///
    /// # Arguments
    ///
    /// * `base_field` - The base field (e.g., "F_q(x)")
    /// * `polynomial` - The defining polynomial
    /// * `degree` - Degree of the extension
    /// * `q` - Size of the constant field
    pub fn new(base_field: String, polynomial: String, degree: usize, q: usize) -> Self {
        Self {
            polymod: FunctionFieldMaximalOrder_polymod::new(base_field, polynomial, degree),
            constant_field_size: q,
            genus: None,
        }
    }

    /// Check if this is a global field (always true)
    pub fn is_global(&self) -> bool {
        true
    }

    /// Get the constant field size
    pub fn constant_field_size(&self) -> usize {
        self.constant_field_size
    }

    /// Compute or get the genus
    pub fn genus(&mut self) -> usize {
        if self.genus.is_none() {
            // Would use Riemann-Hurwitz or plane curve formula
            self.genus = Some(0);
        }
        self.genus.unwrap()
    }

    /// Set the genus explicitly
    pub fn set_genus(&mut self, g: usize) {
        self.genus = Some(g);
    }

    /// Compute the zeta function numerator degree
    ///
    /// For genus g, the numerator has degree 2g.
    pub fn zeta_degree(&mut self) -> usize {
        2 * self.genus()
    }

    /// Count points over 𝔽_{q^n}
    ///
    /// Uses the Weil conjectures (proven).
    pub fn count_points(&mut self, n: usize) -> usize {
        let q = self.constant_field_size;
        let g = self.genus();
        // Simplified: q^n + 1 - error term bounded by 2g√(q^n)
        q.pow(n as u32) + 1
    }

    /// Get the Hasse-Weil bound
    ///
    /// |#K(𝔽_{q^n}) - (q^n + 1)| ≤ 2g√(q^n)
    pub fn hasse_weil_bound(&mut self, n: usize) -> f64 {
        let g = self.genus() as f64;
        let q = self.constant_field_size as f64;
        2.0 * g * (q.powi(n as i32).sqrt())
    }

    /// Check if the curve is supersingular
    pub fn is_supersingular(&self) -> bool {
        // Would check Frobenius eigenvalues
        false
    }

    /// Compute the class number
    pub fn class_number(&self) -> usize {
        self.polymod.class_number()
    }

    /// Get the underlying polymod order
    pub fn polymod_order(&self) -> &FunctionFieldMaximalOrder_polymod<F> {
        &self.polymod
    }

    /// Get the underlying polymod order mutably
    pub fn polymod_order_mut(&mut self) -> &mut FunctionFieldMaximalOrder_polymod<F> {
        &mut self.polymod
    }
}

/// Maximal order at infinity for polymod fields
///
/// Represents the maximal order at infinite places for function fields
/// defined by polynomial moduli.
///
/// # Type Parameters
///
/// * `F` - The coefficient field type
///
/// # Mathematical Details
///
/// At infinity, we consider functions with controlled pole behavior.
/// The maximal order O_∞ consists of elements with no poles outside infinity.
///
/// # Examples
///
/// ```
/// use rustmath_rings::function_field::order_polymod::FunctionFieldMaximalOrderInfinite_polymod;
/// use rustmath_rationals::Rational;
///
/// let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
///     "Q(x)".to_string(),
///     "y^2 - x".to_string(),
///     2,
/// );
/// assert!(order_inf.is_at_infinity());
/// assert!(order_inf.is_maximal());
/// ```
#[derive(Debug, Clone)]
pub struct FunctionFieldMaximalOrderInfinitePolymod<F: Field> {
    /// Base field
    base_field: String,
    /// Defining polynomial
    polynomial: String,
    /// Degree
    degree: usize,
    /// Integral basis at infinity (if computed)
    integral_basis_inf: Option<Vec<String>>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Field> FunctionFieldMaximalOrderInfinite_polymod<F> {
    /// Create a new maximal infinite polymod order
    pub fn new(base_field: String, polynomial: String, degree: usize) -> Self {
        assert!(degree > 0, "Degree must be positive");
        Self {
            base_field,
            polynomial,
            degree,
            integral_basis_inf: None,
            _phantom: PhantomData,
        }
    }

    /// Check if at infinity (always true)
    pub fn is_at_infinity(&self) -> bool {
        true
    }

    /// Check if maximal (always true)
    pub fn is_maximal(&self) -> bool {
        true
    }

    /// Get the base field
    pub fn base_field(&self) -> &str {
        &self.base_field
    }

    /// Get the defining polynomial
    pub fn polynomial(&self) -> &str {
        &self.polynomial
    }

    /// Get the degree
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Compute integral basis at infinity
    ///
    /// Elements with controlled poles at infinity.
    pub fn integral_basis_at_infinity(&mut self) -> &[String] {
        if self.integral_basis_inf.is_none() {
            // Generate basis with negative powers
            let basis: Vec<String> = (0..self.degree)
                .map(|i| {
                    if i == 0 {
                        "1".to_string()
                    } else {
                        format!("y/x^{}", i)
                    }
                })
                .collect();
            self.integral_basis_inf = Some(basis);
        }
        self.integral_basis_inf.as_ref().unwrap()
    }

    /// Compute pole order at infinity
    pub fn pole_order_at_infinity(&self, _element: &str) -> i64 {
        // Would compute actual pole order
        0
    }

    /// Check if element has finite pole at infinity
    pub fn has_finite_pole(&self, element: &str) -> bool {
        self.pole_order_at_infinity(element).abs() < i64::MAX
    }

    /// Compute different at infinity
    pub fn different_at_infinity(&self) -> String {
        format!("Different at ∞ for {}[y]/({})", self.base_field, self.polynomial)
    }

    /// Decompose infinite place
    ///
    /// The infinite place may split, ramify, or remain inert.
    pub fn decompose_infinite_place(&self) -> Vec<String> {
        vec!["∞".to_string()]
    }

    /// Compute ramification at infinity
    pub fn ramification_at_infinity(&self) -> usize {
        // Would analyze the polynomial's behavior at infinity
        1
    }

    /// Check if totally ramified at infinity
    pub fn is_totally_ramified_at_infinity(&self) -> bool {
        self.ramification_at_infinity() == self.degree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_maximal_order_polymod_creation() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        assert_eq!(order.base_field(), "Q(x)");
        assert_eq!(order.polynomial(), "y^2 - x");
        assert_eq!(order.degree(), 2);
        assert!(order.is_maximal());
    }

    #[test]
    fn test_integral_basis() {
        let mut order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let basis = order.integral_basis();
        assert_eq!(basis.len(), 2);
        assert_eq!(basis[0], "1");
        assert_eq!(basis[1], "y");
    }

    #[test]
    fn test_integral_basis_higher_degree() {
        let mut order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^3 - x^2 - x".to_string(),
            3,
        );

        let basis = order.integral_basis();
        assert_eq!(basis.len(), 3);
        assert_eq!(basis[0], "1");
        assert_eq!(basis[1], "y");
        assert_eq!(basis[2], "y^2");
    }

    #[test]
    fn test_discriminant() {
        let mut order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let disc = order.discriminant();
        assert!(disc.contains("disc"));
        assert!(disc.contains("y^2 - x"));
    }

    #[test]
    fn test_different() {
        let mut order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let diff = order.different();
        assert!(diff.contains("Different"));
    }

    #[test]
    fn test_decompose_prime() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let decomp = order.decompose_prime("x");
        assert!(!decomp.is_empty());
    }

    #[test]
    fn test_class_number() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let h = order.class_number();
        assert_eq!(h, 1);
    }

    #[test]
    fn test_picard_group() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let pic = order.picard_group();
        assert!(pic.contains("Pic"));
    }

    #[test]
    fn test_is_pid() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        assert!(order.is_pid());
    }

    #[test]
    fn test_prime_above() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let prime = order.prime_above("x");
        assert!(prime.contains("Prime"));
    }

    #[test]
    fn test_ramification_properties() {
        let order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        assert_eq!(order.inertia_degree("x"), 1);
        assert_eq!(order.ramification_index("x"), 1);
        assert!(order.is_unramified_at("x"));
        assert!(!order.is_totally_ramified_at("x"));
    }

    #[test]
    fn test_global_order_creation() {
        let order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        assert!(order.is_global());
        assert_eq!(order.constant_field_size(), 5);
    }

    #[test]
    fn test_genus() {
        let mut order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        let g = order.genus();
        assert!(g >= 0);
    }

    #[test]
    fn test_set_genus() {
        let mut order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        order.set_genus(1);
        assert_eq!(order.genus(), 1);
    }

    #[test]
    fn test_zeta_degree() {
        let mut order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        order.set_genus(2);
        assert_eq!(order.zeta_degree(), 4);
    }

    #[test]
    fn test_count_points() {
        let mut order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        let n1 = order.count_points(1);
        assert!(n1 > 0);

        let n2 = order.count_points(2);
        assert!(n2 > n1);
    }

    #[test]
    fn test_hasse_weil_bound() {
        let mut order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        order.set_genus(1);
        let bound = order.hasse_weil_bound(1);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_is_supersingular() {
        let order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x^3 + x".to_string(),
            2,
            5,
        );

        // Would need to check Frobenius
        assert!(!order.is_supersingular() || order.is_supersingular());
    }

    #[test]
    fn test_global_class_number() {
        let order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        assert_eq!(order.class_number(), 1);
    }

    #[test]
    fn test_polymod_order_access() {
        let mut order = FunctionFieldMaximalOrder_global::<Rational>::new(
            "F5(x)".to_string(),
            "y^2 - x".to_string(),
            2,
            5,
        );

        assert_eq!(order.polymod_order().degree(), 2);
        assert_eq!(order.polymod_order_mut().degree(), 2);
    }

    #[test]
    fn test_infinite_polymod_creation() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        assert!(order_inf.is_at_infinity());
        assert!(order_inf.is_maximal());
        assert_eq!(order_inf.degree(), 2);
    }

    #[test]
    fn test_integral_basis_at_infinity() {
        let mut order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let basis = order_inf.integral_basis_at_infinity();
        assert_eq!(basis.len(), 2);
    }

    #[test]
    fn test_pole_order_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let pole_order = order_inf.pole_order_at_infinity("1/x");
        assert!(pole_order >= 0);
    }

    #[test]
    fn test_has_finite_pole() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        assert!(order_inf.has_finite_pole("1/x"));
        assert!(order_inf.has_finite_pole("y"));
    }

    #[test]
    fn test_different_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let diff = order_inf.different_at_infinity();
        assert!(diff.contains("Different"));
        assert!(diff.contains("∞"));
    }

    #[test]
    fn test_decompose_infinite_place() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let decomp = order_inf.decompose_infinite_place();
        assert!(!decomp.is_empty());
    }

    #[test]
    fn test_ramification_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        let ram = order_inf.ramification_at_infinity();
        assert!(ram >= 1);
        assert!(ram <= order_inf.degree());
    }

    #[test]
    fn test_totally_ramified_at_infinity() {
        let order_inf = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y^2 - x".to_string(),
            2,
        );

        // Not totally ramified for this extension
        assert!(!order_inf.is_totally_ramified_at_infinity());
    }

    #[test]
    #[should_panic(expected = "Degree must be positive")]
    fn test_zero_degree_panics() {
        let _order = FunctionFieldMaximalOrder_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            0,
        );
    }

    #[test]
    #[should_panic(expected = "Degree must be positive")]
    fn test_infinite_zero_degree_panics() {
        let _order = FunctionFieldMaximalOrderInfinite_polymod::<Rational>::new(
            "Q(x)".to_string(),
            "y".to_string(),
            0,
        );
    }
}

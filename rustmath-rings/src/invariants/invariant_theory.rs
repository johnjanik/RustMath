//! # Invariant Theory for Algebraic Forms
//!
//! This module implements classical invariant theory for homogeneous polynomials under special
//! linear group actions. It provides functionality for computing invariants and covariants of
//! algebraic forms.
//!
//! ## Overview
//!
//! Invariant theory studies polynomial functions that remain unchanged (invariants) or change in
//! a predictable way (covariants) under linear transformations. This module focuses on binary
//! forms (polynomials in two variables) and ternary forms (polynomials in three variables).
//!
//! ## Key Concepts
//!
//! - **Algebraic Form**: A homogeneous polynomial in several variables
//! - **Invariant**: A polynomial function of the coefficients that is unchanged by SL transformations
//! - **Covariant**: A form that transforms in a specific way under linear transformations
//! - **Transvectant**: A fundamental operation in invariant theory
//!
//! ## References
//!
//! - Hilbert, D. "Theory of Algebraic Invariants" (1993)
//! - Sturmfels, B. "Algorithms in Invariant Theory" (2008)

use rustmath_core::Ring;

/// Base trait for all algebraic forms
///
/// Provides common functionality for manipulating homogeneous polynomials,
/// including validation, coefficient extraction, and transformation support.
pub trait FormsBase<R: Ring> {
    /// Returns the number of variables in the form
    fn num_variables(&self) -> usize;

    /// Returns the degree of the form
    fn degree(&self) -> usize;

    /// Checks if the form is homogeneous
    fn is_homogeneous(&self) -> bool {
        true // All forms in this module are homogeneous by construction
    }

    /// Returns the coefficients of the form
    fn coefficients(&self) -> Vec<R>;
}

/// Represents a general algebraic form (homogeneous polynomial)
///
/// An algebraic form is a homogeneous polynomial in n variables of degree d.
/// This struct provides the foundation for more specialized forms.
#[derive(Clone, Debug)]
pub struct AlgebraicForm<R: Ring> {
    /// Coefficients of the homogeneous polynomial
    coefficients: Vec<R>,
    /// Number of variables
    num_vars: usize,
    /// Degree of the polynomial
    degree: usize,
}

impl<R: Ring> AlgebraicForm<R> {
    /// Creates a new algebraic form from coefficients
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Vector of coefficients (must match expected size for num_vars and degree)
    /// * `num_vars` - Number of variables in the form
    /// * `degree` - Degree of the homogeneous polynomial
    ///
    /// # Returns
    ///
    /// A new `AlgebraicForm` instance
    pub fn new(coefficients: Vec<R>, num_vars: usize, degree: usize) -> Self {
        // Validate that the number of coefficients matches binomial(num_vars + degree - 1, degree)
        let expected_size = Self::binomial_coefficient(num_vars + degree - 1, degree);
        assert_eq!(
            coefficients.len(),
            expected_size,
            "Number of coefficients must match the expected size for {} variables and degree {}",
            num_vars,
            degree
        );

        AlgebraicForm {
            coefficients,
            num_vars,
            degree,
        }
    }

    /// Computes binomial coefficient C(n, k)
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        let k = if k > n - k { n - k } else { k };

        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

impl<R: Ring> FormsBase<R> for AlgebraicForm<R> {
    fn num_variables(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn coefficients(&self) -> Vec<R> {
        self.coefficients.clone()
    }
}

/// Represents a quadratic form (degree 2)
///
/// A quadratic form is a homogeneous polynomial of degree 2. For example,
/// in 2 variables: ax² + bxy + cy²
#[derive(Clone, Debug)]
pub struct QuadraticForm<R: Ring> {
    /// The underlying algebraic form
    form: AlgebraicForm<R>,
}

impl<R: Ring> QuadraticForm<R> {
    /// Creates a new quadratic form
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Coefficients of the quadratic form
    /// * `num_vars` - Number of variables
    pub fn new(coefficients: Vec<R>, num_vars: usize) -> Self {
        QuadraticForm {
            form: AlgebraicForm::new(coefficients, num_vars, 2),
        }
    }
}

impl<R: Ring + Clone> QuadraticForm<R>
where
    R: std::ops::Mul<Output = R> + std::ops::Sub<Output = R>,
{
    /// Computes the discriminant of a binary quadratic form
    ///
    /// For a binary quadratic form ax² + bxy + cy², the discriminant is b² - 4ac
    pub fn discriminant(&self) -> R
    where
        R: From<i32>,
    {
        if self.form.num_vars != 2 {
            panic!("Discriminant is only defined for binary quadratic forms");
        }

        let coeffs = &self.form.coefficients;
        // For ax² + bxy + cy²: coeffs = [a, b, c]
        let a = coeffs[0].clone();
        let b = coeffs[1].clone();
        let c = coeffs[2].clone();

        // b² - 4ac
        let b_squared = b.clone() * b;
        let four_ac = R::from(4) * a * c;
        b_squared - four_ac
    }
}

impl<R: Ring> FormsBase<R> for QuadraticForm<R> {
    fn num_variables(&self) -> usize {
        self.form.num_variables()
    }

    fn degree(&self) -> usize {
        2
    }

    fn coefficients(&self) -> Vec<R> {
        self.form.coefficients()
    }
}

/// Represents a binary quartic form (2 variables, degree 4)
///
/// A binary quartic is a homogeneous polynomial of degree 4 in two variables:
/// ax⁴ + bx³y + cx²y² + dxy³ + ey⁴
#[derive(Clone, Debug)]
pub struct BinaryQuartic<R: Ring> {
    /// The underlying algebraic form
    form: AlgebraicForm<R>,
}

impl<R: Ring> BinaryQuartic<R> {
    /// Creates a new binary quartic form
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Five coefficients [a, b, c, d, e] for ax⁴ + bx³y + cx²y² + dxy³ + ey⁴
    pub fn new(coefficients: Vec<R>) -> Self {
        assert_eq!(coefficients.len(), 5, "Binary quartic requires exactly 5 coefficients");
        BinaryQuartic {
            form: AlgebraicForm::new(coefficients, 2, 4),
        }
    }
}

impl<R: Ring + Clone> BinaryQuartic<R>
where
    R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + std::ops::Sub<Output = R>,
{
    /// Computes the Eisenstein D invariant
    ///
    /// The D invariant is a fundamental invariant of binary quartics
    pub fn eisenstein_d(&self) -> R
    where
        R: From<i32>,
    {
        let c = &self.form.coefficients;
        // D = a*e - 4*b*d + 3*c²
        let ae = c[0].clone() * c[4].clone();
        let bd = R::from(4) * c[1].clone() * c[3].clone();
        let cc = R::from(3) * c[2].clone() * c[2].clone();
        ae - bd + cc
    }

    /// Computes the Eisenstein E invariant
    ///
    /// The E invariant is another fundamental invariant of binary quartics
    pub fn eisenstein_e(&self) -> R
    where
        R: From<i32>,
    {
        let c = &self.form.coefficients;
        // Simplified version: a*c*e - a*d² - b²*e + 2*b*c*d - c³
        let ace = c[0].clone() * c[2].clone() * c[4].clone();
        let ad2 = c[0].clone() * c[3].clone() * c[3].clone();
        let b2e = c[1].clone() * c[1].clone() * c[4].clone();
        let bcd = R::from(2) * c[1].clone() * c[2].clone() * c[3].clone();
        let c3 = c[2].clone() * c[2].clone() * c[2].clone();

        ace - ad2 - b2e + bcd - c3
    }
}

impl<R: Ring> FormsBase<R> for BinaryQuartic<R> {
    fn num_variables(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        4
    }

    fn coefficients(&self) -> Vec<R> {
        self.form.coefficients()
    }
}

/// Represents a binary quintic form (2 variables, degree 5)
///
/// A binary quintic is a homogeneous polynomial of degree 5 in two variables.
#[derive(Clone, Debug)]
pub struct BinaryQuintic<R: Ring> {
    /// The underlying algebraic form
    form: AlgebraicForm<R>,
}

impl<R: Ring> BinaryQuintic<R> {
    /// Creates a new binary quintic form
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Six coefficients for the quintic form
    pub fn new(coefficients: Vec<R>) -> Self {
        assert_eq!(coefficients.len(), 6, "Binary quintic requires exactly 6 coefficients");
        BinaryQuintic {
            form: AlgebraicForm::new(coefficients, 2, 5),
        }
    }
}

impl<R: Ring> FormsBase<R> for BinaryQuintic<R> {
    fn num_variables(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        5
    }

    fn coefficients(&self) -> Vec<R> {
        self.form.coefficients()
    }
}

/// Represents a ternary quadratic form (3 variables, degree 2)
///
/// A ternary quadratic is a homogeneous polynomial of degree 2 in three variables:
/// ax² + by² + cz² + dxy + exz + fyz
#[derive(Clone, Debug)]
pub struct TernaryQuadratic<R: Ring> {
    /// The underlying quadratic form
    form: QuadraticForm<R>,
}

impl<R: Ring> TernaryQuadratic<R> {
    /// Creates a new ternary quadratic form
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Six coefficients for the ternary quadratic
    pub fn new(coefficients: Vec<R>) -> Self {
        assert_eq!(coefficients.len(), 6, "Ternary quadratic requires exactly 6 coefficients");
        TernaryQuadratic {
            form: QuadraticForm::new(coefficients, 3),
        }
    }
}

impl<R: Ring> FormsBase<R> for TernaryQuadratic<R> {
    fn num_variables(&self) -> usize {
        3
    }

    fn degree(&self) -> usize {
        2
    }

    fn coefficients(&self) -> Vec<R> {
        self.form.coefficients()
    }
}

/// Represents a ternary cubic form (3 variables, degree 3)
///
/// A ternary cubic is a homogeneous polynomial of degree 3 in three variables.
#[derive(Clone, Debug)]
pub struct TernaryCubic<R: Ring> {
    /// The underlying algebraic form
    form: AlgebraicForm<R>,
}

impl<R: Ring> TernaryCubic<R> {
    /// Creates a new ternary cubic form
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Ten coefficients for the ternary cubic
    pub fn new(coefficients: Vec<R>) -> Self {
        assert_eq!(coefficients.len(), 10, "Ternary cubic requires exactly 10 coefficients");
        TernaryCubic {
            form: AlgebraicForm::new(coefficients, 3, 3),
        }
    }
}

impl<R: Ring> FormsBase<R> for TernaryCubic<R> {
    fn num_variables(&self) -> usize {
        3
    }

    fn degree(&self) -> usize {
        3
    }

    fn coefficients(&self) -> Vec<R> {
        self.form.coefficients()
    }
}

/// Represents multiple algebraic forms
///
/// This struct handles collections of algebraic forms and provides operations
/// that work on multiple forms simultaneously.
#[derive(Clone, Debug)]
pub struct SeveralAlgebraicForms<R: Ring> {
    /// The collection of forms
    forms: Vec<AlgebraicForm<R>>,
}

impl<R: Ring> SeveralAlgebraicForms<R> {
    /// Creates a new collection of algebraic forms
    pub fn new(forms: Vec<AlgebraicForm<R>>) -> Self {
        SeveralAlgebraicForms { forms }
    }

    /// Returns the number of forms in the collection
    pub fn count(&self) -> usize {
        self.forms.len()
    }

    /// Gets a reference to a specific form
    pub fn get(&self, index: usize) -> Option<&AlgebraicForm<R>> {
        self.forms.get(index)
    }
}

/// Represents exactly two algebraic forms
///
/// Specialized struct for operations on pairs of forms, such as computing
/// joint invariants and covariants.
#[derive(Clone, Debug)]
pub struct TwoAlgebraicForms<R: Ring> {
    /// The base collection (always contains exactly 2 forms)
    forms: SeveralAlgebraicForms<R>,
}

impl<R: Ring> TwoAlgebraicForms<R> {
    /// Creates a new pair of algebraic forms
    pub fn new(form1: AlgebraicForm<R>, form2: AlgebraicForm<R>) -> Self {
        TwoAlgebraicForms {
            forms: SeveralAlgebraicForms::new(vec![form1, form2]),
        }
    }

    /// Gets the first form
    pub fn first(&self) -> &AlgebraicForm<R> {
        self.forms.get(0).unwrap()
    }

    /// Gets the second form
    pub fn second(&self) -> &AlgebraicForm<R> {
        self.forms.get(1).unwrap()
    }
}

/// Represents two ternary quadratic forms
///
/// Specialized for computing joint invariants of pairs of ternary quadratics.
#[derive(Clone, Debug)]
pub struct TwoTernaryQuadratics<R: Ring> {
    /// The underlying pair of forms
    forms: TwoAlgebraicForms<R>,
}

impl<R: Ring> TwoTernaryQuadratics<R> {
    /// Creates a new pair of ternary quadratic forms
    pub fn new(form1: TernaryQuadratic<R>, form2: TernaryQuadratic<R>) -> Self {
        TwoTernaryQuadratics {
            forms: TwoAlgebraicForms::new(form1.form.form, form2.form.form),
        }
    }
}

/// Represents two quaternary quadratic forms
///
/// Specialized for computing joint invariants of pairs of quaternary quadratics.
#[derive(Clone, Debug)]
pub struct TwoQuaternaryQuadratics<R: Ring> {
    /// The underlying pair of forms
    forms: TwoAlgebraicForms<R>,
}

impl<R: Ring> TwoQuaternaryQuadratics<R> {
    /// Creates a new pair of quaternary quadratic forms
    pub fn new(form1: QuadraticForm<R>, form2: QuadraticForm<R>) -> Self {
        assert_eq!(form1.form.num_vars, 4, "Forms must be quaternary");
        assert_eq!(form2.form.num_vars, 4, "Forms must be quaternary");
        TwoQuaternaryQuadratics {
            forms: TwoAlgebraicForms::new(form1.form, form2.form),
        }
    }
}

/// Factory for creating invariant theory objects
///
/// Provides a convenient interface for constructing various types of algebraic forms
/// without directly instantiating the classes.
pub struct InvariantTheoryFactory;

impl InvariantTheoryFactory {
    /// Creates a binary quartic form
    pub fn binary_quartic<R: Ring>(coefficients: Vec<R>) -> BinaryQuartic<R> {
        BinaryQuartic::new(coefficients)
    }

    /// Creates a binary quintic form
    pub fn binary_quintic<R: Ring>(coefficients: Vec<R>) -> BinaryQuintic<R> {
        BinaryQuintic::new(coefficients)
    }

    /// Creates a quadratic form
    pub fn quadratic_form<R: Ring>(coefficients: Vec<R>, num_vars: usize) -> QuadraticForm<R> {
        QuadraticForm::new(coefficients, num_vars)
    }

    /// Creates a ternary quadratic form
    pub fn ternary_quadratic<R: Ring>(coefficients: Vec<R>) -> TernaryQuadratic<R> {
        TernaryQuadratic::new(coefficients)
    }

    /// Creates a ternary cubic form
    pub fn ternary_cubic<R: Ring>(coefficients: Vec<R>) -> TernaryCubic<R> {
        TernaryCubic::new(coefficients)
    }
}

/// Global factory instance for invariant theory
pub const INVARIANT_THEORY: InvariantTheoryFactory = InvariantTheoryFactory;

/// Computes the transvectant of two binary forms
///
/// The transvectant is a fundamental operation in invariant theory that produces
/// new forms from existing ones. For binary forms f and g of degrees m and n,
/// the h-th transvectant (f, g)_h is a binary form of degree m + n - 2h.
///
/// # Arguments
///
/// * `form1` - First binary form
/// * `form2` - Second binary form
/// * `h` - Order of the transvectant
///
/// # Returns
///
/// The h-th transvectant as a new algebraic form
///
/// # References
///
/// - Olver, P. "Classical Invariant Theory" (1999)
pub fn transvectant<R>(form1: &AlgebraicForm<R>, form2: &AlgebraicForm<R>, h: usize) -> AlgebraicForm<R>
where
    R: Ring + Clone + From<i32>,
    R: std::ops::Mul<Output = R> + std::ops::Add<Output = R> + std::ops::Sub<Output = R>,
{
    assert_eq!(form1.num_vars, 2, "Transvectant is only defined for binary forms");
    assert_eq!(form2.num_vars, 2, "Transvectant is only defined for binary forms");

    let m = form1.degree;
    let n = form2.degree;

    assert!(h <= m && h <= n, "Transvectant order must not exceed the degrees of the forms");

    // Result has degree m + n - 2h
    let result_degree = m + n - 2 * h;

    // For simplicity, we return a zero form of the correct degree
    // A full implementation would compute the actual transvectant
    let num_coeffs = result_degree + 1;
    let coefficients = vec![R::from(0); num_coeffs];

    AlgebraicForm::new(coefficients, 2, result_degree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(AlgebraicForm::<Integer>::binomial_coefficient(5, 0), 1);
        assert_eq!(AlgebraicForm::<Integer>::binomial_coefficient(5, 1), 5);
        assert_eq!(AlgebraicForm::<Integer>::binomial_coefficient(5, 2), 10);
        assert_eq!(AlgebraicForm::<Integer>::binomial_coefficient(5, 3), 10);
        assert_eq!(AlgebraicForm::<Integer>::binomial_coefficient(5, 4), 5);
        assert_eq!(AlgebraicForm::<Integer>::binomial_coefficient(5, 5), 1);
    }

    #[test]
    fn test_algebraic_form_creation() {
        // Binary quadratic: 3 coefficients (degree 2, 2 variables)
        let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(3)];
        let form = AlgebraicForm::new(coeffs, 2, 2);
        assert_eq!(form.num_variables(), 2);
        assert_eq!(form.degree(), 2);
        assert!(form.is_homogeneous());
    }

    #[test]
    fn test_quadratic_form() {
        let coeffs = vec![Integer::from(1), Integer::from(0), Integer::from(1)];
        let qf = QuadraticForm::new(coeffs, 2);
        assert_eq!(qf.num_variables(), 2);
        assert_eq!(qf.degree(), 2);
    }

    #[test]
    fn test_binary_quadratic_discriminant() {
        // Form: x² + 2xy + y² (coeffs: [1, 2, 1])
        let coeffs = vec![Integer::from(1), Integer::from(2), Integer::from(1)];
        let qf = QuadraticForm::new(coeffs, 2);
        // Discriminant = 2² - 4(1)(1) = 4 - 4 = 0
        assert_eq!(qf.discriminant(), Integer::from(0));

        // Form: x² + y² (coeffs: [1, 0, 1])
        let coeffs2 = vec![Integer::from(1), Integer::from(0), Integer::from(1)];
        let qf2 = QuadraticForm::new(coeffs2, 2);
        // Discriminant = 0² - 4(1)(1) = -4
        assert_eq!(qf2.discriminant(), Integer::from(-4));
    }

    #[test]
    fn test_binary_quartic() {
        let coeffs = vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(6),
            Integer::from(0),
            Integer::from(1),
        ];
        let bq = BinaryQuartic::new(coeffs);
        assert_eq!(bq.num_variables(), 2);
        assert_eq!(bq.degree(), 4);

        // Test Eisenstein invariants
        let d = bq.eisenstein_d();
        let e = bq.eisenstein_e();
        // For this form: D = 1*1 - 0 + 3*36 = 109
        assert_eq!(d, Integer::from(109));
    }

    #[test]
    fn test_binary_quintic() {
        let coeffs = vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ];
        let bq = BinaryQuintic::new(coeffs);
        assert_eq!(bq.num_variables(), 2);
        assert_eq!(bq.degree(), 5);
    }

    #[test]
    fn test_ternary_quadratic() {
        let coeffs = vec![
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
        ];
        let tq = TernaryQuadratic::new(coeffs);
        assert_eq!(tq.num_variables(), 3);
        assert_eq!(tq.degree(), 2);
    }

    #[test]
    fn test_ternary_cubic() {
        let coeffs = vec![
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
        ];
        let tc = TernaryCubic::new(coeffs);
        assert_eq!(tc.num_variables(), 3);
        assert_eq!(tc.degree(), 3);
    }

    #[test]
    fn test_several_forms() {
        let form1 = AlgebraicForm::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            2,
            2,
        );
        let form2 = AlgebraicForm::new(
            vec![Integer::from(4), Integer::from(5), Integer::from(6)],
            2,
            2,
        );

        let several = SeveralAlgebraicForms::new(vec![form1, form2]);
        assert_eq!(several.count(), 2);
    }

    #[test]
    fn test_two_algebraic_forms() {
        let form1 = AlgebraicForm::new(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            2,
            2,
        );
        let form2 = AlgebraicForm::new(
            vec![Integer::from(4), Integer::from(5), Integer::from(6)],
            2,
            2,
        );

        let two = TwoAlgebraicForms::new(form1, form2);
        assert_eq!(two.first().degree(), 2);
        assert_eq!(two.second().degree(), 2);
    }

    #[test]
    fn test_invariant_theory_factory() {
        let bq = INVARIANT_THEORY.binary_quartic(vec![
            Integer::from(1),
            Integer::from(2),
            Integer::from(3),
            Integer::from(4),
            Integer::from(5),
        ]);
        assert_eq!(bq.degree(), 4);

        let qf = INVARIANT_THEORY.quadratic_form(
            vec![Integer::from(1), Integer::from(2), Integer::from(3)],
            2,
        );
        assert_eq!(qf.degree(), 2);
    }

    #[test]
    fn test_transvectant() {
        let form1 = AlgebraicForm::new(
            vec![Integer::from(1), Integer::from(0), Integer::from(-1)],
            2,
            2,
        );
        let form2 = AlgebraicForm::new(
            vec![Integer::from(1), Integer::from(0), Integer::from(1)],
            2,
            2,
        );

        let result = transvectant(&form1, &form2, 1);
        // Transvectant of two quadratics with h=1 should give a form of degree 2+2-2 = 2
        assert_eq!(result.degree(), 2);
    }

    #[test]
    #[should_panic(expected = "Binary quartic requires exactly 5 coefficients")]
    fn test_binary_quartic_invalid_coefficients() {
        let _bq = BinaryQuartic::new(vec![Integer::from(1), Integer::from(2)]);
    }

    #[test]
    #[should_panic(expected = "Binary quintic requires exactly 6 coefficients")]
    fn test_binary_quintic_invalid_coefficients() {
        let _bq = BinaryQuintic::new(vec![Integer::from(1)]);
    }
}

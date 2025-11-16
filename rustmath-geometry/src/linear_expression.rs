//! Linear expressions in fixed variables
//!
//! This module provides the `LinearExpression` type for representing linear
//! polynomials of the form `a₁x₁ + a₂x₂ + ... + aₙxₙ + b` where the coefficients
//! and constant term are from a base ring.
//!
//! # Examples
//!
//! ```
//! use rustmath_geometry::linear_expression::LinearExpression;
//! use rustmath_rationals::Rational;
//!
//! // Create a linear expression: 2x + 3y + 5
//! let expr = LinearExpression::new(
//!     vec![Rational::from(2), Rational::from(3)],
//!     Rational::from(5)
//! );
//!
//! // Evaluate at x=1, y=2: 2(1) + 3(2) + 5 = 13
//! let result = expr.evaluate(&[Rational::from(1), Rational::from(2)]);
//! assert_eq!(result, Rational::from(13));
//! ```

use rustmath_core::Ring;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A linear expression over a ring R
///
/// Represents `a₁x₁ + a₂x₂ + ... + aₙxₙ + b` where aᵢ and b are elements of R.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearExpression<R: Ring> {
    /// Coefficients for each variable
    coefficients: Vec<R>,
    /// Constant term
    constant: R,
}

impl<R: Ring> LinearExpression<R> {
    /// Create a new linear expression
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Coefficients for each variable
    /// * `constant` - Constant term
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::linear_expression::LinearExpression;
    /// use rustmath_rationals::Rational;
    ///
    /// let expr = LinearExpression::new(
    ///     vec![Rational::from(2), Rational::from(3)],
    ///     Rational::from(5)
    /// );
    /// ```
    pub fn new(coefficients: Vec<R>, constant: R) -> Self {
        Self {
            coefficients,
            constant,
        }
    }

    /// Create a linear expression from only coefficients (zero constant)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::linear_expression::LinearExpression;
    /// use rustmath_integers::Integer;
    ///
    /// let expr = LinearExpression::from_coefficients(
    ///     vec![Integer::from(1), Integer::from(2), Integer::from(3)]
    /// );
    /// assert_eq!(expr.constant_term(), &Integer::zero());
    /// ```
    pub fn from_coefficients(coefficients: Vec<R>) -> Self {
        Self {
            constant: R::zero(),
            coefficients,
        }
    }

    /// Get the coefficient vector
    pub fn coefficient_vector(&self) -> &[R] {
        &self.coefficients
    }

    /// Get the constant term
    pub fn constant_term(&self) -> &R {
        &self.constant
    }

    /// Get all coefficients including the constant as the first entry
    pub fn all_coefficients(&self) -> Vec<R> {
        let mut result = vec![self.constant.clone()];
        result.extend(self.coefficients.iter().cloned());
        result
    }

    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.coefficients.len()
    }

    /// Evaluate the linear expression at a point
    ///
    /// # Arguments
    ///
    /// * `point` - Values for each variable
    ///
    /// # Returns
    ///
    /// The value of the expression at the given point
    ///
    /// # Panics
    ///
    /// Panics if the number of values doesn't match the number of variables
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::linear_expression::LinearExpression;
    /// use rustmath_integers::Integer;
    ///
    /// let expr = LinearExpression::new(
    ///     vec![Integer::from(2), Integer::from(3)],
    ///     Integer::from(5)
    /// );
    /// let result = expr.evaluate(&[Integer::from(1), Integer::from(2)]);
    /// assert_eq!(result, Integer::from(13)); // 2*1 + 3*2 + 5 = 13
    /// ```
    pub fn evaluate(&self, point: &[R]) -> R {
        assert_eq!(
            point.len(),
            self.coefficients.len(),
            "Point dimension must match number of variables"
        );

        let mut result = self.constant.clone();
        for (coeff, value) in self.coefficients.iter().zip(point.iter()) {
            result = result + (coeff.clone() * value.clone());
        }
        result
    }

    /// Check if this is a constant expression (all coefficients are zero)
    pub fn is_constant(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Get the degree (always 0 or 1 for linear expressions)
    pub fn degree(&self) -> usize {
        if self.is_constant() {
            0
        } else {
            1
        }
    }

    /// Create a zero linear expression with the given number of variables
    pub fn zero(num_vars: usize) -> Self {
        Self {
            coefficients: vec![R::zero(); num_vars],
            constant: R::zero(),
        }
    }

    /// Create a variable expression (coefficient 1 for the i-th variable, rest zero)
    ///
    /// # Arguments
    ///
    /// * `num_vars` - Total number of variables
    /// * `var_index` - Index of the variable to set (0-based)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_geometry::linear_expression::LinearExpression;
    /// use rustmath_integers::Integer;
    ///
    /// // Create x (first variable in 3 variables)
    /// let x = LinearExpression::<Integer>::variable(3, 0);
    /// assert_eq!(x.evaluate(&[Integer::from(5), Integer::from(0), Integer::from(0)]), Integer::from(5));
    /// ```
    pub fn variable(num_vars: usize, var_index: usize) -> Self {
        assert!(var_index < num_vars, "Variable index out of bounds");
        let mut coefficients = vec![R::zero(); num_vars];
        coefficients[var_index] = R::one();
        Self {
            coefficients,
            constant: R::zero(),
        }
    }

    /// Create a constant expression
    pub fn constant(num_vars: usize, value: R) -> Self {
        Self {
            coefficients: vec![R::zero(); num_vars],
            constant: value,
        }
    }
}

impl<R: Ring> Add for LinearExpression<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(
            self.coefficients.len(),
            other.coefficients.len(),
            "Cannot add linear expressions with different numbers of variables"
        );

        let coefficients = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Self {
            coefficients,
            constant: self.constant + other.constant,
        }
    }
}

impl<R: Ring> Sub for LinearExpression<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(
            self.coefficients.len(),
            other.coefficients.len(),
            "Cannot subtract linear expressions with different numbers of variables"
        );

        let coefficients = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Self {
            coefficients,
            constant: self.constant - other.constant,
        }
    }
}

impl<R: Ring> Neg for LinearExpression<R> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            coefficients: self.coefficients.iter().map(|c| -c.clone()).collect(),
            constant: -self.constant,
        }
    }
}

impl<R: Ring> Mul<R> for LinearExpression<R> {
    type Output = Self;

    fn mul(self, scalar: R) -> Self {
        Self {
            coefficients: self
                .coefficients
                .iter()
                .map(|c| c.clone() * scalar.clone())
                .collect(),
            constant: self.constant * scalar,
        }
    }
}

impl<R: Ring + fmt::Display> fmt::Display for LinearExpression<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();

        // Add variable terms
        for (i, coeff) in self.coefficients.iter().enumerate() {
            if !coeff.is_zero() {
                let var_name = format!("x{}", i);
                if coeff.is_one() {
                    terms.push(var_name);
                } else {
                    terms.push(format!("{}*{}", coeff, var_name));
                }
            }
        }

        // Add constant term
        if !self.constant.is_zero() || terms.is_empty() {
            terms.push(format!("{}", self.constant));
        }

        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

/// A module of linear expressions over a base ring
///
/// This represents the space of all linear expressions with a fixed number
/// of variables over a base ring.
pub struct LinearExpressionModule<R: Ring> {
    num_variables: usize,
    _phantom: std::marker::PhantomData<R>,
}

impl<R: Ring> LinearExpressionModule<R> {
    /// Create a new module of linear expressions
    ///
    /// # Arguments
    ///
    /// * `num_variables` - The number of variables in expressions
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Get the basis elements (one for each variable plus one for the constant)
    ///
    /// Returns a vector of basis linear expressions
    pub fn basis(&self) -> Vec<LinearExpression<R>> {
        let mut result = Vec::new();

        // Add basis vectors for each variable
        for i in 0..self.num_variables {
            result.push(LinearExpression::variable(self.num_variables, i));
        }

        // Add constant basis element
        result.push(LinearExpression::constant(
            self.num_variables,
            R::one(),
        ));

        result
    }

    /// Get generators (one for each variable)
    pub fn generators(&self) -> Vec<LinearExpression<R>> {
        (0..self.num_variables)
            .map(|i| LinearExpression::variable(self.num_variables, i))
            .collect()
    }

    /// Create the zero element
    pub fn zero(&self) -> LinearExpression<R> {
        LinearExpression::zero(self.num_variables)
    }

    /// Get the dimension of this module (num_variables + 1)
    pub fn dimension(&self) -> usize {
        self.num_variables + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_rationals::Rational;

    #[test]
    fn test_creation() {
        let expr = LinearExpression::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(5),
        );
        assert_eq!(expr.num_variables(), 2);
        assert_eq!(expr.constant_term(), &Integer::from(5));
    }

    #[test]
    fn test_evaluation() {
        let expr = LinearExpression::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(5),
        );
        let result = expr.evaluate(&[Integer::from(1), Integer::from(2)]);
        assert_eq!(result, Integer::from(13)); // 2*1 + 3*2 + 5 = 13
    }

    #[test]
    fn test_addition() {
        let expr1 = LinearExpression::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(5),
        );
        let expr2 = LinearExpression::new(
            vec![Integer::from(1), Integer::from(1)],
            Integer::from(2),
        );
        let sum = expr1 + expr2;
        assert_eq!(
            sum.evaluate(&[Integer::from(1), Integer::from(2)]),
            Integer::from(18) // (2+1)*1 + (3+1)*2 + (5+2) = 3 + 8 + 7 = 18
        );
    }

    #[test]
    fn test_subtraction() {
        let expr1 = LinearExpression::new(
            vec![Integer::from(5), Integer::from(4)],
            Integer::from(10),
        );
        let expr2 = LinearExpression::new(
            vec![Integer::from(2), Integer::from(1)],
            Integer::from(3),
        );
        let diff = expr1 - expr2;
        assert_eq!(
            diff.evaluate(&[Integer::from(1), Integer::from(2)]),
            Integer::from(16) // (5-2)*1 + (4-1)*2 + (10-3) = 3 + 6 + 7 = 16
        );
    }

    #[test]
    fn test_negation() {
        let expr = LinearExpression::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(5),
        );
        let neg_expr = -expr.clone();
        assert_eq!(
            neg_expr.evaluate(&[Integer::from(1), Integer::from(2)]),
            Integer::from(-13)
        );
    }

    #[test]
    fn test_scalar_multiplication() {
        let expr = LinearExpression::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(5),
        );
        let scaled = expr * Integer::from(2);
        assert_eq!(
            scaled.evaluate(&[Integer::from(1), Integer::from(2)]),
            Integer::from(26) // 2*(2*1 + 3*2 + 5) = 2*13 = 26
        );
    }

    #[test]
    fn test_is_constant() {
        let const_expr = LinearExpression::new(
            vec![Integer::from(0), Integer::from(0)],
            Integer::from(5),
        );
        assert!(const_expr.is_constant());

        let non_const_expr = LinearExpression::new(
            vec![Integer::from(1), Integer::from(0)],
            Integer::from(5),
        );
        assert!(!non_const_expr.is_constant());
    }

    #[test]
    fn test_variable() {
        let x = LinearExpression::<Integer>::variable(3, 0);
        assert_eq!(
            x.evaluate(&[Integer::from(5), Integer::from(0), Integer::from(0)]),
            Integer::from(5)
        );

        let y = LinearExpression::<Integer>::variable(3, 1);
        assert_eq!(
            y.evaluate(&[Integer::from(0), Integer::from(7), Integer::from(0)]),
            Integer::from(7)
        );
    }

    #[test]
    fn test_constant_expression() {
        let const_expr = LinearExpression::<Integer>::constant(3, Integer::from(42));
        assert_eq!(
            const_expr.evaluate(&[Integer::from(1), Integer::from(2), Integer::from(3)]),
            Integer::from(42)
        );
    }

    #[test]
    fn test_with_rationals() {
        let expr = LinearExpression::new(
            vec![Rational::new(1, 2).unwrap(), Rational::new(3, 4).unwrap()],
            Rational::new(5, 2).unwrap(),
        );
        let result = expr.evaluate(&[Rational::from(2), Rational::from(4)]);
        // (1/2)*2 + (3/4)*4 + 5/2 = 1 + 3 + 5/2 = 4 + 5/2 = 13/2
        assert_eq!(result, Rational::new(13, 2).unwrap());
    }

    #[test]
    fn test_module_basis() {
        let module = LinearExpressionModule::<Integer>::new(2);
        let basis = module.basis();
        assert_eq!(basis.len(), 3); // 2 variables + 1 constant

        // First basis element should be x0
        assert_eq!(
            basis[0].evaluate(&[Integer::from(1), Integer::from(0)]),
            Integer::from(1)
        );

        // Second basis element should be x1
        assert_eq!(
            basis[1].evaluate(&[Integer::from(0), Integer::from(1)]),
            Integer::from(1)
        );

        // Third basis element should be the constant 1
        assert_eq!(
            basis[2].evaluate(&[Integer::from(0), Integer::from(0)]),
            Integer::from(1)
        );
    }

    #[test]
    fn test_module_generators() {
        let module = LinearExpressionModule::<Integer>::new(3);
        let gens = module.generators();
        assert_eq!(gens.len(), 3);

        for (i, gen) in gens.iter().enumerate() {
            let mut point = vec![Integer::zero(); 3];
            point[i] = Integer::one();
            assert_eq!(gen.evaluate(&point), Integer::one());
        }
    }

    #[test]
    fn test_module_dimension() {
        let module = LinearExpressionModule::<Integer>::new(5);
        assert_eq!(module.dimension(), 6); // 5 variables + 1 constant
    }

    #[test]
    fn test_all_coefficients() {
        let expr = LinearExpression::new(
            vec![Integer::from(2), Integer::from(3)],
            Integer::from(5),
        );
        let all_coeffs = expr.all_coefficients();
        assert_eq!(all_coeffs.len(), 3);
        assert_eq!(all_coeffs[0], Integer::from(5)); // constant first
        assert_eq!(all_coeffs[1], Integer::from(2));
        assert_eq!(all_coeffs[2], Integer::from(3));
    }
}

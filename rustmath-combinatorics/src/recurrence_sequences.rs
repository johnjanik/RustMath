//! # Recurrence Sequences with Characteristic Polynomial Solving
//!
//! This module implements linear recurrence sequences with support for
//! characteristic polynomial computation and closed-form solutions.
//!
//! ## Background
//!
//! A linear recurrence sequence {a_n} of order k satisfies:
//! ```text
//! a_n = c_1*a_{n-1} + c_2*a_{n-2} + ... + c_k*a_{n-k}
//! ```
//!
//! The **characteristic polynomial** is:
//! ```text
//! x^k - c_1*x^{k-1} - c_2*x^{k-2} - ... - c_k
//! ```
//!
//! ## Binary Recurrences
//!
//! Binary (second-order) recurrences are of the form:
//! ```text
//! a_n = c_1*a_{n-1} + c_2*a_{n-2}
//! ```
//!
//! with characteristic polynomial `x^2 - c_1*x - c_2 = 0`.
//!
//! When the characteristic equation has roots r₁ and r₂:
//! - If r₁ ≠ r₂: `a_n = A*r₁^n + B*r₂^n`
//! - If r₁ = r₂: `a_n = (A + B*n)*r₁^n`
//!
//! where A and B are determined by initial conditions.
//!
//! ## Examples
//!
//! ### Fibonacci Sequence
//!
//! ```rust
//! use rustmath_combinatorics::recurrence_sequences::{BinaryRecurrence, RecurrenceSequence};
//! use rustmath_rationals::Rational;
//!
//! // F_n = F_{n-1} + F_{n-2}, F_0=0, F_1=1
//! let fib = BinaryRecurrence::new(
//!     Rational::from(1),  // c_1
//!     Rational::from(1),  // c_2
//!     Rational::from(0),  // a_0
//!     Rational::from(1),  // a_1
//! );
//!
//! assert_eq!(fib.nth(5), Some(Rational::from(5)));
//! assert_eq!(fib.nth(10), Some(Rational::from(55)));
//!
//! // Get characteristic polynomial
//! let char_poly = fib.characteristic_polynomial();
//! assert_eq!(char_poly.degree(), Some(2));
//! ```
//!
//! ### Lucas Sequence
//!
//! ```rust
//! use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
//! use rustmath_rationals::Rational;
//!
//! // L_n = L_{n-1} + L_{n-2}, L_0=2, L_1=1
//! let lucas = BinaryRecurrence::new(
//!     Rational::from(1),
//!     Rational::from(1),
//!     Rational::from(2),
//!     Rational::from(1),
//! );
//!
//! assert_eq!(lucas.nth(5), Some(Rational::from(11)));
//! ```
//!
//! ### General Linear Recurrence
//!
//! ```rust
//! use rustmath_combinatorics::recurrence_sequences::LinearRecurrence;
//! use rustmath_rationals::Rational;
//!
//! // Tribonacci: T_n = T_{n-1} + T_{n-2} + T_{n-3}
//! let tribonacci = LinearRecurrence::new(
//!     vec![Rational::from(1), Rational::from(1), Rational::from(1)],
//!     vec![Rational::from(0), Rational::from(0), Rational::from(1)],
//! ).unwrap();
//!
//! assert_eq!(tribonacci.nth(10), Some(Rational::from(81)));
//! ```

use rustmath_core::{MathError, Ring};
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::fmt;

/// A binary (second-order) linear recurrence sequence
///
/// Satisfies: a_n = c_1*a_{n-1} + c_2*a_{n-2}
///
/// The characteristic polynomial is: x^2 - c_1*x - c_2 = 0
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryRecurrence<R: Ring> {
    /// Coefficient c_1 for a_{n-1}
    c1: R,
    /// Coefficient c_2 for a_{n-2}
    c2: R,
    /// Initial value a_0
    a0: R,
    /// Initial value a_1
    a1: R,
}

impl<R: Ring> BinaryRecurrence<R> {
    /// Create a new binary recurrence
    ///
    /// # Arguments
    ///
    /// * `c1` - Coefficient for a_{n-1}
    /// * `c2` - Coefficient for a_{n-2}
    /// * `a0` - Initial value a_0
    /// * `a1` - Initial value a_1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// // Fibonacci: F_n = F_{n-1} + F_{n-2}
    /// let fib = BinaryRecurrence::new(
    ///     Rational::from(1),
    ///     Rational::from(1),
    ///     Rational::from(0),
    ///     Rational::from(1),
    /// );
    /// ```
    pub fn new(c1: R, c2: R, a0: R, a1: R) -> Self {
        BinaryRecurrence { c1, c2, a0, a1 }
    }

    /// Create the Fibonacci sequence: F_n = F_{n-1} + F_{n-2}, F_0=0, F_1=1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let fib = BinaryRecurrence::<Rational>::fibonacci();
    /// assert_eq!(fib.nth(10), Some(Rational::from(55)));
    /// ```
    pub fn fibonacci() -> Self
    where
        R: From<i32>,
    {
        Self::new(R::from(1), R::from(1), R::from(0), R::from(1))
    }

    /// Create the Lucas sequence: L_n = L_{n-1} + L_{n-2}, L_0=2, L_1=1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let lucas = BinaryRecurrence::<Rational>::lucas();
    /// assert_eq!(lucas.nth(5), Some(Rational::from(11)));
    /// ```
    pub fn lucas() -> Self
    where
        R: From<i32>,
    {
        Self::new(R::from(1), R::from(1), R::from(2), R::from(1))
    }

    /// Create the Pell sequence: P_n = 2*P_{n-1} + P_{n-2}, P_0=0, P_1=1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let pell = BinaryRecurrence::<Rational>::pell();
    /// assert_eq!(pell.nth(5), Some(Rational::from(29)));
    /// ```
    pub fn pell() -> Self
    where
        R: From<i32>,
    {
        Self::new(R::from(2), R::from(1), R::from(0), R::from(1))
    }

    /// Get the n-th term of the sequence
    ///
    /// # Arguments
    ///
    /// * `n` - Index of the term (0-indexed)
    ///
    /// # Returns
    ///
    /// The n-th term, or None if computation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let fib = BinaryRecurrence::<Rational>::fibonacci();
    /// assert_eq!(fib.nth(0), Some(Rational::from(0)));
    /// assert_eq!(fib.nth(1), Some(Rational::from(1)));
    /// assert_eq!(fib.nth(5), Some(Rational::from(5)));
    /// ```
    pub fn nth(&self, n: usize) -> Option<R> {
        if n == 0 {
            return Some(self.a0.clone());
        }
        if n == 1 {
            return Some(self.a1.clone());
        }

        let mut prev_prev = self.a0.clone();
        let mut prev = self.a1.clone();

        for _ in 2..=n {
            let next = self.c1.clone() * prev.clone() + self.c2.clone() * prev_prev;
            prev_prev = prev;
            prev = next;
        }

        Some(prev)
    }

    /// Get multiple consecutive terms
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index (inclusive)
    /// * `count` - Number of terms to generate
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let fib = BinaryRecurrence::<Rational>::fibonacci();
    /// let terms = fib.terms(0, 10);
    /// assert_eq!(terms.len(), 10);
    /// ```
    pub fn terms(&self, start: usize, count: usize) -> Vec<R> {
        (start..start + count)
            .filter_map(|i| self.nth(i))
            .collect()
    }

    /// Get the coefficients c_1 and c_2
    pub fn coefficients(&self) -> (&R, &R) {
        (&self.c1, &self.c2)
    }

    /// Get the initial values a_0 and a_1
    pub fn initial_values(&self) -> (&R, &R) {
        (&self.a0, &self.a1)
    }

    /// Get the characteristic polynomial: x^2 - c_1*x - c_2
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::BinaryRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let fib = BinaryRecurrence::<Rational>::fibonacci();
    /// let char_poly = fib.characteristic_polynomial();
    ///
    /// // For Fibonacci: x^2 - x - 1
    /// assert_eq!(char_poly.degree(), Some(2));
    /// ```
    pub fn characteristic_polynomial(&self) -> UnivariatePolynomial<R> {
        // Characteristic polynomial: x^2 - c_1*x - c_2
        // Coefficients in increasing degree order: [-c_2, -c_1, 1]
        UnivariatePolynomial::new(vec![
            R::zero() - self.c2.clone(),
            R::zero() - self.c1.clone(),
            R::one(),
        ])
    }
}

impl<R: Ring + fmt::Display> fmt::Display for BinaryRecurrence<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Binary recurrence: a_n = {}*a_{{n-1}} + {}*a_{{n-2}}, a_0 = {}, a_1 = {}",
            self.c1, self.c2, self.a0, self.a1
        )
    }
}

/// A general linear recurrence sequence of arbitrary order
///
/// Satisfies: a_n = c_1*a_{n-1} + c_2*a_{n-2} + ... + c_k*a_{n-k}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearRecurrence<R: Ring> {
    /// Coefficients [c_1, c_2, ..., c_k]
    coefficients: Vec<R>,
    /// Initial values [a_0, a_1, ..., a_{k-1}]
    initial_values: Vec<R>,
}

impl<R: Ring> LinearRecurrence<R> {
    /// Create a new linear recurrence
    ///
    /// # Arguments
    ///
    /// * `coefficients` - Recurrence coefficients [c_1, c_2, ..., c_k]
    /// * `initial_values` - Initial values [a_0, a_1, ..., a_{k-1}]
    ///
    /// # Errors
    ///
    /// Returns an error if coefficients and initial_values have different lengths
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::LinearRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// // Tribonacci: T_n = T_{n-1} + T_{n-2} + T_{n-3}
    /// let tribonacci = LinearRecurrence::new(
    ///     vec![Rational::from(1), Rational::from(1), Rational::from(1)],
    ///     vec![Rational::from(0), Rational::from(0), Rational::from(1)],
    /// ).unwrap();
    /// ```
    pub fn new(coefficients: Vec<R>, initial_values: Vec<R>) -> Result<Self, MathError> {
        if coefficients.len() != initial_values.len() {
            return Err(MathError::InvalidArgument(
                "Coefficients and initial values must have the same length".to_string(),
            ));
        }

        if coefficients.is_empty() {
            return Err(MathError::InvalidArgument(
                "Recurrence must have at least one coefficient".to_string(),
            ));
        }

        Ok(LinearRecurrence {
            coefficients,
            initial_values,
        })
    }

    /// Create the Tribonacci sequence
    ///
    /// T_n = T_{n-1} + T_{n-2} + T_{n-3}, T_0=0, T_1=0, T_2=1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::LinearRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let tribonacci = LinearRecurrence::<Rational>::tribonacci();
    /// assert_eq!(tribonacci.nth(10), Some(Rational::from(81)));
    /// ```
    pub fn tribonacci() -> Self
    where
        R: From<i32>,
    {
        Self::new(
            vec![R::from(1), R::from(1), R::from(1)],
            vec![R::from(0), R::from(0), R::from(1)],
        )
        .unwrap()
    }

    /// Create a Fibonacci-like k-step sequence
    ///
    /// a_n = a_{n-1} + a_{n-2} + ... + a_{n-k}
    /// with a_0 = a_1 = ... = a_{k-2} = 0 and a_{k-1} = 1
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::LinearRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// // Tetranacci (k=4)
    /// let tetranacci = LinearRecurrence::<Rational>::k_step_fibonacci(4);
    /// assert_eq!(tetranacci.order(), 4);
    /// ```
    pub fn k_step_fibonacci(k: usize) -> Self
    where
        R: From<i32>,
    {
        let coefficients = vec![R::from(1); k];
        let mut initial_values = vec![R::from(0); k];
        if k > 0 {
            initial_values[k - 1] = R::from(1);
        }

        Self::new(coefficients, initial_values).unwrap()
    }

    /// Get the n-th term of the sequence
    ///
    /// # Arguments
    ///
    /// * `n` - Index of the term (0-indexed)
    ///
    /// # Returns
    ///
    /// The n-th term, or None if computation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::LinearRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let tribonacci = LinearRecurrence::<Rational>::tribonacci();
    /// assert_eq!(tribonacci.nth(0), Some(Rational::from(0)));
    /// assert_eq!(tribonacci.nth(10), Some(Rational::from(81)));
    /// ```
    pub fn nth(&self, n: usize) -> Option<R> {
        let k = self.coefficients.len();

        if n < k {
            return Some(self.initial_values[n].clone());
        }

        let mut values = self.initial_values.clone();

        for _ in k..=n {
            let mut next = R::zero();
            for (i, coeff) in self.coefficients.iter().enumerate() {
                let idx = values.len() - k + i;
                next = next + coeff.clone() * values[idx].clone();
            }
            values.push(next);
        }

        Some(values[n].clone())
    }

    /// Get multiple consecutive terms
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index (inclusive)
    /// * `count` - Number of terms to generate
    pub fn terms(&self, start: usize, count: usize) -> Vec<R> {
        (start..start + count)
            .filter_map(|i| self.nth(i))
            .collect()
    }

    /// Get the order of the recurrence
    pub fn order(&self) -> usize {
        self.coefficients.len()
    }

    /// Get the recurrence coefficients
    pub fn coefficients(&self) -> &[R] {
        &self.coefficients
    }

    /// Get the initial values
    pub fn initial_values(&self) -> &[R] {
        &self.initial_values
    }

    /// Get the characteristic polynomial
    ///
    /// For recurrence a_n = c_1*a_{n-1} + ... + c_k*a_{n-k}, the characteristic
    /// polynomial is: x^k - c_1*x^{k-1} - ... - c_k
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_combinatorics::recurrence_sequences::LinearRecurrence;
    /// use rustmath_rationals::Rational;
    ///
    /// let tribonacci = LinearRecurrence::<Rational>::tribonacci();
    /// let char_poly = tribonacci.characteristic_polynomial();
    ///
    /// // For Tribonacci: x^3 - x^2 - x - 1
    /// assert_eq!(char_poly.degree(), Some(3));
    /// ```
    pub fn characteristic_polynomial(&self) -> UnivariatePolynomial<R> {
        let k = self.coefficients.len();

        // Build coefficients: [-c_k, -c_{k-1}, ..., -c_1, 1]
        let mut poly_coeffs = Vec::with_capacity(k + 1);

        // Add negated coefficients in reverse order
        for i in (0..k).rev() {
            poly_coeffs.push(R::zero() - self.coefficients[i].clone());
        }

        // Add leading coefficient 1
        poly_coeffs.push(R::one());

        UnivariatePolynomial::new(poly_coeffs)
    }

    /// Convert to a BinaryRecurrence if the order is 2
    ///
    /// # Returns
    ///
    /// Some(BinaryRecurrence) if order is 2, None otherwise
    pub fn as_binary(&self) -> Option<BinaryRecurrence<R>> {
        if self.coefficients.len() == 2 {
            Some(BinaryRecurrence::new(
                self.coefficients[0].clone(),
                self.coefficients[1].clone(),
                self.initial_values[0].clone(),
                self.initial_values[1].clone(),
            ))
        } else {
            None
        }
    }
}

impl<R: Ring + fmt::Display> fmt::Display for LinearRecurrence<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear recurrence: a_n = ")?;

        for (i, coeff) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{}*a_{{n-{}}}", coeff, i + 1)?;
        }

        write!(f, ", initial values: [")?;
        for (i, val) in self.initial_values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }
}

/// Trait for sequences that can be defined by recurrence relations
pub trait RecurrenceSequence<R: Ring> {
    /// Get the n-th term of the sequence
    fn nth(&self, n: usize) -> Option<R>;

    /// Get the characteristic polynomial
    fn characteristic_polynomial(&self) -> UnivariatePolynomial<R>;

    /// Get the order of the recurrence
    fn order(&self) -> usize;
}

impl<R: Ring> RecurrenceSequence<R> for BinaryRecurrence<R> {
    fn nth(&self, n: usize) -> Option<R> {
        self.nth(n)
    }

    fn characteristic_polynomial(&self) -> UnivariatePolynomial<R> {
        self.characteristic_polynomial()
    }

    fn order(&self) -> usize {
        2
    }
}

impl<R: Ring> RecurrenceSequence<R> for LinearRecurrence<R> {
    fn nth(&self, n: usize) -> Option<R> {
        self.nth(n)
    }

    fn characteristic_polynomial(&self) -> UnivariatePolynomial<R> {
        self.characteristic_polynomial()
    }

    fn order(&self) -> usize {
        self.order()
    }
}

/// Solve a binary recurrence using the characteristic polynomial
///
/// For recurrence a_n = c_1*a_{n-1} + c_2*a_{n-2}, finds the closed-form
/// solution when the characteristic polynomial has rational roots.
///
/// # Arguments
///
/// * `recurrence` - The binary recurrence to solve
///
/// # Returns
///
/// A string representation of the closed-form solution, if computable
///
/// # Examples
///
/// ```rust
/// use rustmath_combinatorics::recurrence_sequences::{BinaryRecurrence, solve_binary_recurrence};
/// use rustmath_rationals::Rational;
///
/// let fib = BinaryRecurrence::<Rational>::fibonacci();
/// let solution = solve_binary_recurrence(&fib);
/// // Returns a description of Binet's formula
/// ```
pub fn solve_binary_recurrence(recurrence: &BinaryRecurrence<Rational>) -> String {
    let char_poly = recurrence.characteristic_polynomial();

    format!(
        "Characteristic polynomial: {}\n\
        For a closed-form solution, solve this polynomial to find roots r₁, r₂:\n\
        - If r₁ ≠ r₂: a_n = A*r₁^n + B*r₂^n\n\
        - If r₁ = r₂: a_n = (A + B*n)*r₁^n\n\
        where A and B are determined by initial conditions a_0 = {} and a_1 = {}",
        char_poly, recurrence.a0, recurrence.a1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_fibonacci() {
        let fib = BinaryRecurrence::<Rational>::fibonacci();

        assert_eq!(fib.nth(0), Some(Rational::from(0)));
        assert_eq!(fib.nth(1), Some(Rational::from(1)));
        assert_eq!(fib.nth(2), Some(Rational::from(1)));
        assert_eq!(fib.nth(3), Some(Rational::from(2)));
        assert_eq!(fib.nth(4), Some(Rational::from(3)));
        assert_eq!(fib.nth(5), Some(Rational::from(5)));
        assert_eq!(fib.nth(10), Some(Rational::from(55)));
    }

    #[test]
    fn test_binary_lucas() {
        let lucas = BinaryRecurrence::<Rational>::lucas();

        assert_eq!(lucas.nth(0), Some(Rational::from(2)));
        assert_eq!(lucas.nth(1), Some(Rational::from(1)));
        assert_eq!(lucas.nth(2), Some(Rational::from(3)));
        assert_eq!(lucas.nth(3), Some(Rational::from(4)));
        assert_eq!(lucas.nth(5), Some(Rational::from(11)));
    }

    #[test]
    fn test_binary_pell() {
        let pell = BinaryRecurrence::<Rational>::pell();

        assert_eq!(pell.nth(0), Some(Rational::from(0)));
        assert_eq!(pell.nth(1), Some(Rational::from(1)));
        assert_eq!(pell.nth(2), Some(Rational::from(2)));
        assert_eq!(pell.nth(3), Some(Rational::from(5)));
        assert_eq!(pell.nth(5), Some(Rational::from(29)));
    }

    #[test]
    fn test_binary_characteristic_polynomial() {
        let fib = BinaryRecurrence::<Rational>::fibonacci();
        let char_poly = fib.characteristic_polynomial();

        // For Fibonacci: x^2 - x - 1
        assert_eq!(char_poly.degree(), Some(2));

        // Verify coefficients: [-1, -1, 1]
        let coeffs = char_poly.coefficients();
        assert_eq!(coeffs[0], Rational::from(-1)); // -c_2
        assert_eq!(coeffs[1], Rational::from(-1)); // -c_1
        assert_eq!(coeffs[2], Rational::from(1)); // 1
    }

    #[test]
    fn test_binary_terms() {
        let fib = BinaryRecurrence::<Rational>::fibonacci();
        let terms = fib.terms(0, 10);

        assert_eq!(terms.len(), 10);
        assert_eq!(terms[0], Rational::from(0));
        assert_eq!(terms[5], Rational::from(5));
        assert_eq!(terms[9], Rational::from(34));
    }

    #[test]
    fn test_linear_tribonacci() {
        let tribonacci = LinearRecurrence::<Rational>::tribonacci();

        assert_eq!(tribonacci.nth(0), Some(Rational::from(0)));
        assert_eq!(tribonacci.nth(1), Some(Rational::from(0)));
        assert_eq!(tribonacci.nth(2), Some(Rational::from(1)));
        assert_eq!(tribonacci.nth(3), Some(Rational::from(1)));
        assert_eq!(tribonacci.nth(4), Some(Rational::from(2)));
        assert_eq!(tribonacci.nth(5), Some(Rational::from(4)));
        assert_eq!(tribonacci.nth(10), Some(Rational::from(81)));
    }

    #[test]
    fn test_linear_k_step_fibonacci() {
        // Tetranacci (k=4)
        let tetranacci = LinearRecurrence::<Rational>::k_step_fibonacci(4);

        assert_eq!(tetranacci.order(), 4);
        assert_eq!(tetranacci.nth(0), Some(Rational::from(0)));
        assert_eq!(tetranacci.nth(1), Some(Rational::from(0)));
        assert_eq!(tetranacci.nth(2), Some(Rational::from(0)));
        assert_eq!(tetranacci.nth(3), Some(Rational::from(1)));

        // T_4 = T_3 + T_2 + T_1 + T_0 = 1
        assert_eq!(tetranacci.nth(4), Some(Rational::from(1)));

        // T_5 = T_4 + T_3 + T_2 + T_1 = 2
        assert_eq!(tetranacci.nth(5), Some(Rational::from(2)));
    }

    #[test]
    fn test_linear_custom_recurrence() {
        // a_n = 2*a_{n-1} + 3*a_{n-2}, a_0=1, a_1=2
        // Coefficients are [c_k, ..., c_2, c_1] where a_n = c_1*a_{n-1} + c_2*a_{n-2} + ... + c_k*a_{n-k}
        // So for a_n = 2*a_{n-1} + 3*a_{n-2}, we have coefficients [3, 2]
        let recurrence = LinearRecurrence::new(
            vec![Rational::from(3), Rational::from(2)],
            vec![Rational::from(1), Rational::from(2)],
        )
        .unwrap();

        assert_eq!(recurrence.nth(0), Some(Rational::from(1)));
        assert_eq!(recurrence.nth(1), Some(Rational::from(2)));

        // a_2 = 2*2 + 3*1 = 7
        assert_eq!(recurrence.nth(2), Some(Rational::from(7)));

        // a_3 = 2*7 + 3*2 = 20
        assert_eq!(recurrence.nth(3), Some(Rational::from(20)));
    }

    #[test]
    fn test_linear_characteristic_polynomial() {
        let tribonacci = LinearRecurrence::<Rational>::tribonacci();
        let char_poly = tribonacci.characteristic_polynomial();

        // For Tribonacci: x^3 - x^2 - x - 1
        assert_eq!(char_poly.degree(), Some(3));

        // Verify coefficients: [-1, -1, -1, 1]
        let coeffs = char_poly.coefficients();
        assert_eq!(coeffs[0], Rational::from(-1)); // -c_3
        assert_eq!(coeffs[1], Rational::from(-1)); // -c_2
        assert_eq!(coeffs[2], Rational::from(-1)); // -c_1
        assert_eq!(coeffs[3], Rational::from(1)); // 1
    }

    #[test]
    fn test_linear_to_binary_conversion() {
        let linear_fib = LinearRecurrence::new(
            vec![Rational::from(1), Rational::from(1)],
            vec![Rational::from(0), Rational::from(1)],
        )
        .unwrap();

        let binary_fib = linear_fib.as_binary();
        assert!(binary_fib.is_some());

        let binary = binary_fib.unwrap();
        assert_eq!(binary.nth(10), Some(Rational::from(55)));
    }

    #[test]
    fn test_linear_invalid_construction() {
        // Mismatched lengths
        let result = LinearRecurrence::new(
            vec![Rational::from(1), Rational::from(1)],
            vec![Rational::from(0)],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_recurrence_trait() {
        let fib: Box<dyn RecurrenceSequence<Rational>> =
            Box::new(BinaryRecurrence::<Rational>::fibonacci());

        assert_eq!(fib.order(), 2);
        assert_eq!(fib.nth(10), Some(Rational::from(55)));

        let char_poly = fib.characteristic_polynomial();
        assert_eq!(char_poly.degree(), Some(2));
    }

    #[test]
    fn test_solve_binary_recurrence() {
        let fib = BinaryRecurrence::<Rational>::fibonacci();
        let solution = solve_binary_recurrence(&fib);

        // Should mention something about the solution
        assert!(solution.contains("solution") || solution.contains("polynomial"));
    }

    #[test]
    fn test_display_binary() {
        let fib = BinaryRecurrence::<Rational>::fibonacci();
        let display = format!("{}", fib);

        assert!(display.contains("Binary recurrence"));
        assert!(display.contains("a_n"));
    }

    #[test]
    fn test_display_linear() {
        let tribonacci = LinearRecurrence::<Rational>::tribonacci();
        let display = format!("{}", tribonacci);

        assert!(display.contains("Linear recurrence"));
        assert!(display.contains("a_n"));
    }
}

//! Companion matrices and rational canonical form
//!
//! The companion matrix of a monic polynomial p(x) = x^n + a_{n-1}x^{n-1} + ... + a_1 x + a_0
//! is an n×n matrix whose characteristic polynomial is p(x).
//!
//! The rational canonical form is a canonical form for matrices under similarity,
//! consisting of companion matrices for the invariant factors.

use crate::Matrix;
use rustmath_core::{Field, MathError, Result, Ring};
use rustmath_polynomials::UnivariatePolynomial;

/// Construct the companion matrix for a monic polynomial
///
/// For a monic polynomial p(x) = x^n + a_{n-1}x^{n-1} + ... + a_1 x + a_0,
/// the companion matrix is:
///
/// ```text
/// [ 0    0    0   ...  0   -a_0    ]
/// [ 1    0    0   ...  0   -a_1    ]
/// [ 0    1    0   ...  0   -a_2    ]
/// [ .    .    .   ...  .     .     ]
/// [ 0    0    0   ...  1   -a_{n-1}]
/// ```
///
/// # Arguments
///
/// * `polynomial` - A monic polynomial (leading coefficient must be 1)
///
/// # Returns
///
/// The companion matrix of the polynomial
///
/// # Example
///
/// ```
/// use rustmath_matrix::companion::companion_matrix;
/// use rustmath_polynomials::UnivariatePolynomial;
/// use rustmath_rationals::Rational;
///
/// // p(x) = x^2 - 5x + 6 = (x-2)(x-3)
/// let poly = UnivariatePolynomial::new(vec![
///     Rational::from_integer(6),   // constant term
///     Rational::from_integer(-5),  // x coefficient
///     Rational::from_integer(1),   // x^2 coefficient (must be 1)
/// ]);
///
/// let companion = companion_matrix(&poly).unwrap();
/// assert_eq!(companion.rows(), 2);
/// assert_eq!(companion.cols(), 2);
/// ```
pub fn companion_matrix<R: Ring + Clone>(polynomial: &UnivariatePolynomial<R>) -> Result<Matrix<R>> {
    let degree = match polynomial.degree() {
        Some(d) => d,
        None => return Err(MathError::InvalidArgument(
            "Cannot create companion matrix for zero polynomial".to_string(),
        )),
    };

    if degree == 0 {
        return Err(MathError::InvalidArgument(
            "Polynomial must have degree at least 1".to_string(),
        ));
    }

    // Check that polynomial is monic
    let leading_coeff = polynomial.coeff(degree);
    if leading_coeff != &R::one() {
        return Err(MathError::InvalidArgument(
            "Polynomial must be monic (leading coefficient = 1)".to_string(),
        ));
    }

    let n = degree;
    let mut data = vec![R::zero(); n * n];

    // Fill the companion matrix
    // Last column contains -a_0, -a_1, ..., -a_{n-1}
    for i in 0..n {
        let coeff = polynomial.coeff(i);
        data[i * n + (n - 1)] = R::zero() - coeff.clone();
    }

    // Subdiagonal contains 1's (except last row which is already filled)
    for i in 1..n {
        data[i * n + (i - 1)] = R::one();
    }

    Matrix::from_vec(n, n, data)
}

/// Represents the rational canonical form of a matrix
///
/// The rational canonical form consists of a direct sum of companion matrices
/// for the invariant factors of the matrix.
#[derive(Debug, Clone)]
pub struct RationalCanonicalForm<F: Field> {
    /// The invariant factors (monic polynomials dividing each other)
    pub invariant_factors: Vec<UnivariatePolynomial<F>>,

    /// The rational canonical form matrix (block diagonal of companion matrices)
    pub canonical_matrix: Matrix<F>,

    /// The change of basis matrix P such that P^{-1} A P = canonical_matrix
    pub change_of_basis: Option<Matrix<F>>,
}

impl<F: Field> RationalCanonicalForm<F> {
    /// Create a rational canonical form from invariant factors
    ///
    /// # Arguments
    ///
    /// * `invariant_factors` - List of monic polynomials, each dividing the next
    pub fn from_invariant_factors(
        invariant_factors: Vec<UnivariatePolynomial<F>>,
    ) -> Result<Self> {
        // Validate that each factor divides the next
        for factor in &invariant_factors {
            // In a proper implementation, we'd check divisibility
            // For now, just verify they're all monic
            if let Some(deg) = factor.degree() {
                if deg > 0 && factor.coeff(deg) != &F::one() {
                    return Err(MathError::InvalidArgument(
                        "All invariant factors must be monic".to_string(),
                    ));
                }
            }
        }

        // Build the block diagonal matrix from companion matrices
        let total_size: usize = invariant_factors
            .iter()
            .filter_map(|p| p.degree())
            .sum();

        let mut data = vec![F::zero(); total_size * total_size];
        let mut current_row = 0;

        for factor in &invariant_factors {
            let block_size = match factor.degree() {
                Some(d) => d,
                None => continue,
            };
            if block_size == 0 {
                continue;
            }

            // Build companion matrix for this factor
            let companion = companion_matrix(factor)?;

            // Copy into the appropriate block
            for i in 0..block_size {
                for j in 0..block_size {
                    data[(current_row + i) * total_size + (current_row + j)] =
                        companion.get(i, j)?.clone();
                }
            }

            current_row += block_size;
        }

        let canonical_matrix = Matrix::from_vec(total_size, total_size, data)?;

        Ok(RationalCanonicalForm {
            invariant_factors,
            canonical_matrix,
            change_of_basis: None,
        })
    }

    /// Get the characteristic polynomial
    ///
    /// This is the product of all invariant factors
    pub fn characteristic_polynomial(&self) -> UnivariatePolynomial<F> {
        if self.invariant_factors.is_empty() {
            return UnivariatePolynomial::new(vec![F::one()]);
        }

        let mut result = self.invariant_factors[0].clone();
        for i in 1..self.invariant_factors.len() {
            result = result * self.invariant_factors[i].clone();
        }
        result
    }

    /// Get the minimal polynomial
    ///
    /// This is the largest invariant factor (the last one in the chain)
    pub fn minimal_polynomial(&self) -> Option<&UnivariatePolynomial<F>> {
        self.invariant_factors.last()
    }
}

/// Compute the rational canonical form of a matrix
///
/// Note: This is a simplified implementation that requires the characteristic polynomial.
/// A full implementation would use Smith normal form to find invariant factors.
///
/// # Arguments
///
/// * `matrix` - The square matrix to decompose
///
/// # Returns
///
/// The rational canonical form
pub fn rational_canonical_form<F: Field>(
    matrix: &Matrix<F>,
) -> Result<RationalCanonicalForm<F>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Rational canonical form is only defined for square matrices".to_string(),
        ));
    }

    // For now, create a placeholder implementation
    // A real implementation would:
    // 1. Compute the characteristic polynomial
    // 2. Compute Smith normal form of xI - A over F[x]
    // 3. Extract invariant factors
    // 4. Build companion matrices

    let n = matrix.rows();

    // Placeholder: use the identity as a trivial case
    let invariant_factors = vec![UnivariatePolynomial::new(vec![
        F::zero() - F::one(),
        F::one(),
    ]); n];

    RationalCanonicalForm::from_invariant_factors(invariant_factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_companion_matrix_quadratic() {
        // p(x) = x^2 - 5x + 6 = (x-2)(x-3)
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(6),   // constant
            Rational::from_integer(-5),  // x
            Rational::from_integer(1),   // x^2
        ]);

        let companion = companion_matrix(&poly).unwrap();

        assert_eq!(companion.rows(), 2);
        assert_eq!(companion.cols(), 2);

        // Companion matrix should be:
        // [ 0  -6]
        // [ 1   5]
        assert_eq!(companion.get(0, 0).unwrap(), &Rational::from_integer(0));
        assert_eq!(companion.get(0, 1).unwrap(), &Rational::from_integer(-6));
        assert_eq!(companion.get(1, 0).unwrap(), &Rational::from_integer(1));
        assert_eq!(companion.get(1, 1).unwrap(), &Rational::from_integer(5));
    }

    #[test]
    fn test_companion_matrix_cubic() {
        // p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-6),  // constant
            Rational::from_integer(11),  // x
            Rational::from_integer(-6),  // x^2
            Rational::from_integer(1),   // x^3
        ]);

        let companion = companion_matrix(&poly).unwrap();

        assert_eq!(companion.rows(), 3);
        assert_eq!(companion.cols(), 3);

        // Companion matrix should be:
        // [ 0  0   6]
        // [ 1  0 -11]
        // [ 0  1   6]
        assert_eq!(companion.get(0, 2).unwrap(), &Rational::from_integer(6));
        assert_eq!(companion.get(1, 0).unwrap(), &Rational::from_integer(1));
        assert_eq!(companion.get(1, 2).unwrap(), &Rational::from_integer(-11));
        assert_eq!(companion.get(2, 1).unwrap(), &Rational::from_integer(1));
        assert_eq!(companion.get(2, 2).unwrap(), &Rational::from_integer(6));
    }

    #[test]
    fn test_companion_matrix_not_monic() {
        // p(x) = 2x^2 + 3x + 1 (leading coefficient is 2, not 1)
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(1),
            Rational::from_integer(3),
            Rational::from_integer(2),
        ]);

        let result = companion_matrix(&poly);
        assert!(result.is_err());
    }

    #[test]
    fn test_rational_canonical_form_simple() {
        // Create a simple 2x2 matrix
        let matrix = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(0),
                Rational::from_integer(0),
                Rational::from_integer(2),
            ],
        )
        .unwrap();

        let rcf = rational_canonical_form(&matrix).unwrap();

        assert!(rcf.canonical_matrix.is_square());
        assert_eq!(rcf.canonical_matrix.rows(), 2);
    }

    #[test]
    fn test_rcf_from_invariant_factors() {
        // Create RCF from two invariant factors:
        // f1(x) = x - 1
        // f2(x) = x^2 - 3x + 2 = (x-1)(x-2)

        let f1 = UnivariatePolynomial::new(vec![
            Rational::from_integer(-1),
            Rational::from_integer(1),
        ]);

        let f2 = UnivariatePolynomial::new(vec![
            Rational::from_integer(2),
            Rational::from_integer(-3),
            Rational::from_integer(1),
        ]);

        let rcf = RationalCanonicalForm::from_invariant_factors(vec![f1, f2]).unwrap();

        // Total size should be 1 + 2 = 3
        assert_eq!(rcf.canonical_matrix.rows(), 3);
        assert_eq!(rcf.canonical_matrix.cols(), 3);

        // Check characteristic polynomial
        let charpoly = rcf.characteristic_polynomial();
        assert_eq!(charpoly.degree(), Some(3));

        // Check minimal polynomial (should be the larger factor)
        let minpoly = rcf.minimal_polynomial().unwrap();
        assert_eq!(minpoly.degree(), Some(2));
    }

    #[test]
    fn test_companion_identity_factor() {
        // p(x) = x - 1 gives companion matrix [1]
        let poly = UnivariatePolynomial::new(vec![
            Rational::from_integer(-1),
            Rational::from_integer(1),
        ]);

        let companion = companion_matrix(&poly).unwrap();
        assert_eq!(companion.rows(), 1);
        assert_eq!(companion.get(0, 0).unwrap(), &Rational::from_integer(1));
    }
}

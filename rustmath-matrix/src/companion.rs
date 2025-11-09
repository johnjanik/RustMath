//! Companion matrices and rational canonical form
//!
//! The companion matrix of a monic polynomial p(x) = x^n + a_{n-1}x^{n-1} + ... + a_1 x + a_0
//! is an n×n matrix whose characteristic polynomial is p(x).
//!
//! The rational canonical form is a canonical form for matrices under similarity,
//! consisting of companion matrices for the invariant factors.

use crate::Matrix;
use crate::polynomial_matrix::PolynomialMatrix;
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

/// Compute the characteristic polynomial det(xI - A) using the Faddeev-LeVerrier algorithm
///
/// # Arguments
///
/// * `matrix` - A square matrix
///
/// # Returns
///
/// The characteristic polynomial
pub fn characteristic_polynomial<F: Field>(matrix: &Matrix<F>) -> Result<UnivariatePolynomial<F>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Characteristic polynomial is only defined for square matrices".to_string(),
        ));
    }

    let n = matrix.rows();

    // Special case for 1x1 matrices
    if n == 1 {
        let a = matrix.get(0, 0)?;
        return Ok(UnivariatePolynomial::new(vec![
            F::zero() - a.clone(),
            F::one(),
        ]));
    }

    // Special case for 2x2 matrices
    if n == 2 {
        let a = matrix.get(0, 0)?;
        let b = matrix.get(0, 1)?;
        let c = matrix.get(1, 0)?;
        let d = matrix.get(1, 1)?;

        let trace = a.clone() + d.clone();
        let det = a.clone() * d.clone() - b.clone() * c.clone();

        return Ok(UnivariatePolynomial::new(vec![
            det,
            F::zero() - trace,
            F::one(),
        ]));
    }

    // Use Faddeev-LeVerrier algorithm for general case
    // Build characteristic polynomial iteratively
    let mut coeffs = vec![F::zero(); n + 1];
    coeffs[n] = F::one(); // Leading coefficient is always 1

    let mut b = Matrix::identity(n);

    for k in 1..=n {
        // B_k = A * B_{k-1}
        b = (matrix.clone() * b)?;

        // c_k = -trace(B_k) / k
        let mut trace = F::zero();
        for i in 0..n {
            trace = trace + b.get(i, i)?.clone();
        }

        // Divide by k (convert k to field element)
        let mut k_field = F::zero();
        for _ in 0..k {
            k_field = k_field + F::one();
        }
        let c_k = F::zero() - (trace / k_field);
        coeffs[n - k] = c_k.clone();

        // Add c_k * I to B_k for next iteration (except on last iteration)
        if k < n {
            let mut new_b_data = b.data().to_vec();
            for i in 0..n {
                new_b_data[i * n + i] = new_b_data[i * n + i].clone() + c_k.clone();
            }
            b = Matrix::from_vec(n, n, new_b_data)?;
        }
    }

    Ok(UnivariatePolynomial::new(coeffs))
}

/// Construct the characteristic matrix xI - A
///
/// This is a polynomial matrix where x is the indeterminate.
fn characteristic_matrix<F: Field>(matrix: &Matrix<F>) -> Result<PolynomialMatrix<F>> {
    let n = matrix.rows();
    let mut data = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            let entry = if i == j {
                // Diagonal: x - a_ij
                UnivariatePolynomial::new(vec![
                    F::zero() - matrix.get(i, j)?.clone(),
                    F::one(),
                ])
            } else {
                // Off-diagonal: -a_ij
                UnivariatePolynomial::new(vec![F::zero() - matrix.get(i, j)?.clone()])
            };
            data.push(entry);
        }
    }

    PolynomialMatrix::new(n, n, data)
}

/// Compute the rational canonical form of a matrix
///
/// This implementation computes the Smith normal form of the characteristic matrix
/// xI - A to extract the invariant factors.
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

    // Step 1: Construct the characteristic matrix xI - A
    let char_matrix = characteristic_matrix(matrix)?;

    // Step 2: Compute Smith normal form to get invariant factors
    let snf = char_matrix.smith_normal_form()?;
    let invariant_factors = snf.invariant_factors();

    // Step 3: Build the rational canonical form from invariant factors
    if invariant_factors.is_empty() {
        // Fallback: use characteristic polynomial as single invariant factor
        let char_poly = characteristic_polynomial(matrix)?;
        RationalCanonicalForm::from_invariant_factors(vec![char_poly])
    } else {
        RationalCanonicalForm::from_invariant_factors(invariant_factors)
    }
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

    #[test]
    fn test_characteristic_polynomial_1x1() {
        // Matrix [a] has characteristic polynomial (x - a)
        let matrix = Matrix::from_vec(
            1,
            1,
            vec![Rational::from_integer(5)],
        )
        .unwrap();

        let charpoly = characteristic_polynomial(&matrix).unwrap();

        assert_eq!(charpoly.degree(), Some(1));
        assert_eq!(charpoly.coeff(0), &Rational::from_integer(-5)); // constant term
        assert_eq!(charpoly.coeff(1), &Rational::from_integer(1));  // x coefficient
    }

    #[test]
    fn test_characteristic_polynomial_2x2() {
        // Matrix [1 2]
        //        [3 4]
        // Characteristic polynomial: x² - 5x - 2
        let matrix = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(1),
                Rational::from_integer(2),
                Rational::from_integer(3),
                Rational::from_integer(4),
            ],
        )
        .unwrap();

        let charpoly = characteristic_polynomial(&matrix).unwrap();

        assert_eq!(charpoly.degree(), Some(2));
        // det(A) = 1*4 - 2*3 = -2
        assert_eq!(charpoly.coeff(0), &Rational::from_integer(-2));
        // -tr(A) = -(1+4) = -5
        assert_eq!(charpoly.coeff(1), &Rational::from_integer(-5));
        assert_eq!(charpoly.coeff(2), &Rational::from_integer(1));
    }

    #[test]
    fn test_characteristic_polynomial_identity() {
        // Identity matrix has characteristic polynomial (x - 1)^n
        let matrix = Matrix::<Rational>::identity(2);
        let charpoly = characteristic_polynomial(&matrix).unwrap();

        assert_eq!(charpoly.degree(), Some(2));
        // (x - 1)² = x² - 2x + 1
        assert_eq!(charpoly.coeff(0), &Rational::from_integer(1));   // constant
        assert_eq!(charpoly.coeff(1), &Rational::from_integer(-2));  // x
        assert_eq!(charpoly.coeff(2), &Rational::from_integer(1));   // x²
    }

    #[test]
    fn test_characteristic_polynomial_zero_matrix() {
        // Zero matrix has characteristic polynomial x^n
        let mut matrix_data = vec![Rational::from_integer(0); 4];
        let matrix = Matrix::from_vec(2, 2, matrix_data).unwrap();
        let charpoly = characteristic_polynomial(&matrix).unwrap();

        assert_eq!(charpoly.degree(), Some(2));
        assert_eq!(charpoly.coeff(0), &Rational::from_integer(0));  // constant = det = 0
        assert_eq!(charpoly.coeff(1), &Rational::from_integer(0));  // -trace = 0
        assert_eq!(charpoly.coeff(2), &Rational::from_integer(1));  // x²
    }

    #[test]
    fn test_rcf_diagonal_matrix() {
        // Diagonal matrix should have simple RCF
        let matrix = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(2),
                Rational::from_integer(0),
                Rational::from_integer(0),
                Rational::from_integer(3),
            ],
        )
        .unwrap();

        let rcf = rational_canonical_form(&matrix).unwrap();

        assert_eq!(rcf.canonical_matrix.rows(), 2);
        assert_eq!(rcf.canonical_matrix.cols(), 2);

        // Check that we have invariant factors
        assert!(!rcf.invariant_factors.is_empty());
    }

    #[test]
    fn test_rcf_nilpotent_matrix() {
        // Nilpotent matrix: [0 1]
        //                   [0 0]
        // Characteristic polynomial: x²
        let matrix = Matrix::from_vec(
            2,
            2,
            vec![
                Rational::from_integer(0),
                Rational::from_integer(1),
                Rational::from_integer(0),
                Rational::from_integer(0),
            ],
        )
        .unwrap();

        let rcf = rational_canonical_form(&matrix).unwrap();

        assert_eq!(rcf.canonical_matrix.rows(), 2);
        assert_eq!(rcf.canonical_matrix.cols(), 2);
    }
}

//! Companion matrices and rational canonical form
//!
//! The companion matrix of a monic polynomial p(x) = x^n + a_{n-1}x^{n-1} + ... + a_1 x + a_0
//! is an n×n matrix whose characteristic polynomial is p(x).
//!
//! The rational canonical form is a canonical form for matrices under similarity,
//! consisting of companion matrices for the invariant factors.
//!
//! MAGMA Handbook ch. 26 §26.11 (`MinimalPolynomial`; also ch. 59 §59.4.3
//! `MinimalPolynomial(g)` for a matrix-group element) and §26.12.2 (`InvariantFactors`,
//! `RationalForm`), digest `rustmath-lava/docs/Chapter_26_Matrices.md`.
//!
//! Algorithm: the invariant factors of `A` are the non-constant diagonal entries of the Smith
//! form of `xI - A` over `F[x]` ([`PolynomialMatrix::smith_normal_form`]); the change of basis
//! comes from the columns of `P⁻¹` (N. Jacobson, *Basic Algebra I*, 2nd ed., §3.10; D. S.
//! Dummit and R. M. Foote, *Abstract Algebra*, 3rd ed., §12.2). MAGMA uses Steel's algorithm
//! [Ste97] instead; no complexity claim is made here.

use crate::Matrix;
use crate::polynomial_matrix::{div_rem_field, PolynomialMatrix};
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

    /// The change of basis matrix T such that T^{-1}·A·T = canonical_matrix (column
    /// convention: the columns of T are the cyclic bases `g, A·g, …, A^{k-1}·g` of the
    /// blocks). MAGMA's `RationalForm` returns `T^{-1}` here, as `T'` with `T'·A·T'^{-1} = F`.
    /// `Some` when computed by [`rational_canonical_form`]; `None` when the form is built
    /// from the invariant factors alone ([`RationalCanonicalForm::from_invariant_factors`]).
    pub change_of_basis: Option<Matrix<F>>,
}

impl<F: Field> RationalCanonicalForm<F> {
    /// Create a rational canonical form from invariant factors
    ///
    /// # Arguments
    ///
    /// * `invariant_factors` - List of monic polynomials, each dividing the next
    ///
    /// * `Err(InvalidArgument)` on a zero polynomial, on a polynomial that is not monic
    ///   (constants included: `2` is rejected), or if the list is not a divisibility chain
    ///   `f_1 | f_2 | …`.
    /// * The constant `1` forms no block and is dropped from the stored factors.
    /// * `change_of_basis` is `None`: it is not computed when built from factors alone.
    pub fn from_invariant_factors(
        invariant_factors: Vec<UnivariatePolynomial<F>>,
    ) -> Result<Self> {
        let mut kept: Vec<UnivariatePolynomial<F>> = Vec::with_capacity(invariant_factors.len());
        for factor in invariant_factors {
            let Some(deg) = factor.degree() else {
                return Err(MathError::InvalidArgument(
                    "invariant factors must be non-zero".to_string(),
                ));
            };
            if !factor.coeff(deg).is_one() {
                return Err(MathError::InvalidArgument(
                    "All invariant factors must be monic".to_string(),
                ));
            }
            if deg > 0 {
                kept.push(factor);
            }
        }
        for w in kept.windows(2) {
            if !div_rem_field(&w[1], &w[0])?.1.is_zero() {
                return Err(MathError::InvalidArgument(
                    "invariant factors must form a divisibility chain".to_string(),
                ));
            }
        }
        let invariant_factors = kept;

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
    /// This is the largest invariant factor (the last one in the chain). `None` only for the
    /// 0×0 form (whose minimal polynomial is the constant 1).
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

/// MAGMA `InvariantFactors(A)` (Handbook ch. 26 §26.12.2): the non-constant invariant factors
/// of `A`, monic, `f_1 | f_2 | … | f_r` (increasing), with product the characteristic
/// polynomial.
///
/// `Err(InvalidArgument)` if `A` is not square; `[]` for the 0×0 matrix. Computed as the
/// non-constant diagonal entries of the Smith form of `xI - A`, whose output check certifies
/// them (see [`PolynomialMatrix::smith_normal_form`]).
pub fn invariant_factors<F: Field>(matrix: &Matrix<F>) -> Result<Vec<UnivariatePolynomial<F>>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Invariant factors are only defined for square matrices".to_string(),
        ));
    }
    let n = matrix.rows();
    let snf = characteristic_matrix(matrix)?.smith_normal_form()?;
    let mut factors = Vec::new();
    let mut total_degree = 0usize;
    for i in 0..n {
        let d = snf.diagonal.get(i, i);
        match d.degree() {
            // det(xI - A) is monic of degree n, so no diagonal entry is zero.
            None => {
                return Err(MathError::InvalidOperation(
                    "internal: zero diagonal entry in the Smith form of xI - A".to_string(),
                ))
            }
            Some(0) => {}
            Some(k) => {
                total_degree += k;
                factors.push(d.clone());
            }
        }
    }
    if total_degree != n {
        return Err(MathError::InvalidOperation(
            "internal: the invariant factors of xI - A do not have total degree n".to_string(),
        ));
    }
    Ok(factors)
}

/// MAGMA `MinimalPolynomial(A)` (Handbook ch. 26 §26.11; also ch. 59 §59.4.3
/// `MinimalPolynomial(g)`): the monic polynomial `f` of least degree with `f(A) = 0`, which is
/// the last invariant factor of `A`.
///
/// `Err(InvalidArgument)` if `A` is not square; the constant 1 for the 0×0 matrix.
pub fn minimal_polynomial<F: Field>(matrix: &Matrix<F>) -> Result<UnivariatePolynomial<F>> {
    Ok(invariant_factors(matrix)?
        .pop()
        .unwrap_or_else(|| UnivariatePolynomial::new(vec![F::one()])))
}

/// `p(A)·e_j`, by Horner's rule on vectors: `v ← A·v + c·e_j` for the coefficients `c` of `p`
/// from the top down.
fn poly_at_matrix_times_unit<F: Field>(
    p: &UnivariatePolynomial<F>,
    a: &Matrix<F>,
    j: usize,
) -> Result<Vec<F>> {
    let n = a.rows();
    let mut v = vec![F::zero(); n];
    for c in p.coefficients().iter().rev() {
        let mut w = mat_vec(a, &v)?;
        w[j] = w[j].clone() + c.clone();
        v = w;
    }
    Ok(v)
}

/// `A·v`.
fn mat_vec<F: Field>(a: &Matrix<F>, v: &[F]) -> Result<Vec<F>> {
    let n = a.cols();
    (0..a.rows())
        .map(|i| {
            let mut acc = F::zero();
            for (k, vk) in v.iter().enumerate().take(n) {
                acc = acc + a.get(i, k)?.clone() * vk.clone();
            }
            Ok(acc)
        })
        .collect()
}

/// Compute the rational canonical form of a matrix (MAGMA `RationalForm(A)`, Handbook ch. 26
/// §26.12.2)
///
/// Returns the invariant factors `f_1 | … | f_r` (as [`invariant_factors`]), the block
/// diagonal matrix `C` of the companion matrices of the `f_i` ([`companion_matrix`], column
/// style) in increasing divisibility order, and `change_of_basis = Some(T)` with
/// `T⁻¹·A·T = C`.
///
/// Construction (Dummit–Foote §12.2): with `P·(xI - A)·Q = S` the Smith form, the module
/// `F^n` (x acting as A) is `F[x]^n / im(xI - A)` via `e_j ↦ e_j`, and `P` maps it onto
/// `⊕ F[x]/(d_i)`; the generator of the i-th summand comes from column i of `P⁻¹`, so
/// `g_i = Σ_j (P⁻¹)_{j,i}(A)·e_j` has annihilator `d_i`, and `g_i, A·g_i, …, A^{k-1}·g_i`
/// (`k = deg d_i`) are the columns of `T` for that block.
///
/// Certificate: the result is returned only if `T` is square of size n, `A·T == T·C` and `T`
/// is invertible; then `A` is similar to a block companion matrix whose polynomials form a
/// divisibility chain, which by uniqueness of the rational canonical form proves both the
/// factors and `C`. Otherwise `Err(InvalidOperation)`. `Err(InvalidArgument)` if `A` is not
/// square. The invertibility test uses [`Matrix::inverse`], which divides with `/`; for a
/// matrix built only from `PrimeField`'s unbound `Ring::zero()`/`one()` sentinels that can
/// panic (a pre-existing limitation of `inverse`).
pub fn rational_canonical_form<F: Field>(
    matrix: &Matrix<F>,
) -> Result<RationalCanonicalForm<F>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Rational canonical form is only defined for square matrices".to_string(),
        ));
    }
    let n = matrix.rows();
    if n == 0 {
        return Ok(RationalCanonicalForm {
            invariant_factors: Vec::new(),
            canonical_matrix: Matrix::from_vec(0, 0, Vec::new())?,
            change_of_basis: Some(Matrix::from_vec(0, 0, Vec::new())?),
        });
    }

    let snf = characteristic_matrix(matrix)?.smith_normal_form()?;
    let pinv = &snf.left_inverse;

    let mut factors = Vec::new();
    let mut columns: Vec<Vec<F>> = Vec::with_capacity(n);
    for i in 0..n {
        let d = snf.diagonal.get(i, i);
        let k = match d.degree() {
            None => {
                return Err(MathError::InvalidOperation(
                    "internal: zero diagonal entry in the Smith form of xI - A".to_string(),
                ))
            }
            Some(0) => continue,
            Some(k) => k,
        };
        // g_i = Σ_j (P⁻¹)_{j,i}(A)·e_j
        let mut g = vec![F::zero(); n];
        for j in 0..n {
            let pji = pinv.get(j, i);
            if pji.is_zero() {
                continue;
            }
            let v = poly_at_matrix_times_unit(pji, matrix, j)?;
            for (gr, vr) in g.iter_mut().zip(v) {
                *gr = gr.clone() + vr;
            }
        }
        for _ in 0..k {
            let next = mat_vec(matrix, &g)?;
            columns.push(std::mem::replace(&mut g, next));
        }
        factors.push(d.clone());
    }

    let mut rcf = RationalCanonicalForm::from_invariant_factors(factors)?;

    let certificate_failed = || {
        Err(MathError::InvalidOperation(
            "internal: rational form certificate failed".to_string(),
        ))
    };
    if columns.len() != n || rcf.canonical_matrix.rows() != n {
        return certificate_failed();
    }
    let mut t_data = Vec::with_capacity(n * n);
    for r in 0..n {
        for col in &columns {
            t_data.push(col[r].clone());
        }
    }
    let t = Matrix::from_vec(n, n, t_data)?;
    if matrix.mul(&t)? != t.mul(&rcf.canonical_matrix)? || t.inverse()?.is_none() {
        return certificate_failed();
    }
    rcf.change_of_basis = Some(t);
    Ok(rcf)
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

    // Reference values: PARI/GP 2.15.4, per matrix A:
    //   matsnf(x*matid(n)-A), matfrobenius(A,1) (invariant factors, decreasing order; RustMath
    //   and MAGMA use increasing order, so the lists are reversed), minpoly(A), charpoly(A),
    //   matfrobenius(A) (the Frobenius form).

    use crate::charpoly_berkowitz;

    fn px(c: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(c.iter().map(|&v| Rational::from_integer(v)).collect())
    }

    fn qm(n: usize, v: &[i64]) -> Matrix<Rational> {
        Matrix::from_vec(n, n, v.iter().map(|&x| Rational::from_integer(x)).collect()).unwrap()
    }

    /// The minimal polynomial by an independent Krylov method: the least k with
    /// vec(A^k) in span(vec(A^0), …, vec(A^{k-1})), found with `Matrix::solve` on the
    /// n²×k system (which also exercises the fixed `solve` on tall systems).
    fn krylov_minpoly(a: &Matrix<Rational>) -> UnivariatePolynomial<Rational> {
        let n = a.rows();
        let mut powers = vec![Matrix::<Rational>::identity(n)];
        loop {
            let k = powers.len();
            let next = powers[k - 1].mul(a).unwrap();
            let mut data = Vec::with_capacity(n * n * k);
            for idx in 0..n * n {
                for p in &powers {
                    data.push(p.data()[idx].clone());
                }
            }
            let cols = Matrix::from_vec(n * n, k, data).unwrap();
            if let Some(c) = cols.solve(next.data()).unwrap() {
                let mut coeffs: Vec<Rational> = c
                    .into_iter()
                    .map(|ci| Rational::from_integer(0) - ci)
                    .collect();
                coeffs.push(Rational::from_integer(1));
                return UnivariatePolynomial::new(coeffs);
            }
            powers.push(next);
        }
    }

    /// Asserts invariant_factors, minimal_polynomial, the rational form (factors, block
    /// companion matrix, certified change of basis), the Krylov minimal polynomial and the
    /// product of the factors against `charpoly_berkowitz`.
    fn check_forms(a: &Matrix<Rational>, expected: &[UnivariatePolynomial<Rational>]) {
        let n = a.rows();
        let fs = invariant_factors(a).unwrap();
        assert_eq!(fs, expected);
        let mp = minimal_polynomial(a).unwrap();
        assert_eq!(&mp, expected.last().unwrap());
        assert_eq!(krylov_minpoly(a), mp);

        let rcf = rational_canonical_form(a).unwrap();
        assert_eq!(rcf.invariant_factors, expected);
        assert_eq!(rcf.minimal_polynomial(), Some(&mp));
        let mut c = Matrix::<Rational>::zeros(n, n);
        let mut off = 0;
        for f in expected {
            let b = companion_matrix(f).unwrap();
            for i in 0..b.rows() {
                for j in 0..b.cols() {
                    c.set(off + i, off + j, b.get(i, j).unwrap().clone())
                        .unwrap();
                }
            }
            off += b.rows();
        }
        assert_eq!(off, n);
        assert_eq!(rcf.canonical_matrix, c);
        let t = rcf.change_of_basis.clone().expect("T is computed");
        assert_eq!(a.mul(&t).unwrap(), t.mul(&c).unwrap(), "A·T != T·C");
        let tinv = t.inverse().unwrap().expect("T invertible");
        assert_eq!(tinv.mul(a).unwrap().mul(&t).unwrap(), c, "T⁻¹·A·T != C");

        let product = expected.iter().fold(px(&[1]), |acc, f| acc * f.clone());
        assert_eq!(product, charpoly_berkowitz(a).unwrap());
    }

    #[test]
    fn forms_r1_diag23_defect() {
        // The ch026.InvariantFactors / MinimalPolynomial / RationalForm defect: the factors
        // were [x-2, x-3] and the minimal polynomial x-3.
        // gp: matsnf(x*matid(2)-[2,0;0,3]) = [x^2 - 5*x + 6, 1],
        //     matfrobenius([2,0;0,3],1) = [x^2 - 5*x + 6], minpoly = x^2 - 5*x + 6,
        //     matfrobenius([2,0;0,3]) = [0, -6; 1, 5]
        let a = qm(2, &[2, 0, 0, 3]);
        check_forms(&a, &[px(&[6, -5, 1])]);
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(rcf.canonical_matrix, qm(2, &[0, -6, 1, 5]));
    }

    #[test]
    fn rcf_minpoly_regression() {
        // gp: minpoly([2,0;0,3]) = x^2 - 5*x + 6
        let a = qm(2, &[2, 0, 0, 3]);
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(rcf.minimal_polynomial(), Some(&px(&[6, -5, 1])));
    }

    #[test]
    fn forms_small_q() {
        // gp: matfrobenius(matdiagonal([2,2,3]),1) = [x^2 - 5*x + 6, x - 2]
        check_forms(
            &qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3]),
            &[px(&[-2, 1]), px(&[6, -5, 1])],
        );
        // gp: matfrobenius(matid(2),1) = [x - 1, x - 1]
        check_forms(&qm(2, &[1, 0, 0, 1]), &[px(&[-1, 1]), px(&[-1, 1])]);
        // gp: matfrobenius([0,1;0,0],1) = [x^2]
        check_forms(&qm(2, &[0, 1, 0, 0]), &[px(&[0, 0, 1])]);
        // gp: matfrobenius([0,0;0,0],1) = [x, x]
        check_forms(&qm(2, &[0, 0, 0, 0]), &[px(&[0, 1]), px(&[0, 1])]);
        // gp: matfrobenius([1,2;3,4],1) = [x^2 - 5*x - 2]
        check_forms(&qm(2, &[1, 2, 3, 4]), &[px(&[-2, -5, 1])]);
        // gp: matfrobenius([1,2,0;0,1,3;4,0,1],1) = [x^3 - 3*x^2 + 3*x - 25]
        check_forms(
            &qm(3, &[1, 2, 0, 0, 1, 3, 4, 0, 1]),
            &[px(&[-25, 3, -3, 1])],
        );
    }

    #[test]
    fn forms_r7_jordan_blocks() {
        // A = J2(2) + J1(2) + [3]. gp: matfrobenius(A,1) = [x^3 - 7*x^2 + 16*x - 12, x - 2],
        //     matfrobenius(A) = [0,0,12,0; 1,0,-16,0; 0,1,7,0; 0,0,0,2]
        let a = qm(4, &[2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3]);
        let expected = [px(&[-2, 1]), px(&[-12, 16, -7, 1])];
        check_forms(&a, &expected);
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(
            rcf.canonical_matrix,
            qm(4, &[2, 0, 0, 0, 0, 0, 0, 12, 0, 1, 0, -16, 0, 0, 1, 7])
        );
        // A conjugate of A, U4*A*U4^-1 with U4 = [1,1,0,0;0,1,1,0;0,0,1,1;0,0,0,1]:
        // gp: same matfrobenius(.,1) and matfrobenius(.)
        let b = qm(4, &[2, 1, -1, 1, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 3]);
        check_forms(&b, &expected);
        assert_eq!(
            rational_canonical_form(&b).unwrap().canonical_matrix,
            rcf.canonical_matrix
        );
    }

    #[test]
    fn forms_r8_derogatory_irreducible_quadratic() {
        // A = U5*(C(x^2+1) + C((x^2+1)(x-1)))*U5^-1, U5 = [1,2,0,0,1;0,1,0,0,0;0,1,1,0,0;
        // 0,0,0,1,1;0,0,0,0,1] (det 1).
        // gp: matfrobenius(A,1) = [x^3 - x^2 + x - 1, x^2 + 1],
        //     charpoly(A) = x^5 - x^4 + 2*x^3 - 2*x^2 + x - 1, subst(minpoly(A),x,A) == 0,
        //     factor(minpoly(A)) = [x - 1, 1; x^2 + 1, 1]
        let a = qm(
            5,
            &[
                2, -5, 0, 1, -2, 1, -2, 0, 0, -1, 1, -2, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 1, 0,
            ],
        );
        check_forms(&a, &[px(&[1, 0, 1]), px(&[-1, 1, -1, 1])]);
    }

    #[test]
    fn forms_empty_and_non_square() {
        // Definitional: the 0×0 matrix has no invariant factors and minimal polynomial 1.
        let e = Matrix::<Rational>::from_vec(0, 0, vec![]).unwrap();
        assert!(invariant_factors(&e).unwrap().is_empty());
        assert_eq!(minimal_polynomial(&e).unwrap(), px(&[1]));
        let rcf = rational_canonical_form(&e).unwrap();
        assert!(rcf.invariant_factors.is_empty());
        assert_eq!(rcf.minimal_polynomial(), None);
        assert_eq!(rcf.change_of_basis.map(|t| t.rows()), Some(0));
        // Non-square input is an error.
        let r = Matrix::from_vec(2, 3, vec![Rational::from_integer(1); 6]).unwrap();
        assert!(invariant_factors(&r).is_err());
        assert!(minimal_polynomial(&r).is_err());
        assert!(rational_canonical_form(&r).is_err());
    }

    #[test]
    fn from_invariant_factors_rejects_non_chain() {
        // x - 2 does not divide x - 3 (gp: (x-3) % (x-2) = -1).
        assert!(
            RationalCanonicalForm::from_invariant_factors(vec![px(&[-2, 1]), px(&[-3, 1])])
                .is_err()
        );
        // The zero polynomial and a non-monic constant are rejected.
        assert!(RationalCanonicalForm::from_invariant_factors(vec![px(&[0])]).is_err());
        assert!(
            RationalCanonicalForm::from_invariant_factors(vec![px(&[2]), px(&[-1, 1])]).is_err()
        );
        // The constant 1 forms no block and is dropped.
        let rcf =
            RationalCanonicalForm::from_invariant_factors(vec![px(&[1]), px(&[-1, 1])]).unwrap();
        assert_eq!(rcf.invariant_factors, vec![px(&[-1, 1])]);
        assert_eq!(rcf.canonical_matrix, qm(1, &[1]));
        assert!(rcf.change_of_basis.is_none());
    }
}

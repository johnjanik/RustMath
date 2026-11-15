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
//! Algorithm: spinning a fixed basis of start vectors under `A` (Krylov iteration; W.
//! Keller-Gehrig, Theoret. Comput. Sci. 36 (1985) 309–317 \[KG85\]; crate module `krylov`) gives a
//! presentation `F^n ≅ F[x]^r / im(R)` of the module `F^n` (x acting as `A`) by r generators and
//! an r×r upper-triangular relation matrix `R` over `F[x]`. Usually r = 1, and then the
//! presentation is the answer (`A` is cyclic). Otherwise the cyclic decomposition by maximal
//! vectors (crate module `frobenius`; K. Hoffman and R. Kunze, *Linear Algebra*, 2nd ed.,
//! §7.2) peels off one cyclic summand per invariant factor, the minimal polynomial first, with
//! the r generators as candidates; it needs no Smith form and no factorization. The change of
//! basis of the rational form comes from the clean generators of the summands. MAGMA uses
//! Steel's algorithm \[Ste97\] (A. Steel, J. Symbolic Comp. 24 (1997) 409–432) instead, which
//! factors the characteristic polynomial. Storjohann's O(n³) Frobenius form (ISSAC 1998; PhD
//! thesis, ETH Zürich 2000, ch. 9) is not implemented.
//!
//! Complexity: the main path costs `O((r + 1)·n³)` field operations (r ≤ n), so `O(n⁴)` in the
//! worst case and `O(n³)` for bounded r (module docs of `frobenius`). This bound holds for the
//! main path only; the fallbacks below have none. Over Q no bit-complexity bound is proved: the
//! echelon entries and coordinates are quotients of minors of the matrix of stored vectors, but
//! within one level the merges apply polynomials in `A` of growing degree (`frobenius`, "Over
//! Q"). Measured in debug (this crate's tests, idle machine):
//! * `CONJDIAG20_S7 = U·diag(1, …, 20)·U⁻¹` (U unimodular, `R` 2×2 with diagonal degrees 13 and
//!   7): [`invariant_factors`] 0.05 s and [`rational_canonical_form`] 0.14 s. The Smith form of
//!   `R` (the main path before) took 81 to 102 s in debug when the case was found, and takes
//!   3.2 s (rational form 5.7 s) on this tree.
//! * `W·B·W⁻¹` with r = n (module docs of `krylov`), n = 20: 0.05 s and 0.10 s; n = 30: 0.4 s
//!   and 0.8 s. With the Smith form of `R`, n = 20 took 31.5 s and 101 s on this tree (no answer
//!   in 300 s when the case was found), n = 14 took 3.5 s and 19 s (95 s for the factors then).
//! * Structured r = n inputs with merges (`W·Q·J·Q⁻¹·W⁻¹`, J a Jordan matrix with eigenvalues
//!   1, 2 and 5, Q sparse unit upper triangular), n = 30: up to 1.8 s and 2.6 s over 60 inputs.
//!   These are examples, not a ceiling: the time grows with the number of merges per level.
//! * Inputs with many small Krylov blocks and many invariant factors (scalar, or diagonal with
//!   few distinct values: r ≈ s ≈ n) cost `Θ(n⁴)` here, while the Smith form of their diagonal
//!   `R` costs about n³: for `2·I` with n = 100 over a word-size GF(7), 0.71 s against 0.05 s
//!   for [`invariant_factors`] and 0.75 s against 0.10 s for [`rational_canonical_form`]; over
//!   `PrimeField` GF(7), 2.5 s and 4.3 s (about 0.2 s and 2.4 s with the Smith form of `R`,
//!   measured before this change); for `3·I` with n = 40 over Q, 0.10 s against 0.01 s.
//!
//! Fallback: if the peeling returns `Err`, [`invariant_factors`], [`minimal_polynomial`] and
//! [`rational_canonical_form`] use the Smith form of the Krylov relation matrix `R` (the main
//! path before), and if that returns `Err` too, the classical presentation by `xI - A` on the
//! unit vectors (same certificates); the result or error of the last stage is returned. Over an
//! exact field (Q, bound GF(p) or GF(p^n)) the first stage does not fail. Two kinds of input
//! reach the fallbacks. First, matrices built only from the unbound `Ring::zero()`/`one()`
//! sentinels of `PrimeField` or of the GF(p^n) type `FiniteFieldElement`, computed in Z, where
//! only ±1 are invertible and the ±1 start vectors can create non-unit pivots (module docs of
//! `krylov`, "Exactness"). Second, a rounding `Field` impl (`Real`) whose checks fail.

use crate::charpoly_berkowitz;
use crate::frobenius::{peel, PeelDepth};
use crate::krylov::{
    field_constants, krylov_presentation, linearly_independent, mat_vec, poly_mat_vec,
};
use crate::polynomial_matrix::div_rem_field;
use crate::polynomial_matrix::PolynomialMatrix;
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

    /// The rational canonical form matrix `C`: block diagonal of companion matrices.
    ///
    /// Block shape: column-style companion matrices ([`companion_matrix`]: 1s on the
    /// subdiagonal, `-a_0, …, -a_{k-1}` in the last column), the block shape of PARI/GP
    /// `matfrobenius`. Block order: increasing divisibility order `f_1 | f_2 | … | f_r`, the
    /// order of MAGMA's `RationalForm` and `InvariantFactors` (Handbook ch. 26 §26.12.2: each
    /// block's polynomial divides the next). PARI lists the blocks in the opposite, decreasing
    /// order: for `A = J₂(2) ⊕ J₁(2) ⊕ [3]`, `matfrobenius(A)` is
    /// `[0,0,12,0; 1,0,-16,0; 0,1,7,0; 0,0,0,2]` and `C` is
    /// `[2,0,0,0; 0,0,0,12; 0,1,0,-16; 0,0,1,7]`. MAGMA's `CompanionMatrix` (ch. 23) has the
    /// 1s on the superdiagonal and the coefficients in the last row, so the form `F` returned
    /// by MAGMA's `RationalForm` is `Cᵀ` (same block order).
    pub canonical_matrix: Matrix<F>,

    /// The change of basis `T` with `T⁻¹·A·T = canonical_matrix` (column convention: the
    /// columns of `T` are the cyclic bases `g, A·g, …, A^{k-1}·g` of the blocks).
    ///
    /// Transforms are not unique; a matrix with another system's defining property can still
    /// differ from the one that system returns.
    /// * PARI/GP `matfrobenius(A, 2)` returns `[F_P, B]` with `A = B⁻¹·F_P·B`, where `F_P` has
    ///   the blocks of `C` in decreasing order. `T⁻¹` has this property for `C`
    ///   (`C = (T⁻¹)·A·(T⁻¹)⁻¹`); with `Π` a block permutation matrix with `Π·C·Π⁻¹ = F_P`
    ///   (it reverses the order of the blocks and keeps each block intact; equal invariant
    ///   factors give equal blocks, which any such `Π` may exchange), `Π·T⁻¹` has it for
    ///   `F_P`.
    /// * MAGMA's `RationalForm` returns `T_M` with `T_M·A·T_M⁻¹ = F` (row convention: the rows
    ///   of `T_M` for a block are `w, w·A, …, w·A^{k-1}`), where `F = Cᵀ`. `T⁻¹` has this
    ///   property iff `C` is symmetric, that is, iff every invariant factor is linear or of
    ///   the form `x² + b·x − 1` (block `[0, 1; 1, −b]`); for example `A = [1,1;1,0]`,
    ///   `C = [0,1;1,1]`. In general a matrix with MAGMA's property comes from the change of
    ///   basis `T₂` of `Aᵀ`: `T₂⁻¹·Aᵀ·T₂ = C` gives `T₂ᵀ·A·(T₂ᵀ)⁻¹ = Cᵀ = F`.
    ///
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
/// This is a polynomial matrix where x is the indeterminate. Its columns are relations of the
/// module `F^n` (x acting as `A`) on the unit vectors: column j maps to
/// `A·e_j - Σ_i a_ij·e_i = 0`. It is the presentation of the classical (fallback) path.
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

/// The non-constant diagonal entries of the Smith form of a relation matrix `R` presenting
/// `F^n` (x acting as an n×n `A`), without transforms: the invariant factors of `A`. A zero
/// diagonal entry or a total degree other than n is `Err(InvalidOperation)`.
fn factors_from_relations<F: Field>(
    relations: &PolynomialMatrix<F>,
    n: usize,
) -> Result<Vec<UnivariatePolynomial<F>>> {
    let mut factors = Vec::new();
    let mut total_degree = 0usize;
    for d in relations.smith_form_diagonal()? {
        match d.degree() {
            // det R is monic of degree n, so no diagonal entry is zero.
            None => {
                return Err(MathError::InvalidOperation(
                    "internal: zero diagonal entry in the Smith form of the relation matrix"
                        .to_string(),
                ))
            }
            Some(0) => {}
            Some(k) => {
                total_degree += k;
                factors.push(d);
            }
        }
    }
    if total_degree != n {
        return Err(MathError::InvalidOperation(
            "internal: the invariant factors do not have total degree n".to_string(),
        ));
    }
    Ok(factors)
}

/// The invariant factors of a square `A` by the Krylov path (fast path).
fn invariant_factors_krylov<F: Field>(matrix: &Matrix<F>) -> Result<Vec<UnivariatePolynomial<F>>> {
    factors_from_relations(&krylov_presentation(matrix)?.relations, matrix.rows())
}

/// The invariant factors of a square `A` from the Smith form of `xI - A` (classical path).
fn invariant_factors_classical<F: Field>(
    matrix: &Matrix<F>,
) -> Result<Vec<UnivariatePolynomial<F>>> {
    factors_from_relations(&characteristic_matrix(matrix)?, matrix.rows())
}

/// The invariant factors of a square `A` by the peeling of `frobenius.rs` (main path).
fn invariant_factors_peeled<F: Field>(matrix: &Matrix<F>) -> Result<Vec<UnivariatePolynomial<F>>> {
    Ok(peel(matrix, PeelDepth::Full)?.invariant_factors())
}

/// The minimal polynomial of a square, non-empty `A` by the first level of the peeling of
/// `frobenius.rs` (main path).
fn minimal_polynomial_peeled<F: Field>(matrix: &Matrix<F>) -> Result<UnivariatePolynomial<F>> {
    peel(matrix, PeelDepth::MinimalPolynomial)?
        .minimal_polynomial()
        .ok_or_else(|| {
            MathError::InvalidOperation(
                "internal: the peeling found no level for a non-empty matrix".to_string(),
            )
        })
}

/// The fallback chain behind the public functions: the Smith form of the Krylov relation
/// matrix `R`, then the Smith form of `xI - A`.
fn invariant_factors_fallback<F: Field>(
    matrix: &Matrix<F>,
) -> Result<Vec<UnivariatePolynomial<F>>> {
    invariant_factors_krylov(matrix).or_else(|_| invariant_factors_classical(matrix))
}

/// MAGMA `InvariantFactors(A)` (Handbook ch. 26 §26.12.2): the non-constant invariant factors
/// of `A`, monic, `f_1 | f_2 | … | f_r` (increasing), with product the characteristic
/// polynomial.
///
/// `Err(InvalidArgument)` if `A` is not square; `[]` for the 0×0 matrix. Computed by the
/// cyclic decomposition by maximal vectors (module docs; crate module `frobenius`): a Krylov
/// presentation gives r generators of `F^n` as an `F[x]`-module (for r = 1 its one relation is
/// the answer), and each level peels off a cyclic summand whose generator is maximal modulo the
/// summands found so far; its relative minimal polynomial is the next invariant factor, from
/// the largest (the minimal polynomial) down. Every claim is checked by exact computation in
/// the run: each spin relation, the annihilator of each merged vector, an exponent certificate
/// per level (every candidate is annihilated by the level's factor modulo the earlier
/// summands), the divisibility chain and the total degree n. A failed check is
/// `Err(InvalidOperation)`, never a value.
///
/// Complexity: `O((r + 1)·n³)` field operations, r ≤ n the number of Krylov blocks (usually 1
/// for a random matrix; n for the inputs described in the module docs of `krylov`). The bound
/// holds for this main path only; the fallback has none, and over Q no bit-complexity bound is
/// proved. Measured times are in the module docs, including the inputs with r ≈ s ≈ n (scalar,
/// diagonal with few distinct values) where the Smith form of `R` was faster.
///
/// Fallback: if the peeling returns `Err`, the factors come from the Smith form of the r×r
/// relation matrix `R` (without transforms; its output check certifies the diagonal, see
/// [`PolynomialMatrix::smith_normal_form`]), and if that returns `Err` too, from the Smith form
/// of `xI - A` (the classical path, equally certified); the result or error of the last stage is
/// returned. Over an exact field (Q, bound GF(p) or GF(p^n)) the peeling does not fail. Two
/// kinds of input reach the fallbacks. First, matrices built only from the unbound
/// `Ring::zero()`/`one()` sentinels of `PrimeField` or of the GF(p^n) type
/// `FiniteFieldElement`, which are computed in Z: there the ±1 start vectors can create a
/// non-unit pivot (a 3-cycle, `J_3(1)` or the all-ones 3×3 matrix, for example), while the
/// constant pivots of `xI - A` are entries of `A`. In Z semantics a result is only returned when
/// every inverted pivot is ±1, so it is the result over Z and, reduced, over GF(p) for every p
/// (module docs of `krylov` and `frobenius`, "Exactness"); a non-unit pivot in every stage (for
/// example `[0, 0; 2, 0]`, or the upper all-ones 5×5 matrix) is `Err`. Second, a rounding
/// `Field` impl (`Real`) whose checks fail: the result is that of the first stage that passes
/// its checks, with no accuracy claim, and it can be `Err`.
pub fn invariant_factors<F: Field>(matrix: &Matrix<F>) -> Result<Vec<UnivariatePolynomial<F>>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Invariant factors are only defined for square matrices".to_string(),
        ));
    }
    invariant_factors_peeled(matrix).or_else(|_| invariant_factors_fallback(matrix))
}

/// MAGMA `MinimalPolynomial(A)` (Handbook ch. 26 §26.11; also ch. 59 §59.4.3
/// `MinimalPolynomial(g)`): the monic polynomial `f` of least degree with `f(A) = 0`, which is
/// the last invariant factor of `A`.
///
/// `Err(InvalidArgument)` if `A` is not square; the constant 1 for the 0×0 matrix. Computed by
/// the first level of the peeling of [`invariant_factors`] only: a maximal vector, whose
/// minimal polynomial (with the exponent certificate of that level) is `μ_A`. Complexity:
/// `O((r + 1)·n³)` field operations, as for [`invariant_factors`] (less when the minimal
/// polynomial has small degree). The bound holds for this main path only; the fallback has
/// none.
///
/// Fallback: if the peeling returns `Err`, the last invariant factor of the fallback chain of
/// [`invariant_factors`] (the Smith form of `R`, then of `xI - A`), with the same triggers
/// (unbound sentinels computed in Z, a rounding `Field` impl). In Z semantics a peeled result is
/// `μ` over Z and, reduced, over GF(p) for every p (module docs of `frobenius`, "Exactness").
/// Since only the first level runs, this can answer where [`invariant_factors`] returns `Err`.
pub fn minimal_polynomial<F: Field>(matrix: &Matrix<F>) -> Result<UnivariatePolynomial<F>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Invariant factors are only defined for square matrices".to_string(),
        ));
    }
    if matrix.rows() == 0 {
        return Ok(UnivariatePolynomial::new(vec![F::one()]));
    }
    minimal_polynomial_peeled(matrix).or_else(|_| {
        Ok(invariant_factors_fallback(matrix)?
            .pop()
            .unwrap_or_else(|| UnivariatePolynomial::new(vec![F::one()])))
    })
}

/// Compute the rational canonical form of a matrix (MAGMA `RationalForm(A)`, Handbook ch. 26
/// §26.12.2)
///
/// Returns the invariant factors `f_1 | … | f_r` (as [`invariant_factors`]), the block
/// diagonal matrix `C` of the companion matrices of the `f_i` ([`companion_matrix`], column
/// style) in increasing divisibility order, and `change_of_basis = Some(T)` with
/// `T⁻¹·A·T = C`. MAGMA's `F` is `Cᵀ`; `T⁻¹` has MAGMA's defining property `T_M·A·T_M⁻¹ = F`
/// only when `C` is symmetric, and PARI's (`B·A·B⁻¹ = F_P`) up to the block order; see
/// [`RationalCanonicalForm::canonical_matrix`] and [`RationalCanonicalForm::change_of_basis`].
///
/// Construction: the peeling of [`invariant_factors`] (crate module `frobenius`) gives maximal
/// vectors `v_ℓ` with relations `f_ℓ(A)·v_ℓ = Σ_{i<ℓ} c_{ℓi}(A)·v_i` on the Krylov vectors
/// already found. Every `c_{ℓi}` is divisible by `f_ℓ` (a consequence of the cyclic
/// decomposition theorem; each division is checked to be exact), and the clean generators
/// `g_ℓ = v_ℓ - Σ_i (c_{ℓi}/f_ℓ)(A)·v_i` satisfy `f_ℓ(A)·g_ℓ = 0` and `F^n = ⊕_ℓ K(g_ℓ)`,
/// `K(g) = span(g, A·g, …)`. The columns of `T` are `g, A·g, …, A^{k-1}·g` block by block
/// (`k = deg f_ℓ`). For a cyclic `A` (r = 1) `T` is the Krylov basis of the first start vector.
///
/// When `T` is not needed,
/// `RationalCanonicalForm::from_invariant_factors(invariant_factors(&a)?)` gives the same
/// factors and `C` (with `change_of_basis = None`) and skips the transforms and the
/// certificate.
///
/// Fallback: if the peeled construction returns `Err`, the Krylov construction runs (the Smith
/// form of the relation matrix `R` with `P⁻¹`, as before), and if that returns `Err` too, the
/// same construction on the classical presentation by `xI - A` with the unit vectors as
/// generators; the result or error of the last stage is returned. Over an exact field (Q, bound
/// GF(p) or GF(p^n)) the first stage does not fail. Two kinds of input reach the fallbacks.
/// First, matrices built only from the unbound `Ring::zero()`/`one()` sentinels of `PrimeField`
/// or `FiniteFieldElement`, computed in Z, where only ±1 are invertible: a stage that meets
/// another pivot returns `Err`. There a result is returned only when every inverted pivot is ±1
/// and `det T = ±1`, so it is the result over Z and, reduced, over GF(p) for every p. Second, a
/// rounding `Field` impl (`Real`) whose relation checks or certificate fail; the result is then
/// whatever stage passes its checks, with no accuracy claim, and it can be `Err` (the 8×8
/// matrix of the `krylov` tests). The Krylov construction (`rcf_from_presentation`): with
/// `P·R·Q = S` the Smith form, the generator of the i-th summand comes from column i of `P⁻¹`
/// (Dummit–Foote §12.1–12.2), reduced modulo `im R` so that it is a combination of the Krylov
/// basis vectors, and taken from `P⁻¹·D⁻¹` (`D` the leading coefficients that make the pivots
/// monic) to keep out scalar factors of hundreds of digits over Q.
///
/// Certificate: the result is returned only if `T` is square of size n, `A·T == T·C` and the
/// columns of `T` are linearly independent; then `A` is similar to a block companion matrix
/// whose polynomials form a divisibility chain, which by uniqueness of the rational canonical
/// form proves both the factors and `C`. Otherwise `Err(InvalidOperation)`.
/// `Err(InvalidArgument)` if `A` is not square. The independence test is an exact incremental
/// elimination that inverts its pivots with `Field::inverse` (never `/`). In Z semantics its
/// pivot (the first non-zero entry of a reduced column) can be a non-unit such as ±2 although
/// `T` is unimodular; when the elimination returns that `Err`, independence is decided by the
/// division-free determinant instead: `T` passes iff the constant term `(-1)^n·det T` of
/// [`charpoly_berkowitz`]`(T)` is invertible, which in Z semantics means `det T = ±1`, so `T`
/// is invertible over Z and modulo every p.
///
/// Complexity: `O((r + 1)·n³)` field operations for the peeling, the clean generators, `T` and
/// the certificate (the main path; module docs). Over Q no bit-complexity bound is proved. The
/// fallbacks have no bound: there the cost of the transform grows with the entries of `P⁻¹`
/// (for `A = U·diag(1, …, 20)·U⁻¹`, U unimodular, r = 2 or 3, the Krylov construction took
/// 1–6 s in debug, and on the `W·B·W⁻¹` input with n = 14, 19 s on this tree). Measured times
/// of the main path are in the module docs.
pub fn rational_canonical_form<F: Field>(
    matrix: &Matrix<F>,
) -> Result<RationalCanonicalForm<F>> {
    if !matrix.is_square() {
        return Err(MathError::InvalidArgument(
            "Rational canonical form is only defined for square matrices".to_string(),
        ));
    }
    if matrix.rows() == 0 {
        return Ok(RationalCanonicalForm {
            invariant_factors: Vec::new(),
            canonical_matrix: Matrix::from_vec(0, 0, Vec::new())?,
            change_of_basis: Some(Matrix::from_vec(0, 0, Vec::new())?),
        });
    }
    rational_canonical_form_peeled(matrix)
        .or_else(|_| rational_canonical_form_krylov(matrix))
        .or_else(|_| rational_canonical_form_classical(matrix))
}

/// [`rational_canonical_form`] of a square, non-empty `A` by the peeling of `frobenius.rs`
/// (main path): its invariant factors and clean generators, then [`certified_form`].
fn rational_canonical_form_peeled<F: Field>(
    matrix: &Matrix<F>,
) -> Result<RationalCanonicalForm<F>> {
    let peeling = peel(matrix, PeelDepth::Full)?;
    let generators = peeling.generators(matrix)?;
    certified_form(matrix, peeling.invariant_factors(), generators)
}

/// [`rational_canonical_form`] of a square, non-empty `A` from the Smith form of the Krylov
/// relation matrix (first fallback).
fn rational_canonical_form_krylov<F: Field>(
    matrix: &Matrix<F>,
) -> Result<RationalCanonicalForm<F>> {
    let presentation = krylov_presentation(matrix)?;
    rcf_from_presentation(
        matrix,
        &presentation.generators,
        &presentation.relations,
        Some(&presentation.basis),
    )
}

/// [`rational_canonical_form`] of a square, non-empty `A` from the presentation by `xI - A`
/// on the unit vectors (classical path, second fallback).
fn rational_canonical_form_classical<F: Field>(
    matrix: &Matrix<F>,
) -> Result<RationalCanonicalForm<F>> {
    let n = matrix.rows();
    let (z, one, _) = field_constants(matrix.data().iter());
    let units: Vec<Vec<F>> = (0..n)
        .map(|j| {
            let mut e = vec![z.clone(); n];
            e[j] = one.clone();
            e
        })
        .collect();
    rcf_from_presentation(matrix, &units, &characteristic_matrix(matrix)?, None)
}

/// `φ(p) = Σ_j p_j(A)·w_j` for `p ∈ F[x]^r` and a Krylov presentation (`relations` = `R`,
/// upper triangular with monic diagonal entries of degrees `k_j`; `basis` the Krylov basis
/// `u_{off_j+t} = A^t·w_j` of [`KrylovPresentation::basis`]), without evaluating at `A`.
///
/// `p` is first reduced modulo `im R` by back substitution from the last row: with
/// `p_j = q·R[j][j] + rem` (`R[j][j]` monic, so the division only inverts its leading 1),
/// `p ← p − q·(column j of R)`. Column j of `R` is zero below row j, so the rows already
/// reduced keep `deg p_s < k_s`, and at the end `deg p_j < k_j` for every j. Each step
/// subtracts an element of `im R ⊆ ker φ`, so `φ(p)` is unchanged, and
/// `φ(p) = Σ_j Σ_{t<k_j} coeff_t(p_j)·u_{off_j+t}`: `O(n²)` field operations instead of
/// `deg p_j` products by `A` per generator. (`P⁻¹` from the Smith form of `R` can have
/// entries of degree far above `k_j` with large coefficients over Q when r > 1; evaluating
/// those at `A` by Horner's rule was the bottleneck of [`rational_canonical_form`].)
///
/// `Err(InvalidOperation)` if the shapes do not fit or a reduced entry does not have degree
/// `< k_j` (impossible for the output of `krylov_presentation`); errors from `Field::inverse`
/// propagate.
///
/// [`KrylovPresentation::basis`]: crate::krylov::KrylovPresentation::basis
fn image_in_krylov_basis<F: Field>(
    mut p: Vec<UnivariatePolynomial<F>>,
    relations: &PolynomialMatrix<F>,
    basis: &[Vec<F>],
    z: &F,
) -> Result<Vec<F>> {
    let fail = || {
        Err(MathError::InvalidOperation(
            "internal: the Krylov presentation does not fit its basis".to_string(),
        ))
    };
    let r = relations.rows();
    if p.len() != r || relations.cols() != r {
        return fail();
    }
    for j in (0..r).rev() {
        let (q, rem) = div_rem_field(&p[j], relations.get(j, j))?;
        if q.is_zero() {
            continue;
        }
        for (i, pi) in p.iter_mut().enumerate().take(j) {
            let rij = relations.get(i, j);
            if !rij.is_zero() {
                *pi = pi.clone() - q.clone() * rij.clone();
            }
        }
        p[j] = rem;
    }
    let n = basis.len();
    let mut g = vec![z.clone(); n];
    let mut off = 0;
    for (j, pj) in p.iter().enumerate() {
        let Some(k) = relations.get(j, j).degree() else {
            return fail();
        };
        if off + k > n || pj.degree().is_some_and(|d| d >= k) {
            return fail();
        }
        for (t, c) in pj.coefficients().iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let u = &basis[off + t];
            if u.len() != n {
                return fail();
            }
            for (gs, us) in g.iter_mut().zip(u) {
                *gs = gs.clone() + c.clone() * us.clone();
            }
        }
        off += k;
    }
    if off != n {
        return fail();
    }
    Ok(g)
}

/// `φ(p) = Σ_j p_j(A)·w_j` for any presentation, by Horner's rule on each `p_j(A)·w_j`
/// (`krylov::poly_mat_vec`).
fn image_by_horner<F: Field>(
    p: &[UnivariatePolynomial<F>],
    matrix: &Matrix<F>,
    generators: &[Vec<F>],
    z: &F,
) -> Vec<F> {
    let mut g = vec![z.clone(); matrix.rows()];
    for (pj, w) in p.iter().zip(generators) {
        if pj.is_zero() {
            continue;
        }
        let v = poly_mat_vec(pj, matrix, w, z);
        for (gr, vr) in g.iter_mut().zip(v) {
            *gr = gr.clone() + vr;
        }
    }
    g
}

/// Whether the columns of the square `t` (given as `columns`) are linearly independent, as
/// the certificate of [`rational_canonical_form`] needs it.
///
/// First the exact elimination of [`linearly_independent`]. That elimination takes the first
/// non-zero entry of each reduced column as its pivot, so in Z semantics (a matrix of unbound
/// `PrimeField`/GF(p^n) sentinels) it can meet a non-unit pivot such as ±2 although `T` is
/// unimodular, and returns `Err`. Then the answer comes from the division-free determinant
/// instead: `T` is accepted iff the constant term `(-1)^n·det T` of [`charpoly_berkowitz`]`(T)`
/// is invertible (`Field::inverse` succeeds). Over a field this is `det T ≠ 0`; in Z semantics
/// it is `det T = ±1`, so `T` is invertible over Z and modulo every p.
fn independent_columns<F: Field>(t: &Matrix<F>, columns: &[Vec<F>]) -> Result<bool> {
    match linearly_independent(columns) {
        Ok(independent) => Ok(independent),
        Err(_) => Ok(charpoly_berkowitz(t)?.coeff(0).inverse().is_ok()),
    }
}

/// The certified rational form of a square, non-empty `A` from a presentation
/// `F^n ≅ F[x]^r / im(relations)`, `φ: e_j ↦ generators[j]` (see [`rational_canonical_form`]).
///
/// The Smith form `S` of `R = relations` comes with `X = P⁻¹·D⁻¹`
/// ([`PolynomialMatrix::smith_form_with_scaled_left_inverse`]; `D` diagonal and invertible),
/// and `g_i = φ(column i of X)`: a unit multiple of `φ(column i of P⁻¹)`, so a generator of
/// the i-th summand `F[x]/(d_i)` all the same. With `basis = Some(u)` the presentation is a
/// Krylov one (`R` upper triangular with monic diagonal, `u` its Krylov basis) and `g_i` is
/// computed by [`image_in_krylov_basis`]; with `None` (the classical presentation by
/// `xI - A`) by [`image_by_horner`]. Both give the same vector.
fn rcf_from_presentation<F: Field>(
    matrix: &Matrix<F>,
    generators: &[Vec<F>],
    relations: &PolynomialMatrix<F>,
    basis: Option<&[Vec<F>]>,
) -> Result<RationalCanonicalForm<F>> {
    let n = matrix.rows();
    let (z, _, _) = field_constants(matrix.data().iter());
    let r = relations.rows();
    let (smith, x) = relations.smith_form_with_scaled_left_inverse()?;

    let mut factors = Vec::new();
    let mut gens: Vec<Vec<F>> = Vec::new();
    for i in 0..r {
        let d = smith.get(i, i);
        match d.degree() {
            None => {
                return Err(MathError::InvalidOperation(
                    "internal: zero diagonal entry in the Smith form of the relation matrix"
                        .to_string(),
                ))
            }
            Some(0) => continue,
            Some(_) => {}
        }
        // g_i = φ(column i of X) = Σ_j X_{j,i}(A)·w_j
        let column: Vec<UnivariatePolynomial<F>> = (0..r).map(|j| x.get(j, i).clone()).collect();
        let g = match basis {
            Some(u) => image_in_krylov_basis(column, relations, u, &z)?,
            None => image_by_horner(&column, matrix, generators, &z),
        };
        if g.len() != n {
            return Err(MathError::InvalidOperation(
                "internal: generator image of the wrong length".to_string(),
            ));
        }
        gens.push(g);
        factors.push(d.clone());
    }
    certified_form(matrix, factors, gens)
}

/// The certified rational form of a square, non-empty `A` from its non-constant invariant
/// factors `f_1 | … | f_s` (increasing) and generators `g_1, …, g_s` in the same order, where
/// `g_i` should generate the i-th cyclic summand (`f_i(A)·g_i = 0`): the columns of `T` are
/// `g_i, A·g_i, …, A^{k-1}·g_i` block by block (`k = deg f_i`), and `C` comes from the factors
/// ([`RationalCanonicalForm::from_invariant_factors`]). The result is returned only if `T` is
/// n×n, `A·T == T·C` and its columns are independent ([`independent_columns`]); otherwise
/// `Err(InvalidOperation("internal: rational form certificate failed"))`.
fn certified_form<F: Field>(
    matrix: &Matrix<F>,
    factors: Vec<UnivariatePolynomial<F>>,
    generators: Vec<Vec<F>>,
) -> Result<RationalCanonicalForm<F>> {
    let n = matrix.rows();
    let (z, _, _) = field_constants(matrix.data().iter());
    let certificate_failed = || {
        Err(MathError::InvalidOperation(
            "internal: rational form certificate failed".to_string(),
        ))
    };
    if factors.len() != generators.len() {
        return certificate_failed();
    }
    let mut columns: Vec<Vec<F>> = Vec::with_capacity(n);
    for (f, mut g) in factors.iter().zip(generators) {
        let k = match f.degree() {
            Some(k) if k > 0 => k,
            _ => return certificate_failed(),
        };
        if g.len() != n || columns.len() + k > n {
            return certificate_failed();
        }
        for _ in 0..k {
            let next = mat_vec(matrix, &g, &z);
            columns.push(std::mem::replace(&mut g, next));
        }
    }

    let mut rcf = RationalCanonicalForm::from_invariant_factors(factors)?;
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
    if matrix.mul(&t)? != t.mul(&rcf.canonical_matrix)? || !independent_columns(&t, &columns)? {
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
        let matrix_data = vec![Rational::from_integer(0); 4];
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
    //   matfrobenius(A) (the Frobenius form, blocks in decreasing order; reversed here).

    use crate::charpoly_berkowitz;
    use crate::krylov::krylov_presentation;

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
        // Over Q the main path itself succeeds (the fallback is not what is being tested), and
        // so does the first fallback (the Smith form of the Krylov relation matrix).
        assert_eq!(invariant_factors_peeled(a).unwrap(), expected);
        assert_eq!(invariant_factors_krylov(a).unwrap(), expected);
        let mp = minimal_polynomial(a).unwrap();
        assert_eq!(&mp, expected.last().unwrap());
        assert_eq!(minimal_polynomial_peeled(a).unwrap(), mp);
        assert_eq!(krylov_minpoly(a), mp);

        let rcf = rational_canonical_form(a).unwrap();
        assert_eq!(rcf.invariant_factors, expected);
        assert_eq!(rcf.minimal_polynomial(), Some(&mp));
        // The public T is the peeled one; the Krylov Smith path gives another certified T.
        assert_eq!(
            rational_canonical_form_peeled(a).unwrap().change_of_basis,
            rcf.change_of_basis
        );
        let krylov = rational_canonical_form_krylov(a).unwrap();
        assert_eq!(krylov.invariant_factors, expected);
        let tk = krylov.change_of_basis.expect("T is computed");
        assert_eq!(
            a.mul(&tk).unwrap(),
            tk.mul(&krylov.canonical_matrix).unwrap(),
            "Krylov path: A·T != T·C"
        );
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
        //     matfrobenius(A) = [0,0,12,0; 1,0,-16,0; 0,1,7,0; 0,0,0,2] (PARI's decreasing
        //     block order; C below lists the blocks increasingly)
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
    fn rcf_magma_row_convention_via_transpose() {
        // MAGMA's CompanionMatrix (Handbook ch. 23 digest) has 1s on the superdiagonal and
        // -a_0, …, -a_{k-1} in the last row, so MAGMA's RationalForm F is the transpose of
        // `canonical_matrix`, and MAGMA's T satisfies T·A·T⁻¹ = F. Such a T comes from the
        // change of basis T₂ of Aᵀ: T₂⁻¹·Aᵀ·T₂ = C gives T₂ᵀ·A·(T₂ᵀ)⁻¹ = Cᵀ.
        // A = J2(2) + J1(2) + [3]. gp: matfrobenius(A~) = [0,0,12,0; 1,0,-16,0; 0,1,7,0;
        // 0,0,0,2] (decreasing block order); in increasing order and transposed this is
        // f_magma below (superdiagonal 1s, last row 12, -16, 7 for x^3 - 7*x^2 + 16*x - 12).
        let a = qm(4, &[2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3]);
        let f_magma = qm(4, &[2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 12, -16, 7]);
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(rcf.canonical_matrix.transpose(), f_magma);
        assert_ne!(rcf.canonical_matrix, f_magma);
        // T⁻¹ conjugates A to C, not to MAGMA's F.
        let t = rcf.change_of_basis.clone().expect("T is computed");
        let tinv = t.inverse().unwrap().expect("T invertible");
        assert_eq!(tinv.mul(&a).unwrap().mul(&t).unwrap(), rcf.canonical_matrix);
        // T₂ᵀ from the rational form of Aᵀ has MAGMA's defining property.
        let rcf_t = rational_canonical_form(&a.transpose()).unwrap();
        assert_eq!(rcf_t.canonical_matrix, rcf.canonical_matrix);
        let tm = rcf_t.change_of_basis.expect("T₂ is computed").transpose();
        let tm_inv = tm.inverse().unwrap().expect("T₂ᵀ invertible");
        assert_eq!(tm.mul(&a).unwrap().mul(&tm_inv).unwrap(), f_magma);
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

    // ---------------------------------------------------------------------------------------
    // Speed over Q (Krylov presentation) and the rational-form conventions.
    //
    // Test matrices, row-major. DENSE12 and DENSE20: entries in [-5, 5]. UPTRI14: upper
    // triangular, diagonal 1..14, strict upper part in [-5, 5]. DEROG12:
    // U·(C(x^3+x+1) ⊕ C(x-2))^{⊕3}·U⁻¹, U unimodular. NILP15: U·(J1(0) ⊕ … ⊕ J5(0))·U⁻¹, U
    // unimodular. In gp, with v the entries of an n×n one: A = matrix(n, n, i, j, v[(i-1)*n+j]).
    // ---------------------------------------------------------------------------------------

    #[rustfmt::skip]
    const DENSE12: [i64; 144] = [
        0, 2, -2, -1, 1, 3, -5, 3, 5, 1, -4, 1,
        -4, -5, 0, -1, 5, 1, 2, -4, -1, 0, 1, -5,
        1, 4, 0, -2, 1, 1, 2, 5, -3, -5, 1, 2,
        5, -4, 2, 2, 2, -4, 3, 1, 5, 3, -4, -4,
        5, 5, 0, -5, -1, 2, -3, -5, -1, 3, 1, -4,
        -1, 1, 5, -2, -5, -5, 3, 3, 0, -4, -3, -1,
        2, 0, -3, 3, 2, 0, -4, 2, 1, -5, 4, -2,
        3, -2, -2, 5, 2, 0, 2, 0, -4, 4, -2, -4,
        -1, 2, 0, 2, 4, -1, -5, -2, 3, 5, 5, -2,
        -4, 5, -2, -2, 1, -5, 5, 1, -4, 1, -5, 4,
        -5, -2, 0, 0, -2, 0, 2, -2, -2, 1, -2, -5,
        5, 5, -4, 4, -1, -5, -4, 4, 5, -2, 0, 0,
    ];
    #[rustfmt::skip]
    const DENSE20: [i64; 400] = [
        5, -4, 5, -4, 3, 1, 0, 2, -5, 3, -1, -5, 3, -1, 2, -3, -4, -5, 3, -5,
        0, 1, 4, 2, 3, -4, 3, 5, -1, -5, 2, -3, -5, 0, -1, 4, 4, 2, -1, -3,
        -5, -2, 4, 0, 5, 2, 5, 0, -4, 1, 5, -5, 0, 5, 0, -2, 2, 3, 2, 5,
        1, 4, 5, 0, -3, -3, 1, -5, 4, -3, -4, -4, -4, -2, -4, 5, -5, 2, -4, -4,
        -2, 3, 1, 3, -4, 1, -5, -3, 1, -4, 4, 5, -1, 1, 4, -5, 1, -2, 2, 2,
        3, 0, 1, -4, -3, -3, 4, -3, 4, -4, 0, -3, -4, -5, 5, -4, 1, 0, -1, -4,
        0, 1, 3, 5, -5, 1, -1, -1, 3, -3, -1, -2, -5, -3, 3, -5, 1, -3, -2, 5,
        0, -4, 0, -4, -5, 3, 3, 3, 5, 1, 0, 5, 5, 3, -1, 2, 1, 5, 2, -1,
        -5, -5, -1, 4, -5, 2, -3, -4, 4, -1, 0, 3, 1, 2, 2, -4, 3, -5, 0, -5,
        0, 2, 3, -2, 4, 0, -1, 2, -1, -3, 3, 0, 1, 3, -3, 1, -5, -5, -2, 5,
        4, -1, 5, 2, 0, 5, 3, 3, -2, -5, 3, 2, 1, 3, 3, 0, 2, 0, -3, -4,
        -3, -5, -2, -1, -3, 5, -1, -4, 5, -3, -2, -4, 5, -5, 1, -4, -4, 0, -2, 4,
        3, 5, -4, -3, -3, 1, 5, 0, -4, -5, -4, 5, 1, -2, -4, -1, 1, 3, -4, -1,
        1, 3, -3, -5, 4, -5, 1, -5, 0, -1, -2, 1, 5, 3, 3, -5, 5, -4, 3, 5,
        -3, -4, 3, 0, -5, 1, 0, 3, -4, 5, -4, -4, 0, -5, 3, 5, 3, -4, -1, -4,
        0, 4, 2, -1, 3, -1, -1, 5, -3, 5, -2, -2, 2, -2, -5, 3, -1, 2, 2, 4,
        4, 3, -3, -4, -1, 0, 2, -5, 3, -2, -1, 0, 0, -2, -3, -2, -1, 4, -2, -5,
        4, 3, -5, -3, -3, 3, 5, 1, 5, -4, 2, -2, 0, 1, -2, -4, 2, -4, 2, 3,
        0, -5, 0, 0, -4, 1, 1, -4, 4, -4, -2, -5, 0, 2, 5, 0, -5, -4, 2, 1,
        0, 1, -5, -3, -5, -4, 0, 5, -3, -3, -5, 4, -4, 3, 0, 3, -1, 4, -2, 1,
    ];
    #[rustfmt::skip]
    const UPTRI14: [i64; 196] = [
        1, 0, -4, -2, 4, 0, 4, 0, -4, -4, 1, -5, -5, -2,
        0, 2, 3, -3, 4, 0, 1, 0, -1, -5, 2, -5, 0, -1,
        0, 0, 3, 3, 1, -1, 4, -1, -1, 0, 4, 2, -3, 5,
        0, 0, 0, 4, -2, -5, -1, -1, 0, -1, 3, -1, 3, 1,
        0, 0, 0, 0, 5, 2, -1, -1, -3, -2, 4, 5, 4, -5,
        0, 0, 0, 0, 0, 6, -4, -4, 0, -4, 4, 4, -4, 2,
        0, 0, 0, 0, 0, 0, 7, 3, -4, -4, 0, 3, 3, -2,
        0, 0, 0, 0, 0, 0, 0, 8, -4, 1, -2, 4, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 9, -1, 2, -4, 5, -2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 10, -4, 5, 5, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 2, -3, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 1, 3,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14,
    ];
    #[rustfmt::skip]
    const DEROG12: [i64; 144] = [
        2, -4, 7, 0, -12, 0, 0, 0, -21, -10, -2, 0,
        -2, 4, -9, 2, 12, 2, -2, 0, 16, 8, 1, 2,
        0, 1, 0, 0, 2, 0, 0, 0, 5, 2, 0, 0,
        0, 0, -1, 2, -4, 0, 0, 0, -1, -2, -2, 0,
        2, 2, 8, 1, 0, 1, -1, 0, 6, 1, -3, 1,
        -1, 2, -4, 1, 1, 1, -1, -4, 14, 3, -8, 1,
        -3, 2, -14, 3, 7, 2, -1, -2, 10, 3, -2, 3,
        0, -2, 2, -1, -6, -1, 1, 2, -12, -5, 5, -1,
        1, 0, 4, 0, -2, 0, 0, 0, -1, -1, -1, 0,
        -3, -4, -12, -2, -2, -2, 2, 0, -12, -3, 5, -2,
        0, 0, -1, 0, 2, 0, 0, 0, 0, 1, 0, 0,
        -4, -4, -16, -2, 4, -2, 2, 0, -12, -2, 6, 0,
    ];
    #[rustfmt::skip]
    const NILP15: [i64; 225] = [
        -1, 2, 1, -2, 4, -2, -3, -5, 0, -1, 0, -4, -1, 1, 0,
        -1, 4, 2, -1, 1, -2, -1, -2, 1, 0, 0, -1, -2, 0, 2,
        0, 0, 0, -1, -2, 0, 0, 2, 1, 0, 0, 2, 0, -1, 1,
        -1, 2, 1, -2, 4, -2, -3, -5, 0, -1, 0, -4, -1, 1, 0,
        1, -2, 0, -3, -5, 3, 1, 5, 3, -1, 0, 5, 1, -4, 3,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, -2, -1, 2, -3, 3, 3, 4, 0, 1, 0, 3, 1, -1, 0,
        1, 0, 0, -3, -5, 2, 1, 5, 3, -1, 0, 5, 0, -4, 3,
        -2, 2, 1, -1, 6, -4, -4, -7, -1, 0, 0, -6, -1, 3, 1,
        -1, 0, 0, 1, 1, -2, -1, -1, -1, 1, 0, -1, 0, 2, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        -3, 8, 4, -2, 3, -6, -3, -5, 2, 0, 0, -3, -4, 1, 4,
        0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// The block diagonal matrix of the column-style companion matrices of `fs`, in order.
    fn block_companion(fs: &[UnivariatePolynomial<Rational>], n: usize) -> Matrix<Rational> {
        let mut c = Matrix::<Rational>::zeros(n, n);
        let mut off = 0;
        for f in fs {
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
        c
    }

    /// `invariant_factors`, `minimal_polynomial` and `rational_canonical_form` on a larger
    /// matrix, timed together against a debug-friendly 60 s bound (at least 20x the measured
    /// debug time; the old xI - A path did not finish DENSE12 in 95 s even in release). The
    /// certificate is re-checked independently of the Krylov elimination: `C` is the block
    /// companion matrix of `expected`, `A·T == T·C`, `Matrix::rank(T) == n`, and the product of
    /// the factors equals `charpoly_berkowitz(A)`.
    fn check_forms_timed(a: &Matrix<Rational>, expected: &[UnivariatePolynomial<Rational>]) {
        check_forms_bounded(a, expected, 60);
    }

    /// [`check_forms_timed`] with a bound of `bound_secs` seconds on the three public calls.
    fn check_forms_bounded(
        a: &Matrix<Rational>,
        expected: &[UnivariatePolynomial<Rational>],
        bound_secs: u64,
    ) {
        let n = a.rows();
        let start = std::time::Instant::now();
        let fs = invariant_factors(a).unwrap();
        let mp = minimal_polynomial(a).unwrap();
        let rcf = rational_canonical_form(a).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < bound_secs,
            "n = {n}: the three functions took {elapsed:?} (bound {bound_secs} s)"
        );
        assert_eq!(fs, expected);
        assert_eq!(&mp, expected.last().unwrap());
        assert_eq!(rcf.invariant_factors, expected);
        // The main path itself succeeds (no fallback).
        assert!(krylov_presentation(a).is_ok());
        assert_eq!(invariant_factors_peeled(a).unwrap(), expected);
        assert_eq!(rcf.canonical_matrix, block_companion(expected, n));
        let t = rcf.change_of_basis.clone().expect("T is computed");
        assert_eq!(
            a.mul(&t).unwrap(),
            t.mul(&rcf.canonical_matrix).unwrap(),
            "A·T != T·C"
        );
        assert_eq!(t.rank().unwrap(), n, "T singular");
        let product = expected.iter().fold(px(&[1]), |acc, f| acc * f.clone());
        assert_eq!(product, charpoly_berkowitz(a).unwrap());
    }

    /// gp: matfrobenius(dense12,1) = [x^12 + 11*x^11 + 83*x^10 + 427*x^9 + 3658*x^8
    /// + 125485*x^7 + 755007*x^6 + 3188519*x^5 + 22410764*x^4 - 30671985*x^3 - 215552386*x^2
    /// - 614931624*x - 4228284306], equal to minpoly(dense12) and charpoly(dense12).
    fn dense12_minpoly() -> UnivariatePolynomial<Rational> {
        px(&[
            -4228284306,
            -614931624,
            -215552386,
            -30671985,
            22410764,
            3188519,
            755007,
            125485,
            3658,
            427,
            83,
            11,
            1,
        ])
    }

    /// gp: Vecrev(minpoly(dense20)) (= Vecrev(charpoly(dense20)); matfrobenius(dense20,1) has
    /// one entry).
    fn dense20_minpoly() -> UnivariatePolynomial<Rational> {
        px(&[
            603020736141495660,
            -2883540038105079373,
            -946672124533905457,
            -72831001880833123,
            17694097771293239,
            40492637320332,
            284130608439815,
            -43574319521858,
            -4601776037495,
            369427153998,
            64281565916,
            -1667016978,
            -67158136,
            -11106759,
            3928983,
            -235499,
            -37171,
            3434,
            -10,
            -13,
            1,
        ])
    }

    /// A deterministic n×n matrix with entries in [-5, 5] (64-bit LCG, MMIX constants).
    fn lcg_dense(n: usize, seed: u64) -> Matrix<Rational> {
        let mut state = seed;
        let v: Vec<i64> = (0..n * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % 11) as i64 - 5
            })
            .collect();
        qm(n, &v)
    }

    #[test]
    fn forms_dense12_q() {
        // T-a1. gp: matfrobenius(dense12,1) = [f], minpoly(dense12) == f, charpoly(dense12) == f
        // (f = dense12_minpoly()).
        let a = qm(12, &DENSE12);
        let f = dense12_minpoly();
        check_forms_timed(&a, std::slice::from_ref(&f));
        assert_eq!(
            rational_canonical_form(&a).unwrap().canonical_matrix,
            companion_matrix(&f).unwrap()
        );
    }

    #[test]
    fn forms_dense20_q_timing() {
        // T-a2. gp: Vecrev(minpoly(dense20)) == Vecrev(charpoly(dense20)) == the coefficients
        // of dense20_minpoly() (all fit in i64: the largest is 2883540038105079373 < 2^63).
        check_forms_timed(&qm(20, &DENSE20), &[dense20_minpoly()]);
    }

    #[test]
    fn forms_uptri14_q() {
        // T-a3: spinning the unit vectors would give r = n = 14 here (R = xI - A).
        // gp: minpoly(uptri14) = charpoly(uptri14), factor(charpoly(uptri14)) = the 14 factors
        //     x - i, Vecrev(charpoly(uptri14)) = [87178291200, -283465647360, 392156797824,
        //     -310989260400, 159721605680, -56663366760, 14409322928, -2681453775, 368411615,
        //     -37312275, 2749747, -143325, 5005, -105, 1]
        let f = px(&[
            87178291200,
            -283465647360,
            392156797824,
            -310989260400,
            159721605680,
            -56663366760,
            14409322928,
            -2681453775,
            368411615,
            -37312275,
            2749747,
            -143325,
            5005,
            -105,
            1,
        ]);
        let product = (1..=14).fold(px(&[1]), |acc, i| acc * px(&[-i, 1]));
        assert_eq!(f, product);
        let a = qm(14, &UPTRI14);
        check_forms_timed(&a, &[f]);
        assert_eq!(krylov_presentation(&a).unwrap().relations.rows(), 1);
    }

    #[test]
    fn forms_derog12_q() {
        // T-a4. gp: matfrobenius(derog12,1) = [f, f, f] with f = x^4 - 2*x^3 + x^2 - x - 2,
        //     factor(charpoly(derog12)) = [x - 2, 3; x^3 + x + 1, 3]
        let a = qm(12, &DEROG12);
        let f = px(&[-2, -1, 1, -2, 1]);
        check_forms(&a, &[f.clone(), f.clone(), f]);
        // r >= the number of invariant factors (a theorem), so the r > 1 Smith path runs.
        assert!(krylov_presentation(&a).unwrap().relations.rows() >= 3);
    }

    #[test]
    fn forms_nilp15_q() {
        // T-a5. gp: matfrobenius(nilp15,1) = [x^5, x^4, x^3, x^2, x], nilp15^5 == 0,
        //     nilp15^4 != 0
        let a = qm(15, &NILP15);
        let expected: Vec<_> = (1..=5)
            .map(|k| {
                let mut c = vec![0; k + 1];
                c[k] = 1;
                px(&c)
            })
            .collect();
        check_forms(&a, &expected);
        assert!(krylov_presentation(&a).unwrap().relations.rows() >= 5);
    }

    /// The non-constant diagonal of the classical Smith form of `xI - A` (the previous
    /// implementation, kept as an independent oracle).
    fn smith_oracle(a: &Matrix<Rational>) -> Vec<UnivariatePolynomial<Rational>> {
        let snf = characteristic_matrix(a)
            .unwrap()
            .smith_normal_form()
            .unwrap();
        (0..a.rows())
            .map(|i| snf.diagonal.get(i, i).clone())
            .filter(|d| d.degree().is_some_and(|k| k > 0))
            .collect()
    }

    /// The 69 matrices of `invariant_factors_agree_with_smith_of_xi_minus_a`: 9 small ones and
    /// 60 deterministic ones, n = 1..5 (dense, sparse, diagonal with repeated entries, upper
    /// triangular with repeated diagonal).
    fn agreement_matrices() -> Vec<Matrix<Rational>> {
        let mut mats = vec![
            qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3]),
            qm(2, &[1, 0, 0, 1]),
            qm(2, &[0, 1, 0, 0]),
            qm(2, &[0, 0, 0, 0]),
            qm(2, &[1, 2, 3, 4]),
            qm(3, &[1, 2, 0, 0, 1, 3, 4, 0, 1]),
            qm(4, &[2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3]),
            qm(4, &[2, 1, -1, 1, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 3]),
            qm(
                5,
                &[
                    2, -5, 0, 1, -2, 1, -2, 0, 0, -1, 1, -2, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 1,
                    0,
                ],
            ),
        ];
        // 60 deterministic matrices, n = 1..5: dense, sparse, diagonal with repeated entries,
        // upper triangular with repeated diagonal.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        for t in 0..60usize {
            let n = 1 + t % 5;
            let kind = (t / 5) % 4;
            let mut v = vec![0i64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let val = next(7) as i64 - 3;
                    let keep = match kind {
                        0 => true,
                        1 => next(4) == 0,
                        2 => i == j,
                        _ => i <= j,
                    };
                    if keep {
                        v[i * n + j] = if kind >= 2 && i == j {
                            next(2) as i64
                        } else {
                            val
                        };
                    }
                }
            }
            mats.push(qm(n, &v));
        }
        mats
    }

    #[test]
    fn invariant_factors_agree_with_smith_of_xi_minus_a() {
        // T-a6: the peeled path and the Krylov Smith path (called directly, so the fallback
        // cannot mask a failure) against the Smith form of xI - A, and the classical rational
        // form (the last fallback) against the Krylov one: same factors and C, each with its
        // own certified T.
        let mats = agreement_matrices();
        assert_eq!(mats.len(), 69);
        for a in &mats {
            let expected = smith_oracle(a);
            assert_eq!(invariant_factors_peeled(a).unwrap(), expected, "A = {a:?}");
            assert_eq!(invariant_factors_krylov(a).unwrap(), expected, "A = {a:?}");
            assert_eq!(invariant_factors(a).unwrap(), expected, "A = {a:?}");
            let peeled = rational_canonical_form_peeled(a).unwrap();
            assert_eq!(peeled.invariant_factors, expected, "A = {a:?}");
            let fast = rational_canonical_form_krylov(a).unwrap();
            assert_eq!(fast.invariant_factors, expected, "A = {a:?}");
            let classical = rational_canonical_form_classical(a).unwrap();
            assert_eq!(classical.invariant_factors, expected, "A = {a:?}");
            assert_eq!(
                classical.canonical_matrix, fast.canonical_matrix,
                "A = {a:?}"
            );
            assert_eq!(peeled.canonical_matrix, fast.canonical_matrix, "A = {a:?}");
            for rcf in [&peeled, &fast, &classical] {
                let t = rcf.change_of_basis.clone().expect("T is computed");
                assert_eq!(
                    inv(&t).mul(a).unwrap().mul(&t).unwrap(),
                    rcf.canonical_matrix,
                    "A = {a:?}"
                );
            }
        }
    }

    #[test]
    #[ignore = "timing report: cargo test --release -p rustmath-matrix --lib \
                release_timing_dense_q -- --ignored --nocapture"]
    fn release_timing_dense_q() {
        // T-a8: prints the times of the three functions; the n = 12 and n = 20 values are the gp
        // ones above, n = 30 is certificate-only (rational_canonical_form returns only a
        // certified result).
        let prod20 = (1..=20).fold(px(&[1]), |acc, i| acc * px(&[-i, 1]));
        let cases = [
            ("DENSE12", qm(12, &DENSE12), Some(dense12_minpoly())),
            ("DENSE20", qm(20, &DENSE20), Some(dense20_minpoly())),
            ("LCG30", lcg_dense(30, 30), None),
            // The 2×2 relation matrix case and the r = n case of the peeled-path tests.
            (
                "CONJDIAG20_S7",
                qm(20, &CONJDIAG20_S7),
                Some(prod20.clone()),
            ),
            (
                "WBW20",
                conj_start_basis(&qm(20, &upper_b(20, |i| i as i64 + 1))),
                Some(prod20),
            ),
        ];
        for (name, a, expected) in cases {
            let t0 = std::time::Instant::now();
            let fs = invariant_factors(&a).unwrap();
            let t1 = t0.elapsed();
            let mp = minimal_polynomial(&a).unwrap();
            let t2 = t0.elapsed() - t1;
            let rcf = rational_canonical_form(&a).unwrap();
            let t3 = t0.elapsed() - t1 - t2;
            println!(
                "{name}: invariant_factors {t1:?}, minimal_polynomial {t2:?}, \
                 rational_canonical_form {t3:?}"
            );
            assert_eq!(rcf.invariant_factors, fs);
            assert_eq!(fs.last(), Some(&mp));
            let degree: usize = fs.iter().filter_map(|f| f.degree()).sum();
            assert_eq!(degree, a.rows());
            if let Some(f) = expected {
                assert_eq!(fs, vec![f]);
            }
        }
        // The previous path (classical Smith form of xI - A), for comparison.
        for n in [6, 8, 10] {
            let a = lcg_dense(n, n as u64);
            let t0 = std::time::Instant::now();
            let old = smith_oracle(&a);
            let t_old = t0.elapsed();
            let t0 = std::time::Instant::now();
            let new = invariant_factors(&a).unwrap();
            let t_new = t0.elapsed();
            assert_eq!(old, new);
            println!("n = {n}: Smith form of xI - A {t_old:?}, Krylov path {t_new:?}");
        }
    }

    /// `m⁻¹`, asserting that it exists.
    fn inv(m: &Matrix<Rational>) -> Matrix<Rational> {
        m.inverse().unwrap().expect("invertible")
    }

    #[test]
    fn rcf_symmetric_c_inverse_has_magma_property() {
        // T-b1. When C is symmetric, T⁻¹ has MAGMA's defining property T_M·A·T_M⁻¹ = F = Cᵀ.
        // gp: A = [1,1;1,0]; matfrobenius(A) = [0, 1; 1, 1], matfrobenius(A,1) = [x^2 - x - 1]
        let a = qm(2, &[1, 1, 1, 0]);
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(rcf.invariant_factors, vec![px(&[-1, -1, 1])]);
        let c = rcf.canonical_matrix.clone();
        assert_eq!(c, qm(2, &[0, 1, 1, 1]));
        assert_eq!(c, c.transpose());
        let tm = inv(rcf.change_of_basis.as_ref().expect("T is computed"));
        assert_eq!(tm.mul(&a).unwrap().mul(&inv(&tm)).unwrap(), c.transpose());

        // A3 = U·[1,0,0;0,0,1;0,1,0]·U⁻¹, U = [1,2,0;0,1,1;0,0,1].
        // gp: matfrobenius(A3,1) = [x^2 - 1, x - 1], matfrobenius(A3) = [0,1,0; 1,0,0; 0,0,1]
        let a3 = qm(3, &[1, -2, 4, 0, 1, 0, 0, 1, -1]);
        let rcf = rational_canonical_form(&a3).unwrap();
        assert_eq!(rcf.invariant_factors, vec![px(&[-1, 1]), px(&[-1, 0, 1])]);
        let c = rcf.canonical_matrix.clone();
        assert_eq!(c, qm(3, &[1, 0, 0, 0, 0, 1, 0, 1, 0]));
        assert_eq!(c, c.transpose());
        let tm = inv(rcf.change_of_basis.as_ref().expect("T is computed"));
        assert_eq!(tm.mul(&a3).unwrap().mul(&inv(&tm)).unwrap(), c.transpose());
        // PARI's Frobenius form lists the blocks in decreasing order, so it differs from C.
        assert_ne!(c, qm(3, &[0, 1, 0, 1, 0, 0, 0, 0, 1]));
    }

    /// The n×n rational matrix with entries `num/den`.
    fn qmr(n: usize, v: &[(i64, i64)]) -> Matrix<Rational> {
        Matrix::from_vec(
            n,
            n,
            v.iter()
                .map(|&(a, b)| Rational::new(a, b).unwrap())
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn rcf_pari_transform_up_to_block_order() {
        // T-b2. PARI: matfrobenius(M,2) = [F_P, B] with M = B^-1*F_P*B (?matfrobenius).
        // A4 = J2(2) + J1(2) + [3]. gp: B4 = matfrobenius(A4,2)[2] = [-6,6,9,4;
        //     5,-7/2,-15/2,-4; -1,1/2,3/2,1; 0,1/2,1/2,0]; PB = B4 with the rows in the order
        //     [4,1,2,3] (Π⁻¹·B4, Π·C·Π⁻¹ = F_P); PB*A4*PB^-1 = [2,0,0,0; 0,0,0,12; 0,1,0,-16;
        //     0,0,1,7] = C
        let a4 = qm(4, &[2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3]);
        let rcf = rational_canonical_form(&a4).unwrap();
        let c = rcf.canonical_matrix.clone();
        assert_eq!(
            c,
            qm(4, &[2, 0, 0, 0, 0, 0, 0, 12, 0, 1, 0, -16, 0, 0, 1, 7])
        );
        let pb = qmr(
            4,
            &[
                (0, 1),
                (1, 2),
                (1, 2),
                (0, 1),
                (-6, 1),
                (6, 1),
                (9, 1),
                (4, 1),
                (5, 1),
                (-7, 2),
                (-15, 2),
                (-4, 1),
                (-1, 1),
                (1, 2),
                (3, 2),
                (1, 1),
            ],
        );
        assert_eq!(pb.mul(&a4).unwrap().mul(&inv(&pb)).unwrap(), c);
        // T⁻¹ has PARI's defining property for C (T⁻¹·A·T = C).
        let t = rcf.change_of_basis.clone().expect("T is computed");
        assert_eq!(inv(&t).mul(&a4).unwrap().mul(&t).unwrap(), c);
        // One block (C = F_P): gp: A = [1,1;1,0]; matfrobenius(A,2) = [[0,1;1,1], [1,-1;0,1]],
        //     B*A*B^-1 == F_P
        let a = qm(2, &[1, 1, 1, 0]);
        let b = qm(2, &[1, -1, 0, 1]);
        let c = rational_canonical_form(&a).unwrap().canonical_matrix;
        assert_eq!(b.mul(&a).unwrap().mul(&inv(&b)).unwrap(), c);
    }

    // ---------------------------------------------------------------------------------------
    // Review follow-ups: the change of basis for small r > 1 (reduction modulo im R, P⁻¹
    // without the monic scalings), the Krylov path over a prime field, and the determinant
    // fallback of the independence certificate.
    // ---------------------------------------------------------------------------------------

    /// CONJDIAG20 = U·diag(1, …, 20)·U⁻¹ with U a product of 80 random elementary integer row
    /// operations (unimodular), row-major; in gp: A = matrix(20, 20, i, j, v[(i-1)*20+j]).
    #[rustfmt::skip]
    const CONJDIAG20: [i64; 400] = [
        9, -56, 0, -22, -66, -96, 14, 30, -14, 70,
        32, -14, 28, 28, 0, 72, 8, -4, -8, 0,
        -20, 322, 64, -892, 414, -1628, -320, -160, 136, -32,
        568, -160, 544, -384, 32, -536, -108, -156, -424, 64,
        -16, 32, -29, 108, 32, 160, -20, 0, -96, -56,
        -56, -4, -36, 8, -16, -32, 0, 0, 12, -32,
        8, 32, 0, -4, 38, -4, -16, -16, 0, -32,
        4, 0, 0, -16, 0, -48, -8, -8, -16, 0,
        8, -360, -32, 402, -443, 618, 232, 184, -96, 248,
        -220, 52, -216, 292, -16, 544, 92, 92, 240, -32,
        0, -88, 0, 0, -110, 6, 44, 44, 0, 88,
        0, 0, 0, 44, 0, 132, 22, 22, 44, 0,
        96, 222, 0, 190, 211, 814, 5, -110, 114, -364,
        -266, 114, -228, -123, 0, -246, 2, 133, 266, 0,
        -62, -202, 0, -226, -212, -660, 31, 104, -59, 316,
        212, -70, 168, 73, 0, 245, 18, -53, -120, 0,
        -30, -120, 0, -60, -120, -360, 0, 60, -51, 180,
        120, -60, 120, 60, 0, 127, 0, -60, -120, 0,
        -12, -40, 32, -490, -31, -1010, -88, 24, 40, 218,
        348, -108, 328, -92, 16, 8, -16, -64, -184, 32,
        12, -172, -16, 227, -226, 496, 140, 84, -20, 88,
        -161, 54, -164, 142, -8, 284, 56, 80, 188, -16,
        -72, -240, 32, -500, -225, -1278, 6, 98, -18, 456,
        426, -112, 302, 80, 16, 252, -8, -150, -326, 32,
        8, -16, 16, -54, -16, -80, 10, 0, 48, 28,
        28, 2, 21, -4, 8, 16, 0, 0, -6, 16,
        -36, 16, 0, 18, 29, 38, -2, -14, 6, -16,
        -18, 6, -12, 6, 0, -24, -4, -4, -8, 0,
        16, 40, 56, -108, 58, -151, -16, -36, 66, -16,
        56, 4, 70, -44, 31, -76, -18, -18, -48, 22,
        -30, -120, 0, -60, -120, -360, 0, 60, -60, 180,
        120, -60, 120, 60, 0, 136, 0, -60, -120, 0,
        232, 188, -32, 694, 84, 2254, 218, -82, 142, -588,
        -744, 312, -792, 74, -16, -60, 87, 309, 702, -32,
        -136, 68, -64, -190, 100, -292, -60, -36, -168, 4,
        76, -26, 84, -62, -32, -80, -6, -10, -62, -64,
        48, -14, 32, 147, -20, 216, 20, 8, 84, -30,
        -64, 13, -42, 21, 16, 10, -2, 9, 39, 32,
        98, 344, 16, 126, 344, 1000, 10, -180, 261, -512,
        -332, 182, -359, -184, 8, -365, 0, 180, 354, 36,
    ];

    #[test]
    fn forms_conjdiag20_q_small_r_rcf_timing() {
        // The ±1 start vector w_0 spans only an 18-dimensional Krylov space here, so the
        // relation matrix R is 2×2 although A has one invariant factor. P⁻¹ of the Smith form
        // of R has entries of degree up to 34 whose coefficients have numerators and
        // denominators of about 200 digits; before the reduction modulo im R and the scaled
        // P⁻¹ (702d0a5), evaluating them at A took rational_canonical_form 30 s in debug
        // (invariant_factors: 0.3 s), and the Krylov Smith path then took 2.1 s. The peeled
        // path (the main path now) needs no P⁻¹: measured 0.17 s in debug.
        // Bound: 15 s, half the old time.
        // gp: matfrobenius(A,1) == [prod(i=1,20,x-i)] (= 1), minpoly(A) == charpoly(A) (= 1)
        let n = 20;
        let a = qm(n, &CONJDIAG20);
        let f = (1..=20).fold(px(&[1]), |acc, i| acc * px(&[-i, 1]));
        assert_eq!(krylov_presentation(&a).unwrap().relations.rows(), 2);
        let start = std::time::Instant::now();
        let rcf = rational_canonical_form(&a).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 15,
            "rational_canonical_form took {elapsed:?}"
        );
        assert_eq!(rcf.invariant_factors, vec![f.clone()]);
        assert_eq!(invariant_factors(&a).unwrap(), vec![f.clone()]);
        // The certificate, re-checked independently of the Krylov elimination.
        assert_eq!(rcf.canonical_matrix, companion_matrix(&f).unwrap());
        let t = rcf.change_of_basis.clone().expect("T is computed");
        assert_eq!(
            a.mul(&t).unwrap(),
            t.mul(&rcf.canonical_matrix).unwrap(),
            "A·T != T·C"
        );
        assert_eq!(t.rank().unwrap(), n, "T singular");
        assert_eq!(f, charpoly_berkowitz(&a).unwrap());
    }

    #[test]
    fn krylov_basis_images_equal_horner_images() {
        // φ(p) = Σ_j p_j(A)·w_j computed two ways: image_in_krylov_basis (reduction modulo im R,
        // then coordinates in the Krylov basis) and image_by_horner (no reduction). Columns:
        // those of X = P⁻¹·D⁻¹ (what rational_canonical_form uses), those of the true P⁻¹, and
        // p_j = x^(2n+j) + j + 1 (degree far above every k_j).
        let mut mats = vec![
            qm(3, &[2, 0, 0, 0, 2, 0, 0, 0, 3]),
            qm(3, &[5, 0, 0, 0, 5, 0, 0, 0, 5]),
            qm(3, &[1, -2, 4, 0, 1, 0, 0, 1, -1]),
            qm(4, &[2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3]),
            qm(4, &[2, 1, -1, 1, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 3]),
            qm(
                5,
                &[
                    2, -5, 0, 1, -2, 1, -2, 0, 0, -1, 1, -2, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 1,
                    0,
                ],
            ),
            qm(12, &DEROG12),
            qm(15, &NILP15),
        ];
        mats.push(lcg_dense(6, 7));
        let mut max_r = 0;
        for a in &mats {
            let n = a.rows();
            let pr = krylov_presentation(a).unwrap();
            let r = pr.relations.rows();
            max_r = max_r.max(r);
            let (z, _, _) = field_constants(a.data().iter());
            let (_, x) = pr.relations.smith_form_with_scaled_left_inverse().unwrap();
            let pinv = pr.relations.smith_normal_form().unwrap().left_inverse;
            let mut cols: Vec<Vec<UnivariatePolynomial<Rational>>> = Vec::new();
            for m in [&x, &pinv] {
                for i in 0..r {
                    cols.push((0..r).map(|j| m.get(j, i).clone()).collect());
                }
            }
            cols.push(
                (0..r)
                    .map(|j| {
                        let mut c = vec![0; 2 * n + j + 1];
                        c[0] = j as i64 + 1;
                        c[2 * n + j] = 1;
                        px(&c)
                    })
                    .collect(),
            );
            for col in cols {
                assert_eq!(
                    image_in_krylov_basis(col.clone(), &pr.relations, &pr.basis, &z).unwrap(),
                    image_by_horner(&col, a, &pr.generators, &z),
                    "A = {a:?}"
                );
            }
        }
        assert!(max_r >= 5, "no presentation with r >= 5 was exercised");
    }

    /// Z with `Field::inverse` defined only for the units ±1: the semantics of a matrix of
    /// unbound `PrimeField` sentinels (rustmath-finitefields), here without the dependency.
    /// `/` is never used by the code under test (it inverts with `Field::inverse`).
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Zs(i64);
    impl std::fmt::Display for Zs {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::ops::Add for Zs {
        type Output = Zs;
        fn add(self, o: Zs) -> Zs {
            Zs(self.0 + o.0)
        }
    }
    impl std::ops::Sub for Zs {
        type Output = Zs;
        fn sub(self, o: Zs) -> Zs {
            Zs(self.0 - o.0)
        }
    }
    impl std::ops::Mul for Zs {
        type Output = Zs;
        fn mul(self, o: Zs) -> Zs {
            Zs(self.0 * o.0)
        }
    }
    impl std::ops::Neg for Zs {
        type Output = Zs;
        fn neg(self) -> Zs {
            Zs(-self.0)
        }
    }
    impl std::ops::Div for Zs {
        type Output = Zs;
        #[allow(clippy::suspicious_arithmetic_impl)] // multiplication by the inverse
        fn div(self, o: Zs) -> Zs {
            self * o.inverse().expect("Zs: division by a non-unit")
        }
    }
    impl Ring for Zs {
        fn zero() -> Zs {
            Zs(0)
        }
        fn one() -> Zs {
            Zs(1)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_one(&self) -> bool {
            self.0 == 1
        }
    }
    impl rustmath_core::CommutativeRing for Zs {}
    impl rustmath_core::IntegralDomain for Zs {}
    impl Field for Zs {
        fn inverse(&self) -> Result<Zs> {
            match self.0 {
                1 | -1 => Ok(*self),
                v => Err(MathError::InvalidArgument(format!(
                    "{v} is not a unit of Z"
                ))),
            }
        }
    }

    /// The square matrix whose columns are `cols` (entries in `Zs`).
    fn zs_columns(cols: &[[i64; 2]]) -> (Matrix<Zs>, Vec<Vec<Zs>>) {
        let vs: Vec<Vec<Zs>> = cols
            .iter()
            .map(|c| c.iter().map(|&v| Zs(v)).collect())
            .collect();
        let n = vs.len();
        let t = Matrix::from_vec(n, n, (0..n * n).map(|k| vs[k % n][k / n]).collect()).unwrap();
        (t, vs)
    }

    #[test]
    fn independence_certificate_falls_back_to_the_determinant() {
        // T = [2, 1; 1, 1] (columns (2, 1) and (1, 1)): det T = 1, so T is unimodular, but the
        // elimination takes the first non-zero entry 2 of the first column as its pivot.
        let (t, cols) = zs_columns(&[[2, 1], [1, 1]]);
        assert!(linearly_independent(&cols).is_err());
        assert!(independent_columns(&t, &cols).unwrap());
        // T = [2, 0; 0, 1]: det 2, independent over Q but not invertible over Z: rejected.
        let (t, cols) = zs_columns(&[[2, 0], [0, 1]]);
        assert!(linearly_independent(&cols).is_err());
        assert!(!independent_columns(&t, &cols).unwrap());
        // T = [2, 4; 1, 2]: det 0: rejected.
        let (t, cols) = zs_columns(&[[2, 1], [4, 2]]);
        assert!(linearly_independent(&cols).is_err());
        assert!(!independent_columns(&t, &cols).unwrap());
        // Unit pivots: decided by the elimination itself.
        let (t, cols) = zs_columns(&[[1, 3], [0, -1]]);
        assert!(linearly_independent(&cols).unwrap());
        assert!(independent_columns(&t, &cols).unwrap());
        let (t, cols) = zs_columns(&[[1, 3], [-1, -3]]);
        assert!(!linearly_independent(&cols).unwrap());
        assert!(!independent_columns(&t, &cols).unwrap());
    }

    /// A test-local prime field GF(P) (P a small prime), so that the Krylov path itself can be
    /// called over a finite field: the `PrimeField` tests in rustmath-finitefields reach only
    /// the public functions, whose classical fallback would mask a Krylov failure.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Gf<const P: u64>(u64);
    impl<const P: u64> Gf<P> {
        fn new(v: i64) -> Self {
            Gf(v.rem_euclid(P as i64) as u64)
        }
    }
    impl<const P: u64> std::fmt::Display for Gf<P> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} mod {P}", self.0)
        }
    }
    impl<const P: u64> std::ops::Add for Gf<P> {
        type Output = Self;
        fn add(self, o: Self) -> Self {
            Gf((self.0 + o.0) % P)
        }
    }
    impl<const P: u64> std::ops::Sub for Gf<P> {
        type Output = Self;
        fn sub(self, o: Self) -> Self {
            Gf((self.0 + P - o.0) % P)
        }
    }
    impl<const P: u64> std::ops::Mul for Gf<P> {
        type Output = Self;
        fn mul(self, o: Self) -> Self {
            Gf(self.0 * o.0 % P)
        }
    }
    impl<const P: u64> std::ops::Neg for Gf<P> {
        type Output = Self;
        fn neg(self) -> Self {
            Gf((P - self.0) % P)
        }
    }
    impl<const P: u64> std::ops::Div for Gf<P> {
        type Output = Self;
        #[allow(clippy::suspicious_arithmetic_impl)] // multiplication by the inverse
        fn div(self, o: Self) -> Self {
            self * o.inverse().expect("Gf: division by zero")
        }
    }
    impl<const P: u64> Ring for Gf<P> {
        fn zero() -> Self {
            Gf(0)
        }
        fn one() -> Self {
            Gf(1 % P)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
        fn is_one(&self) -> bool {
            self.0 == 1
        }
    }
    impl<const P: u64> rustmath_core::CommutativeRing for Gf<P> {}
    impl<const P: u64> rustmath_core::IntegralDomain for Gf<P> {}
    impl<const P: u64> Field for Gf<P> {
        fn inverse(&self) -> Result<Self> {
            if self.0 == 0 {
                return Err(MathError::DivisionByZero);
            }
            // Fermat: a^(P-2).
            let mut r = Gf(1);
            for _ in 0..P - 2 {
                r = r * *self;
            }
            Ok(r)
        }
    }

    /// The Krylov path itself over GF(P) on the integer matrix `v` reduced mod P: the factors
    /// (`expected`, coefficients low to high, increasing order), `C`, the certified `T`
    /// (re-checked with `Matrix::inverse`), the product against `charpoly_berkowitz`, and
    /// r ≥ the number of factors. Returns r.
    fn check_krylov_gfp<const P: u64>(n: usize, v: &[i64], expected: &[&[i64]]) -> usize {
        let a = Matrix::from_vec(n, n, v.iter().map(|&x| Gf::<P>::new(x)).collect()).unwrap();
        let fs: Vec<UnivariatePolynomial<Gf<P>>> = expected
            .iter()
            .map(|c| UnivariatePolynomial::new(c.iter().map(|&x| Gf::<P>::new(x)).collect()))
            .collect();
        assert_eq!(
            invariant_factors_krylov(&a).unwrap(),
            fs,
            "P = {P}, A = {v:?}"
        );
        assert_eq!(
            invariant_factors_peeled(&a).unwrap(),
            fs,
            "P = {P}, A = {v:?}"
        );
        assert_eq!(
            &minimal_polynomial_peeled(&a).unwrap(),
            fs.last().unwrap(),
            "P = {P}, A = {v:?}"
        );
        let peeled = rational_canonical_form_peeled(&a).unwrap();
        let rcf = rational_canonical_form_krylov(&a).unwrap();
        assert_eq!(rcf.invariant_factors, fs, "P = {P}, A = {v:?}");
        assert_eq!(peeled.invariant_factors, fs, "P = {P}, A = {v:?}");
        assert_eq!(peeled.canonical_matrix, rcf.canonical_matrix, "P = {P}");
        let tp = peeled.change_of_basis.expect("T is computed");
        assert_eq!(
            a.mul(&tp).unwrap(),
            tp.mul(&peeled.canonical_matrix).unwrap(),
            "P = {P}: peeled A·T != T·C"
        );
        assert!(
            tp.inverse().unwrap().is_some(),
            "P = {P}: peeled T singular"
        );
        let mut c = Matrix::<Gf<P>>::zeros(n, n);
        let mut off = 0;
        for f in &fs {
            let b = companion_matrix(f).unwrap();
            for i in 0..b.rows() {
                for j in 0..b.cols() {
                    c.set(off + i, off + j, *b.get(i, j).unwrap()).unwrap();
                }
            }
            off += b.rows();
        }
        assert_eq!(off, n);
        assert_eq!(rcf.canonical_matrix, c, "P = {P}, A = {v:?}");
        let t = rcf.change_of_basis.clone().expect("T is computed");
        assert_eq!(
            a.mul(&t).unwrap(),
            t.mul(&c).unwrap(),
            "P = {P}: A·T != T·C"
        );
        assert!(t.inverse().unwrap().is_some(), "P = {P}: T singular");
        let product = fs.iter().fold(
            UnivariatePolynomial::new(vec![Gf::<P>::new(1)]),
            |acc, f| acc * f.clone(),
        );
        assert_eq!(product, charpoly_berkowitz(&a).unwrap(), "P = {P}");
        let r = krylov_presentation(&a).unwrap().relations.rows();
        assert!(r >= fs.len(), "P = {P}: r = {r} < {} factors", fs.len());
        r
    }

    #[test]
    fn krylov_path_over_prime_field_matches_pari() {
        // gp, for each (A, p): Vecrev of each entry of Vecrev(lift(matfrobenius(A*Mod(1,p),1)))
        // (increasing order, coefficients low to high), and lift(minpoly(A*Mod(1,p))) equal
        // to the last entry. dense12 mod 7: [[6,1,0,6,5,5,1,3,4,0,6,4,1]]. Here r = 2 with a
        // single invariant factor (the start vector w_0 alone does not span GF(7)^12), so the
        // change of basis goes through the reduction modulo im R of a non-trivial P⁻¹ column.
        assert_eq!(
            check_krylov_gfp::<7>(12, &DENSE12, &[&[6, 1, 0, 6, 5, 5, 1, 3, 4, 0, 6, 4, 1]]),
            2
        );
        // uptri14 mod 2: [[0,1,1], [0,0,1,0,1], [0,0,0,0,1,0,0,0,1]]
        check_krylov_gfp::<2>(
            14,
            &UPTRI14,
            &[&[0, 1, 1], &[0, 0, 1, 0, 1], &[0, 0, 0, 0, 1, 0, 0, 0, 1]],
        );
        // uptri14 mod 3: [[0,2,1], [0,2,0,1], [0,0,2,2,0,0,0,0,1,1]]
        check_krylov_gfp::<3>(
            14,
            &UPTRI14,
            &[&[0, 2, 1], &[0, 2, 0, 1], &[0, 0, 2, 2, 0, 0, 0, 0, 1, 1]],
        );
        // derog12 mod 2: [[0,1,1,0,1]] x 3; derog12 mod 3: [[1,2,1,1,1]] x 3
        let f2: &[i64] = &[0, 1, 1, 0, 1];
        check_krylov_gfp::<2>(12, &DEROG12, &[f2, f2, f2]);
        let f3: &[i64] = &[1, 2, 1, 1, 1];
        check_krylov_gfp::<3>(12, &DEROG12, &[f3, f3, f3]);
        // nilp15 mod 2: [[0,1], [0,0,1], [0,0,0,1], [0,0,0,0,1], [0,0,0,0,0,1]]
        assert!(
            check_krylov_gfp::<2>(
                15,
                &NILP15,
                &[
                    &[0, 1],
                    &[0, 0, 1],
                    &[0, 0, 0, 1],
                    &[0, 0, 0, 0, 1],
                    &[0, 0, 0, 0, 0, 1]
                ],
            ) >= 5
        );
        // gf5d = V5*(C(x^2+2) + C((x^2+2)(x+1)))*V5^-1 (rustmath-finitefields
        // rational_form_gf5_derogatory) mod 5: [[2,0,1], [2,2,1,1]]
        check_krylov_gfp::<5>(
            5,
            &[
                0, 3, 0, 0, 2, 4, 0, 2, 3, 2, 0, 0, 0, 0, 3, 0, 2, 1, 0, 2, 1, 0, 3, 1, 4,
            ],
            &[&[2, 0, 1], &[2, 2, 1, 1]],
        );
        // [0,0;2,0] mod 2: [[0,1], [0,1]]; mod 3: [[0,0,1]]
        check_krylov_gfp::<2>(2, &[0, 0, 2, 0], &[&[0, 1], &[0, 1]]);
        check_krylov_gfp::<3>(2, &[0, 0, 2, 0], &[&[0, 0, 1]]);
    }

    /// 64 deterministic matrices over GF(P), n = 1..8: dense, sparse, scalar plus sparse, and
    /// block diagonal with repeated blocks (derogatory, r > 1 for small P), with their integer
    /// entries.
    fn gfp_matrices<const P: u64>(seed: u64) -> Vec<(Vec<i64>, Matrix<Gf<P>>)> {
        let mut state = seed;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let mut out = Vec::with_capacity(64);
        for t in 0..64usize {
            let n = 1 + t % 8;
            let kind = (t / 8) % 4;
            let mut v = vec![0i64; n * n];
            let blk = 1 + n / 3;
            for i in 0..n {
                for j in 0..n {
                    let x = next(P) as i64;
                    v[i * n + j] = match kind {
                        0 => x,
                        1 => i64::from(next(4) == 0) * x,
                        2 => i64::from(i == j) * 2 + i64::from(next(5) == 0) * x,
                        // the same block on the diagonal, repeated
                        _ if i / blk == j / blk => v[(i % blk) * n + (j % blk)],
                        _ => 0,
                    };
                    if kind == 3 && i < blk && j < blk {
                        v[i * n + j] = x;
                    }
                }
            }
            let a = Matrix::from_vec(n, n, v.iter().map(|&x| Gf::<P>::new(x)).collect()).unwrap();
            out.push((v, a));
        }
        out
    }

    /// The Krylov path and the peeled path over GF(P) against the classical path (Smith form
    /// of `xI - A`) on the matrices of [`gfp_matrices`]. Returns the largest r seen.
    fn krylov_vs_classical_gfp<const P: u64>(seed: u64) -> usize {
        let mut max_r = 0;
        for (v, a) in gfp_matrices::<P>(seed) {
            max_r = max_r.max(krylov_presentation(&a).unwrap().relations.rows());
            let expected = invariant_factors_classical(&a).unwrap();
            assert_eq!(
                invariant_factors_krylov(&a).unwrap(),
                expected,
                "P = {P}, A = {v:?}"
            );
            assert_eq!(
                invariant_factors_peeled(&a).unwrap(),
                expected,
                "P = {P}, A = {v:?}"
            );
            let krylov = rational_canonical_form_krylov(&a).unwrap();
            let peeled = rational_canonical_form_peeled(&a).unwrap();
            for rcf in [&krylov, &peeled] {
                assert_eq!(rcf.invariant_factors, expected, "P = {P}, A = {v:?}");
                let t = rcf.change_of_basis.clone().expect("T is computed");
                assert_eq!(
                    a.mul(&t).unwrap(),
                    t.mul(&rcf.canonical_matrix).unwrap(),
                    "P = {P}, A = {v:?}"
                );
                assert!(t.inverse().unwrap().is_some(), "P = {P}, A = {v:?}");
            }
        }
        max_r
    }

    #[test]
    fn krylov_path_over_prime_field_agrees_with_classical() {
        assert!(krylov_vs_classical_gfp::<2>(11) >= 3);
        assert!(krylov_vs_classical_gfp::<3>(12) >= 3);
        assert!(krylov_vs_classical_gfp::<5>(13) >= 2);
        assert!(krylov_vs_classical_gfp::<7>(14) >= 2);
    }

    // ---------------------------------------------------------------------------------------
    // The peeled path (frobenius.rs): the inputs on which the Smith form of the Krylov
    // relation matrix had no bound, multi-level peeling, exact fields, Z semantics, and the
    // handbook example H26E9.
    // ---------------------------------------------------------------------------------------

    /// CONJDIAG20_S7 = U·diag(1, …, 20)·U⁻¹, U a product of 80 elementary integer row
    /// operations (seed 7 of the CONJDIAG20 generator), row-major; in gp:
    /// A = matrix(20, 20, i, j, v[(i-1)*20+j]).
    #[rustfmt::skip]
    const CONJDIAG20_S7: [i64; 400] = [
        -90, -140, -74, -2706, -428, -336, 98, -598, 817, 0,
        482, -214, 554, 72, 468, 0, 817, 586, -533, -1208,
        0, -20, -1, -30, -6, -2, 0, -6, 12, 0,
        46, -3, 28, 0, 2, 0, 12, 2, -39, -6,
        0, 0, 57, 438, 54, 106, 0, 110, -108, 0,
        -92, 27, -110, -56, -78, 0, -108, -72, 63, 190,
        0, 0, -21, 424, 46, 8, 0, 74, -148, 0,
        -74, 23, -74, 0, -8, 0, -148, -58, 1, 74,
        0, 0, 37, -421, -73, 210, -44, -94, 224, 0,
        94, -39, 94, -20, -122, 0, 232, 41, -36, -8,
        -26, -52, 3, -1098, -154, -72, 28, -226, 358, 0,
        217, -77, 226, 18, 112, 0, 358, 202, -151, -370,
        -52, -104, -36, -1362, -216, -142, 63, -304, 420, 0,
        286, -108, 304, 36, 208, 0, 420, 288, -300, -592,
        0, -33, 25, -552, -66, 0, 0, -86, 206, 0,
        139, -33, 121, 0, 0, 0, 206, 72, -34, -94,
        -2, -4, -8, 4, 16, -80, 20, 10, -21, 0,
        -10, 8, -10, 8, 48, 0, -30, 2, 12, -20,
        -18, -33, -58, 500, 98, -384, 100, 120, -312, 10,
        -123, 44, -123, 22, 256, 0, -328, -4, 1, -108,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -104, 1256, 218, -486, 88, 272, -616, 0,
        -272, 114, -272, 40, 310, 0, -632, -130, 71, 52,
        0, -33, -1, -50, -10, -2, 0, -10, 20, 0,
        52, -5, 45, 0, 2, 0, 20, 4, -33, -10,
        -13, -26, 5, -49, -10, 60, 14, -16, 35, 0,
        16, -5, 16, 1, -36, 0, 35, 10, -44, -10,
        -26, -52, -8, -829, -122, -59, 28, -180, 266, 0,
        171, -61, 180, 18, 99, 0, 266, 161, -150, -310,
        0, 0, 0, 2, 0, 0, 0, 0, -1, 0,
        0, 0, 0, 0, 0, 16, -1, 0, 0, 0,
        2, 4, -34, 810, 76, 96, -20, 138, -258, 0,
        -138, 38, -138, -8, -64, 0, -249, -118, -10, 168,
        0, 0, -76, 190, 28, -74, 0, 0, -112, 0,
        -18, 14, 0, 56, 46, 0, -112, -16, -61, -52,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 19, 0,
        0, 0, -28, -42, 0, -14, 0, -28, 0, 0,
        19, 0, 28, 28, 0, 0, 0, 0, -30, -22,
    ];

    /// `W·B·W⁻¹`, W the start basis of the Krylov presentation (columns `w_j`): spinning `w_j`
    /// under it is spinning `e_j` under B, so an upper triangular B gives r = n and `R = xI - B`.
    fn conj_start_basis<F: Field>(b: &Matrix<F>) -> Matrix<F> {
        let n = b.rows();
        let k = field_constants(b.data().iter());
        let ws = crate::krylov::start_basis(n, &k);
        let w = Matrix::from_vec(
            n,
            n,
            (0..n * n).map(|idx| ws[idx % n][idx / n].clone()).collect(),
        )
        .unwrap();
        let winv = w.inverse().unwrap().expect("W is unimodular");
        w.mul(b).unwrap().mul(&winv).unwrap()
    }

    /// B upper triangular, n×n, row-major: `B[i][i] = diag(i)`, `B[i][j] = ((3i + 5j) mod 7) - 3`
    /// for j > i (0-indexed; the B of `krylov::tests::fixed_start_basis_worst_case_has_r_equal_n`).
    fn upper_b(n: usize, diag: impl Fn(usize) -> i64) -> Vec<i64> {
        let mut b = vec![0i64; n * n];
        for i in 0..n {
            b[i * n + i] = diag(i);
            for j in (i + 1)..n {
                b[i * n + j] = ((3 * i + 5 * j) % 7) as i64 - 3;
            }
        }
        b
    }

    /// `Π_{i=1}^{n} (x - i)` over Q.
    fn prod_x_minus_i(n: i64) -> UnivariatePolynomial<Rational> {
        (1..=n).fold(px(&[1]), |acc, i| acc * px(&[-i, 1]))
    }

    /// U, n×n, a product of `4n` elementary integer row operations `row_i += c·row_j` with
    /// `c ∈ [-2, 2]` (64-bit LCG, MMIX constants; unimodular).
    fn lcg_unimodular(n: usize, seed: u64) -> Matrix<Rational> {
        let mut state = seed * 1000 + 7;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let mut u = Matrix::<Rational>::identity(n);
        for _ in 0..4 * n {
            let i = next(n as u64) as usize;
            let mut j = next(n as u64) as usize;
            if i == j {
                j = (i + 1) % n;
            }
            let c = Rational::from_integer(next(5) as i64 - 2);
            for t in 0..n {
                let v = u.get(i, t).unwrap().clone() + c.clone() * u.get(j, t).unwrap().clone();
                u.set(i, t, v).unwrap();
            }
        }
        u
    }

    /// `U·M·U⁻¹`.
    fn conj(u: &Matrix<Rational>, m: &Matrix<Rational>) -> Matrix<Rational> {
        u.mul(m).unwrap().mul(&inv(u)).unwrap()
    }

    #[test]
    fn peeled_forms_conjdiag20_s7_was_81_s() {
        // The HANDOFF case: the relation matrix R is 2×2 with diagonal degrees [13, 7], and the
        // Smith form of R took invariant_factors 81 to 102 s and rational_canonical_form about
        // 170 s in debug at c79a582 (depending on load). The peeled path needs no Smith form.
        // gp (v = CONJDIAG20_S7, A = matrix(20,20,i,j,v[(i-1)*20+j])):
        //     matfrobenius(A,1) == [prod(i=1,20,x-i)] (1), minpoly(A) == charpoly(A) (1),
        //     matdet(A) == 20! (1)
        let a = qm(20, &CONJDIAG20_S7);
        let pr = krylov_presentation(&a).unwrap();
        assert_eq!(pr.relations.rows(), 2);
        let degrees: Vec<usize> = (0..2)
            .map(|j| pr.relations.get(j, j).degree().unwrap())
            .collect();
        assert_eq!(degrees, vec![13, 7]);
        let f = prod_x_minus_i(20);
        assert_eq!(invariant_factors_peeled(&a).unwrap(), vec![f.clone()]);
        // The three public calls: measured 0.3 s in debug.
        check_forms_bounded(&a, &[f], 20);
    }

    #[test]
    fn peeled_forms_fixed_start_basis_worst_case_n20() {
        // The documented worst case of the fixed start basis at n = 20: r = 20 and R = xI - B.
        // At c79a582 invariant_factors and rational_canonical_form did not finish in 300 s
        // (n = 14: 95 s for invariant_factors).
        // gp: B = matrix(20,20,i,j, if(i==j, i, if(j>i, ((3*(i-1)+5*(j-1))%7)-3, 0)));
        //     matfrobenius(B,1) == [prod(i=1,20,x-i)] (1)
        let n = 20;
        let b = upper_b(n, |i| i as i64 + 1);
        let a = conj_start_basis(&qm(n, &b));
        let pr = krylov_presentation(&a).unwrap();
        assert_eq!(pr.relations.rows(), n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    px(&[-b[i * n + j], 1])
                } else {
                    px(&[-b[i * n + j]])
                };
                assert_eq!(
                    pr.relations.get(i, j),
                    &expected,
                    "R != xI - B at ({i}, {j})"
                );
            }
        }
        // The three public calls: measured 0.2 s in debug.
        check_forms_bounded(&a, &[prod_x_minus_i(20)], 20);
    }

    #[test]
    fn peeled_forms_r_equals_n_repeated_eigenvalues() {
        // r = n = 20 with two invariant factors (diagonal (i mod 3) + 1).
        // gp: B = matrix(20,20,i,j, if(i==j, (i-1)%3+1, if(j>i, ((3*(i-1)+5*(j-1))%7)-3, 0)));
        //     matfrobenius(B,1) == [(x-1)^7*(x-2)^7*(x-3)^5, x-3] (1)
        let n = 20;
        let a = conj_start_basis(&qm(n, &upper_b(n, |i| (i % 3) as i64 + 1)));
        assert_eq!(krylov_presentation(&a).unwrap().relations.rows(), n);
        let pow = |f: UnivariatePolynomial<Rational>, e: usize| {
            (0..e).fold(px(&[1]), |acc, _| acc * f.clone())
        };
        let big = pow(px(&[-1, 1]), 7) * pow(px(&[-2, 1]), 7) * pow(px(&[-3, 1]), 5);
        // The three public calls: measured 0.6 s in debug.
        check_forms_bounded(&a, &[px(&[-3, 1]), big], 20);
    }

    #[test]
    fn peeled_merges_arrow_n20() {
        // A = W·B·W⁻¹, B[i][i] = i + 1, B[0][j] = (5j mod 7) - 3 (j ≥ 1), other entries 0:
        // r = n, and the last generator (spun first) has a minimal polynomial of degree 2, so
        // merges must run. At c79a582 invariant_factors took 0.04 s and rational_canonical_form
        // 21.4 s in debug (measured under load, about 2× the bound below: a fast machine could
        // pass this test at c79a582, so a pass alone does not show the fix).
        // gp: B = matrix(20,20,i,j, if(i==j, i, if(i==1 && j>1, ((5*(j-1))%7)-3, 0)));
        //     matfrobenius(B,1) == [prod(i=1,20,x-i)] (1)
        let n = 20;
        let mut b = vec![0i64; n * n];
        for i in 0..n {
            b[i * n + i] = i as i64 + 1;
        }
        // Row 0: B[0][j] for j ≥ 1.
        for (j, b0j) in b.iter_mut().enumerate().take(n).skip(1) {
            *b0j = ((5 * j) % 7) as i64 - 3;
        }
        let a = conj_start_basis(&qm(n, &b));
        let pr = krylov_presentation(&a).unwrap();
        assert_eq!(pr.relations.rows(), n);
        let z = Rational::from_integer(0);
        let w = pr.generators[n - 1].clone();
        let aw = mat_vec(&a, &w, &z);
        let a2w = mat_vec(&a, &aw, &z);
        assert!(linearly_independent(&[w.clone(), aw.clone()]).unwrap());
        assert!(!linearly_independent(&[w, aw, a2w]).unwrap());
        // The three public calls: measured 0.7 s in debug.
        check_forms_bounded(&a, &[prod_x_minus_i(20)], 10);
    }

    #[test]
    fn peeled_multilevel_families() {
        // Multi-level peeling: 7, 7 and 40 levels.
        // gp: D = matconcat(matdiagonal(vector(14,k, if(k%2, matcompanion(x^3+x+1),
        //         matcompanion(x-2))))); matfrobenius(D,1) == vector(7,k,x^4-2*x^3+x^2-x-2) (1)
        //     N = matconcat(matdiagonal(vector(7,k,matcompanion(x^k))));
        //         matfrobenius(N,1) == vector(7,k,x^(8-k)) (1)
        //     matfrobenius(3*matid(40),1) == vector(40,k,x-3) (1)
        // (x^3+x+1)(x-2) = x^4 - 2x^3 + x^2 - x - 2.
        let block_diag = |blocks: &[UnivariatePolynomial<Rational>]| {
            let n: usize = blocks.iter().map(|f| f.degree().unwrap()).sum();
            block_companion(blocks, n)
        };
        let mut derog = Vec::new();
        for _ in 0..7 {
            derog.push(px(&[1, 1, 0, 1]));
            derog.push(px(&[-2, 1]));
        }
        let d = block_diag(&derog);
        let a = conj(&lcg_unimodular(28, 3), &d);
        let f = px(&[-2, -1, 1, -2, 1]);
        check_forms_bounded(&a, &vec![f; 7], 60);
        let nilp: Vec<_> = (1..=7usize)
            .map(|k| {
                let mut c = vec![0; k + 1];
                c[k] = 1;
                px(&c)
            })
            .collect();
        let a = conj(&lcg_unimodular(28, 5), &block_diag(&nilp));
        assert!(krylov_presentation(&a).unwrap().relations.rows() >= 7);
        check_forms_bounded(&a, &nilp, 60);
        let mut s = Matrix::<Rational>::zeros(40, 40);
        for i in 0..40 {
            s.set(i, i, Rational::from_integer(3)).unwrap();
        }
        check_forms_bounded(&s, &vec![px(&[-3, 1]); 40], 60);
    }

    #[test]
    fn peeled_path_over_prime_field_r_equals_n() {
        // The fixed-start-basis worst case over GF(101) at n = 60 (r = 60).
        // gp: B = matrix(60,60,i,j, if(i==j, i, if(j>i, ((3*(i-1)+5*(j-1))%7)-3, 0)))*Mod(1,101);
        //     matfrobenius(B,1) == [Mod(1,101)*prod(i=1,60,x-i)] (1)
        let n = 60;
        let b = upper_b(n, |i| i as i64 + 1);
        let bm = Matrix::from_vec(n, n, b.iter().map(|&x| Gf::<101>::new(x)).collect()).unwrap();
        let a = conj_start_basis(&bm);
        assert_eq!(krylov_presentation(&a).unwrap().relations.rows(), n);
        let f = (1..=60).fold(
            UnivariatePolynomial::new(vec![Gf::<101>::new(1)]),
            |acc, i| acc * UnivariatePolynomial::new(vec![Gf::<101>::new(-i), Gf::<101>::new(1)]),
        );
        let start = std::time::Instant::now();
        assert_eq!(invariant_factors_peeled(&a).unwrap(), vec![f.clone()]);
        assert_eq!(minimal_polynomial(&a).unwrap(), f);
        let rcf = rational_canonical_form(&a).unwrap();
        // Measured: 0.05 s in debug (the Smith form of R: 0.26 s for the factors alone).
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 30, "took {elapsed:?}");
        assert_eq!(rcf.invariant_factors, vec![f.clone()]);
        assert_eq!(rcf.canonical_matrix, companion_matrix(&f).unwrap());
        let t = rcf.change_of_basis.expect("T is computed");
        assert_eq!(a.mul(&t).unwrap(), t.mul(&rcf.canonical_matrix).unwrap());
        assert!(t.inverse().unwrap().is_some(), "T singular");
    }

    /// `peel` at both depths is Ok over an exact field and agrees with `expected`, and the
    /// clean generators give the certified form.
    fn check_peel_exact<F: Field>(a: &Matrix<F>, expected: &[UnivariatePolynomial<F>]) {
        let full = peel(a, PeelDepth::Full).unwrap();
        assert_eq!(full.invariant_factors(), expected);
        assert_eq!(full.minimal_polynomial().as_ref(), expected.last());
        let first = peel(a, PeelDepth::MinimalPolynomial).unwrap();
        assert_eq!(first.minimal_polynomial().as_ref(), expected.last());
        let gens = full.generators(a).unwrap();
        let rcf = certified_form(a, full.invariant_factors(), gens).unwrap();
        assert_eq!(rcf.invariant_factors, expected);
    }

    #[test]
    fn peel_exact_fields_never_fall_back() {
        // Over an exact field the first stage answers (no fallback): the 69 matrices of
        // invariant_factors_agree_with_smith_of_xi_minus_a over Q, and the 4×64 matrices of
        // krylov_vs_classical_gfp, against the classical Smith form of xI - A.
        for a in agreement_matrices() {
            check_peel_exact(&a, &smith_oracle(&a));
        }
        fn over<const P: u64>(seed: u64) {
            for (_, a) in gfp_matrices::<P>(seed) {
                check_peel_exact(&a, &invariant_factors_classical(&a).unwrap());
            }
        }
        over::<2>(11);
        over::<3>(12);
        over::<5>(13);
        over::<7>(14);
    }

    #[test]
    fn peeled_generators_are_clean() {
        // (F7): each clean generator g_ℓ has f_ℓ(A)·g_ℓ = 0, and the cyclic bases
        // g, A·g, …, A^{k-1}·g of all levels together are independent. Inputs with several
        // levels: DEROG12 (3 levels), NILP15 (5), and the 5×5 of
        // forms_r8_derogatory_irreducible_quadratic (2).
        let mats = [
            qm(12, &DEROG12),
            qm(15, &NILP15),
            qm(
                5,
                &[
                    2, -5, 0, 1, -2, 1, -2, 0, 0, -1, 1, -2, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 1,
                    0,
                ],
            ),
        ];
        let z = Rational::from_integer(0);
        for (a, levels) in mats.iter().zip([3, 5, 2]) {
            let n = a.rows();
            let p = peel(a, PeelDepth::Full).unwrap();
            let fs = p.invariant_factors();
            assert_eq!(fs.len(), levels);
            let gs = p.generators(a).unwrap();
            assert_eq!(gs.len(), levels);
            let mut cols = Vec::new();
            for (f, g) in fs.iter().zip(&gs) {
                assert!(
                    poly_mat_vec(f, a, g, &z).iter().all(|x| x.is_zero()),
                    "f(A)·g != 0 for f = {f}"
                );
                let mut v = g.clone();
                for _ in 0..f.degree().unwrap() {
                    let next = mat_vec(a, &v, &z);
                    cols.push(std::mem::replace(&mut v, next));
                }
            }
            assert_eq!(cols.len(), n);
            assert!(linearly_independent(&cols).unwrap());
        }
    }

    #[test]
    fn h26e9_invariant_factors_and_rational_form() {
        // MAGMA Handbook H26E9 (V2.x H51E8), over GF(5):
        //   a = [0 2 4 2 0; 2 2 2 3 3; 3 4 4 1 3; 0 0 0 0 1; 0 0 0 1 0];
        //   r, t, f := RationalForm(a); r = [1 0 0 0 0; 0 0 1 0 0; 0 1 0 0 0; 0 0 0 0 1;
        //   0 0 0 1 0], t = [1 3 0 2 1; 2 1 2 2 0; 3 4 3 4 1; 1 0 0 0 0; 0 2 4 2 0],
        //   f = [x + 4, x^2 + 4, x^2 + 4].
        // gp: a = Mod([0,2,4,2,0; 2,2,2,3,3; 3,4,4,1,3; 0,0,0,0,1; 0,0,0,1,0], 5);
        //     matfrobenius(a,1) = [x^2 + Mod(4,5), x^2 + Mod(4,5), x + Mod(4,5)],
        //     t*a*t^-1 == r (1, MAGMA's row convention).
        let g5 = |v: &[i64]| Matrix::from_vec(5, 5, v.iter().map(|&x| Gf::<5>::new(x)).collect());
        let a = g5(&[
            0, 2, 4, 2, 0, 2, 2, 2, 3, 3, 3, 4, 4, 1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        ])
        .unwrap();
        let r = g5(&[
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        ])
        .unwrap();
        let t = g5(&[
            1, 3, 0, 2, 1, 2, 1, 2, 2, 0, 3, 4, 3, 4, 1, 1, 0, 0, 0, 0, 0, 2, 4, 2, 0,
        ])
        .unwrap();
        let gp5 =
            |c: &[i64]| UnivariatePolynomial::new(c.iter().map(|&x| Gf::<5>::new(x)).collect());
        let f = vec![gp5(&[4, 1]), gp5(&[4, 0, 1]), gp5(&[4, 0, 1])];
        assert_eq!(invariant_factors(&a).unwrap(), f);
        assert_eq!(invariant_factors_peeled(&a).unwrap(), f);
        assert_eq!(minimal_polynomial(&a).unwrap(), gp5(&[4, 0, 1]));
        assert!(krylov_presentation(&a).unwrap().relations.rows() >= 3);
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(rcf.invariant_factors, f);
        // C is the block companion matrix of MAGMA's f; it is symmetric here, so it equals
        // MAGMA's r (= Cᵀ).
        assert_eq!(rcf.canonical_matrix, r);
        let tt = rcf.change_of_basis.expect("T is computed");
        assert_eq!(a.mul(&tt).unwrap(), tt.mul(&r).unwrap(), "A·T != T·C");
        assert!(tt.inverse().unwrap().is_some(), "T singular");
        // MAGMA's own transform satisfies t·a = r·t in our arithmetic.
        assert_eq!(t.mul(&a).unwrap(), r.mul(&t).unwrap());
    }

    /// The natural sentinel-only families of rustmath-finitefields (`tests/matrix_forms_gfp.rs`,
    /// `sentinel_families`) of size n: cyclic shift, reversal, transposition (1 2), all-ones,
    /// upper and lower all-ones, nilpotent shift, `J_n(1)` and lower `J_n(-1)`.
    fn sentinel_families(n: usize) -> Vec<Vec<i64>> {
        let mut fams = Vec::new();
        let mut v = vec![0; n * n];
        for i in 0..n {
            v[i * n + (i + 1) % n] = 1;
        }
        fams.push(v);
        let mut v = vec![0; n * n];
        for i in 0..n {
            v[i * n + n - 1 - i] = 1;
        }
        fams.push(v);
        let mut v = vec![0; n * n];
        for i in 2..n {
            v[i * n + i] = 1;
        }
        v[1] = 1;
        v[n] = 1;
        fams.push(v);
        fams.push(vec![1; n * n]);
        fams.push((0..n * n).map(|k| i64::from(k / n <= k % n)).collect());
        fams.push((0..n * n).map(|k| i64::from(k / n >= k % n)).collect());
        let mut v = vec![0; n * n];
        for i in 0..n - 1 {
            v[i * n + i + 1] = 1;
        }
        fams.push(v);
        let mut v = vec![0; n * n];
        for i in 0..n {
            v[i * n + i] = 1;
            if i + 1 < n {
                v[i * n + i + 1] = 1;
            }
        }
        fams.push(v);
        let mut v = vec![0; n * n];
        for i in 0..n {
            v[i * n + i] = -1;
            if i + 1 < n {
                v[(i + 1) * n + i] = 1;
            }
        }
        fams.push(v);
        fams
    }

    fn zs_matrix(n: usize, v: &[i64]) -> Matrix<Zs> {
        Matrix::from_vec(n, n, v.iter().map(|&x| Zs(x)).collect()).unwrap()
    }

    fn zs_poly(c: &[i64]) -> UnivariatePolynomial<Zs> {
        UnivariatePolynomial::new(c.iter().map(|&x| Zs(x)).collect())
    }

    /// The Z-semantics results reduced mod P against the bound GF(P) results of the same integer
    /// matrix (the classical Smith form of xI - A over GF(P) as the oracle).
    fn zs_results_reduce_mod<const P: u64>(
        n: usize,
        v: &[i64],
        factors: Option<&[UnivariatePolynomial<Zs>]>,
        minpoly: Option<&UnivariatePolynomial<Zs>>,
    ) {
        let reduce = |f: &UnivariatePolynomial<Zs>| {
            UnivariatePolynomial::new(f.coefficients().iter().map(|c| Gf::<P>::new(c.0)).collect())
        };
        let a = Matrix::from_vec(n, n, v.iter().map(|&x| Gf::<P>::new(x)).collect()).unwrap();
        let expected = invariant_factors_classical(&a).unwrap();
        if let Some(fs) = factors {
            let got: Vec<_> = fs.iter().map(reduce).collect();
            assert_eq!(got, expected, "P = {P}, A = {v:?}");
        }
        if let Some(mp) = minpoly {
            assert_eq!(Some(&reduce(mp)), expected.last(), "P = {P}, A = {v:?}");
        }
    }

    /// Runs the peeled path on the sentinel-like `Zs` matrix `v` (Z semantics) and checks every
    /// answer against GF(2), GF(3), GF(5), GF(7); a rational form must have `A·T == T·C` and
    /// `det T = ±1`. Returns which of (invariant factors, minimal polynomial, rational form)
    /// answered.
    fn check_zs_peeled(n: usize, v: &[i64]) -> [bool; 3] {
        let a = zs_matrix(n, v);
        let fs = invariant_factors_peeled(&a).ok();
        let mp = minimal_polynomial_peeled(&a).ok();
        if let (Some(fs), Some(mp)) = (&fs, &mp) {
            assert_eq!(fs.last(), Some(mp), "A = {v:?}");
        }
        zs_results_reduce_mod::<2>(n, v, fs.as_deref(), mp.as_ref());
        zs_results_reduce_mod::<3>(n, v, fs.as_deref(), mp.as_ref());
        zs_results_reduce_mod::<5>(n, v, fs.as_deref(), mp.as_ref());
        zs_results_reduce_mod::<7>(n, v, fs.as_deref(), mp.as_ref());
        let rcf = rational_canonical_form_peeled(&a).ok();
        if let Some(rcf) = &rcf {
            assert_eq!(Some(&rcf.invariant_factors), fs.as_ref(), "A = {v:?}");
            let t = rcf.change_of_basis.clone().expect("T is computed");
            assert_eq!(a.mul(&t).unwrap(), t.mul(&rcf.canonical_matrix).unwrap());
            let det = charpoly_berkowitz(&t).unwrap().coeff(0).0;
            assert!(det == 1 || det == -1, "A = {v:?}: det T = {det}");
        }
        [fs.is_some(), mp.is_some(), rcf.is_some()]
    }

    #[test]
    fn peeled_sentinel_results_are_the_results_mod_every_p() {
        // frobenius.rs, "Exactness": in Z semantics (Zs: only ±1 invertible, the semantics of a
        // matrix of unbound PrimeField sentinels) a peeled result is the one over Z, and
        // reduced mod p it is the one over GF(p). Checked on the 45 natural families
        // n = 2..6 and on 600 deterministic matrices with entries in {-1, 0, 1}, n = 2..6,
        // against the bound GF(2), GF(3), GF(5), GF(7) results (gp equivalent:
        // matfrobenius(A*Mod(1,p),1) == Mod(1,p)*matfrobenius(A,1) whenever we answer).
        let mut answered = [0usize; 3];
        for n in 2..=6usize {
            for v in sentinel_families(n) {
                for (k, ok) in check_zs_peeled(n, &v).into_iter().enumerate() {
                    answered[k] += usize::from(ok);
                }
            }
        }
        // Measured: 19 of 45 answer (each of invariant factors, minimal polynomial, rational
        // form); the other 26 reach the fallbacks (rustmath-finitefields
        // forms_sentinel_natural_families checks the public chain on all 45).
        assert!(answered.iter().all(|&k| k >= 19), "{answered:?}");
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let mut random = [0usize; 3];
        for t in 0..600usize {
            let n = 2 + t % 5;
            let v: Vec<i64> = (0..n * n).map(|_| next(3) as i64 - 1).collect();
            for (k, ok) in check_zs_peeled(n, &v).into_iter().enumerate() {
                random[k] += usize::from(ok);
            }
        }
        // Measured: 107 of 600 answer (each of the three).
        assert!(random.iter().all(|&k| k >= 100), "{random:?}");
    }

    #[test]
    fn peeled_sentinel_answers_where_the_smith_paths_err() {
        // A 4×4 sentinel-like matrix (Z semantics) on which both Smith paths meet the pivot 2
        // ("2 is not a unit of Z"), so c79a582 returned Err for all three functions, while the
        // peeled path answers; the answer is right over Z and mod every p (frobenius.rs,
        // "Exactness").
        // gp: A = [0,0,0,0; -1,1,0,0; 1,0,0,0; 0,-1,1,0]; matfrobenius(A,1) = [x^4 - x^3],
        //     lift(matfrobenius(A*Mod(1,p),1)) == [x^4 + (p-1)*x^3] for p = 2, 3, 5, 7 (1)
        let v = [0, 0, 0, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0, -1, 1, 0];
        let a = zs_matrix(4, &v);
        assert!(invariant_factors_fallback(&a).is_err());
        assert!(rational_canonical_form_krylov(&a).is_err());
        assert!(rational_canonical_form_classical(&a).is_err());
        let f = zs_poly(&[0, 0, 0, -1, 1]);
        assert_eq!(invariant_factors(&a).unwrap(), vec![f.clone()]);
        assert_eq!(minimal_polynomial(&a).unwrap(), f);
        assert_eq!(check_zs_peeled(4, &v), [true, true, true]);
    }
}

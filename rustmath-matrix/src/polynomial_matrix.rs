//! Polynomial matrices and Smith normal form over polynomial rings
//!
//! This module provides operations on matrices whose entries are polynomials `F[x]` over a
//! field, which is essential for computing the rational canonical form.
//!
//! MAGMA Handbook ch. 26 §26.12.3 (`SmithForm`, `ElementaryDivisors` over a Euclidean ring)
//! and §26.12.2 (`InvariantFactors`, `RationalForm`), digest
//! `rustmath-lava/docs/Chapter_26_Matrices.md`.
//!
//! Algorithm: classical Smith reduction by elementary row and column operations over the
//! Euclidean domain `F[x]` (the degree is the Euclidean function), tracking `P`, `P⁻¹` and `Q`
//! with `P·M·Q = S`. N. Jacobson, *Basic Algebra I*, 2nd ed., §3.7 (equivalence of matrices
//! over a p.i.d.); H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138,
//! §2.4, Algorithm 2.4.14 (Smith normal form over Z, which applies verbatim with the degree in
//! place of `|·|`). MAGMA itself uses Steel's algorithm [Ste97] (J. Symbolic Comp. 24 (1997)
//! 409–432) for canonical forms over fields; this module does not implement it and makes no
//! complexity claim (coefficient and degree growth over Q are not controlled).

use rustmath_core::{Field, MathError, Result};
use rustmath_polynomials::UnivariatePolynomial;

/// A matrix whose entries are univariate polynomials
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialMatrix<F: Field> {
    rows: usize,
    cols: usize,
    data: Vec<UnivariatePolynomial<F>>,
}

impl<F: Field> PolynomialMatrix<F> {
    /// Create a new polynomial matrix
    pub fn new(rows: usize, cols: usize, data: Vec<UnivariatePolynomial<F>>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::InvalidArgument(
                "Data length must equal rows * cols".to_string(),
            ));
        }

        Ok(Self { rows, cols, data })
    }

    /// Create a zero polynomial matrix
    pub fn zero(rows: usize, cols: usize) -> Self {
        let data = vec![UnivariatePolynomial::new(vec![F::zero()]); rows * cols];
        Self { rows, cols, data }
    }

    /// Create an identity polynomial matrix (diagonal matrix with 1s)
    pub fn identity(size: usize) -> Self {
        let mut data = vec![UnivariatePolynomial::new(vec![F::zero()]); size * size];
        for i in 0..size {
            data[i * size + i] = UnivariatePolynomial::new(vec![F::one()]);
        }
        Self {
            rows: size,
            cols: size,
            data,
        }
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of cols
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get a reference to an entry
    pub fn get(&self, row: usize, col: usize) -> &UnivariatePolynomial<F> {
        &self.data[row * self.cols + col]
    }

    /// Get a mutable reference to an entry
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut UnivariatePolynomial<F> {
        &mut self.data[row * self.cols + col]
    }

    /// Set an entry
    pub fn set(&mut self, row: usize, col: usize, value: UnivariatePolynomial<F>) {
        self.data[row * self.cols + col] = value;
    }

    /// Swap two rows
    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }

        for col in 0..self.cols {
            let idx1 = row1 * self.cols + col;
            let idx2 = row2 * self.cols + col;
            self.data.swap(idx1, idx2);
        }
    }

    /// Swap two columns
    pub fn swap_cols(&mut self, col1: usize, col2: usize) {
        if col1 == col2 {
            return;
        }

        for row in 0..self.rows {
            let idx1 = row * self.cols + col1;
            let idx2 = row * self.cols + col2;
            self.data.swap(idx1, idx2);
        }
    }

    /// Check if all entries are zero
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|p| p.is_zero())
    }

    /// Get the maximum degree of any entry in the matrix
    pub fn max_degree(&self) -> Option<usize> {
        self.data
            .iter()
            .filter_map(|p| p.degree())
            .max()
    }

    /// Matrix product `self · other`.
    ///
    /// `Err(DimensionMismatch)` if `self.cols() != other.rows()`.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch(format!(
                "cannot multiply a {}x{} polynomial matrix by a {}x{} one",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut data = Vec::with_capacity(self.rows * other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = UnivariatePolynomial::zero();
                for k in 0..self.cols {
                    acc = acc + self.get(i, k).clone() * other.get(k, j).clone();
                }
                data.push(acc);
            }
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            data,
        })
    }

    /// Compute the Smith normal form of the polynomial matrix (MAGMA `SmithForm`,
    /// ch. 26 §26.12.3)
    ///
    /// Returns `S = P·M·Q` with `P` (m×m) and `Q` (n×n) unimodular, together with `P⁻¹`. `S` is
    /// diagonal, its non-zero diagonal entries are monic, the zero ones are trailing, and
    /// `d_1 | d_2 | … | d_r`: the unique Smith form of `M`. Any shape is accepted, including
    /// `0×n`, `m×0` and the zero matrix.
    ///
    /// Why the result can be trusted: every operation the loop performs is unimodular
    /// whatever its control-flow decisions (row/column swaps, `row_i -= q·row_k` and
    /// `col_j -= q·col_k` for any `q`, `row_k += row_i`, scaling by `c⁻¹` with `c` invertible),
    /// and each is applied to `S` and to the transform, so `S = P·M·Q` holds by construction.
    /// Before returning, `S` is checked unconditionally to be diagonal, monic, with trailing
    /// zeros and the divisibility chain (tested with exact division). A diagonal, monic,
    /// divisibility-chained matrix equivalent to `M` is its Smith form, by uniqueness
    /// (Jacobson §3.7), so a passing check certifies `S`. When the check fails the result is
    /// `Err(InvalidOperation)`, never a value. For `PrimeField`'s modulus-0 `Ring::zero()` /
    /// `Ring::one()` sentinels, `is_zero()` can only answer "non-zero" wrongly, never "zero",
    /// so the check stays truthful there too. Errors from `Field::inverse` propagate.
    ///
    /// Termination: each pass either leaves a remainder of smaller degree than the pivot or
    /// repairs divisibility (after which the next pass does), so the pivot degree `d` drops at
    /// least every two passes and a stage takes at most `2·d₀+1` passes. A stage exceeding
    /// `2·d₀+3` passes (only possible for a broken `Field` impl) returns `Err`.
    pub fn smith_normal_form(&self) -> Result<PolynomialSmithNormalForm<F>> {
        let (m, n) = (self.rows, self.cols);
        let mut s = self.clone();
        let mut p = Self::identity(m);
        let mut pinv = Self::identity(m);
        let mut q = Self::identity(n);

        // P is updated by left multiplication with an elementary E; P⁻¹ by right
        // multiplication with E⁻¹:
        //   swap rows k, i          ->  swap columns k, i of P⁻¹
        //   row_i -= c·row_k        ->  col_k += c·col_i of P⁻¹
        //   row_k += row_i          ->  col_i -= col_k of P⁻¹
        //   row_k *= c⁻¹            ->  col_k *= c of P⁻¹
        'outer: for k in 0..m.min(n) {
            let mut passes = 0usize;
            let mut cap: Option<usize> = None;
            loop {
                // The first entry of least degree among the non-zero S[i][j], i, j >= k,
                // scanned row-major from (k, k) with a strict '<', so (k, k) wins ties.
                let Some((pi, pj)) = s.find_min_degree_entry(k) else {
                    break 'outer; // the rest of S is zero
                };
                let d = s.get(pi, pj).degree().unwrap_or(0);
                let cap = *cap.get_or_insert(2 * d + 3);
                passes += 1;
                if passes > cap {
                    return Err(MathError::InvalidOperation(
                        "internal: F[x] Smith reduction made no progress".to_string(),
                    ));
                }
                if pi != k {
                    s.swap_rows(k, pi);
                    p.swap_rows(k, pi);
                    pinv.swap_cols(k, pi);
                }
                if pj != k {
                    s.swap_cols(k, pj);
                    q.swap_cols(k, pj);
                }
                // Row k and column k are not touched by the operations of this pass except
                // through S[k][k], which stays fixed.
                let pivot = s.get(k, k).clone();

                let mut dirty = false;
                // Clear column k by row operations.
                for i in (k + 1)..m {
                    if s.get(i, k).is_zero() {
                        continue;
                    }
                    let (quo, rem) = div_rem_field(s.get(i, k), &pivot)?;
                    s.row_sub_mul(i, k, &quo);
                    p.row_sub_mul(i, k, &quo);
                    pinv.col_add_mul(k, i, &quo);
                    dirty |= !rem.is_zero();
                }
                // Clear row k by column operations.
                for j in (k + 1)..n {
                    if s.get(k, j).is_zero() {
                        continue;
                    }
                    let (quo, rem) = div_rem_field(s.get(k, j), &pivot)?;
                    s.col_sub_mul(j, k, &quo);
                    q.col_sub_mul(j, k, &quo);
                    dirty |= !rem.is_zero();
                }
                if dirty {
                    continue; // a remainder of smaller degree than the pivot exists
                }

                // Row k and column k are clear. Repair divisibility: if the pivot does not
                // divide some S[i][j] with i, j > k, add row i to row k; the next pass then
                // leaves a remainder of smaller degree.
                let mut bad_row = None;
                'scan: for i in (k + 1)..m {
                    for j in (k + 1)..n {
                        if !div_rem_field(s.get(i, j), &pivot)?.1.is_zero() {
                            bad_row = Some(i);
                            break 'scan;
                        }
                    }
                }
                if let Some(i) = bad_row {
                    s.row_add(k, i);
                    p.row_add(k, i);
                    pinv.col_sub(i, k);
                    continue;
                }

                // Normalise the pivot to be monic.
                let c = pivot
                    .leading_coefficient()
                    .cloned()
                    .ok_or(MathError::DivisionByZero)?;
                if !c.is_one() {
                    let ci = c.inverse()?;
                    s.scale_row(k, &ci);
                    p.scale_row(k, &ci);
                    pinv.scale_col(k, &c);
                }
                break;
            }
        }

        check_smith_structure(&s)?;

        Ok(PolynomialSmithNormalForm {
            diagonal: s,
            left_transform: p,
            right_transform: q,
            left_inverse: pinv,
        })
    }

    /// Find the entry with minimal non-zero degree in the submatrix starting at (k, k):
    /// the first one in row-major order among those of least degree.
    fn find_min_degree_entry(&self, k: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, (usize, usize))> = None;
        for i in k..self.rows {
            for j in k..self.cols {
                let poly = self.get(i, j);
                if let Some(deg) = poly.degree() {
                    if best.is_none_or(|(b, _)| deg < b) {
                        best = Some((deg, (i, j)));
                    }
                }
            }
        }
        best.map(|(_, pos)| pos)
    }

    /// row_i -= c·row_k
    fn row_sub_mul(&mut self, i: usize, k: usize, c: &UnivariatePolynomial<F>) {
        for j in 0..self.cols {
            let v = self.get(i, j).clone() - c.clone() * self.get(k, j).clone();
            self.set(i, j, v);
        }
    }

    /// row_dst += row_src
    fn row_add(&mut self, dst: usize, src: usize) {
        for j in 0..self.cols {
            let v = self.get(dst, j).clone() + self.get(src, j).clone();
            self.set(dst, j, v);
        }
    }

    /// col_j -= c·col_k
    fn col_sub_mul(&mut self, j: usize, k: usize, c: &UnivariatePolynomial<F>) {
        for i in 0..self.rows {
            let v = self.get(i, j).clone() - c.clone() * self.get(i, k).clone();
            self.set(i, j, v);
        }
    }

    /// col_dst += c·col_src
    fn col_add_mul(&mut self, dst: usize, src: usize, c: &UnivariatePolynomial<F>) {
        for i in 0..self.rows {
            let v = self.get(i, dst).clone() + c.clone() * self.get(i, src).clone();
            self.set(i, dst, v);
        }
    }

    /// col_dst -= col_src
    fn col_sub(&mut self, dst: usize, src: usize) {
        for i in 0..self.rows {
            let v = self.get(i, dst).clone() - self.get(i, src).clone();
            self.set(i, dst, v);
        }
    }

    /// row_i *= c
    fn scale_row(&mut self, i: usize, c: &F) {
        for j in 0..self.cols {
            let v = self.get(i, j).scalar_mul(c);
            self.set(i, j, v);
        }
    }

    /// col_j *= c
    fn scale_col(&mut self, j: usize, c: &F) {
        for i in 0..self.rows {
            let v = self.get(i, j).scalar_mul(c);
            self.set(i, j, v);
        }
    }
}

/// The unconditional output check of [`PolynomialMatrix::smith_normal_form`]: `s` is
/// diagonal, its non-zero diagonal entries are monic, its zero diagonal entries are trailing,
/// and each non-zero diagonal entry divides the next.
fn check_smith_structure<F: Field>(s: &PolynomialMatrix<F>) -> Result<()> {
    let fail = || {
        Err(MathError::InvalidOperation(
            "internal: F[x] Smith form failed its structural check".to_string(),
        ))
    };
    for i in 0..s.rows {
        for j in 0..s.cols {
            if i != j && !s.get(i, j).is_zero() {
                return fail();
            }
        }
    }
    let r = s.rows.min(s.cols);
    let diag: Vec<&UnivariatePolynomial<F>> = (0..r).map(|i| s.get(i, i)).collect();
    for d in &diag {
        if let Some(lc) = d.leading_coefficient() {
            if !lc.is_one() {
                return fail();
            }
        }
    }
    for w in diag.windows(2) {
        match (w[0].is_zero(), w[1].is_zero()) {
            (true, false) => return fail(), // zeros must be trailing
            (false, false) if !div_rem_field(w[1], w[0])?.1.is_zero() => return fail(),
            _ => {}
        }
    }
    Ok(())
}

/// Result of Smith normal form computation for polynomial matrices
#[derive(Debug, Clone)]
pub struct PolynomialSmithNormalForm<F: Field> {
    /// The diagonal matrix `S = P·M·Q` (the Smith normal form)
    pub diagonal: PolynomialMatrix<F>,
    /// Left transformation matrix `P` (m×m, unimodular)
    pub left_transform: PolynomialMatrix<F>,
    /// Right transformation matrix `Q` (n×n, unimodular)
    pub right_transform: PolynomialMatrix<F>,
    /// `P⁻¹` (needed for the change of basis of the rational canonical form)
    pub left_inverse: PolynomialMatrix<F>,
}

impl<F: Field> PolynomialSmithNormalForm<F> {
    /// All non-zero diagonal entries `d_1 | d_2 | … | d_r` of `S`, monic, unit 1s included
    /// (MAGMA `ElementaryDivisors` over `F[x]`); `r` is the rank.
    pub fn invariant_factors(&self) -> Vec<UnivariatePolynomial<F>> {
        let min_dim = self.diagonal.rows.min(self.diagonal.cols);
        (0..min_dim)
            .map(|i| self.diagonal.get(i, i))
            .filter(|d| !d.is_zero())
            .cloned()
            .collect()
    }
}

/// Euclidean division over a field: `(q, r)` with `a = q·b + r` and `deg r < deg b`.
///
/// `Err(DivisionByZero)` if `b = 0`; errors from `Field::inverse` of `lc(b)` propagate.
/// `lc(b)⁻¹` is computed once with [`Field::inverse`] (never `/`). Each step cancels the top
/// coefficient of `r` by construction (it is popped, not computed), so the loop runs at most
/// `deg a - deg b + 1` times for any `Field` impl, including ones with rounding arithmetic.
pub(crate) fn div_rem_field<F: Field>(
    a: &UnivariatePolynomial<F>,
    b: &UnivariatePolynomial<F>,
) -> Result<(UnivariatePolynomial<F>, UnivariatePolynomial<F>)> {
    let Some(db) = b.degree() else {
        return Err(MathError::DivisionByZero);
    };
    let Some(da) = a.degree() else {
        return Ok((UnivariatePolynomial::zero(), UnivariatePolynomial::zero()));
    };
    if da < db {
        return Ok((UnivariatePolynomial::zero(), a.clone()));
    }
    let bc = b.coefficients();
    let inv = bc[db].inverse()?;
    let mut quo = vec![F::zero(); da - db + 1];
    // Working remainder, trimmed: its last entry is non-zero (or it is empty for 0).
    let mut r: Vec<F> = a.coefficients().to_vec();
    while r.len() > db {
        let top = r.len() - 1; // deg r >= deg b
        let c = r[top].clone() * inv.clone();
        let pos = top - db;
        // r -= c·x^pos·b on the coefficients below the top; the top one is dropped.
        for (j, bj) in bc.iter().enumerate().take(db) {
            r[pos + j] = r[pos + j].clone() - c.clone() * bj.clone();
        }
        quo[pos] = quo[pos].clone() + c;
        r.pop();
        while r.last().is_some_and(|x| x.is_zero()) {
            r.pop();
        }
    }
    Ok((UnivariatePolynomial::new(quo), UnivariatePolynomial::new(r)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    #[test]
    fn test_polynomial_matrix_creation() {
        let p1 = UnivariatePolynomial::new(vec![Rational::from_integer(1)]);
        let p0 = UnivariatePolynomial::new(vec![Rational::from_integer(0)]);

        let data = vec![p1.clone(), p0.clone(), p0.clone(), p1.clone()];
        let pm = PolynomialMatrix::new(2, 2, data).unwrap();

        assert_eq!(pm.rows(), 2);
        assert_eq!(pm.cols(), 2);
    }

    #[test]
    fn test_polynomial_division() {
        // Divide x^2 - 1 by x - 1, should give x + 1 with remainder 0
        let dividend = UnivariatePolynomial::new(vec![
            Rational::from_integer(-1),  // constant
            Rational::from_integer(0),   // x
            Rational::from_integer(1),   // x^2
        ]);

        let divisor = UnivariatePolynomial::new(vec![
            Rational::from_integer(-1),  // constant
            Rational::from_integer(1),   // x
        ]);

        let (q, r) = div_rem_field(&dividend, &divisor).unwrap();

        assert_eq!(q.degree(), Some(1));
        assert!(r.is_zero() || r.degree() == Some(0) && r.coeff(0) == &Rational::from_integer(0));
        assert_eq!(q, px(&[1, 1]));
        assert!(r.is_zero());
    }

    // Reference values: PARI/GP 2.15.4. `matsnf(M)` lists the Smith form in decreasing
    // divisibility order (RustMath and MAGMA use increasing order, so the lists are reversed)
    // and refuses non-square polynomial matrices ("inconsistent dimensions in gsmithall"), so
    // the non-square cases below are padded with a zero row or column for PARI.

    /// The polynomial with integer coefficients `c` (low to high) over Q.
    fn px(c: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(c.iter().map(|&v| Rational::from_integer(v)).collect())
    }

    fn pm(
        rows: usize,
        cols: usize,
        entries: Vec<UnivariatePolynomial<Rational>>,
    ) -> PolynomialMatrix<Rational> {
        PolynomialMatrix::new(rows, cols, entries).unwrap()
    }

    /// Runs the Smith form and asserts every certificate: P·M·Q = S, P·P⁻¹ = I, diagonal,
    /// monic non-zero diagonal entries, trailing zeros, divisibility chain. Returns the
    /// diagonal of S.
    fn check_snf(m: &PolynomialMatrix<Rational>) -> Vec<UnivariatePolynomial<Rational>> {
        let snf = m.smith_normal_form().unwrap();
        let s = &snf.diagonal;
        let (p, q, pinv) = (&snf.left_transform, &snf.right_transform, &snf.left_inverse);
        assert_eq!(&p.mul(m).unwrap().mul(q).unwrap(), s, "P·M·Q != S");
        assert_eq!(
            p.mul(pinv).unwrap(),
            PolynomialMatrix::identity(m.rows()),
            "P·P⁻¹ != I"
        );
        assert_eq!(
            pinv.mul(p).unwrap(),
            PolynomialMatrix::identity(m.rows()),
            "P⁻¹·P != I"
        );
        for i in 0..s.rows() {
            for j in 0..s.cols() {
                if i != j {
                    assert!(s.get(i, j).is_zero(), "S not diagonal at ({i}, {j})");
                }
            }
        }
        let d: Vec<_> = (0..s.rows().min(s.cols()))
            .map(|i| s.get(i, i).clone())
            .collect();
        for w in d.windows(2) {
            if !w[1].is_zero() {
                assert!(!w[0].is_zero(), "zeros not trailing");
                assert!(
                    div_rem_field(&w[1], &w[0]).unwrap().1.is_zero(),
                    "chain broken"
                );
            }
        }
        for e in &d {
            if let Some(lc) = e.leading_coefficient() {
                assert_eq!(lc, &Rational::from_integer(1), "not monic");
            }
        }
        d
    }

    #[test]
    fn div_rem_field_non_monic_divisor() {
        // gp: divrem(2*x^3+3*x+1, 2*x+4) = [x^2 - 2*x + 11/2, -21]~
        let a = px(&[1, 3, 0, 2]);
        let b = px(&[4, 2]);
        let (q, r) = div_rem_field(&a, &b).unwrap();
        let expected_q = UnivariatePolynomial::new(vec![
            Rational::new(11, 2).unwrap(),
            Rational::from_integer(-2),
            Rational::from_integer(1),
        ]);
        assert_eq!(q, expected_q);
        assert_eq!(r, px(&[-21]));
        assert_eq!(q * b + r, a);
    }

    #[test]
    fn div_rem_field_edges() {
        assert!(matches!(
            div_rem_field(&px(&[1, 1]), &px(&[0])),
            Err(MathError::DivisionByZero)
        ));
        let (q, r) = div_rem_field(&px(&[0]), &px(&[1, 1])).unwrap();
        assert!(q.is_zero() && r.is_zero());
        let (q, r) = div_rem_field(&px(&[1, 1]), &px(&[1, 0, 1])).unwrap();
        assert!(q.is_zero());
        assert_eq!(r, px(&[1, 1]));
        // Division by the constant 2: gp divrem(x^2+3, 2) = [1/2*x^2 + 3/2, 0]~
        let (q, r) = div_rem_field(&px(&[3, 0, 1]), &px(&[2])).unwrap();
        assert_eq!(q.scalar_mul(&Rational::from_integer(2)), px(&[3, 0, 1]));
        assert!(r.is_zero());
    }

    #[test]
    fn mul_dimension_mismatch() {
        let a = PolynomialMatrix::<Rational>::identity(2);
        let b = PolynomialMatrix::<Rational>::identity(3);
        assert!(matches!(a.mul(&b), Err(MathError::DimensionMismatch(_))));
    }

    #[test]
    fn snf_xi_minus_diag23() {
        // The ch026.InvariantFactors defect: this used to give [x-2, x-3].
        // gp: matsnf([x-2,0;0,x-3]) = [x^2 - 5*x + 6, 1]
        let m = pm(2, 2, vec![px(&[-2, 1]), px(&[0]), px(&[0]), px(&[-3, 1])]);
        assert_eq!(check_snf(&m), vec![px(&[1]), px(&[6, -5, 1])]);
        let snf = m.smith_normal_form().unwrap();
        assert_eq!(snf.invariant_factors(), vec![px(&[1]), px(&[6, -5, 1])]);
    }

    #[test]
    fn snf_g1() {
        // gp: matsnf([x^2-1, x+1; x-1, x^2+x]) = [x^4 + x^3 - 2*x^2 - x + 1, 1]
        let m = pm(
            2,
            2,
            vec![px(&[-1, 0, 1]), px(&[1, 1]), px(&[-1, 1]), px(&[0, 1, 1])],
        );
        assert_eq!(check_snf(&m), vec![px(&[1]), px(&[1, -1, -2, 1, 1])]);
    }

    #[test]
    fn snf_non_square() {
        // gp: matsnf(matconcat([[x,x,0;0,x^2,x^2];[0,0,0]])) = [0, x^2, x]
        let h2 = pm(
            2,
            3,
            vec![
                px(&[0, 1]),
                px(&[0, 1]),
                px(&[0]),
                px(&[0]),
                px(&[0, 0, 1]),
                px(&[0, 0, 1]),
            ],
        );
        assert_eq!(check_snf(&h2), vec![px(&[0, 1]), px(&[0, 0, 1])]);
        // H3 = H2 transposed; gp: matsnf(matconcat([mattranspose(H2),[0,0,0]~])) = [0, x^2, x]
        let h3 = pm(
            3,
            2,
            vec![
                px(&[0, 1]),
                px(&[0]),
                px(&[0, 1]),
                px(&[0, 0, 1]),
                px(&[0]),
                px(&[0, 0, 1]),
            ],
        );
        assert_eq!(check_snf(&h3), vec![px(&[0, 1]), px(&[0, 0, 1])]);
        // gp: matsnf(matconcat([[x-1,x^2-1,x^3-1;x^2-1,x^3-1,x^4-1];[0,0,0]])) =
        //     [0, x^2 - x, x - 1]
        let k2 = pm(
            2,
            3,
            vec![
                px(&[-1, 1]),
                px(&[-1, 0, 1]),
                px(&[-1, 0, 0, 1]),
                px(&[-1, 0, 1]),
                px(&[-1, 0, 0, 1]),
                px(&[-1, 0, 0, 0, 1]),
            ],
        );
        assert_eq!(check_snf(&k2), vec![px(&[-1, 1]), px(&[0, -1, 1])]);
        // gp: matsnf(matconcat([[x,x+1,0;0,x,x+1];[0,0,0]])) = [0, 1, 1]
        let g2 = pm(
            2,
            3,
            vec![
                px(&[0, 1]),
                px(&[1, 1]),
                px(&[0]),
                px(&[0]),
                px(&[0, 1]),
                px(&[1, 1]),
            ],
        );
        assert_eq!(check_snf(&g2), vec![px(&[1]), px(&[1])]);
    }

    #[test]
    fn snf_rank_deficient_and_order_repair() {
        // gp: matsnf([x, x^2; x^2, x^3]) = [0, x], matrank = 1
        let g4 = pm(
            2,
            2,
            vec![
                px(&[0, 1]),
                px(&[0, 0, 1]),
                px(&[0, 0, 1]),
                px(&[0, 0, 0, 1]),
            ],
        );
        assert_eq!(check_snf(&g4), vec![px(&[0, 1]), px(&[0])]);
        assert_eq!(
            g4.smith_normal_form().unwrap().invariant_factors(),
            vec![px(&[0, 1])]
        );
        // gp: matsnf([x^3,0;0,x^2]) = [x^3, x^2]
        let g5 = pm(
            2,
            2,
            vec![px(&[0, 0, 0, 1]), px(&[0]), px(&[0]), px(&[0, 0, 1])],
        );
        assert_eq!(check_snf(&g5), vec![px(&[0, 0, 1]), px(&[0, 0, 0, 1])]);
    }

    #[test]
    fn snf_g6_g7() {
        // gp: matsnf(matdiagonal([x^2+1, x^2-1, x+1])) = [x^4 - 1, x + 1, 1]
        let z = || px(&[0]);
        let g6 = pm(
            3,
            3,
            vec![
                px(&[1, 0, 1]),
                z(),
                z(),
                z(),
                px(&[-1, 0, 1]),
                z(),
                z(),
                z(),
                px(&[1, 1]),
            ],
        );
        assert_eq!(
            check_snf(&g6),
            vec![px(&[1]), px(&[1, 1]), px(&[-1, 0, 0, 0, 1])]
        );
        // Non-monic entries. gp: matsnf([2*x+2, 4; 6, 3*x]) = [x^2 + x - 4, 1],
        //     matdet = 6*x^2 + 6*x - 24
        let g7 = pm(2, 2, vec![px(&[2, 2]), px(&[4]), px(&[6]), px(&[0, 3])]);
        assert_eq!(check_snf(&g7), vec![px(&[1]), px(&[-4, 1, 1])]);
    }

    #[test]
    fn snf_zero_and_empty() {
        // gp: matsnf([0,0;0,0]) = [0, 0]
        let z = PolynomialMatrix::<Rational>::zero(2, 2);
        assert_eq!(check_snf(&z), vec![px(&[0]), px(&[0])]);
        assert!(z
            .smith_normal_form()
            .unwrap()
            .invariant_factors()
            .is_empty());
        // Definitional: 0×0, 0×2 and 2×0 matrices.
        let e = PolynomialMatrix::<Rational>::zero(0, 0);
        assert!(check_snf(&e).is_empty());
        assert!(check_snf(&PolynomialMatrix::<Rational>::zero(0, 2)).is_empty());
        assert!(check_snf(&PolynomialMatrix::<Rational>::zero(2, 0)).is_empty());
    }
}

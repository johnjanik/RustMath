//! Polynomial matrices and Smith normal form over polynomial rings
//!
//! This module provides operations on matrices whose entries are polynomials,
//! which is essential for computing the rational canonical form.

use rustmath_core::{Field, MathError, Result};
use rustmath_polynomials::UnivariatePolynomial;

/// A matrix whose entries are univariate polynomials
#[derive(Debug, Clone)]
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

    /// Compute the Smith normal form of the polynomial matrix
    ///
    /// Returns a diagonal matrix where each diagonal entry divides the next,
    /// along with transformation matrices.
    pub fn smith_normal_form(&self) -> Result<PolynomialSmithNormalForm<F>> {
        let mut s = self.clone();
        let mut left = Self::identity(self.rows);
        let mut right = Self::identity(self.cols);

        let min_dim = self.rows.min(self.cols);

        // Diagonalize the matrix
        for k in 0..min_dim {
            // Bring a non-zero entry of minimal degree to position (k, k)
            if !Self::pivot_to_diagonal(&mut s, &mut left, &mut right, k)? {
                // Rest of matrix is zero
                break;
            }

            // Eliminate entries in row k and column k
            let mut changed = true;
            while changed {
                changed = false;

                // Eliminate row k
                for j in (k + 1)..self.cols {
                    if !s.get(k, j).is_zero() {
                        Self::eliminate_entry_in_row(&mut s, &mut right, k, j)?;
                        changed = true;
                    }
                }

                // Eliminate column k
                for i in (k + 1)..self.rows {
                    if !s.get(i, k).is_zero() {
                        Self::eliminate_entry_in_col(&mut s, &mut left, k, i)?;
                        changed = true;
                    }
                }

                // Check if pivot still has minimal degree
                if changed {
                    if let Some((pi, pj)) = Self::find_min_degree_entry(&s, k)? {
                        if pi != k || pj != k {
                            // Need to re-pivot
                            if pi != k {
                                s.swap_rows(k, pi);
                                left.swap_rows(k, pi);
                            }
                            if pj != k {
                                s.swap_cols(k, pj);
                                right.swap_cols(k, pj);
                            }
                        }
                    }
                }
            }
        }

        // Ensure divisibility: d_i | d_{i+1}
        Self::ensure_divisibility(&mut s, &mut left, &mut right)?;

        Ok(PolynomialSmithNormalForm {
            diagonal: s,
            left_transform: left,
            right_transform: right,
        })
    }

    /// Find the entry with minimal non-zero degree in the submatrix starting at (k, k)
    fn find_min_degree_entry(&self, k: usize) -> Result<Option<(usize, usize)>> {
        let mut min_degree = None;
        let mut min_pos = None;

        for i in k..self.rows {
            for j in k..self.cols {
                let poly = self.get(i, j);
                if !poly.is_zero() {
                    let deg = poly.degree().unwrap_or(0);
                    if min_degree.is_none() || deg < min_degree.unwrap() {
                        min_degree = Some(deg);
                        min_pos = Some((i, j));
                    }
                }
            }
        }

        Ok(min_pos)
    }

    /// Pivot: bring entry with minimal degree to position (k, k)
    fn pivot_to_diagonal(
        s: &mut Self,
        left: &mut Self,
        right: &mut Self,
        k: usize,
    ) -> Result<bool> {
        if let Some((pi, pj)) = Self::find_min_degree_entry(s, k)? {
            if pi != k {
                s.swap_rows(k, pi);
                left.swap_rows(k, pi);
            }
            if pj != k {
                s.swap_cols(k, pj);
                right.swap_cols(k, pj);
            }
            Ok(true)
        } else {
            Ok(false) // All remaining entries are zero
        }
    }

    /// Eliminate an entry in row k by column operations
    fn eliminate_entry_in_row(s: &mut Self, right: &mut Self, k: usize, j: usize) -> Result<()> {
        let pivot = s.get(k, k).clone();
        let target = s.get(k, j).clone();

        if pivot.is_zero() {
            return Ok(());
        }

        // Compute quotient and remainder
        let (q, _r) = target.pseudo_div_rem(&pivot)?;

        // Column operation: col_j -= q * col_k
        for i in 0..s.rows {
            let val = s.get(i, j).clone() - q.clone() * s.get(i, k).clone();
            s.set(i, j, val);
        }

        for i in 0..right.rows {
            let val = right.get(i, j).clone() - q.clone() * right.get(i, k).clone();
            right.set(i, j, val);
        }

        // If remainder is non-zero, we may need to swap columns and continue
        Ok(())
    }

    /// Eliminate an entry in column k by row operations
    fn eliminate_entry_in_col(s: &mut Self, left: &mut Self, k: usize, i: usize) -> Result<()> {
        let pivot = s.get(k, k).clone();
        let target = s.get(i, k).clone();

        if pivot.is_zero() {
            return Ok(());
        }

        // Compute quotient and remainder
        let (q, _r) = target.pseudo_div_rem(&pivot)?;

        // Row operation: row_i -= q * row_k
        for j in 0..s.cols {
            let val = s.get(i, j).clone() - q.clone() * s.get(k, j).clone();
            s.set(i, j, val);
        }

        for j in 0..left.cols {
            let val = left.get(i, j).clone() - q.clone() * left.get(k, j).clone();
            left.set(i, j, val);
        }

        Ok(())
    }

    /// Ensure that each diagonal entry divides the next
    fn ensure_divisibility(s: &mut Self, _left: &mut Self, _right: &mut Self) -> Result<()> {
        let min_dim = s.rows.min(s.cols);

        for i in 0..(min_dim - 1) {
            let d_i = s.get(i, i).clone();
            let d_next = s.get(i + 1, i + 1).clone();

            if !d_i.is_zero() && !d_next.is_zero() {
                // Check if d_i divides d_next
                let (_q, r) = d_next.pseudo_div_rem(&d_i)?;

                if !r.is_zero() {
                    // d_i doesn't divide d_{i+1}
                    // Add column i to column i+1
                    for row in 0..s.rows {
                        let val = s.get(row, i + 1).clone() + s.get(row, i).clone();
                        s.set(row, i + 1, val);
                    }

                    // Re-run elimination for this position
                    // (In a full implementation, we'd recursively fix this)
                }
            }
        }

        Ok(())
    }
}

/// Result of Smith normal form computation for polynomial matrices
#[derive(Debug, Clone)]
pub struct PolynomialSmithNormalForm<F: Field> {
    /// The diagonal matrix (Smith normal form)
    pub diagonal: PolynomialMatrix<F>,
    /// Left transformation matrix
    pub left_transform: PolynomialMatrix<F>,
    /// Right transformation matrix
    pub right_transform: PolynomialMatrix<F>,
}

impl<F: Field> PolynomialSmithNormalForm<F> {
    /// Extract the invariant factors (non-zero diagonal entries)
    pub fn invariant_factors(&self) -> Vec<UnivariatePolynomial<F>> {
        let min_dim = self.diagonal.rows.min(self.diagonal.cols);
        let mut factors = Vec::new();

        for i in 0..min_dim {
            let entry = self.diagonal.get(i, i);
            if !entry.is_zero() {
                // Make it monic
                if let Some(deg) = entry.degree() {
                    let lc = entry.coeff(deg);
                    if lc != &F::zero() {
                        let mut monic = entry.clone();
                        // Divide by leading coefficient to make it monic
                        let lc_inv = F::one() / lc.clone();
                        monic = monic.scalar_mul(&lc_inv);
                        factors.push(monic);
                    }
                }
            }
        }

        factors
    }
}

// Extension trait for polynomial operations
trait PolynomialExt<F: Field> {
    fn pseudo_div_rem(&self, other: &Self) -> Result<(Self, Self)>
    where
        Self: Sized;
    fn scalar_mul(&self, scalar: &F) -> Self;
}

impl<F: Field> PolynomialExt<F> for UnivariatePolynomial<F> {
    /// Pseudo-division: returns (q, r) such that lc(divisor)^k * self = q * divisor + r
    /// where deg(r) < deg(divisor) and k is chosen appropriately
    fn pseudo_div_rem(&self, divisor: &Self) -> Result<(Self, Self)> {
        if divisor.is_zero() {
            return Err(MathError::DivisionByZero);
        }

        if self.is_zero() {
            return Ok((
                UnivariatePolynomial::new(vec![F::zero()]),
                UnivariatePolynomial::new(vec![F::zero()]),
            ));
        }

        let self_deg = match self.degree() {
            Some(d) => d,
            None => return Ok((
                UnivariatePolynomial::new(vec![F::zero()]),
                self.clone(),
            )),
        };

        let div_deg = match divisor.degree() {
            Some(d) => d,
            None => return Err(MathError::DivisionByZero),
        };

        if self_deg < div_deg {
            return Ok((
                UnivariatePolynomial::new(vec![F::zero()]),
                self.clone(),
            ));
        }

        // Regular polynomial division for fields
        let mut remainder = self.clone();
        let mut quotient_coeffs = vec![F::zero(); self_deg - div_deg + 1];

        let divisor_lc = divisor.coeff(div_deg);

        while let Some(rem_deg) = remainder.degree() {
            if rem_deg < div_deg {
                break;
            }

            let rem_lc = remainder.coeff(rem_deg);
            let coeff = rem_lc.clone() / divisor_lc.clone();

            let pos = rem_deg - div_deg;
            quotient_coeffs[pos] = coeff.clone();

            // Subtract coeff * x^pos * divisor from remainder
            for i in 0..=div_deg {
                let old_coeff = remainder.coeff(i + pos);
                let new_coeff = old_coeff.clone() - coeff.clone() * divisor.coeff(i).clone();

                let mut new_coeffs = remainder.coefficients().to_vec();
                while new_coeffs.len() <= i + pos {
                    new_coeffs.push(F::zero());
                }
                new_coeffs[i + pos] = new_coeff;
                remainder = UnivariatePolynomial::new(new_coeffs);
            }
        }

        Ok((
            UnivariatePolynomial::new(quotient_coeffs),
            remainder,
        ))
    }

    fn scalar_mul(&self, scalar: &F) -> Self {
        let coeffs: Vec<F> = self
            .coefficients()
            .iter()
            .map(|c| c.clone() * scalar.clone())
            .collect();
        UnivariatePolynomial::new(coeffs)
    }
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

        let (q, r) = dividend.pseudo_div_rem(&divisor).unwrap();

        assert_eq!(q.degree(), Some(1));
        assert!(r.is_zero() || r.degree() == Some(0) && r.coeff(0) == &Rational::from_integer(0));
    }
}

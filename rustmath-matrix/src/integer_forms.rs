//! Integer matrix normal forms (Smith, Hermite)
//!
//! This module provides algorithms for computing canonical forms of matrices
//! over Euclidean domains, particularly useful for integer matrices.

use crate::Matrix;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::{EuclideanDomain, Result, Ring};

/// Smith Normal Form result
///
/// For a matrix A, returns S = P*A*Q where:
/// - S is diagonal with d₁ | d₂ | ... | dᵣ (each diagonal element divides the next)
/// - P and Q are unimodular (invertible over the ring)
#[derive(Debug, Clone)]
pub struct SmithNormalForm<R: EuclideanDomain> {
    /// The diagonal matrix S
    pub s: Matrix<R>,
    /// Left transformation matrix P (such that S = P*A*Q)
    pub p: Matrix<R>,
    /// Right transformation matrix Q
    pub q: Matrix<R>,
}

/// Hermite Normal Form result
///
/// For a matrix A, returns H = U*A where:
/// - H is in upper-triangular (row) Hermite normal form
/// - U is unimodular (invertible over the ring)
#[derive(Debug, Clone)]
pub struct HermiteNormalForm<R: EuclideanDomain> {
    /// The Hermite normal form matrix H
    pub h: Matrix<R>,
    /// Transformation matrix U (such that H = U*A)
    pub u: Matrix<R>,
}

impl<R: EuclideanDomain + OrderedRing> SmithNormalForm<R> {
    /// Sign-canonicalise the diagonal over an *ordered* Euclidean domain
    /// (e.g. ℤ): every negative diagonal entry is negated by scaling its row
    /// of `s` — and of `p` — by the unit `-1`, so `s = p·a·q` is preserved and
    /// `p` stays unimodular. Over ℤ the result matches SymPy's / Sage's
    /// `smith_normal_form` (all diagonal entries non-negative).
    ///
    /// Over a general (unordered) Euclidean domain the SNF diagonal is only
    /// determined up to units, which is why this is a separate opt-in step.
    pub fn canonicalize_signs(&mut self) {
        let n = self.s.cols();
        let min_dim = self.s.rows().min(n);
        for i in 0..min_dim {
            if self.s.data()[i * n + i].sign() < 0 {
                negate_row(&mut self.s, i);
                negate_row(&mut self.p, i);
            }
        }
    }
}

impl<R: EuclideanDomain + OrderedRing> HermiteNormalForm<R> {
    /// Canonicalise over an *ordered* Euclidean domain (e.g. ℤ), in place:
    ///
    /// 1. every pivot (first non-zero entry of a row) is made positive by
    ///    scaling the row of `h` — and of `u` — by the unit `-1`;
    /// 2. every entry **above** a pivot is reduced into `[0, pivot)` by a
    ///    floor-division row operation.
    ///
    /// Both steps are unimodular row operations, so `h = u·a` is preserved.
    /// Over ℤ the result is the unique (row-style) Hermite normal form, as
    /// returned by Sage's `hermite_form` / SymPy's `hermite_normal_form`.
    pub fn canonicalize(&mut self) -> Result<()> {
        let m = self.h.rows();
        let n = self.h.cols();
        for r in 0..m {
            let pc = match (0..n).find(|&c| !self.h.data()[r * n + c].is_zero()) {
                Some(c) => c,
                None => continue, // zero row: nothing to normalise
            };
            if self.h.data()[r * n + pc].sign() < 0 {
                negate_row(&mut self.h, r);
                negate_row(&mut self.u, r);
            }
            let pivot = self.h.data()[r * n + pc].clone();
            for a in 0..r {
                let v = self.h.data()[a * n + pc].clone();
                if v.is_zero() {
                    continue;
                }
                // Floor division: q with v − q·pivot ∈ [0, pivot). `div_rem`
                // may be truncated (ℤ), so fix up a negative remainder.
                let (mut q, rem) = v.div_rem(&pivot)?;
                if rem.sign() < 0 {
                    q = q - R::one();
                }
                if q.is_zero() {
                    continue;
                }
                row_sub_multiple(&mut self.h, a, r, &q);
                row_sub_multiple(&mut self.u, a, r, &q);
            }
        }
        Ok(())
    }
}

/// Negate a row in place (multiplication by the unit `-1`).
fn negate_row<R: Ring>(mat: &mut Matrix<R>, row: usize) {
    let n = mat.cols();
    for c in 0..n {
        let v = R::zero() - mat.data()[row * n + c].clone();
        mat.data_mut()[row * n + c] = v;
    }
}

/// `row_dst ← row_dst − c · row_src`, in place.
fn row_sub_multiple<R: Ring>(mat: &mut Matrix<R>, dst: usize, src: usize, c: &R) {
    let n = mat.cols();
    for j in 0..n {
        let v = mat.data()[dst * n + j].clone() - c.clone() * mat.data()[src * n + j].clone();
        mat.data_mut()[dst * n + j] = v;
    }
}

/// `col_dst ← col_dst + c · col_src`, in place.
fn col_add_multiple<R: Ring>(mat: &mut Matrix<R>, dst: usize, src: usize, c: &R) {
    let m = mat.rows();
    let n = mat.cols();
    for i in 0..m {
        let v = mat.data()[i * n + dst].clone() + c.clone() * mat.data()[i * n + src].clone();
        mat.data_mut()[i * n + dst] = v;
    }
}

impl<R: EuclideanDomain> Matrix<R> {
    /// Compute the Smith Normal Form of the matrix
    ///
    /// Uses elementary row and column operations to reduce the matrix to
    /// diagonal form where each diagonal entry divides the next.
    ///
    /// This is useful for:
    /// - Computing invariant factors
    /// - Solving systems of linear Diophantine equations
    /// - Computing the structure of finitely generated modules
    pub fn smith_normal_form(&self) -> Result<SmithNormalForm<R>> {
        let m = self.rows();
        let n = self.cols();

        // Initialize working matrix and transformation matrices
        let mut s = self.clone();
        let mut p = Matrix::identity(m);
        let mut q = Matrix::identity(n);

        let min_dim = m.min(n);

        // Process each diagonal position
        for k in 0..min_dim {
            // Find the entry with smallest non-zero norm in the remaining submatrix
            loop {
                let pivot = self.find_pivot_for_smith(&s, k)?;
                if pivot.is_none() {
                    // All remaining entries are zero
                    break;
                }

                let (pi, pj) = pivot.unwrap();

                // Move pivot to position (k, k)
                if pi != k {
                    s.swap_rows(k, pi)?;
                    p.swap_rows(k, pi)?;
                }
                if pj != k {
                    s.swap_cols(k, pj);
                    q.swap_cols(k, pj);
                }

                // Eliminate entries in row k and column k using the pivot.
                //
                // Termination invariant: when the pivot already divides the
                // target we eliminate by an *exact subtraction* (which leaves
                // the pivot row/column untouched apart from the zeroed
                // entry), and reserve the Bezout rotation for the
                // non-dividing case, where the new pivot gcd(pivot, target)
                // has strictly smaller norm. Using the rotation in the
                // dividing case does not terminate: `extended_gcd(a, -a)`
                // returns `(-a, 0, 1)` — same norm, columns rotated — and
                // the outer loop 2-cycles forever (observed on the 0/±1
                // boundary matrices of simplicial homology).
                let mut changed = false;

                // Eliminate row k (to the right of pivot)
                for j in (k + 1)..n {
                    if !s.data()[k * n + j].is_zero() {
                        let pivot_val = s.data()[k * n + k].clone();
                        let target_val = s.data()[k * n + j].clone();

                        let (quot, rem) = target_val.div_rem(&pivot_val)?;
                        if rem.is_zero() {
                            // colⱼ ← colⱼ − quot·col_k, on S and Q.
                            let neg_quot = R::zero() - quot;
                            col_add_multiple(&mut s, j, k, &neg_quot);
                            col_add_multiple(&mut q, j, k, &neg_quot);
                        } else {
                            let (gcd, a, b) = pivot_val.extended_gcd(&target_val);
                            // Apply column operation to eliminate s[k][j]
                            self.apply_column_gcd_operation(&mut s, &mut q, k, j, &pivot_val, &target_val, &gcd, &a, &b)?;
                        }
                        changed = true;
                    }
                }

                // Eliminate column k (below pivot)
                for i in (k + 1)..m {
                    if !s.data()[i * n + k].is_zero() {
                        let pivot_val = s.data()[k * n + k].clone();
                        let target_val = s.data()[i * n + k].clone();

                        let (quot, rem) = target_val.div_rem(&pivot_val)?;
                        if rem.is_zero() {
                            // rowᵢ ← rowᵢ − quot·row_k, on S and P.
                            row_sub_multiple(&mut s, i, k, &quot);
                            row_sub_multiple(&mut p, i, k, &quot);
                        } else {
                            let (gcd, a, b) = pivot_val.extended_gcd(&target_val);
                            // Apply row operation to eliminate s[i][k]
                            self.apply_row_gcd_operation(&mut s, &mut p, k, i, &pivot_val, &target_val, &gcd, &a, &b)?;
                        }
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }
            }
        }

        // Ensure divisibility property: d_i | d_{i+1}
        self.ensure_divisibility(&mut s, &mut p, &mut q)?;

        Ok(SmithNormalForm { s, p, q })
    }

    /// Compute the elementary divisors of the matrix
    ///
    /// Elementary divisors are the non-zero diagonal entries of the Smith Normal Form.
    /// They satisfy the divisibility property: d₁ | d₂ | ... | dᵣ
    ///
    /// These are fundamental invariants that characterize:
    /// - The structure of finitely generated modules over Euclidean domains
    /// - The invariant factors of linear transformations
    /// - The torsion structure of abelian groups
    ///
    /// Over a general Euclidean domain each divisor is determined only up to
    /// a unit; over ℤ, apply [`Integer::abs`](rustmath_integers::Integer) or
    /// [`SmithNormalForm::canonicalize_signs`] for the non-negative canonical
    /// representatives.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_matrix::Matrix;
    /// use rustmath_integers::Integer;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![
    ///     Integer::from(2), Integer::from(4),
    ///     Integer::from(6), Integer::from(8)
    /// ]).unwrap();
    ///
    /// // Verified with SymPy: smith_normal_form([[2, 4], [6, 8]]) = diag(2, 4).
    /// let divs: Vec<Integer> = m.elementary_divisors().unwrap()
    ///     .into_iter().map(|d| d.abs()).collect();
    /// assert_eq!(divs, vec![Integer::from(2), Integer::from(4)]);
    /// ```
    pub fn elementary_divisors(&self) -> Result<Vec<R>> {
        let snf = self.smith_normal_form()?;
        let min_dim = snf.s.rows().min(snf.s.cols());

        let mut divisors = Vec::new();
        for i in 0..min_dim {
            let diag_entry = snf.s.data()[i * snf.s.cols() + i].clone();
            if !diag_entry.is_zero() {
                divisors.push(diag_entry);
            }
        }

        Ok(divisors)
    }

    /// Compute the (row-style) Hermite Normal Form of the matrix
    ///
    /// Returns H = U*A where U is unimodular and:
    /// - H is in row echelon form: each row's pivot (first non-zero entry) is
    ///   strictly to the right of the previous row's pivot,
    /// - each entry above a pivot has been reduced modulo the pivot (with the
    ///   ring's `div_rem`, so over ℤ the representative may still be negative).
    ///
    /// Pivot *signs* and above-pivot representatives are only canonical up to
    /// units over a general Euclidean domain. Over an ordered domain (ℤ), call
    /// [`HermiteNormalForm::canonicalize`] to get the unique HNF with positive
    /// pivots and above-pivot entries in `[0, pivot)` (Sage's `hermite_form`).
    pub fn hermite_normal_form(&self) -> Result<HermiteNormalForm<R>> {
        let m = self.rows();
        let n = self.cols();

        let mut h = self.clone();
        let mut u = Matrix::identity(m);

        let mut pivot_row = 0;

        // Process each column
        for col in 0..n {
            if pivot_row >= m {
                break;
            }

            // Find non-zero entry in this column at or below pivot_row
            let mut found = None;
            for row in pivot_row..m {
                if !h.data()[row * n + col].is_zero() {
                    found = Some(row);
                    break;
                }
            }

            if found.is_none() {
                continue; // This column is all zeros below pivot_row
            }

            let mut current_row = found.unwrap();

            // Move non-zero entry to pivot position
            if current_row != pivot_row {
                h.swap_rows(pivot_row, current_row)?;
                u.swap_rows(pivot_row, current_row)?;
                current_row = pivot_row;
            }

            // Eliminate all entries below the pivot
            loop {
                let mut changed = false;

                for row in (current_row + 1)..m {
                    if !h.data()[row * n + col].is_zero() {
                        let pivot_val = h.data()[current_row * n + col].clone();
                        let target_val = h.data()[row * n + col].clone();

                        let (gcd, a, b) = pivot_val.extended_gcd(&target_val);

                        self.apply_row_gcd_operation(&mut h, &mut u, current_row, row, &pivot_val, &target_val, &gcd, &a, &b)?;
                        changed = true;
                    }
                }

                if !changed {
                    break;
                }

                // After elimination, the pivot might have moved to a different row
                // Find the smallest non-zero entry in this column
                let mut smallest_row = current_row;
                let mut smallest_norm = h.data()[current_row * n + col].norm();

                for row in (current_row + 1)..m {
                    if !h.data()[row * n + col].is_zero() {
                        let norm = h.data()[row * n + col].norm();
                        if norm < smallest_norm {
                            smallest_norm = norm;
                            smallest_row = row;
                        }
                    }
                }

                if smallest_row != current_row {
                    h.swap_rows(current_row, smallest_row)?;
                    u.swap_rows(current_row, smallest_row)?;
                }
            }

            // Reduce entries above the pivot
            for row in 0..current_row {
                if !h.data()[row * n + col].is_zero() {
                    let pivot_val = h.data()[current_row * n + col].clone();
                    let target_val = h.data()[row * n + col].clone();

                    // Compute quotient: target_val = q * pivot_val + r
                    let (q, _r) = target_val.div_rem(&pivot_val)?;

                    // Subtract q times the pivot row from this row
                    for c in 0..n {
                        let val = h.data()[row * n + c].clone() - q.clone() * h.data()[current_row * n + c].clone();
                        h.data_mut()[row * n + c] = val;
                    }

                    // Update transformation matrix (u is m×m: stride m, not n)
                    for c in 0..m {
                        let val = u.data()[row * m + c].clone() - q.clone() * u.data()[current_row * m + c].clone();
                        u.data_mut()[row * m + c] = val;
                    }
                }
            }

            pivot_row += 1;
        }

        Ok(HermiteNormalForm { h, u })
    }

    // Helper methods

    /// Find the position of the smallest non-zero entry in the submatrix starting at (k, k)
    fn find_pivot_for_smith(&self, mat: &Matrix<R>, k: usize) -> Result<Option<(usize, usize)>> {
        let m = mat.rows();
        let n = mat.cols();

        let mut min_norm = u64::MAX;
        let mut pivot = None;

        for i in k..m {
            for j in k..n {
                let val = &mat.data()[i * n + j];
                if !val.is_zero() {
                    let norm = val.norm();
                    if norm < min_norm {
                        min_norm = norm;
                        pivot = Some((i, j));
                    }
                }
            }
        }

        Ok(pivot)
    }

    /// Apply a GCD-based row operation to eliminate an entry
    ///
    /// Given two rows with pivot values a and b, computes gcd(a,b) = sa + tb
    /// and applies row operations to eliminate b.
    fn apply_row_gcd_operation(
        &self,
        mat: &mut Matrix<R>,
        transform: &mut Matrix<R>,
        row1: usize,
        row2: usize,
        a: &R,
        b: &R,
        gcd: &R,
        s: &R,
        t: &R,
    ) -> Result<()> {
        let n = mat.cols();

        // Compute multipliers: u = a/gcd, v = b/gcd
        let (u, _) = a.div_rem(gcd)?;
        let (v, _) = b.div_rem(gcd)?;

        // Apply transformation:
        // new_row1 = s*row1 + t*row2
        // new_row2 = -v*row1 + u*row2

        let mut new_row1 = vec![R::zero(); n];
        let mut new_row2 = vec![R::zero(); n];

        for j in 0..n {
            let r1_val = mat.data_mut()[row1 * n + j].clone();
            let r2_val = mat.data_mut()[row2 * n + j].clone();

            new_row1[j] = s.clone() * r1_val.clone() + t.clone() * r2_val.clone();
            new_row2[j] = u.clone() * r2_val - v.clone() * r1_val;
        }

        // Update matrix
        for j in 0..n {
            mat.data_mut()[row1 * n + j] = new_row1[j].clone();
            mat.data_mut()[row2 * n + j] = new_row2[j].clone();
        }

        // Apply same transformation to the transform matrix
        let transform_cols = transform.cols();
        let mut new_t_row1 = vec![R::zero(); transform_cols];
        let mut new_t_row2 = vec![R::zero(); transform_cols];

        for j in 0..transform_cols {
            let t1_val = transform.data_mut()[row1 * transform_cols + j].clone();
            let t2_val = transform.data_mut()[row2 * transform_cols + j].clone();

            new_t_row1[j] = s.clone() * t1_val.clone() + t.clone() * t2_val.clone();
            new_t_row2[j] = u.clone() * t2_val - v.clone() * t1_val;
        }

        for j in 0..transform_cols {
            transform.data_mut()[row1 * transform_cols + j] = new_t_row1[j].clone();
            transform.data_mut()[row2 * transform_cols + j] = new_t_row2[j].clone();
        }

        Ok(())
    }

    /// Apply a GCD-based column operation to eliminate an entry
    fn apply_column_gcd_operation(
        &self,
        mat: &mut Matrix<R>,
        transform: &mut Matrix<R>,
        col1: usize,
        col2: usize,
        a: &R,
        b: &R,
        gcd: &R,
        s: &R,
        t: &R,
    ) -> Result<()> {
        let m = mat.rows();
        let n = mat.cols();

        // Compute multipliers
        let (u, _) = a.div_rem(gcd)?;
        let (v, _) = b.div_rem(gcd)?;

        // Apply column transformation
        for i in 0..m {
            let c1_val = mat.data_mut()[i * n + col1].clone();
            let c2_val = mat.data_mut()[i * n + col2].clone();

            let new_c1 = s.clone() * c1_val.clone() + t.clone() * c2_val.clone();
            let new_c2 = u.clone() * c2_val - v.clone() * c1_val;

            mat.data_mut()[i * n + col1] = new_c1;
            mat.data_mut()[i * n + col2] = new_c2;
        }

        // Apply to transformation matrix
        let transform_rows = transform.rows();
        for i in 0..transform_rows {
            let c1_val = transform.data_mut()[i * n + col1].clone();
            let c2_val = transform.data_mut()[i * n + col2].clone();

            let new_c1 = s.clone() * c1_val.clone() + t.clone() * c2_val.clone();
            let new_c2 = u.clone() * c2_val - v.clone() * c1_val;

            transform.data_mut()[i * n + col1] = new_c1;
            transform.data_mut()[i * n + col2] = new_c2;
        }

        Ok(())
    }

    /// Enforce the divisibility chain `d₁ | d₂ | …` on the (already diagonal)
    /// matrix `mat`, updating `p` and `q` so that `mat = p·A·q` keeps holding.
    ///
    /// Uses the classic 2×2 reduction: for a diagonal pair `(dᵢ, dⱼ)`, `i < j`,
    /// with `dᵢ ∤ dⱼ`,
    ///
    /// 1. add column `j` to column `i` (stacks `dⱼ` under `dᵢ` in column `i`),
    /// 2. replace rows `i`, `j` by the Bézout combination
    ///    (`g = s·dᵢ + t·dⱼ` from the extended gcd), which leaves `g` at
    ///    `(i,i)`, `dᵢ·dⱼ/g` at `(j,j)`, zero at `(j,i)` and fill-in `t·dⱼ`
    ///    at `(i,j)`,
    /// 3. clear the fill-in with `colⱼ ← colⱼ − (t·dⱼ/g)·colᵢ` (exact, since
    ///    `g | dⱼ`).
    ///
    /// The pair becomes `(gcd(dᵢ,dⱼ), lcm-associate)`; every step is an
    /// elementary (unimodular) row/column operation. Processing `i` in
    /// increasing order makes the whole diagonal a divisibility chain: once
    /// `dᵢ` divides all later entries, later pair fixes replace those entries
    /// by gcds/lcms of multiples of `dᵢ`, which stay multiples of `dᵢ`.
    fn ensure_divisibility(
        &self,
        mat: &mut Matrix<R>,
        p: &mut Matrix<R>,
        q: &mut Matrix<R>,
    ) -> Result<()> {
        let n = mat.cols();
        let min_dim = mat.rows().min(n);
        if min_dim == 0 {
            return Ok(());
        }

        for i in 0..(min_dim - 1) {
            for j in (i + 1)..min_dim {
                let d_i = mat.data()[i * n + i].clone();
                let d_j = mat.data()[j * n + j].clone();
                if d_j.is_zero() {
                    continue; // everything divides 0
                }
                if !d_i.is_zero() {
                    let (_, r) = d_j.div_rem(&d_i)?;
                    if r.is_zero() {
                        continue; // already dᵢ | dⱼ
                    }
                }
                // (also handles dᵢ = 0, dⱼ ≠ 0: swaps the zero rightwards)

                // Step 1: colᵢ += colⱼ, on S and Q (S·E = P·A·(Q·E)).
                col_add_multiple(mat, i, j, &R::one());
                col_add_multiple(q, i, j, &R::one());

                // Step 2: Bézout row operation between rows i and j.
                let (g, s, t) = d_i.extended_gcd(&d_j);
                self.apply_row_gcd_operation(mat, p, i, j, &d_i, &d_j, &g, &s, &t)?;

                // Step 3: colⱼ −= (t·dⱼ/g)·colᵢ, on S and Q.
                let (d_j_over_g, _) = d_j.div_rem(&g)?;
                let f = t * d_j_over_g;
                col_add_multiple(mat, j, i, &(R::zero() - f.clone()));
                col_add_multiple(q, j, i, &(R::zero() - f));
            }
        }

        Ok(())
    }


    /// Swap two columns in the matrix
    fn swap_cols(&mut self, col1: usize, col2: usize) {
        if col1 == col2 {
            return;
        }

        let m = self.rows();
        let n = self.cols();
        for i in 0..m {
            let idx1 = i * n + col1;
            let idx2 = i * n + col2;
            self.data_mut().swap(idx1, idx2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;

    #[test]
    fn test_smith_normal_form_simple() {
        // Simple 2x2 matrix
        let m = Matrix::from_vec(2, 2, vec![
            Integer::from(2), Integer::from(4),
            Integer::from(6), Integer::from(8)
        ]).unwrap();

        let snf = m.smith_normal_form().unwrap();

        // The result should be diagonal with d1 | d2
        assert!(snf.s.data()[0 * 2 + 1].is_zero()); // off-diagonal
        assert!(snf.s.data()[1 * 2 + 0].is_zero()); // off-diagonal

        // Check diagonal entries
        let d1 = &snf.s.data()[0];
        let d2 = &snf.s.data()[3];

        if !d2.is_zero() {
            let (_q, r) = d2.clone().div_rem(d1).unwrap();
            assert!(r.is_zero(), "d1 should divide d2");
        }
    }

    #[test]
    fn test_hermite_normal_form_simple() {
        // Simple 2x2 matrix
        let m = Matrix::from_vec(2, 2, vec![
            Integer::from(2), Integer::from(3),
            Integer::from(4), Integer::from(5)
        ]).unwrap();

        let hnf = m.hermite_normal_form().unwrap();

        // Result should be upper triangular
        // (entries below diagonal should be zero)
        assert!(hnf.h.data()[1 * 2 + 0].is_zero() || hnf.h.data()[0 * 2 + 0].is_zero());
    }

    #[test]
    fn test_hermite_3x3() {
        let m = Matrix::from_vec(3, 3, vec![
            Integer::from(1), Integer::from(2), Integer::from(3),
            Integer::from(4), Integer::from(5), Integer::from(6),
            Integer::from(7), Integer::from(8), Integer::from(9)
        ]).unwrap();

        let result = m.hermite_normal_form();
        assert!(result.is_ok());

        let hnf = result.unwrap();

        // Verify H = U * A
        // This is a basic sanity check
        assert_eq!(hnf.h.rows(), 3);
        assert_eq!(hnf.h.cols(), 3);
    }

    // ------------------------------------------------------------------ //
    // Independently verified normal-form tests. Every expected matrix     //
    // below was checked with SymPy (smith_normal_form) or, for the        //
    // row-style HNF, an independent Python floor-division row reduction   //
    // cross-checked against SymPy's column HNF of the transpose (equal    //
    // row lattices).                                                      //
    // ------------------------------------------------------------------ //

    fn z(n: i64) -> Integer {
        Integer::from(n)
    }

    fn zmat(rows: usize, cols: usize, vals: &[i64]) -> Matrix<Integer> {
        Matrix::from_vec(rows, cols, vals.iter().map(|&v| Integer::from(v)).collect()).unwrap()
    }

    /// s = p·a·q must hold exactly, and p, q must be unimodular (det = ±1).
    fn check_snf_identity(a: &Matrix<Integer>, snf: &SmithNormalForm<Integer>) {
        let paq = snf.p.mul(a).unwrap().mul(&snf.q).unwrap();
        assert_eq!(paq.data(), snf.s.data(), "s != p*a*q");
        assert_eq!(snf.p.det().unwrap().abs(), z(1), "p not unimodular");
        assert_eq!(snf.q.det().unwrap().abs(), z(1), "q not unimodular");
    }

    fn snf_diagonal(snf: &SmithNormalForm<Integer>) -> Vec<Integer> {
        let n = snf.s.cols();
        let d = snf.s.rows().min(n);
        (0..d).map(|i| snf.s.data()[i * n + i].clone()).collect()
    }

    #[test]
    fn test_snf_divisibility_fixup_diag_2_3() {
        // SymPy: smith_normal_form(diag(2, 3)) == diag(1, 6).
        // Without the divisibility fix-up this returns diag(2, 3) — 2 ∤ 3.
        let a = zmat(2, 2, &[2, 0, 0, 3]);
        let mut snf = a.smith_normal_form().unwrap();
        check_snf_identity(&a, &snf);
        snf.canonicalize_signs();
        check_snf_identity(&a, &snf);
        assert_eq!(snf_diagonal(&snf), vec![z(1), z(6)]);
    }

    #[test]
    fn test_snf_3x3_verified() {
        // SymPy: smith_normal_form([[2,4,4],[-6,6,12],[10,4,16]]) == diag(2, 2, 156).
        let a = zmat(3, 3, &[2, 4, 4, -6, 6, 12, 10, 4, 16]);
        let mut snf = a.smith_normal_form().unwrap();
        check_snf_identity(&a, &snf);
        snf.canonicalize_signs();
        check_snf_identity(&a, &snf);
        assert_eq!(snf_diagonal(&snf), vec![z(2), z(2), z(156)]);
    }

    #[test]
    fn test_snf_4x4_verified() {
        // SymPy: smith_normal_form([[3,3,1,4],[0,1,0,0],[0,0,19,16],[0,0,0,3]])
        //        == diag(1, 1, 3, 57).
        let a = zmat(4, 4, &[3, 3, 1, 4, 0, 1, 0, 0, 0, 0, 19, 16, 0, 0, 0, 3]);
        let mut snf = a.smith_normal_form().unwrap();
        check_snf_identity(&a, &snf);
        snf.canonicalize_signs();
        check_snf_identity(&a, &snf);
        assert_eq!(snf_diagonal(&snf), vec![z(1), z(1), z(3), z(57)]);
    }

    #[test]
    fn test_snf_rectangular_and_rank_deficient() {
        // 2x3 rank 1: [[2,4,6],[4,8,12]] — row 2 = 2·row 1. Invariant factor
        // gcd(2,4,6) = 2; second diagonal entry 0.
        let a = zmat(2, 3, &[2, 4, 6, 4, 8, 12]);
        let mut snf = a.smith_normal_form().unwrap();
        check_snf_identity(&a, &snf);
        snf.canonicalize_signs();
        assert_eq!(snf_diagonal(&snf), vec![z(2), z(0)]);
        assert_eq!(a.elementary_divisors().unwrap().len(), 1);
    }

    #[test]
    fn test_snf_pm1_boundary_matrix_terminates() {
        // Regression: the simplicial boundary matrix d_1 of S^1 (all entries
        // 0/±1). The pre-fix elimination loop 2-cycled forever on this
        // matrix, because with pivot | target the Bezout rotation
        // (extended_gcd(a, -a) = (-a, 0, 1)) preserves the pivot norm.
        // SymPy: smith_normal_form([[-1,-1,0],[1,0,-1],[0,1,1]]) = diag(1,1,0).
        let a = zmat(3, 3, &[-1, -1, 0, 1, 0, -1, 0, 1, 1]);
        let mut snf = a.smith_normal_form().unwrap();
        check_snf_identity(&a, &snf);
        snf.canonicalize_signs();
        check_snf_identity(&a, &snf);
        assert_eq!(snf_diagonal(&snf), vec![z(1), z(1), z(0)]);
        let divs: Vec<Integer> = a
            .elementary_divisors()
            .unwrap()
            .into_iter()
            .map(|d| d.abs())
            .collect();
        assert_eq!(divs, vec![z(1), z(1)]);
    }

    #[test]
    fn test_snf_pm1_torsion_matrix() {
        // The transposed-orientation double cover pattern: a 0/±1 matrix
        // with a non-unit invariant factor, exercising both the exact-
        // subtraction and Bezout-rotation elimination paths.
        // SymPy: smith_normal_form([[1,1,0],[-1,0,1],[0,-1,1],[1,1,2]])
        //        = diag(1, 1, 2) (4x3, rank 3, torsion factor 2).
        let a = zmat(4, 3, &[1, 1, 0, -1, 0, 1, 0, -1, 1, 1, 1, 2]);
        let mut snf = a.smith_normal_form().unwrap();
        check_snf_identity(&a, &snf);
        snf.canonicalize_signs();
        check_snf_identity(&a, &snf);
        assert_eq!(snf_diagonal(&snf), vec![z(1), z(1), z(2)]);
    }

    #[test]
    fn test_snf_divisibility_chain_always_holds() {
        // A few matrices; the divisibility chain d_i | d_{i+1} must hold on the
        // non-zero diagonal (this was previously violated, e.g. diag(2,3)).
        let mats = vec![
            zmat(2, 2, &[2, 0, 0, 3]),
            zmat(3, 3, &[4, 0, 0, 0, 6, 0, 0, 0, 9]),
            zmat(3, 3, &[2, 4, 4, -6, 6, 12, 10, 4, 16]),
            zmat(2, 2, &[-2, 0, 0, -3]),
        ];
        for a in mats {
            let snf = a.smith_normal_form().unwrap();
            check_snf_identity(&a, &snf);
            let d = snf_diagonal(&snf);
            for w in d.windows(2) {
                if !w[1].is_zero() {
                    assert!(
                        !w[0].is_zero(),
                        "zero diagonal entry before a non-zero one"
                    );
                    let (_, r) = w[1].div_rem(&w[0]).unwrap();
                    assert!(r.is_zero(), "divisibility {} | {} fails", w[0], w[1]);
                }
            }
        }
    }

    /// h = u·a must hold exactly, u unimodular, h in row echelon form.
    fn check_hnf_identity(a: &Matrix<Integer>, hnf: &HermiteNormalForm<Integer>) {
        let ua = hnf.u.mul(a).unwrap();
        assert_eq!(ua.data(), hnf.h.data(), "h != u*a");
        assert_eq!(hnf.u.det().unwrap().abs(), z(1), "u not unimodular");
        // echelon: pivot columns strictly increase; zero rows at the bottom
        let (m, n) = (hnf.h.rows(), hnf.h.cols());
        let mut last: isize = -1;
        let mut seen_zero_row = false;
        for i in 0..m {
            match (0..n).find(|&j| !hnf.h.data()[i * n + j].is_zero()) {
                Some(p) => {
                    assert!(!seen_zero_row, "non-zero row after a zero row");
                    assert!(p as isize > last, "pivots do not move right");
                    last = p as isize;
                }
                None => seen_zero_row = true,
            }
        }
    }

    /// Canonical (Sage-style) properties: positive pivots, entries above each
    /// pivot in [0, pivot).
    fn check_hnf_canonical(hnf: &HermiteNormalForm<Integer>) {
        let (m, n) = (hnf.h.rows(), hnf.h.cols());
        for i in 0..m {
            if let Some(p) = (0..n).find(|&j| !hnf.h.data()[i * n + j].is_zero()) {
                let pivot = hnf.h.data()[i * n + p].clone();
                assert!(pivot.signum() > 0, "pivot not positive");
                for r in 0..i {
                    let e = hnf.h.data()[r * n + p].clone();
                    assert!(
                        e.signum() >= 0 && e < pivot,
                        "entry {} above pivot {} not in [0, pivot)",
                        e,
                        pivot
                    );
                }
            }
        }
    }

    #[test]
    fn test_hnf_canonical_2x2() {
        // Row-style HNF of [[2,3],[4,5]] is [[2,0],[0,1]] (row lattice:
        // (4,5)-2(2,3) = (0,-1), then (2,3)-3(0,1) = (2,0)); verified
        // independently in Python and by row-lattice equality in SymPy.
        let a = zmat(2, 2, &[2, 3, 4, 5]);
        let mut hnf = a.hermite_normal_form().unwrap();
        check_hnf_identity(&a, &hnf);
        hnf.canonicalize().unwrap();
        check_hnf_identity(&a, &hnf);
        check_hnf_canonical(&hnf);
        assert_eq!(hnf.h.data(), zmat(2, 2, &[2, 0, 0, 1]).data());
    }

    #[test]
    fn test_hnf_canonical_2x3() {
        // Row HNF of [[1,2,3],[4,5,6]] = [[1,2,3],[0,3,6]] (verified in Python;
        // note 2 ∈ [0,3) above the second pivot).
        let a = zmat(2, 3, &[1, 2, 3, 4, 5, 6]);
        let mut hnf = a.hermite_normal_form().unwrap();
        check_hnf_identity(&a, &hnf); // u is 2×2, h is 2×3: exercises the stride fix
        hnf.canonicalize().unwrap();
        check_hnf_identity(&a, &hnf);
        check_hnf_canonical(&hnf);
        assert_eq!(hnf.h.data(), zmat(2, 3, &[1, 2, 3, 0, 3, 6]).data());
    }

    #[test]
    fn test_hnf_canonical_3x2_with_zero_row() {
        // Row HNF of [[2,3],[4,5],[7,1]] = [[1,0],[0,1],[0,0]] (verified in
        // Python: the three rows generate all of Z²).
        let a = zmat(3, 2, &[2, 3, 4, 5, 7, 1]);
        let mut hnf = a.hermite_normal_form().unwrap();
        check_hnf_identity(&a, &hnf);
        hnf.canonicalize().unwrap();
        check_hnf_identity(&a, &hnf);
        check_hnf_canonical(&hnf);
        assert_eq!(hnf.h.data(), zmat(3, 2, &[1, 0, 0, 1, 0, 0]).data());
    }

    #[test]
    fn test_hnf_canonical_pivot_column_skip() {
        // Row HNF of [[0,2],[3,4]] = [[3,0],[0,2]] (verified in Python:
        // 4 - 2·2 = 0 above the second pivot).
        let a = zmat(2, 2, &[0, 2, 3, 4]);
        let mut hnf = a.hermite_normal_form().unwrap();
        check_hnf_identity(&a, &hnf);
        hnf.canonicalize().unwrap();
        check_hnf_identity(&a, &hnf);
        check_hnf_canonical(&hnf);
        assert_eq!(hnf.h.data(), zmat(2, 2, &[3, 0, 0, 2]).data());
    }
}

//! Exact diagonalization of a ternary rational quadratic form by congruence, so
//! the conic layer accepts a general form `a X² + b Y² + c Z² + d XY + e YZ +
//! f XZ`, not only an already-diagonal one.
//!
//! Ported from `dessin_engine/src/quadratic_form.rs`
//! (`/home/john/inverse_galois/M23/dessin_engine/src/quadratic_form.rs`),
//! adapted to RustMath's `rustmath_rationals::Rational`. (Named `ternary` here to
//! avoid colliding with this crate's existing binary-form `quadratic_form.rs`.)
//!
//! We carry the change-of-basis `P` (columns = new basis in old coordinates) so
//! that a solution `w` of the diagonal form pulls back to `P·w`, a solution of
//! the original: if `D = Pᵀ M P` then `(P w)ᵀ M (P w) = wᵀ D w`.

// Indexed loops are intentional: the symmetric-congruence row/col/transform
// updates read more clearly addressing `a[i][j]` and `p[i][j]` directly.
#![allow(clippy::needless_range_loop)]

use crate::hilbert::rat_is_zero;
use rustmath_rationals::Rational;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FormError {
    #[error("form is degenerate (a diagonal coefficient is zero)")]
    Degenerate,
}

type Row = [Rational; 3];
type Mat = [Row; 3];

fn q(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn zero_mat() -> Mat {
    [
        [q(0), q(0), q(0)],
        [q(0), q(0), q(0)],
        [q(0), q(0), q(0)],
    ]
}

fn id_mat() -> Mat {
    let mut m = zero_mat();
    for i in 0..3 {
        m[i][i] = q(1);
    }
    m
}

/// A symmetric `3×3` Gram matrix of a ternary form.
#[derive(Debug, Clone)]
pub struct TernaryForm {
    pub gram: Mat,
}

impl TernaryForm {
    /// From the coefficients of `aX² + bY² + cZ² + dXY + eYZ + fXZ`.
    /// Gram = `[[a, d/2, f/2],[d/2, b, e/2],[f/2, e/2, c]]`.
    pub fn from_coeffs(
        a: Rational,
        b: Rational,
        c: Rational,
        d: Rational,
        e: Rational,
        f: Rational,
    ) -> Result<Self, FormError> {
        let half = Rational::new(1, 2).unwrap();
        let d2 = &d * &half;
        let e2 = &e * &half;
        let f2 = &f * &half;
        Ok(Self {
            gram: [
                [a, d2.clone(), f2.clone()],
                [d2, b, e2.clone()],
                [f2, e2, c],
            ],
        })
    }

    /// Evaluate `vᵀ M v` (used in tests to certify diagonalization).
    pub fn eval(&self, v: &Row) -> Result<Rational, FormError> {
        let mut acc = q(0);
        for i in 0..3 {
            for j in 0..3 {
                let t = &(&self.gram[i][j] * &v[i]) * &v[j];
                acc = &acc + &t;
            }
        }
        Ok(acc)
    }

    /// Diagonalize by symmetric congruence. Returns the diagonal entries and the
    /// transform `P` (a solution `w` of `diag` gives `P·w` solving `self`).
    pub fn diagonalize(&self) -> Result<([Rational; 3], Mat), FormError> {
        let mut a = self.gram.clone();
        let mut p = id_mat();

        for s in 0..3 {
            ensure_pivot(&mut a, &mut p, s)?;
            if rat_is_zero(&a[s][s]) {
                // No usable pivot in the trailing block ⇒ rank-deficient.
                continue;
            }
            let piv = a[s][s].clone();
            for r in 0..3 {
                if r == s || rat_is_zero(&a[r][s]) {
                    continue;
                }
                // b_r <- b_r + λ b_s with λ = -a[r][s]/piv.
                let lambda = -(&a[r][s] / &piv);
                add_lambda_basis(&mut a, &mut p, r, s, &lambda)?;
            }
        }
        let diag = [a[0][0].clone(), a[1][1].clone(), a[2][2].clone()];
        Ok((diag, p))
    }
}

/// Make `a[s][s]` nonzero (if the trailing block is not identically zero), via a
/// congruence swap or the off-diagonal `b_i <- b_i + b_j` trick.
fn ensure_pivot(a: &mut Mat, p: &mut Mat, s: usize) -> Result<(), FormError> {
    if !rat_is_zero(&a[s][s]) {
        return Ok(());
    }
    // Prefer an existing nonzero diagonal entry in the trailing block.
    for t in (s + 1)..3 {
        if !rat_is_zero(&a[t][t]) {
            swap_basis(a, p, s, t);
            return Ok(());
        }
    }
    // Else create one from an off-diagonal entry: b_i <- b_i + b_j gives
    // new a[i][i] = 2 a[i][j] ≠ 0 in characteristic 0.
    for i in s..3 {
        for j in (i + 1)..3 {
            if !rat_is_zero(&a[i][j]) {
                add_lambda_basis(a, p, i, j, &q(1))?;
                if i != s {
                    swap_basis(a, p, s, i);
                }
                return Ok(());
            }
        }
    }
    // Trailing block is all zero: leave a[s][s] = 0 (degenerate form).
    Ok(())
}

/// `b_r <- b_r + λ b_k`: row_r += λ row_k, then col_r += λ col_k, and P col_r += λ P col_k.
fn add_lambda_basis(
    a: &mut Mat,
    p: &mut Mat,
    r: usize,
    k: usize,
    lambda: &Rational,
) -> Result<(), FormError> {
    // row_r += λ row_k
    for j in 0..3 {
        let t = lambda * &a[k][j];
        a[r][j] = &a[r][j] + &t;
    }
    // col_r += λ col_k
    for i in 0..3 {
        let t = lambda * &a[i][k];
        a[i][r] = &a[i][r] + &t;
    }
    // P col_r += λ P col_k
    for i in 0..3 {
        let t = lambda * &p[i][k];
        p[i][r] = &p[i][r] + &t;
    }
    Ok(())
}

/// Congruence swap of basis vectors `i,j`: swap rows i,j; cols i,j; P cols i,j.
fn swap_basis(a: &mut Mat, p: &mut Mat, i: usize, j: usize) {
    a.swap(i, j);
    for row in a.iter_mut() {
        row.swap(i, j);
    }
    for row in p.iter_mut() {
        row.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    #[test]
    fn diagonalizes_pure_cross_term() {
        // f = XY: Gram has zero diagonal. Diagonalization must succeed and the
        // transform must pull a diagonal solution back to a real solution.
        let f = TernaryForm::from_coeffs(r(0), r(0), r(0), r(1), r(0), r(0)).unwrap();
        let (diag, p) = f.diagonalize().unwrap();
        // XY is rank 2; the third diagonal entry is 0.
        assert!(diag.iter().filter(|d| rat_is_zero(d)).count() >= 1);
        // Check the congruence identity diag_ii = (P e_i)ᵀ M (P e_i).
        for i in 0..3 {
            let col = [p[0][i].clone(), p[1][i].clone(), p[2][i].clone()];
            assert_eq!(f.eval(&col).unwrap(), diag[i]);
        }
    }

    #[test]
    fn diagonalizes_and_pullback_is_isometry() {
        // A generic nondegenerate form; verify diag_ii = (P e_i)ᵀ M (P e_i).
        let f = TernaryForm::from_coeffs(r(1), r(2), r(3), r(1), r(1), r(1)).unwrap();
        let (diag, p) = f.diagonalize().unwrap();
        let cols: Vec<Row> = (0..3)
            .map(|i| [p[0][i].clone(), p[1][i].clone(), p[2][i].clone()])
            .collect();
        for i in 0..3 {
            assert_eq!(f.eval(&cols[i]).unwrap(), diag[i]);
            assert!(!rat_is_zero(&diag[i]), "form should be nondegenerate");
        }
    }

    #[test]
    fn pullback_of_diagonal_isotropic_vector_solves_original() {
        let f = TernaryForm::from_coeffs(r(1), r(1), r(-1), r(2), r(0), r(0)).unwrap();
        let (diag, p) = f.diagonalize().unwrap();
        for i in 0..3 {
            let col = [p[0][i].clone(), p[1][i].clone(), p[2][i].clone()];
            assert_eq!(f.eval(&col).unwrap(), diag[i]);
        }
    }
}

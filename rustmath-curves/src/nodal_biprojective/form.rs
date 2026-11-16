//! Bihomogeneous forms over `Z` and mod p (MAGMA ch. 114: curves in a product of projective
//! lines are given by one bihomogeneous equation). `coeffs[i][j]` is the coefficient of
//! `X^i Y^j` of the affine chart of the form of bidegree `(dx, dy)`, `0 <= i <= dx`,
//! `0 <= j <= dy`.

use super::{check_prime, CountError};
use rustmath_integers::Integer;
use rustmath_polynomials::zx;

/// A bihomogeneous form `F(X, Y) = sum F[i][j] X^i Y^j` over `Z` of bidegree `(dx, dy)`; the
/// curve `V(F)` in `P^1 x P^1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BihomogeneousForm {
    coeffs: Vec<Vec<Integer>>,
}

fn check_shape<T>(rows: &[Vec<T>]) -> Result<(usize, usize), CountError> {
    if rows.is_empty() || rows[0].is_empty() {
        return Err(CountError::Shape("empty coefficient array".into()));
    }
    let w = rows[0].len();
    if rows.iter().any(|r| r.len() != w) {
        return Err(CountError::Shape("rows of different lengths".into()));
    }
    if rows.len() > 4097 || w > 4097 {
        return Err(CountError::Shape("degree above 4096".into()));
    }
    Ok((rows.len() - 1, w - 1))
}

impl BihomogeneousForm {
    /// From `coeffs[i][j]` (coefficient of `X^i Y^j`). The array must be rectangular and the form
    /// nonzero; its dimensions fix the bidegree.
    pub fn new(coeffs: Vec<Vec<Integer>>) -> Result<Self, CountError> {
        check_shape(&coeffs)?;
        if coeffs.iter().all(|r| r.iter().all(|c| c.is_zero())) {
            return Err(CountError::ZeroForm);
        }
        Ok(BihomogeneousForm { coeffs })
    }

    /// From decimal strings (`rows[i][j]`, the coefficient of `X^i Y^j`).
    pub fn from_decimal_rows(rows: &[Vec<String>]) -> Result<Self, CountError> {
        let mut coeffs = Vec::with_capacity(rows.len());
        for r in rows {
            let mut row = Vec::with_capacity(r.len());
            for s in r {
                row.push(
                    Integer::from_decimal_str(s.trim())
                        .ok_or_else(|| CountError::Shape(format!("not an integer: {s}")))?,
                );
            }
            coeffs.push(row);
        }
        Self::new(coeffs)
    }

    /// The coefficient array.
    pub fn coeffs(&self) -> &[Vec<Integer>] {
        &self.coeffs
    }

    /// `(dx, dy)`.
    pub fn bidegree(&self) -> (usize, usize) {
        (self.coeffs.len() - 1, self.coeffs[0].len() - 1)
    }

    /// The arithmetic genus `(dx - 1)(dy - 1)` of a curve of bidegree `(dx, dy)` with
    /// `dx, dy >= 1` (adjunction on `P^1 x P^1`).
    pub fn arithmetic_genus(&self) -> usize {
        let (dx, dy) = self.bidegree();
        dx.saturating_sub(1) * dy.saturating_sub(1)
    }

    /// The content (gcd of the coefficients, positive).
    pub fn content(&self) -> Integer {
        let mut g = Integer::zero();
        for r in &self.coeffs {
            for c in r {
                g = g.gcd(c);
            }
        }
        g
    }

    /// `F / content(F)`.
    pub fn primitive_part(&self) -> Self {
        let c = self.content();
        BihomogeneousForm {
            coeffs: self
                .coeffs
                .iter()
                .map(|r| r.iter().map(|x| x.clone() / c.clone()).collect())
                .collect(),
        }
    }

    /// The form with the roles of `X` and `Y` exchanged.
    pub fn transpose(&self) -> Self {
        let (dx, dy) = self.bidegree();
        let mut coeffs = vec![vec![Integer::zero(); dx + 1]; dy + 1];
        for (i, r) in self.coeffs.iter().enumerate() {
            for (j, c) in r.iter().enumerate() {
                coeffs[j][i] = c.clone();
            }
        }
        BihomogeneousForm { coeffs }
    }

    /// `F(X, (aY + b)/(cY + d)) (cY + d)^dy`, divided by its (positive) content. For
    /// `ad - bc != 0` this is the same curve after an automorphism of the `Y`-line over `Q`.
    pub fn mobius_y(&self, a: i64, b: i64, c: i64, d: i64) -> Result<Self, CountError> {
        if (a as i128) * (d as i128) - (b as i128) * (c as i128) == 0 {
            return Err(CountError::Shape("Moebius map with ad - bc = 0".into()));
        }
        let (_, dy) = self.bidegree();
        let num = [Integer::from(b), Integer::from(a)];
        let den = [Integer::from(d), Integer::from(c)];
        // pw_num[j] = (aY + b)^j, pw_den[j] = (cY + d)^j
        let mut pw_num = vec![vec![Integer::one()]];
        let mut pw_den = vec![vec![Integer::one()]];
        for j in 1..=dy {
            pw_num.push(zx::mul(&pw_num[j - 1], &num));
            pw_den.push(zx::mul(&pw_den[j - 1], &den));
        }
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for r in &self.coeffs {
            let mut acc = vec![Integer::zero(); dy + 1];
            for (j, cij) in r.iter().enumerate() {
                if cij.is_zero() {
                    continue;
                }
                let t = zx::scalar_mul(&zx::mul(&pw_num[j], &pw_den[dy - j]), cij);
                for (l, v) in t.into_iter().enumerate() {
                    acc[l] = acc[l].clone() + v;
                }
            }
            coeffs.push(acc);
        }
        Ok(BihomogeneousForm::new(coeffs)?.primitive_part())
    }

    fn with_positive_leading(self) -> Self {
        let first = self
            .coeffs
            .iter()
            .flat_map(|r| r.iter())
            .find(|c| !c.is_zero())
            .map(|c| c.signum())
            .unwrap_or(1);
        if first < 0 {
            BihomogeneousForm {
                coeffs: self
                    .coeffs
                    .iter()
                    .map(|r| r.iter().map(|x| -x.clone()).collect())
                    .collect(),
            }
        } else {
            self
        }
    }

    /// True iff `other = lambda self` for a nonzero rational `lambda` (same bidegree).
    pub fn equals_up_to_scalar(&self, other: &Self) -> bool {
        if self.bidegree() != other.bidegree() {
            return false;
        }
        let a = self.primitive_part().with_positive_leading();
        let b = other.primitive_part().with_positive_leading();
        a == b
    }

    /// True iff the content is 1 (a hypothesis of Lemma 9.1 of the owner's paper, johnjanik/M23
    /// at `2d11038`, `paper/golay_m23.tex`, `\label{lem:reduction}`).
    pub fn is_primitive(&self) -> bool {
        self.content() == Integer::one()
    }

    /// The reduction mod p of a primitive form: `Err(NotPrimitive { content })` unless the
    /// content is 1, otherwise as [`BihomogeneousForm::reduce`]. Use it for the certificate of
    /// [`certify`](fn@super::certify), whose Lemma 9.1 needs `F` primitive.
    pub fn reduce_primitive(&self, p: u64) -> Result<FormModP, CountError> {
        let content = self.content();
        if content != Integer::one() {
            return Err(CountError::NotPrimitive { content });
        }
        self.reduce(p)
    }

    /// The reduction mod p, an odd prime below `2^16`. It does not check that the form is
    /// primitive ([`BihomogeneousForm::reduce_primitive`] does).
    pub fn reduce(&self, p: u64) -> Result<FormModP, CountError> {
        check_prime(p)?;
        let pi = Integer::from(p);
        let coeffs = self
            .coeffs
            .iter()
            .map(|r| {
                r.iter()
                    .map(|c| {
                        let m = c % &pi;
                        let v = m.to_i64().expect("residue fits i64");
                        v.rem_euclid(p as i64) as u64
                    })
                    .collect()
            })
            .collect();
        FormModP::new(p, coeffs)
    }
}

/// A bihomogeneous form over `F_p`, `p` an odd prime below `2^16`, coefficients in `[0, p)`.
///
/// For [`certify`](fn@super::certify) and
/// [`normalization_point_counts`](super::normalization_point_counts), the hypotheses of Lemma 9.1
/// (owner's paper, `\label{lem:reduction}`) that this type cannot check are the caller's: the
/// form is the reduction of a primitive form of the same bidegree (over `Z`, or over the ring of
/// integers of a number field at a prime of residue degree one), whose zero set over the
/// generic fibre is birational to the curve of the genus passed in. Locally at p primitivity is
/// `F mod p != 0`, which [`FormModP::new`] checks (`ZeroForm`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormModP {
    p: u64,
    coeffs: Vec<Vec<u64>>,
}

impl FormModP {
    /// From `coeffs[i][j]` (read modulo `p`). Errors: `p` not an odd prime below `2^16`, a
    /// non-rectangular array, the zero form.
    pub fn new(p: u64, coeffs: Vec<Vec<u64>>) -> Result<Self, CountError> {
        check_prime(p)?;
        check_shape(&coeffs)?;
        let coeffs: Vec<Vec<u64>> = coeffs
            .into_iter()
            .map(|r| r.into_iter().map(|c| c % p).collect())
            .collect();
        if coeffs.iter().all(|r| r.iter().all(|&c| c == 0)) {
            return Err(CountError::ZeroForm);
        }
        Ok(FormModP { p, coeffs })
    }

    /// The prime.
    pub fn p(&self) -> u64 {
        self.p
    }

    /// The coefficient array, `coeffs[i][j]` of `X^i Y^j`.
    pub fn coeffs(&self) -> &[Vec<u64>] {
        &self.coeffs
    }

    /// `(dx, dy)`.
    pub fn bidegree(&self) -> (usize, usize) {
        (self.coeffs.len() - 1, self.coeffs[0].len() - 1)
    }

    /// `(dx - 1)(dy - 1)`.
    pub fn arithmetic_genus(&self) -> usize {
        let (dx, dy) = self.bidegree();
        dx.saturating_sub(1) * dy.saturating_sub(1)
    }

    /// The form with `X` and `Y` exchanged.
    pub fn transpose(&self) -> Self {
        let (dx, dy) = self.bidegree();
        let mut coeffs = vec![vec![0u64; dx + 1]; dy + 1];
        for (i, r) in self.coeffs.iter().enumerate() {
            for (j, &c) in r.iter().enumerate() {
                coeffs[j][i] = c;
            }
        }
        FormModP { p: self.p, coeffs }
    }

    /// `F(X, 1/Y) Y^dy`: the chart at `Y = infinity`.
    pub fn reverse_y(&self) -> Self {
        FormModP {
            p: self.p,
            coeffs: self
                .coeffs
                .iter()
                .map(|r| r.iter().rev().cloned().collect())
                .collect(),
        }
    }

    /// `F(1/X, Y) X^dx`: the chart at `X = infinity`.
    pub fn reverse_x(&self) -> Self {
        FormModP {
            p: self.p,
            coeffs: self.coeffs.iter().rev().cloned().collect(),
        }
    }
}

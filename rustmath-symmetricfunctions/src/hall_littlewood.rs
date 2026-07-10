//! Hall-Littlewood symmetric functions `P_lambda(x; t)` and `Q_lambda(x; t)`.
//!
//! Follows Macdonald, *Symmetric Functions and Hall Polynomials*, 2nd ed.,
//! Chapter III. The Hall-Littlewood `P_lambda` are the unique family with
//!
//! 1. `P_lambda = m_lambda + sum_{mu < lambda} u_{lambda mu}(t) m_mu`
//!    (strictly dominance-lower monomial terms), and
//! 2. orthogonality w.r.t. the t-deformed Hall inner product, defined on power
//!    sums by `<p_rho, p_sigma>_t = delta_{rho sigma} z_rho prod_i 1/(1 - t^{rho_i})`
//!    (Macdonald III (4.11)).
//!
//! `Q_lambda = b_lambda(t) P_lambda` with `b_lambda(t) = prod_{i>=1} phi_{m_i(lambda)}(t)`,
//! `phi_r(t) = (1-t)(1-t^2)...(1-t^r)` (Macdonald III (2.11)-(2.12)); equivalently
//! `b_lambda = 1 / <P_lambda, P_lambda>_t`.
//!
//! # Coefficient representation (design decision)
//!
//! Coefficients are computed in the exact rational function field `Q(t)`
//! ([`TRat`] = [`RatFunc<Rational>`]) because Gram-Schmidt divides by norms.
//! By Macdonald III (2.7)/(4.9) the *final* monomial-expansion coefficients of
//! `P_lambda` and `Q_lambda` lie in `Z[t]`; the per-partition constructors
//! [`hall_littlewood_p`] / [`hall_littlewood_q`] therefore return coefficients
//! as *polynomials in t with exact `Rational` coefficients*
//! ([`Poly<Rational>`], low-degree-first), and return an honest error if a
//! computed coefficient failed to be polynomial (mathematically impossible;
//! never silently truncated). The matrix-level API keeps the `Q(t)` form.
//!
//! # Algorithm
//!
//! Sequential Gram-Schmidt over `Q(t)` on the monomial basis, processed along a
//! linear extension of dominance order (ascending: `(1^n)` first, `(n)` last).
//! Given that an orthogonal, dominance-unitriangular family exists (Macdonald
//! III (2.6) + (4.9)), Gram-Schmidt along *any* linear extension reproduces
//! exactly that family: by induction, `m_lambda - P_lambda` lies in the span of
//! the already-processed `P_mu`, and subtracting the orthogonal projection
//! yields `P_lambda` itself. The internal `partitions_order(n)` (descending
//! lexicographic) refines dominance, so its reverse is a valid processing order.
//!
//! The construction is fully generic in `n = |lambda|` (no artificial rank
//! cap); it is exhaustively verified against independently computed tables for
//! all `|lambda| <= 4` in the tests. Cost grows with the number of partitions
//! of `n` (Gram-Schmidt is `O(p(n)^2)` inner products of length `p(n)`).

use crate::classical_bases::{
    centralizer_size, partitions_order, schur_to_monomial_matrix, transition_matrix,
    ClassicalBasis,
};
use crate::ratfunc::{Poly, RatFunc};
use rustmath_combinatorics::Partition;
use rustmath_core::{Field, MathError, Result, Ring};
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// The exact rational function field `Q(t)`.
pub type TRat = RatFunc<Rational>;

/// Embed a `Rational` constant into `Q(t)`.
pub fn t_constant(r: Rational) -> TRat {
    RatFunc::constant(r)
}

/// The generator `t` of `Q(t)`.
pub fn t_var() -> TRat {
    RatFunc::var()
}

/// The polynomial `1 - t^r` in `Q[t]`.
fn one_minus_t_pow(r: usize) -> Poly<Rational> {
    Poly::one().sub(&Poly::var_pow(r))
}

/// `phi_r(t) = (1 - t)(1 - t^2) ... (1 - t^r)` (Macdonald III (2.12); `phi_0 = 1`).
pub fn phi_polynomial(r: usize) -> Poly<Rational> {
    let mut acc = Poly::one();
    for j in 1..=r {
        acc = acc.mul(&one_minus_t_pow(j));
    }
    acc
}

/// `b_lambda(t) = prod_{i >= 1} phi_{m_i(lambda)}(t)` where `m_i` is the
/// multiplicity of the part `i` (Macdonald III (2.12)). Satisfies
/// `Q_lambda = b_lambda P_lambda` and `<P_lambda, P_lambda>_t = 1/b_lambda`.
pub fn hl_b_polynomial(lambda: &Partition) -> Poly<Rational> {
    let mut mult: HashMap<usize, usize> = HashMap::new();
    for &part in lambda.parts() {
        *mult.entry(part).or_insert(0) += 1;
    }
    let mut acc = Poly::one();
    for (_, m) in mult {
        acc = acc.mul(&phi_polynomial(m));
    }
    acc
}

/// The Hall-Littlewood weight `prod_i 1/(1 - t^{rho_i})` on the power-sum
/// diagonal (the full diagonal entry is `z_rho` times this).
fn hall_weight(rho: &Partition) -> TRat {
    let mut den = Poly::one();
    for &r in rho.parts() {
        den = den.mul(&one_minus_t_pow(r));
    }
    RatFunc::new(Poly::one(), den).expect("1 - t^r is nonzero in Q[t]")
}

// ---------------------------------------------------------------------------
// Generic deformed Gram-Schmidt engine (shared with the Macdonald basis)
// ---------------------------------------------------------------------------

fn lift_matrix<F: Field>(m: &Matrix<Rational>, embed: &dyn Fn(&Rational) -> F) -> Result<Matrix<F>> {
    let (rows, cols) = (m.rows(), m.cols());
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(embed(m.get(i, j)?));
        }
    }
    Matrix::from_vec(rows, cols, data)
}

/// Build the `P -> monomial` transition matrix at degree `n` for the family
/// orthogonal w.r.t. the deformed inner product
/// `<p_rho, p_sigma> = delta z_rho * weight(rho)`, unitriangular w.r.t.
/// dominance order in the monomial basis.
///
/// Rows and columns are indexed by [`partitions_order`]`(n)`; entry `(i, j)` is
/// the coefficient of `m_{parts[j]}` in `P_{parts[i]}`.
pub(crate) fn deformed_p_to_monomial_matrix<F: Field>(
    n: usize,
    weight: &dyn Fn(&Partition) -> F,
    embed: &dyn Fn(&Rational) -> F,
) -> Result<Matrix<F>> {
    let parts = partitions_order(n);
    let k = parts.len();

    // m_lambda expanded in power sums (exact rational engine from classical_bases).
    let m2p = lift_matrix(
        &transition_matrix(ClassicalBasis::Monomial, ClassicalBasis::PowerSum, n),
        embed,
    )?;
    let p2m = lift_matrix(
        &transition_matrix(ClassicalBasis::PowerSum, ClassicalBasis::Monomial, n),
        embed,
    )?;

    // Diagonal of the deformed inner product in the power-sum basis.
    let mut w = Vec::with_capacity(k);
    for rho in &parts {
        let z = embed(&Rational::from(centralizer_size(rho)));
        w.push(z * weight(rho));
    }

    let ip = |a: &[F], b: &[F]| -> F {
        let mut acc = F::zero();
        for i in 0..k {
            if a[i].is_zero() || b[i].is_zero() {
                continue;
            }
            acc = acc + a[i].clone() * b[i].clone() * w[i].clone();
        }
        acc
    };

    // Gram-Schmidt in ascending dominance order = reversed partitions_order
    // (descending lex refines dominance; see module docs).
    let mut pvecs: Vec<Option<Vec<F>>> = vec![None; k];
    for i in (0..k).rev() {
        let mut v: Vec<F> = (0..k)
            .map(|j| m2p.get(i, j).map(|x| x.clone()))
            .collect::<Result<Vec<F>>>()?;
        for prev in pvecs.iter().skip(i + 1) {
            let pj = prev.as_ref().expect("processed in reverse order");
            let num = ip(&v, pj);
            if num.is_zero() {
                continue;
            }
            let norm = ip(pj, pj);
            let c = num
                * norm.inverse().map_err(|_| {
                    MathError::NumericalError(
                        "deformed Gram-Schmidt: vanishing norm (inner product degenerate)"
                            .to_string(),
                    )
                })?;
            for l in 0..k {
                v[l] = v[l].clone() - c.clone() * pj[l].clone();
            }
        }
        pvecs[i] = Some(v);
    }

    // Convert P rows from power-sum coordinates back to monomial coordinates.
    let mut data = Vec::with_capacity(k * k);
    for i in 0..k {
        let v = pvecs[i].as_ref().expect("all rows processed");
        for mu in 0..k {
            let mut acc = F::zero();
            for rho in 0..k {
                if v[rho].is_zero() {
                    continue;
                }
                acc = acc + v[rho].clone() * p2m.get(rho, mu)?.clone();
            }
            data.push(acc);
        }
    }
    Matrix::from_vec(k, k, data)
}

// ---------------------------------------------------------------------------
// Public Hall-Littlewood API
// ---------------------------------------------------------------------------

/// The `HallLittlewoodP -> Monomial` transition matrix at weight `n` over `Q(t)`:
/// entry `(i, j)` is the coefficient of `m_{parts[j]}` in `P_{parts[i]}(x; t)`,
/// with rows/columns indexed by [`partitions_order`]`(n)`. The entries are
/// provably polynomials in `t` with integer coefficients.
pub fn hall_littlewood_p_to_monomial_matrix(n: usize) -> Result<Matrix<TRat>> {
    deformed_p_to_monomial_matrix(n, &hall_weight, &|r| t_constant(r.clone()))
}

/// The `HallLittlewoodQ -> Monomial` transition matrix: row `lambda` of the `P`
/// matrix scaled by `b_lambda(t)`.
pub fn hall_littlewood_q_to_monomial_matrix(n: usize) -> Result<Matrix<TRat>> {
    let p = hall_littlewood_p_to_monomial_matrix(n)?;
    let parts = partitions_order(n);
    let k = parts.len();
    let mut data = Vec::with_capacity(k * k);
    for (i, lam) in parts.iter().enumerate() {
        let b = RatFunc::from_poly(hl_b_polynomial(lam));
        for j in 0..k {
            data.push(b.clone() * p.get(i, j)?.clone());
        }
    }
    Matrix::from_vec(k, k, data)
}

/// The `HallLittlewoodP -> basis` transition matrix over `Q(t)` for any
/// classical target basis, composed as `(P -> m) * (m -> basis)`.
pub fn hall_littlewood_p_to_classical_matrix(
    n: usize,
    basis: ClassicalBasis,
) -> Result<Matrix<TRat>> {
    let p2m = hall_littlewood_p_to_monomial_matrix(n)?;
    let m2x = lift_matrix(
        &transition_matrix(ClassicalBasis::Monomial, basis, n),
        &|r| t_constant(r.clone()),
    )?;
    p2m.mul(&m2x)
}

/// The `HallLittlewoodP -> Schur` transition matrix over `Q(t)`. Specializes to
/// the identity at `t = 0` (where `P_lambda = s_lambda`).
pub fn hall_littlewood_p_to_schur_matrix(n: usize) -> Result<Matrix<TRat>> {
    hall_littlewood_p_to_classical_matrix(n, ClassicalBasis::Schur)
}

/// The `Schur -> HallLittlewoodP` transition matrix: its `(lambda, mu)` entry
/// is the Kostka-Foulkes polynomial `K_{lambda mu}(t)`, i.e.
/// `s_lambda = sum_mu K_{lambda mu}(t) P_mu(x; t)` (Macdonald III (6.5)).
pub fn schur_to_hall_littlewood_p_matrix(n: usize) -> Result<Matrix<TRat>> {
    let s2m = lift_matrix(&schur_to_monomial_matrix(n), &|r| t_constant(r.clone()))?;
    let p2m = hall_littlewood_p_to_monomial_matrix(n)?;
    let m2hl = p2m
        .inverse()?
        .ok_or_else(|| {
            MathError::NumericalError(
                "P -> monomial matrix is unitriangular and must be invertible".to_string(),
            )
        })?;
    s2m.mul(&m2hl)
}

/// The Kostka-Foulkes polynomial `K_{lambda mu}(t)`, as a polynomial in `t`
/// with exact rational (in fact non-negative integer) coefficients.
pub fn kostka_foulkes_polynomial(lambda: &Partition, mu: &Partition) -> Result<Poly<Rational>> {
    if lambda.sum() != mu.sum() {
        return Ok(Poly::zero());
    }
    let n = lambda.sum();
    let parts = partitions_order(n);
    let mat = schur_to_hall_littlewood_p_matrix(n)?;
    let i = parts.iter().position(|p| p == lambda).ok_or_else(|| {
        MathError::InvalidArgument("lambda is not a partition of n".to_string())
    })?;
    let j = parts.iter().position(|p| p == mu).ok_or_else(|| {
        MathError::InvalidArgument("mu is not a partition of n".to_string())
    })?;
    let entry = mat.get(i, j)?;
    entry
        .as_polynomial()
        .cloned()
        .ok_or_else(|| non_polynomial_error("Kostka-Foulkes"))
}

fn non_polynomial_error(what: &str) -> MathError {
    MathError::NumericalError(format!(
        "{} coefficient failed to reduce to a polynomial in t; \
         this contradicts Macdonald III (2.7) — please report",
        what
    ))
}

fn row_as_polynomial_map(
    mat: &Matrix<TRat>,
    parts: &[Partition],
    lambda: &Partition,
    what: &str,
) -> Result<HashMap<Partition, Poly<Rational>>> {
    let i = parts.iter().position(|p| p == lambda).ok_or_else(|| {
        MathError::InvalidArgument(format!("{:?} is not a valid partition index", lambda))
    })?;
    let mut out = HashMap::new();
    for (j, mu) in parts.iter().enumerate() {
        let entry = mat.get(i, j)?;
        if entry.is_zero() {
            continue;
        }
        let poly = entry
            .as_polynomial()
            .cloned()
            .ok_or_else(|| non_polynomial_error(what))?;
        out.insert(mu.clone(), poly);
    }
    Ok(out)
}

/// The Hall-Littlewood polynomial `P_lambda(x; t)` expanded in the monomial
/// basis: a map `mu -> u_{lambda mu}(t)` with `u_{lambda lambda} = 1` and
/// support only on `mu <= lambda` in dominance order. Coefficients are exact
/// polynomials in `t` (low-degree-first `Rational` coefficients).
pub fn hall_littlewood_p(lambda: &Partition) -> Result<HashMap<Partition, Poly<Rational>>> {
    let n = lambda.sum();
    let mat = hall_littlewood_p_to_monomial_matrix(n)?;
    row_as_polynomial_map(&mat, &partitions_order(n), lambda, "Hall-Littlewood P")
}

/// The Hall-Littlewood polynomial `Q_lambda(x; t) = b_lambda(t) P_lambda(x; t)`
/// expanded in the monomial basis, with exact polynomial-in-`t` coefficients.
pub fn hall_littlewood_q(lambda: &Partition) -> Result<HashMap<Partition, Poly<Rational>>> {
    let n = lambda.sum();
    let mat = hall_littlewood_q_to_monomial_matrix(n)?;
    row_as_polynomial_map(&mat, &partitions_order(n), lambda, "Hall-Littlewood Q")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(v: Vec<usize>) -> Partition {
        Partition::new(v)
    }

    fn rp(coeffs: &[i64]) -> Poly<Rational> {
        Poly::from_coeffs(coeffs.iter().map(|&c| Rational::from(c)).collect())
    }

    /// Off-diagonal monomial-expansion coefficients of P_lambda for |lambda| <= 4,
    /// as (lambda, mu, coeffs of u_{lambda mu}(t) low-degree-first).
    ///
    /// Independently verified (sympy): Gram-Schmidt on the Hall inner product
    /// over Q(t) AND the charge-statistic Kostka-Foulkes route agree on these;
    /// they also match Macdonald III.2 examples (e.g. P_2 = m_2 + (1-t) m_11).
    fn expected_hl_offdiag(n: usize) -> Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> {
        match n {
            2 => vec![(vec![2], vec![1, 1], vec![1, -1])],
            3 => vec![
                (vec![3], vec![2, 1], vec![1, -1]),
                (vec![3], vec![1, 1, 1], vec![1, -2, 1]),
                (vec![2, 1], vec![1, 1, 1], vec![2, -1, -1]),
            ],
            4 => vec![
                (vec![4], vec![3, 1], vec![1, -1]),
                (vec![4], vec![2, 2], vec![1, -1]),
                (vec![4], vec![2, 1, 1], vec![1, -2, 1]),
                (vec![4], vec![1, 1, 1, 1], vec![1, -3, 3, -1]),
                (vec![3, 1], vec![2, 2], vec![1, -1]),
                (vec![3, 1], vec![2, 1, 1], vec![2, -2]),
                (vec![3, 1], vec![1, 1, 1, 1], vec![3, -5, 1, 1]),
                (vec![2, 2], vec![2, 1, 1], vec![1, -1]),
                (vec![2, 2], vec![1, 1, 1, 1], vec![2, -3, 0, 1]),
                (vec![2, 1, 1], vec![1, 1, 1, 1], vec![3, -1, -1, -1]),
            ],
            _ => vec![],
        }
    }

    #[test]
    fn test_hl_p_monomial_table_n_le_4() {
        for n in 1..=4usize {
            let parts = partitions_order(n);
            let mat = hall_littlewood_p_to_monomial_matrix(n).unwrap();
            let expected = expected_hl_offdiag(n);
            for (i, lam) in parts.iter().enumerate() {
                for (j, mu) in parts.iter().enumerate() {
                    let entry = mat.get(i, j).unwrap();
                    if i == j {
                        assert!(entry.is_one(), "P_{:?} diagonal", lam);
                        continue;
                    }
                    let exp = expected
                        .iter()
                        .find(|(l, m, _)| part(l.clone()) == *lam && part(m.clone()) == *mu);
                    match exp {
                        Some((_, _, coeffs)) => {
                            assert_eq!(
                                *entry,
                                RatFunc::from_poly(rp(coeffs)),
                                "P_{:?} coefficient of m_{:?}",
                                lam,
                                mu
                            );
                        }
                        None => {
                            assert!(entry.is_zero(), "P_{:?} at m_{:?} must vanish", lam, mu);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_hl_p_coeffs_are_integer_polynomials() {
        // Macdonald III (2.7): the u_{lambda mu}(t) lie in Z[t].
        for n in 0..=4usize {
            let parts = partitions_order(n);
            let mat = hall_littlewood_p_to_monomial_matrix(n).unwrap();
            for i in 0..parts.len() {
                for j in 0..parts.len() {
                    let entry = mat.get(i, j).unwrap();
                    let poly = entry.as_polynomial().unwrap_or_else(|| {
                        panic!("non-polynomial HL coefficient at n={} ({},{})", n, i, j)
                    });
                    for c in poly.coeffs() {
                        assert!(c.is_integer(), "non-integer coefficient {:?}", c);
                    }
                }
            }
        }
    }

    #[test]
    fn test_hl_p_t0_is_schur() {
        // P_lambda(x; 0) = s_lambda: the monomial expansion at t=0 is the
        // Kostka matrix (schur_to_monomial_matrix).
        for n in 1..=4usize {
            let parts = partitions_order(n);
            let mat = hall_littlewood_p_to_monomial_matrix(n).unwrap();
            let kostka = schur_to_monomial_matrix(n);
            for i in 0..parts.len() {
                for j in 0..parts.len() {
                    let v = mat.get(i, j).unwrap().eval(&Rational::zero()).unwrap();
                    assert_eq!(
                        v,
                        *kostka.get(i, j).unwrap(),
                        "t=0 mismatch at n={} ({},{})",
                        n,
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_hl_p_t1_is_monomial() {
        // P_lambda(x; 1) = m_lambda: the matrix specializes to the identity.
        for n in 1..=4usize {
            let parts = partitions_order(n);
            let mat = hall_littlewood_p_to_monomial_matrix(n).unwrap();
            for i in 0..parts.len() {
                for j in 0..parts.len() {
                    let v = mat.get(i, j).unwrap().eval(&Rational::one()).unwrap();
                    let expected = if i == j { Rational::one() } else { Rational::zero() };
                    assert_eq!(v, expected, "t=1 mismatch at n={} ({},{})", n, i, j);
                }
            }
        }
    }

    #[test]
    fn test_hl_q_is_b_times_p() {
        for n in 1..=4usize {
            let parts = partitions_order(n);
            let p = hall_littlewood_p_to_monomial_matrix(n).unwrap();
            let q = hall_littlewood_q_to_monomial_matrix(n).unwrap();
            for (i, lam) in parts.iter().enumerate() {
                let b = RatFunc::from_poly(hl_b_polynomial(lam));
                for j in 0..parts.len() {
                    assert_eq!(
                        *q.get(i, j).unwrap(),
                        b.clone() * p.get(i, j).unwrap().clone()
                    );
                }
            }
        }
    }

    #[test]
    fn test_hl_q_column_and_elementary() {
        // P_(1^n) = e_n = m_(1^n) exactly, so Q_(1^n) = phi_n(t) e_n
        // ("elementary" facet of the Hall-Littlewood Q family).
        for n in 1..=4usize {
            let ones = part(vec![1; n]);
            let q = hall_littlewood_q(&ones).unwrap();
            assert_eq!(q.len(), 1);
            assert_eq!(q[&ones], phi_polynomial(n));
            let p = hall_littlewood_p(&ones).unwrap();
            assert_eq!(p.len(), 1);
            assert_eq!(p[&ones], Poly::one());
        }
        // Q_(2) = (1-t) m_2 + (1-t)^2 m_11 (b_(2) = phi_1 = 1 - t).
        let q2 = hall_littlewood_q(&part(vec![2])).unwrap();
        assert_eq!(q2[&part(vec![2])], rp(&[1, -1]));
        assert_eq!(q2[&part(vec![1, 1])], rp(&[1, -2, 1]));
    }

    #[test]
    fn test_hl_q_t1_normalized_limit() {
        // b_lambda(0) = 1, so Q_lambda(x; 0) = s_lambda too.
        // At t = 1, Q_lambda vanishes to order l(lambda): each coefficient is
        // divisible by (1-t)^{l(lambda)}, and the quotient at t=1 is
        // (prod_i m_i(lambda)!) * [mu == lambda] — the "augmented monomial"
        // normalized limit. (The bare Q_lambda(x; 1) is 0 for lambda != ();
        // the true elementary connection is Q_(1^n) = phi_n e_n, tested above.)
        for n in 1..=4usize {
            let parts = partitions_order(n);
            let q = hall_littlewood_q_to_monomial_matrix(n).unwrap();
            for (i, lam) in parts.iter().enumerate() {
                assert!(hl_b_polynomial(lam).eval(&Rational::zero()).is_one());
                let l = lam.length();
                let one_minus_t_pow_l = rp(&[1, -1]).pow(l);
                // multiplicity factorial product
                let mut mult: HashMap<usize, usize> = HashMap::new();
                for &pt in lam.parts() {
                    *mult.entry(pt).or_insert(0) += 1;
                }
                let mut fact = 1i64;
                for (_, m) in mult {
                    for x in 1..=m as i64 {
                        fact *= x;
                    }
                }
                for (j, mu) in parts.iter().enumerate() {
                    let entry = q.get(i, j).unwrap();
                    if entry.is_zero() {
                        continue;
                    }
                    let poly = entry.as_polynomial().expect("Q coeffs are polynomials");
                    let quotient = poly
                        .div_exact(&one_minus_t_pow_l)
                        .expect("Q coefficients divisible by (1-t)^{l(lambda)}");
                    let at_one = quotient.eval(&Rational::one());
                    let expected = if lam == mu {
                        Rational::from(fact)
                    } else {
                        Rational::zero()
                    };
                    assert_eq!(at_one, expected, "normalized t=1 limit {:?} {:?}", lam, mu);
                }
            }
        }
    }

    /// Kostka-Foulkes table for |lambda| <= 4, independently verified two ways
    /// (charge statistic and Gram-Schmidt inversion in sympy); matches the
    /// classical published tables (e.g. K_{(2,1),(1^3)} = t + t^2,
    /// K_{(3,1),(1^4)} = t^3 + t^4 + t^5, K_{(2,2),(1^4)} = t^2 + t^4).
    fn expected_kf_offdiag(n: usize) -> Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> {
        match n {
            2 => vec![(vec![2], vec![1, 1], vec![0, 1])],
            3 => vec![
                (vec![3], vec![2, 1], vec![0, 1]),
                (vec![3], vec![1, 1, 1], vec![0, 0, 0, 1]),
                (vec![2, 1], vec![1, 1, 1], vec![0, 1, 1]),
            ],
            4 => vec![
                (vec![4], vec![3, 1], vec![0, 1]),
                (vec![4], vec![2, 2], vec![0, 0, 1]),
                (vec![4], vec![2, 1, 1], vec![0, 0, 0, 1]),
                (vec![4], vec![1, 1, 1, 1], vec![0, 0, 0, 0, 0, 0, 1]),
                (vec![3, 1], vec![2, 2], vec![0, 1]),
                (vec![3, 1], vec![2, 1, 1], vec![0, 1, 1]),
                (vec![3, 1], vec![1, 1, 1, 1], vec![0, 0, 0, 1, 1, 1]),
                (vec![2, 2], vec![2, 1, 1], vec![0, 1]),
                (vec![2, 2], vec![1, 1, 1, 1], vec![0, 0, 1, 0, 1]),
                (vec![2, 1, 1], vec![1, 1, 1, 1], vec![0, 1, 1, 1]),
            ],
            _ => vec![],
        }
    }

    #[test]
    fn test_kostka_foulkes_table_n_le_4() {
        for n in 1..=4usize {
            let parts = partitions_order(n);
            let mat = schur_to_hall_littlewood_p_matrix(n).unwrap();
            let expected = expected_kf_offdiag(n);
            for (i, lam) in parts.iter().enumerate() {
                for (j, mu) in parts.iter().enumerate() {
                    let entry = mat.get(i, j).unwrap();
                    if i == j {
                        assert!(entry.is_one(), "K_{:?}{:?}", lam, mu);
                        continue;
                    }
                    let exp = expected
                        .iter()
                        .find(|(l, m, _)| part(l.clone()) == *lam && part(m.clone()) == *mu);
                    match exp {
                        Some((_, _, coeffs)) => {
                            assert_eq!(
                                *entry,
                                RatFunc::from_poly(rp(coeffs)),
                                "K_{:?},{:?}(t)",
                                lam,
                                mu
                            )
                        }
                        None => assert!(entry.is_zero(), "K_{:?},{:?} must vanish", lam, mu),
                    }
                }
            }
        }
    }

    #[test]
    fn test_kostka_foulkes_polynomial_accessor() {
        // K_{(2,1),(1,1,1)}(t) = t + t^2 ; different sizes give 0.
        let k = kostka_foulkes_polynomial(&part(vec![2, 1]), &part(vec![1, 1, 1])).unwrap();
        assert_eq!(k, rp(&[0, 1, 1]));
        let z = kostka_foulkes_polynomial(&part(vec![2]), &part(vec![1, 1, 1])).unwrap();
        assert!(z.is_zero());
    }

    #[test]
    fn test_hl_p_to_schur_and_back() {
        // schur_to_HL * HL_to_schur = identity; HL_to_schur at t=0 is the identity.
        for n in 1..=4usize {
            let a = schur_to_hall_littlewood_p_matrix(n).unwrap();
            let b = hall_littlewood_p_to_schur_matrix(n).unwrap();
            let prod = a.mul(&b).unwrap();
            let k = partitions_order(n).len();
            for i in 0..k {
                for j in 0..k {
                    let expected = if i == j { TRat::one() } else { TRat::zero() };
                    assert_eq!(*prod.get(i, j).unwrap(), expected);
                    let at0 = b.get(i, j).unwrap().eval(&Rational::zero()).unwrap();
                    let e0 = if i == j { Rational::one() } else { Rational::zero() };
                    assert_eq!(at0, e0, "P->s at t=0 must be identity");
                }
            }
        }
    }

    #[test]
    fn test_hl_p_in_powersum_and_elementary() {
        // partitions_order(2) = [(2), (1,1)].
        // P_2 = ((1+t)/2) p_2 + ((1-t)/2) p_{11} (from m_2 = p_2,
        // m_11 = (p_11 - p_2)/2); P_(11) = m_11 = -p_2/2 + p_11/2.
        let mat = hall_littlewood_p_to_classical_matrix(2, ClassicalBasis::PowerSum).unwrap();
        let half = Rational::new(1i64, 2i64).unwrap();
        let one_plus_t_half = RatFunc::from_poly(Poly::from_coeffs(vec![half.clone(), half.clone()]));
        let one_minus_t_half =
            RatFunc::from_poly(Poly::from_coeffs(vec![half.clone(), -half.clone()]));
        assert_eq!(*mat.get(0, 0).unwrap(), one_plus_t_half);
        assert_eq!(*mat.get(0, 1).unwrap(), one_minus_t_half);
        assert_eq!(*mat.get(1, 0).unwrap(), RatFunc::constant(-half.clone()));
        assert_eq!(*mat.get(1, 1).unwrap(), RatFunc::constant(half));

        // P_(1^n) = e_n: in the elementary basis the row of (1^n) is the unit
        // vector at the single-part partition (n) (since e_{(n)} = e_n).
        for n in 2..=4usize {
            let parts = partitions_order(n);
            let e = hall_littlewood_p_to_classical_matrix(n, ClassicalBasis::Elementary).unwrap();
            let i_ones = parts.iter().position(|p| *p == part(vec![1; n])).unwrap();
            let j_n = parts.iter().position(|p| *p == part(vec![n])).unwrap();
            for j in 0..parts.len() {
                let expected = if j == j_n { TRat::one() } else { TRat::zero() };
                assert_eq!(*e.get(i_ones, j).unwrap(), expected, "P_(1^{}) = e_{}", n, n);
            }
        }
    }

    #[test]
    fn test_hl_degree_zero_and_one() {
        let empty = part(vec![]);
        let p = hall_littlewood_p(&empty).unwrap();
        assert_eq!(p.len(), 1);
        assert!(p[&empty].is_one());
        let p1 = hall_littlewood_p(&part(vec![1])).unwrap();
        assert_eq!(p1.len(), 1);
        assert!(p1[&part(vec![1])].is_one());
    }
}

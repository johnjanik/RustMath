//! Hermite–Padé / rational-function reconstruction over `ℚ`.
//!
//! Given a truncated power series `c₀ + c₁u + … + c_{N-1} u^{N-1}` (the slice
//! `series` holds the `cᵢ`) and target degrees `num_deg = m`, `den_deg = n`,
//! [`hermite_pade`] finds polynomials `p` (deg ≤ m) and `q` (deg ≤ n), `q ≠ 0`,
//! with
//!
//! ```text
//!     q(u)·series(u) − p(u) = O(u^{m+n+1}).
//! ```
//!
//! The coefficient of `u^k` in `q·series` is `Σ_j q_j c_{k-j}`. For `k = 0..=m`
//! this simply *defines* `p_k`; for `k > m` it must vanish. Rather than use only
//! the minimal `n × (n+1)` block of vanishing equations — which always admits a
//! nonzero solution and so can never *reject* a series — we impose the vanishing
//! condition on **every** available tail coefficient `k = m+1 .. N-1`. This turns
//! the reconstruction into an honest recognizer: a nonzero denominator exists iff
//! the first `N` coefficients are genuinely consistent with a degree-`(m, n)`
//! rational function. A truncated `exp(u)` (whose Hankel determinants are all
//! nonzero) therefore yields only the trivial solution once `N > m+n+1`, and we
//! return `None` instead of fabricating a spurious approximant.
//!
//! The denominator is normalized to `q(0) = 1`. If the only solutions have
//! `q(0) = 0` (degenerate — no genuine power-series denominator), we return
//! `None`.

use rustmath_core::Ring;
use rustmath_rationals::Rational;

/// Reduce `a` to reduced row-echelon form over `ℚ`.
///
/// Returns the RREF matrix together with the list of pivot columns (in
/// increasing order, one per pivot row).
fn rref(mut a: Vec<Vec<Rational>>) -> (Vec<Vec<Rational>>, Vec<usize>) {
    let rows = a.len();
    if rows == 0 {
        return (a, Vec::new());
    }
    let cols = a[0].len();
    let mut pivot_cols = Vec::new();
    let mut r = 0;

    for c in 0..cols {
        if r >= rows {
            break;
        }
        // Find a pivot in column `c` at or below row `r`.
        let mut piv = None;
        for (i, row) in a.iter().enumerate().skip(r) {
            if !row[c].is_zero() {
                piv = Some(i);
                break;
            }
        }
        let piv = match piv {
            Some(i) => i,
            None => continue,
        };
        a.swap(r, piv);

        // Scale the pivot row so the pivot entry becomes 1.
        let inv = a[r][c].reciprocal().expect("pivot entry is nonzero");
        for j in 0..cols {
            a[r][j] = a[r][j].clone() * inv.clone();
        }

        // Eliminate column `c` from every other row.
        for i in 0..rows {
            if i != r && !a[i][c].is_zero() {
                let factor = a[i][c].clone();
                for j in 0..cols {
                    a[i][j] = a[i][j].clone() - factor.clone() * a[r][j].clone();
                }
            }
        }

        pivot_cols.push(c);
        r += 1;
    }

    (a, pivot_cols)
}

/// Basis of the right null space of `a` (which has `cols` columns), over `ℚ`.
fn nullspace_basis(a: Vec<Vec<Rational>>, cols: usize) -> Vec<Vec<Rational>> {
    let (rref_a, pivot_cols) = rref(a);
    let free_cols: Vec<usize> = (0..cols).filter(|c| !pivot_cols.contains(c)).collect();

    let mut basis = Vec::with_capacity(free_cols.len());
    for &f in &free_cols {
        let mut v = vec![Rational::zero(); cols];
        v[f] = Rational::one();
        for (row, &pc) in pivot_cols.iter().enumerate() {
            v[pc] = -rref_a[row][f].clone();
        }
        basis.push(v);
    }
    basis
}

/// Hermite–Padé reconstruction of a rational function from its power series.
///
/// `series` holds the coefficients `c₀, c₁, …, c_{N-1}` (increasing degree) of a
/// truncated power series. Returns `Some((p_coeffs, q_coeffs))`, both in
/// increasing-degree order (`p_coeffs` has length `num_deg + 1`, `q_coeffs` has
/// length `den_deg + 1`), with the denominator normalized so `q_coeffs[0] == 1`
/// and
///
/// ```text
///     q(u)·series(u) − p(u) ≡ 0   (mod u^N),
/// ```
///
/// i.e. the reconstructed rational function reproduces **every** supplied
/// coefficient. Returns `None` when
///
/// * `num_deg + den_deg + 1 > series.len()` (not enough data), or
/// * no nonzero denominator of degree ≤ `den_deg` reproduces all `N`
///   coefficients (the series is not a degree-`(num_deg, den_deg)` rational
///   function to that order), or
/// * every solution is degenerate with `q(0) = 0`.
pub fn hermite_pade(
    series: &[Rational],
    num_deg: usize,
    den_deg: usize,
) -> Option<(Vec<Rational>, Vec<Rational>)> {
    let n_terms = series.len();
    let m = num_deg;
    let n = den_deg;

    // Need at least m + n + 1 coefficients to pin down p and q.
    if n_terms == 0 || m + n + 1 > n_terms {
        return None;
    }

    // Coefficient accessor with `c_i = 0` for i outside the known range.
    let c = |i: usize| -> Rational {
        if i < n_terms {
            series[i].clone()
        } else {
            Rational::zero()
        }
    };

    // Homogeneous system in the unknowns q_0 .. q_n: for every tail index
    // k = m+1 .. n_terms-1 require  Σ_{j=0}^{n} q_j c_{k-j} = 0.
    let mut mat: Vec<Vec<Rational>> = Vec::with_capacity(n_terms.saturating_sub(m + 1));
    for k in (m + 1)..n_terms {
        let mut row = Vec::with_capacity(n + 1);
        for j in 0..=n {
            // c_{k-j}, with the convention c_i = 0 for i < 0.
            if j <= k {
                row.push(c(k - j));
            } else {
                row.push(Rational::zero());
            }
        }
        mat.push(row);
    }

    let basis = nullspace_basis(mat, n + 1);

    // Pick a null-space vector with nonzero constant term so we can normalize
    // q(0) = 1. If none exists, the only denominators vanish at 0 (degenerate).
    let q_vec = basis.into_iter().find(|v| !v[0].is_zero())?;

    let inv0 = q_vec[0].reciprocal().expect("q(0) is nonzero here");
    let q: Vec<Rational> = q_vec.into_iter().map(|x| x * inv0.clone()).collect();

    // Numerator: p_k = Σ_{j=0}^{min(n,k)} q_j c_{k-j} for k = 0 .. m.
    let mut p = Vec::with_capacity(m + 1);
    for k in 0..=m {
        let mut acc = Rational::zero();
        for j in 0..=n.min(k) {
            acc = acc + q[j].clone() * c(k - j);
        }
        p.push(acc);
    }

    Some((p, q))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::UnivariatePolynomial;

    fn q(num: i64, den: i64) -> Rational {
        Rational::new(num, den).unwrap()
    }

    fn poly(coeffs: &[Rational]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.to_vec())
    }

    #[test]
    fn recovers_known_rational() {
        // p/q = (1 + 2u) / (1 - u + 3u^2). Independently expanded in sympy:
        //   series = [1, 3, 0, -9, -9, 18, 45, -9, ...]
        // Take N = 2*max(deg)+2 = 6 terms.
        let series: Vec<Rational> = [1, 3, 0, -9, -9, 18]
            .iter()
            .map(|&x| q(x, 1))
            .collect();

        let (p_rec, q_rec) = hermite_pade(&series, 1, 2).expect("series is rational");

        let p_true = [q(1, 1), q(2, 1)];
        let q_true = [q(1, 1), q(-1, 1), q(3, 1)];

        // Recovered up to a common scalar: p*q_rec == p_rec*q_true.
        let lhs = poly(&p_true) * poly(&q_rec);
        let rhs = poly(&p_rec) * poly(&q_true);
        assert_eq!(lhs, rhs);

        // Normalization pins it exactly (q_true(0) = 1 already).
        assert_eq!(poly(&p_rec), poly(&p_true));
        assert_eq!(poly(&q_rec), poly(&q_true));
    }

    #[test]
    fn recovered_denominator_is_normalized() {
        let series: Vec<Rational> = [1, 3, 0, -9, -9, 18]
            .iter()
            .map(|&x| q(x, 1))
            .collect();
        let (_p, q_rec) = hermite_pade(&series, 1, 2).unwrap();
        assert_eq!(q_rec[0], q(1, 1));
    }

    #[test]
    fn rejects_exp_series() {
        // exp(u) = Σ u^k / k!. Its Hankel determinants are all nonzero, so it is
        // not a low-degree rational function. With more terms than the minimal
        // m+n+1, the reconstruction must fail rather than fabricate.
        fn factorial(k: u64) -> i64 {
            (1..=k as i64).product::<i64>().max(1)
        }
        let n_terms = 6;
        let series: Vec<Rational> = (0..n_terms).map(|k| q(1, factorial(k))).collect();

        // sympy: null space is trivial for these (rows >= cols, full column rank).
        assert!(hermite_pade(&series, 1, 1).is_none());
        assert!(hermite_pade(&series, 2, 2).is_none());
    }

    #[test]
    fn too_few_terms_returns_none() {
        // num_deg + den_deg + 1 = 4 > 3 terms available.
        let series: Vec<Rational> = [1, 3, 0].iter().map(|&x| q(x, 1)).collect();
        assert!(hermite_pade(&series, 1, 2).is_none());
    }

    #[test]
    fn recovers_polynomial_with_zero_denominator_degree() {
        // A pure polynomial 1 + 2u is rational with q = 1 (den_deg = 0), provided
        // the tail coefficients all vanish.
        let series: Vec<Rational> = [1, 2, 0, 0].iter().map(|&x| q(x, 1)).collect();
        let (p_rec, q_rec) = hermite_pade(&series, 1, 0).expect("polynomial is rational");
        assert_eq!(poly(&p_rec), poly(&[q(1, 1), q(2, 1)]));
        assert_eq!(poly(&q_rec), poly(&[q(1, 1)]));

        // But if a later coefficient is nonzero it is *not* a degree-(1,0)
        // rational, so reconstruction must fail.
        let series2: Vec<Rational> = [1, 2, 0, 5].iter().map(|&x| q(x, 1)).collect();
        assert!(hermite_pade(&series2, 1, 0).is_none());
    }
}

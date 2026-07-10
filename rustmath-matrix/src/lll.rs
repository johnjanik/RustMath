//! LLL lattice basis reduction over ℤ.
//!
//! The crate had Hermite/Smith normal forms and Gram–Schmidt but no LLL — the
//! lattice reduction needed for short-vector / small-coefficient problems (e.g. the
//! `polredabs`-style reduction of a number field's `T₂` lattice in Phase 3).
//!
//! Implementation: classic Lenstra–Lenstra–Lovász with `δ = 3/4`. Basis vectors
//! are exact `Vec<Integer>` (rows); the Gram–Schmidt coefficients steering the
//! reduction are computed in `f64`. Because every basis update is an exact integer
//! operation (`b_k ← b_k − r·b_j`, swaps), the returned transform `U` is exactly
//! unimodular and `reduced = U · input` holds exactly regardless of float rounding;
//! the float GSO only affects how many steps are taken, never correctness of the
//! lattice. An iteration cap guards against pathological non-termination.

use rustmath_integers::Integer;

const DELTA: f64 = 0.75;

fn to_f64(x: &Integer) -> f64 {
    x.to_f64().unwrap_or_else(|| {
        // fall back via bit length for very large values (sign-preserving)
        let bits = x.bit_length() as i32;
        let s = if x.signum() < 0 { -1.0 } else { 1.0 };
        s * 2f64.powi(bits.min(1023))
    })
}

/// Exact integer dot product of two equal-length vectors.
fn dot(a: &[Integer], b: &[Integer]) -> Integer {
    let mut s = Integer::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        s = s + x.clone() * y.clone();
    }
    s
}

/// Gram–Schmidt in f64 from exact integer vectors. Returns `(mu, bstar_norm2)`
/// where `mu[i][j]` are the GSO coefficients (`mu[i][i] = 1`) and `bnorm[i]` is
/// `‖b_i*‖²`.
fn gram_schmidt(b: &[Vec<Integer>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = b.len();
    let mut mu = vec![vec![0.0f64; n]; n];
    let mut bnorm = vec![0.0f64; n];
    let mut bstar: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let bi: Vec<f64> = b[i].iter().map(to_f64).collect();
        let mut v = bi.clone();
        for j in 0..i {
            // mu_ij = <b_i, b_j*> / ‖b_j*‖²  ; use exact <b_i,b_j> projected
            let mut dotp = 0.0;
            for d in 0..v.len() {
                dotp += bi[d] * bstar[j][d];
            }
            let m = if bnorm[j] > 0.0 { dotp / bnorm[j] } else { 0.0 };
            mu[i][j] = m;
            for d in 0..v.len() {
                v[d] -= m * bstar[j][d];
            }
        }
        mu[i][i] = 1.0;
        let mut nrm = 0.0;
        for d in 0..v.len() {
            nrm += v[d] * v[d];
        }
        bnorm[i] = nrm;
        bstar.push(v);
    }
    let _ = dot; // exact dot available for callers/tests
    (mu, bnorm)
}

/// LLL-reduce the lattice spanned by `basis` (rows). Returns
/// `(reduced, u)` with `reduced = u · basis` and `u` unimodular.
/// `basis` must be non-empty with equal-length rows.
pub fn lll_reduce(basis: &[Vec<Integer>]) -> (Vec<Vec<Integer>>, Vec<Vec<Integer>>) {
    let n = basis.len();
    let mut b: Vec<Vec<Integer>> = basis.to_vec();
    // u = identity
    let mut u: Vec<Vec<Integer>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { Integer::one() } else { Integer::zero() }).collect())
        .collect();
    if n <= 1 {
        return (b, u);
    }

    let (mut mu, mut bnorm) = gram_schmidt(&b);
    let cap = 1000 * n * n + 1000;
    let mut iters = 0usize;
    let mut k = 1usize;
    while k < n {
        iters += 1;
        if iters > cap {
            break;
        }
        // size-reduce b[k] against b[k-1..=0]
        let mut changed = false;
        for j in (0..k).rev() {
            if mu[k][j].abs() > 0.5 {
                let r = mu[k][j].round();
                let ri = Integer::from(r as i64);
                if !ri.is_zero() {
                    for d in 0..b[k].len() {
                        b[k][d] = b[k][d].clone() - ri.clone() * b[j][d].clone();
                    }
                    for d in 0..n {
                        u[k][d] = u[k][d].clone() - ri.clone() * u[j][d].clone();
                    }
                    changed = true;
                }
            }
        }
        if changed {
            let g = gram_schmidt(&b);
            mu = g.0;
            bnorm = g.1;
        }
        // Lovász condition
        if bnorm[k] >= (DELTA - mu[k][k - 1] * mu[k][k - 1]) * bnorm[k - 1] {
            k += 1;
        } else {
            b.swap(k, k - 1);
            u.swap(k, k - 1);
            let g = gram_schmidt(&b);
            mu = g.0;
            bnorm = g.1;
            k = if k > 1 { k - 1 } else { 1 };
        }
    }
    (b, u)
}

/// Convenience: the reduced basis only.
pub fn lll_reduced_basis(basis: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    lll_reduce(basis).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[i64]) -> Vec<Integer> {
        xs.iter().map(|&x| Integer::from(x)).collect()
    }

    fn norm2(x: &[Integer]) -> Integer {
        dot(x, x)
    }

    /// Check the LLL-reduced conditions on the float GSO of `b`.
    fn is_lll_reduced(b: &[Vec<Integer>]) -> bool {
        let (mu, bnorm) = gram_schmidt(b);
        let n = b.len();
        for i in 1..n {
            for j in 0..i {
                if mu[i][j].abs() > 0.5 + 1e-6 {
                    return false;
                }
            }
        }
        for k in 1..n {
            if bnorm[k] < (DELTA - mu[k][k - 1] * mu[k][k - 1]) * bnorm[k - 1] - 1e-6 {
                return false;
            }
        }
        true
    }

    /// Product of GSO norms = det(Gram); a lattice invariant under unimodular maps.
    fn gram_det_approx(b: &[Vec<Integer>]) -> f64 {
        gram_schmidt(b).1.iter().product()
    }

    #[test]
    fn test_classic_2d() {
        // A skewed basis of Z^2; LLL should find a near-orthogonal short basis.
        let basis = vec![v(&[1, 1, 1]), v(&[-1, 0, 2])];
        let (red, u) = lll_reduce(&basis);
        assert!(is_lll_reduced(&red));
        // reduced = u · basis
        for i in 0..2 {
            for d in 0..3 {
                let mut acc = Integer::zero();
                for j in 0..2 {
                    acc = acc + u[i][j].clone() * basis[j][d].clone();
                }
                assert_eq!(acc, red[i][d]);
            }
        }
    }

    #[test]
    fn test_reduces_a_known_bad_basis() {
        // Famous LLL example (Cohen): rows of a 3x3 lattice.
        let basis = vec![v(&[1, 0, 0]), v(&[0, 1, 0]), v(&[3, 4, 5])];
        let (red, _u) = lll_reduce(&basis);
        assert!(is_lll_reduced(&red));
        // lattice invariant preserved
        assert!((gram_det_approx(&basis) - gram_det_approx(&red)).abs() < 1e-3);
    }

    #[test]
    fn test_shortens_vectors() {
        // A clearly reducible basis: second vector is first + tiny perturbation.
        let basis = vec![v(&[100, 1]), v(&[101, 1])];
        let (red, _) = lll_reduce(&basis);
        assert!(is_lll_reduced(&red));
        // shortest reduced vector should be much shorter than the input minimum
        let min_red = red.iter().map(|x| norm2(x)).min().unwrap();
        assert!(min_red < Integer::from(100));
    }

    #[test]
    fn test_identity_stays_reduced() {
        let basis = vec![v(&[1, 0, 0]), v(&[0, 1, 0]), v(&[0, 0, 1])];
        let (red, u) = lll_reduce(&basis);
        assert!(is_lll_reduced(&red));
        // U should be a signed permutation (unimodular); det magnitude 1 ⇒ each
        // reduced vector still a unit vector
        for r in &red {
            assert_eq!(norm2(r), Integer::one());
        }
        let _ = u;
    }
}

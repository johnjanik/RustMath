//! Lattice short-vector enumeration (Fincke–Pohst) and closest-vector search.
//!
//! Given a lattice basis (rows), [`short_vectors`] returns every nonzero lattice
//! vector of squared length `≤ bound`, [`shortest_vector`] solves SVP, and
//! [`closest_vector`] solves CVP. The lattice is LLL-reduced first (Phase 3b) for an
//! efficient enumeration; bounds come from an `f64` Gram–Schmidt, exact lengths are
//! checked over `ℤ`. References: Fincke–Pohst; Cohen, Alg. 2.7.5/2.7.7.

use crate::lll::lll_reduce;
use rustmath_integers::Integer;

fn to_f64(x: &Integer) -> f64 {
    x.to_f64().unwrap_or(0.0)
}

fn dot(a: &[Integer], b: &[Integer]) -> Integer {
    let mut s = Integer::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        s = s + x.clone() * y.clone();
    }
    s
}

/// Gram–Schmidt over `f64`: `mu[i][j]` (`i>j`) and `bnorm[i] = ‖b_i*‖²`.
fn gram_schmidt(b: &[Vec<Integer>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = b.len();
    let mut mu = vec![vec![0.0f64; n]; n];
    let mut bnorm = vec![0.0f64; n];
    let mut bstar: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let bi: Vec<f64> = b[i].iter().map(to_f64).collect();
        let mut v = bi.clone();
        for j in 0..i {
            let mut d = 0.0;
            for k in 0..v.len() {
                d += bi[k] * bstar[j][k];
            }
            let m = if bnorm[j] > 0.0 { d / bnorm[j] } else { 0.0 };
            mu[i][j] = m;
            for k in 0..v.len() {
                v[k] -= m * bstar[j][k];
            }
        }
        let nrm: f64 = v.iter().map(|x| x * x).sum();
        bnorm[i] = nrm;
        bstar.push(v);
    }
    (mu, bnorm)
}

/// Linear combination `Σ xᵢ·basis[i]`.
fn combine(x: &[i64], basis: &[Vec<Integer>]) -> Vec<Integer> {
    let dim = basis[0].len();
    let mut out = vec![Integer::zero(); dim];
    for (i, &xi) in x.iter().enumerate() {
        if xi != 0 {
            let c = Integer::from(xi);
            for d in 0..dim {
                out[d] = out[d].clone() + c.clone() * basis[i][d].clone();
            }
        }
    }
    out
}

/// Canonical sign: negate so the first nonzero entry is positive (dedupes `±v`).
fn canonical(mut v: Vec<Integer>) -> Vec<Integer> {
    for c in &v {
        if c.signum() < 0 {
            v = v.iter().map(|x| -x.clone()).collect();
            break;
        } else if c.signum() > 0 {
            break;
        }
    }
    v
}

/// All nonzero lattice vectors with squared length `≤ bound` (one per `±` pair).
pub fn short_vectors(basis: &[Vec<Integer>], bound: &Integer) -> Vec<Vec<Integer>> {
    let (red, _u) = lll_reduce(basis);
    let n = red.len();
    let (mu, bnorm) = gram_schmidt(&red);
    let c = to_f64(bound) + 1e-6;
    let mut x = vec![0i64; n];
    let mut raw: Vec<Vec<i64>> = Vec::new();
    enumerate(n as isize - 1, &mut x, &mu, &bnorm, 0.0, c, n, &mut raw);

    use std::collections::HashSet;
    let mut seen: HashSet<Vec<Integer>> = HashSet::new();
    let mut out = Vec::new();
    for xc in raw {
        let v = combine(&xc, &red);
        if dot(&v, &v) <= *bound {
            let cv = canonical(v);
            if cv.iter().any(|e| !e.is_zero()) && seen.insert(cv.clone()) {
                out.push(cv);
            }
        }
    }
    out
}

/// Fincke–Pohst recursion: enumerate integer coordinate vectors `x` with
/// `Σ_i B_i·(x_i + U_i)² ≤ C`, `U_i = Σ_{j>i} mu[j][i]·x_j`.
#[allow(clippy::too_many_arguments)]
fn enumerate(
    i: isize,
    x: &mut [i64],
    mu: &[Vec<f64>],
    bnorm: &[f64],
    rho_above: f64,
    c: f64,
    n: usize,
    out: &mut Vec<Vec<i64>>,
) {
    if i < 0 {
        if x.iter().any(|&v| v != 0) {
            out.push(x.to_vec());
        }
        return;
    }
    let iu = i as usize;
    let mut u = 0.0f64;
    for j in (iu + 1)..n {
        u += mu[j][iu] * x[j] as f64;
    }
    let rem = c - rho_above;
    if rem < -1e-9 || bnorm[iu] <= 0.0 {
        return;
    }
    let width = (rem / bnorm[iu]).max(0.0).sqrt();
    let lo = (-width - u).ceil() as i64;
    let hi = (width - u).floor() as i64;
    for xi in lo..=hi {
        x[iu] = xi;
        let d = xi as f64 + u;
        enumerate(i - 1, x, mu, bnorm, rho_above + bnorm[iu] * d * d, c, n, out);
    }
    x[iu] = 0;
}

/// A shortest nonzero lattice vector (SVP).
pub fn shortest_vector(basis: &[Vec<Integer>]) -> Vec<Integer> {
    let (red, _u) = lll_reduce(basis);
    // bound: length of the shortest reduced basis vector
    let mut bound = dot(&red[0], &red[0]);
    for v in &red {
        let d = dot(v, v);
        if !d.is_zero() && d < bound {
            bound = d;
        }
    }
    let mut best: Option<Vec<Integer>> = None;
    for v in short_vectors(basis, &bound) {
        let d = dot(&v, &v);
        match &best {
            Some(b) if dot(b, b) <= d => {}
            _ => best = Some(v),
        }
    }
    best.unwrap_or_else(|| red[0].clone())
}

/// A closest lattice vector to `target` (CVP) by enumeration around the Babai
/// nearest-plane point.
pub fn closest_vector(basis: &[Vec<Integer>], target: &[Integer]) -> Vec<Integer> {
    let (red, _u) = lll_reduce(basis);
    let n = red.len();
    let (mu, bnorm) = gram_schmidt(&red);
    let tf: Vec<f64> = target.iter().map(to_f64).collect();
    // GSO coordinates of the target: tc[i] = <t, b_i*>/‖b_i*‖²
    let bstar = compute_bstar(&red, &mu);
    let tc: Vec<f64> = (0..n)
        .map(|i| {
            let mut d = 0.0;
            for k in 0..tf.len() {
                d += tf[k] * bstar[i][k];
            }
            if bnorm[i] > 0.0 {
                d / bnorm[i]
            } else {
                0.0
            }
        })
        .collect();
    // initial radius from Babai rounding
    let babai = babai_round(&tc, &mu, n);
    let bv = combine(&babai, &red);
    let mut radius2 = diff_norm2(&bv, target);
    let mut best = bv;
    let mut x = vec![0i64; n];
    cvp_enum(n as isize - 1, &mut x, &mu, &bnorm, &tc, 0.0, &mut radius2, &red, target, &mut best, n);
    best
}

fn compute_bstar(b: &[Vec<Integer>], mu: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = b.len();
    let mut bstar: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v: Vec<f64> = b[i].iter().map(to_f64).collect();
        for j in 0..i {
            for k in 0..v.len() {
                v[k] -= mu[i][j] * bstar[j][k];
            }
        }
        bstar.push(v);
    }
    bstar
}

fn babai_round(tc: &[f64], mu: &[Vec<f64>], n: usize) -> Vec<i64> {
    let mut x = vec![0i64; n];
    for i in (0..n).rev() {
        let mut u = 0.0;
        for j in (i + 1)..n {
            u += mu[j][i] * x[j] as f64;
        }
        x[i] = (tc[i] - u).round() as i64;
    }
    x
}

fn diff_norm2(v: &[Integer], t: &[Integer]) -> Integer {
    let mut s = Integer::zero();
    for k in 0..v.len() {
        let d = v[k].clone() - t[k].clone();
        s = s + d.clone() * d;
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn cvp_enum(
    i: isize,
    x: &mut [i64],
    mu: &[Vec<f64>],
    bnorm: &[f64],
    tc: &[f64],
    rho_above: f64,
    radius2: &mut Integer,
    red: &[Vec<Integer>],
    target: &[Integer],
    best: &mut Vec<Integer>,
    n: usize,
) {
    if i < 0 {
        let v = combine(x, red);
        let d = diff_norm2(&v, target);
        if d < *radius2 {
            *radius2 = d;
            *best = v;
        }
        return;
    }
    let iu = i as usize;
    let mut u = 0.0f64;
    for j in (iu + 1)..n {
        u += mu[j][iu] * x[j] as f64;
    }
    let center = tc[iu] - u;
    let rem = to_f64(radius2) - rho_above;
    if rem < -1e-9 || bnorm[iu] <= 0.0 {
        return;
    }
    let width = (rem / bnorm[iu]).max(0.0).sqrt() + 1e-9;
    let lo = (center - width).ceil() as i64;
    let hi = (center + width).floor() as i64;
    for xi in lo..=hi {
        x[iu] = xi;
        let d = xi as f64 - center;
        cvp_enum(
            i - 1,
            x,
            mu,
            bnorm,
            tc,
            rho_above + bnorm[iu] * d * d,
            radius2,
            red,
            target,
            best,
            n,
        );
    }
    x[iu] = 0;
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

    /// Brute-force short vectors of a lattice for cross-checking (small bases).
    fn brute(basis: &[Vec<Integer>], bound: i64, range: i64) -> std::collections::HashSet<Vec<Integer>> {
        let n = basis.len();
        let mut set = std::collections::HashSet::new();
        let mut idx = vec![-range; n];
        loop {
            let xc: Vec<i64> = idx.clone();
            let vv = combine(&xc, basis);
            if dot(&vv, &vv) <= Integer::from(bound) && vv.iter().any(|e| !e.is_zero()) {
                set.insert(canonical(vv));
            }
            let mut p = 0;
            while p < n {
                idx[p] += 1;
                if idx[p] > range {
                    idx[p] = -range;
                    p += 1;
                } else {
                    break;
                }
            }
            if p == n {
                break;
            }
        }
        set
    }

    #[test]
    fn short_vectors_z2() {
        // Z^2, bound 2: vectors of norm² ≤ 2: (1,0),(0,1),(1,1),(1,-1)
        let basis = vec![v(&[1, 0]), v(&[0, 1])];
        let got: std::collections::HashSet<Vec<Integer>> =
            short_vectors(&basis, &Integer::from(2)).into_iter().collect();
        let expect = brute(&basis, 2, 3);
        assert_eq!(got, expect);
        assert_eq!(got.len(), 4);
    }

    #[test]
    fn short_vectors_skewed() {
        // a skewed 2D lattice; cross-check against brute force
        let basis = vec![v(&[2, 1]), v(&[1, 3])];
        for bound in [5i64, 10, 20, 50] {
            let got: std::collections::HashSet<Vec<Integer>> =
                short_vectors(&basis, &Integer::from(bound)).into_iter().collect();
            assert_eq!(got, brute(&basis, bound, 8), "bound {bound}");
        }
    }

    #[test]
    fn shortest_vector_finds_minimum() {
        // a rank-2 lattice of determinant 4 (not unimodular)
        let basis = vec![v(&[2, 0]), v(&[1, 2])];
        let s = shortest_vector(&basis);
        // brute-force minimum nonzero norm²
        let min = brute(&basis, 100, 12).iter().map(|v| norm2(v)).min().unwrap();
        assert_eq!(norm2(&s), min);
        assert_eq!(min, Integer::from(4)); // (2,0)
    }

    #[test]
    fn closest_vector_cvp() {
        // lattice 5Z^2, target (12, 8) → closest is (10, 10), dist² 4+4=8? check (10,10) vs (15,10)..
        let basis = vec![v(&[5, 0]), v(&[0, 5])];
        let c = closest_vector(&basis, &v(&[12, 8]));
        // nearest multiples of 5: (10,10) dist²=4+4=8; (15,10) dist²=9+4=13; (10,5) 4+9=13
        assert_eq!(c, v(&[10, 10]));
    }
}

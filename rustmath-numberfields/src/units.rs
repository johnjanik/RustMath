//! Unit group and regulator of a number field `K = ℚ[x]/(f)`.
//!
//! By Dirichlet's unit theorem `O_K^× ≅ μ_K × ℤ^{r₁+r₂−1}`. The **regulator** is the
//! covolume of the unit lattice `Log(O_K^×)` in the trace-zero hyperplane, where
//! `Log(α) = (eᵢ·log|σᵢ(α)|)` (`eᵢ = 1` real, `2` complex). Fundamental units are
//! found by enumerating small units (`|N(α)| = 1`); the regulator is the absolute
//! determinant of the shortest independent log vectors (a fundamental system for
//! small fields). For small fields this is exact-to-precision; large fundamental
//! units (rank ≥ 1 with big regulator) are out of reach of the small-coefficient
//! search — the sub-exponential frontier. Validated against `gp bnfinit`.

use crate::classgroup::element_norm;
use crate::round2::{maximal_order_data, OrderData};
use rustmath_integers::Integer;
use rustmath_polynomials::real_roots::count_real_roots_int;

/// Signature `(r₁, r₂)`: real embeddings and complex-conjugate pairs.
pub fn signature(f: &[Integer]) -> (usize, usize) {
    let n = f.len() - 1;
    let r1 = count_real_roots_int(f);
    (r1, (n - r1) / 2)
}

/// Unit rank `r₁ + r₂ − 1`.
pub fn unit_rank(f: &[Integer]) -> usize {
    let (r1, r2) = signature(f);
    r1 + r2 - 1
}

// minimal complex arithmetic for the embeddings
#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }
    fn sub(self, o: C) -> C {
        C { re: self.re - o.re, im: self.im - o.im }
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C { re: (self.re * o.re + self.im * o.im) / d, im: (self.im * o.re - self.re * o.im) / d }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

fn roots(f: &[Integer]) -> Vec<C> {
    let n = f.len() - 1;
    let c: Vec<f64> = f.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let eval = |z: C| {
        let mut acc = C { re: 1.0, im: 0.0 };
        for k in (0..n).rev() {
            acc = acc.mul(z).add(C { re: c[k], im: 0.0 });
        }
        acc
    };
    let mut z: Vec<C> = (0..n)
        .map(|k| {
            let s = C { re: 0.4, im: 0.9 };
            let mut p = C { re: 1.0, im: 0.0 };
            for _ in 0..k {
                p = p.mul(s);
            }
            p
        })
        .collect();
    for _ in 0..300 {
        let mut md = 0.0f64;
        for i in 0..n {
            let mut den = C { re: 1.0, im: 0.0 };
            for j in 0..n {
                if j != i {
                    den = den.mul(z[i].sub(z[j]));
                }
            }
            let d = eval(z[i]).div(den);
            z[i] = z[i].sub(d);
            md = md.max(d.abs());
        }
        if md < 1e-13 {
            break;
        }
    }
    z
}

fn embeddings(rts: &[C]) -> (Vec<C>, Vec<C>) {
    let mut reals = Vec::new();
    let mut cplx = Vec::new();
    let mut used = vec![false; rts.len()];
    for i in 0..rts.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        if rts[i].im.abs() < 1e-6 {
            reals.push(rts[i]);
        } else {
            cplx.push(rts[i]);
            let mut best = usize::MAX;
            let mut bd = f64::INFINITY;
            for j in 0..rts.len() {
                if j != i && !used[j] {
                    let d = (rts[i].re - rts[j].re).abs() + (rts[i].im + rts[j].im).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
            }
            if best != usize::MAX {
                used[best] = true;
            }
        }
    }
    (reals, cplx)
}

/// `Log(α) = (eᵢ·log|σᵢ(α)|)`, length `r₁+r₂` (real embeddings then complex pairs).
fn log_embedding(ord: &OrderData, alpha: &[Integer], reals: &[C], cplx: &[C]) -> Vec<f64> {
    let n = ord.n;
    let dd = ord.d.to_f64().unwrap_or(1.0);
    // power-coord numerators: pow[i] = Σ_k w[i][k]·α_k
    let pow: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0f64;
            for k in 0..n {
                s += ord.w[i][k].to_f64().unwrap_or(0.0) * alpha[k].to_f64().unwrap_or(0.0);
            }
            s
        })
        .collect();
    let sigma = |r: C| -> C {
        let mut acc = C { re: 0.0, im: 0.0 };
        let mut pw = C { re: 1.0, im: 0.0 };
        for &coef in &pow {
            acc = acc.add(C { re: coef, im: 0.0 }.mul(pw));
            pw = pw.mul(r);
        }
        C { re: acc.re / dd, im: acc.im / dd }
    };
    let mut out = Vec::with_capacity(reals.len() + cplx.len());
    for r in reals {
        out.push(sigma(*r).abs().ln());
    }
    for r in cplx {
        out.push(2.0 * sigma(*r).abs().ln());
    }
    out
}

/// Enumerate units of `O_K` with integral-basis coordinates in `[−b, b]ⁿ`
/// (those with `|N(α)| = 1`), as `(coords, Log(α))`.
fn small_units(
    ord: &OrderData,
    reals: &[C],
    cplx: &[C],
    b: i64,
) -> Vec<(Vec<Integer>, Vec<f64>)> {
    let n = ord.n;
    let mut out = Vec::new();
    let mut idx = vec![-b; n];
    loop {
        let alpha: Vec<Integer> = idx.iter().map(|&x| Integer::from(x)).collect();
        if alpha.iter().any(|x| !x.is_zero()) && element_norm(ord, &alpha).abs().is_one() {
            let lv = log_embedding(ord, &alpha, reals, cplx);
            out.push((alpha, lv));
        }
        let mut p = 0;
        while p < n {
            idx[p] += 1;
            if idx[p] > b {
                idx[p] = -b;
                p += 1;
            } else {
                break;
            }
        }
        if p == n {
            break;
        }
    }
    out
}

/// The regulator of `K`. `1.0` for unit rank 0 (imaginary quadratic / ℚ). For
/// rank `≥ 1`, the covolume of the unit log lattice from small units. `None` if no
/// fundamental system is found within the small-coefficient search.
pub fn regulator(f: &[Integer]) -> Option<f64> {
    let ord = maximal_order_data(f);
    let (r1, r2) = signature(f);
    let rank = r1 + r2 - 1;
    if rank == 0 {
        return Some(1.0);
    }
    let rts = roots(f);
    let (reals, cplx) = embeddings(&rts);
    let b = if f.len() - 1 <= 2 { 40 } else { 6 };
    let units = small_units(&ord, &reals, &cplx, b);
    // log vectors projected to the first `rank` coordinates (drop one — trace-zero)
    let mut vecs: Vec<Vec<f64>> = units
        .iter()
        .filter(|(_a, lv)| lv.iter().any(|x| x.abs() >= 1e-7)) // skip torsion
        .map(|(_a, lv)| lv[..rank].to_vec())
        .collect();
    // shortest first
    vecs.sort_by(|a, b| {
        let na: f64 = a.iter().map(|x| x * x).sum();
        let nb: f64 = b.iter().map(|x| x * x).sum();
        na.partial_cmp(&nb).unwrap()
    });
    // greedily pick `rank` independent vectors (a fundamental system for small fields)
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut ortho: Vec<Vec<f64>> = Vec::new(); // Gram–Schmidt of basis
    for v in vecs {
        let mut r = v.clone();
        for o in &ortho {
            let dot: f64 = r.iter().zip(o).map(|(a, b)| a * b).sum();
            let nn: f64 = o.iter().map(|x| x * x).sum();
            if nn > 1e-18 {
                for i in 0..r.len() {
                    r[i] -= dot / nn * o[i];
                }
            }
        }
        if r.iter().map(|x| x * x).sum::<f64>().sqrt() > 1e-6 {
            ortho.push(r);
            basis.push(v);
            if basis.len() == rank {
                break;
            }
        }
    }
    if basis.len() < rank {
        return None;
    }
    Some(det_f64(&basis).abs())
}

/// Determinant of a small `f64` matrix by Gaussian elimination.
fn det_f64(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut det = 1.0;
    for i in 0..n {
        let mut piv = i;
        for k in i + 1..n {
            if a[k][i].abs() > a[piv][i].abs() {
                piv = k;
            }
        }
        if a[piv][i].abs() < 1e-15 {
            return 0.0;
        }
        if piv != i {
            a.swap(piv, i);
            det = -det;
        }
        det *= a[i][i];
        for k in i + 1..n {
            let f = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= f * a[i][j];
            }
        }
    }
    det
}

/// Number of roots of unity `w_K = |μ_K|`: the units with all `|σᵢ(α)| = 1`.
pub fn roots_of_unity_count(f: &[Integer]) -> usize {
    let ord = maximal_order_data(f);
    let rts = roots(f);
    let (reals, cplx) = embeddings(&rts);
    let b = if f.len() - 1 <= 2 { 6 } else { 4 };
    let units = small_units(&ord, &reals, &cplx, b);
    let mut count = 0usize;
    for (_a, lv) in &units {
        if lv.iter().all(|x| x.abs() < 1e-7) {
            count += 1; // |σ(α)| = 1 for every embedding ⇒ root of unity
        }
    }
    count.max(2) // ±1 always present
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn signature_and_rank() {
        assert_eq!(signature(&iz(&[-2, 0, 1])), (2, 0)); // Q(sqrt2): totally real
        assert_eq!(signature(&iz(&[1, 0, 1])), (0, 1)); // Q(i)
        assert_eq!(signature(&iz(&[-1, -1, 0, 1])), (1, 1)); // complex cubic
        assert_eq!(unit_rank(&iz(&[-2, 0, 1])), 1);
        assert_eq!(unit_rank(&iz(&[1, 0, 1])), 0);
    }

    #[test]
    fn regulator_matches_gp() {
        assert!(close(regulator(&iz(&[1, 0, 1])).unwrap(), 1.0)); // Q(i), rank 0
        assert!(close(regulator(&iz(&[5, 0, 1])).unwrap(), 1.0)); // Q(sqrt-5), rank 0
        assert!(close(regulator(&iz(&[-2, 0, 1])).unwrap(), 0.88137359)); // Q(sqrt2)
        assert!(close(regulator(&iz(&[-5, 0, 1])).unwrap(), 0.48121183)); // Q(sqrt5)
        assert!(close(regulator(&iz(&[-3, 0, 1])).unwrap(), 1.31695790)); // Q(sqrt3)
        assert!(close(regulator(&iz(&[-1, -1, 0, 1])).unwrap(), 0.28119957)); // x^3-x-1
    }

    #[test]
    fn roots_of_unity() {
        assert_eq!(roots_of_unity_count(&iz(&[1, 0, 1])), 4); // Q(i): ±1, ±i
        assert_eq!(roots_of_unity_count(&iz(&[-2, 0, 1])), 2); // Q(sqrt2): ±1
        assert_eq!(roots_of_unity_count(&iz(&[5, 0, 1])), 2); // Q(sqrt-5): ±1
    }
}

/// Sign vector of `α` at the real embeddings (`0` if `σᵢ(α) > 0`, `1` if `< 0`).
fn sign_vector(ord: &OrderData, alpha: &[Integer], reals: &[C]) -> Vec<u8> {
    let n = ord.n;
    let dd = ord.d.to_f64().unwrap_or(1.0);
    let pow: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0f64;
            for k in 0..n {
                s += ord.w[i][k].to_f64().unwrap_or(0.0) * alpha[k].to_f64().unwrap_or(0.0);
            }
            s
        })
        .collect();
    reals
        .iter()
        .map(|r| {
            let mut acc = 0.0f64;
            let mut pw = 1.0f64;
            for &coef in &pow {
                acc += coef * pw;
                pw *= r.re;
            }
            if acc / dd < 0.0 {
                1
            } else {
                0
            }
        })
        .collect()
}

/// `F₂`-rank of a set of sign vectors.
fn f2_rank(vecs: &[Vec<u8>], dim: usize) -> usize {
    let mut rows: Vec<Vec<u8>> = vecs.to_vec();
    let mut rank = 0usize;
    for col in 0..dim {
        let mut piv = None;
        for r in rank..rows.len() {
            if rows[r][col] == 1 {
                piv = Some(r);
                break;
            }
        }
        if let Some(p) = piv {
            rows.swap(rank, p);
            for r in 0..rows.len() {
                if r != rank && rows[r][col] == 1 {
                    for c in 0..dim {
                        rows[r][c] ^= rows[rank][c];
                    }
                }
            }
            rank += 1;
        }
    }
    rank
}

/// The narrow class number `h⁺ = h · 2^{r₁ − rank(sign map)}`, where the sign map
/// sends `O_K^×` to its vector of signs at the real places. Equals `h` for totally
/// imaginary fields and when a unit of norm `−1` exists; doubles (per independent
/// sign) otherwise. `None` if the class number is unavailable. Validated vs
/// `gp bnfnarrow`.
pub fn narrow_class_number(f: &[Integer]) -> Option<usize> {
    let cg = crate::classgroup::class_group(f)?;
    let h: usize = cg.iter().product::<usize>().max(1);
    let (r1, _r2) = signature(f);
    if r1 == 0 {
        return Some(h);
    }
    let ord = maximal_order_data(f);
    let rts = roots(f);
    let (reals, cplx) = embeddings(&rts);
    let b = if f.len() - 1 <= 2 { 30 } else { 6 };
    let units = small_units(&ord, &reals, &cplx, b);
    let sign_vecs: Vec<Vec<u8>> = units.iter().map(|(a, _)| sign_vector(&ord, a, &reals)).collect();
    let rank = f2_rank(&sign_vecs, r1);
    Some(h * (1usize << (r1 - rank)))
}

#[cfg(test)]
mod narrow_tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn narrow_class_number_matches_gp() {
        assert_eq!(narrow_class_number(&iz(&[-2, 0, 1])), Some(1)); // Q(sqrt2)
        assert_eq!(narrow_class_number(&iz(&[-3, 0, 1])), Some(2)); // Q(sqrt3): norm+1 unit
        assert_eq!(narrow_class_number(&iz(&[-5, 0, 1])), Some(1)); // Q(sqrt5)
        assert_eq!(narrow_class_number(&iz(&[1, 0, 1])), Some(1)); // Q(i)
        assert_eq!(narrow_class_number(&iz(&[5, 0, 1])), Some(2)); // Q(sqrt-5)
        assert_eq!(narrow_class_number(&iz(&[-10, 0, 1])), Some(2)); // Q(sqrt10)
        assert_eq!(narrow_class_number(&iz(&[6, -1, 1])), Some(3)); // Q(sqrt-23)
        assert_eq!(narrow_class_number(&iz(&[-79, 0, 1])), Some(6)); // Q(sqrt79): h=3, h+=6
    }
}

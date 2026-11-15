//! `#X0_p(F_q)` for a curve in `P^1 x P^1` by fibring over one of the two lines, with
//! Frobenius-orbit weighting.
//!
//! Reference algorithm: `m23golay/curve.py`, `count_points` (johnjanik/M23 at `d05475b`), with
//! `fibre = "V"` for [`Fibration::FibresInX`] on the M23 form (`X = T` of degree 4, `Y = V`).
//! Root counts: `rustmath_finitefields::small_field` (MAGMA ch. 21; the WP-F kernel).

use super::{CountError, Degeneracy, FormModP};
use rayon::prelude::*;
use rustmath_finitefields::small_field::{count_distinct_roots, SmallFq, SmallFqElt};

/// Which variable the fibres are polynomials in; the other variable is the base.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fibration {
    /// Sum over `y` in `P^1(F_q)` of the roots in `x` (fibres of degree `dx`). For the M23 form
    /// this is `curve.py` with `fibre = "V"` (quartic fibres).
    FibresInX,
    /// Sum over `x` in `P^1(F_q)` of the roots in `y` (fibres of degree `dy`).
    FibresInY,
}

/// Distinct roots in `P^1(F_q)` of `sum c_i x^i`, a binary form of formal degree `m`.
#[inline(always)]
fn projective_roots<const K: usize>(
    fq: &SmallFq<K>,
    c: &[SmallFqElt<K>],
    m: usize,
) -> Option<usize> {
    let mut d = m as isize;
    while d >= 0 && fq.is_zero(&c[d as usize]) {
        d -= 1;
    }
    if d < 0 {
        return None;
    }
    let at_infinity = usize::from((d as usize) < m);
    Some(at_infinity + count_distinct_roots(fq, &c[..=(d as usize)]).expect("nonzero"))
}

/// The fibre coefficients `c_i(v) = sum_j cols[i][j] v^j` at the base point `v`.
#[inline(always)]
fn fibre_at<const K: usize>(
    fq: &SmallFq<K>,
    cols: &[Vec<u64>],
    v: &SmallFqElt<K>,
    pw: &mut [SmallFqElt<K>],
    out: &mut [SmallFqElt<K>],
) {
    let n = pw.len() - 1;
    pw[0] = fq.one();
    for j in 1..=n {
        pw[j] = fq.mul(&pw[j - 1], v);
    }
    for (i, col) in cols.iter().enumerate() {
        // Lazy accumulation: at most 4097 products below 2^32 each.
        let mut acc = [0u64; K];
        for (j, &cij) in col.iter().enumerate() {
            if cij == 0 {
                continue;
            }
            for u in 0..K {
                acc[u] += cij * pw[j][u];
            }
        }
        let mut r = [0u64; K];
        for u in 0..K {
            r[u] = fq.reduce(acc[u]);
        }
        out[i] = r;
    }
}

fn chunk_count<const K: usize>(
    fq: &SmallFq<K>,
    cols: &[Vec<u64>],
    lo: u64,
    hi: u64,
) -> Result<u64, CountError> {
    let m = cols.len() - 1;
    let n = cols[0].len() - 1;
    let mut pw = vec![[0u64; K]; n + 1];
    let mut fib = vec![[0u64; K]; m + 1];
    let mut total = 0u64;
    for code in lo..hi {
        let v = fq.from_code(code);
        // Process v only if its code is the least in its Frobenius orbit; weight by the size.
        let mut size = 1u64;
        let mut minimal = true;
        let mut x = v;
        for _ in 1..K {
            x = fq.frobenius(&x);
            let c = fq.code(&x);
            if c == code {
                break;
            }
            if c < code {
                minimal = false;
                break;
            }
            size += 1;
        }
        if !minimal {
            continue;
        }
        fibre_at(fq, cols, &v, &mut pw, &mut fib);
        match projective_roots(fq, &fib, m) {
            Some(r) => total += size * r as u64,
            None => {
                return Err(CountError::DegenerateModel(Degeneracy::FibreComponent {
                    place: format!(
                        "base point with code {code} in GF({}^{K})",
                        fq.characteristic()
                    ),
                }))
            }
        }
    }
    Ok(total)
}

/// `cols[i]` = the coefficients in the base variable of the `i`-th power of the fibre variable.
fn count_k<const K: usize>(p: u64, cols: &[Vec<u64>]) -> Result<u64, CountError> {
    let fq = SmallFq::<K>::new(p).map_err(|_| CountError::FieldTooLarge { p, k: K as u32 })?;
    let q = fq.order();
    let chunk = 1u64 << 14;
    let nchunks = q.div_ceil(chunk);
    let parts: Result<Vec<u64>, CountError> = (0..nchunks)
        .into_par_iter()
        .map(|c| chunk_count(&fq, cols, c * chunk, ((c + 1) * chunk).min(q)))
        .collect();
    let mut total: u64 = parts?.iter().sum();
    // The base point at infinity: the leading coefficients in the base variable.
    let m = cols.len() - 1;
    let n = cols[0].len() - 1;
    let lead: Vec<SmallFqElt<K>> = (0..=m).map(|i| fq.from_fp(cols[i][n])).collect();
    match projective_roots(&fq, &lead, m) {
        Some(r) => total += r as u64,
        None => {
            return Err(CountError::DegenerateModel(Degeneracy::FibreComponent {
                place: "base point at infinity".into(),
            }))
        }
    }
    Ok(total)
}

/// `#X0_p(F_q)`, `q = p^k`, the number of `F_q`-points of the plane model `V(F mod p)` in
/// `P^1 x P^1` (not of its normalization), by summing the distinct roots in `P^1(F_q)` of every
/// fibre. Runs on the current rayon pool, in chunks of `2^14` base points.
///
/// Errors: `k = 0`, `k > 8` or `p^k >= 2^63` (`FieldTooLarge`); a fibre that vanishes
/// identically (`DegenerateModel(FibreComponent)`).
pub fn plane_point_count(f: &FormModP, k: u32, fib: Fibration) -> Result<u64, CountError> {
    let p = f.p();
    let g = match fib {
        Fibration::FibresInX => f.clone(),
        Fibration::FibresInY => f.transpose(),
    };
    let cols = g.coeffs();
    match k {
        1 => count_k::<1>(p, cols),
        2 => count_k::<2>(p, cols),
        3 => count_k::<3>(p, cols),
        4 => count_k::<4>(p, cols),
        5 => count_k::<5>(p, cols),
        6 => count_k::<6>(p, cols),
        7 => count_k::<7>(p, cols),
        8 => count_k::<8>(p, cols),
        _ => Err(CountError::FieldTooLarge { p, k }),
    }
}

//! Factorization of `f ∈ ℤ[x]` over `ℚ_p` into local factors with their
//! `(e, f, root-valuation)` data, and the Frobenius cycle type at `p`.
//!
//! Generalizes the p-adelic engine's `PadicFactorizer.swift`, which handled only
//! the single-segment unramified and pure-Eisenstein shapes and deferred every
//! mixed multi-segment polygon. Here each Newton-polygon segment is processed via
//! its **Ore residual polynomial**: a segment of slope `−h/e` (lowest terms) and
//! horizontal length `ℓ = e·t` has a residual polynomial `R ∈ F_p[y]` of degree
//! `t`; in the *regular* case (`R` separable) each irreducible factor of `R` of
//! degree `s` yields one `ℚ_p`-factor of `f` with ramification `e`, residue degree
//! `s`, and root valuation `h/e`. This uniformly covers unramified (slope 0,
//! `e = 1`), Eisenstein (`deg R = 1`), and regular mixed polygons. The genuinely
//! non-regular case (`R` inseparable — `p` divides the index, higher-order Montes
//! required) is reported as [`PadicError::NonRegularSegment`].
//!
//! For `p` unramified in `f` (`p ∤ disc f`), the residue degrees of the factors
//! are the cycle type of Frobenius at `p` — exposed cheaply by [`cycle_type`], the
//! native replacement for the PARI-via-Sage Frobenius screen.

use crate::fp_factor;
use crate::newton::{newton_polygon, Segment};
use rustmath_integers::Integer;

/// One irreducible factor of `f` over `ℚ_p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalFactor {
    /// Ramification index `e ≥ 1`.
    pub e: usize,
    /// Residue degree `f ≥ 1`.
    pub f: usize,
    /// Degree over `ℚ_p` (`= e·f`).
    pub degree: usize,
    /// Common root valuation `h/e` (in lowest terms).
    pub root_val_num: i64,
    pub root_val_den: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicError {
    ZeroPolynomial,
    /// A segment's residual polynomial is inseparable — `p` divides the index
    /// `[O_K : ℤ[α]]` and first-order Ore data is insufficient (higher-order
    /// Montes required). The first-order `(e,f)` for this segment is not returned.
    NonRegularSegment { slope_num: i64, slope_den: i64 },
    Internal(String),
}

/// Reduce `f` to `F_p[x]` coefficients in `[0, p)`, little-endian, trimmed.
fn reduce_mod_p(f: &[Integer], p: i64) -> Vec<i64> {
    let pi = Integer::from(p);
    let v: Vec<i64> = f
        .iter()
        .map(|c| {
            let r = (c.clone() % pi.clone()).to_i64();
            ((r % p) + p) % p
        })
        .collect();
    fp_factor::trim(&v)
}

/// Frobenius cycle type of `f` at an **unramified** prime `p`: the sorted multiset
/// of residue degrees of `f` mod `p`. Returns `None` when `p` is ramified or
/// divides the leading coefficient (i.e. `f` is not separable mod `p`), since the
/// cycle type is only defined at unramified primes.
pub fn cycle_type(f: &[Integer], p: i64) -> Option<Vec<usize>> {
    let n = (f.len() as i64) - 1;
    let fp = reduce_mod_p(f, p);
    if fp_factor::degree(&fp) != n {
        return None; // p divides the leading coefficient
    }
    // separable mod p?  gcd(f, f') constant
    let g = fp_factor::gcd(&fp, &fp_factor::derivative_of(&fp, p), p);
    if fp_factor::degree(&g) != 0 {
        return None; // p is ramified
    }
    let mut degs: Vec<usize> = fp_factor::factor(&fp, p)
        .iter()
        .map(|g| (g.len() - 1).max(0))
        .collect();
    degs.sort_unstable();
    Some(degs)
}

/// Ore residual polynomial of a segment, in `F_p[y]` (little-endian, `[0, p)`).
fn residual_polynomial(f: &[Integer], p: i64, seg: &Segment, e: usize, h: i64) -> Vec<i64> {
    let i0 = seg.from.i;
    let v0 = seg.from.v;
    let t = seg.length / e;
    let pi = Integer::from(p);
    let mut r = vec![0i64; t + 1];
    for j in 0..=t {
        let idx = i0 + j * e;
        let coeff = &f[idx];
        if coeff.is_zero() {
            continue; // above the hull
        }
        let exponent = v0 - (j as i64) * h; // ≥ 0 along the segment
        // a_idx / p^exponent  (exact; reduces above-hull points to 0 mod p)
        let mut q = coeff.clone();
        for _ in 0..exponent {
            q = q / pi.clone();
        }
        let m = (q % pi.clone()).to_i64();
        r[j] = ((m % p) + p) % p;
    }
    fp_factor::trim(&r)
}

/// Is the `F_p[y]` polynomial separable (square-free)?
fn is_separable_fp(r: &[i64], p: i64) -> bool {
    if fp_factor::degree(r) <= 0 {
        return true;
    }
    let d = fp_factor::derivative_of(r, p);
    if fp_factor::is_zero(&d) {
        return false; // a p-th power
    }
    fp_factor::degree(&fp_factor::gcd(r, &d, p)) == 0
}

/// Factor `f` over `ℚ_p`, returning the local factors with `(e, f, root valuation)`.
///
/// Regular segments are fully resolved. A non-regular segment (inseparable
/// residual polynomial) yields [`PadicError::NonRegularSegment`].
pub fn padic_factor(f: &[Integer], p: i64) -> Result<Vec<LocalFactor>, PadicError> {
    let polygon = newton_polygon(f, p).ok_or(PadicError::ZeroPolynomial)?;
    let mut out = Vec::new();
    for seg in &polygon.segments {
        // slope = slope_num/slope_den (≤ 0 for the lower hull); root valuation = h/e
        let e = seg.slope_den as usize;
        let h = -seg.slope_num; // ≥ 0
        if e == 0 || seg.length % e != 0 {
            return Err(PadicError::Internal(format!(
                "segment length {} not divisible by ramification {}",
                seg.length, e
            )));
        }
        let r = residual_polynomial(f, p, seg, e, h);
        if !is_separable_fp(&r, p) {
            return Err(PadicError::NonRegularSegment {
                slope_num: seg.slope_num,
                slope_den: seg.slope_den,
            });
        }
        for g in fp_factor::factor(&r, p) {
            let s = (g.len() - 1).max(0);
            out.push(LocalFactor {
                e,
                f: s,
                degree: e * s,
                root_val_num: h, // gcd(h,e)=1 since slope is in lowest terms
                root_val_den: seg.slope_den,
            });
        }
    }
    Ok(out)
}

/// Convenience: the sorted multiset of `(e, f)` pairs from [`padic_factor`].
pub fn ramification_type(f: &[Integer], p: i64) -> Result<Vec<(usize, usize)>, PadicError> {
    let mut ef: Vec<(usize, usize)> = padic_factor(f, p)?.iter().map(|lf| (lf.e, lf.f)).collect();
    ef.sort_unstable();
    Ok(ef)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(cs: &[i64]) -> Vec<Integer> {
        cs.iter().map(|&c| Integer::from(c)).collect()
    }

    #[test]
    fn test_cycle_type_unramified() {
        // x^2 + 1 at p=5: splits (4 = -1 is a QR mod 5) → two linear → [1,1]
        assert_eq!(cycle_type(&p(&[1, 0, 1]), 5), Some(vec![1, 1]));
        // x^2 + 1 at p=7: inert (-1 not a QR mod 7) → one quadratic → [2]
        assert_eq!(cycle_type(&p(&[1, 0, 1]), 7), Some(vec![2]));
    }

    #[test]
    fn test_cycle_type_ramified_none() {
        // x^2 + 1 at p=2: ramified (x^2+1 = (x+1)^2 mod 2) → None
        assert_eq!(cycle_type(&p(&[1, 0, 1]), 2), None);
    }

    #[test]
    fn test_padic_unramified_split() {
        // x^2 + 1 over Q_5: two unramified linear factors (e=1,f=1)
        let ef = ramification_type(&p(&[1, 0, 1]), 5).unwrap();
        assert_eq!(ef, vec![(1, 1), (1, 1)]);
    }

    #[test]
    fn test_padic_unramified_inert() {
        // x^2 + 1 over Q_7: one unramified factor (e=1,f=2)
        let ef = ramification_type(&p(&[1, 0, 1]), 7).unwrap();
        assert_eq!(ef, vec![(1, 2)]);
    }

    #[test]
    fn test_padic_eisenstein_totally_ramified() {
        // x^3 + 3x + 3 over Q_3: Eisenstein → totally ramified (e=3,f=1)
        let ef = ramification_type(&p(&[3, 3, 0, 1]), 3).unwrap();
        assert_eq!(ef, vec![(3, 1)]);
    }

    #[test]
    fn test_padic_mixed_segments() {
        // (x^2+3)(x-1) = x^3 - x^2 + 3x - 3 over Q_3:
        //   x-1 is unramified (e=1,f=1); x^2+3 is Eisenstein (e=2,f=1).
        let ef = ramification_type(&p(&[-3, 3, -1, 1]), 3).unwrap();
        assert_eq!(ef, vec![(1, 1), (2, 1)]);
    }

    #[test]
    fn test_padic_unramified_split_qr() {
        // x^2 + 2 over Q_3: −2 ≡ 1 is a QR mod 3, so x^2+2 = (x-1)(x+1) mod 3 →
        // two unramified linear factors (e=1,f=1). (x^2+5, where −5≡1 likewise.)
        let ef = ramification_type(&p(&[2, 0, 1]), 3).unwrap();
        assert_eq!(ef, vec![(1, 1), (1, 1)]);
    }
}

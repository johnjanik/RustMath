//! KMSV §5 (genus 0): assemble the rational Belyi map from the modular forms.
//!
//! Given a basis of S_k(Γ) (from [`super::modular_forms_hp::recover_forms`]), echelonize
//! by w-valuation to get g(w) = w^m + O(w^{m+2e}), h(w) = w^{m+e} + O(w^{m+2e}) (eq 5.5),
//! form the coordinate x(w) on X(Γ) ≅ P¹, and (§5b) match φ_Δ(x(w)) = φ(w) against the
//! exact hypergeometric series to recover the rational map Φ(x). This module is the
//! genus-0 assembly; hp complex power-series arithmetic is done in place on Vec<Complex>.

use rug::{Complex, Float};

/// First index with |coeff| above `tol` (the w-valuation); `len` if all tiny.
fn valuation(s: &[Complex], tol: f64) -> usize {
    s.iter()
        .position(|c| {
            let re = c.real().to_f64();
            let im = c.imag().to_f64();
            (re * re + im * im).sqrt() > tol
        })
        .unwrap_or(s.len())
}

/// Truncated product of two power series to `len` terms.
fn series_mul(a: &[Complex], b: &[Complex], len: usize, prec: u32) -> Vec<Complex> {
    let mut out = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    for i in 0..a.len().min(len) {
        for j in 0..b.len().min(len - i) {
            let t = Complex::with_val(prec, &a[i] * &b[j]);
            out[i + j] = Complex::with_val(prec, &out[i + j] + &t);
        }
    }
    out
}

/// Reciprocal 1/s of a unit power series (s[0] ≠ 0), to `len` terms.
fn unit_recip(s: &[Complex], len: usize, prec: u32) -> Vec<Complex> {
    let mut r = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    let s0_inv = Complex::with_val(prec, Complex::with_val(prec, (1.0, 0.0)) / &s[0]);
    r[0] = s0_inv.clone();
    for n in 1..len {
        let mut acc = Complex::with_val(prec, (0.0, 0.0));
        for j in 1..=n {
            if j < s.len() {
                let t = Complex::with_val(prec, &s[j] * &r[n - j]);
                acc = Complex::with_val(prec, &acc + &t);
            }
        }
        r[n] = Complex::with_val(prec, -Complex::with_val(prec, &s0_inv * &acc));
    }
    r
}

/// Power-series quotient num/den (valuation(num) ≥ valuation(den)), to `len` terms.
fn series_div(num: &[Complex], den: &[Complex], len: usize, prec: u32, tol: f64) -> Vec<Complex> {
    let vn = valuation(num, tol);
    let vd = valuation(den, tol);
    assert!(vn >= vd, "series_div: numerator valuation {vn} < denominator {vd}");
    let num_s = &num[vn..];
    let den_s = &den[vd..];
    let inv = unit_recip(den_s, len, prec);
    let q = series_mul(num_s, &inv, len, prec);
    // shift up by (vn - vd)
    let shift = vn - vd;
    let mut out = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    for i in 0..len.saturating_sub(shift) {
        out[i + shift] = q[i].clone();
    }
    out
}

/// Convert recovered forms (coefficient vectors b, f = (1−w)^k Σ b_n w^n) into ordinary
/// power series Σ c_n w^n by multiplying out the (1−w)^k automorphy factor.
pub fn forms_to_series(forms: &[Vec<Complex>], k: i64, prec: u32) -> Vec<Vec<Complex>> {
    let len = forms[0].len();
    // (1−w)^k = Σ_j binom(k,j) (−1)^j w^j
    let mut omw = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    let mut binom = 1i128;
    for j in 0..=(k as usize).min(len - 1) {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        omw[j] = Complex::with_val(prec, (sign * binom as f64, 0.0));
        binom = binom * (k as i128 - j as i128) / (j as i128 + 1);
    }
    forms.iter().map(|b| series_mul(&omw, b, len, prec)).collect()
}

/// Echelonize a set of power series into reduced row-echelon form by w-valuation:
/// each returned series is monic at a distinct increasing valuation, with that valuation
/// zeroed in all others. For dim d this yields leading valuations 0,1,…,d−1 (when the
/// space is spanned by such). Returns the series sorted by valuation.
pub fn echelonize(mut rows: Vec<Vec<Complex>>, prec: u32, tol: f64) -> Vec<Vec<Complex>> {
    let d = rows.len();
    let len = rows[0].len();
    let mut pivot_of_row = vec![0usize; d];
    let mut done = 0usize;
    while done < d {
        // pick the row (≥ done) of smallest valuation
        let mut best = done;
        let mut best_val = valuation(&rows[done], tol);
        for r in (done + 1)..d {
            let v = valuation(&rows[r], tol);
            if v < best_val {
                best_val = v;
                best = r;
            }
        }
        rows.swap(done, best);
        let piv = best_val;
        pivot_of_row[done] = piv;
        // normalize row `done` to be monic at its pivot
        let lead_inv = Complex::with_val(prec, Complex::with_val(prec, (1.0, 0.0)) / &rows[done][piv]);
        for j in 0..len {
            rows[done][j] = Complex::with_val(prec, &rows[done][j] * &lead_inv);
        }
        // eliminate this pivot column from every other row
        for r in 0..d {
            if r != done {
                let factor = rows[r][piv].clone();
                let fnorm = {
                    let re = factor.real().to_f64();
                    let im = factor.imag().to_f64();
                    (re * re + im * im).sqrt()
                };
                if fnorm > 0.0 {
                    for j in 0..len {
                        let t = Complex::with_val(prec, &factor * &rows[done][j]);
                        rows[r][j] = Complex::with_val(prec, &rows[r][j] - &t);
                    }
                }
            }
        }
        done += 1;
    }
    // sort by valuation
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by_key(|&i| valuation(&rows[i], tol));
    order.into_iter().map(|i| rows[i].clone()).collect()
}

/// The coordinate x(w) = Θ^e·h(w)/(g(w) + c·h(w)) on X(Γ) (eq 5.5), with c the
/// coefficient of w^{m+2e} in h — the `+c·h` cancels the w^{m+e} term so x(w) =
/// (Θw)^e + O(w^{3e}). Here Θ is left as 1 (an overall scale). For the (5,3,3) genus-0
/// case g,h are the valuation-1 and valuation-2 echelon forms (m=1, e=1, so c = [w³]h).
pub fn coordinate_x(echelon: &[Vec<Complex>], m: usize, e: usize, prec: u32, tol: f64) -> Vec<Complex> {
    let len = echelon[0].len();
    let g = echelon.iter().find(|s| valuation(s, tol) == m).expect("valuation-m form");
    let h = echelon.iter().find(|s| valuation(s, tol) == m + e).expect("valuation-(m+e) form");
    let c = h[m + 2 * e].clone(); // coefficient of w^{m+2e} in h
    // denom = g + c·h
    let mut denom = g.clone();
    for j in 0..len {
        let t = Complex::with_val(prec, &c * &h[j]);
        denom[j] = Complex::with_val(prec, &denom[j] + &t);
    }
    series_div(h, &denom, len, prec, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::coset_graph::CosetGraph;
    use crate::belyi::modular_forms_hp::recover_forms;
    use crate::belyi::triangle_group::TriangleGroup;
    use crate::belyi::triangle_group_hp::TriangleGroupHp;

    // Validate the forms→coordinate pipeline against the paper's (5,3,3) x(w) (eq 5.10):
    //   x(w) = Θw − (3/2)Θ³w³ + (81/16)Θ⁵w⁵ − (189/16)Θ⁷w⁷ + …
    // The Θ-INDEPENDENT signature: x is odd, and c₅ = (9/4)c₃², c₇ = (7/2)c₃³.
    #[test]
    fn coordinate_x_matches_paper_5_3_3() {
        let prec = 256u32;
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);

        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        assert_eq!(forms.len(), 3);
        let series = forms_to_series(&forms, 6, prec);
        let ech = echelonize(series, prec, 1e-25);
        // leading valuations should be 0,1,2
        let vals: Vec<usize> = ech.iter().map(|s| valuation(s, 1e-25)).collect();
        assert_eq!(vals, vec![0, 1, 2], "echelon valuations");

        let x = coordinate_x(&ech, 1, 1, prec, 1e-25);
        let cf = |n: usize| -> num_complex::Complex64 {
            num_complex::Complex64::new(x[n].real().to_f64(), x[n].imag().to_f64())
        };
        let (c1, c3, c5, c7) = (cf(1), cf(3), cf(5), cf(7));
        // x is odd: even coefficients vanish
        for n in [0usize, 2, 4, 6, 8] {
            assert!(cf(n).norm() < 1e-12, "x[{n}] should vanish, got {}", cf(n).norm());
        }
        // leading coefficient is nonzero
        assert!(c1.norm() > 1e-6);
        // Θ-independent relations from eq 5.10
        let r5 = (c5 - 2.25 * c3 * c3).norm() / c5.norm();
        let r7 = (c7 - 3.5 * c3 * c3 * c3).norm() / c7.norm();
        assert!(r5 < 1e-10, "c5 = (9/4)c3² failed, rel {r5:.2e}");
        assert!(r7 < 1e-10, "c7 = (7/2)c3³ failed, rel {r7:.2e}");
    }
}

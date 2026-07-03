//! KMSV §5 (genus 0): assemble the rational Belyi map from the modular forms.
//!
//! Given a basis of S_k(Γ) (from [`super::modular_forms_hp::recover_forms`]), echelonize
//! by w-valuation to get g(w) = w^m + O(w^{m+2e}), h(w) = w^{m+e} + O(w^{m+2e}) (eq 5.5),
//! form the coordinate x(w) on X(Γ) ≅ P¹, and (§5b) match φ_Δ(x(w)) = φ(w) against the
//! exact hypergeometric series to recover the rational map Φ(x). This module is the
//! genus-0 assembly; hp complex power-series arithmetic is done in place on Vec<Complex>.

use super::mp_svd::{jacobi_svd, JacobiSvdOptions, MpC, MpMatrix};
use super::triangle_group_hp::TriangleGroupHp;
use rug::{Complex, Float};

/// Solve for the rational Belyi map Φ(x) = P(x)/Q(x) (deg ≤ d each) from the power
/// series x(w) and φ(w), via the linear relation φ(w)·Q(x(w)) − P(x(w)) = 0. The
/// coefficient-of-w^n equations form a homogeneous system whose 1-D null space (hp SVD)
/// gives (P, Q) up to scale. Returns (P coeffs p_0..p_d, Q coeffs q_0..q_d).
pub fn solve_belyi_map(x: &[Complex], phi: &[Complex], d: usize, prec: u32) -> (Vec<Complex>, Vec<Complex>) {
    let len = x.len();
    // x^i for i = 0..=d
    let mut xpow = vec![vec![Complex::with_val(prec, (1.0, 0.0)); len]];
    for i in 1..=d {
        xpow.push(series_mul(&xpow[i - 1], x, len, prec));
    }
    // φ·x^i
    let phixi: Vec<Vec<Complex>> = (0..=d).map(|i| series_mul(phi, &xpow[i], len, prec)).collect();
    let ncols = 2 * (d + 1); // [q_0..q_d, p_0..p_d]
    let nrows = ncols + 6;
    let mut data = Vec::with_capacity(nrows * ncols);
    for n in 0..nrows {
        for i in 0..=d {
            data.push(MpC::new(phixi[i][n].real().clone(), phixi[i][n].imag().clone()));
        }
        for i in 0..=d {
            data.push(MpC::new(
                Float::with_val(prec, -xpow[i][n].real()),
                Float::with_val(prec, -xpow[i][n].imag()),
            ));
        }
    }
    let mat = MpMatrix::from_row_major(nrows, ncols, prec, data).expect("system matrix");
    let opt = JacobiSvdOptions::new(prec, 80, "1e-70", "1e-40");
    let svd = jacobi_svd(&mat, &opt).expect("svd");
    let last = ncols - 1; // smallest singular value ⇒ null vector
    let take = |i: usize| {
        let z = svd.v.get(i, last);
        Complex::with_val(prec, (z.re.clone(), z.im.clone()))
    };
    let q: Vec<Complex> = (0..=d).map(take).collect();
    let p: Vec<Complex> = (0..=d).map(|i| take(d + 1 + i)).collect();
    (p, q)
}

/// Convert an exact `Rational` to a high-precision `Float` (via decimal strings).
fn rat_to_float(r: &rustmath_rationals::Rational, prec: u32) -> Float {
    let num = Float::with_val(prec, Float::parse(r.numerator().to_string()).unwrap());
    let den = Float::with_val(prec, Float::parse(r.denominator().to_string()).unwrap());
    Float::with_val(prec, &num / &den)
}

/// κ (eq 4.13): κ = ((μ−1)/(μ+1))·Γ(2−C)Γ(C−A)Γ(C−B)/(Γ(1−A)Γ(1−B)Γ(C)), with
/// A,B,C the hypergeometric parameters (eq 4.10). Computed in hp via the rug Γ-function.
pub fn kappa(tg: &TriangleGroupHp) -> Float {
    let prec = tg.prec;
    let f = |v: f64| Float::with_val(prec, v);
    let inv = |x: f64| Float::with_val(prec, f(1.0) / f(x));
    let (a, b, c) = (tg.a as f64, tg.b as f64, tg.c as f64);
    let big_a = Float::with_val(prec, f(0.5) * Float::with_val(prec, f(1.0) + inv(a) - inv(b) - inv(c)));
    let big_b = Float::with_val(prec, f(0.5) * Float::with_val(prec, f(1.0) + inv(a) - inv(b) + inv(c)));
    let big_c = Float::with_val(prec, f(1.0) + inv(a));
    let gam = |x: Float| x.gamma();
    let num = gam(Float::with_val(prec, f(2.0) - &big_c))
        * gam(Float::with_val(prec, Float::with_val(prec, &big_c - &big_a)))
        * gam(Float::with_val(prec, &big_c - &big_b));
    let den = gam(Float::with_val(prec, f(1.0) - &big_a))
        * gam(Float::with_val(prec, f(1.0) - &big_b))
        * gam(big_c.clone());
    let ratio = Float::with_val(
        prec,
        Float::with_val(prec, &tg.mu - f(1.0)) / Float::with_val(prec, &tg.mu + f(1.0)),
    );
    Float::with_val(prec, &ratio * Float::with_val(prec, num / den))
}

/// The Δ-uniformizer φ as a power series in w: φ(w) = Σ φ_n (w/κ)^{a·n} (eq 4.16),
/// with the exact rational coefficients φ_n from the hypergeometric expansion scaled by
/// κ^{-p}. Real coefficients when κ is real.
pub fn phi_w(a: i64, b: i64, c: i64, kappa: &Float, prec: u32, len: usize) -> Vec<Complex> {
    let phi_u = super::hypergeometric::phi_in_u(a, b, c, len);
    let inv_kappa = Float::with_val(prec, Float::with_val(prec, 1.0) / kappa);
    let mut kpow = vec![Float::with_val(prec, 1.0); len];
    for p in 1..len {
        kpow[p] = Float::with_val(prec, &kpow[p - 1] * &inv_kappa);
    }
    let mut out = vec![Complex::with_val(prec, (0.0, 0.0)); len];
    for p in 0..len {
        let rf = rat_to_float(phi_u.coeff(p), prec);
        let coeff = Float::with_val(prec, &rf * &kpow[p]);
        out[p] = Complex::with_val(prec, (coeff, 0.0));
    }
    out
}

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

    // Shared (5,3,3) setup: returns (x_paper = Θ·x₀, φ(w), Θ).
    fn setup_5_3_3(prec: u32) -> (Vec<Complex>, Vec<Complex>, Complex) {
        use rug::float::Constant;
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        let ech = echelonize(forms_to_series(&forms, 6, prec), prec, 1e-25);
        let x0 = coordinate_x(&ech, 1, 1, prec, 1e-25);
        let len = x0.len();
        let kap = kappa(&tg);
        let alpha = Float::with_val(prec, Float::with_val(prec, 81.0) / 2.0);
        let root = (alpha.clone().ln() / Float::with_val(prec, 5.0)).exp();
        let pi = Float::with_val(prec, Constant::Pi);
        let ang = Float::with_val(prec, Float::with_val(prec, &pi * 2.0) / 5.0);
        let scale = Float::with_val(prec, Float::with_val(prec, 1.0) / Float::with_val(prec, &root * &kap));
        let theta = Complex::with_val(
            prec,
            (
                Float::with_val(prec, &scale * ang.clone().cos()),
                Float::with_val(prec, &scale * ang.clone().sin()),
            ),
        );
        let xp: Vec<Complex> = x0.iter().map(|z| Complex::with_val(prec, z * &theta)).collect();
        let phi = phi_w(5, 3, 3, &kap, prec, len);
        (xp, phi, theta)
    }

    // The SOLVER (not verifier): recover Φ = P/Q from x(w), φ(w) with no knowledge of the
    // answer, then check it equals the paper's 648x⁵/(324x⁵+405x⁴−120x²+16).
    #[test]
    fn solve_map_recovers_paper_5_3_3() {
        let prec = 256u32;
        let (xp, phi, _theta) = setup_5_3_3(prec);
        let (p, q) = solve_belyi_map(&xp, &phi, 5, prec);
        // normalize by q₅ (leading denom coeff)
        let q5 = q[5].clone();
        let norm = |z: &Complex| -> num_complex::Complex64 {
            let r = Complex::with_val(prec, z / &q5);
            num_complex::Complex64::new(r.real().to_f64(), r.imag().to_f64())
        };
        // expected (÷324): p = [0,0,0,0,0, 648/324=2]; q = [16/324,0,−120/324,0,405/324,1]
        let exp_p = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let exp_q = [16.0 / 324.0, 0.0, -120.0 / 324.0, 0.0, 405.0 / 324.0, 1.0];
        for i in 0..=5 {
            let dp = (norm(&p[i]) - num_complex::Complex64::new(exp_p[i], 0.0)).norm();
            let dq = (norm(&q[i]) - num_complex::Complex64::new(exp_q[i], 0.0)).norm();
            assert!(dp < 1e-6, "P[{i}] = {} (want {})", norm(&p[i]), exp_p[i]);
            assert!(dq < 1e-6, "Q[{i}] = {} (want {})", norm(&q[i]), exp_q[i]);
        }
    }

    // §5b end-to-end: with κ (rug Γ) and Θ = (81/2)^{1/5}·e^{2πi/5}/κ, verify the paper's
    // Belyi map Φ(x) = 648x⁵/(324x⁵+405x⁴−120x²+16) satisfies Φ(Θ·x₀(w)) = φ(w), where
    // φ(w) is the exact hypergeometric Δ-uniformizer. Validates κ, φ(w), and the whole
    // §5 assembly against ground truth.
    #[test]
    fn belyi_map_matches_paper_5_3_3() {
        use rug::float::Constant;
        let prec = 256u32;
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        let ech = echelonize(forms_to_series(&forms, 6, prec), prec, 1e-25);
        let x0 = coordinate_x(&ech, 1, 1, prec, 1e-25);
        let len = x0.len();

        let kap = kappa(&tg);
        assert!((kap.to_f64() - 0.37630).abs() < 1e-4, "κ = {} (expected ≈0.3763)", kap.to_f64());

        // Θ = e^{2πi/5} / ((81/2)^{1/5} · κ)   [fixes 40.5·Θ⁵ = κ⁻⁵ at leading order]
        let alpha = Float::with_val(prec, Float::with_val(prec, 81.0) / 2.0);
        let root = (alpha.clone().ln() / Float::with_val(prec, 5.0)).exp();
        let pi = Float::with_val(prec, Constant::Pi);
        let ang = Float::with_val(prec, Float::with_val(prec, &pi * 2.0) / 5.0);
        let scale = Float::with_val(prec, Float::with_val(prec, 1.0) / Float::with_val(prec, &root * &kap));
        let theta = Complex::with_val(
            prec,
            (
                Float::with_val(prec, &scale * ang.clone().cos()),
                Float::with_val(prec, &scale * ang.clone().sin()),
            ),
        );
        // sanity: Θ ≈ 0.3917053 + 1.205545 i (eq 5.9)
        assert!((theta.real().to_f64() - 0.3917053).abs() < 1e-4, "Θ_re = {}", theta.real().to_f64());
        assert!((theta.imag().to_f64() - 1.205545).abs() < 1e-4, "Θ_im = {}", theta.imag().to_f64());

        // x_paper = Θ · x₀
        let xp: Vec<Complex> = x0.iter().map(|z| Complex::with_val(prec, z * &theta)).collect();
        // Φ(xp) = 648 xp⁵ / (324 xp⁵ + 405 xp⁴ − 120 xp² + 16)
        let x2 = series_mul(&xp, &xp, len, prec);
        let x4 = series_mul(&x2, &x2, len, prec);
        let x5 = series_mul(&x4, &xp, len, prec);
        let cc = |re: f64| Complex::with_val(prec, (re, 0.0));
        let mut num = vec![Complex::with_val(prec, (0.0, 0.0)); len];
        let mut den = vec![Complex::with_val(prec, (0.0, 0.0)); len];
        for j in 0..len {
            num[j] = Complex::with_val(prec, &x5[j] * cc(648.0));
            let t324 = Complex::with_val(prec, &x5[j] * cc(324.0));
            let t405 = Complex::with_val(prec, &x4[j] * cc(405.0));
            let t120 = Complex::with_val(prec, &x2[j] * cc(-120.0));
            den[j] = Complex::with_val(prec, Complex::with_val(prec, &t324 + &t405) + &t120);
        }
        den[0] = Complex::with_val(prec, &den[0] + cc(16.0));
        let phi_of_x = series_div(&num, &den, len, prec, 1e-25);

        let phi = phi_w(5, 3, 3, &kap, prec, len);
        // Compare in the accurate low-order range (higher coeffs degrade as the N=48
        // form error ~ρ^N is amplified by the series division and 5th powers).
        let mut worst = 0f64;
        let mut worst_n = 0;
        for n in 0..11 {
            let d = Complex::with_val(prec, &phi_of_x[n] - &phi[n]);
            let m = (d.real().to_f64().powi(2) + d.imag().to_f64().powi(2)).sqrt();
            if m > worst {
                worst = m;
                worst_n = n;
            }
        }
        // ~2e-8 at N=48 (truncation-limited — tightens with N); decisive vs the exact map.
        assert!(worst < 1e-7, "Φ(Θ·x₀(w)) ≠ φ(w): worst coeff diff {worst:.2e} at n={worst_n}");
    }
}

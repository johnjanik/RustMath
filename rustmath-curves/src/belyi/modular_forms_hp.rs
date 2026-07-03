//! KMSV §4 (Path A) at high precision (rug/MPFR): the modular-forms eigenvector
//! matrix assembled in `rug::Complex` so the null space resolves to the truncation
//! floor ρ^N (≈10⁻³⁰ at N≈110) instead of f64's ρ^{-N}-limited ~10⁻³.
//!
//! Geometry strategy: the *combinatorial* reduction (which generators to apply) is
//! decided in f64 (robust for generic sample points), and the resulting word is
//! *evaluated* in hp — so we never need an hp `atan2`/`floor` reduction, only hp
//! matrix arithmetic. Coset reps are rebuilt in hp from the words recorded by
//! [`CosetGraph::compactify`].

use super::coset_graph::{CosetGraph, Gen};
use super::mp_svd::{jacobi_svd, JacobiSvdOptions, MpC, MpMatrix};
use super::triangle_group::TriangleGroup;
use super::triangle_group_hp::{MobiusHp, TriangleGroupHp};
use num_complex::Complex64;
use rug::float::Constant;
use rug::{Complex, Float};

fn cmod_f64(z: &Complex) -> f64 {
    let re = z.real().to_f64();
    let im = z.imag().to_f64();
    (re * re + im * im).sqrt()
}

fn wp(z: &Complex, prec: u32) -> Complex {
    let ic = Complex::with_val(prec, (0.0, 1.0));
    let num = Complex::with_val(prec, z - &ic);
    let den = Complex::with_val(prec, z + &ic);
    Complex::with_val(prec, &num / &den)
}

fn wp_inv(w: &Complex, prec: u32) -> Complex {
    let ic = Complex::with_val(prec, (0.0, 1.0));
    let one = Complex::with_val(prec, (1.0, 0.0));
    let num = Complex::with_val(prec, &ic * Complex::with_val(prec, &one + w));
    let den = Complex::with_val(prec, &one - w);
    Complex::with_val(prec, &num / &den)
}

/// hp integer power of a complex number (n ≥ 0).
fn cpow_u(z: &Complex, n: u32, prec: u32) -> Complex {
    let mut r = Complex::with_val(prec, (1.0, 0.0));
    for _ in 0..n {
        r = Complex::with_val(prec, &r * z);
    }
    r
}

/// Build an hp Möbius from a generator word (right-multiplied in word order, matching
/// how compactify accumulates reps).
fn word_to_hp(word: &[Gen], da: &MobiusHp, dai: &MobiusHp, db: &MobiusHp, dbi: &MobiusHp) -> MobiusHp {
    let mut m = MobiusHp::identity(da.prec);
    for g in word {
        let gm = match g {
            Gen::A => da,
            Gen::AInv => dai,
            Gen::B => db,
            Gen::BInv => dbi,
        };
        m = m.mul(gm);
    }
    m
}

/// δ from a reduction word `ops` (left-multiplied, matching `reduce_to_base`).
fn delta_from_ops(ops: &[(bool, i32)], tg: &TriangleGroupHp) -> MobiusHp {
    let mut delta = MobiusHp::identity(tg.prec);
    for &(is_a, pw) in ops {
        let g = if is_a {
            tg.delta_a.pow_signed(pw)
        } else {
            tg.delta_b.pow_signed(pw)
        };
        delta = g.mul(&delta);
    }
    delta
}

/// hp coset representatives α_i, rebuilt from the words recorded by `compactify`.
fn reps_hp(cg: &CosetGraph, tg: &TriangleGroupHp) -> Vec<MobiusHp> {
    let dai = tg.delta_a.inverse();
    let dbi = tg.delta_b.inverse();
    cg.rep_words
        .iter()
        .map(|w| word_to_hp(w, &tg.delta_a, &dai, &tg.delta_b, &dbi))
        .collect()
}

/// hp radius ρ of a circle (in the w_{z_a} chart) containing the compact domain D_Γ.
pub fn domain_radius_hp(cg: &CosetGraph, tg: &TriangleGroupHp) -> Float {
    let prec = tg.prec;
    let reps = reps_hp(cg, tg);
    let ncz = Complex::with_val(prec, (Float::with_val(prec, -tg.z_c.real()), tg.z_c.imag().clone()));
    let verts = [tg.z_a.clone(), tg.z_b.clone(), tg.z_c.clone(), ncz];
    let mut rho = Float::with_val(prec, 0.0);
    for rep in &reps {
        for v in &verts {
            let r = wp(&rep.apply(v), prec);
            let m = Float::with_val(prec, cmod_f64(&r));
            if m > rho {
                rho = m;
            }
        }
    }
    // recompute the max modulus at full precision for the winning value
    let mut rho_hp = Float::with_val(prec, 0.0);
    for rep in &reps {
        for v in &verts {
            let r = wp(&rep.apply(v), prec);
            let re = r.real();
            let im = r.imag();
            let m = Float::with_val(prec, Float::with_val(prec, re * re) + Float::with_val(prec, im * im)).sqrt();
            if m > rho_hp {
                rho_hp = m;
            }
        }
    }
    rho_hp
}

/// Assemble the preconditioned matrix M = ρ^{n−r}·(A−I) (eq 4.6) in hp. Its null space
/// is S_k(Γ) (the ρ^{n−r} scaling fixes the −I diagonal and conditions the A part).
/// Also returns ρ (for un-scaling recovered vectors: b_n = ρ^{-n} y_n). `cg` must be
/// compactified.
pub fn assemble_scaled_ami(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    rho_scale: f64,
) -> (Vec<Vec<Complex>>, Float) {
    let prec = tg.prec;
    let dim = big_n + 1;
    let ku = k as u32;
    let rho = Float::with_val(prec, domain_radius_hp(cg, tg) * rho_scale);
    let reps = reps_hp(cg, tg);

    let pi = Float::with_val(prec, Constant::Pi);
    let two_pi = Float::with_val(prec, 2.0 * &pi);
    let one = Complex::with_val(prec, (1.0, 0.0));

    // Sum over the Q sample points, parallelised (each sample is an independent outer-product
    // contribution a[n][r] += base · w_m^{-n} · w'_m^r; O(Q·dim²) total). rayon fold/reduce.
    use rayon::prelude::*;
    let zero_mat = || vec![vec![Complex::with_val(prec, (0.0, 0.0)); dim]; dim];
    let mut a = (1..=q)
        .into_par_iter()
        .fold(&zero_mat, |mut acc, m| {
            // w_m = ρ · exp(2πi m/Q)
            let theta = Float::with_val(prec, &two_pi * (m as f64)) / (q as f64);
            let ct = theta.clone().cos();
            let st = theta.clone().sin();
            let wm = Complex::with_val(prec, (Float::with_val(prec, &rho * &ct), Float::with_val(prec, &rho * &st)));
            let zm = wp_inv(&wm, prec);
            let zm64 = Complex64::new(zm.real().to_f64(), zm.imag().to_f64());
            // combinatorial reduction word (f64), evaluated in hp
            let (_, ops) = tg64.reduce_to_base(zm64);
            let i = cg.coset_from_ops(&ops);
            let delta = delta_from_ops(&ops, tg);
            let gamma = reps[i].mul(&delta);
            let zpm = gamma.apply(&zm);
            let wpm = wp(&zpm, prec);
            // j(γ,z) = c z + d
            let jm = Complex::with_val(prec, &zm * &gamma.c) + Complex::with_val(prec, (&gamma.d, 0.0));
            let jm_neg_k = cpow_u(&Complex::with_val(prec, &one / &jm), ku, prec);
            let omw_pm_k = cpow_u(&Complex::with_val(prec, &one - &wpm), ku, prec);
            let omw_m_k = cpow_u(&Complex::with_val(prec, &one - &wm), ku, prec);
            let base = Complex::with_val(prec, &jm_neg_k * &omw_pm_k) / Complex::with_val(prec, omw_m_k);
            // (w'_m)^r and w_m^{-n}
            let mut wpm_r = vec![Complex::with_val(prec, (1.0, 0.0)); dim];
            for r in 1..dim {
                wpm_r[r] = Complex::with_val(prec, &wpm_r[r - 1] * &wpm);
            }
            let inv_wm = Complex::with_val(prec, &one / &wm);
            let mut wm_neg_n = vec![Complex::with_val(prec, (1.0, 0.0)); dim];
            for n in 1..dim {
                wm_neg_n[n] = Complex::with_val(prec, &wm_neg_n[n - 1] * &inv_wm);
            }
            for n in 0..dim {
                let cn = Complex::with_val(prec, &base * &wm_neg_n[n]);
                for r in 0..dim {
                    let term = Complex::with_val(prec, &cn * &wpm_r[r]);
                    acc[n][r] = Complex::with_val(prec, &acc[n][r] + &term);
                }
            }
            acc
        })
        .reduce(&zero_mat, |mut a1, a2| {
            for n in 0..dim {
                for r in 0..dim {
                    a1[n][r] = Complex::with_val(prec, &a1[n][r] + &a2[n][r]);
                }
            }
            a1
        });
    // divide by Q, precondition a[n][r] ·= ρ^{n−r}, subtract I
    let mut rho_pow = vec![Float::with_val(prec, 1.0); dim];
    for n in 1..dim {
        rho_pow[n] = Float::with_val(prec, &rho_pow[n - 1] * &rho);
    }
    let qf = Float::with_val(prec, q as f64);
    for n in 0..dim {
        for r in 0..dim {
            let scale = Float::with_val(prec, &rho_pow[n] / &rho_pow[r]) / &qf;
            a[n][r] = Complex::with_val(prec, &a[n][r] * &scale);
        }
        a[n][n] = Complex::with_val(prec, &a[n][n] - &one);
    }
    (a, rho)
}

/// dim S_k(Γ) via Gauss–Jordan pivots (unreliable for small σ — see mp_svd; kept for
/// the k=2 sanity check where full rank is unambiguous).
pub fn nullity_s_k(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    tol: f64,
    rho_scale: f64,
) -> (usize, Vec<f64>) {
    let (a, _rho) = assemble_scaled_ami(tg64, tg, cg, k, big_n, q, rho_scale);
    null_space_hp(a, tol, tg.prec)
}

/// dim S_k(Γ) via the hp complex SVD of the preconditioned A−I: the number of singular
/// values ≤ `threshold`. Returns (nullity, all singular values descending). This is the
/// reliable rank test (LU pivots floor at ~10⁻⁴; the true small σ ~10⁻²⁹).
pub fn dim_s_k_svd(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    threshold_decimal: &str,
    tol_decimal: &str,
    rho_scale: f64,
) -> (usize, Vec<Float>) {
    let prec = tg.prec;
    let (a, _rho) = assemble_scaled_ami(tg64, tg, cg, k, big_n, q, rho_scale);
    let dim = a.len();
    // to row-major MpMatrix
    let mut data = Vec::with_capacity(dim * dim);
    for row in &a {
        for z in row {
            data.push(MpC::new(Float::with_val(prec, z.real()), Float::with_val(prec, z.imag())));
        }
    }
    let mat = MpMatrix::from_row_major(dim, dim, prec, data).expect("square matrix");
    let opt = JacobiSvdOptions::new(prec, 80, tol_decimal, "1e-40");
    let svd = jacobi_svd(&mat, &opt).expect("svd");
    let threshold = Float::with_val(prec, Float::parse(threshold_decimal).expect("threshold"));
    let nullity = svd.numerical_nullity_indices(&threshold).len();
    (nullity, svd.sigma)
}

/// Recover a basis of S_k(Γ) as coefficient vectors b (with f = (1−w)^k Σ b_n w^n)
/// from the hp SVD of the preconditioned A−I: take the small-σ right singular vectors
/// y and un-scale b_n = ρ^{-n} y_n. Returns dim S_k(Γ) vectors, each length N+1 — the
/// modular forms themselves, which §5 echelonizes into the coordinate x(w).
pub fn recover_forms(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    threshold_decimal: &str,
    tol_decimal: &str,
    rho_scale: f64,
) -> Vec<Vec<Complex>> {
    let prec = tg.prec;
    let (a, rho) = assemble_scaled_ami(tg64, tg, cg, k, big_n, q, rho_scale);
    let dim = a.len();
    let mut data = Vec::with_capacity(dim * dim);
    for row in &a {
        for z in row {
            data.push(MpC::new(Float::with_val(prec, z.real()), Float::with_val(prec, z.imag())));
        }
    }
    let mat = MpMatrix::from_row_major(dim, dim, prec, data).expect("square matrix");
    let opt = JacobiSvdOptions::new(prec, 80, tol_decimal, "1e-40");
    let svd = jacobi_svd(&mat, &opt).expect("svd");
    let threshold = Float::with_val(prec, Float::parse(threshold_decimal).expect("threshold"));
    let ker = svd.right_nullspace_basis(&threshold);

    let mut rho_pow = vec![Float::with_val(prec, 1.0); dim];
    for n in 1..dim {
        rho_pow[n] = Float::with_val(prec, &rho_pow[n - 1] * &rho);
    }
    let mut forms = Vec::with_capacity(ker.cols);
    for f in 0..ker.cols {
        let mut b = Vec::with_capacity(dim);
        for n in 0..dim {
            let y = ker.get(n, f).div_real(&rho_pow[n]); // b_n = ρ^{-n} y_n
            b.push(Complex::with_val(prec, (y.re, y.im)));
        }
        forms.push(b);
    }
    forms
}

/// Nullity + sorted pivot magnitudes of a square hp complex matrix (Gauss–Jordan,
/// partial pivoting).
fn null_space_hp(mut m: Vec<Vec<Complex>>, tol: f64, prec: u32) -> (usize, Vec<f64>) {
    let rows = m.len();
    let cols = m[0].len();
    let mut pivots: Vec<f64> = Vec::new();
    let mut rank = 0usize;
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let mut best = r;
        let mut best_val = cmod_f64(&m[r][c]);
        for i in (r + 1)..rows {
            let v = cmod_f64(&m[i][c]);
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        if best_val < tol {
            continue;
        }
        m.swap(r, best);
        pivots.push(best_val);
        rank += 1;
        let piv = m[r][c].clone();
        for j in c..cols {
            m[r][j] = Complex::with_val(prec, &m[r][j] / &piv);
        }
        for i in 0..rows {
            if i != r && cmod_f64(&m[i][c]) > 0.0 {
                let f = m[i][c].clone();
                for j in c..cols {
                    let t = Complex::with_val(prec, &f * &m[r][j]);
                    m[i][j] = Complex::with_val(prec, &m[i][j] - &t);
                }
            }
        }
        r += 1;
    }
    pivots.sort_by(|a, b| b.partial_cmp(a).unwrap());
    (cols - rank, pivots)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREC: u32 = 256;

    fn setup_5_3_3() -> (TriangleGroup, TriangleGroupHp, CosetGraph) {
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, PREC);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        (tg64, tg, cg)
    }

    // Dump the [2,12,5] preconditioned M = ρ^{n−r}(A−I) to a raw-f64 file for an external
    // FP64 SVD (numpy / cuSOLVER). Env: M_N (=big_n), M_K (=weight), M_OUT (=path).
    // Format: u32 dim, then dim*dim*2 f64 (row-major, re/im interleaved, little-endian).
    #[test]
    #[ignore]
    fn dump_2_12_5_matrix() {
        use std::io::Write;
        let n: usize = std::env::var("M_N").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
        let k: i64 = std::env::var("M_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let out = std::env::var("M_OUT").unwrap_or_else(|_| "/tmp/m_2_12_5.bin".into());
        let prec: u32 = std::env::var("M_PREC").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
        let s0: Vec<usize> = vec![0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6];
        let s1: Vec<usize> = vec![14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6];
        let tg64 = TriangleGroup::new(2, 12, 5);
        let tg = TriangleGroupHp::new(2, 12, 5, prec);
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify_with(&tg64, 0.996, 40);
        let q = 2 * n + 8;
        let (a, rho) = assemble_scaled_ami(&tg64, &tg, &cg, k, n, q, 1.0);
        let dim = a.len();
        eprintln!("[2,12,5] k={k} N={n} dim={dim} ρ={:.6} → {out}", rho.to_f64());
        let mut buf: Vec<u8> = Vec::with_capacity(4 + dim * dim * 16);
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
        for row in &a {
            for z in row {
                buf.extend_from_slice(&z.real().to_f64().to_le_bytes());
                buf.extend_from_slice(&z.imag().to_f64().to_le_bytes());
            }
        }
        let mut f = std::fs::File::create(&out).expect("create");
        f.write_all(&buf).expect("write");
    }

    // Dump the [2,12,5] preconditioned M in EXTENDED precision (double-double / triple-double)
    // for a GPU Ozaki + Ogita–Aishima refined SVD. Past the FP64 wall (N>~1953) the f64 dump's
    // own rounding (ρ^{−N}·1e-16) exceeds the truncation floor, so each rug entry is split into
    // `M_LIMBS` non-overlapping f64 limbs (2=dd ~1e-32, 3=td ~1e-48) via recursive round-to-nearest.
    // Format: u32 dim, u8 nlimbs, then row-major entries; each entry = re limbs then im limbs,
    //   nlimbs f64 each (little-endian). File size = 5 + dim*dim*2*nlimbs*8.
    // Assemble at M_PREC ≥ 106 (dd) / ≥ 159 (td) bits so the limbs carry real information.
    #[test]
    #[ignore]
    fn dump_2_12_5_matrix_ext() {
        use std::io::Write;
        let n: usize = std::env::var("M_N").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
        let k: i64 = std::env::var("M_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let out = std::env::var("M_OUT").unwrap_or_else(|_| "/tmp/m_2_12_5_ext.bin".into());
        let prec: u32 = std::env::var("M_PREC").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
        let nlimbs: usize = std::env::var("M_LIMBS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let s0: Vec<usize> = vec![0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6];
        let s1: Vec<usize> = vec![14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6];
        let tg64 = TriangleGroup::new(2, 12, 5);
        let tg = TriangleGroupHp::new(2, 12, 5, prec);
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify_with(&tg64, 0.996, 40);
        let q = 2 * n + 8;
        let (a, rho) = assemble_scaled_ami(&tg64, &tg, &cg, k, n, q, 1.0);
        let dim = a.len();
        eprintln!("[2,12,5] EXT k={k} N={n} dim={dim} limbs={nlimbs} prec={prec} ρ={:.6} → {out}", rho.to_f64());
        // recursively split a full-precision Float into `nlimbs` non-overlapping f64 limbs
        let split = |x: &Float| -> Vec<f64> {
            let mut limbs = Vec::with_capacity(nlimbs);
            let mut rem = Float::with_val(prec, x);
            for _ in 0..nlimbs {
                let hi = rem.to_f64();
                limbs.push(hi);
                rem = Float::with_val(prec, &rem - hi);
            }
            limbs
        };
        let mut buf: Vec<u8> = Vec::with_capacity(5 + dim * dim * 2 * nlimbs * 8);
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
        buf.push(nlimbs as u8);
        for row in &a {
            for z in row {
                for l in split(z.real()) { buf.extend_from_slice(&l.to_le_bytes()); }
                for l in split(z.imag()) { buf.extend_from_slice(&l.to_le_bytes()); }
            }
        }
        let mut f = std::fs::File::create(&out).expect("create");
        f.write_all(&buf).expect("write");
    }

    // The hp coset reps rebuilt from words match the f64 reps (validates word tracking).
    #[test]
    fn hp_reps_match_f64() {
        let (_, tg, cg) = setup_5_3_3();
        let reps = reps_hp(&cg, &tg);
        for (i, rep) in reps.iter().enumerate() {
            let f = &cg.reps[i];
            assert!(
                (rep.a.to_f64() - f.a).abs() < 1e-10
                    && (rep.b.to_f64() - f.b).abs() < 1e-10
                    && (rep.c.to_f64() - f.c).abs() < 1e-10
                    && (rep.d.to_f64() - f.d).abs() < 1e-10,
                "hp rep {i} disagrees with f64"
            );
        }
    }

    // The hp matrix assembly is correct where LU-pivot rank detection is reliable:
    // dim S_2(Γ) = 0, i.e. A−I is full rank with all pivots O(1) (well clear of any
    // threshold). This exercises the whole hp path — embedding, word-reconstructed
    // reps, hp reduction-by-word, hp assembly + preconditioning.
    #[test]
    fn k2_full_rank_hp() {
        let (tg64, tg, cg) = setup_5_3_3();
        let (nullity, pivots) = nullity_s_k(&tg64, &tg, &cg, 2, 48, 96, 1e-8, 1.0);
        assert_eq!(nullity, 0, "dim S_2 should be 0");
        assert!(*pivots.last().unwrap() > 1e-1, "all pivots should be O(1) for k=2");
    }

    // The payoff: with the hp complex SVD, the EXACT dim S_k(Γ) = 0, 1, 3 for
    // k = 2, 4, 6 resolves cleanly — the small singular values collapse to ~ρ^N,
    // far below the O(1) rank ones (LU pivots could not do this).
    #[test]
    fn dim_s_k_svd_5_3_3() {
        let (tg64, tg, cg) = setup_5_3_3();
        let (n, q) = (48usize, 96usize);
        // ρ^48 ≈ 5·10⁻¹⁴; threshold 10⁻⁸ separates the null σ from the O(1) rank σ.
        for (k, expected) in [(2i64, 0usize), (4, 1), (6, 3)] {
            let (nullity, sigma) = dim_s_k_svd(&tg64, &tg, &cg, k, n, q, "1e-8", "1e-70", 1.0);
            let smallest = sigma.last().unwrap().to_f64();
            let largest = sigma.first().unwrap().to_f64();
            assert_eq!(
                nullity, expected,
                "dim S_{k} = {nullity} (expected {expected}); σ_min={smallest:.2e}, σ_max={largest:.2e}"
            );
        }
    }

    // Recover the actual weight-6 forms and check one is genuinely Γ-modular: pick z₁
    // well inside the domain and a side-pairing γ ∈ Γ, then verify the reconstructed
    // f(z) = (1−w)^6 Σ b_n w^n satisfies f(γz₁) = j(γ,z₁)^6 f(z₁). (This is INDEPENDENT
    // of the assembly — it directly tests modularity of the recovered coefficients.)
    #[test]
    fn recovered_form_is_modular() {
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, PREC);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        // side pairings (valid Γ elements) captured before compactify
        let gammas: Vec<_> = cg.side_pairings.iter().map(|s| s.gamma).collect();
        cg.compactify(&tg64);

        let forms = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        assert_eq!(forms.len(), 3, "dim S_6 = 3");
        let b: Vec<Complex64> = forms[0]
            .iter()
            .map(|z| Complex64::new(z.real().to_f64(), z.imag().to_f64()))
            .collect();
        let i_c = Complex64::new(0.0, 1.0);
        let eval = |z: Complex64| -> Complex64 {
            let w = (z - i_c) / (z + i_c);
            let mut s = Complex64::new(0.0, 0.0);
            let mut wn = Complex64::new(1.0, 0.0);
            for bn in &b {
                s += bn * wn;
                wn *= w;
            }
            (Complex64::new(1.0, 0.0) - w).powu(6) * s
        };
        // z₁ near the centre z_a = i (small |w|), so both f(z₁) and f(γz₁) are accurate.
        let z1 = Complex64::new(0.05, 1.1);
        let fz1 = eval(z1);
        // best-behaved side pairing (γz₁ closest to the domain) should satisfy automorphy.
        let mut best = f64::INFINITY;
        for g in &gammas {
            let gz = g.apply(z1);
            let j = Complex64::new(g.c, 0.0) * z1 + Complex64::new(g.d, 0.0);
            let resid = (eval(gz) - j.powu(6) * fz1).norm() / fz1.norm();
            best = best.min(resid);
        }
        // ~3e-5 here (limited by series truncation at |w(γz₁)|~0.8, i.e. 0.8^48), versus
        // ~10³ for an LU-noise vector — unambiguous evidence the form is genuinely modular.
        assert!(best < 1e-4, "recovered form not modular: best automorphy residual {best:.2e}");
    }
}

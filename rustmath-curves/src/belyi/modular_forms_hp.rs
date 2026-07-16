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

/// conjugate of a complex (negate imaginary part).
fn cconj(z: &Complex, prec: u32) -> Complex {
    Complex::with_val(prec, (z.real().clone(), Float::with_val(prec, -z.imag())))
}

/// Disk chart centered at an arbitrary upper-half-plane point `ctr`:
///   w = (z − ctr)/(z − conj ctr).
/// ctr = z_a = i recovers `wp` exactly; ctr = z_b = μi is the order-12 chart, ctr = z_c the
/// order-5 chart.  dz/dw = (ctr − conj ctr)/(1−w)² for any center, so the weight-k automorphy
/// factor stays (1−w)^k — only the (constant) 2i·Im(ctr) scale differs, and it cancels in the ratio.
fn wp_c(z: &Complex, ctr: &Complex, prec: u32) -> Complex {
    let ctr_bar = cconj(ctr, prec);
    let num = Complex::with_val(prec, z - ctr);
    let den = Complex::with_val(prec, z - &ctr_bar);
    Complex::with_val(prec, &num / &den)
}

/// inverse of `wp_c`:  z = (ctr − w·conj ctr)/(1 − w).
fn wp_c_inv(w: &Complex, ctr: &Complex, prec: u32) -> Complex {
    let ctr_bar = cconj(ctr, prec);
    let one = Complex::with_val(prec, (1.0, 0.0));
    let num = Complex::with_val(prec, ctr - Complex::with_val(prec, w * &ctr_bar));
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
    domain_radius_hp_centered(cg, tg, &tg.z_a)
}

/// hp radius ρ of a circle (in the w_{ctr} chart) containing the compact domain D_Γ.
pub fn domain_radius_hp_centered(cg: &CosetGraph, tg: &TriangleGroupHp, ctr: &Complex) -> Float {
    let prec = tg.prec;
    let reps = reps_hp(cg, tg);
    let ncz = Complex::with_val(prec, (Float::with_val(prec, -tg.z_c.real()), tg.z_c.imag().clone()));
    let verts = [tg.z_a.clone(), tg.z_b.clone(), tg.z_c.clone(), ncz];
    let mut rho_hp = Float::with_val(prec, 0.0);
    for rep in &reps {
        for v in &verts {
            let r = wp_c(&rep.apply(v), ctr, prec);
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
    ctr: &Complex,
) -> (Vec<Vec<Complex>>, Float) {
    let prec = tg.prec;
    let dim = big_n + 1;
    let ku = k as u32;
    let rho = Float::with_val(prec, domain_radius_hp_centered(cg, tg, ctr) * rho_scale);
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
            let zm = wp_c_inv(&wm, ctr, prec);
            let zm64 = Complex64::new(zm.real().to_f64(), zm.imag().to_f64());
            // combinatorial reduction word (f64), evaluated in hp
            let (_, ops) = tg64.reduce_to_base(zm64);
            let i = cg.coset_from_ops(&ops);
            let delta = delta_from_ops(&ops, tg);
            let gamma = reps[i].mul(&delta);
            let zpm = gamma.apply(&zm);
            let wpm = wp_c(&zpm, ctr, prec);
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

/// Streamed, constant-memory assembly of the preconditioned M = ρ^{n−r}(A−I),
/// written directly as the EXT limb dump (u32 dim, u8 nlimbs, row-major dd/td limbs)
/// with per-row crash checkpointing.
///
/// Uses the circle structure of the sample points: w_m = ρ e^{2πim/Q} gives
/// w_m^{−n} = ρ^{−n} e^{−2πimn/Q}, and the ρ^{n−r} preconditioner cancels the ρ^{−n},
/// leaving
///   M[n][r] = (1/Q) Σ_m base_m · e^{−2πimn/Q} · (w′_m/ρ)^r  −  δ_{nr}.
/// Each row needs only the Q per-sample scalars (base_m, u_m = w′_m/ρ) and the Q-periodic
/// root-of-unity table E[j] = e^{−2πij/Q} (phase index (m·n) mod Q is exact integer
/// arithmetic — no error growth across rows). Worker memory is O(Q + dim) rug values,
/// unlike `assemble_scaled_ami` whose rayon fold clones a dim² accumulator per split
/// (~1.6 GB each at dim=3001 — the OOM at N=3000).
///
/// Completed rows are recorded in `{out}.progress` (params line, then one row index per
/// line); rerunning with the same params skips them. Returns ρ.
pub fn dump_scaled_ami_streamed(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    rho_scale: f64,
    ctr: &Complex,
    nlimbs: usize,
    out: &str,
) -> Float {
    use rayon::prelude::*;
    use std::io::Write;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let prec = tg.prec;
    let dim = big_n + 1;
    let ku = k as u32;
    let rho = Float::with_val(prec, domain_radius_hp_centered(cg, tg, ctr) * rho_scale);
    let reps = reps_hp(cg, tg);
    let one = Complex::with_val(prec, (1.0, 0.0));
    let pi = Float::with_val(prec, Constant::Pi);
    let two_pi = Float::with_val(prec, 2.0 * &pi);
    let qf = Float::with_val(prec, q as f64);

    // root-of-unity table E[j] = e^{−2πij/Q}
    let unit: Vec<Complex> = (0..q)
        .into_par_iter()
        .map(|j| {
            let theta = Float::with_val(prec, &two_pi * (j as f64)) / (q as f64);
            let (s, c) = theta.sin_cos(Float::new(prec));
            Complex::with_val(prec, (c, Float::with_val(prec, -s)))
        })
        .collect();

    // per-sample scalars: c0_m = base_m/Q and u_m = w′_m/ρ, for m = 1..=Q
    let samples: Vec<(Complex, Complex)> = (1..=q)
        .into_par_iter()
        .map(|m| {
            let theta = Float::with_val(prec, &two_pi * (m as f64)) / (q as f64);
            let (s, c) = theta.sin_cos(Float::new(prec));
            let wm = Complex::with_val(prec, (Float::with_val(prec, &rho * &c), Float::with_val(prec, &rho * &s)));
            let zm = wp_c_inv(&wm, ctr, prec);
            let zm64 = Complex64::new(zm.real().to_f64(), zm.imag().to_f64());
            let (_, ops) = tg64.reduce_to_base(zm64);
            let i = cg.coset_from_ops(&ops);
            let delta = delta_from_ops(&ops, tg);
            let gamma = reps[i].mul(&delta);
            let zpm = gamma.apply(&zm);
            let wpm = wp_c(&zpm, ctr, prec);
            let jm = Complex::with_val(prec, &zm * &gamma.c) + Complex::with_val(prec, (&gamma.d, 0.0));
            let jm_neg_k = cpow_u(&Complex::with_val(prec, &one / &jm), ku, prec);
            let omw_pm_k = cpow_u(&Complex::with_val(prec, &one - &wpm), ku, prec);
            let omw_m_k = cpow_u(&Complex::with_val(prec, &one - &wm), ku, prec);
            let base = Complex::with_val(prec, &jm_neg_k * &omw_pm_k) / Complex::with_val(prec, omw_m_k);
            let c0 = Complex::with_val(prec, &base / &qf);
            let u = Complex::with_val(prec, &wpm / &rho);
            (c0, u)
        })
        .collect();

    // output file: header immediately, rows pwritten at exact offsets as they finish
    let row_bytes = dim * 2 * nlimbs * 8;
    let total = 5 + dim * row_bytes;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(out)
        .expect("open output");
    file.set_len(total as u64).expect("set_len");
    let mut header = Vec::with_capacity(5);
    header.extend_from_slice(&(dim as u32).to_le_bytes());
    header.push(nlimbs as u8);
    file.write_at(&header, 0).expect("write header");

    // progress sidecar: params line, then one completed row index per line
    let progress_path = format!("{out}.progress");
    let params_line = format!("dim={dim} nlimbs={nlimbs} prec={prec} q={q} k={k}");
    let mut done = vec![false; dim];
    let mut n_done = 0usize;
    if let Ok(text) = std::fs::read_to_string(&progress_path) {
        let mut lines = text.lines();
        match lines.next() {
            Some(first) if first == params_line => {
                for l in lines {
                    if let Ok(n) = l.trim().parse::<usize>() {
                        if n < dim && !done[n] {
                            done[n] = true;
                            n_done += 1;
                        }
                    }
                }
                eprintln!("[streamed] resume: {n_done}/{dim} rows already done");
            }
            Some(first) => panic!(
                "progress file {progress_path} is for different params\n  have: {first}\n  want: {params_line}\nremove it (and the output) to start over"
            ),
            None => {}
        }
    }
    let mut progress = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&progress_path)
        .expect("open progress");
    if n_done == 0 {
        writeln!(progress, "{params_line}").expect("write params");
        progress.flush().expect("flush progress");
    }
    let progress = Mutex::new(progress);
    let counter = AtomicUsize::new(n_done);

    let todo: Vec<usize> = (0..dim).filter(|&n| !done[n]).collect();
    todo.into_par_iter().for_each(|n| {
        // c_m = c0_m · e^{−2πimn/Q}; acc[r] = Σ_m c_m u_m^r via a running product
        let mut v: Vec<Complex> = samples
            .iter()
            .enumerate()
            .map(|(idx, (c0, _))| {
                let m = idx + 1;
                Complex::with_val(prec, c0 * &unit[(m * n) % q])
            })
            .collect();
        let mut row: Vec<u8> = Vec::with_capacity(row_bytes);
        let mut split_into = |x: &Float, buf: &mut Vec<u8>| {
            let mut rem = Float::with_val(prec, x);
            for _ in 0..nlimbs {
                let hi = rem.to_f64();
                buf.extend_from_slice(&hi.to_le_bytes());
                rem = Float::with_val(prec, &rem - hi);
            }
        };
        for r in 0..dim {
            if r > 0 {
                for (vm, (_, u)) in v.iter_mut().zip(samples.iter()) {
                    *vm *= u;
                }
            }
            let mut acc = Complex::with_val(prec, (0.0, 0.0));
            for vm in &v {
                acc += vm;
            }
            if r == n {
                acc -= &one;
            }
            split_into(acc.real(), &mut row);
            split_into(acc.imag(), &mut row);
        }
        file.write_at(&row, (5 + n * row_bytes) as u64).expect("write row");
        {
            let mut p = progress.lock().unwrap();
            writeln!(p, "{n}").expect("write progress");
            p.flush().expect("flush progress");
        }
        let c = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if c % 100 == 0 || c == dim {
            eprintln!("[streamed] {c}/{dim} rows");
        }
    });
    file.sync_data().expect("sync");
    rho
}

/// Explicit parameters for one atlas chart assembly (probe or full dump).
///
/// This is the env-var surface the `dump_2_12_5_matrix_ext_streamed` harness has
/// always driven, lifted into a struct so the `belyi-atlas` binary and the test
/// call the SAME setup + assembly path.  Any matrix produced through
/// [`run_atlas_dump`] is bit-identical to the trusted streamed harness for the
/// same resolved parameters.
pub struct AtlasDumpParams {
    /// 0-based image list of s0 (order a).
    pub s0: Vec<usize>,
    /// 0-based image list of s1 (order b).
    pub s1: Vec<usize>,
    /// Triangle-group orders (a, b, c).
    pub abc: (u32, u32, u32),
    /// Basepoint coset: rebase by the transposition (0 base); 0 = no rebase.
    pub base: usize,
    /// Series length N (dim = N + 1).
    pub n: usize,
    /// Weight k.
    pub k: i64,
    /// mpfr precision (bits).
    pub prec: u32,
    /// Extended double-double limbs.
    pub nlimbs: usize,
    /// Compactify BFS prune radius.
    pub r_prune: f64,
    /// Compactify word cap.
    pub l_max: usize,
    /// Expansion vertex: "a"/"b"/"c" (base vertices) or "a2"/"b2"/"c2" (coset rep
    /// applied to the vertex, selecting a satellite preimage).
    pub center: String,
    /// Coset index for the "*2" satellite centers.
    pub coset: usize,
    /// Output matrix path.
    pub out: String,
}

/// Result of an atlas assembly: convergence radius, chart center, matrix dim.
pub struct AtlasDumpResult {
    pub rho: Float,
    pub ctr: Complex,
    pub dim: usize,
}

/// Build the triangle group, compactified coset graph, and chart center for `p`
/// exactly as the streamed harness does (the `(0 base)` rebase, `compactify_with`,
/// and the center dispatch, verbatim). Extracted from [`run_atlas_dump`] so the
/// solve driver ([`super::genus0_map::run_and_persist_belyi`]) shares the setup;
/// `p.n`, `p.nlimbs`, `p.out` are not consumed here.
pub fn atlas_setup(p: &AtlasDumpParams) -> (TriangleGroup, TriangleGroupHp, CosetGraph, Complex) {
    let mut s0 = p.s0.clone();
    let mut s1 = p.s1.clone();
    assert_eq!(s0.len(), s1.len(), "s0, s1 degrees differ");
    let (oa, ob, oc) = p.abc;

    // Rebase by (0 base): re-mark coset `base` as the basepoint.
    if p.base != 0 {
        let base = p.base;
        let q = |x: usize| if x == 0 { base } else if x == base { 0 } else { x };
        let (o0, o1) = (s0.clone(), s1.clone());
        for x in 0..o0.len() {
            s0[q(x)] = q(o0[x]);
            s1[q(x)] = q(o1[x]);
        }
    }

    let tg64 = TriangleGroup::new(oa, ob, oc);
    let tg = TriangleGroupHp::new(oa, ob, oc, p.prec);
    let mut cg = CosetGraph::build(&tg64, &s0, &s1);
    cg.compactify_with(&tg64, p.r_prune, p.l_max);

    let ctr = match p.center.as_str() {
        "b" => tg.z_b.clone(),
        "c" => tg.z_c.clone(),
        "b2" => reps_hp(&cg, &tg)[p.coset].apply(&tg.z_b),
        "c2" => reps_hp(&cg, &tg)[p.coset].apply(&tg.z_c),
        "a2" => reps_hp(&cg, &tg)[p.coset].apply(&tg.z_a),
        _ => tg.z_a.clone(),
    };
    (tg64, tg, cg, ctr)
}

/// Set up the triangle group + coset graph exactly as the streamed harness does,
/// pick the requested chart center, and run [`dump_scaled_ami_streamed`].
///
/// The `q = 2 N + 8` over-sampling, the `(0 base)` rebase, and the center
/// dispatch all match the harness verbatim, so callers get identical numerics.
pub fn run_atlas_dump(p: &AtlasDumpParams) -> AtlasDumpResult {
    let (tg64, tg, cg, ctr) = atlas_setup(p);
    let q = 2 * p.n + 8;
    let rho = dump_scaled_ami_streamed(&tg64, &tg, &cg, p.k, p.n, q, 1.0, &ctr, p.nlimbs, &p.out);
    AtlasDumpResult { rho, ctr, dim: p.n + 1 }
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
    let (a, _rho) = assemble_scaled_ami(tg64, tg, cg, k, big_n, q, rho_scale, &tg.z_a);
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
    dim_s_k_svd_centered(tg64, tg, cg, k, big_n, q, threshold_decimal, tol_decimal, rho_scale, &tg.z_a)
}

/// [`dim_s_k_svd`] in the disk chart centered at an arbitrary `ctr` (dim S_k is
/// chart-independent; ρ — and hence the N needed for a given accuracy — is not).
#[allow(clippy::too_many_arguments)]
pub fn dim_s_k_svd_centered(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    threshold_decimal: &str,
    tol_decimal: &str,
    rho_scale: f64,
    ctr: &Complex,
) -> (usize, Vec<Float>) {
    let prec = tg.prec;
    let (a, _rho) = assemble_scaled_ami(tg64, tg, cg, k, big_n, q, rho_scale, ctr);
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
    recover_forms_centered(tg64, tg, cg, k, big_n, q, threshold_decimal, tol_decimal, rho_scale, &tg.z_a)
}

/// [`recover_forms`] in the disk chart centered at an arbitrary `ctr` (e.g. `tg.z_b`
/// for the order-b vertex chart). The kernel DIMENSION is chart-independent; the
/// coefficient vectors are expansions in the `ctr` chart's w and differ between charts.
/// `assemble_scaled_ami` already accepts `ctr`; this threads it the rest of the way.
#[allow(clippy::too_many_arguments)]
pub fn recover_forms_centered(
    tg64: &TriangleGroup,
    tg: &TriangleGroupHp,
    cg: &CosetGraph,
    k: i64,
    big_n: usize,
    q: usize,
    threshold_decimal: &str,
    tol_decimal: &str,
    rho_scale: f64,
    ctr: &Complex,
) -> Vec<Vec<Complex>> {
    let prec = tg.prec;
    let (a, rho) = assemble_scaled_ami(tg64, tg, cg, k, big_n, q, rho_scale, ctr);
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
        let (a, rho) = assemble_scaled_ami(&tg64, &tg, &cg, k, n, q, 1.0, &tg.z_a);
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
        // M_CENTER selects the expansion vertex: a = order-2 (default), b = order-12, c = order-5.
        let center = std::env::var("M_CENTER").unwrap_or_else(|_| "a".into());
        let ctr = match center.as_str() {
            "b" => tg.z_b.clone(),
            "c" => tg.z_c.clone(),
            _ => tg.z_a.clone(),
        };
        let (a, rho) = assemble_scaled_ami(&tg64, &tg, &cg, k, n, q, 1.0, &ctr);
        let dim = a.len();
        eprintln!("[2,12,5] EXT center={center} k={k} N={n} dim={dim} limbs={nlimbs} prec={prec} ρ={:.6} → {out}", rho.to_f64());
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

    // Same dump as `dump_2_12_5_matrix_ext`, via the streamed constant-memory assembly
    // (row-checkpointed to M_OUT.progress — rerun with identical params to resume).
    #[test]
    #[ignore]
    fn dump_2_12_5_matrix_ext_streamed() {
        let n: usize = std::env::var("M_N").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
        let k: i64 = std::env::var("M_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let out = std::env::var("M_OUT").unwrap_or_else(|_| "/tmp/m_2_12_5_ext.bin".into());
        let prec: u32 = std::env::var("M_PREC").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
        let nlimbs: usize = std::env::var("M_LIMBS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let mut s0: Vec<usize> = vec![0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6];
        let mut s1: Vec<usize> = vec![14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6];
        // M_S0/M_S1: comma-separated 0-based permutation override — selects a DIFFERENT
        // passport member (e.g. the achiral (2A,12B,5A) dessin) through the same machinery.
        if let (Ok(a), Ok(b)) = (std::env::var("M_S0"), std::env::var("M_S1")) {
            s0 = a.split(',').map(|t| t.trim().parse().expect("M_S0")).collect();
            s1 = b.split(',').map(|t| t.trim().parse().expect("M_S1")).collect();
            // degree = permutation length (24 for M24-class dessins, 23 for M23, ...)
            assert_eq!(s0.len(), s1.len());
            eprintln!("[tri] permutations OVERRIDDEN from M_S0/M_S1 (degree {})", s0.len());
        }
        // M_ABC: "a,b,c" triangle-group order override (e.g. "3,4,3" for the (3B,4C,3A)
        // passport). s0 must have order a, s1 order b, (s0 s1)^-1 order c.
        let (oa, ob, oc): (u32, u32, u32) = if let Ok(t) = std::env::var("M_ABC") {
            let v: Vec<u32> = t.split(',').map(|x| x.trim().parse().expect("M_ABC")).collect();
            eprintln!("[tri] group orders OVERRIDDEN: ({}, {}, {})", v[0], v[1], v[2]);
            (v[0], v[1], v[2])
        } else {
            (2, 12, 5)
        };
        // M_BASE=i: conjugate both permutations by the transposition (0 i), i.e. re-mark coset i
        // as the basepoint. Stab(0) becomes the CONJUGATE subgroup α_i Γ' α_i^{-1} — the same
        // curve via z ↦ α_i z — and compactify re-centers the domain around the new base cell.
        // With i in the second 12-cycle of s1, M_CENTER=b then charts the second order-12 point
        // at a well-centered radius (the raw M_CENTER=b2 route sits at ρ≈0.9995+, needing N≈14000).
        if let Some(base) = std::env::var("M_BASE").ok().and_then(|s| s.parse::<usize>().ok()) {
            if base != 0 {
                let p = |x: usize| if x == 0 { base } else if x == base { 0 } else { x };
                let (o0, o1) = (s0.clone(), s1.clone());
                for x in 0..o0.len() {
                    s0[p(x)] = p(o0[x]);
                    s1[p(x)] = p(o1[x]);
                }
                eprintln!("[2,12,5] rebased: coset {base} is now the basepoint (conjugated by (0 {base}))");
            }
        }
        let tg64 = TriangleGroup::new(oa, ob, oc);
        let tg = TriangleGroupHp::new(oa, ob, oc, prec);
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        // M_RPRUNE / M_LMAX: compactify BFS prune radius and word cap (larger triangle
        // groups, e.g. (3,4,8) area 7/24, need r_prune > 0.996 to reach every coset).
        let r_prune: f64 = std::env::var("M_RPRUNE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.996);
        let l_max: usize = std::env::var("M_LMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
        cg.compactify_with(&tg64, r_prune, l_max);
        let q = 2 * n + 8;
        // M_CENTER: a/b/c = the vertex charts; b2 = the SECOND order-12 preimage (the other
        // 12-cycle of s1), i.e. rep_i(z_b) for coset i from M_COSET (default 1, which lies in
        // the cycle (1 2 22 5 8 18 17 7 15 12 3 9)).
        let center = std::env::var("M_CENTER").unwrap_or_else(|_| "a".into());
        let ctr = match center.as_str() {
            "b" => tg.z_b.clone(),
            "b2" => {
                let coset: usize = std::env::var("M_COSET").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
                let ctr = reps_hp(&cg, &tg)[coset].apply(&tg.z_b);
                eprintln!("[2,12,5] b2 center: coset {coset} rep applied to z_b → {:.45}", ctr);
                ctr
            }
            "c2" => {
                let coset: usize = std::env::var("M_COSET").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
                let ctr = reps_hp(&cg, &tg)[coset].apply(&tg.z_c);
                eprintln!("[2,12,5] c2 center: coset {coset} rep applied to z_c → {:.45}", ctr);
                ctr
            }
            "a2" => {
                let coset: usize = std::env::var("M_COSET").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
                let ctr = reps_hp(&cg, &tg)[coset].apply(&tg.z_a);
                eprintln!("[2,12,5] a2 center: coset {coset} rep applied to z_a → {:.45}", ctr);
                ctr
            }
            "c" => tg.z_c.clone(),
            _ => tg.z_a.clone(),
        };
        eprintln!("[2,12,5] EXT-streamed center={center} k={k} N={n} limbs={nlimbs} prec={prec} → {out}");
        eprintln!("[2,12,5] ctr_full = {:.45}", ctr);
        let rho = dump_scaled_ami_streamed(&tg64, &tg, &cg, k, n, q, 1.0, &ctr, nlimbs, &out);
        eprintln!("[2,12,5] EXT-streamed done: dim={} ρ={:.6} ρ_full={:.45}", n + 1, rho.to_f64(), rho);
    }

    // Print the hp coset representatives + triangle data for a frame (M_S0/M_S1/M_ABC/
    // M_BASE/M_RPRUNE/M_LMAX as in the streamed dump). Needed for MULTI-FRAME atlases:
    // the exact transition between frame B and frame B' is  rep_{B'}[j'] ∘ R_v^t ∘ rep_B[j]^{-1}
    // (t = sheet branch), so the Python glue layer needs these matrices at full precision.
    #[test]
    #[ignore]
    fn print_reps_348() {
        let prec: u32 = std::env::var("M_PREC").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
        let mut s0: Vec<usize> = vec![0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6];
        let mut s1: Vec<usize> = vec![14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6];
        if let (Ok(a), Ok(b)) = (std::env::var("M_S0"), std::env::var("M_S1")) {
            s0 = a.split(',').map(|t| t.trim().parse().expect("M_S0")).collect();
            s1 = b.split(',').map(|t| t.trim().parse().expect("M_S1")).collect();
        }
        let (oa, ob, oc): (u32, u32, u32) = if let Ok(t) = std::env::var("M_ABC") {
            let v: Vec<u32> = t.split(',').map(|x| x.trim().parse().expect("M_ABC")).collect();
            (v[0], v[1], v[2])
        } else {
            (2, 12, 5)
        };
        if let Some(base) = std::env::var("M_BASE").ok().and_then(|s| s.parse::<usize>().ok()) {
            if base != 0 {
                let p = |x: usize| if x == 0 { base } else if x == base { 0 } else { x };
                let (o0, o1) = (s0.clone(), s1.clone());
                for x in 0..o0.len() {
                    s0[p(x)] = p(o0[x]);
                    s1[p(x)] = p(o1[x]);
                }
            }
        }
        let tg64 = TriangleGroup::new(oa, ob, oc);
        let tg = TriangleGroupHp::new(oa, ob, oc, prec);
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        let r_prune: f64 = std::env::var("M_RPRUNE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.996);
        let l_max: usize = std::env::var("M_LMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
        cg.compactify_with(&tg64, r_prune, l_max);
        println!("z_a = {:.45}", tg.z_a);
        println!("z_b = {:.45}", tg.z_b);
        println!("z_c = {:.45}", tg.z_c);
        for (nm, m) in [("delta_a", &tg.delta_a), ("delta_b", &tg.delta_b), ("delta_c", &tg.delta_c)] {
            println!("{nm} = [{:.45}, {:.45}, {:.45}, {:.45}]", m.a, m.b, m.c, m.d);
        }
        let reps = reps_hp(&cg, &tg);
        for (j, m) in reps.iter().enumerate() {
            println!("rep[{j}] = [{:.45}, {:.45}, {:.45}, {:.45}]", m.a, m.b, m.c, m.d);
        }
    }

    // Control dump: the KMSV (5,3,3) paper case through the same streamed path.
    // dim S_6 = 3, so the form map should be the RNC-2 in P^2 — the diagnostic for
    // whether the (2,12,5) non-RNC deviation is real geometry or a framework artifact.
    #[test]
    #[ignore]
    fn dump_5_3_3_matrix_ext_streamed() {
        let n: usize = std::env::var("M_N").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
        let k: i64 = std::env::var("M_K").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
        let out = std::env::var("M_OUT").unwrap_or_else(|_| "/tmp/m_5_3_3_ext.bin".into());
        let prec: u32 = std::env::var("M_PREC").ok().and_then(|s| s.parse().ok()).unwrap_or(140);
        let nlimbs: usize = std::env::var("M_LIMBS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let s0: Vec<usize> = vec![4, 0, 1, 2, 3];
        let s1: Vec<usize> = vec![1, 2, 0, 3, 4];
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, prec);
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        let q = 2 * n + 8;
        let center = std::env::var("M_CENTER").unwrap_or_else(|_| "a".into());
        let ctr = match center.as_str() {
            "b" => tg.z_b.clone(),
            "c" => tg.z_c.clone(),
            _ => tg.z_a.clone(),
        };
        eprintln!("[5,3,3] EXT-streamed center={center} k={k} N={n} limbs={nlimbs} prec={prec} → {out}");
        let rho = dump_scaled_ami_streamed(&tg64, &tg, &cg, k, n, q, 1.0, &ctr, nlimbs, &out);
        eprintln!("[5,3,3] EXT-streamed done: dim={} ρ={:.6} ρ_full={:.45}", n + 1, rho.to_f64(), rho);
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

    // recover_forms_centered at ctr = z_a is the SAME code path recover_forms now
    // delegates to; the two calls must agree. The comparison is numeric rather than
    // bitwise only because the rayon fold/reduce in assemble_scaled_ami reassociates
    // hp additions between runs (~1e-75 matrix noise, ~1e-60 after the ρ^{-n}
    // un-scaling) — 1e-30 relative is dozens of decades above that, and dozens below
    // any mathematically meaningful difference.
    #[test]
    fn recover_forms_centered_z_a_matches_uncentered() {
        let (tg64, tg, cg) = setup_5_3_3();
        let f1 = recover_forms(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0);
        let f2 = recover_forms_centered(&tg64, &tg, &cg, 6, 48, 96, "1e-8", "1e-70", 1.0, &tg.z_a);
        assert_eq!(f1.len(), 3);
        assert_eq!(f1.len(), f2.len());
        let mut worst = 0f64;
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                let d = Complex::with_val(PREC, x - y);
                let scale = 1.0 + cmod_f64(x);
                worst = worst.max(cmod_f64(&d) / scale);
            }
        }
        assert!(worst < 1e-30, "z_a-centered recover_forms diverged from recover_forms: {worst:.2e}");
    }

    // dim S_k is chart-independent — the honest cross-chart invariant (coefficients
    // are chart expansions and DO differ). The z_b chart has measured ρ_b ≈ 0.826608
    // (vs ρ_a ≈ 0.528936), so N = 96 gives 96·log10(1/ρ_b) ≈ 7.9 decimal digits;
    // SolveParams(256, 96, 7) derives threshold 1e-5, which the kernel σ ~1e-8 clear
    // by three decades.
    #[test]
    fn dim_s_k_chart_independent_5_3_3() {
        use crate::belyi::solve::SolveParams;
        let (tg64, tg, cg) = setup_5_3_3();
        let sp = SolveParams::new(PREC, 96, 7).expect("z_b chart binding");
        let rho_b = domain_radius_hp_centered(&cg, &tg, &tg.z_b).to_f64();
        sp.check_rho(rho_b).expect("N = 96 must cover 7 digits at the measured ρ_b");
        let (dim_b, sigma) = dim_s_k_svd_centered(
            &tg64, &tg, &cg, 6, sp.big_n, 2 * sp.big_n,
            &sp.threshold_decimal, &sp.tol_decimal, 1.0, &tg.z_b,
        );
        let smallest = sigma.last().unwrap().to_f64();
        assert_eq!(dim_b, 3, "dim S_6 in the z_b chart (z_a chart gives 3); σ_min={smallest:.2e}");
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

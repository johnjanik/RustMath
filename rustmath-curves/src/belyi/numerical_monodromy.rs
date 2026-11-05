//! Numerical monodromy of a solved factorized Belyi map — the honest "is it the
//! right map" check. Given a [2,12,5] solution as `FactorizedRoots`, we form the
//! rational map `φ = N/D` with `N = A²B`, `D = λR⁵S`, and recover the monodromy
//! permutations `σ₀, σ₁` by analytically continuing the 24-point fiber `φ⁻¹(p)`
//! around loops encircling `0` and `1`. This uses ONLY the numerical roots — it does
//! not consult the flag triangulation the seed was built from — so a match to the
//! target passport (`2⁸1⁸ | 12² | 5⁴1⁴`, transitive, primitive) and to the original
//! dessin would be a genuine, non-circular verification.
//!
//! Evaluation is in FACTORED / logarithmic-derivative form: the power-basis
//! coefficients of `N`,`D` are astronomically large (products of 24 roots spanning
//! distance ~5, magnitudes ~1e28), so power-basis evaluation loses all accuracy.
//! Products `∏(z−root)` stay machine-relative accurate at any magnitude.

use super::factorized_residual::FactorizedRoots;
use num_complex::Complex64;
use rug::ops::Pow;
use rug::{Complex, Float};

// ============================ f64 factored-form ============================

/// `g(z) = N(z) − p·D(z)` and `g'(z)`, factored form.
/// `N = ∏(z−aᵢ)²·∏(z−bⱼ)`, `D = λ·∏(z−rₖ)⁵·∏(z−sₗ)`.
fn g_and_deriv(sol: &FactorizedRoots, z: Complex64, p: Complex64) -> (Complex64, Complex64) {
    let mut nprod = Complex64::new(1.0, 0.0);
    let mut ln = Complex64::new(0.0, 0.0); // N'/N
    for &a in &sol.roots_a {
        let w = z - a;
        nprod *= w * w;
        ln += 2.0 / w;
    }
    for &b in &sol.roots_b {
        let w = z - b;
        nprod *= w;
        ln += 1.0 / w;
    }
    let mut dprod = sol.lambda;
    let mut ld = Complex64::new(0.0, 0.0); // D'/D
    for &r in &sol.roots_r {
        let w = z - r;
        dprod *= w.powu(5);
        ld += 5.0 / w;
    }
    for &s in &sol.roots_s {
        let w = z - s;
        dprod *= w;
        ld += 1.0 / w;
    }
    let g = nprod - p * dprod;
    let gp = nprod * ln - p * (dprod * ld);
    (g, gp)
}

/// `φ(z) = N(z)/D(z)` in factored form.
pub fn phi(sol: &FactorizedRoots, z: Complex64) -> Complex64 {
    let mut nprod = Complex64::new(1.0, 0.0);
    for &a in &sol.roots_a {
        let w = z - a;
        nprod *= w * w;
    }
    for &b in &sol.roots_b {
        nprod *= z - b;
    }
    let mut dprod = sol.lambda;
    for &r in &sol.roots_r {
        dprod *= (z - r).powu(5);
    }
    for &s in &sol.roots_s {
        dprod *= z - s;
    }
    nprod / dprod
}

fn newton_factored(sol: &FactorizedRoots, p: Complex64, z0: Complex64) -> Option<Complex64> {
    let mut z = z0;
    for _ in 0..80 {
        let (g, gp) = g_and_deriv(sol, z, p);
        if gp.norm() < 1e-300 {
            return None;
        }
        let dz = g / gp;
        z -= dz;
        if !z.re.is_finite() || !z.im.is_finite() {
            return None;
        }
        if dz.norm() < 1e-14 * (1.0 + z.norm()) {
            return Some(z);
        }
    }
    Some(z)
}

/// The full 24-point fiber `φ⁻¹(p)` via a grid of Newton starts + dedupe. The
/// ramification is crushed into a tiny "pinhole" around the centroid of the finite
/// critical points; the grid is centered there and scaled to the cluster extent.
pub fn fiber_over_factored(sol: &FactorizedRoots, p: Complex64) -> Vec<Complex64> {
    let mut pts: Vec<Complex64> = Vec::new();
    pts.extend(&sol.roots_a);
    pts.extend(&sol.roots_b);
    pts.extend(&sol.roots_r);
    pts.extend(&sol.roots_s);
    let cen = pts.iter().sum::<Complex64>() / (pts.len() as f64);
    let ext = pts
        .iter()
        .map(|z| (z - cen).norm())
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let radius = ext * 30.0;
    let mut roots: Vec<Complex64> = Vec::new();
    let m = 201;
    for gi in 0..m {
        for gj in 0..m {
            let x = cen.re - radius + 2.0 * radius * (gi as f64) / ((m - 1) as f64);
            let y = cen.im - radius + 2.0 * radius * (gj as f64) / ((m - 1) as f64);
            if let Some(z) = newton_factored(sol, p, Complex64::new(x, y)) {
                if (phi(sol, z) - p).norm() < 1e-6 * (1.0 + p.norm())
                    && roots.iter().all(|r| (r - z).norm() > 1e-4 * ext)
                {
                    roots.push(z);
                }
            }
        }
    }
    roots
}

fn match_nearest(base: &[Complex64], moved: &[Complex64]) -> (Vec<usize>, f64) {
    let mut perm = vec![0usize; moved.len()];
    let mut conf = f64::INFINITY;
    for (i, m) in moved.iter().enumerate() {
        let (mut d1, mut d2, mut j1) = (f64::INFINITY, f64::INFINITY, 0usize);
        for (j, b) in base.iter().enumerate() {
            let d = (m - b).norm();
            if d < d1 {
                d2 = d1;
                d1 = d;
                j1 = j;
            } else if d < d2 {
                d2 = d;
            }
        }
        perm[i] = j1;
        if d1 > 0.0 {
            conf = conf.min(d2 / d1);
        }
    }
    (perm, conf)
}

/// Track the fiber (factored f64) around a CCW circle centered at `center` through
/// `p* = φ(base[0])`. Returns `(perm, confidence, is_bijection)`.
pub fn track_loop_factored(
    sol: &FactorizedRoots,
    base_fiber: &[Complex64],
    center: Complex64,
    n_steps: usize,
) -> (Vec<usize>, f64, bool) {
    let p_star = phi(sol, base_fiber[0]);
    let radius = (p_star - center).norm();
    let theta0 = (p_star - center).arg();
    let mut cur = base_fiber.to_vec();
    for step in 1..=n_steps {
        let theta = theta0 + 2.0 * std::f64::consts::PI * (step as f64) / (n_steps as f64);
        let p = center + Complex64::from_polar(radius, theta);
        for zi in cur.iter_mut() {
            if let Some(z) = newton_factored(sol, p, *zi) {
                *zi = z;
            }
        }
    }
    let (perm, conf) = match_nearest(base_fiber, &cur);
    let bij = is_bijection(&perm);
    (perm, conf, bij)
}

// ============================ hp factored-form (the pinhole needs it) ============================

fn c64_to_hp(z: Complex64, prec: u32) -> Complex {
    Complex::with_val(prec, (z.re, z.im))
}

fn cmod_hp(z: &Complex, prec: u32) -> Float {
    let re = Float::with_val(prec, z.real());
    let im = Float::with_val(prec, z.imag());
    let r2 = Float::with_val(prec, &re * &re);
    let i2 = Float::with_val(prec, &im * &im);
    Float::with_val(prec, &r2 + &i2).sqrt()
}

fn cmul(a: &Complex, b: &Complex, prec: u32) -> Complex {
    Complex::with_val(prec, a * b)
}
fn cdiv_scalar(k: f64, w: &Complex, prec: u32) -> Complex {
    let kk = Complex::with_val(prec, (k, 0.0));
    Complex::with_val(prec, &kk / w)
}

/// hp roots of the solution, converted once.
pub struct SolHp {
    a: Vec<Complex>,
    b: Vec<Complex>,
    r: Vec<Complex>,
    s: Vec<Complex>,
    lambda: Complex,
    prec: u32,
}

pub fn sol_to_hp(sol: &FactorizedRoots, prec: u32) -> SolHp {
    let cv = |v: &[Complex64]| v.iter().map(|&z| c64_to_hp(z, prec)).collect::<Vec<_>>();
    SolHp {
        a: cv(&sol.roots_a),
        b: cv(&sol.roots_b),
        r: cv(&sol.roots_r),
        s: cv(&sol.roots_s),
        lambda: c64_to_hp(sol.lambda, prec),
        prec,
    }
}

fn g_and_deriv_hp(s: &SolHp, z: &Complex, p: &Complex) -> (Complex, Complex) {
    let prec = s.prec;
    let mut nprod = Complex::with_val(prec, (1.0, 0.0));
    let mut ln = Complex::with_val(prec, (0.0, 0.0));
    for ai in &s.a {
        let w = Complex::with_val(prec, z - ai);
        nprod = cmul(&nprod, &cmul(&w, &w, prec), prec);
        ln += cdiv_scalar(2.0, &w, prec);
    }
    for bi in &s.b {
        let w = Complex::with_val(prec, z - bi);
        nprod = cmul(&nprod, &w, prec);
        ln += cdiv_scalar(1.0, &w, prec);
    }
    let mut dprod = s.lambda.clone();
    let mut ld = Complex::with_val(prec, (0.0, 0.0));
    for ri in &s.r {
        let w = Complex::with_val(prec, z - ri);
        let mut w5 = w.clone();
        for _ in 0..4 {
            w5 = cmul(&w5, &w, prec);
        }
        dprod = cmul(&dprod, &w5, prec);
        ld += cdiv_scalar(5.0, &w, prec);
    }
    for si in &s.s {
        let w = Complex::with_val(prec, z - si);
        dprod = cmul(&dprod, &w, prec);
        ld += cdiv_scalar(1.0, &w, prec);
    }
    let pd = cmul(p, &dprod, prec);
    let g = Complex::with_val(prec, &nprod - &pd);
    let t1 = cmul(&nprod, &ln, prec);
    let dl = cmul(&dprod, &ld, prec);
    let t2 = cmul(p, &dl, prec);
    let gp = Complex::with_val(prec, &t1 - &t2);
    (g, gp)
}

/// `φ(z) = N(z)/D(z)` in hp factored form.
pub fn phi_hp(s: &SolHp, z: &Complex) -> Complex {
    let prec = s.prec;
    let mut nprod = Complex::with_val(prec, (1.0, 0.0));
    for ai in &s.a {
        let w = Complex::with_val(prec, z - ai);
        nprod = cmul(&nprod, &cmul(&w, &w, prec), prec);
    }
    for bi in &s.b {
        nprod = cmul(&nprod, &Complex::with_val(prec, z - bi), prec);
    }
    let mut dprod = s.lambda.clone();
    for ri in &s.r {
        let w = Complex::with_val(prec, z - ri);
        let mut w5 = w.clone();
        for _ in 0..4 {
            w5 = cmul(&w5, &w, prec);
        }
        dprod = cmul(&dprod, &w5, prec);
    }
    for si in &s.s {
        dprod = cmul(&dprod, &Complex::with_val(prec, z - si), prec);
    }
    Complex::with_val(prec, &nprod / &dprod)
}

fn newton_factored_hp(s: &SolHp, p: &Complex, z0: &Complex) -> Complex {
    let prec = s.prec;
    let tol = Float::with_val(prec, 10.0).pow(-((prec as f64 * 0.28) as i32));
    let mut z = z0.clone();
    for _ in 0..100 {
        let (g, gp) = g_and_deriv_hp(s, &z, p);
        if cmod_hp(&gp, prec) < Float::with_val(prec, 1e-290) {
            break;
        }
        let dz = Complex::with_val(prec, &g / &gp);
        z -= &dz;
        if cmod_hp(&dz, prec) < tol {
            break;
        }
    }
    z
}

fn match_nearest_hp(base: &[Complex], moved: &[Complex], prec: u32) -> (Vec<usize>, f64) {
    let mut perm = vec![0usize; moved.len()];
    let mut conf = f64::INFINITY;
    for (i, m) in moved.iter().enumerate() {
        let (mut d1, mut d2, mut j1) = (f64::INFINITY, f64::INFINITY, 0usize);
        for (j, b) in base.iter().enumerate() {
            let diff = Complex::with_val(prec, m - b);
            let dd = cmod_hp(&diff, prec).to_f64();
            if dd < d1 {
                d2 = d1;
                d1 = dd;
                j1 = j;
            } else if dd < d2 {
                d2 = dd;
            }
        }
        perm[i] = j1;
        if d1 > 0.0 {
            conf = conf.min(d2 / d1);
        }
    }
    (perm, conf)
}

/// hp path-tracking of the (pinhole) fiber around a CCW circle centered at `center`
/// through `p* = φ(base[0])`. Base fiber given as f64 approximations, polished in hp.
pub fn track_loop_factored_hp(
    s: &SolHp,
    base_fiber_f64: &[Complex64],
    center: Complex64,
    n_steps: usize,
) -> (Vec<usize>, f64, bool) {
    let prec = s.prec;
    let mut cur: Vec<Complex> = base_fiber_f64.iter().map(|&z| c64_to_hp(z, prec)).collect();
    let p_star = phi_hp(s, &cur[0]);
    let p_star_f64 = Complex64::new(p_star.real().to_f64(), p_star.imag().to_f64());
    let radius = (p_star_f64 - center).norm();
    let theta0 = (p_star_f64 - center).arg();
    let center_hp = c64_to_hp(center, prec);
    for zi in cur.iter_mut() {
        *zi = newton_factored_hp(s, &p_star, zi);
    }
    let base_hp = cur.clone();
    for step in 1..=n_steps {
        let theta = theta0 + 2.0 * std::f64::consts::PI * (step as f64) / (n_steps as f64);
        let pr = Float::with_val(prec, radius * theta.cos());
        let pi = Float::with_val(prec, radius * theta.sin());
        let p = Complex::with_val(prec, (pr, pi)) + &center_hp;
        for zi in cur.iter_mut() {
            *zi = newton_factored_hp(s, &p, zi);
        }
    }
    let (perm, conf) = match_nearest_hp(&base_hp, &cur, prec);
    let bij = is_bijection(&perm);
    (perm, conf, bij)
}

/// Diagnostic: minimum inter-sheet distance seen anywhere on the loop (hp). If far
/// below the solution accuracy the sheets crush together and no tracker can follow.
pub fn loop_min_approach_hp(
    s: &SolHp,
    base_fiber_f64: &[Complex64],
    center: Complex64,
    n_steps: usize,
) -> f64 {
    let prec = s.prec;
    let mut cur: Vec<Complex> = base_fiber_f64.iter().map(|&z| c64_to_hp(z, prec)).collect();
    let p_star = phi_hp(s, &cur[0]);
    let p_star_f64 = Complex64::new(p_star.real().to_f64(), p_star.imag().to_f64());
    let radius = (p_star_f64 - center).norm();
    let theta0 = (p_star_f64 - center).arg();
    let center_hp = c64_to_hp(center, prec);
    for zi in cur.iter_mut() {
        *zi = newton_factored_hp(s, &p_star, zi);
    }
    let mut min_app = f64::INFINITY;
    for step in 1..=n_steps {
        let theta = theta0 + 2.0 * std::f64::consts::PI * (step as f64) / (n_steps as f64);
        let pr = Float::with_val(prec, radius * theta.cos());
        let pi = Float::with_val(prec, radius * theta.sin());
        let p = Complex::with_val(prec, (pr, pi)) + &center_hp;
        for zi in cur.iter_mut() {
            *zi = newton_factored_hp(s, &p, zi);
        }
        for i in 0..cur.len() {
            for j in (i + 1)..cur.len() {
                let diff = Complex::with_val(prec, &cur[i] - &cur[j]);
                min_app = min_app.min(cmod_hp(&diff, prec).to_f64());
            }
        }
    }
    min_app
}

// ============================ permutation utilities ============================

fn is_bijection(perm: &[usize]) -> bool {
    let mut seen = vec![false; perm.len()];
    for &j in perm {
        if j >= perm.len() || seen[j] {
            return false;
        }
        seen[j] = true;
    }
    true
}

/// Cycle-type multiset (sorted descending) of a permutation given as `perm[i]=image`.
pub fn cycle_type(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut seen = vec![false; n];
    let mut cyc = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut len = 0;
        let mut x = start;
        while !seen[x] {
            seen[x] = true;
            x = perm[x];
            len += 1;
        }
        cyc.push(len);
    }
    cyc.sort_unstable_by(|a, b| b.cmp(a));
    cyc
}

/// Compose permutations: `(g∘f)[i] = g[f[i]]`.
pub fn compose(g: &[usize], f: &[usize]) -> Vec<usize> {
    f.iter().map(|&i| g[i]).collect()
}

pub fn inverse(p: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; p.len()];
    for (i, &j) in p.iter().enumerate() {
        inv[j] = i;
    }
    inv
}

/// Is `⟨generators⟩` transitive on `{0..n}`?
pub fn is_transitive(n: usize, gens: &[&[usize]]) -> bool {
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1;
    while let Some(x) = stack.pop() {
        for g in gens {
            let y = g[x];
            if !seen[y] {
                seen[y] = true;
                count += 1;
                stack.push(y);
            }
        }
    }
    count == n
}

/// Is the transitive action primitive (no nontrivial block containing `0`)?
pub fn is_primitive(n: usize, gens: &[&[usize]]) -> bool {
    if !is_transitive(n, gens) {
        return false;
    }
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != c {
            let nx = parent[c];
            parent[c] = r;
            c = nx;
        }
        r
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    for k in 1..n {
        let mut parent: Vec<usize> = (0..n).collect();
        union(&mut parent, 0, k);
        loop {
            let mut changed = false;
            for a in 0..n {
                for b in (a + 1)..n {
                    if find(&mut parent, a) == find(&mut parent, b) {
                        for g in gens {
                            let (ga, gb) = (g[a], g[b]);
                            if find(&mut parent, ga) != find(&mut parent, gb) {
                                union(&mut parent, ga, gb);
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let r0 = find(&mut parent, 0);
        let size = (0..n).filter(|&x| find(&mut parent, x) == r0).count();
        if size != 1 && size != n {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_type_and_utils() {
        // (0 1 2)(3 4) on 5 points
        let p = vec![1, 2, 0, 4, 3];
        assert_eq!(cycle_type(&p), vec![3, 2]);
        let inv = inverse(&p);
        assert_eq!(compose(&p, &inv), vec![0, 1, 2, 3, 4]);
        assert!(is_bijection(&p));
        assert!(!is_bijection(&[0, 0, 1]));
    }

    #[test]
    fn transitivity_and_primitivity() {
        // full cycle is transitive and (on prime 5) primitive
        let c5: Vec<usize> = vec![1, 2, 3, 4, 0];
        assert!(is_transitive(5, &[&c5]));
        assert!(is_primitive(5, &[&c5]));
        // (0 2)(1 3) + (0 1)(2 3) generate an imprimitive group on 4 pts (blocks {0,2},{1,3})
        let a: Vec<usize> = vec![2, 3, 0, 1];
        let b: Vec<usize> = vec![1, 0, 3, 2];
        assert!(is_transitive(4, &[&a, &b]));
        assert!(!is_primitive(4, &[&a, &b]));
    }
}

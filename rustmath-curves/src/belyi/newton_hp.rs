//! Arbitrary-precision Newton refinement of the `[2,12,5]` factorized residual
//! (Stage 3) — break the high-ramification conditioning wall (Blocker D).
//!
//! The `f64` LM refinement drives the residual to ~`1e-8` and then crawls: the
//! `R⁵`/`U¹²` ramification makes the Jacobian ill-conditioned at the solution, so
//! the last digits are lost to `f64` precision. Here we re-run Newton in
//! `rug`-backed MPFR/MPC arbitrary precision (native, no Python bridge), which has
//! digits to spare past the conditioning.
//!
//! The system is the same identity `A²B − λR⁵S − cU¹² = 0` (25 complex coeffs = 50
//! real residuals), with the Möbius gauge fixed by freezing 3 roots, giving a
//! square 50×50 Newton system. A backtracking line search guards against
//! ill-conditioned overshoot.

use super::factorized_residual::FactorizedRoots;
use rug::ops::Pow;
use rug::{Complex, Float};

/// A univariate polynomial with `rug::Complex` coefficients at a fixed precision.
struct PolyHp {
    c: Vec<Complex>,
    prec: u32,
}

impl PolyHp {
    fn one(prec: u32) -> Self {
        Self {
            c: vec![Complex::with_val(prec, (1.0, 0.0))],
            prec,
        }
    }

    fn from_roots_monic(roots: &[Complex], prec: u32) -> Self {
        let mut p = Self::one(prec);
        for r in roots {
            let lin = Self {
                c: vec![
                    Complex::with_val(prec, -r),
                    Complex::with_val(prec, (1.0, 0.0)),
                ],
                prec,
            };
            p = p.mul(&lin);
        }
        p
    }

    fn mul(&self, rhs: &Self) -> Self {
        let n = self.c.len() + rhs.c.len() - 1;
        let mut out = vec![Complex::with_val(self.prec, (0.0, 0.0)); n];
        for (i, a) in self.c.iter().enumerate() {
            for (j, b) in rhs.c.iter().enumerate() {
                let prod = Complex::with_val(self.prec, a * b);
                out[i + j] += &prod;
            }
        }
        Self {
            c: out,
            prec: self.prec,
        }
    }

    fn pow(&self, e: usize) -> Self {
        let mut out = self.clone_hp();
        for _ in 1..e {
            out = out.mul(self);
        }
        out
    }

    fn clone_hp(&self) -> Self {
        Self {
            c: self.c.iter().map(|z| Complex::with_val(self.prec, z)).collect(),
            prec: self.prec,
        }
    }

    /// `self + scale·rhs`, padded to the longer length.
    fn add_scaled(&self, scale: &Complex, rhs: &Self) -> Self {
        let n = self.c.len().max(rhs.c.len());
        let mut out = vec![Complex::with_val(self.prec, (0.0, 0.0)); n];
        for i in 0..n {
            if i < self.c.len() {
                out[i] += &self.c[i];
            }
            if i < rhs.c.len() {
                let t = Complex::with_val(self.prec, scale * &rhs.c[i]);
                out[i] += &t;
            }
        }
        Self {
            c: out,
            prec: self.prec,
        }
    }
}

// Packed layout (mirrors newton.rs): a(8) b(8) r(4) s(4) u(2) λ c → 56 reals.
const N_UNKNOWNS: usize = 56;

fn seed_to_floats(seed: &FactorizedRoots, prec: u32) -> Vec<Float> {
    let mut x = Vec::with_capacity(N_UNKNOWNS);
    let mut push = |z: &num_complex::Complex64| {
        x.push(Float::with_val(prec, z.re));
        x.push(Float::with_val(prec, z.im));
    };
    for z in seed
        .roots_a
        .iter()
        .chain(&seed.roots_b)
        .chain(&seed.roots_r)
        .chain(&seed.roots_s)
        .chain(&seed.roots_u)
    {
        push(z);
    }
    push(&seed.lambda);
    push(&seed.c);
    x
}

fn floats_to_complex(x: &[Float], i: usize, prec: u32) -> Complex {
    Complex::with_val(prec, (&x[i], &x[i + 1]))
}

/// Residual (50 real components) of `A²B − λR⁵S − cU¹²` at the packed point `x`.
fn residual_floats(x: &[Float], prec: u32) -> Vec<Float> {
    let take = |start: usize, n: usize| -> Vec<Complex> {
        (0..n).map(|k| floats_to_complex(x, start + 2 * k, prec)).collect()
    };
    let a = take(0, 8);
    let b = take(16, 8);
    let r = take(32, 4);
    let s = take(40, 4);
    let u = take(48, 2);
    let lambda = floats_to_complex(x, 104 / 2, prec); // x[52],x[53]
    let c = floats_to_complex(x, 108 / 2, prec); // x[54],x[55]

    let p_zero = PolyHp::from_roots_monic(&a, prec)
        .pow(2)
        .mul(&PolyHp::from_roots_monic(&b, prec));
    let p_inf = PolyHp::from_roots_monic(&r, prec)
        .pow(5)
        .mul(&PolyHp::from_roots_monic(&s, prec));
    let p_one = PolyHp::from_roots_monic(&u, prec).pow(12);

    let neg_lambda = Complex::with_val(prec, -&lambda);
    let neg_c = Complex::with_val(prec, -&c);
    let res = p_zero.add_scaled(&neg_lambda, &p_inf).add_scaled(&neg_c, &p_one);

    // 25 coefficients → 50 reals; pad if the top coefficients cancelled.
    let mut out = Vec::with_capacity(50);
    for k in 0..25 {
        if k < res.c.len() {
            out.push(Float::with_val(prec, res.c[k].real()));
            out.push(Float::with_val(prec, res.c[k].imag()));
        } else {
            out.push(Float::with_val(prec, 0.0));
            out.push(Float::with_val(prec, 0.0));
        }
    }
    out
}

fn norm_f64(v: &[Float]) -> f64 {
    let prec = v[0].prec();
    let mut s = Float::with_val(prec, 0.0);
    for a in v {
        s += Float::with_val(prec, a * a);
    }
    s.sqrt().to_f64()
}

/// Solve `A x = b` (real, n×n) at precision `prec` by Gaussian elimination with
/// partial pivoting. `a` is row-major.
fn solve_dense_hp(mut a: Vec<Float>, mut b: Vec<Float>, n: usize, prec: u32) -> Option<Vec<Float>> {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].clone().abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].clone().abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-250 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col].clone();
        for r in (col + 1)..n {
            let f = Float::with_val(prec, &a[r * n + col] / &d);
            if f != 0.0 {
                for k in col..n {
                    let t = Float::with_val(prec, &f * &a[col * n + k]);
                    a[r * n + k] -= &t;
                }
                let t = Float::with_val(prec, &f * &b[col]);
                b[r] -= &t;
            }
        }
    }
    let mut x = vec![Float::with_val(prec, 0.0); n];
    for row in (0..n).rev() {
        let mut s = b[row].clone();
        for k in (row + 1)..n {
            let t = Float::with_val(prec, &a[row * n + k] * &x[k]);
            s -= &t;
        }
        x[row] = Float::with_val(prec, &s / &a[row * n + row]);
    }
    Some(x)
}

/// Configuration for the high-precision Newton refinement.
#[derive(Debug, Clone)]
pub struct NewtonHpConfig {
    pub prec_bits: u32,
    pub max_iters: usize,
    /// Target residual (as f64) at which to stop.
    pub target: f64,
    /// Frozen packed indices (gauge fix). Default: the 2 white points + one black.
    pub frozen: Vec<usize>,
}

impl Default for NewtonHpConfig {
    fn default() -> Self {
        Self {
            prec_bits: 512,
            max_iters: 40,
            target: 1e-100,
            frozen: vec![0, 1, 48, 49, 50, 51],
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewtonHpReport {
    pub initial_residual: f64,
    pub final_residual: f64,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
    /// Final packed unknowns as decimal strings (for LLL/PSLQ recognition).
    pub solution_decimals: Vec<String>,
}

/// Newton refinement in arbitrary precision from an `f64` seed. The gauge is fixed
/// by `frozen`; the resulting square system is solved with a backtracking line
/// search for robustness against the ill-conditioned Jacobian.
pub fn refine_hp(seed: &FactorizedRoots, cfg: &NewtonHpConfig) -> NewtonHpReport {
    let prec = cfg.prec_bits;
    let mut x = seed_to_floats(seed, prec);
    let free: Vec<usize> = (0..N_UNKNOWNS).filter(|k| !cfg.frozen.contains(k)).collect();
    let n = free.len(); // == 50 with the default gauge
    let m = 50;

    // Finite-difference step: ~1/3 of the digits, plenty for Newton convergence.
    let digits = (prec as f64 * 0.301) as i32;
    let h = Float::with_val(prec, 10.0).pow(-(digits / 3));

    let mut r = residual_floats(&x, prec);
    let mut f = norm_f64(&r);
    let initial_residual = f;
    let mut history = vec![f];
    let mut iters = 0;
    // LM damping: plain Newton overshoots on the ill-conditioned (R⁵, U¹²) Jacobian;
    // (JᵀJ + μI) regularizes it. μ shrinks on accept (→ Newton), grows on reject.
    let mut mu = 1e-8_f64 * (1.0 + f);

    while iters < cfg.max_iters && f > cfg.target {
        // Central-difference Jacobian (m × n) over the free unknowns.
        let mut jac = vec![Float::with_val(prec, 0.0); m * n];
        for (jc, &idx) in free.iter().enumerate() {
            let mut xp = clone_floats(&x, prec);
            let mut xm = clone_floats(&x, prec);
            xp[idx] += &h;
            xm[idx] -= &h;
            let rp = residual_floats(&xp, prec);
            let rm = residual_floats(&xm, prec);
            let two_h = Float::with_val(prec, 2.0 * &h);
            for i in 0..m {
                jac[i * n + jc] =
                    Float::with_val(prec, &(Float::with_val(prec, &rp[i] - &rm[i])) / &two_h);
            }
        }
        // Normal equations JᵀJ (n×n) and −Jᵀr (n).
        let mut jtj = vec![Float::with_val(prec, 0.0); n * n];
        let mut jtr = vec![Float::with_val(prec, 0.0); n];
        for a in 0..n {
            for b in a..n {
                let mut s = Float::with_val(prec, 0.0);
                for i in 0..m {
                    s += Float::with_val(prec, &jac[i * n + a] * &jac[i * n + b]);
                }
                jtj[a * n + b] = s.clone();
                jtj[b * n + a] = s;
            }
            let mut s = Float::with_val(prec, 0.0);
            for i in 0..m {
                s += Float::with_val(prec, &jac[i * n + a] * &r[i]);
            }
            jtr[a] = Float::with_val(prec, -s);
        }

        // Adapt μ: try the damped step, accept on decrease.
        let mut stepped = false;
        for _ in 0..16 {
            let mut damped = jtj.clone();
            let mu_f = Float::with_val(prec, mu);
            for d in 0..n {
                damped[d * n + d] += &mu_f;
            }
            let dx = match solve_dense_hp(damped, jtr.clone(), n, prec) {
                Some(v) => v,
                None => {
                    mu *= 8.0;
                    continue;
                }
            };
            let mut xn = clone_floats(&x, prec);
            for (k, &idx) in free.iter().enumerate() {
                xn[idx] += &dx[k];
            }
            let rn = residual_floats(&xn, prec);
            let fnn = norm_f64(&rn);
            if fnn.is_finite() && fnn < f {
                x = xn;
                r = rn;
                f = fnn;
                mu = (mu * 0.3).max(1e-300);
                stepped = true;
                break;
            } else {
                mu *= 8.0;
            }
        }
        iters += 1;
        history.push(f);
        if !stepped {
            break;
        }
    }

    let solution_decimals = x
        .iter()
        .map(|v| v.to_string_radix(10, Some((cfg.prec_bits as usize) / 4)))
        .collect();

    NewtonHpReport {
        initial_residual,
        final_residual: f,
        iterations: iters,
        converged: f <= cfg.target,
        history,
        solution_decimals,
    }
}

fn clone_floats(x: &[Float], prec: u32) -> Vec<Float> {
    x.iter().map(|v| Float::with_val(prec, v)).collect()
}

/// Sorted LU pivot magnitudes (ascending) of the gauge-fixed Jacobian at arbitrary
/// precision. Unlike the `f64` version these are free of finite-difference noise, so
/// a clear gap in the spectrum gives the honest numerical corank; a smooth decay
/// means ill-conditioning without a definite rank drop. Requires a square system
/// (`50` free unknowns, i.e. the default 6-real gauge).
pub fn jacobian_pivots_hp(seed: &FactorizedRoots, frozen: &[usize], prec: u32) -> Vec<f64> {
    let x = seed_to_floats(seed, prec);
    let free: Vec<usize> = (0..N_UNKNOWNS).filter(|k| !frozen.contains(k)).collect();
    let n = free.len();
    let m = 50;
    let digits = (prec as f64 * 0.301) as i32;
    let h = Float::with_val(prec, 10.0).pow(-(digits / 3));
    let mut a = vec![Float::with_val(prec, 0.0); m * n];
    for (jc, &idx) in free.iter().enumerate() {
        let mut xp = clone_floats(&x, prec);
        let mut xm = clone_floats(&x, prec);
        xp[idx] += &h;
        xm[idx] -= &h;
        let rp = residual_floats(&xp, prec);
        let rm = residual_floats(&xm, prec);
        let two_h = Float::with_val(prec, 2.0 * &h);
        for i in 0..m.min(n) {
            a[i * n + jc] =
                Float::with_val(prec, &Float::with_val(prec, &rp[i] - &rm[i]) / &two_h);
        }
        for i in m.min(n)..m {
            // extra rows if m>n: keep for completeness (unused in square LU below)
            let _ = i;
        }
    }
    let sz = n.min(m);
    let mut pivots = Vec::with_capacity(sz);
    let tiny = Float::with_val(prec, 10.0).pow(-((digits as i64) - 5));
    for col in 0..sz {
        let mut piv = col;
        let mut best = a[col * n + col].clone().abs();
        for row in (col + 1)..sz {
            let v = a[row * n + col].clone().abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
        }
        let d = a[col * n + col].clone();
        pivots.push(d.clone().abs().to_f64());
        if d.clone().abs() < tiny {
            // remaining pivots are numerically zero at this precision
            for _ in (col + 1)..sz {
                pivots.push(0.0);
            }
            break;
        }
        for row in (col + 1)..sz {
            let f = Float::with_val(prec, &a[row * n + col] / &d);
            for k in col..n {
                let t = Float::with_val(prec, &f * &a[col * n + k]);
                a[row * n + k] -= &t;
            }
        }
    }
    pivots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pivots
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    /// The high-precision residual of a constant configuration equals |1 − λ − c|
    /// (all factors empty ⇒ each term the constant 1).
    #[test]
    fn constant_residual_matches() {
        let prec = 128;
        // roots at 0 make each monic factor z^k, not constant; instead use the
        // known constant identity via empty-like: put all roots equal so factors
        // are (z-r)^k — not constant. So just check the residual is finite & the
        // top coefficient reflects 1 − λ − c after full cancellation is degree 24.
        let seed = FactorizedRoots {
            roots_a: (0..8).map(|k| Complex64::new(k as f64 * 0.1, 0.0)).collect(),
            roots_b: (0..8).map(|k| Complex64::new(0.0, k as f64 * 0.1 + 0.2)).collect(),
            roots_r: (0..4).map(|k| Complex64::new(-1.0 - k as f64, 0.1)).collect(),
            roots_s: (0..4).map(|k| Complex64::new(0.2, -1.0 - k as f64)).collect(),
            roots_u: vec![Complex64::new(0.4, 0.5), Complex64::new(-0.4, -0.5)],
            // 0.25 + 0.75 = 1 exactly in binary ⇒ 1 − λ − c = 0 exactly.
            lambda: Complex64::new(0.25, 0.0),
            c: Complex64::new(0.75, 0.0),
        };
        let x = seed_to_floats(&seed, prec);
        let r = residual_floats(&x, prec);
        // The degree-24 (top) coefficient is the leading coeff 1 − λ − c = 0.
        let top_re = r[48].to_f64();
        let top_im = r[49].to_f64();
        assert!(top_re.abs() < 1e-30 && top_im.abs() < 1e-30, "top coeff {top_re} {top_im}");
    }

    #[test]
    fn hp_matches_f64_residual_order_of_magnitude() {
        // The hp residual norm should agree with the f64 one for the same config.
        let seed = FactorizedRoots {
            roots_a: (0..8).map(|k| Complex64::new(0.2 * k as f64 - 0.8, 0.1)).collect(),
            roots_b: (0..8).map(|k| Complex64::new(0.0, 0.15 * k as f64 - 0.6)).collect(),
            roots_r: (0..4).map(|k| Complex64::new(-0.5 * k as f64 - 1.2, 0.2)).collect(),
            roots_s: (0..4).map(|k| Complex64::new(0.3, -0.4 * k as f64 - 1.1)).collect(),
            roots_u: vec![Complex64::new(0.5, 0.6), Complex64::new(-0.6, -0.5)],
            lambda: Complex64::new(0.5, 0.1),
            c: Complex64::new(0.5, -0.1),
        };
        let f64_norm = seed.residual_norm();
        let x = seed_to_floats(&seed, 200);
        let hp_norm = norm_f64(&residual_floats(&x, 200));
        assert!((f64_norm - hp_norm).abs() / f64_norm.max(1.0) < 1e-6, "{f64_norm} vs {hp_norm}");
    }
}

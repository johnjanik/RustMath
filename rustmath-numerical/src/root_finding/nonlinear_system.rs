//! Multivariate nonlinear systems: damped Gauss–Newton and the **true-root
//! detector**.
//!
//! Ported from the M23 Belyi construction campaign
//! (`/home/john/inverse_galois/M23/dessin_build/`, `solve_roots.py` +
//! the pure-Newton diagnostic). See `RustMath/M23_BELYI_CONIC_PORT_SPEC.md` §P1.
//!
//! Motivation: least-squares Belyi solvers routinely converge to *non-root*
//! stationary points of ‖r‖² (e.g. a near-degenerate configuration with two
//! ramification points colliding). Such a point has small residual yet is not a
//! solution. The campaign's diagnostic: from a genuine simple root, a **full
//! (undamped) Gauss–Newton step drives the residual quadratically to 0**; from a
//! spurious minimum the full step either stalls at a nonzero residual (∇½‖r‖²≈0
//! with r≠0) or *blows up* (the observed `6.5e-10 → 1.0`). `classify_candidate`
//! turns that into a verdict so no spurious cover is ever accepted downstream.
//!
//! Everything is `f64`; a caller with complex unknowns supplies a real residual
//! (interleave re/im), exactly as the campaign did. High-precision refinement of
//! a `TrueRoot` is a separate (BigFloat) stage.

/// Residual function: maps `n` real unknowns to `m` real residual components
/// (`m >= n`). Square (`m == n`) or over-determined.
pub type Residual<'a> = dyn Fn(&[f64]) -> Vec<f64> + 'a;

/// Configuration for the damped Gauss–Newton solver / detector.
#[derive(Debug, Clone)]
pub struct NewtonConfig {
    pub max_iters: usize,
    /// Residual ∞-norm below which we declare a root.
    pub root_tol: f64,
    /// Finite-difference step for the Jacobian.
    pub fd_step: f64,
    /// Minimum accepted line-search fraction before giving up a damped step.
    pub min_step_frac: f64,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            root_tol: 1e-12,
            fd_step: 1e-7,
            min_step_frac: 1e-10,
        }
    }
}

/// Outcome of a damped Gauss–Newton solve.
#[derive(Debug, Clone)]
pub struct NewtonSystemResult {
    pub x: Vec<f64>,
    pub residual_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Verdict of the true-root detector.
#[derive(Debug, Clone, PartialEq)]
pub enum RootClass {
    /// Full Gauss–Newton drove the residual to ~0: a genuine isolated root.
    TrueRoot { residual_norm: f64, iterations: usize },
    /// Stationary point of ‖r‖² with residual bounded away from 0: not a root.
    SpuriousMinimum { residual_norm: f64 },
    /// Full Newton diverged from the candidate: not a root.
    Diverged { residual_norm: f64 },
}

impl RootClass {
    pub fn is_true_root(&self) -> bool {
        matches!(self, RootClass::TrueRoot { .. })
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, &x| m.max(x.abs()))
}

/// Forward-difference Jacobian of `f` at `x` (m×n, row-major).
fn jacobian(f: &Residual, x: &[f64], fx: &[f64], h: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = fx.len();
    let mut j = vec![vec![0.0; n]; m];
    let mut xp = x.to_vec();
    for col in 0..n {
        let saved = xp[col];
        xp[col] = saved + h;
        let fp = f(&xp);
        xp[col] = saved;
        for row in 0..m {
            j[row][col] = (fp[row] - fx[row]) / h;
        }
    }
    j
}

/// Solve the Gauss–Newton normal equations `(JᵀJ) dx = -Jᵀr` for the step `dx`.
/// For a square well-conditioned system this equals the plain Newton step.
/// Returns `None` if the normal matrix is singular.
fn gauss_newton_step(j: &[Vec<f64>], r: &[f64]) -> Option<Vec<f64>> {
    let m = j.len();
    if m == 0 {
        return None;
    }
    let n = j[0].len();
    // A = JᵀJ  (n×n),  b = -Jᵀr  (n)
    let mut a = vec![vec![0.0; n]; n];
    let mut b = vec![0.0; n];
    for col in 0..n {
        for k in 0..m {
            b[col] -= j[k][col] * r[k];
        }
        for col2 in 0..n {
            let mut s = 0.0;
            for k in 0..m {
                s += j[k][col] * j[k][col2];
            }
            a[col][col2] = s;
        }
    }
    solve_dense(a, b)
}

/// Gaussian elimination with partial pivoting. Returns `None` if singular.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let diag = a[col][col];
        for j in col..n {
            a[col][j] /= diag;
        }
        b[col] /= diag;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

/// Damped Gauss–Newton with backtracking line search. General workhorse solver.
pub fn newton_system(x0: &[f64], f: &Residual, cfg: &NewtonConfig) -> NewtonSystemResult {
    let mut x = x0.to_vec();
    let mut r = f(&x);
    let mut norm = inf_norm(&r);
    for it in 0..cfg.max_iters {
        if norm < cfg.root_tol {
            return NewtonSystemResult { x, residual_norm: norm, iterations: it, converged: true };
        }
        let j = jacobian(f, &x, &r, cfg.fd_step);
        let Some(dx) = gauss_newton_step(&j, &r) else {
            break;
        };
        // backtracking line search on the residual ∞-norm
        let mut t = 1.0;
        let mut accepted = false;
        while t > cfg.min_step_frac {
            let xt: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + t * d).collect();
            let rt = f(&xt);
            if inf_norm(&rt) < norm {
                x = xt;
                r = rt;
                norm = inf_norm(&r);
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    NewtonSystemResult { x, residual_norm: norm, iterations: cfg.max_iters, converged: norm < cfg.root_tol }
}

/// Levenberg–Marquardt least-squares (trust-region between Gauss–Newton and
/// gradient descent via the damping `λ`). Robust *candidate finder* from a poor
/// start: it will happily settle into a spurious minimum of ‖r‖² — that is
/// exactly why its output must be passed through [`classify_candidate`] before a
/// cover is accepted (the M23-campaign lesson).
pub fn levenberg_marquardt(x0: &[f64], f: &Residual, cfg: &NewtonConfig) -> NewtonSystemResult {
    let mut x = x0.to_vec();
    let mut r = f(&x);
    let mut cost = l2(&r);
    let mut lambda = 1e-3;
    for it in 0..cfg.max_iters {
        if inf_norm(&r) < cfg.root_tol {
            return NewtonSystemResult { x, residual_norm: inf_norm(&r), iterations: it, converged: true };
        }
        let j = jacobian(f, &x, &r, cfg.fd_step);
        let n = x.len();
        let m = r.len();
        // Build JᵀJ + λ·diag(JᵀJ) and -Jᵀr
        let mut a = vec![vec![0.0; n]; n];
        let mut b = vec![0.0; n];
        for col in 0..n {
            for k in 0..m {
                b[col] -= j[k][col] * r[k];
            }
            for col2 in 0..n {
                let mut s = 0.0;
                for k in 0..m {
                    s += j[k][col] * j[k][col2];
                }
                a[col][col2] = s;
            }
        }
        for i in 0..n {
            a[i][i] += lambda * a[i][i].max(1e-12);
        }
        let Some(dx) = solve_dense(a, b) else {
            lambda *= 10.0;
            if lambda > 1e12 { break; }
            continue;
        };
        let xt: Vec<f64> = x.iter().zip(&dx).map(|(a, d)| a + d).collect();
        let rt = f(&xt);
        let ct = l2(&rt);
        if ct < cost {
            x = xt;
            r = rt;
            cost = ct;
            lambda = (lambda * 0.5).max(1e-12);
        } else {
            lambda *= 2.0;
            if lambda > 1e12 { break; }
        }
    }
    NewtonSystemResult { x, residual_norm: inf_norm(&r), iterations: cfg.max_iters, converged: inf_norm(&r) < cfg.root_tol }
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// **True-root detector.** Given a candidate `x0` (typically a low-residual point
/// returned by a least-squares/LM solver), run *undamped* Gauss–Newton and decide
/// whether `x0` sits at a genuine root or at a spurious minimum of ‖r‖².
///
/// - residual reaches `root_tol`  → `TrueRoot`
/// - residual grows past `blowup_factor × max(start, 1)` → `Diverged`
/// - step vanishes (`< min_step_frac`) with residual `> root_tol` → `SpuriousMinimum`
/// - `max_probe` steps exhausted without reaching `root_tol` → `SpuriousMinimum`
pub fn classify_candidate(
    x0: &[f64],
    f: &Residual,
    cfg: &NewtonConfig,
    max_probe: usize,
    blowup_factor: f64,
) -> RootClass {
    let mut x = x0.to_vec();
    let mut r = f(&x);
    let start = inf_norm(&r);
    let blowup = blowup_factor * start.max(1.0);
    for it in 0..max_probe {
        let norm = inf_norm(&r);
        if norm < cfg.root_tol {
            return RootClass::TrueRoot { residual_norm: norm, iterations: it };
        }
        if norm > blowup {
            return RootClass::Diverged { residual_norm: norm };
        }
        let j = jacobian(f, &x, &r, cfg.fd_step);
        let Some(dx) = gauss_newton_step(&j, &r) else {
            return RootClass::SpuriousMinimum { residual_norm: norm };
        };
        // undamped full step — the diagnostic that separates roots from minima
        let step_size = inf_norm(&dx);
        if step_size < cfg.min_step_frac {
            return RootClass::SpuriousMinimum { residual_norm: norm };
        }
        for (xi, di) in x.iter_mut().zip(&dx) {
            *xi += di;
        }
        r = f(&x);
    }
    RootClass::SpuriousMinimum { residual_norm: inf_norm(&r) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // f(x,y) = [x^2 + y^2 - 1, x - y]  — roots at (±1/√2, ±1/√2).
    fn circle_line(v: &[f64]) -> Vec<f64> {
        vec![v[0] * v[0] + v[1] * v[1] - 1.0, v[0] - v[1]]
    }

    #[test]
    fn newton_system_finds_square_root() {
        let cfg = NewtonConfig::default();
        let res = newton_system(&[0.9, 0.4], &circle_line, &cfg);
        assert!(res.converged, "residual {}", res.residual_norm);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((res.x[0] - s).abs() < 1e-8 && (res.x[1] - s).abs() < 1e-8);
    }

    #[test]
    fn detector_accepts_genuine_root() {
        let cfg = NewtonConfig::default();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // start very near a genuine root
        let c = classify_candidate(&[s + 1e-4, s - 1e-4], &circle_line, &cfg, 30, 1e6);
        assert!(c.is_true_root(), "got {:?}", c);
    }

    // Over-determined inconsistent system: [x-1, x-2, x-3]. ‖r‖² minimised at
    // x=2 with residual √2 ≠ 0 — a spurious minimum, not a root.
    fn inconsistent(v: &[f64]) -> Vec<f64> {
        vec![v[0] - 1.0, v[0] - 2.0, v[0] - 3.0]
    }

    #[test]
    fn detector_rejects_spurious_minimum() {
        let cfg = NewtonConfig::default();
        // x = 2 is the least-squares minimiser (∇ = 0) but residual = √2.
        let c = classify_candidate(&[2.0], &inconsistent, &cfg, 30, 1e6);
        assert!(
            matches!(c, RootClass::SpuriousMinimum { .. }),
            "got {:?}",
            c
        );
    }

    // A system with no real solution: x^2 + 1 = 0 — any candidate is a non-root,
    // so the undamped probe must diverge or stall, never report TrueRoot.
    fn no_real_root(v: &[f64]) -> Vec<f64> {
        vec![v[0] * v[0] + 1.0]
    }

    #[test]
    fn detector_rejects_no_real_root_system() {
        let cfg = NewtonConfig::default();
        let c = classify_candidate(&[0.3], &no_real_root, &cfg, 40, 1e3);
        assert!(!c.is_true_root(), "system with no real root wrongly accepted: {:?}", c);
    }
}

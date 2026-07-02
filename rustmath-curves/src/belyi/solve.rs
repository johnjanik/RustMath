//! Native numerical gate for Belyi solve candidates.
//!
//! The `[2,12,5]` cover is solved by a **complex** parameter homotopy (the emitted
//! Julia job, [`crate::belyi::pipeline::assemble_2_12_5_homotopy`]). Before a
//! returned candidate is trusted by the decide chain, it must be screened: a
//! genuine cover is an *isolated root* of the pinned system, whereas a spurious
//! least-squares minimum of `‖F‖²` is not. Feeding the latter into `exactify`
//! would invite a false positive — precisely the failure mode that killed
//! `cover0_lm.npy` in the M23 campaign (residual `6.5e-10` yet **not** a root:
//! undamped Newton escaped to `1.0`).
//!
//! This module wires `rustmath_numerical`'s true-root detector into that gate. It
//! evaluates the integer-coefficient [`PolySystem`] at a *complex* point — the
//! `n` complex coordinates encoded as `2n` reals `[re₀, im₀, re₁, im₁, …]`, each
//! complex equation contributing its real and imaginary parts — and runs
//! [`classify_candidate`]. A square complex system (`n` unknowns, `n` equations,
//! as [`crate::belyi::pinned::pinned_system_2_12_5`]) yields a square `2n×2n` real
//! system, exactly what the detector expects.

use rustmath_numerical::homotopy::{CoordinateReIm, NumericalSolution};
use rustmath_numerical::root_finding::{
    classify_candidate, levenberg_marquardt, NewtonConfig, NewtonSystemResult, RootClass,
};
use rustmath_polynomials::poly_system::PolySystem;

/// Complex number as `(re, im)` in `f64`.
type Cf = (f64, f64);

#[inline]
fn cadd(a: Cf, b: Cf) -> Cf {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn cmul(a: Cf, b: Cf) -> Cf {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Evaluate `system` at the complex point encoded in `x` (interleaved
/// `[re, im, …]`), returning the real residual `[Re F₀, Im F₀, Re F₁, Im F₁, …]`.
///
/// Coordinates beyond `x`'s length are treated as `0` (defensive; the caller is
/// expected to supply `2·num_variables` entries).
pub fn eval_real_residual(system: &PolySystem, x: &[f64]) -> Vec<f64> {
    let n = system.num_variables();
    let z: Vec<Cf> = (0..n)
        .map(|i| (x.get(2 * i).copied().unwrap_or(0.0), x.get(2 * i + 1).copied().unwrap_or(0.0)))
        .collect();

    let mut out = Vec::with_capacity(2 * system.num_equations());
    for poly in system.polynomials() {
        let mut acc: Cf = (0.0, 0.0);
        for (mono, coeff) in poly.terms() {
            // Coefficients of the pinned/parameter systems are small integers
            // (products of monic-form ±1s and multinomial factors); to_f64 fits.
            let c = coeff
                .to_f64()
                .expect("Belyi system coefficient fits in f64");
            let mut term: Cf = (c, 0.0);
            for (&v, &e) in mono.iter_exponents() {
                let zv = z.get(v).copied().unwrap_or((0.0, 0.0));
                for _ in 0..e {
                    term = cmul(term, zv);
                }
            }
            acc = cadd(acc, term);
        }
        out.push(acc.0);
        out.push(acc.1);
    }
    out
}

/// Decode a [`NumericalSolution`] into the interleaved real vector
/// `[re₀, im₀, re₁, im₁, …]`. Unparseable decimals default to `0.0`.
pub fn numerical_solution_to_real(sol: &NumericalSolution) -> Vec<f64> {
    let mut x = Vec::with_capacity(2 * sol.coordinates_re_im_decimal.len());
    for c in &sol.coordinates_re_im_decimal {
        x.push(c.re.trim().parse::<f64>().unwrap_or(0.0));
        x.push(c.im.trim().parse::<f64>().unwrap_or(0.0));
    }
    x
}

/// Re-encode an interleaved real vector `[re₀, im₀, …]` as a [`NumericalSolution`]
/// with the given residual ∞-norm and a `native-refined` path status.
pub fn real_to_numerical_solution(x: &[f64], residual_norm: f64) -> NumericalSolution {
    let coordinates_re_im_decimal = x
        .chunks_exact(2)
        .map(|p| CoordinateReIm {
            re: format!("{}", p[0]),
            im: format!("{}", p[1]),
        })
        .collect();
    NumericalSolution {
        coordinates_re_im_decimal,
        residual_norm_decimal: format!("{residual_norm:e}"),
        path_status: "native-refined".to_string(),
    }
}

/// **The gate.** Classify a numerical candidate against `system` with the
/// true-root detector: `TrueRoot` ⇒ a genuine isolated cover (undamped Newton
/// drives the residual to ~0); `SpuriousMinimum`/`Diverged` ⇒ reject.
///
/// `max_probe` bounds the undamped-Newton probe; `blowup` is the divergence
/// factor (relative to the starting residual).
pub fn gate_numerical_candidate(
    sol: &NumericalSolution,
    system: &PolySystem,
    cfg: &NewtonConfig,
    max_probe: usize,
    blowup: f64,
) -> RootClass {
    let x0 = numerical_solution_to_real(sol);
    let f = |x: &[f64]| eval_real_residual(system, x);
    classify_candidate(&x0, &f, cfg, max_probe, blowup)
}

/// Convenience gate with campaign-tested defaults
/// (`NewtonConfig::default`, 60 probe steps, `1e12` blowup).
pub fn gate_default(sol: &NumericalSolution, system: &PolySystem) -> RootClass {
    gate_numerical_candidate(sol, system, &NewtonConfig::default(), 60, 1e12)
}

/// Polish a candidate with Levenberg–Marquardt against `system` (the native
/// refinement step); pair with [`gate_numerical_candidate`] to confirm the
/// polished point is a genuine root before accepting it.
pub fn refine_candidate(
    sol: &NumericalSolution,
    system: &PolySystem,
    cfg: &NewtonConfig,
) -> NewtonSystemResult {
    let x0 = numerical_solution_to_real(sol);
    let f = |x: &[f64]| eval_real_residual(system, x);
    levenberg_marquardt(&x0, &f, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_numerical::homotopy::CoordinateReIm;

    fn sol_from(pairs: &[(f64, f64)]) -> NumericalSolution {
        NumericalSolution {
            coordinates_re_im_decimal: pairs
                .iter()
                .map(|&(re, im)| CoordinateReIm {
                    re: format!("{re}"),
                    im: format!("{im}"),
                })
                .collect(),
            residual_norm_decimal: "0.0".into(),
            path_status: "candidate".into(),
        }
    }

    // z^2 + 1 = 0  (one complex variable, one complex equation): roots ±i.
    fn z_sq_plus_1() -> PolySystem {
        PolySystem::from_terms(1, &[vec![(vec![2], 1), (vec![0], 1)]])
    }

    // Inconsistent complex system in one variable: {z - 1 = 0, z - 2 = 0}.
    fn inconsistent() -> PolySystem {
        PolySystem::from_terms(
            1,
            &[
                vec![(vec![1], 1), (vec![0], -1)],
                vec![(vec![1], 1), (vec![0], -2)],
            ],
        )
    }

    #[test]
    fn residual_of_i_is_zero_for_z_sq_plus_1() {
        // z = i encoded as (re=0, im=1); z^2 + 1 = 0.
        let r = eval_real_residual(&z_sq_plus_1(), &[0.0, 1.0]);
        assert!(r.iter().all(|v| v.abs() < 1e-12), "residual {r:?}");
    }

    #[test]
    fn gate_accepts_genuine_complex_root() {
        // Start slightly off i; the detector must confirm a true root.
        let sol = sol_from(&[(0.01, 0.98)]);
        let class = gate_default(&sol, &z_sq_plus_1());
        assert!(class.is_true_root(), "genuine root rejected: {class:?}");
    }

    #[test]
    fn gate_rejects_spurious_minimum() {
        // Least-squares minimum of {z-1, z-2} sits near z = 1.5, residual != 0.
        let sol = sol_from(&[(1.5, 0.0)]);
        let class = gate_default(&sol, &inconsistent());
        assert!(!class.is_true_root(), "spurious minimum accepted: {class:?}");
    }

    #[test]
    fn refine_then_gate_confirms_root() {
        let sys = z_sq_plus_1();
        let cfg = NewtonConfig::default();
        let refined = refine_candidate(&sol_from(&[(0.2, 0.7)]), &sys, &cfg);
        let sol2 = real_to_numerical_solution(&refined.x, refined.residual_norm);
        assert!(gate_default(&sol2, &sys).is_true_root());
    }
}

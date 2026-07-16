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

use super::mp_svd::JacobiSvdOptions;
use rustmath_numerical::homotopy::{CoordinateReIm, NumericalSolution};
use rustmath_numerical::root_finding::{
    classify_candidate, levenberg_marquardt, NewtonConfig, NewtonSystemResult, RootClass,
};
use rustmath_polynomials::poly_system::PolySystem;

// ---------------------------------------------------------------------------
// SolveParams: the N-vs-precision binding for the hp solve spine (item A7)
// ---------------------------------------------------------------------------

/// Decimal digits representable at `prec_bits` of mpfr precision:
/// `D = floor(prec_bits · log10 2)`. prec = 256 → D = 77.
pub fn decimal_capacity(prec_bits: u32) -> usize {
    (prec_bits as f64 * std::f64::consts::LOG10_2).floor() as usize
}

/// Jacobi off-diagonal convergence tolerance derived from `prec_bits`:
/// `1e-(D−7)` — 7 guard digits above the representation floor (sweeps past that
/// only churn rounding noise). prec = 256 → `"1e-70"`, the trusted (5,3,3) literal.
pub fn jacobi_tol_decimal(prec_bits: u32) -> String {
    format!("1e-{}", decimal_capacity(prec_bits).saturating_sub(7))
}

/// σ-cluster tolerance derived from `prec_bits`: `1e-(ceil(D/2)+1)` — singular
/// values agreeing to half the working precision (plus one guard decade) are a
/// cluster, whose individual vectors are not canonical (use the subspace).
/// prec = 256 → `"1e-40"`, the trusted (5,3,3) literal.
pub fn cluster_tol_decimal(prec_bits: u32) -> String {
    format!("1e-{}", decimal_capacity(prec_bits).div_ceil(2) + 1)
}

/// Why a [`SolveParams`] request was rejected. Every variant states the exact
/// inequality that failed — an under-precisioned run would otherwise return
/// coefficients dominated by amplified rounding noise while *looking* converged.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// `digits < 4`: the derived threshold `1e-(ceil(digits/2)+1)` no longer
    /// separates the kernel σ (~`1e-digits`) from the O(1) rank σ.
    DigitsTooSmall { digits: usize, min: usize },
    /// `decimal_capacity(prec_bits) < 2·digits + ceil(log10(N+1)) + 8`.
    InsufficientPrecision {
        prec_bits: u32,
        needed_bits: u32,
        digits: usize,
        big_n: usize,
    },
    /// `ρ^N > 10^-digits`: the series truncation floor sits above the requested
    /// accuracy (`achievable_digits = N·log10(1/ρ) < digits`).
    TruncationTooCoarse {
        big_n: usize,
        rho: f64,
        achievable_digits: f64,
        requested_digits: usize,
    },
}

/// Coupled parameters for the §4/§5 hp solve chain, binding the series
/// truncation `N`, the working precision, and the SVD tolerances so they are
/// mutually consistent instead of free-floating literals.
///
/// Derivation (generalizing the trusted (5,3,3) test literals
/// `prec = 256`, `N = 48`, `"1e-8"`, `"1e-70"`, `"1e-40"`):
///
/// * `D = floor(prec · log10 2)` decimal digits of working precision (256 → 77).
/// * `tol_decimal = 1e-(D−7)`: Jacobi sweep convergence, 7 guard digits above
///   the rounding floor (77 → `1e-70`).
/// * `cluster_decimal = 1e-(ceil(D/2)+1)`: σ clustered at half working
///   precision (77 → `1e-40`).
/// * `threshold_decimal = 1e-(ceil(digits/2)+1)`: the kernel/rank separator.
///   The kernel σ sit at the truncation floor ~`10^-digits`, the rank σ are
///   O(1); the geometric mean pushed one decade toward the floor separates them
///   (digits = 13 → `1e-8`; for (5,3,3), ρ^48 ≈ 5·10⁻¹⁴, i.e. digits = 13).
/// * Accepted only if `D ≥ 2·digits + ceil(log10(N+1)) + 8`: the un-scaling
///   `b_n = ρ^{-n} y_n` in `recover_forms` amplifies representation noise
///   `10^-D` by up to `ρ^{-N} ≈ 10^digits`, so trusting the result at the
///   truncation floor `10^-digits` needs `D ≥ 2·digits`, plus `ceil(log10(N+1))`
///   for length-(N+1) accumulations and 8 guard digits.
///
/// `digits` is the requested decimal accuracy: the caller must also choose `N`
/// large enough that `ρ^N ≤ 10^-digits`. ρ is only known once the coset graph is
/// compactified, so that binding is checked separately by [`Self::check_rho`].
#[derive(Debug, Clone, PartialEq)]
pub struct SolveParams {
    /// mpfr working precision in bits.
    pub prec_bits: u32,
    /// Series truncation N (matrix dim = N + 1).
    pub big_n: usize,
    /// Requested decimal accuracy (the truncation floor `10^-digits`).
    pub digits: usize,
    /// Kernel/rank σ separation threshold (derived, see type docs).
    pub threshold_decimal: String,
    /// Jacobi off-diagonal convergence tolerance (derived).
    pub tol_decimal: String,
    /// σ-cluster tolerance (derived).
    pub cluster_decimal: String,
    /// Jacobi sweep cap (the (5,3,3)-tested default: 80).
    pub max_sweeps: usize,
    /// Extra rows over columns in the `solve_belyi_map` system
    /// (`nrows = ncols + extra_rows`; the (5,3,3)-tested default: 6).
    pub extra_rows: usize,
}

impl SolveParams {
    /// Jacobi sweep cap used by every trusted (5,3,3) run.
    pub const DEFAULT_MAX_SWEEPS: usize = 80;
    /// Row surplus used by every trusted (5,3,3) run of `solve_belyi_map`.
    pub const DEFAULT_EXTRA_ROWS: usize = 6;

    /// Bind (`prec_bits`, `N`, `digits`) as documented on the type, rejecting an
    /// under-precisioned or under-separated request with the exact failed bound.
    pub fn new(prec_bits: u32, n: usize, digits: usize) -> Result<SolveParams, ParamError> {
        const MIN_DIGITS: usize = 4;
        if digits < MIN_DIGITS {
            return Err(ParamError::DigitsTooSmall { digits, min: MIN_DIGITS });
        }
        let cap = decimal_capacity(prec_bits);
        let acc_digits = ((n + 1) as f64).log10().ceil() as usize;
        let needed_cap = 2 * digits + acc_digits + 8;
        if cap < needed_cap {
            let needed_bits = (needed_cap as f64 / std::f64::consts::LOG10_2).ceil() as u32;
            return Err(ParamError::InsufficientPrecision {
                prec_bits,
                needed_bits,
                digits,
                big_n: n,
            });
        }
        Ok(SolveParams {
            prec_bits,
            big_n: n,
            digits,
            threshold_decimal: format!("1e-{}", digits.div_ceil(2) + 1),
            tol_decimal: jacobi_tol_decimal(prec_bits),
            cluster_decimal: cluster_tol_decimal(prec_bits),
            max_sweeps: Self::DEFAULT_MAX_SWEEPS,
            extra_rows: Self::DEFAULT_EXTRA_ROWS,
        })
    }

    /// The N-vs-ρ half of the binding: `ρ^N ≤ 10^-digits`, i.e.
    /// `N·log10(1/ρ) ≥ digits`. Call once the compactified domain radius ρ is
    /// known; a failure means the truncation floor is above the requested
    /// accuracy and the run must not proceed.
    pub fn check_rho(&self, rho: f64) -> Result<(), ParamError> {
        if !(rho > 0.0 && rho < 1.0) {
            return Err(ParamError::TruncationTooCoarse {
                big_n: self.big_n,
                rho,
                achievable_digits: 0.0,
                requested_digits: self.digits,
            });
        }
        let achievable = self.big_n as f64 * (1.0 / rho).log10();
        if achievable < self.digits as f64 {
            return Err(ParamError::TruncationTooCoarse {
                big_n: self.big_n,
                rho,
                achievable_digits: achievable,
                requested_digits: self.digits,
            });
        }
        Ok(())
    }

    /// The [`JacobiSvdOptions`] this binding prescribes.
    pub fn svd_options(&self) -> JacobiSvdOptions {
        JacobiSvdOptions::new(
            self.prec_bits,
            self.max_sweeps,
            &self.tol_decimal,
            &self.cluster_decimal,
        )
    }
}

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

    // The binding reproduces the trusted (5,3,3) literals: prec = 256, N = 48
    // (ρ ≈ 0.5289, ρ^48 ≈ 5e-14 ⇒ digits = 13) must derive exactly the
    // threshold/tol/cluster strings the hand-tuned tests have always used.
    #[test]
    fn solve_params_reproduce_5_3_3_literals() {
        let sp = SolveParams::new(256, 48, 13).expect("(5,3,3) config must be accepted");
        assert_eq!(sp.threshold_decimal, "1e-8");
        assert_eq!(sp.tol_decimal, "1e-70");
        assert_eq!(sp.cluster_decimal, "1e-40");
        assert_eq!(sp.max_sweeps, 80);
        assert_eq!(sp.extra_rows, 6);
        // measured (5,3,3) z_a-chart radius: 48·log10(1/0.528936) ≈ 13.28 ≥ 13.
        sp.check_rho(0.528936).expect("N = 48 covers 13 digits at ρ ≈ 0.5289");
        // one digit more is NOT covered by N = 48 at that ρ (13.28 < 14):
        let sp14 = SolveParams::new(256, 48, 14).expect("precision itself is sufficient");
        assert!(matches!(
            sp14.check_rho(0.528936),
            Err(ParamError::TruncationTooCoarse { .. })
        ));
        // ρ ≥ 1 (non-contracting chart) can never satisfy the binding.
        assert!(sp.check_rho(1.0).is_err());
    }

    // Under-precisioned requests err with the exact failed bound: digits = 13,
    // N = 48 needs D ≥ 2·13 + ceil(log10 49) + 8 = 36 decimal digits ⇒ 120 bits.
    #[test]
    fn solve_params_reject_under_precision() {
        match SolveParams::new(100, 48, 13) {
            Err(ParamError::InsufficientPrecision { prec_bits, needed_bits, .. }) => {
                assert_eq!(prec_bits, 100);
                assert_eq!(needed_bits, 120);
            }
            other => panic!("expected InsufficientPrecision, got {other:?}"),
        }
        // exactly at the bound is accepted
        assert!(SolveParams::new(120, 48, 13).is_ok());
        // digits below the separability floor are rejected outright
        assert!(matches!(
            SolveParams::new(256, 48, 3),
            Err(ParamError::DigitsTooSmall { .. })
        ));
    }

    // The [2,12,5] production binding: prec = 400 bits, N = 3000, digits = 12 is
    // consistent (D = 120 ≥ 36) and the measured z_a-chart ρ ≈ 0.9906 gives
    // 3000·log10(1/0.9906) ≈ 12.3 ≥ 12, while N = 2500 would NOT reach 12 digits.
    #[test]
    fn solve_params_bind_2_12_5_production() {
        let sp = SolveParams::new(400, 3000, 12).expect("production binding");
        sp.check_rho(0.9906).expect("N = 3000 covers 12 digits at ρ ≈ 0.9906");
        let sp_short = SolveParams::new(400, 2500, 12).unwrap();
        assert!(matches!(
            sp_short.check_rho(0.9906),
            Err(ParamError::TruncationTooCoarse { .. })
        ));
    }
}

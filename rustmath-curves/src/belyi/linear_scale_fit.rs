//! Linear least-squares fit of the scalars `(λ, c)` before Newton.
//!
//! Given approximate factor roots, the polynomial identity
//! `A²B − λ R⁵S − c U¹² = 0` is *linear* in `(λ, c)` once the factors are fixed.
//! Solving the 2×2 normal equations for the best `(λ, c)` gives a much better
//! Newton seed than arbitrary defaults. Adapted from the P2 review note.

use num_complex::Complex64;

use super::factorized_residual::FactorizedPolys;

#[derive(Debug, Clone)]
pub struct ScaleFitReport {
    pub lambda: Complex64,
    pub c: Complex64,
    pub residual_norm: f64,
}

fn hermitian_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum()
}

/// Least-squares solve of `P0 = λ·Pinf + c·P1` over ℂ, returning `(λ, c)` and the
/// residual norm of `P0 − λ·Pinf − c·P1`.
pub fn fit_lambda_c_from_vectors(
    p0: &[Complex64],
    p_inf: &[Complex64],
    p_one: &[Complex64],
) -> ScaleFitReport {
    // Normal equations  M · [λ, c]ᵀ = b, with M Hermitian:
    //   [<pinf,pinf> <pinf,p1>] [λ]   [<pinf,p0>]
    //   [<p1,pinf>   <p1,p1>  ] [c] = [<p1,p0>  ]
    let a00 = hermitian_dot(p_inf, p_inf);
    let a01 = hermitian_dot(p_inf, p_one);
    let a10 = hermitian_dot(p_one, p_inf);
    let a11 = hermitian_dot(p_one, p_one);
    let b0 = hermitian_dot(p_inf, p0);
    let b1 = hermitian_dot(p_one, p0);

    let det = a00 * a11 - a01 * a10;
    let (lambda, c) = if det.norm() > 1e-30 {
        (
            (b0 * a11 - a01 * b1) / det,
            (a00 * b1 - b0 * a10) / det,
        )
    } else {
        (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0))
    };

    // Residual P0 − λ·Pinf − c·P1.
    let n = p0.len().max(p_inf.len()).max(p_one.len());
    let mut res_sq = 0.0;
    for i in 0..n {
        let z0 = p0.get(i).copied().unwrap_or_default();
        let zi = p_inf.get(i).copied().unwrap_or_default();
        let z1 = p_one.get(i).copied().unwrap_or_default();
        res_sq += (z0 - lambda * zi - c * z1).norm_sqr();
    }

    ScaleFitReport {
        lambda,
        c,
        residual_norm: res_sq.sqrt(),
    }
}

/// Fit `(λ, c)` for a factorization by expanding the three degree-24 parts.
pub fn fit_lambda_c(polys: &FactorizedPolys) -> ScaleFitReport {
    let p0 = polys.p_zero().pad_to(25);
    let p_inf = polys.p_inf().pad_to(25);
    let p_one = polys.p_one().pad_to(25);
    fit_lambda_c_from_vectors(&p0, &p_inf, &p_one)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn recovers_exact_scalars_when_consistent() {
        // Construct P0 = 2·Pinf + 3·P1 exactly ⇒ fit returns (2, 3), residual 0.
        let p_inf = vec![c(1.0, 0.0), c(0.0, 1.0), c(2.0, -1.0), c(-1.0, 0.5)];
        let p_one = vec![c(0.5, 0.0), c(1.0, 1.0), c(-1.0, 0.0), c(0.0, 2.0)];
        let lam = c(2.0, 0.0);
        let cc = c(3.0, 0.0);
        let p0: Vec<Complex64> = (0..4).map(|i| lam * p_inf[i] + cc * p_one[i]).collect();

        let fit = fit_lambda_c_from_vectors(&p0, &p_inf, &p_one);
        assert!((fit.lambda - lam).norm() < 1e-10, "λ = {}", fit.lambda);
        assert!((fit.c - cc).norm() < 1e-10, "c = {}", fit.c);
        assert!(fit.residual_norm < 1e-10, "residual {}", fit.residual_norm);
    }

    #[test]
    fn recovers_complex_scalars() {
        let p_inf = vec![c(1.0, 0.0), c(0.3, -0.7), c(2.0, 1.0)];
        let p_one = vec![c(0.0, 1.0), c(1.0, 0.2), c(-0.5, 0.5)];
        let lam = c(1.5, -0.5);
        let cc = c(-0.25, 0.75);
        let p0: Vec<Complex64> = (0..3).map(|i| lam * p_inf[i] + cc * p_one[i]).collect();
        let fit = fit_lambda_c_from_vectors(&p0, &p_inf, &p_one);
        assert!((fit.lambda - lam).norm() < 1e-10);
        assert!((fit.c - cc).norm() < 1e-10);
    }

    #[test]
    fn fit_reduces_residual_for_factorized_polys() {
        use super::super::factorized_residual::FactorizedRoots;
        // Arbitrary factors; the LS fit residual must be <= the residual at (1,1).
        let mut roots = FactorizedRoots {
            roots_a: (0..8).map(|k| c(0.1 * k as f64, 0.0)).collect(),
            roots_b: (0..8).map(|k| c(0.0, 0.2 * k as f64 + 0.3)).collect(),
            roots_r: (0..4).map(|k| c(-0.5 * k as f64 - 1.0, 0.1)).collect(),
            roots_s: (0..4).map(|k| c(0.2, -0.3 * k as f64 - 1.0)).collect(),
            roots_u: vec![c(0.4, 0.6), c(-0.6, -0.4)],
            lambda: c(1.0, 0.0),
            c: c(1.0, 0.0),
        };
        let base = roots.residual_norm();
        let fit = fit_lambda_c(&roots.to_polys());
        roots.lambda = fit.lambda;
        roots.c = fit.c;
        assert!(
            roots.residual_norm() <= base + 1e-12,
            "fit did not reduce residual: {} vs {}",
            roots.residual_norm(),
            base
        );
        assert!((roots.residual_norm() - fit.residual_norm).abs() < 1e-6);
    }
}

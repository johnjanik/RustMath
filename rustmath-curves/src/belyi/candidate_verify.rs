//! Candidate Belyi-map verifier — the "artifact hunt" machinery.
//!
//! Given a candidate three-point cover from *any* source (literature, a
//! collaborator, a modular-functions computation à la Monien/Vatuzov, or our own
//! packing→Newton pipeline), certify it by checking the defining polynomial
//! identity and the ramification degrees.
//!
//! A degree-`d` Belyi map `φ = N/D` with `φ − 1 = O/D` satisfies, for suitable
//! scalars, the single polynomial identity
//!
//! ```text
//!   P_zero − λ · P_pole − c · P_one = 0
//! ```
//!
//! where `P_zero` (zeros of `φ`), `P_pole` (poles), `P_one` (zeros of `φ − 1`) are
//! the three degree-`d` products of ramification factors. This covers both:
//! * our `[2,12,5]`:  `A²B − λ R⁵S − c U¹² = 0`;
//! * Vatuzov's M₂₄ `(2¹²|3⁶1⁶|21,3)`:  `P₃³P₁ − c·z³ − 1·Q₂² = 0`.
//!
//! The check is numeric (complex `f64`) so it works over any field via an
//! embedding; exact recognition happens elsewhere.

use super::factorized_residual::PolyC;
use num_complex::Complex64;

/// A verification verdict for a candidate three-point cover.
#[derive(Debug, Clone)]
pub struct CandidateReport {
    pub degree: usize,
    pub deg_zero: usize,
    pub deg_pole: usize,
    pub deg_one: usize,
    /// `‖P_zero − λ·P_pole − c·P_one‖` (coefficient 2-norm).
    pub identity_residual: f64,
    /// All three parts share the map degree.
    pub degrees_consistent: bool,
    /// Identity residual within tolerance.
    pub identity_holds: bool,
}

impl CandidateReport {
    pub fn passes(&self) -> bool {
        self.degrees_consistent && self.identity_holds
    }
}

/// Verify the Belyi identity `P_zero − λ·P_pole − c·P_one = 0` numerically.
pub fn verify_identity(
    p_zero: &PolyC,
    p_pole: &PolyC,
    p_one: &PolyC,
    lambda: Complex64,
    c: Complex64,
    tol: f64,
) -> CandidateReport {
    let residual = p_zero
        .add_scaled(-lambda, p_pole)
        .add_scaled(-c, p_one);
    let identity_residual = residual.c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

    let dz = p_zero.degree();
    let dp = p_pole.degree();
    let d1 = p_one.degree();
    // The map degree is the common degree of the three parts (poles may be lower
    // when φ has a pole at ∞; then deg_pole < degree and the difference is the
    // order of the pole at ∞ — still consistent as long as zero/one parts match).
    let degree = dz.max(dp).max(d1);
    let degrees_consistent = dz == degree && d1 == degree && dp <= degree;

    CandidateReport {
        degree,
        deg_zero: dz,
        deg_pole: dp,
        deg_one: d1,
        identity_residual,
        degrees_consistent,
        identity_holds: identity_residual < tol,
    }
}

/// Convenience: build the three parts from factor `(polynomial, multiplicity)`
/// lists and verify. `zero = ∏ zᵢ^{mᵢ}` etc.
pub fn verify_factored(
    zero_factors: &[(PolyC, usize)],
    pole_factors: &[(PolyC, usize)],
    one_factors: &[(PolyC, usize)],
    lambda: Complex64,
    c: Complex64,
    tol: f64,
) -> CandidateReport {
    let build = |fs: &[(PolyC, usize)]| {
        let mut p = PolyC::one();
        for (f, m) in fs {
            if *m > 0 {
                p = p.mul(&f.pow(*m));
            }
        }
        p
    };
    verify_identity(&build(zero_factors), &build(pole_factors), &build(one_factors), lambda, c, tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// The power map φ = zⁿ is Belyi: over 0 and ∞ fully ramified, over 1 the n-th
    /// roots of unity. Identity: zⁿ − 1·1 − 1·(zⁿ − 1) = 0.
    #[test]
    fn power_map_is_certified() {
        let n = 5;
        let z_n = PolyC::from_roots_monic(&vec![c(0.0, 0.0); n]); // z^5
        let pole = PolyC::one(); // pole only at ∞
        let one = {
            // z^5 - 1
            let mut p = z_n.clone();
            p.c[0] -= c(1.0, 0.0);
            p
        };
        let rep = verify_identity(&z_n, &pole, &one, c(1.0, 0.0), c(1.0, 0.0), 1e-12);
        assert!(rep.identity_holds, "residual {}", rep.identity_residual);
        assert_eq!(rep.deg_zero, 5);
        assert_eq!(rep.deg_one, 5);
        assert!(rep.passes());
    }

    /// A wrong scalar must fail the identity (guards against a vacuous verifier).
    #[test]
    fn wrong_scalar_is_rejected() {
        let n = 5;
        let z_n = PolyC::from_roots_monic(&vec![c(0.0, 0.0); n]);
        let pole = PolyC::one();
        let mut one = z_n.clone();
        one.c[0] -= c(1.0, 0.0);
        let rep = verify_identity(&z_n, &pole, &one, c(2.0, 0.0), c(1.0, 0.0), 1e-12);
        assert!(!rep.identity_holds, "should reject wrong λ");
        assert!(!rep.passes());
    }

    /// Factored entry point on the Chebyshev-style identity T with a squared factor.
    /// φ = (x²)·1 style: verify build-from-multiplicities matches direct.
    #[test]
    fn factored_matches_direct() {
        // zero = (z-1)² (z-2), pole = 1, one = zero - 1·pole? construct consistent:
        // Use φ = z³: zero=z³ (mult), pole=1, one=z³-1.
        let z = PolyC {
            c: vec![c(0.0, 0.0), c(1.0, 0.0)],
        };
        let zero = z.pow(3);
        let pole = PolyC::one();
        let mut one = zero.clone();
        one.c[0] -= c(1.0, 0.0);
        let direct = verify_identity(&zero, &pole, &one, c(1.0, 0.0), c(1.0, 0.0), 1e-12);
        let fact = verify_factored(
            &[(z.clone(), 3)],
            &[(PolyC::one(), 1)],
            &[(one.clone(), 1)],
            c(1.0, 0.0),
            c(1.0, 0.0),
            1e-12,
        );
        assert!(direct.identity_holds && fact.identity_holds);
        assert!((direct.identity_residual - fact.identity_residual).abs() < 1e-12);
    }
}

//! Exactification: turn an untrusted numerical candidate into certified exact
//! data.
//!
//! Ported from `dessin_engine/src/exactification.rs::exactify`
//! (`/home/john/inverse_galois/M23/dessin_engine`). The `recognize_complex_algebraic`
//! LLL search itself now lives in `rustmath-numberfields` (Wave 1); this module
//! is the *orchestration*: for each coordinate of a homotopy candidate it calls
//! [`recognize_complex_algebraic`](rustmath_numberfields::recognize::recognize_complex_algebraic),
//! then **certifies** — rational coordinates by exact back-substitution into the
//! original [`PolySystem`] via
//! [`is_exact_solution`](rustmath_polynomials::poly_system::PolySystem::is_exact_solution);
//! algebraic coordinates are returned as per-coordinate minimal polynomials (the
//! common-field embedding + exact substitution over `L` is the S2/S3 follow-up).
//!
//! Numerical recognition is heuristic; the *certificate* is the exact check.
//! Anything unrecognized or unverified is reported as such, never as a result.

use crate::homotopy::NumericalSolution;
use num_bigint::BigInt;
use rustmath_integers::Integer;
use rustmath_numberfields::recognize::recognize_complex_algebraic;
use rustmath_polynomials::poly_system::PolySystem;
use rustmath_rationals::Rational;

/// The result of exactifying one numerical candidate against the exact system.
///
/// Same discipline as the reference implementation: only `CertifiedRational` is a
/// decided, exactly-verified solution; everything else is honest about *why* the
/// candidate did not resolve to a certified rational point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactifyOutcome {
    /// All coordinates rational and the point is an exact zero of the system.
    CertifiedRational(Vec<Rational>),
    /// Coordinates recognized as algebraic (per-coordinate minimal polynomials,
    /// integer coeffs ascending); the common-field embedding + exact substitution
    /// over `L` is the follow-up.
    AlgebraicCoordinates(Vec<Vec<BigInt>>),
    /// A coordinate could not be recognized within the degree bound.
    RecognitionFailed,
    /// Coordinates rational but the point is not an exact zero (a spurious path).
    SubstitutionFailed,
}

/// Exactify a numerical candidate against the exact system.
///
/// `max_deg` bounds the algebraic degree searched per coordinate. Only when
/// every coordinate is degree-1 (rational) and the assembled point is an exact
/// zero do we return [`ExactifyOutcome::CertifiedRational`].
pub fn exactify(sol: &NumericalSolution, system: &PolySystem, max_deg: usize) -> ExactifyOutcome {
    let mut minpolys: Vec<Vec<BigInt>> = Vec::with_capacity(sol.coordinates_re_im_decimal.len());
    for coord in &sol.coordinates_re_im_decimal {
        let re: f64 = coord.re.parse().unwrap_or(f64::NAN);
        let im: f64 = coord.im.parse().unwrap_or(f64::NAN);
        match recognize_complex_algebraic(re, im, max_deg) {
            Some(p) => minpolys.push(p),
            None => return ExactifyOutcome::RecognitionFailed,
        }
    }

    // All degree-1 ⇒ rational coordinates: c0 + c1·x = 0 ⇒ x = -c0/c1.
    if minpolys.iter().all(|p| p.len() == 2) {
        let pt: Vec<Rational> = minpolys
            .iter()
            .map(|p| {
                let num = Integer::from(-p[0].clone());
                let den = Integer::from(p[1].clone());
                Rational::new(num, den).expect("degree-1 leading coeff is nonzero")
            })
            .collect();
        return if system.is_exact_solution(&pt) {
            ExactifyOutcome::CertifiedRational(pt)
        } else {
            ExactifyOutcome::SubstitutionFailed
        };
    }

    ExactifyOutcome::AlgebraicCoordinates(minpolys)
}

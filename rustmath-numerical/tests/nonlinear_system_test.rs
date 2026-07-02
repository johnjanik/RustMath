//! Integration tests for the multivariate Newton + true-root detector
//! (`root_finding::nonlinear_system`). Ported from the M23 Belyi campaign; see
//! `RustMath/M23_BELYI_CONIC_PORT_SPEC.md` §P1. Kept as an integration test so it
//! compiles against the (green) library, independent of unrelated in-crate
//! test-code build debt.

use rustmath_numerical::root_finding::{
    classify_candidate, levenberg_marquardt, newton_system, NewtonConfig, RootClass,
};

// f(x,y) = [x^2 + y^2 - 1, x - y]  — roots at (±1/√2, ±1/√2).
fn circle_line(v: &[f64]) -> Vec<f64> {
    vec![v[0] * v[0] + v[1] * v[1] - 1.0, v[0] - v[1]]
}

// Over-determined inconsistent: [x-1, x-2, x-3]; ‖r‖² minimised at x=2, residual √2.
fn inconsistent(v: &[f64]) -> Vec<f64> {
    vec![v[0] - 1.0, v[0] - 2.0, v[0] - 3.0]
}

// A system with no real solution: x^2 + 1 = 0. Any candidate is a non-root.
fn no_real_root(v: &[f64]) -> Vec<f64> {
    vec![v[0] * v[0] + 1.0]
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
    let c = classify_candidate(&[s + 1e-4, s - 1e-4], &circle_line, &cfg, 30, 1e6);
    assert!(c.is_true_root(), "genuine root wrongly rejected: {:?}", c);
}

#[test]
fn detector_rejects_spurious_minimum() {
    let cfg = NewtonConfig::default();
    let c = classify_candidate(&[2.0], &inconsistent, &cfg, 30, 1e6);
    assert!(
        matches!(c, RootClass::SpuriousMinimum { .. }),
        "spurious least-squares minimum wrongly accepted: {:?}",
        c
    );
}

#[test]
fn lm_then_classify_confirms_genuine_root() {
    // The P1 construction workflow: LM finds a low-residual candidate from a poor
    // start, then the detector confirms it is a genuine root before acceptance.
    let cfg = NewtonConfig::default();
    let found = levenberg_marquardt(&[2.0, -1.5], &circle_line, &cfg);
    assert!(found.residual_norm < 1e-6, "LM residual {}", found.residual_norm);
    let c = classify_candidate(&found.x, &circle_line, &cfg, 30, 1e6);
    assert!(c.is_true_root(), "LM candidate wrongly rejected: {:?}", c);
}

#[test]
fn lm_then_classify_rejects_inconsistent_minimum() {
    // LM settles into the spurious least-squares minimum of an inconsistent
    // system; the detector must reject it (the M23-campaign guardrail).
    let cfg = NewtonConfig::default();
    let found = levenberg_marquardt(&[10.0], &inconsistent, &cfg);
    let c = classify_candidate(&found.x, &inconsistent, &cfg, 30, 1e6);
    assert!(!c.is_true_root(), "LM spurious minimum wrongly accepted: {:?}", c);
}

#[test]
fn detector_rejects_no_real_root_system() {
    let cfg = NewtonConfig::default();
    let c = classify_candidate(&[0.3], &no_real_root, &cfg, 40, 1e3);
    // No real root exists: must be SpuriousMinimum or Diverged, never TrueRoot.
    assert!(
        !c.is_true_root(),
        "system with no real root wrongly accepted: {:?}",
        c
    );
}

//! Known-answer Stauduhar descent tests (degrees 3, 4, 5).
//!
//! Each polynomial's Galois group is classical and verifiable by hand; the
//! descent must return the exact `nTt` label.

use rustmath_galois::descent::{galois_group, Config};
use rustmath_integers::Integer;

fn iz(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

fn run(coeffs: &[i64]) -> (Option<String>, usize) {
    let cfg = Config::default();
    let res = galois_group(&iz(coeffs), &cfg).expect("descent succeeded");
    (res.label, res.order)
}

// ---- degree 3 ----

#[test]
fn cubic_x3_minus_2_is_s3() {
    // x³ − 2: disc = −108 (not square) ⇒ S₃ = 3T2.
    let (label, order) = run(&[-2, 0, 0, 1]);
    assert_eq!(order, 6);
    assert_eq!(label.as_deref(), Some("3T2"));
}

#[test]
fn cubic_x3_minus_3x_minus_1_is_c3() {
    // x³ − 3x − 1: disc = 81 = 9² (square) ⇒ C₃ = 3T1.
    let (label, order) = run(&[-1, -3, 0, 1]);
    assert_eq!(order, 3);
    assert_eq!(label.as_deref(), Some("3T1"));
}

// ---- degree 4 ----

#[test]
fn quartic_x4_plus_1_is_v4() {
    // x⁴ + 1 (Φ₈): Galois group V₄ = 4T2 (order 4).
    let (label, order) = run(&[1, 0, 0, 0, 1]);
    assert_eq!(order, 4);
    assert_eq!(label.as_deref(), Some("4T2"));
}

#[test]
fn quartic_x4_minus_2_is_d4() {
    // x⁴ − 2: Galois group D₄ = 4T3 (order 8).
    let (label, order) = run(&[-2, 0, 0, 0, 1]);
    assert_eq!(order, 8);
    assert_eq!(label.as_deref(), Some("4T3"));
}

#[test]
fn quartic_x4_plus_x_plus_1_is_s4() {
    // x⁴ + x + 1: Galois group S₄ = 4T5 (order 24).
    let (label, order) = run(&[1, 1, 0, 0, 1]);
    assert_eq!(order, 24);
    assert_eq!(label.as_deref(), Some("4T5"));
}

#[test]
fn quartic_cyclic_c4() {
    // x⁴ + x³ + x² + x + 1 is Φ₅, degree 4, Galois group C₄ = 4T1 (order 4).
    let (label, order) = run(&[1, 1, 1, 1, 1]);
    assert_eq!(order, 4);
    assert_eq!(label.as_deref(), Some("4T1"));
}

#[test]
fn quartic_a4() {
    // x⁴ + 8x + 12: classical A₄ quartic (disc a perfect square, group 4T4).
    let (label, order) = run(&[12, 8, 0, 0, 1]);
    assert_eq!(order, 12);
    assert_eq!(label.as_deref(), Some("4T4"));
}

// ---- degree 5 ----

#[test]
fn quintic_x5_minus_2_is_f20() {
    // x⁵ − 2: Galois group F₂₀ = 5T3 (order 20).
    let (label, order) = run(&[-2, 0, 0, 0, 0, 1]);
    assert_eq!(order, 20);
    assert_eq!(label.as_deref(), Some("5T3"));
}

#[test]
fn quintic_x5_minus_x_minus_1_is_s5() {
    // x⁵ − x − 1: Galois group S₅ = 5T5 (order 120).
    let (label, order) = run(&[-1, -1, 0, 0, 0, 1]);
    assert_eq!(order, 120);
    assert_eq!(label.as_deref(), Some("5T5"));
}

#[test]
fn quintic_cyclic_c5() {
    // Φ₁₁ restricted / the degree-5 cyclic field x⁵+x⁴−4x³−3x²+3x+1
    // (the real subfield of Q(ζ₁₁)) has Galois group C₅ = 5T1.
    let (label, order) = run(&[1, 3, -3, -4, 1, 1]);
    assert_eq!(order, 5);
    assert_eq!(label.as_deref(), Some("5T1"));
}

#[test]
fn quintic_d5() {
    // x⁵ − 5x + 12 has Galois group D₅ = 5T2 (order 10).
    let (label, order) = run(&[12, -5, 0, 0, 0, 1]);
    assert_eq!(order, 10);
    assert_eq!(label.as_deref(), Some("5T2"));
}

// ---- descent soundness invariants ----

#[test]
fn descent_steps_are_monotone_decreasing() {
    let cfg = Config::default();
    let res = galois_group(&iz(&[-2, 0, 0, 1]), &cfg).unwrap(); // x³−2, S₃
    // every recorded step strictly decreases the order (a genuine descent)
    for (from, to, idx) in &res.steps {
        assert!(to < from, "step did not descend: {from} -> {to}");
        assert!(*idx >= 2, "index must be ≥ 2 for a proper maximal descent");
    }
}

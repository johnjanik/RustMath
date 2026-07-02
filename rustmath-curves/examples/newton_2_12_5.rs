//! Convergence experiment: seed the [2,12,5] factorized residual from the
//! flag-native circle packing and refine with Levenberg–Marquardt.
//!
//! Run: cargo run --release -p rustmath-curves --example newton_2_12_5

use rustmath_curves::belyi::flag_packing::flag_pack;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    default_gauge_freeze, factorized_roots_from_flag_layout, lm_refine_gauge, NewtonConfig,
};
use rustmath_curves::belyi::packing::PackingConfig;

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];

fn main() {
    let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
    let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
    let tri = flag_triangulation(&s0, &s1).unwrap();

    println!("=== flag-native circle packing ===");
    let layout = flag_pack(&tri, &PackingConfig::default());
    println!(
        "  disk relaxation: converged={} iters={} max_angle_err={:.3e}",
        layout.packing.converged, layout.packing.iterations, layout.packing.max_angle_error
    );
    println!("  carried leaves: {}", layout.carried_leaves.len());

    println!("\n=== seed: packing -> factor roots + (λ,c) fit ===");
    let seed = match factorized_roots_from_flag_layout(&tri, &layout) {
        Ok(r) => r,
        Err(e) => {
            println!("  FAILED to build seed: {e}");
            return;
        }
    };
    println!(
        "  roots (A,B,R,S,U) = ({},{},{},{},{})",
        seed.roots_a.len(),
        seed.roots_b.len(),
        seed.roots_r.len(),
        seed.roots_s.len(),
        seed.roots_u.len()
    );
    println!("  λ = {:.4}  c = {:.4}", seed.lambda, seed.c);
    println!("  seed residual‖·‖ = {:.6e}", seed.residual_norm());

    println!("\n=== Levenberg–Marquardt refinement ===");
    let cfg = NewtonConfig {
        max_iters: 2000,
        tol: 1e-13,
        fd_step: 1e-7,
    };
    let (_out, rep) = lm_refine_gauge(&seed, &cfg, &default_gauge_freeze());
    // Print a sparse trace.
    for (i, f) in rep.history.iter().enumerate() {
        if i == 0 || i == rep.history.len() - 1 || i % 20 == 0 {
            println!("  iter {:>4}: ‖r‖ = {:.6e}", i, f);
        }
    }
    println!(
        "\n  initial ‖r‖ = {:.6e}\n  final   ‖r‖ = {:.6e}\n  iterations = {}\n  converged (<1e-12) = {}",
        rep.initial_residual, rep.final_residual, rep.iterations, rep.converged
    );
    let drop = if rep.final_residual > 0.0 {
        rep.initial_residual / rep.final_residual
    } else {
        f64::INFINITY
    };
    println!("  residual reduction factor = {:.3e}", drop);
    let still_decreasing = rep.history.len() >= 2
        && rep.history[rep.history.len() - 1] < 0.9 * rep.history[rep.history.len() - 2];
    println!(
        "\n  VERDICT: {}",
        if rep.final_residual < 1e-8 {
            "seed lies in a Newton basin — refined to ~machine tolerance."
        } else if drop > 1e5 && (rep.final_residual < 1e-4 || still_decreasing) {
            "IN a Newton basin — residual collapsed and is still decreasing; the f64 \n           tail is limited by high-ramification conditioning (use arbitrary precision \n           + analytic Jacobian for the final digits). The packing seed WORKS."
        } else {
            "seed NOT in a Newton basin — pivot to the modular-functions seed."
        }
    );
}

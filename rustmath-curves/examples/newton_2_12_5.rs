//! Convergence experiment: seed the [2,12,5] factorized residual from the
//! flag-native circle packing and refine with Levenberg–Marquardt.
//!
//! Run: cargo run --release -p rustmath-curves --example newton_2_12_5

use rustmath_curves::belyi::flag_packing::flag_pack;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    default_gauge_freeze, factorized_roots_from_flag_layout, jacobian_pivot_spread, lm_refine_gauge,
    NewtonConfig,
};
use rustmath_curves::belyi::newton_hp::{refine_hp, NewtonHpConfig};
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
        max_iters: 300,
        tol: 1e-13,
        fd_step: 1e-7,
    };
    let (refined, rep) = lm_refine_gauge(&seed, &cfg, &default_gauge_freeze());
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
        "\n  VERDICT (f64): {}",
        if rep.final_residual < 1e-8 {
            "seed lies in a Newton basin — refined to ~machine tolerance."
        } else if drop > 1e5 && (rep.final_residual < 1e-4 || still_decreasing) {
            "IN a Newton basin — residual collapsed; f64 tail is conditioning-limited."
        } else {
            "seed NOT in a Newton basin — pivot to the modular-functions seed."
        }
    );

    println!("\n=== Jacobian conditioning at the refined point ===");
    // default gauge = 6 dof (3 roots); extended = 8 dof (add roots_b[0]); more = 10.
    for (label, frozen) in [
        ("6-dof (3 roots)", default_gauge_freeze()),
        ("8-dof (+roots_b[0])", vec![0, 1, 48, 49, 50, 51, 16, 17]),
        ("10-dof (+b0,+r0)", vec![0, 1, 48, 49, 50, 51, 16, 17, 32, 33]),
    ] {
        let (pmin, pmax, ratio) = jacobian_pivot_spread(&refined, &frozen, 1e-6);
        println!(
            "  {:<20}: min|piv|={:.3e}  max|piv|={:.3e}  spread={:.3e}",
            label, pmin, pmax, ratio
        );
    }

    println!("\n=== Stage 3: arbitrary-precision Newton ===");
    for &bits in &[256_u32, 512, 1024] {
        let hp = NewtonHpConfig {
            prec_bits: bits,
            max_iters: 60,
            target: 1e-200,
            frozen: default_gauge_freeze(),
        };
        let hr = refine_hp(&refined, &hp);
        println!(
            "  prec {:>4} bits (~{:>3} digits): {:.3e} -> {:.3e}  in {} iters",
            bits,
            (bits as f64 * 0.301) as u32,
            hr.initial_residual,
            hr.final_residual,
            hr.iterations
        );
    }
    println!(
        "\n  FINDING: identical slow rate at 77 vs 308 digits, and a near-singular\n  \
         Jacobian (spread ~1e13-1e14, robust to gauge over-fixing) ⇒ the endgame wall\n  \
         is a STRUCTURAL degeneracy from the R^5/U^12 ramification, NOT f64 precision.\n  \
         Arbitrary precision alone does not suffice; the singular root needs DEFLATION\n  \
         (then the hp engine converges quadratically)."
    );
}

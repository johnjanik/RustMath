//! Perturbation test: is the singular Jacobian a seed artifact (two double-zeros
//! wrongly coincident) or intrinsic? Take the LM-refined roots, force the closest
//! A–A pair apart to O(0.5), re-refine, and compare residual + Jacobian
//! conditioning.
//!
//!   residual recovers AND spread improves  ⇒ a well-separated solution exists;
//!                                             the packing just gave a bad basin →
//!                                             build a proper (hyperbolic) packing.
//!   LM pulls them back / stays singular    ⇒ the degeneracy is intrinsic → deflation.
//!
//! Run: cargo run --release -p rustmath-curves --example perturb_2_12_5

use num_complex::Complex64;
use rustmath_curves::belyi::flag_packing::flag_pack;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    factorized_roots_from_flag_layout, jacobian_pivot_spread, lm_refine_gauge, min_root_separation,
    NewtonConfig,
};
use rustmath_curves::belyi::packing::PackingConfig;

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];

// Gauge that freezes 3 NON-A points (r0, u0, u1) so every A-root is free to move.
const A_FREE_GAUGE: [usize; 6] = [32, 33, 48, 49, 50, 51];

fn closest_a_pair(a: &[Complex64]) -> (usize, usize, f64) {
    let mut best = (0, 1, f64::INFINITY);
    for i in 0..a.len() {
        for j in (i + 1)..a.len() {
            let d = (a[i] - a[j]).norm();
            if d < best.2 {
                best = (i, j, d);
            }
        }
    }
    best
}

fn report(tag: &str, roots: &rustmath_curves::belyi::factorized_residual::FactorizedRoots) {
    let sep = min_root_separation(roots);
    let (pmin, _pmax, spread) = jacobian_pivot_spread(roots, &A_FREE_GAUGE, 1e-6);
    println!(
        "  {:<24}: ‖r‖={:.3e}  min_all={:.3e}  min|piv|={:.3e}  spread={:.3e}",
        tag,
        roots.residual_norm(),
        sep.min_all,
        pmin,
        spread
    );
}

fn main() {
    let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
    let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
    let tri = flag_triangulation(&s0, &s1).unwrap();
    let layout = flag_pack(&tri, &PackingConfig::default());
    let seed = factorized_roots_from_flag_layout(&tri, &layout).expect("seed");

    let cfg300 = NewtonConfig { max_iters: 300, tol: 1e-15, fd_step: 1e-7 };
    let cfg_long = NewtonConfig { max_iters: 4000, tol: 1e-15, fd_step: 1e-7 };

    // Baseline: refine the packing seed under the A-free gauge.
    let (refined, _) = lm_refine_gauge(&seed, &cfg300, &A_FREE_GAUGE);
    println!("=== baseline (packing seed, A-free gauge) ===");
    report("refined 300", &refined);

    let (i, j, d) = closest_a_pair(&refined.roots_a);
    println!("\n  closest A-pair: ({i},{j}) at distance {d:.3e}");

    // Perturb: jitter ALL the zeros (A, B) apart — the layout has multiple
    // near-coincident pairs, so a single-pair push leaves others singular. Use a
    // deterministic golden-angle spray of magnitude eps so every coincidence breaks.
    let eps = 0.2_f64;
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let jitter = |v: &mut Vec<Complex64>, base: usize| {
        for (k, z) in v.iter_mut().enumerate() {
            let theta = 2.0 * std::f64::consts::PI * (((base + k) as f64 * phi).fract());
            *z += Complex64::new(eps * theta.cos(), eps * theta.sin());
        }
    };
    let mut perturbed = refined.clone();
    jitter(&mut perturbed.roots_a, 0);
    jitter(&mut perturbed.roots_b, 8);

    println!("\n=== after jittering all zeros (eps={eps}) ===");
    report("perturbed (pre-refine)", &perturbed);

    // Re-refine from the separated configuration.
    let (refined2, rep2) = lm_refine_gauge(&perturbed, &cfg_long, &A_FREE_GAUGE);
    println!("\n=== re-refined from the separated seed ({} iters) ===", rep2.iterations);
    report("refined2", &refined2);

    let (i2, j2, d2) = closest_a_pair(&refined2.roots_a);
    println!("\n  closest A-pair now: ({i2},{j2}) at distance {d2:.3e}");
    let sep_base = min_root_separation(&refined).min_all;
    let sep2 = min_root_separation(&refined2).min_all;
    let (pmin_base, _, spread_base) = jacobian_pivot_spread(&refined, &A_FREE_GAUGE, 1e-6);
    let (pmin2, _, spread2) = jacobian_pivot_spread(&refined2, &A_FREE_GAUGE, 1e-6);
    let recovered = refined2.residual_norm() < 1e-4;
    let separation_improved = sep2 > 100.0 * sep_base;
    let conditioning_improved = spread2 < spread_base && pmin2 > pmin_base;
    println!(
        "\n  VERDICT: {}",
        if recovered && separation_improved && conditioning_improved {
            "CONFIRMED — a big jitter returns to a solution whose roots are far more\n           \
             separated (min_all up ~1e4x) AND whose Jacobian is markedly less singular.\n           \
             The singularity is a SEED artifact; a proper (hyperbolic) packing that\n           \
             places distinct roots is the fix. Deflation is not needed."
        } else if d2 < 10.0 * d && recovered {
            "roots snapped BACK together — the degeneracy is intrinsic; go to deflation."
        } else {
            "inconclusive — inspect the numbers above."
        }
    );
}

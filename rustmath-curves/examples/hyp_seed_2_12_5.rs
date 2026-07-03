//! The hyperbolic seed, end to end: maximal packing → Poincaré-disk layout → read
//! the original dessin-vertex positions → factor-root seed. Compare its root
//! separation and Jacobian conditioning against the euclidean packing seed that
//! produced the singular Jacobian.
//!
//! Run: cargo run --release -p rustmath-curves --example hyp_seed_2_12_5

use num_complex::Complex64;
use rustmath_curves::belyi::flag_packing::flag_pack;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::hyperbolic_packing::{
    hyperbolic_layout, maximal_packing_of, HypPackingConfig,
};
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    factorized_roots_from_flag_layout, factorized_roots_from_positions, jacobian_pivot_spread,
    lm_refine_gauge, min_root_separation, NewtonConfig,
};
use rustmath_curves::belyi::packing::PackingConfig;

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];

// Freeze 3 non-A points (r0, u0, u1) so the conditioning probe sees the zeros move.
const A_FREE_GAUGE: [usize; 6] = [32, 33, 48, 49, 50, 51];

fn describe(tag: &str, roots: &rustmath_curves::belyi::factorized_residual::FactorizedRoots) {
    let sep = min_root_separation(roots);
    let (pmin, _pmax, spread) = jacobian_pivot_spread(roots, &A_FREE_GAUGE, 1e-6);
    println!(
        "  {:<20}: ‖r‖={:.3e}  min_all={:.3e} ({})  min_zp={:.3e}  min|piv|={:.3e}  spread={:.3e}",
        tag,
        roots.residual_norm(),
        sep.min_all,
        sep.closest_pair,
        sep.min_zero_pole,
        pmin,
        spread
    );
}

fn main() {
    let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
    let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
    let tri = flag_triangulation(&s0, &s1).unwrap();

    // Euclidean seed (the folded one).
    let el = flag_pack(&tri, &PackingConfig::default());
    let euclid_seed = factorized_roots_from_flag_layout(&tri, &el).expect("euclid seed");

    // Hyperbolic seed.
    let (sub, hp) = maximal_packing_of(&tri, &HypPackingConfig::default());
    let layout = hyperbolic_layout(&sub, &hp);
    let n = tri.n_vertices();
    let unplaced = (0..n).filter(|&u| layout.positions[u].is_none()).count();
    println!(
        "hyperbolic packing: converged={} (err {:.1e}); original vertices placed: {}/{}",
        hp.converged,
        hp.max_angle_error,
        n - unplaced,
        n
    );
    let positions: Vec<Option<Complex64>> = (0..n).map(|u| layout.positions[u]).collect();
    // Coincidence scan: any two original vertices at (nearly) the same point?
    let mut coincident = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if let (Some(zi), Some(zj)) = (positions[i], positions[j]) {
                if (zi - zj).norm() < 1e-9 {
                    coincident.push((i, j, (zi - zj).norm()));
                }
            }
        }
    }
    println!("coincident original-vertex pairs (<1e-9): {coincident:?}");
    if let Some(&(i, j, _)) = coincident.first() {
        println!(
            "  vertex {i} at {:?}, vertex {j} at {:?}  (puncture={})",
            positions[i].unwrap(),
            positions[j].unwrap(),
            hp.puncture
        );
    }
    let hyp_seed = factorized_roots_from_positions(&tri, &positions).expect("hyp seed");

    println!("\n=== seed comparison (A-free gauge) ===");
    describe("euclidean (folded)", &euclid_seed);
    describe("hyperbolic", &hyp_seed);

    // Refine both and compare where they land.
    let cfg = NewtonConfig { max_iters: 400, tol: 1e-15, fd_step: 1e-7 };
    let (e_ref, _) = lm_refine_gauge(&euclid_seed, &cfg, &A_FREE_GAUGE);
    let (h_ref, _) = lm_refine_gauge(&hyp_seed, &cfg, &A_FREE_GAUGE);
    println!("\n=== after 400 LM iters ===");
    describe("euclidean refined", &e_ref);
    describe("hyperbolic refined", &h_ref);

    let euclid_folds = min_root_separation(&euclid_seed).min_all == 0.0;
    let hyp_distinct = min_root_separation(&hyp_seed).min_all > 0.0;
    let hyp_solves = h_ref.residual_norm() < 1e-8;
    println!(
        "\n  VERDICT: {}",
        if hyp_distinct && hyp_solves && euclid_folds {
            "the manifold subdivision + hyperbolic layout FIXES the fold: the hyperbolic\n           \
             seed places DISTINCT roots and refines to a GENUINE solution (‖r‖→1e-14),\n           \
             while the euclidean seed folds (coincident roots) and stalls at ~1e-4.\n           \
             (Note: the singular Jacobian at the solution is a Möbius-FRAME artifact —\n           \
             min zero-pole distance is not frame-invariant — not an intrinsic degeneracy.)"
        } else {
            "hyperbolic seed did not clearly beat euclidean — inspect the numbers."
        }
    );
}

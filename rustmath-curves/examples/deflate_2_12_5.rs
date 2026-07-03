//! Deflate the intrinsic singular Jacobian at the [2,12,5] solution. Take the
//! hyperbolic-seeded f64 solution (‖r‖~1e-14, singular Jacobian, spread ~1e26),
//! deflate (augment F(x)=0 with J(x)v=0, a·v=1), and check the deflated Jacobian is
//! full rank (finite spread) — the prerequisite for a quadratic hp endgame.
//!
//! Run: cargo run --release -p rustmath-curves --example deflate_2_12_5

use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::hyperbolic_packing::{
    hyperbolic_layout, maximal_packing_of, HypPackingConfig,
};
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    deflate_refine, factorized_roots_from_positions, jacobian_corank, lm_refine_gauge,
    DeflateConfig, NewtonConfig,
};

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];
const A_FREE_GAUGE: [usize; 6] = [32, 33, 48, 49, 50, 51];

fn main() {
    let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
    let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
    let tri = flag_triangulation(&s0, &s1).unwrap();

    // Hyperbolic seed → f64 singular solution.
    let (sub, hp) = maximal_packing_of(&tri, &HypPackingConfig::default());
    let layout = hyperbolic_layout(&sub, &hp);
    let positions: Vec<_> = (0..tri.n_vertices()).map(|u| layout.positions[u]).collect();
    let seed = factorized_roots_from_positions(&tri, &positions).expect("seed");
    let cfg = NewtonConfig { max_iters: 800, tol: 1e-15, fd_step: 1e-7 };
    let (sol, _) = lm_refine_gauge(&seed, &cfg, &A_FREE_GAUGE);
    println!("f64 solution: ‖r‖ = {:.3e}", sol.residual_norm());

    println!("\n=== numerical corank of the base Jacobian (f64, fd=1e-6) ===");
    for tol in [1e-6, 1e-8, 1e-10, 1e-12] {
        let ck = jacobian_corank(&sol, &A_FREE_GAUGE, 1e-6, tol);
        println!("  rel_tol {tol:.0e}: corank = {ck}");
    }

    // Deflate.
    let dcfg = DeflateConfig::default();
    let (defl, rep) = deflate_refine(&sol, &dcfg, &A_FREE_GAUGE);
    println!("\n=== deflation ===");
    println!("  base Jacobian spread     : {:.3e}   (singular)", rep.base_jac_spread);
    println!("  deflated Jacobian spread : {:.3e}   (full rank ⇒ finite)", rep.deflated_jac_spread);
    println!(
        "  base residual ‖F‖: {:.3e} -> {:.3e}   in {} iters",
        rep.initial_residual, rep.final_residual, rep.iterations
    );
    let _ = &defl;
    println!(
        "\n  FINDING: the base Jacobian corank is LARGE (tens, not 1) — as expected from\n  \
         the U^12/R^5 local multiplicity (>= 60). A single corank-1 augmentation cannot\n  \
         restore rank (deflated spread ~ base spread). This needs ITERATED LVZ deflation:\n  \
         at each step take an SVD to read the numerical rank r, add r null vectors and r\n  \
         normalizations, and reapply to the augmented system until the smallest singular\n  \
         value is clearly nonzero — all in high precision (the corank read itself needs\n  \
         an accurate hp Jacobian; the f64 finite-difference count above is only a proxy)."
    );
}

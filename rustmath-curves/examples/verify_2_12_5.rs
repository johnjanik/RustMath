//! Cheap checks before any heavy endgame build:
//!   (1) is the solution a genuine [2,12,5] map — distinct roots within each factor
//!       (real multiplicities) and no zero=pole (genuine degree 24)?
//!   (2) is the f64 corank count stable across finite-difference steps, or is it
//!       noise-dominated (hence unreliable, needing an hp Jacobian)?
//!
//! Run: cargo run --release -p rustmath-curves --example verify_2_12_5

use num_complex::Complex64;
use rustmath_curves::belyi::factorized_residual::FactorizedRoots;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::hyperbolic_packing::{
    hyperbolic_layout, maximal_packing_of, HypPackingConfig,
};
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    factorized_roots_from_positions, jacobian_corank, lm_refine_gauge, NewtonConfig,
};
use rustmath_curves::belyi::newton_hp::jacobian_pivots_hp;

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];
const A_FREE_GAUGE: [usize; 6] = [32, 33, 48, 49, 50, 51];

fn min_within(v: &[Complex64]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..v.len() {
        for j in (i + 1)..v.len() {
            m = m.min((v[i] - v[j]).norm());
        }
    }
    m
}

fn min_between(a: &[Complex64], b: &[Complex64]) -> f64 {
    let mut m = f64::INFINITY;
    for x in a {
        for y in b {
            m = m.min((x - y).norm());
        }
    }
    m
}

fn main() {
    let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
    let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
    let tri = flag_triangulation(&s0, &s1).unwrap();
    let (sub, hp) = maximal_packing_of(&tri, &HypPackingConfig::default());
    let layout = hyperbolic_layout(&sub, &hp);
    let positions: Vec<_> = (0..tri.n_vertices()).map(|u| layout.positions[u]).collect();
    let seed = factorized_roots_from_positions(&tri, &positions).expect("seed");
    let cfg = NewtonConfig { max_iters: 800, tol: 1e-15, fd_step: 1e-7 };
    let (sol, _) = lm_refine_gauge(&seed, &cfg, &A_FREE_GAUGE);
    println!("solution: ‖r‖ = {:.3e}\n", sol.residual_norm());

    println!("=== (1) map verification: within-factor separations ===");
    let FactorizedRoots { roots_a, roots_b, roots_r, roots_s, roots_u, .. } = &sol;
    println!("  A–A (8 double zeros) : {:.3e}", min_within(roots_a));
    println!("  B–B (8 simple zeros) : {:.3e}", min_within(roots_b));
    println!("  R–R (4 order-5 poles): {:.3e}", min_within(roots_r));
    println!("  S–S (4 simple poles) : {:.3e}", min_within(roots_s));
    println!("  U–U (2 white pts)    : {:.3e}", min_within(roots_u));
    let zeros: Vec<Complex64> = roots_a.iter().chain(roots_b).copied().collect();
    let poles: Vec<Complex64> = roots_r.iter().chain(roots_s).copied().collect();
    let ones: Vec<Complex64> = roots_u.clone();
    let zp = min_between(&zeros, &poles);
    let zo = min_between(&zeros, &ones);
    let po = min_between(&poles, &ones);
    println!("  zero–pole gap        : {zp:.3e}   (>0 ⇒ genuine degree 24, no cancellation)");
    println!("  zero–one gap         : {zo:.3e}");
    println!("  pole–one gap         : {po:.3e}");
    let within_ok = [
        min_within(roots_a),
        min_within(roots_b),
        min_within(roots_r),
        min_within(roots_s),
        min_within(roots_u),
    ]
    .iter()
    .all(|&d| d > 1e-3);
    println!(
        "\n  → within-factor roots {}; zero/pole/one loci {} disjoint.",
        if within_ok { "DISTINCT (genuine 2^8 1^8 | 5^4 1^4 | 12^2 multiplicities)" } else { "NOT all distinct (!)" },
        if zp > 0.0 && zo > 0.0 && po > 0.0 { "are" } else { "are NOT" }
    );
    println!(
        "  → ramification profile is genuine by construction+distinctness. What is NOT\n    \
         checked here (needs analytic continuation): the monodromy group = M24 and the\n    \
         braid orbit. That is the remaining 'is it the right map' question."
    );

    println!("\n=== (2) f64 corank stability across finite-difference steps ===");
    println!("  fd_step \\ rel_tol   1e-6   1e-8   1e-10");
    for fd in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8] {
        let c6 = jacobian_corank(&sol, &A_FREE_GAUGE, fd, 1e-6);
        let c8 = jacobian_corank(&sol, &A_FREE_GAUGE, fd, 1e-8);
        let c10 = jacobian_corank(&sol, &A_FREE_GAUGE, fd, 1e-10);
        println!("  {fd:>7.0e}            {c6:>4}   {c8:>4}   {c10:>4}");
    }
    println!(
        "\n  → if the count swings with fd_step, the f64 corank is noise-dominated and\n    \
         UNRELIABLE — the honest corank needs an arbitrary-precision Jacobian (next)."
    );

    println!("\n=== (2b) arbitrary-precision pivot spectrum (the honest corank) ===");
    for prec in [256u32, 512] {
        let piv = jacobian_pivots_hp(&sol, &A_FREE_GAUGE, prec);
        let maxp = piv.iter().cloned().fold(0.0_f64, f64::max);
        let smallest: Vec<String> = piv.iter().take(8).map(|p| format!("{:.1e}", p / maxp)).collect();
        // corank via a clear gap: count pivots below sqrt(machine) at this precision.
        let dig = (prec as f64 * 0.301) as i32;
        let gap_tol = 10f64.powi(-dig / 2);
        let corank = piv.iter().filter(|&&p| p / maxp < gap_tol).count();
        println!(
            "  {prec:>4} bits: 8 smallest |piv|/max = [{}]  →  corank(<1e{}) = {}",
            smallest.join(", "),
            -dig / 2,
            corank
        );
    }
    println!(
        "\n  → read the spectrum: a clean gap (a cluster at ~0, the rest O(1)) pins the\n    \
         corank r for iterated LVZ; a smooth decay means the clustering geometry, not a\n    \
         crisp rank drop, dominates — pointing at a balanced frame / cleaner solution first."
    );
}

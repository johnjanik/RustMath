//! Root-collision diagnostic: as LM drives the residual down, do the factor roots
//! stay separated (⇒ converging to the genuine map; the singular Jacobian is
//! intrinsic and deflation is the right tool) or collide (⇒ converging to a
//! degenerate stratum; the singularity is spurious and the seed/formulation needs
//! fixing)?
//!
//! Run: cargo run --release -p rustmath-curves --example collision_2_12_5

use rustmath_curves::belyi::flag_packing::flag_pack;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::newton::{
    default_gauge_freeze, factorized_roots_from_flag_layout, lm_refine_gauge, min_root_separation,
    NewtonConfig,
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
    let layout = flag_pack(&tri, &PackingConfig::default());
    let seed = factorized_roots_from_flag_layout(&tri, &layout).expect("seed");
    let frozen = default_gauge_freeze();

    println!("  iters      ‖r‖       min_all   min_zero_pole  min_within  closest");
    println!("  -----   ---------   ---------  -------------  ----------  -------");
    for &iters in &[0usize, 50, 200, 1000, 5000, 20000] {
        // Refine fresh from the seed to `iters` LM steps, then measure separation.
        let cfg = NewtonConfig {
            max_iters: iters,
            tol: 1e-15,
            fd_step: 1e-7,
        };
        let (roots, rep) = if iters == 0 {
            (seed.clone(), None)
        } else {
            let (r, rp) = lm_refine_gauge(&seed, &cfg, &frozen);
            (r, Some(rp))
        };
        let sep = min_root_separation(&roots);
        let res = rep.map(|r| r.final_residual).unwrap_or_else(|| seed.residual_norm());
        println!(
            "  {:>5}   {:.3e}   {:.3e}   {:.3e}     {:.3e}    {}",
            iters, res, sep.min_all, sep.min_zero_pole, sep.min_within_type, sep.closest_pair
        );
    }

    println!(
        "\n  READING: if min_all / min_zero_pole hold ~constant while ‖r‖ falls, the roots\n  \
         stay separated ⇒ the seed converges to the genuine map and the singular\n  \
         Jacobian is intrinsic (deflation is correct). If they shrink toward 0 as ‖r‖\n  \
         falls, the seed is heading to a degenerate stratum (fix the seed instead)."
    );
}

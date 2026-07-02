//! Milestone U0 gate: proves the genus-0 Belyi/conic path in `rustmath-curves`
//! compiles and runs **without** the `genus2` feature (i.e. independent of the
//! feature-gated hyperelliptic/Jacobian stack). See CURVES_BUILD_BLOCKERS_SPEC.md
//! and BUILD_STATUS.md.
//!
//! Run: `cargo test -p rustmath-curves --test belyi_build_gate`  (no --features).

use rustmath_curves::belyi::monodromy::{genus_from_branch_cycles, Permutation};
use rustmath_curves::belyi::pipeline::g6_mueller_gate;
use rustmath_curves::belyi::portal::portal_2_12_5;
use rustmath_quadraticforms::conic::VerdictKind;

#[test]
fn belyi_portal_2_12_5_is_degree_24_genus_0() {
    let p = portal_2_12_5();
    assert_eq!(p.degree(), 24, "portal degree");
    assert_eq!(p.genus(), 0, "portal genus");
}

#[test]
fn belyi_2_12_5_passport_has_genus_zero_via_riemann_hurwitz() {
    // sigma_0 = 2^8 1^8, sigma_1 = 12^2, sigma_inf = 5^4 1^4 on 24 sheets.
    let n = 24;
    let sigma0 = Permutation::from_cycles(
        n,
        &[
            vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7],
            vec![8, 9], vec![10, 11], vec![12, 13], vec![14, 15],
            // remaining 8 fixed points (16..=23) are implicit
        ],
    )
    .expect("sigma0");
    let sigma1 = Permutation::from_cycles(
        n,
        &[
            (0..12).collect::<Vec<_>>(),
            (12..24).collect::<Vec<_>>(),
        ],
    )
    .expect("sigma1");
    let sigma_inf = Permutation::from_cycles(
        n,
        &[
            vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9],
            vec![10, 11, 12, 13, 14], vec![15, 16, 17, 18, 19],
            // 20..=23 fixed
        ],
    )
    .expect("sigma_inf");

    let g = genus_from_branch_cycles(n, &[&sigma0, &sigma1, &sigma_inf]).expect("genus");
    // 2g-2 = -2n + defect = -48 + (8 + 22 + 16) = -48 + 46 = -2  =>  g = 0.
    assert_eq!(g, 0, "Riemann–Hurwitz genus of the [2,12,5] passport");
}

#[test]
fn granboulan_mueller_conic_gate_is_locally_empty() {
    // The (12,2,2,2)/(2,12,5) M23-fixed-field conic is x^2+y^2+z^2 = (-1,-1),
    // anisotropic (no Q-point): a LOCAL obstruction at {2, infinity}.
    let verdict = g6_mueller_gate();
    assert_eq!(
        verdict.kind,
        VerdictKind::LocallyEmpty,
        "Granboulan/Müller conic must read as a local obstruction, got {:?}",
        verdict.kind
    );
}

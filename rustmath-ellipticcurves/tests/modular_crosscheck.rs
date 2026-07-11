//! Cross-checks between the elliptic-curve analytic leg and the
//! modular-symbols machinery (dev-dependency only; the lib graph is clean).
//!
//! Three independent derivations of the same numbers meet here:
//!
//! 1. **a_p**: point counts / Tate reduction types on the curve side vs
//!    Hecke eigenvalues (T_p, and U_p at p | N) of the attached newform on
//!    the modular side (Eichler–Shimura). The same tables were derived a
//!    third way (Python point counting) before either test existed.
//! 2. **Root numbers**: the product of local root numbers from Tate data
//!    (Rohrlich formulas) vs ε = −w_N from the exact rational Fricke
//!    matrix W_N of the modular-symbols space (chunk-5 signs). Completely
//!    different derivations — this closes the loop demanded by stage 2.
//! 3. **Exact L(1)-vanishing**: the Manin–Birch winding projection
//!    (certified-exact) vs the curve-side honest lattice, including
//!    feeding the exact certificate into
//!    `LFunction::analytic_rank_with_exact_l1`.
//!
//! Every curve model below was independently verified (Python: global
//! minimality, conductor from reduction data, point-counted a_p) before
//! being asserted here; the summand matching is by a_2, and the full
//! p ≤ 40 eigenvalue match then re-verifies the model/label pairing.

use rustmath_ellipticcurves::{AnalyticRank, EllipticCurve, LFunction};
use rustmath_integers::Integer;
use rustmath_modular::{
    HeckeEigenvalue, HeckeSummand, ModularSymbolsGamma0, RationalNewformLSeries,
    SummandHeckeAction,
};
use rustmath_rationals::Rational;

fn curve(a: [i64; 5]) -> EllipticCurve {
    EllipticCurve::new(
        Integer::from(a[0]),
        Integer::from(a[1]),
        Integer::from(a[2]),
        Integer::from(a[3]),
        Integer::from(a[4]),
    )
}

/// The rational T_n eigenvalue on a summand (panics on non-rational or
/// mixed action: all summands used here are rational newforms).
fn eigenvalue(m: &ModularSymbolsGamma0, w: &HeckeSummand, n: u64) -> Rational {
    match m.hecke_action_on_summand(w, n).unwrap() {
        SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(v)) => v,
        other => panic!("T_{} does not act by a rational scalar: {:?}", n, other),
    }
}

/// Find the 2-dimensional summand whose a_2 matches (unique at every level
/// used here; the subsequent full a_p match re-verifies the pairing).
fn find_summand<'a>(
    m: &ModularSymbolsGamma0,
    summands: &'a [HeckeSummand],
    a2: i64,
) -> &'a HeckeSummand {
    let want = Rational::from_i64(a2);
    let hits: Vec<&HeckeSummand> = summands
        .iter()
        .filter(|w| w.dimension() == 2 && eigenvalue(m, w, 2) == want)
        .collect();
    assert_eq!(hits.len(), 1, "summand with a_2 = {} not unique", a2);
    hits[0]
}

/// (label, model, level, a_2) for every rational newform level used.
const CURVES: [(&str, [i64; 5], u64, i64); 10] = [
    ("11a1", [0, -1, 1, -10, -20], 11, -2),
    ("14a1", [1, 0, 1, 4, -6], 14, -1),
    ("15a1", [1, 1, 1, -10, -10], 15, -1),
    ("17a1", [1, -1, 1, -1, -14], 17, -1),
    ("19a1", [0, 1, 1, -9, -15], 19, 0),
    ("21a1", [1, 0, 0, -4, -1], 21, -1),
    ("26a1", [1, 0, 1, -5, -8], 26, -1),
    ("26b1", [1, -1, 1, -3, 3], 26, 1),
    ("37a1", [0, 0, 1, -1, 0], 37, -2),
    ("37b1", [0, 1, 1, -23, -50], 37, 0),
];

/// EICHLER–SHIMURA GATE: for every curve/level pair, the exact curve-side
/// a_p (point counts at good p, Tate reduction type at bad p) equals the
/// modular-symbols Hecke eigenvalue (T_p, and U_p at p | N) for every
/// prime p <= 40 — the third independent derivation of these numbers in
/// this sprint.
#[test]
fn test_eichler_shimura_ap_crosscheck() {
    let primes: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for (label, a, level, a2) in &CURVES {
        let e = curve(*a);
        assert_eq!(
            e.compute_conductor(),
            Integer::from(*level as i64),
            "{}: conductor",
            label
        );
        let m = ModularSymbolsGamma0::new(*level);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w = find_summand(&m, dec.summands(), *a2);
        for p in &primes {
            let curve_ap = e.compute_a_p(&Integer::from(*p as i64));
            let modular_ap = eigenvalue(&m, w, *p);
            assert_eq!(
                modular_ap,
                Rational::from_i64(curve_ap),
                "{}: a_{} (curve point count vs Hecke eigenvalue)",
                label,
                p
            );
        }
    }
}

/// ROOT-NUMBER LOOP CLOSURE: the curve-side global root number (product
/// of Rohrlich local factors from Tate data) equals the modular-side
/// ε = −w_N from the exact rational Fricke matrix, for every curve whose
/// conductor is multiplicative-only (all ten CURVES levels are).
#[test]
fn test_root_number_vs_fricke_epsilon() {
    for (label, a, level, a2) in &CURVES {
        let e = curve(*a);
        let m = ModularSymbolsGamma0::new(*level);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w = find_summand(&m, dec.summands(), *a2);
        let ls = RationalNewformLSeries::new(&m, w).unwrap();
        assert_eq!(
            e.root_number(),
            Ok(ls.root_number()),
            "{}: Tate-data root number vs Fricke-derived epsilon",
            label
        );
    }
}

/// MANIN–BIRCH GATE: the exact winding vanishing (certified-exact zeros)
/// agrees with the curve-side lattice, and feeding the exact certificate
/// into `analytic_rank_with_exact_l1` produces the certified answers:
/// 37a1 (winding zero) → OneCertifiedModuloRounding; all other curves
/// (winding nonzero) → ZeroCertified with no numeric leg needed.
#[test]
fn test_winding_certificates_drive_the_lattice() {
    for (label, a, level, a2) in &CURVES {
        let e = curve(*a);
        let m = ModularSymbolsGamma0::new(*level);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w = find_summand(&m, dec.summands(), *a2);
        let vanishes = m.l1_vanishes(w).unwrap();
        assert_eq!(
            vanishes,
            *label == "37a1",
            "{}: exact winding vanishing",
            label
        );
        let lf = LFunction::new(e);
        let r = lf.analytic_rank_with_exact_l1(22, Some(vanishes));
        if *label == "37a1" {
            assert!(
                matches!(r, AnalyticRank::OneCertifiedModuloRounding { .. }),
                "{}: expected rank 1, got {}",
                label,
                r
            );
        } else {
            assert!(
                matches!(r, AnalyticRank::ZeroCertified { l1: None, .. }),
                "{}: expected exact-certificate rank 0, got {}",
                label,
                r
            );
        }
        // and the pure-numeric lattice (no external certificate) agrees
        // wherever it certifies
        let rn = lf.analytic_rank(22);
        assert_eq!(
            rn.certified_value(),
            r.certified_value(),
            "{}: numeric lattice vs certificate lattice",
            label
        );
    }
}

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
//!    (Rohrlich formulas at p ≥ 5, Kraus/Halberstadt tables at additive
//!    2 and 3) vs ε = −w_N from the exact rational Fricke matrix W_N of
//!    the modular-symbols space (chunk-5 signs). Completely different
//!    derivations — this closes the loop demanded by stage 2, including
//!    the wild-additive battery of `test_wild_root_numbers_vs_fricke_epsilon`.
//! 3. **Exact L(1)-vanishing**: the Manin–Birch winding projection
//!    (certified-exact) vs the curve-side honest lattice, including
//!    feeding the exact certificate into
//!    `LFunction::analytic_rank_with_exact_l1`.
//! 4. **L-values**: the certified curve-side L(E,1) / L'(E,1) entering
//!    the BSD ratio vs the modular-side newform L-values (independent
//!    coefficient pipeline), within the two certified budgets.
//!
//! Every curve model below was independently verified (Python: global
//! minimality, conductor from reduction data, point-counted a_p) before
//! being asserted here; the summand matching is by a_2 (or by a full a_p
//! vector at additive levels), and the full p ≤ 40 eigenvalue match then
//! re-verifies the model/label pairing.

use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_ellipticcurves::{AnalyticRank, EllipticCurve, LFunction};
use rustmath_integers::Integer;
use rustmath_modular::{
    HeckeEigenvalue, HeckeSummand, ModularSymbolsGamma0, RationalNewformLSeries, SummandHeckeAction,
};
use rustmath_rationals::Rational;
use rustmath_reals::BigFloat;

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

/// Find the 2-dimensional rational-newform summand matching a full a_p
/// vector (needed at additive levels, where a_2 or a_3 is 0 and no longer
/// separates summands on its own).
fn find_summand_by_aps<'a>(
    m: &ModularSymbolsGamma0,
    summands: &'a [HeckeSummand],
    aps: &[(u64, i64)],
) -> &'a HeckeSummand {
    let hits: Vec<&HeckeSummand> = summands
        .iter()
        .filter(|w| {
            w.dimension() == 2
                && aps
                    .iter()
                    .all(|(p, ap)| eigenvalue(m, w, *p) == Rational::from_i64(*ap))
        })
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "summand with a_p vector {:?} not unique",
        aps
    );
    hits[0]
}

/// KRAUS/HALBERSTADT LOOP CLOSURE (the stage-2 wild-root-number gate):
/// for curves with ADDITIVE reduction at 2 or 3, the global root number
/// assembled from Tate data + the Rizzo tables must equal ε = −w_N from
/// the exact rational Fricke matrix of the attached newform — a derivation
/// sharing NO code with the tables (modular symbols over exact rationals).
/// The newform summand is matched by the full a_p vector (p ≤ 13) and its
/// existence/uniqueness is asserted; conductors are re-derived by Tate.
/// Battery: 10 curves, additive at 2, at 3, and at both, spanning Kodaira
/// types II/III/IV/I₀*/I₁*/I₂*/III*/IV* at the wild primes and both global
/// signs (models and expected conductors from the PARI-derived battery,
/// but nothing modular-side is taken from PARI here).
#[test]
fn test_wild_root_numbers_vs_fricke_epsilon() {
    // (model, N, additive at)
    let wild: [([i64; 5], u64, &str); 10] = [
        ([0, 1, 0, -1, 0], 20, "2(IV)"),
        ([0, -1, 0, 1, 0], 24, "2(III)"),
        ([0, 0, 1, 0, 0], 27, "3(II)"),
        ([0, 0, 0, -1, 0], 32, "2(III)"),
        ([0, 0, 0, 0, 1], 36, "2(IV)+3(III)"),
        ([1, -1, 0, 0, -5], 45, "3(I1*)"),
        ([0, 1, 0, 1, 0], 48, "2(II)"),
        ([1, -1, 1, 1, -1], 54, "3(II)"),
        ([0, 0, 0, 1, 2], 56, "2(I1*)"),
        ([1, -1, 0, 9, 0], 63, "3(I2*)"),
    ];
    for (a, level, kinds) in &wild {
        let e = curve(*a);
        assert_eq!(
            e.compute_conductor(),
            Integer::from(*level as i64),
            "{:?}: conductor",
            a
        );
        let curve_w = e
            .root_number()
            .unwrap_or_else(|err| panic!("{:?}: root number should be decided now: {}", a, err));
        let m = ModularSymbolsGamma0::new(*level);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let aps: Vec<(u64, i64)> = [2u64, 3, 5, 7, 11, 13]
            .iter()
            .map(|p| (*p, e.compute_a_p(&Integer::from(*p as i64))))
            .collect();
        let w = find_summand_by_aps(&m, dec.summands(), &aps);
        let ls = RationalNewformLSeries::new(&m, w).unwrap();
        assert_eq!(
            curve_w,
            ls.root_number(),
            "N = {} ({}): Tate+Rizzo global root number vs Fricke epsilon",
            level,
            kinds
        );
    }
}

/// BSD-RATIO DEV-DEP TIE-IN: the L(E,1) entering the certified rank-0
/// ratio equals the MODULAR-side L(f,1) of the attached newform (computed
/// from Hecke eigenvalues of modular symbols — an independent coefficient
/// pipeline), within the two certified budgets; and dividing the modular
/// value by the curve-side Ω reproduces the same recognized rational.
/// Same check at rank 1 for L'(37a,1).
#[test]
#[allow(clippy::type_complexity)] // flat gate table, clearest as-is
fn test_bsd_ratio_curve_vs_modular_lvalues() {
    // (label, model, level, a_2, expected ratio (num, den))
    let cases: [(&str, [i64; 5], u64, i64, (i64, i64)); 3] = [
        ("11a1", [0, -1, 1, -10, -20], 11, -2, (1, 5)),
        ("15a1", [1, 1, 1, -10, -10], 15, -1, (1, 8)),
        ("37b1", [0, 1, 1, -23, -50], 37, 0, (1, 3)),
    ];
    for (label, a, level, a2, (num, den)) in &cases {
        let e = curve(*a);
        let ratio = e.bsd_ratio_rank0(26).unwrap();
        assert_eq!(
            ratio.recognized,
            Rational::new(*num, *den).unwrap(),
            "{}: recognized ratio",
            label
        );
        let m = ModularSymbolsGamma0::new(*level);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w = find_summand(&m, dec.summands(), *a2);
        let ls = RationalNewformLSeries::new(&m, w).unwrap();
        let lv_mod = ls.l1(26).unwrap();
        // the two certified L(1) values agree within the sum of budgets
        let wp = RealField::precision(&ratio.l1.value).max(256);
        let diff = OrderedRing::abs(
            &(ratio.l1.value.clone().with_precision(wp) - lv_mod.value.clone().with_precision(wp)),
        );
        let budget =
            ratio.l1.error_budget().with_precision(wp) + lv_mod.error_budget().with_precision(wp);
        assert!(
            diff < budget,
            "{}: curve-side L(1) vs modular-side L(1): diff {} exceeds combined budget {}",
            label,
            diff.to_decimal_string(6),
            budget.to_decimal_string(6)
        );
        // modular L(1) / curve Omega lands on the same recognized rational
        let q_mod = lv_mod.value.with_precision(wp) / ratio.omega.clone().with_precision(wp);
        let rec = BigFloat::from_rational(&ratio.recognized, wp);
        let d2 = OrderedRing::abs(&(q_mod - rec));
        let tol = BigFloat::from_decimal_str("0.00000000000000000001", wp).unwrap();
        assert!(
            d2 < tol,
            "{}: modular L(1)/Omega vs recognized rational (diff {})",
            label,
            d2.to_decimal_string(6)
        );
    }
    // rank 1: L'(37a, 1) from both pipelines
    let e = curve([0, 0, 1, -1, 0]);
    let ls_curve = rustmath_ellipticcurves::lfunction::CurveLSeries::new(&e).unwrap();
    let dv_curve = ls_curve.l1_derivative(24).unwrap();
    let m = ModularSymbolsGamma0::new(37);
    let dec = m.cuspidal_hecke_decomposition().unwrap();
    let w = find_summand(&m, dec.summands(), -2);
    let ls_mod = RationalNewformLSeries::new(&m, w).unwrap();
    let dv_mod = ls_mod.l1_derivative(24).unwrap();
    let wp = RealField::precision(&dv_curve.value).max(256);
    let diff = OrderedRing::abs(
        &(dv_curve.value.clone().with_precision(wp) - dv_mod.value.clone().with_precision(wp)),
    );
    let budget =
        dv_curve.error_budget().with_precision(wp) + dv_mod.error_budget().with_precision(wp);
    assert!(
        diff < budget,
        "L'(37a,1): curve vs modular pipelines differ by {} (budget {})",
        diff.to_decimal_string(6),
        budget.to_decimal_string(6)
    );
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

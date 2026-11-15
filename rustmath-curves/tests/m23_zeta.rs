//! WP-F: point counts and Frobenius polynomials of the genus 4 curve X of HJLPPZ
//! (`docs/workstreams/m23_golay/HANDOFF.md` Section 3; MAGMA ch. 114 `LPolynomial`,
//! `ZetaFunction`, `NumberOfPlacesOfDegreeOneECF`, digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`).
//!
//! Reference values: `tests/data/m23/zeta_X.json` and `zeta_L.json` (johnjanik/M23 at
//! `d05475b`, computed with Python/FLINT `m23golay/curve.py`, cross-checked with PARI), `h62.json`
//! (same commit), HJLPPZ Prop. 3.14 (45 and 52 at 53). Values marked "new" have no external
//! reference; they are checked by the derived checks (Weil, mod 23, mod 2) and by the counts at
//! `k = 5, 6` predicted from `N_1..N_4` (WP-F design review 1, Section 1).
//!
//! Fast tests run in debug in the normal suite. Opt-in tests:
//!
//! ```text
//! cargo test --release -p rustmath-curves --test m23_zeta -- --ignored full_
//! M23ISGALOIS=/path/to/m23isgalois cargo test --release -p rustmath-curves --test m23_zeta -- --ignored env_
//! ```
//!
//! The `env_` tests read HJLPPZ's `F` and `calF` from the checkout named by `M23ISGALOIS` and
//! panic without it. Without it, the L table, the anchor at 53 and the identity `F o M = c G`
//! do not run; everything else runs on the vendored CC0 model `G`.

mod common;

use common::m23::*;
use rustmath_curves::nodal_biprojective::{
    plane_point_count, BranchRule, Certificate, CountError, CountOptions, Degeneracy, Fibration,
    LocalType, NormalizationCounts, ReductionStatus, Unresolved,
};
use rustmath_curves::weil::{FrobeniusPolynomial, TwoTorsionOrder};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

fn ref_x(p: u64) -> RefRow {
    zeta_x().into_iter().find(|r| r.p == p).expect("row")
}

fn exact() -> CountOptions {
    CountOptions::default()
}

fn unibranch() -> CountOptions {
    CountOptions {
        branch_rule: BranchRule::AssumedUnibranch,
        ..CountOptions::default()
    }
}

fn counts_of(row: &Row) -> &NormalizationCounts {
    row.result
        .as_ref()
        .unwrap_or_else(|e| panic!("p = {}: {e}", row.p))
}

/// Table checks of a counted row against its reference row (counts up to `kmax`, and with
/// `kmax >= 4` the polynomial, Weil, mod 23 and the table's Q candidates).
fn assert_matches_table(r: &NormalizationCounts, reference: &RefRow) {
    let rc = reference.counts.as_ref().expect("reference counts");
    let k = r.counts.len().min(rc.len());
    assert_eq!(&r.counts[..k], &rc[..k], "p = {}: counts", r.p);
    if r.counts.len() >= GENUS {
        let c = checks(r).unwrap().unwrap();
        if let Some(pc) = &reference.p_coeffs {
            assert_eq!(c.frob.coefficients(), &ints(pc)[..], "p = {}: P", r.p);
        }
        assert!(c.weil, "p = {}: Weil", r.p);
        assert_eq!(c.mod23, (0, 0), "p = {}: mod 23", r.p);
        let mut q = c.q.clone();
        q.sort();
        let mut rq = reference.q_candidates.clone();
        rq.sort();
        assert_eq!(q, rq, "p = {}: Q candidates", r.p);
    }
}

fn nodal62(r: &NormalizationCounts) {
    assert_eq!(
        r.status,
        ReductionStatus::Decided(Certificate::Nodal {
            nodes: 62,
            genus_hypothesis: GENUS
        }),
        "p = {}",
        r.p
    );
}

fn non_nodal_types(r: &NormalizationCounts) -> Vec<(LocalType, Option<Vec<usize>>)> {
    match &r.status {
        ReductionStatus::Unresolved(Unresolved::DeltaSumCertified {
            delta_sum: 62,
            non_nodal,
            ..
        }) => non_nodal
            .iter()
            .map(|s| (s.local.clone(), s.branch_orbits.clone()))
            .collect(),
        s => panic!(
            "p = {}: expected DeltaSumCertified with delta-sum 62, got {s:?}",
            r.p
        ),
    }
}

#[test]
fn cc0_models_are_primitive_of_bidegree_4_23() {
    for (name, g) in cc0_models() {
        assert_eq!(g.bidegree(), (4, 23), "{name}");
        assert_eq!(g.content(), Integer::one(), "{name}");
        assert_eq!(g.arithmetic_genus(), 66, "{name}");
    }
}

#[test]
fn reference_polynomials_pass_the_derived_checks() {
    // Every "ok" row of zeta_X.json: Newton's identities give the table's P, which is Weil,
    // satisfies P(chi) = P(chi p) = 0 mod 23 and splits mod 2 as Q Q^rev with the table's Q.
    for row in zeta_x().iter().filter(|r| r.ok) {
        let counts: Vec<Integer> = row
            .counts
            .as_ref()
            .unwrap()
            .iter()
            .map(|&n| Integer::from(n))
            .collect();
        let fp = FrobeniusPolynomial::from_point_counts(&Integer::from(row.p), &counts).unwrap();
        assert_eq!(fp.coefficients(), &ints(row.p_coeffs.as_ref().unwrap())[..]);
        assert!(fp.is_weil());
        let c = chi(row.p);
        assert!(fp.eval_mod(&Integer::from(c), &Integer::from(23)).is_zero());
        assert!(fp
            .eval_mod(&Integer::from(c * row.p as i64), &Integer::from(23))
            .is_zero());
        assert!(!fp.reciprocal_halves_mod2().is_empty());
    }
    // The skipped row at 5 (reference rule): not Weil.
    let r5 = ref_x(5);
    assert!(!r5.ok);
    let c: Vec<Integer> = r5
        .counts
        .unwrap()
        .iter()
        .map(|&n| Integer::from(n))
        .collect();
    let fp = FrobeniusPolynomial::from_point_counts(&Integer::from(5), &c).unwrap();
    assert!(!fp.is_weil());
}

#[test]
fn x_at_7_k6_and_11_k5_nodal() {
    let models = cc0_models();
    // p = 7: G degenerates (disc vanishes identically mod 7), G7 is nodal with 62 nodes.
    let row = x_row(&models, 7, 6, &exact());
    assert_eq!(row.model, "G7");
    let r = counts_of(&row);
    nodal62(r);
    assert_matches_table(r, &ref_x(7));
    assert_eq!(
        r.predicted_counts_agree(),
        Some(true),
        "k = 5, 6 predicted at 7"
    );
    // p = 11: G is nodal.
    let row = x_row(&models, 11, 5, &exact());
    assert_eq!(row.model, "G");
    let r = counts_of(&row);
    nodal62(r);
    assert_matches_table(r, &ref_x(11));
    assert_eq!(
        r.predicted_counts_agree(),
        Some(true),
        "k = 5 predicted at 11"
    );
    // P mod 2 squarefree at 11: order 15 on J[2] (HANDOFF 3.3 item 5), Decided with the row.
    assert_eq!(
        checks(r).unwrap().unwrap().two_torsion,
        TwoTorsionOrder::DeterminedByP(15)
    );
    // HANDOFF Section 1 link: the T-coordinates of the 62 nodes are the roots of h62 mod 11
    // (the Moebius map changes only V, and is invertible mod 11). The product of the
    // characteristic polynomials of the nodes' T-coordinates is h62 mod 11 made monic.
    let p = 11i64;
    let h: Vec<i64> = h62()
        .iter()
        .map(|c| (c % &Integer::from(p)).to_i64().unwrap().rem_euclid(p))
        .collect();
    let h = fp_factor::make_monic(&fp_factor::trim(&h), p);
    let mut prod = vec![1i64];
    for s in &r.locus.points {
        assert_eq!(s.local, LocalType::Node);
        let cp: Vec<i64> = s
            .x_charpoly
            .as_ref()
            .expect("finite T")
            .iter()
            .map(|&c| c as i64)
            .collect();
        prod = fp_factor::mul(&prod, &cp, p);
    }
    assert_eq!(
        fp_factor::trim(&prod),
        h,
        "nodes over the roots of h62 mod 11"
    );
}

#[test]
fn x_at_17_and_19_k3_nodal() {
    let models = cc0_models();
    for p in [17u64, 19] {
        let row = x_row(&models, p, 3, &exact());
        let r = counts_of(&row);
        nodal62(r);
        assert_matches_table(r, &ref_x(p));
    }
}

#[test]
fn x_at_13_and_31_one_cusp() {
    // The table's D rows: one ordinary cusp (A2, delta 1) and 61 nodes. Exact and reference
    // rules agree (a cusp is unibranch); the status is Unresolved under the project rule.
    let models = cc0_models();
    for p in [13u64, 31] {
        let row = x_row(&models, p, 3, &exact());
        let r = counts_of(&row);
        assert_eq!(
            non_nodal_types(r),
            vec![(LocalType::A(2), Some(vec![1]))],
            "p = {p}"
        );
        assert_eq!(r.locus.node_count(), 61);
        assert_eq!(r.branch_rule, BranchRule::Exact);
        assert_matches_table(r, &ref_x(p));
        let u = x_row(&models, p, 3, &unibranch());
        let ru = counts_of(&u);
        assert_eq!(ru.counts, r.counts);
        assert_eq!(ru.branch_rule, BranchRule::AssumedUnibranch);
        assert!(!ru.status.is_decided());
    }
}

#[test]
fn x_at_5_tacnode_with_conjugate_branches() {
    // G at 5: 60 nodes and a tacnode (A3, delta 2) with conjugate branches. The exact counts are
    // new (WP-F design 7.1): 7, 37, 169, 561, and N_5, N_6 = 3197, 15709 are the values predicted
    // by P (design review 1, Section 1).
    let models = cc0_models();
    let row = x_row(&models, 5, 6, &exact());
    assert_eq!(row.model, "G");
    let r = counts_of(&row);
    assert_eq!(non_nodal_types(r), vec![(LocalType::A(3), Some(vec![2]))]);
    assert_eq!(r.locus.node_count(), 60);
    assert_eq!(r.counts, vec![7, 37, 169, 561, 3197, 15709]);
    assert_eq!(r.predicted_counts_agree(), Some(true));
    let c = checks(r).unwrap().unwrap();
    assert_eq!(
        c.frob.coefficients(),
        &ints(&[1, 1, 6, 20, 16, 100, 150, 125, 625])[..]
    );
    assert!(c.weil);
    assert_eq!(c.mod23, (0, 0));
    let mut q = c.q.clone();
    q.sort();
    assert_eq!(q, vec!["x^4+x^2+x+1", "x^4+x^3+x^2+1"]);
    // The reference rule (tacnode counted once) gives the table's skipped row 8, 36, 170, 560:
    // not Weil, and its predictions at k = 5, 6 fail.
    let u = x_row(&models, 5, 6, &unibranch());
    let ru = counts_of(&u);
    assert_eq!(&ru.counts[..4], &ref_x(5).counts.unwrap()[..]);
    assert!(!checks(ru).unwrap().unwrap().weil);
    assert_eq!(ru.predicted_counts_agree(), Some(false));
}

#[test]
fn x_at_3_and_23_refused() {
    // F mod 3 = (V - 1)^23 T (T - 1)^3 (design 5); G and its Moebius images degenerate too.
    // At 23 the discriminant also vanishes identically for every model (review 1, 3.5).
    let models = cc0_models();
    for p in [3u64, 23] {
        let row = x_row(&models, p, 1, &exact());
        assert_eq!(
            row.result.err(),
            Some(CountError::DegenerateModel(
                Degeneracy::DiscriminantVanishes
            )),
            "p = {p}"
        );
    }
}

/// Brute-force `#X0(F_p)` over `P^1 x P^1(F_p)` for a form mod p.
fn brute_force(f: &rustmath_curves::nodal_biprojective::FormModP) -> u64 {
    let p = f.p();
    let eval = |g: &rustmath_curves::nodal_biprojective::FormModP, x: u64, y: u64| -> u64 {
        let mut acc = 0u64;
        for row in g.coeffs().iter().rev() {
            let mut inner = 0u64;
            for &c in row.iter().rev() {
                inner = (inner * y + c) % p;
            }
            acc = (acc * x + inner) % p;
        }
        acc
    };
    let (rx, ry) = (f.reverse_x(), f.reverse_y());
    let rxy = rx.reverse_y();
    let mut n = 0;
    for x in 0..p {
        for y in 0..p {
            n += u64::from(eval(f, x, y) == 0);
        }
        n += u64::from(eval(&ry, x, 0) == 0);
    }
    for y in 0..p {
        n += u64::from(eval(&rx, 0, y) == 0);
    }
    n + u64::from(eval(&rxy, 0, 0) == 0)
}

#[test]
fn both_fibrations_and_brute_force() {
    // #X0(F_q) does not depend on the fibration: the T-fibration (degree 23 fibres, the vector
    // root-count path) and the V-fibration (quartic fibres) agree; brute force at k = 1.
    let models = cc0_models();
    for (p, name) in [(7u64, "G7"), (11, "G")] {
        let g = &models.iter().find(|m| m.0 == name).unwrap().1;
        let f = g.reduce(p).unwrap();
        for k in 1..=2 {
            let a = plane_point_count(&f, k, Fibration::FibresInX).unwrap();
            let b = plane_point_count(&f, k, Fibration::FibresInY).unwrap();
            assert_eq!(a, b, "{name} at p = {p}, k = {k}");
            if k == 1 {
                assert_eq!(a, brute_force(&f), "{name} at p = {p}: brute force");
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Opt-in: the external data of HJLPPZ (M23ISGALOIS).
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs M23ISGALOIS (a checkout of shaowuz/m23isgalois)"]
fn env_f_equals_g_by_mobius() {
    // F(T, (967W - 3)/(81 - 2267W)) (81 - 2267W)^23 = c G(T, W) with c > 0 (design Section 3):
    // mobius_y divides by the positive content, so the identity is exact equality.
    let dir = require_m23isgalois();
    let f = model_f(&dir);
    assert_eq!(f.content(), Integer::one());
    let g = model_g();
    assert_eq!(f.mobius_y(967, -3, -2267, 81).unwrap(), g);
    // and the inverse map takes G back to F
    assert!(g
        .mobius_y(81, 3, 2267, 967)
        .unwrap()
        .equals_up_to_scalar(&f));
}

#[test]
#[ignore = "needs M23ISGALOIS; run in release"]
fn env_x_from_f_k3() {
    // X rows 7..79 at k <= 3 recomputed from HJLPPZ's F equal the table (F and G are isomorphic
    // models at every prime not dividing 71526 = 2 3 7 13 131).
    let dir = require_m23isgalois();
    let f = model_f(&dir);
    let models = vec![("F", f)];
    for reference in zeta_x().iter().filter(|r| r.ok) {
        let row = x_row(&models, reference.p, 3, &exact());
        let r = counts_of(&row);
        assert_matches_table(r, reference);
    }
}

fn l_row(calf: &[Vec<[Integer; 6]>], p: u64, s0: u64, kmax: u32, opts: &CountOptions) -> Row {
    let result = calf_mod(calf, p, s0).and_then(|f| {
        rustmath_curves::nodal_biprojective::normalization_point_counts(&f, GENUS, kmax, opts)
    });
    Row {
        p,
        s0: Some(s0),
        model: "calF".into(),
        result,
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; run in release"]
fn env_l_table() {
    // zeta_L.json: refusals at 5 and 11; Nodal at 7, 19, 41, 43 with the table values; at 31 the
    // reference-rule counts equal the table but fail mod 23, the exact counts are new.
    let dir = require_m23isgalois();
    let calf = model_calf(&dir);
    let (rows, _) = zeta_l();
    for reference in &rows {
        let s0 = reference.s0.unwrap();
        assert!(degree_one_primes(reference.p).contains(&s0));
        let row = l_row(&calf, reference.p, s0, 4, &exact());
        if !reference.ok {
            assert_eq!(
                row.result.err(),
                Some(CountError::DegenerateModel(
                    Degeneracy::DiscriminantVanishes
                )),
                "p = {}",
                reference.p
            );
            continue;
        }
        let r = counts_of(&row);
        if reference.p == 31 {
            // 57 nodes, an ordinary triple point (one rational and two conjugate tangents) and a
            // tacnode with conjugate branches.
            let mut t = non_nodal_types(r);
            t.sort_by_key(|x| format!("{x:?}"));
            assert_eq!(
                t,
                vec![
                    (LocalType::A(3), Some(vec![2])),
                    (LocalType::OrdinaryMultiple(3), Some(vec![1, 2])),
                ]
            );
            assert_eq!(r.locus.node_count(), 57);
            assert_eq!(r.counts, vec![43, 1017, 30265, 923729]);
            let c = checks(r).unwrap().unwrap();
            assert_eq!(c.frob.coefficients()[1..5], ints(&[11, 88, 682, 4438])[..]);
            assert!(c.weil);
            assert_eq!(c.mod23, (0, 0));
            let u = l_row(&calf, 31, s0, 4, &unibranch());
            let ru = counts_of(&u);
            assert_eq!(&ru.counts, reference.counts.as_ref().unwrap());
            let cu = checks(ru).unwrap().unwrap();
            assert!(cu.weil, "the reference P is Weil");
            assert_eq!(cu.mod23, (15, 7), "but fails the mod 23 check");
            continue;
        }
        nodal62(r);
        assert_eq!(
            &r.counts,
            reference.counts.as_ref().unwrap(),
            "p = {}",
            reference.p
        );
        let c = checks(r).unwrap().unwrap();
        assert_eq!(
            c.frob.coefficients(),
            &ints(reference.p_coeffs.as_ref().unwrap())[..]
        );
        assert!(c.weil);
        assert_eq!(c.mod23, (0, 0));
        let mut q = c.q.clone();
        q.sort();
        let mut rq = reference.q_candidates.clone();
        rq.sort();
        assert_eq!(q, rq);
    }
}

#[test]
#[ignore = "needs M23ISGALOIS"]
fn env_anchor_53() {
    // HJLPPZ Prop. 3.14 and test_core.py: the three degree-one primes over 53 give 45, 52, 57.
    let dir = require_m23isgalois();
    let calf = model_calf(&dir);
    let (_, check) = zeta_l();
    let s0s = degree_one_primes(53);
    assert_eq!(s0s, vec![21, 36, 41]);
    let mut n1 = Vec::new();
    for s0 in s0s {
        let row = l_row(&calf, 53, s0, 1, &exact());
        let r = counts_of(&row);
        nodal62(r);
        n1.push(r.counts[0]);
    }
    assert_eq!(n1, vec![52, 57, 45]);
    n1.sort();
    assert_eq!(n1, check);
}

#[test]
#[ignore = "needs M23ISGALOIS; release, about 15 s on 4 threads"]
fn env_l31_k5() {
    // Review 1, Section 1: at (31, s - 12) N_5 predicted from the exact N_1..N_4 is counted
    // (28604633); the reference rule's prediction (28598074) is not (28604634).
    let dir = require_m23isgalois();
    let calf = model_calf(&dir);
    let r = l_row(&calf, 31, 12, 5, &exact());
    let r = counts_of(&r);
    assert_eq!(r.counts[4], 28604633);
    assert_eq!(r.predicted_counts_agree(), Some(true));
    let u = l_row(&calf, 31, 12, 5, &unibranch());
    let u = counts_of(&u);
    assert_eq!(u.counts[4], 28604634);
    assert_eq!(u.predicted_counts_agree(), Some(false));
}

// ------------------------------------------------------------------------------------------
// Opt-in release runs on the CC0 models.
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "release, about 1 min on 4 threads"]
fn full_x_table_k4() {
    // All 18 rows of zeta_X.json at k = 4 from [G, G7, G13]; 13, 31, 41, 73 have one cusp.
    let models = cc0_models();
    for reference in zeta_x().iter().filter(|r| r.ok) {
        let row = x_row(&models, reference.p, 4, &exact());
        let r = counts_of(&row);
        if [13, 31, 41, 73].contains(&reference.p) {
            assert_eq!(non_nodal_types(r), vec![(LocalType::A(2), Some(vec![1]))]);
        } else {
            nodal62(r);
        }
        assert_matches_table(r, reference);
        assert_eq!(&r.counts, reference.counts.as_ref().unwrap());
    }
    // p = 5 at k = 4: the exact counts (new).
    let r = x_row(&models, 5, 4, &exact());
    assert_eq!(counts_of(&r).counts, vec![7, 37, 169, 561]);
}

#[test]
#[ignore = "release, about 2 min on 4 threads"]
fn full_x_k5_31_41() {
    // Review 1, 3.3: at 31 and 41 (one cusp each) the counts at k = 5 are predicted by P.
    let models = cc0_models();
    for p in [31u64, 41] {
        let row = x_row(&models, p, 5, &exact());
        assert_eq!(
            counts_of(&row).predicted_counts_agree(),
            Some(true),
            "p = {p}"
        );
    }
}

#[test]
#[ignore = "release"]
fn full_both_fibrations_k3() {
    // The T-fibration (degree 23 fibres) at k = 3 for p <= 31 gives the same plane counts.
    let models = cc0_models();
    for p in [7u64, 11, 13, 17, 19, 29, 31] {
        let (forms_status, best) = {
            let forms: Vec<(String, rustmath_curves::nodal_biprojective::FormModP)> = models
                .iter()
                .filter_map(|(n, g)| g.reduce(p).ok().map(|f| (n.to_string(), f)))
                .collect();
            let (_, b) = select_model(&forms);
            (forms, b)
        };
        let f = &forms_status[best.expect("a model")].1;
        for k in 1..=3 {
            assert_eq!(
                plane_point_count(f, k, Fibration::FibresInX).unwrap(),
                plane_point_count(f, k, Fibration::FibresInY).unwrap(),
                "p = {p}, k = {k}"
            );
        }
    }
}

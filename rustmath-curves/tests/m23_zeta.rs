//! WP-F: point counts and Frobenius polynomials of the genus 4 curve X of HJLPPZ
//! (`docs/workstreams/m23_golay/HANDOFF.md` Section 3; MAGMA ch. 114 `LPolynomial`,
//! `ZetaFunction`, `NumberOfPlacesOfDegreeOneECF`, digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`).
//!
//! Reference values: `tests/data/m23/zeta_X.json` and `zeta_L.json` (johnjanik/M23 at
//! `2d11038`, computed with Python/FLINT `m23golay/curve.py`, cross-checked with PARI; each row
//! with the singular points of the owner's classifier), `h62.json` (`d05475b`, unchanged at
//! `2d11038`), the owner's recount `data/rustmath_crosscheck.json` (`2d11038`), HJLPPZ
//! Prop. 3.14 (45 and 52 at 53). The rows at X 5 and L (31, 12) of `d05475b`, made with the
//! assumed-unibranch rule, are stated in the tests that check that rule. Values marked "new"
//! have no external reference; they are checked by the derived checks (Weil, mod 23, mod 2) and
//! by the counts at `k = 5, 6` predicted from `N_1..N_4` (WP-F design review 1, Section 1).
//!
//! Statuses: Decided by Lemma 9.1 of the owner's paper (johnjanik/M23 at `2d11038`,
//! `paper/golay_m23.tex`, `\label{lem:reduction}`), with the nodal or the delta-sum certificate
//! (the owner's decision on F-9, 2026-11-15). The irreducibility evidence of every M23 form is a
//! fibre with `gcd(4, 23) = 1` (F-9 design, Section 3.2; PARI `wpf/gp/m23_cc0.gp`).
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
    plane_point_count, AbsoluteWitness, BasePoint, BranchRule, Certificate, CountError,
    CountOptions, Degeneracy, Fibration, FormModP, IrreducibleFibre, LocalType,
    NormalizationCounts, ReductionStatus,
};
use rustmath_curves::weil::{FrobeniusPolynomial, RankBound, TwoTorsionOrder};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;
use std::collections::BTreeMap;

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

fn ref_x(p: u64) -> RefRow {
    zeta_x().into_iter().find(|r| r.p == p).expect("row")
}

fn exact() -> CountOptions {
    CountOptions::default()
}

/// The counts of the reference rule of `curve.py` at `d05475b` (every non-nodal point once).
fn unibranch(r: &NormalizationCounts) -> Vec<u64> {
    r.counts_under(BranchRule::AssumedUnibranch)
        .expect("every point classified")
}

fn types(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries.iter().map(|&(k, n)| (k.to_string(), n)).collect()
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

/// Decided by the nodal certificate with 62 nodes; absolute irreducibility by a fibre and
/// `gcd(4, 23) = 1`; exact counts.
fn nodal62(r: &NormalizationCounts) {
    match &r.status {
        ReductionStatus::Decided(Certificate::Nodal {
            nodes: 62,
            genus_hypothesis: GENUS,
            irreducibility,
            ..
        }) => assert_eq!(
            irreducibility.absolute,
            AbsoluteWitness::Bidegree { gcd: 1 },
            "p = {}",
            r.p
        ),
        s => panic!("p = {}: expected Decided: nodal (62 nodes), got {s:?}", r.p),
    }
    assert_eq!(r.branch_rule, BranchRule::Exact);
    assert_eq!(r.status.label(), "Decided: nodal (62 nodes)");
}

/// Decided by the delta-sum certificate (delta-sum 62); the non-nodal points with their branch
/// orbits. The owner's decision on F-9 (2026-11-15) made this status Decided: it was
/// `Unresolved(DeltaSumCertified)` before.
fn delta_sum62(r: &NormalizationCounts) -> Vec<(LocalType, Option<Vec<usize>>)> {
    assert_eq!(r.branch_rule, BranchRule::Exact);
    match &r.status {
        ReductionStatus::Decided(Certificate::DeltaSum {
            delta_sum: 62,
            genus_hypothesis: GENUS,
            non_nodal,
            irreducibility,
            ..
        }) => {
            assert_eq!(
                irreducibility.absolute,
                AbsoluteWitness::Bidegree { gcd: 1 },
                "p = {}",
                r.p
            );
            non_nodal
                .iter()
                .map(|s| (s.local.clone(), s.branch_orbits.clone()))
                .collect()
        }
        s => panic!(
            "p = {}: expected Decided by the delta-sum certificate (delta-sum 62), got {s:?}",
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
    // Every row of zeta_X.json at 2d11038 is "ok" (p = 5 included, corrected by the owner, F-8):
    // Newton's identities give the table's P, which is Weil, satisfies P(chi) = P(chi p) = 0
    // mod 23 and splits mod 2 as Q Q^rev with the table's Q.
    let rows = zeta_x();
    assert_eq!(rows.len(), 19);
    assert!(rows.iter().all(|r| r.ok));
    for row in &rows {
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
}

#[test]
fn x_certificates_equal_the_owners() {
    // For every row of zeta_X.json (2d11038) the best CC0 model at p is Decided, and its
    // singular points in the owner's labels equal the row's `certificate.singular_points` (the
    // owner's classifier by blow-ups, m23golay/curve.py). Certificates only, no counts.
    let models = cc0_models();
    for row in zeta_x() {
        let forms: Vec<(String, FormModP)> = models
            .iter()
            .filter_map(|(n, g)| g.reduce_primitive(row.p).ok().map(|f| (n.to_string(), f)))
            .collect();
        let (statuses, best) = select_model(&forms);
        let b = best.unwrap_or_else(|| panic!("p = {}: every model refused", row.p));
        let (status, locus) = statuses[b].as_ref().unwrap();
        let owner = row.singular_points.as_ref().expect("certificate");
        assert_eq!(&owner_types(locus), owner, "p = {}", row.p);
        let nodal_only = owner.keys().all(|k| k == "A_1");
        match status {
            ReductionStatus::Decided(Certificate::Nodal { .. }) => assert!(nodal_only),
            ReductionStatus::Decided(Certificate::DeltaSum { .. }) => assert!(!nodal_only),
            s => panic!("p = {}: {s:?}", row.p),
        }
        assert_eq!(
            status.certificate().unwrap().irreducibility().absolute,
            AbsoluteWitness::Bidegree { gcd: 1 }
        );
    }
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
    // The table's D rows: one ordinary cusp (A2, delta 1) and 61 nodes, as the owner's
    // classifier finds ({A_1: 61, A_2: 1}). Decided by the delta-sum certificate (the owner's
    // decision on F-9; it was Unresolved before). Exact and reference rules agree (a cusp is
    // unibranch).
    let models = cc0_models();
    for p in [13u64, 31] {
        let row = x_row(&models, p, 3, &exact());
        let r = counts_of(&row);
        assert_eq!(
            delta_sum62(r),
            vec![(LocalType::A(2), Some(vec![1]))],
            "p = {p}"
        );
        assert_eq!(r.locus.node_count(), 61);
        assert_eq!(
            r.status.label(),
            "Decided: delta-sum, Lemma 9.1 (61 nodes, A2)"
        );
        assert_eq!(owner_types(&r.locus), types(&[("A_1", 61), ("A_2", 1)]));
        assert_matches_table(r, &ref_x(p));
        assert_eq!(unibranch(r), r.counts);
    }
    // At 13 the model is G13 (G and G7 have a horizontal line component mod 13).
    assert_eq!(x_row(&models, 13, 1, &exact()).model, "G13");
}

#[test]
fn x_at_5_tacnode_with_conjugate_branches() {
    // G at 5: 60 nodes and a tacnode (A3, delta 2) with conjugate branches; the owner's
    // classifier finds {A_1: 60, A_3: 1}. The counts 7, 37, 169, 561 are the reference row at
    // 2d11038 (corrected by the owner, F-8, after WP-F), and N_5, N_6 = 3197, 15709 are the
    // values predicted by P (design review 1, Section 1). Decided by the delta-sum certificate:
    // X has good reduction at 5 (the owner's decision on F-9; it was Unresolved before).
    let models = cc0_models();
    let row = x_row(&models, 5, 6, &exact());
    assert_eq!(row.model, "G");
    let r = counts_of(&row);
    assert_eq!(delta_sum62(r), vec![(LocalType::A(3), Some(vec![2]))]);
    assert_eq!(r.locus.node_count(), 60);
    assert_eq!(
        r.status.label(),
        "Decided: delta-sum, Lemma 9.1 (60 nodes, A3 conjugate branches)"
    );
    assert_eq!(owner_types(&r.locus), types(&[("A_1", 60), ("A_3", 1)]));
    // No F_5-rational fibre of G is irreducible of full degree in either ruling; the x-fibre over
    // the root of y^2 + 2 is (PARI, F-9 design `wpf/gp/m23_p5.gp`: the factor degrees of every
    // rational fibre, and the first x-fibre irreducible over F_25 is the one over y^2 + 2; PARI
    // factors G mod 5 as one factor of bidegree (4, 23), `m23_cc0.gp`).
    assert_eq!(
        r.status.certificate().unwrap().irreducibility().fibre,
        IrreducibleFibre {
            fibration: Fibration::FibresInX,
            base: BasePoint::Closed(vec![2, 0, 1]),
        }
    );
    assert_matches_table(r, &ref_x(5));
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
    // The reference rule (tacnode counted once) gives the skipped row of zeta_X.json at
    // d05475b, 8, 36, 170, 560: not Weil, and its predictions at k = 5, 6 fail.
    let u = unibranch(r);
    assert_eq!(&u[..4], &[8, 36, 170, 560]);
    let cu = checks_of_counts(5, &u).unwrap().unwrap();
    assert!(!cu.weil);
    assert_eq!(cu.predicted, Some(false));
}

#[test]
fn x_at_3_and_23_refused() {
    // F mod 3 = (V - 1)^23 T (T - 1)^3 (design 5); G and its Moebius images degenerate too.
    // p = 3 waits for the model of WP-A step A0 (follow-up F-1).
    // At 23 the discriminant also vanishes identically for every model (review 1, 3.5). This
    // refusal is final (follow-up F-5, closed by the owner's decision of 2026-11-15): by Le Lay
    // (frontier-math-m23-cover-investigation at 3f6ab4b, Section 5 of paper/main.tex,
    // "Constructing the harmonic lift", paper/harmonic-construction.tex,
    // `sec:harmonic-construction`, with Lemma `lem:marked-distance`; Section 4, "Reduction at
    // 23", for the local structure) X acquires good reduction at 23 only over the tamely
    // ramified extension K(rho), rho^15 = pi, so there is no smooth model over F_23 to count on.
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
fn brute_force(f: &FormModP) -> u64 {
    let p = f.p();
    let eval = |g: &FormModP, x: u64, y: u64| -> u64 {
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
// C-5: the Frobenius module Jac[2] (design C57), and C-7: the JSON rows.
// ------------------------------------------------------------------------------------------

/// Every counted row of X (p = 5..131) and of the covers over L (p <= 127):
/// `(cover, p, s0, N_1..N_4)`.
/// Source: RustMath's tables after F-9 (`docs/workstreams/m23_golay/wpf/followup_f9/F9_impl.md`
/// Sections 3.1 and 3.2, RustMath at `88f9174`). The rows with p <= 79 of X and the L rows at
/// 7, 19, 31, 41, 43 equal the owner's `zeta_X.json` / `zeta_L.json` (checked by
/// `c5_rows_equal_the_owners`); the owner recounted X at 83, 97, 107
/// (`full_x_owner_recount_83_97_107`). The others are RustMath's counts, checked by the Weil,
/// mod 23 and mod 2 tests.
const ROWS: &[(char, u64, u64, [u64; 4])] = &[
    ('X', 5, 0, [7, 37, 169, 561]),
    ('X', 7, 0, [11, 69, 383, 2433]),
    ('X', 11, 0, [15, 133, 1368, 14905]),
    ('X', 13, 0, [11, 161, 2135, 29089]),
    ('X', 17, 0, [21, 285, 4923, 82985]),
    ('X', 19, 0, [17, 405, 7265, 130913]),
    ('X', 29, 0, [35, 789, 24773, 708785]),
    ('X', 31, 0, [33, 993, 29640, 926713]),
    ('X', 37, 0, [43, 1433, 50881, 1881609]),
    ('X', 41, 0, [42, 1780, 69408, 2827728]),
    ('X', 43, 0, [44, 1996, 80192, 3422024]),
    ('X', 47, 0, [47, 2349, 104321, 4882281]),
    ('X', 53, 0, [54, 2804, 148704, 7881808]),
    ('X', 59, 0, [72, 3560, 205056, 12120780]),
    ('X', 61, 0, [64, 3740, 226852, 13855856]),
    ('X', 67, 0, [80, 4460, 302192, 20163724]),
    ('X', 71, 0, [68, 5048, 358256, 25432076]),
    ('X', 73, 0, [83, 5541, 390089, 28393065]),
    ('X', 79, 0, [77, 6369, 491636, 38964585]),
    ('X', 83, 0, [91, 6885, 574375, 47483041]),
    ('X', 89, 0, [104, 7942, 707906, 62766210]),
    ('X', 97, 0, [101, 9385, 910820, 88546849]),
    ('X', 101, 0, [114, 10456, 1030926, 104090928]),
    ('X', 103, 0, [96, 10648, 1092594, 112536824]),
    ('X', 107, 0, [116, 11482, 1226360, 131139882]),
    ('X', 109, 0, [104, 11986, 1294118, 141216370]),
    ('X', 113, 0, [128, 13016, 1444142, 163058328]),
    ('X', 127, 0, [127, 16501, 2047933, 260199313]),
    ('X', 131, 0, [144, 17064, 2246976, 294472176]),
    ('L', 7, 5, [7, 45, 343, 2545]),
    ('L', 19, 3, [15, 413, 6849, 130169]),
    ('L', 31, 12, [43, 1017, 30265, 923729]),
    ('L', 41, 8, [40, 1790, 68866, 2828954]),
    ('L', 43, 15, [43, 1829, 78799, 3422769]),
    ('L', 47, 13, [54, 2316, 104076, 4889096]),
    ('L', 53, 21, [52, 2862, 150562, 7889314]),
    ('L', 53, 36, [57, 2925, 147507, 7890353]),
    ('L', 53, 41, [45, 2885, 147972, 7888001]),
    ('L', 61, 52, [51, 3909, 226185, 13840033]),
    ('L', 67, 15, [71, 4661, 302150, 20149585]),
    ('L', 67, 42, [62, 4614, 301154, 20166970]),
    ('L', 71, 37, [56, 5114, 358256, 25415418]),
    ('L', 71, 67, [78, 5210, 358566, 25433810]),
    ('L', 79, 15, [94, 6510, 493546, 38940122]),
    ('L', 79, 40, [82, 6316, 494788, 38952656]),
    ('L', 79, 46, [70, 6334, 492232, 38966242]),
    ('L', 89, 57, [94, 8088, 707974, 62746864]),
    ('L', 89, 69, [94, 8090, 705808, 62734682]),
    ('L', 97, 76, [103, 9497, 913861, 88569977]),
    ('L', 101, 74, [109, 10305, 1024111, 104049033]),
    ('L', 101, 98, [115, 10117, 1027081, 104049377]),
    ('L', 103, 93, [117, 10897, 1091211, 112578169]),
    ('L', 127, 63, [135, 16457, 2050377, 260132721]),
    ('L', 127, 74, [130, 16210, 2047378, 260120418]),
];

/// The expected analysis of a row (design C57 Section 4): the possible orders of Frobenius on
/// `Jac[2]`, the reason when decided, the types of `Jac[2]` and of `W` at the eigenvalue 1, and
/// the (8-letter, 15-point) cycle types of the classes of `F` on `W`. Computed by the design's
/// prototype (`design/proto/c5.py`, Python) and independently by its review (`review1/rowcheck.py`:
/// the conjugacy classes of `Sp(8, 2)` from GAP with `dim ker (M^k - 1)` by rank, and the
/// stable Lagrangians of the surviving classes); the valuations are PARI/GP's
/// (`tests/data/two_torsion/jac2_v2_pari.txt`, checked by `c5_bounds_equal_pari`).
struct E {
    cover: char,
    p: u64,
    s0: u64,
    orders: &'static [u64],
    reason: &'static str,
    j1: &'static [&'static [u32]],
    w1: &'static [&'static [u32]],
    classes: &'static [(&'static str, &'static str)],
}

#[rustfmt::skip]
const EXPECTED: &[E] = &[
    E { cover: 'X', p: 5, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 7, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 11, s0: 0, orders: &[15], reason: "P mod 2 squarefree", j1: &[], w1: &[], classes: &[("3 5", "15")] },
    E { cover: 'X', p: 13, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 17, s0: 0, orders: &[14], reason: "v2 #Jac(F_p) = 1", j1: &[&[2]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 19, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 29, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 31, s0: 0, orders: &[15], reason: "P mod 2 squarefree", j1: &[], w1: &[], classes: &[("3 5", "15")] },
    E { cover: 'X', p: 37, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 41, s0: 0, orders: &[6, 12], reason: "", j1: &[&[4], &[2, 2]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 43, s0: 0, orders: &[12], reason: "v2 #Jac(F_p) = 1", j1: &[&[4]], w1: &[&[2]], classes: &[("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 47, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 53, s0: 0, orders: &[6, 12], reason: "", j1: &[&[4], &[2, 2]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 59, s0: 0, orders: &[10], reason: "v2 #Jac(F_p^5) = 4", j1: &[], w1: &[], classes: &[("1^3 5", "5^3")] },
    E { cover: 'X', p: 61, s0: 0, orders: &[6, 12], reason: "", j1: &[&[4], &[2, 2], &[2, 1, 1]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 67, s0: 0, orders: &[10], reason: "v2 #Jac(F_p^5) = 4", j1: &[], w1: &[], classes: &[("1^3 5", "5^3")] },
    E { cover: 'X', p: 71, s0: 0, orders: &[10], reason: "v2 #Jac(F_p^5) = 4", j1: &[], w1: &[], classes: &[("1^3 5", "5^3")] },
    E { cover: 'X', p: 73, s0: 0, orders: &[14], reason: "v2 #Jac(F_p) = 1", j1: &[&[2]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 79, s0: 0, orders: &[15], reason: "P mod 2 squarefree", j1: &[], w1: &[], classes: &[("3 5", "15")] },
    E { cover: 'X', p: 83, s0: 0, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 89, s0: 0, orders: &[8], reason: "v2 #Jac(F_p) = 1", j1: &[&[8]], w1: &[&[4]], classes: &[("1^2 2 4", "1 2 4^3")] },
    E { cover: 'X', p: 97, s0: 0, orders: &[15], reason: "P mod 2 squarefree", j1: &[], w1: &[], classes: &[("3 5", "15")] },
    E { cover: 'X', p: 101, s0: 0, orders: &[3, 6, 12], reason: "", j1: &[&[4], &[2, 2], &[2, 1, 1], &[1, 1, 1, 1]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 103, s0: 0, orders: &[12], reason: "v2 #Jac(F_p) = 1", j1: &[&[4]], w1: &[&[2]], classes: &[("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 107, s0: 0, orders: &[4, 8], reason: "", j1: &[&[8], &[6, 2], &[6, 1, 1], &[4, 4], &[4, 2, 2], &[4, 2, 1, 1], &[4, 1, 1, 1, 1], &[3, 3, 2], &[3, 3, 1, 1]], w1: &[&[4], &[3, 1], &[2, 2], &[2, 1, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("1^4 2^2", "1^3 2^6"), ("2^4", "1^7 2^4"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'X', p: 109, s0: 0, orders: &[8], reason: "v2 #Jac'(F_p^4) = 6", j1: &[&[8], &[6, 2], &[6, 1, 1]], w1: &[&[4], &[3, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'X', p: 113, s0: 0, orders: &[12], reason: "v2 #Jac(F_p) = 1", j1: &[&[4]], w1: &[&[2]], classes: &[("2 6", "1 2 3^2 6")] },
    E { cover: 'X', p: 127, s0: 0, orders: &[14], reason: "v2 #Jac(F_p) = 1", j1: &[&[2]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'X', p: 131, s0: 0, orders: &[6, 12], reason: "", j1: &[&[4], &[2, 2], &[2, 1, 1]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'L', p: 7, s0: 5, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 19, s0: 3, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 31, s0: 12, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 41, s0: 8, orders: &[8], reason: "v2 #Jac'(F_p^4) = 6", j1: &[&[8], &[6, 2], &[6, 1, 1]], w1: &[&[4], &[3, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'L', p: 43, s0: 15, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 47, s0: 13, orders: &[12], reason: "v2 #Jac(F_p) = 1", j1: &[&[4]], w1: &[&[2]], classes: &[("2 6", "1 2 3^2 6")] },
    E { cover: 'L', p: 53, s0: 21, orders: &[8], reason: "v2 #Jac(F_p) = 1", j1: &[&[8]], w1: &[&[4]], classes: &[("1^2 2 4", "1 2 4^3")] },
    E { cover: 'L', p: 53, s0: 36, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 53, s0: 41, orders: &[15], reason: "P mod 2 squarefree", j1: &[], w1: &[], classes: &[("3 5", "15")] },
    E { cover: 'L', p: 61, s0: 52, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 67, s0: 15, orders: &[15], reason: "P mod 2 squarefree", j1: &[], w1: &[], classes: &[("3 5", "15")] },
    E { cover: 'L', p: 67, s0: 42, orders: &[4, 8], reason: "", j1: &[&[8], &[6, 2], &[4, 4]], w1: &[&[4], &[3, 1], &[2, 2]], classes: &[("1^2 2 4", "1 2 4^3"), ("1^4 2^2", "1^3 2^6"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'L', p: 71, s0: 37, orders: &[4, 8], reason: "", j1: &[&[8], &[6, 2], &[6, 1, 1], &[4, 4], &[4, 2, 2], &[3, 3, 2]], w1: &[&[4], &[3, 1], &[2, 2], &[2, 1, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("1^4 2^2", "1^3 2^6"), ("2^4", "1^7 2^4"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'L', p: 71, s0: 67, orders: &[4, 8], reason: "", j1: &[&[8], &[6, 2], &[4, 4]], w1: &[&[4], &[3, 1], &[2, 2]], classes: &[("1^2 2 4", "1 2 4^3"), ("1^4 2^2", "1^3 2^6"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'L', p: 79, s0: 15, orders: &[2, 4, 8], reason: "", j1: &[&[8], &[6, 2], &[6, 1, 1], &[4, 4], &[4, 2, 2], &[4, 2, 1, 1], &[3, 3, 2], &[3, 3, 1, 1], &[2, 2, 2, 2]], w1: &[&[4], &[3, 1], &[2, 2], &[2, 1, 1], &[1, 1, 1, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("1^4 2^2", "1^3 2^6"), ("1^8", "1^15"), ("2^4", "1^7 2^4"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'L', p: 79, s0: 40, orders: &[6, 12], reason: "", j1: &[&[4], &[2, 2], &[2, 1, 1]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'L', p: 79, s0: 46, orders: &[8], reason: "v2 #Jac(F_p) = 1", j1: &[&[8]], w1: &[&[4]], classes: &[("1^2 2 4", "1 2 4^3")] },
    E { cover: 'L', p: 89, s0: 57, orders: &[3, 6, 12], reason: "", j1: &[&[4], &[2, 2], &[2, 1, 1], &[1, 1, 1, 1]], w1: &[&[2], &[1, 1]], classes: &[("1^2 3^2", "1^3 3^4"), ("2 6", "1 2 3^2 6")] },
    E { cover: 'L', p: 89, s0: 69, orders: &[8], reason: "v2 #Jac'(F_p^4) = 6", j1: &[&[8], &[6, 2]], w1: &[&[4], &[3, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("4^2", "1^3 2^2 4^2")] },
    E { cover: 'L', p: 97, s0: 76, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 101, s0: 74, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 101, s0: 98, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 103, s0: 93, orders: &[14], reason: "v2 #Jac(F_p) = 1", j1: &[&[2]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 127, s0: 63, orders: &[7, 14], reason: "", j1: &[&[2], &[1, 1]], w1: &[&[1]], classes: &[("1 7", "1 7^2")] },
    E { cover: 'L', p: 127, s0: 74, orders: &[4, 8], reason: "", j1: &[&[8], &[6, 2], &[6, 1, 1], &[4, 4], &[4, 2, 2], &[4, 2, 1, 1], &[4, 1, 1, 1, 1], &[3, 3, 2], &[3, 3, 1, 1]], w1: &[&[4], &[3, 1], &[2, 2], &[2, 1, 1]], classes: &[("1^2 2 4", "1 2 4^3"), ("1^4 2^2", "1^3 2^6"), ("2^4", "1^7 2^4"), ("4^2", "1^3 2^2 4^2")] },
];

fn row_frob(cover: char, p: u64, s0: u64) -> FrobeniusPolynomial {
    let r = ROWS
        .iter()
        .find(|r| r.0 == cover && r.1 == p && r.2 == s0)
        .unwrap_or_else(|| panic!("row {cover} {p} {s0}"));
    FrobeniusPolynomial::from_counts_u64(p, &r.3).unwrap()
}

fn x_frob(p: u64) -> FrobeniusPolynomial {
    row_frob('X', p, 0)
}

fn rank_bound(k: u64, twisted: bool, v2: u32) -> RankBound {
    RankBound { k, twisted, v2 }
}

#[test]
fn c5_rows_equal_the_owners() {
    // The constant table equals zeta_X.json and zeta_L.json wherever the owner has the row.
    let mut n = 0;
    for r in zeta_x() {
        let row = ROWS.iter().find(|x| x.0 == 'X' && x.1 == r.p).expect("row");
        assert_eq!(&row.3[..], &r.counts.unwrap()[..], "X {}", r.p);
        n += 1;
    }
    let (l, _) = zeta_l();
    for r in l.iter().filter(|r| r.ok) {
        let s0 = r.s0.unwrap();
        let row = ROWS
            .iter()
            .find(|x| x.0 == 'L' && x.1 == r.p && x.2 == s0)
            .expect("row");
        assert_eq!(&row.3[..], &r.counts.clone().unwrap()[..], "L {} {s0}", r.p);
        n += 1;
    }
    assert_eq!(n, 24);
    assert_eq!(ROWS.len(), 54);
    assert_eq!(EXPECTED.len(), 54);
}

#[test]
fn c5_acceptance() {
    // REQUESTS C-5: with k = 1 alone (v2 #Jac(F_p) = 1) the order is 14 at 17, 73, 127, 12 at
    // 43, 103, 113 and 8 at 89; the four order 15 rows are unchanged (P mod 2 squarefree).
    for (p, order) in [
        (17, 14),
        (73, 14),
        (127, 14),
        (43, 12),
        (103, 12),
        (113, 12),
        (89, 8),
    ] {
        assert_eq!(
            x_frob(p).two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByTwoRank {
                order,
                bounds: vec![rank_bound(1, false, 1)],
            },
            "p = {p}"
        );
    }
    for p in [11, 31, 79, 97] {
        assert_eq!(
            x_frob(p).two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByP(15),
            "p = {p}"
        );
    }
}

#[test]
fn c5_beyond_requests_1_2() {
    // Decided by k > 1 or by a quadratic twist (design C57 4.4; PARI/GP valuations quoted from
    // jac2_v2_pari.txt: X 59 "k5:4/4", X 109 "k4:13/6", L 41 8 "k4:18/6", L 89 69 "k4:10/6").
    for p in [59, 67, 71] {
        assert_eq!(
            x_frob(p).two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByTwoRank {
                order: 10,
                bounds: vec![rank_bound(5, false, 4)],
            },
            "p = {p}"
        );
    }
    for fp in [x_frob(109), row_frob('L', 41, 8), row_frob('L', 89, 69)] {
        assert_eq!(
            fp.two_torsion_frobenius_order(),
            TwoTorsionOrder::DeterminedByTwoRank {
                order: 8,
                bounds: vec![rank_bound(4, true, 6)],
            },
            "p = {}",
            fp.p()
        );
    }
    let j = jac2_of(&x_frob(109), true, true);
    assert_eq!(j.w_types_one(), vec![vec![4], vec![3, 1]]);
    match x_frob(107).two_torsion_frobenius_order() {
        TwoTorsionOrder::Unresolved { candidates, .. } => assert_eq!(candidates, vec![4, 8]),
        o => panic!("107: {o:?}"),
    }
}

#[test]
fn c5_order_text_says_what_was_tried() {
    // REQUESTS Section 4: an Unresolved order says what was tried (C57 review 2, n1). The
    // orders are those of EXPECTED; N_max is PARI's (jac2_v2_pari.txt: X 7 "N=14", X 107 "N=8").
    let text = |p: u64, decided: bool| jac2_of(&x_frob(p), true, decided).order_text();
    assert_eq!(text(11, true), "15 (Decided; P mod 2 squarefree)");
    assert_eq!(text(17, true), "14 (Decided; v2 #Jac(F_p) = 1)");
    assert_eq!(
        text(17, false),
        "14 (conditional on the counts; v2 #Jac(F_p) = 1)"
    );
    assert_eq!(
        text(7, true),
        "Unresolved: 7 or 14 (tried: v2 #Jac(F_p^k) and v2 #Jac'(F_p^k) for every k | 14)"
    );
    assert_eq!(
        text(107, true),
        "Unresolved: 4 or 8 (tried: v2 #Jac(F_p^k) and v2 #Jac'(F_p^k) for every k | 8)"
    );
    assert_eq!(
        jac2_of(&x_frob(7), false, true).order_text(),
        "not applicable (P is not a Weil polynomial)"
    );
    // Skipped bounds are named. P = T^2 - 2T + q, q = 21 * 2^128 + 1 prime, the Frobenius
    // polynomial of an ordinary elliptic curve (a = 2 prime to q, |a| <= 2 sqrt(q)), with
    // P mod 2 = (x + 1)^2. PARI/GP (tests/data/two_torsion/skipped_ec.gp, skipped_ec_pari.txt):
    // v2 #E(F_q) = 128 with elementary divisors 2^0, 2^128; v2 #E(F_q^2) = 130 (2^1, 2^129);
    // v2 #E'(F_q) = 2, v2 #E'(F_q^2) = 2. Both module types of E[2] pass the twisted bounds.
    let q = Integer::from(21) * Integer::from(2).pow(128) + Integer::from(1);
    let fp = FrobeniusPolynomial::from_coefficients(
        &q,
        vec![Integer::from(1), Integer::from(-2), q.clone()],
    )
    .unwrap();
    assert!(fp.is_weil());
    let j = jac2_of(&fp, true, true);
    let a = j.analysis.as_ref().unwrap();
    assert_eq!(a.skipped_bounds, vec![(1, false), (2, false)]);
    assert_eq!(
        a.bounds,
        vec![rank_bound(1, true, 2), rank_bound(2, true, 2)]
    );
    assert_eq!(
        j.order_text(),
        "Unresolved: 1 or 2 (tried: v2 #Jac(F_p^k) and v2 #Jac'(F_p^k) for every k | 2; \
         skipped, only v2 >= 128 seen: #Jac(F_p), #Jac(F_p^2))"
    );
    assert_eq!(j.json()["order_candidates"], serde_json::json!([1, 2]));
}

#[test]
fn c5_bounds_equal_pari() {
    // Every bound v2 #Jac(F_p^k), v2 #Jac'(F_p^k), k | N_max, on all 54 rows equals PARI/GP's
    // (polresultant; tests/data/two_torsion/v2.gp, jac2_v2_pari.txt).
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/two_torsion/jac2_v2_pari.txt"),
    )
    .unwrap();
    let mut n = 0;
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let cover = w[0].chars().next().unwrap();
        let (p, s0): (u64, u64) = (w[1].parse().unwrap(), w[2].parse().unwrap());
        let nmax: u64 = w[3].trim_start_matches("N=").parse().unwrap();
        let fp = row_frob(cover, p, s0);
        let a = fp.two_torsion_analysis().unwrap();
        assert_eq!(a.max_order, nmax, "{line}");
        let mut expect = Vec::new();
        for kv in &w[4..] {
            let (k, v) = kv.trim_start_matches('k').split_once(':').unwrap();
            let (u, t) = v.split_once('/').unwrap();
            expect.push((
                k.parse::<u64>().unwrap(),
                u.parse::<u32>().unwrap(),
                t.parse::<u32>().unwrap(),
            ));
        }
        if a.factors.iter().all(|f| f.multiplicity == 1) {
            assert!(a.bounds.is_empty(), "{line}: squarefree, no bound computed");
        } else {
            let mut got: Vec<(u64, bool, u32)> =
                a.bounds.iter().map(|b| (b.k, b.twisted, b.v2)).collect();
            got.sort();
            let mut want: Vec<(u64, bool, u32)> = expect
                .iter()
                .flat_map(|&(k, u, t)| [(k, false, u), (k, true, t)])
                .collect();
            want.sort();
            assert_eq!(got, want, "{line}");
        }
        n += 1;
    }
    assert_eq!(n, 54);
}

#[test]
fn c5_matches_the_owners_predictions() {
    // data/a8_frobenius_predictions.json (johnjanik/M23 at e0d25b8, vendored): for every X row,
    // #Jac(F_p) and its v2 equal his; the Q candidates are his pair; where he decides the order,
    // ours is equal; our cycle types on W (8 letters, 15 points) are among his, and equal
    // except at 107 and 109, where the k > 1 and twisted bounds narrow them.
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(data_dir().join("a8_frobenius_predictions.json")).unwrap(),
    )
    .unwrap();
    let primes = v["primes"].as_object().unwrap();
    assert_eq!(primes.len(), 29);
    for (ps, r) in primes {
        let p: u64 = ps.parse().unwrap();
        let fp = x_frob(p);
        let n = fp.jacobian_order(1);
        assert_eq!(
            n,
            Integer::from(r["order_of_Jac_Fp"].as_u64().unwrap()),
            "p = {p}"
        );
        assert_eq!(
            n.valuation(&Integer::from(2)) as u64,
            r["v2_order_of_Jac_Fp"].as_u64().unwrap()
        );
        let mut theirs: Vec<String> = r["Q_or_reverse"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        theirs.sort();
        theirs.dedup();
        let mut ours: Vec<String> = fp
            .reciprocal_halves_mod2()
            .iter()
            .map(|q| rustmath_curves::weil::f2_poly_string(q))
            .collect();
        ours.sort();
        assert_eq!(ours, theirs, "p = {p}");
        let j = jac2_of(&fp, true, true);
        if let Some(o) = r.get("order_on_Jac2") {
            assert_eq!(j.decided_order(), Some(o.as_u64().unwrap()), "p = {p}");
        }
        let strs = |k: &str| -> Vec<String> {
            r[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        };
        let (c8, c15) = (strs("cycle_types_8"), strs("cycle_types_15"));
        let mut o8: Vec<String> = j.w_cycle_types().iter().map(|c| c.0.clone()).collect();
        let mut o15: Vec<String> = j.w_cycle_types().iter().map(|c| c.1.clone()).collect();
        o8.sort();
        o8.dedup();
        o15.sort();
        o15.dedup();
        assert!(
            o8.iter().all(|c| c8.contains(c)),
            "p = {p}: {o8:?} vs {c8:?}"
        );
        assert!(
            o15.iter().all(|c| c15.contains(c)),
            "p = {p}: {o15:?} vs {c15:?}"
        );
        if p != 107 && p != 109 {
            assert_eq!(o8, c8, "p = {p}");
            assert_eq!(o15, c15, "p = {p}");
        } else {
            assert!(o8.len() < c8.len(), "p = {p}");
        }
    }
}

/// The checks of the analysis of one row against its expected values (shared by `c5_all_rows`
/// and `full_x_table_k4`).
fn assert_jac2(j: &Jac2, e: &E) {
    let what = format!("{} {} {}", e.cover, e.p, e.s0);
    let a = j.analysis.as_ref().unwrap();
    assert_eq!(a.orders(), e.orders.to_vec(), "{what}: orders");
    assert_eq!(j.reason(), e.reason, "{what}: reason");
    let j1: Option<Vec<Vec<u32>>> =
        (!e.j1.is_empty()).then(|| e.j1.iter().map(|t| t.to_vec()).collect());
    assert_eq!(a.eigenvalue_one_types(), j1, "{what}: J at 1");
    let w1: Vec<Vec<u32>> = if e.w1.is_empty() {
        vec![vec![]]
    } else {
        e.w1.iter().map(|t| t.to_vec()).collect()
    };
    assert_eq!(j.w_types_one(), w1, "{what}: W at 1");
    let classes: Vec<(String, String)> = e
        .classes
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    assert_eq!(j.w_cycle_types(), classes, "{what}: classes");
    assert!(j.w_error.is_none(), "{what}");
}

#[test]
fn c5_all_rows() {
    // The full table of design C57 Section 4 on all 54 counted rows (every one Decided).
    for e in EXPECTED {
        let fp = row_frob(e.cover, e.p, e.s0);
        assert!(fp.is_weil());
        let j = jac2_of(&fp, true, true);
        assert_jac2(&j, e);
        let decided = e.orders.len() == 1;
        assert_eq!(
            j.order_status(),
            if decided { "Decided" } else { "Unresolved" }
        );
        // A Decided order needs a Decided row: the same P on a row that is not Decided is only
        // conditional on the counts.
        let jc = jac2_of(&fp, true, false);
        assert_eq!(
            jc.order_status(),
            if decided {
                "conditional on the counts"
            } else {
                "Unresolved"
            }
        );
    }
}

/// Every key of the owner's row is in ours with the same value.
fn assert_owner_keys(ours: &serde_json::Value, owner: &serde_json::Value) {
    for (k, v) in owner.as_object().unwrap() {
        assert_eq!(ours.get(k), Some(v), "p = {}: key {k}", owner["p"]);
    }
}

#[test]
fn json_rows_carry_the_owners_keys() {
    // row_json (REQUESTS C-7) at X 5 (A_3, delta 62), 7 and 11 (62 nodes), counted in debug with
    // k = 4, and at 23 (refused): every key of the owner's rows of zeta_X.json (2d11038) is
    // present with the same value, the order of "P mod 2" and "Q candidates" included; the
    // legacy keys of rustmath_wpf_tables.json and jac2 are there.
    let owner: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(data_dir().join("zeta_X.json")).unwrap())
            .unwrap();
    let models = cc0_models();
    for p in [5u64, 7, 11] {
        let row = x_row(&models, p, 4, &exact());
        let ours = row_json(&row, None);
        let theirs = owner
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["p"] == p)
            .unwrap();
        assert_owner_keys(&ours, theirs);
        for k in [
            "a",
            "P_mod_2",
            "Q",
            "model",
            "rustmath_status",
            "irreducibility",
            "jac2",
            "weil",
            "trusted_hypotheses",
        ] {
            assert!(ours.get(k).is_some(), "p = {p}: {k}");
        }
        let a: Vec<i64> = theirs["P"].as_array().unwrap()[1..5]
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect();
        assert_eq!(ours["a"], serde_json::json!(a));
        assert_eq!(ours["Q"], theirs["Q candidates"][0]);
        assert_eq!(ours["decided"], serde_json::json!(true));
    }
    let row = x_row(&models, 23, 1, &exact());
    let ours = row_json(&row, None);
    assert_eq!(ours["status"], "refused");
    assert_eq!(ours["model"], "all");
    assert_eq!(
        ours["reason"],
        "model degenerate at this prime: the discriminant vanishes identically mod p"
    );
    assert!(ours.get("Q").is_none() && ours.get("counts").is_none());
    // With k = 3 there is no P, so no analysis: jac2 is null.
    let row = x_row(&models, 17, 3, &exact());
    assert_eq!(row_json(&row, None)["jac2"], serde_json::Value::Null);
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
        assert!(r.status.is_decided(), "F at p = {}", reference.p);
        assert_eq!(
            &owner_types(&r.locus),
            reference.singular_points.as_ref().unwrap(),
            "F at p = {}",
            reference.p
        );
    }
}

fn l_row(calf: &[Vec<[Integer; 6]>], p: u64, s0: u64, kmax: u32) -> Row {
    let result = calf_mod(calf, p, s0).and_then(|f| {
        rustmath_curves::nodal_biprojective::normalization_point_counts(&f, GENUS, kmax, &exact())
    });
    assert_genus_consistent(&result, &format!("calF at p = {p}, s0 = {s0}"));
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
    // zeta_L.json (2d11038): refusals at (5, 3), (11, 2), (23, 5), (23, 15); Decided: nodal at
    // 7, 19, 41, 43 with the table values; at (31, 12) Decided by the delta-sum certificate (the
    // owner's decision on F-9) with the corrected row (F-8). The singular points of every counted
    // row equal the owner's.
    let dir = require_m23isgalois();
    let calf = model_calf(&dir);
    let (rows, _) = zeta_l();
    for reference in &rows {
        let s0 = reference.s0.unwrap();
        assert!(degree_one_primes(reference.p).contains(&s0));
        let row = l_row(&calf, reference.p, s0, 4);
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
        assert_eq!(
            &owner_types(&r.locus),
            reference.singular_points.as_ref().unwrap(),
            "p = {}",
            reference.p
        );
        if reference.p == 31 {
            // 57 nodes, an ordinary triple point (one rational and two conjugate tangents) and a
            // tacnode with conjugate branches.
            let mut t = delta_sum62(r);
            t.sort_by_key(|x| format!("{x:?}"));
            assert_eq!(
                t,
                vec![
                    (LocalType::A(3), Some(vec![2])),
                    (LocalType::OrdinaryMultiple(3), Some(vec![1, 2])),
                ]
            );
            assert_eq!(r.locus.node_count(), 57);
            assert_eq!(
                r.status.label(),
                "Decided: delta-sum, Lemma 9.1 (57 nodes, ordinary 3-fold point [1, 2]; A3 \
                 conjugate branches)"
            );
            assert_eq!(r.counts, vec![43, 1017, 30265, 923729]);
            assert_eq!(&r.counts, reference.counts.as_ref().unwrap());
            let c = checks(r).unwrap().unwrap();
            assert_eq!(
                c.frob.coefficients(),
                &ints(reference.p_coeffs.as_ref().unwrap())[..]
            );
            assert_eq!(c.frob.coefficients()[1..5], ints(&[11, 88, 682, 4438])[..]);
            assert!(c.weil);
            assert_eq!(c.mod23, (0, 0));
            // The reference rule gives the row of zeta_L.json at d05475b, 44, 1014, 30266,
            // 923726: a Weil polynomial that fails the mod 23 check.
            let u = unibranch(r);
            assert_eq!(u, vec![44, 1014, 30266, 923726]);
            let cu = checks_of_counts(31, &u).unwrap().unwrap();
            assert!(cu.weil, "the reference P is Weil");
            assert_eq!(cu.mod23, (15, 7), "but fails the mod 23 check");
            let e = EXPECTED
                .iter()
                .find(|e| e.cover == 'L' && e.p == 31 && e.s0 == 12)
                .unwrap();
            assert_jac2(&c.jac2, e);
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
        // C-5: the analysis of Jac[2] equals the table of c5_all_rows (at (41, 8) the order 8 is
        // decided by the quadratic twist over F_(p^4)).
        let e = EXPECTED
            .iter()
            .find(|e| e.cover == 'L' && e.p == reference.p && e.s0 == s0)
            .unwrap();
        assert_jac2(&c.jac2, e);
        assert!(c.jac2.row_decided);
        // C-7: the JSON row carries the owner's keys with his values (his L rows have no
        // `certified` and no `chi_minus23`).
        let ours = row_json(&row, None);
        for k in [
            "p",
            "s0",
            "status",
            "counts",
            "P",
            "P(chi) mod 23",
            "P(chi p) mod 23",
            "P mod 2",
            "Q candidates",
            "golay_split",
        ] {
            let theirs = &zeta_l_json(reference.p, s0)[k];
            assert_eq!(&ours[k], theirs, "L {} {s0}: {k}", reference.p);
        }
    }
}

/// The owner's row of `zeta_L.json` at `(p, s0)`.
fn zeta_l_json(p: u64, s0: u64) -> serde_json::Value {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(data_dir().join("zeta_L.json")).unwrap())
            .unwrap();
    v["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["p"] == p && r["s0"] == s0)
        .unwrap()
        .clone()
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
        let row = l_row(&calf, 53, s0, 1);
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
    let r = l_row(&calf, 31, 12, 5);
    let r = counts_of(&r);
    assert!(r.status.is_decided());
    assert_eq!(r.counts[4], 28604633);
    assert_eq!(r.predicted_counts_agree(), Some(true));
    let u = unibranch(r);
    assert_eq!(u[4], 28604634);
    let fu = FrobeniusPolynomial::from_counts_u64(31, &u[..GENUS]).unwrap();
    assert_eq!(fu.point_counts(5)[4], Integer::from(28598074));
    assert!(!fu.predicts(&ints(&u.iter().map(|&n| n as i64).collect::<Vec<_>>())));
}

// ------------------------------------------------------------------------------------------
// Opt-in release runs on the CC0 models.
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "release, about 1.5 min on 4 threads"]
fn full_x_table_k4() {
    // All 19 rows of zeta_X.json (2d11038) at k = 4 from [G, G7, G13]: Decided by the delta-sum
    // certificate at 5 (A3, conjugate branches) and at 13, 31, 41, 73 (one cusp each), by the
    // nodal certificate elsewhere; the singular points equal the owner's on every row.
    let models = cc0_models();
    let rows = zeta_x();
    assert_eq!(rows.len(), 19);
    for reference in &rows {
        let row = x_row(&models, reference.p, 4, &exact());
        let r = counts_of(&row);
        if reference.p == 5 {
            assert_eq!(delta_sum62(r), vec![(LocalType::A(3), Some(vec![2]))]);
        } else if [13, 31, 41, 73].contains(&reference.p) {
            assert_eq!(delta_sum62(r), vec![(LocalType::A(2), Some(vec![1]))]);
        } else {
            nodal62(r);
        }
        assert_eq!(
            &owner_types(&r.locus),
            reference.singular_points.as_ref().unwrap(),
            "p = {}",
            reference.p
        );
        assert_matches_table(r, reference);
        assert_eq!(&r.counts, reference.counts.as_ref().unwrap());
        // C-5: the analysis of Jac[2] of the counted row equals the table of c5_all_rows.
        let c = checks(r).unwrap().unwrap();
        let e = EXPECTED
            .iter()
            .find(|e| e.cover == 'X' && e.p == reference.p)
            .unwrap();
        assert_jac2(&c.jac2, e);
        assert!(c.jac2.row_decided);
    }
}

#[test]
#[ignore = "release, about 2.5 min on 4 threads"]
fn full_x_owner_recount_83_97_107() {
    // The owner recounted three rows of RustMath's extended table with his implementation
    // (johnjanik/M23 at 2d11038, data/rustmath_crosscheck.json): 83: 91, 6885, 574375,
    // 47483041; 97: 101, 9385, 910820, 88546849; 107: 116, 11482, 1226360, 131139882. The paper
    // (paper/golay_m23.tex at 2d11038): "at 107 the model has 60 nodes and a tacnode".
    let models = cc0_models();
    let cases: [(u64, [u64; 4]); 3] = [
        (83, [91, 6885, 574375, 47483041]),
        (97, [101, 9385, 910820, 88546849]),
        (107, [116, 11482, 1226360, 131139882]),
    ];
    for (p, owner) in cases {
        let row = x_row(&models, p, 4, &exact());
        let r = counts_of(&row);
        assert_eq!(r.counts, owner, "p = {p}");
        if p == 107 {
            assert_eq!(delta_sum62(r), vec![(LocalType::A(3), Some(vec![1, 1]))]);
            assert_eq!(r.locus.node_count(), 60);
            assert_eq!(
                r.status.label(),
                "Decided: delta-sum, Lemma 9.1 (60 nodes, A3 rational branches)"
            );
        } else {
            nodal62(r);
        }
    }
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
            let forms: Vec<(String, FormModP)> = models
                .iter()
                .filter_map(|(n, g)| g.reduce_primitive(p).ok().map(|f| (n.to_string(), f)))
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

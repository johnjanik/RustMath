//! Point counts on small curves in `P^1 x P^1` against PARI/GP (MAGMA ch. 114 `LPolynomial`,
//! `IsNode`, `IsCusp`, `IsOrdinarySingularity`; digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`).
//!
//! The curves `V^2 + b(T) V + c(T) = 0` have bidegree (4,2) (fibre variable `X = T`, degree 4;
//! `Y = V`). Their singular points are those of `y^2 = D(T)`, `D = b^2 - 4c`, `y = 2V + b`, and the
//! normalization is `y^2 = D / (square part)`. Reference values: PARI/GP 2.15
//! `hyperellcharpoly` of the normalization, and `ellap`. The scripts and their output are in
//! `docs/workstreams/m23_golay/wpf/gp/`: `small_curves.gp` and `branch.gp` (curves A to I2, E1,
//! E2; reproduced in the WP-F design review), `tangent.gp` (curve T3), and `small_curves4.gp`,
//! which reads `small_curves2.gp` (curves J, J', J2, K1, K2). The PARI output is quoted at each
//! case. `P` is descending: `[1, a_1, ..., p^g]`.
//!
//! The curves of WP-F code review 1, Section 2.2 are `X^2 + b(Y) X + c(Y) = 0`, of bidegree
//! (2, dy), found by the review's search script (`adv2.gp`, not kept). Their reference values
//! were rerun from `b` and `c` alone by `verify.gp` in the same folder (PARI/GP 2.15.4).
//!
//! Statuses follow Lemma 9.1 of the owner's paper (johnjanik/M23 at `2d11038`,
//! `paper/golay_m23.tex`, `\label{lem:reduction}`): Decided needs the delta-sum `p_a - g` and
//! the absolute irreducibility of `F mod p`. By the owner's decision on follow-up F-9
//! (2026-11-15) both the nodal and the delta-sum certificate are Decided. The reducible
//! reductions of [`reducible_reductions_are_never_decided`] come from the F-9 design, Section
//! 3.3, with their PARI values from `regress.gp` and `f9_tests.gp` (same folder).
//! MAGMA ch. 114.8.1 `IsAbsolutelyIrreducible`.

// The case tables are tuples on purpose: one line per curve, as in the PARI/GP scripts.
#![allow(clippy::type_complexity)]

use rustmath_curves::nodal_biprojective::{
    absolute_irreducibility, certify, normalization_point_counts, plane_point_count,
    singular_locus, AbsoluteWitness, BasePoint, BihomogeneousForm, BranchRule, Certificate,
    CountError, CountOptions, Degeneracy, Fibration, FormModP, Irreducibility,
    IrreducibilityEvidence, IrreducibilityGap, IrreducibleFibre, LocalType, NormalizationCounts,
    ReductionStatus, Unresolved,
};
use rustmath_curves::weil::FrobeniusPolynomial;
use rustmath_integers::Integer;

/// `a(T) V^2 + b(T) V + c(T)` (ascending in `T`, degree <= 4) as a (4,2) form mod p.
fn quad_form(p: u64, a: &[i64], b: &[i64], c: &[i64]) -> FormModP {
    let mut cols = vec![vec![0u64; 3]; 5];
    for (j, poly) in [c, b, a].iter().enumerate() {
        for (i, &x) in poly.iter().enumerate() {
            cols[i][j] = x.rem_euclid(p as i64) as u64;
        }
    }
    FormModP::new(p, cols).unwrap()
}

/// `X^2 + b(Y) X + c(Y)` (ascending in `Y`, `b` and `c` of length `dy + 1`) as a (2, dy) form
/// mod p.
fn monic_quadratic_in_x(p: u64, b: &[u64], c: &[u64]) -> FormModP {
    let mut a = vec![0u64; b.len()];
    a[0] = 1;
    FormModP::new(p, vec![c.to_vec(), b.to_vec(), a]).unwrap()
}

/// Brute-force `#X0(F_p)` over `P^1 x P^1(F_p)` (independent of the fibre counter).
fn brute_force(f: &FormModP) -> u64 {
    let p = f.p();
    let eval = |g: &FormModP, x: u64, y: u64| -> u64 {
        let mut acc = 0u64;
        for (i, row) in g.coeffs().iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                let t = c * pow(x, i as u64, p) % p * pow(y, j as u64, p) % p;
                acc = (acc + t) % p;
            }
        }
        acc
    };
    let rx = f.reverse_x();
    let ry = f.reverse_y();
    let rxy = rx.reverse_y();
    let mut n = 0;
    for x in 0..p {
        for y in 0..p {
            n += u64::from(eval(f, x, y) == 0);
        }
        n += u64::from(eval(&ry, x, 0) == 0); // (x, inf)
    }
    for y in 0..p {
        n += u64::from(eval(&rx, 0, y) == 0); // (inf, y)
    }
    n + u64::from(eval(&rxy, 0, 0) == 0)
}

fn pow(b: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1 % p;
    let mut b = b % p;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % p;
        }
        b = b * b % p;
        e >>= 1;
    }
    r
}

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

/// Count with the exact rule up to `kmax`, check `P` against PARI, the predictions beyond `g`,
/// the Weil test, both fibrations and brute force; return the result for further checks.
fn check(
    name: &str,
    f: &FormModP,
    genus: usize,
    kmax: u32,
    expect_p: &[i64],
) -> rustmath_curves::nodal_biprojective::NormalizationCounts {
    let r = normalization_point_counts(f, genus, kmax, &CountOptions::default())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(r.counts.len(), kmax as usize, "{name}: counts");
    let fp = r.frobenius_polynomial().unwrap().unwrap();
    assert_eq!(fp.coefficients(), &ints(expect_p)[..], "{name}: P = {fp}");
    assert!(fp.is_weil(), "{name}: Weil");
    if kmax as usize > genus {
        assert_eq!(
            r.predicted_counts_agree(),
            Some(true),
            "{name}: predictions"
        );
    }
    // both fibrations and brute force give the same plane count
    let n1 = r.plane_counts[0];
    assert_eq!(
        plane_point_count(f, 1, Fibration::FibresInY).unwrap(),
        n1,
        "{name}"
    );
    assert_eq!(brute_force(f), n1, "{name}: brute force");
    if kmax >= 2 {
        assert_eq!(
            plane_point_count(f, 2, Fibration::FibresInY).unwrap(),
            r.plane_counts[1],
            "{name}: T-fibration k = 2"
        );
    }
    r
}

/// Decided by the nodal certificate with `nodes` nodes over `F_p-bar`; exact counts. Returns the
/// irreducibility evidence.
fn decided_nodal(r: &NormalizationCounts, nodes: usize) -> IrreducibilityEvidence {
    assert_eq!(r.branch_rule, BranchRule::Exact);
    match &r.status {
        ReductionStatus::Decided(Certificate::Nodal {
            nodes: n,
            genus_hypothesis,
            irreducibility,
            ..
        }) => {
            assert_eq!(*n, nodes);
            assert_eq!(*genus_hypothesis, r.genus);
            assert_eq!(r.status.label(), format!("Decided: nodal ({nodes} nodes)"));
            irreducibility.clone()
        }
        s => panic!("expected Decided: nodal ({nodes} nodes), got {s:?}"),
    }
}

/// Decided by the delta-sum certificate: the local types of the non-nodal points; exact counts.
/// The owner's decision on F-9 (2026-11-15) made this status Decided (it was
/// `Unresolved(DeltaSumCertified)` before).
fn decided_delta_sum(r: &NormalizationCounts) -> Vec<LocalType> {
    assert_eq!(r.branch_rule, BranchRule::Exact);
    match &r.status {
        ReductionStatus::Decided(Certificate::DeltaSum {
            non_nodal,
            delta_sum,
            genus_hypothesis,
            ..
        }) => {
            assert_eq!(*genus_hypothesis, r.genus);
            assert_eq!(Some(*delta_sum), r.locus.delta_sum());
            assert!(r
                .status
                .label()
                .starts_with("Decided: delta-sum, Lemma 9.1 ("));
            non_nodal.iter().map(|s| s.local.clone()).collect()
        }
        s => panic!("expected Decided by the delta-sum certificate, got {s:?}"),
    }
}

/// The value and the gradient of `f` at an affine point (for checking a smooth point).
fn value_and_gradient(f: &FormModP, x: u64, y: u64) -> (u64, u64, u64) {
    let p = f.p();
    let (mut v, mut fx, mut fy) = (0u64, 0u64, 0u64);
    for (i, row) in f.coeffs().iter().enumerate() {
        for (j, &c) in row.iter().enumerate() {
            let m = |e_x: u64, e_y: u64| c * pow(x, e_x, p) % p * pow(y, e_y, p) % p;
            v = (v + m(i as u64, j as u64)) % p;
            if i > 0 {
                fx = (fx + i as u64 % p * m(i as u64 - 1, j as u64)) % p;
            }
            if j > 0 {
                fy = (fy + j as u64 % p * m(i as u64, j as u64 - 1)) % p;
            }
        }
    }
    (v, fx, fy)
}

fn from_rows(p: u64, rows: &[&[u64]]) -> FormModP {
    FormModP::new(p, rows.iter().map(|r| r.to_vec()).collect()).unwrap()
}

const B11: [i64; 5] = [5, 1, 0, 3, 1]; // T^4 + 3T^3 + T + 5
const B13: [i64; 5] = [1, 7, 0, 1, 2]; // 2T^4 + T^3 + 7T + 1

#[test]
fn smooth_genus_3() {
    // A: p = 11, a = 2T^4 + T + 3, b = T^3 + 5T^2 + 1, c = T^4 + 4T^2 + 7.
    // PARI: charpoly = x^6 + 5*x^4 - 6*x^3 + 55*x^2 + 1331.
    let f = quad_form(11, &[3, 1, 0, 0, 2], &[1, 0, 5, 1], &[7, 0, 4, 0, 1]);
    let r = check("A", &f, 3, 3, &[1, 0, 5, -6, 55, 0, 1331]);
    let ev = decided_nodal(&r, 0);
    // gcd(4, 2) = 2: absolute irreducibility by a smooth F_11-point, checked here on the curve.
    match ev.absolute {
        AbsoluteWitness::SmoothRationalPoint {
            x: Some(x),
            y: Some(y),
        } => {
            let (v, fx, fy) = value_and_gradient(&f, x, y);
            assert_eq!(v, 0, "A: ({x}, {y}) on the curve");
            assert!(fx != 0 || fy != 0, "A: ({x}, {y}) smooth");
        }
        w => panic!("A: expected an affine smooth F_p-point, got {w:?}"),
    }
    // With a genus below the true one the delta-sum is too small: an error, not a status.
    assert_eq!(
        normalization_point_counts(&f, 2, 1, &CountOptions::default()).unwrap_err(),
        CountError::GenusInconsistent {
            found: 0,
            expected: 1
        }
    );
}

#[test]
fn nodes_with_rational_and_conjugate_branches() {
    // PARI (small_curves.gp, branch.gp): B node at 2 conjugate; C node at 4 rational;
    // C2 node at 5 conjugate; G, H (p = 13) conjugate.
    let cases: [(&str, u64, &[i64], &[i64], [i64; 5], Vec<usize>); 5] = [
        (
            "B",
            11,
            &B11,
            &[0, 0, 9, 2, 6],
            [1, 3, 14, 33, 121],
            vec![2],
        ),
        (
            "C",
            11,
            &B11,
            &[7, 5, 9, 2, 6],
            [1, 2, 4, 22, 121],
            vec![1, 1],
        ),
        (
            "C2",
            11,
            &B11,
            &[8, 5, 9, 2, 6],
            [1, 4, 13, 44, 121],
            vec![2],
        ),
        (
            "G",
            13,
            &B13,
            &[10, 10, 1, 5, 3],
            [1, 1, -8, 13, 169],
            vec![2],
        ),
        (
            "H",
            13,
            &B13,
            &[0, 5, 1, 5, 3],
            [1, 7, 29, 91, 169],
            vec![2],
        ),
    ];
    for (name, p, b, c, pp, orbits) in cases {
        let f = quad_form(p, &[1], b, c);
        let r = check(name, &f, 2, 3, &pp);
        decided_nodal(&r, 1);
        assert_eq!(r.locus.points[0].branch_orbits, Some(orbits), "{name}");
        assert_eq!(r.locus.points[0].residue_degree, 1);
    }
}

#[test]
fn cusp_and_tacnodes() {
    // D: A2 at 3, PARI charpoly x^4 + 10*x^2 + 121.
    let f = quad_form(11, &[1], &B11, &[1, 8, 3, 2, 6]);
    let r = check("D", &f, 2, 3, &[1, 0, 10, 0, 121]);
    assert_eq!(decided_delta_sum(&r), vec![LocalType::A(2)]);
    assert_eq!(
        r.status.label(),
        "Decided: delta-sum, Lemma 9.1 (0 nodes, A2)"
    );
    // A2 is unibranch: the reference rule gives the same counts. (Before the owner's decision
    // on F-9 this test also asserted that the status under that rule is not Decided; the rule is
    // now never the rule of a status: `counts` with a Decided status are exact, and the rule's
    // counts are only `counts_under`.)
    assert_eq!(
        r.counts_under(BranchRule::AssumedUnibranch),
        Some(r.counts.clone())
    );
    assert_eq!(r.counts_under(BranchRule::Exact), Some(r.counts.clone()));
    // A3 (tacnodes), genus 1: E at 1 conjugate (x^2 - x + 11), F at 6 rational (x^2 + 2x + 11),
    // I at 2 conjugate (x^2 - 6x + 13), I2 at 3 conjugate (x^2 - 5x + 13).
    let cases: [(&str, u64, &[i64], &[i64], [i64; 3], Vec<usize>); 4] = [
        ("E", 11, &B11, &[0, 9, 4, 6, 6], [1, -1, 11], vec![2]),
        ("F", 11, &B11, &[10, 5, 7, 0, 6], [1, 2, 11], vec![1, 1]),
        ("I", 13, &B13, &[7, 0, 12, 0, 3], [1, -6, 13], vec![2]),
        ("I2", 13, &B13, &[0, 10, 12, 8, 3], [1, -5, 13], vec![2]),
    ];
    for (name, p, b, c, pp, orbits) in cases {
        let f = quad_form(p, &[1], b, c);
        let r = check(name, &f, 1, 3, &pp);
        assert_eq!(decided_delta_sum(&r), vec![LocalType::A(3)], "{name}");
        assert_eq!(r.locus.non_nodal()[0].branch_orbits, Some(orbits), "{name}");
        // the reference rule counts the tacnode once: wrong by +-1 at k = 1, so P changes
        let u = r.counts_under(BranchRule::AssumedUnibranch).unwrap();
        assert_ne!(u[0], r.counts[0], "{name}: unibranch rule differs");
    }
}

#[test]
fn tacnode_tangent_to_the_fibre() {
    // Review 1, 3.1: F = V^2 + (T^3 + 2T^4) V + 3T^4 at p = 11 has one A3 at (0, 0) whose tangent
    // is the fibre V = 0, so gcd(f, f_T, F_V) = T^3. PARI (tangent.gp): the normalization
    // y^2 = 4T^4 + 4T^3 + T^2 - 12 has a_p = 4, N_1 = 8; h(0) is not a square (conjugate).
    let f = quad_form(11, &[1], &[0, 0, 0, 1, 2], &[0, 0, 0, 0, 3]);
    let r = check("T3", &f, 1, 3, &[1, -4, 11]);
    assert_eq!(r.counts[0], 8);
    assert_eq!(decided_delta_sum(&r), vec![LocalType::A(3)]);
    let s = &r.locus.non_nodal()[0];
    assert_eq!(s.coordinates, Some((Some(0), Some(0))));
    assert_eq!(s.branch_orbits, Some(vec![2]));
}

#[test]
fn points_of_residue_degree_two() {
    // small_curves4.gp, p = 11, T^2 - 2 = T^2 + 9 mod 11 (2 is a nonsquare mod 11):
    // J (k = 4): b = T^4 + 3T^3 + T + 5, c = [5, 4, 5, 0, 4]; D = (T^2 + 9)^2 (quartic):
    //   one node of degree 2 over a degree 2 place of the V-line; h(sqrt 2) nonsquare in F_121:
    //   conjugate branches. charpoly x^2 - 6*x + 11.
    // J' (k = 2): c = [8, 4, 2, 0, 2]; h(sqrt 2) square: rational branches; x^2 + 4*x + 11.
    // J2 (k = 1): b = T^4 + 3T^2 + 5 (even), c = [2, 0, 10, 0, 1]: the two conjugate nodes lie
    //   in the fibre over one rational V (residue field K[T]/(T^2 - 2)); x^2 - 6*x + 11.
    // K1: b even, c = [0, 0, 1, 0, 6]; D = (T^2 + 1)(T^2 + 9)^3: two conjugate cusps in one
    //   rational fibre; x^2 + 4*x + 11.
    let cases: [(&str, &[i64], &[i64], [i64; 3], LocalType, Vec<usize>); 4] = [
        (
            "J",
            &B11,
            &[5, 4, 5, 0, 4],
            [1, -6, 11],
            LocalType::Node,
            vec![2],
        ),
        (
            "J'",
            &B11,
            &[8, 4, 2, 0, 2],
            [1, 4, 11],
            LocalType::Node,
            vec![1, 1],
        ),
        (
            "J2",
            &[5, 0, 3, 0, 1],
            &[2, 0, 10, 0, 1],
            [1, -6, 11],
            LocalType::Node,
            vec![1, 1],
        ),
        (
            "K1",
            &[5, 0, 3, 0, 1],
            &[0, 0, 1, 0, 6],
            [1, 4, 11],
            LocalType::A(2),
            vec![1],
        ),
    ];
    for (name, b, c, pp, ty, orbits) in cases {
        let f = quad_form(11, &[1], b, c);
        let r = check(name, &f, 1, 4, &pp);
        assert_eq!(r.locus.points.len(), 1, "{name}: one closed point");
        let s = &r.locus.points[0];
        assert_eq!(s.residue_degree, 2, "{name}");
        assert_eq!(s.local, ty, "{name}");
        assert_eq!(s.branch_orbits, Some(orbits), "{name}");
        if ty == LocalType::Node {
            decided_nodal(&r, 2);
        } else {
            assert_eq!(decided_delta_sum(&r), vec![LocalType::A(2)]);
            assert_eq!(
                r.status.label(),
                "Decided: delta-sum, Lemma 9.1 (0 nodes, A2 at a point of degree 2)"
            );
        }
    }
}

#[test]
fn split_fibre_with_a_cusp_and_a_node() {
    // K2 (small_curves4.gp), p = 13: b = (T - 1)(T - 4)(T^2 + 2T + 7) + 3, b(1) = b(4) = 3, so a
    // cusp at T = 1 and a node at T = 4 lie in the same fibre V = -3/2. c = [12, 6, 1, 12, 7].
    // D = (T+3)(T+7)(T+8)(T+9)^2(T+12)^3; h(4) is a square: rational branches.
    // PARI: charpoly x^2 + 2*x + 13.
    let f = quad_form(13, &[1], &[5, 12, 1, 10, 1], &[12, 6, 1, 12, 7]);
    let r = check("K2", &f, 1, 3, &[1, 2, 13]);
    assert_eq!(r.locus.points.len(), 2);
    let v0 = 5u64; // -3/2 mod 13 = 5
    let mut seen: Vec<(LocalType, Option<(Option<u64>, Option<u64>)>)> = r
        .locus
        .points
        .iter()
        .map(|s| (s.local.clone(), s.coordinates))
        .collect();
    seen.sort_by_key(|s| s.1);
    assert_eq!(
        seen,
        vec![
            (LocalType::A(2), Some((Some(1), Some(v0)))),
            (LocalType::Node, Some((Some(4), Some(v0)))),
        ]
    );
    assert_eq!(decided_delta_sum(&r), vec![LocalType::A(2)]);
}

#[test]
fn charts_at_infinity() {
    // The same curves after T -> 1/T and V -> 1/V: isomorphic models, so the same counts; the
    // singular points move to T = infinity, V = infinity.
    let cases: [(&str, u64, &[i64], &[i64], usize); 5] = [
        ("B", 11, &B11, &[0, 0, 9, 2, 6], 2),
        ("C", 11, &B11, &[7, 5, 9, 2, 6], 2),
        ("D", 11, &B11, &[1, 8, 3, 2, 6], 2),
        ("E", 11, &B11, &[0, 9, 4, 6, 6], 1),
        ("F", 11, &B11, &[10, 5, 7, 0, 6], 1),
    ];
    for (name, p, b, c, g) in cases {
        let f = quad_form(p, &[1], b, c);
        let base = normalization_point_counts(&f, g, 2, &CountOptions::default()).unwrap();
        for (chart, h) in [
            ("T -> 1/T", f.reverse_x()),
            ("V -> 1/V", f.reverse_y()),
            ("both", f.reverse_x().reverse_y()),
        ] {
            let r = normalization_point_counts(&h, g, 2, &CountOptions::default())
                .unwrap_or_else(|e| panic!("{name} {chart}: {e}"));
            assert_eq!(r.counts, base.counts, "{name} {chart}");
            assert_eq!(
                r.status.is_decided(),
                base.status.is_decided(),
                "{name} {chart}"
            );
        }
    }
}

#[test]
fn elliptic_curves_with_singular_points_at_infinity() {
    // E1: V^2 = T^3 + 3T + 5 as a (3,2) form: an A2 cusp at (inf, inf).
    // PARI: ellap(ellinit([0,0,0,3,5]), p) = 1, 3, 5 at p = 7, 11, 13.
    for (p, ap) in [(7u64, 1i64), (11, 3), (13, 5)] {
        let mut cols = vec![vec![0u64; 3]; 4];
        cols[0][2] = 1;
        cols[3][0] = p - 1;
        cols[1][0] = p - 3;
        cols[0][0] = p - 5;
        let f = FormModP::new(p, cols).unwrap();
        let r = check("E1", &f, 1, 3, &[1, -ap, p as i64]);
        assert_eq!(decided_delta_sum(&r), vec![LocalType::A(2)]);
        let s = r.locus.non_nodal()[0];
        assert_eq!(s.coordinates, Some((None, None)));
        // gcd(3, 2) = 1: a fibre irreducible of full degree settles absolute irreducibility.
        assert_eq!(
            r.status.certificate().unwrap().irreducibility().absolute,
            AbsoluteWitness::Bidegree { gcd: 1 }
        );
    }
    // E2: V^3 = T^3 + 2T + 3 as a (3,3) form: an ordinary triple point at (inf, inf).
    // PARI: e = ellfromeqn(y^3 - (x^3 + 2*x + 3)) = [0,0,27,0,-251]; ellap = -5, 7, -8 at
    // p = 7, 13, 19.
    let e2 = |p: u64| {
        let mut cols = vec![vec![0u64; 4]; 4];
        cols[0][3] = 1;
        cols[3][0] = p - 1;
        cols[1][0] = p - 2;
        cols[0][0] = p - 3;
        FormModP::new(p, cols).unwrap()
    };
    for (p, ap) in [(7u64, -5i64), (13, 7), (19, -8)] {
        let r = check("E2", &e2(p), 1, 3, &[1, -ap, p as i64]);
        assert_eq!(decided_delta_sum(&r), vec![LocalType::OrdinaryMultiple(3)]);
    }
    // p = 11: x^3 + 2x + 3 has a double root mod 11 (disc = -4*8 - 27*9 = -275 = 0 mod 11), so
    // the curve acquires further singular points: delta-sum 4 > p_a - g = 3.
    let r = normalization_point_counts(&e2(11), 1, 1, &CountOptions::default()).unwrap();
    match r.status {
        ReductionStatus::Unresolved(Unresolved::DeltaMismatch {
            found: 4,
            expected: 3,
            found_is_lower_bound: false,
        }) => {}
        s => panic!("E2 at 11: {s:?}"),
    }
}

#[test]
fn refusals() {
    // A fibre component: (V - 1)(V + T^4 + 1) = V^2 + T^4 V - T^4 - 1.
    let f = quad_form(11, &[1], &[0, 0, 0, 0, 1], &[-1, 0, 0, 0, -1]);
    assert!(matches!(
        singular_locus(&f),
        Err(CountError::DegenerateModel(
            Degeneracy::FibreComponent { .. }
        ))
    ));
    assert!(matches!(
        plane_point_count(&f, 1, Fibration::FibresInX),
        Err(CountError::DegenerateModel(
            Degeneracy::FibreComponent { .. }
        ))
    ));
    // D = 0: (V + T^2)^2.
    let f = quad_form(11, &[1], &[0, 0, 2], &[0, 0, 0, 0, 1]);
    assert_eq!(
        singular_locus(&f).unwrap_err(),
        CountError::DegenerateModel(Degeneracy::DiscriminantVanishes)
    );
    // p = 2, p not prime, p too large, field too large, fibre degree.
    assert_eq!(
        FormModP::new(2, vec![vec![1, 1], vec![1, 0]]).unwrap_err(),
        CountError::CharacteristicTwo
    );
    assert_eq!(
        FormModP::new(9, vec![vec![1, 1], vec![1, 0]]).unwrap_err(),
        CountError::NotPrime(9)
    );
    assert_eq!(
        FormModP::new(65537, vec![vec![1, 1], vec![1, 0]]).unwrap_err(),
        CountError::PrimeTooLarge(65537)
    );
    let f = quad_form(11, &[1], &B11, &[0, 0, 9, 2, 6]);
    assert!(matches!(
        normalization_point_counts(&f, 2, 9, &CountOptions::default()),
        Err(CountError::FieldTooLarge { .. })
    ));
    let quintic = FormModP::new(11, vec![vec![1, 1]; 6]).unwrap();
    assert_eq!(
        singular_locus(&quintic).unwrap_err(),
        CountError::FibreDegree(5)
    );
    // 65521^4 > 2^63: the word-size kernel refuses k = 4 at the largest prime below 2^16.
    let big = quad_form(65521, &[1], &B11, &[0, 0, 9, 2, 6]);
    assert_eq!(
        plane_point_count(&big, 4, Fibration::FibresInX).unwrap_err(),
        CountError::FieldTooLarge { p: 65521, k: 4 }
    );
    assert_eq!(
        normalization_point_counts(&big, 2, 4, &CountOptions::default()).unwrap_err(),
        CountError::FieldTooLarge { p: 65521, k: 4 }
    );
}

#[test]
fn weil_polynomial_of_b_from_counts() {
    // Cross-check of the weil module on curve B: P from N_1, N_2 equals the PARI charpoly.
    let f = quad_form(11, &[1], &B11, &[0, 0, 9, 2, 6]);
    let r = normalization_point_counts(&f, 2, 2, &CountOptions::default()).unwrap();
    let fp = FrobeniusPolynomial::from_counts_u64(11, &r.counts).unwrap();
    assert_eq!(fp.coefficients(), &ints(&[1, 3, 14, 33, 121])[..]);
}

#[test]
fn unclassified_e6_points() {
    // Review 1, 2.2 and 3.1: F = Y^3 - X^4 of bidegree (4,3), p_a = 6, at p = 7, 11, 13. Its
    // singular points (0, 0) and (inf, inf) are both E6 (y^3 = x^4), which the classifier does
    // not identify. The curve is rational: t -> (t^3, t^4) is a bijection from P^1 (inverse
    // t = Y/X), so #X0(F_q) = q + 1. PARI (verify.gp): brute force gives p affine points
    // plus (inf, inf), so p + 1, at p = 7, 11, 13.
    let e6 = |p: u64| {
        let mut cols = vec![vec![0u64; 4]; 5];
        cols[0][3] = 1;
        cols[4][0] = p - 1;
        FormModP::new(p, cols).unwrap()
    };
    for p in [7u64, 11, 13] {
        let f = e6(p);
        let line: Vec<u64> = (1..=3u32).map(|k| p.pow(k) + 1).collect();
        // By default: Unclassified and no counts at all.
        let r = normalization_point_counts(&f, 0, 3, &CountOptions::default()).unwrap();
        assert!(
            matches!(
                r.status,
                ReductionStatus::Unresolved(Unresolved::Unclassified(_))
            ),
            "E6 at {p}: {:?}",
            r.status
        );
        assert!(r.counts.is_empty(), "E6 at {p}: no counts");
        assert_eq!(r.plane_counts, line, "E6 at {p}: plane counts");
        assert_eq!(brute_force(&f), p + 1, "E6 at {p}: brute force");
        assert_eq!(r.locus.points.len(), 2, "E6 at {p}");
        for s in &r.locus.points {
            assert!(
                matches!(s.local, LocalType::Unclassified(_)),
                "E6 at {p}: {}",
                s.describe()
            );
            assert_eq!(s.delta, None, "E6 at {p}");
        }
        // unibranch_if_unclassified: counts under the assumed rule, never Decided. E6 is
        // unibranch, so here these counts happen to be q + 1.
        let opts = CountOptions {
            unibranch_if_unclassified: true,
            ..CountOptions::default()
        };
        let u = normalization_point_counts(&f, 0, 3, &opts).unwrap();
        assert_eq!(u.branch_rule, BranchRule::AssumedUnibranch, "E6 at {p}");
        assert!(!u.status.is_decided(), "E6 at {p}");
        assert_eq!(
            u.status, r.status,
            "E6 at {p}: the option does not change the status"
        );
        assert_eq!(u.counts, line, "E6 at {p}");
        // As the reduction of a smooth (4,3) curve over Q (genus 6, p_a - g = 0), the two points
        // count at least 1 each: a delta-sum of at least 2, a lower bound.
        let s = normalization_point_counts(&f, 6, 1, &CountOptions::default()).unwrap();
        assert_eq!(
            s.status,
            ReductionStatus::Unresolved(Unresolved::DeltaMismatch {
                found: 2,
                expected: 0,
                found_is_lower_bound: true,
            }),
            "E6 at {p}, genus 6"
        );
        assert!(s.counts.is_empty(), "E6 at {p}, genus 6: no counts");
    }
    // The criterion still applies to E6 (it needs no classification): at 11 no F_11-rational
    // fibre is irreducible of full degree in either ruling, and the x-fibre over the root of
    // y^2 + y + 6 is irreducible over F_121, the first in the order of the search (PARI,
    // f9_tests.gp; unit.gp). gcd(4, 3) = 1.
    match absolute_irreducibility(&e6(11)).unwrap() {
        Irreducibility::Absolute(ev) => {
            assert_eq!(
                ev.fibre,
                IrreducibleFibre {
                    fibration: Fibration::FibresInX,
                    base: BasePoint::Closed(vec![6, 1, 1]),
                }
            );
            assert_eq!(ev.absolute, AbsoluteWitness::Bidegree { gcd: 1 });
        }
        r => panic!("E6 at 11: {r:?}"),
    }
}

/// A curve of WP-F code review 1, Section 2.2 at p = 13: `X^2 + b(Y) X + c(Y)`, with its
/// PARI/GP reference (quoted at [`review_cases`]) and its one singular closed point.
struct ReviewCase {
    name: &'static str,
    b: &'static [u64],
    c: &'static [u64],
    genus: usize,
    /// `hyperellcharpoly` of the normalization, descending.
    charpoly: &'static [i64],
    /// `N_1, ..., N_6` of the normalization.
    counts: [u64; 6],
    local: LocalType,
    degree: usize,
    orbits: &'static [usize],
}

/// Follow-up F-11, the A_k part: odd k over a residue field of degree > 1, and the precision
/// doubling of the classifier.
// F = X^2 + b(Y) X + c(Y) at p = 13 with D = b^2 - 4c = S(Y)^2 H(Y). The singular points lie
// over the roots y0 of S, the normalization is y^2 = H, and the branches at y0 are rational iff
// h(y0) = (D / S^ord)(y0) is a square in F_13(y0). PARI (verify.gp): D = factor(D) mod 13,
// charpoly = hyperellcharpoly(Mod(1,13)*H), N_k = 13^k + 1 - polsym(charpoly, 6)[k+1], and
// issquare(subst(h, 'Y, ffgen(S))).
// node3_sq0: D = (Y+5)(Y^3+2)^2(Y^3+8Y^2+Y+11), H = Y^4+2Y^2+3Y+3, charpoly x^2 + x + 13,
//   N = 15, 195, 2160, 28275, 372075, 4829760. h(y0) not a square in F_13^3: the branches are
//   conjugate over F_13^3 and rational over F_13^6.
// node3_sq1: D = (Y+5)(Y^3+2)^2(Y^3+8Y^2+4Y+9), H = Y^4+5Y^2+3Y+6, charpoly x^2 - x + 13,
//   N = 13, 195, 2236, 28275, 370513, 4829760. h(y0) a square: rational branches.
// A3_d2_sq0: D = (Y+1)(Y+8)(Y^2+2)^4(Y^2+7Y+7), H = Y^4+3Y^3+2Y+4, charpoly x^2 + 4x + 13,
//   N = 18, 180, 2106, 28800, 371538, 4822740. h(y0) not a square in F_13^2: conjugate branches.
// A3_d2_sq1: D = (Y+1)(Y^2+2)^4(Y^3+2Y^2+11Y+6), H = Y^4+3Y^3+4Y+6, charpoly x^2 - 5x + 13,
//   N = 9, 171, 2268, 28899, 372069, 4826304. h(y0) a square: rational branches.
// A15_sq1: D = (Y-1)^16 (Y^2+Y+2), genus 0, N = 14, 170, 2198, 28562, 371294, 4826810
//   (13^k + 1). h(1) = 4, a square.
// A15_sq0: D = (Y-1)^16 (Y+6)(Y+9), genus 0, N as for A15_sq1. h(1) = 5, not a square.
fn review_cases() -> Vec<ReviewCase> {
    let a15 = [14, 170, 2198, 28562, 371294, 4826810];
    vec![
        ReviewCase {
            name: "node3_sq0",
            b: &[7, 1, 10, 1, 0, 1],
            c: &[6, 7, 4, 12, 2, 0],
            genus: 1,
            charpoly: &[1, 1, 13],
            counts: [15, 195, 2160, 28275, 372075, 4829760],
            local: LocalType::Node,
            degree: 3,
            orbits: &[2],
        },
        ReviewCase {
            name: "node3_sq1",
            b: &[11, 8, 10, 9, 0, 1],
            c: &[8, 2, 1, 12, 5, 0],
            genus: 1,
            charpoly: &[1, -1, 13],
            counts: [13, 195, 2236, 28275, 370513, 4829760],
            local: LocalType::Node,
            degree: 3,
            orbits: &[1, 1],
        },
        ReviewCase {
            name: "A3_d2_sq0",
            b: &[12, 3, 1, 3, 11, 8, 1],
            c: &[7, 10, 12, 11, 7, 4, 0],
            genus: 1,
            charpoly: &[1, 4, 13],
            counts: [18, 180, 2106, 28800, 371538, 4822740],
            local: LocalType::A(3),
            degree: 2,
            orbits: &[2],
        },
        ReviewCase {
            name: "A3_d2_sq1",
            b: &[1, 4, 7, 4, 11, 8, 1],
            c: &[12, 12, 5, 11, 2, 5, 0],
            genus: 1,
            charpoly: &[1, -5, 13],
            counts: [9, 171, 2268, 28899, 372069, 4826304],
            local: LocalType::A(3),
            degree: 2,
            orbits: &[1, 1],
        },
        ReviewCase {
            name: "A15_sq1",
            b: &[12, 2, 9, 8, 12, 9, 5, 7, 12, 1],
            c: &[3, 10, 2, 12, 12, 8, 5, 12, 11, 0],
            genus: 0,
            charpoly: &[1],
            counts: a15,
            local: LocalType::A(15),
            degree: 1,
            orbits: &[1, 1],
        },
        ReviewCase {
            name: "A15_sq0",
            b: &[11, 12, 10, 0, 3, 8, 0, 1, 6, 1],
            c: &[7, 2, 3, 11, 12, 7, 11, 0, 12, 0],
            genus: 0,
            charpoly: &[1],
            counts: a15,
            local: LocalType::A(15),
            degree: 1,
            orbits: &[2],
        },
    ]
}

/// Counts up to `kmax` against PARI, the closed point, its branches and the status.
fn check_review_case(case: &ReviewCase, kmax: u32) {
    let name = case.name;
    let f = monic_quadratic_in_x(13, case.b, case.c);
    let r = check(name, &f, case.genus, kmax, case.charpoly);
    assert_eq!(r.counts, &case.counts[..kmax as usize], "{name}: counts");
    assert_eq!(r.locus.points.len(), 1, "{name}: one closed point");
    let s = &r.locus.points[0];
    assert_eq!(s.residue_degree, case.degree, "{name}");
    assert_eq!(s.local, case.local, "{name}");
    assert_eq!(s.branch_orbits.as_deref(), Some(case.orbits), "{name}");
    if case.local == LocalType::Node {
        decided_nodal(&r, case.degree);
    } else {
        assert_eq!(decided_delta_sum(&r), vec![case.local.clone()], "{name}");
    }
}

#[test]
fn node_of_degree_3_tacnode_of_degree_2_and_a15() {
    // k = 3 already separates the two node classes (3 more or 3 fewer points than the plane
    // model). k = 6 is in node_of_degree_3_up_to_k6 (release). The A3 at a degree 2 place needs
    // k = 2 (class) and k = 4 (branches rational); the A15 needs k = 1 and 2.
    for case in review_cases() {
        let kmax = match case.local {
            LocalType::Node => 3,
            LocalType::A(3) => 4,
            _ => 3,
        };
        check_review_case(&case, kmax);
    }
}

#[test]
#[ignore = "release, about 2 s (k = 6 at p = 13 takes about 20 s per curve in debug)"]
fn node_of_degree_3_up_to_k6() {
    // The conjugate branches of node3_sq0 become rational over F_13^6.
    for case in review_cases().iter().filter(|c| c.local == LocalType::Node) {
        check_review_case(case, 6);
    }
}

/// The regression curves of the F-9 design, Section 3.3, at p = 13 (PARI/GP `regress.gp`,
/// output `regress.out.txt`; the search that found (i) and (ii) is `regress_search.gp`).
/// Coefficient rows `[i][j]` of `x^i y^j`.
const RED_I: [&[u64]; 5] = [
    &[8, 3, 0, 10, 5],
    &[5, 4, 3, 4, 0],
    &[8, 3, 0, 8, 0],
    &[5, 7, 8, 0, 11],
    &[5, 7, 5, 5, 1],
];
const RED_II: [&[u64]; 5] = [
    &[7, 11, 4, 3, 6],
    &[9, 3, 1, 11, 0],
    &[7, 6, 2, 0, 12],
    &[5, 0, 9, 1, 4],
    &[3, 0, 0, 11, 1],
];

#[test]
fn reducible_reductions_are_never_decided() {
    // Lemma 9.1 needs F mod p absolutely irreducible. Each curve below has a singular locus
    // whose delta-sum is p_a - g for the genus input, so the delta tally alone (the certificate
    // before F-9) would call it Decided. Each is reducible over F_p-bar; none is Decided now.
    let opts = CountOptions::default();

    // (i) F = E1 E2, two smooth (2,2) curves over F_13 meeting transversally in 8 points
    // (PARI: factor over F_13 gives two factors of bidegree (2,2); Res_x(E1, E2) squarefree of
    // degree 8). Bidegree (4,4), p_a = 9, genus input 1: 8 nodes, delta 8 = p_a - g.
    let f1 = from_rows(13, &RED_I);
    let locus = singular_locus(&f1).unwrap();
    assert_eq!(locus.node_count(), 8);
    assert_eq!(locus.delta_sum(), Some(8));
    let r = normalization_point_counts(&f1, 1, 2, &opts).unwrap();
    assert_eq!(
        r.status,
        ReductionStatus::Unresolved(Unresolved::IrreducibilityNotShown(
            IrreducibilityGap::NoIrreducibleFibre
        ))
    );
    assert_eq!(
        r.status.label(),
        "Unresolved: absolute irreducibility not shown (no irreducible fibre among the first 64 \
         closed points of each degree <= 4 in either ruling)"
    );
    // The counts are those of the normalization E1 + E2 (PARI: #E1 = 9, 171, #E2 = 10, 180),
    // and the plane model has 18 points over F_13 (PARI brute force); the genus-1 polynomial
    // T^2 + 5T + 13 from N_1 = 19 is Weil but predicts N_2 = 171, not 351.
    assert_eq!(r.counts, vec![19, 351]);
    assert_eq!(r.plane_counts[0], 18);
    assert_eq!(brute_force(&f1), 18);
    assert_eq!(r.predicted_counts_agree(), Some(false));
    assert_eq!(
        absolute_irreducibility(&f1),
        Ok(Irreducibility::NotShown(
            IrreducibilityGap::NoIrreducibleFibre
        ))
    );

    // (ii) F = E E^sigma for a smooth (2,2) curve E over F_169 = F_13(w), w^2 = 2: irreducible
    // over F_13, two conjugate factors over F_169 (PARI). 8 nodes (E meets E^sigma in 8
    // points), no F_13-point at all (PARI brute force 0), N_2 = 2 #E(F_169) = 290.
    let f2 = from_rows(13, &RED_II);
    let r = normalization_point_counts(&f2, 1, 2, &opts).unwrap();
    assert_eq!(r.locus.node_count(), 8);
    let x_over_0 = IrreducibleFibre {
        fibration: Fibration::FibresInX,
        base: BasePoint::Rational(0),
    };
    assert_eq!(
        r.status,
        ReductionStatus::Unresolved(Unresolved::IrreducibilityNotShown(
            IrreducibilityGap::NoSmoothRationalPoint {
                fibre: x_over_0,
                gcd: 4
            }
        ))
    );
    assert_eq!(
        r.status.label(),
        "Unresolved: absolute irreducibility not shown (irreducible over F_p by the x-fibre over \
         y = 0; gcd(dx, dy) = 4 and no smooth F_p-point)"
    );
    assert_eq!(r.counts, vec![0, 290]);
    assert_eq!(r.plane_counts[0], 0);
    assert_eq!(brute_force(&f2), 0);

    // (iii) F = (x - 3) G, G smooth of bidegree (3,2) and genus 2 (PARI: factor gives x - 3
    // and G). Bidegree (4,2), p_a = 3, genus input 1: singular_locus finds 2 nodes (the line
    // meets G in a closed point of degree 2), delta 2 = p_a - g. A vertical line: refused.
    let f3 = from_rows(
        13,
        &[
            &[5, 10, 11],
            &[4, 5, 12],
            &[1, 0, 2],
            &[7, 1, 10],
            &[2, 0, 1],
        ],
    );
    let locus = singular_locus(&f3).unwrap();
    assert_eq!(locus.node_count(), 2);
    assert_eq!(locus.delta_sum(), Some(2));
    let vertical = CountError::DegenerateModel(Degeneracy::FibreComponent {
        place: "x-place g(x) = [10, 1] (ascending)".into(),
    });
    assert_eq!(certify(&f3, 1).unwrap_err(), vertical);
    assert_eq!(
        normalization_point_counts(&f3, 1, 2, &opts).unwrap_err(),
        vertical
    );
    // (iii') G alone read as a (4,2) form: the line x = infinity is a component (the bidegree
    // drops). Again 2 nodes and delta 2 = p_a - 1.
    let f4 = from_rows(
        13,
        &[&[7, 1, 5], &[1, 3, 2], &[0, 1, 0], &[2, 0, 1], &[0, 0, 0]],
    );
    assert_eq!(singular_locus(&f4).unwrap().delta_sum(), Some(2));
    assert_eq!(
        certify(&f4, 1).unwrap_err(),
        CountError::DegenerateModel(Degeneracy::FibreComponent {
            place: "x = infinity".into()
        })
    );
    // (iii'') the line y = infinity: (x^4 + 2x + 1) y + (x^3 + 5) as a (4,2) form, refused by
    // singular_locus already (unchanged).
    let f5 = from_rows(
        13,
        &[&[5, 1, 0], &[0, 2, 0], &[0, 0, 0], &[1, 0, 0], &[0, 1, 0]],
    );
    let horizontal = CountError::DegenerateModel(Degeneracy::FibreComponent {
        place: "y = infinity".into(),
    });
    assert_eq!(singular_locus(&f5).unwrap_err(), horizontal);
    assert_eq!(certify(&f5, 0).unwrap_err(), horizontal);

    // (v) x^2 + 11 of bidegree (2, 0): two conjugate vertical lines (PARI: irreducible over
    // F_13). A bidegree with a zero entry is a shape error on every path.
    let fv = from_rows(13, &[&[11], &[0], &[1]]);
    assert!(matches!(certify(&fv, 0), Err(CountError::Shape(_))));
    assert!(matches!(
        normalization_point_counts(&fv, 0, 1, &opts),
        Err(CountError::Shape(_))
    ));
    assert!(matches!(
        absolute_irreducibility(&fv),
        Err(CountError::Shape(_))
    ));
}

#[test]
fn smooth_22_curve_certified_by_a_smooth_point() {
    // (iv) E1 of (i) alone, a smooth (2,2) curve of genus 1 over F_13 (PARI: charpoly
    // x^2 - 5x + 13, #E1 = 9, 171, 2268). gcd(2, 2) = 2, so a fibre is not enough; the first
    // smooth F_13-point in the order of the search is (3, 6) (PARI, f9_tests.gp).
    let e1 = from_rows(13, &[&[11, 9, 11], &[6, 12, 8], &[5, 11, 1]]);
    let r = check("E1 (2,2)", &e1, 1, 3, &[1, -5, 13]);
    assert_eq!(r.counts, vec![9, 171, 2268]);
    let ev = decided_nodal(&r, 0);
    assert_eq!(
        ev.fibre,
        IrreducibleFibre {
            fibration: Fibration::FibresInX,
            base: BasePoint::Rational(0),
        }
    );
    assert_eq!(
        ev.absolute,
        AbsoluteWitness::SmoothRationalPoint {
            x: Some(3),
            y: Some(6)
        }
    );
    let (v, fx, fy) = value_and_gradient(&e1, 3, 6);
    assert_eq!(v, 0);
    assert!(fx != 0 || fy != 0);
    assert_eq!(
        ev.to_string(),
        "x-fibre over y = 0 irreducible of full degree; smooth F_p-point (3, 6)"
    );
}

#[test]
fn absolute_irreducibility_settled_by_a_residue_degree() {
    // F-9 review 1, m3: gcd(a, b) = 2 settled by a fibre over a point of even degree (Claim B:
    // the m conjugate factors fall into gcd(m, d) orbits over F_(p^d), and m | 2).
    // S1 (p = 3, bidegree (2,2)): (y^2 + y + 1) x^2 + (y + 1) x + (2y + 1). PARI (f9_tests.gp):
    // one factor over F_3 and over F_9; no F_3-rational fibre is irreducible of full degree in
    // either ruling; the x-fibre over the root of y^2 + 1 is irreducible over F_9; the
    // x-discriminant y^3 + y^2 + 2y is squarefree (one simple root at infinity), so S1 is smooth
    // of genus 1: charpoly x^2 + 2x + 3, N = 6, 12, 18 (brute force and hyperellcharpoly).
    let s1 = from_rows(3, &[&[1, 2, 0], &[1, 1, 0], &[1, 1, 1]]);
    let r = check("S1", &s1, 1, 3, &[1, 2, 3]);
    assert_eq!(r.counts, vec![6, 12, 18]);
    let ev = decided_nodal(&r, 0);
    assert_eq!(
        ev.fibre,
        IrreducibleFibre {
            fibration: Fibration::FibresInX,
            base: BasePoint::Closed(vec![1, 0, 1]),
        }
    );
    assert_eq!(ev.absolute, AbsoluteWitness::Bidegree { gcd: 2 });
    assert_eq!(
        ev.to_string(),
        "x-fibre over the point g(y) = [1, 0, 1] of degree 2 irreducible of full degree; \
         gcd(dx, dy) = 2, settled by the residue degree 2"
    );
    // S2 (p = 3, bidegree (4,2)): (2y^2 + 2) x^4 + (2y^2 + 2y + 1) x^3 + (y^2 + 2y + 1) x^2 +
    // (2y^2 + 2y + 2) x + (2y^2 + 2). PARI (f9_tests.gp): one factor over F_3 and over F_9; the
    // x-fibre over y = 1 is irreducible, there is no smooth F_3-point, and the first x-fibre over
    // a point of degree 2 that is irreducible over F_9 is the one over y^2 + 2y + 2.
    let s2 = from_rows(
        3,
        &[&[2, 0, 2], &[2, 2, 2], &[1, 2, 1], &[1, 2, 2], &[2, 0, 2]],
    );
    match absolute_irreducibility(&s2).unwrap() {
        Irreducibility::Absolute(ev) => {
            assert_eq!(
                ev.fibre,
                IrreducibleFibre {
                    fibration: Fibration::FibresInX,
                    base: BasePoint::Closed(vec![2, 2, 1]),
                }
            );
            assert_eq!(ev.absolute, AbsoluteWitness::Bidegree { gcd: 2 });
        }
        r => panic!("S2: {r:?}"),
    }
}

#[test]
fn primitive_forms() {
    // Lemma 9.1 needs a primitive F: reduce_primitive refuses content 2 and accepts the
    // primitive part.
    let n = |v: i64| Integer::from(v);
    let g = BihomogeneousForm::new(vec![vec![n(2), n(4)], vec![n(6), n(0)], vec![n(2), n(-2)]])
        .unwrap();
    assert!(!g.is_primitive());
    let e = g.reduce_primitive(13).unwrap_err();
    assert_eq!(e, CountError::NotPrimitive { content: n(2) });
    assert_eq!(
        e.to_string(),
        "the form is not primitive (content 2); Lemma 9.1 needs content 1"
    );
    let h = g.primitive_part();
    assert!(h.is_primitive());
    assert_eq!(h.reduce_primitive(13).unwrap(), h.reduce(13).unwrap());
    // content 13 at p = 13: not primitive (reduce alone gives the zero form)
    let z = BihomogeneousForm::new(vec![vec![n(13), n(26)], vec![n(39), n(0)]]).unwrap();
    assert_eq!(
        z.reduce_primitive(13).unwrap_err(),
        CountError::NotPrimitive { content: n(13) }
    );
    assert_eq!(z.reduce(13).unwrap_err(), CountError::ZeroForm);
}

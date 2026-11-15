//! Point counts on small curves in `P^1 x P^1` against PARI/GP (MAGMA ch. 114 `LPolynomial`,
//! `IsNode`, `IsCusp`, `IsOrdinarySingularity`; digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`).
//!
//! The curves `V^2 + b(T) V + c(T) = 0` have bidegree (4,2) (fibre variable `X = T`, degree 4;
//! `Y = V`). Their singular points are those of `y^2 = D(T)`, `D = b^2 - 4c`, `y = 2V + b`, and the
//! normalization is `y^2 = D / (square part)`. Reference values: PARI/GP 2.15
//! `hyperellcharpoly` of the normalization, and `ellap`, from the scripts
//! `scratchpad/m23/design/WPF/gp/small_curves.gp` (curves A to I2, E1, E2; reproduced in the WP-F
//! design review), `scratchpad/m23/design/WPF/review1/tangent.gp` (curve T3), and
//! `scratchpad/m23/impl/gp/small_curves4.gp` (curves J, J', J2, K1, K2). The PARI output is quoted
//! at each case. `P` is descending: `[1, a_1, ..., p^g]`.

// The case tables are tuples on purpose: one line per curve, as in the PARI/GP scripts.
#![allow(clippy::type_complexity)]

use rustmath_curves::nodal_biprojective::{
    normalization_point_counts, plane_point_count, singular_locus, BranchRule, Certificate,
    CountError, CountOptions, Degeneracy, Fibration, FormModP, LocalType, ReductionStatus,
    Unresolved,
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

fn decided_nodal(r: &rustmath_curves::nodal_biprojective::NormalizationCounts, nodes: usize) {
    assert_eq!(
        r.status,
        ReductionStatus::Decided(Certificate::Nodal {
            nodes,
            genus_hypothesis: r.genus
        })
    );
    assert_eq!(r.branch_rule, BranchRule::Exact);
}

fn delta_certified(r: &rustmath_curves::nodal_biprojective::NormalizationCounts) -> Vec<LocalType> {
    match &r.status {
        ReductionStatus::Unresolved(Unresolved::DeltaSumCertified { non_nodal, .. }) => {
            non_nodal.iter().map(|s| s.local.clone()).collect()
        }
        s => panic!("expected DeltaSumCertified, got {s:?}"),
    }
}

const B11: [i64; 5] = [5, 1, 0, 3, 1]; // T^4 + 3T^3 + T + 5
const B13: [i64; 5] = [1, 7, 0, 1, 2]; // 2T^4 + T^3 + 7T + 1

#[test]
fn smooth_genus_3() {
    // A: p = 11, a = 2T^4 + T + 3, b = T^3 + 5T^2 + 1, c = T^4 + 4T^2 + 7.
    // PARI: charpoly = x^6 + 5*x^4 - 6*x^3 + 55*x^2 + 1331.
    let f = quad_form(11, &[3, 1, 0, 0, 2], &[1, 0, 5, 1], &[7, 0, 4, 0, 1]);
    let r = check("A", &f, 3, 3, &[1, 0, 5, -6, 55, 0, 1331]);
    decided_nodal(&r, 0);
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
    assert_eq!(delta_certified(&r), vec![LocalType::A(2)]);
    assert_eq!(r.branch_rule, BranchRule::Exact);
    // A2 is unibranch: the reference rule gives the same counts (but stays Unresolved).
    let opts = CountOptions {
        branch_rule: BranchRule::AssumedUnibranch,
        ..CountOptions::default()
    };
    let u = normalization_point_counts(&f, 2, 3, &opts).unwrap();
    assert_eq!(u.counts, r.counts);
    assert_eq!(u.branch_rule, BranchRule::AssumedUnibranch);
    assert!(!u.status.is_decided());
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
        assert_eq!(delta_certified(&r), vec![LocalType::A(3)], "{name}");
        assert_eq!(r.locus.non_nodal()[0].branch_orbits, Some(orbits), "{name}");
        // the reference rule counts the tacnode once: wrong by +-1 at k = 1, so P changes
        let u = normalization_point_counts(
            &f,
            1,
            3,
            &CountOptions {
                branch_rule: BranchRule::AssumedUnibranch,
                ..CountOptions::default()
            },
        )
        .unwrap();
        assert_ne!(u.counts[0], r.counts[0], "{name}: unibranch rule differs");
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
    assert_eq!(delta_certified(&r), vec![LocalType::A(3)]);
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
            assert_eq!(delta_certified(&r), vec![LocalType::A(2)]);
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
    assert_eq!(delta_certified(&r), vec![LocalType::A(2)]);
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
        assert_eq!(delta_certified(&r), vec![LocalType::A(2)]);
        let s = r.locus.non_nodal()[0];
        assert_eq!(s.coordinates, Some((None, None)));
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
        assert_eq!(delta_certified(&r), vec![LocalType::OrdinaryMultiple(3)]);
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

//! The 57 three-point class vectors of M23 (WP-P, HANDOFF 4.1 and 4.2 steps 4 to 6): the
//! enumeration, the counting identity `#survivors = n |C(b)|`, generation by transitivity, the
//! real points and the genus, against the owner's `data/m23_three_point_nielsen.json` and
//! `data/m23_three_point_passports.json` (vendored in `tests/data/m23/`, README there).
//! MAGMA ch. 58 (permutation groups) and ch. 91 (characters, structure constants).
//!
//! Fast tests run in debug. The full table is opt-in (release):
//!
//! ```text
//! cargo test --release -p rustmath-groups --test m23_three_point -- --ignored --exact full_57_rows
//! ```

mod m23_common;

use std::collections::BTreeSet;
use std::time::Instant;

use m23_common::{data_dir, read_nielsen_rows, read_passport_rows, NielsenRow};
use rustmath_groups::m23::classes::ORBIT_CAP;
use rustmath_groups::m23::three_point::ClassVectorCount;
use rustmath_groups::m23::{
    count_class_vector, enumerate_class_vectors, genus23, structure_count, AtlasTable,
    ClassIdentifier, ClassLists, ClassVector, CountStatus, M23Class, ThreePointError,
    TrustedResult, VectorKind, M23,
};
use rustmath_rationals::Rational;

/// GAP 4.12.1: `SizesCentralizers(CharacterTable("M23"))`.
const GAP_CENTRALIZERS: [u64; 17] = [
    10200960, 2688, 180, 32, 15, 12, 14, 14, 8, 11, 11, 14, 14, 15, 15, 23, 23,
];

fn setup() -> (M23, ClassIdentifier) {
    let g = M23::published().unwrap();
    let id = ClassIdentifier::new(&g).unwrap();
    (g, id)
}

fn kind_of(s: &str) -> VectorKind {
    match s {
        "rational" => VectorKind::Rational,
        "conjugate_pair" => VectorKind::ConjugatePair,
        other => panic!("unknown kind {other}"),
    }
}

fn vector_of(row: &NielsenRow) -> ClassVector {
    ClassVector {
        kind: kind_of(&row.kind),
        classes: row.classes,
    }
}

fn nielsen_rows() -> Vec<NielsenRow> {
    read_nielsen_rows(&data_dir().join("m23_three_point_nielsen.json")).unwrap()
}

/// Compares a count with the owner's row: n, size, real, genus, and the checks that hold on
/// every row (status, divisibility, parity, a = c adds nothing, |C(b)| equal to GAP's).
fn check_row(c: &ClassVectorCount, row: &NielsenRow) {
    let v = c.vector;
    assert_eq!(
        c.structure_constant.to_string(),
        row.structure_constant,
        "n of {v}"
    );
    assert_eq!(c.size, row.size, "size of {v}");
    assert_eq!(c.real, row.real, "real of {v}");
    assert_eq!(c.genus23, row.genus23, "genus of {v}");
    assert_eq!(c.status, CountStatus::Decided);
    assert_eq!(
        Rational::from(c.survivors as i64),
        &c.structure_constant * &Rational::from(c.centralizer_order as i64),
        "identity of {v}"
    );
    assert_eq!(
        c.centralizer_order,
        GAP_CENTRALIZERS[c.b_class.index()],
        "|C(b)| of {v}"
    );
    assert_eq!(c.generating % c.centralizer_order, 0);
    assert_eq!(c.real_involutions % c.centralizer_order, 0);
    assert_eq!((c.size - c.real) % 2, 0, "size - real even for {v}");
    assert_eq!(c.a_equals_c_generating, 0, "a = c never generates ({v})");
    let want = match v.kind {
        VectorKind::Rational => TrustedResult::ConjugationThreeRealPoints,
        VectorKind::ConjugatePair => TrustedResult::ConjugationConjugatePair,
    };
    assert_eq!(c.trusted, vec![TrustedResult::AtlasMaximalSubgroups, want]);
}

// The set of class vectors of genus >= 0 and n > 0 equals the 57 rows of both owner files
// (m23_three_point_nielsen.json and m23_three_point_passports.json, the latter from GAP).
#[test]
fn class_vector_enumeration_is_the_57() {
    let (_g, id) = setup();
    let t = AtlasTable::m23();
    let ours = enumerate_class_vectors(t, &id).unwrap();
    assert_eq!(ours.len(), 57);
    let ours_set: BTreeSet<ClassVector> = ours.iter().copied().collect();
    let rows = nielsen_rows();
    let theirs: BTreeSet<ClassVector> = rows.iter().map(vector_of).collect();
    assert_eq!(ours_set, theirs);
    let passports: BTreeSet<ClassVector> =
        read_passport_rows(&data_dir().join("m23_three_point_passports.json"))
            .unwrap()
            .into_iter()
            .map(|(k, classes, _)| ClassVector {
                kind: kind_of(&k),
                classes,
            })
            .collect();
    assert_eq!(ours_set, passports);
    assert_eq!(
        ours.iter()
            .filter(|v| v.kind == VectorKind::Rational)
            .count(),
        27
    );
    for row in &rows {
        let v = vector_of(row);
        v.validate().unwrap();
        assert_eq!(genus23(&id, &v).unwrap(), row.genus23, "{v}");
        assert_eq!(
            t.structure_constant(v.classes).unwrap().to_string(),
            row.structure_constant,
            "{v}"
        );
    }
}

// The 18 rows whose a class is 2A or 3A, with the class lists as conjugation orbits (debug):
// among them (2A,23A,23B) 7/3, (2A,14A,14B) 17/3, (2A,7A,7B) 0/0 and (3A,5A,6A) 70/4.
#[test]
fn eighteen_small_rows() {
    let (g, id) = setup();
    let t = AtlasTable::m23();
    let t0 = Instant::now();
    let lists = ClassLists::by_orbit(&g, &id, &[M23Class::C2A, M23Class::C3A], ORBIT_CAP).unwrap();
    assert_eq!(lists.get(M23Class::C2A).unwrap().len(), 3795);
    assert_eq!(lists.get(M23Class::C3A).unwrap().len(), 56672);
    let mut done = 0;
    for row in nielsen_rows() {
        let v = vector_of(&row);
        let s = match structure_count(&g, &id, t, &lists, &v) {
            Ok(s) => s,
            Err(ThreePointError::ClassListMissing(_)) => continue,
            Err(e) => panic!("{v}: {e}"),
        };
        let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
        assert_eq!(s.survivors, c.survivors);
        check_row(&c, &row);
        done += 1;
    }
    assert_eq!(done, 18);
    let named = [
        ([M23Class::C2A, M23Class::C23A, M23Class::C23B], 7u64, 3u64),
        ([M23Class::C2A, M23Class::C14A, M23Class::C14B], 17, 3),
        ([M23Class::C2A, M23Class::C7A, M23Class::C7B], 0, 0),
    ];
    for (classes, size, real) in named {
        let v = ClassVector {
            kind: VectorKind::ConjugatePair,
            classes,
        };
        let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
        assert_eq!((c.size, c.real), (size, real), "{v}");
    }
    // (2A, 7A, 7B): 301 survivors, n = 43/2, none generating (design 9)
    let v = ClassVector {
        kind: VectorKind::ConjugatePair,
        classes: [M23Class::C2A, M23Class::C7A, M23Class::C7B],
    };
    let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
    assert_eq!((c.survivors, c.generating), (301, 0));
    assert_eq!(c.structure_constant, Rational::new(43, 2).unwrap());
    let v = ClassVector {
        kind: VectorKind::Rational,
        classes: [M23Class::C3A, M23Class::C5A, M23Class::C6A],
    };
    let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
    assert_eq!(
        (c.size, c.real, c.b_class, c.a_class),
        (70, 4, M23Class::C6A, M23Class::C3A)
    );
    println!("18 rows in debug: {:?}", t0.elapsed());
}

// The cheap Err rows of design Section 9 and review 2, n1 and n2.
#[test]
fn degenerate_cases() {
    let (g, id) = setup();
    let t = AtlasTable::m23();
    let lists = ClassLists::by_orbit(&g, &id, &[M23Class::C1A, M23Class::C2A], ORBIT_CAP).unwrap();
    // n1: invalid shapes are refused
    for (kind, classes) in [
        (
            VectorKind::ConjugatePair,
            [M23Class::C3A, M23Class::C7A, M23Class::C7A],
        ),
        (
            VectorKind::ConjugatePair,
            [M23Class::C7A, M23Class::C3A, M23Class::C3A],
        ),
        (
            VectorKind::ConjugatePair,
            [M23Class::C2A, M23Class::C7A, M23Class::C11B],
        ),
        (
            VectorKind::Rational,
            [M23Class::C2A, M23Class::C7A, M23Class::C7B],
        ),
    ] {
        let v = ClassVector { kind, classes };
        assert!(
            matches!(v.validate(), Err(ThreePointError::InvalidClassVector(_))),
            "{v}"
        );
        assert!(matches!(
            count_class_vector(&g, &id, t, &lists, &v),
            Err(ThreePointError::InvalidClassVector(_))
        ));
        assert!(matches!(
            structure_count(&g, &id, t, &lists, &v),
            Err(ThreePointError::InvalidClassVector(_))
        ));
    }
    // all orders in {1, 11, 23}: no size, but structure_count works (n2); n = 1/|G|
    let v = ClassVector {
        kind: VectorKind::Rational,
        classes: [M23Class::C1A; 3],
    };
    assert_eq!(
        count_class_vector(&g, &id, t, &lists, &v),
        Err(ThreePointError::GenerationCriterionNotApplicable)
    );
    let s = structure_count(&g, &id, t, &lists, &v).unwrap();
    assert_eq!(s.survivors, 1);
    assert_eq!(s.structure_constant, Rational::new(1, 10_200_960).unwrap());
    assert_eq!(s.centralizer_order, 10_200_960);
    // n = 0 and negative genus: (2A, 2A, 8A) has no survivors (no involution inverts 8A)
    let v = ClassVector {
        kind: VectorKind::Rational,
        classes: [M23Class::C2A, M23Class::C2A, M23Class::C8A],
    };
    let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
    assert_eq!((c.survivors, c.size, c.real), (0, 0, 0));
    assert_eq!(c.structure_constant, Rational::from(0i64));
    assert!(c.genus23 < 0);
    // a missing class list
    let v = ClassVector {
        kind: VectorKind::ConjugatePair,
        classes: [M23Class::C4A, M23Class::C23A, M23Class::C23B],
    };
    assert_eq!(
        count_class_vector(&g, &id, t, &lists, &v),
        Err(ThreePointError::ClassListMissing(M23Class::C4A))
    );
    // the real count needs 2A
    let only3 = ClassLists::by_orbit(&g, &id, &[M23Class::C3A], ORBIT_CAP).unwrap();
    let v = ClassVector {
        kind: VectorKind::ConjugatePair,
        classes: [M23Class::C3A, M23Class::C23A, M23Class::C23B],
    };
    assert_eq!(
        count_class_vector(&g, &id, t, &lists, &v),
        Err(ThreePointError::ClassListMissing(M23Class::C3A))
    );
    assert_eq!(
        count_class_vector(&g, &id, t, &only3, &v),
        Err(ThreePointError::ClassListMissing(M23Class::C2A))
    );
    assert!(structure_count(&g, &id, t, &only3, &v).is_ok());
}

/// All 57 rows of m23_three_point_nielsen.json exactly, with the identity, divisibility and
/// parity on each, |C(b)| computed by RustMath and equal to GAP's; and the a = c case on the
/// 21 rational multisets that contain 2A. Design: 3.6 s, 78 MB (release).
#[test]
#[ignore]
fn full_57_rows() {
    let t0 = Instant::now();
    let (g, id) = setup();
    let t = AtlasTable::m23();
    let rational = [
        M23Class::C2A,
        M23Class::C3A,
        M23Class::C4A,
        M23Class::C5A,
        M23Class::C6A,
        M23Class::C8A,
    ];
    let t1 = Instant::now();
    let lists = ClassLists::by_enumeration(&g, &id, &rational).unwrap();
    println!("the six rational class lists: {:?}", t1.elapsed());
    let t2 = Instant::now();
    let mut worst = (std::time::Duration::ZERO, String::new());
    let mut lines = Vec::new();
    for row in nielsen_rows() {
        let v = vector_of(&row);
        let tr = Instant::now();
        let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
        check_row(&c, &row);
        let el = tr.elapsed();
        if el > worst.0 {
            worst = (el, v.to_string());
        }
        lines.push(format!(
            "{:<15} {:<16} g={} n={} |C(b)|={} surv={} size={} real={} reproduced",
            v.kind.name(),
            v.to_string(),
            c.genus23,
            c.structure_constant,
            c.centralizer_order,
            c.survivors,
            c.size,
            c.real
        ));
    }
    assert_eq!(lines.len(), 57);
    for l in &lines {
        println!("{l}");
    }
    println!(
        "57/57 rows reproduced: {:?}; slowest {:?} {}",
        t2.elapsed(),
        worst.0,
        worst.1
    );
    // the a = c case on every rational multiset containing 2A (any genus)
    let t3 = Instant::now();
    let mut ac = Vec::new();
    for (i, &x) in rational.iter().enumerate() {
        for &y in &rational[i..] {
            let v = ClassVector {
                kind: VectorKind::Rational,
                classes: [M23Class::C2A, x, y],
            };
            let c = count_class_vector(&g, &id, t, &lists, &v).unwrap();
            assert_eq!(c.a_equals_c_generating, 0, "{v}");
            if c.a_class == M23Class::C2A {
                ac.push(format!("{v}:{}", c.a_equals_c_candidates));
            }
        }
    }
    println!(
        "a = c candidates, none generating: {} ({:?})",
        ac.join(" "),
        t3.elapsed()
    );
    println!(
        "full_57_rows total {:?}, VmHWM {} MB",
        t0.elapsed(),
        m23_common::vm_hwm_mb()
    );
}

//! M23 on the published generators, its classes and centralizers (WP-P, HANDOFF 4.2 steps 1
//! to 3). MAGMA ch. 58.
//!
//! Fast tests run in debug. The opt-in release tests label all 10,200,960 elements:
//!
//! ```text
//! cargo test --release -p rustmath-groups --test m23_classes -- --ignored --exact full_census
//! cargo test --release -p rustmath-groups --test m23_classes -- --ignored --exact full_family_11
//! ```
//!
//! References: GAP 4.12.1 with CTblLib 1.3.7, `CharacterTable("M23")`
//! (`SizesConjugacyClasses`, `SizesCentralizers`, written out below); the owner's
//! `m23golay/m23.py` (`ORDER`, the base orbit lengths of HANDOFF 4.2 step 1) and
//! `data/group_facts.json` (`signs`), vendored in `tests/data/m23/`.

mod m23_common;

use std::time::Instant;

use rustmath_coding::cyclic_golay::GolaySign;
use rustmath_combinatorics::designs::is_steiner_system;
use rustmath_groups::m23::classes::{signature, ORBIT_CAP, SIFT_BUDGET};
use rustmath_groups::m23::group::{
    from_cycles_1based, identity23, inverse, power, rmul, M23_ORDER,
};
use rustmath_groups::m23::{
    census, centralizer_order_by_sifting, conjugacy_class_by_orbit, ClassIdentifier, M23Class,
    M23Error, M23,
};
use rustmath_groups::transitive23::cycle_type;

/// GAP 4.12.1: `SizesConjugacyClasses(CharacterTable("M23"))`, ATLAS order.
const GAP_CLASS_SIZES: [u64; 17] = [
    1, 3795, 56672, 318780, 680064, 850080, 728640, 728640, 1275120, 927360, 927360, 728640,
    728640, 680064, 680064, 443520, 443520,
];
/// GAP 4.12.1: `SizesCentralizers(CharacterTable("M23"))`, ATLAS order.
const GAP_CENTRALIZERS: [u64; 17] = [
    10200960, 2688, 180, 32, 15, 12, 14, 14, 8, 11, 11, 14, 14, 15, 15, 23, 23,
];

fn setup() -> (M23, ClassIdentifier) {
    let g = M23::published().expect("M23 on the published generators");
    let id = ClassIdentifier::new(&g).expect("class identifier");
    (g, id)
}

// The owner's m23.py: ORDER = 10_200_960; HANDOFF 4.2 step 1: base of 6 points with orbit
// lengths 23, 22, 21, 20, 3, 16 for the published generators. GAP: Size(Group(g1, g2)) =
// 10200960 (design review 2, rv2_rows.g).
#[test]
fn m23_published_order_and_base() {
    let g = M23::published().unwrap();
    assert_eq!(g.chain().order().to_string(), "10200960");
    assert_eq!(M23_ORDER, 10_200_960);
    assert_eq!(g.base_orbit_lengths(), vec![23, 22, 21, 20, 3, 16]);
    assert_eq!(rmul(&rmul(&g.g1(), &g.g2()), &g.g3()), identity23());
    assert_eq!(g.g3(), inverse(&rmul(&g.g1(), &g.g2())));
    assert!(g.contains(&g.g1()) && g.contains(&g.g2()) && g.contains(&g.g3()));
    // An odd permutation is not in M23.
    let t = from_cycles_1based(&[[1u8, 2]]).unwrap();
    assert!(!g.contains(&t));
}

// 253 heptads forming an S(4,7,23) (Witt 1938; owner's verify_group_theory.py); the even Golay
// code has dimension 11 and weight distribution 1 + 506 x^8 + 1288 x^12 + 253 x^16
// (MacWilliams, Sloane 1977, ch. 20).
#[test]
fn steiner_and_even_golay_code() {
    let g = M23::published().unwrap();
    let h = g.heptads();
    assert_eq!(h.len(), 253);
    assert!(h.iter().all(|m| m.count_ones() == 7));
    let blocks: Vec<Vec<usize>> = h
        .iter()
        .map(|&m| (0..23).filter(|&i| (m >> i) & 1 == 1).collect())
        .collect();
    assert!(is_steiner_system(23, 4, &blocks).unwrap());
    let basis = g.even_golay_basis();
    assert_eq!(basis.len(), 11);
    let mut weights = [0u64; 24];
    for m in 0u32..(1 << 11) {
        let w = (0..11)
            .filter(|&k| (m >> k) & 1 == 1)
            .fold(0u32, |w, k| w ^ basis[k].0);
        weights[w.count_ones() as usize] += 1;
    }
    let mut want = [0u64; 24];
    want[0] = 1;
    want[8] = 506;
    want[12] = 1288;
    want[16] = 253;
    assert_eq!(weights, want);
    // the Steiner system is preserved by the generators
    for p in [g.g1(), g.g2()] {
        let mut img: Vec<u32> = h
            .iter()
            .map(|&m| rustmath_groups::m23::group::mask_image(m, &p))
            .collect();
        img.sort_unstable();
        assert_eq!(img, h);
    }
}

// group_facts.json, signs: g2 (23A) has sign -1 and g3 (23B) has sign +1. The conventions of
// design 3.3: 23A is the class of g2, 7A and 15A have Golay trace 1, 14A squares into 7A.
#[test]
fn split_invariants_separate() {
    let (g, id) = setup();
    let facts = m23_common::vendored("group_facts.json");
    assert_eq!(facts["signs"]["g2 (23A of HJLPPZ)"].as_i64(), Some(-1));
    assert_eq!(facts["signs"]["g3 (23B of HJLPPZ)"].as_i64(), Some(1));
    assert_eq!(id.cyclic_steiner_sign(&g.g2()).unwrap(), GolaySign::Minus);
    assert_eq!(id.cyclic_steiner_sign(&g.g3()).unwrap(), GolaySign::Plus);
    assert_eq!(id.representative(M23Class::C23A), g.g2());
    assert_eq!(id.class_of_member(&g.g2()).unwrap(), M23Class::C23A);
    assert_eq!(id.class_of_member(&g.g3()).unwrap(), M23Class::C23B);
    assert_eq!(id.class_of_member(&g.g1()).unwrap(), M23Class::C2A);
    for c in M23Class::ALL {
        let x = id.representative(c);
        assert!(g.contains(&x), "{c} representative in M23");
        assert_eq!(id.class_of_member(&x).unwrap(), c, "{c}");
        assert_eq!(
            id.class_of_member(&inverse(&x)).unwrap(),
            c.inverse(),
            "{c}^-1"
        );
        let ct: Vec<u8> = cycle_type(&x).iter().map(|&l| l as u8).collect();
        assert_eq!(ct, c.cycle_type(), "{c} cycle type");
        assert_eq!(rustmath_groups::m23::group::order(&x), c.element_order());
    }
    // the trace separates 7, 14, 15, 23, with value 1 on 7A, 14A, 15A
    for (a, b) in [
        (M23Class::C7A, M23Class::C7B),
        (M23Class::C14A, M23Class::C14B),
        (M23Class::C15A, M23Class::C15B),
        (M23Class::C23A, M23Class::C23B),
    ] {
        let (ta, tb) = (
            id.golay_trace(&id.representative(a)),
            id.golay_trace(&id.representative(b)),
        );
        assert_ne!(ta, tb, "trace separates {a} {b}");
    }
    for c in [M23Class::C7A, M23Class::C14A, M23Class::C15A] {
        assert_eq!(id.golay_trace(&id.representative(c)), 1, "{c}");
    }
    let x14 = id.representative(M23Class::C14A);
    assert_eq!(
        id.class_of_member(&rmul(&x14, &x14)).unwrap(),
        M23Class::C7A
    );
    // the trace does not separate 11AB; the family does
    let x11 = id.representative(M23Class::C11A);
    assert_eq!(id.golay_trace(&x11), id.golay_trace(&inverse(&x11)));
    let fa = id.heptad_family_11(&x11).unwrap();
    let fb = id.heptad_family_11(&inverse(&x11)).unwrap();
    assert_ne!(fa, fb);
    // the family is preserved by g -> g^k exactly for k a square mod 11 (HANDOFF 4.2 step 3)
    for k in 1..11i64 {
        let fk = id.heptad_family_11(&power(&x11, k)).unwrap();
        let square = [1, 3, 4, 5, 9].contains(&k);
        assert_eq!(fk == fa, square, "k = {k}");
    }
    // the sign separates 23AB and agrees with the trace on the representatives
    assert_ne!(
        id.cyclic_steiner_sign(&g.g2()).unwrap(),
        id.cyclic_steiner_sign(&inverse(&g.g2())).unwrap()
    );
}

// GAP: SizesCentralizers(CharacterTable("M23")). 1A, 2A, 3A by conjugation orbits (1, 3,795,
// 56,672 elements), 4A..23B by sifting C_Sym(x).
#[test]
fn centralizers_match_gap() {
    let (g, id) = setup();
    let t0 = Instant::now();
    for c in M23Class::ALL {
        assert_eq!(
            id.centralizer_order(&g, c).unwrap(),
            GAP_CENTRALIZERS[c.index()],
            "{c}"
        );
        assert_eq!(
            GAP_CLASS_SIZES[c.index()] * GAP_CENTRALIZERS[c.index()],
            M23_ORDER,
            "{c}"
        );
    }
    // cached: a second call is free
    assert_eq!(id.centralizer_order(&g, M23Class::C2A).unwrap(), 2688);
    for (c, len) in [
        (M23Class::C1A, 1usize),
        (M23Class::C2A, 3795),
        (M23Class::C3A, 56672),
    ] {
        let orbit = conjugacy_class_by_orbit(&g, &id.representative(c), ORBIT_CAP).unwrap();
        assert_eq!(orbit.len(), len, "{c}");
        assert!(orbit.iter().all(|x| id.class_of_member(x).unwrap() == c));
    }
    println!("17 centralizers and 3 orbits: {:?}", t0.elapsed());
}

// The cheap Err cases of design Section 9 that concern the classes.
#[test]
fn degenerate_class_cases() {
    let (g, id) = setup();
    // a 2^8 1^7 involution outside M23: called 2A by class_of_member (unspecified), refused by
    // class_of_checked
    let inv_out = from_cycles_1based(&[
        [1u8, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
    ])
    .unwrap();
    assert!(!g.contains(&inv_out));
    assert_eq!(id.class_of_member(&inv_out).unwrap(), M23Class::C2A);
    assert_eq!(id.class_of_checked(&g, &inv_out), Err(M23Error::NotInM23));
    // a 22-cycle has the signature of the 11-cycles
    let c22: Vec<u8> = (1..=22).collect();
    let c22 = from_cycles_1based(&[c22]).unwrap();
    assert_eq!(signature(&c22), (1, 1, 1));
    assert!(id.class_of_member(&c22).is_ok());
    assert_eq!(id.class_of_checked(&g, &c22), Err(M23Error::NotInM23));
    assert!(matches!(
        id.heptad_family_11(&c22),
        Err(M23Error::InvariantFailed(_))
    ));
    // a transposition has a signature outside the 12 types
    let t = from_cycles_1based(&[[1u8, 2]]).unwrap();
    assert_eq!(id.class_of_member(&t), Err(M23Error::NotInM23));
    // a 23-cycle outside M23: the cyclic sign fails
    let c23: Vec<u8> = (1..=23).collect();
    let c23 = from_cycles_1based(&[c23]).unwrap();
    assert!(!g.contains(&c23));
    assert!(matches!(
        id.cyclic_steiner_sign(&c23),
        Err(M23Error::InvariantFailed(_))
    ));
    assert!(matches!(
        id.cyclic_steiner_sign(&g.g1()),
        Err(M23Error::InvariantFailed(_))
    ));
    // budgets: C_Sym(1) = Sym(23) has 23! elements (saturates u64), the 4A class has 318,780
    match centralizer_order_by_sifting(&g, &identity23(), SIFT_BUDGET) {
        Err(M23Error::BudgetExceeded { needed, budget }) => {
            assert_eq!(needed, u64::MAX);
            assert_eq!(budget, SIFT_BUDGET);
        }
        other => panic!("{other:?}"),
    }
    // review 1: a saturated size is above every budget, u64::MAX included (no 23! loop)
    assert_eq!(
        centralizer_order_by_sifting(&g, &identity23(), u64::MAX),
        Err(M23Error::BudgetExceeded {
            needed: u64::MAX,
            budget: u64::MAX
        })
    );
    assert!(matches!(
        centralizer_order_by_sifting(&g, &id.representative(M23Class::C2A), SIFT_BUDGET),
        Err(M23Error::BudgetExceeded { .. })
    ));
    assert!(matches!(
        conjugacy_class_by_orbit(&g, &id.representative(M23Class::C4A), 1000),
        Err(M23Error::BudgetExceeded { budget: 1000, .. })
    ));
    assert_eq!(
        centralizer_order_by_sifting(&g, &inv_out, SIFT_BUDGET),
        Err(M23Error::NotInM23)
    );
    // malformed cycles
    assert!(from_cycles_1based(&[[1u8, 24]]).is_err());
    assert!(from_cycles_1based(&[[1u8, 2], [2, 3]]).is_err());
    assert!(from_cycles_1based(&[[0u8, 2]]).is_err());
    // names
    for c in M23Class::ALL {
        assert_eq!(M23Class::from_name(c.name()), Some(c));
        assert_eq!(M23Class::from_name(&c.name().to_lowercase()), Some(c));
        assert_eq!(c.inverse().inverse(), c);
        assert_eq!(c.is_rational(), c.inverse() == c);
    }
    assert_eq!(M23Class::from_name("9A"), None);
}

/// Labels of all 10,200,960 elements (fast 11 test): the 17 class sizes; in the same pass the
/// cyclic sign equals the trace on all 887,040 elements of order 23; then the 17 centralizers
/// by scanning. Reference: GAP (constants above). Design: 8.9 s, 3 MB (release).
#[test]
#[ignore]
fn full_census() {
    let t0 = Instant::now();
    let (g, id) = setup();
    let c = census(&g, &id).unwrap();
    assert_eq!(c.class_sizes, GAP_CLASS_SIZES);
    assert_eq!(c.sign_agrees_on_23, 887_040);
    assert_eq!(c.centralizer_orders, GAP_CENTRALIZERS);
    assert_eq!(c.class_sizes.iter().sum::<u64>(), M23_ORDER);
    println!(
        "full_census total {:?}, VmHWM {} MB",
        t0.elapsed(),
        m23_common::vm_hwm_mb()
    );
}

/// The fast 11 test equals the family test on all 1,854,720 elements of order 11 (927,360 in
/// 11A). Redundant by the exactness argument of classes.rs; kept as its own opt-in test.
/// Design: 40.3 s, 3 MB (release).
#[test]
#[ignore]
fn full_family_11() {
    let t0 = Instant::now();
    let (g, id) = setup();
    let fam_a = id
        .heptad_family_11(&id.representative(M23Class::C11A))
        .unwrap();
    let (mut n11, mut na) = (0u64, 0u64);
    g.for_each_element(|p| {
        if signature(p) == (1, 1, 1) {
            n11 += 1;
            let fast = id.class_of_member(p).unwrap() == M23Class::C11A;
            let slow = id.heptad_family_11(p).unwrap() == fam_a;
            assert_eq!(fast, slow);
            na += fast as u64;
        }
    });
    assert_eq!((n11, na), (1_854_720, 927_360));
    println!(
        "full_family_11 total {:?}, VmHWM {} MB",
        t0.elapsed(),
        m23_common::vm_hwm_mb()
    );
}

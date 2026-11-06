//! Integration tests for Agent G's additions (`perm_predicates`, `transitive23`).
//!
//! These live in `tests/` (a separate compilation unit linking the built library)
//! so they run green independently of the crate's pre-existing, unrelated
//! unit-test (`#[cfg(test)]`) compile breakage in other modules (`free_group`,
//! `additive_abelian_wrapper`, …) which Agent G is not permitted to touch.

use rustmath_groups::perm_predicates as pp;
use rustmath_groups::transitive23 as t23;
use rustmath_groups::transitive23::Group23;

// --------------------------- perm_predicates ------------------------------- //

fn cyc(n: usize) -> Vec<usize> {
    (0..n).map(|i| (i + 1) % n).collect()
}

#[test]
fn c23_primitive() {
    let gens = vec![cyc(23)];
    assert!(pp::is_transitive(&gens, 23));
    assert_eq!(pp::orbits(&gens, 23).len(), 1);
    assert!(pp::block_systems(&gens, 23).is_empty());
    assert!(pp::is_primitive(&gens, 23));
}

#[test]
fn c6_imprimitive_two_systems() {
    let gens = vec![cyc(6)];
    assert!(pp::is_transitive(&gens, 6));
    assert!(!pp::is_primitive(&gens, 6));
    let sys = pp::block_systems(&gens, 6);
    assert_eq!(sys.len(), 2);
    assert!(sys.contains(&vec![vec![0, 3], vec![1, 4], vec![2, 5]]));
    assert!(sys.contains(&vec![vec![0, 2, 4], vec![1, 3, 5]]));
}

#[test]
fn degree6_blocks_of_three() {
    // <(0 1 2)(3 4 5), (0 3)(1 4)(2 5)>: blocks {0,1,2},{3,4,5}.
    let a = vec![1, 2, 0, 4, 5, 3];
    let b = vec![3, 4, 5, 0, 1, 2];
    let gens = vec![a, b];
    assert!(pp::is_transitive(&gens, 6));
    assert!(!pp::is_primitive(&gens, 6));
    assert!(pp::block_systems(&gens, 6).contains(&vec![vec![0, 1, 2], vec![3, 4, 5]]));
}

#[test]
fn s4_primitive_and_stabilizer_is_s3() {
    let gens = vec![
        vec![1, 0, 2, 3],
        vec![0, 2, 1, 3],
        vec![0, 1, 3, 2],
    ];
    assert!(pp::is_transitive(&gens, 4));
    assert!(pp::is_primitive(&gens, 4));

    let stab = pp::stabilizer(&gens, 4, 0);
    for s in &stab {
        assert_eq!(s[0], 0);
    }
    // |Stab_{S4}(0)| = |S3| = 6
    fn order(gens: &[Vec<usize>], n: usize) -> usize {
        use std::collections::HashSet;
        let id: Vec<usize> = (0..n).collect();
        let mut set = HashSet::new();
        set.insert(id.clone());
        let mut fr = vec![id];
        while let Some(g) = fr.pop() {
            for s in gens {
                let h: Vec<usize> = g.iter().map(|&i| s[i]).collect();
                if set.insert(h.clone()) {
                    fr.push(h);
                }
            }
        }
        set.len()
    }
    assert_eq!(order(&stab, 4), 6);
}

#[test]
fn intransitive_orbits() {
    let gens = vec![vec![1, 0, 2, 3]];
    assert!(!pp::is_transitive(&gens, 4));
    assert_eq!(pp::orbits(&gens, 4), vec![vec![0, 1], vec![2], vec![3]]);
}

// ----------------------------- transitive23 -------------------------------- //

/// x -> a*x + b mod 23, as an image-list Perm on residues {0,…,22}.
fn affine(a: u64, b: u64) -> t23::Perm {
    let mut p = [0u8; 23];
    for i in 0..23u64 {
        p[i as usize] = ((a * i + b) % 23) as u8;
    }
    p
}

#[test]
fn affine_group_sets_match_closure() {
    let t = affine(1, 1); // translation, 23-cycle
    // C23
    assert_eq!(
        t23::cycle_type_set(&[t], 100).unwrap(),
        Group23::C23.type_set().unwrap()
    );
    // D23 = <t, x -> -x>
    assert_eq!(
        t23::cycle_type_set(&[t, affine(22, 0)], 100).unwrap(),
        Group23::D23.type_set().unwrap()
    );
    // F23 = <t, x -> 2x> (2 has order 11 mod 23)
    assert_eq!(
        t23::cycle_type_set(&[t, affine(2, 0)], 1000).unwrap(),
        Group23::F23.type_set().unwrap()
    );
    // AGL(1,23) = <t, x -> 5x> (5 primitive root, order 22)
    assert_eq!(
        t23::cycle_type_set(&[t, affine(5, 0)], 1000).unwrap(),
        Group23::AGL23.type_set().unwrap()
    );
}

fn full(g: Group23) -> Vec<Vec<usize>> {
    g.type_set().unwrap().into_iter().collect()
}

#[test]
fn classify_all_seven_via_fingerprints() {
    // even chain (disc a square)
    assert_eq!(t23::classify(&full(Group23::C23), true), Some(Group23::C23));
    assert_eq!(t23::classify(&full(Group23::F23), true), Some(Group23::F23));
    assert_eq!(t23::classify(&full(Group23::M23), true), Some(Group23::M23));
    assert_eq!(
        t23::classify(&[vec![1; 20].into_iter().chain([3]).collect()], true),
        Some(Group23::A23)
    ); // a lone 3-cycle: 3·1^20, even, not an M23 type

    // odd chain (disc not a square)
    assert_eq!(t23::classify(&full(Group23::D23), false), Some(Group23::D23));
    assert_eq!(
        t23::classify(&full(Group23::AGL23), false),
        Some(Group23::AGL23)
    );
    assert_eq!(
        t23::classify(&[vec![1; 21].into_iter().chain([2]).collect()], false),
        Some(Group23::S23)
    ); // a transposition: 2·1^21, odd, not an AGL type
}

#[test]
fn m23_vs_a23_vs_s23() {
    // exact M23 fingerprint, square -> M23
    assert_eq!(t23::classify(&full(Group23::M23), true), Some(Group23::M23));

    // M23 fingerprint + an even type it lacks -> A23
    let mut a = full(Group23::M23);
    a.push(vec![1; 20].into_iter().chain([3]).collect());
    assert_eq!(t23::classify(&a, true), Some(Group23::A23));

    // M23 fingerprint + an odd type (=> disc not a square) -> S23
    let mut s = full(Group23::M23);
    s.push(vec![1; 21].into_iter().chain([2]).collect());
    assert_eq!(t23::classify(&s, false), Some(Group23::S23));
}

#[test]
fn candidates_sound_and_odd_type_contradiction() {
    assert_eq!(
        t23::candidates(&full(Group23::C23), true),
        vec![Group23::C23, Group23::F23, Group23::M23, Group23::A23]
    );
    // an odd type is impossible in any subgroup of A23
    let odd: Vec<Vec<usize>> = vec![vec![1; 21].into_iter().chain([2]).collect()];
    assert_eq!(t23::classify(&odd, true), None);
}

#[test]
fn finite_groups_have_distinct_type_sets() {
    let finite = [
        Group23::C23,
        Group23::D23,
        Group23::F23,
        Group23::AGL23,
        Group23::M23,
    ];
    for (i, &a) in finite.iter().enumerate() {
        for &b in &finite[i + 1..] {
            assert_ne!(a.type_set(), b.type_set());
        }
    }
}

// --------------------------- transitive24 part-3 --------------------------- //

/// `Db::load_default` reads the gitignored generator atlas
/// `data/transitive_24.jsonl` (built by GAP, see `data/README.md`). When the file
/// is present, check the real data; when it is absent, it must fail with a clear
/// error pointing at the cycle-type support instead of mis-parsing that file.
#[test]
fn transitive24_db_load_default_loads_or_fails_gracefully() {
    use rustmath_groups::transitive24::{cycle_type_set, group_closure, Db};
    use rustmath_integers::Integer;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/transitive_24.jsonl");
    let started = std::time::Instant::now();
    let res = Db::load_default();
    let elapsed = started.elapsed();

    if !path.exists() {
        assert!(res.is_err());
        let msg = format!("{}", res.err().unwrap());
        assert!(msg.contains("CycleTypeSupport::load_default"), "msg was: {msg}");
        return;
    }

    let db = res.expect("transitive_24.jsonl is present and must parse");
    println!("Db::load_default: {} groups in {:?}", db.groups.len(), elapsed);
    // Reference values: GAP TransGrp (NrTransitiveGroups(24) = 25000, Size,
    // IsPrimitive, IsomorphismGroups with MathieuGroup(24) / PSL(2,23)), which
    // uses LMFDB's numbering in degree 24.
    assert_eq!(db.groups.len(), 25000);
    let ts: std::collections::BTreeSet<usize> = db.groups.iter().map(|g| g.t).collect();
    assert_eq!(ts.len(), 25000);
    assert_eq!((*ts.iter().next().unwrap(), *ts.iter().last().unwrap()), (1, 25000));
    let group = |t: usize| db.groups.iter().find(|g| g.t == t).unwrap();

    // 24T1 = C24: order 24, imprimitive, contains the odd 24-cycle.
    let c24 = group(1);
    assert_eq!(c24.order, Integer::from(24));
    assert!(!c24.primitive);
    assert_eq!(c24.parity, -1);
    assert!(c24.solvable);
    // The stored generators really generate a group of that order.
    assert_eq!(group_closure(&c24.gens, 1000).unwrap().len(), 24);
    assert!(cycle_type_set(&c24.gens, 1000).unwrap().contains(&vec![24]));

    // 24T7817 = PSL(2,23) on the projective line: order 6072, primitive, simple
    // (so all elements are even and it is not solvable).
    let psl = group(7817);
    assert_eq!(psl.order, Integer::from(6072));
    assert!(psl.primitive);
    assert_eq!(psl.parity, 1);
    assert!(!psl.solvable);
    assert_eq!(group_closure(&psl.gens, 10_000).unwrap().len(), 6072);

    // 24T24680 = M24: order 244823040, primitive, simple.
    let m24 = group(24680);
    assert_eq!(m24.order, Integer::from(244_823_040i64));
    assert!(m24.primitive);
    assert_eq!(m24.parity, 1);
    assert!(!m24.solvable);
}

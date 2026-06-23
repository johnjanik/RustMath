//! Phase 4 validation: native Frobenius cycle-type narrowing of Gal(f).
//!  - perm/closure machinery (computes cycle types from generators);
//!  - native group_closure cycle-type sets agree with the precomputed support;
//!  - sharp narrowing (CycleTypeSupport) is sound (true Galois group always a
//!    candidate) and pins Gal(f) to its Frobenius-blind class on real LMFDB fields.
use rustmath_groups::transitive24::*;
use rustmath_integers::Integer;
use rustmath_polynomials::padic_factor::cycle_type as poly_cycle_type;
use std::collections::BTreeSet;

#[rustfmt::skip]
const POLYS: &[(usize, [i64;25])] = &[
    (65, [1, -6, 21, -58, 140, -280, 471, -696, 914, -1074, 1176, -1228, 1221, -1140, 966, -744, 537, -376, 265, -178, 107, -54, 21, -6, 1]),
    (39, [1, -6, 22, -61, 136, -248, 377, -454, 384, -108, -321, 721, -885, 721, -321, -108, 384, -454, 377, -248, 136, -61, 22, -6, 1]),
    (50, [1, -10, 50, -164, 387, -682, 918, -952, 741, -354, -86, 440, -577, 440, -86, -354, 741, -952, 918, -682, 387, -164, 50, -10, 1]),
    (49, [1, 8, 28, 44, -19, -206, -272, 178, 757, 290, -944, -820, 711, 888, -388, -512, 155, 136, -6, -10, -33, 10, 10, -6, 1]),
    (21, [1, 0, 0, 0, -10, 0, 0, 0, -27, 0, 0, 0, -18, 0, 0, 0, -38, 0, 0, 0, 8, 0, 0, 0, 1]),
    (72, [1, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 8, 0, 0, -3, 0, 0, 3, 0, 0, -3, 0, 0, 1]),
    (2, [1, -1, 0, 0, 0, 1, -1, 1, -1, 0, 1, -1, 1, -1, 1, 0, -1, 1, -1, 1, 0, 0, 0, -1, 1]),
    (22, [1, -3, 9, -18, 27, -18, -4, 60, -153, 279, -405, 519, -579, 519, -405, 279, -153, 60, -4, -18, 27, -18, 9, -3, 1]),
];

fn small_prime(n:i64)->bool{ if n<2 {return false;} let mut d=2; while d*d<=n { if n%d==0 {return false;} d+=1;} true }

fn observed_types(coeffs:&[i64])->Vec<Vec<usize>>{
    let f: Vec<Integer> = coeffs.iter().map(|&x| Integer::from(x)).collect();
    let mut obs: Vec<Vec<usize>> = Vec::new();
    let mut p = 2i64;
    while p < 2000 {
        if small_prime(p) { if let Some(ct)=poly_cycle_type(&f,p){ if !obs.contains(&ct){obs.push(ct);} } }
        p += 1;
    }
    obs
}

#[test]
fn perm_basics() {
    let p = perm_from_cycles(&[vec![1,2,3], vec![4,5]]);
    let ct = cycle_type(&p);
    assert_eq!(ct.iter().sum::<usize>(), 24);
    assert!(ct.contains(&2) && ct.contains(&3));
    assert!(is_odd_type(&vec![24]));
    assert!(!is_odd_type(&vec![2;12]));
    assert_eq!(type_order(&vec![2,3,4]), 12);
    let c = perm_from_cycles(&[(1..=24).collect()]);
    assert_eq!(group_closure(&[c], 1000).unwrap().len(), 24);
}

#[test]
fn db_and_support_load() {
    let db = Db::load_default().expect("transitive_24.jsonl present");
    assert_eq!(db.groups.len(), 25000);
    assert_eq!(db.groups.iter().find(|g| g.t==1).unwrap().order, Integer::from(24));
    assert!(db.groups.iter().find(|g| g.t==24680).unwrap().primitive); // M24
    let sup = CycleTypeSupport::load_default().expect("cycletypes file present");
    assert_eq!(sup.by_t.len(), 25000);
}

#[test]
fn native_closure_matches_precomputed_support() {
    // For the (small) validation groups, the native group_closure cycle-type set
    // must equal the precomputed support — cross-validates both.
    let db = Db::load_default().unwrap();
    let sup = CycleTypeSupport::load_default().unwrap();
    for (t, _) in POLYS {
        let g = db.groups.iter().find(|g| g.t==*t).unwrap();
        let native: BTreeSet<Vec<usize>> = cycle_type_set(&g.gens, 100_000).expect("small group");
        let pre: BTreeSet<Vec<usize>> = sup.by_t[t].iter().cloned().collect();
        assert_eq!(native, pre, "native vs precomputed cycle-type set 24T{}", t);
    }
}

#[test]
fn sharp_narrowing_sound_and_blind_class() {
    let sup = CycleTypeSupport::load_default().unwrap();
    for (true_t, coeffs) in POLYS {
        let obs = observed_types(coeffs);
        let superset = sup.candidates(&obs);          // sound: support ⊇ observed
        let blind = sup.blind_class(&obs);            // sharp: support == observed
        assert!(superset.contains(true_t), "true group 24T{} must be in the sound superset", true_t);
        assert!(blind.contains(true_t), "true group 24T{} must be in its blind class (enough primes)", true_t);
        println!("24T{}: {} observed types -> {} sound superset, {} blind class", true_t, obs.len(), superset.len(), blind.len());
    }
}

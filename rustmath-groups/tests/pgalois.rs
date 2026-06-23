//! Validation of the general-degree local Galois group namer.

use rustmath_groups::pgalois::*;
use rustmath_integers::Integer;

fn iz(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

#[test]
fn transitive_group_orbit_signatures_and_parity() {
    // Degree-5 transitive groups: k=1 always [5]; k=2 (pairs) separates the
    // 2-transitive groups ([10]) from the imprimitive ([5,5]).
    let groups = transitive_groups(5);
    let by = |name: &str| groups.iter().find(|g| g.name == name).unwrap().clone();

    assert_eq!(orbit_signature(&by("C5").gens, 5), vec![vec![5], vec![5, 5]]);
    assert_eq!(orbit_signature(&by("D5").gens, 5), vec![vec![5], vec![5, 5]]);
    assert_eq!(orbit_signature(&by("F20").gens, 5), vec![vec![5], vec![10]]);
    assert_eq!(orbit_signature(&by("A5").gens, 5), vec![vec![5], vec![10]]);
    assert_eq!(orbit_signature(&by("S5").gens, 5), vec![vec![5], vec![10]]);

    // Parity (⊆ A_5): C5, D5, A5 even; F20, S5 contain odd permutations.
    assert!(group_in_alternating(&by("C5").gens));
    assert!(group_in_alternating(&by("D5").gens));
    assert!(group_in_alternating(&by("A5").gens));
    assert!(!group_in_alternating(&by("F20").gens));
    assert!(!group_in_alternating(&by("S5").gens));
}

#[test]
fn unramified_local_galois_matches_pari_gp() {
    // x⁵ − x − 1 (disc 2869 = 19·151, global S5). Frobenius cycle type = factor
    // degrees of f mod p; |Gal(f/ℚ_p)| = lcm. Cross-checked vs PARI/GP factormod.
    let f = iz(&[-1, -1, 0, 0, 0, 1]);
    let u = |p: i64| unramified_local_galois_group(&f, p).unwrap();

    assert_eq!(u(11).frobenius_cycle_type, vec![5]); // irreducible ⇒ C5
    assert_eq!(u(11).order, 5);
    assert!(u(11).transitive);

    assert_eq!(u(13).frobenius_cycle_type, vec![5]);

    assert_eq!(u(7).frobenius_cycle_type, vec![2, 3]); // C6 as (2,3) on 5 points
    assert_eq!(u(7).order, 6);
    assert!(!u(7).transitive);

    assert_eq!(u(23).frobenius_cycle_type, vec![1, 4]);
    assert_eq!(u(23).order, 4);

    assert_eq!(u(47).frobenius_cycle_type, vec![1, 1, 3]);
    assert_eq!(u(47).order, 3);

    // 19 | disc ⇒ ramified ⇒ None.
    assert!(unramified_local_galois_group(&f, 19).is_none());
}

#[test]
fn candidates_exact_at_unramified_and_sound_at_ramified() {
    // Unramified, f irreducible mod p ⇒ exactly C5 (5T1).
    let f = iz(&[-1, -1, 0, 0, 0, 1]); // x⁵−x−1
    let (labels, exact) = local_galois_candidates(&f, 11);
    assert_eq!(labels, vec!["5T1"]);
    assert!(exact);

    // Ramified quartic x⁴−2 over ℚ_2: disc −2048 nonsquare ⇒ Gal ⊄ A_4. The sound
    // candidate set is the odd-containing transitive quartic groups {C4, D4, S4}; the
    // true group D4 is included (full separation needs the resolvent/tower machinery).
    let (labels4, exact4) = local_galois_candidates(&iz(&[-2, 0, 0, 0, 1]), 2);
    assert!(labels4.contains(&"4T3")); // D4 retained (sound)
    assert!(!labels4.contains(&"4T2")); // V4 (even) excluded
    assert!(!labels4.contains(&"4T4")); // A4 (even) excluded
    assert!(!exact4); // not yet unique from parity alone
}

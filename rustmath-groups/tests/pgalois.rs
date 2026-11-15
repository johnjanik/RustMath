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

// ---------------------------------------------------------------------------------------
// Prime checks, non-monic and non-primitive inputs (follow-up unit P1). References,
// PARI/GP 2.15, with deg(F) = vecsort(apply(poldegree, F[,1]~)):
//   poldisc(2*x^2+x+1) = -7,           deg(factorpadic(2*x^2+x+1, 2, 8)) = [1, 1]
//   poldisc(2*x^3+x^2+x+1) = -83,      deg(factorpadic(2*x^3+x^2+x+1, 2, 8)) = [1, 2]
//   poldisc(3*x^4+x^3+2*x+1) = -563,   deg(factorpadic(3*x^4+x^3+2*x+1, 3, 8)) = [1, 3]
//   poldisc(2*x+1) = 1,                deg(factorpadic(2*x+1, 2, 8)) = [1]
//   factormod(x^2+1, 7) irreducible,   deg(factorpadic(x^2+1, 7, 8)) = [2]
//   poldisc(6*x+6) = poldisc(2*x) = poldisc(-2*x+4) = 1, one factorpadic factor of degree 1
//   poldisc(3*x^2+3) = -36, poldisc(x^2+1) = -4, deg(factorpadic(3*x^2+3, 3, 8)) = [2]
//   poldisc(x^4+x+1) = 229, deg(factormod(x^4+x+1, 3)) = [1, 3] = deg(factorpadic(3*(x^4+x+1), 3, 8))
//   poldisc(x^5-x-1) = 2869, deg(factormod(x^5-x-1, 2)) = [2, 3], deg(factormod(x^5-x-1, 3)) = [5]
// ---------------------------------------------------------------------------------------

#[test]
#[should_panic(expected = "pgalois::unramified_local_galois_group: p = 15 is not a prime")]
fn unramified_rejects_composite_p() {
    unramified_local_galois_group(&iz(&[1, 0, 1]), 15);
}

#[test]
#[should_panic(expected = "is not a prime")]
fn unramified_rejects_p_zero() {
    unramified_local_galois_group(&iz(&[1, 0, 1]), 0);
}

#[test]
#[should_panic(expected = "is not a prime")]
fn unramified_rejects_p_one() {
    unramified_local_galois_group(&iz(&[1, 0, 1]), 1);
}

#[test]
#[should_panic(expected = "pgalois::local_galois_candidates: p = 15 is not a prime")]
fn candidates_reject_composite_p() {
    local_galois_candidates(&iz(&[-1, -1, 0, 0, 0, 1]), 15);
}

#[test]
fn unramified_non_monic_adds_root_at_infinity() {
    let u = |f: &[i64], p: i64| unramified_local_galois_group(&iz(f), p).unwrap();
    let a = u(&[1, 1, 2], 2);
    assert_eq!((a.frobenius_cycle_type, a.order, a.transitive), (vec![1, 1], 1, false));
    let b = u(&[1, 1, 1, 2], 2);
    assert_eq!((b.frobenius_cycle_type, b.order, b.transitive), (vec![1, 2], 2, false));
    let c = u(&[1, 2, 0, 1, 3], 3);
    assert_eq!((c.frobenius_cycle_type, c.order, c.transitive), (vec![1, 3], 3, false));
    let d = u(&[1, 2], 2);
    assert_eq!((d.frobenius_cycle_type, d.order, d.transitive), (vec![1], 1, true));
}

#[test]
fn unramified_untrimmed_input() {
    let u = unramified_local_galois_group(&iz(&[1, 0, 1, 0]), 7).unwrap();
    assert_eq!((u.frobenius_cycle_type, u.order, u.transitive), (vec![2], 2, true));
}

#[test]
fn unramified_linear_with_content() {
    for (f, p) in [(vec![6, 6], 3), (vec![0, 2], 2), (vec![4, -2], 2)] {
        let u = unramified_local_galois_group(&iz(&f), p).unwrap();
        assert_eq!(
            (u.frobenius_cycle_type, u.order, u.transitive),
            (vec![1], 1, true),
            "{f:?} at {p}"
        );
    }
}

#[test]
fn unramified_uses_primitive_part() {
    let u = unramified_local_galois_group(&iz(&[3, 0, 3]), 3).unwrap();
    assert_eq!((u.frobenius_cycle_type, u.order, u.transitive), (vec![2], 2, true));
}

#[test]
fn candidates_use_primitive_part() {
    assert_eq!(local_galois_candidates(&iz(&[3, 3, 0, 0, 3]), 3), (vec![], true));
    assert_eq!(local_galois_candidates(&iz(&[-2, -2, 0, 0, 0, 2]), 2), (vec![], true));
    assert_eq!(local_galois_candidates(&iz(&[-2, -2, 0, 0, 0, 2]), 3), (vec!["5T1"], true));
}

#[test]
fn candidates_constant_makes_no_claim() {
    assert_eq!(local_galois_candidates(&iz(&[5]), 3), (vec![], false));
    assert_eq!(local_galois_candidates(&iz(&[]), 3), (vec![], false));
}

#[test]
#[should_panic(expected = "f must have degree >= 1")]
fn unramified_rejects_constant() {
    unramified_local_galois_group(&iz(&[5]), 3);
}

#[test]
#[should_panic(expected = "pgalois::is_padic_square: p = 15 is not a prime")]
fn is_padic_square_rejects_composite() {
    is_padic_square(&Integer::from(4), 15);
}

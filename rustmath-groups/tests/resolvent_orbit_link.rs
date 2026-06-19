//! End-to-end link (Module 18/19): the irreducible-factor degrees of a
//! polynomial's k-subset resolvent equal the orbit lengths of its Galois group on
//! k-subsets. Validates the polynomial side (rustmath-polynomials) against the
//! group side (rustmath-groups) at k = 2 and k = 3.

use rustmath_groups::ksubset_orbits::orbit_lengths_on_ksubsets;
use rustmath_groups::transitive24::{perm_from_cycles, Perm};
use rustmath_integers::Integer;
use rustmath_polynomials::resolvent::{
    pair_sum_resolvent, resolvent_orbit_signature, subset_sum_resolvent,
};

fn iz(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}
fn pc(cycles: &[&[u8]]) -> Perm {
    let v: Vec<Vec<u8>> = cycles.iter().map(|c| c.to_vec()).collect();
    perm_from_cycles(&v)
}

#[test]
fn s5_resolvent_degrees_match_group_orbits() {
    // x⁵ − x − 1 has Galois group S₅, which is transitive on both 2- and 3-subsets.
    let f = iz(&[-1, -1, 0, 0, 0, 1]);
    let s5 = [pc(&[&[1, 2]]), pc(&[&[1, 2, 3, 4, 5]])];

    // k = 2
    let poly2 = resolvent_orbit_signature(&pair_sum_resolvent(&f)).unwrap();
    let grp2 = orbit_lengths_on_ksubsets(&s5, 5, 2);
    assert_eq!(poly2, grp2);
    assert_eq!(poly2, vec![10]);

    // k = 3
    let poly3 = resolvent_orbit_signature(&subset_sum_resolvent(&f, 3)).unwrap();
    let grp3 = orbit_lengths_on_ksubsets(&s5, 5, 3);
    assert_eq!(poly3, grp3);
    assert_eq!(poly3, vec![10]);
}

#[test]
fn c4_resolvent_splits_match_group_orbits() {
    // Φ₅ = x⁴+x³+x²+x+1 has Galois group C₄ on its 4 roots; the pair-sum resolvent
    // splits as [2, 4] — exactly C₄'s orbit lengths on pairs.
    let phi5 = iz(&[1, 1, 1, 1, 1]);
    let c4 = [pc(&[&[1, 2, 4, 3]])];

    let poly2 = resolvent_orbit_signature(&pair_sum_resolvent(&phi5)).unwrap();
    let grp2 = orbit_lengths_on_ksubsets(&c4, 4, 2);
    assert_eq!(poly2, grp2);
    assert_eq!(poly2, vec![2, 4]);
}

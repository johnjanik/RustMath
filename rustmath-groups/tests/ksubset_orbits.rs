//! Validation of `k`-subset orbit lengths (Module 18 group side) against
//! hand-computed actions and the PARI/GP-validated resolvent factor degrees.

use rustmath_groups::ksubset_orbits::{orbit_lengths_on_ksubsets, orbit_lengths_on_pairs};
use rustmath_groups::transitive24::{perm_from_cycles, Perm};

fn pc(cycles: &[&[u8]]) -> Perm {
    let v: Vec<Vec<u8>> = cycles.iter().map(|c| c.to_vec()).collect();
    perm_from_cycles(&v)
}

#[test]
fn s4_is_transitive_on_pairs() {
    // S₄ = ⟨(1 2), (1 2 3 4)⟩ on 4 points: one orbit of length C(4,2)=6.
    // Matches the irreducible degree-6 pair-sum resolvent of x⁴+x+1 (PARI/GP).
    let gens = [pc(&[&[1, 2]]), pc(&[&[1, 2, 3, 4]])];
    assert_eq!(orbit_lengths_on_pairs(&gens, 4), vec![6]);
}

#[test]
fn c4_pairs_split_2_and_4() {
    // C₄ = ⟨(1 2 4 3)⟩ (Galois group of Φ₅ on its 4 roots): pairs split into a
    // length-2 orbit and a length-4 orbit — exactly the PARI/GP pair-sum resolvent
    // factor degrees [2, 4] of Φ₅.
    let gens = [pc(&[&[1, 2, 4, 3]])];
    assert_eq!(orbit_lengths_on_pairs(&gens, 4), vec![2, 4]);
}

#[test]
fn a4_is_transitive_on_pairs() {
    // A₄ = ⟨(1 2 3), (2 3 4)⟩ is 2-transitive: single orbit of length 6.
    let gens = [pc(&[&[1, 2, 3]]), pc(&[&[2, 3, 4]])];
    assert_eq!(orbit_lengths_on_pairs(&gens, 4), vec![6]);
}

#[test]
fn klein_four_double_transposition() {
    // V₄ = ⟨(1 2)(3 4), (1 3)(2 4)⟩: pairs {12,34}, {13,24}, {14,23} — three
    // length-2 orbits.
    let gens = [pc(&[&[1, 2], &[3, 4]]), pc(&[&[1, 3], &[2, 4]])];
    assert_eq!(orbit_lengths_on_pairs(&gens, 4), vec![2, 2, 2]);
}

#[test]
fn dihedral_5_pairs_and_triples() {
    // D₅ = ⟨(1 2 3 4 5), (2 5)(3 4)⟩ on 5 points.
    let gens = [pc(&[&[1, 2, 3, 4, 5]]), pc(&[&[2, 5], &[3, 4]])];
    let l = orbit_lengths_on_pairs(&gens, 5);
    assert_eq!(l.iter().sum::<usize>(), 10); // C(5,2)
    assert_eq!(l, vec![5, 5]);
    let l3 = orbit_lengths_on_ksubsets(&gens, 5, 3);
    assert_eq!(l3.iter().sum::<usize>(), 10); // C(5,3)
}

#[test]
fn full_symmetric_24_on_pairs() {
    // S₂₄ = ⟨(1 2), (1 2 … 24)⟩: one orbit of length C(24,2)=276.
    let big: Vec<u8> = (1..=24).collect();
    let gens = [pc(&[&[1, 2]]), pc(&[&big])];
    assert_eq!(orbit_lengths_on_pairs(&gens, 24), vec![276]);
}

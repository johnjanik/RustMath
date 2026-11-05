//! Unit tests for the generic permutation layer.

use rustmath_galois::perm::*;

fn p(v: &[usize]) -> Perm {
    v.to_vec()
}

#[test]
fn compose_and_inverse() {
    let a = from_cycles(4, &[vec![0, 1, 2, 3]]); // (0 1 2 3): 0->1,1->2,2->3,3->0
    assert_eq!(a, p(&[1, 2, 3, 0]));
    let b = from_cycles(4, &[vec![0, 1]]); // (0 1)
    assert_eq!(b, p(&[1, 0, 2, 3]));
    // (a∘b)[i] = a[b[i]]: apply b then a.
    let ab = compose(&a, &b);
    // b: 0->1,1->0 ; then a: 1->2,0->1 ⇒ 0->2, 1->1
    assert_eq!(ab[0], 2);
    assert_eq!(ab[1], 1);
    // inverse
    let ai = inverse(&a);
    assert_eq!(compose(&a, &ai), identity(4));
    assert_eq!(compose(&ai, &a), identity(4));
}

#[test]
fn cycle_type_and_parity() {
    // (0 1 2 3): one 4-cycle ⇒ type [4], odd (4-1=3 transpositions).
    let a = from_cycles(4, &[vec![0, 1, 2, 3]]);
    assert_eq!(cycle_type(&a), vec![4]);
    assert!(is_odd(&a));
    // (0 1)(2 3): type [2,2], even.
    let b = from_cycles(4, &[vec![0, 1], vec![2, 3]]);
    assert_eq!(cycle_type(&b), vec![2, 2]);
    assert!(!is_odd(&b));
    // identity: type [1,1,1,1], even.
    assert_eq!(cycle_type(&identity(4)), vec![1, 1, 1, 1]);
    assert!(!is_odd(&identity(4)));
}

#[test]
fn symmetric_and_alternating_orders() {
    for n in 2..=5usize {
        let sn = group_order(n, &sym_gens(n), 5000).unwrap();
        let factorial: usize = (1..=n).product();
        assert_eq!(sn, factorial, "|S_{n}| wrong");
        let an = group_order(n, &alt_gens(n), 5000).unwrap();
        assert_eq!(an, factorial / 2, "|A_{n}| wrong");
    }
}

#[test]
fn coset_reps_count_matches_index() {
    // S_4 / A_4 has 2 cosets; S_4 / (point stabiliser S_3) has 4.
    let s4 = group_closure(4, &sym_gens(4), 100).unwrap();
    let a4 = group_closure(4, &alt_gens(4), 100).unwrap();
    let reps = coset_reps(&s4, &a4);
    assert_eq!(reps.len(), 2);
    assert_eq!(index(&s4, &a4), 2);
}

#[test]
fn conjugation_preserves_order() {
    let s4 = group_closure(4, &sym_gens(4), 100).unwrap();
    let a4 = group_closure(4, &alt_gens(4), 100).unwrap();
    // conjugate A_4 by a transposition: still order 12, still a subgroup of S_4.
    let tau = from_cycles(4, &[vec![0, 1]]);
    let conj = conjugate(&tau, &a4);
    let mut cs = conj.clone();
    cs.sort();
    cs.dedup();
    assert_eq!(cs.len(), 12);
    // A_4 is normal in S_4, so the conjugate equals A_4.
    let mut a4s = a4.clone();
    a4s.sort();
    assert_eq!(cs, a4s);
    let _ = s4;
}

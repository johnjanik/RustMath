//! Tests for the subgroup lattice, maximal-subgroup computation, and the
//! small-degree transitive-group identifier.

use rustmath_galois::labels::{fingerprint, identify, transitive_groups};
use rustmath_galois::perm::{alt_gens, group_closure, sym_gens};
use rustmath_galois::subgroups::maximal_subgroups;

#[test]
fn maximal_subgroups_of_s3() {
    // S_3 (order 6): maximal subgroups are A_3 (order 3) and three C_2 (order 2).
    let s3 = group_closure(3, &sym_gens(3), 100).unwrap();
    let max = maximal_subgroups(3, &s3);
    let orders: Vec<usize> = max.iter().map(|m| m.len()).collect();
    let count3 = orders.iter().filter(|&&o| o == 3).count();
    let count2 = orders.iter().filter(|&&o| o == 2).count();
    assert_eq!(count3, 1, "one A_3");
    assert_eq!(count2, 3, "three C_2");
    assert_eq!(max.len(), 4);
}

#[test]
fn maximal_subgroups_of_s4() {
    // S_4 (order 24): maximal subgroups are A_4 (12), three D_4 (8), four S_3 (6).
    let s4 = group_closure(4, &sym_gens(4), 100).unwrap();
    let max = maximal_subgroups(4, &s4);
    let count12 = max.iter().filter(|m| m.len() == 12).count();
    let count8 = max.iter().filter(|m| m.len() == 8).count();
    let count6 = max.iter().filter(|m| m.len() == 6).count();
    assert_eq!(count12, 1, "one A_4");
    assert_eq!(count8, 3, "three D_4");
    assert_eq!(count6, 4, "four S_3");
}

#[test]
fn fingerprints_separate_all_small_transitive_groups() {
    // No two distinct nTt of the same degree share a fingerprint.
    for n in 3..=5usize {
        let groups = transitive_groups(n);
        let fps: Vec<_> = groups
            .iter()
            .map(|g| fingerprint(n, &g.gens, 1000).unwrap())
            .collect();
        for i in 0..fps.len() {
            for j in (i + 1)..fps.len() {
                assert_ne!(
                    fps[i], fps[j],
                    "{} and {} share a fingerprint",
                    groups[i].label, groups[j].label
                );
            }
        }
    }
}

#[test]
fn identify_named_groups() {
    // Identify S_n and A_n directly from their standard generators.
    assert_eq!(identify(3, &sym_gens(3), 1000), Some(("3T2", "S3")));
    assert_eq!(identify(3, &alt_gens(3), 1000), Some(("3T1", "C3")));
    assert_eq!(identify(4, &sym_gens(4), 1000), Some(("4T5", "S4")));
    assert_eq!(identify(4, &alt_gens(4), 1000), Some(("4T4", "A4")));
    assert_eq!(identify(5, &sym_gens(5), 1000), Some(("5T5", "S5")));
    assert_eq!(identify(5, &alt_gens(5), 1000), Some(("5T4", "A5")));
}

#[test]
fn identify_each_table_entry_round_trips() {
    for n in 3..=5usize {
        for g in transitive_groups(n) {
            let id = identify(n, &g.gens, 2000);
            assert_eq!(id, Some((g.label, g.name)), "round-trip failed for {}", g.label);
        }
    }
}

//! Degree-24 narrowing test on the 24T2672 example polynomial.
//!
//! This test requires the degree-24 atlas data files
//! (`rustmath-groups/data/transitive_24.jsonl` and `transitive24_cycletypes.jsonl`).
//! It is `#[ignore]`d by default so the suite runs without the ~30 MB data; run
//! with `cargo test -p rustmath-galois -- --ignored` to exercise it.

use rustmath_galois::deg24::narrow_degree24;
use rustmath_integers::Integer;

fn iz(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

#[test]
#[ignore = "needs the degree-24 atlas data files"]
fn narrow_24t2672_example() {
    // Ascending coefficients of the 24T2672 example polynomial.
    let f = iz(&[
        3, 0, -75, 0, 537, 0, -873, 0, 789, 0, -1212, 0, 2551, 0, -2137, 0, 117, 0, 322, 0, 27, 0,
        -13, 0, 1,
    ]);
    let out = narrow_degree24(&f, 600).expect("atlas loaded");
    // Soundness: the cycle-type candidate class must contain the true group.
    assert!(
        out.candidate_class.contains(&2672),
        "candidate class must contain 24T2672 (sound), got {} candidates",
        out.candidate_class.len()
    );
    // Narrowing must not be empty and must still contain the true group.
    assert!(out.narrowed.contains(&2672), "narrowing dropped the true group (unsound!)");
    // The resolvent descent should strictly narrow (or already be unique).
    assert!(
        out.narrowed.len() <= out.candidate_class.len(),
        "narrowing increased the candidate set"
    );
    eprintln!(
        "24T2672 example: {} cycle-type candidates -> {} after resolvent descent (unique_t = {:?})",
        out.candidate_class.len(),
        out.narrowed.len(),
        out.unique_t
    );
    for (k, sig, remaining) in &out.steps {
        eprintln!("  k={k} orbit signature {sig:?} -> {remaining} candidates");
    }
}

use rustmath_groups::solvtower::{factor_signature, solvable_tower};
use rustmath_groups::transitive24::perm_from_cycles;

// NOTE: perm_from_cycles uses 1-indexed points (cyc[k]-1 internally).
fn sig(gens: &[[u8;24]], cap: usize) -> (bool, Option<usize>, Vec<(usize,usize)>) {
    let t = solvable_tower(gens, cap);
    let mut f = factor_signature(&t); f.sort();
    (t.solvable, t.order, f)
}
#[test] fn s3_chief() {
    let g = [perm_from_cycles(&[vec![1,2,3]]), perm_from_cycles(&[vec![1,2]])];
    let (sv, ord, f) = sig(&g, 100_000);
    assert!(sv); assert_eq!(ord, Some(6)); assert_eq!(f, vec![(2,1),(3,1)]);
}
#[test] fn s4_chief() {
    let g = [perm_from_cycles(&[vec![1,2,3,4]]), perm_from_cycles(&[vec![1,2]])];
    let (sv, ord, f) = sig(&g, 100_000);
    assert!(sv); assert_eq!(ord, Some(24)); assert_eq!(f, vec![(2,1),(2,2),(3,1)]);
}
#[test] fn a5_not_solvable() {
    let g = [perm_from_cycles(&[vec![1,2,3]]), perm_from_cycles(&[vec![3,4,5]])];
    let t = solvable_tower(&g, 100_000);
    assert!(!t.solvable, "A5 must be reported non-solvable");
}

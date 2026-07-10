//! Representations of the symmetric group: Young's seminormal form (MAGMA Chapter 92).
//!
//! For the symmetric group `S_n`, the irreducible representations are indexed by partitions
//! `λ ⊢ n`. This module builds the **seminormal** representing matrices over the rationals
//! (rustmath-matrix `Matrix<Rational>`), together with the hook-length dimension formula
//! and a bridge to the Murnaghan–Nakayama character values (which are the traces of these
//! matrices).
//!
//! | MAGMA intrinsic                          | function here                              |
//! |------------------------------------------|-------------------------------------------|
//! | `SymmetricRepresentationSeminormal`      | [`seminormal_representation`]              |
//! | (adjacent generator `s_i`)               | [`seminormal_transposition`]              |
//! | hook-length dimension `SymmetricCharacterValue` on `id` | [`hook_length_dimension`]   |
//! | `SymmetricCharacterValue`                | [`symmetric_character_value`]             |
//!
//! The seminormal construction follows James–Kerber [JK81, §3.3]. For an adjacent
//! transposition `s_i = (i, i+1)` acting on the basis of standard tableaux `{e_T}`:
//!
//! * let `d = c(i+1) − c(i)` be the axial distance in `T`, where `c(cell) = col − row`;
//! * the diagonal entry is `M[T][T] = 1/d`;
//! * if `s_i·T` (swap the cells holding `i` and `i+1`) is again standard, the off-diagonal
//!   entry in column `T` is `1` when `d < 0` and `1 − 1/d²` when `d > 0`.
//!
//! These generator matrices are involutions satisfying the braid relations, hence extend
//! to a genuine representation via any factorisation of a permutation into adjacent
//! transpositions. Young's **orthogonal** form (which needs `sqrt` / cyclotomic entries and
//! therefore is not rational) is intentionally *not* provided here; see the crate README /
//! integrator notes.
//!
//! Reference: MAGMA Handbook, Chapter 92; James & Kerber, *The Representation Theory of the
//! Symmetric Group* [JK81].

use crate::partitions::Partition;
use crate::permutations::Permutation;
use crate::tableaux::{standard_tableaux, Tableau};
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// The hook length of the cell `(i, j)` (0-indexed) of `λ`, or `None` if off the diagram.
pub fn hook_length(partition: &Partition, i: usize, j: usize) -> Option<usize> {
    let hooks = partition.hook_lengths();
    hooks.get(i).and_then(|row| row.get(j)).copied()
}

/// The dimension of the irreducible `S_n`-module indexed by `λ`, i.e. the number of
/// standard Young tableaux of shape `λ`, via the hook-length formula `n! / ∏ hooks`.
///
/// Computed with arbitrary-precision `Integer` (no `usize` overflow for large `n`).
pub fn hook_length_dimension(partition: &Partition) -> Integer {
    let n = partition.sum();
    let mut numerator = Integer::one();
    for k in 2..=n {
        numerator = numerator * Integer::from(k as u64);
    }
    let mut denominator = Integer::one();
    for row in partition.hook_lengths() {
        for h in row {
            denominator = denominator * Integer::from(h as u64);
        }
    }
    numerator / denominator
}

/// Locate the (row, col) of a value in a tableau (0-indexed), if present.
fn position_of(t: &Tableau, value: usize) -> Option<(usize, usize)> {
    for (r, row) in t.rows().iter().enumerate() {
        for (c, &x) in row.iter().enumerate() {
            if x == value {
                return Some((r, c));
            }
        }
    }
    None
}

/// Swap the cells holding `i` and `i+1` in `t`, returning the new tableau (may or may not
/// be standard).
fn swap_values(t: &Tableau, i: usize) -> Tableau {
    let mut rows: Vec<Vec<usize>> = t.rows().to_vec();
    for row in rows.iter_mut() {
        for x in row.iter_mut() {
            if *x == i {
                *x = i + 1;
            } else if *x == i + 1 {
                *x = i;
            }
        }
    }
    Tableau::new(rows).unwrap()
}

fn key(t: &Tableau) -> Vec<Vec<usize>> {
    t.rows().to_vec()
}

/// Seminormal matrix of the adjacent transposition `s_i = (i, i+1)` (with `1 ≤ i ≤ n-1`)
/// acting on the irreducible `S_n`-module indexed by `λ`, over the rationals.
///
/// The basis is the set of standard tableaux of shape `λ` in the fixed order returned by
/// [`standard_tableaux`].
pub fn seminormal_transposition(partition: &Partition, i: usize) -> Matrix<Rational> {
    let tableaux = standard_tableaux(partition);
    let dim = tableaux.len();
    let mut index: HashMap<Vec<Vec<usize>>, usize> = HashMap::new();
    for (idx, t) in tableaux.iter().enumerate() {
        index.insert(key(t), idx);
    }

    let mut data = vec![Rational::zero(); dim * dim];
    let at = |data: &mut Vec<Rational>, r: usize, c: usize, v: Rational| {
        data[r * dim + c] = v;
    };

    for (col, t) in tableaux.iter().enumerate() {
        let (ri, ci) = position_of(t, i).expect("value i present");
        let (rj, cj) = position_of(t, i + 1).expect("value i+1 present");
        let content_i = ci as i64 - ri as i64;
        let content_j = cj as i64 - rj as i64;
        let d = content_j - content_i; // axial distance, nonzero for standard tableaux

        // Diagonal entry 1/d.
        at(&mut data, col, col, Rational::new(1i64, d).unwrap());

        // Off-diagonal: only when swapping i, i+1 keeps the tableau standard, i.e. the two
        // cells share neither a row nor a column.
        let same_row = ri == rj;
        let same_col = ci == cj;
        if !same_row && !same_col {
            let t2 = swap_values(t, i);
            let row = *index.get(&key(&t2)).expect("swapped tableau is standard");
            let off = if d < 0 {
                Rational::from_integer(1)
            } else {
                Rational::new(d * d - 1, d * d).unwrap()
            };
            at(&mut data, row, col, off);
        }
    }

    Matrix::from_vec(dim, dim, data).unwrap()
}

/// Express a permutation as a word in the adjacent generators `s_1, ..., s_{n-1}`
/// (1-indexed), such that `perm = s_{w[0]} ∘ s_{w[1]} ∘ ⋯` (function composition).
fn adjacent_word(perm: &Permutation) -> Vec<usize> {
    let n = perm.size();
    let mut w: Vec<usize> = (0..n).map(|i| perm.apply(i).unwrap()).collect();
    let mut recorded = Vec::new();
    let mut sorted = false;
    while !sorted {
        sorted = true;
        for k in 0..n.saturating_sub(1) {
            if w[k] > w[k + 1] {
                w.swap(k, k + 1);
                recorded.push(k + 1); // abstract generator s_{k+1}
                sorted = false;
            }
        }
    }
    recorded.reverse();
    recorded
}

/// Young's **seminormal** representing matrix of `perm ∈ S_n` in the irreducible indexed by
/// the partition `λ` of `n`, over the rationals.
///
/// Requires `perm.size() == λ.sum()`. Returns the `d × d` identity-shaped matrix (`d` =
/// number of standard tableaux) for the identity permutation.
pub fn seminormal_representation(partition: &Partition, perm: &Permutation) -> Matrix<Rational> {
    let dim = standard_tableaux(partition).len();
    let word = adjacent_word(perm);
    let mut acc: Matrix<Rational> = Matrix::identity(dim);
    for g in word {
        let m = seminormal_transposition(partition, g);
        acc = acc.mul(&m).unwrap();
    }
    acc
}

/// The trace of a square rational matrix.
pub fn trace(m: &Matrix<Rational>) -> Rational {
    let n = m.rows();
    let mut s = Rational::zero();
    for i in 0..n {
        s = s + m.get(i, i).unwrap().clone();
    }
    s
}

/// The cycle type of a permutation, as a partition of `n` (including 1-cycles).
fn cycle_type(perm: &Permutation) -> Partition {
    let mut lengths: Vec<usize> = perm.cycles().iter().map(|c| c.len()).collect();
    let counted: usize = lengths.iter().sum();
    // `cycles()` may omit fixed points; pad with 1-cycles up to n.
    for _ in counted..perm.size() {
        lengths.push(1);
    }
    Partition::new(lengths)
}

/// The ways to remove a border strip (rim hook) of length `k` from `λ`, as `(μ, height)`
/// pairs where `height` is the leg length (rows spanned minus one). Implemented with the
/// abacus / beta-set method: `μ` is reached by decreasing a single first-column hook length
/// `β_i` by `k` when `β_i − k` is not already a bead; the height is the number of beads
/// strictly between `β_i − k` and `β_i`.
fn border_strip_removals(shape: &Partition, k: usize) -> Vec<(Partition, usize)> {
    let parts = shape.parts();
    let r = parts.len();
    if r == 0 || k == 0 {
        return Vec::new();
    }
    // Beta-numbers β_i = λ_i + (r-1-i), strictly decreasing and distinct.
    let beta: Vec<usize> = (0..r).map(|i| parts[i] + (r - 1 - i)).collect();
    let beadset: std::collections::HashSet<usize> = beta.iter().copied().collect();

    let mut results = Vec::new();
    for i in 0..r {
        if beta[i] < k {
            continue;
        }
        let nb = beta[i] - k;
        if beadset.contains(&nb) {
            continue;
        }
        // Height = number of beads strictly between nb and beta[i].
        let height = beta.iter().filter(|&&b| nb < b && b < beta[i]).count();
        let mut new_beta = beta.clone();
        new_beta[i] = nb;
        new_beta.sort_unstable_by(|a, b| b.cmp(a)); // descending
        let new_parts: Vec<usize> = (0..r)
            .map(|j| new_beta[j] - (r - 1 - j))
            .filter(|&p| p > 0)
            .collect();
        results.push((Partition::new(new_parts), height));
    }
    results
}

/// The Murnaghan–Nakayama recursion for the irreducible character `χ^λ` evaluated at a
/// permutation of the given `cycle_type`.
///
/// This is a *correct* implementation (via [`border_strip_removals`]); it is provided here
/// rather than reusing `symmetric_group_representations::murnaghan_nakayama`, whose
/// border-strip enumeration over-counts on column-shaped rim hooks (see integrator notes).
pub fn mn_character(shape: &Partition, cycle_type_parts: &Partition) -> i64 {
    if shape.sum() == 0 {
        return if cycle_type_parts.sum() == 0 { 1 } else { 0 };
    }
    let parts = cycle_type_parts.parts();
    if parts.is_empty() {
        return 0;
    }
    let k = parts[0];
    let rest = Partition::new(parts[1..].to_vec());
    let mut result = 0i64;
    for (mu, height) in border_strip_removals(shape, k) {
        let sign = if height % 2 == 0 { 1 } else { -1 };
        result += sign * mn_character(&mu, &rest);
    }
    result
}

/// `SymmetricCharacterValue(λ, perm)` — the value of the irreducible character indexed by
/// `λ` on `perm`, via the (correct) Murnaghan–Nakayama recursion. Equals the trace of
/// [`seminormal_representation`].
pub fn symmetric_character_value(partition: &Partition, perm: &Permutation) -> i64 {
    mn_character(partition, &cycle_type(perm))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Rational {
        Rational::new(n, d).unwrap()
    }

    fn perm(v: Vec<usize>) -> Permutation {
        Permutation::from_vec(v).unwrap()
    }

    #[test]
    fn test_hook_length_dimension() {
        // λ = [2,1] ⊢ 3 has dimension 2.
        assert_eq!(hook_length_dimension(&Partition::new(vec![2, 1])), Integer::from(2));
        // λ = [3,2] ⊢ 5 has dimension 5.
        assert_eq!(hook_length_dimension(&Partition::new(vec![3, 2])), Integer::from(5));
        // λ = [n] trivial, [1^n] sign: dimension 1.
        assert_eq!(hook_length_dimension(&Partition::new(vec![4])), Integer::from(1));
        assert_eq!(hook_length_dimension(&Partition::new(vec![1, 1, 1, 1])), Integer::from(1));
        // λ = [2,2] ⊢ 4 has dimension 2.
        assert_eq!(hook_length_dimension(&Partition::new(vec![2, 2])), Integer::from(2));
    }

    #[test]
    fn test_mn_character_ground_truth() {
        // S_3 character table. Classes: [1,1,1], [2,1], [3].
        let c111 = Partition::new(vec![1, 1, 1]);
        let c21 = Partition::new(vec![2, 1]);
        let c3 = Partition::new(vec![3]);
        // Trivial [3]: (1, 1, 1).
        let triv = Partition::new(vec![3]);
        assert_eq!(mn_character(&triv, &c111), 1);
        assert_eq!(mn_character(&triv, &c21), 1);
        assert_eq!(mn_character(&triv, &c3), 1);
        // Standard [2,1]: (2, 0, -1).
        let std = Partition::new(vec![2, 1]);
        assert_eq!(mn_character(&std, &c111), 2);
        assert_eq!(mn_character(&std, &c21), 0);
        assert_eq!(mn_character(&std, &c3), -1);
        // Sign [1,1,1]: (1, -1, 1).
        let sgn = Partition::new(vec![1, 1, 1]);
        assert_eq!(mn_character(&sgn, &c111), 1);
        assert_eq!(mn_character(&sgn, &c21), -1);
        assert_eq!(mn_character(&sgn, &c3), 1);

        // S_4 dimensions (value on the identity class [1,1,1,1]).
        let id4 = Partition::new(vec![1, 1, 1, 1]);
        for (lam, dim) in [
            (vec![4], 1),
            (vec![3, 1], 3),
            (vec![2, 2], 2),
            (vec![2, 1, 1], 3),
            (vec![1, 1, 1, 1], 1),
        ] {
            assert_eq!(mn_character(&Partition::new(lam.clone()), &id4), dim, "dim {:?}", lam);
        }
    }

    #[test]
    fn test_adjacent_word_reconstruction() {
        // Reconstruct the permutation from its adjacent-transposition word.
        for p in [
            perm(vec![0, 1, 2, 3]),
            perm(vec![2, 0, 1, 3]),
            perm(vec![3, 2, 1, 0]),
            perm(vec![1, 3, 0, 2]),
        ] {
            let n = p.size();
            let word = adjacent_word(&p);
            let mut acc = Permutation::identity(n);
            for g in word {
                // s_g swaps 0-indexed positions g-1 and g.
                let mut v: Vec<usize> = (0..n).collect();
                v.swap(g - 1, g);
                let s = Permutation::from_vec(v).unwrap();
                acc = acc.compose(&s).unwrap();
            }
            assert_eq!(acc, p);
        }
    }

    #[test]
    fn test_seminormal_s3_21() {
        let lambda = Partition::new(vec![2, 1]);
        // standard_tableaux order for [2,1]: determine indices via the generators.
        // s_1 = (1,2) swaps values 1,2; s_2 = (2,3).
        let m1 = seminormal_transposition(&lambda, 1);
        let m2 = seminormal_transposition(&lambda, 2);
        // Both are involutions.
        let id = Matrix::<Rational>::identity(2);
        assert_eq!(m1.mul(&m1).unwrap(), id);
        assert_eq!(m2.mul(&m2).unwrap(), id);
        // Braid relation s1 s2 s1 = s2 s1 s2.
        let lhs = m1.mul(&m2).unwrap().mul(&m1).unwrap();
        let rhs = m2.mul(&m1).unwrap().mul(&m2).unwrap();
        assert_eq!(lhs, rhs);
        // Traces are integer character values: χ(s_i) for [2,1] on a transposition = 0.
        assert_eq!(trace(&m1), r(0, 1));
        assert_eq!(trace(&m2), r(0, 1));
    }

    #[test]
    fn test_seminormal_trace_matches_character() {
        // For several partitions, trace(ρ_λ(σ)) must equal SymmetricCharacterValue(λ, σ).
        let cases = vec![
            (Partition::new(vec![2, 1]), 3),
            (Partition::new(vec![3, 1]), 4),
            (Partition::new(vec![2, 2]), 4),
            (Partition::new(vec![3, 2]), 5),
            (Partition::new(vec![4, 1]), 5),
        ];
        for (lambda, n) in cases {
            for p in crate::permutations::all_permutations(n) {
                let m = seminormal_representation(&lambda, &p);
                let tr = trace(&m);
                let chi = symmetric_character_value(&lambda, &p);
                assert_eq!(
                    tr,
                    Rational::from_i64(chi),
                    "trace mismatch for λ={:?}, σ={:?}",
                    lambda.parts(),
                    p.as_slice()
                );
            }
        }
    }

    #[test]
    fn test_trivial_and_sign_reps() {
        let n = 4;
        let trivial = Partition::new(vec![4]);
        let sign = Partition::new(vec![1, 1, 1, 1]);
        for p in crate::permutations::all_permutations(n) {
            let mt = seminormal_representation(&trivial, &p);
            assert_eq!(mt, Matrix::<Rational>::identity(1));
            let ms = seminormal_representation(&sign, &p);
            assert_eq!(ms.get(0, 0).unwrap().clone(), Rational::from_i64(p.sign() as i64));
        }
    }

    #[test]
    fn test_representation_is_homomorphism() {
        // ρ(σ·τ) = ρ(σ)·ρ(τ).
        let lambda = Partition::new(vec![3, 2]);
        let sigma = perm(vec![1, 0, 3, 4, 2]);
        let tau = perm(vec![4, 3, 2, 1, 0]);
        let prod = sigma.compose(&tau).unwrap();
        let lhs = seminormal_representation(&lambda, &prod);
        let rhs = seminormal_representation(&lambda, &sigma)
            .mul(&seminormal_representation(&lambda, &tau))
            .unwrap();
        assert_eq!(lhs, rhs);
    }
}

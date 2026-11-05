//! Frobenius **short cosets** — the #1 Stauduhar speed lever (Elsenhans 2014,
//! Klüners 2014, Geißler–Klüners 2000).
//!
//! In Stauduhar's method, descending `G → H` normally tests the relative
//! invariant on *all* `[G:H]` cosets of `G/H`. When a Frobenius element
//! `σ ∈ Gal(f)` is known (as an explicit permutation in the current root
//! labeling), only the **short cosets**
//!
//! ```text
//!     (G/H)_σ = { τH : σ ∈ τ H τ⁻¹ }   ( ⟺  τ⁻¹ σ τ ∈ H )
//! ```
//!
//! can possibly be the descent target: the true Galois group lies in some
//! conjugate `cHc⁻¹`, and `σ ∈ Gal(f) ⊆ cHc⁻¹`, so the true coset `c` is always
//! short. Every non-short coset is provably *not* the descent target, so the
//! invariant never needs to be evaluated there. In imprimitive degree-24 cases
//! this collapses an index that can be in the millions down to a handful (or
//! one) coset — the difference between a sub-second test and the >300 s blowup
//! of evaluating a full degree-24 relative invariant.
//!
//! Two Frobenius elements are available essentially for free:
//!   * **Complex conjugation** ([`conjugation_perm`]) — an element of `Gal(f)`
//!     in the *complex*-root labeling used by the small-degree descent. Its
//!     cycle type is `(#real roots) · 1 + (#conjugate pairs) · 2`.
//!   * **p-adic Frobenius** — in a p-adic `GaloisCtx` (a future module), the
//!     `p`-power Frobenius permutes the roots within each irreducible factor of
//!     `f mod p` as a cycle; this is the labeling Magma/OSCAR use.
//!
//! Soundness caveat for the *speedup*: evaluating only short cosets is sound
//! **provided the relative invariant is known separable for `(G,H)`** (so that a
//! rational value at a short coset is automatically a simple root). Establishing
//! that separability independently — by a precomputed invariant or a cheap
//! mod-`p` collision test (Tschirnhaus preselection) — is what lets the engine
//! skip building the full resolvent. This module supplies the group-theory core;
//! the separable-invariant layer consumes it.

use crate::perm::{compose, coset_reps, inverse, Perm};
use rustmath_polynomials::root_label::BigComplex;
use std::collections::HashSet;

/// The **short cosets** `(G/H)_σ = { τH : τ⁻¹ σ τ ∈ H }`, returned as the same
/// canonical coset representatives [`coset_reps`] produces (lexicographically
/// least per coset). `group` and `subgroup` are explicit element lists; `sigma`
/// is a Frobenius element of `Gal(f)` as a permutation in the current labeling.
///
/// The result is always a subset of `coset_reps(group, subgroup)`, and — when
/// `σ` is genuinely in `Gal(f)` — always contains the true descent coset. If it
/// is **empty**, `Gal(f)` lies in no conjugate of `H`, so the `G → H` descent
/// can be rejected outright with no invariant evaluation at all.
pub fn short_cosets(group: &[Perm], subgroup: &[Perm], sigma: &Perm) -> Vec<Perm> {
    let h: HashSet<&Perm> = subgroup.iter().collect();
    coset_reps(group, subgroup)
        .into_iter()
        .filter(|tau| {
            // σ ∈ τHτ⁻¹  ⟺  τ⁻¹ σ τ ∈ H. With compose(a,b) = apply b then a,
            // the group product τ⁻¹·σ·τ is compose(inverse(τ), compose(σ, τ)).
            let conj = compose(&inverse(tau), &compose(sigma, tau));
            h.contains(&conj)
        })
        .collect()
}

/// Sound, evaluation-free rejection: `true` iff `Gal(f)` *cannot* lie in any
/// conjugate of `H` (the short-coset set is empty), given the Frobenius element
/// `σ`. A `true` verdict lets the descent skip `H` entirely.
pub fn descent_impossible(group: &[Perm], subgroup: &[Perm], sigma: &Perm) -> bool {
    short_cosets(group, subgroup, sigma).is_empty()
}

/// The number of short cosets — the count of relative-invariant evaluations the
/// `G → H` descent will need (vs. the full index `[G:H]`).
pub fn short_coset_count(group: &[Perm], subgroup: &[Perm], sigma: &Perm) -> usize {
    short_cosets(group, subgroup, sigma).len()
}

/// **Complex conjugation** as a permutation of the (complex-)labeled roots — a
/// free Frobenius element of `Gal(f)` for the small-degree complex descent.
///
/// Root `i` maps to the root `j` closest to its complex conjugate `(re_i, −im_i)`.
/// Real roots (`im ≈ 0`) are fixed. Matching is done in `f64` (robust because the
/// roots are well separated; the exact value is irrelevant — only the pairing is).
/// The result is an involution: its cycle type is `[1]·(#real) + [2]·(#pairs)`.
pub fn conjugation_perm(roots: &[BigComplex]) -> Perm {
    let n = roots.len();
    let zf: Vec<(f64, f64)> = roots.iter().map(|z| (z.re.to_f64(), z.im.to_f64())).collect();
    // Scale tolerance to the root magnitudes so it is meaningful for any field.
    let mut sep = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (zf[i].0 - zf[j].0).hypot(zf[i].1 - zf[j].1);
            if d < sep {
                sep = d;
            }
        }
    }
    let real_tol = if sep.is_finite() { sep * 0.25 } else { 1e-9 };
    let mut p = vec![0usize; n];
    for i in 0..n {
        if zf[i].1.abs() <= real_tol {
            p[i] = i; // real root: fixed
            continue;
        }
        // find the closest root to the conjugate (re_i, -im_i)
        let target = (zf[i].0, -zf[i].1);
        let mut best = i;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            let d = (zf[j].0 - target.0).hypot(zf[j].1 - target.1);
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        p[i] = best;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perm::{cycle_type, from_cycles, group_closure, identity, sym_gens};
    use rustmath_integers::Integer;
    use rustmath_polynomials::root_label::complex_roots;

    fn s_n(n: usize) -> Vec<Perm> {
        group_closure(n, &sym_gens(n), 10000).unwrap()
    }

    #[test]
    fn short_cosets_subset_of_all() {
        let g = s_n(3);
        let h = group_closure(3, &[from_cycles(3, &[vec![0, 1]])], 100).unwrap(); // ⟨(01)⟩
        let sigma = from_cycles(3, &[vec![0, 1]]);
        let short = short_cosets(&g, &h, &sigma);
        let all = coset_reps(&g, &h);
        assert!(short.len() <= all.len());
        for s in &short {
            assert!(all.contains(s));
        }
        // (01) is in H itself and in some conjugates; at least the identity coset
        // is short (id⁻¹·σ·id = σ = (01) ∈ H).
        assert!(!short.is_empty());
    }

    #[test]
    fn identity_sigma_makes_all_cosets_short() {
        let g = s_n(4);
        let h = group_closure(4, &[from_cycles(4, &[vec![0, 1]]), from_cycles(4, &[vec![2, 3]])], 100).unwrap();
        let id = identity(4);
        let short = short_cosets(&g, &h, &id);
        let all = coset_reps(&g, &h);
        // identity lies in every conjugate τHτ⁻¹, so every coset is short.
        assert_eq!(short.len(), all.len());
    }

    #[test]
    fn impossible_descent_detected() {
        // H = ⟨(012)⟩ ≅ C_3 in S_3 has no transposition in any conjugate, so a
        // transposition Frobenius makes the G→H descent impossible.
        let g = s_n(3);
        let h = group_closure(3, &[from_cycles(3, &[vec![0, 1, 2]])], 100).unwrap();
        let sigma = from_cycles(3, &[vec![0, 1]]);
        assert!(descent_impossible(&g, &h, &sigma));
        assert_eq!(short_coset_count(&g, &h, &sigma), 0);
    }

    #[test]
    fn conjugation_perm_cycle_types() {
        // x^2 - 2: two real roots → identity (cycle type all 1s)
        let f = [Integer::from(-2), Integer::from(0), Integer::from(1)];
        let r = complex_roots(&f, 200).roots;
        let cj = conjugation_perm(&r);
        assert_eq!(cycle_type(&cj), vec![1, 1]);

        // x^2 + 1: one conjugate pair → a single 2-cycle
        let f = [Integer::from(1), Integer::from(0), Integer::from(1)];
        let r = complex_roots(&f, 200).roots;
        let cj = conjugation_perm(&r);
        assert_eq!(cycle_type(&cj), vec![2]);

        // x^3 - 2: one real + one conjugate pair → cycle type [1, 2]
        let f = [Integer::from(-2), Integer::from(0), Integer::from(0), Integer::from(1)];
        let r = complex_roots(&f, 200).roots;
        let cj = conjugation_perm(&r);
        assert_eq!(cycle_type(&cj), vec![1, 2]);

        // x^4 + 1: two conjugate pairs → cycle type [2, 2]; and it's an involution
        let f = [Integer::from(1), Integer::from(0), Integer::from(0), Integer::from(0), Integer::from(1)];
        let r = complex_roots(&f, 200).roots;
        let cj = conjugation_perm(&r);
        assert_eq!(cycle_type(&cj), vec![2, 2]);
        assert_eq!(compose(&cj, &cj), identity(4)); // involution
    }
}

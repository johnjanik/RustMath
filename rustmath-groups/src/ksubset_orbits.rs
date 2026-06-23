//! Orbit lengths of a permutation group on `k`-subsets — the combinatorial
//! invariant measured by the linear (subset-sum) resolvent of Module 18.
//!
//! For `f ∈ ℤ[x]` with roots `α₁,…,αₙ` and Galois group `G = Gal(f) ⊆ Sₙ` acting on
//! the roots, the sorted multiset of orbit lengths of `G` on `k`-subsets equals the
//! degrees of the irreducible factors of the `k`-subset resolvent
//! `∏_{|S|=k}(Y − Σ_{i∈S} αᵢ)` over `ℚ` — **provided** the subset-sums are distinct
//! (the resolvent is separable). This is the bridge from
//! [`rustmath_polynomials::resolvent`] to group identification: two transitive
//! groups sharing a Frobenius cycle-type profile (a *blind class*) are routinely
//! separated by their pair-orbit signatures.
//!
//! Cross-validated against PARI/GP: `Φ₅` (Galois `C₄ = ⟨(1 2 4 3)⟩`) has pair-sum
//! resolvent factor degrees `[2, 4]`, exactly the orbit lengths of `C₄` on pairs;
//! `x⁴+x+1` (Galois `S₄`) gives a single degree-6 factor = the lone `S₄`-orbit on
//! pairs.
//!
//! Subsets of `{0,…,n−1}` are encoded as `u32` bitmasks (`n ≤ 24 = DEG`).

use crate::transitive24::{Perm, DEG};
use std::collections::HashSet;

/// Image of a subset (bitmask) under `g`: bit `i` ↦ bit `g[i]`.
fn apply(g: &Perm, mask: u32) -> u32 {
    let mut out = 0u32;
    let mut m = mask;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        out |= 1u32 << g[i];
        m &= m - 1;
    }
    out
}

/// All `k`-subset bitmasks of `{0,…,n−1}` (`0 ≤ k ≤ n ≤ DEG`), enumerated directly
/// in `O(C(n,k))` via Gosper's hack (next bitmask with the same popcount) — not by
/// scanning all `2ⁿ` masks, so degree-24 stays cheap.
fn ksubset_masks(n: usize, k: usize) -> Vec<u32> {
    assert!(n <= DEG && k <= n, "need k ≤ n ≤ {DEG}");
    let mut out = Vec::new();
    if k == 0 {
        out.push(0);
        return out;
    }
    let limit = if n == 32 { u32::MAX } else { 1u32 << n };
    let mut x: u32 = (1u32 << k) - 1; // lowest k-subset: bits 0..k-1
    while x < limit {
        out.push(x);
        // Gosper's hack: next integer with the same number of set bits.
        let c = x & x.wrapping_neg();
        let r = x + c;
        x = (((x ^ r) >> 2) / c) | r;
    }
    out
}

/// Sorted (ascending) orbit lengths of the group `⟨gens⟩` acting on the `k`-subsets
/// of `{0,…,n−1}`. The lengths sum to `C(n, k)`. Orbits are computed by closing
/// each subset under the generators (no need to enumerate the whole group).
pub fn orbit_lengths_on_ksubsets(gens: &[Perm], n: usize, k: usize) -> Vec<usize> {
    let masks = ksubset_masks(n, k);
    let mut visited: HashSet<u32> = HashSet::new();
    let mut lengths = Vec::new();
    for &start in &masks {
        if visited.contains(&start) {
            continue;
        }
        // BFS/DFS over the orbit of `start`.
        let mut stack = vec![start];
        visited.insert(start);
        let mut count = 0usize;
        while let Some(cur) = stack.pop() {
            count += 1;
            for g in gens {
                let nxt = apply(g, cur);
                if visited.insert(nxt) {
                    stack.push(nxt);
                }
            }
        }
        lengths.push(count);
    }
    lengths.sort_unstable();
    lengths
}

/// Orbit lengths of `⟨gens⟩` on **2-subsets** (pairs) of `{0,…,n−1}` — the
/// invariant matched by `rustmath_polynomials::resolvent::pair_sum_resolvent`.
pub fn orbit_lengths_on_pairs(gens: &[Perm], n: usize) -> Vec<usize> {
    orbit_lengths_on_ksubsets(gens, n, 2)
}

// Tests live in `tests/ksubset_orbits.rs`: the crate's other inline `#[cfg(test)]`
// modules do not compile, so unit tests for this module run as an integration test
// (same pattern as `tests/galois_narrowing.rs`).

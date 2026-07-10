//! Stauduhar descent — the blocked completion path (documented seam, honest
//! stub).
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups): MAGMA's `GaloisGroup` is the
//!   Stauduhar method with p-adic root approximation (Fieker–Klüners).
//! - SageMath: `sage.rings.number_field.galois_group` (delegates hard cases to
//!   PARI/GAP).
//! - Stauduhar, R.: "The determination of Galois groups", Math. Comp. 27
//!   (1973); Fieker–Klüners, "Computation of Galois groups of rational
//!   polynomials", LMS J. Comput. Math. 17 (2014).
//!
//! # The algorithm this crate is missing
//!
//! Starting from `G₀ = S_n`, repeatedly test, for each maximal transitive
//! subgroup `H < G` (up to conjugacy), whether `Gal(f) ≤ H^σ` for some coset
//! `σ`, by evaluating an `H`-invariant polynomial `I(x₁,…,x_n)` at labelled
//! approximations of the roots for every coset representative and checking
//! whether the value is a rational integer (with a proven error bound); descend
//! while any test succeeds. Termination identifies `Gal(f)` exactly — this is
//! what upgrades every `Unresolved` of this crate (e.g. the F20-vs-S5 pair at
//! degree 5, everything at degree ≥ 6 that is not `A_n`/`S_n`) to `Decided`.
//!
//! # The exact seam, piece by piece
//!
//! 1. **Transitive-subgroup lattice with cosets.** For each degree `n`: the
//!    maximal-subgroup chains of transitive groups plus coset representatives.
//!    `rustmath-groups` (READ-ONLY / frozen for this sprint) already carries
//!    committed transitive-group tables at selected degrees
//!    (`transitive23`/`transitive24` modules) but not the
//!    maximal-subgroup/coset machinery; a BSGS (base and strong generating
//!    set) implementation there is the natural provider. Required API shape:
//!    `maximal_transitive_subgroups(n, t) -> Vec<(t', Vec<CosetRep>)>`.
//! 2. **Relative resolvent / invariant evaluation.** For each pair `H < G` a
//!    primitive `H`-invariant `I ∈ ℤ[x₁,…,x_n]` with its `G`-orbit; generic
//!    constructions exist (sums of monomial orbits). Evaluation needs root
//!    approximations to certified precision.
//! 3. **Certified root labelling.** `rustmath_complex::aberth_roots` (already
//!    landed and verified) provides certified arbitrary-precision complex
//!    roots; the seam is a wrapper fixing a root ordering, tracking it through
//!    Tschirnhausen transformations, and bounding `|I(α)|` to prove/refute
//!    integrality of the resolvent value (or the p-adic analogue via
//!    `rustmath-padics`).
//! 4. **Tschirnhausen fallback** for degenerate invariant values (repeated
//!    resolvent roots), as in [`crate::quintic`]'s `s`-fallback but general.
//!
//! Until those land, this function refuses honestly rather than guess.

use crate::types::GaloisGroupResult;
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;

/// Stauduhar descent through the transitive lattice of degree `n`. **Not yet
/// implemented** — see the module documentation for the exact missing pieces
/// and where each plugs in. Always returns `Err(MathError::NotImplemented)`.
pub fn stauduhar_descent(f: &[Integer]) -> Result<GaloisGroupResult> {
    let n = f.len().saturating_sub(1);
    Err(MathError::NotImplemented(format!(
        "Stauduhar descent (degree {n}): blocked on (1) maximal-transitive-subgroup lattice + \
         coset representatives + BSGS in the frozen rustmath-groups crate, (2) generic \
         H-invariant construction, (3) certified root labelling on top of \
         rustmath_complex::aberth_roots; see rustmath-galois/src/stauduhar.rs for the seam"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stauduhar_refuses_honestly() {
        let f: Vec<Integer> = [-2i64, 0, 0, 0, 0, 1].iter().map(|&x| Integer::from(x)).collect();
        assert!(matches!(
            stauduhar_descent(&f),
            Err(MathError::NotImplemented(_))
        ));
    }
}

//! Galois groups of **reducible** polynomials: the splitting field of a
//! product is the compositum of the factors' splitting fields, and the group
//! embeds in the direct product as a subdirect product (Goursat). This module
//! decides the cases where the subdirect-product analysis is a finite exact
//! computation on discriminant square classes:
//!
//! - all nonlinear factors quadratic → `Gal = C2^m`, `m` = the `𝔽₂`-rank of the
//!   factor discriminants in `ℚ*/ℚ*²`;
//! - one cubic + one quadratic (+ linear factors) → `C6`, `S3`, or `S3×C2`,
//!   decided by whether the cubic is cyclic and whether the quadratic's
//!   discriminant class coincides with the cubic's (`ℚ(√disc₃) = ℚ(√d₂)`);
//! - anything else with ≥ 2 nonlinear factors → honest `Unresolved`.
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups): `GaloisGroup` of reducible
//!   polynomials (direct-product scaffolding).
//! - SageMath: `sage.rings.number_field.galois_group`.

use crate::types::{Candidates, Evidence, GaloisGroupResult, GroupId};
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;

/// `𝔽₂`-rank of the given discriminants in `ℚ*/ℚ*²`: greedily keep a
/// discriminant iff no product with a subset of the kept ones is a perfect
/// square. Exact; cost `O(2^rank)` big-integer products.
fn square_class_rank(discs: &[Integer]) -> Result<usize> {
    let mut indep: Vec<Integer> = Vec::new();
    for d in discs {
        if indep.len() >= 16 {
            return Err(MathError::NotSupported(
                "more than 16 independent quadratic discriminants".to_string(),
            ));
        }
        let mut dependent = false;
        for mask in 0u32..(1u32 << indep.len()) {
            let mut prod = d.clone();
            for (i, e) in indep.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    prod = prod * e.clone();
                }
            }
            if prod.is_perfect_square() {
                dependent = true;
                break;
            }
        }
        if !dependent {
            indep.push(d.clone());
        }
    }
    Ok(indep.len())
}

/// Classify a squarefree polynomial with ≥ 2 distinct nonlinear irreducible
/// factors (each **monic**, monicized by the caller; linear factors
/// contribute nothing to the splitting field and are already dropped).
/// `total_degree` is the degree of the trimmed input.
pub fn classify_composite(
    nonlinear: &[Vec<Integer>],
    total_degree: usize,
    mut ev: Evidence,
) -> Result<GaloisGroupResult> {
    debug_assert!(nonlinear.len() >= 2);
    let degs: Vec<usize> = nonlinear.iter().map(|g| g.len() - 1).collect();
    ev.notes.push(format!(
        "reducible input: {} nonlinear irreducible factors of degrees {degs:?}; the group is \
         the Galois group of the compositum of their splitting fields",
        nonlinear.len()
    ));

    let discs: Vec<Integer> = nonlinear.iter().map(|g| discriminant(g)).collect();

    // Parity of the whole action: disc(∏ fᵢ) = ∏ disc(fᵢ) · ∏_{i<j} Res(fᵢ,fⱼ)²,
    // so the square class of the product of factor discriminants decides G ⊆ A_n.
    let disc_prod = discs
        .iter()
        .fold(Integer::one(), |acc, d| acc * d.clone());
    let in_alternating = disc_prod.is_perfect_square();

    // Case 1: all factors quadratic → multiquadratic field, G = C2^m.
    if degs.iter().all(|&d| d == 2) {
        let m = square_class_rank(&discs)?;
        // m ≥ 1: an irreducible quadratic has a non-square discriminant.
        let name = if m == 1 { "C2".to_string() } else { format!("C2^{m}") };
        ev.notes.push(format!(
            "all nonlinear factors quadratic with discriminants {:?}; their square classes \
             span an 𝔽₂-space of rank {m}, so the compositum is multiquadratic of degree 2^{m} \
             and Gal = C2^{m} (elementary abelian)",
            discs.iter().map(|d| d.to_string()).collect::<Vec<_>>()
        ));
        return Ok(GaloisGroupResult::Decided {
            group: GroupId {
                degree: total_degree,
                order: Some(1u128 << m),
                name,
                t_number: None,
            },
            evidence: ev,
        });
    }

    // Case 2: exactly one cubic and one quadratic.
    if nonlinear.len() == 2 && degs.contains(&3) && degs.contains(&2) {
        let (i3, i2) = if degs[0] == 3 { (0, 1) } else { (1, 0) };
        let d3 = &discs[i3];
        let d2 = &discs[i2];
        let cubic_cyclic = !d3.is_zero() && d3.is_perfect_square();
        let (name, order, note) = if cubic_cyclic {
            (
                "C6".to_string(),
                6u128,
                "cubic factor is cyclic (square discriminant); C3 and C2 share no common \
                 quotient (Goursat), so the subdirect product is the full C3 × C2 = C6"
                    .to_string(),
            )
        } else if (d3.clone() * d2.clone()).is_perfect_square() {
            (
                "S3".to_string(),
                6u128,
                "cubic factor has group S3 and ℚ(√disc₃) = ℚ(√d₂) (disc₃·d₂ is a square): \
                 the quadratic's splitting field is the quadratic subfield of the cubic's, \
                 so the compositum group is the order-6 fiber product ≅ S3"
                    .to_string(),
            )
        } else {
            (
                "S3xC2".to_string(),
                12u128,
                "cubic factor has group S3 and ℚ(√disc₃) ≠ ℚ(√d₂) (disc₃·d₂ not a square): \
                 the only common quotient C2 is not identified, so Gal = S3 × C2 (order 12)"
                    .to_string(),
            )
        };
        ev.notes.push(note);
        return Ok(GaloisGroupResult::Decided {
            group: GroupId { degree: total_degree, order: Some(order), name, t_number: None },
            evidence: ev,
        });
    }

    // Everything else: honest refusal.
    Ok(GaloisGroupResult::Unresolved {
        candidates: Candidates::Unknown {
            degree: total_degree,
            transitive: false,
            contained_in_alternating: Some(in_alternating),
        },
        ruled_out: Vec::new(),
        evidence: ev,
        blocked_on: "general subdirect-product (Goursat) analysis of a compositum with ≥ 2 \
                     nonlinear factors is not implemented; it needs per-factor groups plus \
                     the lattice of common quotients and field intersections"
            .to_string(),
    })
}

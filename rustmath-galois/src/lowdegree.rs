//! Exact Galois-group classification in degrees 2, 3, 4 — the classical
//! resolvent decision procedures. Every branch is a complete exact decision;
//! nothing here is probabilistic and nothing returns `Unresolved`.
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups): the special-cased
//!   `GaloisGroup` decision for `deg f ≤ 4`.
//! - SageMath: `sage.rings.number_field.galois_group`.
//! - Kappe, L.-C. and Warren, B.: "An Elementary Test for the Galois Group of a
//!   Quartic Polynomial", Amer. Math. Monthly 96 (1989) 133–137 — the C4/D4
//!   disambiguation. (Criterion re-verified empirically for this port against
//!   sympy's `galois_group` on 700+ irreducible quartics covering all five
//!   groups.)
//!
//! Input contract for every function here: monic irreducible `f ∈ ℤ[x]` of the
//! stated degree, with `evidence.discriminant` / `disc_is_square` prefilled by
//! the caller for that exact polynomial.

use crate::types::{Evidence, GaloisGroupResult, GroupId};
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::zassenhaus;

fn decided(group: GroupId, mut ev: Evidence, note: String) -> GaloisGroupResult {
    ev.notes.push(note);
    GaloisGroupResult::Decided { group, evidence: ev }
}

/// Degree 2, irreducible ⟹ `C2` (the only transitive subgroup of `S₂`).
pub fn classify_quadratic(ev: Evidence) -> GaloisGroupResult {
    decided(
        GroupId::new(2, 2, "C2", Some(1)),
        ev,
        "irreducible quadratic: Gal = C2 = S2 (2T1)".to_string(),
    )
}

/// Degree 3, irreducible: `C3` iff `disc(f)` is a perfect square, else `S3`.
pub fn classify_cubic(ev: Evidence) -> Result<GaloisGroupResult> {
    let disc_sq = ev
        .disc_is_square
        .ok_or_else(|| MathError::InvalidOperation("cubic classifier needs disc".into()))?;
    Ok(if disc_sq {
        decided(
            GroupId::new(3, 3, "C3", Some(1)),
            ev,
            "irreducible cubic with square discriminant: Gal ⊆ A3, transitive ⟹ C3 (3T1)"
                .to_string(),
        )
    } else {
        decided(
            GroupId::new(3, 6, "S3", Some(2)),
            ev,
            "irreducible cubic with non-square discriminant: Gal ⊄ A3, transitive ⟹ S3 (3T2)"
                .to_string(),
        )
    })
}

/// Integer roots of a **monic** polynomial in `ℤ[x]` (rational roots of a monic
/// integer polynomial are integers), via full factorization. Multiplicity-free
/// under the separability precondition.
fn monic_integer_roots(f: &[Integer]) -> Result<Vec<Integer>> {
    let (_, factors) = zassenhaus::factor(f).map_err(|_| {
        MathError::NotSupported("factor recombination limit exceeded".to_string())
    })?;
    let mut roots = Vec::new();
    for (g, mult) in factors {
        if g.len() == 2 {
            if !g[1].is_one() {
                // Cannot happen for monic f (factors are primitive with positive
                // leading coefficient and their product is monic).
                return Err(MathError::InvalidOperation(
                    "non-monic linear factor of a monic polynomial".to_string(),
                ));
            }
            for _ in 0..mult {
                roots.push(-g[0].clone());
            }
        }
    }
    Ok(roots)
}

/// Degree 4, irreducible: full classification into `C4/V4/D4/A4/S4` via the
/// resolvent cubic `R(y) = y³ − b y² + (ac − 4d) y − (a²d + c² − 4bd)` (roots
/// `α₁α₂+α₃α₄`, `α₁α₃+α₂α₄`, `α₁α₄+α₂α₃`; `disc R = disc f`, so `R` is
/// separable exactly when `f` is):
///
/// - `R` irreducible: `A4` if `disc f` is a square, else `S4`;
/// - `R` splits into three rational roots: `V4`;
/// - `R` has exactly one rational root `β`: `C4` or `D4`, separated by
///   Kappe–Warren: `C4` iff both `x² − βx + d` and `x² + ax + (b − β)` split
///   over `ℚ(√disc f)` — for rational `δ = u² − 4v` that is `δ = 0`, `δ` a
///   square, or `δ·disc` a square.
pub fn classify_quartic(f: &[Integer], ev: Evidence) -> Result<GaloisGroupResult> {
    debug_assert_eq!(f.len(), 5);
    debug_assert!(f[4].is_one());
    let (a, b, c, d) = (&f[3], &f[2], &f[1], &f[0]);
    let disc = ev
        .discriminant
        .clone()
        .ok_or_else(|| MathError::InvalidOperation("quartic classifier needs disc".into()))?;
    let disc_sq = ev.disc_is_square == Some(true);

    // Resolvent cubic, little-endian: y³ − b y² + (ac − 4d) y − (a²d + c² − 4bd).
    let four = Integer::from(4);
    let rc: Vec<Integer> = vec![
        -(a.clone() * a.clone() * d.clone() + c.clone() * c.clone()
            - four.clone() * b.clone() * d.clone()),
        a.clone() * c.clone() - four.clone() * d.clone(),
        -b.clone(),
        Integer::one(),
    ];
    let roots = monic_integer_roots(&rc)?;

    match roots.len() {
        0 => {
            if disc_sq {
                Ok(decided(
                    GroupId::new(4, 12, "A4", Some(4)),
                    ev,
                    "resolvent cubic irreducible over ℚ and disc(f) a square ⟹ A4 (4T4)"
                        .to_string(),
                ))
            } else {
                Ok(decided(
                    GroupId::new(4, 24, "S4", Some(5)),
                    ev,
                    "resolvent cubic irreducible over ℚ and disc(f) not a square ⟹ S4 (4T5)"
                        .to_string(),
                ))
            }
        }
        3 => {
            if !disc_sq {
                return Err(MathError::InvalidOperation(
                    "internal contradiction: resolvent cubic split but disc(f) not a square \
                     (V4 ⊆ A4 forces a square discriminant)"
                        .to_string(),
                ));
            }
            Ok(decided(
                GroupId::new(4, 4, "V4", Some(2)),
                ev,
                "resolvent cubic splits into three rational roots ⟹ V4 (4T2)".to_string(),
            ))
        }
        1 => {
            if disc_sq {
                return Err(MathError::InvalidOperation(
                    "internal contradiction: resolvent cubic with exactly one rational root \
                     but square disc(f) (C4/D4 have non-square discriminant)"
                        .to_string(),
                ));
            }
            let beta = &roots[0];
            // Kappe–Warren quadratics: x² − βx + d and x² + ax + (b − β).
            let delta1 = beta.clone() * beta.clone() - four.clone() * d.clone();
            let delta2 =
                a.clone() * a.clone() - four.clone() * (b.clone() - beta.clone());
            let splits_over_q_sqrt_disc = |delta: &Integer| -> bool {
                delta.is_zero()
                    || delta.is_perfect_square()
                    || (delta.clone() * disc.clone()).is_perfect_square()
            };
            if splits_over_q_sqrt_disc(&delta1) && splits_over_q_sqrt_disc(&delta2) {
                Ok(decided(
                    GroupId::new(4, 4, "C4", Some(1)),
                    ev,
                    format!(
                        "resolvent cubic has unique rational root β = {beta}; Kappe–Warren: \
                         both x²−βx+d and x²+ax+(b−β) split over ℚ(√disc) ⟹ C4 (4T1)"
                    ),
                ))
            } else {
                Ok(decided(
                    GroupId::new(4, 8, "D4", Some(3)),
                    ev,
                    format!(
                        "resolvent cubic has unique rational root β = {beta}; Kappe–Warren: \
                         not both auxiliary quadratics split over ℚ(√disc) ⟹ D4 (4T3)"
                    ),
                ))
            }
        }
        k => Err(MathError::InvalidOperation(format!(
            "separable rational cubic with {k} rational roots is impossible"
        ))),
    }
}

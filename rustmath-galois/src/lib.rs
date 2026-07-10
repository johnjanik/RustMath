//! # rustmath-galois — Galois groups of univariate polynomials over ℚ
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups): `GaloisGroup`,
//!   `GaloisProof`, transitive-group identification `nTk`.
//! - SageMath: `sage.rings.number_field.galois_group`.
//!
//! Executable subset delivered here (PLAN Phase-3 keystone 2):
//!
//! - **Degrees 2–4**: complete exact decision by classical resolvents
//!   (discriminant square test; resolvent cubic factorization pattern with the
//!   Kappe–Warren C4/D4 disambiguation). See [`lowdegree`].
//! - **Degree 5**: Frobenius cycle-type sieve over the complete 5T1–5T5 table,
//!   sharpened by an exact ordered-pair resolvent; decides C5/D5/F20-ruled
//!   cases/A5/S5 whenever the exhibited elements and exact invariants force
//!   uniqueness, and returns the F20-vs-S5 blind spot as `Unresolved`. See
//!   [`quintic`].
//! - **General degree n**: the sieve infrastructure (cycle types via Dedekind
//!   reduction, parity/discriminant filtering) plus Jordan-criterion
//!   certificates that decide `A_n`/`S_n` by exhibiting elements; everything
//!   else is an honest `Unresolved`-with-evidence. See [`sieve`].
//! - **Reducible inputs**: multiquadratic (`C2^m`) and cubic×quadratic
//!   composita decided exactly; the general Goursat analysis is documented as
//!   out of scope. See [`reducible`].
//! - **Stauduhar descent**: the completion path is specified — with the exact
//!   seam into the frozen `rustmath-groups` (BSGS, transitive lattices) and
//!   `rustmath_complex::aberth_roots` (certified root labelling) — in
//!   [`stauduhar`], which refuses honestly until those land.
//!
//! ## Honesty contract
//!
//! [`GaloisGroupResult`] separates `Decided` (with a complete certificate
//! trail) from `Unresolved` (with everything that was *proven* ruled out and
//! the exact remaining candidate set). A bounded search that finds nothing —
//! e.g. a sieve that never sees an S5-only cycle type — **never** decides; only
//! exhibited elements and exact invariants do.
//!
//! ```
//! use rustmath_galois::galois_group;
//! use rustmath_integers::Integer;
//!
//! // x⁴ + 1 has Galois group V4.
//! let f: Vec<Integer> = [1i64, 0, 0, 0, 1].iter().map(|&c| Integer::from(c)).collect();
//! assert_eq!(galois_group(&f).unwrap().decided_name(), Some("V4"));
//! ```

#![forbid(unsafe_code)]

pub mod lowdegree;
pub mod quintic;
pub mod reducible;
pub mod sieve;
pub mod stauduhar;
pub mod types;

pub use types::{Candidates, CycleType, Evidence, GaloisGroupResult, GroupId};

use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::{zassenhaus, zx};

/// Monicize an irreducible `g ∈ ℤ[x]` with positive leading coefficient `a`:
/// `h(y) = a^{n−1} g(y/a)`, i.e. `h_k = g_k · a^{n−1−k}`. `h` is monic and
/// integral, its roots are `a·(roots of g)`, so the splitting field and the
/// Galois action on roots are unchanged.
fn monicize(g: &[Integer]) -> Vec<Integer> {
    let n = g.len() - 1;
    let a = &g[n];
    if a.is_one() {
        return g.to_vec();
    }
    (0..=n)
        .map(|k| {
            if k == n {
                Integer::one()
            } else {
                g[k].clone() * a.pow((n - 1 - k) as u32)
            }
        })
        .collect()
}

/// Classification of a **monic irreducible** `g` of degree ≥ 1, with the
/// evidence carrying the discriminant facts for `g` itself.
fn classify_irreducible(g: &[Integer], mut ev: Evidence) -> Result<GaloisGroupResult> {
    let n = g.len() - 1;
    if n == 1 {
        ev.notes.push("degree 1: splitting field is ℚ".to_string());
        return Ok(GaloisGroupResult::Decided {
            group: GroupId::new(1, 1, "C1", Some(1)),
            evidence: ev,
        });
    }
    let d = discriminant(g);
    if d.is_zero() {
        // Impossible for an irreducible polynomial in characteristic 0.
        return Err(MathError::InvalidOperation(
            "irreducible polynomial with zero discriminant".to_string(),
        ));
    }
    let disc_sq = d.is_perfect_square();
    ev.discriminant = Some(d);
    ev.disc_is_square = Some(disc_sq);
    ev.notes.push(format!(
        "disc is {} perfect square ⟹ G {} A_{n}",
        if disc_sq { "a" } else { "not a" },
        if disc_sq { "⊆" } else { "⊄" }
    ));
    match n {
        2 => Ok(lowdegree::classify_quadratic(ev)),
        3 => lowdegree::classify_cubic(ev),
        4 => lowdegree::classify_quartic(g, ev),
        5 => quintic::classify_quintic(g, ev),
        _ => sieve::classify_general(g, ev),
    }
}

/// Galois group over ℚ of the splitting field of `f ∈ ℤ[x]` (little-endian
/// coefficients, `f[i]` = coefficient of `xⁱ`; need not be monic or
/// squarefree — content, multiplicities and rational roots do not change the
/// splitting field and are normalized away with a note in the evidence).
///
/// Returns `Err` only for empty/zero/constant input or on internal failure;
/// mathematical indeterminacy is reported as
/// [`GaloisGroupResult::Unresolved`], never as an error and never as a guess.
pub fn galois_group(f: &[Integer]) -> Result<GaloisGroupResult> {
    let ft = zx::trim(f);
    if ft.is_empty() || zx::is_zero(&ft) {
        return Err(MathError::InvalidArgument(
            "the zero polynomial has no splitting field".to_string(),
        ));
    }
    let n = zx::degree(&ft);
    if n == 0 {
        return Err(MathError::InvalidArgument(
            "a nonzero constant has no roots; its Galois group is not defined here".to_string(),
        ));
    }
    let n = n as usize;

    let (_content, factors) = zassenhaus::factor(&ft).map_err(|_| {
        MathError::NotSupported("factor recombination limit exceeded".to_string())
    })?;
    let mut ev = Evidence { degree: n, ..Default::default() };
    if factors.iter().any(|(_, m)| *m > 1) {
        ev.notes.push(
            "input is not squarefree; the group is computed for its radical (same splitting \
             field)"
                .to_string(),
        );
    }
    // Distinct irreducible factors (primitive, positive leading coefficient).
    let distinct: Vec<Vec<Integer>> = factors.into_iter().map(|(g, _)| g).collect();
    let nonlinear: Vec<&Vec<Integer>> = distinct.iter().filter(|g| g.len() > 2).collect();
    let n_linear = distinct.len() - nonlinear.len();

    ev.irreducible = distinct.len() == 1 && distinct[0].len() == ft.len();

    match nonlinear.len() {
        0 => {
            ev.notes.push(
                "all irreducible factors are linear: the splitting field is ℚ".to_string(),
            );
            Ok(GaloisGroupResult::Decided {
                group: GroupId { degree: n, order: Some(1), name: "C1".to_string(), t_number: None },
                evidence: ev,
            })
        }
        1 => {
            if n_linear > 0 {
                ev.notes.push(format!(
                    "linear factors contribute nothing to the splitting field; the group is \
                     that of the unique nonlinear irreducible factor (degree {})",
                    nonlinear[0].len() - 1
                ));
            }
            let g = monicize(nonlinear[0]);
            if !g[..].eq(&nonlinear[0][..]) {
                ev.notes.push(
                    "factor monicized via y ↦ a^{n−1}·g(y/a) (roots scaled by the leading \
                     coefficient; splitting field unchanged)"
                        .to_string(),
                );
            }
            classify_irreducible(&g, ev)
        }
        _ => {
            let monic: Vec<Vec<Integer>> = nonlinear.iter().map(|g| monicize(g)).collect();
            reducible::classify_composite(&monic, n, ev)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    fn name_of(v: &[i64]) -> String {
        galois_group(&iz(v))
            .unwrap()
            .decided_name()
            .expect("expected a decided group")
            .to_string()
    }

    // ---------------- degenerate and trivial inputs ---------------- //

    #[test]
    fn zero_and_constant_polynomials_are_errors() {
        assert!(galois_group(&iz(&[])).is_err());
        assert!(galois_group(&iz(&[0])).is_err());
        assert!(galois_group(&iz(&[7])).is_err());
    }

    #[test]
    fn linear_and_split_polynomials_are_trivial() {
        assert_eq!(name_of(&[3, 2]), "C1"); // 2x + 3
        assert_eq!(name_of(&[-4, 0, 1]), "C1"); // (x−2)(x+2)
                                                // 4x³ − 3x − 1 = (x − 1)(2x + 1)²: non-monic, non-squarefree, all roots rational.
        assert_eq!(name_of(&[-1, -3, 0, 4]), "C1");
    }

    #[test]
    fn non_squarefree_input_uses_radical() {
        // (x² + 1)²
        let r = galois_group(&iz(&[1, 0, 2, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("C2"));
        assert!(r
            .evidence()
            .notes
            .iter()
            .any(|s| s.contains("not squarefree")));
    }

    // ---------------- degree 2 ---------------- //

    #[test]
    fn quadratics() {
        assert_eq!(name_of(&[1, 0, 1]), "C2"); // x² + 1
        assert_eq!(name_of(&[-2, 0, 1]), "C2"); // x² − 2
        assert_eq!(name_of(&[2, 0, 3]), "C2"); // 3x² + 2 (non-monic)
    }

    // ---------------- degree 3 (sympy-verified) ---------------- //

    #[test]
    fn cubics() {
        assert_eq!(name_of(&[-1, -3, 0, 1]), "C3"); // x³ − 3x − 1, disc = 81
        assert_eq!(name_of(&[-2, 0, 0, 1]), "S3"); // x³ − 2, disc = −108
        assert_eq!(name_of(&[-3, -3, 0, 2]), "S3"); // 2x³ − 3x − 3 (non-monic, Eisenstein at 3)
    }

    // ---------------- degree 4 (all five groups; sympy-verified) -------- //

    #[test]
    fn quartic_v4() {
        assert_eq!(name_of(&[1, 0, 0, 0, 1]), "V4"); // x⁴ + 1
    }

    #[test]
    fn quartic_s4() {
        assert_eq!(name_of(&[1, 1, 0, 0, 1]), "S4"); // x⁴ + x + 1
    }

    #[test]
    fn quartic_c4() {
        assert_eq!(name_of(&[1, 1, 1, 1, 1]), "C4"); // Φ₅ = x⁴ + x³ + x² + x + 1
    }

    #[test]
    fn quartic_d4() {
        assert_eq!(name_of(&[-2, 0, 0, 0, 1]), "D4"); // x⁴ − 2
    }

    #[test]
    fn quartic_a4() {
        assert_eq!(name_of(&[12, 8, 0, 0, 1]), "A4"); // x⁴ + 8x + 12
    }

    // ---------------- degree 5 (sympy-verified) ---------------- //

    #[test]
    fn quintic_s5() {
        // x⁵ − x − 1: the (3,2) type at p = 2 plus non-square disc decide S5.
        let r = galois_group(&iz(&[-1, -1, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("S5"));
        let ev = r.evidence();
        assert_eq!(ev.disc_is_square, Some(false));
        assert!(ev.frobenius_types.iter().any(|(_, t)| t == &vec![3, 2]));
    }

    #[test]
    fn quintic_c5_lehmer() {
        // x⁵ + x⁴ − 4x³ − 3x² + 3x + 1 (Lehmer): C5, decided by the
        // ordered-pair resolvent signature [5,5,5,5].
        let r = galois_group(&iz(&[1, 3, -3, -4, 1, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("C5"));
        let ev = r.evidence();
        assert_eq!(ev.disc_is_square, Some(true));
        assert!(ev
            .resolvent_signatures
            .iter()
            .any(|(_, sig)| sig == &vec![5, 5, 5, 5]));
    }

    #[test]
    fn quintic_d5() {
        // x⁵ − 5x + 12: D5 (sympy-verified), decided by the ordered-pair
        // resolvent signature [10,10] (the (2,2,1) type alone only rules out C5).
        let r = galois_group(&iz(&[12, -5, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("D5"));
        assert!(r
            .evidence()
            .resolvent_signatures
            .iter()
            .any(|(_, sig)| sig == &vec![10, 10]));
    }

    #[test]
    fn quintic_a5() {
        // x⁵ + 20x + 16: A5 (sympy-verified); square disc + the (3,1,1) type
        // at p = 7 rule out everything else — no resolvent needed.
        let r = galois_group(&iz(&[16, 20, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("A5"));
        assert!(r.evidence().resolvent_signatures.is_empty());
    }

    #[test]
    fn quintic_f20_is_honestly_unresolved() {
        // x⁵ − 2 has Gal = F20 (sympy-verified), but parity, the sieve and the
        // ordered-pair resolvent cannot separate F20 from S5. The honest answer
        // is Unresolved with exactly {F20, S5} remaining — a bounded search
        // that finds no S5-only cycle type must NOT decide F20.
        let r = galois_group(&iz(&[-2, 0, 0, 0, 0, 1])).unwrap();
        assert!(!r.is_decided());
        let mut names = r.candidate_names().unwrap();
        names.sort_unstable();
        assert_eq!(names, vec!["F20", "S5"]);
        match &r {
            GaloisGroupResult::Unresolved { ruled_out, blocked_on, evidence, .. } => {
                // C5, D5, A5 are ruled out by parity (disc not a square).
                for g in ["C5", "D5", "A5"] {
                    assert!(
                        ruled_out.iter().any(|(id, _)| id.name == g),
                        "{g} should be ruled out"
                    );
                }
                assert!(blocked_on.contains("Cayley") || blocked_on.contains("Stauduhar"));
                // The exact ordered-pair signature [20] was computed and recorded.
                assert!(evidence
                    .resolvent_signatures
                    .iter()
                    .any(|(_, sig)| sig == &vec![20]));
            }
            GaloisGroupResult::Decided { .. } => unreachable!(),
        }
    }

    // ---------------- degree ≥ 6: Jordan certificates (sympy-verified) -- //

    #[test]
    fn sextic_s6() {
        // x⁶ − x − 1: S6 (order 720, sympy-verified). Decided via the
        // (5,1)-cycle (2-transitivity) plus a transposition power.
        let r = galois_group(&iz(&[-1, -1, 0, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("S6"));
    }

    #[test]
    fn sextic_a6() {
        // x⁶ + 24x − 20: A6 (order 360, sympy-verified). Decided via the
        // (5,1)-cycle plus a 3-cycle power plus square discriminant.
        let r = galois_group(&iz(&[-20, 24, 0, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("A6"));
        assert_eq!(r.evidence().disc_is_square, Some(true));
    }

    #[test]
    fn octic_s8() {
        // x⁸ − x − 1: S8 (Osada: Gal(xⁿ − x − 1) = S_n for all n; the (5,3)
        // type at p = 3 powers to a 5-cycle with 4 < 5 ≤ 5 = n−3 — Jordan).
        let r = galois_group(&iz(&[-1, -1, 0, 0, 0, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("S8"));
        assert_eq!(
            match &r {
                GaloisGroupResult::Decided { group, .. } => group.order,
                _ => None,
            },
            Some(40320)
        );
    }

    // ---------------- reducible composita ---------------- //

    #[test]
    fn multiquadratic_composita() {
        // (x² − 2)(x² − 3) → V4 = C2².
        let f = zx::mul(&iz(&[-2, 0, 1]), &iz(&[-3, 0, 1]));
        assert_eq!(galois_group(&f).unwrap().decided_name(), Some("C2^2"));
        // (x² − 2)(x² − 8): 2·8 = 16 is a square — same field, C2.
        let f = zx::mul(&iz(&[-2, 0, 1]), &iz(&[-8, 0, 1]));
        assert_eq!(galois_group(&f).unwrap().decided_name(), Some("C2"));
        // (x² − 2)(x² − 3)(x² − 6): 6 = 2·3 is dependent — still rank 2.
        let f = zx::mul(
            &zx::mul(&iz(&[-2, 0, 1]), &iz(&[-3, 0, 1])),
            &iz(&[-6, 0, 1]),
        );
        assert_eq!(galois_group(&f).unwrap().decided_name(), Some("C2^2"));
    }

    #[test]
    fn cubic_times_quadratic() {
        // (x³ − 2)(x² + 3): ℚ(∛2, ζ₃) already contains √−3 → S3 (order 6).
        let f = zx::mul(&iz(&[-2, 0, 0, 1]), &iz(&[3, 0, 1]));
        assert_eq!(galois_group(&f).unwrap().decided_name(), Some("S3"));
        // (x³ − 2)(x² + 1): ℚ(∛2, ζ₃, i) has degree 12 → S3 × C2.
        let f = zx::mul(&iz(&[-2, 0, 0, 1]), &iz(&[1, 0, 1]));
        assert_eq!(galois_group(&f).unwrap().decided_name(), Some("S3xC2"));
        // (x³ − 3x − 1)(x² + 1): cyclic cubic × C2 → C6.
        let f = zx::mul(&iz(&[-1, -3, 0, 1]), &iz(&[1, 0, 1]));
        assert_eq!(galois_group(&f).unwrap().decided_name(), Some("C6"));
    }

    #[test]
    fn two_nonquadratic_factors_are_unresolved() {
        // (x³ − 2)(x³ − 3): Goursat analysis not implemented → honest Unresolved.
        let f = zx::mul(&iz(&[-2, 0, 0, 1]), &iz(&[-3, 0, 0, 1]));
        let r = galois_group(&f).unwrap();
        assert!(!r.is_decided());
    }

    #[test]
    fn quartic_times_linear_delegates() {
        // (x⁴ + 1)(x − 3) → V4, with a note about the linear factor.
        let f = zx::mul(&iz(&[1, 0, 0, 0, 1]), &iz(&[-3, 1]));
        let r = galois_group(&f).unwrap();
        assert_eq!(r.decided_name(), Some("V4"));
        assert!(r.evidence().notes.iter().any(|s| s.contains("linear factors")));
    }

    // ---------------- t-number bookkeeping ---------------- //

    #[test]
    fn t_numbers_for_the_5t_table() {
        let cases: [(&[i64], u32); 3] = [
            (&[1, 3, -3, -4, 1, 1], 1),  // C5 = 5T1
            (&[12, -5, 0, 0, 0, 1], 2),  // D5 = 5T2
            (&[-1, -1, 0, 0, 0, 1], 5),  // S5 = 5T5
        ];
        for (coeffs, t) in cases {
            match galois_group(&iz(coeffs)).unwrap() {
                GaloisGroupResult::Decided { group, .. } => {
                    assert_eq!(group.t_number, Some(t));
                    assert_eq!(group.degree, 5);
                }
                GaloisGroupResult::Unresolved { .. } => panic!("expected decided"),
            }
        }
    }
}

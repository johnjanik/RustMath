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
//! - **Degree 5**: complete exact decision. Frobenius cycle-type sieve over
//!   the 5T1–5T5 table, sharpened by an exact ordered-pair resolvent, with
//!   the F20-vs-S5 blind spot closed by the Cayley–Dummit sextic resolvent
//!   (rational root ⟺ solvable ⟺ `Gal ⊆ F20` up to conjugacy). All five
//!   groups C5/D5/F20/A5/S5 are decided. See [`quintic`] and [`cayley`].
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

pub mod cayley;
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
    fn quintic_f20_decided_by_cayley_resolvent() {
        // x⁵ − 2 has Gal = F20 (gp polgalois: [20, -1, 1, "F(5) = 5:4"]).
        // Parity rules out C5/D5/A5; the ordered-pair resolvent cannot see
        // F20 vs S5 (both 2-transitive, signature [20]); the Cayley–Dummit
        // sextic resolvent (squarefree, rational root 0 — gp-verified)
        // certifies solvability and decides F20.
        let r = galois_group(&iz(&[-2, 0, 0, 0, 0, 1])).unwrap();
        assert_eq!(r.decided_name(), Some("F20"));
        let ev = r.evidence();
        assert_eq!(ev.disc_is_square, Some(false));
        // The exact ordered-pair signature [20] was computed and recorded…
        assert!(ev.resolvent_signatures.iter().any(|(_, sig)| sig == &vec![20]));
        // …and the Cayley sextic factored as [1, 5]: Gal fixes one F20-coset.
        assert!(ev
            .resolvent_signatures
            .iter()
            .any(|(desc, sig)| desc.contains("Cayley") && sig == &vec![1, 5]));
        match &r {
            GaloisGroupResult::Decided { group, .. } => {
                assert_eq!(group.order, Some(20));
                assert_eq!(group.t_number, Some(3));
            }
            GaloisGroupResult::Unresolved { .. } => unreachable!(),
        }
    }

    /// The full gp `polgalois` battery: 16 structured gates covering all five
    /// groups and 20 random irreducible quintics compared blindly. Every
    /// verdict below was computed with PARI/GP 2.17.3 `polgalois`
    /// (setrand(20260716) for the random block; coefficients uniform in
    /// [−10, 10]; only irreducible draws kept).
    #[test]
    fn quintic_battery_matches_gp_polgalois() {
        let cases: [(&[i64], &str); 36] = [
            // structured gates (gp-verified individually)
            (&[-2, 0, 0, 0, 0, 1], "F20"),
            (&[-1, -1, 0, 0, 0, 1], "S5"),
            (&[16, 20, 0, 0, 0, 1], "A5"),
            (&[1, 3, -3, -4, 1, 1], "C5"),
            (&[12, -5, 0, 0, 0, 1], "D5"),
            (&[12, 15, 0, 0, 0, 1], "F20"),
            (&[-12, -5, 0, 0, 0, 1], "D5"),
            (&[-30, 0, 0, 0, 0, 1], "F20"),
            (&[-29, 0, 0, 0, 0, 1], "F20"),
            (&[-28, 0, 0, 0, 0, 1], "F20"),
            (&[-27, 0, 0, 0, 0, 1], "F20"),
            (&[-16, 20, 0, 0, 0, 1], "A5"),
            // C5: polsubcyclo(p, 5) for p = 31, 41, 61, 71
            (&[5, 1, -21, -12, 1, 1], "C5"),
            (&[-9, 21, 5, -16, 1, 1], "C5"),
            (&[-13, 41, -17, -24, 1, 1], "C5"),
            (&[1, 25, 37, -28, 1, 1], "C5"),
            // 20 random irreducible quintics, verdicts taken blindly from gp
            (&[-8, -2, -3, 2, 1, 1], "S5"),
            (&[10, -10, -9, -6, -1, 1], "S5"),
            (&[4, 1, 1, -5, 0, 1], "S5"),
            (&[9, -4, -1, 6, 5, 1], "S5"),
            (&[3, 8, 7, 0, 1, 1], "S5"),
            (&[3, 10, 7, -7, -5, 1], "S5"),
            (&[-9, 10, -7, -5, 6, 1], "S5"),
            (&[-2, -2, 4, 8, -5, 1], "S5"),
            (&[4, 9, -10, -9, -9, 1], "S5"),
            (&[-3, -9, 3, -7, -4, 1], "S5"),
            (&[3, -3, 9, 4, 3, 1], "S5"),
            (&[8, 1, -4, 1, -9, 1], "S5"),
            (&[-7, 8, 2, 1, -3, 1], "S5"),
            (&[-9, -4, -5, 8, 10, 1], "S5"),
            (&[5, -2, -7, 4, -10, 1], "S5"),
            (&[8, 7, 4, -6, -4, 1], "S5"),
            (&[3, -6, 6, -3, -7, 1], "S5"),
            (&[-2, 8, 10, -6, 10, 1], "S5"),
            (&[5, -6, 4, 8, 5, 1], "S5"),
            (&[4, 7, 3, -3, 10, 1], "S5"),
        ];
        for (coeffs, want) in cases {
            let r = galois_group(&iz(coeffs)).unwrap();
            assert_eq!(
                r.decided_name(),
                Some(want),
                "polgalois disagreement on {coeffs:?}"
            );
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
        let cases: [(&[i64], u32); 5] = [
            (&[1, 3, -3, -4, 1, 1], 1),  // C5 = 5T1
            (&[12, -5, 0, 0, 0, 1], 2),  // D5 = 5T2
            (&[-2, 0, 0, 0, 0, 1], 3),   // F20 = 5T3
            (&[16, 20, 0, 0, 0, 1], 4),  // A5 = 5T4
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

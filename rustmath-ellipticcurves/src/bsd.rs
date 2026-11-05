//! Birch and Swinnerton-Dyer conjecture verification, wired to the real
//! components of this crate:
//!
//! * algebraic rank — certified 2-descent ([`crate::rank`]);
//! * analytic rank — the honest lattice ([`crate::lfunction`]), with the
//!   root number complete at every prime ([`crate::rootnumber`] +
//!   Kraus/Halberstadt);
//! * Ω — the AGM real period of the minimal model ([`crate::period`]);
//! * regulator — Néron–Tate heights over `BigFloat` ([`crate::height`]),
//!   on the descent's infinite-order witness points (NOT saturated — see
//!   [`BSDVerifier::verify_conjecture`]);
//! * |T|, c_p — exact torsion and Tate data;
//! * Ш — the analytic order ASSUMING BSD ([`crate::bsdratio`], which
//!   labels its conditionality), when a rank-0/rank-1 assembly applies;
//!   otherwise an honest `Err` is carried in the result.

use crate::bsdratio::AnalyticShaAssumingBSD;
use crate::curve::{EllipticCurve, Point};
use crate::lfunction::{AnalyticRank, LFunction};
use crate::rank::RankBoundResult;
use rustmath_integers::Integer;
use rustmath_reals::bigfloat::BigFloat;

/// Decimal digits used for the numeric analytic-rank layer in BSD checks.
const BSD_ANALYTIC_DIGITS: usize = 26;

/// Working precision (bits) for the Ω / regulator legs of the verifier.
const BSD_PREC_BITS: u64 = 128;

/// Result of BSD conjecture verification. The numeric legs are certified
/// `BigFloat`s (see the respective modules for the error models); the Sha
/// leg is the analytic order assuming BSD, or an honest reason why no
/// assembly applies.
#[derive(Debug, Clone)]
pub struct BSDResult {
    pub algebraic_rank: u32,
    /// The analytic rank in the honest lattice (certified 0/1, at-least-2,
    /// or unresolved with reason) — never a bare fabricated integer.
    pub analytic_rank: AnalyticRank,
    /// The analytic order of Ш ASSUMING BSD (labeled as such in its type),
    /// or the honest reason no rank-0/rank-1 assembly applies.
    pub sha_analytic: Result<AnalyticShaAssumingBSD, String>,
    /// Regulator of the descent witnesses (1 for rank 0; ĥ-based; the
    /// witnesses are not saturated, so this is the regulator of the
    /// sublattice they generate).
    pub regulator: BigFloat,
    /// The real period Ω_E of the global minimal model (AGM, certified).
    pub real_period: BigFloat,
    pub tamagawa_numbers: Vec<u32>,
    pub torsion_order: u32,
}

impl BSDResult {
    /// Check if the ranks agree (weak BSD): true only when the analytic
    /// rank is CERTIFIED and equals the algebraic rank. An unresolved
    /// analytic rank never "agrees".
    pub fn ranks_agree(&self) -> bool {
        self.analytic_rank.certified_value() == Some(self.algebraic_rank)
    }

    /// The analytic order of Ш (assuming BSD), when an assembly applied.
    pub fn sha_order(&self) -> Option<u32> {
        self.sha_analytic.as_ref().ok().map(|s| s.order)
    }
}

/// BSD conjecture verifier
pub struct BSDVerifier {
    curve: EllipticCurve,
    computed_rank: Option<u32>,
    analytic_rank: Option<AnalyticRank>,
    generators: Vec<Point>,
}

impl BSDVerifier {
    /// Create a new BSD verifier for a curve
    pub fn new(curve: EllipticCurve) -> Self {
        Self {
            curve,
            computed_rank: None,
            analytic_rank: None,
            generators: Vec::new(),
        }
    }

    /// Verify the BSD conjecture: certified algebraic rank (2-descent),
    /// the honest analytic-rank lattice, real Ω / regulator / torsion /
    /// Tamagawa legs, and the analytic Ш assuming BSD where a rank-0 or
    /// rank-1 assembly applies.
    ///
    /// Rank-1 caveat (recorded in the Ш provenance): the regulator input
    /// is the descent's infinite-order witness point, which is NOT
    /// saturated; if it is m·(generator) the Ш assembly evaluates to
    /// Ш_an/m² and fails its integrality check with an honest `Err`
    /// rather than fabricating an order.
    ///
    /// # Panics
    ///
    /// Panics (honest refusal, never a guess) when the algebraic rank is
    /// undetermined — see [`Self::compute_algebraic_rank`].
    pub fn verify_conjecture(&mut self) -> BSDResult {
        let algebraic_rank = self.compute_algebraic_rank();
        let analytic_rank = self.compute_analytic_rank();
        let sha_analytic = self.compute_sha_analytic();
        let regulator = self.compute_regulator();
        let real_period = self.curve.real_period(BSD_PREC_BITS);
        let tamagawa = self.compute_tamagawa_numbers();
        let torsion = self.torsion_order();

        BSDResult {
            algebraic_rank,
            analytic_rank,
            sha_analytic,
            regulator,
            real_period,
            tamagawa_numbers: tamagawa,
            torsion_order: torsion,
        }
    }

    /// Compute the algebraic rank via genuine 2-descent (see [`crate::rank`]).
    ///
    /// Returns the exact rank when the certified descent interval collapses
    /// (`lower == upper`), recording the infinite-order witness points found
    /// by the descent as `self.generators` (a subset of a generating set —
    /// saturation is not performed).
    ///
    /// # Panics
    ///
    /// Honest refusal, never a guess:
    /// * when the interval stays open (`lower < upper`): everywhere-locally-
    ///   solvable torsors without rational points, i.e. a possible
    ///   nontrivial Sha[2] obstruction;
    /// * when the curve has no rational 2-torsion (2-descent over Q does
    ///   not apply; number-field descent is out of scope).
    fn compute_algebraic_rank(&mut self) -> u32 {
        match self.curve.rank_bounds() {
            RankBoundResult::Bounds(b) => {
                if b.lower == b.upper {
                    self.generators = b.infinite_order_points(&self.curve);
                    self.computed_rank = Some(b.lower);
                    b.lower
                } else {
                    panic!(
                        "algebraic rank undetermined: 2-descent certifies rank ∈ [{}, {}] \
                         (everywhere-locally-solvable torsors without rational points — \
                         possible nontrivial Sha[2]); refusing to fabricate an exact rank",
                        b.lower, b.upper
                    )
                }
            }
            RankBoundResult::Unresolved { reason } => {
                panic!("algebraic rank undetermined: {}", reason)
            }
        }
    }

    /// Compute the analytic rank in the honest lattice (see
    /// [`crate::lfunction::AnalyticRank`]): certified 0 via a nonzero
    /// L(1), certified 1 via the exact ε = −1 zero plus a certified
    /// nonzero L'(1), or an honest unresolved outcome — never a bare
    /// fabricated integer.
    fn compute_analytic_rank(&mut self) -> AnalyticRank {
        let l_function = LFunction::new(self.curve.clone());
        let rank = l_function.analytic_rank(BSD_ANALYTIC_DIGITS);
        self.analytic_rank = Some(rank.clone());
        rank
    }

    /// The analytic order of Ш assuming BSD (REAL now — this replaces the
    /// old `estimate_sha_size` facade): rank 0 → the certified L(1)/Ω
    /// recognition; rank 1 with a descent witness → the L′/(Ω·Reg)
    /// assembly with the witness as (unsaturated) generator. `Err` with
    /// the honest reason otherwise. Must be called after the rank legs.
    fn compute_sha_analytic(&mut self) -> Result<AnalyticShaAssumingBSD, String> {
        let an = self
            .analytic_rank
            .clone()
            .expect("compute_sha_analytic called before compute_analytic_rank");
        match an.certified_value() {
            Some(0) => self.curve.analytic_sha_rank0(BSD_ANALYTIC_DIGITS),
            Some(1) => match self.generators.first() {
                Some(gen) => self
                    .curve
                    .analytic_sha_rank1(gen, BSD_ANALYTIC_DIGITS)
                    .map_err(|e| {
                        format!(
                            "rank-1 Sha assembly with the (unsaturated) descent \
                             witness failed: {}",
                            e
                        )
                    }),
                None => Err(
                    "analytic rank 1 but no infinite-order witness available (descent \
                     found none); cannot form the rank-1 regulator"
                        .to_string(),
                ),
            },
            _ => Err(format!("no Sha assembly applies: analytic rank is {}", an)),
        }
    }

    /// The regulator of the descent's infinite-order witness points
    /// (REAL Néron–Tate heights over BigFloat, Sage/LMFDB normalization;
    /// 1 for rank 0). The witnesses are not saturated: this is the
    /// regulator of the sublattice they generate.
    fn compute_regulator(&mut self) -> BigFloat {
        self.curve.regulator(&self.generators, BSD_PREC_BITS)
    }

    /// Compute Tamagawa numbers c_p at the bad primes (ascending order of
    /// p), via Tate's algorithm (see `crate::tate`).
    ///
    /// Primes dividing the discriminant of the given model at which the
    /// curve actually has good reduction (non-minimal model) are skipped.
    /// If the list would be empty it contains a single 1, preserving the
    /// invariant that the Tamagawa product over the returned list is the
    /// true product over all primes.
    fn compute_tamagawa_numbers(&self) -> Vec<u32> {
        let mut tamagawa = Vec::new();

        for (p, _) in rustmath_integers::prime::factor(&self.curve.discriminant.abs()) {
            let local = self.curve.local_data(&p);
            if local.conductor_exponent > 0 {
                tamagawa.push(local.tamagawa_number);
            }
        }

        if tamagawa.is_empty() {
            vec![1]
        } else {
            tamagawa
        }
    }

    /// Order of the torsion subgroup E(Q)_tors (exact, via minimal model +
    /// reduction bound + Lutz–Nagell; see [`crate::torsion`]).
    fn torsion_order(&self) -> u32 {
        self.curve.torsion_subgroup().order
    }

    /// Check if the weak BSD conjecture holds: `Ok(true/false)` when the
    /// analytic rank is certified (comparing it to the descent-certified
    /// algebraic rank), `Err` with the honest reason when it is not.
    pub fn check_weak_bsd(&mut self) -> Result<bool, String> {
        let alg_rank = self.compute_algebraic_rank();
        let an_rank = self.compute_analytic_rank();
        match an_rank.certified_value() {
            Some(v) => Ok(v == alg_rank),
            None => Err(format!(
                "weak BSD undecidable here: analytic rank is {}",
                an_rank
            )),
        }
    }

    /// Generate a BSD report
    pub fn generate_report(&mut self) -> String {
        let result = self.verify_conjecture();

        let sha_line = match &result.sha_analytic {
            Ok(sha) => format!("{} (assuming BSD)", sha.order),
            Err(reason) => format!("no assembly applies: {}", reason),
        };
        format!(
            "BSD Conjecture Verification Report\n\
             =====================================\n\
             Curve: {}\n\
             Discriminant: {}\n\
             Conductor: {}\n\n\
             Algebraic Rank: {}\n\
             Analytic Rank: {}\n\
             Ranks Agree: {}\n\n\
             Regulator (descent witnesses, unsaturated): {}\n\
             Real Period Omega: {}\n\
             Torsion Order: {}\n\
             Tamagawa Numbers: {:?}\n\n\
             Analytic |Sha|: {}\n",
            self.curve,
            self.curve.discriminant,
            self.curve.conductor.as_ref().unwrap_or(&Integer::zero()),
            result.algebraic_rank,
            result.analytic_rank,
            result.ranks_agree(),
            result.regulator.to_decimal_string(20),
            result.real_period.to_decimal_string(20),
            result.torsion_order,
            result.tamagawa_numbers,
            sha_line
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsd_verifier_creation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));

        let verifier = BSDVerifier::new(curve);
        assert!(verifier.computed_rank.is_none());
    }

    #[test]
    fn test_algebraic_rank_computation() {
        // Real now (was an `unimplemented!` facade): y² = x³ − x has rank
        // exactly 0, certified by 2-descent (interval collapses to [0, 0];
        // Python-verified).
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let mut verifier = BSDVerifier::new(curve);
        let rank = verifier.compute_algebraic_rank();
        assert_eq!(rank, 0);
        assert_eq!(verifier.computed_rank, Some(0));
        // rank 0: no infinite-order generators
        assert!(verifier.generators.is_empty());
    }

    /// The analytic rank legs, now decided everywhere: y² = x³ − 1
    /// (N = 144, additive at 2 and 3) MOVED from an honest Unresolved to a
    /// certified analytic rank 0 by the Kraus/Halberstadt tables (w = +1,
    /// L(1) = 1.21432532394… ≠ 0; PARI + mpmath derived, gated in
    /// lfunction::tests::test_l1_wild_additive_movers); 11a1 stays
    /// certified 0 via its nonzero L(1).
    #[test]
    fn test_analytic_rank_computation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(0), Integer::from(-1));
        let mut verifier = BSDVerifier::new(curve);
        let rank = verifier.compute_analytic_rank();
        assert!(matches!(rank, AnalyticRank::ZeroCertified { .. }));
        assert_eq!(rank.certified_value(), Some(0));

        let e11 = EllipticCurve::new(
            Integer::from(0),
            Integer::from(-1),
            Integer::from(1),
            Integer::from(-10),
            Integer::from(-20),
        );
        let mut verifier = BSDVerifier::new(e11);
        let rank = verifier.compute_analytic_rank();
        assert_eq!(rank.certified_value(), Some(0));
    }

    /// END-TO-END verify_conjecture on y² = x³ + x (N = 64, additive at
    /// 2 — decidable only with the wild tables): certified ranks 0/0,
    /// real Ω = 3.70814935460274383686… (PARI E.omega + independent AGM
    /// derivation, both before this test), regulator exactly 1 (rank 0),
    /// |T| = 2, c₂ = 1, and Ш_an = 1 assuming BSD (L/Ω = 1/4 recognized).
    #[test]
    fn test_verify_conjecture_end_to_end_x3px() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(1), Integer::from(0));
        let mut verifier = BSDVerifier::new(curve);
        let result = verifier.verify_conjecture();
        assert_eq!(result.algebraic_rank, 0);
        assert_eq!(result.analytic_rank.certified_value(), Some(0));
        assert!(result.ranks_agree());
        assert_eq!(result.torsion_order, 2);
        assert_eq!(result.tamagawa_numbers, vec![1]);
        let omega_truth =
            BigFloat::from_decimal_str("3.7081493546027438368677006943905200924351976470435", 192)
                .unwrap();
        let d =
            rustmath_core::ordering::OrderedRing::abs(&(result.real_period.clone() - omega_truth));
        let tol = BigFloat::from_decimal_str("0.000000000000000000000001", 192).unwrap();
        assert!(d < tol, "Omega to 24 digits, got {}", result.real_period);
        assert_eq!(result.sha_order(), Some(1), "Sha_an = 1 assuming BSD");
        let sha = result.sha_analytic.as_ref().unwrap();
        assert!(sha.provenance.contains("ASSUMING BSD"));
    }

    #[test]
    fn test_tamagawa_numbers() {
        // y² = x³ - x + 1: disc = -368 = -2⁴·23; Tate gives type IV at 2
        // with c₂ = 3 and I1 at 23 with c₂₃ = 1 (PARI/GP-verified).
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));

        let verifier = BSDVerifier::new(curve);
        let tamagawa = verifier.compute_tamagawa_numbers();
        assert_eq!(tamagawa, vec![3, 1]);

        // 15a1: c₃ = 2 (non-split I4), c₅ = 4 (split I4); PARI-verified.
        let e15 = EllipticCurve::new(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
            Integer::from(-10),
            Integer::from(-10),
        );
        let verifier = BSDVerifier::new(e15);
        assert_eq!(verifier.compute_tamagawa_numbers(), vec![2, 4]);
    }

    /// Weak BSD verifies END TO END on 15a1 — and now ALSO on y² = x³ + 1
    /// (N = 36, additive at 2 and 3), which MOVED from an honest Err
    /// ("root number unresolved") to Ok(true): the Kraus/Halberstadt
    /// tables give w = +1, L(1) = 0.70109105266… ≠ 0 certifies analytic
    /// rank 0, and 2-descent certifies algebraic rank 0.
    #[test]
    fn test_weak_bsd() {
        let e15 = EllipticCurve::new(
            Integer::from(1),
            Integer::from(1),
            Integer::from(1),
            Integer::from(-10),
            Integer::from(-10),
        );
        let mut verifier = BSDVerifier::new(e15);
        assert_eq!(verifier.check_weak_bsd(), Ok(true));

        let curve = EllipticCurve::from_short_weierstrass(Integer::from(0), Integer::from(1));
        let mut verifier = BSDVerifier::new(curve);
        assert_eq!(
            verifier.check_weak_bsd(),
            Ok(true),
            "y² = x³ + 1 moved from Unresolved to a decided weak-BSD instance"
        );
    }

    #[test]
    fn test_bsd_result() {
        let e65 = EllipticCurve::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(-1),
            Integer::from(0),
        );
        let analytic = LFunction::new(e65).analytic_rank(20);
        assert_eq!(analytic.certified_value(), Some(1), "65a1 analytic rank 1");
        let result = BSDResult {
            algebraic_rank: 1,
            analytic_rank: analytic,
            sha_analytic: Err("not assembled in this literal".to_string()),
            regulator: BigFloat::one_prec(64),
            real_period: BigFloat::from_decimal_str("2.5", 64).unwrap(),
            tamagawa_numbers: vec![1],
            torsion_order: 1,
        };

        assert!(result.ranks_agree());
        assert_eq!(result.sha_order(), None, "Err carries no order");

        // an unresolved analytic rank never "agrees"
        let result2 = BSDResult {
            algebraic_rank: 0,
            analytic_rank: AnalyticRank::Unresolved {
                reason: "test".to_string(),
            },
            sha_analytic: Err("test".to_string()),
            regulator: BigFloat::one_prec(64),
            real_period: BigFloat::from_decimal_str("2.5", 64).unwrap(),
            tamagawa_numbers: vec![1],
            torsion_order: 1,
        };
        assert!(!result2.ranks_agree());
    }

    /// UN-IGNORED (was a facade victim): report generation is real now.
    /// Re-pointed from y² = x³ − x + 1 (whose 2-descent is still an honest
    /// refusal — no rational 2-torsion) to y² = x³ − x, where every leg
    /// works: certified ranks 0/0, real Ω, Ш_an = 1 assuming BSD.
    #[test]
    fn test_generate_report() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(0));

        let mut verifier = BSDVerifier::new(curve);
        let report = verifier.generate_report();

        assert!(report.contains("BSD Conjecture"));
        assert!(report.contains("Algebraic Rank: 0"));
        assert!(report.contains("Analytic Rank: 0"));
        assert!(report.contains("Ranks Agree: true"));
        assert!(report.contains("Analytic |Sha|: 1 (assuming BSD)"));
    }
}

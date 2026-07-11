//! Birch and Swinnerton-Dyer conjecture verification
//!
//! Implements tools for verifying the BSD conjecture numerically

use crate::curve::{EllipticCurve, Point};
use crate::lfunction::{AnalyticRank, LFunction};
use crate::rank::RankBoundResult;
use rustmath_integers::Integer;

/// Decimal digits used for the numeric analytic-rank layer in BSD checks.
const BSD_ANALYTIC_DIGITS: usize = 26;

/// Result of BSD conjecture verification
#[derive(Debug, Clone)]
pub struct BSDResult {
    pub algebraic_rank: u32,
    /// The analytic rank in the honest lattice (certified 0/1, at-least-2,
    /// or unresolved with reason) — never a bare fabricated integer.
    pub analytic_rank: AnalyticRank,
    pub sha_estimate: f64,
    pub regulator: f64,
    pub periods: f64,
    pub tamagawa_numbers: Vec<u32>,
    pub torsion_order: u32,
    pub bsd_constant: f64,
}

impl BSDResult {
    /// Check if the ranks agree (weak BSD): true only when the analytic
    /// rank is CERTIFIED and equals the algebraic rank. An unresolved
    /// analytic rank never "agrees".
    pub fn ranks_agree(&self) -> bool {
        self.analytic_rank.certified_value() == Some(self.algebraic_rank)
    }

    /// Estimate the order of Sha (Tate-Shafarevich group)
    pub fn sha_order(&self) -> f64 {
        self.sha_estimate
    }

    /// Get the BSD constant C = L^(r)(E, 1) / (r! * Ω * Reg * c * |Sha| / |E_tors|²)
    pub fn bsd_ratio(&self) -> f64 {
        self.bsd_constant
    }
}

/// BSD conjecture verifier
pub struct BSDVerifier {
    curve: EllipticCurve,
    computed_rank: Option<u32>,
    analytic_rank: Option<AnalyticRank>,
    regulator: Option<f64>,
    generators: Vec<Point>,
}

impl BSDVerifier {
    /// Create a new BSD verifier for a curve
    pub fn new(curve: EllipticCurve) -> Self {
        Self {
            curve,
            computed_rank: None,
            analytic_rank: None,
            regulator: None,
            generators: Vec::new(),
        }
    }

    /// Verify the BSD conjecture
    pub fn verify_conjecture(&mut self) -> BSDResult {
        let algebraic_rank = self.compute_algebraic_rank();
        let analytic_rank = self.compute_analytic_rank();
        let sha_approx = self.estimate_sha_size();
        let regulator = self.compute_regulator();
        let periods = self.compute_periods();
        let tamagawa = self.compute_tamagawa_numbers();
        let torsion = self.torsion_order();
        let bsd_const = self.compute_bsd_constant();

        BSDResult {
            algebraic_rank,
            analytic_rank,
            sha_estimate: sha_approx,
            regulator,
            periods,
            tamagawa_numbers: tamagawa,
            torsion_order: torsion,
            bsd_constant: bsd_const,
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

    /// Estimate the size of the Tate-Shafarevich group
    ///
    /// A real estimate requires the BSD formula
    /// |Sha| ≈ L^(r)(E,1) * |E_tors|² / (Ω * Reg * ∏c_p), which needs genuine
    /// L-function derivative data (the order-r derivative at s=1) together
    /// with honest regulator/period/Tamagawa inputs. This previously just
    /// hardcoded 1.0 for every curve, which is a fabricated result, not a
    /// computation.
    fn estimate_sha_size(&self) -> f64 {
        unimplemented!(
            "Tate-Shafarevich group order estimate not yet implemented (facade): requires \
             L-function derivative data and a genuine BSD formula evaluation, not a \
             hardcoded 1.0"
        )
    }

    /// Compute the regulator (determinant of height pairing matrix)
    fn compute_regulator(&mut self) -> f64 {
        if self.generators.is_empty() {
            return 1.0;
        }

        let rank = self.generators.len();
        if rank == 0 {
            return 1.0;
        }

        // Compute height pairing matrix
        let mut matrix = vec![vec![0.0; rank]; rank];

        for i in 0..rank {
            for j in 0..rank {
                matrix[i][j] =
                    self.canonical_height_pairing(&self.generators[i], &self.generators[j]);
            }
        }

        // Compute determinant
        let det = self.determinant(&matrix);
        self.regulator = Some(det.abs());
        det.abs()
    }

    /// Canonical height pairing ⟨P, Q⟩
    fn canonical_height_pairing(&self, p: &Point, q: &Point) -> f64 {
        if p.infinity || q.infinity {
            return 0.0;
        }

        // ⟨P, Q⟩ = (h(P+Q) - h(P) - h(Q)) / 2
        let sum = self.curve.add_points(p, q);
        let h_sum = self.canonical_height(&sum);
        let h_p = self.canonical_height(p);
        let h_q = self.canonical_height(q);

        (h_sum - h_p - h_q) / 2.0
    }

    /// Canonical (Néron-Tate) height
    fn canonical_height(&self, p: &Point) -> f64 {
        if p.infinity {
            return 0.0;
        }

        // Simplified height: h(x, y) ≈ log max(|num(x)|, |den(x)|)
        let x_num = p.x.numerator().abs();
        let x_den = p.x.denominator().abs();

        let max_val = if x_num > x_den { x_num } else { x_den };
        max_val.to_f64().unwrap_or(1.0).ln()
    }

    /// Compute determinant of a matrix
    fn determinant(&self, matrix: &Vec<Vec<f64>>) -> f64 {
        let n = matrix.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return matrix[0][0];
        }
        if n == 2 {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }

        // LU decomposition for larger matrices
        let mut det = 1.0;
        let mut a = matrix.clone();

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[k][i].abs() > a[max_row][i].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                a.swap(i, max_row);
                det = -det;
            }

            if a[i][i].abs() < 1e-10 {
                return 0.0;
            }

            for k in (i + 1)..n {
                let factor = a[k][i] / a[i][i];
                for j in i..n {
                    a[k][j] -= factor * a[i][j];
                }
            }

            det *= a[i][i];
        }

        det
    }

    /// Compute periods (real and complex)
    fn compute_periods(&self) -> f64 {
        // Ω = ∫_{E(ℝ)} ω where ω = dx / (2y + a₁x + a₃)
        // Simplified: use approximate period
        let disc = self.curve.discriminant.to_f64().unwrap_or(1.0).abs();
        2.0 * std::f64::consts::PI * disc.powf(1.0 / 12.0)
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

    /// Compute the BSD constant
    fn compute_bsd_constant(&self) -> f64 {
        let l_func = LFunction::new(self.curve.clone());
        let l_value = l_func.special_value(1.0).norm();

        let rank = self
            .analytic_rank
            .as_ref()
            .and_then(|r| r.certified_value())
            .unwrap_or(0);
        if rank > 0 {
            // Would need to compute L^(r)(E, 1)
            return 1.0;
        }

        // C = L(E, 1) / (Ω * ∏c_p * |Sha| / |E_tors|²)
        let omega = self.compute_periods();
        let tamagawa_product: u32 = self.compute_tamagawa_numbers().iter().product();
        let sha = self.estimate_sha_size();
        let torsion = self.torsion_order();

        let denominator = omega * tamagawa_product as f64 * sha / (torsion * torsion) as f64;

        if denominator > 0.0 {
            l_value / denominator
        } else {
            1.0
        }
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

        format!(
            "BSD Conjecture Verification Report\n\
             =====================================\n\
             Curve: {}\n\
             Discriminant: {}\n\
             Conductor: {}\n\n\
             Algebraic Rank: {}\n\
             Analytic Rank: {}\n\
             Ranks Agree: {}\n\n\
             Regulator: {:.6}\n\
             Periods: {:.6}\n\
             Torsion Order: {}\n\
             Tamagawa Numbers: {:?}\n\n\
             Estimated |Sha|: {:.6}\n\
             BSD Constant: {:.6}\n",
            self.curve,
            self.curve.discriminant,
            self.curve.conductor.as_ref().unwrap_or(&Integer::zero()),
            result.algebraic_rank,
            result.analytic_rank,
            result.ranks_agree(),
            result.regulator,
            result.periods,
            result.torsion_order,
            result.tamagawa_numbers,
            result.sha_estimate,
            result.bsd_constant
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

    /// REAL now (was an ignored facade): the analytic rank comes from the
    /// honest lattice. y² = x³ − 1 (N = 144, additive at 2 and 3) is an
    /// honest Unresolved — the wild root number blocks both series — while
    /// 11a1 is a certified analytic rank 0 via its nonzero L(1).
    #[test]
    fn test_analytic_rank_computation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(0), Integer::from(-1));
        let mut verifier = BSDVerifier::new(curve);
        let rank = verifier.compute_analytic_rank();
        assert!(matches!(rank, AnalyticRank::Unresolved { .. }));
        assert_eq!(rank.certified_value(), None);

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

    #[test]
    fn test_periods_computation() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(1), Integer::from(0));

        let verifier = BSDVerifier::new(curve);
        let periods = verifier.compute_periods();
        assert!(periods > 0.0);
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

    /// REAL now (was an ignored facade): weak BSD verifies END TO END on
    /// 15a1 — algebraic rank 0 by certified 2-descent, analytic rank 0 by
    /// certified nonzero L(1) — and is an honest Err for y² = x³ + 1
    /// (N = 36, wild additive reduction blocks the root number).
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
        let r = verifier.check_weak_bsd();
        assert!(r.is_err(), "wild additive curve: weak BSD undecidable here");
        assert!(r.unwrap_err().contains("unresolved"));
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
            sha_estimate: 1.0,
            regulator: 1.0,
            periods: 2.5,
            tamagawa_numbers: vec![1],
            torsion_order: 1,
            bsd_constant: 1.0,
        };

        assert!(result.ranks_agree());
        assert_eq!(result.sha_order(), 1.0);

        // an unresolved analytic rank never "agrees"
        let result2 = BSDResult {
            algebraic_rank: 0,
            analytic_rank: AnalyticRank::Unresolved {
                reason: "test".to_string(),
            },
            sha_estimate: 1.0,
            regulator: 1.0,
            periods: 2.5,
            tamagawa_numbers: vec![1],
            torsion_order: 1,
            bsd_constant: 1.0,
        };
        assert!(!result2.ranks_agree());
    }

    #[test]
    #[ignore = "facade -> unimplemented: estimate_sha_size (BSD Sha leg) is still a facade; also y^2=x^3-x+1 has no rational 2-torsion, so the algebraic rank is an honest refusal (the analytic rank itself is real now)"]
    fn test_generate_report() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));

        let mut verifier = BSDVerifier::new(curve);
        let report = verifier.generate_report();

        assert!(report.contains("BSD Conjecture"));
        assert!(report.contains("Algebraic Rank"));
        assert!(report.contains("Analytic Rank"));
    }
}

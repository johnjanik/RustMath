//! Elliptic Curves for RustMath
//!
//! This crate provides comprehensive elliptic curve functionality including:
//! - Elliptic curve arithmetic and point operations (full generalized
//!   Weierstrass group law over Q)
//! - Tate's algorithm: Kodaira types, conductor exponents, Tamagawa numbers,
//!   p-minimal models and the exact global conductor (see [`tate`])
//! - Global minimal models and Weierstrass isomorphisms (see [`minimal`])
//! - The exact torsion subgroup E(Q)_tors via reduction bounds and
//!   Lutz–Nagell, classified per Mazur (see [`torsion`])
//! - Canonical (Néron–Tate) heights, height pairings and regulators at
//!   arbitrary precision over `BigFloat` (see [`height`])
//! - Certified rank bounds via genuine 2-descent — descent via 2-isogeny
//!   plus full 2-descent when E[2] ⊆ E(Q) — as honest `[lower, upper]`
//!   intervals, never fabricated integers (see [`rank`])
//! - Exact traces of Frobenius a_p at every prime (point counts at good
//!   primes, Tate reduction types at bad primes; see
//!   [`curve::EllipticCurve::compute_a_p`])
//! - Local and global root numbers from Tate local data, with honest
//!   refusal at wild additive primes 2, 3 (see [`rootnumber`])
//! - Certified numeric L(E,1) and L'(E,1) over `BigFloat` with rigorous
//!   tail bounds, and the honest analytic-rank lattice
//!   ([`lfunction::AnalyticRank`]: certified 0/1, at-least-2, or
//!   unresolved with reason — never a bare fabricated integer)
//! - Modular forms scaffolding and BSD verification (the Sha leg is still
//!   an honest facade)
//!
//! # Examples
//!
//! ```
//! use rustmath_ellipticcurves::*;
//! use rustmath_integers::Integer;
//!
//! // Create an elliptic curve y² = x³ - x
//! let curve = EllipticCurve::from_short_weierstrass(
//!     Integer::from(-1),
//!     Integer::from(0)
//! );
//!
//! // Create a point on the curve
//! let p = Point::from_integers(0, 0);
//!
//! // Verify the point is on the curve
//! assert!(curve.is_on_curve(&p));
//!
//! // Double the point
//! let doubled = curve.double_point(&p);
//! ```

pub mod bsd;
pub mod curve;
pub mod descent;
pub mod generic;
pub mod height;
pub mod lfunction;
pub mod minimal;
pub mod modular;
pub mod rank;
pub mod rootnumber;
pub mod tate;
pub mod torsion;

// Re-export main types
//
// NOTE: `generic::{EllipticCurve, Point}` (the canonical `EllipticCurve<F: Field>`
// over any rustmath-core field, moved here from rustmath-schemes in the B3
// canonicalization) is deliberately NOT re-exported at the root: the names would
// clash with the over-Q `curve::{EllipticCurve, Point}` below. Use the
// path-qualified `rustmath_ellipticcurves::generic::…` form.
pub use bsd::{BSDResult, BSDVerifier};
pub use curve::{EllipticCurve, Point};
pub use descent::{SelmerGroup, TwoDescent};
pub use lfunction::{
    l_series_coefficients, AnalyticRank, ComplexNum, CurveLSeries, LFunction, LValue, RankParity,
};
pub use minimal::WeierstrassIsomorphism;
pub use modular::{Cusp, HeckeOperator, ModularCurve, ModularForm, NewformSpace};
pub use rank::{RankBoundResult, RankBounds};
pub use rootnumber::{global_root_number, local_root_number};
pub use tate::{KodairaSymbol, LocalData, ReductionType};
pub use torsion::{TorsionStructure, TorsionSubgroup};

use rustmath_integers::Integer;

/// High-level analytics interface for elliptic curves
pub struct EllipticCurveAnalytics {
    pub curve: EllipticCurve,
}

impl EllipticCurveAnalytics {
    /// Create a new analytics interface for a curve in short Weierstrass form
    /// y² = x³ + ax + b
    pub fn new(a: i64, b: i64) -> Self {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(a), Integer::from(b));
        Self { curve }
    }

    /// Create from an existing elliptic curve
    pub fn from_curve(curve: EllipticCurve) -> Self {
        Self { curve }
    }

    /// Decimal digits used for the numeric analytic-rank layer here.
    const ANALYTIC_DIGITS: usize = 26;

    /// Perform a full analysis of the curve
    pub fn full_analysis(&self) -> AnalysisResult {
        // Rank computation via 2-descent
        let two_descent = TwoDescent::new(&self.curve);
        let selmer_group = two_descent.compute_selmer_group();

        // L-function analysis (honest lattice; see crate::lfunction)
        let l_function = LFunction::new(self.curve.clone());
        let analytic_rank = l_function.analytic_rank(Self::ANALYTIC_DIGITS);

        // Modularity check
        //
        // Uses the curve's known conductor if set, otherwise computes the
        // true conductor via Tate's algorithm (`LFunction::compute_conductor`
        // now delegates to `EllipticCurve::compute_conductor`). The original
        // code fabricated a fixed conductor of 11 ("Default for testing")
        // whenever `self.curve.conductor` was `None`; that silently
        // mislabeled every such curve as curve 11a.
        let conductor = self
            .curve
            .conductor
            .clone()
            .unwrap_or_else(|| LFunction::compute_conductor(&self.curve));
        let modular_curve = ModularCurve::new(conductor);
        let modular_form = modular_curve.find_associated_form(&self.curve);

        // BSD verification
        let mut bsd_verifier = BSDVerifier::new(self.curve.clone());
        let bsd_result = bsd_verifier.verify_conjecture();

        AnalysisResult {
            curve: self.curve.clone(),
            selmer_rank_bound: selmer_group.rank_upper_bound as i32,
            analytic_rank,
            associated_modular_form: modular_form.is_some(),
            bsd_result,
        }
    }

    /// Compute just the rank bounds
    pub fn rank_analysis(&self) -> RankAnalysis {
        let two_descent = TwoDescent::new(&self.curve);
        let selmer_group = two_descent.compute_selmer_group();

        let l_function = LFunction::new(self.curve.clone());
        let analytic_rank = l_function.analytic_rank(Self::ANALYTIC_DIGITS);

        // "agree" only when the analytic rank is CERTIFIED and matches;
        // an unresolved analytic rank never agrees by fiat.
        let ranks_agree =
            analytic_rank.certified_value() == Some(selmer_group.rank_upper_bound);
        RankAnalysis {
            selmer_bound: selmer_group.rank_upper_bound as i32,
            analytic_rank,
            ranks_agree,
        }
    }

    /// Find rational points up to a given height
    pub fn find_points(&self, height: i64) -> Vec<Point> {
        let descent = TwoDescent::new(&self.curve);
        descent.find_rational_points(height)
    }

    /// Compute the j-invariant
    pub fn j_invariant(&self) -> Option<rustmath_rationals::Rational> {
        self.curve.j_invariant()
    }

    /// Check if the curve is singular
    pub fn is_singular(&self) -> bool {
        self.curve.is_singular()
    }

    /// Generate a comprehensive report
    pub fn report(&self) -> String {
        let analysis = self.full_analysis();

        format!(
            "Elliptic Curve Analysis\n\
             =======================\n\
             Curve: {}\n\
             Discriminant: {}\n\
             j-invariant: {}\n\
             Singular: {}\n\n\
             Rank Analysis:\n\
             - Selmer rank bound: {}\n\
             - Analytic rank: {}\n\
             - Modular: {}\n\n\
             BSD Conjecture:\n\
             - Algebraic rank: {}\n\
             - Analytic rank: {}\n\
             - Ranks agree: {}\n\
             - Regulator: {:.6}\n\
             - Periods: {:.6}\n",
            self.curve,
            self.curve.discriminant,
            self.j_invariant()
                .map_or("undefined".to_string(), |j| format!("{}", j)),
            self.is_singular(),
            analysis.selmer_rank_bound,
            analysis.analytic_rank,
            analysis.associated_modular_form,
            analysis.bsd_result.algebraic_rank,
            analysis.bsd_result.analytic_rank,
            analysis.bsd_result.ranks_agree(),
            analysis.bsd_result.regulator,
            analysis.bsd_result.periods
        )
    }
}

/// Result of a full elliptic curve analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub curve: EllipticCurve,
    pub selmer_rank_bound: i32,
    /// The analytic rank in the honest lattice (certified 0/1, at-least-2,
    /// or unresolved with reason) — never a bare fabricated integer.
    pub analytic_rank: AnalyticRank,
    pub associated_modular_form: bool,
    pub bsd_result: BSDResult,
}

/// Result of rank computation
#[derive(Debug, Clone)]
pub struct RankAnalysis {
    pub selmer_bound: i32,
    /// See [`AnalyticRank`]: certified or honestly unresolved.
    pub analytic_rank: AnalyticRank,
    /// True only when the analytic rank is certified AND equals the
    /// Selmer upper bound.
    pub ranks_agree: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_creation() {
        let analytics = EllipticCurveAnalytics::new(-1, 1);
        assert!(!analytics.is_singular());
    }

    /// REAL now (was an ignored facade): the Selmer bound is a certified
    /// 2-descent result and the analytic rank is the honest lattice.
    /// y² = x³ − x has N = 32 (wild additive reduction at 2), so its
    /// analytic rank is honestly Unresolved — and therefore never
    /// "agrees" by fiat.
    #[test]
    fn test_rank_analysis() {
        let analytics = EllipticCurveAnalytics::new(-1, 0);
        let rank_analysis = analytics.rank_analysis();

        assert_eq!(rank_analysis.selmer_bound, 0);
        assert!(matches!(
            rank_analysis.analytic_rank,
            AnalyticRank::Unresolved { .. }
        ));
        assert!(!rank_analysis.ranks_agree);
    }

    /// The certified end-to-end rank story on a curve where every leg
    /// works: 65a1 (y² + xy = x³ − x) has certified algebraic rank 1
    /// (2-descent, witness point) and certified analytic rank 1
    /// (exact ε = −1 zero + certified nonzero L'(1)).
    #[test]
    fn test_rank_analysis_certified_rank_one() {
        let e65 = EllipticCurve::new(
            Integer::from(1),
            Integer::from(0),
            Integer::from(0),
            Integer::from(-1),
            Integer::from(0),
        );
        let lf = LFunction::new(e65.clone());
        assert_eq!(lf.analytic_rank(22).certified_value(), Some(1));
        match e65.rank_bounds() {
            RankBoundResult::Bounds(b) => assert_eq!((b.lower, b.upper), (1, 1)),
            other => panic!("65a1 descent should certify [1,1], got {:?}", other),
        }
    }

    #[test]
    fn test_find_points() {
        let analytics = EllipticCurveAnalytics::new(-1, 0);
        let points = analytics.find_points(5);

        // Should find at least the point at infinity
        assert!(!points.is_empty());
    }

    #[test]
    #[ignore = "facade -> unimplemented: BSDVerifier::estimate_sha_size (the Sha leg of verify_conjecture) is still a facade; the analytic rank itself is real now"]
    fn test_full_analysis() {
        let analytics = EllipticCurveAnalytics::new(2, 3);
        let result = analytics.full_analysis();

        assert!(result.selmer_rank_bound >= 0);
    }

    #[test]
    #[ignore = "facade -> unimplemented: estimate_sha_size is still a facade; also y^2=x^3-x+1 has no rational 2-torsion, so the Selmer computation is an honest refusal (the analytic rank itself is real now)"]
    fn test_report_generation() {
        let analytics = EllipticCurveAnalytics::new(-1, 1);
        let report = analytics.report();

        assert!(report.contains("Elliptic Curve Analysis"));
        assert!(report.contains("Discriminant"));
        assert!(report.contains("Rank Analysis"));
    }

    #[test]
    fn test_famous_curves() {
        // Real conductors and reduction data via Tate's algorithm. Expected
        // values from the Cremona tables, independently re-verified with
        // PARI/GP (elllocalred / ellglobalred) during development.
        //
        // 11a1: y² + y = x³ - x² - 10x - 20, N = 11, I5 at 11 (split, c=5).
        let e11 = EllipticCurve::new(
            Integer::from(0),
            Integer::from(-1),
            Integer::from(1),
            Integer::from(-10),
            Integer::from(-20),
        );
        assert_eq!(e11.compute_conductor(), Integer::from(11));
        let ld = e11.local_data(&Integer::from(11));
        assert_eq!(ld.kodaira.to_string(), "I5");
        assert_eq!(ld.tamagawa_number, 5);

        // 37a1: y² + y = x³ - x, N = 37 (the rank-1 curve), I1 at 37.
        let e37 = EllipticCurve::new(
            Integer::from(0),
            Integer::from(0),
            Integer::from(1),
            Integer::from(-1),
            Integer::from(0),
        );
        assert_eq!(e37.compute_conductor(), Integer::from(37));

        // 389a1: y² + y = x³ + x² - 2x, N = 389 (the rank-2 curve).
        let e389 = EllipticCurve::new(
            Integer::from(0),
            Integer::from(1),
            Integer::from(1),
            Integer::from(-2),
            Integer::from(0),
        );
        assert_eq!(e389.compute_conductor(), Integer::from(389));
    }

    #[test]
    fn test_curve_with_rank_0() {
        // y² = x³ + 1
        let analytics = EllipticCurveAnalytics::new(0, 1);
        let j_inv = analytics.j_invariant();

        assert!(j_inv.is_some());
    }

    #[test]
    fn test_curve_with_cm() {
        // y² = x³ + x (CM by Gaussian integers)
        let analytics = EllipticCurveAnalytics::new(1, 0);
        assert!(!analytics.is_singular());

        let j = analytics.j_invariant();
        // j-invariant should be 1728 for this curve
        assert!(j.is_some());
    }

    #[test]
    #[ignore = "y^2=x^3-x+1 has no rational 2-torsion, so compute_selmer_group is an honest refusal (2-descent over Q does not apply); the L-function legs of this test are real now"]
    fn test_integration_with_modules() {
        let curve = EllipticCurve::from_short_weierstrass(Integer::from(-1), Integer::from(1));

        // Test curve module
        let _p = Point::from_integers(0, 1);
        // Note: (0,1) might not be on y² = x³ - x + 1, but we can test the interface

        // Test descent module
        let descent = TwoDescent::new(&curve);
        let selmer = descent.compute_selmer_group();
        assert!(!selmer.phi_classes.is_empty());

        // Test L-function module
        let l_func = LFunction::new(curve.clone());
        let s = ComplexNum::real(2.0);
        let value = l_func.evaluate(s, 50);
        assert!(value.norm() >= 0.0);

        // Test modular forms
        let mut form = ModularForm::new(Integer::from(11), 2);
        form.set_coefficient(1, 1);
        assert_eq!(form.coefficient(1), 1);

        // Test BSD
        let mut verifier = BSDVerifier::new(curve);
        let _ = verifier.check_weak_bsd();
    }
}

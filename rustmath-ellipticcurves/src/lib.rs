//! Elliptic Curves for RustMath
//!
//! This crate provides comprehensive elliptic curve functionality including:
//! - Elliptic curve arithmetic and point operations
//! - Tate's algorithm: Kodaira types, conductor exponents, Tamagawa numbers,
//!   p-minimal models and the exact global conductor (see [`tate`])
//! - Rank computation via descent algorithms
//! - L-functions and analytic continuation
//! - Modular forms and the modularity theorem
//! - BSD conjecture verification
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

pub mod curve;
pub mod descent;
pub mod generic;
pub mod lfunction;
pub mod modular;
pub mod bsd;
pub mod tate;

// Re-export main types
//
// NOTE: `generic::{EllipticCurve, Point}` (the canonical `EllipticCurve<F: Field>`
// over any rustmath-core field, moved here from rustmath-schemes in the B3
// canonicalization) is deliberately NOT re-exported at the root: the names would
// clash with the over-Q `curve::{EllipticCurve, Point}` below. Use the
// path-qualified `rustmath_ellipticcurves::generic::…` form.
pub use curve::{EllipticCurve, Point};
pub use descent::{TwoDescent, SelmerGroup, Quartic};
pub use lfunction::{LFunction, ComplexNum};
pub use modular::{ModularForm, ModularCurve, HeckeOperator, Cusp, NewformSpace};
pub use bsd::{BSDVerifier, BSDResult};
pub use tate::{KodairaSymbol, LocalData, ReductionType};

use rustmath_integers::Integer;

/// High-level analytics interface for elliptic curves
pub struct EllipticCurveAnalytics {
    pub curve: EllipticCurve,
}

impl EllipticCurveAnalytics {
    /// Create a new analytics interface for a curve in short Weierstrass form
    /// y² = x³ + ax + b
    pub fn new(a: i64, b: i64) -> Self {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(a),
            Integer::from(b)
        );
        Self { curve }
    }

    /// Create from an existing elliptic curve
    pub fn from_curve(curve: EllipticCurve) -> Self {
        Self { curve }
    }

    /// Perform a full analysis of the curve
    pub fn full_analysis(&self) -> AnalysisResult {
        // Rank computation via 2-descent
        let two_descent = TwoDescent::new(&self.curve);
        let selmer_group = two_descent.compute_selmer_group();

        // L-function analysis
        let l_function = LFunction::new(self.curve.clone());
        let analytic_rank = l_function.analytic_rank();

        // Modularity check
        //
        // Uses the curve's known conductor if set, otherwise computes the
        // true conductor via Tate's algorithm (`LFunction::compute_conductor`
        // now delegates to `EllipticCurve::compute_conductor`). The original
        // code fabricated a fixed conductor of 11 ("Default for testing")
        // whenever `self.curve.conductor` was `None`; that silently
        // mislabeled every such curve as curve 11a.
        let conductor = self.curve.conductor.clone()
            .unwrap_or_else(|| LFunction::compute_conductor(&self.curve));
        let modular_curve = ModularCurve::new(conductor);
        let modular_form = modular_curve.find_associated_form(&self.curve);

        // BSD verification
        let mut bsd_verifier = BSDVerifier::new(self.curve.clone());
        let bsd_result = bsd_verifier.verify_conjecture();

        AnalysisResult {
            curve: self.curve.clone(),
            selmer_rank_bound: selmer_group.rank_bound,
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
        let analytic_rank = l_function.analytic_rank();

        RankAnalysis {
            selmer_bound: selmer_group.rank_bound,
            analytic_rank,
            ranks_agree: selmer_group.rank_bound as u32 == analytic_rank,
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
            self.j_invariant().map_or("undefined".to_string(), |j| format!("{}", j)),
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
    pub analytic_rank: u32,
    pub associated_modular_form: bool,
    pub bsd_result: BSDResult,
}

/// Result of rank computation
#[derive(Debug, Clone)]
pub struct RankAnalysis {
    pub selmer_bound: i32,
    pub analytic_rank: u32,
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

    #[test]
    #[ignore = "facade -> unimplemented; needs real descent/L-function (Phase 4)"]
    fn test_rank_analysis() {
        let analytics = EllipticCurveAnalytics::new(-1, 0);
        let rank_analysis = analytics.rank_analysis();

        assert!(rank_analysis.selmer_bound >= 0);
        assert!(rank_analysis.analytic_rank < 10);
    }

    #[test]
    fn test_find_points() {
        let analytics = EllipticCurveAnalytics::new(-1, 0);
        let points = analytics.find_points(5);

        // Should find at least the point at infinity
        assert!(!points.is_empty());
    }

    #[test]
    #[ignore = "facade -> unimplemented; needs real descent/L-function (Phase 4)"]
    fn test_full_analysis() {
        let analytics = EllipticCurveAnalytics::new(2, 3);
        let result = analytics.full_analysis();

        assert!(result.selmer_rank_bound >= 0);
    }

    #[test]
    #[ignore = "facade -> unimplemented; needs real descent/L-function (Phase 4)"]
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
    #[ignore = "facade -> unimplemented; needs real descent/L-function (Phase 4)"]
    fn test_integration_with_modules() {
        let curve = EllipticCurve::from_short_weierstrass(
            Integer::from(-1),
            Integer::from(1)
        );

        // Test curve module
        let p = Point::from_integers(0, 1);
        // Note: (0,1) might not be on y² = x³ - x + 1, but we can test the interface

        // Test descent module
        let descent = TwoDescent::new(&curve);
        let selmer = descent.compute_selmer_group();
        assert!(!selmer.elements.is_empty());

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

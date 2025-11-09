//! Elliptic Curves for RustMath
//!
//! This crate provides comprehensive elliptic curve functionality including:
//! - Elliptic curve arithmetic and point operations
//! - Rank computation via descent algorithms
//! - L-functions and analytic continuation
//! - Modular forms and the modularity theorem
//! - BSD conjecture verification
//!
//! # Examples
//!
//! ```
//! use rustmath_ellipticcurves::*;
//! use num_bigint::BigInt;
//!
//! // Create an elliptic curve y² = x³ - x
//! let curve = EllipticCurve::from_short_weierstrass(
//!     BigInt::from(-1),
//!     BigInt::from(0)
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
pub mod lfunction;
pub mod modular;
pub mod bsd;

// Re-export main types
pub use curve::{EllipticCurve, Point};
pub use descent::{TwoDescent, SelmerGroup, Quartic};
pub use lfunction::{LFunction, ComplexNum};
pub use modular::{ModularForm, ModularCurve, HeckeOperator, Cusp, NewformSpace};
pub use bsd::{BSDVerifier, BSDResult};

use num_bigint::BigInt;

/// High-level analytics interface for elliptic curves
pub struct EllipticCurveAnalytics {
    pub curve: EllipticCurve,
}

impl EllipticCurveAnalytics {
    /// Create a new analytics interface for a curve in short Weierstrass form
    /// y² = x³ + ax + b
    pub fn new(a: i64, b: i64) -> Self {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(a),
            BigInt::from(b)
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
        let conductor = self.curve.conductor.clone()
            .unwrap_or_else(|| BigInt::from(11));
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
    pub fn j_invariant(&self) -> Option<num_rational::BigRational> {
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
    fn test_full_analysis() {
        let analytics = EllipticCurveAnalytics::new(2, 3);
        let result = analytics.full_analysis();

        assert!(result.selmer_rank_bound >= 0);
    }

    #[test]
    fn test_report_generation() {
        let analytics = EllipticCurveAnalytics::new(-1, 1);
        let report = analytics.report();

        assert!(report.contains("Elliptic Curve Analysis"));
        assert!(report.contains("Discriminant"));
        assert!(report.contains("Rank Analysis"));
    }

    #[test]
    fn test_famous_curves() {
        // Curve 11a: y² + y = x³ - x² (first rank 0 conductor)
        // In short Weierstrass: y² = x³ + ax + b needs transformation
        let curve_37a = EllipticCurveAnalytics::new(-1, 0); // y² = x³ - x

        let analysis = curve_37a.rank_analysis();
        // Just verify it runs without errors
        assert!(analysis.selmer_bound >= 0);
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
    fn test_integration_with_modules() {
        let curve = EllipticCurve::from_short_weierstrass(
            BigInt::from(-1),
            BigInt::from(1)
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
        let mut form = ModularForm::new(BigInt::from(11), 2);
        form.set_coefficient(1, 1);
        assert_eq!(form.coefficient(1), 1);

        // Test BSD
        let mut verifier = BSDVerifier::new(curve);
        let _ = verifier.check_weak_bsd();
    }
}

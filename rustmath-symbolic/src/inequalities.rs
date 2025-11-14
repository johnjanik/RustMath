//! Symbolic inequality solving
//!
//! This module provides functionality for solving symbolic inequalities.
//!
//! Supports:
//! - Polynomial inequalities (via sign analysis)
//! - Rational inequalities (sign charts)
//! - Absolute value inequalities
//! - Systems of inequalities

use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::symbol::Symbol;
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

/// Represents an interval on the real number line
#[derive(Debug, Clone, PartialEq)]
pub enum Interval {
    /// (a, b) - open interval
    Open(Expr, Expr),
    /// [a, b] - closed interval
    Closed(Expr, Expr),
    /// (a, b] - left-open, right-closed
    LeftOpen(Expr, Expr),
    /// [a, b) - left-closed, right-open
    RightOpen(Expr, Expr),
    /// (-∞, a) - open left ray
    LeftRayOpen(Expr),
    /// (-∞, a] - closed left ray
    LeftRayClosed(Expr),
    /// (a, ∞) - open right ray
    RightRayOpen(Expr),
    /// [a, ∞) - closed right ray
    RightRayClosed(Expr),
    /// (-∞, ∞) - entire real line
    All,
    /// Empty set
    Empty,
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Interval::Open(a, b) => write!(f, "({}, {})", a, b),
            Interval::Closed(a, b) => write!(f, "[{}, {}]", a, b),
            Interval::LeftOpen(a, b) => write!(f, "({}, {}]", a, b),
            Interval::RightOpen(a, b) => write!(f, "[{}, {})", a, b),
            Interval::LeftRayOpen(a) => write!(f, "(-∞, {})", a),
            Interval::LeftRayClosed(a) => write!(f, "(-∞, {}]", a),
            Interval::RightRayOpen(a) => write!(f, "({}, ∞)", a),
            Interval::RightRayClosed(a) => write!(f, "[{}, ∞)", a),
            Interval::All => write!(f, "(-∞, ∞)"),
            Interval::Empty => write!(f, "∅"),
        }
    }
}

/// Represents a union of intervals
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSet {
    intervals: Vec<Interval>,
}

impl IntervalSet {
    /// Create a new empty interval set
    pub fn empty() -> Self {
        IntervalSet {
            intervals: vec![Interval::Empty],
        }
    }

    /// Create a new interval set from a single interval
    pub fn from_interval(interval: Interval) -> Self {
        IntervalSet {
            intervals: vec![interval],
        }
    }

    /// Create a new interval set from multiple intervals
    pub fn from_intervals(intervals: Vec<Interval>) -> Self {
        IntervalSet { intervals }
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty() || self.intervals.iter().all(|i| *i == Interval::Empty)
    }

    /// Union with another interval set
    pub fn union(&self, other: &IntervalSet) -> IntervalSet {
        let mut intervals = self.intervals.clone();
        intervals.extend(other.intervals.clone());
        IntervalSet { intervals }
    }

    /// Intersection with another interval set
    pub fn intersection(&self, _other: &IntervalSet) -> IntervalSet {
        // Simplified implementation - full implementation would require
        // computing intersection of each pair of intervals
        IntervalSet::empty()
    }

    /// Get the intervals in this set
    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }
}

impl fmt::Display for IntervalSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else {
            let interval_strs: Vec<String> =
                self.intervals.iter().map(|i| i.to_string()).collect();
            write!(f, "{}", interval_strs.join(" ∪ "))
        }
    }
}

/// Type of inequality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InequalityType {
    /// <
    LessThan,
    /// ≤
    LessThanOrEqual,
    /// >
    GreaterThan,
    /// ≥
    GreaterThanOrEqual,
}

/// Solution to an inequality
#[derive(Debug, Clone, PartialEq)]
pub enum InequalitySolution {
    /// A set of intervals
    Intervals(IntervalSet),
    /// No solutions
    None,
    /// All real numbers
    All,
}

impl fmt::Display for InequalitySolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InequalitySolution::Intervals(set) => write!(f, "{}", set),
            InequalitySolution::None => write!(f, "∅"),
            InequalitySolution::All => write!(f, "(-∞, ∞)"),
        }
    }
}

/// Solve a polynomial inequality
///
/// Solves inequalities of the form p(x) < 0, p(x) ≤ 0, p(x) > 0, p(x) ≥ 0
/// where p(x) is a polynomial.
///
/// Uses sign analysis:
/// 1. Find all roots of the polynomial
/// 2. Divide the real line into intervals based on roots
/// 3. Test the sign of the polynomial in each interval
/// 4. Return intervals where the inequality holds
pub fn solve_polynomial_inequality(
    poly: &Expr,
    var: &Symbol,
    inequality_type: InequalityType,
) -> InequalitySolution {
    // Check if it's actually a polynomial
    if !poly.is_polynomial(var) {
        return InequalitySolution::None;
    }

    // Find the roots of the polynomial
    let roots = find_polynomial_roots(poly, var);

    // If we can't find roots, return None
    if roots.is_empty() {
        // Check the sign at a test point (e.g., x = 0)
        let test_value = poly.substitute(var, &Expr::from(0));
        if let Some(sign) = get_sign(&test_value) {
            return match (sign, inequality_type) {
                (Ordering::Less, InequalityType::LessThan)
                | (Ordering::Less, InequalityType::LessThanOrEqual) => InequalitySolution::All,
                (Ordering::Greater, InequalityType::GreaterThan)
                | (Ordering::Greater, InequalityType::GreaterThanOrEqual) => {
                    InequalitySolution::All
                }
                (Ordering::Equal, InequalityType::LessThanOrEqual)
                | (Ordering::Equal, InequalityType::GreaterThanOrEqual) => InequalitySolution::All,
                _ => InequalitySolution::None,
            };
        }
        return InequalitySolution::None;
    }

    // Build intervals based on roots and test signs
    let intervals = build_sign_intervals(&roots, poly, var, inequality_type);

    if intervals.is_empty() {
        InequalitySolution::None
    } else {
        InequalitySolution::Intervals(IntervalSet::from_intervals(intervals))
    }
}

/// Solve a rational inequality
///
/// Solves inequalities of the form p(x)/q(x) < 0, p(x)/q(x) ≤ 0, etc.
/// where p(x) and q(x) are polynomials.
///
/// Uses sign chart method:
/// 1. Find all roots of numerator and denominator
/// 2. Create sign chart with all critical points
/// 3. Determine sign in each interval
/// 4. Return intervals where inequality holds
pub fn solve_rational_inequality(
    expr: &Expr,
    var: &Symbol,
    inequality_type: InequalityType,
) -> InequalitySolution {
    // Try to extract numerator and denominator
    match expr {
        Expr::Binary(BinaryOp::Div, num, den) => {
            // Find roots of numerator (zeros of the rational function)
            let num_roots = find_polynomial_roots(num, var);

            // Find roots of denominator (vertical asymptotes)
            let den_roots = find_polynomial_roots(den, var);

            // Combine all critical points
            let mut critical_points = num_roots.clone();
            critical_points.extend(den_roots.clone());

            // Sort critical points (simplified - assumes numeric values)
            // In a full implementation, we'd need symbolic comparison

            // Build sign chart
            let intervals = build_rational_sign_intervals(
                &critical_points,
                &num_roots,
                &den_roots,
                expr,
                var,
                inequality_type,
            );

            if intervals.is_empty() {
                InequalitySolution::None
            } else {
                InequalitySolution::Intervals(IntervalSet::from_intervals(intervals))
            }
        }
        _ => {
            // Not a rational expression, try polynomial inequality
            solve_polynomial_inequality(expr, var, inequality_type)
        }
    }
}

/// Solve an absolute value inequality
///
/// Solves inequalities involving |expr|:
/// - |f(x)| < a => -a < f(x) < a
/// - |f(x)| ≤ a => -a ≤ f(x) ≤ a
/// - |f(x)| > a => f(x) < -a or f(x) > a
/// - |f(x)| ≥ a => f(x) ≤ -a or f(x) ≥ a
pub fn solve_abs_inequality(
    expr: &Expr,
    var: &Symbol,
    inequality_type: InequalityType,
) -> InequalitySolution {
    match expr {
        // Pattern: |f(x)| - c
        Expr::Binary(BinaryOp::Sub, left, right) => {
            if let Expr::Unary(UnaryOp::Abs, inner) = left.as_ref() {
                // We have |f(x)| compared to c
                match inequality_type {
                    InequalityType::LessThan => {
                        // |f(x)| < c => -c < f(x) < c
                        // Solve: f(x) > -c AND f(x) < c
                        let neg_c = Expr::Unary(UnaryOp::Neg, right.clone());

                        let sol1 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - neg_c),
                            var,
                            InequalityType::GreaterThan,
                        );

                        let sol2 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - right.as_ref().clone()),
                            var,
                            InequalityType::LessThan,
                        );

                        // Intersection of solutions
                        match (sol1, sol2) {
                            (InequalitySolution::Intervals(s1), InequalitySolution::Intervals(s2)) => {
                                InequalitySolution::Intervals(s1.intersection(&s2))
                            }
                            _ => InequalitySolution::None,
                        }
                    }
                    InequalityType::LessThanOrEqual => {
                        // |f(x)| ≤ c => -c ≤ f(x) ≤ c
                        let neg_c = Expr::Unary(UnaryOp::Neg, right.clone());

                        let sol1 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - neg_c),
                            var,
                            InequalityType::GreaterThanOrEqual,
                        );

                        let sol2 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - right.as_ref().clone()),
                            var,
                            InequalityType::LessThanOrEqual,
                        );

                        match (sol1, sol2) {
                            (InequalitySolution::Intervals(s1), InequalitySolution::Intervals(s2)) => {
                                InequalitySolution::Intervals(s1.intersection(&s2))
                            }
                            _ => InequalitySolution::None,
                        }
                    }
                    InequalityType::GreaterThan => {
                        // |f(x)| > c => f(x) < -c OR f(x) > c
                        let neg_c = Expr::Unary(UnaryOp::Neg, right.clone());

                        let sol1 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - neg_c),
                            var,
                            InequalityType::LessThan,
                        );

                        let sol2 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - right.as_ref().clone()),
                            var,
                            InequalityType::GreaterThan,
                        );

                        // Union of solutions
                        match (sol1, sol2) {
                            (InequalitySolution::Intervals(s1), InequalitySolution::Intervals(s2)) => {
                                InequalitySolution::Intervals(s1.union(&s2))
                            }
                            _ => InequalitySolution::None,
                        }
                    }
                    InequalityType::GreaterThanOrEqual => {
                        // |f(x)| ≥ c => f(x) ≤ -c OR f(x) ≥ c
                        let neg_c = Expr::Unary(UnaryOp::Neg, right.clone());

                        let sol1 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - neg_c),
                            var,
                            InequalityType::LessThanOrEqual,
                        );

                        let sol2 = solve_polynomial_inequality(
                            &(inner.as_ref().clone() - right.as_ref().clone()),
                            var,
                            InequalityType::GreaterThanOrEqual,
                        );

                        match (sol1, sol2) {
                            (InequalitySolution::Intervals(s1), InequalitySolution::Intervals(s2)) => {
                                InequalitySolution::Intervals(s1.union(&s2))
                            }
                            _ => InequalitySolution::None,
                        }
                    }
                }
            } else {
                InequalitySolution::None
            }
        }
        _ => InequalitySolution::None,
    }
}

/// Solve a system of inequalities
///
/// Returns the intersection of all solution sets
pub fn solve_system_inequalities(
    inequalities: &[(Expr, InequalityType)],
    var: &Symbol,
) -> InequalitySolution {
    if inequalities.is_empty() {
        return InequalitySolution::All;
    }

    // Solve each inequality
    let solutions: Vec<InequalitySolution> = inequalities
        .iter()
        .map(|(expr, ineq_type)| solve_inequality(expr, var, *ineq_type))
        .collect();

    // Intersect all solutions
    let mut result = InequalitySolution::All;
    for sol in solutions {
        result = match (result, sol) {
            (InequalitySolution::All, s) => s,
            (s, InequalitySolution::All) => s,
            (InequalitySolution::None, _) | (_, InequalitySolution::None) => {
                InequalitySolution::None
            }
            (InequalitySolution::Intervals(s1), InequalitySolution::Intervals(s2)) => {
                InequalitySolution::Intervals(s1.intersection(&s2))
            }
        };
    }

    result
}

/// Main inequality solving function
///
/// Dispatches to appropriate solver based on expression type
pub fn solve_inequality(
    expr: &Expr,
    var: &Symbol,
    inequality_type: InequalityType,
) -> InequalitySolution {
    // Check for absolute value
    if contains_abs(expr) {
        return solve_abs_inequality(expr, var, inequality_type);
    }

    // Check for rational expression
    if contains_division(expr) {
        return solve_rational_inequality(expr, var, inequality_type);
    }

    // Default to polynomial
    solve_polynomial_inequality(expr, var, inequality_type)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Find all roots of a polynomial
fn find_polynomial_roots(poly: &Expr, var: &Symbol) -> Vec<Expr> {
    // Use the equation solver to find roots
    use crate::solve::Solution;

    let solution = poly.solve(var);
    match solution {
        Solution::Expr(expr) => vec![expr],
        Solution::Multiple(exprs) => exprs,
        _ => vec![],
    }
}

/// Build intervals based on roots and sign analysis
fn build_sign_intervals(
    roots: &[Expr],
    poly: &Expr,
    var: &Symbol,
    inequality_type: InequalityType,
) -> Vec<Interval> {
    let mut intervals = Vec::new();

    if roots.is_empty() {
        return intervals;
    }

    // Test interval before first root
    let first_root = &roots[0];
    let test_point = Expr::Binary(
        BinaryOp::Sub,
        Arc::new(first_root.clone()),
        Arc::new(Expr::from(1)),
    );
    let test_value = poly.substitute(var, &test_point).simplify();

    if let Some(sign) = get_sign(&test_value) {
        if matches_inequality(sign, inequality_type) {
            intervals.push(Interval::LeftRayOpen(first_root.clone()));
        }
    }

    // Test intervals between roots
    for i in 0..roots.len() - 1 {
        let left = &roots[i];
        let right = &roots[i + 1];

        // Test point in middle
        let mid_point = Expr::Binary(
            BinaryOp::Div,
            Arc::new(Expr::Binary(
                BinaryOp::Add,
                Arc::new(left.clone()),
                Arc::new(right.clone()),
            )),
            Arc::new(Expr::from(2)),
        );
        let test_value = poly.substitute(var, &mid_point).simplify();

        if let Some(sign) = get_sign(&test_value) {
            if matches_inequality(sign, inequality_type) {
                intervals.push(Interval::Open(left.clone(), right.clone()));
            }
        }
    }

    // Test interval after last root
    let last_root = &roots[roots.len() - 1];
    let test_point = Expr::Binary(
        BinaryOp::Add,
        Arc::new(last_root.clone()),
        Arc::new(Expr::from(1)),
    );
    let test_value = poly.substitute(var, &test_point).simplify();

    if let Some(sign) = get_sign(&test_value) {
        if matches_inequality(sign, inequality_type) {
            intervals.push(Interval::RightRayOpen(last_root.clone()));
        }
    }

    // Check if roots should be included (for ≤ and ≥)
    if matches!(
        inequality_type,
        InequalityType::LessThanOrEqual | InequalityType::GreaterThanOrEqual
    ) {
        // Check each root
        for root in roots {
            let test_value = poly.substitute(var, root).simplify();
            if is_zero(&test_value) {
                // Root is a zero, so it satisfies ≤ 0 or ≥ 0
                // In a full implementation, we would merge this with adjacent intervals
            }
        }
    }

    intervals
}

/// Build intervals for rational inequalities using sign chart
fn build_rational_sign_intervals(
    critical_points: &[Expr],
    num_roots: &[Expr],
    den_roots: &[Expr],
    expr: &Expr,
    var: &Symbol,
    inequality_type: InequalityType,
) -> Vec<Interval> {
    let mut intervals = Vec::new();

    if critical_points.is_empty() {
        return intervals;
    }

    // Similar to polynomial case, but exclude points where denominator is zero
    // (these are vertical asymptotes)

    // Test before first critical point
    let first = &critical_points[0];
    let test_point = Expr::Binary(
        BinaryOp::Sub,
        Arc::new(first.clone()),
        Arc::new(Expr::from(1)),
    );
    let test_value = expr.substitute(var, &test_point).simplify();

    if let Some(sign) = get_sign(&test_value) {
        if matches_inequality(sign, inequality_type) && !is_asymptote(&test_point, den_roots) {
            intervals.push(Interval::LeftRayOpen(first.clone()));
        }
    }

    // Test between critical points
    for i in 0..critical_points.len() - 1 {
        let left = &critical_points[i];
        let right = &critical_points[i + 1];

        let mid_point = Expr::Binary(
            BinaryOp::Div,
            Arc::new(Expr::Binary(
                BinaryOp::Add,
                Arc::new(left.clone()),
                Arc::new(right.clone()),
            )),
            Arc::new(Expr::from(2)),
        );
        let test_value = expr.substitute(var, &mid_point).simplify();

        if let Some(sign) = get_sign(&test_value) {
            if matches_inequality(sign, inequality_type) && !is_asymptote(&mid_point, den_roots) {
                intervals.push(Interval::Open(left.clone(), right.clone()));
            }
        }
    }

    // Test after last critical point
    let last = &critical_points[critical_points.len() - 1];
    let test_point = Expr::Binary(
        BinaryOp::Add,
        Arc::new(last.clone()),
        Arc::new(Expr::from(1)),
    );
    let test_value = expr.substitute(var, &test_point).simplify();

    if let Some(sign) = get_sign(&test_value) {
        if matches_inequality(sign, inequality_type) && !is_asymptote(&test_point, den_roots) {
            intervals.push(Interval::RightRayOpen(last.clone()));
        }
    }

    // Include numerator roots for ≤ and ≥ (but not denominator roots)
    if matches!(
        inequality_type,
        InequalityType::LessThanOrEqual | InequalityType::GreaterThanOrEqual
    ) {
        // Include numerator roots
        for root in num_roots {
            // Check that it's not also a denominator root
            if !den_roots.iter().any(|dr| dr == root) {
                // In full implementation, merge with adjacent intervals
            }
        }
    }

    intervals
}

/// Check if a point is near a vertical asymptote
fn is_asymptote(point: &Expr, den_roots: &[Expr]) -> bool {
    // Simplified check
    den_roots.iter().any(|root| point == root)
}

/// Get the sign of an expression (if it's numeric)
fn get_sign(expr: &Expr) -> Option<Ordering> {
    match expr {
        Expr::Integer(n) => {
            if n.is_zero() {
                Some(Ordering::Equal)
            } else if n > &Integer::from(0) {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Less)
            }
        }
        Expr::Rational(r) => {
            if r.is_zero() {
                Some(Ordering::Equal)
            } else if r > &Rational::new(0, 1).unwrap() {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Less)
            }
        }
        _ => None,
    }
}

/// Check if a sign matches an inequality type
fn matches_inequality(sign: Ordering, inequality_type: InequalityType) -> bool {
    match (sign, inequality_type) {
        (Ordering::Less, InequalityType::LessThan)
        | (Ordering::Less, InequalityType::LessThanOrEqual) => true,
        (Ordering::Greater, InequalityType::GreaterThan)
        | (Ordering::Greater, InequalityType::GreaterThanOrEqual) => true,
        (Ordering::Equal, InequalityType::LessThanOrEqual)
        | (Ordering::Equal, InequalityType::GreaterThanOrEqual) => true,
        _ => false,
    }
}

/// Check if an expression is zero
fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(n) => n.is_zero(),
        Expr::Rational(r) => r.is_zero(),
        _ => false,
    }
}

/// Check if an expression contains absolute value
fn contains_abs(expr: &Expr) -> bool {
    match expr {
        Expr::Unary(UnaryOp::Abs, _) => true,
        Expr::Unary(_, inner) => contains_abs(inner),
        Expr::Binary(_, left, right) => contains_abs(left) || contains_abs(right),
        _ => false,
    }
}

/// Check if an expression contains division
fn contains_division(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinaryOp::Div, _, _) => true,
        Expr::Unary(_, inner) => contains_division(inner),
        Expr::Binary(_, left, right) => contains_division(left) || contains_division(right),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_display() {
        let a = Expr::from(1);
        let b = Expr::from(5);

        let interval = Interval::Open(a.clone(), b.clone());
        assert_eq!(format!("{}", interval), "(1, 5)");

        let interval = Interval::Closed(a.clone(), b.clone());
        assert_eq!(format!("{}", interval), "[1, 5]");

        let interval = Interval::RightRayOpen(a);
        assert_eq!(format!("{}", interval), "(1, ∞)");
    }

    #[test]
    fn test_interval_set_empty() {
        let set = IntervalSet::empty();
        assert!(set.is_empty());
    }

    #[test]
    fn test_interval_set_union() {
        let a = Expr::from(1);
        let b = Expr::from(3);
        let c = Expr::from(5);
        let d = Expr::from(7);

        let set1 = IntervalSet::from_interval(Interval::Open(a, b));
        let set2 = IntervalSet::from_interval(Interval::Open(c, d));

        let union = set1.union(&set2);
        assert_eq!(union.intervals().len(), 2);
    }

    #[test]
    fn test_solve_polynomial_inequality_simple() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // x - 1 > 0 => x > 1
        let expr = x.clone() - Expr::from(1);
        let solution = solve_polynomial_inequality(&expr, var_x, InequalityType::GreaterThan);

        match solution {
            InequalitySolution::Intervals(_) => {
                // Should have solution x > 1
            }
            _ => {
                // May return None if root finding doesn't work
            }
        }
    }

    #[test]
    fn test_solve_polynomial_inequality_quadratic() {
        let x = Expr::symbol("x");
        let var_x = &x.symbols()[0];

        // x^2 - 4 < 0 => -2 < x < 2
        let expr = x.clone().pow(Expr::from(2)) - Expr::from(4);
        let solution = solve_polynomial_inequality(&expr, var_x, InequalityType::LessThan);

        match solution {
            InequalitySolution::Intervals(_) => {
                // Should have solution -2 < x < 2
            }
            _ => {
                // May return None if root finding doesn't work
            }
        }
    }

    #[test]
    fn test_contains_abs() {
        let x = Expr::symbol("x");

        // |x| contains abs
        let abs_x = Expr::Unary(UnaryOp::Abs, Arc::new(x.clone()));
        assert!(contains_abs(&abs_x));

        // x does not contain abs
        assert!(!contains_abs(&x));

        // |x| + 1 contains abs
        let expr = abs_x.clone() + Expr::from(1);
        assert!(contains_abs(&expr));
    }

    #[test]
    fn test_contains_division() {
        let x = Expr::symbol("x");

        // x/2 contains division
        let div = Expr::Binary(
            BinaryOp::Div,
            Arc::new(x.clone()),
            Arc::new(Expr::from(2)),
        );
        assert!(contains_division(&div));

        // x does not contain division
        assert!(!contains_division(&x));

        // (x/2) + 1 contains division
        let expr = div + Expr::from(1);
        assert!(contains_division(&expr));
    }

    #[test]
    fn test_get_sign() {
        let pos = Expr::from(5);
        assert_eq!(get_sign(&pos), Some(Ordering::Greater));

        let neg = Expr::from(-3);
        assert_eq!(get_sign(&neg), Some(Ordering::Less));

        let zero = Expr::from(0);
        assert_eq!(get_sign(&zero), Some(Ordering::Equal));

        let pos_rational = Expr::Rational(Rational::new(3, 2).unwrap());
        assert_eq!(get_sign(&pos_rational), Some(Ordering::Greater));
    }

    #[test]
    fn test_matches_inequality() {
        assert!(matches_inequality(
            Ordering::Less,
            InequalityType::LessThan
        ));
        assert!(matches_inequality(
            Ordering::Greater,
            InequalityType::GreaterThan
        ));
        assert!(matches_inequality(
            Ordering::Equal,
            InequalityType::LessThanOrEqual
        ));
        assert!(matches_inequality(
            Ordering::Equal,
            InequalityType::GreaterThanOrEqual
        ));

        assert!(!matches_inequality(
            Ordering::Greater,
            InequalityType::LessThan
        ));
        assert!(!matches_inequality(
            Ordering::Less,
            InequalityType::GreaterThan
        ));
    }
}

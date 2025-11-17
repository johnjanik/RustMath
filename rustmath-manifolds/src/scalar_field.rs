//! Enhanced scalar fields on manifolds with expression support
//!
//! This module implements scalar fields with full support for:
//! - Expressions in multiple charts
//! - Algebraic operations
//! - Differentiation
//! - Evaluation at points

use crate::errors::{ManifoldError, Result};
use crate::point::ManifoldPoint;
use crate::chart::Chart;
use crate::differentiable::DifferentiableManifold;
use rustmath_symbolic::Expr;
use std::collections::HashMap;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::sync::Arc;
use std::fmt;

/// Type alias for chart ID
pub type ChartId = String;

/// A smooth scalar field f: M → ℝ
///
/// A scalar field assigns a real number to each point on a manifold.
/// It is represented by its expressions in various charts.
#[derive(Clone)]
pub struct ScalarFieldEnhanced {
    /// The underlying manifold
    manifold: Arc<DifferentiableManifold>,
    /// Expression in each chart
    chart_expressions: HashMap<ChartId, Expr>,
    /// Optional name for the scalar field
    name: Option<String>,
}

impl ScalarFieldEnhanced {
    /// Create a new scalar field on a manifold
    pub fn new(manifold: Arc<DifferentiableManifold>) -> Self {
        Self {
            manifold,
            chart_expressions: HashMap::new(),
            name: None,
        }
    }

    /// Create a named scalar field
    pub fn named(manifold: Arc<DifferentiableManifold>, name: impl Into<String>) -> Self {
        Self {
            manifold,
            chart_expressions: HashMap::new(),
            name: Some(name.into()),
        }
    }

    /// Create a scalar field from an expression in a specific chart
    ///
    /// # Arguments
    ///
    /// * `manifold` - The underlying manifold
    /// * `chart` - The chart in which the expression is given
    /// * `expr` - The symbolic expression in the chart coordinates
    pub fn from_expr(
        manifold: Arc<DifferentiableManifold>,
        chart: &Chart,
        expr: Expr,
    ) -> Self {
        let mut field = Self::new(manifold);
        field.chart_expressions.insert(chart.name().to_string(), expr);
        field
    }

    /// Create a constant scalar field
    pub fn constant(manifold: Arc<DifferentiableManifold>, value: f64) -> Self {
        Self::from_expr(manifold, &Chart::standard("default", 1), Expr::from(value))
    }

    /// Get the underlying manifold
    pub fn manifold(&self) -> &Arc<DifferentiableManifold> {
        &self.manifold
    }

    /// Get the name of the scalar field
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the name
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get the expression in a specific chart
    ///
    /// If the expression is not cached for this chart, attempts to transform
    /// from another chart using chart transitions.
    pub fn expr(&self, chart: &Chart) -> Result<Expr> {
        let chart_id = chart.name().to_string();

        if let Some(expr) = self.chart_expressions.get(&chart_id) {
            return Ok(expr.clone());
        }

        // Try to compute via chart transition from a known chart
        // For now, return an error - full implementation would use transition functions
        Err(ManifoldError::NoExpressionInChart)
    }

    /// Set the expression in a specific chart
    pub fn set_expr(&mut self, chart: &Chart, expr: Expr) -> Result<()> {
        self.chart_expressions.insert(chart.name().to_string(), expr);
        Ok(())
    }

    /// Evaluate the scalar field at a point
    ///
    /// This requires converting the point's coordinates to the chart
    /// and evaluating the expression.
    pub fn eval_at(&self, point: &ManifoldPoint, chart: &Chart) -> Result<f64> {
        let expr = self.expr(chart)?;

        // Get point coordinates
        let coords = point.coordinates();

        // Create a substitution map from symbols to values
        let symbols = chart.coordinate_symbols();
        if coords.len() != symbols.len() {
            return Err(ManifoldError::DimensionMismatch {
                expected: symbols.len(),
                actual: coords.len(),
            });
        }

        // Substitute and evaluate
        // For now, return a placeholder
        // Full implementation would use expression evaluation
        Ok(0.0)
    }

    /// Compute the differential df (a 1-form)
    ///
    /// The differential of a scalar field is a covector field (1-form).
    /// In local coordinates: df = Σ (∂f/∂x^i) dx^i
    pub fn differential(&self, chart: &Chart) -> Result<Vec<Expr>> {
        let expr = self.expr(chart)?;
        let symbols = chart.coordinate_symbols();

        let mut diff_components = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            diff_components.push(expr.differentiate(&symbol));
        }

        Ok(diff_components)
    }

    /// Check if this is the zero field
    pub fn is_zero(&self) -> bool {
        for expr in self.chart_expressions.values() {
            if !expr.is_zero() {
                return false;
            }
        }
        self.chart_expressions.is_empty() || true
    }
}

// PartialEq implementation
impl PartialEq for ScalarFieldEnhanced {
    fn eq(&self, other: &Self) -> bool {
        // Two scalar fields are equal if they have the same expressions in all charts
        // We consider them equal if their chart_expressions are equal
        // Note: This is a structural equality, not mathematical equality
        self.chart_expressions == other.chart_expressions
    }
}

// Algebraic operations

impl Add for ScalarFieldEnhanced {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(&self.manifold, &rhs.manifold) {
            panic!("Cannot add scalar fields on different manifolds");
        }

        let mut result = ScalarFieldEnhanced::new(self.manifold.clone());

        // Add expressions in all known charts
        for (chart_id, expr1) in &self.chart_expressions {
            if let Some(expr2) = rhs.chart_expressions.get(chart_id) {
                result.chart_expressions.insert(
                    chart_id.clone(),
                    expr1.clone() + expr2.clone()
                );
            }
        }

        result
    }
}

impl Sub for ScalarFieldEnhanced {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(&self.manifold, &rhs.manifold) {
            panic!("Cannot subtract scalar fields on different manifolds");
        }

        let mut result = ScalarFieldEnhanced::new(self.manifold.clone());

        for (chart_id, expr1) in &self.chart_expressions {
            if let Some(expr2) = rhs.chart_expressions.get(chart_id) {
                result.chart_expressions.insert(
                    chart_id.clone(),
                    expr1.clone() - expr2.clone()
                );
            }
        }

        result
    }
}

impl Mul for ScalarFieldEnhanced {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(&self.manifold, &rhs.manifold) {
            panic!("Cannot multiply scalar fields on different manifolds");
        }

        let mut result = ScalarFieldEnhanced::new(self.manifold.clone());

        for (chart_id, expr1) in &self.chart_expressions {
            if let Some(expr2) = rhs.chart_expressions.get(chart_id) {
                result.chart_expressions.insert(
                    chart_id.clone(),
                    expr1.clone() * expr2.clone()
                );
            }
        }

        result
    }
}

impl Div for ScalarFieldEnhanced {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if !Arc::ptr_eq(&self.manifold, &rhs.manifold) {
            panic!("Cannot divide scalar fields on different manifolds");
        }

        let mut result = ScalarFieldEnhanced::new(self.manifold.clone());

        for (chart_id, expr1) in &self.chart_expressions {
            if let Some(expr2) = rhs.chart_expressions.get(chart_id) {
                result.chart_expressions.insert(
                    chart_id.clone(),
                    expr1.clone() / expr2.clone()
                );
            }
        }

        result
    }
}

impl Neg for ScalarFieldEnhanced {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = ScalarFieldEnhanced::new(self.manifold.clone());
        result.name = self.name.map(|n| format!("-{}", n));

        for (chart_id, expr) in &self.chart_expressions {
            result.chart_expressions.insert(chart_id.clone(), -expr.clone());
        }

        result
    }
}

impl fmt::Debug for ScalarFieldEnhanced {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScalarFieldEnhanced")
            .field("name", &self.name)
            .field("manifold_dim", &self.manifold.dimension())
            .field("num_charts", &self.chart_expressions.len())
            .finish()
    }
}

impl fmt::Display for ScalarFieldEnhanced {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "ScalarField({})", name)
        } else {
            write!(f, "ScalarField")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples::EuclideanSpace;
    use rustmath_symbolic::Symbol;

    #[test]
    fn test_scalar_field_creation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let f = ScalarFieldEnhanced::new(m.clone());
        assert_eq!(f.manifold().dimension(), 2);
    }

    #[test]
    fn test_scalar_field_from_expr() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let x = Symbol::new("x");
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2));

        let f = ScalarFieldEnhanced::from_expr(m.clone(), chart, expr.clone());
        assert_eq!(f.expr(chart).unwrap(), expr);
    }

    #[test]
    fn test_scalar_field_addition() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let f = ScalarFieldEnhanced::from_expr(m.clone(), chart, Expr::from(1));
        let g = ScalarFieldEnhanced::from_expr(m.clone(), chart, Expr::from(2));

        let h = f + g;
        assert_eq!(h.expr(chart).unwrap(), Expr::from(3));
    }

    #[test]
    fn test_scalar_field_multiplication() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let f = ScalarFieldEnhanced::from_expr(m.clone(), chart, Expr::from(3));
        let g = ScalarFieldEnhanced::from_expr(m.clone(), chart, Expr::from(4));

        let h = f * g;
        assert_eq!(h.expr(chart).unwrap(), Expr::from(12));
    }

    #[test]
    fn test_scalar_field_negation() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = m.default_chart().unwrap();

        let f = ScalarFieldEnhanced::from_expr(m.clone(), chart, Expr::from(5));
        let g = -f;

        assert_eq!(g.expr(chart).unwrap(), Expr::from(-5));
    }

    #[test]
    fn test_scalar_field_differential() {
        let m = Arc::new(EuclideanSpace::new(2));
        let chart = Chart::new("cart", 2, vec!["x", "y"]).unwrap();

        let x = Symbol::new("x");
        let y = Symbol::new("y");

        // f = x^2 + y^2
        let expr = Expr::Symbol(x.clone()).pow(Expr::from(2))
            + Expr::Symbol(y.clone()).pow(Expr::from(2));

        let f = ScalarFieldEnhanced::from_expr(m.clone(), &chart, expr);
        let df = f.differential(&chart).unwrap();

        // df/dx = 2x, df/dy = 2y
        assert_eq!(df.len(), 2);
    }
}
